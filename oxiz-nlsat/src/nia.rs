//! Non-linear Integer Arithmetic (NIA) solver.
//!
//! This module extends the NLSAT solver with integer constraints using
//! branch-and-bound and cutting planes.
//!
//! ## Key Components
//!
//! - **Branch and Bound**: Enumerate integer solutions
//! - **Cutting Planes**: Add constraints to eliminate non-integer solutions
//! - **Mixed Constraints**: Combine real and integer variables
//!
//! ## Reference
//!
//! - Z3's NIA solver in `nlsat/nlsat_solver.cpp`
//! - Branch-and-bound for mixed integer non-linear programming (MINLP)

use crate::solver::{AtomId, Model, NlsatSolver, SolverResult};
use crate::types::{Atom, AtomKind, Literal};
use num_rational::BigRational;
use num_traits::ToPrimitive;
use oxiz_math::polynomial::{Polynomial, Var};

/// Integer variable type specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    /// Real-valued variable (no integer constraint).
    Real,
    /// Integer-valued variable.
    Integer,
}

/// Configuration for NIA solver.
#[derive(Debug, Clone)]
pub struct NiaConfig {
    /// Maximum number of branch-and-bound nodes to explore.
    pub max_nodes: usize,
    /// Maximum depth of the branch-and-bound tree.
    pub max_depth: usize,
    /// Enable cutting planes.
    ///
    /// Currently has no observable effect: [`NiaSolver::add_cutting_plane`]
    /// is an honest no-op (see its doc comment for why a real
    /// tableau-derived Gomory cut is out of scope for the CAD-based
    /// [`NlsatSolver`] this solver wraps). Kept as a harmless, forward-
    /// compatible toggle for a future real implementation.
    pub enable_cutting_planes: bool,
    /// Branching variable selection strategy.
    pub branching_strategy: BranchingStrategy,
    /// Reserved for a future f64-tolerance-based heuristic.
    ///
    /// No soundness-relevant decision in this module uses this value:
    /// both `NiaSolver::is_integer_solution` (the Sat/Unsat gate) and
    /// `NiaSolver::select_branching_variable`'s candidacy test use
    /// exact rational integrality (`BigRational::is_integer`) instead, so
    /// that a value merely *close* to an integer (e.g.
    /// `1_000_000_000_001 / 1_000_000_000_000`) is never mistaken for one.
    pub int_tolerance: f64,
}

impl Default for NiaConfig {
    fn default() -> Self {
        Self {
            max_nodes: 10_000,
            max_depth: 100,
            enable_cutting_planes: true,
            branching_strategy: BranchingStrategy::MostFractional,
            int_tolerance: 1e-6,
        }
    }
}

/// Strategy for selecting which variable to branch on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingStrategy {
    /// Branch on variable with most fractional value.
    MostFractional,
    /// Branch on variable with least fractional value.
    LeastFractional,
    /// Branch on variable with smallest domain.
    SmallestDomain,
    /// Branch on first fractional variable found.
    FirstFractional,
}

/// Branch-and-bound node.
///
/// A node does NOT hold a reference to any [`NlsatSolver`]; instead it
/// records the path of bound constraints (relative to the shared base
/// problem) that must hold for this branch. This is essential for
/// soundness: [`NlsatSolver`] has no clause-retraction (push/pop) API, so
/// asserting a branch bound directly onto a solver shared across sibling
/// branches would permanently leak that bound into every other branch (see
/// the audit finding this module fixes). Each node is instead solved by
/// rebuilding a fresh [`NlsatSolver`] from the shared base problem plus
/// exactly this node's own `path`.
#[derive(Debug, Clone)]
struct BranchNode {
    /// Decision level in the solver.
    level: u32,
    /// Current depth in the tree.
    depth: usize,
    /// Path of bound constraints from the root to this node:
    /// `(var, bound, is_lower)` where `is_lower == true` means `var >= bound`
    /// and `is_lower == false` means `var <= bound`.
    ///
    /// A variable may appear more than once here: unlike an earlier design
    /// that permanently forbade re-branching a variable once split (via a
    /// `branched_vars: HashSet<Var>` hard filter), a still-fractional
    /// integer variable can legitimately need several rounds of
    /// floor/ceil splitting before its value settles on an integer (see
    /// the NIA-4 audit finding). Termination remains guaranteed without an
    /// explicit "already branched" guard because each additional split on
    /// the same variable is a strict refinement of its bound: if `x <=
    /// bound` already holds and the relaxation still returns a fractional
    /// `value < bound`, then `floor(value) < bound` too (bound-width
    /// strictly shrinks by at least one integer unit every time), and
    /// symmetrically for `x >= bound` / `ceil(value) > bound`. That
    /// monotonic narrowing, together with the existing `max_depth` /
    /// `max_nodes` limits, bounds the search.
    path: Vec<(Var, BigRational, bool)>,
}

/// A faithful, replayable snapshot of the arithmetic variables, atoms, and
/// non-learned (i.e. originally asserted, not CDCL-derived) clauses of an
/// [`NlsatSolver`]'s problem.
///
/// Used to rebuild a fresh, correctly scoped [`NlsatSolver`] for each
/// branch-and-bound node without mutating (or leaking state into) the
/// shared base solver.
struct ProblemSnapshot {
    /// Number of arithmetic variables in the base problem.
    num_arith_vars: u32,
    /// One slot per boolean variable index (`0..num_bool_vars`):
    /// `Some((poly, kind))` if that variable backs a single-factor
    /// inequality atom, `None` if it's a "free" boolean variable with no
    /// associated atom. Replaying slots in index order via
    /// `new_ineq_atom`/`new_bool_var` reproduces the exact original
    /// variable numbering, since both allocate the next sequential
    /// boolean variable deterministically.
    atom_slots: Vec<Option<(Polynomial, AtomKind)>>,
    /// All non-learned clauses of the base problem.
    clauses: Vec<Vec<Literal>>,
}

/// Non-linear integer arithmetic solver.
///
/// Extends NLSAT with integer constraints using branch-and-bound.
pub struct NiaSolver {
    /// Underlying NLSAT solver for real arithmetic.
    nlsat: NlsatSolver,
    /// Variable types (Real or Integer).
    var_types: Vec<VarType>,
    /// NIA-specific configuration.
    config: NiaConfig,
    /// Statistics.
    stats: NiaStats,
}

/// Statistics for NIA solver.
#[derive(Debug, Clone, Default)]
pub struct NiaStats {
    /// Number of branch-and-bound nodes explored.
    pub nodes_explored: usize,
    /// Number of cutting planes added.
    pub cutting_planes: usize,
    /// Number of integer solutions found.
    pub integer_solutions: usize,
    /// Maximum depth reached.
    pub max_depth_reached: usize,
}

impl NiaSolver {
    /// Create a new NIA solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(NiaConfig::default())
    }

    /// Create a new NIA solver with custom configuration.
    pub fn with_config(config: NiaConfig) -> Self {
        Self {
            nlsat: NlsatSolver::new(),
            var_types: Vec::new(),
            config,
            stats: NiaStats::default(),
        }
    }

    /// Get the underlying NLSAT solver.
    pub fn nlsat(&self) -> &NlsatSolver {
        &self.nlsat
    }

    /// Get mutable reference to underlying NLSAT solver.
    pub fn nlsat_mut(&mut self) -> &mut NlsatSolver {
        &mut self.nlsat
    }

    /// Set variable type (Real or Integer).
    pub fn set_var_type(&mut self, var: Var, var_type: VarType) {
        // Ensure we have enough space
        while self.var_types.len() <= var as usize {
            self.var_types.push(VarType::Real);
        }
        self.var_types[var as usize] = var_type;
    }

    /// Get variable type.
    pub fn var_type(&self, var: Var) -> VarType {
        self.var_types
            .get(var as usize)
            .copied()
            .unwrap_or(VarType::Real)
    }

    /// Check if a variable is integer-typed.
    pub fn is_integer_var(&self, var: Var) -> bool {
        self.var_type(var) == VarType::Integer
    }

    /// Solve with integer constraints.
    ///
    /// Uses branch-and-bound to find integer solutions.
    pub fn solve(&mut self) -> SolverResult {
        // Reset statistics
        self.stats = NiaStats::default();

        // First solve the real relaxation
        let real_result = self.nlsat.solve();

        match real_result {
            SolverResult::Unsat => {
                // Real relaxation is UNSAT, so integer problem is also UNSAT
                SolverResult::Unsat
            }
            SolverResult::Unknown => SolverResult::Unknown,
            SolverResult::Sat => {
                // Check if the solution satisfies integer constraints
                if let Some(model) = self.nlsat.get_model() {
                    if self.is_integer_solution(&model) {
                        // Lucky! Real solution is already integer
                        self.stats.integer_solutions += 1;
                        return SolverResult::Sat;
                    }

                    // Need to branch
                    return self.branch_and_bound();
                }
                SolverResult::Unknown
            }
        }
    }

    /// Take a faithful, replayable snapshot of `self.nlsat`'s current
    /// problem (arithmetic variable count, atoms, and non-learned clauses).
    ///
    /// Returns `None` if the current problem cannot be guaranteed to
    /// replay identically onto a fresh [`NlsatSolver`] (e.g. it contains
    /// atoms not reachable through the public `new_ineq_atom` API, such as
    /// root atoms produced internally during CAD-based solving, or stray
    /// boolean variables not tied 1:1 to atoms). Branch-and-bound scoping
    /// depends on exact replay fidelity for soundness, so callers must
    /// treat `None` as "cannot safely branch" rather than guessing.
    fn snapshot_problem(&self) -> Option<ProblemSnapshot> {
        let num_atoms = self.nlsat.num_atoms();
        let num_bool_vars = self.nlsat.num_bool_vars() as usize;
        let mut atom_slots: Vec<Option<(Polynomial, AtomKind)>> = vec![None; num_bool_vars];

        for id in 0..num_atoms as AtomId {
            match self.nlsat.get_atom(id)? {
                Atom::Ineq(ineq) if ineq.factors.len() == 1 && !ineq.factors[0].is_even => {
                    let slot = atom_slots.get_mut(ineq.bool_var as usize)?;
                    *slot = Some((ineq.factors[0].poly.clone(), ineq.kind));
                }
                // Root atoms or multi-factor/even-power atoms are not
                // reproducible via `new_ineq_atom`; bail out honestly
                // rather than silently dropping or mis-replaying them.
                _ => return None,
            }
        }

        let clauses: Vec<Vec<Literal>> = self
            .nlsat
            .clauses()
            .clauses()
            .iter()
            .filter(|c| !c.is_learned())
            .map(|c| c.literals().to_vec())
            .collect();

        Some(ProblemSnapshot {
            num_arith_vars: self.nlsat.num_arith_vars(),
            atom_slots,
            clauses,
        })
    }

    /// Rebuild a fresh [`NlsatSolver`] from `snapshot` plus the branch bound
    /// constraints in `path`. Each branch node gets its own solver built
    /// this way, so bounds from one branch can never leak into a sibling.
    fn rebuild_solver(
        snapshot: &ProblemSnapshot,
        path: &[(Var, BigRational, bool)],
    ) -> NlsatSolver {
        let mut solver = NlsatSolver::new();

        for _ in 0..snapshot.num_arith_vars {
            solver.new_arith_var();
        }

        for slot in &snapshot.atom_slots {
            match slot {
                Some((poly, kind)) => {
                    solver.new_ineq_atom(poly.clone(), *kind);
                }
                None => {
                    solver.new_bool_var();
                }
            }
        }

        for clause in &snapshot.clauses {
            solver.add_clause(clause.clone());
        }

        for &(var, ref bound, is_lower) in path {
            // For x >= bound: NOT(x - bound < 0)
            // For x <= bound: NOT(x - bound > 0)
            let x = Polynomial::from_var(var);
            let poly = Polynomial::sub(&x, &Polynomial::constant(bound.clone()));

            if is_lower {
                let atom_id = solver.new_ineq_atom(poly, AtomKind::Lt);
                let lit = solver.atom_literal(atom_id, false);
                solver.add_clause(vec![lit]);
            } else {
                let atom_id = solver.new_ineq_atom(poly, AtomKind::Gt);
                let lit = solver.atom_literal(atom_id, false);
                solver.add_clause(vec![lit]);
            }
        }

        solver
    }

    /// Branch-and-bound search for integer solutions.
    ///
    /// Each node is solved against a *freshly rebuilt* [`NlsatSolver`]
    /// (via [`Self::rebuild_solver`]) scoped to exactly that node's branch
    /// path, so sibling branch bounds never leak into each other. Gomory
    /// cuts, which are valid across the whole integer hull (not just the
    /// current fractional vertex), are instead asserted permanently onto
    /// the shared base solver `self.nlsat`, which is re-snapshotted before
    /// every node so all pending branches benefit from them.
    fn branch_and_bound(&mut self) -> SolverResult {
        let root_node = BranchNode {
            level: 0,
            depth: 0,
            path: Vec::new(),
        };

        let mut stack = vec![root_node];
        // Tracks whether every explored branch was conclusively resolved
        // (Sat or Unsat). If any node was Unknown (e.g. depth-limited or
        // solver-inconclusive), the overall search cannot honestly claim
        // Unsat once the stack is exhausted.
        let mut fully_explored = true;

        while let Some(node) = stack.pop() {
            self.stats.nodes_explored += 1;
            self.stats.max_depth_reached = self.stats.max_depth_reached.max(node.depth);

            // Check limits
            if self.stats.nodes_explored >= self.config.max_nodes {
                return SolverResult::Unknown;
            }
            if node.depth >= self.config.max_depth {
                // Pruned due to depth limit, not proven infeasible.
                fully_explored = false;
                continue;
            }

            // Snapshot the current base problem (original constraints plus
            // any Gomory cuts accumulated so far) and rebuild a solver
            // scoped to exactly this node's own branch path.
            let Some(snapshot) = self.snapshot_problem() else {
                // Cannot faithfully replay the problem (see
                // `snapshot_problem` docs) — branching cannot be proven
                // sound, so honestly report Unknown instead of guessing.
                return SolverResult::Unknown;
            };
            let mut node_solver = Self::rebuild_solver(&snapshot, &node.path);
            let result = node_solver.solve();

            match result {
                SolverResult::Unsat => {
                    // This branch is infeasible - backtrack
                    continue;
                }
                SolverResult::Unknown => {
                    // Inconclusive - try other branches, but remember we
                    // cannot claim full exploration.
                    fully_explored = false;
                    continue;
                }
                SolverResult::Sat => {
                    // Check if solution is integer
                    if let Some(model) = node_solver.get_model() {
                        if self.is_integer_solution(&model) {
                            // Found an integer solution! Publish the node's
                            // solver as the authoritative one so
                            // `self.nlsat().get_model()` reflects it.
                            self.stats.integer_solutions += 1;
                            self.nlsat = node_solver;
                            return SolverResult::Sat;
                        }

                        // If cutting planes are enabled, attempt to cut off
                        // the fractional LP solution before branching. A
                        // Gomory cut tightens the relaxation and can prune
                        // many branches. Cuts are asserted onto the shared
                        // base solver (never onto `node_solver`, which is
                        // scoped to this branch only) since they remain
                        // valid for the entire integer hull.
                        if self.config.enable_cutting_planes {
                            let _ = self.add_cutting_plane(&model);
                        }

                        // Solution is not integer - need to branch
                        if let Some(branch_var) = self.select_branching_variable(&model) {
                            // Get current value of branch variable
                            if let Some(value) = model.arith_value(branch_var) {
                                // Create two branches: x <= floor(value) and x >= ceil(value)
                                let (floor_val, ceil_val) = self.floor_ceil(value);

                                // Push ceil branch first (stack is LIFO, so will explore floor first)
                                if ceil_val > floor_val {
                                    Self::push_branch(
                                        &mut stack, &node, branch_var, &ceil_val,
                                        true, // >= ceil
                                    );
                                }

                                // Push floor branch
                                Self::push_branch(
                                    &mut stack, &node, branch_var, &floor_val,
                                    false, // <= floor
                                );
                            }
                        } else {
                            // `is_integer_solution` above rejected this model
                            // as non-integral, yet `select_branching_variable`
                            // (which uses the same exact rational test for
                            // candidacy) found no branchable variable. That
                            // happens for real when the relaxation pinned an
                            // integer-typed variable to an exact *algebraic*
                            // point: irrational, so certainly not an integer,
                            // but with no rational value to take a floor and
                            // ceiling of, so nothing to branch on either.
                            // Branching on an algebraic point's approximation
                            // is a later phase. Either way this node's fate is
                            // genuinely unresolved -- it must NOT be silently
                            // dropped as if it were refuted, or the search
                            // could wrongly conclude Unsat once the stack
                            // empties.
                            fully_explored = false;
                            continue;
                        }
                    }
                }
            }
        }

        // Exhausted search space. Only report Unsat if every branch was
        // conclusively refuted; otherwise some region of the search space
        // was never actually ruled out.
        if fully_explored {
            SolverResult::Unsat
        } else {
            SolverResult::Unknown
        }
    }

    /// Select which variable to branch on.
    ///
    /// Candidacy is gated on the *exact* rational integrality test
    /// (`!value.is_integer()`), matching [`Self::is_integer_solution`]'s
    /// authoritative soundness gate. It intentionally does NOT reuse the
    /// lossy f64 `int_tolerance` window: a value that is f64-near-integer
    /// but not exactly integral (e.g. `1_000_000_000_001 /
    /// 1_000_000_000_000`) is rejected by `is_integer_solution` and must
    /// still be selectable here, or the search would find no candidate for
    /// a genuinely fractional node and silently drop it (see the NIA-3
    /// audit finding). A variable may be selected again after already
    /// having been branched on -- see the `path` doc comment on
    /// [`BranchNode`] for why that remains sound and terminating.
    fn select_branching_variable(&self, model: &Model) -> Option<Var> {
        let mut candidates: Vec<(Var, BigRational, f64)> = Vec::new();

        for var in 0..self.nlsat.num_arith_vars() {
            // Skip if not integer-typed
            if !self.is_integer_var(var) {
                continue;
            }

            // Get value from model.
            if let Some(value) = model.arith_value(var)
                && !value.is_integer()
            {
                // `frac` is used only to rank candidates by "how
                // fractional" they are for the search-order heuristics
                // below (ties broken by proximity to 0.5, etc.); it is
                // non-authoritative, so the f64 approximation is fine
                // here even though it was wrong as a candidacy gate.
                let frac = self.fractional_part(value);
                candidates.push((var, value.clone(), frac));
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // Select based on strategy
        match self.config.branching_strategy {
            BranchingStrategy::MostFractional => {
                // Pick variable with fractional part closest to 0.5
                candidates.sort_by(|a, b| {
                    let dist_a = (a.2 - 0.5).abs();
                    let dist_b = (b.2 - 0.5).abs();
                    dist_a
                        .partial_cmp(&dist_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Some(candidates[0].0)
            }
            BranchingStrategy::LeastFractional => {
                // Pick variable with smallest fractional part
                candidates
                    .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
                Some(candidates[0].0)
            }
            BranchingStrategy::FirstFractional => Some(candidates[0].0),
            BranchingStrategy::SmallestDomain => {
                // For now, just pick first (domain analysis would require more info)
                Some(candidates[0].0)
            }
        }
    }

    /// Push a child branch node onto the stack, recording the new bound
    /// constraint in its `path` (relative to the shared base problem).
    ///
    /// Unlike the old design, this never touches any [`NlsatSolver`]
    /// directly: the constraint is only materialized when the node is
    /// popped and solved via [`Self::rebuild_solver`], which keeps sibling
    /// branches from ever seeing each other's bounds.
    fn push_branch(
        stack: &mut Vec<BranchNode>,
        parent: &BranchNode,
        var: Var,
        bound: &BigRational,
        is_lower: bool, // true for x >= bound, false for x <= bound
    ) {
        let mut new_path = parent.path.clone();
        new_path.push((var, bound.clone(), is_lower));

        stack.push(BranchNode {
            level: parent.level + 1,
            depth: parent.depth + 1,
            path: new_path,
        });
    }

    /// Check if a model satisfies all integer constraints.
    ///
    /// This is the authoritative soundness gate for branch-and-bound: once it
    /// returns `true`, the caller reports `model` as the solver's final `Sat`
    /// answer, so it must check *exact* integrality (`BigRational::is_integer`,
    /// i.e. denominator `== 1` in lowest terms) rather than an f64-tolerance
    /// heuristic. Such a heuristic would convert the exact rational to `f64`
    /// (itself precision-losing for large numerators/denominators) and accept
    /// anything within `int_tolerance` of a whole number, so a genuinely
    /// fractional value such as `1_000_000_000_001 / 1_000_000_000_000` would
    /// be wrongly reported as an integer solution, making the solver return
    /// `Sat` with a model that does not actually satisfy the integrality
    /// constraint. Every soundness-relevant integrality check in this module
    /// -- this method and [`Self::select_branching_variable`]'s candidacy
    /// gate -- uses the same exact test for that reason; only the
    /// non-authoritative candidate-ranking heuristic
    /// ([`Self::fractional_part`], used solely to order/tie-break already-
    /// exact-confirmed candidates) remains `f64`-based, since an approximate
    /// ranking only affects search efficiency, never correctness.
    fn is_integer_solution(&self, model: &Model) -> bool {
        for var in 0..self.nlsat.num_arith_vars() {
            if !self.is_integer_var(var) {
                continue; // Skip real variables
            }

            // An exact real-algebraic witness (`√2`) is irrational, so it is
            // never an integer -- and `arith_value` deliberately reports it as
            // absent (see `oxiz_nlsat::solver::Model`). Without this check the
            // loop below would skip the variable and read "no rational value"
            // as "no integrality violation", making QF_NIA `x*x = 2` come back
            // Sat. The relaxation being algebraically satisfiable says nothing
            // about the integer problem.
            if model.arith_point(var).is_some_and(|p| !p.is_rational()) {
                return false;
            }

            if let Some(value) = model.arith_value(var)
                && !value.is_integer()
            {
                return false;
            }
        }
        true
    }

    /// Get the fractional part of a rational number.
    ///
    /// Non-authoritative: used only to rank already exact-confirmed
    /// fractional candidates in [`Self::select_branching_variable`] (e.g.
    /// picking the one closest to `0.5` for [`BranchingStrategy::MostFractional`]).
    /// Never used as an integrality *test* -- see [`Self::is_integer_solution`].
    fn fractional_part(&self, value: &BigRational) -> f64 {
        // Convert to f64 for fractional part calculation
        let val_f64 = value.numer().to_f64().unwrap_or(0.0) / value.denom().to_f64().unwrap_or(1.0);
        (val_f64 - val_f64.floor()).abs()
    }

    /// Compute floor and ceiling of a rational number.
    ///
    /// Uses [`BigRational::floor`]/[`BigRational::ceil`] rather than raw
    /// `BigInt` division, which truncates toward zero and gives the wrong
    /// answer for negative non-integral values (e.g. `-3/2` truncates to
    /// `-1`, but `floor(-3/2) == -2`). A branch bound of `x <= -1` would
    /// fail to exclude the fractional relaxation point `x = -1.5`, causing
    /// branch-and-bound to loop on the same fractional solution.
    fn floor_ceil(&self, value: &BigRational) -> (BigRational, BigRational) {
        (value.floor(), value.ceil())
    }

    /// Get statistics.
    pub fn stats(&self) -> &NiaStats {
        &self.stats
    }

    /// Attempt to add Gomory cutting planes to eliminate the current
    /// fractional solution.
    ///
    /// This is a deliberate, honest no-op: it always returns `false` and
    /// never touches `self.nlsat`. A prior implementation built a
    /// fabricated "row" for `oxiz_math::lp::cutting_planes::
    /// CuttingPlaneGenerator::generate_gomory_cut`, consisting solely of
    /// the identity term `coefficient 1 for var, RHS = value`. That is not
    /// a real simplex-tableau row (which must express a
    /// *basic* fractional variable as a combination of the problem's
    /// *non-basic* variables); a coefficient of exactly `1` always has zero
    /// fractional part, so the generator's cut was always empty and the
    /// call was dead code in practice. Worse, had a non-empty cut ever come
    /// out of a fabricated row, asserting it permanently onto the shared
    /// `self.nlsat` (as the old code did) would have been UNSOUND: the
    /// resulting "cut" would not actually be implied by the problem's real
    /// constraints, so it could exclude genuine integer solutions and turn
    /// a satisfiable problem into a reported `Unsat`.
    ///
    /// [`NlsatSolver`] is a CAD-based nonlinear decision procedure, not a
    /// simplex/LP engine, so it has no tableau to derive a real Gomory row
    /// from; producing one would require building a separate LP-relaxation
    /// subsystem, which is out of scope here. `oxiz-theories`'s LIA solver
    /// (`lia/cuts.rs`) is a different situation: it does maintain a real
    /// simplex tableau, and its GMI/Chvatal-Gomory cut generator derives
    /// genuine tableau-backed inequalities from it, wired into
    /// `LiaSolver::check` as root-node cuts. That machinery has no
    /// counterpart here -- there is no tableau in this module to feed it --
    /// so `add_cutting_plane` stays a documented no-op rather than
    /// fabricating an unsound cut from a non-tableau "row". The
    /// tableau-backed nonlinear cut engine, when one lands, is expected at
    /// `oxiz_theories::arithmetic::nla`; until then, branch-and-bound alone
    /// (see `Self::branch_and_bound`) remains sound and, for bounded
    /// problems, complete without cutting planes here.
    ///
    /// Kept as a callable method (rather than deleted outright) so
    /// `NiaConfig::enable_cutting_planes` remains a harmless, meaningful
    /// toggle for existing callers and so a future tableau-backed
    /// implementation has a stable call site to land in;
    /// `stats().cutting_planes` is never incremented by this method.
    pub fn add_cutting_plane(&mut self, _model: &Model) -> bool {
        false
    }
}

impl Default for NiaSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    #[test]
    fn test_nia_solver_new() {
        let solver = NiaSolver::new();
        assert_eq!(solver.stats().nodes_explored, 0);
    }

    #[test]
    fn test_nia_var_types() {
        let mut solver = NiaSolver::new();

        solver.set_var_type(0, VarType::Integer);
        solver.set_var_type(1, VarType::Real);

        assert_eq!(solver.var_type(0), VarType::Integer);
        assert_eq!(solver.var_type(1), VarType::Real);
        assert!(solver.is_integer_var(0));
        assert!(!solver.is_integer_var(1));
    }

    #[test]
    fn test_nia_fractional_part() {
        let solver = NiaSolver::new();

        let val1 = BigRational::new(5.into(), 2.into()); // 2.5
        let frac1 = solver.fractional_part(&val1);
        assert!((frac1 - 0.5).abs() < 0.01);

        let val2 = BigRational::from_integer(3.into()); // 3.0
        let frac2 = solver.fractional_part(&val2);
        assert!(frac2 < 0.01);
    }

    #[test]
    fn test_nia_floor_ceil() {
        let solver = NiaSolver::new();

        let val = BigRational::new(7.into(), 2.into()); // 3.5
        let (floor, ceil) = solver.floor_ceil(&val);

        assert_eq!(floor, rat(3));
        assert_eq!(ceil, rat(4));
    }

    #[test]
    fn test_nia_simple_integer() {
        let mut solver = NiaSolver::new();

        // x is integer, x - 1 = 0  => x = 1
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        let x = Polynomial::from_var(var_x);
        let poly = Polynomial::sub(&x, &Polynomial::constant(rat(1)));
        let atom = solver.nlsat_mut().new_ineq_atom(poly, AtomKind::Eq);
        let lit = solver.nlsat().atom_literal(atom, true);
        solver.nlsat_mut().add_clause(vec![lit]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        if let Some(model) = solver.nlsat().get_model() {
            let x_val = model
                .arith_value(var_x)
                .expect("test operation should succeed");
            assert_eq!(x_val, &rat(1));
        }
    }

    #[test]
    fn test_nia_fractional_infeasible() {
        let mut solver = NiaSolver::new();

        // x is integer
        // x - 0.5 = 0  => x = 0.5 (not integer, should be UNSAT)
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        let x = Polynomial::from_var(var_x);
        let half = BigRational::new(1.into(), 2.into());
        let poly = Polynomial::sub(&x, &Polynomial::constant(half));
        let atom = solver.nlsat_mut().new_ineq_atom(poly, AtomKind::Eq);
        let lit = solver.nlsat().atom_literal(atom, true);
        solver.nlsat_mut().add_clause(vec![lit]);

        let result = solver.solve();
        // Real relaxation gives x = 0.5, but no integer solution exists
        assert_eq!(result, SolverResult::Unsat);
    }

    #[test]
    fn test_branching_strategy() {
        let config = NiaConfig {
            branching_strategy: BranchingStrategy::MostFractional,
            ..Default::default()
        };

        let solver = NiaSolver::with_config(config);
        assert_eq!(
            solver.config.branching_strategy,
            BranchingStrategy::MostFractional
        );
    }

    #[test]
    fn test_nia_stats() {
        let solver = NiaSolver::new();
        let stats = solver.stats();

        assert_eq!(stats.nodes_explored, 0);
        assert_eq!(stats.cutting_planes, 0);
        assert_eq!(stats.integer_solutions, 0);
    }

    /// Regression test for the NIA-5 audit finding: `add_cutting_plane` is
    /// an honest no-op (see its doc comment), so even a problem that
    /// genuinely exercises branch-and-bound with
    /// `enable_cutting_planes: true` (the default) must never report any
    /// cutting planes added. This pins the no-op so a future change can't
    /// silently reintroduce the old fabricated-identity-row cut path
    /// without a deliberate, reviewed test update.
    #[test]
    fn test_add_cutting_plane_is_pinned_as_a_no_op() {
        let mut solver = NiaSolver::new();
        assert!(solver.config.enable_cutting_planes);

        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        // x - 0.5 = 0  =>  real relaxation x = 0.5 (fractional), forcing a
        // branch-and-bound round in which `enable_cutting_planes` gates a
        // call to `add_cutting_plane`.
        let x = Polynomial::from_var(var_x);
        let half = BigRational::new(1.into(), 2.into());
        let poly = Polynomial::sub(&x, &Polynomial::constant(half));
        let atom = solver.nlsat_mut().new_ineq_atom(poly, AtomKind::Eq);
        let lit = solver.nlsat().atom_literal(atom, true);
        solver.nlsat_mut().add_clause(vec![lit]);

        let _ = solver.solve();
        assert_eq!(
            solver.stats().cutting_planes,
            0,
            "add_cutting_plane is a documented no-op; it must never report \
             a cut being added"
        );

        // Also exercise the method directly with a hand-built model.
        let mut arith_values = std::collections::HashMap::new();
        arith_values.insert(var_x, BigRational::new(3.into(), 2.into())); // 1.5
        let model = Model {
            bool_values: std::collections::HashMap::new(),
            arith_values,
            ..Default::default()
        };
        assert!(!solver.add_cutting_plane(&model));
        assert_eq!(solver.stats().cutting_planes, 0);
    }

    // Regression test: a value within f64-tolerance distance of an integer
    // (as `int_tolerance` used to gate on) must NOT be conflated with
    // genuine, exact integrality -- this is the exact rational distinction
    // that `is_integer_solution` and `select_branching_variable` now both
    // rely on.
    #[test]
    fn test_near_integer_by_f64_tolerance_is_not_exactly_integer() {
        let near_one = BigRational::from_integer(1.into())
            + BigRational::new(1.into(), 1_000_000_000_000i64.into());

        // Within `NiaConfig::default().int_tolerance` (1e-6) of an integer
        // when viewed through f64...
        let frac = (near_one.numer().to_f64().unwrap_or(0.0)
            / near_one.denom().to_f64().unwrap_or(1.0))
        .fract()
        .abs();
        assert!(frac < NiaConfig::default().int_tolerance);
        // ...but it genuinely is not an integer.
        assert!(!near_one.is_integer());
    }

    // Regression test for the item: previously `is_integer_solution` used
    // `is_near_integer`'s f64-tolerance heuristic as the authoritative
    // soundness check, so a value within `int_tolerance` (default `1e-6`) of
    // a whole number but not actually integral would be wrongly accepted as
    // an "integer solution", making the solver return `Sat` for a genuinely
    // `Unsat` integer problem. `x = 1 + 1/10^12` is the *unique* real
    // solution here (an equality constraint) and is not an integer, so the
    // correct answer is `Unsat`.
    #[test]
    fn test_nia_near_integer_but_not_exact_is_rejected() {
        let mut solver = NiaSolver::new();
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        let x = Polynomial::from_var(var_x);
        let near_one = BigRational::from_integer(1.into())
            + BigRational::new(1.into(), 1_000_000_000_000i64.into());
        let poly = Polynomial::sub(&x, &Polynomial::constant(near_one));
        let atom = solver.nlsat_mut().new_ineq_atom(poly, AtomKind::Eq);
        let lit = solver.nlsat().atom_literal(atom, true);
        solver.nlsat_mut().add_clause(vec![lit]);

        let result = solver.solve();
        assert_eq!(
            result,
            SolverResult::Unsat,
            "x = 1 + 1e-12 is not an integer; the solver must not accept it \
             as an integer solution just because it is within int_tolerance \
             of one"
        );
    }

    /// Regression test for the audit finding: `create_branch` used to add
    /// BOTH the floor (`x <= bound`) and ceil (`x >= bound + 1`) constraints
    /// as permanent unit clauses to the SAME shared [`NlsatSolver`]. Once
    /// both siblings were pushed, the shared solver held the contradictory
    /// pair `x <= 0` and `x >= 1` simultaneously, so every subsequent node
    /// (in either branch) solved an unsatisfiable, over-constrained problem
    /// and `branch_and_bound` always returned `Unsat`.
    ///
    /// This test directly exercises the fixed scoping mechanism
    /// (`snapshot_problem` + `rebuild_solver`) and checks that two sibling
    /// branch solvers built from the same snapshot never see each other's
    /// bound, and that the shared base solver is left completely
    /// unaffected by either branch.
    #[test]
    fn test_branch_bounds_do_not_leak_between_siblings() {
        let mut solver = NiaSolver::new();
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        // Only constraint on the base problem: x >= 0 (i.e. NOT(x < 0)).
        let x = Polynomial::from_var(var_x);
        let atom = solver.nlsat_mut().new_ineq_atom(x, AtomKind::Lt);
        let lit = solver.nlsat().atom_literal(atom, false);
        solver.nlsat_mut().add_clause(vec![lit]);

        let snapshot = solver
            .snapshot_problem()
            .expect("simple linear problem must be snapshot-able");

        // Sibling branch paths that used to be asserted permanently onto
        // ONE shared solver: floor branch "x <= 0" and ceil branch "x >= 1".
        let floor_path = vec![(var_x, BigRational::from_integer(0.into()), false)];
        let ceil_path = vec![(var_x, BigRational::from_integer(1.into()), true)];

        let mut floor_solver = NiaSolver::rebuild_solver(&snapshot, &floor_path);
        let mut ceil_solver = NiaSolver::rebuild_solver(&snapshot, &ceil_path);

        // Each branch is independently satisfiable (x=0 and x=1
        // respectively). If bounds leaked between siblings, at least one
        // of these would incorrectly be Unsat.
        assert_eq!(floor_solver.solve(), SolverResult::Sat);
        assert_eq!(ceil_solver.solve(), SolverResult::Sat);

        // The shared base solver must remain untouched by either branch.
        assert_eq!(solver.nlsat_mut().solve(), SolverResult::Sat);
        if let Some(model) = solver.nlsat().get_model() {
            // Base problem only asserts x >= 0; a negative value would mean
            // a branch bound leaked into the shared solver.
            assert!(model.arith_value(var_x).is_none_or(|v| *v >= rat(0)));
        }
    }

    /// End-to-end sanity check covering `NiaSolver::solve()`'s public API
    /// for a disjunctive constraint with one fractional and one integer
    /// alternative: `(x = 3/2) OR (x = 2)`. Whether or not the solver's
    /// initial witness happens to be the fractional disjunct (routing
    /// through `branch_and_bound`) or the integer one directly, the only
    /// answer consistent with the problem is `Sat` with `x = 2` — the
    /// pre-fix bound-leaking bug could turn this into a wrong `Unsat`
    /// whenever branching was exercised (see
    /// `test_branch_bounds_do_not_leak_between_siblings` for a
    /// branching-path-targeted reproduction of that exact defect).
    #[test]
    fn test_nia_strict_lower_bound_finds_integer_solution() {
        let mut solver = NiaSolver::new();
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        let x = Polynomial::from_var(var_x);
        let half_poly = Polynomial::sub(
            &x,
            &Polynomial::constant(BigRational::new(3.into(), 2.into())),
        );
        let two_poly = Polynomial::sub(&x, &Polynomial::constant(rat(2)));
        let half_atom = solver.nlsat_mut().new_ineq_atom(half_poly, AtomKind::Eq);
        let two_atom = solver.nlsat_mut().new_ineq_atom(two_poly, AtomKind::Eq);
        let half_lit = solver.nlsat().atom_literal(half_atom, true);
        let two_lit = solver.nlsat().atom_literal(two_atom, true);
        solver.nlsat_mut().add_clause(vec![half_lit, two_lit]);

        let result = solver.solve();
        assert_eq!(
            result,
            SolverResult::Sat,
            "(x = 3/2) OR (x = 2) has the integer solution x = 2"
        );

        if let Some(model) = solver.nlsat().get_model() {
            let x_val = model
                .arith_value(var_x)
                .expect("model must assign integer var x");
            assert_eq!(*x_val, rat(2), "the only integer-satisfying model is x = 2");
            assert_eq!(
                x_val.denom(),
                &num_bigint::BigInt::from(1),
                "returned solution must be integer, got {x_val}"
            );
        }
    }

    #[test]
    fn test_nia_floor_ceil_negative() {
        let solver = NiaSolver::new();

        // floor(-3/2) == -2, ceil(-3/2) == -1 (NOT -1 and 0, which is what
        // truncating BigInt division toward zero would incorrectly give).
        let val = BigRational::new((-3).into(), 2.into());
        let (floor, ceil) = solver.floor_ceil(&val);

        assert_eq!(floor, rat(-2));
        assert_eq!(ceil, rat(-1));
    }

    #[test]
    fn test_nia_floor_ceil_negative_integer() {
        let solver = NiaSolver::new();

        // An already-integral negative value: floor == ceil == the value.
        let val = rat(-4);
        let (floor, ceil) = solver.floor_ceil(&val);

        assert_eq!(floor, rat(-4));
        assert_eq!(ceil, rat(-4));
    }

    /// Regression test for the NIA-4 audit finding: two *successive*
    /// rounds of branching on the same variable, driven directly through
    /// `push_branch` (mirroring how `branch_and_bound` would grow the
    /// path), must each still find that variable a valid branch candidate
    /// as long as its relaxed value remains fractional, converging on the
    /// correct final integer bound.
    #[test]
    fn test_push_branch_twice_on_same_var_preserves_candidacy() {
        let mut solver = NiaSolver::new();
        let var_x = solver.nlsat_mut().new_arith_var();
        solver.set_var_type(var_x, VarType::Integer);

        let root = BranchNode {
            level: 0,
            depth: 0,
            path: Vec::new(),
        };

        // Round 1: relaxation returns x = 1.5 -> branch to x >= 2.
        let mut stack = Vec::new();
        NiaSolver::push_branch(&mut stack, &root, var_x, &rat(2), true);
        let round1_node = stack.pop().expect("push_branch must push a node");
        assert_eq!(round1_node.path, vec![(var_x, rat(2), true)]);

        let mut model1_values = std::collections::HashMap::new();
        model1_values.insert(var_x, BigRational::new(5.into(), 2.into())); // 2.5
        let model1 = Model {
            bool_values: std::collections::HashMap::new(),
            arith_values: model1_values,
            ..Default::default()
        };
        // Still fractional under the round-1 bound -> must be selectable
        // again (this is exactly what the old `branched_vars` filter
        // would have forbidden).
        assert_eq!(solver.select_branching_variable(&model1), Some(var_x));

        // Round 2: relaxation returns x = 2.5 -> branch to x >= 3.
        NiaSolver::push_branch(&mut stack, &round1_node, var_x, &rat(3), true);
        let round2_node = stack.pop().expect("push_branch must push a node");
        assert_eq!(
            round2_node.path,
            vec![(var_x, rat(2), true), (var_x, rat(3), true)],
            "the path must accumulate both rounds' bounds on the same var"
        );

        // Round 2's relaxation finally lands on an integer -> no longer a
        // candidate.
        let mut model2_values = std::collections::HashMap::new();
        model2_values.insert(var_x, rat(3));
        let model2 = Model {
            bool_values: std::collections::HashMap::new(),
            arith_values: model2_values,
            ..Default::default()
        };
        assert_eq!(
            solver.select_branching_variable(&model2),
            None,
            "an exactly-integer value must not be selected for further branching"
        );
    }
}
