//! Model-Based Interpolation (MBI)
//!
//! Provides Craig interpolation for the propositional (Boolean) fragment.
//!
//! Given an unsatisfiable conjunction `A ∧ B`, an interpolant `I` satisfies
//!
//! 1. `A ⇒ I`,
//! 2. `I ∧ B` is unsatisfiable, and
//! 3. every variable of `I` is shared between `A` and `B`.
//!
//! ## What is (and is not) supported
//!
//! Only the **purely propositional** fragment is handled soundly: formulas
//! built from `true`/`false`, Boolean variables, and the connectives
//! `not`/`and`/`or`/`implies`/`xor`/`ite`/`=` over Boolean subterms. For that
//! fragment the interpolant is computed exactly as
//!
//! ```text
//! I = ∃(vars(A) \ shared). A
//! ```
//!
//! which is a genuine Craig interpolant: `A ⇒ I` holds by existential
//! weakening, `vars(I) ⊆ shared` by construction, and `I ∧ B` is
//! unsatisfiable whenever `A ∧ B` is (the `A`-local and `B`-local variables
//! are disjoint, so any shared model of `I ∧ B` lifts to a model of `A ∧ B`).
//! The existential over the finite set of Boolean `A`-local variables is
//! computed by exact expansion `∃v. φ ≡ φ[v:=⊤] ∨ φ[v:=⊥]`.
//!
//! Every returned interpolant is additionally **validated** against the
//! interpolant conditions by exhaustive evaluation over the (finite) variable
//! set before it is handed back — so a fabricated or otherwise incorrect
//! result can never escape this module. In particular, if `A ∧ B` is actually
//! satisfiable (no interpolant exists) validation fails and `None` is
//! returned. Any formula outside the supported fragment, or one with more than
//! [`MAX_VALIDATION_VARS`] variables (where exhaustive validation is
//! infeasible), yields an honest `None`.
//!
//! Reference: Z3's interpolation (`src/muz/spacer` / `theory_interpolant.cpp`);
//! McMillan, "Interpolation and SAT-based Model Checking" (2003).

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use std::collections::{HashMap, HashSet};

/// Maximum number of distinct variables for which an interpolant candidate is
/// exhaustively validated (`2^MAX_VALIDATION_VARS` assignments). Beyond this
/// bound validation is infeasible, so `interpolate` returns `None` rather than
/// an unvalidated result.
pub const MAX_VALIDATION_VARS: usize = 20;

/// Work item of the iterative propositional evaluator.
#[derive(Debug, Clone, Copy)]
enum EvalStep {
    /// Visit a term: memoize it if it is a leaf, otherwise schedule its
    /// operands and a follow-up [`EvalStep::Combine`].
    Enter(TermId),
    /// Combine already-evaluated operands into this term's value.
    Combine(TermId),
}

/// Configuration for MBI
#[derive(Debug, Clone)]
pub struct MbiConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Enable model minimization
    pub minimize_models: bool,
    /// Enable interpolant simplification
    pub simplify_interpolants: bool,
}

impl Default for MbiConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            minimize_models: true,
            simplify_interpolants: true,
        }
    }
}

/// An interpolant from MBI
#[derive(Debug, Clone)]
pub struct MbiInterpolant {
    /// The interpolant formula
    pub formula: TermId,
    /// Variables in the interpolant
    pub variables: HashSet<TermId>,
    /// Number of iterations to compute
    pub iterations: u32,
}

impl MbiInterpolant {
    /// Create a new interpolant
    pub fn new(formula: TermId, variables: HashSet<TermId>) -> Self {
        Self {
            formula,
            variables,
            iterations: 0,
        }
    }

    /// Get the formula
    pub fn formula(&self) -> TermId {
        self.formula
    }

    /// Get the variables
    pub fn variables(&self) -> &HashSet<TermId> {
        &self.variables
    }
}

/// Statistics for MBI
#[derive(Debug, Clone, Default)]
pub struct MbiStats {
    /// Number of interpolation attempts
    pub attempts: u64,
    /// Number of successful interpolations
    pub successes: u64,
    /// Number of models examined
    pub models_examined: u64,
    /// Total iterations
    pub total_iterations: u64,
}

/// Model-Based Interpolation solver
#[derive(Debug)]
pub struct MbiSolver {
    /// Configuration
    config: MbiConfig,
    /// Statistics
    stats: MbiStats,
    /// Common variables (interface between A and B)
    common_vars: HashSet<TermId>,
    /// Current model assignment
    current_model: HashMap<TermId, bool>,
}

impl MbiSolver {
    /// Create a new MBI solver
    pub fn new() -> Self {
        Self {
            config: MbiConfig::default(),
            stats: MbiStats::default(),
            common_vars: HashSet::new(),
            current_model: HashMap::new(),
        }
    }

    /// Create with configuration
    pub fn with_config(config: MbiConfig) -> Self {
        Self {
            config,
            stats: MbiStats::default(),
            common_vars: HashSet::new(),
            current_model: HashMap::new(),
        }
    }

    /// Compute a validated Craig interpolant for `A ∧ B` (which must be
    /// unsatisfiable).
    ///
    /// Returns `Some(I)` with
    ///
    /// 1. `A ⇒ I`,
    /// 2. `I ∧ B` unsatisfiable, and
    /// 3. `vars(I) ⊆ vars(A) ∩ vars(B)`,
    ///
    /// or `None` when the inputs are outside the supported propositional
    /// fragment, are too large to validate, or admit no interpolant (e.g.
    /// `A ∧ B` is satisfiable). The returned interpolant is always validated;
    /// this method never returns an unchecked or fabricated formula.
    pub fn interpolate(
        &mut self,
        formula_a: TermId,
        formula_b: TermId,
        manager: &mut TermManager,
    ) -> Option<MbiInterpolant> {
        self.stats.attempts += 1;

        // Only the purely propositional fragment is handled soundly.
        if !self.is_boolean_formula(formula_a, manager)
            || !self.is_boolean_formula(formula_b, manager)
        {
            return None;
        }

        // Collect variables and the shared interface.
        let vars_a = self.collect_variables(formula_a, manager);
        let vars_b = self.collect_variables(formula_b, manager);
        self.common_vars = vars_a.intersection(&vars_b).copied().collect();

        let all_vars: Vec<TermId> = vars_a.union(&vars_b).copied().collect();
        if all_vars.len() > MAX_VALIDATION_VARS {
            // Cannot validate exhaustively; refuse rather than guess.
            return None;
        }

        // A-local variables: present in A but not shared with B.
        let a_local: Vec<TermId> = vars_a.difference(&self.common_vars).copied().collect();

        // I = ∃(a_local). A, computed by exact Boolean expansion.
        let interpolant = self.project_existential(formula_a, &a_local, manager);
        self.stats.total_iterations += a_local.len() as u64;

        // Validate the three interpolant conditions exhaustively. This also
        // rejects the case where A ∧ B is satisfiable (no interpolant exists).
        if !self.validate(formula_a, formula_b, interpolant, &all_vars, manager) {
            return None;
        }

        let variables = self.collect_variables(interpolant, manager);
        self.stats.successes += 1;
        self.stats.models_examined += 1u64 << all_vars.len().min(MAX_VALIDATION_VARS);

        Some(MbiInterpolant {
            formula: interpolant,
            variables,
            iterations: a_local.len() as u32,
        })
    }

    /// Eliminate a set of Boolean variables existentially by exact expansion:
    /// `∃v. φ ≡ φ[v:=⊤] ∨ φ[v:=⊥]`.
    fn project_existential(
        &self,
        formula: TermId,
        vars: &[TermId],
        manager: &mut TermManager,
    ) -> TermId {
        let true_id = manager.mk_true();
        let false_id = manager.mk_false();
        let mut current = formula;
        for &v in vars {
            let mut map_true = FxHashMap::default();
            map_true.insert(v, true_id);
            let mut map_false = FxHashMap::default();
            map_false.insert(v, false_id);
            let with_true = manager.substitute(current, &map_true);
            let with_false = manager.substitute(current, &map_false);
            current = manager.mk_or([with_true, with_false]);
        }
        current
    }

    /// Exhaustively check the interpolant conditions for the candidate `i`
    /// over all assignments to `all_vars`.
    fn validate(
        &self,
        formula_a: TermId,
        formula_b: TermId,
        interpolant: TermId,
        all_vars: &[TermId],
        manager: &TermManager,
    ) -> bool {
        // Condition 3: vars(I) ⊆ shared.
        let ivars = self.collect_variables(interpolant, manager);
        if !ivars.is_subset(&self.common_vars) {
            return false;
        }

        let n = all_vars.len();
        if n > MAX_VALIDATION_VARS {
            return false;
        }

        for mask in 0u64..(1u64 << n) {
            let mut assign: HashMap<TermId, bool> = HashMap::with_capacity(n);
            for (bit, &v) in all_vars.iter().enumerate() {
                assign.insert(v, (mask >> bit) & 1 == 1);
            }

            let a_val = match self.eval_bool(formula_a, &assign, manager) {
                Some(b) => b,
                None => return false,
            };
            let i_val = match self.eval_bool(interpolant, &assign, manager) {
                Some(b) => b,
                None => return false,
            };
            let b_val = match self.eval_bool(formula_b, &assign, manager) {
                Some(b) => b,
                None => return false,
            };

            // Condition 1: A ⇒ I.
            if a_val && !i_val {
                return false;
            }
            // Condition 2: I ∧ B unsatisfiable.
            if i_val && b_val {
                return false;
            }
        }

        true
    }

    /// Evaluate a propositional formula under a total Boolean assignment.
    ///
    /// Returns `None` if the formula contains a node outside the supported
    /// propositional fragment or a variable missing from `assign`.
    fn eval_bool(
        &self,
        term: TermId,
        assign: &HashMap<TermId, bool>,
        manager: &TermManager,
    ) -> Option<bool> {
        // Explicit stack, plus a per-call memo keyed on `TermId`. The
        // assignment is fixed for the duration of the call and no binder is
        // ever entered (binders are outside the supported fragment), so the
        // memo is sound; without it a shared DAG is re-expanded as a tree.
        let mut memo: HashMap<TermId, bool> = HashMap::new();
        let mut stack = vec![EvalStep::Enter(term)];

        while let Some(step) = stack.pop() {
            match step {
                EvalStep::Enter(id) => {
                    if memo.contains_key(&id) {
                        continue;
                    }
                    let t = manager.get(id)?;
                    match &t.kind {
                        TermKind::True => {
                            memo.insert(id, true);
                        }
                        TermKind::False => {
                            memo.insert(id, false);
                        }
                        TermKind::Var(_) => {
                            let value = assign.get(&id).copied()?;
                            memo.insert(id, value);
                        }
                        TermKind::Not(a) => {
                            stack.push(EvalStep::Combine(id));
                            stack.push(EvalStep::Enter(*a));
                        }
                        TermKind::And(args) | TermKind::Or(args) => {
                            stack.push(EvalStep::Combine(id));
                            for &a in args.iter() {
                                stack.push(EvalStep::Enter(a));
                            }
                        }
                        TermKind::Implies(a, b) | TermKind::Xor(a, b) | TermKind::Eq(a, b) => {
                            stack.push(EvalStep::Combine(id));
                            stack.push(EvalStep::Enter(*a));
                            stack.push(EvalStep::Enter(*b));
                        }
                        TermKind::Ite(c, _, _) => {
                            // Only the taken branch is ever evaluated, so an
                            // unsupported branch on the untaken side must not
                            // turn the whole evaluation into `None`.
                            stack.push(EvalStep::Combine(id));
                            stack.push(EvalStep::Enter(*c));
                        }
                        // Anything outside the propositional fragment is not
                        // evaluable here; report it honestly rather than
                        // defaulting to a Boolean value.
                        _ => return None,
                    }
                }
                EvalStep::Combine(id) => {
                    let t = manager.get(id)?;
                    let value = match &t.kind {
                        TermKind::Not(a) => !memo.get(a).copied()?,
                        TermKind::And(args) => {
                            let mut acc = true;
                            for a in args.iter() {
                                acc &= memo.get(a).copied()?;
                            }
                            acc
                        }
                        TermKind::Or(args) => {
                            let mut acc = false;
                            for a in args.iter() {
                                acc |= memo.get(a).copied()?;
                            }
                            acc
                        }
                        TermKind::Implies(a, b) => {
                            !memo.get(a).copied()? || memo.get(b).copied()?
                        }
                        TermKind::Xor(a, b) => memo.get(a).copied()? ^ memo.get(b).copied()?,
                        TermKind::Eq(a, b) => memo.get(a).copied()? == memo.get(b).copied()?,
                        TermKind::Ite(c, then_b, else_b) => {
                            let branch = if memo.get(c).copied()? {
                                *then_b
                            } else {
                                *else_b
                            };
                            match memo.get(&branch).copied() {
                                Some(value) => value,
                                None => {
                                    // Branch not evaluated yet: schedule it
                                    // and revisit this node afterwards.
                                    stack.push(EvalStep::Combine(id));
                                    stack.push(EvalStep::Enter(branch));
                                    continue;
                                }
                            }
                        }
                        _ => return None,
                    };
                    memo.insert(id, value);
                }
            }
        }

        memo.get(&term).copied()
    }

    /// Check whether `term` lies entirely within the supported propositional
    /// fragment (Boolean variables and Boolean connectives).
    fn is_boolean_formula(&self, term: TermId, manager: &TermManager) -> bool {
        // Explicit stack plus a visited set: the recursive form both
        // overflowed on deep formulas and re-expanded shared subterms
        // exponentially. Every node must be in the fragment, so a single
        // failure is decisive and the traversal order is irrelevant.
        let bool_sort = manager.sorts.bool_sort;
        let mut stack = vec![term];
        let mut visited: HashSet<TermId> = HashSet::new();

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let Some(t) = manager.get(id) else {
                return false;
            };
            match &t.kind {
                TermKind::True | TermKind::False => {}
                TermKind::Var(_) => {
                    if t.sort != bool_sort {
                        return false;
                    }
                }
                TermKind::Not(a) => stack.push(*a),
                TermKind::And(args) | TermKind::Or(args) => {
                    stack.extend(args.iter().copied());
                }
                TermKind::Implies(a, b) | TermKind::Xor(a, b) | TermKind::Eq(a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                TermKind::Ite(c, then_b, else_b) => {
                    stack.push(*c);
                    stack.push(*then_b);
                    stack.push(*else_b);
                }
                // Everything else is outside the propositional fragment.
                _ => return false,
            }
        }

        true
    }

    /// Collect the variables occurring in a formula.
    fn collect_variables(&self, formula: TermId, manager: &TermManager) -> HashSet<TermId> {
        let mut vars = HashSet::new();
        let mut worklist = vec![formula];
        let mut visited = HashSet::new();

        while let Some(term) = worklist.pop() {
            if visited.contains(&term) {
                continue;
            }
            visited.insert(term);

            let t = match manager.get(term) {
                Some(t) => t,
                None => continue,
            };

            match &t.kind {
                TermKind::Var(_) => {
                    vars.insert(term);
                }
                // Descend through *every* other kind via the exhaustive
                // child accessor. The previous per-kind list silently
                // dropped the children of function applications, array,
                // string, bit-vector and floating-point operations, `let`
                // and `match` — so variables under them were never reported,
                // and the caller (interpolant validation) then quantified
                // over an incomplete variable set.
                other => worklist.extend(crate::ast::traversal::get_children(other)),
            }
        }

        vars
    }

    /// Set a model for interpolation
    pub fn set_model(&mut self, model: HashMap<TermId, bool>) {
        self.current_model = model;
    }

    /// Get the common variables
    pub fn common_variables(&self) -> &HashSet<TermId> {
        &self.common_vars
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &MbiConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &MbiStats {
        &self.stats
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.common_vars.clear();
        self.current_model.clear();
    }

    /// Compute a validated sequence interpolant for a formula chain.
    ///
    /// Given formulas `A_1, ..., A_n` whose conjunction is unsatisfiable,
    /// returns interpolants `I_1, ..., I_{n-1}` satisfying
    ///
    /// - `A_1 ⇒ I_1`,
    /// - `I_i ∧ A_{i+1} ⇒ I_{i+1}` for `1 ≤ i < n-1`, and
    /// - `I_{n-1} ∧ A_n` is unsatisfiable.
    ///
    /// Each `I_i` is the validated pairwise interpolant of the prefix
    /// `A_1 ∧ ... ∧ A_i` against the suffix `A_{i+1} ∧ ... ∧ A_n`; the
    /// inductive linkage condition is then verified exhaustively. Returns
    /// `None` if any step is outside the supported propositional fragment, is
    /// too large to validate, or fails the sequence conditions.
    pub fn sequence_interpolate(
        &mut self,
        formulas: &[TermId],
        manager: &mut TermManager,
    ) -> Option<Vec<MbiInterpolant>> {
        if formulas.len() < 2 {
            return None;
        }

        let mut interpolants = Vec::new();

        for i in 0..formulas.len() - 1 {
            let prefix = self.conjoin(&formulas[..=i], manager);
            let suffix = self.conjoin(&formulas[i + 1..], manager);

            interpolants.push(self.interpolate(prefix, suffix, manager)?);
        }

        // Verify the inductive linkage I_i ∧ A_{i+1} ⇒ I_{i+1} exhaustively.
        if !self.validate_sequence(formulas, &interpolants, manager) {
            return None;
        }

        Some(interpolants)
    }

    /// Verify the inductive sequence-interpolant conditions over the union of
    /// all variables (exhaustively). Returns `false` if the union exceeds
    /// [`MAX_VALIDATION_VARS`] or any condition is violated.
    fn validate_sequence(
        &self,
        formulas: &[TermId],
        interpolants: &[MbiInterpolant],
        manager: &TermManager,
    ) -> bool {
        if interpolants.len() + 1 != formulas.len() {
            return false;
        }

        // Union of all variables across the chain and interpolants.
        let mut all: HashSet<TermId> = HashSet::new();
        for &f in formulas {
            all.extend(self.collect_variables(f, manager));
        }
        for interp in interpolants {
            all.extend(interp.variables.iter().copied());
        }
        let all_vars: Vec<TermId> = all.into_iter().collect();
        let n = all_vars.len();
        if n > MAX_VALIDATION_VARS {
            return false;
        }

        for mask in 0u64..(1u64 << n) {
            let mut assign: HashMap<TermId, bool> = HashMap::with_capacity(n);
            for (bit, &v) in all_vars.iter().enumerate() {
                assign.insert(v, (mask >> bit) & 1 == 1);
            }

            // I_i ∧ A_{i+1} ⇒ I_{i+1} for interior links.
            for i in 0..interpolants.len().saturating_sub(1) {
                let ii = self.eval_bool(interpolants[i].formula, &assign, manager);
                let a_next = self.eval_bool(formulas[i + 1], &assign, manager);
                let i_next = self.eval_bool(interpolants[i + 1].formula, &assign, manager);
                match (ii, a_next, i_next) {
                    (Some(ii), Some(a_next), Some(i_next)) => {
                        if ii && a_next && !i_next {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }

        true
    }

    /// Conjoin a sequence of formulas
    fn conjoin(&self, formulas: &[TermId], manager: &mut TermManager) -> TermId {
        if formulas.is_empty() {
            return manager.mk_true();
        }
        if formulas.len() == 1 {
            return formulas[0];
        }
        manager.mk_and(formulas.iter().copied())
    }

    /// Tree interpolation is not supported.
    ///
    /// A sound tree interpolant requires the per-node interpolants to satisfy
    /// the tree linkage conditions; this module only implements binary and
    /// (validated) sequence interpolation. Rather than return a fabricated
    /// (e.g. trivially `true`) interpolant, this method honestly returns
    /// `None`.
    pub fn tree_interpolate(
        &mut self,
        _root: TermId,
        _children: &[TermId],
        _manager: &mut TermManager,
    ) -> Option<MbiInterpolant> {
        None
    }
}

impl Default for MbiSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbi_config() {
        let config = MbiConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!(config.minimize_models);
        assert!(config.simplify_interpolants);
    }

    #[test]
    fn test_mbi_interpolant() {
        let t = TermId::from(1u32);
        let mut vars = HashSet::new();
        vars.insert(t);

        let interp = MbiInterpolant::new(t, vars);
        assert_eq!(interp.formula(), t);
        assert_eq!(interp.iterations, 0);
    }

    #[test]
    fn test_mbi_solver_creation() {
        let solver = MbiSolver::new();
        assert_eq!(solver.stats().attempts, 0);
        assert!(solver.common_variables().is_empty());
    }

    #[test]
    fn test_mbi_stats() {
        let stats = MbiStats::default();
        assert_eq!(stats.attempts, 0);
        assert_eq!(stats.successes, 0);
    }

    #[test]
    fn test_mbi_with_config() {
        let config = MbiConfig {
            max_iterations: 50,
            minimize_models: false,
            simplify_interpolants: true,
        };
        let solver = MbiSolver::with_config(config.clone());
        assert_eq!(solver.config.max_iterations, 50);
        assert!(!solver.config.minimize_models);
    }

    #[test]
    fn test_mbi_reset() {
        let mut solver = MbiSolver::new();
        solver.common_vars.insert(TermId::from(1u32));
        solver.reset();
        assert!(solver.common_variables().is_empty());
    }

    #[test]
    fn test_mbi_set_model() {
        let mut solver = MbiSolver::new();
        let mut model = HashMap::new();
        model.insert(TermId::from(1u32), true);
        solver.set_model(model);
        assert!(!solver.current_model.is_empty());
    }

    /// Brute-force check that `formula` is unsatisfiable under all Boolean
    /// assignments to `vars`.
    fn is_unsat(
        solver: &MbiSolver,
        formula: TermId,
        vars: &[TermId],
        manager: &TermManager,
    ) -> bool {
        for mask in 0u64..(1u64 << vars.len()) {
            let mut assign = HashMap::new();
            for (bit, &v) in vars.iter().enumerate() {
                assign.insert(v, (mask >> bit) & 1 == 1);
            }
            if solver.eval_bool(formula, &assign, manager) == Some(true) {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_interpolate_basic_shared() {
        // A: x ∧ y   B: ¬x   (shared: x). A ∧ B unsat.
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", bool_sort);
        let y = m.mk_var("y", bool_sort);
        let a = m.mk_and([x, y]);
        let not_x = m.mk_not(x);
        let b = not_x;

        let mut solver = MbiSolver::new();
        let interp = solver
            .interpolate(a, b, &mut m)
            .expect("interpolant should exist");

        // I must only mention shared variable x.
        assert!(interp.variables().iter().all(|&v| v == x));

        // Verify A ⇒ I and I ∧ B unsat by brute force.
        let all = [x, y];
        let not_i = m.mk_not(interp.formula());
        let a_and_not_i = m.mk_and([a, not_i]);
        assert!(is_unsat(&solver, a_and_not_i, &all, &m), "A ⇒ I must hold");
        let i_and_b = m.mk_and([interp.formula(), b]);
        assert!(is_unsat(&solver, i_and_b, &all, &m), "I ∧ B must be unsat");
    }

    #[test]
    fn test_interpolate_satisfiable_returns_none() {
        // A: x   B: y   (A ∧ B satisfiable) -> no interpolant.
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", bool_sort);
        let y = m.mk_var("y", bool_sort);

        let mut solver = MbiSolver::new();
        assert!(solver.interpolate(x, y, &mut m).is_none());
    }

    #[test]
    fn test_interpolate_non_boolean_returns_none() {
        // Arithmetic atom is outside the supported fragment.
        let mut m = TermManager::new();
        let int_sort = m.sorts.int_sort;
        let x = m.mk_var("x", int_sort);
        let zero = m.mk_int(0);
        let a = m.mk_gt(x, zero);
        let b = m.mk_le(x, zero);

        let mut solver = MbiSolver::new();
        assert!(solver.interpolate(a, b, &mut m).is_none());
    }

    #[test]
    fn test_interpolate_a_local_eliminated() {
        // A: (x ∧ z)   B: (¬x)   z is A-local, must be eliminated: I ≡ x.
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", bool_sort);
        let z = m.mk_var("z", bool_sort);
        let a = m.mk_and([x, z]);
        let b = m.mk_not(x);

        let mut solver = MbiSolver::new();
        let interp = solver
            .interpolate(a, b, &mut m)
            .expect("interpolant should exist");
        // No A-local variable z may appear.
        assert!(!interp.variables().contains(&z));
        assert!(interp.variables().iter().all(|&v| v == x));
    }

    #[test]
    fn test_sequence_interpolate() {
        // A1: x, A2: (¬x ∨ y), A3: ¬y. Conjunction unsat.
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", bool_sort);
        let y = m.mk_var("y", bool_sort);
        let a1 = x;
        let nx = m.mk_not(x);
        let a2 = m.mk_or([nx, y]);
        let a3 = m.mk_not(y);

        let mut solver = MbiSolver::new();
        let seq = solver
            .sequence_interpolate(&[a1, a2, a3], &mut m)
            .expect("sequence interpolants should exist");
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_tree_interpolate_is_none() {
        let mut m = TermManager::new();
        let bool_sort = m.sorts.bool_sort;
        let x = m.mk_var("x", bool_sort);
        let mut solver = MbiSolver::new();
        assert!(solver.tree_interpolate(x, &[x], &mut m).is_none());
    }
}

#[cfg(test)]
mod deep_walk_tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_collect_variables_sees_apply_arguments() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let zero = manager.mk_int(0);
        let formula = manager.mk_gt(fx, zero);

        let solver = MbiSolver::new();
        let vars = solver.collect_variables(formula, &manager);
        assert!(vars.contains(&x), "variable under an application was lost");
    }

    #[test]
    fn test_eval_bool_shared_dag_is_fast() {
        // 55 levels of a two-strand DAG: without memoization this is 2^55
        // evaluations.
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let p = manager.mk_var("p", bool_sort);
        let q = manager.mk_var("q", bool_sort);
        let (mut a, mut b) = (p, q);
        for _ in 0..55 {
            let next_a = manager.mk_implies(a, b);
            let next_b = manager.mk_implies(b, a);
            a = next_a;
            b = next_b;
        }

        let solver = MbiSolver::new();
        let mut assign = HashMap::new();
        assign.insert(p, true);
        assign.insert(q, false);
        assert!(solver.eval_bool(a, &assign, &manager).is_some());
        assert!(solver.is_boolean_formula(a, &manager));
    }

    #[test]
    fn test_eval_bool_ite_does_not_evaluate_untaken_branch() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let int_sort = manager.sorts.int_sort;
        let p = manager.mk_var("p", bool_sort);
        let q = manager.mk_var("q", bool_sort);
        // The else-branch is outside the propositional fragment.
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let unsupported = manager.mk_gt(x, zero);
        let ite = manager.mk_ite(p, q, unsupported);

        let solver = MbiSolver::new();
        let mut assign = HashMap::new();
        assign.insert(p, true);
        assign.insert(q, false);
        assert_eq!(solver.eval_bool(ite, &assign, &manager), Some(false));

        // Taking the unsupported branch is reported honestly.
        assign.insert(p, false);
        assert_eq!(solver.eval_bool(ite, &assign, &manager), None);
    }

    #[test]
    fn test_eval_bool_deep_nesting_does_not_overflow() {
        // 128 KiB / 6_250 levels: the house ratio for a genuinely nested
        // `Not` chain (see oxiz-spacer/src/walk.rs's `DEEP_STACK`/`DEEP_DEPTH`).
        // `mk_not` folds double negation (`not(not(p)) == p`), so a loop of
        // `mk_not` calls oscillates between depth 0 and 1 and never actually
        // nests -- intern the `Not` nodes directly so the chain really is
        // this deep. `DEEP_DEPTH` is even so the parity below stays meaningful.
        const DEEP_STACK: usize = 1 << 17;
        const DEEP_DEPTH: usize = 6_250;

        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let bool_sort = manager.sorts.bool_sort;
                let p = manager.mk_var("p", bool_sort);
                let mut formula = p;
                for _ in 0..DEEP_DEPTH {
                    formula = manager.intern_term(TermKind::Not(formula), bool_sort);
                }

                let solver = MbiSolver::new();
                let mut assign = HashMap::new();
                assign.insert(p, true);
                let value = solver.eval_bool(formula, &assign, &manager);
                let is_bool = solver.is_boolean_formula(formula, &manager);
                (value, is_bool)
            })
            .expect("thread spawn should succeed");

        let (value, is_bool) = handle.join().expect("deep evaluation must not overflow");
        // DEEP_DEPTH negations is an even number of negations.
        assert_eq!(value, Some(true));
        assert!(is_bool);
    }
}
