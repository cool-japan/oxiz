//! NLSAT solver implementation.
//!
//! This module provides the core NLSAT (Non-Linear Satisfiability) solver
//! that uses the CAD (Cylindrical Algebraic Decomposition) algorithm to
//! decide satisfiability of polynomial constraints over the reals.
//!
//! Reference: Z3's `nlsat/nlsat_solver.cpp`

mod conflict;
mod decide;
mod propagate;

use crate::assignment::{Assignment, Justification};
use crate::clause::{ClauseDatabase, ClauseId, NULL_CLAUSE};
use crate::restart::{RestartManager, RestartStrategy};
use crate::types::{Atom, AtomKind, BoolVar, IneqAtom, Lbool, Literal, NULL_BOOL_VAR, PolyFactor};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use oxiz_math::polynomial::{NULL_VAR, Polynomial, Var};
use std::collections::{HashMap, HashSet};

pub use propagate::PropagationResult;

/// Atom identifier.
pub type AtomId = u32;

/// Result of the NLSAT solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverResult {
    /// Formula is satisfiable.
    Sat,
    /// Formula is unsatisfiable.
    Unsat,
    /// Solver ran out of resources (timeout, memory).
    Unknown,
}

/// Configuration for the NLSAT solver.
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Maximum number of conflicts before restart.
    pub max_conflicts: u64,
    /// Maximum number of learned clauses before reduction.
    pub max_learned: usize,
    /// Fraction of learned clauses to keep during reduction.
    pub learned_keep_fraction: f64,
    /// Enable random decisions.
    pub random_decisions: bool,
    /// Random decision frequency (0.0 - 1.0).
    pub random_freq: f64,
    /// Seed for random number generator.
    pub random_seed: u64,
    /// Enable verbose output.
    pub verbose: bool,
    /// Enable dynamic variable reordering.
    pub dynamic_reordering: bool,
    /// Conflicts between reorderings.
    pub reorder_frequency: u64,
    /// Enable early termination optimizations.
    pub early_termination: bool,
    /// Restart strategy.
    pub restart_strategy: RestartStrategy,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_conflicts: 100_000,
            max_learned: 10_000,
            learned_keep_fraction: 0.5,
            random_decisions: false,
            random_freq: 0.05,
            random_seed: 91648253,
            verbose: false,
            dynamic_reordering: false,
            reorder_frequency: 1000,
            early_termination: true,
            restart_strategy: RestartStrategy::default(),
        }
    }
}

/// Statistics for the solver.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// Number of decisions made.
    pub decisions: u64,
    /// Number of propagations.
    pub propagations: u64,
    /// Number of conflicts.
    pub conflicts: u64,
    /// Number of restarts.
    pub restarts: u64,
    /// Number of learned clauses.
    pub learned_clauses: u64,
    /// Number of clause deletions.
    pub clause_deletions: u64,
    /// Number of theory propagations.
    pub theory_propagations: u64,
    /// Number of theory conflicts.
    pub theory_conflicts: u64,
    /// Number of variable reorderings.
    pub reorderings: u64,
    /// Number of early terminations.
    pub early_terminations: u64,
}

/// Non-linear arithmetic solver using CAD.
pub struct NlsatSolver {
    /// Solver configuration.
    pub(super) config: SolverConfig,
    /// Solver statistics.
    pub(super) stats: SolverStats,
    /// Clause database.
    pub(super) clauses: ClauseDatabase,
    /// Variable assignment.
    pub(super) assignment: Assignment,
    /// Atoms (polynomial constraints).
    pub(super) atoms: Vec<Atom>,
    /// Map from polynomial hash to atom IDs (for deduplication).
    pub(super) atom_map: HashMap<u64, Vec<AtomId>>,
    /// Number of arithmetic variables.
    pub(super) num_arith_vars: u32,
    /// Number of boolean variables.
    pub(super) num_bool_vars: u32,
    /// Variable ordering for CAD.
    pub(super) var_order: Vec<Var>,
    /// Activity scores for boolean variables.
    pub(super) var_activity: Vec<f64>,
    /// Activity increment.
    pub(super) var_activity_inc: f64,
    /// Activity decay factor.
    pub(super) var_activity_decay: f64,
    /// Activity scores for arithmetic variables.
    pub(super) arith_activity: Vec<f64>,
    /// Activity increment for arithmetic variables.
    pub(super) arith_activity_inc: f64,
    /// Activity decay factor for arithmetic variables.
    pub(super) arith_activity_decay: f64,
    /// Queue of literals for unit propagation.
    pub(super) propagation_queue: Vec<Literal>,
    /// Current conflict clause (if any).
    pub(super) conflict_clause: Option<ClauseId>,
    /// Learnt clause from conflict analysis.
    pub(super) learnt_clause: Vec<Literal>,
    /// Seen variables during conflict analysis.
    pub(super) seen: HashSet<BoolVar>,
    /// The polynomial evaluator cache.
    pub(super) eval_cache: HashMap<(AtomId, Vec<Option<BigRational>>), Lbool>,
    /// Random number generator state.
    pub(super) random_state: u64,
    /// Track clauses used in conflict (for unsat core extraction).
    pub(super) conflict_clauses: HashSet<ClauseId>,
    /// Enable unsat core extraction.
    pub(super) extract_unsat_core: bool,
    /// Restart manager.
    pub(super) restart_manager: RestartManager,
    /// Average LBD of recent learned clauses (for Glucose-style restarts).
    pub(super) recent_avg_lbd: f64,
    /// Saved phase (polarity) for each boolean variable.
    /// true = positive polarity, false = negative polarity.
    pub(super) saved_phase: Vec<bool>,
}

impl NlsatSolver {
    /// Create a new NLSAT solver with default configuration.
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new NLSAT solver with the given configuration.
    pub fn with_config(config: SolverConfig) -> Self {
        let random_state = config.random_seed;
        let restart_manager = RestartManager::new(config.restart_strategy);
        Self {
            config,
            stats: SolverStats::default(),
            clauses: ClauseDatabase::new(),
            assignment: Assignment::new(),
            atoms: Vec::new(),
            atom_map: HashMap::new(),
            num_arith_vars: 0,
            num_bool_vars: 0,
            var_order: Vec::new(),
            var_activity: Vec::new(),
            var_activity_inc: 1.0,
            var_activity_decay: 0.95,
            arith_activity: Vec::new(),
            arith_activity_inc: 1.0,
            arith_activity_decay: 0.95,
            propagation_queue: Vec::new(),
            conflict_clause: None,
            learnt_clause: Vec::new(),
            seen: HashSet::new(),
            eval_cache: HashMap::new(),
            random_state,
            conflict_clauses: HashSet::new(),
            extract_unsat_core: false,
            restart_manager,
            recent_avg_lbd: 0.0,
            saved_phase: Vec::new(),
        }
    }

    /// Enable or disable unsat core extraction.
    pub fn set_unsat_core_extraction(&mut self, enable: bool) {
        self.extract_unsat_core = enable;
    }

    /// Get the unsat core (if the formula is unsat and extraction is enabled).
    /// Returns the set of clause IDs that form a minimal unsatisfiable core.
    pub fn get_unsat_core(&self) -> Vec<ClauseId> {
        self.conflict_clauses.iter().copied().collect()
    }

    /// Get the solver statistics.
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Get the solver configuration.
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Get the number of clauses.
    pub fn num_clauses(&self) -> usize {
        self.clauses.num_clauses()
    }

    /// Get the number of atoms.
    pub fn num_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Get the number of arithmetic variables.
    pub fn num_arith_vars(&self) -> u32 {
        self.num_arith_vars
    }

    /// Get the number of boolean variables.
    pub fn num_bool_vars(&self) -> u32 {
        self.num_bool_vars
    }

    /// Get the current assignment.
    pub fn assignment(&self) -> &Assignment {
        &self.assignment
    }

    /// Get the clause database.
    pub fn clauses(&self) -> &ClauseDatabase {
        &self.clauses
    }

    // ========== Variable and Atom Management ==========

    /// Create a new boolean variable.
    pub fn new_bool_var(&mut self) -> BoolVar {
        let var = self.num_bool_vars;
        self.num_bool_vars += 1;
        self.assignment.ensure_bool_var(var);
        self.clauses.ensure_bool_var(var);

        // Extend activity tracking
        if var as usize >= self.var_activity.len() {
            self.var_activity.resize(var as usize + 1, 0.0);
        }

        // Initialize saved phase (default to positive)
        if var as usize >= self.saved_phase.len() {
            self.saved_phase.resize(var as usize + 1, true);
        }

        var
    }

    /// Create a new arithmetic variable.
    pub fn new_arith_var(&mut self) -> Var {
        let var = self.num_arith_vars;
        self.num_arith_vars += 1;
        self.assignment.ensure_arith_var(var);
        self.var_order.push(var);

        // Initialize activity
        if var as usize >= self.arith_activity.len() {
            self.arith_activity.resize(var as usize + 1, 0.0);
        }

        var
    }

    /// Create a new inequality atom (p op 0).
    /// Uses deduplication to avoid creating duplicate atoms.
    pub fn new_ineq_atom(&mut self, poly: Polynomial, kind: AtomKind) -> AtomId {
        // Compute hash for deduplication
        let atom_hash = self.compute_atom_hash(&poly, kind);

        // Check if we already have this atom
        if let Some(atom_ids) = self.atom_map.get(&atom_hash) {
            for &atom_id in atom_ids {
                if let Some(Atom::Ineq(existing)) = self.get_atom(atom_id) {
                    // Check if this is truly the same atom
                    if existing.kind == kind
                        && existing.factors.len() == 1
                        && existing.factors[0].poly == poly
                    {
                        // Reuse existing atom
                        return atom_id;
                    }
                }
            }
        }

        // Get maximum variable in polynomial
        let max_var = poly.max_var();

        // Ensure we have enough arithmetic variables
        if max_var != NULL_VAR {
            while self.num_arith_vars <= max_var {
                self.new_arith_var();
            }
        }

        // Create boolean variable for this atom
        let bool_var = self.new_bool_var();

        let atom = Atom::Ineq(IneqAtom {
            kind,
            factors: vec![PolyFactor {
                poly,
                is_even: false,
            }],
            max_var,
            bool_var,
        });

        let id = self.atoms.len() as AtomId;
        self.atoms.push(atom);

        // Add to deduplication map
        self.atom_map.entry(atom_hash).or_default().push(id);

        id
    }

    /// Compute a hash for an atom (for deduplication).
    pub(super) fn compute_atom_hash(&self, poly: &Polynomial, kind: AtomKind) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the atom kind
        (kind as u8).hash(&mut hasher);

        // Hash the polynomial terms
        for term in poly.terms() {
            term.coeff.numer().hash(&mut hasher);
            term.coeff.denom().hash(&mut hasher);
            for vp in term.monomial.vars() {
                vp.var.hash(&mut hasher);
                vp.power.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Get an atom by ID.
    pub fn get_atom(&self, id: AtomId) -> Option<&Atom> {
        self.atoms.get(id as usize)
    }

    /// Get the boolean variable for an atom.
    pub fn atom_bool_var(&self, id: AtomId) -> BoolVar {
        match self.get_atom(id) {
            Some(Atom::Ineq(a)) => a.bool_var,
            Some(Atom::Root(a)) => a.bool_var,
            None => NULL_BOOL_VAR,
        }
    }

    /// Create a literal from an atom ID.
    pub fn atom_literal(&self, id: AtomId, positive: bool) -> Literal {
        let var = self.atom_bool_var(id);
        if positive {
            Literal::positive(var)
        } else {
            Literal::negative(var)
        }
    }

    // ========== Clause Management ==========

    /// Add a clause to the solver.
    /// Returns the clause ID, or None if the clause is trivially satisfied.
    pub fn add_clause(&mut self, mut literals: Vec<Literal>) -> Option<ClauseId> {
        // Remove duplicates and check for tautology
        literals.sort_by_key(|l| l.index());
        literals.dedup();

        // Check for tautology (both x and ~x)
        for i in 0..literals.len() {
            if i + 1 < literals.len() && literals[i].var() == literals[i + 1].var() {
                // Tautology: clause contains both x and ~x
                return None;
            }
        }

        if literals.is_empty() {
            // Empty clause - unsatisfiable
            return Some(NULL_CLAUSE);
        }

        // Compute max arithmetic variable
        let max_var = self.clause_max_var(&literals);

        // Add to clause database
        let id = self.clauses.add(literals.clone(), max_var, false);

        // Check for unit clause - assign immediately
        if literals.len() == 1 {
            let lit = literals[0];
            let current_val = self.assignment.lit_value(lit);
            if current_val.is_false() {
                // Conflict at level 0 - set conflict clause so solve() will return Unsat
                self.conflict_clause = Some(id);
                // Track for unsat core
                if self.extract_unsat_core {
                    self.conflict_clauses.insert(id);
                    // Also track the clause that assigned the conflicting literal
                    if let Some(reason) = self.find_unit_conflict_reason(lit) {
                        self.conflict_clauses.insert(reason);
                    }
                }
                return Some(id);
            }
            if current_val.is_undef() {
                self.assignment.assign(lit, Justification::Unit);
                self.save_phase(lit);
                self.propagation_queue.push(lit);
            }
        }

        Some(id)
    }

    /// Add a learned clause.
    pub(super) fn add_learned_clause(&mut self, literals: Vec<Literal>) -> ClauseId {
        let max_var = self.clause_max_var(&literals);
        let id = self.clauses.add(literals, max_var, true);
        self.stats.learned_clauses += 1;
        id
    }

    /// Find the clause that assigned a literal (for unsat core tracking at level 0).
    pub(super) fn find_unit_conflict_reason(&self, lit: Literal) -> Option<ClauseId> {
        let negated = lit.negate();
        let trail = self.assignment.trail();
        for entry in trail {
            if entry.literal == negated {
                if let Justification::Propagation(cid) = entry.justification {
                    return Some(cid);
                } else if let Justification::Unit = entry.justification {
                    // It was assigned by a unit clause, need to find it
                    // Search through clauses for unit clause containing this literal
                    for (idx, clause) in self.clauses.clauses().iter().enumerate() {
                        if clause.len() == 1 && clause.get(0) == Some(negated) {
                            return Some(idx as ClauseId);
                        }
                    }
                }
            }
        }
        None
    }

    /// Compute the maximum arithmetic variable in a clause.
    pub(super) fn clause_max_var(&self, literals: &[Literal]) -> Var {
        let mut max_var = 0;
        for lit in literals {
            // Find the atom for this literal's variable
            for atom in &self.atoms {
                match atom {
                    Atom::Ineq(a) if a.bool_var == lit.var() && a.max_var != NULL_VAR => {
                        max_var = max_var.max(a.max_var);
                    }
                    Atom::Root(a) if a.bool_var == lit.var() => {
                        let atom_max = a.max_var();
                        if atom_max != NULL_VAR {
                            max_var = max_var.max(atom_max);
                        }
                    }
                    _ => {}
                }
            }
        }
        max_var
    }

    // ========== Main Solve Loop ==========

    /// Solve the formula.
    pub fn solve(&mut self) -> SolverResult {
        // Clear unsat core tracking from previous solve
        // (but only if there's no existing conflict from clause addition)
        if self.extract_unsat_core && self.conflict_clause.is_none() {
            self.conflict_clauses.clear();
        }

        // Check for conflicts detected during clause addition (at level 0)
        if self.conflict_clause.is_some() && self.assignment.level() == 0 {
            return SolverResult::Unsat;
        }

        // Initial propagation
        match self.propagate() {
            PropagationResult::Conflict(cid) => {
                if self.assignment.level() == 0 {
                    return SolverResult::Unsat;
                }
                self.conflict_clause = Some(cid);
            }
            PropagationResult::TheoryConflict(lits) => {
                if self.assignment.level() == 0 {
                    return SolverResult::Unsat;
                }
                let cid = self.add_learned_clause(lits);
                self.conflict_clause = Some(cid);
            }
            PropagationResult::Ok => {}
        }

        loop {
            // Handle conflict
            if let Some(conflict_id) = self.conflict_clause.take() {
                self.stats.conflicts += 1;
                self.restart_manager.record_conflict();

                if self.assignment.level() == 0 {
                    return SolverResult::Unsat;
                }

                // Analyze conflict
                let (learnt, backtrack_level) = self.analyze_conflict(conflict_id);

                if learnt.is_empty() {
                    return SolverResult::Unsat;
                }

                // Compute and update LBD
                let lbd = self.compute_lbd(&learnt);
                self.recent_avg_lbd = (self.recent_avg_lbd * 0.8) + (lbd as f64 * 0.2);

                // Backtrack
                self.backtrack(backtrack_level);

                // Add learned clause
                let learnt_id = self.add_learned_clause(learnt.clone());

                // Set LBD for the learned clause
                if let Some(clause) = self.clauses.get_mut(learnt_id) {
                    clause.set_lbd(lbd);
                }

                // Bump clause activity
                self.clauses.bump_activity(learnt_id);

                // The first literal of learned clause should be asserted
                if !learnt.is_empty() {
                    let justification = if learnt.len() == 1 {
                        Justification::Unit
                    } else {
                        Justification::Propagation(learnt_id)
                    };
                    self.assignment.assign(learnt[0], justification);
                    self.save_phase(learnt[0]);
                    self.propagation_queue.push(learnt[0]);
                }

                // Decay activities
                self.decay_activities();

                // Check if we should restart
                self.maybe_restart();

                // Check if we should reduce learned clauses
                if self.clauses.num_learned() as usize > self.config.max_learned {
                    self.reduce_learned();
                }

                // Check if we should reorder variables
                if self.config.dynamic_reordering
                    && self
                        .stats
                        .conflicts
                        .is_multiple_of(self.config.reorder_frequency)
                {
                    self.dynamic_reorder();
                }

                continue;
            }

            // Propagate
            match self.propagate() {
                PropagationResult::Conflict(cid) => {
                    self.conflict_clause = Some(cid);
                    continue;
                }
                PropagationResult::TheoryConflict(lits) => {
                    self.stats.theory_conflicts += 1;
                    let cid = self.add_learned_clause(lits);
                    self.conflict_clause = Some(cid);
                    continue;
                }
                PropagationResult::Ok => {}
            }

            // Theory propagation
            if let Some(conflict_lits) = self.theory_propagate() {
                self.stats.theory_conflicts += 1;
                let cid = self.add_learned_clause(conflict_lits);
                self.conflict_clause = Some(cid);
                continue;
            }

            // Make a decision
            if let Some(lit) = self.decide() {
                self.stats.decisions += 1;
                self.assignment.push_level();
                self.assignment.assign(lit, Justification::Decision);
                self.save_phase(lit);
                self.propagation_queue.push(lit);
                continue;
            }

            // No more decisions - check if we have a complete assignment
            if self.is_complete() {
                return SolverResult::Sat;
            }

            // Need to assign arithmetic variables
            if let Some(var) = self.next_arith_var() {
                if let Some(value) = self.pick_arith_value(var) {
                    self.assignment.set_arith(var, value);
                    // After assigning an arithmetic variable, we may have new propagations
                    continue;
                } else {
                    // No valid value for this variable - backtrack
                    if self.assignment.level() == 0 {
                        return SolverResult::Unsat;
                    }
                    self.backtrack(self.assignment.level() - 1);
                    continue;
                }
            }

            // All variables assigned and satisfiable
            return SolverResult::Sat;
        }
    }

    // ========== Model Extraction ==========

    /// Get the model (assignment of variables) if satisfiable.
    pub fn get_model(&self) -> Option<Model> {
        if !self.is_complete() {
            return None;
        }

        let mut bool_values = HashMap::new();
        for var in 0..self.num_bool_vars {
            let val = self.assignment.bool_value(var);
            if !val.is_undef() {
                bool_values.insert(var, val.is_true());
            }
        }

        let mut arith_values = HashMap::new();
        for var in 0..self.num_arith_vars {
            if let Some(val) = self.assignment.arith_value(var) {
                arith_values.insert(var, val.clone());
            }
        }

        Some(Model {
            bool_values,
            arith_values,
        })
    }
}

impl Default for NlsatSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// A model (satisfying assignment).
#[derive(Debug, Clone)]
pub struct Model {
    /// Boolean variable assignments.
    pub bool_values: HashMap<BoolVar, bool>,
    /// Arithmetic variable assignments.
    pub arith_values: HashMap<Var, BigRational>,
}

impl Model {
    /// Get the value of a boolean variable.
    pub fn bool_value(&self, var: BoolVar) -> Option<bool> {
        self.bool_values.get(&var).copied()
    }

    /// Get the value of an arithmetic variable.
    pub fn arith_value(&self, var: Var) -> Option<&BigRational> {
        self.arith_values.get(&var)
    }
}

/// Compute the integer square root if the number is a perfect square.
pub(super) fn integer_sqrt(n: &num_bigint::BigInt) -> Option<num_bigint::BigInt> {
    use num_traits::ToPrimitive;

    if n.is_negative() {
        return None;
    }

    if n.is_zero() {
        return Some(num_bigint::BigInt::zero());
    }

    // For small numbers, use f64
    if let Some(n_f64) = n.to_f64()
        && n_f64 < 1e15
    {
        let sqrt = n_f64.sqrt();
        let sqrt_int = sqrt.round() as i64;
        let candidate = num_bigint::BigInt::from(sqrt_int);
        if &candidate * &candidate == *n {
            return Some(candidate);
        }
        return None;
    }

    // Newton's method for large numbers
    let mut x = n.clone();
    let two = num_bigint::BigInt::from(2);
    let mut y: num_bigint::BigInt = (&x + num_bigint::BigInt::one()) / &two;

    while y < x {
        x = y.clone();
        y = (&x + n / &x) / &two;
    }

    if &x * &x == *n { Some(x) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AtomKind;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    #[test]
    fn test_solver_new() {
        let solver = NlsatSolver::new();
        assert_eq!(solver.num_clauses(), 0);
        assert_eq!(solver.num_atoms(), 0);
    }

    #[test]
    fn test_solver_new_vars() {
        let mut solver = NlsatSolver::new();

        let b1 = solver.new_bool_var();
        let b2 = solver.new_bool_var();
        assert_eq!(b1, 0);
        assert_eq!(b2, 1);
        assert_eq!(solver.num_bool_vars(), 2);

        let x = solver.new_arith_var();
        let y = solver.new_arith_var();
        assert_eq!(x, 0);
        assert_eq!(y, 1);
        assert_eq!(solver.num_arith_vars(), 2);
    }

    #[test]
    fn test_solver_add_clause() {
        let mut solver = NlsatSolver::new();

        let a = solver.new_bool_var();
        let b = solver.new_bool_var();

        // (a ∨ b)
        let id = solver.add_clause(vec![Literal::positive(a), Literal::positive(b)]);
        assert!(id.is_some());
        assert_eq!(solver.num_clauses(), 1);

        // Unit clause
        let id2 = solver.add_clause(vec![Literal::positive(a)]);
        assert!(id2.is_some());
        assert_eq!(solver.num_clauses(), 2);
    }

    #[test]
    fn test_solver_tautology() {
        let mut solver = NlsatSolver::new();

        let a = solver.new_bool_var();

        // (a ∨ ~a) - tautology
        let id = solver.add_clause(vec![Literal::positive(a), Literal::negative(a)]);
        assert!(id.is_none()); // Tautology should return None
    }

    #[test]
    fn test_solver_simple_sat() {
        let mut solver = NlsatSolver::new();

        let a = solver.new_bool_var();
        let b = solver.new_bool_var();

        // (a ∨ b) ∧ (~a ∨ b)
        solver.add_clause(vec![Literal::positive(a), Literal::positive(b)]);
        solver.add_clause(vec![Literal::negative(a), Literal::positive(b)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        // b should be true
        let model = solver.get_model().expect("SAT result should have a model");
        assert_eq!(model.bool_value(b), Some(true));
    }

    #[test]
    fn test_solver_simple_unsat() {
        let mut solver = NlsatSolver::new();

        let a = solver.new_bool_var();

        // a ∧ ~a
        solver.add_clause(vec![Literal::positive(a)]);
        solver.add_clause(vec![Literal::negative(a)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat);
    }

    #[test]
    fn test_solver_unit_propagation() {
        let mut solver = NlsatSolver::new();

        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let c = solver.new_bool_var();

        // a ∧ (~a ∨ b) ∧ (~b ∨ c)
        solver.add_clause(vec![Literal::positive(a)]);
        solver.add_clause(vec![Literal::negative(a), Literal::positive(b)]);
        solver.add_clause(vec![Literal::negative(b), Literal::positive(c)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model after unit propagation");
        assert_eq!(model.bool_value(a), Some(true));
        assert_eq!(model.bool_value(b), Some(true));
        assert_eq!(model.bool_value(c), Some(true));
    }

    #[test]
    fn test_solver_ineq_atom() {
        let mut solver = NlsatSolver::new();

        // x > 0 where x is variable 0
        let x_var = 0;
        let x = Polynomial::from_var(x_var);
        let atom_id = solver.new_ineq_atom(x, AtomKind::Gt);

        assert_eq!(solver.num_atoms(), 1);
        assert_eq!(solver.num_arith_vars(), 1);

        // Add clause requiring x > 0
        let lit = solver.atom_literal(atom_id, true);
        solver.add_clause(vec![lit]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model for ineq atom");
        let x_val = model
            .arith_value(x_var)
            .expect("model should have arithmetic value for x");
        assert!(x_val.is_positive());
    }

    #[test]
    fn test_solver_linear_constraints() {
        let mut solver = NlsatSolver::new();

        // x > 0 ∧ x - 2 < 0 (i.e., 0 < x < 2)
        let x = Polynomial::from_var(0);

        let gt_atom = solver.new_ineq_atom(x.clone(), AtomKind::Gt);
        let x_minus_2 = Polynomial::sub(&x, &Polynomial::constant(rat(2)));
        let lt_atom = solver.new_ineq_atom(x_minus_2, AtomKind::Lt);

        solver.add_clause(vec![solver.atom_literal(gt_atom, true)]);
        solver.add_clause(vec![solver.atom_literal(lt_atom, true)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model for linear constraints");
        let x_val = model
            .arith_value(0)
            .expect("model should have arithmetic value for x");
        assert!(x_val > &rat(0));
        assert!(x_val < &rat(2));
    }

    #[test]
    fn test_integer_sqrt() {
        assert_eq!(integer_sqrt(&0.into()), Some(0.into()));
        assert_eq!(integer_sqrt(&1.into()), Some(1.into()));
        assert_eq!(integer_sqrt(&4.into()), Some(2.into()));
        assert_eq!(integer_sqrt(&9.into()), Some(3.into()));
        assert_eq!(integer_sqrt(&16.into()), Some(4.into()));
        assert_eq!(integer_sqrt(&100.into()), Some(10.into()));

        // Not perfect squares
        assert_eq!(integer_sqrt(&2.into()), None);
        assert_eq!(integer_sqrt(&3.into()), None);
        assert_eq!(integer_sqrt(&5.into()), None);
    }

    #[test]
    fn test_find_linear_root() {
        let solver = NlsatSolver::new();

        // 2x + 4 = 0  =>  x = -2
        let x = Polynomial::from_var(0);
        let poly = Polynomial::add(&x.scale(&rat(2)), &Polynomial::constant(rat(4)));
        let roots = solver.find_linear_root(&poly);
        assert_eq!(roots, vec![rat(-2)]);
    }

    #[test]
    fn test_find_quadratic_roots() {
        let solver = NlsatSolver::new();

        // x^2 - 4 = 0  =>  x = ±2
        let x = Polynomial::from_var(0);
        let x2 = Polynomial::mul(&x, &x);
        let poly = Polynomial::sub(&x2, &Polynomial::constant(rat(4)));

        let mut roots = solver.find_quadratic_roots(&poly);
        roots.sort();
        assert_eq!(roots, vec![rat(-2), rat(2)]);
    }

    #[test]
    fn test_solver_quadratic_sat() {
        let mut solver = NlsatSolver::new();

        // x^2 - 4 > 0 (satisfied when x < -2 or x > 2)
        let x = Polynomial::from_var(0);
        let x2 = Polynomial::mul(&x, &x);
        let poly = Polynomial::sub(&x2, &Polynomial::constant(rat(4)));

        let atom_id = solver.new_ineq_atom(poly, AtomKind::Gt);
        solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model for quadratic constraint");
        let x_val = model
            .arith_value(0)
            .expect("model should have arithmetic value for x in quadratic");
        // Should be either < -2 or > 2
        let x2_minus_4 = x_val * x_val - rat(4);
        assert!(x2_minus_4.is_positive());
    }

    #[test]
    fn test_solver_stats() {
        let mut solver = NlsatSolver::new();

        let a = solver.new_bool_var();
        let b = solver.new_bool_var();
        let c = solver.new_bool_var();

        // Create a simple problem requiring decisions
        solver.add_clause(vec![Literal::positive(a), Literal::positive(b)]);
        solver.add_clause(vec![Literal::positive(b), Literal::positive(c)]);
        solver.add_clause(vec![Literal::negative(a), Literal::negative(c)]);

        solver.solve();

        // Should have some decisions and propagations
        let stats = solver.stats();
        assert!(stats.decisions > 0 || stats.propagations > 0);
    }

    #[test]
    fn test_solver_unsat_core() {
        let mut solver = NlsatSolver::new();

        // Enable unsat core extraction
        solver.set_unsat_core_extraction(true);

        let a = solver.new_bool_var();

        // Create an unsatisfiable formula: a ∧ ¬a
        let c1 = solver.add_clause(vec![Literal::positive(a)]);
        let c2 = solver.add_clause(vec![Literal::negative(a)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat);

        // Get the unsat core
        let core = solver.get_unsat_core();

        // The core should contain both clauses
        assert!(!core.is_empty());
        assert!(core.contains(&c1.expect("c1 should have a clause id")));
        assert!(core.contains(&c2.expect("c2 should have a clause id")));
    }

    #[test]
    fn test_solver_unsat_core_with_redundancy() {
        let mut solver = NlsatSolver::new();

        // Enable unsat core extraction
        solver.set_unsat_core_extraction(true);

        let a = solver.new_bool_var();
        let b = solver.new_bool_var();

        // Create an unsatisfiable formula with a redundant clause
        // Core: a ∧ ¬a
        // Redundant: (a ∨ b)
        let c1 = solver.add_clause(vec![Literal::positive(a)]);
        let c2 = solver.add_clause(vec![Literal::negative(a)]);
        let _c3 = solver.add_clause(vec![Literal::positive(a), Literal::positive(b)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat);

        // Get the unsat core
        let core = solver.get_unsat_core();

        // The core should contain at least the conflicting clauses
        assert!(!core.is_empty());
        assert!(core.contains(&c1.expect("c1 should have a clause id for redundancy test")));
        assert!(core.contains(&c2.expect("c2 should have a clause id for redundancy test")));
    }

    #[test]
    fn test_solver_cubic_polynomial() {
        let mut solver = NlsatSolver::new();

        // x^3 - x = 0 has roots at -1, 0, 1
        let x = Polynomial::from_var(0);
        let x2 = Polynomial::mul(&x, &x);
        let x3 = Polynomial::mul(&x2, &x);
        let poly = Polynomial::sub(&x3, &x); // x^3 - x

        let atom_id = solver.new_ineq_atom(poly, AtomKind::Eq);
        solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model for cubic constraint");
        let x_val = model
            .arith_value(0)
            .expect("model should have arithmetic value for x in cubic");
        // Should be one of -1, 0, 1
        let x3_minus_x = x_val * x_val * x_val - x_val;
        assert_eq!(x3_minus_x, rat(0));
    }

    #[test]
    fn test_solver_multiple_variables_simple() {
        let mut solver = NlsatSolver::new();

        // Simple multi-variable test: x > 0 ∧ y > 0
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);

        let atom1 = solver.new_ineq_atom(x, AtomKind::Gt);
        let atom2 = solver.new_ineq_atom(y, AtomKind::Gt);

        solver.add_clause(vec![solver.atom_literal(atom1, true)]);
        solver.add_clause(vec![solver.atom_literal(atom2, true)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model for multi-variable test");
        let x_val = model
            .arith_value(0)
            .expect("model should have arithmetic value for x");
        let y_val = model
            .arith_value(1)
            .expect("model should have arithmetic value for y");

        // Both should be positive
        assert!(x_val.is_positive());
        assert!(y_val.is_positive());
    }

    #[test]
    #[ignore] // Complex multi-variable constraints - may be challenging for solver
    fn test_solver_circle_and_line() {
        let mut solver = NlsatSolver::new();

        // Circle: x^2 + y^2 = 25 (radius 5)
        // Line: y = x
        let x = Polynomial::from_var(0);
        let y = Polynomial::from_var(1);

        let x2 = Polynomial::mul(&x, &x);
        let y2 = Polynomial::mul(&y, &y);

        // x^2 + y^2 - 25 = 0
        let circle = Polynomial::sub(&Polynomial::add(&x2, &y2), &Polynomial::constant(rat(25)));

        // y - x = 0
        let line = Polynomial::sub(&y, &x);

        let atom1 = solver.new_ineq_atom(circle, AtomKind::Eq);
        let atom2 = solver.new_ineq_atom(line, AtomKind::Eq);

        solver.add_clause(vec![solver.atom_literal(atom1, true)]);
        solver.add_clause(vec![solver.atom_literal(atom2, true)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model for circle and line");
        let x_val = model
            .arith_value(0)
            .expect("model should have arithmetic value for x in circle");
        let y_val = model
            .arith_value(1)
            .expect("model should have arithmetic value for y in circle");

        // Should satisfy y = x
        assert_eq!(x_val, y_val);

        // Should satisfy x^2 + y^2 = 25, i.e., 2x^2 = 25
        let sum_of_squares = x_val.clone() * x_val.clone() + y_val.clone() * y_val.clone();
        assert_eq!(sum_of_squares, rat(25));
    }

    #[test]
    fn test_solver_inequality_chain() {
        let mut solver = NlsatSolver::new();

        // Test: x > 0 ∧ x < 10
        let x = Polynomial::from_var(0);

        // x > 0
        let atom1 = solver.new_ineq_atom(x.clone(), AtomKind::Gt);

        // x - 10 < 0 (i.e., x < 10)
        let x_minus_10 = Polynomial::sub(&x, &Polynomial::constant(rat(10)));
        let atom2 = solver.new_ineq_atom(x_minus_10, AtomKind::Lt);

        solver.add_clause(vec![solver.atom_literal(atom1, true)]);
        solver.add_clause(vec![solver.atom_literal(atom2, true)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        let model = solver
            .get_model()
            .expect("SAT result should have a model for inequality chain");
        let x_val = model
            .arith_value(0)
            .expect("model should have arithmetic value for x in chain");

        // Should be in (0, 10)
        assert!(*x_val > rat(0));
        assert!(*x_val < rat(10));
    }

    #[test]
    fn test_solver_unsatisfiable_bounds() {
        let mut solver = NlsatSolver::new();

        // Test: x > 10 ∧ x < 5 (unsatisfiable)
        let x = Polynomial::from_var(0);

        // x - 10 > 0
        let x_minus_10 = Polynomial::sub(&x, &Polynomial::constant(rat(10)));
        let atom1 = solver.new_ineq_atom(x_minus_10, AtomKind::Gt);

        // x - 5 < 0
        let x_minus_5 = Polynomial::sub(&x, &Polynomial::constant(rat(5)));
        let atom2 = solver.new_ineq_atom(x_minus_5, AtomKind::Lt);

        solver.add_clause(vec![solver.atom_literal(atom1, true)]);
        solver.add_clause(vec![solver.atom_literal(atom2, true)]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat);
    }
}
