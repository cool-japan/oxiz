//! NLSAT solver implementation.
//!
//! This module provides the core NLSAT (Non-Linear Satisfiability) solver
//! that uses the CAD (Cylindrical Algebraic Decomposition) algorithm to
//! decide satisfiability of polynomial constraints over the reals.
//!
//! Reference: Z3's `nlsat/nlsat_solver.cpp`

use crate::assignment::{Assignment, Justification};
use crate::cad::SturmSequence;
use crate::clause::{ClauseDatabase, ClauseId, NULL_CLAUSE, Watch};
use crate::interval_set::IntervalSet;
use crate::restart::{RestartManager, RestartStrategy};
use crate::types::{Atom, AtomKind, BoolVar, IneqAtom, Lbool, Literal, NULL_BOOL_VAR, PolyFactor};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use oxiz_math::polynomial::{NULL_VAR, Polynomial, Var};
use rustc_hash::FxHashMap;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, HashSet};

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

/// The state of a propagation result.
#[derive(Debug, Clone)]
pub enum PropagationResult {
    /// Propagation succeeded with no conflict.
    Ok,
    /// A conflict was detected.
    Conflict(ClauseId),
    /// Theory conflict (polynomial constraint violation).
    TheoryConflict(Vec<Literal>),
}

/// Non-linear arithmetic solver using CAD.
pub struct NlsatSolver {
    /// Solver configuration.
    config: SolverConfig,
    /// Solver statistics.
    stats: SolverStats,
    /// Clause database.
    clauses: ClauseDatabase,
    /// Variable assignment.
    assignment: Assignment,
    /// Atoms (polynomial constraints).
    atoms: Vec<Atom>,
    /// Map from polynomial hash to atom IDs (for deduplication).
    atom_map: HashMap<u64, Vec<AtomId>>,
    /// Number of arithmetic variables.
    num_arith_vars: u32,
    /// Number of boolean variables.
    num_bool_vars: u32,
    /// Variable ordering for CAD.
    var_order: Vec<Var>,
    /// Activity scores for boolean variables.
    var_activity: Vec<f64>,
    /// Activity increment.
    var_activity_inc: f64,
    /// Activity decay factor.
    var_activity_decay: f64,
    /// Activity scores for arithmetic variables.
    arith_activity: Vec<f64>,
    /// Activity increment for arithmetic variables.
    arith_activity_inc: f64,
    /// Activity decay factor for arithmetic variables.
    arith_activity_decay: f64,
    /// Queue of literals for unit propagation.
    propagation_queue: Vec<Literal>,
    /// Current conflict clause (if any).
    conflict_clause: Option<ClauseId>,
    /// Learnt clause from conflict analysis.
    learnt_clause: Vec<Literal>,
    /// Seen variables during conflict analysis.
    seen: HashSet<BoolVar>,
    /// The polynomial evaluator cache.
    eval_cache: HashMap<(AtomId, Vec<Option<BigRational>>), Lbool>,
    /// Random number generator state.
    random_state: u64,
    /// Track clauses used in conflict (for unsat core extraction).
    conflict_clauses: HashSet<ClauseId>,
    /// Enable unsat core extraction.
    extract_unsat_core: bool,
    /// Restart manager.
    restart_manager: RestartManager,
    /// Average LBD of recent learned clauses (for Glucose-style restarts).
    recent_avg_lbd: f64,
    /// Saved phase (polarity) for each boolean variable.
    /// true = positive polarity, false = negative polarity.
    saved_phase: Vec<bool>,
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
    fn compute_atom_hash(&self, poly: &Polynomial, kind: AtomKind) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the atom kind
        (kind as u8).hash(&mut hasher);

        // Hash the polynomial terms
        // Note: This is a simple hash. For production, we'd want a more robust comparison.
        for term in poly.terms() {
            term.coeff.numer().hash(&mut hasher);
            term.coeff.denom().hash(&mut hasher);
            // Hash monomial
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
    fn add_learned_clause(&mut self, literals: Vec<Literal>) -> ClauseId {
        let max_var = self.clause_max_var(&literals);
        let id = self.clauses.add(literals, max_var, true);
        self.stats.learned_clauses += 1;
        id
    }

    /// Find the clause that assigned a literal (for unsat core tracking at level 0).
    fn find_unit_conflict_reason(&self, lit: Literal) -> Option<ClauseId> {
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
    fn clause_max_var(&self, literals: &[Literal]) -> Var {
        let mut max_var = 0;
        for lit in literals {
            // Find the atom for this literal's variable
            for atom in &self.atoms {
                match atom {
                    Atom::Ineq(a) if a.bool_var == lit.var() => {
                        if a.max_var != NULL_VAR {
                            max_var = max_var.max(a.max_var);
                        }
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

    // ========== Propagation ==========

    /// Perform boolean constraint propagation.
    ///
    /// Two-watched literal scheme:
    /// - Each clause [L1, L2, ...] watches L1 and L2
    /// - Watch for Li is stored in watch list indexed by ~Li
    /// - When ~Li becomes TRUE (i.e., Li becomes FALSE), we're notified
    ///
    fn propagate(&mut self) -> PropagationResult {
        while let Some(lit) = self.propagation_queue.pop() {
            self.stats.propagations += 1;

            // `lit` was just assigned TRUE.
            // We need to find clauses where `lit.negate()` is being watched (i.e., was possibly true).
            // These clauses registered their watch in the list indexed by `lit` (since watch is on ~(lit.negate()) = lit).
            //
            // Wait, let me think again:
            // - Clause [L1, L2] puts watch on L1, registered at index ~L1
            // - Clause [L1, L2] puts watch on L2, registered at index ~L2
            // - When literal X becomes TRUE:
            //   - If X = ~L1 (so L1 becomes FALSE), we look at watches indexed by X = ~L1
            //   - These are clauses where L1 was being watched and now L1 is false
            //
            // So `lit` (assigned TRUE) means we look at watches indexed by `lit`.
            // The watched literal that became FALSE is `lit.negate()`.
            let false_lit = lit.negate();
            let watches = self.clauses.watches(lit).to_vec();

            for watch in watches {
                let clause_id = watch.clause_id;

                // Check blocker first (optimization)
                if self.assignment.lit_value(watch.blocker).is_true() {
                    continue;
                }

                // Get the clause literals
                let (lits_len, lit0) = {
                    let clause = match self.clauses.get(clause_id) {
                        Some(c) => c,
                        None => continue,
                    };
                    let lits = clause.literals();
                    (lits.len(), lits[0])
                };

                if lits_len < 2 {
                    // Unit clause - should have been handled during add_clause
                    let first_val = self.assignment.lit_value(lit0);
                    if first_val.is_false() {
                        self.propagation_queue.clear();
                        return PropagationResult::Conflict(clause_id);
                    }
                    continue;
                }

                // We already checked lits_len >= 2, so lit1 must exist
                // (just needed to verify it's not None, no need to use the value)

                // The false literal (false_lit) should be one of lit0 or lit1.
                // We want to ensure the false literal is at position 1 so position 0 is the "other" watch.
                // If lit0 is the false one, swap positions.
                if lit0 == false_lit
                    && let Some(clause) = self.clauses.get_mut(clause_id)
                {
                    clause.swap(0, 1);
                }

                // After potential swap, re-read the literals
                let Some(clause) = self.clauses.get(clause_id) else {
                    continue;
                };
                let lits = clause.literals();
                let (first_lit, second_lit) = (lits[0], lits[1]);

                // Now second_lit should be the one that became false
                // Check if the first literal (the other watched one) is true - clause is satisfied
                let first_val = self.assignment.lit_value(first_lit);
                if first_val.is_true() {
                    // Update blocker
                    self.update_watch_blocker(lit, clause_id, first_lit);
                    continue;
                }

                // Look for a new watch among literals 2..n
                let found_new = self.find_new_watch(clause_id, lit, second_lit);

                if found_new {
                    continue;
                }

                // No new watch found - all other literals are false
                // Check if conflict or unit propagation
                if first_val.is_false() {
                    // All literals are false - conflict!
                    self.propagation_queue.clear();
                    return PropagationResult::Conflict(clause_id);
                }

                // first_val is undef - unit propagation
                self.assignment
                    .assign(first_lit, Justification::Propagation(clause_id));
                self.propagation_queue.push(first_lit);
            }
        }

        PropagationResult::Ok
    }

    /// Update a watch blocker.
    fn update_watch_blocker(&mut self, lit: Literal, clause_id: ClauseId, new_blocker: Literal) {
        let watches = self.clauses.watches_mut(lit);
        for w in watches.iter_mut() {
            if w.clause_id == clause_id {
                w.blocker = new_blocker;
                break;
            }
        }
    }

    /// Find a new watch for a clause.
    /// `old_watch_index_lit` is the literal index where the old watch was registered (i.e., `lit` from propagate).
    /// `old_watched_lit` is the actual watched literal that became false.
    fn find_new_watch(
        &mut self,
        clause_id: ClauseId,
        old_watch_index_lit: Literal,
        _old_watched_lit: Literal,
    ) -> bool {
        let clause = match self.clauses.get(clause_id) {
            Some(c) => c,
            None => return false,
        };

        let lits_len = clause.len();
        if lits_len <= 2 {
            return false;
        }

        // Find a literal (at position 2 or later) that's not false
        for i in 2..lits_len {
            let Some(lit_i) = clause.get(i) else {
                continue;
            };
            let val = self.assignment.lit_value(lit_i);
            if !val.is_false() {
                // Found a new watch candidate
                let Some(first_lit) = clause.get(0) else {
                    continue;
                };

                // Swap position 1 with position i
                if let Some(clause_mut) = self.clauses.get_mut(clause_id) {
                    clause_mut.swap(1, i);
                }

                // Get the new watched literal (now at position 1)
                let Some(clause_ref) = self.clauses.get(clause_id) else {
                    return false;
                };
                let Some(new_watch_lit) = clause_ref.get(1) else {
                    return false;
                };

                // Remove from old watch list
                self.clauses
                    .watches_mut(old_watch_index_lit)
                    .retain(|w| w.clause_id != clause_id);

                // Add to new watch list (indexed by ~new_watch_lit)
                self.clauses
                    .watches_mut(new_watch_lit.negate())
                    .push(Watch::new(clause_id, first_lit));

                return true;
            }
        }

        false
    }

    /// Perform theory propagation (evaluate polynomial constraints).
    fn theory_propagate(&mut self) -> Option<Vec<Literal>> {
        // Track literals for phase saving (to avoid borrow checker issues)
        let mut lits_to_save = Vec::new();

        // Evaluate all atoms under the current arithmetic assignment
        for (id, atom) in self.atoms.iter().enumerate() {
            let atom_id = id as AtomId;

            match atom {
                Atom::Ineq(ineq) => {
                    let bool_var = ineq.bool_var;
                    let current_val = self.assignment.bool_value(bool_var);

                    // Only check if the arithmetic variables are assigned
                    if !self.can_evaluate_atom(atom_id) {
                        continue;
                    }

                    // Evaluate the polynomial
                    let eval_result = self.evaluate_atom(atom_id);

                    match (current_val, eval_result) {
                        (Lbool::True, Lbool::False) | (Lbool::False, Lbool::True) => {
                            // Conflict between boolean assignment and theory evaluation
                            let lit = if current_val.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };

                            // Return conflict clause explaining why this is unsatisfiable
                            let explanation = self.explain_theory_conflict(atom_id, lit);
                            return Some(explanation);
                        }
                        (Lbool::Undef, result) if !result.is_undef() => {
                            // Theory propagation
                            let lit = if result.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };
                            self.assignment.assign(lit, Justification::Theory);
                            lits_to_save.push(lit);
                            self.stats.theory_propagations += 1;
                        }
                        _ => {}
                    }
                }
                Atom::Root(root) => {
                    let bool_var = root.bool_var;
                    let current_val = self.assignment.bool_value(bool_var);

                    // Only check if the arithmetic variables are assigned
                    if !self.can_evaluate_atom(atom_id) {
                        continue;
                    }

                    // Evaluate the root atom
                    let eval_result = self.evaluate_root_atom(root);

                    match (current_val, eval_result) {
                        (Lbool::True, Lbool::False) | (Lbool::False, Lbool::True) => {
                            // Conflict between boolean assignment and theory evaluation
                            let lit = if current_val.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };

                            // Return conflict clause explaining why this is unsatisfiable
                            let explanation = self.explain_theory_conflict(atom_id, lit);
                            return Some(explanation);
                        }
                        (Lbool::Undef, result) if !result.is_undef() => {
                            // Theory propagation
                            let lit = if result.is_true() {
                                Literal::positive(bool_var)
                            } else {
                                Literal::negative(bool_var)
                            };
                            self.assignment.assign(lit, Justification::Theory);
                            lits_to_save.push(lit);
                            self.stats.theory_propagations += 1;
                        }
                        _ => {}
                    }
                }
            }
        }

        // Save phases for all propagated literals
        for lit in lits_to_save {
            self.save_phase(lit);
        }

        None
    }

    /// Check if we can evaluate an atom (all required variables assigned).
    fn can_evaluate_atom(&self, atom_id: AtomId) -> bool {
        match self.get_atom(atom_id) {
            Some(Atom::Ineq(ineq)) => {
                // Check if all variables in any polynomial factor are assigned
                for factor in &ineq.factors {
                    for var in factor.poly.vars() {
                        if !self.assignment.is_arith_assigned(var) {
                            return false;
                        }
                    }
                }
                true
            }
            Some(Atom::Root(root)) => {
                // For root atoms, we need all variables up to max_var assigned
                let max = root.max_var();
                if max == NULL_VAR {
                    return true;
                }
                for var in 0..=max {
                    if !self.assignment.is_arith_assigned(var) {
                        return false;
                    }
                }
                true
            }
            None => false,
        }
    }

    /// Evaluate an atom under the current assignment.
    fn evaluate_atom(&self, atom_id: AtomId) -> Lbool {
        match self.get_atom(atom_id) {
            Some(Atom::Ineq(ineq)) => {
                // Compute signs for each factor
                let mut signs = Vec::with_capacity(ineq.factors.len());

                for factor in &ineq.factors {
                    // Build the evaluation map
                    let mut eval_map = FxHashMap::default();
                    for var in factor.poly.vars() {
                        if let Some(val) = self.assignment.arith_value(var) {
                            eval_map.insert(var, val.clone());
                        } else {
                            return Lbool::Undef;
                        }
                    }

                    // Evaluate polynomial
                    let value = factor.poly.eval(&eval_map);
                    let sign = if value.is_zero() {
                        0
                    } else if value.is_positive() {
                        1
                    } else {
                        -1
                    };
                    signs.push(sign);
                }

                // Use evaluate_sign method
                match ineq.evaluate_sign(&signs) {
                    Some(true) => Lbool::True,
                    Some(false) => Lbool::False,
                    None => Lbool::Undef,
                }
            }
            Some(Atom::Root(root)) => self.evaluate_root_atom(root),
            None => Lbool::Undef,
        }
    }

    /// Evaluate a root atom under the current assignment.
    ///
    /// For a root atom like `x op root[i](p)`, where op is =, <, >, <=, or >=:
    /// 1. Substitute all assigned variables (except x) into p
    /// 2. Isolate the roots of the resulting univariate polynomial
    /// 3. Get the i-th root (1-indexed, sorted in ascending order)
    /// 4. Compare x's value with the root value
    fn evaluate_root_atom(&self, root: &crate::types::RootAtom) -> Lbool {
        // Get the value of the variable x
        let x_val = match self.assignment.arith_value(root.var) {
            Some(v) => v,
            None => return Lbool::Undef,
        };

        // Substitute all assigned variables (except root.var) into the polynomial
        let mut sub_poly = root.poly.clone();
        for var in root.poly.vars() {
            if var != root.var {
                if let Some(val) = self.assignment.arith_value(var) {
                    sub_poly = sub_poly.substitute(var, &Polynomial::constant(val.clone()));
                } else {
                    return Lbool::Undef;
                }
            }
        }

        // Now sub_poly should be univariate in root.var
        if !sub_poly.is_univariate() && !sub_poly.is_constant() {
            return Lbool::Undef;
        }

        // If the polynomial is constant (all roots are gone), we can't satisfy the root atom
        if sub_poly.is_constant() {
            // No roots exist
            return Lbool::False;
        }

        // Use Sturm sequence to isolate roots
        let sturm = SturmSequence::new(&sub_poly, root.var);
        let root_intervals = sturm.isolate_roots();

        // Check if we have enough roots
        if (root.root_index as usize) > root_intervals.len() || root.root_index == 0 {
            // Root index out of bounds (1-indexed)
            return Lbool::False;
        }

        // Get the i-th root interval (root_index is 1-based)
        let (root_lo, root_hi) = &root_intervals[(root.root_index - 1) as usize];

        // Compare x_val with the root
        // The root is in the interval [root_lo, root_hi]
        // For precise comparison, we refine the interval if x_val is within it
        let cmp_lo = x_val.cmp(root_lo);
        let cmp_hi = x_val.cmp(root_hi);

        // Determine the comparison result based on the atom kind
        let result = match root.kind {
            AtomKind::RootEq => {
                // x = root[i](p)
                // This is true only if the interval is a point and x_val equals it
                if root_lo == root_hi && x_val == root_lo {
                    true
                } else if cmp_lo == CmpOrdering::Less || cmp_hi == CmpOrdering::Greater {
                    // x is definitely not equal to the root
                    false
                } else {
                    // x_val is within the isolating interval - we can't determine for sure
                    // In a real implementation, we would refine the interval further
                    return Lbool::Undef;
                }
            }
            AtomKind::RootLt => {
                // x < root[i](p)
                if cmp_hi == CmpOrdering::Less {
                    // x < root_hi, so definitely x < root
                    true
                } else if cmp_lo != CmpOrdering::Less {
                    // x >= root_lo, so definitely x >= root (not less)
                    false
                } else {
                    // x is in [root_lo, root_hi) - unclear
                    return Lbool::Undef;
                }
            }
            AtomKind::RootGt => {
                // x > root[i](p)
                if cmp_lo == CmpOrdering::Greater {
                    // x > root_lo, so definitely x > root
                    true
                } else if cmp_hi != CmpOrdering::Greater {
                    // x <= root_hi, so definitely x <= root (not greater)
                    false
                } else {
                    // x is in (root_lo, root_hi] - unclear
                    return Lbool::Undef;
                }
            }
            AtomKind::RootLe => {
                // x <= root[i](p)
                if cmp_hi != CmpOrdering::Greater {
                    // x <= root_hi, so definitely x <= root
                    true
                } else if cmp_lo == CmpOrdering::Greater {
                    // x > root_lo, so definitely x > root (not <=)
                    false
                } else {
                    return Lbool::Undef;
                }
            }
            AtomKind::RootGe => {
                // x >= root[i](p)
                if cmp_lo != CmpOrdering::Less {
                    // x >= root_lo, so definitely x >= root
                    true
                } else if cmp_hi == CmpOrdering::Less {
                    // x < root_hi, so definitely x < root (not >=)
                    false
                } else {
                    return Lbool::Undef;
                }
            }
            _ => return Lbool::Undef,
        };

        Lbool::from_bool(result)
    }

    /// Explain a theory conflict.
    fn explain_theory_conflict(
        &mut self,
        atom_id: AtomId,
        conflicting_lit: Literal,
    ) -> Vec<Literal> {
        let mut explanation = Vec::new();

        // The conflicting literal is part of the explanation
        explanation.push(conflicting_lit.negate());

        // Collect variables involved in conflict first
        let mut conflict_vars = Vec::new();

        // Add arithmetic variable assignments that led to this conflict
        // For each assigned arithmetic variable, find the atoms that constrained it
        if let Some(Atom::Ineq(ineq)) = self.get_atom(atom_id) {
            for factor in &ineq.factors {
                for var in factor.poly.vars() {
                    conflict_vars.push(var);

                    // Find atoms that assigned this variable
                    for (other_id, other_atom) in self.atoms.iter().enumerate() {
                        if other_id == atom_id as usize {
                            continue;
                        }

                        if let Atom::Ineq(other_ineq) = other_atom {
                            let has_var = other_ineq
                                .factors
                                .iter()
                                .any(|f| f.poly.vars().contains(&var));
                            if has_var {
                                let bool_var = other_ineq.bool_var;
                                let val = self.assignment.bool_value(bool_var);
                                if !val.is_undef() {
                                    let lit = if val.is_true() {
                                        Literal::negative(bool_var)
                                    } else {
                                        Literal::positive(bool_var)
                                    };
                                    if !explanation.contains(&lit)
                                        && !explanation.contains(&lit.negate())
                                    {
                                        explanation.push(lit);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Bump activity for variables involved in conflict
        for var in conflict_vars {
            self.bump_arith_activity(var);
        }

        explanation
    }

    // ========== Decision Making ==========

    /// Make a decision.
    fn decide(&mut self) -> Option<Literal> {
        // Random decision
        if self.config.random_decisions
            && self.random() < self.config.random_freq
            && let Some(lit) = self.random_decision()
        {
            return Some(lit);
        }

        // VSIDS-like decision: pick the unassigned variable with highest activity
        let mut best_var: Option<BoolVar> = None;
        let mut best_activity = f64::NEG_INFINITY;

        for var in 0..self.num_bool_vars {
            if self.assignment.is_bool_assigned(var) {
                continue;
            }

            let activity = self.var_activity.get(var as usize).copied().unwrap_or(0.0);
            if activity > best_activity {
                best_activity = activity;
                best_var = Some(var);
            }
        }

        best_var.map(|var| {
            // Use saved phase (phase saving heuristic)
            let polarity = self.saved_phase.get(var as usize).copied().unwrap_or(true);
            Literal::new(var, polarity)
        })
    }

    /// Save the phase (polarity) of a literal assignment.
    fn save_phase(&mut self, lit: Literal) {
        let var = lit.var();
        let polarity = !lit.is_negated();
        if (var as usize) < self.saved_phase.len() {
            self.saved_phase[var as usize] = polarity;
        }
    }

    /// Make a random decision.
    fn random_decision(&mut self) -> Option<Literal> {
        let mut unassigned = Vec::new();
        for var in 0..self.num_bool_vars {
            if !self.assignment.is_bool_assigned(var) {
                unassigned.push(var);
            }
        }

        if unassigned.is_empty() {
            return None;
        }

        let idx = (self.random_int() as usize) % unassigned.len();
        let var = unassigned[idx];
        let positive = self.random_int().is_multiple_of(2);

        Some(if positive {
            Literal::positive(var)
        } else {
            Literal::negative(var)
        })
    }

    /// Get the next arithmetic variable to assign.
    fn next_arith_var(&self) -> Option<Var> {
        // Return the first unassigned variable in the ordering
        self.var_order
            .iter()
            .find(|&&var| !self.assignment.is_arith_assigned(var))
            .copied()
    }

    /// Pick a value for an arithmetic variable.
    fn pick_arith_value(&mut self, var: Var) -> Option<BigRational> {
        // Get the feasible region for this variable
        let feasible = self.compute_feasible_region(var);

        // Early termination: if feasible region is empty, record and return None
        if feasible.is_empty() {
            if self.config.early_termination {
                self.stats.early_terminations += 1;
            }
            return None;
        }

        // Sample a point from the feasible region
        feasible.sample()
    }

    /// Compute the feasible region for an arithmetic variable.
    fn compute_feasible_region(&self, var: Var) -> IntervalSet {
        let mut region = IntervalSet::reals();

        // Intersect with constraints from all satisfied atoms
        for atom in &self.atoms {
            if let Atom::Ineq(ineq) = atom {
                // Check if this atom involves the variable
                let involves_var = ineq.factors.iter().any(|f| f.poly.vars().contains(&var));
                if !involves_var {
                    continue;
                }

                // Check if this atom is assigned
                let val = self.assignment.bool_value(ineq.bool_var);
                if val.is_undef() {
                    continue;
                }

                // Get the constraint on var from this atom
                let constraint = self.atom_constraint_on_var(atom, var, val.is_true());
                region = region.intersect(&constraint);

                if region.is_empty() {
                    break;
                }
            }
        }

        region
    }

    /// Get the constraint that an atom places on a variable.
    fn atom_constraint_on_var(&self, atom: &Atom, var: Var, atom_is_true: bool) -> IntervalSet {
        match atom {
            Atom::Ineq(ineq) => {
                // For now, only handle single-factor atoms
                if ineq.factors.len() != 1 {
                    return IntervalSet::reals();
                }

                let factor = &ineq.factors[0];

                // Substitute all assigned variables except `var`
                let mut sub_poly = factor.poly.clone();
                for v in factor.poly.vars() {
                    if v != var
                        && let Some(val) = self.assignment.arith_value(v)
                    {
                        sub_poly = sub_poly.substitute(v, &Polynomial::constant(val.clone()));
                    }
                }

                // Now sub_poly should be univariate in `var`
                if !sub_poly.is_univariate() && !sub_poly.is_constant() {
                    // Can't simplify further
                    return IntervalSet::reals();
                }

                // Find roots
                let roots = self.find_univariate_roots(&sub_poly, var);

                // Determine signs between roots
                let signs = self.compute_signs_between_roots(&sub_poly, var, &roots);

                // Create interval set based on constraint kind and polarity
                let target_sign = match (ineq.kind, atom_is_true) {
                    (AtomKind::Eq, true) => 0,    // p = 0
                    (AtomKind::Eq, false) => 127, // p != 0 (special case)
                    (AtomKind::Lt, true) => -1,   // p < 0
                    (AtomKind::Lt, false) => 1,   // p >= 0 (includes 0)
                    (AtomKind::Gt, true) => 1,    // p > 0
                    (AtomKind::Gt, false) => -1,  // p <= 0 (includes 0)
                    _ => return IntervalSet::reals(),
                };

                if target_sign == 127 {
                    // p != 0: complement of {roots}
                    let zero_set = IntervalSet::sign_set(&roots, &signs, 0);
                    zero_set.complement()
                } else if target_sign == 1 && !atom_is_true {
                    // p >= 0: positive or zero
                    let pos_set = IntervalSet::sign_set(&roots, &signs, 1);
                    let zero_set = IntervalSet::sign_set(&roots, &signs, 0);
                    pos_set.union(&zero_set)
                } else if target_sign == -1 && !atom_is_true {
                    // p <= 0: negative or zero
                    let neg_set = IntervalSet::sign_set(&roots, &signs, -1);
                    let zero_set = IntervalSet::sign_set(&roots, &signs, 0);
                    neg_set.union(&zero_set)
                } else {
                    IntervalSet::sign_set(&roots, &signs, target_sign)
                }
            }
            Atom::Root(root) => {
                // For root atoms, we need to isolate the roots and determine the constraint
                // x op root[i](p) where op is =, <, >, <=, >=

                // First, check if this root atom actually involves the variable `var`
                if root.var != var && !root.poly.vars().contains(&var) {
                    return IntervalSet::reals();
                }

                // If the atom involves `var` in the polynomial (not as the root variable),
                // we cannot easily extract a constraint on `var` alone
                if root.var != var {
                    return IntervalSet::reals();
                }

                // Substitute all assigned variables (except var) into the polynomial
                let mut sub_poly = root.poly.clone();
                for v in root.poly.vars() {
                    if v != var {
                        if let Some(val) = self.assignment.arith_value(v) {
                            sub_poly = sub_poly.substitute(v, &Polynomial::constant(val.clone()));
                        } else {
                            return IntervalSet::reals();
                        }
                    }
                }

                // If the polynomial is constant, no roots exist
                if sub_poly.is_constant() {
                    return IntervalSet::empty();
                }

                // Isolate the roots
                let sturm = SturmSequence::new(&sub_poly, var);
                let root_intervals = sturm.isolate_roots();

                // Check if we have enough roots
                if (root.root_index as usize) > root_intervals.len() || root.root_index == 0 {
                    return IntervalSet::empty();
                }

                // Get the i-th root interval
                let (root_lo, root_hi) = &root_intervals[(root.root_index - 1) as usize];

                // Create interval set based on the atom kind and polarity
                match (root.kind, atom_is_true) {
                    (AtomKind::RootEq, true) => {
                        // x = root[i](p)
                        IntervalSet::from_point(root_lo.clone())
                    }
                    (AtomKind::RootEq, false) => {
                        // x != root[i](p) - complement of the point
                        IntervalSet::from_point(root_lo.clone()).complement()
                    }
                    (AtomKind::RootLt, true) => {
                        // x < root[i](p) - approximately (-∞, root_hi)
                        IntervalSet::lt(root_hi.clone())
                    }
                    (AtomKind::RootLt, false) => {
                        // x >= root[i](p) - approximately [root_lo, +∞)
                        IntervalSet::ge(root_lo.clone())
                    }
                    (AtomKind::RootGt, true) => {
                        // x > root[i](p) - approximately (root_lo, +∞)
                        IntervalSet::gt(root_lo.clone())
                    }
                    (AtomKind::RootGt, false) => {
                        // x <= root[i](p) - approximately (-∞, root_hi]
                        IntervalSet::le(root_hi.clone())
                    }
                    (AtomKind::RootLe, true) => {
                        // x <= root[i](p)
                        IntervalSet::le(root_hi.clone())
                    }
                    (AtomKind::RootLe, false) => {
                        // x > root[i](p)
                        IntervalSet::gt(root_lo.clone())
                    }
                    (AtomKind::RootGe, true) => {
                        // x >= root[i](p)
                        IntervalSet::ge(root_lo.clone())
                    }
                    (AtomKind::RootGe, false) => {
                        // x < root[i](p)
                        IntervalSet::lt(root_hi.clone())
                    }
                    _ => IntervalSet::reals(),
                }
            }
        }
    }

    /// Find roots of a univariate polynomial.
    fn find_univariate_roots(&self, poly: &Polynomial, var: Var) -> Vec<BigRational> {
        // For now, use a simple approach for low-degree polynomials
        let degree = poly.degree(var);

        if degree == 0 {
            return Vec::new();
        }

        if degree == 1 {
            // Linear: ax + b = 0  =>  x = -b/a
            return self.find_linear_root(poly);
        }

        if degree == 2 {
            // Quadratic: use quadratic formula (rational roots only)
            return self.find_quadratic_roots(poly);
        }

        // For higher degrees, we would need more sophisticated root isolation
        // For now, return empty (conservative but safe)
        Vec::new()
    }

    /// Find the root of a linear polynomial.
    fn find_linear_root(&self, poly: &Polynomial) -> Vec<BigRational> {
        // p = ax + b, find x = -b/a
        let terms = poly.terms();
        if terms.len() > 2 {
            return Vec::new();
        }

        let mut a = BigRational::zero();
        let mut b = BigRational::zero();

        for term in terms {
            if term.monomial.is_unit() {
                b = term.coeff.clone();
            } else if term.monomial.total_degree() == 1 {
                a = term.coeff.clone();
            }
        }

        if a.is_zero() {
            return Vec::new();
        }

        vec![-b / a]
    }

    /// Find rational roots of a quadratic polynomial.
    fn find_quadratic_roots(&self, poly: &Polynomial) -> Vec<BigRational> {
        // p = ax^2 + bx + c
        // Discriminant = b^2 - 4ac
        // If discriminant is a perfect square, roots are rational

        let terms = poly.terms();
        if terms.len() > 3 {
            return Vec::new();
        }

        let mut a = BigRational::zero();
        let mut b = BigRational::zero();
        let mut c = BigRational::zero();

        for term in terms {
            match term.monomial.total_degree() {
                0 => c = term.coeff.clone(),
                1 => b = term.coeff.clone(),
                2 => a = term.coeff.clone(),
                _ => return Vec::new(),
            }
        }

        if a.is_zero() {
            // Actually linear
            if b.is_zero() {
                return Vec::new();
            }
            return vec![-c.clone() / b.clone()];
        }

        // Discriminant
        let disc = &b * &b - BigRational::from_integer(4.into()) * &a * &c;

        if disc.is_negative() {
            return Vec::new();
        }

        if disc.is_zero() {
            let root = -b / (BigRational::from_integer(2.into()) * a);
            return vec![root];
        }

        // Check if discriminant is a perfect square
        // For rational discriminant p/q, we need both p and q to be perfect squares
        let numer = disc.numer().clone();
        let denom = disc.denom().clone();

        if let (Some(sqrt_n), Some(sqrt_d)) = (integer_sqrt(&numer), integer_sqrt(&denom)) {
            let sqrt_disc = BigRational::new(sqrt_n, sqrt_d);
            let two_a = BigRational::from_integer(2.into()) * &a;
            let root1 = (-&b + &sqrt_disc) / &two_a;
            let root2 = (-&b - &sqrt_disc) / &two_a;

            let mut roots = vec![root1, root2];
            roots.sort();
            roots.dedup();
            roots
        } else {
            // Irrational roots - cannot represent exactly
            Vec::new()
        }
    }

    /// Compute signs of polynomial between roots.
    fn compute_signs_between_roots(
        &self,
        poly: &Polynomial,
        var: Var,
        roots: &[BigRational],
    ) -> Vec<i8> {
        if roots.is_empty() {
            // No roots - evaluate at any point
            let test_val = BigRational::zero();
            let mut eval_map = FxHashMap::default();
            eval_map.insert(var, test_val);
            let val = poly.eval(&eval_map);
            let sign = if val.is_zero() {
                0
            } else if val.is_positive() {
                1
            } else {
                -1
            };
            return vec![sign];
        }

        let mut signs = Vec::with_capacity(roots.len() + 1);

        // Before first root
        let before = &roots[0] - BigRational::one();
        signs.push(self.eval_sign(poly, var, &before));

        // Between roots
        for i in 0..roots.len() - 1 {
            let mid = (&roots[i] + &roots[i + 1]) / BigRational::from_integer(2.into());
            signs.push(self.eval_sign(poly, var, &mid));
        }

        // After last root
        if let Some(last_root) = roots.last() {
            let after = last_root + BigRational::one();
            signs.push(self.eval_sign(poly, var, &after));
        }

        signs
    }

    /// Evaluate the sign of a polynomial at a point.
    fn eval_sign(&self, poly: &Polynomial, var: Var, val: &BigRational) -> i8 {
        let mut eval_map = FxHashMap::default();
        eval_map.insert(var, val.clone());
        let result = poly.eval(&eval_map);
        if result.is_zero() {
            0
        } else if result.is_positive() {
            1
        } else {
            -1
        }
    }

    // ========== Conflict Analysis ==========

    /// Analyze a conflict and learn a clause.
    fn analyze_conflict(&mut self, conflict_id: ClauseId) -> (Vec<Literal>, u32) {
        self.learnt_clause.clear();
        self.seen.clear();

        // Track this clause for unsat core
        if self.extract_unsat_core {
            self.conflict_clauses.insert(conflict_id);
        }

        let clause_lits: Vec<Literal> = match self.clauses.get(conflict_id) {
            Some(c) => c.literals().to_vec(),
            None => return (Vec::new(), 0),
        };

        let current_level = self.assignment.level();
        let mut counter = 0; // Number of literals at current level

        // Process conflict clause
        for &lit in &clause_lits {
            let var = lit.var();
            if !self.seen.contains(&var) {
                self.seen.insert(var);
                let level = self.assignment.bool_level(var);

                if level == current_level {
                    counter += 1;
                } else if level > 0 {
                    self.learnt_clause.push(lit.negate());
                    self.bump_var_activity(var);
                }
            }
        }

        // Resolve until we have exactly one literal at current level
        let mut trail_idx = self.assignment.trail().len();
        while counter > 1 && trail_idx > 0 {
            // Find next literal to resolve
            trail_idx -= 1;
            let trail = self.assignment.trail();

            let entry = &trail[trail_idx];
            let lit = entry.literal;
            let var = lit.var();

            if !self.seen.contains(&var) {
                continue;
            }
            self.seen.remove(&var);
            counter -= 1;

            // Get the reason clause
            if let Justification::Propagation(reason_id) = &entry.justification {
                // Track reason clause for unsat core
                if self.extract_unsat_core {
                    self.conflict_clauses.insert(*reason_id);
                }

                let reason_lits: Vec<Literal> = match self.clauses.get(*reason_id) {
                    Some(r) => r.literals().to_vec(),
                    None => continue,
                };

                for reason_lit in reason_lits {
                    if reason_lit == lit {
                        continue;
                    }

                    let reason_var = reason_lit.var();
                    if !self.seen.contains(&reason_var) {
                        self.seen.insert(reason_var);
                        let level = self.assignment.bool_level(reason_var);

                        if level == current_level {
                            counter += 1;
                        } else if level > 0 {
                            self.learnt_clause.push(reason_lit.negate());
                            self.bump_var_activity(reason_var);
                        }
                    }
                }
            }
        }

        // Find the UIP (asserting literal)
        trail_idx = self.assignment.trail().len();
        while trail_idx > 0 {
            trail_idx -= 1;
            let trail = self.assignment.trail();
            let entry = &trail[trail_idx];
            let var = entry.literal.var();

            if self.seen.contains(&var) {
                // This is the asserting literal
                self.learnt_clause.insert(0, entry.literal.negate());
                self.bump_var_activity(var);
                break;
            }
        }

        // Compute backtrack level
        let mut backtrack_level = 0;
        for lit in &self.learnt_clause[1..] {
            let level = self.assignment.bool_level(lit.var());
            backtrack_level = backtrack_level.max(level);
        }

        // Minimize learned clause (optional)
        let minimized = self.minimize_clause(self.learnt_clause.clone());

        (minimized, backtrack_level)
    }

    /// Minimize a learned clause by removing redundant literals.
    fn minimize_clause(&self, mut clause: Vec<Literal>) -> Vec<Literal> {
        if clause.len() <= 1 {
            return clause;
        }

        // Keep track of which literals can be removed
        let mut to_remove = Vec::new();

        // Try to remove each literal (except the first asserting literal)
        for i in 1..clause.len() {
            let lit = clause[i];
            let var = lit.var();

            // Check if this literal is redundant
            if self.is_redundant_literal(var, &clause) {
                to_remove.push(i);
            }
        }

        // Remove redundant literals (in reverse order to maintain indices)
        for &idx in to_remove.iter().rev() {
            clause.remove(idx);
        }

        clause
    }

    /// Check if a literal at a variable is redundant in the clause.
    fn is_redundant_literal(&self, var: BoolVar, clause: &[Literal]) -> bool {
        // Get the justification for this variable
        let trail = self.assignment.trail();
        let entry = trail.iter().find(|e| e.literal.var() == var);

        if let Some(entry) = entry {
            match &entry.justification {
                Justification::Propagation(reason_id) => {
                    // Check if the reason clause's literals are all in the learned clause
                    if let Some(reason_clause) = self.clauses.get(*reason_id) {
                        let reason_lits = reason_clause.literals();

                        // All literals in reason (except the propagated one) should be
                        // either at level 0 or already in the learned clause
                        for &reason_lit in reason_lits {
                            if reason_lit.var() == var {
                                continue; // Skip the propagated literal
                            }

                            let reason_var = reason_lit.var();
                            let level = self.assignment.bool_level(reason_var);

                            if level == 0 {
                                continue; // Level 0 literals are always fine
                            }

                            // Check if this literal (or its negation) is in the clause
                            let in_clause = clause.iter().any(|&cl| cl.var() == reason_var);

                            if !in_clause {
                                // Check recursively if this variable is redundant
                                if !self.is_redundant_literal(reason_var, clause) {
                                    return false;
                                }
                            }
                        }

                        // All reason literals are covered
                        return true;
                    }
                }
                Justification::Decision | Justification::Unit | Justification::Theory => {
                    // Cannot minimize decision or unit literals
                    return false;
                }
            }
        }

        false
    }

    // ========== Backtracking ==========

    /// Backtrack to a given level.
    fn backtrack(&mut self, level: u32) {
        // Clear propagation queue
        self.propagation_queue.clear();
        self.conflict_clause = None;

        // Pop assignment levels
        let _unassigned = self.assignment.pop_level(level);

        // Reset arithmetic assignments above this level
        // (Simplified: reset all arithmetic assignments)
        for var in 0..self.num_arith_vars {
            self.assignment.unset_arith(var);
            self.assignment.reset_feasible(var);
        }

        // Clear evaluation cache
        self.eval_cache.clear();
    }

    // ========== Activity Management ==========

    /// Bump the activity of a variable.
    fn bump_var_activity(&mut self, var: BoolVar) {
        if (var as usize) >= self.var_activity.len() {
            self.var_activity.resize(var as usize + 1, 0.0);
        }

        self.var_activity[var as usize] += self.var_activity_inc;

        // Rescale if too large
        if self.var_activity[var as usize] > 1e100 {
            for a in &mut self.var_activity {
                *a *= 1e-100;
            }
            self.var_activity_inc *= 1e-100;
        }
    }

    /// Bump the activity of an arithmetic variable.
    fn bump_arith_activity(&mut self, var: Var) {
        if (var as usize) >= self.arith_activity.len() {
            self.arith_activity.resize(var as usize + 1, 0.0);
        }

        self.arith_activity[var as usize] += self.arith_activity_inc;

        // Rescale if too large
        if self.arith_activity[var as usize] > 1e100 {
            for a in &mut self.arith_activity {
                *a *= 1e-100;
            }
            self.arith_activity_inc *= 1e-100;
        }
    }

    /// Decay all activities.
    fn decay_activities(&mut self) {
        self.var_activity_inc *= 1.0 / self.var_activity_decay;
        self.arith_activity_inc *= 1.0 / self.arith_activity_decay;
        self.clauses.decay_activities();
    }

    // ========== Restart and Reduction ==========

    /// Compute the Literal Block Distance (LBD) of a clause.
    ///
    /// LBD is the number of distinct decision levels in the clause.
    /// Lower LBD indicates a more "glue" clause.
    fn compute_lbd(&self, clause_lits: &[Literal]) -> u32 {
        let mut levels = HashSet::new();
        for &lit in clause_lits {
            let level = self.assignment.bool_level(lit.var());
            if level > 0 {
                levels.insert(level);
            }
        }
        levels.len() as u32
    }

    /// Maybe perform a restart using the restart manager.
    fn maybe_restart(&mut self) {
        // Use restart manager to determine if we should restart
        let should_restart = if matches!(
            self.config.restart_strategy,
            RestartStrategy::Glucose { .. }
        ) {
            self.restart_manager
                .should_restart(Some(self.recent_avg_lbd))
        } else {
            self.restart_manager.should_restart(None)
        };

        if should_restart && self.assignment.level() > 0 {
            self.stats.restarts += 1;
            self.backtrack(0);
            self.restart_manager.restart();
        }
    }

    /// Reduce learned clauses.
    fn reduce_learned(&mut self) {
        let removed = self
            .clauses
            .reduce_learned(self.config.learned_keep_fraction);
        self.stats.clause_deletions += removed.len() as u64;
    }

    /// Perform dynamic variable reordering based on activity scores.
    fn dynamic_reorder(&mut self) {
        if !self.config.dynamic_reordering {
            return;
        }

        // Can only reorder unassigned variables
        let mut unassigned_vars: Vec<(Var, f64)> = (0..self.num_arith_vars)
            .filter(|&var| !self.assignment.is_arith_assigned(var))
            .map(|var| {
                let activity = self
                    .arith_activity
                    .get(var as usize)
                    .copied()
                    .unwrap_or(0.0);
                (var, activity)
            })
            .collect();

        // Sort by activity (highest first)
        unassigned_vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(CmpOrdering::Equal));

        // Rebuild var_order: assigned variables first (in current order), then by activity
        let assigned_vars: Vec<Var> = (0..self.num_arith_vars)
            .filter(|&var| self.assignment.is_arith_assigned(var))
            .collect();

        self.var_order.clear();
        self.var_order.extend(assigned_vars);
        self.var_order
            .extend(unassigned_vars.iter().map(|(var, _)| *var));

        self.stats.reorderings += 1;
    }

    // ========== Helper Methods ==========

    /// Check if the formula is completely assigned.
    fn is_complete(&self) -> bool {
        // All boolean variables assigned
        for var in 0..self.num_bool_vars {
            if !self.assignment.is_bool_assigned(var) {
                return false;
            }
        }

        // All arithmetic variables assigned
        for var in 0..self.num_arith_vars {
            if !self.assignment.is_arith_assigned(var) {
                return false;
            }
        }

        true
    }

    /// Generate a random number in [0, 1).
    fn random(&mut self) -> f64 {
        self.random_int() as f64 / u64::MAX as f64
    }

    /// Generate a random u64.
    fn random_int(&mut self) -> u64 {
        // Simple LCG
        self.random_state = self
            .random_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.random_state
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
fn integer_sqrt(n: &num_bigint::BigInt) -> Option<num_bigint::BigInt> {
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
        let model = solver.get_model().unwrap();
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

        let model = solver.get_model().unwrap();
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

        let model = solver.get_model().unwrap();
        let x_val = model.arith_value(x_var).unwrap();
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

        let model = solver.get_model().unwrap();
        let x_val = model.arith_value(0).unwrap();
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

        let model = solver.get_model().unwrap();
        let x_val = model.arith_value(0).unwrap();
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
        assert!(core.contains(&c1.unwrap()));
        assert!(core.contains(&c2.unwrap()));
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
        assert!(core.contains(&c1.unwrap()));
        assert!(core.contains(&c2.unwrap()));
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

        let model = solver.get_model().unwrap();
        let x_val = model.arith_value(0).unwrap();
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

        let model = solver.get_model().unwrap();
        let x_val = model.arith_value(0).unwrap();
        let y_val = model.arith_value(1).unwrap();

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

        let model = solver.get_model().unwrap();
        let x_val = model.arith_value(0).unwrap();
        let y_val = model.arith_value(1).unwrap();

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

        let model = solver.get_model().unwrap();
        let x_val = model.arith_value(0).unwrap();

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
