//! CDCL SAT Solver

mod conflict;
mod decide;
mod learn;
mod propagate;

use crate::chb::CHB;
use crate::chrono::ChronoBacktrack;
use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{LBool, Lit, Var};
use crate::lrb::LRB;
use crate::memory_opt::{MemoryAction, MemoryOptimizer};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::trail::{Reason, Trail};
use crate::vsids::VSIDS;
use crate::watched::{WatchLists, Watcher};
use smallvec::SmallVec;

/// Binary implication graph for efficient binary clause propagation
/// For each literal L, stores the list of literals that are implied when L is false
/// (i.e., for binary clause (~L v M), when L is assigned false, M must be true)
#[derive(Debug, Clone)]
pub(super) struct BinaryImplicationGraph {
    /// implications[lit] = list of (implied_lit, clause_id) pairs
    implications: Vec<Vec<(Lit, ClauseId)>>,
}

impl BinaryImplicationGraph {
    fn new(num_vars: usize) -> Self {
        Self {
            implications: vec![Vec::new(); num_vars * 2],
        }
    }

    fn resize(&mut self, num_vars: usize) {
        self.implications.resize(num_vars * 2, Vec::new());
    }

    fn add(&mut self, lit: Lit, implied: Lit, clause_id: ClauseId) {
        self.implications[lit.code() as usize].push((implied, clause_id));
    }

    fn get(&self, lit: Lit) -> &[(Lit, ClauseId)] {
        &self.implications[lit.code() as usize]
    }

    fn clear(&mut self) {
        for implications in &mut self.implications {
            implications.clear();
        }
    }
}

/// Result from a theory check
#[derive(Debug, Clone)]
pub enum TheoryCheckResult {
    /// Theory is satisfied under current assignment
    Sat,
    /// Theory detected a conflict, returns conflict clause literals
    Conflict(SmallVec<[Lit; 8]>),
    /// Theory propagated new literals (lit, reason clause)
    Propagated(Vec<(Lit, SmallVec<[Lit; 8]>)>),
}

/// Callback trait for theory solvers
/// The CDCL(T) solver implements this to receive theory callbacks
pub trait TheoryCallback {
    /// Called when a literal is assigned
    /// Returns a theory check result
    fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult;

    /// Called after propagation is complete to do a full theory check
    fn final_check(&mut self) -> TheoryCheckResult;

    /// Called when the decision level increases
    fn on_new_level(&mut self, _level: u32) {}

    /// Called when backtracking
    fn on_backtrack(&mut self, level: u32);
}

/// Result of SAT solving
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown (e.g., timeout, resource limit)
    Unknown,
}

/// Solver configuration
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// Restart interval (number of conflicts)
    pub restart_interval: u64,
    /// Restart multiplier for geometric restarts
    pub restart_multiplier: f64,
    /// Clause deletion threshold
    pub clause_deletion_threshold: usize,
    /// Variable decay factor
    pub var_decay: f64,
    /// Clause decay factor
    pub clause_decay: f64,
    /// Random polarity probability (0.0 to 1.0)
    pub random_polarity_prob: f64,
    /// Restart strategy: "luby" or "geometric"
    pub restart_strategy: RestartStrategy,
    /// Enable lazy hyper-binary resolution
    pub enable_lazy_hyper_binary: bool,
    /// Use CHB instead of VSIDS for branching
    pub use_chb_branching: bool,
    /// Use LRB (Learning Rate Branching) for branching
    pub use_lrb_branching: bool,
    /// Enable inprocessing (periodic preprocessing during search)
    pub enable_inprocessing: bool,
    /// Inprocessing interval (number of conflicts between inprocessing)
    pub inprocessing_interval: u64,
    /// Enable chronological backtracking
    pub enable_chronological_backtrack: bool,
    /// Chronological backtracking threshold (max distance from assertion level)
    pub chrono_backtrack_threshold: u32,
}

/// Restart strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Luby sequence restarts
    Luby,
    /// Geometric restarts
    Geometric,
    /// Glucose-style dynamic restarts based on LBD
    Glucose,
    /// Local restarts based on LBD trail
    LocalLbd,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            restart_interval: 100,
            restart_multiplier: 1.5,
            clause_deletion_threshold: 10000,
            var_decay: 0.95,
            clause_decay: 0.999,
            random_polarity_prob: 0.02,
            restart_strategy: RestartStrategy::Luby,
            enable_lazy_hyper_binary: true,
            use_chb_branching: false,
            use_lrb_branching: false,
            enable_inprocessing: false,
            inprocessing_interval: 5000,
            enable_chronological_backtrack: true,
            chrono_backtrack_threshold: 100,
        }
    }
}

/// Statistics for the solver
#[derive(Debug, Default, Clone)]
pub struct SolverStats {
    /// Number of decisions made
    pub decisions: u64,
    /// Number of propagations
    pub propagations: u64,
    /// Number of conflicts
    pub conflicts: u64,
    /// Number of restarts
    pub restarts: u64,
    /// Number of learned clauses
    pub learned_clauses: u64,
    /// Number of deleted clauses
    pub deleted_clauses: u64,
    /// Number of binary clauses learned
    pub binary_clauses: u64,
    /// Number of unit clauses learned
    pub unit_clauses: u64,
    /// Total LBD of learned clauses
    pub total_lbd: u64,
    /// Number of clause minimizations
    pub minimizations: u64,
    /// Literals removed by minimization
    pub literals_removed: u64,
    /// Number of chronological backtracks
    pub chrono_backtracks: u64,
    /// Number of non-chronological backtracks
    pub non_chrono_backtracks: u64,
}

impl SolverStats {
    /// Get average LBD of learned clauses
    #[must_use]
    pub fn avg_lbd(&self) -> f64 {
        if self.learned_clauses == 0 {
            0.0
        } else {
            self.total_lbd as f64 / self.learned_clauses as f64
        }
    }

    /// Get average decisions per conflict
    #[must_use]
    pub fn avg_decisions_per_conflict(&self) -> f64 {
        if self.conflicts == 0 {
            0.0
        } else {
            self.decisions as f64 / self.conflicts as f64
        }
    }

    /// Get propagations per conflict
    #[must_use]
    pub fn propagations_per_conflict(&self) -> f64 {
        if self.conflicts == 0 {
            0.0
        } else {
            self.propagations as f64 / self.conflicts as f64
        }
    }

    /// Get clause deletion ratio
    #[must_use]
    pub fn deletion_ratio(&self) -> f64 {
        if self.learned_clauses == 0 {
            0.0
        } else {
            self.deleted_clauses as f64 / self.learned_clauses as f64
        }
    }

    /// Get chronological backtrack ratio
    #[must_use]
    pub fn chrono_backtrack_ratio(&self) -> f64 {
        let total = self.chrono_backtracks + self.non_chrono_backtracks;
        if total == 0 {
            0.0
        } else {
            self.chrono_backtracks as f64 / total as f64
        }
    }

    /// Display formatted statistics
    pub fn display(&self) {
        println!("========== Solver Statistics ==========");
        println!("Decisions:              {:>12}", self.decisions);
        println!("Propagations:           {:>12}", self.propagations);
        println!("Conflicts:              {:>12}", self.conflicts);
        println!("Restarts:               {:>12}", self.restarts);
        println!("Learned clauses:        {:>12}", self.learned_clauses);
        println!("  - Unit clauses:       {:>12}", self.unit_clauses);
        println!("  - Binary clauses:     {:>12}", self.binary_clauses);
        println!("Deleted clauses:        {:>12}", self.deleted_clauses);
        println!("Minimizations:          {:>12}", self.minimizations);
        println!("Literals removed:       {:>12}", self.literals_removed);
        println!("Chrono backtracks:      {:>12}", self.chrono_backtracks);
        println!("Non-chrono backtracks:  {:>12}", self.non_chrono_backtracks);
        println!("---------------------------------------");
        println!("Avg LBD:                {:>12.2}", self.avg_lbd());
        println!(
            "Avg decisions/conflict: {:>12.2}",
            self.avg_decisions_per_conflict()
        );
        println!(
            "Propagations/conflict:  {:>12.2}",
            self.propagations_per_conflict()
        );
        println!(
            "Deletion ratio:         {:>12.2}%",
            self.deletion_ratio() * 100.0
        );
        println!(
            "Chrono backtrack ratio: {:>12.2}%",
            self.chrono_backtrack_ratio() * 100.0
        );
        println!("=======================================");
    }
}

/// CDCL SAT Solver
#[derive(Debug)]
pub struct Solver {
    /// Configuration
    pub(super) config: SolverConfig,
    /// Number of variables
    pub(super) num_vars: usize,
    /// Clause database
    pub(super) clauses: ClauseDatabase,
    /// Assignment trail
    pub(super) trail: Trail,
    /// Watch lists
    pub(super) watches: WatchLists,
    /// VSIDS branching heuristic
    pub(super) vsids: VSIDS,
    /// CHB branching heuristic
    pub(super) chb: CHB,
    /// LRB branching heuristic
    pub(super) lrb: LRB,
    /// Statistics
    pub(super) stats: SolverStats,
    /// Learnt clause for conflict analysis
    pub(super) learnt: SmallVec<[Lit; 16]>,
    /// Seen flags for conflict analysis
    pub(super) seen: Vec<bool>,
    /// Analyze stack
    pub(super) analyze_stack: Vec<Lit>,
    /// Current restart threshold
    pub(super) restart_threshold: u64,
    /// Assertions stack for incremental solving (number of original clauses)
    pub(super) assertion_levels: Vec<usize>,
    /// Trail sizes at each assertion level (for proper pop backtracking)
    pub(super) assertion_trail_sizes: Vec<usize>,
    /// Clause IDs added at each assertion level (for proper pop)
    pub(super) assertion_clause_ids: Vec<Vec<ClauseId>>,
    /// Model (if sat)
    pub(super) model: Vec<LBool>,
    /// Whether formula is trivially unsatisfiable
    pub(super) trivially_unsat: bool,
    /// Phase saving: last polarity assigned to each variable
    pub(super) phase: Vec<bool>,
    /// Luby sequence index for restarts
    pub(super) luby_index: u64,
    /// Level marks for LBD computation
    pub(super) level_marks: Vec<u32>,
    /// Current mark counter for LBD computation
    pub(super) lbd_mark: u32,
    /// Learned clause IDs for deletion
    pub(super) learned_clause_ids: Vec<ClauseId>,
    /// Number of conflicts since last clause deletion
    pub(super) conflicts_since_deletion: u64,
    /// PRNG state (xorshift64)
    pub(super) rng_state: u64,
    /// For Glucose-style restarts: average LBD of recent conflicts
    pub(super) recent_lbd_sum: u64,
    /// Number of conflicts contributing to recent_lbd_sum
    pub(super) recent_lbd_count: u64,
    /// Binary implication graph for fast binary clause propagation
    pub(super) binary_graph: BinaryImplicationGraph,
    /// Global average LBD for local restarts
    pub(super) global_lbd_sum: u64,
    /// Number of conflicts contributing to global LBD
    pub(super) global_lbd_count: u64,
    /// Conflicts since last local restart
    pub(super) conflicts_since_local_restart: u64,
    /// Conflicts since last inprocessing
    pub(super) conflicts_since_inprocessing: u64,
    /// Chronological backtracking helper
    pub(super) chrono_backtrack: ChronoBacktrack,
    /// Clause activity bump increment (for MapleSAT-style clause bumping)
    pub(super) clause_bump_increment: f64,
    /// Memory optimizer with size-class pools for clause allocation
    pub(super) memory_optimizer: MemoryOptimizer,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

impl Solver {
    /// Create a new solver
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(SolverConfig::default())
    }

    /// Create a new solver with configuration
    #[must_use]
    pub fn with_config(config: SolverConfig) -> Self {
        let chrono_enabled = config.enable_chronological_backtrack;
        let chrono_threshold = config.chrono_backtrack_threshold;

        Self {
            restart_threshold: config.restart_interval,
            config,
            num_vars: 0,
            clauses: ClauseDatabase::new(),
            trail: Trail::new(0),
            watches: WatchLists::new(0),
            vsids: VSIDS::new(0),
            chb: CHB::new(0),
            lrb: LRB::new(0),
            stats: SolverStats::default(),
            learnt: SmallVec::new(),
            seen: Vec::new(),
            analyze_stack: Vec::new(),
            assertion_levels: vec![0],
            assertion_trail_sizes: vec![0],
            assertion_clause_ids: vec![Vec::new()],
            model: Vec::new(),
            trivially_unsat: false,
            phase: Vec::new(),
            luby_index: 0,
            level_marks: Vec::new(),
            lbd_mark: 0,
            learned_clause_ids: Vec::new(),
            conflicts_since_deletion: 0,
            rng_state: 0x853c_49e6_748f_ea9b, // Random seed
            recent_lbd_sum: 0,
            recent_lbd_count: 0,
            binary_graph: BinaryImplicationGraph::new(0),
            global_lbd_sum: 0,
            global_lbd_count: 0,
            conflicts_since_local_restart: 0,
            conflicts_since_inprocessing: 0,
            chrono_backtrack: ChronoBacktrack::new(chrono_enabled, chrono_threshold),
            clause_bump_increment: 1.0,
            memory_optimizer: MemoryOptimizer::new(),
        }
    }

    /// Create a new variable
    pub fn new_var(&mut self) -> Var {
        let var = Var::new(self.num_vars as u32);
        self.num_vars += 1;
        self.trail.resize(self.num_vars);
        self.watches.resize(self.num_vars);
        self.binary_graph.resize(self.num_vars);
        self.vsids.insert(var);
        self.chb.insert(var);
        self.lrb.resize(self.num_vars);
        self.seen.resize(self.num_vars, false);
        self.model.resize(self.num_vars, LBool::Undef);
        self.phase.resize(self.num_vars, false); // Default phase: negative
        // Resize level_marks to at least num_vars (enough for decision levels)
        if self.level_marks.len() < self.num_vars {
            self.level_marks.resize(self.num_vars, 0);
        }
        var
    }

    /// Ensure we have at least n variables
    pub fn ensure_vars(&mut self, n: usize) {
        while self.num_vars < n {
            self.new_var();
        }
    }

    /// Add a clause
    pub fn add_clause(&mut self, lits: impl IntoIterator<Item = Lit>) -> bool {
        let mut clause_lits: SmallVec<[Lit; 8]> = lits.into_iter().collect();

        // Ensure we have all variables
        for lit in &clause_lits {
            let var_idx = lit.var().index();
            if var_idx >= self.num_vars {
                self.ensure_vars(var_idx + 1);
            }
        }

        // Remove duplicates and check for tautology
        clause_lits.sort_by_key(|l| l.code());
        clause_lits.dedup();

        // Check for tautology (x and ~x in same clause)
        for i in 0..clause_lits.len() {
            for j in (i + 1)..clause_lits.len() {
                if clause_lits[i] == clause_lits[j].negate() {
                    return true; // Tautology - always satisfied
                }
            }
        }

        // Handle special cases
        match clause_lits.len() {
            0 => {
                self.trivially_unsat = true;
                return false; // Empty clause - unsat
            }
            1 => {
                // Unit clause - enqueue at decision level 0
                // Unit clauses must be assigned at level 0 to survive backtracking.
                // After solve(), current_level may be > 0, so we must backtrack first.
                let lit = clause_lits[0];

                if self.trail.lit_value(lit).is_false() {
                    // The literal conflicts with the current trail.
                    // Check if the conflict is at decision level 0 (permanent constraint)
                    // or from a previous solve (can be retried after backtrack).
                    let var = lit.var();
                    let level = self.trail.level(var);
                    if level == 0 {
                        // Conflict with a level-0 assignment - truly UNSAT
                        self.trivially_unsat = true;
                        return false;
                    } else {
                        // Conflict with higher-level assignment from previous solve.
                        // Backtrack to root and assign the new unit literal at level 0.
                        self.backtrack_to_root();
                        self.trail.assign_decision(lit);
                        return true;
                    }
                }

                if self.trail.lit_value(lit).is_true() {
                    // Already satisfied - check if at level 0
                    let var = lit.var();
                    let level = self.trail.level(var);
                    if level == 0 {
                        // Already assigned at level 0, nothing to do
                        return true;
                    }
                    // Assigned at higher level - backtrack and reassign at level 0
                    self.backtrack_to_root();
                    self.trail.assign_decision(lit);
                    return true;
                }

                // Variable is unassigned - backtrack to level 0 first to ensure
                // the assignment is at level 0 (survives future backtracks)
                if self.trail.decision_level() > 0 {
                    self.backtrack_to_root();
                }
                self.trail.assign_decision(lit);
                return true;
            }
            2 => {
                // Binary clause - check if it conflicts with current assignment
                let lit0 = clause_lits[0];
                let lit1 = clause_lits[1];
                let val0 = self.trail.lit_value(lit0);
                let val1 = self.trail.lit_value(lit1);

                // If clause is satisfied, just add it
                if val0.is_true() || val1.is_true() {
                    // Clause already satisfied by current assignment
                    let clause_id = self.clauses.add_original(clause_lits.iter().copied());
                    if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                        current_level_clauses.push(clause_id);
                    }
                    self.binary_graph.add(lit0.negate(), lit1, clause_id);
                    self.binary_graph.add(lit1.negate(), lit0, clause_id);
                    self.watches
                        .add(lit0.negate(), Watcher::new(clause_id, lit1));
                    self.watches
                        .add(lit1.negate(), Watcher::new(clause_id, lit0));
                    return true;
                }

                // If both literals are false, we have a conflict
                if val0.is_false() && val1.is_false() {
                    // Check if both are at level 0
                    let level0 = self.trail.level(lit0.var());
                    let level1 = self.trail.level(lit1.var());

                    if level0 == 0 && level1 == 0 {
                        // Conflict at level 0 - UNSAT
                        self.trivially_unsat = true;
                        return false;
                    }

                    // Backtrack to level 0 and add clause
                    // The clause will be propagated on next solve()
                    self.backtrack_to_root();
                }

                // If one literal is false and one undefined, propagate
                // after adding the clause (via next solve())

                let clause_id = self.clauses.add_original(clause_lits.iter().copied());
                if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                    current_level_clauses.push(clause_id);
                }
                self.binary_graph.add(lit0.negate(), lit1, clause_id);
                self.binary_graph.add(lit1.negate(), lit0, clause_id);
                self.watches
                    .add(lit0.negate(), Watcher::new(clause_id, lit1));
                self.watches
                    .add(lit1.negate(), Watcher::new(clause_id, lit0));
                return true;
            }
            _ => {}
        }

        // Add clause (3+ literals)
        // Check if clause is satisfied or conflicts with current assignment
        let num_false = clause_lits
            .iter()
            .filter(|&l| self.trail.lit_value(*l).is_false())
            .count();
        let has_true = clause_lits
            .iter()
            .any(|l| self.trail.lit_value(*l).is_true());

        if !has_true && num_false == clause_lits.len() {
            // All literals are false - conflict
            // Check if all at level 0
            let all_at_zero = clause_lits.iter().all(|l| self.trail.level(l.var()) == 0);
            if all_at_zero {
                self.trivially_unsat = true;
                return false;
            }
            // Backtrack to level 0
            self.backtrack_to_root();
        }

        let clause_id = self.clauses.add_original(clause_lits.iter().copied());

        // Track clause for incremental solving
        if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
            current_level_clauses.push(clause_id);
        }

        // Set up watches - prefer non-false literals for watching
        let lit0 = clause_lits[0];
        let lit1 = clause_lits[1];

        self.watches
            .add(lit0.negate(), Watcher::new(clause_id, lit1));
        self.watches
            .add(lit1.negate(), Watcher::new(clause_id, lit0));

        true
    }

    /// Add a clause from DIMACS literals
    pub fn add_clause_dimacs(&mut self, lits: &[i32]) -> bool {
        self.add_clause(lits.iter().map(|&l| Lit::from_dimacs(l)))
    }

    /// Solve the SAT problem
    pub fn solve(&mut self) -> SolverResult {
        // Check if trivially unsatisfiable
        if self.trivially_unsat {
            return SolverResult::Unsat;
        }

        // Initial propagation
        if self.propagate().is_some() {
            return SolverResult::Unsat;
        }

        loop {
            // Propagate
            if let Some(conflict) = self.propagate() {
                self.stats.conflicts += 1;
                self.conflicts_since_inprocessing += 1;

                if self.trail.decision_level() == 0 {
                    return SolverResult::Unsat;
                }

                // Analyze conflict
                let (backtrack_level, learnt_clause) = self.analyze(conflict);

                // Backtrack with phase saving
                self.backtrack_with_phase_saving(backtrack_level);

                // Learn clause
                if learnt_clause.len() == 1 {
                    // Store unit learned clause in database for persistence
                    let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
                    self.stats.learned_clauses += 1;
                    self.stats.unit_clauses += 1;
                    self.learned_clause_ids.push(clause_id);

                    // Track for incremental solving
                    if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                        current_level_clauses.push(clause_id);
                    }

                    self.trail.assign_decision(learnt_clause[0]);
                } else {
                    // Compute LBD for the learned clause
                    let lbd = self.compute_lbd(&learnt_clause);

                    // Track recent LBD for Glucose-style and local restarts
                    self.recent_lbd_sum += u64::from(lbd);
                    self.recent_lbd_count += 1;
                    self.global_lbd_sum += u64::from(lbd);
                    self.global_lbd_count += 1;

                    // Reset recent LBD tracking periodically
                    if self.recent_lbd_count >= 5000 {
                        self.recent_lbd_sum /= 2;
                        self.recent_lbd_count /= 2;
                    }

                    let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
                    self.stats.learned_clauses += 1;

                    // Set LBD score for the clause
                    if let Some(clause) = self.clauses.get_mut(clause_id) {
                        clause.lbd = lbd;
                    }

                    // Track learned clause for potential deletion
                    self.learned_clause_ids.push(clause_id);

                    // Track clause for incremental solving
                    if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                        current_level_clauses.push(clause_id);
                    }

                    // Watch first two literals
                    let lit0 = learnt_clause[0];
                    let lit1 = learnt_clause[1];
                    self.watches
                        .add(lit0.negate(), Watcher::new(clause_id, lit1));
                    self.watches
                        .add(lit1.negate(), Watcher::new(clause_id, lit0));

                    // Propagate the asserting literal
                    self.trail.assign_propagation(learnt_clause[0], clause_id);
                }

                // Decay activities
                self.vsids.decay();
                self.chb.decay();
                self.lrb.decay();
                self.lrb.on_conflict();
                self.clauses.decay_activity(self.config.clause_decay);
                // Increase clause bump increment (inverse of decay)
                self.clause_bump_increment /= self.config.clause_decay;

                // Track conflicts for clause deletion
                self.conflicts_since_deletion += 1;

                // Periodic clause database reduction
                if self.conflicts_since_deletion >= self.config.clause_deletion_threshold as u64 {
                    self.reduce_clause_database();
                    self.conflicts_since_deletion = 0;

                    // Vivification after clause database reduction (at level 0 after restart)
                    if self.stats.restarts.is_multiple_of(10) {
                        let saved_level = self.trail.decision_level();
                        if saved_level == 0 {
                            self.vivify_clauses();
                        }
                    }
                }

                // Check for restart
                if self.stats.conflicts >= self.restart_threshold {
                    self.restart();
                }

                // Periodic inprocessing
                if self.config.enable_inprocessing
                    && self.conflicts_since_inprocessing >= self.config.inprocessing_interval
                {
                    self.inprocess();
                    self.conflicts_since_inprocessing = 0;
                }
            } else {
                // No conflict - try to decide
                if let Some(var) = self.pick_branch_var() {
                    self.stats.decisions += 1;
                    self.trail.new_decision_level();

                    // Use phase saving with random polarity
                    let polarity = if self.rand_bool(self.config.random_polarity_prob) {
                        // Random polarity
                        self.rand_bool(0.5)
                    } else {
                        // Saved phase
                        self.phase[var.index()]
                    };
                    let lit = if polarity {
                        Lit::pos(var)
                    } else {
                        Lit::neg(var)
                    };
                    self.trail.assign_decision(lit);
                } else {
                    // All variables assigned - SAT
                    self.save_model();
                    return SolverResult::Sat;
                }
            }
        }
    }

    /// Solve with assumptions and return unsat core if UNSAT
    ///
    /// This is the key method for MaxSAT: it solves under assumptions and
    /// if the result is UNSAT, returns the subset of assumptions in the core.
    ///
    /// # Arguments
    /// * `assumptions` - Literals that must be true
    ///
    /// # Returns
    /// * `(SolverResult, Option<Vec<Lit>>)` - Result and unsat core (if UNSAT)
    pub fn solve_with_assumptions(
        &mut self,
        assumptions: &[Lit],
    ) -> (SolverResult, Option<Vec<Lit>>) {
        if self.trivially_unsat {
            return (SolverResult::Unsat, Some(Vec::new()));
        }

        // Ensure all assumption variables exist
        for &lit in assumptions {
            while self.num_vars <= lit.var().index() {
                self.new_var();
            }
        }

        // Initial propagation at level 0
        if self.propagate().is_some() {
            return (SolverResult::Unsat, Some(Vec::new()));
        }

        // Create a new decision level for assumptions
        let assumption_level_start = self.trail.decision_level();

        // Assign assumptions as decisions
        for (i, &lit) in assumptions.iter().enumerate() {
            // Check if already assigned
            let value = self.trail.lit_value(lit);
            if value.is_true() {
                continue; // Already satisfied
            }
            if value.is_false() {
                // Conflict with assumption - extract core from conflicting assumptions
                let core = self.extract_assumption_core(assumptions, i);
                self.backtrack(assumption_level_start);
                return (SolverResult::Unsat, Some(core));
            }

            // Make decision for assumption
            self.trail.new_decision_level();
            self.trail.assign_decision(lit);

            // Propagate after each assumption
            if let Some(_conflict) = self.propagate() {
                // Conflict during assumption propagation
                let core = self.analyze_assumption_conflict(assumptions);
                self.backtrack(assumption_level_start);
                return (SolverResult::Unsat, Some(core));
            }
        }

        // Now solve normally
        loop {
            if let Some(conflict) = self.propagate() {
                self.stats.conflicts += 1;

                // Check if conflict involves assumptions
                let backtrack_level = self.analyze_conflict_level(conflict);

                if backtrack_level <= assumption_level_start {
                    // Conflict forces backtracking past assumptions - UNSAT
                    let core = self.analyze_assumption_conflict(assumptions);
                    self.backtrack(assumption_level_start);
                    return (SolverResult::Unsat, Some(core));
                }

                let (bt_level, learnt_clause) = self.analyze(conflict);
                self.backtrack_with_phase_saving(bt_level.max(assumption_level_start + 1));
                self.learn_clause(learnt_clause);

                self.vsids.decay();
                self.clauses.decay_activity(self.config.clause_decay);
                self.handle_clause_deletion_and_restart_limited(assumption_level_start);
            } else {
                // No conflict - try to decide
                if let Some(var) = self.pick_branch_var() {
                    self.stats.decisions += 1;
                    self.trail.new_decision_level();

                    let polarity = if self.rand_bool(self.config.random_polarity_prob) {
                        self.rand_bool(0.5)
                    } else {
                        self.phase.get(var.index()).copied().unwrap_or(false)
                    };
                    let lit = if polarity {
                        Lit::pos(var)
                    } else {
                        Lit::neg(var)
                    };
                    self.trail.assign_decision(lit);
                } else {
                    // All variables assigned - SAT
                    self.save_model();
                    self.backtrack(assumption_level_start);
                    return (SolverResult::Sat, None);
                }
            }
        }
    }

    /// Solve with theory integration via callbacks
    ///
    /// This implements the CDCL(T) loop:
    /// 1. BCP (Boolean Constraint Propagation)
    /// 2. Theory propagation (via callback)
    /// 3. On conflict: analyze and learn
    /// 4. Decision
    /// 5. Final theory check when all vars assigned
    pub fn solve_with_theory<T: TheoryCallback>(&mut self, theory: &mut T) -> SolverResult {
        if self.trivially_unsat {
            return SolverResult::Unsat;
        }

        // Initial propagation
        if self.propagate().is_some() {
            return SolverResult::Unsat;
        }

        // Track how many assignments have been sent to the theory.
        // We only send NEW assignments (not previously processed ones) to avoid
        // duplicate theory constraints that would cause spurious UNSAT.
        let mut theory_processed: usize = 0;

        loop {
            // Boolean propagation
            if let Some(conflict) = self.propagate() {
                self.stats.conflicts += 1;

                if self.trail.decision_level() == 0 {
                    return SolverResult::Unsat;
                }

                let (backtrack_level, learnt_clause) = self.analyze(conflict);
                theory.on_backtrack(backtrack_level);
                self.backtrack_with_phase_saving(backtrack_level);
                // After backtrack, the trail may be shorter; update processed count
                theory_processed = theory_processed.min(self.trail.assignments().len());
                self.learn_clause(learnt_clause);

                self.vsids.decay();
                self.clauses.decay_activity(self.config.clause_decay);
                self.handle_clause_deletion_and_restart();
                continue;
            }

            // Theory propagation check after each assignment
            loop {
                // Get only NEW (unprocessed) assignments and notify theory
                let assignments = self.trail.assignments().to_vec();
                let mut theory_conflict = None;
                let mut theory_propagations = Vec::new();

                // Check only NEW assignments with theory (skip already-processed ones).
                // Guard against stale theory_processed after backtracks/restarts.
                let safe_start = theory_processed.min(assignments.len());
                for &lit in &assignments[safe_start..] {
                    match theory.on_assignment(lit) {
                        TheoryCheckResult::Sat => {}
                        TheoryCheckResult::Conflict(conflict_lits) => {
                            theory_conflict = Some(conflict_lits);
                            break;
                        }
                        TheoryCheckResult::Propagated(props) => {
                            theory_propagations.extend(props);
                        }
                    }
                }
                // Update processed count
                theory_processed = assignments.len();

                // Handle theory conflict
                if let Some(conflict_lits) = theory_conflict {
                    self.stats.conflicts += 1;

                    if self.trail.decision_level() == 0 {
                        return SolverResult::Unsat;
                    }

                    let (backtrack_level, learnt_clause) =
                        self.analyze_theory_conflict(&conflict_lits);

                    // Empty learned clause signals all-level-0 conflict = fundamental UNSAT
                    if learnt_clause.is_empty() {
                        self.trivially_unsat = true;
                        return SolverResult::Unsat;
                    }

                    theory.on_backtrack(backtrack_level);
                    self.backtrack_with_phase_saving(backtrack_level);
                    // After backtrack, update theory_processed to trail length
                    theory_processed = theory_processed.min(self.trail.assignments().len());
                    self.learn_clause(learnt_clause);

                    self.vsids.decay();
                    self.clauses.decay_activity(self.config.clause_decay);
                    self.handle_clause_deletion_and_restart();
                    continue;
                }

                // Handle theory propagations
                let mut made_propagation = false;
                for (lit, reason_lits) in theory_propagations {
                    if !self.trail.is_assigned(lit.var()) {
                        // Add reason clause and propagate
                        let clause_id = self.add_theory_reason_clause(&reason_lits, lit);
                        self.trail.assign_propagation(lit, clause_id);
                        made_propagation = true;
                    }
                }

                if made_propagation {
                    // Re-run Boolean propagation
                    if let Some(conflict) = self.propagate() {
                        self.stats.conflicts += 1;

                        if self.trail.decision_level() == 0 {
                            return SolverResult::Unsat;
                        }

                        let (backtrack_level, learnt_clause) = self.analyze(conflict);
                        theory.on_backtrack(backtrack_level);
                        self.backtrack_with_phase_saving(backtrack_level);
                        // After backtrack, the trail is shorter; update processed count
                        theory_processed = theory_processed.min(self.trail.assignments().len());
                        self.learn_clause(learnt_clause);

                        self.vsids.decay();
                        self.clauses.decay_activity(self.config.clause_decay);
                        self.handle_clause_deletion_and_restart();
                    }
                    continue;
                }

                break;
            }

            // Try to decide
            if let Some(var) = self.pick_branch_var() {
                self.stats.decisions += 1;
                self.trail.new_decision_level();
                let new_level = self.trail.decision_level();
                theory.on_new_level(new_level);

                let polarity = if self.rand_bool(self.config.random_polarity_prob) {
                    self.rand_bool(0.5)
                } else {
                    self.phase[var.index()]
                };
                let lit = if polarity {
                    Lit::pos(var)
                } else {
                    Lit::neg(var)
                };
                self.trail.assign_decision(lit);
            } else {
                // All variables assigned - do final theory check
                match theory.final_check() {
                    TheoryCheckResult::Sat => {
                        self.save_model();
                        return SolverResult::Sat;
                    }
                    TheoryCheckResult::Conflict(conflict_lits) => {
                        self.stats.conflicts += 1;

                        if self.trail.decision_level() == 0 {
                            return SolverResult::Unsat;
                        }

                        let (backtrack_level, learnt_clause) =
                            self.analyze_theory_conflict(&conflict_lits);

                        // If all conflict literals are at level 0, analyze_theory_conflict
                        // returns an empty learned clause as a signal of fundamental UNSAT.
                        if learnt_clause.is_empty() {
                            self.trivially_unsat = true;
                            return SolverResult::Unsat;
                        }

                        theory.on_backtrack(backtrack_level);
                        self.backtrack_with_phase_saving(backtrack_level);
                        // After backtrack, update theory_processed
                        theory_processed = theory_processed.min(self.trail.assignments().len());
                        self.learn_clause(learnt_clause);

                        self.vsids.decay();
                        self.clauses.decay_activity(self.config.clause_decay);
                        self.handle_clause_deletion_and_restart();
                    }
                    TheoryCheckResult::Propagated(props) => {
                        // Handle late propagations
                        for (lit, reason_lits) in props {
                            if !self.trail.is_assigned(lit.var()) {
                                let clause_id = self.add_theory_reason_clause(&reason_lits, lit);
                                self.trail.assign_propagation(lit, clause_id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get the model (if sat)
    #[must_use]
    pub fn model(&self) -> &[LBool] {
        &self.model
    }

    /// Get the value of a variable in the model
    #[must_use]
    pub fn model_value(&self, var: Var) -> LBool {
        self.model.get(var.index()).copied().unwrap_or(LBool::Undef)
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Get memory optimizer statistics
    #[must_use]
    pub fn memory_opt_stats(&self) -> &crate::memory_opt::MemoryOptStats {
        self.memory_optimizer.stats()
    }

    /// Get number of variables
    #[must_use]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get number of clauses
    #[must_use]
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Push a new assertion level (for incremental solving)
    ///
    /// This saves the current state so that clauses added after this point
    /// can be removed with pop(). Automatically backtracks to decision level 0
    /// to ensure a clean state for adding new constraints.
    pub fn push(&mut self) {
        // Backtrack to level 0 to ensure clean state
        // This is necessary because solve() may leave assignments on the trail
        // Use phase-saving backtrack to properly re-insert variables into decision heaps
        self.backtrack_with_phase_saving(0);

        self.assertion_levels.push(self.clauses.num_original());
        self.assertion_trail_sizes.push(self.trail.size());
        self.assertion_clause_ids.push(Vec::new());
    }

    /// Pop to previous assertion level
    pub fn pop(&mut self) {
        if self.assertion_levels.len() > 1 {
            self.assertion_levels.pop();

            // Get the trail size to backtrack to
            let trail_size = self.assertion_trail_sizes.pop().unwrap_or(0);

            // Remove all clauses added at this assertion level
            if let Some(clause_ids_to_remove) = self.assertion_clause_ids.pop() {
                for clause_id in clause_ids_to_remove {
                    // Remove from clause database
                    self.clauses.remove(clause_id);

                    // Remove from learned clause tracking if it's a learned clause
                    self.learned_clause_ids.retain(|&id| id != clause_id);

                    // Note: Watch lists will be cleaned up naturally during propagation
                    // as they check if clauses are deleted before using them
                }
            }

            // Backtrack trail to the exact size it was at push()
            // This properly handles unit clauses that were added after push
            // Note: backtrack_to_size clears values but doesn't re-insert into heaps,
            // so we need to manually re-insert unassigned variables.
            let current_size = self.trail.size();
            if current_size > trail_size {
                // Collect variables that will be unassigned
                let mut unassigned_vars = Vec::new();
                for i in trail_size..current_size {
                    let lit = self.trail.assignments()[i];
                    unassigned_vars.push(lit.var());
                }

                self.trail.backtrack_to_size(trail_size);

                // Re-insert unassigned variables into decision heaps
                for var in unassigned_vars {
                    if !self.vsids.contains(var) {
                        self.vsids.insert(var);
                    }
                    if !self.chb.contains(var) {
                        self.chb.insert(var);
                    }
                    self.lrb.unassign(var);
                }
            }

            // Ensure we're at decision level 0 with proper heap re-insertion
            self.backtrack_with_phase_saving(0);

            // Clear the trivially_unsat flag as we've removed problematic clauses
            self.trivially_unsat = false;
        }
    }

    /// Backtrack to decision level 0 (for AllSAT enumeration)
    ///
    /// This is necessary after a SAT result before adding blocking clauses
    /// to ensure the new clauses can trigger propagation correctly.
    /// Uses phase-saving backtrack to properly re-insert unassigned variables
    /// into the decision heaps (VSIDS, CHB, LRB).
    pub fn backtrack_to_root(&mut self) {
        self.backtrack_with_phase_saving(0);
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.clauses = ClauseDatabase::new();
        self.trail.clear();
        self.watches.clear();
        self.vsids.clear();
        self.chb.clear();
        self.stats = SolverStats::default();
        self.learnt.clear();
        self.seen.clear();
        self.analyze_stack.clear();
        self.assertion_levels.clear();
        self.assertion_levels.push(0);
        self.assertion_trail_sizes.clear();
        self.assertion_trail_sizes.push(0);
        self.assertion_clause_ids.clear();
        self.assertion_clause_ids.push(Vec::new());
        self.model.clear();
        self.num_vars = 0;
        self.restart_threshold = self.config.restart_interval;
        self.trivially_unsat = false;
        self.phase.clear();
        self.luby_index = 0;
        self.level_marks.clear();
        self.lbd_mark = 0;
        self.learned_clause_ids.clear();
        self.conflicts_since_deletion = 0;
        self.rng_state = 0x853c_49e6_748f_ea9b;
        self.recent_lbd_sum = 0;
        self.recent_lbd_count = 0;
        self.binary_graph.clear();
        self.global_lbd_sum = 0;
        self.global_lbd_count = 0;
        self.conflicts_since_local_restart = 0;
    }

    /// Get the current trail (for theory solvers)
    #[must_use]
    pub fn trail(&self) -> &Trail {
        &self.trail
    }

    /// Get the current decision level
    #[must_use]
    pub fn decision_level(&self) -> u32 {
        self.trail.decision_level()
    }

    /// Debug method: print all learned clauses
    pub fn debug_print_learned_clauses(&self) {
        println!(
            "=== Learned Clauses ({}) ===",
            self.learned_clause_ids.len()
        );
        for (i, &cid) in self.learned_clause_ids.iter().enumerate() {
            if let Some(clause) = self.clauses.get(cid)
                && !clause.deleted
            {
                let lits: Vec<String> = clause
                    .lits
                    .iter()
                    .map(|lit| {
                        let var = lit.var().index();
                        if lit.is_pos() {
                            format!("v{}", var)
                        } else {
                            format!("~v{}", var)
                        }
                    })
                    .collect();
                println!(
                    "  Learned {}: ({}), LBD={}",
                    i,
                    lits.join(" | "),
                    clause.lbd
                );
            }
        }
    }

    /// Debug method: print binary implication graph entries
    pub fn debug_print_binary_graph(&self) {
        println!("=== Binary Implication Graph ===");
        for lit_code in 0..(self.num_vars * 2) {
            let lit = Lit::from_code(lit_code as u32);
            let implications = self.binary_graph.get(lit);
            if !implications.is_empty() {
                let lit_str = if lit.is_pos() {
                    format!("v{}", lit.var().index())
                } else {
                    format!("~v{}", lit.var().index())
                };
                for &(implied, _cid) in implications {
                    let impl_str = if implied.is_pos() {
                        format!("v{}", implied.var().index())
                    } else {
                        format!("~v{}", implied.var().index())
                    };
                    println!("  {} -> {}", lit_str, impl_str);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_sat() {
        let mut solver = Solver::new();
        assert_eq!(solver.solve(), SolverResult::Sat);
    }

    #[test]
    fn test_simple_sat() {
        let mut solver = Solver::new();
        let _x = solver.new_var();
        let _y = solver.new_var();

        // x or y
        solver.add_clause_dimacs(&[1, 2]);
        // not x or y
        solver.add_clause_dimacs(&[-1, 2]);

        assert_eq!(solver.solve(), SolverResult::Sat);
        assert!(solver.model_value(Var::new(1)).is_true()); // y must be true
    }

    #[test]
    fn test_simple_unsat() {
        let mut solver = Solver::new();
        let _x = solver.new_var();

        // x
        solver.add_clause_dimacs(&[1]);
        // not x
        solver.add_clause_dimacs(&[-1]);

        assert_eq!(solver.solve(), SolverResult::Unsat);
    }

    #[test]
    fn test_pigeonhole_2_1() {
        // 2 pigeons, 1 hole - UNSAT
        let mut solver = Solver::new();
        let _p1h1 = solver.new_var(); // pigeon 1 in hole 1
        let _p2h1 = solver.new_var(); // pigeon 2 in hole 1

        // Each pigeon must be in some hole
        solver.add_clause_dimacs(&[1]); // p1 in h1
        solver.add_clause_dimacs(&[2]); // p2 in h1

        // No hole can have two pigeons
        solver.add_clause_dimacs(&[-1, -2]); // not (p1h1 and p2h1)

        assert_eq!(solver.solve(), SolverResult::Unsat);
    }

    #[test]
    fn test_3sat_random() {
        let mut solver = Solver::new();
        for _ in 0..10 {
            solver.new_var();
        }

        // Random 3-SAT instance (likely SAT)
        solver.add_clause_dimacs(&[1, 2, 3]);
        solver.add_clause_dimacs(&[-1, 4, 5]);
        solver.add_clause_dimacs(&[2, -3, 6]);
        solver.add_clause_dimacs(&[-4, 7, 8]);
        solver.add_clause_dimacs(&[5, -6, 9]);
        solver.add_clause_dimacs(&[-7, 8, 10]);
        solver.add_clause_dimacs(&[1, -8, -9]);
        solver.add_clause_dimacs(&[-2, 3, -10]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_luby_sequence() {
        // Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
        assert_eq!(Solver::luby(0), 1);
        assert_eq!(Solver::luby(1), 1);
        assert_eq!(Solver::luby(2), 2);
        assert_eq!(Solver::luby(3), 1);
        assert_eq!(Solver::luby(4), 1);
        assert_eq!(Solver::luby(5), 2);
        assert_eq!(Solver::luby(6), 4);
        assert_eq!(Solver::luby(7), 1);
    }

    #[test]
    fn test_phase_saving() {
        let mut solver = Solver::new();
        for _ in 0..5 {
            solver.new_var();
        }

        // Set up a problem where phase saving helps
        solver.add_clause_dimacs(&[1, 2]);
        solver.add_clause_dimacs(&[-1, 3]);
        solver.add_clause_dimacs(&[-2, 4]);
        solver.add_clause_dimacs(&[-3, -4, 5]);
        solver.add_clause_dimacs(&[-5, 1]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_lbd_computation() {
        // Test that clause deletion can handle a problem that generates learned clauses
        let mut solver = Solver::with_config(SolverConfig {
            clause_deletion_threshold: 5, // Trigger deletion quickly
            ..SolverConfig::default()
        });

        for _ in 0..20 {
            solver.new_var();
        }

        // A harder problem to generate more conflicts and learned clauses
        // PHP(3,2): 3 pigeons, 2 holes - UNSAT
        // Variables: p_i_h (pigeon i in hole h)
        // p11=1, p12=2, p21=3, p22=4, p31=5, p32=6

        // Each pigeon must be in some hole
        solver.add_clause_dimacs(&[1, 2]); // p1 in h1 or h2
        solver.add_clause_dimacs(&[3, 4]); // p2 in h1 or h2
        solver.add_clause_dimacs(&[5, 6]); // p3 in h1 or h2

        // No hole can have two pigeons
        solver.add_clause_dimacs(&[-1, -3]); // not (p1h1 and p2h1)
        solver.add_clause_dimacs(&[-1, -5]); // not (p1h1 and p3h1)
        solver.add_clause_dimacs(&[-3, -5]); // not (p2h1 and p3h1)
        solver.add_clause_dimacs(&[-2, -4]); // not (p1h2 and p2h2)
        solver.add_clause_dimacs(&[-2, -6]); // not (p1h2 and p3h2)
        solver.add_clause_dimacs(&[-4, -6]); // not (p2h2 and p3h2)

        let result = solver.solve();
        assert_eq!(result, SolverResult::Unsat);
        // Verify we had some conflicts (and thus learned clauses)
        assert!(solver.stats().conflicts > 0);
    }

    #[test]
    fn test_clause_activity_decay() {
        let mut solver = Solver::new();
        for _ in 0..10 {
            solver.new_var();
        }

        // Add some clauses
        solver.add_clause_dimacs(&[1, 2, 3]);
        solver.add_clause_dimacs(&[-1, 4, 5]);
        solver.add_clause_dimacs(&[-2, -3, 6]);

        // Solve (should be SAT)
        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);
    }

    #[test]
    fn test_clause_minimization() {
        // Test that clause minimization works correctly on a problem
        // that will generate learned clauses
        let mut solver = Solver::new();

        for _ in 0..15 {
            solver.new_var();
        }

        // A problem structure that generates conflicts and learned clauses
        // Graph coloring with 3 colors on 5 vertices
        // Vertices: 1-5, Colors: R(0-4), G(5-9), B(10-14)

        // Each vertex has at least one color
        solver.add_clause_dimacs(&[1, 6, 11]); // v1: R or G or B
        solver.add_clause_dimacs(&[2, 7, 12]); // v2
        solver.add_clause_dimacs(&[3, 8, 13]); // v3
        solver.add_clause_dimacs(&[4, 9, 14]); // v4
        solver.add_clause_dimacs(&[5, 10, 15]); // v5

        // At most one color per vertex (pairwise exclusion)
        solver.add_clause_dimacs(&[-1, -6]); // v1: not (R and G)
        solver.add_clause_dimacs(&[-1, -11]); // v1: not (R and B)
        solver.add_clause_dimacs(&[-6, -11]); // v1: not (G and B)

        solver.add_clause_dimacs(&[-2, -7]);
        solver.add_clause_dimacs(&[-2, -12]);
        solver.add_clause_dimacs(&[-7, -12]);

        solver.add_clause_dimacs(&[-3, -8]);
        solver.add_clause_dimacs(&[-3, -13]);
        solver.add_clause_dimacs(&[-8, -13]);

        // Adjacent vertices have different colors (edges: 1-2, 2-3, 3-4, 4-5)
        solver.add_clause_dimacs(&[-1, -2]); // edge 1-2: not both R
        solver.add_clause_dimacs(&[-6, -7]); // edge 1-2: not both G
        solver.add_clause_dimacs(&[-11, -12]); // edge 1-2: not both B

        solver.add_clause_dimacs(&[-2, -3]); // edge 2-3
        solver.add_clause_dimacs(&[-7, -8]);
        solver.add_clause_dimacs(&[-12, -13]);

        let result = solver.solve();
        assert_eq!(result, SolverResult::Sat);

        // The solver may or may not have conflicts/learned clauses depending on
        // the decision heuristic. The key is that the result is correct.
        // If there are learned clauses, minimization would have been applied.
    }

    /// A simple theory callback that does nothing (pure SAT)
    struct NullTheory;

    impl TheoryCallback for NullTheory {
        fn on_assignment(&mut self, _lit: Lit) -> TheoryCheckResult {
            TheoryCheckResult::Sat
        }

        fn final_check(&mut self) -> TheoryCheckResult {
            TheoryCheckResult::Sat
        }

        fn on_backtrack(&mut self, _level: u32) {}
    }

    #[test]
    fn test_solve_with_theory_sat() {
        let mut solver = Solver::new();
        let mut theory = NullTheory;

        let _x = solver.new_var();
        let _y = solver.new_var();

        // x or y
        solver.add_clause_dimacs(&[1, 2]);
        // not x or y
        solver.add_clause_dimacs(&[-1, 2]);

        assert_eq!(solver.solve_with_theory(&mut theory), SolverResult::Sat);
        assert!(solver.model_value(Var::new(1)).is_true()); // y must be true
    }

    #[test]
    fn test_solve_with_theory_unsat() {
        let mut solver = Solver::new();
        let mut theory = NullTheory;

        let _x = solver.new_var();

        // x
        solver.add_clause_dimacs(&[1]);
        // not x
        solver.add_clause_dimacs(&[-1]);

        assert_eq!(solver.solve_with_theory(&mut theory), SolverResult::Unsat);
    }

    /// A theory that forces x0 => x1 (if x0 is true, x1 must be true)
    struct ImplicationTheory {
        /// Track if x0 is assigned true
        x0_true: bool,
    }

    impl ImplicationTheory {
        fn new() -> Self {
            Self { x0_true: false }
        }
    }

    impl TheoryCallback for ImplicationTheory {
        fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult {
            // If x0 becomes true, propagate x1
            if lit.var().index() == 0 && lit.is_pos() {
                self.x0_true = true;
                // Propagate: x1 must be true because x0 is true
                // The reason is: ~x0 (if x0 were false, we wouldn't need x1)
                let reason: SmallVec<[Lit; 8]> = smallvec::smallvec![Lit::pos(Var::new(0))];
                return TheoryCheckResult::Propagated(vec![(Lit::pos(Var::new(1)), reason)]);
            }
            TheoryCheckResult::Sat
        }

        fn final_check(&mut self) -> TheoryCheckResult {
            TheoryCheckResult::Sat
        }

        fn on_backtrack(&mut self, _level: u32) {
            self.x0_true = false;
        }
    }

    #[test]
    fn test_theory_propagation() {
        let mut solver = Solver::new();
        let mut theory = ImplicationTheory::new();

        let _x0 = solver.new_var();
        let _x1 = solver.new_var();

        // Force x0 to be true
        solver.add_clause_dimacs(&[1]);

        let result = solver.solve_with_theory(&mut theory);
        assert_eq!(result, SolverResult::Sat);

        // x0 should be true (forced by clause)
        assert!(solver.model_value(Var::new(0)).is_true());
        // x1 should also be true (propagated by theory)
        assert!(solver.model_value(Var::new(1)).is_true());
    }

    /// Theory that says x0 and x1 can't both be true
    struct MutexTheory {
        x0_true: Option<Lit>,
        x1_true: Option<Lit>,
    }

    impl MutexTheory {
        fn new() -> Self {
            Self {
                x0_true: None,
                x1_true: None,
            }
        }
    }

    impl TheoryCallback for MutexTheory {
        fn on_assignment(&mut self, lit: Lit) -> TheoryCheckResult {
            if lit.var().index() == 0 && lit.is_pos() {
                self.x0_true = Some(lit);
            }
            if lit.var().index() == 1 && lit.is_pos() {
                self.x1_true = Some(lit);
            }

            // If both are true, conflict
            if self.x0_true.is_some() && self.x1_true.is_some() {
                // Conflict clause: ~x0 or ~x1 (at least one must be false)
                let conflict: SmallVec<[Lit; 8]> = smallvec::smallvec![
                    Lit::pos(Var::new(0)), // x0 is true (we negate in conflict)
                    Lit::pos(Var::new(1))  // x1 is true
                ];
                return TheoryCheckResult::Conflict(conflict);
            }
            TheoryCheckResult::Sat
        }

        fn final_check(&mut self) -> TheoryCheckResult {
            if self.x0_true.is_some() && self.x1_true.is_some() {
                let conflict: SmallVec<[Lit; 8]> =
                    smallvec::smallvec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))];
                return TheoryCheckResult::Conflict(conflict);
            }
            TheoryCheckResult::Sat
        }

        fn on_backtrack(&mut self, _level: u32) {
            self.x0_true = None;
            self.x1_true = None;
        }
    }

    #[test]
    fn test_theory_conflict() {
        let mut solver = Solver::new();
        let mut theory = MutexTheory::new();

        let _x0 = solver.new_var();
        let _x1 = solver.new_var();

        // Force both x0 and x1 to be true (should cause theory conflict)
        solver.add_clause_dimacs(&[1]);
        solver.add_clause_dimacs(&[2]);

        let result = solver.solve_with_theory(&mut theory);
        assert_eq!(result, SolverResult::Unsat);
    }

    #[test]
    fn test_solve_with_assumptions_sat() {
        let mut solver = Solver::new();

        let x0 = solver.new_var();
        let x1 = solver.new_var();

        // x0 \/ x1
        solver.add_clause([Lit::pos(x0), Lit::pos(x1)]);

        // Assume x0 = true
        let assumptions = [Lit::pos(x0)];
        let (result, core) = solver.solve_with_assumptions(&assumptions);

        assert_eq!(result, SolverResult::Sat);
        assert!(core.is_none());
    }

    #[test]
    fn test_solve_with_assumptions_unsat() {
        let mut solver = Solver::new();

        let x0 = solver.new_var();
        let x1 = solver.new_var();

        // x0 -> ~x1 (encoded as ~x0 \/ ~x1)
        solver.add_clause([Lit::neg(x0), Lit::neg(x1)]);

        // Assume both x0 = true and x1 = true (should be UNSAT)
        let assumptions = [Lit::pos(x0), Lit::pos(x1)];
        let (result, core) = solver.solve_with_assumptions(&assumptions);

        assert_eq!(result, SolverResult::Unsat);
        assert!(core.is_some());
        let core = core.expect("UNSAT result must have conflict core");
        // Core should contain at least one of the conflicting assumptions
        assert!(!core.is_empty());
    }

    #[test]
    fn test_solve_with_assumptions_core_extraction() {
        let mut solver = Solver::new();

        let x0 = solver.new_var();
        let x1 = solver.new_var();
        let x2 = solver.new_var();

        // ~x0 (x0 must be false)
        solver.add_clause([Lit::neg(x0)]);

        // Assume x0 = true, x1 = true, x2 = true
        // Only x0 should be in the core
        let assumptions = [Lit::pos(x0), Lit::pos(x1), Lit::pos(x2)];
        let (result, core) = solver.solve_with_assumptions(&assumptions);

        assert_eq!(result, SolverResult::Unsat);
        assert!(core.is_some());
        let core = core.expect("UNSAT result must have conflict core");
        // x0 should be in the core
        assert!(core.contains(&Lit::pos(x0)));
    }

    #[test]
    fn test_solve_with_assumptions_incremental() {
        let mut solver = Solver::new();

        let x0 = solver.new_var();
        let x1 = solver.new_var();

        // x0 \/ x1
        solver.add_clause([Lit::pos(x0), Lit::pos(x1)]);

        // First: assume ~x0 (should be SAT with x1 = true)
        let (result1, _) = solver.solve_with_assumptions(&[Lit::neg(x0)]);
        assert_eq!(result1, SolverResult::Sat);

        // Second: assume ~x0 and ~x1 (should be UNSAT)
        let (result2, core2) = solver.solve_with_assumptions(&[Lit::neg(x0), Lit::neg(x1)]);
        assert_eq!(result2, SolverResult::Unsat);
        assert!(core2.is_some());

        // Third: assume x0 (should be SAT again)
        let (result3, _) = solver.solve_with_assumptions(&[Lit::pos(x0)]);
        assert_eq!(result3, SolverResult::Sat);
    }

    #[test]
    fn test_push_pop_simple() {
        let mut solver = Solver::new();

        let x0 = solver.new_var();

        // Should be SAT (x0 can be true or false)
        assert_eq!(solver.solve(), SolverResult::Sat);

        // Push and add unit clause: x0
        solver.push();
        solver.add_clause([Lit::pos(x0)]);
        assert_eq!(solver.solve(), SolverResult::Sat);
        assert!(solver.model_value(x0).is_true());

        // Pop - should be SAT again
        solver.pop();
        let result = solver.solve();
        assert_eq!(
            result,
            SolverResult::Sat,
            "After pop, expected SAT but got {:?}. trivially_unsat={}",
            result,
            solver.trivially_unsat
        );
    }

    #[test]
    fn test_push_pop_incremental() {
        let mut solver = Solver::new();

        let x0 = solver.new_var();
        let x1 = solver.new_var();
        let x2 = solver.new_var();

        // Base level: x0 \/ x1
        solver.add_clause([Lit::pos(x0), Lit::pos(x1)]);
        assert_eq!(solver.solve(), SolverResult::Sat);

        // Push and add: ~x0
        solver.push();
        solver.add_clause([Lit::neg(x0)]);
        assert_eq!(solver.solve(), SolverResult::Sat);
        // x1 must be true
        assert!(solver.model_value(x1).is_true());

        // Push again and add: ~x1 (should be UNSAT)
        solver.push();
        solver.add_clause([Lit::neg(x1)]);
        assert_eq!(solver.solve(), SolverResult::Unsat);

        // Pop back one level (remove ~x1, keep ~x0)
        solver.pop();
        assert_eq!(solver.solve(), SolverResult::Sat);
        assert!(solver.model_value(x1).is_true());

        // Pop back to base level (remove ~x0)
        solver.pop();
        assert_eq!(solver.solve(), SolverResult::Sat);
        // Either x0 or x1 can be true now

        // Push and add different clause: x0 /\ x2
        solver.push();
        solver.add_clause([Lit::pos(x0)]);
        solver.add_clause([Lit::pos(x2)]);
        assert_eq!(solver.solve(), SolverResult::Sat);
        assert!(solver.model_value(x0).is_true());
        assert!(solver.model_value(x2).is_true());

        // Pop and verify clauses are removed
        solver.pop();
        assert_eq!(solver.solve(), SolverResult::Sat);
    }

    #[test]
    fn test_push_pop_with_learned_clauses() {
        let mut solver = Solver::new();

        let x0 = solver.new_var();
        let x1 = solver.new_var();
        let x2 = solver.new_var();

        // Create a formula that will cause learning
        // (x0 \/ x1) /\ (~x0 \/ x2) /\ (~x1 \/ x2)
        solver.add_clause([Lit::pos(x0), Lit::pos(x1)]);
        solver.add_clause([Lit::neg(x0), Lit::pos(x2)]);
        solver.add_clause([Lit::neg(x1), Lit::pos(x2)]);

        assert_eq!(solver.solve(), SolverResult::Sat);

        // Push and add conflicting clause
        solver.push();
        solver.add_clause([Lit::neg(x2)]);

        // This should be UNSAT and cause clause learning
        assert_eq!(solver.solve(), SolverResult::Unsat);

        // Pop - learned clauses from this level should be removed
        solver.pop();

        // Should be SAT again
        assert_eq!(solver.solve(), SolverResult::Sat);
    }
}
