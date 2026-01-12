//! CDCL SAT Solver

use crate::chb::CHB;
use crate::chrono::ChronoBacktrack;
use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{LBool, Lit, Var};
use crate::lrb::LRB;
use crate::trail::{Reason, Trail};
use crate::vsids::VSIDS;
use crate::watched::{WatchLists, Watcher};
use smallvec::SmallVec;

/// Binary implication graph for efficient binary clause propagation
/// For each literal L, stores the list of literals that are implied when L is false
/// (i.e., for binary clause (~L v M), when L is assigned false, M must be true)
#[derive(Debug, Clone)]
struct BinaryImplicationGraph {
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
    config: SolverConfig,
    /// Number of variables
    num_vars: usize,
    /// Clause database
    clauses: ClauseDatabase,
    /// Assignment trail
    trail: Trail,
    /// Watch lists
    watches: WatchLists,
    /// VSIDS branching heuristic
    vsids: VSIDS,
    /// CHB branching heuristic
    chb: CHB,
    /// LRB branching heuristic
    lrb: LRB,
    /// Statistics
    stats: SolverStats,
    /// Learnt clause for conflict analysis
    learnt: SmallVec<[Lit; 16]>,
    /// Seen flags for conflict analysis
    seen: Vec<bool>,
    /// Analyze stack
    analyze_stack: Vec<Lit>,
    /// Current restart threshold
    restart_threshold: u64,
    /// Assertions stack for incremental solving (number of original clauses)
    assertion_levels: Vec<usize>,
    /// Trail sizes at each assertion level (for proper pop backtracking)
    assertion_trail_sizes: Vec<usize>,
    /// Clause IDs added at each assertion level (for proper pop)
    assertion_clause_ids: Vec<Vec<ClauseId>>,
    /// Model (if sat)
    model: Vec<LBool>,
    /// Whether formula is trivially unsatisfiable
    trivially_unsat: bool,
    /// Phase saving: last polarity assigned to each variable
    phase: Vec<bool>,
    /// Luby sequence index for restarts
    luby_index: u64,
    /// Level marks for LBD computation
    level_marks: Vec<u32>,
    /// Current mark counter for LBD computation
    lbd_mark: u32,
    /// Learned clause IDs for deletion
    learned_clause_ids: Vec<ClauseId>,
    /// Number of conflicts since last clause deletion
    conflicts_since_deletion: u64,
    /// PRNG state (xorshift64)
    rng_state: u64,
    /// For Glucose-style restarts: average LBD of recent conflicts
    recent_lbd_sum: u64,
    /// Number of conflicts contributing to recent_lbd_sum
    recent_lbd_count: u64,
    /// Binary implication graph for fast binary clause propagation
    binary_graph: BinaryImplicationGraph,
    /// Global average LBD for local restarts
    global_lbd_sum: u64,
    /// Number of conflicts contributing to global LBD
    global_lbd_count: u64,
    /// Conflicts since last local restart
    conflicts_since_local_restart: u64,
    /// Conflicts since last inprocessing
    conflicts_since_inprocessing: u64,
    /// Chronological backtracking helper
    chrono_backtrack: ChronoBacktrack,
    /// Clause activity bump increment (for MapleSAT-style clause bumping)
    clause_bump_increment: f64,
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
                // Unit clause - enqueue
                let lit = clause_lits[0];
                if self.trail.lit_value(lit).is_false() {
                    self.trivially_unsat = true;
                    return false;
                }
                if !self.trail.is_assigned(lit.var()) {
                    self.trail.assign_decision(lit);
                }
                return true;
            }
            2 => {
                // Binary clause - add to binary implication graph
                let clause_id = self.clauses.add_original(clause_lits.iter().copied());

                // Track clause for incremental solving
                if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
                    current_level_clauses.push(clause_id);
                }

                let lit0 = clause_lits[0];
                let lit1 = clause_lits[1];

                // (~lit0 => lit1) and (~lit1 => lit0)
                self.binary_graph.add(lit0.negate(), lit1, clause_id);
                self.binary_graph.add(lit1.negate(), lit0, clause_id);

                // Also set up watches for consistency
                self.watches
                    .add(lit0.negate(), Watcher::new(clause_id, lit1));
                self.watches
                    .add(lit1.negate(), Watcher::new(clause_id, lit0));

                return true;
            }
            _ => {}
        }

        // Add clause (3+ literals)
        let clause_id = self.clauses.add_original(clause_lits.iter().copied());

        // Track clause for incremental solving
        if let Some(current_level_clauses) = self.assertion_clause_ids.last_mut() {
            current_level_clauses.push(clause_id);
        }

        // Set up watches
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

    /// Extract a core of assumptions that caused a conflict
    fn extract_assumption_core(&self, assumptions: &[Lit], conflict_idx: usize) -> Vec<Lit> {
        // The conflicting assumption and any assumptions it depends on
        let mut core = Vec::new();
        let conflict_lit = assumptions[conflict_idx];

        // Find assumptions that led to this conflict
        for &lit in &assumptions[..=conflict_idx] {
            if self.seen.get(lit.var().index()).copied().unwrap_or(false) || lit == conflict_lit {
                core.push(lit);
            }
        }

        // If core is empty, just return the conflicting assumption
        if core.is_empty() {
            core.push(conflict_lit);
        }

        core
    }

    /// Analyze conflict to find assumptions in the unsat core
    fn analyze_assumption_conflict(&mut self, assumptions: &[Lit]) -> Vec<Lit> {
        // Use seen flags to mark which assumptions are in the conflict
        let mut core = Vec::new();

        // Walk back through the trail to find conflicting assumptions
        for &lit in assumptions {
            let var = lit.var();
            if var.index() < self.trail.assignments().len() {
                let value = self.trail.lit_value(lit);
                // If the negation of an assumption is implied, it's in the core
                if value.is_false() || self.seen.get(var.index()).copied().unwrap_or(false) {
                    core.push(lit);
                }
            }
        }

        // If no specific core found, return all assumptions
        if core.is_empty() {
            core.extend(assumptions.iter().copied());
        }

        core
    }

    /// Get the minimum backtrack level for a conflict
    fn analyze_conflict_level(&self, conflict: ClauseId) -> u32 {
        let clause = match self.clauses.get(conflict) {
            Some(c) => c,
            None => return 0,
        };

        let mut min_level = u32::MAX;
        for lit in clause.lits.iter().copied() {
            let level = self.trail.level(lit.var());
            if level > 0 && level < min_level {
                min_level = level;
            }
        }

        if min_level == u32::MAX { 0 } else { min_level }
    }

    /// Handle clause deletion and restart, but don't backtrack past assumptions
    fn handle_clause_deletion_and_restart_limited(&mut self, min_level: u32) {
        self.conflicts_since_deletion += 1;

        if self.conflicts_since_deletion >= self.config.clause_deletion_threshold as u64 {
            self.reduce_clause_database();
            self.conflicts_since_deletion = 0;
        }

        if self.stats.conflicts >= self.restart_threshold {
            // Limited restart - don't backtrack past assumptions
            self.backtrack(min_level);
            self.stats.restarts += 1;
            self.luby_index += 1;
            self.restart_threshold =
                self.stats.conflicts + self.config.restart_interval * Self::luby(self.luby_index);
        }
    }

    /// Unit propagation using two-watched literals
    fn propagate(&mut self) -> Option<ClauseId> {
        while let Some(lit) = self.trail.next_to_propagate() {
            self.stats.propagations += 1;

            // First, propagate binary implications (faster)
            let binary_implications = self.binary_graph.get(lit).to_vec();
            for &(implied_lit, clause_id) in &binary_implications {
                let value = self.trail.lit_value(implied_lit);
                if value.is_false() {
                    // Conflict in binary clause
                    return Some(clause_id);
                } else if !value.is_defined() {
                    // Propagate
                    self.trail.assign_propagation(implied_lit, clause_id);

                    // Lazy hyper-binary resolution: check if we can learn a binary clause
                    if self.config.enable_lazy_hyper_binary {
                        self.check_hyper_binary_resolution(lit, implied_lit, clause_id);
                    }
                }
            }

            // Get watches for the negation of the propagated literal
            let watches = std::mem::take(self.watches.get_mut(lit));

            let mut i = 0;
            while i < watches.len() {
                let watcher = watches[i];

                // Check blocker
                if self.trail.lit_value(watcher.blocker).is_true() {
                    i += 1;
                    continue;
                }

                let clause = match self.clauses.get_mut(watcher.clause) {
                    Some(c) if !c.deleted => c,
                    _ => {
                        i += 1;
                        continue;
                    }
                };

                // Make sure the false literal is at position 1
                if clause.lits[0] == lit.negate() {
                    clause.swap(0, 1);
                }

                // If first watch is true, clause is satisfied
                let first = clause.lits[0];
                if self.trail.lit_value(first).is_true() {
                    // Update blocker
                    self.watches
                        .get_mut(lit)
                        .push(Watcher::new(watcher.clause, first));
                    i += 1;
                    continue;
                }

                // Look for a new watch
                let mut found = false;
                for j in 2..clause.lits.len() {
                    let l = clause.lits[j];
                    if !self.trail.lit_value(l).is_false() {
                        clause.swap(1, j);
                        self.watches
                            .add(clause.lits[1].negate(), Watcher::new(watcher.clause, first));
                        found = true;
                        break;
                    }
                }

                if found {
                    i += 1;
                    continue;
                }

                // No new watch found - clause is unit or conflicting
                self.watches
                    .get_mut(lit)
                    .push(Watcher::new(watcher.clause, first));

                if self.trail.lit_value(first).is_false() {
                    // Conflict
                    // Put remaining watches back
                    for j in (i + 1)..watches.len() {
                        self.watches.get_mut(lit).push(watches[j]);
                    }
                    return Some(watcher.clause);
                } else {
                    // Unit propagation
                    self.trail.assign_propagation(first, watcher.clause);

                    // Lazy hyper-binary resolution
                    if self.config.enable_lazy_hyper_binary {
                        self.check_hyper_binary_resolution(lit, first, watcher.clause);
                    }
                }

                i += 1;
            }
        }

        None
    }

    /// Analyze conflict and learn clause
    fn analyze(&mut self, conflict: ClauseId) -> (u32, SmallVec<[Lit; 16]>) {
        self.learnt.clear();
        self.learnt.push(Lit::from_code(0)); // Placeholder for asserting literal

        let mut counter = 0;
        let mut p = None;
        let mut index = self.trail.assignments().len();
        let current_level = self.trail.decision_level();

        // Reset seen flags
        for s in &mut self.seen {
            *s = false;
        }

        let mut reason_clause = conflict;

        loop {
            // Process reason clause (must exist, as it's either conflict or a propagation reason)
            let Some(clause) = self.clauses.get(reason_clause) else {
                break; // Should not happen in valid state
            };
            let start = if p.is_some() { 1 } else { 0 };
            let is_learned = clause.learned;

            // Record clause usage for tier promotion and bump activity (if it's a learned clause)
            if is_learned && let Some(clause_mut) = self.clauses.get_mut(reason_clause) {
                clause_mut.record_usage();
                // Promote to Core if LBD ≤ 2 (GLUE clause)
                if clause_mut.lbd <= 2 {
                    clause_mut.promote_to_core();
                }
                // Bump clause activity (MapleSAT-style)
                clause_mut.activity += self.clause_bump_increment;
            }

            let Some(clause) = self.clauses.get(reason_clause) else {
                break;
            };
            for &lit in &clause.lits[start..] {
                let var = lit.var();
                let level = self.trail.level(var);

                if !self.seen[var.index()] && level > 0 {
                    self.seen[var.index()] = true;
                    self.vsids.bump(var);
                    self.chb.bump(var);
                    self.lrb.on_reason(var);

                    if level == current_level {
                        counter += 1;
                    } else {
                        self.learnt.push(lit.negate());
                    }
                }
            }

            // Find next literal to analyze
            let mut current_lit;
            loop {
                index -= 1;
                current_lit = self.trail.assignments()[index];
                p = Some(current_lit);
                if self.seen[current_lit.var().index()] {
                    break;
                }
            }

            counter -= 1;
            if counter == 0 {
                break;
            }

            let var = current_lit.var();
            match self.trail.reason(var) {
                Reason::Propagation(c) => reason_clause = c,
                _ => break,
            }
        }

        // Set asserting literal (p is guaranteed to be Some at this point)
        if let Some(lit) = p {
            self.learnt[0] = lit.negate();
        }

        // Minimize learnt clause using recursive resolution
        self.minimize_learnt_clause();

        // Calculate assertion level (traditional backtrack level)
        let assertion_level = if self.learnt.len() == 1 {
            0
        } else {
            // Find second highest level
            let mut max_level = 0;
            let mut max_idx = 1;
            for (i, &lit) in self.learnt.iter().enumerate().skip(1) {
                let level = self.trail.level(lit.var());
                if level > max_level {
                    max_level = level;
                    max_idx = i;
                }
            }
            // Move second watch to position 1
            self.learnt.swap(1, max_idx);
            max_level
        };

        // Apply chronological backtracking if enabled
        let backtrack_level = self.chrono_backtrack.compute_backtrack_level(
            &self.trail,
            &self.learnt,
            assertion_level,
        );

        // Track chronological vs non-chronological backtracks
        if backtrack_level != assertion_level {
            self.stats.chrono_backtracks += 1;
        } else {
            self.stats.non_chrono_backtracks += 1;
        }

        (backtrack_level, self.learnt.clone())
    }

    /// Pick next variable to branch on
    fn pick_branch_var(&mut self) -> Option<Var> {
        if self.config.use_lrb_branching {
            // Use LRB branching
            while let Some(var) = self.lrb.select() {
                if !self.trail.is_assigned(var) {
                    self.lrb.on_assign(var);
                    return Some(var);
                }
            }
        } else if self.config.use_chb_branching {
            // Use CHB branching
            // Rebuild heap periodically to reflect score changes
            if self.stats.decisions.is_multiple_of(100) {
                self.chb.rebuild_heap();
            }

            while let Some(var) = self.chb.pop_max() {
                if !self.trail.is_assigned(var) {
                    return Some(var);
                }
            }
        } else {
            // Use VSIDS branching
            while let Some(var) = self.vsids.pop_max() {
                if !self.trail.is_assigned(var) {
                    return Some(var);
                }
            }
        }
        None
    }

    /// Minimize the learned clause by removing redundant literals
    ///
    /// A literal can be removed if it is implied by the remaining literals.
    /// We use a recursive check: a literal l is redundant if its reason clause
    /// contains only literals that are either:
    /// - Already in the learnt clause (marked as seen)
    /// - At decision level 0 (always true in the learned clause context)
    /// - Themselves redundant (recursive check)
    ///
    /// This also performs clause strengthening by checking for stronger implications
    fn minimize_learnt_clause(&mut self) {
        if self.learnt.len() <= 2 {
            // Don't minimize very small clauses
            return;
        }

        let original_len = self.learnt.len();

        // Mark all literals in the learned clause as "in clause"
        // We use analyze_stack to track literals to check
        self.analyze_stack.clear();

        // Phase 1: Basic minimization - remove redundant literals
        let mut j = 1; // Write position
        for i in 1..self.learnt.len() {
            let lit = self.learnt[i];
            if self.lit_is_redundant(lit) {
                // Skip this literal (it's redundant)
            } else {
                // Keep this literal
                self.learnt[j] = lit;
                j += 1;
            }
        }
        self.learnt.truncate(j);

        // Phase 2: Clause strengthening - check for self-subsuming resolution
        // If the clause contains both l and ~l' where l' is in a reason clause,
        // we might be able to strengthen the clause
        self.strengthen_learnt_clause();

        // Track minimization statistics
        let final_len = self.learnt.len();
        if final_len < original_len {
            self.stats.minimizations += 1;
            self.stats.literals_removed += (original_len - final_len) as u64;
        }
    }

    /// Strengthen the learned clause using on-the-fly self-subsuming resolution
    fn strengthen_learnt_clause(&mut self) {
        if self.learnt.len() <= 2 {
            return;
        }

        // Check each literal to see if we can strengthen by resolution
        let mut j = 1;
        for i in 1..self.learnt.len() {
            let lit = self.learnt[i];
            let var = lit.var();

            // Check if this literal can be strengthened
            if let Reason::Propagation(reason_id) = self.trail.reason(var)
                && let Some(reason_clause) = self.clauses.get(reason_id)
                && reason_clause.lits.len() == 2
            {
                // Binary reason: one literal is lit, the other is the implied literal
                let other_lit = if reason_clause.lits[0] == lit.negate() {
                    reason_clause.lits[1]
                } else if reason_clause.lits[1] == lit.negate() {
                    reason_clause.lits[0]
                } else {
                    // Keep the literal
                    self.learnt[j] = lit;
                    j += 1;
                    continue;
                };

                // If other_lit is already in the learned clause at level 0,
                // we can remove lit
                if self.trail.level(other_lit.var()) == 0 && self.seen[other_lit.var().index()] {
                    // Skip this literal (strengthened)
                    continue;
                }
            }

            // Keep this literal
            self.learnt[j] = lit;
            j += 1;
        }
        self.learnt.truncate(j);
    }

    /// Check if a literal is redundant in the learned clause
    ///
    /// A literal is redundant if its reason clause only contains:
    /// - Literals marked as seen (in the learned clause)
    /// - Literals at decision level 0
    /// - Literals that are themselves redundant (recursive)
    fn lit_is_redundant(&mut self, lit: Lit) -> bool {
        let var = lit.var();

        // Decision variables and theory propagations are not redundant
        let reason = match self.trail.reason(var) {
            Reason::Decision => return false,
            Reason::Theory => return false, // Theory propagations can't be minimized
            Reason::Propagation(c) => c,
        };

        let reason_clause = match self.clauses.get(reason) {
            Some(c) => c,
            None => return false,
        };

        // Check all literals in the reason clause
        for &reason_lit in &reason_clause.lits {
            if reason_lit == lit.negate() {
                // Skip the literal we're analyzing
                continue;
            }

            let reason_var = reason_lit.var();

            // Level 0 literals are always OK
            if self.trail.level(reason_var) == 0 {
                continue;
            }

            // If the literal is in the learned clause (seen), it's OK
            if self.seen[reason_var.index()] {
                continue;
            }

            // Otherwise, this literal prevents minimization
            // (A full recursive check would be more powerful but more expensive)
            return false;
        }

        true
    }

    /// Backtrack with phase saving
    fn backtrack_with_phase_saving(&mut self, level: u32) {
        // Save phases before backtracking
        let phase = &mut self.phase;
        let lrb = &mut self.lrb;
        self.trail.backtrack_to_with_callback(level, |lit| {
            let var = lit.var();
            if var.index() < phase.len() {
                phase[var.index()] = lit.is_pos();
            }
            // Re-insert variable into LRB heap
            lrb.unassign(var);
        });
    }

    /// Backtrack to a given level without saving phases
    fn backtrack(&mut self, level: u32) {
        self.trail.backtrack_to(level);
    }

    /// Compute the Luby sequence value for index i (1-indexed: luby(1)=1, luby(2)=1, ...)
    /// Sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    /// For 0-indexed input, we add 1 internally.
    fn luby(i: u64) -> u64 {
        let i = i + 1; // Convert to 1-indexed

        // Find k such that 2^k - 1 >= i
        let mut k = 1u32;
        while (1u64 << k) - 1 < i {
            k += 1;
        }

        let seq_len = (1u64 << k) - 1;

        if i == seq_len {
            // i is exactly 2^k - 1, return 2^(k-1)
            1u64 << (k - 1)
        } else {
            // Recurse: luby(i) = luby(i - (2^(k-1) - 1))
            // The sequence up to 2^k - 1 is: luby(1..2^(k-1)-1), luby(1..2^(k-1)-1), 2^(k-1)
            let half_len = (1u64 << (k - 1)) - 1;
            if i <= half_len {
                Self::luby(i - 1) // Already 0-indexed internally
            } else if i <= 2 * half_len {
                Self::luby(i - half_len - 1)
            } else {
                1u64 << (k - 1)
            }
        }
    }

    /// Restart
    fn restart(&mut self) {
        self.stats.restarts += 1;
        self.backtrack_with_phase_saving(0);

        // Calculate next restart threshold based on strategy
        match self.config.restart_strategy {
            RestartStrategy::Luby => {
                self.luby_index += 1;
                self.restart_threshold = self.stats.conflicts
                    + Self::luby(self.luby_index) * self.config.restart_interval;
            }
            RestartStrategy::Geometric => {
                let current_interval = if self.restart_threshold > self.stats.conflicts {
                    self.restart_threshold - self.stats.conflicts
                } else {
                    self.config.restart_interval
                };
                let next_interval =
                    (current_interval as f64 * self.config.restart_multiplier) as u64;
                self.restart_threshold = self.stats.conflicts + next_interval;
            }
            RestartStrategy::Glucose => {
                // Glucose-style dynamic restarts based on LBD
                // Restart when recent average LBD is higher than global average
                // For now, use geometric with dynamic adjustment
                let current_interval = if self.restart_threshold > self.stats.conflicts {
                    self.restart_threshold - self.stats.conflicts
                } else {
                    self.config.restart_interval
                };

                // Adjust based on recent LBD trend
                let next_interval = if self.recent_lbd_count > 50 {
                    let recent_avg = self.recent_lbd_sum / self.recent_lbd_count.max(1);
                    // If recent LBD is low (good), increase interval; if high, decrease
                    if recent_avg < 5 {
                        // Good quality clauses - increase interval
                        ((current_interval as f64) * 1.1) as u64
                    } else {
                        // Poor quality clauses - decrease interval
                        ((current_interval as f64) * 0.9) as u64
                    }
                } else {
                    current_interval
                };

                self.restart_threshold = self.stats.conflicts + next_interval.max(100);
            }
            RestartStrategy::LocalLbd => {
                // Local restarts based on LBD
                // Check if we should do a local restart
                self.conflicts_since_local_restart += 1;

                if self.conflicts_since_local_restart >= 50 && self.should_local_restart() {
                    // Perform local restart - backtrack to a safe level, not to 0
                    let local_level = self.compute_local_restart_level();
                    self.backtrack_with_phase_saving(local_level);
                    self.conflicts_since_local_restart = 0;
                    // Reset recent LBD for next window
                    self.recent_lbd_sum = 0;
                    self.recent_lbd_count = 0;
                } else {
                    // Standard restart if too many conflicts
                    let current_interval = if self.restart_threshold > self.stats.conflicts {
                        self.restart_threshold - self.stats.conflicts
                    } else {
                        self.config.restart_interval
                    };
                    self.restart_threshold = self.stats.conflicts + current_interval;
                }
                return; // Don't do full backtrack to 0
            }
        }

        // Re-add all unassigned variables to VSIDS heap
        for i in 0..self.num_vars {
            let var = Var::new(i as u32);
            if !self.trail.is_assigned(var) && !self.vsids.contains(var) {
                self.vsids.insert(var);
            }
        }
    }

    /// Check if we should perform a local restart
    /// Returns true if recent average LBD is significantly higher than global average
    fn should_local_restart(&self) -> bool {
        if self.recent_lbd_count < 50 || self.global_lbd_count < 100 {
            return false;
        }

        let recent_avg = self.recent_lbd_sum / self.recent_lbd_count.max(1);
        let global_avg = self.global_lbd_sum / self.global_lbd_count.max(1);

        // Local restart if recent average is 1.25x higher than global average
        recent_avg * 4 > global_avg * 5
    }

    /// Compute the level to backtrack to for local restart
    /// Use a level that preserves some of the search progress
    fn compute_local_restart_level(&self) -> u32 {
        let current_level = self.trail.decision_level();

        // Backtrack to about 20% of current depth to preserve some work
        if current_level > 5 {
            current_level / 5
        } else {
            0
        }
    }

    /// Compute LBD (Literal Block Distance) of a clause
    /// LBD is the number of distinct decision levels in the clause
    fn compute_lbd(&mut self, lits: &[Lit]) -> u32 {
        self.lbd_mark += 1;
        let mark = self.lbd_mark;

        let mut count = 0u32;
        for &lit in lits {
            let level = self.trail.level(lit.var()) as usize;
            if level < self.level_marks.len() && self.level_marks[level] != mark {
                self.level_marks[level] = mark;
                count += 1;
            }
        }

        count
    }

    /// Reduce the learned clause database using tier-based deletion strategy
    /// - Core tier (Tier 1): Rarely deleted, only if very inactive
    /// - Mid tier (Tier 2): Delete ~30% based on activity
    /// - Local tier (Tier 3): Delete ~75% based on activity
    fn reduce_clause_database(&mut self) {
        use crate::clause::ClauseTier;

        let mut core_candidates: Vec<(ClauseId, f64)> = Vec::new();
        let mut mid_candidates: Vec<(ClauseId, f64)> = Vec::new();
        let mut local_candidates: Vec<(ClauseId, f64)> = Vec::new();

        for &cid in &self.learned_clause_ids {
            if let Some(clause) = self.clauses.get(cid) {
                if clause.deleted {
                    continue;
                }

                // Don't delete binary clauses (very useful)
                if clause.lits.len() <= 2 {
                    continue;
                }

                // Check if clause is currently a reason for any assignment
                // (We can't delete reason clauses)
                let is_reason = clause.lits.iter().any(|&lit| {
                    let var = lit.var();
                    if self.trail.is_assigned(var) {
                        matches!(self.trail.reason(var), Reason::Propagation(r) if r == cid)
                    } else {
                        false
                    }
                });

                if !is_reason {
                    match clause.tier {
                        ClauseTier::Core => core_candidates.push((cid, clause.activity)),
                        ClauseTier::Mid => mid_candidates.push((cid, clause.activity)),
                        ClauseTier::Local => local_candidates.push((cid, clause.activity)),
                    }
                }
            }
        }

        // Sort by activity (ascending) - delete low-activity clauses first
        core_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        mid_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        local_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Delete different percentages from each tier
        // Core: Delete bottom 10% (very conservative)
        let num_core_delete = core_candidates.len() / 10;
        // Mid: Delete bottom 30%
        let num_mid_delete = (mid_candidates.len() * 3) / 10;
        // Local: Delete bottom 75% (very aggressive)
        let num_local_delete = (local_candidates.len() * 3) / 4;

        for (cid, _) in core_candidates.iter().take(num_core_delete) {
            self.clauses.remove(*cid);
            self.stats.deleted_clauses += 1;
        }

        for (cid, _) in mid_candidates.iter().take(num_mid_delete) {
            self.clauses.remove(*cid);
            self.stats.deleted_clauses += 1;
        }

        for (cid, _) in local_candidates.iter().take(num_local_delete) {
            self.clauses.remove(*cid);
            self.stats.deleted_clauses += 1;
        }

        // Clean up learned_clause_ids (remove deleted clauses)
        self.learned_clause_ids
            .retain(|&cid| self.clauses.get(cid).is_some_and(|c| !c.deleted));
    }

    /// Save the model
    fn save_model(&mut self) {
        self.model.resize(self.num_vars, LBool::Undef);
        for i in 0..self.num_vars {
            self.model[i] = self.trail.value(Var::new(i as u32));
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
        self.trail.backtrack_to(0);

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
            self.trail.backtrack_to_size(trail_size);

            // Ensure we're at decision level 0
            self.trail.backtrack_to(0);

            // Clear the trivially_unsat flag as we've removed problematic clauses
            self.trivially_unsat = false;
        }
    }

    /// Backtrack to decision level 0 (for AllSAT enumeration)
    ///
    /// This is necessary after a SAT result before adding blocking clauses
    /// to ensure the new clauses can trigger propagation correctly.
    pub fn backtrack_to_root(&mut self) {
        self.trail.backtrack_to(0);
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
                self.learn_clause(learnt_clause);

                self.vsids.decay();
                self.clauses.decay_activity(self.config.clause_decay);
                self.handle_clause_deletion_and_restart();
                continue;
            }

            // Theory propagation check after each assignment
            loop {
                // Get newly assigned literals and notify theory
                let assignments = self.trail.assignments().to_vec();
                let mut theory_conflict = None;
                let mut theory_propagations = Vec::new();

                // Check each new assignment with theory
                for &lit in &assignments {
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

                // Handle theory conflict
                if let Some(conflict_lits) = theory_conflict {
                    self.stats.conflicts += 1;

                    if self.trail.decision_level() == 0 {
                        return SolverResult::Unsat;
                    }

                    let (backtrack_level, learnt_clause) =
                        self.analyze_theory_conflict(&conflict_lits);
                    theory.on_backtrack(backtrack_level);
                    self.backtrack_with_phase_saving(backtrack_level);
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
                        theory.on_backtrack(backtrack_level);
                        self.backtrack_with_phase_saving(backtrack_level);
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

    /// Analyze a theory conflict (given as a list of literals that are all false)
    fn analyze_theory_conflict(&mut self, conflict_lits: &[Lit]) -> (u32, SmallVec<[Lit; 16]>) {
        self.learnt.clear();
        self.learnt.push(Lit::from_code(0)); // Placeholder

        let mut counter = 0;
        let current_level = self.trail.decision_level();

        // Reset seen flags
        for s in &mut self.seen {
            *s = false;
        }

        // Process conflict literals
        for &lit in conflict_lits {
            let var = lit.var();
            let level = self.trail.level(var);

            if !self.seen[var.index()] && level > 0 {
                self.seen[var.index()] = true;
                self.vsids.bump(var);
                self.chb.bump(var);

                if level == current_level {
                    counter += 1;
                } else {
                    self.learnt.push(lit.negate());
                }
            }
        }

        // Find UIP by walking back through trail
        let mut index = self.trail.assignments().len();
        let mut p = None;

        while counter > 0 {
            index -= 1;
            let current_lit = self.trail.assignments()[index];
            p = Some(current_lit);
            let var = current_lit.var();

            if self.seen[var.index()] {
                counter -= 1;

                if counter > 0
                    && let Reason::Propagation(reason_clause) = self.trail.reason(var)
                    && let Some(clause) = self.clauses.get(reason_clause)
                {
                    // Get reason and process its literals
                    for &lit in &clause.lits[1..] {
                        let reason_var = lit.var();
                        let level = self.trail.level(reason_var);

                        if !self.seen[reason_var.index()] && level > 0 {
                            self.seen[reason_var.index()] = true;
                            self.vsids.bump(reason_var);
                            self.chb.bump(reason_var);

                            if level == current_level {
                                counter += 1;
                            } else {
                                self.learnt.push(lit.negate());
                            }
                        }
                    }
                }
            }
        }

        // Set asserting literal
        if let Some(uip) = p {
            self.learnt[0] = uip.negate();
        }

        // Minimize
        self.minimize_learnt_clause();

        // Calculate backtrack level
        let backtrack_level = if self.learnt.len() == 1 {
            0
        } else {
            let mut max_level = 0;
            let mut max_idx = 1;
            for (i, &lit) in self.learnt.iter().enumerate().skip(1) {
                let level = self.trail.level(lit.var());
                if level > max_level {
                    max_level = level;
                    max_idx = i;
                }
            }
            self.learnt.swap(1, max_idx);
            max_level
        };

        (backtrack_level, self.learnt.clone())
    }

    /// Add a theory reason clause
    /// The clause is: reason_lits[0] OR reason_lits[1] OR ... OR propagated_lit
    fn add_theory_reason_clause(&mut self, reason_lits: &[Lit], propagated_lit: Lit) -> ClauseId {
        let mut clause_lits: SmallVec<[Lit; 8]> = SmallVec::new();
        clause_lits.push(propagated_lit);
        for &lit in reason_lits {
            clause_lits.push(lit.negate());
        }

        let clause_id = self.clauses.add_learned(clause_lits.iter().copied());

        // Set up watches
        if clause_lits.len() >= 2 {
            let lit0 = clause_lits[0];
            let lit1 = clause_lits[1];
            self.watches
                .add(lit0.negate(), Watcher::new(clause_id, lit1));
            self.watches
                .add(lit1.negate(), Watcher::new(clause_id, lit0));
        }

        clause_id
    }

    /// Learn a clause and set up watches
    /// Includes on-the-fly subsumption check
    fn learn_clause(&mut self, learnt_clause: SmallVec<[Lit; 16]>) {
        if learnt_clause.len() == 1 {
            self.trail.assign_decision(learnt_clause[0]);
            self.stats.unit_clauses += 1;
            self.stats.learned_clauses += 1;
        } else if learnt_clause.len() == 2 {
            // Binary learned clause - add to binary implication graph
            let lbd = self.compute_lbd(&learnt_clause);
            let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
            self.stats.learned_clauses += 1;
            self.stats.binary_clauses += 1;
            self.stats.total_lbd += lbd as u64;

            if let Some(clause) = self.clauses.get_mut(clause_id) {
                clause.lbd = lbd;
            }

            self.learned_clause_ids.push(clause_id);

            let lit0 = learnt_clause[0];
            let lit1 = learnt_clause[1];

            // Add to binary graph
            self.binary_graph.add(lit0.negate(), lit1, clause_id);
            self.binary_graph.add(lit1.negate(), lit0, clause_id);

            self.watches
                .add(lit0.negate(), Watcher::new(clause_id, lit1));
            self.watches
                .add(lit1.negate(), Watcher::new(clause_id, lit0));

            self.trail.assign_propagation(learnt_clause[0], clause_id);
        } else {
            let lbd = self.compute_lbd(&learnt_clause);
            self.stats.total_lbd += lbd as u64;
            let clause_id = self.clauses.add_learned(learnt_clause.iter().copied());
            self.stats.learned_clauses += 1;

            if let Some(clause) = self.clauses.get_mut(clause_id) {
                clause.lbd = lbd;
            }

            self.learned_clause_ids.push(clause_id);

            let lit0 = learnt_clause[0];
            let lit1 = learnt_clause[1];
            self.watches
                .add(lit0.negate(), Watcher::new(clause_id, lit1));
            self.watches
                .add(lit1.negate(), Watcher::new(clause_id, lit0));

            self.trail.assign_propagation(learnt_clause[0], clause_id);

            // On-the-fly subsumption: check if this new clause subsumes existing clauses
            if learnt_clause.len() <= 5 && lbd <= 3 {
                self.check_subsumption(clause_id);
            }
        }
    }

    /// Check if the given clause subsumes any existing clauses
    /// A clause C subsumes C' if all literals of C are in C'
    fn check_subsumption(&mut self, new_clause_id: ClauseId) {
        let new_clause = match self.clauses.get(new_clause_id) {
            Some(c) => c.lits.clone(),
            None => return,
        };

        if new_clause.len() > 10 {
            return; // Don't check subsumption for large clauses (too expensive)
        }

        // Check against learned clauses only
        let mut to_remove = Vec::new();
        for &cid in &self.learned_clause_ids {
            if cid == new_clause_id {
                continue;
            }

            if let Some(clause) = self.clauses.get(cid) {
                if clause.deleted || clause.lits.len() < new_clause.len() {
                    continue;
                }

                // Check if new_clause subsumes clause
                if new_clause.iter().all(|&lit| clause.lits.contains(&lit)) {
                    to_remove.push(cid);
                }
            }
        }

        // Remove subsumed clauses
        for cid in to_remove {
            self.clauses.remove(cid);
            self.stats.deleted_clauses += 1;
        }
    }

    /// Handle clause deletion check and restart check
    fn handle_clause_deletion_and_restart(&mut self) {
        self.conflicts_since_deletion += 1;

        if self.conflicts_since_deletion >= self.config.clause_deletion_threshold as u64 {
            self.reduce_clause_database();
            self.conflicts_since_deletion = 0;
        }

        if self.stats.conflicts >= self.restart_threshold {
            self.restart();
        }
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

    /// Generate a random u64 using xorshift64
    fn rand_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generate a random f64 in [0, 1)
    fn rand_f64(&mut self) -> f64 {
        const MAX: f64 = u64::MAX as f64;
        (self.rand_u64() as f64) / MAX
    }

    /// Generate a random boolean with given probability of being true
    fn rand_bool(&mut self, probability: f64) -> bool {
        self.rand_f64() < probability
    }

    /// Check for hyper-binary resolution opportunity
    /// When propagating `implied` due to `lit` being assigned, check if we can
    /// learn a binary clause by resolving the reason clauses
    fn check_hyper_binary_resolution(&mut self, _lit: Lit, implied: Lit, reason_id: ClauseId) {
        // Only check at higher decision levels to avoid overhead
        if self.trail.decision_level() < 2 {
            return;
        }

        // Get the reason clause
        let reason_clause = match self.clauses.get(reason_id) {
            Some(c) if c.lits.len() >= 2 && c.lits.len() <= 4 => c.lits.clone(),
            _ => return,
        };

        // Check if we can derive a binary clause
        // Look for literals in the reason clause that are assigned at the current level
        let current_level = self.trail.decision_level();
        let mut current_level_lits = SmallVec::<[Lit; 4]>::new();

        for &reason_lit in &reason_clause {
            if reason_lit != implied {
                let var = reason_lit.var();
                if self.trail.level(var) == current_level {
                    current_level_lits.push(reason_lit);
                }
            }
        }

        // If there's exactly one literal from the current level besides the implied one,
        // and all others are at level 0 or earlier, we might have a hyper-binary opportunity
        if current_level_lits.len() == 1 {
            let other_lit = current_level_lits[0];

            // Check if we can create a useful binary clause
            // We can learn: ~other_lit => implied (or equivalently: other_lit v implied)
            let binary_clause_lits = [other_lit.negate(), implied];

            // Check if this binary clause is new and useful
            if !self.has_binary_implication(other_lit, implied) {
                // Learn this binary clause on-the-fly
                let clause_id = self.clauses.add_learned(binary_clause_lits.iter().copied());
                self.binary_graph.add(other_lit, implied, clause_id);
                self.binary_graph
                    .add(implied.negate(), other_lit.negate(), clause_id);
                self.stats.learned_clauses += 1;
            }
        }
    }

    /// Check if a binary implication already exists
    fn has_binary_implication(&self, from_lit: Lit, to_lit: Lit) -> bool {
        self.binary_graph
            .get(from_lit)
            .iter()
            .any(|(lit, _)| *lit == to_lit)
    }

    /// Vivification: try to strengthen clauses by checking if some literals are redundant
    /// This is an inprocessing technique that should be called periodically
    fn vivify_clauses(&mut self) {
        if self.trail.decision_level() != 0 {
            return; // Only vivify at decision level 0
        }

        let mut vivified_count = 0;
        let max_vivifications = 100; // Limit to avoid too much overhead

        // Try to vivify some learned clauses
        let clause_ids: Vec<ClauseId> = self
            .learned_clause_ids
            .iter()
            .copied()
            .take(max_vivifications)
            .collect();

        for clause_id in clause_ids {
            if vivified_count >= max_vivifications {
                break;
            }

            let clause_lits = match self.clauses.get(clause_id) {
                Some(c) if !c.deleted && c.lits.len() > 2 => c.lits.clone(),
                _ => continue,
            };

            // Try to find redundant literals in the clause
            // Assign all literals except one to false and see if we can derive the last one
            for skip_idx in 0..clause_lits.len() {
                // Save current state
                let saved_level = self.trail.decision_level();

                // Assign all literals except skip_idx to false
                self.trail.new_decision_level();
                let mut conflict = false;

                for (i, &lit) in clause_lits.iter().enumerate() {
                    if i == skip_idx {
                        continue;
                    }

                    let value = self.trail.lit_value(lit);
                    if value.is_true() {
                        // Clause is already satisfied
                        conflict = false;
                        break;
                    } else if value.is_false() {
                        // Already false
                        continue;
                    } else {
                        // Assign to false
                        self.trail.assign_decision(lit.negate());

                        // Propagate
                        if self.propagate().is_some() {
                            conflict = true;
                            break;
                        }
                    }
                }

                // Backtrack
                self.backtrack(saved_level);

                if conflict
                    && let Some(clause) = self.clauses.get_mut(clause_id)
                    && clause.lits.len() > 2
                {
                    // The literal at skip_idx is implied by the rest
                    // We can remove it from the clause (vivification succeeded)
                    clause.lits.remove(skip_idx);
                    vivified_count += 1;
                    break; // Done with this clause
                }
            }
        }
    }

    /// Perform inprocessing (apply preprocessing during search)
    fn inprocess(&mut self) {
        use crate::preprocessing::Preprocessor;

        // Only inprocess at decision level 0
        if self.trail.decision_level() != 0 {
            return;
        }

        // Create preprocessor with current number of variables
        let mut preprocessor = Preprocessor::new(self.num_vars);

        // Apply lightweight preprocessing techniques
        let _pure_elim = preprocessor.pure_literal_elimination(&mut self.clauses);
        let _subsumption = preprocessor.subsumption_elimination(&mut self.clauses);

        // On-the-fly clause strengthening
        self.strengthen_clauses_inprocessing();

        // Rebuild watch lists for any modified clauses
        // This is a simplified approach - in a full implementation,
        // we would track which clauses were removed and update watches incrementally
    }

    /// On-the-fly clause strengthening during inprocessing
    ///
    /// Try to remove literals from clauses by checking if they're redundant.
    /// A literal is redundant if the clause is satisfied when it's assigned to false.
    fn strengthen_clauses_inprocessing(&mut self) {
        if self.trail.decision_level() != 0 {
            return;
        }

        let max_clauses_to_strengthen = 50; // Limit to avoid overhead
        let mut strengthened_count = 0;

        // Collect candidate clauses (learned clauses with LBD > 2)
        let mut candidates: Vec<(ClauseId, u32)> = Vec::new();

        for &clause_id in &self.learned_clause_ids {
            if let Some(clause) = self.clauses.get(clause_id)
                && !clause.deleted
                && clause.lits.len() > 3
                && clause.lbd > 2
            {
                candidates.push((clause_id, clause.lbd));
            }
        }

        // Sort by LBD (prioritize higher LBD clauses for strengthening)
        candidates.sort_by_key(|(_, lbd)| std::cmp::Reverse(*lbd));

        for (clause_id, _) in candidates.iter().take(max_clauses_to_strengthen) {
            if strengthened_count >= max_clauses_to_strengthen {
                break;
            }

            let clause_lits = match self.clauses.get(*clause_id) {
                Some(c) if !c.deleted && c.lits.len() > 3 => c.lits.clone(),
                _ => continue,
            };

            // Try to remove each literal by checking if the remaining clause is still valid
            let mut literals_to_remove = Vec::new();

            for (i, &lit) in clause_lits.iter().enumerate() {
                // Save current trail state
                let saved_level = self.trail.decision_level();

                // Try assigning this literal to false
                self.trail.new_decision_level();
                self.trail.assign_decision(lit.negate());

                // Propagate
                let conflict = self.propagate();

                // Backtrack
                self.backtrack(saved_level);

                if conflict.is_some() {
                    // Assigning this literal to false causes a conflict
                    // This means the rest of the clause implies this literal
                    // So this literal can potentially be removed (strengthening)

                    // But we need to be careful: only remove if the remaining clause
                    // is still non-tautological and non-empty
                    let mut remaining: Vec<Lit> = clause_lits
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &l)| l)
                        .collect();

                    // Check if remaining clause is still valid (at least 2 literals)
                    if remaining.len() >= 2 {
                        // Check it's not a tautology
                        remaining.sort_by_key(|l| l.code());
                        let mut is_tautology = false;
                        for k in 0..remaining.len() - 1 {
                            if remaining[k] == remaining[k + 1].negate() {
                                is_tautology = true;
                                break;
                            }
                        }

                        if !is_tautology {
                            literals_to_remove.push(i);
                            break; // Only remove one literal at a time
                        }
                    }
                }
            }

            // Apply strengthening if we found literals to remove
            if !literals_to_remove.is_empty() {
                // First, remove literals
                if let Some(clause) = self.clauses.get_mut(*clause_id) {
                    // Remove literals in reverse order to preserve indices
                    for &idx in literals_to_remove.iter().rev() {
                        if idx < clause.lits.len() {
                            clause.lits.remove(idx);
                        }
                    }
                }

                // Then, recompute LBD (after the mutable borrow ends)
                if let Some(clause) = self.clauses.get(*clause_id) {
                    let lits_clone = clause.lits.clone();
                    let new_lbd = self.compute_lbd(&lits_clone);

                    // Now update the LBD
                    if let Some(clause) = self.clauses.get_mut(*clause_id) {
                        clause.lbd = new_lbd;
                    }

                    strengthened_count += 1;
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

        // Verify we had some learning (and thus minimization opportunities)
        assert!(solver.stats().conflicts > 0 || solver.stats().learned_clauses > 0);
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
        let core = core.unwrap();
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
        let core = core.unwrap();
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
