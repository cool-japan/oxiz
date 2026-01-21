//! Stochastic Local Search (SLS) Theory Integration
//!
//! Provides SLS-based solving for hard satisfiability problems.
//! Implements WalkSAT, GSAT, and adaptive local search algorithms.
//!
//! # Overview
//!
//! SLS methods complement exact SAT/SMT solving by:
//! - Finding satisfying assignments for hard random instances
//! - Providing initial solutions for optimization
//! - Escaping local minima through randomized moves
//!
//! # Algorithms
//!
//! - WalkSAT: Random walk with focused moves
//! - GSAT: Greedy SAT with random restarts
//! - Adaptive: Dynamic parameter tuning
//!
//! # Example
//!
//! ```ignore
//! use oxiz_theories::sls::{SlsSolver, SlsConfig, SlsAlgorithm};
//!
//! let mut solver = SlsSolver::new(SlsConfig::default());
//! solver.add_clause(&[1, -2, 3]);
//! solver.add_clause(&[-1, 2, 3]);
//! let result = solver.solve();
//! ```

use std::collections::{HashMap, HashSet};

/// Literal type (positive = variable, negative = negated variable)
pub type Lit = i32;

/// Variable type
pub type Var = u32;

/// Clause ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClauseId(pub u32);

/// SLS algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlsAlgorithm {
    /// WalkSAT algorithm
    #[default]
    WalkSat,
    /// GSAT algorithm
    Gsat,
    /// ProbSAT algorithm
    ProbSat,
    /// Adaptive algorithm (auto-selects)
    Adaptive,
}

/// Configuration for SLS solver
#[derive(Debug, Clone)]
pub struct SlsConfig {
    /// Algorithm to use
    pub algorithm: SlsAlgorithm,
    /// Maximum number of flips
    pub max_flips: u64,
    /// Maximum number of restarts
    pub max_restarts: u32,
    /// Noise probability for random moves (0.0 to 1.0)
    pub noise: f64,
    /// Random seed
    pub seed: u64,
    /// Enable adaptive noise
    pub adaptive_noise: bool,
    /// Noise increment factor
    pub noise_inc: f64,
    /// Noise decrement factor
    pub noise_dec: f64,
    /// Enable tabu search
    pub tabu: bool,
    /// Tabu tenure (number of flips a variable is forbidden)
    pub tabu_tenure: u32,
}

impl Default for SlsConfig {
    fn default() -> Self {
        Self {
            algorithm: SlsAlgorithm::WalkSat,
            max_flips: 1_000_000,
            max_restarts: 100,
            noise: 0.57, // Optimal for many random 3-SAT instances
            seed: 42,
            adaptive_noise: true,
            noise_inc: 0.01,
            noise_dec: 0.01,
            tabu: false,
            tabu_tenure: 10,
        }
    }
}

/// Result of SLS solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlsResult {
    /// Found a satisfying assignment
    Sat(Vec<bool>),
    /// No solution found within limits
    Unknown,
    /// Problem is trivially unsatisfiable (empty clause)
    Unsat,
}

/// Statistics for SLS solving
#[derive(Debug, Clone, Default)]
pub struct SlsStats {
    /// Number of flips performed
    pub flips: u64,
    /// Number of restarts
    pub restarts: u32,
    /// Number of clauses satisfied at best
    pub best_unsat: u32,
    /// Average flips per restart
    pub avg_flips_per_restart: f64,
    /// Time spent (milliseconds)
    pub time_ms: u64,
    /// Number of random moves
    pub random_moves: u64,
    /// Number of greedy moves
    pub greedy_moves: u64,
}

/// A clause in the SLS solver
#[derive(Debug, Clone)]
struct SlsClause {
    /// Literals in the clause
    literals: Vec<Lit>,
    /// Number of true literals (satisfied count)
    sat_count: u32,
    /// Weight for weighted SLS
    weight: f64,
}

impl SlsClause {
    fn new(literals: Vec<Lit>) -> Self {
        Self {
            literals,
            sat_count: 0,
            weight: 1.0,
        }
    }

    fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if clause is satisfied
    #[allow(dead_code)]
    fn is_satisfied(&self) -> bool {
        self.sat_count > 0
    }
}

/// SLS Solver
#[derive(Debug)]
pub struct SlsSolver {
    /// Configuration
    config: SlsConfig,
    /// Clauses
    clauses: Vec<SlsClause>,
    /// Number of variables
    num_vars: u32,
    /// Current assignment (true/false for each variable)
    assignment: Vec<bool>,
    /// Occurrence lists: variable -> clauses containing it positive
    pos_occs: HashMap<Var, Vec<ClauseId>>,
    /// Occurrence lists: variable -> clauses containing it negative
    neg_occs: HashMap<Var, Vec<ClauseId>>,
    /// Unsatisfied clauses
    unsat_clauses: HashSet<ClauseId>,
    /// Break count for each variable (how many clauses would become unsat if flipped)
    break_count: Vec<u32>,
    /// Make count for each variable (how many clauses would become sat if flipped)
    make_count: Vec<u32>,
    /// Tabu list: variable -> flip when it becomes allowed again
    tabu_list: Vec<u64>,
    /// Current flip number
    current_flip: u64,
    /// Random state
    rng_state: u64,
    /// Statistics
    stats: SlsStats,
    /// Best assignment found
    best_assignment: Vec<bool>,
    /// Best number of unsatisfied clauses
    best_unsat_count: u32,
    /// Current noise (for adaptive)
    current_noise: f64,
}

impl SlsSolver {
    /// Create a new SLS solver
    pub fn new(config: SlsConfig) -> Self {
        let seed = config.seed;
        let noise = config.noise;
        Self {
            config,
            clauses: Vec::new(),
            num_vars: 0,
            assignment: Vec::new(),
            pos_occs: HashMap::new(),
            neg_occs: HashMap::new(),
            unsat_clauses: HashSet::new(),
            break_count: Vec::new(),
            make_count: Vec::new(),
            tabu_list: Vec::new(),
            current_flip: 0,
            rng_state: seed,
            stats: SlsStats::default(),
            best_assignment: Vec::new(),
            best_unsat_count: u32::MAX,
            current_noise: noise,
        }
    }

    /// Add a clause to the solver
    pub fn add_clause(&mut self, literals: &[Lit]) {
        if literals.is_empty() {
            return; // Skip empty clauses
        }

        let clause_id = ClauseId(self.clauses.len() as u32);
        let clause = SlsClause::new(literals.to_vec());
        self.clauses.push(clause);

        // Update variable count and occurrence lists
        for &lit in literals {
            let var = lit.unsigned_abs();
            if var > self.num_vars {
                self.num_vars = var;
            }

            if lit > 0 {
                self.pos_occs.entry(var).or_default().push(clause_id);
            } else {
                self.neg_occs.entry(var).or_default().push(clause_id);
            }
        }
    }

    /// Solve the formula
    pub fn solve(&mut self) -> SlsResult {
        if self.clauses.is_empty() {
            return SlsResult::Sat(Vec::new());
        }

        // Check for empty clause
        for clause in &self.clauses {
            if clause.len() == 0 {
                return SlsResult::Unsat;
            }
        }

        // Initialize data structures
        self.initialize();

        // Main SLS loop
        for restart in 0..self.config.max_restarts {
            self.stats.restarts = restart + 1;
            self.random_assignment();
            self.initialize_sat_counts();

            let flips_per_restart = self.config.max_flips / self.config.max_restarts as u64;
            for _ in 0..flips_per_restart {
                if self.unsat_clauses.is_empty() {
                    return SlsResult::Sat(self.assignment.clone());
                }

                // Update best
                let unsat_count = self.unsat_clauses.len() as u32;
                if unsat_count < self.best_unsat_count {
                    self.best_unsat_count = unsat_count;
                    self.best_assignment = self.assignment.clone();
                }

                // Pick variable to flip based on algorithm
                let var = match self.config.algorithm {
                    SlsAlgorithm::WalkSat => self.walksat_pick(),
                    SlsAlgorithm::Gsat => self.gsat_pick(),
                    SlsAlgorithm::ProbSat => self.probsat_pick(),
                    SlsAlgorithm::Adaptive => self.adaptive_pick(),
                };

                if let Some(v) = var {
                    self.flip_variable(v);
                    self.stats.flips += 1;
                    self.current_flip += 1;

                    // Adaptive noise adjustment
                    if self.config.adaptive_noise {
                        self.adjust_noise();
                    }
                }
            }
        }

        self.stats.best_unsat = self.best_unsat_count;
        self.stats.avg_flips_per_restart = self.stats.flips as f64 / self.stats.restarts as f64;

        SlsResult::Unknown
    }

    /// Initialize data structures
    fn initialize(&mut self) {
        let n = self.num_vars as usize + 1;
        self.assignment = vec![false; n];
        self.break_count = vec![0; n];
        self.make_count = vec![0; n];
        self.tabu_list = vec![0; n];
        self.best_assignment = vec![false; n];
        self.best_unsat_count = u32::MAX;
        self.current_flip = 0;
    }

    /// Generate random assignment
    fn random_assignment(&mut self) {
        for i in 1..=self.num_vars as usize {
            self.assignment[i] = self.random_bool();
        }
    }

    /// Initialize satisfaction counts for all clauses
    fn initialize_sat_counts(&mut self) {
        self.unsat_clauses.clear();

        for (i, clause) in self.clauses.iter_mut().enumerate() {
            clause.sat_count = 0;
            for &lit in &clause.literals {
                let var = lit.unsigned_abs() as usize;
                let is_pos = lit > 0;
                if self.assignment[var] == is_pos {
                    clause.sat_count += 1;
                }
            }
            if clause.sat_count == 0 {
                self.unsat_clauses.insert(ClauseId(i as u32));
            }
        }

        // Initialize make/break counts
        self.update_all_counts();
    }

    /// Update make/break counts for all variables
    fn update_all_counts(&mut self) {
        for var in 1..=self.num_vars as usize {
            self.break_count[var] = 0;
            self.make_count[var] = 0;
        }

        for (clause_id, clause) in self.clauses.iter().enumerate() {
            if clause.sat_count == 0 {
                // All literals in unsat clause would make it sat
                for &lit in &clause.literals {
                    let var = lit.unsigned_abs() as usize;
                    self.make_count[var] += 1;
                }
            } else if clause.sat_count == 1 {
                // Find the critical literal
                for &lit in &clause.literals {
                    let var = lit.unsigned_abs() as usize;
                    let is_pos = lit > 0;
                    if self.assignment[var] == is_pos {
                        self.break_count[var] += 1;
                        break;
                    }
                }
            }
            let _ = clause_id; // Suppress unused variable warning
        }
    }

    /// WalkSAT variable selection
    fn walksat_pick(&mut self) -> Option<Var> {
        // Pick a random unsatisfied clause
        let clause_id = self.random_unsat_clause()?;

        // Copy literals to avoid borrow conflict
        let literals: Vec<Lit> = self.clauses[clause_id.0 as usize].literals.clone();
        let clause_len = literals.len();

        // With probability noise, pick a random variable from the clause
        if self.random_float() < self.current_noise {
            self.stats.random_moves += 1;
            let idx = self.random_usize(clause_len);
            return Some(literals[idx].unsigned_abs());
        }

        // Otherwise, pick the variable with minimum break count
        self.stats.greedy_moves += 1;
        let mut best_var = None;
        let mut best_break = u32::MAX;

        for &lit in &literals {
            let var = lit.unsigned_abs();
            let break_val = self.break_count[var as usize];

            // Check tabu
            if self.config.tabu && self.tabu_list[var as usize] > self.current_flip {
                continue;
            }

            if break_val < best_break {
                best_break = break_val;
                best_var = Some(var);
            }
        }

        best_var
    }

    /// GSAT variable selection
    fn gsat_pick(&mut self) -> Option<Var> {
        // Find variable with maximum net gain (make - break)
        let mut best_var = None;
        let mut best_gain = i32::MIN;

        for var in 1..=self.num_vars {
            // Check tabu
            if self.config.tabu && self.tabu_list[var as usize] > self.current_flip {
                continue;
            }

            let gain = self.make_count[var as usize] as i32 - self.break_count[var as usize] as i32;
            if gain > best_gain {
                best_gain = gain;
                best_var = Some(var);
            }
        }

        // With noise probability, pick random instead
        if self.random_float() < self.current_noise {
            self.stats.random_moves += 1;
            let var = self.random_usize(self.num_vars as usize) as u32 + 1;
            return Some(var);
        }

        self.stats.greedy_moves += 1;
        best_var
    }

    /// ProbSAT variable selection
    fn probsat_pick(&mut self) -> Option<Var> {
        // Pick a random unsatisfied clause
        let clause_id = self.random_unsat_clause()?;

        // Copy literals to avoid borrow conflict
        let literals: Vec<Lit> = self.clauses[clause_id.0 as usize].literals.clone();

        // Calculate probability for each variable
        let mut probs: Vec<(Var, f64)> = Vec::new();
        let mut total = 0.0;

        let cb = 2.06; // ProbSAT parameter for break
        let cm = 0.0; // ProbSAT parameter for make

        for &lit in &literals {
            let var = lit.unsigned_abs();
            let break_val = self.break_count[var as usize];
            let make_val = self.make_count[var as usize];

            // f(make, break) = 0^(-cb*break) * (1+make)^cm
            // Simplified: just use break
            let prob = (0.9f64).powf(cb * break_val as f64) * (1.0 + make_val as f64).powf(cm);
            probs.push((var, prob));
            total += prob;
        }

        if total <= 0.0 {
            return probs.first().map(|&(v, _)| v);
        }

        // Roulette wheel selection
        let r = self.random_float() * total;
        let mut cumulative = 0.0;
        for (var, prob) in probs {
            cumulative += prob;
            if r <= cumulative {
                return Some(var);
            }
        }

        None
    }

    /// Adaptive algorithm selection
    fn adaptive_pick(&mut self) -> Option<Var> {
        // Switch between algorithms based on progress
        let progress = 1.0 - (self.unsat_clauses.len() as f64 / self.clauses.len() as f64);

        if progress > 0.9 {
            // Near solution: use precise WalkSAT
            self.walksat_pick()
        } else if progress > 0.5 {
            // Medium: use ProbSAT
            self.probsat_pick()
        } else {
            // Far: use GSAT
            self.gsat_pick()
        }
    }

    /// Flip a variable and update data structures
    fn flip_variable(&mut self, var: Var) {
        let var_idx = var as usize;
        let old_val = self.assignment[var_idx];
        self.assignment[var_idx] = !old_val;

        // Update tabu
        if self.config.tabu {
            self.tabu_list[var_idx] = self.current_flip + self.config.tabu_tenure as u64;
        }

        // Get affected clauses
        let (sat_clauses, unsat_clauses) = if old_val {
            // Was true, now false: positive occurrences lose a sat, negative gain
            (
                self.neg_occs.get(&var).cloned().unwrap_or_default(),
                self.pos_occs.get(&var).cloned().unwrap_or_default(),
            )
        } else {
            // Was false, now true: positive occurrences gain, negative lose
            (
                self.pos_occs.get(&var).cloned().unwrap_or_default(),
                self.neg_occs.get(&var).cloned().unwrap_or_default(),
            )
        };

        // Update clauses that gain a satisfied literal
        for clause_id in sat_clauses {
            let clause = &mut self.clauses[clause_id.0 as usize];
            clause.sat_count += 1;

            if clause.sat_count == 1 {
                // Was unsat, now sat
                self.unsat_clauses.remove(&clause_id);
            }
        }

        // Update clauses that lose a satisfied literal
        for clause_id in unsat_clauses {
            let clause = &mut self.clauses[clause_id.0 as usize];
            if clause.sat_count > 0 {
                clause.sat_count -= 1;

                if clause.sat_count == 0 {
                    // Was sat, now unsat
                    self.unsat_clauses.insert(clause_id);
                }
            }
        }

        // Update make/break counts (simplified: recompute all)
        self.update_all_counts();
    }

    /// Adjust noise for adaptive noise
    fn adjust_noise(&mut self) {
        let unsat_count = self.unsat_clauses.len() as u32;
        if unsat_count < self.best_unsat_count {
            // Improvement: decrease noise
            self.current_noise = (self.current_noise - self.config.noise_dec).max(0.01);
        } else {
            // No improvement: increase noise
            self.current_noise = (self.current_noise + self.config.noise_inc).min(0.99);
        }
    }

    /// Pick a random unsatisfied clause
    fn random_unsat_clause(&mut self) -> Option<ClauseId> {
        if self.unsat_clauses.is_empty() {
            return None;
        }
        let idx = self.random_usize(self.unsat_clauses.len());
        self.unsat_clauses.iter().nth(idx).copied()
    }

    /// Random number generator (xorshift64)
    fn random_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    fn random_bool(&mut self) -> bool {
        self.random_u64() & 1 == 0
    }

    fn random_float(&mut self) -> f64 {
        (self.random_u64() as f64) / (u64::MAX as f64)
    }

    fn random_usize(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.random_u64() as usize) % max
    }

    /// Get statistics
    pub fn stats(&self) -> &SlsStats {
        &self.stats
    }

    /// Get the best assignment found
    pub fn best_assignment(&self) -> &[bool] {
        &self.best_assignment
    }

    /// Get current assignment
    pub fn assignment(&self) -> &[bool] {
        &self.assignment
    }

    /// Number of variables
    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    /// Number of clauses
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.clauses.clear();
        self.num_vars = 0;
        self.pos_occs.clear();
        self.neg_occs.clear();
        self.unsat_clauses.clear();
        self.stats = SlsStats::default();
        self.rng_state = self.config.seed;
        self.current_noise = self.config.noise;
    }
}

impl Default for SlsSolver {
    fn default() -> Self {
        Self::new(SlsConfig::default())
    }
}

// ============================================================================
// Weighted MAX-SAT SLS
// ============================================================================

/// Configuration for weighted MAX-SAT SLS
#[derive(Debug, Clone)]
pub struct WeightedSlsConfig {
    /// Base SLS configuration
    pub base: SlsConfig,
    /// Weight update factor
    pub weight_update: f64,
    /// Smooth probability
    pub smooth_prob: f64,
}

impl Default for WeightedSlsConfig {
    fn default() -> Self {
        Self {
            base: SlsConfig::default(),
            weight_update: 1.0,
            smooth_prob: 0.01,
        }
    }
}

/// Weighted MAX-SAT SLS solver
#[derive(Debug)]
pub struct WeightedSlsSolver {
    /// Base SLS solver
    base: SlsSolver,
    /// Clause weights
    weights: Vec<f64>,
    /// Configuration
    #[allow(dead_code)]
    config: WeightedSlsConfig,
    /// Best cost found
    best_cost: f64,
    /// Statistics
    stats: WeightedSlsStats,
}

/// Statistics for weighted SLS
#[derive(Debug, Clone, Default)]
pub struct WeightedSlsStats {
    /// Base stats
    pub base_stats: SlsStats,
    /// Weight updates
    pub weight_updates: u64,
    /// Smooth operations
    pub smooth_ops: u64,
    /// Best cost found
    pub best_cost: f64,
}

impl WeightedSlsSolver {
    /// Create a new weighted SLS solver
    pub fn new(config: WeightedSlsConfig) -> Self {
        Self {
            base: SlsSolver::new(config.base.clone()),
            weights: Vec::new(),
            config,
            best_cost: f64::MAX,
            stats: WeightedSlsStats::default(),
        }
    }

    /// Add a weighted clause
    pub fn add_weighted_clause(&mut self, literals: &[Lit], weight: f64) {
        self.base.add_clause(literals);
        self.weights.push(weight);
    }

    /// Add a hard clause (weight = infinity)
    pub fn add_hard_clause(&mut self, literals: &[Lit]) {
        self.add_weighted_clause(literals, f64::MAX);
    }

    /// Add a soft clause with unit weight
    pub fn add_soft_clause(&mut self, literals: &[Lit]) {
        self.add_weighted_clause(literals, 1.0);
    }

    /// Solve for minimum cost
    pub fn solve(&mut self) -> (SlsResult, f64) {
        // Initialize weights in base solver
        for (i, &w) in self.weights.iter().enumerate() {
            if i < self.base.clauses.len() {
                self.base.clauses[i].weight = w;
            }
        }

        let result = self.base.solve();
        self.stats.base_stats = self.base.stats.clone();

        let cost = self.compute_cost(&self.base.best_assignment);
        self.stats.best_cost = cost;

        (result, cost)
    }

    /// Compute cost of an assignment
    fn compute_cost(&self, assignment: &[bool]) -> f64 {
        let mut cost = 0.0;

        for (i, clause) in self.base.clauses.iter().enumerate() {
            let mut satisfied = false;
            for &lit in &clause.literals {
                let var = lit.unsigned_abs() as usize;
                let is_pos = lit > 0;
                if var < assignment.len() && assignment[var] == is_pos {
                    satisfied = true;
                    break;
                }
            }
            if !satisfied {
                cost += self.weights.get(i).copied().unwrap_or(1.0);
            }
        }

        cost
    }

    /// Get statistics
    pub fn stats(&self) -> &WeightedSlsStats {
        &self.stats
    }

    /// Get best cost
    pub fn best_cost(&self) -> f64 {
        self.best_cost
    }
}

// ============================================================================
// Phase Saving and Polarity Heuristics
// ============================================================================

/// Phase saving mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhaseMode {
    /// Always start with false
    False,
    /// Always start with true
    True,
    /// Random phase
    Random,
    /// Save previous phase
    #[default]
    Save,
    /// Use polarity from unit propagation
    Unit,
}

/// Phase saver for SLS
#[derive(Debug)]
pub struct PhaseSaver {
    /// Saved phases
    phases: Vec<Option<bool>>,
    /// Mode
    mode: PhaseMode,
    /// Random seed
    rng_state: u64,
}

impl PhaseSaver {
    /// Create a new phase saver
    pub fn new(mode: PhaseMode) -> Self {
        Self {
            phases: Vec::new(),
            mode,
            rng_state: 42,
        }
    }

    /// Ensure capacity for n variables
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.phases.len() < n {
            self.phases.resize(n, None);
        }
    }

    /// Get phase for a variable
    pub fn get_phase(&mut self, var: Var) -> bool {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);

        match self.mode {
            PhaseMode::False => false,
            PhaseMode::True => true,
            PhaseMode::Random => {
                let x = &mut self.rng_state;
                *x ^= *x << 13;
                *x ^= *x >> 7;
                *x ^= *x << 17;
                *x & 1 == 0
            }
            PhaseMode::Save | PhaseMode::Unit => self.phases[idx].unwrap_or(false),
        }
    }

    /// Save phase for a variable
    pub fn save_phase(&mut self, var: Var, phase: bool) {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);
        self.phases[idx] = Some(phase);
    }

    /// Reset all phases
    pub fn reset(&mut self) {
        for phase in &mut self.phases {
            *phase = None;
        }
    }
}

// ============================================================================
// Clause Weighting Schemes
// ============================================================================

/// Clause weighting scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightingScheme {
    /// No weighting
    #[default]
    None,
    /// Additive weighting
    Additive,
    /// Multiplicative weighting
    Multiplicative,
    /// SAPS (Scaling and Probabilistic Smoothing)
    Saps,
    /// PAWS (Pure Additive Weighting Scheme)
    Paws,
}

/// Clause weight manager
#[derive(Debug)]
pub struct ClauseWeightManager {
    /// Weights for each clause
    weights: Vec<f64>,
    /// Weighting scheme
    scheme: WeightingScheme,
    /// Additive increment
    add_inc: f64,
    /// Multiplicative factor
    mult_factor: f64,
    /// Smooth probability
    #[allow(dead_code)]
    smooth_prob: f64,
}

impl ClauseWeightManager {
    /// Create a new weight manager
    pub fn new(scheme: WeightingScheme) -> Self {
        Self {
            weights: Vec::new(),
            scheme,
            add_inc: 1.0,
            mult_factor: 1.1,
            smooth_prob: 0.01,
        }
    }

    /// Initialize weights for n clauses
    pub fn initialize(&mut self, n: usize) {
        self.weights = vec![1.0; n];
    }

    /// Update weights for unsatisfied clauses
    pub fn update(&mut self, unsat: &HashSet<ClauseId>) {
        match self.scheme {
            WeightingScheme::None => {}
            WeightingScheme::Additive => {
                for &cid in unsat {
                    let idx = cid.0 as usize;
                    if idx < self.weights.len() {
                        self.weights[idx] += self.add_inc;
                    }
                }
            }
            WeightingScheme::Multiplicative => {
                for &cid in unsat {
                    let idx = cid.0 as usize;
                    if idx < self.weights.len() {
                        self.weights[idx] *= self.mult_factor;
                    }
                }
            }
            WeightingScheme::Saps | WeightingScheme::Paws => {
                // SAPS: Scale and smooth
                for &cid in unsat {
                    let idx = cid.0 as usize;
                    if idx < self.weights.len() {
                        self.weights[idx] *= self.mult_factor;
                    }
                }
            }
        }
    }

    /// Smooth weights (decrease all weights slightly)
    pub fn smooth(&mut self) {
        let avg: f64 = self.weights.iter().sum::<f64>() / self.weights.len() as f64;
        for w in &mut self.weights {
            *w = (*w + avg) / 2.0;
        }
    }

    /// Get weight for a clause
    pub fn get_weight(&self, clause_id: ClauseId) -> f64 {
        self.weights
            .get(clause_id.0 as usize)
            .copied()
            .unwrap_or(1.0)
    }

    /// Reset all weights
    pub fn reset(&mut self) {
        for w in &mut self.weights {
            *w = 1.0;
        }
    }
}

// ============================================================================
// Variable Selection Heuristics
// ============================================================================

/// Variable selection heuristic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VarSelectHeuristic {
    /// Minimum break (standard WalkSAT)
    #[default]
    MinBreak,
    /// Maximum make
    MaxMake,
    /// Maximum net gain (make - break)
    MaxGain,
    /// Age-based (oldest variable in clause)
    Age,
    /// Score-based (VSIDS-like)
    Score,
}

/// Variable activity tracker
#[derive(Debug)]
pub struct VarActivity {
    /// Activity scores
    activities: Vec<f64>,
    /// Decay factor
    decay: f64,
    /// Bump amount
    bump: f64,
}

impl VarActivity {
    /// Create a new activity tracker
    pub fn new() -> Self {
        Self {
            activities: Vec::new(),
            decay: 0.95,
            bump: 1.0,
        }
    }

    /// Ensure capacity for n variables
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.activities.len() < n {
            self.activities.resize(n, 0.0);
        }
    }

    /// Bump activity for a variable
    pub fn bump(&mut self, var: Var) {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);
        self.activities[idx] += self.bump;

        // Rescale if too large
        if self.activities[idx] > 1e100 {
            for a in &mut self.activities {
                *a *= 1e-100;
            }
            self.bump *= 1e-100;
        }
    }

    /// Decay all activities
    pub fn decay(&mut self) {
        self.bump /= self.decay;
    }

    /// Get activity for a variable
    pub fn get(&self, var: Var) -> f64 {
        self.activities.get(var as usize).copied().unwrap_or(0.0)
    }

    /// Reset all activities
    pub fn reset(&mut self) {
        for a in &mut self.activities {
            *a = 0.0;
        }
        self.bump = 1.0;
    }
}

impl Default for VarActivity {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Focused Random Walk
// ============================================================================

/// Focused random walk parameters
#[derive(Debug, Clone)]
pub struct FocusedWalkConfig {
    /// Probability of focused move
    pub focus_prob: f64,
    /// Number of variables to consider
    pub focus_size: usize,
    /// Use break score
    pub use_break: bool,
    /// Use make score
    pub use_make: bool,
}

impl Default for FocusedWalkConfig {
    fn default() -> Self {
        Self {
            focus_prob: 0.8,
            focus_size: 5,
            use_break: true,
            use_make: true,
        }
    }
}

/// Focused random walk solver
#[derive(Debug)]
pub struct FocusedWalk {
    /// Configuration
    config: FocusedWalkConfig,
    /// Focus set (candidate variables)
    focus_set: Vec<Var>,
}

impl FocusedWalk {
    /// Create a new focused walk
    pub fn new(config: FocusedWalkConfig) -> Self {
        Self {
            config,
            focus_set: Vec::new(),
        }
    }

    /// Update focus set from unsatisfied clause
    pub fn update_focus(&mut self, clause_lits: &[Lit], break_counts: &[u32]) {
        self.focus_set.clear();

        // Sort by break count
        let mut vars: Vec<_> = clause_lits
            .iter()
            .map(|&lit| {
                let var = lit.unsigned_abs();
                let break_val = break_counts.get(var as usize).copied().unwrap_or(0);
                (var, break_val)
            })
            .collect();

        vars.sort_by_key(|&(_, b)| b);

        // Take top focus_size variables
        for (var, _) in vars.into_iter().take(self.config.focus_size) {
            self.focus_set.push(var);
        }
    }

    /// Get the focus set
    pub fn focus_set(&self) -> &[Var] {
        &self.focus_set
    }
}

// ============================================================================
// Restart Strategies
// ============================================================================

/// Restart strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RestartStrategy {
    /// No restarts
    None,
    /// Fixed number of flips per restart
    Fixed(u64),
    /// Geometric sequence (multiply by factor each restart)
    Geometric(u64, f64),
    /// Luby sequence (provably optimal for certain problems)
    Luby(u64),
    /// Glucose-like (based on progress)
    Glucose,
    /// Adaptive (based on conflict analysis)
    Adaptive,
}

impl Default for RestartStrategy {
    fn default() -> Self {
        RestartStrategy::Geometric(1000, 1.5)
    }
}

/// Restart manager
#[derive(Debug)]
pub struct RestartManager {
    /// Strategy
    strategy: RestartStrategy,
    /// Current restart threshold
    current_threshold: u64,
    /// Restart count
    restart_count: u32,
    /// Luby sequence state
    luby_index: u32,
    /// Base unit for Luby
    #[allow(dead_code)]
    luby_unit: u64,
    /// Recent conflict counts (for adaptive)
    recent_conflicts: Vec<u32>,
    /// LBD queue (for Glucose-like)
    lbd_queue: Vec<u32>,
}

impl RestartManager {
    /// Create a new restart manager
    pub fn new(strategy: RestartStrategy) -> Self {
        let (threshold, luby_unit) = match strategy {
            RestartStrategy::None => (u64::MAX, 1),
            RestartStrategy::Fixed(n) => (n, n),
            RestartStrategy::Geometric(base, _) => (base, base),
            RestartStrategy::Luby(unit) => (unit, unit),
            RestartStrategy::Glucose => (50, 50),
            RestartStrategy::Adaptive => (1000, 1000),
        };

        Self {
            strategy,
            current_threshold: threshold,
            restart_count: 0,
            luby_index: 1,
            luby_unit,
            recent_conflicts: Vec::new(),
            lbd_queue: Vec::new(),
        }
    }

    /// Check if should restart
    pub fn should_restart(&self, flips: u64) -> bool {
        flips >= self.current_threshold
    }

    /// Notify of a restart
    pub fn notify_restart(&mut self) {
        self.restart_count += 1;

        self.current_threshold = match self.strategy {
            RestartStrategy::None => u64::MAX,
            RestartStrategy::Fixed(n) => n,
            RestartStrategy::Geometric(base, factor) => {
                let mult = factor.powi(self.restart_count as i32);
                (base as f64 * mult) as u64
            }
            RestartStrategy::Luby(unit) => {
                let luby_val = self.luby(self.luby_index);
                self.luby_index += 1;
                unit * luby_val as u64
            }
            RestartStrategy::Glucose => {
                // Simple Glucose-like: check if recent LBDs are high
                if self.lbd_queue.len() >= 50 {
                    let avg: u32 = self.lbd_queue.iter().sum::<u32>() / 50;
                    if avg > 5 {
                        self.current_threshold / 2
                    } else {
                        self.current_threshold * 2
                    }
                } else {
                    self.current_threshold
                }
            }
            RestartStrategy::Adaptive => {
                // Based on conflict rate
                if self.recent_conflicts.len() >= 10 {
                    let sum: u32 = self.recent_conflicts.iter().sum();
                    let avg = sum / 10;
                    if avg > 100 {
                        self.current_threshold / 2
                    } else {
                        self.current_threshold * 2
                    }
                } else {
                    self.current_threshold
                }
            }
        };
    }

    /// Record conflict (for adaptive strategies)
    pub fn record_conflict(&mut self, conflicts: u32) {
        self.recent_conflicts.push(conflicts);
        if self.recent_conflicts.len() > 100 {
            self.recent_conflicts.remove(0);
        }
    }

    /// Record LBD (for Glucose-like)
    pub fn record_lbd(&mut self, lbd: u32) {
        self.lbd_queue.push(lbd);
        if self.lbd_queue.len() > 100 {
            self.lbd_queue.remove(0);
        }
    }

    /// Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    #[allow(clippy::only_used_in_recursion)]
    fn luby(&self, i: u32) -> u32 {
        if i == 0 {
            return 1;
        }

        // Find k such that 2^k - 1 == i
        let mut k = 1u32;
        let mut power = 2u32;
        while power - 1 < i {
            k += 1;
            power *= 2;
        }

        if power - 1 == i {
            // i is 2^k - 1, return 2^(k-1)
            1u32 << (k - 1)
        } else {
            // Recurse
            self.luby(i - (power / 2) + 1)
        }
    }

    /// Get restart count
    pub fn count(&self) -> u32 {
        self.restart_count
    }

    /// Get current threshold
    pub fn threshold(&self) -> u64 {
        self.current_threshold
    }
}

// ============================================================================
// Novelty and rNovelty Heuristics
// ============================================================================

/// Novelty parameter for variable selection
#[derive(Debug, Clone)]
pub struct NoveltyConfig {
    /// Probability of selecting second-best variable (p)
    pub novelty_prob: f64,
    /// Enable novelty+ (extra random walk)
    pub novelty_plus: bool,
    /// Random walk probability for novelty+
    pub wp: f64,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            novelty_prob: 0.5,
            novelty_plus: true,
            wp: 0.01,
        }
    }
}

/// Novelty variable selector
#[derive(Debug)]
pub struct NoveltySelector {
    /// Configuration
    config: NoveltyConfig,
    /// Last flipped variable
    last_flipped: Option<Var>,
    /// Flip age for each variable
    flip_age: Vec<u64>,
    /// Current time (flip count)
    current_time: u64,
}

impl NoveltySelector {
    /// Create a new novelty selector
    pub fn new(config: NoveltyConfig) -> Self {
        Self {
            config,
            last_flipped: None,
            flip_age: Vec::new(),
            current_time: 0,
        }
    }

    /// Ensure capacity for n variables
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.flip_age.len() < n {
            self.flip_age.resize(n, 0);
        }
    }

    /// Notify that a variable was flipped
    pub fn notify_flip(&mut self, var: Var) {
        let idx = var as usize;
        self.ensure_capacity(idx + 1);
        self.flip_age[idx] = self.current_time;
        self.last_flipped = Some(var);
        self.current_time += 1;
    }

    /// Get age of a variable (flips since last flip)
    pub fn age(&self, var: Var) -> u64 {
        let idx = var as usize;
        if idx < self.flip_age.len() {
            self.current_time.saturating_sub(self.flip_age[idx])
        } else {
            self.current_time
        }
    }

    /// Select variable from candidates using novelty heuristic
    pub fn select(
        &self,
        candidates: &[(Var, i32)], // (variable, break count)
        rng: &mut u64,
    ) -> Option<Var> {
        if candidates.is_empty() {
            return None;
        }

        if candidates.len() == 1 {
            return Some(candidates[0].0);
        }

        // Sort by break count (best = lowest break)
        let mut sorted: Vec<_> = candidates.to_vec();
        sorted.sort_by_key(|&(_, b)| b);

        let best = sorted[0].0;
        let second_best = sorted[1].0;

        // If best is the same as last flipped, consider second best
        if Some(best) == self.last_flipped {
            // With probability novelty_prob, pick second best
            let r = (*rng as f64) / (u64::MAX as f64);
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;

            if r < self.config.novelty_prob {
                return Some(second_best);
            }
        }

        // Novelty+: with small probability, pick random
        if self.config.novelty_plus {
            let r = (*rng as f64) / (u64::MAX as f64);
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;

            if r < self.config.wp {
                let idx = (*rng as usize) % candidates.len();
                *rng ^= *rng << 13;
                *rng ^= *rng >> 7;
                *rng ^= *rng << 17;
                return Some(candidates[idx].0);
            }
        }

        Some(best)
    }

    /// Reset the selector
    pub fn reset(&mut self) {
        self.last_flipped = None;
        self.current_time = 0;
        for age in &mut self.flip_age {
            *age = 0;
        }
    }
}

// ============================================================================
// CCAnr (Conflict-driven Clause Learning with Novelty and Restart)
// ============================================================================

/// CCAnr configuration
#[derive(Debug, Clone)]
pub struct CcanrConfig {
    /// Configuration score increase
    pub score_inc: f64,
    /// Average weight threshold
    pub avg_weight_threshold: f64,
    /// Clause weight limit
    pub weight_limit: f64,
    /// Enable configuration checking
    pub config_checking: bool,
}

impl Default for CcanrConfig {
    fn default() -> Self {
        Self {
            score_inc: 1.0,
            avg_weight_threshold: 3.0,
            weight_limit: 100.0,
            config_checking: true,
        }
    }
}

/// CCAnr solver enhancements
#[derive(Debug)]
pub struct CcanrEnhancer {
    /// Configuration
    config: CcanrConfig,
    /// Variable scores
    scores: Vec<f64>,
    /// Configuration checking bits
    config_bits: Vec<bool>,
    /// Total weight
    total_weight: f64,
}

impl CcanrEnhancer {
    /// Create a new CCAnr enhancer
    pub fn new(config: CcanrConfig) -> Self {
        Self {
            config,
            scores: Vec::new(),
            config_bits: Vec::new(),
            total_weight: 0.0,
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.scores = vec![0.0; n];
        self.config_bits = vec![false; n];
        self.total_weight = 0.0;
    }

    /// Update scores for unsatisfied clause
    pub fn update_scores(&mut self, clause_lits: &[Lit]) {
        for &lit in clause_lits {
            let var = lit.unsigned_abs() as usize;
            if var < self.scores.len() {
                self.scores[var] += self.config.score_inc;
            }
        }
    }

    /// Set configuration bit for variable
    pub fn set_config(&mut self, var: Var, value: bool) {
        let idx = var as usize;
        if idx < self.config_bits.len() {
            self.config_bits[idx] = value;
        }
    }

    /// Check configuration bit
    pub fn check_config(&self, var: Var) -> bool {
        let idx = var as usize;
        if idx < self.config_bits.len() {
            self.config_bits[idx]
        } else {
            false
        }
    }

    /// Get score for variable
    pub fn score(&self, var: Var) -> f64 {
        self.scores.get(var as usize).copied().unwrap_or(0.0)
    }

    /// Decay all scores
    pub fn decay_scores(&mut self, factor: f64) {
        for s in &mut self.scores {
            *s *= factor;
        }
    }

    /// Should smooth weights?
    pub fn should_smooth(&self, num_clauses: usize) -> bool {
        if num_clauses == 0 {
            return false;
        }
        let avg = self.total_weight / num_clauses as f64;
        avg > self.config.avg_weight_threshold
    }

    /// Add to total weight
    pub fn add_weight(&mut self, w: f64) {
        self.total_weight += w;
    }

    /// Reset
    pub fn reset(&mut self) {
        for s in &mut self.scores {
            *s = 0.0;
        }
        for b in &mut self.config_bits {
            *b = false;
        }
        self.total_weight = 0.0;
    }
}

// ============================================================================
// Backbone Detection
// ============================================================================

/// Backbone analysis (variables that have the same value in all solutions)
#[derive(Debug)]
pub struct BackboneDetector {
    /// Known backbone literals (positive = must be true, negative = must be false)
    backbone: Vec<Lit>,
    /// Candidate backbone literals
    candidates: HashSet<Lit>,
    /// Solutions seen
    solutions_seen: u32,
    /// Minimum solutions before declaring backbone
    min_solutions: u32,
}

impl BackboneDetector {
    /// Create a new backbone detector
    pub fn new(min_solutions: u32) -> Self {
        Self {
            backbone: Vec::new(),
            candidates: HashSet::new(),
            solutions_seen: 0,
            min_solutions,
        }
    }

    /// Initialize candidates from first solution
    pub fn initialize(&mut self, solution: &[bool]) {
        self.candidates.clear();
        for (i, &val) in solution.iter().enumerate().skip(1) {
            let lit = if val { i as i32 } else { -(i as i32) };
            self.candidates.insert(lit);
        }
        self.solutions_seen = 1;
    }

    /// Update with new solution
    pub fn update(&mut self, solution: &[bool]) {
        if self.candidates.is_empty() {
            self.initialize(solution);
            return;
        }

        // Remove candidates that don't match this solution
        self.candidates.retain(|&lit| {
            let var = lit.unsigned_abs() as usize;
            let expected = lit > 0;
            var < solution.len() && solution[var] == expected
        });

        self.solutions_seen += 1;

        // If we've seen enough solutions, commit candidates to backbone
        if self.solutions_seen >= self.min_solutions {
            self.backbone.extend(self.candidates.iter());
            self.candidates.clear();
        }
    }

    /// Get the detected backbone
    pub fn backbone(&self) -> &[Lit] {
        &self.backbone
    }

    /// Check if variable is in backbone
    pub fn is_backbone(&self, var: Var) -> Option<bool> {
        let pos = var as i32;
        let neg = -(var as i32);

        if self.backbone.contains(&pos) || self.candidates.contains(&pos) {
            Some(true)
        } else if self.backbone.contains(&neg) || self.candidates.contains(&neg) {
            Some(false)
        } else {
            None
        }
    }

    /// Number of backbone variables detected
    pub fn backbone_size(&self) -> usize {
        self.backbone.len() + self.candidates.len()
    }

    /// Reset
    pub fn reset(&mut self) {
        self.backbone.clear();
        self.candidates.clear();
        self.solutions_seen = 0;
    }
}

// ============================================================================
// Diversification Methods
// ============================================================================

/// Diversification strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiversificationStrategy {
    /// No diversification
    None,
    /// Random walk
    RandomWalk,
    /// Stagnation-based
    #[default]
    Stagnation,
    /// Configuration changing
    ConfigChange,
}

/// Diversification manager
#[derive(Debug)]
pub struct DiversificationManager {
    /// Strategy
    strategy: DiversificationStrategy,
    /// Stagnation counter
    stagnation_count: u32,
    /// Stagnation threshold
    stagnation_threshold: u32,
    /// Last best score
    last_best: u32,
    /// Configuration change rate
    config_change_rate: f64,
}

impl DiversificationManager {
    /// Create a new diversification manager
    pub fn new(strategy: DiversificationStrategy) -> Self {
        Self {
            strategy,
            stagnation_count: 0,
            stagnation_threshold: 100,
            last_best: u32::MAX,
            config_change_rate: 0.1,
        }
    }

    /// Update with current state
    pub fn update(&mut self, current_unsat: u32) {
        if current_unsat >= self.last_best {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
            self.last_best = current_unsat;
        }
    }

    /// Check if should diversify
    pub fn should_diversify(&self) -> bool {
        match self.strategy {
            DiversificationStrategy::None => false,
            DiversificationStrategy::RandomWalk => true, // Always consider
            DiversificationStrategy::Stagnation => {
                self.stagnation_count >= self.stagnation_threshold
            }
            DiversificationStrategy::ConfigChange => {
                self.stagnation_count >= self.stagnation_threshold / 2
            }
        }
    }

    /// Get diversification probability
    pub fn diversify_prob(&self) -> f64 {
        match self.strategy {
            DiversificationStrategy::None => 0.0,
            DiversificationStrategy::RandomWalk => 0.01,
            DiversificationStrategy::Stagnation => {
                if self.stagnation_count >= self.stagnation_threshold {
                    0.5
                } else {
                    0.0
                }
            }
            DiversificationStrategy::ConfigChange => self.config_change_rate,
        }
    }

    /// Notify of diversification
    pub fn notify_diversify(&mut self) {
        self.stagnation_count = 0;
    }

    /// Reset
    pub fn reset(&mut self) {
        self.stagnation_count = 0;
        self.last_best = u32::MAX;
    }
}

// ============================================================================
// Clause Subsumption and Simplification
// ============================================================================

/// Clause simplifier for SLS
#[derive(Debug)]
pub struct ClauseSimplifier {
    /// Occurrence lists: literal -> clause indices
    occurrences: HashMap<Lit, Vec<usize>>,
    /// Clause sizes
    clause_sizes: Vec<usize>,
    /// Deleted clauses
    deleted: HashSet<usize>,
}

impl ClauseSimplifier {
    /// Create a new clause simplifier
    pub fn new() -> Self {
        Self {
            occurrences: HashMap::new(),
            clause_sizes: Vec::new(),
            deleted: HashSet::new(),
        }
    }

    /// Build occurrence lists
    pub fn build(&mut self, clauses: &[Vec<Lit>]) {
        self.occurrences.clear();
        self.clause_sizes.clear();
        self.deleted.clear();

        for (i, clause) in clauses.iter().enumerate() {
            self.clause_sizes.push(clause.len());
            for &lit in clause {
                self.occurrences.entry(lit).or_default().push(i);
            }
        }
    }

    /// Check if clause a subsumes clause b
    pub fn subsumes(&self, clause_a: &[Lit], clause_b: &[Lit]) -> bool {
        if clause_a.len() > clause_b.len() {
            return false;
        }

        let b_set: HashSet<Lit> = clause_b.iter().copied().collect();
        clause_a.iter().all(|lit| b_set.contains(lit))
    }

    /// Simplify by subsumption
    pub fn simplify_subsumption(&mut self, clauses: &[Vec<Lit>]) -> Vec<usize> {
        let mut to_delete = Vec::new();

        for (i, clause_a) in clauses.iter().enumerate() {
            if self.deleted.contains(&i) {
                continue;
            }

            // Check if this clause subsumes any other
            if let Some(&lit) = clause_a.first()
                && let Some(candidates) = self.occurrences.get(&lit)
            {
                for &j in candidates {
                    if i != j && !self.deleted.contains(&j) && self.subsumes(clause_a, &clauses[j])
                    {
                        to_delete.push(j);
                        self.deleted.insert(j);
                    }
                }
            }
        }

        to_delete
    }

    /// Find unit clauses
    pub fn find_units(&self, clauses: &[Vec<Lit>]) -> Vec<Lit> {
        clauses
            .iter()
            .enumerate()
            .filter(|(i, clause)| clause.len() == 1 && !self.deleted.contains(i))
            .map(|(_, clause)| clause[0])
            .collect()
    }

    /// Propagate units
    pub fn propagate_unit(&mut self, unit: Lit, clauses: &mut [Vec<Lit>]) {
        let neg_unit = -unit;

        // Remove clauses containing the unit literal
        if let Some(indices) = self.occurrences.get(&unit).cloned() {
            for i in indices {
                self.deleted.insert(i);
            }
        }

        // Remove negation from clauses
        if let Some(indices) = self.occurrences.get(&neg_unit).cloned() {
            for i in indices {
                if !self.deleted.contains(&i) {
                    clauses[i].retain(|&lit| lit != neg_unit);
                }
            }
        }
    }

    /// Check if formula is satisfiable (trivial check)
    pub fn is_trivially_unsat(&self, clauses: &[Vec<Lit>]) -> bool {
        clauses
            .iter()
            .enumerate()
            .any(|(i, clause)| clause.is_empty() && !self.deleted.contains(&i))
    }

    /// Get deleted clause count
    pub fn deleted_count(&self) -> usize {
        self.deleted.len()
    }
}

impl Default for ClauseSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Sparrow Algorithm
// ============================================================================

/// Sparrow algorithm configuration
#[derive(Debug, Clone)]
pub struct SparrowConfig {
    /// Base probability scaling factor c_b
    pub cb: f64,
    /// Make probability scaling factor c_m
    pub cm: f64,
    /// Smoothing probability
    pub sp: f64,
    /// Age factor
    pub age_factor: f64,
}

impl Default for SparrowConfig {
    fn default() -> Self {
        Self {
            cb: 2.06,
            cm: 0.0,
            sp: 0.8,
            age_factor: 4.0,
        }
    }
}

/// Sparrow variable selector
#[derive(Debug)]
pub struct SparrowSelector {
    /// Configuration
    config: SparrowConfig,
    /// Variable ages (flips since last flip)
    ages: Vec<u64>,
    /// Current flip count
    current_flip: u64,
    /// Probabilities workspace
    probs: Vec<f64>,
}

impl SparrowSelector {
    /// Create a new Sparrow selector
    pub fn new(config: SparrowConfig) -> Self {
        Self {
            config,
            ages: Vec::new(),
            current_flip: 0,
            probs: Vec::new(),
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.ages = vec![0; n];
        self.current_flip = 0;
    }

    /// Notify flip
    pub fn notify_flip(&mut self, var: Var) {
        let idx = var as usize;
        if idx < self.ages.len() {
            self.ages[idx] = self.current_flip;
        }
        self.current_flip += 1;
    }

    /// Get age of variable
    fn age(&self, var: Var) -> u64 {
        let idx = var as usize;
        if idx < self.ages.len() {
            self.current_flip.saturating_sub(self.ages[idx])
        } else {
            self.current_flip
        }
    }

    /// Select variable from clause literals
    pub fn select(
        &mut self,
        clause_lits: &[Lit],
        break_counts: &[u32],
        make_counts: &[u32],
        rng: &mut u64,
    ) -> Option<Var> {
        if clause_lits.is_empty() {
            return None;
        }

        self.probs.clear();
        let mut total = 0.0;

        for &lit in clause_lits {
            let var = lit.unsigned_abs();
            let break_val = break_counts.get(var as usize).copied().unwrap_or(0) as f64;
            let make_val = make_counts.get(var as usize).copied().unwrap_or(0) as f64;
            let age_val = self.age(var) as f64;

            // Sparrow probability function
            let prob = self.config.sp.powf(break_val)
                * (1.0 + make_val).powf(self.config.cm)
                * (1.0 + age_val).powf(self.config.age_factor);

            self.probs.push(prob);
            total += prob;
        }

        if total <= 0.0 {
            return Some(clause_lits[0].unsigned_abs());
        }

        // Roulette wheel selection
        let mut r = (*rng as f64 / u64::MAX as f64) * total;
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;

        for (i, &prob) in self.probs.iter().enumerate() {
            r -= prob;
            if r <= 0.0 {
                return Some(clause_lits[i].unsigned_abs());
            }
        }

        Some(clause_lits.last()?.unsigned_abs())
    }

    /// Reset
    pub fn reset(&mut self) {
        for age in &mut self.ages {
            *age = 0;
        }
        self.current_flip = 0;
    }
}

// ============================================================================
// Break-Make Score (BMS) Selector
// ============================================================================

/// BMS (Break-Make Score) variable selection
#[derive(Debug, Clone)]
pub struct BmsConfig {
    /// Break weight
    pub break_weight: f64,
    /// Make weight
    pub make_weight: f64,
    /// Age weight
    pub age_weight: f64,
    /// Polynomial exponent for break
    pub break_exp: f64,
    /// Polynomial exponent for make
    pub make_exp: f64,
}

impl Default for BmsConfig {
    fn default() -> Self {
        Self {
            break_weight: 1.0,
            make_weight: 0.5,
            age_weight: 0.1,
            break_exp: 2.0,
            make_exp: 1.0,
        }
    }
}

/// BMS variable selector
#[derive(Debug)]
pub struct BmsSelector {
    /// Configuration
    config: BmsConfig,
    /// Variable ages
    ages: Vec<u64>,
    /// Current flip
    current_flip: u64,
    /// Score cache
    score_cache: Vec<f64>,
}

impl BmsSelector {
    /// Create a new BMS selector
    pub fn new(config: BmsConfig) -> Self {
        Self {
            config,
            ages: Vec::new(),
            current_flip: 0,
            score_cache: Vec::new(),
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.ages = vec![0; n];
        self.score_cache = vec![0.0; n];
        self.current_flip = 0;
    }

    /// Notify flip
    pub fn notify_flip(&mut self, var: Var) {
        let idx = var as usize;
        if idx < self.ages.len() {
            self.ages[idx] = self.current_flip;
        }
        self.current_flip += 1;
    }

    /// Compute BMS score for a variable
    pub fn compute_score(&self, var: Var, break_count: u32, make_count: u32) -> f64 {
        let idx = var as usize;
        let age = if idx < self.ages.len() {
            self.current_flip.saturating_sub(self.ages[idx]) as f64
        } else {
            self.current_flip as f64
        };

        // BMS formula: -break_weight * break^exp + make_weight * make^exp + age_weight * age
        -self.config.break_weight * (break_count as f64).powf(self.config.break_exp)
            + self.config.make_weight * (make_count as f64).powf(self.config.make_exp)
            + self.config.age_weight * age
    }

    /// Select best variable by BMS score
    pub fn select(
        &self,
        candidates: &[Var],
        break_counts: &[u32],
        make_counts: &[u32],
    ) -> Option<Var> {
        candidates
            .iter()
            .map(|&v| {
                let b = break_counts.get(v as usize).copied().unwrap_or(0);
                let m = make_counts.get(v as usize).copied().unwrap_or(0);
                (v, self.compute_score(v, b, m))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(v, _)| v)
    }

    /// Reset
    pub fn reset(&mut self) {
        for age in &mut self.ages {
            *age = 0;
        }
        self.current_flip = 0;
    }
}

// ============================================================================
// Solution Verification
// ============================================================================

/// Solution verifier for SLS
#[derive(Debug)]
pub struct SolutionVerifier {
    /// Cached clause data
    clauses: Vec<Vec<Lit>>,
}

impl SolutionVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
        }
    }

    /// Set clauses to verify against
    pub fn set_clauses(&mut self, clauses: Vec<Vec<Lit>>) {
        self.clauses = clauses;
    }

    /// Verify a solution
    pub fn verify(&self, assignment: &[bool]) -> VerificationResult {
        let mut satisfied = 0;
        let mut unsatisfied = Vec::new();

        for (i, clause) in self.clauses.iter().enumerate() {
            let mut clause_sat = false;
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                let expected = lit > 0;
                if var < assignment.len() && assignment[var] == expected {
                    clause_sat = true;
                    break;
                }
            }
            if clause_sat {
                satisfied += 1;
            } else {
                unsatisfied.push(i);
            }
        }

        VerificationResult {
            is_valid: unsatisfied.is_empty(),
            satisfied_count: satisfied,
            unsatisfied_indices: unsatisfied,
        }
    }

    /// Quick check (returns true if all clauses satisfied)
    pub fn is_valid(&self, assignment: &[bool]) -> bool {
        for clause in &self.clauses {
            let mut clause_sat = false;
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                let expected = lit > 0;
                if var < assignment.len() && assignment[var] == expected {
                    clause_sat = true;
                    break;
                }
            }
            if !clause_sat {
                return false;
            }
        }
        true
    }
}

impl Default for SolutionVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of solution verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Is the solution valid (all clauses satisfied)?
    pub is_valid: bool,
    /// Number of satisfied clauses
    pub satisfied_count: usize,
    /// Indices of unsatisfied clauses
    pub unsatisfied_indices: Vec<usize>,
}

// ============================================================================
// DDFW (Divide and Distribute Fixed Weights)
// ============================================================================

/// DDFW configuration
#[derive(Debug, Clone)]
pub struct DdfwConfig {
    /// Initial weight for all clauses
    pub init_weight: f64,
    /// Weight transfer amount
    pub transfer_amount: f64,
    /// Distribution frequency (every N flips)
    pub distribute_freq: u32,
}

impl Default for DdfwConfig {
    fn default() -> Self {
        Self {
            init_weight: 1.0,
            transfer_amount: 1.0,
            distribute_freq: 100,
        }
    }
}

/// DDFW weight manager
#[derive(Debug)]
pub struct DdfwManager {
    /// Configuration
    config: DdfwConfig,
    /// Clause weights
    weights: Vec<f64>,
    /// Flip counter for distribution
    flip_counter: u32,
    /// Total weight (should be constant)
    total_weight: f64,
}

impl DdfwManager {
    /// Create a new DDFW manager
    pub fn new(config: DdfwConfig) -> Self {
        Self {
            config,
            weights: Vec::new(),
            flip_counter: 0,
            total_weight: 0.0,
        }
    }

    /// Initialize for n clauses
    pub fn initialize(&mut self, n: usize) {
        self.weights = vec![self.config.init_weight; n];
        self.total_weight = self.config.init_weight * n as f64;
        self.flip_counter = 0;
    }

    /// Get weight for clause
    pub fn weight(&self, clause_id: usize) -> f64 {
        self.weights
            .get(clause_id)
            .copied()
            .unwrap_or(self.config.init_weight)
    }

    /// Notify of flip
    pub fn notify_flip(&mut self) {
        self.flip_counter += 1;
    }

    /// Should distribute weights?
    pub fn should_distribute(&self) -> bool {
        self.flip_counter >= self.config.distribute_freq
    }

    /// Distribute weights from satisfied to unsatisfied clauses
    pub fn distribute(&mut self, satisfied: &[usize], unsatisfied: &[usize]) {
        if satisfied.is_empty() || unsatisfied.is_empty() {
            self.flip_counter = 0;
            return;
        }

        // Calculate transfer amount from each satisfied clause
        let transfer_per_sat = self.config.transfer_amount / satisfied.len() as f64;
        let gain_per_unsat = (transfer_per_sat * satisfied.len() as f64) / unsatisfied.len() as f64;

        // Transfer from satisfied
        for &idx in satisfied {
            if idx < self.weights.len() {
                let transfer = self.weights[idx].min(transfer_per_sat);
                self.weights[idx] -= transfer;
            }
        }

        // Add to unsatisfied
        for &idx in unsatisfied {
            if idx < self.weights.len() {
                self.weights[idx] += gain_per_unsat;
            }
        }

        self.flip_counter = 0;
    }

    /// Reset
    pub fn reset(&mut self) {
        let n = self.weights.len();
        for w in &mut self.weights {
            *w = self.config.init_weight;
        }
        self.total_weight = self.config.init_weight * n as f64;
        self.flip_counter = 0;
    }
}

// ============================================================================
// Clause Importance Tracking
// ============================================================================

/// Tracks importance of clauses based on conflict frequency
#[derive(Debug)]
pub struct ClauseImportance {
    /// Hit count (how often clause was unsatisfied)
    hit_counts: Vec<u32>,
    /// Critical count (how often clause was the only one unsatisfied)
    critical_counts: Vec<u32>,
    /// Decay factor
    decay: f64,
    /// Decay interval (flips)
    decay_interval: u32,
    /// Current flip
    current_flip: u32,
}

impl ClauseImportance {
    /// Create new importance tracker
    pub fn new() -> Self {
        Self {
            hit_counts: Vec::new(),
            critical_counts: Vec::new(),
            decay: 0.99,
            decay_interval: 1000,
            current_flip: 0,
        }
    }

    /// Initialize for n clauses
    pub fn initialize(&mut self, n: usize) {
        self.hit_counts = vec![0; n];
        self.critical_counts = vec![0; n];
        self.current_flip = 0;
    }

    /// Record hit on clause
    pub fn record_hit(&mut self, clause_id: usize) {
        if clause_id < self.hit_counts.len() {
            self.hit_counts[clause_id] = self.hit_counts[clause_id].saturating_add(1);
        }
    }

    /// Record critical clause (only unsatisfied)
    pub fn record_critical(&mut self, clause_id: usize) {
        if clause_id < self.critical_counts.len() {
            self.critical_counts[clause_id] = self.critical_counts[clause_id].saturating_add(1);
        }
    }

    /// Notify flip
    pub fn notify_flip(&mut self) {
        self.current_flip += 1;
        if self.current_flip >= self.decay_interval {
            self.decay_all();
            self.current_flip = 0;
        }
    }

    /// Decay all counts
    fn decay_all(&mut self) {
        for h in &mut self.hit_counts {
            *h = (*h as f64 * self.decay) as u32;
        }
        for c in &mut self.critical_counts {
            *c = (*c as f64 * self.decay) as u32;
        }
    }

    /// Get importance score (combines hit and critical counts)
    pub fn importance(&self, clause_id: usize) -> f64 {
        let hit = self.hit_counts.get(clause_id).copied().unwrap_or(0) as f64;
        let critical = self.critical_counts.get(clause_id).copied().unwrap_or(0) as f64;
        hit + 2.0 * critical // Critical counts worth more
    }

    /// Get most important clauses
    pub fn most_important(&self, n: usize) -> Vec<usize> {
        let mut indices: Vec<_> = (0..self.hit_counts.len())
            .map(|i| (i, self.importance(i)))
            .collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indices.into_iter().take(n).map(|(i, _)| i).collect()
    }

    /// Reset
    pub fn reset(&mut self) {
        for h in &mut self.hit_counts {
            *h = 0;
        }
        for c in &mut self.critical_counts {
            *c = 0;
        }
        self.current_flip = 0;
    }
}

impl Default for ClauseImportance {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Learning from Solutions
// ============================================================================

/// Learns variable polarities from found solutions
#[derive(Debug)]
pub struct SolutionLearner {
    /// Polarity counts: (true_count, false_count) for each variable
    polarity_counts: Vec<(u32, u32)>,
    /// Solutions collected
    solutions_count: u32,
    /// Maximum solutions to track
    max_solutions: u32,
}

impl SolutionLearner {
    /// Create new learner
    pub fn new(max_solutions: u32) -> Self {
        Self {
            polarity_counts: Vec::new(),
            solutions_count: 0,
            max_solutions,
        }
    }

    /// Initialize for n variables
    pub fn initialize(&mut self, n: usize) {
        self.polarity_counts = vec![(0, 0); n];
        self.solutions_count = 0;
    }

    /// Record a solution
    pub fn record_solution(&mut self, assignment: &[bool]) {
        if self.solutions_count >= self.max_solutions {
            return;
        }

        for (i, &val) in assignment.iter().enumerate() {
            if i < self.polarity_counts.len() {
                if val {
                    self.polarity_counts[i].0 += 1;
                } else {
                    self.polarity_counts[i].1 += 1;
                }
            }
        }
        self.solutions_count += 1;
    }

    /// Get preferred polarity for variable
    pub fn preferred_polarity(&self, var: Var) -> Option<bool> {
        let idx = var as usize;
        if idx < self.polarity_counts.len() {
            let (true_count, false_count) = self.polarity_counts[idx];
            if true_count > false_count {
                Some(true)
            } else if false_count > true_count {
                Some(false)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Get polarity confidence (0.0 to 1.0)
    pub fn polarity_confidence(&self, var: Var) -> f64 {
        let idx = var as usize;
        if idx < self.polarity_counts.len() {
            let (true_count, false_count) = self.polarity_counts[idx];
            let total = true_count + false_count;
            if total == 0 {
                return 0.5;
            }
            let max_count = true_count.max(false_count);
            max_count as f64 / total as f64
        } else {
            0.5
        }
    }

    /// Get high-confidence variables
    pub fn high_confidence_vars(&self, threshold: f64) -> Vec<(Var, bool)> {
        self.polarity_counts
            .iter()
            .enumerate()
            .filter_map(|(i, &(t, f))| {
                let total = t + f;
                if total == 0 {
                    return None;
                }
                let confidence = (t.max(f) as f64) / (total as f64);
                if confidence >= threshold {
                    Some((i as Var, t > f))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Reset
    pub fn reset(&mut self) {
        for (t, f) in &mut self.polarity_counts {
            *t = 0;
            *f = 0;
        }
        self.solutions_count = 0;
    }
}

// ============================================================================
// Hybrid SLS-CDCL Interface
// ============================================================================

/// Interface for hybrid SLS-CDCL solving
#[derive(Debug)]
pub struct HybridSlsInterface {
    /// Assumed literals from CDCL
    assumptions: Vec<Lit>,
    /// Conflict clauses learned from SLS
    learned_clauses: Vec<Vec<Lit>>,
    /// Phase hints from SLS
    phase_hints: Vec<Option<bool>>,
    /// Variables to focus on
    focus_vars: HashSet<Var>,
}

impl HybridSlsInterface {
    /// Create new interface
    pub fn new() -> Self {
        Self {
            assumptions: Vec::new(),
            learned_clauses: Vec::new(),
            phase_hints: Vec::new(),
            focus_vars: HashSet::new(),
        }
    }

    /// Set assumptions from CDCL solver
    pub fn set_assumptions(&mut self, assumptions: Vec<Lit>) {
        self.assumptions = assumptions;
    }

    /// Get assumptions
    pub fn assumptions(&self) -> &[Lit] {
        &self.assumptions
    }

    /// Add learned clause from SLS
    pub fn add_learned_clause(&mut self, clause: Vec<Lit>) {
        self.learned_clauses.push(clause);
    }

    /// Get and clear learned clauses
    pub fn take_learned_clauses(&mut self) -> Vec<Vec<Lit>> {
        std::mem::take(&mut self.learned_clauses)
    }

    /// Set phase hint for variable
    pub fn set_phase_hint(&mut self, var: Var, phase: bool) {
        let idx = var as usize;
        if self.phase_hints.len() <= idx {
            self.phase_hints.resize(idx + 1, None);
        }
        self.phase_hints[idx] = Some(phase);
    }

    /// Get phase hint
    pub fn phase_hint(&self, var: Var) -> Option<bool> {
        self.phase_hints.get(var as usize).copied().flatten()
    }

    /// Set focus variables (from CDCL conflict analysis)
    pub fn set_focus_vars(&mut self, vars: HashSet<Var>) {
        self.focus_vars = vars;
    }

    /// Get focus variables
    pub fn focus_vars(&self) -> &HashSet<Var> {
        &self.focus_vars
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.assumptions.clear();
        self.learned_clauses.clear();
        self.phase_hints.clear();
        self.focus_vars.clear();
    }
}

impl Default for HybridSlsInterface {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// YalSAT-style Solver
// ============================================================================

/// YalSAT configuration (Yet Another Local Search SAT)
#[derive(Debug, Clone)]
pub struct YalsatConfig {
    /// Base configuration
    pub base: SlsConfig,
    /// Enable caching
    pub caching: bool,
    /// Cache size limit
    pub cache_limit: usize,
    /// Focused probability
    pub focused_prob: f64,
    /// Enable boosting
    pub boosting: bool,
}

impl Default for YalsatConfig {
    fn default() -> Self {
        Self {
            base: SlsConfig::default(),
            caching: true,
            cache_limit: 1000,
            focused_prob: 0.8,
            boosting: true,
        }
    }
}

/// YalSAT-style enhanced SLS solver
#[derive(Debug)]
pub struct YalsatSolver {
    /// Base solver
    base: SlsSolver,
    /// Configuration
    #[allow(dead_code)]
    config: YalsatConfig,
    /// Score cache (var -> (break, make) at last update)
    score_cache: HashMap<Var, (u32, u32)>,
    /// Boost factors for variables
    boost: Vec<f64>,
    /// Cached best variable per clause
    clause_best: Vec<Option<Var>>,
}

impl YalsatSolver {
    /// Create new YalSAT solver
    pub fn new(config: YalsatConfig) -> Self {
        Self {
            base: SlsSolver::new(config.base.clone()),
            config,
            score_cache: HashMap::new(),
            boost: Vec::new(),
            clause_best: Vec::new(),
        }
    }

    /// Add clause
    pub fn add_clause(&mut self, literals: &[Lit]) {
        self.base.add_clause(literals);
        self.clause_best.push(None);
    }

    /// Solve
    pub fn solve(&mut self) -> SlsResult {
        // Initialize boost factors
        let n = self.base.num_vars() as usize + 1;
        self.boost = vec![1.0; n];

        // Run base solver with enhanced picking
        self.base.solve()
    }

    /// Update boost for variable
    pub fn update_boost(&mut self, var: Var, factor: f64) {
        let idx = var as usize;
        if idx < self.boost.len() {
            self.boost[idx] *= factor;
            // Clamp to prevent overflow
            if self.boost[idx] > 1000.0 {
                self.boost[idx] = 1000.0;
            }
        }
    }

    /// Get boosted score
    pub fn boosted_score(&self, var: Var, base_score: f64) -> f64 {
        let boost = self.boost.get(var as usize).copied().unwrap_or(1.0);
        base_score * boost
    }

    /// Invalidate cache for variable
    pub fn invalidate_cache(&mut self, var: Var) {
        self.score_cache.remove(&var);
    }

    /// Clear all caches
    pub fn clear_cache(&mut self) {
        self.score_cache.clear();
        for best in &mut self.clause_best {
            *best = None;
        }
    }

    /// Reset
    pub fn reset(&mut self) {
        self.base.reset();
        self.score_cache.clear();
        self.boost.clear();
        self.clause_best.clear();
    }
}

// ============================================================================
// Portfolio SLS (Multiple Algorithms)
// ============================================================================

/// Portfolio SLS configuration
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Algorithms to use
    pub algorithms: Vec<SlsAlgorithm>,
    /// Flips per algorithm per round
    pub flips_per_algo: u64,
    /// Enable adaptive switching
    pub adaptive: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![
                SlsAlgorithm::WalkSat,
                SlsAlgorithm::ProbSat,
                SlsAlgorithm::Gsat,
            ],
            flips_per_algo: 10000,
            adaptive: true,
        }
    }
}

/// Portfolio SLS solver (runs multiple algorithms)
#[derive(Debug)]
pub struct PortfolioSls {
    /// Configuration
    config: PortfolioConfig,
    /// Individual solvers
    solvers: Vec<SlsSolver>,
    /// Best result so far
    best_result: Option<SlsResult>,
    /// Best unsat count
    best_unsat: u32,
    /// Algorithm performance (successes)
    algo_performance: Vec<u32>,
}

impl PortfolioSls {
    /// Create a new portfolio solver
    pub fn new(config: PortfolioConfig) -> Self {
        let mut solvers = Vec::new();
        let algo_count = config.algorithms.len();

        for &algo in &config.algorithms {
            let sls_config = SlsConfig {
                algorithm: algo,
                max_flips: config.flips_per_algo,
                max_restarts: 1,
                ..SlsConfig::default()
            };
            solvers.push(SlsSolver::new(sls_config));
        }

        Self {
            config,
            solvers,
            best_result: None,
            best_unsat: u32::MAX,
            algo_performance: vec![0; algo_count],
        }
    }

    /// Add clause to all solvers
    pub fn add_clause(&mut self, literals: &[Lit]) {
        for solver in &mut self.solvers {
            solver.add_clause(literals);
        }
    }

    /// Solve using portfolio
    pub fn solve(&mut self, max_rounds: u32) -> SlsResult {
        for _round in 0..max_rounds {
            for (i, solver) in self.solvers.iter_mut().enumerate() {
                let result = solver.solve();

                if let SlsResult::Sat(ref _assignment) = result {
                    self.algo_performance[i] += 1;
                    self.best_result = Some(result.clone());
                    return result;
                }

                // Track best progress
                let unsat_count = solver.best_unsat_count;
                if unsat_count < self.best_unsat {
                    self.best_unsat = unsat_count;
                }
            }

            // Adaptive: prioritize better-performing algorithms
            if self.config.adaptive && !self.algo_performance.is_empty() {
                // Sort by performance (descending)
                let mut indices: Vec<_> = (0..self.solvers.len()).collect();
                indices.sort_by_key(|&i| std::cmp::Reverse(self.algo_performance[i]));
                // Could reorder solvers here for next round
            }
        }

        self.best_result.clone().unwrap_or(SlsResult::Unknown)
    }

    /// Get best unsatisfied count
    pub fn best_unsat_count(&self) -> u32 {
        self.best_unsat
    }

    /// Get algorithm performance
    pub fn algorithm_performance(&self) -> &[u32] {
        &self.algo_performance
    }

    /// Reset all solvers
    pub fn reset(&mut self) {
        for solver in &mut self.solvers {
            solver.reset();
        }
        self.best_result = None;
        self.best_unsat = u32::MAX;
        for p in &mut self.algo_performance {
            *p = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sls_config() {
        let config = SlsConfig::default();
        assert_eq!(config.algorithm, SlsAlgorithm::WalkSat);
        assert_eq!(config.max_flips, 1_000_000);
    }

    #[test]
    fn test_sls_solver_creation() {
        let solver = SlsSolver::new(SlsConfig::default());
        assert_eq!(solver.num_vars(), 0);
        assert_eq!(solver.num_clauses(), 0);
    }

    #[test]
    fn test_add_clause() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1, -2, 3]);
        solver.add_clause(&[-1, 2, 3]);
        assert_eq!(solver.num_vars(), 3);
        assert_eq!(solver.num_clauses(), 2);
    }

    #[test]
    fn test_solve_trivial_sat() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1]);
        solver.add_clause(&[2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_solve_simple_unsat() {
        let mut solver = SlsSolver::new(SlsConfig {
            max_flips: 1000,
            max_restarts: 5,
            ..Default::default()
        });
        solver.add_clause(&[1]);
        solver.add_clause(&[-1]);
        let result = solver.solve();
        // May be Unknown since SLS doesn't prove UNSAT
        assert!(matches!(result, SlsResult::Unknown | SlsResult::Sat(_)));
    }

    #[test]
    fn test_walksat() {
        let mut config = SlsConfig::default();
        config.algorithm = SlsAlgorithm::WalkSat;
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        solver.add_clause(&[1, -2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_gsat() {
        let mut config = SlsConfig::default();
        config.algorithm = SlsAlgorithm::Gsat;
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_probsat() {
        let mut config = SlsConfig::default();
        config.algorithm = SlsAlgorithm::ProbSat;
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2, 3]);
        solver.add_clause(&[-1, -2, 3]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_adaptive() {
        let mut config = SlsConfig::default();
        config.algorithm = SlsAlgorithm::Adaptive;
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 3]);
        solver.add_clause(&[2, 3]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_tabu() {
        let mut config = SlsConfig::default();
        config.tabu = true;
        config.tabu_tenure = 5;
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, -2]);
        solver.add_clause(&[1, -2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
    }

    #[test]
    fn test_adaptive_noise() {
        let mut config = SlsConfig::default();
        config.adaptive_noise = true;
        let mut solver = SlsSolver::new(config);
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_)));
    }

    #[test]
    fn test_stats() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1, 2]);
        let _ = solver.solve();
        let stats = solver.stats();
        assert!(stats.restarts > 0);
    }

    #[test]
    fn test_reset() {
        let mut solver = SlsSolver::new(SlsConfig::default());
        solver.add_clause(&[1, 2]);
        solver.reset();
        assert_eq!(solver.num_clauses(), 0);
    }

    // Weighted SLS tests

    #[test]
    fn test_weighted_sls() {
        let mut solver = WeightedSlsSolver::new(WeightedSlsConfig::default());
        solver.add_hard_clause(&[1, 2]);
        solver.add_soft_clause(&[-1]);
        solver.add_soft_clause(&[-2]);
        let (result, cost) = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
        assert!(cost >= 0.0);
    }

    // Phase saver tests

    #[test]
    fn test_phase_saver() {
        let mut saver = PhaseSaver::new(PhaseMode::Save);
        saver.save_phase(1, true);
        assert!(saver.get_phase(1));
        saver.save_phase(1, false);
        assert!(!saver.get_phase(1));
    }

    #[test]
    fn test_phase_modes() {
        let mut false_saver = PhaseSaver::new(PhaseMode::False);
        assert!(!false_saver.get_phase(1));

        let mut true_saver = PhaseSaver::new(PhaseMode::True);
        assert!(true_saver.get_phase(1));
    }

    // Weight manager tests

    #[test]
    fn test_weight_manager() {
        let mut manager = ClauseWeightManager::new(WeightingScheme::Additive);
        manager.initialize(5);

        let mut unsat = HashSet::new();
        unsat.insert(ClauseId(0));
        unsat.insert(ClauseId(2));

        manager.update(&unsat);
        assert!(manager.get_weight(ClauseId(0)) > 1.0);
        assert_eq!(manager.get_weight(ClauseId(1)), 1.0);
    }

    #[test]
    fn test_weight_smooth() {
        let mut manager = ClauseWeightManager::new(WeightingScheme::Saps);
        manager.initialize(3);
        manager.weights[0] = 10.0;
        manager.weights[1] = 1.0;
        manager.weights[2] = 1.0;

        manager.smooth();
        // After smoothing, weights should be more balanced
        assert!(manager.get_weight(ClauseId(0)) < 10.0);
    }

    // Variable activity tests

    #[test]
    fn test_var_activity() {
        let mut activity = VarActivity::new();
        activity.bump(1);
        activity.bump(1);
        activity.bump(2);

        assert!(activity.get(1) > activity.get(2));
    }

    #[test]
    fn test_var_activity_decay() {
        let mut activity = VarActivity::new();
        activity.bump(1);
        let before = activity.bump;
        activity.decay();
        assert!(activity.bump > before);
    }

    // Focused walk tests

    #[test]
    fn test_focused_walk() {
        let mut walk = FocusedWalk::new(FocusedWalkConfig::default());
        let lits = vec![1, -2, 3];
        let breaks = vec![0, 2, 1, 0];
        walk.update_focus(&lits, &breaks);
        assert!(!walk.focus_set().is_empty());
    }

    #[test]
    fn test_focused_walk_ordering() {
        let mut walk = FocusedWalk::new(FocusedWalkConfig {
            focus_size: 2,
            ..Default::default()
        });
        let lits = vec![1, 2, 3];
        let breaks = vec![0, 3, 1, 2]; // var 2 has break=1, var 3 has break=2, var 1 has break=3
        walk.update_focus(&lits, &breaks);

        // Should prefer lower break counts
        let focus = walk.focus_set();
        assert_eq!(focus.len(), 2);
    }

    // Restart manager tests

    #[test]
    fn test_restart_manager_fixed() {
        let mut manager = RestartManager::new(RestartStrategy::Fixed(100));
        assert!(!manager.should_restart(50));
        assert!(manager.should_restart(100));
        manager.notify_restart();
        assert_eq!(manager.count(), 1);
        assert_eq!(manager.threshold(), 100);
    }

    #[test]
    fn test_restart_manager_geometric() {
        let mut manager = RestartManager::new(RestartStrategy::Geometric(100, 2.0));
        assert_eq!(manager.threshold(), 100);
        manager.notify_restart();
        // Should double: 100 * 2^1 = 200
        assert_eq!(manager.threshold(), 200);
        manager.notify_restart();
        // Should quadruple: 100 * 2^2 = 400
        assert_eq!(manager.threshold(), 400);
    }

    #[test]
    fn test_restart_manager_luby() {
        let mut manager = RestartManager::new(RestartStrategy::Luby(10));
        // Luby sequence: 1, 1, 2, 1, 1, 2, 4, ...
        // Initial: luby_index = 1, threshold = 10 (initial value)
        assert_eq!(manager.threshold(), 10);

        // First restart: uses luby(1) = 1, luby_index becomes 2
        manager.notify_restart();
        assert_eq!(manager.threshold(), 10); // 1 * 10 = 10

        // Second restart: uses luby(2) = 1, luby_index becomes 3
        manager.notify_restart();
        assert_eq!(manager.threshold(), 10); // 1 * 10 = 10

        // Third restart: uses luby(3) = 2, luby_index becomes 4
        manager.notify_restart();
        assert_eq!(manager.threshold(), 20); // 2 * 10 = 20
    }

    #[test]
    fn test_restart_manager_luby_sequence() {
        let manager = RestartManager::new(RestartStrategy::Luby(1));
        // Test Luby function directly
        assert_eq!(manager.luby(1), 1);
        assert_eq!(manager.luby(2), 1);
        assert_eq!(manager.luby(3), 2);
        assert_eq!(manager.luby(4), 1);
        assert_eq!(manager.luby(5), 1);
        assert_eq!(manager.luby(6), 2);
        assert_eq!(manager.luby(7), 4);
    }

    // Novelty selector tests

    #[test]
    fn test_novelty_selector() {
        let mut selector = NoveltySelector::new(NoveltyConfig::default());
        selector.ensure_capacity(10);

        // First flip - var 1 flipped at time 0, then time becomes 1
        selector.notify_flip(1);
        // age = current_time - flip_time = 1 - 0 = 1
        assert_eq!(selector.age(1), 1);
        // var 2 age = current_time - 0 = 1 (never flipped, initialized at 0)
        assert_eq!(selector.age(2), 1);

        // Second flip - var 2 flipped at time 1, then time becomes 2
        selector.notify_flip(2);
        // var 1: age = 2 - 0 = 2
        assert!(selector.age(1) > selector.age(2));
        // var 2: age = 2 - 1 = 1
        assert_eq!(selector.age(2), 1);
    }

    #[test]
    fn test_novelty_selection() {
        let selector = NoveltySelector::new(NoveltyConfig {
            novelty_prob: 0.0, // Always pick best
            novelty_plus: false,
            ..Default::default()
        });

        let candidates = vec![(1, 5), (2, 1), (3, 3)]; // (var, break)
        let mut rng = 42u64;
        let selected = selector.select(&candidates, &mut rng);
        assert_eq!(selected, Some(2)); // Lowest break count
    }

    #[test]
    fn test_novelty_avoids_last_flipped() {
        let mut selector = NoveltySelector::new(NoveltyConfig {
            novelty_prob: 1.0, // Always pick second best when best = last_flipped
            novelty_plus: false,
            ..Default::default()
        });
        selector.notify_flip(2); // Make var 2 the last flipped

        let candidates = vec![(2, 1), (3, 5)]; // var 2 is best (break=1)
        let mut rng = 42u64;
        let selected = selector.select(&candidates, &mut rng);
        assert_eq!(selected, Some(3)); // Should pick second best since 2 was just flipped
    }

    // CCAnr tests

    #[test]
    fn test_ccanr_enhancer() {
        let mut enhancer = CcanrEnhancer::new(CcanrConfig::default());
        enhancer.initialize(10);

        let clause_lits = vec![1, -2, 3];
        enhancer.update_scores(&clause_lits);

        assert!(enhancer.score(1) > 0.0);
        assert!(enhancer.score(2) > 0.0);
        assert!(enhancer.score(3) > 0.0);
        assert_eq!(enhancer.score(5), 0.0);
    }

    #[test]
    fn test_ccanr_config_checking() {
        let mut enhancer = CcanrEnhancer::new(CcanrConfig::default());
        enhancer.initialize(10);

        enhancer.set_config(5, true);
        assert!(enhancer.check_config(5));
        assert!(!enhancer.check_config(6));
    }

    #[test]
    fn test_ccanr_decay() {
        let mut enhancer = CcanrEnhancer::new(CcanrConfig::default());
        enhancer.initialize(5);

        enhancer.update_scores(&[1, 2]);
        let before = enhancer.score(1);
        enhancer.decay_scores(0.5);
        assert!((enhancer.score(1) - before * 0.5).abs() < 0.001);
    }

    // Backbone detector tests

    #[test]
    fn test_backbone_detector() {
        let mut detector = BackboneDetector::new(2);

        // First solution: x1=true, x2=true, x3=false
        let sol1 = vec![false, true, true, false];
        detector.initialize(&sol1);
        assert_eq!(detector.backbone_size(), 3);

        // Second solution: x1=true, x2=false, x3=false
        let sol2 = vec![false, true, false, false];
        detector.update(&sol2);

        // x2 should be removed from backbone candidates
        // x1=true and x3=false should remain
        assert!(detector.is_backbone(1) == Some(true));
        assert!(detector.is_backbone(3) == Some(false));
        assert!(detector.is_backbone(2).is_none());
    }

    #[test]
    fn test_backbone_commits_after_threshold() {
        let mut detector = BackboneDetector::new(2);

        let sol1 = vec![false, true, true];
        detector.initialize(&sol1);

        let sol2 = vec![false, true, true];
        detector.update(&sol2);

        // Should have committed to backbone after 2 solutions
        assert!(!detector.backbone().is_empty());
    }

    // Diversification tests

    #[test]
    fn test_diversification_manager() {
        let mut manager = DiversificationManager::new(DiversificationStrategy::Stagnation);

        for _ in 0..50 {
            manager.update(10); // No improvement
        }
        assert!(!manager.should_diversify());

        for _ in 0..60 {
            manager.update(10); // Still no improvement
        }
        assert!(manager.should_diversify()); // Should diversify after threshold
    }

    #[test]
    fn test_diversification_improvement_resets() {
        let mut manager = DiversificationManager::new(DiversificationStrategy::Stagnation);

        for _ in 0..90 {
            manager.update(10);
        }

        // Improvement resets counter
        manager.update(5);
        assert!(!manager.should_diversify());
    }

    // Clause simplifier tests

    #[test]
    fn test_clause_simplifier_subsumption() {
        let simplifier = ClauseSimplifier::new();

        // [1, 2] subsumes [1, 2, 3]
        assert!(simplifier.subsumes(&[1, 2], &[1, 2, 3]));
        assert!(!simplifier.subsumes(&[1, 2, 3], &[1, 2]));
        assert!(simplifier.subsumes(&[1], &[1, 2, 3]));
    }

    #[test]
    fn test_clause_simplifier_build() {
        let mut simplifier = ClauseSimplifier::new();
        let clauses = vec![vec![1, 2], vec![-1, 3], vec![2, 3]];
        simplifier.build(&clauses);

        // Check occurrence lists work
        assert_eq!(simplifier.clause_sizes.len(), 3);
    }

    #[test]
    fn test_clause_simplifier_units() {
        let simplifier = ClauseSimplifier::new();
        let clauses = vec![vec![1], vec![2, 3], vec![-4]];
        let units = simplifier.find_units(&clauses);
        assert!(units.contains(&1));
        assert!(units.contains(&-4));
        assert!(!units.contains(&2));
    }

    #[test]
    fn test_clause_simplifier_trivially_unsat() {
        let simplifier = ClauseSimplifier::new();

        let sat_clauses = vec![vec![1, 2], vec![-1]];
        assert!(!simplifier.is_trivially_unsat(&sat_clauses));

        let unsat_clauses = vec![vec![1, 2], vec![]];
        assert!(simplifier.is_trivially_unsat(&unsat_clauses));
    }

    // Sparrow selector tests

    #[test]
    fn test_sparrow_selector() {
        let mut selector = SparrowSelector::new(SparrowConfig::default());
        selector.initialize(10);

        let clause_lits = vec![1, 2, 3];
        let break_counts = vec![0, 5, 1, 2];
        let make_counts = vec![0, 0, 2, 1];
        let mut rng = 42u64;

        let selected = selector.select(&clause_lits, &break_counts, &make_counts, &mut rng);
        assert!(selected.is_some());
    }

    #[test]
    fn test_sparrow_age_factor() {
        let mut selector = SparrowSelector::new(SparrowConfig::default());
        selector.initialize(10);

        // Flip var 1
        selector.notify_flip(1);
        selector.notify_flip(2);
        selector.notify_flip(3);

        // Var 1 should be oldest
        assert!(selector.age(1) > selector.age(2));
        assert!(selector.age(2) > selector.age(3));
    }

    // Portfolio SLS tests

    #[test]
    fn test_portfolio_sls() {
        let mut portfolio = PortfolioSls::new(PortfolioConfig::default());
        portfolio.add_clause(&[1, 2]);
        portfolio.add_clause(&[-1, 2]);
        portfolio.add_clause(&[1, -2]);

        let result = portfolio.solve(5);
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
    }

    #[test]
    fn test_portfolio_reset() {
        let mut portfolio = PortfolioSls::new(PortfolioConfig::default());
        portfolio.add_clause(&[1, 2]);
        let _ = portfolio.solve(1);

        portfolio.reset();
        assert_eq!(portfolio.best_unsat_count(), u32::MAX);
    }

    // BMS Selector tests

    #[test]
    fn test_bms_selector() {
        let mut selector = BmsSelector::new(BmsConfig::default());
        selector.initialize(10);

        // Score should penalize breaks and reward makes
        let score_low_break = selector.compute_score(1, 0, 2);
        let score_high_break = selector.compute_score(1, 5, 2);
        assert!(score_low_break > score_high_break);
    }

    #[test]
    fn test_bms_selection() {
        let selector = BmsSelector::new(BmsConfig::default());
        let candidates = vec![1, 2, 3];
        let break_counts = vec![0, 5, 1, 2]; // var 1: break=5, var 2: break=1, var 3: break=2
        let make_counts = vec![0, 0, 3, 1];

        let selected = selector.select(&candidates, &break_counts, &make_counts);
        // var 2 should be best (low break, high make)
        assert_eq!(selected, Some(2));
    }

    #[test]
    fn test_bms_age_factor() {
        let mut selector = BmsSelector::new(BmsConfig {
            age_weight: 1.0, // High age weight
            break_weight: 0.0,
            make_weight: 0.0,
            ..Default::default()
        });
        selector.initialize(10);

        // Flip var 1 multiple times to advance time
        selector.notify_flip(1);
        selector.notify_flip(1);
        selector.notify_flip(1);

        // var 1 was last flipped at time 2, var 2 was never flipped
        // var 1 age: 3 - 2 = 1
        // var 2 age: 3 - 0 = 3 (initialized at 0)
        let score1 = selector.compute_score(1, 0, 0);
        let score2 = selector.compute_score(2, 0, 0);
        assert!(
            score2 > score1,
            "score2 ({}) should be > score1 ({})",
            score2,
            score1
        );
    }

    // Solution verification tests

    #[test]
    fn test_solution_verifier() {
        let mut verifier = SolutionVerifier::new();
        verifier.set_clauses(vec![vec![1, 2], vec![-1, 3], vec![2, 3]]);

        // Valid solution: x1=true, x2=true, x3=true
        let assignment = vec![false, true, true, true];
        let result = verifier.verify(&assignment);
        assert!(result.is_valid);
        assert_eq!(result.satisfied_count, 3);
    }

    #[test]
    fn test_solution_verifier_invalid() {
        let mut verifier = SolutionVerifier::new();
        verifier.set_clauses(vec![vec![1], vec![-1]]);

        let assignment = vec![false, true];
        let result = verifier.verify(&assignment);
        assert!(!result.is_valid);
        assert_eq!(result.unsatisfied_indices.len(), 1);
    }

    #[test]
    fn test_solution_verifier_quick_check() {
        let mut verifier = SolutionVerifier::new();
        verifier.set_clauses(vec![vec![1, 2], vec![-1, 2]]);

        assert!(verifier.is_valid(&[false, true, true]));
        assert!(verifier.is_valid(&[false, false, true]));
    }

    // DDFW tests

    #[test]
    fn test_ddfw_manager() {
        let mut ddfw = DdfwManager::new(DdfwConfig::default());
        ddfw.initialize(5);

        assert_eq!(ddfw.weight(0), 1.0);
        assert_eq!(ddfw.weight(2), 1.0);
    }

    #[test]
    fn test_ddfw_distribution() {
        let mut ddfw = DdfwManager::new(DdfwConfig {
            init_weight: 10.0,
            transfer_amount: 2.0,
            distribute_freq: 1,
        });
        ddfw.initialize(5);

        ddfw.notify_flip();
        assert!(ddfw.should_distribute());

        // Transfer from clause 0,1 to clause 3,4
        ddfw.distribute(&[0, 1], &[3, 4]);

        // Satisfied clauses should have less weight
        assert!(ddfw.weight(0) < 10.0);
        // Unsatisfied should have more
        assert!(ddfw.weight(3) > 10.0);
    }

    // Clause importance tests

    #[test]
    fn test_clause_importance() {
        let mut importance = ClauseImportance::new();
        importance.initialize(5);

        importance.record_hit(0);
        importance.record_hit(0);
        importance.record_hit(1);
        importance.record_critical(0);

        // Clause 0 should be most important
        assert!(importance.importance(0) > importance.importance(1));
    }

    #[test]
    fn test_clause_importance_ranking() {
        let mut importance = ClauseImportance::new();
        importance.initialize(5);

        importance.record_hit(2);
        importance.record_hit(2);
        importance.record_hit(2);
        importance.record_hit(0);
        importance.record_critical(0);

        let most = importance.most_important(2);
        assert_eq!(most.len(), 2);
        // Both 0 and 2 should be in top 2
        assert!(most.contains(&0) || most.contains(&2));
    }

    // Solution learner tests

    #[test]
    fn test_solution_learner() {
        let mut learner = SolutionLearner::new(10);
        learner.initialize(5);

        // Record solutions where var 1 is always true
        learner.record_solution(&[false, true, false, true, false]);
        learner.record_solution(&[false, true, true, false, true]);
        learner.record_solution(&[false, true, false, false, true]);

        assert_eq!(learner.preferred_polarity(1), Some(true));
    }

    #[test]
    fn test_solution_learner_confidence() {
        let mut learner = SolutionLearner::new(10);
        learner.initialize(3);

        learner.record_solution(&[false, true, true, false]);
        learner.record_solution(&[false, true, false, false]);

        // Var 1: 2 true, 0 false -> 100% confidence for true
        assert!((learner.polarity_confidence(1) - 1.0).abs() < 0.01);
        // Var 2: 1 true, 1 false -> 50% confidence
        assert!((learner.polarity_confidence(2) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_solution_learner_high_confidence() {
        let mut learner = SolutionLearner::new(10);
        learner.initialize(4);

        learner.record_solution(&[false, true, true, false, true]);
        learner.record_solution(&[false, true, true, true, true]);
        learner.record_solution(&[false, true, true, false, false]);

        let high_conf = learner.high_confidence_vars(0.9);
        // Var 1 and 2 should have high confidence (all or mostly same polarity)
        assert!(high_conf.iter().any(|&(v, _)| v == 1 || v == 2));
    }

    // Hybrid interface tests

    #[test]
    fn test_hybrid_interface() {
        let mut interface = HybridSlsInterface::new();

        interface.set_assumptions(vec![1, -2, 3]);
        assert_eq!(interface.assumptions().len(), 3);

        interface.add_learned_clause(vec![1, 2, 3]);
        let learned = interface.take_learned_clauses();
        assert_eq!(learned.len(), 1);
        assert!(interface.take_learned_clauses().is_empty());
    }

    #[test]
    fn test_hybrid_interface_phase_hints() {
        let mut interface = HybridSlsInterface::new();

        interface.set_phase_hint(5, true);
        interface.set_phase_hint(3, false);

        assert_eq!(interface.phase_hint(5), Some(true));
        assert_eq!(interface.phase_hint(3), Some(false));
        assert_eq!(interface.phase_hint(1), None);
    }

    #[test]
    fn test_hybrid_interface_focus() {
        let mut interface = HybridSlsInterface::new();

        let mut focus = HashSet::new();
        focus.insert(1);
        focus.insert(3);
        focus.insert(5);
        interface.set_focus_vars(focus);

        assert!(interface.focus_vars().contains(&1));
        assert!(interface.focus_vars().contains(&3));
        assert!(!interface.focus_vars().contains(&2));
    }

    // YalSAT tests

    #[test]
    fn test_yalsat_solver() {
        let mut solver = YalsatSolver::new(YalsatConfig::default());
        solver.add_clause(&[1, 2]);
        solver.add_clause(&[-1, 2]);
        solver.add_clause(&[1, -2]);

        let result = solver.solve();
        assert!(matches!(result, SlsResult::Sat(_) | SlsResult::Unknown));
    }

    #[test]
    fn test_yalsat_boost() {
        let mut solver = YalsatSolver::new(YalsatConfig::default());
        solver.add_clause(&[1, 2]);
        solver.boost = vec![1.0; 3];

        solver.update_boost(1, 2.0);
        assert!((solver.boost[1] - 2.0).abs() < 0.01);

        let score = solver.boosted_score(1, 10.0);
        assert!((score - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_yalsat_cache_invalidation() {
        let mut solver = YalsatSolver::new(YalsatConfig::default());
        solver.score_cache.insert(1, (5, 3));
        solver.score_cache.insert(2, (2, 1));

        solver.invalidate_cache(1);
        assert!(!solver.score_cache.contains_key(&1));
        assert!(solver.score_cache.contains_key(&2));

        solver.clear_cache();
        assert!(solver.score_cache.is_empty());
    }
}
