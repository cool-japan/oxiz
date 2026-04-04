//! Core SLS types: literals, variables, clauses, configuration, and the main SlsSolver.

#[allow(unused_imports)]
use crate::prelude::*;
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
pub(super) struct SlsClause {
    /// Literals in the clause
    pub literals: Vec<Lit>,
    /// Number of true literals (satisfied count)
    pub sat_count: u32,
    /// Weight for weighted SLS
    pub weight: f64,
}

impl SlsClause {
    pub fn new(literals: Vec<Lit>) -> Self {
        Self {
            literals,
            sat_count: 0,
            weight: 1.0,
        }
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if clause is satisfied
    #[allow(dead_code)]
    pub fn is_satisfied(&self) -> bool {
        self.sat_count > 0
    }
}

/// SLS Solver
#[derive(Debug)]
pub struct SlsSolver {
    /// Configuration
    pub(super) config: SlsConfig,
    /// Clauses
    pub(super) clauses: Vec<SlsClause>,
    /// Number of variables
    pub(super) num_vars: u32,
    /// Current assignment (true/false for each variable)
    pub(super) assignment: Vec<bool>,
    /// Occurrence lists: variable -> clauses containing it positive
    pub(super) pos_occs: HashMap<Var, Vec<ClauseId>>,
    /// Occurrence lists: variable -> clauses containing it negative
    pub(super) neg_occs: HashMap<Var, Vec<ClauseId>>,
    /// Unsatisfied clauses
    pub(super) unsat_clauses: HashSet<ClauseId>,
    /// Break count for each variable (how many clauses would become unsat if flipped)
    pub(super) break_count: Vec<u32>,
    /// Make count for each variable (how many clauses would become sat if flipped)
    pub(super) make_count: Vec<u32>,
    /// Tabu list: variable -> flip when it becomes allowed again
    pub(super) tabu_list: Vec<u64>,
    /// Current flip number
    pub(super) current_flip: u64,
    /// Random state
    pub(super) rng_state: u64,
    /// Statistics
    pub(super) stats: SlsStats,
    /// Best assignment found
    pub(super) best_assignment: Vec<bool>,
    /// Best number of unsatisfied clauses
    pub(super) best_unsat_count: u32,
    /// Current noise (for adaptive)
    pub(super) current_noise: f64,
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
    pub(super) fn initialize(&mut self) {
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
    pub(super) fn random_assignment(&mut self) {
        for i in 1..=self.num_vars as usize {
            self.assignment[i] = self.random_bool();
        }
    }

    /// Initialize satisfaction counts for all clauses
    pub(super) fn initialize_sat_counts(&mut self) {
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
    pub(super) fn update_all_counts(&mut self) {
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
    pub(super) fn walksat_pick(&mut self) -> Option<Var> {
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
    pub(super) fn gsat_pick(&mut self) -> Option<Var> {
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
    pub(super) fn probsat_pick(&mut self) -> Option<Var> {
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
    pub(super) fn adaptive_pick(&mut self) -> Option<Var> {
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
    pub(super) fn flip_variable(&mut self, var: Var) {
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
    pub(super) fn adjust_noise(&mut self) {
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
    pub(super) fn random_unsat_clause(&mut self) -> Option<ClauseId> {
        if self.unsat_clauses.is_empty() {
            return None;
        }
        let idx = self.random_usize(self.unsat_clauses.len());
        self.unsat_clauses.iter().nth(idx).copied()
    }

    /// Random number generator (xorshift64)
    pub(super) fn random_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    pub(super) fn random_bool(&mut self) -> bool {
        self.random_u64() & 1 == 0
    }

    pub(super) fn random_float(&mut self) -> f64 {
        (self.random_u64() as f64) / (u64::MAX as f64)
    }

    pub(super) fn random_usize(&mut self, max: usize) -> usize {
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
