//! Stochastic Local Search for MaxSAT.
//!
//! SLS is a fast incomplete method for finding approximate solutions to MaxSAT
//! problems. It uses local search with randomization to escape local optima.
//!
//! Reference: Z3's local search implementations and SLS MaxSAT solvers

use crate::maxsat::{SoftClause, Weight};
use rand::Rng;
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Errors from SLS
#[derive(Error, Debug)]
pub enum SlsError {
    /// No solution found within limits
    #[error("no solution found")]
    NoSolution,
    /// Iteration limit exceeded
    #[error("iteration limit exceeded")]
    IterationLimit,
}

/// Result of SLS
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlsResult {
    /// Found solution (may not be optimal)
    Satisfiable,
    /// Could not find solution
    Unknown,
}

/// Configuration for SLS
#[derive(Debug, Clone)]
pub struct SlsConfig {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Maximum number of flips
    pub max_flips: u32,
    /// Noise parameter (0.0 to 1.0)
    pub noise: f64,
    /// Random walk probability
    pub random_walk_prob: f64,
    /// Tabu tenure
    pub tabu_tenure: u32,
    /// Random seed
    pub random_seed: u64,
}

impl Default for SlsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            max_flips: 100000,
            noise: 0.2,
            random_walk_prob: 0.05,
            tabu_tenure: 10,
            random_seed: 42,
        }
    }
}

/// Statistics from SLS
#[derive(Debug, Clone, Default)]
pub struct SlsStats {
    /// Number of iterations
    pub iterations: u32,
    /// Number of variable flips
    pub flips: u32,
    /// Best cost found
    pub best_cost: Option<Weight>,
    /// Number of restarts
    pub restarts: u32,
}

/// Variable assignment
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Variable values (variable index -> bool)
    values: Vec<bool>,
}

impl Assignment {
    /// Create a random assignment
    fn random(num_vars: usize, rng: &mut rand::rngs::StdRng) -> Self {
        Self {
            values: (0..num_vars).map(|_| rng.random_bool(0.5)).collect(),
        }
    }

    /// Get value of a variable
    fn get(&self, var: usize) -> bool {
        self.values.get(var).copied().unwrap_or(false)
    }

    /// Flip a variable
    fn flip(&mut self, var: usize) {
        if var < self.values.len() {
            self.values[var] = !self.values[var];
        }
    }
}

/// Clause evaluation cache
#[derive(Debug)]
struct ClauseCache {
    /// Whether each hard clause is satisfied
    hard_satisfied: Vec<bool>,
    /// Whether each soft clause is satisfied
    soft_satisfied: Vec<bool>,
}

impl ClauseCache {
    fn new(num_hard: usize, num_soft: usize) -> Self {
        Self {
            hard_satisfied: vec![false; num_hard],
            soft_satisfied: vec![false; num_soft],
        }
    }

    /// Check if all hard clauses are satisfied
    fn all_hard_satisfied(&self) -> bool {
        self.hard_satisfied.iter().all(|&x| x)
    }
}

/// SLS MaxSAT solver
pub struct SlsSolver {
    /// Hard clauses (as literal vectors)
    hard_clauses: Vec<Vec<i32>>,
    /// Soft clauses
    soft_clauses: Vec<SoftClause>,
    /// Configuration
    config: SlsConfig,
    /// Statistics
    stats: SlsStats,
    /// Random number generator
    rng: rand::rngs::StdRng,
    /// Current assignment
    assignment: Option<Assignment>,
    /// Best assignment found
    best_assignment: Option<Assignment>,
    /// Best cost found
    best_cost: Weight,
    /// Clause cache
    cache: Option<ClauseCache>,
    /// Tabu list (variable -> iteration when it was flipped)
    tabu: FxHashMap<usize, u32>,
    /// Number of variables
    num_vars: usize,
}

impl SlsSolver {
    /// Create a new SLS solver
    pub fn new() -> Self {
        Self::with_config(SlsConfig::default())
    }

    /// Create a new SLS solver with configuration
    pub fn with_config(config: SlsConfig) -> Self {
        use rand::SeedableRng;
        let seed = config.random_seed;
        Self {
            hard_clauses: Vec::new(),
            soft_clauses: Vec::new(),
            config,
            stats: SlsStats::default(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            assignment: None,
            best_assignment: None,
            best_cost: Weight::Infinite,
            cache: None,
            tabu: FxHashMap::default(),
            num_vars: 0,
        }
    }

    /// Add a hard clause
    pub fn add_hard(&mut self, lits: Vec<i32>) {
        // Update num_vars first
        for &lit in &lits {
            let var = lit.unsigned_abs() as usize;
            if var > self.num_vars {
                self.num_vars = var;
            }
        }
        self.hard_clauses.push(lits);
    }

    /// Add a soft clause
    pub fn add_soft(&mut self, clause: SoftClause) {
        self.soft_clauses.push(clause);
    }

    /// Initialize solver
    fn initialize(&mut self) {
        // Update num_vars from soft clauses
        for clause in &self.soft_clauses {
            for lit in &clause.lits {
                let var = lit.to_dimacs().unsigned_abs() as usize;
                if var > self.num_vars {
                    self.num_vars = var;
                }
            }
        }

        // Create random initial assignment
        self.assignment = Some(Assignment::random(self.num_vars + 1, &mut self.rng));

        // Initialize cache
        self.cache = Some(ClauseCache::new(
            self.hard_clauses.len(),
            self.soft_clauses.len(),
        ));

        // Evaluate initial assignment
        self.update_cache();
        let cost = self.compute_cost();
        self.best_cost = cost.clone();
        self.best_assignment = self.assignment.clone();
        self.stats.best_cost = Some(cost);
    }

    /// Solve using SLS
    pub fn solve(&mut self) -> Result<SlsResult, SlsError> {
        self.initialize();

        for iter in 0..self.config.max_iterations {
            self.stats.iterations = iter + 1;

            // Update cache
            self.update_cache();

            // Check if all hard clauses are satisfied
            if !self
                .cache
                .as_ref()
                .expect("cache initialized at solve start")
                .all_hard_satisfied()
            {
                // Pick a violated hard clause and flip a variable from it
                self.flip_from_violated_hard();
                continue;
            }

            // All hard clauses satisfied, try to improve soft clause satisfaction
            let cost = self.compute_cost();

            // Update best if improved
            if cost < self.best_cost {
                self.best_cost = cost.clone();
                self.best_assignment = self.assignment.clone();
                self.stats.best_cost = Some(cost.clone());
            }

            // Check if solved optimally (all clauses satisfied)
            if cost.is_zero() {
                return Ok(SlsResult::Satisfiable);
            }

            // Pick a move
            if self.rng.random_bool(self.config.random_walk_prob) {
                // Random walk: flip a random variable from a violated soft clause
                self.random_walk();
            } else {
                // Greedy or noise move
                self.greedy_move();
            }

            self.stats.flips += 1;

            if self.stats.flips >= self.config.max_flips {
                break;
            }
        }

        // Restore best assignment
        if let Some(best) = &self.best_assignment {
            self.assignment = Some(best.clone());
        }

        Ok(SlsResult::Satisfiable)
    }

    /// Update the clause satisfaction cache
    fn update_cache(&mut self) {
        let assignment = self
            .assignment
            .as_ref()
            .expect("assignment initialized at solve start");

        // Update hard clauses
        for (idx, clause) in self.hard_clauses.iter().enumerate() {
            let satisfied = Self::is_clause_satisfied_static(clause, assignment);
            if let Some(cache) = self.cache.as_mut() {
                cache.hard_satisfied[idx] = satisfied;
            }
        }

        // Update soft clauses
        for (idx, clause) in self.soft_clauses.iter().enumerate() {
            let lits: Vec<i32> = clause.lits.iter().map(|l| l.to_dimacs()).collect();
            let satisfied = Self::is_clause_satisfied_static(&lits, assignment);
            if let Some(cache) = self.cache.as_mut() {
                cache.soft_satisfied[idx] = satisfied;
            }
        }
    }

    /// Check if a clause is satisfied (static version)
    fn is_clause_satisfied_static(clause: &[i32], assignment: &Assignment) -> bool {
        clause.iter().any(|&lit| {
            let var = lit.unsigned_abs() as usize;
            let val = assignment.get(var);
            if lit > 0 { val } else { !val }
        })
    }

    /// Compute current cost (sum of weights of violated soft clauses)
    fn compute_cost(&self) -> Weight {
        let cache = self
            .cache
            .as_ref()
            .expect("cache initialized at solve start");
        let mut cost = Weight::zero();

        for (idx, &satisfied) in cache.soft_satisfied.iter().enumerate() {
            if !satisfied {
                cost = cost.add(&self.soft_clauses[idx].weight);
            }
        }

        cost
    }

    /// Flip a variable from a violated hard clause
    fn flip_from_violated_hard(&mut self) {
        let cache = self
            .cache
            .as_ref()
            .expect("cache initialized at solve start");

        // Find violated hard clauses
        let violated: Vec<usize> = cache
            .hard_satisfied
            .iter()
            .enumerate()
            .filter(|(_, sat)| !**sat)
            .map(|(idx, _)| idx)
            .collect();

        if violated.is_empty() {
            return;
        }

        // Pick a random violated clause
        let clause_idx = violated[self.rng.random_range(0..violated.len())];
        let clause = &self.hard_clauses[clause_idx];

        // Pick a random variable from the clause
        let lit_idx = self.rng.random_range(0..clause.len());
        let var = clause[lit_idx].unsigned_abs() as usize;

        // Flip it
        if let Some(ref mut assignment) = self.assignment {
            assignment.flip(var);
            self.tabu.insert(var, self.stats.flips);
        }
    }

    /// Random walk: flip a random variable from a violated soft clause
    fn random_walk(&mut self) {
        let cache = self
            .cache
            .as_ref()
            .expect("cache initialized at solve start");

        // Find violated soft clauses
        let violated: Vec<usize> = cache
            .soft_satisfied
            .iter()
            .enumerate()
            .filter(|(_, sat)| !**sat)
            .map(|(idx, _)| idx)
            .collect();

        if violated.is_empty() {
            return;
        }

        // Pick a random violated soft clause
        let clause_idx = violated[self.rng.random_range(0..violated.len())];
        let clause = &self.soft_clauses[clause_idx];

        if clause.lits.is_empty() {
            return;
        }

        // Pick a random variable from it
        let lit_idx = self.rng.random_range(0..clause.lits.len());
        let var = clause.lits[lit_idx].to_dimacs().unsigned_abs() as usize;

        // Flip it
        if let Some(ref mut assignment) = self.assignment {
            assignment.flip(var);
            self.tabu.insert(var, self.stats.flips);
        }
    }

    /// Greedy move with noise
    fn greedy_move(&mut self) {
        // Try flipping each variable and pick the one with best improvement
        let mut best_var = None;
        let mut best_delta = Weight::Infinite;

        for var in 1..=self.num_vars {
            // Skip tabu variables
            if let Some(&tabu_iter) = self.tabu.get(&var)
                && self.stats.flips - tabu_iter < self.config.tabu_tenure
            {
                continue;
            }

            // Try flipping this variable
            if let Some(assignment) = self.assignment.as_mut() {
                assignment.flip(var);
            }
            self.update_cache();
            let new_cost = self.compute_cost();
            let delta = new_cost.clone(); // Simplified: should compute actual delta

            if delta < best_delta {
                best_delta = delta;
                best_var = Some(var);
            }

            // Flip back
            if let Some(assignment) = self.assignment.as_mut() {
                assignment.flip(var);
            }
        }

        // Apply best move (or noise move)
        if self.rng.random_bool(self.config.noise) {
            // Noise move: pick random variable
            let var = self.rng.random_range(1..=self.num_vars);
            if let Some(assignment) = self.assignment.as_mut() {
                assignment.flip(var);
                self.tabu.insert(var, self.stats.flips);
            }
        } else if let Some(var) = best_var {
            // Greedy move
            if let Some(assignment) = self.assignment.as_mut() {
                assignment.flip(var);
                self.tabu.insert(var, self.stats.flips);
            }
        }
    }

    /// Get the best cost found
    pub fn best_cost(&self) -> &Weight {
        &self.best_cost
    }

    /// Get statistics
    pub fn stats(&self) -> &SlsStats {
        &self.stats
    }

    /// Get the best assignment
    pub fn best_assignment(&self) -> Option<&Assignment> {
        self.best_assignment.as_ref()
    }
}

impl Default for SlsSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maxsat::SoftId;
    use oxiz_sat::Lit;

    #[test]
    fn test_sls_solver_new() {
        let solver = SlsSolver::new();
        assert_eq!(solver.stats().iterations, 0);
        assert_eq!(*solver.best_cost(), Weight::Infinite);
    }

    #[test]
    fn test_sls_simple() {
        let mut solver = SlsSolver::new();

        // Add soft clauses
        solver.add_soft(SoftClause::new(
            SoftId(0),
            [Lit::from_dimacs(1)],
            Weight::from(1),
        ));
        solver.add_soft(SoftClause::new(
            SoftId(1),
            [Lit::from_dimacs(-1)],
            Weight::from(1),
        ));

        let result = solver.solve();
        assert!(result.is_ok());

        // Should find a solution with cost 1 (one clause must be violated)
        assert_eq!(*solver.best_cost(), Weight::from(1));
    }

    #[test]
    fn test_sls_config() {
        let config = SlsConfig {
            max_iterations: 5000,
            max_flips: 50000,
            noise: 0.3,
            random_walk_prob: 0.1,
            tabu_tenure: 15,
            random_seed: 123,
        };

        let solver = SlsSolver::with_config(config);
        assert_eq!(solver.config.max_iterations, 5000);
        assert_eq!(solver.config.noise, 0.3);
    }
}
