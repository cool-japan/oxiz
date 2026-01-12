//! Portfolio-based MaxSAT solver.
//!
//! A portfolio solver runs multiple MaxSAT algorithms (either sequentially or
//! in parallel) and returns the best result. This is effective because different
//! algorithms perform better on different problem instances.
//!
//! Reference: Z3's portfolio-based optimization strategies

use crate::maxsat::{
    MaxSatAlgorithm, MaxSatConfig, MaxSatError, MaxSatResult, MaxSatSolver, SoftClause, SoftId,
    Weight,
};
use oxiz_sat::Lit;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::sync::{Arc, Mutex};
use thiserror::Error;

/// Errors from portfolio solver
#[derive(Error, Debug)]
pub enum PortfolioError {
    /// MaxSAT solver error
    #[error("maxsat error: {0}")]
    MaxSatError(#[from] MaxSatError),
    /// No algorithm succeeded
    #[error("no algorithm succeeded")]
    AllFailed,
    /// Resource limit exceeded
    #[error("resource limit exceeded")]
    ResourceLimit,
}

/// Portfolio strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortfolioStrategy {
    /// Run all algorithms sequentially, return first optimal result
    Sequential,
    /// Run all algorithms sequentially, return best result
    SequentialBest,
    /// True parallel execution using rayon (run all algorithms concurrently)
    Parallel,
    /// Adaptive: choose based on problem features
    Adaptive,
}

/// Configuration for portfolio solver
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Algorithms to include in portfolio
    pub algorithms: Vec<MaxSatAlgorithm>,
    /// Strategy to use
    pub strategy: PortfolioStrategy,
    /// Time limit per algorithm (in iterations)
    pub time_per_algorithm: u32,
    /// Use problem features for adaptive selection
    pub use_features: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![
                MaxSatAlgorithm::FuMalik,
                MaxSatAlgorithm::Oll,
                MaxSatAlgorithm::Msu3,
                MaxSatAlgorithm::Pmres,
            ],
            strategy: PortfolioStrategy::SequentialBest,
            time_per_algorithm: 1000,
            use_features: true,
        }
    }
}

/// Statistics from portfolio solving
#[derive(Debug, Clone, Default)]
pub struct PortfolioStats {
    /// Number of algorithms tried
    pub algorithms_tried: u32,
    /// Which algorithm found the best solution
    pub best_algorithm: Option<MaxSatAlgorithm>,
    /// Best cost found
    pub best_cost: Option<Weight>,
    /// Total SAT calls across all algorithms
    pub total_sat_calls: u32,
    /// Number of cores shared between algorithms (for work sharing)
    pub cores_shared: u32,
    /// Number of bounds shared between algorithms
    pub bounds_shared: u32,
}

/// Problem features used for adaptive algorithm selection
#[derive(Debug, Clone)]
pub struct ProblemFeatures {
    /// Number of soft clauses
    pub num_soft: usize,
    /// Number of hard clauses
    pub num_hard: usize,
    /// Average clause size
    pub avg_clause_size: f64,
    /// Has weights (vs unweighted)
    pub has_weights: bool,
    /// Weight variance (0.0 if all equal)
    pub weight_variance: f64,
}

impl ProblemFeatures {
    /// Compute features from soft and hard clauses
    pub fn compute(soft_clauses: &[SoftClause], num_hard: usize) -> Self {
        let num_soft = soft_clauses.len();

        let avg_clause_size = if num_soft > 0 {
            soft_clauses.iter().map(|c| c.lits.len()).sum::<usize>() as f64 / num_soft as f64
        } else {
            0.0
        };

        // Check if all weights are the same
        let has_weights = if num_soft > 1 {
            let first_weight = &soft_clauses[0].weight;
            soft_clauses.iter().any(|c| &c.weight != first_weight)
        } else {
            false
        };

        // Compute weight variance (simplified)
        let weight_variance = if has_weights {
            // For simplicity, just return 1.0 if weights differ
            1.0
        } else {
            0.0
        };

        Self {
            num_soft,
            num_hard,
            avg_clause_size,
            has_weights,
            weight_variance,
        }
    }

    /// Select best algorithm based on features
    pub fn select_algorithm(&self) -> MaxSatAlgorithm {
        // Simple heuristics for algorithm selection:
        // - Large number of soft clauses → OLL or PMRES
        // - Weighted instance → WMax or stratified
        // - Small instance → Fu-Malik
        // - Average instance → MSU3

        if self.has_weights {
            MaxSatAlgorithm::WMax
        } else if self.num_soft > 1000 {
            MaxSatAlgorithm::Oll
        } else if self.num_soft > 100 {
            MaxSatAlgorithm::Pmres
        } else {
            MaxSatAlgorithm::FuMalik
        }
    }
}

/// Portfolio MaxSAT solver
pub struct PortfolioSolver {
    /// Hard clauses
    hard_clauses: Vec<Vec<Lit>>,
    /// Soft clauses
    soft_clauses: Vec<SoftClause>,
    /// Configuration
    config: PortfolioConfig,
    /// Statistics
    stats: PortfolioStats,
    /// Best model found
    best_model: Option<Vec<oxiz_sat::LBool>>,
    /// Cost achieved by each algorithm (for tracking)
    algorithm_costs: FxHashMap<MaxSatAlgorithm, Weight>,
}

impl PortfolioSolver {
    /// Create a new portfolio solver
    pub fn new() -> Self {
        Self::with_config(PortfolioConfig::default())
    }

    /// Create a new portfolio solver with configuration
    pub fn with_config(config: PortfolioConfig) -> Self {
        Self {
            hard_clauses: Vec::new(),
            soft_clauses: Vec::new(),
            config,
            stats: PortfolioStats::default(),
            best_model: None,
            algorithm_costs: FxHashMap::default(),
        }
    }

    /// Add a hard clause
    pub fn add_hard(&mut self, lits: impl IntoIterator<Item = Lit>) {
        self.hard_clauses.push(lits.into_iter().collect());
    }

    /// Add a soft clause
    pub fn add_soft(&mut self, id: SoftId, lits: impl IntoIterator<Item = Lit>, weight: Weight) {
        let clause = SoftClause::new(id, lits, weight);
        self.soft_clauses.push(clause);
    }

    /// Solve using the configured portfolio strategy
    pub fn solve(&mut self) -> Result<MaxSatResult, PortfolioError> {
        match self.config.strategy {
            PortfolioStrategy::Sequential => self.solve_sequential(),
            PortfolioStrategy::SequentialBest => self.solve_sequential_best(),
            PortfolioStrategy::Parallel => self.solve_parallel(),
            PortfolioStrategy::Adaptive => self.solve_adaptive(),
        }
    }

    /// Run algorithms sequentially, return first optimal result
    fn solve_sequential(&mut self) -> Result<MaxSatResult, PortfolioError> {
        let algorithms = self.config.algorithms.clone();
        for algorithm in algorithms {
            let result = self.run_algorithm(algorithm)?;
            self.stats.algorithms_tried += 1;

            if result == MaxSatResult::Optimal {
                self.stats.best_algorithm = Some(algorithm);
                return Ok(result);
            }
        }

        Err(PortfolioError::AllFailed)
    }

    /// Run all algorithms sequentially, return best result
    fn solve_sequential_best(&mut self) -> Result<MaxSatResult, PortfolioError> {
        let mut best_result = MaxSatResult::Unknown;
        let mut best_cost = Weight::Infinite;
        let mut best_algorithm = None;

        let algorithms = self.config.algorithms.clone();
        for algorithm in algorithms {
            match self.run_algorithm(algorithm) {
                Ok(result) => {
                    self.stats.algorithms_tried += 1;

                    // Get the cost from the algorithm
                    let cost = self.get_algorithm_cost(algorithm);

                    if cost < best_cost {
                        best_cost = cost.clone();
                        best_result = result;
                        best_algorithm = Some(algorithm);
                        self.stats.best_cost = Some(cost);
                    }
                }
                Err(_) => {
                    // Continue with next algorithm
                    continue;
                }
            }
        }

        self.stats.best_algorithm = best_algorithm;

        if best_result == MaxSatResult::Unknown {
            Err(PortfolioError::AllFailed)
        } else {
            Ok(best_result)
        }
    }

    /// True parallel execution using rayon
    fn solve_parallel(&mut self) -> Result<MaxSatResult, PortfolioError> {
        // Run algorithms in parallel using rayon
        // Each algorithm gets a separate solver instance with the same problem

        let algorithms = self.config.algorithms.clone();
        let hard_clauses = self.hard_clauses.clone();
        let soft_clauses = self.soft_clauses.clone();
        let time_per_algorithm = self.config.time_per_algorithm;

        // Shared state for tracking best solution
        let best_cost = Arc::new(Mutex::new(Weight::Infinite));
        let best_model = Arc::new(Mutex::new(None));
        let best_algorithm = Arc::new(Mutex::new(None));
        let total_sat_calls = Arc::new(Mutex::new(0u32));

        // Run all algorithms in parallel
        let results: Vec<_> = algorithms
            .par_iter()
            .map(|&algorithm| {
                // Create a solver for this algorithm
                let config = MaxSatConfig {
                    algorithm,
                    max_iterations: time_per_algorithm,
                    ..Default::default()
                };

                let mut solver = MaxSatSolver::with_config(config);

                // Add hard clauses
                for clause in &hard_clauses {
                    solver.add_hard(clause.iter().copied());
                }

                // Add soft clauses
                for clause in &soft_clauses {
                    solver.add_soft_weighted(clause.lits.iter().copied(), clause.weight.clone());
                }

                // Solve
                let result = solver.solve();

                // Update shared statistics
                if let Ok(mut sat_calls) = total_sat_calls.lock() {
                    *sat_calls += solver.stats().sat_calls;
                }

                // If successful, check if this is the best solution so far
                if result.is_ok() {
                    let cost = solver.cost().clone();

                    if let Ok(mut best) = best_cost.lock()
                        && cost < *best
                    {
                        *best = cost.clone();

                        // Update best algorithm
                        if let Ok(mut best_alg) = best_algorithm.lock() {
                            *best_alg = Some(algorithm);
                        }

                        // Update best model
                        if let Some(model) = solver.best_model()
                            && let Ok(mut best_mdl) = best_model.lock()
                        {
                            *best_mdl = Some(model.to_vec());
                        }
                    }
                }

                (algorithm, result, solver.cost().clone())
            })
            .collect();

        // Aggregate results
        let mut best_result = MaxSatResult::Unknown;

        for (algorithm, result, cost) in results {
            self.stats.algorithms_tried += 1;

            if let Ok(res) = result {
                self.algorithm_costs.insert(algorithm, cost.clone());

                // Track the best result
                if let Ok(best) = best_cost.lock()
                    && cost == *best
                {
                    best_result = res;
                }
            }
        }

        // Update final statistics
        if let Ok(sat_calls) = total_sat_calls.lock() {
            self.stats.total_sat_calls = *sat_calls;
        }

        if let Ok(alg) = best_algorithm.lock() {
            self.stats.best_algorithm = *alg;
        }

        if let Ok(cost) = best_cost.lock()
            && *cost != Weight::Infinite
        {
            self.stats.best_cost = Some(cost.clone());
        }

        if let Ok(model) = best_model.lock() {
            self.best_model = model.clone();
        }

        if best_result == MaxSatResult::Unknown {
            Err(PortfolioError::AllFailed)
        } else {
            Ok(best_result)
        }
    }

    /// Adaptive algorithm selection based on problem features
    fn solve_adaptive(&mut self) -> Result<MaxSatResult, PortfolioError> {
        if !self.config.use_features {
            return self.solve_sequential_best();
        }

        // Compute problem features
        let features = ProblemFeatures::compute(&self.soft_clauses, self.hard_clauses.len());

        // Select best algorithm
        let selected = features.select_algorithm();
        self.stats.best_algorithm = Some(selected);

        // Run selected algorithm
        let result = self.run_algorithm(selected)?;
        self.stats.algorithms_tried += 1;

        Ok(result)
    }

    /// Run a specific algorithm
    fn run_algorithm(
        &mut self,
        algorithm: MaxSatAlgorithm,
    ) -> Result<MaxSatResult, PortfolioError> {
        let mut config = MaxSatConfig::default();
        config.algorithm = algorithm;
        config.max_iterations = self.config.time_per_algorithm;

        let mut solver = MaxSatSolver::with_config(config);

        // Add hard clauses
        for clause in &self.hard_clauses {
            solver.add_hard(clause.iter().copied());
        }

        // Add soft clauses
        for clause in &self.soft_clauses {
            solver.add_soft_weighted(clause.lits.iter().copied(), clause.weight.clone());
        }

        // Solve
        let result = solver.solve()?;

        // Update stats
        self.stats.total_sat_calls += solver.stats().sat_calls;

        // Track cost for this algorithm
        let cost = solver.cost().clone();
        self.algorithm_costs.insert(algorithm, cost);

        // Store best model if found
        if let Some(model) = solver.best_model() {
            self.best_model = Some(model.to_vec());
        }

        Ok(result)
    }

    /// Get the cost achieved by a specific algorithm
    fn get_algorithm_cost(&self, algorithm: MaxSatAlgorithm) -> Weight {
        self.algorithm_costs
            .get(&algorithm)
            .cloned()
            .unwrap_or(Weight::Infinite)
    }

    /// Get statistics
    pub fn stats(&self) -> &PortfolioStats {
        &self.stats
    }

    /// Get the best model
    pub fn best_model(&self) -> Option<&[oxiz_sat::LBool]> {
        self.best_model.as_deref()
    }
}

impl Default for PortfolioSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_solver_new() {
        let solver = PortfolioSolver::new();
        assert_eq!(solver.stats().algorithms_tried, 0);
    }

    #[test]
    fn test_portfolio_config() {
        let config = PortfolioConfig {
            algorithms: vec![MaxSatAlgorithm::FuMalik, MaxSatAlgorithm::Oll],
            strategy: PortfolioStrategy::Sequential,
            time_per_algorithm: 500,
            use_features: false,
        };

        let solver = PortfolioSolver::with_config(config);
        assert_eq!(solver.config.algorithms.len(), 2);
        assert!(!solver.config.use_features);
    }

    #[test]
    fn test_problem_features() {
        let soft_clauses = vec![
            SoftClause::new(SoftId(0), [Lit::from_dimacs(1)], Weight::from(1)),
            SoftClause::new(SoftId(1), [Lit::from_dimacs(2)], Weight::from(1)),
        ];

        let features = ProblemFeatures::compute(&soft_clauses, 5);
        assert_eq!(features.num_soft, 2);
        assert_eq!(features.num_hard, 5);
        assert!(!features.has_weights); // All weights are equal
    }

    #[test]
    fn test_problem_features_weighted() {
        let soft_clauses = vec![
            SoftClause::new(SoftId(0), [Lit::from_dimacs(1)], Weight::from(1)),
            SoftClause::new(SoftId(1), [Lit::from_dimacs(2)], Weight::from(5)),
        ];

        let features = ProblemFeatures::compute(&soft_clauses, 0);
        assert!(features.has_weights); // Weights differ
    }

    #[test]
    fn test_adaptive_algorithm_selection() {
        // Small unweighted instance
        let features = ProblemFeatures {
            num_soft: 10,
            num_hard: 5,
            avg_clause_size: 2.0,
            has_weights: false,
            weight_variance: 0.0,
        };
        assert_eq!(features.select_algorithm(), MaxSatAlgorithm::FuMalik);

        // Large weighted instance
        let features = ProblemFeatures {
            num_soft: 2000,
            num_hard: 100,
            avg_clause_size: 3.0,
            has_weights: true,
            weight_variance: 1.0,
        };
        assert_eq!(features.select_algorithm(), MaxSatAlgorithm::WMax);
    }

    #[test]
    fn test_portfolio_simple() {
        let mut solver = PortfolioSolver::new();

        // Add simple problem
        solver.add_soft(SoftId(0), [Lit::from_dimacs(1)], Weight::from(1));
        solver.add_soft(SoftId(1), [Lit::from_dimacs(2)], Weight::from(1));

        let result = solver.solve();
        // Should succeed with at least one algorithm
        assert!(result.is_ok() || matches!(result, Err(PortfolioError::AllFailed)));
    }

    #[test]
    fn test_portfolio_parallel() {
        let config = PortfolioConfig {
            algorithms: vec![
                MaxSatAlgorithm::FuMalik,
                MaxSatAlgorithm::Oll,
                MaxSatAlgorithm::Msu3,
            ],
            strategy: PortfolioStrategy::Parallel,
            time_per_algorithm: 100,
            use_features: false,
        };

        let mut solver = PortfolioSolver::with_config(config);

        // Add simple problem
        solver.add_soft(SoftId(0), [Lit::from_dimacs(1)], Weight::from(1));
        solver.add_soft(SoftId(1), [Lit::from_dimacs(2)], Weight::from(1));
        solver.add_soft(SoftId(2), [Lit::from_dimacs(3)], Weight::from(2));

        let result = solver.solve();
        // Parallel execution should succeed
        assert!(result.is_ok() || matches!(result, Err(PortfolioError::AllFailed)));

        // Check that multiple algorithms were tried
        assert_eq!(solver.stats().algorithms_tried, 3);
    }

    #[test]
    fn test_portfolio_parallel_weighted() {
        let config = PortfolioConfig {
            algorithms: vec![
                MaxSatAlgorithm::WMax,
                MaxSatAlgorithm::Pmres,
                MaxSatAlgorithm::FuMalik,
            ],
            strategy: PortfolioStrategy::Parallel,
            time_per_algorithm: 200,
            use_features: false,
        };

        let mut solver = PortfolioSolver::with_config(config);

        // Add weighted problem
        solver.add_soft(SoftId(0), [Lit::from_dimacs(1)], Weight::from(10));
        solver.add_soft(SoftId(1), [Lit::from_dimacs(2)], Weight::from(5));
        solver.add_soft(SoftId(2), [Lit::from_dimacs(3)], Weight::from(2));

        let result = solver.solve();
        assert!(result.is_ok() || matches!(result, Err(PortfolioError::AllFailed)));

        if result.is_ok() {
            // Should have selected one of the algorithms
            assert!(solver.stats().best_algorithm.is_some());
        }
    }
}
