//! Hybrid solver combining Stochastic Local Search (SLS) with exact MaxSAT.
//!
//! This hybrid approach uses SLS to quickly find good approximate solutions,
//! then switches to exact methods to prove optimality or improve the solution.
//!
//! Reference: Z3's portfolio-based optimization strategies

use crate::maxsat::{MaxSatConfig, MaxSatError, MaxSatResult, MaxSatSolver, SoftClause, Weight};
use crate::sls::{SlsConfig, SlsError, SlsSolver};
use thiserror::Error;

/// Errors from hybrid solver
#[derive(Error, Debug)]
pub enum HybridError {
    /// SLS phase error
    #[error("SLS error: {0}")]
    SlsError(#[from] SlsError),
    /// Exact solver error
    #[error("exact solver error: {0}")]
    ExactError(#[from] MaxSatError),
    /// No solution found
    #[error("no solution found")]
    NoSolution,
}

/// Hybrid solver strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridStrategy {
    /// SLS first, then exact if time permits
    SlsFirst,
    /// Parallel SLS and exact (simulated)
    Parallel,
    /// SLS to get upper bound, exact to prove optimality
    SlsGuided,
}

/// Configuration for hybrid solver
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// SLS configuration
    pub sls_config: SlsConfig,
    /// Exact MaxSAT configuration
    pub maxsat_config: MaxSatConfig,
    /// Hybrid strategy to use
    pub strategy: HybridStrategy,
    /// Maximum time for SLS phase (in iterations)
    pub sls_max_iterations: u32,
    /// Switch to exact if SLS finds solution within this cost
    pub switch_threshold: Option<Weight>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            sls_config: SlsConfig::default(),
            maxsat_config: MaxSatConfig::default(),
            strategy: HybridStrategy::SlsFirst,
            sls_max_iterations: 5000,
            switch_threshold: None,
        }
    }
}

/// Statistics from hybrid solving
#[derive(Debug, Clone, Default)]
pub struct HybridStats {
    /// SLS iterations used
    pub sls_iterations: u32,
    /// Exact solver iterations used
    pub exact_iterations: u32,
    /// Best cost from SLS
    pub sls_best_cost: Option<Weight>,
    /// Final cost
    pub final_cost: Option<Weight>,
    /// Whether exact solver was used
    pub used_exact: bool,
}

/// Hybrid MaxSAT solver
pub struct HybridSolver {
    /// Soft clauses
    soft_clauses: Vec<SoftClause>,
    /// Configuration
    config: HybridConfig,
    /// Statistics
    stats: HybridStats,
    /// Best cost found
    best_cost: Weight,
}

impl HybridSolver {
    /// Create a new hybrid solver
    pub fn new() -> Self {
        Self::with_config(HybridConfig::default())
    }

    /// Create a new hybrid solver with configuration
    pub fn with_config(config: HybridConfig) -> Self {
        Self {
            soft_clauses: Vec::new(),
            config,
            stats: HybridStats::default(),
            best_cost: Weight::Infinite,
        }
    }

    /// Add a soft clause
    pub fn add_soft(&mut self, clause: SoftClause) {
        self.soft_clauses.push(clause);
    }

    /// Solve using hybrid approach
    pub fn solve(&mut self) -> Result<MaxSatResult, HybridError> {
        match self.config.strategy {
            HybridStrategy::SlsFirst => self.solve_sls_first(),
            HybridStrategy::Parallel => self.solve_parallel(),
            HybridStrategy::SlsGuided => self.solve_sls_guided(),
        }
    }

    /// SLS-first strategy: run SLS, then exact if needed
    fn solve_sls_first(&mut self) -> Result<MaxSatResult, HybridError> {
        // Phase 1: Run SLS to get a good approximate solution
        let sls_result = self.run_sls_phase();

        match sls_result {
            Ok(cost) => {
                self.stats.sls_best_cost = Some(cost.clone());
                self.best_cost = cost.clone();

                // Check if we should switch to exact
                if let Some(ref threshold) = self.config.switch_threshold
                    && cost <= *threshold
                {
                    // Cost is good enough, try exact solver to prove optimality
                    return self.run_exact_phase();
                }

                self.stats.final_cost = Some(cost);
                Ok(MaxSatResult::Satisfiable)
            }
            Err(_) => {
                // SLS failed, try exact solver
                self.run_exact_phase()
            }
        }
    }

    /// Parallel strategy (simulated): run both and take best result
    fn solve_parallel(&mut self) -> Result<MaxSatResult, HybridError> {
        // In a real implementation, this would run in parallel threads
        // For now, we simulate by running SLS first with a time limit
        let sls_result = self.run_sls_phase();

        // Also run exact solver
        let exact_result = self.run_exact_phase();

        // Return the better result
        match (sls_result, exact_result) {
            (Ok(_sls_cost), Ok(_)) => {
                // Exact solver gives provably optimal result
                Ok(MaxSatResult::Optimal)
            }
            (Ok(sls_cost), Err(_)) => {
                self.best_cost = sls_cost.clone();
                self.stats.final_cost = Some(sls_cost);
                Ok(MaxSatResult::Satisfiable)
            }
            (Err(_), Ok(_)) => Ok(MaxSatResult::Optimal),
            (Err(e), Err(_)) => Err(e.into()),
        }
    }

    /// SLS-guided strategy: use SLS to guide exact solver
    fn solve_sls_guided(&mut self) -> Result<MaxSatResult, HybridError> {
        // First, run SLS to get an upper bound
        if let Ok(cost) = self.run_sls_phase() {
            self.stats.sls_best_cost = Some(cost.clone());
            self.best_cost = cost;
        }

        // Then run exact solver, which can use the SLS bound
        self.run_exact_phase()
    }

    /// Run SLS phase
    fn run_sls_phase(&mut self) -> Result<Weight, SlsError> {
        let mut sls = SlsSolver::with_config(self.config.sls_config.clone());

        // Add hard clauses (none in this simplified version)
        // Add soft clauses
        for clause in &self.soft_clauses {
            sls.add_soft(clause.clone());
        }

        // Solve
        sls.solve()?;

        self.stats.sls_iterations = sls.stats().iterations;
        Ok(sls.best_cost().clone())
    }

    /// Run exact MaxSAT phase
    fn run_exact_phase(&mut self) -> Result<MaxSatResult, HybridError> {
        self.stats.used_exact = true;

        let mut exact = MaxSatSolver::with_config(self.config.maxsat_config.clone());

        // Add soft clauses
        for clause in &self.soft_clauses {
            exact.add_soft_weighted(clause.lits.iter().copied(), clause.weight.clone());
        }

        // Solve
        let result = exact.solve()?;

        self.best_cost = exact.cost().clone();
        self.stats.final_cost = Some(self.best_cost.clone());
        self.stats.exact_iterations = exact.stats().cores_extracted;

        Ok(result)
    }

    /// Get the best cost found
    pub fn best_cost(&self) -> &Weight {
        &self.best_cost
    }

    /// Get statistics
    pub fn stats(&self) -> &HybridStats {
        &self.stats
    }
}

impl Default for HybridSolver {
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
    fn test_hybrid_solver_new() {
        let solver = HybridSolver::new();
        assert_eq!(*solver.best_cost(), Weight::Infinite);
        assert!(!solver.stats().used_exact);
    }

    #[test]
    fn test_hybrid_simple() {
        let mut solver = HybridSolver::new();

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

        // Should find a solution with cost 1
        assert_eq!(*solver.best_cost(), Weight::from(1));
    }

    #[test]
    fn test_hybrid_config() {
        let config = HybridConfig {
            sls_max_iterations: 1000,
            strategy: HybridStrategy::SlsGuided,
            ..Default::default()
        };

        let solver = HybridSolver::with_config(config);
        assert_eq!(solver.config.sls_max_iterations, 1000);
        assert_eq!(solver.config.strategy, HybridStrategy::SlsGuided);
    }
}
