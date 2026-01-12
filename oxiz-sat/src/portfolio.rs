//! Parallel portfolio solving
//!
//! This module implements parallel portfolio solving where multiple solver
//! instances run concurrently with different configurations, and the first
//! to find a result wins.

use crate::literal::Lit;
use crate::solver::{RestartStrategy, Solver, SolverConfig, SolverResult};

#[cfg(test)]
use crate::literal::Var;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Result from a portfolio solver thread
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    /// Solver ID that found the result
    pub solver_id: usize,
    /// The result
    pub result: SolverResult,
    /// Time taken to solve
    pub time: Duration,
    /// Number of conflicts
    pub conflicts: u64,
}

/// Configuration for a portfolio solver instance
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Base solver configuration
    pub solver_config: SolverConfig,
    /// Solver ID
    pub id: usize,
    /// Optional name for this configuration
    pub name: String,
}

impl PortfolioConfig {
    /// Create a new portfolio configuration
    pub fn new(id: usize, name: String, solver_config: SolverConfig) -> Self {
        Self {
            solver_config,
            id,
            name,
        }
    }

    /// Create focused configuration (exploitative)
    pub fn focused(id: usize) -> Self {
        let config = SolverConfig {
            restart_strategy: RestartStrategy::Glucose,
            random_polarity_prob: 0.01,
            ..Default::default()
        };
        Self::new(id, format!("Focused-{}", id), config)
    }

    /// Create diversified configuration (explorative)
    pub fn diversified(id: usize) -> Self {
        let config = SolverConfig {
            restart_strategy: RestartStrategy::Luby,
            restart_interval: 100,
            random_polarity_prob: 0.1,
            ..Default::default()
        };
        Self::new(id, format!("Diversified-{}", id), config)
    }

    /// Create aggressive restart configuration
    pub fn aggressive_restart(id: usize) -> Self {
        let config = SolverConfig {
            restart_strategy: RestartStrategy::Geometric,
            restart_interval: 50,
            restart_multiplier: 1.1,
            random_polarity_prob: 0.05,
            ..Default::default()
        };
        Self::new(id, format!("AggressiveRestart-{}", id), config)
    }

    /// Create conservative configuration
    pub fn conservative(id: usize) -> Self {
        let config = SolverConfig {
            restart_strategy: RestartStrategy::Luby,
            restart_interval: 200,
            random_polarity_prob: 0.02,
            ..Default::default()
        };
        Self::new(id, format!("Conservative-{}", id), config)
    }
}

/// Portfolio solver that runs multiple solvers in parallel
pub struct PortfolioSolver {
    /// Number of solver threads
    num_solvers: usize,
    /// Portfolio configurations
    configs: Vec<PortfolioConfig>,
    /// Timeout for solving (None = no timeout)
    timeout: Option<Duration>,
}

impl PortfolioSolver {
    /// Create a new portfolio solver
    pub fn new(num_solvers: usize) -> Self {
        Self {
            num_solvers,
            configs: Vec::new(),
            timeout: None,
        }
    }

    /// Set timeout for solving
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = Some(timeout);
    }

    /// Add a configuration to the portfolio
    pub fn add_config(&mut self, config: PortfolioConfig) {
        self.configs.push(config);
    }

    /// Generate default diverse portfolio
    pub fn generate_default_portfolio(&mut self) {
        self.configs.clear();
        let strategies_per_type = self.num_solvers / 4;
        let mut id = 0;

        for _ in 0..strategies_per_type {
            self.configs.push(PortfolioConfig::focused(id));
            id += 1;
        }

        for _ in 0..strategies_per_type {
            self.configs.push(PortfolioConfig::diversified(id));
            id += 1;
        }

        for _ in 0..strategies_per_type {
            self.configs.push(PortfolioConfig::aggressive_restart(id));
            id += 1;
        }

        // Fill remaining with conservative
        while id < self.num_solvers {
            self.configs.push(PortfolioConfig::conservative(id));
            id += 1;
        }
    }

    /// Solve a formula in parallel using the portfolio
    pub fn solve(&self, clauses: Vec<Vec<Lit>>, num_vars: usize) -> Option<PortfolioResult> {
        if self.configs.is_empty() {
            return None;
        }

        let (tx, rx): (Sender<PortfolioResult>, Receiver<PortfolioResult>) = channel();
        let should_stop = Arc::new(Mutex::new(false));
        let mut handles: Vec<JoinHandle<()>> = Vec::new();

        // Spawn solver threads
        for config in &self.configs {
            let tx = tx.clone();
            let should_stop = Arc::clone(&should_stop);
            let clauses = clauses.clone();
            let config = config.clone();

            let handle = thread::spawn(move || {
                // Create solver with configuration
                let mut solver = Solver::with_config(config.solver_config);

                // Add variables
                for _ in 0..num_vars {
                    solver.new_var();
                }

                // Add clauses
                for clause in clauses {
                    solver.add_clause(clause);
                }

                // Solve
                let thread_start = Instant::now();

                // Check if another solver found the answer
                if *should_stop.lock().unwrap() {
                    return;
                }

                let result = solver.solve();

                // We found a result!
                let elapsed = thread_start.elapsed();
                let stats = solver.stats();

                let portfolio_result = PortfolioResult {
                    solver_id: config.id,
                    result,
                    time: elapsed,
                    conflicts: stats.conflicts,
                };

                // Notify other threads to stop
                *should_stop.lock().unwrap() = true;

                // Send result
                let _ = tx.send(portfolio_result);
            });

            handles.push(handle);
        }

        // Drop the sender so the channel closes when all threads finish
        drop(tx);

        // Wait for first result or timeout
        let result = if let Some(timeout) = self.timeout {
            match rx.recv_timeout(timeout) {
                Ok(result) => Some(result),
                Err(_) => {
                    // Timeout - signal all threads to stop
                    *should_stop.lock().unwrap() = true;
                    None
                }
            }
        } else {
            rx.recv().ok()
        };

        // Wait for all threads to finish
        for handle in handles {
            let _ = handle.join();
        }

        result
    }

    /// Solve with assumptions
    pub fn solve_with_assumptions(
        &self,
        clauses: Vec<Vec<Lit>>,
        num_vars: usize,
        assumptions: &[Lit],
    ) -> Option<PortfolioResult> {
        if self.configs.is_empty() {
            return None;
        }

        let (tx, rx): (Sender<PortfolioResult>, Receiver<PortfolioResult>) = channel();
        let should_stop = Arc::new(Mutex::new(false));
        let mut handles: Vec<JoinHandle<()>> = Vec::new();

        // Spawn solver threads
        for config in &self.configs {
            let tx = tx.clone();
            let should_stop = Arc::clone(&should_stop);
            let clauses = clauses.clone();
            let assumptions = assumptions.to_vec();
            let config = config.clone();

            let handle = thread::spawn(move || {
                let mut solver = Solver::with_config(config.solver_config);

                for _ in 0..num_vars {
                    solver.new_var();
                }

                for clause in clauses {
                    solver.add_clause(clause);
                }

                let thread_start = Instant::now();

                if *should_stop.lock().unwrap() {
                    return;
                }

                let (result, _core) = solver.solve_with_assumptions(&assumptions);

                let elapsed = thread_start.elapsed();
                let stats = solver.stats();

                let portfolio_result = PortfolioResult {
                    solver_id: config.id,
                    result,
                    time: elapsed,
                    conflicts: stats.conflicts,
                };

                *should_stop.lock().unwrap() = true;
                let _ = tx.send(portfolio_result);
            });

            handles.push(handle);
        }

        drop(tx);

        let result = if let Some(timeout) = self.timeout {
            match rx.recv_timeout(timeout) {
                Ok(result) => Some(result),
                Err(_) => {
                    *should_stop.lock().unwrap() = true;
                    None
                }
            }
        } else {
            rx.recv().ok()
        };

        for handle in handles {
            let _ = handle.join();
        }

        result
    }

    /// Get number of solvers in the portfolio
    pub fn num_solvers(&self) -> usize {
        self.num_solvers
    }

    /// Get configurations
    pub fn configs(&self) -> &[PortfolioConfig] {
        &self.configs
    }
}

impl Default for PortfolioSolver {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let mut portfolio = Self::new(num_cpus);
        portfolio.generate_default_portfolio();
        portfolio
    }
}

/// Portfolio statistics
#[derive(Debug, Clone, Default)]
pub struct PortfolioStats {
    /// Total solves attempted
    pub total_solves: u64,
    /// Number of SAT results
    pub sat_results: u64,
    /// Number of UNSAT results
    pub unsat_results: u64,
    /// Number of timeouts
    pub timeouts: u64,
    /// Total time spent solving
    pub total_time: Duration,
    /// Solver ID frequency (which solver found results most often)
    pub solver_wins: Vec<(usize, u64)>,
}

impl PortfolioStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a solve result
    pub fn record(&mut self, result: Option<&PortfolioResult>) {
        self.total_solves += 1;

        if let Some(r) = result {
            match r.result {
                SolverResult::Sat => self.sat_results += 1,
                SolverResult::Unsat => self.unsat_results += 1,
                SolverResult::Unknown => {}
            }
            self.total_time += r.time;

            // Update solver wins
            let mut found = false;
            for (id, count) in &mut self.solver_wins {
                if *id == r.solver_id {
                    *count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                self.solver_wins.push((r.solver_id, 1));
            }
        } else {
            self.timeouts += 1;
        }
    }

    /// Get average solve time
    pub fn avg_time(&self) -> Duration {
        if self.total_solves == 0 {
            return Duration::from_secs(0);
        }
        self.total_time / self.total_solves as u32
    }

    /// Get best performing solver ID
    pub fn best_solver(&self) -> Option<usize> {
        self.solver_wins
            .iter()
            .max_by_key(|(_, count)| count)
            .map(|(id, _)| *id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_config() {
        let config = PortfolioConfig::focused(0);
        assert_eq!(config.id, 0);
        assert!(config.name.contains("Focused"));
    }

    #[test]
    fn test_portfolio_solver_creation() {
        let mut portfolio = PortfolioSolver::new(4);
        portfolio.generate_default_portfolio();
        assert_eq!(portfolio.configs().len(), 4);
    }

    #[test]
    fn test_portfolio_default() {
        let portfolio = PortfolioSolver::default();
        assert!(!portfolio.configs().is_empty());
    }

    #[test]
    fn test_portfolio_stats() {
        let mut stats = PortfolioStats::new();
        assert_eq!(stats.total_solves, 0);

        stats.record(None); // Timeout
        assert_eq!(stats.timeouts, 1);
    }

    #[test]
    fn test_simple_sat_parallel() {
        let mut portfolio = PortfolioSolver::new(2);
        portfolio.generate_default_portfolio();
        portfolio.set_timeout(Duration::from_secs(5));

        // Simple SAT formula: (x0 ∨ x1)
        let clauses = vec![vec![Lit::pos(Var(0)), Lit::pos(Var(1))]];

        let result = portfolio.solve(clauses, 2);
        assert!(result.is_some());

        if let Some(r) = result {
            assert!(matches!(r.result, SolverResult::Sat));
        }
    }
}
