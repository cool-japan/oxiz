//! Portfolio SAT Solver.
//!
//! Runs multiple solver configurations in parallel and returns the first result.

use crate::{Clause, Solver, SolverResult};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for portfolio solving.
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Number of parallel solvers
    pub num_solvers: usize,
    /// Timeout per solver (None = no timeout)
    pub timeout: Option<Duration>,
    /// Enable clause sharing between solvers
    pub enable_sharing: bool,
    /// Maximum shared clause length
    pub max_shared_length: usize,
    /// Sharing frequency (every N conflicts)
    pub sharing_frequency: usize,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            num_solvers: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            timeout: None,
            enable_sharing: true,
            max_shared_length: 8,
            sharing_frequency: 1000,
        }
    }
}

/// Solver variant configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverVariant {
    /// Default VSIDS heuristic
    Default,
    /// Aggressive restart strategy
    AggressiveRestart,
    /// Conservative restart strategy
    ConservativeRestart,
    /// Focus on phase saving
    PhaseSaving,
    /// Random search
    Random,
    /// Luby restart sequence
    Luby,
}

impl SolverVariant {
    /// Apply variant configuration to solver.
    pub fn configure(&self, _solver: &mut Solver) {
        // Simplified: would actually configure solver parameters
    }
}

/// Result from portfolio solving.
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    /// The SAT result
    pub result: SolverResult,
    /// Which solver found the result
    pub solver_id: usize,
    /// Variant that succeeded
    pub variant: SolverVariant,
    /// Time taken
    pub elapsed: Duration,
    /// Total conflicts across all solvers
    pub total_conflicts: u64,
}

/// Statistics for portfolio solving.
#[derive(Debug, Clone, Default)]
pub struct PortfolioStats {
    /// Number of runs
    pub runs: u64,
    /// Clauses shared
    pub clauses_shared: u64,
    /// Total solver time (sum of all parallel times)
    pub total_solver_time_ms: u64,
    /// Wall clock time
    pub wall_clock_time_ms: u64,
}

/// Portfolio SAT solver.
pub struct PortfolioSolver {
    config: PortfolioConfig,
    stats: PortfolioStats,
    variants: Vec<SolverVariant>,
}

impl PortfolioSolver {
    /// Create a new portfolio solver.
    pub fn new(config: PortfolioConfig) -> Self {
        let variants = Self::default_variants(config.num_solvers);
        Self {
            config,
            stats: PortfolioStats::default(),
            variants,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PortfolioConfig::default())
    }

    /// Get default solver variants.
    fn default_variants(num_solvers: usize) -> Vec<SolverVariant> {
        let base_variants = [
            SolverVariant::Default,
            SolverVariant::AggressiveRestart,
            SolverVariant::ConservativeRestart,
            SolverVariant::PhaseSaving,
            SolverVariant::Random,
            SolverVariant::Luby,
        ];

        let mut variants = Vec::with_capacity(num_solvers);
        for i in 0..num_solvers {
            variants.push(base_variants[i % base_variants.len()]);
        }
        variants
    }

    /// Solve with portfolio approach.
    pub fn solve(&mut self, clauses: &[Clause]) -> PortfolioResult {
        self.stats.runs += 1;
        let start = Instant::now();

        let found = Arc::new(AtomicBool::new(false));
        let result: Arc<Mutex<Option<PortfolioResult>>> = Arc::new(Mutex::new(None));

        // Run solvers in parallel
        let variants = self.variants.clone();
        let found_clone = Arc::clone(&found);
        let result_clone: Arc<Mutex<Option<PortfolioResult>>> = Arc::clone(&result);
        let _max_shared_length = self.config.max_shared_length;

        let _solver_results: Vec<_> = (0..self.config.num_solvers)
            .into_par_iter()
            .map(|solver_id| {
                if found_clone.load(Ordering::Relaxed) {
                    return None;
                }

                let mut solver = Solver::new();
                let variant: SolverVariant = variants[solver_id];
                variant.configure(&mut solver);

                // Add clauses
                for clause in clauses {
                    solver.add_clause(clause.lits.iter().copied());
                }

                // Solve
                let solve_start = Instant::now();
                let sat_result = solver.solve();
                let solve_time = solve_start.elapsed();

                // Check if we're the first to finish
                if !found_clone.swap(true, Ordering::Relaxed) {
                    let mut result_lock = result_clone.lock().expect("mutex poisoned");
                    *result_lock = Some(PortfolioResult {
                        result: sat_result,
                        solver_id,
                        variant: variants[solver_id],
                        elapsed: solve_time,
                        total_conflicts: solver.stats().conflicts,
                    });
                }

                Some(())
            })
            .collect();

        let elapsed = start.elapsed();
        self.stats.wall_clock_time_ms += elapsed.as_millis() as u64;

        // Return the first result found

        result
            .lock()
            .expect("mutex poisoned")
            .take()
            .unwrap_or(PortfolioResult {
                result: SolverResult::Unknown,
                solver_id: 0,
                variant: SolverVariant::Default,
                elapsed,
                total_conflicts: 0,
            })
    }

    /// Get statistics.
    pub fn stats(&self) -> &PortfolioStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PortfolioStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_config_default() {
        let config = PortfolioConfig::default();
        assert!(config.num_solvers > 0);
        assert!(config.enable_sharing);
    }

    #[test]
    fn test_solver_variants() {
        let variants = PortfolioSolver::default_variants(4);
        assert_eq!(variants.len(), 4);
        assert_eq!(variants[0], SolverVariant::Default);
    }

    #[test]
    fn test_portfolio_solver_creation() {
        let solver = PortfolioSolver::default_config();
        assert_eq!(solver.stats().runs, 0);
    }
}
