//! Portfolio solver for running multiple Spacer strategies.
//!
//! The portfolio solver runs multiple solving strategies concurrently
//! and returns the first successful result. This improves robustness
//! by trying different parameter configurations.
//!
//! Reference: Z3's portfolio mode and strategy scheduling

use crate::chc::ChcSystem;
use crate::pdr::{Spacer, SpacerConfig, SpacerError, SpacerResult};
use oxiz_core::TermManager;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur in the portfolio solver
#[derive(Error, Debug)]
pub enum PortfolioError {
    /// All strategies failed
    #[error("all strategies failed")]
    AllFailed,
    /// Timeout occurred
    #[error("timeout after {0:?}")]
    Timeout(Duration),
    /// Spacer error
    #[error("spacer error: {0}")]
    Spacer(#[from] SpacerError),
}

/// A solving strategy with configuration
#[derive(Debug, Clone)]
pub struct Strategy {
    /// Name of the strategy
    pub name: String,
    /// Configuration for this strategy
    pub config: SpacerConfig,
    /// Time budget for this strategy (None = unlimited)
    pub time_budget: Option<Duration>,
}

impl Strategy {
    /// Create a new strategy
    pub fn new(name: impl Into<String>, config: SpacerConfig) -> Self {
        Self {
            name: name.into(),
            config,
            time_budget: None,
        }
    }

    /// Set time budget for this strategy
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = Some(budget);
        self
    }

    /// Default aggressive strategy
    /// Uses small resource limits for quick exploration
    pub fn aggressive() -> Self {
        Self::new(
            "aggressive",
            SpacerConfig {
                max_level: 50,
                max_pobs: 1000,
                max_smt_queries: 10000,
                use_inductive_gen: false,
                use_cegar: false,
                verbosity: 0,
            },
        )
        .with_time_budget(Duration::from_secs(5))
    }

    /// Default conservative strategy
    /// Uses larger resource limits and more advanced features
    pub fn conservative() -> Self {
        Self::new(
            "conservative",
            SpacerConfig {
                max_level: 1000,
                max_pobs: 100000,
                max_smt_queries: 10_000_000,
                use_inductive_gen: true,
                use_cegar: true,
                verbosity: 0,
            },
        )
        .with_time_budget(Duration::from_secs(60))
    }

    /// Balanced strategy (default)
    pub fn balanced() -> Self {
        Self::new(
            "balanced",
            SpacerConfig {
                max_level: 200,
                max_pobs: 10000,
                max_smt_queries: 100000,
                use_inductive_gen: true,
                use_cegar: false,
                verbosity: 0,
            },
        )
        .with_time_budget(Duration::from_secs(20))
    }

    /// BMC-like strategy (shallow but wide exploration)
    pub fn bmc_like() -> Self {
        Self::new(
            "bmc",
            SpacerConfig {
                max_level: 100,
                max_pobs: 5000,
                max_smt_queries: 50000,
                use_inductive_gen: false,
                use_cegar: false,
                verbosity: 0,
            },
        )
        .with_time_budget(Duration::from_secs(10))
    }
}

/// Result from a portfolio run
#[derive(Debug, Clone)]
pub struct PortfolioResult {
    /// The result from the winning strategy
    pub result: SpacerResult,
    /// Name of the strategy that succeeded
    pub strategy_name: String,
    /// Time taken by the winning strategy
    pub time_taken: Duration,
    /// Number of strategies attempted
    pub strategies_tried: usize,
}

/// Portfolio solver that runs multiple strategies
pub struct PortfolioSolver {
    /// Strategies to try
    strategies: Vec<Strategy>,
    /// Global timeout for all strategies
    global_timeout: Option<Duration>,
}

impl PortfolioSolver {
    /// Create a new portfolio solver with default strategies
    pub fn new() -> Self {
        Self {
            strategies: vec![
                Strategy::aggressive(),
                Strategy::balanced(),
                Strategy::conservative(),
            ],
            global_timeout: None,
        }
    }

    /// Create an empty portfolio (no default strategies)
    pub fn empty() -> Self {
        Self {
            strategies: Vec::new(),
            global_timeout: None,
        }
    }

    /// Add a strategy to the portfolio
    pub fn add_strategy(&mut self, strategy: Strategy) {
        self.strategies.push(strategy);
    }

    /// Set global timeout for the entire portfolio run
    pub fn with_global_timeout(mut self, timeout: Duration) -> Self {
        self.global_timeout = Some(timeout);
        self
    }

    /// Run the portfolio solver sequentially
    ///
    /// Tries each strategy in order until one succeeds or all fail.
    /// This is simpler than parallel execution and avoids resource contention.
    pub fn solve_sequential(
        &self,
        terms: &mut TermManager,
        system: &ChcSystem,
    ) -> Result<PortfolioResult, PortfolioError> {
        let global_start = Instant::now();
        let mut strategies_tried = 0;

        for strategy in &self.strategies {
            // Check global timeout
            if let Some(global_timeout) = self.global_timeout
                && global_start.elapsed() >= global_timeout
            {
                return Err(PortfolioError::Timeout(global_timeout));
            }

            strategies_tried += 1;
            let strategy_start = Instant::now();

            tracing::info!("Trying strategy '{}' ({})", strategy.name, strategies_tried);

            // Create a solver with this strategy's configuration
            let mut spacer = Spacer::with_config(terms, system, strategy.config.clone());

            // Try to solve with this strategy
            match spacer.solve() {
                Ok(SpacerResult::Safe) => {
                    let time_taken = strategy_start.elapsed();
                    tracing::info!("Strategy '{}' succeeded in {:?}", strategy.name, time_taken);
                    return Ok(PortfolioResult {
                        result: SpacerResult::Safe,
                        strategy_name: strategy.name.clone(),
                        time_taken,
                        strategies_tried,
                    });
                }
                Ok(SpacerResult::Unsafe) => {
                    let time_taken = strategy_start.elapsed();
                    tracing::info!(
                        "Strategy '{}' found counterexample in {:?}",
                        strategy.name,
                        time_taken
                    );
                    return Ok(PortfolioResult {
                        result: SpacerResult::Unsafe,
                        strategy_name: strategy.name.clone(),
                        time_taken,
                        strategies_tried,
                    });
                }
                Ok(SpacerResult::Unknown) => {
                    tracing::debug!(
                        "Strategy '{}' returned unknown after {:?}",
                        strategy.name,
                        strategy_start.elapsed()
                    );
                    // Continue to next strategy
                }
                Err(e) => {
                    tracing::warn!("Strategy '{}' failed: {}", strategy.name, e);
                    // Continue to next strategy
                }
            }

            // Check strategy-specific timeout
            if let Some(budget) = strategy.time_budget
                && strategy_start.elapsed() >= budget
            {
                tracing::debug!(
                    "Strategy '{}' exceeded time budget {:?}",
                    strategy.name,
                    budget
                );
            }
        }

        // All strategies failed
        Err(PortfolioError::AllFailed)
    }

    /// Get the number of strategies in the portfolio
    pub fn strategy_count(&self) -> usize {
        self.strategies.len()
    }

    /// Get the strategies
    pub fn strategies(&self) -> &[Strategy] {
        &self.strategies
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
    use crate::chc::{ChcSystem, PredicateApp};

    #[test]
    fn test_strategy_creation() {
        let aggressive = Strategy::aggressive();
        assert_eq!(aggressive.name, "aggressive");
        assert!(aggressive.time_budget.is_some());

        let conservative = Strategy::conservative();
        assert_eq!(conservative.name, "conservative");
        assert!(conservative.config.use_inductive_gen);
    }

    #[test]
    fn test_portfolio_creation() {
        let portfolio = PortfolioSolver::new();
        assert_eq!(portfolio.strategy_count(), 3);

        let empty = PortfolioSolver::empty();
        assert_eq!(empty.strategy_count(), 0);
    }

    #[test]
    fn test_portfolio_add_strategy() {
        let mut portfolio = PortfolioSolver::empty();
        portfolio.add_strategy(Strategy::balanced());
        assert_eq!(portfolio.strategy_count(), 1);
    }

    #[test]
    #[ignore = "Requires complete arithmetic theory integration"]
    fn test_portfolio_simple_safe() {
        let mut terms = TermManager::new();
        let mut system = ChcSystem::new();

        // Simple safe system: x = 0 => Inv(x), Inv(x) /\ x < 0 => false
        let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
        let x = terms.mk_var("x", terms.sorts.int_sort);
        let zero = terms.mk_int(0);
        let init_constraint = terms.mk_eq(x, zero);

        system.add_init_rule(
            [("x".to_string(), terms.sorts.int_sort)],
            init_constraint,
            inv,
            [x],
        );

        let neg_constraint = terms.mk_lt(x, zero);
        system.add_query(
            [("x".to_string(), terms.sorts.int_sort)],
            [PredicateApp::new(inv, [x])],
            neg_constraint,
        );

        let portfolio = PortfolioSolver::new();
        let result = portfolio.solve_sequential(&mut terms, &system);

        assert!(result.is_ok());
        let portfolio_result = result.unwrap();
        assert_eq!(portfolio_result.result, SpacerResult::Safe);
        assert!(portfolio_result.strategies_tried > 0);
    }

    #[test]
    fn test_portfolio_with_timeout() {
        let portfolio = PortfolioSolver::new().with_global_timeout(Duration::from_secs(30));

        assert!(portfolio.global_timeout.is_some());
        assert_eq!(portfolio.global_timeout.unwrap(), Duration::from_secs(30));
    }
}
