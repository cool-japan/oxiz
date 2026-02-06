//! Portfolio Solving with ML-Guided Tactic Selection
//!
//! Run multiple tactics in parallel and select best based on ML predictions.

use super::{FormulaFeatures, TacticId};
use crate::MLStats;

/// Portfolio configuration
#[derive(Debug, Clone)]
pub struct PortfolioConfig {
    /// Number of parallel tactics
    pub num_parallel: usize,
    /// Time limit per tactic (seconds)
    pub time_limit: f64,
    /// Use ML for tactic selection
    pub use_ml: bool,
}

impl Default for PortfolioConfig {
    fn default() -> Self {
        Self {
            num_parallel: 4,
            time_limit: 60.0,
            use_ml: true,
        }
    }
}

/// Result from a tactic in portfolio
#[derive(Debug, Clone)]
pub struct TacticResult {
    /// Tactic ID
    pub tactic_id: TacticId,
    /// Was successful?
    pub success: bool,
    /// Time taken (seconds)
    pub time: f64,
    /// Number of conflicts
    pub conflicts: usize,
}

/// Portfolio solver
pub struct Portfolio {
    /// Configuration
    config: PortfolioConfig,
    /// Statistics
    stats: MLStats,
}

impl Portfolio {
    /// Create a new portfolio solver
    pub fn new(config: PortfolioConfig) -> Self {
        Self {
            config,
            stats: MLStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(PortfolioConfig::default())
    }

    /// Select tactics to run in parallel
    pub fn select_tactics(&self, features: &FormulaFeatures) -> Vec<TacticId> {
        // For now, return first N tactics
        // In full implementation, use ML to select best tactics
        (0..self.config.num_parallel).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &MLStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::default_config();
        assert_eq!(portfolio.config.num_parallel, 4);
    }

    #[test]
    fn test_portfolio_select_tactics() {
        let portfolio = Portfolio::default_config();
        let features = FormulaFeatures::default();

        let tactics = portfolio.select_tactics(&features);
        assert_eq!(tactics.len(), 4);
    }
}
