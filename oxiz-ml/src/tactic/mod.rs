//! ML-Guided Tactic Selection
//!
//! Select best solving tactics based on formula features.

mod formula_features;
mod portfolio;
mod selector;

pub use formula_features::{FeatureExtractor, FormulaFeatures};
pub use portfolio::{Portfolio, PortfolioConfig, TacticResult};
pub use selector::{TacticConfig, TacticId, TacticSelector};

/// Tactic selection decision
#[derive(Debug, Clone, Copy)]
pub struct TacticSelection {
    /// Selected tactic ID
    pub tactic_id: TacticId,
    /// Confidence in this selection
    pub confidence: f64,
    /// Estimated solve time (seconds)
    pub estimated_time: f64,
}

impl TacticSelection {
    /// Create a new tactic selection
    pub fn new(tactic_id: TacticId, confidence: f64, estimated_time: f64) -> Self {
        Self {
            tactic_id,
            confidence: confidence.clamp(0.0, 1.0),
            estimated_time: estimated_time.max(0.0),
        }
    }
}

/// Tactic feedback for learning
#[derive(Debug, Clone, Copy)]
pub struct TacticFeedback {
    /// Was the tactic successful?
    pub was_successful: bool,
    /// Actual solve time (seconds)
    pub actual_time: f64,
    /// Number of conflicts
    pub conflicts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tactic_selection() {
        let selection = TacticSelection::new(1, 0.8, 10.5);
        assert_eq!(selection.tactic_id, 1);
        assert_eq!(selection.confidence, 0.8);
    }
}
