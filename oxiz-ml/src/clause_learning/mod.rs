//! ML-Guided Clause Learning and Deletion
//!
//! Predict clause usefulness and guide deletion policies.

mod deletion_policy;
mod usefulness_predictor;

pub use deletion_policy::{DeletionConfig, DeletionDecision, DeletionPolicy};
pub use usefulness_predictor::{ClauseFeatures, UsefulnessConfig, UsefulnessPredictor};

/// Clause ID type
pub type ClauseId = usize;

/// Usefulness prediction
#[derive(Debug, Clone, Copy)]
pub struct UsefulnessPrediction {
    /// Predicted usefulness score (0.0 = useless, 1.0 = very useful)
    pub score: f64,
    /// Confidence in prediction
    pub confidence: f64,
    /// Should keep this clause?
    pub should_keep: bool,
}

impl UsefulnessPrediction {
    /// Create a new usefulness prediction
    pub fn new(score: f64, confidence: f64) -> Self {
        let should_keep = score > 0.5;
        Self {
            score: score.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            should_keep,
        }
    }
}

/// Clause feedback for learning
#[derive(Debug, Clone, Copy)]
pub struct ClauseFeedback {
    /// Was the clause actually useful?
    pub was_useful: bool,
    /// Number of times used in conflicts
    pub usage_count: usize,
    /// Clause age when deleted/evaluated
    pub age: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usefulness_prediction() {
        let pred = UsefulnessPrediction::new(0.8, 0.9);
        assert!(pred.should_keep);
        assert!(pred.score > 0.5);
    }
}
