//! ML-Guided Branching Heuristics
//!
//! Learn optimal variable selection and polarity choices from solver history.

mod features;
mod learner;
mod vsids_ml;

pub use features::{BranchingFeatures, FeatureCache, FeatureExtractor};
pub use learner::{BranchingConfig, BranchingLearner, BranchingStats};
pub use vsids_ml::{MLEnhancedVSIDS, VSIDSConfig};

/// Variable ID type (re-export from oxiz-sat)
pub type VarId = usize;

/// Branching decision
#[derive(Debug, Clone, Copy)]
pub struct BranchingDecision {
    /// Variable to branch on
    pub variable: VarId,
    /// Suggested polarity (true = positive, false = negative)
    pub polarity: bool,
    /// Confidence score
    pub confidence: f64,
}

impl BranchingDecision {
    /// Create a new branching decision
    pub fn new(variable: VarId, polarity: bool, confidence: f64) -> Self {
        Self {
            variable,
            polarity,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if this decision is confident enough
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Feedback for learning
#[derive(Debug, Clone, Copy)]
pub struct BranchingFeedback {
    /// Was the branching decision good? (led to progress)
    pub was_good: bool,
    /// Number of conflicts after this branch
    pub conflicts_after: usize,
    /// Number of propagations after this branch
    pub propagations_after: usize,
    /// Time to next conflict (us)
    pub time_to_conflict_us: u64,
}

impl BranchingFeedback {
    /// Compute a reward score from feedback
    pub fn reward_score(&self) -> f64 {
        if self.was_good {
            // More propagations and fewer conflicts = better
            let prop_score = (1.0 + self.propagations_after as f64).ln();
            let conflict_penalty = (self.conflicts_after as f64) * 0.5;
            (prop_score - conflict_penalty).max(0.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branching_decision() {
        let decision = BranchingDecision::new(5, true, 0.8);
        assert_eq!(decision.variable, 5);
        assert!(decision.polarity);
        assert!(decision.is_confident(0.7));
        assert!(!decision.is_confident(0.9));
    }

    #[test]
    fn test_branching_feedback_reward() {
        let good_feedback = BranchingFeedback {
            was_good: true,
            conflicts_after: 1,
            propagations_after: 100,
            time_to_conflict_us: 1000,
        };
        assert!(good_feedback.reward_score() > 0.0);

        let bad_feedback = BranchingFeedback {
            was_good: false,
            conflicts_after: 10,
            propagations_after: 5,
            time_to_conflict_us: 100,
        };
        assert_eq!(bad_feedback.reward_score(), 0.0);
    }
}
