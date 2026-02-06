//! ML-Guided Restart Policy Learning
//!
//! Learn when to restart the solver for maximum efficiency.

mod adaptive;
mod policy_learner;

pub use adaptive::{AdaptiveConfig, AdaptiveRestart};
pub use policy_learner::{RestartConfig, RestartFeatures, RestartPolicyLearner};

/// Restart decision
#[derive(Debug, Clone, Copy)]
pub struct RestartDecision {
    /// Should restart now?
    pub should_restart: bool,
    /// Confidence in this decision
    pub confidence: f64,
    /// Predicted benefit of restarting
    pub expected_benefit: f64,
}

impl RestartDecision {
    /// Create a new restart decision
    pub fn new(should_restart: bool, confidence: f64, expected_benefit: f64) -> Self {
        Self {
            should_restart,
            confidence: confidence.clamp(0.0, 1.0),
            expected_benefit,
        }
    }

    /// Check if confident enough to restart
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Restart feedback for learning
#[derive(Debug, Clone, Copy)]
pub struct RestartFeedback {
    /// Did restarting help?
    pub was_beneficial: bool,
    /// Conflicts before restart
    pub conflicts_before: usize,
    /// Conflicts saved after restart
    pub conflicts_saved: usize,
    /// Time gained (us)
    pub time_saved_us: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restart_decision() {
        let decision = RestartDecision::new(true, 0.8, 100.0);
        assert!(decision.should_restart);
        assert!(decision.is_confident(0.7));
    }
}
