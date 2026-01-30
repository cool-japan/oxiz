//! Restart Policy Learner
//!
//! Learn optimal restart policies using ML.

use super::{RestartDecision, RestartFeedback};
use crate::models::{LinearRegression, Model, ModelError};
use crate::{MLStats, RESTART_FEATURE_SIZE};

/// Restart features
#[derive(Debug, Clone)]
pub struct RestartFeatures {
    /// Feature vector
    pub features: Vec<f64>,
}

impl RestartFeatures {
    /// Extract features from solver state
    pub fn extract(
        conflicts: usize,
        avg_lbd: f64,
        conflict_rate: f64,
        propagation_rate: f64,
        trail_size: usize,
        decision_level: usize,
        restarts_so_far: usize,
    ) -> Self {
        let mut features = Vec::with_capacity(RESTART_FEATURE_SIZE);

        // 1. Conflicts since last restart (log scale)
        features.push((1.0 + conflicts as f64).ln() / 10.0);

        // 2. Average LBD (normalized)
        features.push(avg_lbd / 20.0);

        // 3. Conflict rate
        features.push(conflict_rate);

        // 4. Propagation rate
        features.push(propagation_rate);

        // 5. Trail utilization
        features.push(trail_size as f64 / 1000.0);

        // 6. Decision level (normalized)
        features.push(decision_level as f64 / 100.0);

        // 7. Restarts so far (log scale)
        features.push((1.0 + restarts_so_far as f64).ln() / 10.0);

        // 8. LBD trend (simplified as current LBD)
        features.push(avg_lbd / 20.0);

        // 9. Conflict/restart ratio
        let conf_restart_ratio = if restarts_so_far > 0 {
            conflicts as f64 / restarts_so_far as f64
        } else {
            conflicts as f64
        };
        features.push((1.0 + conf_restart_ratio).ln() / 10.0);

        // 10. Progress indicator (propagation - conflict rate)
        features.push((propagation_rate - conflict_rate).clamp(0.0, 1.0));

        // Pad to RESTART_FEATURE_SIZE
        features.resize(RESTART_FEATURE_SIZE, 0.0);

        Self { features }
    }
}

/// Restart policy configuration
#[derive(Debug, Clone)]
pub struct RestartConfig {
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Enable online learning
    pub online_learning: bool,
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            learning_rate: 0.01,
            online_learning: true,
        }
    }
}

/// Restart policy learner
pub struct RestartPolicyLearner {
    /// ML model
    model: LinearRegression,
    /// Configuration
    config: RestartConfig,
    /// Statistics
    stats: MLStats,
}

impl RestartPolicyLearner {
    /// Create a new restart policy learner
    pub fn new(config: RestartConfig) -> Self {
        Self {
            model: LinearRegression::new(RESTART_FEATURE_SIZE),
            config,
            stats: MLStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(RestartConfig::default())
    }

    /// Predict whether to restart
    pub fn predict_restart(&mut self, features: &RestartFeatures) -> RestartDecision {
        let start = std::time::Instant::now();

        let prediction = self.model.predict(&features.features);
        let score = prediction.first().copied().unwrap_or(0.0);

        // Score > 0.5 suggests restart is beneficial
        let should_restart = score > 0.5;
        let confidence = if should_restart { score } else { 1.0 - score };

        let expected_benefit = (score - 0.5) * 100.0; // Estimated conflicts saved

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_prediction_time(elapsed);

        RestartDecision::new(should_restart, confidence, expected_benefit)
    }

    /// Learn from feedback
    pub fn learn_from_feedback(&mut self, features: &RestartFeatures, feedback: RestartFeedback) {
        if !self.config.online_learning {
            return;
        }

        let start = std::time::Instant::now();

        // Target: 1.0 if beneficial, 0.0 otherwise
        let target = if feedback.was_beneficial { 1.0 } else { 0.0 };

        if let Err(e) = self.model.train(&features.features, &[target]) {
            eprintln!("Training error: {}", e);
        }

        if feedback.was_beneficial {
            self.stats.record_correct();
        } else {
            self.stats.record_incorrect();
        }

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_training_time(elapsed);
    }

    /// Get statistics
    pub fn stats(&self) -> &MLStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MLStats::default();
    }

    /// Save model
    pub fn save_model(&self) -> Result<Vec<u8>, ModelError> {
        self.model.save()
    }

    /// Load model
    pub fn load_model(&mut self, data: &[u8]) -> Result<(), ModelError> {
        self.model.load(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restart_features_extract() {
        let features = RestartFeatures::extract(
            100, // conflicts
            3.5, // avg_lbd
            0.3, // conflict_rate
            0.7, // propagation_rate
            50,  // trail_size
            20,  // decision_level
            5,   // restarts_so_far
        );

        assert_eq!(features.features.len(), RESTART_FEATURE_SIZE);
        assert!(features.features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_restart_policy_learner_creation() {
        let learner = RestartPolicyLearner::default_config();
        assert_eq!(learner.stats.predictions, 0);
    }

    #[test]
    fn test_restart_policy_learner_predict() {
        let mut learner = RestartPolicyLearner::default_config();
        let features = RestartFeatures::extract(100, 5.0, 0.5, 0.5, 50, 20, 5);

        let decision = learner.predict_restart(&features);
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
    }

    #[test]
    fn test_restart_policy_learner_learn() {
        let mut learner = RestartPolicyLearner::default_config();
        let features = RestartFeatures::extract(100, 5.0, 0.5, 0.5, 50, 20, 5);

        let feedback = RestartFeedback {
            was_beneficial: true,
            conflicts_before: 100,
            conflicts_saved: 20,
            time_saved_us: 1000,
        };

        learner.learn_from_feedback(&features, feedback);
        assert_eq!(learner.stats.correct, 1);
    }
}
