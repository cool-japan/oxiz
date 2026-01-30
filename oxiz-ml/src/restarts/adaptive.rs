//! Adaptive Restart Strategy
//!
//! Dynamically adjust restart intervals based on solver progress.

use super::{RestartDecision, RestartFeatures, RestartFeedback, RestartPolicyLearner};
use crate::MLStats;

/// Adaptive restart configuration
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Base restart interval
    pub base_interval: usize,
    /// Maximum restart interval
    pub max_interval: usize,
    /// Geometric multiplier for restart intervals
    pub multiplier: f64,
    /// Use ML guidance
    pub use_ml: bool,
    /// Minimum conflicts between restarts
    pub min_conflicts: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            base_interval: 100,
            max_interval: 100_000,
            multiplier: 1.5,
            use_ml: true,
            min_conflicts: 50,
        }
    }
}

/// Adaptive restart strategy
pub struct AdaptiveRestart {
    /// Configuration
    config: AdaptiveConfig,
    /// ML policy learner (optional)
    learner: Option<RestartPolicyLearner>,
    /// Current restart interval
    current_interval: usize,
    /// Conflicts since last restart
    conflicts_since_restart: usize,
    /// Total restarts
    total_restarts: usize,
    /// Average LBD tracker
    lbd_sum: f64,
    lbd_count: usize,
    /// Statistics
    stats: MLStats,
}

impl AdaptiveRestart {
    /// Create a new adaptive restart strategy
    pub fn new(config: AdaptiveConfig, learner: Option<RestartPolicyLearner>) -> Self {
        let current_interval = config.base_interval;

        Self {
            config,
            learner,
            current_interval,
            conflicts_since_restart: 0,
            total_restarts: 0,
            lbd_sum: 0.0,
            lbd_count: 0,
            stats: MLStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        let config = AdaptiveConfig::default();
        let learner = if config.use_ml {
            Some(RestartPolicyLearner::default_config())
        } else {
            None
        };

        Self::new(config, learner)
    }

    /// Create with ML learner
    pub fn with_ml(config: AdaptiveConfig) -> Self {
        let learner = Some(RestartPolicyLearner::default_config());
        Self::new(config, learner)
    }

    /// Check if should restart
    pub fn should_restart(
        &mut self,
        conflict_rate: f64,
        propagation_rate: f64,
        trail_size: usize,
        decision_level: usize,
    ) -> RestartDecision {
        let start = std::time::Instant::now();

        // Minimum conflicts check
        if self.conflicts_since_restart < self.config.min_conflicts {
            let elapsed = start.elapsed().as_micros() as u64;
            self.stats.record_prediction_time(elapsed);
            return RestartDecision::new(false, 1.0, 0.0);
        }

        let avg_lbd = if self.lbd_count > 0 {
            self.lbd_sum / self.lbd_count as f64
        } else {
            5.0
        };

        // Try ML prediction first
        if let Some(ref mut learner) = self.learner {
            let features = RestartFeatures::extract(
                self.conflicts_since_restart,
                avg_lbd,
                conflict_rate,
                propagation_rate,
                trail_size,
                decision_level,
                self.total_restarts,
            );

            let ml_decision = learner.predict_restart(&features);

            // If ML is confident, use it
            if ml_decision.is_confident(0.6) {
                let elapsed = start.elapsed().as_micros() as u64;
                self.stats.record_prediction_time(elapsed);
                return ml_decision;
            }
        }

        // Fallback to geometric restart schedule
        let should_restart = self.conflicts_since_restart >= self.current_interval;
        let confidence = 0.5; // Medium confidence for schedule-based restart

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_prediction_time(elapsed);

        RestartDecision::new(should_restart, confidence, 0.0)
    }

    /// Execute restart (update state)
    pub fn do_restart(&mut self) {
        self.total_restarts += 1;
        self.conflicts_since_restart = 0;

        // Update restart interval (geometric sequence)
        self.current_interval = ((self.current_interval as f64 * self.config.multiplier) as usize)
            .min(self.config.max_interval);

        // Reset LBD tracking
        self.lbd_sum = 0.0;
        self.lbd_count = 0;
    }

    /// Record a conflict
    pub fn record_conflict(&mut self, lbd: f64) {
        self.conflicts_since_restart += 1;
        self.lbd_sum += lbd;
        self.lbd_count += 1;
    }

    /// Learn from restart feedback
    pub fn learn_from_feedback(&mut self, features: &RestartFeatures, feedback: RestartFeedback) {
        if let Some(ref mut learner) = self.learner {
            learner.learn_from_feedback(features, feedback);

            // Update stats
            if feedback.was_beneficial {
                self.stats.record_correct();
            } else {
                self.stats.record_incorrect();
            }
        }
    }

    /// Reset strategy (e.g., when changing problem)
    pub fn reset(&mut self) {
        self.current_interval = self.config.base_interval;
        self.conflicts_since_restart = 0;
        self.total_restarts = 0;
        self.lbd_sum = 0.0;
        self.lbd_count = 0;
        self.stats = MLStats::default();
    }

    /// Get current restart interval
    pub fn current_interval(&self) -> usize {
        self.current_interval
    }

    /// Get total restarts
    pub fn total_restarts(&self) -> usize {
        self.total_restarts
    }

    /// Get statistics
    pub fn stats(&self) -> &MLStats {
        &self.stats
    }

    /// Get ML learner statistics if available
    pub fn ml_stats(&self) -> Option<&MLStats> {
        self.learner.as_ref().map(|l| l.stats())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_restart_creation() {
        let restart = AdaptiveRestart::default_config();
        assert_eq!(restart.total_restarts, 0);
        assert_eq!(restart.current_interval, 100);
    }

    #[test]
    fn test_adaptive_restart_min_conflicts() {
        let mut restart = AdaptiveRestart::default_config();

        // Should not restart if below minimum conflicts
        let decision = restart.should_restart(0.5, 0.5, 50, 20);
        assert!(!decision.should_restart);
    }

    #[test]
    fn test_adaptive_restart_schedule() {
        let mut restart = AdaptiveRestart::default_config();

        // Record enough conflicts
        for _ in 0..100 {
            restart.record_conflict(3.0);
        }

        // Should consider restarting
        let decision = restart.should_restart(0.5, 0.5, 50, 20);
        // Decision depends on ML or schedule
    }

    #[test]
    fn test_adaptive_restart_do_restart() {
        let mut restart = AdaptiveRestart::default_config();

        restart.do_restart();

        assert_eq!(restart.total_restarts, 1);
        assert_eq!(restart.conflicts_since_restart, 0);
        assert!(restart.current_interval > 100); // Increased by multiplier
    }

    #[test]
    fn test_adaptive_restart_record_conflict() {
        let mut restart = AdaptiveRestart::default_config();

        restart.record_conflict(3.0);
        restart.record_conflict(5.0);

        assert_eq!(restart.conflicts_since_restart, 2);
        assert_eq!(restart.lbd_count, 2);
    }

    #[test]
    fn test_adaptive_restart_reset() {
        let mut restart = AdaptiveRestart::default_config();

        restart.record_conflict(3.0);
        restart.do_restart();

        restart.reset();

        assert_eq!(restart.total_restarts, 0);
        assert_eq!(restart.conflicts_since_restart, 0);
        assert_eq!(restart.current_interval, 100);
    }
}
