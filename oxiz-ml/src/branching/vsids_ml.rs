//! ML-Enhanced VSIDS (Variable State Independent Decaying Sum)
//!
//! Combines traditional VSIDS heuristic with ML predictions.

use super::{BranchingDecision, BranchingFeedback, BranchingLearner, VarId};
use crate::MLStats;
use rustc_hash::FxHashMap;

/// VSIDS configuration
#[derive(Debug, Clone)]
pub struct VSIDSConfig {
    /// Activity decay factor
    pub decay_factor: f64,
    /// Initial activity boost
    pub initial_boost: f64,
    /// ML weight (0.0 = pure VSIDS, 1.0 = pure ML)
    pub ml_weight: f64,
    /// Minimum confidence to trust ML
    pub min_ml_confidence: f64,
}

impl Default for VSIDSConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.95,
            initial_boost: 1.0,
            ml_weight: 0.3, // Blend VSIDS and ML
            min_ml_confidence: 0.6,
        }
    }
}

/// ML-enhanced VSIDS branching heuristic
pub struct MLEnhancedVSIDS {
    /// Variable activities
    activities: FxHashMap<VarId, f64>,
    /// Activity increment
    activity_inc: f64,
    /// ML learner
    ml_learner: Option<BranchingLearner>,
    /// Configuration
    config: VSIDSConfig,
    /// Statistics
    stats: MLStats,
}

impl MLEnhancedVSIDS {
    /// Create a new ML-enhanced VSIDS
    pub fn new(config: VSIDSConfig, ml_learner: Option<BranchingLearner>) -> Self {
        Self {
            activities: FxHashMap::default(),
            activity_inc: 1.0,
            ml_learner,
            config,
            stats: MLStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(VSIDSConfig::default(), None)
    }

    /// Create with ML learner
    pub fn with_ml(config: VSIDSConfig, learner: BranchingLearner) -> Self {
        Self::new(config, Some(learner))
    }

    /// Select next branching variable
    pub fn select_variable(&mut self, candidates: &[VarId]) -> Option<BranchingDecision> {
        if candidates.is_empty() {
            return None;
        }

        let start = std::time::Instant::now();

        // Get VSIDS decision
        let vsids_var = self.select_vsids(candidates);

        // Get ML decision if available
        let decision = if let Some(ref mut learner) = self.ml_learner {
            if let Some(ml_decision) = learner.predict_branch(candidates) {
                // Blend VSIDS and ML
                if ml_decision.confidence >= self.config.min_ml_confidence {
                    // High confidence ML: use ML with some VSIDS influence
                    if self.pseudo_random() < self.config.ml_weight {
                        ml_decision
                    } else {
                        BranchingDecision::new(vsids_var, ml_decision.polarity, 0.7)
                    }
                } else {
                    // Low confidence ML: mostly VSIDS
                    BranchingDecision::new(vsids_var, ml_decision.polarity, 0.6)
                }
            } else {
                // No ML prediction: pure VSIDS
                BranchingDecision::new(vsids_var, true, 0.5)
            }
        } else {
            // No ML learner: pure VSIDS
            BranchingDecision::new(vsids_var, true, 0.5)
        };

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_prediction_time(elapsed);

        Some(decision)
    }

    /// Select variable using pure VSIDS
    fn select_vsids(&self, candidates: &[VarId]) -> VarId {
        let mut best_var = candidates[0];
        let mut best_activity = self.activities.get(&candidates[0]).copied().unwrap_or(0.0);

        for &var in &candidates[1..] {
            let activity = self.activities.get(&var).copied().unwrap_or(0.0);
            if activity > best_activity {
                best_activity = activity;
                best_var = var;
            }
        }

        best_var
    }

    /// Bump variable activity
    pub fn bump_activity(&mut self, var: VarId) {
        let activity = self.activities.entry(var).or_insert(0.0);
        *activity += self.activity_inc;

        // Rescale if activities get too large
        if *activity > 1e100 {
            self.rescale_activities();
        }

        // Update ML learner if available
        if let Some(ref mut learner) = self.ml_learner {
            learner.bump_activity(var, self.activity_inc);
        }
    }

    /// Decay all activities
    pub fn decay_activities(&mut self) {
        self.activity_inc /= self.config.decay_factor;

        if let Some(ref mut learner) = self.ml_learner {
            learner.decay_activities(self.config.decay_factor);
        }
    }

    /// Rescale all activities to prevent overflow
    fn rescale_activities(&mut self) {
        let scale = 1e-100;
        for activity in self.activities.values_mut() {
            *activity *= scale;
        }
        self.activity_inc *= scale;
    }

    /// Learn from feedback
    pub fn learn_from_feedback(&mut self, var: VarId, feedback: BranchingFeedback) {
        if let Some(ref mut learner) = self.ml_learner {
            learner.learn_from_feedback(var, feedback);

            if feedback.was_good {
                self.stats.record_correct();
            } else {
                self.stats.record_incorrect();
            }
        }
    }

    /// Update with conflict
    pub fn update_conflict(&mut self, var: VarId, lbd: f64) {
        self.bump_activity(var);

        if let Some(ref mut learner) = self.ml_learner {
            learner.update_conflict(var, lbd);
        }
    }

    /// Update with propagation
    pub fn update_propagation(&mut self, var: VarId) {
        if let Some(ref mut learner) = self.ml_learner {
            learner.update_propagation(var);
        }
    }

    /// Update with decision
    pub fn update_decision(&mut self, var: VarId, polarity: bool, level: usize) {
        if let Some(ref mut learner) = self.ml_learner {
            learner.update_decision(var, polarity, level);
        }
    }

    /// Get activity score for a variable
    pub fn get_activity(&self, var: VarId) -> f64 {
        self.activities.get(&var).copied().unwrap_or(0.0)
    }

    /// Get statistics
    pub fn stats(&self) -> &MLStats {
        &self.stats
    }

    /// Get ML learner statistics if available
    pub fn ml_stats(&self) -> Option<&MLStats> {
        self.ml_learner.as_ref().map(|l| &l.stats().ml_stats)
    }

    /// Pseudo-random number for blending
    fn pseudo_random(&self) -> f64 {
        // Simple pseudo-random based on stats
        let val = (self.stats.predictions * 2654435761) % 1000;
        val as f64 / 1000.0
    }

    /// Clear all statistics
    pub fn clear_stats(&mut self) {
        self.stats = MLStats::default();
        if let Some(ref mut learner) = self.ml_learner {
            learner.reset_stats();
        }
    }

    /// Reset activities
    pub fn reset(&mut self) {
        self.activities.clear();
        self.activity_inc = 1.0;
        self.stats = MLStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_vsids_creation() {
        let vsids = MLEnhancedVSIDS::default_config();
        assert_eq!(vsids.stats.predictions, 0);
    }

    #[test]
    fn test_ml_vsids_bump_activity() {
        let mut vsids = MLEnhancedVSIDS::default_config();

        vsids.bump_activity(0);
        vsids.bump_activity(0);
        vsids.bump_activity(1);

        assert!(vsids.get_activity(0) > vsids.get_activity(1));
    }

    #[test]
    fn test_ml_vsids_select_variable() {
        let mut vsids = MLEnhancedVSIDS::default_config();

        vsids.bump_activity(1);
        vsids.bump_activity(1);

        let decision = vsids.select_variable(&[0, 1, 2]);
        assert!(decision.is_some());
        assert_eq!(decision.unwrap().variable, 1);
    }

    #[test]
    fn test_ml_vsids_decay() {
        let mut vsids = MLEnhancedVSIDS::default_config();

        vsids.bump_activity(0);
        let initial_inc = vsids.activity_inc;

        vsids.decay_activities();
        // In VSIDS, activity_inc is divided by decay_factor (< 1),
        // so it INCREASES after decay to give more weight to recent events
        assert!(vsids.activity_inc > initial_inc);
    }

    #[test]
    fn test_ml_vsids_rescale() {
        let mut vsids = MLEnhancedVSIDS::default_config();

        *vsids.activities.entry(0).or_insert(0.0) = 1e101;
        vsids.bump_activity(0);

        // Should have rescaled
        assert!(vsids.get_activity(0) < 1e100);
    }
}
