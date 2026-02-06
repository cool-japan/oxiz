//! Tactic Selector
//!
//! Select best tactic for a formula using ML.

use super::{FormulaFeatures, TacticFeedback, TacticSelection};
use crate::models::{DecisionTree, Model, ModelError, SplitCriterion, TreeConfig};
use crate::{MLStats, TACTIC_FEATURE_SIZE};

/// Tactic ID type
pub type TacticId = usize;

/// Tactic selector configuration
#[derive(Debug, Clone)]
pub struct TacticConfig {
    /// Number of available tactics
    pub num_tactics: usize,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Enable online learning
    pub online_learning: bool,
}

impl Default for TacticConfig {
    fn default() -> Self {
        Self {
            num_tactics: 5, // Default: 5 tactics
            min_confidence: 0.6,
            online_learning: true,
        }
    }
}

/// ML-based tactic selector
pub struct TacticSelector {
    /// Decision tree for tactic selection
    model: DecisionTree,
    /// Configuration
    config: TacticConfig,
    /// Statistics
    stats: MLStats,
    /// Tactic performance history (for estimation)
    tactic_times: Vec<Vec<f64>>,
}

impl TacticSelector {
    /// Create a new tactic selector
    pub fn new(config: TacticConfig) -> Self {
        let tree_config = TreeConfig {
            max_depth: 8,
            min_samples_split: 5,
            min_samples_leaf: 2,
            criterion: SplitCriterion::Entropy,
            max_features: 0,
        };

        let model = DecisionTree::new(TACTIC_FEATURE_SIZE, tree_config);
        let num_tactics = config.num_tactics;

        Self {
            model,
            config,
            stats: MLStats::default(),
            tactic_times: vec![Vec::new(); num_tactics],
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(TacticConfig::default())
    }

    /// Select best tactic for a formula
    pub fn select_tactic(&mut self, features: &FormulaFeatures) -> TacticSelection {
        let start = std::time::Instant::now();

        // Get prediction from model
        let prediction = self.model.predict(&features.features);
        let tactic_score = prediction.first().copied().unwrap_or(0.0);

        // Map score to tactic ID
        let tactic_id = (tactic_score.abs() * self.config.num_tactics as f64) as usize
            % self.config.num_tactics;

        // Estimate time based on historical data
        let estimated_time = self.estimate_time(tactic_id);

        // Confidence is medium for decision tree
        let confidence = 0.7;

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_prediction_time(elapsed);

        TacticSelection::new(tactic_id, confidence, estimated_time)
    }

    /// Estimate solve time for a tactic
    fn estimate_time(&self, tactic_id: TacticId) -> f64 {
        if tactic_id >= self.tactic_times.len() {
            return 10.0; // Default estimate
        }

        let times = &self.tactic_times[tactic_id];
        if times.is_empty() {
            return 10.0;
        }

        // Use median as robust estimate
        let mut sorted_times = times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted_times[sorted_times.len() / 2]
    }

    /// Learn from feedback
    pub fn learn_from_feedback(
        &mut self,
        features: &FormulaFeatures,
        tactic_id: TacticId,
        feedback: TacticFeedback,
    ) {
        // Record actual time for future estimation
        if tactic_id < self.tactic_times.len() {
            self.tactic_times[tactic_id].push(feedback.actual_time);

            // Keep only recent history
            if self.tactic_times[tactic_id].len() > 100 {
                self.tactic_times[tactic_id].remove(0);
            }
        }

        // Update statistics
        if feedback.was_successful {
            self.stats.record_correct();
        } else {
            self.stats.record_incorrect();
        }
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
    fn test_tactic_selector_creation() {
        let selector = TacticSelector::default_config();
        assert_eq!(selector.stats.predictions, 0);
    }

    #[test]
    fn test_tactic_selector_select() {
        let mut selector = TacticSelector::default_config();
        let features = FormulaFeatures::default();

        let selection = selector.select_tactic(&features);
        assert!(selection.tactic_id < selector.config.num_tactics);
    }

    #[test]
    fn test_tactic_selector_learn() {
        let mut selector = TacticSelector::default_config();
        let features = FormulaFeatures::default();

        let feedback = TacticFeedback {
            was_successful: true,
            actual_time: 5.0,
            conflicts: 1000,
        };

        selector.learn_from_feedback(&features, 0, feedback);
        assert_eq!(selector.stats.correct, 1);
    }
}
