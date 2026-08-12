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

/// Minimum number of accumulated training samples before the selector will
/// (re)fit its decision tree. Matches the tree's `min_samples_split` so a fit
/// has enough data to actually branch rather than collapse to a single leaf.
const MIN_TRAIN_SAMPLES: usize = 5;

/// Retrain the model after this many new successful samples have been
/// recorded since the last fit — an amortized schedule so learning happens
/// periodically rather than on every single feedback event.
const RETRAIN_INTERVAL: usize = 4;

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
    /// Supervised training set: `(feature vector, target)` pairs where the
    /// target decodes (via `TacticSelector::encode_target`) back to the
    /// tactic that succeeded for those features. Accumulated from feedback
    /// and used to refit `model`.
    training_data: Vec<(Vec<f64>, f64)>,
    /// New successful samples recorded since the last retrain.
    samples_since_retrain: usize,
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
            training_data: Vec::new(),
            samples_since_retrain: 0,
        }
    }

    /// Encode a tactic id as the continuous training target that
    /// [`TacticSelector::select_tactic`]'s score->tactic mapping decodes back
    /// to `tactic_id`. Placing the target at the mid-point of the tactic's
    /// score band (`(id + 0.5) / num_tactics`) makes the decode robust to the
    /// small averaging the decision-tree leaf applies.
    fn encode_target(&self, tactic_id: TacticId) -> f64 {
        let n = self.config.num_tactics.max(1) as f64;
        (tactic_id as f64 + 0.5) / n
    }

    /// Number of accumulated supervised training samples.
    pub fn training_samples(&self) -> usize {
        self.training_data.len()
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(TacticConfig::default())
    }

    /// Select best tactic for a formula
    pub fn select_tactic(&mut self, features: &FormulaFeatures) -> TacticSelection {
        let start = oxiz_time::Instant::now();

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

    /// Learn from feedback.
    ///
    /// Besides updating the time history and accuracy counters, a *successful*
    /// outcome is recorded as a supervised training example (these features
    /// → this tactic) and the underlying decision tree is periodically refit
    /// from the accumulated examples (see `MIN_TRAIN_SAMPLES` /
    /// `RETRAIN_INTERVAL`). This is what makes the selector genuinely
    /// *learn*: after enough consistent feedback, [`TacticSelector::select_tactic`]'s
    /// prediction adapts toward the tactics that actually worked. Unsuccessful
    /// outcomes are not used as positive labels (there is no single "correct"
    /// alternative to point at), but still update the accuracy statistics.
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

            // Record a supervised example only when the feature vector matches
            // the model's expected input width, so a malformed vector can never
            // poison the training set.
            if features.features.len() == TACTIC_FEATURE_SIZE {
                let target = self.encode_target(tactic_id);
                self.training_data.push((features.features.clone(), target));
                self.samples_since_retrain += 1;

                if self.config.online_learning
                    && self.training_data.len() >= MIN_TRAIN_SAMPLES
                    && self.samples_since_retrain >= RETRAIN_INTERVAL
                {
                    self.retrain();
                }
            }
        } else {
            self.stats.record_incorrect();
        }
    }

    /// Refit the decision tree from the accumulated supervised training set.
    ///
    /// Best-effort: a fit error (e.g. a transient dimension mismatch) is
    /// swallowed rather than propagated, since tactic selection must never
    /// fail a solve — it just means the model keeps its previous structure.
    fn retrain(&mut self) {
        let inputs: Vec<Vec<f64>> = self.training_data.iter().map(|(f, _)| f.clone()).collect();
        let targets: Vec<Vec<f64>> = self.training_data.iter().map(|(_, t)| vec![*t]).collect();

        let start = oxiz_time::Instant::now();
        // The tree refits from scratch from `training_data` each time, so
        // clear its own online buffer first to avoid double-counting samples.
        self.model.clear_training_buffer();
        if self.model.train_batch(&inputs, &targets).is_ok() {
            self.stats
                .record_training_time(start.elapsed().as_micros() as u64);
            self.samples_since_retrain = 0;
        }
    }

    /// Force an immediate retrain from the accumulated feedback, regardless of
    /// the periodic schedule (useful right before persisting a model).
    pub fn retrain_now(&mut self) {
        if self.training_data.len() >= MIN_TRAIN_SAMPLES {
            self.retrain();
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

    /// End-to-end learning: after consistent feedback that tactic 1 works for
    /// one family of formulas and tactic 4 for another, the trained selector
    /// must actually predict those tactics for the respective feature
    /// vectors — proving `learn_from_feedback` really trains the model.
    #[test]
    fn test_selector_learns_to_pick_successful_tactic() {
        let mut selector = TacticSelector::default_config();

        // Two clearly distinguishable feature vectors (differ in slot 0).
        let mut a = vec![0.0; TACTIC_FEATURE_SIZE];
        a[0] = 0.1;
        let feature_a = FormulaFeatures::from_vec(a);

        let mut b = vec![0.0; TACTIC_FEATURE_SIZE];
        b[0] = 0.9;
        let feature_b = FormulaFeatures::from_vec(b);

        let ok = |t: f64| TacticFeedback {
            was_successful: true,
            actual_time: t,
            conflicts: 10,
        };

        for _ in 0..6 {
            selector.learn_from_feedback(&feature_a, 1, ok(1.0));
            selector.learn_from_feedback(&feature_b, 4, ok(2.0));
        }
        selector.retrain_now();

        assert_eq!(selector.training_samples(), 12);
        assert_eq!(selector.stats.correct, 12);

        let pick_a = selector.select_tactic(&feature_a);
        let pick_b = selector.select_tactic(&feature_b);
        assert_eq!(
            pick_a.tactic_id, 1,
            "selector should learn tactic 1 for feature family A"
        );
        assert_eq!(
            pick_b.tactic_id, 4,
            "selector should learn tactic 4 for feature family B"
        );
    }

    /// A model round-trips through save/load with its learned structure
    /// intact, so persistence across runs is meaningful.
    #[test]
    fn test_selector_model_persists_learning() {
        let mut selector = TacticSelector::default_config();
        let mut a = vec![0.0; TACTIC_FEATURE_SIZE];
        a[0] = 0.2;
        let feature_a = FormulaFeatures::from_vec(a);
        for _ in 0..8 {
            selector.learn_from_feedback(
                &feature_a,
                3,
                TacticFeedback {
                    was_successful: true,
                    actual_time: 1.0,
                    conflicts: 1,
                },
            );
        }
        selector.retrain_now();
        let saved = selector.save_model().expect("save should succeed");

        let mut restored = TacticSelector::default_config();
        restored.load_model(&saved).expect("load should succeed");
        assert_eq!(
            restored.select_tactic(&feature_a).tactic_id,
            selector.select_tactic(&feature_a).tactic_id
        );
    }
}
