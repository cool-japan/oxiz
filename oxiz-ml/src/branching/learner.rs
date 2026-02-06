//! Branching Learner with ML Models
//!
//! Main interface for ML-guided branching decisions.

use super::features::{BranchingFeatures, FeatureExtractor};
use super::{BranchingDecision, BranchingFeedback, VarId};
use crate::models::{LinearRegression, Model, ModelError, NeuralNetwork};
use crate::{BRANCHING_FEATURE_SIZE, MLStats};
use serde::{Deserialize, Serialize};

/// Branching learner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchingConfig {
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Learning rate for online updates
    pub learning_rate: f64,
    /// Use neural network (vs linear model)
    pub use_neural_network: bool,
    /// Enable online learning
    pub online_learning: bool,
    /// Exploration rate (epsilon-greedy)
    pub exploration_rate: f64,
}

impl Default for BranchingConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            learning_rate: 0.01,
            use_neural_network: true,
            online_learning: true,
            exploration_rate: 0.1,
        }
    }
}

/// Branching learner statistics
#[derive(Debug, Clone, Default)]
pub struct BranchingStats {
    /// ML statistics
    pub ml_stats: MLStats,
    /// Number of exploration decisions
    pub explorations: usize,
    /// Number of exploitation decisions
    pub exploitations: usize,
    /// Average reward per decision
    pub avg_reward: f64,
    /// Total reward accumulated
    pub total_reward: f64,
}

impl BranchingStats {
    /// Record a reward
    pub fn record_reward(&mut self, reward: f64) {
        self.total_reward += reward;
        let n = (self.explorations + self.exploitations) as f64;
        if n > 0.0 {
            self.avg_reward = self.total_reward / n;
        }
    }
}

/// ML-based branching learner
pub struct BranchingLearner {
    /// Feature extractor
    feature_extractor: FeatureExtractor,
    /// ML model (either neural network or linear)
    model: Box<dyn Model + Send + Sync>,
    /// Configuration
    config: BranchingConfig,
    /// Statistics
    stats: BranchingStats,
    /// Pseudo-random state for exploration
    rand_state: u64,
}

impl BranchingLearner {
    /// Create a new branching learner
    pub fn new(config: BranchingConfig) -> Result<Self, ModelError> {
        let feature_extractor = FeatureExtractor::new(BRANCHING_FEATURE_SIZE);

        let model: Box<dyn Model + Send + Sync> = if config.use_neural_network {
            // Create a small neural network [15 -> 8 -> 4 -> 1]
            Box::new(NeuralNetwork::simple(vec![
                BRANCHING_FEATURE_SIZE,
                8,
                4,
                1,
            ])?)
        } else {
            // Use linear regression
            Box::new(LinearRegression::new(BRANCHING_FEATURE_SIZE))
        };

        Ok(Self {
            feature_extractor,
            model,
            config,
            stats: BranchingStats::default(),
            rand_state: 12345,
        })
    }

    /// Create with default configuration
    pub fn default_config() -> Result<Self, ModelError> {
        Self::new(BranchingConfig::default())
    }

    /// Create with a neural network
    pub fn new_with_neural_net(
        layer_sizes: Vec<usize>,
        learning_rate: f64,
    ) -> Result<Self, ModelError> {
        let config = BranchingConfig {
            learning_rate,
            use_neural_network: true,
            ..Default::default()
        };

        let feature_extractor = FeatureExtractor::new(BRANCHING_FEATURE_SIZE);
        let model = Box::new(NeuralNetwork::simple(layer_sizes)?);

        Ok(Self {
            feature_extractor,
            model,
            config,
            stats: BranchingStats::default(),
            rand_state: 12345,
        })
    }

    /// Predict best branching decision from candidates
    pub fn predict_branch(&mut self, candidates: &[VarId]) -> Option<BranchingDecision> {
        if candidates.is_empty() {
            return None;
        }

        let start = std::time::Instant::now();

        // Epsilon-greedy exploration
        if self.should_explore() {
            self.stats.explorations += 1;
            let idx = self.random_index(candidates.len());
            let var = candidates[idx];
            let features = self.feature_extractor.extract(var);
            let polarity = self.predict_polarity(&features);

            let elapsed = start.elapsed().as_micros() as u64;
            self.stats.ml_stats.record_prediction_time(elapsed);

            return Some(BranchingDecision::new(var, polarity, 0.5));
        }

        // Exploitation: use ML model
        self.stats.exploitations += 1;

        let mut best_var = candidates[0];
        let mut best_score = f64::NEG_INFINITY;

        for &var in candidates {
            let features = self.feature_extractor.extract(var);
            let prediction = self.model.predict(&features.features);

            if !prediction.is_empty() {
                let score = prediction[0];
                if score > best_score {
                    best_score = score;
                    best_var = var;
                }
            }
        }

        let features = self.feature_extractor.extract(best_var);
        let polarity = self.predict_polarity(&features);
        let confidence = self.score_to_confidence(best_score);

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.ml_stats.record_prediction_time(elapsed);

        Some(BranchingDecision::new(best_var, polarity, confidence))
    }

    /// Predict polarity based on phase consistency feature
    fn predict_polarity(&self, features: &BranchingFeatures) -> bool {
        if features.features.len() > 4 {
            // Feature index 4 is phase consistency
            features.features[4] > 0.5
        } else {
            true
        }
    }

    /// Convert score to confidence
    fn score_to_confidence(&self, score: f64) -> f64 {
        // Sigmoid transformation
        let confidence = 1.0 / (1.0 + (-5.0 * score).exp());
        confidence.clamp(0.0, 1.0)
    }

    /// Learn from feedback
    pub fn learn_from_feedback(&mut self, var: VarId, feedback: BranchingFeedback) {
        if !self.config.online_learning {
            return;
        }

        let start = std::time::Instant::now();

        let reward = feedback.reward_score();
        self.stats.record_reward(reward);

        // Extract features and train
        let features = self.feature_extractor.extract(var);
        let target = vec![reward];

        if let Err(e) = self.model.train(&features.features, &target) {
            // Log error but don't fail
            eprintln!("Training error: {}", e);
        }

        // Validate prediction
        if feedback.was_good {
            self.stats.ml_stats.record_correct();
        } else {
            self.stats.ml_stats.record_incorrect();
        }

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.ml_stats.record_training_time(elapsed);
    }

    /// Update feature extractor with solver events
    pub fn update_conflict(&mut self, var: VarId, lbd: f64) {
        self.feature_extractor.update_conflict(var, lbd);
    }

    /// Update propagation
    pub fn update_propagation(&mut self, var: VarId) {
        self.feature_extractor.update_propagation(var);
    }

    /// Update decision
    pub fn update_decision(&mut self, var: VarId, polarity: bool, level: usize) {
        self.feature_extractor.update_decision(var, polarity, level);
    }

    /// Bump activity
    pub fn bump_activity(&mut self, var: VarId, amount: f64) {
        self.feature_extractor.bump_activity(var, amount);
    }

    /// Decay activities
    pub fn decay_activities(&mut self, factor: f64) {
        self.feature_extractor.decay_all_activities(factor);
    }

    /// Check if should explore (epsilon-greedy)
    fn should_explore(&mut self) -> bool {
        // Pseudo-random number generation
        self.rand_state = self.rand_state.wrapping_mul(1103515245).wrapping_add(12345);
        let rand_val = ((self.rand_state / 65536) % 1000) as f64 / 1000.0;
        rand_val < self.config.exploration_rate
    }

    /// Get pseudo-random index
    fn random_index(&mut self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        self.rand_state = self.rand_state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.rand_state / 65536) % max as u64) as usize
    }

    /// Get statistics
    pub fn stats(&self) -> &BranchingStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BranchingStats::default();
    }

    /// Save model
    pub fn save_model(&self) -> Result<Vec<u8>, ModelError> {
        self.model.save()
    }

    /// Load model
    pub fn load_model(&mut self, data: &[u8]) -> Result<(), ModelError> {
        self.model.load(data)
    }

    /// Get configuration
    pub fn config(&self) -> &BranchingConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branching_learner_creation() {
        let learner = BranchingLearner::default_config().unwrap();
        assert!(learner.stats().ml_stats.predictions == 0);
    }

    #[test]
    fn test_branching_learner_predict() {
        let mut learner = BranchingLearner::default_config().unwrap();
        let candidates = vec![0, 1, 2];

        let decision = learner.predict_branch(&candidates);
        assert!(decision.is_some());

        let decision = decision.unwrap();
        assert!(candidates.contains(&decision.variable));
    }

    #[test]
    fn test_branching_learner_learn() {
        let mut learner = BranchingLearner::default_config().unwrap();

        let feedback = BranchingFeedback {
            was_good: true,
            conflicts_after: 1,
            propagations_after: 50,
            time_to_conflict_us: 1000,
        };

        learner.learn_from_feedback(0, feedback);
        assert!(learner.stats().total_reward > 0.0);
    }

    #[test]
    fn test_branching_learner_updates() {
        let mut learner = BranchingLearner::default_config().unwrap();

        learner.update_conflict(0, 3.0);
        learner.update_propagation(0);
        learner.update_decision(0, true, 5);

        assert!(learner.feature_extractor.num_variables() > 0);
    }

    #[test]
    fn test_branching_learner_save_load() {
        let learner = BranchingLearner::default_config().unwrap();

        let saved = learner.save_model().unwrap();
        assert!(!saved.is_empty());

        let mut learner2 = BranchingLearner::default_config().unwrap();
        learner2.load_model(&saved).unwrap();
    }
}
