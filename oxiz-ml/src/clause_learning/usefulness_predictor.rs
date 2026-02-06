//! Clause Usefulness Predictor
#![allow(clippy::too_many_arguments)] // ML feature extraction
//!
//! Predict how useful a learned clause will be.

use super::{ClauseFeedback, UsefulnessPrediction};
use crate::models::{LinearRegression, Model, ModelError};
use crate::{CLAUSE_FEATURE_SIZE, MLStats};

/// Clause features for ML prediction
#[derive(Debug, Clone)]
pub struct ClauseFeatures {
    /// Feature vector
    pub features: Vec<f64>,
}

impl ClauseFeatures {
    /// Extract features from a clause
    pub fn extract(
        lbd: usize,
        size: usize,
        activity: f64,
        age: usize,
        usage_count: usize,
        glue_count: usize,
        decision_level: usize,
        backtrack_level: usize,
    ) -> Self {
        let mut features = Vec::with_capacity(CLAUSE_FEATURE_SIZE);

        // 1. LBD (Literal Block Distance) - key quality metric
        features.push(lbd as f64 / 20.0);

        // 2. Clause size (normalized)
        features.push(size as f64 / 50.0);

        // 3. Activity score (normalized)
        features.push(activity.min(1.0));

        // 4. Age (log scale)
        features.push((1.0 + age as f64).ln() / 10.0);

        // 5. Usage count (how often used in conflicts)
        features.push((1.0 + usage_count as f64).ln() / 5.0);

        // 6. Glue count (number of different decision levels)
        features.push(glue_count as f64 / 20.0);

        // 7. Decision level where learned
        features.push(decision_level as f64 / 100.0);

        // 8. Backtrack level
        features.push(backtrack_level as f64 / 100.0);

        // 9. LBD/size ratio (quality indicator)
        let lbd_size_ratio = if size > 0 {
            lbd as f64 / size as f64
        } else {
            1.0
        };
        features.push(lbd_size_ratio);

        // 10. Usage rate (usage/age)
        let usage_rate = if age > 0 {
            usage_count as f64 / age as f64
        } else {
            0.0
        };
        features.push(usage_rate.min(1.0));

        // 11. Is it a glue clause? (LBD <= 2)
        features.push(if lbd <= 2 { 1.0 } else { 0.0 });

        // 12. Recency (inverse of age)
        features.push(1.0 / (1.0 + age as f64));

        // Pad to CLAUSE_FEATURE_SIZE
        features.resize(CLAUSE_FEATURE_SIZE, 0.0);

        Self { features }
    }
}

/// Usefulness predictor configuration
#[derive(Debug, Clone)]
pub struct UsefulnessConfig {
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Enable online learning
    pub online_learning: bool,
}

impl Default for UsefulnessConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            learning_rate: 0.01,
            online_learning: true,
        }
    }
}

/// Clause usefulness predictor
pub struct UsefulnessPredictor {
    /// ML model
    model: LinearRegression,
    /// Configuration
    config: UsefulnessConfig,
    /// Statistics
    stats: MLStats,
}

impl UsefulnessPredictor {
    /// Create a new usefulness predictor
    pub fn new(config: UsefulnessConfig) -> Self {
        Self {
            model: LinearRegression::new(CLAUSE_FEATURE_SIZE),
            config,
            stats: MLStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(UsefulnessConfig::default())
    }

    /// Predict clause usefulness
    pub fn predict_usefulness(&mut self, features: &ClauseFeatures) -> UsefulnessPrediction {
        let start = std::time::Instant::now();

        let prediction = self.model.predict(&features.features);
        let score = prediction.first().copied().unwrap_or(0.5);

        // Normalize score to [0, 1]
        let normalized_score = (score.clamp(0.0, 1.0) + 1.0) / 2.0;

        // Confidence based on how far from 0.5
        let confidence = (normalized_score - 0.5).abs() * 2.0;

        let elapsed = start.elapsed().as_micros() as u64;
        self.stats.record_prediction_time(elapsed);

        UsefulnessPrediction::new(normalized_score, confidence)
    }

    /// Learn from feedback
    pub fn learn_from_feedback(&mut self, features: &ClauseFeatures, feedback: ClauseFeedback) {
        if !self.config.online_learning {
            return;
        }

        let start = std::time::Instant::now();

        // Compute target based on actual usefulness
        let usage_rate = if feedback.age > 0 {
            (feedback.usage_count as f64 / feedback.age as f64).min(1.0)
        } else {
            0.0
        };

        let target = if feedback.was_useful {
            (0.5 + usage_rate * 0.5).min(1.0)
        } else {
            (0.5 - usage_rate * 0.5).max(0.0)
        };

        if let Err(e) = self.model.train(&features.features, &[target]) {
            eprintln!("Training error: {}", e);
        }

        if feedback.was_useful {
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
    fn test_clause_features_extract() {
        let features = ClauseFeatures::extract(
            3,   // lbd
            10,  // size
            0.5, // activity
            100, // age
            5,   // usage_count
            2,   // glue_count
            20,  // decision_level
            15,  // backtrack_level
        );

        assert_eq!(features.features.len(), CLAUSE_FEATURE_SIZE);
        assert!(features.features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_usefulness_predictor_creation() {
        let predictor = UsefulnessPredictor::default_config();
        assert_eq!(predictor.stats.predictions, 0);
    }

    #[test]
    fn test_usefulness_predictor_predict() {
        let mut predictor = UsefulnessPredictor::default_config();
        let features = ClauseFeatures::extract(3, 10, 0.5, 100, 5, 2, 20, 15);

        let prediction = predictor.predict_usefulness(&features);
        assert!(prediction.score >= 0.0 && prediction.score <= 1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[test]
    fn test_usefulness_predictor_learn() {
        let mut predictor = UsefulnessPredictor::default_config();
        let features = ClauseFeatures::extract(3, 10, 0.5, 100, 5, 2, 20, 15);

        let feedback = ClauseFeedback {
            was_useful: true,
            usage_count: 10,
            age: 100,
        };

        predictor.learn_from_feedback(&features, feedback);
        assert_eq!(predictor.stats.correct, 1);
    }
}
