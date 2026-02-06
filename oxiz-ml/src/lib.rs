//! OxiZ-ML: Machine Learning-Guided Heuristics for SMT Solving
//!
//! This crate provides ML-guided heuristics for various aspects of SMT solving:
//! - **Branching**: Learn optimal variable selection and polarity choices
//! - **Restarts**: Adaptive restart policies based on solver progress
//! - **Clause Learning**: Predict clause usefulness for deletion decisions
//! - **Tactic Selection**: Choose best tactics based on formula features
//!
//! # Architecture
//!
//! The ML system is designed to have minimal overhead (<10% of solve time) by:
//! - Using lightweight models (small NNs, decision trees, linear models)
//! - Caching feature computations
//! - Supporting both online and offline learning
//! - Providing fast inference paths
//!
//! # Models
//!
//! Three types of models are provided:
//! - **Neural Networks**: For complex pattern recognition (2-3 hidden layers)
//! - **Decision Trees**: For fast, interpretable decisions
//! - **Linear Models**: For simple, ultra-fast predictions
//!
//! # Training
//!
//! - **Online Learning**: Update models during solving (minimal overhead)
//! - **Offline Training**: Train on collected solve traces
//! - **Transfer Learning**: Pre-trained models for common problem classes
//!
//! # Examples
//!
//! ## ML-Guided Branching
//!
//! ```rust
//! use oxiz_ml::branching::{BranchingLearner, FeatureExtractor};
//! use oxiz_ml::models::NeuralNetwork;
//!
//! // Create a branching learner with a neural network
//! let mut learner = BranchingLearner::new_with_neural_net(
//!     vec![10, 8, 4, 1],  // Layer sizes
//!     0.01,                // Learning rate
//! );
//!
//! // During solving, extract features and predict
//! // let features = extract_features(&solver_state);
//! // let best_var = learner.predict_branch(&features);
//! ```
//!
//! ## Tactic Selection
//!
//! ```rust,ignore
//! use oxiz_ml::tactic::TacticSelector;
//!
//! // Create a tactic selector
//! let config = TacticConfig::default();
//! let selector = TacticSelector::new(config);
//!
//! // Select best tactic for a formula
//! // let features = extract_formula_features(&formula);
//! // let tactic_id = selector.select_tactic(&features);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

/// Core ML model implementations
pub mod models;

/// Branching heuristic learning
pub mod branching;

/// Restart policy learning
pub mod restarts;

/// Clause learning and deletion policies
pub mod clause_learning;

/// Tactic selection
pub mod tactic;

/// Training infrastructure
pub mod training;

// Re-export commonly used types
pub use branching::{BranchingFeatures, BranchingLearner};
pub use clause_learning::{DeletionPolicy, UsefulnessPredictor};
pub use models::{DecisionTree, LinearModel, Model, NeuralNetwork};
pub use restarts::{AdaptiveRestart, RestartPolicyLearner};
pub use tactic::{FormulaFeatures, TacticSelector};
pub use training::{DataCollector, OfflineTrainer, OnlineLearner};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default minimum confidence threshold for ML predictions
pub const DEFAULT_MIN_CONFIDENCE: f64 = 0.6;

/// Default learning rate for online learning
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Default feature vector size for branching
pub const BRANCHING_FEATURE_SIZE: usize = 15;

/// Default feature vector size for restart prediction
pub const RESTART_FEATURE_SIZE: usize = 10;

/// Default feature vector size for clause usefulness
pub const CLAUSE_FEATURE_SIZE: usize = 12;

/// Default feature vector size for tactic selection
pub const TACTIC_FEATURE_SIZE: usize = 20;

/// ML prediction result
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction<T> {
    /// The predicted value
    pub value: T,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
}

impl<T> Prediction<T> {
    /// Create a new prediction with confidence
    pub fn new(value: T, confidence: f64) -> Self {
        Self {
            value,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Check if prediction meets minimum confidence threshold
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Map the prediction value using a function
    pub fn map<U, F>(self, f: F) -> Prediction<U>
    where
        F: FnOnce(T) -> U,
    {
        Prediction {
            value: f(self.value),
            confidence: self.confidence,
        }
    }
}

/// Common statistics for ML components
#[derive(Debug, Clone, Default)]
pub struct MLStats {
    /// Total predictions made
    pub predictions: usize,
    /// Correct predictions (validated)
    pub correct: usize,
    /// Incorrect predictions (validated)
    pub incorrect: usize,
    /// Total training updates
    pub training_updates: usize,
    /// Average prediction time (microseconds)
    pub avg_prediction_time_us: f64,
    /// Average training time (microseconds)
    pub avg_training_time_us: f64,
    /// Total inference time (microseconds)
    pub total_inference_time_us: u64,
    /// Total training time (microseconds)
    pub total_training_time_us: u64,
}

impl MLStats {
    /// Calculate prediction accuracy
    pub fn accuracy(&self) -> f64 {
        let validated = self.correct + self.incorrect;
        if validated == 0 {
            0.0
        } else {
            self.correct as f64 / validated as f64
        }
    }

    /// Get overhead percentage relative to total time
    pub fn overhead_percentage(&self, total_solver_time_us: u64) -> f64 {
        if total_solver_time_us == 0 {
            0.0
        } else {
            let ml_time = self.total_inference_time_us + self.total_training_time_us;
            (ml_time as f64 / total_solver_time_us as f64) * 100.0
        }
    }

    /// Update average prediction time
    pub fn record_prediction_time(&mut self, time_us: u64) {
        self.predictions += 1;
        self.total_inference_time_us += time_us;
        self.avg_prediction_time_us = self.total_inference_time_us as f64 / self.predictions as f64;
    }

    /// Update average training time
    pub fn record_training_time(&mut self, time_us: u64) {
        self.training_updates += 1;
        self.total_training_time_us += time_us;
        self.avg_training_time_us =
            self.total_training_time_us as f64 / self.training_updates as f64;
    }

    /// Record a correct prediction
    pub fn record_correct(&mut self) {
        self.correct += 1;
    }

    /// Record an incorrect prediction
    pub fn record_incorrect(&mut self) {
        self.incorrect += 1;
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_creation() {
        let pred = Prediction::new(42, 0.85);
        assert_eq!(pred.value, 42);
        assert_eq!(pred.confidence, 0.85);
    }

    #[test]
    fn test_prediction_confidence_clamping() {
        let pred1 = Prediction::new(1, 1.5);
        assert_eq!(pred1.confidence, 1.0);

        let pred2 = Prediction::new(2, -0.5);
        assert_eq!(pred2.confidence, 0.0);
    }

    #[test]
    fn test_prediction_is_confident() {
        let pred = Prediction::new(42, 0.75);
        assert!(pred.is_confident(0.6));
        assert!(!pred.is_confident(0.8));
    }

    #[test]
    fn test_prediction_map() {
        let pred = Prediction::new(10, 0.9);
        let mapped = pred.map(|x| x * 2);
        assert_eq!(mapped.value, 20);
        assert_eq!(mapped.confidence, 0.9);
    }

    #[test]
    fn test_ml_stats_accuracy() {
        let mut stats = MLStats::default();
        assert_eq!(stats.accuracy(), 0.0);

        stats.correct = 7;
        stats.incorrect = 3;
        assert_eq!(stats.accuracy(), 0.7);
    }

    #[test]
    fn test_ml_stats_overhead() {
        let stats = MLStats {
            total_inference_time_us: 1000,
            total_training_time_us: 500,
            ..Default::default()
        };

        let overhead = stats.overhead_percentage(100_000);
        assert!((overhead - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_ml_stats_record_times() {
        let mut stats = MLStats::default();

        stats.record_prediction_time(100);
        stats.record_prediction_time(200);

        assert_eq!(stats.predictions, 2);
        assert_eq!(stats.avg_prediction_time_us, 150.0);

        stats.record_training_time(500);
        stats.record_training_time(300);

        assert_eq!(stats.training_updates, 2);
        assert_eq!(stats.avg_training_time_us, 400.0);
    }
}
