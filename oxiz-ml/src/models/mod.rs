//! Core ML Model Implementations
//!
//! This module provides lightweight ML models optimized for SMT solving:
//! - Neural networks with backpropagation
//! - Decision trees with pruning
//! - Linear models (regression and classification)
//!
//! All models are designed for:
//! - Fast inference (<100μs per prediction)
//! - Online learning capability
//! - Serialization/deserialization
//! - Pure Rust implementation

pub mod activation;
pub mod decision_tree;
pub mod linear_model;
pub mod loss;
pub mod neural_network;
pub mod optimizer;
pub mod tensor;

pub use activation::{Activation, ActivationFn};
pub use decision_tree::{DecisionNode, DecisionTree, SplitCriterion, TreeConfig};
pub use linear_model::{LinearModel, LinearRegression, LogisticRegression};
pub use loss::{Loss, LossFn};
pub use neural_network::{Layer, NetworkConfig, NeuralNetwork};
pub use optimizer::{AdaGrad, Adam, Optimizer, SGD};
pub use tensor::{Tensor, TensorOps};

/// Common trait for all ML models
pub trait Model {
    /// Input feature dimension
    fn input_dim(&self) -> usize;

    /// Output dimension
    fn output_dim(&self) -> usize;

    /// Forward pass: predict output from input features
    fn predict(&self, input: &[f64]) -> Vec<f64>;

    /// Training step: update model parameters
    fn train(&mut self, input: &[f64], target: &[f64]) -> Result<f64, ModelError>;

    /// Batch training
    fn train_batch(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[Vec<f64>],
    ) -> Result<f64, ModelError> {
        if inputs.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: inputs.len(),
                got: targets.len(),
            });
        }

        let mut total_loss = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            total_loss += self.train(input, target)?;
        }

        Ok(total_loss / inputs.len() as f64)
    }

    /// Get number of trainable parameters
    fn num_parameters(&self) -> usize;

    /// Save model to bytes
    fn save(&self) -> Result<Vec<u8>, ModelError>;

    /// Load model from bytes
    fn load(&mut self, data: &[u8]) -> Result<(), ModelError>;
}

/// ML model errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ModelError {
    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        got: usize,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Training error
    #[error("Training error: {0}")]
    TrainingError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Numerical error (NaN, Inf)
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Empty input
    #[error("Empty input provided")]
    EmptyInput,

    /// Invalid tree structure
    #[error("Invalid tree structure: {0}")]
    InvalidTree(String),
}

/// Result type for model operations
pub type ModelResult<T> = Result<T, ModelError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_error_display() {
        let err = ModelError::DimensionMismatch {
            expected: 10,
            got: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }
}
