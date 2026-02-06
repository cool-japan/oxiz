//! Linear Models for Fast Inference
#![allow(clippy::needless_range_loop, clippy::manual_memcpy)] // Matrix algorithms
//!
//! Provides linear regression and logistic regression.
//! Ultra-fast inference (<1μs per prediction).

use super::optimizer::Optimizer;
use super::tensor::{Tensor, TensorOps};
use super::{Model, ModelError, ModelResult};
use serde::{Deserialize, Serialize};

/// Linear regression model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Weight vector
    weights: Tensor,
    /// Bias term
    bias: f64,
    /// Input dimension
    input_dim: usize,
    /// Regularization parameter (L2)
    regularization: f64,
}

impl LinearRegression {
    /// Create new linear regression model
    pub fn new(input_dim: usize) -> Self {
        Self {
            weights: Tensor::zeros(&[input_dim]),
            bias: 0.0,
            input_dim,
            regularization: 0.0,
        }
    }

    /// Create with L2 regularization
    pub fn with_regularization(input_dim: usize, lambda: f64) -> Self {
        Self {
            weights: Tensor::zeros(&[input_dim]),
            bias: 0.0,
            input_dim,
            regularization: lambda,
        }
    }

    /// Fit using normal equations (closed-form solution)
    #[allow(non_snake_case)] // Mathematical notation: X, XtX, Xty are conventional
    pub fn fit_closed_form(&mut self, features: &[Vec<f64>], targets: &[f64]) -> ModelResult<()> {
        if features.is_empty() || targets.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        if features.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: features.len(),
                got: targets.len(),
            });
        }

        let n = features.len();
        let d = self.input_dim;

        // Add bias column to features (X becomes [X, 1])
        let mut X = vec![vec![0.0; d + 1]; n];
        for i in 0..n {
            for j in 0..d {
                X[i][j] = features[i][j];
            }
            X[i][d] = 1.0; // Bias term
        }

        // Compute X^T * X
        let mut XtX = vec![vec![0.0; d + 1]; d + 1];
        for i in 0..(d + 1) {
            for j in 0..(d + 1) {
                for k in 0..n {
                    XtX[i][j] += X[k][i] * X[k][j];
                }
                // Add regularization to diagonal (except bias)
                if i == j && i < d {
                    XtX[i][j] += self.regularization;
                }
            }
        }

        // Compute X^T * y
        let mut Xty = vec![0.0; d + 1];
        for i in 0..(d + 1) {
            for k in 0..n {
                Xty[i] += X[k][i] * targets[k];
            }
        }

        // Solve XtX * w = Xty using Gaussian elimination
        let w = self.solve_linear_system(&XtX, &Xty)?;

        // Extract weights and bias
        for i in 0..d {
            self.weights.data[i] = w[i];
        }
        self.bias = w[d];

        Ok(())
    }

    /// Solve linear system Ax = b using Gaussian elimination
    #[allow(non_snake_case)] // Mathematical notation: A is the coefficient matrix
    fn solve_linear_system(&self, A: &[Vec<f64>], b: &[f64]) -> ModelResult<Vec<f64>> {
        let n = A.len();
        if n == 0 || b.len() != n {
            return Err(ModelError::InvalidConfig(
                "Invalid matrix dimensions".to_string(),
            ));
        }

        // Create augmented matrix [A | b]
        let mut aug = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = A[i][j];
            }
            aug[i][n] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[k][i].abs() > aug[max_row][i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            aug.swap(i, max_row);

            // Check for singular matrix
            if aug[i][i].abs() < 1e-10 {
                return Err(ModelError::NumericalError("Singular matrix".to_string()));
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = aug[k][i] / aug[i][i];
                for j in i..(n + 1) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = aug[i][n];
            for j in (i + 1)..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }

        Ok(x)
    }

    /// Fit using gradient descent
    pub fn fit_gradient_descent<O: Optimizer>(
        &mut self,
        features: &[Vec<f64>],
        targets: &[f64],
        optimizer: &mut O,
        epochs: usize,
    ) -> ModelResult<Vec<f64>> {
        if features.is_empty() || targets.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        let mut losses = Vec::new();

        for _ in 0..epochs {
            let mut total_loss = 0.0;

            for (feature, &target) in features.iter().zip(targets.iter()) {
                // Forward pass
                let pred = self.predict_single(feature);

                // Loss (MSE)
                let error = pred - target;
                total_loss += 0.5 * error * error;

                // Gradients
                let feature_tensor = Tensor::from_slice(feature);
                let grad_weights = feature_tensor.scale(error);
                let grad_bias = error;

                // Update weights
                optimizer.step(0, &mut self.weights, &grad_weights);

                // Update bias
                self.bias -= optimizer.learning_rate() * grad_bias;
            }

            losses.push(total_loss / features.len() as f64);
        }

        Ok(losses)
    }

    /// Predict for single sample (internal, no bounds check)
    fn predict_single(&self, features: &[f64]) -> f64 {
        let mut result = self.bias;
        for i in 0..features.len().min(self.weights.data.len()) {
            result += self.weights.data[i] * features[i];
        }
        result
    }

    /// Get weights
    pub fn weights(&self) -> &[f64] {
        &self.weights.data
    }

    /// Get bias
    pub fn bias(&self) -> f64 {
        self.bias
    }
}

/// Logistic regression model (binary classification)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    /// Weight vector
    weights: Tensor,
    /// Bias term
    bias: f64,
    /// Input dimension
    input_dim: usize,
    /// Regularization parameter (L2)
    regularization: f64,
}

impl LogisticRegression {
    /// Create new logistic regression model
    pub fn new(input_dim: usize) -> Self {
        Self {
            weights: Tensor::zeros(&[input_dim]),
            bias: 0.0,
            input_dim,
            regularization: 0.0,
        }
    }

    /// Create with L2 regularization
    pub fn with_regularization(input_dim: usize, lambda: f64) -> Self {
        Self {
            weights: Tensor::zeros(&[input_dim]),
            bias: 0.0,
            input_dim,
            regularization: lambda,
        }
    }

    /// Sigmoid function
    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Fit using gradient descent
    pub fn fit<O: Optimizer>(
        &mut self,
        features: &[Vec<f64>],
        targets: &[f64],
        optimizer: &mut O,
        epochs: usize,
    ) -> ModelResult<Vec<f64>> {
        if features.is_empty() || targets.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        let mut losses = Vec::new();

        for _ in 0..epochs {
            let mut total_loss = 0.0;

            for (feature, &target) in features.iter().zip(targets.iter()) {
                // Forward pass
                let z = self.predict_logit(feature);
                let pred = self.sigmoid(z);

                // Cross-entropy loss
                let pred_clamped = pred.clamp(1e-15, 1.0 - 1e-15);
                let loss =
                    -(target * pred_clamped.ln() + (1.0 - target) * (1.0 - pred_clamped).ln());
                total_loss += loss;

                // Gradient
                let error = pred - target;

                // Update weights
                let feature_tensor = Tensor::from_slice(feature);
                let mut grad_weights = feature_tensor.scale(error);

                // Add regularization gradient
                if self.regularization > 0.0 {
                    let reg_grad = self.weights.scale(self.regularization);
                    grad_weights = grad_weights.add(&reg_grad)?;
                }

                optimizer.step(0, &mut self.weights, &grad_weights);

                // Update bias
                self.bias -= optimizer.learning_rate() * error;
            }

            losses.push(total_loss / features.len() as f64);
        }

        Ok(losses)
    }

    /// Predict logit (before sigmoid)
    fn predict_logit(&self, features: &[f64]) -> f64 {
        let mut result = self.bias;
        for i in 0..features.len().min(self.weights.data.len()) {
            result += self.weights.data[i] * features[i];
        }
        result
    }

    /// Predict probability
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let logit = self.predict_logit(features);
        self.sigmoid(logit)
    }

    /// Get weights
    pub fn weights(&self) -> &[f64] {
        &self.weights.data
    }

    /// Get bias
    pub fn bias(&self) -> f64 {
        self.bias
    }
}

/// Unified linear model trait
pub trait LinearModel {
    /// Fit model to data
    fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) -> ModelResult<()>;

    /// Get model weights
    fn weights(&self) -> &[f64];

    /// Get bias term
    fn bias(&self) -> f64;
}

impl Model for LinearRegression {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        1
    }

    fn predict(&self, input: &[f64]) -> Vec<f64> {
        vec![self.predict_single(input)]
    }

    fn train(&mut self, input: &[f64], target: &[f64]) -> ModelResult<f64> {
        if target.len() != 1 {
            return Err(ModelError::DimensionMismatch {
                expected: 1,
                got: target.len(),
            });
        }

        // Online gradient descent step
        let pred = self.predict_single(input);
        let error = pred - target[0];
        let loss = 0.5 * error * error;

        // Update weights: w = w - lr * error * x
        let lr = 0.01; // Default learning rate for online updates
        for i in 0..input.len().min(self.weights.data.len()) {
            self.weights.data[i] -= lr * error * input[i];
        }
        self.bias -= lr * error;

        Ok(loss)
    }

    fn num_parameters(&self) -> usize {
        self.input_dim + 1 // weights + bias
    }

    fn save(&self) -> ModelResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| ModelError::SerializationError(e.to_string()))
    }

    fn load(&mut self, data: &[u8]) -> ModelResult<()> {
        let loaded: LinearRegression = serde_json::from_slice(data)
            .map_err(|e| ModelError::SerializationError(e.to_string()))?;

        self.weights = loaded.weights;
        self.bias = loaded.bias;
        self.input_dim = loaded.input_dim;
        self.regularization = loaded.regularization;

        Ok(())
    }
}

impl Model for LogisticRegression {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        1
    }

    fn predict(&self, input: &[f64]) -> Vec<f64> {
        vec![self.predict_proba(input)]
    }

    fn train(&mut self, input: &[f64], target: &[f64]) -> ModelResult<f64> {
        if target.len() != 1 {
            return Err(ModelError::DimensionMismatch {
                expected: 1,
                got: target.len(),
            });
        }

        // Online gradient descent step
        let pred = self.predict_proba(input);
        let pred_clamped = pred.clamp(1e-15, 1.0 - 1e-15);
        let loss = -(target[0] * pred_clamped.ln() + (1.0 - target[0]) * (1.0 - pred_clamped).ln());

        let error = pred - target[0];

        // Update weights
        let lr = 0.01;
        for i in 0..input.len().min(self.weights.data.len()) {
            self.weights.data[i] -= lr * error * input[i];
        }
        self.bias -= lr * error;

        Ok(loss)
    }

    fn num_parameters(&self) -> usize {
        self.input_dim + 1
    }

    fn save(&self) -> ModelResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| ModelError::SerializationError(e.to_string()))
    }

    fn load(&mut self, data: &[u8]) -> ModelResult<()> {
        let loaded: LogisticRegression = serde_json::from_slice(data)
            .map_err(|e| ModelError::SerializationError(e.to_string()))?;

        self.weights = loaded.weights;
        self.bias = loaded.bias;
        self.input_dim = loaded.input_dim;
        self.regularization = loaded.regularization;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::optimizer::SGD;
    use super::*;

    #[test]
    fn test_linear_regression_creation() {
        let model = LinearRegression::new(5);
        assert_eq!(model.input_dim(), 5);
        assert_eq!(model.num_parameters(), 6);
    }

    #[test]
    fn test_linear_regression_predict() {
        let mut model = LinearRegression::new(2);
        model.weights.data = vec![2.0, 3.0];
        model.bias = 1.0;

        let pred = model.predict(&[1.0, 1.0]);
        assert_eq!(pred[0], 6.0); // 2*1 + 3*1 + 1 = 6
    }

    #[test]
    fn test_linear_regression_fit_simple() {
        let mut model = LinearRegression::new(1);

        let features = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![2.0, 4.0, 6.0]; // y = 2*x

        model.fit_closed_form(&features, &targets).unwrap();

        // Check if learned approximately y = 2*x
        let pred = model.predict(&[4.0]);
        assert!((pred[0] - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_linear_regression_gradient_descent() {
        let mut model = LinearRegression::new(1);
        let mut optimizer = SGD::new(0.01);

        let features = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![2.0, 4.0, 6.0];

        let losses = model
            .fit_gradient_descent(&features, &targets, &mut optimizer, 100)
            .unwrap();

        // Loss should decrease
        assert!(losses.last().unwrap() < losses.first().unwrap());
    }

    #[test]
    fn test_logistic_regression_creation() {
        let model = LogisticRegression::new(5);
        assert_eq!(model.input_dim(), 5);
    }

    #[test]
    fn test_logistic_regression_sigmoid() {
        let model = LogisticRegression::new(1);
        assert_eq!(model.sigmoid(0.0), 0.5);
        assert!(model.sigmoid(10.0) > 0.99);
        assert!(model.sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_logistic_regression_predict() {
        let mut model = LogisticRegression::new(2);
        model.weights.data = vec![1.0, 1.0];
        model.bias = 0.0;

        let prob = model.predict_proba(&[1.0, 1.0]);
        assert!(prob > 0.5); // Positive weights -> high probability
    }

    #[test]
    fn test_logistic_regression_fit() {
        let mut model = LogisticRegression::new(2);
        let mut optimizer = SGD::new(0.1);

        // Simple linearly separable data
        let features = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
        ];
        let targets = vec![1.0, 1.0, 0.0, 0.0];

        let losses = model.fit(&features, &targets, &mut optimizer, 50).unwrap();

        // Loss should decrease
        assert!(losses.last().unwrap() < losses.first().unwrap());
    }

    #[test]
    fn test_linear_regression_save_load() {
        let mut model = LinearRegression::new(2);
        model.weights.data = vec![1.0, 2.0];
        model.bias = 3.0;

        let saved = model.save().unwrap();

        let mut model2 = LinearRegression::new(2);
        model2.load(&saved).unwrap();

        assert_eq!(model2.weights.data, vec![1.0, 2.0]);
        assert_eq!(model2.bias, 3.0);
    }
}
