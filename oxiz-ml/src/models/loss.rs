//! Loss Functions for Training

/// Loss function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Loss {
    /// Mean Squared Error (for regression)
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Binary Cross-Entropy (for binary classification)
    BinaryCrossEntropy,
    /// Categorical Cross-Entropy (for multi-class classification)
    CategoricalCrossEntropy,
    /// Huber loss (robust to outliers)
    Huber,
}

impl Loss {
    /// Compute loss value
    pub fn compute(&self, prediction: f64, target: f64) -> f64 {
        match self {
            Loss::MSE => {
                let diff = prediction - target;
                0.5 * diff * diff
            }
            Loss::MAE => (prediction - target).abs(),
            Loss::BinaryCrossEntropy => {
                // Clip to avoid log(0)
                let pred = prediction.clamp(1e-15, 1.0 - 1e-15);
                -(target * pred.ln() + (1.0 - target) * (1.0 - pred).ln())
            }
            Loss::CategoricalCrossEntropy => {
                // For single prediction (softmax not needed here)
                let pred = prediction.clamp(1e-15, 1.0);
                -target * pred.ln()
            }
            Loss::Huber => {
                let delta = 1.0;
                let diff = (prediction - target).abs();
                if diff <= delta {
                    0.5 * diff * diff
                } else {
                    delta * (diff - 0.5 * delta)
                }
            }
        }
    }

    /// Compute gradient (derivative with respect to prediction)
    pub fn gradient(&self, prediction: f64, target: f64) -> f64 {
        match self {
            Loss::MSE => prediction - target,
            Loss::MAE => {
                if prediction > target {
                    1.0
                } else if prediction < target {
                    -1.0
                } else {
                    0.0
                }
            }
            Loss::BinaryCrossEntropy => {
                let pred = prediction.clamp(1e-15, 1.0 - 1e-15);
                -(target / pred) + (1.0 - target) / (1.0 - pred)
            }
            Loss::CategoricalCrossEntropy => {
                let pred = prediction.clamp(1e-15, 1.0);
                -target / pred
            }
            Loss::Huber => {
                let delta = 1.0;
                let diff = prediction - target;
                if diff.abs() <= delta {
                    diff
                } else if diff > 0.0 {
                    delta
                } else {
                    -delta
                }
            }
        }
    }

    /// Compute loss for vectors
    pub fn compute_vec(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(predictions.len(), targets.len());
        let sum: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &targ)| self.compute(pred, targ))
            .sum();

        sum / predictions.len() as f64
    }

    /// Compute gradient for vectors
    pub fn gradient_vec(&self, predictions: &[f64], targets: &[f64]) -> Vec<f64> {
        assert_eq!(predictions.len(), targets.len());
        predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &targ)| self.gradient(pred, targ))
            .collect()
    }
}

/// Trait for loss functions
pub trait LossFn: Send + Sync {
    /// Compute loss
    fn compute(&self, prediction: f64, target: f64) -> f64;

    /// Compute gradient
    fn gradient(&self, prediction: f64, target: f64) -> f64;

    /// Compute loss for batch
    fn compute_batch(&self, predictions: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(predictions.len(), targets.len());
        let sum: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &targ)| self.compute(pred, targ))
            .sum();

        sum / predictions.len() as f64
    }
}

impl LossFn for Loss {
    fn compute(&self, prediction: f64, target: f64) -> f64 {
        Loss::compute(self, prediction, target)
    }

    fn gradient(&self, prediction: f64, target: f64) -> f64 {
        Loss::gradient(self, prediction, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let loss = Loss::MSE;

        // Perfect prediction
        assert_eq!(loss.compute(5.0, 5.0), 0.0);

        // Error of 2
        assert_eq!(loss.compute(5.0, 3.0), 2.0); // 0.5 * (2^2) = 2.0

        // Gradient
        assert_eq!(loss.gradient(5.0, 3.0), 2.0);
        assert_eq!(loss.gradient(3.0, 5.0), -2.0);
    }

    #[test]
    fn test_mae() {
        let loss = Loss::MAE;

        assert_eq!(loss.compute(5.0, 5.0), 0.0);
        assert_eq!(loss.compute(5.0, 3.0), 2.0);
        assert_eq!(loss.compute(3.0, 5.0), 2.0);

        assert_eq!(loss.gradient(5.0, 3.0), 1.0);
        assert_eq!(loss.gradient(3.0, 5.0), -1.0);
        assert_eq!(loss.gradient(5.0, 5.0), 0.0);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let loss = Loss::BinaryCrossEntropy;

        // Perfect prediction
        let l1 = loss.compute(1.0, 1.0);
        assert!(l1 < 1e-10);

        let l2 = loss.compute(0.0, 0.0);
        assert!(l2 < 1e-10);

        // Moderate error
        let l3 = loss.compute(0.5, 1.0);
        assert!(l3 > 0.0);
    }

    #[test]
    fn test_huber() {
        let loss = Loss::Huber;

        // Small error (quadratic)
        let l1 = loss.compute(0.5, 0.0);
        assert_eq!(l1, 0.125); // 0.5 * 0.5^2

        // Large error (linear)
        let l2 = loss.compute(5.0, 0.0);
        assert_eq!(l2, 4.5); // 1.0 * (5.0 - 0.5)
    }

    #[test]
    fn test_compute_vec() {
        let loss = Loss::MSE;
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];

        let result = loss.compute_vec(&predictions, &targets);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_gradient_vec() {
        let loss = Loss::MSE;
        let predictions = vec![3.0, 5.0, 7.0];
        let targets = vec![2.0, 4.0, 6.0];

        let grads = loss.gradient_vec(&predictions, &targets);
        assert_eq!(grads, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mse_symmetry() {
        let loss = Loss::MSE;
        let l1 = loss.compute(5.0, 3.0);
        let l2 = loss.compute(3.0, 5.0);
        assert_eq!(l1, l2); // MSE is symmetric in error magnitude
    }

    #[test]
    fn test_gradient_at_zero() {
        let loss = Loss::MSE;
        assert_eq!(loss.gradient(5.0, 5.0), 0.0);
    }
}
