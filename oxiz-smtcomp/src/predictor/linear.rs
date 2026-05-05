//! Linear regression difficulty model.
//!
//! Trains a linear model `log1p(runtime) = w · x + b` using full-batch
//! gradient descent with Armijo backtracking line-search and L2
//! regularisation on the weights (not the bias).

use std::time::Instant;

use super::class::DifficultyClass;
use super::dataset::Dataset;
use super::features::{FEATURE_DIM, FeatureNormalizer, Features};
use super::models::DifficultyModel;
use super::report::{TrainingConfig, TrainingReport};

/// Linear regressor predicting `log1p(runtime_s)` from normalised features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LinearRegressor {
    /// Weight vector (length `FEATURE_DIM`).
    pub weights: Vec<f64>,
    /// Bias term.
    pub bias: f64,
    /// Feature normalizer fitted on training data.
    pub normalizer: FeatureNormalizer,
    /// L2 regularisation coefficient applied to weights only.
    pub l2: f64,
    /// Whether `fit` has been called at least once.
    pub is_fitted: bool,
}

impl LinearRegressor {
    /// Create an unfitted regressor with the given L2 coefficient.
    #[must_use]
    pub fn new(l2: f64) -> Self {
        Self {
            weights: vec![0.0; FEATURE_DIM],
            bias: 0.0,
            normalizer: FeatureNormalizer::default(),
            l2,
            is_fitted: false,
        }
    }

    /// Deserialise from a JSON string produced by [`to_json`].
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] if the string is not valid JSON or the
    /// schema does not match.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Compute the dot product of `normalized` with `self.weights` plus bias.
    fn predict_raw(&self, normalized: &[f64]) -> f64 {
        let dot: f64 = self
            .weights
            .iter()
            .zip(normalized.iter())
            .map(|(w, x)| w * x)
            .sum();
        dot + self.bias
    }

    /// MSE loss on `log1p(runtime)` plus L2 penalty.
    fn compute_loss(&self, dataset: &Dataset, normalized_batch: &[Vec<f64>]) -> f64 {
        let n = dataset.samples.len();
        if n == 0 {
            return 0.0;
        }

        let mse: f64 = dataset
            .samples
            .iter()
            .zip(normalized_batch.iter())
            .map(|(s, x)| {
                let y = s.runtime_seconds.ln_1p();
                let y_hat = self.predict_raw(x);
                let err = y_hat - y;
                err * err
            })
            .sum::<f64>()
            / n as f64;

        let l2_penalty: f64 = self.weights.iter().map(|w| w * w).sum::<f64>() * self.l2;
        mse + l2_penalty
    }
}

impl DifficultyModel for LinearRegressor {
    fn name(&self) -> &'static str {
        "linear"
    }

    fn predict_runtime(&self, features: &Features) -> f64 {
        let norm = self.normalizer.normalize(features);
        let log_rt = self.predict_raw(&norm);
        log_rt.exp_m1().max(0.0)
    }

    fn fit(
        &mut self,
        dataset: &Dataset,
        config: &TrainingConfig,
        _rng: &mut dyn rand::Rng,
    ) -> TrainingReport {
        let start = Instant::now();
        let n = dataset.samples.len();

        if n == 0 {
            self.is_fitted = true;
            return TrainingReport {
                final_loss: 0.0,
                epochs: 0,
                mae_seconds: 0.0,
                class_accuracy: 0.0,
                k_fold_mean_mae: None,
                time_to_train: start.elapsed(),
            };
        }

        // 1. Fit normalizer
        let all_features: Vec<Features> =
            dataset.samples.iter().map(|s| s.features.clone()).collect();
        self.normalizer = FeatureNormalizer::fit(&all_features);

        // 2. Pre-normalise all features
        let normalized: Vec<Vec<f64>> = dataset
            .samples
            .iter()
            .map(|s| self.normalizer.normalize(&s.features))
            .collect();

        // 3. Initialise weights to small values
        self.weights = vec![0.01; FEATURE_DIM];
        self.bias = 0.0;

        let nf = n as f64;
        let mut prev_loss = self.compute_loss(dataset, &normalized);
        let mut patience_count = 0usize;
        let mut actual_epochs = 0usize;

        for _epoch in 0..config.max_epochs {
            actual_epochs += 1;

            // Compute gradients (full batch)
            let mut grad_w = [0.0f64; FEATURE_DIM];
            let mut grad_b = 0.0f64;

            for (s, x) in dataset.samples.iter().zip(normalized.iter()) {
                let y = s.runtime_seconds.ln_1p();
                let y_hat = self.predict_raw(x);
                let residual = y_hat - y;
                grad_b += residual;
                for (j, &xj) in x.iter().enumerate() {
                    grad_w[j] += residual * xj;
                }
            }

            // Average + L2 gradient on weights
            for (gw, w) in grad_w.iter_mut().zip(self.weights.iter()) {
                *gw = *gw / nf + 2.0 * self.l2 * w;
            }
            grad_b /= nf;

            // Armijo backtracking line-search
            let mut lr = config.learning_rate;
            let grad_norm_sq: f64 = grad_w.iter().map(|g| g * g).sum::<f64>() + grad_b * grad_b;
            let armijo_c = 0.5;
            let armijo_rho = 0.5;

            loop {
                // Tentative update
                let mut new_w = self.weights.clone();
                for (nw, &gw) in new_w.iter_mut().zip(grad_w.iter()) {
                    *nw -= lr * gw;
                }
                let new_b = self.bias - lr * grad_b;

                let old_w = std::mem::replace(&mut self.weights, new_w);
                let old_b = self.bias;
                self.bias = new_b;

                let new_loss = self.compute_loss(dataset, &normalized);
                let sufficient_decrease = prev_loss - armijo_c * lr * grad_norm_sq;

                if new_loss <= sufficient_decrease || lr < 1e-12 {
                    // Accept step
                    prev_loss = new_loss;
                    break;
                }

                // Reject step, restore
                self.weights = old_w;
                self.bias = old_b;
                lr *= armijo_rho;
            }

            // Early stopping
            let loss_improvement = prev_loss - self.compute_loss(dataset, &normalized);
            if loss_improvement.abs() < config.convergence_tol {
                patience_count += 1;
                if patience_count >= config.convergence_patience {
                    break;
                }
            } else {
                patience_count = 0;
            }
        }

        let final_loss = self.compute_loss(dataset, &normalized);

        // Compute MAE and class accuracy on training set
        let mut total_ae = 0.0f64;
        let mut correct = 0usize;

        for (s, x) in dataset.samples.iter().zip(normalized.iter()) {
            let predicted_rt = self.predict_raw(x).exp_m1().max(0.0);
            total_ae += (predicted_rt - s.runtime_seconds).abs();
            let pred_class = DifficultyClass::from_runtime_seconds(predicted_rt);
            let actual_class = DifficultyClass::from_runtime_seconds(s.runtime_seconds);
            if pred_class == actual_class {
                correct += 1;
            }
        }

        let mae_seconds = total_ae / nf;
        let class_accuracy = correct as f64 / nf;

        self.is_fitted = true;

        TrainingReport {
            final_loss,
            epochs: actual_epochs,
            mae_seconds,
            class_accuracy,
            k_fold_mean_mae: None,
            time_to_train: start.elapsed(),
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).expect("serialization infallible for LinearRegressor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::dataset::Sample;
    use rand::SeedableRng;

    fn make_dataset_linear(n: usize) -> Dataset {
        // runtime = exp(0.5 * atom_count / 100) - 1  (roughly linear in log-space)
        let mut ds = Dataset::new();
        for i in 0..n {
            let f = Features {
                atom_count: i as f64,
                ..Default::default()
            };
            let rt = (0.5 * i as f64 / 100.0_f64).exp() - 1.0;
            ds.push(Sample {
                features: f,
                runtime_seconds: rt.max(0.001),
                status: crate::benchmark::BenchmarkStatus::Sat,
            });
        }
        ds
    }

    #[test]
    fn test_linear_fits_and_predicts() {
        let ds = make_dataset_linear(50);
        let mut model = LinearRegressor::new(1e-3);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let report = model.fit(&ds, &config, &mut rng);
        assert!(model.is_fitted);
        assert!(report.epochs > 0);
        assert!(report.mae_seconds >= 0.0);
    }

    #[test]
    fn test_linear_json_round_trip() {
        let ds = make_dataset_linear(20);
        let mut model = LinearRegressor::new(1e-3);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        model.fit(&ds, &config, &mut rng);

        let json = model.to_json();
        let restored = LinearRegressor::from_json(&json).expect("json parse failed");
        let f = Features::default();
        let p1 = model.predict_runtime(&f);
        let p2 = restored.predict_runtime(&f);
        assert!((p1 - p2).abs() < 1e-9);
    }
}
