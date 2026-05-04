//! k-Nearest Neighbours regression model.
//!
//! A lazy (instance-based) learner that predicts runtime via inverse-distance
//! weighted averaging of the `k` nearest neighbours in normalised feature space.
//! Theory bits (indices 0..10) receive 2× weight in the distance metric to
//! give priority to theory-level similarity over structural features.

use std::time::Instant;

use super::class::DifficultyClass;
use super::dataset::Dataset;
use super::features::{FeatureNormalizer, Features};
use super::models::DifficultyModel;
use super::report::{TrainingConfig, TrainingReport};

/// Small constant added to distances to prevent division-by-zero.
const EPSILON: f64 = 1e-9;

/// k-NN regressor operating on normalised features.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KnnRegressor {
    /// Number of neighbours to consider.
    pub k: usize,
    /// Normalizer fitted on training data.
    pub normalizer: FeatureNormalizer,
    /// Stored training points as `(normalised_features, log1p_runtime)`.
    pub training_samples: Vec<(Vec<f64>, f64)>,
    /// Whether `fit` has been called.
    pub is_fitted: bool,
}

impl KnnRegressor {
    /// Create a new unfitted k-NN regressor.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            normalizer: FeatureNormalizer::default(),
            training_samples: Vec::new(),
            is_fitted: false,
        }
    }

    /// Deserialise from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`serde_json::Error`] on malformed JSON or schema mismatch.
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Weighted Euclidean distance between two normalised feature vectors.
    ///
    /// Theory bits (indices 0..10) are weighted 2×; remaining features 1×.
    ///
    /// `sqrt(Σ w_i * (a_i - b_i)^2)`
    fn weighted_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .enumerate()
            .map(|(i, (&ai, &bi))| {
                let w = if i < 10 { 2.0 } else { 1.0 };
                let diff = ai - bi;
                w * diff * diff
            })
            .sum::<f64>()
            .sqrt()
    }
}

impl DifficultyModel for KnnRegressor {
    fn name(&self) -> &'static str {
        "knn"
    }

    fn predict_runtime(&self, features: &Features) -> f64 {
        if self.training_samples.is_empty() {
            return 0.0;
        }

        let norm = self.normalizer.normalize(features);
        let k = self.k.min(self.training_samples.len());

        // Compute distances to all training points
        let mut distances: Vec<(f64, f64)> = self
            .training_samples
            .iter()
            .map(|(x, y)| (Self::weighted_euclidean_distance(&norm, x), *y))
            .collect();

        // Partial sort: move the k smallest distances to the front
        distances.select_nth_unstable_by(k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        let neighbours = &distances[..k];

        // Inverse-distance weighted average of log1p_runtime
        let (weighted_sum, weight_sum) = neighbours.iter().fold(
            (0.0f64, 0.0f64),
            |(ws, w_total), &(d, y)| {
                let inv_d = 1.0 / (d + EPSILON);
                (ws + y * inv_d, w_total + inv_d)
            },
        );

        if weight_sum < EPSILON {
            return 0.0;
        }

        let log_rt = weighted_sum / weight_sum;
        log_rt.exp_m1().max(0.0)
    }

    fn fit(
        &mut self,
        dataset: &Dataset,
        _config: &TrainingConfig,
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

        // Fit normalizer
        let all_features: Vec<Features> = dataset
            .samples
            .iter()
            .map(|s| s.features.clone())
            .collect();
        self.normalizer = FeatureNormalizer::fit(&all_features);

        // Store normalised training samples
        self.training_samples = dataset
            .samples
            .iter()
            .map(|s| {
                let x = self.normalizer.normalize(&s.features);
                let y = s.runtime_seconds.ln_1p();
                (x, y)
            })
            .collect();

        // Evaluate on training set (leave-all-in, not cross-validated)
        let mut total_ae = 0.0f64;
        let mut correct = 0usize;
        let nf = n as f64;

        for s in &dataset.samples {
            let predicted_rt = self.predict_runtime(&s.features);
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
            final_loss: 0.0,
            epochs: 1,
            mae_seconds,
            class_accuracy,
            k_fold_mean_mae: None,
            time_to_train: start.elapsed(),
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).expect("serialization infallible for KnnRegressor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkStatus;
    use crate::predictor::dataset::{Dataset, Sample};
    use rand::SeedableRng;

    fn make_sample(atom: f64, rt: f64) -> Sample {
        Sample {
            features: Features { atom_count: atom, ..Default::default() },
            runtime_seconds: rt,
            status: BenchmarkStatus::Sat,
        }
    }

    #[test]
    fn test_knn_k1_nearest() {
        let mut ds = Dataset::new();
        ds.push(make_sample(0.0, 0.05)); // trivial
        ds.push(make_sample(50.0, 5.0)); // medium
        ds.push(make_sample(100.0, 30.0)); // hard

        let mut model = KnnRegressor::new(1);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);

        // Query exactly at atom=0 should be closest to first sample
        let query = Features { atom_count: 0.0, ..Default::default() };
        let rt = model.predict_runtime(&query);
        // Should be close to 0.05
        assert!(rt < 1.0, "Expected trivial runtime, got {rt}");
    }

    #[test]
    fn test_knn_handles_zero_distance() {
        let mut ds = Dataset::new();
        ds.push(make_sample(42.0, 1.0));
        ds.push(make_sample(42.0, 1.0)); // duplicate

        let mut model = KnnRegressor::new(2);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);

        let query = Features { atom_count: 42.0, ..Default::default() };
        let rt = model.predict_runtime(&query);
        assert!(rt.is_finite(), "NaN/Inf when distance is zero: got {rt}");
    }
}
