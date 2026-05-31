//! Training dataset for ML difficulty predictors.
//!
//! [`Dataset`] stores [`Sample`]s, each pairing a [`Features`] vector with the
//! observed runtime of the corresponding benchmark run.  Utility methods
//! support shuffling, train/test split, and k-fold cross-validation.

use super::features::Features;
use crate::benchmark::{BenchmarkStatus, SingleResult};
use crate::loader::BenchmarkMeta;
use rand::Rng;
use rand::seq::SliceRandom;

/// A single labelled training example.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Feature vector extracted from the benchmark metadata.
    pub features: Features,
    /// Observed solver runtime in seconds.
    pub runtime_seconds: f64,
    /// Final solver verdict for this run.
    pub status: BenchmarkStatus,
}

/// A collection of [`Sample`]s used for training or evaluation.
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// Ordered list of training samples.
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// Create an empty dataset.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a dataset by joining benchmark metadata with run results.
    ///
    /// * Only `Sat` and `Unsat` results are included — timeouts, errors, and
    ///   unknowns are excluded to avoid polluting the training signal.
    /// * Matching is done by `path` equality.
    #[must_use]
    pub fn from_results(metas: &[BenchmarkMeta], results: &[SingleResult]) -> Self {
        use std::collections::HashMap;
        let meta_map: HashMap<_, _> = metas.iter().map(|m| (&m.path, m)).collect();

        let samples = results
            .iter()
            .filter(|r| matches!(r.status, BenchmarkStatus::Sat | BenchmarkStatus::Unsat))
            .filter_map(|r| {
                let meta = meta_map.get(&r.path)?;
                let features = Features::from_meta(meta);
                let runtime_seconds = r.time.as_secs_f64();
                Some(Sample {
                    features,
                    runtime_seconds,
                    status: r.status,
                })
            })
            .collect();

        Self { samples }
    }

    /// Append a single sample to the dataset.
    pub fn push(&mut self, sample: Sample) {
        self.samples.push(sample);
    }

    /// Number of samples in the dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// `true` if the dataset contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Randomly shuffle and split into two datasets with `train_frac` in the
    /// first partition (clamped to `[0, 1]`).
    ///
    /// Use a seeded `rng` for reproducibility.
    #[must_use]
    pub fn shuffle_split(&self, train_frac: f64, rng: &mut impl Rng) -> (Dataset, Dataset) {
        let mut indices: Vec<usize> = (0..self.samples.len()).collect();
        indices.shuffle(rng);

        let train_frac = train_frac.clamp(0.0, 1.0);
        let n_train = ((self.samples.len() as f64) * train_frac).round() as usize;
        let n_train = n_train.min(self.samples.len());

        let train_samples = indices[..n_train]
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();
        let test_samples = indices[n_train..]
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();

        (
            Dataset {
                samples: train_samples,
            },
            Dataset {
                samples: test_samples,
            },
        )
    }

    /// Generate `k` non-overlapping (train, validation) fold pairs.
    ///
    /// Each pair uses `(k-1)/k` of the data for training and `1/k` for
    /// validation.  If `k == 0` or `k > n`, returns an empty `Vec`.
    #[must_use]
    pub fn k_fold(&self, k: usize) -> Vec<(Dataset, Dataset)> {
        let n = self.samples.len();
        if k == 0 || k > n {
            return Vec::new();
        }

        let fold_size = n / k;
        let mut folds = Vec::with_capacity(k);

        for fold_idx in 0..k {
            let val_start = fold_idx * fold_size;
            let val_end = if fold_idx == k - 1 {
                n
            } else {
                val_start + fold_size
            };

            let val_samples: Vec<Sample> = self.samples[val_start..val_end].to_vec();
            let train_samples: Vec<Sample> = self.samples[..val_start]
                .iter()
                .chain(self.samples[val_end..].iter())
                .cloned()
                .collect();

            folds.push((
                Dataset {
                    samples: train_samples,
                },
                Dataset {
                    samples: val_samples,
                },
            ));
        }

        folds
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_sample(rt: f64) -> Sample {
        Sample {
            features: Features::default(),
            runtime_seconds: rt,
            status: BenchmarkStatus::Sat,
        }
    }

    #[test]
    fn test_dataset_push_len() {
        let mut ds = Dataset::new();
        assert!(ds.is_empty());
        ds.push(make_sample(1.0));
        ds.push(make_sample(2.0));
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_shuffle_split_sizes() {
        let mut ds = Dataset::new();
        for i in 0..10 {
            ds.push(make_sample(i as f64));
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (train, test) = ds.shuffle_split(0.8, &mut rng);
        assert_eq!(train.len() + test.len(), 10);
        assert_eq!(train.len(), 8);
    }

    #[test]
    fn test_k_fold_counts() {
        let mut ds = Dataset::new();
        for i in 0..10 {
            ds.push(make_sample(i as f64));
        }
        let folds = ds.k_fold(5);
        assert_eq!(folds.len(), 5);
        for (train, val) in &folds {
            assert_eq!(train.len() + val.len(), 10);
            assert_eq!(val.len(), 2);
        }
    }

    #[test]
    fn test_k_fold_zero_returns_empty() {
        let mut ds = Dataset::new();
        ds.push(make_sample(1.0));
        assert!(ds.k_fold(0).is_empty());
    }
}
