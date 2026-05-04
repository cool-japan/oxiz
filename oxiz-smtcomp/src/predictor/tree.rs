//! Regression tree difficulty model.
//!
//! Builds a CART-style regression tree using variance reduction as the
//! splitting criterion.  Predictions are the mean `log1p(runtime)` of the
//! leaf partition, inverse-transformed via `exp_m1`.

use std::time::Instant;

use super::class::DifficultyClass;
use super::dataset::Dataset;
use super::features::{FEATURE_DIM, FeatureNormalizer, Features};
use super::models::DifficultyModel;
use super::report::{TrainingConfig, TrainingReport};

/// A single node in the regression tree.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TreeNode {
    /// Terminal node — stores the mean `log1p(runtime)` of all training
    /// samples that fell into this partition.
    Leaf {
        /// Mean target value (in log1p-space).
        value: f64,
    },
    /// Internal split node.
    Split {
        /// Index into the normalised feature vector (0 .. `FEATURE_DIM`).
        feature_idx: usize,
        /// Split threshold; samples with `x[feature_idx] <= threshold` go left.
        threshold: f64,
        /// Left subtree (≤ threshold).
        left: Box<TreeNode>,
        /// Right subtree (> threshold).
        right: Box<TreeNode>,
    },
}

/// Regression tree trained with variance-reduction splitting.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegressionTree {
    /// Root node (None before fitting).
    pub root: Option<TreeNode>,
    /// Maximum tree depth (0 = unlimited; practical default 8).
    pub max_depth: usize,
    /// Minimum samples required at a node to attempt a split.
    pub min_samples_split: usize,
    /// Feature normalizer fitted on training data.
    pub normalizer: FeatureNormalizer,
    /// Whether `fit` has been called.
    pub is_fitted: bool,
}

impl RegressionTree {
    /// Create an unfitted regression tree with the given depth limit and
    /// minimum-samples-to-split constraint.
    #[must_use]
    pub fn new(max_depth: usize, min_samples_split: usize) -> Self {
        Self {
            root: None,
            max_depth,
            min_samples_split: min_samples_split.max(2),
            normalizer: FeatureNormalizer::default(),
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

    /// Recursively build a tree node from `samples`.
    ///
    /// `samples[i] = (normalised_features, log1p_runtime)`.
    fn build_node(
        samples: &[(Vec<f64>, f64)],
        depth: usize,
        max_depth: usize,
        min_samples_split: usize,
    ) -> TreeNode {
        let n = samples.len();
        let mean_y = samples.iter().map(|(_, y)| y).sum::<f64>() / n as f64;

        // Base cases: too few samples, depth exceeded, or zero variance
        if n < min_samples_split
            || (max_depth > 0 && depth >= max_depth)
            || variance(samples) < f64::EPSILON
        {
            return TreeNode::Leaf { value: mean_y };
        }

        let mut best_reduction = 0.0f64;
        let mut best_feature = 0usize;
        let mut best_threshold = 0.0f64;
        let parent_var = variance(samples) * n as f64;

        for feat_idx in 0..FEATURE_DIM {
            // Collect sorted unique values for candidate threshold midpoints
            let mut vals: Vec<f64> = samples.iter().map(|(x, _)| x[feat_idx]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Remove near-duplicate values to reduce redundant splits
            vals.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

            for window in vals.windows(2) {
                let threshold = (window[0] + window[1]) / 2.0;
                let left: Vec<_> = samples
                    .iter()
                    .filter(|(x, _)| x[feat_idx] <= threshold)
                    .cloned()
                    .collect();
                let right: Vec<_> = samples
                    .iter()
                    .filter(|(x, _)| x[feat_idx] > threshold)
                    .cloned()
                    .collect();

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let reduction = parent_var
                    - variance(&left) * left.len() as f64
                    - variance(&right) * right.len() as f64;

                if reduction > best_reduction {
                    best_reduction = reduction;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                }
            }
        }

        // No beneficial split found → leaf
        if best_reduction <= 0.0 {
            return TreeNode::Leaf { value: mean_y };
        }

        let left_samples: Vec<_> = samples
            .iter()
            .filter(|(x, _)| x[best_feature] <= best_threshold)
            .cloned()
            .collect();
        let right_samples: Vec<_> = samples
            .iter()
            .filter(|(x, _)| x[best_feature] > best_threshold)
            .cloned()
            .collect();

        TreeNode::Split {
            feature_idx: best_feature,
            threshold: best_threshold,
            left: Box::new(Self::build_node(
                &left_samples,
                depth + 1,
                max_depth,
                min_samples_split,
            )),
            right: Box::new(Self::build_node(
                &right_samples,
                depth + 1,
                max_depth,
                min_samples_split,
            )),
        }
    }

    /// Traverse the tree to predict a value for `features`.
    fn predict_node(node: &TreeNode, features: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf { value } => *value,
            TreeNode::Split {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if features[*feature_idx] <= *threshold {
                    Self::predict_node(left, features)
                } else {
                    Self::predict_node(right, features)
                }
            }
        }
    }

    /// Compute the depth of the tree rooted at `node`.
    #[must_use]
    pub fn tree_depth(node: &TreeNode) -> usize {
        match node {
            TreeNode::Leaf { .. } => 0,
            TreeNode::Split { left, right, .. } => {
                1 + Self::tree_depth(left).max(Self::tree_depth(right))
            }
        }
    }
}

/// Population variance of the `y` values in `samples`.
fn variance(samples: &[(Vec<f64>, f64)]) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let mean = samples.iter().map(|(_, y)| y).sum::<f64>() / n as f64;
    samples
        .iter()
        .map(|(_, y)| {
            let diff = y - mean;
            diff * diff
        })
        .sum::<f64>()
        / n as f64
}

impl DifficultyModel for RegressionTree {
    fn name(&self) -> &'static str {
        "tree"
    }

    fn predict_runtime(&self, features: &Features) -> f64 {
        let root = match &self.root {
            Some(r) => r,
            None => return 0.0,
        };
        let norm = self.normalizer.normalize(features);
        let log_rt = Self::predict_node(root, &norm);
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

        // Build training samples in normalised space
        let training: Vec<(Vec<f64>, f64)> = dataset
            .samples
            .iter()
            .map(|s| {
                let x = self.normalizer.normalize(&s.features);
                let y = s.runtime_seconds.ln_1p();
                (x, y)
            })
            .collect();

        self.root = Some(Self::build_node(
            &training,
            0,
            self.max_depth,
            self.min_samples_split,
        ));

        // Evaluate on training set
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

        self.is_fitted = true;

        TrainingReport {
            final_loss: 0.0,
            epochs: 1,
            mae_seconds: total_ae / nf,
            class_accuracy: correct as f64 / nf,
            k_fold_mean_mae: None,
            time_to_train: start.elapsed(),
        }
    }

    fn to_json(&self) -> String {
        serde_json::to_string(self).expect("serialization infallible for RegressionTree")
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
    fn test_tree_fits_pure_split() {
        // Two perfectly separable groups
        let mut ds = Dataset::new();
        for _ in 0..10 {
            ds.push(make_sample(0.0, 0.05)); // trivial
            ds.push(make_sample(100.0, 30.0)); // hard
        }
        let mut model = RegressionTree::new(4, 2);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);
        assert!(model.is_fitted);
        assert!(model.root.is_some());

        let q_trivial = Features { atom_count: 0.0, ..Default::default() };
        let rt_trivial = model.predict_runtime(&q_trivial);
        assert!(rt_trivial < 1.0, "Expected trivial pred, got {rt_trivial}");

        let q_hard = Features { atom_count: 100.0, ..Default::default() };
        let rt_hard = model.predict_runtime(&q_hard);
        assert!(rt_hard > 1.0, "Expected hard pred, got {rt_hard}");
    }

    #[test]
    fn test_tree_max_depth_respected() {
        let mut ds = Dataset::new();
        for i in 0..30 {
            ds.push(make_sample(i as f64, i as f64 * 0.5));
        }
        let max_d = 3usize;
        let mut model = RegressionTree::new(max_d, 2);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);
        if let Some(ref root) = model.root {
            assert!(
                RegressionTree::tree_depth(root) <= max_d,
                "Tree depth {} exceeds max {}",
                RegressionTree::tree_depth(root),
                max_d,
            );
        }
    }

    #[test]
    fn test_tree_min_samples_split_respected() {
        // Only 3 samples with min_samples_split = 4 → should produce a leaf at root
        let mut ds = Dataset::new();
        ds.push(make_sample(0.0, 0.05));
        ds.push(make_sample(50.0, 5.0));
        ds.push(make_sample(100.0, 30.0));

        let mut model = RegressionTree::new(10, 4);
        let config = TrainingConfig::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        model.fit(&ds, &config, &mut rng);
        assert!(model.is_fitted);
        // With 3 samples and min_split=4, root should be a leaf
        assert!(
            matches!(model.root, Some(TreeNode::Leaf { .. })),
            "Expected leaf at root with n < min_samples_split"
        );
    }
}
