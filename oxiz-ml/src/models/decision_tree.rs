//! Decision Tree Implementation
//!
//! Fast decision trees for classification and regression.
//! Optimized for quick inference (<10μs per prediction).

use super::{Model, ModelError, ModelResult};
use serde::{Deserialize, Serialize};

/// Split criterion for decision trees
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitCriterion {
    /// Gini impurity (for classification)
    Gini,
    /// Information gain / entropy (for classification)
    Entropy,
    /// Mean squared error (for regression)
    MSE,
    /// Mean absolute error (for regression)
    MAE,
}

impl SplitCriterion {
    /// Compute impurity/error for a set of values
    pub fn compute(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        match self {
            SplitCriterion::Gini => {
                // For binary classification: Gini = 1 - sum(p_i^2)
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let p = mean.clamp(0.0, 1.0);
                1.0 - (p * p + (1.0 - p) * (1.0 - p))
            }
            SplitCriterion::Entropy => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let p = mean.clamp(1e-15, 1.0 - 1e-15);
                -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
            }
            SplitCriterion::MSE => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
            }
            SplitCriterion::MAE => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                values.iter().map(|v| (v - mean).abs()).sum::<f64>() / values.len() as f64
            }
        }
    }
}

/// A node in the decision tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionNode {
    /// Internal node with split condition
    Internal {
        /// Feature index to split on
        feature_idx: usize,
        /// Threshold value
        threshold: f64,
        /// Left child (feature <= threshold)
        left: Box<DecisionNode>,
        /// Right child (feature > threshold)
        right: Box<DecisionNode>,
    },
    /// Leaf node with prediction value
    Leaf {
        /// Prediction value
        value: f64,
        /// Number of samples in this leaf
        num_samples: usize,
    },
}

impl Drop for DecisionNode {
    /// Iterative drop.
    ///
    /// `build_tree` is iterative and `TreeConfig::max_depth` is a public field,
    /// so a tree can be arbitrarily unbalanced; the compiler-generated recursive
    /// `drop_in_place` would then abort the process at scope exit, after the
    /// model has already been used, with no diagnostic. Each node is dismantled
    /// into a shallow shell before being released.
    fn drop(&mut self) {
        /// Detach a node's children, leaving a shell that drops trivially.
        fn dismantle(node: &mut DecisionNode, out: &mut Vec<DecisionNode>) {
            /// Replace a boxed child with a leaf and hand the child over.
            fn take(slot: &mut Box<DecisionNode>, out: &mut Vec<DecisionNode>) {
                out.push(std::mem::replace(
                    slot.as_mut(),
                    DecisionNode::Leaf {
                        value: 0.0,
                        num_samples: 0,
                    },
                ));
            }

            match node {
                DecisionNode::Leaf { .. } => {}
                DecisionNode::Internal { left, right, .. } => {
                    take(left, out);
                    take(right, out);
                }
            }
        }

        let mut pending = Vec::new();
        dismantle(self, &mut pending);
        while let Some(mut node) = pending.pop() {
            dismantle(&mut node, &mut pending);
        }
    }
}

impl DecisionNode {
    /// Predict for a single sample.
    ///
    /// Iterative: prediction follows a single root-to-leaf path, and a tree
    /// loaded from an untrusted model file (or trained with a large
    /// `TreeConfig::max_depth`) can be arbitrarily unbalanced. `-> f64` has no
    /// error channel, so a depth cap here could only return a wrong prediction.
    pub fn predict(&self, features: &[f64]) -> f64 {
        let mut node = self;
        loop {
            match node {
                DecisionNode::Internal {
                    feature_idx,
                    threshold,
                    left,
                    right,
                } => {
                    let Some(&feature) = features.get(*feature_idx) else {
                        // Handle dimension mismatch gracefully
                        return 0.0;
                    };

                    node = if feature <= *threshold { left } else { right };
                }
                DecisionNode::Leaf { value, .. } => return *value,
            }
        }
    }

    /// Count total nodes in tree.
    ///
    /// Iterative (explicit heap stack); see [`DecisionNode::predict`].
    pub fn count_nodes(&self) -> usize {
        let mut stack: Vec<&DecisionNode> = vec![self];
        let mut count = 0usize;

        while let Some(node) = stack.pop() {
            count = count.saturating_add(1);
            if let DecisionNode::Internal { left, right, .. } = node {
                stack.push(left);
                stack.push(right);
            }
        }

        count
    }

    /// Get maximum depth of tree.
    ///
    /// Iterative (explicit heap stack): the depth of the deepest leaf, which is
    /// exactly what `1 + max(left, right)` computed recursively.
    pub fn max_depth(&self) -> usize {
        let mut stack: Vec<(&DecisionNode, usize)> = vec![(self, 0)];
        let mut deepest = 0usize;

        while let Some((node, depth)) = stack.pop() {
            match node {
                DecisionNode::Internal { left, right, .. } => {
                    let child_depth = depth.saturating_add(1);
                    stack.push((left, child_depth));
                    stack.push((right, child_depth));
                }
                DecisionNode::Leaf { .. } => deepest = deepest.max(depth),
            }
        }

        deepest
    }
}

/// Tree configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeConfig {
    /// Maximum tree depth
    pub max_depth: usize,
    /// Minimum samples required to split
    pub min_samples_split: usize,
    /// Minimum samples required in a leaf
    pub min_samples_leaf: usize,
    /// Split criterion
    pub criterion: SplitCriterion,
    /// Maximum number of features to consider per split (0 = all features)
    pub max_features: usize,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::MSE,
            max_features: 0, // Use all features
        }
    }
}

/// Decision tree for classification/regression
///
/// # Training model
///
/// A decision tree is inherently a *batch* learner: [`DecisionTree::fit`]
/// rebuilds the whole tree from a full dataset in one shot. It has no true
/// per-sample incremental update rule (unlike a linear model or a neural
/// network, which nudge weights along a gradient).
///
/// To still support the [`Model::train`]/[`Model::train_batch`] online
/// interface honestly, this tree keeps an internal **accumulating training
/// buffer** (`DecisionTree::train_buffer`). Every `train`/`train_batch`
/// call appends its sample(s) to that buffer and then *refits the entire
/// tree from the accumulated buffer* (an incremental-refit policy: the model
/// genuinely reflects every sample seen so far, at O(n) rebuild cost per
/// call). This is correct but grows more expensive as the buffer grows, so
/// for a one-shot large dataset prefer calling [`DecisionTree::fit`] directly;
/// use [`DecisionTree::clear_training_buffer`] to forget accumulated online
/// samples. The buffer is intentionally **not serialized** — only the fitted
/// tree structure is (see the `#[serde(skip)]` below) — so a saved/loaded
/// model reproduces identical predictions without carrying its raw training
/// history around.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    /// Root node of the tree
    root: Option<DecisionNode>,
    /// Configuration
    config: TreeConfig,
    /// Input dimension
    input_dim: usize,
    /// Output dimension (always 1 for decision trees)
    output_dim: usize,
    /// Accumulating buffer of `(features, target)` samples fed through the
    /// online [`Model::train`]/[`Model::train_batch`] interface. Refitting
    /// the tree from this buffer is what makes incremental training real (see
    /// the type-level docs). Not serialized: a persisted model only needs its
    /// fitted structure.
    #[serde(skip, default)]
    train_buffer: Vec<(Vec<f64>, f64)>,
}

impl DecisionTree {
    /// Create a new decision tree
    pub fn new(input_dim: usize, config: TreeConfig) -> Self {
        Self {
            root: None,
            config,
            input_dim,
            output_dim: 1,
            train_buffer: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config(input_dim: usize) -> Self {
        Self::new(input_dim, TreeConfig::default())
    }

    /// Fit the tree to training data
    pub fn fit(&mut self, features: &[Vec<f64>], targets: &[f64]) -> ModelResult<()> {
        if features.is_empty() || targets.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        if features.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: features.len(),
                got: targets.len(),
            });
        }

        // Verify input dimension
        if !features.is_empty() && features[0].len() != self.input_dim {
            return Err(ModelError::DimensionMismatch {
                expected: self.input_dim,
                got: features[0].len(),
            });
        }

        // Build tree recursively
        let indices: Vec<usize> = (0..features.len()).collect();
        self.root = Some(self.build_tree(features, targets, &indices, 0)?);

        Ok(())
    }

    /// Build the tree.
    ///
    /// Iterative (explicit heap stack): `TreeConfig::max_depth` is a public
    /// field, so the only structural bound on depth is the sample count, and a
    /// pathological split sequence produces a chain one sample deep per level.
    fn build_tree(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> ModelResult<DecisionNode> {
        /// Work item for the iterative tree builder.
        enum BuildTask {
            /// Build the subtree for this index set.
            Split {
                /// Sample indices reaching this node.
                indices: Vec<usize>,
                /// Depth of this node below the root.
                depth: usize,
            },
            /// Join the two most recently built subtrees under an internal node.
            Join {
                /// Feature the node splits on.
                feature_idx: usize,
                /// Split threshold.
                threshold: f64,
            },
        }

        let mut tasks = vec![BuildTask::Split {
            indices: indices.to_vec(),
            depth,
        }];
        let mut built: Vec<DecisionNode> = Vec::new();

        while let Some(task) = tasks.pop() {
            let (node_indices, node_depth) = match task {
                BuildTask::Join {
                    feature_idx,
                    threshold,
                } => {
                    let right = built.pop();
                    let left = built.pop();
                    let (Some(left), Some(right)) = (left, right) else {
                        return Err(ModelError::EmptyInput);
                    };
                    built.push(DecisionNode::Internal {
                        feature_idx,
                        threshold,
                        left: Box::new(left),
                        right: Box::new(right),
                    });
                    continue;
                }
                BuildTask::Split { indices, depth } => (indices, depth),
            };

            if node_indices.is_empty() {
                return Err(ModelError::EmptyInput);
            }

            // Extract values for current indices
            let values: Vec<f64> = node_indices.iter().map(|&i| targets[i]).collect();
            let leaf_value = values.iter().sum::<f64>() / values.len() as f64;

            // Stopping criteria
            let should_stop = node_depth >= self.config.max_depth
                || node_indices.len() < self.config.min_samples_split
                || self.is_pure(&values);

            if should_stop {
                // Create leaf with mean value
                built.push(DecisionNode::Leaf {
                    value: leaf_value,
                    num_samples: node_indices.len(),
                });
                continue;
            }

            // Find best split
            let Some((best_feature, best_threshold)) =
                self.find_best_split(features, targets, &node_indices)
            else {
                // No valid split found, create leaf
                built.push(DecisionNode::Leaf {
                    value: leaf_value,
                    num_samples: node_indices.len(),
                });
                continue;
            };

            // Split data
            let (left_indices, right_indices) =
                self.split_data(features, &node_indices, best_feature, best_threshold);

            // Check minimum leaf size
            if left_indices.len() < self.config.min_samples_leaf
                || right_indices.len() < self.config.min_samples_leaf
            {
                built.push(DecisionNode::Leaf {
                    value: leaf_value,
                    num_samples: node_indices.len(),
                });
                continue;
            }

            // Build left and right subtrees, then join them (left completes first).
            tasks.push(BuildTask::Join {
                feature_idx: best_feature,
                threshold: best_threshold,
            });
            tasks.push(BuildTask::Split {
                indices: right_indices,
                depth: node_depth + 1,
            });
            tasks.push(BuildTask::Split {
                indices: left_indices,
                depth: node_depth + 1,
            });
        }

        built.pop().ok_or(ModelError::EmptyInput)
    }

    /// Check if all values are the same (pure node)
    fn is_pure(&self, values: &[f64]) -> bool {
        if values.is_empty() {
            return true;
        }
        let first = values[0];
        values.iter().all(|&v| (v - first).abs() < 1e-10)
    }

    /// Find best split
    fn find_best_split(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        indices: &[usize],
    ) -> Option<(usize, f64)> {
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        // Current impurity
        let values: Vec<f64> = indices.iter().map(|&i| targets[i]).collect();
        let current_impurity = self.config.criterion.compute(&values);

        // Determine which features to consider
        let num_features = if self.config.max_features > 0 {
            self.config.max_features.min(self.input_dim)
        } else {
            self.input_dim
        };

        // Try each feature
        for feature_idx in 0..num_features {
            // Get unique thresholds (midpoints between consecutive values)
            let mut feature_values: Vec<f64> =
                indices.iter().map(|&i| features[i][feature_idx]).collect();
            feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            feature_values.dedup();

            // Try each threshold
            for i in 0..feature_values.len().saturating_sub(1) {
                let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

                // Split data
                let (left_indices, right_indices) =
                    self.split_data(features, indices, feature_idx, threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Compute impurity of split
                let left_values: Vec<f64> = left_indices.iter().map(|&i| targets[i]).collect();
                let right_values: Vec<f64> = right_indices.iter().map(|&i| targets[i]).collect();

                let left_impurity = self.config.criterion.compute(&left_values);
                let right_impurity = self.config.criterion.compute(&right_values);

                // Weighted impurity
                let n = indices.len() as f64;
                let n_left = left_indices.len() as f64;
                let n_right = right_indices.len() as f64;

                let weighted_impurity =
                    (n_left / n) * left_impurity + (n_right / n) * right_impurity;
                let gain = current_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        if best_gain > 0.0 {
            Some((best_feature, best_threshold))
        } else {
            None
        }
    }

    /// Split data based on feature and threshold
    fn split_data(
        &self,
        features: &[Vec<f64>],
        indices: &[usize],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left = Vec::new();
        let mut right = Vec::new();

        for &idx in indices {
            if features[idx][feature_idx] <= threshold {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }

        (left, right)
    }

    /// Get tree structure information
    pub fn info(&self) -> TreeInfo {
        if let Some(ref root) = self.root {
            TreeInfo {
                num_nodes: root.count_nodes(),
                max_depth: root.max_depth(),
                num_leaves: self.count_leaves(root),
            }
        } else {
            TreeInfo {
                num_nodes: 0,
                max_depth: 0,
                num_leaves: 0,
            }
        }
    }

    /// Count number of leaf nodes.
    ///
    /// Iterative (explicit heap stack); see [`DecisionNode::predict`].
    fn count_leaves(&self, node: &DecisionNode) -> usize {
        let mut stack: Vec<&DecisionNode> = vec![node];
        let mut leaves = 0usize;

        while let Some(current) = stack.pop() {
            match current {
                DecisionNode::Internal { left, right, .. } => {
                    stack.push(left);
                    stack.push(right);
                }
                DecisionNode::Leaf { .. } => leaves = leaves.saturating_add(1),
            }
        }

        leaves
    }

    /// Number of samples currently held in the online training buffer.
    ///
    /// These are the samples accumulated through [`Model::train`] /
    /// [`Model::train_batch`] since the last [`DecisionTree::clear_training_buffer`]
    /// (or construction). The buffer is what the tree refits from on each
    /// online update; see the type-level documentation.
    pub fn training_buffer_len(&self) -> usize {
        self.train_buffer.len()
    }

    /// Discard every sample accumulated through the online training interface.
    ///
    /// The already-fitted tree is left untouched — this only forgets the raw
    /// history that future online [`Model::train`] calls would otherwise keep
    /// refitting from, so it is the way to bound the incremental-refit cost
    /// once a model has stabilised.
    pub fn clear_training_buffer(&mut self) {
        self.train_buffer.clear();
    }

    /// Refit the tree from the whole accumulated online training buffer and
    /// return the resulting mean-squared training loss.
    ///
    /// This is the shared core of the online [`Model::train`] /
    /// [`Model::train_batch`] path: every sample seen so far participates, so
    /// the refitted tree is not clobbered by the most recent sample (the bug
    /// the previous single-sample `fit` had) but genuinely reflects the full
    /// history.
    fn refit_from_buffer(&mut self) -> ModelResult<f64> {
        if self.train_buffer.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        let features: Vec<Vec<f64>> = self.train_buffer.iter().map(|(f, _)| f.clone()).collect();
        let targets: Vec<f64> = self.train_buffer.iter().map(|(_, t)| *t).collect();

        self.fit(&features, &targets)?;
        Ok(self.mean_squared_loss(&features, &targets))
    }

    /// Mean squared error of the current tree over `(features, targets)`.
    ///
    /// A real, non-fabricated training metric (the previous `train` always
    /// returned a hardcoded `0.0`).
    fn mean_squared_loss(&self, features: &[Vec<f64>], targets: &[f64]) -> f64 {
        if features.is_empty() {
            return 0.0;
        }
        let mut sum_sq = 0.0;
        for (feature, &target) in features.iter().zip(targets.iter()) {
            let predicted = self.predict(feature).first().copied().unwrap_or(0.0);
            let diff = predicted - target;
            sum_sq += diff * diff;
        }
        sum_sq / features.len() as f64
    }
}

/// Tree information
#[derive(Debug, Clone)]
pub struct TreeInfo {
    /// Total number of nodes
    pub num_nodes: usize,
    /// Maximum depth
    pub max_depth: usize,
    /// Number of leaf nodes
    pub num_leaves: usize,
}

impl Model for DecisionTree {
    fn input_dim(&self) -> usize {
        self.input_dim
    }

    fn output_dim(&self) -> usize {
        self.output_dim
    }

    fn predict(&self, input: &[f64]) -> Vec<f64> {
        if let Some(ref root) = self.root {
            vec![root.predict(input)]
        } else {
            vec![0.0]
        }
    }

    fn train(&mut self, input: &[f64], target: &[f64]) -> ModelResult<f64> {
        // A decision tree has no per-sample incremental update rule, so
        // "online" training here means: remember this sample and refit the
        // whole tree from every sample accumulated so far (see the
        // type-level docs on the incremental-refit policy). This fixes the
        // previous behaviour, which refit the tree on *only* the current
        // sample — discarding all prior structure — and always reported a
        // hardcoded loss of 0.0.
        if target.is_empty() {
            return Err(ModelError::EmptyInput);
        }
        if input.len() != self.input_dim {
            return Err(ModelError::DimensionMismatch {
                expected: self.input_dim,
                got: input.len(),
            });
        }
        self.train_buffer.push((input.to_vec(), target[0]));
        self.refit_from_buffer()
    }

    fn train_batch(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> ModelResult<f64> {
        // Batch training is the decision tree's native mode: accumulate every
        // sample and refit the whole tree once, rather than looping the
        // single-sample `train` (which would pointlessly rebuild the tree
        // once per sample). The returned loss is the real mean-squared error
        // over the accumulated buffer.
        if inputs.len() != targets.len() {
            return Err(ModelError::DimensionMismatch {
                expected: inputs.len(),
                got: targets.len(),
            });
        }
        if inputs.is_empty() {
            return Err(ModelError::EmptyInput);
        }
        for (input, target) in inputs.iter().zip(targets.iter()) {
            if target.is_empty() {
                return Err(ModelError::EmptyInput);
            }
            if input.len() != self.input_dim {
                return Err(ModelError::DimensionMismatch {
                    expected: self.input_dim,
                    got: input.len(),
                });
            }
            self.train_buffer.push((input.clone(), target[0]));
        }
        self.refit_from_buffer()
    }

    fn num_parameters(&self) -> usize {
        // Decision trees don't have traditional parameters
        // Return number of nodes as a proxy
        if let Some(ref root) = self.root {
            root.count_nodes()
        } else {
            0
        }
    }

    fn save(&self) -> ModelResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| ModelError::SerializationError(e.to_string()))
    }

    fn load(&mut self, data: &[u8]) -> ModelResult<()> {
        let loaded: DecisionTree = serde_json::from_slice(data)
            .map_err(|e| ModelError::SerializationError(e.to_string()))?;

        self.root = loaded.root;
        self.config = loaded.config;
        self.input_dim = loaded.input_dim;
        self.output_dim = loaded.output_dim;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_criterion_gini() {
        let criterion = SplitCriterion::Gini;
        let pure = vec![1.0, 1.0, 1.0];
        assert_eq!(criterion.compute(&pure), 0.0);

        let mixed = vec![0.0, 1.0];
        assert!(criterion.compute(&mixed) > 0.0);
    }

    #[test]
    fn test_split_criterion_mse() {
        let criterion = SplitCriterion::MSE;
        let uniform = vec![2.0, 2.0, 2.0];
        assert_eq!(criterion.compute(&uniform), 0.0);

        let varied = vec![1.0, 2.0, 3.0];
        assert!(criterion.compute(&varied) > 0.0);
    }

    #[test]
    fn test_decision_node_predict() {
        let leaf = DecisionNode::Leaf {
            value: 5.0,
            num_samples: 10,
        };
        assert_eq!(leaf.predict(&[1.0, 2.0]), 5.0);
    }

    #[test]
    fn test_decision_tree_creation() {
        let tree = DecisionTree::default_config(5);
        assert_eq!(tree.input_dim(), 5);
        assert_eq!(tree.output_dim(), 1);
    }

    #[test]
    fn test_decision_tree_fit_simple() {
        let mut tree = DecisionTree::default_config(1);

        let features = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        tree.fit(&features, &targets)
            .expect("test operation should succeed");

        assert!(tree.root.is_some());
    }

    #[test]
    fn test_decision_tree_predict() {
        let mut tree = DecisionTree::default_config(1);

        let features = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![10.0, 20.0, 30.0];

        tree.fit(&features, &targets)
            .expect("test operation should succeed");

        let pred = tree.predict(&[1.5]);
        assert!(pred[0] >= 10.0 && pred[0] <= 20.0);
    }

    #[test]
    fn test_decision_tree_info() {
        let mut tree = DecisionTree::default_config(2);

        let features = vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![2.0, 2.0],
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        tree.fit(&features, &targets)
            .expect("test operation should succeed");

        let info = tree.info();
        assert!(info.num_nodes > 0);
        assert!(info.num_leaves > 0);
    }

    /// A decision tree must reach high *training* accuracy on a cleanly
    /// separable synthetic dataset when driven through the online
    /// `train_batch` interface — the previous `train` impl refit on only the
    /// last sample, so it could never learn a multi-sample rule.
    #[test]
    fn test_decision_tree_batch_train_reaches_high_accuracy() {
        let tree_config = TreeConfig {
            max_depth: 6,
            min_samples_split: 2,
            min_samples_leaf: 1,
            criterion: SplitCriterion::Gini,
            max_features: 0,
        };
        let mut tree = DecisionTree::new(2, tree_config);

        // 20-sample linearly separable set: label 1.0 iff x0 + x1 > 10.
        let mut inputs: Vec<Vec<f64>> = Vec::new();
        let mut targets: Vec<Vec<f64>> = Vec::new();
        for i in 0..20 {
            let x0 = (i % 5) as f64 * 2.0; // 0,2,4,6,8
            let x1 = (i / 5) as f64 * 3.0; // 0,3,6,9
            let label = if x0 + x1 > 10.0 { 1.0 } else { 0.0 };
            inputs.push(vec![x0, x1]);
            targets.push(vec![label]);
        }

        let loss = tree
            .train_batch(&inputs, &targets)
            .expect("batch training should succeed");
        assert!(loss.is_finite(), "loss must be a real number, got {loss}");

        let mut correct = 0usize;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let predicted = tree.predict(input)[0];
            // Threshold the regression output at 0.5 to recover the class.
            let predicted_label = if predicted >= 0.5 { 1.0 } else { 0.0 };
            if (predicted_label - target[0]).abs() < 1e-9 {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / inputs.len() as f64;
        assert!(
            accuracy > 0.9,
            "expected >90% train accuracy on separable data, got {accuracy}"
        );
        assert_eq!(tree.training_buffer_len(), 20);
    }

    /// Online `train` must *accumulate* samples and refit from all of them,
    /// not clobber the tree with only the most recent sample.
    #[test]
    fn test_decision_tree_online_train_accumulates_samples() {
        let mut tree = DecisionTree::default_config(1);

        // Two clearly different (x -> y) samples. If `train` kept only the
        // last one (the old bug), the tree would be a single leaf predicting
        // the last target for every input.
        tree.train(&[0.0], &[0.0]).expect("train sample 1");
        tree.train(&[10.0], &[100.0]).expect("train sample 2");
        tree.train(&[0.0], &[0.0]).expect("train sample 3");
        tree.train(&[10.0], &[100.0]).expect("train sample 4");

        assert_eq!(tree.training_buffer_len(), 4);

        // A tree that only remembered the last sample would predict ~100 here.
        let low = tree.predict(&[0.0])[0];
        let high = tree.predict(&[10.0])[0];
        assert!(
            low < 50.0 && high > 50.0,
            "tree should separate the two accumulated classes: low={low}, high={high}"
        );

        tree.clear_training_buffer();
        assert_eq!(tree.training_buffer_len(), 0);
    }

    /// The reported training loss must be a genuine metric, not a hardcoded
    /// `0.0`: a dataset with two contradictory targets for the same input
    /// cannot be fit exactly, so the loss is strictly positive.
    #[test]
    fn test_decision_tree_train_loss_is_real() {
        let mut tree = DecisionTree::default_config(1);
        let inputs = vec![vec![1.0], vec![1.0]];
        let targets = vec![vec![0.0], vec![10.0]];
        let loss = tree
            .train_batch(&inputs, &targets)
            .expect("training should succeed");
        assert!(
            loss > 0.0,
            "contradictory targets must yield a positive loss, got {loss}"
        );
    }

    #[test]
    fn test_decision_tree_save_load() {
        let mut tree = DecisionTree::default_config(2);

        let features = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let targets = vec![1.0, 2.0];

        tree.fit(&features, &targets)
            .expect("test operation should succeed");

        let saved = tree.save().expect("test operation should succeed");

        let mut tree2 = DecisionTree::default_config(2);
        tree2.load(&saved).expect("test operation should succeed");

        let pred1 = tree.predict(&[1.5, 1.5]);
        let pred2 = tree2.predict(&[1.5, 1.5]);

        assert_eq!(pred1, pred2);
    }

    /// Stack size for the deep-tree regression tests. A stack overflow aborts
    /// the process rather than failing a test, so *returning at all* is the
    /// assertion; the small stack makes a surviving recursion detectable.
    const DEEP_TREE_STACK: usize = 1 << 17;

    /// Build a right-leaning chain of `depth` internal nodes.
    fn deep_chain(depth: usize) -> DecisionNode {
        let mut node = DecisionNode::Leaf {
            value: 42.0,
            num_samples: 1,
        };
        for _ in 0..depth {
            node = DecisionNode::Internal {
                feature_idx: 0,
                threshold: 0.5,
                left: Box::new(DecisionNode::Leaf {
                    value: -1.0,
                    num_samples: 1,
                }),
                right: Box::new(node),
            };
        }
        node
    }

    #[test]
    fn deep_tree_walks_and_drop_survive_a_small_stack() {
        std::thread::Builder::new()
            .name("ml_deep_tree".to_string())
            .stack_size(DEEP_TREE_STACK)
            .spawn(|| {
                let depth = 12_500;
                let tree = deep_chain(depth);

                // Semantic pins: the same values the recursive versions produced.
                assert_eq!(tree.count_nodes(), 2 * depth + 1);
                assert_eq!(tree.max_depth(), depth);
                // features[0] > threshold at every level, so we walk to the
                // bottom-right leaf.
                assert!((tree.predict(&[1.0]) - 42.0).abs() < f64::EPSILON);
                // ... and left at the first level.
                assert!((tree.predict(&[0.0]) + 1.0).abs() < f64::EPSILON);
                // Dimension mismatch is still reported as 0.0.
                assert!(tree.predict(&[]).abs() < f64::EPSILON);

                // `tree` is dropped here: iterative Drop, not 100k drop frames.
            })
            .expect("test thread should spawn")
            .join()
            .expect("test thread should not panic");
    }

    #[test]
    fn shallow_tree_walk_values_are_unchanged() {
        let tree = DecisionNode::Internal {
            feature_idx: 1,
            threshold: 2.0,
            left: Box::new(DecisionNode::Leaf {
                value: 10.0,
                num_samples: 3,
            }),
            right: Box::new(DecisionNode::Internal {
                feature_idx: 0,
                threshold: 1.0,
                left: Box::new(DecisionNode::Leaf {
                    value: 20.0,
                    num_samples: 2,
                }),
                right: Box::new(DecisionNode::Leaf {
                    value: 30.0,
                    num_samples: 1,
                }),
            }),
        };
        assert_eq!(tree.count_nodes(), 5);
        assert_eq!(tree.max_depth(), 2);
        assert!((tree.predict(&[0.0, 1.0]) - 10.0).abs() < f64::EPSILON);
        assert!((tree.predict(&[0.5, 3.0]) - 20.0).abs() < f64::EPSILON);
        assert!((tree.predict(&[5.0, 3.0]) - 30.0).abs() < f64::EPSILON);
    }
}
