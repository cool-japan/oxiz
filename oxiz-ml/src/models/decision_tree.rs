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

impl DecisionNode {
    /// Predict for a single sample
    pub fn predict(&self, features: &[f64]) -> f64 {
        match self {
            DecisionNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if *feature_idx >= features.len() {
                    // Handle dimension mismatch gracefully
                    return 0.0;
                }

                if features[*feature_idx] <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
            DecisionNode::Leaf { value, .. } => *value,
        }
    }

    /// Count total nodes in tree
    pub fn count_nodes(&self) -> usize {
        match self {
            DecisionNode::Internal { left, right, .. } => {
                1 + left.count_nodes() + right.count_nodes()
            }
            DecisionNode::Leaf { .. } => 1,
        }
    }

    /// Get maximum depth of tree
    pub fn max_depth(&self) -> usize {
        match self {
            DecisionNode::Internal { left, right, .. } => {
                1 + left.max_depth().max(right.max_depth())
            }
            DecisionNode::Leaf { .. } => 0,
        }
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
}

impl DecisionTree {
    /// Create a new decision tree
    pub fn new(input_dim: usize, config: TreeConfig) -> Self {
        Self {
            root: None,
            config,
            input_dim,
            output_dim: 1,
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

    /// Recursively build tree
    fn build_tree(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        indices: &[usize],
        depth: usize,
    ) -> ModelResult<DecisionNode> {
        if indices.is_empty() {
            return Err(ModelError::EmptyInput);
        }

        // Extract values for current indices
        let values: Vec<f64> = indices.iter().map(|&i| targets[i]).collect();

        // Stopping criteria
        let should_stop = depth >= self.config.max_depth
            || indices.len() < self.config.min_samples_split
            || self.is_pure(&values);

        if should_stop {
            // Create leaf with mean value
            let value = values.iter().sum::<f64>() / values.len() as f64;
            return Ok(DecisionNode::Leaf {
                value,
                num_samples: indices.len(),
            });
        }

        // Find best split
        if let Some((best_feature, best_threshold)) =
            self.find_best_split(features, targets, indices)
        {
            // Split data
            let (left_indices, right_indices) =
                self.split_data(features, indices, best_feature, best_threshold);

            // Check minimum leaf size
            if left_indices.len() < self.config.min_samples_leaf
                || right_indices.len() < self.config.min_samples_leaf
            {
                let value = values.iter().sum::<f64>() / values.len() as f64;
                return Ok(DecisionNode::Leaf {
                    value,
                    num_samples: indices.len(),
                });
            }

            // Recursively build left and right subtrees
            let left = Box::new(self.build_tree(features, targets, &left_indices, depth + 1)?);
            let right = Box::new(self.build_tree(features, targets, &right_indices, depth + 1)?);

            Ok(DecisionNode::Internal {
                feature_idx: best_feature,
                threshold: best_threshold,
                left,
                right,
            })
        } else {
            // No valid split found, create leaf
            let value = values.iter().sum::<f64>() / values.len() as f64;
            Ok(DecisionNode::Leaf {
                value,
                num_samples: indices.len(),
            })
        }
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

    /// Count number of leaf nodes
    fn count_leaves(&self, node: &DecisionNode) -> usize {
        match node {
            DecisionNode::Internal { left, right, .. } => {
                self.count_leaves(left) + self.count_leaves(right)
            }
            DecisionNode::Leaf { .. } => 1,
        }
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
        // For online learning, we'd need to rebuild the tree or use incremental methods
        // For now, just return 0 (decision trees are typically trained in batch)
        self.fit(&[input.to_vec()], target)?;
        Ok(0.0)
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

        tree.fit(&features, &targets).unwrap();

        assert!(tree.root.is_some());
    }

    #[test]
    fn test_decision_tree_predict() {
        let mut tree = DecisionTree::default_config(1);

        let features = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets = vec![10.0, 20.0, 30.0];

        tree.fit(&features, &targets).unwrap();

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

        tree.fit(&features, &targets).unwrap();

        let info = tree.info();
        assert!(info.num_nodes > 0);
        assert!(info.num_leaves > 0);
    }

    #[test]
    fn test_decision_tree_save_load() {
        let mut tree = DecisionTree::default_config(2);

        let features = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let targets = vec![1.0, 2.0];

        tree.fit(&features, &targets).unwrap();

        let saved = tree.save().unwrap();

        let mut tree2 = DecisionTree::default_config(2);
        tree2.load(&saved).unwrap();

        let pred1 = tree.predict(&[1.5, 1.5]);
        let pred2 = tree2.predict(&[1.5, 1.5]);

        assert_eq!(pred1, pred2);
    }
}
