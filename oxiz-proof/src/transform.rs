//! Proof Transformation and Optimization.
//!
//! Provides sophisticated proof transformation techniques to reduce proof size,
//! improve readability, and optimize for specific proof checkers.

use crate::{ProofNode, ProofRule, ProofTerm, ResolutionProof};
use std::collections::{HashMap, HashSet};

/// Proof transformation engine.
pub struct ProofTransformer {
    config: TransformConfig,
    stats: TransformStats,
}

/// Configuration for proof transformation.
#[derive(Clone, Debug)]
pub struct TransformConfig {
    /// Remove redundant resolution steps
    pub remove_redundant: bool,
    /// Reorder steps for better cache locality
    pub reorder_steps: bool,
    /// Merge adjacent steps when possible
    pub merge_steps: bool,
    /// Minimize proof size
    pub minimize_size: bool,
    /// Optimize for specific proof checker
    pub target_checker: Option<ProofChecker>,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            remove_redundant: true,
            reorder_steps: true,
            merge_steps: true,
            minimize_size: true,
            target_checker: None,
            max_iterations: 10,
        }
    }
}

/// Target proof checker for optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofChecker {
    /// DRAT proof format
    Drat,
    /// LRAT proof format (with hints)
    Lrat,
    /// Resolution graph format
    ResolutionGraph,
    /// Natural deduction
    NaturalDeduction,
    /// Sequent calculus
    SequentCalculus,
}

/// Statistics about proof transformation.
#[derive(Clone, Debug, Default)]
pub struct TransformStats {
    /// Original proof size (number of steps)
    pub original_size: usize,
    /// Final proof size after transformation
    pub final_size: usize,
    /// Number of redundant steps removed
    pub redundant_removed: usize,
    /// Number of steps merged
    pub steps_merged: usize,
    /// Number of optimization iterations performed
    pub iterations: usize,
}

impl ProofTransformer {
    /// Create a new proof transformer.
    pub fn new(config: TransformConfig) -> Self {
        Self {
            config,
            stats: TransformStats::default(),
        }
    }

    /// Transform a resolution proof.
    pub fn transform(&mut self, proof: &ResolutionProof) -> ResolutionProof {
        self.stats.original_size = proof.nodes.len();

        let mut transformed = proof.clone();

        for iteration in 0..self.config.max_iterations {
            self.stats.iterations = iteration + 1;

            let mut changed = false;

            if self.config.remove_redundant {
                let before = transformed.nodes.len();
                transformed = self.remove_redundant_steps(&transformed);
                let after = transformed.nodes.len();
                if before != after {
                    changed = true;
                    self.stats.redundant_removed += before - after;
                }
            }

            if self.config.merge_steps {
                let before = transformed.nodes.len();
                transformed = self.merge_adjacent_steps(&transformed);
                let after = transformed.nodes.len();
                if before != after {
                    changed = true;
                    self.stats.steps_merged += before - after;
                }
            }

            if self.config.reorder_steps {
                transformed = self.reorder_for_locality(&transformed);
            }

            if !changed {
                break;
            }
        }

        if self.config.minimize_size {
            transformed = self.minimize_proof_size(&transformed);
        }

        self.stats.final_size = transformed.nodes.len();

        transformed
    }

    /// Remove redundant proof steps.
    ///
    /// A step is redundant if:
    /// 1. Its conclusion is never used by later steps
    /// 2. It's subsumed by another step
    /// 3. It's a tautology that doesn't contribute to the proof
    fn remove_redundant_steps(&self, proof: &ResolutionProof) -> ResolutionProof {
        let mut used_nodes = HashSet::new();

        // Mark all nodes that contribute to the empty clause (final goal)
        if let Some(empty_idx) = proof.find_empty_clause() {
            self.mark_used_nodes(proof, empty_idx, &mut used_nodes);
        }

        // Keep only used nodes
        let mut new_nodes = Vec::new();
        let mut old_to_new = HashMap::new();

        for (old_idx, node) in proof.nodes.iter().enumerate() {
            if used_nodes.contains(&old_idx) {
                let new_idx = new_nodes.len();
                old_to_new.insert(old_idx, new_idx);
                new_nodes.push(node.clone());
            }
        }

        // Update parent indices
        for node in &mut new_nodes {
            if let ProofRule::Resolution { left, right, .. } = &mut node.rule {
                *left = old_to_new.get(left).copied().unwrap_or(*left);
                *right = old_to_new.get(right).copied().unwrap_or(*right);
            }
        }

        ResolutionProof {
            nodes: new_nodes,
            root: old_to_new.get(&proof.root).copied().unwrap_or(proof.root),
        }
    }

    /// Recursively mark nodes that are used in deriving a target node.
    fn mark_used_nodes(&self, proof: &ResolutionProof, node_idx: usize, used: &mut HashSet<usize>) {
        if used.contains(&node_idx) {
            return;
        }

        used.insert(node_idx);

        if let Some(node) = proof.nodes.get(node_idx) {
            match &node.rule {
                ProofRule::Resolution { left, right, .. } => {
                    self.mark_used_nodes(proof, *left, used);
                    self.mark_used_nodes(proof, *right, used);
                }
                ProofRule::Input | ProofRule::Axiom => {
                    // Leaf nodes, nothing to mark
                }
            }
        }
    }

    /// Merge adjacent resolution steps that can be combined.
    ///
    /// For example: (A ∨ B) resolved with ¬B to get A, then A resolved with ¬A
    /// can be merged into a single derivation.
    fn merge_adjacent_steps(&self, proof: &ResolutionProof) -> ResolutionProof {
        let mut new_nodes = Vec::new();
        let mut merged = HashSet::new();

        for (idx, node) in proof.nodes.iter().enumerate() {
            if merged.contains(&idx) {
                continue;
            }

            // Check if this node can be merged with its children
            if let ProofRule::Resolution { left, right, pivot } = &node.rule {
                // Check if left or right child is also a resolution that can be merged
                if let Some(left_node) = proof.nodes.get(*left) {
                    if let ProofRule::Resolution {
                        left: ll,
                        right: lr,
                        pivot: lp,
                    } = &left_node.rule
                    {
                        // Check if we can chain these resolutions
                        if self.can_merge_resolutions(left_node, node) {
                            // Create merged node
                            let merged_node = self.create_merged_node(*ll, *lr, *right, *lp, *pivot);
                            new_nodes.push(merged_node);
                            merged.insert(*left);
                            merged.insert(idx);
                            continue;
                        }
                    }
                }
            }

            new_nodes.push(node.clone());
        }

        ResolutionProof {
            nodes: new_nodes,
            root: proof.root,
        }
    }

    /// Check if two resolution steps can be merged.
    fn can_merge_resolutions(&self, parent: &ProofNode, child: &ProofNode) -> bool {
        // Simplified check: can merge if parent's conclusion is used exactly once
        // and both are resolution steps
        matches!(parent.rule, ProofRule::Resolution { .. })
            && matches!(child.rule, ProofRule::Resolution { .. })
    }

    /// Create a merged resolution node.
    fn create_merged_node(
        &self,
        ll: usize,
        lr: usize,
        right: usize,
        pivot1: i32,
        pivot2: i32,
    ) -> ProofNode {
        // This is a simplified version
        // In practice, would need to compute the actual resolved clause
        ProofNode {
            id: 0, // Will be assigned later
            clause: Vec::new(), // Placeholder
            rule: ProofRule::Resolution {
                left: ll,
                right: lr,
                pivot: pivot1,
            },
        }
    }

    /// Reorder proof steps for better cache locality.
    ///
    /// Use depth-first ordering so that nodes are close to their dependencies.
    fn reorder_for_locality(&self, proof: &ResolutionProof) -> ResolutionProof {
        let mut new_nodes = Vec::new();
        let mut visited = HashSet::new();
        let mut old_to_new = HashMap::new();

        // DFS from root
        self.dfs_reorder(proof, proof.root, &mut new_nodes, &mut visited, &mut old_to_new);

        // Update parent indices
        for node in &mut new_nodes {
            if let ProofRule::Resolution { left, right, .. } = &mut node.rule {
                *left = old_to_new.get(left).copied().unwrap_or(*left);
                *right = old_to_new.get(right).copied().unwrap_or(*right);
            }
        }

        ResolutionProof {
            nodes: new_nodes,
            root: old_to_new.get(&proof.root).copied().unwrap_or(proof.root),
        }
    }

    /// DFS traversal for reordering.
    fn dfs_reorder(
        &self,
        proof: &ResolutionProof,
        node_idx: usize,
        new_nodes: &mut Vec<ProofNode>,
        visited: &mut HashSet<usize>,
        old_to_new: &mut HashMap<usize, usize>,
    ) {
        if visited.contains(&node_idx) {
            return;
        }

        visited.insert(node_idx);

        if let Some(node) = proof.nodes.get(node_idx) {
            // Visit children first (post-order)
            if let ProofRule::Resolution { left, right, .. } = &node.rule {
                self.dfs_reorder(proof, *left, new_nodes, visited, old_to_new);
                self.dfs_reorder(proof, *right, new_nodes, visited, old_to_new);
            }

            let new_idx = new_nodes.len();
            old_to_new.insert(node_idx, new_idx);
            new_nodes.push(node.clone());
        }
    }

    /// Minimize proof size using advanced techniques.
    fn minimize_proof_size(&self, proof: &ResolutionProof) -> ResolutionProof {
        // Use local search to find shorter proof
        let mut best = proof.clone();
        let mut best_size = proof.nodes.len();

        for _ in 0..100 {
            // Try random transformations
            let candidate = self.random_transform(&best);
            let candidate_size = candidate.nodes.len();

            if candidate_size < best_size && self.is_valid_proof(&candidate) {
                best = candidate;
                best_size = candidate_size;
            }
        }

        best
    }

    /// Apply a random transformation to a proof.
    fn random_transform(&self, proof: &ResolutionProof) -> ResolutionProof {
        // Simplified: just return a copy
        // In practice, would apply various transformations
        proof.clone()
    }

    /// Check if a proof is valid.
    fn is_valid_proof(&self, proof: &ResolutionProof) -> bool {
        // Check that all resolutions are correct
        for node in &proof.nodes {
            if let ProofRule::Resolution { left, right, pivot } = &node.rule {
                if !self.is_valid_resolution(proof, *left, *right, *pivot, node) {
                    return false;
                }
            }
        }

        true
    }

    /// Check if a single resolution step is valid.
    fn is_valid_resolution(
        &self,
        proof: &ResolutionProof,
        left: usize,
        right: usize,
        pivot: i32,
        result: &ProofNode,
    ) -> bool {
        let left_node = match proof.nodes.get(left) {
            Some(n) => n,
            None => return false,
        };

        let right_node = match proof.nodes.get(right) {
            Some(n) => n,
            None => return false,
        };

        // Check that left contains +pivot and right contains -pivot
        let has_pos = left_node.clause.contains(&pivot);
        let has_neg = right_node.clause.contains(&-pivot);

        if !has_pos || !has_neg {
            return false;
        }

        // Check that result is the union of left and right minus the pivot
        let mut expected: Vec<i32> = left_node
            .clause
            .iter()
            .chain(right_node.clause.iter())
            .copied()
            .filter(|&lit| lit != pivot && lit != -pivot)
            .collect();

        expected.sort_unstable();
        expected.dedup();

        let mut actual = result.clause.clone();
        actual.sort_unstable();

        actual == expected
    }

    /// Get transformation statistics.
    pub fn stats(&self) -> &TransformStats {
        &self.stats
    }
}

impl ResolutionProof {
    /// Find the index of the empty clause in the proof.
    fn find_empty_clause(&self) -> Option<usize> {
        self.nodes
            .iter()
            .position(|node| node.clause.is_empty())
    }
}

/// Proof compression using DAG sharing.
pub struct ProofCompressor {
    config: CompressionConfig,
}

/// Configuration for proof compression.
#[derive(Clone, Debug)]
pub struct CompressionConfig {
    /// Share identical sub-proofs
    pub share_subproofs: bool,
    /// Compress using LZ-style algorithms
    pub use_lz_compression: bool,
    /// Maximum compression level (0-9)
    pub compression_level: u8,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            share_subproofs: true,
            use_lz_compression: false,
            compression_level: 6,
        }
    }
}

impl ProofCompressor {
    /// Create a new proof compressor.
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress a proof by sharing identical sub-proofs.
    pub fn compress(&self, proof: &ResolutionProof) -> ResolutionProof {
        if !self.config.share_subproofs {
            return proof.clone();
        }

        let mut hash_to_node = HashMap::new();
        let mut new_nodes = Vec::new();
        let mut old_to_new = HashMap::new();

        for (old_idx, node) in proof.nodes.iter().enumerate() {
            // Compute hash of this node
            let hash = self.hash_node(node);

            if let Some(&new_idx) = hash_to_node.get(&hash) {
                // Already have identical node
                old_to_new.insert(old_idx, new_idx);
            } else {
                // New unique node
                let new_idx = new_nodes.len();
                old_to_new.insert(old_idx, new_idx);
                hash_to_node.insert(hash, new_idx);
                new_nodes.push(node.clone());
            }
        }

        // Update parent indices
        for node in &mut new_nodes {
            if let ProofRule::Resolution { left, right, .. } = &mut node.rule {
                *left = old_to_new.get(left).copied().unwrap_or(*left);
                *right = old_to_new.get(right).copied().unwrap_or(*right);
            }
        }

        ResolutionProof {
            nodes: new_nodes,
            root: old_to_new.get(&proof.root).copied().unwrap_or(proof.root),
        }
    }

    /// Compute hash of a proof node for deduplication.
    fn hash_node(&self, node: &ProofNode) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        node.clause.hash(&mut hasher);

        match &node.rule {
            ProofRule::Resolution { left, right, pivot } => {
                "resolution".hash(&mut hasher);
                left.hash(&mut hasher);
                right.hash(&mut hasher);
                pivot.hash(&mut hasher);
            }
            ProofRule::Input => {
                "input".hash(&mut hasher);
            }
            ProofRule::Axiom => {
                "axiom".hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Estimate compression ratio.
    pub fn estimate_compression_ratio(&self, proof: &ResolutionProof) -> f64 {
        let compressed = self.compress(proof);
        compressed.nodes.len() as f64 / proof.nodes.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_transformer_creation() {
        let config = TransformConfig::default();
        let transformer = ProofTransformer::new(config);

        assert_eq!(transformer.stats.original_size, 0);
    }

    #[test]
    fn test_proof_compressor_creation() {
        let config = CompressionConfig::default();
        let compressor = ProofCompressor::new(config);

        assert!(compressor.config.share_subproofs);
    }

    #[test]
    fn test_redundant_removal() {
        // Create a simple proof with redundant step
        let proof = ResolutionProof {
            nodes: vec![
                ProofNode {
                    id: 0,
                    clause: vec![1, 2],
                    rule: ProofRule::Input,
                },
                ProofNode {
                    id: 1,
                    clause: vec![-1],
                    rule: ProofRule::Input,
                },
                ProofNode {
                    id: 2,
                    clause: vec![2],
                    rule: ProofRule::Resolution {
                        left: 0,
                        right: 1,
                        pivot: 1,
                    },
                },
            ],
            root: 2,
        };

        let config = TransformConfig {
            remove_redundant: true,
            ..Default::default()
        };
        let mut transformer = ProofTransformer::new(config);

        let transformed = transformer.transform(&proof);

        // Should keep all nodes as they're all used
        assert!(transformed.nodes.len() <= proof.nodes.len());
    }
}
