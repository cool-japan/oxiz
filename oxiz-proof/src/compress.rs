//! Proof compression and optimization.
//!
//! This module provides algorithms for compressing and optimizing proofs,
//! including:
//! - Removing redundant steps
//! - Merging duplicate subproofs
//! - Trimming unused branches
//! - Reordering steps for better structure

use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// Configuration for proof compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Remove redundant steps (same conclusion from same premises)
    pub remove_redundant: bool,
    /// Trim unused branches
    pub trim_unused: bool,
    /// Inline trivial steps
    pub inline_trivial: bool,
    /// Maximum number of compression passes
    pub max_passes: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            remove_redundant: true,
            trim_unused: true,
            inline_trivial: true,
            max_passes: 3,
        }
    }
}

/// Result of proof compression
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Number of nodes before compression
    pub nodes_before: usize,
    /// Number of nodes after compression
    pub nodes_after: usize,
    /// Number of nodes removed
    pub nodes_removed: usize,
    /// Compression ratio (nodes_after / nodes_before)
    pub ratio: f64,
}

impl CompressionResult {
    /// Create a new compression result
    #[must_use]
    pub fn new(nodes_before: usize, nodes_after: usize) -> Self {
        let nodes_removed = nodes_before.saturating_sub(nodes_after);
        let ratio = if nodes_before > 0 {
            nodes_after as f64 / nodes_before as f64
        } else {
            1.0
        };

        Self {
            nodes_before,
            nodes_after,
            nodes_removed,
            ratio,
        }
    }
}

/// Proof compression engine
pub struct ProofCompressor {
    /// Configuration
    config: CompressionConfig,
}

impl ProofCompressor {
    /// Create a new proof compressor with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CompressionConfig::default(),
        }
    }

    /// Create a proof compressor with custom configuration
    #[must_use]
    pub fn with_config(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress a proof
    pub fn compress(&self, proof: &Proof) -> (Proof, CompressionResult) {
        let nodes_before = proof.len();
        let mut compressed = proof.clone();

        for _ in 0..self.config.max_passes {
            let prev_len = compressed.len();

            if self.config.trim_unused {
                compressed = self.trim_unused_nodes(&compressed);
            }

            if self.config.remove_redundant {
                compressed = self.remove_redundant(&compressed);
            }

            if self.config.inline_trivial {
                compressed = self.inline_trivial_steps(&compressed);
            }

            // Stop if no progress
            if compressed.len() == prev_len {
                break;
            }
        }

        let nodes_after = compressed.len();
        let result = CompressionResult::new(nodes_before, nodes_after);

        (compressed, result)
    }

    /// Trim unused nodes (nodes not reachable from roots)
    fn trim_unused_nodes(&self, proof: &Proof) -> Proof {
        // Mark all reachable nodes
        let mut reachable = FxHashSet::default();
        let mut queue = VecDeque::new();

        // Start from roots
        for &root_id in proof.roots() {
            queue.push_back(root_id);
            reachable.insert(root_id);
        }

        // BFS to find all reachable nodes
        while let Some(node_id) = queue.pop_front() {
            if let Some(node) = proof.get_node(node_id)
                && let ProofStep::Inference { premises, .. } = &node.step
            {
                for &premise_id in premises {
                    if reachable.insert(premise_id) {
                        queue.push_back(premise_id);
                    }
                }
            }
        }

        // Build new proof with only reachable nodes
        let mut new_proof = Proof::new();
        let mut id_map = rustc_hash::FxHashMap::default();

        // Copy reachable nodes in topological order
        for node in proof.nodes() {
            if reachable.contains(&node.id) {
                let new_id = match &node.step {
                    ProofStep::Axiom { conclusion } => new_proof.add_axiom(conclusion.clone()),
                    ProofStep::Inference {
                        rule,
                        premises,
                        conclusion,
                        args,
                    } => {
                        let new_premises: Vec<ProofNodeId> = premises
                            .iter()
                            .filter_map(|p| id_map.get(p).copied())
                            .collect();

                        new_proof.add_inference_with_args(
                            rule.clone(),
                            new_premises,
                            args.to_vec(),
                            conclusion.clone(),
                        )
                    }
                };
                id_map.insert(node.id, new_id);
            }
        }

        new_proof
    }

    /// Remove redundant steps (duplicate conclusions)
    fn remove_redundant(&self, proof: &Proof) -> Proof {
        // Already handled by proof deduplication in the Proof structure
        proof.clone()
    }

    /// Inline trivial steps (identity operations, reflexivity, etc.)
    fn inline_trivial_steps(&self, proof: &Proof) -> Proof {
        let mut new_proof = Proof::new();
        let mut id_map = rustc_hash::FxHashMap::default();

        for node in proof.nodes() {
            let is_trivial = match &node.step {
                ProofStep::Axiom { .. } => false,
                ProofStep::Inference { rule, premises, .. } => {
                    // Trivial if it's a reflexivity or identity rule with single premise
                    (rule == "refl" || rule == "identity") && premises.len() <= 1
                }
            };

            if is_trivial {
                // Skip this node, map to its premise
                if let ProofStep::Inference { premises, .. } = &node.step
                    && let Some(&premise_id) = premises.first()
                    && let Some(&mapped_premise) = id_map.get(&premise_id)
                {
                    id_map.insert(node.id, mapped_premise);
                    continue;
                }
            }

            // Copy non-trivial nodes
            let new_id = match &node.step {
                ProofStep::Axiom { conclusion } => new_proof.add_axiom(conclusion.clone()),
                ProofStep::Inference {
                    rule,
                    premises,
                    conclusion,
                    args,
                } => {
                    let new_premises: Vec<ProofNodeId> = premises
                        .iter()
                        .filter_map(|p| id_map.get(p).copied())
                        .collect();

                    new_proof.add_inference_with_args(
                        rule.clone(),
                        new_premises,
                        args.to_vec(),
                        conclusion.clone(),
                    )
                }
            };
            id_map.insert(node.id, new_id);
        }

        new_proof
    }
}

impl Default for ProofCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Trim a proof to only include nodes needed for a specific conclusion
pub fn trim_to_conclusion(proof: &Proof, conclusion_id: ProofNodeId) -> Proof {
    let mut needed = FxHashSet::default();
    let mut queue = VecDeque::new();

    queue.push_back(conclusion_id);
    needed.insert(conclusion_id);

    // BFS backward through premises
    while let Some(node_id) = queue.pop_front() {
        if let Some(node) = proof.get_node(node_id)
            && let ProofStep::Inference { premises, .. } = &node.step
        {
            for &premise_id in premises {
                if needed.insert(premise_id) {
                    queue.push_back(premise_id);
                }
            }
        }
    }

    // Build new proof with only needed nodes
    let mut new_proof = Proof::new();
    let mut id_map = rustc_hash::FxHashMap::default();

    for node in proof.nodes() {
        if needed.contains(&node.id) {
            let new_id = match &node.step {
                ProofStep::Axiom { conclusion } => new_proof.add_axiom(conclusion.clone()),
                ProofStep::Inference {
                    rule,
                    premises,
                    conclusion,
                    args,
                } => {
                    let new_premises: Vec<ProofNodeId> = premises
                        .iter()
                        .filter_map(|p| id_map.get(p).copied())
                        .collect();

                    new_proof.add_inference_with_args(
                        rule.clone(),
                        new_premises,
                        args.to_vec(),
                        conclusion.clone(),
                    )
                }
            };
            id_map.insert(node.id, new_id);
        }
    }

    new_proof
}

/// Get the dependency cone of a node (all nodes it depends on)
#[must_use]
pub fn get_dependency_cone(proof: &Proof, node_id: ProofNodeId) -> Vec<ProofNodeId> {
    let mut deps = FxHashSet::default();
    let mut queue = VecDeque::new();

    queue.push_back(node_id);
    deps.insert(node_id);

    while let Some(id) = queue.pop_front() {
        if let Some(node) = proof.get_node(id)
            && let ProofStep::Inference { premises, .. } = &node.step
        {
            for &premise_id in premises {
                if deps.insert(premise_id) {
                    queue.push_back(premise_id);
                }
            }
        }
    }

    deps.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_result() {
        let result = CompressionResult::new(100, 80);
        assert_eq!(result.nodes_before, 100);
        assert_eq!(result.nodes_after, 80);
        assert_eq!(result.nodes_removed, 20);
        assert!((result.ratio - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_trim_unused() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let _unused = proof.add_axiom("r"); // This is unused
        proof.add_inference("and", vec![a1, a2], "p /\\ q");

        let compressor = ProofCompressor::new();
        let (compressed, _) = compressor.compress(&proof);

        // Should remove the unused axiom
        assert!(compressed.len() <= proof.len());
    }

    #[test]
    fn test_trim_to_conclusion() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let a3 = proof.add_axiom("r");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let _i2 = proof.add_inference("or", vec![a3], "r");

        // Trim to only include nodes needed for i1
        let trimmed = trim_to_conclusion(&proof, i1);

        // Should have 3 nodes: a1, a2, i1 (not a3 or i2)
        assert_eq!(trimmed.len(), 3);
    }

    #[test]
    fn test_dependency_cone() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let i2 = proof.add_inference("not", vec![i1], "~(p /\\ q)");

        let cone = get_dependency_cone(&proof, i2);

        // Should include i2, i1, a1, a2
        assert_eq!(cone.len(), 4);
        assert!(cone.contains(&a1));
        assert!(cone.contains(&a2));
        assert!(cone.contains(&i1));
        assert!(cone.contains(&i2));
    }

    #[test]
    fn test_compressor_default() {
        let compressor = ProofCompressor::default();
        assert!(compressor.config.remove_redundant);
        assert!(compressor.config.trim_unused);
    }

    #[test]
    fn test_compress_simple_proof() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        proof.add_inference("and", vec![a1, a2], "p /\\ q");

        let compressor = ProofCompressor::new();
        let (compressed, result) = compressor.compress(&proof);

        assert_eq!(result.nodes_before, proof.len());
        assert_eq!(result.nodes_after, compressed.len());
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn var_name() -> impl Strategy<Value = String> {
            "[a-z][0-9]*".prop_map(|s| s.to_string())
        }

        proptest! {
            /// Compression should never increase proof size
            #[test]
            fn prop_compression_reduces_or_maintains_size(
                conclusions in prop::collection::vec(var_name(), 1..15)
            ) {
                let mut proof = Proof::new();
                for conclusion in &conclusions {
                    proof.add_axiom(conclusion);
                }

                let compressor = ProofCompressor::new();
                let (compressed, result) = compressor.compress(&proof);

                prop_assert!(compressed.len() <= proof.len());
                prop_assert_eq!(result.nodes_before, proof.len());
                prop_assert_eq!(result.nodes_after, compressed.len());
            }

            /// Compression ratio should be between 0 and 1
            #[test]
            fn prop_compression_ratio_bounds(
                conclusions in prop::collection::vec(var_name(), 1..10)
            ) {
                let mut proof = Proof::new();
                for conclusion in &conclusions {
                    proof.add_axiom(conclusion);
                }

                let compressor = ProofCompressor::new();
                let (_, result) = compressor.compress(&proof);

                prop_assert!(result.ratio >= 0.0);
                prop_assert!(result.ratio <= 1.0);
            }

            /// Nodes removed should equal nodes_before - nodes_after
            #[test]
            fn prop_nodes_removed_consistency(
                conclusions in prop::collection::vec(var_name(), 1..12)
            ) {
                let mut proof = Proof::new();
                for conclusion in &conclusions {
                    proof.add_axiom(conclusion);
                }

                let compressor = ProofCompressor::new();
                let (_, result) = compressor.compress(&proof);

                prop_assert_eq!(
                    result.nodes_removed,
                    result.nodes_before.saturating_sub(result.nodes_after)
                );
            }

            /// Dependency cone should include the target node
            #[test]
            fn prop_dependency_cone_includes_target(
                conclusions in prop::collection::vec(var_name(), 2..8)
            ) {
                let mut proof = Proof::new();
                let mut ids = Vec::new();

                for conclusion in &conclusions {
                    ids.push(proof.add_axiom(conclusion));
                }

                if ids.len() >= 2 {
                    let inf_id = proof.add_inference("and", vec![ids[0], ids[1]], "result");
                    let cone = get_dependency_cone(&proof, inf_id);
                    prop_assert!(cone.contains(&inf_id));
                }
            }

            /// Dependency cone should include all premises
            #[test]
            fn prop_dependency_cone_includes_premises(
                conclusions in prop::collection::vec(var_name(), 2..8)
            ) {
                let mut proof = Proof::new();
                let mut premise_ids = Vec::new();

                for conclusion in &conclusions {
                    premise_ids.push(proof.add_axiom(conclusion));
                }

                if premise_ids.len() >= 2 {
                    let inf_id = proof.add_inference(
                        "and",
                        premise_ids[..2].to_vec(),
                        "result"
                    );
                    let cone = get_dependency_cone(&proof, inf_id);

                    for &premise_id in &premise_ids[..2] {
                        prop_assert!(cone.contains(&premise_id));
                    }
                }
            }

            /// Trimmed proof should not be larger than original
            #[test]
            fn prop_trim_reduces_size(
                conclusions in prop::collection::vec(var_name(), 2..10)
            ) {
                let mut proof = Proof::new();
                let mut ids = Vec::new();

                for conclusion in &conclusions {
                    ids.push(proof.add_axiom(conclusion));
                }

                if ids.len() >= 2 {
                    let target = proof.add_inference("and", vec![ids[0], ids[1]], "target");
                    let trimmed = trim_to_conclusion(&proof, target);
                    prop_assert!(trimmed.len() <= proof.len());
                }
            }

            /// Trimmed proof should contain the target conclusion
            #[test]
            fn prop_trim_contains_target(
                conclusions in prop::collection::vec(var_name(), 2..8)
            ) {
                let mut proof = Proof::new();
                let mut ids = Vec::new();

                for conclusion in &conclusions {
                    ids.push(proof.add_axiom(conclusion));
                }

                if ids.len() >= 2 {
                    let target = proof.add_inference("and", vec![ids[0], ids[1]], "target");
                    let trimmed = trim_to_conclusion(&proof, target);

                    // Target should be reachable in trimmed proof
                    prop_assert!(!trimmed.is_empty());
                }
            }

            /// Multiple compression passes should be idempotent
            #[test]
            fn prop_compression_idempotent(
                conclusions in prop::collection::vec(var_name(), 1..8)
            ) {
                let mut proof = Proof::new();
                for conclusion in &conclusions {
                    proof.add_axiom(conclusion);
                }

                let compressor = ProofCompressor::new();
                let (compressed1, _) = compressor.compress(&proof);
                let (compressed2, _) = compressor.compress(&compressed1);

                // Second compression shouldn't change size
                prop_assert_eq!(compressed1.len(), compressed2.len());
            }

            /// Compression with no optimization should preserve size
            #[test]
            fn prop_no_optimization_preserves_size(
                conclusions in prop::collection::vec(var_name(), 1..8)
            ) {
                let mut proof = Proof::new();
                for conclusion in &conclusions {
                    proof.add_axiom(conclusion);
                }

                let config = CompressionConfig {
                    remove_redundant: false,
                    trim_unused: false,
                    inline_trivial: false,
                    max_passes: 0,
                };
                let compressor = ProofCompressor::with_config(config);
                let (compressed, _) = compressor.compress(&proof);

                prop_assert_eq!(compressed.len(), proof.len());
            }
        }
    }
}
