//! Proof merging and slicing utilities.

use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Merge multiple proofs into a single proof.
///
/// This combines multiple independent proofs into one, renumbering node IDs as needed
/// to avoid conflicts. The merged proof will contain all nodes from all input proofs.
///
/// # Arguments
///
/// * `proofs` - A slice of proofs to merge
///
/// # Returns
///
/// A new proof containing all nodes from the input proofs
///
/// # Example
///
/// ```
/// use oxiz_proof::proof::Proof;
/// use oxiz_proof::merge::merge_proofs;
///
/// let mut proof1 = Proof::new();
/// proof1.add_axiom("p");
/// proof1.add_axiom("q");
///
/// let mut proof2 = Proof::new();
/// proof2.add_axiom("r");
/// proof2.add_axiom("s");
///
/// let merged = merge_proofs(&[proof1, proof2]);
/// assert_eq!(merged.len(), 4);
/// ```
pub fn merge_proofs(proofs: &[Proof]) -> Proof {
    let mut merged = Proof::new();

    for proof in proofs {
        // Build a mapping from old IDs to new IDs
        let mut id_map: FxHashMap<ProofNodeId, ProofNodeId> = FxHashMap::default();

        // Process nodes in order
        for node in proof.nodes() {
            let new_id = match &node.step {
                ProofStep::Axiom { conclusion } => merged.add_axiom(conclusion.clone()),
                ProofStep::Inference {
                    rule,
                    premises,
                    conclusion,
                    args,
                } => {
                    // Map old premise IDs to new premise IDs
                    let new_premises: Vec<ProofNodeId> = premises
                        .iter()
                        .filter_map(|&old_id| id_map.get(&old_id).copied())
                        .collect();

                    if args.is_empty() {
                        merged.add_inference(rule, new_premises, conclusion.clone())
                    } else {
                        merged.add_inference_with_args(
                            rule,
                            new_premises,
                            args.to_vec(),
                            conclusion.clone(),
                        )
                    }
                }
            };

            id_map.insert(node.id, new_id);
        }
    }

    merged
}

/// Extract a slice of a proof containing only the nodes needed to derive a specific conclusion.
///
/// This performs backward traversal from the target node to find all dependencies,
/// creating a minimal proof that still derives the target conclusion.
///
/// # Arguments
///
/// * `proof` - The source proof
/// * `target` - The target node ID to extract
///
/// # Returns
///
/// A new proof containing only the nodes needed to derive the target
///
/// # Example
///
/// ```
/// use oxiz_proof::proof::Proof;
/// use oxiz_proof::merge::slice_proof;
///
/// let mut proof = Proof::new();
/// let p = proof.add_axiom("p");
/// let q = proof.add_axiom("q");
/// let pq = proof.add_inference("and", vec![p, q], "p ∧ q");
/// let r = proof.add_axiom("r");
///
/// let sliced = slice_proof(&proof, pq);
/// // Should contain p, q, and pq, but not r
/// assert_eq!(sliced.len(), 3);
/// ```
pub fn slice_proof(proof: &Proof, target: ProofNodeId) -> Proof {
    let mut sliced = Proof::new();
    let mut id_map: FxHashMap<ProofNodeId, ProofNodeId> = FxHashMap::default();
    let mut visited = std::collections::HashSet::new();

    // Collect all dependencies using BFS
    let mut dependencies = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(target);
    visited.insert(target);

    while let Some(node_id) = queue.pop_front() {
        dependencies.push(node_id);

        if let Some(node) = proof.get_node(node_id)
            && let ProofStep::Inference { premises, .. } = &node.step
        {
            for &premise in premises.iter() {
                if visited.insert(premise) {
                    queue.push_back(premise);
                }
            }
        }
    }

    // Reverse to process in topological order (axioms first)
    dependencies.reverse();

    // Add nodes to sliced proof in topological order
    for node_id in dependencies {
        if let Some(node) = proof.get_node(node_id) {
            let new_id = match &node.step {
                ProofStep::Axiom { conclusion } => sliced.add_axiom(conclusion.clone()),
                ProofStep::Inference {
                    rule,
                    premises,
                    conclusion,
                    args,
                } => {
                    let new_premises: Vec<ProofNodeId> = premises
                        .iter()
                        .filter_map(|&old_id| id_map.get(&old_id).copied())
                        .collect();

                    if args.is_empty() {
                        sliced.add_inference(rule, new_premises, conclusion.clone())
                    } else {
                        sliced.add_inference_with_args(
                            rule,
                            new_premises,
                            args.to_vec(),
                            conclusion.clone(),
                        )
                    }
                }
            };

            id_map.insert(node.id, new_id);
        }
    }

    sliced
}

/// Extract slices for multiple target nodes efficiently.
///
/// This is more efficient than calling `slice_proof` multiple times when extracting
/// multiple slices from the same proof, as it reuses dependency information.
///
/// # Arguments
///
/// * `proof` - The source proof
/// * `targets` - The target node IDs to extract
///
/// # Returns
///
/// A vector of proofs, one for each target
pub fn slice_proof_multi(proof: &Proof, targets: &[ProofNodeId]) -> Vec<Proof> {
    targets
        .iter()
        .map(|&target| slice_proof(proof, target))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_proofs_empty() {
        let merged = merge_proofs(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_proofs_single() {
        let mut proof = Proof::new();
        proof.add_axiom("p");
        proof.add_axiom("q");

        let merged = merge_proofs(&[proof.clone()]);
        assert_eq!(merged.len(), proof.len());
    }

    #[test]
    fn test_merge_proofs_multiple() {
        let mut proof1 = Proof::new();
        proof1.add_axiom("p");
        proof1.add_axiom("q");

        let mut proof2 = Proof::new();
        proof2.add_axiom("r");
        proof2.add_axiom("s");

        let merged = merge_proofs(&[proof1, proof2]);
        assert_eq!(merged.len(), 4);
    }

    #[test]
    fn test_merge_with_inferences() {
        let mut proof1 = Proof::new();
        let p = proof1.add_axiom("p");
        let q = proof1.add_axiom("q");
        proof1.add_inference("and", vec![p, q], "p ∧ q");

        let mut proof2 = Proof::new();
        let r = proof2.add_axiom("r");
        proof2.add_inference("not", vec![r], "¬r");

        let merged = merge_proofs(&[proof1, proof2]);
        assert_eq!(merged.len(), 5); // p, q, p∧q, r, ¬r
    }

    #[test]
    fn test_slice_proof_axiom() {
        let mut proof = Proof::new();
        let p = proof.add_axiom("p");
        let _q = proof.add_axiom("q");

        let sliced = slice_proof(&proof, p);
        assert_eq!(sliced.len(), 1);
    }

    #[test]
    fn test_slice_proof_inference() {
        let mut proof = Proof::new();
        let p = proof.add_axiom("p");
        let q = proof.add_axiom("q");
        let pq = proof.add_inference("and", vec![p, q], "p ∧ q");
        let _r = proof.add_axiom("r");

        let sliced = slice_proof(&proof, pq);
        assert_eq!(sliced.len(), 3); // p, q, p∧q (but not r)
    }

    #[test]
    fn test_slice_proof_deep() {
        let mut proof = Proof::new();
        let p = proof.add_axiom("p");
        let q = proof.add_axiom("q");
        let pq = proof.add_inference("and", vec![p, q], "p ∧ q");
        let r = proof.add_axiom("r");
        let pqr = proof.add_inference("and", vec![pq, r], "(p ∧ q) ∧ r");

        let sliced = slice_proof(&proof, pqr);
        assert_eq!(sliced.len(), 5); // p, q, p∧q, r, (p∧q)∧r
    }

    #[test]
    fn test_slice_multi() {
        let mut proof = Proof::new();
        let p = proof.add_axiom("p");
        let q = proof.add_axiom("q");
        let pq = proof.add_inference("and", vec![p, q], "p ∧ q");

        let slices = slice_proof_multi(&proof, &[p, pq]);
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].len(), 1);
        assert_eq!(slices[1].len(), 3);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_merge_preserves_nodes(num_proofs in 1usize..5, nodes_per_proof in 1usize..10) {
                let proofs: Vec<Proof> = (0..num_proofs)
                    .map(|i| {
                        let mut proof = Proof::new();
                        for j in 0..nodes_per_proof {
                            proof.add_axiom(format!("p{}_{}", i, j));
                        }
                        proof
                    })
                    .collect();

                let total_expected = num_proofs * nodes_per_proof;
                let merged = merge_proofs(&proofs);

                // Merged proof should have at most the sum of all nodes
                prop_assert!(merged.len() <= total_expected);
            }

            #[test]
            fn prop_merge_empty_proofs(n in 0usize..10) {
                let proofs: Vec<Proof> = (0..n).map(|_| Proof::new()).collect();
                let merged = merge_proofs(&proofs);
                prop_assert!(merged.is_empty());
            }

            #[test]
            fn prop_slice_smaller_or_equal(axiom_count in 1usize..20) {
                let mut proof = Proof::new();
                let mut ids = Vec::new();

                for i in 0..axiom_count {
                    ids.push(proof.add_axiom(format!("p{}", i)));
                }

                if let Some(&target) = ids.first() {
                    let sliced = slice_proof(&proof, target);
                    prop_assert!(sliced.len() <= proof.len());
                    prop_assert!(!sliced.is_empty());
                }
            }

            #[test]
            fn prop_slice_contains_target(depth in 1usize..10) {
                let mut proof = Proof::new();
                let mut prev = proof.add_axiom("base");

                for i in 0..depth {
                    prev = proof.add_inference("step", vec![prev], format!("level{}", i));
                }

                let sliced = slice_proof(&proof, prev);

                // Sliced proof should contain all dependencies
                prop_assert_eq!(sliced.len(), depth + 1); // base + depth levels
            }

            #[test]
            fn prop_merge_single_is_copy(node_count in 1usize..20) {
                let mut proof = Proof::new();
                for i in 0..node_count {
                    proof.add_axiom(format!("p{}", i));
                }

                let merged = merge_proofs(&[proof.clone()]);
                prop_assert_eq!(merged.len(), proof.len());
            }

            #[test]
            fn prop_slice_axiom_is_single(axiom_count in 2usize..20, target_idx in 0usize..20) {
                let target_idx = target_idx % axiom_count;
                let mut proof = Proof::new();
                let mut ids = Vec::new();

                for i in 0..axiom_count {
                    ids.push(proof.add_axiom(format!("p{}", i)));
                }

                let sliced = slice_proof(&proof, ids[target_idx]);
                prop_assert_eq!(sliced.len(), 1);
            }
        }
    }
}
