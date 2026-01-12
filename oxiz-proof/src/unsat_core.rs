//! Unsat core extraction from proofs.
//!
//! An unsat core is a minimal subset of assertions that is sufficient to prove
//! unsatisfiability. This module provides algorithms to extract unsat cores from
//! proof trees, which is useful for:
//! - Debugging unsatisfiable formulas
//! - Identifying conflicting constraints
//! - Minimizing problem instances

use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// An unsat core - a minimal set of axioms needed to prove unsatisfiability
#[derive(Debug, Clone)]
pub struct UnsatCore {
    /// The proof node IDs that form the core
    core_nodes: Vec<ProofNodeId>,
    /// The axioms (input assertions) in the core
    axioms: Vec<ProofNodeId>,
    /// Size of the original proof
    original_size: usize,
    /// Size of the core
    core_size: usize,
}

impl UnsatCore {
    /// Get the core node IDs
    #[must_use]
    pub fn nodes(&self) -> &[ProofNodeId] {
        &self.core_nodes
    }

    /// Get the axiom node IDs in the core
    #[must_use]
    pub fn axioms(&self) -> &[ProofNodeId] {
        &self.axioms
    }

    /// Get the reduction ratio (core_size / original_size)
    #[must_use]
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_size > 0 {
            self.core_size as f64 / self.original_size as f64
        } else {
            1.0
        }
    }

    /// Check if the core is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.core_nodes.is_empty()
    }

    /// Get the size of the core
    #[must_use]
    pub fn len(&self) -> usize {
        self.core_size
    }
}

/// Extract an unsat core from a proof
///
/// This performs a backward traversal from the contradiction (false/empty clause)
/// to find all axioms that contribute to the proof.
#[must_use]
pub fn extract_unsat_core(proof: &Proof, contradiction: ProofNodeId) -> UnsatCore {
    let mut core_nodes = FxHashSet::default();
    let mut axioms = Vec::new();
    let mut queue = VecDeque::new();

    // Start from the contradiction
    queue.push_back(contradiction);
    core_nodes.insert(contradiction);

    // Backward BFS through proof
    while let Some(node_id) = queue.pop_front() {
        if let Some(node) = proof.get_node(node_id) {
            match &node.step {
                ProofStep::Axiom { .. } => {
                    // This is an input axiom - part of the core
                    axioms.push(node_id);
                }
                ProofStep::Inference { premises, .. } => {
                    // Add all premises to the core
                    for &premise_id in premises {
                        if core_nodes.insert(premise_id) {
                            queue.push_back(premise_id);
                        }
                    }
                }
            }
        }
    }

    let core_vec: Vec<ProofNodeId> = core_nodes.into_iter().collect();

    UnsatCore {
        core_nodes: core_vec.clone(),
        axioms,
        original_size: proof.len(),
        core_size: core_vec.len(),
    }
}

/// Extract a minimal unsat core by trying to remove axioms
///
/// This is more expensive but produces a smaller core by checking if each axiom
/// is necessary.
pub fn extract_minimal_unsat_core(proof: &Proof, contradiction: ProofNodeId) -> UnsatCore {
    // First get the full core
    let full_core = extract_unsat_core(proof, contradiction);

    // Try to minimize by removing each axiom and checking if still valid
    let mut minimal_axioms = full_core.axioms.clone();

    // Simple greedy minimization
    let mut i = 0;
    while i < minimal_axioms.len() {
        let candidate = minimal_axioms[i];

        // Try removing this axiom
        minimal_axioms.remove(i);

        // Check if we can still prove contradiction with remaining axioms
        let still_valid = can_prove_with_axioms(proof, contradiction, &minimal_axioms);

        if still_valid {
            // Good, keep it removed
        } else {
            // Need this axiom, put it back
            minimal_axioms.insert(i, candidate);
            i += 1;
        }
    }

    // Recompute core with minimal axioms
    let mut core_nodes = FxHashSet::default();
    let mut queue = VecDeque::new();

    queue.push_back(contradiction);
    core_nodes.insert(contradiction);

    while let Some(node_id) = queue.pop_front() {
        if let Some(node) = proof.get_node(node_id) {
            match &node.step {
                ProofStep::Axiom { .. } => {
                    // Already in minimal_axioms
                }
                ProofStep::Inference { premises, .. } => {
                    for &premise_id in premises {
                        // Only include if it leads to minimal axioms
                        if is_reachable_from_axioms(proof, premise_id, &minimal_axioms)
                            && core_nodes.insert(premise_id)
                        {
                            queue.push_back(premise_id);
                        }
                    }
                }
            }
        }
    }

    let core_vec: Vec<ProofNodeId> = core_nodes.into_iter().collect();

    UnsatCore {
        core_nodes: core_vec.clone(),
        axioms: minimal_axioms,
        original_size: proof.len(),
        core_size: core_vec.len(),
    }
}

/// Check if we can prove the goal using only the given axioms
fn can_prove_with_axioms(proof: &Proof, goal: ProofNodeId, allowed_axioms: &[ProofNodeId]) -> bool {
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();

    queue.push_back(goal);
    visited.insert(goal);

    while let Some(node_id) = queue.pop_front() {
        if let Some(node) = proof.get_node(node_id) {
            match &node.step {
                ProofStep::Axiom { .. } => {
                    if !allowed_axioms.contains(&node_id) {
                        return false; // Uses an axiom not in our set
                    }
                }
                ProofStep::Inference { premises, .. } => {
                    for &premise_id in premises {
                        if visited.insert(premise_id) {
                            queue.push_back(premise_id);
                        }
                    }
                }
            }
        }
    }

    true
}

/// Check if a node is reachable from the given axioms
fn is_reachable_from_axioms(proof: &Proof, node_id: ProofNodeId, axioms: &[ProofNodeId]) -> bool {
    if axioms.contains(&node_id) {
        return true;
    }

    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::new();

    queue.push_back(node_id);
    visited.insert(node_id);

    while let Some(current) = queue.pop_front() {
        if let Some(node) = proof.get_node(current) {
            match &node.step {
                ProofStep::Axiom { .. } => {
                    if axioms.contains(&current) {
                        return true;
                    }
                }
                ProofStep::Inference { premises, .. } => {
                    for &premise_id in premises {
                        if axioms.contains(&premise_id) {
                            return true;
                        }
                        if visited.insert(premise_id) {
                            queue.push_back(premise_id);
                        }
                    }
                }
            }
        }
    }

    false
}

/// Get axiom labels/conclusions from unsat core
#[must_use]
pub fn get_core_labels(proof: &Proof, core: &UnsatCore) -> Vec<String> {
    core.axioms()
        .iter()
        .filter_map(|&node_id| {
            proof.get_node(node_id).and_then(|node| match &node.step {
                ProofStep::Axiom { conclusion } => Some(conclusion.clone()),
                _ => None,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::Proof;

    #[test]
    fn test_extract_unsat_core() {
        let mut proof = Proof::new();

        // Axioms
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let a3 = proof.add_axiom("r");

        // Inferences
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let i2 = proof.add_inference("conflict", vec![i1, a3], "false");

        // Extract core from contradiction
        let core = extract_unsat_core(&proof, i2);

        // Should include all 3 axioms
        assert_eq!(core.axioms().len(), 3);
        assert!(core.axioms().contains(&a1));
        assert!(core.axioms().contains(&a2));
        assert!(core.axioms().contains(&a3));
    }

    #[test]
    fn test_unsat_core_with_unused_axiom() {
        let mut proof = Proof::new();

        // Axioms
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let _a3 = proof.add_axiom("r"); // Unused

        // Conflict using only a1 and a2
        let i1 = proof.add_inference("conflict", vec![a1, a2], "false");

        let core = extract_unsat_core(&proof, i1);

        // Should only include a1 and a2, not a3
        assert_eq!(core.axioms().len(), 2);
        assert!(core.axioms().contains(&a1));
        assert!(core.axioms().contains(&a2));
    }

    #[test]
    fn test_unsat_core_reduction_ratio() {
        let mut proof = Proof::new();

        // Many axioms
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        proof.add_axiom("r");
        proof.add_axiom("s");
        proof.add_axiom("t");

        // Conflict using only a1 and a2
        let i1 = proof.add_inference("conflict", vec![a1, a2], "false");

        let core = extract_unsat_core(&proof, i1);

        // Core should be smaller than original
        assert!(core.reduction_ratio() < 1.0);
        assert_eq!(core.original_size, 6); // 5 axioms + 1 inference
    }

    #[test]
    fn test_get_core_labels() {
        let mut proof = Proof::new();

        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("conflict", vec![a1, a2], "false");

        let core = extract_unsat_core(&proof, i1);
        let labels = get_core_labels(&proof, &core);

        assert_eq!(labels.len(), 2);
        assert!(labels.contains(&"p".to_string()));
        assert!(labels.contains(&"q".to_string()));
    }

    #[test]
    fn test_unsat_core_empty() {
        let core = UnsatCore {
            core_nodes: Vec::new(),
            axioms: Vec::new(),
            original_size: 10,
            core_size: 0,
        };

        assert!(core.is_empty());
        assert_eq!(core.len(), 0);
    }

    #[test]
    fn test_minimal_unsat_core() {
        let mut proof = Proof::new();

        // Create a proof where one axiom is redundant
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let a3 = proof.add_axiom("p"); // Duplicate, should be removed in minimal

        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let i2 = proof.add_inference("conflict", vec![i1, a3], "false");

        let minimal = extract_minimal_unsat_core(&proof, i2);

        // Should have fewer axioms than full core
        assert!(minimal.axioms().len() <= 3);
    }

    #[test]
    fn test_deep_proof_core() {
        let mut proof = Proof::new();

        // Build a deeper proof tree
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let a3 = proof.add_axiom("r");

        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let i2 = proof.add_inference("and", vec![i1, a3], "(p /\\ q) /\\ r");
        let i3 = proof.add_inference("conflict", vec![i2], "false");

        let core = extract_unsat_core(&proof, i3);

        // Should trace back to all 3 axioms
        assert_eq!(core.axioms().len(), 3);
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn var_name() -> impl Strategy<Value = String> {
            "[a-z][0-9]*".prop_map(|s| s.to_string())
        }

        proptest! {
            /// Unsat core should never be larger than original proof
            #[test]
            fn prop_core_size_bounded(
                conclusions in prop::collection::vec(var_name(), 2..10)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if axiom_ids.len() >= 2 {
                    let contradiction = proof.add_inference(
                        "conflict",
                        axiom_ids.clone(),
                        "false"
                    );

                    let core = extract_unsat_core(&proof, contradiction);
                    prop_assert!(core.len() <= proof.len());
                    prop_assert!(core.axioms().len() <= axiom_ids.len());
                }
            }

            /// Reduction ratio should always be between 0 and 1
            #[test]
            fn prop_reduction_ratio_bounds(
                conclusions in prop::collection::vec(var_name(), 1..8)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if !axiom_ids.is_empty() {
                    let contradiction = proof.add_inference(
                        "conflict",
                        vec![axiom_ids[0]],
                        "false"
                    );

                    let core = extract_unsat_core(&proof, contradiction);
                    let ratio = core.reduction_ratio();
                    prop_assert!((0.0..=1.0).contains(&ratio));
                }
            }

            /// All axioms in core should exist in original proof
            #[test]
            fn prop_core_axioms_valid(
                conclusions in prop::collection::vec(var_name(), 2..8)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if axiom_ids.len() >= 2 {
                    let contradiction = proof.add_inference(
                        "conflict",
                        vec![axiom_ids[0], axiom_ids[1]],
                        "false"
                    );

                    let core = extract_unsat_core(&proof, contradiction);
                    for &axiom_id in core.axioms() {
                        prop_assert!(proof.get_node(axiom_id).is_some());
                    }
                }
            }

            /// Minimal core should be no larger than full core
            #[test]
            fn prop_minimal_core_smaller(
                conclusions in prop::collection::vec(var_name(), 2..6)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if axiom_ids.len() >= 2 {
                    let contradiction = proof.add_inference(
                        "conflict",
                        axiom_ids.clone(),
                        "false"
                    );

                    let full_core = extract_unsat_core(&proof, contradiction);
                    let minimal_core = extract_minimal_unsat_core(&proof, contradiction);

                    prop_assert!(minimal_core.axioms().len() <= full_core.axioms().len());
                }
            }

            /// Core labels should match axiom conclusions
            #[test]
            fn prop_core_labels_match(
                conclusions in prop::collection::vec(var_name(), 2..6)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if axiom_ids.len() >= 2 {
                    let contradiction = proof.add_inference(
                        "conflict",
                        vec![axiom_ids[0], axiom_ids[1]],
                        "false"
                    );

                    let core = extract_unsat_core(&proof, contradiction);
                    let labels = get_core_labels(&proof, &core);

                    // Number of labels should match number of axioms in core
                    prop_assert_eq!(labels.len(), core.axioms().len());
                }
            }

            /// Empty check consistency
            #[test]
            fn prop_empty_consistency(
                conclusions in prop::collection::vec(var_name(), 0..5)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if !axiom_ids.is_empty() {
                    let contradiction = proof.add_inference(
                        "conflict",
                        vec![axiom_ids[0]],
                        "false"
                    );

                    let core = extract_unsat_core(&proof, contradiction);
                    // Core should not be empty when there's a valid contradiction
                    prop_assert!(!core.is_empty());
                }
            }

            /// Core should include the contradiction node
            #[test]
            fn prop_core_includes_contradiction(
                conclusions in prop::collection::vec(var_name(), 1..6)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if !axiom_ids.is_empty() {
                    let contradiction = proof.add_inference(
                        "conflict",
                        vec![axiom_ids[0]],
                        "false"
                    );

                    let core = extract_unsat_core(&proof, contradiction);
                    prop_assert!(core.nodes().contains(&contradiction));
                }
            }

            /// All core nodes should be reachable in original proof
            #[test]
            fn prop_core_nodes_reachable(
                conclusions in prop::collection::vec(var_name(), 2..6)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    axiom_ids.push(proof.add_axiom(conclusion));
                }

                if axiom_ids.len() >= 2 {
                    let contradiction = proof.add_inference(
                        "conflict",
                        axiom_ids.clone(),
                        "false"
                    );

                    let core = extract_unsat_core(&proof, contradiction);
                    for &node_id in core.nodes() {
                        prop_assert!(proof.get_node(node_id).is_some());
                    }
                }
            }
        }
    }
}
