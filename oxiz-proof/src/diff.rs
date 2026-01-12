//! Proof diffing utilities for comparing two proofs.

use crate::proof::{Proof, ProofNodeId, ProofStep};
use std::fmt;

/// Represents a difference between two proofs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofDiff {
    /// Node exists only in the first proof
    OnlyInFirst(ProofNodeId, String),
    /// Node exists only in the second proof
    OnlyInSecond(ProofNodeId, String),
    /// Node exists in both but with different steps
    Different {
        id1: ProofNodeId,
        id2: ProofNodeId,
        conclusion1: String,
        conclusion2: String,
    },
    /// Structural difference in proof shape
    StructuralDifference(String),
}

impl fmt::Display for ProofDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProofDiff::OnlyInFirst(id, conclusion) => {
                write!(f, "- [{}] {}", id, conclusion)
            }
            ProofDiff::OnlyInSecond(id, conclusion) => {
                write!(f, "+ [{}] {}", id, conclusion)
            }
            ProofDiff::Different {
                id1,
                id2,
                conclusion1,
                conclusion2,
            } => {
                write!(f, "~ [{}] {} ≠ [{}] {}", id1, conclusion1, id2, conclusion2)
            }
            ProofDiff::StructuralDifference(msg) => {
                write!(f, "! {}", msg)
            }
        }
    }
}

/// Compare two proofs and return their differences.
///
/// This function compares two proofs node-by-node and identifies:
/// - Nodes that exist only in the first proof
/// - Nodes that exist only in the second proof
/// - Nodes with matching IDs but different content
/// - Structural differences
///
/// # Arguments
///
/// * `proof1` - The first proof
/// * `proof2` - The second proof
///
/// # Returns
///
/// A vector of differences found between the proofs
///
/// # Example
///
/// ```
/// use oxiz_proof::proof::Proof;
/// use oxiz_proof::diff::diff_proofs;
///
/// let mut proof1 = Proof::new();
/// proof1.add_axiom("p");
/// proof1.add_axiom("q");
///
/// let mut proof2 = Proof::new();
/// proof2.add_axiom("p");
/// proof2.add_axiom("r");
///
/// let diffs = diff_proofs(&proof1, &proof2);
/// assert!(!diffs.is_empty());
/// ```
pub fn diff_proofs(proof1: &Proof, proof2: &Proof) -> Vec<ProofDiff> {
    let mut diffs = Vec::new();

    // Check for size differences
    if proof1.len() != proof2.len() {
        diffs.push(ProofDiff::StructuralDifference(format!(
            "Size mismatch: {} vs {} nodes",
            proof1.len(),
            proof2.len()
        )));
    }

    // Check for root count differences
    if proof1.roots().len() != proof2.roots().len() {
        diffs.push(ProofDiff::StructuralDifference(format!(
            "Root count mismatch: {} vs {}",
            proof1.roots().len(),
            proof2.roots().len()
        )));
    }

    // Build conclusion maps for both proofs
    let mut conclusions1: std::collections::HashMap<String, ProofNodeId> =
        std::collections::HashMap::new();
    let mut conclusions2: std::collections::HashMap<String, ProofNodeId> =
        std::collections::HashMap::new();

    for node in proof1.nodes() {
        let conclusion = match &node.step {
            ProofStep::Axiom { conclusion } => conclusion.clone(),
            ProofStep::Inference { conclusion, .. } => conclusion.clone(),
        };
        conclusions1.insert(conclusion, node.id);
    }

    for node in proof2.nodes() {
        let conclusion = match &node.step {
            ProofStep::Axiom { conclusion } => conclusion.clone(),
            ProofStep::Inference { conclusion, .. } => conclusion.clone(),
        };
        conclusions2.insert(conclusion, node.id);
    }

    // Find conclusions only in proof1
    for (conclusion, id) in &conclusions1 {
        if !conclusions2.contains_key(conclusion) {
            diffs.push(ProofDiff::OnlyInFirst(*id, conclusion.clone()));
        }
    }

    // Find conclusions only in proof2
    for (conclusion, id) in &conclusions2 {
        if !conclusions1.contains_key(conclusion) {
            diffs.push(ProofDiff::OnlyInSecond(*id, conclusion.clone()));
        }
    }

    diffs
}

/// Compute similarity metrics between two proofs.
#[derive(Debug, Clone, Copy)]
pub struct ProofSimilarity {
    /// Jaccard similarity based on conclusions (0.0 to 1.0)
    pub jaccard_similarity: f64,
    /// Ratio of common nodes to total nodes (0.0 to 1.0)
    pub node_overlap: f64,
    /// Structural similarity based on depth and shape (0.0 to 1.0)
    pub structural_similarity: f64,
}

impl fmt::Display for ProofSimilarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Proof Similarity Metrics:")?;
        writeln!(
            f,
            "  Jaccard similarity: {:.2}%",
            self.jaccard_similarity * 100.0
        )?;
        writeln!(f, "  Node overlap: {:.2}%", self.node_overlap * 100.0)?;
        writeln!(
            f,
            "  Structural similarity: {:.2}%",
            self.structural_similarity * 100.0
        )
    }
}

/// Compute similarity metrics between two proofs.
///
/// # Arguments
///
/// * `proof1` - The first proof
/// * `proof2` - The second proof
///
/// # Returns
///
/// Similarity metrics comparing the two proofs
pub fn compute_similarity(proof1: &Proof, proof2: &Proof) -> ProofSimilarity {
    // Collect conclusions from both proofs
    let mut conclusions1 = std::collections::HashSet::new();
    let mut conclusions2 = std::collections::HashSet::new();

    for node in proof1.nodes() {
        let conclusion = match &node.step {
            ProofStep::Axiom { conclusion } => conclusion.clone(),
            ProofStep::Inference { conclusion, .. } => conclusion.clone(),
        };
        conclusions1.insert(conclusion);
    }

    for node in proof2.nodes() {
        let conclusion = match &node.step {
            ProofStep::Axiom { conclusion } => conclusion.clone(),
            ProofStep::Inference { conclusion, .. } => conclusion.clone(),
        };
        conclusions2.insert(conclusion);
    }

    // Compute Jaccard similarity
    let intersection = conclusions1.intersection(&conclusions2).count();
    let union = conclusions1.union(&conclusions2).count();
    let jaccard_similarity = if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    };

    // Compute node overlap
    let total_nodes = proof1.len() + proof2.len();
    let node_overlap = if total_nodes == 0 {
        1.0
    } else {
        (2 * intersection) as f64 / total_nodes as f64
    };

    // Compute structural similarity based on depth
    let stats1 = proof1.stats();
    let stats2 = proof2.stats();

    let depth_diff = (stats1.max_depth as f64 - stats2.max_depth as f64).abs();
    let max_depth = stats1.max_depth.max(stats2.max_depth) as f64;
    let depth_similarity = if max_depth == 0.0 {
        1.0
    } else {
        1.0 - (depth_diff / max_depth)
    };

    let root_diff = (stats1.root_count as f64 - stats2.root_count as f64).abs();
    let max_roots = stats1.root_count.max(stats2.root_count) as f64;
    let root_similarity = if max_roots == 0.0 {
        1.0
    } else {
        1.0 - (root_diff / max_roots)
    };

    let structural_similarity = (depth_similarity + root_similarity) / 2.0;

    ProofSimilarity {
        jaccard_similarity,
        node_overlap,
        structural_similarity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_identical_proofs() {
        let mut proof1 = Proof::new();
        proof1.add_axiom("p");
        proof1.add_axiom("q");

        let mut proof2 = Proof::new();
        proof2.add_axiom("p");
        proof2.add_axiom("q");

        let diffs = diff_proofs(&proof1, &proof2);
        // Should have no content differences, only possibly ID differences
        assert!(
            diffs
                .iter()
                .all(|d| matches!(d, ProofDiff::StructuralDifference(_)))
                || diffs.is_empty()
        );
    }

    #[test]
    fn test_diff_different_proofs() {
        let mut proof1 = Proof::new();
        proof1.add_axiom("p");
        proof1.add_axiom("q");

        let mut proof2 = Proof::new();
        proof2.add_axiom("p");
        proof2.add_axiom("r");

        let diffs = diff_proofs(&proof1, &proof2);
        assert!(!diffs.is_empty());
    }

    #[test]
    fn test_similarity_identical() {
        let mut proof1 = Proof::new();
        proof1.add_axiom("p");
        proof1.add_axiom("q");

        let mut proof2 = Proof::new();
        proof2.add_axiom("p");
        proof2.add_axiom("q");

        let sim = compute_similarity(&proof1, &proof2);
        assert_eq!(sim.jaccard_similarity, 1.0);
        assert_eq!(sim.node_overlap, 1.0);
    }

    #[test]
    fn test_similarity_disjoint() {
        let mut proof1 = Proof::new();
        proof1.add_axiom("p");
        proof1.add_axiom("q");

        let mut proof2 = Proof::new();
        proof2.add_axiom("r");
        proof2.add_axiom("s");

        let sim = compute_similarity(&proof1, &proof2);
        assert_eq!(sim.jaccard_similarity, 0.0);
    }

    #[test]
    fn test_similarity_partial_overlap() {
        let mut proof1 = Proof::new();
        proof1.add_axiom("p");
        proof1.add_axiom("q");

        let mut proof2 = Proof::new();
        proof2.add_axiom("p");
        proof2.add_axiom("r");

        let sim = compute_similarity(&proof1, &proof2);
        assert!(sim.jaccard_similarity > 0.0 && sim.jaccard_similarity < 1.0);
    }

    #[test]
    fn test_diff_display() {
        let diff = ProofDiff::OnlyInFirst(ProofNodeId(0), "p".to_string());
        let display = format!("{}", diff);
        assert!(display.contains("p"));
    }

    #[test]
    fn test_similarity_display() {
        let sim = ProofSimilarity {
            jaccard_similarity: 0.75,
            node_overlap: 0.8,
            structural_similarity: 0.9,
        };
        let display = format!("{}", sim);
        assert!(display.contains("75"));
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_similarity_bounds(size in 1usize..20) {
                let mut proof1 = Proof::new();
                let mut proof2 = Proof::new();

                for i in 0..size {
                    proof1.add_axiom(format!("p{}", i));
                    proof2.add_axiom(format!("q{}", i));
                }

                let sim = compute_similarity(&proof1, &proof2);

                // All similarity metrics should be between 0 and 1
                prop_assert!(sim.jaccard_similarity >= 0.0 && sim.jaccard_similarity <= 1.0);
                prop_assert!(sim.node_overlap >= 0.0 && sim.node_overlap <= 1.0);
                prop_assert!(sim.structural_similarity >= 0.0 && sim.structural_similarity <= 1.0);
            }

            #[test]
            fn prop_identical_proofs_similarity(size in 1usize..20) {
                let mut proof1 = Proof::new();
                for i in 0..size {
                    proof1.add_axiom(format!("p{}", i));
                }

                let mut proof2 = Proof::new();
                for i in 0..size {
                    proof2.add_axiom(format!("p{}", i));
                }

                let sim = compute_similarity(&proof1, &proof2);

                // Identical proofs should have perfect similarity
                prop_assert_eq!(sim.jaccard_similarity, 1.0);
                prop_assert_eq!(sim.node_overlap, 1.0);
            }

            #[test]
            fn prop_disjoint_proofs_zero_jaccard(size1 in 1usize..10, size2 in 1usize..10) {
                let mut proof1 = Proof::new();
                for i in 0..size1 {
                    proof1.add_axiom(format!("p{}", i));
                }

                let mut proof2 = Proof::new();
                for i in 0..size2 {
                    proof2.add_axiom(format!("q{}", i));
                }

                let sim = compute_similarity(&proof1, &proof2);

                // Disjoint proofs should have zero Jaccard similarity
                prop_assert_eq!(sim.jaccard_similarity, 0.0);
            }

            #[test]
            fn prop_diff_empty_proofs(_n in 0usize..1) {
                let proof1 = Proof::new();
                let proof2 = Proof::new();

                let diffs = diff_proofs(&proof1, &proof2);

                // Empty proofs should have no differences
                prop_assert!(diffs.is_empty());
            }

            #[test]
            fn prop_diff_self_is_empty(size in 1usize..20) {
                let mut proof = Proof::new();
                for i in 0..size {
                    proof.add_axiom(format!("p{}", i));
                }

                let diffs = diff_proofs(&proof, &proof);

                // A proof compared with itself should only have structural diffs (if any)
                prop_assert!(diffs.iter().all(|d| matches!(d, ProofDiff::StructuralDifference(_))) || diffs.is_empty());
            }

            #[test]
            fn prop_similarity_symmetric(size in 1usize..15) {
                let mut proof1 = Proof::new();
                let mut proof2 = Proof::new();

                for i in 0..size {
                    proof1.add_axiom(format!("p{}", i));
                    if i % 2 == 0 {
                        proof2.add_axiom(format!("p{}", i));
                    } else {
                        proof2.add_axiom(format!("q{}", i));
                    }
                }

                let sim1 = compute_similarity(&proof1, &proof2);
                let sim2 = compute_similarity(&proof2, &proof1);

                // Similarity should be symmetric
                prop_assert_eq!(sim1.jaccard_similarity, sim2.jaccard_similarity);
                prop_assert_eq!(sim1.node_overlap, sim2.node_overlap);
            }
        }
    }
}
