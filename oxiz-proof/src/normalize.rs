//! Proof normalization utilities for canonical representation.

use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::FxHashMap;

/// Normalize a proof by reordering axioms alphabetically and renumbering nodes.
///
/// This creates a canonical representation of the proof that is independent of
/// the order in which axioms and inferences were added.
///
/// # Arguments
///
/// * `proof` - The proof to normalize
///
/// # Returns
///
/// A normalized proof with axioms in alphabetical order
///
/// # Example
///
/// ```
/// use oxiz_proof::proof::Proof;
/// use oxiz_proof::normalize::normalize_proof;
///
/// let mut proof = Proof::new();
/// proof.add_axiom("q");
/// proof.add_axiom("p");
/// proof.add_axiom("r");
///
/// let normalized = normalize_proof(&proof);
/// // Axioms will be in order: p, q, r
/// ```
pub fn normalize_proof(proof: &Proof) -> Proof {
    let mut normalized = Proof::new();
    let mut id_map: FxHashMap<ProofNodeId, ProofNodeId> = FxHashMap::default();

    // Collect and sort axioms
    let mut axioms: Vec<(ProofNodeId, String)> = Vec::new();
    let mut inferences: Vec<(ProofNodeId, &ProofStep)> = Vec::new();

    for node in proof.nodes() {
        match &node.step {
            ProofStep::Axiom { conclusion } => {
                axioms.push((node.id, conclusion.clone()));
            }
            ProofStep::Inference { .. } => {
                inferences.push((node.id, &node.step));
            }
        }
    }

    // Sort axioms alphabetically by conclusion
    axioms.sort_by(|a, b| a.1.cmp(&b.1));

    // Add sorted axioms to normalized proof
    for (old_id, conclusion) in axioms {
        let new_id = normalized.add_axiom(conclusion);
        id_map.insert(old_id, new_id);
    }

    // Process inferences in topological order
    let mut added = std::collections::HashSet::new();
    let mut changed = true;

    while changed && added.len() < inferences.len() {
        changed = false;

        for &(old_id, step) in &inferences {
            if added.contains(&old_id) {
                continue;
            }

            if let ProofStep::Inference {
                rule,
                premises,
                conclusion,
                args,
            } = step
            {
                // Check if all premises have been added
                let all_premises_ready = premises.iter().all(|&p| id_map.contains_key(&p));

                if all_premises_ready {
                    let new_premises: Vec<ProofNodeId> = premises
                        .iter()
                        .filter_map(|&old_p| id_map.get(&old_p).copied())
                        .collect();

                    let new_id = if args.is_empty() {
                        normalized.add_inference(rule, new_premises, conclusion.clone())
                    } else {
                        normalized.add_inference_with_args(
                            rule,
                            new_premises,
                            args.to_vec(),
                            conclusion.clone(),
                        )
                    };

                    id_map.insert(old_id, new_id);
                    added.insert(old_id);
                    changed = true;
                }
            }
        }
    }

    normalized
}

/// Canonicalize conclusion strings by removing extra whitespace.
///
/// This standardizes conclusion formatting for easier comparison.
///
/// # Arguments
///
/// * `proof` - The proof to canonicalize
///
/// # Returns
///
/// A new proof with canonicalized conclusions
pub fn canonicalize_conclusions(proof: &Proof) -> Proof {
    let mut canonical = Proof::new();
    let mut id_map: FxHashMap<ProofNodeId, ProofNodeId> = FxHashMap::default();

    for node in proof.nodes() {
        let new_id = match &node.step {
            ProofStep::Axiom { conclusion } => {
                let canonical_conclusion = canonicalize_string(conclusion);
                canonical.add_axiom(canonical_conclusion)
            }
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                args,
            } => {
                let canonical_conclusion = canonicalize_string(conclusion);
                let new_premises: Vec<ProofNodeId> = premises
                    .iter()
                    .filter_map(|&old_p| id_map.get(&old_p).copied())
                    .collect();

                if args.is_empty() {
                    canonical.add_inference(rule, new_premises, canonical_conclusion)
                } else {
                    canonical.add_inference_with_args(
                        rule,
                        new_premises,
                        args.to_vec(),
                        canonical_conclusion,
                    )
                }
            }
        };

        id_map.insert(node.id, new_id);
    }

    canonical
}

/// Canonicalize a string by removing extra whitespace and trimming.
fn canonicalize_string(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_proof_axioms() {
        let mut proof = Proof::new();
        proof.add_axiom("z");
        proof.add_axiom("a");
        proof.add_axiom("m");

        let normalized = normalize_proof(&proof);

        // Check that axioms are sorted
        let nodes: Vec<_> = normalized.nodes().iter().collect();
        assert_eq!(nodes.len(), 3);

        if let ProofStep::Axiom { conclusion } = &nodes[0].step {
            assert_eq!(conclusion, "a");
        } else {
            panic!("Expected axiom");
        }

        if let ProofStep::Axiom { conclusion } = &nodes[1].step {
            assert_eq!(conclusion, "m");
        } else {
            panic!("Expected axiom");
        }

        if let ProofStep::Axiom { conclusion } = &nodes[2].step {
            assert_eq!(conclusion, "z");
        } else {
            panic!("Expected axiom");
        }
    }

    #[test]
    fn test_normalize_proof_with_inferences() {
        let mut proof = Proof::new();
        let z = proof.add_axiom("z");
        let a = proof.add_axiom("a");
        proof.add_inference("and", vec![z, a], "z ∧ a");

        let normalized = normalize_proof(&proof);

        assert_eq!(normalized.len(), 3);
    }

    #[test]
    fn test_canonicalize_conclusions() {
        let mut proof = Proof::new();
        proof.add_axiom("p   q   r"); // Extra spaces
        proof.add_axiom("  x  y  "); // Leading/trailing spaces

        let canonical = canonicalize_conclusions(&proof);

        let nodes: Vec<_> = canonical.nodes().iter().collect();
        if let ProofStep::Axiom { conclusion } = &nodes[0].step {
            assert_eq!(conclusion, "p q r");
        } else {
            panic!("Expected axiom");
        }

        if let ProofStep::Axiom { conclusion } = &nodes[1].step {
            assert_eq!(conclusion, "x y");
        } else {
            panic!("Expected axiom");
        }
    }

    #[test]
    fn test_canonicalize_string() {
        assert_eq!(canonicalize_string("  a  b  c  "), "a b c");
        assert_eq!(canonicalize_string("x"), "x");
        assert_eq!(canonicalize_string("  "), "");
    }

    #[test]
    fn test_normalize_empty_proof() {
        let proof = Proof::new();
        let normalized = normalize_proof(&proof);
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_canonicalize_empty_proof() {
        let proof = Proof::new();
        let canonical = canonicalize_conclusions(&proof);
        assert!(canonical.is_empty());
    }
}
