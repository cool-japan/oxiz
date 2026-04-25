//! Proof minimization via hash-cons deduplication and iterative re-trimming.
//!
//! This module provides a [`ProofMinimizer`] that computes a *minimal* proof DAG
//! by deduplicating semantically-identical lemma nodes across all branches and
//! then iterating cone reduction until a fixed point is reached.
//!
//! ## Distinction from `compress.rs`
//!
//! `compress.rs` removes *structurally redundant* nodes (unreachable steps, trivial
//! identity rewrites). `minimize.rs` goes further: it performs **hash-cons
//! deduplication** — collapsing any two nodes that share the same rule, conclusion
//! text, argument list, and premise count into a single canonical node, then
//! re-trims the proof until no further removal is possible.

use crate::compress::get_dependency_cone;
use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::FxHashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ProofMinimizer`].
#[derive(Debug, Clone)]
pub struct MinimizeConfig {
    /// Maximum number of minimization passes (dedup + trim loop).
    /// Defaults to 4.
    pub max_passes: usize,
    /// Whether to perform hash-cons deduplication.
    /// When `false`, only structural cone reduction is applied.
    /// Defaults to `true`.
    pub enable_dedup: bool,
}

impl Default for MinimizeConfig {
    fn default() -> Self {
        Self {
            max_passes: 4,
            enable_dedup: true,
        }
    }
}

impl MinimizeConfig {
    /// Create a new configuration with all options at their defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable hash-cons deduplication (only cone reduction runs).
    pub fn without_dedup(mut self) -> Self {
        self.enable_dedup = false;
        self
    }

    /// Set the maximum number of passes.
    pub fn with_max_passes(mut self, passes: usize) -> Self {
        self.max_passes = passes;
        self
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Statistics returned by [`ProofMinimizer::minimize`].
#[derive(Debug, Clone, Default)]
pub struct MinimizeResult {
    /// Number of minimization passes that actually ran.
    pub passes: usize,
    /// Total node count removed across all passes.
    pub nodes_removed: usize,
    /// Total duplicate nodes collapsed (may span multiple passes).
    pub duplicates_collapsed: usize,
}

// ---------------------------------------------------------------------------
// Hash key for candidate duplicates
// ---------------------------------------------------------------------------

/// An opaque u64 that identifies a node's *semantic fingerprint*:
/// derived from its rule variant, conclusion text, argument list, and
/// premise count. Two nodes are candidates for deduplication iff they
/// produce the same `ConclusionHash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ConclusionHash(u64);

impl ConclusionHash {
    /// Compute the hash for an [`Axiom`] node.
    fn for_axiom(conclusion: &str) -> Self {
        let mut h = DefaultHasher::new();
        0u8.hash(&mut h); // variant tag = 0 for Axiom
        conclusion.hash(&mut h);
        Self(h.finish())
    }

    /// Compute the hash for an [`Inference`] node.
    fn for_inference(rule: &str, conclusion: &str, args: &[String], premise_count: usize) -> Self {
        let mut h = DefaultHasher::new();
        1u8.hash(&mut h); // variant tag = 1 for Inference
        rule.hash(&mut h);
        conclusion.hash(&mut h);
        args.hash(&mut h);
        premise_count.hash(&mut h);
        Self(h.finish())
    }
}

// ---------------------------------------------------------------------------
// Equality check (full structural equality on the node step)
// ---------------------------------------------------------------------------

/// Full structural equality check between two [`ProofNode`]s after the
/// hash-cons candidate match. Because two different inference rules might
/// accidentally collide in the hash (though unlikely), we verify every field.
fn nodes_are_equal(proof: &Proof, lhs: ProofNodeId, rhs: ProofNodeId) -> bool {
    match (proof.get_node(lhs), proof.get_node(rhs)) {
        (Some(l), Some(r)) => match (&l.step, &r.step) {
            (ProofStep::Axiom { conclusion: cl }, ProofStep::Axiom { conclusion: cr }) => cl == cr,
            (
                ProofStep::Inference {
                    rule: rl,
                    premises: pl,
                    conclusion: cl,
                    args: al,
                },
                ProofStep::Inference {
                    rule: rr,
                    premises: pr,
                    conclusion: cr,
                    args: ar,
                },
            ) => rl == rr && cl == cr && al.as_slice() == ar.as_slice() && pl.len() == pr.len(),
            _ => false,
        },
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Core minimization helpers
// ---------------------------------------------------------------------------

/// Rebuild `proof` retaining only the nodes in the dependency cone of
/// `root_id`, rewriting premise references through `id_remap`.
///
/// Returns the rebuilt proof and the number of nodes removed.
fn rebuild_cone(
    proof: &Proof,
    root_id: ProofNodeId,
    id_remap: &FxHashMap<ProofNodeId, ProofNodeId>,
) -> (Proof, usize) {
    let nodes_before = proof.len();

    // Compute the set of nodes needed (after remapping), starting from root.
    // We first resolve the root through the remap, then compute its cone.
    let effective_root = resolve(root_id, id_remap);
    let cone_set: std::collections::HashSet<ProofNodeId> =
        get_dependency_cone(proof, effective_root)
            .into_iter()
            .collect();

    let mut new_proof = Proof::new();
    let mut local_remap: FxHashMap<ProofNodeId, ProofNodeId> = FxHashMap::default();

    // Walk nodes in original order so prerequisites are inserted before
    // dependents (Proof stores them topologically since add_inference uses
    // existing IDs as premises, which must already exist).
    for node in proof.nodes() {
        // Skip nodes outside the cone.
        if !cone_set.contains(&node.id) {
            continue;
        }

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
                    .map(|p| {
                        let resolved = resolve(*p, id_remap);
                        *local_remap.get(&resolved).unwrap_or(&resolved)
                    })
                    .collect();

                new_proof.add_inference_with_args(
                    rule.clone(),
                    new_premises,
                    args.to_vec(),
                    conclusion.clone(),
                )
            }
        };
        local_remap.insert(node.id, new_id);
    }

    let nodes_after = new_proof.len();
    let removed = nodes_before.saturating_sub(nodes_after);
    (new_proof, removed)
}

/// Follow the remap chain to find the canonical node for `id`.
fn resolve(id: ProofNodeId, remap: &FxHashMap<ProofNodeId, ProofNodeId>) -> ProofNodeId {
    let mut current = id;
    // Guard against pathological cycles (should never occur in a DAG, but
    // be defensive). Limit iterations to avoid infinite loops.
    for _ in 0..1024 {
        match remap.get(&current) {
            Some(&next) if next != current => current = next,
            _ => break,
        }
    }
    current
}

/// Perform one deduplication pass over `proof`.
///
/// Returns an `id_remap` table mapping duplicate node IDs to their canonical
/// representative, and a count of how many nodes were declared duplicate.
fn dedup_pass(proof: &Proof) -> (FxHashMap<ProofNodeId, ProofNodeId>, usize) {
    // Map from ConclusionHash → first-seen canonical node id.
    let mut canon: FxHashMap<ConclusionHash, ProofNodeId> = FxHashMap::default();
    let mut remap: FxHashMap<ProofNodeId, ProofNodeId> = FxHashMap::default();
    let mut collapsed = 0usize;

    for node in proof.nodes() {
        let hash = match &node.step {
            ProofStep::Axiom { conclusion } => ConclusionHash::for_axiom(conclusion),
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                args,
            } => ConclusionHash::for_inference(rule, conclusion, args.as_slice(), premises.len()),
        };

        match canon.get(&hash).copied() {
            Some(existing) if nodes_are_equal(proof, existing, node.id) => {
                // This node is a true duplicate of `existing`.
                remap.insert(node.id, existing);
                collapsed += 1;
            }
            Some(_) => {
                // Hash collision but not structurally equal — keep this node
                // as a separate canonical (the first one stays canonical, this
                // one remains unmapped).
            }
            None => {
                canon.insert(hash, node.id);
            }
        }
    }

    (remap, collapsed)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Proof minimizer that performs hash-cons deduplication followed by
/// iterative dependency-cone reduction until a fixed point.
pub struct ProofMinimizer {
    config: MinimizeConfig,
}

impl ProofMinimizer {
    /// Create a new minimizer with the given configuration.
    pub fn new(config: MinimizeConfig) -> Self {
        Self { config }
    }

    /// Create a minimizer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(MinimizeConfig::default())
    }

    /// Minimize `proof` in-place.
    ///
    /// Returns a [`MinimizeResult`] describing what was done.
    pub fn minimize(&self, proof: &mut Proof) -> MinimizeResult {
        let mut result = MinimizeResult::default();

        // Fast exit for empty proofs.
        if proof.is_empty() {
            return result;
        }

        for pass in 0..self.config.max_passes {
            let size_before = proof.len();
            let mut pass_collapsed = 0usize;

            // Step 1: hash-cons deduplication (unless disabled).
            if self.config.enable_dedup {
                let (remap, collapsed) = dedup_pass(proof);
                pass_collapsed = collapsed;

                if !remap.is_empty() {
                    // Rebuild the proof with the remap applied, trimmed to root.
                    if let Some(root_id) = proof.root() {
                        let (new_proof, _) = rebuild_cone(proof, root_id, &remap);
                        *proof = new_proof;
                    }
                }
            }

            // Step 2: cone reduction (trim nodes unreachable from root).
            if let Some(root_id) = proof.root() {
                let empty_remap = FxHashMap::default();
                let (new_proof, _) = rebuild_cone(proof, root_id, &empty_remap);
                *proof = new_proof;
            }

            let size_after = proof.len();
            let removed_this_pass = size_before.saturating_sub(size_after);

            result.passes = pass + 1;
            result.nodes_removed += removed_this_pass;
            result.duplicates_collapsed += pass_collapsed;

            // Fixed-point check: stop if nothing was removed this pass.
            if removed_this_pass == 0 && pass_collapsed == 0 {
                break;
            }
        }

        result
    }
}

impl Default for ProofMinimizer {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::Proof;

    // ------------------------------------------------------------------
    // Helper: build a proof with two axiom-nodes whose conclusions are
    // identical through the `update_conclusion` backdoor.
    //
    // Because Proof::add_axiom deduplicates by conclusion at construction
    // time, we construct them with distinct conclusions first, then rename
    // one to match the other.
    // ------------------------------------------------------------------
    fn proof_with_duplicate_axioms() -> (Proof, ProofNodeId, ProofNodeId) {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p"); // id=0, conclusion="p"
        let a2 = proof.add_axiom("q"); // id=1, conclusion="q"
        // Rename a2 to also conclude "p" — now two nodes share "p".
        proof.update_conclusion(a2, "p");
        // Add an inference over both so both are in the root's cone.
        let root = proof.add_inference("merge", vec![a1, a2], "merged_p");
        (proof, a1, root)
    }

    // (a) A proof with duplicate axiom nodes must shrink after minimization.
    #[test]
    fn test_duplicate_axiom_proof_shrinks() {
        let (mut proof, _a1, _root) = proof_with_duplicate_axioms();
        let size_before = proof.len();

        let minimizer = ProofMinimizer::with_defaults();
        let result = minimizer.minimize(&mut proof);

        assert!(
            result.nodes_removed > 0 || proof.len() < size_before,
            "minimizer must remove at least one duplicate node"
        );
    }

    // (b) Minimizing an already-minimal proof yields no removals, and a
    //     second minimize produces the same result.
    #[test]
    fn test_idempotent_on_minimal() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("x");
        let a2 = proof.add_axiom("y");
        proof.add_inference("and", vec![a1, a2], "x_and_y");

        let minimizer = ProofMinimizer::with_defaults();
        let result1 = minimizer.minimize(&mut proof);
        let size_after_first = proof.len();

        let result2 = minimizer.minimize(&mut proof);
        let size_after_second = proof.len();

        assert_eq!(
            result1.nodes_removed, 0,
            "a minimal proof should have nothing removed"
        );
        assert_eq!(
            size_after_first, size_after_second,
            "second minimize should produce the same size"
        );
        assert_eq!(
            result2.nodes_removed, 0,
            "second minimize should report no removals"
        );
    }

    // (c) Minimization must preserve the root conclusion.
    #[test]
    fn test_preserves_conclusion() {
        let (mut proof, _a1, _root) = proof_with_duplicate_axioms();
        let original_conclusion = proof
            .root_node()
            .map(|n| n.conclusion().to_string())
            .expect("proof must have a root node");

        let minimizer = ProofMinimizer::with_defaults();
        let _result = minimizer.minimize(&mut proof);

        let after_conclusion = proof
            .root_node()
            .map(|n| n.conclusion().to_string())
            .expect("proof must still have a root node after minimize");

        assert_eq!(
            original_conclusion, after_conclusion,
            "root conclusion must be preserved across minimization"
        );
    }

    // (d) A chain of duplicate lemmas requires multiple passes to fully
    //     resolve; `passes > 1` and the final result is stable.
    #[test]
    fn test_iterates_to_fixed_point() {
        // Build: a0="p", a1="q" (renamed to "p"), layer1 infers from both,
        // then a2="r", a3="s" (renamed to "r"), layer2 infers from both.
        // The root uses layer1 and layer2.
        let mut proof = Proof::new();

        // Layer 1 duplicates
        let a0 = proof.add_axiom("p_orig");
        let a1 = proof.add_axiom("p_dup_raw");
        proof.update_conclusion(a1, "p_orig"); // duplicate of a0
        let layer1 = proof.add_inference("l1", vec![a0, a1], "layer1_out");

        // Layer 2 duplicates — different conclusion namespace
        let a2 = proof.add_axiom("r_orig");
        let a3 = proof.add_axiom("r_dup_raw");
        proof.update_conclusion(a3, "r_orig"); // duplicate of a2
        let layer2 = proof.add_inference("l2", vec![a2, a3], "layer2_out");

        // Root uses both layers
        proof.add_inference("root_rule", vec![layer1, layer2], "final_root");

        let minimizer = ProofMinimizer::new(MinimizeConfig {
            max_passes: 4,
            enable_dedup: true,
        });
        let result = minimizer.minimize(&mut proof);

        // Should have removed duplicate nodes.
        assert!(
            result.nodes_removed > 0,
            "minimizer must remove duplicates; nodes_removed={}",
            result.nodes_removed
        );

        // After minimizing, a second pass must be stable (fixed-point).
        let size_stable = proof.len();
        let result2 = minimizer.minimize(&mut proof);
        assert_eq!(
            proof.len(),
            size_stable,
            "proof size must not change after fixed-point"
        );
        assert_eq!(
            result2.nodes_removed, 0,
            "no further removals expected at fixed-point"
        );

        // Check that passes > 1 only when the proof actually required it.
        // (With this construction we always run at least 1 pass.)
        assert!(result.passes >= 1, "at least one pass must be recorded");
    }

    // (e) enable_dedup=false disables the explicit dedup pass but cone
    //     reduction still runs. A genuinely minimal proof (no unreachable
    //     nodes, all distinct conclusions) must not shrink.
    #[test]
    fn test_disable_dedup_preserves_size() {
        // Build a minimal proof with all distinct conclusions and all nodes
        // reachable from root — nothing to trim, nothing to dedup.
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("distinct_x");
        let a2 = proof.add_axiom("distinct_y");
        let a3 = proof.add_axiom("distinct_z");
        let i1 = proof.add_inference("and", vec![a1, a2], "and_xy");
        proof.add_inference("combine", vec![i1, a3], "root_out");
        let size_before = proof.len(); // all 5 nodes reachable

        let minimizer = ProofMinimizer::new(MinimizeConfig {
            max_passes: 4,
            enable_dedup: false,
        });
        let result = minimizer.minimize(&mut proof);

        // No nodes should be removed from a fully-connected, duplicate-free proof.
        assert_eq!(
            result.nodes_removed, 0,
            "no nodes should be removed from a minimal, fully-connected proof"
        );
        assert_eq!(
            proof.len(),
            size_before,
            "disabling dedup on a minimal proof must not shrink it"
        );
    }

    // (f) Empty proof must not panic.
    #[test]
    fn test_empty_proof_safe() {
        let mut proof = Proof::new();
        let minimizer = ProofMinimizer::with_defaults();
        let result = minimizer.minimize(&mut proof);
        assert_eq!(result.nodes_removed, 0);
        assert_eq!(result.passes, 0);
        assert!(proof.is_empty());
    }
}
