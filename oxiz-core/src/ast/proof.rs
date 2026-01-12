//! Proof representation for UNSAT results
//!
//! A proof represents a derivation of unsatisfiability from the input assertions.
//! Proofs can be represented as resolution trees or natural deduction proofs.

use crate::ast::TermId;
use rustc_hash::FxHashMap;

/// Unique identifier for a proof node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProofId(pub u64);

/// A proof node representing a derivation step
#[derive(Debug, Clone)]
pub struct ProofNode {
    /// Unique ID for this proof node
    pub id: ProofId,
    /// The rule used in this derivation step
    pub rule: ProofRule,
    /// The conclusion (derived formula)
    pub conclusion: TermId,
    /// References to premises (sub-proofs)
    pub premises: Vec<ProofId>,
    /// Optional metadata (e.g., hints, axiom names)
    pub metadata: FxHashMap<String, String>,
}

/// Proof rules for different inference steps
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofRule {
    /// Assumption from input assertions
    Assume {
        /// Name of the assumption
        name: Option<String>,
    },
    /// Boolean resolution
    Resolution {
        /// Resolved literal
        pivot: TermId,
    },
    /// Modus ponens
    ModusPonens,
    /// Boolean tautology
    Tautology,
    /// Arithmetic inequality
    ArithInequality,
    /// Theory lemma
    TheoryLemma {
        /// Theory name (e.g., "LIA", "BV")
        theory: String,
    },
    /// Contradiction derived
    Contradiction,
    /// Rewriting/Simplification
    Rewrite,
    /// Substitution
    Substitution,
    /// Symmetry (a = b ⊢ b = a)
    Symmetry,
    /// Transitivity ((a = b) ∧ (b = c) ⊢ a = c)
    Transitivity,
    /// Congruence (f(a) = f(b) from a = b)
    Congruence,
    /// Reflexivity (⊢ a = a)
    Reflexivity,
    /// Instantiation of quantified formula
    Instantiation {
        /// Terms used for instantiation
        terms: Vec<TermId>,
    },
    /// Other/Custom rule
    Custom {
        /// Rule name
        name: String,
    },
}

/// A proof object representing a complete proof tree
#[derive(Debug, Clone)]
pub struct Proof {
    /// All nodes in the proof
    nodes: FxHashMap<ProofId, ProofNode>,
    /// The root of the proof (typically a contradiction)
    root: ProofId,
    /// Next ID to assign
    next_id: u64,
}

impl Proof {
    /// Create a new empty proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: FxHashMap::default(),
            root: ProofId(0),
            next_id: 0,
        }
    }

    /// Create a proof with a root node
    #[must_use]
    pub fn with_root(root: ProofNode) -> Self {
        let root_id = root.id;
        let mut nodes = FxHashMap::default();
        nodes.insert(root_id, root);
        Self {
            nodes,
            root: root_id,
            next_id: root_id.0 + 1,
        }
    }

    /// Generate a new unique proof ID
    pub fn new_id(&mut self) -> ProofId {
        let id = ProofId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a proof node
    pub fn add_node(&mut self, node: ProofNode) {
        self.nodes.insert(node.id, node);
    }

    /// Get a proof node by ID
    #[must_use]
    pub fn get_node(&self, id: ProofId) -> Option<&ProofNode> {
        self.nodes.get(&id)
    }

    /// Get the root node
    #[must_use]
    pub fn root(&self) -> ProofId {
        self.root
    }

    /// Set the root node
    pub fn set_root(&mut self, root: ProofId) {
        self.root = root;
    }

    /// Get all nodes in the proof
    #[must_use]
    pub fn nodes(&self) -> &FxHashMap<ProofId, ProofNode> {
        &self.nodes
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the number of proof steps
    #[must_use]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Validate the structural integrity of the proof
    ///
    /// Checks that:
    /// - All premise references exist in the proof
    /// - The proof is well-founded (no cycles)
    /// - The root node exists
    pub fn validate_structure(&self) -> Result<(), String> {
        // Check root exists
        if !self.nodes.contains_key(&self.root) {
            return Err(format!("Root node {:?} does not exist in proof", self.root));
        }

        // Check all premise references exist
        for (id, node) in &self.nodes {
            for premise_id in &node.premises {
                if !self.nodes.contains_key(premise_id) {
                    return Err(format!(
                        "Node {:?} references non-existent premise {:?}",
                        id, premise_id
                    ));
                }
            }
        }

        // Check for cycles using DFS
        let mut visited = rustc_hash::FxHashSet::default();
        let mut rec_stack = rustc_hash::FxHashSet::default();

        fn has_cycle(
            node_id: ProofId,
            nodes: &FxHashMap<ProofId, ProofNode>,
            visited: &mut rustc_hash::FxHashSet<ProofId>,
            rec_stack: &mut rustc_hash::FxHashSet<ProofId>,
        ) -> bool {
            if rec_stack.contains(&node_id) {
                return true;
            }
            if visited.contains(&node_id) {
                return false;
            }

            visited.insert(node_id);
            rec_stack.insert(node_id);

            if let Some(node) = nodes.get(&node_id) {
                for &premise_id in &node.premises {
                    if has_cycle(premise_id, nodes, visited, rec_stack) {
                        return true;
                    }
                }
            }

            rec_stack.remove(&node_id);
            false
        }

        for node_id in self.nodes.keys() {
            if has_cycle(*node_id, &self.nodes, &mut visited, &mut rec_stack) {
                return Err(format!(
                    "Cycle detected in proof starting from node {:?}",
                    node_id
                ));
            }
        }

        Ok(())
    }

    /// Get all leaf nodes (nodes with no premises)
    #[must_use]
    pub fn leaves(&self) -> Vec<ProofId> {
        self.nodes
            .iter()
            .filter_map(|(id, node)| {
                if node.premises.is_empty() {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute the height of the proof tree
    #[must_use]
    pub fn height(&self) -> usize {
        fn compute_height(
            node_id: ProofId,
            nodes: &FxHashMap<ProofId, ProofNode>,
            memo: &mut FxHashMap<ProofId, usize>,
        ) -> usize {
            if let Some(&h) = memo.get(&node_id) {
                return h;
            }

            let node = match nodes.get(&node_id) {
                Some(n) => n,
                None => return 0,
            };

            if node.premises.is_empty() {
                memo.insert(node_id, 0);
                return 0;
            }

            let max_premise_height = node
                .premises
                .iter()
                .map(|&p| compute_height(p, nodes, memo))
                .max()
                .unwrap_or(0);

            let height = max_premise_height + 1;
            memo.insert(node_id, height);
            height
        }

        let mut memo = FxHashMap::default();
        compute_height(self.root, &self.nodes, &mut memo)
    }

    /// Remove unreachable nodes (nodes not reachable from root)
    ///
    /// Returns the number of nodes removed
    pub fn prune_unreachable(&mut self) -> usize {
        // Find all reachable nodes via DFS from root
        let mut reachable = rustc_hash::FxHashSet::default();
        let mut stack = vec![self.root];

        while let Some(node_id) = stack.pop() {
            if reachable.insert(node_id)
                && let Some(node) = self.nodes.get(&node_id)
            {
                for &premise_id in &node.premises {
                    stack.push(premise_id);
                }
            }
        }

        // Remove unreachable nodes
        let initial_size = self.nodes.len();
        self.nodes.retain(|id, _| reachable.contains(id));
        initial_size - self.nodes.len()
    }

    /// Remove redundant proof steps
    ///
    /// This simplifies the proof by removing nodes that have only one premise
    /// and perform trivial transformations (like Rewrite with no change).
    /// Returns the number of nodes removed.
    pub fn simplify(&mut self) -> usize {
        let mut removed = 0;
        let mut changed = true;

        while changed {
            changed = false;
            let mut replacements = FxHashMap::default();

            // Find nodes that can be eliminated
            for (node_id, node) in &self.nodes {
                // Skip the root
                if *node_id == self.root {
                    continue;
                }

                // Check if this is a trivial rewrite with one premise
                if node.premises.len() == 1 {
                    match &node.rule {
                        ProofRule::Rewrite | ProofRule::Substitution => {
                            // Can replace this node with its premise
                            replacements.insert(*node_id, node.premises[0]);
                            changed = true;
                        }
                        _ => {}
                    }
                }
            }

            // Apply replacements
            for (old_id, new_id) in &replacements {
                // Update all nodes that reference old_id
                for node in self.nodes.values_mut() {
                    for premise in &mut node.premises {
                        if *premise == *old_id {
                            *premise = *new_id;
                        }
                    }
                }

                // Update root if necessary
                if self.root == *old_id {
                    self.root = *new_id;
                }

                // Remove the old node
                self.nodes.remove(old_id);
                removed += 1;
            }
        }

        removed
    }

    /// Minimize the proof by removing unnecessary assumptions
    ///
    /// This performs a bottom-up analysis to identify which assumptions
    /// (leaf nodes) are actually used in deriving the conclusion.
    /// Returns a new proof containing only the necessary nodes.
    #[must_use]
    pub fn minimize(&self) -> Proof {
        // Start from root and trace back to find all used nodes
        let mut used = rustc_hash::FxHashSet::default();
        let mut stack = vec![self.root];

        while let Some(node_id) = stack.pop() {
            if used.insert(node_id)
                && let Some(node) = self.nodes.get(&node_id)
            {
                for &premise_id in &node.premises {
                    stack.push(premise_id);
                }
            }
        }

        // Build new proof with only used nodes
        let mut minimized = Proof::new();
        minimized.root = self.root;
        minimized.next_id = self.next_id;

        for node_id in &used {
            if let Some(node) = self.nodes.get(node_id) {
                minimized.nodes.insert(*node_id, node.clone());
            }
        }

        minimized
    }

    /// Get statistics about the proof
    #[must_use]
    pub fn statistics(&self) -> ProofStatistics {
        let mut assumptions = 0;
        let mut resolutions = 0;
        let mut theory_lemmas = 0;
        let mut rewrites = 0;
        let mut substitutions = 0;
        let mut congruences = 0;
        let mut instantiations = 0;
        let mut other_rules = 0;

        // Count rule types
        for node in self.nodes.values() {
            match &node.rule {
                ProofRule::Assume { .. } => assumptions += 1,
                ProofRule::Resolution { .. } => resolutions += 1,
                ProofRule::TheoryLemma { .. } => theory_lemmas += 1,
                ProofRule::Rewrite => rewrites += 1,
                ProofRule::Substitution => substitutions += 1,
                ProofRule::Congruence => congruences += 1,
                ProofRule::Instantiation { .. } => instantiations += 1,
                _ => other_rules += 1,
            }
        }

        ProofStatistics {
            total_nodes: self.nodes.len(),
            height: self.height(),
            assumptions,
            resolutions,
            theory_lemmas,
            rewrites,
            substitutions,
            congruences,
            instantiations,
            other_rules,
        }
    }
}

/// Statistics about a proof
#[derive(Debug, Default, Clone)]
pub struct ProofStatistics {
    /// Total number of proof nodes
    pub total_nodes: usize,
    /// Height of the proof tree
    pub height: usize,
    /// Number of assumptions
    pub assumptions: usize,
    /// Number of resolution steps
    pub resolutions: usize,
    /// Number of theory lemmas
    pub theory_lemmas: usize,
    /// Number of rewrites
    pub rewrites: usize,
    /// Number of substitutions
    pub substitutions: usize,
    /// Number of congruence steps
    pub congruences: usize,
    /// Number of quantifier instantiations
    pub instantiations: usize,
    /// Number of other rules
    pub other_rules: usize,
}

impl Default for Proof {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofNode {
    /// Create a new proof node
    #[must_use]
    pub fn new(id: ProofId, rule: ProofRule, conclusion: TermId) -> Self {
        Self {
            id,
            rule,
            conclusion,
            premises: Vec::new(),
            metadata: FxHashMap::default(),
        }
    }

    /// Create a proof node with premises
    #[must_use]
    pub fn with_premises(
        id: ProofId,
        rule: ProofRule,
        conclusion: TermId,
        premises: Vec<ProofId>,
    ) -> Self {
        Self {
            id,
            rule,
            conclusion,
            premises,
            metadata: FxHashMap::default(),
        }
    }

    /// Add a premise to this proof node
    pub fn add_premise(&mut self, premise: ProofId) {
        self.premises.push(premise);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermId;

    #[test]
    fn test_empty_proof() {
        let proof = Proof::new();
        assert!(proof.is_empty());
        assert_eq!(proof.size(), 0);
    }

    #[test]
    fn test_create_proof_node() {
        let node = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        assert_eq!(node.id, ProofId(0));
        assert_eq!(node.conclusion, TermId(1));
        assert!(node.premises.is_empty());
    }

    #[test]
    fn test_proof_with_root() {
        let root_node = ProofNode::new(ProofId(0), ProofRule::Contradiction, TermId(99));
        let proof = Proof::with_root(root_node);

        assert!(!proof.is_empty());
        assert_eq!(proof.size(), 1);
        assert_eq!(proof.root(), ProofId(0));
    }

    #[test]
    fn test_add_proof_nodes() {
        let mut proof = Proof::new();

        let node1 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("A".to_string()),
            },
            TermId(1),
        );
        let node2 = ProofNode::new(ProofId(1), ProofRule::ModusPonens, TermId(2));

        proof.add_node(node1);
        proof.add_node(node2);
        proof.set_root(ProofId(1));

        assert_eq!(proof.size(), 2);
        assert_eq!(proof.root(), ProofId(1));
    }

    #[test]
    fn test_proof_node_with_premises() {
        let node = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: TermId(5) },
            TermId(3),
            vec![ProofId(0), ProofId(1)],
        );

        assert_eq!(node.premises.len(), 2);
        assert_eq!(node.premises[0], ProofId(0));
        assert_eq!(node.premises[1], ProofId(1));
    }

    #[test]
    fn test_proof_metadata() {
        let mut node = ProofNode::new(
            ProofId(0),
            ProofRule::TheoryLemma {
                theory: "LIA".to_string(),
            },
            TermId(1),
        );

        node.add_metadata("source".to_string(), "farkas".to_string());
        node.add_metadata("hint".to_string(), "coefficient_vector".to_string());

        assert_eq!(node.metadata.len(), 2);
        assert_eq!(node.metadata.get("source"), Some(&"farkas".to_string()));
    }

    #[test]
    fn test_validate_structure_valid_proof() {
        let mut proof = Proof::new();

        // Build a simple valid proof tree:
        //       node2 (root)
        //       /    \
        //   node0   node1
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        let node1 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("H2".to_string()),
            },
            TermId(2),
        );
        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: TermId(3) },
            TermId(4),
            vec![ProofId(0), ProofId(1)],
        );

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.set_root(ProofId(2));

        assert!(proof.validate_structure().is_ok());
    }

    #[test]
    fn test_validate_structure_missing_premise() {
        let mut proof = Proof::new();

        // Node references a non-existent premise
        let node = ProofNode::with_premises(
            ProofId(0),
            ProofRule::Resolution { pivot: TermId(1) },
            TermId(2),
            vec![ProofId(99)], // This premise doesn't exist
        );

        proof.add_node(node);
        proof.set_root(ProofId(0));

        assert!(proof.validate_structure().is_err());
    }

    #[test]
    fn test_validate_structure_missing_root() {
        let mut proof = Proof::new();

        let node = ProofNode::new(ProofId(0), ProofRule::Assume { name: None }, TermId(1));

        proof.add_node(node);
        proof.set_root(ProofId(99)); // Non-existent root

        assert!(proof.validate_structure().is_err());
    }

    #[test]
    fn test_validate_structure_cycle() {
        let mut proof = Proof::new();

        // Create a cycle: node0 -> node1 -> node0
        let mut node0 = ProofNode::new(ProofId(0), ProofRule::ModusPonens, TermId(1));
        node0.add_premise(ProofId(1));

        let mut node1 = ProofNode::new(ProofId(1), ProofRule::ModusPonens, TermId(2));
        node1.add_premise(ProofId(0));

        proof.add_node(node0);
        proof.add_node(node1);
        proof.set_root(ProofId(0));

        assert!(proof.validate_structure().is_err());
    }

    #[test]
    fn test_proof_leaves() {
        let mut proof = Proof::new();

        // Build proof tree with leaves at node0 and node1
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        let node1 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("H2".to_string()),
            },
            TermId(2),
        );
        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: TermId(3) },
            TermId(4),
            vec![ProofId(0), ProofId(1)],
        );

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.set_root(ProofId(2));

        let leaves = proof.leaves();
        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&ProofId(0)));
        assert!(leaves.contains(&ProofId(1)));
    }

    #[test]
    fn test_proof_height() {
        let mut proof = Proof::new();

        // Build a proof tree of height 2:
        //         node3 (root)
        //            |
        //         node2
        //         /    \
        //     node0   node1
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        let node1 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("H2".to_string()),
            },
            TermId(2),
        );
        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: TermId(3) },
            TermId(4),
            vec![ProofId(0), ProofId(1)],
        );
        let node3 = ProofNode::with_premises(
            ProofId(3),
            ProofRule::Contradiction,
            TermId(5),
            vec![ProofId(2)],
        );

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.add_node(node3);
        proof.set_root(ProofId(3));

        assert_eq!(proof.height(), 2);
    }

    #[test]
    fn test_proof_height_single_node() {
        let proof = Proof::with_root(ProofNode::new(ProofId(0), ProofRule::Tautology, TermId(1)));

        assert_eq!(proof.height(), 0);
    }

    #[test]
    fn test_prune_unreachable() {
        let mut proof = Proof::new();

        // Build a proof with unreachable nodes
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        let node1 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("H2".to_string()),
            },
            TermId(2),
        );
        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: TermId(3) },
            TermId(4),
            vec![ProofId(0)], // Only uses node0, not node1
        );

        // Add an unreachable node
        let node3 = ProofNode::new(
            ProofId(3),
            ProofRule::Assume {
                name: Some("unused".to_string()),
            },
            TermId(5),
        );

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.add_node(node3);
        proof.set_root(ProofId(2));

        assert_eq!(proof.size(), 4);

        // Prune unreachable nodes
        let removed = proof.prune_unreachable();

        // Should remove node1 and node3 (both unreachable)
        assert_eq!(removed, 2);
        assert_eq!(proof.size(), 2);
        assert!(proof.get_node(ProofId(0)).is_some());
        assert!(proof.get_node(ProofId(1)).is_none());
        assert!(proof.get_node(ProofId(2)).is_some());
        assert!(proof.get_node(ProofId(3)).is_none());
    }

    #[test]
    fn test_simplify_redundant_rewrites() {
        let mut proof = Proof::new();

        // Build a proof with redundant rewrite steps
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );

        // Redundant rewrite (single premise)
        let node1 =
            ProofNode::with_premises(ProofId(1), ProofRule::Rewrite, TermId(2), vec![ProofId(0)]);

        // Another redundant rewrite
        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Substitution,
            TermId(3),
            vec![ProofId(1)],
        );

        // Final resolution
        let node3 = ProofNode::with_premises(
            ProofId(3),
            ProofRule::Contradiction,
            TermId(4),
            vec![ProofId(2)],
        );

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.add_node(node3);
        proof.set_root(ProofId(3));

        assert_eq!(proof.size(), 4);

        // Simplify the proof
        let removed = proof.simplify();

        // Should remove the two redundant rewrites
        assert_eq!(removed, 2);
        assert_eq!(proof.size(), 2);

        // node3 should now directly reference node0
        let node3_simplified = proof.get_node(ProofId(3)).unwrap();
        assert_eq!(node3_simplified.premises, vec![ProofId(0)]);
    }

    #[test]
    fn test_minimize_proof() {
        let mut proof = Proof::new();

        // Build a proof with extra assumptions
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("used".to_string()),
            },
            TermId(1),
        );

        let node1 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("unused".to_string()),
            },
            TermId(2),
        );

        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::ModusPonens,
            TermId(3),
            vec![ProofId(0)], // Only uses node0
        );

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.set_root(ProofId(2));

        assert_eq!(proof.size(), 3);

        // Minimize the proof
        let minimized = proof.minimize();

        // Should only contain node0 and node2
        assert_eq!(minimized.size(), 2);
        assert!(minimized.get_node(ProofId(0)).is_some());
        assert!(minimized.get_node(ProofId(1)).is_none());
        assert!(minimized.get_node(ProofId(2)).is_some());
        assert_eq!(minimized.root(), ProofId(2));
    }

    #[test]
    fn test_proof_statistics() {
        let mut proof = Proof::new();

        // Build a proof with various rule types
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        let node1 = ProofNode::new(
            ProofId(1),
            ProofRule::Assume {
                name: Some("H2".to_string()),
            },
            TermId(2),
        );
        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Resolution { pivot: TermId(3) },
            TermId(4),
            vec![ProofId(0), ProofId(1)],
        );
        let node3 = ProofNode::with_premises(
            ProofId(3),
            ProofRule::TheoryLemma {
                theory: "LIA".to_string(),
            },
            TermId(5),
            vec![ProofId(2)],
        );
        let node4 =
            ProofNode::with_premises(ProofId(4), ProofRule::Rewrite, TermId(6), vec![ProofId(3)]);

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.add_node(node3);
        proof.add_node(node4);
        proof.set_root(ProofId(4));

        let stats = proof.statistics();

        assert_eq!(stats.total_nodes, 5);
        assert_eq!(stats.assumptions, 2);
        assert_eq!(stats.resolutions, 1);
        assert_eq!(stats.theory_lemmas, 1);
        assert_eq!(stats.rewrites, 1);
        assert_eq!(stats.height, 3);
    }

    #[test]
    fn test_simplify_preserves_validity() {
        let mut proof = Proof::new();

        // Build a valid proof with redundant steps
        let node0 = ProofNode::new(
            ProofId(0),
            ProofRule::Assume {
                name: Some("H1".to_string()),
            },
            TermId(1),
        );
        let node1 =
            ProofNode::with_premises(ProofId(1), ProofRule::Rewrite, TermId(2), vec![ProofId(0)]);
        let node2 = ProofNode::with_premises(
            ProofId(2),
            ProofRule::Contradiction,
            TermId(3),
            vec![ProofId(1)],
        );

        proof.add_node(node0);
        proof.add_node(node1);
        proof.add_node(node2);
        proof.set_root(ProofId(2));

        // Validate before simplification
        assert!(proof.validate_structure().is_ok());

        // Simplify
        proof.simplify();

        // Validate after simplification
        assert!(proof.validate_structure().is_ok());
    }

    #[test]
    fn test_minimize_empty_proof() {
        let proof = Proof::new();
        let minimized = proof.minimize();
        assert_eq!(minimized.size(), 0);
    }
}
