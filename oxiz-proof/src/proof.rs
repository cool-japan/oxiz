//! Core proof representation.

use crate::metadata::ProofMetadata;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::fmt;
use std::io::{self, Write};

/// Unique identifier for a proof node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProofNodeId(pub u32);

impl fmt::Display for ProofNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p{}", self.0)
    }
}

/// A proof tree with explicit node tracking.
///
/// This structure maintains a DAG (Directed Acyclic Graph) of proof steps,
/// allowing for efficient premise tracking and proof compression.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Proof {
    /// All nodes in the proof
    nodes: Vec<ProofNode>,
    /// Root nodes (conclusions with no dependents)
    roots: Vec<ProofNodeId>,
    /// Mapping from terms to proof nodes (for deduplication)
    #[cfg_attr(feature = "serde", serde(skip))]
    term_cache: FxHashMap<String, ProofNodeId>,
    /// Metadata for proof nodes
    metadata: FxHashMap<ProofNodeId, ProofMetadata>,
    /// Next available node ID
    next_id: u32,
}

/// A single node in the proof tree
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProofNode {
    /// Unique identifier
    pub id: ProofNodeId,
    /// The proof step
    pub step: ProofStep,
    /// IDs of nodes that depend on this one (optimized for 0-2 dependents)
    pub dependents: SmallVec<[ProofNodeId; 2]>,
    /// Depth in the proof tree (for compression)
    pub depth: u32,
}

/// A single proof step.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ProofStep {
    /// Axiom (no premises)
    Axiom {
        /// The conclusion term
        conclusion: String,
    },
    /// Inference rule application
    Inference {
        /// Rule name
        rule: String,
        /// Premise node IDs (optimized for 1-4 premises)
        premises: SmallVec<[ProofNodeId; 4]>,
        /// The conclusion term
        conclusion: String,
        /// Optional arguments to the rule (optimized for 0-2 args)
        args: SmallVec<[String; 2]>,
    },
}

impl ProofNode {
    /// Get the conclusion of this proof node.
    pub fn conclusion(&self) -> &str {
        match &self.step {
            ProofStep::Axiom { conclusion } => conclusion,
            ProofStep::Inference { conclusion, .. } => conclusion,
        }
    }
}

impl Proof {
    /// Create a new empty proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            term_cache: FxHashMap::default(),
            metadata: FxHashMap::default(),
            next_id: 0,
        }
    }

    /// Allocate a new node ID
    fn alloc_id(&mut self) -> ProofNodeId {
        let id = ProofNodeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add an axiom to the proof
    pub fn add_axiom(&mut self, conclusion: impl Into<String>) -> ProofNodeId {
        let conclusion = conclusion.into();

        // Check cache for deduplication
        if let Some(&existing_id) = self.term_cache.get(&conclusion) {
            return existing_id;
        }

        let id = self.alloc_id();
        let node = ProofNode {
            id,
            step: ProofStep::Axiom {
                conclusion: conclusion.clone(),
            },
            dependents: SmallVec::new(),
            depth: 0,
        };

        self.nodes.push(node);
        self.term_cache.insert(conclusion, id);
        self.roots.push(id);
        id
    }

    /// Add an inference step to the proof
    pub fn add_inference(
        &mut self,
        rule: impl Into<String>,
        premises: Vec<ProofNodeId>,
        conclusion: impl Into<String>,
    ) -> ProofNodeId {
        self.add_inference_with_args(rule, premises, Vec::new(), conclusion)
    }

    /// Add an inference step with arguments
    pub fn add_inference_with_args(
        &mut self,
        rule: impl Into<String>,
        premises: Vec<ProofNodeId>,
        args: Vec<String>,
        conclusion: impl Into<String>,
    ) -> ProofNodeId {
        let conclusion = conclusion.into();
        let rule = rule.into();

        // Check cache for deduplication
        if let Some(&existing_id) = self.term_cache.get(&conclusion) {
            return existing_id;
        }

        let id = self.alloc_id();

        // Calculate depth (max of premise depths + 1)
        let depth = premises
            .iter()
            .filter_map(|p| self.get_node(*p))
            .map(|n| n.depth)
            .max()
            .unwrap_or(0)
            + 1;

        let node = ProofNode {
            id,
            step: ProofStep::Inference {
                rule,
                premises: SmallVec::from_vec(premises.clone()),
                conclusion: conclusion.clone(),
                args: SmallVec::from_vec(args),
            },
            dependents: SmallVec::new(),
            depth,
        };

        // Update premise dependents
        for premise_id in &premises {
            if let Some(premise_node) = self.get_node_mut(*premise_id) {
                premise_node.dependents.push(id);
            }
        }

        self.nodes.push(node);
        self.term_cache.insert(conclusion, id);
        self.roots.push(id);
        id
    }

    /// Get a proof node by ID
    #[must_use]
    pub fn get_node(&self, id: ProofNodeId) -> Option<&ProofNode> {
        self.nodes.get(id.0 as usize)
    }

    /// Get a mutable proof node by ID
    fn get_node_mut(&mut self, id: ProofNodeId) -> Option<&mut ProofNode> {
        self.nodes.get_mut(id.0 as usize)
    }

    /// Update the conclusion of a proof node.
    ///
    /// Returns `true` if the node was found and updated, `false` otherwise.
    /// This is useful for proof simplification and transformation.
    pub fn update_conclusion(
        &mut self,
        id: ProofNodeId,
        new_conclusion: impl Into<String>,
    ) -> bool {
        if let Some(node) = self.get_node_mut(id) {
            let new_conclusion = new_conclusion.into();
            match &mut node.step {
                ProofStep::Axiom { conclusion } => *conclusion = new_conclusion,
                ProofStep::Inference { conclusion, .. } => *conclusion = new_conclusion,
            }
            // Update term cache
            if let Some(old_term) = self
                .term_cache
                .iter()
                .find_map(|(k, v)| if *v == id { Some(k.clone()) } else { None })
            {
                self.term_cache.remove(&old_term);
            }
            true
        } else {
            false
        }
    }

    /// Get all proof nodes
    #[must_use]
    pub fn nodes(&self) -> &[ProofNode] {
        &self.nodes
    }

    /// Get the root nodes
    #[must_use]
    pub fn roots(&self) -> &[ProofNodeId] {
        &self.roots
    }

    /// Get the primary root (last added root)
    #[must_use]
    pub fn root(&self) -> Option<ProofNodeId> {
        self.roots.last().copied()
    }

    /// Get the primary root node
    #[must_use]
    pub fn root_node(&self) -> Option<&ProofNode> {
        self.root().and_then(|id| self.get_node(id))
    }

    /// Get the number of nodes
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Get the maximum depth of the proof tree
    #[must_use]
    pub fn depth(&self) -> u32 {
        self.stats().max_depth
    }

    /// Get the node count (alias for len())
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.len()
    }

    /// Get all leaf nodes (axioms - nodes with no premises)
    #[must_use]
    pub fn leaf_nodes(&self) -> Vec<ProofNodeId> {
        self.nodes
            .iter()
            .filter(|node| matches!(node.step, ProofStep::Axiom { .. }))
            .map(|node| node.id)
            .collect()
    }

    /// Get the premises of a node (if it's an inference)
    #[must_use]
    pub fn premises(&self, node_id: ProofNodeId) -> Option<&[ProofNodeId]> {
        self.get_node(node_id).and_then(|node| {
            if let ProofStep::Inference { premises, .. } = &node.step {
                Some(premises.as_slice())
            } else {
                None
            }
        })
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear the proof
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.roots.clear();
        self.term_cache.clear();
        self.metadata.clear();
        self.next_id = 0;
    }

    /// Get proof statistics
    #[must_use]
    pub fn stats(&self) -> ProofStats {
        let mut stats = ProofStats {
            total_nodes: self.nodes.len(),
            axiom_count: 0,
            inference_count: 0,
            max_depth: 0,
            avg_depth: 0.0,
            root_count: self.roots.len(),
            max_premises: 0,
            avg_premises: 0.0,
            leaf_count: 0,
        };

        let mut total_depth = 0u64;
        let mut total_premises = 0usize;

        for node in &self.nodes {
            match &node.step {
                ProofStep::Axiom { .. } => stats.axiom_count += 1,
                ProofStep::Inference { premises, .. } => {
                    stats.inference_count += 1;
                    let premise_count = premises.len();
                    stats.max_premises = stats.max_premises.max(premise_count);
                    total_premises += premise_count;
                }
            }
            stats.max_depth = stats.max_depth.max(node.depth);
            total_depth += u64::from(node.depth);

            // Count leaves (nodes with no dependents)
            if node.dependents.is_empty() {
                stats.leaf_count += 1;
            }
        }

        if !self.nodes.is_empty() {
            stats.avg_depth = total_depth as f64 / self.nodes.len() as f64;
        }

        if stats.inference_count > 0 {
            stats.avg_premises = total_premises as f64 / stats.inference_count as f64;
        }

        stats
    }

    /// Write the proof in a human-readable format
    pub fn write<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "; Proof with {} nodes", self.len())?;
        writeln!(writer)?;

        for node in &self.nodes {
            match &node.step {
                ProofStep::Axiom { conclusion } => {
                    writeln!(writer, "{}: (axiom {})", node.id, conclusion)?;
                }
                ProofStep::Inference {
                    rule,
                    premises,
                    conclusion,
                    args,
                } => {
                    write!(writer, "{}: ({} [", node.id, rule)?;
                    for (i, p) in premises.iter().enumerate() {
                        if i > 0 {
                            write!(writer, ", ")?;
                        }
                        write!(writer, "{}", p)?;
                    }
                    write!(writer, "]")?;
                    if !args.is_empty() {
                        write!(writer, " :args [")?;
                        for (i, arg) in args.iter().enumerate() {
                            if i > 0 {
                                write!(writer, ", ")?;
                            }
                            write!(writer, "{}", arg)?;
                        }
                        write!(writer, "]")?;
                    }
                    writeln!(writer, " => {})", conclusion)?;
                }
            }
        }

        Ok(())
    }

    /// Convert to string
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        let mut buf = Vec::new();
        self.write(&mut buf)
            .expect("writing to Vec should not fail");
        String::from_utf8(buf).expect("proof output is UTF-8")
    }

    /// Add multiple axioms at once and return their IDs
    pub fn add_axioms(
        &mut self,
        conclusions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Vec<ProofNodeId> {
        conclusions.into_iter().map(|c| self.add_axiom(c)).collect()
    }

    /// Find all nodes with a specific rule
    #[must_use]
    pub fn find_nodes_by_rule(&self, rule: &str) -> Vec<ProofNodeId> {
        self.nodes
            .iter()
            .filter(|node| matches!(&node.step, ProofStep::Inference { rule: r, .. } if r == rule))
            .map(|node| node.id)
            .collect()
    }

    /// Find all nodes with a specific conclusion
    #[must_use]
    pub fn find_nodes_by_conclusion(&self, conclusion: &str) -> Vec<ProofNodeId> {
        self.nodes
            .iter()
            .filter(|node| match &node.step {
                ProofStep::Axiom { conclusion: c } => c == conclusion,
                ProofStep::Inference { conclusion: c, .. } => c == conclusion,
            })
            .map(|node| node.id)
            .collect()
    }

    /// Get all direct children (premises) of a node
    #[must_use]
    pub fn get_children(&self, node_id: ProofNodeId) -> Vec<ProofNodeId> {
        self.premises(node_id)
            .map(|p| p.to_vec())
            .unwrap_or_default()
    }

    /// Get all direct parents (dependents) of a node
    #[must_use]
    pub fn get_parents(&self, node_id: ProofNodeId) -> Vec<ProofNodeId> {
        self.get_node(node_id)
            .map(|node| node.dependents.to_vec())
            .unwrap_or_default()
    }

    /// Check if one node is an ancestor of another
    #[must_use]
    pub fn is_ancestor(&self, ancestor: ProofNodeId, descendant: ProofNodeId) -> bool {
        if ancestor == descendant {
            return false;
        }

        let mut visited = rustc_hash::FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(descendant);

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }

            if current == ancestor {
                return true;
            }

            if let Some(premises) = self.premises(current) {
                for &premise in premises {
                    queue.push_back(premise);
                }
            }
        }

        false
    }

    /// Get all ancestors of a node (all nodes it depends on)
    #[must_use]
    pub fn get_all_ancestors(&self, node_id: ProofNodeId) -> Vec<ProofNodeId> {
        let mut ancestors = Vec::new();
        let mut visited = rustc_hash::FxHashSet::default();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(node_id);

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }

            if current != node_id {
                ancestors.push(current);
            }

            if let Some(premises) = self.premises(current) {
                for &premise in premises {
                    queue.push_back(premise);
                }
            }
        }

        ancestors
    }

    /// Count nodes of a specific rule type
    #[must_use]
    pub fn count_rule(&self, rule: &str) -> usize {
        self.nodes
            .iter()
            .filter(|node| matches!(&node.step, ProofStep::Inference { rule: r, .. } if r == rule))
            .count()
    }

    /// Rebuild the internal term cache (useful after deserialization)
    #[cfg(feature = "serde")]
    pub fn rebuild_cache(&mut self) {
        self.term_cache.clear();
        for node in &self.nodes {
            let conclusion = match &node.step {
                ProofStep::Axiom { conclusion } => conclusion,
                ProofStep::Inference { conclusion, .. } => conclusion,
            };
            self.term_cache.insert(conclusion.clone(), node.id);
        }
    }

    // Metadata API

    /// Set metadata for a proof node.
    pub fn set_metadata(&mut self, node_id: ProofNodeId, metadata: ProofMetadata) {
        self.metadata.insert(node_id, metadata);
    }

    /// Get metadata for a proof node.
    #[must_use]
    pub fn get_metadata(&self, node_id: ProofNodeId) -> Option<&ProofMetadata> {
        self.metadata.get(&node_id)
    }

    /// Get mutable metadata for a proof node.
    pub fn get_metadata_mut(&mut self, node_id: ProofNodeId) -> Option<&mut ProofMetadata> {
        self.metadata.get_mut(&node_id)
    }

    /// Remove metadata for a proof node.
    pub fn remove_metadata(&mut self, node_id: ProofNodeId) -> Option<ProofMetadata> {
        self.metadata.remove(&node_id)
    }

    /// Check if a node has metadata.
    #[must_use]
    pub fn has_metadata(&self, node_id: ProofNodeId) -> bool {
        self.metadata.contains_key(&node_id)
    }

    /// Get all nodes with a specific tag.
    #[must_use]
    pub fn nodes_with_tag(&self, tag: &str) -> Vec<ProofNodeId> {
        self.metadata
            .iter()
            .filter(|(_, meta)| meta.has_tag(tag))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all nodes with a specific priority.
    #[must_use]
    pub fn nodes_with_priority(&self, priority: crate::metadata::Priority) -> Vec<ProofNodeId> {
        self.metadata
            .iter()
            .filter(|(_, meta)| meta.priority() == priority)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all nodes with a specific difficulty.
    #[must_use]
    pub fn nodes_with_difficulty(
        &self,
        difficulty: crate::metadata::Difficulty,
    ) -> Vec<ProofNodeId> {
        self.metadata
            .iter()
            .filter(|(_, meta)| meta.difficulty() == difficulty)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all nodes with a specific strategy.
    #[must_use]
    pub fn nodes_with_strategy(&self, strategy: crate::metadata::Strategy) -> Vec<ProofNodeId> {
        self.metadata
            .iter()
            .filter(|(_, meta)| meta.has_strategy(strategy))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get or create metadata for a node.
    pub fn get_or_create_metadata(&mut self, node_id: ProofNodeId) -> &mut ProofMetadata {
        self.metadata.entry(node_id).or_default()
    }
}

impl Default for Proof {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a proof
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProofStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of axiom nodes
    pub axiom_count: usize,
    /// Number of inference nodes
    pub inference_count: usize,
    /// Maximum depth in the proof tree
    pub max_depth: u32,
    /// Average depth
    pub avg_depth: f64,
    /// Number of root nodes
    pub root_count: usize,
    /// Maximum number of premises for a single inference
    pub max_premises: usize,
    /// Average number of premises per inference
    pub avg_premises: f64,
    /// Number of nodes with no dependents (leaves)
    pub leaf_count: usize,
}

impl fmt::Display for ProofStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Proof Statistics:")?;
        writeln!(f, "  Total nodes: {}", self.total_nodes)?;
        writeln!(f, "  Axioms: {}", self.axiom_count)?;
        writeln!(f, "  Inferences: {}", self.inference_count)?;
        writeln!(f, "  Roots: {}", self.root_count)?;
        writeln!(f, "  Leaves: {}", self.leaf_count)?;
        writeln!(f, "  Max depth: {}", self.max_depth)?;
        writeln!(f, "  Avg depth: {:.2}", self.avg_depth)?;
        writeln!(f, "  Max premises: {}", self.max_premises)?;
        writeln!(f, "  Avg premises: {:.2}", self.avg_premises)
    }
}

impl fmt::Display for ProofNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.id, self.step)
    }
}

impl fmt::Display for ProofStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProofStep::Axiom { conclusion } => write!(f, "(axiom {})", conclusion),
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                args,
            } => {
                write!(f, "({} [", rule)?;
                for (i, p) in premises.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "]")?;
                if !args.is_empty() {
                    write!(f, " :args {:?}", args)?;
                }
                write!(f, " => {})", conclusion)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_creation() {
        let proof = Proof::new();
        assert!(proof.is_empty());
        assert_eq!(proof.len(), 0);
    }

    #[test]
    fn test_add_axiom() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("p");

        assert_eq!(proof.len(), 1);
        let node = proof.get_node(id).unwrap();
        assert_eq!(node.id, ProofNodeId(0));
        assert_eq!(node.depth, 0);
    }

    #[test]
    fn test_add_inference() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");

        assert_eq!(proof.len(), 3);
        let node = proof.get_node(i1).unwrap();
        assert_eq!(node.depth, 1);
    }

    #[test]
    fn test_proof_deduplication() {
        let mut proof = Proof::new();
        let id1 = proof.add_axiom("p");
        let id2 = proof.add_axiom("p"); // Same conclusion

        assert_eq!(id1, id2);
        assert_eq!(proof.len(), 1); // Only one node created
    }

    #[test]
    fn test_proof_depth() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let i2 = proof.add_inference("not", vec![i1], "~(p /\\ q)");

        let node = proof.get_node(i2).unwrap();
        assert_eq!(node.depth, 2);
    }

    #[test]
    fn test_proof_stats() {
        let mut proof = Proof::new();
        proof.add_axiom("p");
        proof.add_axiom("q");
        let a1 = proof.add_axiom("r");
        let a2 = proof.add_axiom("s");
        proof.add_inference("and", vec![a1, a2], "r /\\ s");

        let stats = proof.stats();
        assert_eq!(stats.total_nodes, 5);
        assert_eq!(stats.axiom_count, 4);
        assert_eq!(stats.inference_count, 1);
        assert_eq!(stats.max_depth, 1);
    }

    #[test]
    fn test_proof_clear() {
        let mut proof = Proof::new();
        proof.add_axiom("p");
        proof.add_axiom("q");

        assert_eq!(proof.len(), 2);
        proof.clear();
        assert!(proof.is_empty());
    }

    #[test]
    fn test_proof_display() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        proof.add_inference("not", vec![a1], "~p");

        let output = proof.to_string_repr();
        assert!(output.contains("axiom"));
        assert!(output.contains("not"));
    }

    #[test]
    fn test_dependent_tracking() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");

        let node_a1 = proof.get_node(a1).unwrap();
        assert!(node_a1.dependents.contains(&i1));

        let node_a2 = proof.get_node(a2).unwrap();
        assert!(node_a2.dependents.contains(&i1));
    }

    // Metadata tests
    #[test]
    fn test_metadata_set_get() {
        use crate::metadata::{Difficulty, Priority};

        let mut proof = Proof::new();
        let id = proof.add_axiom("p");

        let meta = ProofMetadata::new()
            .with_priority(Priority::High)
            .with_difficulty(Difficulty::Easy);

        proof.set_metadata(id, meta);

        let retrieved = proof.get_metadata(id).unwrap();
        assert_eq!(retrieved.priority(), Priority::High);
        assert_eq!(retrieved.difficulty(), Difficulty::Easy);
    }

    #[test]
    fn test_metadata_tags() {
        let mut proof = Proof::new();
        let id1 = proof.add_axiom("p");
        let id2 = proof.add_axiom("q");

        proof.set_metadata(id1, ProofMetadata::new().with_tag("important"));
        proof.set_metadata(id2, ProofMetadata::new().with_tag("trivial"));

        let important_nodes = proof.nodes_with_tag("important");
        assert_eq!(important_nodes.len(), 1);
        assert_eq!(important_nodes[0], id1);
    }

    #[test]
    fn test_metadata_priority_filter() {
        use crate::metadata::Priority;

        let mut proof = Proof::new();
        let id1 = proof.add_axiom("p");
        let id2 = proof.add_axiom("q");
        let id3 = proof.add_axiom("r");

        proof.set_metadata(id1, ProofMetadata::new().with_priority(Priority::High));
        proof.set_metadata(id2, ProofMetadata::new().with_priority(Priority::High));
        proof.set_metadata(id3, ProofMetadata::new().with_priority(Priority::Low));

        let high_priority = proof.nodes_with_priority(Priority::High);
        assert_eq!(high_priority.len(), 2);
    }

    #[test]
    fn test_metadata_strategy_filter() {
        use crate::metadata::Strategy;

        let mut proof = Proof::new();
        let id1 = proof.add_axiom("p");
        let id2 = proof.add_axiom("q");

        proof.set_metadata(
            id1,
            ProofMetadata::new().with_strategy(Strategy::Resolution),
        );
        proof.set_metadata(id2, ProofMetadata::new().with_strategy(Strategy::Theory));

        let resolution_nodes = proof.nodes_with_strategy(Strategy::Resolution);
        assert_eq!(resolution_nodes.len(), 1);
        assert_eq!(resolution_nodes[0], id1);
    }

    #[test]
    fn test_metadata_remove() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("p");

        proof.set_metadata(id, ProofMetadata::new().with_tag("temp"));
        assert!(proof.has_metadata(id));

        proof.remove_metadata(id);
        assert!(!proof.has_metadata(id));
    }

    #[test]
    fn test_metadata_get_or_create() {
        use crate::metadata::Priority;

        let mut proof = Proof::new();
        let id = proof.add_axiom("p");

        let meta = proof.get_or_create_metadata(id);
        meta.set_priority(Priority::VeryHigh);

        let retrieved = proof.get_metadata(id).unwrap();
        assert_eq!(retrieved.priority(), Priority::VeryHigh);
    }

    #[test]
    fn test_metadata_clear() {
        let mut proof = Proof::new();
        let id = proof.add_axiom("p");
        proof.set_metadata(id, ProofMetadata::new().with_tag("test"));

        proof.clear();
        assert!(!proof.has_metadata(id));
    }

    // Property-based tests
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        // Strategy for generating variable names
        fn var_name() -> impl Strategy<Value = String> {
            "[a-z][0-9]*".prop_map(|s| s.to_string())
        }

        // Strategy for generating rule names
        fn rule_name() -> impl Strategy<Value = String> {
            prop_oneof![
                Just("and".to_string()),
                Just("or".to_string()),
                Just("not".to_string()),
                Just("imp".to_string()),
                Just("resolution".to_string()),
            ]
        }

        proptest! {
            /// Adding axioms should never fail and always increase size
            #[test]
            fn prop_axiom_increases_size(conclusion in var_name()) {
                let mut proof = Proof::new();
                let initial_len = proof.len();
                proof.add_axiom(&conclusion);
                prop_assert!(proof.len() > initial_len || proof.len() == 1);
            }

            /// Axiom depth is always 0
            #[test]
            fn prop_axiom_depth_zero(conclusion in var_name()) {
                let mut proof = Proof::new();
                let id = proof.add_axiom(&conclusion);
                let node = proof.get_node(id).unwrap();
                prop_assert_eq!(node.depth, 0);
            }

            /// Deduplication: Adding same axiom twice gives same ID
            #[test]
            fn prop_axiom_deduplication(conclusion in var_name()) {
                let mut proof = Proof::new();
                let id1 = proof.add_axiom(&conclusion);
                let id2 = proof.add_axiom(&conclusion);
                prop_assert_eq!(id1, id2);
                prop_assert_eq!(proof.len(), 1);
            }

            /// Inference depth is always greater than max premise depth
            #[test]
            fn prop_inference_depth(
                rule in rule_name(),
                conclusions in prop::collection::vec(var_name(), 1..5)
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                for conclusion in &conclusions {
                    let id = proof.add_axiom(conclusion);
                    axiom_ids.push(id);
                }

                if !axiom_ids.is_empty() {
                    let inf_id = proof.add_inference(
                        &rule,
                        axiom_ids.clone(),
                        format!("({} {})", rule, conclusions.join(" "))
                    );
                    let inf_node = proof.get_node(inf_id).unwrap();

                    for &premise_id in &axiom_ids {
                        let premise_node = proof.get_node(premise_id).unwrap();
                        prop_assert!(inf_node.depth > premise_node.depth);
                    }
                }
            }

            /// Stats should be consistent with actual proof structure
            #[test]
            fn prop_stats_consistency(
                axiom_count in 1..10_usize,
                inference_count in 0..5_usize
            ) {
                let mut proof = Proof::new();
                let mut axiom_ids = Vec::new();

                // Add axioms
                for i in 0..axiom_count {
                    let id = proof.add_axiom(format!("p{}", i));
                    axiom_ids.push(id);
                }

                // Add inferences
                for i in 0..inference_count {
                    if axiom_ids.len() >= 2 {
                        let premises = vec![axiom_ids[0], axiom_ids[1]];
                        let id = proof.add_inference(
                            "and",
                            premises,
                            format!("q{}", i)
                        );
                        axiom_ids.push(id);
                    }
                }

                let stats = proof.stats();
                prop_assert_eq!(stats.axiom_count, axiom_count);
                prop_assert!(stats.inference_count <= inference_count);
                prop_assert_eq!(stats.total_nodes, stats.axiom_count + stats.inference_count);
            }

            /// Node count equals axioms plus inferences
            #[test]
            fn prop_node_count_invariant(
                axioms in prop::collection::vec(var_name(), 0..10)
            ) {
                let mut proof = Proof::new();
                for axiom in &axioms {
                    proof.add_axiom(axiom);
                }

                let stats = proof.stats();
                prop_assert_eq!(proof.len(), stats.total_nodes);
                prop_assert_eq!(stats.total_nodes, stats.axiom_count + stats.inference_count);
            }

            /// All nodes should be reachable
            #[test]
            fn prop_all_nodes_reachable(
                conclusions in prop::collection::vec(var_name(), 1..8)
            ) {
                let mut proof = Proof::new();
                for conclusion in &conclusions {
                    proof.add_axiom(conclusion);
                }

                for i in 0..proof.len() {
                    let id = ProofNodeId(i as u32);
                    prop_assert!(proof.get_node(id).is_some());
                }
            }

            /// Clearing proof resets everything
            #[test]
            fn prop_clear_resets(
                conclusions in prop::collection::vec(var_name(), 1..10)
            ) {
                let mut proof = Proof::new();
                for conclusion in &conclusions {
                    proof.add_axiom(conclusion);
                }

                proof.clear();
                prop_assert!(proof.is_empty());
                prop_assert_eq!(proof.len(), 0);
                prop_assert_eq!(proof.next_id, 0);
            }

            /// Max depth is at least as large as any individual node depth
            #[test]
            fn prop_max_depth_bound(
                conclusions in prop::collection::vec(var_name(), 1..8)
            ) {
                let mut proof = Proof::new();
                let mut ids = Vec::new();

                for conclusion in &conclusions {
                    ids.push(proof.add_axiom(conclusion));
                }

                // Add some inferences
                if ids.len() >= 2 {
                    let id = proof.add_inference("and", vec![ids[0], ids[1]], "result");
                    ids.push(id);
                }

                let stats = proof.stats();
                for id in &ids {
                    if let Some(node) = proof.get_node(*id) {
                        prop_assert!(stats.max_depth >= node.depth);
                    }
                }
            }

            /// Premise IDs must exist in the proof
            #[test]
            fn prop_premise_validity(
                rule in rule_name(),
                conclusions in prop::collection::vec(var_name(), 2..6)
            ) {
                let mut proof = Proof::new();
                let mut premise_ids = Vec::new();

                for conclusion in &conclusions {
                    premise_ids.push(proof.add_axiom(conclusion));
                }

                if premise_ids.len() >= 2 {
                    let inf_id = proof.add_inference(
                        &rule,
                        premise_ids.clone(),
                        "result"
                    );

                    if let Some(premises) = proof.premises(inf_id) {
                        for &premise_id in premises {
                            prop_assert!(proof.get_node(premise_id).is_some());
                        }
                    }
                }
            }
        }
    }
}
