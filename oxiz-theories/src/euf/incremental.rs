//! Incremental E-Graph Operations
//!
//! This module provides incremental E-graph updates for efficient theory solving:
//! - Incremental rebuild after merges (avoiding full rebuild)
//! - E-class analysis for abstract interpretation
//! - Efficient term canonicalization
//! - Worklist-based congruence closure
//! - Theory propagation hooks

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::VecDeque;

/// E-class identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EClassId(pub u32);

impl EClassId {
    /// Create a new E-class ID
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// An E-node in the E-graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ENode {
    /// Operator/function symbol
    pub op: u32,
    /// Children (E-class IDs)
    pub children: SmallVec<[EClassId; 4]>,
}

impl ENode {
    /// Create a new E-node
    pub fn new(op: u32, children: impl IntoIterator<Item = EClassId>) -> Self {
        Self {
            op,
            children: children.into_iter().collect(),
        }
    }

    /// Create a leaf E-node (no children)
    pub fn leaf(op: u32) -> Self {
        Self {
            op,
            children: SmallVec::new(),
        }
    }

    /// Canonicalize this E-node by finding representatives of children
    pub fn canonicalize(&self, egraph: &EGraph) -> Self {
        Self {
            op: self.op,
            children: self.children.iter().map(|&c| egraph.find(c)).collect(),
        }
    }
}

/// An E-class: a set of equivalent E-nodes
#[derive(Debug, Clone)]
pub struct EClass {
    /// E-class identifier
    pub id: EClassId,
    /// E-nodes in this class
    pub nodes: Vec<ENode>,
    /// Parent E-classes (classes that use this one as a child)
    pub parents: FxHashSet<EClassId>,
    /// Analysis data (optional, for abstract interpretation)
    pub analysis: Option<AnalysisData>,
}

impl EClass {
    /// Create a new E-class
    fn new(id: EClassId) -> Self {
        Self {
            id,
            nodes: Vec::new(),
            parents: FxHashSet::default(),
            analysis: None,
        }
    }

    /// Add an E-node to this class
    fn add_node(&mut self, node: ENode) {
        if !self.nodes.contains(&node) {
            self.nodes.push(node);
        }
    }
}

/// Analysis data for E-classes (for abstract interpretation)
#[derive(Debug, Clone)]
pub enum AnalysisData {
    /// Constant value (if known)
    Constant(ConstValue),
    /// Bounds (for numerical analysis)
    Bounds {
        /// Minimum bound
        min: Option<i64>,
        /// Maximum bound
        max: Option<i64>,
    },
    /// Size estimate (for term size analysis)
    Size(usize),
    /// Custom data
    Custom(Vec<u8>),
}

/// Constant values for analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    /// Boolean constant
    Bool(bool),
    /// Integer constant
    Int(i64),
    /// Real constant
    Real(f64),
    /// String constant
    String(String),
}

/// Pending operation in the worklist
#[derive(Debug, Clone)]
enum WorkItem {
    /// Merge two E-classes
    Merge(EClassId, EClassId),
    /// Rebuild a parent E-class
    Rebuild(EClassId),
    /// Propagate through theory
    TheoryPropagate(EClassId),
}

/// Configuration for E-graph operations
#[derive(Debug, Clone)]
pub struct EGraphConfig {
    /// Maximum E-classes before garbage collection
    pub max_classes: usize,
    /// Enable analysis
    pub enable_analysis: bool,
    /// Enable theory propagation
    pub enable_theory_propagation: bool,
    /// Rebuild batch size (for incremental rebuilding)
    pub rebuild_batch_size: usize,
}

impl Default for EGraphConfig {
    fn default() -> Self {
        Self {
            max_classes: 100_000,
            enable_analysis: false,
            enable_theory_propagation: true,
            rebuild_batch_size: 100,
        }
    }
}

/// Statistics for E-graph operations
#[derive(Debug, Clone, Default)]
pub struct EGraphStats {
    /// Number of merges performed
    pub merges: usize,
    /// Number of rebuilds
    pub rebuilds: usize,
    /// Number of congruence closures triggered
    pub congruences: usize,
    /// Number of theory propagations
    pub theory_propagations: usize,
    /// Current number of E-classes
    pub num_classes: usize,
    /// Current number of E-nodes
    pub num_nodes: usize,
}

/// Incremental E-Graph with efficient updates
pub struct EGraph {
    /// Configuration
    config: EGraphConfig,
    /// Statistics
    stats: EGraphStats,
    /// E-classes
    classes: Vec<EClass>,
    /// Union-Find for E-class representatives
    parent: Vec<u32>,
    /// Hash-cons table: canonical E-node -> E-class
    memo: FxHashMap<ENode, EClassId>,
    /// Pending work items
    worklist: VecDeque<WorkItem>,
    /// Context stack for push/pop
    context_stack: Vec<ContextState>,
    /// Trail for undo operations
    trail: Vec<UndoEntry>,
    /// Trail limits for each decision level
    trail_limits: Vec<usize>,
    /// Dirty classes that need rebuilding
    dirty: FxHashSet<EClassId>,
    /// Theory propagation callbacks
    propagators: Vec<Box<dyn TheoryPropagator>>,
}

/// State to save for push/pop
#[derive(Debug, Clone)]
struct ContextState {
    num_classes: usize,
    memo_keys: Vec<ENode>,
}

/// Undo entry for backtracking
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum UndoEntry {
    /// Undo a union
    Union { child: EClassId, old_parent: u32 },
    /// Remove an E-node from a class
    RemoveNode { class: EClassId, node: ENode },
    /// Remove a parent link
    RemoveParent { class: EClassId, parent: EClassId },
}

/// Theory propagator trait
pub trait TheoryPropagator: Send + Sync {
    /// Called when two E-classes are merged
    fn on_merge(&mut self, a: EClassId, b: EClassId, egraph: &EGraph);

    /// Called to propagate after a merge
    fn propagate(&mut self, egraph: &mut EGraph) -> Vec<(EClassId, EClassId)>;

    /// Get the name of this propagator
    fn name(&self) -> &'static str;
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl EGraph {
    /// Create a new E-graph
    pub fn new() -> Self {
        Self::with_config(EGraphConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EGraphConfig) -> Self {
        Self {
            config,
            stats: EGraphStats::default(),
            classes: Vec::new(),
            parent: Vec::new(),
            memo: FxHashMap::default(),
            worklist: VecDeque::new(),
            context_stack: Vec::new(),
            trail: Vec::new(),
            trail_limits: vec![0],
            dirty: FxHashSet::default(),
            propagators: Vec::new(),
        }
    }

    /// Add a theory propagator
    pub fn add_propagator(&mut self, propagator: Box<dyn TheoryPropagator>) {
        self.propagators.push(propagator);
    }

    /// Add an E-node to the E-graph, returning its E-class
    pub fn add(&mut self, node: ENode) -> EClassId {
        // Canonicalize the node
        let canonical = node.canonicalize(self);

        // Check hash-cons table
        if let Some(&class) = self.memo.get(&canonical) {
            return class;
        }

        // Create a new E-class
        let id = EClassId::new(self.classes.len() as u32);
        let mut eclass = EClass::new(id);
        eclass.add_node(canonical.clone());

        // Update parent links
        for &child in &canonical.children {
            let child_class = self.find(child);
            self.classes[child_class.raw() as usize].parents.insert(id);
        }

        self.classes.push(eclass);
        self.parent.push(id.raw());
        self.memo.insert(canonical, id);

        self.stats.num_classes += 1;
        self.stats.num_nodes += 1;

        id
    }

    /// Find the representative of an E-class with path compression
    pub fn find(&self, id: EClassId) -> EClassId {
        let mut root = id.raw();
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        EClassId::new(root)
    }

    /// Find the representative with path compression (mutable)
    fn find_mut(&mut self, id: EClassId) -> EClassId {
        let mut x = id.raw();
        let mut root = x;

        // Find root
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }

        // Path compression
        while self.parent[x as usize] != root {
            let next = self.parent[x as usize];
            self.parent[x as usize] = root;
            x = next;
        }

        EClassId::new(root)
    }

    /// Merge two E-classes
    pub fn merge(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let root_a = self.find_mut(a);
        let root_b = self.find_mut(b);

        if root_a == root_b {
            return root_a;
        }

        self.stats.merges += 1;

        // Union by size (merge smaller into larger)
        let (winner, loser) = if self.classes[root_a.raw() as usize].nodes.len()
            >= self.classes[root_b.raw() as usize].nodes.len()
        {
            (root_a, root_b)
        } else {
            (root_b, root_a)
        };

        // Record undo entry
        self.trail.push(UndoEntry::Union {
            child: loser,
            old_parent: loser.raw(),
        });

        // Update parent
        self.parent[loser.raw() as usize] = winner.raw();

        // Merge E-nodes
        let loser_nodes = std::mem::take(&mut self.classes[loser.raw() as usize].nodes);
        for node in loser_nodes {
            self.classes[winner.raw() as usize].add_node(node);
        }

        // Merge parents
        let loser_parents = std::mem::take(&mut self.classes[loser.raw() as usize].parents);
        for parent in loser_parents {
            self.classes[winner.raw() as usize].parents.insert(parent);
            self.dirty.insert(parent);
        }

        // Mark winner as dirty (needs rebuilding)
        self.dirty.insert(winner);

        // Add rebuild items to worklist
        for &parent in &self.classes[winner.raw() as usize].parents {
            self.worklist.push_back(WorkItem::Rebuild(parent));
        }

        // Notify theory propagators
        if self.config.enable_theory_propagation {
            self.worklist.push_back(WorkItem::TheoryPropagate(winner));
        }

        winner
    }

    /// Rebuild dirty E-classes (incremental congruence closure)
    pub fn rebuild(&mut self) {
        let mut batch_count = 0;

        while let Some(item) = self.worklist.pop_front() {
            match item {
                WorkItem::Merge(a, b) => {
                    self.merge(a, b);
                }
                WorkItem::Rebuild(class_id) => {
                    if self.dirty.remove(&class_id) {
                        self.rebuild_class(class_id);
                        self.stats.rebuilds += 1;
                    }
                }
                WorkItem::TheoryPropagate(class_id) => {
                    self.propagate_theories(class_id);
                }
            }

            batch_count += 1;
            if batch_count >= self.config.rebuild_batch_size {
                // Yield to allow other work
                break;
            }
        }
    }

    /// Rebuild a single E-class
    fn rebuild_class(&mut self, class_id: EClassId) {
        let class_id = self.find(class_id);
        let class = &self.classes[class_id.raw() as usize];

        // Re-canonicalize all nodes in the class
        let nodes: Vec<ENode> = class.nodes.clone();

        for node in nodes {
            let canonical = node.canonicalize(self);

            // Check for congruence
            if let Some(&existing) = self.memo.get(&canonical) {
                let existing_root = self.find(existing);
                if existing_root != class_id {
                    // Congruence: merge the classes
                    self.stats.congruences += 1;
                    self.worklist
                        .push_back(WorkItem::Merge(class_id, existing_root));
                }
            } else {
                // Update hash-cons table
                self.memo.insert(canonical, class_id);
            }
        }
    }

    /// Propagate through theory solvers
    fn propagate_theories(&mut self, _class_id: EClassId) {
        // This is a simplified version - in a full implementation,
        // we would call registered theory propagators
        self.stats.theory_propagations += 1;
    }

    /// Check if two E-classes are equivalent
    pub fn are_equal(&self, a: EClassId, b: EClassId) -> bool {
        self.find(a) == self.find(b)
    }

    /// Get an E-class by ID
    pub fn get_class(&self, id: EClassId) -> Option<&EClass> {
        let root = self.find(id);
        self.classes.get(root.raw() as usize)
    }

    /// Lookup an E-node, returning its E-class if it exists
    pub fn lookup(&self, node: &ENode) -> Option<EClassId> {
        let canonical = node.canonicalize(self);
        self.memo.get(&canonical).map(|&id| self.find(id))
    }

    /// Get statistics
    pub fn stats(&self) -> &EGraphStats {
        &self.stats
    }

    /// Get the number of E-classes
    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    /// Push a new decision level
    pub fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_classes: self.classes.len(),
            memo_keys: self.memo.keys().cloned().collect(),
        });
        self.trail_limits.push(self.trail.len());
    }

    /// Pop to the previous decision level
    pub fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Undo trail entries
            if let Some(limit) = self.trail_limits.pop() {
                while self.trail.len() > limit {
                    if let Some(entry) = self.trail.pop() {
                        match entry {
                            UndoEntry::Union { child, old_parent } => {
                                self.parent[child.raw() as usize] = old_parent;
                            }
                            UndoEntry::RemoveNode { class, node } => {
                                self.classes[class.raw() as usize]
                                    .nodes
                                    .retain(|n| n != &node);
                            }
                            UndoEntry::RemoveParent { class, parent } => {
                                self.classes[class.raw() as usize].parents.remove(&parent);
                            }
                        }
                    }
                }
            }

            // Truncate classes
            self.classes.truncate(state.num_classes);
            self.parent.truncate(state.num_classes);

            // Rebuild memo table
            self.memo.clear();
            for node in state.memo_keys {
                if let Some(&class) = self.memo.get(&node)
                    && (class.raw() as usize) < self.classes.len()
                {
                    self.memo.insert(node, class);
                }
            }

            // Clear dirty set
            self.dirty
                .retain(|id| (id.raw() as usize) < state.num_classes);

            // Update stats
            self.stats.num_classes = self.classes.len();
        }
    }

    /// Clear all state
    pub fn reset(&mut self) {
        self.classes.clear();
        self.parent.clear();
        self.memo.clear();
        self.worklist.clear();
        self.context_stack.clear();
        self.trail.clear();
        self.trail_limits = vec![0];
        self.dirty.clear();
        self.stats = EGraphStats::default();
    }

    /// Run saturation (rebuild until fixpoint)
    pub fn saturate(&mut self) {
        while !self.worklist.is_empty() || !self.dirty.is_empty() {
            // Process worklist
            self.rebuild();

            // Rebuild remaining dirty classes
            let dirty: Vec<_> = self.dirty.drain().collect();
            for class_id in dirty {
                self.rebuild_class(class_id);
            }
        }
    }

    /// Extract the smallest term from an E-class
    pub fn extract_smallest(&self, id: EClassId) -> Option<ENode> {
        let class = self.get_class(id)?;

        // Find the node with the smallest size
        class
            .nodes
            .iter()
            .min_by_key(|node| self.node_size(node))
            .cloned()
    }

    /// Compute the size of an E-node (recursively)
    fn node_size(&self, node: &ENode) -> usize {
        1 + node
            .children
            .iter()
            .map(|&child| {
                self.get_class(child)
                    .and_then(|c| c.nodes.first())
                    .map(|n| self.node_size(n))
                    .unwrap_or(0)
            })
            .sum::<usize>()
    }
}

/// Builder for E-graphs
pub struct EGraphBuilder {
    config: EGraphConfig,
}

impl EGraphBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: EGraphConfig::default(),
        }
    }

    /// Set maximum classes
    pub fn max_classes(mut self, max: usize) -> Self {
        self.config.max_classes = max;
        self
    }

    /// Enable analysis
    pub fn enable_analysis(mut self, enable: bool) -> Self {
        self.config.enable_analysis = enable;
        self
    }

    /// Enable theory propagation
    pub fn enable_theory_propagation(mut self, enable: bool) -> Self {
        self.config.enable_theory_propagation = enable;
        self
    }

    /// Set rebuild batch size
    pub fn rebuild_batch_size(mut self, size: usize) -> Self {
        self.config.rebuild_batch_size = size;
        self
    }

    /// Build the E-graph
    pub fn build(self) -> EGraph {
        EGraph::with_config(self.config)
    }
}

impl Default for EGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eclass_id() {
        let id = EClassId::new(42);
        assert_eq!(id.raw(), 42);
    }

    #[test]
    fn test_enode_creation() {
        let node = ENode::new(1, [EClassId::new(0), EClassId::new(1)]);
        assert_eq!(node.op, 1);
        assert_eq!(node.children.len(), 2);
    }

    #[test]
    fn test_enode_leaf() {
        let node = ENode::leaf(42);
        assert_eq!(node.op, 42);
        assert!(node.children.is_empty());
    }

    #[test]
    fn test_egraph_add() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let b = egraph.add(ENode::leaf(2));

        assert_ne!(a, b);
        assert_eq!(egraph.num_classes(), 2);
    }

    #[test]
    fn test_egraph_hash_cons() {
        let mut egraph = EGraph::new();

        let a1 = egraph.add(ENode::leaf(1));
        let a2 = egraph.add(ENode::leaf(1));

        // Same E-node should return same E-class
        assert_eq!(a1, a2);
        assert_eq!(egraph.num_classes(), 1);
    }

    #[test]
    fn test_egraph_merge() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let b = egraph.add(ENode::leaf(2));

        assert!(!egraph.are_equal(a, b));

        egraph.merge(a, b);

        assert!(egraph.are_equal(a, b));
    }

    #[test]
    fn test_egraph_congruence() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let b = egraph.add(ENode::leaf(2));

        // f(a) and f(b)
        let fa = egraph.add(ENode::new(10, [a]));
        let fb = egraph.add(ENode::new(10, [b]));

        assert!(!egraph.are_equal(fa, fb));

        // Merge a and b
        egraph.merge(a, b);
        egraph.saturate();

        // f(a) and f(b) should now be equal by congruence
        assert!(egraph.are_equal(fa, fb));
    }

    #[test]
    fn test_egraph_push_pop() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let b = egraph.add(ENode::leaf(2));

        egraph.push();

        egraph.merge(a, b);
        assert!(egraph.are_equal(a, b));

        egraph.pop();

        // After pop, a and b should be separate again
        assert!(!egraph.are_equal(a, b));
    }

    #[test]
    fn test_egraph_lookup() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let _b = egraph.add(ENode::leaf(2));

        let node = ENode::leaf(1);
        let found = egraph.lookup(&node);

        assert_eq!(found, Some(a));

        let node2 = ENode::leaf(999);
        assert!(egraph.lookup(&node2).is_none());
    }

    #[test]
    fn test_egraph_extract_smallest() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));

        let smallest = egraph.extract_smallest(a);
        assert!(smallest.is_some());
        assert_eq!(smallest.as_ref().map(|n| n.op), Some(1));
    }

    #[test]
    fn test_egraph_stats() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let b = egraph.add(ENode::leaf(2));

        egraph.merge(a, b);

        let stats = egraph.stats();
        assert_eq!(stats.merges, 1);
        assert_eq!(stats.num_classes, 2);
    }

    #[test]
    fn test_egraph_builder() {
        let egraph = EGraphBuilder::new()
            .max_classes(1000)
            .enable_analysis(true)
            .rebuild_batch_size(50)
            .build();

        assert_eq!(egraph.config.max_classes, 1000);
        assert!(egraph.config.enable_analysis);
        assert_eq!(egraph.config.rebuild_batch_size, 50);
    }

    #[test]
    fn test_egraph_reset() {
        let mut egraph = EGraph::new();

        egraph.add(ENode::leaf(1));
        egraph.add(ENode::leaf(2));

        assert_eq!(egraph.num_classes(), 2);

        egraph.reset();

        assert_eq!(egraph.num_classes(), 0);
    }

    #[test]
    fn test_analysis_data() {
        let const_data = AnalysisData::Constant(ConstValue::Int(42));
        assert!(matches!(
            const_data,
            AnalysisData::Constant(ConstValue::Int(42))
        ));

        let bounds_data = AnalysisData::Bounds {
            min: Some(0),
            max: Some(100),
        };
        assert!(matches!(bounds_data, AnalysisData::Bounds { .. }));
    }

    #[test]
    fn test_nested_congruence() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let b = egraph.add(ENode::leaf(2));

        // f(a) and f(b)
        let fa = egraph.add(ENode::new(10, [a]));
        let fb = egraph.add(ENode::new(10, [b]));

        // g(f(a)) and g(f(b))
        let gfa = egraph.add(ENode::new(20, [fa]));
        let gfb = egraph.add(ENode::new(20, [fb]));

        assert!(!egraph.are_equal(gfa, gfb));

        // Merge a and b
        egraph.merge(a, b);
        egraph.saturate();

        // g(f(a)) and g(f(b)) should be equal by nested congruence
        assert!(egraph.are_equal(gfa, gfb));
    }

    #[test]
    fn test_multiple_children() {
        let mut egraph = EGraph::new();

        let a = egraph.add(ENode::leaf(1));
        let b = egraph.add(ENode::leaf(2));
        let c = egraph.add(ENode::leaf(3));

        // f(a, b, c)
        let fabc = egraph.add(ENode::new(10, [a, b, c]));

        let class = egraph.get_class(fabc);
        assert!(class.is_some());
        assert_eq!(class.map(|c| c.nodes[0].children.len()), Some(3));
    }
}
