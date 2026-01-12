//! E-graph (Equality Graph) based equality reasoning
//!
//! E-graphs are a data structure for efficiently representing and reasoning about
//! equivalence classes of terms. They support:
//! - Efficient congruence closure
//! - Term rewriting and simplification
//! - Equality saturation
//! - Pattern matching and extraction
//!
//! Reference: "Equality Saturation: A New Approach to Optimization" and Z3's E-matching

use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// E-class ID representing an equivalence class of terms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EClassId(pub u32);

/// E-node representing a term in the E-graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ENode {
    /// The term kind (operator)
    pub kind: ENodeKind,
    /// The children e-class IDs
    pub children: Vec<EClassId>,
}

/// E-node kind (simplified term representation)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ENodeKind {
    /// Variable
    Var(String),
    /// Constant (integer)
    IntConst(i64),
    /// Boolean constant
    BoolConst(bool),
    /// Addition
    Add,
    /// Multiplication
    Mul,
    /// Subtraction
    Sub,
    /// Negation
    Neg,
    /// Logical And
    And,
    /// Logical Or
    Or,
    /// Logical Not
    Not,
    /// Equality
    Eq,
    /// Less than
    Lt,
    /// If-then-else
    Ite,
    /// Array select
    Select,
    /// Array store
    Store,
}

/// E-class representing an equivalence class of terms
#[derive(Debug, Clone)]
pub struct EClass {
    /// The ID of this e-class
    pub id: EClassId,
    /// The canonical e-node representative
    pub nodes: Vec<ENode>,
    /// Parent e-classes (for congruence closure)
    pub parents: FxHashSet<EClassId>,
}

impl EClass {
    /// Create a new e-class
    #[must_use]
    pub fn new(id: EClassId) -> Self {
        Self {
            id,
            nodes: Vec::new(),
            parents: FxHashSet::default(),
        }
    }

    /// Add an e-node to this class
    pub fn add_node(&mut self, node: ENode) {
        if !self.nodes.contains(&node) {
            self.nodes.push(node);
        }
    }
}

/// E-graph for equality reasoning
#[derive(Debug, Clone)]
pub struct EGraph {
    /// E-classes indexed by ID
    classes: FxHashMap<EClassId, EClass>,
    /// Union-find structure for e-class merging
    unionfind: FxHashMap<EClassId, EClassId>,
    /// Hashcons: maps e-nodes to their e-class
    hashcons: FxHashMap<ENode, EClassId>,
    /// Pending merges for congruence closure
    pending: VecDeque<(EClassId, EClassId)>,
    /// Next e-class ID
    next_id: u32,
    /// Worklist for equality saturation
    worklist: VecDeque<EClassId>,
}

impl EGraph {
    /// Create a new empty E-graph
    #[must_use]
    pub fn new() -> Self {
        Self {
            classes: FxHashMap::default(),
            unionfind: FxHashMap::default(),
            hashcons: FxHashMap::default(),
            pending: VecDeque::new(),
            next_id: 0,
            worklist: VecDeque::new(),
        }
    }

    /// Generate a fresh e-class ID
    fn fresh_id(&mut self) -> EClassId {
        let id = EClassId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Find the canonical representative of an e-class (with path compression)
    pub fn find(&mut self, id: EClassId) -> EClassId {
        if let Some(&parent) = self.unionfind.get(&id)
            && parent != id
        {
            let root = self.find(parent);
            self.unionfind.insert(id, root);
            return root;
        }
        id
    }

    /// Add an e-node to the e-graph, returning its e-class ID
    pub fn add(&mut self, node: ENode) -> EClassId {
        // Canonicalize children
        let canonical_node = ENode {
            kind: node.kind.clone(),
            children: node.children.iter().map(|&c| self.find(c)).collect(),
        };

        // Check if this e-node already exists
        if let Some(&existing_id) = self.hashcons.get(&canonical_node) {
            return self.find(existing_id);
        }

        // Create a new e-class
        let id = self.fresh_id();
        let mut eclass = EClass::new(id);
        eclass.add_node(canonical_node.clone());

        // Update parent pointers
        for &child in &canonical_node.children {
            let child_id = self.find(child);
            if let Some(child_class) = self.classes.get_mut(&child_id) {
                child_class.parents.insert(id);
            }
        }

        self.classes.insert(id, eclass);
        self.unionfind.insert(id, id);
        self.hashcons.insert(canonical_node, id);
        self.worklist.push_back(id);

        id
    }

    /// Merge two e-classes
    pub fn merge(&mut self, id1: EClassId, id2: EClassId) -> EClassId {
        let root1 = self.find(id1);
        let root2 = self.find(id2);

        if root1 == root2 {
            return root1;
        }

        // Union: make root2 point to root1
        self.unionfind.insert(root2, root1);

        // Merge the e-classes
        let (class2_nodes, class2_parents) = if let Some(class2) = self.classes.remove(&root2) {
            (class2.nodes, class2.parents)
        } else {
            (Vec::new(), FxHashSet::default())
        };

        if let Some(class1) = self.classes.get_mut(&root1) {
            for node in class2_nodes {
                class1.add_node(node);
            }
            // Merge parent sets
            class1.parents.extend(class2_parents);
        }

        // Add to pending for congruence closure
        self.pending.push_back((root1, root2));

        root1
    }

    /// Perform congruence closure
    pub fn rebuild(&mut self) {
        while let Some((id1, id2)) = self.pending.pop_front() {
            self.process_merge(id1, id2);
        }
        // Rebuild hashcons to reflect canonicalized nodes
        self.rebuild_hashcons();
    }

    /// Rebuild the hashcons table after merges
    fn rebuild_hashcons(&mut self) {
        // Clear and rebuild hashcons with canonicalized nodes
        let mut new_hashcons = FxHashMap::default();

        // Collect all nodes from all classes
        let mut nodes_to_process = Vec::new();
        for (class_id, eclass) in &self.classes {
            for node in &eclass.nodes {
                nodes_to_process.push((*class_id, node.clone()));
            }
        }

        // Re-canonicalize and check for congruent nodes
        for (class_id, node) in nodes_to_process {
            let canonical_node = ENode {
                kind: node.kind.clone(),
                children: node.children.iter().map(|&c| self.find(c)).collect(),
            };

            // If this canonical form already exists in another class, merge them
            if let Some(&existing_class) = new_hashcons.get(&canonical_node) {
                let class_root = self.find(class_id);
                let existing_root = self.find(existing_class);
                if class_root != existing_root {
                    // Found congruent nodes in different classes - merge them
                    self.merge(class_root, existing_root);
                }
            } else {
                let class_root = self.find(class_id);
                new_hashcons.insert(canonical_node, class_root);
            }
        }

        self.hashcons = new_hashcons;
    }

    /// Process a merge for congruence closure
    fn process_merge(&mut self, id1: EClassId, _id2: EClassId) {
        let root = self.find(id1);

        // Collect all parents from the merged class
        let parents: Vec<_> = self
            .classes
            .get(&root)
            .map(|c| c.parents.iter().copied().collect())
            .unwrap_or_default();

        // Check all pairs of parents for congruence
        for i in 0..parents.len() {
            for j in (i + 1)..parents.len() {
                let p1 = parents[i];
                let p2 = parents[j];

                if self.are_congruent(p1, p2) {
                    let p1_root = self.find(p1);
                    let p2_root = self.find(p2);
                    if p1_root != p2_root {
                        self.merge(p1_root, p2_root);
                    }
                }
            }
        }
    }

    /// Check if two e-classes are congruent
    fn are_congruent(&mut self, id1: EClassId, id2: EClassId) -> bool {
        // Clone the nodes to avoid borrow checker issues
        let nodes1 = self
            .classes
            .get(&id1)
            .map(|c| c.nodes.clone())
            .unwrap_or_default();
        let nodes2 = self
            .classes
            .get(&id2)
            .map(|c| c.nodes.clone())
            .unwrap_or_default();

        // Check if any nodes in class1 are congruent with any nodes in class2
        for node1 in &nodes1 {
            for node2 in &nodes2 {
                if self.nodes_congruent(node1, node2) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if two e-nodes are congruent
    ///
    /// Two nodes are congruent if they have the same operator and
    /// their children are in the same equivalence classes
    fn nodes_congruent(&mut self, node1: &ENode, node2: &ENode) -> bool {
        if node1.kind != node2.kind {
            return false;
        }

        if node1.children.len() != node2.children.len() {
            return false;
        }

        // Re-canonicalize children and compare
        node1
            .children
            .iter()
            .zip(&node2.children)
            .all(|(&c1, &c2)| {
                let canon1 = self.find(c1);
                let canon2 = self.find(c2);
                canon1 == canon2
            })
    }

    /// Extract a canonical term from an e-class
    ///
    /// This finds the "best" representative term from an e-class
    /// (typically the smallest or simplest)
    pub fn extract(&self, id: EClassId) -> Option<&ENode> {
        let root_id = self.unionfind.get(&id).copied().unwrap_or(id);
        self.classes
            .get(&root_id)
            .and_then(|class| class.nodes.first())
    }

    /// Get all e-nodes in an e-class
    #[must_use]
    pub fn get_class(&self, id: EClassId) -> Option<&EClass> {
        let root_id = self.unionfind.get(&id).copied().unwrap_or(id);
        self.classes.get(&root_id)
    }

    /// Check if two e-classes are equal
    pub fn equiv(&mut self, id1: EClassId, id2: EClassId) -> bool {
        self.find(id1) == self.find(id2)
    }

    /// Get statistics about the E-graph
    #[must_use]
    pub fn statistics(&self) -> EGraphStats {
        let mut total_nodes = 0;
        for class in self.classes.values() {
            total_nodes += class.nodes.len();
        }

        EGraphStats {
            num_eclasses: self.classes.len(),
            num_enodes: total_nodes,
            pending_merges: self.pending.len(),
        }
    }

    /// Convert a TermId from TermManager to an E-class ID
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) -> Option<EClassId> {
        let t = manager.get(term)?;

        let (kind, children) = match &t.kind {
            TermKind::Var(name) => {
                let name_str = manager.resolve_str(*name);
                (ENodeKind::Var(name_str.to_string()), Vec::new())
            }
            TermKind::IntConst(val) => {
                // Convert BigInt to i64 (truncate if needed)
                let val_i64 = val.to_string().parse::<i64>().unwrap_or(0);
                (ENodeKind::IntConst(val_i64), Vec::new())
            }
            TermKind::True => (ENodeKind::BoolConst(true), Vec::new()),
            TermKind::False => (ENodeKind::BoolConst(false), Vec::new()),
            TermKind::Add(args) => {
                let child_ids: Vec<_> = args
                    .iter()
                    .filter_map(|&arg| self.add_term(arg, manager))
                    .collect();
                (ENodeKind::Add, child_ids)
            }
            TermKind::Mul(args) => {
                let child_ids: Vec<_> = args
                    .iter()
                    .filter_map(|&arg| self.add_term(arg, manager))
                    .collect();
                (ENodeKind::Mul, child_ids)
            }
            TermKind::Sub(a, b) => {
                let a_id = self.add_term(*a, manager)?;
                let b_id = self.add_term(*b, manager)?;
                (ENodeKind::Sub, vec![a_id, b_id])
            }
            TermKind::Neg(a) => {
                let a_id = self.add_term(*a, manager)?;
                (ENodeKind::Neg, vec![a_id])
            }
            TermKind::And(args) => {
                let child_ids: Vec<_> = args
                    .iter()
                    .filter_map(|&arg| self.add_term(arg, manager))
                    .collect();
                (ENodeKind::And, child_ids)
            }
            TermKind::Or(args) => {
                let child_ids: Vec<_> = args
                    .iter()
                    .filter_map(|&arg| self.add_term(arg, manager))
                    .collect();
                (ENodeKind::Or, child_ids)
            }
            TermKind::Not(a) => {
                let a_id = self.add_term(*a, manager)?;
                (ENodeKind::Not, vec![a_id])
            }
            TermKind::Eq(a, b) => {
                let a_id = self.add_term(*a, manager)?;
                let b_id = self.add_term(*b, manager)?;
                (ENodeKind::Eq, vec![a_id, b_id])
            }
            TermKind::Lt(a, b) => {
                let a_id = self.add_term(*a, manager)?;
                let b_id = self.add_term(*b, manager)?;
                (ENodeKind::Lt, vec![a_id, b_id])
            }
            TermKind::Ite(c, t, e) => {
                let c_id = self.add_term(*c, manager)?;
                let t_id = self.add_term(*t, manager)?;
                let e_id = self.add_term(*e, manager)?;
                (ENodeKind::Ite, vec![c_id, t_id, e_id])
            }
            TermKind::Select(a, i) => {
                let a_id = self.add_term(*a, manager)?;
                let i_id = self.add_term(*i, manager)?;
                (ENodeKind::Select, vec![a_id, i_id])
            }
            TermKind::Store(a, i, v) => {
                let a_id = self.add_term(*a, manager)?;
                let i_id = self.add_term(*i, manager)?;
                let v_id = self.add_term(*v, manager)?;
                (ENodeKind::Store, vec![a_id, i_id, v_id])
            }
            _ => return None,
        };

        let node = ENode { kind, children };
        Some(self.add(node))
    }

    /// Assert an equality between two terms
    pub fn assert_eq(&mut self, term1: TermId, term2: TermId, manager: &TermManager) {
        if let (Some(id1), Some(id2)) =
            (self.add_term(term1, manager), self.add_term(term2, manager))
        {
            self.merge(id1, id2);
            self.rebuild();
        }
    }

    /// Extract the lowest-cost e-node from an e-class using the given cost function
    ///
    /// This implements a greedy extraction algorithm that finds the e-node with
    /// the minimum cost in the given e-class.
    ///
    /// # Arguments
    ///
    /// * `eclass_id` - The e-class to extract from
    /// * `cost_fn` - The cost function to use
    ///
    /// # Returns
    ///
    /// The lowest-cost e-node in the e-class, or None if the e-class doesn't exist
    pub fn extract_best<F>(&self, eclass_id: EClassId, cost_fn: F) -> Option<ENode>
    where
        F: Fn(&ENodeKind, &[EClassId]) -> u64,
    {
        let canonical_id = *self.unionfind.get(&eclass_id).unwrap_or(&eclass_id);
        let eclass = self.classes.get(&canonical_id)?;

        eclass
            .nodes
            .iter()
            .min_by_key(|node| cost_fn(&node.kind, &node.children))
            .cloned()
    }

    /// Extract the lowest-cost e-node from each e-class
    ///
    /// This computes a cost for each e-class by finding the minimum-cost e-node
    /// in that class, considering the costs of children recursively.
    ///
    /// # Arguments
    ///
    /// * `node_cost` - Base cost function for different node kinds
    ///
    /// # Returns
    ///
    /// A map from e-class IDs to their lowest-cost e-nodes
    pub fn extract_all<F>(&self, node_cost: F) -> FxHashMap<EClassId, (ENode, u64)>
    where
        F: Fn(&ENodeKind) -> u64,
    {
        let mut costs: FxHashMap<EClassId, u64> = FxHashMap::default();
        let mut best_nodes: FxHashMap<EClassId, ENode> = FxHashMap::default();
        let mut changed = true;

        // Iteratively compute costs until fixed point
        let max_iterations = 100;
        let mut iterations = 0;

        while changed && iterations < max_iterations {
            changed = false;
            iterations += 1;

            for (eclass_id, eclass) in &self.classes {
                let canonical_id = self.find_canonical(*eclass_id);

                let mut min_cost = u64::MAX;
                let mut best_node = None;

                for node in &eclass.nodes {
                    // Calculate cost of this node (base cost + children costs)
                    let base_cost = node_cost(&node.kind);
                    let mut total_cost = base_cost;

                    for &child in &node.children {
                        let child_canonical = self.find_canonical(child);
                        if let Some(&child_cost) = costs.get(&child_canonical) {
                            total_cost = total_cost.saturating_add(child_cost);
                        } else {
                            // Child cost not computed yet, use base cost
                            total_cost = total_cost.saturating_add(1);
                        }
                    }

                    if total_cost < min_cost {
                        min_cost = total_cost;
                        best_node = Some(node.clone());
                    }
                }

                // Update if cost improved
                if let Some(node) = best_node {
                    let old_cost = costs.get(&canonical_id).copied().unwrap_or(u64::MAX);
                    if min_cost < old_cost {
                        costs.insert(canonical_id, min_cost);
                        best_nodes.insert(canonical_id, node);
                        changed = true;
                    }
                }
            }
        }

        // Combine results
        best_nodes
            .into_iter()
            .map(|(id, node)| {
                let cost = costs.get(&id).copied().unwrap_or(u64::MAX);
                (id, (node, cost))
            })
            .collect()
    }

    /// Find the canonical e-class ID (helper for extraction)
    fn find_canonical(&self, id: EClassId) -> EClassId {
        let mut current = id;
        while let Some(&parent) = self.unionfind.get(&current) {
            if parent == current {
                return current;
            }
            current = parent;
        }
        current
    }
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about an E-graph
#[derive(Debug, Default, Clone)]
pub struct EGraphStats {
    /// Number of e-classes
    pub num_eclasses: usize,
    /// Number of e-nodes
    pub num_enodes: usize,
    /// Number of pending merges
    pub pending_merges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_egraph() {
        let egraph = EGraph::new();
        assert_eq!(egraph.classes.len(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut egraph = EGraph::new();
        let node = ENode {
            kind: ENodeKind::IntConst(42),
            children: Vec::new(),
        };
        let id = egraph.add(node);
        assert!(egraph.get_class(id).is_some());
    }

    #[test]
    fn test_merge_classes() {
        let mut egraph = EGraph::new();

        let node1 = ENode {
            kind: ENodeKind::IntConst(1),
            children: Vec::new(),
        };
        let node2 = ENode {
            kind: ENodeKind::IntConst(2),
            children: Vec::new(),
        };

        let id1 = egraph.add(node1);
        let id2 = egraph.add(node2);

        assert!(!egraph.equiv(id1, id2));

        egraph.merge(id1, id2);

        assert!(egraph.equiv(id1, id2));
    }

    #[test]
    fn test_congruence_closure() {
        let mut egraph = EGraph::new();

        // Create a and b
        let a = egraph.add(ENode {
            kind: ENodeKind::Var("a".to_string()),
            children: Vec::new(),
        });
        let b = egraph.add(ENode {
            kind: ENodeKind::Var("b".to_string()),
            children: Vec::new(),
        });

        // Create f(a) and f(b)
        let fa = egraph.add(ENode {
            kind: ENodeKind::Add,
            children: vec![a],
        });
        let fb = egraph.add(ENode {
            kind: ENodeKind::Add,
            children: vec![b],
        });

        // Assert a = b
        egraph.merge(a, b);
        egraph.rebuild();

        // f(a) should equal f(b) by congruence
        assert!(egraph.equiv(fa, fb));
    }

    #[test]
    fn test_statistics() {
        let mut egraph = EGraph::new();

        egraph.add(ENode {
            kind: ENodeKind::IntConst(1),
            children: Vec::new(),
        });
        egraph.add(ENode {
            kind: ENodeKind::IntConst(2),
            children: Vec::new(),
        });

        let stats = egraph.statistics();
        assert_eq!(stats.num_eclasses, 2);
        assert_eq!(stats.num_enodes, 2);
    }

    #[test]
    fn test_add_term() {
        let mut manager = TermManager::new();
        let mut egraph = EGraph::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add(vec![x, y]);

        let sum_id = egraph.add_term(sum, &manager);
        assert!(sum_id.is_some());

        let stats = egraph.statistics();
        assert!(stats.num_eclasses >= 1);
    }

    #[test]
    fn test_assert_equality() {
        let mut manager = TermManager::new();
        let mut egraph = EGraph::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);

        egraph.assert_eq(x, y, &manager);

        // x and y should be in the same e-class
        if let (Some(x_id), Some(y_id)) =
            (egraph.add_term(x, &manager), egraph.add_term(y, &manager))
        {
            assert!(egraph.equiv(x_id, y_id));
        }
    }

    #[test]
    fn test_extract_best_simple() {
        let mut egraph = EGraph::new();

        // Create two constants in the same e-class
        let const1 = ENode {
            kind: ENodeKind::IntConst(1),
            children: Vec::new(),
        };
        let const2 = ENode {
            kind: ENodeKind::IntConst(2),
            children: Vec::new(),
        };

        let id1 = egraph.add(const1.clone());
        let id2 = egraph.add(const2);

        // Merge them
        egraph.merge(id1, id2);

        // Cost function: prefer smaller constants
        let cost_fn = |kind: &ENodeKind, children: &[EClassId]| -> u64 {
            match kind {
                ENodeKind::IntConst(n) => n.unsigned_abs(),
                _ => children.len() as u64,
            }
        };

        // Extract should return const1 (cost 1) rather than const2 (cost 2)
        let best = egraph.extract_best(id1, cost_fn);
        assert!(best.is_some());
        assert_eq!(best.unwrap().kind, ENodeKind::IntConst(1));
    }

    #[test]
    fn test_extract_best_with_children() {
        let mut egraph = EGraph::new();

        // Create variable
        let var_x = ENode {
            kind: ENodeKind::Var("x".to_string()),
            children: Vec::new(),
        };
        let x_id = egraph.add(var_x);

        // Create two equivalent expressions: x + 0 and x
        let zero = ENode {
            kind: ENodeKind::IntConst(0),
            children: Vec::new(),
        };
        let zero_id = egraph.add(zero);

        let add_x_0 = ENode {
            kind: ENodeKind::Add,
            children: vec![x_id, zero_id],
        };
        let add_id = egraph.add(add_x_0);

        // Merge add_id with x_id (since x + 0 = x)
        egraph.merge(add_id, x_id);

        // Cost function: prefer fewer children
        let cost_fn =
            |_kind: &ENodeKind, children: &[EClassId]| -> u64 { (children.len() as u64 + 1) * 10 };

        // Extract should prefer var_x (no children) over add_x_0 (two children)
        let best = egraph.extract_best(x_id, cost_fn);
        assert!(best.is_some());
        let best_node = best.unwrap();
        assert_eq!(best_node.kind, ENodeKind::Var("x".to_string()));
        assert_eq!(best_node.children.len(), 0);
    }

    #[test]
    fn test_extract_all() {
        let mut egraph = EGraph::new();

        // Create a simple expression: 1 + 2
        let one = ENode {
            kind: ENodeKind::IntConst(1),
            children: Vec::new(),
        };
        let two = ENode {
            kind: ENodeKind::IntConst(2),
            children: Vec::new(),
        };

        let id1 = egraph.add(one);
        let id2 = egraph.add(two);

        let add = ENode {
            kind: ENodeKind::Add,
            children: vec![id1, id2],
        };
        let add_id = egraph.add(add);

        // Simple cost function: constants cost 1, operations cost 10
        let node_cost = |kind: &ENodeKind| -> u64 {
            match kind {
                ENodeKind::IntConst(_) | ENodeKind::Var(_) | ENodeKind::BoolConst(_) => 1,
                _ => 10,
            }
        };

        let extracted = egraph.extract_all(node_cost);

        // Should have 3 e-classes (1, 2, and 1+2)
        assert_eq!(extracted.len(), 3);

        // Constants should have cost 1
        if let Some((_, cost)) = extracted.get(&id1) {
            assert_eq!(*cost, 1);
        }

        // Add should have cost 12 (10 for add + 1 for each child)
        if let Some((_, cost)) = extracted.get(&add_id) {
            assert_eq!(*cost, 12);
        }
    }

    #[test]
    fn test_extract_all_with_merge() {
        let mut egraph = EGraph::new();

        // Create x and y
        let x = ENode {
            kind: ENodeKind::Var("x".to_string()),
            children: Vec::new(),
        };
        let y = ENode {
            kind: ENodeKind::Var("y".to_string()),
            children: Vec::new(),
        };

        let x_id = egraph.add(x.clone());
        let y_id = egraph.add(y);

        // Merge x and y
        egraph.merge(x_id, y_id);
        egraph.rebuild();

        // Cost function: prefer shorter variable names
        let node_cost = |kind: &ENodeKind| -> u64 {
            match kind {
                ENodeKind::Var(name) => name.len() as u64,
                _ => 100,
            }
        };

        let extracted = egraph.extract_all(node_cost);

        // Should pick the one with shorter name (both x and y have length 1)
        assert!(!extracted.is_empty());
        for (_, (node, cost)) in extracted {
            if matches!(node.kind, ENodeKind::Var(_)) {
                assert_eq!(cost, 1);
            }
        }
    }

    #[test]
    fn test_find_canonical() {
        let mut egraph = EGraph::new();

        let node1 = ENode {
            kind: ENodeKind::IntConst(1),
            children: Vec::new(),
        };
        let node2 = ENode {
            kind: ENodeKind::IntConst(2),
            children: Vec::new(),
        };

        let id1 = egraph.add(node1);
        let id2 = egraph.add(node2);

        // Before merge, canonical IDs should be themselves
        assert_eq!(egraph.find_canonical(id1), id1);
        assert_eq!(egraph.find_canonical(id2), id2);

        // After merge, both should point to the same canonical ID
        let merged = egraph.merge(id1, id2);
        assert_eq!(egraph.find_canonical(id1), merged);
        assert_eq!(egraph.find_canonical(id2), merged);
    }
}
