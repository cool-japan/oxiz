//! Resolution Graph Analysis
//!
//! Analyzes the structure of resolution proofs to improve clause learning
//! and branching decisions. This module builds and analyzes resolution DAGs
//! (Directed Acyclic Graphs) to identify patterns that indicate good vs bad
//! decisions during search.
//!
//! Key features:
//! - Resolution DAG construction from conflict analysis
//! - Graph-based clause quality metrics
//! - Variable importance scoring based on resolution structure
//! - Resolution pattern detection for better learning

use crate::literal::{Lit, Var};
#[allow(unused_imports)]
use crate::prelude::*;

/// Node in the resolution graph
#[derive(Debug, Clone)]
pub struct ResolutionNode {
    /// Unique ID for this node
    id: usize,
    /// The clause at this node (None for decision nodes)
    clause: Option<Vec<Lit>>,
    /// IDs of parent nodes (clauses that were resolved to produce this)
    parents: Vec<usize>,
    /// The variable that was resolved on (if this is a resolution node)
    resolved_var: Option<Var>,
    /// Decision level where this clause was derived
    decision_level: usize,
    /// Whether this is a decision node
    is_decision: bool,
}

impl ResolutionNode {
    /// Create a new resolution node
    pub fn new(id: usize, clause: Vec<Lit>, decision_level: usize) -> Self {
        Self {
            id,
            clause: Some(clause),
            parents: Vec::new(),
            resolved_var: None,
            decision_level,
            is_decision: false,
        }
    }

    /// Create a decision node
    pub fn decision(id: usize, literal: Lit, decision_level: usize) -> Self {
        Self {
            id,
            clause: Some(vec![literal]),
            parents: Vec::new(),
            resolved_var: None,
            decision_level,
            is_decision: true,
        }
    }

    /// Mark this node as a resolution of two parent clauses
    pub fn add_resolution(&mut self, parent1: usize, parent2: usize, resolved_var: Var) {
        self.parents.push(parent1);
        self.parents.push(parent2);
        self.resolved_var = Some(resolved_var);
    }

    /// Get the clause at this node
    pub fn clause(&self) -> Option<&[Lit]> {
        self.clause.as_deref()
    }

    /// Get the parent node IDs
    pub fn parents(&self) -> &[usize] {
        &self.parents
    }

    /// Get the variable this node resolved on
    pub fn resolved_var(&self) -> Option<Var> {
        self.resolved_var
    }

    /// Check if this is a decision node
    pub fn is_decision(&self) -> bool {
        self.is_decision
    }

    /// Get the decision level
    pub fn decision_level(&self) -> usize {
        self.decision_level
    }
}

/// Resolution Graph for analyzing proof structure
#[derive(Debug)]
pub struct ResolutionGraph {
    /// All nodes in the graph
    nodes: Vec<ResolutionNode>,
    /// Map from clause hash to node ID for deduplication
    clause_map: HashMap<u64, usize>,
    /// Statistics
    stats: GraphStats,
}

/// Statistics about the resolution graph
#[derive(Debug, Default, Clone)]
pub struct GraphStats {
    /// Total number of resolution steps
    pub resolutions: usize,
    /// Total number of decision nodes
    pub decisions: usize,
    /// Maximum graph depth (longest path from leaf to root)
    pub max_depth: usize,
    /// Average number of parents per node
    pub avg_parents: f64,
    /// Variables that participate in many resolutions
    pub frequent_vars: HashMap<Var, usize>,
}

impl ResolutionGraph {
    /// Create a new resolution graph
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            clause_map: HashMap::new(),
            stats: GraphStats::default(),
        }
    }

    /// Add a clause node to the graph
    pub fn add_clause(&mut self, clause: Vec<Lit>, decision_level: usize) -> usize {
        let hash = Self::hash_clause(&clause);

        // Check if we already have this clause
        if let Some(&node_id) = self.clause_map.get(&hash) {
            return node_id;
        }

        let node_id = self.nodes.len();
        let node = ResolutionNode::new(node_id, clause, decision_level);

        self.nodes.push(node);
        self.clause_map.insert(hash, node_id);

        node_id
    }

    /// Add a decision node to the graph
    pub fn add_decision(&mut self, literal: Lit, decision_level: usize) -> usize {
        let node_id = self.nodes.len();
        let node = ResolutionNode::decision(node_id, literal, decision_level);

        self.nodes.push(node);
        self.stats.decisions += 1;

        node_id
    }

    /// Record a resolution between two clauses
    pub fn add_resolution(
        &mut self,
        parent1_id: usize,
        parent2_id: usize,
        resolved_var: Var,
        result_clause: Vec<Lit>,
        decision_level: usize,
    ) -> usize {
        let result_id = self.add_clause(result_clause, decision_level);

        // Update the result node to record the resolution
        if let Some(node) = self.nodes.get_mut(result_id)
            && node.parents.is_empty()
        {
            // Only add parents if not already set (for deduplication)
            node.add_resolution(parent1_id, parent2_id, resolved_var);
            self.stats.resolutions += 1;

            // Track variable frequency
            *self.stats.frequent_vars.entry(resolved_var).or_insert(0) += 1;
        }

        result_id
    }

    /// Compute graph depth starting from a given node.
    ///
    /// The depth of a node is 1 for a leaf (no parents) and
    /// `1 + max(depth of parents)` otherwise.
    ///
    /// A `node_id` that is not in the graph has no resolution depth and
    /// reports `0`; the previous implementation indexed the node vector
    /// directly and panicked instead, which is not acceptable in a public
    /// method taking an arbitrary index.
    pub fn compute_depth(&self, node_id: usize) -> usize {
        let mut memo = HashMap::new();
        self.compute_depth_memo(node_id, &mut memo)
    }

    /// Depth computation over an explicit heap stack, memoized on node id.
    ///
    /// This replaces a recursive walk with two defects:
    ///
    /// * Depth was the resolution-DAG depth, unguarded. The return type is
    ///   `usize`, so there was no channel through which a depth cap could
    ///   report giving up — a cap could only return a silently wrong depth.
    /// * The `visited` set was shared across the whole walk and never
    ///   unwound, so the *second* and later visits to a shared parent
    ///   returned `0` rather than that parent's depth. Any resolution DAG
    ///   with sharing — i.e. every non-trivial one — got an understated
    ///   depth. `memo` now carries each node's real depth, so sharing is
    ///   exploited instead of corrupting the answer.
    ///
    /// `on_path` keeps the original cycle behaviour: a parent that is still
    /// being expanded further down the current path contributes `0`, so a
    /// (malformed) cyclic graph terminates rather than looping.
    fn compute_depth_memo(&self, node_id: usize, memo: &mut HashMap<usize, usize>) -> usize {
        enum Step {
            /// Start resolving this node.
            Enter(usize),
            /// Every parent of this node has been resolved; fold them.
            Exit(usize),
        }

        let mut on_path: HashSet<usize> = HashSet::new();
        let mut stack = vec![Step::Enter(node_id)];

        while let Some(step) = stack.pop() {
            match step {
                Step::Enter(id) => {
                    if memo.contains_key(&id) {
                        continue;
                    }
                    if !on_path.insert(id) {
                        // Cycle: this node is still being expanded above us.
                        continue;
                    }
                    let Some(node) = self.nodes.get(id) else {
                        // Not a node of this graph: no depth.
                        memo.insert(id, 0);
                        on_path.remove(&id);
                        continue;
                    };
                    if node.parents.is_empty() {
                        memo.insert(id, 1); // Leaf node
                        on_path.remove(&id);
                        continue;
                    }
                    stack.push(Step::Exit(id));
                    for &parent_id in &node.parents {
                        stack.push(Step::Enter(parent_id));
                    }
                }
                Step::Exit(id) => {
                    let max_parent_depth = self.nodes.get(id).map_or(0, |node| {
                        node.parents
                            .iter()
                            // A parent with no memo entry is one that is
                            // still on the current path, i.e. a cycle edge;
                            // it contributes 0, as in the recursive form.
                            .map(|parent_id| memo.get(parent_id).copied().unwrap_or(0))
                            .max()
                            .unwrap_or(0)
                    });
                    memo.insert(id, max_parent_depth + 1);
                    on_path.remove(&id);
                }
            }
        }

        memo.get(&node_id).copied().unwrap_or(0)
    }

    /// Analyze the graph and update statistics
    pub fn analyze(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        // Compute maximum depth. A single memo shared across all start
        // nodes makes this linear in the DAG; the previous code allocated a
        // fresh visited set per node, making `analyze` quadratic.
        let mut memo = HashMap::new();
        let max_depth = (0..self.nodes.len())
            .map(|id| self.compute_depth_memo(id, &mut memo))
            .max()
            .unwrap_or(0);
        self.stats.max_depth = max_depth;

        // Compute average number of parents
        let total_parents: usize = self.nodes.iter().map(|n| n.parents.len()).sum();
        self.stats.avg_parents = total_parents as f64 / self.nodes.len() as f64;
    }

    /// Get the top-k most frequently resolved variables
    pub fn get_frequent_vars(&self, k: usize) -> Vec<(Var, usize)> {
        let mut vars: Vec<_> = self
            .stats
            .frequent_vars
            .iter()
            .map(|(&var, &count)| (var, count))
            .collect();

        vars.sort_by_key(|item| std::cmp::Reverse(item.1));
        vars.truncate(k);
        vars
    }

    /// Compute clause quality based on resolution graph structure
    ///
    /// Lower scores indicate better quality clauses:
    /// - Shorter resolution paths are better (fewer resolution steps)
    /// - Clauses involving frequently-resolved variables are more important
    /// - Clauses at lower decision levels are more general
    pub fn clause_quality(&self, node_id: usize) -> f64 {
        if node_id >= self.nodes.len() {
            return f64::MAX;
        }

        let node = &self.nodes[node_id];
        let depth = self.compute_depth(node_id) as f64;
        let decision_level = node.decision_level as f64;

        // Count how many literals involve frequently-resolved variables
        let freq_score = if let Some(clause) = node.clause() {
            clause
                .iter()
                .filter_map(|lit| {
                    self.stats
                        .frequent_vars
                        .get(&lit.var())
                        .map(|&count| count as f64)
                })
                .sum::<f64>()
        } else {
            0.0
        };

        // Quality = depth + decision_level / (1 + freq_score)
        // Lower is better
        depth + decision_level / (1.0 + freq_score)
    }

    /// Find redundant resolutions in the graph
    ///
    /// Returns node IDs of resolutions that could be eliminated
    pub fn find_redundant_resolutions(&self) -> Vec<usize> {
        let mut redundant = Vec::new();

        for (i, node) in self.nodes.iter().enumerate() {
            if node.parents.len() < 2 {
                continue; // Not a resolution node
            }

            // Check if this resolution could be bypassed
            // A resolution is redundant if we can reach the same clause
            // through a shorter path
            if self.has_shorter_path(i) {
                redundant.push(i);
            }
        }

        redundant
    }

    /// Check if there's a shorter path to derive the same clause
    fn has_shorter_path(&self, node_id: usize) -> bool {
        let node = &self.nodes[node_id];
        let Some(target_clause) = node.clause() else {
            return false;
        };

        let target_hash = Self::hash_clause(target_clause);

        // BFS from all leaf nodes to see if we can reach this clause faster
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut depths = HashMap::new();

        // Start from leaf nodes (nodes with no parents)
        for (id, n) in self.nodes.iter().enumerate() {
            if n.parents.is_empty() && id != node_id {
                queue.push_back(id);
                depths.insert(id, 0);
            }
        }

        while let Some(current_id) = queue.pop_front() {
            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id);

            let current_depth = depths[&current_id];

            // Check if this node has the same clause
            if let Some(clause) = self.nodes[current_id].clause()
                && Self::hash_clause(clause) == target_hash
                && current_depth < self.compute_depth(node_id)
            {
                return true; // Found a shorter path
            }

            // Explore children (nodes that use this as a parent)
            for (child_id, child) in self.nodes.iter().enumerate() {
                if child.parents.contains(&current_id) && !visited.contains(&child_id) {
                    queue.push_back(child_id);
                    depths.insert(child_id, current_depth + 1);
                }
            }
        }

        false
    }

    /// Hash a clause for deduplication
    fn hash_clause(clause: &[Lit]) -> u64 {
        use core::hash::BuildHasher;

        let mut sorted = clause.to_vec();
        sorted.sort_unstable_by_key(|lit| lit.code());

        let build = core::hash::BuildHasherDefault::<rustc_hash::FxHasher>::default();
        build.hash_one(&sorted)
    }

    /// Get statistics about the graph
    pub fn stats(&self) -> &GraphStats {
        &self.stats
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.clause_map.clear();
        self.stats = GraphStats::default();
    }

    /// Get the number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: usize) -> Option<&ResolutionNode> {
        self.nodes.get(node_id)
    }
}

impl Default for ResolutionGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Resolution Graph Analyzer
///
/// Provides high-level analysis of resolution graphs to guide solver decisions
#[derive(Debug)]
pub struct ResolutionAnalyzer {
    /// The resolution graph being analyzed
    graph: ResolutionGraph,
    /// Variable scores based on resolution frequency
    var_scores: HashMap<Var, f64>,
    /// Whether analysis is enabled
    enabled: bool,
}

impl ResolutionAnalyzer {
    /// Create a new resolution analyzer
    pub fn new() -> Self {
        Self {
            graph: ResolutionGraph::new(),
            var_scores: HashMap::new(),
            enabled: true,
        }
    }

    /// Enable or disable analysis
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if analysis is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the resolution graph
    pub fn graph(&self) -> &ResolutionGraph {
        &self.graph
    }

    /// Get mutable access to the resolution graph
    pub fn graph_mut(&mut self) -> &mut ResolutionGraph {
        &mut self.graph
    }

    /// Analyze the current graph and update variable scores
    pub fn analyze(&mut self) {
        if !self.enabled {
            return;
        }

        self.graph.analyze();

        // Update variable scores based on resolution frequency and graph structure
        self.var_scores.clear();

        for (&var, &count) in &self.graph.stats.frequent_vars {
            // Variables that appear in many resolutions are more important
            let frequency_score = count as f64;

            // Also consider the quality of clauses they appear in
            let quality_score: f64 = self
                .graph
                .nodes
                .iter()
                .filter(|node| {
                    node.clause()
                        .map(|c| c.iter().any(|lit| lit.var() == var))
                        .unwrap_or(false)
                })
                .map(|node| 1.0 / (1.0 + self.graph.clause_quality(node.id)))
                .sum();

            self.var_scores.insert(var, frequency_score + quality_score);
        }
    }

    /// Get the importance score for a variable
    ///
    /// Higher scores indicate more important variables for branching
    pub fn variable_importance(&self, var: Var) -> f64 {
        self.var_scores.get(&var).copied().unwrap_or(0.0)
    }

    /// Get the top-k most important variables
    pub fn get_important_vars(&self, k: usize) -> Vec<(Var, f64)> {
        let mut vars: Vec<_> = self
            .var_scores
            .iter()
            .map(|(&var, &score)| (var, score))
            .collect();

        vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        vars.truncate(k);
        vars
    }

    /// Clear the analyzer state
    pub fn clear(&mut self) {
        self.graph.clear();
        self.var_scores.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &GraphStats {
        self.graph.stats()
    }
}

impl Default for ResolutionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_graph_creation() {
        let graph = ResolutionGraph::new();
        assert_eq!(graph.num_nodes(), 0);
    }

    #[test]
    fn test_add_clause() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);
        let v1 = Var(1);

        let clause1 = vec![Lit::pos(v0), Lit::pos(v1)];
        let id1 = graph.add_clause(clause1.clone(), 0);

        assert_eq!(id1, 0);
        assert_eq!(graph.num_nodes(), 1);

        // Adding same clause should return same ID
        let id2 = graph.add_clause(clause1, 0);
        assert_eq!(id1, id2);
        assert_eq!(graph.num_nodes(), 1);
    }

    #[test]
    fn test_add_decision() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);

        let id = graph.add_decision(Lit::pos(v0), 1);
        assert_eq!(id, 0);
        assert_eq!(graph.num_nodes(), 1);

        let node = graph
            .get_node(id)
            .expect("Decision node must exist in graph");
        assert!(node.is_decision());
        assert_eq!(node.decision_level(), 1);
    }

    #[test]
    fn test_resolution() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);
        let v1 = Var(1);

        // Clause 1: x0 ∨ x1
        let clause1 = vec![Lit::pos(v0), Lit::pos(v1)];
        let id1 = graph.add_clause(clause1, 0);

        // Clause 2: ~x0 ∨ x1
        let clause2 = vec![Lit::neg(v0), Lit::pos(v1)];
        let id2 = graph.add_clause(clause2, 0);

        // Resolution on x0 produces: x1
        let result = vec![Lit::pos(v1)];
        let id3 = graph.add_resolution(id1, id2, v0, result, 1);

        assert_eq!(graph.num_nodes(), 3);

        let node = graph
            .get_node(id3)
            .expect("Resolution node must exist in graph");
        assert_eq!(node.parents().len(), 2);
        assert_eq!(node.resolved_var(), Some(v0));
        assert_eq!(graph.stats().resolutions, 1);
    }

    #[test]
    fn test_compute_depth() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);
        let v1 = Var(1);

        // Build a simple resolution chain
        let id1 = graph.add_clause(vec![Lit::pos(v0)], 0);
        let id2 = graph.add_clause(vec![Lit::neg(v0), Lit::pos(v1)], 0);
        let id3 = graph.add_resolution(id1, id2, v0, vec![Lit::pos(v1)], 1);

        assert_eq!(graph.compute_depth(id1), 1); // Leaf
        assert_eq!(graph.compute_depth(id2), 1); // Leaf
        assert_eq!(graph.compute_depth(id3), 2); // One level above leaves
    }

    /// A node id that is not in the graph has no depth. This used to index
    /// the node vector directly and panic.
    #[test]
    fn test_compute_depth_unknown_node_is_zero() {
        let graph = ResolutionGraph::new();
        assert_eq!(graph.compute_depth(42), 0);
    }

    /// Regression: the previous walk shared one `visited` set across the
    /// whole traversal and never unwound it, so the *second* visit to a
    /// shared parent contributed depth 0 and the reported depth was too
    /// small. Here `mid1` and `mid2` share the leaf `root`, and the top
    /// node has both as parents.
    #[test]
    fn test_compute_depth_counts_shared_parents_correctly() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);
        let v1 = Var(1);
        let v2 = Var(2);

        let leaf_a = graph.add_clause(vec![Lit::pos(v0)], 0);
        let leaf_b = graph.add_clause(vec![Lit::neg(v0), Lit::pos(v1)], 0);
        let mid1 = graph.add_resolution(leaf_a, leaf_b, v0, vec![Lit::pos(v1)], 1);
        let leaf_c = graph.add_clause(vec![Lit::neg(v1), Lit::pos(v2)], 0);
        let mid2 = graph.add_resolution(mid1, leaf_c, v1, vec![Lit::pos(v2)], 2);
        let top = graph.add_resolution(mid1, mid2, v2, vec![Lit::pos(v2), Lit::neg(v0)], 3);

        assert_eq!(graph.compute_depth(mid1), 2);
        assert_eq!(graph.compute_depth(mid2), 3);
        // 1 + max(depth(mid1)=2, depth(mid2)=3) = 4. The old walk visited
        // `mid1` first, marked its subtree, and then scored `mid2` as 0,
        // reporting 3.
        assert_eq!(graph.compute_depth(top), 4);
    }

    /// A 12_500-deep resolution chain on a 128 KiB stack: the assertion is
    /// that `compute_depth` returns at all (a stack overflow aborts the
    /// process), plus that the depth it reports is exact.
    #[test]
    fn test_compute_depth_deep_chain_does_not_overflow() {
        let worker = std::thread::Builder::new().stack_size(1 << 17).spawn(|| {
            const CHAIN: usize = 12_500;
            let mut graph = ResolutionGraph::new();
            let v0 = Var(0);
            let mut current = graph.add_clause(vec![Lit::pos(v0)], 0);
            for level in 1..=CHAIN {
                let side = graph.add_clause(vec![Lit::neg(Var(level as u32))], 0);
                current = graph.add_resolution(
                    current,
                    side,
                    v0,
                    vec![Lit::pos(Var(level as u32))],
                    level,
                );
            }
            (graph.compute_depth(current), CHAIN)
        });
        let (depth, chain) = match worker.map(std::thread::JoinHandle::join) {
            Ok(Ok(result)) => result,
            _ => panic!("deep-chain depth worker thread did not complete"),
        };
        assert_eq!(depth, chain + 1);
    }

    #[test]
    fn test_analyze() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);
        let v1 = Var(1);

        let id1 = graph.add_clause(vec![Lit::pos(v0)], 0);
        let id2 = graph.add_clause(vec![Lit::neg(v0), Lit::pos(v1)], 0);
        graph.add_resolution(id1, id2, v0, vec![Lit::pos(v1)], 1);

        graph.analyze();

        assert_eq!(graph.stats().resolutions, 1);
        assert!(graph.stats().max_depth > 0);
    }

    #[test]
    fn test_frequent_vars() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);
        let v1 = Var(1);
        let v2 = Var(2);

        // Multiple resolutions on v0
        let id1 = graph.add_clause(vec![Lit::pos(v0)], 0);
        let id2 = graph.add_clause(vec![Lit::neg(v0), Lit::pos(v1)], 0);
        graph.add_resolution(id1, id2, v0, vec![Lit::pos(v1)], 1);

        let id3 = graph.add_clause(vec![Lit::pos(v0), Lit::pos(v2)], 0);
        let id4 = graph.add_clause(vec![Lit::neg(v0)], 0);
        graph.add_resolution(id3, id4, v0, vec![Lit::pos(v2)], 1);

        let freq = graph.get_frequent_vars(10);
        assert!(!freq.is_empty());
        assert_eq!(freq[0].0, v0); // v0 should be most frequent
        assert_eq!(freq[0].1, 2); // Resolved twice
    }

    #[test]
    fn test_resolution_analyzer() {
        let mut analyzer = ResolutionAnalyzer::new();
        assert!(analyzer.is_enabled());

        let v0 = Var(0);
        let v1 = Var(1);

        let id1 = analyzer.graph_mut().add_clause(vec![Lit::pos(v0)], 0);
        let id2 = analyzer
            .graph_mut()
            .add_clause(vec![Lit::neg(v0), Lit::pos(v1)], 0);
        analyzer
            .graph_mut()
            .add_resolution(id1, id2, v0, vec![Lit::pos(v1)], 1);

        analyzer.analyze();

        // v0 should have some importance since it was resolved on
        assert!(analyzer.variable_importance(v0) > 0.0);
    }

    #[test]
    fn test_important_vars() {
        let mut analyzer = ResolutionAnalyzer::new();
        let v0 = Var(0);
        let v1 = Var(1);
        let v2 = Var(2);

        // Create multiple resolutions
        let id1 = analyzer.graph_mut().add_clause(vec![Lit::pos(v0)], 0);
        let id2 = analyzer
            .graph_mut()
            .add_clause(vec![Lit::neg(v0), Lit::pos(v1)], 0);
        analyzer
            .graph_mut()
            .add_resolution(id1, id2, v0, vec![Lit::pos(v1)], 1);

        let id3 = analyzer
            .graph_mut()
            .add_clause(vec![Lit::pos(v0), Lit::pos(v2)], 0);
        let id4 = analyzer.graph_mut().add_clause(vec![Lit::neg(v0)], 0);
        analyzer
            .graph_mut()
            .add_resolution(id3, id4, v0, vec![Lit::pos(v2)], 1);

        analyzer.analyze();

        let important = analyzer.get_important_vars(2);
        assert!(!important.is_empty());
        assert_eq!(important[0].0, v0); // v0 should be most important
    }

    #[test]
    fn test_clear() {
        let mut analyzer = ResolutionAnalyzer::new();
        let v0 = Var(0);

        analyzer.graph_mut().add_clause(vec![Lit::pos(v0)], 0);
        assert_eq!(analyzer.graph().num_nodes(), 1);

        analyzer.clear();
        assert_eq!(analyzer.graph().num_nodes(), 0);
    }

    #[test]
    fn test_clause_quality() {
        let mut graph = ResolutionGraph::new();
        let v0 = Var(0);
        let v1 = Var(1);

        let id1 = graph.add_clause(vec![Lit::pos(v0)], 0);
        let id2 = graph.add_clause(vec![Lit::neg(v0), Lit::pos(v1)], 0);
        let id3 = graph.add_resolution(id1, id2, v0, vec![Lit::pos(v1)], 1);

        graph.analyze();

        // Leaf clauses should have better quality (lower score) than derived clauses
        let quality1 = graph.clause_quality(id1);
        let quality3 = graph.clause_quality(id3);

        assert!(quality1 <= quality3);
    }

    #[test]
    fn test_disabled_analyzer() {
        let mut analyzer = ResolutionAnalyzer::new();
        analyzer.set_enabled(false);
        assert!(!analyzer.is_enabled());

        let v0 = Var(0);
        analyzer.graph_mut().add_clause(vec![Lit::pos(v0)], 0);

        // Analyze should do nothing when disabled
        analyzer.analyze();
        assert!(analyzer.var_scores.is_empty());
    }
}
