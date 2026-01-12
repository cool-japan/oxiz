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
use std::collections::{HashMap, HashSet, VecDeque};

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

    /// Compute graph depth starting from a given node
    pub fn compute_depth(&self, node_id: usize) -> usize {
        let mut visited = HashSet::new();
        self.compute_depth_recursive(node_id, &mut visited)
    }

    /// Recursive depth computation with cycle detection
    fn compute_depth_recursive(&self, node_id: usize, visited: &mut HashSet<usize>) -> usize {
        if visited.contains(&node_id) {
            return 0; // Cycle detected (shouldn't happen in DAG)
        }

        visited.insert(node_id);

        let node = &self.nodes[node_id];
        if node.parents.is_empty() {
            return 1; // Leaf node
        }

        let max_parent_depth = node
            .parents
            .iter()
            .map(|&parent_id| self.compute_depth_recursive(parent_id, visited))
            .max()
            .unwrap_or(0);

        max_parent_depth + 1
    }

    /// Analyze the graph and update statistics
    pub fn analyze(&mut self) {
        if self.nodes.is_empty() {
            return;
        }

        // Compute maximum depth
        self.stats.max_depth = (0..self.nodes.len())
            .map(|id| self.compute_depth(id))
            .max()
            .unwrap_or(0);

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

        vars.sort_by(|a, b| b.1.cmp(&a.1));
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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut sorted = clause.to_vec();
        sorted.sort_unstable_by_key(|lit| lit.code());

        let mut hasher = DefaultHasher::new();
        sorted.hash(&mut hasher);
        hasher.finish()
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

        vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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

        let node = graph.get_node(id).unwrap();
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

        let node = graph.get_node(id3).unwrap();
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
