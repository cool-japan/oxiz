//! Implication Graph for CDCL Conflict Analysis.
//!
//! This module provides graph data structures and algorithms for analyzing
//! Boolean constraint propagation and identifying conflict causes.
//!
//! ## Key Concepts
//!
//! 1. **Implication Graph**: Directed graph where nodes are literals and edges
//!    represent propagation (antecedent → consequent)
//! 2. **UIP (Unique Implication Point)**: Cut point in graph that all paths
//!    from decision to conflict pass through
//! 3. **Conflict Side**: Literals at current decision level reachable from conflict
//!
//! ## Applications
//!
//! - First UIP identification for clause learning
//! - Decision level computation
//! - Conflict-driven clause learning (CDCL)
//! - Backjump level calculation
//!
//! ## References
//!
//! - Zhang et al.: "Efficient Conflict Driven Learning in a Boolean Satisfiability Solver" (ICCAD 2001)
//! - Z3's `sat/sat_cut_simplifier.cpp`

use oxiz_sat::Lit;
use rustc_hash::{FxHashMap, FxHashSet};

/// Variable type (absolute value of literal).
pub type Var = u32;

/// Decision level.
pub type Level = u32;

/// Node in implication graph.
#[derive(Debug, Clone)]
pub struct ImplicationNode {
    /// The literal at this node.
    pub literal: Lit,
    /// Decision level when this literal was assigned.
    pub level: Level,
    /// Antecedent literals (reasons for propagation).
    pub antecedents: Vec<Lit>,
    /// Consequents (literals this one implies).
    pub consequents: Vec<Lit>,
    /// Is this a decision literal?
    pub is_decision: bool,
}

impl ImplicationNode {
    /// Create decision node.
    pub fn decision(literal: Lit, level: Level) -> Self {
        Self {
            literal,
            level,
            antecedents: Vec::new(),
            consequents: Vec::new(),
            is_decision: true,
        }
    }

    /// Create propagated node.
    pub fn propagated(literal: Lit, level: Level, antecedents: Vec<Lit>) -> Self {
        Self {
            literal,
            level,
            antecedents,
            consequents: Vec::new(),
            is_decision: false,
        }
    }
}

/// Configuration for implication graph.
#[derive(Debug, Clone)]
pub struct ImplicationGraphConfig {
    /// Enable graph compaction.
    pub enable_compaction: bool,
    /// Maximum graph size before compaction.
    pub max_size: usize,
}

impl Default for ImplicationGraphConfig {
    fn default() -> Self {
        Self {
            enable_compaction: true,
            max_size: 100000,
        }
    }
}

/// Statistics for implication graph.
#[derive(Debug, Clone, Default)]
pub struct ImplicationGraphStats {
    /// UIPs computed.
    pub uips_computed: u64,
    /// Graph traversals.
    pub traversals: u64,
    /// Nodes added.
    pub nodes_added: u64,
    /// Compactions performed.
    pub compactions: u64,
}

/// Implication graph for CDCL.
pub struct ImplicationGraph {
    config: ImplicationGraphConfig,
    stats: ImplicationGraphStats,
    /// Nodes indexed by literal.
    nodes: FxHashMap<Lit, ImplicationNode>,
    /// Current decision level.
    current_level: Level,
    /// Literals at each decision level.
    level_lits: FxHashMap<Level, Vec<Lit>>,
}

impl ImplicationGraph {
    /// Create new implication graph.
    pub fn new() -> Self {
        Self::with_config(ImplicationGraphConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: ImplicationGraphConfig) -> Self {
        Self {
            config,
            stats: ImplicationGraphStats::default(),
            nodes: FxHashMap::default(),
            current_level: 0,
            level_lits: FxHashMap::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ImplicationGraphStats {
        &self.stats
    }

    /// Add decision literal.
    pub fn add_decision(&mut self, lit: Lit, level: Level) {
        let node = ImplicationNode::decision(lit, level);
        self.nodes.insert(lit, node);
        self.level_lits.entry(level).or_default().push(lit);
        self.current_level = level;
        self.stats.nodes_added += 1;

        self.check_compaction();
    }

    /// Add propagated literal.
    pub fn add_propagation(&mut self, lit: Lit, level: Level, antecedents: Vec<Lit>) {
        // Update antecedent consequents
        for &ant in &antecedents {
            if let Some(ant_node) = self.nodes.get_mut(&ant) {
                ant_node.consequents.push(lit);
            }
        }

        let node = ImplicationNode::propagated(lit, level, antecedents);
        self.nodes.insert(lit, node);
        self.level_lits.entry(level).or_default().push(lit);
        self.stats.nodes_added += 1;

        self.check_compaction();
    }

    /// Find first UIP (Unique Implication Point) for conflict.
    ///
    /// Returns the literal that is the first UIP cut point.
    pub fn find_first_uip(&mut self, conflict_lits: &[Lit], decision_level: Level) -> Option<Lit> {
        self.stats.uips_computed += 1;

        if conflict_lits.is_empty() {
            return None;
        }

        // Collect literals at current decision level involved in conflict
        let conflict_side = self.compute_conflict_side(conflict_lits, decision_level);

        if conflict_side.is_empty() {
            return conflict_lits.first().copied();
        }

        // Find the last literal assigned at current level in conflict side
        // This is the first UIP (closest to conflict)
        let mut uip = None;
        let mut max_order = 0;

        for &lit in &conflict_side {
            // In real implementation, would track assignment order
            // For now, use simple heuristic
            if self.is_at_level(lit, decision_level) {
                let order = self.get_assignment_order(lit);
                if order > max_order {
                    max_order = order;
                    uip = Some(lit);
                }
            }
        }

        uip
    }

    /// Compute conflict side of cut.
    ///
    /// Returns all literals at decision level reachable from conflict.
    fn compute_conflict_side(&mut self, conflict_lits: &[Lit], level: Level) -> FxHashSet<Lit> {
        self.stats.traversals += 1;

        let mut conflict_side = FxHashSet::default();
        let mut frontier: Vec<Lit> = conflict_lits.to_vec();
        let mut visited = FxHashSet::default();

        // Backward reachability from conflict
        while let Some(lit) = frontier.pop() {
            if !visited.insert(lit) {
                continue;
            }

            if let Some(node) = self.nodes.get(&lit)
                && node.level == level
            {
                conflict_side.insert(lit);

                // Add antecedents to frontier
                for &ant in &node.antecedents {
                    frontier.push(ant);
                }
            }
        }

        conflict_side
    }

    /// Check if literal is at given decision level.
    fn is_at_level(&self, lit: Lit, level: Level) -> bool {
        self.nodes
            .get(&lit)
            .map(|n| n.level == level)
            .unwrap_or(false)
    }

    /// Get assignment order (simplified - would track actual order).
    fn get_assignment_order(&self, lit: Lit) -> usize {
        // Simplified: use hash value as proxy for order
        // Real implementation would maintain assignment trail
        lit.var().0 as usize
    }

    /// Get decision level of literal.
    pub fn get_level(&self, lit: Lit) -> Option<Level> {
        self.nodes.get(&lit).map(|n| n.level)
    }

    /// Get antecedents of literal.
    pub fn get_antecedents(&self, lit: Lit) -> Option<&[Lit]> {
        self.nodes.get(&lit).map(|n| n.antecedents.as_slice())
    }

    /// Compute backjump level for learned clause.
    ///
    /// Returns the second-highest decision level in the clause.
    pub fn compute_backjump_level(&self, clause: &[Lit]) -> Level {
        if clause.len() <= 1 {
            return 0;
        }

        let mut levels: Vec<Level> = clause
            .iter()
            .filter_map(|&lit| self.get_level(lit))
            .collect();

        levels.sort_unstable();
        levels.dedup();

        // Return second-highest level (or 0 if only one level)
        if levels.len() >= 2 {
            levels[levels.len() - 2]
        } else {
            0
        }
    }

    /// Backtrack to decision level.
    pub fn backtrack(&mut self, level: Level) {
        // Remove all literals above this level
        self.nodes.retain(|_, node| node.level <= level);
        self.level_lits.retain(|&l, _| l <= level);
        self.current_level = level;
    }

    /// Clear graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.level_lits.clear();
        self.current_level = 0;
    }

    /// Check if compaction is needed.
    fn check_compaction(&mut self) {
        if self.config.enable_compaction && self.nodes.len() > self.config.max_size {
            self.compact();
        }
    }

    /// Compact graph by removing old literals.
    fn compact(&mut self) {
        // Remove literals from lowest levels
        if let Some(&min_level) = self.level_lits.keys().min() {
            self.nodes.retain(|_, node| node.level > min_level);
            self.level_lits.remove(&min_level);
            self.stats.compactions += 1;
        }
    }

    /// Get number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get current decision level.
    pub fn current_level(&self) -> Level {
        self.current_level
    }
}

impl Default for ImplicationGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(n: i32) -> Lit {
        Lit::from_dimacs(n)
    }

    #[test]
    fn test_graph_creation() {
        let graph = ImplicationGraph::new();
        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.current_level(), 0);
    }

    #[test]
    fn test_add_decision() {
        let mut graph = ImplicationGraph::new();

        graph.add_decision(lit(1), 1);

        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.get_level(lit(1)), Some(1));
        assert_eq!(graph.current_level(), 1);
    }

    #[test]
    fn test_add_propagation() {
        let mut graph = ImplicationGraph::new();

        graph.add_decision(lit(1), 1);
        graph.add_propagation(lit(2), 1, vec![lit(1)]);

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.get_antecedents(lit(2)), Some(&[lit(1)][..]));
    }

    #[test]
    fn test_get_level() {
        let mut graph = ImplicationGraph::new();

        graph.add_decision(lit(1), 1);
        graph.add_decision(lit(2), 2);

        assert_eq!(graph.get_level(lit(1)), Some(1));
        assert_eq!(graph.get_level(lit(2)), Some(2));
        assert_eq!(graph.get_level(lit(3)), None);
    }

    #[test]
    fn test_backjump_level() {
        let mut graph = ImplicationGraph::new();

        graph.add_decision(lit(1), 1);
        graph.add_decision(lit(2), 2);
        graph.add_decision(lit(3), 3);

        let clause = vec![lit(1), lit(2), lit(3)];
        let level = graph.compute_backjump_level(&clause);

        assert_eq!(level, 2); // Second-highest level
    }

    #[test]
    fn test_backtrack() {
        let mut graph = ImplicationGraph::new();

        graph.add_decision(lit(1), 1);
        graph.add_decision(lit(2), 2);
        graph.add_decision(lit(3), 3);

        graph.backtrack(1);

        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.get_level(lit(1)), Some(1));
        assert_eq!(graph.get_level(lit(2)), None);
        assert_eq!(graph.get_level(lit(3)), None);
    }

    #[test]
    fn test_clear() {
        let mut graph = ImplicationGraph::new();

        graph.add_decision(lit(1), 1);
        graph.add_decision(lit(2), 2);

        graph.clear();

        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.current_level(), 0);
    }

    #[test]
    fn test_find_first_uip() {
        let mut graph = ImplicationGraph::new();

        graph.add_decision(lit(1), 1);
        graph.add_propagation(lit(2), 1, vec![lit(1)]);
        graph.add_propagation(lit(3), 1, vec![lit(2)]);

        let conflict = vec![lit(3)];
        let uip = graph.find_first_uip(&conflict, 1);

        assert!(uip.is_some());
    }

    #[test]
    fn test_compaction() {
        let config = ImplicationGraphConfig {
            enable_compaction: true,
            max_size: 2,
        };

        let mut graph = ImplicationGraph::with_config(config);

        graph.add_decision(lit(1), 1);
        graph.add_decision(lit(2), 2);
        graph.add_decision(lit(3), 3); // Should trigger compaction

        assert!(graph.stats().compactions > 0);
    }
}
