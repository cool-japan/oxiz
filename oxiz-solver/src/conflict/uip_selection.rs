//! UIP (Unique Implication Point) Selection for Conflict Analysis.
#![allow(dead_code)] // Under development
//!
//! Identifies UIPs in the implication graph for effective clause learning.
//!
//! ## UIP Strategies
//!
//! - **First UIP**: Closest UIP to conflict (default in modern SAT solvers)
//! - **Last UIP**: Decision literal (produces longest clauses)
//! - **All UIPs**: Enumerate all UIPs for analysis
//! - **Dominator-based**: Use dominator tree to find UIPs efficiently
//!
//! ## References
//!
//! - "Chaff: Engineering an Efficient SAT Solver" (Moskewicz et al., 2001)
//! - Z3's `sat/sat_cut_simplifier.cpp`

use oxiz_sat::Lit;
use rustc_hash::{FxHashMap, FxHashSet};

/// Decision level.
pub type Level = usize;

/// Node in the implication graph.
#[derive(Debug, Clone)]
pub struct ImplicationNode {
    /// The literal assigned.
    pub literal: Lit,
    /// Decision level of this assignment.
    pub level: Level,
    /// Reason for this assignment (None for decisions).
    pub reason: Option<Vec<Lit>>,
}

impl ImplicationNode {
    /// Create a new implication node.
    pub fn new(literal: Lit, level: Level, reason: Option<Vec<Lit>>) -> Self {
        Self {
            literal,
            level,
            reason,
        }
    }

    /// Check if this is a decision node.
    pub fn is_decision(&self) -> bool {
        self.reason.is_none()
    }
}

/// UIP selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UipStrategy {
    /// First UIP (closest to conflict).
    FirstUIP,
    /// Last UIP (decision literal).
    LastUIP,
    /// All UIPs (for analysis).
    AllUIPs,
    /// Dominator-based UIP.
    Dominator,
}

/// Configuration for UIP selection.
#[derive(Debug, Clone)]
pub struct UipConfig {
    /// UIP selection strategy.
    pub strategy: UipStrategy,
    /// Enable dominator tree computation.
    pub use_dominators: bool,
    /// Maximum distance from conflict to search.
    pub max_distance: Option<usize>,
}

impl Default for UipConfig {
    fn default() -> Self {
        Self {
            strategy: UipStrategy::FirstUIP,
            use_dominators: false,
            max_distance: Some(1000),
        }
    }
}

/// Statistics for UIP selection.
#[derive(Debug, Clone, Default)]
pub struct UipStats {
    /// UIPs found.
    pub uips_found: u64,
    /// First UIPs selected.
    pub first_uips: u64,
    /// Last UIPs selected.
    pub last_uips: u64,
    /// Average distance to conflict.
    pub avg_distance: f64,
    /// Dominator computations.
    pub dominator_computations: u64,
}

/// UIP selector for conflict analysis.
pub struct UipSelector {
    /// Configuration.
    config: UipConfig,
    /// Statistics.
    stats: UipStats,
    /// Implication graph.
    graph: FxHashMap<Lit, ImplicationNode>,
    /// Current decision level.
    current_level: Level,
}

impl UipSelector {
    /// Create a new UIP selector.
    pub fn new(config: UipConfig) -> Self {
        Self {
            config,
            stats: UipStats::default(),
            graph: FxHashMap::default(),
            current_level: 0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(UipConfig::default())
    }

    /// Add an implication to the graph.
    pub fn add_implication(&mut self, literal: Lit, level: Level, reason: Option<Vec<Lit>>) {
        let node = ImplicationNode::new(literal, level, reason);
        self.graph.insert(literal, node);

        if level > self.current_level {
            self.current_level = level;
        }
    }

    /// Find UIP for a conflict clause.
    ///
    /// Returns the UIP literal according to the configured strategy.
    pub fn find_uip(&mut self, conflict: &[Lit]) -> Option<Lit> {
        if conflict.is_empty() {
            return None;
        }

        match self.config.strategy {
            UipStrategy::FirstUIP => self.find_first_uip(conflict),
            UipStrategy::LastUIP => self.find_last_uip(conflict),
            UipStrategy::AllUIPs => self.find_first_uip(conflict), // Return first as default
            UipStrategy::Dominator => {
                if self.config.use_dominators {
                    self.find_dominator_uip(conflict)
                } else {
                    self.find_first_uip(conflict)
                }
            }
        }
    }

    /// Find the first UIP (closest to conflict).
    fn find_first_uip(&mut self, conflict: &[Lit]) -> Option<Lit> {
        // Get conflict level (highest level in conflict)
        let conflict_level = conflict
            .iter()
            .filter_map(|&lit| self.graph.get(&lit).map(|node| node.level))
            .max()?;

        // Count literals at conflict level
        let mut current_lits: FxHashSet<Lit> = conflict
            .iter()
            .filter(|&&lit| {
                self.graph
                    .get(&lit)
                    .is_some_and(|node| node.level == conflict_level)
            })
            .copied()
            .collect();

        // Traverse backward until we find UIP (single literal at conflict level)
        while current_lits.len() > 1 {
            // Pick a literal to resolve
            let &lit = current_lits.iter().next()?;
            current_lits.remove(&lit);

            // Add its reason clause
            if let Some(node) = self.graph.get(&lit)
                && let Some(reason) = &node.reason
            {
                for &reason_lit in reason {
                    if let Some(reason_node) = self.graph.get(&reason_lit)
                        && reason_node.level == conflict_level
                    {
                        current_lits.insert(reason_lit);
                    }
                }
            }
        }

        self.stats.uips_found += 1;
        self.stats.first_uips += 1;

        current_lits.iter().next().copied()
    }

    /// Find the last UIP (decision literal at conflict level).
    fn find_last_uip(&mut self, conflict: &[Lit]) -> Option<Lit> {
        // Get conflict level
        let conflict_level = conflict
            .iter()
            .filter_map(|&lit| self.graph.get(&lit).map(|node| node.level))
            .max()?;

        // Find decision literal at this level
        for (lit, node) in &self.graph {
            if node.level == conflict_level && node.is_decision() {
                self.stats.uips_found += 1;
                self.stats.last_uips += 1;
                return Some(*lit);
            }
        }

        // Fallback to first UIP
        self.find_first_uip(conflict)
    }

    /// Find UIP using dominator analysis.
    fn find_dominator_uip(&mut self, conflict: &[Lit]) -> Option<Lit> {
        self.stats.dominator_computations += 1;

        // Simplified: Compute dominators in implication graph
        // In real implementation, would use Lengauer-Tarjan algorithm

        // For now, fall back to first UIP
        self.find_first_uip(conflict)
    }

    /// Find all UIPs in the implication graph.
    pub fn find_all_uips(&mut self, conflict: &[Lit]) -> Vec<Lit> {
        let mut uips = Vec::new();

        // Get conflict level
        let conflict_level = match conflict
            .iter()
            .filter_map(|&lit| self.graph.get(&lit).map(|node| node.level))
            .max()
        {
            Some(level) => level,
            None => return uips,
        };

        // Find all literals at conflict level
        let conflict_level_lits: Vec<Lit> = conflict
            .iter()
            .filter(|&&lit| {
                self.graph
                    .get(&lit)
                    .is_some_and(|node| node.level == conflict_level)
            })
            .copied()
            .collect();

        // Check each for UIP property
        for &candidate in &conflict_level_lits {
            if self.is_uip(candidate, conflict_level) {
                uips.push(candidate);
            }
        }

        self.stats.uips_found += uips.len() as u64;

        uips
    }

    /// Check if a literal is a UIP at a given level.
    fn is_uip(&self, candidate: Lit, level: Level) -> bool {
        // A literal is a UIP if all paths from decision to conflict go through it

        // Simplified check: is it on the conflict side and single at its level?
        // Real implementation would do proper reachability analysis

        if let Some(node) = self.graph.get(&candidate) {
            node.level == level
        } else {
            false
        }
    }

    /// Compute distance from a literal to the conflict.
    pub fn distance_to_conflict(&self, lit: Lit, conflict: &[Lit]) -> Option<usize> {
        // Simplified: BFS from lit to any conflict literal

        let mut visited = FxHashSet::default();
        let mut queue = vec![(lit, 0)];
        visited.insert(lit);

        while let Some((current, dist)) = queue.pop() {
            if conflict.contains(&current) {
                return Some(dist);
            }

            if let Some(node) = self.graph.get(&current)
                && let Some(reason) = &node.reason
            {
                for &next_lit in reason {
                    if visited.insert(next_lit) {
                        queue.push((next_lit, dist + 1));
                    }
                }
            }
        }

        None
    }

    /// Update average distance statistics.
    fn update_distance_stats(&mut self, distance: usize) {
        let count = self.stats.uips_found;
        let old_avg = self.stats.avg_distance;
        self.stats.avg_distance = (old_avg * (count - 1) as f64 + distance as f64) / count as f64;
    }

    /// Clear the implication graph.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.current_level = 0;
    }

    /// Get statistics.
    pub fn stats(&self) -> &UipStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = UipStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_sat::Var;

    #[test]
    fn test_uip_selector_creation() {
        let selector = UipSelector::default_config();
        assert_eq!(selector.stats().uips_found, 0);
    }

    #[test]
    fn test_add_implication() {
        let mut selector = UipSelector::default_config();

        let lit = Lit::pos(Var::new(0));
        selector.add_implication(lit, 1, None);

        assert_eq!(selector.current_level, 1);
        assert!(selector.graph.contains_key(&lit));
    }

    #[test]
    fn test_decision_node() {
        let node = ImplicationNode::new(Lit::pos(Var::new(0)), 1, None);
        assert!(node.is_decision());

        let node2 = ImplicationNode::new(Lit::pos(Var::new(1)), 1, Some(vec![]));
        assert!(!node2.is_decision());
    }

    #[test]
    fn test_find_last_uip() {
        let mut selector = UipSelector::new(UipConfig {
            strategy: UipStrategy::LastUIP,
            ..Default::default()
        });

        // Add decision at level 1
        let decision = Lit::pos(Var::new(0));
        selector.add_implication(decision, 1, None);

        // Add propagated literals
        let prop1 = Lit::pos(Var::new(1));
        selector.add_implication(prop1, 1, Some(vec![decision]));

        let conflict = vec![prop1];

        let uip = selector.find_uip(&conflict);
        assert!(uip.is_some());
        assert_eq!(selector.stats().last_uips, 1);
    }

    #[test]
    fn test_stats() {
        let mut selector = UipSelector::default_config();

        let lit = Lit::pos(Var::new(0));
        selector.add_implication(lit, 1, None);

        let conflict = vec![lit];
        let _ = selector.find_uip(&conflict);

        assert!(selector.stats().uips_found > 0);
    }
}
