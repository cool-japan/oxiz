//! Bellman-Ford Algorithm for Difference Logic
//!
//! Implements shortest path computation with negative cycle detection.

use super::graph::{ConstraintGraph, DiffEdge, DiffVar};
use num_rational::Rational64;
use std::collections::{HashMap, VecDeque};

/// A negative cycle detected in the constraint graph
#[derive(Debug, Clone)]
pub struct NegativeCycle {
    /// Edge indices forming the cycle
    pub edges: Vec<usize>,
    /// Total weight of the cycle (negative)
    pub total_weight: Rational64,
}

impl NegativeCycle {
    /// Create a new negative cycle
    pub fn new(edges: Vec<usize>, total_weight: Rational64) -> Self {
        Self {
            edges,
            total_weight,
        }
    }

    /// Get the constraint indices in the cycle
    pub fn constraint_indices(&self) -> &[usize] {
        &self.edges
    }
}

/// Result of Bellman-Ford computation
#[derive(Debug, Clone)]
pub enum BellmanFordResult {
    /// Shortest paths found (no negative cycle)
    Distances(HashMap<DiffVar, Rational64>),
    /// Negative cycle detected
    NegativeCycle(NegativeCycle),
}

/// Bellman-Ford algorithm for shortest paths
///
/// Computes shortest paths from a source node to all other nodes,
/// or detects a negative cycle if one exists.
#[derive(Debug)]
pub struct BellmanFord {
    /// Distance from source to each node
    distances: HashMap<DiffVar, Rational64>,
    /// Parent edge for each node (for path reconstruction)
    parent_edge: HashMap<DiffVar, usize>,
    /// Whether distances are valid
    valid: bool,
    /// Detected negative cycle (if any)
    negative_cycle: Option<NegativeCycle>,
}

impl BellmanFord {
    /// Create a new Bellman-Ford solver
    pub fn new() -> Self {
        Self {
            distances: HashMap::new(),
            parent_edge: HashMap::new(),
            valid: false,
            negative_cycle: None,
        }
    }

    /// Initialize distances for the graph
    fn initialize(&mut self, graph: &ConstraintGraph) {
        self.distances.clear();
        self.parent_edge.clear();
        self.negative_cycle = None;

        // Source distance is 0
        self.distances
            .insert(DiffVar::SOURCE, Rational64::from_integer(0));

        // Initialize all nodes from source edges (they have 0-weight edges from source)
        for edge in graph.get_edges(DiffVar::SOURCE) {
            self.distances.insert(edge.to, edge.weight);
        }

        // If any node wasn't connected from source, give it distance 0
        for node in graph.vars() {
            self.distances
                .entry(node)
                .or_insert_with(|| Rational64::from_integer(0));
        }
    }

    /// Run the Bellman-Ford algorithm
    pub fn run(&mut self, graph: &ConstraintGraph) -> BellmanFordResult {
        self.initialize(graph);

        let n = graph.num_vars() + 1; // +1 for source

        // Relax edges |V| - 1 times
        for _ in 0..n {
            let mut changed = false;

            for edge in graph.all_edges() {
                if let Some(&dist_from) = self.distances.get(&edge.from) {
                    let new_dist = dist_from + edge.weight;
                    let current_dist = self.distances.get(&edge.to).copied();

                    let should_update = match current_dist {
                        None => true,
                        Some(d) => new_dist < d,
                    };

                    if should_update {
                        self.distances.insert(edge.to, new_dist);
                        self.parent_edge.insert(edge.to, edge.constraint_idx);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Check for negative cycles (one more iteration)
        for edge in graph.all_edges() {
            if let Some(&dist_from) = self.distances.get(&edge.from) {
                let new_dist = dist_from + edge.weight;
                if let Some(&dist_to) = self.distances.get(&edge.to) {
                    // For negative cycle detection, check if we can still improve
                    // Note: strict edges would need special handling with infinitesimals
                    if new_dist < dist_to {
                        // Found a negative cycle - extract it
                        let cycle = self.extract_negative_cycle(graph, edge);
                        self.negative_cycle = Some(cycle.clone());
                        self.valid = false;
                        return BellmanFordResult::NegativeCycle(cycle);
                    }
                }
            }
        }

        self.valid = true;
        BellmanFordResult::Distances(self.distances.clone())
    }

    /// Extract the negative cycle containing the given edge
    fn extract_negative_cycle(
        &self,
        graph: &ConstraintGraph,
        trigger_edge: &DiffEdge,
    ) -> NegativeCycle {
        // Use cycle detection: follow parent edges from trigger_edge.to
        // until we find a repeated node, then extract the cycle
        let mut current = trigger_edge.to;

        // First, go back n steps to ensure we're in the cycle
        for _ in 0..=graph.num_vars() {
            if let Some(&edge_idx) = self.parent_edge.get(&current) {
                if let Some(constraint) = graph.get_constraint(edge_idx) {
                    current = constraint.y;
                }
            } else {
                break;
            }
        }

        // Now extract the cycle
        let cycle_start = current;
        let mut cycle_edges = Vec::new();
        let mut total_weight = Rational64::from_integer(0);

        // Extract edges in the cycle by following parent pointers
        #[allow(clippy::while_let_loop)]
        loop {
            if let Some(&edge_idx) = self.parent_edge.get(&current) {
                if let Some(constraint) = graph.get_constraint(edge_idx) {
                    // Find the edge for this constraint
                    for edge in graph.get_edges(constraint.y) {
                        if edge.constraint_idx == edge_idx {
                            cycle_edges.push(edge_idx);
                            total_weight += edge.weight;
                            break;
                        }
                    }
                    current = constraint.y;

                    if current == cycle_start && !cycle_edges.is_empty() {
                        break;
                    }

                    // Safety check to avoid infinite loop
                    if cycle_edges.len() > graph.num_vars() as usize + 1 {
                        break;
                    }
                }
            } else {
                break;
            }
        }

        // If we didn't find a proper cycle, just use the trigger edge
        if cycle_edges.is_empty() {
            cycle_edges.push(trigger_edge.constraint_idx);
            total_weight = trigger_edge.weight;
        }

        NegativeCycle::new(cycle_edges, total_weight)
    }

    /// Get the distance to a node (if computed)
    pub fn get_distance(&self, node: DiffVar) -> Option<Rational64> {
        if self.valid {
            self.distances.get(&node).copied()
        } else {
            None
        }
    }

    /// Get all distances (if valid)
    pub fn distances(&self) -> Option<&HashMap<DiffVar, Rational64>> {
        if self.valid {
            Some(&self.distances)
        } else {
            None
        }
    }

    /// Get the detected negative cycle (if any)
    pub fn negative_cycle(&self) -> Option<&NegativeCycle> {
        self.negative_cycle.as_ref()
    }

    /// Check if distances are valid (no negative cycle)
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}

impl Default for BellmanFord {
    fn default() -> Self {
        Self::new()
    }
}

/// SPFA (Shortest Path Faster Algorithm) - optimized Bellman-Ford
///
/// Uses a queue-based approach that's often faster in practice.
#[derive(Debug)]
pub struct Spfa {
    /// Distance from source to each node
    distances: HashMap<DiffVar, Rational64>,
    /// Parent edge for each node
    parent_edge: HashMap<DiffVar, usize>,
    /// Whether node is in queue
    in_queue: HashMap<DiffVar, bool>,
    /// Visit count per node (for cycle detection)
    visit_count: HashMap<DiffVar, u32>,
}

impl Spfa {
    /// Create a new SPFA solver
    pub fn new() -> Self {
        Self {
            distances: HashMap::new(),
            parent_edge: HashMap::new(),
            in_queue: HashMap::new(),
            visit_count: HashMap::new(),
        }
    }

    /// Run SPFA algorithm
    pub fn run(&mut self, graph: &ConstraintGraph) -> BellmanFordResult {
        self.distances.clear();
        self.parent_edge.clear();
        self.in_queue.clear();
        self.visit_count.clear();

        let n = graph.num_vars() + 1;
        let mut queue = VecDeque::new();

        // Initialize source
        self.distances
            .insert(DiffVar::SOURCE, Rational64::from_integer(0));
        queue.push_back(DiffVar::SOURCE);
        self.in_queue.insert(DiffVar::SOURCE, true);
        self.visit_count.insert(DiffVar::SOURCE, 1);

        while let Some(u) = queue.pop_front() {
            self.in_queue.insert(u, false);

            for edge in graph.get_edges(u) {
                let dist_u = match self.distances.get(&u) {
                    Some(&d) => d,
                    None => continue,
                };

                let new_dist = dist_u + edge.weight;
                let current_dist = self.distances.get(&edge.to).copied();

                let should_update = match current_dist {
                    None => true,
                    Some(d) => new_dist < d,
                };

                if should_update {
                    self.distances.insert(edge.to, new_dist);
                    self.parent_edge.insert(edge.to, edge.constraint_idx);

                    if !self.in_queue.get(&edge.to).copied().unwrap_or(false) {
                        queue.push_back(edge.to);
                        self.in_queue.insert(edge.to, true);

                        let count = self.visit_count.entry(edge.to).or_insert(0);
                        *count += 1;

                        // If a node is visited more than n times, there's a negative cycle
                        if *count > n {
                            // Extract negative cycle
                            let cycle = self.extract_cycle(graph, edge.to);
                            return BellmanFordResult::NegativeCycle(cycle);
                        }
                    }
                }
            }
        }

        BellmanFordResult::Distances(self.distances.clone())
    }

    /// Extract negative cycle starting from a node known to be in a cycle
    fn extract_cycle(&self, graph: &ConstraintGraph, start: DiffVar) -> NegativeCycle {
        let mut cycle_edges = Vec::new();
        let mut total_weight = Rational64::from_integer(0);
        let mut visited: HashMap<DiffVar, bool> = HashMap::new();

        let mut current = start;

        // Find the cycle
        loop {
            if visited.get(&current).copied().unwrap_or(false) {
                break;
            }
            visited.insert(current, true);

            if let Some(&edge_idx) = self.parent_edge.get(&current) {
                if let Some(constraint) = graph.get_constraint(edge_idx) {
                    for edge in graph.get_edges(constraint.y) {
                        if edge.constraint_idx == edge_idx {
                            cycle_edges.push(edge_idx);
                            total_weight += edge.weight;
                            break;
                        }
                    }
                    current = constraint.y;
                } else {
                    break;
                }
            } else {
                break;
            }

            if current == start {
                break;
            }

            if cycle_edges.len() > graph.num_vars() as usize + 1 {
                break;
            }
        }

        NegativeCycle::new(cycle_edges, total_weight)
    }

    /// Get distances (if computed successfully)
    #[allow(dead_code)]
    pub fn distances(&self) -> &HashMap<DiffVar, Rational64> {
        &self.distances
    }
}

impl Default for Spfa {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff_logic::graph::DiffConstraint;
    use oxiz_core::ast::TermId;

    fn create_test_graph() -> ConstraintGraph {
        let mut graph = ConstraintGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let term_z = TermId::from(3u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);
        let z = graph.get_or_create_var(term_z);

        // x - y ≤ 3
        graph.add_constraint(DiffConstraint::new_leq(
            x,
            y,
            Rational64::from_integer(3),
            origin,
        ));

        // y - z ≤ 2
        graph.add_constraint(DiffConstraint::new_leq(
            y,
            z,
            Rational64::from_integer(2),
            origin,
        ));

        // z - x ≤ 1
        graph.add_constraint(DiffConstraint::new_leq(
            z,
            x,
            Rational64::from_integer(1),
            origin,
        ));

        graph
    }

    #[test]
    fn test_bellman_ford_no_cycle() {
        let graph = create_test_graph();
        let mut bf = BellmanFord::new();
        let result = bf.run(&graph);

        match result {
            BellmanFordResult::Distances(dists) => {
                assert!(!dists.is_empty());
                // Sum of cycle weights: 3 + 2 + 1 = 6 > 0, so no negative cycle
            }
            BellmanFordResult::NegativeCycle(_) => {
                panic!("Should not detect negative cycle");
            }
        }
    }

    #[test]
    fn test_bellman_ford_negative_cycle() {
        let mut graph = ConstraintGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let term_z = TermId::from(3u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);
        let z = graph.get_or_create_var(term_z);

        // Create a negative cycle:
        // x - y ≤ -1
        // y - z ≤ -1
        // z - x ≤ -1
        // Total: -3 < 0
        graph.add_constraint(DiffConstraint::new_leq(
            x,
            y,
            Rational64::from_integer(-1),
            origin,
        ));
        graph.add_constraint(DiffConstraint::new_leq(
            y,
            z,
            Rational64::from_integer(-1),
            origin,
        ));
        graph.add_constraint(DiffConstraint::new_leq(
            z,
            x,
            Rational64::from_integer(-1),
            origin,
        ));

        let mut bf = BellmanFord::new();
        let result = bf.run(&graph);

        match result {
            BellmanFordResult::Distances(_) => {
                panic!("Should detect negative cycle");
            }
            BellmanFordResult::NegativeCycle(cycle) => {
                assert!(!cycle.edges.is_empty());
                assert!(cycle.total_weight < Rational64::from_integer(0));
            }
        }
    }

    #[test]
    fn test_spfa_no_cycle() {
        let graph = create_test_graph();
        let mut spfa = Spfa::new();
        let result = spfa.run(&graph);

        match result {
            BellmanFordResult::Distances(dists) => {
                assert!(!dists.is_empty());
            }
            BellmanFordResult::NegativeCycle(_) => {
                panic!("Should not detect negative cycle");
            }
        }
    }

    #[test]
    fn test_spfa_negative_cycle() {
        let mut graph = ConstraintGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);

        // Simple negative cycle: x - y ≤ -1, y - x ≤ -1
        graph.add_constraint(DiffConstraint::new_leq(
            x,
            y,
            Rational64::from_integer(-1),
            origin,
        ));
        graph.add_constraint(DiffConstraint::new_leq(
            y,
            x,
            Rational64::from_integer(-1),
            origin,
        ));

        let mut spfa = Spfa::new();
        let result = spfa.run(&graph);

        match result {
            BellmanFordResult::Distances(_) => {
                panic!("Should detect negative cycle");
            }
            BellmanFordResult::NegativeCycle(cycle) => {
                assert!(!cycle.edges.is_empty());
            }
        }
    }
}
