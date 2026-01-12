//! UTVPI Theory Solver
//!
//! Implements satisfiability checking for UTVPI constraints using
//! Bellman-Ford algorithm on the doubled graph.

use super::graph::{DoubledGraph, DoubledNode, Sign, UtConstraint, UtEdge};
use num_rational::Rational64;
use oxiz_core::ast::TermId;
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for UTVPI solver
#[derive(Debug, Clone)]
pub struct UtvpiConfig {
    /// Use SPFA instead of standard Bellman-Ford
    pub use_spfa: bool,
    /// Enable propagation of tight bounds
    pub propagate_bounds: bool,
    /// Enable lemma learning from conflicts
    pub learn_lemmas: bool,
}

impl Default for UtvpiConfig {
    fn default() -> Self {
        Self {
            use_spfa: true,
            propagate_bounds: true,
            learn_lemmas: true,
        }
    }
}

/// Statistics for UTVPI solver
#[derive(Debug, Clone, Default)]
pub struct UtvpiStats {
    /// Number of constraints added
    pub constraints_added: u64,
    /// Number of consistency checks
    pub checks: u64,
    /// Number of conflicts detected
    pub conflicts: u64,
    /// Number of propagations
    pub propagations: u64,
    /// Number of push operations
    pub pushes: u64,
    /// Number of pop operations
    pub pops: u64,
}

/// Result of UTVPI solver operations
#[derive(Debug, Clone)]
pub enum UtvpiResult {
    /// Satisfiable, no conflicts
    Ok,
    /// Conflict detected, returns constraint indices forming the conflict
    Conflict(Vec<usize>),
    /// Unknown (e.g., resource limit)
    Unknown,
}

/// A negative cycle in the doubled graph
#[derive(Debug, Clone)]
pub struct UtvpiNegativeCycle {
    /// Edge indices forming the cycle
    pub edges: Vec<usize>,
    /// Total weight of the cycle (for diagnostics)
    #[allow(dead_code)]
    pub total_weight: Rational64,
}

/// UTVPI Theory Solver
#[derive(Debug)]
pub struct UtvpiSolver {
    /// Configuration
    config: UtvpiConfig,
    /// Doubled graph
    graph: DoubledGraph,
    /// Distances from source
    distances: HashMap<DoubledNode, Rational64>,
    /// Parent edge for path reconstruction
    parent_edge: HashMap<DoubledNode, usize>,
    /// Are distances valid?
    distances_valid: bool,
    /// Statistics
    stats: UtvpiStats,
}

impl UtvpiSolver {
    /// Create a new UTVPI solver
    pub fn new(is_integer: bool) -> Self {
        Self {
            config: UtvpiConfig::default(),
            graph: DoubledGraph::new(is_integer),
            distances: HashMap::new(),
            parent_edge: HashMap::new(),
            distances_valid: false,
            stats: UtvpiStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(is_integer: bool, config: UtvpiConfig) -> Self {
        Self {
            config,
            graph: DoubledGraph::new(is_integer),
            distances: HashMap::new(),
            parent_edge: HashMap::new(),
            distances_valid: false,
            stats: UtvpiStats::default(),
        }
    }

    /// Get or create a variable for a term
    pub fn get_or_create_var(&mut self, term: TermId) -> u32 {
        self.distances_valid = false;
        self.graph.get_or_create_var(term)
    }

    /// Add a UTVPI constraint
    pub fn add_constraint(&mut self, constraint: UtConstraint) -> usize {
        self.stats.constraints_added += 1;
        self.distances_valid = false;
        self.graph.add_constraint(constraint)
    }

    /// Add constraint: x - y ≤ c
    pub fn add_diff(&mut self, x: u32, y: u32, bound: Rational64, origin: TermId) -> usize {
        self.add_constraint(UtConstraint::diff(x, y, bound, origin))
    }

    /// Add constraint: x + y ≤ c
    pub fn add_sum(&mut self, x: u32, y: u32, bound: Rational64, origin: TermId) -> usize {
        self.add_constraint(UtConstraint::sum(x, y, bound, origin))
    }

    /// Add constraint: -x - y ≤ c
    pub fn add_neg_sum(&mut self, x: u32, y: u32, bound: Rational64, origin: TermId) -> usize {
        self.add_constraint(UtConstraint::neg_sum(x, y, bound, origin))
    }

    /// Add constraint: x ≤ c
    pub fn add_upper(&mut self, x: u32, bound: Rational64, origin: TermId) -> usize {
        self.add_constraint(UtConstraint::upper(x, bound, origin))
    }

    /// Add constraint: -x ≤ c (i.e., x ≥ -c)
    pub fn add_lower(&mut self, x: u32, bound: Rational64, origin: TermId) -> usize {
        self.add_constraint(UtConstraint::lower(x, bound, origin))
    }

    /// Add general UTVPI constraint: ax + by ≤ c
    pub fn add_general(
        &mut self,
        x: u32,
        a: Sign,
        y: u32,
        b: Sign,
        bound: Rational64,
        origin: TermId,
    ) -> usize {
        self.add_constraint(UtConstraint::new(x, a, y, b, bound, origin))
    }

    /// Check consistency
    pub fn check(&mut self) -> UtvpiResult {
        self.stats.checks += 1;

        let result = if self.config.use_spfa {
            self.run_spfa()
        } else {
            self.run_bellman_ford()
        };

        match result {
            Ok(()) => {
                self.distances_valid = true;
                UtvpiResult::Ok
            }
            Err(cycle) => {
                self.stats.conflicts += 1;
                UtvpiResult::Conflict(cycle.edges)
            }
        }
    }

    /// Run standard Bellman-Ford algorithm
    fn run_bellman_ford(&mut self) -> Result<(), UtvpiNegativeCycle> {
        self.distances.clear();
        self.parent_edge.clear();

        // Initialize distances
        self.distances
            .insert(DoubledNode::SOURCE, Rational64::from_integer(0));

        // Initialize all nodes with 0 (source is connected to all)
        for node in self.graph.all_nodes() {
            self.distances.insert(node, Rational64::from_integer(0));
        }

        // Add source edges
        for edge in self.graph.get_edges(DoubledNode::SOURCE) {
            self.distances.insert(edge.to, edge.weight);
        }

        // Number of nodes = 2 * num_vars + 1 (source)
        let n = (self.graph.num_nodes() + 1) as usize;

        // Relax edges n times
        for _ in 0..n {
            let mut changed = false;

            for edge in self.graph.all_edges() {
                if let Some(&dist_from) = self.distances.get(&edge.from) {
                    let new_dist = dist_from + edge.weight;
                    let should_update = match self.distances.get(&edge.to) {
                        None => true,
                        Some(&d) => new_dist < d,
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

        // Check for negative cycles
        for edge in self.graph.all_edges() {
            if let Some(&dist_from) = self.distances.get(&edge.from) {
                let new_dist = dist_from + edge.weight;
                if let Some(&dist_to) = self.distances.get(&edge.to)
                    && new_dist < dist_to
                {
                    return Err(self.extract_cycle(edge));
                }
            }
        }

        Ok(())
    }

    /// Run SPFA (Shortest Path Faster Algorithm)
    fn run_spfa(&mut self) -> Result<(), UtvpiNegativeCycle> {
        self.distances.clear();
        self.parent_edge.clear();

        let n = self.graph.num_nodes() + 1;
        let mut queue: VecDeque<DoubledNode> = VecDeque::new();
        let mut in_queue: HashSet<DoubledNode> = HashSet::new();
        let mut visit_count: HashMap<DoubledNode, u32> = HashMap::new();

        // Initialize source
        self.distances
            .insert(DoubledNode::SOURCE, Rational64::from_integer(0));
        queue.push_back(DoubledNode::SOURCE);
        in_queue.insert(DoubledNode::SOURCE);
        visit_count.insert(DoubledNode::SOURCE, 1);

        while let Some(u) = queue.pop_front() {
            in_queue.remove(&u);

            for edge in self.graph.get_edges(u) {
                let dist_u = match self.distances.get(&u) {
                    Some(&d) => d,
                    None => continue,
                };

                let new_dist = dist_u + edge.weight;
                let should_update = match self.distances.get(&edge.to) {
                    None => true,
                    Some(&d) => new_dist < d,
                };

                if should_update {
                    self.distances.insert(edge.to, new_dist);
                    self.parent_edge.insert(edge.to, edge.constraint_idx);

                    if !in_queue.contains(&edge.to) {
                        queue.push_back(edge.to);
                        in_queue.insert(edge.to);

                        let count = visit_count.entry(edge.to).or_insert(0);
                        *count += 1;

                        // If a node is visited more than n times, negative cycle exists
                        if *count > n {
                            return Err(self.extract_cycle_from_node(edge.to));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract negative cycle from a triggering edge
    fn extract_cycle(&self, trigger_edge: &UtEdge) -> UtvpiNegativeCycle {
        let mut cycle_edges = Vec::new();
        let mut total_weight = Rational64::from_integer(0);
        let mut visited: HashSet<DoubledNode> = HashSet::new();
        let mut current = trigger_edge.to;

        // Go back to find a node in the cycle
        for _ in 0..=self.graph.num_nodes() {
            if let Some(&edge_idx) = self.parent_edge.get(&current) {
                if let Some(constraint) = self.graph.get_constraint(edge_idx) {
                    // Find the source node for this constraint
                    current = self.get_source_node_for_constraint(constraint);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let cycle_start = current;

        // Extract the cycle
        loop {
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);

            if let Some(&edge_idx) = self.parent_edge.get(&current) {
                if let Some(constraint) = self.graph.get_constraint(edge_idx) {
                    cycle_edges.push(edge_idx);

                    // Find edge weight
                    for edge in self
                        .graph
                        .get_edges(self.get_source_node_for_constraint(constraint))
                    {
                        if edge.constraint_idx == edge_idx && edge.to == current {
                            total_weight += edge.weight;
                            break;
                        }
                    }

                    current = self.get_source_node_for_constraint(constraint);

                    if current == cycle_start && !cycle_edges.is_empty() {
                        break;
                    }

                    if cycle_edges.len() > self.graph.num_nodes() as usize + 1 {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if cycle_edges.is_empty() {
            cycle_edges.push(trigger_edge.constraint_idx);
            total_weight = trigger_edge.weight;
        }

        UtvpiNegativeCycle {
            edges: cycle_edges,
            total_weight,
        }
    }

    /// Extract cycle from a node known to be in a cycle
    fn extract_cycle_from_node(&self, start: DoubledNode) -> UtvpiNegativeCycle {
        let mut cycle_edges = Vec::new();
        let mut total_weight = Rational64::from_integer(0);
        let mut visited: HashSet<DoubledNode> = HashSet::new();
        let mut current = start;

        loop {
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);

            if let Some(&edge_idx) = self.parent_edge.get(&current) {
                if let Some(constraint) = self.graph.get_constraint(edge_idx) {
                    cycle_edges.push(edge_idx);

                    let source = self.get_source_node_for_constraint(constraint);
                    for edge in self.graph.get_edges(source) {
                        if edge.constraint_idx == edge_idx && edge.to == current {
                            total_weight += edge.weight;
                            break;
                        }
                    }

                    current = source;

                    if current == start {
                        break;
                    }

                    if cycle_edges.len() > self.graph.num_nodes() as usize + 1 {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        UtvpiNegativeCycle {
            edges: cycle_edges,
            total_weight,
        }
    }

    /// Get source node for a constraint (helper for cycle extraction)
    fn get_source_node_for_constraint(&self, constraint: &UtConstraint) -> DoubledNode {
        // The source depends on the constraint type
        match (constraint.a, constraint.b) {
            (Sign::Positive, Sign::Negative) => DoubledNode::positive(constraint.y),
            (Sign::Negative, Sign::Positive) => DoubledNode::positive(constraint.x),
            (Sign::Positive, Sign::Positive) => DoubledNode::negative(constraint.y),
            (Sign::Negative, Sign::Negative) => DoubledNode::positive(constraint.y),
            (Sign::Positive, Sign::Zero) => DoubledNode::negative(constraint.x),
            (Sign::Negative, Sign::Zero) => DoubledNode::positive(constraint.x),
            (Sign::Zero, Sign::Positive) => DoubledNode::negative(constraint.y),
            (Sign::Zero, Sign::Negative) => DoubledNode::positive(constraint.y),
            (Sign::Zero, Sign::Zero) => DoubledNode::SOURCE,
        }
    }

    /// Push a new decision level
    pub fn push(&mut self) {
        self.stats.pushes += 1;
        self.graph.push();
    }

    /// Pop to a previous level
    pub fn pop(&mut self, levels: u32) {
        if levels > 0 {
            self.stats.pops += 1;
            self.distances_valid = false;
            self.graph.pop(levels);
        }
    }

    /// Get variable value from the model
    pub fn get_value(&self, var: u32) -> Option<Rational64> {
        if !self.distances_valid {
            return None;
        }

        // The value of x is (d(x⁺) - d(x⁻)) / 2
        let pos = DoubledNode::positive(var);
        let neg = DoubledNode::negative(var);

        let d_pos = self.distances.get(&pos)?;
        let d_neg = self.distances.get(&neg)?;

        Some((*d_pos - *d_neg) / Rational64::from_integer(2))
    }

    /// Get the model (all variable assignments)
    pub fn get_model(&self) -> HashMap<u32, Rational64> {
        let mut model = HashMap::new();

        if !self.distances_valid {
            return model;
        }

        for var in 0..self.graph.num_vars() {
            if let Some(value) = self.get_value(var) {
                model.insert(var, value);
            }
        }

        model
    }

    /// Get the implied upper bound for a variable
    pub fn get_upper_bound(&self, var: u32) -> Option<Rational64> {
        if !self.distances_valid {
            return None;
        }

        // Upper bound: d(x⁺) (distance to positive node)
        let pos = DoubledNode::positive(var);
        self.distances.get(&pos).copied()
    }

    /// Get the implied lower bound for a variable
    pub fn get_lower_bound(&self, var: u32) -> Option<Rational64> {
        if !self.distances_valid {
            return None;
        }

        // Lower bound: -d(x⁻)
        let neg = DoubledNode::negative(var);
        self.distances.get(&neg).map(|d| -*d)
    }

    /// Get statistics
    pub fn stats(&self) -> &UtvpiStats {
        &self.stats
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.graph.reset();
        self.distances.clear();
        self.parent_edge.clear();
        self.distances_valid = false;
    }

    /// Number of variables
    pub fn num_vars(&self) -> u32 {
        self.graph.num_vars()
    }

    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.graph.num_constraints()
    }

    /// Current decision level
    pub fn current_level(&self) -> u32 {
        self.graph.current_level()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rational64 {
        Rational64::from_integer(n)
    }

    #[test]
    fn test_solver_creation() {
        let solver = UtvpiSolver::new(true);
        assert_eq!(solver.num_vars(), 0);
        assert_eq!(solver.num_constraints(), 0);
    }

    #[test]
    fn test_satisfiable_diff() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        // x - y ≤ 5 and y - x ≤ 3
        solver.add_diff(x, y, r(5), origin);
        solver.add_diff(y, x, r(3), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }

    #[test]
    fn test_satisfiable_sum() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        // x + y ≤ 10
        solver.add_sum(x, y, r(10), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }

    #[test]
    fn test_unsatisfiable_diff() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        // x - y ≤ -1 and y - x ≤ -1
        // This means x < y and y < x, contradiction
        solver.add_diff(x, y, r(-1), origin);
        solver.add_diff(y, x, r(-1), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Conflict(_)));
    }

    #[test]
    fn test_unsatisfiable_sum() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        // x + y ≤ -1 and -x - y ≤ -1
        // This means x + y ≤ -1 and x + y ≥ 1, contradiction
        solver.add_sum(x, y, r(-1), origin);
        solver.add_neg_sum(x, y, r(-1), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Conflict(_)));
    }

    #[test]
    fn test_bounds() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));

        // x ≤ 10 and x ≥ 5 (i.e., -x ≤ -5)
        solver.add_upper(x, r(10), origin);
        solver.add_lower(x, r(-5), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }

    #[test]
    fn test_unsatisfiable_bounds() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));

        // x ≤ 5 and x ≥ 10 (i.e., -x ≤ -10)
        solver.add_upper(x, r(5), origin);
        solver.add_lower(x, r(-10), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Conflict(_)));
    }

    #[test]
    fn test_model_extraction() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        // x - y ≤ 5
        solver.add_diff(x, y, r(5), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));

        let model = solver.get_model();
        assert!(!model.is_empty());
    }

    #[test]
    fn test_push_pop() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        // Level 0: x - y ≤ 5
        solver.add_diff(x, y, r(5), origin);
        assert_eq!(solver.check(), UtvpiResult::Ok);

        solver.push();
        assert_eq!(solver.current_level(), 1);

        // Level 1: y - x ≤ -10 (would make x - y ≥ 10, conflict with ≤ 5)
        solver.add_diff(y, x, r(-10), origin);
        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Conflict(_)));

        // Pop back to level 0
        solver.pop(1);
        assert_eq!(solver.current_level(), 0);

        // Should be satisfiable again
        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }

    #[test]
    fn test_reset() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        solver.add_upper(x, r(5), origin);

        solver.reset();

        assert_eq!(solver.num_vars(), 0);
        assert_eq!(solver.num_constraints(), 0);
    }

    #[test]
    fn test_triangle() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));
        let z = solver.get_or_create_var(TermId::from(3u32));

        // x - y ≤ 3
        // y - z ≤ 2
        // z - x ≤ 1
        // Sum: 0 ≤ 6 (satisfiable)
        solver.add_diff(x, y, r(3), origin);
        solver.add_diff(y, z, r(2), origin);
        solver.add_diff(z, x, r(1), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }

    #[test]
    fn test_negative_triangle() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));
        let z = solver.get_or_create_var(TermId::from(3u32));

        // x - y ≤ -1
        // y - z ≤ -1
        // z - x ≤ -1
        // Sum: 0 ≤ -3 (unsatisfiable - negative cycle)
        solver.add_diff(x, y, r(-1), origin);
        solver.add_diff(y, z, r(-1), origin);
        solver.add_diff(z, x, r(-1), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Conflict(_)));
    }

    #[test]
    fn test_mixed_constraints() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        // x - y ≤ 5 (difference)
        // x + y ≤ 10 (sum)
        // x ≤ 8 (upper bound)
        solver.add_diff(x, y, r(5), origin);
        solver.add_sum(x, y, r(10), origin);
        solver.add_upper(x, r(8), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }

    #[test]
    fn test_bounds_extraction() {
        let mut solver = UtvpiSolver::new(true);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));

        // x ≤ 10
        solver.add_upper(x, r(10), origin);
        // x ≥ 3 (i.e., -x ≤ -3)
        solver.add_lower(x, r(-3), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));

        // Check bounds
        if let Some(upper) = solver.get_upper_bound(x) {
            assert!(upper <= r(10));
        }
        if let Some(lower) = solver.get_lower_bound(x) {
            assert!(lower >= r(3));
        }
    }

    #[test]
    fn test_spfa_mode() {
        let config = UtvpiConfig {
            use_spfa: true,
            ..Default::default()
        };
        let mut solver = UtvpiSolver::with_config(true, config);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        solver.add_diff(x, y, r(5), origin);
        solver.add_diff(y, x, r(3), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }

    #[test]
    fn test_bellman_ford_mode() {
        let config = UtvpiConfig {
            use_spfa: false,
            ..Default::default()
        };
        let mut solver = UtvpiSolver::with_config(true, config);
        let origin = TermId::from(100u32);

        let x = solver.get_or_create_var(TermId::from(1u32));
        let y = solver.get_or_create_var(TermId::from(2u32));

        solver.add_diff(x, y, r(5), origin);
        solver.add_diff(y, x, r(3), origin);

        let result = solver.check();
        assert!(matches!(result, UtvpiResult::Ok));
    }
}

impl PartialEq for UtvpiResult {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (UtvpiResult::Ok, UtvpiResult::Ok) => true,
            (UtvpiResult::Unknown, UtvpiResult::Unknown) => true,
            (UtvpiResult::Conflict(a), UtvpiResult::Conflict(b)) => a == b,
            _ => false,
        }
    }
}
