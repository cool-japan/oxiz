//! Constraint Graph for Difference Logic
//!
//! Represents difference constraints as a weighted directed graph.

use num_rational::Rational64;
use oxiz_core::ast::TermId;
use std::collections::HashMap;

/// Variable identifier in difference logic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiffVar(pub u32);

impl DiffVar {
    /// Virtual source node for Bellman-Ford
    pub const SOURCE: Self = Self(u32::MAX);

    /// Create a new variable
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the variable ID
    pub fn id(self) -> u32 {
        self.0
    }

    /// Check if this is the source node
    pub fn is_source(self) -> bool {
        self == Self::SOURCE
    }
}

impl From<u32> for DiffVar {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

/// Type of difference constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstraintType {
    /// x - y ≤ c (non-strict)
    LeqConst,
    /// x - y < c (strict) - converted to ≤ (c - ε) for integers
    LtConst,
}

/// A difference constraint: x - y ≤ c or x - y < c
#[derive(Debug, Clone)]
pub struct DiffConstraint {
    /// Left variable (x in x - y ≤ c)
    pub x: DiffVar,
    /// Right variable (y in x - y ≤ c)
    pub y: DiffVar,
    /// Constant bound (c)
    pub bound: Rational64,
    /// Constraint type (≤ or <)
    pub constraint_type: ConstraintType,
    /// Original term ID for explanations
    pub origin: TermId,
    /// Decision level when added
    pub level: u32,
    /// Whether this is asserted (not just propagated)
    pub asserted: bool,
}

impl DiffConstraint {
    /// Create a new constraint x - y ≤ c
    pub fn new_leq(x: DiffVar, y: DiffVar, bound: Rational64, origin: TermId) -> Self {
        Self {
            x,
            y,
            bound,
            constraint_type: ConstraintType::LeqConst,
            origin,
            level: 0,
            asserted: false,
        }
    }

    /// Create a new constraint x - y < c
    pub fn new_lt(x: DiffVar, y: DiffVar, bound: Rational64, origin: TermId) -> Self {
        Self {
            x,
            y,
            bound,
            constraint_type: ConstraintType::LtConst,
            origin,
            level: 0,
            asserted: false,
        }
    }

    /// Get the effective bound (for strict constraints, we use bound - 1 for integers)
    pub fn effective_bound(&self, is_integer: bool) -> Rational64 {
        match self.constraint_type {
            ConstraintType::LeqConst => self.bound,
            ConstraintType::LtConst => {
                if is_integer {
                    // For integers: x < c means x ≤ c - 1
                    self.bound - Rational64::from_integer(1)
                } else {
                    // For reals: keep strict bound (handled specially)
                    self.bound
                }
            }
        }
    }
}

/// Edge in the constraint graph
#[derive(Debug, Clone)]
pub struct DiffEdge {
    /// Source node (y in the constraint x - y ≤ c)
    pub from: DiffVar,
    /// Target node (x in the constraint x - y ≤ c)
    pub to: DiffVar,
    /// Edge weight (c in the constraint x - y ≤ c)
    pub weight: Rational64,
    /// Constraint index for explanation
    pub constraint_idx: usize,
    /// Whether this is a strict edge (for real arithmetic)
    pub strict: bool,
}

impl DiffEdge {
    /// Create a new edge
    pub fn new(from: DiffVar, to: DiffVar, weight: Rational64, constraint_idx: usize) -> Self {
        Self {
            from,
            to,
            weight,
            constraint_idx,
            strict: false,
        }
    }

    /// Create a new strict edge
    pub fn new_strict(
        from: DiffVar,
        to: DiffVar,
        weight: Rational64,
        constraint_idx: usize,
    ) -> Self {
        Self {
            from,
            to,
            weight,
            constraint_idx,
            strict: true,
        }
    }
}

/// Constraint graph for difference logic
///
/// Represents constraints as a weighted directed graph where:
/// - Each variable is a node
/// - Constraint x - y ≤ c is edge (y → x) with weight c
#[derive(Debug, Clone)]
pub struct ConstraintGraph {
    /// Number of variables
    num_vars: u32,
    /// Variable to node mapping
    var_to_node: HashMap<TermId, DiffVar>,
    /// Node to variable mapping
    node_to_var: HashMap<DiffVar, TermId>,
    /// Adjacency list: edges[node] = list of outgoing edges
    edges: HashMap<DiffVar, Vec<DiffEdge>>,
    /// All constraints (for backtracking)
    constraints: Vec<DiffConstraint>,
    /// Active constraint indices by level (for incremental solving)
    active_by_level: Vec<Vec<usize>>,
    /// Current decision level
    current_level: u32,
    /// Whether this is integer arithmetic
    is_integer: bool,
}

impl ConstraintGraph {
    /// Create a new constraint graph
    pub fn new(is_integer: bool) -> Self {
        let mut edges = HashMap::new();
        // Always add edges from source
        edges.insert(DiffVar::SOURCE, Vec::new());

        Self {
            num_vars: 0,
            var_to_node: HashMap::new(),
            node_to_var: HashMap::new(),
            edges,
            constraints: Vec::new(),
            active_by_level: vec![Vec::new()],
            current_level: 0,
            is_integer,
        }
    }

    /// Get or create a node for a term
    pub fn get_or_create_var(&mut self, term: TermId) -> DiffVar {
        if let Some(&var) = self.var_to_node.get(&term) {
            return var;
        }

        let var = DiffVar::new(self.num_vars);
        self.num_vars += 1;
        self.var_to_node.insert(term, var);
        self.node_to_var.insert(var, term);
        self.edges.insert(var, Vec::new());

        // Add edge from source to new node with weight 0
        self.edges
            .get_mut(&DiffVar::SOURCE)
            .expect("Source node should exist")
            .push(DiffEdge::new(
                DiffVar::SOURCE,
                var,
                Rational64::from_integer(0),
                0,
            ));

        var
    }

    /// Get the term for a variable
    pub fn get_term(&self, var: DiffVar) -> Option<TermId> {
        self.node_to_var.get(&var).copied()
    }

    /// Get the variable for a term
    pub fn get_var(&self, term: TermId) -> Option<DiffVar> {
        self.var_to_node.get(&term).copied()
    }

    /// Add a constraint x - y ≤ c
    ///
    /// Creates edge (y → x) with weight c
    pub fn add_constraint(&mut self, constraint: DiffConstraint) -> usize {
        let idx = self.constraints.len();

        // Get effective bound
        let weight = constraint.effective_bound(self.is_integer);

        // Create edge from y to x with weight c
        // This represents: x ≤ y + c, i.e., dist[x] ≤ dist[y] + c
        let edge = if constraint.constraint_type == ConstraintType::LtConst && !self.is_integer {
            DiffEdge::new_strict(constraint.y, constraint.x, weight, idx)
        } else {
            DiffEdge::new(constraint.y, constraint.x, weight, idx)
        };

        self.edges.entry(constraint.y).or_default().push(edge);

        // Track constraint for backtracking
        while self.active_by_level.len() <= self.current_level as usize {
            self.active_by_level.push(Vec::new());
        }
        self.active_by_level[self.current_level as usize].push(idx);

        self.constraints.push(constraint);
        idx
    }

    /// Get all edges from a node
    pub fn get_edges(&self, from: DiffVar) -> impl Iterator<Item = &DiffEdge> {
        self.edges.get(&from).into_iter().flatten()
    }

    /// Get all edges in the graph
    pub fn all_edges(&self) -> impl Iterator<Item = &DiffEdge> {
        self.edges.values().flatten()
    }

    /// Get a constraint by index
    pub fn get_constraint(&self, idx: usize) -> Option<&DiffConstraint> {
        self.constraints.get(idx)
    }

    /// Get all constraints
    pub fn constraints(&self) -> &[DiffConstraint] {
        &self.constraints
    }

    /// Number of variables (excluding source)
    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// All variable nodes
    pub fn vars(&self) -> impl Iterator<Item = DiffVar> + '_ {
        (0..self.num_vars).map(DiffVar::new)
    }

    /// All nodes including source
    pub fn nodes(&self) -> impl Iterator<Item = DiffVar> + '_ {
        std::iter::once(DiffVar::SOURCE).chain(self.vars())
    }

    /// Push a new decision level
    pub fn push(&mut self) {
        self.current_level += 1;
        while self.active_by_level.len() <= self.current_level as usize {
            self.active_by_level.push(Vec::new());
        }
    }

    /// Pop to a previous decision level
    pub fn pop(&mut self, levels: u32) {
        if levels == 0 {
            return;
        }

        let target_level = self.current_level.saturating_sub(levels);

        // Remove edges for constraints at higher levels
        for level in (target_level + 1)..=self.current_level {
            if let Some(indices) = self.active_by_level.get(level as usize) {
                for &idx in indices {
                    if let Some(constraint) = self.constraints.get(idx) {
                        // Remove the edge for this constraint
                        if let Some(edges) = self.edges.get_mut(&constraint.y) {
                            edges.retain(|e| e.constraint_idx != idx);
                        }
                    }
                }
            }
        }

        // Truncate active_by_level and constraints
        self.active_by_level.truncate(target_level as usize + 1);

        // Note: We keep constraints in the vector but they're effectively deactivated
        // because their edges are removed

        self.current_level = target_level;
    }

    /// Current decision level
    pub fn current_level(&self) -> u32 {
        self.current_level
    }

    /// Is integer arithmetic?
    pub fn is_integer(&self) -> bool {
        self.is_integer
    }

    /// Clear all constraints (for reset)
    pub fn clear(&mut self) {
        self.constraints.clear();
        self.active_by_level.clear();
        self.active_by_level.push(Vec::new());
        self.current_level = 0;

        // Clear all edges except source → variable edges
        for (var, edges) in &mut self.edges {
            if var.is_source() {
                // Keep source edges but reset to weight 0
                edges.clear();
                for node in 0..self.num_vars {
                    edges.push(DiffEdge::new(
                        DiffVar::SOURCE,
                        DiffVar::new(node),
                        Rational64::from_integer(0),
                        0,
                    ));
                }
            } else {
                edges.clear();
            }
        }
    }

    /// Full reset including variables
    pub fn reset(&mut self) {
        self.num_vars = 0;
        self.var_to_node.clear();
        self.node_to_var.clear();
        self.edges.clear();
        self.edges.insert(DiffVar::SOURCE, Vec::new());
        self.constraints.clear();
        self.active_by_level.clear();
        self.active_by_level.push(Vec::new());
        self.current_level = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_graph_creation() {
        let graph = ConstraintGraph::new(true);
        assert_eq!(graph.num_vars(), 0);
        assert!(graph.is_integer());
    }

    #[test]
    fn test_variable_creation() {
        let mut graph = ConstraintGraph::new(true);
        let term1 = TermId::from(1u32);
        let term2 = TermId::from(2u32);

        let var1 = graph.get_or_create_var(term1);
        let var2 = graph.get_or_create_var(term2);

        assert_eq!(graph.num_vars(), 2);
        assert_ne!(var1, var2);
        assert_eq!(graph.get_var(term1), Some(var1));
        assert_eq!(graph.get_term(var1), Some(term1));
    }

    #[test]
    fn test_add_constraint() {
        let mut graph = ConstraintGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);

        // Add constraint x - y ≤ 5
        let constraint = DiffConstraint::new_leq(x, y, Rational64::from_integer(5), origin);
        let idx = graph.add_constraint(constraint);

        assert_eq!(graph.num_constraints(), 1);
        assert!(graph.get_constraint(idx).is_some());

        // Check that edge y → x exists with weight 5
        let edges: Vec<_> = graph.get_edges(y).collect();
        assert!(!edges.is_empty());
        assert_eq!(edges[0].to, x);
        assert_eq!(edges[0].weight, Rational64::from_integer(5));
    }

    #[test]
    fn test_strict_constraint_integer() {
        let mut graph = ConstraintGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);

        // Add constraint x - y < 5 (becomes x - y ≤ 4 for integers)
        let constraint = DiffConstraint::new_lt(x, y, Rational64::from_integer(5), origin);
        let effective = constraint.effective_bound(true);

        assert_eq!(effective, Rational64::from_integer(4));
    }

    #[test]
    fn test_push_pop() {
        let mut graph = ConstraintGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);

        // Level 0: add constraint x - y ≤ 5
        let constraint1 = DiffConstraint::new_leq(x, y, Rational64::from_integer(5), origin);
        graph.add_constraint(constraint1);
        assert_eq!(graph.num_constraints(), 1);

        // Push to level 1
        graph.push();
        assert_eq!(graph.current_level(), 1);

        // Level 1: add constraint x - y ≤ 3
        let constraint2 = DiffConstraint::new_leq(x, y, Rational64::from_integer(3), origin);
        graph.add_constraint(constraint2);
        assert_eq!(graph.num_constraints(), 2);

        // Pop to level 0
        graph.pop(1);
        assert_eq!(graph.current_level(), 0);

        // Edge for constraint2 should be removed
        let edges: Vec<_> = graph
            .get_edges(y)
            .filter(|e| e.constraint_idx == 1)
            .collect();
        assert!(edges.is_empty());
    }
}
