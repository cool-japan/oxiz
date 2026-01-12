//! Doubled Graph for UTVPI Constraints
//!
//! Represents UTVPI constraints using a doubled graph where each variable
//! x is represented by two nodes: x⁺ (positive) and x⁻ (negative).

use num_rational::Rational64;
use oxiz_core::ast::TermId;
use std::collections::HashMap;

/// Node in the doubled graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DoubledNode {
    /// Original variable ID
    pub var_id: u32,
    /// Is this the positive (true) or negative (false) node
    pub positive: bool,
}

impl DoubledNode {
    /// Create a positive node (x⁺)
    pub fn positive(var_id: u32) -> Self {
        Self {
            var_id,
            positive: true,
        }
    }

    /// Create a negative node (x⁻)
    pub fn negative(var_id: u32) -> Self {
        Self {
            var_id,
            positive: false,
        }
    }

    /// Get the complementary node (x⁺ → x⁻ or x⁻ → x⁺)
    pub fn complement(self) -> Self {
        Self {
            var_id: self.var_id,
            positive: !self.positive,
        }
    }

    /// Virtual source node
    pub const SOURCE: Self = Self {
        var_id: u32::MAX,
        positive: true,
    };

    /// Check if this is the source node
    pub fn is_source(self) -> bool {
        self.var_id == u32::MAX
    }
}

/// Sign of a coefficient in UTVPI constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    /// Coefficient is +1
    Positive,
    /// Coefficient is -1
    Negative,
    /// Coefficient is 0 (variable not present)
    Zero,
}

/// A UTVPI constraint: ax + by ≤ c where a, b ∈ {-1, 0, 1}
#[derive(Debug, Clone)]
pub struct UtConstraint {
    /// First variable (x)
    pub x: u32,
    /// Coefficient of x (-1, 0, or 1)
    pub a: Sign,
    /// Second variable (y)
    pub y: u32,
    /// Coefficient of y (-1, 0, or 1)
    pub b: Sign,
    /// Constant bound (c)
    pub bound: Rational64,
    /// Original term ID for explanations
    pub origin: TermId,
    /// Decision level when added
    pub level: u32,
    /// Is this a strict inequality (<)?
    pub strict: bool,
}

impl UtConstraint {
    /// Create a new UTVPI constraint ax + by ≤ c
    pub fn new(x: u32, a: Sign, y: u32, b: Sign, bound: Rational64, origin: TermId) -> Self {
        Self {
            x,
            a,
            y,
            b,
            bound,
            origin,
            level: 0,
            strict: false,
        }
    }

    /// Create x - y ≤ c (difference constraint)
    pub fn diff(x: u32, y: u32, bound: Rational64, origin: TermId) -> Self {
        Self::new(x, Sign::Positive, y, Sign::Negative, bound, origin)
    }

    /// Create x + y ≤ c (sum constraint)
    pub fn sum(x: u32, y: u32, bound: Rational64, origin: TermId) -> Self {
        Self::new(x, Sign::Positive, y, Sign::Positive, bound, origin)
    }

    /// Create -x - y ≤ c
    pub fn neg_sum(x: u32, y: u32, bound: Rational64, origin: TermId) -> Self {
        Self::new(x, Sign::Negative, y, Sign::Negative, bound, origin)
    }

    /// Create x ≤ c (unary constraint)
    pub fn upper(x: u32, bound: Rational64, origin: TermId) -> Self {
        Self::new(x, Sign::Positive, 0, Sign::Zero, bound, origin)
    }

    /// Create -x ≤ c (equivalent to x ≥ -c)
    pub fn lower(x: u32, bound: Rational64, origin: TermId) -> Self {
        Self::new(x, Sign::Negative, 0, Sign::Zero, bound, origin)
    }

    /// Get effective bound (for strict constraints, subtract 1 for integers)
    pub fn effective_bound(&self, is_integer: bool) -> Rational64 {
        if self.strict && is_integer {
            self.bound - Rational64::from_integer(1)
        } else {
            self.bound
        }
    }

    /// Check if this is a unary constraint (only one variable)
    pub fn is_unary(&self) -> bool {
        self.a == Sign::Zero || self.b == Sign::Zero
    }
}

/// Edge in the doubled graph
#[derive(Debug, Clone)]
pub struct UtEdge {
    /// Source node
    pub from: DoubledNode,
    /// Target node
    pub to: DoubledNode,
    /// Edge weight
    pub weight: Rational64,
    /// Constraint index for explanation
    pub constraint_idx: usize,
}

impl UtEdge {
    /// Create a new edge
    pub fn new(
        from: DoubledNode,
        to: DoubledNode,
        weight: Rational64,
        constraint_idx: usize,
    ) -> Self {
        Self {
            from,
            to,
            weight,
            constraint_idx,
        }
    }
}

/// Doubled graph for UTVPI constraints
#[derive(Debug)]
pub struct DoubledGraph {
    /// Number of variables
    num_vars: u32,
    /// Variable to term mapping
    var_to_term: HashMap<u32, TermId>,
    /// Term to variable mapping
    term_to_var: HashMap<TermId, u32>,
    /// Adjacency list: edges by source node
    edges: HashMap<DoubledNode, Vec<UtEdge>>,
    /// All constraints
    constraints: Vec<UtConstraint>,
    /// Active constraint indices by level
    active_by_level: Vec<Vec<usize>>,
    /// Current decision level
    current_level: u32,
    /// Is integer arithmetic
    is_integer: bool,
}

impl DoubledGraph {
    /// Create a new doubled graph
    pub fn new(is_integer: bool) -> Self {
        let mut edges = HashMap::new();
        edges.insert(DoubledNode::SOURCE, Vec::new());

        Self {
            num_vars: 0,
            var_to_term: HashMap::new(),
            term_to_var: HashMap::new(),
            edges,
            constraints: Vec::new(),
            active_by_level: vec![Vec::new()],
            current_level: 0,
            is_integer,
        }
    }

    /// Get or create a variable ID for a term
    pub fn get_or_create_var(&mut self, term: TermId) -> u32 {
        if let Some(&var) = self.term_to_var.get(&term) {
            return var;
        }

        let var = self.num_vars;
        self.num_vars += 1;
        self.var_to_term.insert(var, term);
        self.term_to_var.insert(term, var);

        // Create entries for both positive and negative nodes
        let pos = DoubledNode::positive(var);
        let neg = DoubledNode::negative(var);
        self.edges.entry(pos).or_default();
        self.edges.entry(neg).or_default();

        // Add edges from source to both nodes with weight 0
        self.edges
            .get_mut(&DoubledNode::SOURCE)
            .expect("Source should exist")
            .push(UtEdge::new(
                DoubledNode::SOURCE,
                pos,
                Rational64::from_integer(0),
                0,
            ));
        self.edges
            .get_mut(&DoubledNode::SOURCE)
            .expect("Source should exist")
            .push(UtEdge::new(
                DoubledNode::SOURCE,
                neg,
                Rational64::from_integer(0),
                0,
            ));

        var
    }

    /// Get the term for a variable
    pub fn get_term(&self, var: u32) -> Option<TermId> {
        self.var_to_term.get(&var).copied()
    }

    /// Get the variable for a term
    pub fn get_var(&self, term: TermId) -> Option<u32> {
        self.term_to_var.get(&term).copied()
    }

    /// Add a UTVPI constraint
    pub fn add_constraint(&mut self, constraint: UtConstraint) -> usize {
        let idx = self.constraints.len();
        let weight = constraint.effective_bound(self.is_integer);

        // Translate constraint to edges based on signs
        match (constraint.a, constraint.b) {
            // x - y ≤ c: edges (y⁺ → x⁺, c) and (x⁻ → y⁻, c)
            (Sign::Positive, Sign::Negative) => {
                self.add_edge(
                    DoubledNode::positive(constraint.y),
                    DoubledNode::positive(constraint.x),
                    weight,
                    idx,
                );
                self.add_edge(
                    DoubledNode::negative(constraint.x),
                    DoubledNode::negative(constraint.y),
                    weight,
                    idx,
                );
            }
            // -x + y ≤ c (same as y - x ≤ c): edges (x⁺ → y⁺, c) and (y⁻ → x⁻, c)
            (Sign::Negative, Sign::Positive) => {
                self.add_edge(
                    DoubledNode::positive(constraint.x),
                    DoubledNode::positive(constraint.y),
                    weight,
                    idx,
                );
                self.add_edge(
                    DoubledNode::negative(constraint.y),
                    DoubledNode::negative(constraint.x),
                    weight,
                    idx,
                );
            }
            // x + y ≤ c: edges (y⁻ → x⁺, c) and (x⁻ → y⁺, c)
            (Sign::Positive, Sign::Positive) => {
                self.add_edge(
                    DoubledNode::negative(constraint.y),
                    DoubledNode::positive(constraint.x),
                    weight,
                    idx,
                );
                self.add_edge(
                    DoubledNode::negative(constraint.x),
                    DoubledNode::positive(constraint.y),
                    weight,
                    idx,
                );
            }
            // -x - y ≤ c: edges (y⁺ → x⁻, c) and (x⁺ → y⁻, c)
            (Sign::Negative, Sign::Negative) => {
                self.add_edge(
                    DoubledNode::positive(constraint.y),
                    DoubledNode::negative(constraint.x),
                    weight,
                    idx,
                );
                self.add_edge(
                    DoubledNode::positive(constraint.x),
                    DoubledNode::negative(constraint.y),
                    weight,
                    idx,
                );
            }
            // x ≤ c (unary): edge (x⁻ → x⁺, 2c)
            (Sign::Positive, Sign::Zero) => {
                self.add_edge(
                    DoubledNode::negative(constraint.x),
                    DoubledNode::positive(constraint.x),
                    weight * Rational64::from_integer(2),
                    idx,
                );
            }
            // -x ≤ c (unary): edge (x⁺ → x⁻, 2c)
            (Sign::Negative, Sign::Zero) => {
                self.add_edge(
                    DoubledNode::positive(constraint.x),
                    DoubledNode::negative(constraint.x),
                    weight * Rational64::from_integer(2),
                    idx,
                );
            }
            // y ≤ c (unary, x coefficient is zero)
            (Sign::Zero, Sign::Positive) => {
                self.add_edge(
                    DoubledNode::negative(constraint.y),
                    DoubledNode::positive(constraint.y),
                    weight * Rational64::from_integer(2),
                    idx,
                );
            }
            // -y ≤ c (unary)
            (Sign::Zero, Sign::Negative) => {
                self.add_edge(
                    DoubledNode::positive(constraint.y),
                    DoubledNode::negative(constraint.y),
                    weight * Rational64::from_integer(2),
                    idx,
                );
            }
            // Both zero - degenerate case (0 ≤ c)
            (Sign::Zero, Sign::Zero) => {
                // No edges needed, just check if c >= 0
            }
        }

        // Track for backtracking
        while self.active_by_level.len() <= self.current_level as usize {
            self.active_by_level.push(Vec::new());
        }
        self.active_by_level[self.current_level as usize].push(idx);

        self.constraints.push(constraint);
        idx
    }

    /// Add an edge to the graph
    fn add_edge(&mut self, from: DoubledNode, to: DoubledNode, weight: Rational64, idx: usize) {
        self.edges
            .entry(from)
            .or_default()
            .push(UtEdge::new(from, to, weight, idx));
    }

    /// Get edges from a node
    pub fn get_edges(&self, from: DoubledNode) -> impl Iterator<Item = &UtEdge> {
        self.edges.get(&from).into_iter().flatten()
    }

    /// Get all edges
    pub fn all_edges(&self) -> impl Iterator<Item = &UtEdge> {
        self.edges.values().flatten()
    }

    /// Get a constraint by index
    pub fn get_constraint(&self, idx: usize) -> Option<&UtConstraint> {
        self.constraints.get(idx)
    }

    /// Number of variables
    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    /// Number of nodes (2 * num_vars)
    pub fn num_nodes(&self) -> u32 {
        self.num_vars * 2
    }

    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// All positive nodes
    pub fn positive_nodes(&self) -> impl Iterator<Item = DoubledNode> + '_ {
        (0..self.num_vars).map(DoubledNode::positive)
    }

    /// All negative nodes
    pub fn negative_nodes(&self) -> impl Iterator<Item = DoubledNode> + '_ {
        (0..self.num_vars).map(DoubledNode::negative)
    }

    /// All nodes
    pub fn all_nodes(&self) -> impl Iterator<Item = DoubledNode> + '_ {
        self.positive_nodes().chain(self.negative_nodes())
    }

    /// Push a new decision level
    pub fn push(&mut self) {
        self.current_level += 1;
        while self.active_by_level.len() <= self.current_level as usize {
            self.active_by_level.push(Vec::new());
        }
    }

    /// Pop to a previous level
    pub fn pop(&mut self, levels: u32) {
        if levels == 0 {
            return;
        }

        let target_level = self.current_level.saturating_sub(levels);

        // Remove edges for constraints at higher levels
        for level in (target_level + 1)..=self.current_level {
            if let Some(indices) = self.active_by_level.get(level as usize) {
                for &idx in indices {
                    // Remove all edges with this constraint index
                    for edges in self.edges.values_mut() {
                        edges.retain(|e| e.constraint_idx != idx);
                    }
                }
            }
        }

        self.active_by_level.truncate(target_level as usize + 1);
        self.current_level = target_level;
    }

    /// Current decision level
    pub fn current_level(&self) -> u32 {
        self.current_level
    }

    /// Is integer arithmetic
    pub fn is_integer(&self) -> bool {
        self.is_integer
    }

    /// Reset the graph
    pub fn reset(&mut self) {
        self.num_vars = 0;
        self.var_to_term.clear();
        self.term_to_var.clear();
        self.edges.clear();
        self.edges.insert(DoubledNode::SOURCE, Vec::new());
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
    fn test_doubled_node() {
        let pos = DoubledNode::positive(5);
        let neg = DoubledNode::negative(5);

        assert!(pos.positive);
        assert!(!neg.positive);
        assert_eq!(pos.var_id, neg.var_id);
        assert_eq!(pos.complement(), neg);
        assert_eq!(neg.complement(), pos);
    }

    #[test]
    fn test_graph_creation() {
        let graph = DoubledGraph::new(true);
        assert_eq!(graph.num_vars(), 0);
        assert!(graph.is_integer());
    }

    #[test]
    fn test_add_variable() {
        let mut graph = DoubledGraph::new(true);
        let term1 = TermId::from(1u32);
        let term2 = TermId::from(2u32);

        let var1 = graph.get_or_create_var(term1);
        let var2 = graph.get_or_create_var(term2);

        assert_eq!(graph.num_vars(), 2);
        assert_ne!(var1, var2);
        assert_eq!(graph.get_term(var1), Some(term1));
        assert_eq!(graph.get_var(term1), Some(var1));
    }

    #[test]
    fn test_diff_constraint() {
        let mut graph = DoubledGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);

        // x - y ≤ 5
        let constraint = UtConstraint::diff(x, y, Rational64::from_integer(5), origin);
        graph.add_constraint(constraint);

        assert_eq!(graph.num_constraints(), 1);

        // Check edges: (y⁺ → x⁺) and (x⁻ → y⁻)
        let edges_from_y_pos: Vec<_> = graph.get_edges(DoubledNode::positive(y)).collect();
        assert!(!edges_from_y_pos.is_empty());
    }

    #[test]
    fn test_sum_constraint() {
        let mut graph = DoubledGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);

        // x + y ≤ 10
        let constraint = UtConstraint::sum(x, y, Rational64::from_integer(10), origin);
        graph.add_constraint(constraint);

        assert_eq!(graph.num_constraints(), 1);

        // Check edges: (y⁻ → x⁺) and (x⁻ → y⁺)
        let edges_from_y_neg: Vec<_> = graph.get_edges(DoubledNode::negative(y)).collect();
        assert!(!edges_from_y_neg.is_empty());
    }

    #[test]
    fn test_unary_constraint() {
        let mut graph = DoubledGraph::new(true);
        let term_x = TermId::from(1u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);

        // x ≤ 7
        let constraint = UtConstraint::upper(x, Rational64::from_integer(7), origin);
        graph.add_constraint(constraint);

        assert_eq!(graph.num_constraints(), 1);

        // Check edge: (x⁻ → x⁺, 14) (doubled for unary)
        let edges_from_x_neg: Vec<_> = graph.get_edges(DoubledNode::negative(x)).collect();
        assert!(!edges_from_x_neg.is_empty());
        assert_eq!(edges_from_x_neg[0].weight, Rational64::from_integer(14));
    }

    #[test]
    fn test_push_pop() {
        let mut graph = DoubledGraph::new(true);
        let term_x = TermId::from(1u32);
        let term_y = TermId::from(2u32);
        let origin = TermId::from(100u32);

        let x = graph.get_or_create_var(term_x);
        let y = graph.get_or_create_var(term_y);

        // Level 0
        graph.add_constraint(UtConstraint::diff(
            x,
            y,
            Rational64::from_integer(5),
            origin,
        ));

        graph.push();
        assert_eq!(graph.current_level(), 1);

        // Level 1
        graph.add_constraint(UtConstraint::sum(
            x,
            y,
            Rational64::from_integer(10),
            origin,
        ));
        assert_eq!(graph.num_constraints(), 2);

        graph.pop(1);
        assert_eq!(graph.current_level(), 0);
    }
}
