//! Dense Difference Logic Solver
//!
//! Uses a dense adjacency matrix for small problems (≤ 50 variables).
//! More cache-friendly than sparse representation for small graphs.

use super::bellman_ford::NegativeCycle;
use super::graph::{DiffConstraint, DiffVar};
use num_rational::Rational64;
use oxiz_core::ast::TermId;
use std::collections::HashMap;

/// Maximum number of variables for dense representation
const MAX_DENSE_VARS: usize = 50;

/// Infinity value for unreachable nodes
fn infinity() -> Rational64 {
    Rational64::new(i64::MAX / 2, 1)
}

/// Dense difference logic solver using Floyd-Warshall
#[derive(Debug)]
pub struct DenseDiffLogic {
    /// Number of variables
    num_vars: usize,
    /// Variable to index mapping
    var_to_idx: HashMap<TermId, usize>,
    /// Index to variable mapping
    idx_to_var: Vec<TermId>,
    /// Distance matrix: dist[i][j] = shortest path from i to j
    dist: Vec<Vec<Rational64>>,
    /// Next matrix for path reconstruction
    next: Vec<Vec<Option<usize>>>,
    /// Constraint origins for each edge
    edge_constraint: Vec<Vec<Option<usize>>>,
    /// All constraints
    constraints: Vec<DiffConstraint>,
    /// Is integer arithmetic
    is_integer: bool,
}

impl DenseDiffLogic {
    /// Create a new dense solver
    pub fn new(is_integer: bool) -> Self {
        Self {
            num_vars: 0,
            var_to_idx: HashMap::new(),
            idx_to_var: Vec::new(),
            dist: Vec::new(),
            next: Vec::new(),
            edge_constraint: Vec::new(),
            constraints: Vec::new(),
            is_integer,
        }
    }

    /// Check if the number of variables is suitable for dense representation
    pub fn is_suitable(num_vars: usize) -> bool {
        num_vars <= MAX_DENSE_VARS
    }

    /// Get or create index for a term
    pub fn get_or_create_idx(&mut self, term: TermId) -> usize {
        if let Some(&idx) = self.var_to_idx.get(&term) {
            return idx;
        }

        let idx = self.num_vars;
        self.num_vars += 1;
        self.var_to_idx.insert(term, idx);
        self.idx_to_var.push(term);

        // Expand matrices
        let inf = infinity();

        // Add new row
        let mut new_row = vec![inf; self.num_vars];
        new_row[idx] = Rational64::from_integer(0); // Self-loop is 0
        self.dist.push(new_row);
        self.next.push(vec![None; self.num_vars]);
        self.edge_constraint.push(vec![None; self.num_vars]);

        // Expand existing rows
        for i in 0..idx {
            self.dist[i].push(inf);
            self.next[i].push(None);
            self.edge_constraint[i].push(None);
        }

        idx
    }

    /// Get index for a term
    pub fn get_idx(&self, term: TermId) -> Option<usize> {
        self.var_to_idx.get(&term).copied()
    }

    /// Add a constraint x - y ≤ c
    pub fn add_constraint(&mut self, constraint: DiffConstraint) -> Option<NegativeCycle> {
        // Get indices
        let term_x = self.get_var_term(constraint.x)?;
        let term_y = self.get_var_term(constraint.y)?;

        let x_idx = self.get_or_create_idx(term_x);
        let y_idx = self.get_or_create_idx(term_y);

        let weight = constraint.effective_bound(self.is_integer);
        let constraint_idx = self.constraints.len();
        self.constraints.push(constraint);

        // Add edge y → x with weight c
        // (This represents x ≤ y + c)
        if weight < self.dist[y_idx][x_idx] {
            self.dist[y_idx][x_idx] = weight;
            self.next[y_idx][x_idx] = Some(x_idx);
            self.edge_constraint[y_idx][x_idx] = Some(constraint_idx);

            // Run incremental Floyd-Warshall update
            return self.incremental_update(y_idx, x_idx);
        }

        None
    }

    /// Get term for a DiffVar (helper for constraint processing)
    fn get_var_term(&self, var: DiffVar) -> Option<TermId> {
        // For now, we assume DiffVar ID maps directly to term ID
        // In practice, this should use a proper mapping
        Some(TermId::from(var.id()))
    }

    /// Incremental Floyd-Warshall update after adding edge (u, v)
    fn incremental_update(&mut self, u: usize, v: usize) -> Option<NegativeCycle> {
        let n = self.num_vars;
        let new_dist = self.dist[u][v];

        // Check for negative self-loop
        for i in 0..n {
            let via_uv = self.dist[i][u].checked_add(&new_dist)?;
            let via_uv_i = via_uv.checked_add(&self.dist[v][i])?;

            if via_uv_i < self.dist[i][i] {
                self.dist[i][i] = via_uv_i;
                if via_uv_i < Rational64::from_integer(0) {
                    // Negative cycle detected
                    return Some(self.extract_cycle(i));
                }
            }
        }

        // Update all pairs
        for i in 0..n {
            for j in 0..n {
                let dist_i_u = self.dist[i][u];
                let dist_v_j = self.dist[v][j];

                if dist_i_u < infinity() && dist_v_j < infinity() {
                    let new_path = dist_i_u + new_dist + dist_v_j;
                    if new_path < self.dist[i][j] {
                        self.dist[i][j] = new_path;
                        self.next[i][j] = self.next[i][u];
                    }
                }
            }
        }

        None
    }

    /// Extract a negative cycle starting from a node on the cycle
    fn extract_cycle(&self, start: usize) -> NegativeCycle {
        let mut cycle_edges = Vec::new();
        let mut total_weight = Rational64::from_integer(0);
        let mut current = start;
        let mut visited = vec![false; self.num_vars];

        loop {
            if visited[current] {
                break;
            }
            visited[current] = true;

            // Find next node in the cycle
            let mut found = false;
            for j in 0..self.num_vars {
                if self.dist[current][j] < infinity()
                    && j != current
                    && let Some(constraint_idx) = self.edge_constraint[current][j]
                {
                    // Check if this edge is part of a path back to start
                    let via_j = self.dist[current][j] + self.dist[j][start];
                    if via_j < infinity() {
                        cycle_edges.push(constraint_idx);
                        total_weight += self.dist[current][j];
                        current = j;
                        found = true;
                        break;
                    }
                }
            }

            if !found || current == start {
                break;
            }

            if cycle_edges.len() > self.num_vars {
                break;
            }
        }

        NegativeCycle::new(cycle_edges, total_weight)
    }

    /// Get the distance between two variables
    pub fn get_distance(&self, from: TermId, to: TermId) -> Option<Rational64> {
        let from_idx = self.var_to_idx.get(&from)?;
        let to_idx = self.var_to_idx.get(&to)?;

        let d = self.dist[*from_idx][*to_idx];
        if d < infinity() { Some(d) } else { None }
    }

    /// Check if the current constraint set is satisfiable
    pub fn check_sat(&self) -> bool {
        // Check for negative cycles (negative diagonal)
        for i in 0..self.num_vars {
            if self.dist[i][i] < Rational64::from_integer(0) {
                return false;
            }
        }
        true
    }

    /// Get a model (variable assignments)
    pub fn get_model(&self) -> HashMap<TermId, Rational64> {
        let mut model = HashMap::new();

        // Use distances from virtual source (node 0 or first node)
        // For simplicity, we assign each variable its shortest distance from a reference point
        if self.num_vars == 0 {
            return model;
        }

        // Use node 0 as reference (assigned value 0)
        let reference_term = self.idx_to_var[0];
        model.insert(reference_term, Rational64::from_integer(0));

        // Other variables: value = dist[0][i] (if reachable)
        for i in 1..self.num_vars {
            let term = self.idx_to_var[i];
            let d = self.dist[0][i];
            if d < infinity() {
                model.insert(term, d);
            } else {
                // Unreachable, assign 0
                model.insert(term, Rational64::from_integer(0));
            }
        }

        model
    }

    /// Number of variables
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.constraints.clear();

        // Reset distance matrix to initial state
        let inf = infinity();
        for i in 0..self.num_vars {
            for j in 0..self.num_vars {
                self.dist[i][j] = if i == j {
                    Rational64::from_integer(0)
                } else {
                    inf
                };
                self.next[i][j] = if i == j { Some(j) } else { None };
                self.edge_constraint[i][j] = None;
            }
        }
    }
}

/// Trait extension for Rational64
trait RationalExt {
    fn checked_add(&self, other: &Self) -> Option<Self>
    where
        Self: Sized;
}

impl RationalExt for Rational64 {
    fn checked_add(&self, other: &Self) -> Option<Self> {
        // Simple overflow check
        let inf = infinity();
        if *self >= inf || *other >= inf {
            return Some(inf);
        }
        Some(*self + *other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_creation() {
        let dense = DenseDiffLogic::new(true);
        assert_eq!(dense.num_vars(), 0);
        assert!(dense.check_sat());
    }

    #[test]
    fn test_is_suitable() {
        assert!(DenseDiffLogic::is_suitable(10));
        assert!(DenseDiffLogic::is_suitable(50));
        assert!(!DenseDiffLogic::is_suitable(51));
        assert!(!DenseDiffLogic::is_suitable(100));
    }

    #[test]
    fn test_add_variables() {
        let mut dense = DenseDiffLogic::new(true);
        let term1 = TermId::from(1u32);
        let term2 = TermId::from(2u32);

        let idx1 = dense.get_or_create_idx(term1);
        let idx2 = dense.get_or_create_idx(term2);

        assert_eq!(dense.num_vars(), 2);
        assert_ne!(idx1, idx2);
        assert_eq!(dense.get_idx(term1), Some(idx1));
        assert_eq!(dense.get_idx(term2), Some(idx2));
    }

    #[test]
    fn test_distance_query() {
        let mut dense = DenseDiffLogic::new(true);
        let term1 = TermId::from(1u32);
        let term2 = TermId::from(2u32);

        let _ = dense.get_or_create_idx(term1);
        let _ = dense.get_or_create_idx(term2);

        // Initially, only self-distances are 0
        assert_eq!(
            dense.get_distance(term1, term1),
            Some(Rational64::from_integer(0))
        );
        assert_eq!(dense.get_distance(term1, term2), None);
    }

    #[test]
    fn test_get_model() {
        let mut dense = DenseDiffLogic::new(true);
        let term1 = TermId::from(1u32);
        let term2 = TermId::from(2u32);

        let _ = dense.get_or_create_idx(term1);
        let _ = dense.get_or_create_idx(term2);

        let model = dense.get_model();
        assert_eq!(model.len(), 2);
        // First variable should be 0 (reference)
        assert_eq!(model.get(&term1), Some(&Rational64::from_integer(0)));
    }
}
