//! Difference Logic Theory Solver
//!
//! Main solver implementation that integrates with the CDCL(T) framework.

use super::bellman_ford::{BellmanFord, BellmanFordResult, NegativeCycle, Spfa};
use super::dense::DenseDiffLogic;
use super::graph::{ConstraintGraph, ConstraintType, DiffConstraint, DiffVar};
use num_rational::Rational64;
use oxiz_core::ast::TermId;
use std::collections::HashMap;

/// Configuration for the difference logic solver
#[derive(Debug, Clone)]
pub struct DiffLogicConfig {
    /// Use SPFA instead of standard Bellman-Ford
    pub use_spfa: bool,
    /// Use dense representation for small problems
    pub use_dense: bool,
    /// Threshold for switching to dense representation
    pub dense_threshold: usize,
    /// Enable theory propagation
    pub propagate: bool,
    /// Maximum propagation rounds per call
    pub max_propagation_rounds: usize,
}

impl Default for DiffLogicConfig {
    fn default() -> Self {
        Self {
            use_spfa: true,
            use_dense: true,
            dense_threshold: 50,
            propagate: true,
            max_propagation_rounds: 10,
        }
    }
}

/// Statistics for the difference logic solver
#[derive(Debug, Clone, Default)]
pub struct DiffLogicStats {
    /// Number of constraints added
    pub constraints_added: usize,
    /// Number of propagations
    pub propagations: usize,
    /// Number of conflicts detected
    pub conflicts: usize,
    /// Number of consistency checks
    pub checks: usize,
    /// Number of model queries
    pub model_queries: usize,
}

/// Result of a difference logic operation
#[derive(Debug, Clone)]
pub enum DiffLogicResult {
    /// Operation succeeded
    Ok,
    /// Conflict detected with explanation
    Conflict(Vec<TermId>),
    /// Theory propagation (implied constraint)
    Propagation {
        /// The implied constraint
        implied: TermId,
        /// Reason for the implication
        reason: Vec<TermId>,
    },
}

/// Difference Logic Theory Solver
///
/// Handles constraints of the form x - y ≤ c (or x - y < c).
#[derive(Debug)]
pub struct DiffLogicSolver {
    /// Configuration
    config: DiffLogicConfig,
    /// Constraint graph (sparse representation)
    graph: ConstraintGraph,
    /// Dense solver (for small problems)
    dense: Option<DenseDiffLogic>,
    /// Bellman-Ford solver
    bf: BellmanFord,
    /// SPFA solver
    spfa: Spfa,
    /// Current distances (cached from last check)
    distances: HashMap<DiffVar, Rational64>,
    /// Whether distances are up-to-date
    distances_valid: bool,
    /// Statistics
    stats: DiffLogicStats,
    /// Decision level stack for backtracking
    level_stack: Vec<usize>,
    /// Current decision level
    current_level: u32,
    /// Pending constraints to process
    pending: Vec<usize>,
    /// Term to constraint mapping
    term_to_constraint: HashMap<TermId, Vec<usize>>,
}

impl DiffLogicSolver {
    /// Create a new solver with default configuration
    pub fn new(is_integer: bool) -> Self {
        Self::with_config(is_integer, DiffLogicConfig::default())
    }

    /// Create a new solver with custom configuration
    pub fn with_config(is_integer: bool, config: DiffLogicConfig) -> Self {
        Self {
            graph: ConstraintGraph::new(is_integer),
            dense: if config.use_dense {
                Some(DenseDiffLogic::new(is_integer))
            } else {
                None
            },
            bf: BellmanFord::new(),
            spfa: Spfa::new(),
            distances: HashMap::new(),
            distances_valid: false,
            stats: DiffLogicStats::default(),
            level_stack: vec![0],
            current_level: 0,
            pending: Vec::new(),
            term_to_constraint: HashMap::new(),
            config,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &DiffLogicConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &DiffLogicStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DiffLogicStats::default();
    }

    /// Register a variable (term)
    pub fn register_var(&mut self, term: TermId) -> DiffVar {
        self.distances_valid = false;
        self.graph.get_or_create_var(term)
    }

    /// Add a constraint x - y ≤ c
    pub fn add_leq(
        &mut self,
        x: TermId,
        y: TermId,
        c: Rational64,
        origin: TermId,
    ) -> DiffLogicResult {
        self.add_constraint_internal(x, y, c, ConstraintType::LeqConst, origin)
    }

    /// Add a constraint x - y < c
    pub fn add_lt(
        &mut self,
        x: TermId,
        y: TermId,
        c: Rational64,
        origin: TermId,
    ) -> DiffLogicResult {
        self.add_constraint_internal(x, y, c, ConstraintType::LtConst, origin)
    }

    /// Internal method to add a constraint
    fn add_constraint_internal(
        &mut self,
        x_term: TermId,
        y_term: TermId,
        c: Rational64,
        constraint_type: ConstraintType,
        origin: TermId,
    ) -> DiffLogicResult {
        self.stats.constraints_added += 1;
        self.distances_valid = false;

        // Get or create variables
        let x = self.graph.get_or_create_var(x_term);
        let y = self.graph.get_or_create_var(y_term);

        // Create constraint
        let mut constraint = match constraint_type {
            ConstraintType::LeqConst => DiffConstraint::new_leq(x, y, c, origin),
            ConstraintType::LtConst => DiffConstraint::new_lt(x, y, c, origin),
        };
        constraint.level = self.current_level;
        constraint.asserted = true;

        // Add to graph
        let idx = self.graph.add_constraint(constraint);

        // Track constraint by origin term
        self.term_to_constraint.entry(origin).or_default().push(idx);

        // Add to pending for next propagation
        self.pending.push(idx);

        // Quick check: does this create an obvious conflict?
        // A simple heuristic: if x == y and c < 0, immediate conflict
        if x == y && c < Rational64::from_integer(0) {
            self.stats.conflicts += 1;
            return DiffLogicResult::Conflict(vec![origin]);
        }

        DiffLogicResult::Ok
    }

    /// Check consistency of current constraints
    pub fn check(&mut self) -> DiffLogicResult {
        self.stats.checks += 1;

        // Always use Bellman-Ford/SPFA for now (dense solver is for future optimization)
        // Run Bellman-Ford or SPFA
        let result = if self.config.use_spfa {
            self.spfa.run(&self.graph)
        } else {
            self.bf.run(&self.graph)
        };

        match result {
            BellmanFordResult::Distances(dists) => {
                // Convert to our format
                self.distances = dists;
                self.distances_valid = true;
                DiffLogicResult::Ok
            }
            BellmanFordResult::NegativeCycle(cycle) => {
                self.stats.conflicts += 1;
                // Extract conflict clause from cycle
                let conflict = self.cycle_to_conflict(&cycle);
                DiffLogicResult::Conflict(conflict)
            }
        }
    }

    /// Convert a negative cycle to a conflict clause
    fn cycle_to_conflict(&self, cycle: &NegativeCycle) -> Vec<TermId> {
        let mut conflict = Vec::new();

        for &idx in cycle.constraint_indices() {
            if let Some(constraint) = self.graph.get_constraint(idx) {
                conflict.push(constraint.origin);
            }
        }

        conflict
    }

    /// Propagate implied bounds
    pub fn propagate(&mut self) -> Vec<DiffLogicResult> {
        let mut results = Vec::new();

        if !self.config.propagate {
            return results;
        }

        // Ensure distances are computed
        if !self.distances_valid
            && let DiffLogicResult::Conflict(c) = self.check()
        {
            results.push(DiffLogicResult::Conflict(c));
            return results;
        }

        // Theory propagation: if dist[x] - dist[y] ≤ c is tighter
        // than an unasserted constraint, propagate it
        // (This is a simplified version - full implementation would
        // require tracking unasserted constraints)

        self.pending.clear();
        results
    }

    /// Get the current value of a variable in the model
    pub fn get_value(&mut self, term: TermId) -> Option<Rational64> {
        self.stats.model_queries += 1;

        if !self.distances_valid
            && let DiffLogicResult::Conflict(_) = self.check()
        {
            return None;
        }

        if let Some(var) = self.graph.get_var(term) {
            self.distances.get(&var).copied()
        } else {
            None
        }
    }

    /// Get a complete model
    pub fn get_model(&mut self) -> HashMap<TermId, Rational64> {
        let mut model = HashMap::new();

        if !self.distances_valid
            && let DiffLogicResult::Conflict(_) = self.check()
        {
            return model;
        }

        for (var, dist) in &self.distances {
            if !var.is_source()
                && let Some(term) = self.graph.get_term(*var)
            {
                model.insert(term, *dist);
            }
        }

        model
    }

    /// Push a new decision level
    pub fn push(&mut self) {
        self.current_level += 1;
        self.graph.push();
        self.level_stack.push(self.graph.num_constraints());
    }

    /// Pop to a previous decision level
    pub fn pop(&mut self, levels: u32) {
        if levels == 0 {
            return;
        }

        self.graph.pop(levels);
        self.current_level = self.current_level.saturating_sub(levels);

        // Truncate level stack
        let target = (self.current_level + 1) as usize;
        if target < self.level_stack.len() {
            self.level_stack.truncate(target);
        }

        self.distances_valid = false;
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.graph.reset();
        if let Some(ref mut dense) = self.dense {
            dense.clear();
        }
        self.distances.clear();
        self.distances_valid = false;
        self.pending.clear();
        self.term_to_constraint.clear();
        self.level_stack = vec![0];
        self.current_level = 0;
        self.reset_stats();
    }

    /// Get the explanation for a constraint
    pub fn explain(&self, constraint_origin: TermId) -> Vec<TermId> {
        let mut explanation = Vec::new();

        if let Some(indices) = self.term_to_constraint.get(&constraint_origin) {
            for &idx in indices {
                if let Some(constraint) = self.graph.get_constraint(idx) {
                    // For a difference constraint, the explanation is the path
                    // from y to x in the constraint graph
                    // Simplified: just return the constraint itself
                    explanation.push(constraint.origin);
                }
            }
        }

        explanation
    }

    /// Check if a potential constraint would cause a conflict
    pub fn would_conflict(&mut self, x: TermId, y: TermId, c: Rational64, strict: bool) -> bool {
        // Quick check: get current bounds and see if new constraint conflicts
        if !self.distances_valid
            && let DiffLogicResult::Conflict(_) = self.check()
        {
            return true;
        }

        let x_var = self.graph.get_var(x);
        let y_var = self.graph.get_var(y);

        match (x_var, y_var) {
            (Some(xv), Some(yv)) => {
                // Current constraint implies: x ≤ y + dist[x] - dist[y]
                // New constraint: x - y ≤ c (or < c)
                // Combined: we need to check if the cycle y → x → ... → y is negative

                let dx = self
                    .distances
                    .get(&xv)
                    .copied()
                    .unwrap_or(Rational64::from_integer(0));
                let dy = self
                    .distances
                    .get(&yv)
                    .copied()
                    .unwrap_or(Rational64::from_integer(0));

                // The implied bound from current solution
                let current_diff = dx - dy;

                // Check if adding x - y ≤ c would create a negative cycle
                // This happens if there's a path y → ... → x with weight w
                // and w + c < 0

                // For simplicity, check if the triangle inequality is violated
                // This is conservative but fast
                if strict {
                    c <= -current_diff
                } else {
                    c < -current_diff
                }
            }
            _ => false,
        }
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
        self.current_level
    }

    /// Is this an integer solver?
    pub fn is_integer(&self) -> bool {
        self.graph.is_integer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(id: u32) -> TermId {
        TermId::from(id)
    }

    #[test]
    fn test_solver_creation() {
        let solver = DiffLogicSolver::new(true);
        assert_eq!(solver.num_vars(), 0);
        assert_eq!(solver.num_constraints(), 0);
        assert!(solver.is_integer());
    }

    #[test]
    fn test_add_constraint() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let origin = term(100);

        // x - y ≤ 5
        let result = solver.add_leq(x, y, Rational64::from_integer(5), origin);
        assert!(matches!(result, DiffLogicResult::Ok));
        assert_eq!(solver.num_constraints(), 1);
    }

    #[test]
    fn test_consistency_check() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let z = term(3);
        let o1 = term(100);
        let o2 = term(101);
        let o3 = term(102);

        // Add consistent constraints
        // x - y ≤ 3
        // y - z ≤ 2
        // z - x ≤ 1
        // Total cycle: 3 + 2 + 1 = 6 ≥ 0, so consistent
        solver.add_leq(x, y, Rational64::from_integer(3), o1);
        solver.add_leq(y, z, Rational64::from_integer(2), o2);
        solver.add_leq(z, x, Rational64::from_integer(1), o3);

        let result = solver.check();
        assert!(matches!(result, DiffLogicResult::Ok));
    }

    #[test]
    fn test_conflict_detection() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let z = term(3);
        let o1 = term(100);
        let o2 = term(101);
        let o3 = term(102);

        // Add inconsistent constraints (negative cycle)
        // x - y ≤ -1
        // y - z ≤ -1
        // z - x ≤ -1
        // Total cycle: -3 < 0, so inconsistent
        solver.add_leq(x, y, Rational64::from_integer(-1), o1);
        solver.add_leq(y, z, Rational64::from_integer(-1), o2);
        solver.add_leq(z, x, Rational64::from_integer(-1), o3);

        let result = solver.check();
        assert!(matches!(result, DiffLogicResult::Conflict(_)));
    }

    #[test]
    fn test_model_extraction() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let origin = term(100);

        // x - y ≤ 5
        solver.add_leq(x, y, Rational64::from_integer(5), origin);

        let result = solver.check();
        assert!(matches!(result, DiffLogicResult::Ok));

        let model = solver.get_model();
        assert!(!model.is_empty());

        // Check that the constraint is satisfied
        if let (Some(&vx), Some(&vy)) = (model.get(&x), model.get(&y)) {
            assert!(vx - vy <= Rational64::from_integer(5));
        }
    }

    #[test]
    fn test_push_pop() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let o1 = term(100);
        let o2 = term(101);

        // Level 0: x - y ≤ 5
        solver.add_leq(x, y, Rational64::from_integer(5), o1);
        assert_eq!(solver.num_constraints(), 1);

        // Push to level 1
        solver.push();
        assert_eq!(solver.current_level(), 1);

        // Level 1: x - y ≤ 3
        solver.add_leq(x, y, Rational64::from_integer(3), o2);
        assert_eq!(solver.num_constraints(), 2);

        // Pop to level 0
        solver.pop(1);
        assert_eq!(solver.current_level(), 0);

        // Should still be consistent
        let result = solver.check();
        assert!(matches!(result, DiffLogicResult::Ok));
    }

    #[test]
    fn test_strict_constraint() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let origin = term(100);

        // x - y < 5 (becomes x - y ≤ 4 for integers)
        let result = solver.add_lt(x, y, Rational64::from_integer(5), origin);
        assert!(matches!(result, DiffLogicResult::Ok));

        let check_result = solver.check();
        assert!(matches!(check_result, DiffLogicResult::Ok));
    }

    #[test]
    fn test_immediate_conflict() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let origin = term(100);

        // x - x ≤ -1 (immediate conflict)
        let result = solver.add_leq(x, x, Rational64::from_integer(-1), origin);
        assert!(matches!(result, DiffLogicResult::Conflict(_)));
    }

    #[test]
    fn test_would_conflict() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let origin = term(100);

        // x - y ≤ 0
        solver.add_leq(x, y, Rational64::from_integer(0), origin);
        solver.check();

        // Adding y - x ≤ -1 would create a conflict
        // (cycle x - y ≤ 0, y - x ≤ -1 has total -1 < 0)
        let would_conflict = solver.would_conflict(y, x, Rational64::from_integer(-1), false);
        assert!(would_conflict);
    }

    #[test]
    fn test_reset() {
        let mut solver = DiffLogicSolver::new(true);
        let x = term(1);
        let y = term(2);
        let origin = term(100);

        solver.add_leq(x, y, Rational64::from_integer(5), origin);
        assert_eq!(solver.num_constraints(), 1);

        solver.reset();
        assert_eq!(solver.num_constraints(), 0);
        assert_eq!(solver.num_vars(), 0);
    }

    #[test]
    fn test_real_arithmetic() {
        let mut solver = DiffLogicSolver::new(false); // Real arithmetic
        let x = term(1);
        let y = term(2);
        let origin = term(100);

        // x - y < 0.5 (remains strict for reals)
        let result = solver.add_lt(x, y, Rational64::new(1, 2), origin);
        assert!(matches!(result, DiffLogicResult::Ok));
    }
}
