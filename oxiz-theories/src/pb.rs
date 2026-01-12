//! Pseudo-Boolean and Cardinality Constraint Solver
//!
//! Supports:
//! - **Cardinality Constraints**: `x1 + x2 + ... + xn >= k` (unit coefficients)
//! - **Pseudo-Boolean Constraints**: `c1*x1 + c2*x2 + ... + cn*xn >= k` (weighted)
//!
//! # Examples
//!
//! ```text
//! // At least 2 of {x, y, z} must be true
//! Cardinality: x + y + z >= 2
//!
//! // Weighted constraint
//! PB: 3*x + 2*y + z >= 5
//! ```

use oxiz_core::ast::TermId;
use std::collections::HashMap;

/// A weighted literal in a PB constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightedLiteral {
    /// The literal (term ID)
    pub lit: TermId,
    /// The weight/coefficient
    pub weight: u64,
}

impl WeightedLiteral {
    /// Create a new weighted literal
    pub fn new(lit: TermId, weight: u64) -> Self {
        Self { lit, weight }
    }
}

/// Type of pseudo-Boolean constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PbConstraintKind {
    /// Cardinality: all weights are 1
    Cardinality,
    /// General pseudo-Boolean with arbitrary weights
    PseudoBoolean,
}

/// A pseudo-Boolean constraint: sum(c_i * x_i) >= k
#[derive(Debug, Clone)]
pub struct PbConstraint {
    /// Unique identifier
    pub id: usize,
    /// Weighted literals
    pub lits: Vec<WeightedLiteral>,
    /// Threshold (right-hand side)
    pub k: u64,
    /// Kind of constraint
    pub kind: PbConstraintKind,
    /// Whether this is a learned constraint
    pub learned: bool,
}

impl PbConstraint {
    /// Create a new cardinality constraint
    pub fn cardinality(id: usize, lits: Vec<TermId>, k: u64) -> Self {
        let wlits = lits
            .into_iter()
            .map(|lit| WeightedLiteral::new(lit, 1))
            .collect();
        Self {
            id,
            lits: wlits,
            k,
            kind: PbConstraintKind::Cardinality,
            learned: false,
        }
    }

    /// Create a new pseudo-Boolean constraint
    pub fn pseudo_boolean(id: usize, lits: Vec<WeightedLiteral>, k: u64) -> Self {
        // Check if all weights are 1
        let is_card = lits.iter().all(|wl| wl.weight == 1);
        let kind = if is_card {
            PbConstraintKind::Cardinality
        } else {
            PbConstraintKind::PseudoBoolean
        };

        Self {
            id,
            lits,
            k,
            kind,
            learned: false,
        }
    }

    /// Normalize the constraint (remove duplicates, sort by weight)
    pub fn normalize(&mut self) {
        // Remove duplicates by summing weights
        let mut lit_weights: HashMap<TermId, u64> = HashMap::new();
        for wl in &self.lits {
            *lit_weights.entry(wl.lit).or_insert(0) += wl.weight;
        }

        self.lits = lit_weights
            .into_iter()
            .map(|(lit, weight)| WeightedLiteral::new(lit, weight))
            .collect();

        // Sort by weight (descending) for better propagation
        self.lits.sort_by(|a, b| b.weight.cmp(&a.weight));

        // Update kind
        let is_card = self.lits.iter().all(|wl| wl.weight == 1);
        self.kind = if is_card {
            PbConstraintKind::Cardinality
        } else {
            PbConstraintKind::PseudoBoolean
        };
    }

    /// Get the slack (how much we can lose and still satisfy)
    ///
    /// slack = (sum of all weights) - k
    pub fn slack(&self) -> u64 {
        let total: u64 = self.lits.iter().map(|wl| wl.weight).sum();
        total.saturating_sub(self.k)
    }

    /// Check if the constraint is trivially satisfied
    pub fn is_trivial(&self) -> bool {
        self.k == 0
    }

    /// Check if the constraint is unsatisfiable
    pub fn is_unsat(&self) -> bool {
        let total: u64 = self.lits.iter().map(|wl| wl.weight).sum();
        total < self.k
    }
}

/// Statistics for PB solver
#[derive(Debug, Clone, Default)]
pub struct PbStats {
    /// Number of cardinality constraints
    pub num_card: usize,
    /// Number of PB constraints
    pub num_pb: usize,
    /// Number of propagations
    pub num_propagations: usize,
    /// Number of conflicts
    pub num_conflicts: usize,
    /// Number of normalizations
    pub num_normalizations: usize,
}

impl PbStats {
    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Configuration for PB solver
#[derive(Debug, Clone)]
pub struct PbConfig {
    /// Enable automatic normalization
    pub auto_normalize: bool,
    /// Enable subsumption checking
    pub enable_subsumption: bool,
    /// Maximum constraint size
    pub max_constraint_size: usize,
}

impl Default for PbConfig {
    fn default() -> Self {
        Self {
            auto_normalize: true,
            enable_subsumption: true,
            max_constraint_size: 10000,
        }
    }
}

/// Pseudo-Boolean solver
pub struct PbSolver {
    /// Configuration
    config: PbConfig,
    /// Statistics
    stats: PbStats,
    /// Constraints
    constraints: Vec<PbConstraint>,
    /// Next constraint ID
    next_id: usize,
    /// Current assignments (lit -> value)
    assignments: HashMap<TermId, bool>,
    /// Context stack for push/pop
    context_stack: Vec<usize>,
}

impl PbSolver {
    /// Create a new PB solver
    pub fn new() -> Self {
        Self::with_config(PbConfig::default())
    }

    /// Create a new solver with configuration
    pub fn with_config(config: PbConfig) -> Self {
        Self {
            config,
            stats: PbStats::default(),
            constraints: Vec::new(),
            next_id: 0,
            assignments: HashMap::new(),
            context_stack: Vec::new(),
        }
    }

    /// Add a cardinality constraint
    pub fn add_cardinality(&mut self, lits: Vec<TermId>, k: u64) -> usize {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        let mut constraint = PbConstraint::cardinality(id, lits, k);

        if self.config.auto_normalize {
            constraint.normalize();
            self.stats.num_normalizations = self.stats.num_normalizations.saturating_add(1);
        }

        self.constraints.push(constraint);
        self.stats.num_card = self.stats.num_card.saturating_add(1);
        id
    }

    /// Add a pseudo-Boolean constraint
    pub fn add_pb_constraint(&mut self, lits: Vec<WeightedLiteral>, k: u64) -> usize {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        let mut constraint = PbConstraint::pseudo_boolean(id, lits, k);

        if self.config.auto_normalize {
            constraint.normalize();
            self.stats.num_normalizations = self.stats.num_normalizations.saturating_add(1);
        }

        self.constraints.push(constraint);
        self.stats.num_pb = self.stats.num_pb.saturating_add(1);
        id
    }

    /// Assign a value to a literal
    pub fn assign(&mut self, lit: TermId, value: bool) {
        self.assignments.insert(lit, value);
    }

    /// Get the assignment of a literal
    pub fn get_assignment(&self, lit: TermId) -> Option<bool> {
        self.assignments.get(&lit).copied()
    }

    /// Check if a constraint is satisfied under current assignments
    pub fn is_satisfied(&self, constraint_id: usize) -> Option<bool> {
        let constraint = self.constraints.get(constraint_id)?;

        let mut sum = 0u64;
        let mut has_unassigned = false;

        for wl in &constraint.lits {
            match self.get_assignment(wl.lit) {
                Some(true) => sum = sum.saturating_add(wl.weight),
                Some(false) => {}
                None => has_unassigned = true,
            }
        }

        if sum >= constraint.k {
            Some(true) // Already satisfied
        } else if !has_unassigned {
            Some(false) // All assigned, still not satisfied
        } else {
            None // Unknown
        }
    }

    /// Propagate unit constraints
    ///
    /// Returns newly propagated assignments
    pub fn propagate(&mut self) -> Vec<(TermId, bool)> {
        let mut propagated = Vec::new();

        for constraint in &self.constraints {
            if constraint.is_trivial() {
                continue;
            }

            let mut sum = 0u64;
            let mut unassigned = Vec::new();

            for wl in &constraint.lits {
                match self.get_assignment(wl.lit) {
                    Some(true) => sum = sum.saturating_add(wl.weight),
                    Some(false) => {}
                    None => unassigned.push(wl),
                }
            }

            // Check if constraint forces remaining literals
            if sum < constraint.k {
                let remaining = constraint.k.saturating_sub(sum);

                // If all unassigned literals must be true
                let min_remaining: u64 = unassigned.iter().map(|wl| wl.weight).sum();
                if min_remaining > 0 && min_remaining <= remaining {
                    // All unassigned must be true
                    for wl in unassigned {
                        if self.assignments.insert(wl.lit, true).is_none() {
                            propagated.push((wl.lit, true));
                        }
                    }
                }
            }
        }

        if !propagated.is_empty() {
            self.stats.num_propagations =
                self.stats.num_propagations.saturating_add(propagated.len());
        }

        propagated
    }

    /// Check for conflicts
    ///
    /// Returns constraint IDs that are unsatisfiable
    pub fn check_conflicts(&mut self) -> Vec<usize> {
        let mut conflicts = Vec::new();

        for constraint in &self.constraints {
            let mut max_sum = 0u64;

            for wl in &constraint.lits {
                match self.get_assignment(wl.lit) {
                    Some(true) => max_sum = max_sum.saturating_add(wl.weight),
                    Some(false) => {}
                    None => max_sum = max_sum.saturating_add(wl.weight), // Could be true
                }
            }

            if max_sum < constraint.k {
                conflicts.push(constraint.id);
            }
        }

        if !conflicts.is_empty() {
            self.stats.num_conflicts = self.stats.num_conflicts.saturating_add(conflicts.len());
        }

        conflicts
    }

    /// Get a constraint by ID
    pub fn get_constraint(&self, id: usize) -> Option<&PbConstraint> {
        self.constraints.iter().find(|c| c.id == id)
    }

    /// Push a new context level
    pub fn push(&mut self) {
        self.context_stack.push(self.assignments.len());
    }

    /// Pop context levels
    pub fn pop(&mut self, levels: usize) {
        for _ in 0..levels {
            if let Some(size) = self.context_stack.pop() {
                // Remove assignments added after push
                let to_remove: Vec<_> = self.assignments.keys().skip(size).copied().collect();
                for lit in to_remove {
                    self.assignments.remove(&lit);
                }
            }
        }
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.constraints.clear();
        self.assignments.clear();
        self.context_stack.clear();
        self.stats.reset();
        self.next_id = 0;
    }

    /// Get statistics
    pub fn stats(&self) -> &PbStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &PbConfig {
        &self.config
    }

    /// Get number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }
}

impl Default for PbSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_literal() {
        let lit = TermId::new(1);
        let wl = WeightedLiteral::new(lit, 5);
        assert_eq!(wl.lit, lit);
        assert_eq!(wl.weight, 5);
    }

    #[test]
    fn test_cardinality_constraint() {
        let lits = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let constraint = PbConstraint::cardinality(0, lits, 2);

        assert_eq!(constraint.kind, PbConstraintKind::Cardinality);
        assert_eq!(constraint.k, 2);
        assert_eq!(constraint.lits.len(), 3);
        assert!(constraint.lits.iter().all(|wl| wl.weight == 1));
    }

    #[test]
    fn test_pb_constraint() {
        let lits = vec![
            WeightedLiteral::new(TermId::new(1), 3),
            WeightedLiteral::new(TermId::new(2), 2),
            WeightedLiteral::new(TermId::new(3), 1),
        ];
        let constraint = PbConstraint::pseudo_boolean(0, lits, 5);

        assert_eq!(constraint.kind, PbConstraintKind::PseudoBoolean);
        assert_eq!(constraint.k, 5);
    }

    #[test]
    fn test_normalize() {
        let lits = vec![
            WeightedLiteral::new(TermId::new(1), 3),
            WeightedLiteral::new(TermId::new(1), 2), // Duplicate
            WeightedLiteral::new(TermId::new(2), 1),
        ];
        let mut constraint = PbConstraint::pseudo_boolean(0, lits, 5);
        constraint.normalize();

        // Should combine duplicate weights: 3 + 2 = 5
        assert_eq!(constraint.lits.len(), 2);
        let lit1_weight = constraint
            .lits
            .iter()
            .find(|wl| wl.lit == TermId::new(1))
            .map(|wl| wl.weight);
        assert_eq!(lit1_weight, Some(5));
    }

    #[test]
    fn test_slack() {
        let lits = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let constraint = PbConstraint::cardinality(0, lits, 2);
        // Total = 3, k = 2, slack = 1
        assert_eq!(constraint.slack(), 1);
    }

    #[test]
    fn test_trivial_constraint() {
        let lits = vec![TermId::new(1)];
        let constraint = PbConstraint::cardinality(0, lits, 0);
        assert!(constraint.is_trivial());
    }

    #[test]
    fn test_unsat_constraint() {
        let lits = vec![TermId::new(1), TermId::new(2)];
        let constraint = PbConstraint::cardinality(0, lits, 5); // Impossible: 2 < 5
        assert!(constraint.is_unsat());
    }

    #[test]
    fn test_solver_creation() {
        let solver = PbSolver::new();
        assert_eq!(solver.num_constraints(), 0);
        assert_eq!(solver.stats().num_card, 0);
    }

    #[test]
    fn test_add_cardinality() {
        let mut solver = PbSolver::new();
        let lits = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let id = solver.add_cardinality(lits, 2);

        assert_eq!(solver.num_constraints(), 1);
        assert_eq!(solver.stats().num_card, 1);

        let constraint = solver.get_constraint(id).expect("constraint exists");
        assert_eq!(constraint.kind, PbConstraintKind::Cardinality);
    }

    #[test]
    fn test_add_pb_constraint() {
        let mut solver = PbSolver::new();
        let lits = vec![
            WeightedLiteral::new(TermId::new(1), 3),
            WeightedLiteral::new(TermId::new(2), 2),
        ];
        let id = solver.add_pb_constraint(lits, 4);

        assert_eq!(solver.num_constraints(), 1);
        assert_eq!(solver.stats().num_pb, 1);

        let constraint = solver.get_constraint(id).expect("constraint exists");
        assert_eq!(constraint.kind, PbConstraintKind::PseudoBoolean);
    }

    #[test]
    fn test_assign_and_check() {
        let mut solver = PbSolver::new();
        let lits = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let id = solver.add_cardinality(lits, 2);

        // Assign x1=true, x2=true, x3=false
        solver.assign(TermId::new(1), true);
        solver.assign(TermId::new(2), true);
        solver.assign(TermId::new(3), false);

        // Should be satisfied (2 >= 2)
        assert_eq!(solver.is_satisfied(id), Some(true));
    }

    #[test]
    fn test_unsatisfied_constraint() {
        let mut solver = PbSolver::new();
        let lits = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        let id = solver.add_cardinality(lits, 2);

        // Assign all to false
        solver.assign(TermId::new(1), false);
        solver.assign(TermId::new(2), false);
        solver.assign(TermId::new(3), false);

        // Should be unsatisfied (0 < 2)
        assert_eq!(solver.is_satisfied(id), Some(false));
    }

    #[test]
    fn test_propagate() {
        let mut solver = PbSolver::new();
        let lits = vec![TermId::new(1), TermId::new(2), TermId::new(3)];
        solver.add_cardinality(lits, 3);

        // All three must be true
        let propagated = solver.propagate();

        assert_eq!(propagated.len(), 3);
        assert_eq!(solver.stats().num_propagations, 3);
    }

    #[test]
    fn test_check_conflicts() {
        let mut solver = PbSolver::new();
        let lits = vec![TermId::new(1), TermId::new(2)];
        solver.add_cardinality(lits.clone(), 3); // Impossible: 2 < 3

        // Assign both to false
        solver.assign(TermId::new(1), false);
        solver.assign(TermId::new(2), false);

        let conflicts = solver.check_conflicts();
        assert!(!conflicts.is_empty());
        assert_eq!(solver.stats().num_conflicts, 1);
    }

    #[test]
    fn test_push_pop() {
        let mut solver = PbSolver::new();

        solver.push();
        solver.assign(TermId::new(1), true);
        assert_eq!(solver.get_assignment(TermId::new(1)), Some(true));

        solver.pop(1);
        assert_eq!(solver.get_assignment(TermId::new(1)), None);
    }

    #[test]
    fn test_reset() {
        let mut solver = PbSolver::new();
        let lits = vec![TermId::new(1), TermId::new(2)];
        solver.add_cardinality(lits, 1);
        solver.assign(TermId::new(1), true);

        solver.reset();

        assert_eq!(solver.num_constraints(), 0);
        assert_eq!(solver.get_assignment(TermId::new(1)), None);
    }
}
