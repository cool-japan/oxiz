//! Core Theory Trait for SMT Solvers.
//!
//! Defines the interface that all theory solvers must implement for
//! integration into the CDCL(T) framework.

use crate::literal::Lit;
use crate::{Sort, TermId};

/// Result of a theory check operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TheoryCheckResult {
    /// The current assignment is satisfiable in the theory.
    Sat,
    /// The current assignment is unsatisfiable.
    Unsat,
    /// Unable to determine (incomplete theory solver).
    Unknown,
}

/// Result of a propagation attempt.
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Literals propagated by the theory.
    pub propagated: Vec<Lit>,
    /// Explanation for each propagation (antecedents).
    pub explanations: Vec<Vec<Lit>>,
}

/// Conflict explanation from a theory.
#[derive(Debug, Clone)]
pub struct TheoryConflict {
    /// Core literals that caused the conflict.
    pub core: Vec<Lit>,
    /// Optional proof/justification.
    pub proof: Option<String>,
}

/// Core trait for theory solvers in CDCL(T).
///
/// This trait defines the minimal interface required for a theory solver
/// to participate in CDCL(T) solving.
pub trait Theory: Send + Sync {
    /// Get the name of this theory (e.g., "EUF", "LRA", "BV").
    fn name(&self) -> &str;

    /// Reset the theory solver to initial state.
    fn reset(&mut self);

    /// Notify the theory of a new decision/propagation at the given decision level.
    ///
    /// The theory should update its internal state to reflect this assignment.
    fn assert_literal(&mut self, lit: Lit, level: usize);

    /// Backtrack to the given decision level.
    ///
    /// The theory should undo all assertions made at levels > `level`.
    fn backtrack(&mut self, level: usize);

    /// Check consistency of current theory assignment.
    ///
    /// Returns:
    /// - `Sat` if the assignment is theory-consistent
    /// - `Unsat` if inconsistent (call `get_conflict()` for explanation)
    /// - `Unknown` if the solver cannot determine (incomplete)
    fn check(&mut self) -> TheoryCheckResult;

    /// Get the conflict explanation after `check()` returns `Unsat`.
    ///
    /// The conflict should be a minimal set of literals whose conjunction
    /// is theory-inconsistent.
    fn get_conflict(&self) -> Option<TheoryConflict>;

    /// Attempt to propagate literals based on current assignment.
    ///
    /// The theory may deduce additional literals that must be true given
    /// the current partial assignment.
    fn propagate(&mut self) -> PropagationResult;

    /// Explain why a literal was propagated.
    ///
    /// Returns the set of literals (antecedents) that imply `lit`.
    fn explain_propagation(&self, lit: Lit) -> Vec<Lit>;

    /// Get a model (satisfying assignment) for theory variables.
    ///
    /// Called after `check()` returns `Sat` to extract concrete values.
    fn get_model(&self) -> TheoryModel;

    /// Check if the solver supports incremental solving.
    fn supports_incremental(&self) -> bool {
        true
    }

    /// Check if the solver supports unsat core generation.
    fn supports_unsat_core(&self) -> bool {
        false
    }

    /// Push a new assertion scope.
    ///
    /// Only called if `supports_incremental()` returns true.
    fn push(&mut self) {
        // Default: do nothing
    }

    /// Pop the last assertion scope.
    ///
    /// Only called if `supports_incremental()` returns true.
    fn pop(&mut self) {
        // Default: do nothing
    }
}

/// Model produced by a theory solver.
#[derive(Debug, Clone, Default)]
pub struct TheoryModel {
    /// Variable assignments: term ID -> value term ID.
    pub assignments: std::collections::HashMap<TermId, TermId>,
}

impl TheoryModel {
    /// Create an empty model.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an assignment to the model.
    pub fn assign(&mut self, var: TermId, value: TermId) {
        self.assignments.insert(var, value);
    }

    /// Get the value assigned to a variable.
    pub fn get(&self, var: TermId) -> Option<TermId> {
        self.assignments.get(&var).copied()
    }

    /// Check if a variable has an assignment.
    pub fn contains(&self, var: TermId) -> bool {
        self.assignments.contains_key(&var)
    }

    /// Number of assignments in the model.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Check if the model is empty.
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

/// Extension trait for theories that support quantifier reasoning.
pub trait QuantifiedTheory: Theory {
    /// Instantiate a quantified formula based on current model.
    ///
    /// Returns new ground instances to add to the solver.
    fn instantiate_quantifiers(&mut self) -> Vec<TermId>;

    /// Get a set of relevant ground terms for quantifier instantiation.
    fn get_ground_terms(&self, sort: &Sort) -> Vec<TermId>;
}

/// Extension trait for theories that support optimization.
pub trait OptimizingTheory: Theory {
    /// Set an objective function to minimize.
    fn set_objective(&mut self, objective: TermId, minimize: bool);

    /// Get the current objective value (if bounded).
    fn get_objective_value(&self) -> Option<String>;

    /// Tighten the objective bound.
    fn tighten_bound(&mut self, upper_bound: Option<String>, lower_bound: Option<String>);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock theory implementation for testing
    struct MockTheory {
        name: String,
        check_result: TheoryCheckResult,
    }

    impl MockTheory {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                check_result: TheoryCheckResult::Sat,
            }
        }
    }

    impl Theory for MockTheory {
        fn name(&self) -> &str {
            &self.name
        }

        fn reset(&mut self) {}

        fn assert_literal(&mut self, _lit: Lit, _level: usize) {}

        fn backtrack(&mut self, _level: usize) {}

        fn check(&mut self) -> TheoryCheckResult {
            self.check_result
        }

        fn get_conflict(&self) -> Option<TheoryConflict> {
            None
        }

        fn propagate(&mut self) -> PropagationResult {
            PropagationResult {
                propagated: Vec::new(),
                explanations: Vec::new(),
            }
        }

        fn explain_propagation(&self, _lit: Lit) -> Vec<Lit> {
            Vec::new()
        }

        fn get_model(&self) -> TheoryModel {
            TheoryModel::new()
        }
    }

    #[test]
    fn test_theory_trait() {
        let mut theory = MockTheory::new("test");
        assert_eq!(theory.name(), "test");
        assert_eq!(theory.check(), TheoryCheckResult::Sat);
        assert!(theory.supports_incremental());
    }

    #[test]
    fn test_theory_model() {
        let mut model = TheoryModel::new();
        assert!(model.is_empty());

        model.assign(TermId::new(0), TermId::new(1));
        assert_eq!(model.len(), 1);
        assert!(model.contains(TermId::new(0)));
        assert_eq!(model.get(TermId::new(0)), Some(TermId::new(1)));
    }

    #[test]
    fn test_propagation_result() {
        let result = PropagationResult {
            propagated: vec![],
            explanations: vec![],
        };
        assert!(result.propagated.is_empty());
    }
}
