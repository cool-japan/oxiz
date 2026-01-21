//! Core tactic types and traits.

use crate::ast::TermId;
use crate::error::Result;

/// A goal represents a formula to be solved
#[derive(Debug, Clone)]
pub struct Goal {
    /// The assertions in this goal
    pub assertions: Vec<TermId>,
    /// Model precision (for optimization)
    pub precision: Precision,
}

/// Precision level for model generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Under approximation - may miss solutions
    Under,
    /// Exact solution required
    #[default]
    Precise,
    /// Over approximation - may include spurious solutions
    Over,
}

impl Goal {
    /// Create a new goal with the given assertions
    #[must_use]
    pub fn new(assertions: Vec<TermId>) -> Self {
        Self {
            assertions,
            precision: Precision::Precise,
        }
    }

    /// Create an empty goal (trivially satisfiable)
    #[must_use]
    pub fn empty() -> Self {
        Self {
            assertions: Vec::new(),
            precision: Precision::Precise,
        }
    }

    /// Add an assertion to the goal
    pub fn add(&mut self, term: TermId) {
        self.assertions.push(term);
    }

    /// Check if the goal is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.assertions.is_empty()
    }

    /// Get the number of assertions
    #[must_use]
    pub fn len(&self) -> usize {
        self.assertions.len()
    }
}

/// Result of applying a tactic
#[derive(Debug)]
pub enum TacticResult {
    /// The goal was solved (sat/unsat)
    Solved(SolveResult),
    /// The goal was transformed into sub-goals
    SubGoals(Vec<Goal>),
    /// The tactic does not apply to this goal
    NotApplicable,
    /// The tactic failed with an error
    Failed(String),
}

/// Solve result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveResult {
    /// Satisfiable
    Sat,
    /// Unsatisfiable
    Unsat,
    /// Unknown
    Unknown,
}

/// A tactic transforms goals into sub-goals
pub trait Tactic: Send + Sync {
    /// Get the name of this tactic
    fn name(&self) -> &str;

    /// Apply the tactic to a goal
    fn apply(&self, goal: &Goal) -> Result<TacticResult>;

    /// Get a description of the tactic
    fn description(&self) -> &str {
        ""
    }
}
