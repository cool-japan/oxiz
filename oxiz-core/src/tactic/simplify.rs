//! Simplification tactics.

use super::core::*;
use crate::ast::{TermId, TermManager};
use crate::error::Result;

/// Simplification tactic - simplifies boolean and arithmetic expressions
pub struct SimplifyTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> SimplifyTactic<'a> {
    /// Create a new simplify tactic with a term manager
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply simplification to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let simplified: Vec<TermId> = goal
            .assertions
            .iter()
            .map(|&term| self.manager.simplify(term))
            .collect();

        // Check if all assertions simplified to true
        let all_true = simplified.iter().all(|&t| t == self.manager.mk_true());
        if all_true {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        // Check if any assertion simplified to false
        let any_false = simplified.iter().any(|&t| t == self.manager.mk_false());
        if any_false {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = simplified
            .into_iter()
            .filter(|&t| t != self.manager.mk_true())
            .collect();

        // Check if anything changed
        if filtered == goal.assertions {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// A stateless simplify tactic that uses an owned manager
#[derive(Debug, Default)]
pub struct StatelessSimplifyTactic;

impl Tactic for StatelessSimplifyTactic {
    fn name(&self) -> &str {
        "simplify"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        // Real simplification requires the mutable SimplifyTactic
        Ok(TacticResult::SubGoals(vec![(*goal).clone()]))
    }

    fn description(&self) -> &str {
        "Simplifies boolean and arithmetic expressions"
    }
}
