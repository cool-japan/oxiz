//! Aggressive simplification tactic.

use super::core::*;
use crate::ast::{TermId, TermManager};
use crate::error::Result;
use crate::simplification::{AggressiveSimplifier, SimplificationConfig};

/// Simplification tactic with more expensive preprocessing rewrites enabled.
pub struct AggressiveSimplifyTactic<'a> {
    manager: &'a mut TermManager,
    config: SimplificationConfig,
}

impl<'a> AggressiveSimplifyTactic<'a> {
    /// Create a new aggressive simplification tactic.
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            config: SimplificationConfig { aggressive: true },
        }
    }

    /// Create a tactic with an explicit simplification configuration.
    pub fn with_config(manager: &'a mut TermManager, config: SimplificationConfig) -> Self {
        Self { manager, config }
    }

    /// Apply aggressive simplification to a goal.
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut simplifier = AggressiveSimplifier::new(self.manager, self.config);
        let simplified: Vec<TermId> = goal
            .assertions
            .iter()
            .map(|&term| simplifier.simplify_term(term))
            .collect();

        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        if simplified.iter().all(|&term| term == true_id) {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }
        if simplified.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        let filtered: Vec<TermId> = simplified
            .into_iter()
            .filter(|&term| term != true_id)
            .collect();

        if filtered == goal.assertions {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// Stateless aggressive simplification tactic.
#[derive(Debug, Default)]
pub struct StatelessAggressiveSimplifyTactic;

impl Tactic for StatelessAggressiveSimplifyTactic {
    fn name(&self) -> &str {
        "aggressive-simplify"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Applies aggressive Boolean and arithmetic simplifications"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggressive_simplify_ite_same_branches() {
        let mut manager = TermManager::new();
        let cond = manager.mk_var("cond", manager.sorts.bool_sort);
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let ite = manager.mk_ite(cond, x, x);

        let goal = Goal::new(vec![ite]);
        let mut tactic = AggressiveSimplifyTactic::new(&mut manager);
        let result = tactic
            .apply_mut(&goal)
            .expect("test operation should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert_eq!(goals[0].assertions, vec![x]);
            }
            TacticResult::NotApplicable => {
                assert_eq!(goal.assertions, vec![x]);
            }
            other => panic!("expected subgoal, got {other:?}"),
        }
    }
}
