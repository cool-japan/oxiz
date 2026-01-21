//! Goal splitting tactics.

use super::core::*;
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;

/// Goal splitting tactic
pub struct SplitTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> SplitTactic<'a> {
    /// Create a new split tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Find a good boolean term to split on
    fn find_split_candidate(&self, goal: &Goal) -> Option<TermId> {
        use crate::ast::traversal::collect_subterms;
        use rustc_hash::FxHashSet;

        let mut candidates = Vec::new();

        // Collect all boolean subterms from all assertions
        for &assertion in &goal.assertions {
            let subterms = collect_subterms(assertion, self.manager);

            for term_id in subterms {
                if let Some(term) = self.manager.get(term_id) {
                    // Only consider boolean terms that are not constants
                    if term.sort == self.manager.sorts.bool_sort {
                        match &term.kind {
                            TermKind::True | TermKind::False => {}
                            _ => {
                                candidates.push(term_id);
                            }
                        }
                    }
                }
            }
        }

        // Remove duplicates
        let unique: FxHashSet<TermId> = candidates.into_iter().collect();

        // Prefer variables over complex terms
        for &candidate in &unique {
            if let Some(term) = self.manager.get(candidate)
                && matches!(term.kind, TermKind::Var(_))
            {
                return Some(candidate);
            }
        }

        // Return any candidate if no variables found
        unique.into_iter().next()
    }

    /// Apply case splitting to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        // Find a boolean term to split on
        let Some(split_var) = self.find_split_candidate(goal) else {
            return Ok(TacticResult::NotApplicable);
        };

        // Create two sub-goals:
        // 1. goal ∧ split_var
        // 2. goal ∧ ¬split_var

        let true_goal = {
            let mut assertions = goal.assertions.clone();
            assertions.push(split_var);
            Goal {
                assertions,
                precision: goal.precision,
            }
        };

        let false_goal = {
            let not_split_var = self.manager.mk_not(split_var);
            let mut assertions = goal.assertions.clone();
            assertions.push(not_split_var);
            Goal {
                assertions,
                precision: goal.precision,
            }
        };

        Ok(TacticResult::SubGoals(vec![true_goal, false_goal]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessSplitTactic;

impl Tactic for StatelessSplitTactic {
    fn name(&self) -> &str {
        "split"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Performs case splitting on boolean subterms"
    }
}
