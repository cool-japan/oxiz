//! Eliminate unconstrained variables tactic.

use super::core::*;
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;

/// Eliminate unconstrained variables tactic
pub struct EliminateUnconstrainedTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> EliminateUnconstrainedTactic<'a> {
    /// Create a new eliminate-unconstrained tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Count occurrences of each variable in the goal
    fn count_variable_occurrences(&self, goal: &Goal) -> rustc_hash::FxHashMap<TermId, usize> {
        use crate::ast::traversal::collect_subterms;
        use rustc_hash::FxHashMap;

        let mut counts: FxHashMap<TermId, usize> = FxHashMap::default();

        for &assertion in &goal.assertions {
            let subterms = collect_subterms(assertion, self.manager);
            for term_id in subterms {
                if let Some(term) = self.manager.get(term_id)
                    && matches!(term.kind, TermKind::Var(_))
                {
                    *counts.entry(term_id).or_insert(0) += 1;
                }
            }
        }

        counts
    }

    /// Check if a variable appears in an eliminable context
    fn is_eliminable(&self, var: TermId, assertion: TermId) -> bool {
        if let Some(term) = self.manager.get(assertion) {
            match &term.kind {
                TermKind::Eq(lhs, rhs) => {
                    if *lhs == var {
                        !self.contains_var(var, *rhs)
                    } else if *rhs == var {
                        !self.contains_var(var, *lhs)
                    } else {
                        false
                    }
                }
                TermKind::Or(args) => args.contains(&var),
                TermKind::And(args) => args.iter().any(|&arg| self.is_eliminable(var, arg)),
                _ => false,
            }
        } else {
            false
        }
    }

    /// Check if a variable appears in a term
    fn contains_var(&self, var: TermId, term: TermId) -> bool {
        use crate::ast::traversal::contains_term;
        contains_term(term, var, self.manager)
    }

    /// Eliminate a variable from assertions
    fn eliminate_variable(&mut self, var: TermId, assertions: &[TermId]) -> Vec<TermId> {
        let mut result = Vec::new();

        for &assertion in assertions {
            if let Some(term) = self.manager.get(assertion) {
                match &term.kind {
                    TermKind::Eq(lhs, rhs) if *lhs == var || *rhs == var => {
                        continue;
                    }
                    TermKind::Or(args) if args.contains(&var) => {
                        result.push(self.manager.mk_true());
                    }
                    _ => {
                        result.push(assertion);
                    }
                }
            } else {
                result.push(assertion);
            }
        }

        result
    }

    /// Apply the eliminate-unconstrained tactic
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let counts = self.count_variable_occurrences(goal);

        let mut candidates: Vec<TermId> = counts
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(&var, _)| var)
            .collect();

        if candidates.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        candidates.retain(|&var| {
            goal.assertions
                .iter()
                .any(|&assertion| self.is_eliminable(var, assertion))
        });

        if candidates.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut new_assertions = goal.assertions.clone();
        for var in candidates {
            new_assertions = self.eliminate_variable(var, &new_assertions);
        }

        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        if new_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        let filtered: Vec<TermId> = new_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        if filtered.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        if filtered == goal.assertions {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessEliminateUnconstrainedTactic;

impl Tactic for StatelessEliminateUnconstrainedTactic {
    fn name(&self) -> &str {
        "elim-uncnstr"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Eliminates unconstrained variables from the formula"
    }
}
