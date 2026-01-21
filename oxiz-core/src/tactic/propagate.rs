//! Value propagation tactics.

use super::core::*;
use crate::error::Result;
use crate::ast::{TermId, TermManager};

/// Value propagation tactic
pub struct PropagateValuesTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> PropagateValuesTactic<'a> {
    /// Create a new propagate-values tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply value propagation to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        use crate::ast::TermKind;
        use rustc_hash::FxHashMap;

        // Phase 1: Collect equalities of the form (= var constant)
        let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();

        for &assertion in &goal.assertions {
            if let Some(term) = self.manager.get(assertion)
                && let TermKind::Eq(lhs, rhs) = &term.kind
            {
                let lhs_term = self.manager.get(*lhs);
                let rhs_term = self.manager.get(*rhs);

                match (lhs_term.map(|t| &t.kind), rhs_term.map(|t| &t.kind)) {
                    // x = constant
                    (Some(TermKind::Var(_)), Some(k)) if is_constant(k) => {
                        subst.insert(*lhs, *rhs);
                    }
                    // constant = x
                    (Some(k), Some(TermKind::Var(_))) if is_constant(k) => {
                        subst.insert(*rhs, *lhs);
                    }
                    _ => {}
                }
            }
        }

        // No substitutions found
        if subst.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 2: Apply substitutions to all assertions
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());
        let mut changed = false;

        for &assertion in &goal.assertions {
            let substituted = self.manager.substitute(assertion, &subst);
            let simplified = self.manager.simplify(substituted);

            if simplified != assertion {
                changed = true;
            }
            new_assertions.push(simplified);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        // Check for trivially true/false
        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        // Check if any assertion is false
        if new_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = new_assertions
            .into_iter()
            .filter(|&a| a != true_id)
            .collect();

        // If all assertions are true, goal is SAT
        if filtered.is_empty() {
            return Ok(TacticResult::Solved(SolveResult::Sat));
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: filtered,
            precision: goal.precision,
        }]))
    }
}

/// Check if a term kind is a constant
fn is_constant(kind: &crate::ast::TermKind) -> bool {
    use crate::ast::TermKind;
    matches!(
        kind,
        TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
    )
}

/// Stateless version for the Tactic trait (placeholder)
#[derive(Debug, Default)]
pub struct StatelessPropagateValuesTactic;

impl Tactic for StatelessPropagateValuesTactic {
    fn name(&self) -> &str {
        "propagate-values"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Propagates constant values through the formula"
    }
}
