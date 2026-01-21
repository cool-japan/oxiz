//! Context solver simplification tactics.

use super::core::*;
use crate::ast::{TermId, TermManager};
use crate::error::Result;

/// Context-based solver simplification tactic
pub struct CtxSolverSimplifyTactic<'a> {
    manager: &'a mut TermManager,
    /// Maximum number of iterations
    max_iterations: usize,
}

impl<'a> CtxSolverSimplifyTactic<'a> {
    /// Create a new context-solver-simplify tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            max_iterations: 10,
        }
    }

    /// Create with custom max iterations
    pub fn with_max_iterations(manager: &'a mut TermManager, max_iterations: usize) -> Self {
        Self {
            manager,
            max_iterations,
        }
    }

    /// Extract equalities from context that can be used for substitution
    fn extract_substitutions(
        &self,
        assertions: &[TermId],
        skip_index: usize,
    ) -> rustc_hash::FxHashMap<TermId, TermId> {
        use crate::ast::TermKind;
        use rustc_hash::FxHashMap;

        let mut subst: FxHashMap<TermId, TermId> = FxHashMap::default();

        for (i, &assertion) in assertions.iter().enumerate() {
            if i == skip_index {
                continue;
            }

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
                    // x = y (prefer lower term ID as representative)
                    (Some(TermKind::Var(_)), Some(TermKind::Var(_))) => {
                        if lhs.0 > rhs.0 {
                            subst.insert(*lhs, *rhs);
                        } else {
                            subst.insert(*rhs, *lhs);
                        }
                    }
                    _ => {}
                }
            }
        }

        subst
    }

    /// Apply context-dependent simplification to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        if goal.assertions.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        let mut current_assertions = goal.assertions.clone();
        let mut changed = false;

        // Iterate until fixpoint or max iterations
        for _ in 0..self.max_iterations {
            let mut iteration_changed = false;
            let mut new_assertions = Vec::with_capacity(current_assertions.len());

            for i in 0..current_assertions.len() {
                // Extract substitutions from other assertions
                let subst = self.extract_substitutions(&current_assertions, i);

                if subst.is_empty() {
                    new_assertions.push(current_assertions[i]);
                    continue;
                }

                // Apply substitution and simplify
                let substituted = self.manager.substitute(current_assertions[i], &subst);
                let simplified = self.manager.simplify(substituted);

                if simplified != current_assertions[i] {
                    iteration_changed = true;
                    changed = true;
                }
                new_assertions.push(simplified);
            }

            current_assertions = new_assertions;

            if !iteration_changed {
                break;
            }
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        // Check for trivially true/false
        let true_id = self.manager.mk_true();
        let false_id = self.manager.mk_false();

        // Check if any assertion is false
        if current_assertions.contains(&false_id) {
            return Ok(TacticResult::Solved(SolveResult::Unsat));
        }

        // Filter out true assertions
        let filtered: Vec<TermId> = current_assertions
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

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessCtxSolverSimplifyTactic;

impl Tactic for StatelessCtxSolverSimplifyTactic {
    fn name(&self) -> &str {
        "ctx-solver-simplify"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        // Without a term manager, we can only return the goal unchanged
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "Simplifies assertions using other assertions as context"
    }
}
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

