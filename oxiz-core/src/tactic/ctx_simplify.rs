//! Context solver simplification tactics.

use super::core::*;
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
#[allow(unused_imports)]
use crate::prelude::*;
use smallvec::SmallVec;

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
    ) -> crate::prelude::FxHashMap<TermId, TermId> {
        use crate::ast::TermKind;
        use crate::prelude::FxHashMap;

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

    fn simplify_with_context(&mut self, term: TermId) -> TermId {
        let assignments = FxHashMap::default();
        let mut cache = FxHashMap::default();
        self.simplify_with_assignments(term, &assignments, &mut cache)
    }

    fn simplify_with_assignments(
        &mut self,
        term: TermId,
        assignments: &FxHashMap<TermId, bool>,
        cache: &mut FxHashMap<(TermId, u64), TermId>,
    ) -> TermId {
        let cache_key = (term, assignment_fingerprint(assignments));
        if let Some(&cached) = cache.get(&cache_key) {
            return cached;
        }

        let simplified = match self.manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Implies(cond, rhs)) => {
                let cond = self.simplify_with_assignments(cond, assignments, cache);
                let mut rhs_assignments = assignments.clone();
                record_assignment(cond, true, self.manager, &mut rhs_assignments);
                let rhs = self.simplify_with_assignments(rhs, &rhs_assignments, cache);
                self.manager.mk_implies(cond, rhs)
            }
            Some(TermKind::And(args)) => {
                let mut scoped = assignments.clone();
                let simplified_args: SmallVec<[TermId; 4]> = args
                    .into_iter()
                    .map(|arg| {
                        let rewritten = self.simplify_with_assignments(arg, &scoped, cache);
                        record_assignment(rewritten, true, self.manager, &mut scoped);
                        rewritten
                    })
                    .collect();
                self.manager.mk_and(simplified_args)
            }
            Some(TermKind::Or(args)) => {
                let simplified_args: SmallVec<[TermId; 4]> = args
                    .into_iter()
                    .map(|arg| self.simplify_with_assignments(arg, assignments, cache))
                    .collect();
                self.manager.mk_or(simplified_args)
            }
            Some(TermKind::Not(arg)) => {
                let arg = self.simplify_with_assignments(arg, assignments, cache);
                self.manager.mk_not(arg)
            }
            Some(TermKind::Ite(cond, then_branch, else_branch)) => {
                if let Some(value) = evaluate_condition(cond, assignments, self.manager) {
                    let chosen = if value { then_branch } else { else_branch };
                    self.simplify_with_assignments(chosen, assignments, cache)
                } else {
                    let cond = self.simplify_with_assignments(cond, assignments, cache);
                    if let Some(value) = evaluate_condition(cond, assignments, self.manager) {
                        let chosen = if value { then_branch } else { else_branch };
                        self.simplify_with_assignments(chosen, assignments, cache)
                    } else {
                        let then_branch =
                            self.simplify_with_assignments(then_branch, assignments, cache);
                        let else_branch =
                            self.simplify_with_assignments(else_branch, assignments, cache);
                        self.manager.mk_ite(cond, then_branch, else_branch)
                    }
                }
            }
            Some(TermKind::Eq(lhs, rhs)) => {
                let lhs = self.simplify_with_assignments(lhs, assignments, cache);
                let rhs = self.simplify_with_assignments(rhs, assignments, cache);
                self.manager.mk_eq(lhs, rhs)
            }
            Some(TermKind::Add(args)) => {
                let args: SmallVec<[TermId; 4]> = args
                    .into_iter()
                    .map(|arg| self.simplify_with_assignments(arg, assignments, cache))
                    .collect();
                self.manager.mk_add(args)
            }
            Some(TermKind::Sub(lhs, rhs)) => {
                let lhs = self.simplify_with_assignments(lhs, assignments, cache);
                let rhs = self.simplify_with_assignments(rhs, assignments, cache);
                self.manager.mk_sub(lhs, rhs)
            }
            Some(TermKind::Lt(lhs, rhs)) => {
                let lhs = self.simplify_with_assignments(lhs, assignments, cache);
                let rhs = self.simplify_with_assignments(rhs, assignments, cache);
                self.manager.mk_lt(lhs, rhs)
            }
            Some(TermKind::Le(lhs, rhs)) => {
                let lhs = self.simplify_with_assignments(lhs, assignments, cache);
                let rhs = self.simplify_with_assignments(rhs, assignments, cache);
                self.manager.mk_le(lhs, rhs)
            }
            Some(TermKind::Gt(lhs, rhs)) => {
                let lhs = self.simplify_with_assignments(lhs, assignments, cache);
                let rhs = self.simplify_with_assignments(rhs, assignments, cache);
                self.manager.mk_gt(lhs, rhs)
            }
            Some(TermKind::Ge(lhs, rhs)) => {
                let lhs = self.simplify_with_assignments(lhs, assignments, cache);
                let rhs = self.simplify_with_assignments(rhs, assignments, cache);
                self.manager.mk_ge(lhs, rhs)
            }
            _ => term,
        };

        let normalized = self.manager.simplify(simplified);
        cache.insert(cache_key, normalized);
        normalized
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

                let context_rewritten = self.simplify_with_context(current_assertions[i]);
                let substituted = if subst.is_empty() {
                    context_rewritten
                } else {
                    self.manager.substitute(context_rewritten, &subst)
                };
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

fn record_assignment(
    condition: TermId,
    value: bool,
    manager: &TermManager,
    assignments: &mut FxHashMap<TermId, bool>,
) {
    assignments.insert(condition, value);
    if let Some(term) = manager.get(condition) {
        match &term.kind {
            TermKind::Not(inner) => {
                assignments.insert(*inner, !value);
            }
            TermKind::Var(_) => {
                assignments.insert(condition, value);
            }
            _ => {}
        }
    }
}

fn evaluate_condition(
    condition: TermId,
    assignments: &FxHashMap<TermId, bool>,
    manager: &TermManager,
) -> Option<bool> {
    if let Some(&value) = assignments.get(&condition) {
        return Some(value);
    }

    match manager.get(condition).map(|term| &term.kind) {
        Some(TermKind::True) => Some(true),
        Some(TermKind::False) => Some(false),
        Some(TermKind::Not(inner)) => evaluate_condition(*inner, assignments, manager).map(|v| !v),
        _ => None,
    }
}

fn assignment_fingerprint(assignments: &FxHashMap<TermId, bool>) -> u64 {
    let mut pairs: Vec<(u32, bool)> = assignments.iter().map(|(k, v)| (k.0, *v)).collect();
    pairs.sort_unstable_by_key(|(term_id, _)| *term_id);
    let mut hash = 0_u64;
    for (term_id, value) in pairs {
        hash = hash
            .wrapping_mul(1_099_511_628_211)
            .wrapping_add(u64::from(term_id) << 1)
            .wrapping_add(u64::from(value));
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ctx_dep_rewrite_dead_branch() {
        let mut manager = TermManager::new();
        let cond = manager.mk_var("cond", manager.sorts.bool_sort);
        let then_branch = manager.mk_var("then", manager.sorts.int_sort);
        let else_branch = manager.mk_var("else", manager.sorts.int_sort);
        let ite = manager.mk_ite(cond, then_branch, else_branch);
        let guarded = manager.mk_implies(cond, ite);
        let goal = Goal::new(vec![guarded]);

        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("test operation should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                let expected = manager.mk_implies(cond, then_branch);
                assert_eq!(goals[0].assertions, vec![expected]);
            }
            other => panic!("expected rewritten implication, got {other:?}"),
        }
    }
}
