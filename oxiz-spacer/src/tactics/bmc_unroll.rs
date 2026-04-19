use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::error::Result;
use oxiz_core::tactic::{Goal, TacticResult};
use smallvec::SmallVec;

/// Minimal BMC unrolling engine over `(init, trans, property)` formulas.
pub struct BmcEngine<'a> {
    manager: &'a mut TermManager,
}

impl<'a> BmcEngine<'a> {
    /// Create a new unrolling engine.
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Unroll a transition system encoded as `(init, trans, property)` assertions.
    pub fn unroll(&mut self, assertions: &[TermId], depth: usize) -> Option<Vec<TermId>> {
        if assertions.len() < 3 {
            return None;
        }

        let init = assertions[0];
        let trans = assertions[1];
        let property = assertions[2];
        let mut flattened = Vec::with_capacity(depth.saturating_mul(2).saturating_add(2));

        flattened.push(self.rename_term(init, 0));
        for step in 0..depth {
            flattened.push(self.rename_term(trans, step));
        }
        for step in 0..=depth {
            flattened.push(self.rename_term(property, step));
        }

        Some(flattened)
    }

    fn rename_term(&mut self, term: TermId, step: usize) -> TermId {
        let mut cache = std::collections::HashMap::new();
        self.rename_term_cached(term, step, &mut cache)
    }

    fn rename_term_cached(
        &mut self,
        term: TermId,
        step: usize,
        cache: &mut std::collections::HashMap<(TermId, usize), TermId>,
    ) -> TermId {
        if let Some(&cached) = cache.get(&(term, step)) {
            return cached;
        }

        let renamed = match self.manager.get(term).cloned() {
            Some(node) => match node.kind {
                TermKind::Var(name) => {
                    let source = self.manager.resolve_str(name);
                    let (base, target_step) = next_state_name(source, step);
                    self.manager.mk_var(&format!("{base}@{target_step}"), node.sort)
                }
                TermKind::Not(arg) => {
                    let arg = self.rename_term_cached(arg, step, cache);
                    self.manager.mk_not(arg)
                }
                TermKind::And(args) => {
                    let args: SmallVec<[TermId; 4]> = args
                        .into_iter()
                        .map(|arg| self.rename_term_cached(arg, step, cache))
                        .collect();
                    self.manager.mk_and(args)
                }
                TermKind::Or(args) => {
                    let args: SmallVec<[TermId; 4]> = args
                        .into_iter()
                        .map(|arg| self.rename_term_cached(arg, step, cache))
                        .collect();
                    self.manager.mk_or(args)
                }
                TermKind::Implies(lhs, rhs) => {
                    let lhs = self.rename_term_cached(lhs, step, cache);
                    let rhs = self.rename_term_cached(rhs, step, cache);
                    self.manager.mk_implies(lhs, rhs)
                }
                TermKind::Eq(lhs, rhs) => {
                    let lhs = self.rename_term_cached(lhs, step, cache);
                    let rhs = self.rename_term_cached(rhs, step, cache);
                    self.manager.mk_eq(lhs, rhs)
                }
                TermKind::Ite(cond, then_branch, else_branch) => {
                    let cond = self.rename_term_cached(cond, step, cache);
                    let then_branch = self.rename_term_cached(then_branch, step, cache);
                    let else_branch = self.rename_term_cached(else_branch, step, cache);
                    self.manager.mk_ite(cond, then_branch, else_branch)
                }
                TermKind::Add(args) => {
                    let args: SmallVec<[TermId; 4]> = args
                        .into_iter()
                        .map(|arg| self.rename_term_cached(arg, step, cache))
                        .collect();
                    self.manager.mk_add(args)
                }
                TermKind::Sub(lhs, rhs) => {
                    let lhs = self.rename_term_cached(lhs, step, cache);
                    let rhs = self.rename_term_cached(rhs, step, cache);
                    self.manager.mk_sub(lhs, rhs)
                }
                TermKind::Mul(args) => {
                    let args: SmallVec<[TermId; 4]> = args
                        .into_iter()
                        .map(|arg| self.rename_term_cached(arg, step, cache))
                        .collect();
                    self.manager.mk_mul(args)
                }
                TermKind::Lt(lhs, rhs) => {
                    let lhs = self.rename_term_cached(lhs, step, cache);
                    let rhs = self.rename_term_cached(rhs, step, cache);
                    self.manager.mk_lt(lhs, rhs)
                }
                TermKind::Le(lhs, rhs) => {
                    let lhs = self.rename_term_cached(lhs, step, cache);
                    let rhs = self.rename_term_cached(rhs, step, cache);
                    self.manager.mk_le(lhs, rhs)
                }
                TermKind::Gt(lhs, rhs) => {
                    let lhs = self.rename_term_cached(lhs, step, cache);
                    let rhs = self.rename_term_cached(rhs, step, cache);
                    self.manager.mk_gt(lhs, rhs)
                }
                TermKind::Ge(lhs, rhs) => {
                    let lhs = self.rename_term_cached(lhs, step, cache);
                    let rhs = self.rename_term_cached(rhs, step, cache);
                    self.manager.mk_ge(lhs, rhs)
                }
                _ => term,
            },
            None => term,
        };

        cache.insert((term, step), renamed);
        renamed
    }
}

/// Tactic wrapper around the unrolling engine.
pub struct BmcUnrollTactic<'a> {
    manager: &'a mut TermManager,
    /// Number of transition steps to unroll.
    pub depth: usize,
}

impl<'a> BmcUnrollTactic<'a> {
    /// Create a new BMC unrolling tactic using the default depth of 5.
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager, depth: 5 }
    }

    /// Create a tactic with an explicit depth.
    pub fn with_depth(manager: &'a mut TermManager, depth: usize) -> Self {
        Self { manager, depth }
    }

    /// Create a tactic from an optional solver option string.
    pub fn from_option(manager: &'a mut TermManager, bmc_depth: Option<&str>) -> Self {
        let depth = bmc_depth
            .and_then(|raw| raw.parse::<usize>().ok())
            .unwrap_or(5);
        Self { manager, depth }
    }

    /// Apply BMC unrolling to a goal.
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut engine = BmcEngine::new(self.manager);
        let Some(assertions) = engine.unroll(&goal.assertions, self.depth) else {
            return Ok(TacticResult::NotApplicable);
        };
        if assertions.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions,
            precision: goal.precision,
        }]))
    }
}

fn next_state_name(source: &str, step: usize) -> (String, usize) {
    if let Some(base) = source.strip_suffix("_next") {
        (base.to_string(), step + 1)
    } else if let Some(base) = source.strip_suffix('\'') {
        (base.to_string(), step + 1)
    } else {
        (source.to_string(), step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bmc_unroll_produces_formula() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let x_next = manager.mk_var("x_next", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let one = manager.mk_int(1);

        let init = manager.mk_eq(x, zero);
        let x_plus_one = manager.mk_add([x, one]);
        let trans = manager.mk_eq(x_next, x_plus_one);
        let property = manager.mk_ge(x, zero);
        let goal = Goal::new(vec![init, trans, property]);

        let mut tactic = BmcUnrollTactic::with_depth(&mut manager, 2);
        let result = tactic.apply_mut(&goal).expect("test operation should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert!(!goals[0].assertions.is_empty());
            }
            other => panic!("expected unrolled goal, got {other:?}"),
        }
    }

    /// Verify that a doubly-next-state variable `x_next_next` is renamed correctly.
    /// `next_state_name` strips only the final `_next` suffix, so `x_next_next`
    /// becomes `x_next@{step+1}` — the inner `_next` is treated as part of the base
    /// name, not stripped again.
    #[test]
    fn test_bmc_unroll_handles_nested_next_state() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // `x_next_next` ends with `_next`, so it strips to base `x_next` and
        // target step = step + 1 = 1 (at depth 0).
        let x_nn = manager.mk_var("x_next_next", int_sort);
        let zero = manager.mk_int(0);

        // A minimal 3-assertion goal: init / trans / property
        let init = manager.mk_eq(x_nn, zero);
        let trans = manager.mk_eq(x_nn, zero);
        let property = manager.mk_ge(x_nn, zero);
        let goal = Goal::new(vec![init, trans, property]);

        let mut tactic = BmcUnrollTactic::with_depth(&mut manager, 1);
        let result = tactic.apply_mut(&goal).expect("tactic should apply on 3-assertion goal");

        let TacticResult::SubGoals(goals) = result else {
            panic!("expected SubGoals, not NotApplicable or Solved");
        };
        assert_eq!(goals.len(), 1);
        // After unrolling depth=1, the renamed assertions should reference variables
        // of the form `x_next@N`.  Inspect the first assertion (init at step 0):
        // `x_next_next` → base `x_next`, target step 1 → var name `x_next@1`.
        let first_assertion = goals[0].assertions[0];
        // Traverse the assertion to find the variable name.
        let found = find_var_name_in_term(&manager, first_assertion, "x_next@1");
        assert!(found, "expected variable `x_next@1` in renamed assertion");
    }

    /// Helper: recursively search for a variable whose name contains `needle`.
    fn find_var_name_in_term(manager: &TermManager, term: TermId, needle: &str) -> bool {
        let Some(node) = manager.get(term) else {
            return false;
        };
        match &node.kind {
            TermKind::Var(sym) => manager.resolve_str(*sym).contains(needle),
            TermKind::Eq(a, b) | TermKind::Le(a, b) | TermKind::Lt(a, b)
            | TermKind::Ge(a, b) | TermKind::Gt(a, b) | TermKind::Sub(a, b)
            | TermKind::Implies(a, b) => {
                find_var_name_in_term(manager, *a, needle)
                    || find_var_name_in_term(manager, *b, needle)
            }
            TermKind::Not(a) => find_var_name_in_term(manager, *a, needle),
            TermKind::And(args) | TermKind::Or(args) | TermKind::Add(args)
            | TermKind::Mul(args) => args
                .iter()
                .any(|&a| find_var_name_in_term(manager, a, needle)),
            TermKind::Ite(c, t, e) => {
                find_var_name_in_term(manager, *c, needle)
                    || find_var_name_in_term(manager, *t, needle)
                    || find_var_name_in_term(manager, *e, needle)
            }
            _ => false,
        }
    }

    /// Applying `BmcUnrollTactic::apply_mut` twice yields the same *shape*
    /// (one subgoal, same assertion count) even though variable names diverge
    /// on the second application (the `@N` suffix is not a `_next`/`'` marker,
    /// so step-suffixes accumulate).
    #[test]
    fn test_bmc_unroll_idempotent_under_reapply() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let x_next = manager.mk_var("x_next", int_sort);
        let zero = manager.mk_int(0);
        let one = manager.mk_int(1);

        let init = manager.mk_eq(x, zero);
        let x_plus_one = manager.mk_add([x, one]);
        let trans = manager.mk_eq(x_next, x_plus_one);
        let property = manager.mk_ge(x, zero);
        let goal = Goal::new(vec![init, trans, property]);

        // First application
        let mut tactic = BmcUnrollTactic::with_depth(&mut manager, 2);
        let first = tactic
            .apply_mut(&goal)
            .expect("first application should succeed");
        let TacticResult::SubGoals(first_goals) = first else {
            panic!("expected SubGoals after first application");
        };
        let first_assertion_count = first_goals[0].assertions.len();

        // Second application: re-apply on the subgoal produced by the first run.
        let subgoal = &first_goals[0];
        let mut tactic2 = BmcUnrollTactic::with_depth(&mut manager, 2);
        let second = tactic2
            .apply_mut(subgoal)
            .expect("second application should succeed");
        let TacticResult::SubGoals(second_goals) = second else {
            panic!("expected SubGoals after second application");
        };

        // Both results contain exactly one subgoal.
        assert_eq!(first_goals.len(), 1);
        assert_eq!(second_goals.len(), 1);
        // The assertion count is stable: unroll reads only assertions[0..3].
        assert_eq!(second_goals[0].assertions.len(), first_assertion_count);
    }

    /// `BmcUnrollTactic::from_option(Some("8"))` must configure depth = 8.
    #[test]
    fn test_bmc_unroll_from_option_depth() {
        let mut manager = TermManager::new();
        let tactic = BmcUnrollTactic::from_option(&mut manager, Some("8"));
        assert_eq!(tactic.depth, 8, "depth should be parsed from the option string");
    }
}
