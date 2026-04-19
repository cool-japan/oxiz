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
    depth: usize,
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
        let trans = manager.mk_eq(x_next, manager.mk_add([x, one]));
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
}
