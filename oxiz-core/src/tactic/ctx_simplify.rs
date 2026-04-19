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

        // Dead-branch ITE elimination post-pass (context = all sibling assertions)
        let ite_rewritten = eliminate_dead_ite_branches(&current_assertions, self.manager);
        let ite_changed = ite_rewritten != current_assertions;
        if ite_changed {
            current_assertions = ite_rewritten;
            changed = true;
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

/// Eliminate dead ITE branches using the set of sibling assertions as context.
///
/// For each assertion in `assertions`, recursively rewrites any ITE sub-term
/// whose condition (or its negation) is already in the context set, replacing
/// the ITE by the live branch.  Returns a new assertion list; if no change
/// occurred the returned `Vec` will be equal (by value) to the input slice.
fn eliminate_dead_ite_branches(
    assertions: &[TermId],
    manager: &mut TermManager,
) -> Vec<TermId> {
    let ctx: FxHashSet<TermId> = assertions.iter().copied().collect();
    assertions
        .iter()
        .map(|&term_id| rewrite_ite_in_context(term_id, &ctx, 0, manager))
        .collect()
}

/// Recursively rewrite ITE nodes within `term_id` using `ctx` as the set of
/// known-true assertions.  `depth` prevents unbounded recursion.
fn rewrite_ite_in_context(
    term_id: TermId,
    ctx: &FxHashSet<TermId>,
    depth: usize,
    manager: &mut TermManager,
) -> TermId {
    // Hard cap — safe to return original (sound: we just don't simplify deeper)
    if depth > 32 {
        return term_id;
    }

    let kind = match manager.get(term_id) {
        Some(t) => t.kind.clone(),
        None => return term_id,
    };

    match kind {
        TermKind::Ite(cond, then_branch, else_branch) => {
            let not_cond = manager.mk_not(cond);
            if ctx.contains(&cond) {
                // Condition is known true — take then-branch
                rewrite_ite_in_context(then_branch, ctx, depth + 1, manager)
            } else if ctx.contains(&not_cond) {
                // Condition is known false — take else-branch
                rewrite_ite_in_context(else_branch, ctx, depth + 1, manager)
            } else {
                // Descend into branches with augmented, non-overlapping contexts
                let mut ctx_then = ctx.clone();
                ctx_then.insert(cond);
                let new_then =
                    rewrite_ite_in_context(then_branch, &ctx_then, depth + 1, manager);

                let mut ctx_else = ctx.clone();
                ctx_else.insert(not_cond);
                let new_else =
                    rewrite_ite_in_context(else_branch, &ctx_else, depth + 1, manager);

                if new_then == then_branch && new_else == else_branch {
                    term_id // no structural change
                } else {
                    manager.mk_ite(cond, new_then, new_else)
                }
            }
        }
        // Non-ITE terms: no rewrite at this level
        _ => term_id,
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

    // ── EP-3: dead-branch ITE elimination tests ──────────────────────────────

    /// Goal: [cond, If(cond, foo, bar)] → ITE replaced by `foo` because
    /// `cond` is present in the sibling-assertion context.
    #[test]
    fn test_ite_eliminates_when_cond_in_context() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let cond = manager.mk_var("cond", bool_sort);
        let foo = manager.mk_var("foo", bool_sort);
        let bar = manager.mk_var("bar", bool_sort);
        let ite = manager.mk_ite(cond, foo, bar);

        let goal = Goal::new(vec![cond, ite]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("tactic should not error");

        // The ITE should be eliminated to `foo`; `cond` (true) filters out
        // or remains, but either way `bar` must not appear in the assertions.
        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                for &a in &goals[0].assertions {
                    assert_ne!(a, bar, "dead branch `bar` should have been eliminated");
                }
                assert!(
                    goals[0].assertions.contains(&foo),
                    "live branch `foo` should remain"
                );
            }
            TacticResult::Solved(SolveResult::Sat) => {
                // All assertions collapsed to true — also acceptable
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    /// Goal: [Not(cond), If(cond, foo, bar)] → ITE replaced by `bar` because
    /// `Not(cond)` means the condition is known false.
    #[test]
    fn test_ite_eliminates_when_neg_cond_in_context() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let cond = manager.mk_var("cond2", bool_sort);
        let foo = manager.mk_var("foo2", bool_sort);
        let bar = manager.mk_var("bar2", bool_sort);
        let not_cond = manager.mk_not(cond);
        let ite = manager.mk_ite(cond, foo, bar);

        let goal = Goal::new(vec![not_cond, ite]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("tactic should not error");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                for &a in &goals[0].assertions {
                    assert_ne!(a, foo, "dead branch `foo` should have been eliminated");
                }
                assert!(
                    goals[0].assertions.contains(&bar),
                    "live branch `bar` should remain"
                );
            }
            TacticResult::Solved(SolveResult::Sat) => {}
            other => panic!("unexpected result: {other:?}"),
        }
    }

    /// Goal: [a, If(a, If(b, p, q), r)]
    /// Outer ITE is eliminated (a is in context) giving If(b, p, q).
    /// The inner ITE is NOT eliminated because `b` is not in the root context.
    #[test]
    fn test_ite_descends_with_augmented_ctx() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let a = manager.mk_var("a3", bool_sort);
        let b = manager.mk_var("b3", bool_sort);
        let p = manager.mk_var("p3", bool_sort);
        let q = manager.mk_var("q3", bool_sort);
        let r = manager.mk_var("r3", bool_sort);

        let inner_ite = manager.mk_ite(b, p, q);
        let outer_ite = manager.mk_ite(a, inner_ite, r);

        let goal = Goal::new(vec![a, outer_ite]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("tactic should not error");

        // After eliminating outer ITE, we should get `inner_ite` (= If(b,p,q))
        // in the assertions, and `r` should not be present.
        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                for &assertion in &goals[0].assertions {
                    assert_ne!(assertion, r, "`r` is dead and must not appear");
                }
                // inner_ite should be in the assertions (b not in root ctx)
                assert!(
                    goals[0].assertions.contains(&inner_ite),
                    "inner ITE If(b,p,q) should remain intact"
                );
            }
            TacticResult::Solved(SolveResult::Sat) => {}
            other => panic!("unexpected result: {other:?}"),
        }
    }

    /// Construct a 35-deep nested ITE chain and run the tactic.
    /// The test asserts the tactic completes without panicking or looping.
    #[test]
    fn test_ite_recursion_depth_cap() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;

        // Build: ITE(c, ITE(c, ITE(c, ... (35 deep) ..., base), base), base)
        let cond = manager.mk_var("deep_cond", bool_sort);
        let base = manager.mk_var("deep_base", bool_sort);
        let mut term = base;
        for _ in 0..35 {
            term = manager.mk_ite(cond, term, base);
        }

        let goal = Goal::new(vec![cond, term]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);
        // Must not panic or loop; result value is unconstrained
        let result = tactic.apply_mut(&goal);
        assert!(
            result.is_ok(),
            "tactic must not error on deep ITE: {result:?}"
        );
    }

    /// Running `apply_mut` on a goal that resolves to `Solved(Unsat)` must
    /// preserve that status even after the ITE post-pass runs.  Also verifies
    /// the tactic does not accidentally flip statuses when no ITEs are present.
    #[test]
    fn test_apply_mut_status_preserved() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // x = 5, x < 3  →  contradiction  →  Unsat
        // (This already resolves to Unsat without any ITEs, so the post-pass
        //  must leave the result intact.)
        let x = manager.mk_var("x_ep3", int_sort);
        let five = manager.mk_int(5);
        let three = manager.mk_int(3);
        let eq = manager.mk_eq(x, five);
        let lt = manager.mk_lt(x, three);

        let goal = Goal::new(vec![eq, lt]);
        let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("tactic should not error");

        assert!(
            matches!(result, TacticResult::Solved(SolveResult::Unsat)),
            "expected Unsat status preserved, got {result:?}"
        );
    }
}
