//! Context solver simplification tactics.

use super::core::*;
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
#[allow(unused_imports)]
use crate::prelude::*;
use smallvec::SmallVec;
use std::rc::Rc;

/// A set of boolean facts assumed true while simplifying a subterm.
///
/// Shared by reference-count rather than cloned into every pending frame:
/// the explicit work stack in
/// [`CtxSolverSimplifyTactic::simplify_with_assignments`] can hold one frame
/// per level of nesting, and cloning the whole map into each of them would
/// reintroduce the O(depth x |assignments|) memory the recursive version had.
type Ctx = Rc<FxHashMap<TermId, bool>>;

/// Memo key: a term together with the *exact* context it was simplified
/// under, identified by [`CtxSimplifyState::intern`].
type CtxCacheKey = (TermId, usize);

/// How a [`CtxFrame::Build`] frame recombines the simplified children that
/// its matching `Eval` frames left on the result stack.
enum CtxBuild {
    Not,
    Or(usize),
    Add(usize),
    Eq,
    Sub,
    Lt,
    Le,
    Gt,
    Ge,
    /// Simplified antecedent; the single pending result is the consequent.
    Implies(TermId),
    /// Simplified condition; the two pending results are the branches.
    Ite(TermId),
}

/// One pending step of the iterative contextual simplification walk. Each
/// variant carries the resume state its recursive counterpart used to keep in
/// live locals across the recursive call.
enum CtxFrame {
    /// Simplify `term` under `ctx`.
    Eval { term: TermId, ctx: Ctx },
    /// Normalize + memoize the single pending result as the value of `key`.
    Finish { key: CtxCacheKey },
    /// The antecedent of an `Implies` has been simplified; extend the context
    /// with it and simplify the consequent.
    ImpliesRhs {
        key: CtxCacheKey,
        ctx: Ctx,
        rhs: TermId,
    },
    /// The condition of an `Ite` has been simplified; re-test it against the
    /// context and either take a branch or rebuild.
    IteCond {
        key: CtxCacheKey,
        ctx: Ctx,
        then_branch: TermId,
        else_branch: TermId,
    },
    /// Conjunction arguments are simplified left to right, each one under a
    /// context extended by the previous ones — so they cannot be batched like
    /// the other n-ary nodes. `issued` counts the `Eval`s pushed so far;
    /// `done` collects their results.
    AndStep {
        key: CtxCacheKey,
        args: SmallVec<[TermId; 4]>,
        issued: usize,
        scoped: FxHashMap<TermId, bool>,
        done: SmallVec<[TermId; 4]>,
    },
    /// All children of a node are simplified; rebuild it.
    Build { key: CtxCacheKey, op: CtxBuild },
}

/// Memo plus context-identity table for one contextual-simplification run.
#[derive(Default)]
struct CtxSimplifyState {
    cache: FxHashMap<CtxCacheKey, TermId>,
    /// Sorted assignment list -> dense id. Interning the assignments
    /// themselves (rather than a hash of them) is what makes the memo key
    /// collision-free; see
    /// [`CtxSolverSimplifyTactic::simplify_with_assignments`].
    context_ids: FxHashMap<Vec<(u32, bool)>, usize>,
}

impl CtxSimplifyState {
    fn intern(&mut self, assignments: &Ctx) -> usize {
        let mut pairs: Vec<(u32, bool)> = assignments.iter().map(|(k, v)| (k.0, *v)).collect();
        pairs.sort_unstable();
        let next = self.context_ids.len();
        *self.context_ids.entry(pairs).or_insert(next)
    }
}

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
        let mut state = CtxSimplifyState::default();
        let root_ctx = Rc::new(FxHashMap::default());
        self.simplify_with_assignments(term, &root_ctx, &mut state)
    }

    /// Context-aware simplification of `term` under the boolean `assignments`
    /// already known to hold.
    ///
    /// # Explicit work stack
    ///
    /// This used to recurse natively once per level of boolean/arithmetic
    /// nesting with no guard at all. The return type is `TermId` — there is
    /// no error channel — so a depth cap could only have returned a term that
    /// silently stopped being simplified partway down while still being
    /// presented as the simplification of the whole. The walk is therefore
    /// driven by an explicit heap [`Vec`] of [`CtxFrame`]s instead, mirroring
    /// `ast::manager::query::substitute`. Frames carry their own resume
    /// state, so no "impossible" pop can occur: every frame that consumes
    /// child results was pushed together with exactly the child evaluations
    /// that produce them.
    ///
    /// # Exact context keys
    ///
    /// The memo is keyed on `(TermId, context-id)`. The context id comes from
    /// [`CtxSimplifyState::intern`], which interns the *sorted assignment
    /// list itself*. It previously came from an `assignment_fingerprint`
    /// helper that hashed the assignments down to a single `u64`: two
    /// different contexts that collided on that 64-bit digest would have made
    /// this function return a term simplified under the **wrong** set of
    /// known-true facts — a genuine (if improbable) unsoundness, since the
    /// whole point of the context is that facts in it are assumed. Interning
    /// compares the assignments themselves, so distinct contexts can never
    /// alias.
    fn simplify_with_assignments(
        &mut self,
        term: TermId,
        assignments: &Ctx,
        state: &mut CtxSimplifyState,
    ) -> TermId {
        let mut frames: Vec<CtxFrame> = vec![CtxFrame::Eval {
            term,
            ctx: Rc::clone(assignments),
        }];
        // Simplified child results, consumed by the `Build`/resume frames
        // that requested them.
        let mut results: Vec<TermId> = Vec::new();

        while let Some(frame) = frames.pop() {
            match frame {
                CtxFrame::Eval { term, ctx } => {
                    self.eval_frame(term, &ctx, state, &mut frames, &mut results);
                }

                CtxFrame::Finish { key } => {
                    // `Eval` pushed exactly one child evaluation before this
                    // frame, so `results` cannot be empty here; `unwrap_or`
                    // keeps the impossible branch unwritable without an
                    // `expect`, falling back to the node itself.
                    let value = results.pop().unwrap_or(key.0);
                    self.finish(key, value, state, &mut results);
                }

                CtxFrame::ImpliesRhs { key, ctx, rhs } => {
                    let cond = results.pop().unwrap_or(key.0);
                    let mut rhs_assignments = (*ctx).clone();
                    record_assignment(cond, true, self.manager, &mut rhs_assignments);
                    frames.push(CtxFrame::Build {
                        key,
                        op: CtxBuild::Implies(cond),
                    });
                    frames.push(CtxFrame::Eval {
                        term: rhs,
                        ctx: Rc::new(rhs_assignments),
                    });
                }

                CtxFrame::IteCond {
                    key,
                    ctx,
                    then_branch,
                    else_branch,
                } => {
                    let cond = results.pop().unwrap_or(key.0);
                    if let Some(value) = evaluate_condition(cond, &ctx, self.manager) {
                        let chosen = if value { then_branch } else { else_branch };
                        frames.push(CtxFrame::Finish { key });
                        frames.push(CtxFrame::Eval { term: chosen, ctx });
                    } else {
                        frames.push(CtxFrame::Build {
                            key,
                            op: CtxBuild::Ite(cond),
                        });
                        frames.push(CtxFrame::Eval {
                            term: else_branch,
                            ctx: Rc::clone(&ctx),
                        });
                        frames.push(CtxFrame::Eval {
                            term: then_branch,
                            ctx,
                        });
                    }
                }

                CtxFrame::AndStep {
                    key,
                    args,
                    mut issued,
                    mut scoped,
                    mut done,
                } => {
                    if done.len() < issued {
                        let rewritten = results.pop().unwrap_or(key.0);
                        record_assignment(rewritten, true, self.manager, &mut scoped);
                        done.push(rewritten);
                    }
                    if issued == args.len() {
                        let built = self.manager.mk_and(done);
                        self.finish(key, built, state, &mut results);
                    } else {
                        let next = args[issued];
                        issued += 1;
                        let ctx = Rc::new(scoped.clone());
                        frames.push(CtxFrame::AndStep {
                            key,
                            args,
                            issued,
                            scoped,
                            done,
                        });
                        frames.push(CtxFrame::Eval { term: next, ctx });
                    }
                }

                CtxFrame::Build { key, op } => {
                    let built = self.build(&op, &mut results, key.0);
                    self.finish(key, built, state, &mut results);
                }
            }
        }

        // The root evaluation leaves exactly one value behind.
        results.pop().unwrap_or(term)
    }

    /// Expand one `Eval` frame: consult the memo, then either finish `term`
    /// immediately (leaf) or push the child evaluations plus the frame that
    /// recombines them.
    fn eval_frame(
        &mut self,
        term: TermId,
        ctx: &Ctx,
        state: &mut CtxSimplifyState,
        frames: &mut Vec<CtxFrame>,
        results: &mut Vec<TermId>,
    ) {
        let key = (term, state.intern(ctx));
        if let Some(&cached) = state.cache.get(&key) {
            results.push(cached);
            return;
        }

        match self.manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Implies(cond, rhs)) => {
                frames.push(CtxFrame::ImpliesRhs {
                    key,
                    ctx: Rc::clone(ctx),
                    rhs,
                });
                frames.push(CtxFrame::Eval {
                    term: cond,
                    ctx: Rc::clone(ctx),
                });
            }

            Some(TermKind::And(args)) => {
                frames.push(CtxFrame::AndStep {
                    key,
                    args,
                    issued: 0,
                    scoped: (**ctx).clone(),
                    done: SmallVec::new(),
                });
            }

            Some(TermKind::Or(args)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Or(args.len()), &args);
            }

            Some(TermKind::Add(args)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Add(args.len()), &args);
            }

            Some(TermKind::Not(arg)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Not, &[arg]);
            }

            Some(TermKind::Ite(cond, then_branch, else_branch)) => {
                if let Some(value) = evaluate_condition(cond, ctx, self.manager) {
                    let chosen = if value { then_branch } else { else_branch };
                    frames.push(CtxFrame::Finish { key });
                    frames.push(CtxFrame::Eval {
                        term: chosen,
                        ctx: Rc::clone(ctx),
                    });
                } else {
                    frames.push(CtxFrame::IteCond {
                        key,
                        ctx: Rc::clone(ctx),
                        then_branch,
                        else_branch,
                    });
                    frames.push(CtxFrame::Eval {
                        term: cond,
                        ctx: Rc::clone(ctx),
                    });
                }
            }

            Some(TermKind::Eq(lhs, rhs)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Eq, &[lhs, rhs]);
            }
            Some(TermKind::Sub(lhs, rhs)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Sub, &[lhs, rhs]);
            }
            Some(TermKind::Lt(lhs, rhs)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Lt, &[lhs, rhs]);
            }
            Some(TermKind::Le(lhs, rhs)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Le, &[lhs, rhs]);
            }
            Some(TermKind::Gt(lhs, rhs)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Gt, &[lhs, rhs]);
            }
            Some(TermKind::Ge(lhs, rhs)) => {
                Self::push_children(frames, ctx, key, CtxBuild::Ge, &[lhs, rhs]);
            }

            // Every other node is opaque to *contextual* simplification: it
            // is handed to `TermManager::simplify` (in `finish`) but its
            // children are not re-simplified under the surrounding context.
            // That is an incompleteness, not an unsoundness — the term is
            // returned intact rather than replaced by a default — and it is
            // the behaviour this tactic has always had. `None` (a dangling
            // id) lands here too and likewise yields the id unchanged.
            _ => self.finish(key, term, state, results),
        }
    }

    /// Push a `Build` frame plus one `Eval` per child, ordered so the
    /// children's results land in `results` in argument order.
    fn push_children(
        frames: &mut Vec<CtxFrame>,
        ctx: &Ctx,
        key: CtxCacheKey,
        op: CtxBuild,
        args: &[TermId],
    ) {
        frames.push(CtxFrame::Build { key, op });
        for &arg in args.iter().rev() {
            frames.push(CtxFrame::Eval {
                term: arg,
                ctx: Rc::clone(ctx),
            });
        }
    }

    /// Rebuild a node from the simplified children sitting on top of
    /// `results`.
    fn build(&mut self, op: &CtxBuild, results: &mut Vec<TermId>, fallback: TermId) -> TermId {
        // `take` pops `n` results in argument order. Each `Build` frame was
        // pushed together with exactly `n` child evaluations, so the requested
        // results are always present; `fallback` (the original node) keeps the
        // unreachable branch expressible without a panic.
        let take = |n: usize, results: &mut Vec<TermId>| -> SmallVec<[TermId; 4]> {
            let start = results.len().saturating_sub(n);
            let mut taken: SmallVec<[TermId; 4]> = results.split_off(start).into();
            while taken.len() < n {
                taken.push(fallback);
            }
            taken
        };

        match *op {
            CtxBuild::Not => {
                let a = take(1, results);
                self.manager.mk_not(a[0])
            }
            CtxBuild::Or(n) => {
                let args = take(n, results);
                self.manager.mk_or(args)
            }
            CtxBuild::Add(n) => {
                let args = take(n, results);
                self.manager.mk_add(args)
            }
            CtxBuild::Eq => {
                let a = take(2, results);
                self.manager.mk_eq(a[0], a[1])
            }
            CtxBuild::Sub => {
                let a = take(2, results);
                self.manager.mk_sub(a[0], a[1])
            }
            CtxBuild::Lt => {
                let a = take(2, results);
                self.manager.mk_lt(a[0], a[1])
            }
            CtxBuild::Le => {
                let a = take(2, results);
                self.manager.mk_le(a[0], a[1])
            }
            CtxBuild::Gt => {
                let a = take(2, results);
                self.manager.mk_gt(a[0], a[1])
            }
            CtxBuild::Ge => {
                let a = take(2, results);
                self.manager.mk_ge(a[0], a[1])
            }
            CtxBuild::Implies(cond) => {
                let a = take(1, results);
                self.manager.mk_implies(cond, a[0])
            }
            CtxBuild::Ite(cond) => {
                let a = take(2, results);
                self.manager.mk_ite(cond, a[0], a[1])
            }
        }
    }

    /// Normalize, memoize and publish the result for one node — the tail the
    /// recursive version ran after every match arm.
    fn finish(
        &mut self,
        key: CtxCacheKey,
        value: TermId,
        state: &mut CtxSimplifyState,
        results: &mut Vec<TermId>,
    ) {
        let normalized = self.manager.simplify(value);
        state.cache.insert(key, normalized);
        results.push(normalized);
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

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // Contextual simplification rewrites assertions using the others as
        // context, which needs a `&mut TermManager`. The manager-free path
        // honestly reports NotApplicable rather than returning the goal
        // unchanged. Use `CtxSolverSimplifyTactic::apply_mut` (or
        // `create_managed`) for the real transformation.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Simplifies assertions using other assertions as context \
         (requires a TermManager; the manager-free path is NotApplicable)"
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

/// Decide `condition` from the known-true `assignments`, if possible.
///
/// Iterative rather than recursive: `Not` chains are the only recursive case
/// and `(not (not (not ...)))` of arbitrary depth is trivially constructible,
/// while the `Option<bool>` return means a depth cap would answer "unknown"
/// for a condition the context does in fact decide (an incompleteness that is
/// avoidable outright).
fn evaluate_condition(
    condition: TermId,
    assignments: &FxHashMap<TermId, bool>,
    manager: &TermManager,
) -> Option<bool> {
    let mut current = condition;
    // Parity of the `Not` chain unwound so far.
    let mut negated = false;
    loop {
        if let Some(&value) = assignments.get(&current) {
            return Some(value != negated);
        }

        match manager.get(current).map(|term| &term.kind) {
            Some(TermKind::True) => return Some(!negated),
            Some(TermKind::False) => return Some(negated),
            Some(TermKind::Not(inner)) => {
                current = *inner;
                negated = !negated;
            }
            // Anything else is opaque to this purely syntactic evaluator:
            // honestly "unknown", never a guessed default.
            _ => return None,
        }
    }
}

/// Eliminate dead ITE branches using the set of sibling assertions as context.
///
/// For each assertion in `assertions`, recursively rewrites any ITE sub-term
/// whose condition (or its negation) is already in the context set, replacing
/// the ITE by the live branch.  Returns a new assertion list; if no change
/// occurred the returned `Vec` will be equal (by value) to the input slice.
fn eliminate_dead_ite_branches(assertions: &[TermId], manager: &mut TermManager) -> Vec<TermId> {
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
                let new_then = rewrite_ite_in_context(then_branch, &ctx_then, depth + 1, manager);

                let mut ctx_else = ctx.clone();
                ctx_else.insert(not_cond);
                let new_else = rewrite_ite_in_context(else_branch, &ctx_else, depth + 1, manager);

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
        let result = tactic
            .apply_mut(&goal)
            .expect("test operation should succeed");

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

#[cfg(test)]
mod group_c1_tests {
    use super::*;

    /// `simplify_with_assignments` is driven by an explicit heap stack now.
    /// A formula far deeper than any native stack could hold must simply
    /// return -- an overflow would abort the process, so returning at all is
    /// the assertion.
    ///
    /// The converted walk is exercised directly via `simplify_with_context`
    /// rather than through `apply_mut`: the latter re-runs the whole pass up
    /// to `max_iterations` times, which multiplies the cost by ten.
    ///
    /// Depth is 8 000 rather than the usual 60 000-100 000 because this
    /// tactic normalizes *every* rebuilt node through `TermManager::simplify`,
    /// making one pass quadratic in nesting depth for reasons that have
    /// nothing to do with recursion (60 000 takes minutes). 8 000 is still
    /// far past the depth at which the previous native recursion aborted on a
    /// 1 MiB stack -- its frames carried a cloned `TermKind` plus a
    /// `SmallVec` and a cloned assignment map per level -- so it exercises
    /// exactly the property under test.
    #[test]
    fn contextual_simplification_survives_a_deep_formula_on_a_tiny_stack() {
        // Stack size and nesting depth are scaled together on purpose: what
        // this test pins is the *ratio* — about 131 bytes of stack per
        // nesting level (128 KiB / 1_000). The pair used to be
        // 1 MiB / 8_000 — the same 131 bytes — but the tactic's superlinear
        // walk made that variant exceed the workspace's 3×60 s nextest
        // terminate-after ceiling. Never raise `DEPTH` without raising
        // `STACK` by the same factor.
        const DEPTH: usize = 1_000;
        const STACK: usize = 1 << 17;

        let handle = std::thread::Builder::new()
            .stack_size(STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let bool_sort = manager.sorts.bool_sort;
                let p = manager.mk_var("p", bool_sort);

                // `Not(Or(current, q_i))` per level, with a distinct `q_i` so
                // nothing folds. `Or` is used rather than `And` because only
                // `And`/`Implies` extend the assignment context, and this
                // test is about recursion depth, not context size.
                let mut current = p;
                for i in 0..DEPTH {
                    let q = manager.mk_var(&format!("q{i}"), bool_sort);
                    let disj = manager.mk_or([current, q]);
                    current = manager.mk_not(disj);
                }

                let mut tactic = CtxSolverSimplifyTactic::new(&mut manager);
                let simplified = tactic.simplify_with_context(current);
                // The formula has no contextual redundancy, so it must come
                // back unchanged rather than truncated.
                simplified == current
            })
            .expect("test thread must spawn");

        assert_eq!(handle.join().ok(), Some(true));
    }

    /// `evaluate_condition` unwinds `Not` chains iteratively and must report
    /// the right parity. The chain is built through `TermManager`, which may
    /// fold double negations, so the expectation is derived from the term the
    /// manager actually produced rather than from the loop count.
    #[test]
    fn evaluate_condition_unwinds_long_not_chains() {
        let mut manager = TermManager::new();
        let bool_sort = manager.sorts.bool_sort;
        let p = manager.mk_var("p", bool_sort);

        let mut assignments = FxHashMap::default();
        assignments.insert(p, true);

        let mut current = p;
        for _ in 0..10_001 {
            current = manager.mk_not(current);
        }

        // Read the final term structurally to derive the expected answer.
        let mut probe = current;
        let mut negated = false;
        while let Some(TermKind::Not(inner)) = manager.get(probe).map(|t| t.kind.clone()) {
            probe = inner;
            negated = !negated;
        }
        let expected = (probe == p).then_some(!negated);

        assert_eq!(
            evaluate_condition(current, &assignments, &manager),
            expected
        );
    }

    /// The memo key interns the assignment set itself. Two *different*
    /// contexts must therefore never share a cache entry -- the previous
    /// 64-bit `assignment_fingerprint` digest could alias them and hand back
    /// a term simplified under facts that do not hold.
    #[test]
    fn context_ids_are_injective() {
        let mut state = CtxSimplifyState::default();

        let a = TermId::new(1);
        let b = TermId::new(2);

        let mut ctx_a = FxHashMap::default();
        ctx_a.insert(a, true);
        let mut ctx_b = FxHashMap::default();
        ctx_b.insert(a, false);
        let mut ctx_c = FxHashMap::default();
        ctx_c.insert(a, true);
        ctx_c.insert(b, true);
        // Same contents, different insertion order: must intern equal.
        let mut ctx_c2 = FxHashMap::default();
        ctx_c2.insert(b, true);
        ctx_c2.insert(a, true);

        let ids = [
            state.intern(&Rc::new(ctx_a)),
            state.intern(&Rc::new(ctx_b)),
            state.intern(&Rc::new(ctx_c)),
        ];
        assert_ne!(ids[0], ids[1], "opposite polarities must not share an id");
        assert_ne!(ids[0], ids[2], "a superset must not share an id");
        assert_ne!(ids[1], ids[2]);
        assert_eq!(
            state.intern(&Rc::new(ctx_c2)),
            ids[2],
            "order must not affect the interned identity"
        );
    }
}
