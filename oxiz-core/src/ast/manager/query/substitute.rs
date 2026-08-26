//! Iterative, capture-avoiding term substitution.
//!
//! Split out of `ast/manager/query.rs`. [`TermManager::substitute`] used to
//! recurse natively once per level of term nesting, guarded by a shared
//! depth cap (`MAX_QUERY_RECURSION_DEPTH` = 1000, since removed) that bailed
//! out by returning the term *unchanged* past the cap. That guard was sound
//! for [`TermManager::simplify`] (see `super::simplify`; a missed
//! optimisation, not a wrong answer) but not for `substitute`: a caller that
//! substitutes and silently gets back a not-fully-substituted term will
//! reason about the wrong formula. A workspace-wide caller audit found
//! several call sites where that would be a genuine soundness exposure --
//! not merely degraded output -- including:
//!
//! * proof-instantiation checking (`oxiz-theories::checking::proof`), which
//!   recomputes `body[subst]` to verify a claimed proof step; a
//!   not-fully-substituted recomputation could make the checker accept an
//!   invalid instantiation step;
//! * quantifier elimination (`oxiz-core::qe::*`, `oxiz-theories::array::
//!   quantifier_elim`), which returns the substituted formula directly as
//!   the "equivalent" quantifier-free result;
//! * Spacer/PDR inductive-invariant checking and BMC unrolling
//!   (`oxiz-spacer::{pdr,bmc,invariant,smt}`), where an under-substituted
//!   lemma or transition constraint could make a non-inductive candidate
//!   look inductive;
//! * Skolemization (`oxiz-core::ast::normal_forms`), where a stray
//!   unsubstituted bound variable would corrupt the pre-CNF pipeline.
//!
//! So this is converted to an explicit heap stack rather than capped: a
//! `Vec`-backed work list replaces the native call stack, and can grow
//! arbitrarily (bounded by memory, not the fixed native stack), removing
//! the need for a cap at all.
//!
//! # Preserving capture-avoidance
//!
//! The recursive version alpha-renamed bound variables to avoid capturing a
//! replacement term's free variables (see [`TermManager::prepare_binder_subst`],
//! unchanged below) and, whenever descending into a binder
//! (`Forall`/`Exists`/`Let`/`Match`) under a *different* effective
//! substitution than its parent, allocated a fresh, throwaway memo cache
//! for that scope (a local `inner_cache`) -- because whether a given
//! `TermId` has already been "substituted" depends on *which* substitution
//! is in force, not just its identity. The iterative version reproduces
//! this exactly via [`SubstContext`]: context `0` is the outer
//! `(subst, cache)` pair passed in by the caller, and descending into a
//! binder scope with a nonempty effective substitution opens a new context
//! (a fresh owned substitution map plus a fresh, empty cache) rather than
//! reusing the parent's. A binder scope whose effective substitution is
//! *empty* (every relevant mapping was shadowed away) resolves to the
//! original term immediately, without visiting its body at all -- matching
//! the recursive version's `None => id` short-circuit precisely.

use super::TermManager;
use crate::ast::term::{MatchCase, TermId, TermKind};
use crate::ast::traversal::{collect_free_vars_including_patterns, collect_subterms, get_children};
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use smallvec::SmallVec;

/// Result of [`TermManager::prepare_binder_subst`]: the effective
/// substitution to apply inside the binder scope, paired with any fresh
/// `(name, sort)` bindings introduced by alpha-renaming to avoid capture.
type BinderSubstPrep = (FxHashMap<TermId, TermId>, SmallVec<[(Spur, SortId); 2]>);

/// One substitution scope opened while iteratively walking a term: the
/// substitution map in effect (always owned -- the outer, caller-supplied
/// map is cloned once up front, see [`TermManager::substitute_cached`]) and
/// the memo cache for terms resolved under it.
///
/// A binder scope's cache is always fresh and is discarded once that
/// scope's work is done, exactly mirroring the recursive version's local
/// `inner_cache`: a subterm's substituted value depends on *which*
/// effective substitution is in force, not just its `TermId`, so caches
/// must not be shared across scopes with different substitutions.
struct SubstContext {
    subst: FxHashMap<TermId, TermId>,
    cache: FxHashMap<TermId, TermId>,
}

impl SubstContext {
    /// `id`'s substituted value under this context, if already known
    /// without visiting its children: a direct substitution-map hit
    /// (checked first, matching the order the recursive version's entry
    /// checks used) or an already-memoized `cache` entry.
    fn resolved(&self, id: TermId) -> Option<TermId> {
        self.subst
            .get(&id)
            .copied()
            .or_else(|| self.cache.get(&id).copied())
    }
}

/// Per-case outcome of expanding a `Match` node's cases (see
/// [`TermManager::expand_match`]): each case independently either survives
/// untouched (its bound names shadowed away every relevant mapping, exactly
/// as the recursive `subst_match`'s `None => new_cases.push(case)` arm) or
/// needs its body rebuilt from a value resolved in its own fresh context.
enum MatchCasePlan {
    Unchanged(MatchCase),
    Rewrite {
        constructor: Option<Spur>,
        new_bound: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        ctx: usize,
    },
}

/// One pending step of the iterative substitution walk. `ctx`/`body_ctx`
/// index into the `contexts` arena threaded through
/// [`TermManager::substitute_cached`]; see [`SubstContext`].
enum SubstStep {
    /// Resolve `id`'s substituted value under `contexts[ctx]`. A no-op if
    /// already resolved; otherwise schedules the matching combine step
    /// (and, for binder nodes, may open a fresh context) plus whichever
    /// children need resolving first.
    Expand { id: TermId, ctx: usize },
    /// Rebuild a non-binder node from its already-resolved children (see
    /// [`TermManager::rebuild_substituted`]) and memoize the result.
    Combine {
        id: TermId,
        ctx: usize,
        kind: TermKind,
        sort: SortId,
    },
    /// Rebuild a `Forall` (`is_exists = false`) or `Exists` node from a
    /// body/patterns resolved in `body_ctx`.
    CombineQuantifier {
        id: TermId,
        ctx: usize,
        sort: SortId,
        is_exists: bool,
        new_vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        body_ctx: usize,
    },
    /// Rebuild a `Let` node. Binding values always resolve in `ctx` (the
    /// outer scope); `body` resolves in `body_ctx` when one was opened for
    /// it, or is reused unchanged (`body_ctx: None`) when every relevant
    /// mapping was shadowed away.
    CombineLet {
        id: TermId,
        ctx: usize,
        sort: SortId,
        bindings: SmallVec<[(Spur, TermId); 2]>,
        final_names: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        body_ctx: Option<usize>,
    },
    /// Rebuild a `Match` node. `scrutinee` resolves in `ctx`; each case
    /// follows its own [`MatchCasePlan`].
    CombineMatch {
        id: TermId,
        ctx: usize,
        sort: SortId,
        scrutinee: TermId,
        case_plans: SmallVec<[MatchCasePlan; 4]>,
    },
}

impl TermManager {
    /// Substitute variables in a term according to a mapping.
    pub fn substitute(&mut self, id: TermId, subst: &FxHashMap<TermId, TermId>) -> TermId {
        self.substitute_cached(id, subst, &mut FxHashMap::default())
    }

    /// Substitute with memoization, using an explicit heap stack instead of
    /// native recursion (see the module doc comment).
    ///
    /// Every `TermKind` variant is handled explicitly in
    /// [`TermManager::rebuild_substituted`] -- there is deliberately no
    /// catch-all arm, so a newly added variant fails to compile there
    /// rather than being silently skipped (which would drop solved
    /// equations while leaving occurrences in place, yielding wrong
    /// sat/unsat results and wrong models).
    ///
    /// Substitution is capture-avoiding: descending into a binder
    /// (`Forall`, `Exists`, `Let`, `Match`) drops shadowed variables from
    /// the substitution domain and alpha-renames any bound variable whose
    /// name would otherwise capture a free variable of a replacement term.
    pub(in crate::ast::manager) fn substitute_cached(
        &mut self,
        id: TermId,
        subst: &FxHashMap<TermId, TermId>,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        // Fast path matching the entry checks every recursive call used to
        // make: a direct substitution-map hit, or already memoized from a
        // prior call sharing this `cache` (e.g. `SubstitutionBuilder::apply`
        // in `ast/manager/mod.rs`, which reuses one cache across many
        // `substitute_cached` calls for the same mapping).
        if let Some(&replacement) = subst.get(&id) {
            return replacement;
        }
        if let Some(&result) = cache.get(&id) {
            return result;
        }

        // `contexts[0]` is the outer scope: the caller's `cache` (taken,
        // put back before returning) paired with an owned clone of `subst`
        // (cloned once, up front, so every `SubstContext` -- including ones
        // opened later for binder scopes -- uniformly owns its map).
        let mut contexts: Vec<SubstContext> = vec![SubstContext {
            subst: subst.clone(),
            cache: core::mem::take(cache),
        }];
        let mut work: Vec<SubstStep> = vec![SubstStep::Expand { id, ctx: 0 }];

        while let Some(step) = work.pop() {
            self.run_substitute_step(step, &mut contexts, &mut work);
        }

        // By now `work` is empty, so every step it ever held -- including
        // the root's `Expand`/`Combine` -- has run to completion, so the
        // root is resolved in `contexts[0]` (`unwrap_or` is a defensive
        // fallback for that structurally unreachable case).
        let result = contexts[0].resolved(id).unwrap_or(id);
        let SubstContext {
            cache: outer_cache, ..
        } = contexts.swap_remove(0);
        *cache = outer_cache;
        result
    }

    /// Dispatch one [`SubstStep`], possibly pushing more steps (and, for
    /// binder nodes, more [`SubstContext`]s) onto `work`/`contexts`.
    fn run_substitute_step(
        &mut self,
        step: SubstStep,
        contexts: &mut Vec<SubstContext>,
        work: &mut Vec<SubstStep>,
    ) {
        match step {
            SubstStep::Expand { id, ctx } => {
                if contexts[ctx].resolved(id).is_some() {
                    return;
                }

                let (kind, sort) = match self.get(id) {
                    Some(term) => (term.kind.clone(), term.sort),
                    None => {
                        // Matches the recursive version's `None => return id`.
                        contexts[ctx].cache.insert(id, id);
                        return;
                    }
                };

                match kind {
                    // ===== Leaves: nothing to substitute into =====
                    TermKind::True
                    | TermKind::False
                    | TermKind::IntConst(_)
                    | TermKind::RealConst(_)
                    | TermKind::BitVecConst { .. }
                    | TermKind::Var(_)
                    | TermKind::StringLit(_)
                    | TermKind::FpLit { .. }
                    | TermKind::FpPlusInfinity { .. }
                    | TermKind::FpMinusInfinity { .. }
                    | TermKind::FpPlusZero { .. }
                    | TermKind::FpMinusZero { .. }
                    | TermKind::FpNaN { .. } => {
                        contexts[ctx].cache.insert(id, id);
                    }

                    // ===== Binders: capture-avoiding, handled specially =====
                    TermKind::Forall {
                        vars,
                        body,
                        patterns,
                    } => {
                        self.expand_quantifier(
                            id, ctx, sort, false, vars, body, patterns, contexts, work,
                        );
                    }
                    TermKind::Exists {
                        vars,
                        body,
                        patterns,
                    } => {
                        self.expand_quantifier(
                            id, ctx, sort, true, vars, body, patterns, contexts, work,
                        );
                    }
                    TermKind::Let { bindings, body } => {
                        self.expand_let(id, ctx, sort, bindings, body, contexts, work);
                    }
                    TermKind::Match { scrutinee, cases } => {
                        self.expand_match(id, ctx, sort, scrutinee, cases, contexts, work);
                    }

                    // ===== Everything else: generic children via get_children =====
                    other => {
                        let children = get_children(&other);
                        work.push(SubstStep::Combine {
                            id,
                            ctx,
                            kind: other,
                            sort,
                        });
                        for &child in children.iter().rev() {
                            if contexts[ctx].resolved(child).is_none() {
                                work.push(SubstStep::Expand { id: child, ctx });
                            }
                        }
                    }
                }
            }

            SubstStep::Combine {
                id,
                ctx,
                kind,
                sort,
            } => {
                let result = self.rebuild_substituted(&contexts[ctx], id, kind, sort);
                contexts[ctx].cache.insert(id, result);
            }

            SubstStep::CombineQuantifier {
                id,
                ctx,
                sort,
                is_exists,
                new_vars,
                body,
                patterns,
                body_ctx,
            } => {
                let new_body = contexts[body_ctx].resolved(body).unwrap_or(body);
                let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                    .iter()
                    .map(|pattern| {
                        pattern
                            .iter()
                            .map(|&t| contexts[body_ctx].resolved(t).unwrap_or(t))
                            .collect()
                    })
                    .collect();
                let result = if is_exists {
                    self.intern(
                        TermKind::Exists {
                            vars: new_vars,
                            body: new_body,
                            patterns: new_patterns,
                        },
                        sort,
                    )
                } else {
                    self.intern(
                        TermKind::Forall {
                            vars: new_vars,
                            body: new_body,
                            patterns: new_patterns,
                        },
                        sort,
                    )
                };
                contexts[ctx].cache.insert(id, result);
            }

            SubstStep::CombineLet {
                id,
                ctx,
                sort,
                bindings,
                final_names,
                body,
                body_ctx,
            } => {
                let new_values: SmallVec<[TermId; 2]> = bindings
                    .iter()
                    .map(|&(_, value)| contexts[ctx].resolved(value).unwrap_or(value))
                    .collect();
                let new_body = match body_ctx {
                    None => body,
                    Some(bctx) => contexts[bctx].resolved(body).unwrap_or(body),
                };
                let new_bindings: SmallVec<[(Spur, TermId); 2]> = final_names
                    .iter()
                    .zip(new_values.iter())
                    .map(|(&(name, _), &value)| (name, value))
                    .collect();
                let result = self.intern(
                    TermKind::Let {
                        bindings: new_bindings,
                        body: new_body,
                    },
                    sort,
                );
                contexts[ctx].cache.insert(id, result);
            }

            SubstStep::CombineMatch {
                id,
                ctx,
                sort,
                scrutinee,
                case_plans,
            } => {
                let new_scrutinee = contexts[ctx].resolved(scrutinee).unwrap_or(scrutinee);
                let new_cases: SmallVec<[MatchCase; 4]> = case_plans
                    .into_iter()
                    .map(|plan| match plan {
                        MatchCasePlan::Unchanged(case) => case,
                        MatchCasePlan::Rewrite {
                            constructor,
                            new_bound,
                            body,
                            ctx: bctx,
                        } => {
                            let new_body = contexts[bctx].resolved(body).unwrap_or(body);
                            let bindings: SmallVec<[Spur; 4]> =
                                new_bound.iter().map(|&(name, _)| name).collect();
                            MatchCase {
                                constructor,
                                bindings,
                                body: new_body,
                            }
                        }
                    })
                    .collect();
                let result = self.intern(
                    TermKind::Match {
                        scrutinee: new_scrutinee,
                        cases: new_cases,
                    },
                    sort,
                );
                contexts[ctx].cache.insert(id, result);
            }
        }
    }

    /// Expand step for a `Forall`/`Exists` node (`is_exists` selects which):
    /// computes the capture-avoiding binder substitution once (mirrors the
    /// recursive `subst_quantifier_parts`), then either resolves
    /// immediately to `id` unchanged -- nothing in `subst` survives
    /// shadowing, matching the recursive version's `None => id`
    /// short-circuit, *without visiting the body/patterns at all* -- or
    /// opens a fresh context for the body and patterns and schedules them
    /// alongside a `CombineQuantifier`.
    #[allow(clippy::too_many_arguments)]
    fn expand_quantifier(
        &mut self,
        id: TermId,
        ctx: usize,
        sort: SortId,
        is_exists: bool,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        contexts: &mut Vec<SubstContext>,
        work: &mut Vec<SubstStep>,
    ) {
        // Trigger patterns are siblings of `body` in this same scope, so
        // their free variables must be visible to the fresh-name choice --
        // otherwise a rename can pick a name that a trigger still uses,
        // capturing it (see `prepare_binder_subst`).
        let pattern_terms: SmallVec<[TermId; 4]> =
            patterns.iter().flat_map(|p| p.iter().copied()).collect();
        let prep = self.prepare_binder_subst(&vars, body, &pattern_terms, &contexts[ctx].subst);
        let Some((effective, new_vars)) = prep else {
            contexts[ctx].cache.insert(id, id);
            return;
        };

        let body_ctx = contexts.len();
        contexts.push(SubstContext {
            subst: effective,
            cache: FxHashMap::default(),
        });

        // `work` is a LIFO stack: the combine step must be pushed *before*
        // its children so that it ends up underneath them and is only
        // popped (and run) once every child above it has been resolved.
        // Pushing children first would let the combine step run first
        // instead, reading not-yet-resolved children back as unchanged --
        // exactly the silent-no-op bug this whole conversion exists to
        // avoid.
        work.push(SubstStep::CombineQuantifier {
            id,
            ctx,
            sort,
            is_exists,
            new_vars,
            body,
            patterns: patterns.clone(),
            body_ctx,
        });

        if contexts[body_ctx].resolved(body).is_none() {
            work.push(SubstStep::Expand {
                id: body,
                ctx: body_ctx,
            });
        }
        for pattern in &patterns {
            for &t in pattern {
                if contexts[body_ctx].resolved(t).is_none() {
                    work.push(SubstStep::Expand {
                        id: t,
                        ctx: body_ctx,
                    });
                }
            }
        }
    }

    /// Expand step for a `Let` node (mirrors the recursive `subst_let`).
    /// Binding values are always resolved in the outer `ctx`; the bound
    /// names' sorts come from each binding's *original* (pre-substitution)
    /// value -- substitution is sort-preserving, so this is equivalent to
    /// using the substituted value's sort, but doesn't need to wait for it
    /// to be computed, letting the body's binder-substitution prep run
    /// unconditionally rather than depend on the binding-value results.
    #[allow(clippy::too_many_arguments)]
    fn expand_let(
        &mut self,
        id: TermId,
        ctx: usize,
        sort: SortId,
        bindings: SmallVec<[(Spur, TermId); 2]>,
        body: TermId,
        contexts: &mut Vec<SubstContext>,
        work: &mut Vec<SubstStep>,
    ) {
        let bound: SmallVec<[(Spur, SortId); 2]> = bindings
            .iter()
            .map(|&(name, value)| {
                (
                    name,
                    self.get(value).map_or(self.sorts.bool_sort, |t| t.sort),
                )
            })
            .collect();

        // A `Let` scope has no sibling term positions: the bound values are
        // resolved in the *outer* scope, not this one.
        let prep = self.prepare_binder_subst(&bound, body, &[], &contexts[ctx].subst);
        let (final_names, body_ctx): (SmallVec<[(Spur, SortId); 2]>, Option<usize>) = match prep {
            None => (bound, None),
            Some((effective, new_bound)) => {
                let bctx = contexts.len();
                contexts.push(SubstContext {
                    subst: effective,
                    cache: FxHashMap::default(),
                });
                (new_bound, Some(bctx))
            }
        };

        // Combine pushed before its children -- see the comment in
        // `expand_quantifier`.
        work.push(SubstStep::CombineLet {
            id,
            ctx,
            sort,
            bindings: bindings.clone(),
            final_names,
            body,
            body_ctx,
        });

        for &(_, value) in &bindings {
            if contexts[ctx].resolved(value).is_none() {
                work.push(SubstStep::Expand { id: value, ctx });
            }
        }
        if let Some(bctx) = body_ctx
            && contexts[bctx].resolved(body).is_none()
        {
            work.push(SubstStep::Expand {
                id: body,
                ctx: bctx,
            });
        }
    }

    /// Expand step for a `Match` node (mirrors the recursive
    /// `subst_match`). The scrutinee is resolved in the outer `ctx`; each
    /// case is planned independently, since different cases can bind
    /// different names and so can differ on whether capture-avoidance opens
    /// a fresh context for that case's body.
    #[allow(clippy::too_many_arguments)]
    fn expand_match(
        &mut self,
        id: TermId,
        ctx: usize,
        sort: SortId,
        scrutinee: TermId,
        cases: SmallVec<[MatchCase; 4]>,
        contexts: &mut Vec<SubstContext>,
        work: &mut Vec<SubstStep>,
    ) {
        let mut case_plans: SmallVec<[MatchCasePlan; 4]> = SmallVec::new();
        // Children to expand are collected first and pushed only after the
        // combine step below (see the comment in `expand_quantifier`), but
        // opening each case's context has to happen here regardless, since
        // `case_plans` (needed by the combine step) records which context
        // index each rewritten case's body resolves in.
        let mut pending: SmallVec<[(TermId, usize); 4]> = SmallVec::new();
        for case in cases {
            let mut bound: SmallVec<[(Spur, SortId); 2]> = SmallVec::new();
            for &name in &case.bindings {
                let var_sort = self
                    .find_var_sort(case.body, name)
                    .unwrap_or(self.sorts.bool_sort);
                bound.push((name, var_sort));
            }

            // A `Match` case's only term position in its own scope is its
            // body (the scrutinee is resolved in the outer scope), so there
            // are no sibling terms.
            match self.prepare_binder_subst(&bound, case.body, &[], &contexts[ctx].subst) {
                None => case_plans.push(MatchCasePlan::Unchanged(case)),
                Some((effective, new_bound)) => {
                    let bctx = contexts.len();
                    contexts.push(SubstContext {
                        subst: effective,
                        cache: FxHashMap::default(),
                    });
                    pending.push((case.body, bctx));
                    case_plans.push(MatchCasePlan::Rewrite {
                        constructor: case.constructor,
                        new_bound,
                        body: case.body,
                        ctx: bctx,
                    });
                }
            }
        }

        // Combine pushed before its children -- see the comment in
        // `expand_quantifier`.
        work.push(SubstStep::CombineMatch {
            id,
            ctx,
            sort,
            scrutinee,
            case_plans,
        });

        if contexts[ctx].resolved(scrutinee).is_none() {
            work.push(SubstStep::Expand { id: scrutinee, ctx });
        }
        for (case_body, bctx) in pending {
            if contexts[bctx].resolved(case_body).is_none() {
                work.push(SubstStep::Expand {
                    id: case_body,
                    ctx: bctx,
                });
            }
        }
    }

    /// Rebuild a non-binder node from its already-resolved children (see
    /// [`SubstStep::Combine`]). Mirrors the recursive `substitute_cached`'s
    /// match one-for-one: every non-leaf, non-binder `TermKind` variant
    /// gets its own arm calling the same `mk_*`/`intern` constructor the
    /// recursive version used, with each recursive call replaced by a
    /// `ctx`-scoped lookup (`sub`) of the already-computed child value.
    /// Leaves and binders are resolved directly during `Expand` and are
    /// never scheduled as a `Combine`, so those arms are unreachable here;
    /// they are still listed explicitly (no catch-all) so that a newly
    /// added `TermKind` variant fails to compile, exactly like the
    /// recursive version's original exhaustiveness guarantee.
    fn rebuild_substituted(
        &mut self,
        ctx: &SubstContext,
        id: TermId,
        kind: TermKind,
        sort: SortId,
    ) -> TermId {
        let sub = |t: TermId| ctx.resolved(t).unwrap_or(t);
        match kind {
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::Var(_)
            | TermKind::StringLit(_)
            | TermKind::FpLit { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpNaN { .. } => {
                unreachable!("leaves are resolved directly in Expand, never scheduled as Combine")
            }

            // ===== Boolean connectives =====
            TermKind::Not(a) => {
                let a = sub(a);
                self.mk_not(a)
            }
            TermKind::And(args) => {
                let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
                self.mk_and(new)
            }
            TermKind::Or(args) => {
                let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
                self.mk_or(new)
            }
            TermKind::Xor(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_xor(a, b)
            }
            TermKind::Implies(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_implies(a, b)
            }
            TermKind::Ite(c, t, e) => {
                let c = sub(c);
                let t = sub(t);
                let e = sub(e);
                self.mk_ite(c, t, e)
            }

            // ===== Equality / distinct =====
            TermKind::Eq(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_eq(a, b)
            }
            TermKind::Distinct(args) => {
                let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
                self.mk_distinct(new)
            }

            // ===== Arithmetic =====
            TermKind::Neg(a) => {
                let a = sub(a);
                self.mk_neg(a)
            }
            TermKind::Add(args) => {
                let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
                self.mk_add(new)
            }
            TermKind::Sub(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_sub(a, b)
            }
            TermKind::Mul(args) => {
                let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
                self.mk_mul(new)
            }
            TermKind::Div(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_div(a, b)
            }
            TermKind::Mod(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_mod(a, b)
            }
            TermKind::Lt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_lt(a, b)
            }
            TermKind::Le(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_le(a, b)
            }
            TermKind::Gt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_gt(a, b)
            }
            TermKind::Ge(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_ge(a, b)
            }

            // ===== Arrays =====
            TermKind::Select(arr, idx) => {
                let arr = sub(arr);
                let idx = sub(idx);
                self.mk_select(arr, idx)
            }
            TermKind::Store(arr, idx, val) => {
                let arr = sub(arr);
                let idx = sub(idx);
                let val = sub(val);
                self.mk_store(arr, idx, val)
            }

            // ===== Bit vectors =====
            TermKind::BvConcat(a, b) => {
                let a = sub(a);
                let b = sub(b);
                // A sort-preserving substitution can never reach the error
                // branch: the operands were BV-sorted before substitution,
                // and a substitution that preserves sorts leaves them
                // BV-sorted.
                //
                // The `id` fallback covers only that unreachable case, by
                // declining to apply an ill-typed map and keeping the
                // original term. That may leave a substitution unapplied,
                // which is a weaker result but a sound one — unlike the
                // fabricated 32-bit-wide term this used to build, which is
                // ill-sorted and can flip an answer between sat and unsat.
                self.try_mk_bv_concat(a, b).unwrap_or(id)
            }
            TermKind::BvExtract { high, low, arg } => {
                let arg = sub(arg);
                self.mk_bv_extract(high, low, arg)
            }
            TermKind::BvNot(a) => {
                let a = sub(a);
                self.mk_bv_not(a)
            }
            TermKind::BvAnd(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_and(a, b)
            }
            TermKind::BvOr(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_or(a, b)
            }
            TermKind::BvXor(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_xor(a, b)
            }
            TermKind::BvAdd(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_add(a, b)
            }
            TermKind::BvSub(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_sub(a, b)
            }
            TermKind::BvMul(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_mul(a, b)
            }
            TermKind::BvUdiv(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_udiv(a, b)
            }
            TermKind::BvSdiv(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_sdiv(a, b)
            }
            TermKind::BvUrem(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_urem(a, b)
            }
            TermKind::BvSrem(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_srem(a, b)
            }
            TermKind::BvShl(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_shl(a, b)
            }
            TermKind::BvLshr(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_lshr(a, b)
            }
            TermKind::BvAshr(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_ashr(a, b)
            }
            TermKind::BvUlt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_ult(a, b)
            }
            TermKind::BvUle(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_ule(a, b)
            }
            TermKind::BvSlt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_slt(a, b)
            }
            TermKind::BvSle(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_bv_sle(a, b)
            }

            // ===== Strings =====
            TermKind::StrConcat(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_concat(a, b)
            }
            TermKind::StrLen(a) => {
                let a = sub(a);
                self.mk_str_len(a)
            }
            TermKind::StrSubstr(s, i, n) => {
                let s = sub(s);
                let i = sub(i);
                let n = sub(n);
                self.mk_str_substr(s, i, n)
            }
            TermKind::StrAt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_at(a, b)
            }
            TermKind::StrContains(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_contains(a, b)
            }
            TermKind::StrPrefixOf(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_prefixof(a, b)
            }
            TermKind::StrSuffixOf(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_suffixof(a, b)
            }
            TermKind::StrIndexOf(s, t, o) => {
                let s = sub(s);
                let t = sub(t);
                let o = sub(o);
                self.mk_str_indexof(s, t, o)
            }
            TermKind::StrReplace(s, p, r) => {
                let s = sub(s);
                let p = sub(p);
                let r = sub(r);
                self.mk_str_replace(s, p, r)
            }
            TermKind::StrReplaceAll(s, p, r) => {
                let s = sub(s);
                let p = sub(p);
                let r = sub(r);
                self.mk_str_replace_all(s, p, r)
            }
            TermKind::StrToInt(a) => {
                let a = sub(a);
                self.mk_str_to_int(a)
            }
            TermKind::IntToStr(a) => {
                let a = sub(a);
                self.mk_int_to_str(a)
            }
            TermKind::StrInRe(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_in_re(a, b)
            }
            TermKind::StrReplaceRe(s, re, r) => {
                let s = sub(s);
                let re = sub(re);
                let r = sub(r);
                self.mk_str_replace_re(s, re, r)
            }
            TermKind::StrReplaceReAll(s, re, r) => {
                let s = sub(s);
                let re = sub(re);
                let r = sub(r);
                self.mk_str_replace_re_all(s, re, r)
            }
            TermKind::StrLt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_lt(a, b)
            }
            TermKind::StrLe(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_str_le(a, b)
            }
            TermKind::StrToCode(a) => {
                let a = sub(a);
                self.mk_str_to_code(a)
            }
            TermKind::StrFromCode(a) => {
                let a = sub(a);
                self.mk_str_from_code(a)
            }

            // ===== Floating point =====
            TermKind::FpAbs(a) => {
                let a = sub(a);
                self.mk_fp_abs(a)
            }
            TermKind::FpNeg(a) => {
                let a = sub(a);
                self.mk_fp_neg(a)
            }
            TermKind::FpSqrt(rm, a) => {
                let a = sub(a);
                self.mk_fp_sqrt(rm, a)
            }
            TermKind::FpRoundToIntegral(rm, a) => {
                let a = sub(a);
                self.mk_fp_round_to_integral(rm, a)
            }
            TermKind::FpAdd(rm, a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_add(rm, a, b)
            }
            TermKind::FpSub(rm, a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_sub(rm, a, b)
            }
            TermKind::FpMul(rm, a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_mul(rm, a, b)
            }
            TermKind::FpDiv(rm, a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_div(rm, a, b)
            }
            TermKind::FpRem(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_rem(a, b)
            }
            TermKind::FpMin(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_min(a, b)
            }
            TermKind::FpMax(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_max(a, b)
            }
            TermKind::FpLeq(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_leq(a, b)
            }
            TermKind::FpLt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_lt(a, b)
            }
            TermKind::FpGeq(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_geq(a, b)
            }
            TermKind::FpGt(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_gt(a, b)
            }
            TermKind::FpEq(a, b) => {
                let a = sub(a);
                let b = sub(b);
                self.mk_fp_eq(a, b)
            }
            TermKind::FpFma(rm, a, b, c) => {
                let a = sub(a);
                let b = sub(b);
                let c = sub(c);
                self.mk_fp_fma(rm, a, b, c)
            }
            TermKind::FpIsNormal(a) => {
                let a = sub(a);
                self.mk_fp_is_normal(a)
            }
            TermKind::FpIsSubnormal(a) => {
                let a = sub(a);
                self.mk_fp_is_subnormal(a)
            }
            TermKind::FpIsZero(a) => {
                let a = sub(a);
                self.mk_fp_is_zero(a)
            }
            TermKind::FpIsInfinite(a) => {
                let a = sub(a);
                self.mk_fp_is_infinite(a)
            }
            TermKind::FpIsNaN(a) => {
                let a = sub(a);
                self.mk_fp_is_nan(a)
            }
            TermKind::FpIsNegative(a) => {
                let a = sub(a);
                self.mk_fp_is_negative(a)
            }
            TermKind::FpIsPositive(a) => {
                let a = sub(a);
                self.mk_fp_is_positive(a)
            }
            TermKind::FpToReal(a) => {
                let a = sub(a);
                self.mk_fp_to_real(a)
            }
            TermKind::FpToFp { rm, arg, eb, sb } => {
                let arg = sub(arg);
                self.mk_fp_to_fp(rm, arg, eb, sb)
            }
            TermKind::FpToSBV { rm, arg, width } => {
                let arg = sub(arg);
                self.mk_fp_to_sbv(rm, arg, width)
            }
            TermKind::FpToUBV { rm, arg, width } => {
                let arg = sub(arg);
                self.mk_fp_to_ubv(rm, arg, width)
            }
            TermKind::RealToFp { rm, arg, eb, sb } => {
                let arg = sub(arg);
                self.mk_real_to_fp(rm, arg, eb, sb)
            }
            TermKind::SBVToFp { rm, arg, eb, sb } => {
                let arg = sub(arg);
                self.mk_sbv_to_fp(rm, arg, eb, sb)
            }
            TermKind::UBVToFp { rm, arg, eb, sb } => {
                let arg = sub(arg);
                self.mk_ubv_to_fp(rm, arg, eb, sb)
            }

            // ===== Uninterpreted function application =====
            TermKind::Apply { func, args } => {
                let args: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
                self.intern(TermKind::Apply { func, args }, sort)
            }

            // ===== Algebraic datatypes =====
            TermKind::DtConstructor { constructor, args } => {
                let args: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
                self.intern(TermKind::DtConstructor { constructor, args }, sort)
            }
            TermKind::DtTester { constructor, arg } => {
                let arg = sub(arg);
                self.intern(TermKind::DtTester { constructor, arg }, sort)
            }
            TermKind::DtSelector { selector, arg } => {
                let arg = sub(arg);
                self.intern(TermKind::DtSelector { selector, arg }, sort)
            }

            TermKind::Forall { .. }
            | TermKind::Exists { .. }
            | TermKind::Let { .. }
            | TermKind::Match { .. } => {
                unreachable!(
                    "binders are handled by dedicated expand_*/Combine* steps, never scheduled as a plain Combine"
                )
            }
        }
    }

    /// Build the effective substitution to apply inside a binder scope.
    ///
    /// Drops entries whose source is one of `bound` (the bound variable is
    /// shadowed) and, when a bound variable's name would capture a free
    /// variable of some replacement term, alpha-renames that bound variable to
    /// a fresh name (extending the returned substitution with the renaming).
    ///
    /// `sibling_terms` lists further term positions that belong to this same
    /// binder scope but are not reachable from `body` -- in practice a
    /// `Forall`/`Exists` node's trigger patterns, which are siblings of the
    /// body rather than subterms of it. Their free variables must be part of
    /// the set a freshly generated binder name avoids: a trigger occurrence is
    /// a live occurrence, so handing out its name as "fresh" captures it.
    /// `Let` and `Match` scopes pass an empty slice.
    ///
    /// Every free-variable query here is deliberately the *pattern-aware*
    /// [`collect_free_vars_including_patterns`]: this function is the
    /// capture-avoidance name-clash detector, and a name it fails to see is a
    /// name it will happily reuse. Over-reporting only costs a different fresh
    /// name; under-reporting captures.
    ///
    /// Returns `None` when the resulting substitution is empty (nothing to do,
    /// no capture) so the caller can preserve the original term.
    pub(super) fn prepare_binder_subst(
        &mut self,
        bound: &[(Spur, SortId)],
        body: TermId,
        sibling_terms: &[TermId],
        subst: &FxHashMap<TermId, TermId>,
    ) -> Option<BinderSubstPrep> {
        // Effective substitution: drop entries whose source is a bound variable.
        let mut effective: FxHashMap<TermId, TermId> = FxHashMap::default();
        for (&from, &to) in subst {
            let shadowed = self.get(from).is_some_and(|t| match &t.kind {
                TermKind::Var(name) => bound
                    .iter()
                    .any(|(bound_name, bound_sort)| bound_name == name && *bound_sort == t.sort),
                _ => false,
            });
            if !shadowed {
                effective.insert(from, to);
            }
        }
        if effective.is_empty() {
            return None;
        }

        // Names occurring free in the replacement range.
        let mut range_free: FxHashSet<Spur> = FxHashSet::default();
        for &to in effective.values() {
            for var in collect_free_vars_including_patterns(to, self) {
                if let Some(TermKind::Var(name)) = self.get(var).map(|t| &t.kind) {
                    range_free.insert(*name);
                }
            }
        }

        // Bound variables whose name would capture a replacement's free variable.
        let capturing: SmallVec<[usize; 2]> = bound
            .iter()
            .enumerate()
            .filter(|(_, (name, _))| range_free.contains(name))
            .map(|(index, _)| index)
            .collect();

        if capturing.is_empty() {
            return Some((effective, bound.iter().copied().collect()));
        }

        // Names that a freshly generated binder name must avoid: the
        // replacement range's, this binder's own, and everything occurring
        // free anywhere else in the scope being rebuilt -- the body plus any
        // sibling term positions (trigger patterns).
        let mut avoid = range_free;
        for (name, _) in bound {
            avoid.insert(*name);
        }
        for &scope_term in core::iter::once(&body).chain(sibling_terms) {
            for var in collect_free_vars_including_patterns(scope_term, self) {
                if let Some(TermKind::Var(name)) = self.get(var).map(|t| &t.kind) {
                    avoid.insert(*name);
                }
            }
        }

        let mut new_bound: SmallVec<[(Spur, SortId); 2]> = bound.iter().copied().collect();
        for index in capturing {
            let (name, var_sort) = bound[index];
            let (fresh_name, fresh_var) = self.fresh_var(name, var_sort, &avoid);
            avoid.insert(fresh_name);
            let old_var = self.intern(TermKind::Var(name), var_sort);
            effective.insert(old_var, fresh_var);
            new_bound[index] = (fresh_name, var_sort);
        }
        Some((effective, new_bound))
    }

    /// Create a fresh variable derived from `base` whose name is not in `avoid`.
    fn fresh_var(&mut self, base: Spur, sort: SortId, avoid: &FxHashSet<Spur>) -> (Spur, TermId) {
        let base_name = self.resolve_str(base).to_string();
        let mut counter: u64 = 0;
        loop {
            let candidate = format!("{base_name}!{counter}");
            let name = self.intern_str(&candidate);
            if !avoid.contains(&name) {
                let var = self.intern(TermKind::Var(name), sort);
                return (name, var);
            }
            counter += 1;
        }
    }

    /// Find the sort of a variable named `target` by locating an occurrence in
    /// `term` (used to reconstruct fresh binders for `match` cases, whose
    /// bindings do not carry sort information directly).
    pub(super) fn find_var_sort(&self, term: TermId, target: Spur) -> Option<SortId> {
        for sub in collect_subterms(term, self) {
            if let Some(t) = self.get(sub)
                && let TermKind::Var(name) = t.kind
                && name == target
            {
                return Some(t.sort);
            }
        }
        None
    }
}
