//! Iterative engine behind [`super::Substitution::apply`].
//!
//! # What this replaces
//!
//! `Substitution::apply_recursive` was a native `Result`-returning recursion
//! that spent one call frame per level of the term it walked. Its production
//! caller is [`crate::ematching::quantifier_inst::EmatchEngine::match_round`]
//! (reached from `Solver::check` via the E-matching round), which applies the
//! substitution to a *quantifier body* -- a term whose depth comes straight
//! from the input formula and is therefore unbounded. A deep enough body
//! overflowed the native stack and aborted the process.
//!
//! It is converted here to an explicit heap work list, so depth is bounded by
//! available memory rather than by the fixed native stack.
//!
//! # Why this does not delegate to [`crate::ast::TermManager::substitute`]
//!
//! `TermManager::substitute` is the workspace's canonical capture-avoiding
//! iterative substitution, and delegating to it was evaluated first. It does
//! not fit this caller:
//!
//! * **Key space.** `substitute` maps `TermId -> TermId`; a [`Substitution`]
//!   maps a variable *name* (`Spur`) to a term, and replaces every `Var` with
//!   that name whatever its sort. One name can denote several distinct `Var`
//!   `TermId`s (one per sort it was interned at), so the two key spaces are
//!   not in bijection: building the `TermId` map would mean scanning the whole
//!   manager for `Var` nodes on every application.
//! * **Shadowing granularity.** A binder shadows a name here (`vars.iter().
//!   any(|(v, _)| v == name)`), whereas `prepare_binder_subst` shadows on the
//!   `(name, sort)` pair. Delegating would let a same-name/different-sort
//!   binder stop shadowing, changing which occurrences get replaced.
//! * **Alpha-renaming.** `substitute` renames binders to avoid capture and so
//!   can return a body with freshly generated variable names. E-matching maps
//!   quantified variables to *ground* terms drawn from the term pool, which
//!   have no free variables to capture, so the rename never buys anything here
//!   and would only make instantiation lemmas differ syntactically from the
//!   terms the rest of the engine (fingerprints, the dedup cache) keys on.
//! * **Error channel.** `apply` reports a dangling `TermId` as
//!   [`OxizError::EmatchError`]; `substitute` has no error channel and returns
//!   such a term unchanged.
//!
//! So the walk is converted in place, reusing `substitute`'s structure: a
//! `Vec` work list of [`Step`]s, a scope arena ([`ApplyScope`]) so that each
//! binder body is resolved under its own effective substitution *and its own
//! memo cache*, and a `Combine` step pushed before the children it consumes.
//!
//! # Two fixes that come with the conversion
//!
//! * **Memoization.** The recursion had none, so a shared subterm was
//!   re-expanded once per path reaching it -- exponential on a DAG-shaped
//!   body, which is the normal shape of a term produced by a hash-consing
//!   manager. Each scope now memoizes by `TermId`.
//! * **No silent catch-all.** The recursion ended in `_ => Ok(term)`, which
//!   silently returned bit-vector, string, floating-point, datatype, `Let`,
//!   `Match`, `Xor`, `Mod` and `Distinct` nodes *unsubstituted*: an
//!   instantiation lemma with the quantified variable still free in it, i.e.
//!   a wrong formula handed to the solver. [`rebuild`] is exhaustive over
//!   `TermKind` with no catch-all arm, so a newly added variant fails to
//!   compile rather than being dropped.

use crate::ast::traversal::get_children;
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::{OxizError, Result};
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use smallvec::SmallVec;

/// One substitution scope opened while walking a term: the name bindings in
/// force inside it, plus the memo cache for terms resolved under them.
///
/// A binder opens a fresh scope whose cache starts empty and is never shared
/// with its parent: whether a given `TermId` has already been substituted
/// depends on *which* bindings are in force, not on its identity alone.
struct ApplyScope {
    bindings: FxHashMap<Spur, TermId>,
    cache: FxHashMap<TermId, TermId>,
}

impl ApplyScope {
    /// `id`'s substituted value under this scope, or `id` itself when it has
    /// not been resolved (only ever queried after the resolving step has run,
    /// so the fallback is the structurally unreachable case).
    fn resolved(&self, id: TermId) -> TermId {
        self.cache.get(&id).copied().unwrap_or(id)
    }
}

/// One pending step of the iterative walk. `scope`/`body_scope` index the
/// scope arena threaded through [`apply_bindings`].
enum Step {
    /// Resolve `id` under `scopes[scope]`, scheduling its children and the
    /// matching combine step when it is not already resolved.
    Expand { id: TermId, scope: usize },
    /// Rebuild a non-binder node from its resolved children.
    Combine {
        id: TermId,
        scope: usize,
        kind: TermKind,
        sort: SortId,
    },
    /// Rebuild a `Forall` (`is_exists == false`) or `Exists` node.
    CombineQuantifier {
        id: TermId,
        scope: usize,
        sort: SortId,
        is_exists: bool,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        body_scope: usize,
    },
    /// Rebuild a `Let` node: bound values resolve in the outer scope, the
    /// body in `body_scope`.
    CombineLet {
        id: TermId,
        scope: usize,
        sort: SortId,
        bindings: SmallVec<[(Spur, TermId); 2]>,
        body: TermId,
        body_scope: usize,
    },
    /// Rebuild a `Match` node: the scrutinee resolves in the outer scope,
    /// each case body in its own scope. `kind` is the original
    /// `TermKind::Match` (carried whole so the case list can be cloned and
    /// patched without naming the private `MatchCase` type), and
    /// `body_scopes` runs parallel to its cases.
    CombineMatch {
        id: TermId,
        scope: usize,
        sort: SortId,
        kind: TermKind,
        body_scopes: SmallVec<[usize; 4]>,
    },
}

/// Apply `bindings` (variable name -> replacement term) to `term`.
///
/// Returns the original `TermId` for any node none of whose children changed,
/// preserving the structural sharing the recursive version provided.
pub(super) fn apply_bindings(
    bindings: &FxHashMap<Spur, TermId>,
    term: TermId,
    manager: &mut TermManager,
) -> Result<TermId> {
    let mut scopes: Vec<ApplyScope> = vec![ApplyScope {
        bindings: bindings.clone(),
        cache: FxHashMap::default(),
    }];
    let mut work: Vec<Step> = vec![Step::Expand { id: term, scope: 0 }];

    while let Some(step) = work.pop() {
        run_step(step, manager, &mut scopes, &mut work)?;
    }

    // `work` is empty, so the root's `Expand` (and any combine it scheduled)
    // has run: the root is resolved in scope 0.
    Ok(scopes[0].resolved(term))
}

/// Dispatch one [`Step`], possibly pushing further steps and scopes.
fn run_step(
    step: Step,
    manager: &mut TermManager,
    scopes: &mut Vec<ApplyScope>,
    work: &mut Vec<Step>,
) -> Result<()> {
    match step {
        Step::Expand { id, scope } => expand(id, scope, manager, scopes, work),

        Step::Combine {
            id,
            scope,
            kind,
            sort,
        } => {
            let result = rebuild(manager, &scopes[scope], id, kind, sort);
            scopes[scope].cache.insert(id, result);
            Ok(())
        }

        Step::CombineQuantifier {
            id,
            scope,
            sort,
            is_exists,
            vars,
            body,
            patterns,
            body_scope,
        } => {
            let new_body = scopes[body_scope].resolved(body);
            let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                .iter()
                .map(|pattern| {
                    pattern
                        .iter()
                        .map(|&t| scopes[body_scope].resolved(t))
                        .collect()
                })
                .collect();
            let result = if new_body == body && new_patterns == patterns {
                id
            } else if is_exists {
                manager.intern(
                    TermKind::Exists {
                        vars,
                        body: new_body,
                        patterns: new_patterns,
                    },
                    sort,
                )
            } else {
                manager.intern(
                    TermKind::Forall {
                        vars,
                        body: new_body,
                        patterns: new_patterns,
                    },
                    sort,
                )
            };
            scopes[scope].cache.insert(id, result);
            Ok(())
        }

        Step::CombineLet {
            id,
            scope,
            sort,
            bindings,
            body,
            body_scope,
        } => {
            let new_bindings: SmallVec<[(Spur, TermId); 2]> = bindings
                .iter()
                .map(|&(name, value)| (name, scopes[scope].resolved(value)))
                .collect();
            let new_body = scopes[body_scope].resolved(body);
            let result = if new_body == body && new_bindings == bindings {
                id
            } else {
                manager.intern(
                    TermKind::Let {
                        bindings: new_bindings,
                        body: new_body,
                    },
                    sort,
                )
            };
            scopes[scope].cache.insert(id, result);
            Ok(())
        }

        Step::CombineMatch {
            id,
            scope,
            sort,
            kind,
            body_scopes,
        } => {
            let TermKind::Match {
                scrutinee,
                mut cases,
            } = kind
            else {
                // `CombineMatch` is only ever pushed from the `Match` arm of
                // `expand`, carrying that very node's kind.
                unreachable!("CombineMatch always carries a TermKind::Match");
            };
            let new_scrutinee = scopes[scope].resolved(scrutinee);
            let mut changed = new_scrutinee != scrutinee;
            for (case, &body_scope) in cases.iter_mut().zip(body_scopes.iter()) {
                let new_body = scopes[body_scope].resolved(case.body);
                changed |= new_body != case.body;
                case.body = new_body;
            }
            let result = if changed {
                manager.intern(
                    TermKind::Match {
                        scrutinee: new_scrutinee,
                        cases,
                    },
                    sort,
                )
            } else {
                id
            };
            scopes[scope].cache.insert(id, result);
            Ok(())
        }
    }
}

/// Expand step: resolve a leaf directly, or schedule a node's children plus
/// the combine step that consumes them.
fn expand(
    id: TermId,
    scope: usize,
    manager: &mut TermManager,
    scopes: &mut Vec<ApplyScope>,
    work: &mut Vec<Step>,
) -> Result<()> {
    if scopes[scope].cache.contains_key(&id) {
        return Ok(());
    }

    let (kind, sort) = match manager.get(id) {
        Some(term) => (term.kind.clone(), term.sort),
        None => {
            return Err(OxizError::EmatchError(format!(
                "Term {id:?} not found in manager"
            )));
        }
    };

    match kind {
        // A bound variable resolves to its replacement; anything else stays.
        TermKind::Var(name) => {
            let result = scopes[scope].bindings.get(&name).copied().unwrap_or(id);
            scopes[scope].cache.insert(id, result);
        }

        // Leaves: nothing to substitute into.
        TermKind::True
        | TermKind::False
        | TermKind::IntConst(_)
        | TermKind::RealConst(_)
        | TermKind::BitVecConst { .. }
        | TermKind::StringLit(_)
        | TermKind::FpLit { .. }
        | TermKind::FpPlusInfinity { .. }
        | TermKind::FpMinusInfinity { .. }
        | TermKind::FpPlusZero { .. }
        | TermKind::FpMinusZero { .. }
        | TermKind::FpNaN { .. } => {
            scopes[scope].cache.insert(id, id);
        }

        TermKind::Forall {
            vars,
            body,
            patterns,
        } => expand_quantifier(id, scope, sort, false, vars, body, patterns, scopes, work),
        TermKind::Exists {
            vars,
            body,
            patterns,
        } => expand_quantifier(id, scope, sort, true, vars, body, patterns, scopes, work),
        TermKind::Let { bindings, body } => {
            expand_let(id, scope, sort, bindings, body, scopes, work);
        }
        // Inlined rather than split into an `expand_match` helper: naming the
        // case type in a signature is impossible from here (`ast::term` is a
        // private module), so the case list stays behind type inference.
        TermKind::Match { scrutinee, cases } => {
            let mut body_scopes: SmallVec<[usize; 4]> = SmallVec::new();
            for case in &cases {
                let effective = unshadowed(&scopes[scope].bindings, |name| {
                    case.bindings.contains(&name)
                });
                body_scopes.push(scopes.len());
                scopes.push(ApplyScope {
                    bindings: effective,
                    cache: FxHashMap::default(),
                });
            }
            let pending: SmallVec<[(TermId, usize); 4]> = cases
                .iter()
                .map(|case| case.body)
                .zip(body_scopes.iter().copied())
                .collect();

            work.push(Step::CombineMatch {
                id,
                scope,
                sort,
                kind: TermKind::Match { scrutinee, cases },
                body_scopes,
            });

            if !scopes[scope].cache.contains_key(&scrutinee) {
                work.push(Step::Expand {
                    id: scrutinee,
                    scope,
                });
            }
            for (body, body_scope) in pending {
                if !scopes[body_scope].cache.contains_key(&body) {
                    work.push(Step::Expand {
                        id: body,
                        scope: body_scope,
                    });
                }
            }
        }

        other => {
            let children = get_children(&other);
            // `work` is LIFO: the combine step goes on first so that it sits
            // *below* the children and only runs once they are all resolved.
            work.push(Step::Combine {
                id,
                scope,
                kind: other,
                sort,
            });
            for &child in children.iter().rev() {
                if !scopes[scope].cache.contains_key(&child) {
                    work.push(Step::Expand { id: child, scope });
                }
            }
        }
    }

    Ok(())
}

/// Bindings still in force inside a binder that binds `names`: every mapping
/// whose variable name is not shadowed by one of them.
fn unshadowed(
    outer: &FxHashMap<Spur, TermId>,
    names: impl Fn(Spur) -> bool,
) -> FxHashMap<Spur, TermId> {
    outer
        .iter()
        .filter(|(name, _)| !names(**name))
        .map(|(&name, &value)| (name, value))
        .collect()
}

/// Expand a `Forall`/`Exists` node: open a scope for the body and its trigger
/// patterns (which are siblings of the body, in the same binder scope) under
/// the unshadowed bindings.
#[allow(clippy::too_many_arguments)]
fn expand_quantifier(
    id: TermId,
    scope: usize,
    sort: SortId,
    is_exists: bool,
    vars: SmallVec<[(Spur, SortId); 2]>,
    body: TermId,
    patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    scopes: &mut Vec<ApplyScope>,
    work: &mut Vec<Step>,
) {
    let effective = unshadowed(&scopes[scope].bindings, |name| {
        vars.iter().any(|(v, _)| *v == name)
    });
    let body_scope = scopes.len();
    scopes.push(ApplyScope {
        bindings: effective,
        cache: FxHashMap::default(),
    });

    work.push(Step::CombineQuantifier {
        id,
        scope,
        sort,
        is_exists,
        vars,
        body,
        patterns: patterns.clone(),
        body_scope,
    });

    if !scopes[body_scope].cache.contains_key(&body) {
        work.push(Step::Expand {
            id: body,
            scope: body_scope,
        });
    }
    for pattern in &patterns {
        for &t in pattern {
            if !scopes[body_scope].cache.contains_key(&t) {
                work.push(Step::Expand {
                    id: t,
                    scope: body_scope,
                });
            }
        }
    }
}

/// Expand a `Let` node: bound values belong to the outer scope, the body to a
/// scope where the let-bound names are shadowed away.
fn expand_let(
    id: TermId,
    scope: usize,
    sort: SortId,
    bindings: SmallVec<[(Spur, TermId); 2]>,
    body: TermId,
    scopes: &mut Vec<ApplyScope>,
    work: &mut Vec<Step>,
) {
    let effective = unshadowed(&scopes[scope].bindings, |name| {
        bindings.iter().any(|(n, _)| *n == name)
    });
    let body_scope = scopes.len();
    scopes.push(ApplyScope {
        bindings: effective,
        cache: FxHashMap::default(),
    });

    work.push(Step::CombineLet {
        id,
        scope,
        sort,
        bindings: bindings.clone(),
        body,
        body_scope,
    });

    for &(_, value) in &bindings {
        if !scopes[scope].cache.contains_key(&value) {
            work.push(Step::Expand { id: value, scope });
        }
    }
    if !scopes[body_scope].cache.contains_key(&body) {
        work.push(Step::Expand {
            id: body,
            scope: body_scope,
        });
    }
}

/// Rebuild a non-binder node from its already-resolved children.
///
/// Returns `id` unchanged when no child changed -- the structural sharing the
/// recursive version's `if changed { .. } else { Ok(term) }` arms provided,
/// applied uniformly here rather than per variant.
///
/// The match is exhaustive with no catch-all: the retired recursion's
/// `_ => Ok(term)` arm silently returned bit-vector, string, floating-point,
/// datatype, `Xor`, `Mod` and `Distinct` nodes *unsubstituted*, which for an
/// E-matching instantiation means emitting a lemma in which the quantified
/// variable is still free. Listing every variant makes a newly added one a
/// compile error instead of a fresh instance of that bug.
#[allow(clippy::too_many_lines)]
fn rebuild(
    manager: &mut TermManager,
    scope: &ApplyScope,
    id: TermId,
    kind: TermKind,
    sort: SortId,
) -> TermId {
    let children = get_children(&kind);
    if children.iter().all(|&child| scope.resolved(child) == child) {
        return id;
    }
    let sub = |t: TermId| scope.resolved(t);

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
            unreachable!("leaves are resolved directly in `expand`, never scheduled as a Combine")
        }

        // ===== Boolean connectives =====
        TermKind::Not(a) => {
            let a = sub(a);
            manager.mk_not(a)
        }
        TermKind::And(args) => {
            let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
            manager.mk_and(new)
        }
        TermKind::Or(args) => {
            let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
            manager.mk_or(new)
        }
        TermKind::Xor(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_xor(a, b)
        }
        TermKind::Implies(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_implies(a, b)
        }
        TermKind::Ite(c, t, e) => {
            let c = sub(c);
            let t = sub(t);
            let e = sub(e);
            manager.mk_ite(c, t, e)
        }

        // ===== Equality / distinct =====
        TermKind::Eq(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_eq(a, b)
        }
        TermKind::Distinct(args) => {
            let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
            manager.mk_distinct(new)
        }

        // ===== Arithmetic =====
        TermKind::Neg(a) => {
            let a = sub(a);
            manager.mk_neg(a)
        }
        TermKind::Add(args) => {
            let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
            manager.mk_add(new)
        }
        TermKind::Sub(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_sub(a, b)
        }
        TermKind::Mul(args) => {
            let new: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
            manager.mk_mul(new)
        }
        TermKind::Div(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_div(a, b)
        }
        TermKind::Mod(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_mod(a, b)
        }
        TermKind::Lt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_lt(a, b)
        }
        TermKind::Le(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_le(a, b)
        }
        TermKind::Gt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_gt(a, b)
        }
        TermKind::Ge(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_ge(a, b)
        }

        // ===== Arrays =====
        TermKind::Select(arr, idx) => {
            let arr = sub(arr);
            let idx = sub(idx);
            manager.mk_select(arr, idx)
        }
        TermKind::Store(arr, idx, val) => {
            let arr = sub(arr);
            let idx = sub(idx);
            let val = sub(val);
            manager.mk_store(arr, idx, val)
        }

        // ===== Bit vectors =====
        TermKind::BvConcat(a, b) => {
            let a = sub(a);
            let b = sub(b);
            // A sort-preserving substitution can never reach the error
            // branch: the operands were BV-sorted before substitution, and a
            // substitution that preserves sorts leaves them BV-sorted.
            //
            // The `id` fallback is a last resort for that unreachable case,
            // not a good outcome: the early return above means some child
            // *did* change, so `id` is the original, unsubstituted term —
            // the very shape this module's header calls out as leaving a
            // quantified variable free in an instantiation. It is still
            // preferable to the fabricated 32-bit-wide term this used to
            // build, because a wrong-width term is unsound in the theory
            // rather than merely too weak, and an ill-typed map is a caller
            // bug that cannot be repaired here.
            manager.try_mk_bv_concat(a, b).unwrap_or(id)
        }
        TermKind::BvExtract { high, low, arg } => {
            let arg = sub(arg);
            manager.mk_bv_extract(high, low, arg)
        }
        TermKind::BvNot(a) => {
            let a = sub(a);
            manager.mk_bv_not(a)
        }
        TermKind::BvAnd(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_and(a, b)
        }
        TermKind::BvOr(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_or(a, b)
        }
        TermKind::BvXor(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_xor(a, b)
        }
        TermKind::BvAdd(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_add(a, b)
        }
        TermKind::BvSub(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_sub(a, b)
        }
        TermKind::BvMul(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_mul(a, b)
        }
        TermKind::BvUdiv(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_udiv(a, b)
        }
        TermKind::BvSdiv(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_sdiv(a, b)
        }
        TermKind::BvUrem(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_urem(a, b)
        }
        TermKind::BvSrem(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_srem(a, b)
        }
        TermKind::BvShl(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_shl(a, b)
        }
        TermKind::BvLshr(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_lshr(a, b)
        }
        TermKind::BvAshr(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_ashr(a, b)
        }
        TermKind::BvUlt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_ult(a, b)
        }
        TermKind::BvUle(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_ule(a, b)
        }
        TermKind::BvSlt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_slt(a, b)
        }
        TermKind::BvSle(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_bv_sle(a, b)
        }

        // ===== Strings =====
        TermKind::StrConcat(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_concat(a, b)
        }
        TermKind::StrLen(a) => {
            let a = sub(a);
            manager.mk_str_len(a)
        }
        TermKind::StrSubstr(s, i, n) => {
            let s = sub(s);
            let i = sub(i);
            let n = sub(n);
            manager.mk_str_substr(s, i, n)
        }
        TermKind::StrAt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_at(a, b)
        }
        TermKind::StrContains(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_contains(a, b)
        }
        TermKind::StrPrefixOf(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_prefixof(a, b)
        }
        TermKind::StrSuffixOf(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_suffixof(a, b)
        }
        TermKind::StrIndexOf(s, t, o) => {
            let s = sub(s);
            let t = sub(t);
            let o = sub(o);
            manager.mk_str_indexof(s, t, o)
        }
        TermKind::StrReplace(s, p, r) => {
            let s = sub(s);
            let p = sub(p);
            let r = sub(r);
            manager.mk_str_replace(s, p, r)
        }
        TermKind::StrReplaceAll(s, p, r) => {
            let s = sub(s);
            let p = sub(p);
            let r = sub(r);
            manager.mk_str_replace_all(s, p, r)
        }
        TermKind::StrToInt(a) => {
            let a = sub(a);
            manager.mk_str_to_int(a)
        }
        TermKind::IntToStr(a) => {
            let a = sub(a);
            manager.mk_int_to_str(a)
        }
        TermKind::StrInRe(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_in_re(a, b)
        }
        TermKind::StrReplaceRe(s, re, r) => {
            let s = sub(s);
            let re = sub(re);
            let r = sub(r);
            manager.mk_str_replace_re(s, re, r)
        }
        TermKind::StrReplaceReAll(s, re, r) => {
            let s = sub(s);
            let re = sub(re);
            let r = sub(r);
            manager.mk_str_replace_re_all(s, re, r)
        }
        TermKind::StrLt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_lt(a, b)
        }
        TermKind::StrLe(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_str_le(a, b)
        }
        TermKind::StrToCode(a) => {
            let a = sub(a);
            manager.mk_str_to_code(a)
        }
        TermKind::StrFromCode(a) => {
            let a = sub(a);
            manager.mk_str_from_code(a)
        }

        // ===== Floating point =====
        TermKind::FpAbs(a) => {
            let a = sub(a);
            manager.mk_fp_abs(a)
        }
        TermKind::FpNeg(a) => {
            let a = sub(a);
            manager.mk_fp_neg(a)
        }
        TermKind::FpSqrt(rm, a) => {
            let a = sub(a);
            manager.mk_fp_sqrt(rm, a)
        }
        TermKind::FpRoundToIntegral(rm, a) => {
            let a = sub(a);
            manager.mk_fp_round_to_integral(rm, a)
        }
        TermKind::FpAdd(rm, a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_add(rm, a, b)
        }
        TermKind::FpSub(rm, a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_sub(rm, a, b)
        }
        TermKind::FpMul(rm, a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_mul(rm, a, b)
        }
        TermKind::FpDiv(rm, a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_div(rm, a, b)
        }
        TermKind::FpRem(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_rem(a, b)
        }
        TermKind::FpMin(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_min(a, b)
        }
        TermKind::FpMax(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_max(a, b)
        }
        TermKind::FpLeq(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_leq(a, b)
        }
        TermKind::FpLt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_lt(a, b)
        }
        TermKind::FpGeq(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_geq(a, b)
        }
        TermKind::FpGt(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_gt(a, b)
        }
        TermKind::FpEq(a, b) => {
            let a = sub(a);
            let b = sub(b);
            manager.mk_fp_eq(a, b)
        }
        TermKind::FpFma(rm, a, b, c) => {
            let a = sub(a);
            let b = sub(b);
            let c = sub(c);
            manager.mk_fp_fma(rm, a, b, c)
        }
        TermKind::FpIsNormal(a) => {
            let a = sub(a);
            manager.mk_fp_is_normal(a)
        }
        TermKind::FpIsSubnormal(a) => {
            let a = sub(a);
            manager.mk_fp_is_subnormal(a)
        }
        TermKind::FpIsZero(a) => {
            let a = sub(a);
            manager.mk_fp_is_zero(a)
        }
        TermKind::FpIsInfinite(a) => {
            let a = sub(a);
            manager.mk_fp_is_infinite(a)
        }
        TermKind::FpIsNaN(a) => {
            let a = sub(a);
            manager.mk_fp_is_nan(a)
        }
        TermKind::FpIsNegative(a) => {
            let a = sub(a);
            manager.mk_fp_is_negative(a)
        }
        TermKind::FpIsPositive(a) => {
            let a = sub(a);
            manager.mk_fp_is_positive(a)
        }
        TermKind::FpToReal(a) => {
            let a = sub(a);
            manager.mk_fp_to_real(a)
        }
        TermKind::FpToFp { rm, arg, eb, sb } => {
            let arg = sub(arg);
            manager.mk_fp_to_fp(rm, arg, eb, sb)
        }
        TermKind::FpToSBV { rm, arg, width } => {
            let arg = sub(arg);
            manager.mk_fp_to_sbv(rm, arg, width)
        }
        TermKind::FpToUBV { rm, arg, width } => {
            let arg = sub(arg);
            manager.mk_fp_to_ubv(rm, arg, width)
        }
        TermKind::RealToFp { rm, arg, eb, sb } => {
            let arg = sub(arg);
            manager.mk_real_to_fp(rm, arg, eb, sb)
        }
        TermKind::SBVToFp { rm, arg, eb, sb } => {
            let arg = sub(arg);
            manager.mk_sbv_to_fp(rm, arg, eb, sb)
        }
        TermKind::UBVToFp { rm, arg, eb, sb } => {
            let arg = sub(arg);
            manager.mk_ubv_to_fp(rm, arg, eb, sb)
        }

        // ===== Uninterpreted function application =====
        TermKind::Apply { func, args } => {
            let args: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
            manager.intern(TermKind::Apply { func, args }, sort)
        }

        // ===== Algebraic datatypes =====
        TermKind::DtConstructor { constructor, args } => {
            let args: SmallVec<[TermId; 4]> = args.iter().map(|&a| sub(a)).collect();
            manager.intern(TermKind::DtConstructor { constructor, args }, sort)
        }
        TermKind::DtTester { constructor, arg } => {
            let arg = sub(arg);
            manager.intern(TermKind::DtTester { constructor, arg }, sort)
        }
        TermKind::DtSelector { selector, arg } => {
            let arg = sub(arg);
            manager.intern(TermKind::DtSelector { selector, arg }, sort)
        }

        TermKind::Forall { .. }
        | TermKind::Exists { .. }
        | TermKind::Let { .. }
        | TermKind::Match { .. } => {
            unreachable!(
                "binders open their own scope in `expand` and are rebuilt by a dedicated \
                 Combine* step, never by a plain Combine"
            )
        }
    }
}
