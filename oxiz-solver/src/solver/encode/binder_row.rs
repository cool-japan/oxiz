//! Read-over-write expansion **under a binder**.
//!
//! # The hole this closes
//!
//! `(select (store a i v) k)` is not a primitive: the theory of arrays defines
//! it by the read-over-write axiom
//!
//! ```text
//! (select (store a i v) k)  ≡  (ite (= i k) v (select a k))
//! ```
//!
//! For a **ground** term the solver never needs the right-hand side: the term
//! is a root of [`Solver::instantiate_array_axioms`](crate::solver::Solver),
//! which emits exactly that case split lazily, on demand, for the pairs the
//! search actually needs.
//!
//! Under a binder there is no such root.  `array_axioms::ground_children`
//! stops at `Forall`/`Exists`/`Let`/`Match` — a `select` mentioning a bound
//! variable is an instance of nothing — so the lazy axiom never fires, and the
//! only thing that can refute the assertion is an MBQI instantiation at a
//! value that falsifies it.  MBQI picks its instances from the candidate
//! model, and in the candidate model the whole read is one opaque value of the
//! element sort, so every value of the binder looks equally satisfying and the
//! fixpoint reports `Satisfied`.  The result is a wrong `sat`:
//!
//! ```text
//! (declare-const a (Array (_ BitVec 7) (_ BitVec 7)))
//! (assert (= a ((as const (Array (_ BitVec 7) (_ BitVec 7))) (_ bv0 7))))
//! (assert (forall ((i (_ BitVec 7)))
//!           (= (select (store a i (_ bv5 7)) (_ bv1 7)) (_ bv5 7))))
//! ```
//!
//! `a` is pinned to the all-zero array, so at `i = #b0000000` the `store`
//! misses index 1 and the read is `#b0000000 ≠ #b0000101`: the script is
//! **unsatisfiable**, and the tree answered `sat`.
//!
//! Expanding the read-over-write *before* instantiation removes the opaque
//! value: the body becomes `(= (ite (= i #b1) #b5 (select a #b1)) #b5)`, whose
//! only array term, `(select a #b1)`, is ground.  The ground array solver pins
//! it to `#b0` from `a`'s definition, the body collapses to `(= i #b1)`, and
//! *any* instantiation but `i = #b1` refutes the quantifier.  Measured on the
//! repro above: `sat` (wrong) before, `unsat` after.
//!
//! # Why this is an equivalence, not an approximation
//!
//! The rewrite is the array axiom itself, so it holds in every model at every
//! polarity, with no side condition — exactly like the case split
//! `instantiate_array_axioms` emits for the ground term.  Declining to apply
//! it therefore costs completeness and never soundness, which is what every
//! guard below does when it is unsure.
//!
//! # Why it is confined to binder bodies
//!
//! Applying it to ground terms as well would be sound, and wrong for cost: the
//! ground path decides *which* case splits the search needs and emits only
//! those, while an eager expansion pays for every store of every chain in
//! every model.  The defect is only above the binder, so the rewrite is only
//! above the binder.
//!
//! # The guards, and what each of them costs
//!
//! * A quantifier is a candidate only when, **at each of its own
//!   occurrences**, the binders it would be spliced across bind no name that
//!   occurs free in it — which is what makes
//!   [`TermManager::substitute`](oxiz_core::ast::TermManager::substitute)
//!   incapable of alpha-renaming anything.  The chain checked runs from the
//!   occurrence outward and *stops at the nearest enclosing candidate
//!   quantifier*, because the rewrite is spliced into that quantifier's body
//!   and therefore never crosses its binder.  This replaced
//!   [`finite_expand::binder_names`]'s whole-assertion over-approximation,
//!   which a script could *evade*: adding the valid, contentless conjunct
//!   `(forall ((a (_ BitVec 7))) (= a a))` beside a real quantifier whose
//!   only free variable is the array `a` put the name `a` in the bound set
//!   and declined the real quantifier — a wrong `sat` where the same two
//!   quantifiers as two assertions are `unsat` (`#P2b-55` (a)).
//! * Nested quantifiers are rewritten **innermost-first**, and the enclosing
//!   binder is rebuilt from its rewritten body rather than spliced across:
//!   `(forall ((j …)) (forall ((i …)) φ))` used to yield no rewrite at all,
//!   because the outer walk stops at the inner binder and splicing the inner
//!   one back would cross the outer binder (`#P2b-55` (b)).  Inside one
//!   quantifier's body the read-over-write walk still stops at a deeper
//!   binder: the reads in that scope belong to the inner quantifier's own
//!   turn, which has already happened.
//! * A quantifier carrying a trigger (`:pattern`) that mentions a term this
//!   pass would rewrite is declined outright.  Rewriting the body while
//!   leaving the trigger alone would disable that trigger, and rewriting the
//!   trigger would produce an `ite`, which is not a legal pattern.
//! * The store chain is peeled at most [`MAX_STORE_PEEL_DEPTH`] deep and at
//!   most [`MAX_REWRITES_PER_QUANTIFIER`] reads are rewritten per quantifier,
//!   so an adversarial term cannot turn one assertion into an exponential one.
//!
//! Reference: Z3's `smt/smt_theory_array` read-over-write axiom
//! (`array_rewriter::mk_select_core`) and the eager `(ite (= i j) v (select a
//! j))` expansion its `rewriter.expand_select_store` option performs.

use oxiz_core::ast::traversal::{
    collect_free_vars_including_patterns, contains_term, get_children,
};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use smallvec::SmallVec;

#[allow(unused_imports)]
use crate::prelude::*;

use oxiz_core::interner::Spur;

/// Maximum number of `store`s one read is peeled through.
///
/// A chain of `n` stores expands to `n` nested `ite`s, so the cost is linear
/// and the cap only matters for adversarial input.  Past it the remaining base
/// keeps its `select`, which is still the same term — declining costs
/// completeness only.
const MAX_STORE_PEEL_DEPTH: usize = 16;

/// Maximum number of read-over-write terms rewritten inside one quantifier.
///
/// A body carrying more than this is left entirely alone rather than half
/// rewritten, so the cost of the pass stays bounded by the assertion's size.
const MAX_REWRITES_PER_QUANTIFIER: usize = 256;

/// Maximum number of quantifier sub-terms one assertion may carry before the
/// pass declines it outright, mirroring
/// [`finite_expand`](super::finite_expand)'s cap for the same reason: the
/// candidate filter costs one free-variable walk per quantifier.
const MAX_CANDIDATE_QUANTIFIERS: usize = 64;

/// `term` with every read-over-write occurring under a binder expanded, or
/// `None` when nothing qualified.
///
/// The result is **equivalent** to `term`, and the caller asserts it *beside*
/// `term` rather than in place of it — see
/// [`Solver::assert_binder_row_lemma`](crate::solver::Solver) for the
/// measurement that forced that choice.
pub(crate) fn expand_reads_over_writes_under_binders(
    term: TermId,
    manager: &mut TermManager,
) -> Option<TermId> {
    let quantifiers = rewritable_quantifiers(term, manager);
    if quantifiers.is_empty() {
        return None;
    }

    // Innermost-first, so that an enclosing quantifier is rebuilt from a body
    // whose own binders have already been rewritten.  `splice` accumulates the
    // rewrites; applying it to a *body* never crosses that body's own binder,
    // which is what makes the nested case expressible at all.
    let mut splice: FxHashMap<TermId, TermId> = FxHashMap::default();
    for quantifier in quantifiers {
        let Some(rewritten) = rewrite_quantifier(quantifier, &splice, manager) else {
            continue;
        };
        splice.insert(quantifier, rewritten);
    }
    if splice.is_empty() {
        return None;
    }

    let current = manager.substitute(term, &splice);
    (current != term).then_some(current)
}

/// The quantifiers of `term` this pass may rewrite, innermost-first.
///
/// A quantifier qualifies when, at every one of its occurrences, the binders
/// its rewrite would be *spliced across* bind no name that occurs free in it.
/// The chain of such binders runs from the occurrence outward and stops at the
/// nearest enclosing quantifier that is itself a candidate, because the
/// rewrite is spliced into that quantifier's body and so never crosses its
/// binder.  Since "is a candidate" appears on both sides, the filter is a
/// downward fixpoint: everything starts as a candidate and a decline can only
/// lengthen someone else's chain, so at most one pass per quantifier is needed
/// and the iteration terminates.
///
/// The rewritten quantifier has exactly the original's free variables, so a
/// splice that cannot capture the original cannot capture the replacement
/// either — which is the property the whole filter exists to establish.
fn rewritable_quantifiers(term: TermId, manager: &TermManager) -> Vec<TermId> {
    let occurrences = quantifier_occurrences(term, manager);
    if occurrences.is_empty() || occurrences.len() > MAX_CANDIDATE_QUANTIFIERS {
        return Vec::new();
    }

    let mut free_names: FxHashMap<TermId, FxHashSet<Spur>> = FxHashMap::default();
    for occurrence in &occurrences {
        let names: FxHashSet<Spur> =
            collect_free_vars_including_patterns(occurrence.quantifier, manager)
                .iter()
                .filter_map(|&free| match manager.get(free).map(|t| &t.kind) {
                    Some(TermKind::Var(name)) => Some(*name),
                    _ => None,
                })
                .collect();
        free_names.insert(occurrence.quantifier, names);
    }

    let mut declined: FxHashSet<TermId> = FxHashSet::default();
    loop {
        let mut changed = false;
        for occurrence in &occurrences {
            if declined.contains(&occurrence.quantifier) {
                continue;
            }
            let Some(free) = free_names.get(&occurrence.quantifier) else {
                continue;
            };
            if occurrence
                .chains
                .iter()
                .any(|chain| crosses_a_capturing_binder(chain, free, &declined))
            {
                declined.insert(occurrence.quantifier);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    occurrences
        .iter()
        .filter(|occurrence| !declined.contains(&occurrence.quantifier))
        .map(|occurrence| occurrence.quantifier)
        .collect()
}

/// Does splicing a replacement whose free names are `free` into this
/// occurrence cross a binder that would capture it?
///
/// `chain` runs from the occurrence outward.  The scan stops at the first
/// enclosing quantifier that is still a candidate: the rewrite is spliced into
/// *that* quantifier's body, so its own binder is never crossed.
fn crosses_a_capturing_binder(
    chain: &[EnclosingBinder],
    free: &FxHashSet<Spur>,
    declined: &FxHashSet<TermId>,
) -> bool {
    for binder in chain {
        if let Some(quantifier) = binder.quantifier
            && !declined.contains(&quantifier)
        {
            return false;
        }
        if binder.names.iter().any(|name| free.contains(name)) {
            return true;
        }
    }
    false
}

/// One binder enclosing an occurrence: the names it binds, and its term when
/// it is a quantifier (a `Let` or a `Match` is never a splice target, so it is
/// recorded with `None` and its names always count).
#[derive(Clone)]
struct EnclosingBinder {
    quantifier: Option<TermId>,
    names: SmallVec<[Spur; 2]>,
}

/// One quantifier sub-term of the assertion, with the enclosing-binder chain
/// of each of its occurrences (innermost binder first).
struct QuantifierOccurrence {
    quantifier: TermId,
    chains: Vec<Vec<EnclosingBinder>>,
}

/// Every quantifier sub-term of `term`, innermost-first, each with the
/// enclosing-binder chain of every position it occurs at.
///
/// Iterative with an explicit stack over a *tree* walk rather than a DAG walk:
/// the same hash-consed quantifier can sit in two different binder scopes, and
/// the chain is exactly what distinguishes them, so a `visited` set keyed by
/// term alone would throw away the question being asked.  The cap on the
/// number of distinct quantifiers (checked by the caller) bounds the work.
fn quantifier_occurrences(term: TermId, manager: &TermManager) -> Vec<QuantifierOccurrence> {
    let mut found: FxHashMap<TermId, Vec<Vec<EnclosingBinder>>> = FxHashMap::default();
    let mut order: Vec<TermId> = Vec::new();
    let mut depths: FxHashMap<TermId, usize> = FxHashMap::default();
    let mut stack: Vec<(TermId, usize)> = vec![(term, 0)];
    let mut chain: Vec<EnclosingBinder> = Vec::new();
    let mut visited: FxHashSet<(TermId, usize)> = FxHashSet::default();

    while let Some((current, depth)) = stack.pop() {
        chain.truncate(depth);
        if !visited.insert((current, depth)) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        let binder = binder_bound_names(current, &kind, manager);
        if matches!(kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
            let mut outward = chain.clone();
            outward.reverse();
            if !found.contains_key(&current) {
                order.push(current);
            }
            let entry = found.entry(current).or_default();
            entry.push(outward);
            let seen = depths.entry(current).or_insert(depth);
            *seen = (*seen).max(depth);
            if order.len() > MAX_CANDIDATE_QUANTIFIERS {
                return Vec::new();
            }
        }
        let child_depth = match binder {
            Some(binder) => {
                chain.truncate(depth);
                chain.push(binder);
                depth + 1
            }
            None => depth,
        };
        for child in get_children(&kind) {
            stack.push((child, child_depth));
        }
    }

    // Innermost-first: a quantifier that *contains* another occurs at a
    // strictly smaller binder depth than the one it contains, so ordering by
    // deepest occurrence descending rewrites the inner one first.
    order.sort_by_key(|q| core::cmp::Reverse(depths.get(q).copied().unwrap_or(0)));
    order
        .into_iter()
        .filter_map(|quantifier| {
            found
                .remove(&quantifier)
                .map(|chains| QuantifierOccurrence { quantifier, chains })
        })
        .collect()
}

/// The names `term` binds over its children, or `None` when it binds nothing.
fn binder_bound_names(
    term: TermId,
    kind: &TermKind,
    manager: &TermManager,
) -> Option<EnclosingBinder> {
    match kind {
        TermKind::Forall { vars, .. } | TermKind::Exists { vars, .. } => Some(EnclosingBinder {
            quantifier: Some(term),
            names: vars.iter().map(|&(name, _)| name).collect(),
        }),
        TermKind::Let { bindings, .. } => Some(EnclosingBinder {
            quantifier: None,
            names: bindings.iter().map(|&(name, _)| name).collect(),
        }),
        TermKind::Match { cases, .. } => {
            let names: SmallVec<[Spur; 2]> = cases
                .iter()
                .flat_map(|case| case.bindings.iter().copied())
                .collect();
            let _ = manager;
            Some(EnclosingBinder {
                quantifier: None,
                names,
            })
        }
        _ => None,
    }
}

/// One quantifier with every read-over-write in its own scope expanded, or
/// `None` when it carries none or a guard declined it.
///
/// `inner` carries the rewrites of the quantifiers nested inside this one,
/// already built (the sweep is innermost-first).  Applying it to the **body**
/// rather than to the quantifier is what keeps the nested case sound: the
/// substitution starts inside this binder, so this binder is never crossed and
/// `substitute` has no reason to alpha-rename it.
fn rewrite_quantifier(
    quantifier: TermId,
    inner: &FxHashMap<TermId, TermId>,
    manager: &mut TermManager,
) -> Option<TermId> {
    let (vars, body, patterns, is_exists) = match manager.get(quantifier).map(|t| t.kind.clone())? {
        TermKind::Forall {
            vars,
            body,
            patterns,
        } => (vars, body, patterns, false),
        TermKind::Exists {
            vars,
            body,
            patterns,
        } => (vars, body, patterns, true),
        _ => return None,
    };

    let body = if inner.is_empty() {
        body
    } else {
        manager.substitute(body, inner)
    };

    let new_body = match read_over_write_map(body, manager) {
        Some(map) => {
            // A trigger mentioning a term this pass rewrites would be left
            // pointing at a term the body no longer contains.  Decline the
            // whole quantifier rather than disable its trigger.
            for pattern in &patterns {
                for &pattern_term in pattern {
                    if map
                        .keys()
                        .any(|&key| contains_term(pattern_term, key, manager))
                    {
                        return None;
                    }
                }
            }
            manager.substitute(body, &map)
        }
        // No read-over-write of this quantifier's own, but an inner one was
        // rewritten: rebuilding the binder around the new body is the whole
        // point of the innermost-first sweep.
        None => body,
    };

    let names: Vec<(String, SortId)> = vars
        .iter()
        .map(|&(name, sort)| (manager.resolve_str(name).to_string(), sort))
        .collect();
    let patterns: Vec<Vec<TermId>> = patterns
        .iter()
        .map(|pattern| pattern.iter().copied().collect())
        .collect();
    let bound = names.iter().map(|(name, sort)| (name.as_str(), *sort));

    let rebuilt = if is_exists {
        manager.mk_exists_with_patterns(bound, new_body, patterns)
    } else {
        manager.mk_forall_with_patterns(bound, new_body, patterns)
    };
    (rebuilt != quantifier).then_some(rebuilt)
}

/// Every read-over-write in `body`'s *own* binder scope, mapped to its
/// expansion.
///
/// The walk stops at a deeper binder: see the module doc for why rewriting
/// inside one would be dropped by `substitute`'s capture avoidance.
fn read_over_write_map(
    body: TermId,
    manager: &mut TermManager,
) -> Option<FxHashMap<TermId, TermId>> {
    let mut stack: Vec<TermId> = vec![body];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut targets: Vec<(TermId, TermId, TermId)> = Vec::new();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        if matches!(
            kind,
            TermKind::Forall { .. }
                | TermKind::Exists { .. }
                | TermKind::Let { .. }
                | TermKind::Match { .. }
        ) {
            continue;
        }
        if let TermKind::Select(array, index) = kind {
            if matches!(
                manager.get(array).map(|t| &t.kind),
                Some(TermKind::Store(..))
            ) {
                targets.push((current, array, index));
                if targets.len() > MAX_REWRITES_PER_QUANTIFIER {
                    return None;
                }
            }
        }
        stack.extend(get_children(&kind));
    }

    if targets.is_empty() {
        return None;
    }

    let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
    for (select, array, index) in targets {
        let expanded = peel_store_chain(array, index, manager);
        if expanded != select {
            map.insert(select, expanded);
        }
    }
    (!map.is_empty()).then_some(map)
}

/// `(select <array> <index>)` with the `store` chain on `array` peeled into
/// nested `ite`s, innermost store last.
fn peel_store_chain(array: TermId, index: TermId, manager: &mut TermManager) -> TermId {
    let mut base = array;
    let mut frames: SmallVec<[(TermId, TermId); 8]> = SmallVec::new();
    while frames.len() < MAX_STORE_PEEL_DEPTH {
        let Some(TermKind::Store(inner, store_index, value)) =
            manager.get(base).map(|t| t.kind.clone())
        else {
            break;
        };
        frames.push((store_index, value));
        base = inner;
    }

    let mut result = manager.mk_select(base, index);
    while let Some((store_index, value)) = frames.pop() {
        let guard = manager.mk_eq(store_index, index);
        result = manager.mk_ite(guard, value, result);
    }
    result
}

#[cfg(test)]
mod tests;
