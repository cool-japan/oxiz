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
//! * A quantifier is a candidate only when its free variables are disjoint
//!   from every name bound anywhere in the assertion, which is
//!   [`finite_expand::binder_names`]'s over-approximation and what makes
//!   splicing the rewritten quantifier back with
//!   [`TermManager::substitute`](oxiz_core::ast::TermManager::substitute)
//!   incapable of alpha-renaming anything.
//! * Inside one quantifier's body the walk stops at a *deeper* binder: a
//!   replacement that mentions the inner binder's variable would make
//!   `substitute` alpha-rename that binder and silently drop the rewrite.
//!   The inner quantifier gets its own turn on a later sweep, once the outer
//!   one has been expanded or instantiated away.
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

use super::finite_expand::binder_names;

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

    let mut current = term;
    for quantifier in quantifiers {
        let Some(rewritten) = rewrite_quantifier(quantifier, manager) else {
            continue;
        };
        let mut splice: FxHashMap<TermId, TermId> = FxHashMap::default();
        splice.insert(quantifier, rewritten);
        current = manager.substitute(current, &splice);
    }

    (current != term).then_some(current)
}

/// The quantifiers of `term` this pass may rewrite.
///
/// The filter is [`finite_expand::expandable_quantifiers`]'s: a quantifier
/// qualifies only when none of its free variables shares a name with a binder
/// anywhere in `term`.  That is what makes splicing the rewritten quantifier
/// back in safe — the rewritten quantifier has exactly the original's free
/// variables, so a `substitute` that cannot capture the original cannot
/// capture the replacement either.
fn rewritable_quantifiers(term: TermId, manager: &TermManager) -> Vec<TermId> {
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut quantifiers: Vec<TermId> = Vec::new();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
            continue;
        };
        if matches!(kind, TermKind::Forall { .. } | TermKind::Exists { .. }) {
            quantifiers.push(current);
            if quantifiers.len() > MAX_CANDIDATE_QUANTIFIERS {
                return Vec::new();
            }
        }
        stack.extend(get_children(&kind));
    }
    if quantifiers.is_empty() {
        return quantifiers;
    }

    let bound_anywhere = binder_names(term, manager);
    quantifiers.retain(|&quantifier| {
        !collect_free_vars_including_patterns(quantifier, manager)
            .iter()
            .filter_map(|&free| match manager.get(free).map(|t| &t.kind) {
                Some(TermKind::Var(name)) => Some(*name),
                _ => None,
            })
            .any(|name| bound_anywhere.contains(&name))
    });
    quantifiers
}

/// One quantifier with every read-over-write in its own scope expanded, or
/// `None` when it carries none or a guard declined it.
fn rewrite_quantifier(quantifier: TermId, manager: &mut TermManager) -> Option<TermId> {
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

    let map = read_over_write_map(body, manager)?;

    // A trigger mentioning a term this pass rewrites would be left pointing at
    // a term the body no longer contains.  Decline the whole quantifier rather
    // than disable its trigger.
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

    let new_body = manager.substitute(body, &map);
    if new_body == body {
        return None;
    }

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
