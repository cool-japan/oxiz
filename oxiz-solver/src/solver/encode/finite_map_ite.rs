//! Flattening for finite-map lookup spines built out of nested equality
//! `ite`.
//!
//! Tool generators frequently encode a finite lookup table as a right-leaning
//! chain of guarded `ite`:
//!
//! ```text
//! (ite (= idx 1) v1 (ite (= idx 2) v2 (ite (= idx 3) v3 default)))
//! ```
//!
//! Handed to [`super::bool_euf_encoding::Solver::eliminate_nonbool_ite`]
//! as-is, each nesting level gets its *own* fresh mux variable, so recovering
//! the table's value at a chosen index costs chasing an `n`-deep chain of
//! one-directional implications. Because none of those per-level guards are
//! declared mutually exclusive, CDCL(T) can explore combinations of them
//! before arithmetic ever refutes an inconsistent one — the search cost grows
//! with the chain's depth in a way a flat encoding does not need to pay.
//!
//! This pass recognises such a chain (a "lookup spine") ending at a
//! non-matching default branch and rewrites it, in one step, into:
//!
//! * a single fresh result constant `r`,
//! * one flat implication per key, `(= idx k_i) => (= r v_i)`, and
//! * one implication for the fallthrough, `(and (not (= idx k_1)) …) => (= r
//!   default)`.
//!
//! Every guard now names `idx` directly rather than a chain of nested
//! predecessors, which is both a smaller encoding and a shape
//! [`super::super::int_case_split`]'s existing case-split refinement and the
//! at-most-one clauses added here (see [`Solver::flatten_lookup_spines`]) can
//! act on directly.
//!
//! # What this pass deliberately does not attempt
//!
//! It does not try to recognise 0/1-valued arithmetic idioms (`max`, `abs`,
//! saturating select, …) layered over a flattened result, and it does not
//! maintain a cross-assertion alias cache for `define-fun` bodies. Both are
//! narrow, benchmark-family-specific tunings layered on top of the same idea
//! upstream; the cost/benefit was judged not to clear the bar here — see the
//! session report for the reasoning.

use rustc_hash::{FxHashMap, FxHashSet};

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_sat::Lit;

use super::bool_euf_encoding::{collect_ground_subterms, needs_ite_elimination};
use super::*;

/// Minimum number of distinct keys a right-leaning equality-`ite` chain must
/// carry before flattening it is worth the rewrite. Shorter chains are left
/// for the generic muxer in `eliminate_nonbool_ite`, which handles a
/// two-or-three-deep `ite` just fine on its own.
const MIN_LOOKUP_ARMS: usize = 3;

/// Above this many distinct keys, skip the pairwise at-most-one clauses for
/// an index's domain (still keeps the defining implications and the
/// candidate registration for the lazy case-split refinement). Pairwise AMO
/// is `O(n^2)`; a wide table would otherwise spend more clauses on mutual
/// exclusion bookkeeping than the table itself contains entries.
const MAX_AMO_ARMS: usize = 24;

/// One recognised lookup spine: `root` is the outermost `ite`, `index` is the
/// term every guard in the chain compares against a distinct integer key,
/// `arms` is `(key, value)` in first-occurrence order (a repeated key further
/// down the chain is dead — `ite` picks the first match — and is dropped
/// during matching, not here), `default` is the terminal non-matching
/// branch, and `chain_nodes` is every `ite` node from `root` down to (but not
/// including) `default`, used to keep a shorter, nested spine match from
/// being selected independently once its nodes are already claimed by a
/// longer one.
struct LookupSpine {
    root: TermId,
    index: TermId,
    arms: Vec<(i64, TermId)>,
    default: TermId,
    chain_nodes: Vec<TermId>,
}

/// If `a`/`b` is `(index, IntConst)` in either order, return `(index, key)`.
/// `None` when neither side (or both sides) is an integer literal — the
/// guard is then not a key comparison this pass understands.
fn split_index_and_key(a: TermId, b: TermId, manager: &TermManager) -> Option<(TermId, i64)> {
    use num_traits::ToPrimitive;
    let a_kind = &manager.get(a)?.kind;
    let b_kind = &manager.get(b)?.kind;
    match (a_kind, b_kind) {
        (TermKind::IntConst(n), TermKind::IntConst(_)) => {
            // Both constant: not an index comparison at all (would collapse
            // to True/False by simplification anyway); decline rather than
            // guess which side is "the index".
            let _ = n;
            None
        }
        (TermKind::IntConst(n), _) => Some((b, n.to_i64()?)),
        (_, TermKind::IntConst(n)) => Some((a, n.to_i64()?)),
        _ => None,
    }
}

/// Try to read `root` as the head of a lookup spine. `None` when `root` is
/// not an eligible `ite`, its guard is not an `(= index key)` comparison
/// over an `Int`-sorted index, or fewer than [`MIN_LOOKUP_ARMS`] distinct
/// keys chain together before the guard shape breaks (that break point is
/// exactly `default`).
fn match_lookup_spine(root: TermId, manager: &TermManager) -> Option<LookupSpine> {
    let root_node = manager.get(root)?;
    let TermKind::Ite(cond0, then0, else0) = &root_node.kind else {
        return None;
    };
    if !needs_ite_elimination(root_node.sort, manager) {
        return None;
    }
    let cond0_node = manager.get(*cond0)?;
    let TermKind::Eq(a0, b0) = &cond0_node.kind else {
        return None;
    };
    let (index, first_key) = split_index_and_key(*a0, *b0, manager)?;
    if manager.get(index)?.sort != manager.sorts.int_sort {
        return None;
    }

    let mut seen_keys: FxHashSet<i64> = FxHashSet::default();
    seen_keys.insert(first_key);
    let mut arms = vec![(first_key, *then0)];
    let mut chain_nodes = vec![root];
    let mut cursor = *else0;

    while let Some(node) = manager.get(cursor) {
        let TermKind::Ite(cond, then_branch, else_branch) = &node.kind else {
            break;
        };
        if node.sort != root_node.sort {
            break;
        }
        let Some(cond_node) = manager.get(*cond) else {
            break;
        };
        let TermKind::Eq(a, b) = &cond_node.kind else {
            break;
        };
        let Some((this_index, key)) = split_index_and_key(*a, *b, manager) else {
            break;
        };
        if this_index != index {
            break;
        }
        chain_nodes.push(cursor);
        if seen_keys.insert(key) {
            arms.push((key, *then_branch));
        }
        // A repeated key's arm is unreachable (the first match already wins)
        // and is intentionally dropped rather than re-added: keeping it would
        // force its value equal to the first arm's, which is not what the
        // original chain means.
        cursor = *else_branch;
    }

    if arms.len() < MIN_LOOKUP_ARMS {
        return None;
    }
    Some(LookupSpine {
        root,
        index,
        arms,
        default: cursor,
        chain_nodes,
    })
}

/// Of a set of overlapping candidate spines, keep the maximal ones: sort
/// longest-first and skip any candidate whose root was already claimed by an
/// earlier (longer-or-equal) selection's chain. A spine nested inside
/// another's default branch, or reachable only through an arm value, is
/// unaffected by this and is free to be selected independently.
fn select_maximal_spines(mut candidates: Vec<LookupSpine>) -> Vec<LookupSpine> {
    candidates.sort_by_key(|spine| core::cmp::Reverse(spine.arms.len()));
    let mut claimed: FxHashSet<TermId> = FxHashSet::default();
    let mut chosen = Vec::with_capacity(candidates.len());
    for spine in candidates {
        if claimed.contains(&spine.root) {
            continue;
        }
        claimed.extend(spine.chain_nodes.iter().copied());
        chosen.push(spine);
    }
    chosen
}

impl Solver {
    /// Rewrite every maximal lookup spine reachable from `term` (outside a
    /// binder) into a fresh result constant plus flat key implications,
    /// register each spine's index for the finite-domain case-split
    /// refinement in [`super::super::int_case_split`], and add the sound,
    /// unconditional pairwise at-most-one clauses over its keys (see
    /// [`MAX_AMO_ARMS`]).
    ///
    /// Runs before [`Solver::eliminate_nonbool_ite`] (see [`Solver::assert`]):
    /// once a spine here is rewritten to `(= r ...)` implications, none of
    /// its `ite` nodes remain in the term for the generic muxer to expand
    /// into its own, per-level fresh variables.
    pub(super) fn flatten_lookup_spines(
        &mut self,
        term: TermId,
        manager: &mut TermManager,
    ) -> TermId {
        let candidates: Vec<LookupSpine> = collect_ground_subterms(term, manager)
            .into_iter()
            .filter_map(|st| match_lookup_spine(st, manager))
            .collect();
        if candidates.is_empty() {
            return term;
        }
        let spines = select_maximal_spines(candidates);

        let mut result_of: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut side_conditions: Vec<TermId> = Vec::new();

        for (ordinal, spine) in spines.iter().enumerate() {
            let sort = manager
                .get(spine.root)
                .map(|t| t.sort)
                .unwrap_or(manager.sorts.int_sort);
            let result = manager.mk_var(
                &oxiz_core::smtlib::reserved_name("lookup", &format!("{}-{ordinal}", spine.root.0)),
                sort,
            );
            result_of.insert(spine.root, result);

            let mut not_any_key: Vec<TermId> = Vec::with_capacity(spine.arms.len());
            let mut key_lits: Vec<Lit> = Vec::with_capacity(spine.arms.len());
            for &(key, value) in &spine.arms {
                let key_term = manager.mk_int(key);
                let key_eq = manager.mk_eq(spine.index, key_term);
                let val_eq = manager.mk_eq(result, value);
                side_conditions.push(manager.mk_implies(key_eq, val_eq));
                not_any_key.push(manager.mk_not(key_eq));
                if spine.arms.len() <= MAX_AMO_ARMS {
                    // `Solver::encode`: see the same change in
                    // `int_case_split::assert_value_disjunction` — the depth-0
                    // entry point is the one choke point the array-refinement
                    // guard is computed at, and every caller of it has to go
                    // through `encode` for that to hold.
                    key_lits.push(self.encode(key_eq, manager));
                }
            }
            let none_matched = manager.mk_and(not_any_key);
            let default_eq = manager.mk_eq(result, spine.default);
            side_conditions.push(manager.mk_implies(none_matched, default_eq));

            // Sound, unconditional pairwise mutual exclusion: `idx` cannot
            // equal two distinct integer literals at once, regardless of any
            // other hypothesis, so this needs no entailment check the way a
            // case-split disjunction would (see `split_narrow_int_domains`'s
            // module doc for why *that* direction needs one).
            for i in 0..key_lits.len() {
                for j in (i + 1)..key_lits.len() {
                    self.sat
                        .add_clause([key_lits[i].negate(), key_lits[j].negate()]);
                }
            }
            if self.config.enable_domain_first_branching {
                self.push_branch_priority(&key_lits);
            }

            self.lookup_index_terms.insert(spine.index);
        }

        let rewritten = manager.substitute(term, &result_of);
        let mut parts = Vec::with_capacity(1 + side_conditions.len());
        parts.push(rewritten);
        // A side condition can itself mention another chosen spine's root
        // (one table's arm value, or its default, containing another table),
        // so resolve every side condition through the *complete* map, built
        // above from every spine regardless of which order they were
        // processed in.
        for side in &mut side_conditions {
            *side = manager.substitute(*side, &result_of);
        }
        parts.extend(side_conditions);
        manager.mk_and(parts)
    }
}

#[cfg(test)]
mod tests;
