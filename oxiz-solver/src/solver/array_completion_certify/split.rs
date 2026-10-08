//! A universal over an **infinite** index sort, certified by splitting each
//! bound variable into the points the interpretation names and the region it
//! does not (`#P2b-51`, decision (54)(ii)).
//!
//! Over `Int` and `Real` the completion interprets an array as a default plus
//! finitely many points, `(store … ((as const A) d) p₁ v₁ …)`.  The validity
//! query over such a chain is exact in principle and weak in practice: a
//! `Real`-indexed chain read at a fresh constant is `unknown` on every build
//! (recheck 13, `t07_real_replay`), so a model that satisfies `∀x. a[x] > 0`
//! could not be certified at all.  But the chain's reading is known in
//! closed form — `vᵢ` at `pᵢ`, `d` everywhere else — so the universal can be
//! decided pointwise instead:
//!
//! * at every tuple of **named** points (each bound variable at one of the
//!   store indices of an array it reads) the instance is closed, and is
//!   evaluated exactly;
//! * where a bound variable `x` is **unnamed** (different from every named
//!   point) every read `select(V, x)` is `V`'s default, so the instance with
//!   those reads replaced is array-free; it is evaluated when `x` no longer
//!   occurs, and otherwise discharged as the arithmetic validity query
//!   `(x ≠ p₁ ∧ … ∧ x ≠ pₙ) ⇒ instance`.
//!
//! Together the tuples cover every assignment of the bound variables, so the
//! universal holds exactly when every tuple does.  The split applies only
//! where every read under the binder is at a bare bound variable or at a
//! ground index, of an array the substitution interprets as a literal value;
//! anything else (`a[x + 1]`, a `store` under the binder, an array left
//! symbolic) falls back to the validity query unchanged.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortKind;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{Combinations, is_array_value, peel_universal_with_vars, run_query};
use crate::solver::{Solver, SolverResult};

/// Tuples of points one universal may be split into.
const MAX_SPLIT_TUPLES: usize = 256;

/// `Some(true)` when the universal holds under `substitution` at every tuple,
/// `Some(false)` when one tuple is not certified (an exact evaluation shows it
/// false, or its query is not refuted), and `None` when the split does not
/// apply (the caller then runs the validity query).
#[allow(clippy::too_many_arguments)]
pub(super) fn universal_by_points(
    solver: &Solver,
    universal: TermId,
    position: usize,
    substitution: &FxHashMap<TermId, TermId>,
    hypotheses: &[TermId],
    manager: &mut TermManager,
    logic: Option<&str>,
    queries: &mut usize,
) -> Option<bool> {
    let (body, vars) = peel_universal_with_vars(universal, position, manager)?;
    if vars.is_empty() || !vars.iter().all(|&var| has_infinite_sort(var, manager)) {
        return None;
    }
    let body = manager.substitute(body, substitution);
    let var_set: FxHashSet<TermId> = vars.iter().copied().collect();
    let reads = reads_under_binder(body, &var_set, manager)?;

    // The named points of each bound variable: every store index of every
    // array it reads.
    let mut named: FxHashMap<TermId, Vec<TermId>> = FxHashMap::default();
    for &(_, array, index) in &reads {
        let list = named.entry(index).or_default();
        for point in store_indices(array, manager) {
            if !list.contains(&point) {
                list.push(point);
            }
        }
    }
    // Option lists: `Some(point)` per named point, then `None` (unnamed).
    let mut option_lists: Vec<Vec<Option<TermId>>> = Vec::with_capacity(vars.len());
    let mut total: usize = 1;
    for var in &vars {
        let mut list: Vec<Option<TermId>> = named
            .get(var)
            .map(|points| points.iter().copied().map(Some).collect())
            .unwrap_or_default();
        list.push(None);
        total = total.saturating_mul(list.len());
        option_lists.push(list);
    }
    if total > MAX_SPLIT_TUPLES {
        return None;
    }
    for tuple in option_tuples(&option_lists) {
        let mut reduce: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut assign: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut hypotheses: Vec<TermId> = hypotheses.to_vec();
        let mut unnamed: Vec<TermId> = Vec::new();
        for (&var, choice) in vars.iter().zip(tuple.iter()) {
            match choice {
                Some(point) => {
                    assign.insert(var, *point);
                }
                None => {
                    unnamed.push(var);
                    for &(read, array, index) in &reads {
                        if index == var {
                            reduce.insert(
                                read,
                                super::super::array_axioms::const_array_default(
                                    innermost_const(array, manager)?,
                                    manager,
                                )?,
                            );
                        }
                    }
                    for &point in named.get(&var).map_or(&[][..], Vec::as_slice) {
                        let equal = manager.mk_eq(var, point);
                        hypotheses.push(manager.mk_not(equal));
                    }
                }
            }
        }
        let reduced = manager.substitute(body, &reduce);
        let instance = manager.substitute(reduced, &assign);
        let mentions_unnamed = manager
            .free_vars_including_patterns(instance)
            .into_iter()
            .any(|var| unnamed.contains(&var));
        if !mentions_unnamed {
            match super::evaluate_closed(solver, instance, manager) {
                Some(true) => continue,
                Some(false) => return Some(false),
                None => {}
            }
        }
        hypotheses.push(manager.mk_not(instance));
        let refutation = manager.mk_and(hypotheses);
        // Where the split applies it decides: an instance it cannot refute is
        // not certified, and the validity query over the store chain it would
        // fall back to asks the same question with arrays in it (over `Int`,
        // `∀i. a[i] = 2·i` kept a sub-solver's simplex busy for minutes).
        if !matches!(
            run_query(refutation, manager, logic, queries),
            Some(SolverResult::Unsat)
        ) {
            return Some(false);
        }
    }
    Some(true)
}

/// Whether the point split applies to `universal` whatever the completion
/// interprets `arrays` as: every bound variable has an infinite sort, and every
/// read under the binder that mentions one is a read of one of `arrays` at a
/// bare bound variable.  The candidate-point search (`infinite`) proposes only
/// where this holds, so its certificates never fall back to a validity query
/// over a store chain read at a compound index (`∀i. a[i + 1] = a[i] + 1` kept a
/// sub-solver's simplex busy for minutes).
pub(super) fn applies_to(
    universal: TermId,
    position: usize,
    arrays: &FxHashSet<TermId>,
    manager: &mut TermManager,
) -> bool {
    let Some((body, vars)) = peel_universal_with_vars(universal, position, manager) else {
        return false;
    };
    if vars.is_empty() || !vars.iter().all(|&var| has_infinite_sort(var, manager)) {
        return false;
    }
    let var_set: FxHashSet<TermId> = vars.iter().copied().collect();
    let mentions = |term: TermId, manager: &TermManager| -> bool {
        manager
            .free_vars_including_patterns(term)
            .into_iter()
            .any(|var| var_set.contains(&var))
    };
    let mut stack: Vec<TermId> = vec![body];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            return false;
        };
        match &data.kind {
            TermKind::Select(array, index) if mentions(*index, manager) => {
                if !var_set.contains(index) || !arrays.contains(array) {
                    return false;
                }
            }
            TermKind::Store(..) if mentions(current, manager) => return false,
            kind => {
                children.clear();
                children.extend(oxiz_core::ast::traversal::get_children(kind));
                stack.extend(children.iter().copied());
            }
        }
    }
    true
}

/// Whether `term`'s sort is `Int` or `Real`.
fn has_infinite_sort(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|data| {
        matches!(
            manager.sorts.get(data.sort).map(|s| &s.kind),
            Some(SortKind::Int | SortKind::Real)
        )
    })
}

/// Every read `select(V, x)` of `body` whose index mentions a bound variable,
/// as `(read, V, x)`; `None` when one is not of the split's shape (the index
/// is not a bare bound variable, or `V` is not a literal array value), or a
/// `store` under the binder mentions one.
fn reads_under_binder(
    body: TermId,
    vars: &FxHashSet<TermId>,
    manager: &TermManager,
) -> Option<Vec<(TermId, TermId, TermId)>> {
    let mentions = |term: TermId| -> bool {
        manager
            .free_vars_including_patterns(term)
            .into_iter()
            .any(|var| vars.contains(&var))
    };
    let mut out: Vec<(TermId, TermId, TermId)> = Vec::new();
    let mut stack: Vec<TermId> = vec![body];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let data = manager.get(current)?;
        match &data.kind {
            TermKind::Select(array, index) if mentions(*index) => {
                if !vars.contains(index) || !is_array_value(*array, manager) {
                    return None;
                }
                out.push((current, *array, *index));
            }
            TermKind::Store(..) if mentions(current) => return None,
            kind => {
                children.clear();
                children.extend(oxiz_core::ast::traversal::get_children(kind));
                stack.extend(children.iter().copied());
            }
        }
    }
    out.sort_unstable_by_key(|&(read, _, _)| read.raw());
    Some(out)
}

/// The literal store indices of an array value, outermost first.
fn store_indices(array: TermId, manager: &TermManager) -> Vec<TermId> {
    let mut out: Vec<TermId> = Vec::new();
    let mut current = array;
    while let Some(TermKind::Store(inner, index, _)) = manager.get(current).map(|d| &d.kind) {
        out.push(*index);
        current = *inner;
    }
    out
}

/// The array constant at the bottom of a store chain.
fn innermost_const(array: TermId, manager: &TermManager) -> Option<TermId> {
    let mut current = array;
    while let Some(TermKind::Store(inner, _, _)) = manager.get(current).map(|d| &d.kind) {
        current = *inner;
    }
    super::is_const_array(current, manager).then_some(current)
}

/// Every tuple of the per-variable option lists, in odometer order.
fn option_tuples(lists: &[Vec<Option<TermId>>]) -> Vec<Vec<Option<TermId>>> {
    // `Combinations` works over `TermId`s; index the options instead.
    let indexed: Vec<Vec<TermId>> = lists
        .iter()
        .map(|list| (0..list.len() as u32).map(TermId).collect())
        .collect();
    Combinations::with_cap(&indexed, MAX_SPLIT_TUPLES)
        .map(|combination| {
            combination
                .iter()
                .zip(lists.iter())
                .map(|(slot, list)| list.get(slot.raw() as usize).copied().flatten())
                .collect()
        })
        .collect()
}
