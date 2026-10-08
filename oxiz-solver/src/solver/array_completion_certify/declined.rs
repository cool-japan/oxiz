//! `#P2b-63`: the goals the completion used to **decline**, now interpreted
//! from the candidate model where the universals do not look.
//!
//! Decision (40) runs the completion on every quantified array `Sat`, but a
//! goal it declined kept the candidate model — and that model could falsify
//! its own universal.  Three decline rules were the reach of `#P2b-51`:
//!
//! * **an uninterpreted function anywhere in the goal** — an application
//!   survives into the certificate's validity query, where the sub-solver may
//!   interpret it any way it likes;
//! * **more than three array-sorted free variables** — the search is a
//!   product over per-array pools;
//! * **an `exists`** — certifying one needs a witness.
//!
//! Each is lifted the same way: whatever no universal reads through a binder
//! is given the value the candidate model already gives it, and only the rest
//! is completed.
//!
//! * A **ground application** `f(k⃗)` (no bound variable in it) whose
//!   arguments are literals or scalars the candidate pins is replaced by its
//!   model value.  The published interpretation of `f` is untouched — it is
//!   the candidate's, and the certificate verified the goal at exactly the
//!   values it prints.  Two applications whose arguments evaluate alike must
//!   carry one value (a function), or the attempt declines.
//! * An array read **only** at ground indices (never under a binder at a
//!   bound index, never stored into, compared or passed on) is not completed
//!   when the goal has more arrays than the search takes: each of its reads is
//!   replaced by its model value, under the same congruence check, and the
//!   array keeps the candidate's interpretation.
//! * An `exists` at positive polarity is certified by a **witness**: its body,
//!   with the bound variables replaced by one of the goal's points (the
//!   pinned search's points plus one representative they do not name), must
//!   be valid under the completion — `ψ(w)` entails `∃x. ψ(x)`.  An `exists`
//!   anywhere else still declines.
//!
//! Nothing here weakens the certificate's argument: a replaced term is a
//! symbol of the interpretation fixed at a value, which the validity query
//! then reads as that value; a witness is an instance, and an instance of a
//! valid body is valid.  What it changes is only which goals get a
//! certificate at all.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::smtlib::reserved_name;
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{Combinations, Goal, is_const_array, is_literal, run_query};
use crate::solver::SolverResult;

/// The head of a point of a function-like symbol: an uninterpreted function
/// or an array variable read at ground indices.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Head {
    Function(Spur),
    Array(TermId),
}

/// Witness tuples one `exists` may try with a query (a product over its
/// bound variables).
const MAX_WITNESSES: usize = 8;

/// Witness tuples one `exists` may try by evaluation alone: an interpretation
/// the certificate checks names its points, and each is one exact evaluation.
const MAX_EVALUATED_WITNESSES: usize = 64;

/// Whether some assertion carries an application of an uninterpreted
/// function (the array constant `((as const A) d)` is interpreted and does
/// not count).
pub(super) fn has_uninterpreted_application(assertions: &[TermId], manager: &TermManager) -> bool {
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = assertions.to_vec();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if matches!(data.kind, TermKind::Apply { .. }) && !is_const_array(current, manager) {
            return true;
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    false
}

/// What the walk of [`interpret_from_candidate`] found.
#[derive(Default)]
struct Occurrences {
    /// Ground uninterpreted applications, in first-encounter order.
    applications: Vec<TermId>,
    /// Every ground read `(select a k)` of each array variable.
    reads: FxHashMap<TermId, Vec<TermId>>,
    /// Array variables with an occurrence other than a ground read.
    must_complete: FxHashSet<TermId>,
}

/// Split `arrays` into the ones to complete and the ground terms to fix at
/// their candidate values; `None` when the goal cannot be interpreted that
/// way (a non-ground application, a `let` / `match`, a value the candidate
/// model does not give as a literal, two readings of one function or array
/// that disagree).  `fix_arrays` says whether read-only arrays are fixed too
/// (only when the goal has more arrays than the search takes).
pub(super) fn interpret_from_candidate(
    assertions: &[TermId],
    arrays: &[TermId],
    assignments: &FxHashMap<TermId, TermId>,
    scalar_pins: &FxHashMap<TermId, TermId>,
    fix_arrays: bool,
    manager: &TermManager,
) -> Option<(Vec<TermId>, FxHashMap<TermId, TermId>)> {
    let array_set: FxHashSet<TermId> = arrays.iter().copied().collect();
    let found = walk_occurrences(assertions, &array_set, manager)?;

    let mut ground_values: FxHashMap<TermId, TermId> = FxHashMap::default();
    // (head, evaluated arguments) -> value: one value per point of a function.
    let mut table: FxHashMap<(Head, Vec<TermId>), TermId> = FxHashMap::default();
    for &application in &found.applications {
        let TermKind::Apply { func, args } = &manager.get(application)?.kind else {
            return None;
        };
        let head = Head::Function(*func);
        let key: Vec<TermId> = args
            .iter()
            .map(|&arg| literal_value(arg, scalar_pins, manager))
            .collect::<Option<_>>()?;
        let value = assignments
            .get(&application)
            .copied()
            .filter(|&value| is_literal(value, manager))?;
        if *table.entry((head, key)).or_insert(value) != value {
            return None;
        }
        ground_values.insert(application, value);
    }

    let mut completed: Vec<TermId> = Vec::with_capacity(arrays.len());
    for &array in arrays {
        let fixed = fix_arrays
            && !found.must_complete.contains(&array)
            && fix_array_reads(
                array,
                found.reads.get(&array).map_or(&[][..], Vec::as_slice),
                assignments,
                scalar_pins,
                &mut table,
                &mut ground_values,
                manager,
            );
        if !fixed {
            completed.push(array);
        }
    }
    Some((completed, ground_values))
}

/// Record the candidate value of every read of `array` in `ground_values`,
/// or leave everything untouched and answer `false` when one read cannot be
/// fixed (its index or value is not a literal, or two reads disagree).
fn fix_array_reads(
    array: TermId,
    reads: &[TermId],
    assignments: &FxHashMap<TermId, TermId>,
    scalar_pins: &FxHashMap<TermId, TermId>,
    table: &mut FxHashMap<(Head, Vec<TermId>), TermId>,
    ground_values: &mut FxHashMap<TermId, TermId>,
    manager: &TermManager,
) -> bool {
    let mut staged: Vec<((Head, Vec<TermId>), TermId, TermId)> = Vec::with_capacity(reads.len());
    for &read in reads {
        let Some(TermKind::Select(_, index)) = manager.get(read).map(|d| &d.kind) else {
            return false;
        };
        let Some(index) = literal_value(*index, scalar_pins, manager) else {
            return false;
        };
        let Some(value) = assignments
            .get(&read)
            .copied()
            .filter(|&value| is_literal(value, manager))
        else {
            return false;
        };
        staged.push(((Head::Array(array), vec![index]), read, value));
    }
    let mut local: FxHashMap<(Head, Vec<TermId>), TermId> = FxHashMap::default();
    for (key, _, value) in &staged {
        let existing = table.get(key).or_else(|| local.get(key)).copied();
        if existing.is_some_and(|previous| previous != *value) {
            return false;
        }
        local.insert(key.clone(), *value);
    }
    for (key, read, value) in staged {
        table.insert(key, value);
        ground_values.insert(read, value);
    }
    true
}

/// The literal `term` denotes under the scalar pins: itself if a literal, its
/// pin if a pinned scalar.
fn literal_value(
    term: TermId,
    scalar_pins: &FxHashMap<TermId, TermId>,
    manager: &TermManager,
) -> Option<TermId> {
    if is_literal(term, manager) {
        return Some(term);
    }
    scalar_pins
        .get(&term)
        .copied()
        .filter(|&value| is_literal(value, manager))
}

/// Walk the assertions with the binder scope in hand: every ground
/// uninterpreted application, every ground read of an array variable, and
/// every array variable met anywhere else.  `None` on a non-ground
/// application or on a `let` / `match` (whose scope this walk does not
/// model).
fn walk_occurrences(
    assertions: &[TermId],
    arrays: &FxHashSet<TermId>,
    manager: &TermManager,
) -> Option<Occurrences> {
    let mut found = Occurrences::default();
    // Scope 0 binds nothing; each binder opens the union of its parent's
    // names and its own.
    let mut scopes: Vec<FxHashSet<Spur>> = vec![FxHashSet::default()];
    let mut stack: Vec<(TermId, usize)> = assertions.iter().map(|&a| (a, 0)).collect();
    let mut visited: FxHashSet<(TermId, usize)> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    let mut seen_applications: FxHashSet<TermId> = FxHashSet::default();
    let mut seen_reads: FxHashSet<TermId> = FxHashSet::default();
    while let Some((current, scope)) = stack.pop() {
        if !visited.insert((current, scope)) {
            continue;
        }
        let data = manager.get(current)?;
        match &data.kind {
            TermKind::Forall { vars, body, .. } | TermKind::Exists { vars, body, .. } => {
                let mut names = scopes.get(scope)?.clone();
                names.extend(vars.iter().map(|&(name, _)| name));
                scopes.push(names);
                stack.push((*body, scopes.len() - 1));
            }
            TermKind::Let { .. } | TermKind::Match { .. } => return None,
            TermKind::Apply { .. } if !is_const_array(current, manager) => {
                if mentions_scope(current, scopes.get(scope)?, manager) {
                    return None;
                }
                if seen_applications.insert(current) {
                    found.applications.push(current);
                }
            }
            TermKind::Select(array, index) if arrays.contains(array) => {
                if mentions_scope(*index, scopes.get(scope)?, manager) {
                    found.must_complete.insert(*array);
                } else if seen_reads.insert(current) {
                    found.reads.entry(*array).or_default().push(current);
                }
                stack.push((*index, scope));
            }
            kind => {
                children.clear();
                children.extend(oxiz_core::ast::traversal::get_children(kind));
                for &child in &children {
                    if arrays.contains(&child) {
                        found.must_complete.insert(child);
                    } else {
                        stack.push((child, scope));
                    }
                }
            }
        }
    }
    found.applications.sort_unstable_by_key(|term| term.raw());
    for reads in found.reads.values_mut() {
        reads.sort_unstable_by_key(|term| term.raw());
    }
    Some(found)
}

/// Whether `term` mentions a variable named in `scope`.
fn mentions_scope(term: TermId, scope: &FxHashSet<Spur>, manager: &TermManager) -> bool {
    if scope.is_empty() {
        return false;
    }
    manager
        .free_vars_including_patterns(term)
        .iter()
        .any(|&var| matches!(manager.get(var).map(|d| &d.kind), Some(TermKind::Var(name)) if scope.contains(name)))
}

impl Goal {
    /// Certify the positive-polarity `exists` `existential` true under
    /// `substitution` (the completion, the pins, the fixed ground values) by
    /// a witness among the goal's points.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn certify_existential(
        &self,
        solver: &crate::solver::Solver,
        existential: TermId,
        position: usize,
        substitution: &FxHashMap<TermId, TermId>,
        extra_points: &FxHashMap<SortId, Vec<TermId>>,
        manager: &mut TermManager,
        logic: Option<&str>,
        queries: &mut usize,
    ) -> bool {
        // The points of the arrays this `exists` reads come first: a
        // candidate's Skolem witness sits at one of them.
        let mentioned: FxHashSet<TermId> = manager
            .free_vars_including_patterns(existential)
            .into_iter()
            .collect();
        let mut extra: FxHashMap<SortId, Vec<TermId>> = FxHashMap::default();
        for (array, value) in substitution {
            if !mentioned.contains(array) {
                continue;
            }
            for (sort, points) in
                super::completion_points(&core::iter::once((*array, *value)).collect(), manager)
            {
                let slot = extra.entry(sort).or_default();
                for point in points {
                    if !slot.contains(&point) {
                        slot.push(point);
                    }
                }
            }
        }
        for (&sort, points) in extra_points {
            let slot = extra.entry(sort).or_default();
            for &point in points {
                if !slot.contains(&point) {
                    slot.push(point);
                }
            }
        }
        for list in extra.values_mut() {
            list.sort_unstable_by_key(|term| term.raw());
        }
        let Some(witnesses) = self.existential_witnesses_capped(
            existential,
            position,
            &extra,
            MAX_EVALUATED_WITNESSES,
            manager,
        ) else {
            return false;
        };
        // Every witness is evaluated (cheap); only the first
        // `MAX_WITNESSES` of those the evaluation cannot close cost a query.
        let mut open: Vec<TermId> = Vec::new();
        for instance in witnesses {
            let instance = manager.substitute(instance, substitution);
            match super::evaluate_closed(solver, instance, manager) {
                Some(true) => return true,
                Some(false) => {}
                None => {
                    if open.len() < MAX_WITNESSES {
                        open.push(instance);
                    }
                }
            }
        }
        for instance in open {
            let negated = manager.mk_not(instance);
            let negated = self.under_hypotheses(negated, manager);
            match run_query(negated, manager, logic, queries) {
                Some(SolverResult::Unsat) => return true,
                Some(_) => {}
                None => return false,
            }
        }
        false
    }

    /// The instances of `existential`'s body at the witness tuples (the
    /// goal's points plus one unnamed representative per bound variable's
    /// sort), each still over the goal's free symbols; `None` when the
    /// `exists` is not one this module handles (an alternation under it, a
    /// sort with no point).  Shared by the certificate and by the sample the
    /// evaluation pre-filter and the fill query read, so all three look at
    /// the same witnesses.
    pub(super) fn existential_witnesses(
        &self,
        existential: TermId,
        position: usize,
        manager: &mut TermManager,
    ) -> Option<Vec<TermId>> {
        self.existential_witnesses_with(existential, position, &FxHashMap::default(), manager)
    }

    /// [`Self::existential_witnesses`] with `extra` points tried as well —
    /// the store indices of the interpretation being certified, which is
    /// where a candidate's Skolem witness sits (decision (54)(ii)).
    pub(super) fn existential_witnesses_with(
        &self,
        existential: TermId,
        position: usize,
        extra: &FxHashMap<SortId, Vec<TermId>>,
        manager: &mut TermManager,
    ) -> Option<Vec<TermId>> {
        self.existential_witnesses_capped(existential, position, extra, MAX_WITNESSES, manager)
    }

    /// [`Self::existential_witnesses_with`] with at most `cap` witness tuples.
    fn existential_witnesses_capped(
        &self,
        existential: TermId,
        position: usize,
        extra: &FxHashMap<SortId, Vec<TermId>>,
        cap: usize,
        manager: &mut TermManager,
    ) -> Option<Vec<TermId>> {
        let (body, vars) = peel_existential_with_vars(existential, position, manager)?;
        let points = self.points_by_sort(manager);
        let mut lists: Vec<Vec<TermId>> = Vec::with_capacity(vars.len());
        for &var in &vars {
            let sort = manager.get(var)?.sort;
            // The interpretation's own points first: the witness cap must not
            // cut off the one point a Skolem witness is printed at.
            let mut list: Vec<TermId> = extra.get(&sort).cloned().unwrap_or_default();
            for &point in points.get(&sort).map_or(&[][..], Vec::as_slice) {
                if !list.contains(&point) {
                    list.push(point);
                }
            }
            if let Some(representative) = super::pinned::unnamed_point(sort, &list, manager) {
                list.push(representative);
            }
            if list.is_empty() {
                return None;
            }
            lists.push(list);
        }
        let mut instances: Vec<TermId> = Vec::new();
        for combination in Combinations::with_cap(&lists, cap) {
            let map: FxHashMap<TermId, TermId> = vars
                .iter()
                .copied()
                .zip(combination.iter().copied())
                .collect();
            instances.push(manager.substitute(body, &map));
        }
        Some(instances)
    }
}

/// The body of a chain of consecutive `exists`s, with every bound variable
/// replaced by a fresh reserved constant, and the constants in binder order;
/// `None` when a quantifier survives the peel.
fn peel_existential_with_vars(
    term: TermId,
    position: usize,
    manager: &mut TermManager,
) -> Option<(TermId, Vec<TermId>)> {
    let mut current = term;
    let mut bound: Vec<(Spur, SortId)> = Vec::new();
    while let Some(TermKind::Exists { vars, body, .. }) =
        manager.get(current).map(|d| d.kind.clone())
    {
        bound.extend(vars.iter().copied());
        current = body;
    }
    if super::contains_quantifier(current, manager) {
        return None;
    }
    let mut rename: FxHashMap<TermId, TermId> = FxHashMap::default();
    let mut fresh_vars: Vec<TermId> = Vec::with_capacity(bound.len());
    for (index, (name, sort)) in bound.into_iter().enumerate() {
        let bound_name = manager.resolve_str(name).to_string();
        let bound_var = manager.mk_var(&bound_name, sort);
        let fresh = manager.mk_var(
            &reserved_name("qwitx", &format!("{position}_{index}")),
            sort,
        );
        match rename.insert(bound_var, fresh) {
            None => fresh_vars.push(fresh),
            Some(previous) => {
                if let Some(slot) = fresh_vars.iter_mut().find(|var| **var == previous) {
                    *slot = fresh;
                }
            }
        }
    }
    Some((manager.substitute(current, &rename), fresh_vars))
}
