//! The open regions a `Real` guard cuts, bracketed by terms (`#P2b-75`).
//!
//! Over `Int` and bit-vectors `t ± 1` beside every guard ground `t` is enough
//! for the projection `π(v) = max{s ∈ S : s ≤ v}` to keep every value in the
//! region of the guards it was in: a value past a boundary `g` is at least
//! `g + 1`, which is in the relevant set.  Over `Real` it is not: between
//! `1.0 < x < 2.0` lies no `t ± 1`, and `(forall ((q Real)) (=> (and (> q 1.0)
//! (< q 2.0)) false))` was certified `sat`.  Every open region between two
//! guard values holds the midpoint of the two, so the midpoint of every pair
//! of guard grounds (as a term, `(* 1/2 (+ a b))`, so it stays in its region
//! however the model moves `a` and `b`) plus `t ± 1` gives every region a
//! point, and the projection sends every value outside the relevant set to
//! the point of its region.
//!
//! That projection is not monotone once the relevant set also holds points
//! inside a region (a function's arguments), so a goal that orders two `Real`
//! bound variables (`x ≤ y`, which the fragment admits for its monotone
//! projection) keeps the monotone one, `π(v) = max{s ∈ S : s ≤ v}`, which
//! preserves every non-strict bound, equality and strict upper bound — and is
//! declined (the caller keeps its ordinary path, never a `sat`) only when one
//! of its `Real` guards holds on an interval open at its lower end (`x > g`,
//! `x ≠ g`), the one shape neither projection preserves for it.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;

use super::super::QuantifiedFormula;
use crate::prelude::*;

/// Guard grounds of one `Real` sort beyond which the pairwise midpoints are
/// not enumerated (the goal declines instead).
const MAX_REAL_GUARD_GROUNDS: usize = 12;

/// Add the midpoint of every pair of `Real` guard grounds to `relevant`, per
/// sort; `false` when the goal must decline (too many grounds, or an order
/// between two `Real` bound variables beside a guard open at its lower end).
pub(super) fn add_real_midpoints(
    quantifiers: &[QuantifiedFormula],
    relevant: &mut FxHashMap<SortId, Vec<TermId>>,
    manager: &mut TermManager,
) -> bool {
    let real = manager.sorts.real_sort;
    let mut grounds: Vec<TermId> = Vec::new();
    let mut ordered_pair = false;
    let mut open_lower = false;
    for quantifier in quantifiers {
        if !quantifier.is_universal || !quantifier.can_instantiate() {
            continue;
        }
        let vars: FxHashSet<Spur> = quantifier.bound_vars.iter().map(|(n, _)| *n).collect();
        let reals: FxHashSet<Spur> = quantifier
            .bound_vars
            .iter()
            .filter(|&&(_, sort)| sort == real)
            .map(|(n, _)| *n)
            .collect();
        if reals.is_empty() {
            continue;
        }
        let body = super::peel_ground_premises(quantifier.body, &vars, manager);
        let guard = match manager.get(body).map(|t| t.kind.clone()) {
            Some(TermKind::Implies(g, _)) => g,
            _ => body,
        };
        let mut found: Vec<TermId> = Vec::new();
        super::collect_guard_ground_terms(guard, &reals, manager, &mut found);
        grounds.extend(found);
        ordered_pair |= orders_two_bound_variables(guard, &reals, manager);
        open_lower |= holds_on_an_open_lower_interval(guard, &reals, manager);
    }
    grounds.sort_unstable_by_key(|term| term.raw());
    grounds.dedup();
    if grounds.is_empty() {
        return true;
    }
    if (ordered_pair && open_lower) || grounds.len() > MAX_REAL_GUARD_GROUNDS {
        return false;
    }
    let half = manager.mk_real(num_rational::Rational64::new(1, 2));
    let mut points: Vec<TermId> = Vec::new();
    for (i, &a) in grounds.iter().enumerate() {
        for &b in &grounds[i + 1..] {
            let point = match (real_literal(a, manager), real_literal(b, manager)) {
                (Some(x), Some(y)) => match num_traits::CheckedAdd::checked_add(&x, &y) {
                    Some(sum) => manager.mk_real(sum * num_rational::Rational64::new(1, 2)),
                    None => continue,
                },
                _ => {
                    let sum = manager.mk_add([a, b]);
                    manager.mk_mul([half, sum])
                }
            };
            points.push(point);
        }
    }
    let bucket = relevant.entry(real).or_default();
    for point in points {
        if !bucket.contains(&point) {
            bucket.push(point);
        }
    }
    true
}

/// The polarity a sub-term of a guard occurs at.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Polarity {
    Positive,
    Negative,
    Both,
}

/// Whether some comparison of a bound variable of `vars` with a ground term
/// holds, at the polarity it occurs at in `guard`, on an interval OPEN at its
/// lower end (`x > g`, `x ≠ g`), or at a polarity the walk cannot pin down.
///
/// Those are the only guard atoms the monotone projection `π(v) = max{s ∈ S :
/// s ≤ v}` does not preserve over `Real` (no finite set has a point in
/// `(g, v]` for every `v > g`): a non-strict bound, an equality and a strict
/// upper bound are preserved by it — `π(v) ≤ v`, and every guard ground and
/// `g - 1` is in the relevant set — so a goal ordering two `Real` bound
/// variables keeps the monotone projection, and is certifiable, unless one of
/// these occurs (`03_monotonicity`: `0 ≤ x ≤ 100 ∧ 0 ≤ y ≤ 100 ∧ x ≤ y`).
fn holds_on_an_open_lower_interval(
    guard: TermId,
    vars: &FxHashSet<Spur>,
    manager: &TermManager,
) -> bool {
    let is_var = |term: TermId| matches!(manager.get(term).map(|t| &t.kind), Some(TermKind::Var(n)) if vars.contains(n));
    let ground = |term: TermId| {
        !manager
            .free_vars_including_patterns(term)
            .iter()
            .any(|&v| is_var(v))
    };
    let mut stack: Vec<(TermId, Polarity)> = vec![(guard, Polarity::Positive)];
    let mut visited: FxHashSet<(TermId, Polarity)> = FxHashSet::default();
    while let Some((current, polarity)) = stack.pop() {
        if !visited.insert((current, polarity)) {
            continue;
        }
        let Some(node) = manager.get(current) else {
            continue;
        };
        let flip = match polarity {
            Polarity::Positive => Polarity::Negative,
            Polarity::Negative => Polarity::Positive,
            Polarity::Both => Polarity::Both,
        };
        // `(var, relation, ground)` read left to right: `Lt` / `Le` / `Gt` /
        // `Ge` / `Eq` of the variable against the ground side.
        let compared = match &node.kind {
            TermKind::Lt(l, r) => Some((*l, *r, "<")),
            TermKind::Le(l, r) => Some((*l, *r, "<=")),
            TermKind::Gt(l, r) => Some((*l, *r, ">")),
            TermKind::Ge(l, r) => Some((*l, *r, ">=")),
            TermKind::Eq(l, r) => Some((*l, *r, "=")),
            _ => None,
        };
        if let Some((l, r, op)) = compared {
            let oriented = if is_var(l) && ground(r) {
                Some(op)
            } else if is_var(r) && ground(l) {
                Some(match op {
                    "<" => ">",
                    "<=" => ">=",
                    ">" => "<",
                    ">=" => "<=",
                    other => other,
                })
            } else {
                None
            };
            if let Some(op) = oriented {
                let effective = match polarity {
                    Polarity::Both => return true,
                    Polarity::Positive => op,
                    Polarity::Negative => match op {
                        "<" => ">=",
                        "<=" => ">",
                        ">" => "<=",
                        ">=" => "<",
                        _ => "!=",
                    },
                };
                if matches!(effective, ">" | "!=") {
                    return true;
                }
                continue;
            }
        }
        match &node.kind {
            TermKind::Not(a) => stack.push((*a, flip)),
            TermKind::And(args) | TermKind::Or(args) => {
                stack.extend(args.iter().map(|&a| (a, polarity)));
            }
            TermKind::Implies(l, r) => {
                stack.push((*l, flip));
                stack.push((*r, polarity));
            }
            TermKind::Ite(c, t, e) => {
                stack.push((*c, Polarity::Both));
                stack.push((*t, polarity));
                stack.push((*e, polarity));
            }
            kind => stack.extend(
                oxiz_core::ast::traversal::get_children(kind)
                    .into_iter()
                    .map(|child| (child, Polarity::Both)),
            ),
        }
    }
    false
}

/// Whether `guard` compares two of `vars` with an order (`≤ < ≥ >`), at any
/// depth.
fn orders_two_bound_variables(
    guard: TermId,
    vars: &FxHashSet<Spur>,
    manager: &TermManager,
) -> bool {
    let is_var = |term: TermId| matches!(manager.get(term).map(|t| &t.kind), Some(TermKind::Var(n)) if vars.contains(n));
    let mut stack: Vec<TermId> = vec![guard];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(node) = manager.get(current) else {
            continue;
        };
        if let TermKind::Le(l, r) | TermKind::Lt(l, r) | TermKind::Ge(l, r) | TermKind::Gt(l, r) =
            &node.kind
            && is_var(*l)
            && is_var(*r)
        {
            return true;
        }
        stack.extend(oxiz_core::ast::traversal::get_children(&node.kind));
    }
    false
}

/// The value of a `Real` (or integer) literal.
fn real_literal(term: TermId, manager: &TermManager) -> Option<num_rational::Rational64> {
    match &manager.get(term)?.kind {
        TermKind::RealConst(value) => Some(*value),
        TermKind::IntConst(value) => i64::try_from(value.clone())
            .ok()
            .map(num_rational::Rational64::from_integer),
        _ => None,
    }
}
