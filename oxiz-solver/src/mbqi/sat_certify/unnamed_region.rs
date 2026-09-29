//! Completing the relevant set of an **array index** position (`#P2b-60`).
//!
//! # The wrong `sat` this closes
//!
//! ```text
//! (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))
//! (assert (= a1 (store ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0) #b0000000 #b1)))
//! (assert (forall ((i (_ BitVec 7))) (= (select a1 i) #b1)))
//! ```
//!
//! is unsatisfiable (`a1` reads `#b0` at `#b0000001`), and it was answered
//! `sat` (`c4b04b7` and crates.io 0.3.3 answer `unknown`).  The universal is
//! essentially uninterpreted — its bound variable occurs only as an array
//! index — so [`super::collect_fragment_instances`] instantiated it over the
//! relevant set of Ge & de Moura: the index terms the model already reads,
//! here `#b0000000` alone.  One instance, satisfied; the set is saturated; the
//! certifier concluded `sat`.  Adding a single unrelated ground read at
//! `#b0000001` anywhere in the script turned the answer into the correct
//! `unsat`, which is what located the defect.  The same shape over an `Int`,
//! a `Real` and a declared index sort was a wrong `sat` too — the `Int` and
//! declared-sort spellings on `c4b04b7` and crates.io 0.3.3 as well.
//!
//! # Why the relevant set is not complete for arrays
//!
//! The completeness argument behind the relevant set extends a model `M` of
//! the finite instance set to the whole domain by a *projection* `π` that
//! sends every index to one the instance set covers, and reads every
//! uninterpreted function through it.  For that `M'` to still satisfy the
//! ground part, every value the ground part *pins* must be a fixed point of
//! `π`, and `π` must send an index to one the ground part treats alike.  An
//! uninterpreted function is pinned only where a term names it, and the terms
//! the model reads are exactly those.  An array is not such a function:
//! `store`, the array constant `((as const A) d)` and extensionality pin its
//! value at indices the model need not read (a `store` index nobody reads
//! back), and at indices **no** term names at all (every point of an array
//! constant's default).  If `π` sends an unnamed index `j` to a named `k`, `M'`
//! reads `store(b, k, v)` at `j` as `v` where array semantics demands `b[j]`,
//! so the "model" the certificate vouches for is not a model.
//!
//! # The repair
//!
//! For every sort a bound variable indexes an array at, the relevant set is
//! completed with:
//!
//! * **every named index value** — each literal of the sort in an index
//!   position anywhere in the goal (a `select` / `store` index) or compared
//!   with a bound variable of the sort in a quantifier body (a guard constant),
//!   and the model's value of each non-literal term in either position — so
//!   every value the ground part can pin is instantiated, hence a fixed point;
//! * **representatives of the unnamed region** — over a sort a quantifier does
//!   not order, one value no term names (the smallest, over bit-vectors, `Int`
//!   and `Real`); over an *ordered* one, one per gap of every order a guard can
//!   use: `r ± 1` beside every named `r` over `Int` and bit-vectors (mod `2^w`,
//!   plus the four ends of the unsigned and signed orders, so a signed guard's
//!   wrap-around gap is covered too), and the midpoints of consecutive named
//!   values plus `min - 1` and `max + 1` over `Real`;
//! * over a **declared** sort, every ground term of the sort the goal spells
//!   outside a binder: a model of such a sort needs no element beyond them, so
//!   there is no unnamed region to represent, only the named elements to cover;
//! * over `Bool`, both values, and over an enumeration datatype (every
//!   constructor nullary), every constructor — exhaustive;
//! * over a declared sort, also every constant of the sort the candidate model
//!   assigns — the Skolem witness of an asserted `exists` appears only in the
//!   encoded assertion, never in the goal's spelling (`#P2b-64` (2));
//! * over a datatype with a field, or any other index sort (floating point,
//!   arrays, strings), **nothing is complete**, so the caller declines to
//!   certify (re-fix pass 13, `#P2b-64` (1)): no single value is known to be
//!   unnamed there, and until then the certifier concluded `sat` over the
//!   spelled terms alone.
//!
//! `π` then sends every unnamed index to an unnamed representative in the same
//! gap of every order, where every array term reads what it reads at every
//! other unnamed index of that gap, and fixes every named index: `M'` is a
//! model.
//!
//! Every point added is an **instance of a universal**, which is always a
//! sound consequence: the change can only make an unsatisfiable goal show its
//! conflict or cost a certification its cap, never produce a `sat`.  The points
//! are chosen from the goal's own index terms rather than from every literal
//! the model reads, because the instance at a representative makes the model
//! read it: a choice keyed on the model's reads would move the representative
//! every round and the set would never saturate.  They are added only at the
//! saturation re-check (`MBQIIntegration::certify_or_refine`).

use num_bigint::BigInt;
use num_rational::Rational64;
use num_traits::{CheckedAdd, CheckedDiv, CheckedSub};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::{SortId, SortKind};

use super::super::QuantifiedFormula;
use super::super::model_completion::CompletedModel;
use crate::prelude::*;

/// What the points are chosen from, per sort a bound variable indexes an
/// array at.
#[derive(Default)]
struct Named {
    /// Literal terms naming an index value.
    literals: Vec<TermId>,
    /// Non-literal terms in an index or guard position (their model values
    /// name index values).
    terms: Vec<TermId>,
    /// Whether some quantifier orders a bound variable of the sort.
    ordered: bool,
}

/// Add the points of the module docs to `relevant` for every sort a bound
/// variable indexes an array at.  `goal` is the assertions in scope.
///
/// `false` when some such sort has no complete set of points (a datatype
/// with a field): the saturation test must then not conclude anything.
pub(super) fn add_unnamed_index_points(
    quantifiers: &[QuantifiedFormula],
    model: &CompletedModel,
    goal: &[TermId],
    relevant: &mut FxHashMap<SortId, Vec<TermId>>,
    manager: &mut TermManager,
) -> bool {
    let (order, mut named) = index_sorts(quantifiers, manager);
    if order.is_empty() {
        return true;
    }
    let bound_names: FxHashSet<Spur> = quantifiers
        .iter()
        .flat_map(|q| q.bound_vars.iter().map(|(name, _)| *name))
        .collect();
    collect_named_indices(goal, &bound_names, &mut named, manager);
    // The relevant set's own non-literal index terms name values too (a
    // declared index constant the model reads, an extensionality witness).
    for (&sort, entry) in named.iter_mut() {
        if let Some(terms) = relevant.get(&sort) {
            entry
                .terms
                .extend(terms.iter().copied().filter(|&t| !is_literal(t, manager)));
        }
    }
    for sort in order {
        let Some(kind) = manager.sorts.get(sort).map(|s| s.kind.clone()) else {
            continue;
        };
        let Some(entry) = named.get(&sort) else {
            continue;
        };
        let mut values: Vec<TermId> = entry.literals.clone();
        for &term in &entry.terms {
            if let Some(value) = model.eval(term).filter(|&v| is_literal(v, manager)) {
                values.push(value);
            }
        }
        values.sort_unstable_by_key(|term| term.raw());
        values.dedup();
        let mut points: Vec<TermId> = values.clone();
        match kind {
            SortKind::Bool => points.extend([manager.mk_false(), manager.mk_true()]),
            SortKind::BitVec(width) => {
                let taken: FxHashSet<BigInt> = values
                    .iter()
                    .filter_map(|&v| int_value(v, manager))
                    .collect();
                for value in bv_representatives(width, &taken, entry.ordered) {
                    points.push(manager.mk_bitvec(value, width));
                }
            }
            SortKind::Int => {
                let taken: FxHashSet<BigInt> = values
                    .iter()
                    .filter_map(|&v| int_value(v, manager))
                    .collect();
                for value in int_representatives(&taken, entry.ordered) {
                    points.push(manager.mk_int(value));
                }
            }
            SortKind::Real => {
                let taken: Vec<Rational64> = values
                    .iter()
                    .filter_map(|&v| real_value(v, manager))
                    .collect();
                for value in real_representatives(&taken, entry.ordered) {
                    points.push(manager.mk_real(value));
                }
            }
            SortKind::Uninterpreted(_) => {
                points.extend(ground_terms_of_sort(goal, sort, manager));
                // A constant of the sort the goal never spells outside a
                // binder is still an element the model must cover: the
                // Skolem witness of an asserted `exists` lives only in the
                // *encoded* assertion (`encode::exists_skolem`), and a
                // binder-sort witness only in the candidate pool.  Missing
                // them made `∃x. x ≠ u` beside `∀x. a[x] = 1` over
                // `a = store(K0, u, 1)` saturate at `{u}` alone (`#P2b-64`
                // (2)).  Constants only — a model key that is an application
                // or a read is an instance term, and choosing points by
                // those would move them every round.
                points.extend(model_constants_of_sort(model, sort, &bound_names, manager));
            }
            SortKind::Datatype(_) => {
                // An enumeration (every constructor nullary) is its own
                // finite domain: every constructor is a point.  A datatype
                // with a field has no finite domain to enumerate, and no
                // single representative of the values no term names is
                // known to be unnamed (a field value can coincide with a
                // named term's), so the saturation cannot be certified
                // there at all: the caller is told to decline (`#P2b-64`
                // (1): `∀x:L. a[x] = 1` beside `a = store(K0, nil, 1)` was
                // certified over `{nil}` and answered `sat`).
                match enumeration_constructors(sort, manager) {
                    Some(constructors) => points.extend(constructors),
                    None => return false,
                }
            }
            // No representative is known for any other index sort, so no
            // saturation over it is a certificate.
            _ => return false,
        }
        let bucket = relevant.entry(sort).or_default();
        for point in points {
            if !bucket.contains(&point) {
                bucket.push(point);
            }
        }
    }
    true
}

/// The sorts a bound variable indexes an array at, in first-encounter order,
/// with the guard literals / terms of each and whether a quantifier orders a
/// bound variable of the sort.
fn index_sorts(
    quantifiers: &[QuantifiedFormula],
    manager: &TermManager,
) -> (Vec<SortId>, FxHashMap<SortId, Named>) {
    let mut order: Vec<SortId> = Vec::new();
    let mut named: FxHashMap<SortId, Named> = FxHashMap::default();
    for quantifier in quantifiers {
        if !quantifier.is_universal || !quantifier.can_instantiate() {
            continue;
        }
        let names: FxHashMap<Spur, SortId> = quantifier.bound_vars.iter().copied().collect();
        let bound_sort = |term: TermId| -> Option<SortId> {
            match manager.get(term).map(|t| &t.kind) {
                Some(TermKind::Var(name)) => names.get(name).copied(),
                _ => None,
            }
        };
        let mentions_bound = |term: TermId| -> bool {
            manager.free_vars_including_patterns(term).iter().any(|&v| {
                matches!(manager.get(v).map(|t| &t.kind), Some(TermKind::Var(n)) if names.contains_key(n))
            })
        };
        let mut visited: FxHashSet<TermId> = FxHashSet::default();
        let mut stack: Vec<TermId> = vec![quantifier.body];
        let mut children: Vec<TermId> = Vec::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(data) = manager.get(current) else {
                continue;
            };
            match &data.kind {
                TermKind::Select(_, index) | TermKind::Store(_, index, _) => {
                    if let Some(sort) = bound_sort(*index)
                        && !order.contains(&sort)
                    {
                        order.push(sort);
                    }
                }
                TermKind::Eq(l, r)
                | TermKind::Le(l, r)
                | TermKind::Lt(l, r)
                | TermKind::Ge(l, r)
                | TermKind::Gt(l, r)
                | TermKind::BvUlt(l, r)
                | TermKind::BvUle(l, r)
                | TermKind::BvSlt(l, r)
                | TermKind::BvSle(l, r) => {
                    let is_order = !matches!(data.kind, TermKind::Eq(..));
                    for (side, other) in [(*l, *r), (*r, *l)] {
                        let Some(sort) = bound_sort(side) else {
                            continue;
                        };
                        let entry = named.entry(sort).or_default();
                        entry.ordered |= is_order;
                        if !mentions_bound(other) {
                            if is_literal(other, manager) {
                                entry.literals.push(other);
                            } else {
                                entry.terms.push(other);
                            }
                        }
                    }
                }
                _ => {}
            }
            children.clear();
            children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
            stack.extend(children.iter().copied());
        }
    }
    named.retain(|sort, _| order.contains(sort));
    for &sort in &order {
        named.entry(sort).or_default();
    }
    (order, named)
}

/// Every term in a `select` / `store` index position of the goal (bodies
/// included) whose sort is one of `named`'s and which mentions no variable a
/// tracked quantifier binds: literals go to `literals`, the rest to `terms`.
///
/// A declared constant that happens to share a name with a bound variable is
/// read as bound and left out — that costs this module one named value, and
/// the relevant set (which the caller folds in) still carries it wherever the
/// model reads it.
fn collect_named_indices(
    goal: &[TermId],
    bound_names: &FxHashSet<Spur>,
    named: &mut FxHashMap<SortId, Named>,
    manager: &TermManager,
) {
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = goal.to_vec();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if let TermKind::Select(_, index) | TermKind::Store(_, index, _) = &data.kind
            && let Some(sort) = manager.get(*index).map(|t| t.sort)
            && let Some(entry) = named.get_mut(&sort)
        {
            if is_literal(*index, manager) {
                entry.literals.push(*index);
            } else if !manager.free_vars_including_patterns(*index).iter().any(|&v| {
                matches!(manager.get(v).map(|t| &t.kind), Some(TermKind::Var(n)) if bound_names.contains(n))
            }) {
                entry.terms.push(*index);
            }
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
}

/// Bit-vector representatives: unordered, the smallest unnamed value; ordered,
/// `r ± 1 (mod 2^w)` beside every named `r` and the four ends of the unsigned
/// and signed orders, each only if unnamed.
fn bv_representatives(width: u32, taken: &FxHashSet<BigInt>, ordered: bool) -> Vec<BigInt> {
    let modulus = BigInt::from(1u8) << width;
    let mut out: Vec<BigInt> = Vec::new();
    if !ordered {
        let mut value = BigInt::from(0u8);
        while value < modulus {
            if !taken.contains(&value) {
                out.push(value);
                break;
            }
            value += 1u8;
        }
        return out;
    }
    let half = BigInt::from(1u8) << width.saturating_sub(1);
    let mut candidates: Vec<BigInt> =
        vec![BigInt::from(0u8), &modulus - 1u8, half.clone(), &half - 1u8];
    for r in taken {
        candidates.push((r + 1u8 + &modulus) % &modulus);
        candidates.push((r - 1u8 + &modulus) % &modulus);
    }
    for value in candidates {
        if value >= BigInt::from(0u8) && value < modulus && !taken.contains(&value) {
            out.push(value);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `Int` representatives: unordered, the smallest non-negative unnamed value;
/// ordered, `r ± 1` beside every named `r`, or `0` when nothing is named.
fn int_representatives(taken: &FxHashSet<BigInt>, ordered: bool) -> Vec<BigInt> {
    let mut out: Vec<BigInt> = Vec::new();
    if !ordered || taken.is_empty() {
        let mut value = BigInt::from(0u8);
        while taken.contains(&value) {
            value += 1u8;
        }
        out.push(value);
        return out;
    }
    for r in taken {
        for candidate in [r + 1u8, r - 1u8] {
            if !taken.contains(&candidate) {
                out.push(candidate);
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// `Real` representatives: unordered, the smallest non-negative integer not
/// named; ordered, the midpoint of every two consecutive named values plus
/// `min - 1` and `max + 1`, or `0` when nothing is named.
fn real_representatives(taken: &[Rational64], ordered: bool) -> Vec<Rational64> {
    let mut sorted: Vec<Rational64> = taken.to_vec();
    sorted.sort();
    sorted.dedup();
    if !ordered || sorted.is_empty() {
        let mut value: i64 = 0;
        while sorted.contains(&Rational64::from_integer(value)) {
            value = value.saturating_add(1);
        }
        return vec![Rational64::from_integer(value)];
    }
    // Checked throughout: two literals with large denominators overflow
    // `i64`, and a candidate that cannot be represented is simply skipped —
    // a point is only an instance, so a missing one costs completeness of
    // the saturation test (the certifier then keeps refining), never a
    // verdict it would not otherwise reach.
    let mut out: Vec<Rational64> = Vec::new();
    let one = Rational64::from_integer(1);
    let two = Rational64::from_integer(2);
    if let (Some(min), Some(max)) = (sorted.first(), sorted.last()) {
        out.extend(min.checked_sub(&one));
        out.extend(max.checked_add(&one));
    }
    for pair in sorted.windows(2) {
        if let [low, high] = pair {
            out.extend(low.checked_add(high).and_then(|sum| sum.checked_div(&two)));
        }
    }
    out
}

/// Every ground term of `sort` the goal spells outside a binder.
fn ground_terms_of_sort(goal: &[TermId], sort: SortId, manager: &TermManager) -> Vec<TermId> {
    let mut out: Vec<TermId> = Vec::new();
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack: Vec<TermId> = goal.to_vec();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if matches!(
            data.kind,
            TermKind::Forall { .. } | TermKind::Exists { .. } | TermKind::Let { .. }
        ) {
            continue;
        }
        if data.sort == sort {
            out.push(current);
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    out.sort_unstable_by_key(|term| term.raw());
    out
}

/// Every constant (`Var`) of `sort` the candidate model assigns, other than a
/// bound variable's name, in term-id order.
fn model_constants_of_sort(
    model: &CompletedModel,
    sort: SortId,
    bound_names: &FxHashSet<Spur>,
    manager: &TermManager,
) -> Vec<TermId> {
    let mut out: Vec<TermId> = model
        .assignments
        .keys()
        .copied()
        .filter(|&term| {
            manager.get(term).is_some_and(|data| {
                data.sort == sort
                    && matches!(&data.kind, TermKind::Var(name) if !bound_names.contains(name))
            })
        })
        .collect();
    out.sort_unstable_by_key(|term| term.raw());
    out
}

/// Every constructor of `sort` as a term, when `sort` is a datatype whose
/// constructors are all nullary; `None` otherwise.
fn enumeration_constructors(sort: SortId, manager: &mut TermManager) -> Option<Vec<TermId>> {
    let name = manager.sorts.datatype_name(sort)?.to_string();
    let names: Vec<String> = {
        let def = manager.sorts.get_datatype(&name)?;
        if def
            .constructors
            .iter()
            .any(|ctor| !ctor.selectors.is_empty())
        {
            return None;
        }
        def.constructors
            .iter()
            .map(|ctor| manager.resolve_str(ctor.name).to_string())
            .collect()
    };
    Some(
        names
            .iter()
            .map(|ctor| manager.mk_dt_constructor(ctor, [], sort))
            .collect(),
    )
}

/// Whether `term` is a literal of a scalar sort.
fn is_literal(term: TermId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|t| {
        matches!(
            t.kind,
            TermKind::BitVecConst { .. }
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::True
                | TermKind::False
        )
    })
}

/// The value of a bit-vector or integer literal.
fn int_value(term: TermId, manager: &TermManager) -> Option<BigInt> {
    match &manager.get(term)?.kind {
        TermKind::BitVecConst { value, .. } => Some(value.clone()),
        TermKind::IntConst(value) => Some(value.clone()),
        _ => None,
    }
}

/// The value of a real (or integer, read as real) literal.
fn real_value(term: TermId, manager: &TermManager) -> Option<Rational64> {
    match &manager.get(term)?.kind {
        TermKind::RealConst(value) => Some(*value),
        TermKind::IntConst(value) => i64::try_from(value.clone())
            .ok()
            .map(Rational64::from_integer),
        _ => None,
    }
}
