//! A representative of the values no term names, over a datatype **with a
//! field** (`#P2b-75`, completing `#P2b-64` (1)'s decline).
//!
//! `unnamed_region` covers an enumeration datatype by listing every
//! constructor, and used to decline every other datatype: no single value was
//! known to be unnamed.  That made a guard `¬(x = nil)` over a list binder
//! either a wrong `sat` (`(=> (not (= x nil)) false)` with only `nil` in the
//! relevant set, every instance vacuous) or, once declined, an `unknown` on a
//! satisfiable goal every earlier build certified.
//!
//! A bound variable of a datatype sort occurs in the fragment only as an
//! uninterpreted function's or an array's argument and on one side of an
//! equality guard (`eu_walk` rejects a selector or tester over it), so the
//! projection needs one value distinct from every *named* value — every value
//! the goal's terms of the sort denote — and nothing else: an equality guard
//! then reads the same at every unnamed value and at the representative.
//! This module builds one: it enumerates ground values of the sort in size
//! order (constructors in declaration order, scalar fields from `0` upward)
//! until it holds one more distinct value than there are named ones, and
//! returns the first that is not named.  A named term whose value is not a
//! ground constructor value, or a sort whose values cannot be spelled (an
//! uninterpreted field), leaves the caller to decline as before.

use num_bigint::BigInt;
use num_rational::Rational64;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::{SortId, SortKind};

use super::super::model_completion::CompletedModel;

/// Constructor nesting the enumeration may use.
const MAX_DEPTH: usize = 4;

/// The ground value `term` denotes, when it is a constructor value itself or
/// the model gives it one; `None` otherwise.
pub(super) fn named_value(
    term: TermId,
    model: &CompletedModel,
    manager: &TermManager,
) -> Option<TermId> {
    if is_ground_value(term, manager) {
        return Some(term);
    }
    model
        .eval(term)
        .filter(|&value| is_ground_value(value, manager))
}

/// One value of `sort` that is none of `named`, or `None` when the
/// enumeration cannot produce one.
pub(super) fn unnamed_value(
    sort: SortId,
    named: &[TermId],
    manager: &mut TermManager,
) -> Option<TermId> {
    let wanted = named.len().saturating_add(1);
    enumerate(sort, wanted, MAX_DEPTH, manager)
        .into_iter()
        .find(|value| !named.contains(value))
}

/// Up to `count` distinct ground values of `sort`, smallest first.
fn enumerate(sort: SortId, count: usize, depth: usize, manager: &mut TermManager) -> Vec<TermId> {
    let Some(kind) = manager.sorts.get(sort).map(|s| s.kind.clone()) else {
        return Vec::new();
    };
    let bound = u32::try_from(count).unwrap_or(u32::MAX);
    match kind {
        SortKind::Bool => [manager.mk_false(), manager.mk_true()]
            .into_iter()
            .take(count)
            .collect(),
        SortKind::Int => (0..bound).map(|v| manager.mk_int(v)).collect(),
        SortKind::Real => (0..bound)
            .map(|v| manager.mk_real(Rational64::from_integer(i64::from(v))))
            .collect(),
        SortKind::BitVec(width) => {
            let modulus = BigInt::from(1u8) << width;
            (0..bound)
                .map(BigInt::from)
                .take_while(|v| *v < modulus)
                .map(|v| manager.mk_bitvec(v, width))
                .collect()
        }
        SortKind::Datatype(_) => enumerate_datatype(sort, count, depth, manager),
        _ => Vec::new(),
    }
}

/// [`enumerate`] over a datatype: every nullary constructor first, then each
/// constructor with fields over the product of its fields' values (one
/// nesting level fewer for a datatype field).
fn enumerate_datatype(
    sort: SortId,
    count: usize,
    depth: usize,
    manager: &mut TermManager,
) -> Vec<TermId> {
    let Some(name) = manager.sorts.datatype_name(sort).map(str::to_string) else {
        return Vec::new();
    };
    let constructors: Vec<(String, Vec<SortId>)> = match manager.sorts.get_datatype(&name) {
        Some(def) => def
            .constructors
            .iter()
            .map(|ctor| {
                (
                    manager.resolve_str(ctor.name).to_string(),
                    ctor.selectors.iter().map(|&(_, field)| field).collect(),
                )
            })
            .collect(),
        None => return Vec::new(),
    };
    let mut out: Vec<TermId> = Vec::new();
    for (ctor, fields) in &constructors {
        if fields.is_empty() && out.len() < count {
            out.push(manager.mk_dt_constructor(ctor, [], sort));
        }
    }
    if depth == 0 {
        return out;
    }
    for (ctor, fields) in &constructors {
        if fields.is_empty() {
            continue;
        }
        let mut lists: Vec<Vec<TermId>> = Vec::with_capacity(fields.len());
        for &field in fields {
            let values = enumerate(field, count, depth - 1, manager);
            if values.is_empty() {
                lists.clear();
                break;
            }
            lists.push(values);
        }
        if lists.len() != fields.len() {
            continue;
        }
        // The product in odometer order, stopping once `count` are held.
        let mut positions: Vec<usize> = vec![0; lists.len()];
        'product: while out.len() < count {
            let args: Vec<TermId> = positions
                .iter()
                .zip(&lists)
                .map(|(&i, list)| list[i])
                .collect();
            let value = manager.mk_dt_constructor(ctor, args, sort);
            if !out.contains(&value) {
                out.push(value);
            }
            for slot in (0..positions.len()).rev() {
                positions[slot] += 1;
                if positions[slot] < lists[slot].len() {
                    continue 'product;
                }
                positions[slot] = 0;
            }
            break;
        }
        if out.len() >= count {
            break;
        }
    }
    out
}

/// Whether `term` is a ground value: a scalar literal, or a constructor
/// applied to ground values.
pub(super) fn is_ground_value(term: TermId, manager: &TermManager) -> bool {
    let mut stack: Vec<TermId> = vec![term];
    while let Some(current) = stack.pop() {
        match manager.get(current).map(|t| &t.kind) {
            Some(
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. },
            ) => {}
            Some(TermKind::DtConstructor { args, .. }) => stack.extend(args.iter().copied()),
            _ => return false,
        }
    }
    true
}
