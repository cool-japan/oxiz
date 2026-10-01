//! Equalities between two literal array **values**, decided exactly before
//! the certificate evaluates or queries anything.
//!
//! An interpretation the certificate checks is a literal array value — a
//! constant array over a literal default with `store`s of literal indices and
//! values on top — so an assertion `a = (store b k v)` becomes, once the
//! interpretation is substituted, an equality between two such values.  The
//! model evaluator does not decide array equality, so every such assertion
//! used to cost a quantifier-free query, and over an `Int` index that query is
//! extensionality over two store chains: `bench/…/AUFLIA/array_update.smt2`'s
//! pinned model kept a sub-solver busy for minutes once the candidate-point
//! search (`infinite`) proposed chains for both arrays.
//!
//! Two literal values are equal exactly when their reads agree everywhere,
//! and a read is known in closed form: the stored value at a stored index, the
//! default elsewhere.  So they are equal iff they read alike at every index
//! either one stores to and — unless those indices exhaust a finite index
//! sort — their defaults are equal.  Nothing here is a heuristic: a pair this
//! module cannot read (a non-literal leaf, an index sort whose size it does
//! not know) is left for the evaluator and the query exactly as before.

use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortKind;
use rustc_hash::{FxHashMap, FxHashSet};

use super::is_array_value;

/// Stored indices a value may carry before this module leaves it alone.
const MAX_STORED_POINTS: usize = 4_096;

/// `term` with every equality between two literal array values replaced by
/// its truth value (see the module docs).
pub(super) fn fold_array_value_equalities(term: TermId, manager: &mut TermManager) -> TermId {
    let mut replacements: FxHashMap<TermId, TermId> = FxHashMap::default();
    let mut stack: Vec<TermId> = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        if let TermKind::Eq(left, right) = data.kind
            && is_array_value(left, manager)
            && is_array_value(right, manager)
        {
            if let Some(equal) = array_values_equal(left, right, manager) {
                let value = if equal {
                    manager.mk_true()
                } else {
                    manager.mk_false()
                };
                replacements.insert(current, value);
            }
            continue;
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    if replacements.is_empty() {
        term
    } else {
        manager.substitute(term, &replacements)
    }
}

/// A literal's value, independent of how the term was interned.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Bv(u32, BigInt),
    Int(BigInt),
    Real(num_rational::Rational64),
    Bool(bool),
}

fn key(term: TermId, manager: &TermManager) -> Option<Key> {
    match &manager.get(term)?.kind {
        TermKind::BitVecConst { value, width } => Some(Key::Bv(
            *width,
            oxiz_core::ast::bv_wrap_unsigned(value, *width),
        )),
        TermKind::IntConst(value) => Some(Key::Int(value.clone())),
        TermKind::RealConst(value) => Some(Key::Real(*value)),
        TermKind::True => Some(Key::Bool(true)),
        TermKind::False => Some(Key::Bool(false)),
        _ => None,
    }
}

/// Whether two literal array values are equal, or `None` when that cannot be
/// read off them.
fn array_values_equal(left: TermId, right: TermId, manager: &TermManager) -> Option<bool> {
    if left == right {
        return Some(true);
    }
    let (left_default, left_points) = read_value(left, manager)?;
    let (right_default, right_points) = read_value(right, manager)?;
    let mut indices: Vec<Key> = left_points.keys().cloned().collect();
    for index in right_points.keys() {
        if !left_points.contains_key(index) {
            indices.push(index.clone());
        }
    }
    for index in &indices {
        let left_read = left_points.get(index).unwrap_or(&left_default);
        let right_read = right_points.get(index).unwrap_or(&right_default);
        if left_read != right_read {
            return Some(false);
        }
    }
    if left_default == right_default {
        return Some(true);
    }
    // The defaults differ: equal only if the stored indices cover a finite
    // index sort entirely.
    let sort = manager.get(left)?.sort;
    let Some(SortKind::Array { domain, .. }) = manager.sorts.get(sort).map(|s| &s.kind) else {
        return None;
    };
    let size: Option<BigInt> = match manager.sorts.get(*domain).map(|s| &s.kind) {
        Some(SortKind::Bool) => Some(BigInt::from(2u8)),
        Some(SortKind::BitVec(width)) => Some(BigInt::from(1u8) << *width),
        Some(SortKind::Int | SortKind::Real) => None,
        _ => return None,
    };
    match size {
        Some(size) if BigInt::from(indices.len()) >= size => Some(true),
        _ => Some(false),
    }
}

/// The default and the `index -> value` map a literal array value reads,
/// outermost store winning, keyed by value; `None` when a leaf is not a
/// literal or the chain is longer than [`MAX_STORED_POINTS`].
fn read_value(value: TermId, manager: &TermManager) -> Option<(Key, FxHashMap<Key, Key>)> {
    let mut points: FxHashMap<Key, Key> = FxHashMap::default();
    let mut current = value;
    loop {
        let data = manager.get(current)?;
        match &data.kind {
            TermKind::Store(inner, index, stored) => {
                let index = key(*index, manager)?;
                let stored = key(*stored, manager)?;
                points.entry(index).or_insert(stored);
                if points.len() > MAX_STORED_POINTS {
                    return None;
                }
                current = *inner;
            }
            _ => {
                let default = super::super::array_axioms::const_array_default(current, manager)?;
                return Some((key(default, manager)?, points));
            }
        }
    }
}
