//! Counterexample candidates from the points a quantifier body's own guards
//! name (`#P2b-70`).
//!
//! The counterexample search instantiates a bound variable with the values
//! the candidate model gives terms of its sort.  Nothing makes that set reach
//! every region the body's guards carve: beside `a = (store ((as const …) #b0)
//! #b0000101 #b1)`, the universal `∀i. i <s 0 ⇒ a[i] = #b1` is false at every
//! negative `i`, and the only bit-vector values the model held were `#b0000000`
//! and `#b0000101` — both outside the guard, where the body holds — so no
//! round ever produced the refuting instance and the check ended `unknown`.
//! Whether some other term happened to take a negative value was a matter of
//! the search's trajectory: HEAD `c702310` refuted all five spellings (`bvslt`,
//! `bvuge`, `bvugt`, `bvule`, a negated `bvult`) and re-fix pass 12's memoised
//! `assert_neq` gate did not.
//!
//! The guards name their own boundaries.  For every comparison between the
//! bound variable and a literal `c` in the body, `c`, `c + 1` and `c − 1`
//! (mod `2ʷ` over a bit-vector) lie on both sides of it, so a body that is
//! false somewhere in a region one of its guards delimits is false at one of
//! those points or at a point the model already names — the same
//! representatives `sat_certify::augment_guard_grounds` and
//! `sat_certify::unnamed_region` use.  They are tried only where the model's
//! own values found no counterexample, as extra combinations; a counterexample
//! found here is an ordinary instance of the asserted universal, so it can
//! only strengthen the search, and the phase never changes whether the round
//! counts as fully evaluated.

use super::*;

/// Guard points one bound variable may add.
const MAX_GUARD_POINTS: usize = 12;

/// Combinations the guard phase may evaluate per quantifier.
const MAX_GUARD_COMBINATIONS: usize = 64;

impl CounterExampleGenerator {
    /// Counterexamples at combinations that give at least one bound variable
    /// a guard point (see the module docs); empty when the body names none.
    pub(super) fn guard_point_counterexamples(
        &mut self,
        quantifier: &QuantifiedFormula,
        candidates: &[Vec<TermId>],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<CounterExample> {
        let mut lists: Vec<Vec<(TermId, bool)>> = Vec::with_capacity(quantifier.bound_vars.len());
        let mut any_new = false;
        for (index, &(name, sort)) in quantifier.bound_vars.iter().enumerate() {
            let existing: Vec<TermId> = candidates
                .get(index)
                .map(|list| {
                    list.iter()
                        .copied()
                        .take(self.max_candidates_per_var)
                        .collect()
                })
                .unwrap_or_default();
            let mut list: Vec<(TermId, bool)> =
                existing.iter().map(|&term| (term, false)).collect();
            for point in guard_points(quantifier.body, name, sort, manager) {
                if list.len() >= existing.len() + MAX_GUARD_POINTS {
                    break;
                }
                if !list.iter().any(|&(term, _)| term == point) {
                    list.push((point, true));
                    any_new = true;
                }
            }
            if list.is_empty() {
                return Vec::new();
            }
            lists.push(list);
        }
        if !any_new {
            return Vec::new();
        }

        let mut found: Vec<CounterExample> = Vec::new();
        let mut evaluated = 0usize;
        let mut cursor: Vec<usize> = vec![0; lists.len()];
        loop {
            let combination: Vec<(TermId, bool)> = cursor
                .iter()
                .zip(lists.iter())
                .filter_map(|(&slot, list)| list.get(slot).copied())
                .collect();
            if combination.len() == lists.len() && combination.iter().any(|&(_, new)| new) {
                evaluated += 1;
                let mut assignment: FxHashMap<Spur, TermId> = FxHashMap::default();
                for (index, &(value, _)) in combination.iter().enumerate() {
                    if let Some(name) = quantifier.var_name(index) {
                        assignment.insert(name, value);
                    }
                }
                let substituted = self.apply_substitution(quantifier.body, &assignment, manager);
                let value = self.evaluate_under_model(substituted, model, manager);
                if self.is_counterexample(value, quantifier.is_universal, manager) {
                    let witnesses: Vec<TermId> = combination.iter().map(|&(v, _)| v).collect();
                    let mut cex = CounterExample::new(
                        quantifier.term,
                        assignment,
                        witnesses,
                        model.generation,
                    );
                    cex.body_value = Some(value);
                    cex.calculate_quality(manager);
                    found.push(cex);
                    if found.len() >= self.max_cex_per_quantifier {
                        break;
                    }
                }
                if evaluated >= MAX_GUARD_COMBINATIONS {
                    break;
                }
            }
            // Odometer, first position fastest.
            let mut position = 0usize;
            loop {
                let Some(slot) = cursor.get_mut(position) else {
                    return found;
                };
                *slot += 1;
                if *slot < lists.get(position).map_or(0, Vec::len) {
                    break;
                }
                *slot = 0;
                position += 1;
            }
        }
        found
    }
}

/// `c`, `c + 1` and `c − 1` for every literal `c` the bound variable `name`
/// (of sort `sort`) is compared with in `body`, outside nested binders, in
/// ascending value order.
fn guard_points(body: TermId, name: Spur, sort: SortId, manager: &mut TermManager) -> Vec<TermId> {
    let is_bound = |term: TermId, manager: &TermManager| -> bool {
        manager.get(term).is_some_and(|data| {
            data.sort == sort && matches!(data.kind, TermKind::Var(n) if n == name)
        })
    };
    let mut literals: Vec<TermId> = Vec::new();
    let mut stack: Vec<TermId> = vec![body];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut children: Vec<TermId> = Vec::new();
    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(data) = manager.get(current) else {
            continue;
        };
        let pairs: SmallVec<[(TermId, TermId); 4]> = match &data.kind {
            TermKind::Forall { .. } | TermKind::Exists { .. } => continue,
            TermKind::Eq(l, r)
            | TermKind::BvUlt(l, r)
            | TermKind::BvUle(l, r)
            | TermKind::BvSlt(l, r)
            | TermKind::BvSle(l, r)
            | TermKind::Lt(l, r)
            | TermKind::Le(l, r)
            | TermKind::Gt(l, r)
            | TermKind::Ge(l, r) => SmallVec::from_slice(&[(*l, *r)]),
            TermKind::Distinct(args) => {
                let mut out: SmallVec<[(TermId, TermId); 4]> = SmallVec::new();
                for (i, &left) in args.iter().enumerate() {
                    for &right in args.iter().skip(i + 1) {
                        out.push((left, right));
                    }
                }
                out
            }
            // A read over a write compares the write's index with the read's:
            // `(select (store a i v) c)` is `v` exactly where `i = c` (re-fix
            // pass 15; `qeq120/q0005`, whose refuting instance is `i = c`,
            // was reached only through E-matching over terms a declined
            // certificate round had interned).
            TermKind::Select(array, index) => {
                let mut out: SmallVec<[(TermId, TermId); 4]> = SmallVec::new();
                let mut level = *array;
                while let Some(TermKind::Store(inner, written, _)) =
                    manager.get(level).map(|t| &t.kind)
                {
                    out.push((*written, *index));
                    level = *inner;
                }
                out
            }
            _ => SmallVec::new(),
        };
        for (left, right) in pairs {
            if is_bound(left, manager) && is_literal_of(right, sort, manager) {
                literals.push(right);
            } else if is_bound(right, manager) && is_literal_of(left, sort, manager) {
                literals.push(left);
            }
        }
        children.clear();
        children.extend(oxiz_core::ast::traversal::get_children(&data.kind));
        stack.extend(children.iter().copied());
    }
    let mut keyed: Vec<(BigInt, TermId)> = Vec::new();
    for literal in literals {
        for point in neighbourhood(literal, manager) {
            if let Some(key) = literal_value(point, manager)
                && !keyed.iter().any(|&(_, term)| term == point)
            {
                keyed.push((key, point));
            }
        }
    }
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    keyed.into_iter().map(|(_, term)| term).collect()
}

/// Whether `term` is a bit-vector or integer literal of `sort`.
fn is_literal_of(term: TermId, sort: SortId, manager: &TermManager) -> bool {
    manager.get(term).is_some_and(|data| {
        data.sort == sort
            && matches!(
                data.kind,
                TermKind::BitVecConst { .. } | TermKind::IntConst(_)
            )
    })
}

/// A bit-vector or integer literal's value.
fn literal_value(term: TermId, manager: &TermManager) -> Option<BigInt> {
    match &manager.get(term)?.kind {
        TermKind::BitVecConst { value, .. } | TermKind::IntConst(value) => Some(value.clone()),
        _ => None,
    }
}

/// `c`, `c + 1` and `c − 1` (mod `2ʷ` over a bit-vector of width `w`).
fn neighbourhood(literal: TermId, manager: &mut TermManager) -> Vec<TermId> {
    match manager.get(literal).map(|data| data.kind.clone()) {
        Some(TermKind::BitVecConst { value, width }) => {
            let modulus = BigInt::from(1u8) << width;
            let up = (&value + 1u8) % &modulus;
            let down = (&value + &modulus - 1u8) % &modulus;
            vec![
                literal,
                manager.mk_bitvec(up, width),
                manager.mk_bitvec(down, width),
            ]
        }
        Some(TermKind::IntConst(value)) => vec![
            literal,
            manager.mk_int(&value + 1u8),
            manager.mk_int(&value - 1u8),
        ],
        _ => vec![literal],
    }
}
