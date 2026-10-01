//! Fresh values for the classes no theory valued (`#P2b-71`), and the
//! function tables they key (re-fix pass 15, decisions (64), (68)).
//!
//! # What is minted
//!
//! A numeric or bit-vector term the arithmetic and bit-vector solvers never
//! constrained has no value of its own — the model builder records the sort
//! default and marks it so (`Model::set_default`).  Two such terms the
//! congruence closure kept in *different* classes then printed the same
//! value, and every table keyed by them described a function that is not
//! one: `(f (g k)) = 1` beside `(f (g j)) = 2` printed `g` constantly `0`, so
//! `f` had to be `1` and `2` at `0`.
//!
//! Pass 14 minted values for **variables** only (a declared constant the
//! congruence closure knows, an index or argument position, an `exists`
//! Skolem constant).  This pass reaches every class through its
//! **application and `select` members** too, innermost first, and it keys
//! every one of them by the **point** it reads — the function and the values
//! of its arguments, or the array's class and the value of its index:
//!
//! * a class a theory valued (a model entry that is not a default, or a
//!   literal member) keeps that value;
//! * an application whose point another application already has — theory
//!   valued or minted — takes that application's value: the printed table
//!   is a function, and two applications at one point are one entry
//!   (without this, a fresh value for `(f 4)` beside `(f y)` with `y = 4`
//!   in the arithmetic model and in another class of the closure gave `f`
//!   two entries at `4`);
//! * every other class gets the default it already prints where no other
//!   class uses that value, else the smallest value of its sort nothing
//!   else uses.
//!
//! Distinct classes therefore print distinct values (keeping every
//! disequality the search decided), one point prints one value (keeping
//! every table a function), and the printers — the constants, the tables,
//! the arrays and `(get-value)` — all read the one model that results.  A
//! goal where the values still make the printed model falsify an assertion
//! is caught by `printed_check` and printed without them.

use super::*;
use crate::prelude::{FxHashMap, FxHashSet};
use num_rational::Rational64;

/// A literal's value, as the key the minting avoids collisions on.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum ValueKey {
    Int(BigInt),
    Real(Rational64),
    Bv(u32, BigInt),
}

/// Values tried per sort before minting gives up on a finite sort.
const MAX_MINT_SCAN: u64 = 1 << 16;

/// One argument's value, as far as a point needs it: a value term, or the
/// class of an uninterpreted-sort term (whose witness the printers assign
/// per class).
#[derive(Clone, PartialEq, Eq, Hash)]
enum ArgValue {
    Term(TermId),
    Class(super::class_values::ClassKey),
}

/// Where an application or a read takes its value.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Point {
    Apply(String, Vec<ArgValue>),
    Select(super::class_values::ClassKey, ArgValue),
}

impl Context {
    /// The fresh values of `#P2b-71` for `model`, as `(term, value)`: see
    /// the module docs.
    pub(super) fn mint_fresh_values(
        &mut self,
        model: &crate::solver::Model,
    ) -> Vec<(TermId, TermId)> {
        let bound = self.bound_only_variables();
        let variables = self.variable_candidates(model, &bound);
        let members = self.member_terms(model, &bound);
        let mintable_members: Vec<TermId> = members
            .iter()
            .copied()
            .filter(|&term| {
                (model.get(term).is_none() || model.is_defaulted(term)) && self.mintable_sort(term)
            })
            .collect();
        if variables.is_empty() && mintable_members.is_empty() {
            return Vec::new();
        }
        let chosen = self.chosen_class_values(model);
        let minting: FxHashSet<TermId> = variables
            .iter()
            .chain(mintable_members.iter())
            .copied()
            .collect();
        let mut used = self.values_in_use(model, &minting);
        let mut current = model.clone();
        let mut by_class: FxHashMap<super::class_values::ClassKey, TermId> = FxHashMap::default();
        let mut out: Vec<(TermId, TermId)> = Vec::new();

        // Variables first: they are the leaves every point is keyed by.
        for term in variables {
            let key = self.class_key(term);
            let value = match chosen.get(&key).or_else(|| by_class.get(&key)) {
                Some(&value) => Some(value),
                None => self.fresh_or_default(term, model, &mut used),
            };
            let Some(value) = value else {
                continue;
            };
            by_class.insert(key, value);
            if model.get(term) != Some(value) {
                current.set(term, value);
                out.push((term, value));
            }
        }

        // Applications and reads, innermost first; at each depth the points
        // a theory valued are registered before any fresh value is chosen.
        let mut points: FxHashMap<Point, TermId> = FxHashMap::default();
        let mut by_depth: std::collections::BTreeMap<usize, Vec<TermId>> =
            std::collections::BTreeMap::new();
        let depths = self.nesting_depths(&members);
        for &term in &members {
            by_depth
                .entry(depths.get(&term).copied().unwrap_or(0))
                .or_default()
                .push(term);
        }
        for (_, level) in by_depth {
            for &term in &level {
                if minting.contains(&term) {
                    continue;
                }
                let Some(value) = current.get(term).filter(|&v| self.is_point_value(v)) else {
                    continue;
                };
                if let Some(point) = self.point_of(term, &current, &by_class, &chosen) {
                    points.entry(point).or_insert(value);
                }
            }
            for &term in &level {
                if !minting.contains(&term) {
                    continue;
                }
                let key = self.class_key(term);
                let point = self.point_of(term, &current, &by_class, &chosen);
                let value = match chosen
                    .get(&key)
                    .or_else(|| point.as_ref().and_then(|p| points.get(p)))
                    .or_else(|| by_class.get(&key))
                {
                    Some(&value) => Some(value),
                    None => self.fresh_or_default(term, model, &mut used),
                };
                let Some(value) = value else {
                    continue;
                };
                if let Some(point) = point {
                    points.entry(point).or_insert(value);
                }
                by_class.insert(key, value);
                if model.get(term) != Some(value) {
                    current.set(term, value);
                    out.push((term, value));
                }
            }
        }
        out
    }

    /// The variables pass 14 minted for: every `Int` / `Real` / bit-vector
    /// declared constant the congruence closure knows, and every declared
    /// constant or `exists` Skolem constant read as an index or passed as an
    /// argument, the model leaves unvalued or only defaulted — in term-id
    /// order.
    fn variable_candidates(
        &self,
        model: &crate::solver::Model,
        bound: &super::array_model::BoundOnly,
    ) -> Vec<TermId> {
        let mut positions: FxHashSet<TermId> = FxHashSet::default();
        for &term in model.assignments().keys() {
            match self.terms.get(term).map(|t| &t.kind) {
                Some(TermKind::Select(_, index)) => {
                    positions.insert(*index);
                }
                Some(TermKind::Apply { args, .. }) => positions.extend(args.iter().copied()),
                _ => {}
            }
        }
        let declared: Vec<TermId> = self.declared_consts.iter().map(|d| d.term).collect();
        // A position counts when it is a declared constant or the Skolem
        // constant of an asserted `exists` (the witness the published model
        // has to show); the search's other internal symbols (binder-sort
        // witnesses, instantiation constants) print nowhere.
        let mut out: Vec<TermId> = declared
            .iter()
            .copied()
            .filter(|&term| self.solver.euf_class_representative(term).is_some())
            .chain(
                positions
                    .into_iter()
                    .filter(|&term| declared.contains(&term) || self.is_exists_skolem(term)),
            )
            .filter(|&term| {
                matches!(
                    self.terms.get(term).map(|t| &t.kind),
                    Some(TermKind::Var(_))
                ) && (model.get(term).is_none() || model.is_defaulted(term))
                    && self.mintable_sort(term)
                    && !self.mentions_bound_variable(term, bound)
            })
            .collect();
        out.sort_unstable_by_key(|term| term.raw());
        out.dedup();
        out
    }

    /// Every application of a tabulated function and every read of an array
    /// the model does not interpret outright, the congruence closure interned
    /// and no bound variable occurs in — any sort, since a point is keyed by
    /// every argument — in term-id order.  An application of a function the
    /// model does not tabulate (a recursive definition's) is its
    /// definition's to value; a read of an array a completion interpreted is
    /// that value's to give.
    fn member_terms(
        &self,
        model: &crate::solver::Model,
        bound: &super::array_model::BoundOnly,
    ) -> Vec<TermId> {
        let tabulated: FxHashSet<&str> = self
            .declared_funs
            .iter()
            .filter(|d| !d.interpreted && !d.arg_sorts.is_empty())
            .map(|d| d.name.as_str())
            .collect();
        self.solver
            .euf_interned_terms()
            .into_iter()
            .filter(|&term| match self.terms.get(term).map(|t| &t.kind) {
                Some(TermKind::Apply { func, .. }) => {
                    tabulated.contains(self.terms.resolve_str(*func))
                }
                Some(TermKind::Select(array, _)) => model.get(*array).is_none(),
                _ => false,
            })
            .filter(|&term| !self.mentions_bound_variable(term, bound))
            .collect()
    }

    /// The value a theory chose for each class: a model entry that is not a
    /// default, or a literal member.
    fn chosen_class_values(
        &self,
        model: &crate::solver::Model,
    ) -> FxHashMap<super::class_values::ClassKey, TermId> {
        let mut chosen: FxHashMap<super::class_values::ClassKey, TermId> = FxHashMap::default();
        let mut entries: Vec<(TermId, TermId)> = model
            .assignments()
            .iter()
            .filter(|&(&term, _)| !model.is_defaulted(term))
            .map(|(&term, &value)| (term, value))
            .collect();
        entries.sort_unstable_by_key(|&(term, _)| term.raw());
        for (term, value) in entries {
            if self.value_key(value).is_some() {
                chosen.entry(self.class_key(term)).or_insert(value);
            }
        }
        for index in 0..(self.terms.len() as u32) {
            let term = TermId(index);
            if self.value_key(term).is_some() {
                chosen.entry(self.class_key(term)).or_insert(term);
            }
        }
        chosen
    }

    /// The default `model` prints for `term`, where no other class uses its
    /// value, else the smallest unused value of its sort.
    fn fresh_or_default(
        &mut self,
        term: TermId,
        model: &crate::solver::Model,
        used: &mut FxHashSet<ValueKey>,
    ) -> Option<TermId> {
        if let Some(value) = model.get(term)
            && let Some(key) = self.value_key(value)
            && used.insert(key)
        {
            return Some(value);
        }
        let sort = self.terms.get(term)?.sort;
        self.fresh_value(sort, used)
    }

    /// Nesting depth of every member: an application or read over no member
    /// is at depth 0, one over a member at depth `d` at `d + 1`.
    fn nesting_depths(&self, members: &[TermId]) -> FxHashMap<TermId, usize> {
        let member_set: FxHashSet<TermId> = members.iter().copied().collect();
        let mut depth: FxHashMap<TermId, usize> = FxHashMap::default();
        for &root in members {
            let mut stack: Vec<(TermId, bool)> = vec![(root, false)];
            while let Some((term, expanded)) = stack.pop() {
                if depth.contains_key(&term) {
                    continue;
                }
                let Some(data) = self.terms.get(term) else {
                    depth.insert(term, 0);
                    continue;
                };
                let children = oxiz_core::ast::traversal::get_children(&data.kind);
                if !expanded {
                    stack.push((term, true));
                    for &child in &children {
                        if !depth.contains_key(&child) {
                            stack.push((child, false));
                        }
                    }
                    continue;
                }
                let inner = children
                    .iter()
                    .map(|child| depth.get(child).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                let own = if member_set.contains(&term) {
                    inner + 1
                } else {
                    inner
                };
                depth.insert(term, own);
            }
        }
        depth
    }

    /// The point `term` reads: its function and argument values, or its
    /// array's class and index value; `None` when an argument has no value
    /// yet.
    fn point_of(
        &mut self,
        term: TermId,
        current: &crate::solver::Model,
        by_class: &FxHashMap<super::class_values::ClassKey, TermId>,
        chosen: &FxHashMap<super::class_values::ClassKey, TermId>,
    ) -> Option<Point> {
        match self.terms.get(term)?.kind.clone() {
            TermKind::Apply { func, args } => {
                let name = self.terms.resolve_str(func).to_string();
                let mut values: Vec<ArgValue> = Vec::with_capacity(args.len());
                for arg in args {
                    values.push(self.arg_value(arg, current, by_class, chosen)?);
                }
                Some(Point::Apply(name, values))
            }
            TermKind::Select(array, index) => Some(Point::Select(
                self.class_key(array),
                self.arg_value(index, current, by_class, chosen)?,
            )),
            _ => None,
        }
    }

    /// One argument's value for [`Self::point_of`].
    fn arg_value(
        &mut self,
        arg: TermId,
        current: &crate::solver::Model,
        by_class: &FxHashMap<super::class_values::ClassKey, TermId>,
        chosen: &FxHashMap<super::class_values::ClassKey, TermId>,
    ) -> Option<ArgValue> {
        if self.is_point_value(arg) {
            return Some(ArgValue::Term(arg));
        }
        let sort = self.terms.get(arg)?.sort;
        if self.is_uninterpreted_sort(sort) {
            return Some(ArgValue::Class(self.class_key(arg)));
        }
        if let Some(value) = current.get(arg).filter(|&v| self.is_point_value(v)) {
            return Some(ArgValue::Term(value));
        }
        let key = self.class_key(arg);
        if let Some(&value) = by_class.get(&key).or_else(|| chosen.get(&key)) {
            return Some(ArgValue::Term(value));
        }
        let value = self.solver.eval_in_model(arg, current, &self.terms, 0)?;
        let term = match (value, self.terms.sorts.get(sort).map(|s| s.kind.clone())?) {
            (crate::solver::EvalVal::Bool(flag), _) => self.terms.mk_bool(flag),
            (crate::solver::EvalVal::Num(number), SortKind::Int) if *number.denom() == 1 => {
                self.terms.mk_int(*number.numer())
            }
            (crate::solver::EvalVal::Num(number), SortKind::Real) => self.terms.mk_real(number),
            (crate::solver::EvalVal::Bv { value, width }, _) => self.terms.mk_bitvec(value, width),
            _ => return None,
        };
        Some(ArgValue::Term(term))
    }

    /// Whether `term` is a value a point can be keyed by: a numeric,
    /// bit-vector or Boolean literal, or a constructor value.
    fn is_point_value(&self, term: TermId) -> bool {
        self.value_key(term).is_some()
            || matches!(
                self.terms.get(term).map(|t| &t.kind),
                Some(TermKind::True | TermKind::False)
            )
            || super::array_model::is_constructor_value(term, &self.terms)
    }

    /// Whether `term` is the Skolem constant of an asserted `exists`
    /// (`encode::exists_skolem`).
    fn is_exists_skolem(&self, term: TermId) -> bool {
        match self.terms.get(term).map(|t| &t.kind) {
            Some(TermKind::Var(name)) => {
                oxiz_core::smtlib::is_reserved_tag(self.terms.resolve_str(*name), "sk")
            }
            _ => false,
        }
    }

    /// Whether `term`'s sort is one values are minted for.
    fn mintable_sort(&self, term: TermId) -> bool {
        self.terms.get(term).is_some_and(|t| {
            matches!(
                self.terms.sorts.get(t.sort).map(|s| &s.kind),
                Some(SortKind::Int | SortKind::Real | SortKind::BitVec(_))
            )
        })
    }

    /// Every value a fresh one must avoid: each literal the congruence
    /// closure knows (a class of its own that a disequality may separate from
    /// another), the model value of every term not about to be minted, the
    /// value the model folds for every other term the closure interned (a
    /// compound argument such as `(+ x 2)` prints the value it folds to), and
    /// the default of every declared constant that prints one (unvalued and
    /// not about to be minted).  A literal the closure never saw — an element
    /// value, a constant array's default — was never compared with anything.
    fn values_in_use(
        &self,
        model: &crate::solver::Model,
        minting: &FxHashSet<TermId>,
    ) -> FxHashSet<ValueKey> {
        let mut used: FxHashSet<ValueKey> = FxHashSet::default();
        for index in 0..(self.terms.len() as u32) {
            let term = TermId(index);
            if let Some(key) = self.value_key(term)
                && self.solver.euf_class_representative(term).is_some()
            {
                used.insert(key);
            }
        }
        for (term, &value) in model.assignments() {
            if minting.contains(term) {
                continue;
            }
            if let Some(key) = self.value_key(value) {
                used.insert(key);
            }
        }
        for term in self.solver.euf_interned_terms() {
            if minting.contains(&term) || model.get(term).is_some() || !self.mintable_sort(term) {
                continue;
            }
            let Some(value) = self.solver.eval_in_model(term, model, &self.terms, 0) else {
                continue;
            };
            match value {
                crate::solver::EvalVal::Num(number) => {
                    if *number.denom() == 1 {
                        used.insert(ValueKey::Int(BigInt::from(*number.numer())));
                    }
                    used.insert(ValueKey::Real(number));
                }
                crate::solver::EvalVal::Bv { value, width } => {
                    used.insert(ValueKey::Bv(width, value));
                }
                crate::solver::EvalVal::Bool(_) => {}
            }
        }
        for decl in &self.declared_consts {
            if model.get(decl.term).is_some() || minting.contains(&decl.term) {
                continue;
            }
            match self.terms.sorts.get(decl.sort).map(|s| &s.kind) {
                Some(SortKind::Int) => {
                    used.insert(ValueKey::Int(BigInt::zero()));
                }
                Some(SortKind::Real) => {
                    used.insert(ValueKey::Real(Rational64::from_integer(0)));
                }
                Some(SortKind::BitVec(width)) => {
                    used.insert(ValueKey::Bv(*width, BigInt::zero()));
                }
                _ => {}
            }
        }
        used
    }

    /// The [`ValueKey`] of a literal term, or `None` for anything else.
    fn value_key(&self, term: TermId) -> Option<ValueKey> {
        match &self.terms.get(term)?.kind {
            TermKind::IntConst(value) => Some(ValueKey::Int(value.clone())),
            TermKind::RealConst(value) => Some(ValueKey::Real(*value)),
            TermKind::BitVecConst { value, width } => Some(ValueKey::Bv(
                *width,
                oxiz_core::ast::bv_wrap_unsigned(value, *width),
            )),
            _ => None,
        }
    }

    /// The smallest value of `sort` (from zero upward) not in `used`, marked
    /// used; `None` when a finite sort has none left within the scan bound.
    fn fresh_value(&mut self, sort: SortId, used: &mut FxHashSet<ValueKey>) -> Option<TermId> {
        let kind = self.terms.sorts.get(sort).map(|s| s.kind.clone())?;
        let mut candidate: u64 = 0;
        while candidate < MAX_MINT_SCAN {
            let key = match kind {
                SortKind::Int => ValueKey::Int(BigInt::from(candidate)),
                SortKind::Real => {
                    ValueKey::Real(Rational64::from_integer(i64::try_from(candidate).ok()?))
                }
                SortKind::BitVec(width) => {
                    if width < 64 && candidate >= (1u64 << width) {
                        return None;
                    }
                    ValueKey::Bv(width, BigInt::from(candidate))
                }
                _ => return None,
            };
            if used.insert(key.clone()) {
                return Some(match key {
                    ValueKey::Int(value) => self.terms.mk_int(value),
                    ValueKey::Real(value) => self.terms.mk_real(value),
                    ValueKey::Bv(width, value) => self.terms.mk_bitvec(value, width),
                });
            }
            candidate += 1;
        }
        None
    }
}
