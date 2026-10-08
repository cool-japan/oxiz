//! A selector or a tester applied to a datatype **constructor application**,
//! decided at construction (`#P2b-82`, re-fix pass 16).
//!
//! Datatypes are free (SMT-LIB 2.6 §4.2.3): `(hd (cons a b))` is `a` and
//! `((_ is cons) (cons a b))` is `true` in every model, `((_ is nil) (cons a
//! b))` is `false`.  Only a selector of the constructor that built the value
//! is determined; `(hd nil)` is left alone — SMT-LIB leaves it unspecified.
//!
//! Folding where the term is built, as `dt_eq` does for an equality of two
//! constructor applications, is what reaches every consumer: the datatype
//! axioms' selector-over-constructor lemma folded `(distinct 1 (hd (cons 1
//! l1)))` for the plain atom, but the congruence closure met `(f (hd (cons 1
//! l1)))` as an application whose argument was an opaque selector node and
//! never merged it with `(f 1)` — `(distinct (f 1) (f (hd (cons 1 l1))))`
//! answered `sat` on every build (0.3.3, `c4b04b7`, `c702310`; z3 `unsat`).
//! Substitution rebuilds selectors and testers through the same two entry
//! points, so a term built by instantiating a binder folds as well, and
//! [`TermManager::fold_constructor_accessors`] folds a term built before (the
//! printed-model certificate calls it on the assertions it closes, recheck
//! 15's `wh2`: `(fst (mk k blue))` over the printed `k`).
//!
//! The declaration's constructor and selector names are read as spurs of this
//! manager's interner — the convention of the parser, which declares every
//! datatype this way, and of every consumer in the solver
//! (`solver::dt_axioms::resolve_decl`, `Context`'s datatype registration).
//! A declaration made through the sort manager's own interner (some of this
//! crate's unit tests do) is outside that convention, and the fold's answer
//! for it is not defined.

use super::super::term::{TermId, TermKind};
use super::super::traversal::get_children;
use super::TermManager;
use crate::interner::Spur;
use crate::prelude::*;
use crate::sort::SortId;

/// How many rounds [`TermManager::fold_constructor_accessors`] runs: one
/// round folds every accessor whose argument is already a constructor
/// application, and a fold can expose one more level.
const MAX_FOLD_ROUNDS: usize = 64;

impl TermManager {
    /// `(selector arg)` of sort `result_sort`, folded to the field when `arg`
    /// is an application of the constructor `selector` belongs to.
    pub fn mk_dt_selector_spur(
        &mut self,
        selector: Spur,
        arg: TermId,
        result_sort: SortId,
    ) -> TermId {
        if let Some(field) = self.selector_over_constructor(selector, arg, result_sort) {
            return field;
        }
        // `#P2b-90`: a selector of a datatype `ite` reads its branches.
        if let Some(lifted) = self.lift_over_dt_ite(arg, &mut |manager, branch| {
            manager.mk_dt_selector_spur(selector, branch, result_sort)
        }) {
            return lifted;
        }
        self.intern(TermKind::DtSelector { selector, arg }, result_sort)
    }

    /// `((_ is constructor) arg)`, folded to `true` / `false` when `arg` is an
    /// application of a constructor of the same datatype.
    pub fn mk_dt_tester_spur(&mut self, constructor: Spur, arg: TermId) -> TermId {
        if let Some(holds) = self.tester_over_constructor(constructor, arg) {
            return if holds { self.true_id } else { self.false_id };
        }
        // `#P2b-90`: a tester of a datatype `ite` reads its branches.
        if let Some(lifted) = self.lift_over_dt_ite(arg, &mut |manager, branch| {
            manager.mk_dt_tester_spur(constructor, branch)
        }) {
            return lifted;
        }
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::DtTester { constructor, arg }, bool_sort)
    }

    /// `term` with every selector of the constructor that built its argument
    /// replaced by the field, and every tester of a constructor application by
    /// `true` / `false`, until none is left (at most `MAX_FOLD_ROUNDS`
    /// rounds).
    pub fn fold_constructor_accessors(&mut self, term: TermId) -> TermId {
        let mut current = term;
        for _ in 0..MAX_FOLD_ROUNDS {
            let map = self.foldable_accessors(current);
            if map.is_empty() {
                break;
            }
            current = self.substitute(current, &map);
        }
        current
    }

    /// Every selector / tester sub-term of `term` that folds, with its fold.
    fn foldable_accessors(&mut self, term: TermId) -> FxHashMap<TermId, TermId> {
        let mut stack: Vec<TermId> = vec![term];
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut found: Vec<TermId> = Vec::new();
        while let Some(current) = stack.pop() {
            if !seen.insert(current) {
                continue;
            }
            let Some(node) = self.get(current) else {
                continue;
            };
            if matches!(
                node.kind,
                TermKind::DtSelector { .. } | TermKind::DtTester { .. }
            ) {
                found.push(current);
            }
            stack.extend(get_children(&node.kind));
        }
        found.sort_unstable();
        let mut map: FxHashMap<TermId, TermId> = FxHashMap::default();
        for accessor in found {
            let folded = match self.get(accessor).map(|t| (t.kind.clone(), t.sort)) {
                Some((TermKind::DtSelector { selector, arg }, sort)) => {
                    self.selector_over_constructor(selector, arg, sort)
                }
                Some((TermKind::DtTester { constructor, arg }, _)) => self
                    .tester_over_constructor(constructor, arg)
                    .map(|holds| if holds { self.true_id } else { self.false_id }),
                _ => None,
            };
            if let Some(folded) = folded {
                map.insert(accessor, folded);
            }
        }
        map
    }

    /// The field `selector` reads from the constructor application `arg`, or
    /// `None` when `arg` is not one, `selector` is not a selector of its
    /// constructor, or the field's sort is not `result_sort`.
    fn selector_over_constructor(
        &self,
        selector: Spur,
        arg: TermId,
        result_sort: SortId,
    ) -> Option<TermId> {
        let node = self.get(arg)?;
        let TermKind::DtConstructor { constructor, args } = &node.kind else {
            return None;
        };
        let name = self.sorts.datatype_name(node.sort)?;
        let def = self.sorts.get_datatype(name)?;
        let declared = def.constructors.iter().find(|c| c.name == *constructor)?;
        if declared.selectors.len() != args.len() {
            return None;
        }
        let position = declared
            .selectors
            .iter()
            .position(|&(field, _)| field == selector)?;
        let field = *args.get(position)?;
        (self.get(field)?.sort == result_sort).then_some(field)
    }

    /// Whether the constructor application `arg` was built by `constructor`,
    /// or `None` when `arg` is not a constructor application or `constructor`
    /// is not a constructor of its datatype.
    fn tester_over_constructor(&self, constructor: Spur, arg: TermId) -> Option<bool> {
        let node = self.get(arg)?;
        let TermKind::DtConstructor {
            constructor: built, ..
        } = &node.kind
        else {
            return None;
        };
        let name = self.sorts.datatype_name(node.sort)?;
        let def = self.sorts.get_datatype(name)?;
        if !def.constructors.iter().any(|c| c.name == constructor) {
            return None;
        }
        Some(*built == constructor)
    }
}

#[cfg(test)]
mod tests {
    use super::super::TermManager;
    use crate::ast::term::TermKind;
    use crate::prelude::*;
    use crate::sort::{DataTypeConstructor, SortId};
    use smallvec::smallvec;

    /// `L = nil | cons(hd Int, tl L)` in `manager`, with its sort.
    fn list(manager: &mut TermManager) -> SortId {
        let sort = manager.sorts.mk_datatype_sort("L");
        let int = manager.sorts.int_sort;
        let cons = DataTypeConstructor {
            name: manager.intern_str("cons"),
            selectors: smallvec![
                (manager.intern_str("hd"), int),
                (manager.intern_str("tl"), sort)
            ],
        };
        let nil = DataTypeConstructor {
            name: manager.intern_str("nil"),
            selectors: smallvec![],
        };
        manager.sorts.declare_datatype("L", vec![nil, cons]);
        sort
    }

    #[test]
    fn a_selector_of_the_building_constructor_is_its_field() {
        let mut m = TermManager::new();
        let sort = list(&mut m);
        let int = m.sorts.int_sort;
        let one = m.mk_int(1);
        let l1 = m.mk_var("l1", sort);
        let cell = m.mk_dt_constructor("cons", [one, l1], sort);
        assert_eq!(m.mk_dt_selector("hd", cell, int), one);
        assert_eq!(m.mk_dt_selector("tl", cell, sort), l1);
        // `(hd nil)` is unspecified: kept as a selector node.
        let nil = m.mk_dt_constructor("nil", [], sort);
        let head_of_nil = m.mk_dt_selector("hd", nil, int);
        assert!(matches!(
            m.get(head_of_nil).map(|t| &t.kind),
            Some(TermKind::DtSelector { .. })
        ));
        // Over an opaque list, a selector node too.
        let head = m.mk_dt_selector("hd", l1, int);
        assert!(matches!(
            m.get(head).map(|t| &t.kind),
            Some(TermKind::DtSelector { .. })
        ));
    }

    #[test]
    fn a_tester_of_a_constructor_application_is_a_literal() {
        let mut m = TermManager::new();
        let sort = list(&mut m);
        let one = m.mk_int(1);
        let l1 = m.mk_var("l1", sort);
        let cell = m.mk_dt_constructor("cons", [one, l1], sort);
        assert_eq!(m.mk_dt_tester("cons", cell), m.mk_true());
        assert_eq!(m.mk_dt_tester("nil", cell), m.mk_false());
        let opaque = m.mk_dt_tester("nil", l1);
        assert!(matches!(
            m.get(opaque).map(|t| &t.kind),
            Some(TermKind::DtTester { .. })
        ));
    }

    /// A node interned past the builder (the pass's input) folds too.
    #[test]
    fn the_pass_folds_an_accessor_interned_unfolded() {
        let mut m = TermManager::new();
        let sort = list(&mut m);
        let int = m.sorts.int_sort;
        let bool_sort = m.sorts.bool_sort;
        let one = m.mk_int(1);
        let l1 = m.mk_var("l1", sort);
        let cell = m.mk_dt_constructor("cons", [one, l1], sort);
        let hd = m.intern_str("hd");
        let nil_name = m.intern_str("nil");
        let raw_head = m.intern_term(
            TermKind::DtSelector {
                selector: hd,
                arg: cell,
            },
            int,
        );
        let raw_tester = m.intern_term(
            TermKind::DtTester {
                constructor: nil_name,
                arg: cell,
            },
            bool_sort,
        );
        assert_eq!(m.fold_constructor_accessors(raw_head), one);
        assert_eq!(m.fold_constructor_accessors(raw_tester), m.mk_false());
    }

    /// What a printed model's certificate needs: `(fst p)` once `p` is its
    /// value, and a nested accessor chain in one call.
    #[test]
    fn a_substituted_value_folds_through_nested_accessors() {
        let mut m = TermManager::new();
        let sort = list(&mut m);
        let int_sort = m.sorts.int_sort;
        let seven = m.mk_int(7);
        let l1 = m.mk_var("l1", sort);
        let l2 = m.mk_var("l2", sort);
        let tail = m.mk_dt_selector("tl", l1, sort);
        let head_of_tail = m.mk_dt_selector("hd", tail, int_sort);
        let inner = m.mk_dt_constructor("cons", [seven, l2], sort);
        let value = m.mk_dt_constructor("cons", [seven, inner], sort);
        let mut subst: FxHashMap<_, _> = FxHashMap::default();
        subst.insert(l1, value);
        let closed = m.substitute(head_of_tail, &subst);
        assert_eq!(m.fold_constructor_accessors(closed), seven);
    }
}
