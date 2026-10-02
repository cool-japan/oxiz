//! Pulling apart two datatype values the search proved *distinct* but the
//! reconstruction rendered identically (`Solver::separate_disequal_dt_values`).
//!
//! The repair re-values one numeric field no assertion mentions.  "No
//! assertion mentions it" is not "nothing constrains it": the datatype axioms'
//! congruence lemmas put the same accessor in atoms of their own — `(=> (= (h
//! 2) l2) (= (hd (h 2)) (hd l2)))` — and the search decides those atoms like any
//! other.  A bump that ignored them moved `(hd l2)` from the `2` the tableau
//! had chosen to `3`, equal to `(hd (h 2))`, which the search had kept apart
//! (`(> (hd (h 2)) (hd l2))` true, `(= (hd (h 2)) (hd l2))` false): the model
//! gate's negated-equality check then refused the candidate as a
//! falsification, the refused candidates were blocked, and the blocking
//! clauses downgraded the next check's refutation to `unknown` (recheck 15's
//! `u03`: a trivially false tester after a `push`/`pop`, `unsat` on `c702310`,
//! `unknown` on re-fix pass 15's tree).  A bump is now kept only when it makes
//! no numeric equality the search decided `false` hold
//! ([`Solver::keeps_negated_equalities`], the gate's own check); the next
//! value (up to [`MAX_SEPARATION_OFFSET`] away), then the next field, is tried
//! otherwise.  An arithmetic atom the search decided `true` over an accessor no
//! assertion reads is not protected: moving such a field is the repair's whole
//! purpose, and protecting it too left `(distinct l2 l3)` printed as two equal
//! lists (`gen_dt.py` seed 30093154, `d00037`).
//!
//! # A field whose sort is a datatype (re-fix pass 18)
//!
//! Only numeric fields used to move, so a disequality decided over an
//! enumeration-valued field could not be repaired: `(not ((_ is empty) b1))`
//! beside `(not (= b1 (full red)))` over `Box = empty | full (item Col)`
//! rebuilt `b1` as `(full red)` — `(item b1)` is a term the datatype axioms
//! introduce and do not axiomatise again, so it took its sort's default.
//! Re-fix pass 17's net withheld that model; under decision (85) a model the
//! reader shows false makes the check `unknown`, and twenty of the 696
//! formulas of `solver_final_regression::test_dt_differential_against_brute_force_oracle`
//! answered `unknown`.  A datatype-sorted field no assertion mentions now
//! moves too, to each other constructor's ground default in declaration
//! order, and a move is kept only when it leaves strictly fewer
//! decided disequalities holding between equal values; when the right side
//! of a pair cannot move (a constructor literal), the left side is tried.

use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use oxiz_sat::LBool;

use super::super::EvalVal;
use super::super::Solver;
use super::super::dt_axioms::DeclInfo;
use super::super::model_eval::EvalOutcome;
use super::super::types::{Constraint, Model};
use super::{assertion_subterms, equality_class};

/// The search's decided datatype equalities, as an adjacency map.
type Adjacency = FxHashMap<TermId, Vec<TermId>>;

/// How far a separated field may be moved from the value it was rebuilt with
/// before the repair gives the field up (each step is one candidate value).
const MAX_SEPARATION_OFFSET: i64 = 8;

impl Solver {
    /// Pull apart two datatype values that the search proved *distinct* but the
    /// reconstruction rendered identically.
    ///
    /// Distinctness of two applications of the *same* constructor lives entirely
    /// in their fields, and a field the theory never pinned has no witness of its
    /// own: `(assert (not (= p q)))` over `(mk-pair (fst Int) (snd Int))` leaves
    /// the linear solver free to report the same value for `(fst p)` and
    /// `(fst q)` (it discharges disequalities by case split, not by separating
    /// witnesses — the same effect `Solver::eval_in_model` documents for
    /// `distinct`), so both sides reconstructed to `(mk-pair 0 0)`.
    ///
    /// The repair only ever re-values a field whose accessor occurs in *no*
    /// assertion — one nothing in the formula constrains, where every value of
    /// the sort is equally legitimate — so it can only turn a wrong witness into
    /// a right one, never the reverse.  See [`Solver::separate_dt_value`] for
    /// why that, and not the arithmetic solver's `value()`, is the pin test.
    pub(super) fn separate_disequal_dt_values(
        &self,
        decls: &FxHashMap<SortId, DeclInfo>,
        model: &mut Model,
        manager: &mut TermManager,
    ) {
        let (disequal, equal_adjacency) = self.decided_dt_equalities(model, manager);
        if disequal.is_empty() {
            return;
        }
        let asserted = assertion_subterms(&self.assertions, manager);
        for &(left, right) in &disequal {
            let (Some(left_value), Some(right_value)) = (model.get(left), model.get(right)) else {
                continue;
            };
            if left_value != right_value {
                continue;
            }
            // The whole class the search proved equal to `right` has to move
            // together, or the repair would break one of those equalities while
            // fixing the disequality.  A class containing *both* sides is a
            // contradictory assignment the repair must not paper over.
            // … and the terms the congruence closure merged with it.
            let class = self.separation_class(right, &equal_adjacency, manager);
            if class.contains(&left) {
                continue;
            }
            let Some(sort) = manager.get(right).map(|node| node.sort) else {
                continue;
            };
            if self.separate_dt_value(right, sort, &class, &asserted, decls, model, manager) {
                continue;
            }
            // A datatype-valued field (see the module docs), of the right side
            // and then of the left.
            if self.separate_by_datatype_field(
                right,
                sort,
                &class,
                &asserted,
                decls,
                (&disequal, &equal_adjacency),
                model,
                manager,
            ) {
                continue;
            }
            let left_class = self.separation_class(left, &equal_adjacency, manager);
            if left_class.contains(&right) {
                continue;
            }
            self.separate_by_datatype_field(
                left,
                sort,
                &left_class,
                &asserted,
                decls,
                (&disequal, &equal_adjacency),
                model,
                manager,
            );
        }
    }

    /// `term`'s decided-equality class and its congruence-class mates.
    fn separation_class(
        &self,
        term: TermId,
        equal_adjacency: &FxHashMap<TermId, Vec<TermId>>,
        manager: &TermManager,
    ) -> Vec<TermId> {
        let mut class = equality_class(term, equal_adjacency);
        let sort = manager.get(term).map(|node| node.sort);
        for mate in self.dt_class_mates(term, sort, manager) {
            if !class.contains(&mate) {
                class.push(mate);
            }
        }
        class
    }

    /// Re-value one datatype-sorted field of `term`'s reconstructed value no
    /// assertion mentions, so that fewer of the search's decided
    /// disequalities hold between equal values, and no decided equality comes
    /// to hold between different ones (see the module docs).  `true` when a
    /// field moved.
    #[allow(clippy::too_many_arguments)]
    fn separate_by_datatype_field(
        &self,
        term: TermId,
        sort: SortId,
        class: &[TermId],
        asserted: &FxHashSet<TermId>,
        decls: &FxHashMap<SortId, DeclInfo>,
        (disequal, equal_adjacency): (&[(TermId, TermId)], &Adjacency),
        model: &mut Model,
        manager: &mut TermManager,
    ) -> bool {
        let Some(decl) = decls.get(&sort) else {
            return false;
        };
        // A class holding a ground constructor value is that value: moving it
        // would make the literal print another.
        if class
            .iter()
            .any(|&member| member != term && is_ground_dt_value(member, manager))
        {
            return false;
        }
        let Some(TermKind::DtConstructor { constructor, args }) = model
            .get(term)
            .and_then(|value| manager.get(value))
            .map(|node| node.kind.clone())
        else {
            return false;
        };
        let name = manager.resolve_str(constructor).to_string();
        let Some(index) = decl.constructors.iter().position(|c| c.name == name) else {
            return false;
        };
        let fields: Vec<(String, SortId)> = decl.constructors[index]
            .fields
            .iter()
            .map(|field| (field.selector.clone(), field.sort))
            .collect();
        if fields.len() != args.len() {
            return false;
        }
        let before = violated_disequalities(disequal, model);
        if before == 0 {
            return false;
        }
        let broken_before = broken_equalities(equal_adjacency, model);
        for (position, (selector, field_sort)) in fields.into_iter().enumerate() {
            if !manager.sorts.is_datatype(field_sort) {
                continue;
            }
            let accessor = manager.mk_dt_selector(&selector, term, field_sort);
            // A member that is a constructor application holds the field as
            // its own argument — whatever spelling the encoder gave that
            // argument (a purified one is no assertion's sub-term) — so the
            // field is not free, unless that argument is the field of a class
            // member itself (the reconstruction axiom's `(full (item b))`).
            let selector_spur = manager.intern_str(&selector);
            let pinned = class.iter().any(|&member| {
                let member_accessor = manager.mk_dt_selector(&selector, member, field_sort);
                if asserted.contains(&member_accessor) {
                    return true;
                }
                let Some(TermKind::DtConstructor {
                    args: member_args, ..
                }) = manager.get(member).map(|node| &node.kind)
                else {
                    return false;
                };
                let Some(&field) = member_args.get(position) else {
                    return true;
                };
                !matches!(
                    manager.get(field).map(|node| &node.kind),
                    Some(TermKind::DtSelector { selector: read, arg })
                        if *read == selector_spur && class.contains(arg)
                )
            });
            if pinned || asserted.contains(&accessor) {
                continue;
            }
            let Some(field_decl) = super::super::dt_axioms::resolve_decl(field_sort, manager)
            else {
                continue;
            };
            let current = args[position];
            for candidate_ctor in &field_decl.constructors {
                let mut field_args: Vec<TermId> = Vec::with_capacity(candidate_ctor.fields.len());
                for field in &candidate_ctor.fields {
                    match super::ground_default_term(manager, field.sort) {
                        Some(value) => field_args.push(value),
                        None => break,
                    }
                }
                if field_args.len() != candidate_ctor.fields.len() {
                    continue;
                }
                let candidate =
                    manager.mk_dt_constructor(&candidate_ctor.name, field_args, field_sort);
                if candidate == current {
                    continue;
                }
                let mut new_args = args.to_vec();
                new_args[position] = candidate;
                let value = manager.mk_dt_constructor(&name, new_args, sort);
                let mut trial = model.clone();
                trial.set(accessor, candidate);
                trial.set(term, value);
                for &member in class {
                    trial.set(member, value);
                }
                if violated_disequalities(disequal, &trial) >= before
                    || broken_equalities(equal_adjacency, &trial) > broken_before
                {
                    continue;
                }
                *model = trial;
                return true;
            }
        }
        false
    }

    /// Re-value one numeric field of `term`'s reconstructed value so that the
    /// value changes.
    ///
    /// Only fields of the outermost constructor are considered, and only
    /// `Int`/`Real` ones whose accessor occurs in *no* assertion — for every
    /// member of the equality class, so a field pinned through a term the search
    /// proved equal to `term` is left alone too.  That is the criterion that
    /// makes the repair safe: a term the assertions never mention has no
    /// user constraint on it at all (the only lemmas that speak about it are
    /// datatype axioms, which the repair moves *towards* satisfying), and it is
    /// also invisible to [`Solver::model_refutes_assertions`], which evaluates
    /// nothing but the assertions.  Note that the arithmetic solver's own
    /// `value()` is *not* a usable pin test: it reports a value for every
    /// tableau variable, constrained or not — reporting `0` for both `(fst p)`
    /// and `(fst q)` is exactly how the collision arises.
    ///
    /// A datatype all of whose scalar fields are pinned keeps its colliding
    /// value rather than acquiring a fabricated one.
    ///
    /// `class` is every term the search proved equal to `term`; all of them
    /// receive the new value so the repair cannot break an equality.
    ///
    /// `true` when a field moved.
    #[allow(clippy::too_many_arguments)]
    fn separate_dt_value(
        &self,
        term: TermId,
        sort: SortId,
        class: &[TermId],
        asserted: &FxHashSet<TermId>,
        decls: &FxHashMap<SortId, DeclInfo>,
        model: &mut Model,
        manager: &mut TermManager,
    ) -> bool {
        let Some(decl) = decls.get(&sort) else {
            return false;
        };
        let Some(TermKind::DtConstructor { constructor, args }) = model
            .get(term)
            .and_then(|value| manager.get(value))
            .cloned()
            .map(|node| node.kind)
        else {
            return false;
        };
        let name = manager.resolve_str(constructor).to_string();
        let Some(index) = decl.constructors.iter().position(|c| c.name == name) else {
            return false;
        };
        let fields: Vec<(String, SortId)> = decl.constructors[index]
            .fields
            .iter()
            .map(|field| (field.selector.clone(), field.sort))
            .collect();
        if fields.len() != args.len() {
            return false;
        }
        let int_sort = manager.sorts.int_sort;
        let real_sort = manager.sorts.real_sort;
        for (position, (selector, field_sort)) in fields.into_iter().enumerate() {
            if field_sort != int_sort && field_sort != real_sort {
                continue;
            }
            let accessor = manager.mk_dt_selector(&selector, term, field_sort);
            let pinned = class.iter().any(|&member| {
                let member_accessor = manager.mk_dt_selector(&selector, member, field_sort);
                asserted.contains(&member_accessor)
            });
            if pinned || asserted.contains(&accessor) {
                continue;
            }
            let current = args[position];
            let Some(bumped) = (1..=MAX_SEPARATION_OFFSET).find_map(|offset| {
                let bumped = match manager.get(current).map(|node| node.kind.clone()) {
                    Some(TermKind::IntConst(value)) => manager.mk_int(value + offset),
                    Some(TermKind::RealConst(value)) => {
                        manager.mk_real(value + num_rational::Rational64::from_integer(offset))
                    }
                    _ => return None,
                };
                self.keeps_negated_equalities(accessor, bumped, model, manager)
                    .then_some(bumped)
            }) else {
                continue;
            };
            let mut new_args = args.to_vec();
            new_args[position] = bumped;
            let value = manager.mk_dt_constructor(&name, new_args, sort);
            // Publish the field too, so `(get-value ((fst q)))` reports the very
            // number `(get-model)` printed inside `q`.
            model.set(accessor, bumped);
            model.set(term, value);
            for &member in class {
                model.set(member, value);
            }
            return true;
        }
        false
    }

    /// Whether giving `accessor` the value `bumped` makes no numeric equality
    /// the search decided **false** hold in the model — the one check the
    /// ground gate runs on atoms that are not assertions
    /// (`Solver::model_violates_negated_equality`).  A bump that broke one was
    /// refused by the gate as a falsification (`#P2b-86`).
    fn keeps_negated_equalities(
        &self,
        accessor: TermId,
        bumped: TermId,
        model: &Model,
        manager: &TermManager,
    ) -> bool {
        let mut trial = model.clone();
        trial.set(accessor, bumped);
        self.violated_negated_equalities(&trial, manager)
            <= self.violated_negated_equalities(model, manager)
    }

    /// How many numeric equality atoms the search decided `false` hold in
    /// `model` (both sides folding to one number).
    fn violated_negated_equalities(&self, model: &Model, manager: &TermManager) -> usize {
        let mut violated = 0usize;
        for (&var, constraint) in &self.var_to_constraint {
            let Constraint::Eq(lhs, rhs) = *constraint else {
                continue;
            };
            if self.sat.model_value(var) != LBool::False {
                continue;
            }
            let numeric = manager.get(lhs).is_some_and(|t| {
                t.sort == manager.sorts.int_sort || t.sort == manager.sorts.real_sort
            });
            if !numeric {
                continue;
            }
            if let (EvalOutcome::Value(EvalVal::Num(left)), EvalOutcome::Value(EvalVal::Num(right))) = (
                self.eval_in_model_outcome(lhs, model, manager, 0),
                self.eval_in_model_outcome(rhs, model, manager, 0),
            ) && left == right
            {
                violated += 1;
            }
        }
        violated
    }
}

/// How many of the decided disequalities `disequal` hold between two equal
/// model values.
fn violated_disequalities(disequal: &[(TermId, TermId)], model: &Model) -> usize {
    disequal
        .iter()
        .filter(
            |&&(left, right)| match (model.get(left), model.get(right)) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            },
        )
        .count()
}

/// How many of the decided equalities in `adjacency` hold between two
/// different model values.
fn broken_equalities(adjacency: &Adjacency, model: &Model) -> usize {
    let mut broken = 0usize;
    for (&left, neighbours) in adjacency {
        for &right in neighbours {
            if left < right
                && let (Some(a), Some(b)) = (model.get(left), model.get(right))
                && a != b
            {
                broken += 1;
            }
        }
    }
    broken
}

/// Whether `term` is a ground datatype value: a constructor applied to
/// literals and ground datatype values only.
fn is_ground_dt_value(term: TermId, manager: &TermManager) -> bool {
    let mut stack = vec![term];
    let mut steps = 0usize;
    while let Some(current) = stack.pop() {
        steps += 1;
        if steps > 4_096 {
            return false;
        }
        match manager.get(current).map(|node| &node.kind) {
            Some(TermKind::DtConstructor { args, .. }) => stack.extend(args.iter().copied()),
            Some(
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. },
            ) => {}
            _ => return false,
        }
    }
    true
}
