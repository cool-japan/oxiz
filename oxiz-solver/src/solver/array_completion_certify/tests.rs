//! White-box tests for the completion hook at `check`'s exit (decisions (40)
//! and (41)): what a certified completion replaces in the published model,
//! and that a failed one replaces nothing.  The end-to-end verdict and
//! model-replay pins live in `oxiz-solver/tests/round4_pass12_completion_pins.rs`
//! and in the inverted `round4_pass6` / `round4_pass11` recheck pins.

use super::*;
use oxiz_core::ast::TermManager;

/// `∀i:(_ BitVec 6). a[i] = ((_ extract 0 0) i)` — satisfiable, decided by
/// `finite_expand` (64 points), and satisfied by no default-plus-pins
/// interpretation the goal names a point for: the goal spells no index
/// literal at all and `a` must alternate.
fn alternating_goal(tm: &mut TermManager) -> (Solver, TermId) {
    let mut solver = Solver::new();
    let index = tm.sorts.bitvec(6);
    let element = tm.sorts.bitvec(1);
    let array_sort = tm.sorts.array(index, element);
    let a = tm.mk_var("a", array_sort);
    let i = tm.mk_var("i", index);
    let read = tm.mk_select(a, i);
    let low_bit = tm.mk_bv_extract(0, 0, i);
    let body = tm.mk_eq(read, low_bit);
    let universal = tm.mk_forall([("i", index)], body);
    solver.assert(universal, tm);
    (solver, a)
}

#[test]
fn a_certificate_that_cannot_pass_leaves_the_model_untouched() {
    let mut tm = TermManager::new();
    let (mut solver, _a) = alternating_goal(&mut tm);
    assert_eq!(solver.check(&mut tm), SolverResult::Sat);
    let before = solver.model.as_ref().map(|m| m.assignments().clone());
    assert!(
        before.is_some(),
        "the expansion decides this goal and publishes a model"
    );
    assert!(
        !solver.certify_sat_by_array_completion(&mut tm),
        "no constant and no default-plus-pins interpretation satisfies an \
         alternating array, so the certificate must not pass"
    );
    let after = solver.model.as_ref().map(|m| m.assignments().clone());
    assert_eq!(
        before, after,
        "a failed or declined certificate must leave the published model \
         exactly as it was (decision (40))"
    );
}

#[test]
fn an_installed_completion_drops_every_entry_derived_from_the_old_interpretation() {
    let mut tm = TermManager::new();
    let mut solver = Solver::new();
    let index = tm.sorts.bitvec(7);
    let bool_sort = tm.sorts.bool_sort;
    let array_sort = tm.sorts.array(index, bool_sort);
    let a = tm.mk_var("a", array_sort);
    let p = tm.mk_var("p", bool_sort);
    let two = tm.mk_bitvec(2u8, 7);
    let read_two = tm.mk_select(a, two);
    let ground = tm.mk_or(vec![p, read_two]);
    let i = tm.mk_var("i", index);
    let read_i = tm.mk_select(a, i);
    let universal = tm.mk_forall([("i", index)], read_i);
    solver.assert(ground, &mut tm);
    solver.assert(universal, &mut tm);
    assert_eq!(solver.check(&mut tm), SolverResult::Sat);
    let Some(model) = solver.model.as_ref() else {
        panic!("a `sat` must publish a model");
    };
    let installed = model.get(a);
    assert!(
        installed.is_some_and(|value| is_array_value(value, &tm)),
        "`a` must be bound to the certified array value, got {installed:?}"
    );
    let completed: FxHashSet<TermId> = [a].into_iter().collect();
    let stale: Vec<TermId> = model
        .assignments()
        .keys()
        .copied()
        .filter(|&term| term != a && mentions_any(term, &completed, &tm))
        .collect();
    assert!(
        stale.is_empty(),
        "no entry derived from the replaced interpretation may survive the \
         install: {stale:?}"
    );
}

#[test]
fn an_array_value_is_a_literal_store_chain_over_a_literal_constant() {
    let mut tm = TermManager::new();
    let index = tm.sorts.bitvec(7);
    let element = tm.sorts.bitvec(1);
    let array_sort = tm.sorts.array(index, element);
    let one = tm.mk_bitvec(1u8, 1);
    let zero = tm.mk_bitvec(0u8, 1);
    let k = tm.mk_bitvec(5u8, 7);
    let constant = const_array(array_sort, one, &mut tm);
    let chain = tm.mk_store(constant, k, zero);
    assert!(is_array_value(constant, &tm));
    assert!(is_array_value(chain, &tm));
    let x = tm.mk_var("x", element);
    let symbolic_default = const_array(array_sort, x, &mut tm);
    let symbolic_value = tm.mk_store(constant, k, x);
    let a = tm.mk_var("a", array_sort);
    let over_a_variable = tm.mk_store(a, k, zero);
    assert!(
        !is_array_value(symbolic_default, &tm)
            && !is_array_value(symbolic_value, &tm)
            && !is_array_value(over_a_variable, &tm)
            && !is_array_value(a, &tm),
        "only a closed, literal interpretation is an installed array value"
    );
}
