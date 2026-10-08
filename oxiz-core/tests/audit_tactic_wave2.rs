//! Wave-2 tactic soundness / model-converter regression tests.
//!
//! Covers:
//! - P4-1103: Ackermannization must not collapse function applications whose
//!   arguments are quantifier-bound (unsound); such symbols are excluded.
//! - P4-1104: Fourier-Motzkin must not report Sat via a real-domain
//!   elimination over integer variables (real-feasible but int-infeasible).
//! - P4-1109: variable-eliminating tactics expose a `ModelConverter` that
//!   reconstructs eliminated variables (solve-eqs) or drops the fresh
//!   Ackermann variables (ackermannize) so a sub-goal model lifts to the
//!   original goal.

use oxiz_core::ast::TermManager;
use oxiz_core::tactic::{
    AckermannizeTactic, FourierMotzkinTactic, Goal, SolveEqsTactic, SolveResult, TacticModel,
    TacticResult,
};

// ─────────────────────────────────────────────────────────────────────────────
// P4-1103: Ackermannization + quantifier-bound arguments
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn ackermann_ground_applications_are_transformed() {
    // Two ground applications of `f` must be Ackermannized into fresh
    // variables plus a functional-consistency constraint.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let a = manager.mk_var("a", int_sort);
    let b = manager.mk_var("b", int_sort);
    let zero = manager.mk_int(0);
    let fa = manager.mk_apply("f", [a], int_sort);
    let fb = manager.mk_apply("f", [b], int_sort);
    let g1 = manager.mk_ge(fa, zero);
    let g2 = manager.mk_ge(fb, zero);
    let goal = Goal::new(vec![g1, g2]);

    let mut tactic = AckermannizeTactic::new(&mut manager);
    let result = tactic.apply_mut(&goal).expect("apply_mut must not error");
    match result {
        TacticResult::SubGoals(goals) => {
            assert_eq!(goals.len(), 1);
            // Original 2 assertions + at least one congruence constraint.
            assert!(
                goals[0].assertions.len() >= 3,
                "expected a functional-consistency constraint to be added"
            );
        }
        other => panic!("expected SubGoals for ground applications, got {other:?}"),
    }
}

#[test]
fn ackermann_skips_function_with_only_quantified_occurrence() {
    // `forall x. f(x) >= 0`: the sole application of `f` depends on the bound
    // variable `x`, so it must NOT be Ackermannized (that would collapse a
    // family of values into one ground variable). NotApplicable is the honest
    // outcome.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let zero = manager.mk_int(0);
    let fx = manager.mk_apply("f", [x], int_sort);
    let body = manager.mk_ge(fx, zero);
    let forall = manager.mk_forall([("x", int_sort)], body);
    let goal = Goal::new(vec![forall]);

    let mut tactic = AckermannizeTactic::new(&mut manager);
    let result = tactic.apply_mut(&goal).expect("apply_mut must not error");
    assert!(
        matches!(result, TacticResult::NotApplicable),
        "a function applied only to bound variables must not be Ackermannized, got {result:?}"
    );
}

#[test]
fn ackermann_excludes_symbol_with_mixed_ground_and_quantified_occurrence() {
    // `f(a) >= 0  AND  (forall x. f(x) <= 10)`: `f` occurs both ground and
    // quantifier-bound. Ackermannizing only the ground `f(a)` while leaving
    // the quantified `f(x)` as the real `f` would decouple the fresh variable
    // from `f` (unsound), so `f` must be excluded entirely -> NotApplicable.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let a = manager.mk_var("a", int_sort);
    let x = manager.mk_var("x", int_sort);
    let zero = manager.mk_int(0);
    let ten = manager.mk_int(10);

    let fa = manager.mk_apply("f", [a], int_sort);
    let ground = manager.mk_ge(fa, zero); // f(a) >= 0 (ground)

    let fx = manager.mk_apply("f", [x], int_sort);
    let le = manager.mk_le(fx, ten); // f(x) <= 10
    let forall = manager.mk_forall([("x", int_sort)], le);

    let goal = Goal::new(vec![ground, forall]);

    let mut tactic = AckermannizeTactic::new(&mut manager);
    let result = tactic.apply_mut(&goal).expect("apply_mut must not error");
    assert!(
        matches!(result, TacticResult::NotApplicable),
        "a symbol with any quantified occurrence must be fully excluded, got {result:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// P4-1104: Fourier-Motzkin over integer variables
// ─────────────────────────────────────────────────────────────────────────────

/// Build the two-inequality encoding of `2x = c`: `2x <= c AND 2x >= c`.
fn two_x_equals(manager: &mut TermManager, x: oxiz_core::ast::TermId, c: i64) -> Goal {
    let two = manager.mk_int(2);
    let cc = manager.mk_int(c);
    let two_x = manager.mk_mul([two, x]);
    let le = manager.mk_le(two_x, cc); // 2x <= c
    let ge = manager.mk_ge(two_x, cc); // 2x >= c
    Goal::new(vec![le, ge])
}

#[test]
fn fm_integer_2x_equals_1_is_not_sat() {
    // 1 <= 2x <= 1 over the INTEGERS has no solution (x = 1/2 is real-only).
    // Fourier-Motzkin (a real-domain method) must NOT report Sat here.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let goal = two_x_equals(&mut manager, x, 1);

    let mut tactic = FourierMotzkinTactic::new(&mut manager);
    let result = tactic.apply_mut(&goal).expect("apply_mut must not error");
    assert!(
        !matches!(result, TacticResult::Solved(SolveResult::Sat)),
        "2x = 1 over Int must not be reported Sat, got {result:?}"
    );
}

#[test]
fn fm_real_2x_equals_1_is_sat() {
    // Over the REALS, 1 <= 2x <= 1 is satisfiable (x = 1/2). The integer
    // guard must not over-restrict real variables.
    let mut manager = TermManager::new();
    let real_sort = manager.sorts.real_sort;
    let x = manager.mk_var("x", real_sort);
    let goal = two_x_equals(&mut manager, x, 1);

    let mut tactic = FourierMotzkinTactic::new(&mut manager);
    let result = tactic.apply_mut(&goal).expect("apply_mut must not error");
    assert!(
        matches!(result, TacticResult::Solved(SolveResult::Sat)),
        "2x = 1 over Real must be Sat, got {result:?}"
    );
}

#[test]
fn fm_integer_unit_coefficient_bounds_still_eliminated() {
    // 1 <= x <= 5 over the integers is satisfiable and exactly eliminable
    // (unit coefficients), so the exact-shadow path must still conclude Sat.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let one = manager.mk_int(1);
    let five = manager.mk_int(5);
    let lower = manager.mk_ge(x, one); // x >= 1
    let upper = manager.mk_le(x, five); // x <= 5
    let goal = Goal::new(vec![lower, upper]);

    let mut tactic = FourierMotzkinTactic::new(&mut manager);
    let result = tactic.apply_mut(&goal).expect("apply_mut must not error");
    assert!(
        matches!(result, TacticResult::Solved(SolveResult::Sat)),
        "exactly-eliminable integer bounds must still conclude Sat, got {result:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// P4-1109: model converters
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn solve_eqs_converter_reconstructs_constant_eliminated_var() {
    // Goal: x = 5 AND x < y. solve-eqs removes `x = 5` and substitutes,
    // leaving `5 < y`. A model {y = 6} of the sub-goal must convert to a model
    // of the original with x reconstructed to 5.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let five = manager.mk_int(5);
    let eq = manager.mk_eq(x, five); // x = 5
    let lt = manager.mk_lt(x, y); // x < y
    let goal = Goal::new(vec![eq, lt]);

    let (result, converter) = {
        let mut tactic = SolveEqsTactic::new(&mut manager);
        tactic
            .apply_mut_with_converter(&goal)
            .expect("apply must not error")
    };
    assert!(matches!(result, TacticResult::SubGoals(_)));
    let converter = converter.expect("solve-eqs must yield a converter for SubGoals");

    let six = manager.mk_int(6);
    let mut sub_model = TacticModel::new();
    sub_model.set(y, six);

    let full = converter.convert(&sub_model, &mut manager);
    assert_eq!(full.get(y), Some(six), "original y value must be preserved");
    assert_eq!(
        full.get(x),
        Some(five),
        "eliminated x must be reconstructed to 5"
    );
}

#[test]
fn solve_eqs_converter_reconstructs_expression_eliminated_var() {
    // Goal: x = y + 1 AND x >= 3. solve-eqs removes `x = y + 1`. A model
    // {y = 5} must reconstruct x = y + 1 = 6 by evaluating the defining
    // expression under the model.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let one = manager.mk_int(1);
    let three = manager.mk_int(3);
    let y_plus_1 = manager.mk_add([y, one]);
    let eq = manager.mk_eq(x, y_plus_1); // x = y + 1
    let ge = manager.mk_ge(x, three); // x >= 3
    let goal = Goal::new(vec![eq, ge]);

    let (result, converter) = {
        let mut tactic = SolveEqsTactic::new(&mut manager);
        tactic
            .apply_mut_with_converter(&goal)
            .expect("apply must not error")
    };
    assert!(matches!(result, TacticResult::SubGoals(_)));
    let converter = converter.expect("solve-eqs must yield a converter for SubGoals");

    let five = manager.mk_int(5);
    let mut sub_model = TacticModel::new();
    sub_model.set(y, five);

    let full = converter.convert(&sub_model, &mut manager);
    let six = manager.mk_int(6);
    assert_eq!(
        full.get(x),
        Some(six),
        "x must be reconstructed as y + 1 = 6"
    );
}

#[test]
fn ackermann_converter_drops_fresh_variables() {
    // Ackermannizing two ground applications of `f` introduces fresh `!ack_k`
    // variables. The converter must drop those from a sub-goal model (they are
    // not part of the original signature) while preserving the original
    // variables' values.
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let a = manager.mk_var("a", int_sort);
    let b = manager.mk_var("b", int_sort);
    let zero = manager.mk_int(0);
    let fa = manager.mk_apply("f", [a], int_sort);
    let fb = manager.mk_apply("f", [b], int_sort);
    let g1 = manager.mk_ge(fa, zero);
    let g2 = manager.mk_ge(fb, zero);
    let goal = Goal::new(vec![g1, g2]);

    let (result, converter) = {
        let mut tactic = AckermannizeTactic::new(&mut manager);
        tactic
            .apply_mut_with_converter(&goal)
            .expect("apply must not error")
    };
    assert!(matches!(result, TacticResult::SubGoals(_)));
    let converter = converter.expect("ackermannize must yield a converter for SubGoals");

    // Fresh variables are interned by name+sort, so re-creating them yields
    // the same TermIds the tactic allocated.
    let ack0 = manager.mk_var(&oxiz_core::smtlib::reserved_name("ack", "0"), int_sort);
    let ack1 = manager.mk_var(&oxiz_core::smtlib::reserved_name("ack", "1"), int_sort);
    let v_a = manager.mk_int(11);
    let v_b = manager.mk_int(22);
    let v_ack = manager.mk_int(33);

    let mut sub_model = TacticModel::new();
    sub_model.set(a, v_a);
    sub_model.set(b, v_b);
    sub_model.set(ack0, v_ack);
    sub_model.set(ack1, v_ack);

    let full = converter.convert(&sub_model, &mut manager);
    assert_eq!(full.get(a), Some(v_a), "original var a must be preserved");
    assert_eq!(full.get(b), Some(v_b), "original var b must be preserved");
    assert_eq!(full.get(ack0), None, "fresh Ackermann var must be dropped");
    assert_eq!(full.get(ack1), None, "fresh Ackermann var must be dropped");
}
