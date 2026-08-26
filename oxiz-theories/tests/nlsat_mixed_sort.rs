//! Regression tests for two nonlinear-arithmetic dispatch soundness bugs in
//! `oxiz_theories::nlsat`:
//!
//! 1. **PARITY-QF_NIRA-01** — mixed Int/Real nonlinear problems used to force
//!    *every* variable to Integer (the global `integer_mode` flag), so a
//!    genuinely-Real variable that had to take a non-integral value produced a
//!    spurious UNSAT. The fix assigns integrality per variable *sort*.
//!
//! 2. **nia-nra-dispatch-drops-atoms-trusts-sat** — the dispatch used to drop
//!    any top-level term that is not a pure conjunction of translatable
//!    polynomial atoms (e.g. a disjunction) and still trust a `Sat` verdict on
//!    the resulting relaxed subproblem. The fix flags the extraction as
//!    incomplete and refuses to certify Sat (falls through to CDCL(T)).
//!
//! Verdicts are cross-checked against Z3 semantics conceptually; no external
//! solver is invoked.
//!
//! ## Feature gate
//!
//! The whole file exercises `oxiz_theories::nlsat`, which exists only under the
//! `nlsat` feature (on by default) -- that feature is what pulls the
//! `oxiz-nlsat` crate into the graph. A `--no-default-features --features std`
//! build has no such module, so this file compiles to nothing there.
#![cfg(feature = "nlsat")]

use num_rational::Rational64;
use oxiz_core::ast::TermManager;
use oxiz_theories::nlsat::{NlDispatchResult, dispatch_nia_constraints, dispatch_nra_constraints};

// ── PARITY-QF_NIRA-01: mixed Int/Real must not force Real vars to Integer ────

/// `(* x x) = 4 ∧ y = 1.5` with `x : Int`, `y : Real`.
///
/// The unique constraint on `y` is a non-integral real value, so the problem
/// is SAT (x = ±2, y = 1.5). Before the fix the QF_NIRA routing (integer_mode
/// = true) forced `y` to be an integer, making `y = 1.5` unsatisfiable and the
/// whole system a false UNSAT.
#[test]
fn test_qf_nira_int_square_with_real_half_is_sat() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let real_sort = manager.sorts.real_sort;

    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", real_sort);

    let square = manager.mk_mul(vec![x, x]);
    let four = manager.mk_int(4);
    let eq_sq = manager.mk_eq(square, four);

    let three_halves = manager.mk_real(Rational64::new(3, 2)); // 1.5
    let eq_y = manager.mk_eq(y, three_halves);

    // integer_mode = true mirrors the QF_NIRA routing in the solver.
    let result = dispatch_nia_constraints(&[eq_sq, eq_y], &manager, true);

    assert_ne!(
        result,
        Some(NlDispatchResult::Unsat),
        "mixed Int/Real (x*x=4 ∧ y=1.5) must not be reported UNSAT — Real y must stay real"
    );
    assert!(
        matches!(result, Some(NlDispatchResult::Sat { .. })),
        "mixed Int/Real (x*x=4 ∧ y=1.5) is satisfiable (x=±2, y=1.5)"
    );
}

/// `(* y y) = 4 ∧ 0 < x ∧ x < 1` with `y : Int`, `x : Real`.
///
/// Here the Real variable `x` is confined to the open interval `(0, 1)`, which
/// contains no integer, so the instance is SAT *only* because `x` may be
/// non-integral (e.g. x = 1/2, y = 2). Forcing `x` to Integer would wrongly
/// make it UNSAT.
#[test]
fn test_qf_nira_sat_requires_non_integral_real() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let real_sort = manager.sorts.real_sort;

    let x = manager.mk_var("x", real_sort);
    let y = manager.mk_var("y", int_sort);

    // y*y = 4 supplies the nonlinearity so the NIA dispatch engages.
    let y_sq = manager.mk_mul(vec![y, y]);
    let four = manager.mk_int(4);
    let eq_y_sq = manager.mk_eq(y_sq, four);

    // 0 < x < 1  →  x must be strictly between two integers.
    let zero = manager.mk_int(0);
    let one = manager.mk_int(1);
    let x_gt_0 = manager.mk_gt(x, zero);
    let x_lt_1 = manager.mk_lt(x, one);

    let result = dispatch_nia_constraints(&[eq_y_sq, x_gt_0, x_lt_1], &manager, true);

    assert_ne!(
        result,
        Some(NlDispatchResult::Unsat),
        "a Real x in (0,1) must not be rejected as if it were an integer"
    );
    assert!(
        matches!(result, Some(NlDispatchResult::Sat { .. })),
        "y*y=4 ∧ 0<x<1 is satisfiable (y=±2, x non-integral)"
    );
}

/// Control: a *pure* QF_NIA square (all Int) must still be reported SAT — the
/// per-sort change must not regress the genuinely-integer path.
#[test]
fn test_qf_nia_pure_integer_square_still_sat() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let square = manager.mk_mul(vec![x, x]);
    let four = manager.mk_int(4);
    let eq = manager.mk_eq(square, four);

    let result = dispatch_nia_constraints(&[eq], &manager, true);
    assert!(
        matches!(result, Some(NlDispatchResult::Sat { .. })),
        "x*x=4 with x:Int is SAT (x=±2)"
    );
}

// ── nia-nra-dispatch-drops-atoms-trusts-sat: no silent drop → no false Sat ───

/// NIA: `(* x y) = 12 ∧ (x = 100 ∨ y = 100)` with `x, y : Int`.
///
/// This is UNSAT: if x = 100 then y = 12/100 (non-integer); if y = 100 then
/// x = 12/100 (non-integer). The disjunction is *not* a conjunction of
/// polynomial atoms, so the old extractor dropped it and certified SAT on the
/// relaxed `x*y = 12` (which is satisfiable at x=12, y=1). The fix must refuse
/// to report SAT for the relaxed problem.
#[test]
fn test_nia_dropped_disjunction_does_not_fabricate_sat() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);

    let prod = manager.mk_mul(vec![x, y]);
    let twelve = manager.mk_int(12);
    let eq_prod = manager.mk_eq(prod, twelve);

    let hundred = manager.mk_int(100);
    let x_eq_100 = manager.mk_eq(x, hundred);
    let y_eq_100 = manager.mk_eq(y, hundred);
    let disj = manager.mk_or(vec![x_eq_100, y_eq_100]);

    let result = dispatch_nia_constraints(&[eq_prod, disj], &manager, true);

    assert!(
        !matches!(result, Some(NlDispatchResult::Sat { .. })),
        "a dropped disjunction must not let the relaxed x*y=12 be certified SAT"
    );
}

/// NRA: `(* x x) = 4 ∧ (x = 5 ∨ x = 7)` with `x : Real`.
///
/// UNSAT (x = ±2 contradicts x ∈ {5, 7}). Dropping the disjunction leaves the
/// satisfiable `x*x = 4`; the fix must not report SAT on that relaxation.
#[test]
fn test_nra_dropped_disjunction_does_not_fabricate_sat() {
    let mut manager = TermManager::new();
    let real_sort = manager.sorts.real_sort;
    let x = manager.mk_var("x", real_sort);

    let square = manager.mk_mul(vec![x, x]);
    let four = manager.mk_int(4);
    let eq_sq = manager.mk_eq(square, four);

    let five = manager.mk_int(5);
    let seven = manager.mk_int(7);
    let x_eq_5 = manager.mk_eq(x, five);
    let x_eq_7 = manager.mk_eq(x, seven);
    let disj = manager.mk_or(vec![x_eq_5, x_eq_7]);

    let result = dispatch_nra_constraints(&[eq_sq, disj], &manager);

    assert!(
        !matches!(result, Some(NlDispatchResult::Sat { .. })),
        "a dropped disjunction must not let the relaxed x*x=4 be certified SAT"
    );
}

/// NIA: an assertion containing an untranslatable operand (integer `div`) must
/// likewise not be silently dropped into a false SAT.
///
/// `(* x x) = 4 ∧ (div x y) = 3` — the `div` atom does not translate to a
/// polynomial, so the extractor must flag the problem incomplete and refuse to
/// certify SAT on the relaxed `x*x = 4`.
#[test]
fn test_nia_untranslatable_operand_does_not_fabricate_sat() {
    let mut manager = TermManager::new();
    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);

    let square = manager.mk_mul(vec![x, x]);
    let four = manager.mk_int(4);
    let eq_sq = manager.mk_eq(square, four);

    let div_xy = manager.mk_div(x, y);
    let three = manager.mk_int(3);
    let eq_div = manager.mk_eq(div_xy, three);

    let result = dispatch_nia_constraints(&[eq_sq, eq_div], &manager, true);

    assert!(
        !matches!(result, Some(NlDispatchResult::Sat { .. })),
        "an untranslatable div atom must not be dropped into a false SAT"
    );
}
