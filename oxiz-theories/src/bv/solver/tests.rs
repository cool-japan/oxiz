//! Unit tests for the bit-blasting bit-vector solver.
//!
//! Split out of `solver.rs` to keep that file under the 2000-line policy.

use super::*;
// `TheoryCombination` now lives in the sibling `combination` module, so the
// glob above no longer brings the trait into scope; the tests below call
// `get_shared_equalities` / `notify_equality` through it.
use crate::theory::TheoryCombination;

/// Regression (theories-bv, 811-line solver.rs refactor): a genuinely
/// satisfiable 8-bit multiplication + disjunction pattern must not return
/// a false `Unsat` after an earlier probe has run on the same solver.
///
/// Root cause: lazy hyper-binary resolution injected derived binary clauses
/// into the SAT database mid-`solve()`; those clauses were *not* tracked in
/// the learned-clause list, so `check()`'s per-probe `forget_learned_since`
/// cleanup (and the enclosing `pop()`) could not remove them. Left behind,
/// such a clause — implicitly resting on a since-retracted level-0 decision
/// installed by `assert_const` — spuriously forced `Unsat` on the next
/// probe. `embedded_sat_config()` now disables that heuristic (and
/// inprocessing) so the incremental cleanup contract stays exact.
///
/// Drives the same `a = x*3 ∧ a ≠ x ∧ (a = x ∨ a = 7)` disjunction as the
/// `oxiz-solver` integration test `bv_mul_aux_disjunction_const_is_sat_8bit`
/// via explicit `push`/`check`/`pop` branches: branch `a = x` is UNSAT, the
/// following branch `a = 7` is SAT (e.g. x=173: 173*3 = 519 ≡ 7 mod 256).
fn run_mul_disjunction_branches(width: u32) -> (TheoryResult, TheoryResult) {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);
    let three = TermId::new(2);
    let a = TermId::new(3);
    let seven = TermId::new(4);
    solver.new_bv(x, width);
    solver.assert_const(three, 3, width);
    solver.bv_mul(a, x, three);
    assert!(solver.assert_neq(a, x));

    // Disjunct 1: a = x  (=> UNSAT: x*3 = x with x != x is impossible).
    solver.push();
    assert!(solver.assert_eq(a, x));
    let r1 = solver.check().expect("check should succeed");
    solver.pop();

    // Disjunct 2: a = 7  (=> SAT, and must NOT be poisoned by disjunct 1).
    solver.push();
    solver.new_bv(seven, width);
    solver.assert_const(seven, 7, width);
    assert!(solver.assert_eq(a, seven));
    let r2 = solver.check().expect("check should succeed");
    solver.pop();
    (r1, r2)
}

#[test]
fn bv_mul_disjunction_incremental_stays_sat_4bit() {
    let (r1, r2) = run_mul_disjunction_branches(4);
    assert!(matches!(r1, TheoryResult::Unsat(_)), "a=x branch {r1:?}");
    assert!(matches!(r2, TheoryResult::Sat), "a=7 branch {r2:?}");
}

#[test]
fn bv_mul_disjunction_incremental_stays_sat_8bit() {
    let (r1, r2) = run_mul_disjunction_branches(8);
    assert!(matches!(r1, TheoryResult::Unsat(_)), "a=x branch {r1:?}");
    assert!(matches!(r2, TheoryResult::Sat), "a=7 branch {r2:?}");
}

#[test]
fn bv_mul_disjunction_single_check_is_sat_8bit() {
    // The same pattern collapsed into one probe: a = x*3, a != x, a = 7.
    // SAT with x=173 (173*3 = 519 ≡ 7 mod 256, and 7 != 173).
    let mut solver = BvSolver::new();
    let x = TermId::new(1);
    let three = TermId::new(2);
    let a = TermId::new(3);
    solver.new_bv(x, 8);
    solver.assert_const(three, 3, 8);
    solver.bv_mul(a, x, three);
    assert!(solver.assert_neq(a, x));
    solver.assert_const(a, 7, 8);
    let r = solver.check().expect("check should succeed");
    assert!(matches!(r, TheoryResult::Sat), "got {r:?}");
}

#[test]
fn test_bv_eq() {
    let mut solver = BvSolver::new();

    let a = TermId::new(1);
    let b = TermId::new(2);

    solver.new_bv(a, 8);
    solver.new_bv(b, 8);

    // a = 42
    solver.assert_const(a, 42, 8);

    // a = b
    assert!(solver.assert_eq(a, b));

    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    // b should be 42
    assert_eq!(solver.get_value(b), Some(42));
}

#[test]
fn test_bv_neq() {
    let mut solver = BvSolver::new();

    let a = TermId::new(1);
    let b = TermId::new(2);

    solver.new_bv(a, 4);
    solver.new_bv(b, 4);

    // a = 5
    solver.assert_const(a, 5, 4);
    // b = 5
    solver.assert_const(b, 5, 4);
    // a != b (contradiction)
    assert!(solver.assert_neq(a, b));

    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Unsat(_)));
}

// Audit regression (theories-bv): `BvSolver::check()` could return a
// false `Unsat` for a genuinely satisfiable "inverse" bit-vector
// constraint (fixed dividend/quotient, free divisor) because its own
// first internal `solve()` call could hit an unsound learned clause
// (resolved through a bare, clause-less level-0 decision literal
// installed by `assert_const`). `check()` now re-verifies an `Unsat`
// verdict by discarding this probe's learned clauses and retrying once
// before trusting it. This was previously worked around at the test
// level by `#[ignore]`ing `test_bv10_udiv` in `tests/test_bv10.rs`.
#[test]
fn audit_check_recovers_from_first_attempt_false_unsat() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let dividend = TermId::new(1);
    let divisor = TermId::new(2);
    let quotient = TermId::new(3);
    let result = TermId::new(4);

    solver.new_bv(dividend, width);
    solver.new_bv(divisor, width);
    solver.new_bv(quotient, width);
    solver.new_bv(result, width);

    solver.assert_const(dividend, 100, width);
    solver.assert_const(quotient, 5, width);
    solver.bv_udiv(result, dividend, divisor);
    assert!(solver.assert_eq(result, quotient));

    let outcome = solver.check().expect("check should succeed");
    assert!(
        matches!(outcome, TheoryResult::Sat),
        "100 / divisor = 5 is satisfiable (e.g. divisor in 17..=20); got {outcome:?}"
    );

    let d = solver
        .get_value(divisor)
        .expect("divisor should have a value");
    assert_eq!(100 / d, 5, "witness divisor {d} must satisfy 100/d = 5");
}

// Audit regression (theories-bv): `get_value` computed `1u64 << i` for
// every bit index, which panics (debug) or silently wraps to a wrong
// value (release) once a bit-vector is wider than 64 bits. It must now
// honestly report "unavailable" (`None`) instead, while a new
// `get_value_big` correctly returns the full-width value.
#[test]
fn get_value_returns_none_for_width_over_64_get_value_big_is_correct() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);

    solver.new_bv(a, 100);
    // Manually force bit 0 and bit 99 true, everything else false --
    // exercises the exact `i >= 64` shift that used to panic/wrap.
    let bv = solver
        .term_to_bv
        .get(&a)
        .cloned()
        .expect("bv var must exist after new_bv");
    for (i, &var) in bv.bits.iter().enumerate() {
        if i == 0 || i == 99 {
            solver.sat.add_clause([Lit::pos(var)]);
        } else {
            solver.sat.add_clause([Lit::neg(var)]);
        }
    }

    let result = solver.check().expect("test operation should succeed");
    assert!(matches!(result, TheoryResult::Sat));

    assert_eq!(
        solver.get_value(a),
        None,
        "get_value must honestly report unavailable for width > 64, not panic or wrap"
    );

    let mut expected = BigUint::ZERO;
    expected.set_bit(0, true);
    expected.set_bit(99, true);
    assert_eq!(
        solver.get_value_big(a),
        Some(expected),
        "get_value_big must return the correct full-width value"
    );
}

// Audit regression (theories-bv): `extract_model_equalities` (used for
// Nelson-Oppen model-based equality sharing) keyed its value map by
// `u64`, computed via the same panicking/wrapping `1u64 << i` shift for
// bit-vectors wider than 64 bits. It must now correctly detect equal
// wide values via a `BigUint`-keyed map instead.
#[test]
fn extract_model_equalities_handles_width_over_64() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);

    solver.new_bv(a, 70);
    solver.new_bv(b, 70);

    // Force both `a` and `b` to the same 70-bit value (bit 69 set).
    for &term in &[a, b] {
        let bv = solver
            .term_to_bv
            .get(&term)
            .cloned()
            .expect("bv var must exist after new_bv");
        for (i, &var) in bv.bits.iter().enumerate() {
            if i == 69 {
                solver.sat.add_clause([Lit::pos(var)]);
            } else {
                solver.sat.add_clause([Lit::neg(var)]);
            }
        }
    }

    assert!(matches!(solver.sat.solve(), SolverResult::Sat));

    // Must not panic (previously `1u64 << 69` would panic in debug /
    // silently wrap -- and possibly corrupt equality detection -- in
    // release).
    solver.extract_model_equalities();

    let shared = solver.get_shared_equalities();
    assert_eq!(
        shared.len(),
        1,
        "a and b share the same 70-bit value and must be reported equal"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Wide-constant pinning
// ─────────────────────────────────────────────────────────────────────────

/// `assert_const_big` must pin every bit of a value wider than 64 bits.
///
/// `assert_const` used to shift the value by the bit index, so a width-128
/// constant both aborted in debug builds (`shift >= 64`) and pinned the low
/// limb repeated across every limb in release ones.  The model read back here
/// is the exact constant, high limb included.
#[test]
fn assert_const_big_pins_every_bit_at_width_128() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);

    // 2^64 + 5: one bit in each limb, so a low-limb-only pin is detectable.
    let mut value = BigUint::ZERO;
    value.set_bit(64, true);
    value.set_bit(0, true);
    value.set_bit(2, true);

    assert!(solver.assert_const_big(x, &value, 128));
    assert!(matches!(
        solver.check().expect("check must succeed"),
        TheoryResult::Sat
    ));
    assert_eq!(solver.get_value_big(x), Some(value));
}

/// A `u64`-valued constant asserted at a width above 64 must zero-extend, not
/// repeat its limb.
#[test]
fn assert_const_zero_extends_above_64_bits() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);

    assert!(solver.assert_const(x, 5, 96));
    assert!(matches!(
        solver.check().expect("check must succeed"),
        TheoryResult::Sat
    ));
    assert_eq!(solver.get_value_big(x), Some(BigUint::from(5u32)));
}

/// Bits of the literal at or above the declared width are read modulo
/// `2^width`, exactly as SMT-LIB reads an out-of-range numeral.
#[test]
fn assert_const_big_truncates_to_declared_width() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);

    let mut value = BigUint::ZERO;
    value.set_bit(70, true);
    value.set_bit(3, true);

    assert!(solver.assert_const_big(x, &value, 8));
    assert!(matches!(
        solver.check().expect("check must succeed"),
        TheoryResult::Sat
    ));
    assert_eq!(solver.get_value(x), Some(0b0000_1000));
}

/// Pinning a constant at a width the term does not have changes nothing and
/// says so, instead of indexing past the end of the existing bit-vector.
#[test]
fn assert_const_reports_width_mismatch() {
    let mut solver = BvSolver::new();
    let x = TermId::new(1);
    solver.new_bv(x, 8);

    assert!(!solver.assert_const(x, 1, 16));
    assert!(solver.assert_const(x, 1, 8));
}

// ─────────────────────────────────────────────────────────────────────────
// Width mismatch is reported, never asserted
// ─────────────────────────────────────────────────────────────────────────

/// Every binary encoding must answer `false` for operands of different widths.
///
/// These used to be `assert_eq!(va.width, vb.width)`, so a mixed-width term —
/// which the term builder still accepts for `(bvadd x8 y16)` — aborted the
/// whole process instead of being reported as unencodable.
#[test]
fn binary_ops_report_width_mismatch_instead_of_aborting() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);
    let r = TermId::new(3);
    solver.new_bv(a, 8);
    solver.new_bv(b, 16);

    assert!(!solver.bv_add(r, a, b));
    assert!(!solver.bv_sub(r, a, b));
    assert!(!solver.bv_mul(r, a, b));
    assert!(!solver.bv_and(r, a, b));
    assert!(!solver.bv_or(r, a, b));
    assert!(!solver.bv_xor(r, a, b));
    assert!(!solver.bv_udiv(r, a, b));
    assert!(!solver.bv_sdiv(r, a, b));
    assert!(!solver.bv_urem(r, a, b));
    assert!(!solver.bv_srem(r, a, b));
    assert!(!solver.bv_shl(r, a, b));
    assert!(!solver.bv_lshr(r, a, b));
    assert!(!solver.bv_ashr(r, a, b));
    assert!(!solver.assert_eq(a, b));
    assert!(!solver.assert_neq(a, b));
    assert!(!solver.assert_ult(a, b));
    assert!(!solver.assert_ule(a, b));
    assert!(!solver.assert_slt(a, b));
    assert!(!solver.assert_sle(a, b));

    // Nothing was encoded for the mismatched pair.
    assert!(solver.get_bv(r).is_none());
}

/// The equal-width path still encodes and still reports success, so the honest
/// `false` above is not a blanket refusal.
#[test]
fn binary_ops_still_encode_at_equal_widths() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let b = TermId::new(2);
    let r = TermId::new(3);
    solver.new_bv(a, 8);
    solver.new_bv(b, 8);

    assert!(solver.bv_add(r, a, b));
    assert!(solver.assert_eq(a, b));
    assert!(solver.assert_ule(a, b));
    assert!(solver.get_bv(r).is_some());
}

/// An extraction whose range escapes the source vector is malformed; it must
/// be reported, not asserted.
#[test]
fn extract_reports_out_of_range_instead_of_aborting() {
    let mut solver = BvSolver::new();
    let a = TermId::new(1);
    let r = TermId::new(2);
    solver.new_bv(a, 8);

    assert!(!solver.extract(r, a, 8, 0), "high bit is past the source");
    assert!(!solver.extract(r, a, 2, 5), "low above high is not a range");
    assert!(solver.extract(r, a, 5, 2));
}
