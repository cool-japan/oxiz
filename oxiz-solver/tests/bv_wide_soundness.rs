//! Regression tests for bit-vectors wider than 64 bits.
//!
//! Everything here failed before the wide-BV fixes, and every failure had the
//! same root cause: a 64-bit window onto an arbitrary-width value.
//!
//! | Script                                                   | Was      | Must be |
//! |----------------------------------------------------------|----------|---------|
//! | `x = 2^64` ∧ `x <u 1` at width 128                        | `sat`    | `unsat` |
//! | `a = 0`, `b = 2^64`, `(distinct (g a) (g b))` at 128      | `unsat`  | `sat`   |
//! | `(get-value (x))` for a 96-bit `x` pinned to all-ones     | `#x00…0` | exact   |
//!
//! 1. The bit-blaster pinned a `BitVecConst` from `iter_u64_digits().next()`,
//!    so a 128-bit constant became its low limb: `2^64` was asserted as `0`,
//!    which really is `<u 1` — a false `sat` in release builds, and a shift
//!    overflow abort in debug ones.
//! 2. The EUF canonicalisation of BV constants keyed on `(low_64_bits, width)`,
//!    so `0` and `2^64` shared a key at width 128 and were *merged* — with the
//!    merge recorded as tautological.  Congruence then made `(g a)` and `(g b)`
//!    the same node and the disequality became a conflict: a false `unsat`.
//! 3. The model builder read `BvSolver::get_value`, which is `None` above 64
//!    bits, so every wide bit-vector fell through to a default of `0`.
//!
//! The equal-value direction of (2) is pinned as well: two *distinct* term ids
//! holding the same wide constant must still be merged, or the fix would have
//! traded a false `unsat` for a lost congruence.

use oxiz_solver::{Context, SolverResult};

/// Run a single SMT-LIB2 script and return the solver result.
///
/// The verdict is the last `sat` / `unsat` / `unknown` token in the output.
fn run_script(script: &str) -> SolverResult {
    let mut ctx = Context::new();
    let outputs = ctx.execute_script(script).unwrap_or_default();
    for tok in outputs.iter().rev() {
        match tok.trim() {
            "sat" => return SolverResult::Sat,
            "unsat" => return SolverResult::Unsat,
            "unknown" => return SolverResult::Unknown,
            _ => {}
        }
    }
    SolverResult::Unknown
}

/// Every output line of a script, so `(get-value ...)` can be inspected.
fn run_script_output(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script).unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────────────
// 1. Wide constants must be pinned at their full width
// ─────────────────────────────────────────────────────────────────────────

/// `x = 2^64 ∧ x <u 1` at width 128.  Truncating the constant to its low limb
/// pinned `x = 0`, which satisfies `x <u 1`: a false `sat`.
#[test]
fn wide_const_high_limb_is_pinned_128bit() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 128))
(assert (= x (_ bv18446744073709551616 128)))
(assert (bvult x (_ bv1 128)))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// The satisfiable twin: nothing about the fix may turn a real `sat` into
/// `unsat`.  `2^64 <u 2^64 + 1` holds at width 128.
#[test]
fn wide_const_high_limb_stays_sat_when_satisfiable_128bit() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 128))
(assert (= x (_ bv18446744073709551616 128)))
(assert (bvult x (_ bv18446744073709551617 128)))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// Two wide constants that differ *only* above bit 64 must be unequal.
#[test]
fn wide_consts_differing_above_bit_64_are_unequal() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 128))
(assert (= x (_ bv0 128)))
(assert (= x (_ bv18446744073709551616 128)))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

// ─────────────────────────────────────────────────────────────────────────
// 2. EUF canonicalisation of wide BV constants
// ─────────────────────────────────────────────────────────────────────────

/// Distinct wide constants sharing their low 64 bits must stay in distinct EUF
/// classes, so `(distinct (g a) (g b))` is satisfiable.
#[test]
fn distinct_wide_consts_are_not_merged_in_euf() {
    let script = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 128)) (_ BitVec 8))
(declare-const a (_ BitVec 128))
(declare-const b (_ BitVec 128))
(assert (= a (_ bv0 128)))
(assert (= b (_ bv18446744073709551616 128)))
(assert (distinct (g a) (g b)))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Sat);
}

/// The other direction: *equal* wide constants must still be merged, so
/// congruence closure still derives `(g a) = (g b)` and the disequality is a
/// conflict.  Without this the full-value key would have cost a real `unsat`.
#[test]
fn equal_wide_consts_still_merge_in_euf() {
    let script = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 128)) (_ BitVec 8))
(declare-const a (_ BitVec 128))
(declare-const b (_ BitVec 128))
(assert (= a (_ bv18446744073709551616 128)))
(assert (= b (_ bv18446744073709551616 128)))
(assert (distinct (g a) (g b)))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// The same congruence conflict one limb up, so the discriminating bits sit
/// above 64 on *both* sides of the key.
#[test]
fn wide_const_congruence_conflict_above_bit_64() {
    let script = "\
(set-logic QF_UFBV)
(declare-fun g ((_ BitVec 128)) (_ BitVec 8))
(declare-const a (_ BitVec 128))
(declare-const b (_ BitVec 128))
(assert (= a (_ bv36893488147419103232 128)))
(assert (= b (_ bv36893488147419103232 128)))
(assert (distinct (g a) (g b)))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

// ─────────────────────────────────────────────────────────────────────────
// 3. Wide model values
// ─────────────────────────────────────────────────────────────────────────

/// A 96-bit variable pinned to all-ones must be *reported* as all-ones; the
/// model used to read `0` because `get_value` gives up above 64 bits.
#[test]
fn wide_bv_model_value_round_trips_96bit() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 96))
(assert (= x (_ bv79228162514264337593543950335 96)))
(check-sat)
(get-value (x))
";
    let outputs = run_script_output(script);
    let printed = outputs.join("\n");
    assert!(
        printed.contains("#xffffffffffffffffffffffff"),
        "96-bit model value must be the pinned constant, got: {printed}"
    );
}

/// A value whose *only* set bit lives above 64: the low limb is zero, so a
/// 64-bit read cannot tell it apart from `0`.
#[test]
fn wide_bv_model_value_high_limb_only_128bit() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 128))
(assert (= x (_ bv18446744073709551616 128)))
(check-sat)
(get-value (x))
";
    let outputs = run_script_output(script);
    let printed = outputs.join("\n");
    assert!(
        printed.contains("#x00000000000000010000000000000000"),
        "128-bit model value must carry the high limb, got: {printed}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 4. Wide BV structure must not abort or answer falsely
// ─────────────────────────────────────────────────────────────────────────

/// `2^64 + 2^64 = 2^65` at width 128 — a carry that crosses the limb boundary.
#[test]
fn wide_bv_add_crosses_limb_boundary_128bit() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 128))
(assert (= x (_ bv18446744073709551616 128)))
(assert (not (= (bvadd x x) (_ bv36893488147419103232 128))))
(check-sat)
";
    assert_eq!(run_script(script), SolverResult::Unsat);
}

/// A mixed-width application is malformed, but it must never abort the process
/// and must never be answered `unsat` on the strength of a bogus circuit.
#[test]
fn mixed_width_bv_op_does_not_abort() {
    let script = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 8))
(declare-const y (_ BitVec 16))
(assert (= (bvadd x y) y))
(check-sat)
";
    assert_ne!(run_script(script), SolverResult::Unsat);
}

// ─────────────────────────────────────────────────────────────────────────
// 5. `encode_add_const` must read bit `i >= 64` of a `u64` as zero (U-Z11)
// ─────────────────────────────────────────────────────────────────────────
//
// `oxiz-theories/src/bv/solver.rs::encode_add_const` used to read the bit of
// its `u64` constant as `((constant >> i) & 1) == 1` with `i` running over the
// bit-vector width.  Rust's `>>` on a `u64` keeps only the low 6 bits of the
// shift amount, so for `i >= 64` bit `i` was read as bit `i % 64` (release) or
// the process aborted with "attempt to shift right with overflow" (debug).
// All three call sites pass `constant = 1` as the `+1` of a two's-complement
// negation (`bv_neg`, `bv_sub`, and the `~a + 1` helper), so **every**
// `bvsub`/`bvneg` circuit wider than 64 bits was blasted against
// `1 + 2^64 + 2^128 + …`.
//
// That does not merely weaken the circuit — it encodes a *different* formula,
// so it fabricated wrong `sat` answers **and wrong `unsat` proofs**.  Measured
// on the unmodified 0.3.4 tree (`p2-oxiz-034/report.md` §6.1/§6.2, re-measured
// for this suite at widths 65, 96 and 128 — all twelve rows below were wrong
// at every one of the three widths):
//
// | property below                       | correct | 0.3.3/0.3.4 answered |
// |--------------------------------------|---------|----------------------|
// | `(x + x) - x = x`                     | `unsat` | **`sat`**            |
// | `(x + y) - y = x`                     | `unsat` | **`sat`**            |
// | `x = 0 ∧ (x + x) - x = 0`             | `sat`   | **`unsat`** (proof!) |
// | `x = 0 ∧ -x = 0`                      | `sat`   | **`unsat`** (proof!) |
// | `a = b ∧ a - b = 0`                   | `sat`   | **`unsat`** (proof!) |
// | `x = 0 ∧ (x + x) - x = 2^64`          | `unsat` | **`sat`**            |
// | `a >=u b ∧ ¬(a - b <=u a)`            | `unsat` | **`sat`**            |
//
// The three widths are 65 (one bit past the boundary), 96 (a non-multiple of
// 64) and 128 (`u128`, the width cargo-formal's encoder emits for `u128`/`i128`
// Rust code).  Width 64 was correct throughout and stays as the control.

/// Widths exercised by the `encode_add_const` group: just past the `u64`
/// boundary, a non-multiple of 64, and `u128`.
const WIDE_WIDTHS: [u32; 3] = [65, 96, 128];

/// `2^64`, the first value a `u64` cannot hold — the bit the broken shift
/// injected.
const TWO_POW_64: &str = "18446744073709551616";

/// `(not (= (bvsub (bvadd x x) x) x))` — the negation of an identity, so
/// `unsat` at every width.  0.3.3/0.3.4: `sat` at 65, 96 and 128.
#[test]
fn wide_sub_of_add_is_identity_above_64_bits() {
    for w in WIDE_WIDTHS {
        let script = format!(
            "(set-logic QF_BV)\n\
             (declare-const x (_ BitVec {w}))\n\
             (assert (not (= (bvsub (bvadd x x) x) x)))\n\
             (check-sat)\n"
        );
        assert_eq!(
            run_script(&script),
            SolverResult::Unsat,
            "(x + x) - x = x must hold at width {w}"
        );
    }
}

/// The two-variable form `(not (= (bvsub (bvadd x y) y) x))`, so the result is
/// not reachable by folding a repeated operand.  0.3.3/0.3.4: `sat` at 65, 96
/// and 128.
#[test]
fn wide_sub_of_add_two_vars_is_identity_above_64_bits() {
    for w in WIDE_WIDTHS {
        let script = format!(
            "(set-logic QF_BV)\n\
             (declare-const x (_ BitVec {w}))\n\
             (declare-const y (_ BitVec {w}))\n\
             (assert (not (= (bvsub (bvadd x y) y) x)))\n\
             (check-sat)\n"
        );
        assert_eq!(
            run_script(&script),
            SolverResult::Unsat,
            "(x + y) - y = x must hold at width {w}"
        );
    }
}

/// `x = 0 ∧ (x + x) - x = 0` is satisfiable at `x = 0`.  0.3.3/0.3.4 answered
/// **`unsat`** at 65, 96 and 128 — a fabricated proof (report §6.2 W1): the
/// corrupted circuit forces the subtraction to `2^64`.
#[test]
fn wide_zero_sub_of_add_is_satisfiable_above_64_bits() {
    for w in WIDE_WIDTHS {
        let script = format!(
            "(set-logic QF_BV)\n\
             (declare-const x (_ BitVec {w}))\n\
             (assert (= x (_ bv0 {w})))\n\
             (assert (= (bvsub (bvadd x x) x) (_ bv0 {w})))\n\
             (check-sat)\n"
        );
        assert_eq!(
            run_script(&script),
            SolverResult::Sat,
            "x = 0 must satisfy (x + x) - x = 0 at width {w}"
        );
    }
}

/// `x = 0 ∧ -x = 0` is satisfiable at `x = 0`.  0.3.3/0.3.4 answered
/// **`unsat`** at 65, 96 and 128 (report §6.2 W4) — `bvneg` is the second call
/// site of the broken constant.
#[test]
fn wide_neg_of_zero_is_zero_above_64_bits() {
    for w in WIDE_WIDTHS {
        let script = format!(
            "(set-logic QF_BV)\n\
             (declare-const x (_ BitVec {w}))\n\
             (assert (= x (_ bv0 {w})))\n\
             (assert (= (bvneg x) (_ bv0 {w})))\n\
             (check-sat)\n"
        );
        assert_eq!(
            run_script(&script),
            SolverResult::Sat,
            "-0 = 0 must be satisfiable at width {w}"
        );
    }
}

/// `a = b ∧ a - b = 0`, the ordinary "two equal values have zero difference"
/// shape.  0.3.3/0.3.4 answered **`unsat`** at 65, 96 and 128 (report §6.2 v3
/// at 128) — in cargo-formal's verdict mapping that is a silent `proved`.
#[test]
fn wide_equal_operands_have_zero_difference_above_64_bits() {
    for w in WIDE_WIDTHS {
        let script = format!(
            "(set-logic QF_BV)\n\
             (declare-const a (_ BitVec {w}))\n\
             (declare-const b (_ BitVec {w}))\n\
             (assert (= a b))\n\
             (assert (= (bvsub a b) (_ bv0 {w})))\n\
             (check-sat)\n"
        );
        assert_eq!(
            run_script(&script),
            SolverResult::Sat,
            "a = b ∧ a - b = 0 must be satisfiable at width {w}"
        );
    }
}

/// The direct confirmation of the mechanism: asserting the *corrupted* value
/// `2^64` for `(x + x) - x` at `x = 0` must be `unsat`.  0.3.3/0.3.4 answered
/// **`sat`** at 65, 96 and 128 (report §6.2 W2), which is what pins the wrong
/// value the broken constant produced.
#[test]
fn wide_sub_of_add_is_not_two_pow_64_above_64_bits() {
    for w in WIDE_WIDTHS {
        let script = format!(
            "(set-logic QF_BV)\n\
             (declare-const x (_ BitVec {w}))\n\
             (assert (= x (_ bv0 {w})))\n\
             (assert (= (bvsub (bvadd x x) x) (_ bv{TWO_POW_64} {w})))\n\
             (check-sat)\n"
        );
        assert_eq!(
            run_script(&script),
            SolverResult::Unsat,
            "(x + x) - x must be 0, not 2^64, at x = 0 and width {w}"
        );
    }
}

/// A realistic verification condition: "subtracting a smaller unsigned value
/// never grows it", i.e. `a >=u b ⇒ a - b <=u a`.  Negated it must be `unsat`.
/// 0.3.3/0.3.4 answered **`sat`** at 65, 96 and 128 (report §6.2 v1 at 128),
/// i.e. it handed back a bogus counterexample to a true `u128` VC.
#[test]
fn wide_unsigned_subtraction_no_underflow_vc_above_64_bits() {
    for w in WIDE_WIDTHS {
        let script = format!(
            "(set-logic QF_BV)\n\
             (declare-const a (_ BitVec {w}))\n\
             (declare-const b (_ BitVec {w}))\n\
             (assert (bvuge a b))\n\
             (assert (not (bvule (bvsub a b) a)))\n\
             (check-sat)\n"
        );
        assert_eq!(
            run_script(&script),
            SolverResult::Unsat,
            "a >=u b implies a - b <=u a at width {w}"
        );
    }
}

/// Width 64 is the control: it was correct before the fix and must stay
/// correct.  One row of each direction.
#[test]
fn width_64_control_group_is_unchanged() {
    let identity = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 64))
(assert (not (= (bvsub (bvadd x x) x) x)))
(check-sat)
";
    assert_eq!(run_script(identity), SolverResult::Unsat);

    let zero_sub = "\
(set-logic QF_BV)
(declare-const x (_ BitVec 64))
(assert (= x (_ bv0 64)))
(assert (= (bvsub (bvadd x x) x) (_ bv0 64)))
(check-sat)
";
    assert_eq!(run_script(zero_sub), SolverResult::Sat);
}
