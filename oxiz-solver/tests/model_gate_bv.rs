//! End-to-end cover for the bit-vector arm of the model-verification gate.
//!
//! `Solver::model_refutes_assertions` is the last check before a candidate
//! model is reported as `sat`.  Until `solver/model_eval_bv.rs` existed it was
//! *blind to bit-vectors*: `EvalVal` had only `Bool` and `Num` variants, every
//! `Bv*` term fell into `open_in_model`'s closing arm, and `parse_value_term`
//! answered `Undetermined` for a `BitVecConst`.  So for a QF_BV formula the
//! gate returned `false` — "go ahead and report `sat`" — whatever the model
//! said, and OxiZ 0.3.3 and 0.3.4 answered `sat` for
//! `(= a #x0f) ∧ (not (and (bvule a #x0f) (bvule a #x10)))` with the model
//! `a = #b00001111`, which makes that second assertion **false**.
//!
//! Adding an evaluator to a gate is a two-sided risk, and this file covers
//! both sides:
//!
//! * **It must fire when the model really is refuted.**
//!   [`the_u01_family_is_no_longer_answered_sat`] pins that for the three
//!   cargo-formal fixtures the finding came from.
//! * **It must not fire otherwise.**  A gate that refuses a good model costs
//!   a *legal but useless* `unknown` on a formula the solver had solved.  The
//!   bulk of this file is therefore the [`SAT_SCRIPTS`] table: 47 satisfiable
//!   bit-vector scripts across widths 8, 32, 64, 65 and 128, covering every
//!   operator the evaluator folds, each of which must still answer `sat`.
//!
//! # Two things deliberately *not* asserted here
//!
//! * **The u01 family's verdict is `unsat` only once the root cause is
//!   fixed.** The gate's own exit is `unknown` (`check_core` blocks the model
//!   and re-solves, then gives up), so this file asserts "not `sat`" rather
//!   than a particular non-`sat` answer.  That assertion is the one that is
//!   true both before and after the bit-vector solver's scope-rollback fix,
//!   and it is the property that matters: a fabricated counterexample has
//!   become an honest verdict.
//! * **The printed radix.**  `(get-value)`'s `#x` / `#b` choice is itself
//!   under repair, so [`printed_value`] reads the *number* and accepts either
//!   spelling.
//!
//! Every width here is at most 128 so the expectations can be written as
//! `u128` literals, independent of any bignum code in the crate under test.
//!
//! `bvsub` and `bvneg` are deliberately absent above 64 bits: their circuits
//! go through `encode_add_const`, whose `u64` shift is a separate known defect
//! at those widths, and a script that tripped over it would fail here for a
//! reason that has nothing to do with the gate.

use oxiz_solver::{Context, SolverResult};

/// Every output line of a script.
fn run_script_output(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script).unwrap_or_default()
}

/// The verdict of a script: the last `sat` / `unsat` / `unknown` token.
fn verdict(outputs: &[String]) -> SolverResult {
    for token in outputs.iter().rev() {
        match token.trim() {
            "sat" => return SolverResult::Sat,
            "unsat" => return SolverResult::Unsat,
            "unknown" => return SolverResult::Unknown,
            _ => {}
        }
    }
    SolverResult::Unknown
}

/// Run one script and return its verdict.
fn run(script: &str) -> SolverResult {
    verdict(&run_script_output(script))
}

/// The value `(get-value (<name>))` printed for `name`, as a number.
///
/// Radix-agnostic on purpose: which of `#x` and `#b` OxiZ picks for a given
/// width is a separate open finding, and these tests are about the gate, not
/// about the printer.  `None` when the name or its value is missing.
fn printed_value(outputs: &[String], name: &str) -> Option<u128> {
    let joined = outputs.join("\n");
    let after_name = joined.find(name).map(|at| &joined[at + name.len()..])?;
    let after_hash = after_name.find('#').map(|at| &after_name[at + 1..])?;
    let mut chars = after_hash.chars();
    let radix = match chars.next()? {
        'x' => 16,
        'b' => 2,
        _ => return None,
    };
    let digits: String = chars.take_while(char::is_ascii_alphanumeric).collect();
    u128::from_str_radix(&digits, radix).ok()
}

// ─────────────────────────────────────────────────────────────────────────
// The gate must not refuse a good model
// ─────────────────────────────────────────────────────────────────────────

/// Satisfiable bit-vector scripts, one per row: a name for the failure
/// message and the script itself.  Every one of them must answer `sat`.
///
/// The witnesses are pinned wherever a wrong fold would otherwise be masked
/// by the solver simply picking a different model: with `v0` nailed to a
/// constant there is exactly one model, so if the evaluator folds any
/// operation in the script incorrectly the gate refutes that single model and
/// the verdict drops to `unknown`.
const SAT_SCRIPTS: &[(&str, &str)] = &[
    // ---- width 8 --------------------------------------------------------
    (
        "w8 pinned constant",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x0f))(check-sat)",
    ),
    (
        "w8 bvadd",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= (bvadd v0 #x01) #x10))(check-sat)",
    ),
    (
        "w8 bvsub",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x03))(assert (= (bvsub v0 #x05) #xfe))(check-sat)",
    ),
    (
        "w8 bvmul",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= (bvmul v0 #x03) #x21))(check-sat)",
    ),
    (
        "w8 bvudiv by zero is all ones",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x07))(assert (= (bvudiv v0 #x00) #xff))(check-sat)",
    ),
    (
        "w8 bvurem by zero is the dividend",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x07))(assert (= (bvurem v0 #x00) #x07))(check-sat)",
    ),
    (
        "w8 bvudiv and bvurem",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x0d))(assert (= (bvudiv v0 #x03) #x04))\
         (assert (= (bvurem v0 #x03) #x01))(check-sat)",
    ),
    (
        "w8 bvsdiv and bvsrem of a negative dividend",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #xf9))(assert (= (bvsdiv v0 #x02) #xfd))\
         (assert (= (bvsrem v0 #x02) #xff))(check-sat)",
    ),
    (
        "w8 bvshl",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x03))(assert (= (bvshl v0 #x02) #x0c))(check-sat)",
    ),
    (
        "w8 bvlshr at the width",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #xff))(assert (= (bvlshr v0 #x08) #x00))(check-sat)",
    ),
    (
        "w8 bvashr sign fills",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x80))(assert (= (bvashr v0 #x01) #xc0))(check-sat)",
    ),
    (
        "w8 bvashr past the width saturates",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x80))(assert (= (bvashr v0 #xff) #xff))(check-sat)",
    ),
    (
        "w8 bitwise",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x0f))(assert (= (bvnot v0) #xf0))\
         (assert (= (bvand v0 #x33) #x03))(assert (= (bvor v0 #x30) #x3f))\
         (assert (= (bvxor v0 #xff) #xf0))(check-sat)",
    ),
    (
        "w8 concat and extract",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))(declare-const v1 (_ BitVec 8))\
         (assert (= v0 #xab))(assert (= v1 #xcd))\
         (assert (= (concat v0 v1) #xabcd))\
         (assert (= ((_ extract 15 8) (concat v0 v1)) #xab))\
         (assert (= ((_ extract 7 0) (concat v0 v1)) #xcd))(check-sat)",
    ),
    (
        "w8 signed and unsigned disagree",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x80))(assert (bvslt v0 #x00))(assert (bvugt v0 #x00))\
         (assert (bvsle v0 #x80))(assert (bvule v0 #x80))(check-sat)",
    ),
    (
        "w8 disjunction over bv atoms",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x0f))(assert (or (bvugt v0 #x10) (bvule v0 #x0f)))(check-sat)",
    ),
    (
        "w8 negated conjunction, satisfied",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x20))\
         (assert (not (and (bvule v0 #x0f) (bvule v0 #x10))))(check-sat)",
    ),
    (
        "w8 distinct",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))(declare-const v1 (_ BitVec 8))\
         (declare-const v2 (_ BitVec 8))(assert (distinct v0 v1 v2))\
         (assert (= v0 #x01))(assert (= v1 #x02))(check-sat)",
    ),
    (
        "w8 ite over a bv condition",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))(declare-const v1 (_ BitVec 8))\
         (assert (= v1 #x00))(assert (= v0 (ite (bvult v1 #x02) #x11 #x22)))(check-sat)",
    ),
    (
        "w8 bvnand bvnor bvxnor bvcomp",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x0f))(assert (= (bvnand v0 #x33) #xfc))\
         (assert (= (bvnor v0 #x30) #xc0))(assert (= (bvxnor v0 #xff) #x0f))\
         (assert (= (bvcomp v0 #x0f) #b1))(check-sat)",
    ),
    (
        "w8 zero_extend sign_extend rotate repeat",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
         (assert (= v0 #x81))(assert (= ((_ zero_extend 8) v0) #x0081))\
         (assert (= ((_ sign_extend 8) v0) #xff81))\
         (assert (= ((_ rotate_left 1) v0) #x03))\
         (assert (= ((_ rotate_right 1) v0) #xc0))\
         (assert (= ((_ repeat 2) v0) #x8181))(check-sat)",
    ),
    // ---- width 32 -------------------------------------------------------
    (
        "w32 bvadd",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))\
         (assert (= v0 #x0000000f))\
         (assert (= (bvadd v0 #x00000001) #x00000010))(check-sat)",
    ),
    (
        "w32 bvmul",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))\
         (assert (= v0 #x00010001))\
         (assert (= (bvmul v0 #x00000003) #x00030003))(check-sat)",
    ),
    (
        "w32 bvudiv and bvurem",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))\
         (assert (= v0 #x0000000c))(assert (= (bvudiv v0 #x00000003) #x00000004))\
         (assert (= (bvurem v0 #x00000005) #x00000002))(check-sat)",
    ),
    (
        "w32 shifts",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))\
         (assert (= v0 #x80000000))\
         (assert (= (bvlshr v0 #x0000001f) #x00000001))\
         (assert (= (bvashr v0 #x0000001f) #xffffffff))\
         (assert (= (bvshl v0 #x00000001) #x00000000))(check-sat)",
    ),
    (
        "w32 extract",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))\
         (assert (= v0 #x12345678))\
         (assert (= ((_ extract 31 16) v0) #x1234))\
         (assert (= ((_ extract 15 0) v0) #x5678))(check-sat)",
    ),
    (
        "w32 range of comparisons",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))\
         (assert (bvult v0 #x00000010))(assert (bvugt v0 #x00000005))(check-sat)",
    ),
    // ---- width 64 -------------------------------------------------------
    (
        "w64 bvadd",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 64))\
         (assert (= v0 #x000000000000000f))\
         (assert (= (bvadd v0 #x0000000000000001) #x0000000000000010))(check-sat)",
    ),
    (
        "w64 bvmul",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 64))\
         (assert (= v0 #x0000000100000001))\
         (assert (= (bvmul v0 #x0000000000000002) #x0000000200000002))(check-sat)",
    ),
    (
        "w64 bvudiv and bvurem",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 64))\
         (assert (= v0 #x000000000000000d))\
         (assert (= (bvudiv v0 #x0000000000000003) #x0000000000000004))\
         (assert (= (bvurem v0 #x0000000000000003) #x0000000000000001))(check-sat)",
    ),
    (
        "w64 shifts past the width",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 64))\
         (assert (= v0 #x8000000000000000))\
         (assert (= (bvlshr v0 #x000000000000003f) #x0000000000000001))\
         (assert (= (bvashr v0 #x00000000000000ff) #xffffffffffffffff))\
         (assert (= (bvshl v0 #x0000000000000040) #x0000000000000000))(check-sat)",
    ),
    (
        "w64 concat of two 32-bit halves",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))(declare-const v1 (_ BitVec 32))\
         (assert (= v0 #x00000001))(assert (= v1 #xffffffff))\
         (assert (= (concat v0 v1) #x00000001ffffffff))(check-sat)",
    ),
    (
        "w64 unsigned beats signed at the sign bit",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 64))\
         (assert (= v0 #x8000000000000000))\
         (assert (bvugt v0 #x0000000000000001))\
         (assert (bvslt v0 #x0000000000000001))(check-sat)",
    ),
    // ---- width 65: past the 64-bit limb boundary, and not a multiple of 4
    (
        "w65 pinned to 2^64",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
         (assert (= v0 (_ bv18446744073709551616 65)))\
         (assert (bvugt v0 (_ bv18446744073709551615 65)))(check-sat)",
    ),
    (
        "w65 bvadd wraps at 2^65",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
         (assert (= v0 (_ bv18446744073709551616 65)))\
         (assert (= (bvadd v0 v0) (_ bv36893488147419103232 65)))(check-sat)",
    ),
    (
        "w65 bvmul by two",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
         (assert (= v0 (_ bv18446744073709551616 65)))\
         (assert (= (bvmul v0 (_ bv2 65)) (_ bv36893488147419103232 65)))(check-sat)",
    ),
    (
        "w65 bvshl across the limb boundary",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
         (assert (= v0 (_ bv1 65)))\
         (assert (= (bvshl v0 (_ bv64 65)) (_ bv18446744073709551616 65)))\
         (assert (= (bvlshr (bvshl v0 (_ bv64 65)) (_ bv64 65)) (_ bv1 65)))(check-sat)",
    ),
    (
        "w65 bvlshr past the width",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
         (assert (= v0 (_ bv18446744073709551616 65)))\
         (assert (= (bvlshr v0 (_ bv65 65)) (_ bv0 65)))(check-sat)",
    ),
    (
        "w65 bvudiv by zero is all ones",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
         (assert (= v0 (_ bv18446744073709551616 65)))\
         (assert (= (bvudiv v0 (_ bv0 65)) (_ bv36893488147419103231 65)))(check-sat)",
    ),
    (
        "w65 extract across the limb boundary",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
         (assert (= v0 (_ bv18446744073709551616 65)))\
         (assert (= ((_ extract 64 64) v0) #b1))\
         (assert (= ((_ extract 63 0) v0) #x0000000000000000))(check-sat)",
    ),
    // ---- width 128 ------------------------------------------------------
    (
        "w128 high limb is not zero",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 128))\
         (assert (= v0 #x00000000000000010000000000000000))\
         (assert (bvugt v0 #x000000000000000000000000ffffffff))(check-sat)",
    ),
    (
        "w128 bvmul overflows to zero",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 128))\
         (assert (= v0 #x00000000000000010000000000000000))\
         (assert (= (bvmul v0 v0) #x00000000000000000000000000000000))(check-sat)",
    ),
    (
        "w128 bvlshr by 64",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 128))\
         (assert (= v0 #x00000000000000010000000000000000))\
         (assert (= (bvlshr v0 #x00000000000000000000000000000040) \
         #x00000000000000000000000000000001))(check-sat)",
    ),
    (
        "w128 bvudiv by two",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 128))\
         (assert (= v0 #x00000000000000010000000000000000))\
         (assert (= (bvudiv v0 #x00000000000000000000000000000002) \
         #x00000000000000008000000000000000))(check-sat)",
    ),
    (
        "w128 concat of two 64-bit halves",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 64))(declare-const v1 (_ BitVec 64))\
         (assert (= v0 #x0000000000000001))(assert (= v1 #x0000000000000000))\
         (assert (= (concat v0 v1) #x00000000000000010000000000000000))(check-sat)",
    ),
    (
        "w128 extract the high half",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 128))\
         (assert (= v0 #x00000000000000010000000000000000))\
         (assert (= ((_ extract 127 64) v0) #x0000000000000001))\
         (assert (= ((_ extract 63 0) v0) #x0000000000000000))(check-sat)",
    ),
    (
        "w128 all ones is -1 signed",
        "(set-logic QF_BV)(declare-const v0 (_ BitVec 128))\
         (assert (= v0 #xffffffffffffffffffffffffffffffff))\
         (assert (bvslt v0 #x00000000000000000000000000000000))\
         (assert (bvugt v0 #x00000000000000000000000000000000))(check-sat)",
    ),
];

/// Every script in [`SAT_SCRIPTS`] must still answer `sat`.
///
/// This is the *cost* side of the gate.  A model-verification gate that
/// refuses a good model is not unsound — `unknown` is always a legal verdict —
/// but it throws away answers the solver had already found, and a bit-vector
/// evaluator with one wrong width rule or one wrong division-by-zero case
/// would do exactly that on a formula that pins its witness.  Every script in
/// the table answered `sat` on the unmodified 0.3.4 tree as well (measured
/// against `6bdf958`), so a failure here is this module's regression and
/// nobody else's.
#[test]
fn satisfiable_bit_vector_scripts_still_answer_sat() {
    let mut failures: Vec<String> = Vec::new();
    for (name, script) in SAT_SCRIPTS {
        let answer = run(script);
        if answer != SolverResult::Sat {
            failures.push(format!("{name}: {answer:?}"));
        }
    }
    assert!(
        failures.is_empty(),
        "the model gate must not turn a satisfiable bit-vector script into a \
         non-`sat` verdict; {} of {} regressed:\n  {}",
        failures.len(),
        SAT_SCRIPTS.len(),
        failures.join("\n  ")
    );
}

/// The table is the load-bearing part of the test above, so its size and its
/// width coverage are pinned rather than left to drift.  (No 0.3.3/0.3.4
/// behaviour to record: this one is about the test data, not the solver.)
#[test]
fn the_sat_table_covers_every_width_the_design_names() {
    assert!(
        SAT_SCRIPTS.len() >= 30,
        "at least 30 satisfiable scripts, got {}",
        SAT_SCRIPTS.len()
    );
    for prefix in ["w8 ", "w32 ", "w64 ", "w65 ", "w128 "] {
        assert!(
            SAT_SCRIPTS.iter().any(|(name, _)| name.starts_with(prefix)),
            "no script at width prefix {prefix:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// `(get-value)` still reports the right numbers
// ─────────────────────────────────────────────────────────────────────────

/// A satisfiable script whose model is unique must still come back with that
/// model, at every width.
///
/// What 0.3.3 / 0.3.4 answered: `sat` with these same values (measured on a
/// pristine `6bdf958` extract) — the gate must not cost any of them.
///
/// The gate now re-evaluates every assertion against the model before `sat`
/// is reported, so a model the solver builds *and* the gate accepts is a
/// stronger claim than either alone.
#[test]
fn pinned_models_survive_the_gate_at_every_width() {
    let cases: &[(&str, &str, u128)] = &[
        (
            "w8",
            "(set-logic QF_BV)(declare-const v0 (_ BitVec 8))\
             (assert (= (bvadd v0 #x01) #x10))(check-sat)(get-value (v0))",
            0x0f,
        ),
        (
            // The multiplier must be ODD for the model to be unique: `2 * x`
            // is two-to-one modulo `2^32`, so `(= (bvmul v0 #x02) #x08)` has
            // `#x00000004` *and* `#x80000004` as models and either is a
            // correct answer.  `3` is invertible modulo `2^32`.
            "w32",
            "(set-logic QF_BV)(declare-const v0 (_ BitVec 32))\
             (assert (= (bvmul v0 #x00000003) #x0000000c))(check-sat)(get-value (v0))",
            4,
        ),
        (
            "w64",
            "(set-logic QF_BV)(declare-const v0 (_ BitVec 64))\
             (assert (= v0 #xdeadbeefcafebabe))(check-sat)(get-value (v0))",
            0xdead_beef_cafe_babe,
        ),
        (
            "w65",
            "(set-logic QF_BV)(declare-const v0 (_ BitVec 65))\
             (assert (= v0 (_ bv18446744073709551616 65)))(check-sat)(get-value (v0))",
            1u128 << 64,
        ),
        (
            "w128",
            "(set-logic QF_BV)(declare-const v0 (_ BitVec 128))\
             (assert (= v0 #x00000000000000010000000000000000))(check-sat)(get-value (v0))",
            1u128 << 64,
        ),
    ];

    for (name, script, expected) in cases {
        let outputs = run_script_output(script);
        assert_eq!(
            verdict(&outputs),
            SolverResult::Sat,
            "{name} must be sat, got {:?}: {outputs:?}",
            verdict(&outputs)
        );
        assert_eq!(
            printed_value(&outputs, "v0"),
            Some(*expected),
            "{name} model value, from {outputs:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The gate must fire when the model really is refuted
// ─────────────────────────────────────────────────────────────────────────

/// The three cargo-formal conformance fixtures the finding came from.
///
/// Each is unsatisfiable and each was answered **`sat`** by OxiZ 0.3.3 and
/// 0.3.4, with a model that falsifies its own second assertion:
///
/// | fixture | model reported | second assertion under it |
/// |---|---|---|
/// | `u01_boolean_structure_and_ule` | `a = #b00001111` | `(not (and true true))` = false |
/// | `u08_boolean_structure_ult_ugt` | `a = #b00001111` | false |
/// | `u09_boolean_structure_ugt_ult` | `a = #b00001111` | false |
///
/// The assertion here is "**not `sat`**", not "`unsat`", and that is
/// deliberate: this gate's own exit is `unknown` (`check_core` blocks the
/// model, re-solves if it can afford to, and otherwise reports `unknown`),
/// while the bit-vector solver's scope-rollback fix makes the same scripts
/// come back `unsat` outright.  "Not `sat`" is the property both establish and
/// the one that matters — a fabricated counterexample has become an honest
/// verdict.
#[test]
fn the_u01_family_is_no_longer_answered_sat() {
    let fixtures: &[(&str, &str)] = &[
        (
            "u01 (not (and (bvule a #x0f) (bvule a #x10)))",
            "(set-logic QF_BV)(declare-const a (_ BitVec 8))\
             (assert (= a #x0f))\
             (assert (not (and (bvule a #x0f) (bvule a #x10))))(check-sat)",
        ),
        (
            "u08 (not (and (bvult a #x10) (bvugt a #x00)))",
            "(set-logic QF_BV)(declare-const a (_ BitVec 8))\
             (assert (= a #x0f))\
             (assert (not (and (bvult a #x10) (bvugt a #x00))))(check-sat)",
        ),
        (
            "u09 (not (and (bvugt a #x00) (bvult a #x10)))",
            "(set-logic QF_BV)(declare-const a (_ BitVec 8))\
             (assert (= a #x0f))\
             (assert (not (and (bvugt a #x00) (bvult a #x10))))(check-sat)",
        ),
        (
            "de Morgan form of u01",
            "(set-logic QF_BV)(declare-const a (_ BitVec 8))\
             (assert (= a #x0f))\
             (assert (or (not (bvule a #x0f)) (not (bvule a #x10))))(check-sat)",
        ),
        (
            "signed variant of u01",
            "(set-logic QF_BV)(declare-const a (_ BitVec 8))\
             (assert (= a #x0f))\
             (assert (not (and (bvsle a #x0f) (bvsle a #x10))))(check-sat)",
        ),
    ];

    for (name, script) in fixtures {
        assert_ne!(
            run(script),
            SolverResult::Sat,
            "{name} is unsatisfiable; 0.3.3/0.3.4 answered `sat` with a model \
             that falsifies its own second assertion"
        );
    }
}

/// A model that violates a bit-vector assertion outright is refused even when
/// the assertion is the only one in the script.
///
/// `(bvule (bvadd x y) #x02)` with `x = y = #xff` (both pinned) is false —
/// `#xff + #xff` is `#xfe`, which is not `<=u #x02`.  The script really is
/// unsatisfiable, so any answer but `sat` is correct; the point is that the
/// evaluator can now *see* the arithmetic, which it could not before.
///
/// What 0.3.3 / 0.3.4 answered: `unsat` — the bit-vector solver refutes this
/// one on its own, with the gate contributing nothing because `bvadd` and
/// `bvule` were both `Undetermined` to it.  The gate's own contribution is
/// pinned from the other side by the unit test
/// `model_eval_bv::tests::the_u01_model_is_refused`, which drives it directly
/// with the model 0.3.3/0.3.4 waved through.
#[test]
fn an_arithmetic_bit_vector_violation_is_not_reported_sat() {
    let script = "(set-logic QF_BV)\
        (declare-const x (_ BitVec 8))(declare-const y (_ BitVec 8))\
        (assert (= x #xff))(assert (= y #xff))\
        (assert (bvule (bvadd x y) #x02))(check-sat)";
    assert_ne!(run(script), SolverResult::Sat);
}

/// The control for the test above: the same shape with a bound the sum
/// actually meets must still answer `sat`, so the assertion above is not
/// passing merely because the gate refuses every `bvadd`.
///
/// What 0.3.3 / 0.3.4 answered: `sat`, and it must stay `sat`.
#[test]
fn the_same_shape_with_a_satisfied_bound_is_still_sat() {
    let script = "(set-logic QF_BV)\
        (declare-const x (_ BitVec 8))(declare-const y (_ BitVec 8))\
        (assert (= x #xff))(assert (= y #xff))\
        (assert (bvule (bvadd x y) #xfe))(check-sat)";
    assert_eq!(run(script), SolverResult::Sat);
}

/// A formula whose term DAG has heavy sharing must not cost its *tree* size.
///
/// `y1 = x + x`, `y2 = y1 + y1`, ... is one node per level in the hash-consed
/// arena and `2^n` nodes as a tree.  Adding bit-vector arms to the gate's
/// evaluator made that difference reachable for the first time — before them a
/// `bvadd` answered `Undetermined` at the top node without descending — and
/// the evaluator's per-call memo table is what keeps it linear.  Measured
/// without the memo: 90 ms at depth 20, 1.4 s at depth 24, 22.8 s at depth 28,
/// a factor of two per level; depth 60 below would take longer than the age of
/// this project.
///
/// `3 * 2^60` is `0` modulo `2^32`, so the script is satisfiable and the gate
/// really does evaluate the whole chain rather than short-circuiting.
///
/// What 0.3.3 / 0.3.4 answered: `sat`, in about a millisecond — correct, but
/// only because the gate never descended into the chain at all.
#[test]
fn a_shared_bit_vector_dag_is_not_folded_as_a_tree() {
    const DEPTH: usize = 60;
    let mut body = format!("(= y{DEPTH} #x00000000)");
    for level in (1..=DEPTH).rev() {
        let previous = if level == 1 {
            "x".to_string()
        } else {
            format!("y{}", level - 1)
        };
        body = format!("(let ((y{level} (bvadd {previous} {previous}))) {body})");
    }
    let script = format!(
        "(set-logic QF_BV)(declare-const x (_ BitVec 32))\
         (assert (= x #x00000003))(assert {body})(check-sat)"
    );
    assert_eq!(run(&script), SolverResult::Sat);
}

/// Non-bit-vector formulas must be completely unaffected: the arm added to
/// `distinct` bails out to `Undetermined` on the first operand that is not a
/// bit-vector, which is the answer the gate gave for *every* `distinct`
/// before.  An integer `distinct` the LP model happens to satisfy with
/// colliding intermediate values must still be `sat`.
///
/// What 0.3.3 / 0.3.4 answered: `sat`, and it must stay `sat`.
#[test]
fn integer_formulas_are_unaffected() {
    let script = "(set-logic QF_LIA)\
        (declare-const p Int)(declare-const q Int)(declare-const r Int)\
        (assert (distinct p q r))(assert (> p 0))(assert (< r 100))(check-sat)";
    assert_eq!(run(script), SolverResult::Sat);
}
