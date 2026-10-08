//! U-Z14 — the SMT-LIB sort rules for `=`, `distinct`, `ite`, the Boolean
//! connectives and the arithmetic comparisons, enforced at parse time.
//!
//! `oxiz-core/src/smtlib/parser/build.rs` already rejected a width mismatch
//! under the `bvadd` family (`check_bv_binary_widths`). Everything else was
//! interned with no sort test at all: `(= a8 b16)` became an ordinary
//! equality between an 8-bit and a 16-bit term, `(not a8)` became a `Not`
//! over a bit-vector, `(> a8 #x0f)` became a `Gt` node the bit-vector branch
//! of the theory manager recognised but had **no arm for**, so it asserted
//! nothing at all.
//!
//! # What OxiZ 0.3.3 / 0.3.4 answered before this fix
//!
//! Every rejection below was accepted, and every script here that pins its
//! constants to contradictory values answered **`sat`** — a verified wrong
//! verdict, not merely a missing diagnostic. Measured on the 0.3.4 tree at
//! commit `6bdf958` with cargo-formal's conformance runner:
//!
//! | fixture | 0.3.3 / 0.3.4 | correct |
//! |---|---|---|
//! | `u03_width_mismatch_eq.smt2` | `sat` | `error` |
//! | `u04_width_mismatch_literal.smt2` | `sat` | `error` |
//! | `u05_width_mismatch_not.smt2` | `sat` | `error` |
//! | `(> a8 #x0f)` with `(= a8 #x0f)` (report `I0-a` §4.1) | `sat` | `error` |
//!
//! # Where the error surfaces
//!
//! [`parse_script`] is the first thing `oxiz_solver::Context::execute_script`
//! calls, and it propagates the failure with `?`, so an ill-sorted command
//! reaches an embedder as `Err(OxizError::ParseError { .. })` — **not** as an
//! `(error ...)` line in the response vector. These tests therefore assert on
//! `parse_script`'s `Err`, which is exactly the value `execute_script`
//! returns. (Measured end to end: `execute_script` on the u03 fixture returns
//! `Err(parse error at position 102: operands of = must have the same sort,
//! got (_ BitVec 8) and (_ BitVec 16))`.)

use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::parse_script;

/// Parse `script` and return the error message, or panic if it was accepted.
fn rejection(script: &str) -> String {
    let mut manager = TermManager::new();
    match parse_script(script, &mut manager) {
        Ok(commands) => panic!(
            "expected an ill-sorted script to be rejected, but it parsed into \
             {} commands:\n{script}",
            commands.len()
        ),
        Err(err) => err.to_string(),
    }
}

/// Parse `script`, asserting that it is accepted.
fn accepted(script: &str) {
    let mut manager = TermManager::new();
    if let Err(err) = parse_script(script, &mut manager) {
        panic!("expected a well-sorted script to parse, got {err}\n{script}");
    }
}

/// The declarations every bit-vector case below shares.
const BV_HEAD: &str = "(set-logic QF_BV)\n\
                       (declare-const a8 (_ BitVec 8))\n\
                       (declare-const b16 (_ BitVec 16))\n";

/// cargo-formal fixture `u03_width_mismatch_eq.smt2`, verbatim.
///
/// 0.3.3/0.3.4: `sat`.
#[test]
fn u03_equality_between_different_bit_vector_widths_is_rejected() {
    let script = "(set-logic QF_BV)\n\
                  (declare-const a8 (_ BitVec 8))\n\
                  (declare-const b16 (_ BitVec 16))\n\
                  (assert (= a8 b16))\n\
                  (check-sat)\n";
    let message = rejection(script);
    assert!(message.contains("operands of ="), "{message}");
    assert!(message.contains("(_ BitVec 8)"), "{message}");
    assert!(message.contains("(_ BitVec 16)"), "{message}");
}

/// cargo-formal fixture `u04_width_mismatch_literal.smt2`, verbatim: the
/// right-hand side is a four-hex-digit literal, i.e. 16 bits wide.
///
/// 0.3.3/0.3.4: `sat`.
#[test]
fn u04_equality_with_a_wider_literal_is_rejected() {
    let script = "(set-logic QF_BV)\n\
                  (declare-const a8 (_ BitVec 8))\n\
                  (assert (= a8 #x002a))\n\
                  (check-sat)\n";
    let message = rejection(script);
    assert!(message.contains("operands of ="), "{message}");
    assert!(message.contains("(_ BitVec 8)"), "{message}");
    assert!(message.contains("(_ BitVec 16)"), "{message}");
}

/// cargo-formal fixture `u05_width_mismatch_not.smt2`, verbatim.
///
/// 0.3.3/0.3.4: `sat`.
#[test]
fn u05_not_applied_to_a_bit_vector_is_rejected() {
    let script = "(set-logic QF_BV)\n\
                  (declare-const a8 (_ BitVec 8))\n\
                  (assert (not a8))\n\
                  (check-sat)\n";
    let message = rejection(script);
    assert!(message.contains("operands of not"), "{message}");
    assert!(message.contains("Bool"), "{message}");
    assert!(message.contains("(_ BitVec 8)"), "{message}");
}

/// Report `I0-a` §4.1: `(> a8 #x0f)` used to build a `Gt` node that asserted
/// nothing, so `(= a8 #x0f) ∧ (> a8 #x0f)` — a contradiction — answered `sat`.
///
/// 0.3.3/0.3.4: `sat`.
#[test]
fn arithmetic_comparisons_over_bit_vectors_are_rejected() {
    for op in ["<", "<=", ">", ">="] {
        let script =
            format!("{BV_HEAD}(assert (= a8 #x0f))\n(assert ({op} a8 #x0f))\n(check-sat)\n");
        let message = rejection(&script);
        assert!(message.contains(&format!("operands of {op}")), "{message}");
        assert!(message.contains("Int or Real"), "{message}");
        assert!(message.contains("(_ BitVec 8)"), "{message}");
    }
}

/// `distinct` has the same signature as `=` and the same rule.
///
/// 0.3.3/0.3.4: accepted, `sat`.
#[test]
fn distinct_between_different_bit_vector_widths_is_rejected() {
    let script = format!("{BV_HEAD}(assert (distinct a8 b16))\n(check-sat)\n");
    let message = rejection(&script);
    assert!(message.contains("operands of distinct"), "{message}");
    assert!(message.contains("(_ BitVec 16)"), "{message}");
}

/// The Boolean connectives take `Bool` operands only.
///
/// 0.3.3/0.3.4: all four accepted, `sat`.
#[test]
fn boolean_connectives_reject_a_bit_vector_operand() {
    for op in ["and", "or", "=>", "xor"] {
        let script = format!("{BV_HEAD}(assert ({op} a8 true))\n(check-sat)\n");
        let message = rejection(&script);
        assert!(message.contains(&format!("operands of {op}")), "{message}");
        assert!(message.contains("Bool"), "{message}");
        assert!(message.contains("(_ BitVec 8)"), "{message}");
    }
}

/// `ite` needs a `Bool` condition and two branches of one sort.
///
/// 0.3.3/0.3.4: both accepted.
#[test]
fn ite_checks_its_condition_and_its_branches() {
    let condition = format!("{BV_HEAD}(assert (= a8 (ite a8 a8 a8)))\n(check-sat)\n");
    let message = rejection(&condition);
    assert!(message.contains("operands of ite"), "{message}");
    assert!(message.contains("Bool"), "{message}");

    let branches = format!("{BV_HEAD}(assert (= a8 (ite true a8 b16)))\n(check-sat)\n");
    let message = rejection(&branches);
    assert!(message.contains("operands of ite"), "{message}");
    assert!(message.contains("same sort"), "{message}");
    assert!(message.contains("(_ BitVec 16)"), "{message}");
}

/// Two different theories under `=` is the same violation as two widths.
///
/// 0.3.3/0.3.4: accepted.
#[test]
fn equality_across_theories_is_rejected() {
    let script = "(set-logic ALL)\n\
                  (declare-const p Bool)\n\
                  (declare-const i Int)\n\
                  (assert (= p i))\n\
                  (check-sat)\n";
    let message = rejection(script);
    assert!(message.contains("operands of ="), "{message}");
    assert!(message.contains("Bool"), "{message}");
    assert!(message.contains("Int"), "{message}");
}

/// Positive controls: every shape the new checks could plausibly have broken
/// still parses. Without these the checks above could be satisfied by a
/// parser that rejects everything.
#[test]
fn well_sorted_scripts_still_parse() {
    // Bit-vectors of one width, under each of the checked operators.
    accepted(&format!("{BV_HEAD}(assert (= a8 #x0f))\n(check-sat)\n"));
    accepted(&format!(
        "{BV_HEAD}(assert (distinct a8 #x0f))\n(check-sat)\n"
    ));
    accepted(&format!(
        "{BV_HEAD}(assert (not (= a8 #x0f)))\n(check-sat)\n"
    ));
    accepted(&format!(
        "{BV_HEAD}(assert (and (= a8 #x0f) (bvule a8 #xff)))\n(check-sat)\n"
    ));
    accepted(&format!(
        "{BV_HEAD}(assert (or (= a8 #x0f) (xor true false)))\n(check-sat)\n"
    ));
    accepted(&format!(
        "{BV_HEAD}(assert (=> (= a8 #x0f) (bvule a8 #xff)))\n(check-sat)\n"
    ));
    accepted(&format!(
        "{BV_HEAD}(assert (= a8 (ite (= a8 #x0f) a8 #x00)))\n(check-sat)\n"
    ));
    // Equality between two *different* widths is the violation; equality
    // between two 16-bit terms is not.
    accepted(&format!("{BV_HEAD}(assert (= b16 #x002a))\n(check-sat)\n"));

    // Arithmetic, including the chainable n-ary form.
    let int_head = "(set-logic QF_LIA)\n(declare-const i Int)\n(declare-const j Int)\n";
    for op in ["<", "<=", ">", ">="] {
        accepted(&format!("{int_head}(assert ({op} i j))\n(check-sat)\n"));
        accepted(&format!("{int_head}(assert ({op} i j 0))\n(check-sat)\n"));
    }
    accepted(&format!("{int_head}(assert (distinct i j))\n(check-sat)\n"));

    // `Int` and `Real` mix: every SMT-LIB numeral is `Int`-sorted, so
    // rejecting `(= r 1)` for `r : Real` would reject the ordinary spelling.
    let real_head = "(set-logic QF_LRA)\n(declare-const r Real)\n";
    accepted(&format!("{real_head}(assert (= r 1))\n(check-sat)\n"));
    accepted(&format!("{real_head}(assert (< r 1))\n(check-sat)\n"));
    accepted(&format!("{real_head}(assert (= r 1.5))\n(check-sat)\n"));
    accepted(&format!(
        "{real_head}(assert (= r (ite (< r 1) 0 1.0)))\n(check-sat)\n"
    ));

    // Arrays, uninterpreted sorts and quantifiers keep their equalities.
    accepted(
        "(set-logic QF_AUFLIA)\n\
         (declare-const arr (Array Int Int))\n\
         (declare-const brr (Array Int Int))\n\
         (assert (= arr brr))\n\
         (assert (= (select arr 0) 1))\n\
         (check-sat)\n",
    );
    accepted(
        "(set-logic QF_UF)\n\
         (declare-sort S 0)\n\
         (declare-const s S)\n\
         (declare-const t S)\n\
         (assert (distinct s t))\n\
         (check-sat)\n",
    );
    accepted(
        "(set-logic UF)\n\
         (declare-sort S 0)\n\
         (declare-fun p (S) Bool)\n\
         (assert (forall ((x S)) (=> (p x) (p x))))\n\
         (check-sat)\n",
    );
    // A `let` binding carries its bound term's sort through the check.
    accepted(&format!(
        "{BV_HEAD}(assert (let ((v (bvadd a8 #x01))) (= v #x10)))\n(check-sat)\n"
    ));
}
