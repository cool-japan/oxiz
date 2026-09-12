//! U-Z17 — SMT-LIB 2.7's bit-vector overflow predicates.
//!
//! `bvuaddo`, `bvsaddo`, `bvusubo`, `bvssubo`, `bvumulo`, `bvsmulo` and
//! `bvnego` are parsed as *desugarings* into term kinds the bit-blaster
//! already handles (`oxiz-core/src/smtlib/parser/build.rs`,
//! `build_bv_overflow_binary` and the `bvnego` arm of `build_unary`). There is
//! no new `TermKind` and no new circuit, so what these tests really check is
//! that each desugaring denotes the predicate it claims to.
//!
//! # What OxiZ 0.3.3 / 0.3.4 answered before this fix
//!
//! Every script in this file failed to parse. Measured on the 0.3.4 tree at
//! commit `6bdf958`, cargo-formal fixture `u07_bvuaddo_unsupported.smt2`
//! returned
//! `Err(parse error at position 586: unknown function/constant bvuaddo)`
//! from `Context::execute_script`, where the correct answer is `unsat`. The
//! other six names were equally unknown.
//!
//! # Method
//!
//! Both polarities are decided for every case. Pinning `a` and `b` to
//! constants makes the predicate's value determinate, so the *negative*
//! direction (`(assert (not p))` must be `unsat` when `p` holds) is the one
//! that carries the weight: a predicate that asserted nothing at all — the
//! failure mode U-Z14 found for `(> bv bv)` — would answer `sat` there.
//!
//! # Dependency on the U-Z10 fix (wave W1-a)
//!
//! `bvsaddo` and `bvssubo` are the two definitions that are a **conjunction
//! of two bit-vector atoms**, and negating one produces `(not (and A B))` over
//! bit-vector atoms — which is precisely the shape of finding U-Z10 (the
//! `BvSolver::pop` cache-rollback bug, cargo-formal fixture
//! `u01_boolean_structure_and_ule.smt2`). Measured on the W1-c tree *without*
//! W1-a's fix, for `a = 0001`, `b = 0111` at width 4:
//!
//! | script | answer | correct |
//! |---|---|---|
//! | `(not (= s(a) s(b)))` alone | `unsat` | `unsat` |
//! | `(not (distinct s(a) s(a+b)))` alone | `unsat` | `unsat` |
//! | `(not (and <both>))` | **`sat`** | `unsat` |
//!
//! Each conjunct is decided correctly; only their conjunction is not. With
//! W1-a's fix in the tree (`oxiz-theories/src/bv/solver.rs`,
//! `oxiz-solver/src/solver/theory_manager.rs`, `oxiz-theories/src/bv/propagator.rs`
//! taken from the W1-a working tree) all 13 tests in this file pass. The
//! desugarings are therefore correct as written; the four tests that touch
//! `bvsaddo`/`bvssubo` are the ones that need U-Z10 fixed, and they are
//! asserted rather than weakened so that they report the truth either way.
//!
//! The `2 * w`-wide multiplication forms — which the Phase 2b design expected
//! to depend on finding U-Z11 — turn out **not** to: `bvumulo`/`bvsmulo` at
//! width 64 pass without W1-a's fix, because a 128-bit `bvmul` never reaches
//! the `encode_add_const` path that U-Z11 is about.

use oxiz_solver::Context;

// ---------------------------------------------------------------------------
// The reference semantics, width-parametric.
// ---------------------------------------------------------------------------

/// `2^width`, the modulus of `width`-bit arithmetic. Exact for `width <= 32`,
/// which is every width these references are used at.
fn modulus(width: u32) -> u64 {
    1u64 << width
}

/// Reinterpret a `width`-bit unsigned value as two's-complement signed.
fn to_signed(value: u64, width: u32) -> i64 {
    let signed_modulus = i64::try_from(modulus(width)).unwrap_or(i64::MAX);
    let value = i64::try_from(value).unwrap_or(i64::MAX);
    if value >= signed_modulus / 2 {
        value - signed_modulus
    } else {
        value
    }
}

/// Whether `value` fits the `width`-bit signed range.
fn fits_signed(value: i64, width: u32) -> bool {
    let half = i64::try_from(modulus(width)).unwrap_or(i64::MAX) / 2;
    value >= -half && value < half
}

/// The reference value of `predicate` on the `width`-bit operands `a`, `b`.
///
/// `b` is ignored by `bvnego`.
fn reference(predicate: &str, a: u64, b: u64, width: u32) -> bool {
    let (sa, sb) = (to_signed(a, width), to_signed(b, width));
    match predicate {
        // An unsigned sum wraps iff the exact sum reaches the modulus.
        "bvuaddo" => a + b >= modulus(width),
        // An unsigned difference borrows iff the minuend is the smaller.
        "bvusubo" => a < b,
        // An unsigned product wraps iff the exact product reaches the modulus.
        "bvumulo" => a * b >= modulus(width),
        "bvsaddo" => !fits_signed(sa + sb, width),
        "bvssubo" => !fits_signed(sa - sb, width),
        "bvsmulo" => !fits_signed(sa * sb, width),
        // Negation overflows for exactly one value: the most negative one.
        "bvnego" => a == modulus(width) / 2,
        other => panic!("no reference for {other}"),
    }
}

/// The seven predicate names, unary one last.
const BINARY_PREDICATES: [&str; 6] = [
    "bvuaddo", "bvsaddo", "bvusubo", "bvssubo", "bvumulo", "bvsmulo",
];

// ---------------------------------------------------------------------------
// Driving the solver.
// ---------------------------------------------------------------------------

/// Run `script` and reduce its output to one verdict word.
fn decide(script: &str) -> String {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines
            .iter()
            .rev()
            .find(|line| matches!(line.trim(), "sat" | "unsat" | "unknown"))
            .map_or_else(
                || format!("no verdict in {lines:?}"),
                |l| l.trim().to_string(),
            ),
        Err(err) => format!("error: {err}"),
    }
}

/// A script pinning `a` (and, for a binary predicate, `b`) to constants and
/// asserting `predicate` with the given polarity.
fn script_for(predicate: &str, a: u64, b: u64, width: u32, polarity: bool) -> String {
    let binary = predicate != "bvnego";
    let application = if binary {
        format!("({predicate} a b)")
    } else {
        format!("({predicate} a)")
    };
    let asserted = if polarity {
        application
    } else {
        format!("(not {application})")
    };
    let mut script = format!(
        "(set-logic QF_BV)\n\
         (declare-const a (_ BitVec {width}))\n\
         (assert (= a (_ bv{a} {width})))\n"
    );
    if binary {
        script.push_str(&format!(
            "(declare-const b (_ BitVec {width}))\n\
             (assert (= b (_ bv{b} {width})))\n"
        ));
    }
    script.push_str(&format!("(assert {asserted})\n(check-sat)\n"));
    script
}

/// Assert that the solver agrees with [`reference`] on one case, in both
/// polarities.
fn check_case(predicate: &str, a: u64, b: u64, width: u32) {
    let expected = reference(predicate, a, b, width);
    let agreeing = decide(&script_for(predicate, a, b, width, expected));
    assert_eq!(
        agreeing, "sat",
        "({predicate} {a} {b}) at width {width}: reference says {expected}, \
         so asserting {expected} must be sat"
    );
    let contradicting = decide(&script_for(predicate, a, b, width, !expected));
    assert_eq!(
        contradicting, "unsat",
        "({predicate} {a} {b}) at width {width}: reference says {expected}, \
         so asserting {} must be unsat — a `sat` here means the predicate \
         constrained nothing",
        !expected
    );
}

// ---------------------------------------------------------------------------
// The cargo-formal conformance fixture.
// ---------------------------------------------------------------------------

/// cargo-formal fixture `u07_bvuaddo_unsupported.smt2`, verbatim.
///
/// 0.3.3/0.3.4: `Err(parse error ...: unknown function/constant bvuaddo)`.
#[test]
fn u07_fixture_is_unsat() {
    let script = "(set-logic QF_BV)\n\
                  (declare-const a (_ BitVec 8))\n\
                  (declare-const b (_ BitVec 8))\n\
                  (assert (= a #xff))\n\
                  (assert (= b #x01))\n\
                  (assert (not (bvuaddo a b)))\n\
                  (check-sat)\n";
    assert_eq!(decide(script), "unsat");
}

// ---------------------------------------------------------------------------
// Exhaustive differential test at width 4.
// ---------------------------------------------------------------------------

/// All 256 ordered pairs of 4-bit values, for every binary predicate, against
/// the reference. Split one test per predicate so a failure names the culprit
/// and the cases run in parallel.
macro_rules! exhaustive_width_four {
    ($name:ident, $predicate:literal) => {
        /// All 256 ordered pairs of 4-bit operands against the reference.
        ///
        /// 0.3.3/0.3.4: the script did not parse (unknown operator).
        #[test]
        fn $name() {
            for a in 0..16u64 {
                for b in 0..16u64 {
                    check_case($predicate, a, b, 4);
                }
            }
        }
    };
}

exhaustive_width_four!(bvuaddo_matches_the_reference_at_width_four, "bvuaddo");
exhaustive_width_four!(bvsaddo_matches_the_reference_at_width_four, "bvsaddo");
exhaustive_width_four!(bvusubo_matches_the_reference_at_width_four, "bvusubo");
exhaustive_width_four!(bvssubo_matches_the_reference_at_width_four, "bvssubo");
exhaustive_width_four!(bvumulo_matches_the_reference_at_width_four, "bvumulo");
exhaustive_width_four!(bvsmulo_matches_the_reference_at_width_four, "bvsmulo");

/// All 16 4-bit values under `bvnego`; only `1000` (= -8) overflows.
///
/// 0.3.3/0.3.4: the script did not parse (unknown operator).
#[test]
fn bvnego_matches_the_reference_at_width_four() {
    for a in 0..16u64 {
        check_case("bvnego", 0, a, 4);
    }
    // Spelled out, since the whole predicate is this one value.
    assert_eq!(decide(&script_for("bvnego", 8, 0, 4, true)), "sat");
    assert_eq!(decide(&script_for("bvnego", 8, 0, 4, false)), "unsat");
    assert_eq!(decide(&script_for("bvnego", 7, 0, 4, true)), "unsat");
    assert_eq!(decide(&script_for("bvnego", 0, 0, 4, true)), "unsat");
}

// ---------------------------------------------------------------------------
// Width 8, against Rust's own overflow flags.
// ---------------------------------------------------------------------------

/// The reference agrees with `u8`/`i8`'s own overflow reporting.
///
/// This is the reference checking *itself*: `reference` is width-parametric
/// arithmetic, and at width 8 the standard library already knows every answer.
/// Exhaustive over all 65 536 pairs, since it costs no solver calls.
#[test]
fn the_reference_agrees_with_u8_and_i8_at_width_eight() {
    for a in 0..256u64 {
        for b in 0..256u64 {
            let (ua, ub) = (a as u8, b as u8);
            let (ia, ib) = (ua as i8, ub as i8);
            assert_eq!(reference("bvuaddo", a, b, 8), ua.overflowing_add(ub).1);
            assert_eq!(reference("bvusubo", a, b, 8), ua.overflowing_sub(ub).1);
            assert_eq!(reference("bvumulo", a, b, 8), ua.overflowing_mul(ub).1);
            assert_eq!(reference("bvsaddo", a, b, 8), ia.overflowing_add(ib).1);
            assert_eq!(reference("bvssubo", a, b, 8), ia.overflowing_sub(ib).1);
            assert_eq!(reference("bvsmulo", a, b, 8), ia.overflowing_mul(ib).1);
        }
        assert_eq!(
            reference("bvnego", a, 0, 8),
            (a as u8 as i8).checked_neg().is_none()
        );
    }
}

/// A deterministic pseudo-random sequence, so a failure is reproducible from
/// the seed alone. Numerical Recipes' 32-bit linear congruential generator.
fn pseudo_random_pairs(count: usize) -> Vec<(u64, u64)> {
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        u64::from(state >> 24)
    };
    (0..count).map(|_| (next(), next())).collect()
}

/// 200 pseudo-random 8-bit pairs for every predicate, against the reference —
/// which the test above has pinned to `u8`/`i8`'s own overflow flags.
///
/// 0.3.3/0.3.4: the scripts did not parse (unknown operators).
#[test]
fn every_predicate_matches_the_reference_at_width_eight() {
    let pairs = pseudo_random_pairs(200);
    assert_eq!(pairs.len(), 200);
    for predicate in BINARY_PREDICATES {
        for &(a, b) in &pairs {
            check_case(predicate, a, b, 8);
        }
    }
    for &(a, _) in &pairs {
        check_case("bvnego", 0, a, 8);
    }
    // The extreme values every signed predicate turns on, spelled out.
    for predicate in BINARY_PREDICATES {
        for &(a, b) in &[(128u64, 128u64), (128, 1), (127, 1), (255, 255), (0, 0)] {
            check_case(predicate, a, b, 8);
        }
    }
    check_case("bvnego", 0, 128, 8);
    check_case("bvnego", 0, 127, 8);
}

// ---------------------------------------------------------------------------
// Width 64 — the double-width forms.
// ---------------------------------------------------------------------------

/// `bvumulo` and `bvsmulo` at width 64.
///
/// These are the two definitions that build a `2 * w`-bit intermediate
/// product, so at `w = 64` they exercise a 128-bit multiplier — the region
/// finding U-Z11 (wave W1-a, `oxiz-theories/src/bv/solver.rs`
/// `encode_add_const` shifting a `u64` by the bit index) makes unsound for
/// `bvsub`/`bvneg` above 64 bits. This test is expected to pass once W1-a's
/// fix is in the tree; it is asserted, not weakened, so that it reports the
/// truth either way.
///
/// 0.3.3/0.3.4: the scripts did not parse (unknown operators).
#[test]
fn multiplication_overflow_predicates_at_width_sixty_four() {
    const TWO_32: u64 = 1 << 32;
    const TWO_31: u64 = 1 << 31;
    const TWO_62: u64 = 1 << 62;

    // 2^32 * 2^32 = 2^64: the smallest unsigned 64-bit product that overflows.
    assert_eq!(
        decide(&script_for("bvumulo", TWO_32, TWO_32, 64, true)),
        "sat"
    );
    assert_eq!(
        decide(&script_for("bvumulo", TWO_32, TWO_32, 64, false)),
        "unsat"
    );
    // 2^31 * 2 = 2^32: comfortably inside 64 bits.
    assert_eq!(decide(&script_for("bvumulo", TWO_31, 2, 64, false)), "sat");
    assert_eq!(decide(&script_for("bvumulo", TWO_31, 2, 64, true)), "unsat");

    // Signed: 2^62 * 4 = 2^64, past i64::MAX.
    assert_eq!(decide(&script_for("bvsmulo", TWO_62, 4, 64, true)), "sat");
    assert_eq!(
        decide(&script_for("bvsmulo", TWO_62, 4, 64, false)),
        "unsat"
    );
    // Signed, no overflow: 3 * 5.
    assert_eq!(decide(&script_for("bvsmulo", 3, 5, 64, false)), "sat");
    assert_eq!(decide(&script_for("bvsmulo", 3, 5, 64, true)), "unsat");
}

/// The additive and negation predicates at width 64, where no double-width
/// intermediate is built.
///
/// 0.3.3/0.3.4: the scripts did not parse (unknown operators).
#[test]
fn additive_overflow_predicates_at_width_sixty_four() {
    const U64_MAX: u64 = u64::MAX;
    const I64_MIN: u64 = 1 << 63;

    // u64::MAX + 1 wraps.
    assert_eq!(decide(&script_for("bvuaddo", U64_MAX, 1, 64, true)), "sat");
    assert_eq!(
        decide(&script_for("bvuaddo", U64_MAX, 1, 64, false)),
        "unsat"
    );
    assert_eq!(decide(&script_for("bvuaddo", 1, 1, 64, false)), "sat");
    assert_eq!(decide(&script_for("bvuaddo", 1, 1, 64, true)), "unsat");

    // 0 - 1 borrows.
    assert_eq!(decide(&script_for("bvusubo", 0, 1, 64, true)), "sat");
    assert_eq!(decide(&script_for("bvusubo", 0, 1, 64, false)), "unsat");
    assert_eq!(decide(&script_for("bvusubo", 1, 0, 64, false)), "sat");

    // i64::MAX + 1 overflows the signed range; i64::MIN - 1 underflows it.
    assert_eq!(
        decide(&script_for("bvsaddo", I64_MIN - 1, 1, 64, true)),
        "sat"
    );
    assert_eq!(
        decide(&script_for("bvsaddo", I64_MIN - 1, 1, 64, false)),
        "unsat"
    );
    assert_eq!(decide(&script_for("bvssubo", I64_MIN, 1, 64, true)), "sat");
    assert_eq!(
        decide(&script_for("bvssubo", I64_MIN, 1, 64, false)),
        "unsat"
    );

    // Negating i64::MIN overflows; negating anything else does not.
    assert_eq!(decide(&script_for("bvnego", I64_MIN, 0, 64, true)), "sat");
    assert_eq!(
        decide(&script_for("bvnego", I64_MIN, 0, 64, false)),
        "unsat"
    );
    assert_eq!(
        decide(&script_for("bvnego", I64_MIN - 1, 0, 64, false)),
        "sat"
    );
    assert_eq!(
        decide(&script_for("bvnego", I64_MIN - 1, 0, 64, true)),
        "unsat"
    );
}

// ---------------------------------------------------------------------------
// Sort discipline.
// ---------------------------------------------------------------------------

/// The overflow predicates obey the same operand rule as every other binary
/// bit-vector operator (U-Z14): one shared `(_ BitVec w)` sort.
#[test]
fn overflow_predicates_reject_mismatched_operands() {
    let head = "(set-logic QF_BV)\n\
                (declare-const a8 (_ BitVec 8))\n\
                (declare-const b16 (_ BitVec 16))\n";
    for predicate in BINARY_PREDICATES {
        let script = format!("{head}(assert ({predicate} a8 b16))\n(check-sat)\n");
        let verdict = decide(&script);
        assert!(
            verdict.starts_with("error:"),
            "{predicate} accepted mismatched widths: {verdict}"
        );
        assert!(
            verdict.contains("same bit-vector width"),
            "{predicate}: {verdict}"
        );
    }
    let script = format!("{head}(assert (bvnego true))\n(check-sat)\n");
    assert!(decide(&script).starts_with("error:"));
}
