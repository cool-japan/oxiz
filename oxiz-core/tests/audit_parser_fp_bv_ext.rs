//! Regression tests for the P1 parser wave (SMT-LIB `FloatingPoint` /
//! `FixedSizeBitVectors` extensions, sort/lexer robustness).
//!
//! Each test reproduces one specific defect fixed in this wave and asserts
//! the corrected behavior:
//!
//! - FP-TOFP-01: indexed `to_fp`/`to_fp_unsigned`/`fp.to_sbv`/`fp.to_ubv`
//!   must accept their leading rounding-mode argument.
//! - BV-NEG-01: `bvneg` must be recognized (was a Bool-sorted uninterpreted
//!   apply).
//! - BV-EXT-01: `bvnand`/`bvnor`/`bvxnor`/`bvcomp`/`bvsmod` must be
//!   recognized and correctly typed.
//! - FP-CONV-SIB-01: `fp.to_real`, the `(fp sign exp sig)` literal, and the
//!   indexed FP special-value constants must be recognized.
//! - SORT-BUILTIN-01: `RoundingMode` and `RegLan` must not silently become
//!   ordinary uninterpreted sorts. The two reserved names have since diverged:
//!   `RoundingMode` became a first-class sort and is now *declarable* (these
//!   tests pin the acceptance, the case split a symbolic mode compiles into,
//!   and the honest errors for the positions no closure axiom can reach),
//!   while `RegLan` stays reserved because it names the built-in sort every
//!   regular-expression term is interned at.
//! - R1: `parse_sort` must not overflow the stack on deeply nested sorts.
//! - todo-1151: `try_mk_bv_concat` must not silently fabricate a width for a
//!   non-bit-vector operand — it returns a sort error instead.
//! - todo-1174: the lexer must reject leading-zero numerals and must not
//!   truncate `(_ bvN M)` literal values to `i64`.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::smtlib::{Command, Lexer, Printer, parse_script};
use oxiz_core::sort::SortKind;

/// Parse a full script, returning the manager plus the asserted term ids in
/// order.
fn parse_asserts(script: &str) -> (TermManager, Vec<TermId>) {
    let mut manager = TermManager::new();
    let commands = parse_script(script, &mut manager).expect("script should parse");
    let asserts = commands
        .into_iter()
        .filter_map(|c| match c {
            Command::Assert(t) => Some(t),
            _ => None,
        })
        .collect();
    (manager, asserts)
}

fn kind(manager: &TermManager, t: TermId) -> TermKind {
    manager.get(t).expect("term should exist").kind.clone()
}

fn sort_kind(manager: &TermManager, t: TermId) -> SortKind {
    let sort = manager.get(t).expect("term should exist").sort;
    manager
        .sorts
        .get(sort)
        .expect("sort should exist")
        .kind
        .clone()
}

/// Return the two operand term ids of an equality (`mk_eq` canonicalizes
/// operand order by raw `TermId`, so callers must not assume which side is
/// which — this is the same helper pattern used by `audit_parser_terms.rs`).
fn eq_operands(m: &TermManager, t: TermId) -> (TermId, TermId) {
    match kind(m, t) {
        TermKind::Eq(a, b) => (a, b),
        other => panic!("expected equality, got {other:?}"),
    }
}

/// Find, among the two operands of an equality, the one satisfying `pred`.
fn eq_side_matching(m: &TermManager, t: TermId, pred: impl Fn(&TermKind) -> bool) -> TermId {
    let (a, b) = eq_operands(m, t);
    if pred(&kind(m, a)) {
        a
    } else if pred(&kind(m, b)) {
        b
    } else {
        panic!(
            "neither operand matched: {:?} / {:?}",
            kind(m, a),
            kind(m, b)
        )
    }
}

/// Re-print a term via the basic printer and confirm the printed form
/// re-parses, *under a real strict script* that first re-declares every free
/// variable `t` depends on with its original sort (`decls`), to a term of
/// the same sort as `t`. This deliberately avoids `parse_term`'s lenient
/// bare-term mode, which defaults any undeclared symbol to `Bool` and would
/// silently "round-trip" even a wrongly-typed printed form.
fn assert_round_trips(manager: &TermManager, t: TermId, decls: &[&str]) {
    let printer = Printer::new(manager);
    let printed = printer.print_term(t);
    let sort_id = manager.get(t).expect("term should exist").sort;
    let mut sort_str = String::new();
    printer.write_sort(&mut sort_str, sort_id);

    let mut script = String::new();
    for d in decls {
        script.push_str(d);
        script.push('\n');
    }
    script.push_str(&format!("(declare-const __round_trip__ {sort_str})\n"));
    script.push_str(&format!("(assert (= __round_trip__ {printed}))\n"));

    let mut fresh = TermManager::new();
    parse_script(&script, &mut fresh)
        .unwrap_or_else(|e| panic!("round-trip script failed to parse:\n{script}\nerror: {e:?}"));
}

// ---------------------------------------------------------------------------
// FP-TOFP-01: indexed to_fp / to_fp_unsigned / fp.to_sbv / fp.to_ubv must
// accept a rounding-mode first argument.
// ---------------------------------------------------------------------------

#[test]
fn to_fp_from_real_accepts_rounding_mode() {
    let (m, asserts) = parse_asserts(
        r#"
        (set-logic QF_FP)
        (declare-const a (_ FloatingPoint 11 53))
        (assert (= a ((_ to_fp 11 53) RNE 10.0)))
        "#,
    );
    let rhs = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::RealToFp { .. }));
    match kind(&m, rhs) {
        TermKind::RealToFp { eb, sb, .. } => assert_eq!((eb, sb), (11, 53)),
        other => panic!("expected RealToFp, got {other:?}"),
    }
    // Not round-tripped: the pre-existing basic printer formats whole-number
    // `Real` constants (e.g. the `10.0` above) without a decimal point
    // (`{r}` on the underlying `Rational64`, e.g. "10"), which then re-lexes
    // as an `Int` numeral rather than a `Real` decimal — a separate,
    // out-of-scope printer defect (any `Real`-typed round trip through this
    // printer hits it, not just this operator).
}

#[test]
fn to_fp_from_bitvec_dispatches_to_signed_conversion() {
    let (m, asserts) = parse_asserts(
        r#"
        (set-logic QF_FP)
        (declare-const bv (_ BitVec 32))
        (declare-const a (_ FloatingPoint 8 24))
        (assert (= a ((_ to_fp 8 24) RTZ bv)))
        "#,
    );
    let rhs = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::SBVToFp { .. }));
    match kind(&m, rhs) {
        TermKind::SBVToFp { eb, sb, .. } => assert_eq!((eb, sb), (8, 24)),
        other => panic!("expected SBVToFp, got {other:?}"),
    }
    assert_round_trips(&m, rhs, &["(declare-const bv (_ BitVec 32))"]);
}

#[test]
fn to_fp_unsigned_accepts_rounding_mode() {
    let (m, asserts) = parse_asserts(
        r#"
        (set-logic QF_FP)
        (declare-const bv (_ BitVec 16))
        (declare-const a (_ FloatingPoint 5 11))
        (assert (= a ((_ to_fp_unsigned 5 11) RTP bv)))
        "#,
    );
    let rhs = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::UBVToFp { .. }));
    match kind(&m, rhs) {
        TermKind::UBVToFp { eb, sb, .. } => assert_eq!((eb, sb), (5, 11)),
        other => panic!("expected UBVToFp, got {other:?}"),
    }
    assert_round_trips(&m, rhs, &["(declare-const bv (_ BitVec 16))"]);
}

#[test]
fn fp_to_sbv_and_to_ubv_accept_rounding_mode() {
    let (m, asserts) = parse_asserts(
        r#"
        (set-logic QF_FP)
        (declare-const a (_ FloatingPoint 8 24))
        (declare-const b (_ FloatingPoint 8 24))
        (assert (= ((_ fp.to_sbv 32) RTZ a) ((_ fp.to_ubv 32) RTZ b)))
        "#,
    );
    let (lhs, rhs) = eq_operands(&m, asserts[0]);
    let sbv_side = if matches!(kind(&m, lhs), TermKind::FpToSBV { .. }) {
        lhs
    } else {
        rhs
    };
    let ubv_side = if sbv_side == lhs { rhs } else { lhs };
    match kind(&m, sbv_side) {
        TermKind::FpToSBV { width, .. } => assert_eq!(width, 32),
        other => panic!("expected FpToSBV, got {other:?}"),
    }
    match kind(&m, ubv_side) {
        TermKind::FpToUBV { width, .. } => assert_eq!(width, 32),
        other => panic!("expected FpToUBV, got {other:?}"),
    }
    let decls: &[&str] = &[
        "(declare-const a (_ FloatingPoint 8 24))",
        "(declare-const b (_ FloatingPoint 8 24))",
    ];
    assert_round_trips(&m, sbv_side, decls);
    assert_round_trips(&m, ubv_side, decls);
}

// ---------------------------------------------------------------------------
// BV-NEG-01: bvneg must be recognized.
// ---------------------------------------------------------------------------

#[test]
fn bvneg_is_recognized() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= (bvneg x) #x00))
        "#,
    );
    // mk_bv_neg lowers to two's-complement negation `0 - x`; pre-fix this
    // was an unrecognized Bool-sorted uninterpreted apply of "bvneg".
    let neg_side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::BvSub(..)));
    match kind(&m, neg_side) {
        TermKind::BvSub(zero, arg) => {
            assert!(matches!(
                kind(&m, zero),
                TermKind::BitVecConst { width: 8, .. }
            ));
            assert_eq!(sort_kind(&m, arg), SortKind::BitVec(8));
        }
        other => panic!("expected bvneg to lower to BvSub, got {other:?}"),
    }
    assert_eq!(sort_kind(&m, neg_side), SortKind::BitVec(8));
    assert_round_trips(&m, neg_side, &["(declare-const x (_ BitVec 8))"]);
}

// ---------------------------------------------------------------------------
// BV-EXT-01: bvnand, bvnor, bvxnor, bvcomp, bvsmod must be recognized.
// ---------------------------------------------------------------------------

#[test]
fn bvnand_lowers_to_not_and() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const a (_ BitVec 4))
        (declare-const b (_ BitVec 4))
        (declare-const c (_ BitVec 4))
        (assert (= (bvnand a b) c))
        "#,
    );
    let nand_side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::BvNot(..)));
    match kind(&m, nand_side) {
        TermKind::BvNot(inner) => assert!(matches!(kind(&m, inner), TermKind::BvAnd(..))),
        other => panic!("expected bvnand to lower to BvNot(BvAnd(..)), got {other:?}"),
    }
    assert_eq!(sort_kind(&m, nand_side), SortKind::BitVec(4));
    let decls: &[&str] = &[
        "(declare-const a (_ BitVec 4))",
        "(declare-const b (_ BitVec 4))",
    ];
    assert_round_trips(&m, nand_side, decls);
}

#[test]
fn bvnor_lowers_to_not_or() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const a (_ BitVec 4))
        (declare-const b (_ BitVec 4))
        (declare-const c (_ BitVec 4))
        (assert (= (bvnor a b) c))
        "#,
    );
    let nor_side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::BvNot(..)));
    match kind(&m, nor_side) {
        TermKind::BvNot(inner) => assert!(matches!(kind(&m, inner), TermKind::BvOr(..))),
        other => panic!("expected bvnor to lower to BvNot(BvOr(..)), got {other:?}"),
    }
    let decls: &[&str] = &[
        "(declare-const a (_ BitVec 4))",
        "(declare-const b (_ BitVec 4))",
    ];
    assert_round_trips(&m, nor_side, decls);
}

#[test]
fn bvxnor_lowers_to_not_xor() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const a (_ BitVec 4))
        (declare-const b (_ BitVec 4))
        (declare-const c (_ BitVec 4))
        (assert (= (bvxnor a b) c))
        "#,
    );
    let xnor_side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::BvNot(..)));
    match kind(&m, xnor_side) {
        TermKind::BvNot(inner) => assert!(matches!(kind(&m, inner), TermKind::BvXor(..))),
        other => panic!("expected bvxnor to lower to BvNot(BvXor(..)), got {other:?}"),
    }
    let decls: &[&str] = &[
        "(declare-const a (_ BitVec 4))",
        "(declare-const b (_ BitVec 4))",
    ];
    assert_round_trips(&m, xnor_side, decls);
}

#[test]
fn bvcomp_produces_a_single_bit_result() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const a (_ BitVec 4))
        (declare-const b (_ BitVec 4))
        (declare-const c (_ BitVec 1))
        (assert (= (bvcomp a b) c))
        "#,
    );
    let comp_side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::Ite(..)));
    assert_eq!(sort_kind(&m, comp_side), SortKind::BitVec(1));
    let decls: &[&str] = &[
        "(declare-const a (_ BitVec 4))",
        "(declare-const b (_ BitVec 4))",
    ];
    assert_round_trips(&m, comp_side, decls);
}

#[test]
fn bvsmod_is_recognized_and_correctly_typed() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const a (_ BitVec 8))
        (declare-const b (_ BitVec 8))
        (declare-const c (_ BitVec 8))
        (assert (= (bvsmod a b) c))
        "#,
    );
    // Pre-fix this degraded to a Bool-sorted `Apply("bvsmod", ..)`; it must
    // now build a genuine bit-vector-sorted term (the SMT-LIB `bvsmod`
    // definition's outermost connective is an `ite`).
    let smod_side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::Ite(..)));
    assert_eq!(sort_kind(&m, smod_side), SortKind::BitVec(8));
    let decls: &[&str] = &[
        "(declare-const a (_ BitVec 8))",
        "(declare-const b (_ BitVec 8))",
    ];
    assert_round_trips(&m, smod_side, decls);
}

// ---------------------------------------------------------------------------
// FP-CONV-SIB-01: fp.to_real, the (fp ...) literal, and indexed FP special
// values must be recognized.
// ---------------------------------------------------------------------------

#[test]
fn fp_to_real_is_recognized() {
    let (m, asserts) = parse_asserts(
        r#"
        (set-logic QF_FP)
        (declare-const x (_ FloatingPoint 8 24))
        (declare-const r Real)
        (assert (= r (fp.to_real x)))
        "#,
    );
    let rhs = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::FpToReal(_)));
    assert_eq!(sort_kind(&m, rhs), SortKind::Real);
    assert_round_trips(&m, rhs, &["(declare-const x (_ FloatingPoint 8 24))"]);
}

#[test]
fn fp_literal_bit_triple_constructor_is_recognized() {
    // float32: eb=8, sb=24 -> sign is 1 bit, exponent is 8 bits, the
    // significand literal here is the 23 explicit (non-hidden) bits.
    let (m, asserts) = parse_asserts(
        r#"
        (set-logic QF_FP)
        (declare-const x (_ FloatingPoint 8 24))
        (assert (= x (fp #b1 #b10000001 #b01000000000000000000000)))
        "#,
    );
    let rhs = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::FpLit { .. }));
    match kind(&m, rhs) {
        TermKind::FpLit {
            sign,
            exp,
            sig,
            eb,
            sb,
        } => {
            assert!(sign);
            assert_eq!(exp, num_bigint::BigInt::from(0b10000001));
            assert_eq!(sig, num_bigint::BigInt::from(0b01000000000000000000000i64));
            assert_eq!((eb, sb), (8, 24));
        }
        other => panic!("expected FpLit, got {other:?}"),
    }
    assert_eq!(
        sort_kind(&m, rhs),
        SortKind::FloatingPoint { eb: 8, sb: 24 }
    );
    // Not round-tripped: the pre-existing basic printer formats the FpLit's
    // exponent/significand fields with `#b{decimal_value}` instead of an
    // actual zero-padded binary string (a separate, out-of-scope printer
    // defect), so its printed form is not guaranteed to re-lex as a valid
    // `#b`-literal. Structural verification above is the meaningful check
    // for this parser-level fix.
}

#[test]
fn fp_special_value_constants_are_recognized() {
    let (m, asserts) = parse_asserts(
        r#"
        (set-logic QF_FP)
        (declare-const x (_ FloatingPoint 8 24))
        (assert (fp.eq x (_ +oo 8 24)))
        (assert (fp.eq x (_ -oo 8 24)))
        (assert (fp.eq x (_ +zero 8 24)))
        (assert (fp.eq x (_ -zero 8 24)))
        (assert (fp.eq x (_ NaN 8 24)))
        "#,
    );
    let rhs_of = |t: TermId| match kind(&m, t) {
        TermKind::FpEq(_, b) => b,
        other => panic!("expected FpEq, got {other:?}"),
    };
    assert!(matches!(
        kind(&m, rhs_of(asserts[0])),
        TermKind::FpPlusInfinity { eb: 8, sb: 24 }
    ));
    assert!(matches!(
        kind(&m, rhs_of(asserts[1])),
        TermKind::FpMinusInfinity { eb: 8, sb: 24 }
    ));
    assert!(matches!(
        kind(&m, rhs_of(asserts[2])),
        TermKind::FpPlusZero { eb: 8, sb: 24 }
    ));
    assert!(matches!(
        kind(&m, rhs_of(asserts[3])),
        TermKind::FpMinusZero { eb: 8, sb: 24 }
    ));
    assert!(matches!(
        kind(&m, rhs_of(asserts[4])),
        TermKind::FpNaN { eb: 8, sb: 24 }
    ));
    for &a in &asserts {
        assert_round_trips(&m, rhs_of(a), &[]);
    }
}

// ---------------------------------------------------------------------------
// SORT-BUILTIN-01: RoundingMode / RegLan must not silently become ordinary
// uninterpreted sorts.
// ---------------------------------------------------------------------------

/// Parse a script that must fail, returning the lowercased error text.
fn parse_error(script: &str) -> String {
    let mut manager = TermManager::new();
    let err = parse_script(script, &mut manager)
        .expect_err("script was expected to be rejected, but it parsed");
    format!("{err:?}").to_lowercase()
}

/// `RoundingMode` is a first-class sort: a constant may be declared at it.
///
/// This replaces the old pin asserting the *opposite*. Back then rounding
/// modes existed only as literals baked into `fp.*` operators at parse time,
/// so a `RoundingMode`-sorted symbol could not be represented at all and an
/// honest rejection was the best available answer. It is now
/// `SortKind::RoundingMode`, and its five inhabitants are nullary `Var` terms.
#[test]
fn rounding_mode_sort_is_declarable() {
    let mut manager = TermManager::new();
    let commands = parse_script("(declare-const m RoundingMode)", &mut manager)
        .expect("a RoundingMode-sorted constant must be declarable");
    let [Command::DeclareConst(name, sort)] = commands.as_slice() else {
        panic!("expected a single declare-const, got {commands:?}");
    };
    assert_eq!(name, "m");
    // The sort *string* is what `Command::DeclareConst` carries to the solver,
    // where `Context::parse_sort_name` resolves it back. If the two spellings
    // ever disagree, the declared constant and its occurrences in terms end up
    // at different `SortId`s — and, since `mk_var` hash-conses on
    // `(name, sort)`, become two unrelated terms.
    assert_eq!(sort, "RoundingMode");
}

/// Both spellings of a mode denote the *same term*, not merely equal ones.
///
/// `mk_rounding_mode` interns under the canonical long name, so the short
/// alias resolves to that identical `TermId` — which is what lets EUF decide
/// `(= m RNE)` against a mode written either way with no extra axiom.
#[test]
fn short_and_long_rounding_mode_spellings_intern_to_one_term() {
    let (manager, asserts) = parse_asserts(
        "(declare-const m RoundingMode)
         (assert (= m RNE))
         (assert (= m roundNearestTiesToEven))
         (assert (= m RTZ))",
    );
    let [short, long, other] = asserts.as_slice() else {
        panic!("expected three assertions, got {}", asserts.len());
    };
    assert_eq!(
        short, long,
        "RNE and roundNearestTiesToEven must intern to the same term"
    );
    assert_ne!(short, other, "RNE and RTZ must stay different terms");

    // And the mode term itself carries the reserved sort, spelled long.
    let TermKind::Eq(_, mode) = manager.get(*short).expect("assertion").kind else {
        panic!("expected an equality");
    };
    let node = manager.get(mode).expect("mode term");
    assert!(
        manager
            .sorts
            .get(node.sort)
            .is_some_and(oxiz_core::sort::Sort::is_rounding_mode)
    );
    let TermKind::Var(spur) = node.kind else {
        panic!("a rounding mode must be a nullary Var, got {:?}", node.kind);
    };
    assert_eq!(manager.resolve_str(spur), "roundNearestTiesToEven");
}

/// The positions whose finiteness no closure axiom can reach are rejected —
/// honestly, naming the v1 limitation rather than claiming the sort is
/// unsupported.
///
/// Accepting any of these would silently give `RoundingMode` the behaviour of
/// an *infinite* free sort: `(distinct (g 1) .. (g 6))` over
/// `(declare-fun g (Int) RoundingMode)` would answer `sat`.
#[test]
fn rounding_mode_is_rejected_where_no_closure_axiom_can_reach() {
    for (script, position) in [
        ("(declare-fun g (Int) RoundingMode)", "result sort"),
        ("(declare-fun h (RoundingMode) Int)", "argument sort"),
        ("(declare-const a (Array Int RoundingMode))", "array range"),
        ("(declare-const b (Array RoundingMode Int))", "array domain"),
        (
            "(declare-datatype Box ((box (unbox RoundingMode))))",
            "datatype field sort",
        ),
    ] {
        let msg = parse_error(script);
        assert!(
            msg.contains("roundingmode"),
            "{position}: error should name the sort: {msg}"
        );
        assert!(
            msg.contains("nullary"),
            "{position}: error should say nullary declaration position is the supported one: {msg}"
        );
        assert!(
            msg.contains(&position.to_lowercase()),
            "{position}: error should name the offending position: {msg}"
        );
    }
}

/// The nullary spelling of `declare-fun` is a constant, so it is accepted.
#[test]
fn nullary_declare_fun_at_rounding_mode_is_accepted() {
    let mut manager = TermManager::new();
    parse_script(
        "(declare-fun m () RoundingMode)(assert (= m RTZ))",
        &mut manager,
    )
    .expect("a nullary declare-fun at RoundingMode is a constant declaration");
}

/// A symbolic rounding mode compiles into a five-way `ite` case split whose
/// leaves are ordinary concrete `FpAdd` nodes — so the floating-point theory
/// never has to know that a symbolic mode exists.
#[test]
fn symbolic_rounding_mode_expands_to_a_five_way_case_split() {
    let (manager, asserts) = parse_asserts(
        "(declare-const m RoundingMode)
         (declare-const x (_ FloatingPoint 8 24))
         (declare-const y (_ FloatingPoint 8 24))
         (declare-const z (_ FloatingPoint 8 24))
         (assert (= z (fp.add m x y)))",
    );
    let [assertion] = asserts.as_slice() else {
        panic!("expected one assertion");
    };
    let TermKind::Eq(_, rhs) = manager.get(*assertion).expect("assertion").kind else {
        panic!("expected an equality");
    };

    // Walk the else-chain: four `ite` levels, then a bare `FpAdd` — the
    // unguarded final branch the closure axiom is what makes sound.
    let mut modes = Vec::new();
    let mut current = rhs;
    loop {
        match &manager.get(current).expect("case-split node").kind {
            TermKind::Ite(_, then_branch, else_branch) => {
                let TermKind::FpAdd(rm, _, _) = manager.get(*then_branch).expect("branch").kind
                else {
                    panic!("each taken branch must be a concrete fp.add");
                };
                modes.push(rm);
                current = *else_branch;
            }
            TermKind::FpAdd(rm, _, _) => {
                modes.push(*rm);
                break;
            }
            other => panic!("unexpected node in the case split: {other:?}"),
        }
    }
    assert_eq!(
        modes,
        oxiz_core::ast::RoundingMode::ALL.to_vec(),
        "the split must cover all five modes, in order, with RTZ as the final else"
    );
}

/// A `RoundingMode` quantifier binder is *relativized* at parse time.
///
/// Nothing else can do it: the solver's closure axiom attaches to declared
/// constants, and a bound variable has no declaration. Without relativization
/// `forall` would range over an unconstrained free sort — strictly stronger
/// than "for all five modes" — and `exists` could witness with a value that is
/// no rounding mode at all.
#[test]
fn rounding_mode_binders_are_relativized_to_the_five_modes() {
    let (manager, asserts) = parse_asserts(
        "(declare-const p Bool)
         (assert (forall ((m RoundingMode)) p))
         (assert (exists ((m RoundingMode)) p))
         (assert (forall ((i Int)) p))",
    );
    let [universal, existential, untouched] = asserts.as_slice() else {
        panic!("expected three assertions");
    };

    // forall: the body became `(=> closure p)`.
    let TermKind::Forall { body, .. } = manager.get(*universal).expect("forall").kind else {
        panic!("expected a forall");
    };
    assert!(
        matches!(manager.get(body).expect("body").kind, TermKind::Implies(..)),
        "a relativized forall body must be an implication"
    );

    // exists: the body became `(and closure p)`.
    let TermKind::Exists { body, .. } = manager.get(*existential).expect("exists").kind else {
        panic!("expected an exists");
    };
    assert!(
        matches!(manager.get(body).expect("body").kind, TermKind::And(_)),
        "a relativized exists body must be a conjunction"
    );

    // A non-RoundingMode binder is left exactly as it was: the body is the
    // bare `p`, with no guard wrapped around it.
    let TermKind::Forall { body, .. } = manager.get(*untouched).expect("forall").kind else {
        panic!("expected a forall");
    };
    assert!(
        matches!(manager.get(body).expect("body").kind, TermKind::Var(_)),
        "a non-RoundingMode binder must not be relativized"
    );
}

/// `RegLan` stays un-declarable, and the message says *why* honestly.
///
/// The rejection is load-bearing, not a "not implemented" placeholder:
/// `RegLan` names the built-in sort every regular-expression term is interned
/// at (`TermManager::reglan_sort`), so letting a user declare a symbol of that
/// sort would alias a free constant into the regex encoding's own namespace.
/// The message must therefore not claim the sublanguage is missing — it is
/// fully supported, as [`reglan_reserved_name_does_not_disable_re_operators`]
/// asserts on the very next lines.
#[test]
fn reglan_sort_is_honestly_rejected_not_silently_uninterpreted() {
    let mut manager = TermManager::new();
    let err = parse_script("(declare-const r RegLan)", &mut manager)
        .expect_err("declaring a RegLan-sorted constant must not silently succeed");
    let msg = format!("{err:?}").to_lowercase();
    assert!(msg.contains("reglan"), "error should mention RegLan: {msg}");
    assert!(
        msg.contains("reserved"),
        "error must say the name is reserved, not that the theory is missing: {msg}"
    );
    assert!(
        !msg.contains("not yet implemented") && !msg.contains("not implemented"),
        "the re.* sublanguage IS implemented; the message must not claim otherwise: {msg}"
    );
}

/// The counterpart of the rejection above: reserving the *name* must leave the
/// regular-language operators themselves fully usable. `re.++`, `str.to_re`
/// and `re.allchar` still parse under `str.in_re`, each interned at the
/// reserved `RegLan` sort.
#[test]
fn reglan_reserved_name_does_not_disable_re_operators() {
    let (manager, asserts) = parse_asserts(
        r#"(declare-const s String)
           (assert (str.in_re s (re.++ (str.to_re "a") re.allchar)))"#,
    );
    let [membership] = asserts.as_slice() else {
        panic!("expected exactly one assertion, got {}", asserts.len());
    };
    let TermKind::StrInRe(_, re) = manager.get(*membership).expect("assert term").kind else {
        panic!("expected a str.in_re membership atom");
    };

    // The regex operand is a `RegLan`-sorted `re.++` node whose two arguments
    // are `str.to_re` and `re.allchar`, all at the same reserved sort.
    let concat = manager.get(re).expect("regex operand");
    let reglan_sort = concat.sort;
    match &manager.sorts.get(reglan_sort).expect("regex sort").kind {
        SortKind::Uninterpreted(spur) => assert_eq!(
            manager.resolve_str(*spur),
            "RegLan",
            "regex nodes must carry the reserved RegLan sort"
        ),
        other => panic!("regex node must be RegLan-sorted, got {other:?}"),
    }
    let TermKind::Apply { func, ref args } = concat.kind else {
        panic!("expected an re.++ Apply node, got {:?}", concat.kind)
    };
    assert_eq!(manager.resolve_str(func), "re.++");
    let [to_re, allchar] = args.as_slice() else {
        panic!("re.++ must have two operands, got {}", args.len());
    };
    for (arg, expected) in [(to_re, "str.to_re"), (allchar, "re.allchar")] {
        let node = manager.get(*arg).expect("regex argument");
        assert_eq!(node.sort, reglan_sort, "{expected} must be RegLan-sorted");
        let TermKind::Apply { func, .. } = node.kind else {
            panic!("expected {expected} Apply node, got {:?}", node.kind)
        };
        assert_eq!(manager.resolve_str(func), expected);
    }
}

// ---------------------------------------------------------------------------
// R1: parse_sort must not overflow the stack on deeply nested sorts.
// ---------------------------------------------------------------------------

#[test]
fn deeply_nested_array_sort_is_rejected_not_a_stack_overflow() {
    // Nest well past the 512-level guard via the Array sort's *range*
    // position (right-recursion), matching how `(Array (Array (Array ...
    // Int) Int) Int)` grows in practice.
    let depth = 2000;
    let mut sort_expr = "Int".to_string();
    for _ in 0..depth {
        sort_expr = format!("(Array Int {sort_expr})");
    }
    let script = format!("(declare-const x {sort_expr})");
    let mut manager = TermManager::new();
    let err = parse_script(&script, &mut manager)
        .expect_err("pathologically deep nested sort must be rejected, not overflow the stack");
    let msg = format!("{err:?}").to_lowercase();
    assert!(
        msg.contains("deep") || msg.contains("depth"),
        "error should mention nesting depth: {msg}"
    );
}

#[test]
fn moderately_nested_array_sort_still_parses() {
    // A sanity control: nesting well under the 512-level cap must still
    // parse successfully (the guard must not be so tight it rejects
    // legitimate, if unusual, inputs).
    let depth = 20;
    let mut sort_expr = "Int".to_string();
    for _ in 0..depth {
        sort_expr = format!("(Array Int {sort_expr})");
    }
    let script = format!("(declare-const x {sort_expr})");
    let mut manager = TermManager::new();
    parse_script(&script, &mut manager).expect("moderately nested sort should parse");
}

// ---------------------------------------------------------------------------
// todo-1151: try_mk_bv_concat must not silently fabricate a width for a
// non-bit-vector operand.
// ---------------------------------------------------------------------------

#[test]
fn bv_concat_computes_exact_combined_width_for_valid_operands() {
    let mut m = TermManager::new();
    let a = m.mk_bitvec(0i64, 5);
    let b = m.mk_bitvec(0i64, 3);
    let concat = m
        .try_mk_bv_concat(a, b)
        .expect("two bit-vector operands must concat");
    assert_eq!(sort_kind(&m, concat), SortKind::BitVec(8));
}

// A non-bit-vector operand is now a returned error rather than a
// `debug_assert!`. That is strictly stronger than the assertion it replaces:
// the check no longer disappears under `--release` (where the old code fell
// back to a fabricated `32 + 8 = 40`-bit result), so this test needs no
// `cfg(debug_assertions)` gate and covers release builds too.
#[test]
fn bv_concat_rejects_non_bitvector_operand() {
    let mut m = TermManager::new();
    let int_term = m.mk_int(5);
    let bv_term = m.mk_bitvec(3i64, 8);
    let err = m
        .try_mk_bv_concat(int_term, bv_term)
        .expect_err("an Int operand must be rejected, not widened to a fabricated 40 bits");
    let rendered = err.to_string();
    // The message must name both operand sorts, so a caller can tell which
    // side was wrong.
    assert!(
        rendered.contains("Int") && rendered.contains("BitVec(8)"),
        "error should name both operand sorts, got: {rendered}"
    );
}

// The same guarantee at the parser level: an ill-typed `concat` or indexed
// bit-vector operator must surface as a parse error instead of interning a
// term at a fabricated width. These run identically in debug and release.
#[test]
fn parser_rejects_ill_typed_bitvector_operators() {
    for script in [
        // `concat` with a non-bit-vector operand: used to intern a
        // fabricated 32 + 8 = 40-bit term.
        "(declare-const x Int)(declare-const y (_ BitVec 8))(assert (= (concat x y) y))",
        // `zero_extend` prepends zeros via concat.
        "(declare-const x Int)(assert (= ((_ zero_extend 4) x) x))",
        // `repeat` folds the operand into itself via concat.
        "(declare-const x Int)(assert (= ((_ repeat 3) x) x))",
    ] {
        let mut m = TermManager::new();
        let err = parse_script(script, &mut m)
            .err()
            .unwrap_or_else(|| panic!("ill-typed bit-vector application must fail: {script}"));
        // Assert the *specific* error, not merely that something failed:
        // a bare `is_err()` would also pass if some unrelated upstream guard
        // rejected the script, leaving the concat width fabrication untested.
        let rendered = err.to_string();
        assert!(
            rendered.contains("concat") && rendered.contains("Int"),
            "expected the concat sort error naming the Int operand for {script}, got: {rendered}"
        );
    }
}

// ---------------------------------------------------------------------------
// todo-1174: lexer leading-zero numerals and arbitrary-precision (_ bvN M).
// ---------------------------------------------------------------------------

#[test]
fn lexer_rejects_leading_zero_numeral() {
    let mut lexer = Lexer::new("007");
    let _ = lexer.next_token();
    assert!(
        lexer.has_errors(),
        "a leading-zero numeral must be recorded as a lexical error"
    );
}

#[test]
fn lexer_accepts_bare_zero_numeral() {
    let mut lexer = Lexer::new("0");
    let _ = lexer.next_token();
    assert!(!lexer.has_errors(), "a bare '0' is a valid SMT-LIB numeral");
}

#[test]
fn lexer_does_not_flag_leading_zeros_in_a_decimals_fractional_part() {
    // The SMT-LIB `<decimal>` grammar is `<numeral>.0*<numeral>`, so a
    // fractional part like the "001" in "0.001" legitimately starts with
    // zeros and must not be flagged.
    let mut lexer = Lexer::new("0.001");
    let _ = lexer.next_token();
    assert!(
        !lexer.has_errors(),
        "leading zeros in a decimal's fractional part are valid"
    );
}

#[test]
fn indexed_bitvector_literal_supports_values_beyond_i64() {
    // i64::MAX = 9223372036854775807 (19 digits); use a value with more
    // digits than that to confirm the parser no longer truncates to i64.
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 128))
        (assert (= x (_ bv123456789012345678901234567890 128)))
        "#,
    );
    let rhs = eq_side_matching(&m, asserts[0], |k| {
        matches!(k, TermKind::BitVecConst { .. })
    });
    match kind(&m, rhs) {
        TermKind::BitVecConst { value, width } => {
            assert_eq!(width, 128);
            let expected: num_bigint::BigInt = "123456789012345678901234567890"
                .parse()
                .expect("literal should parse as BigInt");
            assert_eq!(value, expected);
        }
        other => panic!("expected BitVecConst, got {other:?}"),
    }
}
