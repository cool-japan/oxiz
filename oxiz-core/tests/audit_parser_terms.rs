//! Regression tests for audited defects in the SMT-LIB term parser
//! (`oxiz-core/src/smtlib/parser/terms.rs`).
//!
//! Each test reproduces a specific soundness bug that the parser used to
//! exhibit and asserts the corrected behavior:
//!
//! 1. `(div a b)` / `(mod a b)` were lowered to subtraction.
//! 2. Real `/`, `abs`, `to_real`, `to_int`, `is_int`, `divisible` fell through
//!    to Bool-sorted uninterpreted applies (or were unsupported).
//! 3. Undeclared symbols silently became fresh Bool variables.
//! 4. Indexed BV ops (`zero_extend`, `sign_extend`, `rotate_*`, `repeat`)
//!    degraded to Bool-sorted generic applies.

use oxiz_core::ast::{TermKind, TermManager};
use oxiz_core::smtlib::{Command, parse_script};
use oxiz_core::sort::SortKind;

/// Parse a full script, returning the manager plus the asserted term ids in
/// order.
fn parse_asserts(script: &str) -> (TermManager, Vec<oxiz_core::ast::TermId>) {
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

fn kind(manager: &TermManager, t: oxiz_core::ast::TermId) -> TermKind {
    manager.get(t).expect("term should exist").kind.clone()
}

/// Return the two operand term ids of an equality (`mk_eq` canonicalizes
/// operand order, so callers must not assume which side is which).
fn eq_operands(
    m: &TermManager,
    t: oxiz_core::ast::TermId,
) -> (oxiz_core::ast::TermId, oxiz_core::ast::TermId) {
    match kind(m, t) {
        TermKind::Eq(a, b) => (a, b),
        other => panic!("expected equality, got {other:?}"),
    }
}

/// Find, among the two operands of an equality, the one satisfying `pred`.
fn eq_side_matching(
    m: &TermManager,
    t: oxiz_core::ast::TermId,
    pred: impl Fn(&TermKind) -> bool,
) -> oxiz_core::ast::TermId {
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

// ---------------------------------------------------------------------------
// Finding 1: div / mod must not become subtraction.
// ---------------------------------------------------------------------------

#[test]
fn div_is_not_subtraction() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const a Int)
        (declare-const b Int)
        (assert (= (div a b) 0))
        "#,
    );
    // One side of the equality must be a Div node, never Sub.
    let (a, b) = eq_operands(&m, asserts[0]);
    let ka = kind(&m, a);
    let kb = kind(&m, b);
    assert!(
        matches!(ka, TermKind::Div(_, _)) || matches!(kb, TermKind::Div(_, _)),
        "expected Div, got {ka:?} / {kb:?} (regression: div lowered to subtraction)"
    );
    assert!(
        !matches!(ka, TermKind::Sub(_, _)) && !matches!(kb, TermKind::Sub(_, _)),
        "div must not be Sub"
    );
}

#[test]
fn mod_is_not_subtraction() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const a Int)
        (declare-const b Int)
        (assert (= (mod a b) 0))
        "#,
    );
    let _ = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::Mod(_, _)));
}

// ---------------------------------------------------------------------------
// Finding 2: real division and Int/Real conversions.
// ---------------------------------------------------------------------------

#[test]
fn real_division_stays_in_arithmetic() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x Real)
        (assert (= (/ x 3.0) 1.0))
        "#,
    );
    let _ = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::Div(_, _)));
}

#[test]
fn abs_lowers_to_ite() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x Int)
        (assert (= (abs x) 5))
        "#,
    );
    let _ = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::Ite(_, _, _)));
}

#[test]
fn to_real_is_value_preserving_and_not_bool_apply() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const n Int)
        (assert (< (to_real n) 2.5))
        "#,
    );
    let TermKind::Lt(lhs, _) = kind(&m, asserts[0]) else {
        panic!("expected <");
    };
    // to_real(n) must be the integer variable itself (the injection), never a
    // Bool-sorted uninterpreted apply.
    match kind(&m, lhs) {
        TermKind::Var(_) => {}
        other => panic!("expected Var for to_real arg, got {other:?}"),
    }
    // Its sort must NOT be Bool.
    let sort = m.get(lhs).unwrap().sort;
    assert!(
        !matches!(m.sorts.get(sort).unwrap().kind, SortKind::Bool),
        "to_real result must not be Bool-sorted"
    );
}

#[test]
fn to_int_is_honest_error_not_silent_wrong() {
    // to_int needs a genuine floor operator we do not have; the parser must
    // reject it rather than silently produce a wrong (Bool-sorted apply) term.
    let mut manager = TermManager::new();
    let script = r#"
        (declare-const x Real)
        (assert (= (to_int x) 3))
    "#;
    assert!(
        parse_script(script, &mut manager).is_err(),
        "to_int must be an honest parse error, not a silent wrong term"
    );
}

#[test]
fn is_int_is_honest_error() {
    let mut manager = TermManager::new();
    let script = r#"
        (declare-const x Real)
        (assert (is_int x))
    "#;
    assert!(parse_script(script, &mut manager).is_err());
}

#[test]
fn divisible_lowers_to_mod_equals_zero() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x Int)
        (assert ((_ divisible 3) x))
        "#,
    );
    // ((_ divisible 3) x) lowers to (= (mod x 3) 0).
    let _ = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::Mod(_, _)));
}

// ---------------------------------------------------------------------------
// Finding 3: undeclared symbols must be rejected in a real script.
// ---------------------------------------------------------------------------

#[test]
fn undeclared_symbol_in_script_is_error() {
    let mut manager = TermManager::new();
    // `typo_var` is never declared; declaring `x` establishes script context.
    let script = r#"
        (declare-const x Int)
        (assert (< x typo_var))
    "#;
    let err = parse_script(script, &mut manager);
    assert!(
        err.is_err(),
        "undeclared symbol must be rejected, not turned into a fresh Bool var"
    );
}

#[test]
fn declared_symbols_still_parse() {
    // Sanity: a fully declared script must still parse fine.
    let mut manager = TermManager::new();
    let script = r#"
        (declare-const x Int)
        (declare-const y Int)
        (assert (< x y))
    "#;
    assert!(parse_script(script, &mut manager).is_ok());
}

#[test]
fn quantifier_bound_vars_not_rejected() {
    // Bound variables are not "declared" via declare-const but must still
    // resolve, even under the strict undeclared-symbol rule.
    let mut manager = TermManager::new();
    let script = r#"
        (declare-const c Int)
        (assert (forall ((i Int)) (> (+ i c) 0)))
    "#;
    assert!(
        parse_script(script, &mut manager).is_ok(),
        "quantifier-bound variable wrongly treated as undeclared"
    );
}

// ---------------------------------------------------------------------------
// Finding 4: indexed BV ops must be real bit-vector terms.
// ---------------------------------------------------------------------------

fn bv_width(m: &TermManager, t: oxiz_core::ast::TermId) -> Option<u32> {
    let sort = m.get(t)?.sort;
    m.sorts.get(sort)?.bitvec_width()
}

/// Return the operand of an equality that is a `BvConcat` (the lowered
/// extend/rotate/repeat term), regardless of canonicalized operand order.
fn concat_side(m: &TermManager, t: oxiz_core::ast::TermId) -> oxiz_core::ast::TermId {
    eq_side_matching(m, t, |k| matches!(k, TermKind::BvConcat(_, _)))
}

#[test]
fn zero_extend_produces_bitvector_of_correct_width() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= ((_ zero_extend 8) x) #x0000))
        "#,
    );
    // 8-bit value zero-extended by 8 => 16-bit bit-vector via concat.
    let side = concat_side(&m, asserts[0]);
    assert_eq!(bv_width(&m, side), Some(16), "zero_extend width wrong");
}

#[test]
fn sign_extend_produces_bitvector_of_correct_width() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= ((_ sign_extend 4) x) #x000))
        "#,
    );
    let side = concat_side(&m, asserts[0]);
    assert_eq!(bv_width(&m, side), Some(12), "sign_extend width wrong");
}

#[test]
fn rotate_left_preserves_width() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= ((_ rotate_left 3) x) #x00))
        "#,
    );
    // A non-trivial rotation is a concat of two extracts, not a Bool apply.
    let side = concat_side(&m, asserts[0]);
    assert_eq!(
        bv_width(&m, side),
        Some(8),
        "rotate_left must preserve width"
    );
}

#[test]
fn repeat_produces_correct_width() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= ((_ repeat 3) x) #x000000))
        "#,
    );
    let side = concat_side(&m, asserts[0]);
    assert_eq!(bv_width(&m, side), Some(24), "repeat width wrong");
}

// ---------------------------------------------------------------------------
// Nesting depth guard: pathological nesting yields an error, not a crash.
//
// The parser collects operands on an explicit heap frame stack rather than by
// recursive descent, so its native stack usage does not grow with the nesting
// depth of the input. `MAX_PARSE_DEPTH` (1024) survives as a *resource* bound,
// and its error message is part of the observable contract.
//
// The recursive-descent version needed ~2.9 KiB of native stack per nesting
// level in the release profile, so reaching the 1024 limit took ~3 MiB — more
// than a libtest thread's ~2 MiB and far more than the ~1 MiB an embedder's
// worker thread may have. The process then died of a stack overflow *before*
// the limit could report anything, which is precisely what the limit exists to
// prevent. `deeply_nested_term_survives_a_one_mib_stack` below is the test that
// actually pins that down; the rest would pass on a big enough stack either
// way.
// ---------------------------------------------------------------------------

/// `(- (- (- ... 0 ...)))` nested `depth` levels deep.
fn nested_minus(depth: usize) -> String {
    let mut s = String::with_capacity(depth * 4 + 1);
    for _ in 0..depth {
        s.push_str("(- ");
    }
    s.push('0');
    for _ in 0..depth {
        s.push(')');
    }
    s
}

/// The deepest `nested_minus` chain accepted inside `(assert (= <chain> 0))`.
///
/// `MAX_PARSE_DEPTH` is 1024 and counts every term position: the asserted `=`
/// is at depth 1 and its operands at depth 2, so an `n`-level chain occupies
/// depths 2..=n+1 and its innermost `0` sits at depth n+2. The deepest chain
/// that fits is therefore n = 1022.
const DEEPEST_ACCEPTED: usize = 1022;

#[test]
fn deeply_nested_term_errors_instead_of_overflowing() {
    // Far deeper than MAX_PARSE_DEPTH.
    let script = format!("(assert (= {} 0))", nested_minus(200_000));
    let mut manager = TermManager::new();
    let err = parse_script(&script, &mut manager)
        .expect_err("deep nesting should be rejected by the depth guard");
    assert!(
        err.to_string().contains("term nesting too deep"),
        "expected the nesting-limit error, got: {err}"
    );
}

#[test]
fn nesting_just_under_the_limit_still_parses() {
    let script = format!("(assert (= {} 0))", nested_minus(DEEPEST_ACCEPTED));
    let mut manager = TermManager::new();
    parse_script(&script, &mut manager)
        .expect("nesting below MAX_PARSE_DEPTH must still be accepted");
}

#[test]
fn nesting_just_over_the_limit_is_rejected() {
    let script = format!("(assert (= {} 0))", nested_minus(DEEPEST_ACCEPTED + 1));
    let mut manager = TermManager::new();
    let err = parse_script(&script, &mut manager)
        .expect_err("nesting above MAX_PARSE_DEPTH must be rejected");
    assert!(
        err.to_string().contains("term nesting too deep"),
        "expected the nesting-limit error, got: {err}"
    );
}

/// The point of the whole exercise: an embedder calling OxiZ from a worker
/// thread with a conventional ~1 MiB stack must get an error *value* back for
/// pathological input, not a process abort.
///
/// A Rust stack overflow is not a panic — it is a fatal runtime abort that
/// `catch_unwind` cannot intercept — so the only way to assert on it is to run
/// the parse on a thread whose stack size is pinned small and observe that the
/// thread returns at all. If the parser ever regains a stack-depth dependence,
/// this test does not fail: it kills the whole test process, which is exactly
/// as loud as it should be.
// STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — this is the
// parser's own depth-guard budget test; scaling the stack down would not
// exercise the guard as an embedder with a conventional 1 MiB stack actually
// sees it.
#[test]
fn deeply_nested_term_survives_a_one_mib_stack() {
    const STACK_SIZE: usize = 1 << 20; // 1 MiB

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let script = format!("(assert (= {} 0))", nested_minus(200_000));
            let mut manager = TermManager::new();
            let err = parse_script(&script, &mut manager)
                .expect_err("deep nesting should be rejected by the depth guard");
            assert!(
                err.to_string().contains("term nesting too deep"),
                "expected the nesting-limit error, got: {err}"
            );
            // A depth the parser accepts must also fit in the same stack, or
            // the limit is not actually usable by such an embedder.
            let ok_script = format!("(assert (= {} 0))", nested_minus(DEEPEST_ACCEPTED));
            let mut manager = TermManager::new();
            parse_script(&ok_script, &mut manager)
                .expect("an accepted nesting depth must parse on a 1 MiB stack too");
        })
        .expect("spawning a 1 MiB-stack thread should succeed");

    handle
        .join()
        .expect("the parse must return on a 1 MiB stack instead of overflowing it");
}

/// Deeply nested *annotations* are the one term-parsing path that still went
/// through a native recursive call after the operand stack was made explicit:
/// an attribute value such as `:pattern (...)` contains terms, so the attribute
/// grammar sits between two term positions. It is driven by the same frame
/// stack now; this pins that down on a small stack as well.
#[test]
fn deeply_nested_annotations_survive_a_small_stack() {
    const STACK_SIZE: usize = 1 << 17; // 128 KiB

    let handle = std::thread::Builder::new()
        .stack_size(STACK_SIZE)
        .spawn(|| {
            let mut s = String::from("0");
            for _ in 0..6_250 {
                s = format!("(! 0 :pattern ({s}))");
            }
            let mut manager = TermManager::new();
            let err = oxiz_core::smtlib::parse_term(&s, &mut manager)
                .expect_err("deeply nested annotations should hit the depth guard");
            assert!(
                err.to_string().contains("term nesting too deep"),
                "expected the nesting-limit error, got: {err}"
            );
        })
        .expect("spawning a 1 MiB-stack thread should succeed");

    handle
        .join()
        .expect("annotation nesting must return on a 1 MiB stack instead of overflowing it");
}

/// Deep nesting must not disturb the *scoped* forms: a `let` chain binds and
/// unbinds through the same frame stack, so a name bound deep inside must not
/// leak back out to an enclosing scope.
#[test]
fn deep_let_nesting_keeps_scopes_balanced() {
    let depth = 500;
    let mut s = String::from("x");
    for i in 0..depth {
        s = format!("(let ((x {i})) {s})");
    }
    let script = format!("(declare-const x Int)(assert (= {s} 0))");
    let mut manager = TermManager::new();
    parse_script(&script, &mut manager).expect("deeply nested lets must parse");

    // After the nested lets close, the outer declaration of `x` must be the one
    // that resolves again.
    let script = format!("(declare-const x Int)(assert (= {s} 0))(assert (= x 1))");
    let mut manager = TermManager::new();
    parse_script(&script, &mut manager).expect("let bindings must not leak out of their scope");
}

// ---------------------------------------------------------------------------
// Finding 5: an undeclared head symbol must not become an unconstrained
// uninterpreted function.
//
// `(assert (not (str.< "abc" "abd")))` used to answer `sat` because `str.<`
// silently became a 2-ary uninterpreted predicate; z3 reports
// `(error "unknown function/constant str.<")`. The rule now is: resolve
// against every declaration table first (define-fun, datatype constructors and
// selectors, declare-fun, declare-const, let), then reject — always in script
// mode, and additionally in bare-term mode when the name lives in a reserved
// SMT-LIB theory namespace.
// ---------------------------------------------------------------------------

/// Parse a script and return the error message, asserting that it failed.
fn parse_script_err(script: &str) -> String {
    let mut manager = TermManager::new();
    match parse_script(script, &mut manager) {
        Ok(_) => panic!("script should have been rejected:\n{script}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn unknown_theory_operators_are_rejected_per_namespace() {
    // One representative undeclared symbol from each reserved namespace.
    for application in [
        "(str.frobnicate s)",
        "(re.frobnicate s)",
        "(seq.nth s 0)",
        "(char.frobnicate s)",
        "(fp.frobnicate s)",
        "(int.frobnicate s)",
        "(bvfrobnicate s)",
    ] {
        let script = format!("(declare-const s Int)(assert (= {application} s))");
        let msg = parse_script_err(&script);
        assert!(
            msg.contains("unknown"),
            "expected an unknown-symbol error for {application}, got: {msg}"
        );
    }
}

#[test]
fn unknown_plain_application_head_is_rejected_in_script_mode() {
    let msg = parse_script_err(
        r#"
        (declare-const x Int)
        (assert (totally_bogus_op x 2))
        "#,
    );
    assert!(
        msg.contains("unknown function/constant totally_bogus_op"),
        "unexpected message: {msg}"
    );
}

#[test]
fn unknown_indexed_identifier_is_rejected() {
    let msg = parse_script_err(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= ((_ frobnicate 3) x) x))
        "#,
    );
    assert!(msg.contains("unknown"), "unexpected message: {msg}");
}

// --- Control tests: QF_UF and friends must not regress. --------------------

#[test]
fn control_declared_function_application_still_uninterpreted() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-fun f (Int Int) Int)
        (declare-const k Int)
        (assert (= (f k 2) 10))
        "#,
    );
    // `f` must still build an uninterpreted Apply, carrying its declared Int
    // return sort (not Bool).
    let side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::Apply { .. }));
    let sort = m.get(side).expect("term exists").sort;
    assert!(
        matches!(m.sorts.get(sort).expect("sort exists").kind, SortKind::Int),
        "declared function application lost its declared return sort"
    );
}

#[test]
fn control_declared_uninterpreted_sort_predicate_still_parses() {
    let mut manager = TermManager::new();
    let script = r#"
        (declare-sort U 0)
        (declare-fun p (U) Bool)
        (declare-const a U)
        (declare-const b U)
        (assert (p a))
        (assert (not (p b)))
        (assert (distinct a b))
    "#;
    assert!(
        parse_script(script, &mut manager).is_ok(),
        "QF_UF script must keep parsing under the strict unknown-symbol rule"
    );
}

#[test]
fn control_declared_symbol_in_reserved_namespace_wins() {
    // Declarations are consulted before the reserved-namespace check, so a
    // user function that happens to be spelled like a theory operator works.
    let mut manager = TermManager::new();
    let script = r#"
        (declare-fun bvmine (Int) Bool)
        (declare-fun str.mine (Int) Bool)
        (assert (bvmine 1))
        (assert (str.mine 2))
    "#;
    assert!(
        parse_script(script, &mut manager).is_ok(),
        "an explicitly declared symbol must resolve even in a theory namespace"
    );
}

#[test]
fn control_define_fun_application_still_expands() {
    let mut manager = TermManager::new();
    let script = r#"
        (define-fun double ((n Int)) Int (* 2 n))
        (declare-const k Int)
        (assert (= (double k) 4))
    "#;
    assert!(parse_script(script, &mut manager).is_ok());
}

#[test]
fn control_datatype_constructor_and_selector_applications_resolve() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-datatypes ((IntList 0)) (((nil) (cons (head Int) (tail IntList)))))
        (declare-const l IntList)
        (assert (= l (cons 1 nil)))
        (assert (= (head l) 10))
        "#,
    );
    let ctor = eq_side_matching(&m, asserts[0], |k| {
        matches!(k, TermKind::DtConstructor { .. })
    });
    assert!(matches!(kind(&m, ctor), TermKind::DtConstructor { .. }));

    let sel = eq_side_matching(&m, asserts[1], |k| matches!(k, TermKind::DtSelector { .. }));
    let sort = m.get(sel).expect("term exists").sort;
    assert!(
        matches!(m.sorts.get(sort).expect("sort exists").kind, SortKind::Int),
        "selector application must carry its declared result sort, not Bool"
    );
}

#[test]
fn control_bare_term_mode_still_allows_free_symbols() {
    // `parse_term` (not `parse_script`) intentionally stays lenient so ad-hoc
    // fragments can mention free symbols.
    let mut manager = TermManager::new();
    assert!(
        oxiz_core::smtlib::parse_term("(some_free_fn a b)", &mut manager).is_ok(),
        "bare-term mode must keep accepting undeclared non-theory symbols"
    );
}

#[test]
fn bare_term_mode_still_rejects_reserved_theory_namespace() {
    let mut manager = TermManager::new();
    assert!(
        oxiz_core::smtlib::parse_term("(str.frobnicate a b c)", &mut manager).is_err(),
        "a reserved theory name is never a legitimate free symbol"
    );
}

// ---------------------------------------------------------------------------
// Finding 6: SMT-LIB Unicode Strings operators and string-literal escapes.
// ---------------------------------------------------------------------------

/// The value of a string-literal term.
fn string_lit(m: &TermManager, t: oxiz_core::ast::TermId) -> String {
    match kind(m, t) {
        TermKind::StringLit(s) => s,
        other => panic!("expected a string literal, got {other:?}"),
    }
}

/// Parse `(assert <term>)` and return the manager plus the asserted term.
fn parse_one_assert(script: &str) -> (TermManager, oxiz_core::ast::TermId) {
    let (m, asserts) = parse_asserts(script);
    let t = *asserts.first().expect("one assertion expected");
    (m, t)
}

#[test]
fn str_lt_ground_folds_in_both_polarities() {
    let (m, t) = parse_one_assert(r#"(assert (str.< "abc" "abd"))"#);
    assert!(matches!(kind(&m, t), TermKind::True), "\"abc\" < \"abd\"");

    let (m, t) = parse_one_assert(r#"(assert (str.< "abd" "abc"))"#);
    assert!(matches!(kind(&m, t), TermKind::False), "\"abd\" !< \"abc\"");

    // A proper prefix is strictly smaller.
    let (m, t) = parse_one_assert(r#"(assert (str.< "ab" "abc"))"#);
    assert!(matches!(kind(&m, t), TermKind::True));
    let (m, t) = parse_one_assert(r#"(assert (str.< "abc" "abc"))"#);
    assert!(matches!(kind(&m, t), TermKind::False));
}

#[test]
fn str_le_ground_folds_in_both_polarities() {
    let (m, t) = parse_one_assert(r#"(assert (str.<= "abc" "abc"))"#);
    assert!(matches!(kind(&m, t), TermKind::True));

    let (m, t) = parse_one_assert(r#"(assert (str.<= "abd" "abc"))"#);
    assert!(matches!(kind(&m, t), TermKind::False));
}

#[test]
fn str_lt_is_chainable() {
    // `(str.< a b c)` abbreviates `(and (str.< a b) (str.< b c))`.
    let (m, t) = parse_one_assert(r#"(assert (str.< "a" "b" "c"))"#);
    assert!(matches!(kind(&m, t), TermKind::True));
    let (m, t) = parse_one_assert(r#"(assert (str.< "a" "c" "b"))"#);
    assert!(matches!(kind(&m, t), TermKind::False));
}

/// A symbolic operand no longer aborts the parse: it builds a real `StrLt` /
/// `StrLe` term the string theory can evaluate.
#[test]
fn str_order_with_symbolic_operand_builds_a_term() {
    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (assert (str.< x "abd"))
        "#,
    );
    assert!(
        matches!(kind(&m, t), TermKind::StrLt(_, _)),
        "symbolic str.< must build a StrLt term"
    );

    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (declare-const y String)
        (assert (str.<= x y))
        "#,
    );
    assert!(
        matches!(kind(&m, t), TermKind::StrLe(_, _)),
        "symbolic str.<= must build a StrLe term"
    );

    // Chaining still expands to a conjunction of binary atoms.
    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (declare-const y String)
        (declare-const z String)
        (assert (str.< x y z))
        "#,
    );
    assert!(
        matches!(kind(&m, t), TermKind::And(_)),
        "a chained str.< must expand to a conjunction"
    );
}

/// The order's structural identities hold without any constant operand:
/// `x < x` is false, `x <= x` is true, nothing is below `""`, and `""` is
/// below everything.  Reference: Z3's `seq_rewriter.cpp` `mk_str_lt`.
#[test]
fn str_order_structural_identities_fold() {
    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (assert (str.< x x))
        "#,
    );
    assert!(matches!(kind(&m, t), TermKind::False), "x < x is false");

    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (assert (str.<= x x))
        "#,
    );
    assert!(matches!(kind(&m, t), TermKind::True), "x <= x is true");

    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (assert (str.< x ""))
        "#,
    );
    assert!(
        matches!(kind(&m, t), TermKind::False),
        "nothing is strictly below the empty string"
    );

    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (assert (str.<= "" x))
        "#,
    );
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "the empty string is below everything"
    );
}

/// `str.to_code` / `str.from_code` with a symbolic operand build terms rather
/// than failing the parse.
#[test]
fn str_code_conversions_with_symbolic_operand_build_terms() {
    let (m, t) = parse_one_assert(
        r#"
        (declare-const x String)
        (assert (= (str.to_code x) 65))
        "#,
    );
    let side = eq_side_matching(&m, t, |k| matches!(k, TermKind::StrToCode(_)));
    assert!(matches!(kind(&m, side), TermKind::StrToCode(_)));

    let (m, t) = parse_one_assert(
        r#"
        (declare-const n Int)
        (assert (= (str.from_code n) "A"))
        "#,
    );
    let side = eq_side_matching(&m, t, |k| matches!(k, TermKind::StrFromCode(_)));
    assert!(matches!(kind(&m, side), TermKind::StrFromCode(_)));
}

/// `str.replace_re` / `str.replace_re_all` parse into dedicated term kinds
/// whose middle operand keeps the `RegLan` regex node.
#[test]
fn str_replace_re_parses_into_terms() {
    let (m, t) =
        parse_one_assert(r#"(assert (= (str.replace_re "abcabc" (str.to_re "b") "X") "aXcabc"))"#);
    let side = eq_side_matching(&m, t, |k| matches!(k, TermKind::StrReplaceRe(_, _, _)));
    assert!(matches!(kind(&m, side), TermKind::StrReplaceRe(_, _, _)));

    let (m, t) = parse_one_assert(
        r#"
        (declare-const s String)
        (assert (= (str.replace_re_all s (re.* (str.to_re "a")) "X") "X"))
        "#,
    );
    let side = eq_side_matching(&m, t, |k| matches!(k, TermKind::StrReplaceReAll(_, _, _)));
    assert!(matches!(kind(&m, side), TermKind::StrReplaceReAll(_, _, _)));
}

/// A surrogate code point is inside the theory's alphabet but not
/// representable in OxiZ's `char`-backed strings, so `str.from_code` declines
/// to fold rather than returning `""` (which has the wrong length).
#[test]
fn str_from_code_leaves_surrogates_unfolded() {
    let (m, t) = parse_one_assert(r#"(assert (= (str.from_code 55296) ""))"#);
    let side = eq_side_matching(&m, t, |k| matches!(k, TermKind::StrFromCode(_)));
    assert!(
        matches!(kind(&m, side), TermKind::StrFromCode(_)),
        "a surrogate code point must not fold to a string literal"
    );

    // The code points on either side of the surrogate block fold normally.
    let (m, t) = parse_one_assert(r#"(assert (= (str.from_code 57344) "\u{e000}"))"#);
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "U+E000 is representable and must fold"
    );
    let (m, t) = parse_one_assert(r#"(assert (= (str.from_code 55295) "\u{d7ff}"))"#);
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "U+D7FF is representable and must fold"
    );
}

#[test]
fn str_to_code_folds_for_constants() {
    // The equality of two identical interned constants folds to `true`, so a
    // `True` term is exactly the evidence that the fold produced the right
    // value; the negative polarity below shows it is not vacuous.
    let (m, t) = parse_one_assert(r#"(assert (= (str.to_code "A") 65))"#);
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "str.to_code \"A\" = 65"
    );
    let (m, t) = parse_one_assert(r#"(assert (= (str.to_code "A") 66))"#);
    assert!(
        !matches!(kind(&m, t), TermKind::True),
        "str.to_code \"A\" must not be 66"
    );

    // Anything that is not a one-character string yields -1.
    let (m, t) = parse_one_assert(r#"(assert (= (str.to_code "") -1))"#);
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "str.to_code \"\" = -1"
    );
    let (m, t) = parse_one_assert(r#"(assert (= (str.to_code "ab") -1))"#);
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "str.to_code \"ab\" = -1"
    );
}

#[test]
fn str_from_code_folds_for_constants() {
    let (m, t) = parse_one_assert(r#"(assert (= (str.from_code 65) "A"))"#);
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "str.from_code 65 must fold to \"A\""
    );
    let (m, t) = parse_one_assert(r#"(assert (= (str.from_code 65) "B"))"#);
    assert!(
        !matches!(kind(&m, t), TermKind::True),
        "str.from_code 65 must not be \"B\""
    );

    // Out of the theory's alphabet => the empty string.
    let (m, t) = parse_one_assert(r#"(assert (= (str.from_code 200000) ""))"#);
    assert!(
        matches!(kind(&m, t), TermKind::True),
        "an out-of-range code point must yield \"\""
    );
}

#[test]
fn str_is_digit_lowers_to_regex_membership() {
    let (m, t) = parse_one_assert(
        r#"
        (declare-const c String)
        (assert (str.is_digit c))
        "#,
    );
    assert!(
        matches!(kind(&m, t), TermKind::StrInRe(_, _)),
        "str.is_digit must lower to a `re.range \"0\" \"9\"` membership"
    );
}

#[test]
fn string_literal_unicode_escapes_are_decoded() {
    // Braced form, unbraced 4-digit form, boundary code points, a non-escape
    // backslash, the doubled-quote escape, and a mixed literal.
    let cases: [(&str, usize); 8] = [
        (r#""\u{e9}""#, 1),
        ("\"\\u0041\"", 1),
        (r#""\u{0}""#, 1),
        (r#""\u{2ffff}""#, 1),
        // One above the alphabet's maximum: not an escape, stands for itself.
        (r#""\u{30000}""#, 9),
        (r#""\q""#, 2),
        ("\"a\"\"b\"", 3),
        ("\"a\\u{62}c\\u0064e\"", 5),
    ];
    for (literal, expected_len) in cases {
        // Wrap in `str.len` so the assertion cannot fold away to `true`
        // before the literal can be inspected.
        let script = format!("(assert (= (str.len {literal}) 999))");
        let (m, asserts) = parse_asserts(&script);
        let len_term = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::StrLen(_)));
        let TermKind::StrLen(inner) = kind(&m, len_term) else {
            panic!("expected StrLen");
        };
        let value = string_lit(&m, inner);
        assert_eq!(
            value.chars().count(),
            expected_len,
            "literal {literal} decoded to {value:?}"
        );
    }
}

#[test]
fn string_literal_escape_changes_str_len() {
    // The defect this guards: `(str.len "\u{e9}")` used to be 6, because the
    // escape was kept as its six literal source characters.
    let (m, t) = parse_one_assert(r#"(assert (= (str.len "\u{e9}") 1))"#);
    let arg = eq_side_matching(&m, t, |k| matches!(k, TermKind::StrLen(_)));
    let TermKind::StrLen(inner) = kind(&m, arg) else {
        panic!("expected StrLen");
    };
    assert_eq!(string_lit(&m, inner).chars().count(), 1);
}

#[test]
fn surrogate_escape_is_rejected() {
    // The SMT-LIB alphabet includes the UTF-16 surrogate range, which OxiZ's
    // string representation cannot hold; rejecting is honest.
    let msg = parse_script_err(r#"(assert (= (str.len "\ud800") 1))"#);
    assert!(msg.contains("surrogate"), "unexpected message: {msg}");
}

/// Round-tripping a string value through the printer and back.
///
/// The defect this guards: both printers used to emit C-style escapes
/// (`s.replace('\\', "\\\\").replace('"', "\\\"")` in `basic.rs` and
/// `pretty.rs`), and `model::Value`'s `Display` emitted none at all. SMT-LIB
/// has no `\"` escape — a quote is written `""` — and no `\\` escape either,
/// so back then:
///
/// * a value containing `"` printed as `\"`, which re-parses as a backslash
///   followed by the end of the literal;
/// * a value containing one `\` printed as `\\`, which re-parses as two;
/// * a non-ASCII code point printed as raw UTF-8 bytes rather than a
///   `\u{...}` escape, so z3 read it back as several characters.
///
/// The printer now emits `"` as `""`, `\` verbatim *unless* the text
/// following it would be read back as a `\u` escape (in which case the
/// backslash is written `\u{5c}`), and every code point outside printable
/// ASCII as `\u{...}` — see `smtlib::printer::format_string_literal`, the one
/// encoder all three sites share.
#[test]
fn string_values_round_trip_through_the_printer() {
    use oxiz_core::smtlib::Printer;

    for literal in [
        r#""\u{e9}""#,
        "\"a\"\"b\"",
        r#""back\slash""#,
        r#""\u{1f600}""#,
    ] {
        let script = format!("(assert (= (str.len {literal}) 999))");
        let (m, asserts) = parse_asserts(&script);
        let len_term = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::StrLen(_)));
        let TermKind::StrLen(inner) = kind(&m, len_term) else {
            panic!("expected StrLen");
        };
        let original = string_lit(&m, inner);

        let printed = Printer::new(&m).print_term(inner);
        let reparsed_script = format!("(assert (= (str.len {printed}) 999))");
        let (m2, asserts2) = parse_asserts(&reparsed_script);
        let len2 = eq_side_matching(&m2, asserts2[0], |k| matches!(k, TermKind::StrLen(_)));
        let TermKind::StrLen(inner2) = kind(&m2, len2) else {
            panic!("expected StrLen");
        };
        assert_eq!(
            string_lit(&m2, inner2),
            original,
            "{literal} printed as {printed} did not round-trip"
        );
    }
}

// ---------------------------------------------------------------------------
// Finding 7: indexed identifiers are theory constructs, so an unrecognised one
// must be an error too — the `str.<` failure mode applies to `(_ f i)` heads
// as well, and there the *name* often looks known.
//
// Guards the standard `((_ extract i j) x)` spelling (which used to fall
// through to a Bool-sorted uninterpreted apply, answering `sat` for
// `(= ((_ extract 3 0) #xab) #xc)` and tripping a `try_mk_bv_concat` sort
// error), its range/sort checks, and the remaining unimplemented indexed
// operators.
// ---------------------------------------------------------------------------

#[test]
fn extract_standard_spelling_is_a_real_bitvector() {
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= ((_ extract 3 0) x) #x0))
        "#,
    );
    let side = eq_side_matching(&m, asserts[0], |k| matches!(k, TermKind::BvExtract { .. }));
    assert_eq!(
        bv_width(&m, side),
        Some(4),
        "((_ extract 3 0) x) must be a 4-bit bit-vector, not a Bool apply"
    );
}

#[test]
fn extract_inside_concat_has_a_known_width() {
    // Regression: an uninterpreted `extract` gave `concat` an operand with no
    // width, which `try_mk_bv_concat` now reports as a sort error.
    let (m, asserts) = parse_asserts(
        r#"
        (declare-const x (_ BitVec 8))
        (assert (= (concat #x0 ((_ extract 3 0) x)) #x00))
        "#,
    );
    let side = concat_side(&m, asserts[0]);
    assert_eq!(bv_width(&m, side), Some(8));
}

#[test]
fn extract_range_and_sort_are_checked() {
    for script in [
        // i < j
        "(declare-const x (_ BitVec 8))(assert (= ((_ extract 0 3) x) #x0))",
        // i >= width
        "(declare-const x (_ BitVec 8))(assert (= ((_ extract 9 0) x) x))",
        // non-bit-vector operand
        "(assert (= ((_ extract 3 0) 5) #x0))",
        // wrong index count
        "(declare-const x (_ BitVec 8))(assert (= ((_ extract 3) x) #x0))",
    ] {
        let msg = parse_script_err(script);
        assert!(
            !msg.is_empty(),
            "malformed extract must be an error: {script}"
        );
    }
}

#[test]
fn unimplemented_indexed_operators_error_rather_than_answer() {
    for script in [
        // `((_ to_fp eb sb) bv)`: the no-rounding-mode bit-pattern form.
        "(declare-const f (_ FloatingPoint 8 24))\
         (assert (= f ((_ to_fp 8 24) #x3f800000)))",
        // `(_ int2bv n)` / `bv2int` are not implemented.
        "(declare-const b (_ BitVec 8))(assert (= ((_ int2bv 8) 5) b))",
        "(declare-const b (_ BitVec 8))(assert (= (bv2int b) 5))",
        // SMT-LIB datatype `match` is not implemented.
        "(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\
         (declare-const l L)\
         (assert (= 1 (match l ((nil 0) ((cons h t) h)))))",
    ] {
        let msg = parse_script_err(script);
        assert!(
            msg.contains("unknown"),
            "unimplemented operator must be reported, got: {msg}"
        );
    }
}
