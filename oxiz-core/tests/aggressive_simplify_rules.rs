//! Integration tests for the new AggressiveSimplifier rewrite rules.
//!
//! Several rule families (Not(Not(a))→a, Implies(true/false,b)→…, Xor identities,
//! Eq(x,x)→true, ITE identities, arithmetic constant folding) are already applied
//! at construction time by the `mk_*` builder methods.  The tests below focus on
//! rules that are *genuinely new* in AggressiveSimplifier:
//!
//!   1. De Morgan push-down: Not(And(a,b)) → Or(Not(a), Not(b))
//!   2. Implies(a, false) → Not(a)
//!   3. Implies(a, a)     → true
//!   4. BvNot(BvNot(x))  → x
//!   5. BvAnd identity rules (0, all_ones, same)
//!   6. BvOr  identity rules (0, all_ones, same)
//!   7. BvXor identity rules (0, same)
//!   8. Boolean-heavy integration goal
//!   9. BV-heavy integration goal

use oxiz_core::ast::TermManager;
use oxiz_core::simplification::{AggressiveSimplifier, SimplificationConfig};

fn aggressive_config() -> SimplificationConfig {
    SimplificationConfig { aggressive: true }
}

// ---------------------------------------------------------------------------
// Rule family 1 — De Morgan push-down: Not(And(a,b)) → Or(Not(a), Not(b))
// ---------------------------------------------------------------------------

#[test]
fn test_de_morgan_not_and_pushdown() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let a = manager.mk_var("a", bool_sort);
    let b = manager.mk_var("b", bool_sort);

    // And(a, b)
    let and_ab = manager.mk_and([a, b]);
    // Not(And(a,b)) — mk_not only collapses Not(Not(_)), Not(true), Not(false);
    // Not(And(…)) survives construction.
    let not_and = manager.mk_not(and_ab);

    // Pre-build expected result so we can compare after simplifier is dropped.
    let not_a = manager.mk_not(a);
    let not_b = manager.mk_not(b);
    let expected = manager.mk_or([not_a, not_b]);

    let result = {
        let mut simplifier = AggressiveSimplifier::new(&mut manager, aggressive_config());
        simplifier.simplify_term(not_and)
    };

    assert_eq!(
        result, expected,
        "Not(And(a,b)) should reduce to Or(Not(a),Not(b))"
    );
}

// ---------------------------------------------------------------------------
// Rule family 2 — Implies(a, false) → Not(a)
// ---------------------------------------------------------------------------

#[test]
fn test_implies_a_false_reduces_to_not_a() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let a = manager.mk_var("a", bool_sort);
    let f = manager.mk_false();

    // mk_implies does NOT simplify Implies(a, false), so this survives construction.
    let implies_af = manager.mk_implies(a, f);
    let expected_not_a = manager.mk_not(a);

    let result = {
        let mut simplifier = AggressiveSimplifier::new(&mut manager, aggressive_config());
        simplifier.simplify_term(implies_af)
    };

    assert_eq!(
        result, expected_not_a,
        "Implies(a, false) should reduce to Not(a)"
    );
}

// ---------------------------------------------------------------------------
// Rule family 3 — Implies(a, a) → true
// ---------------------------------------------------------------------------

#[test]
fn test_implies_reflexive_is_true() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let a = manager.mk_var("a", bool_sort);

    // mk_implies does NOT simplify Implies(a, a), so this survives construction.
    let implies_aa = manager.mk_implies(a, a);
    let true_id = manager.mk_true();

    let result = {
        let mut simplifier = AggressiveSimplifier::new(&mut manager, aggressive_config());
        simplifier.simplify_term(implies_aa)
    };

    assert_eq!(result, true_id, "Implies(a, a) should reduce to true");
}

// ---------------------------------------------------------------------------
// Rule family 4 — BvNot(BvNot(x)) → x
// ---------------------------------------------------------------------------

#[test]
fn test_bv_double_not_eliminates() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let x = manager.mk_var("x", bv8);

    let not_x = manager.mk_bv_not(x);
    let not_not_x = manager.mk_bv_not(not_x);

    let result = {
        let mut simplifier = AggressiveSimplifier::new(&mut manager, aggressive_config());
        simplifier.simplify_term(not_not_x)
    };

    assert_eq!(result, x, "BvNot(BvNot(x)) should reduce to x");
}

// ---------------------------------------------------------------------------
// Rule family 5 — BvAnd identity rules
// ---------------------------------------------------------------------------

#[test]
fn test_bv_and_identities() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let x = manager.mk_var("x", bv8);

    let zero8 = manager.mk_bitvec(0i64, 8);
    let all_ones8 = manager.mk_bitvec(0xFF_i64, 8);

    // BvAnd(x, 0)
    let and_x_zero = manager.mk_bv_and(x, zero8);
    // BvAnd(0, x)
    let and_zero_x = manager.mk_bv_and(zero8, x);
    // BvAnd(x, all_ones)
    let and_x_all = manager.mk_bv_and(x, all_ones8);
    // BvAnd(all_ones, x)
    let and_all_x = manager.mk_bv_and(all_ones8, x);
    // BvAnd(x, x)
    let and_x_x = manager.mk_bv_and(x, x);

    let (r1, r2, r3, r4, r5) = {
        let mut s = AggressiveSimplifier::new(&mut manager, aggressive_config());
        (
            s.simplify_term(and_x_zero),
            s.simplify_term(and_zero_x),
            s.simplify_term(and_x_all),
            s.simplify_term(and_all_x),
            s.simplify_term(and_x_x),
        )
    };

    assert_eq!(r1, zero8, "BvAnd(x, 0) should reduce to 0");
    assert_eq!(r2, zero8, "BvAnd(0, x) should reduce to 0");
    assert_eq!(r3, x, "BvAnd(x, all_ones) should reduce to x");
    assert_eq!(r4, x, "BvAnd(all_ones, x) should reduce to x");
    assert_eq!(r5, x, "BvAnd(x, x) should reduce to x");
}

// ---------------------------------------------------------------------------
// Rule family 6 — BvOr identity rules
// ---------------------------------------------------------------------------

#[test]
fn test_bv_or_identities() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let x = manager.mk_var("x", bv8);

    let zero8 = manager.mk_bitvec(0i64, 8);
    let all_ones8 = manager.mk_bitvec(0xFF_i64, 8);

    let or_x_zero = manager.mk_bv_or(x, zero8);
    let or_zero_x = manager.mk_bv_or(zero8, x);
    let or_x_all = manager.mk_bv_or(x, all_ones8);
    let or_all_x = manager.mk_bv_or(all_ones8, x);
    let or_x_x = manager.mk_bv_or(x, x);

    let (r1, r2, r3, r4, r5) = {
        let mut s = AggressiveSimplifier::new(&mut manager, aggressive_config());
        (
            s.simplify_term(or_x_zero),
            s.simplify_term(or_zero_x),
            s.simplify_term(or_x_all),
            s.simplify_term(or_all_x),
            s.simplify_term(or_x_x),
        )
    };

    assert_eq!(r1, x, "BvOr(x, 0) should reduce to x");
    assert_eq!(r2, x, "BvOr(0, x) should reduce to x");
    assert_eq!(r3, all_ones8, "BvOr(x, all_ones) should reduce to all_ones");
    assert_eq!(r4, all_ones8, "BvOr(all_ones, x) should reduce to all_ones");
    assert_eq!(r5, x, "BvOr(x, x) should reduce to x");
}

// ---------------------------------------------------------------------------
// Rule family 7 — BvXor identity rules
// ---------------------------------------------------------------------------

#[test]
fn test_bv_xor_identities() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let x = manager.mk_var("x", bv8);

    let zero8 = manager.mk_bitvec(0i64, 8);

    let xor_x_x = manager.mk_bv_xor(x, x);
    let xor_zero_x = manager.mk_bv_xor(zero8, x);
    let xor_x_zero = manager.mk_bv_xor(x, zero8);

    let (r1, r2, r3) = {
        let mut s = AggressiveSimplifier::new(&mut manager, aggressive_config());
        (
            s.simplify_term(xor_x_x),
            s.simplify_term(xor_zero_x),
            s.simplify_term(xor_x_zero),
        )
    };

    assert_eq!(r1, zero8, "BvXor(x, x) should reduce to 0");
    assert_eq!(r2, x, "BvXor(0, x) should reduce to x");
    assert_eq!(r3, x, "BvXor(x, 0) should reduce to x");
}

// ---------------------------------------------------------------------------
// Integration test 8 — Boolean-heavy goal
// ---------------------------------------------------------------------------

/// A chain of Boolean rewrites that exercises multiple rule families in one goal.
///
///   Assertion 1: Implies(And(a, b), false)
///     → Not(And(a, b))           [Implies(_, false) → Not(_)]
///     → Or(Not(a), Not(b))       [De Morgan]
///
///   Assertion 2: Implies(a, a)   → true   [reflexivity]
///
/// After simplification, assertion 2 vanishes (is true) and only assertion 1 remains.
#[test]
fn test_integration_boolean_heavy_goal() {
    let mut manager = TermManager::new();
    let bool_sort = manager.sorts.bool_sort;
    let a = manager.mk_var("a", bool_sort);
    let b = manager.mk_var("b", bool_sort);

    let and_ab = manager.mk_and([a, b]);
    let f = manager.mk_false();

    // Implies(And(a,b), false) — mk_implies does NOT simplify this.
    let impl_and_false = manager.mk_implies(and_ab, f);
    // Implies(a, a)
    let impl_aa = manager.mk_implies(a, a);

    // Pre-build expected Or(Not(a), Not(b)).
    let not_a = manager.mk_not(a);
    let not_b = manager.mk_not(b);
    let expected_or = manager.mk_or([not_a, not_b]);
    let true_id = manager.mk_true();

    let (r1, r2) = {
        let mut simplifier = AggressiveSimplifier::new(&mut manager, aggressive_config());
        (
            simplifier.simplify_term(impl_and_false),
            simplifier.simplify_term(impl_aa),
        )
    };

    assert_eq!(r2, true_id, "Implies(a,a) should be true");
    assert_eq!(
        r1, expected_or,
        "Implies(And(a,b),false) should reduce to Or(Not(a),Not(b))"
    );
}

// ---------------------------------------------------------------------------
// Integration test 9 — BV-heavy goal
// ---------------------------------------------------------------------------

/// Exercises BV identity rules across compound expressions.
///
///   BvAnd(BvNot(BvNot(x)), 0xFF) → x
///     [BvNot(BvNot(x))→x, then BvAnd(x, all_ones)→x]
///
///   BvXor(BvOr(x, 0), BvAnd(x, x)) → BvXor(x, x) → 0
///     [BvOr(x,0)→x, BvAnd(x,x)→x, BvXor(x,x)→0]
#[test]
fn test_integration_bv_heavy_goal() {
    let mut manager = TermManager::new();
    let bv8 = manager.sorts.bitvec(8);
    let x = manager.mk_var("x", bv8);

    let zero8 = manager.mk_bitvec(0i64, 8);
    let all_ones8 = manager.mk_bitvec(0xFF_i64, 8);

    // BvAnd(BvNot(BvNot(x)), 0xFF)
    let not_x = manager.mk_bv_not(x);
    let not_not_x = manager.mk_bv_not(not_x);
    let and_dbl_not_all = manager.mk_bv_and(not_not_x, all_ones8);

    // BvXor(BvOr(x, 0), BvAnd(x, x))
    let or_x_zero = manager.mk_bv_or(x, zero8);
    let and_x_x = manager.mk_bv_and(x, x);
    let xor_expr = manager.mk_bv_xor(or_x_zero, and_x_x);

    let (r1, r2) = {
        let mut simplifier = AggressiveSimplifier::new(&mut manager, aggressive_config());
        (
            simplifier.simplify_term(and_dbl_not_all),
            simplifier.simplify_term(xor_expr),
        )
    };

    assert_eq!(r1, x, "BvAnd(BvNot(BvNot(x)), all_ones) should reduce to x");
    assert_eq!(r2, zero8, "BvXor(BvOr(x,0), BvAnd(x,x)) should reduce to 0");
}
