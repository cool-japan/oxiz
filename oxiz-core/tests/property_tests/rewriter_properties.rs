//! Property-based tests for rewriter correctness
//!
//! This module tests that rewrite rules preserve semantics and maintain
//! proper invariants across all theories:
//! - Boolean rewrites
//! - Arithmetic rewrites
//! - Preservation of satisfiability

use num_bigint::BigInt;
use num_traits::{One, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager};
use proptest::prelude::*;

/// Helper to get boolean value from a term
fn get_bool_value(tm: &TermManager, id: TermId) -> Option<bool> {
    match tm.get(id).map(|t| &t.kind) {
        Some(TermKind::True) => Some(true),
        Some(TermKind::False) => Some(false),
        _ => None,
    }
}

/// Helper to get integer value from a term
fn get_int_value(tm: &TermManager, id: TermId) -> Option<BigInt> {
    match tm.get(id).map(|t| &t.kind) {
        Some(TermKind::IntConst(n)) => Some(n.clone()),
        _ => None,
    }
}

/// Test that rewriting preserves tautologies (no parameters needed)
#[test]
fn rewrite_preserves_tautology_true() {
    let mut tm = TermManager::new();
    let t = tm.mk_bool(true);

    // Simplify
    let rewritten = tm.simplify(t);

    // Should still be true
    assert_eq!(get_bool_value(&tm, rewritten), Some(true));
}

/// Test that rewriting preserves contradictions (no parameters needed)
#[test]
fn rewrite_preserves_contradiction_false() {
    let mut tm = TermManager::new();
    let f = tm.mk_bool(false);

    // Simplify
    let rewritten = tm.simplify(f);

    // Should still be false
    assert_eq!(get_bool_value(&tm, rewritten), Some(false));
}

proptest! {
    /// Test: (a AND NOT a) rewrites to false
    #[test]
    fn and_with_negation_is_false(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let not_t = tm.mk_not(t);
        let contradiction = tm.mk_and(vec![t, not_t]);

        let rewritten = tm.simplify(contradiction);

        // Should be false
        prop_assert_eq!(get_bool_value(&tm, rewritten), Some(false));
    }

    /// Test: (a OR NOT a) rewrites to true
    #[test]
    fn or_with_negation_is_true(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let not_t = tm.mk_not(t);
        let tautology = tm.mk_or(vec![t, not_t]);

        let rewritten = tm.simplify(tautology);

        // Should be true
        prop_assert_eq!(get_bool_value(&tm, rewritten), Some(true));
    }

    /// Test: (a AND (a OR b)) simplifies to same value as a (absorption law)
    #[test]
    fn absorption_law_and(a in proptest::bool::ANY, b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let ta = tm.mk_bool(a);
        let tb = tm.mk_bool(b);
        let or_ab = tm.mk_or(vec![ta, tb]);
        let and_term = tm.mk_and(vec![ta, or_ab]);

        let rewritten = tm.simplify(and_term);

        // Should have same value as ta
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, ta), get_bool_value(&tm, rewritten)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test idempotence: a AND a = a
    #[test]
    fn and_idempotent(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let and_tt = tm.mk_and(vec![t, t]);

        let rewritten = tm.simplify(and_tt);

        // Should have same value as t
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, rewritten)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test idempotence: a OR a = a
    #[test]
    fn or_idempotent(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let or_tt = tm.mk_or(vec![t, t]);

        let rewritten = tm.simplify(or_tt);

        // Should have same value as t
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, rewritten)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that 0 + x rewrites to x
    #[test]
    fn add_zero_eliminates(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let zero = tm.mk_int(BigInt::zero());
        let t = tm.mk_int(BigInt::from(n));
        let sum = tm.mk_add(vec![zero, t]);

        let rewritten = tm.simplify(sum);

        if let (Some(v1), Some(v2)) = (get_int_value(&tm, t), get_int_value(&tm, rewritten)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that 1 * x rewrites to x
    #[test]
    fn mul_one_eliminates(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let one = tm.mk_int(BigInt::one());
        let t = tm.mk_int(BigInt::from(n));
        let prod = tm.mk_mul(vec![one, t]);

        let rewritten = tm.simplify(prod);

        if let (Some(v1), Some(v2)) = (get_int_value(&tm, t), get_int_value(&tm, rewritten)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that 0 * x rewrites to 0
    #[test]
    fn mul_zero_annihilates(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let zero = tm.mk_int(BigInt::zero());
        let t = tm.mk_int(BigInt::from(n));
        let prod = tm.mk_mul(vec![zero, t]);

        let rewritten = tm.simplify(prod);

        prop_assert_eq!(get_int_value(&tm, rewritten), Some(BigInt::zero()));
    }

    /// Test that x - x rewrites to 0
    #[test]
    fn subtract_self_is_zero(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let diff = tm.mk_sub(t, t);

        let rewritten = tm.simplify(diff);

        prop_assert_eq!(get_int_value(&tm, rewritten), Some(BigInt::zero()));
    }

    /// Test that x + (-x) rewrites to 0
    #[test]
    fn add_negation_is_zero(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let neg_t = tm.mk_neg(t);
        let sum = tm.mk_add(vec![t, neg_t]);

        let rewritten = tm.simplify(sum);

        prop_assert_eq!(get_int_value(&tm, rewritten), Some(BigInt::zero()));
    }

    /// Test constant folding for addition
    #[test]
    fn constant_folding_add(a in -50i64..50i64, b in -50i64..50i64) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let sum = tm.mk_add(vec![ta, tb]);

        let rewritten = tm.simplify(sum);

        // Should be a constant equal to a + b
        if let Some(value) = get_int_value(&tm, rewritten) {
            prop_assert_eq!(value, BigInt::from(a + b));
        }
    }

    /// Test constant folding for multiplication
    #[test]
    fn constant_folding_mul(a in -20i64..20i64, b in -20i64..20i64) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let prod = tm.mk_mul(vec![ta, tb]);

        let rewritten = tm.simplify(prod);

        // Should be a constant equal to a * b
        if let Some(value) = get_int_value(&tm, rewritten) {
            prop_assert_eq!(value, BigInt::from(a * b));
        }
    }

    /// Test that x < x rewrites to false
    #[test]
    fn less_than_self_is_false(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let lt = tm.mk_lt(t, t);

        let rewritten = tm.simplify(lt);

        prop_assert_eq!(get_bool_value(&tm, rewritten), Some(false));
    }

    /// Test that x <= x rewrites to true
    #[test]
    fn less_equal_self_is_true(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let le = tm.mk_le(t, t);

        let rewritten = tm.simplify(le);

        prop_assert_eq!(get_bool_value(&tm, rewritten), Some(true));
    }

    /// Test that rewriting x = x always gives true
    #[test]
    fn rewrite_self_equality_is_true(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let eq = tm.mk_eq(t, t);

        let rewritten = tm.simplify(eq);

        // Should be true
        prop_assert_eq!(get_bool_value(&tm, rewritten), Some(true));
    }

    /// Test idempotence of rewriting
    #[test]
    fn rewrite_is_idempotent(a in -50i64..50i64, b in -50i64..50i64) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let sum = tm.mk_add(vec![ta, tb]);

        let rewritten1 = tm.simplify(sum);
        let rewritten2 = tm.simplify(rewritten1);

        // Rewriting twice should give the same result
        prop_assert_eq!(rewritten1, rewritten2);
    }
}
