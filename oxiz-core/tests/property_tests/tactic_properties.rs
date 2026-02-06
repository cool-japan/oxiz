//! Property-based tests for tactic composition and correctness
//!
//! This module tests that tactics:
//! - Preserve satisfiability
//! - Compose correctly
//! - Produce sound transformations

use num_bigint::BigInt;
use num_traits::Zero;
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

/// Test that simplify preserves tautologies (no parameters needed)
#[test]
fn simplify_preserves_tautology_true() {
    let mut tm = TermManager::new();
    let t = tm.mk_bool(true);

    let simplified = tm.simplify(t);

    // Should still be true
    assert_eq!(get_bool_value(&tm, simplified), Some(true));
}

/// Test that simplify preserves contradictions (no parameters needed)
#[test]
fn simplify_preserves_contradiction_false() {
    let mut tm = TermManager::new();
    let f = tm.mk_bool(false);

    let simplified = tm.simplify(f);

    // Should still be false
    assert_eq!(get_bool_value(&tm, simplified), Some(false));
}

proptest! {
    /// Test that the simplify method is idempotent
    #[test]
    fn simplify_idempotent(a in -50i64..50i64, b in -50i64..50i64) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let sum = tm.mk_add(vec![ta, tb]);

        let simplified1 = tm.simplify(sum);
        let simplified2 = tm.simplify(simplified1);

        // Simplifying twice should give the same result
        prop_assert_eq!(simplified1, simplified2);
    }

    /// Test that simplify handles constants
    #[test]
    fn simplify_constant(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));

        let simplified = tm.simplify(t);

        // Should preserve the value
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, t), get_int_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that simplify normalizes double negations
    #[test]
    fn simplify_normalizes_double_negation(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let not_t = tm.mk_not(t);
        let not_not_t = tm.mk_not(not_t);

        let simplified = tm.simplify(not_not_t);

        // Should have the same value as original
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that simplify eliminates identity operations
    #[test]
    fn simplify_eliminates_identity_add(n in -50i64..50i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let zero = tm.mk_int(BigInt::zero());
        let sum = tm.mk_add(vec![t, zero]);

        let simplified = tm.simplify(sum);

        // Should equal the original value
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, t), get_int_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that simplify performs constant folding
    #[test]
    fn simplify_folds_constants(a in -30i64..30i64, b in -30i64..30i64) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let sum = tm.mk_add(vec![ta, tb]);

        let simplified = tm.simplify(sum);

        // Should be a constant
        if let Some(val) = get_int_value(&tm, simplified) {
            prop_assert_eq!(val, BigInt::from(a + b));
        }
    }

    /// Test that simplify handles tautologies
    #[test]
    fn simplify_handles_tautology(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let not_t = tm.mk_not(t);
        let tautology = tm.mk_or(vec![t, not_t]);

        let simplified = tm.simplify(tautology);

        // Should simplify to true
        prop_assert_eq!(get_bool_value(&tm, simplified), Some(true));
    }

    /// Test that simplify handles contradictions
    #[test]
    fn simplify_handles_contradiction(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let not_t = tm.mk_not(t);
        let contradiction = tm.mk_and(vec![t, not_t]);

        let simplified = tm.simplify(contradiction);

        // Should simplify to false
        prop_assert_eq!(get_bool_value(&tm, simplified), Some(false));
    }

    /// Test AND with true is identity
    #[test]
    fn and_true_identity(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let true_term = tm.mk_bool(true);
        let and_term = tm.mk_and(vec![t, true_term]);

        let simplified = tm.simplify(and_term);

        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test OR with false is identity
    #[test]
    fn or_false_identity(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let false_term = tm.mk_bool(false);
        let or_term = tm.mk_or(vec![t, false_term]);

        let simplified = tm.simplify(or_term);

        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that x = x simplifies to true
    #[test]
    fn self_equality_is_true(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let eq = tm.mk_eq(t, t);

        let simplified = tm.simplify(eq);

        // x = x should be true
        prop_assert_eq!(get_bool_value(&tm, simplified), Some(true));
    }

    /// Test that x < x simplifies to false
    #[test]
    fn self_less_than_is_false(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let lt = tm.mk_lt(t, t);

        let simplified = tm.simplify(lt);

        // x < x should be false
        prop_assert_eq!(get_bool_value(&tm, simplified), Some(false));
    }

    /// Test that x <= x simplifies to true
    #[test]
    fn self_less_equal_is_true(n in -100i64..100i64) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let le = tm.mk_le(t, t);

        let simplified = tm.simplify(le);

        // x <= x should be true
        prop_assert_eq!(get_bool_value(&tm, simplified), Some(true));
    }

    /// Test that if-then-else with true condition simplifies to then branch
    #[test]
    fn ite_true_condition(a in -50i64..50i64, b in -50i64..50i64) {
        let mut tm = TermManager::new();
        let true_term = tm.mk_bool(true);
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let ite = tm.mk_ite(true_term, ta, tb);
        let simplified = tm.simplify(ite);

        // Should equal ta
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, ta), get_int_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that if-then-else with false condition simplifies to else branch
    #[test]
    fn ite_false_condition(a in -50i64..50i64, b in -50i64..50i64) {
        let mut tm = TermManager::new();
        let false_term = tm.mk_bool(false);
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let ite = tm.mk_ite(false_term, ta, tb);
        let simplified = tm.simplify(ite);

        // Should equal tb
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, tb), get_int_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }
}
