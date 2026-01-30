//! Property-based tests for Quantifier Elimination (QE) soundness
//!
//! This module tests basic QE-related term construction:
//! - Existential and universal quantifier creation
//! - Term simplification
//! - Variable handling

use num_bigint::BigInt;
use num_traits::Zero;
use oxiz_core::ast::{TermManager, traversal};
use proptest::prelude::*;

/// Strategy for generating small integers for QE
fn qe_int_strategy() -> impl Strategy<Value = i64> {
    -20i64..20i64
}

/// Test forall with disjunction (trichotomy) - no parameters needed
#[test]
fn forall_trichotomy() {
    let mut tm = TermManager::new();
    let int_sort = tm.sorts.int_sort;
    let x = tm.mk_var("x", int_sort);
    let zero = tm.mk_int(BigInt::zero());

    // forall x. (x >= 0) OR (x < 0)
    let ge = tm.mk_ge(x, zero);
    let lt = tm.mk_lt(x, zero);
    let or_term = tm.mk_or(vec![ge, lt]);
    let forall = tm.mk_forall([("x", int_sort)], or_term);

    // x should not be free
    let free_vars = traversal::collect_free_vars(forall, &tm);
    assert!(!free_vars.contains(&x));
}

proptest! {
    /// Test that quantifier-free formulas can be created
    #[test]
    fn create_quantifier_free_formula(n in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let x = tm.mk_var("x", tm.sorts.int_sort);
        let c = tm.mk_int(BigInt::from(n));

        // Quantifier-free formula: x = c
        let formula = tm.mk_eq(x, c);

        // Should create successfully
        let free_vars = traversal::collect_free_vars(formula, &tm);
        prop_assert!(free_vars.contains(&x));
    }

    /// Test that existential quantifiers can be created
    #[test]
    fn create_exists_quantifier(n in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let c = tm.mk_int(BigInt::from(n));

        // exists x. x = c
        let body = tm.mk_eq(x, c);
        let exists = tm.mk_exists([("x", int_sort)], body);

        // Quantified variable should not be free in the result
        let free_vars = traversal::collect_free_vars(exists, &tm);
        prop_assert!(!free_vars.contains(&x));
    }

    /// Test that universal quantifiers can be created
    #[test]
    fn create_forall_quantifier(n in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let c = tm.mk_int(BigInt::from(n));

        // forall x. x >= c
        let body = tm.mk_ge(x, c);
        let forall = tm.mk_forall([("x", int_sort)], body);

        // Quantified variable should not be free in the result
        let free_vars = traversal::collect_free_vars(forall, &tm);
        prop_assert!(!free_vars.contains(&x));
    }

    /// Test that nested quantifiers work correctly
    #[test]
    fn nested_quantifiers(a in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let c = tm.mk_int(BigInt::from(a));

        // exists x. exists y. x + y = c
        let sum = tm.mk_add(vec![x, y]);
        let eq = tm.mk_eq(sum, c);
        let exists_y = tm.mk_exists([("y", int_sort)], eq);
        let exists_x = tm.mk_exists([("x", int_sort)], exists_y);

        // Neither x nor y should be free
        let free_vars = traversal::collect_free_vars(exists_x, &tm);
        prop_assert!(!free_vars.contains(&x));
        prop_assert!(!free_vars.contains(&y));
    }

    /// Test that free variables are preserved outside quantifiers
    #[test]
    fn free_vars_preserved(n in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let y = tm.mk_var("y", int_sort);
        let c = tm.mk_int(BigInt::from(n));

        // exists x. (x + y = c)
        let sum = tm.mk_add(vec![x, y]);
        let eq = tm.mk_eq(sum, c);
        let exists = tm.mk_exists([("x", int_sort)], eq);

        // y should still be a free variable
        let free_vars = traversal::collect_free_vars(exists, &tm);
        prop_assert!(free_vars.contains(&y));
        prop_assert!(!free_vars.contains(&x));
    }

    /// Test creating bounded formulas
    #[test]
    fn create_bounded_formula(a in qe_int_strategy(), b in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let x = tm.mk_var("x", tm.sorts.int_sort);
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        // a <= x <= b
        let le_a = tm.mk_le(ta, x);
        let le_b = tm.mk_le(x, tb);
        let bounded = tm.mk_and(vec![le_a, le_b]);

        // Should create successfully
        let free_vars = traversal::collect_free_vars(bounded, &tm);
        prop_assert!(free_vars.contains(&x));
    }

    /// Test existentially quantified bounds
    #[test]
    fn exists_bounded(a in qe_int_strategy(), b in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        // exists x. (a <= x AND x <= b)
        let le_a = tm.mk_le(ta, x);
        let le_b = tm.mk_le(x, tb);
        let body = tm.mk_and(vec![le_a, le_b]);
        let exists = tm.mk_exists([("x", int_sort)], body);

        // x should not be free in the result
        let free_vars = traversal::collect_free_vars(exists, &tm);
        prop_assert!(!free_vars.contains(&x));
    }

    /// Test creating implications
    #[test]
    fn create_implication(a in qe_int_strategy(), b in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let x = tm.mk_var("x", tm.sorts.int_sort);
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        // x >= a => x >= b
        let ge_a = tm.mk_ge(x, ta);
        let ge_b = tm.mk_ge(x, tb);
        let implies = tm.mk_implies(ge_a, ge_b);

        // Should create successfully
        let free_vars = traversal::collect_free_vars(implies, &tm);
        prop_assert!(free_vars.contains(&x));
    }

    /// Test creating universally quantified implication
    #[test]
    fn forall_implication(a in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let zero = tm.mk_int(BigInt::zero());
        let ta = tm.mk_int(BigInt::from(a));

        // forall x. (x >= 0) => (x >= a)
        let ge_zero = tm.mk_ge(x, zero);
        let ge_a = tm.mk_ge(x, ta);
        let implies = tm.mk_implies(ge_zero, ge_a);
        let forall = tm.mk_forall([("x", int_sort)], implies);

        // x should not be free
        let free_vars = traversal::collect_free_vars(forall, &tm);
        prop_assert!(!free_vars.contains(&x));
    }

    /// Test existential with disjunction
    #[test]
    fn exists_disjunction(a in qe_int_strategy(), b in qe_int_strategy()) {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let x = tm.mk_var("x", int_sort);
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        // exists x. (x = a) OR (x = b)
        let eq_a = tm.mk_eq(x, ta);
        let eq_b = tm.mk_eq(x, tb);
        let or_term = tm.mk_or(vec![eq_a, eq_b]);
        let exists = tm.mk_exists([("x", int_sort)], or_term);

        // x should not be free
        let free_vars = traversal::collect_free_vars(exists, &tm);
        prop_assert!(!free_vars.contains(&x));
    }
}
