//! Property-based tests for AST operations
//!
//! This module tests fundamental properties of the AST such as:
//! - Term construction and uniqueness
//! - Substitution correctness
//! - Traversal consistency

use num_bigint::BigInt;
use num_traits::{One, Zero};
use oxiz_core::ast::{TermId, TermKind, TermManager, traversal};
use proptest::prelude::*;
use rustc_hash::FxHashMap;

/// Strategy for generating small integers
fn small_int_strategy() -> impl Strategy<Value = i64> {
    -100i64..100i64
}

/// Strategy for generating variable names
fn var_name_strategy() -> impl Strategy<Value = String> {
    "[a-z][0-9]?".prop_map(|s| s.to_string())
}

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

proptest! {
    // =====================================
    // Term Construction Properties
    // =====================================

    /// Test that creating the same integer constant twice yields the same TermId
    #[test]
    fn integer_constant_uniqueness(n in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t1 = tm.mk_int(BigInt::from(n));
        let t2 = tm.mk_int(BigInt::from(n));
        prop_assert_eq!(t1, t2);
    }

    /// Test that creating the same boolean constant yields the same TermId
    #[test]
    fn boolean_constant_uniqueness(b in proptest::bool::ANY) {
        let tm = TermManager::new();
        let t1 = tm.mk_bool(b);
        let t2 = tm.mk_bool(b);
        prop_assert_eq!(t1, t2);
    }

    /// Test that variables with the same name have the same TermId
    #[test]
    fn variable_uniqueness(name in var_name_strategy()) {
        let mut tm = TermManager::new();
        let sort = tm.sorts.int_sort;
        let v1 = tm.mk_var(&name, sort);
        let v2 = tm.mk_var(&name, sort);
        prop_assert_eq!(v1, v2);
    }

    /// Test that double negation of booleans is handled correctly
    #[test]
    fn double_negation_property(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let not_t = tm.mk_not(t);
        let not_not_t = tm.mk_not(not_t);

        // Double negation should simplify to original
        let t_val = get_bool_value(&tm, t);
        let not_not_val = get_bool_value(&tm, not_not_t);

        if let (Some(v1), Some(v2)) = (t_val, not_not_val) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that AND with true is identity for constants
    #[test]
    fn and_true_identity(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let true_term = tm.mk_bool(true);
        let result = tm.mk_and(vec![t, true_term]);

        // t AND true = t
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, result)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that AND with false is false
    #[test]
    fn and_false_annihilator(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let false_term = tm.mk_bool(false);
        let result = tm.mk_and(vec![t, false_term]);

        // t AND false = false
        prop_assert_eq!(get_bool_value(&tm, result), Some(false));
    }

    /// Test that OR with false is identity for constants
    #[test]
    fn or_false_identity(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let false_term = tm.mk_bool(false);
        let result = tm.mk_or(vec![t, false_term]);

        // t OR false = t
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, result)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that OR with true is true
    #[test]
    fn or_true_annihilator(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let true_term = tm.mk_bool(true);
        let result = tm.mk_or(vec![t, true_term]);

        // t OR true = true
        prop_assert_eq!(get_bool_value(&tm, result), Some(true));
    }

    /// Test that addition with zero is identity for constants
    #[test]
    fn add_zero_identity(n in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let zero = tm.mk_int(BigInt::zero());
        let result = tm.mk_add(vec![t, zero]);

        // t + 0 = t
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, t), get_int_value(&tm, result)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that multiplication with one is identity
    #[test]
    fn mul_one_identity(n in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let one = tm.mk_int(BigInt::one());
        let result = tm.mk_mul(vec![t, one]);

        // t * 1 = t
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, t), get_int_value(&tm, result)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that multiplication with zero is zero
    #[test]
    fn mul_zero_annihilator(n in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let zero = tm.mk_int(BigInt::zero());
        let result = tm.mk_mul(vec![t, zero]);

        // t * 0 = 0 (after simplification)
        let simplified = tm.simplify(result);
        if let Some(v) = get_int_value(&tm, simplified) {
            prop_assert_eq!(v, BigInt::zero());
        }
    }

    /// Test commutativity of addition for constants
    #[test]
    fn add_commutative(a in small_int_strategy(), b in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t1 = tm.mk_int(BigInt::from(a));
        let t2 = tm.mk_int(BigInt::from(b));

        let sum1 = tm.mk_add(vec![t1, t2]);
        let sum2 = tm.mk_add(vec![t2, t1]);

        // a + b = b + a
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, sum1), get_int_value(&tm, sum2)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test commutativity of multiplication for constants
    #[test]
    fn mul_commutative(a in small_int_strategy(), b in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t1 = tm.mk_int(BigInt::from(a));
        let t2 = tm.mk_int(BigInt::from(b));

        let prod1 = tm.mk_mul(vec![t1, t2]);
        let prod2 = tm.mk_mul(vec![t2, t1]);

        // a * b = b * a
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, prod1), get_int_value(&tm, prod2)) {
            prop_assert_eq!(v1, v2);
        }
    }

    // =====================================
    // Substitution Properties
    // =====================================

    /// Test that substituting a variable with itself yields the original term
    #[test]
    fn identity_substitution(name in var_name_strategy()) {
        let mut tm = TermManager::new();
        let sort = tm.sorts.int_sort;
        let var = tm.mk_var(&name, sort);

        let mut subst = FxHashMap::default();
        subst.insert(var, var);

        let result = tm.substitute(var, &subst);
        prop_assert_eq!(var, result);
    }

    /// Test that substitution is idempotent when substituting constants
    #[test]
    fn constant_substitution_idempotent(
        name in var_name_strategy(),
        value in small_int_strategy()
    ) {
        let mut tm = TermManager::new();
        let sort = tm.sorts.int_sort;
        let var = tm.mk_var(&name, sort);
        let const_term = tm.mk_int(BigInt::from(value));

        let mut subst = FxHashMap::default();
        subst.insert(var, const_term);

        let result1 = tm.substitute(var, &subst);
        let result2 = tm.substitute(result1, &subst);

        prop_assert_eq!(result1, result2);
    }

    /// Test that substitution commutes with addition
    #[test]
    fn substitution_commutes_with_add(
        name in var_name_strategy(),
        value in small_int_strategy(),
        n in small_int_strategy()
    ) {
        let mut tm = TermManager::new();
        let sort = tm.sorts.int_sort;
        let var = tm.mk_var(&name, sort);
        let const_n = tm.mk_int(BigInt::from(n));
        let const_value = tm.mk_int(BigInt::from(value));

        // Substitute then add
        let mut subst = FxHashMap::default();
        subst.insert(var, const_value);
        let subst_var = tm.substitute(var, &subst);
        let result1 = tm.mk_add(vec![subst_var, const_n]);

        // Add then substitute
        let sum = tm.mk_add(vec![var, const_n]);
        let result2 = tm.substitute(sum, &subst);

        // Should be equal: subst(x) + n = subst(x + n)
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, result1), get_int_value(&tm, result2)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that substitution distributes over conjunction
    #[test]
    fn substitution_distributes_over_and(
        name in var_name_strategy(),
        b1 in proptest::bool::ANY,
        b2 in proptest::bool::ANY
    ) {
        let mut tm = TermManager::new();
        let sort = tm.sorts.bool_sort;
        let var = tm.mk_var(&name, sort);
        let const_b1 = tm.mk_bool(b1);
        let const_b2 = tm.mk_bool(b2);

        // Create: var AND b1
        let and_term = tm.mk_and(vec![var, const_b1]);

        // Substitute with b2
        let mut subst = FxHashMap::default();
        subst.insert(var, const_b2);
        let result = tm.substitute(and_term, &subst);

        // Should equal: b2 AND b1
        let expected = tm.mk_and(vec![const_b2, const_b1]);

        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, result), get_bool_value(&tm, expected)) {
            prop_assert_eq!(v1, v2);
        }
    }

    // =====================================
    // Comparison Properties
    // =====================================

    /// Test that x = x is always true for constants
    #[test]
    fn equality_reflexive(n in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let eq = tm.mk_eq(t, t);

        // x = x should be true
        prop_assert_eq!(get_bool_value(&tm, eq), Some(true));
    }

    /// Test that if a = b then b = a (symmetry)
    #[test]
    fn equality_symmetric(a in small_int_strategy(), b in small_int_strategy()) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let eq1 = tm.mk_eq(ta, tb);
        let eq2 = tm.mk_eq(tb, ta);

        // (a = b) <=> (b = a)
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, eq1), get_bool_value(&tm, eq2)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that x < y implies not (y < x)
    #[test]
    fn less_than_asymmetric(a in small_int_strategy(), b in small_int_strategy()) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let lt_ab = tm.mk_lt(ta, tb);
        let lt_ba = tm.mk_lt(tb, ta);

        // If a < b, then not (b < a)
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, lt_ab), get_bool_value(&tm, lt_ba))
            && v1
        {
            prop_assert!(!v2);
        }
    }

    /// Test trichotomy: exactly one of x < y, x = y, x > y holds for constants
    #[test]
    fn trichotomy_property(a in small_int_strategy(), b in small_int_strategy()) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let lt = tm.mk_lt(ta, tb);
        let eq = tm.mk_eq(ta, tb);
        let gt = tm.mk_gt(ta, tb);

        if let (Some(v_lt), Some(v_eq), Some(v_gt)) = (
            get_bool_value(&tm, lt),
            get_bool_value(&tm, eq),
            get_bool_value(&tm, gt)
        ) {
            // Exactly one should be true
            let count = [v_lt, v_eq, v_gt].iter().filter(|&&x| x).count();
            prop_assert_eq!(count, 1);
        }
    }

    // =====================================
    // Traversal Properties
    // =====================================

    /// Test that traversing a term and collecting all subterms includes the term itself
    #[test]
    fn traversal_includes_root(n in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));

        let subterms = traversal::collect_subterms(t, &tm);

        prop_assert!(subterms.contains(&t));
    }

    /// Test that the number of unique subterms is reasonable
    #[test]
    fn traversal_subterm_count(
        a in small_int_strategy(),
        b in small_int_strategy(),
        c in small_int_strategy()
    ) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));
        let tc = tm.mk_int(BigInt::from(c));

        // Create: (a + b) * c
        let sum = tm.mk_add(vec![ta, tb]);
        let prod = tm.mk_mul(vec![sum, tc]);

        let subterms = traversal::collect_subterms(prod, &tm);

        // Should have at least 3 terms
        prop_assert!(subterms.len() >= 3);
    }

    /// Test that collecting free variables works correctly
    #[test]
    fn free_variables_collection(
        x_name in var_name_strategy(),
        y_name in var_name_strategy(),
        n in small_int_strategy()
    ) {
        let mut tm = TermManager::new();
        let sort = tm.sorts.int_sort;

        let x = tm.mk_var(&x_name, sort);
        let y = tm.mk_var(&y_name, sort);
        let c = tm.mk_int(BigInt::from(n));

        // Create: x + y + c
        let term = tm.mk_add(vec![x, y, c]);

        let free_vars = traversal::collect_free_vars(term, &tm);

        // Should contain x and y, but not c (which is a constant)
        if x_name != y_name {
            prop_assert!(free_vars.contains(&x));
            prop_assert!(free_vars.contains(&y));
            prop_assert_eq!(free_vars.len(), 2);
        } else {
            // If same name, should only have one variable
            prop_assert_eq!(free_vars.len(), 1);
        }
    }

    // =====================================
    // Simplification Properties
    // =====================================

    /// Test that simplification is idempotent
    #[test]
    fn simplification_idempotent(
        a in small_int_strategy(),
        b in small_int_strategy()
    ) {
        let mut tm = TermManager::new();
        let ta = tm.mk_int(BigInt::from(a));
        let tb = tm.mk_int(BigInt::from(b));

        let sum = tm.mk_add(vec![ta, tb]);
        let simplified1 = tm.simplify(sum);
        let simplified2 = tm.simplify(simplified1);

        prop_assert_eq!(simplified1, simplified2);
    }

    /// Test that simplification preserves semantics
    #[test]
    fn simplification_preserves_value(n in small_int_strategy()) {
        let mut tm = TermManager::new();
        let t = tm.mk_int(BigInt::from(n));
        let zero = tm.mk_int(BigInt::zero());

        // Create: n + 0
        let sum = tm.mk_add(vec![t, zero]);
        let simplified = tm.simplify(sum);

        // Original and simplified should have same value
        if let (Some(v1), Some(v2)) = (get_int_value(&tm, sum), get_int_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }

    /// Test that not(not(x)) simplifies to x for boolean constants
    #[test]
    fn double_negation_simplifies(b in proptest::bool::ANY) {
        let mut tm = TermManager::new();
        let t = tm.mk_bool(b);
        let not_t = tm.mk_not(t);
        let not_not_t = tm.mk_not(not_t);
        let simplified = tm.simplify(not_not_t);

        // Should simplify back to original value
        if let (Some(v1), Some(v2)) = (get_bool_value(&tm, t), get_bool_value(&tm, simplified)) {
            prop_assert_eq!(v1, v2);
        }
    }
}
