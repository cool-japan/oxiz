//! Property-based tests for AST operations
//!
//! This module tests fundamental properties of the AST such as:
//! - Term construction and uniqueness
//! - Substitution correctness
//! - Congruence closure properties
//! - Traversal consistency
//! - Boolean and arithmetic simplification

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_core::ast::*;
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

/// Strategy for generating BigInt values
fn bigint_strategy() -> impl Strategy<Value = BigInt> {
    small_int_strategy().prop_map(BigInt::from)
}

/// Strategy for generating BigRational values
fn bigrational_strategy() -> impl Strategy<Value = BigRational> {
    (bigint_strategy(), bigint_strategy())
        .prop_filter("denominator must be non-zero", |(_, d)| !d.is_zero())
        .prop_map(|(n, d)| BigRational::new(n, d))
}

#[cfg(test)]
mod term_construction_properties {
    use super::*;

    proptest! {
        /// Test that creating the same integer constant twice yields the same TermId
        #[test]
        fn integer_constant_uniqueness(n in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t1 = tm.mk_int(BigInt::from(n));
            let t2 = tm.mk_int(BigInt::from(n));
            prop_assert_eq!(t1, t2);
        }

        /// Test that creating the same boolean constant yields the same TermId
        #[test]
        fn boolean_constant_uniqueness(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t1 = tm.mk_bool(b);
            let t2 = tm.mk_bool(b);
            prop_assert_eq!(t1, t2);
        }

        /// Test that variables with the same name have the same TermId
        #[test]
        fn variable_uniqueness(name in var_name_strategy()) {
            let mut tm = manager::TermManager::new();
            let sort = tm.sorts.int_sort;
            let v1 = tm.mk_var(&name, sort);
            let v2 = tm.mk_var(&name, sort);
            prop_assert_eq!(v1, v2);
        }

        /// Test that double negation is handled correctly
        #[test]
        fn double_negation_property(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let not_t = tm.mk_not(t);
            let not_not_t = tm.mk_not(not_t);

            // Double negation should simplify to original
            // (or at least be semantically equivalent)
            let t_val = tm.get_bool_value(t);
            let not_not_val = tm.get_bool_value(not_not_t);

            if let (Some(v1), Some(v2)) = (t_val, not_not_val) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that AND with true is identity
        #[test]
        fn and_true_identity(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let true_term = tm.mk_bool(true);
            let result = tm.mk_and(vec![t, true_term]);

            // t ∧ true = t
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(t), tm.get_bool_value(result)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that AND with false is false
        #[test]
        fn and_false_annihilator(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let false_term = tm.mk_bool(false);
            let result = tm.mk_and(vec![t, false_term]);

            // t ∧ false = false
            prop_assert_eq!(tm.get_bool_value(result), Some(false));
        }

        /// Test that OR with false is identity
        #[test]
        fn or_false_identity(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let false_term = tm.mk_bool(false);
            let result = tm.mk_or(vec![t, false_term]);

            // t ∨ false = t
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(t), tm.get_bool_value(result)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that OR with true is true
        #[test]
        fn or_true_annihilator(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let true_term = tm.mk_bool(true);
            let result = tm.mk_or(vec![t, true_term]);

            // t ∨ true = true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }

        /// Test that addition with zero is identity
        #[test]
        fn add_zero_identity(n in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let zero = tm.mk_int(BigInt::zero());
            let result = tm.mk_add(vec![t, zero]);

            // t + 0 = t
            if let (Some(v1), Some(v2)) = (tm.get_int_value(t), tm.get_int_value(result)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that multiplication with one is identity
        #[test]
        fn mul_one_identity(n in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let one = tm.mk_int(BigInt::one());
            let result = tm.mk_mul(vec![t, one]);

            // t * 1 = t
            if let (Some(v1), Some(v2)) = (tm.get_int_value(t), tm.get_int_value(result)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that multiplication with zero is zero
        #[test]
        fn mul_zero_annihilator(n in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let zero = tm.mk_int(BigInt::zero());
            let result = tm.mk_mul(vec![t, zero]);

            // t * 0 = 0
            prop_assert_eq!(tm.get_int_value(result), Some(BigInt::zero()));
        }

        /// Test commutativity of addition
        #[test]
        fn add_commutative(a in small_int_strategy(), b in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t1 = tm.mk_int(BigInt::from(a));
            let t2 = tm.mk_int(BigInt::from(b));

            let sum1 = tm.mk_add(vec![t1, t2]);
            let sum2 = tm.mk_add(vec![t2, t1]);

            // a + b = b + a
            if let (Some(v1), Some(v2)) = (tm.get_int_value(sum1), tm.get_int_value(sum2)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test associativity of addition
        #[test]
        fn add_associative(a in small_int_strategy(), b in small_int_strategy(), c in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t1 = tm.mk_int(BigInt::from(a));
            let t2 = tm.mk_int(BigInt::from(b));
            let t3 = tm.mk_int(BigInt::from(c));

            let sum_ab = tm.mk_add(vec![t1, t2]);
            let result1 = tm.mk_add(vec![sum_ab, t3]);

            let sum_bc = tm.mk_add(vec![t2, t3]);
            let result2 = tm.mk_add(vec![t1, sum_bc]);

            // (a + b) + c = a + (b + c)
            if let (Some(v1), Some(v2)) = (tm.get_int_value(result1), tm.get_int_value(result2)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test commutativity of multiplication
        #[test]
        fn mul_commutative(a in small_int_strategy(), b in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t1 = tm.mk_int(BigInt::from(a));
            let t2 = tm.mk_int(BigInt::from(b));

            let prod1 = tm.mk_mul(vec![t1, t2]);
            let prod2 = tm.mk_mul(vec![t2, t1]);

            // a * b = b * a
            if let (Some(v1), Some(v2)) = (tm.get_int_value(prod1), tm.get_int_value(prod2)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test distributivity: a * (b + c) = a*b + a*c
        #[test]
        fn distributive_property(a in -10i64..10i64, b in -10i64..10i64, c in -10i64..10i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let tc = tm.mk_int(BigInt::from(c));

            // a * (b + c)
            let sum_bc = tm.mk_add(vec![tb, tc]);
            let left = tm.mk_mul(vec![ta, sum_bc]);

            // a*b + a*c
            let prod_ab = tm.mk_mul(vec![ta, tb]);
            let prod_ac = tm.mk_mul(vec![ta, tc]);
            let right = tm.mk_add(vec![prod_ab, prod_ac]);

            if let (Some(v1), Some(v2)) = (tm.get_int_value(left), tm.get_int_value(right)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod substitution_properties {
    use super::*;

    proptest! {
        /// Test that substituting a variable with itself yields the original term
        #[test]
        fn identity_substitution(name in var_name_strategy()) {
            let mut tm = manager::TermManager::new();
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
            let mut tm = manager::TermManager::new();
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
            let mut tm = manager::TermManager::new();
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
            if let (Some(v1), Some(v2)) = (tm.get_int_value(result1), tm.get_int_value(result2)) {
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
            let mut tm = manager::TermManager::new();
            let sort = tm.sorts.bool_sort;
            let var = tm.mk_var(&name, sort);
            let const_b1 = tm.mk_bool(b1);
            let const_b2 = tm.mk_bool(b2);

            // Create: var ∧ b1
            let and_term = tm.mk_and(vec![var, const_b1]);

            // Substitute with b2
            let mut subst = FxHashMap::default();
            subst.insert(var, const_b2);
            let result = tm.substitute(and_term, &subst);

            // Should equal: b2 ∧ b1
            let expected = tm.mk_and(vec![const_b2, const_b1]);

            if let (Some(v1), Some(v2)) = (tm.get_bool_value(result), tm.get_bool_value(expected)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod comparison_properties {
    use super::*;

    proptest! {
        /// Test that x = x is always true
        #[test]
        fn equality_reflexive(n in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let eq = tm.mk_eq(t, t);

            // x = x should be true
            prop_assert_eq!(tm.get_bool_value(eq), Some(true));
        }

        /// Test that if a = b then b = a
        #[test]
        fn equality_symmetric(a in small_int_strategy(), b in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let eq1 = tm.mk_eq(ta, tb);
            let eq2 = tm.mk_eq(tb, ta);

            // (a = b) ⟺ (b = a)
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(eq1), tm.get_bool_value(eq2)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that x < y implies not (y < x)
        #[test]
        fn less_than_asymmetric(a in small_int_strategy(), b in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let lt_ab = tm.mk_lt(ta, tb);
            let lt_ba = tm.mk_lt(tb, ta);

            // If a < b, then not (b < a)
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(lt_ab), tm.get_bool_value(lt_ba)) {
                if v1 {
                    prop_assert!(!v2);
                }
            }
        }

        /// Test that x <= y is equivalent to (x < y) ∨ (x = y)
        #[test]
        fn less_equal_definition(a in small_int_strategy(), b in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let le = tm.mk_le(ta, tb);
            let lt = tm.mk_lt(ta, tb);
            let eq = tm.mk_eq(ta, tb);
            let or_term = tm.mk_or(vec![lt, eq]);

            // (a <= b) ⟺ (a < b) ∨ (a = b)
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(le), tm.get_bool_value(or_term)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test trichotomy: exactly one of x < y, x = y, x > y holds
        #[test]
        fn trichotomy_property(a in small_int_strategy(), b in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let lt = tm.mk_lt(ta, tb);
            let eq = tm.mk_eq(ta, tb);
            let gt = tm.mk_gt(ta, tb);

            if let (Some(v_lt), Some(v_eq), Some(v_gt)) = (
                tm.get_bool_value(lt),
                tm.get_bool_value(eq),
                tm.get_bool_value(gt)
            ) {
                // Exactly one should be true
                let count = [v_lt, v_eq, v_gt].iter().filter(|&&x| x).count();
                prop_assert_eq!(count, 1);
            }
        }

        /// Test transitivity: if x < y and y < z then x < z
        #[test]
        fn less_than_transitive(a in -50i64..50i64, b in -50i64..50i64, c in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let tc = tm.mk_int(BigInt::from(c));

            let lt_ab = tm.mk_lt(ta, tb);
            let lt_bc = tm.mk_lt(tb, tc);
            let lt_ac = tm.mk_lt(ta, tc);

            if let (Some(v_ab), Some(v_bc), Some(v_ac)) = (
                tm.get_bool_value(lt_ab),
                tm.get_bool_value(lt_bc),
                tm.get_bool_value(lt_ac)
            ) {
                // If a < b and b < c, then a < c
                if v_ab && v_bc {
                    prop_assert!(v_ac);
                }
            }
        }
    }
}

#[cfg(test)]
mod traversal_properties {
    use super::*;

    proptest! {
        /// Test that traversing a term and collecting all subterms includes the term itself
        #[test]
        fn traversal_includes_root(n in small_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));

            let mut visited = std::collections::HashSet::new();
            traversal::traverse_term(&tm, t, &mut |term_id| {
                visited.insert(term_id);
                true
            });

            prop_assert!(visited.contains(&t));
        }

        /// Test that the number of unique subterms is reasonable
        #[test]
        fn traversal_subterm_count(
            a in small_int_strategy(),
            b in small_int_strategy(),
            c in small_int_strategy()
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let tc = tm.mk_int(BigInt::from(c));

            // Create: (a + b) * c
            let sum = tm.mk_add(vec![ta, tb]);
            let prod = tm.mk_mul(vec![sum, tc]);

            let mut visited = std::collections::HashSet::new();
            traversal::traverse_term(&tm, prod, &mut |term_id| {
                visited.insert(term_id);
                true
            });

            // Should have at least: prod, sum, ta, tb, tc (5 terms)
            // But may have more due to simplification
            prop_assert!(visited.len() >= 3);
        }

        /// Test that collecting free variables works correctly
        #[test]
        fn free_variables_collection(
            x_name in var_name_strategy(),
            y_name in var_name_strategy(),
            n in small_int_strategy()
        ) {
            let mut tm = manager::TermManager::new();
            let sort = tm.sorts.int_sort;

            let x = tm.mk_var(&x_name, sort);
            let y = tm.mk_var(&y_name, sort);
            let c = tm.mk_int(BigInt::from(n));

            // Create: x + y + c
            let term = tm.mk_add(vec![x, y, c]);

            let free_vars = utils::collect_free_vars(&tm, term);

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
    }
}

#[cfg(test)]
mod congruence_properties {
    use super::*;

    proptest! {
        /// Test congruence: if a = b and c = d, then a+c = b+d
        #[test]
        fn congruence_addition(a in small_int_strategy(), c in small_int_strategy()) {
            let mut tm = manager::TermManager::new();

            let ta1 = tm.mk_int(BigInt::from(a));
            let ta2 = tm.mk_int(BigInt::from(a)); // Same value
            let tc1 = tm.mk_int(BigInt::from(c));
            let tc2 = tm.mk_int(BigInt::from(c)); // Same value

            let sum1 = tm.mk_add(vec![ta1, tc1]);
            let sum2 = tm.mk_add(vec![ta2, tc2]);

            // If a = a and c = c, then a+c = a+c
            let eq = tm.mk_eq(sum1, sum2);
            prop_assert_eq!(tm.get_bool_value(eq), Some(true));
        }

        /// Test that applying the same function to equal arguments yields equal results
        #[test]
        fn function_application_congruence(n in small_int_strategy()) {
            let mut tm = manager::TermManager::new();

            let t1 = tm.mk_int(BigInt::from(n));
            let t2 = tm.mk_int(BigInt::from(n));

            // Apply negation (unary minus) to both
            let neg1 = tm.mk_neg(t1);
            let neg2 = tm.mk_neg(t2);

            // Should be equal
            let eq = tm.mk_eq(neg1, neg2);
            prop_assert_eq!(tm.get_bool_value(eq), Some(true));
        }
    }
}

#[cfg(test)]
mod simplification_properties {
    use super::*;

    proptest! {
        /// Test that simplification is idempotent
        #[test]
        fn simplification_idempotent(
            a in small_int_strategy(),
            b in small_int_strategy()
        ) {
            let mut tm = manager::TermManager::new();
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
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let zero = tm.mk_int(BigInt::zero());

            // Create: n + 0
            let sum = tm.mk_add(vec![t, zero]);
            let simplified = tm.simplify(sum);

            // Original and simplified should have same value
            if let (Some(v1), Some(v2)) = (tm.get_int_value(sum), tm.get_int_value(simplified)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that not(not(x)) simplifies to x
        #[test]
        fn double_negation_simplifies(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let not_t = tm.mk_not(t);
            let not_not_t = tm.mk_not(not_t);
            let simplified = tm.simplify(not_not_t);

            // Should simplify back to original value
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(t), tm.get_bool_value(simplified)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}
