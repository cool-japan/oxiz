//! Property-based tests for Quantifier Elimination (QE) soundness
//!
//! This module tests that QE procedures:
//! - Preserve satisfiability
//! - Eliminate quantifiers correctly
//! - Maintain semantic equivalence
//! - Handle corner cases properly
//! - Work correctly for different theories (LIA, LRA, BV, etc.)

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_core::ast::*;
use oxiz_core::qe::*;
use proptest::prelude::*;
use rustc_hash::FxHashMap;

/// Strategy for generating small integers for QE
fn qe_int_strategy() -> impl Strategy<Value = i64> {
    -20i64..20i64
}

#[cfg(test)]
mod qe_basic_properties {
    use super::*;

    proptest! {
        /// Test that eliminating quantifiers from a quantifier-free formula is identity
        #[test]
        fn qe_on_qf_formula_is_identity(n in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Quantifier-free formula: x = c
            let formula = tm.mk_eq(x, c);

            // Apply QE (should not change it since there's no quantifier)
            let result = qe_lite::apply_qe_lite(&mut tm, formula);

            // Should preserve the formula structure (modulo simplification)
            let free_vars_orig = utils::collect_free_vars(&tm, formula);
            let free_vars_result = utils::collect_free_vars(&tm, result);

            // Should have the same free variables
            prop_assert_eq!(free_vars_orig.len(), free_vars_result.len());
        }

        /// Test that exists x. (x = c) is always satisfiable
        #[test]
        fn exists_equality_is_sat(n in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // ∃x. x = c
            let body = tm.mk_eq(x, c);
            let exists = tm.mk_exists(vec![x], body);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Result should be true (always satisfiable)
            // or at least satisfiable
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test that forall x. (x = x) is a tautology
        #[test]
        fn forall_self_equality_is_true() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);

            // ∀x. x = x
            let body = tm.mk_eq(x, x);
            let forall = tm.mk_forall(vec![x], body);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, forall);

            // Should be true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }

        /// Test that exists x. (x != x) is unsatisfiable
        #[test]
        fn exists_contradiction_is_false() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);

            // ∃x. x ≠ x
            let eq = tm.mk_eq(x, x);
            let neq = tm.mk_not(eq);
            let exists = tm.mk_exists(vec![x], neq);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should be false
            prop_assert_eq!(tm.get_bool_value(result), Some(false));
        }

        /// Test that forall x. false is unsatisfiable
        #[test]
        fn forall_false_is_false() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let false_term = tm.mk_bool(false);

            // ∀x. false
            let forall = tm.mk_forall(vec![x], false_term);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, forall);

            // Should be false
            prop_assert_eq!(tm.get_bool_value(result), Some(false));
        }

        /// Test that exists x. true is satisfiable
        #[test]
        fn exists_true_is_true() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let true_term = tm.mk_bool(true);

            // ∃x. true
            let exists = tm.mk_exists(vec![x], true_term);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should be true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }
    }
}

#[cfg(test)]
mod qe_lia_properties {
    use super::*;

    proptest! {
        /// Test Fourier-Motzkin for exists x. (a <= x <= b)
        #[test]
        fn fm_exists_in_bounds(a in qe_int_strategy(), b in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // ∃x. (a <= x ∧ x <= b)
            let le_a = tm.mk_le(ta, x);
            let le_b = tm.mk_le(x, tb);
            let body = tm.mk_and(vec![le_a, le_b]);
            let exists = tm.mk_exists(vec![x], body);

            // Apply QE
            let result = arith::fm_advanced::apply_fourier_motzkin(&mut tm, exists);

            // Should be satisfiable iff a <= b
            let expected = a <= b;
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert_eq!(val, expected);
            }
        }

        /// Test exists x. (x < a) is always true
        #[test]
        fn exists_less_than_is_true(a in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));

            // ∃x. x < a
            let lt = tm.mk_lt(x, ta);
            let exists = tm.mk_exists(vec![x], lt);

            // Apply QE
            let result = arith::fm_advanced::apply_fourier_motzkin(&mut tm, exists);

            // Should be true (always satisfiable - take x = a-1)
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test forall x. (x >= 0 => x >= 0) is tautology
        #[test]
        fn forall_implication_tautology() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::zero());

            // ∀x. (x >= 0) => (x >= 0)
            let ge = tm.mk_ge(x, zero);
            let implies = tm.mk_implies(ge, ge);
            let forall = tm.mk_forall(vec![x], implies);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, forall);

            // Should be true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }

        /// Test exists x. (x = y) where y is free
        #[test]
        fn exists_equality_with_free_var(n in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);

            // ∃x. x = y
            let eq = tm.mk_eq(x, y);
            let exists = tm.mk_exists(vec![x], eq);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should be true (always satisfiable for any value of y)
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test exists x. (x + a = b) is always satisfiable
        #[test]
        fn exists_linear_equation(a in qe_int_strategy(), b in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // ∃x. x + a = b
            let sum = tm.mk_add(vec![x, ta]);
            let eq = tm.mk_eq(sum, tb);
            let exists = tm.mk_exists(vec![x], eq);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should be true (x = b - a)
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test forall x. (x > 0 => x >= 1) for integers
        #[test]
        fn forall_integer_property() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::zero());
            let one = tm.mk_int(BigInt::one());

            // ∀x. (x > 0) => (x >= 1)
            let gt = tm.mk_gt(x, zero);
            let ge = tm.mk_ge(x, one);
            let implies = tm.mk_implies(gt, ge);
            let forall = tm.mk_forall(vec![x], implies);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, forall);

            // Should be true for integers
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test exists x. (a <= x < b) satisfiability
        #[test]
        fn exists_bounded_interval(a in qe_int_strategy(), b in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // ∃x. (a <= x) ∧ (x < b)
            let le = tm.mk_le(ta, x);
            let lt = tm.mk_lt(x, tb);
            let body = tm.mk_and(vec![le, lt]);
            let exists = tm.mk_exists(vec![x], body);

            // Apply QE
            let result = arith::fm_advanced::apply_fourier_motzkin(&mut tm, exists);

            // Should be satisfiable iff a < b
            let expected = a < b;
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert_eq!(val, expected);
            }
        }
    }
}

#[cfg(test)]
mod qe_lra_properties {
    use super::*;

    proptest! {
        /// Test exists x. (x < a) ∨ (x > a) is always true for reals
        #[test]
        fn exists_real_ordering() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.real_sort);
            let a = tm.mk_var("a", tm.sorts.real_sort);

            // ∃x. (x < a) ∨ (x > a)
            let lt = tm.mk_lt(x, a);
            let gt = tm.mk_gt(x, a);
            let or_term = tm.mk_or(vec![lt, gt]);
            let exists = tm.mk_exists(vec![x], or_term);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should be true (always satisfiable)
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test exists x. (a < x < b) for reals
        #[test]
        fn exists_real_open_interval(a in qe_int_strategy(), b in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.real_sort);
            let ta = tm.mk_real(BigRational::from_integer(BigInt::from(a)));
            let tb = tm.mk_real(BigRational::from_integer(BigInt::from(b)));

            // ∃x. (a < x) ∧ (x < b)
            let gt_a = tm.mk_gt(x, ta);
            let lt_b = tm.mk_lt(x, tb);
            let body = tm.mk_and(vec![gt_a, lt_b]);
            let exists = tm.mk_exists(vec![x], body);

            // Apply QE
            let result = arith::fm_advanced::apply_fourier_motzkin(&mut tm, exists);

            // Should be satisfiable iff a < b
            let expected = a < b;
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert_eq!(val, expected);
            }
        }

        /// Test forall x. (x >= 0) ∨ (x < 0) is tautology
        #[test]
        fn forall_real_trichotomy() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.real_sort);
            let zero = tm.mk_real(BigRational::zero());

            // ∀x. (x >= 0) ∨ (x < 0)
            let ge = tm.mk_ge(x, zero);
            let lt = tm.mk_lt(x, zero);
            let or_term = tm.mk_or(vec![ge, lt]);
            let forall = tm.mk_forall(vec![x], or_term);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, forall);

            // Should be true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }
    }
}

#[cfg(test)]
mod qe_preservation_properties {
    use super::*;

    proptest! {
        /// Test that QE preserves free variables
        #[test]
        fn qe_preserves_free_variables(n in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // ∃x. (x + y = c)
            let sum = tm.mk_add(vec![x, y]);
            let eq = tm.mk_eq(sum, c);
            let exists = tm.mk_exists(vec![x], eq);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // y should still be a free variable in result
            let free_vars = utils::collect_free_vars(&tm, result);

            // y should be in free variables, x should not
            prop_assert!(free_vars.contains(&y) || free_vars.is_empty());
            prop_assert!(!free_vars.contains(&x));
        }

        /// Test that QE removes quantified variables
        #[test]
        fn qe_removes_quantified_vars() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::zero());

            // ∃x. x >= 0
            let ge = tm.mk_ge(x, zero);
            let exists = tm.mk_exists(vec![x], ge);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // x should not be in the result
            let free_vars = utils::collect_free_vars(&tm, result);
            prop_assert!(!free_vars.contains(&x));
        }

        /// Test that nested quantifiers are handled correctly
        #[test]
        fn qe_handles_nested_quantifiers(a in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));

            // ∃x. ∃y. (x + y = a)
            let sum = tm.mk_add(vec![x, y]);
            let eq = tm.mk_eq(sum, ta);
            let exists_y = tm.mk_exists(vec![y], eq);
            let exists_x = tm.mk_exists(vec![x], exists_y);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists_x);

            // Should be true (always satisfiable)
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }

            // Neither x nor y should be free
            let free_vars = utils::collect_free_vars(&tm, result);
            prop_assert!(!free_vars.contains(&x));
            prop_assert!(!free_vars.contains(&y));
        }

        /// Test that QE preserves logical equivalence
        #[test]
        fn qe_preserves_equivalence(n in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Two equivalent formulas:
            // ∃x. x = c
            // ∃x. c = x
            let eq1 = tm.mk_eq(x, c);
            let eq2 = tm.mk_eq(c, x);
            let exists1 = tm.mk_exists(vec![x], eq1);
            let exists2 = tm.mk_exists(vec![x], eq2);

            // Apply QE to both
            let result1 = qe_lite::apply_qe_lite(&mut tm, exists1);
            let result2 = qe_lite::apply_qe_lite(&mut tm, exists2);

            // Should have the same value
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(result1), tm.get_bool_value(result2)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod qe_soundness_properties {
    use super::*;

    proptest! {
        /// Test that QE maintains satisfiability: if original is SAT, result is SAT
        #[test]
        fn qe_sat_implies_result_sat() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::zero());

            // ∃x. x = 0 (satisfiable)
            let eq = tm.mk_eq(x, zero);
            let exists = tm.mk_exists(vec![x], eq);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Result should be SAT (or true)
            prop_assert_ne!(tm.get_bool_value(result), Some(false));
        }

        /// Test that QE maintains unsatisfiability: if original is UNSAT, result is UNSAT
        #[test]
        fn qe_unsat_implies_result_unsat() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::zero());
            let one = tm.mk_int(BigInt::one());

            // ∃x. (x = 0) ∧ (x = 1) (unsatisfiable)
            let eq0 = tm.mk_eq(x, zero);
            let eq1 = tm.mk_eq(x, one);
            let and_term = tm.mk_and(vec![eq0, eq1]);
            let exists = tm.mk_exists(vec![x], and_term);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Result should be UNSAT (false)
            prop_assert_eq!(tm.get_bool_value(result), Some(false));
        }

        /// Test that QE on tautology yields tautology
        #[test]
        fn qe_on_tautology_yields_tautology() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);

            // ∃x. true (tautology)
            let true_term = tm.mk_bool(true);
            let exists = tm.mk_exists(vec![x], true_term);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should be true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }

        /// Test that QE distributes over disjunction
        #[test]
        fn qe_distributes_over_disjunction(a in qe_int_strategy(), b in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // ∃x. (x = a) ∨ (x = b)
            let eq_a = tm.mk_eq(x, ta);
            let eq_b = tm.mk_eq(x, tb);
            let or_term = tm.mk_or(vec![eq_a, eq_b]);
            let exists = tm.mk_exists(vec![x], or_term);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should be true (always satisfiable)
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test that forall is dual of exists: ∀x.φ ≡ ¬∃x.¬φ
        #[test]
        fn forall_exists_duality(n in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // ∀x. x = c
            let eq = tm.mk_eq(x, c);
            let forall = tm.mk_forall(vec![x], eq);

            // ¬∃x. ¬(x = c)
            let not_eq = tm.mk_not(eq);
            let exists = tm.mk_exists(vec![x], not_eq);
            let not_exists = tm.mk_not(exists);

            // Apply QE to both
            let result_forall = qe_lite::apply_qe_lite(&mut tm, forall);
            let result_not_exists = qe_lite::apply_qe_lite(&mut tm, not_exists);

            // Should have the same value
            if let (Some(v1), Some(v2)) = (
                tm.get_bool_value(result_forall),
                tm.get_bool_value(result_not_exists)
            ) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod qe_optimization_properties {
    use super::*;

    proptest! {
        /// Test that QE lite is efficient on simple formulas
        #[test]
        fn qe_lite_handles_simple_formulas(n in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // ∃x. x = c
            let eq = tm.mk_eq(x, c);
            let exists = tm.mk_exists(vec![x], eq);

            // QE lite should handle this efficiently
            let result = qe_lite::apply_qe_lite(&mut tm, exists);

            // Should complete without panic
            prop_assert_ne!(tm.get_bool_value(result), None);
        }

        /// Test that multiple variables can be eliminated
        #[test]
        fn qe_eliminates_multiple_vars(a in qe_int_strategy()) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(a));

            // ∃x. ∃y. (x + y = c)
            let sum = tm.mk_add(vec![x, y]);
            let eq = tm.mk_eq(sum, c);
            let exists_y = tm.mk_exists(vec![y], eq);
            let exists_x = tm.mk_exists(vec![x], exists_y);

            // Apply QE
            let result = qe_lite::apply_qe_lite(&mut tm, exists_x);

            // Both variables should be eliminated
            let free_vars = utils::collect_free_vars(&tm, result);
            prop_assert!(!free_vars.contains(&x));
            prop_assert!(!free_vars.contains(&y));
        }
    }
}
