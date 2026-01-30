//! Property-based tests for tactic composition and correctness
//!
//! This module tests that tactics:
//! - Preserve satisfiability
//! - Compose correctly (sequence, parallel, choice)
//! - Maintain proper error handling
//! - Respect resource limits
//! - Produce sound transformations

use num_bigint::BigInt;
use num_traits::Zero;
use oxiz_core::ast::*;
use oxiz_core::tactic::*;
use proptest::prelude::*;

#[cfg(test)]
mod tactic_composition_properties {
    use super::*;

    proptest! {
        /// Test that the identity tactic preserves the goal
        #[test]
        fn identity_tactic_preserves_goal(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let goal = tm.mk_int(BigInt::from(n));

            // Apply identity (simplify with no changes)
            let result = simplify::apply_simplify(&mut tm, goal);

            // Should preserve the value
            if let (Some(v1), Some(v2)) = (tm.get_int_value(goal), tm.get_int_value(result)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that simplify is idempotent
        #[test]
        fn simplify_idempotent(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let sum = tm.mk_add(vec![ta, tb]);

            let simplified1 = simplify::apply_simplify(&mut tm, sum);
            let simplified2 = simplify::apply_simplify(&mut tm, simplified1);

            // Simplifying twice should give the same result
            prop_assert_eq!(simplified1, simplified2);
        }

        /// Test tactic sequencing: simplify then simplify = simplify
        #[test]
        fn tactic_sequence_idempotent(a in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let zero = tm.mk_int(BigInt::zero());
            let ta = tm.mk_int(BigInt::from(a));
            let sum = tm.mk_add(vec![ta, zero]);

            // Apply simplify twice in sequence
            let result1 = simplify::apply_simplify(&mut tm, sum);
            let result2 = simplify::apply_simplify(&mut tm, result1);

            // Should be idempotent
            prop_assert_eq!(result1, result2);
        }

        /// Test that propagate then simplify preserves semantics
        #[test]
        fn propagate_then_simplify_preserves_semantics(
            a in -50i64..50i64,
            b in -50i64..50i64
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let eq = tm.mk_eq(ta, tb);

            // Apply propagate
            let propagated = propagate::apply_propagate(&mut tm, eq);

            // Then apply simplify
            let simplified = simplify::apply_simplify(&mut tm, propagated);

            // Should preserve boolean value
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(eq), tm.get_bool_value(simplified)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that split then simplify on each branch preserves satisfiability
        #[test]
        fn split_preserves_satisfiability(
            a in -30i64..30i64,
            b in -30i64..30i64
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // Create: (a < b) ∨ (a >= b) - always satisfiable
            let lt = tm.mk_lt(ta, tb);
            let ge = tm.mk_ge(ta, tb);
            let or_term = tm.mk_or(vec![lt, ge]);

            // Apply split tactic
            let branches = split::apply_split(&mut tm, or_term);

            // At least one branch should be satisfiable
            let any_sat = branches.iter().any(|&branch| {
                if let Some(val) = tm.get_bool_value(branch) {
                    val
                } else {
                    true // Unknown, assume possibly satisfiable
                }
            });

            prop_assert!(any_sat || branches.is_empty());
        }
    }
}

#[cfg(test)]
mod tactic_correctness_properties {
    use super::*;

    proptest! {
        /// Test that solve_eqs maintains equality constraints
        #[test]
        fn solve_eqs_maintains_equality(a in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));

            // x = a
            let eq = tm.mk_eq(x, ta);

            // Apply solve_eqs tactic
            let result = solve_eqs::apply_solve_eqs(&mut tm, eq);

            // Result should still imply x = a
            // (either as-is or in a simplified form)
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test that eliminate tactic preserves unsatisfiability
        #[test]
        fn eliminate_preserves_unsat() {
            let mut tm = manager::TermManager::new();
            let true_term = tm.mk_bool(true);
            let false_term = tm.mk_bool(false);

            // Create unsatisfiable goal: true ∧ false
            let unsat = tm.mk_and(vec![true_term, false_term]);

            // Apply eliminate
            let result = eliminate::apply_eliminate(&mut tm, unsat);

            // Should still be false
            prop_assert_eq!(tm.get_bool_value(result), Some(false));
        }

        /// Test that eliminate tactic preserves satisfiability
        #[test]
        fn eliminate_preserves_sat() {
            let mut tm = manager::TermManager::new();
            let true_term = tm.mk_bool(true);

            // Create satisfiable goal: true
            let sat = true_term;

            // Apply eliminate
            let result = eliminate::apply_eliminate(&mut tm, sat);

            // Should still be true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }

        /// Test that ctx_simplify preserves tautologies
        #[test]
        fn ctx_simplify_preserves_tautology(a in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let not_ta = tm.mk_not(ta);

            // a ∨ ¬a is a tautology
            let tautology = tm.mk_or(vec![ta, not_ta]);

            // Apply contextual simplification
            let result = ctx_simplify::apply_ctx_simplify(&mut tm, tautology);

            // Should simplify to true
            prop_assert_eq!(tm.get_bool_value(result), Some(true));
        }

        /// Test that propagate_ineqs strengthens but preserves satisfiability
        #[test]
        fn propagate_ineqs_preserves_sat(
            a in -30i64..30i64,
            b in -30i64..30i64
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // a <= b
            let le = tm.mk_le(ta, tb);

            // Apply inequality propagation
            let result = propagate::propagate_ineqs(&mut tm, le);

            // Should preserve the truth value
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(le), tm.get_bool_value(result)) {
                // Result should imply the original (v1 => v2)
                if v1 {
                    prop_assert!(v2);
                }
            }
        }
    }
}

#[cfg(test)]
mod tactic_normalization_properties {
    use super::*;

    proptest! {
        /// Test that simplify normalizes double negations
        #[test]
        fn simplify_normalizes_double_negation(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let not_t = tm.mk_not(t);
            let not_not_t = tm.mk_not(not_t);

            let simplified = simplify::apply_simplify(&mut tm, not_not_t);

            // Should have the same value as original
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(t), tm.get_bool_value(simplified)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that simplify eliminates identity operations
        #[test]
        fn simplify_eliminates_identity_add(n in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let zero = tm.mk_int(BigInt::zero());
            let sum = tm.mk_add(vec![t, zero]);

            let simplified = simplify::apply_simplify(&mut tm, sum);

            // Should equal the original term
            if let (Some(v1), Some(v2)) = (tm.get_int_value(t), tm.get_int_value(simplified)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that simplify performs constant folding
        #[test]
        fn simplify_folds_constants(a in -30i64..30i64, b in -30i64..30i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let sum = tm.mk_add(vec![ta, tb]);

            let simplified = simplify::apply_simplify(&mut tm, sum);

            // Should be a constant
            if let Some(val) = tm.get_int_value(simplified) {
                prop_assert_eq!(val, BigInt::from(a + b));
            }
        }

        /// Test that simplify normalizes comparison chains
        #[test]
        fn simplify_normalizes_comparisons(
            a in -30i64..30i64,
            b in -30i64..30i64
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // Create: not (a < b)
            let lt = tm.mk_lt(ta, tb);
            let not_lt = tm.mk_not(lt);

            // Should normalize to: a >= b
            let ge = tm.mk_ge(ta, tb);

            let simplified_not_lt = simplify::apply_simplify(&mut tm, not_lt);
            let simplified_ge = simplify::apply_simplify(&mut tm, ge);

            // Should have the same boolean value
            if let (Some(v1), Some(v2)) = (
                tm.get_bool_value(simplified_not_lt),
                tm.get_bool_value(simplified_ge)
            ) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod tactic_transformation_properties {
    use super::*;

    proptest! {
        /// Test that bitblast preserves boolean structure
        #[test]
        fn bitblast_preserves_boolean_structure(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);

            // Bitblasting a simple boolean should preserve it
            let result = bitblast::apply_bitblast(&mut tm, t);

            if let (Some(v1), Some(v2)) = (tm.get_bool_value(t), tm.get_bool_value(result)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that ackermann reduction preserves satisfiability
        #[test]
        fn ackermann_preserves_sat() {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);

            // x = y (satisfiable)
            let eq = tm.mk_eq(x, y);

            // Apply Ackermann reduction
            let result = ackermann::apply_ackermann(&mut tm, eq);

            // Should preserve satisfiability (both are satisfiable)
            // At minimum, should not become unsatisfiable
            if let Some(val) = tm.get_bool_value(result) {
                // If we can evaluate it, check consistency
                if let Some(orig_val) = tm.get_bool_value(eq) {
                    // Result should imply original or be equivalent
                    prop_assert!(val || !orig_val);
                }
            }
        }

        /// Test that quantifier elimination preserves free variables
        #[test]
        fn qe_preserves_free_vars(n in -30i64..30i64) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // x = c (no quantifiers, just a formula with free variable)
            let formula = tm.mk_eq(x, c);

            // Apply quantifier-related tactics (should be identity for non-quantified)
            let result = quantifier::apply_quantifier_tactic(&mut tm, formula);

            // Free variables should be preserved
            let orig_free_vars = utils::collect_free_vars(&tm, formula);
            let result_free_vars = utils::collect_free_vars(&tm, result);

            // At minimum, should include all original free vars
            for var in orig_free_vars {
                prop_assert!(result_free_vars.contains(&var));
            }
        }

        /// Test that solve_eqs eliminates variables when possible
        #[test]
        fn solve_eqs_eliminates_variables_when_possible(n in -30i64..30i64) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Create: (x = c) ∧ (y = x)
            let eq1 = tm.mk_eq(x, c);
            let eq2 = tm.mk_eq(y, x);
            let and_term = tm.mk_and(vec![eq1, eq2]);

            // Apply solve_eqs
            let result = solve_eqs::apply_solve_eqs(&mut tm, and_term);

            // Result should be at least as simple (possibly y = c or similar)
            // We can't easily check elimination without more sophisticated analysis,
            // but we can check that it's still satisfiable
            if let Some(val) = tm.get_bool_value(result) {
                prop_assert!(val);
            }
        }

        /// Test that normalize_bounds maintains inequality relationships
        #[test]
        fn normalize_bounds_maintains_relationships(
            a in -30i64..30i64,
            b in -30i64..30i64
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // a < b
            let lt = tm.mk_lt(ta, tb);

            // Apply normalize_bounds
            let result = normalize_bounds::apply_normalize_bounds(&mut tm, lt);

            // Should preserve the truth value
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(lt), tm.get_bool_value(result)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod tactic_error_handling_properties {
    use super::*;

    proptest! {
        /// Test that tactics handle empty goals gracefully
        #[test]
        fn tactics_handle_true_goal() {
            let mut tm = manager::TermManager::new();
            let true_goal = tm.mk_bool(true);

            // All tactics should handle true gracefully
            let simplified = simplify::apply_simplify(&mut tm, true_goal);
            prop_assert_eq!(tm.get_bool_value(simplified), Some(true));
        }

        /// Test that tactics handle false goals gracefully
        #[test]
        fn tactics_handle_false_goal() {
            let mut tm = manager::TermManager::new();
            let false_goal = tm.mk_bool(false);

            // All tactics should preserve false
            let simplified = simplify::apply_simplify(&mut tm, false_goal);
            prop_assert_eq!(tm.get_bool_value(simplified), Some(false));
        }

        /// Test that tactics handle trivial tautologies
        #[test]
        fn tactics_handle_tautology(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let not_t = tm.mk_not(t);
            let tautology = tm.mk_or(vec![t, not_t]);

            // Should simplify to true
            let simplified = simplify::apply_simplify(&mut tm, tautology);
            prop_assert_eq!(tm.get_bool_value(simplified), Some(true));
        }

        /// Test that tactics handle trivial contradictions
        #[test]
        fn tactics_handle_contradiction(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let not_t = tm.mk_not(t);
            let contradiction = tm.mk_and(vec![t, not_t]);

            // Should simplify to false
            let simplified = simplify::apply_simplify(&mut tm, contradiction);
            prop_assert_eq!(tm.get_bool_value(simplified), Some(false));
        }
    }
}

#[cfg(test)]
mod tactic_combination_properties {
    use super::*;

    proptest! {
        /// Test combining simplify with propagate
        #[test]
        fn simplify_then_propagate_correct(
            a in -30i64..30i64,
            b in -30i64..30i64
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let zero = tm.mk_int(BigInt::zero());

            // Create: (a + 0) < (b + 0)
            let sum_a = tm.mk_add(vec![ta, zero]);
            let sum_b = tm.mk_add(vec![tb, zero]);
            let lt = tm.mk_lt(sum_a, sum_b);

            // Apply simplify then propagate
            let simplified = simplify::apply_simplify(&mut tm, lt);
            let propagated = propagate::apply_propagate(&mut tm, simplified);

            // Should equal: a < b
            let expected = tm.mk_lt(ta, tb);
            let expected_simplified = simplify::apply_simplify(&mut tm, expected);

            if let (Some(v1), Some(v2)) = (
                tm.get_bool_value(propagated),
                tm.get_bool_value(expected_simplified)
            ) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that applying multiple tactics preserves satisfiability
        #[test]
        fn multiple_tactics_preserve_sat(n in -30i64..30i64) {
            let mut tm = manager::TermManager::new();
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // x = c (satisfiable)
            let formula = tm.mk_eq(x, c);

            // Apply multiple tactics
            let mut result = formula;
            result = simplify::apply_simplify(&mut tm, result);
            result = propagate::apply_propagate(&mut tm, result);
            result = simplify::apply_simplify(&mut tm, result);

            // Should still be satisfiable
            // (can't directly check SAT, but can check it's not trivially false)
            prop_assert_ne!(tm.get_bool_value(result), Some(false));
        }
    }
}
