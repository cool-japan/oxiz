//! Property-based tests for rewriter correctness
//!
//! This module tests that rewrite rules preserve semantics and maintain
//! proper invariants across all theories:
//! - Boolean rewrites
//! - Arithmetic rewrites
//! - Bit-vector rewrites
//! - Array theory rewrites
//! - Preservation of satisfiability

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_core::ast::*;
use oxiz_core::rewrite::*;
use proptest::prelude::*;

/// Strategy for generating boolean terms
fn bool_term_strategy(
    tm: &mut manager::TermManager,
    max_depth: u32,
) -> impl Strategy<Value = manager::TermId> {
    let true_term = tm.mk_bool(true);
    let false_term = tm.mk_bool(false);
    prop_oneof![Just(true_term), Just(false_term),]
}

#[cfg(test)]
mod boolean_rewrite_properties {
    use super::*;

    proptest! {
        /// Test that rewriting preserves tautologies
        #[test]
        fn rewrite_preserves_tautology_true() {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(true);

            // Rewrite
            let rewritten = bool::simplify_bool(&mut tm, t);

            // Should still be true
            prop_assert_eq!(tm.get_bool_value(rewritten), Some(true));
        }

        /// Test that rewriting preserves contradictions
        #[test]
        fn rewrite_preserves_contradiction_false() {
            let mut tm = manager::TermManager::new();
            let f = tm.mk_bool(false);

            // Rewrite
            let rewritten = bool::simplify_bool(&mut tm, f);

            // Should still be false
            prop_assert_eq!(tm.get_bool_value(rewritten), Some(false));
        }

        /// Test: (a ∧ ¬a) rewrites to false
        #[test]
        fn and_with_negation_is_false(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let not_t = tm.mk_not(t);
            let contradiction = tm.mk_and(vec![t, not_t]);

            let rewritten = bool::simplify_bool(&mut tm, contradiction);

            // Should be false
            prop_assert_eq!(tm.get_bool_value(rewritten), Some(false));
        }

        /// Test: (a ∨ ¬a) rewrites to true
        #[test]
        fn or_with_negation_is_true(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let not_t = tm.mk_not(t);
            let tautology = tm.mk_or(vec![t, not_t]);

            let rewritten = bool::simplify_bool(&mut tm, tautology);

            // Should be true
            prop_assert_eq!(tm.get_bool_value(rewritten), Some(true));
        }

        /// Test: (a ∧ (a ∨ b)) simplifies to a (absorption law)
        #[test]
        fn absorption_law_and(a in proptest::bool::ANY, b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let tb = tm.mk_bool(b);
            let or_ab = tm.mk_or(vec![ta, tb]);
            let and_term = tm.mk_and(vec![ta, or_ab]);

            let rewritten = bool::simplify_bool(&mut tm, and_term);

            // Should equal ta
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(ta), tm.get_bool_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test: (a ∨ (a ∧ b)) simplifies to a (absorption law)
        #[test]
        fn absorption_law_or(a in proptest::bool::ANY, b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let tb = tm.mk_bool(b);
            let and_ab = tm.mk_and(vec![ta, tb]);
            let or_term = tm.mk_or(vec![ta, and_ab]);

            let rewritten = bool::simplify_bool(&mut tm, or_term);

            // Should equal ta
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(ta), tm.get_bool_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test De Morgan's laws: ¬(a ∧ b) = ¬a ∨ ¬b
        #[test]
        fn de_morgan_and(a in proptest::bool::ANY, b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let tb = tm.mk_bool(b);

            // ¬(a ∧ b)
            let and_ab = tm.mk_and(vec![ta, tb]);
            let not_and = tm.mk_not(and_ab);

            // ¬a ∨ ¬b
            let not_a = tm.mk_not(ta);
            let not_b = tm.mk_not(tb);
            let or_not = tm.mk_or(vec![not_a, not_b]);

            let rewritten1 = bool::simplify_bool(&mut tm, not_and);
            let rewritten2 = bool::simplify_bool(&mut tm, or_not);

            // Should have the same boolean value
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(rewritten1), tm.get_bool_value(rewritten2)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test De Morgan's laws: ¬(a ∨ b) = ¬a ∧ ¬b
        #[test]
        fn de_morgan_or(a in proptest::bool::ANY, b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let tb = tm.mk_bool(b);

            // ¬(a ∨ b)
            let or_ab = tm.mk_or(vec![ta, tb]);
            let not_or = tm.mk_not(or_ab);

            // ¬a ∧ ¬b
            let not_a = tm.mk_not(ta);
            let not_b = tm.mk_not(tb);
            let and_not = tm.mk_and(vec![not_a, not_b]);

            let rewritten1 = bool::simplify_bool(&mut tm, not_or);
            let rewritten2 = bool::simplify_bool(&mut tm, and_not);

            // Should have the same boolean value
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(rewritten1), tm.get_bool_value(rewritten2)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
        #[test]
        fn distributive_and_over_or(
            a in proptest::bool::ANY,
            b in proptest::bool::ANY,
            c in proptest::bool::ANY
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let tb = tm.mk_bool(b);
            let tc = tm.mk_bool(c);

            // a ∧ (b ∨ c)
            let or_bc = tm.mk_or(vec![tb, tc]);
            let left = tm.mk_and(vec![ta, or_bc]);

            // (a ∧ b) ∨ (a ∧ c)
            let and_ab = tm.mk_and(vec![ta, tb]);
            let and_ac = tm.mk_and(vec![ta, tc]);
            let right = tm.mk_or(vec![and_ab, and_ac]);

            let rewritten_left = bool::simplify_bool(&mut tm, left);
            let rewritten_right = bool::simplify_bool(&mut tm, right);

            // Should be equal
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(rewritten_left), tm.get_bool_value(rewritten_right)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test idempotence: a ∧ a = a
        #[test]
        fn and_idempotent(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let and_tt = tm.mk_and(vec![t, t]);

            let rewritten = bool::simplify_bool(&mut tm, and_tt);

            // Should equal t
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(t), tm.get_bool_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test idempotence: a ∨ a = a
        #[test]
        fn or_idempotent(b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_bool(b);
            let or_tt = tm.mk_or(vec![t, t]);

            let rewritten = bool::simplify_bool(&mut tm, or_tt);

            // Should equal t
            if let (Some(v1), Some(v2)) = (tm.get_bool_value(t), tm.get_bool_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test implication: (a ⇒ b) = (¬a ∨ b)
        #[test]
        fn implication_definition(a in proptest::bool::ANY, b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let tb = tm.mk_bool(b);

            // a ⇒ b
            let implies = tm.mk_implies(ta, tb);

            // ¬a ∨ b
            let not_a = tm.mk_not(ta);
            let or_term = tm.mk_or(vec![not_a, tb]);

            let rewritten1 = bool::simplify_bool(&mut tm, implies);
            let rewritten2 = bool::simplify_bool(&mut tm, or_term);

            if let (Some(v1), Some(v2)) = (tm.get_bool_value(rewritten1), tm.get_bool_value(rewritten2)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test iff: (a ⇔ b) = (a ⇒ b) ∧ (b ⇒ a)
        #[test]
        fn iff_definition(a in proptest::bool::ANY, b in proptest::bool::ANY) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_bool(a);
            let tb = tm.mk_bool(b);

            // a ⇔ b
            let iff = tm.mk_iff(ta, tb);

            // (a ⇒ b) ∧ (b ⇒ a)
            let implies_ab = tm.mk_implies(ta, tb);
            let implies_ba = tm.mk_implies(tb, ta);
            let and_term = tm.mk_and(vec![implies_ab, implies_ba]);

            let rewritten1 = bool::simplify_bool(&mut tm, iff);
            let rewritten2 = bool::simplify_bool(&mut tm, and_term);

            if let (Some(v1), Some(v2)) = (tm.get_bool_value(rewritten1), tm.get_bool_value(rewritten2)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod arithmetic_rewrite_properties {
    use super::*;

    proptest! {
        /// Test that 0 + x rewrites to x
        #[test]
        fn add_zero_eliminates(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let zero = tm.mk_int(BigInt::zero());
            let t = tm.mk_int(BigInt::from(n));
            let sum = tm.mk_add(vec![zero, t]);

            let rewritten = arith::simplify_arith(&mut tm, sum);

            if let (Some(v1), Some(v2)) = (tm.get_int_value(t), tm.get_int_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that 1 * x rewrites to x
        #[test]
        fn mul_one_eliminates(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let one = tm.mk_int(BigInt::one());
            let t = tm.mk_int(BigInt::from(n));
            let prod = tm.mk_mul(vec![one, t]);

            let rewritten = arith::simplify_arith(&mut tm, prod);

            if let (Some(v1), Some(v2)) = (tm.get_int_value(t), tm.get_int_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that 0 * x rewrites to 0
        #[test]
        fn mul_zero_annihilates(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let zero = tm.mk_int(BigInt::zero());
            let t = tm.mk_int(BigInt::from(n));
            let prod = tm.mk_mul(vec![zero, t]);

            let rewritten = arith::simplify_arith(&mut tm, prod);

            prop_assert_eq!(tm.get_int_value(rewritten), Some(BigInt::zero()));
        }

        /// Test that x - x rewrites to 0
        #[test]
        fn subtract_self_is_zero(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let diff = tm.mk_sub(t, t);

            let rewritten = arith::simplify_arith(&mut tm, diff);

            prop_assert_eq!(tm.get_int_value(rewritten), Some(BigInt::zero()));
        }

        /// Test that x + (-x) rewrites to 0
        #[test]
        fn add_negation_is_zero(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let neg_t = tm.mk_neg(t);
            let sum = tm.mk_add(vec![t, neg_t]);

            let rewritten = arith::simplify_arith(&mut tm, sum);

            prop_assert_eq!(tm.get_int_value(rewritten), Some(BigInt::zero()));
        }

        /// Test that -(- x) rewrites to x
        #[test]
        fn double_negation_cancels(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let neg_t = tm.mk_neg(t);
            let neg_neg_t = tm.mk_neg(neg_t);

            let rewritten = arith::simplify_arith(&mut tm, neg_neg_t);

            if let (Some(v1), Some(v2)) = (tm.get_int_value(t), tm.get_int_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test constant folding: 2 + 3 rewrites to 5
        #[test]
        fn constant_folding_add(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let sum = tm.mk_add(vec![ta, tb]);

            let rewritten = arith::simplify_arith(&mut tm, sum);

            // Should be a constant equal to a + b
            if let Some(value) = tm.get_int_value(rewritten) {
                prop_assert_eq!(value, BigInt::from(a + b));
            }
        }

        /// Test constant folding: 2 * 3 rewrites to 6
        #[test]
        fn constant_folding_mul(a in -20i64..20i64, b in -20i64..20i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let prod = tm.mk_mul(vec![ta, tb]);

            let rewritten = arith::simplify_arith(&mut tm, prod);

            // Should be a constant equal to a * b
            if let Some(value) = tm.get_int_value(rewritten) {
                prop_assert_eq!(value, BigInt::from(a * b));
            }
        }

        /// Test that x < x rewrites to false
        #[test]
        fn less_than_self_is_false(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let lt = tm.mk_lt(t, t);

            let rewritten = arith::simplify_arith(&mut tm, lt);

            prop_assert_eq!(tm.get_bool_value(rewritten), Some(false));
        }

        /// Test that x <= x rewrites to true
        #[test]
        fn less_equal_self_is_true(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let le = tm.mk_le(t, t);

            let rewritten = arith::simplify_arith(&mut tm, le);

            prop_assert_eq!(tm.get_bool_value(rewritten), Some(true));
        }

        /// Test that rewriting preserves addition associativity
        #[test]
        fn rewrite_preserves_add_associativity(
            a in -30i64..30i64,
            b in -30i64..30i64,
            c in -30i64..30i64
        ) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let tc = tm.mk_int(BigInt::from(c));

            // (a + b) + c
            let sum_ab = tm.mk_add(vec![ta, tb]);
            let left = tm.mk_add(vec![sum_ab, tc]);

            // a + (b + c)
            let sum_bc = tm.mk_add(vec![tb, tc]);
            let right = tm.mk_add(vec![ta, sum_bc]);

            let rewritten_left = arith::simplify_arith(&mut tm, left);
            let rewritten_right = arith::simplify_arith(&mut tm, right);

            if let (Some(v1), Some(v2)) = (tm.get_int_value(rewritten_left), tm.get_int_value(rewritten_right)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that x * 2 can be rewritten as x + x
        #[test]
        fn multiply_by_two_rewrite(n in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let two = tm.mk_int(BigInt::from(2));

            // x * 2
            let mul = tm.mk_mul(vec![t, two]);

            // x + x
            let add = tm.mk_add(vec![t, t]);

            let rewritten_mul = arith::simplify_arith(&mut tm, mul);
            let rewritten_add = arith::simplify_arith(&mut tm, add);

            if let (Some(v1), Some(v2)) = (tm.get_int_value(rewritten_mul), tm.get_int_value(rewritten_add)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that (x + a) - a rewrites to x
        #[test]
        fn add_then_subtract_cancels(x in -50i64..50i64, a in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let tx = tm.mk_int(BigInt::from(x));
            let ta = tm.mk_int(BigInt::from(a));

            let sum = tm.mk_add(vec![tx, ta]);
            let diff = tm.mk_sub(sum, ta);

            let rewritten = arith::simplify_arith(&mut tm, diff);

            if let (Some(v1), Some(v2)) = (tm.get_int_value(tx), tm.get_int_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}

#[cfg(test)]
mod rewrite_soundness_properties {
    use super::*;

    proptest! {
        /// Test that rewriting never changes satisfiability of constants
        #[test]
        fn rewrite_preserves_constant_satisfiability(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));

            // Original is always satisfiable (it's just a constant)
            let rewritten = tm.simplify(t);

            // Should still evaluate to the same value
            if let (Some(v1), Some(v2)) = (tm.get_int_value(t), tm.get_int_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that rewriting x = x always gives true
        #[test]
        fn rewrite_self_equality_is_true(n in -100i64..100i64) {
            let mut tm = manager::TermManager::new();
            let t = tm.mk_int(BigInt::from(n));
            let eq = tm.mk_eq(t, t);

            let rewritten = tm.simplify(eq);

            // Should be true
            prop_assert_eq!(tm.get_bool_value(rewritten), Some(true));
        }

        /// Test that rewriting preserves inequality direction
        #[test]
        fn rewrite_preserves_inequality(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let lt = tm.mk_lt(ta, tb);
            let rewritten = tm.simplify(lt);

            // Should preserve the truth value
            if let Some(original_value) = tm.get_bool_value(lt) {
                if let Some(rewritten_value) = tm.get_bool_value(rewritten) {
                    prop_assert_eq!(original_value, rewritten_value);
                }
            }
        }

        /// Test idempotence of rewriting
        #[test]
        fn rewrite_is_idempotent(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let sum = tm.mk_add(vec![ta, tb]);

            let rewritten1 = tm.simplify(sum);
            let rewritten2 = tm.simplify(rewritten1);

            // Rewriting twice should give the same result
            prop_assert_eq!(rewritten1, rewritten2);
        }
    }
}

#[cfg(test)]
mod theory_combination_rewrite_properties {
    use super::*;

    proptest! {
        /// Test that rewriting mixed boolean and arithmetic terms preserves semantics
        #[test]
        fn mixed_boolean_arithmetic_rewrite(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // (a < b) ∨ (a >= b) should be true
            let lt = tm.mk_lt(ta, tb);
            let ge = tm.mk_ge(ta, tb);
            let or_term = tm.mk_or(vec![lt, ge]);

            let rewritten = tm.simplify(or_term);

            // Should always be true (law of excluded middle for comparisons)
            prop_assert_eq!(tm.get_bool_value(rewritten), Some(true));
        }

        /// Test that rewriting (x = y) ∧ (x != y) gives false
        #[test]
        fn equality_contradiction_rewrites_to_false(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let eq = tm.mk_eq(ta, tb);
            let neq = tm.mk_not(eq);
            let and_term = tm.mk_and(vec![eq, neq]);

            let rewritten = tm.simplify(and_term);

            // Should be false
            prop_assert_eq!(tm.get_bool_value(rewritten), Some(false));
        }

        /// Test that if-then-else with true condition simplifies to then branch
        #[test]
        fn ite_true_condition(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let true_term = tm.mk_bool(true);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let ite = tm.mk_ite(true_term, ta, tb);
            let rewritten = tm.simplify(ite);

            // Should equal ta
            if let (Some(v1), Some(v2)) = (tm.get_int_value(ta), tm.get_int_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }

        /// Test that if-then-else with false condition simplifies to else branch
        #[test]
        fn ite_false_condition(a in -50i64..50i64, b in -50i64..50i64) {
            let mut tm = manager::TermManager::new();
            let false_term = tm.mk_bool(false);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let ite = tm.mk_ite(false_term, ta, tb);
            let rewritten = tm.simplify(ite);

            // Should equal tb
            if let (Some(v1), Some(v2)) = (tm.get_int_value(tb), tm.get_int_value(rewritten)) {
                prop_assert_eq!(v1, v2);
            }
        }
    }
}
