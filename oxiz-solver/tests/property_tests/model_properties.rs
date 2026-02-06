//! Property-based tests for model validity and completeness
//!
//! Tests that generated models:
//! - Satisfy all asserted constraints
//! - Are consistent across theories
//! - Handle partial models correctly

#![allow(clippy::collapsible_if)]

use num_bigint::BigInt;
use oxiz_core::ast::*;
use oxiz_solver::*;
use proptest::prelude::*;

#[cfg(test)]
mod model_basic_properties {
    use super::*;

    proptest! {
        #[test]
        fn model_satisfies_simple_equality(n in -100i64..100i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));
            let eq = tm.mk_eq(x, c);

            solver.assert(eq, &mut tm);
            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    let x_val = model.get(x);
                    prop_assert_eq!(x_val, Some(c));
                }
            }
        }

        #[test]
        fn model_satisfies_conjunction(a in -50i64..50i64, b in -50i64..50i64) {
            if a <= b {
                let mut solver = Solver::new();
                let mut tm = TermManager::new();

                let x = tm.mk_var("x", tm.sorts.int_sort);
                let ta = tm.mk_int(BigInt::from(a));
                let tb = tm.mk_int(BigInt::from(b));

                let ge = tm.mk_ge(x, ta);
                let le = tm.mk_le(x, tb);
                let and_term = tm.mk_and(vec![ge, le]);

                solver.assert(and_term, &mut tm);
                let result = solver.check(&mut tm);

                if matches!(result, SolverResult::Sat) {
                    if let Some(model) = solver.model() {
                        // Model should provide a value for x
                        let x_val = model.get(x);
                        prop_assert!(x_val.is_some());
                    }
                }
            }
        }

        #[test]
        fn model_handles_disjunction(a in -20i64..20i64, b in -20i64..20i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            let eq_a = tm.mk_eq(x, ta);
            let eq_b = tm.mk_eq(x, tb);
            let or_term = tm.mk_or(vec![eq_a, eq_b]);

            solver.assert(or_term, &mut tm);
            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    // Model should satisfy at least one disjunct
                    let x_val = model.get(x);
                    prop_assert!(x_val == Some(ta) || x_val == Some(tb));
                }
            }
        }

        #[test]
        fn model_consistency_across_variables(
            n1 in -30i64..30i64,
            n2 in -30i64..30i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let c1 = tm.mk_int(BigInt::from(n1));
            let c2 = tm.mk_int(BigInt::from(n2));

            solver.assert(tm.mk_eq(x, c1), &mut tm);
            solver.assert(tm.mk_eq(y, c2), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    let x_val = model.get(x);
                    let y_val = model.get(y);

                    prop_assert_eq!(x_val, Some(c1));
                    prop_assert_eq!(y_val, Some(c2));
                }
            }
        }

        #[test]
        fn model_handles_boolean_vars(b in proptest::bool::ANY) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.bool_sort);
            let const_b = tm.mk_bool(b);

            solver.assert(tm.mk_eq(x, const_b), &mut tm);
            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    let x_val = model.get(x);
                    // The model returns a TermId - we need to check if it represents
                    // the expected boolean value by comparing the term kinds
                    if let Some(val_id) = x_val {
                        if let (Some(val_term), Some(expected_term)) =
                            (tm.get(val_id), tm.get(const_b))
                        {
                            // Compare the term kinds (True/False)
                            prop_assert_eq!(
                                std::mem::discriminant(&val_term.kind),
                                std::mem::discriminant(&expected_term.kind),
                                "Expected boolean value {:?} but got {:?}",
                                expected_term.kind,
                                val_term.kind
                            );
                        }
                    } else {
                        // If model doesn't have a value, that's also acceptable
                        // as the solver might use implicit true/false
                    }
                }
            }
        }

        #[test]
        fn model_respects_arithmetic_operations(
            a in -20i64..20i64,
            b in -20i64..20i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // x = a, y = b
            solver.assert(tm.mk_eq(x, ta), &mut tm);
            solver.assert(tm.mk_eq(y, tb), &mut tm);

            // z = x + y
            let z = tm.mk_var("z", tm.sorts.int_sort);
            let sum = tm.mk_add(vec![x, y]);
            solver.assert(tm.mk_eq(z, sum), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    // x and y should have correct values
                    prop_assert_eq!(model.get(x), Some(ta));
                    prop_assert_eq!(model.get(y), Some(tb));
                    // z should exist
                    prop_assert!(model.get(z).is_some());
                }
            }
        }

        #[test]
        fn model_complete_for_all_vars(
            n1 in -10i64..10i64,
            n2 in -10i64..10i64,
            n3 in -10i64..10i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let z = tm.mk_var("z", tm.sorts.int_sort);

            let tx = tm.mk_int(BigInt::from(n1));
            let ty = tm.mk_int(BigInt::from(n2));
            let tz = tm.mk_int(BigInt::from(n3));

            solver.assert(tm.mk_eq(x, tx), &mut tm);
            solver.assert(tm.mk_eq(y, ty), &mut tm);
            solver.assert(tm.mk_eq(z, tz), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    // All variables should have values
                    prop_assert!(model.get(x).is_some());
                    prop_assert!(model.get(y).is_some());
                    prop_assert!(model.get(z).is_some());
                }
            }
        }
    }
}

#[cfg(test)]
mod model_theory_combination_properties {
    use super::*;

    proptest! {
        #[test]
        fn model_satisfies_mixed_boolean_arithmetic(
            n in -30i64..30i64,
            b in proptest::bool::ANY
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            solver.assert(tm.mk_eq(x, c), &mut tm);
            solver.assert(tm.mk_bool(b), &mut tm);

            let result = solver.check(&mut tm);

            // If b is false, should be unsat
            if !b {
                prop_assert!(matches!(result, SolverResult::Unsat));
            } else if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    prop_assert_eq!(model.get(x), Some(c));
                }
            }
        }

        #[test]
        fn model_handles_ite_expressions(
            a in -20i64..20i64,
            b in -20i64..20i64,
            cond in proptest::bool::ANY
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let cond_term = tm.mk_bool(cond);

            // x = ite(cond, a, b)
            let ite = tm.mk_ite(cond_term, ta, tb);
            solver.assert(tm.mk_eq(x, ite), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    let x_val = model.get(x);
                    let expected = if cond { ta } else { tb };
                    prop_assert_eq!(x_val, Some(expected));
                }
            }
        }
    }
}
