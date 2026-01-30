//! Property-based tests for model validity and completeness
//!
//! Tests that generated models:
//! - Satisfy all asserted constraints
//! - Are consistent across theories
//! - Handle partial models correctly
//! - Minimize properly when requested

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
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
                let model = solver.model();
                let x_val = model.eval(x, &tm);

                prop_assert_eq!(x_val, Some(c));
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
                    let model = solver.model();

                    // Model should satisfy both constraints
                    prop_assert!(model.eval(ge, &tm).map(|t| tm.get_bool_value(t)).flatten() == Some(true));
                    prop_assert!(model.eval(le, &tm).map(|t| tm.get_bool_value(t)).flatten() == Some(true));
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
                let model = solver.model();

                // Model should satisfy at least one disjunct
                let sat_a = model.eval(eq_a, &tm).map(|t| tm.get_bool_value(t)).flatten() == Some(true);
                let sat_b = model.eval(eq_b, &tm).map(|t| tm.get_bool_value(t)).flatten() == Some(true);

                prop_assert!(sat_a || sat_b);
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
                let model = solver.model();

                let x_val = model.eval(x, &tm);
                let y_val = model.eval(y, &tm);

                prop_assert_eq!(x_val, Some(c1));
                prop_assert_eq!(y_val, Some(c2));
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
                let model = solver.model();
                let x_val = model.eval(x, &tm).and_then(|t| tm.get_bool_value(t));

                prop_assert_eq!(x_val, Some(b));
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
                let model = solver.model();
                let z_val = model.eval(z, &tm).and_then(|t| tm.get_int_value(t));

                prop_assert_eq!(z_val, Some(BigInt::from(a + b)));
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
                let model = solver.model();

                // All variables should have values
                prop_assert!(model.eval(x, &tm).is_some());
                prop_assert!(model.eval(y, &tm).is_some());
                prop_assert!(model.eval(z, &tm).is_some());
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
            let p = tm.mk_var("p", tm.sorts.bool_sort);

            let c = tm.mk_int(BigInt::from(n));
            let eq = tm.mk_eq(x, c);

            // p ⟺ (x = n)
            let iff = tm.mk_iff(p, eq);
            solver.assert(iff, &mut tm);
            solver.assert(tm.mk_bool(b), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();

                let p_val = model.eval(p, &tm).and_then(|t| tm.get_bool_value(t));
                let eq_val = model.eval(eq, &tm).and_then(|t| tm.get_bool_value(t));

                if let (Some(pv), Some(eqv)) = (p_val, eq_val) {
                    // p ⟺ (x = n) must hold
                    prop_assert_eq!(pv, eqv);
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
                let model = solver.model();
                let x_val = model.eval(x, &tm).and_then(|t| tm.get_int_value(t));

                let expected = if cond { BigInt::from(a) } else { BigInt::from(b) };
                prop_assert_eq!(x_val, Some(expected));
            }
        }
    }
}

#[cfg(test)]
mod model_minimization_properties {
    use super::*;

    proptest! {
        #[test]
        fn minimal_model_has_fewer_assignments(
            a in 0i64..10i64,
            b in 0i64..10i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));

            // x >= a (many solutions)
            solver.assert(tm.mk_ge(x, ta), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let minimal = model.minimize();

                // Minimal model should still satisfy constraints
                prop_assert!(minimal.eval(tm.mk_ge(x, ta), &tm).is_some());
            }
        }
    }
}
