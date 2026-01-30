//! Property-based tests for propagation soundness
//!
//! Tests:
//! - Boolean Constraint Propagation (BCP)
//! - Theory propagation
//! - Propagation completeness
//! - Watched literal correctness

use num_bigint::BigInt;
use oxiz_core::ast::*;
use oxiz_solver::*;
use proptest::prelude::*;

#[cfg(test)]
mod bcp_properties {
    use super::*;

    proptest! {
        #[test]
        fn unit_clause_propagates(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);

            // Unit clause: just p
            solver.assert(p, &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let p_val = model.eval(p, &tm).and_then(|t| tm.get_bool_value(t));

                prop_assert_eq!(p_val, Some(true));
            }
        }

        #[test]
        fn binary_clause_propagates(b in proptest::bool::ANY) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);

            // p ∨ q
            solver.assert(tm.mk_or(vec![p, q]), &mut tm);

            // ¬p
            solver.assert(tm.mk_not(p), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let q_val = model.eval(q, &tm).and_then(|t| tm.get_bool_value(t));

                // q must be true (propagated from ¬p)
                prop_assert_eq!(q_val, Some(true));
            }
        }

        #[test]
        fn ternary_clause_propagates(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);
            let r = tm.mk_var("r", tm.sorts.bool_sort);

            // p ∨ q ∨ r
            solver.assert(tm.mk_or(vec![p, q, r]), &mut tm);

            // ¬p ∧ ¬q
            solver.assert(tm.mk_not(p), &mut tm);
            solver.assert(tm.mk_not(q), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let r_val = model.eval(r, &tm).and_then(|t| tm.get_bool_value(t));

                // r must be true
                prop_assert_eq!(r_val, Some(true));
            }
        }

        #[test]
        fn propagation_is_transitive(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);
            let r = tm.mk_var("r", tm.sorts.bool_sort);

            // p → q
            solver.assert(tm.mk_implies(p, q), &mut tm);

            // q → r
            solver.assert(tm.mk_implies(q, r), &mut tm);

            // p
            solver.assert(p, &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let r_val = model.eval(r, &tm).and_then(|t| tm.get_bool_value(t));

                // r must be true (transitively)
                prop_assert_eq!(r_val, Some(true));
            }
        }

        #[test]
        fn propagation_detects_conflict_early(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);

            // p → q
            solver.assert(tm.mk_implies(p, q), &mut tm);

            // p ∧ ¬q (conflict)
            solver.assert(p, &mut tm);
            solver.assert(tm.mk_not(q), &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat));
        }
    }
}

#[cfg(test)]
mod theory_propagation_properties {
    use super::*;

    proptest! {
        #[test]
        fn equality_propagation(n in -10i64..10i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // x = y, x = c
            solver.assert(tm.mk_eq(x, y), &mut tm);
            solver.assert(tm.mk_eq(x, c), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let y_val = model.eval(y, &tm).and_then(|t| tm.get_int_value(t));

                // y should also be c (propagated)
                prop_assert_eq!(y_val, Some(BigInt::from(n)));
            }
        }

        #[test]
        fn inequality_propagation(a in -10i64..10i64, delta in 1i64..5i64) {
            let b = a + delta;

            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // x >= a, x < b
            solver.assert(tm.mk_ge(x, ta), &mut tm);
            solver.assert(tm.mk_lt(x, tb), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let x_val = model.eval(x, &tm).and_then(|t| tm.get_int_value(t));

                if let Some(val) = x_val {
                    // x must be in [a, b)
                    prop_assert!(val >= BigInt::from(a));
                    prop_assert!(val < BigInt::from(b));
                }
            }
        }

        #[test]
        fn transitivity_propagation(
            a in -10i64..10i64,
            b in -10i64..10i64,
            c in -10i64..10i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let z = tm.mk_var("z", tm.sorts.int_sort);

            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let tc = tm.mk_int(BigInt::from(c));

            // x = a, y = b, x = y
            solver.assert(tm.mk_eq(x, ta), &mut tm);
            solver.assert(tm.mk_eq(y, tb), &mut tm);
            solver.assert(tm.mk_eq(x, y), &mut tm);

            let result = solver.check(&mut tm);

            // Should be unsat if a != b
            if a != b {
                prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
            }
        }

        #[test]
        fn arithmetic_bounds_propagation(
            lower in -10i64..0i64,
            upper in 1i64..10i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let tl = tm.mk_int(BigInt::from(lower));
            let tu = tm.mk_int(BigInt::from(upper));

            // x + y = 0, x >= lower, x <= upper
            let sum = tm.mk_add(vec![x, y]);
            let zero = tm.mk_int(BigInt::from(0));
            solver.assert(tm.mk_eq(sum, zero), &mut tm);
            solver.assert(tm.mk_ge(x, tl), &mut tm);
            solver.assert(tm.mk_le(x, tu), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let x_val = model.eval(x, &tm).and_then(|t| tm.get_int_value(t));
                let y_val = model.eval(y, &tm).and_then(|t| tm.get_int_value(t));

                if let (Some(xv), Some(yv)) = (x_val, y_val) {
                    // y = -x (propagated)
                    prop_assert_eq!(xv + yv, BigInt::from(0));

                    // y should be in [-upper, -lower]
                    prop_assert!(yv >= BigInt::from(-upper));
                    prop_assert!(yv <= BigInt::from(-lower));
                }
            }
        }
    }
}

#[cfg(test)]
mod watched_literals_properties {
    use super::*;

    proptest! {
        #[test]
        fn watched_literals_maintain_invariant(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);
            let r = tm.mk_var("r", tm.sorts.bool_sort);

            // (p ∨ q ∨ r)
            solver.assert(tm.mk_or(vec![p, q, r]), &mut tm);

            // Setting p=false should trigger watch update
            solver.push();
            solver.assert(tm.mk_not(p), &mut tm);

            let result = solver.check(&mut tm);

            // Should still be satisfiable
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unknown));
        }

        #[test]
        fn watched_literals_efficient_propagation(n in 1usize..5usize) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let mut vars = Vec::new();
            for i in 0..n {
                let v = tm.mk_var(&format!("x{}", i), tm.sorts.bool_sort);
                vars.push(v);
            }

            // Large clause: x0 ∨ x1 ∨ ... ∨ xn
            solver.assert(tm.mk_or(vars.clone()), &mut tm);

            // Set all but last to false
            for i in 0..n-1 {
                solver.assert(tm.mk_not(vars[i]), &mut tm);
            }

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();
                let last_val = model.eval(vars[n-1], &tm).and_then(|t| tm.get_bool_value(t));

                // Last variable must be true
                prop_assert_eq!(last_val, Some(true));
            }
        }
    }
}

#[cfg(test)]
mod propagation_completeness_properties {
    use super::*;

    proptest! {
        #[test]
        fn all_unit_clauses_propagate(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);
            let r = tm.mk_var("r", tm.sorts.bool_sort);

            // Three unit clauses
            solver.assert(p, &mut tm);
            solver.assert(q, &mut tm);
            solver.assert(r, &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();

                // All should be true
                prop_assert_eq!(model.eval(p, &tm).and_then(|t| tm.get_bool_value(t)), Some(true));
                prop_assert_eq!(model.eval(q, &tm).and_then(|t| tm.get_bool_value(t)), Some(true));
                prop_assert_eq!(model.eval(r, &tm).and_then(|t| tm.get_bool_value(t)), Some(true));
            }
        }

        #[test]
        fn propagation_reaches_fixpoint(n in 1i64..5i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let z = tm.mk_var("z", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // x = y, y = z, z = c
            solver.assert(tm.mk_eq(x, y), &mut tm);
            solver.assert(tm.mk_eq(y, z), &mut tm);
            solver.assert(tm.mk_eq(z, c), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                let model = solver.model();

                // All should be equal to c
                let x_val = model.eval(x, &tm).and_then(|t| tm.get_int_value(t));
                let y_val = model.eval(y, &tm).and_then(|t| tm.get_int_value(t));
                let z_val = model.eval(z, &tm).and_then(|t| tm.get_int_value(t));

                prop_assert_eq!(x_val, Some(BigInt::from(n)));
                prop_assert_eq!(y_val, Some(BigInt::from(n)));
                prop_assert_eq!(z_val, Some(BigInt::from(n)));
            }
        }
    }
}
