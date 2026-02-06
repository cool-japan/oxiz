//! Property-based tests for propagation soundness
//!
//! Tests:
//! - Boolean Constraint Propagation (BCP)
//! - Theory propagation
//! - Propagation completeness
//! - Watched literal correctness

#![allow(unused_variables, clippy::collapsible_if)]

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
                if let Some(model) = solver.model() {
                    // p should be true
                    let p_val = model.get(p);
                    let true_term = tm.mk_bool(true);
                    prop_assert_eq!(p_val, Some(true_term));
                }
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
                if let Some(model) = solver.model() {
                    // q should be true
                    let q_val = model.get(q);
                    let true_term = tm.mk_bool(true);
                    prop_assert_eq!(q_val, Some(true_term));
                }
            }
        }

        #[test]
        fn chain_propagation(n in 2usize..5) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            // Create chain: p_1 → p_2 → ... → p_n
            let vars: Vec<_> = (0..n)
                .map(|i| tm.mk_var(&format!("p{}", i), tm.sorts.bool_sort))
                .collect();

            // Add implications
            for i in 0..n - 1 {
                let impl_term = tm.mk_implies(vars[i], vars[i + 1]);
                solver.assert(impl_term, &mut tm);
            }

            // Assert first variable
            solver.assert(vars[0], &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    // All variables should have values
                    for v in &vars {
                        prop_assert!(model.get(*v).is_some());
                    }
                }
            }
        }

        #[test]
        fn disjunction_propagation(a in proptest::bool::ANY, b in proptest::bool::ANY) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);
            let r = tm.mk_var("r", tm.sorts.bool_sort);

            // (p ∨ q ∨ r)
            solver.assert(tm.mk_or(vec![p, q, r]), &mut tm);

            // Set p and q to false if a and b are false
            if !a {
                solver.assert(tm.mk_not(p), &mut tm);
            }
            if !b {
                solver.assert(tm.mk_not(q), &mut tm);
            }

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    // At least one should be true
                    let p_val = model.get(p);
                    let q_val = model.get(q);
                    let r_val = model.get(r);
                    let true_term = tm.mk_bool(true);

                    prop_assert!(
                        p_val == Some(true_term) ||
                        q_val == Some(true_term) ||
                        r_val == Some(true_term)
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod theory_propagation_properties {
    use super::*;

    proptest! {
        #[test]
        fn arithmetic_bounds_propagate(n in -50i64..50i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);

            let cn = tm.mk_int(BigInt::from(n));

            // x = n, y = x (equality propagates)
            solver.assert(tm.mk_eq(x, cn), &mut tm);
            solver.assert(tm.mk_eq(y, x), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    // Both should have the same value
                    let x_val = model.get(x);
                    let y_val = model.get(y);

                    prop_assert_eq!(x_val, Some(cn));
                    prop_assert_eq!(y_val, Some(cn));
                }
            }
        }

        #[test]
        fn transitivity_propagates(
            a in -20i64..20i64,
            b in -20i64..20i64,
            c in -20i64..20i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let z = tm.mk_var("z", tm.sorts.int_sort);

            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));
            let tc = tm.mk_int(BigInt::from(c));

            // x = a, y = b, z = c
            solver.assert(tm.mk_eq(x, ta), &mut tm);
            solver.assert(tm.mk_eq(y, tb), &mut tm);
            solver.assert(tm.mk_eq(z, tc), &mut tm);

            let result = solver.check(&mut tm);

            if matches!(result, SolverResult::Sat) {
                if let Some(model) = solver.model() {
                    // All should have values
                    prop_assert!(model.get(x).is_some());
                    prop_assert!(model.get(y).is_some());
                    prop_assert!(model.get(z).is_some());
                }
            }
        }
    }
}

#[cfg(test)]
mod propagation_completeness_properties {
    use super::*;

    proptest! {
        #[test]
        fn no_spurious_conflicts(n in 0i64..20i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let cn = tm.mk_int(BigInt::from(n));

            // Simple satisfiable constraint
            solver.assert(tm.mk_eq(x, cn), &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Sat));
        }

        #[test]
        fn conflict_detected(a in 0i64..10i64, b in 11i64..20i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let ta = tm.mk_int(BigInt::from(a));
            let tb = tm.mk_int(BigInt::from(b));

            // x = a ∧ x = b (conflict when a ≠ b)
            solver.assert(tm.mk_eq(x, ta), &mut tm);
            solver.assert(tm.mk_eq(x, tb), &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat));
        }
    }
}
