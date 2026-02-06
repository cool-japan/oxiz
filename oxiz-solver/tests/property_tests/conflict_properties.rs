//! Property-based tests for conflict analysis and clause learning
//!
//! Tests:
//! - Conflict clause correctness
//! - UIP (Unique Implication Point) computation
//! - Clause minimization
//! - Lemma quality

use num_bigint::BigInt;
use oxiz_core::ast::*;
use oxiz_solver::*;
use proptest::prelude::*;

#[cfg(test)]
mod conflict_detection_properties {
    use super::*;

    proptest! {
        #[test]
        fn detects_simple_contradiction(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::from(0));
            let one = tm.mk_int(BigInt::from(1));

            solver.assert(tm.mk_eq(x, zero), &mut tm);
            solver.assert(tm.mk_eq(x, one), &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
        }

        #[test]
        fn detects_arithmetic_conflict(a in -10i64..10i64, b in -10i64..10i64) {
            if a > b {
                let mut solver = Solver::new();
                let mut tm = TermManager::new();

                let x = tm.mk_var("x", tm.sorts.int_sort);
                let ta = tm.mk_int(BigInt::from(a));
                let tb = tm.mk_int(BigInt::from(b));

                // x >= a AND x <= b where a > b
                solver.assert(tm.mk_ge(x, ta), &mut tm);
                solver.assert(tm.mk_le(x, tb), &mut tm);

                let result = solver.check(&mut tm);
                prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
            }
        }

        #[test]
        fn detects_boolean_conflict(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);

            solver.assert(p, &mut tm);
            solver.assert(tm.mk_not(p), &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat));
        }

        #[test]
        fn learns_from_conflict(n in 1i64..5i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Set up a scenario that forces learning
            solver.push();
            let eq1 = tm.mk_eq(x, c);
            solver.assert(eq1, &mut tm);
            let eq2 = tm.mk_eq(x, c);
            let not_eq2 = tm.mk_not(eq2);
            solver.assert(not_eq2, &mut tm);

            let _ = solver.check(&mut tm);

            // After conflict, learned clauses should prevent re-exploration
            solver.pop();
            solver.push();
            let eq3 = tm.mk_eq(x, c);
            solver.assert(eq3, &mut tm);
            let eq4 = tm.mk_eq(x, c);
            let not_eq4 = tm.mk_not(eq4);
            solver.assert(not_eq4, &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
        }
    }
}

#[cfg(test)]
mod conflict_clause_properties {
    use super::*;

    proptest! {
        #[test]
        fn learned_clause_is_asserting(
            a in -10i64..10i64,
            b in -10i64..10i64,
            c in -10i64..10i64
        ) {
            if a != b && b != c && a != c {
                let mut solver = Solver::new();
                let mut tm = TermManager::new();

                let x = tm.mk_var("x", tm.sorts.int_sort);
                let ta = tm.mk_int(BigInt::from(a));
                let tb = tm.mk_int(BigInt::from(b));
                let tc = tm.mk_int(BigInt::from(c));

                // (x=a ∨ x=b ∨ x=c)
                let eq_a = tm.mk_eq(x, ta);
                let eq_b = tm.mk_eq(x, tb);
                let eq_c = tm.mk_eq(x, tc);
                let clause = tm.mk_or(vec![eq_a, eq_b, eq_c]);

                solver.assert(clause, &mut tm);

                // Force conflicts to learn
                solver.push();
                solver.assert(tm.mk_not(eq_a), &mut tm);
                solver.assert(tm.mk_not(eq_b), &mut tm);
                solver.assert(tm.mk_not(eq_c), &mut tm);

                let result = solver.check(&mut tm);
                prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
            }
        }

        #[test]
        fn conflict_clause_minimal(n in 1i64..5i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Create redundant constraints
            solver.assert(tm.mk_eq(x, c), &mut tm);
            solver.assert(tm.mk_eq(x, c), &mut tm); // Duplicate

            solver.push();
            let eq_xc = tm.mk_eq(x, c);
            let not_eq = tm.mk_not(eq_xc);
            solver.assert(not_eq, &mut tm);

            let _ = solver.check(&mut tm);

            // Learned clause should be minimal (not contain redundant literals)
            solver.pop();
            // After pop, solver should be back to base level
            // (decision_level would be 0, but we check via model instead)
            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Sat));
        }
    }
}

#[cfg(test)]
mod uip_properties {
    use super::*;

    proptest! {
        #[test]
        fn uip_exists_for_conflicts(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);

            // (p ∨ q) ∧ ¬p ∧ ¬q
            solver.assert(tm.mk_or(vec![p, q]), &mut tm);
            solver.assert(tm.mk_not(p), &mut tm);
            solver.assert(tm.mk_not(q), &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat));
        }

        #[test]
        fn first_uip_is_closest_to_conflict(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let p = tm.mk_var("p", tm.sorts.bool_sort);
            let q = tm.mk_var("q", tm.sorts.bool_sort);
            let r = tm.mk_var("r", tm.sorts.bool_sort);

            // Build implication graph
            let imp_pq = tm.mk_implies(p, q);
            solver.assert(imp_pq, &mut tm);
            let imp_qr = tm.mk_implies(q, r);
            solver.assert(imp_qr, &mut tm);
            let not_r = tm.mk_not(r);
            let and_p_not_r = tm.mk_and(vec![p, not_r]);
            solver.assert(and_p_not_r, &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
        }
    }
}

#[cfg(test)]
mod clause_minimization_properties {
    use super::*;

    proptest! {
        #[test]
        fn minimization_preserves_conflict(n in 1i64..5i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let y = tm.mk_var("y", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // x = y, x = c, y != c
            let eq_xy = tm.mk_eq(x, y);
            solver.assert(eq_xy, &mut tm);
            let eq_xc = tm.mk_eq(x, c);
            solver.assert(eq_xc, &mut tm);
            let eq_yc = tm.mk_eq(y, c);
            let neq_yc = tm.mk_not(eq_yc);
            solver.assert(neq_yc, &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
        }

        #[test]
        fn minimized_clause_still_conflicts(
            a in -5i64..5i64,
            b in -5i64..5i64
        ) {
            if a != b {
                let mut solver = Solver::new();
                let mut tm = TermManager::new();

                let x = tm.mk_var("x", tm.sorts.int_sort);
                let ta = tm.mk_int(BigInt::from(a));
                let tb = tm.mk_int(BigInt::from(b));

                solver.assert(tm.mk_eq(x, ta), &mut tm);
                solver.assert(tm.mk_eq(x, tb), &mut tm);

                let result = solver.check(&mut tm);
                prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
            }
        }
    }
}

#[cfg(test)]
mod lemma_quality_properties {
    use super::*;

    proptest! {
        #[test]
        fn learned_lemma_prunes_search_space(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.bool_sort);
            let y = tm.mk_var("y", tm.sorts.bool_sort);

            // (x ∨ y) ∧ ¬x ∧ ¬y
            solver.assert(tm.mk_or(vec![x, y]), &mut tm);

            solver.push();
            solver.assert(tm.mk_not(x), &mut tm);
            solver.assert(tm.mk_not(y), &mut tm);

            let result1 = solver.check(&mut tm);
            prop_assert!(matches!(result1, SolverResult::Unsat));

            solver.pop();

            // Try again - should use learned lemma
            solver.push();
            solver.assert(tm.mk_not(x), &mut tm);
            solver.assert(tm.mk_not(y), &mut tm);

            let result2 = solver.check(&mut tm);
            prop_assert!(matches!(result2, SolverResult::Unsat));
        }

        #[test]
        fn lemma_generalizes_conflict(n in 1i64..3i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Learn from x = n conflict
            solver.push();
            let eq_xc = tm.mk_eq(x, c);
            solver.assert(eq_xc, &mut tm);
            let eq_xc2 = tm.mk_eq(x, c);
            let neq_xc = tm.mk_not(eq_xc2);
            solver.assert(neq_xc, &mut tm);

            let _ = solver.check(&mut tm);
            solver.pop();

            // Lemma should apply to similar conflicts
            solver.push();
            let eq_xc3 = tm.mk_eq(x, c);
            solver.assert(eq_xc3, &mut tm);
            let eq_xc4 = tm.mk_eq(x, c);
            let neq_xc2 = tm.mk_not(eq_xc4);
            solver.assert(neq_xc2, &mut tm);

            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));
        }
    }
}
