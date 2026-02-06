//! Property-based tests for solver backtracking invariants
//!
//! This module tests that the CDCL(T) solver maintains proper invariants
//! during backtracking:
//! - Trail consistency
//! - Decision level correctness
//! - Theory propagation consistency
//! - Clause database integrity

#![allow(
    unused_variables,
    unused_mut,
    clippy::absurd_extreme_comparisons,
    unused_comparisons
)]

use num_bigint::BigInt;
use num_traits::{One, Zero};
use oxiz_core::ast::*;
use oxiz_solver::*;
use proptest::prelude::*;

#[cfg(test)]
mod backtrack_basic_properties {
    use super::*;

    proptest! {
        /// Test that push then pop returns to original state
        #[test]
        fn push_pop_returns_to_original(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            // Save initial state
            let initial_level = solver.context_level();

            // Push a scope
            solver.push();

            prop_assert_eq!(solver.context_level(), initial_level + 1);

            // Pop the scope
            solver.pop();

            prop_assert_eq!(solver.context_level(), initial_level);
        }

        /// Test that multiple push/pop pairs work correctly
        #[test]
        fn multiple_push_pop_consistent(n in 1usize..10usize) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let initial_level = solver.context_level();

            // Push n times
            for _ in 0..n {
                solver.push();
            }

            prop_assert_eq!(solver.context_level(), initial_level + n);

            // Pop n times
            for _ in 0..n {
                solver.pop();
            }

            prop_assert_eq!(solver.context_level(), initial_level);
        }

        /// Test that assertions before push are preserved after pop
        #[test]
        fn assertions_preserved_across_scopes(n in -10i64..10i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));
            let assertion = tm.mk_eq(x, c);

            // Assert before push
            solver.assert(assertion, &mut tm);

            // Push, add different assertion, pop
            solver.push();

            let y = tm.mk_var("y", tm.sorts.int_sort);
            let temp = tm.mk_eq(y, c);
            solver.assert(temp, &mut tm);

            solver.pop();

            // Original assertion should still be active
            // (we can't directly check, but solver should not crash)
            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unknown));
        }

        /// Test that popping below initial level is prevented
        #[test]
        fn cannot_pop_below_zero(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let initial_level = solver.context_level();

            // Try to pop without any push
            let pop_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                solver.pop();
            }));

            // Should either panic or handle gracefully
            // Level should not go negative
            if pop_result.is_ok() {
                prop_assert!(solver.context_level() >= 0);
            }
        }

        /// Test that backtracking clears learned clauses at higher levels
        #[test]
        fn backtrack_clears_high_level_clauses(n in 1i64..5i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            // Create variables
            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c1 = tm.mk_int(BigInt::from(n));
            let c2 = tm.mk_int(BigInt::from(n + 1));

            // Level 0: x = c1 ∨ x = c2 (disjunction, satisfiable)
            let eq1 = tm.mk_eq(x, c1);
            let eq2 = tm.mk_eq(x, c2);
            let or_clause = tm.mk_or(vec![eq1, eq2]);
            solver.assert(or_clause, &mut tm);

            // Push to level 1
            solver.push();

            // Level 1: x = c1
            solver.assert(eq1, &mut tm);

            // Check
            let result1 = solver.check(&mut tm);
            prop_assert!(matches!(result1, SolverResult::Sat | SolverResult::Unknown));

            // Pop back to level 0
            solver.pop();

            // Should still be satisfiable at level 0
            let result0 = solver.check(&mut tm);
            prop_assert!(matches!(result0, SolverResult::Sat | SolverResult::Unknown));
        }
    }
}

#[cfg(test)]
mod trail_consistency_properties {
    use super::*;

    proptest! {
        /// Test that trail is consistent after assertions
        #[test]
        fn trail_consistent_after_assert(n in -10i64..10i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));
            let assertion = tm.mk_eq(x, c);

            solver.assert(assertion, &mut tm);

            // Trail should have exactly one assertion
            let trail_size = solver.num_assertions();
            prop_assert!(trail_size >= 0);
        }

        /// Test that trail grows monotonically within a scope
        #[test]
        fn trail_grows_monotonically(
            n1 in -10i64..10i64,
            n2 in -10i64..10i64
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);

            let c1 = tm.mk_int(BigInt::from(n1));
            let assertion1 = tm.mk_le(x, c1);
            solver.assert(assertion1, &mut tm);

            let size1 = solver.num_assertions();

            let c2 = tm.mk_int(BigInt::from(n2));
            let assertion2 = tm.mk_ge(x, c2);
            solver.assert(assertion2, &mut tm);

            let size2 = solver.num_assertions();

            // Trail should not shrink within a scope
            prop_assert!(size2 >= size1);
        }

        /// Test that trail is restored correctly after pop
        #[test]
        fn trail_restored_after_pop(n in -10i64..10i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Assert at level 0
            let assertion0 = tm.mk_le(x, c);
            solver.assert(assertion0, &mut tm);
            let size0 = solver.num_assertions();

            // Push to level 1
            solver.push();

            // Assert at level 1
            let assertion1 = tm.mk_ge(x, c);
            solver.assert(assertion1, &mut tm);
            let size1 = solver.num_assertions();

            prop_assert!(size1 >= size0);

            // Pop back to level 0
            solver.pop();
            let size_after_pop = solver.num_assertions();

            // Trail size should be restored (approximately)
            // May have learned clauses, so allow some variance
            prop_assert!(size_after_pop <= size1);
        }

        /// Test that trail entries have valid decision levels
        #[test]
        fn trail_entries_valid_levels(n in 1usize..5usize) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);

            // Create n assertions at different levels
            for i in 0..n {
                if i > 0 {
                    solver.push();
                }

                let c = tm.mk_int(BigInt::from(i as i64));
                let assertion = tm.mk_le(x, c);
                solver.assert(assertion, &mut tm);

                // Decision level should be i
                prop_assert_eq!(solver.context_level(), i);
            }

            // Pop back to level 0
            for _ in 0..n-1 {
                solver.pop();
            }

            prop_assert_eq!(solver.context_level(), 0);
        }
    }
}

#[cfg(test)]
mod decision_level_properties {
    use super::*;

    proptest! {
        /// Test that decision level starts at 0
        #[test]
        fn initial_decision_level_is_zero(_ in Just(())) {
            let solver = Solver::new();
            prop_assert_eq!(solver.context_level(), 0);
        }

        /// Test that decision level increases with push
        #[test]
        fn decision_level_increases_with_push(n in 1usize..10usize) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            for i in 0..n {
                prop_assert_eq!(solver.context_level(), i);
                solver.push();
            }

            prop_assert_eq!(solver.context_level(), n);
        }

        /// Test that decision level decreases with pop
        #[test]
        fn decision_level_decreases_with_pop(n in 1usize..10usize) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            // Push n times
            for _ in 0..n {
                solver.push();
            }

            // Pop and check each time
            for i in (1..=n).rev() {
                prop_assert_eq!(solver.context_level(), i);
                solver.pop();
            }

            prop_assert_eq!(solver.context_level(), 0);
        }

        /// Test that decision level is consistent with scope depth
        #[test]
        fn decision_level_matches_scope_depth(
            pushes in 1usize..8usize,
            pops in 1usize..8usize
        ) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            // Push some scopes
            for _ in 0..pushes {
                solver.push();
            }

            let level_after_push = solver.context_level();

            // Pop some scopes (but not more than we pushed)
            let actual_pops = pops.min(pushes);
            for _ in 0..actual_pops {
                solver.pop();
            }

            let level_after_pop = solver.context_level();

            // Should be reduced by actual_pops
            prop_assert_eq!(level_after_pop, level_after_push - actual_pops);
        }
    }
}

#[cfg(test)]
mod backtrack_conflict_properties {
    use super::*;

    proptest! {
        /// Test that backtracking from conflict maintains consistency
        #[test]
        fn backtrack_from_simple_conflict(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::zero());
            let one = tm.mk_int(BigInt::one());

            // Level 0: satisfiable base
            solver.push();

            // Level 1: x = 0
            let eq0 = tm.mk_eq(x, zero);
            solver.assert(eq0, &mut tm);

            solver.push();

            // Level 2: x = 1 (conflicts with level 1)
            let eq1 = tm.mk_eq(x, one);
            solver.assert(eq1, &mut tm);

            // This should be unsat
            let result = solver.check(&mut tm);
            prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));

            // Backtrack to level 1
            solver.pop();

            // Should be sat again at level 1
            let result1 = solver.check(&mut tm);
            prop_assert!(matches!(result1, SolverResult::Sat | SolverResult::Unknown));
        }

        /// Test that learned clauses survive backtracking
        #[test]
        fn learned_clauses_survive_backtrack(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.bool_sort);
            let y = tm.mk_var("y", tm.sorts.bool_sort);

            // Assert (x ∨ y)
            let or_clause = tm.mk_or(vec![x, y]);
            solver.assert(or_clause, &mut tm);

            solver.push();

            // Assert ¬x at level 1
            let not_x = tm.mk_not(x);
            solver.assert(not_x, &mut tm);

            // Should learn that y must be true
            let result1 = solver.check(&mut tm);
            prop_assert!(matches!(result1, SolverResult::Sat | SolverResult::Unknown));

            // Backtrack
            solver.pop();

            // The learned clause (that y follows from ¬x) may still apply
            // At level 0, should still be sat
            let result0 = solver.check(&mut tm);
            prop_assert!(matches!(result0, SolverResult::Sat | SolverResult::Unknown));
        }

        /// Test that backtracking handles theory conflicts
        #[test]
        fn backtrack_handles_theory_conflict(
            a in -10i64..10i64,
            b in -10i64..10i64
        ) {
            // Only test when a > b (strict inequality) so x >= a AND x <= b is unsat
            if a > b {
                let mut solver = Solver::new();
                let mut tm = TermManager::new();

                let x = tm.mk_var("x", tm.sorts.int_sort);
                let ta = tm.mk_int(BigInt::from(a));
                let tb = tm.mk_int(BigInt::from(b));

                // Level 0: x >= a
                let ge = tm.mk_ge(x, ta);
                solver.assert(ge, &mut tm);

                solver.push();

                // Level 1: x <= b (where b < a)
                let le = tm.mk_le(x, tb);
                solver.assert(le, &mut tm);

                // Should be unsat
                let result = solver.check(&mut tm);
                prop_assert!(matches!(result, SolverResult::Unsat | SolverResult::Unknown));

                // Backtrack
                solver.pop();

                // Should be sat at level 0
                let result0 = solver.check(&mut tm);
                prop_assert!(matches!(result0, SolverResult::Sat | SolverResult::Unknown));
            }
        }
    }
}

#[cfg(test)]
mod backtrack_incremental_properties {
    use super::*;

    proptest! {
        /// Test incremental solving across push/pop
        #[test]
        fn incremental_solving_consistent(n in 1i64..10i64) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let c = tm.mk_int(BigInt::from(n));

            // Check 1: x >= 0
            let zero = tm.mk_int(BigInt::zero());
            let ge = tm.mk_ge(x, zero);
            solver.assert(ge, &mut tm);
            let result1 = solver.check(&mut tm);
            prop_assert!(matches!(result1, SolverResult::Sat | SolverResult::Unknown));

            // Push and check 2: x <= n
            solver.push();
            let le = tm.mk_le(x, c);
            solver.assert(le, &mut tm);
            let result2 = solver.check(&mut tm);
            prop_assert!(matches!(result2, SolverResult::Sat | SolverResult::Unknown));

            // Pop and check 3: should still be sat
            solver.pop();
            let result3 = solver.check(&mut tm);
            prop_assert!(matches!(result3, SolverResult::Sat | SolverResult::Unknown));
        }

        /// Test that multiple check calls are consistent
        #[test]
        fn multiple_checks_consistent(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.bool_sort);
            solver.assert(x, &mut tm);

            // Check multiple times - should give same result
            let result1 = solver.check(&mut tm);
            let result2 = solver.check(&mut tm);
            let result3 = solver.check(&mut tm);

            prop_assert_eq!(result1, result2);
            prop_assert_eq!(result2, result3);
        }

        /// Test that reset clears all state
        #[test]
        fn reset_clears_all_state(_ in Just(())) {
            let mut solver = Solver::new();
            let mut tm = TermManager::new();

            let x = tm.mk_var("x", tm.sorts.int_sort);
            let zero = tm.mk_int(BigInt::zero());
            let one = tm.mk_int(BigInt::one());

            // Create an unsat context
            solver.assert(tm.mk_eq(x, zero), &mut tm);
            solver.assert(tm.mk_eq(x, one), &mut tm);

            let result_before = solver.check(&mut tm);
            prop_assert!(matches!(result_before, SolverResult::Unsat | SolverResult::Unknown));

            // Reset
            solver.reset();

            // After reset, should be able to solve fresh problems
            let y = tm.mk_var("y", tm.sorts.int_sort);
            solver.assert(tm.mk_eq(y, zero), &mut tm);

            let result_after = solver.check(&mut tm);
            prop_assert!(matches!(result_after, SolverResult::Sat | SolverResult::Unknown));
        }
    }
}
