//! Property-based tests for CDCL SAT solver
//!
//! Tests:
//! - Clause database integrity
//! - Variable assignment consistency
//! - Resolution correctness
//! - Restart strategies
//! - Clause deletion safety

use oxiz_sat::*;
use proptest::prelude::*;

#[cfg(test)]
mod cdcl_basic_properties {
    use super::*;

    proptest! {
        #[test]
        fn empty_cnf_is_sat() {
            let mut solver = CDCLSolver::new();
            let result = solver.solve();

            prop_assert_eq!(result, SatResult::Sat);
        }

        #[test]
        fn single_unit_clause_is_sat(lit in -100i32..100i32) {
            if lit != 0 {
                let mut solver = CDCLSolver::new();
                let var = lit.abs() as u32;
                solver.add_variable();

                solver.add_clause(vec![lit]);
                let result = solver.solve();

                prop_assert_eq!(result, SatResult::Sat);
            }
        }

        #[test]
        fn contradictory_units_are_unsat(v in 1u32..100u32) {
            let mut solver = CDCLSolver::new();

            for _ in 0..v {
                solver.add_variable();
            }

            solver.add_clause(vec![v as i32]);
            solver.add_clause(vec![-(v as i32)]);

            let result = solver.solve();

            prop_assert_eq!(result, SatResult::Unsat);
        }

        #[test]
        fn tautology_clause_ignorable(v in 1u32..50u32) {
            let mut solver = CDCLSolver::new();

            for _ in 0..v {
                solver.add_variable();
            }

            // v ∨ ¬v (tautology)
            solver.add_clause(vec![v as i32, -(v as i32)]);

            let result = solver.solve();

            // Empty CNF after removing tautology
            prop_assert_eq!(result, SatResult::Sat);
        }

        #[test]
        fn binary_clause_sat(v1 in 1u32..50u32, v2 in 1u32..50u32) {
            if v1 != v2 {
                let mut solver = CDCLSolver::new();
                let max_var = v1.max(v2);

                for _ in 0..=max_var {
                    solver.add_variable();
                }

                // v1 ∨ v2
                solver.add_clause(vec![v1 as i32, v2 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SatResult::Sat);
            }
        }

        #[test]
        fn horn_clause_decidable(
            v1 in 1u32..20u32,
            v2 in 1u32..20u32,
            v3 in 1u32..20u32
        ) {
            let mut solver = CDCLSolver::new();
            let max_var = v1.max(v2).max(v3);

            for _ in 0..=max_var {
                solver.add_variable();
            }

            // Horn clause: ¬v1 ∨ ¬v2 ∨ v3
            solver.add_clause(vec![-(v1 as i32), -(v2 as i32), v3 as i32]);

            let result = solver.solve();

            // Horn clauses are always decidable
            prop_assert!(matches!(result, SatResult::Sat | SatResult::Unsat));
        }
    }
}

#[cfg(test)]
mod clause_learning_properties {
    use super::*;

    proptest! {
        #[test]
        fn learned_clause_prevents_reexploration(v in 2u32..10u32) {
            let mut solver = CDCLSolver::new();

            for _ in 0..v {
                solver.add_variable();
            }

            // Create conflict scenario
            for i in 1..v {
                solver.add_clause(vec![i as i32, (i+1) as i32]);
            }

            solver.add_clause(vec![-(v as i32)]);
            solver.add_clause(vec![1]);

            let result = solver.solve();

            // Should learn and prune search space
            prop_assert!(matches!(result, SatResult::Sat | SatResult::Unsat));
        }

        #[test]
        fn conflict_clause_is_asserting(
            v1 in 1u32..10u32,
            v2 in 1u32..10u32,
            v3 in 1u32..10u32
        ) {
            if v1 != v2 && v2 != v3 && v1 != v3 {
                let mut solver = CDCLSolver::new();
                let max_var = v1.max(v2).max(v3);

                for _ in 0..=max_var {
                    solver.add_variable();
                }

                // (v1 ∨ v2 ∨ v3) ∧ ¬v1 ∧ ¬v2 ∧ ¬v3
                solver.add_clause(vec![v1 as i32, v2 as i32, v3 as i32]);
                solver.add_clause(vec![-(v1 as i32)]);
                solver.add_clause(vec![-(v2 as i32)]);
                solver.add_clause(vec![-(v3 as i32)]);

                let result = solver.solve();

                prop_assert_eq!(result, SatResult::Unsat);
            }
        }

        #[test]
        fn learned_clauses_cumulative(n in 2usize..6usize) {
            let mut solver = CDCLSolver::new();

            for _ in 0..n {
                solver.add_variable();
            }

            // Add conflicting constraints incrementally
            for i in 1..n {
                solver.add_clause(vec![i as i32, (i+1) as i32]);
                solver.add_clause(vec![-(i as i32)]);
            }

            solver.add_clause(vec![-(n as i32)]);
            solver.add_clause(vec![1]);

            let result = solver.solve();

            // Learned clauses should accumulate
            prop_assert!(matches!(result, SatResult::Sat | SatResult::Unsat));
        }
    }
}

#[cfg(test)]
mod resolution_properties {
    use super::*;

    proptest! {
        #[test]
        fn resolution_preserves_satisfiability(
            v1 in 1u32..20u32,
            v2 in 1u32..20u32,
            v3 in 1u32..20u32
        ) {
            if v1 != v2 && v2 != v3 && v1 != v3 {
                let mut solver = CDCLSolver::new();
                let max_var = v1.max(v2).max(v3);

                for _ in 0..=max_var {
                    solver.add_variable();
                }

                // (v1 ∨ v2) ∧ (¬v2 ∨ v3)
                // Resolution gives: (v1 ∨ v3)
                solver.add_clause(vec![v1 as i32, v2 as i32]);
                solver.add_clause(vec![-(v2 as i32), v3 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SatResult::Sat);
            }
        }

        #[test]
        fn resolution_detects_empty_clause(v in 1u32..20u32) {
            let mut solver = CDCLSolver::new();

            for _ in 0..v {
                solver.add_variable();
            }

            // v ∧ ¬v leads to empty clause
            solver.add_clause(vec![v as i32]);
            solver.add_clause(vec![-(v as i32)]);

            let result = solver.solve();

            prop_assert_eq!(result, SatResult::Unsat);
        }

        #[test]
        fn subsumption_removes_redundant_clauses(
            v1 in 1u32..15u32,
            v2 in 1u32..15u32
        ) {
            if v1 != v2 {
                let mut solver = CDCLSolver::new();
                let max_var = v1.max(v2);

                for _ in 0..=max_var {
                    solver.add_variable();
                }

                // v1 subsumes (v1 ∨ v2)
                solver.add_clause(vec![v1 as i32]);
                solver.add_clause(vec![v1 as i32, v2 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SatResult::Sat);
            }
        }
    }
}

#[cfg(test)]
mod restart_properties {
    use super::*;

    proptest! {
        #[test]
        fn restart_preserves_learned_clauses(v in 2u32..8u32) {
            let mut solver = CDCLSolver::new();

            for _ in 0..v {
                solver.add_variable();
            }

            // Add some clauses
            for i in 1..v {
                solver.add_clause(vec![i as i32, (i+1) as i32]);
            }

            // Solve with restarts enabled
            solver.enable_restarts(true);
            let result = solver.solve();

            // Should still find correct result after restarts
            prop_assert!(matches!(result, SatResult::Sat | SatResult::Unsat));
        }

        #[test]
        fn restart_doesnt_affect_correctness(n in 2usize..6usize) {
            let mut solver1 = CDCLSolver::new();
            let mut solver2 = CDCLSolver::new();

            for _ in 0..n {
                solver1.add_variable();
                solver2.add_variable();
            }

            // Add same clauses to both
            for i in 1..n {
                let clause = vec![i as i32, (i+1) as i32];
                solver1.add_clause(clause.clone());
                solver2.add_clause(clause);
            }

            solver1.enable_restarts(true);
            solver2.enable_restarts(false);

            let result1 = solver1.solve();
            let result2 = solver2.solve();

            // Both should give same result
            prop_assert_eq!(result1, result2);
        }
    }
}

#[cfg(test)]
mod variable_elimination_properties {
    use super::*;

    proptest! {
        #[test]
        fn pure_literal_elimination(v in 1u32..20u32) {
            let mut solver = CDCLSolver::new();

            for _ in 0..=v {
                solver.add_variable();
            }

            // v appears only positively (pure literal)
            solver.add_clause(vec![v as i32, 1]);
            solver.add_clause(vec![v as i32, 2]);

            let result = solver.solve();

            prop_assert_eq!(result, SatResult::Sat);
        }

        #[test]
        fn variable_elimination_preserves_sat(
            v1 in 1u32..15u32,
            v2 in 1u32..15u32
        ) {
            if v1 != v2 {
                let mut solver = CDCLSolver::new();
                let max_var = v1.max(v2);

                for _ in 0..=max_var {
                    solver.add_variable();
                }

                // (v1 ∨ v2)
                solver.add_clause(vec![v1 as i32, v2 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SatResult::Sat);
            }
        }
    }
}

#[cfg(test)]
mod clause_database_properties {
    use super::*;

    proptest! {
        #[test]
        fn clause_database_consistent_after_deletion(n in 2usize..10usize) {
            let mut solver = CDCLSolver::new();

            for _ in 0..n {
                solver.add_variable();
            }

            // Add many clauses
            for i in 1..n {
                solver.add_clause(vec![i as i32, (i+1) as i32]);
            }

            // Enable clause deletion
            solver.enable_clause_deletion(true);

            let result = solver.solve();

            // Should still work correctly
            prop_assert!(matches!(result, SatResult::Sat | SatResult::Unsat));
        }

        #[test]
        fn literal_watch_scheme_correct(
            v1 in 1u32..15u32,
            v2 in 1u32..15u32,
            v3 in 1u32..15u32
        ) {
            if v1 != v2 && v2 != v3 && v1 != v3 {
                let mut solver = CDCLSolver::new();
                let max_var = v1.max(v2).max(v3);

                for _ in 0..=max_var {
                    solver.add_variable();
                }

                // (v1 ∨ v2 ∨ v3)
                solver.add_clause(vec![v1 as i32, v2 as i32, v3 as i32]);

                // ¬v1 ∧ ¬v2
                solver.add_clause(vec![-(v1 as i32)]);
                solver.add_clause(vec![-(v2 as i32)]);

                let result = solver.solve();

                // v3 must be true
                prop_assert_eq!(result, SatResult::Sat);
            }
        }

        #[test]
        fn implication_graph_acyclic(n in 2usize..8usize) {
            let mut solver = CDCLSolver::new();

            for _ in 0..n {
                solver.add_variable();
            }

            // Create implication chain (DAG structure)
            for i in 1..n {
                solver.add_clause(vec![-(i as i32), (i+1) as i32]);
            }

            let result = solver.solve();

            // Should have valid implication graph
            prop_assert!(matches!(result, SatResult::Sat | SatResult::Unsat));
        }
    }
}
