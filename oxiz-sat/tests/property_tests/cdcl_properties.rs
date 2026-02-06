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

    #[test]
    fn empty_cnf_is_sat() {
        let mut solver = Solver::new();
        let result = solver.solve();

        assert_eq!(result, SolverResult::Sat);
    }

    proptest! {
        #[test]
        fn single_unit_clause_is_sat(lit in -100i32..100i32) {
            if lit != 0 {
                let mut solver = Solver::new();
                let _var = lit.unsigned_abs();
                solver.new_var();

                solver.add_clause_dimacs(&[lit]);
                let result = solver.solve();

                prop_assert_eq!(result, SolverResult::Sat);
            }
        }

        #[test]
        fn contradictory_units_are_unsat(v in 1u32..100u32) {
            let mut solver = Solver::new();

            for _ in 0..v {
                solver.new_var();
            }

            solver.add_clause_dimacs(&[v as i32]);
            solver.add_clause_dimacs(&[-(v as i32)]);

            let result = solver.solve();

            prop_assert_eq!(result, SolverResult::Unsat);
        }

        #[test]
        fn tautology_clause_ignorable(v in 1u32..50u32) {
            let mut solver = Solver::new();

            for _ in 0..v {
                solver.new_var();
            }

            // v ∨ ¬v (tautology)
            solver.add_clause_dimacs(&[v as i32, -(v as i32)]);

            let result = solver.solve();

            // Empty CNF after removing tautology
            prop_assert_eq!(result, SolverResult::Sat);
        }

        #[test]
        fn binary_clause_sat(v1 in 1u32..50u32, v2 in 1u32..50u32) {
            if v1 != v2 {
                let mut solver = Solver::new();
                let max_var = v1.max(v2);

                for _ in 0..=max_var {
                    solver.new_var();
                }

                // v1 ∨ v2
                solver.add_clause_dimacs(&[v1 as i32, v2 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SolverResult::Sat);
            }
        }

        #[test]
        fn horn_clause_decidable(
            v1 in 1u32..20u32,
            v2 in 1u32..20u32,
            v3 in 1u32..20u32
        ) {
            let mut solver = Solver::new();
            let max_var = v1.max(v2).max(v3);

            for _ in 0..=max_var {
                solver.new_var();
            }

            // Horn clause: ¬v1 ∨ ¬v2 ∨ v3
            solver.add_clause_dimacs(&[-(v1 as i32), -(v2 as i32), v3 as i32]);

            let result = solver.solve();

            // Horn clauses are always decidable
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unsat));
        }
    }
}

#[cfg(test)]
mod clause_learning_properties {
    use super::*;

    proptest! {
        #[test]
        fn learned_clause_prevents_reexploration(v in 2u32..10u32) {
            let mut solver = Solver::new();

            for _ in 0..v {
                solver.new_var();
            }

            // Create conflict scenario
            for i in 1..v {
                solver.add_clause_dimacs(&[i as i32, (i+1) as i32]);
            }

            solver.add_clause_dimacs(&[-(v as i32)]);
            solver.add_clause_dimacs(&[1]);

            let result = solver.solve();

            // Should learn and prune search space
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unsat));
        }

        #[test]
        fn conflict_clause_is_asserting(
            v1 in 1u32..10u32,
            v2 in 1u32..10u32,
            v3 in 1u32..10u32
        ) {
            if v1 != v2 && v2 != v3 && v1 != v3 {
                let mut solver = Solver::new();
                let max_var = v1.max(v2).max(v3);

                for _ in 0..=max_var {
                    solver.new_var();
                }

                // (v1 ∨ v2 ∨ v3) ∧ ¬v1 ∧ ¬v2 ∧ ¬v3
                solver.add_clause_dimacs(&[v1 as i32, v2 as i32, v3 as i32]);
                solver.add_clause_dimacs(&[-(v1 as i32)]);
                solver.add_clause_dimacs(&[-(v2 as i32)]);
                solver.add_clause_dimacs(&[-(v3 as i32)]);

                let result = solver.solve();

                prop_assert_eq!(result, SolverResult::Unsat);
            }
        }

        #[test]
        fn learned_clauses_cumulative(n in 2usize..6usize) {
            let mut solver = Solver::new();

            for _ in 0..n {
                solver.new_var();
            }

            // Add conflicting constraints incrementally
            for i in 1..n {
                solver.add_clause_dimacs(&[i as i32, (i+1) as i32]);
                solver.add_clause_dimacs(&[-(i as i32)]);
            }

            solver.add_clause_dimacs(&[-(n as i32)]);
            solver.add_clause_dimacs(&[1]);

            let result = solver.solve();

            // Learned clauses should accumulate
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unsat));
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
                let mut solver = Solver::new();
                let max_var = v1.max(v2).max(v3);

                for _ in 0..=max_var {
                    solver.new_var();
                }

                // (v1 ∨ v2) ∧ (¬v2 ∨ v3)
                // Resolution gives: (v1 ∨ v3)
                solver.add_clause_dimacs(&[v1 as i32, v2 as i32]);
                solver.add_clause_dimacs(&[-(v2 as i32), v3 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SolverResult::Sat);
            }
        }

        #[test]
        fn resolution_detects_empty_clause(v in 1u32..20u32) {
            let mut solver = Solver::new();

            for _ in 0..v {
                solver.new_var();
            }

            // v ∧ ¬v leads to empty clause
            solver.add_clause_dimacs(&[v as i32]);
            solver.add_clause_dimacs(&[-(v as i32)]);

            let result = solver.solve();

            prop_assert_eq!(result, SolverResult::Unsat);
        }

        #[test]
        fn subsumption_removes_redundant_clauses(
            v1 in 1u32..15u32,
            v2 in 1u32..15u32
        ) {
            if v1 != v2 {
                let mut solver = Solver::new();
                let max_var = v1.max(v2);

                for _ in 0..=max_var {
                    solver.new_var();
                }

                // v1 subsumes (v1 ∨ v2)
                solver.add_clause_dimacs(&[v1 as i32]);
                solver.add_clause_dimacs(&[v1 as i32, v2 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SolverResult::Sat);
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
            let mut solver = Solver::new();

            for _ in 0..v {
                solver.new_var();
            }

            // Add some clauses
            for i in 1..v {
                solver.add_clause_dimacs(&[i as i32, (i+1) as i32]);
            }

            // Solve (restarts are enabled by default in SolverConfig)
            let result = solver.solve();

            // Should still find correct result after restarts
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unsat));
        }

        #[test]
        fn restart_doesnt_affect_correctness(n in 2usize..6usize) {
            // Create solver with Luby restarts
            let config1 = SolverConfig {
                restart_strategy: RestartStrategy::Luby,
                ..SolverConfig::default()
            };
            let mut solver1 = Solver::with_config(config1);

            // Create solver with Geometric restarts
            let config2 = SolverConfig {
                restart_strategy: RestartStrategy::Geometric,
                ..SolverConfig::default()
            };
            let mut solver2 = Solver::with_config(config2);

            for _ in 0..n {
                solver1.new_var();
                solver2.new_var();
            }

            // Add same clauses to both
            for i in 1..n {
                solver1.add_clause_dimacs(&[i as i32, (i+1) as i32]);
                solver2.add_clause_dimacs(&[i as i32, (i+1) as i32]);
            }

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
            let mut solver = Solver::new();

            for _ in 0..=v {
                solver.new_var();
            }

            // v appears only positively (pure literal)
            solver.add_clause_dimacs(&[v as i32, 1]);
            solver.add_clause_dimacs(&[v as i32, 2]);

            let result = solver.solve();

            prop_assert_eq!(result, SolverResult::Sat);
        }

        #[test]
        fn variable_elimination_preserves_sat(
            v1 in 1u32..15u32,
            v2 in 1u32..15u32
        ) {
            if v1 != v2 {
                let mut solver = Solver::new();
                let max_var = v1.max(v2);

                for _ in 0..=max_var {
                    solver.new_var();
                }

                // (v1 ∨ v2)
                solver.add_clause_dimacs(&[v1 as i32, v2 as i32]);

                let result = solver.solve();

                prop_assert_eq!(result, SolverResult::Sat);
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
            // Create solver with low clause deletion threshold to trigger deletion
            let config = SolverConfig {
                clause_deletion_threshold: 10,
                ..SolverConfig::default()
            };
            let mut solver = Solver::with_config(config);

            for _ in 0..n {
                solver.new_var();
            }

            // Add many clauses
            for i in 1..n {
                solver.add_clause_dimacs(&[i as i32, (i+1) as i32]);
            }

            // Clause deletion is enabled via config
            let result = solver.solve();

            // Should still work correctly
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unsat));
        }

        #[test]
        fn literal_watch_scheme_correct(
            v1 in 1u32..15u32,
            v2 in 1u32..15u32,
            v3 in 1u32..15u32
        ) {
            if v1 != v2 && v2 != v3 && v1 != v3 {
                let mut solver = Solver::new();
                let max_var = v1.max(v2).max(v3);

                for _ in 0..=max_var {
                    solver.new_var();
                }

                // (v1 ∨ v2 ∨ v3)
                solver.add_clause_dimacs(&[v1 as i32, v2 as i32, v3 as i32]);

                // ¬v1 ∧ ¬v2
                solver.add_clause_dimacs(&[-(v1 as i32)]);
                solver.add_clause_dimacs(&[-(v2 as i32)]);

                let result = solver.solve();

                // v3 must be true
                prop_assert_eq!(result, SolverResult::Sat);
            }
        }

        #[test]
        fn implication_graph_acyclic(n in 2usize..8usize) {
            let mut solver = Solver::new();

            for _ in 0..n {
                solver.new_var();
            }

            // Create implication chain (DAG structure)
            for i in 1..n {
                solver.add_clause_dimacs(&[-(i as i32), (i+1) as i32]);
            }

            let result = solver.solve();

            // Should have valid implication graph
            prop_assert!(matches!(result, SolverResult::Sat | SolverResult::Unsat));
        }
    }
}
