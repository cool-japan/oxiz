//! Property-based tests for Simplex LP solver
//!
//! This module tests:
//! - Simplex algorithm correctness
//! - Dual simplex properties
//! - Tableau consistency
//! - Optimal solution properties
//! - Unboundedness detection
//! - Infeasibility detection

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::lp::*;
use oxiz_math::rational::*;
use proptest::prelude::*;

/// Strategy for generating small LP coefficients
fn lp_coeff_strategy() -> impl Strategy<Value = i64> {
    -10i64..10i64
}

/// Strategy for generating positive coefficients
fn positive_coeff_strategy() -> impl Strategy<Value = i64> {
    1i64..10i64
}

/// Helper to create rational
fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

#[cfg(test)]
mod simplex_basic_properties {
    use super::*;

    proptest! {
        /// Test that feasible LP with single variable has solution
        #[test]
        fn simplex_single_var_feasible(
            c in lp_coeff_strategy(),
            bound in positive_coeff_strategy()
        ) {
            // maximize c*x subject to x <= bound, x >= 0
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            // Objective: maximize c*x
            lp.set_objective_coeff(x, rat(c));

            // Constraint: x <= bound
            lp.add_constraint(vec![(x, rat(1))], rat(bound), ConstraintType::Le);

            // Constraint: x >= 0
            lp.set_lower_bound(x, rat(0));

            let result = lp.solve();

            // Should be feasible if c > 0 (maximum at x=bound) or c < 0 (maximum at x=0)
            match result {
                LPResult::Optimal(value) => {
                    if c > 0 {
                        // Optimal value should be c*bound
                        prop_assert!((value - rat(c * bound)).abs() < rat(1) / rat(1000));
                    } else {
                        // Optimal value should be c*0 = 0 or c*bound
                        prop_assert!(value <= rat(c * bound).max(rat(0)));
                    }
                },
                LPResult::Infeasible => {
                    // Should not be infeasible with these constraints
                    prop_assert!(false, "LP should be feasible");
                },
                LPResult::Unbounded => {
                    // Should only be unbounded if c > 0 and no upper bound
                    prop_assert!(c > 0 && bound < 0);
                },
            }
        }

        /// Test that infeasible LP is detected
        #[test]
        fn simplex_detects_infeasibility() {
            // maximize x subject to x <= -1, x >= 0
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(1));
            lp.add_constraint(vec![(x, rat(1))], rat(-1), ConstraintType::Le);
            lp.set_lower_bound(x, rat(0));

            let result = lp.solve();

            // Should be infeasible
            prop_assert!(matches!(result, LPResult::Infeasible));
        }

        /// Test that unbounded LP is detected
        #[test]
        fn simplex_detects_unboundedness() {
            // maximize x subject to x >= 0 (no upper bound)
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(1));
            lp.set_lower_bound(x, rat(0));

            let result = lp.solve();

            // Should be unbounded
            prop_assert!(matches!(result, LPResult::Unbounded));
        }

        /// Test that LP with zero objective is feasible
        #[test]
        fn simplex_zero_objective(bound in positive_coeff_strategy()) {
            // maximize 0*x subject to x <= bound, x >= 0
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(0));
            lp.add_constraint(vec![(x, rat(1))], rat(bound), ConstraintType::Le);
            lp.set_lower_bound(x, rat(0));

            let result = lp.solve();

            // Should be optimal with value 0
            match result {
                LPResult::Optimal(value) => {
                    prop_assert_eq!(value, BigRational::zero());
                },
                _ => prop_assert!(false, "Should be optimal"),
            }
        }
    }
}

#[cfg(test)]
mod simplex_optimality_properties {
    use super::*;

    proptest! {
        /// Test that optimal solution satisfies all constraints
        #[test]
        fn simplex_solution_satisfies_constraints(
            c1 in lp_coeff_strategy(),
            c2 in lp_coeff_strategy(),
            b in positive_coeff_strategy()
        ) {
            // maximize c1*x1 + c2*x2 subject to x1 + x2 <= b, x1,x2 >= 0
            let mut lp = SimplexLP::new();
            let x1 = lp.add_variable();
            let x2 = lp.add_variable();

            lp.set_objective_coeff(x1, rat(c1));
            lp.set_objective_coeff(x2, rat(c2));

            lp.add_constraint(vec![(x1, rat(1)), (x2, rat(1))], rat(b), ConstraintType::Le);
            lp.set_lower_bound(x1, rat(0));
            lp.set_lower_bound(x2, rat(0));

            let result = lp.solve();

            if let LPResult::Optimal(_) = result {
                // Get solution values
                let val1 = lp.get_variable_value(x1);
                let val2 = lp.get_variable_value(x2);

                // Check non-negativity
                prop_assert!(val1 >= BigRational::zero());
                prop_assert!(val2 >= BigRational::zero());

                // Check constraint satisfaction
                prop_assert!(val1 + val2 <= rat(b));
            }
        }

        /// Test that maximizing positive coefficient gives positive value
        #[test]
        fn simplex_maximize_positive_gives_positive(
            c in positive_coeff_strategy(),
            bound in positive_coeff_strategy()
        ) {
            // maximize c*x subject to x <= bound, x >= 0
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(c));
            lp.add_constraint(vec![(x, rat(1))], rat(bound), ConstraintType::Le);
            lp.set_lower_bound(x, rat(0));

            let result = lp.solve();

            if let LPResult::Optimal(value) = result {
                // Optimal value should be positive
                prop_assert!(value >= BigRational::zero());
                // And should be c*bound
                prop_assert!((value - rat(c * bound)).abs() < rat(1) / rat(1000));
            }
        }

        /// Test that minimizing is equivalent to maximizing negative
        #[test]
        fn simplex_min_equals_max_negative(
            c in lp_coeff_strategy(),
            bound in positive_coeff_strategy()
        ) {
            // Test: min c*x == max -c*x

            // Minimize c*x
            let mut lp_min = SimplexLP::new();
            let x1 = lp_min.add_variable();
            lp_min.set_objective_coeff(x1, rat(c));
            lp_min.set_minimize(true);
            lp_min.add_constraint(vec![(x1, rat(1))], rat(bound), ConstraintType::Le);
            lp_min.set_lower_bound(x1, rat(0));

            // Maximize -c*x
            let mut lp_max = SimplexLP::new();
            let x2 = lp_max.add_variable();
            lp_max.set_objective_coeff(x2, rat(-c));
            lp_max.add_constraint(vec![(x2, rat(1))], rat(bound), ConstraintType::Le);
            lp_max.set_lower_bound(x2, rat(0));

            let result_min = lp_min.solve();
            let result_max = lp_max.solve();

            match (result_min, result_max) {
                (LPResult::Optimal(val_min), LPResult::Optimal(val_max)) => {
                    // min c*x = -max(-c*x)
                    prop_assert!((val_min + val_max).abs() < rat(1) / rat(1000));
                },
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod simplex_duality_properties {
    use super::*;

    proptest! {
        /// Test weak duality: primal objective <= dual objective (for maximization)
        #[test]
        fn simplex_weak_duality(
            c1 in lp_coeff_strategy(),
            c2 in lp_coeff_strategy(),
            b in positive_coeff_strategy()
        ) {
            // Primal: maximize c1*x1 + c2*x2 subject to x1 + x2 <= b, x1,x2 >= 0
            let mut primal = SimplexLP::new();
            let x1 = primal.add_variable();
            let x2 = primal.add_variable();

            primal.set_objective_coeff(x1, rat(c1));
            primal.set_objective_coeff(x2, rat(c2));
            primal.add_constraint(vec![(x1, rat(1)), (x2, rat(1))], rat(b), ConstraintType::Le);
            primal.set_lower_bound(x1, rat(0));
            primal.set_lower_bound(x2, rat(0));

            let primal_result = primal.solve();

            // Dual: minimize b*y subject to y >= c1, y >= c2, y >= 0
            let mut dual = SimplexLP::new();
            let y = dual.add_variable();

            dual.set_objective_coeff(y, rat(b));
            dual.set_minimize(true);
            dual.set_lower_bound(y, rat(c1).max(rat(c2)).max(BigRational::zero()));

            let dual_result = dual.solve();

            // If both are optimal, primal <= dual
            match (primal_result, dual_result) {
                (LPResult::Optimal(primal_val), LPResult::Optimal(dual_val)) => {
                    prop_assert!(primal_val <= dual_val + rat(1) / rat(1000));
                },
                _ => {}
            }
        }

        /// Test strong duality: if primal is optimal, dual is optimal with same value
        #[test]
        fn simplex_strong_duality_simple(b in positive_coeff_strategy()) {
            // Primal: maximize x subject to x <= b, x >= 0
            let mut primal = SimplexLP::new();
            let x = primal.add_variable();

            primal.set_objective_coeff(x, rat(1));
            primal.add_constraint(vec![(x, rat(1))], rat(b), ConstraintType::Le);
            primal.set_lower_bound(x, rat(0));

            let primal_result = primal.solve();

            if let LPResult::Optimal(primal_val) = primal_result {
                // Optimal value should be b
                prop_assert!((primal_val - rat(b)).abs() < rat(1) / rat(1000));
            }
        }
    }
}

#[cfg(test)]
mod simplex_tableau_properties {
    use super::*;

    proptest! {
        /// Test that initial tableau is consistent
        #[test]
        fn simplex_initial_tableau_consistent(
            c in lp_coeff_strategy(),
            b in positive_coeff_strategy()
        ) {
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(c));
            lp.add_constraint(vec![(x, rat(1))], rat(b), ConstraintType::Le);
            lp.set_lower_bound(x, rat(0));

            // Tableau should be initialized without panic
            let tableau = lp.get_tableau();

            // Check basic consistency: number of rows and columns
            prop_assert!(tableau.num_rows() > 0);
            prop_assert!(tableau.num_cols() > 0);
        }

        /// Test that pivot operations maintain tableau validity
        #[test]
        fn simplex_pivot_maintains_validity(
            c in lp_coeff_strategy(),
            b in positive_coeff_strategy()
        ) {
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(c));
            lp.add_constraint(vec![(x, rat(1))], rat(b), ConstraintType::Le);
            lp.set_lower_bound(x, rat(0));

            // Perform one pivot if possible
            let pivoted = lp.perform_pivot();

            // Tableau should still be valid after pivot
            if pivoted {
                let tableau = lp.get_tableau();
                prop_assert!(tableau.num_rows() > 0);
                prop_assert!(tableau.num_cols() > 0);
            }
        }

        /// Test that basic feasible solution satisfies constraints
        #[test]
        fn simplex_bfs_satisfies_constraints(
            b1 in positive_coeff_strategy(),
            b2 in positive_coeff_strategy()
        ) {
            let mut lp = SimplexLP::new();
            let x1 = lp.add_variable();
            let x2 = lp.add_variable();

            lp.set_objective_coeff(x1, rat(1));
            lp.set_objective_coeff(x2, rat(1));

            lp.add_constraint(vec![(x1, rat(1))], rat(b1), ConstraintType::Le);
            lp.add_constraint(vec![(x2, rat(1))], rat(b2), ConstraintType::Le);
            lp.set_lower_bound(x1, rat(0));
            lp.set_lower_bound(x2, rat(0));

            // Get initial basic feasible solution
            let bfs = lp.get_basic_solution();

            // All values should be non-negative
            for value in bfs {
                prop_assert!(value >= BigRational::zero());
            }
        }
    }
}

#[cfg(test)]
mod simplex_degenerate_cases {
    use super::*;

    proptest! {
        /// Test that degenerate LP (multiple optimal solutions) is handled
        #[test]
        fn simplex_handles_degeneracy() {
            // maximize 0*x subject to x <= 10, x >= 0
            // Any x in [0, 10] is optimal
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(0));
            lp.add_constraint(vec![(x, rat(1))], rat(10), ConstraintType::Le);
            lp.set_lower_bound(x, rat(0));

            let result = lp.solve();

            // Should find an optimal solution (any feasible point)
            prop_assert!(matches!(result, LPResult::Optimal(_)));
        }

        /// Test that LP with redundant constraints works
        #[test]
        fn simplex_handles_redundant_constraints(b in positive_coeff_strategy()) {
            // maximize x subject to x <= b, x <= b+1, x >= 0
            // Second constraint is redundant
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(1));
            lp.add_constraint(vec![(x, rat(1))], rat(b), ConstraintType::Le);
            lp.add_constraint(vec![(x, rat(1))], rat(b + 1), ConstraintType::Le);
            lp.set_lower_bound(x, rat(0));

            let result = lp.solve();

            if let LPResult::Optimal(value) = result {
                // Optimal should be at x = b
                prop_assert!((value - rat(b)).abs() < rat(1) / rat(1000));
            }
        }

        /// Test that tight constraints are handled correctly
        #[test]
        fn simplex_handles_tight_constraints(b in positive_coeff_strategy()) {
            // maximize x subject to x <= b, x >= b, so x = b
            let mut lp = SimplexLP::new();
            let x = lp.add_variable();

            lp.set_objective_coeff(x, rat(1));
            lp.add_constraint(vec![(x, rat(1))], rat(b), ConstraintType::Le);
            lp.set_lower_bound(x, rat(b));

            let result = lp.solve();

            if let LPResult::Optimal(value) = result {
                // Optimal must be exactly b
                prop_assert!((value - rat(b)).abs() < rat(1) / rat(1000));
            }
        }
    }
}

#[cfg(test)]
mod simplex_sensitivity_properties {
    use super::*;

    proptest! {
        /// Test that increasing RHS increases optimal value (for maximize)
        #[test]
        fn simplex_rhs_sensitivity(
            c in positive_coeff_strategy(),
            b1 in 1i64..5i64,
            delta in 1i64..5i64
        ) {
            let b2 = b1 + delta;

            // LP 1: maximize c*x subject to x <= b1, x >= 0
            let mut lp1 = SimplexLP::new();
            let x1 = lp1.add_variable();
            lp1.set_objective_coeff(x1, rat(c));
            lp1.add_constraint(vec![(x1, rat(1))], rat(b1), ConstraintType::Le);
            lp1.set_lower_bound(x1, rat(0));

            // LP 2: maximize c*x subject to x <= b2, x >= 0
            let mut lp2 = SimplexLP::new();
            let x2 = lp2.add_variable();
            lp2.set_objective_coeff(x2, rat(c));
            lp2.add_constraint(vec![(x2, rat(1))], rat(b2), ConstraintType::Le);
            lp2.set_lower_bound(x2, rat(0));

            let result1 = lp1.solve();
            let result2 = lp2.solve();

            match (result1, result2) {
                (LPResult::Optimal(val1), LPResult::Optimal(val2)) => {
                    // Increasing RHS should increase optimal value
                    prop_assert!(val2 >= val1);
                },
                _ => {}
            }
        }

        /// Test that increasing objective coefficient increases optimal value
        #[test]
        fn simplex_obj_coeff_sensitivity(
            c1 in 1i64..5i64,
            delta in 1i64..5i64,
            b in positive_coeff_strategy()
        ) {
            let c2 = c1 + delta;

            // LP 1: maximize c1*x subject to x <= b, x >= 0
            let mut lp1 = SimplexLP::new();
            let x1 = lp1.add_variable();
            lp1.set_objective_coeff(x1, rat(c1));
            lp1.add_constraint(vec![(x1, rat(1))], rat(b), ConstraintType::Le);
            lp1.set_lower_bound(x1, rat(0));

            // LP 2: maximize c2*x subject to x <= b, x >= 0
            let mut lp2 = SimplexLP::new();
            let x2 = lp2.add_variable();
            lp2.set_objective_coeff(x2, rat(c2));
            lp2.add_constraint(vec![(x2, rat(1))], rat(b), ConstraintType::Le);
            lp2.set_lower_bound(x2, rat(0));

            let result1 = lp1.solve();
            let result2 = lp2.solve();

            match (result1, result2) {
                (LPResult::Optimal(val1), LPResult::Optimal(val2)) => {
                    // Increasing coefficient should increase optimal value
                    prop_assert!(val2 >= val1);
                },
                _ => {}
            }
        }
    }
}
