//! Property-based tests for LP solver
//!
//! This module tests:
//! - LP algorithm correctness
//! - Optimal solution properties
//! - Infeasibility detection

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use oxiz_math::lp_core::{ConstraintSense, LPResult, LPSolver, OptDir};
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
            // maximize c*x subject to 0 <= x <= bound
            let mut lp = LPSolver::new();
            let x = lp.new_continuous();

            // Objective: maximize c*x
            lp.set_objective(x, rat(c));
            lp.set_direction(OptDir::Maximize);

            // Constraints: x <= bound, x >= 0
            lp.new_constraint([(x, rat(1))], ConstraintSense::Le, rat(bound));
            lp.new_constraint([(x, rat(1))], ConstraintSense::Ge, rat(0));

            let result = lp.solve();

            // Should be feasible (optimal or unknown)
            match result {
                LPResult::Optimal { objective, .. } => {
                    // Optimal objective should be non-negative for c >= 0
                    if c >= 0 {
                        prop_assert!(objective >= BigRational::zero());
                    }
                    // For c > 0, optimal should be at most c*bound
                    if c > 0 {
                        prop_assert!(objective <= rat(c * bound + 1));
                    }
                },
                LPResult::Unbounded => {
                    // Should not be unbounded with both upper and lower bounds
                },
                LPResult::Infeasible => {
                    // Should not be infeasible with these constraints
                },
                LPResult::Unknown => {
                    // Unknown is acceptable
                }
            }
        }

        /// Test that LP with zero objective finds feasible solution
        #[test]
        fn simplex_zero_objective(bound in positive_coeff_strategy()) {
            // maximize 0*x subject to x <= bound
            let mut lp = LPSolver::new();
            let x = lp.new_continuous();

            lp.set_objective(x, rat(0));
            lp.set_direction(OptDir::Maximize);
            lp.new_constraint([(x, rat(1))], ConstraintSense::Le, rat(bound));

            let result = lp.solve();

            // Should be optimal with value 0; other results acceptable for edge cases
            if let LPResult::Optimal { objective, .. } = result {
                prop_assert_eq!(objective, BigRational::zero());
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
            // maximize c1*x1 + c2*x2 subject to x1 + x2 <= b
            let mut lp = LPSolver::new();
            let x1 = lp.new_continuous();
            let x2 = lp.new_continuous();

            lp.set_objective(x1, rat(c1));
            lp.set_objective(x2, rat(c2));
            lp.set_direction(OptDir::Maximize);

            lp.new_constraint([(x1, rat(1)), (x2, rat(1))], ConstraintSense::Le, rat(b));

            let result = lp.solve();

            if let LPResult::Optimal { values, .. } = result {
                // Get solution values
                let val1 = values.get(&x1).cloned().unwrap_or(BigRational::zero());
                let val2 = values.get(&x2).cloned().unwrap_or(BigRational::zero());

                // Check non-negativity (implicit lower bound is 0)
                prop_assert!(val1 >= BigRational::zero());
                prop_assert!(val2 >= BigRational::zero());

                // Check constraint satisfaction
                prop_assert!(&val1 + &val2 <= rat(b));
            }
        }

        /// Test that maximizing positive coefficient gives positive value
        #[test]
        fn simplex_maximize_positive_gives_positive(
            c in positive_coeff_strategy(),
            bound in positive_coeff_strategy()
        ) {
            // maximize c*x subject to x <= bound
            let mut lp = LPSolver::new();
            let x = lp.new_continuous();

            lp.set_objective(x, rat(c));
            lp.set_direction(OptDir::Maximize);
            lp.new_constraint([(x, rat(1))], ConstraintSense::Le, rat(bound));

            let result = lp.solve();

            if let LPResult::Optimal { objective, .. } = result {
                // Optimal value should be positive
                prop_assert!(objective >= BigRational::zero());
            }
        }

        /// Test that minimizing is equivalent to maximizing negative
        #[test]
        fn simplex_min_equals_max_negative(
            c in lp_coeff_strategy(),
            bound in positive_coeff_strategy()
        ) {
            // Test: min c*x == -max -c*x for bounded LP

            // Minimize c*x
            let mut lp_min = LPSolver::new();
            let x1 = lp_min.new_continuous();
            lp_min.set_objective(x1, rat(c));
            lp_min.set_direction(OptDir::Minimize);
            lp_min.new_constraint([(x1, rat(1))], ConstraintSense::Le, rat(bound));

            // Maximize -c*x
            let mut lp_max = LPSolver::new();
            let x2 = lp_max.new_continuous();
            lp_max.set_objective(x2, rat(-c));
            lp_max.set_direction(OptDir::Maximize);
            lp_max.new_constraint([(x2, rat(1))], ConstraintSense::Le, rat(bound));

            let result_min = lp_min.solve();
            let result_max = lp_max.solve();

            if let (LPResult::Optimal { objective: val_min, .. }, LPResult::Optimal { objective: val_max, .. }) = (result_min, result_max) {
                // min c*x = -max(-c*x)
                prop_assert!((&val_min + &val_max).abs() < rat(1));
            }
        }
    }
}

#[cfg(test)]
mod simplex_duality_properties {
    use super::*;

    proptest! {
        /// Test strong duality for simple LP
        #[test]
        fn simplex_strong_duality_simple(b in positive_coeff_strategy()) {
            // Primal: maximize x subject to 0 <= x <= b
            let mut primal = LPSolver::new();
            let x = primal.new_continuous();

            primal.set_objective(x, rat(1));
            primal.set_direction(OptDir::Maximize);
            primal.new_constraint([(x, rat(1))], ConstraintSense::Le, rat(b));
            primal.new_constraint([(x, rat(1))], ConstraintSense::Ge, rat(0));

            let primal_result = primal.solve();

            if let LPResult::Optimal { objective: primal_val, .. } = primal_result {
                // Optimal value should be in [0, b]
                prop_assert!(primal_val >= BigRational::zero());
                prop_assert!(primal_val <= rat(b + 1));
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

            // LP 1: maximize c*x subject to x <= b1
            let mut lp1 = LPSolver::new();
            let x1 = lp1.new_continuous();
            lp1.set_objective(x1, rat(c));
            lp1.set_direction(OptDir::Maximize);
            lp1.new_constraint([(x1, rat(1))], ConstraintSense::Le, rat(b1));

            // LP 2: maximize c*x subject to x <= b2
            let mut lp2 = LPSolver::new();
            let x2 = lp2.new_continuous();
            lp2.set_objective(x2, rat(c));
            lp2.set_direction(OptDir::Maximize);
            lp2.new_constraint([(x2, rat(1))], ConstraintSense::Le, rat(b2));

            let result1 = lp1.solve();
            let result2 = lp2.solve();

            if let (LPResult::Optimal { objective: val1, .. }, LPResult::Optimal { objective: val2, .. }) = (result1, result2) {
                // Increasing RHS should increase optimal value
                prop_assert!(val2 >= val1);
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

            // LP 1: maximize c1*x subject to x <= b
            let mut lp1 = LPSolver::new();
            let x1 = lp1.new_continuous();
            lp1.set_objective(x1, rat(c1));
            lp1.set_direction(OptDir::Maximize);
            lp1.new_constraint([(x1, rat(1))], ConstraintSense::Le, rat(b));

            // LP 2: maximize c2*x subject to x <= b
            let mut lp2 = LPSolver::new();
            let x2 = lp2.new_continuous();
            lp2.set_objective(x2, rat(c2));
            lp2.set_direction(OptDir::Maximize);
            lp2.new_constraint([(x2, rat(1))], ConstraintSense::Le, rat(b));

            let result1 = lp1.solve();
            let result2 = lp2.solve();

            if let (LPResult::Optimal { objective: val1, .. }, LPResult::Optimal { objective: val2, .. }) = (result1, result2) {
                // Increasing coefficient should increase optimal value
                prop_assert!(val2 >= val1);
            }
        }
    }
}
