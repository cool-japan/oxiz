//! Property-based tests for polynomial root-related operations
//!
//! Simplified test file that tests available API methods.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use oxiz_math::polynomial::*;
use proptest::prelude::*;
use rustc_hash::FxHashMap;

/// Strategy for generating polynomial coefficients
fn root_coeff_strategy() -> impl Strategy<Value = i64> {
    -5i64..5i64
}

/// Helper to create rational
fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

#[cfg(test)]
mod polynomial_basic_properties {
    use super::*;

    proptest! {
        /// Test polynomial evaluation
        #[test]
        fn polynomial_evaluation(c0 in root_coeff_strategy(), c1 in root_coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c0, &[]), (c1, &[(0, 1)])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(0));

            // p(0) should be c0
            let result = p.eval(&assignment);
            prop_assert_eq!(result, rat(c0));
        }

        /// Test polynomial zero evaluation
        #[test]
        fn zero_polynomial_evaluation(x in -5i64..5i64) {
            let p = Polynomial::zero();

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            // Zero polynomial evaluates to 0
            let result = p.eval(&assignment);
            prop_assert_eq!(result, BigRational::zero());
        }

        /// Test polynomial addition commutativity
        #[test]
        fn polynomial_add_commutative(
            c1 in root_coeff_strategy(),
            c2 in root_coeff_strategy()
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let sum1 = &p + &q;
            let sum2 = &q + &p;

            prop_assert_eq!(sum1, sum2);
        }

        /// Test polynomial multiplication commutativity
        #[test]
        fn polynomial_mul_commutative(
            c1 in root_coeff_strategy(),
            c2 in root_coeff_strategy()
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let prod1 = &p * &q;
            let prod2 = &q * &p;

            prop_assert_eq!(prod1, prod2);
        }
    }
}

#[cfg(test)]
mod polynomial_derivative_properties {
    use super::*;

    proptest! {
        /// Test derivative of constant is zero
        #[test]
        fn derivative_constant_is_zero(c in root_coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[])]);

            let dp = p.derivative(0);

            prop_assert!(dp.is_zero());
        }

        /// Test derivative of linear is constant
        #[test]
        fn derivative_linear_is_constant(c in root_coeff_strategy()) {
            // p = c*x
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);

            let dp = p.derivative(0);

            // derivative should be constant c
            let expected = Polynomial::from_coeffs_int(&[(c, &[])]);
            prop_assert_eq!(dp, expected);
        }
    }
}

#[cfg(test)]
mod polynomial_gcd_properties {
    use super::*;

    proptest! {
        /// Test GCD univariate is symmetric
        #[test]
        fn gcd_univariate_symmetric(
            c1 in 1i64..8i64,
            c2 in 1i64..8i64
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let g1 = p1.gcd_univariate(&p2);
            let g2 = p2.gcd_univariate(&p1);

            // GCD should be same (both zero or both non-zero)
            prop_assert_eq!(g1.is_zero(), g2.is_zero());
        }

        /// Test GCD with zero
        #[test]
        fn gcd_with_zero(c in 1i64..10i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let zero = Polynomial::zero();

            let g = p.gcd_univariate(&zero);

            // GCD(p, 0) should be related to p
            if !p.is_zero() {
                prop_assert!(!g.is_zero());
            }
        }
    }
}

#[cfg(test)]
mod polynomial_square_free_properties {
    use super::*;

    proptest! {
        /// Test square_free doesn't panic
        #[test]
        fn square_free_works(c in 1i64..10i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 2)])]);

            // Just verify it doesn't panic
            let _sf = p.square_free();
            prop_assert!(true);
        }
    }
}
