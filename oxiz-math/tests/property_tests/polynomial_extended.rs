//! Property-based tests for polynomial operations
//!
//! This module tests:
//! - Polynomial arithmetic properties
//! - Evaluation and interpolation
//! - Derivative properties
//! - Multivariate polynomial properties

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use oxiz_math::polynomial::*;
use proptest::prelude::*;
use rustc_hash::FxHashMap;

/// Strategy for generating small polynomial coefficients
fn coeff_strategy() -> impl Strategy<Value = i64> {
    -10i64..10i64
}

/// Helper to create rational
fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

#[cfg(test)]
mod polynomial_arithmetic_properties {
    use super::*;

    proptest! {
        /// Test that polynomial addition is commutative
        #[test]
        fn poly_add_commutative(c1 in coeff_strategy(), c2 in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let sum1 = &p + &q;
            let sum2 = &q + &p;

            prop_assert_eq!(sum1, sum2);
        }

        /// Test that polynomial addition is associative
        #[test]
        fn poly_add_associative(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            c3 in coeff_strategy()
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);
            let r = Polynomial::from_coeffs_int(&[(c3, &[(0, 1)])]);

            let lhs = &(&p + &q) + &r;
            let rhs = &p + &(&q + &r);

            prop_assert_eq!(lhs, rhs);
        }

        /// Test that polynomial multiplication is commutative
        #[test]
        fn poly_mul_commutative(c1 in coeff_strategy(), c2 in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let prod1 = &p * &q;
            let prod2 = &q * &p;

            prop_assert_eq!(prod1, prod2);
        }

        /// Test that polynomial multiplication is associative
        #[test]
        fn poly_mul_associative(
            c1 in 1i64..5i64,
            c2 in 1i64..5i64,
            c3 in 1i64..5i64
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);
            let r = Polynomial::from_coeffs_int(&[(c3, &[(0, 1)])]);

            let lhs = &(&p * &q) * &r;
            let rhs = &p * &(&q * &r);

            prop_assert_eq!(lhs, rhs);
        }

        /// Test that adding zero doesn't change the polynomial
        #[test]
        fn poly_add_zero_identity(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let zero = Polynomial::zero();

            let result = &p + &zero;

            prop_assert_eq!(result, p);
        }
    }
}

#[cfg(test)]
mod polynomial_evaluation_properties {
    use super::*;

    proptest! {
        /// Test that polynomial evaluation at a point works correctly
        #[test]
        fn poly_eval_at_point(c0 in coeff_strategy(), c1 in coeff_strategy(), x in -5i64..5i64) {
            // p = c0 + c1*x
            let p = Polynomial::from_coeffs_int(&[(c0, &[]), (c1, &[(0, 1)])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = p.eval(&assignment);
            let expected = rat(c0 + c1 * x);

            prop_assert_eq!(result, expected);
        }

        /// Test that zero polynomial evaluates to zero
        #[test]
        fn zero_poly_eval_zero(x in -10i64..10i64) {
            let zero = Polynomial::zero();

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = zero.eval(&assignment);

            prop_assert_eq!(result, BigRational::zero());
        }

        /// Test that constant polynomial evaluates to the constant
        #[test]
        fn constant_poly_eval(c in coeff_strategy(), x in -10i64..10i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = p.eval(&assignment);

            prop_assert_eq!(result, rat(c));
        }
    }
}

#[cfg(test)]
mod polynomial_derivative_properties {
    use super::*;

    // Tests with no parameters go outside proptest! block
    #[test]
    fn poly_deriv_x_is_one() {
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]);
        let dp = p.derivative(0);
        let one = Polynomial::from_coeffs_int(&[(1, &[])]);

        assert_eq!(dp, one);
    }

    #[test]
    fn poly_deriv_x_squared() {
        let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
        let dp = p.derivative(0);
        let expected = Polynomial::from_coeffs_int(&[(2, &[(0, 1)])]);

        assert_eq!(dp, expected);
    }

    proptest! {
        /// Test that derivative of constant is zero
        #[test]
        fn poly_deriv_constant_is_zero(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[])]);
            let dp = p.derivative(0);

            prop_assert_eq!(dp, Polynomial::zero());
        }

        /// Test that derivative of c*x is c
        #[test]
        fn poly_deriv_cx_is_c(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let dp = p.derivative(0);
            let expected = Polynomial::from_coeffs_int(&[(c, &[])]);

            prop_assert_eq!(dp, expected);
        }

        /// Test that derivative of x^n is n*x^(n-1)
        #[test]
        fn poly_deriv_power_rule(n in 2u32..6u32) {
            // p = x^n
            let p = Polynomial::from_coeffs_int(&[(1, &[(0, n)])]);
            let dp = p.derivative(0);

            // expected = n * x^(n-1)
            let expected = Polynomial::from_coeffs_int(&[(n as i64, &[(0, n-1)])]);

            prop_assert_eq!(dp, expected);
        }

        /// Test linearity: d/dx(p + q) = d/dx(p) + d/dx(q)
        #[test]
        fn poly_deriv_additive(
            c1 in coeff_strategy(),
            c2 in coeff_strategy()
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 2)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 2)])]);

            let sum = &p + &q;
            let d_sum = sum.derivative(0);

            let dp = p.derivative(0);
            let dq = q.derivative(0);
            let sum_d = &dp + &dq;

            prop_assert_eq!(d_sum, sum_d);
        }
    }
}

#[cfg(test)]
mod polynomial_gcd_properties {
    use super::*;

    proptest! {
        /// Test GCD univariate is symmetric
        #[test]
        fn poly_gcd_univariate_symmetric(
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
        fn poly_gcd_with_zero(c in 1i64..10i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let zero = Polynomial::zero();

            let g = p.gcd_univariate(&zero);

            // GCD(p, 0) should be related to p
            if !p.is_zero() {
                prop_assert!(!g.is_zero());
            }
        }

        /// Test GCD with self
        #[test]
        fn poly_gcd_with_self(c in 1i64..10i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 2)])]);

            let g = p.gcd_univariate(&p);

            // gcd(p, p) should be non-zero for non-zero p
            if !p.is_zero() {
                prop_assert!(!g.is_zero());
            }
        }
    }
}

#[cfg(test)]
mod multivariate_polynomial_properties {
    use super::*;

    proptest! {
        /// Test that multivariate addition is commutative
        #[test]
        fn mv_poly_add_commutative(
            c1 in coeff_strategy(),
            c2 in coeff_strategy()
        ) {
            // p uses variable 0, q uses variable 1
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(1, 1)])]);

            let sum1 = &p + &q;
            let sum2 = &q + &p;

            prop_assert_eq!(sum1, sum2);
        }

        /// Test multivariate polynomial evaluation
        #[test]
        fn mv_poly_evaluation(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            x in -5i64..5i64,
            y in -5i64..5i64
        ) {
            // p = c1*x + c2*y
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)]), (c2, &[(1, 1)])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));
            assignment.insert(1, rat(y));

            let result = p.eval(&assignment);
            let expected = rat(c1 * x + c2 * y);

            prop_assert_eq!(result, expected);
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
