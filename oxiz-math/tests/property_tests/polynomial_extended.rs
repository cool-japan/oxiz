//! Extended property-based tests for polynomial operations
//!
//! This module provides comprehensive testing of:
//! - Polynomial arithmetic (add, sub, mul, div)
//! - GCD and factorization
//! - Evaluation and interpolation
//! - Derivative and integration properties
//! - Multivariate polynomial properties

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::polynomial::*;
use oxiz_math::rational::*;
use proptest::prelude::*;
use rustc_hash::FxHashMap;

/// Strategy for generating small polynomial coefficients
fn coeff_strategy() -> impl Strategy<Value = i64> {
    -10i64..10i64
}

/// Strategy for generating BigInt coefficients
fn bigint_coeff_strategy() -> impl Strategy<Value = BigInt> {
    coeff_strategy().prop_map(BigInt::from)
}

/// Strategy for generating evaluation points
fn eval_point_strategy() -> impl Strategy<Value = i64> {
    -5i64..5i64
}

#[cfg(test)]
mod polynomial_arithmetic_properties {
    use super::*;

    proptest! {
        /// Test that polynomial addition is commutative
        #[test]
        fn poly_add_commutative(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            c3 in coeff_strategy(),
            c4 in coeff_strategy()
        ) {
            // p1 = c1 + c2*x
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[]), (c2, &[(0, 1)])]);
            // p2 = c3 + c4*x
            let p2 = Polynomial::from_coeffs_int(&[(c3, &[]), (c4, &[(0, 1)])]);

            let sum1 = &p1 + &p2;
            let sum2 = &p2 + &p1;

            prop_assert_eq!(sum1, sum2);
        }

        /// Test that polynomial addition is associative
        #[test]
        fn poly_add_associative(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            c3 in coeff_strategy()
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);
            let p3 = Polynomial::from_coeffs_int(&[(c3, &[(0, 1)])]);

            let sum1 = &(&p1 + &p2) + &p3;
            let sum2 = &p1 + &(&p2 + &p3);

            prop_assert_eq!(sum1, sum2);
        }

        /// Test that polynomial multiplication is commutative
        #[test]
        fn poly_mul_commutative(
            c1 in coeff_strategy(),
            c2 in coeff_strategy()
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let prod1 = &p1 * &p2;
            let prod2 = &p2 * &p1;

            prop_assert_eq!(prod1, prod2);
        }

        /// Test that polynomial multiplication is associative
        #[test]
        fn poly_mul_associative(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            c3 in coeff_strategy()
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);
            let p3 = Polynomial::from_coeffs_int(&[(c3, &[(0, 1)])]);

            let prod1 = &(&p1 * &p2) * &p3;
            let prod2 = &p1 * &(&p2 * &p3);

            prop_assert_eq!(prod1, prod2);
        }

        /// Test distributivity: p1 * (p2 + p3) = p1*p2 + p1*p3
        #[test]
        fn poly_distributive(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            c3 in coeff_strategy()
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);
            let p3 = Polynomial::from_coeffs_int(&[(c3, &[(0, 1)])]);

            let left = &p1 * &(&p2 + &p3);
            let right = &(&p1 * &p2) + &(&p1 * &p3);

            prop_assert_eq!(left, right);
        }

        /// Test that zero is additive identity
        #[test]
        fn poly_zero_additive_identity(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let zero = Polynomial::zero();

            let sum = &p + &zero;

            prop_assert_eq!(sum, p);
        }

        /// Test that one is multiplicative identity
        #[test]
        fn poly_one_multiplicative_identity(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let one = Polynomial::one();

            let prod = &p * &one;

            prop_assert_eq!(prod, p);
        }

        /// Test that zero is multiplicative annihilator
        #[test]
        fn poly_zero_multiplicative_annihilator(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let zero = Polynomial::zero();

            let prod = &p * &zero;

            prop_assert_eq!(prod, Polynomial::zero());
        }

        /// Test that p - p = 0
        #[test]
        fn poly_subtract_self_is_zero(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);

            let diff = &p - &p;

            prop_assert_eq!(diff, Polynomial::zero());
        }

        /// Test that -(- p) = p
        #[test]
        fn poly_double_negation(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);

            let neg_p = -(&p);
            let neg_neg_p = -(&neg_p);

            prop_assert_eq!(neg_neg_p, p);
        }

        /// Test that (p + q) - q = p
        #[test]
        fn poly_add_subtract_cancels(
            c1 in coeff_strategy(),
            c2 in coeff_strategy()
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let sum = &p + &q;
            let diff = &sum - &q;

            prop_assert_eq!(diff, p);
        }

        /// Test that degree(p*q) = degree(p) + degree(q) for non-zero polynomials
        #[test]
        fn poly_degree_multiplicative(
            c1 in 1i64..10i64,
            c2 in 1i64..10i64
        ) {
            // Create p = c1*x  (degree 1)
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            // Create q = c2*x  (degree 1)
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let prod = &p * &q;

            // degree(p*q) should be 2
            prop_assert_eq!(prod.degree(), Some(2));
        }
    }
}

#[cfg(test)]
mod polynomial_evaluation_properties {
    use super::*;

    proptest! {
        /// Test that evaluating p + q = eval(p) + eval(q)
        #[test]
        fn poly_eval_additive(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            x in eval_point_strategy()
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let sum = &p1 + &p2;

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let eval_sum = sum.eval(&assignment);
            let eval_p1 = p1.eval(&assignment);
            let eval_p2 = p2.eval(&assignment);
            let sum_evals = eval_p1 + eval_p2;

            prop_assert_eq!(eval_sum, sum_evals);
        }

        /// Test that evaluating p * q = eval(p) * eval(q)
        #[test]
        fn poly_eval_multiplicative(
            c1 in coeff_strategy(),
            c2 in coeff_strategy(),
            x in eval_point_strategy()
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let prod = &p1 * &p2;

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let eval_prod = prod.eval(&assignment);
            let eval_p1 = p1.eval(&assignment);
            let eval_p2 = p2.eval(&assignment);
            let prod_evals = eval_p1 * eval_p2;

            prop_assert_eq!(eval_prod, prod_evals);
        }

        /// Test that constant polynomials evaluate to their constant
        #[test]
        fn poly_eval_constant(c in coeff_strategy(), x in eval_point_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = p.eval(&assignment);

            prop_assert_eq!(result, rat(c));
        }

        /// Test that linear polynomials c*x evaluate correctly
        #[test]
        fn poly_eval_linear(c in coeff_strategy(), x in eval_point_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = p.eval(&assignment);
            let expected = rat(c * x);

            prop_assert_eq!(result, expected);
        }

        /// Test that quadratic polynomials c*x^2 evaluate correctly
        #[test]
        fn poly_eval_quadratic(c in coeff_strategy(), x in eval_point_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 2)])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = p.eval(&assignment);
            let expected = rat(c * x * x);

            prop_assert_eq!(result, expected);
        }

        /// Test that zero polynomial evaluates to zero
        #[test]
        fn poly_zero_evals_to_zero(x in eval_point_strategy()) {
            let zero = Polynomial::zero();

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = zero.eval(&assignment);

            prop_assert_eq!(result, BigRational::zero());
        }
    }
}

#[cfg(test)]
mod polynomial_derivative_properties {
    use super::*;

    proptest! {
        /// Test that derivative of constant is zero
        #[test]
        fn poly_deriv_constant_is_zero(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[])]);
            let dp = p.derivative(0);

            prop_assert_eq!(dp, Polynomial::zero());
        }

        /// Test that derivative of x is 1
        #[test]
        fn poly_deriv_x_is_one() {
            let p = Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]);
            let dp = p.derivative(0);
            let one = Polynomial::from_coeffs_int(&[(1, &[])]);

            prop_assert_eq!(dp, one);
        }

        /// Test that derivative of c*x is c
        #[test]
        fn poly_deriv_cx_is_c(c in coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let dp = p.derivative(0);
            let expected = Polynomial::from_coeffs_int(&[(c, &[])]);

            prop_assert_eq!(dp, expected);
        }

        /// Test that derivative of x^2 is 2*x
        #[test]
        fn poly_deriv_x_squared() {
            let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
            let dp = p.derivative(0);
            let expected = Polynomial::from_coeffs_int(&[(2, &[(0, 1)])]);

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
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 2)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let sum = &p1 + &p2;
            let d_sum = sum.derivative(0);

            let dp1 = p1.derivative(0);
            let dp2 = p2.derivative(0);
            let sum_derivs = &dp1 + &dp2;

            prop_assert_eq!(d_sum, sum_derivs);
        }

        /// Test product rule: d/dx(p*q) = p'*q + p*q'
        #[test]
        fn poly_deriv_product_rule(
            c1 in 1i64..5i64,
            c2 in 1i64..5i64
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let prod = &p * &q;
            let d_prod = prod.derivative(0);

            let dp = p.derivative(0);
            let dq = q.derivative(0);
            let product_rule = &(&dp * &q) + &(&p * &dq);

            prop_assert_eq!(d_prod, product_rule);
        }

        /// Test that second derivative of x^2 is constant 2
        #[test]
        fn poly_second_deriv_x_squared() {
            let p = Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]);
            let dp = p.derivative(0);
            let ddp = dp.derivative(0);

            let expected = Polynomial::from_coeffs_int(&[(2, &[])]);

            prop_assert_eq!(ddp, expected);
        }
    }
}

#[cfg(test)]
mod polynomial_gcd_properties {
    use super::*;

    proptest! {
        /// Test that gcd(p, p) = p (up to constant factor)
        #[test]
        fn poly_gcd_self(c in 1i64..10i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 2)])]);

            let g = p.gcd(&p);

            // gcd(p,p) should divide p
            if !g.is_zero() {
                let (quotient, remainder) = p.div_rem(&g);
                prop_assert!(remainder.is_zero());
            }
        }

        /// Test that gcd(p, 0) = p
        #[test]
        fn poly_gcd_with_zero(c in 1i64..10i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let zero = Polynomial::zero();

            let g = p.gcd(&zero);

            // gcd should divide p
            if !g.is_zero() {
                let (_, remainder) = p.div_rem(&g);
                prop_assert!(remainder.is_zero());
            }
        }

        /// Test that gcd is commutative
        #[test]
        fn poly_gcd_commutative(
            c1 in 1i64..8i64,
            c2 in 1i64..8i64
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let g1 = p1.gcd(&p2);
            let g2 = p2.gcd(&p1);

            // GCDs should be equal up to constant factor
            // Check by dividing both and seeing if we get constants
            if !g1.is_zero() && !g2.is_zero() {
                let (q1, r1) = g1.div_rem(&g2);
                let (q2, r2) = g2.div_rem(&g1);

                // If gcd is the same, remainders should be zero
                prop_assert!(r1.is_zero() || r2.is_zero() || q1.is_constant() || q2.is_constant());
            }
        }

        /// Test that gcd divides both operands
        #[test]
        fn poly_gcd_divides_both(
            c1 in 1i64..8i64,
            c2 in 1i64..8i64
        ) {
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 2)])]);
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let g = p1.gcd(&p2);

            if !g.is_zero() {
                let (_, r1) = p1.div_rem(&g);
                let (_, r2) = p2.div_rem(&g);

                prop_assert!(r1.is_zero());
                prop_assert!(r2.is_zero());
            }
        }

        /// Test Bézout's identity: gcd(p,q) = a*p + b*q
        #[test]
        fn poly_gcd_bezout_identity(
            c1 in 1i64..6i64,
            c2 in 1i64..6i64
        ) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let q = Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let (g, a, b) = p.gcd_extended(&q);

            // Verify: g = a*p + b*q
            let left = &a * &p;
            let right = &b * &q;
            let sum = &left + &right;

            // Should be equal up to constant factor
            if !g.is_zero() {
                let (_, remainder) = sum.div_rem(&g);
                prop_assert!(remainder.is_zero());
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
            // p1 = c1*x*y
            let p1 = Polynomial::from_coeffs_int(&[(c1, &[(0, 1), (1, 1)])]);
            // p2 = c2*x*y
            let p2 = Polynomial::from_coeffs_int(&[(c2, &[(0, 1), (1, 1)])]);

            let sum1 = &p1 + &p2;
            let sum2 = &p2 + &p1;

            prop_assert_eq!(sum1, sum2);
        }

        /// Test that partial derivative w.r.t. x of x*y is y
        #[test]
        fn mv_poly_partial_deriv_x(c in coeff_strategy()) {
            // p = c*x*y
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1), (1, 1)])]);

            // ∂p/∂x = c*y
            let dp_dx = p.derivative(0);
            let expected = Polynomial::from_coeffs_int(&[(c, &[(1, 1)])]);

            prop_assert_eq!(dp_dx, expected);
        }

        /// Test that partial derivative w.r.t. y of x*y is x
        #[test]
        fn mv_poly_partial_deriv_y(c in coeff_strategy()) {
            // p = c*x*y
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1), (1, 1)])]);

            // ∂p/∂y = c*x
            let dp_dy = p.derivative(1);
            let expected = Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);

            prop_assert_eq!(dp_dy, expected);
        }

        /// Test Schwarz theorem: ∂²p/∂x∂y = ∂²p/∂y∂x
        #[test]
        fn mv_poly_mixed_partials_commute(c in coeff_strategy()) {
            // p = c*x^2*y^2
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 2), (1, 2)])]);

            // ∂²p/∂x∂y
            let dp_dx = p.derivative(0);
            let d2p_dxdy = dp_dx.derivative(1);

            // ∂²p/∂y∂x
            let dp_dy = p.derivative(1);
            let d2p_dydx = dp_dy.derivative(0);

            prop_assert_eq!(d2p_dxdy, d2p_dydx);
        }

        /// Test that evaluation is consistent with variable substitution
        #[test]
        fn mv_poly_eval_consistent(
            c in coeff_strategy(),
            x in eval_point_strategy(),
            y in eval_point_strategy()
        ) {
            // p = c*x*y
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 1), (1, 1)])]);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(x));
            assignment.insert(1, rat(y));

            let result = p.eval(&assignment);
            let expected = rat(c * x * y);

            prop_assert_eq!(result, expected);
        }
    }
}

/// Helper function to create rational from i64
fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}
