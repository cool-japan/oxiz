//! Edge case tests for mathematical operations.
//!
//! These tests focus on boundary conditions, special values,
//! and corner cases that might not be covered by regular tests.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::*;

fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

#[cfg(test)]
mod polynomial_edge_cases {
    use super::*;

    #[test]
    fn test_zero_polynomial_operations() {
        let zero = polynomial::Polynomial::zero();
        let p = polynomial::Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]);

        // Zero + p = p
        assert_eq!(&zero + &p, p);

        // Zero * p = Zero
        assert_eq!(&zero * &p, zero);

        // Derivative of zero is zero
        assert_eq!(zero.derivative(0), zero);
    }

    #[test]
    fn test_constant_polynomial_operations() {
        let c1 = polynomial::Polynomial::from_coeffs_int(&[(5, &[])]);
        let c2 = polynomial::Polynomial::from_coeffs_int(&[(3, &[])]);

        // Constant + Constant
        let sum = &c1 + &c2;
        assert_eq!(sum.total_degree(), 0);

        // Constant * Constant
        let prod = &c1 * &c2;
        assert_eq!(prod.total_degree(), 0);

        // Derivative of constant is zero
        assert_eq!(c1.derivative(0), polynomial::Polynomial::zero());
    }

    #[test]
    fn test_polynomial_multiplication_by_one() {
        let p =
            polynomial::Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (2, &[(0, 1)]), (1, &[])]);
        let one = polynomial::Polynomial::from_coeffs_int(&[(1, &[])]);

        let prod = &p * &one;
        assert_eq!(prod, p);
    }

    #[test]
    fn test_polynomial_gcd_with_zero() {
        let p = polynomial::Polynomial::from_coeffs_int(&[(1, &[(0, 1)])]);
        let zero = polynomial::Polynomial::zero();

        // gcd(p, 0) should be p (up to scalar)
        let g = p.gcd_univariate(&zero);
        assert_eq!(g.total_degree(), p.total_degree());
    }

    #[test]
    fn test_polynomial_eval_at_zero() {
        let p = polynomial::Polynomial::from_coeffs_int(&[
            (1, &[(0, 2)]), // x^2
            (2, &[(0, 1)]), // 2x
            (3, &[]),       // 3
        ]);

        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(0, rat(0));

        // p(0) = 3
        assert_eq!(p.eval(&assignment), rat(3));
    }

    #[test]
    fn test_multivariate_polynomial_single_variable() {
        // Create multivariate polynomial but only use one variable
        let p = polynomial::Polynomial::from_coeffs_int(&[(1, &[(0, 2)]), (2, &[(0, 1)])]);

        assert!(p.is_univariate());
        assert_eq!(p.max_var(), 0);
    }

    #[test]
    fn test_polynomial_high_degree() {
        // Test with a high degree polynomial
        let p = polynomial::Polynomial::from_coeffs_int(&[(1, &[(0, 100)])]);

        assert_eq!(p.total_degree(), 100);

        // Derivative should reduce degree
        let dp = p.derivative(0);
        assert_eq!(dp.total_degree(), 99);
    }
}

#[cfg(test)]
mod rational_edge_cases {
    use super::*;

    #[test]
    fn test_gcd_with_zero() {
        let a = BigInt::from(42);
        let zero = BigInt::zero();

        let g1 = rational::gcd_bigint(a.clone(), zero.clone());
        assert_eq!(g1, BigInt::from(42));

        let g2 = rational::gcd_bigint(zero.clone(), a.clone());
        assert_eq!(g2, BigInt::from(42));
    }

    #[test]
    fn test_gcd_with_one() {
        let a = BigInt::from(42);
        let one = BigInt::one();

        let g = rational::gcd_bigint(a, one);
        assert_eq!(g, BigInt::one());
    }

    #[test]
    fn test_gcd_extended_zero() {
        let a = BigInt::from(10);
        let zero = BigInt::zero();

        let (gcd, x, _y) = rational::gcd_extended(a.clone(), zero.clone());
        assert_eq!(gcd, BigInt::from(10));
        // Verify Bézout identity: gcd = a*x + 0*y
        assert_eq!(&a * &x, gcd);
    }

    #[test]
    fn test_floor_ceil_integers() {
        let n = rat(5);
        assert_eq!(rational::floor(&n), BigInt::from(5));
        assert_eq!(rational::ceil(&n), BigInt::from(5));
    }

    #[test]
    fn test_floor_ceil_negative() {
        let neg_half = BigRational::new(BigInt::from(-1), BigInt::from(2));

        // floor(-0.5) = -1
        assert_eq!(rational::floor(&neg_half), BigInt::from(-1));

        // ceil(-0.5) = 0
        assert_eq!(rational::ceil(&neg_half), BigInt::zero());
    }

    #[test]
    fn test_factorial_zero_and_one() {
        assert_eq!(rational::factorial(0), BigInt::one());
        assert_eq!(rational::factorial(1), BigInt::one());
        assert_eq!(rational::factorial(2), BigInt::from(2));
    }

    #[test]
    fn test_binomial_edge_cases() {
        // C(n, 0) = 1
        assert_eq!(rational::binomial(10, 0), BigInt::one());

        // C(n, n) = 1
        assert_eq!(rational::binomial(10, 10), BigInt::one());

        // C(n, k) = 0 when k > n
        assert_eq!(rational::binomial(5, 10), BigInt::zero());
    }

    #[test]
    fn test_is_prime_small_numbers() {
        assert!(!rational::is_prime(&BigInt::zero(), 20));
        assert!(!rational::is_prime(&BigInt::one(), 20));
        assert!(rational::is_prime(&BigInt::from(2), 20));
        assert!(rational::is_prime(&BigInt::from(3), 20));
        assert!(!rational::is_prime(&BigInt::from(4), 20));
        assert!(rational::is_prime(&BigInt::from(5), 20));
    }

    #[test]
    fn test_divisor_count_one() {
        assert_eq!(rational::divisor_count(&BigInt::one()), BigInt::one());
    }

    #[test]
    fn test_divisor_sum_one() {
        assert_eq!(rational::divisor_sum(&BigInt::one()), BigInt::one());
    }

    #[test]
    fn test_mobius_one() {
        assert_eq!(rational::mobius(&BigInt::one()), 1);
    }

    #[test]
    fn test_euler_totient_one() {
        assert_eq!(rational::euler_totient(&BigInt::one()), BigInt::one());
    }
}

#[cfg(test)]
mod interval_edge_cases {
    use super::*;

    #[test]
    fn test_empty_interval_intersection() {
        let i1 = interval::Interval::closed(rat(1), rat(3));
        let i2 = interval::Interval::closed(rat(5), rat(7));

        let inter = i1.intersect(&i2);
        // Non-overlapping intervals should result in an empty interval
        assert!(inter.is_empty());
    }

    #[test]
    fn test_point_interval() {
        let point = interval::Interval::closed(rat(5), rat(5));

        assert!(point.contains(&rat(5)));
        assert!(!point.contains(&rat(4)));
        assert!(!point.contains(&rat(6)));
    }

    #[test]
    fn test_interval_contains_bounds() {
        let i = interval::Interval::closed(rat(1), rat(10));

        assert!(i.contains(&rat(1)));
        assert!(i.contains(&rat(10)));
        assert!(i.contains(&rat(5)));
    }

    #[test]
    fn test_interval_addition_with_zero() {
        let i = interval::Interval::closed(rat(1), rat(5));
        let zero = interval::Interval::closed(rat(0), rat(0));

        let sum = i.add(&zero);

        // Adding zero interval should preserve the interval
        assert!(sum.contains(&rat(1)));
        assert!(sum.contains(&rat(5)));
    }

    #[test]
    fn test_interval_multiplication_by_zero() {
        let i = interval::Interval::closed(rat(1), rat(5));
        let zero = interval::Interval::closed(rat(0), rat(0));

        let prod = i.mul(&zero);

        // Multiplying by zero should give zero
        assert!(prod.contains(&rat(0)));
    }

    #[test]
    fn test_interval_negative_bounds() {
        let i = interval::Interval::closed(rat(-5), rat(-1));

        assert!(i.contains(&rat(-3)));
        assert!(!i.contains(&rat(0)));
        assert!(!i.contains(&rat(-6)));
    }
}

#[cfg(test)]
mod delta_rational_edge_cases {
    use super::*;

    #[test]
    fn test_delta_rational_zero() {
        let zero = delta_rational::DeltaRational::from_rational(rat(0));
        let zero_plus_delta = delta_rational::DeltaRational::new(rat(0), 1);

        assert!(zero_plus_delta > zero);
    }

    #[test]
    fn test_delta_rational_negative() {
        let neg = delta_rational::DeltaRational::from_rational(rat(-5));
        let zero = delta_rational::DeltaRational::from_rational(rat(0));

        assert!(neg < zero);
    }

    #[test]
    fn test_delta_rational_addition_identity() {
        let dr = delta_rational::DeltaRational::new(rat(5), 2);
        let zero = delta_rational::DeltaRational::from_rational(rat(0));

        let sum = &dr + &zero;
        assert_eq!(sum, dr);
    }
}

#[cfg(test)]
mod matrix_edge_cases {
    use super::*;
    use matrix::Matrix;
    use num_rational::Rational64;

    #[test]
    fn test_matrix_1x1() {
        let m = Matrix::from_vec(1, 1, vec![Rational64::new(5, 1)]);

        assert_eq!(m.get(0, 0), Rational64::new(5, 1));
    }

    #[test]
    fn test_matrix_identity_2x2() {
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational64::new(1, 1),
                Rational64::new(0, 1),
                Rational64::new(0, 1),
                Rational64::new(1, 1),
            ],
        );

        // Gaussian elimination on identity should work
        let (_result, _rank) = m.gaussian_elimination();
        // Should complete successfully
    }

    #[test]
    fn test_matrix_all_zeros() {
        let m = Matrix::from_vec(
            2,
            2,
            vec![
                Rational64::zero(),
                Rational64::zero(),
                Rational64::zero(),
                Rational64::zero(),
            ],
        );

        // Gaussian elimination on zero matrix
        let _result = m.gaussian_elimination();
        // Should handle gracefully
    }
}

#[cfg(test)]
mod simplex_edge_cases {
    use super::*;

    #[test]
    fn test_empty_tableau() {
        let tableau = simplex::SimplexTableau::new();

        // Empty tableau should be feasible
        assert!(tableau.is_feasible());
    }

    #[test]
    fn test_single_variable() {
        let mut tableau = simplex::SimplexTableau::new();
        let x = tableau.fresh_var();

        // Should be able to add a variable
        assert_eq!(x, 0);
    }

    #[test]
    fn test_row_with_zero_coefficients() {
        let row = simplex::Row::from_expr(
            0,
            rat(5),
            rustc_hash::FxHashMap::default(), // No coefficients
        );

        let mut assignment = rustc_hash::FxHashMap::default();
        assignment.insert(1, rat(10));

        // Row with no coefficients should evaluate to constant
        assert_eq!(row.eval(&assignment), rat(5));
    }
}
