//! Property-based tests for mathematical operations.
//!
//! These tests use proptest to verify mathematical properties hold
//! for a wide range of inputs.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::*;
use proptest::prelude::*;

// Helper to create rationals
fn rat(n: i64) -> BigRational {
    BigRational::from_integer(BigInt::from(n))
}

// Strategy for generating reasonable BigInts
fn bigint_strategy() -> impl Strategy<Value = BigInt> {
    (-1000i64..1000i64).prop_map(BigInt::from)
}

// Strategy for generating non-zero BigInts
fn bigint_nonzero_strategy() -> impl Strategy<Value = BigInt> {
    (-1000i64..1000i64)
        .prop_filter("must be non-zero", |&x| x != 0)
        .prop_map(BigInt::from)
}

// Strategy for generating BigRationals
fn bigrational_strategy() -> impl Strategy<Value = BigRational> {
    (bigint_strategy(), bigint_nonzero_strategy()).prop_map(|(n, d)| BigRational::new(n, d))
}

#[cfg(test)]
mod rational_properties {
    use super::*;

    proptest! {
        #[test]
        fn gcd_commutative(a in bigint_strategy(), b in bigint_strategy()) {
            let gcd_ab = rational::gcd_bigint(a.clone(), b.clone());
            let gcd_ba = rational::gcd_bigint(b.clone(), a.clone());
            prop_assert_eq!(gcd_ab, gcd_ba);
        }

        #[test]
        fn gcd_associative(a in bigint_strategy(), b in bigint_strategy(), c in bigint_strategy()) {
            let gcd_ab_c = rational::gcd_bigint(
                rational::gcd_bigint(a.clone(), b.clone()),
                c.clone()
            );
            let gcd_a_bc = rational::gcd_bigint(
                a.clone(),
                rational::gcd_bigint(b.clone(), c.clone())
            );
            prop_assert_eq!(gcd_ab_c, gcd_a_bc);
        }

        #[test]
        fn gcd_binary_equals_euclidean(a in bigint_strategy(), b in bigint_strategy()) {
            let gcd_euclidean = rational::gcd_bigint(a.clone(), b.clone());
            let gcd_binary = rational::gcd_binary(a.clone(), b.clone());
            prop_assert_eq!(gcd_euclidean, gcd_binary);
        }

        #[test]
        fn gcd_extended_property(a in bigint_strategy(), b in bigint_strategy()) {
            let (gcd, x, y) = rational::gcd_extended(a.clone(), b.clone());
            // Verify Bézout's identity: gcd = a*x + b*y
            let result = &a * &x + &b * &y;
            prop_assert_eq!(result, gcd);
        }

        #[test]
        fn floor_ceil_consistency(r in bigrational_strategy()) {
            let f = rational::floor(&r);
            let c = rational::ceil(&r);

            // floor <= rational <= ceil
            prop_assert!(BigRational::from_integer(f.clone()) <= r);
            prop_assert!(r <= BigRational::from_integer(c.clone()));

            // ceil - floor <= 1
            prop_assert!(&c - &f <= BigInt::one());
        }

        #[test]
        fn abs_idempotent(r in bigrational_strategy()) {
            let abs1 = rational::abs(&r);
            let abs2 = rational::abs(&abs1);
            prop_assert_eq!(abs1, abs2);
        }

        #[test]
        fn abs_non_negative(r in bigrational_strategy()) {
            let abs_r = rational::abs(&r);
            prop_assert!(abs_r >= BigRational::zero());
        }
    }
}

#[cfg(test)]
mod interval_properties {
    use super::*;

    proptest! {
        #[test]
        fn intersection_commutative(
            a1 in (-100i64..100i64),
            a2 in (-100i64..100i64),
            b1 in (-100i64..100i64),
            b2 in (-100i64..100i64),
        ) {
            let a_lo = a1.min(a2);
            let a_hi = a1.max(a2);
            let b_lo = b1.min(b2);
            let b_hi = b1.max(b2);

            let i1 = interval::Interval::closed(rat(a_lo), rat(a_hi));
            let i2 = interval::Interval::closed(rat(b_lo), rat(b_hi));

            let int1 = i1.intersect(&i2);
            let int2 = i2.intersect(&i1);

            prop_assert_eq!(int1, int2);
        }

        #[test]
        fn interval_contains_bounds(
            a in (-100i64..100i64),
            b in (-100i64..100i64),
        ) {
            let lo = a.min(b);
            let hi = a.max(b);

            let interval = interval::Interval::closed(rat(lo), rat(hi));

            prop_assert!(interval.contains(&rat(lo)));
            prop_assert!(interval.contains(&rat(hi)));
        }

        #[test]
        fn interval_add_contains_sum(
            a1 in (-50i64..50i64),
            a2 in (-50i64..50i64),
            b1 in (-50i64..50i64),
            b2 in (-50i64..50i64),
        ) {
            let a_lo = a1.min(a2);
            let a_hi = a1.max(a2);
            let b_lo = b1.min(b2);
            let b_hi = b1.max(b2);

            let i1 = interval::Interval::closed(rat(a_lo), rat(a_hi));
            let i2 = interval::Interval::closed(rat(b_lo), rat(b_hi));

            let sum_interval = i1.add(&i2);

            // The sum interval should contain lo+lo and hi+hi
            prop_assert!(sum_interval.contains(&rat(a_lo + b_lo)));
            prop_assert!(sum_interval.contains(&rat(a_hi + b_hi)));
        }
    }
}

#[cfg(test)]
mod polynomial_properties {
    use super::*;

    proptest! {
        #[test]
        fn polynomial_add_commutative(c1 in -10i64..10i64, c2 in -10i64..10i64) {
            let p1 = polynomial::Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = polynomial::Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let sum1 = &p1 + &p2;
            let sum2 = &p2 + &p1;

            prop_assert_eq!(sum1, sum2);
        }

        #[test]
        fn polynomial_add_associative(c1 in -10i64..10i64, c2 in -10i64..10i64, c3 in -10i64..10i64) {
            let p1 = polynomial::Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = polynomial::Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);
            let p3 = polynomial::Polynomial::from_coeffs_int(&[(c3, &[(0, 1)])]);

            let sum1 = &(&p1 + &p2) + &p3;
            let sum2 = &p1 + &(&p2 + &p3);

            prop_assert_eq!(sum1, sum2);
        }

        #[test]
        fn polynomial_mul_commutative(c1 in -10i64..10i64, c2 in -10i64..10i64) {
            let p1 = polynomial::Polynomial::from_coeffs_int(&[(c1, &[(0, 1)])]);
            let p2 = polynomial::Polynomial::from_coeffs_int(&[(c2, &[(0, 1)])]);

            let prod1 = &p1 * &p2;
            let prod2 = &p2 * &p1;

            prop_assert_eq!(prod1, prod2);
        }

        #[test]
        fn polynomial_zero_identity(c in -10i64..10i64) {
            let p = polynomial::Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let zero = polynomial::Polynomial::zero();

            let sum = &p + &zero;

            prop_assert_eq!(sum, p);
        }

        #[test]
        fn polynomial_eval_linear(c in -10i64..10i64, x in -10i64..10i64) {
            // p(x) = c*x should evaluate to c*x at x
            let p = polynomial::Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);

            let mut assignment = rustc_hash::FxHashMap::default();
            assignment.insert(0, rat(x));

            let result = p.eval(&assignment);
            let expected = rat(c * x);

            prop_assert_eq!(result, expected);
        }

        #[test]
        fn polynomial_derivative_constant(c in -10i64..10i64) {
            // Derivative of constant is zero
            let p = polynomial::Polynomial::from_coeffs_int(&[(c, &[])]);
            let dp = p.derivative(0);

            prop_assert_eq!(dp, polynomial::Polynomial::zero());
        }

        #[test]
        fn polynomial_derivative_linear(c in -10i64..10i64) {
            // Derivative of c*x is c
            let p = polynomial::Polynomial::from_coeffs_int(&[(c, &[(0, 1)])]);
            let dp = p.derivative(0);
            let expected = polynomial::Polynomial::from_coeffs_int(&[(c, &[])]);

            prop_assert_eq!(dp, expected);
        }
    }
}

#[cfg(test)]
mod delta_rational_properties {
    use super::*;

    proptest! {
        #[test]
        fn delta_rational_ordering(a in -100i64..100i64, d in -10i64..10i64) {
            let dr1 = delta_rational::DeltaRational::new(rat(a), d);
            let dr2 = delta_rational::DeltaRational::new(rat(a), d);

            prop_assert_eq!(dr1, dr2);
        }

        #[test]
        fn delta_rational_positive_delta(a in -100i64..100i64) {
            let dr_zero = delta_rational::DeltaRational::from_rational(rat(a));
            let dr_pos = delta_rational::DeltaRational::new(rat(a), 1);

            // a + delta > a
            prop_assert!(dr_pos > dr_zero);
        }

        #[test]
        fn delta_rational_add_commutative(
            a in -50i64..50i64,
            b in -50i64..50i64,
            da in -5i64..5i64,
            db in -5i64..5i64,
        ) {
            let dr1 = delta_rational::DeltaRational::new(rat(a), da);
            let dr2 = delta_rational::DeltaRational::new(rat(b), db);

            let sum1 = &dr1 + &dr2;
            let sum2 = &dr2 + &dr1;

            prop_assert_eq!(sum1, sum2);
        }
    }
}

#[cfg(test)]
mod number_theory_properties {
    use super::*;

    proptest! {
        #[test]
        fn factorial_monotonic(n in 1u32..15u32) {
            // n! < (n+1)!
            let fact_n = rational::factorial(n);
            let fact_n1 = rational::factorial(n + 1);

            prop_assert!(fact_n < fact_n1);
        }

        #[test]
        fn binomial_symmetry(n in 5u32..20u32, k in 1u32..5u32) {
            // C(n,k) = C(n,n-k)
            if k <= n {
                let c1 = rational::binomial(n, k);
                let c2 = rational::binomial(n, n - k);

                prop_assert_eq!(c1, c2);
            }
        }

        #[test]
        fn euler_totient_prime(p in 2u32..100u32) {
            // For prime p: φ(p) = p - 1
            let p_big = BigInt::from(p);
            if rational::is_prime(&p_big, 20) {
                let phi = rational::euler_totient(&p_big);
                let expected = p_big - BigInt::one();

                prop_assert_eq!(phi, expected);
            }
        }

        #[test]
        fn divisor_count_positive(n in 1u32..100u32) {
            // τ(n) >= 2 for n > 1
            let n_big = BigInt::from(n);
            let tau = rational::divisor_count(&n_big);

            if n > 1 {
                prop_assert!(tau >= BigInt::from(2));
            }
        }

        #[test]
        fn mobius_bounded(n in 1u32..100u32) {
            // μ(n) ∈ {-1, 0, 1}
            let n_big = BigInt::from(n);
            let mu = rational::mobius(&n_big);

            prop_assert!(mu == -1 || mu == 0 || mu == 1);
        }
    }
}
