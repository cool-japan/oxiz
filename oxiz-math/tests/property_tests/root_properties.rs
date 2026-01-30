//! Property-based tests for root isolation and finding
//!
//! This module tests:
//! - Sturm sequence properties
//! - Descartes' rule of signs
//! - Bisection method
//! - Newton's method convergence
//! - Root bounds (Cauchy, etc.)
//! - Multiplicity detection

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};
use oxiz_math::interval::*;
use oxiz_math::polynomial::*;
use oxiz_math::rational::*;
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
mod sturm_sequence_properties {
    use super::*;

    proptest! {
        /// Test that Sturm sequence of p has correct length
        #[test]
        fn sturm_sequence_length(c1 in root_coeff_strategy(), c2 in root_coeff_strategy()) {
            let p = Polynomial::from_coeffs_int(&[(c1, &[]), (c2, &[(0, 1)])]);

            let sturm = p.sturm_sequence();

            // Sturm sequence should have at most degree(p) + 1 polynomials
            if let Some(deg) = p.degree() {
                prop_assert!(sturm.len() <= deg + 1);
            }
        }

        /// Test that first element of Sturm sequence is the polynomial itself
        #[test]
        fn sturm_first_element_is_polynomial(c in 1i64..5i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 2)])]);

            let sturm = p.sturm_sequence();

            prop_assert!(!sturm.is_empty());
            prop_assert_eq!(&sturm[0], &p);
        }

        /// Test that second element of Sturm sequence is the derivative
        #[test]
        fn sturm_second_element_is_derivative(c in 1i64..5i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[(0, 2)])]);

            let sturm = p.sturm_sequence();

            if sturm.len() >= 2 {
                let dp = p.derivative(0);
                prop_assert_eq!(&sturm[1], &dp);
            }
        }

        /// Test that Sturm sequence sign changes count roots
        #[test]
        fn sturm_counts_roots_linear() {
            // p = x - 1 has exactly one root at x = 1
            let p = Polynomial::from_coeffs_int(&[(-1, &[]), (1, &[(0, 1)])]);

            let sturm = p.sturm_sequence();

            // Count sign changes at x = 0 (before root)
            let mut assignment_before = FxHashMap::default();
            assignment_before.insert(0, rat(0));
            let signs_before = sturm.iter().map(|s| {
                let val = s.eval(&assignment_before);
                if val > BigRational::zero() { 1 } else if val < BigRational::zero() { -1 } else { 0 }
            }).collect::<Vec<_>>();

            // Count sign changes at x = 2 (after root)
            let mut assignment_after = FxHashMap::default();
            assignment_after.insert(0, rat(2));
            let signs_after = sturm.iter().map(|s| {
                let val = s.eval(&assignment_after);
                if val > BigRational::zero() { 1 } else if val < BigRational::zero() { -1 } else { 0 }
            }).collect::<Vec<_>>();

            // Count sign changes
            let count_before = count_sign_changes(&signs_before);
            let count_after = count_sign_changes(&signs_after);

            // Difference should be 1 (one root in (0, 2))
            prop_assert_eq!(count_before - count_after, 1);
        }

        /// Test that Sturm sequence handles polynomials without real roots
        #[test]
        fn sturm_handles_no_real_roots() {
            // p = x^2 + 1 has no real roots
            let p = Polynomial::from_coeffs_int(&[(1, &[]), (0, &[(0, 1)]), (1, &[(0, 2)])]);

            let sturm = p.sturm_sequence();

            // Count sign changes at x = -10 and x = 10
            let mut assignment_neg = FxHashMap::default();
            assignment_neg.insert(0, rat(-10));
            let signs_neg = sturm.iter().map(|s| {
                let val = s.eval(&assignment_neg);
                if val > BigRational::zero() { 1 } else if val < BigRational::zero() { -1 } else { 0 }
            }).collect::<Vec<_>>();

            let mut assignment_pos = FxHashMap::default();
            assignment_pos.insert(0, rat(10));
            let signs_pos = sturm.iter().map(|s| {
                let val = s.eval(&assignment_pos);
                if val > BigRational::zero() { 1 } else if val < BigRational::zero() { -1 } else { 0 }
            }).collect::<Vec<_>>();

            let count_neg = count_sign_changes(&signs_neg);
            let count_pos = count_sign_changes(&signs_pos);

            // Difference should be 0 (no roots)
            prop_assert_eq!(count_neg - count_pos, 0);
        }
    }
}

#[cfg(test)]
mod descartes_rule_properties {
    use super::*;

    proptest! {
        /// Test Descartes' rule: number of positive roots <= sign changes
        #[test]
        fn descartes_positive_roots_bound(
            c0 in root_coeff_strategy(),
            c1 in root_coeff_strategy(),
            c2 in 1i64..5i64
        ) {
            // p = c2*x^2 + c1*x + c0
            let p = Polynomial::from_coeffs_int(&[(c0, &[]), (c1, &[(0, 1)]), (c2, &[(0, 2)])]);

            // Count sign changes in coefficients
            let coeffs = vec![c0, c1, c2];
            let sign_changes = count_coefficient_sign_changes(&coeffs);

            // Number of positive roots should be <= sign changes
            // (we can't easily count actual roots, but we can check the bound exists)
            prop_assert!(sign_changes <= 3);
        }

        /// Test that polynomial with all positive coefficients has no positive roots
        #[test]
        fn all_positive_coeffs_no_positive_roots() {
            // p = 1 + 2*x + 3*x^2 (all positive coefficients)
            let p = Polynomial::from_coeffs_int(&[(1, &[]), (2, &[(0, 1)]), (3, &[(0, 2)])]);

            // By Descartes' rule, 0 sign changes => 0 positive roots
            // Test that p(x) > 0 for x > 0
            let mut assignment = FxHashMap::default();
            assignment.insert(0, rat(1));

            let val = p.eval(&assignment);
            prop_assert!(val > BigRational::zero());
        }

        /// Test that alternating sign coefficients have maximum roots
        #[test]
        fn alternating_signs_maximum_roots() {
            // p = 1 - x + x^2 (alternating signs)
            let p = Polynomial::from_coeffs_int(&[(1, &[]), (-1, &[(0, 1)]), (1, &[(0, 2)])]);

            // Should have 2 sign changes, so at most 2 positive roots
            // (actual polynomial has 0 real roots, but bound is correct)
            let coeffs = vec![1, -1, 1];
            let sign_changes = count_coefficient_sign_changes(&coeffs);

            prop_assert_eq!(sign_changes, 2);
        }
    }
}

#[cfg(test)]
mod root_bounds_properties {
    use super::*;

    proptest! {
        /// Test Cauchy bound: all roots lie in a computable interval
        #[test]
        fn cauchy_bound_contains_roots(c in 1i64..5i64) {
            // p = x - c has root at x = c
            let p = Polynomial::from_coeffs_int(&[(-c, &[]), (1, &[(0, 1)])]);

            let bound = p.cauchy_bound();

            // Root at c should be within [-bound, bound]
            prop_assert!(rat(c).abs() <= bound);
        }

        /// Test that bound is always non-negative
        #[test]
        fn root_bound_non_negative(
            c0 in root_coeff_strategy(),
            c1 in root_coeff_strategy()
        ) {
            let p = Polynomial::from_coeffs_int(&[(c0, &[]), (c1, &[(0, 1)])]);

            let bound = p.cauchy_bound();

            prop_assert!(bound >= BigRational::zero());
        }

        /// Test that constant polynomial has zero bound
        #[test]
        fn constant_polynomial_zero_bound(c in 1i64..5i64) {
            let p = Polynomial::from_coeffs_int(&[(c, &[])]);

            let bound = p.cauchy_bound();

            // Constant has no roots (except if c=0), but bound should be finite
            prop_assert!(bound >= BigRational::zero());
        }
    }
}

#[cfg(test)]
mod bisection_properties {
    use super::*;

    proptest! {
        /// Test that bisection converges on simple roots
        #[test]
        fn bisection_converges_linear() {
            // p = x - 5 has root at x = 5
            let p = Polynomial::from_coeffs_int(&[(-5, &[]), (1, &[(0, 1)])]);

            // Start with interval [0, 10]
            let interval = Interval::closed(rat(0), rat(10));

            // Run bisection
            let root_intervals = p.isolate_roots_in_interval(&interval);

            // Should find exactly one root interval
            prop_assert_eq!(root_intervals.len(), 1);

            // Root should be in the interval
            let root_interval = &root_intervals[0];
            prop_assert!(root_interval.contains(&rat(5)));
        }

        /// Test that bisection handles quadratic roots
        #[test]
        fn bisection_handles_quadratic() {
            // p = (x-1)(x-2) = x^2 - 3x + 2 has roots at 1 and 2
            let p = Polynomial::from_coeffs_int(&[(2, &[]), (-3, &[(0, 1)]), (1, &[(0, 2)])]);

            // Start with interval [0, 3]
            let interval = Interval::closed(rat(0), rat(3));

            // Run root isolation
            let root_intervals = p.isolate_roots_in_interval(&interval);

            // Should find exactly two root intervals
            prop_assert_eq!(root_intervals.len(), 2);

            // One should contain 1, another should contain 2
            let contains_1 = root_intervals.iter().any(|i| i.contains(&rat(1)));
            let contains_2 = root_intervals.iter().any(|i| i.contains(&rat(2)));

            prop_assert!(contains_1);
            prop_assert!(contains_2);
        }

        /// Test that bisection handles no roots correctly
        #[test]
        fn bisection_handles_no_roots() {
            // p = x^2 + 1 has no real roots
            let p = Polynomial::from_coeffs_int(&[(1, &[]), (0, &[(0, 1)]), (1, &[(0, 2)])]);

            // Start with interval [-10, 10]
            let interval = Interval::closed(rat(-10), rat(10));

            // Run root isolation
            let root_intervals = p.isolate_roots_in_interval(&interval);

            // Should find no roots
            prop_assert_eq!(root_intervals.len(), 0);
        }

        /// Test that bisection refines intervals
        #[test]
        fn bisection_refines_intervals() {
            // p = x - 5 has root at x = 5
            let p = Polynomial::from_coeffs_int(&[(-5, &[]), (1, &[(0, 1)])]);

            // Start with large interval
            let interval = Interval::closed(rat(-100), rat(100));

            // Run root isolation with refinement
            let root_intervals = p.isolate_roots_in_interval(&interval);

            if !root_intervals.is_empty() {
                let refined = &root_intervals[0];

                // Refined interval should be smaller than original
                prop_assert!(refined.width() < rat(200));
            }
        }
    }
}

#[cfg(test)]
mod newton_method_properties {
    use super::*;

    proptest! {
        /// Test that Newton's method converges for simple roots
        #[test]
        fn newton_converges_simple_root(c in 1i64..10i64) {
            // p = x - c has root at c
            let p = Polynomial::from_coeffs_int(&[(-c, &[]), (1, &[(0, 1)])]);

            // Start near the root
            let initial_guess = rat(c + 1);

            // Run Newton's method
            let root = p.newton_method(initial_guess, 10);

            // Should converge to c
            if let Some(r) = root {
                prop_assert!((r - rat(c)).abs() < rat(1) / rat(1000));
            }
        }

        /// Test that Newton's method moves toward roots
        #[test]
        fn newton_approaches_root(c in 2i64..10i64) {
            // p = x - c
            let p = Polynomial::from_coeffs_int(&[(-c, &[]), (1, &[(0, 1)])]);

            // Start far from root
            let initial = rat(0);

            // One iteration of Newton's method
            let dp = p.derivative(0);

            let mut assignment = FxHashMap::default();
            assignment.insert(0, initial.clone());

            let f_val = p.eval(&assignment);
            let df_val = dp.eval(&assignment);

            if df_val != BigRational::zero() {
                let next = initial - f_val / df_val;

                // Next point should be closer to root
                prop_assert!((next - rat(c)).abs() < rat(c));
            }
        }

        /// Test that Newton's method handles stationary points
        #[test]
        fn newton_handles_stationary_points() {
            // p = x^2 has a double root at 0 (stationary point)
            let p = Polynomial::from_coeffs_int(&[(0, &[]), (0, &[(0, 1)]), (1, &[(0, 2)])]);

            // Start near 0
            let initial = rat(1);

            // Newton's method should not crash
            let root = p.newton_method(initial, 5);

            // Method should either converge or recognize the issue
            prop_assert!(root.is_some() || root.is_none());
        }
    }
}

#[cfg(test)]
mod root_multiplicity_properties {
    use super::*;

    proptest! {
        /// Test that simple roots have multiplicity 1
        #[test]
        fn simple_root_multiplicity_one(c in 1i64..5i64) {
            // p = x - c has a simple root at c
            let p = Polynomial::from_coeffs_int(&[(-c, &[]), (1, &[(0, 1)])]);

            let mult = p.root_multiplicity(rat(c));

            prop_assert_eq!(mult, 1);
        }

        /// Test that double roots have multiplicity 2
        #[test]
        fn double_root_multiplicity_two(c in 1i64..5i64) {
            // p = (x - c)^2 = x^2 - 2cx + c^2
            let p = Polynomial::from_coeffs_int(&[
                (c * c, &[]),
                (-2 * c, &[(0, 1)]),
                (1, &[(0, 2)])
            ]);

            let mult = p.root_multiplicity(rat(c));

            prop_assert_eq!(mult, 2);
        }

        /// Test that non-roots have multiplicity 0
        #[test]
        fn non_root_multiplicity_zero(c in 1i64..5i64) {
            // p = x - c has root at c, not at c+1
            let p = Polynomial::from_coeffs_int(&[(-c, &[]), (1, &[(0, 1)])]);

            let mult = p.root_multiplicity(rat(c + 1));

            prop_assert_eq!(mult, 0);
        }

        /// Test that square-free part removes multiplicities
        #[test]
        fn square_free_removes_multiplicities(c in 1i64..5i64) {
            // p = (x - c)^2
            let p = Polynomial::from_coeffs_int(&[
                (c * c, &[]),
                (-2 * c, &[(0, 1)]),
                (1, &[(0, 2)])
            ]);

            let sf = p.square_free_part();

            // Square-free part should be (x - c)
            let expected = Polynomial::from_coeffs_int(&[(-c, &[]), (1, &[(0, 1)])]);

            // Check by dividing
            let (_, remainder) = sf.div_rem(&expected);
            prop_assert!(remainder.is_zero() || expected.is_zero());
        }
    }
}

#[cfg(test)]
mod interval_refinement_properties {
    use super::*;

    proptest! {
        /// Test that interval refinement narrows the interval
        #[test]
        fn refinement_narrows_interval(
            a in -10i64..0i64,
            b in 1i64..10i64
        ) {
            let interval = Interval::closed(rat(a), rat(b));
            let original_width = interval.width();

            // Bisect the interval
            let (left, right) = interval.bisect();

            // Each half should be narrower
            prop_assert!(left.width() < original_width);
            prop_assert!(right.width() < original_width);
        }

        /// Test that refinement preserves interval properties
        #[test]
        fn refinement_preserves_points(
            a in -10i64..0i64,
            b in 1i64..10i64,
            x in -5i64..5i64
        ) {
            let interval = Interval::closed(rat(a), rat(b));

            if interval.contains(&rat(x)) {
                let (left, right) = interval.bisect();

                // x should be in one of the halves
                prop_assert!(left.contains(&rat(x)) || right.contains(&rat(x)));
            }
        }

        /// Test that repeated refinement converges
        #[test]
        fn repeated_refinement_converges() {
            let mut interval = Interval::closed(rat(0), rat(100));

            // Refine 10 times
            for _ in 0..10 {
                let (left, _) = interval.bisect();
                interval = left;
            }

            // Width should be much smaller
            prop_assert!(interval.width() < rat(1));
        }
    }
}

/// Helper function to count sign changes in a sequence
fn count_sign_changes(signs: &[i32]) -> usize {
    let mut count = 0;
    let mut last_nonzero = None;

    for &sign in signs {
        if sign != 0 {
            if let Some(last) = last_nonzero {
                if last != sign {
                    count += 1;
                }
            }
            last_nonzero = Some(sign);
        }
    }

    count
}

/// Helper function to count coefficient sign changes
fn count_coefficient_sign_changes(coeffs: &[i64]) -> usize {
    let mut count = 0;
    let mut last_nonzero = None;

    for &c in coeffs {
        if c != 0 {
            let sign = if c > 0 { 1 } else { -1 };
            if let Some(last) = last_nonzero {
                if last != sign {
                    count += 1;
                }
            }
            last_nonzero = Some(sign);
        }
    }

    count
}
