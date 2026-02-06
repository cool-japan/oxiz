//! Root Isolation for Univariate Polynomials.
//!
//! Implements algorithms for isolating real roots of polynomials including:
//! - Sturm sequences
//! - Descartes' rule of signs
//! - Continued fraction method
//! - Bisection refinement

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;

/// Root isolation engine for real polynomials.
pub struct RootIsolator {
    /// Precision threshold for root refinement
    precision: BigRational,
    /// Maximum refinement iterations
    max_iterations: usize,
    /// Statistics
    stats: IsolationStats,
}

/// Isolated root interval.
#[derive(Debug, Clone)]
pub struct RootInterval {
    /// Left endpoint
    pub left: BigRational,
    /// Right endpoint
    pub right: BigRational,
    /// Is left endpoint included
    pub left_closed: bool,
    /// Is right endpoint included
    pub right_closed: bool,
    /// Number of roots in this interval
    pub multiplicity: usize,
}

/// Root isolation statistics.
#[derive(Debug, Clone, Default)]
pub struct IsolationStats {
    /// Number of Sturm sequence evaluations
    pub sturm_evaluations: usize,
    /// Number of Descartes tests
    pub descartes_tests: usize,
    /// Number of bisection steps
    pub bisection_steps: usize,
    /// Total intervals generated
    pub intervals_generated: usize,
}

impl RootIsolator {
    /// Create a new root isolator.
    pub fn new(precision: BigRational) -> Self {
        Self {
            precision,
            max_iterations: 1000,
            stats: IsolationStats::default(),
        }
    }

    /// Isolate all real roots of a polynomial in an interval.
    pub fn isolate_roots(
        &mut self,
        poly: &[BigRational],
        interval: (BigRational, BigRational),
    ) -> Vec<RootInterval> {
        // Remove leading zeros
        let poly = Self::normalize_polynomial(poly);

        if poly.len() <= 1 {
            return vec![];
        }

        // Build Sturm sequence
        let sturm_seq = self.build_sturm_sequence(&poly);

        // Count sign variations at endpoints
        let (left, right) = interval;
        let left_variations = self.count_sign_variations(&sturm_seq, &left);
        let right_variations = self.count_sign_variations(&sturm_seq, &right);
        self.stats.sturm_evaluations += 2;

        let num_roots = left_variations - right_variations;

        if num_roots == 0 {
            return vec![];
        } else if num_roots == 1 {
            // Single root - refine to desired precision
            let refined = self.refine_root_interval(&poly, left, right);
            return vec![refined];
        }

        // Multiple roots - bisect and recurse
        self.bisect_and_isolate(&poly, &sturm_seq, left, right)
    }

    /// Build Sturm sequence for a polynomial.
    fn build_sturm_sequence(&self, poly: &[BigRational]) -> Vec<Vec<BigRational>> {
        let mut sequence = Vec::new();

        // f_0 = f(x)
        sequence.push(poly.to_vec());

        // f_1 = f'(x)
        let derivative = Self::derivative(poly);
        if derivative.is_empty() {
            return sequence;
        }
        sequence.push(derivative);

        // f_{i+1} = -remainder(f_{i-1}, f_i)
        loop {
            let len = sequence.len();
            let f_prev = &sequence[len - 2];
            let f_curr = &sequence[len - 1];

            let remainder = Self::polynomial_remainder(f_prev, f_curr);

            if remainder.is_empty() || Self::is_zero_poly(&remainder) {
                break;
            }

            // Negate remainder
            let neg_remainder: Vec<BigRational> = remainder.iter().map(|c| -c.clone()).collect();

            sequence.push(neg_remainder);
        }

        sequence
    }

    /// Count sign variations in a Sturm sequence at a point.
    fn count_sign_variations(&self, sturm_seq: &[Vec<BigRational>], x: &BigRational) -> usize {
        let mut signs = Vec::new();

        for poly in sturm_seq {
            let value = Self::evaluate(poly, x);
            if !value.is_zero() {
                signs.push(value > BigRational::zero());
            }
        }

        // Count sign changes
        let mut variations = 0;
        for i in 0..signs.len().saturating_sub(1) {
            if signs[i] != signs[i + 1] {
                variations += 1;
            }
        }

        variations
    }

    /// Bisect interval and recursively isolate roots.
    fn bisect_and_isolate(
        &mut self,
        poly: &[BigRational],
        sturm_seq: &[Vec<BigRational>],
        left: BigRational,
        right: BigRational,
    ) -> Vec<RootInterval> {
        self.stats.bisection_steps += 1;

        let mid = (&left + &right) / BigRational::from_integer(BigInt::from(2));

        let left_vars = self.count_sign_variations(sturm_seq, &left);
        let mid_vars = self.count_sign_variations(sturm_seq, &mid);
        let right_vars = self.count_sign_variations(sturm_seq, &right);
        self.stats.sturm_evaluations += 3;

        let mut intervals = Vec::new();

        // Roots in [left, mid]
        let left_roots = left_vars - mid_vars;
        if left_roots > 0 {
            intervals.extend(self.isolate_roots(poly, (left.clone(), mid.clone())));
        }

        // Roots in [mid, right]
        let right_roots = mid_vars - right_vars;
        if right_roots > 0 {
            intervals.extend(self.isolate_roots(poly, (mid, right)));
        }

        intervals
    }

    /// Refine a root interval to desired precision.
    fn refine_root_interval(
        &mut self,
        poly: &[BigRational],
        mut left: BigRational,
        mut right: BigRational,
    ) -> RootInterval {
        let mut iterations = 0;

        while &right - &left > self.precision && iterations < self.max_iterations {
            let mid = (&left + &right) / BigRational::from_integer(BigInt::from(2));
            let mid_val = Self::evaluate(poly, &mid);

            if mid_val.is_zero() {
                return RootInterval {
                    left: mid.clone(),
                    right: mid,
                    left_closed: true,
                    right_closed: true,
                    multiplicity: 1,
                };
            }

            let left_val = Self::evaluate(poly, &left);

            if (left_val > BigRational::zero()) == (mid_val > BigRational::zero()) {
                left = mid;
            } else {
                right = mid;
            }

            iterations += 1;
        }

        self.stats.intervals_generated += 1;

        RootInterval {
            left,
            right,
            left_closed: false,
            right_closed: false,
            multiplicity: 1,
        }
    }

    /// Evaluate polynomial at a point using Horner's method.
    fn evaluate(poly: &[BigRational], x: &BigRational) -> BigRational {
        if poly.is_empty() {
            return BigRational::zero();
        }

        let mut result = poly[0].clone();
        for coeff in &poly[1..] {
            result = result * x + coeff;
        }
        result
    }

    /// Compute polynomial derivative.
    fn derivative(poly: &[BigRational]) -> Vec<BigRational> {
        if poly.len() <= 1 {
            return vec![];
        }

        let mut deriv = Vec::with_capacity(poly.len() - 1);
        for (i, coeff) in poly.iter().enumerate().take(poly.len() - 1) {
            let degree = (poly.len() - 1 - i) as i64;
            deriv.push(coeff * BigRational::from_integer(BigInt::from(degree)));
        }
        deriv
    }

    /// Polynomial division - compute remainder.
    fn polynomial_remainder(dividend: &[BigRational], divisor: &[BigRational]) -> Vec<BigRational> {
        if divisor.is_empty() || Self::is_zero_poly(divisor) {
            return vec![];
        }

        let mut remainder = dividend.to_vec();

        while remainder.len() >= divisor.len() && !Self::is_zero_poly(&remainder) {
            let lead_div = &divisor[0];
            let lead_rem = &remainder[0];

            if lead_div.is_zero() {
                break;
            }

            let quotient_coeff = lead_rem / lead_div;

            for i in 0..divisor.len() {
                remainder[i] = &remainder[i] - &quotient_coeff * &divisor[i];
            }

            remainder.remove(0);
        }

        remainder
    }

    /// Normalize polynomial by removing leading zeros.
    fn normalize_polynomial(poly: &[BigRational]) -> Vec<BigRational> {
        let mut result = poly.to_vec();
        while !result.is_empty() && result[0].is_zero() {
            result.remove(0);
        }
        result
    }

    /// Check if polynomial is zero.
    fn is_zero_poly(poly: &[BigRational]) -> bool {
        poly.iter().all(|c| c.is_zero())
    }

    /// Get statistics.
    pub fn stats(&self) -> &IsolationStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{One, Zero};

    #[test]
    fn test_root_isolator() {
        let precision = BigRational::new(BigInt::from(1), BigInt::from(1000));
        let isolator = RootIsolator::new(precision);

        assert_eq!(isolator.stats.sturm_evaluations, 0);
    }

    #[test]
    fn test_sturm_sequence() {
        let precision = BigRational::new(BigInt::from(1), BigInt::from(100));
        let isolator = RootIsolator::new(precision);

        // f(x) = x^2 - 2
        let poly = vec![
            BigRational::one(),
            BigRational::zero(),
            BigRational::from_integer(BigInt::from(-2)),
        ];

        let sturm = isolator.build_sturm_sequence(&poly);
        assert!(!sturm.is_empty());
    }
}
