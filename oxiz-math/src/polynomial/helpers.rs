//! Helper functions for polynomial operations.

use super::types::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

type Polynomial = super::Polynomial;

/// Compute GCD of two BigInts using Euclidean algorithm.
pub(super) fn gcd_bigint(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let t = &a % &b;
        a = b;
        b = t;
    }
    a.abs()
}

/// Count sign variations in a sequence of polynomials evaluated at a point.
/// Used in Sturm's theorem for root counting.
pub(super) fn count_sign_variations(seq: &[Polynomial], var: Var, point: &BigRational) -> usize {
    let mut signs = Vec::new();

    for poly in seq {
        let val = poly.eval_at(var, point);
        let c = val.constant_term();
        if !c.is_zero() {
            signs.push(if c.is_positive() { 1 } else { -1 });
        }
    }

    // Count sign changes
    let mut variations = 0;
    for i in 1..signs.len() {
        if signs[i] != signs[i - 1] {
            variations += 1;
        }
    }

    variations
}

/// Compute Cauchy's bound for the absolute value of roots of a polynomial.
/// All real roots lie in the interval [-bound, bound].
/// Cauchy's root bound for a univariate polynomial.
/// Returns B such that all roots have absolute value <= B.
/// Bound: 1 + max(|a_i| / |a_n|) for i < n
pub(super) fn cauchy_root_bound(poly: &Polynomial, var: Var) -> BigRational {
    if poly.is_zero() {
        return BigRational::one();
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return BigRational::one();
    }

    let lc = poly.univ_coeff(var, deg);
    if lc.is_zero() {
        return BigRational::one();
    }

    let lc_abs = lc.abs();

    // Cauchy's bound: 1 + max(|a_i| / |a_n|) for i < n
    let mut max_ratio = BigRational::zero();
    for k in 0..deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            let ratio = coeff.abs() / &lc_abs;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
    }

    BigRational::one() + max_ratio
}

/// Fujiwara's root bound for a univariate polynomial.
/// Returns B such that all roots have absolute value <= B.
/// Generally tighter than Cauchy's bound.
///
/// Fujiwara bound: 2 * max(|a_i/a_n|^(1/(n-i))) for i < n
///
/// Reference: Fujiwara, "Über die obere Schranke des absoluten Betrages
/// der Wurzeln einer algebraischen Gleichung" (1916)
pub(super) fn fujiwara_root_bound(poly: &Polynomial, var: Var) -> BigRational {
    if poly.is_zero() {
        return BigRational::one();
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return BigRational::one();
    }

    let lc = poly.univ_coeff(var, deg);
    if lc.is_zero() {
        return BigRational::one();
    }

    let lc_abs = lc.abs();

    // For each coefficient a_i (i < n), compute |a_i/a_n|^(1/(n-i))
    // We approximate this using rational arithmetic
    let mut max_val = BigRational::zero();

    for k in 0..deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            let ratio = coeff.abs() / &lc_abs;
            let exp = deg - k;

            // Approximate ratio^(1/exp) using binary search
            // We want to find x such that x^exp ≈ ratio
            let approx_root = rational_nth_root_approx(&ratio, exp);

            if approx_root > max_val {
                max_val = approx_root;
            }
        }
    }

    // Fujiwara bound is 2 * max
    BigRational::from_integer(BigInt::from(2)) * max_val
}

/// Lagrange's root bound for positive roots of a univariate polynomial.
/// Returns B such that all positive roots are <= B.
///
/// Lagrange bound: max(|a_i/a_n|^(1/(n-i))) where a_i < 0 and i < n
pub(super) fn lagrange_positive_root_bound(poly: &Polynomial, var: Var) -> BigRational {
    if poly.is_zero() {
        return BigRational::one();
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return BigRational::one();
    }

    let lc = poly.univ_coeff(var, deg);
    if lc.is_zero() {
        return BigRational::one();
    }

    let lc_abs = lc.abs();

    // Find the largest |a_i/a_n|^(1/(n-i)) where a_i and a_n have opposite signs
    let mut max_val = BigRational::zero();
    let lc_positive = lc.is_positive();

    for k in 0..deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() && coeff.is_positive() != lc_positive {
            let ratio = coeff.abs() / &lc_abs;
            let exp = deg - k;

            let approx_root = rational_nth_root_approx(&ratio, exp);

            if approx_root > max_val {
                max_val = approx_root;
            }
        }
    }

    if max_val.is_zero() {
        // No negative coefficients found, bound is 1
        BigRational::one()
    } else {
        max_val
    }
}

/// Approximate the nth root of a rational number.
/// Uses binary search to find x such that x^n ≈ target.
/// Returns an upper bound approximation.
pub(super) fn rational_nth_root_approx(target: &BigRational, n: u32) -> BigRational {
    use crate::rational::pow_uint;

    if n == 0 {
        return BigRational::one();
    }
    if n == 1 {
        return target.clone();
    }
    if target.is_zero() {
        return BigRational::zero();
    }

    // Binary search for the nth root
    let mut low = BigRational::zero();
    let mut high = target.clone() + BigRational::one();

    // Limit iterations
    for _ in 0..100 {
        let mid = (&low + &high) / BigRational::from_integer(BigInt::from(2));
        let mid_pow_n = pow_uint(&mid, n);

        if &mid_pow_n == target {
            return mid;
        }

        if mid_pow_n < *target {
            low = mid;
        } else {
            high = mid.clone();
        }

        // Check convergence
        let diff = &high - &low;
        if diff < BigRational::new(BigInt::one(), BigInt::from(1000000)) {
            return high;
        }
    }

    high
}

/// Count sign variations in the coefficients of a univariate polynomial.
/// This is used for Descartes' rule of signs.
///
/// Descartes' rule of signs: The number of positive real roots of a polynomial
/// is either equal to the number of sign variations in the coefficient sequence,
/// or is less than it by a positive even integer.
pub(super) fn count_coefficient_sign_variations(poly: &Polynomial, var: Var) -> usize {
    if poly.is_zero() {
        return 0;
    }

    let deg = poly.degree(var);
    if deg == 0 {
        return 0;
    }

    // Collect non-zero coefficients in order
    let mut coeffs = Vec::new();
    for k in 0..=deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            coeffs.push(coeff);
        }
    }

    // Count sign changes
    let mut variations = 0;
    for i in 1..coeffs.len() {
        if coeffs[i].is_positive() != coeffs[i - 1].is_positive() {
            variations += 1;
        }
    }

    variations
}

/// Apply Descartes' rule of signs to get bounds on the number of positive roots.
/// Returns (lower_bound, upper_bound) for the number of positive real roots.
/// The actual count equals upper_bound or differs by an even number.
pub(super) fn descartes_positive_roots(poly: &Polynomial, var: Var) -> (usize, usize) {
    let variations = count_coefficient_sign_variations(poly, var);

    // The number of positive roots is variations - 2k for some k >= 0
    // So minimum is 0 if variations is even, 1 if variations is odd
    let lower = variations % 2;
    (lower, variations)
}

/// Apply Descartes' rule of signs to get bounds on the number of negative roots.
/// Returns (lower_bound, upper_bound) for the number of negative real roots.
pub(super) fn descartes_negative_roots(poly: &Polynomial, var: Var) -> (usize, usize) {
    // For negative roots, we evaluate p(-x)
    // This means we negate coefficients of odd powers
    let deg = poly.degree(var);
    let mut neg_poly_terms = Vec::new();

    for k in 0..=deg {
        let coeff = poly.univ_coeff(var, k);
        if !coeff.is_zero() {
            let adjusted_coeff = if k % 2 == 1 { -coeff } else { coeff };
            if k == 0 {
                neg_poly_terms.push(Term::constant(adjusted_coeff));
            } else {
                neg_poly_terms.push(Term::new(adjusted_coeff, Monomial::from_var_power(var, k)));
            }
        }
    }

    let neg_poly = Polynomial::from_terms(neg_poly_terms, poly.order);
    descartes_positive_roots(&neg_poly, var)
}
/// Compute the square root of a rational number if it's a perfect square.
/// Returns None if the rational is not a perfect square of another rational.
pub fn rational_sqrt(n: &BigRational) -> Option<BigRational> {
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        return Some(BigRational::zero());
    }

    // For a fraction p/q to have a rational square root,
    // both p and q must be perfect squares
    let numer = n.numer();
    let denom = n.denom();

    let sqrt_numer = integer_sqrt(numer)?;
    let sqrt_denom = integer_sqrt(denom)?;

    Some(BigRational::new(sqrt_numer, sqrt_denom))
}

pub(super) fn integer_sqrt(n: &BigInt) -> Option<BigInt> {
    if n.is_negative() {
        return None;
    }
    if n.is_zero() {
        return Some(BigInt::zero());
    }
    if n.is_one() {
        return Some(BigInt::one());
    }

    // Newton's method for integer square root
    let mut x: BigInt = n.clone();
    let mut y: BigInt = (&x + BigInt::one()) >> 1; // (x + 1) / 2

    while y < x {
        x = y.clone();
        y = (&x + (n / &x)) >> 1;
    }

    // Check if x is exact square root
    if &x * &x == *n { Some(x) } else { None }
}
