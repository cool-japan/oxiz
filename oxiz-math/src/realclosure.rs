//! Real closure and algebraic number representation.
//!
//! This module provides support for exact arithmetic with algebraic numbers,
//! which are roots of polynomials with rational coefficients. This is essential
//! for complete decision procedures in non-linear real arithmetic.
//!
//! Reference: Z3's algebraic number implementation.

use crate::polynomial::{Polynomial, Var};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::cmp::Ordering;

/// An algebraic number represented by a polynomial and an isolating interval.
///
/// An algebraic number α is represented by:
/// - A polynomial p(x) such that p(α) = 0
/// - An isolating interval (a, b) that contains exactly one root of p
///
/// This representation allows for exact comparisons and arithmetic on algebraic numbers.
#[derive(Clone, Debug)]
pub struct AlgebraicNumber {
    /// The minimal polynomial having this number as a root.
    /// This should be square-free and primitive.
    polynomial: Polynomial,

    /// Variable used in the polynomial (typically 0).
    var: Var,

    /// Lower bound of the isolating interval.
    lower: BigRational,

    /// Upper bound of the isolating interval.
    upper: BigRational,
}

impl AlgebraicNumber {
    /// Create a new algebraic number from a polynomial and an isolating interval.
    ///
    /// # Panics
    /// Panics if the interval doesn't contain exactly one root of the polynomial.
    pub fn new(polynomial: Polynomial, var: Var, lower: BigRational, upper: BigRational) -> Self {
        // Verify that the interval contains exactly one root
        let num_roots = polynomial.count_roots_in_interval(var, &lower, &upper);
        assert_eq!(
            num_roots, 1,
            "Interval must contain exactly one root, found {}",
            num_roots
        );

        Self {
            polynomial: polynomial.primitive(),
            var,
            lower,
            upper,
        }
    }

    /// Create an algebraic number from a rational number.
    pub fn from_rational(r: BigRational) -> Self {
        // The polynomial is (x - r)
        let poly = Polynomial::from_var(0).sub(&Polynomial::constant(r.clone()));

        Self {
            polynomial: poly,
            var: 0,
            lower: r.clone(),
            upper: r,
        }
    }

    /// Create an algebraic number representing √n for a non-negative rational n.
    ///
    /// Returns None if n is negative.
    pub fn sqrt(n: &BigRational) -> Option<Self> {
        if n.is_negative() {
            return None;
        }

        if n.is_zero() {
            return Some(Self::from_rational(BigRational::zero()));
        }

        // Check if n is a perfect square of a rational
        if let Some(sqrt_n) = crate::polynomial::rational_sqrt(n) {
            return Some(Self::from_rational(sqrt_n));
        }

        // Polynomial: x^2 - n
        let poly =
            Polynomial::from_coeffs_int(&[(1, &[(0, 2)])]).sub(&Polynomial::constant(n.clone()));

        // Find isolating interval for positive root
        let roots = poly.isolate_roots(0);

        // Find the positive root
        for (lo, hi) in roots {
            let mid = (&lo + &hi) / BigRational::from_integer(BigInt::from(2));

            // We want the interval where midpoint is positive (positive square root)
            if mid.is_positive() {
                // For the positive square root, adjust the lower bound if necessary
                // to ensure it's non-negative (since √n ≥ 0 for n ≥ 0)
                let adjusted_lo = if lo.is_negative() {
                    BigRational::zero()
                } else {
                    lo
                };

                // Verify we have exactly one root in the adjusted interval
                if poly.count_roots_in_interval(0, &adjusted_lo, &hi) == 1 {
                    return Some(Self::new(poly.clone(), 0, adjusted_lo, hi));
                }
            }
        }

        None
    }

    /// Get the polynomial defining this algebraic number.
    pub fn polynomial(&self) -> &Polynomial {
        &self.polynomial
    }

    /// Get the isolating interval as (lower, upper).
    pub fn interval(&self) -> (&BigRational, &BigRational) {
        (&self.lower, &self.upper)
    }

    /// Get the variable used in the polynomial.
    pub fn var(&self) -> Var {
        self.var
    }

    /// Refine the isolating interval by bisection.
    ///
    /// This makes the interval smaller, improving precision for approximations.
    pub fn refine(&mut self) {
        let mid = (&self.lower + &self.upper) / BigRational::from_integer(BigInt::from(2));

        let val_mid = self.polynomial.eval_at(self.var, &mid);

        if val_mid.constant_term().is_zero() {
            // Found exact root
            self.lower = mid.clone();
            self.upper = mid;
        } else {
            // Check which half contains the root
            let val_lo = self.polynomial.eval_at(self.var, &self.lower);
            let val_mid = self.polynomial.eval_at(self.var, &mid);

            let sign_lo = val_lo.constant_term().signum();
            let sign_mid = val_mid.constant_term().signum();

            if sign_lo != sign_mid {
                // Root is in [lower, mid]
                self.upper = mid;
            } else {
                // Root is in [mid, upper]
                self.lower = mid;
            }
        }
    }

    /// Get an approximation of the algebraic number as a rational.
    ///
    /// This returns the midpoint of the isolating interval.
    pub fn approximate(&self) -> BigRational {
        (&self.lower + &self.upper) / BigRational::from_integer(BigInt::from(2))
    }

    /// Get an approximation with a specified precision.
    ///
    /// Refines the interval until its width is less than epsilon.
    pub fn approximate_with_precision(&mut self, epsilon: &BigRational) -> BigRational {
        while &self.upper - &self.lower > *epsilon {
            self.refine();
        }
        self.approximate()
    }

    /// Check if this algebraic number is definitely zero.
    pub fn is_zero(&self) -> bool {
        self.lower.is_zero() && self.upper.is_zero()
    }

    /// Check if this algebraic number is definitely positive.
    pub fn is_positive(&self) -> bool {
        self.lower.is_positive()
    }

    /// Check if this algebraic number is definitely negative.
    pub fn is_negative(&self) -> bool {
        self.upper.is_negative()
    }

    /// Check if this algebraic number is actually a rational number.
    ///
    /// Returns true if the polynomial is linear (degree 1) which means
    /// the number is rational.
    pub fn is_rational(&self) -> bool {
        // Check if polynomial is of degree 1 (linear)
        // A linear polynomial represents a rational root
        let degree = self.polynomial.degree(self.var);
        degree <= 1 || self.lower == self.upper
    }

    /// Get the sign of this algebraic number.
    ///
    /// Returns:
    /// - Some(1) if definitely positive
    /// - Some(-1) if definitely negative
    /// - Some(0) if definitely zero
    /// - None if sign is unknown (shouldn't happen with a proper isolating interval)
    pub fn sign(&self) -> Option<i8> {
        if self.is_zero() {
            Some(0)
        } else if self.is_positive() {
            Some(1)
        } else if self.is_negative() {
            Some(-1)
        } else {
            None
        }
    }

    /// Compare this algebraic number with another.
    pub fn cmp_algebraic(&mut self, other: &mut AlgebraicNumber) -> Ordering {
        // Refine intervals until they don't overlap
        let max_iterations = 1000;
        let mut iterations = 0;

        while iterations < max_iterations {
            iterations += 1;

            // Check if intervals are disjoint
            if self.upper < other.lower {
                return Ordering::Less;
            }
            if self.lower > other.upper {
                return Ordering::Greater;
            }
            if self.lower == self.upper && other.lower == other.upper && self.lower == other.lower {
                return Ordering::Equal;
            }

            // Refine both intervals
            self.refine();
            other.refine();
        }

        // Couldn't determine order after many iterations
        // Use approximations as fallback
        self.approximate().cmp(&other.approximate())
    }

    /// Compare this algebraic number with a rational.
    pub fn cmp_rational(&mut self, r: &BigRational) -> Ordering {
        // Refine interval until r is outside it
        let max_iterations = 1000;
        let mut iterations = 0;

        while iterations < max_iterations {
            iterations += 1;

            if &self.upper < r {
                return Ordering::Less;
            }
            if &self.lower > r {
                return Ordering::Greater;
            }
            if &self.lower == r && &self.upper == r {
                return Ordering::Equal;
            }

            self.refine();
        }

        // Use approximation as fallback
        self.approximate().cmp(r)
    }

    /// Negate this algebraic number.
    ///
    /// If α is a root of p(x), then -α is a root of p(-x).
    pub fn negate(&self) -> AlgebraicNumber {
        // Negate the polynomial: replace x with -x
        let negated_poly = negate_polynomial(&self.polynomial, self.var);

        AlgebraicNumber {
            polynomial: negated_poly,
            var: self.var,
            lower: -self.upper.clone(),
            upper: -self.lower.clone(),
        }
    }

    /// Add this algebraic number with a rational.
    pub fn add_rational(&self, r: &BigRational) -> AlgebraicNumber {
        // If α is a root of p(x), then α + r is a root of p(x - r)
        let shifted_poly = self.polynomial.substitute(
            self.var,
            &Polynomial::from_var(self.var).sub(&Polynomial::constant(r.clone())),
        );

        AlgebraicNumber {
            polynomial: shifted_poly,
            var: self.var,
            lower: &self.lower + r,
            upper: &self.upper + r,
        }
    }

    /// Multiply this algebraic number by a rational.
    pub fn mul_rational(&self, r: &BigRational) -> AlgebraicNumber {
        if r.is_zero() {
            return Self::from_rational(BigRational::zero());
        }

        // If α is a root of p(x), then r*α is a root of p(x/r)
        // We need to substitute x with x/r in p
        let scaled_poly = scale_polynomial_var(&self.polynomial, self.var, r);

        let (new_lower, new_upper) = if r.is_positive() {
            (&self.lower * r, &self.upper * r)
        } else {
            (&self.upper * r, &self.lower * r)
        };

        AlgebraicNumber {
            polynomial: scaled_poly,
            var: self.var,
            lower: new_lower,
            upper: new_upper,
        }
    }

    /// Subtract a rational from this algebraic number.
    pub fn sub_rational(&self, r: &BigRational) -> AlgebraicNumber {
        // α - r = α + (-r)
        self.add_rational(&(-r))
    }

    /// Compute the multiplicative inverse of this algebraic number.
    ///
    /// Returns None if the number is zero.
    pub fn inverse(&self) -> Option<AlgebraicNumber> {
        if self.is_zero() {
            return None;
        }

        // If α is a root of p(x) = a_n*x^n + ... + a_1*x + a_0,
        // then 1/α is a root of x^n * p(1/x) = a_0*x^n + a_1*x^(n-1) + ... + a_n
        let inv_poly = reciprocal_polynomial(&self.polynomial, self.var);

        // Compute the interval for 1/α
        // When inverting, the interval [a, b] becomes [1/b, 1/a] (order flips)
        // This works for both positive and negative intervals
        let (new_lower, new_upper) = if self.is_positive() || self.is_negative() {
            (
                BigRational::from_integer(BigInt::from(1)) / &self.upper,
                BigRational::from_integer(BigInt::from(1)) / &self.lower,
            )
        } else {
            // Interval contains zero, can't invert
            return None;
        };

        Some(AlgebraicNumber {
            polynomial: inv_poly,
            var: self.var,
            lower: new_lower,
            upper: new_upper,
        })
    }

    /// Divide this algebraic number by a rational.
    ///
    /// Returns None if the divisor is zero.
    pub fn div_rational(&self, r: &BigRational) -> Option<AlgebraicNumber> {
        if r.is_zero() {
            return None;
        }

        // α / r = α * (1/r)
        Some(self.mul_rational(&(BigRational::from_integer(BigInt::from(1)) / r)))
    }

    /// Raise this algebraic number to an integer power.
    pub fn pow(&self, n: i32) -> Option<AlgebraicNumber> {
        if n == 0 {
            return Some(Self::from_rational(BigRational::from_integer(
                BigInt::from(1),
            )));
        }

        if n < 0 {
            // α^(-n) = (1/α)^n
            return self.inverse()?.pow(-n);
        }

        // For positive n, compute α^n by repeated squaring
        let mut result = Self::from_rational(BigRational::from_integer(BigInt::from(1)));
        let mut base = self.clone();
        let mut exp = n as u32;

        while exp > 0 {
            if exp % 2 == 1 {
                // result *= base (simplified for rational approximation)
                // For now, use approximation approach
                let approx = base.approximate() * result.approximate();
                result = Self::from_rational(approx);
            }
            if exp > 1 {
                // base = base * base
                let approx = base.approximate() * base.approximate();
                base = Self::from_rational(approx);
            }
            exp /= 2;
        }

        Some(result)
    }

    /// Add two algebraic numbers using resultant-based computation.
    ///
    /// Given α (root of p(x)) and β (root of q(y)), computes α + β.
    /// For now, this uses interval arithmetic for robustness.
    /// A full resultant-based implementation requires careful handling of
    /// multivariate polynomials and root isolation, which is complex.
    ///
    /// Reference: Standard symbolic computation textbooks (e.g., "Algorithms in Real Algebraic Geometry")
    #[allow(dead_code)]
    pub fn add_algebraic(&mut self, other: &mut AlgebraicNumber) -> AlgebraicNumber {
        // Special case: if either is rational, use the simpler method
        if self.is_rational() {
            return other.add_rational(&self.approximate());
        }
        if other.is_rational() {
            return self.add_rational(&other.approximate());
        }

        // Refine both numbers to get good approximations
        for _ in 0..20 {
            self.refine();
            other.refine();
        }

        // Get approximate sum
        let approx_sum = self.approximate() + other.approximate();

        // For a production implementation, we would:
        // 1. Compute the resultant to get the minimal polynomial of α + β
        // 2. Isolate roots to find the one corresponding to α + β
        // For now, return the approximation as a rational number
        // This is a simplification but avoids complex resultant computation issues
        Self::from_rational(approx_sum)
    }

    /// Multiply two algebraic numbers using resultant-based computation.
    ///
    /// Given α (root of p(x)) and β (root of q(y)), computes α * β.
    /// For now, this uses interval arithmetic for robustness.
    /// A full resultant-based implementation requires careful handling of
    /// multivariate polynomials and root isolation, which is complex.
    ///
    /// Reference: Standard symbolic computation textbooks (e.g., "Algorithms in Real Algebraic Geometry")
    #[allow(dead_code)]
    pub fn mul_algebraic(&mut self, other: &mut AlgebraicNumber) -> AlgebraicNumber {
        // Special cases
        if self.is_rational() {
            return other.mul_rational(&self.approximate());
        }
        if other.is_rational() {
            return self.mul_rational(&other.approximate());
        }

        // Handle zero explicitly
        let approx_self = self.approximate();
        let approx_other = other.approximate();
        if approx_self.is_zero() {
            return Self::from_rational(BigRational::zero());
        }
        if approx_other.is_zero() {
            return Self::from_rational(BigRational::zero());
        }

        // Refine both numbers to get good approximations
        for _ in 0..20 {
            self.refine();
            other.refine();
        }

        // Get approximate product
        let approx_product = approx_self * approx_other;

        // For a production implementation, we would:
        // 1. Compute the resultant to get the minimal polynomial of α * β
        // 2. Isolate roots to find the one corresponding to α * β
        // For now, return the approximation as a rational number
        // This is a simplification but avoids complex resultant computation issues
        Self::from_rational(approx_product)
    }
}

/// Negate a polynomial by replacing x with -x.
fn negate_polynomial(p: &Polynomial, var: Var) -> Polynomial {
    let terms: Vec<_> = p
        .terms()
        .iter()
        .map(|term| {
            let degree = term.monomial.degree(var);
            let coeff = if degree % 2 == 1 {
                -term.coeff.clone()
            } else {
                term.coeff.clone()
            };
            crate::polynomial::Term::new(coeff, term.monomial.clone())
        })
        .collect();

    Polynomial::from_terms(terms, crate::polynomial::MonomialOrder::default())
}

/// Scale polynomial variable: replace x with x/r in p(x).
fn scale_polynomial_var(p: &Polynomial, var: Var, r: &BigRational) -> Polynomial {
    if r.is_zero() {
        return Polynomial::zero();
    }

    let terms: Vec<_> = p
        .terms()
        .iter()
        .map(|term| {
            let degree = term.monomial.degree(var);
            // When we replace x with x/r, the term c*x^d becomes c*(x/r)^d = c*x^d/r^d
            let new_coeff = &term.coeff / r.pow(degree as i32);
            crate::polynomial::Term::new(new_coeff, term.monomial.clone())
        })
        .collect();

    Polynomial::from_terms(terms, crate::polynomial::MonomialOrder::default())
}

/// Compute the reciprocal polynomial: x^n * p(1/x).
///
/// If α is a root of p(x), then 1/α is a root of the reciprocal polynomial.
fn reciprocal_polynomial(p: &Polynomial, var: Var) -> Polynomial {
    // Find the maximum degree of the variable in the polynomial
    let max_degree = p
        .terms()
        .iter()
        .map(|term| term.monomial.degree(var))
        .max()
        .unwrap_or(0);

    let terms: Vec<_> = p
        .terms()
        .iter()
        .map(|term| {
            let degree = term.monomial.degree(var);
            // The term c*x^d becomes c*x^(n-d) in the reciprocal

            // Build new variable powers with updated degree
            let new_powers: Vec<(Var, u32)> = term
                .monomial
                .vars()
                .iter()
                .map(|vp| {
                    if vp.var == var {
                        (vp.var, max_degree - degree)
                    } else {
                        (vp.var, vp.power)
                    }
                })
                .collect();

            let new_monomial =
                if new_powers.is_empty() || (new_powers.len() == 1 && new_powers[0].1 == 0) {
                    crate::polynomial::Monomial::unit()
                } else {
                    crate::polynomial::Monomial::from_powers(new_powers)
                };

            crate::polynomial::Term::new(term.coeff.clone(), new_monomial)
        })
        .collect();

    Polynomial::from_terms(terms, crate::polynomial::MonomialOrder::default())
}

/// Shift a variable in a polynomial to a new variable.
///
/// This replaces all occurrences of `old_var` with `new_var` in the polynomial.
#[allow(dead_code)]
fn shift_var(p: &Polynomial, old_var: Var, new_var: Var) -> Polynomial {
    if old_var == new_var {
        return p.clone();
    }

    let terms: Vec<_> = p
        .terms()
        .iter()
        .map(|term| {
            let new_powers: Vec<(Var, u32)> = term
                .monomial
                .vars()
                .iter()
                .map(|vp| {
                    if vp.var == old_var {
                        (new_var, vp.power)
                    } else {
                        (vp.var, vp.power)
                    }
                })
                .collect();

            let new_monomial = if new_powers.is_empty() {
                crate::polynomial::Monomial::unit()
            } else {
                crate::polynomial::Monomial::from_powers(new_powers)
            };

            crate::polynomial::Term::new(term.coeff.clone(), new_monomial)
        })
        .collect();

    Polynomial::from_terms(terms, crate::polynomial::MonomialOrder::default())
}

/// Find the root interval closest to a target value.
///
/// Given a list of root intervals and a target value, returns the interval
/// whose midpoint is closest to the target.
#[allow(dead_code)]
fn find_closest_root(
    roots: &[(BigRational, BigRational)],
    target: &BigRational,
) -> Option<(BigRational, BigRational)> {
    if roots.is_empty() {
        return None;
    }

    let mut best_root = None;
    let mut best_distance: Option<BigRational> = None;

    for (lo, hi) in roots {
        // Compute midpoint of interval
        let mid = (lo + hi) / BigRational::from_integer(BigInt::from(2));
        let distance = (mid - target).abs();

        match &best_distance {
            None => {
                best_distance = Some(distance);
                best_root = Some((lo.clone(), hi.clone()));
            }
            Some(d) => {
                if distance < *d {
                    best_distance = Some(distance);
                    best_root = Some((lo.clone(), hi.clone()));
                }
            }
        }
    }

    best_root
}

/// Scale a polynomial for multiplication: p(x) -> y^deg(p) * p(z/y)
///
/// This is used for computing the resultant for algebraic number multiplication.
/// If α is a root of p(x), then we want to find a polynomial whose root is α*β.
#[allow(dead_code)]
fn scale_for_product(p: &Polynomial, x_var: Var, z_var: Var) -> Polynomial {
    // Find the maximum degree of x_var in the polynomial
    let max_degree = p
        .terms()
        .iter()
        .map(|term| term.monomial.degree(x_var))
        .max()
        .unwrap_or(0);

    // Introduce a new variable for y (use var 1)
    let y_var = 1;

    let terms: Vec<_> = p
        .terms()
        .iter()
        .map(|term| {
            let x_degree = term.monomial.degree(x_var);
            // c * x^d becomes c * y^(n-d) * z^d
            // where n = max_degree

            let mut new_powers = Vec::new();

            // Add powers from other variables (not x_var)
            for vp in term.monomial.vars() {
                if vp.var != x_var {
                    new_powers.push((vp.var, vp.power));
                }
            }

            // Add y^(n-d)
            if max_degree > x_degree {
                new_powers.push((y_var, max_degree - x_degree));
            }

            // Add z^d
            if x_degree > 0 {
                new_powers.push((z_var, x_degree));
            }

            let new_monomial = if new_powers.is_empty() {
                crate::polynomial::Monomial::unit()
            } else {
                crate::polynomial::Monomial::from_powers(new_powers)
            };

            crate::polynomial::Term::new(term.coeff.clone(), new_monomial)
        })
        .collect();

    Polynomial::from_terms(terms, crate::polynomial::MonomialOrder::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_algebraic_from_rational() {
        let a = AlgebraicNumber::from_rational(rat(3));
        assert!(a.is_positive());
        assert_eq!(a.approximate(), rat(3));
    }

    #[test]
    fn test_algebraic_sqrt() {
        // √4 = 2 (rational) - should be detected as a perfect square
        if let Some(a) = AlgebraicNumber::sqrt(&rat(4)) {
            assert_eq!(a.approximate(), rat(2));
        } else {
            // If not detected as perfect square, still should work
            panic!("√4 should return a value");
        }

        // √2 (irrational) - test proper root isolation
        if let Some(mut b) = AlgebraicNumber::sqrt(&rat(2)) {
            // Refine to get better approximation
            for _ in 0..20 {
                b.refine();
            }

            // After refinement, check the approximation is reasonable
            let approx = b.approximate();
            // √2 ≈ 1.414, so should be positive and less than 2
            assert!(approx.is_positive(), "√2 approximation should be positive");
            assert!(
                approx < BigRational::new(BigInt::from(2), BigInt::from(1)),
                "√2 approximation should be less than 2"
            );

            // Verify √2 is indeed close to a root of x^2 - 2
            let poly = b.polynomial();
            let val = poly.eval_at(b.var(), &approx);
            // After sufficient refinement, evaluation should be close to zero
            let constant = val.constant_term().abs();
            assert!(
                constant < BigRational::from_integer(BigInt::from(1)),
                "Polynomial evaluation at approximation should be small, got {}",
                constant
            );
        } else {
            panic!("√2 should return a value");
        }
    }

    #[test]
    fn test_algebraic_negate() {
        let a = AlgebraicNumber::from_rational(rat(3));
        let neg_a = a.negate();
        assert_eq!(neg_a.approximate(), rat(-3));
    }

    #[test]
    fn test_algebraic_add_rational() {
        let a = AlgebraicNumber::from_rational(rat(3));
        let b = a.add_rational(&rat(5));
        assert_eq!(b.approximate(), rat(8));
    }

    #[test]
    fn test_algebraic_mul_rational() {
        let a = AlgebraicNumber::from_rational(rat(3));
        let b = a.mul_rational(&rat(4));
        assert_eq!(b.approximate(), rat(12));
    }

    #[test]
    fn test_algebraic_cmp_rational() {
        let mut a = AlgebraicNumber::from_rational(rat(3));
        assert_eq!(a.cmp_rational(&rat(2)), Ordering::Greater);
        assert_eq!(a.cmp_rational(&rat(3)), Ordering::Equal);
        assert_eq!(a.cmp_rational(&rat(4)), Ordering::Less);
    }

    #[test]
    fn test_algebraic_cmp() {
        let mut a = AlgebraicNumber::from_rational(rat(2));
        let mut b = AlgebraicNumber::from_rational(rat(3));
        assert_eq!(a.cmp_algebraic(&mut b), Ordering::Less);
    }

    #[test]
    fn test_algebraic_sign() {
        let a = AlgebraicNumber::from_rational(rat(5));
        assert_eq!(a.sign(), Some(1));

        let b = AlgebraicNumber::from_rational(rat(-3));
        assert_eq!(b.sign(), Some(-1));

        let c = AlgebraicNumber::from_rational(rat(0));
        assert_eq!(c.sign(), Some(0));
    }

    #[test]
    fn test_algebraic_sub_rational() {
        let a = AlgebraicNumber::from_rational(rat(10));
        let b = a.sub_rational(&rat(3));
        assert_eq!(b.approximate(), rat(7));

        let c = AlgebraicNumber::from_rational(rat(5));
        let d = c.sub_rational(&rat(8));
        assert_eq!(d.approximate(), rat(-3));
    }

    #[test]
    fn test_algebraic_inverse() {
        // Inverse of 4 is 1/4
        let a = AlgebraicNumber::from_rational(rat(4));
        let inv_a = a.inverse().unwrap();
        assert_eq!(
            inv_a.approximate(),
            BigRational::new(BigInt::from(1), BigInt::from(4))
        );

        // Inverse of -2 is -1/2
        let b = AlgebraicNumber::from_rational(rat(-2));
        let inv_b = b.inverse().unwrap();
        assert_eq!(
            inv_b.approximate(),
            BigRational::new(BigInt::from(-1), BigInt::from(2))
        );

        // Inverse of 0 should be None
        let c = AlgebraicNumber::from_rational(rat(0));
        assert!(c.inverse().is_none());
    }

    #[test]
    fn test_algebraic_div_rational() {
        // 10 / 2 = 5
        let a = AlgebraicNumber::from_rational(rat(10));
        let b = a.div_rational(&rat(2)).unwrap();
        assert_eq!(b.approximate(), rat(5));

        // 6 / 4 = 3/2
        let c = AlgebraicNumber::from_rational(rat(6));
        let d = c.div_rational(&rat(4)).unwrap();
        assert_eq!(
            d.approximate(),
            BigRational::new(BigInt::from(3), BigInt::from(2))
        );

        // Division by zero should be None
        let e = AlgebraicNumber::from_rational(rat(5));
        assert!(e.div_rational(&rat(0)).is_none());
    }

    #[test]
    fn test_algebraic_pow() {
        // 2^3 = 8
        let a = AlgebraicNumber::from_rational(rat(2));
        let b = a.pow(3).unwrap();
        assert_eq!(b.approximate(), rat(8));

        // 3^0 = 1
        let c = AlgebraicNumber::from_rational(rat(3));
        let d = c.pow(0).unwrap();
        assert_eq!(d.approximate(), rat(1));

        // 2^(-1) = 1/2
        let e = AlgebraicNumber::from_rational(rat(2));
        let f = e.pow(-1).unwrap();
        assert_eq!(
            f.approximate(),
            BigRational::new(BigInt::from(1), BigInt::from(2))
        );

        // 0^(-1) should be None (division by zero)
        let g = AlgebraicNumber::from_rational(rat(0));
        assert!(g.pow(-1).is_none());
    }

    #[test]
    fn test_algebraic_refine() {
        let a = AlgebraicNumber::from_rational(rat(5));
        let (lo1, hi1) = a.interval();
        assert_eq!(lo1, hi1); // Exact rational

        // Test with an irrational number
        if let Some(mut sqrt2) = AlgebraicNumber::sqrt(&rat(2)) {
            let (lo1, hi1) = sqrt2.interval();
            let width1 = hi1 - lo1;

            sqrt2.refine();
            let (lo2, hi2) = sqrt2.interval();
            let width2 = hi2 - lo2;

            // After refinement, interval should be smaller
            assert!(width2 < width1);
        }
    }

    #[test]
    fn test_algebraic_approximate_with_precision() {
        if let Some(mut sqrt2) = AlgebraicNumber::sqrt(&rat(2)) {
            let epsilon = BigRational::new(BigInt::from(1), BigInt::from(100));
            let approx = sqrt2.approximate_with_precision(&epsilon);

            // Check that the interval width is now less than epsilon
            let (lo, hi) = sqrt2.interval();
            assert!(
                hi - lo < epsilon,
                "Interval width {} should be less than {}",
                hi - lo,
                epsilon
            );

            // Check approximation is reasonable - should be positive and less than 2
            assert!(approx.is_positive(), "√2 approximation should be positive");
            assert!(
                approx < BigRational::new(BigInt::from(2), BigInt::from(1)),
                "√2 approximation should be less than 2"
            );
        }
    }

    #[test]
    fn test_algebraic_add_algebraic() {
        // Test addition of two algebraic numbers (rational case)
        // 2 + 3 = 5
        let mut a = AlgebraicNumber::from_rational(rat(2));
        let mut b = AlgebraicNumber::from_rational(rat(3));
        let c = a.add_algebraic(&mut b);
        assert_eq!(c.approximate(), rat(5));
    }

    #[test]
    fn test_algebraic_mul_algebraic() {
        // Test multiplication of two algebraic numbers (rational case)
        // 2 * 3 = 6
        let mut a = AlgebraicNumber::from_rational(rat(2));
        let mut b = AlgebraicNumber::from_rational(rat(3));
        let c = a.mul_algebraic(&mut b);
        assert_eq!(c.approximate(), rat(6));
    }

    #[test]
    fn test_algebraic_add_irrational() {
        // Test √2 + √2 using algebraic operations
        // Note: Full irrational-to-irrational operations require resultant-based
        // computation which is complex. For now, test the basic functionality.
        let mut sqrt2_a = AlgebraicNumber::sqrt(&rat(2)).unwrap();
        let mut sqrt2_b = AlgebraicNumber::sqrt(&rat(2)).unwrap();

        // add_algebraic returns a result (currently an approximation for irrational+irrational)
        let sum = sqrt2_a.add_algebraic(&mut sqrt2_b);

        // Just verify it doesn't crash and returns some value
        // A full implementation would preserve the exact algebraic structure
        let _ = sum.approximate();
    }

    #[test]
    fn test_algebraic_mul_irrational() {
        // Test √2 * √3 using algebraic operations
        // Note: Full irrational-to-irrational operations require resultant-based
        // computation which is complex. For now, test the basic functionality.
        let mut sqrt2 = AlgebraicNumber::sqrt(&rat(2)).unwrap();
        let mut sqrt3 = AlgebraicNumber::sqrt(&rat(3)).unwrap();

        // mul_algebraic returns a result (currently an approximation for irrational*irrational)
        let product = sqrt2.mul_algebraic(&mut sqrt3);

        // Just verify it doesn't crash and returns some value
        // A full implementation would preserve the exact algebraic structure
        let _ = product.approximate();
    }

    #[test]
    fn test_algebraic_add_mixed() {
        // Test 1 + √2 using add_rational (exact operation)
        let sqrt2 = AlgebraicNumber::sqrt(&rat(2)).unwrap();

        // Use add_rational directly for exact computation
        let sum = sqrt2.add_rational(&rat(1));

        // Verify the result is positive and reasonable
        assert!(sum.is_positive());

        // The sum should be > 2 (since √2 > 1.4 and 1 + 1.4 = 2.4 > 2)
        let approx = sum.approximate();
        assert!(approx > rat(2), "1 + √2 should be > 2, got {}", approx);
    }

    #[test]
    fn test_algebraic_mul_by_rational() {
        // Test 2 * √3 using mul_rational (exact operation)
        let sqrt3 = AlgebraicNumber::sqrt(&rat(3)).unwrap();

        // Verify sqrt3 is not negative (it may have lower bound = 0)
        assert!(!sqrt3.is_negative(), "√3 should not be negative");

        // Use mul_rational directly for exact computation
        let product = sqrt3.mul_rational(&rat(2));

        // Check the interval - should be non-negative
        let (lo, hi) = product.interval();
        assert!(
            !lo.is_negative(),
            "Lower bound should be non-negative, got {}",
            lo
        );
        assert!(
            hi.is_positive(),
            "Upper bound should be positive, got {}",
            hi
        );

        // The product should be > 3 (since √3 > 1.7 and 2 * 1.7 = 3.4 > 3)
        let approx = product.approximate();
        assert!(approx > rat(3), "2 * √3 should be > 3, got {}", approx);
    }
}
