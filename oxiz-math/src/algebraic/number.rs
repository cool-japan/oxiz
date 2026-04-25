//! Algebraic Number Representation.
//!
//! An algebraic number is a root of a polynomial with rational coefficients.
//! We represent it as (polynomial, isolating_interval) where the interval
//! contains exactly one root of the polynomial.
//!
//! ## Representation
//!
//! ```text
//! α = (p(x), [a, b]) where p(α) = 0 and p has exactly one root in [a, b]
//! ```
//!
//! ## References
//!
//! - "Algorithms in Real Algebraic Geometry" (Basu et al., 2006)
//! - Z3's `math/polynomial/algebraic_numbers.h`

use crate::polynomial::root_counting::Polynomial;
#[allow(unused_imports)]
use crate::prelude::*;
use core::cmp::Ordering;
use core::fmt;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Algebraic number error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlgebraicNumberError {
    /// Interval doesn't isolate a single root.
    NonIsolatingInterval,
    /// Polynomial has no roots in interval.
    NoRootInInterval,
    /// Polynomial is zero.
    ZeroPolynomial,
    /// Invalid operation.
    InvalidOperation(String),
}

impl fmt::Display for AlgebraicNumberError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonIsolatingInterval => write!(f, "Interval doesn't isolate a single root"),
            Self::NoRootInInterval => write!(f, "No root in interval"),
            Self::ZeroPolynomial => write!(f, "Polynomial is zero"),
            Self::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl core::error::Error for AlgebraicNumberError {}

/// Algebraic number represented as a root of a univariate polynomial over Q.
#[derive(Debug, Clone)]
pub struct AlgebraicNumber {
    /// Minimal polynomial (irreducible, with rational coefficients).
    pub minimal_poly: Polynomial,
    /// Lower bound of the isolating interval.
    pub lower: BigRational,
    /// Upper bound of the isolating interval.
    pub upper: BigRational,
    /// Number of refinement steps performed (for termination detection).
    refinement_level: usize,
}

impl AlgebraicNumber {
    /// Create a new algebraic number.
    ///
    /// Validates that the interval `[lower, upper]` contains exactly one root
    /// of `poly` using Sturm's theorem before accepting it.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `poly` is the zero polynomial,
    /// - `lower > upper` (degenerate interval ordering), or
    /// - the Sturm sequence shows ≠ 1 root inside `(lower, upper)`.
    pub fn new(
        poly: Polynomial,
        lower: BigRational,
        upper: BigRational,
    ) -> Result<Self, AlgebraicNumberError> {
        // Reject the zero polynomial — it has infinitely many roots.
        if poly.degree() == 0 && poly.coeffs.first().is_none_or(Zero::is_zero) {
            return Err(AlgebraicNumberError::ZeroPolynomial);
        }

        if lower > upper {
            return Err(AlgebraicNumberError::NonIsolatingInterval);
        }

        // Degenerate (point) interval: check whether `lower` is a root.
        if lower == upper {
            if poly.eval(&lower).is_zero() {
                return Ok(Self {
                    minimal_poly: poly,
                    lower,
                    upper,
                    refinement_level: 0,
                });
            }
            return Err(AlgebraicNumberError::NoRootInInterval);
        }

        // Sturm sequence check: the interval must isolate exactly one root.
        let root_count = sturm_root_count(&poly, &lower, &upper);
        if root_count != 1 {
            return Err(AlgebraicNumberError::NonIsolatingInterval);
        }

        Ok(Self {
            minimal_poly: poly,
            lower,
            upper,
            refinement_level: 0,
        })
    }

    /// Create algebraic number from a rational.
    ///
    /// The rational `r` is represented as the root of the linear polynomial `(x - r)`.
    pub fn from_rational(r: BigRational) -> Self {
        // x - r  →  coeffs = [-r, 1]
        let poly = Polynomial::new(vec![-r.clone(), BigRational::from(BigInt::from(1))]);

        Self {
            minimal_poly: poly,
            lower: r.clone(),
            upper: r,
            refinement_level: 0,
        }
    }

    /// Check if this algebraic number is rational.
    pub fn is_rational(&self) -> bool {
        self.minimal_poly.degree() == 1 || self.lower == self.upper
    }

    /// Convert to rational if the number is rational.
    pub fn to_rational(&self) -> Option<BigRational> {
        if self.is_rational() {
            Some(self.lower.clone())
        } else {
            None
        }
    }

    /// Refine the isolating interval by bisection.
    pub fn refine(&mut self) {
        let mid = (&self.lower + &self.upper) / BigRational::from(BigInt::from(2));
        let mid_value = self.minimal_poly.eval(&mid);

        if mid_value.is_zero() {
            // Exact root — collapse to a point.
            self.lower = mid.clone();
            self.upper = mid;
        } else {
            let lower_value = self.minimal_poly.eval(&self.lower);
            if lower_value.signum() != mid_value.signum() {
                // Root is in [lower, mid].
                self.upper = mid;
            } else {
                // Root is in [mid, upper].
                self.lower = mid;
            }
        }

        self.refinement_level += 1;
    }

    /// Refine until the interval width is at most `precision`.
    pub fn refine_to_precision(&mut self, precision: &BigRational) {
        while &self.upper - &self.lower > *precision {
            self.refine();
        }
    }

    /// Width of the current isolating interval.
    pub fn interval_width(&self) -> BigRational {
        &self.upper - &self.lower
    }

    /// Midpoint of the current isolating interval (an approximation of the number).
    pub fn midpoint(&self) -> BigRational {
        (&self.lower + &self.upper) / BigRational::from(BigInt::from(2))
    }

    /// Compare with another algebraic number by interval refinement.
    pub fn compare(&mut self, other: &mut AlgebraicNumber) -> Ordering {
        loop {
            if self.upper < other.lower {
                return Ordering::Less;
            }
            if self.lower > other.upper {
                return Ordering::Greater;
            }

            // Identical representation → definitely equal.
            if self.minimal_poly == other.minimal_poly
                && self.lower == other.lower
                && self.upper == other.upper
            {
                return Ordering::Equal;
            }

            // Intervals overlap; refine both.
            self.refine();
            other.refine();

            if self.refinement_level > 1000 {
                // Cannot separate after 1 000 steps — treat as equal.
                return Ordering::Equal;
            }
        }
    }

    /// Add two algebraic numbers.
    ///
    /// Only rational operands are handled in this implementation.
    ///
    /// # Errors
    ///
    /// Returns `InvalidOperation` when either operand is non-rational (the
    /// general case requires resultant computation, not yet implemented).
    pub fn add(&self, other: &AlgebraicNumber) -> Result<AlgebraicNumber, AlgebraicNumberError> {
        if self.is_rational() && other.is_rational() {
            return Ok(AlgebraicNumber::from_rational(&self.lower + &other.lower));
        }

        Err(AlgebraicNumberError::InvalidOperation(
            "General algebraic addition requires resultant computation (not yet implemented)"
                .to_string(),
        ))
    }

    /// Multiply two algebraic numbers.
    ///
    /// Only rational operands are handled in this implementation.
    ///
    /// # Errors
    ///
    /// Returns `InvalidOperation` when either operand is non-rational.
    pub fn mul(&self, other: &AlgebraicNumber) -> Result<AlgebraicNumber, AlgebraicNumberError> {
        if self.is_rational() && other.is_rational() {
            return Ok(AlgebraicNumber::from_rational(&self.lower * &other.lower));
        }

        Err(AlgebraicNumberError::InvalidOperation(
            "General algebraic multiplication requires resultant computation (not yet implemented)"
                .to_string(),
        ))
    }

    /// Negate an algebraic number.
    ///
    /// If α is a root of p(x), then −α is a root of p(−x).
    pub fn negate(&self) -> AlgebraicNumber {
        let neg_poly = poly_substitute_neg_x(&self.minimal_poly);
        AlgebraicNumber {
            minimal_poly: neg_poly,
            lower: -&self.upper,
            upper: -&self.lower,
            refinement_level: self.refinement_level,
        }
    }

    /// Return the sign of the algebraic number: −1, 0, or 1.
    pub fn signum(&self) -> i32 {
        if self.upper < BigRational::zero() {
            -1
        } else if self.lower > BigRational::zero() {
            1
        } else if self.lower == self.upper && self.lower.is_zero() {
            0
        } else {
            // Interval straddles zero; refine until sign is clear.
            let mut copy = self.clone();
            copy.refine();
            copy.signum()
        }
    }
}

// ─── private helpers ────────────────────────────────────────────────────────

/// Substitute x → −x in a polynomial: p(−x).
///
/// For p(x) = a₀ + a₁x + a₂x² + …, we have p(−x) = a₀ − a₁x + a₂x² − …
fn poly_substitute_neg_x(poly: &Polynomial) -> Polynomial {
    let coeffs: Vec<BigRational> = poly
        .coeffs
        .iter()
        .enumerate()
        .map(|(i, c)| if i % 2 == 0 { c.clone() } else { -c })
        .collect();
    Polynomial::new(coeffs)
}

/// Count distinct real roots of `poly` in the open interval `(lower, upper)`
/// using Sturm's theorem: count = V(lower) − V(upper) where V(x) is the
/// number of sign changes in the Sturm sequence evaluated at x.
fn sturm_root_count(poly: &Polynomial, lower: &BigRational, upper: &BigRational) -> usize {
    let seq = build_sturm_sequence(poly);
    let v_lower = sign_variations(&seq, lower);
    let v_upper = sign_variations(&seq, upper);
    (v_lower as isize - v_upper as isize).unsigned_abs()
}

/// Build the Sturm sequence of a polynomial.
///
/// - p₀ = p
/// - p₁ = p′
/// - pₖ₊₁ = −rem(pₖ₋₁, pₖ)
fn build_sturm_sequence(poly: &Polynomial) -> Vec<Polynomial> {
    let mut seq = vec![poly.clone(), poly.derivative()];

    loop {
        let n = seq.len();
        let last = &seq[n - 1];

        // Terminate when the tail is degree-0 (zero or non-zero constant): any polynomial
        // mod a constant is 0, so the next negated remainder would be 0 anyway.
        if last.degree() == 0 {
            break;
        }

        let remainder = seq[n - 2].remainder(last);
        let negated = Polynomial::new(remainder.coeffs.iter().map(|c| -c).collect());

        if negated.degree() == 0 && negated.coeffs.first().is_none_or(Zero::is_zero) {
            break;
        }

        seq.push(negated);

        // Safety guard against pathological inputs.
        if seq.len() > 1000 {
            break;
        }
    }

    seq
}

/// Count sign variations in the Sturm sequence evaluated at `point`.
///
/// Zero values are skipped (Sturm's theorem convention).
fn sign_variations(seq: &[Polynomial], point: &BigRational) -> usize {
    let signs: Vec<i32> = seq
        .iter()
        .map(|p| {
            let val = p.eval(point);
            if val.is_positive() {
                1
            } else if val.is_negative() {
                -1
            } else {
                0
            }
        })
        .filter(|&s| s != 0)
        .collect();

    signs.windows(2).filter(|w| w[0] != w[1]).count()
}

// ─── trait impls ────────────────────────────────────────────────────────────

impl PartialEq for AlgebraicNumber {
    fn eq(&self, other: &Self) -> bool {
        let mut a = self.clone();
        let mut b = other.clone();
        matches!(a.compare(&mut b), Ordering::Equal)
    }
}

impl Eq for AlgebraicNumber {}

impl PartialOrd for AlgebraicNumber {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AlgebraicNumber {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut a = self.clone();
        let mut b = other.clone();
        a.compare(&mut b)
    }
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from(BigInt::from(n))
    }

    #[test]
    fn test_from_rational() {
        let alg = AlgebraicNumber::from_rational(rat(42));
        assert!(alg.is_rational());
        assert_eq!(alg.to_rational(), Some(rat(42)));
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = AlgebraicNumber::from_rational(rat(3));
        let b = AlgebraicNumber::from_rational(rat(5));

        let sum = a.add(&b).expect("test operation should succeed");
        assert!(sum.is_rational());
        assert_eq!(sum.to_rational(), Some(rat(8)));

        let prod = a.mul(&b).expect("test operation should succeed");
        assert!(prod.is_rational());
        assert_eq!(prod.to_rational(), Some(rat(15)));
    }

    #[test]
    fn test_negate() {
        let a = AlgebraicNumber::from_rational(rat(5));
        let neg_a = a.negate();

        assert!(neg_a.is_rational());
        assert_eq!(neg_a.to_rational(), Some(rat(-5)));
    }

    #[test]
    fn test_compare() {
        let a = AlgebraicNumber::from_rational(rat(3));
        let b = AlgebraicNumber::from_rational(rat(5));

        assert!(a < b);
        assert!(b > a);
        assert_eq!(a, a.clone());
    }

    #[test]
    fn test_signum() {
        let pos = AlgebraicNumber::from_rational(rat(5));
        let neg = AlgebraicNumber::from_rational(rat(-3));
        let zero = AlgebraicNumber::from_rational(rat(0));

        assert_eq!(pos.signum(), 1);
        assert_eq!(neg.signum(), -1);
        assert_eq!(zero.signum(), 0);
    }

    #[test]
    fn test_refine() {
        // x^2 - 2, root in [1, 2]
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let mut alg =
            AlgebraicNumber::new(poly, rat(1), rat(2)).expect("test operation should succeed");

        let initial_width = alg.interval_width();
        alg.refine();
        let refined_width = alg.interval_width();

        assert!(refined_width < initial_width);
    }

    #[test]
    fn test_interval_width() {
        let alg = AlgebraicNumber::from_rational(rat(5));
        assert_eq!(alg.interval_width(), rat(0));
    }

    #[test]
    fn test_new_rejects_non_isolating_interval() {
        // x^2 - 1 has two roots in [-2, 2]
        let poly = Polynomial::new(vec![rat(-1), rat(0), rat(1)]);
        let result = AlgebraicNumber::new(poly, rat(-2), rat(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_new_accepts_isolating_interval() {
        // x^2 - 2 has exactly one root in [1, 2]
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let result = AlgebraicNumber::new(poly, rat(1), rat(2));
        assert!(result.is_ok());
    }

    #[test]
    fn test_sturm_root_count() {
        // x^2 - 2: one positive root between 1 and 2
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let count = sturm_root_count(&poly, &rat(1), &rat(2));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_poly_substitute_neg_x() {
        // p(x) = x^2 - 2 → p(-x) = x^2 - 2 (even polynomial, unchanged)
        let poly = Polynomial::new(vec![rat(-2), rat(0), rat(1)]);
        let neg = poly_substitute_neg_x(&poly);
        assert_eq!(neg.eval(&rat(2)), poly.eval(&rat(2)));

        // p(x) = x^3 → p(-x) = -x^3
        let poly2 = Polynomial::new(vec![rat(0), rat(0), rat(0), rat(1)]);
        let neg2 = poly_substitute_neg_x(&poly2);
        assert_eq!(neg2.eval(&rat(2)), -poly2.eval(&rat(2)));
    }
}
