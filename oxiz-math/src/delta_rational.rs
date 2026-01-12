//! Delta-rational numbers for handling strict inequalities.
//!
//! A delta-rational is a number of the form r + δ*k where:
//! - r is a rational number
//! - δ is an infinitesimal (smaller than any positive rational)
//! - k is an integer coefficient
//!
//! This allows us to represent strict inequalities:
//! - x < 5 becomes x ≤ 5 - δ
//! - x > 5 becomes x ≥ 5 + δ
//!
//! Reference: Z3's delta_rational implementation.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Neg, Sub};

/// A delta-rational number: r + δ*k
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeltaRational {
    /// The rational part.
    pub rational: BigRational,
    /// The infinitesimal coefficient.
    pub delta_coeff: i64,
}

impl DeltaRational {
    /// Create a new delta-rational.
    pub fn new(rational: BigRational, delta_coeff: i64) -> Self {
        Self {
            rational,
            delta_coeff,
        }
    }

    /// Create a delta-rational from a rational (delta_coeff = 0).
    #[inline]
    pub fn from_rational(r: BigRational) -> Self {
        Self::new(r, 0)
    }

    /// Create a delta-rational from an integer.
    #[inline]
    pub fn from_int(n: i64) -> Self {
        Self::from_rational(BigRational::from_integer(BigInt::from(n)))
    }

    /// Create zero (0 + 0*δ).
    #[inline]
    pub fn zero() -> Self {
        Self::from_int(0)
    }

    /// Create one (1 + 0*δ).
    #[inline]
    pub fn one() -> Self {
        Self::from_int(1)
    }

    /// Create the infinitesimal δ (0 + 1*δ).
    #[inline]
    pub fn delta() -> Self {
        Self::new(BigRational::zero(), 1)
    }

    /// Check if this is zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.rational.is_zero() && self.delta_coeff == 0
    }

    /// Check if this is one.
    #[inline]
    pub fn is_one(&self) -> bool {
        self.rational.is_one() && self.delta_coeff == 0
    }

    /// Check if this is purely rational (delta_coeff = 0).
    #[inline]
    pub fn is_rational(&self) -> bool {
        self.delta_coeff == 0
    }

    /// Get the floor of the rational part.
    pub fn floor(&self) -> BigInt {
        crate::rational::floor(&self.rational)
    }

    /// Get the ceiling of the rational part (adjusted for delta).
    pub fn ceil(&self) -> BigInt {
        if self.delta_coeff > 0 {
            // r + δ*k where k > 0, so ceiling is ceil(r)
            crate::rational::ceil(&self.rational)
        } else if self.delta_coeff < 0 {
            // r + δ*k where k < 0, so we need floor if r is integer, else ceil
            if self.rational.is_integer() {
                crate::rational::floor(&self.rational)
            } else {
                crate::rational::ceil(&self.rational)
            }
        } else {
            crate::rational::ceil(&self.rational)
        }
    }

    /// Negate the delta-rational.
    pub fn negate(&self) -> Self {
        Self::new(-&self.rational, -self.delta_coeff)
    }

    /// Add two delta-rationals.
    pub fn add(&self, other: &Self) -> Self {
        Self::new(
            &self.rational + &other.rational,
            self.delta_coeff + other.delta_coeff,
        )
    }

    /// Subtract two delta-rationals.
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(
            &self.rational - &other.rational,
            self.delta_coeff - other.delta_coeff,
        )
    }

    /// Multiply by a rational scalar.
    pub fn mul_rational(&self, r: &BigRational) -> Self {
        if r.is_zero() {
            return Self::zero();
        }

        // (a + δ*k) * r = a*r + δ*(k*r)
        // But delta coefficient must be integer, so we only support integer r for now
        if r.is_integer() {
            let r_int = r.numer().to_string().parse::<i64>().unwrap_or(0);
            Self::new(&self.rational * r, self.delta_coeff * r_int)
        } else {
            // For non-integer rationals, we can't maintain integer delta coefficient
            // Just multiply the rational part
            Self::new(&self.rational * r, 0)
        }
    }

    /// Divide by a positive rational scalar.
    pub fn div_rational(&self, r: &BigRational) -> Option<Self> {
        if r.is_zero() {
            return None;
        }

        // (a + δ*k) / r = a/r + δ*(k/r)
        if r.is_integer() && r.is_positive() {
            let r_int = r.numer().to_string().parse::<i64>().unwrap_or(1);
            if r_int != 0 {
                Some(Self::new(&self.rational / r, self.delta_coeff / r_int))
            } else {
                None
            }
        } else {
            // For non-integer rationals, just divide the rational part
            Some(Self::new(&self.rational / r, 0))
        }
    }

    /// Compare two delta-rationals.
    pub fn cmp_delta(&self, other: &Self) -> Ordering {
        // First compare rational parts
        match self.rational.cmp(&other.rational) {
            Ordering::Equal => {
                // If rational parts are equal, compare delta coefficients
                self.delta_coeff.cmp(&other.delta_coeff)
            }
            ord => ord,
        }
    }

    /// Check if this delta-rational is positive.
    pub fn is_positive(&self) -> bool {
        self.rational.is_positive() || (self.rational.is_zero() && self.delta_coeff > 0)
    }

    /// Check if this delta-rational is negative.
    pub fn is_negative(&self) -> bool {
        self.rational.is_negative() || (self.rational.is_zero() && self.delta_coeff < 0)
    }

    /// Check if this delta-rational is non-negative.
    pub fn is_non_negative(&self) -> bool {
        !self.is_negative()
    }

    /// Check if this delta-rational is non-positive.
    pub fn is_non_positive(&self) -> bool {
        !self.is_positive()
    }

    /// Get the sign: -1, 0, or 1.
    pub fn sign(&self) -> i8 {
        if self.is_positive() {
            1
        } else if self.is_negative() {
            -1
        } else {
            0
        }
    }

    /// Convert to a lower bound for an interval (r if δ >= 0, else r - ε).
    pub fn to_lower_bound(&self) -> BigRational {
        self.rational.clone()
    }

    /// Convert to an upper bound for an interval (r if δ <= 0, else r + ε).
    pub fn to_upper_bound(&self) -> BigRational {
        self.rational.clone()
    }
}

impl Default for DeltaRational {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<BigRational> for DeltaRational {
    fn from(r: BigRational) -> Self {
        Self::from_rational(r)
    }
}

impl From<i64> for DeltaRational {
    fn from(n: i64) -> Self {
        Self::from_int(n)
    }
}

impl PartialOrd for DeltaRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl Ord for DeltaRational {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_delta(other)
    }
}

impl Neg for DeltaRational {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl Neg for &DeltaRational {
    type Output = DeltaRational;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl Add for DeltaRational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        DeltaRational::add(&self, &rhs)
    }
}

impl Add<&DeltaRational> for &DeltaRational {
    type Output = DeltaRational;

    fn add(self, rhs: &DeltaRational) -> Self::Output {
        DeltaRational::add(self, rhs)
    }
}

impl Sub for DeltaRational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        DeltaRational::sub(&self, &rhs)
    }
}

impl Sub<&DeltaRational> for &DeltaRational {
    type Output = DeltaRational;

    fn sub(self, rhs: &DeltaRational) -> Self::Output {
        DeltaRational::sub(self, rhs)
    }
}

impl fmt::Display for DeltaRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.delta_coeff == 0 {
            write!(f, "{}", self.rational)
        } else if self.rational.is_zero() {
            if self.delta_coeff == 1 {
                write!(f, "δ")
            } else if self.delta_coeff == -1 {
                write!(f, "-δ")
            } else {
                write!(f, "{}δ", self.delta_coeff)
            }
        } else if self.delta_coeff == 1 {
            write!(f, "{} + δ", self.rational)
        } else if self.delta_coeff == -1 {
            write!(f, "{} - δ", self.rational)
        } else if self.delta_coeff > 0 {
            write!(f, "{} + {}δ", self.rational, self.delta_coeff)
        } else {
            write!(f, "{} - {}δ", self.rational, -self.delta_coeff)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_delta_basic() {
        let d = DeltaRational::delta();
        assert_eq!(d.rational, BigRational::zero());
        assert_eq!(d.delta_coeff, 1);
    }

    #[test]
    fn test_delta_add() {
        let a = DeltaRational::new(rat(5), 2); // 5 + 2δ
        let b = DeltaRational::new(rat(3), -1); // 3 - δ
        let c = DeltaRational::add(&a, &b); // 8 + δ

        assert_eq!(c.rational, rat(8));
        assert_eq!(c.delta_coeff, 1);
    }

    #[test]
    fn test_delta_sub() {
        let a = DeltaRational::new(rat(5), 2); // 5 + 2δ
        let b = DeltaRational::new(rat(3), -1); // 3 - δ
        let c = DeltaRational::sub(&a, &b); // 2 + 3δ

        assert_eq!(c.rational, rat(2));
        assert_eq!(c.delta_coeff, 3);
    }

    #[test]
    fn test_delta_cmp() {
        let a = DeltaRational::new(rat(5), 0); // 5
        let b = DeltaRational::new(rat(5), 1); // 5 + δ
        let c = DeltaRational::new(rat(5), -1); // 5 - δ

        assert!(a < b);
        assert!(c < a);
        assert!(c < b);
    }

    #[test]
    fn test_delta_zero_cmp() {
        let zero = DeltaRational::zero();
        let delta = DeltaRational::delta();
        let minus_delta = DeltaRational::new(rat(0), -1);

        assert!(minus_delta < zero);
        assert!(zero < delta);
        assert!(minus_delta < delta);
    }

    #[test]
    fn test_delta_sign() {
        let pos = DeltaRational::new(rat(5), 0);
        let neg = DeltaRational::new(rat(-5), 0);
        let delta_pos = DeltaRational::delta();
        let delta_neg = DeltaRational::new(rat(0), -1);

        assert!(pos.is_positive());
        assert!(neg.is_negative());
        assert!(delta_pos.is_positive());
        assert!(delta_neg.is_negative());
    }

    #[test]
    fn test_delta_mul_rational() {
        let a = DeltaRational::new(rat(5), 2); // 5 + 2δ
        let b = a.mul_rational(&rat(3)); // 15 + 6δ

        assert_eq!(b.rational, rat(15));
        assert_eq!(b.delta_coeff, 6);
    }

    #[test]
    fn test_delta_negate() {
        let a = DeltaRational::new(rat(5), 2); // 5 + 2δ
        let b = a.negate(); // -5 - 2δ

        assert_eq!(b.rational, rat(-5));
        assert_eq!(b.delta_coeff, -2);
    }
}
