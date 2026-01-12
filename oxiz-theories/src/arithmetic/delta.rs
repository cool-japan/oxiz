//! Delta-rational numbers for strict inequalities
//!
//! A delta-rational represents a value of the form `r + k*δ` where:
//! - r is a rational number (the "real" part)
//! - k is a rational number (the "delta" coefficient)
//! - δ is an infinitesimally small positive value
//!
//! This allows exact representation of strict inequalities in LRA:
//! - `x < c` becomes `x <= c - δ` (represented as (c, -1))
//! - `x > c` becomes `x >= c + δ` (represented as (c, 1))

use num_rational::Rational64;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A delta-rational number: represents `real + delta * δ` where δ is infinitesimal
#[derive(Debug, Clone, Copy, Default)]
pub struct DeltaRational {
    /// The real part
    pub real: Rational64,
    /// The delta coefficient (multiplied by infinitesimal δ)
    pub delta: Rational64,
}

impl DeltaRational {
    /// Create a new delta-rational from components
    #[must_use]
    pub const fn new(real: Rational64, delta: Rational64) -> Self {
        Self { real, delta }
    }

    /// Create from a rational (delta = 0)
    #[must_use]
    pub fn from_rational(r: Rational64) -> Self {
        Self {
            real: r,
            delta: Rational64::zero(),
        }
    }

    /// Create zero
    #[must_use]
    pub fn zero() -> Self {
        Self {
            real: Rational64::zero(),
            delta: Rational64::zero(),
        }
    }

    /// Create a positive infinitesimal (0 + δ)
    #[must_use]
    pub fn epsilon() -> Self {
        Self {
            real: Rational64::zero(),
            delta: Rational64::one(),
        }
    }

    /// Create a negative infinitesimal (0 - δ)
    #[must_use]
    pub fn neg_epsilon() -> Self {
        Self {
            real: Rational64::zero(),
            delta: -Rational64::one(),
        }
    }

    /// Check if this is exactly zero
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.real.is_zero() && self.delta.is_zero()
    }

    /// Check if this is positive (greater than zero)
    #[must_use]
    pub fn is_positive(&self) -> bool {
        match self.real.cmp(&Rational64::zero()) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => self.delta > Rational64::zero(),
        }
    }

    /// Check if this is negative (less than zero)
    #[must_use]
    pub fn is_negative(&self) -> bool {
        match self.real.cmp(&Rational64::zero()) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => self.delta < Rational64::zero(),
        }
    }

    /// Check if this is non-negative (>= 0)
    #[must_use]
    pub fn is_non_negative(&self) -> bool {
        !self.is_negative()
    }

    /// Check if this is non-positive (<= 0)
    #[must_use]
    pub fn is_non_positive(&self) -> bool {
        !self.is_positive()
    }

    /// Get the floor (largest integer <= this value)
    #[must_use]
    pub fn floor(&self) -> i64 {
        let real_floor = self.real.floor().to_integer();
        // If real is exactly an integer and delta is negative, floor is real - 1
        if self.real.fract().is_zero() && self.delta < Rational64::zero() {
            real_floor - 1
        } else {
            real_floor
        }
    }

    /// Get the ceiling (smallest integer >= this value)
    #[must_use]
    pub fn ceil(&self) -> i64 {
        let real_ceil = self.real.ceil().to_integer();
        // If real is exactly an integer and delta is positive, ceil is real + 1
        if self.real.fract().is_zero() && self.delta > Rational64::zero() {
            real_ceil + 1
        } else {
            real_ceil
        }
    }
}

impl From<Rational64> for DeltaRational {
    fn from(r: Rational64) -> Self {
        Self::from_rational(r)
    }
}

impl From<i64> for DeltaRational {
    fn from(n: i64) -> Self {
        Self::from_rational(Rational64::from_integer(n))
    }
}

impl PartialEq for DeltaRational {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real && self.delta == other.delta
    }
}

impl Eq for DeltaRational {}

impl PartialOrd for DeltaRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeltaRational {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.real.cmp(&other.real) {
            Ordering::Equal => self.delta.cmp(&other.delta),
            other => other,
        }
    }
}

impl Neg for DeltaRational {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            delta: -self.delta,
        }
    }
}

impl Add for DeltaRational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            delta: self.delta + rhs.delta,
        }
    }
}

impl AddAssign for DeltaRational {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.delta += rhs.delta;
    }
}

impl Sub for DeltaRational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            delta: self.delta - rhs.delta,
        }
    }
}

impl SubAssign for DeltaRational {
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.delta -= rhs.delta;
    }
}

impl Mul<Rational64> for DeltaRational {
    type Output = Self;

    fn mul(self, rhs: Rational64) -> Self::Output {
        Self {
            real: self.real * rhs,
            delta: self.delta * rhs,
        }
    }
}

impl MulAssign<Rational64> for DeltaRational {
    fn mul_assign(&mut self, rhs: Rational64) {
        self.real *= rhs;
        self.delta *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_rational_basic() {
        let a = DeltaRational::from_rational(Rational64::from_integer(5));
        let b = DeltaRational::from_rational(Rational64::from_integer(3));

        assert!(a > b);
        assert_eq!(a - b, DeltaRational::from(2));
    }

    #[test]
    fn test_delta_rational_with_epsilon() {
        let five = DeltaRational::from(5);
        let five_minus_eps = DeltaRational::new(Rational64::from_integer(5), -Rational64::one());
        let five_plus_eps = DeltaRational::new(Rational64::from_integer(5), Rational64::one());

        assert!(five_minus_eps < five);
        assert!(five < five_plus_eps);
        assert!(five_minus_eps < five_plus_eps);
    }

    #[test]
    fn test_delta_is_positive_negative() {
        let eps = DeltaRational::epsilon();
        let neg_eps = DeltaRational::neg_epsilon();
        let zero = DeltaRational::zero();

        assert!(eps.is_positive());
        assert!(!eps.is_negative());

        assert!(neg_eps.is_negative());
        assert!(!neg_eps.is_positive());

        assert!(zero.is_zero());
        assert!(!zero.is_positive());
        assert!(!zero.is_negative());
    }

    #[test]
    fn test_delta_floor_ceil() {
        // 5 - ε should have floor 4, ceil 5
        let five_minus_eps = DeltaRational::new(Rational64::from_integer(5), -Rational64::one());
        assert_eq!(five_minus_eps.floor(), 4);
        assert_eq!(five_minus_eps.ceil(), 5);

        // 5 + ε should have floor 5, ceil 6
        let five_plus_eps = DeltaRational::new(Rational64::from_integer(5), Rational64::one());
        assert_eq!(five_plus_eps.floor(), 5);
        assert_eq!(five_plus_eps.ceil(), 6);

        // 5.5 should have floor 5, ceil 6 (delta doesn't matter)
        let five_point_five = DeltaRational::from_rational(Rational64::new(11, 2));
        assert_eq!(five_point_five.floor(), 5);
        assert_eq!(five_point_five.ceil(), 6);
    }

    #[test]
    fn test_delta_arithmetic() {
        let a = DeltaRational::new(Rational64::from_integer(3), Rational64::one());
        let b = DeltaRational::new(Rational64::from_integer(2), -Rational64::one());

        // (3 + δ) + (2 - δ) = 5
        let sum = a + b;
        assert_eq!(sum.real, Rational64::from_integer(5));
        assert_eq!(sum.delta, Rational64::zero());

        // (3 + δ) - (2 - δ) = 1 + 2δ
        let diff = a - b;
        assert_eq!(diff.real, Rational64::from_integer(1));
        assert_eq!(diff.delta, Rational64::from_integer(2));

        // (3 + δ) * 2 = 6 + 2δ
        let scaled = a * Rational64::from_integer(2);
        assert_eq!(scaled.real, Rational64::from_integer(6));
        assert_eq!(scaled.delta, Rational64::from_integer(2));
    }
}
