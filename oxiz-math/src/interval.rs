//! Interval arithmetic for bound propagation.
//!
//! This module provides interval arithmetic operations used for:
//! - Bound propagation in linear and nonlinear arithmetic
//! - Sign determination for polynomials
//! - Root isolation in CAD
//!
//! Reference: Z3's `math/interval/` directory.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// An endpoint of an interval, which can be finite or infinite.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Bound {
    /// Negative infinity.
    NegInf,
    /// A finite rational value.
    Finite(BigRational),
    /// Positive infinity.
    PosInf,
}

impl Bound {
    /// Create a finite bound.
    #[inline]
    pub fn finite(r: BigRational) -> Self {
        Self::Finite(r)
    }

    /// Create a finite bound from an integer.
    #[inline]
    pub fn from_int(n: i64) -> Self {
        Self::Finite(BigRational::from_integer(BigInt::from(n)))
    }

    /// Check if this bound is finite.
    #[inline]
    pub fn is_finite(&self) -> bool {
        matches!(self, Self::Finite(_))
    }

    /// Check if this bound is infinite.
    #[inline]
    pub fn is_infinite(&self) -> bool {
        !self.is_finite()
    }

    /// Check if this bound is negative infinity.
    #[inline]
    pub fn is_neg_inf(&self) -> bool {
        matches!(self, Self::NegInf)
    }

    /// Check if this bound is positive infinity.
    #[inline]
    pub fn is_pos_inf(&self) -> bool {
        matches!(self, Self::PosInf)
    }

    /// Get the finite value, or None if infinite.
    pub fn as_finite(&self) -> Option<&BigRational> {
        match self {
            Self::Finite(r) => Some(r),
            _ => None,
        }
    }

    /// Negate the bound.
    pub fn negate(&self) -> Bound {
        match self {
            Self::NegInf => Self::PosInf,
            Self::PosInf => Self::NegInf,
            Self::Finite(r) => Self::Finite(-r),
        }
    }

    /// Add two bounds.
    pub fn add(&self, other: &Bound) -> Bound {
        match (self, other) {
            (Self::NegInf, Self::PosInf) | (Self::PosInf, Self::NegInf) => {
                panic!("Undefined: inf + (-inf)")
            }
            (Self::NegInf, _) | (_, Self::NegInf) => Self::NegInf,
            (Self::PosInf, _) | (_, Self::PosInf) => Self::PosInf,
            (Self::Finite(a), Self::Finite(b)) => Self::Finite(a + b),
        }
    }

    /// Subtract two bounds.
    pub fn sub(&self, other: &Bound) -> Bound {
        self.add(&other.negate())
    }

    /// Multiply two bounds.
    pub fn mul(&self, other: &Bound) -> Bound {
        match (self, other) {
            (Self::Finite(a), Self::Finite(b)) => Self::Finite(a * b),
            (Self::Finite(a), inf) | (inf, Self::Finite(a)) => {
                if a.is_zero() {
                    // 0 * inf is undefined, but we can treat it as 0 for interval purposes
                    Self::Finite(BigRational::zero())
                } else if a.is_positive() {
                    inf.clone()
                } else {
                    inf.negate()
                }
            }
            (Self::NegInf, Self::NegInf) | (Self::PosInf, Self::PosInf) => Self::PosInf,
            (Self::NegInf, Self::PosInf) | (Self::PosInf, Self::NegInf) => Self::NegInf,
        }
    }

    /// Compare two bounds.
    pub fn cmp_bound(&self, other: &Bound) -> Ordering {
        match (self, other) {
            (Self::NegInf, Self::NegInf) => Ordering::Equal,
            (Self::NegInf, _) => Ordering::Less,
            (_, Self::NegInf) => Ordering::Greater,
            (Self::PosInf, Self::PosInf) => Ordering::Equal,
            (Self::PosInf, _) => Ordering::Greater,
            (_, Self::PosInf) => Ordering::Less,
            (Self::Finite(a), Self::Finite(b)) => a.cmp(b),
        }
    }

    /// Get the minimum of two bounds.
    pub fn min(&self, other: &Bound) -> Bound {
        if self.cmp_bound(other) == Ordering::Less {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Get the maximum of two bounds.
    pub fn max(&self, other: &Bound) -> Bound {
        if self.cmp_bound(other) == Ordering::Greater {
            self.clone()
        } else {
            other.clone()
        }
    }
}

impl PartialOrd for Bound {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl Ord for Bound {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_bound(other)
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegInf => write!(f, "-∞"),
            Self::PosInf => write!(f, "+∞"),
            Self::Finite(r) => write!(f, "{}", r),
        }
    }
}

/// An interval [lo, hi] with optional open/closed endpoints.
/// Represents the set of values x such that lo ≤ x ≤ hi (closed)
/// or lo < x < hi (open), depending on the flags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    /// Lower bound.
    pub lo: Bound,
    /// Upper bound.
    pub hi: Bound,
    /// True if lower bound is open (exclusive).
    pub lo_open: bool,
    /// True if upper bound is open (exclusive).
    pub hi_open: bool,
}

impl Interval {
    /// Create the empty interval.
    pub fn empty() -> Self {
        Self {
            lo: Bound::PosInf,
            hi: Bound::NegInf,
            lo_open: true,
            hi_open: true,
        }
    }

    /// Create the interval containing all reals (-∞, +∞).
    pub fn reals() -> Self {
        Self {
            lo: Bound::NegInf,
            hi: Bound::PosInf,
            lo_open: true,
            hi_open: true,
        }
    }

    /// Create a point interval [a, a].
    pub fn point(a: BigRational) -> Self {
        Self {
            lo: Bound::Finite(a.clone()),
            hi: Bound::Finite(a),
            lo_open: false,
            hi_open: false,
        }
    }

    /// Create a closed interval [a, b].
    pub fn closed(a: BigRational, b: BigRational) -> Self {
        Self {
            lo: Bound::Finite(a),
            hi: Bound::Finite(b),
            lo_open: false,
            hi_open: false,
        }
    }

    /// Create an open interval (a, b).
    pub fn open(a: BigRational, b: BigRational) -> Self {
        Self {
            lo: Bound::Finite(a),
            hi: Bound::Finite(b),
            lo_open: true,
            hi_open: true,
        }
    }

    /// Create a half-open interval [a, b).
    pub fn half_open_right(a: BigRational, b: BigRational) -> Self {
        Self {
            lo: Bound::Finite(a),
            hi: Bound::Finite(b),
            lo_open: false,
            hi_open: true,
        }
    }

    /// Create a half-open interval (a, b].
    pub fn half_open_left(a: BigRational, b: BigRational) -> Self {
        Self {
            lo: Bound::Finite(a),
            hi: Bound::Finite(b),
            lo_open: true,
            hi_open: false,
        }
    }

    /// Create an interval (-∞, b].
    pub fn at_most(b: BigRational) -> Self {
        Self {
            lo: Bound::NegInf,
            hi: Bound::Finite(b),
            lo_open: true,
            hi_open: false,
        }
    }

    /// Create an interval (-∞, b).
    pub fn less_than(b: BigRational) -> Self {
        Self {
            lo: Bound::NegInf,
            hi: Bound::Finite(b),
            lo_open: true,
            hi_open: true,
        }
    }

    /// Create an interval [a, +∞).
    pub fn at_least(a: BigRational) -> Self {
        Self {
            lo: Bound::Finite(a),
            hi: Bound::PosInf,
            lo_open: false,
            hi_open: true,
        }
    }

    /// Create an interval (a, +∞).
    pub fn greater_than(a: BigRational) -> Self {
        Self {
            lo: Bound::Finite(a),
            hi: Bound::PosInf,
            lo_open: true,
            hi_open: true,
        }
    }

    /// Check if the interval is empty.
    pub fn is_empty(&self) -> bool {
        match self.lo.cmp_bound(&self.hi) {
            Ordering::Greater => true,
            Ordering::Equal => self.lo_open || self.hi_open,
            Ordering::Less => false,
        }
    }

    /// Check if the interval is a single point.
    pub fn is_point(&self) -> bool {
        !self.is_empty() && self.lo == self.hi && !self.lo_open && !self.hi_open
    }

    /// Check if the interval is bounded (both endpoints finite).
    pub fn is_bounded(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    /// Check if the interval contains a value.
    pub fn contains(&self, x: &BigRational) -> bool {
        let x_bound = Bound::Finite(x.clone());
        let lo_ok = match self.lo.cmp_bound(&x_bound) {
            Ordering::Less => true,
            Ordering::Equal => !self.lo_open,
            Ordering::Greater => false,
        };
        let hi_ok = match x_bound.cmp_bound(&self.hi) {
            Ordering::Less => true,
            Ordering::Equal => !self.hi_open,
            Ordering::Greater => false,
        };
        lo_ok && hi_ok
    }

    /// Check if this interval contains zero.
    pub fn contains_zero(&self) -> bool {
        self.contains(&BigRational::zero())
    }

    /// Check if all values in the interval are positive.
    pub fn is_positive(&self) -> bool {
        match &self.lo {
            Bound::NegInf => false,
            Bound::PosInf => self.is_empty(),
            Bound::Finite(r) => {
                if r.is_positive() {
                    true
                } else if r.is_zero() {
                    self.lo_open
                } else {
                    false
                }
            }
        }
    }

    /// Check if all values in the interval are negative.
    pub fn is_negative(&self) -> bool {
        match &self.hi {
            Bound::PosInf => false,
            Bound::NegInf => self.is_empty(),
            Bound::Finite(r) => {
                if r.is_negative() {
                    true
                } else if r.is_zero() {
                    self.hi_open
                } else {
                    false
                }
            }
        }
    }

    /// Check if all values are non-negative.
    pub fn is_non_negative(&self) -> bool {
        match &self.lo {
            Bound::NegInf => false,
            Bound::PosInf => true,
            Bound::Finite(r) => !r.is_negative(),
        }
    }

    /// Check if all values are non-positive.
    pub fn is_non_positive(&self) -> bool {
        match &self.hi {
            Bound::PosInf => false,
            Bound::NegInf => true,
            Bound::Finite(r) => !r.is_positive(),
        }
    }

    /// Get the sign of the interval.
    /// Returns Some(1) if all positive, Some(-1) if all negative, Some(0) if only zero.
    /// Returns None if the interval contains values of different signs.
    pub fn sign(&self) -> Option<i8> {
        if self.is_empty() {
            return None;
        }
        if self.is_point()
            && let Bound::Finite(r) = &self.lo
        {
            return Some(if r.is_positive() {
                1
            } else if r.is_negative() {
                -1
            } else {
                0
            });
        }
        if self.is_positive() {
            Some(1)
        } else if self.is_negative() {
            Some(-1)
        } else {
            None
        }
    }

    /// Negate the interval.
    pub fn negate(&self) -> Interval {
        Interval {
            lo: self.hi.negate(),
            hi: self.lo.negate(),
            lo_open: self.hi_open,
            hi_open: self.lo_open,
        }
    }

    /// Add two intervals.
    pub fn add(&self, other: &Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::empty();
        }
        Interval {
            lo: self.lo.add(&other.lo),
            hi: self.hi.add(&other.hi),
            lo_open: self.lo_open || other.lo_open,
            hi_open: self.hi_open || other.hi_open,
        }
    }

    /// Subtract two intervals.
    pub fn sub(&self, other: &Interval) -> Interval {
        self.add(&other.negate())
    }

    /// Multiply two intervals.
    pub fn mul(&self, other: &Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::empty();
        }

        // Handle cases with infinity more carefully
        // For bounded intervals, we compute all four products
        // For unbounded intervals, the logic is more complex

        // Compute all four products and their openness
        let products = [
            (&self.lo, &other.lo, self.lo_open || other.lo_open),
            (&self.lo, &other.hi, self.lo_open || other.hi_open),
            (&self.hi, &other.lo, self.hi_open || other.lo_open),
            (&self.hi, &other.hi, self.hi_open || other.hi_open),
        ];

        // Compute actual product values
        let mut bounds: Vec<(Bound, bool)> = products
            .iter()
            .map(|(a, b, open)| (a.mul(b), *open))
            .collect();

        // Find min and max
        bounds.sort_by(|a, b| a.0.cmp_bound(&b.0));

        let (lo, lo_open) = bounds
            .first()
            .expect("collection validated to be non-empty")
            .clone();
        let (hi, hi_open) = bounds
            .last()
            .expect("collection validated to be non-empty")
            .clone();

        Interval {
            lo,
            hi,
            lo_open,
            hi_open,
        }
    }

    /// Divide two intervals.
    /// Returns None if division by zero occurs (divisor contains zero).
    pub fn div(&self, other: &Interval) -> Option<Interval> {
        if self.is_empty() || other.is_empty() {
            return Some(Interval::empty());
        }

        // Check if divisor contains zero
        if other.contains_zero() {
            // Division by zero is undefined
            // In interval arithmetic, we need to split the interval or return extended intervals
            // For simplicity, return None to indicate undefined
            return None;
        }

        // Division: [a,b] / [c,d] = [a,b] * [1/d, 1/c]
        // When c,d don't contain zero, we compute reciprocals and multiply

        // Compute reciprocal of other
        let recip_lo = match &other.hi {
            Bound::NegInf | Bound::PosInf => Bound::Finite(BigRational::zero()),
            Bound::Finite(r) => {
                if r.is_zero() {
                    return None; // Contains zero
                }
                Bound::Finite(BigRational::one() / r)
            }
        };

        let recip_hi = match &other.lo {
            Bound::NegInf | Bound::PosInf => Bound::Finite(BigRational::zero()),
            Bound::Finite(r) => {
                if r.is_zero() {
                    return None; // Contains zero
                }
                Bound::Finite(BigRational::one() / r)
            }
        };

        let recip = Interval {
            lo: recip_lo,
            hi: recip_hi,
            lo_open: other.hi_open,
            hi_open: other.lo_open,
        };

        Some(self.mul(&recip))
    }

    /// Compute the intersection of two intervals.
    pub fn intersect(&self, other: &Interval) -> Interval {
        if self.is_empty() || other.is_empty() {
            return Interval::empty();
        }

        let (lo, lo_open) = match self.lo.cmp_bound(&other.lo) {
            Ordering::Less => (other.lo.clone(), other.lo_open),
            Ordering::Greater => (self.lo.clone(), self.lo_open),
            Ordering::Equal => (self.lo.clone(), self.lo_open || other.lo_open),
        };

        let (hi, hi_open) = match self.hi.cmp_bound(&other.hi) {
            Ordering::Less => (self.hi.clone(), self.hi_open),
            Ordering::Greater => (other.hi.clone(), other.hi_open),
            Ordering::Equal => (self.hi.clone(), self.hi_open || other.hi_open),
        };

        Interval {
            lo,
            hi,
            lo_open,
            hi_open,
        }
    }

    /// Compute the union of two intervals (if contiguous).
    /// Returns None if the intervals are not contiguous.
    pub fn union(&self, other: &Interval) -> Option<Interval> {
        if self.is_empty() {
            return Some(other.clone());
        }
        if other.is_empty() {
            return Some(self.clone());
        }

        // Check if intervals overlap or are adjacent
        let intersects = !self.intersect(other).is_empty();
        let adjacent_left = self.hi == other.lo && (!self.hi_open || !other.lo_open);
        let adjacent_right = other.hi == self.lo && (!other.hi_open || !self.lo_open);

        if !intersects && !adjacent_left && !adjacent_right {
            return None;
        }

        let (lo, lo_open) = match self.lo.cmp_bound(&other.lo) {
            Ordering::Less => (self.lo.clone(), self.lo_open),
            Ordering::Greater => (other.lo.clone(), other.lo_open),
            Ordering::Equal => (self.lo.clone(), self.lo_open && other.lo_open),
        };

        let (hi, hi_open) = match self.hi.cmp_bound(&other.hi) {
            Ordering::Greater => (self.hi.clone(), self.hi_open),
            Ordering::Less => (other.hi.clone(), other.hi_open),
            Ordering::Equal => (self.hi.clone(), self.hi_open && other.hi_open),
        };

        Some(Interval {
            lo,
            hi,
            lo_open,
            hi_open,
        })
    }

    /// Compute the power of an interval.
    pub fn pow(&self, n: u32) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }
        if n == 0 {
            return Interval::point(BigRational::one());
        }
        if n == 1 {
            return self.clone();
        }

        if n.is_multiple_of(2) {
            // Even power: result is non-negative
            // [a,b]^n depends on whether interval contains 0
            if self.contains_zero() {
                // [min(|a|,|b|)^n, max(|a|,|b|)^n] but includes 0
                let lo_abs = match &self.lo {
                    Bound::Finite(r) => r.abs(),
                    Bound::NegInf => return Interval::at_least(BigRational::zero()),
                    Bound::PosInf => unreachable!(),
                };
                let hi_abs = match &self.hi {
                    Bound::Finite(r) => r.abs(),
                    Bound::PosInf => return Interval::at_least(BigRational::zero()),
                    Bound::NegInf => unreachable!(),
                };
                let max_abs = lo_abs.max(hi_abs);
                Interval {
                    lo: Bound::Finite(BigRational::zero()),
                    hi: Bound::Finite(pow_rational(&max_abs, n)),
                    lo_open: false,
                    hi_open: self.lo_open && self.hi_open,
                }
            } else if self.is_positive() {
                // All positive: [a^n, b^n]
                let lo_val = match &self.lo {
                    Bound::Finite(r) => pow_rational(r, n),
                    _ => return Interval::at_least(BigRational::zero()),
                };
                let hi_val = match &self.hi {
                    Bound::Finite(r) => pow_rational(r, n),
                    _ => return Interval::at_least(lo_val),
                };
                Interval {
                    lo: Bound::Finite(lo_val),
                    hi: Bound::Finite(hi_val),
                    lo_open: self.lo_open,
                    hi_open: self.hi_open,
                }
            } else {
                // All negative: [b^n, a^n]
                let lo_val = match &self.hi {
                    Bound::Finite(r) => pow_rational(r, n),
                    _ => return Interval::at_least(BigRational::zero()),
                };
                let hi_val = match &self.lo {
                    Bound::Finite(r) => pow_rational(r, n),
                    _ => return Interval::at_least(lo_val),
                };
                Interval {
                    lo: Bound::Finite(lo_val),
                    hi: Bound::Finite(hi_val),
                    lo_open: self.hi_open,
                    hi_open: self.lo_open,
                }
            }
        } else {
            // Odd power: preserves sign
            // [a^n, b^n]
            let lo_val = match &self.lo {
                Bound::NegInf => Bound::NegInf,
                Bound::PosInf => Bound::PosInf,
                Bound::Finite(r) => Bound::Finite(pow_rational(r, n)),
            };
            let hi_val = match &self.hi {
                Bound::NegInf => Bound::NegInf,
                Bound::PosInf => Bound::PosInf,
                Bound::Finite(r) => Bound::Finite(pow_rational(r, n)),
            };
            Interval {
                lo: lo_val,
                hi: hi_val,
                lo_open: self.lo_open,
                hi_open: self.hi_open,
            }
        }
    }

    /// Get the midpoint of the interval (if bounded).
    pub fn midpoint(&self) -> Option<BigRational> {
        match (&self.lo, &self.hi) {
            (Bound::Finite(a), Bound::Finite(b)) => {
                Some((a + b) / BigRational::from_integer(BigInt::from(2)))
            }
            _ => None,
        }
    }

    /// Get the width of the interval (if bounded).
    pub fn width(&self) -> Option<BigRational> {
        match (&self.lo, &self.hi) {
            (Bound::Finite(a), Bound::Finite(b)) => Some(b - a),
            _ => None,
        }
    }

    /// Split the interval at a point.
    pub fn split(&self, at: &BigRational) -> (Interval, Interval) {
        if !self.contains(at) {
            return (self.clone(), Interval::empty());
        }

        let left = Interval {
            lo: self.lo.clone(),
            hi: Bound::Finite(at.clone()),
            lo_open: self.lo_open,
            hi_open: false,
        };

        let right = Interval {
            lo: Bound::Finite(at.clone()),
            hi: self.hi.clone(),
            lo_open: false,
            hi_open: self.hi_open,
        };

        (left, right)
    }

    /// Convex hull of two intervals (always returns a single interval).
    /// This is the smallest interval containing both inputs.
    pub fn hull(&self, other: &Interval) -> Interval {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        let (lo, lo_open) = match self.lo.cmp_bound(&other.lo) {
            Ordering::Less => (self.lo.clone(), self.lo_open),
            Ordering::Greater => (other.lo.clone(), other.lo_open),
            Ordering::Equal => (self.lo.clone(), self.lo_open && other.lo_open),
        };

        let (hi, hi_open) = match self.hi.cmp_bound(&other.hi) {
            Ordering::Greater => (self.hi.clone(), self.hi_open),
            Ordering::Less => (other.hi.clone(), other.hi_open),
            Ordering::Equal => (self.hi.clone(), self.hi_open && other.hi_open),
        };

        Interval {
            lo,
            hi,
            lo_open,
            hi_open,
        }
    }

    /// Tighten lower bound to be at least the given value.
    /// Returns a new interval with the tightened bound, or empty if inconsistent.
    pub fn tighten_lower(&self, new_lo: Bound, new_lo_open: bool) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }

        let (lo, lo_open) = match self.lo.cmp_bound(&new_lo) {
            Ordering::Less => (new_lo, new_lo_open),
            Ordering::Greater => (self.lo.clone(), self.lo_open),
            Ordering::Equal => (self.lo.clone(), self.lo_open || new_lo_open),
        };

        Interval {
            lo,
            hi: self.hi.clone(),
            lo_open,
            hi_open: self.hi_open,
        }
    }

    /// Tighten upper bound to be at most the given value.
    /// Returns a new interval with the tightened bound, or empty if inconsistent.
    pub fn tighten_upper(&self, new_hi: Bound, new_hi_open: bool) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }

        let (hi, hi_open) = match self.hi.cmp_bound(&new_hi) {
            Ordering::Greater => (new_hi, new_hi_open),
            Ordering::Less => (self.hi.clone(), self.hi_open),
            Ordering::Equal => (self.hi.clone(), self.hi_open || new_hi_open),
        };

        Interval {
            lo: self.lo.clone(),
            hi,
            lo_open: self.lo_open,
            hi_open,
        }
    }

    /// Propagate bounds from constraint: x + y ∈ result_interval.
    /// Given x and result, tighten y's bounds.
    /// Returns the tightened interval for y.
    pub fn propagate_add(x: &Interval, result: &Interval) -> Interval {
        // y ∈ result - x
        result.sub(x)
    }

    /// Propagate bounds from constraint: x - y ∈ result_interval.
    /// Given x and result, tighten y's bounds.
    /// Returns the tightened interval for y.
    pub fn propagate_sub_left(x: &Interval, result: &Interval) -> Interval {
        // x - y = result => y = x - result
        x.sub(result)
    }

    /// Propagate bounds from constraint: x - y ∈ result_interval.
    /// Given y and result, tighten x's bounds.
    /// Returns the tightened interval for x.
    pub fn propagate_sub_right(y: &Interval, result: &Interval) -> Interval {
        // x - y = result => x = result + y
        result.add(y)
    }

    /// Propagate bounds from constraint: x * y ∈ result_interval.
    /// Given x and result, tighten y's bounds (if x doesn't contain 0).
    /// Returns None if x contains 0, otherwise returns the tightened interval for y.
    pub fn propagate_mul(x: &Interval, result: &Interval) -> Option<Interval> {
        // y ∈ result / x
        Interval::div(result, x)
    }

    /// Propagate bounds from constraint: x / y ∈ result_interval.
    /// Given x and result, tighten y's bounds (if result doesn't contain 0).
    /// Returns None if propagation is impossible, otherwise returns the tightened interval for y.
    pub fn propagate_div_left(x: &Interval, result: &Interval) -> Option<Interval> {
        // x / y = result => y = x / result
        Interval::div(x, result)
    }

    /// Propagate bounds from constraint: x / y ∈ result_interval.
    /// Given y and result, tighten x's bounds.
    /// Returns the tightened interval for x.
    pub fn propagate_div_right(y: &Interval, result: &Interval) -> Interval {
        // x / y = result => x = result * y
        result.mul(y)
    }

    /// Check if this interval is a subset of another interval.
    pub fn is_subset_of(&self, other: &Interval) -> bool {
        if self.is_empty() {
            return true;
        }
        if other.is_empty() {
            return false;
        }

        // Check lower bound
        let lo_ok = match self.lo.cmp_bound(&other.lo) {
            Ordering::Less => false,
            Ordering::Greater => true,
            Ordering::Equal => !self.lo_open || other.lo_open,
        };

        // Check upper bound
        let hi_ok = match self.hi.cmp_bound(&other.hi) {
            Ordering::Greater => false,
            Ordering::Less => true,
            Ordering::Equal => !self.hi_open || other.hi_open,
        };

        lo_ok && hi_ok
    }

    /// Check if two intervals overlap (have non-empty intersection).
    pub fn overlaps(&self, other: &Interval) -> bool {
        !self.intersect(other).is_empty()
    }

    /// Widen the interval by a given amount on each side.
    pub fn widen(&self, amount: &BigRational) -> Interval {
        if self.is_empty() {
            return Interval::empty();
        }

        let lo = match &self.lo {
            Bound::Finite(r) => Bound::Finite(r - amount),
            bound => bound.clone(),
        };

        let hi = match &self.hi {
            Bound::Finite(r) => Bound::Finite(r + amount),
            bound => bound.clone(),
        };

        Interval {
            lo,
            hi,
            lo_open: self.lo_open,
            hi_open: self.hi_open,
        }
    }
}

/// Compute r^n for a rational number.
fn pow_rational(r: &BigRational, n: u32) -> BigRational {
    if n == 0 {
        return BigRational::one();
    }

    let mut result = BigRational::one();
    let mut base = r.clone();
    let mut exp = n;

    while exp > 0 {
        if exp & 1 == 1 {
            result = &result * &base;
        }
        base = &base * &base;
        exp >>= 1;
    }

    result
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "∅")
        } else {
            let lo_bracket = if self.lo_open { '(' } else { '[' };
            let hi_bracket = if self.hi_open { ')' } else { ']' };
            write!(f, "{}{}, {}{}", lo_bracket, self.lo, self.hi, hi_bracket)
        }
    }
}

impl Default for Interval {
    fn default() -> Self {
        Self::reals()
    }
}

impl Neg for Interval {
    type Output = Interval;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl Neg for &Interval {
    type Output = Interval;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl Add for Interval {
    type Output = Interval;

    fn add(self, rhs: Self) -> Self::Output {
        Interval::add(&self, &rhs)
    }
}

impl Add<&Interval> for &Interval {
    type Output = Interval;

    fn add(self, rhs: &Interval) -> Self::Output {
        Interval::add(self, rhs)
    }
}

impl Sub for Interval {
    type Output = Interval;

    fn sub(self, rhs: Self) -> Self::Output {
        Interval::sub(&self, &rhs)
    }
}

impl Sub<&Interval> for &Interval {
    type Output = Interval;

    fn sub(self, rhs: &Interval) -> Self::Output {
        Interval::sub(self, rhs)
    }
}

impl Mul for Interval {
    type Output = Interval;

    fn mul(self, rhs: Self) -> Self::Output {
        Interval::mul(&self, &rhs)
    }
}

impl Mul<&Interval> for &Interval {
    type Output = Interval;

    fn mul(self, rhs: &Interval) -> Self::Output {
        Interval::mul(self, rhs)
    }
}

impl Div for Interval {
    type Output = Option<Interval>;

    fn div(self, rhs: Self) -> Self::Output {
        Interval::div(&self, &rhs)
    }
}

impl Div<&Interval> for &Interval {
    type Output = Option<Interval>;

    fn div(self, rhs: &Interval) -> Self::Output {
        Interval::div(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_interval_empty() {
        let i = Interval::empty();
        assert!(i.is_empty());
        assert!(!i.contains(&rat(0)));
    }

    #[test]
    fn test_interval_point() {
        let i = Interval::point(rat(5));
        assert!(i.is_point());
        assert!(i.contains(&rat(5)));
        assert!(!i.contains(&rat(4)));
        assert!(i.is_positive());
    }

    #[test]
    fn test_interval_closed() {
        let i = Interval::closed(rat(1), rat(5));
        assert!(i.contains(&rat(1)));
        assert!(i.contains(&rat(3)));
        assert!(i.contains(&rat(5)));
        assert!(!i.contains(&rat(0)));
        assert!(!i.contains(&rat(6)));
    }

    #[test]
    fn test_interval_open() {
        let i = Interval::open(rat(1), rat(5));
        assert!(!i.contains(&rat(1)));
        assert!(i.contains(&rat(3)));
        assert!(!i.contains(&rat(5)));
    }

    #[test]
    fn test_interval_add() {
        let a = Interval::closed(rat(1), rat(3));
        let b = Interval::closed(rat(2), rat(4));
        let c = Interval::add(&a, &b);
        assert!(c.contains(&rat(3)));
        assert!(c.contains(&rat(7)));
        assert!(!c.contains(&rat(2)));
        assert!(!c.contains(&rat(8)));
    }

    #[test]
    fn test_interval_mul() {
        let a = Interval::closed(rat(2), rat(3));
        let b = Interval::closed(rat(4), rat(5));
        let c = Interval::mul(&a, &b);
        // [2,3] * [4,5] = [8, 15]
        assert!(c.contains(&rat(8)));
        assert!(c.contains(&rat(15)));
        assert!(!c.contains(&rat(7)));
        assert!(!c.contains(&rat(16)));
    }

    #[test]
    fn test_interval_mul_mixed_signs() {
        let a = Interval::closed(rat(-2), rat(3));
        let b = Interval::closed(rat(-1), rat(2));
        let c = Interval::mul(&a, &b);
        // Products are: -2*-1=2, -2*2=-4, 3*-1=-3, 3*2=6
        // So result is [-4, 6]
        assert!(c.contains(&rat(-4)));
        assert!(c.contains(&rat(6)));
        assert!(c.contains(&rat(0)));
        assert!(!c.contains(&rat(-5)));
        assert!(!c.contains(&rat(7)));
    }

    #[test]
    fn test_interval_negate() {
        let a = Interval::closed(rat(1), rat(5));
        let b = a.negate();
        assert!(b.contains(&rat(-5)));
        assert!(b.contains(&rat(-1)));
        assert!(!b.contains(&rat(0)));
    }

    #[test]
    fn test_interval_intersect() {
        let a = Interval::closed(rat(1), rat(5));
        let b = Interval::closed(rat(3), rat(7));
        let c = a.intersect(&b);
        // [1,5] ∩ [3,7] = [3,5]
        assert!(c.contains(&rat(3)));
        assert!(c.contains(&rat(5)));
        assert!(!c.contains(&rat(2)));
        assert!(!c.contains(&rat(6)));
    }

    #[test]
    fn test_interval_pow_even() {
        let a = Interval::closed(rat(-2), rat(3));
        let b = a.pow(2);
        // [-2,3]^2 = [0, 9]
        assert!(b.contains(&rat(0)));
        assert!(b.contains(&rat(9)));
        assert!(!b.contains(&rat(-1)));
        assert!(!b.contains(&rat(10)));
    }

    #[test]
    fn test_interval_pow_odd() {
        let a = Interval::closed(rat(-2), rat(3));
        let b = a.pow(3);
        // [-2,3]^3 = [-8, 27]
        assert!(b.contains(&rat(-8)));
        assert!(b.contains(&rat(27)));
        assert!(!b.contains(&rat(-9)));
        assert!(!b.contains(&rat(28)));
    }

    #[test]
    fn test_interval_sign() {
        assert_eq!(Interval::closed(rat(1), rat(5)).sign(), Some(1));
        assert_eq!(Interval::closed(rat(-5), rat(-1)).sign(), Some(-1));
        assert_eq!(Interval::closed(rat(-1), rat(1)).sign(), None);
        assert_eq!(Interval::point(rat(0)).sign(), Some(0));
    }

    #[test]
    fn test_interval_unbounded() {
        let a = Interval::at_least(rat(0));
        assert!(a.is_non_negative());
        assert!(a.contains(&rat(0)));
        assert!(a.contains(&rat(1000)));
        assert!(!a.contains(&rat(-1)));

        let b = Interval::less_than(rat(0));
        assert!(b.is_negative());
        assert!(!b.contains(&rat(0)));
        assert!(b.contains(&rat(-1)));
    }

    #[test]
    fn test_interval_midpoint() {
        let i = Interval::closed(rat(2), rat(8));
        assert_eq!(i.midpoint(), Some(rat(5)));
    }

    #[test]
    fn test_interval_div() {
        // [4, 12] / [2, 3] = [4/3, 6]
        let a = Interval::closed(rat(4), rat(12));
        let b = Interval::closed(rat(2), rat(3));
        let c = Interval::div(&a, &b).expect("Division should succeed");
        // 4/3 ≈ 1.333..., 12/2 = 6
        assert!(c.contains(&BigRational::new(BigInt::from(4), BigInt::from(3))));
        assert!(c.contains(&rat(6)));
    }

    #[test]
    fn test_interval_div_by_zero() {
        // Division by interval containing zero should return None
        let a = Interval::closed(rat(1), rat(2));
        let b = Interval::closed(rat(-1), rat(1)); // Contains zero
        assert!(Interval::div(&a, &b).is_none());
    }

    #[test]
    fn test_interval_div_positive() {
        // [6, 12] / [2, 3] = [2, 6]
        let a = Interval::closed(rat(6), rat(12));
        let b = Interval::closed(rat(2), rat(3));
        let c = Interval::div(&a, &b).expect("Division should succeed");
        assert!(c.contains(&rat(2)));
        assert!(c.contains(&rat(6)));
    }

    #[test]
    fn test_interval_hull() {
        let a = Interval::closed(rat(1), rat(3));
        let b = Interval::closed(rat(5), rat(7));
        let hull = a.hull(&b);
        // Hull should be [1, 7]
        assert_eq!(hull.lo, Bound::Finite(rat(1)));
        assert_eq!(hull.hi, Bound::Finite(rat(7)));
        assert!(hull.contains(&rat(1)));
        assert!(hull.contains(&rat(4)));
        assert!(hull.contains(&rat(7)));
    }

    #[test]
    fn test_interval_tighten_lower() {
        let i = Interval::closed(rat(1), rat(10));
        let tightened = i.tighten_lower(Bound::Finite(rat(5)), false);
        // Should become [5, 10]
        assert_eq!(tightened.lo, Bound::Finite(rat(5)));
        assert_eq!(tightened.hi, Bound::Finite(rat(10)));
        assert!(!tightened.contains(&rat(4)));
        assert!(tightened.contains(&rat(5)));
    }

    #[test]
    fn test_interval_tighten_upper() {
        let i = Interval::closed(rat(1), rat(10));
        let tightened = i.tighten_upper(Bound::Finite(rat(6)), false);
        // Should become [1, 6]
        assert_eq!(tightened.lo, Bound::Finite(rat(1)));
        assert_eq!(tightened.hi, Bound::Finite(rat(6)));
        assert!(tightened.contains(&rat(6)));
        assert!(!tightened.contains(&rat(7)));
    }

    #[test]
    fn test_interval_propagate_add() {
        // x + y = result, x = [1, 2], result = [5, 8]
        // Then y = result - x = [5, 8] - [1, 2] = [3, 7]
        let x = Interval::closed(rat(1), rat(2));
        let result = Interval::closed(rat(5), rat(8));
        let y = Interval::propagate_add(&x, &result);
        assert!(y.contains(&rat(3)));
        assert!(y.contains(&rat(7)));
        assert!(!y.contains(&rat(2)));
        assert!(!y.contains(&rat(8)));
    }

    #[test]
    fn test_interval_propagate_sub() {
        // x - y = result, x = [10, 15], result = [2, 5]
        // Then y = x - result = [10, 15] - [2, 5] = [5, 13]
        let x = Interval::closed(rat(10), rat(15));
        let result = Interval::closed(rat(2), rat(5));
        let y = Interval::propagate_sub_left(&x, &result);
        assert!(y.contains(&rat(5)));
        assert!(y.contains(&rat(13)));
    }

    #[test]
    fn test_interval_propagate_mul() {
        // x * y = result, x = [2, 3], result = [10, 18]
        // Then y = result / x = [10, 18] / [2, 3]
        let x = Interval::closed(rat(2), rat(3));
        let result = Interval::closed(rat(10), rat(18));
        let y = Interval::propagate_mul(&x, &result).expect("Propagation should succeed");
        // y should be roughly [10/3, 9] = [3.33..., 9]
        assert!(y.contains(&rat(5)));
    }

    #[test]
    fn test_interval_is_subset_of() {
        let a = Interval::closed(rat(2), rat(5));
        let b = Interval::closed(rat(1), rat(10));
        assert!(a.is_subset_of(&b));
        assert!(!b.is_subset_of(&a));

        let c = Interval::closed(rat(1), rat(10));
        assert!(c.is_subset_of(&c));
    }

    #[test]
    fn test_interval_overlaps() {
        let a = Interval::closed(rat(1), rat(5));
        let b = Interval::closed(rat(3), rat(7));
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));

        let c = Interval::closed(rat(10), rat(15));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_interval_widen() {
        let i = Interval::closed(rat(5), rat(10));
        let widened = i.widen(&rat(2));
        // Should become [3, 12]
        assert_eq!(widened.lo, Bound::Finite(rat(3)));
        assert_eq!(widened.hi, Bound::Finite(rat(12)));
        assert!(widened.contains(&rat(3)));
        assert!(widened.contains(&rat(12)));
        assert!(!widened.contains(&rat(2)));
        assert!(!widened.contains(&rat(13)));
    }
}
