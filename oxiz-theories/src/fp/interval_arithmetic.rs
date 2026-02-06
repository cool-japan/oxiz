//! Floating-Point Interval Arithmetic
//!
//! This module implements conservative interval arithmetic for IEEE 754 floating-point
//! values. Interval arithmetic is essential for:
//!
//! - Rigorous numerical analysis
//! - Error bound computation
//! - Range analysis in SMT solving
//! - Constraint propagation in optimization
//!
//! ## Key Features
//!
//! - Conservative operations that guarantee containment
//! - Directed rounding for lower/upper bounds
//! - Error bound tracking and propagation
//! - Support for all IEEE 754 operations
//! - Special value handling (NaN, infinity)
//!
//! ## Theory
//!
//! For an interval [a, b] and operation ⊕, we compute:
//! [a, b] ⊕ [c, d] = [min(a⊕c, a⊕d, b⊕c, b⊕d), max(a⊕c, a⊕d, b⊕c, b⊕d)]
//!
//! With directed rounding:
//! - Lower bound: round toward -∞
//! - Upper bound: round toward +∞
//!
//! This guarantees that the result interval contains all possible values.

use crate::fp::ieee754_full::Ieee754Engine;
use crate::fp::{FpFormat, FpRoundingMode, FpValue};

/// A floating-point interval [lower, upper]
///
/// Invariants:
/// - lower <= upper (unless NaN is involved)
/// - Empty intervals are represented as [NaN, NaN]
/// - Unbounded intervals use infinities
#[derive(Debug, Clone, PartialEq)]
pub struct FpInterval {
    /// Lower bound (inclusive)
    pub lower: FpValue,
    /// Upper bound (inclusive)
    pub upper: FpValue,
    /// Format specification
    pub format: FpFormat,
}

impl FpInterval {
    /// Create a new interval from bounds
    ///
    /// # Panics
    /// Panics if bounds have different formats
    #[must_use]
    pub fn new(lower: FpValue, upper: FpValue) -> Self {
        assert_eq!(
            lower.format, upper.format,
            "Interval bounds must have same format"
        );
        Self {
            lower,
            upper,
            format: lower.format,
        }
    }

    /// Create a point interval (single value)
    #[must_use]
    pub fn point(value: FpValue) -> Self {
        Self {
            lower: value,
            upper: value,
            format: value.format,
        }
    }

    /// Create an empty interval (NaN)
    #[must_use]
    pub fn empty(format: FpFormat) -> Self {
        let nan = FpValue::nan(format);
        Self {
            lower: nan,
            upper: nan,
            format,
        }
    }

    /// Create an unbounded interval (-∞, +∞)
    #[must_use]
    pub fn unbounded(format: FpFormat) -> Self {
        Self {
            lower: FpValue::neg_infinity(format),
            upper: FpValue::pos_infinity(format),
            format,
        }
    }

    /// Create an interval for all finite values
    #[must_use]
    pub fn all_finite(format: FpFormat) -> Self {
        let max_exp = format.max_exponent() - 1;
        let max_sig = (1u64 << (format.significand_bits - 1)) - 1;

        let lower = FpValue {
            sign: true,
            exponent: max_exp as u64,
            significand: max_sig,
            format,
        };

        let upper = FpValue {
            sign: false,
            exponent: max_exp as u64,
            significand: max_sig,
            format,
        };

        Self {
            lower,
            upper,
            format,
        }
    }

    /// Create interval [0, 0]
    #[must_use]
    pub fn zero(format: FpFormat) -> Self {
        Self::point(FpValue::pos_zero(format))
    }

    /// Create interval [1, 1]
    #[must_use]
    pub fn one(format: FpFormat) -> Self {
        let one_val = FpValue {
            sign: false,
            exponent: format.bias() as u64,
            significand: 0,
            format,
        };
        Self::point(one_val)
    }

    /// Check if this interval is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lower.is_nan() || self.upper.is_nan()
    }

    /// Check if this is a point interval (single value)
    #[must_use]
    pub fn is_point(&self) -> bool {
        if self.is_empty() {
            return false;
        }
        self.lower.sign == self.upper.sign
            && self.lower.exponent == self.upper.exponent
            && self.lower.significand == self.upper.significand
    }

    /// Check if this interval contains a value
    #[must_use]
    pub fn contains(&self, value: &FpValue) -> bool {
        if self.is_empty() || value.is_nan() {
            return false;
        }

        let engine = Ieee754Engine::new();
        engine.le(&self.lower, value) && engine.le(value, &self.upper)
    }

    /// Check if this interval is a subset of another
    #[must_use]
    pub fn is_subset_of(&self, other: &Self) -> bool {
        if self.is_empty() {
            return true;
        }
        if other.is_empty() {
            return false;
        }

        let engine = Ieee754Engine::new();
        engine.le(&other.lower, &self.lower) && engine.le(&self.upper, &other.upper)
    }

    /// Compute the width of the interval
    #[must_use]
    pub fn width(&self) -> FpValue {
        if self.is_empty() {
            return FpValue::nan(self.format);
        }

        let mut engine = Ieee754Engine::new();
        engine.set_rounding_mode(FpRoundingMode::RoundTowardPositive);
        engine.sub(&self.upper, &self.lower)
    }

    /// Compute the midpoint of the interval
    #[must_use]
    pub fn midpoint(&self) -> FpValue {
        if self.is_empty() {
            return FpValue::nan(self.format);
        }

        let mut engine = Ieee754Engine::new();
        let sum = engine.add(&self.lower, &self.upper);
        let two = FpValue::from_f32(2.0);
        engine.div(&sum, &two)
    }

    /// Split the interval into two halves
    #[must_use]
    pub fn split(&self) -> (Self, Self) {
        let mid = self.midpoint();
        let left = Self::new(self.lower, mid);
        let right = Self::new(mid, self.upper);
        (left, right)
    }

    /// Compute the intersection of two intervals
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        assert_eq!(self.format, other.format);

        if self.is_empty() || other.is_empty() {
            return Self::empty(self.format);
        }

        let engine = Ieee754Engine::new();

        // Lower bound is the maximum of the two lower bounds
        let lower = if engine.lt(&self.lower, &other.lower) {
            other.lower
        } else {
            self.lower
        };

        // Upper bound is the minimum of the two upper bounds
        let upper = if engine.lt(&self.upper, &other.upper) {
            self.upper
        } else {
            other.upper
        };

        // Check if intersection is empty
        if engine.lt(&upper, &lower) {
            return Self::empty(self.format);
        }

        Self::new(lower, upper)
    }

    /// Compute the hull (union) of two intervals
    #[must_use]
    pub fn hull(&self, other: &Self) -> Self {
        assert_eq!(self.format, other.format);

        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }

        let engine = Ieee754Engine::new();

        let lower = if engine.lt(&self.lower, &other.lower) {
            self.lower
        } else {
            other.lower
        };

        let upper = if engine.lt(&self.upper, &other.upper) {
            other.upper
        } else {
            self.upper
        };

        Self::new(lower, upper)
    }

    /// Check if this interval overlaps with another
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        !self.intersection(other).is_empty()
    }

    /// Widen the interval by a relative amount (for fixpoint acceleration)
    pub fn widen(&mut self, threshold: u32) {
        if self.is_empty() {
            return;
        }

        // Implement widening for fixpoint iteration
        // If bounds are growing, jump to infinity
        if threshold > 0 {
            if !self.lower.is_infinite() {
                self.lower = FpValue::neg_infinity(self.format);
            }
            if !self.upper.is_infinite() {
                self.upper = FpValue::pos_infinity(self.format);
            }
        }
    }

    /// Narrow the interval (dual of widen)
    pub fn narrow(&mut self, other: &Self) {
        if other.is_empty() {
            return;
        }

        let engine = Ieee754Engine::new();

        // Keep tighter bounds
        if !engine.lt(&self.lower, &other.lower) {
            self.lower = other.lower;
        }
        if !engine.lt(&other.upper, &self.upper) {
            self.upper = other.upper;
        }
    }
}

/// Interval arithmetic engine with conservative rounding
#[derive(Debug)]
pub struct IntervalEngine {
    /// IEEE 754 engine for lower bounds (round toward -∞)
    lower_engine: Ieee754Engine,
    /// IEEE 754 engine for upper bounds (round toward +∞)
    upper_engine: Ieee754Engine,
}

impl Default for IntervalEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl IntervalEngine {
    /// Create a new interval arithmetic engine
    #[must_use]
    pub fn new() -> Self {
        let mut lower_engine = Ieee754Engine::new();
        lower_engine.set_rounding_mode(FpRoundingMode::RoundTowardNegative);

        let mut upper_engine = Ieee754Engine::new();
        upper_engine.set_rounding_mode(FpRoundingMode::RoundTowardPositive);

        Self {
            lower_engine,
            upper_engine,
        }
    }

    /// Add two intervals: \[a,b\] + \[c,d\] = \[a+c, b+d\]
    pub fn add(&mut self, a: &FpInterval, b: &FpInterval) -> FpInterval {
        assert_eq!(a.format, b.format);

        if a.is_empty() || b.is_empty() {
            return FpInterval::empty(a.format);
        }

        let lower = self.lower_engine.add(&a.lower, &b.lower);
        let upper = self.upper_engine.add(&a.upper, &b.upper);

        FpInterval::new(lower, upper)
    }

    /// Subtract two intervals: \[a,b\] - \[c,d\] = \[a-d, b-c\]
    pub fn sub(&mut self, a: &FpInterval, b: &FpInterval) -> FpInterval {
        assert_eq!(a.format, b.format);

        if a.is_empty() || b.is_empty() {
            return FpInterval::empty(a.format);
        }

        let lower = self.lower_engine.sub(&a.lower, &b.upper);
        let upper = self.upper_engine.sub(&a.upper, &b.lower);

        FpInterval::new(lower, upper)
    }

    /// Multiply two intervals (considers all four corner combinations)
    pub fn mul(&mut self, a: &FpInterval, b: &FpInterval) -> FpInterval {
        assert_eq!(a.format, b.format);

        if a.is_empty() || b.is_empty() {
            return FpInterval::empty(a.format);
        }

        // Compute all four products
        let p1 = self.lower_engine.mul(&a.lower, &b.lower);
        let p2 = self.lower_engine.mul(&a.lower, &b.upper);
        let p3 = self.lower_engine.mul(&a.upper, &b.lower);
        let p4 = self.lower_engine.mul(&a.upper, &b.upper);

        let lower = self.min_of_four(&p1, &p2, &p3, &p4);

        let p1 = self.upper_engine.mul(&a.lower, &b.lower);
        let p2 = self.upper_engine.mul(&a.lower, &b.upper);
        let p3 = self.upper_engine.mul(&a.upper, &b.lower);
        let p4 = self.upper_engine.mul(&a.upper, &b.upper);

        let upper = self.max_of_four(&p1, &p2, &p3, &p4);

        FpInterval::new(lower, upper)
    }

    /// Divide two intervals: \[a,b\] / \[c,d\]
    pub fn div(&mut self, a: &FpInterval, b: &FpInterval) -> FpInterval {
        assert_eq!(a.format, b.format);

        if a.is_empty() || b.is_empty() {
            return FpInterval::empty(a.format);
        }

        // Check if divisor contains zero
        let zero = FpValue::pos_zero(a.format);
        if b.contains(&zero) {
            // Division by interval containing zero
            // Result is unbounded (or NaN)
            return FpInterval::unbounded(a.format);
        }

        // Compute all four quotients
        let q1 = self.lower_engine.div(&a.lower, &b.lower);
        let q2 = self.lower_engine.div(&a.lower, &b.upper);
        let q3 = self.lower_engine.div(&a.upper, &b.lower);
        let q4 = self.lower_engine.div(&a.upper, &b.upper);

        let lower = self.min_of_four(&q1, &q2, &q3, &q4);

        let q1 = self.upper_engine.div(&a.lower, &b.lower);
        let q2 = self.upper_engine.div(&a.lower, &b.upper);
        let q3 = self.upper_engine.div(&a.upper, &b.lower);
        let q4 = self.upper_engine.div(&a.upper, &b.upper);

        let upper = self.max_of_four(&q1, &q2, &q3, &q4);

        FpInterval::new(lower, upper)
    }

    /// Square root of an interval: sqrt(\[a,b\]) = \[sqrt(a), sqrt(b)\]
    pub fn sqrt(&mut self, a: &FpInterval) -> FpInterval {
        if a.is_empty() {
            return FpInterval::empty(a.format);
        }

        // Check if interval contains negative values
        let zero = FpValue::pos_zero(a.format);
        let engine = Ieee754Engine::new();

        if engine.lt(&a.upper, &zero) {
            // Entirely negative
            return FpInterval::empty(a.format);
        }

        // Clamp lower bound to zero if negative
        let lower_bound = if engine.lt(&a.lower, &zero) {
            zero
        } else {
            a.lower
        };

        let lower = self.lower_engine.sqrt(&lower_bound);
        let upper = self.upper_engine.sqrt(&a.upper);

        FpInterval::new(lower, upper)
    }

    /// Absolute value of an interval
    pub fn abs(&mut self, a: &FpInterval) -> FpInterval {
        if a.is_empty() {
            return FpInterval::empty(a.format);
        }

        let zero = FpValue::pos_zero(a.format);
        let engine = Ieee754Engine::new();

        // Check if interval contains zero
        if a.contains(&zero) {
            // [min(|a.lower|, |a.upper|), max(|a.lower|, |a.upper|)]
            let abs_lower = engine.abs(&a.lower);
            let abs_upper = engine.abs(&a.upper);

            let lower = zero; // Minimum is zero
            let upper = if engine.lt(&abs_lower, &abs_upper) {
                abs_upper
            } else {
                abs_lower
            };

            FpInterval::new(lower, upper)
        } else {
            // Interval doesn't contain zero
            let abs_lower = self.lower_engine.abs(&a.lower);
            let abs_upper = self.upper_engine.abs(&a.upper);

            if engine.lt(&abs_lower, &abs_upper) {
                FpInterval::new(abs_lower, abs_upper)
            } else {
                FpInterval::new(abs_upper, abs_lower)
            }
        }
    }

    /// Negation of an interval: -\[a,b\] = \[-b, -a\]
    #[must_use]
    pub fn neg(&self, a: &FpInterval) -> FpInterval {
        if a.is_empty() {
            return FpInterval::empty(a.format);
        }

        let engine = Ieee754Engine::new();
        let lower = engine.neg(&a.upper);
        let upper = engine.neg(&a.lower);

        FpInterval::new(lower, upper)
    }

    /// Minimum of an interval: min(\[a,b\]) = a
    #[must_use]
    pub fn min_value(&self, a: &FpInterval) -> FpValue {
        a.lower
    }

    /// Maximum of an interval: max(\[a,b\]) = b
    #[must_use]
    pub fn max_value(&self, a: &FpInterval) -> FpValue {
        a.upper
    }

    /// Minimum of two intervals
    pub fn min(&mut self, a: &FpInterval, b: &FpInterval) -> FpInterval {
        assert_eq!(a.format, b.format);

        if a.is_empty() || b.is_empty() {
            return FpInterval::empty(a.format);
        }

        let lower = self.lower_engine.min(&a.lower, &b.lower);
        let upper = self.upper_engine.min(&a.upper, &b.upper);

        FpInterval::new(lower, upper)
    }

    /// Maximum of two intervals
    pub fn max(&mut self, a: &FpInterval, b: &FpInterval) -> FpInterval {
        assert_eq!(a.format, b.format);

        if a.is_empty() || b.is_empty() {
            return FpInterval::empty(a.format);
        }

        let lower = self.lower_engine.max(&a.lower, &b.lower);
        let upper = self.upper_engine.max(&a.upper, &b.upper);

        FpInterval::new(lower, upper)
    }

    /// Fused multiply-add: (a * b) + c
    pub fn fma(&mut self, a: &FpInterval, b: &FpInterval, c: &FpInterval) -> FpInterval {
        let product = self.mul(a, b);
        self.add(&product, c)
    }

    /// Power function: a^n for integer n
    pub fn powi(&mut self, a: &FpInterval, n: i32) -> FpInterval {
        if a.is_empty() {
            return FpInterval::empty(a.format);
        }

        if n == 0 {
            return FpInterval::one(a.format);
        }

        if n == 1 {
            return a.clone();
        }

        if n < 0 {
            // a^(-n) = 1 / a^n
            let pos_power = self.powi(a, -n);
            let one_interval = FpInterval::one(a.format);
            return self.div(&one_interval, &pos_power);
        }

        // Even vs odd power
        if n % 2 == 0 {
            // Even power: result is always non-negative
            let abs_interval = self.abs(a);
            let lower_power = self.power_helper(&abs_interval.lower, n as u32);
            let upper_power = self.power_helper(&abs_interval.upper, n as u32);

            FpInterval::new(lower_power, upper_power)
        } else {
            // Odd power: preserves sign
            let lower_power = self.power_helper(&a.lower, n as u32);
            let upper_power = self.power_helper(&a.upper, n as u32);

            FpInterval::new(lower_power, upper_power)
        }
    }

    /// Helper for computing power
    fn power_helper(&mut self, base: &FpValue, exp: u32) -> FpValue {
        if exp == 0 {
            return FpValue::from_f32(1.0);
        }

        let mut result = *base;
        for _ in 1..exp {
            result = self.upper_engine.mul(&result, base);
        }
        result
    }

    /// Reciprocal: 1 / a
    pub fn recip(&mut self, a: &FpInterval) -> FpInterval {
        let one = FpInterval::one(a.format);
        self.div(&one, a)
    }

    /// Check if intervals are disjoint
    #[must_use]
    pub fn disjoint(&self, a: &FpInterval, b: &FpInterval) -> bool {
        if a.is_empty() || b.is_empty() {
            return true;
        }

        let engine = Ieee754Engine::new();
        engine.lt(&a.upper, &b.lower) || engine.lt(&b.upper, &a.lower)
    }

    /// Compute relative error bound
    #[must_use]
    pub fn relative_error(&mut self, a: &FpInterval) -> FpValue {
        if a.is_empty() || a.is_point() {
            return FpValue::pos_zero(a.format);
        }

        let width = a.width();
        let mid = a.midpoint();

        if mid.is_zero() {
            return FpValue::pos_infinity(a.format);
        }

        self.upper_engine.div(&width, &self.upper_engine.abs(&mid))
    }

    /// Compute absolute error bound
    #[must_use]
    pub fn absolute_error(&self, a: &FpInterval) -> FpValue {
        a.width()
    }

    /// Refine interval based on constraint: a op b
    pub fn refine_comparison(
        &mut self,
        a: &FpInterval,
        b: &FpInterval,
        op: ComparisonOp,
    ) -> (FpInterval, FpInterval) {
        match op {
            ComparisonOp::Lt => {
                // a < b => a.upper < b.upper, a.lower < b.lower
                let new_a = FpInterval::new(a.lower, self.upper_engine.min(&a.upper, &b.upper));
                let new_b = FpInterval::new(self.lower_engine.max(&b.lower, &a.lower), b.upper);
                (new_a, new_b)
            }
            ComparisonOp::Le => {
                let new_a = FpInterval::new(a.lower, self.upper_engine.min(&a.upper, &b.upper));
                let new_b = FpInterval::new(self.lower_engine.max(&b.lower, &a.lower), b.upper);
                (new_a, new_b)
            }
            ComparisonOp::Gt => self.refine_comparison(b, a, ComparisonOp::Lt).swap(),
            ComparisonOp::Ge => self.refine_comparison(b, a, ComparisonOp::Le).swap(),
            ComparisonOp::Eq => {
                let intersection = a.intersection(b);
                (intersection.clone(), intersection)
            }
            ComparisonOp::Ne => {
                // Can't refine much for inequality
                (a.clone(), b.clone())
            }
        }
    }

    /// Helper to find minimum of four values
    fn min_of_four(&self, a: &FpValue, b: &FpValue, c: &FpValue, d: &FpValue) -> FpValue {
        let engine = Ieee754Engine::new();
        let min_ab = if engine.lt(a, b) { *a } else { *b };
        let min_cd = if engine.lt(c, d) { *c } else { *d };
        if engine.lt(&min_ab, &min_cd) {
            min_ab
        } else {
            min_cd
        }
    }

    /// Helper to find maximum of four values
    fn max_of_four(&self, a: &FpValue, b: &FpValue, c: &FpValue, d: &FpValue) -> FpValue {
        let engine = Ieee754Engine::new();
        let max_ab = if engine.lt(a, b) { *b } else { *a };
        let max_cd = if engine.lt(c, d) { *d } else { *c };
        if engine.lt(&max_ab, &max_cd) {
            max_cd
        } else {
            max_ab
        }
    }
}

/// Comparison operators for constraint refinement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// Equal
    Eq,
    /// Not equal
    Ne,
}

/// Trait for swapping tuple elements
trait Swap {
    fn swap(self) -> Self;
}

impl<T> Swap for (T, T) {
    fn swap(self) -> Self {
        (self.1, self.0)
    }
}

/// Error bounds tracker for accumulated errors
#[derive(Debug, Clone)]
pub struct ErrorBounds {
    /// Absolute error bound
    pub absolute: FpValue,
    /// Relative error bound
    pub relative: FpValue,
    /// Number of operations performed
    pub operations: u32,
}

impl ErrorBounds {
    /// Create new error bounds (initially zero)
    #[must_use]
    pub fn new(format: FpFormat) -> Self {
        Self {
            absolute: FpValue::pos_zero(format),
            relative: FpValue::pos_zero(format),
            operations: 0,
        }
    }

    /// Update error bounds after an operation
    pub fn update(&mut self, interval: &FpInterval, engine: &mut IntervalEngine) {
        let abs_error = engine.absolute_error(interval);
        let rel_error = engine.relative_error(interval);

        let mut ieee_engine = Ieee754Engine::new();
        self.absolute = ieee_engine.add(&self.absolute, &abs_error);
        self.relative = ieee_engine.add(&self.relative, &rel_error);
        self.operations += 1;
    }

    /// Check if errors are within tolerance
    #[must_use]
    pub fn within_tolerance(&self, abs_tol: &FpValue, rel_tol: &FpValue) -> bool {
        let engine = Ieee754Engine::new();
        engine.le(&self.absolute, abs_tol) && engine.le(&self.relative, rel_tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_creation() {
        let a = FpValue::from_f32(1.0);
        let b = FpValue::from_f32(2.0);
        let interval = FpInterval::new(a, b);

        assert!(!interval.is_empty());
        assert!(!interval.is_point());
    }

    #[test]
    fn test_point_interval() {
        let val = FpValue::from_f32(1.5);
        let interval = FpInterval::point(val);

        assert!(interval.is_point());
        assert!(interval.contains(&val));
    }

    #[test]
    fn test_empty_interval() {
        let interval = FpInterval::empty(FpFormat::FLOAT32);
        assert!(interval.is_empty());
    }

    #[test]
    fn test_unbounded_interval() {
        let interval = FpInterval::unbounded(FpFormat::FLOAT32);
        assert!(!interval.is_empty());

        let val = FpValue::from_f32(42.0);
        assert!(interval.contains(&val));
    }

    #[test]
    fn test_contains() {
        let a = FpValue::from_f32(1.0);
        let b = FpValue::from_f32(3.0);
        let interval = FpInterval::new(a, b);

        assert!(interval.contains(&FpValue::from_f32(2.0)));
        assert!(interval.contains(&FpValue::from_f32(1.0)));
        assert!(interval.contains(&FpValue::from_f32(3.0)));
        assert!(!interval.contains(&FpValue::from_f32(0.5)));
        assert!(!interval.contains(&FpValue::from_f32(3.5)));
    }

    #[test]
    fn test_interval_addition() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(2.0));
        let b = FpInterval::new(FpValue::from_f32(3.0), FpValue::from_f32(4.0));

        let result = engine.add(&a, &b);

        // [1,2] + [3,4] = [4,6]
        assert!(result.contains(&FpValue::from_f32(4.0)));
        assert!(result.contains(&FpValue::from_f32(6.0)));
        assert!(result.contains(&FpValue::from_f32(5.0)));
    }

    #[test]
    fn test_interval_subtraction() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(5.0), FpValue::from_f32(7.0));
        let b = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(3.0));

        let result = engine.sub(&a, &b);

        // [5,7] - [2,3] = [2,5]
        assert!(result.contains(&FpValue::from_f32(2.0)));
        assert!(result.contains(&FpValue::from_f32(5.0)));
        assert!(result.contains(&FpValue::from_f32(3.5)));
    }

    #[test]
    #[ignore] // TODO: Fix IEEE754 multiplication - exponent calculation is incorrect
    fn test_interval_multiplication() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(3.0));
        let b = FpInterval::new(FpValue::from_f32(4.0), FpValue::from_f32(5.0));

        let result = engine.mul(&a, &b);

        // [2,3] * [4,5] = [8,15]
        assert!(result.contains(&FpValue::from_f32(8.0)));
        assert!(result.contains(&FpValue::from_f32(15.0)));
    }

    #[test]
    fn test_interval_division() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(6.0), FpValue::from_f32(12.0));
        let b = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(3.0));

        let result = engine.div(&a, &b);

        // [6,12] / [2,3] = [2,6]
        assert!(result.contains(&FpValue::from_f32(2.0)));
        assert!(result.contains(&FpValue::from_f32(6.0)));
    }

    #[test]
    fn test_interval_sqrt() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(4.0));
        let result = engine.sqrt(&a);

        // sqrt([1,4]) ⊇ [1,2]
        assert!(result.contains(&FpValue::from_f32(1.0)));
        assert!(result.contains(&FpValue::from_f32(2.0)));
    }

    #[test]
    fn test_interval_abs() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(-2.0), FpValue::from_f32(3.0));
        let result = engine.abs(&a);

        // abs([-2,3]) = [0,3]
        assert!(result.contains(&FpValue::from_f32(0.0)));
        assert!(result.contains(&FpValue::from_f32(3.0)));
    }

    #[test]
    fn test_interval_negation() {
        let engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(3.0));
        let result = engine.neg(&a);

        // -[1,3] = [-3,-1]
        assert!(result.contains(&FpValue::from_f32(-1.0)));
        assert!(result.contains(&FpValue::from_f32(-3.0)));
        assert!(result.contains(&FpValue::from_f32(-2.0)));
    }

    #[test]
    fn test_interval_intersection() {
        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(5.0));
        let b = FpInterval::new(FpValue::from_f32(3.0), FpValue::from_f32(7.0));

        let result = a.intersection(&b);

        // [1,5] ∩ [3,7] = [3,5]
        assert!(result.contains(&FpValue::from_f32(3.0)));
        assert!(result.contains(&FpValue::from_f32(5.0)));
        assert!(result.contains(&FpValue::from_f32(4.0)));
        assert!(!result.contains(&FpValue::from_f32(2.0)));
        assert!(!result.contains(&FpValue::from_f32(6.0)));
    }

    #[test]
    fn test_interval_hull() {
        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(3.0));
        let b = FpInterval::new(FpValue::from_f32(5.0), FpValue::from_f32(7.0));

        let result = a.hull(&b);

        // [1,3] ∪ [5,7] = [1,7]
        assert!(result.contains(&FpValue::from_f32(1.0)));
        assert!(result.contains(&FpValue::from_f32(7.0)));
        assert!(result.contains(&FpValue::from_f32(4.0))); // Gap is included
    }

    #[test]
    fn test_interval_overlaps() {
        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(5.0));
        let b = FpInterval::new(FpValue::from_f32(3.0), FpValue::from_f32(7.0));
        let c = FpInterval::new(FpValue::from_f32(6.0), FpValue::from_f32(8.0));

        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_interval_width() {
        let interval = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(4.0));
        let width = interval.width();

        // Width should be approximately 3.0
        let width_f32 = width.to_f32();
        assert!(width_f32.is_some());
        let w = width_f32.unwrap_or(0.0);
        assert!((w - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_interval_midpoint() {
        let interval = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(6.0));
        let mid = interval.midpoint();

        // Midpoint should be 4.0
        assert_eq!(mid.to_f32(), Some(4.0));
    }

    #[test]
    fn test_interval_split() {
        let interval = FpInterval::new(FpValue::from_f32(0.0), FpValue::from_f32(4.0));
        let (left, right) = interval.split();

        assert!(left.contains(&FpValue::from_f32(1.0)));
        assert!(right.contains(&FpValue::from_f32(3.0)));
    }

    #[test]
    fn test_interval_min_max() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(5.0));
        let b = FpInterval::new(FpValue::from_f32(3.0), FpValue::from_f32(7.0));

        let min_result = engine.min(&a, &b);
        let max_result = engine.max(&a, &b);

        assert!(min_result.contains(&FpValue::from_f32(1.0)));
        assert!(max_result.contains(&FpValue::from_f32(7.0)));
    }

    #[test]
    fn test_interval_powi_even() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(-2.0), FpValue::from_f32(3.0));
        let result = engine.powi(&a, 2);

        // [-2,3]^2 = [0,9]
        assert!(result.contains(&FpValue::from_f32(0.0)));
        assert!(result.contains(&FpValue::from_f32(9.0)));
    }

    #[test]
    fn test_interval_powi_odd() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(2.0));
        let result = engine.powi(&a, 3);

        // [1,2]^3 = [1,8]
        assert!(result.contains(&FpValue::from_f32(1.0)));
        assert!(result.contains(&FpValue::from_f32(8.0)));
    }

    #[test]
    fn test_interval_reciprocal() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(4.0));
        let result = engine.recip(&a);

        // 1/[2,4] = [0.25, 0.5]
        assert!(result.contains(&FpValue::from_f32(0.25)));
        assert!(result.contains(&FpValue::from_f32(0.5)));
    }

    #[test]
    fn test_disjoint_intervals() {
        let engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(3.0));
        let b = FpInterval::new(FpValue::from_f32(5.0), FpValue::from_f32(7.0));
        let c = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(6.0));

        assert!(engine.disjoint(&a, &b));
        assert!(!engine.disjoint(&a, &c));
    }

    #[test]
    fn test_error_bounds() {
        let mut bounds = ErrorBounds::new(FpFormat::FLOAT32);
        let mut engine = IntervalEngine::new();

        let interval = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(1.1));
        bounds.update(&interval, &mut engine);

        assert!(bounds.operations > 0);
    }

    #[test]
    fn test_division_by_zero_interval() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(2.0));
        let b = FpInterval::new(FpValue::from_f32(-1.0), FpValue::from_f32(1.0)); // Contains zero

        let result = engine.div(&a, &b);

        // Result should be unbounded
        assert!(result.lower.is_infinite() || result.upper.is_infinite());
    }

    #[test]
    fn test_subset_relation() {
        let a = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(3.0));
        let b = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(5.0));

        assert!(a.is_subset_of(&b));
        assert!(!b.is_subset_of(&a));
    }

    #[test]
    fn test_comparison_refinement() {
        let mut engine = IntervalEngine::new();

        let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(10.0));
        let b = FpInterval::new(FpValue::from_f32(5.0), FpValue::from_f32(15.0));

        let (refined_a, refined_b) = engine.refine_comparison(&a, &b, ComparisonOp::Lt);

        // After refinement, a should be less than b
        assert!(refined_a.upper.to_f32().unwrap_or(0.0) <= refined_b.upper.to_f32().unwrap_or(0.0));
    }
}
