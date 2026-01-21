//! Arbitrary Precision Floating-Point Arithmetic (MPFR-like)
//!
//! This module provides an arbitrary precision floating-point type with configurable
//! precision and rounding modes. It is designed for applications requiring extreme
//! precision beyond what IEEE 754 double precision provides.
//!
//! # Features
//!
//! - Configurable precision (number of bits in mantissa)
//! - Multiple rounding modes (RoundNearest, RoundTowardZero, RoundUp, RoundDown)
//! - Basic arithmetic operations (add, sub, mul, div, sqrt)
//! - Comparison operations
//! - Conversion to/from f64
//!
//! # Example
//!
//! ```
//! use oxiz_math::mpfr::{ArbitraryFloat, RoundingMode, Precision};
//!
//! // Create with 128-bit precision
//! let precision = Precision::new(128);
//! let a = ArbitraryFloat::from_f64(3.14159265358979323846, precision);
//! let b = ArbitraryFloat::from_f64(2.71828182845904523536, precision);
//!
//! // Perform addition with rounding toward nearest
//! let sum = a.add(&b, RoundingMode::RoundNearest);
//! ```
//!
//! # Implementation Notes
//!
//! This implementation uses `num-bigint` for arbitrary precision integer arithmetic.
//! The floating-point representation uses:
//! - Sign bit (boolean)
//! - Mantissa (BigInt with specified precision bits)
//! - Exponent (i64 for the binary exponent)
//!
//! This is a simplified MPFR-like implementation suitable for SMT solving scenarios
//! that require extended precision arithmetic.

use num_bigint::BigUint;
use num_integer::Integer;
use num_traits::{One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Precision specification for arbitrary precision floats.
///
/// Specifies the number of bits in the mantissa (significand).
/// Higher precision means more accurate results but slower computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Precision {
    /// Number of bits in the mantissa.
    bits: u32,
}

impl Precision {
    /// Create a new precision specification.
    ///
    /// # Arguments
    /// * `bits` - Number of bits for the mantissa (minimum 1)
    ///
    /// # Example
    /// ```
    /// use oxiz_math::mpfr::Precision;
    /// let p = Precision::new(128); // 128-bit precision
    /// ```
    pub fn new(bits: u32) -> Self {
        assert!(bits >= 1, "Precision must be at least 1 bit");
        Self { bits }
    }

    /// Get the number of bits.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Double precision (53 bits, same as f64).
    pub const DOUBLE: Precision = Precision { bits: 53 };

    /// Extended precision (64 bits, x86 extended precision).
    pub const EXTENDED: Precision = Precision { bits: 64 };

    /// Quadruple precision (113 bits, IEEE 754 quad).
    pub const QUAD: Precision = Precision { bits: 113 };

    /// High precision for SMT solving (256 bits).
    pub const HIGH: Precision = Precision { bits: 256 };
}

impl Default for Precision {
    fn default() -> Self {
        Self::DOUBLE
    }
}

/// Rounding modes for arbitrary precision operations.
///
/// These correspond to IEEE 754 rounding modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default).
    #[default]
    RoundNearest,
    /// Round toward zero (truncation).
    RoundTowardZero,
    /// Round toward positive infinity.
    RoundUp,
    /// Round toward negative infinity.
    RoundDown,
}

/// Special values for arbitrary precision floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecialValue {
    /// Not a special value (regular number).
    None,
    /// Positive infinity.
    PosInfinity,
    /// Negative infinity.
    NegInfinity,
    /// Not a Number.
    NaN,
}

/// Arbitrary precision floating-point number.
///
/// Represents a floating-point number with configurable precision.
/// The internal representation uses a sign, mantissa (BigInt), and exponent.
///
/// Value = (-1)^sign * mantissa * 2^(exponent - precision + 1)
#[derive(Clone)]
pub struct ArbitraryFloat {
    /// Sign: true for negative, false for positive.
    sign: bool,
    /// Mantissa (significand) as a big integer.
    mantissa: BigUint,
    /// Binary exponent.
    exponent: i64,
    /// Precision (number of bits in mantissa).
    precision: Precision,
    /// Special value indicator.
    special: SpecialValue,
}

impl ArbitraryFloat {
    /// Create a new arbitrary precision float from components.
    ///
    /// # Arguments
    /// * `sign` - True for negative, false for positive
    /// * `mantissa` - The significand
    /// * `exponent` - The binary exponent
    /// * `precision` - The precision specification
    fn new(sign: bool, mantissa: BigUint, exponent: i64, precision: Precision) -> Self {
        Self {
            sign,
            mantissa,
            exponent,
            precision,
            special: SpecialValue::None,
        }
    }

    /// Create a zero value with the given precision.
    pub fn zero(precision: Precision) -> Self {
        Self::new(false, BigUint::zero(), 0, precision)
    }

    /// Create a one value with the given precision.
    pub fn one(precision: Precision) -> Self {
        let mantissa = BigUint::one() << (precision.bits() - 1);
        Self::new(false, mantissa, 0, precision)
    }

    /// Create positive infinity.
    pub fn pos_infinity(precision: Precision) -> Self {
        let mut f = Self::zero(precision);
        f.special = SpecialValue::PosInfinity;
        f
    }

    /// Create negative infinity.
    pub fn neg_infinity(precision: Precision) -> Self {
        let mut f = Self::zero(precision);
        f.special = SpecialValue::NegInfinity;
        f
    }

    /// Create NaN (Not a Number).
    pub fn nan(precision: Precision) -> Self {
        let mut f = Self::zero(precision);
        f.special = SpecialValue::NaN;
        f
    }

    /// Check if this value is zero.
    pub fn is_zero(&self) -> bool {
        self.special == SpecialValue::None && self.mantissa.is_zero()
    }

    /// Check if this value is positive infinity.
    pub fn is_pos_infinity(&self) -> bool {
        self.special == SpecialValue::PosInfinity
    }

    /// Check if this value is negative infinity.
    pub fn is_neg_infinity(&self) -> bool {
        self.special == SpecialValue::NegInfinity
    }

    /// Check if this value is any infinity.
    pub fn is_infinity(&self) -> bool {
        self.is_pos_infinity() || self.is_neg_infinity()
    }

    /// Check if this value is NaN.
    pub fn is_nan(&self) -> bool {
        self.special == SpecialValue::NaN
    }

    /// Check if this value is finite (not infinity or NaN).
    pub fn is_finite(&self) -> bool {
        self.special == SpecialValue::None
    }

    /// Check if this value is negative.
    pub fn is_negative(&self) -> bool {
        self.sign && !self.is_zero() && !self.is_nan()
    }

    /// Check if this value is positive.
    pub fn is_positive(&self) -> bool {
        !self.sign && !self.is_zero() && !self.is_nan()
    }

    /// Get the precision of this value.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Create from an f64 value.
    ///
    /// # Arguments
    /// * `value` - The f64 value to convert
    /// * `precision` - Target precision
    ///
    /// # Example
    /// ```
    /// use oxiz_math::mpfr::{ArbitraryFloat, Precision};
    /// let f = ArbitraryFloat::from_f64(3.14159, Precision::new(128));
    /// ```
    pub fn from_f64(value: f64, precision: Precision) -> Self {
        // Handle special values
        if value.is_nan() {
            return Self::nan(precision);
        }
        if value.is_infinite() {
            return if value > 0.0 {
                Self::pos_infinity(precision)
            } else {
                Self::neg_infinity(precision)
            };
        }
        if value == 0.0 {
            // Preserve sign of zero
            let mut z = Self::zero(precision);
            z.sign = value.is_sign_negative();
            return z;
        }

        // Extract IEEE 754 components
        let bits = value.to_bits();
        let sign = (bits >> 63) != 0;
        let exp_bits = ((bits >> 52) & 0x7FF) as i64;
        let mantissa_bits = bits & 0x000F_FFFF_FFFF_FFFF;

        // Compute actual exponent (removing bias of 1023)
        let exponent = if exp_bits == 0 {
            // Subnormal
            1 - 1023 - 52
        } else {
            // Normal
            exp_bits - 1023 - 52
        };

        // Build mantissa (implicit leading 1 for normalized numbers)
        let mantissa = if exp_bits == 0 {
            // Subnormal: no implicit leading 1
            BigUint::from(mantissa_bits)
        } else {
            // Normal: add implicit leading 1
            BigUint::from(mantissa_bits | (1u64 << 52))
        };

        // Scale mantissa to target precision
        let mut result = Self::new(sign, mantissa, exponent, precision);
        result.normalize(RoundingMode::RoundNearest);
        result
    }

    /// Convert to f64.
    ///
    /// May lose precision if the ArbitraryFloat has higher precision than f64.
    ///
    /// # Arguments
    /// * `rounding` - Rounding mode for the conversion
    ///
    /// # Example
    /// ```
    /// use oxiz_math::mpfr::{ArbitraryFloat, Precision, RoundingMode};
    /// let f = ArbitraryFloat::from_f64(3.14159, Precision::new(128));
    /// let back = f.to_f64(RoundingMode::RoundNearest);
    /// ```
    pub fn to_f64(&self, _rounding: RoundingMode) -> f64 {
        // Handle special values
        if self.is_nan() {
            return f64::NAN;
        }
        if self.is_pos_infinity() {
            return f64::INFINITY;
        }
        if self.is_neg_infinity() {
            return f64::NEG_INFINITY;
        }
        if self.is_zero() {
            return if self.sign { -0.0 } else { 0.0 };
        }

        // For f64 conversion, we need mantissa in [2^52, 2^53) and adjust exponent
        // Current: value = mantissa * 2^exponent (mantissa has self.precision bits)
        // Target: value = f64_mantissa * 2^f64_exp (f64_mantissa has 53 bits)

        let precision_bits = self.precision.bits();
        if precision_bits >= 53 {
            // Shift mantissa to 53 bits
            let shift = (precision_bits - 53) as usize;
            let f64_mantissa = (&self.mantissa >> shift).to_f64().unwrap_or(0.0);
            let f64_exponent = self.exponent + shift as i64;

            let result = f64_mantissa * 2.0f64.powi(f64_exponent as i32);
            if self.sign { -result } else { result }
        } else {
            // Mantissa is smaller than 53 bits
            let mantissa_f64 = self.mantissa.to_f64().unwrap_or(0.0);
            let result = mantissa_f64 * 2.0f64.powi(self.exponent as i32);
            if self.sign { -result } else { result }
        }
    }

    /// Normalize the mantissa to have the leading 1 in the correct position.
    fn normalize(&mut self, rounding: RoundingMode) {
        if self.mantissa.is_zero() || !self.is_finite() {
            return;
        }

        let target_bits = self.precision.bits() as u64;
        let current_bits = self.mantissa.bits();

        if current_bits > target_bits {
            // Need to round down
            let shift = current_bits - target_bits;
            let (quotient, remainder) = self.mantissa.div_rem(&(BigUint::one() << shift as usize));

            self.mantissa = quotient;
            self.exponent += shift as i64;

            // Apply rounding
            let half = BigUint::one() << (shift as usize - 1);
            let round_up = match rounding {
                RoundingMode::RoundNearest => {
                    // Round to nearest, ties to even
                    if remainder > half {
                        true
                    } else if remainder == half {
                        // Tie: round to even
                        self.mantissa.bit(0)
                    } else {
                        false
                    }
                }
                RoundingMode::RoundUp => !self.sign && !remainder.is_zero(),
                RoundingMode::RoundDown => self.sign && !remainder.is_zero(),
                RoundingMode::RoundTowardZero => false,
            };

            if round_up {
                self.mantissa += 1u32;
                // Check for overflow after rounding
                if self.mantissa.bits() > target_bits {
                    self.mantissa >>= 1;
                    self.exponent += 1;
                }
            }
        } else if current_bits < target_bits {
            // Shift left to fill precision
            let shift = target_bits - current_bits;
            self.mantissa <<= shift as usize;
            self.exponent -= shift as i64;
        }
    }

    /// Add two arbitrary precision floats.
    ///
    /// # Arguments
    /// * `other` - The other value to add
    /// * `rounding` - Rounding mode for the result
    ///
    /// # Returns
    /// The sum with the higher precision of the two operands.
    pub fn add(&self, other: &Self, rounding: RoundingMode) -> Self {
        // Handle special values
        if self.is_nan() || other.is_nan() {
            return Self::nan(self.precision.max(other.precision));
        }

        // inf + inf = inf (same sign) or NaN (opposite sign)
        if self.is_infinity() || other.is_infinity() {
            if self.is_pos_infinity() {
                if other.is_neg_infinity() {
                    return Self::nan(self.precision);
                }
                return Self::pos_infinity(self.precision);
            }
            if self.is_neg_infinity() {
                if other.is_pos_infinity() {
                    return Self::nan(self.precision);
                }
                return Self::neg_infinity(self.precision);
            }
            if other.is_pos_infinity() {
                return Self::pos_infinity(other.precision);
            }
            return Self::neg_infinity(other.precision);
        }

        // Handle zeros
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        // Use higher precision
        let result_precision = if self.precision.bits() >= other.precision.bits() {
            self.precision
        } else {
            other.precision
        };

        // Align exponents
        let (m1, m2, exp, s1, s2) = self.align_with(other);

        // Perform addition/subtraction based on signs
        let (result_mantissa, result_sign) = if s1 == s2 {
            // Same sign: add mantissas
            (m1 + m2, s1)
        } else {
            // Different signs: subtract mantissas
            match m1.cmp(&m2) {
                Ordering::Greater => (m1 - m2, s1),
                Ordering::Less => (m2 - m1, s2),
                Ordering::Equal => return Self::zero(result_precision),
            }
        };

        let mut result = Self::new(result_sign, result_mantissa, exp, result_precision);
        result.normalize(rounding);
        result
    }

    /// Subtract two arbitrary precision floats.
    pub fn sub(&self, other: &Self, rounding: RoundingMode) -> Self {
        let negated = other.neg();
        self.add(&negated, rounding)
    }

    /// Multiply two arbitrary precision floats.
    pub fn mul(&self, other: &Self, rounding: RoundingMode) -> Self {
        let result_precision = if self.precision.bits() >= other.precision.bits() {
            self.precision
        } else {
            other.precision
        };

        // Handle special values
        if self.is_nan() || other.is_nan() {
            return Self::nan(result_precision);
        }

        // 0 * inf = NaN
        if (self.is_zero() && other.is_infinity()) || (self.is_infinity() && other.is_zero()) {
            return Self::nan(result_precision);
        }

        // Handle infinities
        let result_sign = self.sign != other.sign;
        if self.is_infinity() || other.is_infinity() {
            return if result_sign {
                Self::neg_infinity(result_precision)
            } else {
                Self::pos_infinity(result_precision)
            };
        }

        // Handle zeros
        if self.is_zero() || other.is_zero() {
            let mut z = Self::zero(result_precision);
            z.sign = result_sign;
            return z;
        }

        // Multiply mantissas and add exponents
        // value = (m1 * m2) * 2^(e1 + e2)
        let result_mantissa = &self.mantissa * &other.mantissa;
        let result_exponent = self.exponent + other.exponent;

        let mut result = Self::new(
            result_sign,
            result_mantissa,
            result_exponent,
            result_precision,
        );
        result.normalize(rounding);
        result
    }

    /// Divide two arbitrary precision floats.
    pub fn div(&self, other: &Self, rounding: RoundingMode) -> Self {
        let result_precision = if self.precision.bits() >= other.precision.bits() {
            self.precision
        } else {
            other.precision
        };

        // Handle special values
        if self.is_nan() || other.is_nan() {
            return Self::nan(result_precision);
        }

        let result_sign = self.sign != other.sign;

        // 0/0 = NaN, inf/inf = NaN
        if (self.is_zero() && other.is_zero()) || (self.is_infinity() && other.is_infinity()) {
            return Self::nan(result_precision);
        }

        // x/0 = inf
        if other.is_zero() {
            return if result_sign {
                Self::neg_infinity(result_precision)
            } else {
                Self::pos_infinity(result_precision)
            };
        }

        // 0/x = 0
        if self.is_zero() {
            let mut z = Self::zero(result_precision);
            z.sign = result_sign;
            return z;
        }

        // x/inf = 0
        if other.is_infinity() {
            let mut z = Self::zero(result_precision);
            z.sign = result_sign;
            return z;
        }

        // inf/x = inf
        if self.is_infinity() {
            return if result_sign {
                Self::neg_infinity(result_precision)
            } else {
                Self::pos_infinity(result_precision)
            };
        }

        // Division: shift dividend left for precision, then divide
        let extra_bits = result_precision.bits() as usize + 10; // Extra bits for rounding
        let shifted_dividend = &self.mantissa << extra_bits;
        let result_mantissa = &shifted_dividend / &other.mantissa;
        // value = (dividend << extra_bits) / divisor * 2^exponent
        //       = (m1 / m2) * 2^extra_bits * 2^exponent
        // We want: (m1 / m2) * 2^(e1 - e2)
        // So: exponent = e1 - e2 - extra_bits
        let result_exponent = self.exponent - other.exponent - (extra_bits as i64);

        let mut result = Self::new(
            result_sign,
            result_mantissa,
            result_exponent,
            result_precision,
        );
        result.normalize(rounding);
        result
    }

    /// Compute the square root.
    ///
    /// Uses Newton-Raphson iteration for arbitrary precision square root.
    pub fn sqrt(&self, rounding: RoundingMode) -> Self {
        // Handle special values
        if self.is_nan() || self.is_neg_infinity() || (self.is_negative() && !self.is_zero()) {
            return Self::nan(self.precision);
        }
        if self.is_pos_infinity() {
            return Self::pos_infinity(self.precision);
        }
        if self.is_zero() {
            return Self::zero(self.precision);
        }

        // Use Newton-Raphson: x_{n+1} = (x_n + S/x_n) / 2
        // Start with a reasonable initial guess using f64
        let initial_guess = self.to_f64(RoundingMode::RoundNearest).sqrt();
        let mut x = Self::from_f64(initial_guess, self.precision);

        // Iterate until convergence
        let two = Self::from_f64(2.0, self.precision);
        let tolerance_bits = self.precision.bits() + 10;

        for _ in 0..100 {
            // x_new = (x + self/x) / 2
            let s_div_x = Self::div(self, &x, rounding);
            let sum = Self::add(&x, &s_div_x, rounding);
            let x_new = Self::div(&sum, &two, rounding);

            // Check convergence
            let diff = Self::sub(&x_new, &x, rounding);
            if diff.is_zero()
                || (diff.mantissa.bits() as i64 + diff.exponent
                    < x_new.exponent - tolerance_bits as i64)
            {
                return x_new;
            }

            x = x_new;
        }

        x
    }

    /// Negate the value.
    pub fn neg(&self) -> Self {
        if self.is_nan() {
            return Self::nan(self.precision);
        }
        if self.is_pos_infinity() {
            return Self::neg_infinity(self.precision);
        }
        if self.is_neg_infinity() {
            return Self::pos_infinity(self.precision);
        }

        let mut result = self.clone();
        result.sign = !result.sign;
        result
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        if self.is_nan() {
            return Self::nan(self.precision);
        }
        if self.is_neg_infinity() {
            return Self::pos_infinity(self.precision);
        }

        let mut result = self.clone();
        result.sign = false;
        result
    }

    /// Align two floats to the same exponent for addition.
    /// Returns (m1, m2, common_exp, sign1, sign2).
    fn align_with(&self, other: &Self) -> (BigUint, BigUint, i64, bool, bool) {
        let exp_diff = self.exponent - other.exponent;

        if exp_diff >= 0 {
            // self has larger exponent - shift other's mantissa right
            let shifted_other = &other.mantissa >> (exp_diff as usize);
            (
                self.mantissa.clone(),
                shifted_other,
                self.exponent,
                self.sign,
                other.sign,
            )
        } else {
            // other has larger exponent - shift self's mantissa right
            let shift = (-exp_diff) as usize;
            let shifted_self = &self.mantissa >> shift;
            (
                shifted_self,
                other.mantissa.clone(),
                other.exponent,
                self.sign,
                other.sign,
            )
        }
    }

    /// Compare two arbitrary precision floats.
    ///
    /// Returns None if either value is NaN.
    pub fn partial_compare(&self, other: &Self) -> Option<Ordering> {
        // NaN comparisons
        if self.is_nan() || other.is_nan() {
            return None;
        }

        // Handle infinities
        if self.is_pos_infinity() {
            return if other.is_pos_infinity() {
                Some(Ordering::Equal)
            } else {
                Some(Ordering::Greater)
            };
        }
        if self.is_neg_infinity() {
            return if other.is_neg_infinity() {
                Some(Ordering::Equal)
            } else {
                Some(Ordering::Less)
            };
        }
        if other.is_pos_infinity() {
            return Some(Ordering::Less);
        }
        if other.is_neg_infinity() {
            return Some(Ordering::Greater);
        }

        // Handle zeros
        let self_zero = self.is_zero();
        let other_zero = other.is_zero();
        if self_zero && other_zero {
            return Some(Ordering::Equal);
        }
        if self_zero {
            return if other.sign {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Less)
            };
        }
        if other_zero {
            return if self.sign {
                Some(Ordering::Less)
            } else {
                Some(Ordering::Greater)
            };
        }

        // Compare signs
        if self.sign != other.sign {
            return if self.sign {
                Some(Ordering::Less)
            } else {
                Some(Ordering::Greater)
            };
        }

        // Same sign: compare magnitudes
        let (m1, m2, _, _, _) = self.align_with(other);
        let mag_cmp = m1.cmp(&m2);

        // If negative, reverse the comparison
        if self.sign {
            Some(mag_cmp.reverse())
        } else {
            Some(mag_cmp)
        }
    }

    /// Create from a string representation.
    ///
    /// Supports decimal notation (e.g., "3.14159") and scientific notation (e.g., "3.14e-10").
    pub fn from_str(s: &str, precision: Precision) -> Option<Self> {
        let s = s.trim();

        // Handle special values
        if s.eq_ignore_ascii_case("nan") {
            return Some(Self::nan(precision));
        }
        if s.eq_ignore_ascii_case("inf") || s.eq_ignore_ascii_case("+inf") {
            return Some(Self::pos_infinity(precision));
        }
        if s.eq_ignore_ascii_case("-inf") {
            return Some(Self::neg_infinity(precision));
        }

        // Try parsing as f64 first (simple approach)
        if let Ok(f) = s.parse::<f64>() {
            return Some(Self::from_f64(f, precision));
        }

        None
    }
}

impl fmt::Debug for ArbitraryFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            write!(f, "NaN")
        } else if self.is_pos_infinity() {
            write!(f, "+Inf")
        } else if self.is_neg_infinity() {
            write!(f, "-Inf")
        } else {
            write!(
                f,
                "ArbitraryFloat {{ sign: {}, mantissa: {}, exp: {}, prec: {} }}",
                self.sign,
                self.mantissa,
                self.exponent,
                self.precision.bits()
            )
        }
    }
}

impl fmt::Display for ArbitraryFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            write!(f, "NaN")
        } else if self.is_pos_infinity() {
            write!(f, "+Inf")
        } else if self.is_neg_infinity() {
            write!(f, "-Inf")
        } else {
            // Convert to f64 for display (may lose precision)
            let value = self.to_f64(RoundingMode::RoundNearest);
            if self.sign && !value.is_sign_negative() {
                write!(f, "-{}", value.abs())
            } else {
                write!(f, "{}", value)
            }
        }
    }
}

impl PartialEq for ArbitraryFloat {
    fn eq(&self, other: &Self) -> bool {
        self.partial_compare(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for ArbitraryFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_compare(other)
    }
}

impl Add for ArbitraryFloat {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::add(&self, &rhs, RoundingMode::RoundNearest)
    }
}

impl Add for &ArbitraryFloat {
    type Output = ArbitraryFloat;

    fn add(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::add(self, rhs, RoundingMode::RoundNearest)
    }
}

impl Sub for ArbitraryFloat {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::sub(&self, &rhs, RoundingMode::RoundNearest)
    }
}

impl Sub for &ArbitraryFloat {
    type Output = ArbitraryFloat;

    fn sub(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::sub(self, rhs, RoundingMode::RoundNearest)
    }
}

impl Mul for ArbitraryFloat {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::mul(&self, &rhs, RoundingMode::RoundNearest)
    }
}

impl Mul for &ArbitraryFloat {
    type Output = ArbitraryFloat;

    fn mul(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::mul(self, rhs, RoundingMode::RoundNearest)
    }
}

impl Div for ArbitraryFloat {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::div(&self, &rhs, RoundingMode::RoundNearest)
    }
}

impl Div for &ArbitraryFloat {
    type Output = ArbitraryFloat;

    fn div(self, rhs: Self) -> Self::Output {
        ArbitraryFloat::div(self, rhs, RoundingMode::RoundNearest)
    }
}

impl Neg for ArbitraryFloat {
    type Output = Self;

    fn neg(self) -> Self::Output {
        ArbitraryFloat::neg(&self)
    }
}

impl Neg for &ArbitraryFloat {
    type Output = ArbitraryFloat;

    fn neg(self) -> Self::Output {
        ArbitraryFloat::neg(self)
    }
}

// Helper trait implementations for max on Precision
impl Precision {
    /// Return the maximum of two precisions.
    fn max(self, other: Self) -> Self {
        if self.bits >= other.bits { self } else { other }
    }
}

/// Context for arbitrary precision operations.
///
/// Provides a convenient way to perform multiple operations with the same
/// precision and rounding mode.
#[derive(Debug, Clone)]
pub struct ArbitraryFloatContext {
    /// Default precision for operations.
    pub precision: Precision,
    /// Default rounding mode.
    pub rounding: RoundingMode,
}

impl ArbitraryFloatContext {
    /// Create a new context with the given precision and rounding mode.
    pub fn new(precision: Precision, rounding: RoundingMode) -> Self {
        Self {
            precision,
            rounding,
        }
    }

    /// Create a zero value.
    pub fn zero(&self) -> ArbitraryFloat {
        ArbitraryFloat::zero(self.precision)
    }

    /// Create a one value.
    pub fn one(&self) -> ArbitraryFloat {
        ArbitraryFloat::one(self.precision)
    }

    /// Create from f64.
    pub fn from_f64(&self, value: f64) -> ArbitraryFloat {
        ArbitraryFloat::from_f64(value, self.precision)
    }

    /// Add two values.
    pub fn add(&self, a: &ArbitraryFloat, b: &ArbitraryFloat) -> ArbitraryFloat {
        a.add(b, self.rounding)
    }

    /// Subtract two values.
    pub fn sub(&self, a: &ArbitraryFloat, b: &ArbitraryFloat) -> ArbitraryFloat {
        a.sub(b, self.rounding)
    }

    /// Multiply two values.
    pub fn mul(&self, a: &ArbitraryFloat, b: &ArbitraryFloat) -> ArbitraryFloat {
        a.mul(b, self.rounding)
    }

    /// Divide two values.
    pub fn div(&self, a: &ArbitraryFloat, b: &ArbitraryFloat) -> ArbitraryFloat {
        a.div(b, self.rounding)
    }

    /// Compute square root.
    pub fn sqrt(&self, a: &ArbitraryFloat) -> ArbitraryFloat {
        a.sqrt(self.rounding)
    }
}

impl Default for ArbitraryFloatContext {
    fn default() -> Self {
        Self::new(Precision::DOUBLE, RoundingMode::RoundNearest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn test_precision_constants() {
        assert_eq!(Precision::DOUBLE.bits(), 53);
        assert_eq!(Precision::EXTENDED.bits(), 64);
        assert_eq!(Precision::QUAD.bits(), 113);
        assert_eq!(Precision::HIGH.bits(), 256);
    }

    #[test]
    fn test_from_f64_basic() {
        let prec = Precision::new(64);
        let f = ArbitraryFloat::from_f64(3.14159, prec);
        let back = f.to_f64(RoundingMode::RoundNearest);
        assert!(approx_eq(back, 3.14159));
    }

    #[test]
    fn test_from_f64_special_values() {
        let prec = Precision::new(64);

        let nan = ArbitraryFloat::from_f64(f64::NAN, prec);
        assert!(nan.is_nan());

        let pos_inf = ArbitraryFloat::from_f64(f64::INFINITY, prec);
        assert!(pos_inf.is_pos_infinity());

        let neg_inf = ArbitraryFloat::from_f64(f64::NEG_INFINITY, prec);
        assert!(neg_inf.is_neg_infinity());

        let zero = ArbitraryFloat::from_f64(0.0, prec);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_addition() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(1.5, prec);
        let b = ArbitraryFloat::from_f64(2.5, prec);
        let sum = ArbitraryFloat::add(&a, &b, RoundingMode::RoundNearest);
        assert!(approx_eq(sum.to_f64(RoundingMode::RoundNearest), 4.0));
    }

    #[test]
    fn test_subtraction() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(5.0, prec);
        let b = ArbitraryFloat::from_f64(3.0, prec);
        let diff = ArbitraryFloat::sub(&a, &b, RoundingMode::RoundNearest);
        assert!(approx_eq(diff.to_f64(RoundingMode::RoundNearest), 2.0));
    }

    #[test]
    fn test_multiplication() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(3.0, prec);
        let b = ArbitraryFloat::from_f64(4.0, prec);
        let prod = ArbitraryFloat::mul(&a, &b, RoundingMode::RoundNearest);
        assert!(approx_eq(prod.to_f64(RoundingMode::RoundNearest), 12.0));
    }

    #[test]
    fn test_division() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(10.0, prec);
        let b = ArbitraryFloat::from_f64(4.0, prec);
        let quot = ArbitraryFloat::div(&a, &b, RoundingMode::RoundNearest);
        assert!(approx_eq(quot.to_f64(RoundingMode::RoundNearest), 2.5));
    }

    #[test]
    fn test_sqrt() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(4.0, prec);
        let root = a.sqrt(RoundingMode::RoundNearest);
        assert!(approx_eq(root.to_f64(RoundingMode::RoundNearest), 2.0));
    }

    #[test]
    fn test_sqrt_2() {
        let prec = Precision::new(128);
        let a = ArbitraryFloat::from_f64(2.0, prec);
        let root = a.sqrt(RoundingMode::RoundNearest);
        let expected = 2.0f64.sqrt();
        let result = root.to_f64(RoundingMode::RoundNearest);
        assert!(
            (result - expected).abs() < 1e-14,
            "sqrt(2) = {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn test_negation() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(5.0, prec);
        let neg_a = a.neg();
        assert!(approx_eq(neg_a.to_f64(RoundingMode::RoundNearest), -5.0));
    }

    #[test]
    fn test_abs() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(-5.0, prec);
        let abs_a = a.abs();
        assert!(approx_eq(abs_a.to_f64(RoundingMode::RoundNearest), 5.0));
    }

    #[test]
    fn test_comparison() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(3.0, prec);
        let b = ArbitraryFloat::from_f64(5.0, prec);
        let c = ArbitraryFloat::from_f64(3.0, prec);

        assert!(a < b);
        assert!(b > a);
        assert!(a == c);
        assert!(a <= c);
        assert!(a >= c);
    }

    #[test]
    fn test_comparison_with_nan() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(3.0, prec);
        let nan = ArbitraryFloat::nan(prec);

        assert!(a.partial_compare(&nan).is_none());
        assert!(nan.partial_compare(&a).is_none());
        assert!(nan.partial_compare(&nan).is_none());
    }

    #[test]
    fn test_infinity_operations() {
        let prec = Precision::new(64);
        let pos_inf = ArbitraryFloat::pos_infinity(prec);
        let neg_inf = ArbitraryFloat::neg_infinity(prec);
        let one = ArbitraryFloat::from_f64(1.0, prec);

        // inf + 1 = inf
        let sum = ArbitraryFloat::add(&pos_inf, &one, RoundingMode::RoundNearest);
        assert!(sum.is_pos_infinity());

        // inf + (-inf) = NaN
        let sum2 = ArbitraryFloat::add(&pos_inf, &neg_inf, RoundingMode::RoundNearest);
        assert!(sum2.is_nan());

        // inf * 2 = inf
        let two = ArbitraryFloat::from_f64(2.0, prec);
        let prod = ArbitraryFloat::mul(&pos_inf, &two, RoundingMode::RoundNearest);
        assert!(prod.is_pos_infinity());

        // inf * 0 = NaN
        let zero = ArbitraryFloat::zero(prec);
        let prod2 = ArbitraryFloat::mul(&pos_inf, &zero, RoundingMode::RoundNearest);
        assert!(prod2.is_nan());
    }

    #[test]
    fn test_zero_operations() {
        let prec = Precision::new(64);
        let zero = ArbitraryFloat::zero(prec);
        let one = ArbitraryFloat::from_f64(1.0, prec);

        // 0 + 1 = 1
        let sum = ArbitraryFloat::add(&zero, &one, RoundingMode::RoundNearest);
        assert!(approx_eq(sum.to_f64(RoundingMode::RoundNearest), 1.0));

        // 0 * 1 = 0
        let prod = ArbitraryFloat::mul(&zero, &one, RoundingMode::RoundNearest);
        assert!(prod.is_zero());

        // 1 / 0 = inf
        let quot = ArbitraryFloat::div(&one, &zero, RoundingMode::RoundNearest);
        assert!(quot.is_pos_infinity());

        // 0 / 0 = NaN
        let quot2 = ArbitraryFloat::div(&zero, &zero, RoundingMode::RoundNearest);
        assert!(quot2.is_nan());
    }

    #[test]
    fn test_operator_overloads() {
        let prec = Precision::new(64);
        let a = ArbitraryFloat::from_f64(10.0, prec);
        let b = ArbitraryFloat::from_f64(3.0, prec);

        let sum = a.clone() + b.clone();
        assert!(approx_eq(sum.to_f64(RoundingMode::RoundNearest), 13.0));

        let diff = a.clone() - b.clone();
        assert!(approx_eq(diff.to_f64(RoundingMode::RoundNearest), 7.0));

        let prod = a.clone() * b.clone();
        assert!(approx_eq(prod.to_f64(RoundingMode::RoundNearest), 30.0));

        let quot = a.clone() / b.clone();
        let result = quot.to_f64(RoundingMode::RoundNearest);
        assert!(
            (result - 10.0 / 3.0).abs() < 1e-10,
            "10/3 = {}, expected {}",
            result,
            10.0 / 3.0
        );

        let neg = -a;
        assert!(approx_eq(neg.to_f64(RoundingMode::RoundNearest), -10.0));
    }

    #[test]
    fn test_context() {
        let ctx = ArbitraryFloatContext::new(Precision::new(128), RoundingMode::RoundNearest);

        let a = ctx.from_f64(3.14159);
        let b = ctx.from_f64(2.71828);

        let sum = ctx.add(&a, &b);
        assert!(approx_eq(
            sum.to_f64(RoundingMode::RoundNearest),
            3.14159 + 2.71828
        ));

        let sqrt_2 = ctx.sqrt(&ctx.from_f64(2.0));
        assert!((sqrt_2.to_f64(RoundingMode::RoundNearest) - 2.0f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_from_str() {
        let prec = Precision::new(64);

        let nan = ArbitraryFloat::from_str("NaN", prec).unwrap();
        assert!(nan.is_nan());

        let inf = ArbitraryFloat::from_str("inf", prec).unwrap();
        assert!(inf.is_pos_infinity());

        let neg_inf = ArbitraryFloat::from_str("-inf", prec).unwrap();
        assert!(neg_inf.is_neg_infinity());

        let pi = ArbitraryFloat::from_str("3.14159", prec).unwrap();
        assert!(approx_eq(pi.to_f64(RoundingMode::RoundNearest), 3.14159));
    }

    #[test]
    fn test_high_precision() {
        // Test that higher precision gives more accurate results
        let low_prec = Precision::new(53);
        let high_prec = Precision::new(256);

        // Compute 1/3 * 3 at different precisions
        let one_low = ArbitraryFloat::from_f64(1.0, low_prec);
        let three_low = ArbitraryFloat::from_f64(3.0, low_prec);
        let div_low = ArbitraryFloat::div(&one_low, &three_low, RoundingMode::RoundNearest);
        let result_low = ArbitraryFloat::mul(&div_low, &three_low, RoundingMode::RoundNearest);

        let one_high = ArbitraryFloat::from_f64(1.0, high_prec);
        let three_high = ArbitraryFloat::from_f64(3.0, high_prec);
        let div_high = ArbitraryFloat::div(&one_high, &three_high, RoundingMode::RoundNearest);
        let result_high = ArbitraryFloat::mul(&div_high, &three_high, RoundingMode::RoundNearest);

        // Both should be close to 1, but high precision should be closer
        let error_low = (result_low.to_f64(RoundingMode::RoundNearest) - 1.0).abs();
        let error_high = (result_high.to_f64(RoundingMode::RoundNearest) - 1.0).abs();

        assert!(
            error_high <= error_low + 1e-15,
            "High precision error {} should be <= low precision error {}",
            error_high,
            error_low
        );
    }

    #[test]
    fn test_display() {
        let prec = Precision::new(64);

        let nan = ArbitraryFloat::nan(prec);
        assert_eq!(format!("{}", nan), "NaN");

        let pos_inf = ArbitraryFloat::pos_infinity(prec);
        assert_eq!(format!("{}", pos_inf), "+Inf");

        let neg_inf = ArbitraryFloat::neg_infinity(prec);
        assert_eq!(format!("{}", neg_inf), "-Inf");
    }
}
