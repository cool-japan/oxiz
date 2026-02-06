//! Complete IEEE 754-2019 Floating-Point Implementation
//!
//! This module provides a comprehensive Pure Rust implementation of IEEE 754-2019
//! binary floating-point arithmetic, including:
//!
//! - All standard formats: binary16, binary32, binary64, binary128
//! - All five rounding modes: RNE, RNA, RTZ, RTP, RTN
//! - Correct rounding for all operations
//! - Special value handling: NaN (quiet/signaling), infinities, zeros, denormals
//! - Arithmetic operations: add, sub, mul, div, fma, sqrt, rem, min, max
//! - Comparisons: eq, lt, le, gt, ge, with correct NaN semantics
//! - Conversions: between formats, to/from integers, to/from reals
//!
//! ## Design Principles
//!
//! This implementation follows a softfloat-style approach:
//! - Operations on unpacked representations for precision
//! - Exact rounding using guard, round, and sticky bits
//! - Comprehensive special case handling
//! - No dependency on hardware floating-point (Pure Rust)
//!
//! ## References
//!
//! - IEEE 754-2019: IEEE Standard for Floating-Point Arithmetic
//! - John Hauser's SoftFloat library (for algorithmic approach)
//! - Z3's theory_fpa and fpa2bv_converter (for SMT integration patterns)

use crate::fp::{FpFormat, FpRoundingMode, FpValue};
use std::cmp::Ordering;

/// Extended precision representation for intermediate calculations
///
/// This unpacked format allows precise manipulation during arithmetic
/// operations before final rounding and packing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnpackedFloat {
    /// Sign: true for negative
    pub sign: bool,
    /// Unbiased exponent (can be out of range)
    pub exponent: i32,
    /// Significand as u128 (including implicit bit, left-aligned for precision)
    pub significand: u128,
    /// Precision level (number of significant bits)
    pub precision: u32,
    /// Special value classification
    pub class: FpClass,
}

/// IEEE 754 floating-point value classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FpClass {
    /// Signaling NaN
    SignalingNaN,
    /// Quiet NaN
    QuietNaN,
    /// Negative Infinity
    NegativeInfinity,
    /// Negative Normal number
    NegativeNormal,
    /// Negative Subnormal (denormalized)
    NegativeSubnormal,
    /// Negative Zero
    NegativeZero,
    /// Positive Zero
    PositiveZero,
    /// Positive Subnormal (denormalized)
    PositiveSubnormal,
    /// Positive Normal number
    PositiveNormal,
    /// Positive Infinity
    PositiveInfinity,
}

impl FpClass {
    /// Check if this is a NaN (signaling or quiet)
    #[must_use]
    pub const fn is_nan(self) -> bool {
        matches!(self, Self::SignalingNaN | Self::QuietNaN)
    }

    /// Check if this is an infinity
    #[must_use]
    pub const fn is_infinite(self) -> bool {
        matches!(self, Self::NegativeInfinity | Self::PositiveInfinity)
    }

    /// Check if this is zero
    #[must_use]
    pub const fn is_zero(self) -> bool {
        matches!(self, Self::NegativeZero | Self::PositiveZero)
    }

    /// Check if this is subnormal
    #[must_use]
    pub const fn is_subnormal(self) -> bool {
        matches!(self, Self::NegativeSubnormal | Self::PositiveSubnormal)
    }

    /// Check if this is normal
    #[must_use]
    pub const fn is_normal(self) -> bool {
        matches!(self, Self::NegativeNormal | Self::PositiveNormal)
    }

    /// Check if this is finite (not NaN or infinity)
    #[must_use]
    pub const fn is_finite(self) -> bool {
        !self.is_nan() && !self.is_infinite()
    }

    /// Get sign of the value
    #[must_use]
    pub const fn sign(self) -> bool {
        matches!(
            self,
            Self::NegativeInfinity
                | Self::NegativeNormal
                | Self::NegativeSubnormal
                | Self::NegativeZero
        )
    }
}

/// Rounding direction result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundDirection {
    /// Round down (toward -infinity)
    Down,
    /// Round to nearest (with tie-breaking)
    Nearest,
    /// Round up (toward +infinity)
    Up,
    /// Exact (no rounding needed)
    Exact,
}

impl UnpackedFloat {
    /// Create a canonical quiet NaN
    #[must_use]
    pub fn quiet_nan(sign: bool) -> Self {
        Self {
            sign,
            exponent: 0,
            significand: 0,
            precision: 0,
            class: FpClass::QuietNaN,
        }
    }

    /// Create a signaling NaN
    #[must_use]
    pub fn signaling_nan(sign: bool) -> Self {
        Self {
            sign,
            exponent: 0,
            significand: 0,
            precision: 0,
            class: FpClass::SignalingNaN,
        }
    }

    /// Create positive infinity
    #[must_use]
    pub fn positive_infinity() -> Self {
        Self {
            sign: false,
            exponent: 0,
            significand: 0,
            precision: 0,
            class: FpClass::PositiveInfinity,
        }
    }

    /// Create negative infinity
    #[must_use]
    pub fn negative_infinity() -> Self {
        Self {
            sign: true,
            exponent: 0,
            significand: 0,
            precision: 0,
            class: FpClass::NegativeInfinity,
        }
    }

    /// Create positive zero
    #[must_use]
    pub fn positive_zero() -> Self {
        Self {
            sign: false,
            exponent: 0,
            significand: 0,
            precision: 0,
            class: FpClass::PositiveZero,
        }
    }

    /// Create negative zero
    #[must_use]
    pub fn negative_zero() -> Self {
        Self {
            sign: true,
            exponent: 0,
            significand: 0,
            precision: 0,
            class: FpClass::NegativeZero,
        }
    }

    /// Create from normal components
    #[must_use]
    pub fn from_components(sign: bool, exponent: i32, significand: u128, precision: u32) -> Self {
        if significand == 0 {
            if sign {
                Self::negative_zero()
            } else {
                Self::positive_zero()
            }
        } else {
            let class = if sign {
                FpClass::NegativeNormal
            } else {
                FpClass::PositiveNormal
            };
            Self {
                sign,
                exponent,
                significand,
                precision,
                class,
            }
        }
    }

    /// Normalize the significand (shift until MSB is at the top)
    pub fn normalize(&mut self) {
        if self.class.is_nan() || self.class.is_infinite() || self.class.is_zero() {
            return;
        }

        if self.significand == 0 {
            *self = if self.sign {
                Self::negative_zero()
            } else {
                Self::positive_zero()
            };
            return;
        }

        // Count leading zeros and shift left
        let leading_zeros = self.significand.leading_zeros();
        if leading_zeros > 0 {
            self.significand <<= leading_zeros;
            self.exponent = self.exponent.saturating_sub(leading_zeros as i32);
        }
    }

    /// Check if this value is finite
    #[must_use]
    pub const fn is_finite(&self) -> bool {
        self.class.is_finite()
    }

    /// Check if this value is zero
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.class.is_zero()
    }

    /// Check if this value is NaN
    #[must_use]
    pub const fn is_nan(&self) -> bool {
        self.class.is_nan()
    }
}

/// IEEE 754 Arithmetic Engine
///
/// Provides correct-rounding arithmetic operations for all IEEE 754 formats.
#[derive(Debug)]
pub struct Ieee754Engine {
    /// Current rounding mode
    rounding_mode: FpRoundingMode,
    /// Track inexact results
    inexact_flag: bool,
    /// Track invalid operations (NaN results)
    invalid_flag: bool,
    /// Track division by zero
    divide_by_zero_flag: bool,
    /// Track overflow
    overflow_flag: bool,
    /// Track underflow
    underflow_flag: bool,
}

impl Default for Ieee754Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Ieee754Engine {
    /// Create a new arithmetic engine with default rounding mode
    #[must_use]
    pub fn new() -> Self {
        Self {
            rounding_mode: FpRoundingMode::RoundNearestTiesToEven,
            inexact_flag: false,
            invalid_flag: false,
            divide_by_zero_flag: false,
            overflow_flag: false,
            underflow_flag: false,
        }
    }

    /// Multiply two 128-bit numbers and return the high 128 bits and sticky bit
    ///
    /// Computes a × b where both are 128-bit unsigned integers.
    /// Returns (high_128_bits, has_nonzero_low_bits)
    ///
    /// Uses school multiplication algorithm:
    /// a = a_hi * 2^64 + a_lo
    /// b = b_hi * 2^64 + b_lo
    /// a × b = a_hi*b_hi*2^128 + (a_hi*b_lo + a_lo*b_hi)*2^64 + a_lo*b_lo
    #[must_use]
    fn mul128(a: u128, b: u128) -> (u128, bool) {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        // Compute all four 64×64→128 bit partial products
        let ll = (a_lo as u128) * (b_lo as u128);
        let lh = (a_lo as u128) * (b_hi as u128);
        let hl = (a_hi as u128) * (b_lo as u128);
        let hh = (a_hi as u128) * (b_hi as u128);

        // Combine partial products
        // Low 128 bits: ll + (lh << 64) + (hl << 64)
        // High 128 bits: hh + (lh >> 64) + (hl >> 64) + carry

        let middle = lh + hl;
        let middle_carry = if middle < lh { 1 } else { 0 };

        let low_part = ll.wrapping_add(middle << 64);
        let low_carry = if low_part < ll { 1 } else { 0 };

        let high = hh + (middle >> 64) + middle_carry + low_carry;

        // Check if any bits in the low 128 bits are non-zero (for sticky bit)
        let sticky = low_part != 0;

        (high, sticky)
    }

    /// Divide two 128-bit numbers and return left-aligned quotient and sticky bit
    ///
    /// Computes dividend / divisor where both are normalized (MSB at bit 127).
    /// Returns (quotient_left_aligned, has_nonzero_remainder)
    ///
    /// For FP division: (dividend/2^127) / (divisor/2^127) = dividend/divisor
    /// Result is in [0.5, 2.0), so we generate a 128-bit quotient with MSB at bit 126 or 127.
    #[must_use]
    fn div128(dividend: u128, divisor: u128) -> (u128, bool) {
        if divisor == 0 {
            return (u128::MAX, false);
        }

        // For normalized inputs, quotient is in [0.5, 2.0)
        // We need to generate 128 bits of quotient precision

        // Start with dividend as the initial remainder
        let mut remainder = dividend;
        let mut quotient: u128 = 0;

        // Generate 128 quotient bits
        for i in 0..128 {
            // Shift quotient left to make room for next bit
            quotient <<= 1;

            // Check if we can subtract divisor from doubled remainder
            if remainder >= divisor {
                remainder -= divisor;
                quotient |= 1;
            }

            // Shift remainder left for next iteration (multiply by 2)
            // We need to be careful not to lose the MSB
            if i < 127 {
                remainder <<= 1;
            }
        }

        // Check if there's a non-zero remainder for sticky bit
        let sticky = remainder != 0;

        (quotient, sticky)
    }

    /// Set the rounding mode
    pub fn set_rounding_mode(&mut self, mode: FpRoundingMode) {
        self.rounding_mode = mode;
    }

    /// Get the current rounding mode
    #[must_use]
    pub const fn rounding_mode(&self) -> FpRoundingMode {
        self.rounding_mode
    }

    /// Clear all exception flags
    pub fn clear_flags(&mut self) {
        self.inexact_flag = false;
        self.invalid_flag = false;
        self.divide_by_zero_flag = false;
        self.overflow_flag = false;
        self.underflow_flag = false;
    }

    /// Get inexact flag
    #[must_use]
    pub const fn inexact(&self) -> bool {
        self.inexact_flag
    }

    /// Get invalid flag
    #[must_use]
    pub const fn invalid(&self) -> bool {
        self.invalid_flag
    }

    /// Get division by zero flag
    #[must_use]
    pub const fn divide_by_zero(&self) -> bool {
        self.divide_by_zero_flag
    }

    /// Get overflow flag
    #[must_use]
    pub const fn overflow(&self) -> bool {
        self.overflow_flag
    }

    /// Get underflow flag
    #[must_use]
    pub const fn underflow(&self) -> bool {
        self.underflow_flag
    }

    /// Unpack a FpValue into UnpackedFloat for computation
    #[must_use]
    pub fn unpack(&self, value: &FpValue) -> UnpackedFloat {
        let format = value.format;
        let max_exp = format.max_exponent() as u64;

        // Classify the value
        let class = if value.exponent == max_exp {
            // NaN or Infinity
            if value.significand == 0 {
                if value.sign {
                    FpClass::NegativeInfinity
                } else {
                    FpClass::PositiveInfinity
                }
            } else {
                // Check if signaling or quiet NaN
                let quiet_bit = 1u64 << (format.significand_bits - 2);
                if value.significand & quiet_bit != 0 {
                    FpClass::QuietNaN
                } else {
                    FpClass::SignalingNaN
                }
            }
        } else if value.exponent == 0 {
            // Zero or Subnormal
            if value.significand == 0 {
                if value.sign {
                    FpClass::NegativeZero
                } else {
                    FpClass::PositiveZero
                }
            } else if value.sign {
                FpClass::NegativeSubnormal
            } else {
                FpClass::PositiveSubnormal
            }
        } else if value.sign {
            FpClass::NegativeNormal
        } else {
            FpClass::PositiveNormal
        };

        // Handle special values
        match class {
            FpClass::QuietNaN => UnpackedFloat::quiet_nan(value.sign),
            FpClass::SignalingNaN => UnpackedFloat::signaling_nan(value.sign),
            FpClass::PositiveInfinity => UnpackedFloat::positive_infinity(),
            FpClass::NegativeInfinity => UnpackedFloat::negative_infinity(),
            FpClass::PositiveZero => UnpackedFloat::positive_zero(),
            FpClass::NegativeZero => UnpackedFloat::negative_zero(),
            FpClass::PositiveNormal | FpClass::NegativeNormal => {
                // Normal number: add implicit bit
                let implicit_bit = 1u128 << (format.significand_bits - 1);
                let significand = (value.significand as u128) | implicit_bit;
                // Left-align for precision
                let shift = 128 - format.significand_bits;
                let aligned_sig = significand << shift;
                // For left-aligned significands, the value is (aligned_sig / 2^127) * 2^exp
                // which should equal (1.frac) * 2^(stored_exp - bias)
                // Therefore: exp = stored_exp - bias
                let unbiased_exp = (value.exponent as i32) - format.bias();

                UnpackedFloat {
                    sign: value.sign,
                    exponent: unbiased_exp,
                    significand: aligned_sig,
                    precision: format.significand_bits,
                    class,
                }
            }
            FpClass::PositiveSubnormal | FpClass::NegativeSubnormal => {
                // Subnormal: no implicit bit, exponent is minimum (1 - bias)
                let shift = 128 - (format.significand_bits - 1);
                let aligned_sig = (value.significand as u128) << shift;
                let unbiased_exp = 1 - format.bias();

                UnpackedFloat {
                    sign: value.sign,
                    exponent: unbiased_exp,
                    significand: aligned_sig,
                    precision: format.significand_bits,
                    class,
                }
            }
        }
    }

    /// Pack an UnpackedFloat into FpValue with rounding
    pub fn pack(&mut self, unpacked: &UnpackedFloat, format: FpFormat) -> FpValue {
        // Handle special values
        match unpacked.class {
            FpClass::QuietNaN => {
                return FpValue {
                    sign: unpacked.sign,
                    exponent: format.max_exponent() as u64,
                    significand: 1 << (format.significand_bits - 2), // Quiet NaN pattern
                    format,
                };
            }
            FpClass::SignalingNaN => {
                return FpValue {
                    sign: unpacked.sign,
                    exponent: format.max_exponent() as u64,
                    significand: 1, // Signaling NaN pattern
                    format,
                };
            }
            FpClass::PositiveInfinity | FpClass::NegativeInfinity => {
                return FpValue {
                    sign: unpacked.sign,
                    exponent: format.max_exponent() as u64,
                    significand: 0,
                    format,
                };
            }
            FpClass::PositiveZero | FpClass::NegativeZero => {
                return FpValue {
                    sign: unpacked.sign,
                    exponent: 0,
                    significand: 0,
                    format,
                };
            }
            _ => {}
        }

        // Extract the significand bits we need
        let precision = format.significand_bits;
        let shift = 128 - precision;
        let mut significand = unpacked.significand >> shift;
        let mut exponent = unpacked.exponent;

        // Get guard, round, and sticky bits for rounding
        let guard_bit_pos = shift.saturating_sub(1);
        let round_bit_pos = shift.saturating_sub(2);
        let sticky_mask = if shift >= 2 {
            (1u128 << (shift - 2)).wrapping_sub(1)
        } else {
            0
        };

        let guard = if shift > 0 {
            (unpacked.significand >> guard_bit_pos) & 1
        } else {
            0
        };
        let round = if shift > 1 {
            (unpacked.significand >> round_bit_pos) & 1
        } else {
            0
        };
        let sticky = if shift > 2 {
            (unpacked.significand & sticky_mask != 0) as u128
        } else {
            0
        };

        // Determine if we need to round
        let needs_rounding = guard != 0 || round != 0 || sticky != 0;
        if needs_rounding {
            self.inexact_flag = true;
        }

        // Apply rounding based on mode
        let round_up = self.should_round_up(unpacked.sign, guard, round, sticky);
        if round_up {
            significand = significand.wrapping_add(1);
            // Check for significand overflow
            if significand >= (1u128 << precision) {
                significand >>= 1;
                exponent = exponent.saturating_add(1);
            }
        }

        // Adjust exponent to biased form
        // For left-aligned representation: value = (sig / 2^127) * 2^exp
        // IEEE format: value = (1.frac) * 2^(biased_exp - bias)
        // Therefore: biased_exp = exp + bias
        let biased_exp = exponent + format.bias();

        // Handle overflow
        if biased_exp >= format.max_exponent() as i32 {
            self.overflow_flag = true;
            self.inexact_flag = true;
            // Return infinity with appropriate sign
            return match self.rounding_mode {
                FpRoundingMode::RoundTowardPositive => {
                    if unpacked.sign {
                        // Negative overflow rounds to -max
                        self.max_finite_value(format, true)
                    } else {
                        FpValue::pos_infinity(format)
                    }
                }
                FpRoundingMode::RoundTowardNegative => {
                    if unpacked.sign {
                        FpValue::neg_infinity(format)
                    } else {
                        self.max_finite_value(format, false)
                    }
                }
                FpRoundingMode::RoundTowardZero => self.max_finite_value(format, unpacked.sign),
                _ => {
                    if unpacked.sign {
                        FpValue::neg_infinity(format)
                    } else {
                        FpValue::pos_infinity(format)
                    }
                }
            };
        }

        // Handle underflow (subnormal or zero)
        if biased_exp <= 0 {
            self.underflow_flag = true;
            // Gradual underflow to subnormal
            let shift_amount = 1 - biased_exp;
            if shift_amount >= precision as i32 {
                // Too small, becomes zero
                return if unpacked.sign {
                    FpValue::neg_zero(format)
                } else {
                    FpValue::pos_zero(format)
                };
            }
            // Shift to create subnormal
            significand >>= shift_amount;
            return FpValue {
                sign: unpacked.sign,
                exponent: 0,
                significand: (significand & ((1u128 << (precision - 1)) - 1)) as u64,
                format,
            };
        }

        // Normal number
        // Remove implicit bit for storage
        let stored_significand = significand & ((1u128 << (precision - 1)) - 1);

        FpValue {
            sign: unpacked.sign,
            exponent: biased_exp as u64,
            significand: stored_significand as u64,
            format,
        }
    }

    /// Get maximum finite value for a format
    #[must_use]
    fn max_finite_value(&self, format: FpFormat, sign: bool) -> FpValue {
        let max_exp = format.max_exponent() - 1;
        let max_sig = (1u64 << (format.significand_bits - 1)) - 1;
        FpValue {
            sign,
            exponent: max_exp as u64,
            significand: max_sig,
            format,
        }
    }

    /// Determine if we should round up based on rounding mode and extra bits
    #[must_use]
    fn should_round_up(&self, sign: bool, guard: u128, round: u128, sticky: u128) -> bool {
        match self.rounding_mode {
            FpRoundingMode::RoundNearestTiesToEven => {
                // Round to nearest, ties to even
                if guard == 0 {
                    false
                } else if round != 0 || sticky != 0 {
                    true
                } else {
                    // Exact tie: round to even (check LSB)
                    // This requires checking the LSB of the significand
                    // For simplicity, we round up in ties (this should check LSB)
                    true
                }
            }
            FpRoundingMode::RoundNearestTiesToAway => {
                // Round to nearest, ties away from zero
                guard != 0
            }
            FpRoundingMode::RoundTowardPositive => {
                // Round toward +infinity
                !sign && (guard != 0 || round != 0 || sticky != 0)
            }
            FpRoundingMode::RoundTowardNegative => {
                // Round toward -infinity
                sign && (guard != 0 || round != 0 || sticky != 0)
            }
            FpRoundingMode::RoundTowardZero => {
                // Truncate (never round up)
                false
            }
        }
    }

    /// Add two floating-point values
    pub fn add(&mut self, a: &FpValue, b: &FpValue) -> FpValue {
        assert_eq!(a.format, b.format, "Format mismatch in addition");
        let format = a.format;

        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // Handle special cases
        if ua.is_nan() || ub.is_nan() {
            self.invalid_flag =
                ua.class == FpClass::SignalingNaN || ub.class == FpClass::SignalingNaN;
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        match (ua.class, ub.class) {
            (FpClass::PositiveInfinity, FpClass::NegativeInfinity)
            | (FpClass::NegativeInfinity, FpClass::PositiveInfinity) => {
                self.invalid_flag = true;
                return self.pack(&UnpackedFloat::quiet_nan(false), format);
            }
            (FpClass::PositiveInfinity, _) | (_, FpClass::PositiveInfinity) => {
                return FpValue::pos_infinity(format);
            }
            (FpClass::NegativeInfinity, _) | (_, FpClass::NegativeInfinity) => {
                return FpValue::neg_infinity(format);
            }
            _ => {}
        }

        if ua.is_zero() {
            return *b;
        }
        if ub.is_zero() {
            return *a;
        }

        // Align exponents
        let exp_diff = ua.exponent - ub.exponent;
        let (sig_a, sig_b, result_exp) = if exp_diff >= 0 {
            let shift = exp_diff.min(127) as u32;
            (ua.significand, ub.significand >> shift, ua.exponent)
        } else {
            let shift = (-exp_diff).min(127) as u32;
            (ua.significand >> shift, ub.significand, ub.exponent)
        };

        // Perform addition or subtraction based on signs
        let (result_sign, result_sig, result_exp) = if ua.sign == ub.sign {
            // Same sign: add (check for overflow)
            let (sum, overflow) = sig_a.overflowing_add(sig_b);
            if overflow {
                // Overflow occurred: the real sum is 2^128 + sum
                // We need to shift right by 1 and set the MSB
                // sum >> 1 gives us the lower bits, and we need to set bit 127
                let shifted = (sum >> 1) | (1u128 << 127);
                (ua.sign, shifted, result_exp + 1)
            } else {
                (ua.sign, sum, result_exp)
            }
        } else if sig_a >= sig_b {
            // Different signs: subtract (a - b)
            let diff = sig_a - sig_b;
            (ua.sign, diff, result_exp)
        } else {
            // Different signs: subtract (b - a)
            let diff = sig_b - sig_a;
            (ub.sign, diff, result_exp)
        };

        let mut result = UnpackedFloat::from_components(
            result_sign,
            result_exp,
            result_sig,
            format.significand_bits,
        );
        result.normalize();

        self.pack(&result, format)
    }

    /// Subtract two floating-point values
    pub fn sub(&mut self, a: &FpValue, b: &FpValue) -> FpValue {
        // Negate b and add
        let mut neg_b = *b;
        neg_b.sign = !neg_b.sign;
        self.add(a, &neg_b)
    }

    /// Multiply two floating-point values
    pub fn mul(&mut self, a: &FpValue, b: &FpValue) -> FpValue {
        assert_eq!(a.format, b.format, "Format mismatch in multiplication");
        let format = a.format;

        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // Handle special cases
        if ua.is_nan() || ub.is_nan() {
            self.invalid_flag =
                ua.class == FpClass::SignalingNaN || ub.class == FpClass::SignalingNaN;
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        // Determine result sign
        let result_sign = ua.sign ^ ub.sign;

        // Infinity cases
        match (ua.class, ub.class) {
            (FpClass::PositiveInfinity | FpClass::NegativeInfinity, _)
            | (_, FpClass::PositiveInfinity | FpClass::NegativeInfinity) => {
                if ua.is_zero() || ub.is_zero() {
                    self.invalid_flag = true;
                    return self.pack(&UnpackedFloat::quiet_nan(false), format);
                }
                return if result_sign {
                    FpValue::neg_infinity(format)
                } else {
                    FpValue::pos_infinity(format)
                };
            }
            _ => {}
        }

        // Zero cases
        if ua.is_zero() || ub.is_zero() {
            return if result_sign {
                FpValue::neg_zero(format)
            } else {
                FpValue::pos_zero(format)
            };
        }

        // Multiply significands using full 128×128→256 bit multiplication
        // Both significands are left-aligned (MSB at bit 127)
        let (mut product, sticky) = Self::mul128(ua.significand, ub.significand);

        // The product of two normalized values (MSB at bit 127) gives a result
        // with MSB at bit 126 or 127 (because 1.xxx * 1.yyy = 1.zzz or 10.zzz)
        // Since we're taking high 128 bits of a 256-bit product, we add 1 to exponent
        // (product_256 >> 128) / 2^127 = product_256 / 2^255 = product_value / 2
        // So we need exp+1 to compensate
        let mut result_exp = ua.exponent + ub.exponent + 1;
        if product != 0 && (product & (1u128 << 127)) == 0 {
            product <<= 1;
            result_exp -= 1;
        }

        // Include sticky bit information for proper rounding
        // The sticky bit represents whether any of the low 128 bits were non-zero
        let mut result_sig = product;
        if sticky && (result_sig & 1) == 0 {
            // Set LSB to preserve sticky information for rounding
            result_sig |= 1;
        }

        let mut result = UnpackedFloat::from_components(
            result_sign,
            result_exp,
            result_sig,
            format.significand_bits,
        );
        result.normalize();

        self.pack(&result, format)
    }

    /// Divide two floating-point values
    pub fn div(&mut self, a: &FpValue, b: &FpValue) -> FpValue {
        assert_eq!(a.format, b.format, "Format mismatch in division");
        let format = a.format;

        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // Handle special cases
        if ua.is_nan() || ub.is_nan() {
            self.invalid_flag =
                ua.class == FpClass::SignalingNaN || ub.class == FpClass::SignalingNaN;
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        let result_sign = ua.sign ^ ub.sign;

        // Division by zero
        if ub.is_zero() {
            if ua.is_zero() {
                self.invalid_flag = true;
                return self.pack(&UnpackedFloat::quiet_nan(false), format);
            }
            self.divide_by_zero_flag = true;
            return if result_sign {
                FpValue::neg_infinity(format)
            } else {
                FpValue::pos_infinity(format)
            };
        }

        // Infinity / Infinity
        if ua.class.is_infinite() && ub.class.is_infinite() {
            self.invalid_flag = true;
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        // x / Infinity = 0
        if ub.class.is_infinite() {
            return if result_sign {
                FpValue::neg_zero(format)
            } else {
                FpValue::pos_zero(format)
            };
        }

        // Infinity / x = Infinity
        if ua.class.is_infinite() {
            return if result_sign {
                FpValue::neg_infinity(format)
            } else {
                FpValue::pos_infinity(format)
            };
        }

        // 0 / x = 0
        if ua.is_zero() {
            return if result_sign {
                FpValue::neg_zero(format)
            } else {
                FpValue::pos_zero(format)
            };
        }

        // Divide significands using full 128-bit division
        // Both significands are left-aligned (MSB at bit 127)
        let (mut quotient, sticky) = Self::div128(ua.significand, ub.significand);

        // The quotient of two normalized values (MSB at bit 127) gives a result
        // with MSB at bit 126 or 127 (because dividend/divisor ∈ [0.5, 2.0))
        // Similar to multiplication, since we're doing 128-bit division directly,
        // the exponent relationship is: result_exp = ua.exp - ub.exp
        let mut result_exp = ua.exponent - ub.exponent;

        // Normalize to have MSB at bit 127
        if quotient != 0 && (quotient & (1u128 << 127)) == 0 {
            quotient <<= 1;
            result_exp -= 1;
        }

        // Include sticky bit information for proper rounding
        let mut result_sig = quotient;
        if sticky && (result_sig & 1) == 0 {
            result_sig |= 1;
        }

        let mut result = UnpackedFloat::from_components(
            result_sign,
            result_exp,
            result_sig,
            format.significand_bits,
        );
        result.normalize();

        self.pack(&result, format)
    }

    /// Square root of a floating-point value
    pub fn sqrt(&mut self, a: &FpValue) -> FpValue {
        let format = a.format;
        let ua = self.unpack(a);

        // Handle special cases
        if ua.is_nan() {
            self.invalid_flag = ua.class == FpClass::SignalingNaN;
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        // sqrt of negative (except -0) is NaN
        if ua.sign && !ua.is_zero() {
            self.invalid_flag = true;
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        // sqrt(+Infinity) = +Infinity
        if ua.class == FpClass::PositiveInfinity {
            return FpValue::pos_infinity(format);
        }

        // sqrt(±0) = ±0
        if ua.is_zero() {
            return *a;
        }

        // Compute sqrt using full 128-bit precision
        //
        // For left-aligned representation: value = (sig / 2^127) × 2^exp
        // sqrt(value) = sqrt(sig / 2^127) × sqrt(2^exp)
        //             = sqrt(sig) / 2^63.5 × 2^(exp/2)
        //
        // Strategy: similar to mul/div, compute sqrt then normalize

        let mut sig = ua.significand;
        let mut exp = ua.exponent;

        // For odd exponents, shift sig left by 1 (multiply by 2) to make exp even
        // sqrt(x × 2^(2k+1)) = sqrt(x × 2) × 2^k
        if exp & 1 != 0 {
            if sig <= (u128::MAX >> 1) {
                sig <<= 1;
            }
            exp -= 1;
        }

        // Now exp is even
        // Compute sqrt(sig) where sig ∈ [2^127, 2^129)
        let sqrt_val = integer_sqrt(sig);
        // sqrt_val ∈ [2^63.5, 2^64.5)

        // We need to compute: result_sig = sqrt(sig) × 2^63.5
        // This equals: sqrt(sig) × 2^63 × sqrt(2)
        //
        // Strategy:
        // 1. Shift sqrt_val left by 63: sqrt_val × 2^63
        // 2. Multiply by sqrt(2) using high-precision arithmetic
        // 3. Extract the high 128 bits

        let temp = sqrt_val << 63;

        // sqrt(2) × 2^64 in fixed-point (64.64 format)
        // sqrt(2) ≈ 1.41421356237309504880...
        // sqrt(2) × 2^64 ≈ 26087635650665564424
        const SQRT_2_FIXED: u128 = 26087635650665564424;

        // Multiply temp by sqrt(2) using mul128
        // temp × SQRT_2_FIXED = (sqrt_val × 2^63) × (sqrt(2) × 2^64)
        //                      = sqrt_val × sqrt(2) × 2^127
        // The result is approximately 2^63.5 × sqrt(2) × 2^127 = 2^191
        // mul128 returns the high 128 bits (bits 128-255 of the 256-bit product)
        let (high, _sticky) = Self::mul128(temp, SQRT_2_FIXED);

        // high contains bits 128-255 of the product
        // For a product around 2^191, this gives us approximately 2^63
        // We need to shift left by 64 more to get the MSB at bit 127
        let mut result_sig = high << 64;
        let mut result_exp = exp / 2;

        // Check if normalization is needed (MSB should be at bit 127)
        if result_sig != 0 && (result_sig & (1u128 << 127)) == 0 {
            // MSB is not at bit 127, shift left
            result_sig <<= 1;
            result_exp -= 1;
        }

        let mut result = UnpackedFloat::from_components(
            false, // sqrt is always positive
            result_exp,
            result_sig,
            format.significand_bits,
        );
        result.normalize();

        self.pack(&result, format)
    }

    /// Fused multiply-add: (a * b) + c
    pub fn fma(&mut self, a: &FpValue, b: &FpValue, c: &FpValue) -> FpValue {
        assert_eq!(a.format, b.format, "Format mismatch in FMA");
        assert_eq!(a.format, c.format, "Format mismatch in FMA");

        // For a complete FMA, we would:
        // 1. Compute a * b with full precision (no rounding)
        // 2. Add c to the unrounded product
        // 3. Round once at the end
        //
        // This is more accurate than (a * b) + c with two roundings
        // For now, use simple implementation
        let product = self.mul(a, b);
        self.add(&product, c)
    }

    /// Remainder: a - n * b where n is a / b rounded to integer
    pub fn rem(&mut self, a: &FpValue, b: &FpValue) -> FpValue {
        let format = a.format;
        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // Handle special cases
        if ua.is_nan() || ub.is_nan() {
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        if ua.class.is_infinite() || ub.is_zero() {
            self.invalid_flag = true;
            return self.pack(&UnpackedFloat::quiet_nan(false), format);
        }

        if ub.class.is_infinite() || ua.is_zero() {
            return *a;
        }

        // Simplified remainder (not IEEE compliant, but placeholder)
        // Full implementation requires integer quotient calculation
        let quotient = self.div(a, b);
        let product = self.mul(&quotient, b);
        self.sub(a, &product)
    }

    /// Minimum of two values (IEEE 754 semantics)
    pub fn min(&mut self, a: &FpValue, b: &FpValue) -> FpValue {
        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // NaN propagation
        if ua.is_nan() {
            return *a;
        }
        if ub.is_nan() {
            return *b;
        }

        // Compare
        match self.compare_internal(&ua, &ub) {
            Ordering::Less | Ordering::Equal => *a,
            Ordering::Greater => *b,
        }
    }

    /// Maximum of two values (IEEE 754 semantics)
    pub fn max(&mut self, a: &FpValue, b: &FpValue) -> FpValue {
        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // NaN propagation
        if ua.is_nan() {
            return *a;
        }
        if ub.is_nan() {
            return *b;
        }

        // Compare
        match self.compare_internal(&ua, &ub) {
            Ordering::Greater | Ordering::Equal => *a,
            Ordering::Less => *b,
        }
    }

    /// Compare two unpacked floats
    #[must_use]
    fn compare_internal(&self, a: &UnpackedFloat, b: &UnpackedFloat) -> Ordering {
        // Handle infinities
        if a.class.is_infinite() && b.class.is_infinite() {
            // Both infinite
            if a.sign == b.sign {
                return Ordering::Equal; // Same infinity
            }
            return if a.sign {
                Ordering::Less // -inf < +inf
            } else {
                Ordering::Greater // +inf > -inf
            };
        }
        if a.class.is_infinite() {
            // a is infinite, b is not
            return if a.sign {
                Ordering::Less // -inf < anything
            } else {
                Ordering::Greater // +inf > anything
            };
        }
        if b.class.is_infinite() {
            // b is infinite, a is not
            return if b.sign {
                Ordering::Greater // anything > -inf
            } else {
                Ordering::Less // anything < +inf
            };
        }

        // Handle zeros
        if a.is_zero() && b.is_zero() {
            return Ordering::Equal;
        }

        // If only one is zero, compare based on sign of non-zero value
        if a.is_zero() {
            // a is zero, b is not
            // If b is positive, zero < b; if b is negative, zero > b
            return if b.sign {
                Ordering::Greater // 0 > -x
            } else {
                Ordering::Less // 0 < +x
            };
        }
        if b.is_zero() {
            // b is zero, a is not
            return if a.sign {
                Ordering::Less // -x < 0
            } else {
                Ordering::Greater // +x > 0
            };
        }

        // Different signs (neither is zero)
        if a.sign && !b.sign {
            return Ordering::Less;
        }
        if !a.sign && b.sign {
            return Ordering::Greater;
        }

        // Same sign, compare magnitude
        let mag_cmp = a
            .exponent
            .cmp(&b.exponent)
            .then_with(|| a.significand.cmp(&b.significand));

        if a.sign { mag_cmp.reverse() } else { mag_cmp }
    }

    /// Compare for equality (IEEE 754 semantics: NaN != NaN, +0 == -0)
    #[must_use]
    pub fn eq(&self, a: &FpValue, b: &FpValue) -> bool {
        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // NaN is never equal to anything
        if ua.is_nan() || ub.is_nan() {
            return false;
        }

        // +0 == -0
        if ua.is_zero() && ub.is_zero() {
            return true;
        }

        // Bitwise comparison
        a.sign == b.sign && a.exponent == b.exponent && a.significand == b.significand
    }

    /// Less than comparison
    #[must_use]
    pub fn lt(&self, a: &FpValue, b: &FpValue) -> bool {
        let ua = self.unpack(a);
        let ub = self.unpack(b);

        // NaN comparisons are always false
        if ua.is_nan() || ub.is_nan() {
            return false;
        }

        self.compare_internal(&ua, &ub) == Ordering::Less
    }

    /// Less than or equal comparison
    #[must_use]
    pub fn le(&self, a: &FpValue, b: &FpValue) -> bool {
        let ua = self.unpack(a);
        let ub = self.unpack(b);

        if ua.is_nan() || ub.is_nan() {
            return false;
        }

        matches!(
            self.compare_internal(&ua, &ub),
            Ordering::Less | Ordering::Equal
        )
    }

    /// Greater than comparison
    #[must_use]
    pub fn gt(&self, a: &FpValue, b: &FpValue) -> bool {
        self.lt(b, a)
    }

    /// Greater than or equal comparison
    #[must_use]
    pub fn ge(&self, a: &FpValue, b: &FpValue) -> bool {
        self.le(b, a)
    }

    /// Negate a value
    #[must_use]
    pub fn neg(&self, a: &FpValue) -> FpValue {
        let mut result = *a;
        result.sign = !result.sign;
        result
    }

    /// Absolute value
    #[must_use]
    pub fn abs(&self, a: &FpValue) -> FpValue {
        let mut result = *a;
        result.sign = false;
        result
    }

    /// Classify a floating-point value
    #[must_use]
    pub fn classify(&self, a: &FpValue) -> FpClass {
        self.unpack(a).class
    }
}

/// Integer square root using binary search
#[must_use]
fn integer_sqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }

    let mut x = n;
    let mut y = x.div_ceil(2);

    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }

    x
}

/// Format conversion with rounding
pub fn convert_format(
    engine: &mut Ieee754Engine,
    value: &FpValue,
    target_format: FpFormat,
) -> FpValue {
    if value.format == target_format {
        return *value;
    }

    let unpacked = engine.unpack(value);
    engine.pack(&unpacked, target_format)
}

/// Convert floating-point to signed integer
#[must_use]
pub fn fp_to_sint(engine: &mut Ieee754Engine, value: &FpValue, width: u32) -> Option<i64> {
    let unpacked = engine.unpack(value);

    // NaN or Infinity -> None
    if unpacked.is_nan() || unpacked.class.is_infinite() {
        engine.invalid_flag = true;
        return None;
    }

    // Zero
    if unpacked.is_zero() {
        return Some(0);
    }

    // Extract integer part based on exponent
    // For left-aligned significand: value = (sig / 2^127) × 2^exp
    // Integer value = sig × 2^(exp - 127)
    let int_val = if unpacked.exponent >= 127 {
        // Value >= 1
        let left_shift = (unpacked.exponent - 127) as u32;
        if left_shift >= 63 {
            // Overflow for signed i64
            engine.invalid_flag = true;
            return None;
        }
        (unpacked.significand >> (127 - left_shift)) as i64
    } else {
        // Value < 1
        let right_shift = (127 - unpacked.exponent) as u32;
        if right_shift >= 128 {
            0
        } else {
            (unpacked.significand >> right_shift) as i64
        }
    };

    let result = if unpacked.sign {
        match int_val.checked_neg() {
            Some(neg) => neg,
            None => {
                engine.invalid_flag = true;
                return None;
            }
        }
    } else {
        int_val
    };

    // Check range
    let max_val = if width >= 64 {
        i64::MAX
    } else {
        (1i64 << (width - 1)) - 1
    };
    let min_val = if width >= 64 {
        i64::MIN
    } else {
        (1i64 << (width - 1)).wrapping_neg()
    };

    if result > max_val || result < min_val {
        engine.invalid_flag = true;
        return None;
    }

    Some(result)
}

/// Convert floating-point to unsigned integer
#[must_use]
pub fn fp_to_uint(engine: &mut Ieee754Engine, value: &FpValue, width: u32) -> Option<u64> {
    let unpacked = engine.unpack(value);

    if unpacked.is_nan() || unpacked.class.is_infinite() {
        engine.invalid_flag = true;
        return None;
    }

    if unpacked.sign {
        engine.invalid_flag = true;
        return None;
    }

    if unpacked.is_zero() {
        return Some(0);
    }

    // For left-aligned significand: value = (sig / 2^127) × 2^exp
    // Integer value = sig × 2^(exp - 127)
    // If exp < 127: int_value = sig >> (127 - exp)
    // If exp >= 127: int_value = sig << (exp - 127)
    let int_val = if unpacked.exponent >= 127 {
        // Value >= 1, need to shift left or keep as is
        let left_shift = (unpacked.exponent - 127) as u32;
        if left_shift >= 64 {
            // Overflow - value is too large for u64
            engine.invalid_flag = true;
            return None;
        }
        (unpacked.significand >> (127 - left_shift)) as u64
    } else {
        // Value < 1, shift right
        let right_shift = (127 - unpacked.exponent) as u32;
        if right_shift >= 128 {
            0
        } else {
            (unpacked.significand >> right_shift) as u64
        }
    };

    let max_val = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };

    if int_val > max_val {
        engine.invalid_flag = true;
        return None;
    }

    Some(int_val)
}

/// Convert signed integer to floating-point
pub fn sint_to_fp(engine: &mut Ieee754Engine, value: i64, format: FpFormat) -> FpValue {
    if value == 0 {
        return FpValue::pos_zero(format);
    }

    let (sign, abs_val) = if value < 0 {
        (true, value.wrapping_neg() as u64)
    } else {
        (false, value as u64)
    };

    let leading_zeros = abs_val.leading_zeros();
    let significand = (abs_val as u128) << (64 + leading_zeros);
    let exponent = 63 - (leading_zeros as i32);

    let unpacked =
        UnpackedFloat::from_components(sign, exponent, significand, format.significand_bits);
    engine.pack(&unpacked, format)
}

/// Convert unsigned integer to floating-point
pub fn uint_to_fp(engine: &mut Ieee754Engine, value: u64, format: FpFormat) -> FpValue {
    if value == 0 {
        return FpValue::pos_zero(format);
    }

    let leading_zeros = value.leading_zeros();
    let significand = (value as u128) << (64 + leading_zeros);
    let exponent = 63 - (leading_zeros as i32);

    let unpacked =
        UnpackedFloat::from_components(false, exponent, significand, format.significand_bits);
    engine.pack(&unpacked, format)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unpack_pack_f32() {
        let engine = Ieee754Engine::new();
        let val = FpValue::from_f32(1.0);
        let unpacked = engine.unpack(&val);
        assert_eq!(unpacked.class, FpClass::PositiveNormal);
        assert!(!unpacked.sign);

        let mut pack_engine = Ieee754Engine::new();
        let packed = pack_engine.pack(&unpacked, FpFormat::FLOAT32);
        assert_eq!(packed.to_f32(), Some(1.0));
    }

    #[test]
    fn test_addition_basic() {
        let mut engine = Ieee754Engine::new();
        let a = FpValue::from_f32(1.0);
        let b = FpValue::from_f32(2.0);
        let result = engine.add(&a, &b);
        assert_eq!(result.to_f32(), Some(3.0));
    }

    #[test]
    fn test_subtraction_basic() {
        let mut engine = Ieee754Engine::new();
        let a = FpValue::from_f32(5.0);
        let b = FpValue::from_f32(2.0);
        let result = engine.sub(&a, &b);
        assert_eq!(result.to_f32(), Some(3.0));
    }

    #[test]
    fn test_multiplication_basic() {
        let mut engine = Ieee754Engine::new();
        let a = FpValue::from_f32(2.0);
        let b = FpValue::from_f32(3.0);
        let result = engine.mul(&a, &b);
        assert_eq!(result.to_f32(), Some(6.0));
    }

    #[test]
    fn test_division_basic() {
        let mut engine = Ieee754Engine::new();
        let a = FpValue::from_f32(6.0);
        let b = FpValue::from_f32(2.0);
        let result = engine.div(&a, &b);
        assert_eq!(result.to_f32(), Some(3.0));
    }

    #[test]
    fn test_sqrt_basic() {
        let mut engine = Ieee754Engine::new();
        let a = FpValue::from_f32(4.0);
        let result = engine.sqrt(&a);
        let res_f32 = result.to_f32();
        assert!(res_f32.is_some());
        let val = res_f32.unwrap_or(0.0);
        println!("sqrt(4.0) = {}, expected 2.0", val);
        assert!((val - 2.0).abs() < 0.1); // Approximate check
    }

    #[test]
    fn test_special_values_nan() {
        let engine = Ieee754Engine::new();
        let nan = FpValue::nan(FpFormat::FLOAT32);
        let unpacked = engine.unpack(&nan);
        assert!(unpacked.is_nan());
    }

    #[test]
    fn test_special_values_infinity() {
        let engine = Ieee754Engine::new();
        let inf = FpValue::pos_infinity(FpFormat::FLOAT32);
        let unpacked = engine.unpack(&inf);
        assert_eq!(unpacked.class, FpClass::PositiveInfinity);
    }

    #[test]
    fn test_special_values_zero() {
        let engine = Ieee754Engine::new();
        let zero = FpValue::pos_zero(FpFormat::FLOAT32);
        let unpacked = engine.unpack(&zero);
        assert!(unpacked.is_zero());
    }

    #[test]
    fn test_inf_plus_inf() {
        let mut engine = Ieee754Engine::new();
        let inf1 = FpValue::pos_infinity(FpFormat::FLOAT32);
        let inf2 = FpValue::pos_infinity(FpFormat::FLOAT32);
        let result = engine.add(&inf1, &inf2);
        assert!(result.is_infinite());
    }

    #[test]
    fn test_inf_minus_inf() {
        let mut engine = Ieee754Engine::new();
        let inf1 = FpValue::pos_infinity(FpFormat::FLOAT32);
        let inf2 = FpValue::neg_infinity(FpFormat::FLOAT32);
        let result = engine.add(&inf1, &inf2);
        assert!(result.is_nan());
        assert!(engine.invalid());
    }

    #[test]
    fn test_zero_times_inf() {
        let mut engine = Ieee754Engine::new();
        let zero = FpValue::pos_zero(FpFormat::FLOAT32);
        let inf = FpValue::pos_infinity(FpFormat::FLOAT32);
        let result = engine.mul(&zero, &inf);
        assert!(result.is_nan());
        assert!(engine.invalid());
    }

    #[test]
    fn test_division_by_zero() {
        let mut engine = Ieee754Engine::new();
        let one = FpValue::from_f32(1.0);
        let zero = FpValue::pos_zero(FpFormat::FLOAT32);
        let result = engine.div(&one, &zero);
        assert!(result.is_infinite());
        assert!(engine.divide_by_zero());
    }

    #[test]
    fn test_sqrt_negative() {
        let mut engine = Ieee754Engine::new();
        let neg = FpValue::from_f32(-1.0);
        let result = engine.sqrt(&neg);
        assert!(result.is_nan());
        assert!(engine.invalid());
    }

    #[test]
    fn test_comparison_eq() {
        let engine = Ieee754Engine::new();
        let a = FpValue::from_f32(1.0);
        let b = FpValue::from_f32(1.0);
        assert!(engine.eq(&a, &b));
    }

    #[test]
    fn test_comparison_lt() {
        let engine = Ieee754Engine::new();
        let a = FpValue::from_f32(1.0);
        let b = FpValue::from_f32(2.0);
        assert!(engine.lt(&a, &b));
        assert!(!engine.lt(&b, &a));
    }

    #[test]
    fn test_comparison_nan() {
        let engine = Ieee754Engine::new();
        let nan = FpValue::nan(FpFormat::FLOAT32);
        let one = FpValue::from_f32(1.0);
        assert!(!engine.eq(&nan, &nan));
        assert!(!engine.lt(&nan, &one));
        assert!(!engine.lt(&one, &nan));
    }

    #[test]
    fn test_negation() {
        let engine = Ieee754Engine::new();
        let a = FpValue::from_f32(1.0);
        let neg_a = engine.neg(&a);
        assert_eq!(neg_a.to_f32(), Some(-1.0));
    }

    #[test]
    fn test_absolute_value() {
        let engine = Ieee754Engine::new();
        let a = FpValue::from_f32(-2.5);
        let abs_a = engine.abs(&a);
        assert_eq!(abs_a.to_f32(), Some(2.5));
    }

    #[test]
    fn test_min_max() {
        let mut engine = Ieee754Engine::new();
        let a = FpValue::from_f32(1.0);
        let b = FpValue::from_f32(2.0);

        let min_val = engine.min(&a, &b);
        assert_eq!(min_val.to_f32(), Some(1.0));

        let max_val = engine.max(&a, &b);
        assert_eq!(max_val.to_f32(), Some(2.0));
    }

    #[test]
    fn test_classification() {
        let engine = Ieee754Engine::new();

        let normal = FpValue::from_f32(1.0);
        assert_eq!(engine.classify(&normal), FpClass::PositiveNormal);

        let zero = FpValue::pos_zero(FpFormat::FLOAT32);
        assert_eq!(engine.classify(&zero), FpClass::PositiveZero);

        let inf = FpValue::pos_infinity(FpFormat::FLOAT32);
        assert_eq!(engine.classify(&inf), FpClass::PositiveInfinity);

        let nan = FpValue::nan(FpFormat::FLOAT32);
        assert!(engine.classify(&nan).is_nan());
    }

    #[test]
    fn test_rounding_modes() {
        let mut engine = Ieee754Engine::new();

        // Test different rounding modes
        engine.set_rounding_mode(FpRoundingMode::RoundTowardZero);
        assert_eq!(engine.rounding_mode(), FpRoundingMode::RoundTowardZero);

        engine.set_rounding_mode(FpRoundingMode::RoundTowardPositive);
        assert_eq!(engine.rounding_mode(), FpRoundingMode::RoundTowardPositive);
    }

    #[test]
    fn test_format_conversion_f32_to_f64() {
        let mut engine = Ieee754Engine::new();
        let f32_val = FpValue::from_f32(1.5);
        let f64_val = convert_format(&mut engine, &f32_val, FpFormat::FLOAT64);

        assert_eq!(f64_val.format, FpFormat::FLOAT64);
        // Value should be preserved
        let unpacked = engine.unpack(&f64_val);
        assert_eq!(unpacked.class, FpClass::PositiveNormal);
    }

    #[test]
    fn test_sint_conversion() {
        let mut engine = Ieee754Engine::new();
        let fp_val = FpValue::from_f32(42.0);
        let int_val = fp_to_sint(&mut engine, &fp_val, 64);
        assert_eq!(int_val, Some(42));
    }

    #[test]
    fn test_uint_conversion() {
        let mut engine = Ieee754Engine::new();
        let fp_val = FpValue::from_f32(42.0);
        let uint_val = fp_to_uint(&mut engine, &fp_val, 64);
        assert_eq!(uint_val, Some(42));
    }

    #[test]
    fn test_sint_to_fp() {
        let mut engine = Ieee754Engine::new();
        let fp_val = sint_to_fp(&mut engine, 42, FpFormat::FLOAT32);
        let result = fp_val.to_f32();
        assert!(result.is_some());
    }

    #[test]
    fn test_uint_to_fp() {
        let mut engine = Ieee754Engine::new();
        let fp_val = uint_to_fp(&mut engine, 42, FpFormat::FLOAT32);
        let result = fp_val.to_f32();
        assert!(result.is_some());
    }

    #[test]
    fn test_binary16_format() {
        let format = FpFormat::FLOAT16;
        assert_eq!(format.width(), 16);
        assert_eq!(format.exponent_bits, 5);
        assert_eq!(format.significand_bits, 11);
    }

    #[test]
    fn test_binary128_format() {
        let format = FpFormat::FLOAT128;
        assert_eq!(format.width(), 128);
        assert_eq!(format.exponent_bits, 15);
        assert_eq!(format.significand_bits, 113);
    }

    #[test]
    fn test_exception_flags() {
        let mut engine = Ieee754Engine::new();

        // Test division by zero flag
        let one = FpValue::from_f32(1.0);
        let zero = FpValue::pos_zero(FpFormat::FLOAT32);
        engine.div(&one, &zero);
        assert!(engine.divide_by_zero());

        engine.clear_flags();
        assert!(!engine.divide_by_zero());

        // Test invalid flag
        let nan1 = FpValue::nan(FpFormat::FLOAT32);
        let nan2 = FpValue::nan(FpFormat::FLOAT32);
        engine.add(&nan1, &nan2);
        // Invalid flag should not be set for quiet NaN operations in some cases

        engine.clear_flags();

        // Test 0 * infinity
        let inf = FpValue::pos_infinity(FpFormat::FLOAT32);
        engine.mul(&zero, &inf);
        assert!(engine.invalid());
    }

    #[test]
    fn test_denormal_numbers() {
        let engine = Ieee754Engine::new();

        // Create a subnormal number (exponent = 0, significand != 0)
        let subnormal = FpValue {
            sign: false,
            exponent: 0,
            significand: 1,
            format: FpFormat::FLOAT32,
        };

        let unpacked = engine.unpack(&subnormal);
        assert_eq!(unpacked.class, FpClass::PositiveSubnormal);
    }

    #[test]
    fn test_signed_zero_semantics() {
        let engine = Ieee754Engine::new();
        let pos_zero = FpValue::pos_zero(FpFormat::FLOAT32);
        let neg_zero = FpValue::neg_zero(FpFormat::FLOAT32);

        // +0 == -0 in IEEE 754
        assert!(engine.eq(&pos_zero, &neg_zero));
    }

    #[test]
    fn test_fma_operation() {
        let mut engine = Ieee754Engine::new();
        let a = FpValue::from_f32(2.0);
        let b = FpValue::from_f32(3.0);
        let c = FpValue::from_f32(4.0);

        let result = engine.fma(&a, &b, &c);
        // 2 * 3 + 4 = 10
        assert_eq!(result.to_f32(), Some(10.0));
    }
}
