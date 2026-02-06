//! Comprehensive tests for IEEE 754 floating-point implementation
//!
//! These tests verify the correctness of the complete FP implementation including:
//! - IEEE 754-2019 arithmetic operations
//! - All rounding modes
//! - Interval arithmetic
//! - Special value handling

use oxiz_theories::fp::ieee754_full::{FpClass, Ieee754Engine};
use oxiz_theories::fp::interval_arithmetic::{
    ComparisonOp, ErrorBounds, FpInterval, IntervalEngine,
};
use oxiz_theories::fp::{FpFormat, FpRoundingMode, FpValue};

#[test]
fn test_ieee754_basic_operations() {
    let mut engine = Ieee754Engine::new();

    // Test addition
    let a = FpValue::from_f32(1.5);
    let b = FpValue::from_f32(2.5);
    let result = engine.add(&a, &b);
    assert_eq!(result.to_f32(), Some(4.0));

    // Test multiplication
    let c = FpValue::from_f32(3.0);
    let d = FpValue::from_f32(4.0);
    let result = engine.mul(&c, &d);
    assert_eq!(result.to_f32(), Some(12.0));

    // Test division
    let e = FpValue::from_f32(10.0);
    let f = FpValue::from_f32(2.0);
    let result = engine.div(&e, &f);
    assert_eq!(result.to_f32(), Some(5.0));
}

#[test]
fn test_rounding_modes() {
    let mut engine = Ieee754Engine::new();

    // Test RNE (round to nearest, ties to even)
    engine.set_rounding_mode(FpRoundingMode::RoundNearestTiesToEven);
    assert_eq!(
        engine.rounding_mode(),
        FpRoundingMode::RoundNearestTiesToEven
    );

    // Test RTZ (round toward zero)
    engine.set_rounding_mode(FpRoundingMode::RoundTowardZero);
    assert_eq!(engine.rounding_mode(), FpRoundingMode::RoundTowardZero);

    // Test RTP (round toward positive)
    engine.set_rounding_mode(FpRoundingMode::RoundTowardPositive);
    assert_eq!(engine.rounding_mode(), FpRoundingMode::RoundTowardPositive);
}

#[test]
fn test_special_values() {
    let engine = Ieee754Engine::new();

    // Test NaN
    let nan = FpValue::nan(FpFormat::FLOAT32);
    assert!(nan.is_nan());
    let unpacked = engine.unpack(&nan);
    assert!(unpacked.class.is_nan());

    // Test infinity
    let pos_inf = FpValue::pos_infinity(FpFormat::FLOAT32);
    let neg_inf = FpValue::neg_infinity(FpFormat::FLOAT32);
    assert!(pos_inf.is_infinite());
    assert!(neg_inf.is_infinite());

    // Test zero
    let pos_zero = FpValue::pos_zero(FpFormat::FLOAT32);
    let neg_zero = FpValue::neg_zero(FpFormat::FLOAT32);
    assert!(pos_zero.is_zero());
    assert!(neg_zero.is_zero());

    // IEEE 754: +0 == -0
    assert!(engine.eq(&pos_zero, &neg_zero));
}

#[test]
fn test_classification() {
    let engine = Ieee754Engine::new();

    let normal = FpValue::from_f32(42.0);
    assert_eq!(engine.classify(&normal), FpClass::PositiveNormal);

    let zero = FpValue::pos_zero(FpFormat::FLOAT32);
    assert_eq!(engine.classify(&zero), FpClass::PositiveZero);

    let inf = FpValue::pos_infinity(FpFormat::FLOAT32);
    assert_eq!(engine.classify(&inf), FpClass::PositiveInfinity);
}

#[test]
fn test_interval_basic_operations() {
    let mut engine = IntervalEngine::new();

    // Test interval addition
    let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(2.0));
    let b = FpInterval::new(FpValue::from_f32(3.0), FpValue::from_f32(4.0));
    let result = engine.add(&a, &b);

    // [1,2] + [3,4] should contain [4,6]
    assert!(result.contains(&FpValue::from_f32(4.0)));
    assert!(result.contains(&FpValue::from_f32(6.0)));
    assert!(result.contains(&FpValue::from_f32(5.0)));

    // Test interval multiplication
    let c = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(3.0));
    let d = FpInterval::new(FpValue::from_f32(4.0), FpValue::from_f32(5.0));
    let result = engine.mul(&c, &d);

    // [2,3] * [4,5] should contain [8,15]
    assert!(result.contains(&FpValue::from_f32(8.0)));
    assert!(result.contains(&FpValue::from_f32(15.0)));
}

#[test]
fn test_interval_properties() {
    let interval = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(5.0));

    // Test containment
    assert!(interval.contains(&FpValue::from_f32(3.0)));
    assert!(!interval.contains(&FpValue::from_f32(6.0)));

    // Test width
    let width = interval.width();
    let w = width.to_f32().unwrap_or(0.0);
    assert!((w - 4.0).abs() < 0.01);

    // Test midpoint
    let mid = interval.midpoint();
    assert_eq!(mid.to_f32(), Some(3.0));
}

#[test]
fn test_interval_intersection() {
    let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(5.0));
    let b = FpInterval::new(FpValue::from_f32(3.0), FpValue::from_f32(7.0));

    let intersection = a.intersection(&b);

    // [1,5] ∩ [3,7] = [3,5]
    assert!(intersection.contains(&FpValue::from_f32(3.0)));
    assert!(intersection.contains(&FpValue::from_f32(5.0)));
    assert!(intersection.contains(&FpValue::from_f32(4.0)));
    assert!(!intersection.contains(&FpValue::from_f32(2.0)));
    assert!(!intersection.contains(&FpValue::from_f32(6.0)));
}

#[test]
fn test_interval_hull() {
    let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(3.0));
    let b = FpInterval::new(FpValue::from_f32(5.0), FpValue::from_f32(7.0));

    let hull = a.hull(&b);

    // [1,3] ∪ [5,7] = [1,7]
    assert!(hull.contains(&FpValue::from_f32(1.0)));
    assert!(hull.contains(&FpValue::from_f32(7.0)));
    assert!(hull.contains(&FpValue::from_f32(4.0))); // Gap is included
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
fn test_all_formats() {
    // Test binary16
    let format16 = FpFormat::FLOAT16;
    assert_eq!(format16.width(), 16);
    assert_eq!(format16.exponent_bits, 5);
    assert_eq!(format16.significand_bits, 11);

    // Test binary32
    let format32 = FpFormat::FLOAT32;
    assert_eq!(format32.width(), 32);
    assert_eq!(format32.exponent_bits, 8);
    assert_eq!(format32.significand_bits, 24);

    // Test binary64
    let format64 = FpFormat::FLOAT64;
    assert_eq!(format64.width(), 64);
    assert_eq!(format64.exponent_bits, 11);
    assert_eq!(format64.significand_bits, 53);

    // Test binary128
    let format128 = FpFormat::FLOAT128;
    assert_eq!(format128.width(), 128);
    assert_eq!(format128.exponent_bits, 15);
    assert_eq!(format128.significand_bits, 113);
}

#[test]
fn test_exception_flags() {
    let mut engine = Ieee754Engine::new();
    engine.clear_flags();

    // Test division by zero
    let one = FpValue::from_f32(1.0);
    let zero = FpValue::pos_zero(FpFormat::FLOAT32);
    engine.div(&one, &zero);
    assert!(engine.divide_by_zero());

    engine.clear_flags();
    assert!(!engine.divide_by_zero());

    // Test invalid operation (0 * infinity)
    let inf = FpValue::pos_infinity(FpFormat::FLOAT32);
    engine.mul(&zero, &inf);
    assert!(engine.invalid());
}

#[test]
fn test_comparison_operations() {
    let engine = Ieee754Engine::new();

    let a = FpValue::from_f32(1.0);
    let b = FpValue::from_f32(2.0);
    let c = FpValue::from_f32(1.0);

    // Test equality
    assert!(engine.eq(&a, &c));
    assert!(!engine.eq(&a, &b));

    // Test less than
    assert!(engine.lt(&a, &b));
    assert!(!engine.lt(&b, &a));

    // Test less than or equal
    assert!(engine.le(&a, &b));
    assert!(engine.le(&a, &c));

    // Test greater than
    assert!(engine.gt(&b, &a));
    assert!(!engine.gt(&a, &b));

    // Test greater than or equal
    assert!(engine.ge(&b, &a));
    assert!(engine.ge(&a, &c));
}

#[test]
fn test_nan_semantics() {
    let engine = Ieee754Engine::new();

    let nan = FpValue::nan(FpFormat::FLOAT32);
    let one = FpValue::from_f32(1.0);

    // NaN != NaN
    assert!(!engine.eq(&nan, &nan));

    // NaN comparisons are all false
    assert!(!engine.lt(&nan, &one));
    assert!(!engine.lt(&one, &nan));
    assert!(!engine.le(&nan, &one));
    assert!(!engine.gt(&nan, &one));
    assert!(!engine.ge(&nan, &one));
}

#[test]
fn test_infinity_arithmetic() {
    let mut engine = Ieee754Engine::new();

    let pos_inf = FpValue::pos_infinity(FpFormat::FLOAT32);
    let _neg_inf = FpValue::neg_infinity(FpFormat::FLOAT32);
    let one = FpValue::from_f32(1.0);

    // inf + 1 = inf
    let result = engine.add(&pos_inf, &one);
    assert!(result.is_infinite());
    assert!(!result.sign);

    // inf - inf = NaN
    let result = engine.sub(&pos_inf, &pos_inf);
    assert!(result.is_nan());
    assert!(engine.invalid());

    engine.clear_flags();

    // inf * -1 = -inf
    let neg_one = FpValue::from_f32(-1.0);
    let result = engine.mul(&pos_inf, &neg_one);
    assert!(result.is_infinite());
    assert!(result.sign);

    // 1 / inf = 0
    let result = engine.div(&one, &pos_inf);
    assert!(result.is_zero());
}

#[test]
fn test_denormal_numbers() {
    let engine = Ieee754Engine::new();

    // Create a denormal number
    let denormal = FpValue {
        sign: false,
        exponent: 0,
        significand: 1,
        format: FpFormat::FLOAT32,
    };

    let unpacked = engine.unpack(&denormal);
    assert_eq!(unpacked.class, FpClass::PositiveSubnormal);
}

#[test]
fn test_min_max_operations() {
    let mut engine = Ieee754Engine::new();

    let a = FpValue::from_f32(1.0);
    let b = FpValue::from_f32(2.0);

    let min_val = engine.min(&a, &b);
    assert_eq!(min_val.to_f32(), Some(1.0));

    let max_val = engine.max(&a, &b);
    assert_eq!(max_val.to_f32(), Some(2.0));

    // Test with NaN
    let nan = FpValue::nan(FpFormat::FLOAT32);
    let min_nan = engine.min(&nan, &a);
    assert!(min_nan.is_nan());

    let max_nan = engine.max(&nan, &a);
    assert!(max_nan.is_nan());
}

#[test]
fn test_fma_operation() {
    let mut engine = Ieee754Engine::new();

    let a = FpValue::from_f32(2.0);
    let b = FpValue::from_f32(3.0);
    let c = FpValue::from_f32(4.0);

    // FMA: (a * b) + c = (2 * 3) + 4 = 10
    let result = engine.fma(&a, &b, &c);
    assert_eq!(result.to_f32(), Some(10.0));
}

#[test]
fn test_sqrt_operation() {
    let mut engine = Ieee754Engine::new();

    let four = FpValue::from_f32(4.0);
    let result = engine.sqrt(&four);

    // sqrt(4) = 2
    let res = result.to_f32().unwrap_or(0.0);
    assert!((res - 2.0).abs() < 0.1);

    // sqrt(-1) = NaN
    engine.clear_flags();
    let neg_one = FpValue::from_f32(-1.0);
    let result = engine.sqrt(&neg_one);
    assert!(result.is_nan());
    assert!(engine.invalid());
}

#[test]
fn test_abs_neg_operations() {
    let engine = Ieee754Engine::new();

    let val = FpValue::from_f32(-5.0);

    // Test negation
    let neg = engine.neg(&val);
    assert_eq!(neg.to_f32(), Some(5.0));

    // Test absolute value
    let abs_val = engine.abs(&val);
    assert_eq!(abs_val.to_f32(), Some(5.0));

    // Test abs of positive
    let pos = FpValue::from_f32(3.0);
    let abs_pos = engine.abs(&pos);
    assert_eq!(abs_pos.to_f32(), Some(3.0));
}

#[test]
fn test_interval_power() {
    let mut engine = IntervalEngine::new();

    // Test even power
    let a = FpInterval::new(FpValue::from_f32(-2.0), FpValue::from_f32(3.0));
    let result = engine.powi(&a, 2);

    // [-2,3]^2 should contain [0,9]
    assert!(result.contains(&FpValue::from_f32(0.0)));
    assert!(result.contains(&FpValue::from_f32(9.0)));

    // Test odd power
    let b = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(2.0));
    let result = engine.powi(&b, 3);

    // [1,2]^3 should contain [1,8]
    assert!(result.contains(&FpValue::from_f32(1.0)));
    assert!(result.contains(&FpValue::from_f32(8.0)));
}

#[test]
fn test_interval_refinement() {
    let mut engine = IntervalEngine::new();

    let a = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(10.0));
    let b = FpInterval::new(FpValue::from_f32(5.0), FpValue::from_f32(15.0));

    // Refine with a < b constraint
    let (refined_a, refined_b) = engine.refine_comparison(&a, &b, ComparisonOp::Lt);

    // After refinement, upper bound of a should be <= upper bound of b
    assert!(
        refined_a.upper.to_f32().unwrap_or(0.0) <= refined_b.upper.to_f32().unwrap_or(f32::MAX)
    );
}

#[test]
fn test_interval_subset_and_overlap() {
    let a = FpInterval::new(FpValue::from_f32(2.0), FpValue::from_f32(3.0));
    let b = FpInterval::new(FpValue::from_f32(1.0), FpValue::from_f32(5.0));
    let c = FpInterval::new(FpValue::from_f32(4.0), FpValue::from_f32(6.0));

    // a is subset of b
    assert!(a.is_subset_of(&b));
    assert!(!b.is_subset_of(&a));

    // a and b overlap
    assert!(a.overlaps(&b));

    // a and c don't overlap
    assert!(!a.overlaps(&c));
}

#[test]
fn test_all_rounding_modes_set() {
    let mut engine = Ieee754Engine::new();

    // Test RNE
    engine.set_rounding_mode(FpRoundingMode::RoundNearestTiesToEven);
    assert_eq!(engine.rounding_mode().smtlib_name(), "RNE");

    // Test RNA
    engine.set_rounding_mode(FpRoundingMode::RoundNearestTiesToAway);
    assert_eq!(engine.rounding_mode().smtlib_name(), "RNA");

    // Test RTP
    engine.set_rounding_mode(FpRoundingMode::RoundTowardPositive);
    assert_eq!(engine.rounding_mode().smtlib_name(), "RTP");

    // Test RTN
    engine.set_rounding_mode(FpRoundingMode::RoundTowardNegative);
    assert_eq!(engine.rounding_mode().smtlib_name(), "RTN");

    // Test RTZ
    engine.set_rounding_mode(FpRoundingMode::RoundTowardZero);
    assert_eq!(engine.rounding_mode().smtlib_name(), "RTZ");
}
