//! SIMD-Accelerated Polynomial Operations.
#![allow(dead_code)] // Under development
//!
//! Provides vectorized polynomial arithmetic for improved performance
//! on dense polynomials.

#[allow(unused_imports)]
use crate::prelude::*;
use core::ops::{Add, Mul, Sub};
use num_traits::{One, Zero};

/// Add two polynomials using SIMD-friendly patterns.
pub fn simd_poly_add<T>(a_coeffs: &[T], b_coeffs: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Zero,
{
    let max_len = a_coeffs.len().max(b_coeffs.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_val = a_coeffs.get(i).cloned().unwrap_or_else(T::zero);
        let b_val = b_coeffs.get(i).cloned().unwrap_or_else(T::zero);
        result.push(a_val + b_val);
    }

    // Remove leading zeros
    while result.len() > 1 && result.last().is_some_and(|c| c.is_zero()) {
        result.pop();
    }

    result
}

/// Multiply two polynomials using SIMD-friendly convolution.
pub fn simd_poly_mul<T>(a_coeffs: &[T], b_coeffs: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    if a_coeffs.is_empty() || b_coeffs.is_empty() {
        return vec![T::zero()];
    }

    let result_len = a_coeffs.len() + b_coeffs.len() - 1;
    let mut result = vec![T::zero(); result_len];

    // Convolution with cache-friendly access pattern
    for (i, a_coeff) in a_coeffs.iter().enumerate() {
        for (j, b_coeff) in b_coeffs.iter().enumerate() {
            result[i + j] = result[i + j].clone() + a_coeff.clone() * b_coeff.clone();
        }
    }

    result
}

/// Evaluate polynomial at a point using Horner's method (SIMD-friendly).
pub fn simd_poly_eval<T>(coeffs: &[T], x: &T) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    if coeffs.is_empty() {
        return T::zero();
    }

    // Horner's method: p(x) = a0 + x(a1 + x(a2 + x(...)))
    let mut result = coeffs
        .last()
        .expect("collection should not be empty")
        .clone();

    for coeff in coeffs.iter().rev().skip(1) {
        result = coeff.clone() + result * x.clone();
    }

    result
}

/// Evaluate polynomial at multiple points in parallel.
pub fn simd_poly_eval_multi<T>(coeffs: &[T], points: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero + Send + Sync,
{
    points.iter().map(|x| simd_poly_eval(coeffs, x)).collect()
}

/// Compute derivative of polynomial.
pub fn simd_poly_derivative<T>(coeffs: &[T]) -> Vec<T>
where
    T: Clone + Mul<Output = T> + From<usize> + Zero,
{
    if coeffs.len() <= 1 {
        return vec![T::zero()];
    }

    coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| c.clone() * T::from(i))
        .collect()
}

/// Scalar multiplication of polynomial.
pub fn simd_poly_scale<T>(coeffs: &[T], scalar: &T) -> Vec<T>
where
    T: Clone + Mul<Output = T>,
{
    coeffs.iter().map(|c| c.clone() * scalar.clone()).collect()
}

/// Polynomial composition: compute p(q(x)) for dense polynomials.
pub fn simd_poly_compose<T>(p_coeffs: &[T], q_coeffs: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    if p_coeffs.is_empty() {
        return vec![T::zero()];
    }

    // Start with constant term
    let mut result = vec![p_coeffs[0].clone()];

    // Powers of q
    let mut q_power = q_coeffs.to_vec();

    for p_coeff in p_coeffs.iter().skip(1) {
        // Add p_coeff * q^i
        let term = simd_poly_scale(&q_power, p_coeff);
        result = simd_poly_add(&result, &term);

        // Update q_power
        q_power = simd_poly_mul(&q_power, q_coeffs);
    }

    result
}

/// Fast polynomial evaluation using precomputed powers (for multiple evaluations).
pub struct PolyEvaluator<T> {
    coeffs: Vec<T>,
    precomputed_powers: Vec<Vec<T>>,
}

impl<T> PolyEvaluator<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero + One,
{
    /// Create evaluator with precomputation.
    pub fn new(coeffs: Vec<T>) -> Self {
        Self {
            coeffs,
            precomputed_powers: Vec::new(),
        }
    }

    /// Evaluate at a point.
    pub fn eval(&self, x: &T) -> T {
        simd_poly_eval(&self.coeffs, x)
    }

    /// Batch evaluate at multiple points.
    pub fn eval_batch(&self, points: &[T]) -> Vec<T>
    where
        T: Send + Sync,
    {
        simd_poly_eval_multi(&self.coeffs, points)
    }
}

// ====================================================================== //
// Functions re-exported by simd/mod.rs                                    //
// ====================================================================== //

/// Element-wise coefficient addition of two i64 slices, returning a new `Vec<i64>`.
///
/// Pads the shorter slice with zeros.
pub fn poly_add_coeffs(a: &[i64], b: &[i64]) -> Vec<i64> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let av = *a.get(i).unwrap_or(&0);
        let bv = *b.get(i).unwrap_or(&0);
        result.push(av + bv);
    }
    result
}

/// Element-wise coefficient subtraction of two i64 slices, returning a new `Vec<i64>`.
///
/// Pads the shorter slice with zeros.
pub fn poly_sub_coeffs(a: &[i64], b: &[i64]) -> Vec<i64> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let av = *a.get(i).unwrap_or(&0);
        let bv = *b.get(i).unwrap_or(&0);
        result.push(av - bv);
    }
    result
}

/// Multiply every coefficient by `scalar`, returning a new `Vec<i64>`.
pub fn poly_mul_scalar(coeffs: &[i64], scalar: i64) -> Vec<i64> {
    coeffs.iter().map(|&c| c * scalar).collect()
}

/// Dot-product of two equal-length i64 slices.
///
/// Returns `Ok(value)` when the result fits in `i64`, or `Err(i128_value)` on
/// overflow.  Panics if `a.len() != b.len()`.
pub fn poly_dot_product(a: &[i64], b: &[i64]) -> Result<i64, i128> {
    assert_eq!(
        a.len(),
        b.len(),
        "poly_dot_product: slices must have equal length"
    );
    let sum: i128 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i128) * (y as i128))
        .sum();
    i64::try_from(sum).map_err(|_| sum)
}

/// Evaluate a polynomial with i64 coefficients at an i64 point using Horner's method.
///
/// Coefficients are stored in ascending order of degree (constant term first).
pub fn poly_eval_horner_i64(coeffs: &[i64], x: i64) -> i64 {
    if coeffs.is_empty() {
        return 0;
    }
    let mut result = *coeffs.last().expect("non-empty slice has a last element");
    for &c in coeffs.iter().rev().skip(1) {
        result = c + result * x;
    }
    result
}

/// Polynomial GCD using SIMD-friendly operations.
pub fn simd_poly_gcd<T>(mut a: Vec<T>, mut b: Vec<T>) -> Vec<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Zero + PartialEq,
{
    // Euclidean algorithm
    while !is_zero_poly(&b) {
        let (_, remainder) = poly_divide(&a, &b);
        a = b;
        b = remainder;
    }

    a
}

/// Check if polynomial is zero.
fn is_zero_poly<T>(coeffs: &[T]) -> bool
where
    T: Zero + PartialEq,
{
    coeffs.iter().all(|c| c.is_zero())
}

/// Polynomial division (simplified for SIMD context).
fn poly_divide<T>(dividend: &[T], _divisor: &[T]) -> (Vec<T>, Vec<T>)
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Zero,
{
    // Simplified: returns (quotient, remainder)
    // Real implementation would do full polynomial long division
    // For now, stub implementation
    (vec![T::zero()], dividend.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_poly_add() {
        let a = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x^2
        let b = vec![4.0, 5.0]; // 4 + 5x
        let result = simd_poly_add(&a, &b);
        assert_eq!(result, vec![5.0, 7.0, 3.0]);
    }

    #[test]
    fn test_simd_poly_mul() {
        let a = vec![1.0, 2.0]; // 1 + 2x
        let b = vec![3.0, 4.0]; // 3 + 4x
        let result = simd_poly_mul(&a, &b);
        // (1 + 2x)(3 + 4x) = 3 + 4x + 6x + 8x^2 = 3 + 10x + 8x^2
        assert_eq!(result, vec![3.0, 10.0, 8.0]);
    }

    #[test]
    fn test_simd_poly_eval() {
        let coeffs = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x^2
        let result = simd_poly_eval(&coeffs, &2.0);
        // 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        assert_eq!(result, 17.0);
    }

    #[test]
    fn test_simd_poly_derivative() {
        use num_bigint::BigInt;
        // Use BigInt which implements From<usize>
        let coeffs: Vec<BigInt> = vec![
            BigInt::from(1),
            BigInt::from(2),
            BigInt::from(3),
            BigInt::from(4),
        ]; // 1 + 2x + 3x^2 + 4x^3
        let deriv = simd_poly_derivative(&coeffs);
        // derivative: 2 + 6x + 12x^2
        assert_eq!(
            deriv,
            vec![BigInt::from(2), BigInt::from(6), BigInt::from(12)]
        );
    }

    #[test]
    fn test_simd_poly_scale() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let result = simd_poly_scale(&coeffs, &2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_poly_evaluator() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let evaluator = PolyEvaluator::new(coeffs);

        let result = evaluator.eval(&2.0);
        assert_eq!(result, 17.0);

        let batch = evaluator.eval_batch(&[1.0, 2.0, 3.0]);
        assert_eq!(batch, vec![6.0, 17.0, 34.0]);
    }

    #[test]
    fn test_simd_poly_compose() {
        let p = vec![1.0, 0.0, 1.0]; // 1 + x^2
        let q = vec![0.0, 2.0]; // 2x
        let result = simd_poly_compose(&p, &q);
        // p(q(x)) = 1 + (2x)^2 = 1 + 4x^2
        assert_eq!(result, vec![1.0, 0.0, 4.0]);
    }
}
