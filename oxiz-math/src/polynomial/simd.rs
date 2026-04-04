//! SIMD-friendly polynomial operations.
//!
//! This module provides autovectorization-friendly implementations of
//! polynomial operations using `chunks_exact()` and aligned data patterns
//! that compilers can reliably vectorize on x86_64 (SSE/AVX) and
//! AArch64 (NEON).
//!
//! All operations work on dense coefficient arrays (`&[i64]`) extracted
//! from the main `Polynomial` type, avoiding `BigRational` overhead for
//! the common case where coefficients fit in machine words.

#[allow(unused_imports)]
use crate::prelude::*;

// ====================================================================
// Core SIMD-friendly chunk operations
// ====================================================================

/// Width of the chunk processed per loop iteration.
///
/// 4 x i64 = 256 bits, matching AVX2 register width. On AArch64 with
/// NEON (128-bit), the compiler will typically unroll 2 iterations.
const CHUNK: usize = 4;

/// Element-wise addition of two i64 coefficient arrays.
///
/// The shorter array is logically zero-padded.  The result length equals
/// `max(a.len(), b.len())`.
///
/// This implementation is designed for autovectorization: the hot loop
/// processes elements in chunks of 4 (matching AVX2 register width).
pub fn simd_add_i64(a: &[i64], b: &[i64]) -> Vec<i64> {
    let out_len = a.len().max(b.len());
    let mut result = vec![0i64; out_len];

    let common = a.len().min(b.len());

    // Hot path: process the common prefix in chunks
    let chunks = common / CHUNK;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;
        // Explicit indexing allows the compiler to prove non-aliasing
        result[base] = a[base] + b[base];
        result[base + 1] = a[base + 1] + b[base + 1];
        result[base + 2] = a[base + 2] + b[base + 2];
        result[base + 3] = a[base + 3] + b[base + 3];
    }

    // Scalar tail for the common prefix
    let tail_start = chunks * CHUNK;
    for i in tail_start..common {
        result[i] = a[i] + b[i];
    }

    // Copy the longer tail
    if a.len() > common {
        result[common..out_len].copy_from_slice(&a[common..]);
    } else if b.len() > common {
        result[common..out_len].copy_from_slice(&b[common..]);
    }

    result
}

/// Element-wise subtraction of two i64 coefficient arrays.
pub fn simd_sub_i64(a: &[i64], b: &[i64]) -> Vec<i64> {
    let out_len = a.len().max(b.len());
    let mut result = vec![0i64; out_len];

    let common = a.len().min(b.len());
    let chunks = common / CHUNK;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;
        result[base] = a[base] - b[base];
        result[base + 1] = a[base + 1] - b[base + 1];
        result[base + 2] = a[base + 2] - b[base + 2];
        result[base + 3] = a[base + 3] - b[base + 3];
    }

    let tail_start = chunks * CHUNK;
    for i in tail_start..common {
        result[i] = a[i] - b[i];
    }

    // For subtraction, the tails are different:
    // a is positive, b is negated
    if a.len() > common {
        result[common..out_len].copy_from_slice(&a[common..]);
    } else if b.len() > common {
        for i in common..out_len {
            result[i] = -b[i];
        }
    }

    result
}

/// Scalar multiplication of an i64 coefficient array.
///
/// Multiplies every coefficient by `scalar` in autovectorizable chunks.
pub fn simd_scalar_mul_i64(coeffs: &[i64], scalar: i64) -> Vec<i64> {
    let mut result = vec![0i64; coeffs.len()];

    let chunks = coeffs.len() / CHUNK;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;
        result[base] = coeffs[base] * scalar;
        result[base + 1] = coeffs[base + 1] * scalar;
        result[base + 2] = coeffs[base + 2] * scalar;
        result[base + 3] = coeffs[base + 3] * scalar;
    }

    let tail_start = chunks * CHUNK;
    for i in tail_start..coeffs.len() {
        result[i] = coeffs[i] * scalar;
    }

    result
}

/// Element-wise equality comparison of two i64 coefficient arrays.
///
/// Returns `true` if both arrays represent the same polynomial (treating
/// missing high-degree coefficients as zero).
pub fn simd_coeffs_equal(a: &[i64], b: &[i64]) -> bool {
    let common = a.len().min(b.len());
    let chunks = common / CHUNK;

    // Compare the common prefix in chunks
    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;
        if a[base] != b[base]
            || a[base + 1] != b[base + 1]
            || a[base + 2] != b[base + 2]
            || a[base + 3] != b[base + 3]
        {
            return false;
        }
    }

    let tail_start = chunks * CHUNK;
    for i in tail_start..common {
        if a[i] != b[i] {
            return false;
        }
    }

    // The extra tail of the longer array must be all zeros
    let longer = if a.len() > b.len() { a } else { b };
    for &v in &longer[common..] {
        if v != 0 {
            return false;
        }
    }

    true
}

/// Evaluate a dense polynomial at an i64 point using Horner's method.
///
/// Coefficients are in ascending degree order: `coeffs[i]` is the
/// coefficient of x^i.
pub fn simd_eval_horner(coeffs: &[i64], x: i64) -> i64 {
    if coeffs.is_empty() {
        return 0;
    }

    let mut acc = 0i64;
    for &c in coeffs.iter().rev() {
        acc = acc.wrapping_mul(x).wrapping_add(c);
    }
    acc
}

/// Dot product of two equal-length i64 slices.
///
/// Returns `None` if the slices differ in length.
/// Uses chunk-based accumulation for autovectorization.
pub fn simd_dot_product_i64(a: &[i64], b: &[i64]) -> Option<i128> {
    if a.len() != b.len() {
        return None;
    }

    let mut acc: i128 = 0;
    let chunks = a.len() / CHUNK;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;
        acc += (a[base] as i128) * (b[base] as i128);
        acc += (a[base + 1] as i128) * (b[base + 1] as i128);
        acc += (a[base + 2] as i128) * (b[base + 2] as i128);
        acc += (a[base + 3] as i128) * (b[base + 3] as i128);
    }

    let tail_start = chunks * CHUNK;
    for i in tail_start..a.len() {
        acc += (a[i] as i128) * (b[i] as i128);
    }

    Some(acc)
}

/// Convolution (dense polynomial multiplication) of two i64 coefficient arrays.
///
/// Uses a tiled approach for better cache utilisation on large polynomials.
pub fn simd_convolve_i64(a: &[i64], b: &[i64]) -> Vec<i64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }

    let out_len = a.len() + b.len() - 1;
    let mut result = vec![0i64; out_len];

    // Tile size chosen so that a tile of `b` fits in L1 cache
    const TILE: usize = 64;

    for b_start in (0..b.len()).step_by(TILE) {
        let b_end = (b_start + TILE).min(b.len());
        for (i, &ai) in a.iter().enumerate() {
            let mut j = b_start;
            // Process in chunks of CHUNK within the tile
            let tile_chunks = (b_end - b_start) / CHUNK;
            for _ in 0..tile_chunks {
                result[i + j] += ai * b[j];
                result[i + j + 1] += ai * b[j + 1];
                result[i + j + 2] += ai * b[j + 2];
                result[i + j + 3] += ai * b[j + 3];
                j += CHUNK;
            }
            // Scalar tail within tile
            while j < b_end {
                result[i + j] += ai * b[j];
                j += 1;
            }
        }
    }

    result
}

/// Negate all coefficients using autovectorizable pattern.
pub fn simd_negate_i64(coeffs: &[i64]) -> Vec<i64> {
    let mut result = vec![0i64; coeffs.len()];
    let chunks = coeffs.len() / CHUNK;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;
        result[base] = -coeffs[base];
        result[base + 1] = -coeffs[base + 1];
        result[base + 2] = -coeffs[base + 2];
        result[base + 3] = -coeffs[base + 3];
    }

    let tail_start = chunks * CHUNK;
    for i in tail_start..coeffs.len() {
        result[i] = -coeffs[i];
    }

    result
}

/// Compute the L-infinity norm (maximum absolute coefficient) of a polynomial.
///
/// Useful for coefficient growth estimation in GCD and factorisation.
pub fn simd_linf_norm(coeffs: &[i64]) -> u64 {
    if coeffs.is_empty() {
        return 0;
    }

    let mut max_val: u64 = 0;
    let chunks = coeffs.len() / CHUNK;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK;
        let a0 = (coeffs[base] as i128).unsigned_abs() as u64;
        let a1 = (coeffs[base + 1] as i128).unsigned_abs() as u64;
        let a2 = (coeffs[base + 2] as i128).unsigned_abs() as u64;
        let a3 = (coeffs[base + 3] as i128).unsigned_abs() as u64;
        max_val = max_val.max(a0).max(a1).max(a2).max(a3);
    }

    let tail_start = chunks * CHUNK;
    for &c in &coeffs[tail_start..] {
        max_val = max_val.max((c as i128).unsigned_abs() as u64);
    }

    max_val
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_add_i64_same_len() {
        let a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let b = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let r = simd_add_i64(&a, &b);
        assert_eq!(r, vec![11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_simd_add_i64_different_len() {
        let a = vec![1, 2, 3];
        let b = vec![10, 20, 30, 40, 50];
        let r = simd_add_i64(&a, &b);
        assert_eq!(r, vec![11, 22, 33, 40, 50]);
    }

    #[test]
    fn test_simd_sub_i64() {
        let a = vec![10, 20, 30, 40];
        let b = vec![1, 2, 3, 4];
        let r = simd_sub_i64(&a, &b);
        assert_eq!(r, vec![9, 18, 27, 36]);
    }

    #[test]
    fn test_simd_sub_i64_shorter_a() {
        let a = vec![10, 20];
        let b = vec![1, 2, 3, 4];
        let r = simd_sub_i64(&a, &b);
        assert_eq!(r, vec![9, 18, -3, -4]);
    }

    #[test]
    fn test_simd_scalar_mul_i64() {
        let coeffs = vec![1, 2, 3, 4, 5];
        let r = simd_scalar_mul_i64(&coeffs, 3);
        assert_eq!(r, vec![3, 6, 9, 12, 15]);
    }

    #[test]
    fn test_simd_coeffs_equal_same() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 4];
        assert!(simd_coeffs_equal(&a, &b));
    }

    #[test]
    fn test_simd_coeffs_equal_trailing_zeros() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3, 0, 0];
        assert!(simd_coeffs_equal(&a, &b));
    }

    #[test]
    fn test_simd_coeffs_equal_different() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 5];
        assert!(!simd_coeffs_equal(&a, &b));
    }

    #[test]
    fn test_simd_eval_horner() {
        // p(x) = 1 + 2x + 3x^2, evaluate at x=2
        // = 1 + 4 + 12 = 17
        let coeffs = vec![1, 2, 3];
        assert_eq!(simd_eval_horner(&coeffs, 2), 17);
    }

    #[test]
    fn test_simd_eval_horner_empty() {
        assert_eq!(simd_eval_horner(&[], 42), 0);
    }

    #[test]
    fn test_simd_dot_product_i64() {
        let a = vec![1, 2, 3, 4];
        let b = vec![5, 6, 7, 8];
        // 5 + 12 + 21 + 32 = 70
        assert_eq!(simd_dot_product_i64(&a, &b), Some(70));
    }

    #[test]
    fn test_simd_dot_product_mismatched() {
        let a = vec![1, 2];
        let b = vec![1, 2, 3];
        assert_eq!(simd_dot_product_i64(&a, &b), None);
    }

    #[test]
    fn test_simd_convolve_i64() {
        // (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
        let a = vec![1, 2];
        let b = vec![3, 4];
        let r = simd_convolve_i64(&a, &b);
        assert_eq!(r, vec![3, 10, 8]);
    }

    #[test]
    fn test_simd_convolve_larger() {
        // (1 + x + x^2)(1 + x) = 1 + 2x + 2x^2 + x^3
        let a = vec![1, 1, 1];
        let b = vec![1, 1];
        let r = simd_convolve_i64(&a, &b);
        assert_eq!(r, vec![1, 2, 2, 1]);
    }

    #[test]
    fn test_simd_convolve_empty() {
        assert!(simd_convolve_i64(&[], &[1, 2]).is_empty());
        assert!(simd_convolve_i64(&[1], &[]).is_empty());
    }

    #[test]
    fn test_simd_negate_i64() {
        let coeffs = vec![1, -2, 3, -4, 5];
        let r = simd_negate_i64(&coeffs);
        assert_eq!(r, vec![-1, 2, -3, 4, -5]);
    }

    #[test]
    fn test_simd_linf_norm() {
        let coeffs = vec![1, -7, 3, -4, 5];
        assert_eq!(simd_linf_norm(&coeffs), 7);
    }

    #[test]
    fn test_simd_linf_norm_empty() {
        assert_eq!(simd_linf_norm(&[]), 0);
    }

    #[test]
    fn test_large_add() {
        // Test with a vector larger than CHUNK to exercise both paths
        let a: Vec<i64> = (0..100).collect();
        let b: Vec<i64> = (100..200).collect();
        let r = simd_add_i64(&a, &b);
        for (i, &item) in r.iter().enumerate() {
            assert_eq!(item, (i as i64) + (100 + i as i64));
        }
    }

    #[test]
    fn test_large_scalar_mul() {
        let coeffs: Vec<i64> = (1..=50).collect();
        let r = simd_scalar_mul_i64(&coeffs, -2);
        for (i, &v) in r.iter().enumerate() {
            assert_eq!(v, -2 * (i as i64 + 1));
        }
    }
}
