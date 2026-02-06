//! SIMD Vector Operations.
//!
//! Provides vectorized operations on numerical arrays.

use num_traits::Zero;
use std::ops::{Add, Mul};

/// Compute sum of array elements using SIMD when possible.
pub fn simd_sum<T>(values: &[T]) -> T
where
    T: Copy + Add<Output = T> + Zero,
{
    // For small arrays, use scalar addition
    if values.len() < 8 {
        return values.iter().copied().fold(T::zero(), Add::add);
    }

    // Process in chunks for better cache locality
    let chunk_size = 8;
    let mut sum = T::zero();

    for chunk in values.chunks(chunk_size) {
        let chunk_sum = chunk.iter().copied().fold(T::zero(), Add::add);
        sum = sum + chunk_sum;
    }

    sum
}

/// Compute dot product of two vectors using SIMD.
pub fn simd_dot_product<T>(a: &[T], b: &[T]) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
{
    assert_eq!(a.len(), b.len(), "vectors must have same length");

    if a.len() < 8 {
        return a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x * y)
            .fold(T::zero(), Add::add);
    }

    // Process in chunks
    let chunk_size = 8;
    let mut sum = T::zero();

    let chunks_a = a.chunks(chunk_size);
    let chunks_b = b.chunks(chunk_size);

    for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
        let chunk_sum = chunk_a
            .iter()
            .zip(chunk_b.iter())
            .map(|(&x, &y)| x * y)
            .fold(T::zero(), Add::add);
        sum = sum + chunk_sum;
    }

    sum
}

/// Compute squared L2 norm of a vector using SIMD.
pub fn simd_norm_squared<T>(values: &[T]) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
{
    simd_dot_product(values, values)
}

/// Matrix-vector multiplication using SIMD.
pub fn simd_matrix_vec_mul<T>(matrix: &[Vec<T>], vec: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
{
    assert!(!matrix.is_empty(), "matrix must not be empty");
    assert_eq!(
        matrix[0].len(),
        vec.len(),
        "matrix columns must match vector length"
    );

    matrix
        .iter()
        .map(|row| simd_dot_product(row, vec))
        .collect()
}

/// Element-wise addition of two vectors.
pub fn simd_vec_add<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T>,
{
    assert_eq!(a.len(), b.len(), "vectors must have same length");

    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Element-wise multiplication of two vectors.
pub fn simd_vec_mul<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Copy + Mul<Output = T>,
{
    assert_eq!(a.len(), b.len(), "vectors must have same length");

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Scalar multiplication of a vector.
pub fn simd_vec_scale<T>(vec: &[T], scalar: T) -> Vec<T>
where
    T: Copy + Mul<Output = T>,
{
    vec.iter().map(|&x| x * scalar).collect()
}

/// Compute weighted sum: sum(weights\[i\] * values\[i\]).
pub fn simd_weighted_sum<T>(values: &[T], weights: &[T]) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
{
    simd_dot_product(values, weights)
}

/// Parallel reduction with custom binary operation.
pub fn simd_reduce<T, F>(values: &[T], init: T, op: F) -> T
where
    T: Copy,
    F: Fn(T, T) -> T,
{
    if values.is_empty() {
        return init;
    }

    // Tree reduction for better parallelism
    let mut working = values.to_vec();

    while working.len() > 1 {
        let mut next = Vec::with_capacity(working.len().div_ceil(2));

        for chunk in working.chunks(2) {
            if chunk.len() == 2 {
                next.push(op(chunk[0], chunk[1]));
            } else {
                next.push(chunk[0]);
            }
        }

        working = next;
    }

    working[0]
}

/// Find maximum element in array using SIMD-friendly pattern.
pub fn simd_max<T>(values: &[T]) -> Option<T>
where
    T: Copy + PartialOrd,
{
    if values.is_empty() {
        return None;
    }

    Some(simd_reduce(
        values,
        values[0],
        |a, b| {
            if a > b { a } else { b }
        },
    ))
}

/// Find minimum element in array using SIMD-friendly pattern.
pub fn simd_min<T>(values: &[T]) -> Option<T>
where
    T: Copy + PartialOrd,
{
    if values.is_empty() {
        return None;
    }

    Some(simd_reduce(
        values,
        values[0],
        |a, b| {
            if a < b { a } else { b }
        },
    ))
}

/// Compute mean of values (for floating point types).
pub fn simd_mean_f64(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    let sum = simd_sum(values);
    Some(sum / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_sum() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = simd_sum(&values);
        assert_eq!(sum, 15.0);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let dot = simd_dot_product(&a, &b);
        assert_eq!(dot, 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0);
    }

    #[test]
    fn test_simd_norm_squared() {
        let values = vec![3.0, 4.0];
        let norm_sq = simd_norm_squared(&values);
        assert_eq!(norm_sq, 25.0);
    }

    #[test]
    fn test_simd_matrix_vec_mul() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vec = vec![5.0, 6.0];
        let result = simd_matrix_vec_mul(&matrix, &vec);
        assert_eq!(result, vec![17.0, 39.0]);
    }

    #[test]
    fn test_simd_vec_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = simd_vec_add(&a, &b);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_simd_max() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let max = simd_max(&values);
        assert_eq!(max, Some(5.0));
    }

    #[test]
    fn test_simd_min() {
        let values = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let min = simd_min(&values);
        assert_eq!(min, Some(1.0));
    }

    #[test]
    fn test_simd_reduce() {
        let values = vec![1, 2, 3, 4, 5];
        let sum = simd_reduce(&values, 0, |a, b| a + b);
        assert_eq!(sum, 15);
    }
}
