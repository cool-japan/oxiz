//! SIMD-Optimized Matrix Operations.
#![allow(clippy::needless_range_loop, clippy::type_complexity)] // Matrix algorithms use explicit indexing
//!
//! Provides cache-friendly matrix operations with SIMD-style chunking.

use num_traits::{Float, Zero};
use std::ops::{Add, Mul, Sub};

/// SIMD-friendly matrix-vector multiplication.
pub fn simd_matrix_vec_mul<T>(matrix: &[Vec<T>], vec: &[T]) -> Vec<T>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    let rows = matrix.len();
    if rows == 0 {
        return Vec::new();
    }

    let cols = matrix[0].len();
    if cols != vec.len() {
        panic!("matrix-vector dimension mismatch");
    }

    let mut result = vec![T::zero(); rows];

    // Process in chunks for cache locality
    const CHUNK_SIZE: usize = 8;

    for (i, row) in matrix.iter().enumerate() {
        let mut sum = T::zero();

        for chunk_idx in (0..cols).step_by(CHUNK_SIZE) {
            let chunk_end = (chunk_idx + CHUNK_SIZE).min(cols);

            for j in chunk_idx..chunk_end {
                sum = sum.clone() + row[j].clone() * vec[j].clone();
            }
        }

        result[i] = sum;
    }

    result
}

/// SIMD-friendly matrix-matrix multiplication.
pub fn simd_matrix_mul<T>(a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    let rows_a = a.len();
    if rows_a == 0 {
        return Vec::new();
    }

    let cols_a = a[0].len();
    let rows_b = b.len();
    if rows_b == 0 || cols_a != rows_b {
        panic!("matrix dimension mismatch");
    }

    let cols_b = b[0].len();

    // Transpose B for cache-friendly access
    let b_t = transpose(b);

    let mut result = vec![vec![T::zero(); cols_b]; rows_a];

    const TILE_SIZE: usize = 32;

    // Tiled matrix multiplication
    for i_tile in (0..rows_a).step_by(TILE_SIZE) {
        let i_end = (i_tile + TILE_SIZE).min(rows_a);

        for j_tile in (0..cols_b).step_by(TILE_SIZE) {
            let j_end = (j_tile + TILE_SIZE).min(cols_b);

            for k_tile in (0..cols_a).step_by(TILE_SIZE) {
                let k_end = (k_tile + TILE_SIZE).min(cols_a);

                // Compute tile
                for i in i_tile..i_end {
                    for j in j_tile..j_end {
                        let mut sum = result[i][j].clone();

                        for k in k_tile..k_end {
                            sum = sum.clone() + a[i][k].clone() * b_t[j][k].clone();
                        }

                        result[i][j] = sum;
                    }
                }
            }
        }
    }

    result
}

/// Transpose a matrix.
pub fn transpose<T: Clone>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    if matrix.is_empty() {
        return Vec::new();
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![vec![matrix[0][0].clone(); rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j].clone();
        }
    }

    result
}

/// SIMD-friendly LU decomposition with partial pivoting.
pub fn simd_lu_decomposition<T>(matrix: &[Vec<T>]) -> Option<(Vec<Vec<T>>, Vec<Vec<T>>, Vec<usize>)>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Float,
{
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return None;
    }

    let mut a = matrix.to_vec();
    let mut l = vec![vec![T::zero(); n]; n];
    let mut u = vec![vec![T::zero(); n]; n];
    let mut perm: Vec<usize> = (0..n).collect();

    for i in 0..n {
        l[i][i] = T::one();
    }

    for k in 0..n {
        // Partial pivoting
        let mut max_idx = k;
        let mut max_val = a[k][k].abs();

        for i in (k + 1)..n {
            let val = a[i][k].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        if max_val < T::epsilon() {
            return None; // Singular matrix
        }

        if max_idx != k {
            a.swap(k, max_idx);
            perm.swap(k, max_idx);
            if k > 0 {
                for j in 0..k {
                    let temp = l[k][j];
                    l[k][j] = l[max_idx][j];
                    l[max_idx][j] = temp;
                }
            }
        }

        // Compute L and U
        for j in k..n {
            u[k][j] = a[k][j];

            for s in 0..k {
                u[k][j] = u[k][j] - l[k][s] * u[s][j];
            }
        }

        for i in (k + 1)..n {
            l[i][k] = a[i][k];

            for s in 0..k {
                l[i][k] = l[i][k] - l[i][s] * u[s][k];
            }

            l[i][k] = l[i][k] / u[k][k];
        }
    }

    Some((l, u, perm))
}

/// Solve linear system using LU decomposition.
pub fn simd_lu_solve<T>(l: &[Vec<T>], u: &[Vec<T>], perm: &[usize], b: &[T]) -> Option<Vec<T>>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Float,
{
    let n = l.len();
    if n == 0 || u.len() != n || perm.len() != n || b.len() != n {
        return None;
    }

    // Apply permutation to b
    let mut b_perm = vec![T::zero(); n];
    for i in 0..n {
        b_perm[i] = b[perm[i]];
    }

    // Forward substitution (L * y = b_perm)
    let mut y = vec![T::zero(); n];
    for i in 0..n {
        let mut sum = b_perm[i];
        for j in 0..i {
            sum = sum - l[i][j] * y[j];
        }
        y[i] = sum;
    }

    // Backward substitution (U * x = y)
    let mut x = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum = sum - u[i][j] * x[j];
        }
        x[i] = sum / u[i][i];
    }

    Some(x)
}

/// Compute matrix determinant using LU decomposition.
pub fn simd_determinant<T>(matrix: &[Vec<T>]) -> Option<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Float,
{
    let (_l, u, perm) = simd_lu_decomposition(matrix)?;

    let n = u.len();

    // Det = product of diagonal of U times sign of permutation
    let mut det = T::one();
    for i in 0..n {
        det = det * u[i][i];
    }

    // Count inversions in permutation: pairs (i, j) where i < j but perm[i] > perm[j]
    let mut inversions = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }

    if inversions % 2 == 1 {
        det = T::zero() - det;
    }

    Some(det)
}

/// Matrix inversion using LU decomposition.
pub fn simd_matrix_inverse<T>(matrix: &[Vec<T>]) -> Option<Vec<Vec<T>>>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Float,
{
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return None;
    }

    let (l, u, perm) = simd_lu_decomposition(matrix)?;

    let mut inverse = vec![vec![T::zero(); n]; n];

    // Solve for each column of the identity matrix
    for j in 0..n {
        let mut e = vec![T::zero(); n];
        e[j] = T::one();

        let col = simd_lu_solve(&l, &u, &perm, &e)?;
        for i in 0..n {
            inverse[i][j] = col[i];
        }
    }

    Some(inverse)
}

/// Compute QR decomposition using Gram-Schmidt.
pub fn simd_qr_decomposition<T>(matrix: &[Vec<T>]) -> Option<(Vec<Vec<T>>, Vec<Vec<T>>)>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Float,
{
    let rows = matrix.len();
    if rows == 0 {
        return None;
    }
    let cols = matrix[0].len();

    let mut q = vec![vec![T::zero(); cols]; rows];
    let mut r = vec![vec![T::zero(); cols]; cols];

    for j in 0..cols {
        // Get column j
        let mut v: Vec<T> = matrix.iter().map(|row| row[j]).collect();

        // Orthogonalize against previous columns
        for i in 0..j {
            let q_col: Vec<T> = q.iter().map(|row| row[i]).collect();

            let dot = dot_product(&q_col, &v);
            r[i][j] = dot;

            for k in 0..rows {
                v[k] = v[k] - q_col[k] * dot;
            }
        }

        // Normalize
        let norm = vector_norm(&v);
        if norm < T::epsilon() {
            return None; // Linearly dependent columns
        }

        r[j][j] = norm;

        for k in 0..rows {
            q[k][j] = v[k] / norm;
        }
    }

    Some((q, r))
}

fn dot_product<T>(a: &[T], b: &[T]) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Zero,
{
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.clone() * y.clone())
        .fold(T::zero(), |acc, x| acc + x)
}

fn vector_norm<T>(v: &[T]) -> T
where
    T: Clone + Add<Output = T> + Mul<Output = T> + Float,
{
    let sum_sq = v.iter().map(|x| *x * *x).fold(T::zero(), |acc, x| acc + x);
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];

        let transposed = transpose(&matrix);

        assert_eq!(transposed.len(), 3);
        assert_eq!(transposed[0], vec![1, 4]);
        assert_eq!(transposed[1], vec![2, 5]);
        assert_eq!(transposed[2], vec![3, 6]);
    }

    #[test]
    fn test_matrix_vec_mul() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let vec = vec![2.0, 3.0];

        let result = simd_matrix_vec_mul(&matrix, &vec);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 8.0).abs() < 1e-10);
        assert!((result[1] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let result = simd_matrix_mul(&a, &b);

        assert_eq!(result.len(), 2);
        assert!((result[0][0] - 19.0).abs() < 1e-10);
        assert!((result[0][1] - 22.0).abs() < 1e-10);
        assert!((result[1][0] - 43.0).abs() < 1e-10);
        assert!((result[1][1] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_decomposition() {
        let matrix = vec![vec![4.0, 3.0], vec![6.0, 3.0]];

        let (l, u, _perm) = simd_lu_decomposition(&matrix).expect("LU decomposition failed");

        assert_eq!(l.len(), 2);
        assert_eq!(u.len(), 2);
    }

    #[test]
    fn test_determinant() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let det = simd_determinant(&matrix).expect("determinant computation failed");

        assert!((det - (-2.0)).abs() < 1e-10);
    }
}
