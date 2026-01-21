//! OxiBLAS: High-Performance BLAS Operations for Large-Scale LP
//!
//! This module provides pure Rust implementations of BLAS (Basic Linear Algebra Subprograms)
//! operations optimized for vectorization and cache efficiency. These operations are designed
//! for large-scale linear programming problems with 1000+ variables.
//!
//! # Operations
//!
//! ## Level 1 BLAS (Vector-Vector)
//! - `ddot`: Dot product of two vectors
//! - `dnrm2`: Euclidean norm of a vector
//! - `dscal`: Scale a vector by a scalar
//! - `daxpy`: y = alpha * x + y
//! - `dcopy`: Copy vector x to y
//! - `dswap`: Swap vectors x and y
//! - `idamax`: Index of maximum absolute value
//!
//! ## Level 2 BLAS (Matrix-Vector)
//! - `dgemv`: General matrix-vector multiplication
//! - `dtrsv`: Triangular solve with single vector
//!
//! ## Level 3 BLAS (Matrix-Matrix)
//! - `dgemm`: General matrix-matrix multiplication with blocking
//! - `dtrsm`: Triangular solve with matrix
//!
//! # Example
//!
//! ```
//! use oxiz_math::blas::{ddot, dgemv, dgemm, Transpose};
//!
//! // Level 1: Dot product
//! let x = vec![1.0, 2.0, 3.0];
//! let y = vec![4.0, 5.0, 6.0];
//! let dot = ddot(&x, &y);
//! assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
//!
//! // Level 2: Matrix-vector multiplication
//! let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 row-major
//! let x2 = vec![1.0, 2.0];
//! let mut y2 = vec![0.0, 0.0];
//! dgemv(Transpose::NoTrans, 2, 2, 1.0, &a, &x2, 0.0, &mut y2);
//! ```
//!
//! # Performance Optimizations
//!
//! - **Cache blocking**: DGEMM uses block-wise multiplication for better cache utilization
//! - **Loop unrolling**: Critical inner loops are unrolled for better instruction pipelining
//! - **Memory access patterns**: Designed for row-major storage with stride-aware operations
//! - **SIMD-friendly**: Inner loops are structured for auto-vectorization by the compiler

/// Block size for cache-efficient matrix operations.
/// Chosen to fit L1 cache (typically 32KB) for three blocks of f64s.
const BLOCK_SIZE: usize = 64;

/// Transpose option for matrix operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    /// No transpose (use matrix as-is).
    NoTrans,
    /// Transpose the matrix.
    Trans,
}

/// Side for triangular solve operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Matrix A is on the left: solve AX = B.
    Left,
    /// Matrix A is on the right: solve XA = B.
    Right,
}

/// Upper or lower triangular indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpLo {
    /// Upper triangular matrix.
    Upper,
    /// Lower triangular matrix.
    Lower,
}

/// Diagonal type indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    /// Non-unit diagonal (use actual diagonal elements).
    NonUnit,
    /// Unit diagonal (diagonal elements are assumed to be 1).
    Unit,
}

// =============================================================================
// Level 1 BLAS: Vector-Vector Operations
// =============================================================================

/// DDOT: Compute the dot product of two vectors.
///
/// Returns x^T * y = sum(x[i] * y[i])
///
/// # Arguments
/// * `x` - First input vector
/// * `y` - Second input vector
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use oxiz_math::blas::ddot;
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![4.0, 5.0, 6.0];
/// assert_eq!(ddot(&x, &y), 32.0);
/// ```
#[inline]
pub fn ddot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(
        x.len(),
        y.len(),
        "Vector lengths must match for dot product"
    );

    let n = x.len();
    let mut sum = 0.0;

    // Unroll by 4 for better pipelining
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum += x[idx] * y[idx];
        sum += x[idx + 1] * y[idx + 1];
        sum += x[idx + 2] * y[idx + 2];
        sum += x[idx + 3] * y[idx + 3];
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        sum += x[i] * y[i];
    }

    // Use remainder variable to satisfy clippy (even though we use the range)
    let _ = remainder;

    sum
}

/// DNRM2: Compute the Euclidean (L2) norm of a vector.
///
/// Returns ||x||_2 = sqrt(sum(x[i]^2))
///
/// Uses a numerically stable algorithm to avoid overflow/underflow.
///
/// # Example
/// ```
/// use oxiz_math::blas::dnrm2;
/// let x = vec![3.0, 4.0];
/// assert!((dnrm2(&x) - 5.0).abs() < 1e-10);
/// ```
#[inline]
pub fn dnrm2(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }

    let n = x.len();

    // Find scale factor to avoid overflow/underflow
    let mut scale = 0.0f64;
    for &xi in x {
        let abs_xi = xi.abs();
        if abs_xi > scale {
            scale = abs_xi;
        }
    }

    if scale == 0.0 {
        return 0.0;
    }

    // Compute scaled sum of squares
    let mut sum = 0.0;
    let inv_scale = 1.0 / scale;

    // Unroll by 4
    let chunks = n / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let s0 = x[idx] * inv_scale;
        let s1 = x[idx + 1] * inv_scale;
        let s2 = x[idx + 2] * inv_scale;
        let s3 = x[idx + 3] * inv_scale;
        sum += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
    }

    for s in x.iter().skip(chunks * 4).take(n - chunks * 4) {
        let s = s * inv_scale;
        sum += s * s;
    }

    scale * sum.sqrt()
}

/// DSCAL: Scale a vector by a scalar.
///
/// Computes x = alpha * x
///
/// # Arguments
/// * `alpha` - Scalar multiplier
/// * `x` - Vector to scale (modified in place)
///
/// # Example
/// ```
/// use oxiz_math::blas::dscal;
/// let mut x = vec![1.0, 2.0, 3.0];
/// dscal(2.0, &mut x);
/// assert_eq!(x, vec![2.0, 4.0, 6.0]);
/// ```
#[inline]
pub fn dscal(alpha: f64, x: &mut [f64]) {
    if alpha == 1.0 {
        return;
    }

    if alpha == 0.0 {
        x.fill(0.0);
        return;
    }

    let n = x.len();
    let chunks = n / 4;

    for i in 0..chunks {
        let idx = i * 4;
        x[idx] *= alpha;
        x[idx + 1] *= alpha;
        x[idx + 2] *= alpha;
        x[idx + 3] *= alpha;
    }

    for x_val in x.iter_mut().skip(chunks * 4).take(n - chunks * 4) {
        *x_val *= alpha;
    }
}

/// DAXPY: Compute y = alpha * x + y (vector plus scaled vector).
///
/// # Arguments
/// * `alpha` - Scalar multiplier for x
/// * `x` - Input vector
/// * `y` - Input/output vector (modified in place)
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use oxiz_math::blas::daxpy;
/// let x = vec![1.0, 2.0, 3.0];
/// let mut y = vec![4.0, 5.0, 6.0];
/// daxpy(2.0, &x, &mut y);
/// assert_eq!(y, vec![6.0, 9.0, 12.0]);
/// ```
#[inline]
pub fn daxpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match for DAXPY");

    if alpha == 0.0 {
        return;
    }

    let n = x.len();
    let chunks = n / 4;

    for i in 0..chunks {
        let idx = i * 4;
        y[idx] += alpha * x[idx];
        y[idx + 1] += alpha * x[idx + 1];
        y[idx + 2] += alpha * x[idx + 2];
        y[idx + 3] += alpha * x[idx + 3];
    }

    for i in (chunks * 4)..n {
        y[i] += alpha * x[i];
    }
}

/// DCOPY: Copy vector x to vector y.
///
/// # Arguments
/// * `x` - Source vector
/// * `y` - Destination vector (modified in place)
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use oxiz_math::blas::dcopy;
/// let x = vec![1.0, 2.0, 3.0];
/// let mut y = vec![0.0, 0.0, 0.0];
/// dcopy(&x, &mut y);
/// assert_eq!(y, vec![1.0, 2.0, 3.0]);
/// ```
#[inline]
pub fn dcopy(x: &[f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match for DCOPY");
    y.copy_from_slice(x);
}

/// DSWAP: Swap vectors x and y.
///
/// # Arguments
/// * `x` - First vector (modified in place)
/// * `y` - Second vector (modified in place)
///
/// # Panics
/// Panics if vectors have different lengths.
///
/// # Example
/// ```
/// use oxiz_math::blas::dswap;
/// let mut x = vec![1.0, 2.0, 3.0];
/// let mut y = vec![4.0, 5.0, 6.0];
/// dswap(&mut x, &mut y);
/// assert_eq!(x, vec![4.0, 5.0, 6.0]);
/// assert_eq!(y, vec![1.0, 2.0, 3.0]);
/// ```
#[inline]
pub fn dswap(x: &mut [f64], y: &mut [f64]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match for DSWAP");
    x.swap_with_slice(y);
}

/// IDAMAX: Find index of element with maximum absolute value.
///
/// Returns the index of the first element with the largest absolute value.
/// Returns 0 for empty vectors.
///
/// # Example
/// ```
/// use oxiz_math::blas::idamax;
/// let x = vec![1.0, -5.0, 3.0];
/// assert_eq!(idamax(&x), 1);
/// ```
#[inline]
pub fn idamax(x: &[f64]) -> usize {
    if x.is_empty() {
        return 0;
    }

    let mut max_idx = 0;
    let mut max_val = x[0].abs();

    for (i, &xi) in x.iter().enumerate().skip(1) {
        let abs_xi = xi.abs();
        if abs_xi > max_val {
            max_val = abs_xi;
            max_idx = i;
        }
    }

    max_idx
}

/// DASUM: Compute the sum of absolute values of vector elements.
///
/// Returns sum(|x[i]|)
///
/// # Example
/// ```
/// use oxiz_math::blas::dasum;
/// let x = vec![1.0, -2.0, 3.0];
/// assert_eq!(dasum(&x), 6.0);
/// ```
#[inline]
pub fn dasum(x: &[f64]) -> f64 {
    let n = x.len();
    let mut sum = 0.0;

    let chunks = n / 4;

    for i in 0..chunks {
        let idx = i * 4;
        sum += x[idx].abs();
        sum += x[idx + 1].abs();
        sum += x[idx + 2].abs();
        sum += x[idx + 3].abs();
    }

    for x_val in x.iter().skip(chunks * 4).take(n - chunks * 4) {
        sum += x_val.abs();
    }

    sum
}

// =============================================================================
// Level 2 BLAS: Matrix-Vector Operations
// =============================================================================

/// DGEMV: General matrix-vector multiplication.
///
/// Computes y = alpha * op(A) * x + beta * y
///
/// where op(A) = A if trans == NoTrans, or op(A) = A^T if trans == Trans.
///
/// # Arguments
/// * `trans` - Whether to transpose A
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Matrix A in row-major order (m x n)
/// * `x` - Input vector (n for NoTrans, m for Trans)
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector (m for NoTrans, n for Trans), modified in place
///
/// # Panics
/// Panics if dimensions don't match.
///
/// # Example
/// ```
/// use oxiz_math::blas::{dgemv, Transpose};
/// let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 row-major
/// let x = vec![1.0, 2.0];
/// let mut y = vec![0.0, 0.0];
/// dgemv(Transpose::NoTrans, 2, 2, 1.0, &a, &x, 0.0, &mut y);
/// assert_eq!(y, vec![5.0, 11.0]); // [1*1+2*2, 3*1+4*2]
/// ```
#[allow(clippy::too_many_arguments)]
pub fn dgemv(
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) {
    assert_eq!(a.len(), m * n, "Matrix A size must be m * n");

    match trans {
        Transpose::NoTrans => {
            assert_eq!(x.len(), n, "Vector x length must be n for NoTrans");
            assert_eq!(y.len(), m, "Vector y length must be m for NoTrans");

            // Scale y by beta
            if beta == 0.0 {
                y.fill(0.0);
            } else if beta != 1.0 {
                dscal(beta, y);
            }

            if alpha == 0.0 {
                return;
            }

            // y = alpha * A * x + beta * y
            for (i, y_val) in y.iter_mut().enumerate().take(m) {
                let row_start = i * n;
                let mut sum = 0.0;

                // Unroll inner loop
                let chunks = n / 4;
                for j in 0..chunks {
                    let idx = j * 4;
                    sum += a[row_start + idx] * x[idx];
                    sum += a[row_start + idx + 1] * x[idx + 1];
                    sum += a[row_start + idx + 2] * x[idx + 2];
                    sum += a[row_start + idx + 3] * x[idx + 3];
                }
                for j in (chunks * 4)..n {
                    sum += a[row_start + j] * x[j];
                }

                *y_val += alpha * sum;
            }
        }
        Transpose::Trans => {
            assert_eq!(x.len(), m, "Vector x length must be m for Trans");
            assert_eq!(y.len(), n, "Vector y length must be n for Trans");

            // Scale y by beta
            if beta == 0.0 {
                y.fill(0.0);
            } else if beta != 1.0 {
                dscal(beta, y);
            }

            if alpha == 0.0 {
                return;
            }

            // y = alpha * A^T * x + beta * y
            for (i, x_val) in x.iter().enumerate().take(m) {
                let row_start = i * n;
                let alpha_xi = alpha * x_val;

                // Unroll inner loop
                let chunks = n / 4;
                for j in 0..chunks {
                    let idx = j * 4;
                    y[idx] += alpha_xi * a[row_start + idx];
                    y[idx + 1] += alpha_xi * a[row_start + idx + 1];
                    y[idx + 2] += alpha_xi * a[row_start + idx + 2];
                    y[idx + 3] += alpha_xi * a[row_start + idx + 3];
                }
                for j in (chunks * 4)..n {
                    y[j] += alpha_xi * a[row_start + j];
                }
            }
        }
    }
}

/// DTRSV: Triangular solve with a single vector.
///
/// Solves op(A) * x = b where A is triangular.
///
/// # Arguments
/// * `uplo` - Whether A is upper or lower triangular
/// * `trans` - Whether to use A or A^T
/// * `diag` - Whether A has unit diagonal
/// * `n` - Order of matrix A
/// * `a` - Triangular matrix A in row-major order (n x n)
/// * `b` - Right-hand side vector (modified in place to contain solution)
///
/// # Panics
/// Panics if dimensions don't match.
#[allow(clippy::too_many_arguments)]
pub fn dtrsv(uplo: UpLo, trans: Transpose, diag: Diag, n: usize, a: &[f64], b: &mut [f64]) {
    assert_eq!(a.len(), n * n, "Matrix A size must be n * n");
    assert_eq!(b.len(), n, "Vector b length must be n");

    if n == 0 {
        return;
    }

    match (uplo, trans) {
        (UpLo::Lower, Transpose::NoTrans) | (UpLo::Upper, Transpose::Trans) => {
            // Forward substitution
            for i in 0..n {
                let mut sum = b[i];
                for j in 0..i {
                    let a_ij = if trans == Transpose::Trans {
                        a[j * n + i]
                    } else {
                        a[i * n + j]
                    };
                    sum -= a_ij * b[j];
                }
                if diag == Diag::NonUnit {
                    // Diagonal element is same for both NoTrans and Trans
                    let a_ii = a[i * n + i];
                    b[i] = sum / a_ii;
                } else {
                    b[i] = sum;
                }
            }
        }
        (UpLo::Upper, Transpose::NoTrans) | (UpLo::Lower, Transpose::Trans) => {
            // Backward substitution
            for i in (0..n).rev() {
                let mut sum = b[i];
                for j in (i + 1)..n {
                    let a_ij = if trans == Transpose::Trans {
                        a[j * n + i]
                    } else {
                        a[i * n + j]
                    };
                    sum -= a_ij * b[j];
                }
                if diag == Diag::NonUnit {
                    // Diagonal element is same for both NoTrans and Trans
                    let a_ii = a[i * n + i];
                    b[i] = sum / a_ii;
                } else {
                    b[i] = sum;
                }
            }
        }
    }
}

// =============================================================================
// Level 3 BLAS: Matrix-Matrix Operations
// =============================================================================

/// DGEMM: General matrix-matrix multiplication with cache blocking.
///
/// Computes C = alpha * op(A) * op(B) + beta * C
///
/// where op(X) = X if trans == NoTrans, or op(X) = X^T if trans == Trans.
///
/// Uses block-wise multiplication for better cache utilization on large matrices.
///
/// # Arguments
/// * `trans_a` - Whether to transpose A
/// * `trans_b` - Whether to transpose B
/// * `m` - Number of rows of op(A) and C
/// * `n` - Number of columns of op(B) and C
/// * `k` - Number of columns of op(A) and rows of op(B)
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Matrix A
/// * `b` - Matrix B
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix C (modified in place)
///
/// # Panics
/// Panics if dimensions don't match.
///
/// # Example
/// ```
/// use oxiz_math::blas::{dgemm, Transpose};
/// let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 row-major
/// let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 row-major
/// let mut c = vec![0.0; 4];
/// dgemm(Transpose::NoTrans, Transpose::NoTrans, 2, 2, 2, 1.0, &a, &b, 0.0, &mut c);
/// assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn dgemm(
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    beta: f64,
    c: &mut [f64],
) {
    // Validate dimensions
    let (a_rows, a_cols) = match trans_a {
        Transpose::NoTrans => (m, k),
        Transpose::Trans => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        Transpose::NoTrans => (k, n),
        Transpose::Trans => (n, k),
    };

    assert_eq!(a.len(), a_rows * a_cols, "Matrix A size mismatch");
    assert_eq!(b.len(), b_rows * b_cols, "Matrix B size mismatch");
    assert_eq!(c.len(), m * n, "Matrix C size must be m * n");

    // Scale C by beta
    if beta == 0.0 {
        c.fill(0.0);
    } else if beta != 1.0 {
        for ci in c.iter_mut() {
            *ci *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    // Use blocked algorithm for large matrices
    if m * n * k > BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE {
        dgemm_blocked(trans_a, trans_b, m, n, k, alpha, a, b, c);
    } else {
        dgemm_simple(trans_a, trans_b, m, n, k, alpha, a, b, c);
    }
}

/// Simple (non-blocked) matrix multiplication for small matrices.
#[allow(clippy::too_many_arguments)]
fn dgemm_simple(
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    let (a_cols, b_cols) = match (trans_a, trans_b) {
        (Transpose::NoTrans, Transpose::NoTrans) => (k, n),
        (Transpose::NoTrans, Transpose::Trans) => (k, k),
        (Transpose::Trans, Transpose::NoTrans) => (m, n),
        (Transpose::Trans, Transpose::Trans) => (m, k),
    };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                let a_il = match trans_a {
                    Transpose::NoTrans => a[i * a_cols + l],
                    Transpose::Trans => a[l * a_cols + i],
                };
                let b_lj = match trans_b {
                    Transpose::NoTrans => b[l * b_cols + j],
                    Transpose::Trans => b[j * b_cols + l],
                };
                sum += a_il * b_lj;
            }
            c[i * n + j] += alpha * sum;
        }
    }
}

/// Blocked matrix multiplication for better cache utilization.
#[allow(clippy::too_many_arguments)]
fn dgemm_blocked(
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
) {
    let (a_cols, b_cols) = match (trans_a, trans_b) {
        (Transpose::NoTrans, Transpose::NoTrans) => (k, n),
        (Transpose::NoTrans, Transpose::Trans) => (k, k),
        (Transpose::Trans, Transpose::NoTrans) => (m, n),
        (Transpose::Trans, Transpose::Trans) => (m, k),
    };

    // Block over all three dimensions
    for i0 in (0..m).step_by(BLOCK_SIZE) {
        let i1 = (i0 + BLOCK_SIZE).min(m);

        for j0 in (0..n).step_by(BLOCK_SIZE) {
            let j1 = (j0 + BLOCK_SIZE).min(n);

            for l0 in (0..k).step_by(BLOCK_SIZE) {
                let l1 = (l0 + BLOCK_SIZE).min(k);

                // Multiply block
                for i in i0..i1 {
                    for j in j0..j1 {
                        let mut sum = 0.0;
                        for l in l0..l1 {
                            let a_il = match trans_a {
                                Transpose::NoTrans => a[i * a_cols + l],
                                Transpose::Trans => a[l * a_cols + i],
                            };
                            let b_lj = match trans_b {
                                Transpose::NoTrans => b[l * b_cols + j],
                                Transpose::Trans => b[j * b_cols + l],
                            };
                            sum += a_il * b_lj;
                        }
                        c[i * n + j] += alpha * sum;
                    }
                }
            }
        }
    }
}

/// DTRSM: Triangular solve with multiple right-hand sides.
///
/// Solves op(A) * X = alpha * B (Side::Left) or X * op(A) = alpha * B (Side::Right)
/// where A is triangular.
///
/// # Arguments
/// * `side` - Whether A is on the left or right
/// * `uplo` - Whether A is upper or lower triangular
/// * `trans` - Whether to use A or A^T
/// * `diag` - Whether A has unit diagonal
/// * `m` - Number of rows of B
/// * `n` - Number of columns of B
/// * `alpha` - Scalar multiplier
/// * `a` - Triangular matrix A
/// * `b` - Right-hand side matrix B (modified in place to contain X)
///
/// # Panics
/// Panics if dimensions don't match.
#[allow(clippy::too_many_arguments)]
pub fn dtrsm(
    side: Side,
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    b: &mut [f64],
) {
    let a_size = match side {
        Side::Left => m,
        Side::Right => n,
    };

    assert_eq!(a.len(), a_size * a_size, "Matrix A must be square");
    assert_eq!(b.len(), m * n, "Matrix B size must be m * n");

    // Scale B by alpha
    if alpha != 1.0 {
        for bi in b.iter_mut() {
            *bi *= alpha;
        }
    }

    match side {
        Side::Left => dtrsm_left(uplo, trans, diag, m, n, a, b),
        Side::Right => dtrsm_right(uplo, trans, diag, m, n, a, b),
    }
}

/// Triangular solve: op(A) * X = B (A on the left).
fn dtrsm_left(
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    a: &[f64],
    b: &mut [f64],
) {
    for col in 0..n {
        match (uplo, trans) {
            (UpLo::Lower, Transpose::NoTrans) | (UpLo::Upper, Transpose::Trans) => {
                // Forward substitution
                for i in 0..m {
                    let mut sum = b[i * n + col];
                    for j in 0..i {
                        let a_ij = if trans == Transpose::Trans {
                            a[j * m + i]
                        } else {
                            a[i * m + j]
                        };
                        sum -= a_ij * b[j * n + col];
                    }
                    if diag == Diag::NonUnit {
                        let a_ii = a[i * m + i];
                        b[i * n + col] = sum / a_ii;
                    } else {
                        b[i * n + col] = sum;
                    }
                }
            }
            (UpLo::Upper, Transpose::NoTrans) | (UpLo::Lower, Transpose::Trans) => {
                // Backward substitution
                for i in (0..m).rev() {
                    let mut sum = b[i * n + col];
                    for j in (i + 1)..m {
                        let a_ij = if trans == Transpose::Trans {
                            a[j * m + i]
                        } else {
                            a[i * m + j]
                        };
                        sum -= a_ij * b[j * n + col];
                    }
                    if diag == Diag::NonUnit {
                        let a_ii = a[i * m + i];
                        b[i * n + col] = sum / a_ii;
                    } else {
                        b[i * n + col] = sum;
                    }
                }
            }
        }
    }
}

/// Triangular solve: X * op(A) = B (A on the right).
fn dtrsm_right(
    uplo: UpLo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    a: &[f64],
    b: &mut [f64],
) {
    for row in 0..m {
        match (uplo, trans) {
            (UpLo::Upper, Transpose::NoTrans) | (UpLo::Lower, Transpose::Trans) => {
                // Forward substitution on columns
                for j in 0..n {
                    let mut sum = b[row * n + j];
                    for k in 0..j {
                        let a_kj = if trans == Transpose::Trans {
                            a[j * n + k]
                        } else {
                            a[k * n + j]
                        };
                        sum -= b[row * n + k] * a_kj;
                    }
                    if diag == Diag::NonUnit {
                        let a_jj = a[j * n + j];
                        b[row * n + j] = sum / a_jj;
                    } else {
                        b[row * n + j] = sum;
                    }
                }
            }
            (UpLo::Lower, Transpose::NoTrans) | (UpLo::Upper, Transpose::Trans) => {
                // Backward substitution on columns
                for j in (0..n).rev() {
                    let mut sum = b[row * n + j];
                    for k in (j + 1)..n {
                        let a_kj = if trans == Transpose::Trans {
                            a[j * n + k]
                        } else {
                            a[k * n + j]
                        };
                        sum -= b[row * n + k] * a_kj;
                    }
                    if diag == Diag::NonUnit {
                        let a_jj = a[j * n + j];
                        b[row * n + j] = sum / a_jj;
                    } else {
                        b[row * n + j] = sum;
                    }
                }
            }
        }
    }
}

// =============================================================================
// LP-Specific Operations
// =============================================================================

/// Configuration for BLAS operations in LP context.
#[derive(Debug, Clone)]
pub struct BlasLPConfig {
    /// Block size for matrix operations.
    pub block_size: usize,
    /// Tolerance for numerical zero.
    pub zero_tolerance: f64,
    /// Whether to use pivoting in triangular solves.
    pub use_pivoting: bool,
}

impl Default for BlasLPConfig {
    fn default() -> Self {
        Self {
            block_size: BLOCK_SIZE,
            zero_tolerance: 1e-12,
            use_pivoting: true,
        }
    }
}

/// Compute the reduced cost for simplex pivoting.
///
/// reduced_cost = c - c_B * B^{-1} * A
///
/// where c is the objective, c_B is the basic objective, B^{-1} is the basis inverse,
/// and A is the constraint matrix.
#[allow(dead_code)]
pub fn compute_reduced_cost(
    c: &[f64],
    c_b: &[f64],
    b_inv_a: &[f64],
    m: usize,
    n: usize,
) -> Vec<f64> {
    let mut reduced = c.to_vec();

    // reduced = c - c_B * B^{-1} * A
    // This is essentially reduced = c - gemv(trans, c_B, B^{-1}*A)
    for j in 0..n {
        for i in 0..m {
            reduced[j] -= c_b[i] * b_inv_a[i * n + j];
        }
    }

    reduced
}

/// Solve B * x = b for the simplex basis update.
///
/// Uses LU factorization with partial pivoting for numerical stability.
#[allow(dead_code)]
pub fn solve_basis(b: &[f64], n: usize, rhs: &mut [f64], config: &BlasLPConfig) -> bool {
    // Simple LU factorization with partial pivoting
    let mut lu = b.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();

    // LU factorization
    for k in 0..n - 1 {
        // Find pivot
        if config.use_pivoting {
            let mut max_idx = k;
            let mut max_val = lu[k * n + k].abs();
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_idx = i;
                }
            }

            if max_val < config.zero_tolerance {
                return false; // Singular matrix
            }

            if max_idx != k {
                // Swap rows
                for j in 0..n {
                    lu.swap(k * n + j, max_idx * n + j);
                }
                perm.swap(k, max_idx);
            }
        }

        let pivot = lu[k * n + k];
        if pivot.abs() < config.zero_tolerance {
            return false; // Singular matrix
        }

        // Eliminate below diagonal
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                lu[i * n + j] -= factor * lu[k * n + j];
            }
        }
    }

    // Apply permutation to RHS
    let mut tmp = vec![0.0; n];
    for i in 0..n {
        tmp[i] = rhs[perm[i]];
    }
    rhs.copy_from_slice(&tmp);

    // Forward substitution (L * y = Pb)
    for i in 1..n {
        for j in 0..i {
            rhs[i] -= lu[i * n + j] * rhs[j];
        }
    }

    // Backward substitution (U * x = y)
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            rhs[i] -= lu[i * n + j] * rhs[j];
        }
        rhs[i] /= lu[i * n + i];
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn approx_eq_vec(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(&ai, &bi)| approx_eq(ai, bi))
    }

    // =============================================================================
    // Level 1 BLAS Tests
    // =============================================================================

    #[test]
    fn test_ddot() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];
        let result = ddot(&x, &y);
        assert!(approx_eq(result, 70.0)); // 5 + 12 + 21 + 32
    }

    #[test]
    fn test_ddot_empty() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        assert!(approx_eq(ddot(&x, &y), 0.0));
    }

    #[test]
    fn test_dnrm2() {
        let x = vec![3.0, 4.0];
        assert!(approx_eq(dnrm2(&x), 5.0));
    }

    #[test]
    fn test_dnrm2_large_values() {
        // Test numerical stability with large values
        let scale = 1e150;
        let x = vec![3.0 * scale, 4.0 * scale];
        assert!(approx_eq(dnrm2(&x), 5.0 * scale));
    }

    #[test]
    fn test_dnrm2_small_values() {
        // Test numerical stability with small values
        let scale = 1e-150;
        let x = vec![3.0 * scale, 4.0 * scale];
        assert!(approx_eq(dnrm2(&x), 5.0 * scale));
    }

    #[test]
    fn test_dscal() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        dscal(2.0, &mut x);
        assert!(approx_eq_vec(&x, &[2.0, 4.0, 6.0, 8.0, 10.0]));
    }

    #[test]
    fn test_dscal_zero() {
        let mut x = vec![1.0, 2.0, 3.0];
        dscal(0.0, &mut x);
        assert!(approx_eq_vec(&x, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_dscal_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        let original = x.clone();
        dscal(1.0, &mut x);
        assert!(approx_eq_vec(&x, &original));
    }

    #[test]
    fn test_daxpy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        daxpy(2.0, &x, &mut y);
        assert!(approx_eq_vec(&y, &[12.0, 24.0, 36.0, 48.0, 60.0]));
    }

    #[test]
    fn test_daxpy_zero_alpha() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        let original_y = y.clone();
        daxpy(0.0, &x, &mut y);
        assert!(approx_eq_vec(&y, &original_y));
    }

    #[test]
    fn test_dcopy() {
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        dcopy(&x, &mut y);
        assert!(approx_eq_vec(&y, &x));
    }

    #[test]
    fn test_dswap() {
        let mut x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        dswap(&mut x, &mut y);
        assert!(approx_eq_vec(&x, &[4.0, 5.0, 6.0]));
        assert!(approx_eq_vec(&y, &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_idamax() {
        let x = vec![1.0, -5.0, 3.0, -2.0];
        assert_eq!(idamax(&x), 1);
    }

    #[test]
    fn test_idamax_empty() {
        let x: Vec<f64> = vec![];
        assert_eq!(idamax(&x), 0);
    }

    #[test]
    fn test_dasum() {
        let x = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        assert!(approx_eq(dasum(&x), 15.0));
    }

    // =============================================================================
    // Level 2 BLAS Tests
    // =============================================================================

    #[test]
    fn test_dgemv_notrans() {
        // A = [[1, 2], [3, 4]], x = [5, 6]
        // y = A * x = [1*5+2*6, 3*5+4*6] = [17, 39]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0];
        let mut y = vec![0.0, 0.0];
        dgemv(Transpose::NoTrans, 2, 2, 1.0, &a, &x, 0.0, &mut y);
        assert!(approx_eq_vec(&y, &[17.0, 39.0]));
    }

    #[test]
    fn test_dgemv_trans() {
        // A = [[1, 2], [3, 4]], x = [5, 6]
        // y = A^T * x = [1*5+3*6, 2*5+4*6] = [23, 34]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![5.0, 6.0];
        let mut y = vec![0.0, 0.0];
        dgemv(Transpose::Trans, 2, 2, 1.0, &a, &x, 0.0, &mut y);
        assert!(approx_eq_vec(&y, &[23.0, 34.0]));
    }

    #[test]
    fn test_dgemv_with_beta() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![10.0, 10.0];
        // y = 2 * A * x + 3 * y = 2 * [3, 7] + 3 * [10, 10] = [36, 44]
        dgemv(Transpose::NoTrans, 2, 2, 2.0, &a, &x, 3.0, &mut y);
        assert!(approx_eq_vec(&y, &[36.0, 44.0]));
    }

    #[test]
    fn test_dtrsv_lower() {
        // L = [[2, 0], [1, 3]]
        // Solve L * x = [4, 5]
        // x[0] = 4/2 = 2
        // x[1] = (5 - 1*2)/3 = 1
        let l = vec![2.0, 0.0, 1.0, 3.0];
        let mut b = vec![4.0, 5.0];
        dtrsv(
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            2,
            &l,
            &mut b,
        );
        assert!(approx_eq_vec(&b, &[2.0, 1.0]));
    }

    #[test]
    fn test_dtrsv_upper() {
        // U = [[2, 1], [0, 3]]
        // Solve U * x = [5, 6]
        // x[1] = 6/3 = 2
        // x[0] = (5 - 1*2)/2 = 1.5
        let u = vec![2.0, 1.0, 0.0, 3.0];
        let mut b = vec![5.0, 6.0];
        dtrsv(
            UpLo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            2,
            &u,
            &mut b,
        );
        assert!(approx_eq_vec(&b, &[1.5, 2.0]));
    }

    #[test]
    fn test_dtrsv_unit_diagonal() {
        // L = [[1, 0], [2, 1]] (unit diagonal)
        // Solve L * x = [3, 8]
        // x[0] = 3
        // x[1] = 8 - 2*3 = 2
        let l = vec![1.0, 0.0, 2.0, 1.0];
        let mut b = vec![3.0, 8.0];
        dtrsv(UpLo::Lower, Transpose::NoTrans, Diag::Unit, 2, &l, &mut b);
        assert!(approx_eq_vec(&b, &[3.0, 2.0]));
    }

    // =============================================================================
    // Level 3 BLAS Tests
    // =============================================================================

    #[test]
    fn test_dgemm_basic() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = A * B = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        dgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );
        assert!(approx_eq_vec(&c, &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_dgemm_with_transpose_a() {
        // A = [[1, 3], [2, 4]] (will be transposed to [[1, 2], [3, 4]])
        // B = [[5, 6], [7, 8]]
        // C = A^T * B = [[19, 22], [43, 50]]
        let a = vec![1.0, 3.0, 2.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        dgemm(
            Transpose::Trans,
            Transpose::NoTrans,
            2,
            2,
            2,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );
        assert!(approx_eq_vec(&c, &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_dgemm_with_transpose_b() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 7], [6, 8]] (will be transposed to [[5, 6], [7, 8]])
        // C = A * B^T = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 7.0, 6.0, 8.0];
        let mut c = vec![0.0; 4];
        dgemm(
            Transpose::NoTrans,
            Transpose::Trans,
            2,
            2,
            2,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );
        assert!(approx_eq_vec(&c, &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_dgemm_with_alpha_beta() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // Identity
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![10.0, 20.0, 30.0, 40.0];
        // C = 2 * I * B + 3 * C = 2 * B + 3 * C
        // = [2*1+3*10, 2*2+3*20, 2*3+3*30, 2*4+3*40] = [32, 64, 96, 128]
        dgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            2,
            2.0,
            &a,
            &b,
            3.0,
            &mut c,
        );
        assert!(approx_eq_vec(&c, &[32.0, 64.0, 96.0, 128.0]));
    }

    #[test]
    fn test_dgemm_non_square() {
        // A = [[1, 2, 3], [4, 5, 6]] (2x3)
        // B = [[7, 8], [9, 10], [11, 12]] (3x2)
        // C = A * B (2x2)
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];
        dgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            2,
            2,
            3,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );
        // c[0,0] = 1*7 + 2*9 + 3*11 = 58
        // c[0,1] = 1*8 + 2*10 + 3*12 = 64
        // c[1,0] = 4*7 + 5*9 + 6*11 = 139
        // c[1,1] = 4*8 + 5*10 + 6*12 = 154
        assert!(approx_eq_vec(&c, &[58.0, 64.0, 139.0, 154.0]));
    }

    #[test]
    fn test_dtrsm_left_lower() {
        // L = [[2, 0], [1, 3]]
        // Solve L * X = [[4, 6], [5, 9]]
        // Column 0: [4, 5] -> [2, 1]
        // Column 1: [6, 9] -> [3, 2]
        let l = vec![2.0, 0.0, 1.0, 3.0];
        let mut b = vec![4.0, 6.0, 5.0, 9.0];
        dtrsm(
            Side::Left,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            2,
            2,
            1.0,
            &l,
            &mut b,
        );
        assert!(approx_eq_vec(&b, &[2.0, 3.0, 1.0, 2.0]));
    }

    #[test]
    fn test_dtrsm_with_alpha() {
        // L = [[2, 0], [1, 3]]
        // Solve L * X = 2 * [[4, 6], [5, 9]] = [[8, 12], [10, 18]]
        let l = vec![2.0, 0.0, 1.0, 3.0];
        let mut b = vec![4.0, 6.0, 5.0, 9.0];
        dtrsm(
            Side::Left,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            2,
            2,
            2.0,
            &l,
            &mut b,
        );
        assert!(approx_eq_vec(&b, &[4.0, 6.0, 2.0, 4.0]));
    }

    // =============================================================================
    // LP Integration Tests
    // =============================================================================

    #[test]
    fn test_blas_lp_config() {
        let config = BlasLPConfig::default();
        assert_eq!(config.block_size, BLOCK_SIZE);
        assert!(config.zero_tolerance > 0.0);
        assert!(config.use_pivoting);
    }

    #[test]
    fn test_solve_basis_simple() {
        // B = [[2, 1], [1, 3]]
        // Solve B * x = [5, 7]
        // Expected: x = [8/5, 9/5] = [1.6, 1.8]
        let b = vec![2.0, 1.0, 1.0, 3.0];
        let mut rhs = vec![5.0, 7.0];
        let config = BlasLPConfig::default();
        let success = solve_basis(&b, 2, &mut rhs, &config);
        assert!(success);
        assert!(approx_eq(rhs[0], 1.6));
        assert!(approx_eq(rhs[1], 1.8));
    }

    #[test]
    fn test_identity_operations() {
        // Test that I * x = x
        // Identity matrix in column-major order (BLAS convention)
        let i = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        dgemv(Transpose::NoTrans, 3, 3, 1.0, &i, &x, 0.0, &mut y);
        assert!(approx_eq_vec(&y, &x));
    }

    #[test]
    fn test_large_matrix_blocked() {
        // Test blocked GEMM with a matrix larger than block size
        let n = 100;
        let a: Vec<f64> = (0..n * n).map(|i| (i % 7) as f64).collect();
        let b: Vec<f64> = (0..n * n).map(|i| ((i + 3) % 5) as f64).collect();
        let mut c = vec![0.0; n * n];

        dgemm(
            Transpose::NoTrans,
            Transpose::NoTrans,
            n,
            n,
            n,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );

        // Verify one element manually
        let mut expected = 0.0;
        for k in 0..n {
            expected += a[k] * b[k * n];
        }
        assert!(approx_eq(c[0], expected));
    }
}
