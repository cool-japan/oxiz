//! BLAS abstraction layer for large-scale linear programming.
//!
//! This module provides a trait-based abstraction for matrix operations commonly
//! used in linear programming solvers. It includes:
//!
//! - A `BlasOps` trait defining core matrix operations (gemm, gemv, trsm)
//! - A pure Rust implementation (`RustBlas`) for correctness with exact arithmetic
//! - Configurable thresholds for choosing between inline and BLAS operations
//! - Feature-flagged support for external BLAS libraries (placeholder for future)
//!
//! # Example
//!
//! ```
//! use oxiz_math::blas_ops::{BlasOps, RustBlas, BlasConfig};
//! use oxiz_math::matrix::Matrix;
//! use num_rational::Rational64;
//!
//! // Create matrices
//! let a = Matrix::from_rows(vec![
//!     vec![Rational64::from(1), Rational64::from(2)],
//!     vec![Rational64::from(3), Rational64::from(4)],
//! ]);
//! let b = Matrix::from_rows(vec![
//!     vec![Rational64::from(5), Rational64::from(6)],
//!     vec![Rational64::from(7), Rational64::from(8)],
//! ]);
//!
//! // Use BLAS operations
//! let blas = RustBlas::new();
//! let c = blas.gemm(&a, &b);
//! ```

use crate::matrix::Matrix;
use num_rational::Rational64;
use num_traits::{One, Zero};

/// Configuration for BLAS operation thresholds.
///
/// Controls when to use optimized BLAS routines vs. inline operations.
/// For small matrices, inline operations may be faster due to reduced overhead.
#[derive(Debug, Clone)]
pub struct BlasConfig {
    /// Threshold for matrix-matrix multiplication (gemm).
    /// Use BLAS when m * n * k > this threshold.
    pub gemm_threshold: usize,

    /// Threshold for matrix-vector multiplication (gemv).
    /// Use BLAS when m * n > this threshold.
    pub gemv_threshold: usize,

    /// Threshold for triangular solve (trsm).
    /// Use BLAS when n * nrhs > this threshold.
    pub trsm_threshold: usize,

    /// Block size for blocked operations.
    pub block_size: usize,
}

impl Default for BlasConfig {
    fn default() -> Self {
        Self {
            gemm_threshold: 64 * 64 * 64, // 262144
            gemv_threshold: 256 * 256,    // 65536
            trsm_threshold: 128 * 128,    // 16384
            block_size: 64,
        }
    }
}

impl BlasConfig {
    /// Creates a new configuration with custom thresholds.
    pub fn new(gemm_threshold: usize, gemv_threshold: usize, trsm_threshold: usize) -> Self {
        Self {
            gemm_threshold,
            gemv_threshold,
            trsm_threshold,
            block_size: 64,
        }
    }

    /// Configuration optimized for small matrices (always use inline).
    pub fn small_matrix() -> Self {
        Self {
            gemm_threshold: usize::MAX,
            gemv_threshold: usize::MAX,
            trsm_threshold: usize::MAX,
            block_size: 32,
        }
    }

    /// Configuration optimized for large matrices (always use BLAS when available).
    pub fn large_matrix() -> Self {
        Self {
            gemm_threshold: 0,
            gemv_threshold: 0,
            trsm_threshold: 0,
            block_size: 128,
        }
    }
}

/// Transpose option for BLAS operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transpose {
    /// No transpose.
    NoTrans,
    /// Transpose the matrix.
    Trans,
}

/// Side option for triangular solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Solve AX = B (A is on the left).
    Left,
    /// Solve XA = B (A is on the right).
    Right,
}

/// Upper/Lower triangular indicator.
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

/// Trait for BLAS-like operations on rational matrices.
///
/// This trait abstracts matrix operations commonly needed in linear programming:
/// - `gemm`: General matrix-matrix multiplication (C = alpha * A * B + beta * C)
/// - `gemv`: General matrix-vector multiplication (y = alpha * A * x + beta * y)
/// - `trsm`: Triangular solve with multiple right-hand sides
///
/// Implementations maintain exact rational arithmetic for correctness in LP solvers.
pub trait BlasOps {
    /// General matrix-matrix multiplication.
    ///
    /// Computes C = alpha * op(A) * op(B) + beta * C
    ///
    /// # Arguments
    /// * `trans_a` - Whether to transpose A
    /// * `trans_b` - Whether to transpose B
    /// * `alpha` - Scalar multiplier for A*B
    /// * `a` - Matrix A
    /// * `b` - Matrix B
    /// * `beta` - Scalar multiplier for C
    /// * `c` - Output matrix C (modified in place)
    #[allow(clippy::too_many_arguments)]
    fn gemm_into(
        &self,
        trans_a: Transpose,
        trans_b: Transpose,
        alpha: &Rational64,
        a: &Matrix,
        b: &Matrix,
        beta: &Rational64,
        c: &mut Matrix,
    );

    /// General matrix-matrix multiplication (convenience method).
    ///
    /// Computes C = A * B
    fn gemm(&self, a: &Matrix, b: &Matrix) -> Matrix {
        let m = a.nrows();
        let n = b.ncols();
        let mut c = Matrix::zeros(m, n);
        self.gemm_into(
            Transpose::NoTrans,
            Transpose::NoTrans,
            &Rational64::from(1),
            a,
            b,
            &Rational64::zero(),
            &mut c,
        );
        c
    }

    /// General matrix-vector multiplication.
    ///
    /// Computes y = alpha * op(A) * x + beta * y
    ///
    /// # Arguments
    /// * `trans` - Whether to transpose A
    /// * `alpha` - Scalar multiplier for A*x
    /// * `a` - Matrix A
    /// * `x` - Input vector x
    /// * `beta` - Scalar multiplier for y
    /// * `y` - Output vector y (modified in place)
    fn gemv_into(
        &self,
        trans: Transpose,
        alpha: &Rational64,
        a: &Matrix,
        x: &[Rational64],
        beta: &Rational64,
        y: &mut [Rational64],
    );

    /// General matrix-vector multiplication (convenience method).
    ///
    /// Computes y = A * x
    fn gemv(&self, a: &Matrix, x: &[Rational64]) -> Vec<Rational64> {
        let m = a.nrows();
        let mut y = vec![Rational64::zero(); m];
        self.gemv_into(
            Transpose::NoTrans,
            &Rational64::from(1),
            a,
            x,
            &Rational64::zero(),
            &mut y,
        );
        y
    }

    /// Triangular solve with multiple right-hand sides.
    ///
    /// Solves op(A) * X = alpha * B (Side::Left) or X * op(A) = alpha * B (Side::Right)
    /// where A is triangular.
    ///
    /// # Arguments
    /// * `side` - Whether A is on the left or right
    /// * `uplo` - Whether A is upper or lower triangular
    /// * `trans` - Whether to transpose A
    /// * `diag` - Whether A has unit diagonal
    /// * `alpha` - Scalar multiplier
    /// * `a` - Triangular matrix A
    /// * `b` - Right-hand side B (modified in place to contain solution X)
    #[allow(clippy::too_many_arguments)]
    fn trsm(
        &self,
        side: Side,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        alpha: &Rational64,
        a: &Matrix,
        b: &mut Matrix,
    );

    /// Triangular solve for a single vector.
    ///
    /// Solves op(A) * x = b where A is triangular.
    ///
    /// # Arguments
    /// * `uplo` - Whether A is upper or lower triangular
    /// * `trans` - Whether to transpose A
    /// * `diag` - Whether A has unit diagonal
    /// * `a` - Triangular matrix A
    /// * `b` - Right-hand side b (modified in place to contain solution x)
    fn trsv(&self, uplo: UpLo, trans: Transpose, diag: Diag, a: &Matrix, b: &mut [Rational64]);

    /// Returns the configuration for this BLAS implementation.
    fn config(&self) -> &BlasConfig;

    /// Checks if BLAS should be used for gemm based on matrix dimensions.
    fn should_use_blas_gemm(&self, m: usize, n: usize, k: usize) -> bool {
        m * n * k >= self.config().gemm_threshold
    }

    /// Checks if BLAS should be used for gemv based on matrix dimensions.
    fn should_use_blas_gemv(&self, m: usize, n: usize) -> bool {
        m * n >= self.config().gemv_threshold
    }

    /// Checks if BLAS should be used for trsm based on matrix dimensions.
    fn should_use_blas_trsm(&self, n: usize, nrhs: usize) -> bool {
        n * nrhs >= self.config().trsm_threshold
    }
}

/// Pure Rust implementation of BLAS operations.
///
/// This implementation provides exact rational arithmetic operations,
/// ensuring correctness for linear programming solvers that require
/// exact solutions.
#[derive(Debug, Clone)]
pub struct RustBlas {
    config: BlasConfig,
}

impl Default for RustBlas {
    fn default() -> Self {
        Self::new()
    }
}

impl RustBlas {
    /// Creates a new RustBlas instance with default configuration.
    pub fn new() -> Self {
        Self {
            config: BlasConfig::default(),
        }
    }

    /// Creates a new RustBlas instance with custom configuration.
    pub fn with_config(config: BlasConfig) -> Self {
        Self { config }
    }

    /// Internal helper for matrix dimensions after transpose.
    fn get_dims(m: &Matrix, trans: Transpose) -> (usize, usize) {
        match trans {
            Transpose::NoTrans => (m.nrows(), m.ncols()),
            Transpose::Trans => (m.ncols(), m.nrows()),
        }
    }

    /// Internal helper for accessing matrix element with transpose.
    fn get_elem(m: &Matrix, i: usize, j: usize, trans: Transpose) -> Rational64 {
        match trans {
            Transpose::NoTrans => m.get(i, j),
            Transpose::Trans => m.get(j, i),
        }
    }
}

#[allow(clippy::needless_range_loop)]
impl BlasOps for RustBlas {
    fn gemm_into(
        &self,
        trans_a: Transpose,
        trans_b: Transpose,
        alpha: &Rational64,
        a: &Matrix,
        b: &Matrix,
        beta: &Rational64,
        c: &mut Matrix,
    ) {
        let (m, k_a) = Self::get_dims(a, trans_a);
        let (k_b, n) = Self::get_dims(b, trans_b);

        assert_eq!(
            k_a, k_b,
            "Inner dimensions must match for matrix multiplication"
        );
        assert_eq!(c.nrows(), m, "Output matrix row dimension mismatch");
        assert_eq!(c.ncols(), n, "Output matrix column dimension mismatch");

        let k = k_a;

        // C = alpha * op(A) * op(B) + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum = Rational64::zero();
                for l in 0..k {
                    let a_il = Self::get_elem(a, i, l, trans_a);
                    let b_lj = Self::get_elem(b, l, j, trans_b);
                    sum += a_il * b_lj;
                }
                let c_ij = c.get(i, j);
                c.set(i, j, alpha * sum + beta * c_ij);
            }
        }
    }

    fn gemv_into(
        &self,
        trans: Transpose,
        alpha: &Rational64,
        a: &Matrix,
        x: &[Rational64],
        beta: &Rational64,
        y: &mut [Rational64],
    ) {
        let (m, n) = Self::get_dims(a, trans);

        assert_eq!(x.len(), n, "Vector x dimension must match matrix columns");
        assert_eq!(y.len(), m, "Vector y dimension must match matrix rows");

        // y = alpha * op(A) * x + beta * y
        for i in 0..m {
            let mut sum = Rational64::zero();
            for j in 0..n {
                let a_ij = Self::get_elem(a, i, j, trans);
                sum += a_ij * x[j];
            }
            y[i] = alpha * sum + beta * y[i];
        }
    }

    fn trsm(
        &self,
        side: Side,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        alpha: &Rational64,
        a: &Matrix,
        b: &mut Matrix,
    ) {
        let n = a.nrows();
        assert_eq!(a.nrows(), a.ncols(), "Triangular matrix must be square");

        match side {
            Side::Left => {
                assert_eq!(b.nrows(), n, "B row dimension must match A");
                self.trsm_left(uplo, trans, diag, alpha, a, b);
            }
            Side::Right => {
                assert_eq!(b.ncols(), n, "B column dimension must match A");
                self.trsm_right(uplo, trans, diag, alpha, a, b);
            }
        }
    }

    fn trsv(&self, uplo: UpLo, trans: Transpose, diag: Diag, a: &Matrix, b: &mut [Rational64]) {
        let n = a.nrows();
        assert_eq!(a.nrows(), a.ncols(), "Triangular matrix must be square");
        assert_eq!(b.len(), n, "Vector dimension must match matrix");

        match (uplo, trans) {
            (UpLo::Lower, Transpose::NoTrans) | (UpLo::Upper, Transpose::Trans) => {
                // Forward substitution
                for i in 0..n {
                    let mut sum = b[i];
                    for j in 0..i {
                        let a_ij = Self::get_elem(a, i, j, trans);
                        sum -= a_ij * b[j];
                    }
                    if diag == Diag::NonUnit {
                        let a_ii = Self::get_elem(a, i, i, trans);
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
                        let a_ij = Self::get_elem(a, i, j, trans);
                        sum -= a_ij * b[j];
                    }
                    if diag == Diag::NonUnit {
                        let a_ii = Self::get_elem(a, i, i, trans);
                        b[i] = sum / a_ii;
                    } else {
                        b[i] = sum;
                    }
                }
            }
        }
    }

    fn config(&self) -> &BlasConfig {
        &self.config
    }
}

impl RustBlas {
    /// Triangular solve with B on the left: op(A) * X = alpha * B
    fn trsm_left(
        &self,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        alpha: &Rational64,
        a: &Matrix,
        b: &mut Matrix,
    ) {
        let n = a.nrows();
        let nrhs = b.ncols();

        // Scale B by alpha first
        if !alpha.is_one() {
            for i in 0..n {
                for j in 0..nrhs {
                    let val = b.get(i, j) * alpha;
                    b.set(i, j, val);
                }
            }
        }

        // Solve column by column
        for col in 0..nrhs {
            match (uplo, trans) {
                (UpLo::Lower, Transpose::NoTrans) | (UpLo::Upper, Transpose::Trans) => {
                    // Forward substitution
                    for i in 0..n {
                        let mut sum = b.get(i, col);
                        for j in 0..i {
                            let a_ij = Self::get_elem(a, i, j, trans);
                            sum -= a_ij * b.get(j, col);
                        }
                        if diag == Diag::NonUnit {
                            let a_ii = Self::get_elem(a, i, i, trans);
                            b.set(i, col, sum / a_ii);
                        } else {
                            b.set(i, col, sum);
                        }
                    }
                }
                (UpLo::Upper, Transpose::NoTrans) | (UpLo::Lower, Transpose::Trans) => {
                    // Backward substitution
                    for i in (0..n).rev() {
                        let mut sum = b.get(i, col);
                        for j in (i + 1)..n {
                            let a_ij = Self::get_elem(a, i, j, trans);
                            sum -= a_ij * b.get(j, col);
                        }
                        if diag == Diag::NonUnit {
                            let a_ii = Self::get_elem(a, i, i, trans);
                            b.set(i, col, sum / a_ii);
                        } else {
                            b.set(i, col, sum);
                        }
                    }
                }
            }
        }
    }

    /// Triangular solve with B on the right: X * op(A) = alpha * B
    fn trsm_right(
        &self,
        uplo: UpLo,
        trans: Transpose,
        diag: Diag,
        alpha: &Rational64,
        a: &Matrix,
        b: &mut Matrix,
    ) {
        let n = a.nrows();
        let m = b.nrows();

        // Scale B by alpha first
        if !alpha.is_one() {
            for i in 0..m {
                for j in 0..n {
                    let val = b.get(i, j) * alpha;
                    b.set(i, j, val);
                }
            }
        }

        // Solve row by row
        for row in 0..m {
            match (uplo, trans) {
                (UpLo::Upper, Transpose::NoTrans) | (UpLo::Lower, Transpose::Trans) => {
                    // Forward substitution on columns
                    for j in 0..n {
                        let mut sum = b.get(row, j);
                        for k in 0..j {
                            let a_kj = Self::get_elem(a, k, j, trans);
                            sum -= b.get(row, k) * a_kj;
                        }
                        if diag == Diag::NonUnit {
                            let a_jj = Self::get_elem(a, j, j, trans);
                            b.set(row, j, sum / a_jj);
                        } else {
                            b.set(row, j, sum);
                        }
                    }
                }
                (UpLo::Lower, Transpose::NoTrans) | (UpLo::Upper, Transpose::Trans) => {
                    // Backward substitution on columns
                    for j in (0..n).rev() {
                        let mut sum = b.get(row, j);
                        for k in (j + 1)..n {
                            let a_kj = Self::get_elem(a, k, j, trans);
                            sum -= b.get(row, k) * a_kj;
                        }
                        if diag == Diag::NonUnit {
                            let a_jj = Self::get_elem(a, j, j, trans);
                            b.set(row, j, sum / a_jj);
                        } else {
                            b.set(row, j, sum);
                        }
                    }
                }
            }
        }
    }
}

/// Global BLAS instance for convenient access.
///
/// This provides a thread-local instance of `RustBlas` for cases where
/// you don't want to manage the BLAS instance explicitly.
pub fn get_blas() -> RustBlas {
    RustBlas::new()
}

/// Configurable BLAS instance with custom settings.
pub fn get_blas_with_config(config: BlasConfig) -> RustBlas {
    RustBlas::with_config(config)
}

// Placeholder for feature-gated external BLAS support
// #[cfg(feature = "external-blas")]
// pub struct ExternalBlas {
//     config: BlasConfig,
// }
//
// #[cfg(feature = "external-blas")]
// impl BlasOps for ExternalBlas {
//     // Implementation would delegate to external BLAS library
//     // while converting between Rational64 and f64 for the BLAS calls
// }

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> Rational64 {
        Rational64::from(n)
    }

    #[test]
    fn test_gemm_basic() {
        let blas = RustBlas::new();

        let a = Matrix::from_rows(vec![vec![rat(1), rat(2)], vec![rat(3), rat(4)]]);
        let b = Matrix::from_rows(vec![vec![rat(5), rat(6)], vec![rat(7), rat(8)]]);

        let c = blas.gemm(&a, &b);

        // C = A * B
        // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        assert_eq!(c.get(0, 0), rat(19));
        assert_eq!(c.get(0, 1), rat(22));
        assert_eq!(c.get(1, 0), rat(43));
        assert_eq!(c.get(1, 1), rat(50));
    }

    #[test]
    fn test_gemm_with_transpose() {
        let blas = RustBlas::new();

        let a = Matrix::from_rows(vec![vec![rat(1), rat(3)], vec![rat(2), rat(4)]]);
        let b = Matrix::from_rows(vec![vec![rat(5), rat(7)], vec![rat(6), rat(8)]]);

        // C = A^T * B^T
        let mut c = Matrix::zeros(2, 2);
        blas.gemm_into(
            Transpose::Trans,
            Transpose::Trans,
            &rat(1),
            &a,
            &b,
            &rat(0),
            &mut c,
        );

        // A^T = [1 2]    B^T = [5 6]
        //       [3 4]          [7 8]
        // Same result as test_gemm_basic
        assert_eq!(c.get(0, 0), rat(19));
        assert_eq!(c.get(0, 1), rat(22));
        assert_eq!(c.get(1, 0), rat(43));
        assert_eq!(c.get(1, 1), rat(50));
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        let blas = RustBlas::new();

        let a = Matrix::from_rows(vec![vec![rat(1), rat(2)], vec![rat(3), rat(4)]]);
        let b = Matrix::from_rows(vec![vec![rat(1), rat(0)], vec![rat(0), rat(1)]]);
        let mut c = Matrix::from_rows(vec![vec![rat(10), rat(20)], vec![rat(30), rat(40)]]);

        // C = 2 * A * B + 3 * C
        blas.gemm_into(
            Transpose::NoTrans,
            Transpose::NoTrans,
            &rat(2),
            &a,
            &b,
            &rat(3),
            &mut c,
        );

        // A * B = A (identity)
        // C = 2 * A + 3 * C_old
        // c[0,0] = 2*1 + 3*10 = 32
        // c[0,1] = 2*2 + 3*20 = 64
        // c[1,0] = 2*3 + 3*30 = 96
        // c[1,1] = 2*4 + 3*40 = 128
        assert_eq!(c.get(0, 0), rat(32));
        assert_eq!(c.get(0, 1), rat(64));
        assert_eq!(c.get(1, 0), rat(96));
        assert_eq!(c.get(1, 1), rat(128));
    }

    #[test]
    fn test_gemv_basic() {
        let blas = RustBlas::new();

        let a = Matrix::from_rows(vec![vec![rat(1), rat(2)], vec![rat(3), rat(4)]]);
        let x = vec![rat(5), rat(6)];

        let y = blas.gemv(&a, &x);

        // y = A * x
        // [1 2] * [5] = [1*5+2*6] = [17]
        // [3 4]   [6]   [3*5+4*6]   [39]
        assert_eq!(y[0], rat(17));
        assert_eq!(y[1], rat(39));
    }

    #[test]
    fn test_gemv_with_transpose() {
        let blas = RustBlas::new();

        let a = Matrix::from_rows(vec![vec![rat(1), rat(3)], vec![rat(2), rat(4)]]);
        let x = vec![rat(5), rat(6)];
        let mut y = vec![rat(0), rat(0)];

        // y = A^T * x
        blas.gemv_into(Transpose::Trans, &rat(1), &a, &x, &rat(0), &mut y);

        // A^T = [1 2]
        //       [3 4]
        // Same as test_gemv_basic
        assert_eq!(y[0], rat(17));
        assert_eq!(y[1], rat(39));
    }

    #[test]
    fn test_trsv_lower() {
        let blas = RustBlas::new();

        // Lower triangular matrix
        let l = Matrix::from_rows(vec![
            vec![rat(2), rat(0), rat(0)],
            vec![rat(1), rat(3), rat(0)],
            vec![rat(1), rat(2), rat(4)],
        ]);

        // Solve L * x = b where b = [4, 10, 21]
        // Expected: x = [2, 2.67, 3] approximately, but let's compute exactly
        // x[0] = 4/2 = 2
        // x[1] = (10 - 1*2)/3 = 8/3
        // x[2] = (21 - 1*2 - 2*(8/3))/4 = (21 - 2 - 16/3)/4 = (57/3 - 16/3)/4 = (41/3)/4 = 41/12
        let mut b = vec![rat(4), rat(10), rat(21)];
        blas.trsv(UpLo::Lower, Transpose::NoTrans, Diag::NonUnit, &l, &mut b);

        assert_eq!(b[0], rat(2));
        assert_eq!(b[1], Rational64::new(8, 3));
        assert_eq!(b[2], Rational64::new(41, 12));
    }

    #[test]
    fn test_trsv_upper() {
        let blas = RustBlas::new();

        // Upper triangular matrix
        let u = Matrix::from_rows(vec![
            vec![rat(2), rat(1), rat(1)],
            vec![rat(0), rat(3), rat(2)],
            vec![rat(0), rat(0), rat(4)],
        ]);

        // Solve U * x = b where b = [8, 14, 12]
        // x[2] = 12/4 = 3
        // x[1] = (14 - 2*3)/3 = 8/3
        // x[0] = (8 - 1*(8/3) - 1*3)/2 = (8 - 8/3 - 3)/2 = (24/3 - 8/3 - 9/3)/2 = (7/3)/2 = 7/6
        let mut b = vec![rat(8), rat(14), rat(12)];
        blas.trsv(UpLo::Upper, Transpose::NoTrans, Diag::NonUnit, &u, &mut b);

        assert_eq!(b[2], rat(3));
        assert_eq!(b[1], Rational64::new(8, 3));
        assert_eq!(b[0], Rational64::new(7, 6));
    }

    #[test]
    fn test_trsv_unit_diagonal() {
        let blas = RustBlas::new();

        // Lower triangular with unit diagonal (diagonal values ignored)
        let l = Matrix::from_rows(vec![
            vec![rat(999), rat(0)], // 999 should be ignored
            vec![rat(2), rat(888)], // 888 should be ignored
        ]);

        // Solve L * x = b where L has unit diagonal
        // x[0] = b[0] = 4
        // x[1] = b[1] - 2*x[0] = 10 - 8 = 2
        let mut b = vec![rat(4), rat(10)];
        blas.trsv(UpLo::Lower, Transpose::NoTrans, Diag::Unit, &l, &mut b);

        assert_eq!(b[0], rat(4));
        assert_eq!(b[1], rat(2));
    }

    #[test]
    fn test_trsm_left_lower() {
        let blas = RustBlas::new();

        // Lower triangular matrix
        let l = Matrix::from_rows(vec![vec![rat(2), rat(0)], vec![rat(1), rat(3)]]);

        // Solve L * X = B
        let mut b = Matrix::from_rows(vec![vec![rat(4), rat(6)], vec![rat(7), rat(12)]]);

        blas.trsm(
            Side::Left,
            UpLo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            &rat(1),
            &l,
            &mut b,
        );

        // Column 0: L * x = [4, 7]^T
        // x[0] = 4/2 = 2
        // x[1] = (7 - 1*2)/3 = 5/3
        assert_eq!(b.get(0, 0), rat(2));
        assert_eq!(b.get(1, 0), Rational64::new(5, 3));

        // Column 1: L * x = [6, 12]^T
        // x[0] = 6/2 = 3
        // x[1] = (12 - 1*3)/3 = 3
        assert_eq!(b.get(0, 1), rat(3));
        assert_eq!(b.get(1, 1), rat(3));
    }

    #[test]
    fn test_config_thresholds() {
        let config = BlasConfig::new(100, 50, 25);
        let blas = RustBlas::with_config(config);

        // Test threshold checks
        assert!(!blas.should_use_blas_gemm(3, 3, 3)); // 27 < 100
        assert!(blas.should_use_blas_gemm(5, 5, 5)); // 125 >= 100

        assert!(!blas.should_use_blas_gemv(5, 5)); // 25 < 50
        assert!(blas.should_use_blas_gemv(8, 8)); // 64 >= 50

        assert!(!blas.should_use_blas_trsm(4, 4)); // 16 < 25
        assert!(blas.should_use_blas_trsm(5, 5)); // 25 >= 25
    }

    #[test]
    fn test_config_presets() {
        let small = BlasConfig::small_matrix();
        assert_eq!(small.gemm_threshold, usize::MAX);

        let large = BlasConfig::large_matrix();
        assert_eq!(large.gemm_threshold, 0);
    }

    #[test]
    fn test_identity_multiplication() {
        let blas = RustBlas::new();

        let a = Matrix::from_rows(vec![
            vec![rat(1), rat(2), rat(3)],
            vec![rat(4), rat(5), rat(6)],
            vec![rat(7), rat(8), rat(9)],
        ]);
        let i = Matrix::identity(3);

        // A * I = A
        let result = blas.gemm(&a, &i);
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(result.get(row, col), a.get(row, col));
            }
        }

        // I * A = A
        let result2 = blas.gemm(&i, &a);
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(result2.get(row, col), a.get(row, col));
            }
        }
    }

    #[test]
    fn test_gemm_non_square() {
        let blas = RustBlas::new();

        // 2x3 * 3x2 = 2x2
        let a = Matrix::from_rows(vec![
            vec![rat(1), rat(2), rat(3)],
            vec![rat(4), rat(5), rat(6)],
        ]);
        let b = Matrix::from_rows(vec![
            vec![rat(7), rat(8)],
            vec![rat(9), rat(10)],
            vec![rat(11), rat(12)],
        ]);

        let c = blas.gemm(&a, &b);

        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);

        // c[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        // c[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        // c[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        // c[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
        assert_eq!(c.get(0, 0), rat(58));
        assert_eq!(c.get(0, 1), rat(64));
        assert_eq!(c.get(1, 0), rat(139));
        assert_eq!(c.get(1, 1), rat(154));
    }

    #[test]
    fn test_get_blas_functions() {
        let blas1 = get_blas();
        assert_eq!(
            blas1.config().gemm_threshold,
            BlasConfig::default().gemm_threshold
        );

        let config = BlasConfig::small_matrix();
        let blas2 = get_blas_with_config(config);
        assert_eq!(blas2.config().gemm_threshold, usize::MAX);
    }
}
