//! SIMD-Accelerated Mathematical Operations.
//!
//! Provides vectorized implementations of common mathematical operations
//! using Rust's portable SIMD support.

pub mod matrix_simd;
pub mod polynomial_simd;
pub mod simplex_simd;
pub mod vector_ops;

pub use matrix_simd::{
    simd_determinant, simd_lu_decomposition, simd_lu_solve, simd_matrix_inverse, simd_matrix_mul,
    simd_matrix_vec_mul as matrix_vec_mul_enhanced, simd_qr_decomposition, transpose,
};
pub use polynomial_simd::{simd_poly_add, simd_poly_eval, simd_poly_mul};
pub use simplex_simd::{
    SimplexError, SimplexSolution, SimplexTableau, simd_dual_simplex, simd_simplex_solve,
};
pub use vector_ops::{simd_dot_product, simd_matrix_vec_mul, simd_norm_squared, simd_sum};
