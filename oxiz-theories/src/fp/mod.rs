//! Floating-Point Theory Solver (QF_FP)
//!
//! Implements IEEE 754 floating-point arithmetic for SMT solving.
//! This module provides support for:
//!
//! - Standard floating-point formats (16, 32, 64, 128-bit)
//! - IEEE 754 operations (add, sub, mul, div, fma, sqrt, rem)
//! - Rounding modes (RNE, RNA, RTP, RTN, RTZ)
//! - Special values (NaN, +Inf, -Inf, +0, -0)
//! - Comparisons and predicates
//!
//! ## Implementation Strategy
//!
//! The solver uses bit-blasting to encode floating-point operations
//! into SAT constraints. This is the most common approach for
//! floating-point SMT solving (used by MathSAT, CVC5, etc.).

mod solver;

pub use solver::{FpFormat, FpRoundingMode, FpSolver, FpValue};
