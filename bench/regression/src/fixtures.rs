//! Embedded SMT-LIB2 fixture content for benchmarks and tests.
//!
//! Each constant embeds a fixture file at compile time using `include_str!`.
//! The fixtures cover four additional logics beyond the ones already
//! exercised by the hand-written criterion benchmarks:
//! * QF_BV   – bit-vector arithmetic
//! * QF_LRA  – linear real arithmetic
//! * QF_AUFLIA – arrays with integer indices
//!
//! There is no standalone SMT-LIB2 MaxSAT fixture because the `.wcnf`
//! file in `benchmarks/maxsat_simple.wcnf` uses DIMACS WCNF format rather
//! than SMT-LIB2 syntax.  MaxSAT coverage is therefore kept in the
//! existing `run_maxsat_benchmarks` harness in `src/benchmarks.rs`.

/// Bit-vector benchmark (QF_BV): basic bvadd / bvand / bvor / bvxor.
pub const BV_SIMPLE: &str = include_str!("../benchmarks/bv_simple.smt2");

/// Linear real arithmetic benchmark (QF_LRA): range constraints with rationals.
pub const LRA_SIMPLE: &str = include_str!("../benchmarks/lra_simple.smt2");

/// Array theory benchmark (QF_AUFLIA): select / store round-trip.
pub const ARRAYS_SIMPLE: &str = include_str!("../benchmarks/arrays_simple.smt2");
