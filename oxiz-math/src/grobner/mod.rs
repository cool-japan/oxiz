//! Gröbner Basis Computation
//!
//! This module provides advanced algorithms for computing Gröbner bases,
//! including Buchberger's algorithm with product/chain criteria, F4 algorithm,
//! and syzygy computation.

pub mod buchberger;
pub mod buchberger_enhanced;
pub mod f4;
pub mod syzygy;

// Re-export standard Buchberger algorithm and NRA solver
pub use buchberger::{
    NraSolver, SatResult, grobner_basis, grobner_basis_f4, grobner_basis_f5, ideal_membership,
    reduce, s_polynomial,
};

pub use buchberger_enhanced::{
    BuchbergerConfig, BuchbergerStats, CriticalPair, EnhancedBuchberger, Monomial,
    Polynomial as BuchbergerPolynomial, Term,
};

// TODO: Export f4 types when module exports are finalized
// pub use f4::{F4Config, F4Solver, F4Stats, Matrix, Polynomial as F4Polynomial};

// TODO: Export syzygy types when module exports are finalized
// pub use syzygy::{
//     Module, ModuleElement, SyzygyComputation, SyzygyConfig, SyzygyStats,
//     Vector as SyzygyVector,
// };
