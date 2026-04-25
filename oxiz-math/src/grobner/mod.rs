//! Gröbner Basis Computation
//!
//! This module provides advanced algorithms for computing Gröbner bases,
//! including Buchberger's algorithm with product/chain criteria, F4 algorithm,
//! and syzygy computation.

#[allow(unused_imports)]
use crate::prelude::*;

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

pub use f4::{
    CriticalPair as F4CriticalPair, F4Algorithm, F4Config, F4Stats,
    Monomial as F4Monomial, Polynomial as F4Polynomial, Term as F4Term,
};

pub use syzygy::{
    BuchbergerCriteria, CriticalPair as SyzygyCriticalPair, Syzygy,
    SyzygyComputer, SyzygyStats,
};
