//! Parallel SAT Solver Components.
//!
//! Provides parallel clause simplification, portfolio solving, and
//! parallel proof checking.

pub mod clause_simplify;
pub mod portfolio;
pub mod proof_check;

pub use clause_simplify::{ParallelClauseSimplifier, SimplificationConfig, SimplificationResult};
pub use portfolio::{PortfolioConfig, PortfolioResult, PortfolioSolver, SolverVariant};
pub use proof_check::{ParallelProofChecker, ProofCheckConfig, ProofCheckResult};
