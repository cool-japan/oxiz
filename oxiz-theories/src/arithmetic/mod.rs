//! Linear Arithmetic theory solver
//!
//! Implements the Dual Simplex algorithm for Linear Real Arithmetic (LRA)
//! with extensions for Linear Integer Arithmetic (LIA).

mod delta;
mod gaussian;
mod lia;
mod optimize;
mod simplex;
mod solver;

pub use delta::DeltaRational;
pub use gaussian::{GaussianElimination, LinearEquation};
pub use lia::{HermiteNormalForm, LiaSolver, PseudoBooleanSolver};
pub use optimize::{
    ConstraintSense, LraOptimizer, Objective, ObjectiveBuilder, OptModel, OptResult,
};
pub use simplex::Simplex;
pub use solver::ArithSolver;
