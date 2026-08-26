//! Linear Arithmetic theory solver
//!
//! Implements the Dual Simplex algorithm for Linear Real Arithmetic (LRA)
//! with extensions for Linear Integer Arithmetic (LIA).

#[allow(unused_imports)]
use crate::prelude::*;

mod delta;
mod gaussian;
mod lia;
// Nonlinear integer arithmetic over the linear relaxation (NIA-over-LP). The
// module documents itself; this declaration is gated on `std` because the
// witness type and the exact re-verification `nla` performs before returning
// `sat` both live in `crate::nl_eval`, which is itself std-only.
#[cfg(feature = "std")]
pub mod nla;
mod optimize;
mod simplex;
mod simplex_opt;
mod solver;

pub use delta::DeltaRational;
pub use gaussian::{GaussianElimination, LinearEquation};
pub use lia::{HermiteNormalForm, LiaSolver, PseudoBooleanSolver};
#[cfg(feature = "std")]
pub use nla::{NlaConfig, NlaVerdict, check_assertions as check_nonlinear};
pub use optimize::{
    ConstraintSense, LraOptimizer, Objective, ObjectiveBuilder, OptModel, OptResult,
};
pub use simplex::{LinExpr, Simplex, VarId};
pub use simplex_opt::SimplexOptStatus;
pub use solver::ArithSolver;
