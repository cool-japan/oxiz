//! String theory quantifier elimination.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod constraint_solver;
pub mod plugin;

pub use constraint_solver::{
    ConcatConstraint, ContainsConstraint, LengthBound, RegexConstraint, RegexPattern,
    StringConstraintSolver, StringSolverStats,
};
pub use plugin::{
    LengthOp, StringConstraint, StringQeConfig, StringQePlugin, StringQeStats, VarId as StringVarId,
};
