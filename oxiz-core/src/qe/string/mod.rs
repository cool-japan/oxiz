//! String theory quantifier elimination.

pub mod constraint_solver;
pub mod plugin;

pub use constraint_solver::{
    ConcatConstraint, ContainsConstraint, LengthBound, RegexConstraint, RegexPattern,
    StringConstraintSolver, StringSolverStats,
};
pub use plugin::{
    LengthOp, StringConstraint, StringQeConfig, StringQePlugin, StringQeStats, VarId as StringVarId,
};
