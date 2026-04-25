//! Arithmetic Tactics.
//!
//! This module provides tactics for simplifying and normalizing arithmetic
//! constraints before they reach the theory solver.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod arith_bounds;
pub mod card2bv;
pub mod factor;
pub mod fm_advanced;
pub mod normalize_bounds;
pub mod propagate_ineqs;

pub use card2bv::{
    BvTerm, BvVar, Card2BvConfig, Card2BvStats, Card2BvTactic, CardinalityConstraint,
    EncodingMetadata, EncodingResult, EncodingStrategy,
};
pub use factor::{FactorTactic, FactorTacticConfig, FactorTacticStats, Monomial, Polynomial};
pub use fm_advanced::{
    ConstraintSystem, FmAdvancedConfig, FmAdvancedStats, FmAdvancedTactic,
    LinearInequality as FmLinearInequality,
};
pub use normalize_bounds::{
    BoundType, NormalizeBoundsConfig, NormalizeBoundsStats, NormalizeBoundsTactic, NormalizedBounds,
};
pub use propagate_ineqs::{
    Bound, BoundKind, PropagateIneqsConfig, PropagateIneqsError, PropagateIneqsStats,
    PropagateIneqsTactic,
};
