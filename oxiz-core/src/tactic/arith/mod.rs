//! Arithmetic Tactics.
//!
//! This module provides tactics for simplifying and normalizing arithmetic
//! constraints before they reach the theory solver.

pub mod arith_bounds;
pub mod card2bv;
pub mod factor;
pub mod fm_advanced;
// TODO: Implement normalize_bounds module
// pub mod normalize_bounds;
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
// TODO: Uncomment when normalize_bounds is implemented
// pub use normalize_bounds::{
//     BoundType, NormalizedBounds, NormalizeBoundsConfig, NormalizeBoundsStats,
//     NormalizeBoundsTactic,
// };
pub use propagate_ineqs::{
    Bound, BoundKind, PropagateIneqsConfig, PropagateIneqsError, PropagateIneqsStats,
    PropagateIneqsTactic,
};
