//! BitVector Quantifier Elimination.
//!
//! This module provides quantifier elimination for bitvector formulas.

pub mod elimination_strategies;
pub mod plugin;
pub mod simplification;

pub use elimination_strategies::{
    BvEliminationConfig, BvEliminationEngine, BvEliminationStats, EliminationStrategy,
};
pub use plugin::{BvConstraint, BvQeConfig, BvQePlugin, BvQeStats};
pub use simplification::{BvSimplificationConfig, BvSimplificationStats, BvSimplifier, BvTerm};
