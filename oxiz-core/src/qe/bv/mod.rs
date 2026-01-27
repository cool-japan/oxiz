//! BitVector Quantifier Elimination.
//!
//! This module provides quantifier elimination for bitvector formulas.

pub mod plugin;

pub use plugin::{BvConstraint, BvQeConfig, BvQePlugin, BvQeStats};
