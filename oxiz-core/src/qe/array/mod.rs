//! Array Quantifier Elimination.

pub mod index_abstraction;
pub mod plugin;

pub use plugin::{ArrayConstraint, ArrayId, ArrayQeConfig, ArrayQePlugin, ArrayQeStats, IndexId};
