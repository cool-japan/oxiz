//! Datatype Quantifier Elimination.
//!
//! This module provides quantifier elimination for algebraic datatype formulas.

pub mod plugin;

pub use plugin::{
    Constructor, Datatype, DatatypeConstraint, DatatypeQeConfig, DatatypeQePlugin, DatatypeQeStats,
};
