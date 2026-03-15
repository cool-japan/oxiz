//! Array Quantifier Elimination.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod index_abstraction;
pub mod plugin;
pub mod quantifier_elim;

pub use plugin::{ArrayConstraint, ArrayId, ArrayQeConfig, ArrayQePlugin, ArrayQeStats, IndexId};
pub use quantifier_elim::{
    ArrayProperty, ArrayQeConfig as ArrayQeConfig2, ArrayQeStats as ArrayQeStats2,
    ArrayQuantifierEliminator, ArrayTerm, IndexConstraint, IndexSet,
};
