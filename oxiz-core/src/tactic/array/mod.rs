//! Array Theory Tactics.
//!
//! Tactics for simplifying and normalizing array constraints.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod array_bounds;
pub mod select_store_elim;

pub use array_bounds::{ArrayBoundsConfig, ArrayBoundsStats, ArrayBoundsTactic, IndexBound};
pub use select_store_elim::{SelectStoreElimStats, SelectStoreElimTactic, StoreChain};
