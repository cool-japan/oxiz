//! BitVector Tactics.
//!
//! This module provides tactics for simplifying and normalizing bitvector
//! constraints before they reach the SAT solver.

pub mod bv1_blast;
pub mod bv_bounds;
pub mod bvarray2uf;
pub mod dt2bv;

pub use bv_bounds::{BvBoundsConfig, BvBoundsStats, BvBoundsTactic, Interval as BvInterval};
pub use bv1_blast::{Bv1BlastConfig, Bv1BlastStats, Bv1BlastTactic};
pub use bvarray2uf::{BvArray2UfConfig, BvArray2UfStats, BvArray2UfTactic};
