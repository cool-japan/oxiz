//! BitVector Tactics.
//!
//! This module provides tactics for simplifying and normalizing bitvector
//! constraints before they reach the SAT solver.

pub mod advanced_rewriter;
// TODO: Implement bounds_propagation module
// pub mod bounds_propagation;
pub mod bv1_blast;
pub mod bv_bounds;
pub mod bv_rewriter;
pub mod bvarray2uf;
pub mod dt2bv;

pub use advanced_rewriter::{
    AdvancedBvRewriter, BvOp, Pattern, RewriterConfig as AdvancedRewriterConfig,
    RewriterStats as AdvancedRewriterStats,
};
// TODO: Uncomment when bounds_propagation is implemented
// pub use bounds_propagation::{
//     BoundsConfig, BoundsPropagationTactic, BoundsStats, Interval as BvPropInterval,
// };
pub use bv_bounds::{BvBoundsConfig, BvBoundsStats, BvBoundsTactic, Interval as BvInterval};
pub use bv_rewriter::{BvRewriterConfig, BvRewriterStats, BvRewriterTactic};
pub use bv1_blast::{Bv1BlastConfig, Bv1BlastStats, Bv1BlastTactic};
pub use bvarray2uf::{BvArray2UfConfig, BvArray2UfStats, BvArray2UfTactic};
