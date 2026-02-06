//! Arithmetic Quantifier Elimination.
//!
//! This module implements quantifier elimination procedures for linear
//! arithmetic over integers and reals.

pub mod cooper;
pub mod ferrante_rackoff;
pub mod omega_test;
pub mod qe_lite_arith;
pub mod virtual_term;

pub use cooper::{CooperEliminator, CooperStats};
pub use ferrante_rackoff::{
    FerranteRackoffEliminator, FerranteRackoffStats, Inequality, InequalityType,
};
pub use omega_test::{
    LinearConstraint as OmegaLinearConstraint, OmegaResult, OmegaTestConfig as OmegaConfig,
    OmegaTestStats as OmegaStats, OmegaTester as OmegaTest,
};
pub use virtual_term::{VirtualTermEliminator, VirtualTermStats};
