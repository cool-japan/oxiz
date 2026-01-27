//! Arithmetic Quantifier Elimination.
//!
//! This module implements quantifier elimination procedures for linear
//! arithmetic over integers and reals.

pub mod cooper;
pub mod ferrante_rackoff;
pub mod omega_test;
pub mod qe_lite_arith;
pub mod virtual_term;

pub use cooper::{Constraint, CooperConfig, CooperQE, CooperStats, DnfFormula};
pub use ferrante_rackoff::{
    Conjunction as FrConjunction, DnfFormula as FrDnfFormula, FerranteRackoffConfig,
    FerranteRackoffQE, FerranteRackoffStats, Inequality, InequalityType,
};
pub use omega_test::{
    LinearInequality, LinearSystem, OmegaConfig, OmegaResult, OmegaStats, OmegaTest,
};
pub use virtual_term::{
    LinearConstraint, LinearTerm, VirtualTermSubstitution, VtsConfig, VtsStats,
};
