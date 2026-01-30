//! Arithmetic Quantifier Elimination.
//!
//! This module implements quantifier elimination procedures for linear
//! arithmetic over integers and reals.

// TODO: Implement these modules
// pub mod cooper;
// pub mod ferrante_rackoff;
pub mod omega_test;
pub mod qe_lite_arith;
// pub mod virtual_term;

// TODO: Uncomment when modules are implemented
// pub use cooper::{Constraint, CooperConfig, CooperQE, CooperStats, DnfFormula};
// pub use ferrante_rackoff::{
//     Conjunction as FrConjunction, DnfFormula as FrDnfFormula, FerranteRackoffConfig,
//     FerranteRackoffQE, FerranteRackoffStats, Inequality, InequalityType,
// };
pub use omega_test::{
    LinearConstraint as OmegaLinearConstraint, OmegaResult, OmegaTestConfig as OmegaConfig,
    OmegaTestStats as OmegaStats, OmegaTester as OmegaTest,
};
// pub use virtual_term::{
//     LinearConstraint, LinearTerm, VirtualTermSubstitution, VtsConfig, VtsStats,
// };
