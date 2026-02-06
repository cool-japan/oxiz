//! Linear Programming and Mixed Integer Programming.

pub mod basis_update;
pub mod branch_cut;
pub mod cutting_planes;
pub mod cutting_planes_extended;
pub mod dual_simplex;
pub mod farkas;

// Re-export main types explicitly to avoid ambiguous glob re-exports
// (VarId, CuttingPlane, CutType are defined in multiple modules)
pub use basis_update::{Basis, BasisUpdateConfig, BasisUpdateStats, BasisUpdater, EtaMatrix};
pub use branch_cut::{
    BranchCutConfig, BranchCutResult, BranchCutSolver, BranchCutStats, BranchingStrategy,
    NodeSelection, VarType,
};
pub use cutting_planes::{CuttingPlaneConfig, CuttingPlaneGenerator, CuttingPlaneStats};
pub use cutting_planes_extended::{
    ExtendedCuttingPlaneGenerator, ExtendedCuttingPlanesConfig, ExtendedCuttingPlanesStats,
};
pub use dual_simplex::{DualSimplexResult, DualSimplexSolver, DualSimplexStats};
pub use farkas::{FarkasCertificate, FarkasConfig, FarkasGenerator, FarkasStats, LinearConstraint};

// Use the cutting_planes module's types as canonical
pub use cutting_planes::{CutType, CuttingPlane};
/// Variable identifier (shared across LP modules)
pub type VarId = usize;
/// Constraint identifier (for Farkas proofs)
pub type ConstraintId = usize;
