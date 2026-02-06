//! BitVector theory solver
//!
//! Implements bit-blasting for fixed-width bit vectors with AIG (And-Inverter Graph) representation.
//!
//! # Modules
//!
//! - **aig**: And-Inverter Graph circuit representation
//! - **aig_builder**: Builder for constructing AIG circuits
//! - **propagator**: Word-level interval propagation
//! - **solver**: Main bitvector solver with bit-blasting
//! - **bitblast_advanced**: Advanced bit-blasting strategies (AIG-based, lazy, cuts)
//! - **word_level**: Word-level reasoning without bit-blasting
//! - **division_opt**: Optimized division and modulo operations

mod aig;
mod aig_builder;
mod bitblast_advanced;
mod division_opt;
mod propagator;
mod solver;
mod word_level;

pub use aig::{AigCircuit, AigEdge, AigNode, AigStats, NodeId};
pub use aig_builder::AigBvBuilder;
pub use bitblast_advanced::{AdvancedBitBlaster, AigCut, BitBlastConfig, BitBlastStats, Polarity};
pub use division_opt::{
    BarrettParams, DivisionConfig, DivisionOptimizer, DivisionStats, MontgomeryParams,
};
pub use propagator::{Constraint, Interval, WordLevelPropagator};
pub use solver::BvSolver;
pub use word_level::{
    OverflowInfo, SignInfo, SimplifiedTerm, WidthInfo, WordLevelReasoner, WordLevelStats,
};
