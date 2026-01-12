//! BitVector theory solver
//!
//! Implements bit-blasting for fixed-width bit vectors with AIG (And-Inverter Graph) representation.

mod aig;
mod aig_builder;
mod propagator;
mod solver;

pub use aig::{AigCircuit, AigEdge, AigNode, AigStats, NodeId};
pub use aig_builder::AigBvBuilder;
pub use propagator::{Constraint, Interval, WordLevelPropagator};
pub use solver::BvSolver;
