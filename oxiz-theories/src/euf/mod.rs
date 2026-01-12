//! Equality with Uninterpreted Functions (EUF) theory solver
//!
//! Uses Union-Find with path compression for congruence closure.

mod ematching;
mod incremental;
mod proof;
mod solver;
mod union_find;

pub use ematching::{
    CodeTree, EMatchEngine, Pattern, QuantifiedFormula, Substitution, Trigger, VarId,
};
pub use incremental::{
    AnalysisData, ConstValue, EClass, EClassId, EGraph, EGraphBuilder, EGraphConfig, EGraphStats,
    ENode, TheoryPropagator,
};
pub use proof::{Conflict, ProofForest, ProofManager, ProofStep};
pub use solver::EufSolver;
pub use union_find::UnionFind;
