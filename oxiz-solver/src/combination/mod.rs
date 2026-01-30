//! Theory Combination.

pub mod coordinator;
pub mod equality_propagation;
pub mod nelson_oppen;

pub use coordinator::{
    CoordinatorConfig, CoordinatorStats, EqualityProp, SatResult as CoordSatResult, SharedTerm,
    TheoryCoordinator, TheoryId as CoordTheoryId, TheorySolver,
};
pub use equality_propagation::{
    CongruenceData, CongruenceKey, EClassData, EClassId, EGraph, EqualityPropStats,
    EqualityPropagator, EqualityWatch, Explanation, TheoryExplanation, UnionFind,
};
pub use nelson_oppen::{
    CombinationResult, Equality, NelsonOppen, NelsonOppenConfig, NelsonOppenStats, TheoryId,
};
