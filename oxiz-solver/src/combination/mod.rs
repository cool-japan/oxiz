//! Theory Combination.
//!
//! This module provides comprehensive theory combination infrastructure:
//! - Basic Nelson-Oppen combination
//! - Advanced Nelson-Oppen with non-convex theory support
//! - Optimized propagation engine
//! - Interface equality management
//! - Conflict resolution
//! - Convexity checking and handling
//! - Partition refinement algorithms

pub mod coordinator;
pub mod equality_propagation;
pub mod nelson_oppen;

// Advanced modules
pub mod conflict_resolution;
pub mod convexity;
pub mod interface_eq;
pub mod nelson_oppen_advanced;
pub mod partition_refinement;
pub mod propagation_opt;
pub mod shared_terms_advanced;

// Re-exports from basic modules
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

// Re-exports from advanced modules
pub use nelson_oppen_advanced::{
    AdvancedNelsonOppen, AdvancedNelsonOppenConfig, AdvancedNelsonOppenStats,
    CaseSplit as AdvancedCaseSplit, Disequality, EqualityExplanation, EqualityPriority,
    ModelBasedCombination, PartitionRefinement as AdvancedPartitionRefinement, TermPartition,
    TheoryConflict as AdvancedTheoryConflict, TheoryProperties,
    TheorySolver as AdvancedTheorySolver, UnionFindWithExplanation,
};

pub use propagation_opt::{
    DependencyGraph, IncrementalPropagator, LazyPropagator, Literal, OptimizedPropagationEngine,
    PropagationCache, PropagationConfig, PropagationEvent, PropagationPriority, PropagationReason,
    PropagationStats, PropagationTrail, WatchList,
};

pub use shared_terms_advanced::{
    AdvancedSharedTermsManager, EClass, EClassId as AdvancedEClassId, EGraph as AdvancedEGraph,
    ENode, Equality as SharedEquality, EqualityExplanation as SharedEqualityExplanation,
    InterfaceTermMinimizer, SharedTermInfo, SharedTermsConfig, SharedTermsStats,
};

pub use interface_eq::{
    EqualityMinimizer, EqualityPriority as InterfaceEqualityPriority,
    EqualityScheduler as InterfaceEqualityScheduler, GenerationStrategy, InterfaceEClass,
    InterfaceEquality, InterfaceEqualityConfig, InterfaceEqualityManager, InterfaceEqualityStats,
    SchedulingPolicy as InterfaceSchedulingPolicy,
};

pub use conflict_resolution::{
    ConflictAnalysis, ConflictClause, ConflictResolutionConfig, ConflictResolutionStats,
    ConflictResolver, Explanation as ConflictExplanation, ExplanationGenerator,
    Literal as ConflictLiteral, MinimizationAlgorithm, MultiTheoryConflictAnalyzer, TheoryConflict,
};

pub use convexity::{
    CaseSplitStrategy, ConvexityConfig, ConvexityHandler, ConvexityProperty, ConvexityStats,
    DisjunctiveReasoning, EqualityDisjunction,
    ModelBasedCombination as ConvexityModelBasedCombination, TheoryModel,
};

pub use partition_refinement::{
    ClassId, Partition, PartitionComparator, PartitionEnumerator, PartitionRefinement,
    PartitionRefinementConfig, PartitionRefinementManager, PartitionRefinementStats,
};
