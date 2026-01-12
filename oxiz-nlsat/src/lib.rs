//! # oxiz-nlsat
//!
//! Non-linear arithmetic solver for OxiZ using Cylindrical Algebraic Decomposition (CAD).
//!
//! This crate implements the NLSAT algorithm for solving non-linear real arithmetic
//! (QF_NRA) and non-linear integer arithmetic (QF_NIA) problems.
//!
//! ## Reference
//!
//! Z3's `nlsat/` directory (~180k lines), particularly `nlsat_solver.cpp`.
//!
//! ## Key Components
//!
//! - **Polynomial Constraints**: Representation of non-linear constraints
//! - **Variable Ordering**: Strategies for CAD variable ordering
//! - **Interval Sets**: Representation of solution intervals
//! - **Explanation**: Conflict explanation for CDCL integration

pub mod assignment;
pub mod assumptions;
pub mod asymmetric_literal_addition;
pub mod bce;
pub mod bound_propagation;
pub mod bve;
pub mod cad;
pub mod chrono_bt;
pub mod clause;
pub mod clause_tiers;
pub mod cutting_planes;
pub mod discriminant;
pub mod eval_cache;
pub mod evaluator;
pub mod explain;
pub mod incremental_cad;
pub mod inprocessing;
pub mod interval_set;
pub mod lemma;
pub mod lookahead;
pub mod maxsat;
pub mod monotonicity;
pub mod nia;
pub mod portfolio;
pub mod proof;
pub mod restart;
pub mod root_hints;
pub mod simplify;
pub mod solver;
pub mod structure_analyzer;
pub mod subsumption;
pub mod symmetry;
pub mod theory_conflict;
pub mod types;
pub mod var_order;
pub mod vivification;
pub mod watched_literals;

// Re-exports
pub use assumptions::{AssumptionManager, Scope};
pub use asymmetric_literal_addition::{AlaConfig, AlaEngine, AlaStats};
pub use bce::{BceConfig, BceEngine, BceStats};
pub use bound_propagation::{BoundPropagator, Interval};
pub use bve::{BveConfig, BveEngine, BveStats};
pub use cad::{
    CadCell, CadConfig, CadDecomposer, CadError, CadLifter, CadPoint, CadProjection, ProjectionSet,
    SampleStrategy, SturmSequence,
};
pub use chrono_bt::{
    BacktrackDecision, ChronoBacktracker, ChronoConfig, ChronoStats, ConflictAnalysisResult,
};
pub use clause_tiers::{ClauseTier, ClauseTierConfig, ClauseTierManager, ClauseTierStats};
pub use cutting_planes::{CutStats, CutType, CuttingPlane, CuttingPlaneGenerator};
pub use discriminant::{DiscriminantAnalyzer, DiscriminantSign, DiscriminantStats, RootInfo};
pub use eval_cache::{CachedSign, EvalCache, EvalCacheConfig, EvalCacheStats, SignPattern};
pub use incremental_cad::{CacheStats, CadSnapshot, IncrementalCad};
pub use inprocessing::{InprocessConfig, InprocessStats, Inprocessor};
pub use lookahead::{LookaheadConfig, LookaheadEngine, LookaheadResult, LookaheadStats};
pub use maxsat::{MaxSatConfig, MaxSatResult, MaxSatSolver, MaxSatStats, SoftConstraint};
pub use monotonicity::{
    MonotonicityAnalyzer, MonotonicityDirection, MonotonicityInfo, MonotonicityStats,
};
pub use nia::{BranchingStrategy, NiaConfig, NiaSolver, NiaStats, VarType};
pub use portfolio::{PortfolioConfig, PortfolioResult, PortfolioSolver, PortfolioStats};
pub use proof::{
    CadOperation, Proof, ProofBuilder, ProofError, ProofId, ProofMetadata, ProofRule, ProofStats,
    ProofStep, TheoryExplanation,
};
pub use restart::{RestartManager, RestartStrategy};
pub use root_hints::{
    PolynomialRootHints, RootHint, RootHintsCache, RootHintsConfig, RootHintsStats,
};
pub use solver::NlsatSolver;
pub use structure_analyzer::{ProblemClass, SolverStrategy, StructureAnalyzer, StructureStats};
pub use subsumption::{SubsumptionChecker, SubsumptionConfig, SubsumptionStats};
pub use symmetry::{Permutation, SymmetryDetector, SymmetryGroup, SymmetryStats};
pub use theory_conflict::{
    ConflictInfo, ConflictPatterns, TheoryConflictConfig, TheoryConflictStats,
    TheoryConflictTracker,
};
pub use var_order::{OrderingAnalyzer, OrderingStats, OrderingStrategy, VariableOrdering};
pub use vivification::{VivificationConfig, VivificationStats, Vivifier};
pub use watched_literals::{WatchEntry, WatchStats, WatchedLiterals, WatchedPropagator};
