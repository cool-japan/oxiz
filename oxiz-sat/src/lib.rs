//! OxiZ SAT Solver - High-performance CDCL SAT solver
//!
//! This crate implements a Conflict-Driven Clause Learning (CDCL) SAT solver
//! with the following features:
//! - Two-watched literals scheme for efficient unit propagation
//! - Multiple branching heuristics (VSIDS, LRB, CHB, VMTF)
//! - Clause learning with first-UIP scheme and recursive minimization
//! - Preprocessing (BCE, BVE, subsumption elimination)
//! - Incremental solving (push/pop)
//! - DRAT proof generation
//! - Local search integration
//! - Parallel portfolio solving
//! - AllSAT enumeration
//!
//! # Examples
//!
//! ## Basic SAT Solving
//!
//! ```
//! use oxiz_sat::{Solver, SolverResult, Lit};
//!
//! let mut solver = Solver::new();
//!
//! // Create variables
//! let a = solver.new_var();
//! let b = solver.new_var();
//! let c = solver.new_var();
//!
//! // Add clause: a OR b
//! solver.add_clause([Lit::pos(a), Lit::pos(b)]);
//!
//! // Add clause: NOT a OR c
//! solver.add_clause([Lit::neg(a), Lit::pos(c)]);
//!
//! // Add clause: NOT b OR NOT c
//! solver.add_clause([Lit::neg(b), Lit::neg(c)]);
//!
//! match solver.solve() {
//!     SolverResult::Sat => println!("Satisfiable!"),
//!     SolverResult::Unsat => println!("Unsatisfiable!"),
//!     SolverResult::Unknown => println!("Unknown"),
//! }
//! ```
//!
//! ## Solving with Assumptions
//!
//! ```
//! use oxiz_sat::{Solver, SolverResult, Lit};
//!
//! let mut solver = Solver::new();
//! let a = solver.new_var();
//! let b = solver.new_var();
//!
//! solver.add_clause([Lit::pos(a), Lit::pos(b)]);
//!
//! // Solve assuming a is false
//! let (result, _) = solver.solve_with_assumptions(&[Lit::neg(a)]);
//! assert_eq!(result, SolverResult::Sat); // b must be true
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod activity;
mod agility;
mod allsat;
mod assumptions;
mod asymmetric_branching;
mod autotuning;
mod backbone;
mod benchmark;
mod big;
mod cardinality;
mod cce;
mod chb;
mod chrono;
mod clause;
mod clause_exchange;
mod clause_maintenance;
mod clause_size_manager;
mod community;
mod community_partition;
mod config_presets;
mod cube;
mod cube_solver;
mod dimacs;
mod distillation;
mod drat_inprocessing;
mod dynamic_lbd;
mod els;
mod extended_resolution;
mod gate;
mod gpu;
mod hyper_binary;
mod literal;
mod local_search;
mod lookahead;
mod lrb;
mod maxsat;
mod memory;
mod memory_opt;
mod ml_branching;
mod occurrence;
mod portfolio;
mod preprocessing;
mod profiling;
mod proof;
mod recursive_minimization;
mod reluctant;
mod rephasing;
mod resolution_graph;
mod smoothed_lbd;
mod solver;
mod stabilization;
mod stats_dashboard;
mod subsumption;
mod symmetry;
mod target_phase;
mod trail;
mod trail_saving;
mod unsat_core;
mod vivification;
mod vmtf;
mod vmtf_queue;
mod vsids;
mod watched;
mod xor;

pub use activity::{ActivityStats, ClauseActivityManager, VariableActivityManager};
pub use agility::{AgilityStats, AgilityTracker};
pub use allsat::{AllSatEnumerator, EnumerationConfig, EnumerationResult, EnumerationStats, Model};
pub use assumptions::{
    Assumption, AssumptionContext, AssumptionCoreMinimizer, AssumptionLevel, AssumptionStack,
    AssumptionStats,
};
pub use asymmetric_branching::{AsymmetricBranching, AsymmetricBranchingStats};
pub use autotuning::{
    Autotuner, AutotuningStats, Configuration, Parameter,
    PerformanceMetrics as TuningPerformanceMetrics, TuningStrategy,
};
pub use backbone::{BackboneAlgorithm, BackboneDetector, BackboneFilter, BackboneStats};
pub use benchmark::{BenchmarkHarness, BenchmarkResult};
pub use big::{BigStats, BinaryImplicationGraph};
pub use cardinality::CardinalityEncoder;
pub use cce::{CceStats, CoveredClauseElimination};
pub use clause::{Clause, ClauseDatabase, ClauseDatabaseStats, ClauseId, ClauseTier};
pub use clause_exchange::{ClauseExchangeBuffer, ExchangeConfig, ExchangeStats, SharedClause};
pub use clause_maintenance::{ClauseMaintenance, MaintenanceStats};
pub use clause_size_manager::{ClauseSizeManager, SizeAdjustmentStrategy, SizeManagerStats};
pub use community::{
    Communities, CommunityOrdering, CommunityStats, LouvainDetector, VariableIncidenceGraph,
};
pub use community_partition::{CommunityPartition, PartitionStats};
pub use config_presets::ConfigPreset;
pub use cube::{Cube, CubeConfig, CubeGenerator, CubeResult, CubeSplittingStrategy, CubeStats};
pub use cube_solver::{
    CubeAndConquer, CubeSolveResult, CubeSolverConfig, CubeSolverStats, ParallelCubeSolver,
};
pub use dimacs::{DimacsError, DimacsParser, DimacsWriter};
pub use distillation::{Distillation, DistillationStats};
pub use drat_inprocessing::{DratInprocessingConfig, DratInprocessingStats, DratInprocessor};
pub use dynamic_lbd::{DynamicLbdManager, DynamicLbdStats};
pub use els::{ElsStats, EquivalentLiteralSubstitution};
pub use extended_resolution::{ClauseSubstitution, ExtendedResolution, Extension, ExtensionType};
pub use gate::{GateDetector, GateStats, GateType};
pub use gpu::{
    BatchPropagationResult, ClauseManagementResult, ConflictAnalysisResult,
    CpuReferenceAccelerator, GpuBackend, GpuBenchmark, GpuBenchmarkResult, GpuBenefitAnalysis,
    GpuConfig, GpuError, GpuRecommendation, GpuSolverAccelerator, GpuStats,
};
pub use hyper_binary::{HbrResult, HyperBinaryResolver, HyperBinaryStats};
pub use literal::{LBool, Lit, Var};
pub use local_search::{LocalSearch, LocalSearchConfig, LocalSearchResult, LocalSearchStats};
pub use lookahead::{LookaheadBranching, LookaheadHeuristic, LookaheadStats};
pub use maxsat::{MaxSatClause, MaxSatConfig, MaxSatResult, MaxSatSolver, MaxSatStats, Weight};
pub use memory::{ClauseArena, ClauseRef, MemoryStats};
pub use memory_opt::{MemoryAction, MemoryOptStats, MemoryOptimizer, SizeClass};
pub use ml_branching::{MLBranching, MLBranchingConfig, MLBranchingStats};
pub use occurrence::{OccurrenceList, OccurrenceStats};
pub use portfolio::{PortfolioConfig, PortfolioResult, PortfolioSolver, PortfolioStats};
pub use preprocessing::Preprocessor;
pub use profiling::{AutoTimer, PerformanceMetrics, Profiler, ScopedTimer};
pub use proof::{DratProof, LratProof, ProofTrimmer};
pub use recursive_minimization::{RecursiveMinStats, RecursiveMinimizer};
pub use reluctant::{ReluctantDoubling, ReluctantStats};
pub use rephasing::{RephasingManager, RephasingStats, RephasingStrategy};
pub use resolution_graph::{
    GraphStats as ResolutionGraphStats, ResolutionAnalyzer, ResolutionGraph, ResolutionNode,
};
pub use smoothed_lbd::{SmoothedLbdStats, SmoothedLbdTracker};
pub use solver::{
    RestartStrategy, Solver, SolverConfig, SolverResult, SolverStats, TheoryCallback,
    TheoryCheckResult,
};
pub use stabilization::{
    SearchMode, StabilizationConfig, StabilizationManager, StabilizationStats,
};
pub use stats_dashboard::{StatsAggregator, StatsDashboard};
pub use subsumption::{SubsumptionChecker, SubsumptionStats};
pub use symmetry::{
    AutomorphismDetector, MatrixSymmetry, Permutation, SymmetryBreaker, SymmetryBreakingMethod,
    SymmetryGroup,
};
pub use target_phase::{PhaseMode, TargetPhaseSelector, TargetPhaseStats};
pub use trail::{Reason, Trail};
pub use trail_saving::{SavedTrail, TrailSavingManager, TrailSavingStats};
pub use unsat_core::UnsatCore;
pub use vivification::{Vivification, VivificationStats};
pub use vmtf::{VMTF, VmtfStats};
pub use vmtf_queue::{VmtfBumpQueue, VmtfBumpStats};
pub use xor::{
    GF2Matrix, GF2Row, PropagateResult, XorAddResult, XorClause, XorClauseId, XorConstraint,
    XorDetector, XorManager, XorPropagator, XorPropagatorStats, XorStrengthening, XorSubsumption,
};
