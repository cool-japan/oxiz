//! OxiZ Theory Solvers
//!
//! This crate provides theory solvers for the CDCL(T) framework:
//! - **EUF** (Equality with Uninterpreted Functions) using E-graphs
//! - **LRA/LIA** (Linear Real/Integer Arithmetic) using Simplex
//! - **BV** (BitVectors) using bit-blasting and word-level propagation
//! - **Arrays** (Extensional Arrays) using read-over-write axioms
//! - **FP** (Floating-Point) IEEE 754 using bit-blasting
//! - **Datatypes** (Algebraic Data Types) for lists, trees, enums
//! - **Strings** (Word Equations + Regular Expressions) using Brzozowski derivatives
//! - **Difference Logic** (DL) for constraints x - y ≤ c using Bellman-Ford
//! - **UTVPI** (Unit Two-Variable Per Inequality) for constraints ±x ± y ≤ c
//! - **Pseudo-Boolean** (PB) for cardinality and weighted constraints
//! - **Special Relations** for partial/total orders, transitive closure
//! - Nelson-Oppen theory combination
//!
//! # Architecture
//!
//! Each theory solver implements the [`Theory`] trait which provides:
//! - `assert_literal`: Process a new assignment
//! - `propagate`: Perform theory propagation
//! - `check`: Check consistency and produce conflicts
//! - `explain`: Generate explanations for propagated literals
//! - `backtrack`: Handle backtracking
//!
//! # Examples
//!
//! ## Theory Combination
//!
//! ```ignore
//! use oxiz_theories::{TheoryCombiner, CombinationConfig};
//! use oxiz_theories::euf::EufSolver;
//! use oxiz_theories::arithmetic::LraSolver;
//!
//! // Create a combiner with EUF and LRA
//! let config = CombinationConfig::default();
//! let combiner = TheoryCombiner::new(config);
//! ```
//!
//! ## User Propagator
//!
//! ```ignore
//! use oxiz_theories::user_propagator::{UserPropagator, PropagatorCallback};
//!
//! struct MyPropagator;
//!
//! impl PropagatorCallback for MyPropagator {
//!     fn on_push(&mut self) {}
//!     fn on_pop(&mut self, _num_scopes: u32) {}
//!     fn on_fixed(&mut self, _term: TermId, _value: bool) -> Vec<TermId> {
//!         vec![]
//!     }
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod arithmetic;
pub mod array;
pub mod bv;
pub mod character;
pub mod checking;
pub mod combination;
pub mod config;
pub mod datatype;
pub mod diff_logic;
pub mod error;
pub mod euf;
pub mod fp;
pub mod hashcons;
mod lru_cache;
pub mod pb;
pub mod propagation;
pub mod quantifier;
pub mod recfun;
pub mod simplify;
pub mod sls;
pub mod special_relations;
pub mod string;
mod theory;
pub mod user_propagator;
pub mod utvpi;
pub mod watched;
pub mod wmaxsat;

pub use combination::{Purifier, SharedVar, TheoryCombiner};
pub use config::{
    BranchingHeuristic, BvConfig, CombinationConfig, CombinationMode, LiaConfig, PivotingRule,
    SimplexConfig, TheoryConfig,
};
pub use datatype::{Constructor, DatatypeDecl, DatatypeSolver, DatatypeSort, Field, Selector};
pub use error::{ConflictInfo, ResourceLimit, SolverStats, TheoryError, TheoryResult};
pub use fp::{FpFormat, FpRoundingMode, FpSolver, FpValue};
pub use hashcons::{HashConsTable, HcTerm};
pub use propagation::{Propagation, PropagationPriority, PropagationQueue, PropagationStats};
pub use simplify::{SimplifyContext, SimplifyResult, SimplifyStats};
pub use string::{Regex, RegexOp, StringSolver};
pub use theory::{
    EqualityNotification, Theory, TheoryCombination, TheoryId, TheoryResult as TheoryCheckResult,
};
pub use watched::{WatchList, WatchStats, WatchedConstraint};

// Quantifier solver exports
pub use quantifier::{
    InstantiationLemma, QuantifierConfig, QuantifierSolver, QuantifierStats, TrackedQuantifier,
};

// Recursive function solver exports
pub use recfun::{CaseDef, RecFunConfig, RecFunDef, RecFunId, RecFunSolver, RecFunStats};

// User propagator exports
pub use user_propagator::{
    Consequence, PropagatorContext, PropagatorResult, UserPropagator, UserPropagatorManager,
    UserPropagatorStats,
};

// Special relations exports
pub use special_relations::{
    RelationDef, RelationEdge, RelationKind, RelationProperties, SpecialRelationSolver,
    SpecialRelationStats,
};

// Pseudo-Boolean exports
pub use pb::{PbConfig, PbConstraint, PbConstraintKind, PbSolver, PbStats, WeightedLiteral};

// Difference Logic exports
pub use diff_logic::{
    BellmanFord, ConstraintGraph, DenseDiffLogic, DiffConstraint, DiffEdge, DiffLogicConfig,
    DiffLogicResult, DiffLogicSolver, DiffLogicStats, DiffVar, NegativeCycle,
};

// UTVPI exports
pub use utvpi::{
    DetectedConstraint, DoubledGraph, DoubledNode, Sign, UtConstraint, UtConstraintKind, UtEdge,
    UtvpiConfig, UtvpiDetector, UtvpiDetectorStats, UtvpiResult, UtvpiSolver, UtvpiStats,
};

// Theory checking exports
pub use checking::{
    ArithCheckConfig, ArithChecker, ArrayChecker, BvChecker, CheckResult, CheckerStats,
    CombinedChecker, Literal, ProofChecker, ProofStep, ProofStepKind, QuantChecker, TheoryChecker,
    TheoryKind,
};

// Weighted MaxSAT exports
pub use wmaxsat::{
    SoftClause, WMaxSatConfig, WMaxSatResult, WMaxSatSolver, WMaxSatStats, Weight as WMaxWeight,
};

// Character theory exports
pub use character::{
    AdvancedCharSolver, CaseFoldMode, CaseFolder, CharClass, CharConfig, CharConstraint,
    CharDomain, CharNormalizer, CharResult, CharSolver, CharStats, CharValue, CharVar, CharWidth,
    CodePoint, NormalizationForm, UnicodeBlock, UnicodeCategory, UnicodeScript,
};

// SLS theory exports
pub use sls::{
    // Backbone detection
    BackboneDetector,
    // BMS selector
    BmsConfig,
    BmsSelector,
    // CCAnr enhancements
    CcanrConfig,
    CcanrEnhancer,
    // Core SLS
    ClauseId,
    // Clause importance
    ClauseImportance,
    // Clause simplification
    ClauseSimplifier,
    ClauseWeightManager,
    // DDFW weights
    DdfwConfig,
    DdfwManager,
    // Diversification
    DiversificationManager,
    DiversificationStrategy,
    FocusedWalk,
    FocusedWalkConfig,
    // Hybrid SLS-CDCL
    HybridSlsInterface,
    // Novelty heuristics
    NoveltyConfig,
    NoveltySelector,
    PhaseMode,
    PhaseSaver,
    // Portfolio SLS
    PortfolioConfig,
    PortfolioSls,
    // Restart strategies
    RestartManager,
    RestartStrategy,
    SlsAlgorithm,
    SlsConfig,
    SlsResult,
    SlsSolver,
    SlsStats,
    // Solution learning
    SolutionLearner,
    // Solution verification
    SolutionVerifier,
    // Sparrow algorithm
    SparrowConfig,
    SparrowSelector,
    VarActivity,
    VarSelectHeuristic,
    VerificationResult,
    WeightedSlsConfig,
    WeightedSlsSolver,
    WeightedSlsStats,
    WeightingScheme,
    // YalSAT
    YalsatConfig,
    YalsatSolver,
};
