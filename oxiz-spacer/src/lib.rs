//! # oxiz-spacer
//!
//! Property Directed Reachability (PDR/IC3) engine for OxiZ.
//!
//! This crate implements Spacer, a solver for Constrained Horn Clauses (CHC),
//! which is essential for software verification tasks like loop invariant inference.
//!
//! ## Key Differentiator
//!
//! Spacer is **missing in most Z3 clones**. Having a Pure Rust implementation
//! makes OxiZ uniquely suitable for verification toolchains.
//!
//! ## Reference
//!
//! Z3's `muz/spacer/` directory.
//!
//! ## Use Cases
//!
//! - **Loop Invariant Inference**: Automatically find loop invariants
//! - **Safety Verification**: Prove program safety properties
//! - **Model Checking**: Bounded and unbounded verification
//! - **Interpolation**: Generate Craig interpolants
//!
//! ## Example
//!
//! ```rust,no_run
//! use oxiz_spacer::chc::{ChcSystem, PredicateApp};
//! use oxiz_spacer::pdr::{Spacer, SpacerResult};
//! use oxiz_core::TermManager;
//!
//! let mut terms = TermManager::new();
//! let mut system = ChcSystem::new();
//!
//! // Declare a predicate Inv(x: Int)
//! let inv = system.declare_predicate("Inv", [terms.sorts.int_sort]);
//!
//! // Add init rule: x = 0 => Inv(x)
//! let x = terms.mk_var("x", terms.sorts.int_sort);
//! let zero = terms.mk_int(0);
//! let init_constraint = terms.mk_eq(x, zero);
//! system.add_init_rule(
//!     [("x".to_string(), terms.sorts.int_sort)],
//!     init_constraint,
//!     inv,
//!     [x],
//! );
//!
//! // Add query: Inv(x) /\ x < 0 => false
//! let neg_constraint = terms.mk_lt(x, zero);
//! system.add_query(
//!     [("x".to_string(), terms.sorts.int_sort)],
//!     [PredicateApp::new(inv, [x])],
//!     neg_constraint,
//! );
//!
//! // Solve
//! let mut spacer = Spacer::new(&mut terms, &system);
//! let result = spacer.solve();
//! assert!(matches!(result, Ok(SpacerResult::Safe)));
//! ```

pub mod abduction;
pub mod bmc;
pub mod chc;
pub mod chccomp;
pub mod ctg;
pub mod diagnostics;
pub mod distributed;
pub mod existential;
pub mod frames;
pub mod generalize;
pub mod interp;
pub mod invariant;
pub mod nonlinear;
pub mod parallel;
pub mod parser;
pub mod pdr;
pub mod pob;
pub mod portfolio;
pub mod proof;
pub mod reach;
pub mod recursive;
pub mod smt;
pub mod theory;

// Re-exports for convenience
pub use abduction::{
    AbductionError, AbductionResult, AbductiveSolver, Hypothesis, HypothesisGenerator,
    HypothesisOrigin, HypothesisTemplate, InvariantSynthesizer, PreconditionSynthesizer,
    TemplateParamKind, TemplateStructure,
};
pub use bmc::{Bmc, BmcConfig, BmcError, BmcResult, BmcStats, HybridSolver};
pub use chc::{ChcSystem, PredId, Predicate, PredicateApp, Rule, RuleBody, RuleHead, RuleId};
pub use chccomp::{ChcCompError, ChcCompParser};
pub use ctg::{CtgError, CtgResult, CtgStrengthener};
pub use diagnostics::{InvariantFormatter, StatsReporter, TraceBuffer, TraceEvent, TraceLevel};
pub use distributed::{
    DistributedConfig, DistributedCoordinator, DistributedError, DistributedStats, SharedState,
    WorkItem as DistributedWorkItem, Worker, WorkerMessage,
};
pub use existential::{
    ExistentialError, ExistentialHandler, ExistentialInfo, ExistentialProjector, ExistentialResult,
    SkolemContext, WitnessExtractor,
};
pub use frames::{FrameManager, INFTY_LEVEL, Lemma, LemmaId, PredicateFrames};
pub use generalize::{GeneralizationError, GeneralizationResult, Generalizer};
pub use interp::{Interpolant, InterpolationError, InterpolationResult, Interpolator};
pub use invariant::{
    Candidate, CandidateSource, InferenceResult, InferenceStats, InvariantConfig,
    InvariantInference, TemplateKind,
};
pub use nonlinear::{
    InterpolantTree, NonLinearAnalyzer, NonLinearError, NonLinearLemmaCombiner, NonLinearResult,
    NonLinearRuleAnalysis, NonLinearStats, TreeInterpolant,
};
pub use parallel::{
    ParallelConfig, ParallelError, ParallelFrameSolver, ParallelPropagator, WorkItem, WorkResult,
};
pub use parser::{ChcParser, ParseError};
pub use pdr::{Spacer, SpacerConfig, SpacerError, SpacerResult, SpacerStats};
pub use pob::{Pob, PobId, PobManager, PobQueue};
pub use portfolio::{PortfolioError, PortfolioResult, PortfolioSolver, Strategy};
pub use proof::{ProofBuilder, ProofStep, SafetyProof};
pub use reach::{
    CexState, ConcreteState, ConcreteValue, ConcreteWitness, ConcreteWitnessExtractor,
    Counterexample, Generalization, OverApproximation, Projection, ReachFact, ReachFactId,
    ReachFactStore, ReachabilityChecker, UnderApproximation, WitnessError,
};
pub use recursive::{RecursionKind, RecursiveAnalyzer, RecursiveError, RecursiveInfo};
pub use smt::{MbpResult, Model, SmtError, SmtSolver, SmtStats};
pub use theory::TheoryIntegration;
