//! OxiZ Core - AST, Sorts, and Traits for the SMT Solver
//!
//! This crate provides the foundational types and traits for the OxiZ SMT solver:
//! - Arena-allocated terms with efficient [`TermId`] references
//! - Sort system for type checking
//! - SMT-LIB2 parser and printer (requires `std` feature)
//! - Tactic framework for preprocessing and simplification (requires `std` feature)
//! - Rewrite system for term transformations
//!
//! # Examples
//!
//! ## Creating Terms
//!
//! ```
//! use oxiz_core::ast::TermManager;
//! use num_bigint::BigInt;
//!
//! let mut tm = TermManager::new();
//!
//! // Boolean terms
//! let p = tm.mk_var("p", tm.sorts.bool_sort);
//! let q = tm.mk_var("q", tm.sorts.bool_sort);
//! let and_pq = tm.mk_and(vec![p, q]);
//!
//! // Integer terms
//! let x = tm.mk_var("x", tm.sorts.int_sort);
//! let five = tm.mk_int(BigInt::from(5));
//! let ge = tm.mk_ge(x, five);
//! ```
//!
//! ## Parsing SMT-LIB2
//!
//! ```
//! use oxiz_core::smtlib::parse_script;
//! use oxiz_core::ast::TermManager;
//!
//! let input = r#"
//!     (declare-const x Int)
//!     (assert (> x 0))
//!     (check-sat)
//! "#;
//!
//! let mut tm = TermManager::new();
//! let commands = parse_script(input, &mut tm).unwrap();
//! ```
//!
//! ## Using Tactics
//!
//! ```
//! use oxiz_core::tactic::{Goal, StatelessSimplifyTactic, Tactic};
//! use oxiz_core::ast::TermManager;
//!
//! let mut tm = TermManager::new();
//!
//! // Create a goal with some formula
//! let t = tm.mk_true();
//! let f = tm.mk_false();
//! let or_tf = tm.mk_or(vec![t, f]);
//!
//! let goal = Goal::new(vec![or_tf]);
//! let tactic = StatelessSimplifyTactic;
//!
//! // Apply simplification
//! let result = tactic.apply(&goal);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod interner;
mod prelude;

// === Always-available modules (no_std compatible) ===
pub mod ast;
pub mod config;
pub mod error;
pub mod error_context;
pub mod error_recovery;
pub mod error_utils;
pub mod literal;
pub mod rewrite;
pub mod sort;
pub mod theories;
pub mod traits;
pub mod unsat_core;

// === std-only modules ===
#[cfg(feature = "std")]
pub mod alloc;
#[cfg(feature = "std")]
pub mod datalog;
#[cfg(feature = "std")]
pub mod diagnostics;
#[cfg(feature = "std")]
pub mod ematching;
#[cfg(feature = "std")]
pub mod model;
#[cfg(feature = "std")]
pub mod qe;
#[cfg(feature = "std")]
pub mod resource;
#[cfg(feature = "std")]
pub mod smtlib;
#[cfg(feature = "std")]
pub mod statistics;
#[cfg(feature = "std")]
pub mod tactic;

// === Always-available exports ===
pub use ast::{Term, TermId, TermKind, TermManager};
pub use config::{
    ClauseDeletionStrategy, Config, GeneralParams, PhaseSaving, ResourceLimits, SatParams,
    SimplifyParams,
};
pub use error::{OxizError, Result};
pub use error_context::{ErrorContext, ResultExt};
pub use error_recovery::{
    ErrorBatch, ErrorRecovery, RecoveryConfig, RecoveryResult, RecoveryStats, RecoveryStrategy,
};
pub use error_utils::{
    ContextResult, OptionErrorExt, SortErrorExt, arity_mismatch, error_chain, parse_error,
    sort_mismatch, suggest_fixes, type_error, undefined_symbol, validate_arity,
    validate_arity_range, validate_max_arity, validate_min_arity,
};
pub use literal::{Lit, Var};
pub use sort::{DataTypeConstructor, Sort, SortId, SortKind};
pub use unsat_core::{UnsatCore, UnsatCoreBuilder, UnsatCoreStrategy};

// Rewrite exports
pub use rewrite::{
    ArithRewriter, ArrayRewriter, BoolRewriter, BottomUpRewriter, BvRewriter, CompositeRewriter,
    IteratingRewriter, Monomial, Polynomial, RewriteConfig, RewriteContext, RewriteResult,
    RewriteStats, RewriteStrategy, Rewriter, StringRewriter,
};

// Trait exports
pub use traits::{
    CachedRewriter as CachedRewriterTrait, ConditionalRewriter, FixpointRewriter, Goal,
    GoalMetadata, IterativeTactic, LazyPropagator, ModelBasedPropagator, OptimizingTheory,
    PropagationPriority, PropagationResult as TraitPropagationResult, Propagator,
    PropagatorManager, PropagatorStatus, QuantifiedTheory, RewriteConfig as TraitRewriteConfig,
    RewriteResult as TraitRewriteResult, RewriteStats as TraitRewriteStats,
    Rewriter as RewriterTrait, SequentialRewriter as SequentialRewriterTrait, Tactic,
    TacticCombinator, TacticConfig, TacticResult, TacticStats, Theory, TheoryCheckResult,
    TheoryConflict, TheoryModel, TheoryPropagationResult, WatchedPropagator,
};

// === std-only exports ===
#[cfg(feature = "std")]
pub use diagnostics::{Diagnostic, DiagnosticEmitter, Fix, RelatedDiagnostic, Severity};
#[cfg(feature = "std")]
pub use resource::{LimitStatus, ResourceManager};
#[cfg(feature = "std")]
pub use statistics::{Statistics, Timer};

// Allocator exports
#[cfg(feature = "std")]
pub use alloc::{
    Arena, ArenaConfig, ArenaError, ArenaHandle, ObjectPool, PoolConfig, PoolGuard, PoolStats,
    Region, RegionAllocator, RegionRef, RegionSlice, SharedObjectPool, SharedPoolGuard,
};

// Model exports
#[cfg(feature = "std")]
pub use model::{
    EvalCache, EvalResult, ImplicantConfig, ImplicantExtractor, Model, ModelCompletion,
    ModelCompletionConfig, ModelEvaluator, PrimeImplicant, Value, ValueFactory, ValueFactoryConfig,
};

// QE exports
#[cfg(feature = "std")]
pub use qe::{
    MbiConfig, MbiInterpolant, MbiSolver, MbiStats, QeLiteConfig, QeLiteResult, QeLiteSolver,
    QeLiteStats, TermGraph, TermGraphConfig, TermGraphStats, TermNode, TermNodeKind,
};
