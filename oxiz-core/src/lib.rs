//! OxiZ Core - AST, Sorts, and Tactics for the SMT Solver
//!
//! This crate provides the foundational types and traits for the OxiZ SMT solver:
//! - Arena-allocated terms with efficient [`TermId`] references
//! - Sort system for type checking
//! - SMT-LIB2 parser and printer
//! - Tactic framework for preprocessing and simplification
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

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod ast;
pub mod config;
pub mod datalog;
pub mod error;
pub mod model;
pub mod qe;
pub mod resource;
pub mod rewrite;
pub mod smtlib;
pub mod sort;
pub mod statistics;
pub mod tactic;
pub mod theories;
pub mod unsat_core;

pub use ast::{Term, TermId, TermKind, TermManager};
pub use config::{
    ClauseDeletionStrategy, Config, GeneralParams, PhaseSaving, ResourceLimits, SatParams,
    SimplifyParams,
};
pub use error::{OxizError, Result};
pub use resource::{LimitStatus, ResourceManager};
pub use sort::{DataTypeConstructor, Sort, SortId, SortKind};
pub use statistics::{Statistics, Timer};
pub use unsat_core::{UnsatCore, UnsatCoreBuilder, UnsatCoreStrategy};

// Model exports
pub use model::{
    EvalCache, EvalResult, ImplicantConfig, ImplicantExtractor, Model, ModelCompletion,
    ModelCompletionConfig, ModelEvaluator, PrimeImplicant, Value, ValueFactory, ValueFactoryConfig,
};

// QE exports
pub use qe::{
    MbiConfig, MbiInterpolant, MbiSolver, MbiStats, QeLiteConfig, QeLiteResult, QeLiteSolver,
    QeLiteStats, TermGraph, TermGraphConfig, TermGraphStats, TermNode, TermNodeKind,
};

// Rewrite exports
pub use rewrite::{
    ArithRewriter, ArrayRewriter, BoolRewriter, BottomUpRewriter, BvRewriter, CompositeRewriter,
    IteratingRewriter, Monomial, Polynomial, RewriteConfig, RewriteContext, RewriteResult,
    RewriteStats, RewriteStrategy, Rewriter, StringRewriter,
};
