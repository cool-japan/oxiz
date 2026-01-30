//! Trait-Based Architecture for OxiZ.
//!
//! This module defines core traits that enable modular, composable, and
//! extensible SMT solver components.

pub mod propagator;
pub mod rewriter;
pub mod tactic;
pub mod theory;

pub use theory::{
    OptimizingTheory, PropagationResult as TheoryPropagationResult, QuantifiedTheory, Theory,
    TheoryCheckResult, TheoryConflict, TheoryModel,
};

pub use propagator::{
    LazyPropagator, ModelBasedPropagator, PropagationPriority, PropagationResult, Propagator,
    PropagatorManager, PropagatorStatus, WatchedPropagator,
};

pub use rewriter::{
    CachedRewriter, ConditionalRewriter, FixpointRewriter, RewriteConfig, RewriteResult,
    RewriteStats, Rewriter, SequentialRewriter,
};

pub use tactic::{
    Goal, GoalMetadata, IterativeTactic, Tactic, TacticCombinator, TacticConfig, TacticResult,
    TacticStats,
};
