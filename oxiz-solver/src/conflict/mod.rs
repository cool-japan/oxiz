//! Conflict Analysis and Minimization.
//!
//! This module provides advanced conflict analysis techniques for CDCL(T)
//! solving, including conflict minimization, explanation generation, and
//! relevancy tracking.

pub mod clause_learning;
pub mod explanation_cache;
pub mod explanation_generator;
pub mod implication_graph;
pub mod minimizer;
pub mod recursive_minimizer;
pub mod relevancy;
pub mod theory_explainer;
pub mod uip_selection;

pub use clause_learning::{
    ClauseId, ClauseLearner, ClauseLearningConfig, ClauseLearningStats, ClauseMinimizer,
    ImplicationGraph as ClauseImplicationGraph, ImplicationNode as ClauseImplicationNode,
    LearnedClause, LearnedDatabase,
};

pub use explanation_cache::{
    CacheConfig, CacheKey, CacheStats, Explanation as CachedExplanation, ExplanationCache,
};
pub use explanation_generator::{
    Explanation, ExplanationConfig, ExplanationGenerator, ExplanationStats,
};
pub use implication_graph::{
    ImplicationGraph, ImplicationGraphConfig, ImplicationGraphStats, ImplicationNode, Level, Var,
};
pub use minimizer::{ConflictMinimizer, MinimizerConfig, MinimizerStats};
pub use recursive_minimizer::{
    Lit as RecursiveLit, Reason, RecursiveMinConfig, RecursiveMinStats, RecursiveMinimizer,
};
pub use relevancy::{ImplicationEdge, RelevancyConfig, RelevancyStats, RelevancyTracker};
pub use theory_explainer::{
    ExplainerConfig, ExplainerStats, ExplanationType, TheoryExplainer, TheoryExplanation,
};
pub use uip_selection::{
    ImplicationNode as UipImplicationNode, Level as UipLevel, UipConfig, UipSelector, UipStats,
    UipStrategy,
};
