//! Conflict Analysis and Minimization.
//!
//! This module provides advanced conflict analysis techniques for CDCL(T)
//! solving, including conflict minimization, explanation generation, and
//! relevancy tracking.

pub mod implication_graph;
pub mod minimizer;
pub mod relevancy;
pub mod theory_explainer;

pub use implication_graph::{
    ImplicationGraph, ImplicationGraphConfig, ImplicationGraphStats, ImplicationNode, Level, Var,
};
pub use minimizer::{ConflictMinimizer, MinimizerConfig, MinimizerStats};
pub use relevancy::{ImplicationEdge, RelevancyConfig, RelevancyStats, RelevancyTracker};
pub use theory_explainer::{
    ExplainerConfig, ExplainerStats, ExplanationType, TheoryExplainer, TheoryExplanation,
};
