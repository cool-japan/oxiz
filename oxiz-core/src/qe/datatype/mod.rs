//! Datatype Quantifier Elimination.
//!
//! This module provides quantifier elimination for algebraic datatype formulas.

pub mod case_analysis;
pub mod plugin;

pub use case_analysis::{
    CaseAnalysisConfig, CaseAnalysisResult, CaseAnalysisStats, CaseAnalyzer,
    Constructor as CaseConstructor,
};
pub use plugin::{
    Constructor, Datatype, DatatypeConstraint, DatatypeQeConfig, DatatypeQePlugin, DatatypeQeStats,
};
