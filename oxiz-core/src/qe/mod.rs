//! Quantifier Elimination Enhancements
//!
//! Provides advanced quantifier elimination techniques:
//! - Term Graph: For analyzing term structure in QE
//! - QE Lite: Fast approximate quantifier elimination
//! - MBI: Model-Based Interpolation
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::qe::{TermGraph, QeLite, Mbi};
//!
//! // Build term graph for analysis
//! let graph = TermGraph::from_formula(formula, manager);
//! let important_vars = graph.identify_important_variables();
//!
//! // Try fast QE first
//! let lite = QeLite::new();
//! if let Some(result) = lite.eliminate(formula, vars) {
//!     return result;
//! }
//!
//! // Fall back to full QE
//! ```

mod mbi;
mod qe_lite;
mod term_graph;

pub use mbi::{MbiConfig, MbiInterpolant, MbiSolver, MbiStats};
pub use qe_lite::{QeLiteConfig, QeLiteResult, QeLiteSolver, QeLiteStats};
pub use term_graph::{TermGraph, TermGraphConfig, TermGraphStats, TermNode, TermNodeKind};
