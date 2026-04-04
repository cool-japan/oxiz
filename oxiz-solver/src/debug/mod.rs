//! Debugging Support for OxiZ Solver.
//!
//! This module provides comprehensive debugging tools for SMT solving:
//!
//! - **Visualization**: Solver state snapshots and DOT graph generation
//! - **Tracing**: Event recording and trace generation
//! - **Conflict Explanation**: Human-readable UNSAT and conflict explanations
//! - **Model Minimization**: Finding minimal satisfying models
//!
//! All functionality is feature-gated behind the `debug` feature.

#[allow(unused_imports)]
use crate::prelude::*;

pub mod explain;
pub mod model_min;
pub mod trace;
pub mod visualize;

pub use explain::{ConflictExplainer, ConflictExplanation, UnsatExplanation};
pub use model_min::{ModelMinResult, ModelMinimizer as DebugModelMinimizer};
pub use trace::{SolverTracer, TraceConfig, TraceEvent};
pub use visualize::{ImplicationGraphDot, SolverStateSnapshot};
