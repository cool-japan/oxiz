//! Difference Logic Theory Solver
//!
//! Implements efficient reasoning about difference constraints of the form:
//! - x - y ≤ c (non-strict)
//! - x - y < c (strict, converted to x - y ≤ c - ε)
//!
//! # Algorithm
//!
//! Uses a constraint graph representation where:
//! - Variables are nodes
//! - Constraint `x - y ≤ c` becomes edge `(y → x)` with weight `c`
//!
//! Satisfiability checking via Bellman-Ford:
//! - UNSAT iff the graph contains a negative cycle
//! - Model: shortest distances from a virtual source node
//!
//! # Logics Supported
//!
//! - QF_IDL: Quantifier-Free Integer Difference Logic
//! - QF_RDL: Quantifier-Free Real Difference Logic
//!
//! # References
//!
//! - Cotton, S. & Maler, O. (2006). Fast and Flexible Difference Constraint Propagation
//! - Nieuwenhuis, R. & Oliveras, A. (2005). DPLL(T) with Exhaustive Theory Propagation

mod bellman_ford;
mod dense;
mod graph;
mod solver;

pub use bellman_ford::{BellmanFord, NegativeCycle};
pub use dense::DenseDiffLogic;
pub use graph::{ConstraintGraph, DiffConstraint, DiffEdge, DiffVar};
pub use solver::{DiffLogicConfig, DiffLogicResult, DiffLogicSolver, DiffLogicStats};
