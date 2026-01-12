//! UTVPI (Unit Two-Variable Per Inequality) Theory Solver
//!
//! Implements efficient reasoning about UTVPI constraints of the form:
//! - ±x ± y ≤ c (where coefficients are -1, 0, or 1)
//!
//! # Examples
//!
//! - x - y ≤ 5 (difference constraint)
//! - x + y ≤ 10 (sum constraint)
//! - -x + y ≤ 3 (equivalent to y - x ≤ 3)
//! - x ≤ 7 (single variable, y coefficient is 0)
//!
//! # Algorithm
//!
//! Uses a "doubled graph" transformation where each variable x is
//! represented by two nodes (x⁺, x⁻):
//! - x⁺ represents x
//! - x⁻ represents -x
//!
//! Constraint translation:
//! - x - y ≤ c → edge (y⁺ → x⁺, c) and (x⁻ → y⁻, c)
//! - x + y ≤ c → edge (y⁻ → x⁺, c) and (x⁻ → y⁺, c)
//! - -x - y ≤ c → edge (y⁺ → x⁻, c) and (x⁺ → y⁻, c)
//! - x ≤ c → edge (x⁻ → x⁺, 2c)
//! - -x ≤ c → edge (x⁺ → x⁻, 2c)
//!
//! Satisfiability is checked via Bellman-Ford on the doubled graph.
//! UNSAT iff the graph contains a negative cycle.
//!
//! # Logics Supported
//!
//! - QF_UTVPI: Quantifier-Free UTVPI (subset of QF_LIA)
//!
//! # References
//!
//! - Lahiri & Musuvathi (2005). An Efficient Decision Procedure for UTVPI Constraints

mod detection;
mod graph;
mod solver;

pub use detection::{DetectedConstraint, UtConstraintKind, UtvpiDetector, UtvpiDetectorStats};
pub use graph::{DoubledGraph, DoubledNode, Sign, UtConstraint, UtEdge};
pub use solver::{UtvpiConfig, UtvpiResult, UtvpiSolver, UtvpiStats};
