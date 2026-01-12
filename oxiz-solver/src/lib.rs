//! OxiZ Solver - Main CDCL(T) SMT Solver
//!
//! This crate provides the main solver API that orchestrates:
//! - SAT core (CDCL)
//! - Theory solvers (EUF, LRA, LIA, BV)
//! - Tactic framework
//! - Parallel portfolio solving
//!
//! # Examples
//!
//! ## Basic Boolean Satisfiability
//!
//! ```
//! use oxiz_solver::{Solver, SolverResult};
//! use oxiz_core::ast::TermManager;
//!
//! let mut solver = Solver::new();
//! let mut tm = TermManager::new();
//!
//! // Create boolean variables
//! let p = tm.mk_var("p", tm.sorts.bool_sort);
//! let q = tm.mk_var("q", tm.sorts.bool_sort);
//!
//! // Assert p AND q
//! let formula = tm.mk_and(vec![p, q]);
//! solver.assert(formula, &mut tm);
//!
//! // Check satisfiability
//! match solver.check(&mut tm) {
//!     SolverResult::Sat => println!("Satisfiable!"),
//!     SolverResult::Unsat => println!("Unsatisfiable!"),
//!     SolverResult::Unknown => println!("Unknown"),
//! }
//! ```
//!
//! ## Integer Arithmetic
//!
//! ```
//! use oxiz_solver::{Solver, SolverResult};
//! use oxiz_core::ast::TermManager;
//! use num_bigint::BigInt;
//!
//! let mut solver = Solver::new();
//! let mut tm = TermManager::new();
//!
//! solver.set_logic("QF_LIA");
//!
//! // Create integer variable
//! let x = tm.mk_var("x", tm.sorts.int_sort);
//!
//! // Assert: x >= 5 AND x <= 10
//! let five = tm.mk_int(BigInt::from(5));
//! let ten = tm.mk_int(BigInt::from(10));
//! solver.assert(tm.mk_ge(x, five), &mut tm);
//! solver.assert(tm.mk_le(x, ten), &mut tm);
//!
//! // Should be satisfiable
//! assert_eq!(solver.check(&mut tm), SolverResult::Sat);
//! ```
//!
//! ## Incremental Solving with Push/Pop
//!
//! ```
//! use oxiz_solver::{Solver, SolverResult};
//! use oxiz_core::ast::TermManager;
//!
//! let mut solver = Solver::new();
//! let mut tm = TermManager::new();
//!
//! let p = tm.mk_var("p", tm.sorts.bool_sort);
//! solver.assert(p, &mut tm);
//!
//! // Push a new scope
//! solver.push();
//! let q = tm.mk_var("q", tm.sorts.bool_sort);
//! solver.assert(q, &mut tm);
//! assert_eq!(solver.check(&mut tm), SolverResult::Sat);
//!
//! // Pop back to previous scope
//! solver.pop();
//! assert_eq!(solver.check(&mut tm), SolverResult::Sat);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod context;
mod mbqi;
mod optimization;
mod simplify;
mod solver;

pub use context::Context;
pub use mbqi::{Instantiation, MBQIResult, MBQISolver, MBQIStats, QuantifiedFormula};
pub use optimization::{Objective, ObjectiveKind, OptimizationResult, Optimizer, ParetoPoint};
pub use solver::{Model, Proof, ProofStep, Solver, SolverConfig, SolverResult, TheoryMode};

// Re-export types from oxiz-sat
pub use oxiz_sat::{RestartStrategy, SolverStats};

// Re-export theory combination types from oxiz-theories
pub use oxiz_theories::{EqualityNotification, TheoryCombination};
