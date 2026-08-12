//! OxiZ Solver - Main CDCL(T) SMT Solver
//!
//! This crate provides the main solver API that orchestrates:
//! - SAT core (CDCL)
//! - Theory solvers (EUF, LRA, LIA, BV)
//! - Tactic framework
//! - Parallel portfolio solving

#![allow(clippy::collapsible_if)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::for_kv_map)]
#![allow(clippy::too_many_arguments)]
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

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(not(feature = "std"))]
extern crate alloc;

// The wasm32 clock guard. Every `Instant` / `SystemTime` this crate reads goes
// through `oxiz-time`, whose types ARE `std::time`'s on every target with a
// working clock and frozen stubs on wasm32-unknown-unknown (which has none and
// aborts on `Instant::now()`); see `oxiz_time`'s crate docs.
//
// A missing `"oxiz-time/std"` in this crate's `std` feature list would silently
// hand a *native* build the frozen clock -- timeouts that never fire, timing
// statistics stuck at zero. Catch that here, at compile time, instead of in
// production.
#[cfg(all(
    feature = "std",
    not(all(target_arch = "wasm32", target_os = "unknown"))
))]
const _: () = assert!(
    !oxiz_time::IS_FROZEN,
    "oxiz-time/std must be forwarded from this crate's `std` feature"
);

mod prelude;

mod context;
mod nelson_oppen;
mod optimization;
pub mod resource_limits;
mod simplify;
#[cfg(feature = "std")]
pub mod skolemization;
mod solver;

pub use context::Context;
pub use nelson_oppen::{NelsonOppenCombiner, NelsonOppenStats, TermTheory, TheoryId};
pub use optimization::{Objective, ObjectiveKind, OptimizationResult, Optimizer, ParetoPoint};
#[cfg(feature = "std")]
pub use oxiz_proof::replay::VerificationResult;
pub use solver::{Model, Proof, ProofStep, Solver, SolverConfig, SolverResult, TheoryMode};

// Re-export types from oxiz-sat
pub use oxiz_sat::{RestartStrategy, SolverStats};

// Re-export theory combination types from oxiz-theories
pub use oxiz_theories::{EqualityNotification, TheoryCombination};

// Phase 2 enhancements
pub mod combination;
pub mod conflict;
pub mod delayed_combination;
pub mod model;
pub mod propagation;
pub mod propagation_pipeline;
pub mod shared_terms;

// MBQI module (Model-Based Quantifier Instantiation)
pub mod mbqi;

// Z3 API compatibility layer (std-only)
#[cfg(feature = "std")]
pub mod z3_compat;

// Debugging support: state snapshots / DOT graphs, event tracing, conflict and
// UNSAT explanation, and model minimization.
pub mod debug;

// Read-only structural invariant checks over the solver's own bookkeeping.
pub mod invariants;
