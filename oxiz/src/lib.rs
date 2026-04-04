//! # OxiZ - Next-Generation SMT Solver in Pure Rust
//!
//! **OxiZ** is a high-performance SMT (Satisfiability Modulo Theories) solver written entirely in Pure Rust,
//! designed to achieve feature parity with Z3 while leveraging Rust's safety, performance, and concurrency features.
//!
//! ## Key Features
//!
//! - **Pure Rust Implementation**: No C/C++ dependencies, no FFI, complete memory safety
//! - **Z3-Compatible**: Extensive theory support and familiar API patterns
//! - **High Performance**: Optimized SAT core with advanced heuristics (VSIDS, LRB, CHB)
//! - **Modular Design**: Use only what you need via feature flags
//! - **SMT-LIB2 Support**: Full parser and printer for standard format (requires `std` feature)
//! - **no_std Support**: Core solver compiles for bare-metal targets (RISC-V, zkVM)
//!
//! ## Module Overview
//!
//! ### Core Infrastructure (always available)
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`core`] | Term management, sorts, and SMT-LIB2 parsing |
//! | [`math`] | Mathematical utilities (polynomials, rationals, CAD) |
//! | `sat` | CDCL SAT solver with advanced heuristics |
//! | `theories` | Theory solvers (EUF, LRA, LIA, Arrays, BV, etc.) |
//! | `solver` | Main SMT solver with DPLL(T) |
//!
//! ### Advanced Features (require `std`)
//!
//! | Module | Feature Flag | Description |
//! |--------|--------------|-------------|
//! | `nlsat` | `nlsat` | Nonlinear real arithmetic solver |
//! | `opt` | `optimization` | MaxSMT and optimization |
//! | `spacer` | `spacer` | CHC solver for program verification |
//! | `proof` | `proof` | Proof generation and checking |
//!
//! ## Quick Start
//!
//! ### Installation
//!
//! Add OxiZ to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! oxiz = "0.2.0"  # Default: std + core solver
//! ```
//!
//! For no_std (e.g., zkVM):
//!
//! ```toml
//! [dependencies]
//! oxiz = { version = "0.2.0", default-features = false }
//! ```
//!
//! With additional features:
//!
//! ```toml
//! [dependencies]
//! oxiz = { version = "0.2.0", features = ["nlsat", "optimization"] }
//! ```
//!
//! Or use all features:
//!
//! ```toml
//! [dependencies]
//! oxiz = { version = "0.2.0", features = ["full"] }
//! ```
//!
//! ### Basic SMT Solving
//!
//! ```rust
//! use oxiz::{Solver, TermManager, SolverResult};
//! use num_bigint::BigInt;
//!
//! let mut solver = Solver::new();
//! let mut tm = TermManager::new();
//!
//! // Create variables
//! let x = tm.mk_var("x", tm.sorts.int_sort);
//! let y = tm.mk_var("y", tm.sorts.int_sort);
//!
//! // x + y = 10
//! let sum = tm.mk_add([x, y]);
//! let ten = tm.mk_int(BigInt::from(10));
//! let eq1 = tm.mk_eq(sum, ten);
//!
//! // x > 5
//! let five = tm.mk_int(BigInt::from(5));
//! let gt = tm.mk_gt(x, five);
//!
//! // Assert constraints
//! solver.assert(eq1, &mut tm);
//! solver.assert(gt, &mut tm);
//!
//! // Check satisfiability
//! match solver.check(&mut tm) {
//!     SolverResult::Sat => {
//!         println!("SAT");
//!         if let Some(model) = solver.model() {
//!             println!("Model: {:?}", model);
//!         }
//!     }
//!     SolverResult::Unsat => println!("UNSAT"),
//!     SolverResult::Unknown => println!("Unknown"),
//! }
//! ```
//!
//! ### SMT-LIB2 Format
//!
//! ```rust
//! use oxiz::{Solver, TermManager};
//!
//! let mut solver = Solver::new();
//! let mut tm = TermManager::new();
//!
//! // Parse SMT-LIB2 script
//! let script = r#"
//!     (declare-const x Int)
//!     (declare-const y Int)
//!     (assert (= (+ x y) 10))
//!     (assert (> x 5))
//!     (check-sat)
//! "#;
//!
//! // Execute commands
//! // solver.execute_script(script, &mut tm)?;
//! ```
//!
//! ## Feature Flags
//!
//! | Feature | Description | Default |
//! |---------|-------------|---------|
//! | `std` | Standard library support | Yes |
//! | `nlsat` | Nonlinear real arithmetic (implies std) | |
//! | `optimization` | MaxSMT and optimization (implies std) | |
//! | `spacer` | CHC solver (implies std) | |
//! | `proof` | Proof generation (implies std) | |
//! | `standard` | All common features except SPACER | |
//! | `full` | All features | |
//!
//! ## Theory Support
//!
//! - **EUF**: Equality and uninterpreted functions
//! - **LRA**: Linear real arithmetic (simplex)
//! - **LIA**: Linear integer arithmetic (branch-and-bound, cuts)
//! - **NRA**: Nonlinear real arithmetic (NLSAT, CAD)
//! - **Arrays**: Theory of arrays with extensionality
//! - **BitVectors**: Bit-precise reasoning
//! - **Strings**: String operations with regex
//! - **Datatypes**: Algebraic data types
//! - **Floating-Point**: IEEE 754 semantics

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(not(feature = "std"))]
extern crate alloc;

// Re-export core modules (always available)
pub use oxiz_core as core;
pub use oxiz_math as math;

// Re-export solver components (always available now)
pub use oxiz_sat as sat;
pub use oxiz_solver as solver;
pub use oxiz_solver::{Solver, SolverResult};
pub use oxiz_theories as theories;

// Re-export core types for convenience
pub use oxiz_core::{Sort, SortId, Term, TermId, TermManager};

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub mod easy;

pub use oxiz_solver::resource_limits;

// Re-export advanced features (require std)
#[cfg(feature = "nlsat")]
#[cfg_attr(docsrs, doc(cfg(feature = "nlsat")))]
pub use oxiz_nlsat as nlsat;

#[cfg(feature = "optimization")]
#[cfg_attr(docsrs, doc(cfg(feature = "optimization")))]
pub use oxiz_opt as opt;

#[cfg(feature = "spacer")]
#[cfg_attr(docsrs, doc(cfg(feature = "spacer")))]
pub use oxiz_spacer as spacer;

#[cfg(feature = "proof")]
#[cfg_attr(docsrs, doc(cfg(feature = "proof")))]
pub use oxiz_proof as proof;

/// Current version of OxiZ
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        assert!(!super::VERSION.is_empty());
    }
}
