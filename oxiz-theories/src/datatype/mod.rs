//! Algebraic Datatype Theory Solver
//!
//! This module provides support for user-defined algebraic datatypes (ADTs)
//! in SMT solving. ADTs include:
//!
//! - **Enumerations**: Finite sets of named constants
//! - **Records**: Product types with named fields
//! - **Recursive types**: Lists, trees, and other inductive structures
//!
//! ## Implementation Strategy
//!
//! The solver uses a combination of:
//! - **Constructor tagging**: Each value has a constructor tag
//! - **Selector axioms**: Extracting fields from constructors
//! - **Distinctness**: Different constructors produce different values
//! - **Acyclicity**: No cyclic datatype values in finite models
//!
//! ## SMT-LIB2 Support
//!
//! Supports the `declare-datatypes` command for defining datatypes:
//! ```smt2
//! (declare-datatypes ((List 1)) (
//!   (par (T) ((nil) (cons (head T) (tail (List T)))))))
//! ```

pub mod constructor_theory;
mod solver;

pub use constructor_theory::{
    CaseBranch, ConstructorDefinition, ConstructorId, ConstructorPattern, ConstructorStats,
    ConstructorTheory, DatatypeDefinition, DatatypeId, DisjointnessConstraint, InductionProof,
    InductionProperty, InjectivityConstraint, SelectorDefinition,
};
pub use solver::{Constructor, DatatypeDecl, DatatypeSolver, DatatypeSort, Field, Selector};
