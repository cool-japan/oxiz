//! Theory modules for SMT reasoning
//!
//! This module provides theory-specific reasoning capabilities:
//! - Array theory (select/store axioms)
//! - String theory (concatenation, length, substring, etc.)
//! - BitVector theory (fixed-width arithmetic and bitwise operations)
//! - FloatingPoint theory (IEEE 754 arithmetic)
//! - Datatype theory (algebraic datatypes with constructors, testers, selectors)
//! - Theory combination via Nelson-Oppen
//!
//! Reference: Z3's `src/ast/` and `src/smt/theory_*.h`

pub mod array;
pub mod bitvector;
pub mod combination;
pub mod datatype;
pub mod floatingpoint;
pub mod lemma_cache;
pub mod string;

pub use array::{ArrayAxiom, ArrayTheory};
pub use bitvector::{BitVectorAxiom, BitVectorOp, BitVectorStatistics, BitVectorTheory};
pub use combination::{NelsonOppen, TheoryCombiner};
pub use datatype::{DatatypeAxiom, DatatypeStatistics, DatatypeTheory};
pub use floatingpoint::{
    FloatingPointAxiom, FloatingPointOp, FloatingPointStatistics, FloatingPointTheory,
};
pub use lemma_cache::{CacheStatistics, Lemma, LemmaCache, TheoryId};
pub use string::{StringAxiom, StringTheory};
