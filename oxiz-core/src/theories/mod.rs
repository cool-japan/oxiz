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
//! # Scope
//!
//! This is `oxiz-core`'s own lightweight theory layer: it is self-contained,
//! depends on nothing outside this crate, and is meant for callers that want a
//! little theory reasoning over a [`crate::ast::TermManager`] without pulling
//! in a solver. Each theory registers the terms it understands, instantiates
//! the axioms it can state as terms, propagates equalities, and reports the
//! conflicts it can see.
//!
//! It is deliberately incomplete, and none of these theories is a decision
//! procedure: "no conflict found" never means "satisfiable". The theories that
//! `oxiz-solver` actually runs live in the `oxiz-theories` crate
//! (`oxiz_theories::{bv, fp, datatype, string, array}`) — bit-blasting,
//! word-level propagation, and the rest are there, not here.
//!
//! Reference: Z3's `src/ast/` and `src/smt/theory_*.h`

#[allow(unused_imports)]
use crate::prelude::*;

pub mod array;
pub mod bitvector;
pub mod combination;
pub mod datatype;
pub(crate) mod eq_classes;
pub mod floatingpoint;
pub mod lemma_cache;
pub mod string;

pub use array::{ArrayAxiom, ArrayTheory};
pub use bitvector::{BitVectorAxiom, BitVectorOp, BitVectorStatistics, BitVectorTheory};
pub use combination::{
    CombinationStats, CombinerOutcome, NelsonOppen, Theory, TheoryCombiner, TheoryResult,
};
pub use datatype::{DatatypeAxiom, DatatypeStatistics, DatatypeTheory};
pub use floatingpoint::{
    FloatingPointAxiom, FloatingPointOp, FloatingPointStatistics, FloatingPointTheory, SpecialValue,
};
pub use lemma_cache::{CacheStatistics, Lemma, LemmaCache, TheoryId};
pub use string::{StringAxiom, StringTheory};
