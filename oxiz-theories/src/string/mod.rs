//! String Theory Solver
//!
//! This module provides a solver for the theory of strings (SMT-LIB QF_S, QF_SLIA).
//! It supports:
//!
//! - **String operations**: concatenation, length, substring, indexOf, replace
//! - **String predicates**: prefix, suffix, contains
//! - **Regular expressions**: membership testing using Brzozowski derivatives
//! - **String-Integer conversion**: str.to_int, str.from_int
//! - **Sequence operations**: extract, replace, replace_all, at, unit
//! - **Word equation solving**: Nielsen transformation, Levi's lemma
//! - **Automata-based solving**: NFA/DFA construction, product automaton
//!
//! ## Implementation Strategy
//!
//! The solver uses a combination of:
//! - **Word equations**: Solving string equalities via Levi's lemma and Nielsen transformations
//! - **Length abstraction**: Translating length constraints to linear arithmetic
//! - **Automata-based regex**: Brzozowski derivatives for regex membership
//! - **Conflict-driven refinement**: Lazy instantiation of axioms
//!
//! ## SMT-LIB2 Support
//!
//! Supports standard string operations:
//! ```smt2
//! (declare-const s String)
//! (assert (str.contains s "hello"))
//! (assert (> (str.len s) 10))
//! ```

pub mod automata;
mod regex;
pub mod sequence;
mod solver;
mod unicode;
pub mod word_eq;

// Core exports
pub use regex::{Regex, RegexOp};
pub use solver::StringSolver;
pub use unicode::UnicodeCategory;

// Sequence operation exports
pub use sequence::{
    IntExpr, RegexId, SeqConstraint, SeqConstraintGen, SeqEvaluator, SeqExpr, SeqResult,
    SeqRewriter, StringBuilder,
};

// Word equation exports
pub use word_eq::{
    CaseSplit, Conflict, ConflictReason, LengthAbstraction, LengthConstraint, LinearConstraint,
    Relation, SolveResult as WordEqSolveResult, Substitution, WordEqConfig, WordEqSolver,
    WordEqStats,
};

// Automata exports
pub use automata::{ConstraintAutomaton, Dfa, Label, Nfa, ProductAutomaton, StateId, Transition};
