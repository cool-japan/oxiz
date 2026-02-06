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
//! - **Unicode normalization**: NFC, NFD, NFKC, NFKD
//! - **Advanced regex**: Capture groups, backreferences, lookahead/lookbehind
//! - **Length reasoning**: Constraint propagation and bounds analysis
//! - **Replace operations**: replaceAll, replaceFirst with constraint generation
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

pub mod advanced_regex;
pub mod automata;
pub mod char_ops;
pub mod normalization;
mod regex;
pub mod regex_solver;
pub mod replace_operations;
pub mod sequence;
mod solver;
pub mod string_length_reasoning;
mod unicode;
pub mod word_eq;

// Core exports
pub use regex::{Regex, RegexOp};
pub use regex_solver::{
    Regex as RegexSolverRegex, RegexSolver, RegexSolverConfig, RegexSolverStats, StrVar,
    StringConstraint,
};
pub use solver::{StringAtom, StringExpr, StringSolver};
pub use unicode::UnicodeCategory;

// Sequence operation exports
pub use sequence::{
    IntExpr, RegexId, SeqConstraint, SeqConstraintGen, SeqEvaluator, SeqExpr, SeqResult,
    SeqRewriter, StringBuilder,
};

// Word equation exports
pub use word_eq::{
    CaseSplit, Conflict, ConflictReason, LengthAbstraction,
    LengthConstraint as WordEqLengthConstraint, LinearConstraint, Relation,
    SolveResult as WordEqSolveResult, Substitution, WordEqConfig, WordEqSolver, WordEqStats,
};

// Automata exports
pub use automata::{ConstraintAutomaton, Dfa, Label, Nfa, ProductAutomaton, StateId, Transition};

// Character operations exports
pub use char_ops::{CharAt, CharClass, CharOpSolver, CharOpStats, CodePoint, PredefinedClass};

// Normalization exports
pub use normalization::{
    CombiningClass, Decomposition, NormalizationConstraint, NormalizationForm, NormalizationSolver,
    UnicodeNormalizer,
};

// Advanced regex exports
pub use advanced_regex::{
    AdvancedRegex, BinaryProperty, CaptureGroup, CharacterClass, Condition, Match, RegexBuilder,
    RegexMatcher, UnicodeBlock, UnicodeProperty, UnicodeScript,
};

// Length reasoning exports
pub use string_length_reasoning::{
    ArithmeticConstraint, LengthBound, LengthConstraint, LengthSolver, LengthSolverStats,
    LengthVar, StringOp,
};

// Replace operations exports
pub use replace_operations::{
    Pattern, ReplaceAnalyzer, ReplaceBuilder, ReplaceConstraint, ReplaceConstraintGen, ReplaceMode,
    ReplaceSolver, ReplaceSolverStats,
};
