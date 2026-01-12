//! Theory trait and common types

use oxiz_core::ast::TermId;
use oxiz_core::error::Result;

/// Unique identifier for a theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TheoryId {
    /// Boolean theory (always present)
    Bool,
    /// Equality with Uninterpreted Functions
    EUF,
    /// Linear Real Arithmetic
    LRA,
    /// Linear Integer Arithmetic
    LIA,
    /// BitVectors
    BV,
    /// Arrays
    Arrays,
    /// Floating-Point (IEEE 754)
    FP,
    /// Algebraic Datatypes
    Datatype,
    /// Strings and Regular Expressions
    Strings,
}

/// Result of a theory check
#[derive(Debug, Clone)]
pub enum TheoryResult {
    /// The theory constraints are satisfiable
    Sat,
    /// The theory constraints are unsatisfiable with explanation
    Unsat(Vec<TermId>),
    /// Need to propagate these literals
    Propagate(Vec<(TermId, Vec<TermId>)>),
    /// Unknown (incomplete theory)
    Unknown,
}

/// Trait for theory solvers
pub trait Theory: Send + Sync {
    /// Get the theory identifier
    fn id(&self) -> TheoryId;

    /// Get the name of this theory
    fn name(&self) -> &str;

    /// Check if this theory can handle a term
    fn can_handle(&self, term: TermId) -> bool;

    /// Assert a term as true
    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult>;

    /// Assert a term as false
    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult>;

    /// Check consistency of current assertions
    fn check(&mut self) -> Result<TheoryResult>;

    /// Push a context level
    fn push(&mut self);

    /// Pop a context level
    fn pop(&mut self);

    /// Reset the theory solver
    fn reset(&mut self);

    /// Get the current model (if satisfiable)
    fn get_model(&self) -> Vec<(TermId, TermId)> {
        Vec::new()
    }
}

/// Equality notification for Nelson-Oppen theory combination
/// When one theory derives that two terms are equal, it notifies other theories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EqualityNotification {
    /// Left-hand side of the equality
    pub lhs: TermId,
    /// Right-hand side of the equality
    pub rhs: TermId,
    /// The reason/justification for this equality (for proof generation)
    pub reason: Option<TermId>,
}

/// Interface for theory combination via Nelson-Oppen
/// Theories implement this to participate in equality sharing
pub trait TheoryCombination {
    /// Notify the theory of a new equality derived by another theory
    /// Returns true if the equality was accepted and processed
    fn notify_equality(&mut self, eq: EqualityNotification) -> bool;

    /// Get all equalities derived by this theory that should be shared
    /// These are equalities between terms that appear in multiple theories
    fn get_shared_equalities(&self) -> Vec<EqualityNotification>;

    /// Check if a term is relevant to this theory
    fn is_relevant(&self, term: TermId) -> bool;
}
