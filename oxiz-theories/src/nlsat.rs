//! NLSAT Theory Wrapper
//!
//! This module wraps the NLSAT solver (from oxiz-nlsat) to provide Theory trait
//! implementation for nonlinear arithmetic (QF_NIA and QF_NRA).
//!
//! ## Current Status
//!
//! This is a minimal wrapper that provides the structure for NLSAT integration.
//! Full constraint processing requires refactoring the solver to parse nonlinear
//! constraints (multiplication, polynomial operations) before passing them to theories.
//!
//! ## Architecture
//!
//! - `NlsatTheory`: Main wrapper implementing `Theory` trait
//! - Handles both Real (QF_NRA) and Integer (QF_NIA) nonlinear arithmetic
//! - Delegates to `NlsatSolver` (real) or `NiaSolver` (integer)
//!
//! ## Reference
//!
//! - Z3's NLSAT integration in nlsat/nlsat_explain.cpp
//! - NLSAT solver: oxiz-nlsat::solver::NlsatSolver
//! - Integer solver: oxiz-nlsat::nia::NiaSolver

use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use oxiz_nlsat::nia::NiaSolver;
use oxiz_nlsat::solver::{NlsatSolver, SolverResult};

/// Context state for push/pop support
#[derive(Debug, Clone)]
struct NlsatContextState {
    /// Marker for context level
    level: usize,
}

/// Wrapper around NlsatSolver or NiaSolver
enum NlsatSolverWrapper {
    /// Real arithmetic (QF_NRA)
    Real(NlsatSolver),
    /// Integer arithmetic (QF_NIA)
    Integer(NiaSolver),
}

impl std::fmt::Debug for NlsatSolverWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Real(_) => write!(f, "NlsatSolverWrapper::Real(..)"),
            Self::Integer(_) => write!(f, "NlsatSolverWrapper::Integer(..)"),
        }
    }
}

impl NlsatSolverWrapper {
    fn new(integer: bool) -> Self {
        if integer {
            Self::Integer(NiaSolver::new())
        } else {
            Self::Real(NlsatSolver::new())
        }
    }

    fn solve(&mut self) -> SolverResult {
        match self {
            Self::Real(s) => s.solve(),
            Self::Integer(s) => s.solve(),
        }
    }
}

/// NLSAT Theory Solver for nonlinear arithmetic
///
/// Supports both real (QF_NRA) and integer (QF_NIA) nonlinear arithmetic.
///
/// # Current Implementation
///
/// This is a minimal placeholder that demonstrates the integration pattern.
/// Full implementation requires:
///
/// 1. Solver-side parsing of nonlinear constraints (x*y, x^2, etc.)
/// 2. Term-to-polynomial conversion API
/// 3. Constraint assertion methods that accept parsed polynomials
/// 4. Model extraction and conflict clause generation
///
/// The challenge is that the `Theory` trait doesn't receive a `TermManager`,
/// so term parsing must happen at the solver level before delegation to theories.
#[derive(Debug)]
pub struct NlsatTheory {
    /// Underlying NLSAT solver (Real or Integer variant)
    solver: NlsatSolverWrapper,
    /// Context stack for push/pop
    context_stack: Vec<NlsatContextState>,
    /// Is this integer arithmetic (NIA) or real arithmetic (NRA)?
    is_integer: bool,
    /// Cached result from last check
    last_result: Option<SolverResult>,
    /// Asserted terms (for conflict generation)
    asserted_terms: Vec<TermId>,
}

impl NlsatTheory {
    /// Create a new NLSAT theory solver
    ///
    /// # Arguments
    ///
    /// * `integer` - true for QF_NIA (integer), false for QF_NRA (real)
    pub fn new(integer: bool) -> Self {
        Self {
            solver: NlsatSolverWrapper::new(integer),
            context_stack: Vec::new(),
            is_integer: integer,
            last_result: None,
            asserted_terms: Vec::new(),
        }
    }
}

impl Theory for NlsatTheory {
    fn id(&self) -> TheoryId {
        if self.is_integer {
            TheoryId::NIA
        } else {
            TheoryId::NRA
        }
    }

    fn name(&self) -> &str {
        if self.is_integer { "NIA" } else { "NRA" }
    }

    fn can_handle(&self, _term: TermId) -> bool {
        // NLSAT handles arithmetic terms
        // A full implementation would check if the term contains nonlinear operations
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // Record the term for conflict generation
        self.asserted_terms.push(term);

        // TODO: Parse term into polynomial constraint and assert to NLSAT solver
        // This requires:
        // 1. Access to TermManager (not available in Theory trait)
        // 2. Term-to-polynomial conversion
        // 3. Atom creation in NLSAT solver
        //
        // For now, this is a placeholder that allows the solver to compile.
        // Full implementation requires refactoring constraint processing in the solver.

        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        // Record the negated term
        self.asserted_terms.push(term);

        // TODO: Similar to assert_true, but negate the constraint

        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        // Run NLSAT solver
        let result = self.solver.solve();
        self.last_result = Some(result);

        match result {
            SolverResult::Sat => Ok(TheoryResult::Sat),
            SolverResult::Unsat => {
                // Return all asserted terms as conflict
                // A full implementation would extract a minimal unsat core
                let conflict = self.asserted_terms.clone();
                Ok(TheoryResult::Unsat(conflict))
            }
            SolverResult::Unknown => Ok(TheoryResult::Unknown),
        }
    }

    fn push(&mut self) {
        self.context_stack.push(NlsatContextState {
            level: self.asserted_terms.len(),
        });

        // Note: NLSAT solver doesn't have push/pop yet
        // We simulate it by tracking assertion levels
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Restore state by truncating assertions
            self.asserted_terms.truncate(state.level);
        }
    }

    fn reset(&mut self) {
        *self = Self::new(self.is_integer);
    }

    fn get_model(&self) -> Vec<(TermId, TermId)> {
        // TODO: Extract model from NLSAT solver
        // This requires:
        // 1. Variable-to-term mapping
        // 2. Value-to-term conversion (BigRational to TermId)
        // 3. TermManager to create value terms
        //
        // For now, return empty model
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nlsat_theory_new() {
        let theory_nia = NlsatTheory::new(true);
        assert_eq!(theory_nia.id(), TheoryId::NIA);
        assert_eq!(theory_nia.name(), "NIA");
        assert!(theory_nia.is_integer);

        let theory_nra = NlsatTheory::new(false);
        assert_eq!(theory_nra.id(), TheoryId::NRA);
        assert_eq!(theory_nra.name(), "NRA");
        assert!(!theory_nra.is_integer);
    }

    #[test]
    fn test_nlsat_theory_push_pop() {
        let mut theory = NlsatTheory::new(false);

        assert_eq!(theory.context_stack.len(), 0);

        theory.push();
        assert_eq!(theory.context_stack.len(), 1);

        theory.push();
        assert_eq!(theory.context_stack.len(), 2);

        theory.pop();
        assert_eq!(theory.context_stack.len(), 1);

        theory.pop();
        assert_eq!(theory.context_stack.len(), 0);
    }

    #[test]
    fn test_nlsat_theory_reset() {
        let mut theory = NlsatTheory::new(false);

        let term = TermId::new(1);
        let _ = theory.assert_true(term);

        assert!(!theory.asserted_terms.is_empty());

        theory.reset();

        assert!(theory.asserted_terms.is_empty());
        assert!(theory.context_stack.is_empty());
    }

    #[test]
    fn test_nlsat_theory_can_handle() {
        let theory = NlsatTheory::new(false);

        // For now, returns true for all terms
        let term = TermId::new(1);
        assert!(theory.can_handle(term));
    }

    #[test]
    fn test_nlsat_theory_check_placeholder() {
        let mut theory = NlsatTheory::new(false);

        // Check with no assertions should return Sat
        let result = theory.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }
}
