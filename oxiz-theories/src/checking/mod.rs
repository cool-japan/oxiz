//! Theory Checking Framework
//!
//! Provides verification infrastructure for theory solver correctness:
//! - Conflict certification per theory
//! - Proof step validation
//! - Explanation checking
//! - Integration with DRAT/Alethe proof generation
//!
//! # Design
//!
//! Each theory provides a `TheoryChecker` implementation that can verify:
//! - Conflict clauses are valid (negation is T-unsatisfiable)
//! - Propagation explanations are correct
//! - Model assignments satisfy theory constraints
//!
//! # Example
//!
//! ```ignore
//! use oxiz_theories::checking::{TheoryChecker, CheckResult};
//!
//! let checker = ArithChecker::new();
//! let result = checker.check_conflict(&literals, &explanation);
//! assert!(result.is_valid());
//! ```

mod arith;
mod array;
mod bv;
mod proof;
mod quant;

pub use arith::{ArithCheckConfig, ArithChecker};
pub use array::ArrayChecker;
pub use bv::BvChecker;
pub use proof::{ProofChecker, ProofStep, ProofStepKind};
pub use quant::QuantChecker;

use oxiz_core::ast::TermId;
use std::collections::HashSet;

/// Result of checking a theory inference
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckResult {
    /// The inference is valid
    Valid,
    /// The inference is invalid with reason
    Invalid(String),
    /// Check could not be performed (missing information)
    Unknown(String),
}

impl CheckResult {
    /// Check if the result is valid
    pub fn is_valid(&self) -> bool {
        matches!(self, CheckResult::Valid)
    }

    /// Check if the result is invalid
    pub fn is_invalid(&self) -> bool {
        matches!(self, CheckResult::Invalid(_))
    }

    /// Get error message if invalid
    pub fn error_message(&self) -> Option<&str> {
        match self {
            CheckResult::Invalid(msg) => Some(msg),
            CheckResult::Unknown(msg) => Some(msg),
            CheckResult::Valid => None,
        }
    }
}

/// A literal (signed term)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal {
    /// The term
    pub term: TermId,
    /// Is the literal positive (term = true) or negative (term = false)
    pub positive: bool,
}

impl Literal {
    /// Create a positive literal
    pub fn pos(term: TermId) -> Self {
        Self {
            term,
            positive: true,
        }
    }

    /// Create a negative literal
    pub fn neg(term: TermId) -> Self {
        Self {
            term,
            positive: false,
        }
    }

    /// Negate this literal
    pub fn negate(self) -> Self {
        Self {
            term: self.term,
            positive: !self.positive,
        }
    }
}

/// Trait for theory-specific checkers
pub trait TheoryChecker: Send + Sync {
    /// Name of the theory
    fn name(&self) -> &'static str;

    /// Check a conflict clause
    ///
    /// A conflict clause is valid if the conjunction of negated literals is
    /// T-unsatisfiable.
    fn check_conflict(&self, clause: &[Literal]) -> CheckResult;

    /// Check a propagation explanation
    ///
    /// Given that `explanation => literal`, verify this is correct.
    fn check_propagation(&self, literal: Literal, explanation: &[Literal]) -> CheckResult;

    /// Check that a model satisfies theory constraints
    fn check_model(&self, assignments: &[(TermId, bool)]) -> CheckResult;

    /// Check a lemma (clause that should be T-valid)
    fn check_lemma(&self, clause: &[Literal]) -> CheckResult {
        // Default: check as conflict (all negated should be T-unsat)
        self.check_conflict(clause)
    }

    /// Get statistics
    fn stats(&self) -> CheckerStats;

    /// Reset statistics
    fn reset_stats(&mut self);
}

/// Statistics for theory checking
#[derive(Debug, Clone, Default)]
pub struct CheckerStats {
    /// Number of conflict checks
    pub conflict_checks: u64,
    /// Number of valid conflicts
    pub valid_conflicts: u64,
    /// Number of invalid conflicts
    pub invalid_conflicts: u64,
    /// Number of propagation checks
    pub propagation_checks: u64,
    /// Number of valid propagations
    pub valid_propagations: u64,
    /// Number of model checks
    pub model_checks: u64,
    /// Total checking time in microseconds
    pub check_time_us: u64,
}

impl CheckerStats {
    /// Merge statistics from another checker
    pub fn merge(&mut self, other: &CheckerStats) {
        self.conflict_checks += other.conflict_checks;
        self.valid_conflicts += other.valid_conflicts;
        self.invalid_conflicts += other.invalid_conflicts;
        self.propagation_checks += other.propagation_checks;
        self.valid_propagations += other.valid_propagations;
        self.model_checks += other.model_checks;
        self.check_time_us += other.check_time_us;
    }

    /// Validation success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.conflict_checks + self.propagation_checks;
        if total == 0 {
            1.0
        } else {
            let valid = self.valid_conflicts + self.valid_propagations;
            valid as f64 / total as f64
        }
    }
}

/// Combined theory checker that dispatches to appropriate theory
#[derive(Debug)]
pub struct CombinedChecker {
    /// Arithmetic checker
    pub arith: ArithChecker,
    /// Array checker
    pub array: ArrayChecker,
    /// Bitvector checker
    pub bv: BvChecker,
    /// Quantifier checker
    pub quant: QuantChecker,
    /// Terms that belong to each theory
    theory_terms: std::collections::HashMap<TermId, TheoryKind>,
}

/// Kind of theory a term belongs to
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TheoryKind {
    /// Boolean/propositional
    Bool,
    /// Arithmetic (LIA/LRA)
    Arith,
    /// Arrays
    Array,
    /// Bitvectors
    Bv,
    /// Quantifiers
    Quant,
    /// Uninterpreted functions
    Uf,
}

impl CombinedChecker {
    /// Create a new combined checker
    pub fn new() -> Self {
        Self {
            arith: ArithChecker::new(),
            array: ArrayChecker::new(),
            bv: BvChecker::new(),
            quant: QuantChecker::new(),
            theory_terms: std::collections::HashMap::new(),
        }
    }

    /// Register a term with its theory
    pub fn register_term(&mut self, term: TermId, kind: TheoryKind) {
        self.theory_terms.insert(term, kind);
    }

    /// Get the theory kind for a term
    pub fn get_theory(&self, term: TermId) -> Option<TheoryKind> {
        self.theory_terms.get(&term).copied()
    }

    /// Check a conflict, dispatching to appropriate theory
    pub fn check_conflict(&self, clause: &[Literal]) -> CheckResult {
        // Determine theory from literals
        let theories: HashSet<_> = clause
            .iter()
            .filter_map(|lit| self.theory_terms.get(&lit.term))
            .collect();

        if theories.len() > 1 {
            // Multi-theory conflict - need combined checking
            return CheckResult::Unknown("Multi-theory conflict".to_string());
        }

        match theories.iter().next() {
            Some(TheoryKind::Arith) => self.arith.check_conflict(clause),
            Some(TheoryKind::Array) => self.array.check_conflict(clause),
            Some(TheoryKind::Bv) => self.bv.check_conflict(clause),
            Some(TheoryKind::Quant) => self.quant.check_conflict(clause),
            _ => CheckResult::Valid, // Bool/UF - assume valid
        }
    }

    /// Get combined statistics
    pub fn stats(&self) -> CheckerStats {
        let mut stats = CheckerStats::default();
        stats.merge(&self.arith.stats());
        stats.merge(&self.array.stats());
        stats.merge(&self.bv.stats());
        stats.merge(&self.quant.stats());
        stats
    }

    /// Reset all statistics
    pub fn reset_stats(&mut self) {
        self.arith.reset_stats();
        self.array.reset_stats();
        self.bv.reset_stats();
        self.quant.reset_stats();
    }
}

impl Default for CombinedChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_result() {
        let valid = CheckResult::Valid;
        assert!(valid.is_valid());
        assert!(!valid.is_invalid());
        assert_eq!(valid.error_message(), None);

        let invalid = CheckResult::Invalid("test error".to_string());
        assert!(!invalid.is_valid());
        assert!(invalid.is_invalid());
        assert_eq!(invalid.error_message(), Some("test error"));
    }

    #[test]
    fn test_literal() {
        let t = TermId::from(1u32);
        let pos = Literal::pos(t);
        let neg = Literal::neg(t);

        assert!(pos.positive);
        assert!(!neg.positive);
        assert_eq!(pos.negate(), neg);
        assert_eq!(neg.negate(), pos);
    }

    #[test]
    fn test_checker_stats() {
        let mut stats1 = CheckerStats::default();
        stats1.conflict_checks = 10;
        stats1.valid_conflicts = 8;

        let mut stats2 = CheckerStats::default();
        stats2.conflict_checks = 5;
        stats2.valid_conflicts = 5;

        stats1.merge(&stats2);
        assert_eq!(stats1.conflict_checks, 15);
        assert_eq!(stats1.valid_conflicts, 13);
    }

    #[test]
    fn test_success_rate() {
        let mut stats = CheckerStats::default();
        assert_eq!(stats.success_rate(), 1.0);

        stats.conflict_checks = 10;
        stats.valid_conflicts = 8;
        assert!((stats.success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_combined_checker() {
        let mut checker = CombinedChecker::new();
        let t = TermId::from(1u32);

        checker.register_term(t, TheoryKind::Arith);
        assert_eq!(checker.get_theory(t), Some(TheoryKind::Arith));
    }
}
