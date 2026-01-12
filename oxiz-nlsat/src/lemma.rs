//! Theory lemma interface for NLSAT.
//!
//! This module provides structures for theory lemmas that explain
//! theory conflicts and propagations.
//!
//! Reference: Z3's `nlsat/nlsat_explain.h`

use crate::types::{BoolVar, Literal};
use num_rational::BigRational;
use oxiz_math::polynomial::Var;

/// A theory lemma explaining a conflict or propagation.
#[derive(Debug, Clone)]
pub struct TheoryLemma {
    /// The literals in the lemma (clause).
    pub literals: Vec<Literal>,
    /// The kind of lemma.
    pub kind: LemmaKind,
    /// Explanation data.
    pub explanation: Explanation,
}

impl TheoryLemma {
    /// Create a new conflict lemma.
    pub fn conflict(literals: Vec<Literal>, explanation: Explanation) -> Self {
        Self {
            literals,
            kind: LemmaKind::Conflict,
            explanation,
        }
    }

    /// Create a new propagation lemma.
    pub fn propagation(literals: Vec<Literal>, explanation: Explanation) -> Self {
        Self {
            literals,
            kind: LemmaKind::Propagation,
            explanation,
        }
    }

    /// Get the asserted literal (for unit lemmas).
    pub fn asserted_literal(&self) -> Option<Literal> {
        if self.literals.len() == 1 {
            Some(self.literals[0])
        } else {
            self.literals.first().copied()
        }
    }

    /// Check if this is a conflict lemma.
    pub fn is_conflict(&self) -> bool {
        matches!(self.kind, LemmaKind::Conflict)
    }

    /// Check if this is a propagation lemma.
    pub fn is_propagation(&self) -> bool {
        matches!(self.kind, LemmaKind::Propagation)
    }

    /// Get the size of the lemma.
    pub fn size(&self) -> usize {
        self.literals.len()
    }
}

/// Kind of theory lemma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LemmaKind {
    /// Conflict: all literals are false under current assignment.
    Conflict,
    /// Propagation: all but one literal are false, implying the last is true.
    Propagation,
}

/// Explanation for a theory lemma.
#[derive(Debug, Clone)]
pub enum Explanation {
    /// Polynomial constraint conflict.
    PolynomialConflict {
        /// Variables involved in the conflict.
        vars: Vec<Var>,
        /// Values assigned to variables.
        values: Vec<BigRational>,
        /// Constraint that was violated.
        description: String,
    },
    /// Root constraint conflict.
    RootConflict {
        /// Variable being constrained.
        var: Var,
        /// Root index.
        root_index: u32,
        /// Description.
        description: String,
    },
    /// Sign-based conflict from CAD.
    SignConflict {
        /// Variables and their assigned values.
        assignment: Vec<(Var, BigRational)>,
        /// Sign conditions that conflict.
        conditions: Vec<String>,
    },
    /// Generic explanation.
    Generic {
        /// Description of the conflict/propagation.
        description: String,
    },
}

impl Explanation {
    /// Create a generic explanation.
    pub fn generic(description: impl Into<String>) -> Self {
        Self::Generic {
            description: description.into(),
        }
    }

    /// Create a polynomial conflict explanation.
    pub fn polynomial_conflict(
        vars: Vec<Var>,
        values: Vec<BigRational>,
        description: impl Into<String>,
    ) -> Self {
        Self::PolynomialConflict {
            vars,
            values,
            description: description.into(),
        }
    }

    /// Create a sign conflict explanation.
    pub fn sign_conflict(assignment: Vec<(Var, BigRational)>, conditions: Vec<String>) -> Self {
        Self::SignConflict {
            assignment,
            conditions,
        }
    }
}

/// Interface for theory lemma generation.
pub trait LemmaGenerator {
    /// Generate a lemma explaining a theory conflict.
    fn explain_conflict(&self, conflicting_atoms: &[BoolVar]) -> TheoryLemma;

    /// Generate a lemma for theory propagation.
    fn explain_propagation(&self, propagated: Literal, reason_atoms: &[BoolVar]) -> TheoryLemma;
}

/// Statistics for theory lemma generation.
#[derive(Debug, Clone, Default)]
pub struct LemmaStats {
    /// Number of conflict lemmas generated.
    pub conflicts: u64,
    /// Number of propagation lemmas generated.
    pub propagations: u64,
    /// Total size of conflict lemmas.
    pub conflict_size: u64,
    /// Total size of propagation lemmas.
    pub propagation_size: u64,
    /// Number of minimized lemmas.
    pub minimized: u64,
}

impl LemmaStats {
    /// Create new empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a conflict lemma.
    pub fn record_conflict(&mut self, size: usize) {
        self.conflicts += 1;
        self.conflict_size += size as u64;
    }

    /// Record a propagation lemma.
    pub fn record_propagation(&mut self, size: usize) {
        self.propagations += 1;
        self.propagation_size += size as u64;
    }

    /// Record a minimization.
    pub fn record_minimization(&mut self) {
        self.minimized += 1;
    }

    /// Get average conflict lemma size.
    pub fn avg_conflict_size(&self) -> f64 {
        if self.conflicts == 0 {
            0.0
        } else {
            self.conflict_size as f64 / self.conflicts as f64
        }
    }

    /// Get average propagation lemma size.
    pub fn avg_propagation_size(&self) -> f64 {
        if self.propagations == 0 {
            0.0
        } else {
            self.propagation_size as f64 / self.propagations as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theory_lemma_conflict() {
        let lits = vec![Literal::positive(0), Literal::negative(1)];
        let explanation = Explanation::generic("Test conflict");

        let lemma = TheoryLemma::conflict(lits.clone(), explanation);

        assert!(lemma.is_conflict());
        assert!(!lemma.is_propagation());
        assert_eq!(lemma.literals, lits);
        assert_eq!(lemma.size(), 2);
    }

    #[test]
    fn test_theory_lemma_propagation() {
        let lits = vec![
            Literal::positive(0),
            Literal::negative(1),
            Literal::positive(2),
        ];
        let explanation = Explanation::generic("Test propagation");

        let lemma = TheoryLemma::propagation(lits.clone(), explanation);

        assert!(!lemma.is_conflict());
        assert!(lemma.is_propagation());
        assert_eq!(lemma.size(), 3);
    }

    #[test]
    fn test_lemma_asserted_literal() {
        let lit = Literal::positive(5);
        let lemma = TheoryLemma::conflict(vec![lit], Explanation::generic("Unit"));

        assert_eq!(lemma.asserted_literal(), Some(lit));
    }

    #[test]
    fn test_explanation_generic() {
        let exp = Explanation::generic("test");
        assert!(matches!(exp, Explanation::Generic { .. }));
    }

    #[test]
    fn test_explanation_polynomial() {
        let exp = Explanation::polynomial_conflict(
            vec![0, 1],
            vec![BigRational::from_integer(1.into())],
            "p > 0 violated",
        );
        assert!(matches!(exp, Explanation::PolynomialConflict { .. }));
    }

    #[test]
    fn test_lemma_stats_empty() {
        let stats = LemmaStats::new();
        assert_eq!(stats.conflicts, 0);
        assert_eq!(stats.propagations, 0);
        assert_eq!(stats.avg_conflict_size(), 0.0);
        assert_eq!(stats.avg_propagation_size(), 0.0);
    }

    #[test]
    fn test_lemma_stats_record() {
        let mut stats = LemmaStats::new();

        stats.record_conflict(5);
        stats.record_conflict(3);
        assert_eq!(stats.conflicts, 2);
        assert_eq!(stats.conflict_size, 8);
        assert_eq!(stats.avg_conflict_size(), 4.0);

        stats.record_propagation(10);
        assert_eq!(stats.propagations, 1);
        assert_eq!(stats.avg_propagation_size(), 10.0);
    }

    #[test]
    fn test_lemma_stats_minimization() {
        let mut stats = LemmaStats::new();
        stats.record_minimization();
        stats.record_minimization();
        assert_eq!(stats.minimized, 2);
    }
}
