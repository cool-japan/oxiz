//! Array Theory Checker
//!
//! Validates array theory inferences (select/store axioms).

use super::{CheckResult, CheckerStats, Literal, TheoryChecker};
#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::TermId;
use oxiz_time::Instant;

/// Array theory checker
#[derive(Debug)]
pub struct ArrayChecker {
    stats: CheckerStats,
    /// Whether to check extensionality axiom
    check_extensionality: bool,
}

impl ArrayChecker {
    /// Create a new array checker
    pub fn new() -> Self {
        Self {
            stats: CheckerStats::default(),
            check_extensionality: true,
        }
    }

    /// Create with extensionality checking disabled
    pub fn without_extensionality() -> Self {
        Self {
            stats: CheckerStats::default(),
            check_extensionality: false,
        }
    }

    /// Check array conflict validity
    /// Array conflicts typically involve:
    /// - Read-over-write: select(store(a, i, v), i) = v
    /// - Read-over-write-miss: i != j => select(store(a, i, v), j) = select(a, j)
    /// - Extensionality: (forall i. select(a, i) = select(b, i)) => a = b
    fn check_array_conflict(&self, clause: &[Literal]) -> CheckResult {
        if clause.is_empty() {
            return CheckResult::Invalid("Empty conflict clause".to_string());
        }
        // A propositional tautology is a sound conflict in any theory.
        if super::clause_has_complementary_pair(clause) {
            return CheckResult::Valid;
        }
        // A genuine array conflict must be certified against the select/store
        // (read-over-write / extensionality) axiom instances, which requires
        // the term structure this checker does not have. Report Unknown rather
        // than rubber-stamp it as Valid.
        CheckResult::Unknown("Array conflict requires select/store axiom certification".to_string())
    }

    /// Check array propagation.
    fn check_array_propagation(&self, literal: Literal, explanation: &[Literal]) -> CheckResult {
        if super::explanation_entails(literal, explanation) {
            CheckResult::Valid
        } else {
            CheckResult::Unknown(
                "Array propagation not certifiable from literal identities".to_string(),
            )
        }
    }

    /// Check model for array consistency.
    fn check_array_model(&self, assignments: &[(TermId, bool)]) -> CheckResult {
        if assignments.is_empty() {
            CheckResult::Valid
        } else {
            CheckResult::Unknown("Array model consistency requires term evaluation".to_string())
        }
    }

    /// Enable/disable extensionality checking
    pub fn set_extensionality(&mut self, enabled: bool) {
        self.check_extensionality = enabled;
    }
}

impl Default for ArrayChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TheoryChecker for ArrayChecker {
    fn name(&self) -> &'static str {
        "array"
    }

    fn check_conflict(&self, clause: &[Literal]) -> CheckResult {
        let start = Instant::now();
        let result = self.check_array_conflict(clause);
        let _elapsed = start.elapsed();
        result
    }

    fn check_propagation(&self, literal: Literal, explanation: &[Literal]) -> CheckResult {
        let start = Instant::now();
        let result = self.check_array_propagation(literal, explanation);
        let _elapsed = start.elapsed();
        result
    }

    fn check_model(&self, assignments: &[(TermId, bool)]) -> CheckResult {
        self.check_array_model(assignments)
    }

    fn stats(&self) -> CheckerStats {
        self.stats.clone()
    }

    fn reset_stats(&mut self) {
        self.stats = CheckerStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_checker_creation() {
        let checker = ArrayChecker::new();
        assert_eq!(checker.name(), "array");
        assert!(checker.check_extensionality);
    }

    #[test]
    fn test_array_without_extensionality() {
        let checker = ArrayChecker::without_extensionality();
        assert!(!checker.check_extensionality);
    }

    #[test]
    fn test_array_conflict_empty() {
        let checker = ArrayChecker::new();
        let result = checker.check_conflict(&[]);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_array_conflict_tautology_valid_but_plain_unknown() {
        let checker = ArrayChecker::new();
        let t1 = TermId::from(1u32);
        // Tautology conflict is soundly valid.
        assert!(
            checker
                .check_conflict(&[Literal::pos(t1), Literal::neg(t1)])
                .is_valid()
        );
        // A single opaque literal cannot be certified -> Unknown, not Valid.
        let result = checker.check_conflict(&[Literal::pos(t1)]);
        assert!(!result.is_valid() && !result.is_invalid());
    }

    #[test]
    fn test_array_propagation() {
        let checker = ArrayChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        assert!(
            checker
                .check_propagation(Literal::pos(t1), &[Literal::pos(t1)])
                .is_valid()
        );
        let result = checker.check_propagation(Literal::pos(t1), &[Literal::pos(t2)]);
        assert!(!result.is_valid() && !result.is_invalid());
    }

    #[test]
    fn test_array_model_check() {
        let checker = ArrayChecker::new();
        let t1 = TermId::from(1u32);
        assert!(checker.check_model(&[]).is_valid());
        let result = checker.check_model(&[(t1, true)]);
        assert!(!result.is_valid() && !result.is_invalid());
    }

    #[test]
    fn test_set_extensionality() {
        let mut checker = ArrayChecker::new();
        assert!(checker.check_extensionality);

        checker.set_extensionality(false);
        assert!(!checker.check_extensionality);

        checker.set_extensionality(true);
        assert!(checker.check_extensionality);
    }

    #[test]
    fn test_array_stats() {
        let mut checker = ArrayChecker::new();
        let stats = checker.stats();
        assert_eq!(stats.conflict_checks, 0);

        checker.reset_stats();
        let stats = checker.stats();
        assert_eq!(stats.propagation_checks, 0);
    }
}
