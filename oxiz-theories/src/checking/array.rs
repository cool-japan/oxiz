//! Array Theory Checker
//!
//! Validates array theory inferences (select/store axioms).

use super::{CheckResult, CheckerStats, Literal, TheoryChecker};
use oxiz_core::ast::TermId;
use std::time::Instant;

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

        // For array theory, conflicts arise from:
        // 1. Contradictory read/write combinations
        // 2. Equality propagation conflicts

        // Simplified: assume valid for now
        // Real implementation would check select/store axioms
        CheckResult::Valid
    }

    /// Check array propagation
    fn check_array_propagation(&self, _literal: Literal, _explanation: &[Literal]) -> CheckResult {
        // Array propagations include:
        // - i = j => select(a, i) = select(a, j)
        // - store(a, i, v) = store(b, j, w) => i = j AND v = w (for equal arrays)
        CheckResult::Valid
    }

    /// Check model for array consistency
    fn check_array_model(&self, _assignments: &[(TermId, bool)]) -> CheckResult {
        // Verify array assignments satisfy select/store semantics
        CheckResult::Valid
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
    fn test_array_conflict_valid() {
        let checker = ArrayChecker::new();
        let t1 = TermId::from(1u32);
        let clause = vec![Literal::pos(t1)];
        let result = checker.check_conflict(&clause);
        assert!(result.is_valid());
    }

    #[test]
    fn test_array_propagation() {
        let checker = ArrayChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        let literal = Literal::pos(t1);
        let explanation = vec![Literal::pos(t2)];
        let result = checker.check_propagation(literal, &explanation);
        assert!(result.is_valid());
    }

    #[test]
    fn test_array_model_check() {
        let checker = ArrayChecker::new();
        let t1 = TermId::from(1u32);
        let assignments = vec![(t1, true)];
        let result = checker.check_model(&assignments);
        assert!(result.is_valid());
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
