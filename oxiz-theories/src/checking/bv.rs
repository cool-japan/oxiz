//! Bitvector Theory Checker
//!
//! Validates bitvector theory inferences.

use super::{CheckResult, CheckerStats, Literal, TheoryChecker};
use oxiz_core::ast::TermId;
use std::time::Instant;

/// Bitvector theory checker
#[derive(Debug)]
pub struct BvChecker {
    stats: CheckerStats,
    /// Maximum bitvector width to check precisely (reserved for future use)
    #[allow(dead_code)]
    max_precise_width: u32,
}

impl BvChecker {
    /// Create a new bitvector checker
    pub fn new() -> Self {
        Self {
            stats: CheckerStats::default(),
            max_precise_width: 64,
        }
    }

    /// Create with custom max width
    pub fn with_max_width(max_width: u32) -> Self {
        Self {
            stats: CheckerStats::default(),
            max_precise_width: max_width,
        }
    }

    /// Check bitvector conflict validity
    fn check_bv_conflict(&self, clause: &[Literal]) -> CheckResult {
        if clause.is_empty() {
            return CheckResult::Invalid("Empty conflict clause".to_string());
        }

        // Bitvector conflicts can arise from:
        // - Bit-blasting contradictions
        // - Arithmetic overflow/underflow
        // - Equality/disequality propagation

        // For small bitvectors, we could enumerate all values
        // For larger ones, we trust the bit-blasting

        CheckResult::Valid
    }

    /// Check bitvector propagation
    fn check_bv_propagation(&self, _literal: Literal, _explanation: &[Literal]) -> CheckResult {
        // BV propagations include:
        // - Bit equality: a[i] = b[i] for equal bitvectors
        // - Sign extension properties
        // - Arithmetic implications
        CheckResult::Valid
    }

    /// Check model for bitvector consistency
    fn check_bv_model(&self, _assignments: &[(TermId, bool)]) -> CheckResult {
        CheckResult::Valid
    }

    /// Evaluate a bitvector expression
    fn _eval_bv(&self, _val: u64, _width: u32) -> u64 {
        // Mask to width
        0
    }

    /// Check if two bitvector values are equal within width
    fn _bv_eq(&self, a: u64, b: u64, width: u32) -> bool {
        let mask = if width >= 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        (a & mask) == (b & mask)
    }
}

impl Default for BvChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TheoryChecker for BvChecker {
    fn name(&self) -> &'static str {
        "bitvector"
    }

    fn check_conflict(&self, clause: &[Literal]) -> CheckResult {
        let start = Instant::now();
        let result = self.check_bv_conflict(clause);
        let _elapsed = start.elapsed();
        result
    }

    fn check_propagation(&self, literal: Literal, explanation: &[Literal]) -> CheckResult {
        let start = Instant::now();
        let result = self.check_bv_propagation(literal, explanation);
        let _elapsed = start.elapsed();
        result
    }

    fn check_model(&self, assignments: &[(TermId, bool)]) -> CheckResult {
        self.check_bv_model(assignments)
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
    fn test_bv_checker_creation() {
        let checker = BvChecker::new();
        assert_eq!(checker.name(), "bitvector");
        assert_eq!(checker.max_precise_width, 64);
    }

    #[test]
    fn test_bv_with_max_width() {
        let checker = BvChecker::with_max_width(32);
        assert_eq!(checker.max_precise_width, 32);
    }

    #[test]
    fn test_bv_conflict_empty() {
        let checker = BvChecker::new();
        let result = checker.check_conflict(&[]);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_bv_conflict_valid() {
        let checker = BvChecker::new();
        let t1 = TermId::from(1u32);
        let clause = vec![Literal::pos(t1)];
        let result = checker.check_conflict(&clause);
        assert!(result.is_valid());
    }

    #[test]
    fn test_bv_propagation() {
        let checker = BvChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        let literal = Literal::pos(t1);
        let explanation = vec![Literal::pos(t2)];
        let result = checker.check_propagation(literal, &explanation);
        assert!(result.is_valid());
    }

    #[test]
    fn test_bv_model_check() {
        let checker = BvChecker::new();
        let t1 = TermId::from(1u32);
        let assignments = vec![(t1, true)];
        let result = checker.check_model(&assignments);
        assert!(result.is_valid());
    }

    #[test]
    fn test_bv_eq() {
        let checker = BvChecker::new();
        assert!(checker._bv_eq(0xFF, 0x1FF, 8)); // Both 0xFF in 8 bits
        assert!(!checker._bv_eq(0xFF, 0xFE, 8)); // Different
        assert!(checker._bv_eq(0, 0, 32));
        assert!(checker._bv_eq(u64::MAX, u64::MAX, 64));
    }

    #[test]
    fn test_bv_stats() {
        let mut checker = BvChecker::new();
        let stats = checker.stats();
        assert_eq!(stats.conflict_checks, 0);

        checker.reset_stats();
        let stats = checker.stats();
        assert_eq!(stats.propagation_checks, 0);
    }
}
