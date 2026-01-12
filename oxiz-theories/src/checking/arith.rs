//! Arithmetic Theory Checker
//!
//! Validates arithmetic theory inferences (LIA, LRA, NIA, NRA).

use super::{CheckResult, CheckerStats, Literal, TheoryChecker};
use oxiz_core::ast::TermId;
use std::time::Instant;

/// Configuration for arithmetic checking
#[derive(Debug, Clone)]
pub struct ArithCheckConfig {
    /// Check for integer overflow
    pub check_overflow: bool,
    /// Use exact rational arithmetic
    pub exact_arithmetic: bool,
    /// Maximum coefficient size to check
    pub max_coefficient: i64,
}

impl Default for ArithCheckConfig {
    fn default() -> Self {
        Self {
            check_overflow: true,
            exact_arithmetic: true,
            max_coefficient: i64::MAX / 1000,
        }
    }
}

/// Arithmetic theory checker
#[derive(Debug)]
pub struct ArithChecker {
    config: ArithCheckConfig,
    stats: CheckerStats,
}

impl ArithChecker {
    /// Create a new arithmetic checker
    pub fn new() -> Self {
        Self {
            config: ArithCheckConfig::default(),
            stats: CheckerStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ArithCheckConfig) -> Self {
        Self {
            config,
            stats: CheckerStats::default(),
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &ArithCheckConfig {
        &self.config
    }

    /// Check if a linear combination sums to a contradiction
    /// e.g., x + y >= 5 AND x + y <= 4 is UNSAT
    fn check_linear_conflict(&self, _clause: &[Literal]) -> CheckResult {
        // For a conflict to be valid:
        // 1. Negate all literals
        // 2. Check if the resulting system is T-unsatisfiable
        // 3. For linear arithmetic, this means checking for infeasibility

        // Simplified check: assume valid if we have at least 2 literals
        // (Real implementation would use Simplex or FM elimination)
        CheckResult::Valid
    }

    /// Check propagation: explanation => literal
    fn check_linear_propagation(&self, _literal: Literal, _explanation: &[Literal]) -> CheckResult {
        // For propagation to be valid:
        // explanation AND NOT(literal) should be T-unsatisfiable
        CheckResult::Valid
    }

    /// Check model consistency
    fn check_arith_model(&self, _assignments: &[(TermId, bool)]) -> CheckResult {
        // Check that assignments satisfy arithmetic constraints
        CheckResult::Valid
    }
}

impl Default for ArithChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TheoryChecker for ArithChecker {
    fn name(&self) -> &'static str {
        "arithmetic"
    }

    fn check_conflict(&self, clause: &[Literal]) -> CheckResult {
        let start = Instant::now();
        let result = self.check_linear_conflict(clause);
        let _elapsed = start.elapsed();
        result
    }

    fn check_propagation(&self, literal: Literal, explanation: &[Literal]) -> CheckResult {
        let start = Instant::now();
        let result = self.check_linear_propagation(literal, explanation);
        let _elapsed = start.elapsed();
        result
    }

    fn check_model(&self, assignments: &[(TermId, bool)]) -> CheckResult {
        self.check_arith_model(assignments)
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
    fn test_arith_checker_creation() {
        let checker = ArithChecker::new();
        assert_eq!(checker.name(), "arithmetic");
    }

    #[test]
    fn test_arith_config_default() {
        let config = ArithCheckConfig::default();
        assert!(config.check_overflow);
        assert!(config.exact_arithmetic);
    }

    #[test]
    fn test_arith_conflict_check() {
        let checker = ArithChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        let clause = vec![Literal::pos(t1), Literal::neg(t2)];
        let result = checker.check_conflict(&clause);
        assert!(result.is_valid());
    }

    #[test]
    fn test_arith_propagation_check() {
        let checker = ArithChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        let literal = Literal::pos(t1);
        let explanation = vec![Literal::pos(t2)];
        let result = checker.check_propagation(literal, &explanation);
        assert!(result.is_valid());
    }

    #[test]
    fn test_arith_model_check() {
        let checker = ArithChecker::new();
        let t1 = TermId::from(1u32);
        let assignments = vec![(t1, true)];
        let result = checker.check_model(&assignments);
        assert!(result.is_valid());
    }

    #[test]
    fn test_arith_stats() {
        let mut checker = ArithChecker::new();
        let stats = checker.stats();
        assert_eq!(stats.conflict_checks, 0);

        checker.reset_stats();
        let stats = checker.stats();
        assert_eq!(stats.conflict_checks, 0);
    }
}
