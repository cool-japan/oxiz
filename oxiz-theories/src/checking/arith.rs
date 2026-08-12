//! Arithmetic Theory Checker
//!
//! Validates arithmetic theory inferences (LIA, LRA, NIA, NRA).

use super::{CheckResult, CheckerStats, Literal, TheoryChecker};
#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::TermId;
use oxiz_time::Instant;

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

    /// Check that a conflict clause is T-valid, i.e. the conjunction of its
    /// negated literals is arithmetically unsatisfiable.
    ///
    /// This checker operates on abstract literal identities only (it has no
    /// access to the term manager), so it can soundly certify the two
    /// theory-independent cases — an empty clause is *not* a valid conflict,
    /// and a propositional tautology *is* — but a genuine linear-arithmetic
    /// infeasibility (e.g. `x+y>=5 ∧ x+y<=4`) requires Simplex/Fourier–Motzkin
    /// over the actual atoms and is reported as `Unknown` rather than being
    /// rubber-stamped as `Valid`.
    fn check_linear_conflict(&self, clause: &[Literal]) -> CheckResult {
        if clause.is_empty() {
            return CheckResult::Invalid(
                "Empty conflict clause is trivially satisfiable".to_string(),
            );
        }
        if super::clause_has_complementary_pair(clause) {
            return CheckResult::Valid;
        }
        CheckResult::Unknown(
            "Linear-arithmetic infeasibility not certifiable from literal identities \
             (requires Simplex/Farkas over the atoms)"
                .to_string(),
        )
    }

    /// Check propagation: `explanation => literal`.
    ///
    /// Sound structural certification only: if the explanation already contains
    /// the propagated literal (or is itself contradictory) the entailment
    /// holds; otherwise a real check would negate the literal and refute the
    /// system, which is not available here, so the result is `Unknown`.
    fn check_linear_propagation(&self, literal: Literal, explanation: &[Literal]) -> CheckResult {
        if super::explanation_entails(literal, explanation) {
            CheckResult::Valid
        } else {
            CheckResult::Unknown(
                "Arithmetic propagation not certifiable from literal identities".to_string(),
            )
        }
    }

    /// Check model consistency.
    ///
    /// With no assignments there is nothing to violate (vacuously consistent);
    /// otherwise verifying that the assignment satisfies the arithmetic atoms
    /// requires evaluating the terms, which this checker cannot do, so the
    /// result is `Unknown`.
    fn check_arith_model(&self, assignments: &[(TermId, bool)]) -> CheckResult {
        if assignments.is_empty() {
            CheckResult::Valid
        } else {
            CheckResult::Unknown(
                "Arithmetic model consistency requires term evaluation".to_string(),
            )
        }
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
    fn test_arith_conflict_tautology_is_valid() {
        // A clause containing p and ¬p is a sound theory-independent conflict.
        let checker = ArithChecker::new();
        let t1 = TermId::from(1u32);
        let clause = vec![Literal::pos(t1), Literal::neg(t1)];
        assert!(checker.check_conflict(&clause).is_valid());
    }

    #[test]
    fn test_arith_conflict_empty_is_invalid() {
        let checker = ArithChecker::new();
        assert!(checker.check_conflict(&[]).is_invalid());
    }

    #[test]
    fn test_arith_conflict_non_tautology_is_unknown_not_valid() {
        // Regression: an uncertifiable conflict must NOT be rubber-stamped Valid.
        let checker = ArithChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);
        let clause = vec![Literal::pos(t1), Literal::neg(t2)];
        let result = checker.check_conflict(&clause);
        assert!(!result.is_valid());
        assert!(!result.is_invalid());
    }

    #[test]
    fn test_arith_propagation_check() {
        let checker = ArithChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        // Explanation contains the propagated literal -> entailed -> Valid.
        assert!(
            checker
                .check_propagation(Literal::pos(t1), &[Literal::pos(t1), Literal::pos(t2)])
                .is_valid()
        );
        // Explanation does not entail the literal -> Unknown, never Valid.
        let result = checker.check_propagation(Literal::pos(t1), &[Literal::pos(t2)]);
        assert!(!result.is_valid() && !result.is_invalid());
    }

    #[test]
    fn test_arith_model_check() {
        let checker = ArithChecker::new();
        let t1 = TermId::from(1u32);
        // Empty model is vacuously consistent.
        assert!(checker.check_model(&[]).is_valid());
        // A non-empty model cannot be certified without term evaluation.
        let result = checker.check_model(&[(t1, true)]);
        assert!(!result.is_valid() && !result.is_invalid());
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
