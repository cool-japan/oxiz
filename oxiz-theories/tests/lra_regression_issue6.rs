//! Regression tests for GitHub issue #6: LRA theory correctness bug.
//!
//! The reporter observed that the QF_LRA problem:
//!   (declare-fun x1 () Real)
//!   (assert (<= x1 (- 1.0)))
//!   (assert (= x1 (- 0.25)))
//!
//! was incorrectly reported as SAT with model x1=0.0.
//!
//! The constraints are:
//!   x1 <= -1.0   (x1 must be at most -1)
//!   x1 =  -0.25  (x1 must equal -0.25)
//!
//! Since -0.25 > -1.0, both constraints cannot be satisfied simultaneously:
//! the equality forces x1 = -0.25, but the inequality requires x1 <= -1.0,
//! which -0.25 violates. Therefore the problem is UNSAT.

use num_rational::Rational64;
use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::TheoryCheckResult;
use oxiz_theories::arithmetic::ArithSolver;

/// Regression test for GitHub issue #6.
///
/// Assert: (<= x1 -1.0) AND (= x1 -0.25) must be UNSAT.
/// The equality pins x1 at -0.25, which violates x1 <= -1.
#[test]
fn test_issue_6_lra_unsat() {
    let mut solver = ArithSolver::lra();

    let x1 = TermId(1);
    let reason = TermId(0);

    // Constraint 1: x1 <= -1.0
    // SMT-LIB2: (<= x1 (- 1.0))
    solver.assert_le(
        &[(x1, Rational64::from_integer(1))],
        Rational64::new(-1, 1), // -1.0
        reason,
    );

    // Constraint 2: x1 = -0.25
    // SMT-LIB2: (= x1 (- 0.25)) where -0.25 = -1/4
    solver.assert_eq(
        &[(x1, Rational64::from_integer(1))],
        Rational64::new(-1, 4), // -0.25
        reason,
    );

    // Check satisfiability
    let result = solver.check();
    assert!(result.is_ok(), "check() should not error");

    match result.expect("check() succeeded") {
        TheoryCheckResult::Unsat(_) => {
            // Correct: x1 = -0.25 violates x1 <= -1.0
        }
        TheoryCheckResult::Sat => {
            panic!(
                "Issue #6 regression: solver returned SAT for UNSAT instance.\n\
                 Constraints: x1 <= -1.0 AND x1 = -0.25.\n\
                 -0.25 > -1.0 so they cannot both hold."
            );
        }
        TheoryCheckResult::Unknown | TheoryCheckResult::Propagate(_) => {
            panic!("Solver returned non-final result for a decidable LRA instance");
        }
    }
}

/// Variant: same bug but with order of assertion reversed.
///
/// Assert: (= x1 -0.25) first, then (<= x1 -1.0).
/// Must still be UNSAT regardless of assertion order.
#[test]
fn test_issue_6_lra_unsat_reversed_order() {
    let mut solver = ArithSolver::lra();

    let x1 = TermId(1);
    let reason = TermId(0);

    // Constraint 1 (asserted first): x1 = -0.25
    solver.assert_eq(
        &[(x1, Rational64::from_integer(1))],
        Rational64::new(-1, 4),
        reason,
    );

    // Constraint 2 (asserted second): x1 <= -1.0
    solver.assert_le(
        &[(x1, Rational64::from_integer(1))],
        Rational64::new(-1, 1),
        reason,
    );

    let result = solver.check();
    assert!(result.is_ok(), "check() should not error");

    match result.expect("check() succeeded") {
        TheoryCheckResult::Unsat(_) => {
            // Correct: -0.25 violates x1 <= -1.0
        }
        TheoryCheckResult::Sat => {
            panic!(
                "Issue #6 regression (reversed order): solver returned SAT for UNSAT instance.\n\
                 Constraints: x1 = -0.25 AND x1 <= -1.0."
            );
        }
        TheoryCheckResult::Unknown | TheoryCheckResult::Propagate(_) => {
            panic!("Solver returned non-final result for a decidable LRA instance");
        }
    }
}

/// Sanity check: a satisfiable instance with similar structure must be SAT.
///
/// x1 <= -1.0 AND x1 = -2.0 is SAT because -2.0 <= -1.0.
#[test]
fn test_issue_6_lra_sat_sanity() {
    let mut solver = ArithSolver::lra();

    let x1 = TermId(1);
    let reason = TermId(0);

    // x1 <= -1.0
    solver.assert_le(
        &[(x1, Rational64::from_integer(1))],
        Rational64::new(-1, 1),
        reason,
    );

    // x1 = -2.0  (satisfies x1 <= -1.0 because -2 < -1)
    solver.assert_eq(
        &[(x1, Rational64::from_integer(1))],
        Rational64::new(-2, 1),
        reason,
    );

    let result = solver.check();
    assert!(result.is_ok(), "check() should not error");

    match result.expect("check() succeeded") {
        TheoryCheckResult::Sat => {
            // Correct: x1 = -2.0 satisfies x1 <= -1.0
        }
        TheoryCheckResult::Unsat(_) => {
            panic!(
                "Sanity failure: solver returned UNSAT for SAT instance.\n\
                 Constraints: x1 <= -1.0 AND x1 = -2.0.\n\
                 -2.0 <= -1.0 so this should be satisfiable."
            );
        }
        TheoryCheckResult::Unknown | TheoryCheckResult::Propagate(_) => {
            panic!("Solver returned non-final result for a decidable LRA instance");
        }
    }
}
