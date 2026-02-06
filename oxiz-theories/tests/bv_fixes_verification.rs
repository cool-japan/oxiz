//! Verification tests for BV conflict detection fixes
//! Tests the two critical bug fixes:
//! 1. Signed division/remainder constraint encoding (bv_12)
//! 2. Comparison conflict detection (bv_15)

use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::bv::BvSolver;

#[test]
fn test_ult_conflict_detection() {
    // Bug fix verification: Contradictory unsigned less-than comparisons
    // x < y AND y < x should be UNSAT
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Assert: x < y
    solver.assert_ult(x, y);

    // Assert: y < x (contradictory)
    solver.assert_ult(y, x);

    // Should detect UNSAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {
            // Good! Conflict detected
        }
        other => panic!("Expected UNSAT but got {:?}", other),
    }
}

#[test]
fn test_ult_no_false_positive() {
    // Ensure we don't have false positives
    // x < y should be satisfiable on its own
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Assert: x < y
    solver.assert_ult(x, y);

    // Should be SAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Sat => {
            // Good!
        }
        other => panic!("Expected SAT but got {:?}", other),
    }
}

#[test]
fn test_sdiv_constraint_encoding() {
    // Test that signed division properly encodes the constraint
    // (bvsdiv x y) = quotient means: x = quotient * y + remainder
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let div_result = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Encode: div_result = x / y
    solver.bv_sdiv(div_result, x, y);

    // Fix x and y to specific values
    solver.assert_const(x, 13, 8); // 13
    solver.assert_const(y, 6, 8); // 6

    // 13 / 6 = 2
    // Assert div_result = 2
    solver.assert_const(div_result, 2, 8);

    // Should be SAT (13 / 6 = 2 is correct)
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Sat => {
            // Good!
        }
        other => panic!("Expected SAT for valid division, but got {:?}", other),
    }
}

#[test]
fn test_sdiv_inconsistent() {
    // Test that inconsistent signed division is detected
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let div_result = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Encode: div_result = x / y
    solver.bv_sdiv(div_result, x, y);

    // Fix x and y
    solver.assert_const(x, 13, 8); // 13
    solver.assert_const(y, 6, 8); // 6

    // 13 / 6 = 2, but assert div_result = 3 (wrong!)
    solver.assert_const(div_result, 3, 8);

    // Should be UNSAT (inconsistent)
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {
            // Good! Inconsistency detected
        }
        other => panic!(
            "Expected UNSAT for inconsistent division, but got {:?}",
            other
        ),
    }
}

#[test]
fn test_srem_constraint_encoding() {
    // Test that signed remainder properly encodes the constraint
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let rem_result = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Encode: rem_result = x % y
    solver.bv_srem(rem_result, x, y);

    // Fix x and y
    solver.assert_const(x, 13, 8); // 13
    solver.assert_const(y, 6, 8); // 6

    // 13 % 6 = 1
    solver.assert_const(rem_result, 1, 8);

    // Should be SAT (13 % 6 = 1 is correct)
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Sat => {
            // Good!
        }
        other => panic!("Expected SAT for valid remainder, but got {:?}", other),
    }
}

#[test]
fn test_srem_inconsistent() {
    // Test that inconsistent signed remainder is detected
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let rem_result = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Encode: rem_result = x % y
    solver.bv_srem(rem_result, x, y);

    // Fix x and y
    solver.assert_const(x, 13, 8); // 13
    solver.assert_const(y, 6, 8); // 6

    // 13 % 6 = 1, but assert rem_result = 2 (wrong!)
    solver.assert_const(rem_result, 2, 8);

    // Should be UNSAT (inconsistent)
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {
            // Good! Inconsistency detected
        }
        other => panic!(
            "Expected UNSAT for inconsistent remainder, but got {:?}",
            other
        ),
    }
}

#[test]
fn test_multiple_ult_chaining() {
    // Test that multiple comparisons work correctly
    // x < y AND y < z should be satisfiable
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let z = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);
    solver.new_bv(z, 8);

    // Assert: x < y
    solver.assert_ult(x, y);

    // Assert: y < z
    solver.assert_ult(y, z);

    // Should be SAT (x < y < z is possible)
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Sat => {
            // Good!
        }
        other => panic!("Expected SAT but got {:?}", other),
    }
}

#[test]
fn test_ult_cache_reuse() {
    // Test that the comparison cache works correctly with repeated assertions
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Assert x < y twice
    solver.assert_ult(x, y);
    solver.assert_ult(x, y);

    // Should still be SAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Sat => {
            // Good!
        }
        other => panic!("Expected SAT but got {:?}", other),
    }
}
