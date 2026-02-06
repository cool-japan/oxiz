//! Test cases for BV conflict detection bugs
//! These are minimal reproducers for the failing Z3 parity tests

use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::bv::BvSolver;

#[test]
fn test_bv02_or_conflict() {
    // bv_02.smt2: OR operation with contradictory constraints
    // (bvor a b) = 0b11111111
    // a = 0b10101010
    // b = 0b01010100
    // Should be UNSAT: 0b10101010 OR 0b01010100 = 0b11111110, not 0b11111111

    let mut solver = BvSolver::new();

    let a = TermId::new(1);
    let b = TermId::new(2);
    let result = TermId::new(3);

    // Create variables
    solver.new_bv(a, 8);
    solver.new_bv(b, 8);

    // Encode: result = a | b
    solver.bv_or(result, a, b);

    // Assert constraints
    solver.assert_const(a, 0b10101010, 8);
    solver.assert_const(b, 0b01010100, 8);
    solver.assert_const(result, 0b11111111, 8);

    // Should detect UNSAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {} // Good!
        other => panic!("Expected UNSAT but got {:?}", other),
    }
}

#[test]
fn test_bv06_sub_conflict() {
    // bv_06.smt2: Contradictory subtraction
    // x - y = 100
    // y - x = 100
    // Should be UNSAT

    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let sub1 = TermId::new(3); // x - y
    let sub2 = TermId::new(4); // y - x

    solver.new_bv(x, 16);
    solver.new_bv(y, 16);

    // Encode: sub1 = x - y
    solver.bv_sub(sub1, x, y);
    // Encode: sub2 = y - x
    solver.bv_sub(sub2, y, x);

    // Assert both equal 100
    solver.assert_const(sub1, 100, 16);
    solver.assert_const(sub2, 100, 16);

    // Should detect UNSAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {} // Good!
        other => panic!("Expected UNSAT but got {:?}", other),
    }
}

#[test]
fn test_bv11_urem_conflict() {
    // bv_11.smt2: Invalid remainder constraint
    // x % y = 10
    // y = 5
    // Should be UNSAT: remainder cannot be >= divisor

    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let rem = TermId::new(3);

    solver.new_bv(x, 16);
    solver.new_bv(y, 16);

    // Encode: rem = x % y
    solver.bv_urem(rem, x, y);

    // Assert constraints
    solver.assert_const(y, 5, 16);
    solver.assert_const(rem, 10, 16);

    // Should detect UNSAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {} // Good!
        other => panic!("Expected UNSAT but got {:?}", other),
    }
}

#[test]
fn test_bv12_sdiv_srem_conflict() {
    // bv_12.smt2: Inconsistent signed division and remainder
    // x / y = -2 (signed)
    // x % y = 1 (signed)
    // y = 6
    // Should be UNSAT: the constraints are inconsistent

    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let div = TermId::new(3);
    let rem = TermId::new(4);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Encode: div = x / y (signed)
    solver.bv_sdiv(div, x, y);
    // Encode: rem = x % y (signed)
    solver.bv_srem(rem, x, y);

    // Assert constraints
    solver.assert_const(div, 0xFE, 8); // -2 in two's complement
    solver.assert_const(rem, 0x01, 8); // 1
    solver.assert_const(y, 0x06, 8); // 6

    // Should detect UNSAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {} // Good!
        other => panic!("Expected UNSAT but got {:?}", other),
    }
}

#[test]
fn test_bv13_not_xor() {
    // bv_13.smt2: NOT and XOR operations
    // NOT(x) = y
    // x XOR y = 0xFF
    // NOT(NOT(x)) = x
    // Should be SAT

    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let not_x = TermId::new(3);
    let xor_xy = TermId::new(4);
    let not_not_x = TermId::new(5);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Encode: not_x = NOT(x)
    solver.bv_not(not_x, x);
    // Encode: xor_xy = x XOR y
    solver.bv_xor(xor_xy, x, y);
    // Encode: not_not_x = NOT(NOT(x))
    solver.bv_not(not_not_x, not_x);

    // Assert constraints
    solver.assert_eq(not_x, y);
    solver.assert_const(xor_xy, 0xFF, 8);
    solver.assert_eq(not_not_x, x);

    // Should be SAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Sat => {} // Good!
        other => panic!("Expected SAT but got {:?}", other),
    }
}

#[test]
fn test_bv15_ult_conflict() {
    // bv_15.smt2: Contradictory unsigned comparisons
    // x < y (unsigned)
    // y < x (unsigned)
    // Should be UNSAT

    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // Assert: x < y
    solver.assert_ult(x, y);
    // Assert: y < x
    solver.assert_ult(y, x);

    // Should detect UNSAT
    match solver.check().expect("check should succeed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {} // Good!
        other => panic!("Expected UNSAT but got {:?}", other),
    }
}
