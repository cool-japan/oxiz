//! Minimal test for BV multiplication encoding

use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::bv::BvSolver;

/// Test if basic equality constraint works
#[test]
fn test_basic_equality() {
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // x = 13, y = 13 - should be SAT
    solver.assert_const(x, 13, 8);
    solver.assert_const(y, 13, 8);
    solver.assert_eq(x, y);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT but got {:?}", other),
    }
}

/// Test if basic inequality is detected
#[test]
fn test_basic_inequality_unsat() {
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    // x = 13, y = 6, but assert x == y - should be UNSAT
    solver.assert_const(x, 13, 8);
    solver.assert_const(y, 6, 8);
    solver.assert_eq(x, y);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {}
        other => panic!("Expected UNSAT but got {:?}", other),
    }
}

/// Test simple addition: 7 + 5 = 12
#[test]
fn test_simple_addition() {
    let mut solver = BvSolver::new();

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);

    solver.new_bv(a, 8);
    solver.new_bv(b, 8);
    solver.new_bv(sum, 8);

    solver.assert_const(a, 7, 8);
    solver.assert_const(b, 5, 8);

    // Encode sum = a + b using bv_add
    solver.bv_add(sum, a, b);

    // Assert sum = 12
    solver.assert_const(sum, 12, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT for 7+5=12, got {:?}", other),
    }
}

/// Test wrong addition result: 7 + 5 != 10
#[test]
fn test_wrong_addition() {
    let mut solver = BvSolver::new();

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);

    solver.new_bv(a, 8);
    solver.new_bv(b, 8);
    solver.new_bv(sum, 8);

    solver.assert_const(a, 7, 8);
    solver.assert_const(b, 5, 8);

    // Encode sum = a + b using bv_add
    solver.bv_add(sum, a, b);

    // Assert sum = 10 (wrong!)
    solver.assert_const(sum, 10, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {}
        other => panic!("Expected UNSAT for 7+5!=10, got {:?}", other),
    }
}

/// Test simple multiplication: 2 * 6 = 12
#[test]
fn test_simple_multiplication() {
    let mut solver = BvSolver::new();

    let a = TermId::new(1);
    let b = TermId::new(2);
    let prod = TermId::new(3);

    solver.new_bv(a, 8);
    solver.new_bv(b, 8);
    solver.new_bv(prod, 8);

    solver.assert_const(a, 2, 8);
    solver.assert_const(b, 6, 8);

    // Encode prod = a * b
    solver.bv_mul(prod, a, b);

    // Assert prod = 12
    solver.assert_const(prod, 12, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Sat => {}
        other => panic!("Expected SAT for 2*6=12, got {:?}", other),
    }
}

/// Test wrong multiplication result: 2 * 6 != 13
#[test]
fn test_wrong_multiplication() {
    let mut solver = BvSolver::new();

    let a = TermId::new(1);
    let b = TermId::new(2);
    let prod = TermId::new(3);

    solver.new_bv(a, 8);
    solver.new_bv(b, 8);
    solver.new_bv(prod, 8);

    solver.assert_const(a, 2, 8);
    solver.assert_const(b, 6, 8);

    // Encode prod = a * b
    solver.bv_mul(prod, a, b);

    // Assert prod = 13 (wrong!)
    solver.assert_const(prod, 13, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {}
        other => panic!("Expected UNSAT for 2*6!=13, got {:?}", other),
    }
}
