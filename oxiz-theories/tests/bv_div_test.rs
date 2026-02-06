//! Test for BV division encoding

use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::bv::BvSolver;

/// Test unsigned division: 13 / 6 = 2
#[test]
fn test_unsigned_division_sat() {
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let quot = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    solver.assert_const(x, 13, 8);
    solver.assert_const(y, 6, 8);

    // Encode quot = x / y (unsigned)
    solver.bv_udiv(quot, x, y);

    // Assert quot = 2
    solver.assert_const(quot, 2, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Sat => {
            println!("Unsigned div 13/6=2: SAT");
        }
        other => panic!("Expected SAT for 13/6=2, got {:?}", other),
    }
}

/// Test unsigned division wrong: 13 / 6 != 3
#[test]
fn test_unsigned_division_unsat() {
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let quot = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    solver.assert_const(x, 13, 8);
    solver.assert_const(y, 6, 8);

    // Encode quot = x / y (unsigned)
    solver.bv_udiv(quot, x, y);

    // Assert quot = 3 (wrong!)
    solver.assert_const(quot, 3, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {
            println!("Unsigned div 13/6!=3: UNSAT (correct)");
        }
        other => panic!("Expected UNSAT for 13/6!=3, got {:?}", other),
    }
}

/// Test unsigned remainder: 13 % 6 = 1
#[test]
fn test_unsigned_remainder_sat() {
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let rem = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    solver.assert_const(x, 13, 8);
    solver.assert_const(y, 6, 8);

    // Encode rem = x % y (unsigned)
    solver.bv_urem(rem, x, y);

    // Assert rem = 1
    solver.assert_const(rem, 1, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Sat => {
            println!("Unsigned rem 13%6=1: SAT");
        }
        other => panic!("Expected SAT for 13%6=1, got {:?}", other),
    }
}

/// Test unsigned remainder wrong: 13 % 6 != 2
#[test]
fn test_unsigned_remainder_unsat() {
    let mut solver = BvSolver::new();

    let x = TermId::new(1);
    let y = TermId::new(2);
    let rem = TermId::new(3);

    solver.new_bv(x, 8);
    solver.new_bv(y, 8);

    solver.assert_const(x, 13, 8);
    solver.assert_const(y, 6, 8);

    // Encode rem = x % y (unsigned)
    solver.bv_urem(rem, x, y);

    // Assert rem = 2 (wrong!)
    solver.assert_const(rem, 2, 8);

    match solver.check().expect("check failed") {
        oxiz_theories::TheoryCheckResult::Unsat(_) => {
            println!("Unsigned rem 13%6!=2: UNSAT (correct)");
        }
        other => panic!("Expected UNSAT for 13%6!=2, got {:?}", other),
    }
}
