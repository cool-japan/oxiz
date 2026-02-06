//! Even more minimal add + ult test

use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::bv::BvSolver;

/// Very simple: a + b = 100, no ult
/// Should be SAT
#[test]
fn test_add_sum_only() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    solver.assert_const(target, 100, width);

    solver.bv_add(sum, a, b);
    solver.assert_eq(sum, target);

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
        }
        other => {
            panic!("Should be SAT: {:?}", other);
        }
    }
}

/// a + b = 100, then add a < b constraint after the fact
/// Should be SAT
#[test]
fn test_add_then_ult() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    solver.assert_const(target, 100, width);

    // First: add equation
    solver.bv_add(sum, a, b);
    solver.assert_eq(sum, target);

    // Then: ult constraint
    solver.assert_ult(a, b);

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
            if let (Some(av), Some(bv)) = (a_val, b_val) {
                assert!(av < bv, "a {} should be < b {}", av, bv);
                assert_eq!((av + bv) % 256, 100, "a+b should be 100");
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT");
        }
    }
}

/// ULT first, then add constraint
/// Should be SAT
#[test]
fn test_ult_then_add() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    solver.assert_const(target, 100, width);

    // First: ult constraint
    solver.assert_ult(a, b);

    // Then: add equation
    solver.bv_add(sum, a, b);
    solver.assert_eq(sum, target);

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
            if let (Some(av), Some(bv)) = (a_val, b_val) {
                assert!(av < bv, "a {} should be < b {}", av, bv);
                assert_eq!((av + bv) % 256, 100, "a+b should be 100");
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT");
        }
    }
}

/// Smaller width - 4 bits
/// a + b = 10 (mod 16), a < b
#[test]
fn test_add_ult_4bit() {
    let mut solver = BvSolver::new();
    let width = 4u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    solver.assert_const(target, 10, width);

    solver.bv_add(sum, a, b);
    solver.assert_eq(sum, target);
    solver.assert_ult(a, b);

    // Solutions: a=0,b=10; a=1,b=9; a=2,b=8; a=3,b=7; a=4,b=6
    // (Needs a < b, so a+b=10 => a < 5)

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
            if let (Some(av), Some(bv)) = (a_val, b_val) {
                assert!(av < bv, "a {} should be < b {}", av, bv);
                assert_eq!((av + bv) % 16, 10, "a+b should be 10 mod 16");
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT");
        }
    }
}

/// Even simpler - 2 bits
/// a + b = 2 (mod 4), a < b
/// Solutions: a=0,b=2; a=1,b=1 is not a<b
#[test]
fn test_add_ult_2bit() {
    let mut solver = BvSolver::new();
    let width = 2u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    solver.assert_const(target, 2, width); // 10 in binary

    solver.bv_add(sum, a, b);
    solver.assert_eq(sum, target);
    solver.assert_ult(a, b);

    // Valid solutions:
    // a=0, b=2 (0+2=2, 0<2) ✓
    // a=1, b=1 (1+1=2, but 1<1 is false) ✗

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
            if let (Some(av), Some(bv)) = (a_val, b_val) {
                assert!(av < bv, "a {} should be < b {}", av, bv);
                assert_eq!((av + bv) % 4, 2, "a+b should be 2 mod 4");
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT with a=0, b=2");
        }
    }
}
