//! Minimal test for ULT with unknowns

use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::bv::BvSolver;

/// Minimal: rem < divisor with both unknown but sum constraint
/// 5 * divisor + rem = 100, rem < divisor
/// Should find divisor=20, rem=0
#[test]
fn test_ult_two_unknowns() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let five = TermId::new(1);
    let divisor = TermId::new(2);
    let remainder = TermId::new(3);
    let product = TermId::new(4);
    let sum = TermId::new(5);
    let target = TermId::new(6);

    solver.new_bv(five, width);
    solver.new_bv(divisor, width);
    solver.new_bv(remainder, width);
    solver.new_bv(product, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    solver.assert_const(five, 5, width);
    solver.assert_const(target, 100, width);

    // product = 5 * divisor
    solver.bv_mul(product, five, divisor);

    // sum = product + remainder
    solver.bv_add(sum, product, remainder);

    // sum == 100
    solver.assert_eq(sum, target);

    // remainder < divisor
    solver.assert_ult(remainder, divisor);

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let div_val = solver.get_value(divisor);
            let rem_val = solver.get_value(remainder);
            let prod_val = solver.get_value(product);
            let sum_val = solver.get_value(sum);
            println!(
                "SAT: divisor={:?}, rem={:?}, prod={:?}, sum={:?}",
                div_val, rem_val, prod_val, sum_val
            );

            if let (Some(d), Some(r)) = (div_val, rem_val) {
                println!(
                    "Verification: 5 * {} + {} = {}, rem {} < div {}",
                    d,
                    r,
                    5 * d + r,
                    r,
                    d
                );
                assert!(r < d, "rem {} should be < div {}", r, d);
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT with divisor=20, rem=0 (or other valid solutions)");
        }
    }
}

/// Even simpler: just two unknowns with a < b
/// Should find any a < b
#[test]
fn test_simple_ult_unknowns() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let a = TermId::new(1);
    let b = TermId::new(2);

    solver.new_bv(a, width);
    solver.new_bv(b, width);

    // a < b (both unknown)
    solver.assert_ult(a, b);

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
            if let (Some(av), Some(bv)) = (a_val, b_val) {
                assert!(av < bv, "a={} should be < b={}", av, bv);
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT - there are many solutions where a < b");
        }
    }
}

/// Test: a < b with b constrained to 20
#[test]
fn test_ult_b_constrained() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let a = TermId::new(1);
    let b = TermId::new(2);

    solver.new_bv(a, width);
    solver.new_bv(b, width);

    solver.assert_const(b, 20, width);

    // a < 20
    solver.assert_ult(a, b);

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
            if let Some(av) = a_val {
                assert!(av < 20, "a={} should be < 20", av);
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT - a can be 0..19");
        }
    }
}

/// Test: a < b with a constrained to 0
#[test]
fn test_ult_a_constrained() {
    let mut solver = BvSolver::new();
    let width = 8u32;

    let a = TermId::new(1);
    let b = TermId::new(2);

    solver.new_bv(a, width);
    solver.new_bv(b, width);

    solver.assert_const(a, 0, width);

    // 0 < b
    solver.assert_ult(a, b);

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            println!("SAT: a={:?}, b={:?}", a_val, b_val);
            if let Some(bv) = b_val {
                assert!(bv > 0, "b={} should be > 0", bv);
            }
        }
        other => {
            println!("Result: {:?}", other);
            panic!("Should be SAT - b can be 1..255");
        }
    }
}
