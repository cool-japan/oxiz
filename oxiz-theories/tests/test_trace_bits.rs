//! Trace bit allocation to debug the issue

use oxiz_core::ast::TermId;
use oxiz_theories::Theory;
use oxiz_theories::bv::BvSolver;

/// Trace 4-bit case
#[test]
fn test_trace_4bit() {
    let mut solver = BvSolver::new();
    let width = 4u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    println!("Creating bitvectors...");
    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    println!("Setting target = 10...");
    solver.assert_const(target, 10, width); // 1010 in binary

    println!("Adding: sum = a + b");
    solver.bv_add(sum, a, b);

    println!("Adding: sum == target");
    solver.assert_eq(sum, target);

    println!("Adding: a < b");
    solver.assert_ult(a, b);

    println!("Solving...");
    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            let sum_val = solver.get_value(sum);
            let target_val = solver.get_value(target);
            println!("SAT:");
            println!("  a = {:?} (binary: {:04b})", a_val, a_val.unwrap_or(999));
            println!("  b = {:?} (binary: {:04b})", b_val, b_val.unwrap_or(999));
            println!(
                "  sum = {:?} (binary: {:04b})",
                sum_val,
                sum_val.unwrap_or(999)
            );
            println!(
                "  target = {:?} (binary: {:04b})",
                target_val,
                target_val.unwrap_or(999)
            );

            if let (Some(av), Some(bv)) = (a_val, b_val) {
                let expected_sum = (av + bv) % 16;
                println!(
                    "  Expected sum: a + b = {} + {} = {} (mod 16)",
                    av, bv, expected_sum
                );
            }
        }
        other => {
            println!("Result: {:?}", other);
        }
    }
}

/// Trace 2-bit case (which works)
#[test]
fn test_trace_2bit() {
    let mut solver = BvSolver::new();
    let width = 2u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    println!("Creating bitvectors...");
    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    println!("Setting target = 2...");
    solver.assert_const(target, 2, width); // 10 in binary

    println!("Adding: sum = a + b");
    solver.bv_add(sum, a, b);

    println!("Adding: sum == target");
    solver.assert_eq(sum, target);

    println!("Adding: a < b");
    solver.assert_ult(a, b);

    println!("Solving...");
    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            let sum_val = solver.get_value(sum);
            let target_val = solver.get_value(target);
            println!("SAT:");
            println!("  a = {:?} (binary: {:02b})", a_val, a_val.unwrap_or(999));
            println!("  b = {:?} (binary: {:02b})", b_val, b_val.unwrap_or(999));
            println!(
                "  sum = {:?} (binary: {:02b})",
                sum_val,
                sum_val.unwrap_or(999)
            );
            println!(
                "  target = {:?} (binary: {:02b})",
                target_val,
                target_val.unwrap_or(999)
            );

            if let (Some(av), Some(bv)) = (a_val, b_val) {
                let expected_sum = (av + bv) % 4;
                println!(
                    "  Expected sum: a + b = {} + {} = {} (mod 4)",
                    av, bv, expected_sum
                );
            }
        }
        other => {
            println!("Result: {:?}", other);
        }
    }
}

/// 3-bit test
#[test]
fn test_trace_3bit() {
    let mut solver = BvSolver::new();
    let width = 3u32;

    let a = TermId::new(1);
    let b = TermId::new(2);
    let sum = TermId::new(3);
    let target = TermId::new(4);

    solver.new_bv(a, width);
    solver.new_bv(b, width);
    solver.new_bv(sum, width);
    solver.new_bv(target, width);

    solver.assert_const(target, 5, width); // 101 in binary

    solver.bv_add(sum, a, b);
    solver.assert_eq(sum, target);
    solver.assert_ult(a, b);

    // Solutions: a=0,b=5; a=1,b=4; a=2,b=3

    match solver.check() {
        Ok(oxiz_theories::TheoryCheckResult::Sat) => {
            let a_val = solver.get_value(a);
            let b_val = solver.get_value(b);
            let sum_val = solver.get_value(sum);
            println!("3-bit SAT: a={:?}, b={:?}, sum={:?}", a_val, b_val, sum_val);

            if let (Some(av), Some(bv)) = (a_val, b_val) {
                let expected_sum = (av + bv) % 8;
                println!(
                    "  Verification: {} + {} = {} (mod 8), should be 5",
                    av, bv, expected_sum
                );
                assert!(av < bv, "a {} should be < b {}", av, bv);
                assert_eq!(expected_sum, 5, "a+b should be 5 mod 8");
            }
        }
        other => {
            println!("3-bit Result: {:?}", other);
            panic!("Should be SAT");
        }
    }
}
