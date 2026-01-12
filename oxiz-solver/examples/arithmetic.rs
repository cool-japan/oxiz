//! Arithmetic constraints example
//!
//! This example demonstrates solving with linear integer arithmetic (LIA)
//! and linear real arithmetic (LRA).

use num_bigint::BigInt;
use num_rational::Rational64;
use oxiz_core::ast::TermManager;
use oxiz_solver::{Solver, SolverResult};

fn main() {
    println!("=== Arithmetic Constraints Example ===\n");

    // Example 1: Simple integer constraints
    println!("Example 1: Integer constraints");
    println!("  Find x such that: x >= 5 AND x <= 10");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let five = tm.mk_int(BigInt::from(5));
        let ten = tm.mk_int(BigInt::from(10));

        let c1 = tm.mk_ge(x, five);
        let c2 = tm.mk_le(x, ten);

        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("  Result: SAT");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    println!("  Model: x = {:?}", x_val);
                }
            }
            SolverResult::Unsat => println!("  Result: UNSAT"),
            SolverResult::Unknown => println!("  Result: UNKNOWN"),
        }
    }

    println!();

    // Example 2: Linear system
    println!("Example 2: Linear system");
    println!("  Find x, y such that:");
    println!("    x + y = 10");
    println!("    2*x - y = 5");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);

        // x + y = 10
        let sum = tm.mk_add(vec![x, y]);
        let ten = tm.mk_int(BigInt::from(10));
        let eq1 = tm.mk_eq(sum, ten);

        // 2*x - y = 5
        let two = tm.mk_int(BigInt::from(2));
        let two_x = tm.mk_mul(vec![two, x]);
        let diff = tm.mk_sub(two_x, y);
        let five = tm.mk_int(BigInt::from(5));
        let eq2 = tm.mk_eq(diff, five);

        solver.assert(eq1, &mut tm);
        solver.assert(eq2, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("  Result: SAT");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    let y_val = model.eval(y, &mut tm);
                    println!("  Model: x = {:?}, y = {:?}", x_val, y_val);
                }
            }
            SolverResult::Unsat => println!("  Result: UNSAT"),
            SolverResult::Unknown => println!("  Result: UNKNOWN"),
        }
    }

    println!();

    // Example 3: Inequality constraints
    println!("Example 3: Inequality constraints");
    println!("  Find x, y such that:");
    println!("    x > 0");
    println!("    y > 0");
    println!("    x + y < 100");
    println!("    2*x + 3*y >= 150");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::from(0));

        // x > 0
        let c1 = tm.mk_gt(x, zero);
        // y > 0
        let zero_y = tm.mk_int(BigInt::from(0));
        let c2 = tm.mk_gt(y, zero_y);
        // x + y < 100
        let sum = tm.mk_add(vec![x, y]);
        let hundred = tm.mk_int(BigInt::from(100));
        let c3 = tm.mk_lt(sum, hundred);
        // 2*x + 3*y >= 150
        let two = tm.mk_int(BigInt::from(2));
        let three = tm.mk_int(BigInt::from(3));
        let two_x = tm.mk_mul(vec![two, x]);
        let three_y = tm.mk_mul(vec![three, y]);
        let weighted_sum = tm.mk_add(vec![two_x, three_y]);
        let onefifty = tm.mk_int(BigInt::from(150));
        let c4 = tm.mk_ge(weighted_sum, onefifty);

        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);
        solver.assert(c3, &mut tm);
        solver.assert(c4, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("  Result: SAT");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    let y_val = model.eval(y, &mut tm);
                    println!("  Model: x = {:?}, y = {:?}", x_val, y_val);
                }
            }
            SolverResult::Unsat => println!("  Result: UNSAT (no solution exists)"),
            SolverResult::Unknown => println!("  Result: UNKNOWN"),
        }
    }

    println!();

    // Example 4: Real arithmetic
    println!("Example 4: Real arithmetic");
    println!("  Find x (real) such that: 0.5 <= x <= 1.5");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LRA");

        let x = tm.mk_var("x", tm.sorts.real_sort);
        let half = tm.mk_real(Rational64::new(1, 2));
        let one_half = tm.mk_real(Rational64::new(3, 2));

        let c1 = tm.mk_ge(x, half);
        let c2 = tm.mk_le(x, one_half);

        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("  Result: SAT");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    println!("  Model: x = {:?}", x_val);
                }
            }
            SolverResult::Unsat => println!("  Result: UNSAT"),
            SolverResult::Unknown => println!("  Result: UNKNOWN"),
        }
    }

    println!();

    // Example 5: Unsatisfiable constraints
    println!("Example 5: Unsatisfiable constraints");
    println!("  Find x such that: x > 10 AND x < 5 (impossible)");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let five = tm.mk_int(BigInt::from(5));
        let ten = tm.mk_int(BigInt::from(10));

        let c1 = tm.mk_gt(x, ten);
        let c2 = tm.mk_lt(x, five);

        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => println!("  Result: SAT (unexpected!)"),
            SolverResult::Unsat => println!("  Result: UNSAT (as expected - no solution)"),
            SolverResult::Unknown => println!("  Result: UNKNOWN"),
        }
    }
}
