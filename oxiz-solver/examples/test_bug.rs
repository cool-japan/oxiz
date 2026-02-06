//! Test case for the reported bug: x + y = 10, x > 5, x < 6 should be UNSAT for integers

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_solver::{Solver, SolverResult};

fn main() {
    println!("=== Bug Report Test Case ===\n");

    // Test: x + y = 10, x > 5, x < 6 (UNSAT for integers)
    println!("Test: x + y = 10, x > 5, x < 6 (should be UNSAT)");
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

        // x > 5
        let five = tm.mk_int(BigInt::from(5));
        let c1 = tm.mk_gt(x, five);

        // x < 6
        let six = tm.mk_int(BigInt::from(6));
        let c2 = tm.mk_lt(x, six);

        solver.assert(eq1, &mut tm);
        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("  Result: SAT (BUG! Should be UNSAT)");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    let y_val = model.eval(y, &mut tm);
                    println!("  Model: x = {:?}, y = {:?}", x_val, y_val);
                }
            }
            SolverResult::Unsat => println!("  Result: UNSAT (CORRECT!)"),
            SolverResult::Unknown => println!("  Result: UNKNOWN"),
        }
    }

    println!();

    // Also test: x > 5, x < 6 alone (should be UNSAT for integers)
    println!("Test: x > 5, x < 6 alone (should be UNSAT)");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);

        // x > 5
        let five = tm.mk_int(BigInt::from(5));
        let c1 = tm.mk_gt(x, five);

        // x < 6
        let six = tm.mk_int(BigInt::from(6));
        let c2 = tm.mk_lt(x, six);

        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("  Result: SAT (BUG! Should be UNSAT)");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    println!("  Model: x = {:?}", x_val);
                }
            }
            SolverResult::Unsat => println!("  Result: UNSAT (CORRECT!)"),
            SolverResult::Unknown => println!("  Result: UNKNOWN"),
        }
    }
}
