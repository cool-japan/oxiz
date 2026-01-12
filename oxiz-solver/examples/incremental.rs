//! Incremental solving example
//!
//! This example demonstrates incremental SMT solving with push/pop:
//! - Adding and removing assertions
//! - Scoped solving
//! - Checking multiple related formulas efficiently

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_solver::{Solver, SolverResult};

fn main() {
    println!("=== Incremental Solving Example ===\n");

    // Example 1: Basic push/pop
    println!("Example 1: Basic push/pop");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        let p = tm.mk_var("p", tm.sorts.bool_sort);
        let q = tm.mk_var("q", tm.sorts.bool_sort);

        println!("  Level 0: Assert p");
        solver.assert(p, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => println!("  Check at level 0: SAT"),
            SolverResult::Unsat => println!("  Check at level 0: UNSAT"),
            SolverResult::Unknown => println!("  Check at level 0: UNKNOWN"),
        }

        println!("  Push to level 1");
        solver.push();

        println!("  Assert q at level 1");
        solver.assert(q, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => println!("  Check at level 1: SAT (p ∧ q)"),
            SolverResult::Unsat => println!("  Check at level 1: UNSAT"),
            SolverResult::Unknown => println!("  Check at level 1: UNKNOWN"),
        }

        println!("  Pop back to level 0");
        solver.pop();

        match solver.check(&mut tm) {
            SolverResult::Sat => println!("  Check at level 0 again: SAT (only p)"),
            SolverResult::Unsat => println!("  Check at level 0 again: UNSAT"),
            SolverResult::Unknown => println!("  Check at level 0 again: UNKNOWN"),
        }

        println!("  Context level: {}", solver.context_level());
    }

    println!();

    // Example 2: Multiple scopes
    println!("Example 2: Multiple nested scopes");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::from(0));

        println!("  Level 0: x >= 0");
        let c1 = tm.mk_ge(x, zero);
        solver.assert(c1, &mut tm);
        println!("    Context level: {}", solver.context_level());

        solver.push();
        println!("  Level 1: x <= 10");
        let ten = tm.mk_int(BigInt::from(10));
        let c2 = tm.mk_le(x, ten);
        solver.assert(c2, &mut tm);
        println!("    Context level: {}", solver.context_level());

        solver.push();
        println!("  Level 2: x == 5");
        let five = tm.mk_int(BigInt::from(5));
        let c3 = tm.mk_eq(x, five);
        solver.assert(c3, &mut tm);
        println!("    Context level: {}", solver.context_level());

        match solver.check(&mut tm) {
            SolverResult::Sat => println!("    SAT: x can be 5"),
            SolverResult::Unsat => println!("    UNSAT"),
            SolverResult::Unknown => println!("    UNKNOWN"),
        }

        solver.pop();
        println!("  Back to level 1 (x >= 0, x <= 10)");
        println!("    Context level: {}", solver.context_level());

        solver.pop();
        println!("  Back to level 0 (x >= 0 only)");
        println!("    Context level: {}", solver.context_level());
    }

    println!();

    // Example 3: Exploring different scenarios
    println!("Example 3: Exploring different scenarios");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        solver.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);

        // Base constraints
        let zero = tm.mk_int(BigInt::from(0));
        let c_base1 = tm.mk_ge(x, zero);
        let zero_y = tm.mk_int(BigInt::from(0));
        let c_base2 = tm.mk_ge(y, zero_y);
        solver.assert(c_base1, &mut tm);
        solver.assert(c_base2, &mut tm);

        println!("  Base: x >= 0, y >= 0");

        // Scenario 1: x + y = 10
        solver.push();
        println!("  Scenario 1: x + y = 10");
        let sum = tm.mk_add(vec![x, y]);
        let ten = tm.mk_int(BigInt::from(10));
        let scenario1 = tm.mk_eq(sum, ten);
        solver.assert(scenario1, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("    Result: SAT");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    let y_val = model.eval(y, &mut tm);
                    println!("    Example: x={:?}, y={:?}", x_val, y_val);
                }
            }
            SolverResult::Unsat => println!("    Result: UNSAT"),
            SolverResult::Unknown => println!("    Result: UNKNOWN"),
        }
        solver.pop();

        // Scenario 2: x = 2*y
        solver.push();
        println!("  Scenario 2: x = 2*y");
        let two = tm.mk_int(BigInt::from(2));
        let two_y = tm.mk_mul(vec![two, y]);
        let scenario2 = tm.mk_eq(x, two_y);
        solver.assert(scenario2, &mut tm);

        // Also add bound for concreteness
        let twenty = tm.mk_int(BigInt::from(20));
        let bound = tm.mk_le(x, twenty);
        solver.assert(bound, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("    Result: SAT");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    let y_val = model.eval(y, &mut tm);
                    println!("    Example: x={:?}, y={:?}", x_val, y_val);
                }
            }
            SolverResult::Unsat => println!("    Result: UNSAT"),
            SolverResult::Unknown => println!("    Result: UNKNOWN"),
        }
        solver.pop();

        // Scenario 3: x > y and x < 5
        solver.push();
        println!("  Scenario 3: x > y and x < 5");
        let c1 = tm.mk_gt(x, y);
        let five = tm.mk_int(BigInt::from(5));
        let c2 = tm.mk_lt(x, five);
        solver.assert(c1, &mut tm);
        solver.assert(c2, &mut tm);

        match solver.check(&mut tm) {
            SolverResult::Sat => {
                println!("    Result: SAT");
                if let Some(model) = solver.model() {
                    let x_val = model.eval(x, &mut tm);
                    let y_val = model.eval(y, &mut tm);
                    println!("    Example: x={:?}, y={:?}", x_val, y_val);
                }
            }
            SolverResult::Unsat => println!("    Result: UNSAT"),
            SolverResult::Unknown => println!("    Result: UNKNOWN"),
        }
        solver.pop();
    }

    println!();

    // Example 4: Incremental model checking
    println!("Example 4: Incremental verification");
    println!("  Check if formula holds under different assumptions");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);

        // Property to check: x + y >= 10
        let sum = tm.mk_add(vec![x, y]);
        let ten = tm.mk_int(BigInt::from(10));
        let property = tm.mk_ge(sum, ten);

        println!("  Property: x + y >= 10");

        // Try different assumptions
        let five = tm.mk_int(BigInt::from(5));
        let five_y = tm.mk_int(BigInt::from(5));
        let three = tm.mk_int(BigInt::from(3));
        let seven = tm.mk_int(BigInt::from(7));
        let two = tm.mk_int(BigInt::from(2));
        let two_y = tm.mk_int(BigInt::from(2));

        let assumptions = vec![
            (
                "x >= 5, y >= 5",
                vec![tm.mk_ge(x, five), tm.mk_ge(y, five_y)],
            ),
            (
                "x >= 3, y >= 7",
                vec![tm.mk_ge(x, three), tm.mk_ge(y, seven)],
            ),
            ("x >= 2, y >= 2", vec![tm.mk_ge(x, two), tm.mk_ge(y, two_y)]),
        ];

        for (desc, assums) in assumptions {
            solver.push();
            println!("\n  Testing with assumption: {}", desc);

            for a in &assums {
                solver.assert(*a, &mut tm);
            }

            // Check if property holds (negate it to find counterexample)
            let not_property = tm.mk_not(property);
            solver.assert(not_property, &mut tm);

            match solver.check(&mut tm) {
                SolverResult::Sat => {
                    println!("    Property VIOLATED");
                    if let Some(model) = solver.model() {
                        let x_val = model.eval(x, &mut tm);
                        let y_val = model.eval(y, &mut tm);
                        println!("    Counterexample: x={:?}, y={:?}", x_val, y_val);
                    }
                }
                SolverResult::Unsat => {
                    println!("    Property HOLDS (no counterexample found)");
                }
                SolverResult::Unknown => println!("    Result: UNKNOWN"),
            }

            solver.pop();
        }
    }

    println!();

    // Example 5: Assertion tracking
    println!("Example 5: Assertion tracking");
    {
        let mut solver = Solver::new();
        let mut tm = TermManager::new();

        let p = tm.mk_var("p", tm.sorts.bool_sort);
        let q = tm.mk_var("q", tm.sorts.bool_sort);

        println!("  Initial assertions: {}", solver.num_assertions());
        solver.assert(p, &mut tm);
        println!("  After asserting p: {}", solver.num_assertions());
        solver.assert(q, &mut tm);
        println!("  After asserting q: {}", solver.num_assertions());

        solver.push();
        let r = tm.mk_var("r", tm.sorts.bool_sort);
        solver.assert(r, &mut tm);
        println!("  After push and asserting r: {}", solver.num_assertions());

        solver.pop();
        println!("  After pop: {}", solver.num_assertions());
    }
}
