//! Optimization example
//!
//! This example demonstrates SMT optimization (OMT) features:
//! - Minimization and maximization
//! - Lexicographic optimization (multiple objectives with priorities)
//! - Pareto optimization (multi-objective)

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_solver::{OptimizationResult, Optimizer};

fn main() {
    println!("=== SMT Optimization Example ===\n");

    // Example 1: Simple minimization
    println!("Example 1: Minimize x subject to x >= 5");
    {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let five = tm.mk_int(BigInt::from(5));
        let c1 = tm.mk_ge(x, five);

        optimizer.assert(c1);
        optimizer.minimize(x);

        match optimizer.optimize(&mut tm) {
            OptimizationResult::Optimal { value, .. } => {
                println!("  Result: Optimal");
                println!("  Optimal value: {:?}", value);
                if let Some(term) = tm.get(value) {
                    println!("  Term kind: {:?}", term.kind);
                }
            }
            OptimizationResult::Unbounded => println!("  Result: Unbounded"),
            OptimizationResult::Unsat => println!("  Result: Unsat"),
            OptimizationResult::Unknown => println!("  Result: Unknown"),
        }
    }

    println!();

    // Example 2: Simple maximization
    println!("Example 2: Maximize y subject to y <= 20");
    {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let y = tm.mk_var("y", tm.sorts.int_sort);
        let twenty = tm.mk_int(BigInt::from(20));
        let c1 = tm.mk_le(y, twenty);

        optimizer.assert(c1);
        optimizer.maximize(y);

        match optimizer.optimize(&mut tm) {
            OptimizationResult::Optimal { value, .. } => {
                println!("  Result: Optimal");
                println!("  Optimal value: {:?}", value);
            }
            OptimizationResult::Unbounded => println!("  Result: Unbounded"),
            OptimizationResult::Unsat => println!("  Result: Unsat"),
            OptimizationResult::Unknown => println!("  Result: Unknown"),
        }
    }

    println!();

    // Example 3: Constrained optimization
    println!("Example 3: Minimize x + y subject to:");
    println!("  x >= 0, y >= 0");
    println!("  2*x + y >= 10");
    println!("  x + 2*y >= 10");
    {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::from(0));
        let ten = tm.mk_int(BigInt::from(10));
        let two = tm.mk_int(BigInt::from(2));

        // Bounds
        let c1 = tm.mk_ge(x, zero);
        let zero_y = tm.mk_int(BigInt::from(0));
        let c2 = tm.mk_ge(y, zero_y);

        // Constraints
        let two_x = tm.mk_mul(vec![two, x]);
        let c3_lhs = tm.mk_add(vec![two_x, y]);
        let c3 = tm.mk_ge(c3_lhs, ten);

        let two_for_y = tm.mk_int(BigInt::from(2));
        let two_y = tm.mk_mul(vec![two_for_y, y]);
        let c4_lhs = tm.mk_add(vec![x, two_y]);
        let ten_c4 = tm.mk_int(BigInt::from(10));
        let c4 = tm.mk_ge(c4_lhs, ten_c4);

        optimizer.assert(c1);
        optimizer.assert(c2);
        optimizer.assert(c3);
        optimizer.assert(c4);

        // Minimize x + y
        let objective = tm.mk_add(vec![x, y]);
        optimizer.minimize(objective);

        match optimizer.optimize(&mut tm) {
            OptimizationResult::Optimal { value, model } => {
                println!("  Result: Optimal");
                println!("  Optimal value of (x + y): {:?}", value);
                println!("  Model:");
                let x_val = model.eval(x, &mut tm);
                let y_val = model.eval(y, &mut tm);
                println!("    x = {:?}", x_val);
                println!("    y = {:?}", y_val);
            }
            OptimizationResult::Unbounded => println!("  Result: Unbounded"),
            OptimizationResult::Unsat => println!("  Result: Unsat"),
            OptimizationResult::Unknown => println!("  Result: Unknown"),
        }
    }

    println!();

    // Example 4: Lexicographic optimization
    println!("Example 4: Lexicographic optimization");
    println!("  Minimize x (priority 0), then minimize y (priority 1)");
    println!("  Subject to: x + y >= 10, x >= 0, y >= 0");
    {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::from(0));
        let ten = tm.mk_int(BigInt::from(10));

        let c1 = tm.mk_ge(x, zero);
        let zero_y = tm.mk_int(BigInt::from(0));
        let c2 = tm.mk_ge(y, zero_y);
        let sum = tm.mk_add(vec![x, y]);
        let c3 = tm.mk_ge(sum, ten);

        optimizer.assert(c1);
        optimizer.assert(c2);
        optimizer.assert(c3);

        // Minimize x first (priority 0)
        optimizer.minimize(x);
        // Then minimize y (priority 1)
        optimizer.minimize(y);

        match optimizer.optimize(&mut tm) {
            OptimizationResult::Optimal { value, model } => {
                println!("  Result: Optimal");
                println!("  Final objective value: {:?}", value);
                println!("  Model:");
                let x_val = model.eval(x, &mut tm);
                let y_val = model.eval(y, &mut tm);
                println!("    x = {:?}", x_val);
                println!("    y = {:?}", y_val);
                println!("  (x should be minimized first, then y)");
            }
            OptimizationResult::Unbounded => println!("  Result: Unbounded"),
            OptimizationResult::Unsat => println!("  Result: Unsat"),
            OptimizationResult::Unknown => println!("  Result: Unknown"),
        }
    }

    println!();

    // Example 5: Pareto optimization
    println!("Example 5: Pareto optimization (multi-objective)");
    println!("  Minimize both x and y simultaneously");
    println!("  Subject to: x + y >= 10, x >= 0, y >= 0");
    {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::from(0));
        let ten = tm.mk_int(BigInt::from(10));

        let c1 = tm.mk_ge(x, zero);
        let zero_y = tm.mk_int(BigInt::from(0));
        let c2 = tm.mk_ge(y, zero_y);
        let sum = tm.mk_add(vec![x, y]);
        let c3 = tm.mk_ge(sum, ten);

        optimizer.assert(c1);
        optimizer.assert(c2);
        optimizer.assert(c3);

        optimizer.minimize(x);
        optimizer.minimize(y);

        let pareto_points = optimizer.pareto_optimize(&mut tm);
        println!("  Found {} Pareto-optimal points:", pareto_points.len());
        for (idx, point) in pareto_points.iter().enumerate() {
            println!("  Point {}:", idx + 1);
            if point.values.len() >= 2 {
                println!("    Objectives: {:?}", point.values);
                println!("    Model size: {} assignments", point.model.size());
            }
        }
    }

    println!();

    // Example 6: Unbounded optimization
    println!("Example 6: Unbounded maximization");
    println!("  Maximize x with only x >= 0 (unbounded)");
    {
        let mut optimizer = Optimizer::new();
        let mut tm = TermManager::new();

        optimizer.set_logic("QF_LIA");

        let x = tm.mk_var("x", tm.sorts.int_sort);
        let zero = tm.mk_int(BigInt::from(0));
        let c1 = tm.mk_ge(x, zero);

        optimizer.assert(c1);
        optimizer.maximize(x);

        match optimizer.optimize(&mut tm) {
            OptimizationResult::Optimal { .. } => {
                println!("  Result: Optimal (found finite maximum)");
            }
            OptimizationResult::Unbounded => {
                println!("  Result: Unbounded (as expected - no finite maximum)");
            }
            OptimizationResult::Unsat => println!("  Result: Unsat"),
            OptimizationResult::Unknown => {
                println!("  Result: Unknown (may be unbounded)");
            }
        }
    }
}
