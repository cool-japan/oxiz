//! Simple SAT solving example
//!
//! This example demonstrates basic propositional logic solving.
//! It creates boolean variables and checks satisfiability of formulas.

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_solver::{Solver, SolverResult};

fn main() {
    let mut solver = Solver::new();
    let mut tm = TermManager::new();

    println!("=== Simple SAT Solving Example ===\n");

    // Example 1: Satisfiable formula (p ∧ q)
    println!("Example 1: (p ∧ q)");
    let p = tm.mk_var("p", tm.sorts.bool_sort);
    let q = tm.mk_var("q", tm.sorts.bool_sort);
    let formula1 = tm.mk_and(vec![p, q]);
    solver.assert(formula1, &mut tm);

    match solver.check(&mut tm) {
        SolverResult::Sat => {
            println!("Result: SAT");
            if let Some(model) = solver.model() {
                println!("Model found with {} assignments", model.size());
                let p_val = model.eval(p, &mut tm);
                let q_val = model.eval(q, &mut tm);
                println!("  p = {:?}", p_val);
                println!("  q = {:?}", q_val);
            }
        }
        SolverResult::Unsat => println!("Result: UNSAT"),
        SolverResult::Unknown => println!("Result: UNKNOWN"),
    }

    println!();

    // Example 2: Unsatisfiable formula (p ∧ ¬p)
    println!("Example 2: (p ∧ ¬p)");
    let mut solver2 = Solver::new();
    let mut tm2 = TermManager::new();
    let p2 = tm2.mk_var("p", tm2.sorts.bool_sort);
    let not_p = tm2.mk_not(p2);
    solver2.assert(p2, &mut tm2);
    solver2.assert(not_p, &mut tm2);

    match solver2.check(&mut tm2) {
        SolverResult::Sat => println!("Result: SAT (unexpected)"),
        SolverResult::Unsat => println!("Result: UNSAT (as expected)"),
        SolverResult::Unknown => println!("Result: UNKNOWN"),
    }

    println!();

    // Example 3: More complex formula (p ∨ q) ∧ (¬p ∨ r) ∧ (¬q ∨ ¬r)
    println!("Example 3: (p ∨ q) ∧ (¬p ∨ r) ∧ (¬q ∨ ¬r)");
    let mut solver3 = Solver::new();
    let mut tm3 = TermManager::new();

    let p3 = tm3.mk_var("p", tm3.sorts.bool_sort);
    let q3 = tm3.mk_var("q", tm3.sorts.bool_sort);
    let r3 = tm3.mk_var("r", tm3.sorts.bool_sort);

    let clause1 = tm3.mk_or(vec![p3, q3]);
    let not_p3 = tm3.mk_not(p3);
    let clause2 = tm3.mk_or(vec![not_p3, r3]);
    let not_q3 = tm3.mk_not(q3);
    let not_r3 = tm3.mk_not(r3);
    let clause3 = tm3.mk_or(vec![not_q3, not_r3]);

    solver3.assert(clause1, &mut tm3);
    solver3.assert(clause2, &mut tm3);
    solver3.assert(clause3, &mut tm3);

    match solver3.check(&mut tm3) {
        SolverResult::Sat => {
            println!("Result: SAT");
            if let Some(model) = solver3.model() {
                println!("Model:");
                let p_val = model.eval(p3, &mut tm3);
                let q_val = model.eval(q3, &mut tm3);
                let r_val = model.eval(r3, &mut tm3);
                println!("  p = {:?}", p_val);
                println!("  q = {:?}", q_val);
                println!("  r = {:?}", r_val);
            }
        }
        SolverResult::Unsat => println!("Result: UNSAT"),
        SolverResult::Unknown => println!("Result: UNKNOWN"),
    }

    println!();

    // Example 4: XOR constraint (p ⊕ q)
    println!("Example 4: (p ⊕ q) - XOR constraint");
    let mut solver4 = Solver::new();
    let mut tm4 = TermManager::new();

    let p4 = tm4.mk_var("p", tm4.sorts.bool_sort);
    let q4 = tm4.mk_var("q", tm4.sorts.bool_sort);
    let xor = tm4.mk_xor(p4, q4);

    solver4.assert(xor, &mut tm4);

    match solver4.check(&mut tm4) {
        SolverResult::Sat => {
            println!("Result: SAT");
            if let Some(model) = solver4.model() {
                let p_val = model.eval(p4, &mut tm4);
                let q_val = model.eval(q4, &mut tm4);
                println!("  p = {:?}", p_val);
                println!("  q = {:?}", q_val);
            }
        }
        SolverResult::Unsat => println!("Result: UNSAT"),
        SolverResult::Unknown => println!("Result: UNKNOWN"),
    }

    println!();

    // Example 5: Distinct constraint
    println!("Example 5: distinct(x, y, z) where all must be different");
    let mut solver5 = Solver::new();
    let mut tm5 = TermManager::new();

    let x = tm5.mk_var("x", tm5.sorts.int_sort);
    let y = tm5.mk_var("y", tm5.sorts.int_sort);
    let z = tm5.mk_var("z", tm5.sorts.int_sort);

    let distinct = tm5.mk_distinct(vec![x, y, z]);
    solver5.assert(distinct, &mut tm5);

    // Add some bounds
    let zero = tm5.mk_int(BigInt::from(0));
    let two = tm5.mk_int(BigInt::from(2));
    let x_ge_0 = tm5.mk_ge(x, zero);
    let x_le_2 = tm5.mk_le(x, two);
    let zero_y = tm5.mk_int(BigInt::from(0));
    let two_y = tm5.mk_int(BigInt::from(2));
    let y_ge_0 = tm5.mk_ge(y, zero_y);
    let y_le_2 = tm5.mk_le(y, two_y);
    let zero_z = tm5.mk_int(BigInt::from(0));
    let two_z = tm5.mk_int(BigInt::from(2));
    let z_ge_0 = tm5.mk_ge(z, zero_z);
    let z_le_2 = tm5.mk_le(z, two_z);

    solver5.assert(x_ge_0, &mut tm5);
    solver5.assert(x_le_2, &mut tm5);
    solver5.assert(y_ge_0, &mut tm5);
    solver5.assert(y_le_2, &mut tm5);
    solver5.assert(z_ge_0, &mut tm5);
    solver5.assert(z_le_2, &mut tm5);

    match solver5.check(&mut tm5) {
        SolverResult::Sat => {
            println!("Result: SAT");
            if let Some(model) = solver5.model() {
                println!("Model (all values should be different 0, 1, or 2):");
                let x_val = model.eval(x, &mut tm5);
                let y_val = model.eval(y, &mut tm5);
                let z_val = model.eval(z, &mut tm5);
                println!("  x = {:?}", x_val);
                println!("  y = {:?}", y_val);
                println!("  z = {:?}", z_val);
            }
        }
        SolverResult::Unsat => println!("Result: UNSAT"),
        SolverResult::Unknown => println!("Result: UNKNOWN"),
    }
}
