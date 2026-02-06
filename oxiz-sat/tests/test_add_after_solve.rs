//! Test adding clauses after solve

use oxiz_sat::{Lit, Solver, SolverResult};

/// Simple test: solve, then add conflicting unit clause
#[test]
fn test_add_unit_after_solve() {
    let mut sat = Solver::new();

    let a = sat.new_var();

    // Add a = true
    sat.add_clause([Lit::pos(a)]);

    println!("First solve...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    assert_eq!(r1, SolverResult::Sat);
    println!("a = {}", sat.model()[a.index()].is_true());

    // Now add a = false (conflicting)
    println!("\nAdding a = false...");
    let ok = sat.add_clause([Lit::neg(a)]);
    println!("add_clause returned: {}", ok);

    println!("\nSecond solve...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    // This should be UNSAT because a=true and a=false conflict
    assert_eq!(r2, SolverResult::Unsat, "Should be UNSAT");
}

/// Test: solve, then add non-conflicting unit clause
#[test]
fn test_add_unit_non_conflicting() {
    let mut sat = Solver::new();

    let a = sat.new_var();
    let b = sat.new_var();

    // a or b
    sat.add_clause([Lit::pos(a), Lit::pos(b)]);

    println!("First solve...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    assert_eq!(r1, SolverResult::Sat);
    println!(
        "a = {}, b = {}",
        sat.model()[a.index()].is_true(),
        sat.model()[b.index()].is_true()
    );

    // Add b = true
    println!("\nAdding b = true...");
    sat.add_clause([Lit::pos(b)]);

    println!("\nSecond solve...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);
    println!(
        "a = {}, b = {}",
        sat.model()[a.index()].is_true(),
        sat.model()[b.index()].is_true()
    );
    assert!(sat.model()[b.index()].is_true(), "b should be true");
}

/// Test: solve, then add constraining unit that rules out previous solution
#[test]
fn test_add_unit_rules_out_prev() {
    let mut sat = Solver::new();

    let a = sat.new_var();
    let b = sat.new_var();

    // a or b
    sat.add_clause([Lit::pos(a), Lit::pos(b)]);

    println!("First solve...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    assert_eq!(r1, SolverResult::Sat);
    let a1 = sat.model()[a.index()].is_true();
    let b1 = sat.model()[b.index()].is_true();
    println!("a = {}, b = {}", a1, b1);

    // If the solver found a=true, b=false, then adding a=false should force b=true
    // If the solver found a=true, b=true, then adding a=false should force b=true
    // If the solver found a=false, b=true, then adding a=false is consistent

    println!("\nAdding a = false...");
    sat.add_clause([Lit::neg(a)]);

    println!("\nSecond solve...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);
    let a2 = sat.model()[a.index()].is_true();
    let b2 = sat.model()[b.index()].is_true();
    println!("a = {}, b = {}", a2, b2);

    // a=false, b=true should be the only valid model
    assert!(!a2, "a should be false");
    assert!(b2, "b should be true (since a|b and ~a)");
}

/// Test with XOR constraint
#[test]
fn test_xor_add_unit_after_solve() {
    let mut sat = Solver::new();

    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    // out = a XOR b
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);

    // out = true
    sat.add_clause([Lit::pos(out)]);

    println!("First solve...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    assert_eq!(r1, SolverResult::Sat);
    let a1 = sat.model()[a.index()].is_true();
    let b1 = sat.model()[b.index()].is_true();
    let out1 = sat.model()[out.index()].is_true();
    println!("a = {}, b = {}, out = {}", a1, b1, out1);
    assert_eq!(a1 ^ b1, out1, "XOR constraint should hold");

    // Add a = false
    println!("\nAdding a = false...");
    sat.add_clause([Lit::neg(a)]);

    println!("\nSecond solve...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);
    let a2 = sat.model()[a.index()].is_true();
    let b2 = sat.model()[b.index()].is_true();
    let out2 = sat.model()[out.index()].is_true();
    println!("a = {}, b = {}, out = {}", a2, b2, out2);

    // With a=false and out=true, b must be true (false XOR true = true)
    assert!(!a2, "a should be false");
    assert!(b2, "b should be true (false XOR b = true => b=true)");
    assert!(out2, "out should be true");
    assert_eq!(
        a2 ^ b2,
        out2,
        "XOR constraint should hold after adding unit"
    );
}
