//! Test if learned clauses cause the spurious UNSAT

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Test with no intermediate solve - no learned clauses
#[test]
fn test_no_learned_clauses() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    encode_xor(&mut sat, x, a, b);
    sat.add_clause([Lit::pos(x)]);
    sat.add_clause([Lit::neg(a)]);

    // Single solve, no learned clauses from previous iterations
    let r = sat.solve();
    println!("No learned clauses: {:?}", r);
    assert_eq!(r, SolverResult::Sat);
}

/// Test with solve after XOR - might learn clauses
#[test]
fn test_with_xor_solve() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    encode_xor(&mut sat, x, a, b);

    // First solve may learn clauses
    let r1 = sat.solve();
    println!("After XOR solve: {:?}", r1);
    println!("Learned clauses count: {}", sat.stats().learned_clauses);

    // Add constraints
    sat.add_clause([Lit::pos(x)]);
    sat.add_clause([Lit::neg(a)]);

    // This solve sees learned clauses from first solve
    let r2 = sat.solve();
    println!("With learned clauses: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);
}

/// Test adding both unit clauses before second solve
#[test]
fn test_both_units_before_solve() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    encode_xor(&mut sat, x, a, b);

    // First solve
    let r1 = sat.solve();
    println!("First solve: {:?}", r1);

    // Add BOTH unit clauses before second solve
    sat.add_clause([Lit::pos(x)]);
    sat.add_clause([Lit::neg(a)]);

    // Second solve
    let r2 = sat.solve();
    println!("Both units before solve: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);
}

/// Test adding unit clauses one at a time with solve in between
#[test]
fn test_units_one_at_a_time() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    encode_xor(&mut sat, x, a, b);

    // First solve
    let r1 = sat.solve();
    println!("First solve: {:?}", r1);

    // Add first unit, solve
    sat.add_clause([Lit::pos(x)]);
    let r2 = sat.solve();
    println!("After x=T: {:?}", r2);

    // Add second unit, solve
    sat.add_clause([Lit::neg(a)]);
    let r3 = sat.solve();
    println!("After a=F: {:?}", r3);

    assert_eq!(r3, SolverResult::Sat);
}
