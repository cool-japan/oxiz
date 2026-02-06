//! Test clause addition and propagation

use oxiz_sat::{Lit, Solver, SolverResult};

#[test]
fn test_unit_clause_conflict() {
    let mut sat = Solver::new();

    let a = sat.new_var();

    println!("Adding a=true...");
    let ok1 = sat.add_clause([Lit::pos(a)]);
    println!("Result: {}", ok1);

    println!("Adding a=false...");
    let ok2 = sat.add_clause([Lit::neg(a)]);
    println!("Result: {}", ok2);

    println!("Trying to solve...");
    match sat.solve() {
        SolverResult::Sat => panic!("Should be UNSAT"),
        SolverResult::Unsat => println!("Correctly UNSAT"),
        _ => panic!("Unknown"),
    }
}

#[test]
fn test_xor_then_unit() {
    let mut sat = Solver::new();

    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    println!("Adding XOR: out = a XOR b");
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);

    println!("Adding a=true...");
    sat.add_clause([Lit::pos(a)]);

    println!("Adding b=false...");
    let ok = sat.add_clause([Lit::neg(b)]);
    println!("Result: {}", ok);

    // out should be constrained to true (a XOR b = 1 XOR 0 = 1)
    println!("Solving...");
    match sat.solve() {
        SolverResult::Sat => {
            let model = sat.model();
            let a_val = model[a.index()].is_true();
            let b_val = model[b.index()].is_true();
            let out_val = model[out.index()].is_true();
            println!("SAT: a={}, b={}, out={}", a_val, b_val, out_val);
            assert!(a_val, "a should be true");
            assert!(!b_val, "b should be false");
            assert!(out_val, "out should be true (1 XOR 0 = 1)");
        }
        SolverResult::Unsat => panic!("Should be SAT"),
        _ => panic!("Unknown"),
    }
}

#[test]
fn test_xor_with_conflicting_unit() {
    let mut sat = Solver::new();

    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    println!("Adding XOR: out = a XOR b");
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);

    println!("Adding a=true...");
    sat.add_clause([Lit::pos(a)]);

    println!("Adding b=true...");
    sat.add_clause([Lit::pos(b)]);

    println!("Adding out=true...");
    let ok = sat.add_clause([Lit::pos(out)]);
    println!("Result: {}", ok);

    // This should be UNSAT: a=1, b=1 => out = 1 XOR 1 = 0, but out=1 is required
    println!("Solving...");
    match sat.solve() {
        SolverResult::Sat => {
            let model = sat.model();
            let a_val = model[a.index()].is_true();
            let b_val = model[b.index()].is_true();
            let out_val = model[out.index()].is_true();
            println!("SAT (BUG!): a={}, b={}, out={}", a_val, b_val, out_val);
            panic!("Should be UNSAT");
        }
        SolverResult::Unsat => println!("Correctly UNSAT"),
        _ => panic!("Unknown"),
    }
}

#[test]
fn test_simple_propagation() {
    let mut sat = Solver::new();

    let a = sat.new_var();
    let b = sat.new_var();

    // a => b (i.e., ~a | b)
    sat.add_clause([Lit::neg(a), Lit::pos(b)]);

    // a = true
    sat.add_clause([Lit::pos(a)]);

    // Now b should be implied (unit propagation)
    // Then adding b = false should conflict
    println!("Adding b=false...");
    let ok = sat.add_clause([Lit::neg(b)]);
    println!("Result: {}", ok);

    println!("Solving...");
    match sat.solve() {
        SolverResult::Sat => panic!("Should be UNSAT"),
        SolverResult::Unsat => println!("Correctly UNSAT"),
        _ => panic!("Unknown"),
    }
}
