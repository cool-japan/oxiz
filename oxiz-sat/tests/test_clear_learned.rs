//! Test if clearing learned clauses fixes the issue

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Try with reset between solves
#[test]
fn test_with_reset() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    encode_xor(&mut sat, x, a, b);

    let r1 = sat.solve();
    println!(
        "First solve: {:?}, learned: {}",
        r1,
        sat.stats().learned_clauses
    );

    // Reset and re-add clauses
    sat.reset();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();
    encode_xor(&mut sat, x, a, b);
    sat.add_clause([Lit::pos(x)]);
    sat.add_clause([Lit::neg(a)]);

    let r2 = sat.solve();
    println!("After reset + constraints: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);
}

/// What happens if we just do the second part fresh?
#[test]
fn test_fresh_after_xor() {
    // First solver - just XOR
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();
        encode_xor(&mut sat, x, a, b);

        let r = sat.solve();
        println!(
            "First solver (XOR only): {:?}, learned: {}",
            r,
            sat.stats().learned_clauses
        );
    }

    // Second solver - XOR + constraints
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();
        encode_xor(&mut sat, x, a, b);
        sat.add_clause([Lit::pos(x)]);
        sat.add_clause([Lit::neg(a)]);

        let r = sat.solve();
        println!("Second solver (XOR + constraints): {:?}", r);
        assert_eq!(r, SolverResult::Sat);
    }
}

/// Debug: what does the first solve actually learn?
#[test]
fn test_debug_learning() {
    let mut sat = Solver::new();
    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    encode_xor(&mut sat, x, a, b);

    let r1 = sat.solve();
    println!("First solve: {:?}", r1);
    println!("Stats:");
    println!("  Decisions: {}", sat.stats().decisions);
    println!("  Conflicts: {}", sat.stats().conflicts);
    println!("  Propagations: {}", sat.stats().propagations);
    println!("  Learned clauses: {}", sat.stats().learned_clauses);

    if r1 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: x={}, a={}, b={}",
            m[x.index()].is_true(),
            m[a.index()].is_true(),
            m[b.index()].is_true()
        );
    }
}
