//! Debug the remaining XOR incremental bug

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Debug the incremental issue
#[test]
fn test_debug_incremental_xor() {
    let mut sat = Solver::new();
    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    encode_xor(&mut sat, out, a, b);

    println!(
        "Initial stats: learned={}, conflicts={}",
        sat.stats().learned_clauses,
        sat.stats().conflicts
    );

    // First solve
    let r1 = sat.solve();
    println!(
        "After solve 1: {:?}, learned={}, conflicts={}",
        r1,
        sat.stats().learned_clauses,
        sat.stats().conflicts
    );

    // Add a=false
    sat.add_clause([Lit::neg(a)]);
    println!("After add a=F: trail={:?}", sat.trail().assignments());

    let r2 = sat.solve();
    println!(
        "After solve 2: {:?}, learned={}, conflicts={}",
        r2,
        sat.stats().learned_clauses,
        sat.stats().conflicts
    );

    // Add b=false
    sat.add_clause([Lit::neg(b)]);
    println!("After add b=F: trail={:?}", sat.trail().assignments());

    let r3 = sat.solve();
    println!(
        "After solve 3: {:?}, learned={}, conflicts={}",
        r3,
        sat.stats().learned_clauses,
        sat.stats().conflicts
    );

    // Compare with fresh solver
    println!("\n=== Fresh solver comparison ===");
    {
        let mut fresh = Solver::new();
        let a = fresh.new_var();
        let b = fresh.new_var();
        let out = fresh.new_var();
        encode_xor(&mut fresh, out, a, b);
        fresh.add_clause([Lit::neg(a)]);
        fresh.add_clause([Lit::neg(b)]);
        let r = fresh.solve();
        println!("Fresh result: {:?}", r);
    }

    assert_eq!(r3, SolverResult::Sat);
}

/// Test with intermediate solve between first and second unit clause
#[test]
fn test_intermediate_solve() {
    let mut sat = Solver::new();
    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    encode_xor(&mut sat, out, a, b);

    // Add a=false BEFORE any solve
    sat.add_clause([Lit::neg(a)]);
    println!(
        "After add a=F (no prior solve): trail={:?}",
        sat.trail().assignments()
    );

    // First solve
    let r1 = sat.solve();
    println!("Solve 1: {:?}, learned={}", r1, sat.stats().learned_clauses);

    // Add b=false
    sat.add_clause([Lit::neg(b)]);
    println!("After add b=F: trail={:?}", sat.trail().assignments());

    // Second solve
    let r2 = sat.solve();
    println!("Solve 2: {:?}, learned={}", r2, sat.stats().learned_clauses);

    assert_eq!(r2, SolverResult::Sat);
}
