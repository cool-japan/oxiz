//! Verify XOR constraint behavior

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Fresh solver: XOR + a=F + b=F should be SAT with out=F
#[test]
fn test_xor_a_false_b_false_fresh() {
    let mut sat = Solver::new();
    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    encode_xor(&mut sat, out, a, b);
    sat.add_clause([Lit::neg(a)]); // a = false
    sat.add_clause([Lit::neg(b)]); // b = false

    let r = sat.solve();
    println!("Fresh (XOR + a=F + b=F): {:?}", r);

    // Should be SAT with out = a XOR b = false XOR false = false
    assert_eq!(r, SolverResult::Sat);
    let m = sat.model();
    assert!(!m[a.index()].is_true(), "a should be false");
    assert!(!m[b.index()].is_true(), "b should be false");
    assert!(
        !m[out.index()].is_true(),
        "out should be false (F XOR F = F)"
    );
}

/// Incremental: Same constraints added incrementally
#[test]
fn test_xor_a_false_b_false_incremental() {
    let mut sat = Solver::new();
    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    encode_xor(&mut sat, out, a, b);

    // First solve
    let r1 = sat.solve();
    println!("1. XOR only: {:?}", r1);
    assert_eq!(r1, SolverResult::Sat);

    // Add a=false
    sat.add_clause([Lit::neg(a)]);
    let r2 = sat.solve();
    println!("2. + a=F: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);

    // Add b=false
    sat.add_clause([Lit::neg(b)]);
    let r3 = sat.solve();
    println!("3. + b=F: {:?}", r3);

    // CRITICAL: This should be SAT!
    // out = a XOR b = false XOR false = false
    assert_eq!(
        r3,
        SolverResult::Sat,
        "XOR + a=F + b=F should be SAT with out=F"
    );

    let m = sat.model();
    println!(
        "Model: a={}, b={}, out={}",
        m[a.index()].is_true(),
        m[b.index()].is_true(),
        m[out.index()].is_true()
    );
}
