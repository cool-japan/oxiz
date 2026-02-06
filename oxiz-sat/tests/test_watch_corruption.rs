//! Test for watched literal corruption

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    // Clauses:
    // C0: ~out | ~a | ~b (if out then not both a,b)
    // C1: ~out | a | b (if out then at least one of a,b)
    // C2: out | ~a | b (if not out and not a then b)
    // C3: out | a | ~b (if not out and not b then a)
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Minimal reproduction: just XOR + solve + a=F + solve
#[test]
fn test_xor_solve_af_solve() {
    let mut sat = Solver::new();
    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    encode_xor(&mut sat, out, a, b);

    println!("=== First solve (XOR only) ===");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    if r1 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: out={}, a={}, b={}",
            m[out.index()].is_true(),
            m[a.index()].is_true(),
            m[b.index()].is_true()
        );
    }

    println!("\n=== Adding a=F ===");
    sat.add_clause([Lit::neg(a)]);
    println!("Trail: {:?}", sat.trail().assignments());

    println!("\n=== Second solve (XOR + a=F) ===");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    println!(
        "Conflicts: {}, Learned: {}",
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );

    if r2 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: out={}, a={}, b={}",
            m[out.index()].is_true(),
            m[a.index()].is_true(),
            m[b.index()].is_true()
        );

        // Verify XOR: out = a XOR b
        let av = m[a.index()].is_true();
        let bv = m[b.index()].is_true();
        let outv = m[out.index()].is_true();
        assert_eq!(outv, av ^ bv, "XOR constraint violated!");
        assert!(!av, "a should be false");
    }

    // The second solve should be SAT
    assert_eq!(r2, SolverResult::Sat, "XOR + a=F should be SAT");
}

/// Same but with more detail on the second solve
#[test]
fn test_xor_af_detail() {
    // What the second solve should see:
    // With a=F:
    // C0: ~out | ~a | ~b = ~out | T | ~b = T (satisfied)
    // C1: ~out | a | b = ~out | F | b = ~out | b (binary-ish)
    // C2: out | ~a | b = out | T | b = T (satisfied)
    // C3: out | a | ~b = out | F | ~b = out | ~b (binary-ish)
    //
    // Effective constraints: ~out | b AND out | ~b
    // This means: (out => b) AND (~out => ~b)
    // Which simplifies to: out = b
    //
    // So any assignment where out = b should be SAT.

    let mut sat = Solver::new();
    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    encode_xor(&mut sat, out, a, b);
    sat.solve(); // This might mess up watches

    sat.add_clause([Lit::neg(a)]);

    // At this point, with a=F, we should be able to find a SAT solution
    // where out = b (either both true or both false)

    let r = sat.solve();
    println!("Result: {:?}, conflicts: {}", r, sat.stats().conflicts);

    if r == SolverResult::Sat {
        let m = sat.model();
        let av = m[a.index()].is_true();
        let bv = m[b.index()].is_true();
        let outv = m[out.index()].is_true();
        println!("Model: out={}, a={}, b={}", outv, av, bv);
        println!(
            "With a=F, we expect out=b. Got out={}, b={}, match={}",
            outv,
            bv,
            outv == bv
        );
    }

    assert_eq!(r, SolverResult::Sat);
}
