//! Debug why incremental XOR returns UNSAT

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Debug the UNSAT issue
#[test]
fn test_unsat_debug() {
    let mut sat = Solver::new();

    let xor12 = sat.new_var();
    let a1 = sat.new_var();
    let b1 = sat.new_var();

    encode_xor(&mut sat, xor12, a1, b1);

    // First solve
    let r1 = sat.solve();
    println!("First solve: {:?}", r1);

    // Add xor12=true
    println!("\nAdding xor12=true...");
    let added1 = sat.add_clause([Lit::pos(xor12)]);
    println!("add_clause returned: {}", added1);

    // Second solve
    let r2 = sat.solve();
    println!("Second solve: {:?}", r2);

    // Add a1=false
    println!("\nAdding a1=false...");
    let added2 = sat.add_clause([Lit::neg(a1)]);
    println!("add_clause returned: {}", added2);

    // Check trail state
    println!("Trail: {:?}", sat.trail().assignments());

    // Third solve
    println!("\nThird solve...");
    let r3 = sat.solve();
    println!("Result: {:?}", r3);

    // If UNSAT, try to understand why
    if r3 == SolverResult::Unsat {
        println!("\nUnexpected UNSAT! Let's debug...");

        // Create a fresh solver with same constraints and verify it's SAT
        let mut fresh = Solver::new();
        let fx = fresh.new_var();
        let fa = fresh.new_var();
        let fb = fresh.new_var();
        encode_xor(&mut fresh, fx, fa, fb);
        fresh.add_clause([Lit::pos(fx)]);
        fresh.add_clause([Lit::neg(fa)]);

        let rf = fresh.solve();
        println!("Fresh solver (same constraints): {:?}", rf);
        if rf == SolverResult::Sat {
            let m = fresh.model();
            println!(
                "  Model: x={}, a={}, b={}",
                m[fx.index()].is_true(),
                m[fa.index()].is_true(),
                m[fb.index()].is_true()
            );
        }

        // The issue must be with the incremental solver state
        panic!("Third solve should be SAT, not UNSAT");
    }

    if r3 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: xor12={}, a1={}, b1={}",
            m[xor12.index()].is_true(),
            m[a1.index()].is_true(),
            m[b1.index()].is_true()
        );
    }
}

/// Step by step verification
#[test]
fn test_step_by_step() {
    // Step 1: XOR only
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();
        encode_xor(&mut sat, x, a, b);
        let r = sat.solve();
        println!("Step 1 (XOR only): {:?}", r);
        assert_eq!(r, SolverResult::Sat);
    }

    // Step 2: XOR + x=true
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();
        encode_xor(&mut sat, x, a, b);
        sat.add_clause([Lit::pos(x)]);
        let r = sat.solve();
        println!("Step 2 (XOR + x=T): {:?}", r);
        assert_eq!(r, SolverResult::Sat);
    }

    // Step 3: XOR + x=true + a=false (fresh)
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();
        encode_xor(&mut sat, x, a, b);
        sat.add_clause([Lit::pos(x)]);
        sat.add_clause([Lit::neg(a)]);
        let r = sat.solve();
        println!("Step 3 (XOR + x=T + a=F fresh): {:?}", r);
        assert_eq!(r, SolverResult::Sat);
        if r == SolverResult::Sat {
            let m = sat.model();
            println!(
                "  Model: x={}, a={}, b={}",
                m[x.index()].is_true(),
                m[a.index()].is_true(),
                m[b.index()].is_true()
            );
        }
    }

    // Step 4: XOR + solve + x=true + solve + a=false + solve (incremental)
    {
        let mut sat = Solver::new();
        let x = sat.new_var();
        let a = sat.new_var();
        let b = sat.new_var();
        encode_xor(&mut sat, x, a, b);

        let r1 = sat.solve();
        println!("Step 4a (XOR, solve): {:?}", r1);
        assert_eq!(r1, SolverResult::Sat);

        sat.add_clause([Lit::pos(x)]);
        let r2 = sat.solve();
        println!("Step 4b (+ x=T, solve): {:?}", r2);
        assert_eq!(r2, SolverResult::Sat);

        sat.add_clause([Lit::neg(a)]);
        println!(
            "Step 4c: Trail after adding a=F: {:?}",
            sat.trail().assignments()
        );

        let r3 = sat.solve();
        println!("Step 4c (+ a=F, solve): {:?}", r3);
        assert_eq!(r3, SolverResult::Sat, "Should be SAT with x=T, a=F, b=T");
    }
}
