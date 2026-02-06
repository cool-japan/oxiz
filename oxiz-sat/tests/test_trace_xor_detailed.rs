//! Detailed trace of XOR incremental solving

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Trace the exact issue
#[test]
fn test_xor_detailed_trace() {
    let mut sat = Solver::new();

    let xor12 = sat.new_var();
    let a1 = sat.new_var();
    let b1 = sat.new_var();

    println!(
        "Vars: xor12={}, a1={}, b1={}",
        xor12.index(),
        a1.index(),
        b1.index()
    );

    encode_xor(&mut sat, xor12, a1, b1);

    // First solve
    println!("\n=== First solve (just XOR) ===");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    if r1 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "  xor12={}, a1={}, b1={}",
            m[xor12.index()].is_true(),
            m[a1.index()].is_true(),
            m[b1.index()].is_true()
        );
    }

    // Add xor12=true
    println!("\n=== Adding xor12=true ===");
    sat.add_clause([Lit::pos(xor12)]);
    println!("Trail after add: {:?}", sat.trail().assignments());

    let r2 = sat.solve();
    println!("Second solve result: {:?}", r2);
    if r2 == SolverResult::Sat {
        let m = sat.model();
        let xv = m[xor12.index()].is_true();
        let av = m[a1.index()].is_true();
        let bv = m[b1.index()].is_true();
        println!("  xor12={}, a1={}, b1={}", xv, av, bv);
        println!(
            "  XOR check: {} XOR {} = {}, match={}",
            av,
            bv,
            av ^ bv,
            (av ^ bv) == xv
        );
    }

    // Add a1=false
    println!("\n=== Adding a1=false ===");
    sat.add_clause([Lit::neg(a1)]);
    println!("Trail after add: {:?}", sat.trail().assignments());

    let r3 = sat.solve();
    println!("Third solve result: {:?}", r3);

    // This should be SAT! xor12=T, a1=F => b1=T
    // Let's verify by checking with a fresh solver:
    println!("\n=== Verification with fresh solver ===");
    {
        let mut fresh = Solver::new();
        let fx = fresh.new_var();
        let fa = fresh.new_var();
        let fb = fresh.new_var();
        encode_xor(&mut fresh, fx, fa, fb);
        fresh.add_clause([Lit::pos(fx)]);
        fresh.add_clause([Lit::neg(fa)]);

        let rf = fresh.solve();
        println!("Fresh solver result: {:?}", rf);
        if rf == SolverResult::Sat {
            let m = fresh.model();
            println!(
                "  xor={}, a={}, b={}",
                m[fx.index()].is_true(),
                m[fa.index()].is_true(),
                m[fb.index()].is_true()
            );
        }
    }

    if r3 == SolverResult::Sat {
        let m = sat.model();
        let xv = m[xor12.index()].is_true();
        let av = m[a1.index()].is_true();
        let bv = m[b1.index()].is_true();
        println!("  xor12={}, a1={}, b1={}", xv, av, bv);

        assert!(xv, "xor12 should be true");
        assert!(!av, "a1 should be false");
        assert!(bv, "b1 should be true");
    } else {
        println!("\nERROR: Third solve should be SAT, not {:?}", r3);
        // Debug: what's the state of trivially_unsat?
    }
}
