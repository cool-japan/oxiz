//! Trace incremental solving in detail

use oxiz_sat::{Lit, Solver, SolverResult};

/// Trace XOR solving step by step
#[test]
fn test_trace_xor_incremental() {
    let mut sat = Solver::new();

    let x = sat.new_var(); // 0 - XOR result
    let a = sat.new_var(); // 1
    let b = sat.new_var(); // 2

    println!("=== Setup XOR clauses ===");
    println!("x = a XOR b");
    println!("Clauses:");
    println!("  C0: ~x | ~a | ~b  (if x then not both a,b)");
    println!("  C1: ~x | a | b    (if x then at least one of a,b)");
    println!("  C2: x | ~a | b    (if not x and not a then b)");
    println!("  C3: x | a | ~b    (if not x and not b then a)");

    sat.add_clause([Lit::neg(x), Lit::neg(a), Lit::neg(b)]); // C0
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]); // C1 - KEY
    sat.add_clause([Lit::pos(x), Lit::neg(a), Lit::pos(b)]); // C2
    sat.add_clause([Lit::pos(x), Lit::pos(a), Lit::neg(b)]); // C3

    // First solve
    println!("\n=== First solve (no constraints) ===");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);

    if r1 == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("Model: x={}, a={}, b={}", xv, av, bv);
        println!(
            "XOR check: {} XOR {} = {}, x={}, match={}",
            av,
            bv,
            av ^ bv,
            xv,
            (av ^ bv) == xv
        );

        // Also check all clauses
        let check_clause = |lits: &[Lit]| -> bool {
            lits.iter().any(|&l| {
                let val = model[l.var().index()].is_true();
                if l.is_pos() { val } else { !val }
            })
        };

        let c0 = check_clause(&[Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
        let c1 = check_clause(&[Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
        let c2 = check_clause(&[Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
        let c3 = check_clause(&[Lit::pos(x), Lit::pos(a), Lit::neg(b)]);
        println!(
            "Clause satisfaction: C0={}, C1={}, C2={}, C3={}",
            c0, c1, c2, c3
        );

        assert!(c0 && c1 && c2 && c3, "All clauses must be satisfied");
    }

    // Add x = true
    println!("\n=== Adding unit clause: x = true ===");
    sat.add_clause([Lit::pos(x)]);

    println!("\n=== Second solve (x=true) ===");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);

    if r2 == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("Model: x={}, a={}, b={}", xv, av, bv);
        println!(
            "XOR check: {} XOR {} = {}, x={}, match={}",
            av,
            bv,
            av ^ bv,
            xv,
            (av ^ bv) == xv
        );

        // Check all clauses
        let check_clause = |lits: &[Lit]| -> bool {
            lits.iter().any(|&l| {
                let val = model[l.var().index()].is_true();
                if l.is_pos() { val } else { !val }
            })
        };

        let c0 = check_clause(&[Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
        let c1 = check_clause(&[Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
        let c2 = check_clause(&[Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
        let c3 = check_clause(&[Lit::pos(x), Lit::pos(a), Lit::neg(b)]);
        let u1 = xv; // Unit clause x = true
        println!(
            "Clause satisfaction: C0={}, C1={}, C2={}, C3={}, U1(x=T)={}",
            c0, c1, c2, c3, u1
        );

        // Key assertions
        assert!(xv, "x must be true (unit clause)");
        assert!(av ^ bv, "Exactly one of a,b must be true when x=true");
        assert!(c0 && c1 && c2 && c3, "All XOR clauses must be satisfied");
    }

    // Add a = false
    println!("\n=== Adding unit clause: a = false ===");
    sat.add_clause([Lit::neg(a)]);

    println!("\n=== Third solve (x=true, a=false) ===");
    let r3 = sat.solve();
    println!("Result: {:?}", r3);

    if r3 == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("Model: x={}, a={}, b={}", xv, av, bv);

        // Check all clauses
        let check_clause = |lits: &[Lit]| -> bool {
            lits.iter().any(|&l| {
                let val = model[l.var().index()].is_true();
                if l.is_pos() { val } else { !val }
            })
        };

        let c0 = check_clause(&[Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
        let c1 = check_clause(&[Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
        let c2 = check_clause(&[Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
        let c3 = check_clause(&[Lit::pos(x), Lit::pos(a), Lit::neg(b)]);
        println!(
            "Clause satisfaction: C0={}, C1={}, C2={}, C3={}",
            c0, c1, c2, c3
        );

        // For x=true, a=false:
        // C0: ~x|~a|~b = F|T|~b = T (always satisfied when a=false)
        // C1: ~x|a|b = F|F|b = b (requires b=true!)
        // C2: x|~a|b = T (satisfied by x)
        // C3: x|a|~b = T (satisfied by x)
        // So b MUST be true!

        assert!(xv, "x must be true");
        assert!(!av, "a must be false");
        assert!(bv, "b must be true (required by C1: ~x|a|b with x=T, a=F)");
        assert!(c0 && c1 && c2 && c3, "All XOR clauses must be satisfied");
    } else {
        // If UNSAT, that's wrong
        panic!("Should be SAT with x=true, a=false, b=true");
    }
}
