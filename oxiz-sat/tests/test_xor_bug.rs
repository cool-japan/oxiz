//! Trace the XOR bug

use oxiz_sat::{Lit, Solver, SolverResult, Var};

fn encode_xor(sat: &mut Solver, out: Var, a: Var, b: Var) {
    // out <=> (a XOR b)
    // out = 1 iff exactly one of a, b is 1
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]); // ~out | ~a | ~b
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]); // ~out | a | b
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]); // out | ~a | b
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]); // out | a | ~b
}

/// Minimal XOR bug test
#[test]
fn test_xor_after_solve() {
    let mut sat = Solver::new();

    let a = sat.new_var(); // 0
    let b = sat.new_var(); // 1
    let out = sat.new_var(); // 2

    println!(
        "Vars: a={}, b={}, out={}",
        a.index(),
        b.index(),
        out.index()
    );

    encode_xor(&mut sat, out, a, b);

    // First solve: let the solver pick any values
    println!("\nFirst solve (unconstrained)...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    if r1 == SolverResult::Sat {
        let model = sat.model();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        let outv = model[out.index()].is_true();
        println!(
            "a={}, b={}, out={}, check: {} XOR {} = {}, match={}",
            av,
            bv,
            outv,
            av,
            bv,
            av ^ bv,
            (av ^ bv) == outv
        );
    }

    // Now add constraint: a = false
    println!("\nAdding a=false...");
    sat.add_clause([Lit::neg(a)]);

    println!("\nSecond solve (a=false)...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    if r2 == SolverResult::Sat {
        let model = sat.model();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        let outv = model[out.index()].is_true();
        println!(
            "a={}, b={}, out={}, check: {} XOR {} = {}, match={}",
            av,
            bv,
            outv,
            av,
            bv,
            av ^ bv,
            (av ^ bv) == outv
        );

        assert!(!av, "a should be false");
        assert_eq!(outv, bv, "out should equal b when a=false");
    }

    // Now add: b = false
    println!("\nAdding b=false...");
    sat.add_clause([Lit::neg(b)]);

    println!("\nThird solve (a=false, b=false)...");
    let r3 = sat.solve();
    println!("Result: {:?}", r3);
    if r3 == SolverResult::Sat {
        let model = sat.model();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        let outv = model[out.index()].is_true();
        println!(
            "a={}, b={}, out={}, check: {} XOR {} = {}, match={}",
            av,
            bv,
            outv,
            av,
            bv,
            av ^ bv,
            (av ^ bv) == outv
        );

        // a=false, b=false => out = false XOR false = false
        assert!(!av, "a should be false");
        assert!(!bv, "b should be false");
        assert!(!outv, "out should be false (false XOR false = false)");
    }
}

/// Two XORs in sequence (like in adder)
#[test]
fn test_two_xors_after_solve() {
    let mut sat = Solver::new();

    // First XOR: xor1 = a XOR b
    let a = sat.new_var();
    let b = sat.new_var();
    let xor1 = sat.new_var();
    encode_xor(&mut sat, xor1, a, b);

    // Second XOR: out = xor1 XOR c
    let c = sat.new_var();
    let out = sat.new_var();
    encode_xor(&mut sat, out, xor1, c);

    println!(
        "Vars: a={}, b={}, xor1={}, c={}, out={}",
        a.index(),
        b.index(),
        xor1.index(),
        c.index(),
        out.index()
    );

    // Constraint: out = true, c = false
    // This means xor1 = out XOR c = true XOR false = true
    // So either a=true,b=false or a=false,b=true
    sat.add_clause([Lit::pos(out)]);
    sat.add_clause([Lit::neg(c)]);

    println!("\nFirst solve (out=true, c=false)...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    if r1 == SolverResult::Sat {
        let model = sat.model();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        let xor1v = model[xor1.index()].is_true();
        let cv = model[c.index()].is_true();
        let outv = model[out.index()].is_true();
        println!("a={}, b={}, xor1={}, c={}, out={}", av, bv, xor1v, cv, outv);
        println!(
            "xor1 check: {} XOR {} = {}, match={}",
            av,
            bv,
            av ^ bv,
            (av ^ bv) == xor1v
        );
        println!(
            "out check: {} XOR {} = {}, match={}",
            xor1v,
            cv,
            xor1v ^ cv,
            (xor1v ^ cv) == outv
        );
    }

    // Now add: a = false
    println!("\nAdding a=false...");
    sat.add_clause([Lit::neg(a)]);

    println!("\nSecond solve (out=true, c=false, a=false)...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    if r2 == SolverResult::Sat {
        let model = sat.model();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        let xor1v = model[xor1.index()].is_true();
        let cv = model[c.index()].is_true();
        let outv = model[out.index()].is_true();
        println!("a={}, b={}, xor1={}, c={}, out={}", av, bv, xor1v, cv, outv);
        println!(
            "xor1 check: {} XOR {} = {}, match={}",
            av,
            bv,
            av ^ bv,
            (av ^ bv) == xor1v
        );
        println!(
            "out check: {} XOR {} = {}, match={}",
            xor1v,
            cv,
            xor1v ^ cv,
            (xor1v ^ cv) == outv
        );

        // a=false, out=true, c=false
        // => xor1 = out XOR c = true
        // => a XOR b = true, a=false => b=true
        assert!(!av, "a should be false");
        assert!(!cv, "c should be false");
        assert!(outv, "out should be true");
        assert!(xor1v, "xor1 should be true");
        assert!(bv, "b should be true (since a=false and xor1=true)");
    }

    // Now add: b = false  (this should make it UNSAT!)
    println!("\nAdding b=false...");
    sat.add_clause([Lit::neg(b)]);

    println!("\nThird solve (out=true, c=false, a=false, b=false)...");
    let r3 = sat.solve();
    println!("Result: {:?}", r3);
    // This should be UNSAT:
    // out=true, c=false => xor1=true
    // a=false, b=false => xor1=false (contradiction!)
    assert_eq!(r3, SolverResult::Unsat, "Should be UNSAT");
}
