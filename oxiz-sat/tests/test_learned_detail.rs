//! Detailed debug of what's being learned

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]);
}

/// Debug what's happening step by step
#[test]
fn test_learned_step_by_step() {
    let mut sat = Solver::new();
    let a = sat.new_var();
    let b = sat.new_var();
    let out = sat.new_var();

    println!(
        "Vars: a={}, b={}, out={}",
        a.index(),
        b.index(),
        out.index()
    );

    encode_xor(&mut sat, out, a, b);

    // First solve
    println!("\n=== First solve ===");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    if r1 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: a={}, b={}, out={}",
            m[a.index()].is_true(),
            m[b.index()].is_true(),
            m[out.index()].is_true()
        );
    }
    println!("Trail after solve: {:?}", sat.trail().assignments());
    println!(
        "Stats: decisions={}, conflicts={}, learned={}",
        sat.stats().decisions,
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );

    // Add a=F
    println!("\n=== Add a=F ===");
    sat.add_clause([Lit::neg(a)]);
    println!("Trail after add: {:?}", sat.trail().assignments());

    // Second solve
    println!("\n=== Second solve ===");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    if r2 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: a={}, b={}, out={}",
            m[a.index()].is_true(),
            m[b.index()].is_true(),
            m[out.index()].is_true()
        );
    }
    println!("Trail after solve: {:?}", sat.trail().assignments());
    println!(
        "Stats: decisions={}, conflicts={}, learned={}",
        sat.stats().decisions,
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );

    // Trail values at this point
    let trail = sat.trail();
    for &lit in trail.assignments() {
        let var = lit.var();
        let level = trail.level(var);
        let value = if lit.is_pos() { "T" } else { "F" };
        let name = match var.index() {
            0 => "a",
            1 => "b",
            2 => "out",
            _ => "?",
        };
        println!("  {}={} at level {}", name, value, level);
    }

    // Add b=F
    println!("\n=== Add b=F ===");
    sat.add_clause([Lit::neg(b)]);
    println!("Trail after add: {:?}", sat.trail().assignments());

    // Check trail values
    let trail = sat.trail();
    for &lit in trail.assignments() {
        let var = lit.var();
        let level = trail.level(var);
        let value = if lit.is_pos() { "T" } else { "F" };
        let name = match var.index() {
            0 => "a",
            1 => "b",
            2 => "out",
            _ => "?",
        };
        println!("  {}={} at level {}", name, value, level);
    }

    // Third solve
    println!("\n=== Third solve ===");
    let r3 = sat.solve();
    println!("Result: {:?}", r3);
    println!(
        "Stats: decisions={}, conflicts={}, learned={}",
        sat.stats().decisions,
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );

    if r3 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: a={}, b={}, out={}",
            m[a.index()].is_true(),
            m[b.index()].is_true(),
            m[out.index()].is_true()
        );
    } else if r3 == SolverResult::Unsat {
        println!(
            "UNSAT! But should be SAT with a=F, b=F, out=F (since out = a XOR b = F XOR F = F)"
        );
    }

    // Verify with fresh solver
    println!("\n=== Fresh solver verification ===");
    {
        let mut fresh = Solver::new();
        let a2 = fresh.new_var();
        let b2 = fresh.new_var();
        let out2 = fresh.new_var();
        encode_xor(&mut fresh, out2, a2, b2);
        fresh.add_clause([Lit::neg(a2)]);
        fresh.add_clause([Lit::neg(b2)]);
        let rf = fresh.solve();
        println!("Fresh result: {:?}", rf);
        if rf == SolverResult::Sat {
            let m = fresh.model();
            println!(
                "Fresh model: a={}, b={}, out={}",
                m[a2.index()].is_true(),
                m[b2.index()].is_true(),
                m[out2.index()].is_true()
            );
        }
    }
}
