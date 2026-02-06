//! Debug what clause is being learned

use oxiz_sat::{Lit, Solver, SolverResult};

fn encode_xor(sat: &mut Solver, out: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    sat.add_clause([Lit::neg(out), Lit::neg(a), Lit::neg(b)]); // C0
    sat.add_clause([Lit::neg(out), Lit::pos(a), Lit::pos(b)]); // C1
    sat.add_clause([Lit::pos(out), Lit::neg(a), Lit::pos(b)]); // C2
    sat.add_clause([Lit::pos(out), Lit::pos(a), Lit::neg(b)]); // C3
}

/// Debug test to understand the conflict
#[test]
fn test_debug_learned_clause() {
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
    println!(
        "Stats: decisions={}, conflicts={}, learned={}",
        sat.stats().decisions,
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );

    // Add a=F
    println!("\n=== Adding a=F ===");
    sat.add_clause([Lit::neg(a)]);

    // Print trail after adding
    println!("Trail after add:");
    for &lit in sat.trail().assignments() {
        let var = lit.var();
        let level = sat.trail().level(var);
        let val = if lit.is_pos() { "T" } else { "F" };
        let name = match var.index() {
            0 => "a",
            1 => "b",
            2 => "out",
            _ => "?",
        };
        println!("  {} = {} at level {}", name, val, level);
    }

    // Second solve
    println!("\n=== Second solve ===");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    println!(
        "Stats: decisions={}, conflicts={}, learned={}",
        sat.stats().decisions,
        sat.stats().conflicts,
        sat.stats().learned_clauses
    );

    // Print trail
    println!("Trail after solve:");
    for &lit in sat.trail().assignments() {
        let var = lit.var();
        let level = sat.trail().level(var);
        let val = if lit.is_pos() { "T" } else { "F" };
        let name = match var.index() {
            0 => "a",
            1 => "b",
            2 => "out",
            _ => "?",
        };
        println!("  {} = {} at level {}", name, val, level);
    }

    // Print learned clauses
    println!("\n=== Learned clauses after second solve ===");
    sat.debug_print_learned_clauses();
    sat.debug_print_binary_graph();
    println!("Total learned: {}", sat.stats().learned_clauses);
    println!("Binary clauses: {}", sat.stats().binary_clauses);

    // Add b=F
    println!("\n=== Adding b=F ===");
    sat.add_clause([Lit::neg(b)]);

    println!("Trail after add:");
    for &lit in sat.trail().assignments() {
        let var = lit.var();
        let level = sat.trail().level(var);
        let val = if lit.is_pos() { "T" } else { "F" };
        let name = match var.index() {
            0 => "a",
            1 => "b",
            2 => "out",
            _ => "?",
        };
        println!("  {} = {} at level {}", name, val, level);
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

    if r3 == SolverResult::Unsat {
        println!("BUG: Should be SAT with a=F, b=F, out=F!");
    } else {
        let m = sat.model();
        println!(
            "Model: a={}, b={}, out={}",
            m[a.index()].is_true(),
            m[b.index()].is_true(),
            m[out.index()].is_true()
        );
    }

    // Fresh solver comparison
    println!("\n=== Fresh solver ===");
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

    assert_eq!(r3, SolverResult::Sat, "XOR + a=F + b=F should be SAT");
}
