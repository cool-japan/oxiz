//! Examine trail state during incremental solving

use oxiz_sat::{Lit, Solver, SolverResult};

fn print_trail(sat: &Solver, x: oxiz_sat::Var, a: oxiz_sat::Var, b: oxiz_sat::Var) {
    let trail = sat.trail();
    println!(
        "  Trail assignments: {:?}",
        trail
            .assignments()
            .iter()
            .map(|&l| {
                let var = l.var();
                let name = match var.index() {
                    0 => "x",
                    1 => "a",
                    2 => "b",
                    _ => "?",
                };
                if l.is_pos() {
                    name.to_string()
                } else {
                    format!("~{}", name)
                }
            })
            .collect::<Vec<_>>()
    );
    println!("  Decision level: {}", sat.decision_level());

    // Print values from trail
    let xv = trail.value(x);
    let av = trail.value(a);
    let bv = trail.value(b);
    println!("  Trail values: x={:?}, a={:?}, b={:?}", xv, av, bv);

    // Print levels
    let xl = trail.level(x);
    let al = trail.level(a);
    let bl = trail.level(b);
    println!("  Trail levels: x@{}, a@{}, b@{}", xl, al, bl);
}

/// Trace the incremental solving issue
#[test]
fn test_trace_trail_state() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    println!(
        "Variables: x={}, a={}, b={}",
        x.index(),
        a.index(),
        b.index()
    );

    // Clause: ~x | a | b
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    println!("\nClause added: ~x | a | b");

    // First solve
    println!("\n=== First solve ===");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    print_trail(&sat, x, a, b);

    // Model
    if r1 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "  Model: x={}, a={}, b={}",
            m[x.index()].is_true(),
            m[a.index()].is_true(),
            m[b.index()].is_true()
        );
    }

    // Add x=true
    println!("\n=== Adding x=true ===");
    println!("Before add_clause:");
    print_trail(&sat, x, a, b);

    sat.add_clause([Lit::pos(x)]);
    println!("After add_clause:");
    print_trail(&sat, x, a, b);

    // Second solve
    println!("\n=== Second solve ===");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    print_trail(&sat, x, a, b);

    if r2 == SolverResult::Sat {
        let m = sat.model();
        let xv = m[x.index()].is_true();
        let av = m[a.index()].is_true();
        let bv = m[b.index()].is_true();
        println!("  Model: x={}, a={}, b={}", xv, av, bv);

        // Check clause satisfaction
        let sat_clause = !xv || av || bv;
        println!("  Clause (~x|a|b) satisfied: {}", sat_clause);
        assert!(sat_clause, "Clause must be satisfied!");
    }
}

/// Check what happens to trail after backtrack
#[test]
fn test_backtrack_state() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();

    sat.add_clause([Lit::neg(x), Lit::pos(a)]); // ~x | a (x implies a)

    // First solve
    let r1 = sat.solve();
    println!("First solve: {:?}", r1);
    println!("  Trail: {:?}", sat.trail().assignments());
    println!("  Level: {}", sat.decision_level());

    // Backtrack to root
    sat.backtrack_to_root();
    println!("\nAfter backtrack_to_root:");
    println!("  Trail: {:?}", sat.trail().assignments());
    println!("  Level: {}", sat.decision_level());

    // Add x=true
    sat.add_clause([Lit::pos(x)]);
    println!("\nAfter add_clause([x]):");
    println!("  Trail: {:?}", sat.trail().assignments());

    // Second solve
    let r2 = sat.solve();
    println!("\nSecond solve: {:?}", r2);
    println!("  Trail: {:?}", sat.trail().assignments());

    if r2 == SolverResult::Sat {
        let m = sat.model();
        let xv = m[x.index()].is_true();
        let av = m[a.index()].is_true();
        println!("  Model: x={}, a={}", xv, av);

        // x=T should imply a=T
        assert!(xv, "x should be true");
        assert!(av, "a should be true (implied by x)");
    }
}
