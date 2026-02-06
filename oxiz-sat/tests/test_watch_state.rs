//! Debug watch state

use oxiz_sat::{Lit, Solver, SolverResult};

/// Check if the issue is with model not being updated
#[test]
fn test_model_update() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // Simple: a AND b
    sat.add_clause([Lit::pos(a)]); // a = true
    sat.add_clause([Lit::pos(b)]); // b = true

    let r1 = sat.solve();
    println!("Solve 1: {:?}", r1);
    let m1 = sat.model();
    println!(
        "Model 1: a={}, b={}",
        m1[a.index()].is_true(),
        m1[b.index()].is_true()
    );

    // Now add x as just a free variable, solve again
    sat.add_clause([Lit::pos(x)]);

    let r2 = sat.solve();
    println!("Solve 2: {:?}", r2);
    let m2 = sat.model();
    println!(
        "Model 2: x={}, a={}, b={}",
        m2[x.index()].is_true(),
        m2[a.index()].is_true(),
        m2[b.index()].is_true()
    );

    assert!(m2[x.index()].is_true());
    assert!(m2[a.index()].is_true());
    assert!(m2[b.index()].is_true());
}

/// Minimal XOR incremental test
#[test]
fn test_minimal_xor_incremental() {
    // Test: after first solve, add one unit clause and solve again
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // Single XOR clause that will be violated: ~x | a | b
    // This means: if x=T, then a OR b must be true
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);

    println!("=== Clause: ~x | a | b ===");
    println!("Semantics: if x is true, then a or b must be true");

    // First solve - unconstrained
    println!("\n--- First solve ---");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    if r1 == SolverResult::Sat {
        let m = sat.model();
        println!(
            "Model: x={}, a={}, b={}",
            m[x.index()].is_true(),
            m[a.index()].is_true(),
            m[b.index()].is_true()
        );
    }

    // Add x = true
    println!("\n--- Adding x=true ---");
    sat.add_clause([Lit::pos(x)]);

    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    if r2 == SolverResult::Sat {
        let m = sat.model();
        let xv = m[x.index()].is_true();
        let av = m[a.index()].is_true();
        let bv = m[b.index()].is_true();
        println!("Model: x={}, a={}, b={}", xv, av, bv);

        // Check clause: ~x | a | b = !x || a || b
        let satisfied = !xv || av || bv;
        println!("Clause (~x|a|b) satisfied: {}", satisfied);

        assert!(satisfied, "Clause must be satisfied!");
    }

    // Add a = false
    println!("\n--- Adding a=false ---");
    sat.add_clause([Lit::neg(a)]);

    let r3 = sat.solve();
    println!("Result: {:?}", r3);

    // Should be SAT with b=true
    assert_eq!(r3, SolverResult::Sat);
    if r3 == SolverResult::Sat {
        let m = sat.model();
        let xv = m[x.index()].is_true();
        let av = m[a.index()].is_true();
        let bv = m[b.index()].is_true();
        println!("Model: x={}, a={}, b={}", xv, av, bv);

        assert!(xv, "x should be true");
        assert!(!av, "a should be false");
        assert!(bv, "b should be true");
    }
}

/// Test with just ~x | a | b and unit constraints before ANY solve
#[test]
fn test_single_clause_fresh() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // ~x | a | b with x=T, a=F → b must be T
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x)]);
    sat.add_clause([Lit::neg(a)]);

    let result = sat.solve();
    println!("Fresh result: {:?}", result);

    assert_eq!(result, SolverResult::Sat);
    let m = sat.model();
    let xv = m[x.index()].is_true();
    let av = m[a.index()].is_true();
    let bv = m[b.index()].is_true();
    println!("Model: x={}, a={}, b={}", xv, av, bv);

    // ~x | a | b = F | F | b = b, so b must be true
    assert!(bv, "b must be true");
}

/// Simplest possible incremental test
#[test]
fn test_simplest_incremental() {
    let mut sat = Solver::new();

    let x = sat.new_var();

    // Just x
    sat.add_clause([Lit::pos(x)]);

    let r1 = sat.solve();
    println!("Solve 1: {:?}", r1);
    assert_eq!(r1, SolverResult::Sat);
    let m1 = sat.model();
    println!("Model 1: x={}", m1[x.index()].is_true());
    assert!(m1[x.index()].is_true());

    // Add ~x - should be UNSAT
    sat.add_clause([Lit::neg(x)]);

    let r2 = sat.solve();
    println!("Solve 2: {:?}", r2);
    assert_eq!(r2, SolverResult::Unsat);
}
