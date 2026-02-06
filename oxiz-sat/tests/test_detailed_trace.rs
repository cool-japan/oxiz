//! Very detailed trace of incremental solving

use oxiz_sat::{Lit, Solver, SolverResult};

/// Trace with fresh solver - no incremental
#[test]
fn test_fresh_solver() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // x = a XOR b
    sat.add_clause([Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::pos(a), Lit::neg(b)]);

    // x=true, a=false
    sat.add_clause([Lit::pos(x)]);
    sat.add_clause([Lit::neg(a)]);

    let result = sat.solve();
    println!("Fresh solver result: {:?}", result);

    assert_eq!(result, SolverResult::Sat);
    let model = sat.model();
    let xv = model[x.index()].is_true();
    let av = model[a.index()].is_true();
    let bv = model[b.index()].is_true();
    println!("Model: x={}, a={}, b={}", xv, av, bv);

    assert!(xv, "x should be true");
    assert!(!av, "a should be false");
    assert!(bv, "b should be true");
}

/// Compare with incremental
#[test]
fn test_incremental_comparison() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // x = a XOR b
    sat.add_clause([Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::pos(a), Lit::neg(b)]);

    // First: just XOR
    println!("=== After XOR clauses only ===");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);

    if r1 == SolverResult::Sat {
        let model = sat.model();
        println!(
            "Model: x={}, a={}, b={}",
            model[x.index()].is_true(),
            model[a.index()].is_true(),
            model[b.index()].is_true()
        );
    }

    // Add x=true
    println!("\n=== After adding x=true ===");
    sat.add_clause([Lit::pos(x)]);
    let r2 = sat.solve();
    println!("Result: {:?}", r2);

    if r2 == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("Model: x={}, a={}, b={}", xv, av, bv);

        // Check clause C1: ~x | a | b
        let c1 = !xv || av || bv;
        println!("C1 (~x|a|b) satisfied: {}", c1);

        if !c1 {
            println!("ERROR: C1 violated! x={}, a={}, b={}", xv, av, bv);
        }
    }

    // Add a=false
    println!("\n=== After adding a=false ===");
    sat.add_clause([Lit::neg(a)]);
    let r3 = sat.solve();
    println!("Result: {:?}", r3);

    if r3 == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("Model: x={}, a={}, b={}", xv, av, bv);

        assert!(xv, "x should be true");
        assert!(!av, "a should be false");
        assert!(bv, "b should be true");
    }
}

/// Test model freshness - is the model being updated?
#[test]
fn test_model_freshness() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // x = a XOR b
    sat.add_clause([Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::pos(a), Lit::neg(b)]);

    let r1 = sat.solve();
    assert_eq!(r1, SolverResult::Sat);
    let m1 = sat.model().to_vec();
    println!(
        "After solve 1: x={}, a={}, b={}",
        m1[x.index()].is_true(),
        m1[a.index()].is_true(),
        m1[b.index()].is_true()
    );

    // Add x=true
    sat.add_clause([Lit::pos(x)]);

    // Check model BEFORE solve - should still be old
    let m_before = sat.model().to_vec();
    println!(
        "Before solve 2: x={}, a={}, b={}",
        m_before[x.index()].is_true(),
        m_before[a.index()].is_true(),
        m_before[b.index()].is_true()
    );

    let r2 = sat.solve();
    assert_eq!(r2, SolverResult::Sat);
    let m2 = sat.model().to_vec();
    println!(
        "After solve 2: x={}, a={}, b={}",
        m2[x.index()].is_true(),
        m2[a.index()].is_true(),
        m2[b.index()].is_true()
    );

    // Check C1 satisfaction
    let xv = m2[x.index()].is_true();
    let av = m2[a.index()].is_true();
    let bv = m2[b.index()].is_true();

    // C1: ~x | a | b
    let c1 = !xv || av || bv;
    println!("C1 (~x|a|b) = !{} || {} || {} = {}", xv, av, bv, c1);

    assert!(c1, "C1 must be satisfied!");
}
