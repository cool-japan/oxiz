//! Debug unit clause handling

use oxiz_sat::{Lit, Solver, SolverResult};

/// Simple XOR test: verify constraints work step by step
#[test]
fn test_xor_step_by_step() {
    let mut sat = Solver::new();

    let x = sat.new_var(); // 0 - represents XOR result
    let a = sat.new_var(); // 1
    let b = sat.new_var(); // 2

    println!("Vars: x={}, a={}, b={}", x.index(), a.index(), b.index());

    // x = a XOR b encoding:
    // ~x | ~a | ~b
    // ~x | a | b
    // x | ~a | b
    // x | a | ~b
    sat.add_clause([Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::pos(a), Lit::neg(b)]);

    // Before any solve - just test direct clause addition
    println!("\n=== Test: x=true, a=false should be SAT with b=true ===");

    sat.add_clause([Lit::pos(x)]); // x = true
    sat.add_clause([Lit::neg(a)]); // a = false

    let result = sat.solve();
    println!("Result: {:?}", result);

    if result == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("x={}, a={}, b={}", xv, av, bv);

        // Verify XOR: x = a XOR b
        let expected_x = av ^ bv;
        println!(
            "Verify: {} XOR {} = {}, x={}, match={}",
            av,
            bv,
            expected_x,
            xv,
            expected_x == xv
        );

        assert!(xv, "x should be true");
        assert!(!av, "a should be false");
        assert!(bv, "b should be true (false XOR true = true)");
    } else {
        panic!("Should be SAT!");
    }
}

/// Test that adding unit clauses AFTER solve works
#[test]
fn test_incremental_unit_clauses() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // x = a XOR b
    sat.add_clause([Lit::neg(x), Lit::neg(a), Lit::neg(b)]);
    sat.add_clause([Lit::neg(x), Lit::pos(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::neg(a), Lit::pos(b)]);
    sat.add_clause([Lit::pos(x), Lit::pos(a), Lit::neg(b)]);

    // First solve - unconstrained
    println!("\n=== First solve (unconstrained) ===");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    assert_eq!(r1, SolverResult::Sat);

    // Now add x = true
    println!("\n=== Adding x=true ===");
    sat.add_clause([Lit::pos(x)]);

    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    assert_eq!(r2, SolverResult::Sat);

    if r2 == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("x={}, a={}, b={}", xv, av, bv);

        // x must be true
        assert!(xv, "x should be true");
        // Exactly one of a, b must be true
        assert!(av ^ bv, "Exactly one of a, b must be true when x=true");
    }

    // Now add a = false
    println!("\n=== Adding a=false ===");
    sat.add_clause([Lit::neg(a)]);

    let r3 = sat.solve();
    println!("Result: {:?}", r3);

    // Should be SAT with x=true, a=false, b=true
    assert_eq!(
        r3,
        SolverResult::Sat,
        "Should be SAT with x=true, a=false, b=true"
    );

    if r3 == SolverResult::Sat {
        let model = sat.model();
        let xv = model[x.index()].is_true();
        let av = model[a.index()].is_true();
        let bv = model[b.index()].is_true();
        println!("x={}, a={}, b={}", xv, av, bv);

        assert!(xv, "x should be true");
        assert!(!av, "a should be false");
        assert!(bv, "b should be true (x=true, a=false => b=true)");
    }
}

/// Verify clause satisfaction after model extraction
#[test]
fn test_verify_clauses() {
    let mut sat = Solver::new();

    let x = sat.new_var();
    let a = sat.new_var();
    let b = sat.new_var();

    // Track clauses for verification
    let clauses = vec![
        vec![Lit::neg(x), Lit::neg(a), Lit::neg(b)], // ~x | ~a | ~b
        vec![Lit::neg(x), Lit::pos(a), Lit::pos(b)], // ~x | a | b  (KEY!)
        vec![Lit::pos(x), Lit::neg(a), Lit::pos(b)], // x | ~a | b
        vec![Lit::pos(x), Lit::pos(a), Lit::neg(b)], // x | a | ~b
    ];

    for c in &clauses {
        sat.add_clause(c.iter().copied());
    }

    sat.add_clause([Lit::pos(x)]); // x = true
    sat.add_clause([Lit::neg(a)]); // a = false

    let result = sat.solve();
    println!("Result: {:?}", result);

    if result == SolverResult::Sat {
        let model = sat.model();
        let lit_val = |l: Lit| -> bool {
            let val = model[l.var().index()].is_true();
            if l.is_pos() { val } else { !val }
        };

        println!("\nVerifying all clauses:");
        for (i, clause) in clauses.iter().enumerate() {
            let satisfied = clause.iter().any(|&l| lit_val(l));
            let lits_str: Vec<String> = clause
                .iter()
                .map(|&l| {
                    let name = match l.var().index() {
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
                .collect();
            println!(
                "  Clause {}: {:?} = {}",
                i,
                lits_str,
                if satisfied { "SAT" } else { "VIOLATED!" }
            );
            assert!(satisfied, "Clause {} should be satisfied", i);
        }

        // Also verify unit clauses
        println!("  Unit x=true: {}", lit_val(Lit::pos(x)));
        println!("  Unit a=false: {}", lit_val(Lit::neg(a)));
    }
}
