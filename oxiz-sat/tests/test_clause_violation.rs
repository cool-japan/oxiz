//! Minimal test for clause violation

use oxiz_sat::{Lit, Solver, SolverResult};

/// Minimal case: XOR clauses + constraints that should be UNSAT
/// but solver returns SAT
#[test]
fn test_minimal_clause_violation() {
    let mut sat = Solver::new();

    let xor12 = sat.new_var(); // 0
    let a1 = sat.new_var(); // 1
    let b1 = sat.new_var(); // 2

    println!(
        "Vars: xor12={}, a1={}, b1={}",
        xor12.index(),
        a1.index(),
        b1.index()
    );

    // XOR encoding: xor12 = a1 XOR b1
    sat.add_clause([Lit::neg(xor12), Lit::neg(a1), Lit::neg(b1)]); // ~xor12 | ~a1 | ~b1
    sat.add_clause([Lit::neg(xor12), Lit::pos(a1), Lit::pos(b1)]); // ~xor12 | a1 | b1 ← KEY CLAUSE
    sat.add_clause([Lit::pos(xor12), Lit::neg(a1), Lit::pos(b1)]); // xor12 | ~a1 | b1
    sat.add_clause([Lit::pos(xor12), Lit::pos(a1), Lit::neg(b1)]); // xor12 | a1 | ~b1

    // Now add constraints that should make it UNSAT:
    // xor12 = true, a1 = false, b1 = false
    // This violates: ~xor12 | a1 | b1 = false | false | false = false

    sat.add_clause([Lit::pos(xor12)]); // xor12 = true
    sat.add_clause([Lit::neg(a1)]); // a1 = false
    sat.add_clause([Lit::neg(b1)]); // b1 = false

    println!("Solving...");
    let result = sat.solve();
    println!("Result: {:?}", result);

    // Should be UNSAT!
    assert_eq!(
        result,
        SolverResult::Unsat,
        "Should be UNSAT - clause ~xor12|a1|b1 is violated"
    );
}

/// Same test but add constraints in different order
#[test]
fn test_clause_violation_order2() {
    let mut sat = Solver::new();

    let xor12 = sat.new_var();
    let a1 = sat.new_var();
    let b1 = sat.new_var();

    // Add unit constraints first
    sat.add_clause([Lit::pos(xor12)]);
    sat.add_clause([Lit::neg(a1)]);
    sat.add_clause([Lit::neg(b1)]);

    // Then add XOR clauses
    sat.add_clause([Lit::neg(xor12), Lit::neg(a1), Lit::neg(b1)]);
    sat.add_clause([Lit::neg(xor12), Lit::pos(a1), Lit::pos(b1)]);
    sat.add_clause([Lit::pos(xor12), Lit::neg(a1), Lit::pos(b1)]);
    sat.add_clause([Lit::pos(xor12), Lit::pos(a1), Lit::neg(b1)]);

    let result = sat.solve();
    println!("Order 2 result: {:?}", result);
    assert_eq!(result, SolverResult::Unsat, "Should be UNSAT");
}

/// Test with solve() between
#[test]
fn test_clause_violation_with_solve() {
    let mut sat = Solver::new();

    let xor12 = sat.new_var();
    let a1 = sat.new_var();
    let b1 = sat.new_var();

    // XOR encoding
    sat.add_clause([Lit::neg(xor12), Lit::neg(a1), Lit::neg(b1)]);
    sat.add_clause([Lit::neg(xor12), Lit::pos(a1), Lit::pos(b1)]);
    sat.add_clause([Lit::pos(xor12), Lit::neg(a1), Lit::pos(b1)]);
    sat.add_clause([Lit::pos(xor12), Lit::pos(a1), Lit::neg(b1)]);

    println!("\nFirst solve (unconstrained XOR)...");
    let r1 = sat.solve();
    println!("Result: {:?}", r1);
    if r1 == SolverResult::Sat {
        let model = sat.model();
        println!(
            "xor12={}, a1={}, b1={}",
            model[xor12.index()].is_true(),
            model[a1.index()].is_true(),
            model[b1.index()].is_true()
        );
    }

    // Now add xor12 = true
    println!("\nAdding xor12=true...");
    sat.add_clause([Lit::pos(xor12)]);

    println!("\nSecond solve...");
    let r2 = sat.solve();
    println!("Result: {:?}", r2);
    if r2 == SolverResult::Sat {
        let model = sat.model();
        println!(
            "xor12={}, a1={}, b1={}",
            model[xor12.index()].is_true(),
            model[a1.index()].is_true(),
            model[b1.index()].is_true()
        );
    }

    // Now add a1 = false
    println!("\nAdding a1=false...");
    sat.add_clause([Lit::neg(a1)]);

    println!("\nThird solve...");
    let r3 = sat.solve();
    println!("Result: {:?}", r3);
    if r3 == SolverResult::Sat {
        let model = sat.model();
        println!(
            "xor12={}, a1={}, b1={}",
            model[xor12.index()].is_true(),
            model[a1.index()].is_true(),
            model[b1.index()].is_true()
        );
        // xor12=true, a1=false => b1 must be true
    }

    // Now add b1 = false (this should make it UNSAT)
    println!("\nAdding b1=false...");
    sat.add_clause([Lit::neg(b1)]);

    println!("\nFourth solve...");
    let r4 = sat.solve();
    println!("Result: {:?}", r4);

    assert_eq!(
        r4,
        SolverResult::Unsat,
        "Should be UNSAT after adding b1=false"
    );
}
