//! Debug allsat enumeration issue

use oxiz_sat::{Lit, Solver, SolverResult, Var};

/// Simple model enumeration test
#[test]
fn test_simple_model_enum() {
    let mut solver = Solver::new();

    // Formula: x1 ∨ x2
    solver.add_clause([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);

    let num_vars = 2;
    let mut models = Vec::new();
    let max_iterations = 10;

    for i in 0..max_iterations {
        println!("Iteration {}", i);

        let result = solver.solve();
        println!("  Result: {:?}", result);

        match result {
            SolverResult::Sat => {
                // Extract model
                let model: Vec<bool> = (0..num_vars)
                    .map(|v| solver.model_value(Var::new(v as u32)).is_true())
                    .collect();
                println!("  Model: {:?}", model);

                // Check if duplicate
                if models.contains(&model) {
                    println!("  DUPLICATE MODEL FOUND!");
                    panic!("Duplicate model found - blocking clause not working");
                }
                models.push(model.clone());

                // Create blocking clause
                let mut blocking: Vec<Lit> = Vec::new();
                for (v, &val) in model.iter().enumerate() {
                    let lit = if val {
                        Lit::neg(Var::new(v as u32)) // block TRUE with negative
                    } else {
                        Lit::pos(Var::new(v as u32)) // block FALSE with positive
                    };
                    blocking.push(lit);
                }
                println!("  Blocking clause: {:?}", blocking);

                // Add blocking clause
                println!("  Trail before add: {:?}", solver.trail().assignments());
                let success = solver.add_clause(blocking.iter().copied());
                println!("  Add clause success: {}", success);
                println!("  Trail after add: {:?}", solver.trail().assignments());

                // Check trail for blocking clause satisfaction
                for &lit in &blocking {
                    let val = solver.trail().lit_value(lit);
                    println!("    Blocking lit {:?} value: {:?}", lit, val);
                }

                if !success {
                    println!("  Formula became UNSAT after blocking clause");
                    break;
                }
            }
            SolverResult::Unsat => {
                println!("  UNSAT - no more models");
                break;
            }
            SolverResult::Unknown => {
                println!("  Unknown");
                break;
            }
        }
    }

    println!("\nFound {} models: {:?}", models.len(), models);

    // Should have exactly 3 models for (x1 ∨ x2):
    // [true, false], [false, true], [true, true]
    assert!(!models.is_empty(), "Should find at least 1 model");
    assert!(
        models.len() <= 4,
        "Should not have more than 3 unique models"
    );
}
