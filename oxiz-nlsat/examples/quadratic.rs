//! Solving quadratic constraints with the NLSAT solver.
//!
//! This example shows how to solve: x^2 - 2 = 0
//! Which has solutions x = ±√2

use num_rational::BigRational;
use num_traits::Signed;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::{NlsatSolver, types::AtomKind};

fn main() {
    println!("=== Quadratic Equation Solver ===\n");

    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    // Create x^2 - 2 = 0
    // We need: x * x - 2
    println!("Solving: x² - 2 = 0");

    let x = Polynomial::from_var(0);
    let x_squared = Polynomial::mul(&x, &x);
    let two = Polynomial::constant(BigRational::from_integer(2.into()));
    let poly = Polynomial::sub(&x_squared, &two);

    let atom_id = solver.new_ineq_atom(poly, AtomKind::Eq);
    solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

    // Solve
    println!("Solving...");
    let result = solver.solve();
    println!("Result: {:?}\n", result);

    // Extract model
    if let Some(model) = solver.get_model() {
        if let Some(x_val) = model.arith_value(0) {
            println!("Solution: x ≈ {}", x_val);

            // Verify: x² should equal 2
            let x_squared_val = x_val * x_val;
            let diff = (x_squared_val - BigRational::from_integer(2.into())).abs();
            println!("Verification: x² = {}", x_val * x_val);
            println!("Error: {}", diff);

            if diff < BigRational::from_integer(1.into()) / BigRational::from_integer(1000.into()) {
                println!("✓ Solution verified (within tolerance)");
            }
        }
    }

    println!("\n=== Example completed ===");
}
