//! Basic usage of the NLSAT solver for simple constraints.
//!
//! This example demonstrates how to:
//! - Create a solver instance
//! - Define variables
//! - Add polynomial constraints
//! - Solve and extract models

use num_traits::Signed;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::{NlsatSolver, types::AtomKind};

fn main() {
    println!("=== Basic NLSAT Solver Example ===\n");

    // Create a new solver
    let mut solver = NlsatSolver::new();

    // Create an arithmetic variable x
    let _x = solver.new_arith_var();

    // Add constraint: x > 0
    println!("Adding constraint: x > 0");
    let poly = Polynomial::from_var(0);
    let atom_id = solver.new_ineq_atom(poly, AtomKind::Gt);
    let lit = solver.atom_literal(atom_id, true);
    solver.add_clause(vec![lit]);

    // Solve
    println!("Solving...");
    let result = solver.solve();
    println!("Result: {:?}\n", result);

    // Extract model if SAT
    if let Some(model) = solver.get_model() {
        if let Some(x_val) = model.arith_value(0) {
            println!("Solution: x = {}", x_val);
            println!("Verified: x > 0? {}", x_val.is_positive());
        }
    }

    println!("\n=== Example completed ===");
}
