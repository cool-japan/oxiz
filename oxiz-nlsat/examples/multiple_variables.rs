//! Multi-variable constraints with NLSAT.
//!
//! This example demonstrates solving constraints with multiple variables:
//! x > 0 ∧ y > 0 ∧ x + y < 10

use num_rational::BigRational;
use num_traits::Signed;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::{NlsatSolver, types::AtomKind};

fn main() {
    println!("=== Multi-Variable Constraint Solver ===\n");

    let mut solver = NlsatSolver::new();

    // Create variables
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    println!("Solving system:");
    println!("  x > 0");
    println!("  y > 0");
    println!("  x + y < 10\n");

    // Constraint 1: x > 0
    let x_poly = Polynomial::from_var(0);
    let atom1 = solver.new_ineq_atom(x_poly.clone(), AtomKind::Gt);
    solver.add_clause(vec![solver.atom_literal(atom1, true)]);

    // Constraint 2: y > 0
    let y_poly = Polynomial::from_var(1);
    let atom2 = solver.new_ineq_atom(y_poly.clone(), AtomKind::Gt);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);

    // Constraint 3: x + y - 10 < 0 (i.e., x + y < 10)
    let x_plus_y = Polynomial::add(&x_poly, &y_poly);
    let ten = Polynomial::constant(BigRational::from_integer(10.into()));
    let x_plus_y_minus_10 = Polynomial::sub(&x_plus_y, &ten);
    let atom3 = solver.new_ineq_atom(x_plus_y_minus_10, AtomKind::Lt);
    solver.add_clause(vec![solver.atom_literal(atom3, true)]);

    // Solve
    println!("Solving...");
    let result = solver.solve();
    println!("Result: {:?}\n", result);

    // Extract and verify solution
    if let Some(model) = solver.get_model() {
        if let (Some(x_val), Some(y_val)) = (model.arith_value(0), model.arith_value(1)) {
            println!("Solution:");
            println!("  x = {}", x_val);
            println!("  y = {}", y_val);
            println!("  x + y = {}\n", x_val + y_val);

            // Verify constraints
            println!("Verification:");
            println!("  x > 0? {}", x_val.is_positive());
            println!("  y > 0? {}", y_val.is_positive());
            println!(
                "  x + y < 10? {}",
                (x_val + y_val) < BigRational::from_integer(10.into())
            );
        }
    }

    println!("\n=== Example completed ===");
}
