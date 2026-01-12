//! Non-linear Integer Arithmetic (NIA) solver example.
//!
//! This example demonstrates how to:
//! - Create an NIA solver for integer constraints
//! - Mark variables as integers
//! - Solve non-linear integer problems with branch-and-bound

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::nia::{NiaConfig, NiaSolver, VarType};
use oxiz_nlsat::solver::SolverResult;
use oxiz_nlsat::types::AtomKind;

fn main() {
    println!("=== NIA Solver Example ===\n");
    println!("Solving integer constraint problem:");
    println!("  Find integers x, y such that:");
    println!("  x² + y² ≤ 50");
    println!("  x > 2");
    println!("  y > 2\n");

    // Create NIA solver with custom configuration
    let config = NiaConfig {
        max_nodes: 10000,
        max_depth: 100,
        enable_cutting_planes: true,
        branching_strategy: oxiz_nlsat::nia::BranchingStrategy::MostFractional,
        int_tolerance: 1e-6,
    };
    let mut solver = NiaSolver::with_config(config);

    // Declare arithmetic variables and mark them as integers
    let _x_var = solver.nlsat_mut().new_arith_var();
    let _y_var = solver.nlsat_mut().new_arith_var();

    // Mark variables as integer type
    solver.set_var_type(0, VarType::Integer);
    solver.set_var_type(1, VarType::Integer);

    println!("Variables: x (var 0), y (var 1)");
    println!("Both marked as INTEGER type\n");

    // Create polynomials
    let x = Polynomial::from_var(0);
    let y = Polynomial::from_var(1);
    let two = Polynomial::constant(BigRational::from_integer(BigInt::from(2)));
    let fifty = Polynomial::constant(BigRational::from_integer(BigInt::from(50)));

    // Constraint 1: x > 2
    let x_gt_2 = Polynomial::sub(&x, &two);
    let atom1 = solver.nlsat_mut().new_ineq_atom(x_gt_2, AtomKind::Gt);
    let lit1 = solver.nlsat().atom_literal(atom1, true);
    solver.nlsat_mut().add_clause(vec![lit1]);

    // Constraint 2: y > 2
    let y_gt_2 = Polynomial::sub(&y, &two);
    let atom2 = solver.nlsat_mut().new_ineq_atom(y_gt_2, AtomKind::Gt);
    let lit2 = solver.nlsat().atom_literal(atom2, true);
    solver.nlsat_mut().add_clause(vec![lit2]);

    // Constraint 3: x² + y² ≤ 50, i.e., x² + y² - 50 < 0
    let x_squared = Polynomial::mul(&x, &x);
    let y_squared = Polynomial::mul(&y, &y);
    let sum = Polynomial::add(&x_squared, &y_squared);
    let constraint = Polynomial::sub(&sum, &fifty);
    let atom3 = solver.nlsat_mut().new_ineq_atom(constraint, AtomKind::Lt);
    let lit3 = solver.nlsat().atom_literal(atom3, true);
    solver.nlsat_mut().add_clause(vec![lit3]);

    println!("Solving with branch-and-bound...");
    let result = solver.solve();
    println!("Result: {:?}\n", result);

    // Display statistics
    let stats = solver.stats();
    println!("NIA Statistics:");
    println!("  B&B nodes explored: {}", stats.nodes_explored);
    println!("  Cutting planes added: {}", stats.cutting_planes);
    println!("  Integer solutions found: {}", stats.integer_solutions);
    println!("  Max depth reached: {}", stats.max_depth_reached);

    // Extract and verify solution
    match result {
        SolverResult::Sat => {
            if let Some(model) = solver.nlsat().get_model() {
                if let (Some(x_val), Some(y_val)) = (model.arith_value(0), model.arith_value(1)) {
                    println!("\n✓ Integer solution found:");
                    println!("  x = {}", x_val);
                    println!("  y = {}", y_val);

                    // Check if values are actually integers
                    if let (Some(x_int), Some(y_int)) = (
                        x_val.to_f64().map(|f| f.round()),
                        y_val.to_f64().map(|f| f.round()),
                    ) {
                        println!("  x (rounded) = {}", x_int);
                        println!("  y (rounded) = {}", y_int);
                    }

                    // Verify constraint
                    let sum_of_squares =
                        x_val.clone() * x_val.clone() + y_val.clone() * y_val.clone();
                    let fifty_rat = BigRational::from_integer(BigInt::from(50));
                    println!("\nVerification:");
                    println!("  x² + y² = {}", sum_of_squares);
                    println!("  Is < 50? {}", sum_of_squares < fifty_rat);
                }
            }
        }
        SolverResult::Unsat => {
            println!("\n✗ No integer solution exists");
        }
        SolverResult::Unknown => {
            println!("\n? Solver could not determine satisfiability");
        }
    }

    println!("\n=== Example completed ===");
}
