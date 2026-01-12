//! Portfolio solver example - parallel solving with multiple strategies.
//!
//! This example demonstrates the portfolio-based parallel solving approach.
//! Multiple solver instances run concurrently with different configurations,
//! and the first to find a solution wins.

use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::portfolio::{PortfolioConfig, PortfolioResult, PortfolioSolver};
use oxiz_nlsat::solver::NlsatSolver;
use oxiz_nlsat::types::AtomKind;
use std::time::Duration;

fn main() {
    println!("=== Portfolio Solver Example ===\n");
    println!("This example demonstrates parallel solving with diverse strategies.\n");

    println!("Problem:");
    println!("  x > 1");
    println!("  y > 1");
    println!("  x² - y = 0  (parabola)");
    println!("  x + y < 10\n");

    // Create base solver with the problem
    let mut base_solver = NlsatSolver::new();

    // Declare variables
    base_solver.new_arith_var(); // x = 0
    base_solver.new_arith_var(); // y = 1

    let x = Polynomial::from_var(0);
    let y = Polynomial::from_var(1);
    let one = Polynomial::constant(BigRational::from_integer(BigInt::from(1)));
    let ten = Polynomial::constant(BigRational::from_integer(BigInt::from(10)));

    // Constraint 1: x > 1
    let x_gt_1 = Polynomial::sub(&x, &one);
    let atom1 = base_solver.new_ineq_atom(x_gt_1, AtomKind::Gt);
    base_solver.add_clause(vec![base_solver.atom_literal(atom1, true)]);

    // Constraint 2: y > 1
    let y_gt_1 = Polynomial::sub(&y, &one);
    let atom2 = base_solver.new_ineq_atom(y_gt_1, AtomKind::Gt);
    base_solver.add_clause(vec![base_solver.atom_literal(atom2, true)]);

    // Constraint 3: x² - y = 0
    let x_squared = Polynomial::mul(&x, &x);
    let parabola = Polynomial::sub(&x_squared, &y);
    let atom3 = base_solver.new_ineq_atom(parabola, AtomKind::Eq);
    base_solver.add_clause(vec![base_solver.atom_literal(atom3, true)]);

    // Constraint 4: x + y < 10
    let sum = Polynomial::add(&x, &y);
    let sum_lt_10 = Polynomial::sub(&sum, &ten);
    let atom4 = base_solver.new_ineq_atom(sum_lt_10, AtomKind::Lt);
    base_solver.add_clause(vec![base_solver.atom_literal(atom4, true)]);

    // Create portfolio configuration
    let num_solvers = num_cpus::get().min(4);
    let config = PortfolioConfig {
        num_solvers,
        timeout: Some(Duration::from_secs(5)),
        enable_clause_sharing: true,
        max_shared_lbd: 8,
        share_interval: 1000,
    };

    println!("Portfolio configuration:");
    println!("  Solver threads: {}", config.num_solvers);
    println!("  Timeout: {:?}", config.timeout);
    println!("  Clause sharing: {}", config.enable_clause_sharing);
    println!("  Max shared LBD: {}", config.max_shared_lbd);
    println!("  Share interval: {} conflicts\n", config.share_interval);

    let mut portfolio = PortfolioSolver::new(config, base_solver);

    println!("Starting parallel solving...");
    println!("Each solver uses a different strategy:");
    println!("  - Solver 0: Aggressive geometric restarts");
    println!("  - Solver 1: Conservative Luby restarts");
    println!("  - Solver 2: Fixed interval restarts");
    println!("  - Solver 3+: Various ordering strategies\n");

    let start = std::time::Instant::now();
    let result = portfolio.solve();
    let elapsed = start.elapsed();

    println!("Solving completed in {:?}\n", elapsed);

    match result {
        PortfolioResult::Sat {
            solver_id,
            model: _,
        } => {
            println!("✓ SATISFIABLE");
            println!("  Winning solver: #{}", solver_id);
            println!("  A solution exists for the constraints.");
            println!("  (The parabola y = x² intersects the feasible region)");
        }
        PortfolioResult::Unsat { solver_id, core: _ } => {
            println!("✗ UNSATISFIABLE");
            println!("  Proven by solver: #{}", solver_id);
            println!("  No solution exists.");
        }
        PortfolioResult::Unknown => {
            println!("? UNKNOWN");
            println!("  Timeout or resource limit reached.");
        }
    }

    // Display statistics
    let stats = portfolio.stats();
    println!("\nPortfolio Statistics:");
    println!("  Solvers used: {}", stats.num_solvers);
    println!("  Winning solver: {:?}", stats.winning_solver);
    println!("  Clauses shared: {}", stats.total_shared_clauses);
    println!("  Total time: {:?}", stats.total_time);

    if !stats.conflicts_per_solver.is_empty() {
        println!("\n  Conflicts per solver:");
        for (i, conflicts) in stats.conflicts_per_solver.iter().enumerate() {
            println!("    Solver {}: {} conflicts", i, conflicts);
        }
    }

    println!("\nKey benefits of portfolio solving:");
    println!("  • Diverse strategies explore different search spaces");
    println!("  • First solver to finish terminates all others");
    println!("  • Clause sharing improves learning across solvers");
    println!("  • Parallelizes computational resources effectively");

    println!("\n=== Example completed ===");
}
