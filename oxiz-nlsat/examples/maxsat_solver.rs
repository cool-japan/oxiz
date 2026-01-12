//! Maximum Satisfiability (MaxSAT) solver conceptual example.
//!
//! This example demonstrates the conceptual usage of the MaxSAT solver.
//! The MaxSAT solver finds optimal solutions by minimizing the weight
//! of violated soft constraints while satisfying all hard constraints.
//!
//! Note: This is a conceptual example showing the API structure.
//! For practical polynomial constraint solving, integrate MaxSAT
//! with the NLSAT backend through the proper initialization methods.

use oxiz_nlsat::maxsat::{MaxSatConfig, MaxSatResult, MaxSatSolver};
use oxiz_nlsat::types::Literal;

fn main() {
    println!("=== MaxSAT Solver Conceptual Example ===\n");
    println!("This example demonstrates the MaxSAT solver API structure.\n");

    println!("Concept:");
    println!("  Hard constraints: MUST be satisfied");
    println!("  Soft constraints: Preferences with weights");
    println!("  Goal: Minimize total weight of violated soft constraints\n");

    // Create MaxSAT solver with configuration
    let config = MaxSatConfig {
        max_iterations: 100,
        minimize_cores: true,
        stratify: false,
    };
    let max_iterations = config.max_iterations;
    let minimize_cores = config.minimize_cores;
    let stratify = config.stratify;

    let mut solver = MaxSatSolver::new(config);

    println!("Configuration:");
    println!("  Max iterations: {}", max_iterations);
    println!("  Core minimization: {}", minimize_cores);
    println!("  Stratification: {}\n", stratify);

    // Example literals (in a real scenario, these would be created from atoms)
    let lit1 = Literal::positive(1);
    let lit2 = Literal::positive(2);
    let lit3 = Literal::positive(3);

    // Add hard constraints (must be satisfied)
    println!("Adding hard constraints...");
    solver.add_hard(vec![lit1]);
    solver.add_hard(vec![lit2, lit3]); // Clause: lit2 OR lit3

    // Add soft constraints (preferences with weights)
    println!("Adding soft constraints...");
    solver.add_soft(vec![lit2], 5); // Weight 5 - strong preference
    solver.add_soft(vec![lit3], 2); // Weight 2 - weak preference

    println!("\nSolving MaxSAT problem...");
    let result = solver.solve();

    println!("\nResult:");
    match result {
        MaxSatResult::Optimal { cost, model } => {
            println!("  ✓ OPTIMAL");
            println!("  Cost (violated weight): {}", cost);
            println!("  Boolean assignments: {} literals", model.len());
            println!("\n  Interpretation:");
            println!("    - All hard constraints satisfied");
            println!("    - Soft constraint violations minimized");
            println!("    - Total violated weight: {}", cost);
        }
        MaxSatResult::Unsatisfiable => {
            println!("  ✗ UNSATISFIABLE");
            println!("  Hard constraints cannot be satisfied");
        }
        MaxSatResult::Unknown => {
            println!("  ? UNKNOWN");
            println!("  Could not find solution within resource limits");
        }
    }

    // Display statistics
    let stats = solver.stats();
    println!("\nStatistics:");
    println!("  SAT solver calls: {}", stats.sat_calls);
    println!("  UNSAT cores found: {}", stats.cores_found);
    println!("  Iterations: {}", stats.iterations);
    println!("  Lower bound: {}", stats.lower_bound);
    println!("  Upper bound: {:?}", stats.upper_bound);

    println!("\n=== Real-world usage ===");
    println!("For polynomial constraints:");
    println!("1. Create polynomial atoms using NlsatSolver");
    println!("2. Convert atoms to literals");
    println!("3. Add literals as hard/soft constraints");
    println!("4. Solve to find optimal assignment");

    println!("\n=== Example completed ===");
}
