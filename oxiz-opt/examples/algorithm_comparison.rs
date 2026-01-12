//! Algorithm Comparison Example
//!
//! This example demonstrates comparing different MaxSAT algorithms on the same problem.
//!
//! Run with: cargo run --example algorithm_comparison

use oxiz_opt::{MaxSatAlgorithm, MaxSatConfig, MaxSatSolver};
use oxiz_sat::{Lit, Var};
use std::time::Instant;

fn solve_with_algorithm(algo: MaxSatAlgorithm, name: &str) {
    println!("\n=== {} ===", name);

    let config = MaxSatConfig::builder().algorithm(algo).build();

    // Create a fresh solver with the config
    let mut solver = MaxSatSolver::with_config(config);

    let num_nodes = 5;
    let num_colors = 3;

    let var = |node: u32, color: u32| -> Var { Var::new(node * num_colors + color) };

    // Hard: Each node gets exactly one color
    for node in 0..num_nodes {
        let clause: Vec<_> = (0..num_colors)
            .map(|color| Lit::pos(var(node, color)))
            .collect();
        solver.add_hard(clause);

        for c1 in 0..num_colors {
            for c2 in (c1 + 1)..num_colors {
                solver.add_hard([Lit::neg(var(node, c1)), Lit::neg(var(node, c2))]);
            }
        }
    }

    // Soft: Adjacent nodes should not have the same color
    let edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)];
    for &(n1, n2) in &edges {
        for color in 0..num_colors {
            solver.add_soft([Lit::neg(var(n1, color)), Lit::neg(var(n2, color))]);
        }
    }

    let start = Instant::now();
    match solver.solve() {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!("Result: {}", result);
            println!("Time: {:?}", elapsed);
            println!("Cost (conflicts): {}", solver.cost());

            let stats = solver.stats();
            println!("SAT calls: {}", stats.sat_calls);
            println!("Cores extracted: {}", stats.cores_extracted);
            println!(
                "Core minimization: {} lits removed",
                stats.core_min_lits_removed
            );
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}

fn main() {
    println!("=== MaxSAT Algorithm Comparison ===");
    println!("Problem: Graph coloring with 5 nodes, 3 colors");
    println!("Edges: (0,1), (1,2), (2,3), (3,4), (0,4)");
    println!("Goal: Minimize coloring conflicts");

    solve_with_algorithm(MaxSatAlgorithm::FuMalik, "Fu-Malik Algorithm");
    solve_with_algorithm(MaxSatAlgorithm::Oll, "OLL Algorithm");
    solve_with_algorithm(MaxSatAlgorithm::Msu3, "MSU3 Algorithm");
    solve_with_algorithm(MaxSatAlgorithm::Pmres, "PMRES Algorithm");

    println!("\n=== Summary ===");
    println!("Different algorithms may have different performance characteristics");
    println!("depending on the problem structure:");
    println!("  - Fu-Malik: Simple, general-purpose core-guided");
    println!("  - OLL: Better for instances with many overlapping cores");
    println!("  - MSU3: Simple relaxation strategy");
    println!("  - PMRES: Resolution-based, good for partial MaxSAT");
}
