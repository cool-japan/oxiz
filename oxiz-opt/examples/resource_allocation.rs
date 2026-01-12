//! Resource Allocation Example
//!
//! This example demonstrates using weighted MaxSAT to solve a resource allocation problem.
//!
//! Problem: Allocate limited resources to projects to maximize value.
//! - Each project has a value if assigned
//! - Hard constraint: Total resource usage must not exceed capacity
//! - Soft constraint: Maximize total value
//!
//! Run with: cargo run --example resource_allocation

use oxiz_opt::{MaxSatAlgorithm, MaxSatConfig, MaxSatSolver, Weight};
use oxiz_sat::{Lit, Var};

#[derive(Debug)]
struct Project {
    name: &'static str,
    resource_cost: u32,
    value: u32,
}

fn main() {
    println!("=== Resource Allocation with Weighted MaxSAT ===\n");

    // Define projects
    let projects = [
        Project {
            name: "Project Alpha",
            resource_cost: 3,
            value: 10,
        },
        Project {
            name: "Project Beta",
            resource_cost: 2,
            value: 7,
        },
        Project {
            name: "Project Gamma",
            resource_cost: 4,
            value: 12,
        },
        Project {
            name: "Project Delta",
            resource_cost: 1,
            value: 4,
        },
        Project {
            name: "Project Epsilon",
            resource_cost: 3,
            value: 9,
        },
    ];

    let total_capacity = 7; // Total resource capacity

    // Variable: project i is selected
    let var = |i: usize| -> Var { Var::new(i as u32) };

    // Use WMax algorithm for weighted instances
    let config = MaxSatConfig::builder()
        .algorithm(MaxSatAlgorithm::WMax)
        .build();
    let mut solver = MaxSatSolver::with_config(config);

    // Hard constraint: Resource capacity
    // For simplicity, we encode using at-most-k constraint
    // In practice, this would use cardinality networks or pseudo-boolean constraints

    // For this example, we'll use a simpler approach:
    // Add constraints that prevent resource overflow by forbidding certain combinations

    // Generate all subsets that exceed capacity
    for mask in 0..(1u32 << projects.len()) {
        let mut total_cost = 0;
        let mut selected = Vec::new();

        for (i, project) in projects.iter().enumerate() {
            if mask & (1 << i) != 0 {
                total_cost += project.resource_cost;
                selected.push(i);
            }
        }

        // If this combination exceeds capacity, add a hard clause forbidding it
        if total_cost > total_capacity {
            let clause: Vec<_> = selected.iter().map(|&i| Lit::neg(var(i))).collect();
            solver.add_hard(clause);
        }
    }

    // Soft constraints: Maximize value (add soft clauses for selecting each project)
    for (i, project) in projects.iter().enumerate() {
        println!(
            "Project {}: cost={}, value={}",
            project.name, project.resource_cost, project.value
        );

        // Add soft clause: prefer to select this project with weight = value
        solver.add_soft_weighted([Lit::pos(var(i))], Weight::from(project.value as i64));
    }

    println!("\nTotal capacity: {}", total_capacity);
    println!("\nSolving optimization problem...");

    match solver.solve() {
        Ok(result) => {
            println!("Result: {}\n", result);

            if let Some(model) = solver.best_model() {
                println!("Optimal Allocation:");
                let mut total_cost = 0;
                let mut total_value = 0;

                for (i, project) in projects.iter().enumerate() {
                    let v = var(i);
                    if v.0 < model.len() as u32 && model[v.0 as usize] == oxiz_sat::LBool::True {
                        println!(
                            "  ✓ {} (cost: {}, value: {})",
                            project.name, project.resource_cost, project.value
                        );
                        total_cost += project.resource_cost;
                        total_value += project.value;
                    }
                }

                println!(
                    "\nTotal resource usage: {} / {}",
                    total_cost, total_capacity
                );
                println!("Total value achieved: {}", total_value);
                println!("\nUnachieved value (cost): {}", solver.cost());

                println!("\nStatistics:");
                let stats = solver.stats();
                println!("  SAT calls: {}", stats.sat_calls);
                println!("  Cores extracted: {}", stats.cores_extracted);
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}
