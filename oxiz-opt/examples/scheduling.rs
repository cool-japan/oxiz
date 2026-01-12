//! Scheduling Problem Example
//!
//! This example demonstrates using oxiz-opt to solve a simple scheduling problem.
//!
//! Problem: Schedule tasks to time slots with preferences and constraints.
//! - Hard constraints: No two conflicting tasks in the same slot
//! - Soft constraints: Preferences for specific time slots (weighted)
//!
//! Run with: cargo run --example scheduling

use oxiz_opt::{MaxSatSolver, Weight};
use oxiz_sat::{Lit, Var};

fn main() {
    println!("=== Task Scheduling with MaxSAT ===\n");

    // Problem setup:
    // 3 tasks (A, B, C), 3 time slots (morning, afternoon, evening)
    // Each task needs exactly one time slot

    let num_tasks = 3;
    let num_slots = 3;

    // Variable encoding: var(task, slot) = task * num_slots + slot
    let var = |task: u32, slot: u32| -> Var { Var::new(task * num_slots + slot) };

    let mut solver = MaxSatSolver::new();

    // Hard constraint 1: Each task must be assigned to exactly one slot
    for task in 0..num_tasks {
        // At least one slot
        let clause: Vec<_> = (0..num_slots)
            .map(|slot| Lit::pos(var(task, slot)))
            .collect();
        solver.add_hard(clause);

        // At most one slot (pairwise)
        for slot1 in 0..num_slots {
            for slot2 in (slot1 + 1)..num_slots {
                solver.add_hard([Lit::neg(var(task, slot1)), Lit::neg(var(task, slot2))]);
            }
        }
    }

    // Hard constraint 2: Tasks A and B conflict (can't be in same slot)
    let task_a = 0;
    let task_b = 1;
    for slot in 0..num_slots {
        solver.add_hard([Lit::neg(var(task_a, slot)), Lit::neg(var(task_b, slot))]);
    }

    // Soft constraints: Preferences (higher weight = stronger preference)

    // Task A prefers morning (weight 5)
    solver.add_soft_weighted([Lit::pos(var(0, 0))], Weight::from(5));

    // Task B prefers afternoon (weight 3)
    solver.add_soft_weighted([Lit::pos(var(1, 1))], Weight::from(3));

    // Task C prefers evening (weight 4)
    solver.add_soft_weighted([Lit::pos(var(2, 2))], Weight::from(4));

    // Task A dislikes evening (negative preference, weight 2)
    solver.add_soft_weighted([Lit::neg(var(0, 2))], Weight::from(2));

    // Solve
    println!("Solving scheduling problem...");
    match solver.solve() {
        Ok(result) => {
            println!("Result: {}\n", result);

            if let Some(model) = solver.best_model() {
                println!("Optimal Schedule:");
                let slot_names = ["Morning", "Afternoon", "Evening"];
                let task_names = ["Task A", "Task B", "Task C"];

                for task in 0..num_tasks {
                    for slot in 0..num_slots {
                        let v = var(task, slot);
                        if v.0 < model.len() as u32 && model[v.0 as usize] == oxiz_sat::LBool::True
                        {
                            println!(
                                "  {} -> {}",
                                task_names[task as usize], slot_names[slot as usize]
                            );
                        }
                    }
                }

                println!("\nCost (unsatisfied preferences): {}", solver.cost());
                println!("Statistics:");
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
