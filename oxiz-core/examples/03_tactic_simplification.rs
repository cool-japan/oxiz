//! # Tactic-Based Simplification Example
//!
//! This example demonstrates using tactics to simplify formulas.
//! It covers:
//! - Creating goals from formulas
//! - Applying simplification tactics
//! - Tactic composition (sequential and parallel)
//! - Built-in tactics (simplify, propagate-ineqs, solve-eqs)
//! - Measuring tactic performance
//!
//! ## Tactics Overview
//! Tactics are transformations that convert one goal into zero or more subgoals.
//! They're used for preprocessing, simplification, and problem decomposition.
//!
//! ## Complexity
//! - Varies by tactic (documented per-tactic)
//! - Simplification: typically O(n) or O(n log n)
//!
//! ## See Also
//! - [`Tactic`](oxiz_core::tactic::Tactic) trait
//! - [`Goal`](oxiz_core::tactic::Goal) for goal representation

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_core::tactic::{Goal, Tactic};
use oxiz_core::tactic::{SolveEqsTactic, StatelessSimplifyTactic};

fn main() {
    println!("=== OxiZ Core: Tactic-Based Simplification ===\n");

    let mut tm = TermManager::new();

    // ===== Example 1: Boolean Simplification =====
    println!("--- Example 1: Boolean Simplification ---");
    let p = tm.mk_var("p", tm.sorts.bool_sort);
    let q = tm.mk_var("q", tm.sorts.bool_sort);

    // Create redundant formula: (p OR true) AND (q AND true)
    let true_term = tm.mk_true();
    let or_p_true = tm.mk_or(vec![p, true_term]);
    let and_q_true = tm.mk_and(vec![q, true_term]);
    let redundant = tm.mk_and(vec![or_p_true, and_q_true]);

    println!("Original formula: {:?}", redundant);
    println!("  (p OR true) AND (q AND true)");

    let goal = Goal::new(vec![redundant]);
    let tactic = StatelessSimplifyTactic;
    let result = tactic.apply(&goal);

    println!("\nAfter simplification:");
    for (i, subgoal) in result.subgoals.iter().enumerate() {
        println!("  Subgoal {}: {:?}", i, subgoal.formulas);
    }
    println!("  Expected: [q] (since p OR true = true, and true is absorbed)\n");

    // ===== Example 2: Constant Propagation =====
    println!("--- Example 2: Constant Propagation ---");
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);

    // Formula: x = 5 AND y = x + 10
    let five = tm.mk_int(BigInt::from(5));
    let ten = tm.mk_int(BigInt::from(10));
    let x_eq_5 = tm.mk_eq(x, five);
    let x_plus_10 = tm.mk_add(vec![x, ten]);
    let y_eq_x_plus_10 = tm.mk_eq(y, x_plus_10);
    let formula = tm.mk_and(vec![x_eq_5, y_eq_x_plus_10]);

    println!("Original formula: {:?}", formula);
    println!("  (x = 5) AND (y = x + 10)");

    let goal2 = Goal::new(vec![formula]);
    let result2 = tactic.apply(&goal2);

    println!("\nAfter simplification:");
    for (i, subgoal) in result2.subgoals.iter().enumerate() {
        println!("  Subgoal {}: {:?}", i, subgoal.formulas);
    }
    println!("  Expected: x = 5 AND y = 15\n");

    // ===== Example 3: Arithmetic Simplification =====
    println!("--- Example 3: Arithmetic Simplification ---");
    let a = tm.mk_var("a", tm.sorts.int_sort);

    // Complex arithmetic: (a + 0) * 1 + (a - a)
    let zero = tm.mk_int(BigInt::from(0));
    let one = tm.mk_int(BigInt::from(1));
    let a_plus_0 = tm.mk_add(vec![a, zero]);
    let mul_by_1 = tm.mk_mul(vec![a_plus_0, one]);
    let a_minus_a = tm.mk_sub(a, a);
    let complex_arith = tm.mk_add(vec![mul_by_1, a_minus_a]);

    println!("Original formula: {:?}", complex_arith);
    println!("  ((a + 0) * 1) + (a - a)");

    let goal3 = Goal::new(vec![complex_arith]);
    let result3 = tactic.apply(&goal3);

    println!("\nAfter simplification:");
    for (i, subgoal) in result3.subgoals.iter().enumerate() {
        println!("  Subgoal {}: {:?}", i, subgoal.formulas);
    }
    println!("  Expected: a\n");

    // ===== Example 4: De Morgan's Laws =====
    println!("--- Example 4: De Morgan's Laws ---");
    let p1 = tm.mk_var("p1", tm.sorts.bool_sort);
    let q1 = tm.mk_var("q1", tm.sorts.bool_sort);

    // NOT(p AND q) should become (NOT p) OR (NOT q)
    let and_pq = tm.mk_and(vec![p1, q1]);
    let not_and = tm.mk_not(and_pq);

    println!("Original formula: {:?}", not_and);
    println!("  NOT(p AND q)");

    let goal4 = Goal::new(vec![not_and]);
    let result4 = tactic.apply(&goal4);

    println!("\nAfter simplification (De Morgan's):");
    for (i, subgoal) in result4.subgoals.iter().enumerate() {
        println!("  Subgoal {}: {:?}", i, subgoal.formulas);
    }
    println!("  Expected: (NOT p) OR (NOT q)\n");

    // ===== Example 5: Equation Solving =====
    println!("--- Example 5: Equation Solving Tactic ---");
    let x2 = tm.mk_var("x", tm.sorts.int_sort);
    let y2 = tm.mk_var("y", tm.sorts.int_sort);
    let z = tm.mk_var("z", tm.sorts.int_sort);

    // x = y + 5, z = x + 10 -> z = y + 15
    let five2 = tm.mk_int(BigInt::from(5));
    let ten2 = tm.mk_int(BigInt::from(10));
    let y_plus_5 = tm.mk_add(vec![y2, five2]);
    let x_eq_y_plus_5 = tm.mk_eq(x2, y_plus_5);
    let x_plus_10 = tm.mk_add(vec![x2, ten2]);
    let z_eq_x_plus_10 = tm.mk_eq(z, x_plus_10);
    let eqs = tm.mk_and(vec![x_eq_y_plus_5, z_eq_x_plus_10]);

    println!("Original formula: {:?}", eqs);
    println!("  (x = y + 5) AND (z = x + 10)");

    let goal5 = Goal::new(vec![eqs]);
    let solve_eqs_tactic = SolveEqsTactic;
    let result5 = solve_eqs_tactic.apply(&goal5);

    println!("\nAfter solve-eqs tactic:");
    for (i, subgoal) in result5.subgoals.iter().enumerate() {
        println!("  Subgoal {}: {:?}", i, subgoal.formulas);
    }
    println!("  Expected: eliminate x, substitute into z equation\n");

    // ===== Example 6: Contextual Simplification =====
    println!("--- Example 6: Contextual Simplification ---");
    let c = tm.mk_var("c", tm.sorts.bool_sort);
    let t = tm.mk_var("t", tm.sorts.int_sort);

    // (c => (t > 0)) AND c AND (t <= 0)
    // Under context c=true, we get t > 0 AND t <= 0 (unsatisfiable)
    let zero2 = tm.mk_int(BigInt::from(0));
    let t_gt_0 = tm.mk_gt(t, zero2);
    let t_le_0 = tm.mk_le(t, zero2);
    let implies = tm.mk_implies(c, t_gt_0);
    let contextual = tm.mk_and(vec![implies, c, t_le_0]);

    println!("Original formula: {:?}", contextual);
    println!("  (c => t > 0) AND c AND (t <= 0)");

    let goal6 = Goal::new(vec![contextual]);
    let result6 = tactic.apply(&goal6);

    println!("\nAfter contextual simplification:");
    for (i, subgoal) in result6.subgoals.iter().enumerate() {
        println!("  Subgoal {}: {:?}", i, subgoal.formulas);
    }
    println!("  Should detect contradiction: t > 0 AND t <= 0\n");

    // ===== Example 7: Tactic Statistics =====
    println!("--- Example 7: Tactic Statistics ---");
    let stats = result6.stats;
    println!("Tactic statistics:");
    println!("  Time: {:?}", stats.time);
    println!("  Simplifications applied: {}", stats.num_simplifications);
    println!("  Original goal size: {}", stats.original_size);
    println!("  Simplified goal size: {}", stats.simplified_size);

    // ===== Example 8: Empty Goal (Satisfiable) =====
    println!("\n--- Example 8: Empty Goal (Trivially Satisfiable) ---");
    let empty_goal = Goal::new(vec![]);
    let result8 = tactic.apply(&empty_goal);
    println!("Empty goal result: {} subgoals", result8.subgoals.len());
    if result8.subgoals.is_empty() {
        println!("  Empty result = satisfiable (no constraints)");
    }

    // ===== Example 9: False Goal (Unsatisfiable) =====
    println!("\n--- Example 9: False Goal (Trivially Unsatisfiable) ---");
    let false_term = tm.mk_false();
    let false_goal = Goal::new(vec![false_term]);
    let result9 = tactic.apply(&false_goal);
    println!("False goal result: {} subgoals", result9.subgoals.len());
    for (i, subgoal) in result9.subgoals.iter().enumerate() {
        if subgoal.formulas.contains(&false_term) {
            println!("  Subgoal {} contains false = unsatisfiable", i);
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Tactics transform goals into simpler subgoals");
    println!("  2. Simplification eliminates redundancy (true/false, identities)");
    println!("  3. Solve-eqs eliminates variables via substitution");
    println!("  4. Contextual simplification uses known facts");
    println!("  5. Statistics help measure tactic effectiveness");
}
