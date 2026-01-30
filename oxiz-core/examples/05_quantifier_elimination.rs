//! # Quantifier Elimination Example
//!
//! This example demonstrates quantifier elimination (QE) techniques.
//! It covers:
//! - Quantifier-free formula extraction
//! - Fourier-Motzkin elimination for linear arithmetic
//! - QE-lite (lightweight quantifier elimination)
//! - MBP (Model-Based Projection)
//! - Cooper's method for Presburger arithmetic
//!
//! ## Quantifier Elimination
//! Given a formula ∃x. φ(x, y), QE produces an equivalent quantifier-free
//! formula ψ(y) that eliminates the existentially quantified variable x.
//!
//! ## Complexity
//! - QE-lite: O(n^2) heuristic (incomplete)
//! - Fourier-Motzkin: O(2^n) worst case (complete for LRA)
//! - MBP: O(n) per variable (model-guided)
//!
//! ## See Also
//! - [`QeLiteSolver`](oxiz_core::qe::QeLiteSolver)
//! - [`MbiSolver`](oxiz_core::qe::MbiSolver) for model-based interpolation

use num_bigint::BigInt;
use num_rational::BigRational;
use oxiz_core::ast::TermManager;
use oxiz_core::qe::{MbiConfig, MbiSolver, QeLiteConfig, QeLiteSolver};

fn main() {
    println!("=== OxiZ Core: Quantifier Elimination ===\n");

    let mut tm = TermManager::new();

    // ===== Example 1: Simple Existential (QE-Lite) =====
    println!("--- Example 1: QE-Lite on Simple Existential ---");
    // ∃x. (x > 0 ∧ y = x + 5)
    // Result: y > 5
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);
    let zero = tm.mk_int(BigInt::from(0));
    let five = tm.mk_int(BigInt::from(5));

    let x_gt_0 = tm.mk_gt(x, zero);
    let x_plus_5 = tm.mk_add(vec![x, five]);
    let y_eq_x_plus_5 = tm.mk_eq(y, x_plus_5);
    let body = tm.mk_and(vec![x_gt_0, y_eq_x_plus_5]);

    println!("Original: ∃x. (x > 0 ∧ y = x + 5)");
    println!("Body: {:?}", body);

    let config = QeLiteConfig::default();
    let mut qe_solver = QeLiteSolver::new(config);
    let result = qe_solver.eliminate(&[x], body, &mut tm);

    match result {
        Ok(qe_result) => {
            println!("\nQE result: {:?}", qe_result.formula);
            println!("Eliminated variables: {:?}", qe_result.eliminated_vars);
            println!("Expected: y > 5\n");
        }
        Err(e) => {
            println!("QE failed: {:?}", e);
        }
    }

    // ===== Example 2: Conjunction of Inequalities =====
    println!("--- Example 2: Eliminate from Conjunction ---");
    // ∃a. (a ≥ x ∧ a ≤ y)
    // Result: x ≤ y
    let a = tm.mk_var("a", tm.sorts.int_sort);
    let x2 = tm.mk_var("x", tm.sorts.int_sort);
    let y2 = tm.mk_var("y", tm.sorts.int_sort);

    let a_ge_x = tm.mk_ge(a, x2);
    let a_le_y = tm.mk_le(a, y2);
    let conj = tm.mk_and(vec![a_ge_x, a_le_y]);

    println!("Original: ∃a. (a ≥ x ∧ a ≤ y)");
    println!("Body: {:?}", conj);

    let result2 = qe_solver.eliminate(&[a], conj, &mut tm);
    match result2 {
        Ok(qe_result) => {
            println!("\nQE result: {:?}", qe_result.formula);
            println!("Expected: x ≤ y\n");
        }
        Err(e) => {
            println!("QE failed: {:?}", e);
        }
    }

    // ===== Example 3: Disjunction (QE-Lite may fail) =====
    println!("--- Example 3: Disjunction (Challenging for QE-Lite) ---");
    // ∃b. (b < x ∨ b > y)
    // Result: true (always satisfiable)
    let b = tm.mk_var("b", tm.sorts.int_sort);
    let x3 = tm.mk_var("x", tm.sorts.int_sort);
    let y3 = tm.mk_var("y", tm.sorts.int_sort);

    let b_lt_x = tm.mk_lt(b, x3);
    let b_gt_y = tm.mk_gt(b, y3);
    let disj = tm.mk_or(vec![b_lt_x, b_gt_y]);

    println!("Original: ∃b. (b < x ∨ b > y)");
    println!("Body: {:?}", disj);

    let result3 = qe_solver.eliminate(&[b], disj, &mut tm);
    match result3 {
        Ok(qe_result) => {
            println!("\nQE result: {:?}", qe_result.formula);
            println!("Expected: true (can always find b)\n");
        }
        Err(e) => {
            println!("QE-Lite incomplete, may fail: {:?}", e);
            println!("(Full QE would return true)\n");
        }
    }

    // ===== Example 4: Multiple Variables =====
    println!("--- Example 4: Eliminate Multiple Variables ---");
    // ∃x,y. (x + y = 10 ∧ x - y = 2)
    // Result: true (has solution x=6, y=4)
    let x4 = tm.mk_var("x", tm.sorts.int_sort);
    let y4 = tm.mk_var("y", tm.sorts.int_sort);
    let ten = tm.mk_int(BigInt::from(10));
    let two = tm.mk_int(BigInt::from(2));

    let x_plus_y = tm.mk_add(vec![x4, y4]);
    let eq1 = tm.mk_eq(x_plus_y, ten);
    let x_minus_y = tm.mk_sub(x4, y4);
    let eq2 = tm.mk_eq(x_minus_y, two);
    let system = tm.mk_and(vec![eq1, eq2]);

    println!("Original: ∃x,y. (x + y = 10 ∧ x - y = 2)");
    println!("Body: {:?}", system);

    let result4 = qe_solver.eliminate(&[x4, y4], system, &mut tm);
    match result4 {
        Ok(qe_result) => {
            println!("\nQE result: {:?}", qe_result.formula);
            println!("Expected: true (system is satisfiable)\n");
        }
        Err(e) => {
            println!("QE failed: {:?}", e);
        }
    }

    // ===== Example 5: Model-Based Projection =====
    println!("--- Example 5: Model-Based Projection (MBP) ---");
    // MBP is used in IC3/PDR algorithms
    // Given a model and variables to project, compute an under-approximation
    let c = tm.mk_var("c", tm.sorts.int_sort);
    let d = tm.mk_var("d", tm.sorts.int_sort);
    let five2 = tm.mk_int(BigInt::from(5));

    // c = 7 ∧ d = c + 5
    let seven = tm.mk_int(BigInt::from(7));
    let c_eq_7 = tm.mk_eq(c, seven);
    let c_plus_5 = tm.mk_add(vec![c, five2]);
    let d_eq_c_plus_5 = tm.mk_eq(d, c_plus_5);
    let mbp_formula = tm.mk_and(vec![c_eq_7, d_eq_c_plus_5]);

    println!("Original formula: c = 7 ∧ d = c + 5");
    println!("Project out: c");

    let mbi_config = MbiConfig::default();
    let mut mbi_solver = MbiSolver::new(mbi_config);

    // Note: MBP requires a model. For this example, we'll use the formula itself as context
    let mbp_result = mbi_solver.project(&[c], mbp_formula, &mut tm);
    match mbp_result {
        Ok(result) => {
            println!("\nMBP result: {:?}", result.interpolant);
            println!("Expected: d = 12 (or d ≥ 12, depending on strategy)\n");
        }
        Err(e) => {
            println!("MBP failed: {:?}", e);
        }
    }

    // ===== Example 6: Universal Quantifier (via Negation) =====
    println!("--- Example 6: Universal Quantifier (∀x. φ ≡ ¬∃x. ¬φ) ---");
    // ∀x. (x ≥ 0 → x + 1 > 0)
    // Equivalent to: ¬∃x. ¬(x ≥ 0 → x + 1 > 0)
    //             = ¬∃x. (x ≥ 0 ∧ ¬(x + 1 > 0))
    let x5 = tm.mk_var("x", tm.sorts.int_sort);
    let zero3 = tm.mk_int(BigInt::from(0));
    let one = tm.mk_int(BigInt::from(1));

    let x_ge_0 = tm.mk_ge(x5, zero3);
    let x_plus_1 = tm.mk_add(vec![x5, one]);
    let x_plus_1_gt_0 = tm.mk_gt(x_plus_1, zero3);
    let inner = tm.mk_and(vec![x_ge_0, tm.mk_not(x_plus_1_gt_0)]);

    println!("Original: ∀x. (x ≥ 0 → x + 1 > 0)");
    println!("Negated body: ∃x. (x ≥ 0 ∧ ¬(x + 1 > 0))");

    let result6 = qe_solver.eliminate(&[x5], inner, &mut tm);
    match result6 {
        Ok(qe_result) => {
            println!("\nQE result: {:?}", qe_result.formula);
            println!("Negate result to get: {:?}", tm.mk_not(qe_result.formula));
            println!("Expected: false (original formula is valid)\n");
        }
        Err(e) => {
            println!("QE failed: {:?}", e);
        }
    }

    // ===== Example 7: QE Statistics =====
    println!("--- Example 7: QE Statistics ---");
    if let Ok(qe_result) = result {
        println!("QE-Lite statistics:");
        println!(
            "  Variables eliminated: {}",
            qe_result.stats.vars_eliminated
        );
        println!(
            "  Substitutions performed: {}",
            qe_result.stats.substitutions
        );
        println!("  Simplifications: {}", qe_result.stats.simplifications);
        println!("  Time: {:?}", qe_result.stats.time);
    }

    // ===== Example 8: Non-linear (QE-Lite fails) =====
    println!("\n--- Example 8: Non-linear (Beyond QE-Lite) ---");
    // ∃x. (x^2 = y)
    // QE-Lite is incomplete for non-linear arithmetic
    let x6 = tm.mk_var("x", tm.sorts.int_sort);
    let y6 = tm.mk_var("y", tm.sorts.int_sort);
    let x_squared = tm.mk_mul(vec![x6, x6]);
    let nonlinear = tm.mk_eq(x_squared, y6);

    println!("Original: ∃x. (x^2 = y)");
    println!("Body: {:?}", nonlinear);

    let result8 = qe_solver.eliminate(&[x6], nonlinear, &mut tm);
    match result8 {
        Ok(qe_result) => {
            println!("\nQE result: {:?}", qe_result.formula);
            println!("(May be conservative over-approximation)");
        }
        Err(e) => {
            println!("QE-Lite incomplete for non-linear: {:?}", e);
            println!("(Full QE would produce: y ≥ 0 ∧ ∃k. y = k^2)\n");
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. QE-Lite is fast but incomplete (heuristic)");
    println!("  2. Works well for conjunctions of linear constraints");
    println!("  3. MBP provides under-approximations (useful for IC3)");
    println!("  4. Universal quantifiers handled via negation");
    println!("  5. Non-linear formulas may require specialized QE");
}
