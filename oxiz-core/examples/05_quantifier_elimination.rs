//! # Quantifier Elimination Example
//!
//! This example demonstrates quantifier elimination (QE) techniques.
//! It covers:
//! - Quantifier-free formula extraction
//! - QE-lite (lightweight quantifier elimination)
//! - Building quantified formulas
//! - Elimination strategies overview
//!
//! ## Quantifier Elimination
//! Given a formula exists x. phi(x, y), QE produces an equivalent quantifier-free
//! formula psi(y) that eliminates the existentially quantified variable x.
//!
//! ## Complexity
//! - QE-lite: O(n^2) heuristic (incomplete)
//! - Fourier-Motzkin: O(2^n) worst case (complete for LRA)
//! - MBP: O(n) per variable (model-guided)
//!
//! ## See Also
//! - [`QeLiteSolver`](oxiz_core::qe::QeLiteSolver)
//! - [`MbiSolver`](oxiz_core::qe::MbiSolver) for model-based interpolation

use oxiz_core::ast::TermManager;
use oxiz_core::qe::{QeLiteConfig, QeLiteResult, QeLiteSolver};

fn main() {
    println!("=== OxiZ Core: Quantifier Elimination ===\n");

    let mut tm = TermManager::new();

    // ===== Example 1: Building Quantified Formulas =====
    println!("--- Example 1: Building Quantified Formulas ---");

    let x = tm.mk_var("x", tm.sorts.int_sort);
    let y = tm.mk_var("y", tm.sorts.int_sort);
    let zero = tm.mk_int(0);
    let five = tm.mk_int(5);

    // Build: exists x. (x > 0 AND y = x + 5)
    let x_gt_0 = tm.mk_gt(x, zero);
    let x_plus_5 = tm.mk_add(vec![x, five]);
    let y_eq_x_plus_5 = tm.mk_eq(y, x_plus_5);
    let body = tm.mk_and(vec![x_gt_0, y_eq_x_plus_5]);

    println!("Body formula: x > 0 AND y = x + 5");
    println!("Body term: {:?}", body);

    // Create existential quantifier
    let exists_formula = tm.mk_exists(vec![("x", tm.sorts.int_sort)], body);

    println!("Exists formula: exists x. (x > 0 AND y = x + 5)");
    println!("Exists term: {:?}", exists_formula);
    println!();

    // ===== Example 2: QE-Lite Configuration =====
    println!("--- Example 2: QE-Lite Configuration ---");

    let config = QeLiteConfig::default();
    println!("QeLiteConfig (default):");
    println!("  max_formula_size: {}", config.max_formula_size);
    println!("  equality_substitution: {}", config.equality_substitution);
    println!("  bound_elimination: {}", config.bound_elimination);
    println!("  divisibility_handling: {}", config.divisibility_handling);
    println!();

    // Custom configuration
    let _custom_config = QeLiteConfig {
        max_formula_size: 500,
        equality_substitution: true,
        bound_elimination: true,
        divisibility_handling: false,
    };
    println!("Custom config created: max_formula_size=500, divisibility_handling=false");
    println!();

    // ===== Example 3: QE-Lite Solver =====
    println!("--- Example 3: QE-Lite Solver ---");

    let mut qe_solver = QeLiteSolver::new();
    println!("Created QE-Lite solver");

    // Try elimination on the existential formula
    let result = qe_solver.eliminate(exists_formula, &mut tm);
    print_qe_lite_result("exists x. (x > 0 AND y = x + 5)", &result);
    println!();

    // ===== Example 4: Universal Quantifier =====
    println!("--- Example 4: Universal Quantifier ---");

    // forall x. (x >= 0 => x >= -1)
    let neg_one = tm.mk_int(-1);
    let x_ge_0 = tm.mk_ge(x, zero);
    let x_ge_neg1 = tm.mk_ge(x, neg_one);
    let implies_body = tm.mk_implies(x_ge_0, x_ge_neg1);

    let forall_formula = tm.mk_forall(vec![("x", tm.sorts.int_sort)], implies_body);

    println!("Universal formula: forall x. (x >= 0 => x >= -1)");
    println!("  This is a tautology");

    let result2 = qe_solver.eliminate(forall_formula, &mut tm);
    print_qe_lite_result("forall x. (x >= 0 => x >= -1)", &result2);
    println!();

    // ===== Example 5: QE-Lite Statistics =====
    println!("--- Example 5: QE-Lite Statistics ---");

    let stats = qe_solver.stats();
    println!("QE-Lite statistics:");
    println!("  Attempts: {}", stats.attempts);
    println!("  Successes: {}", stats.successes);
    println!("  Simplifications: {}", stats.simplifications);
    println!("  Equality substitutions: {}", stats.equality_subs);
    println!("  Bound eliminations: {}", stats.bound_elims);
    println!();

    // ===== Example 6: QeLiteResult API =====
    println!("--- Example 6: QeLiteResult API ---");

    let true_term = tm.mk_true();
    let eliminated = QeLiteResult::Eliminated(true_term);
    let simplified = QeLiteResult::Simplified(true_term);
    let unchanged = QeLiteResult::Unchanged;
    let error = QeLiteResult::Error("Test error".to_string());

    println!("QeLiteResult variants:");
    println!(
        "  Eliminated: is_eliminated={}, result_term={:?}",
        eliminated.is_eliminated(),
        eliminated.result_term()
    );
    println!(
        "  Simplified: is_eliminated={}, result_term={:?}",
        simplified.is_eliminated(),
        simplified.result_term()
    );
    println!(
        "  Unchanged: is_eliminated={}, result_term={:?}",
        unchanged.is_eliminated(),
        unchanged.result_term()
    );
    println!(
        "  Error: is_eliminated={}, result_term={:?}",
        error.is_eliminated(),
        error.result_term()
    );
    println!();

    // ===== Example 7: Linear Constraints =====
    println!("--- Example 7: Linear Constraint Bounds ---");

    let a = tm.mk_var("a", tm.sorts.int_sort);
    let b = tm.mk_var("b", tm.sorts.int_sort);
    let ten = tm.mk_int(10);

    // exists a. (a < 10 AND a > 0 AND b = a + 5)
    let a_lt_10 = tm.mk_lt(a, ten);
    let a_gt_0 = tm.mk_gt(a, zero);
    let a_plus_5 = tm.mk_add(vec![a, five]);
    let b_eq_a_plus_5 = tm.mk_eq(b, a_plus_5);
    let linear_body = tm.mk_and(vec![a_lt_10, a_gt_0, b_eq_a_plus_5]);

    let linear_exists = tm.mk_exists(vec![("a", tm.sorts.int_sort)], linear_body);

    println!("Formula: exists a. (a < 10 AND a > 0 AND b = a + 5)");
    println!("  With bounds: 0 < a < 10");
    println!("  After QE: 5 < b < 15");

    let result3 = qe_solver.eliminate(linear_exists, &mut tm);
    print_qe_lite_result("exists a. (0 < a < 10 AND b = a + 5)", &result3);
    println!();

    // ===== Example 8: Reset Statistics =====
    println!("--- Example 8: Reset Statistics ---");

    println!("Before reset: {} attempts", qe_solver.stats().attempts);
    qe_solver.reset_stats();
    println!("After reset: {} attempts", qe_solver.stats().attempts);
    println!();

    // ===== Example 9: QE Algorithms Overview =====
    println!("--- Example 9: QE Algorithms Overview ---");

    println!("Quantifier Elimination Algorithms:");
    println!();
    println!("| Algorithm        | Domain | Complexity    | Use Case              |");
    println!("|------------------|--------|---------------|----------------------|");
    println!("| QE-Lite          | Any    | O(n)          | Simple patterns       |");
    println!("| Cooper           | LIA    | O(2^n)        | Integer arithmetic    |");
    println!("| Ferrante-Rackoff | LRA    | O(n^4)        | Real arithmetic       |");
    println!("| Omega Test       | LIA    | O(n*m) avg    | Integer constraints   |");
    println!("| CAD              | NRA    | Doubly exp    | Polynomial constraints|");
    println!();
    println!("LIA = Linear Integer Arithmetic");
    println!("LRA = Linear Real Arithmetic");
    println!("NRA = Nonlinear Real Arithmetic");
    println!();

    // ===== Example 10: Non-Linear (QE-Lite incomplete) =====
    println!("--- Example 10: Non-Linear (Beyond QE-Lite) ---");

    // exists x. (x * x = y) - non-linear
    let x2 = tm.mk_var("x", tm.sorts.int_sort);
    let y2 = tm.mk_var("y", tm.sorts.int_sort);
    let x_squared = tm.mk_mul(vec![x2, x2]);
    let nonlinear_body = tm.mk_eq(x_squared, y2);

    let nonlinear_exists = tm.mk_exists(vec![("x", tm.sorts.int_sort)], nonlinear_body);

    println!("Formula: exists x. (x^2 = y)");
    println!("  This is non-linear (x*x = y)");
    println!("  QE-Lite is incomplete for non-linear arithmetic");

    let result4 = qe_solver.eliminate(nonlinear_exists, &mut tm);
    print_qe_lite_result("exists x. (x^2 = y)", &result4);
    println!("  Full QE would produce: y >= 0 AND exists k. (y = k^2)");
    println!();

    // ===== Example 11: Nested Quantifiers =====
    println!("--- Example 11: Nested Quantifiers ---");

    println!("Formula: forall x. exists y. (y > x)");
    println!("  Meaning: for every x, there exists a y greater than x");
    println!("  This is TRUE in integers (y = x + 1 works)");
    println!();
    println!("Nested quantifiers require iterative elimination:");
    println!("  1. Eliminate innermost quantifier first");
    println!("  2. Propagate result to outer quantifiers");
    println!("  3. Continue until all quantifiers are eliminated");
    println!();

    println!("=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. QE transforms quantified formulas to quantifier-free form");
    println!("  2. QE-Lite provides fast but incomplete elimination");
    println!("  3. Different algorithms suit different arithmetic theories");
    println!("  4. Statistics help measure QE effectiveness");
    println!("  5. Non-linear formulas may require specialized algorithms");
    println!("  6. Universal quantifiers: forall x. phi = NOT exists x. NOT phi");
}

/// Print QE Lite result in a human-readable format
fn print_qe_lite_result(formula: &str, result: &QeLiteResult) {
    println!("QE for: {}", formula);
    match result {
        QeLiteResult::Eliminated(term) => {
            println!("  Result: ELIMINATED - quantifier removed");
            println!("  Result term: {:?}", term);
        }
        QeLiteResult::Simplified(term) => {
            println!("  Result: SIMPLIFIED - partially reduced");
            println!("  Result term: {:?}", term);
        }
        QeLiteResult::Unchanged => {
            println!("  Result: UNCHANGED - no elimination possible");
        }
        QeLiteResult::Error(msg) => {
            println!("  Result: ERROR - {}", msg);
        }
    }
}
