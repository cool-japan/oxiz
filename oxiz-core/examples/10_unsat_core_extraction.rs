//! # UNSAT Core Extraction Example
//!
//! This example demonstrates unsatisfiable core extraction.
//! It covers:
//! - Computing minimal unsatisfiable subsets
//! - Tracking assertion origins
//! - Different core extraction strategies
//! - Core minimization algorithms
//! - Applications (debugging, abstraction refinement)
//!
//! ## UNSAT Cores
//! An unsatisfiable core is a minimal subset of assertions that is still
//! unsatisfiable. Cores help identify why a formula is unsatisfiable and
//! are used in verification, test case reduction, and debugging.
//!
//! ## Complexity
//! - Linear search: O(n) SAT calls where n is assertion count
//! - Deletion-based: O(n log n) SAT calls
//! - QuickXplain: O(n log n) SAT calls (average)
//!
//! ## See Also
//! - [`UnsatCore`](oxiz_core::unsat_core::UnsatCore)
//! - [`UnsatCoreBuilder`](oxiz_core::unsat_core::UnsatCoreBuilder)

use num_bigint::BigInt;
use oxiz_core::ast::TermManager;
use oxiz_core::unsat_core::{UnsatCore, UnsatCoreBuilder, UnsatCoreStrategy};

fn main() {
    println!("=== OxiZ Core: UNSAT Core Extraction ===\n");

    let mut tm = TermManager::new();

    // ===== Example 1: Simple UNSAT Core =====
    println!("--- Example 1: Simple UNSAT Core ---");

    // Create contradictory assertions
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let zero = tm.mk_int(BigInt::from(0));
    let ten = tm.mk_int(BigInt::from(10));

    // Assertions:
    // 1. x > 10
    // 2. x < 0
    // 3. x >= 0 (redundant for UNSAT, but makes it clearer)
    let a1 = tm.mk_gt(x, ten);
    let a2 = tm.mk_lt(x, zero);
    let a3 = tm.mk_ge(x, zero);

    let assertions = vec![
        ("A1", a1), // x > 10
        ("A2", a2), // x < 0
        ("A3", a3), // x >= 0
    ];

    println!("Assertions:");
    for (name, term) in &assertions {
        println!("  {}: {:?}", name, term);
    }

    // Build UNSAT core
    let mut builder = UnsatCoreBuilder::new(UnsatCoreStrategy::DeletionBased);
    for (name, term) in &assertions {
        builder.add_assertion(name.to_string(), *term);
    }

    println!("\nComputing UNSAT core...");
    // In a real implementation, this would call the solver
    // For this example, we'll construct a core manually
    let core = UnsatCore {
        assertions: vec!["A1".to_string(), "A2".to_string()],
        is_minimal: true,
        size: 2,
    };

    println!("UNSAT Core: {:?}", core.assertions);
    println!(
        "Size: {} (out of {} assertions)",
        core.size,
        assertions.len()
    );
    println!("Is minimal: {}", core.is_minimal);
    println!("\nExplanation: x > 10 and x < 0 are contradictory");

    // ===== Example 2: Larger UNSAT Core =====
    println!("\n--- Example 2: Larger UNSAT Core ---");

    let y = tm.mk_var("y", tm.sorts.int_sort);
    let z = tm.mk_var("z", tm.sorts.int_sort);
    let one = tm.mk_int(BigInt::from(1));
    let five = tm.mk_int(BigInt::from(5));

    // System of constraints:
    // B1: y = z + 5
    // B2: z > 10
    // B3: y < 10
    // B4: y >= 0 (redundant)
    // B5: z >= 0 (redundant)

    let b1 = tm.mk_eq(y, tm.mk_add(vec![z, five]));
    let b2 = tm.mk_gt(z, ten);
    let b3 = tm.mk_lt(y, ten);
    let b4 = tm.mk_ge(y, zero);
    let b5 = tm.mk_ge(z, zero);

    let large_assertions = vec![("B1", b1), ("B2", b2), ("B3", b3), ("B4", b4), ("B5", b5)];

    println!("Assertions:");
    for (name, _) in &large_assertions {
        println!("  {}", name);
    }

    // Core would be: B1, B2, B3 (y = z + 5, z > 10, y < 10)
    let large_core = UnsatCore {
        assertions: vec!["B1".to_string(), "B2".to_string(), "B3".to_string()],
        is_minimal: true,
        size: 3,
    };

    println!("\nUNSAT Core: {:?}", large_core.assertions);
    println!("Explanation: If z > 10, then y = z + 5 > 15, contradicting y < 10");

    // ===== Example 3: Core Extraction Strategies =====
    println!("\n--- Example 3: Core Extraction Strategies ---");

    println!("Strategy: Deletion-Based");
    println!("  Algorithm: Remove assertions one by one");
    println!("  Complexity: O(n) SAT calls");
    println!("  Result: Minimal core (but not necessarily smallest)");

    println!("\nStrategy: QuickXplain");
    println!("  Algorithm: Divide-and-conquer on assertion set");
    println!("  Complexity: O(n log n) SAT calls (average)");
    println!("  Result: Minimal core");

    println!("\nStrategy: Linear Search");
    println!("  Algorithm: Add assertions until UNSAT");
    println!("  Complexity: O(n) SAT calls");
    println!("  Result: Not necessarily minimal");

    // ===== Example 4: Core Minimization =====
    println!("\n--- Example 4: Core Minimization ---");

    // Start with a non-minimal core
    let non_minimal = UnsatCore {
        assertions: vec![
            "A1".to_string(),
            "A2".to_string(),
            "A3".to_string(),
            "A4".to_string(),
        ],
        is_minimal: false,
        size: 4,
    };

    println!("Non-minimal core: {:?}", non_minimal.assertions);

    // Minimize
    let minimized = UnsatCore {
        assertions: vec!["A1".to_string(), "A2".to_string()],
        is_minimal: true,
        size: 2,
    };

    println!("After minimization: {:?}", minimized.assertions);
    println!(
        "Reduced from {} to {} assertions",
        non_minimal.size, minimized.size
    );

    // ===== Example 5: Application - Debugging =====
    println!("\n--- Example 5: Application - Debugging Specifications ---");

    println!("Specification with 20 assertions is UNSAT");
    println!("UNSAT core identifies problematic subset:");
    println!("  Original: 20 assertions");
    println!("  Core: 3 assertions (assertions #5, #12, #18)");
    println!("\nBenefit: Focus debugging on 3 assertions instead of 20");

    // ===== Example 6: Application - Abstraction Refinement =====
    println!("\n--- Example 6: Application - Abstraction Refinement (CEGAR) ---");

    println!("Counter-Example Guided Abstraction Refinement:");
    println!("  1. Abstract model is checked");
    println!("  2. Spurious counterexample found");
    println!("  3. UNSAT core identifies blocking clause");
    println!("  4. Refinement based on core");
    println!("\nUNSAT core guides refinement, avoiding over-refinement");

    // ===== Example 7: Multiple Cores =====
    println!("\n--- Example 7: Multiple UNSAT Cores ---");

    println!("For formula F = C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5:");
    println!("  Core 1: {C1, C2, C3}");
    println!("  Core 2: {C1, C4, C5}");
    println!("  Core 3: {C2, C4}");
    println!("\nDifferent cores highlight different reasons for UNSAT");
    println!("Intersection of all cores: {C1, C2, C4} (essential constraints)");

    // ===== Example 8: Proof-Based Core Extraction =====
    println!("\n--- Example 8: Proof-Based Core Extraction ---");

    println!("Resolution proof:");
    println!("  1. Derive conflict from assertions");
    println!("  2. Trace back used clauses");
    println!("  3. Core = assertions used in proof");
    println!("\nAdvantage: Core computed during proof construction (no extra SAT calls)");

    // ===== Example 9: Core Statistics =====
    println!("\n--- Example 9: UNSAT Core Statistics ---");

    let stats = UnsatCoreStats {
        total_assertions: 50,
        core_size: 7,
        minimization_iterations: 12,
        solver_calls: 23,
        time_ms: 145,
    };

    println!("Core extraction statistics:");
    println!("  Total assertions: {}", stats.total_assertions);
    println!("  Core size: {}", stats.core_size);
    println!(
        "  Reduction: {:.1}%",
        100.0 * (stats.total_assertions - stats.core_size) as f64 / stats.total_assertions as f64
    );
    println!(
        "  Minimization iterations: {}",
        stats.minimization_iterations
    );
    println!("  Solver calls: {}", stats.solver_calls);
    println!("  Time: {} ms", stats.time_ms);

    // ===== Example 10: Core Visualization =====
    println!("\n--- Example 10: Core Visualization ---");

    println!("Assertions and their participation in core:\n");
    let assertions_viz = vec![
        ("x > 0", true),
        ("x < 10", true),
        ("y = x + 5", true),
        ("y < 5", false),
        ("z >= 0", false),
        ("z < y", false),
    ];

    for (assertion, in_core) in assertions_viz {
        let marker = if in_core { "[*]" } else { "[ ]" };
        println!("  {} {}", marker, assertion);
    }

    println!("\n[*] = In UNSAT core");
    println!("[ ] = Not in UNSAT core (can be removed without affecting UNSAT)");

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. UNSAT cores identify minimal unsatisfiable subsets");
    println!("  2. Cores aid in debugging unsatisfiable formulas");
    println!("  3. Different strategies trade off minimality vs. performance");
    println!("  4. Proof-based extraction is efficient");
    println!("  5. Cores guide abstraction refinement in CEGAR");
    println!("  6. Minimization reduces core size further");
}

// Helper struct for example
#[derive(Debug)]
struct UnsatCoreStats {
    total_assertions: usize,
    core_size: usize,
    minimization_iterations: usize,
    solver_calls: usize,
    time_ms: u64,
}
