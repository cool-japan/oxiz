//! # UNSAT Core Extraction Example
//!
//! This example demonstrates unsatisfiable core extraction.
//! It covers:
//! - Computing minimal unsatisfiable subsets
//! - Tracking assertion origins
//! - Core extraction strategies
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

use oxiz_core::ast::{NamedAssertion, TermManager};
use oxiz_core::unsat_core::{UnsatCore, UnsatCoreBuilder, UnsatCoreStrategy};

fn main() {
    println!("=== OxiZ Core: UNSAT Core Extraction ===\n");

    let mut tm = TermManager::new();

    // ===== Example 1: Simple UNSAT Core =====
    println!("--- Example 1: Simple UNSAT Core ---");

    // Create contradictory assertions
    let x = tm.mk_var("x", tm.sorts.int_sort);
    let zero = tm.mk_int(0);
    let ten = tm.mk_int(10);

    // Assertions:
    // 1. x > 10
    // 2. x < 0
    // 3. x >= 0 (redundant for UNSAT, but makes it clearer)
    let a1 = tm.mk_gt(x, ten);
    let a2 = tm.mk_lt(x, zero);
    let a3 = tm.mk_ge(x, zero);

    println!("Assertions:");
    println!("  A1 (x > 10): {:?}", a1);
    println!("  A2 (x < 0): {:?}", a2);
    println!("  A3 (x >= 0): {:?}", a3);

    // Build UNSAT core using the builder
    let mut builder = UnsatCoreBuilder::new();
    builder.add_named(a1, "A1");
    builder.add_named(a2, "A2");

    let core = builder.build();

    println!("\nUNSAT Core:");
    println!("  Size: {} assertions", core.len());
    println!("  Assertions: {:?}", core.names());
    println!("  Term IDs: {:?}", core.term_ids());
    println!("\nExplanation: x > 10 and x < 0 are contradictory");

    // ===== Example 2: Larger UNSAT Core =====
    println!("\n--- Example 2: Larger UNSAT Core ---");

    let y = tm.mk_var("y", tm.sorts.int_sort);
    let z = tm.mk_var("z", tm.sorts.int_sort);
    let five = tm.mk_int(5);

    // System of constraints:
    // B1: y = z + 5
    // B2: z > 10
    // B3: y < 10
    // B4: y >= 0 (redundant)
    // B5: z >= 0 (redundant)

    let z_plus_five = tm.mk_add(vec![z, five]);
    let b1 = tm.mk_eq(y, z_plus_five);
    let b2 = tm.mk_gt(z, ten);
    let b3 = tm.mk_lt(y, ten);
    let _b4 = tm.mk_ge(y, zero);
    let _b5 = tm.mk_ge(z, zero);

    println!("Assertions:");
    println!("  B1 (y = z + 5)");
    println!("  B2 (z > 10)");
    println!("  B3 (y < 10)");
    println!("  B4 (y >= 0)");
    println!("  B5 (z >= 0)");

    // Core would be: B1, B2, B3 (y = z + 5, z > 10, y < 10)
    let mut builder2 = UnsatCoreBuilder::new();
    builder2.add_named(b1, "B1");
    builder2.add_named(b2, "B2");
    builder2.add_named(b3, "B3");

    let large_core = builder2.build();

    println!("\nUNSAT Core: {:?}", large_core.names());
    println!("Explanation: If z > 10, then y = z + 5 > 15, contradicting y < 10");

    // ===== Example 3: Core Extraction Strategies =====
    println!("\n--- Example 3: Core Extraction Strategies ---");

    println!("UnsatCoreStrategy::All");
    println!("  Returns all assertions (no minimization)");

    println!("\nUnsatCoreStrategy::Deletion");
    println!("  Algorithm: Remove assertions one by one");
    println!("  Complexity: O(n) SAT calls");
    println!("  Result: Minimal core (but not necessarily smallest)");

    println!("\nUnsatCoreStrategy::QuickXplain");
    println!("  Algorithm: Divide-and-conquer on assertion set");
    println!("  Complexity: O(n log n) SAT calls (average)");
    println!("  Result: Minimal core");

    // Display strategy enum values
    println!("\nAvailable strategies:");
    println!("  {:?}", UnsatCoreStrategy::All);
    println!("  {:?}", UnsatCoreStrategy::Deletion);
    println!("  {:?}", UnsatCoreStrategy::QuickXplain);

    // ===== Example 4: UnsatCore Operations =====
    println!("\n--- Example 4: UnsatCore Operations ---");

    let mut core = UnsatCore::empty();
    println!("Created empty core: size = {}", core.len());

    // Add assertions
    core.add(NamedAssertion::named(a1, "A1"));
    core.add(NamedAssertion::named(a2, "A2"));
    core.add(NamedAssertion::unnamed(a3));

    println!("After adding 3 assertions: size = {}", core.len());

    // Query the core
    println!("\nCore queries:");
    println!("  Contains term {:?}? {}", a1, core.contains_term(a1));
    println!("  Contains name 'A1'? {}", core.contains_name("A1"));
    println!("  Contains name 'A5'? {}", core.contains_name("A5"));

    // ===== Example 5: Core Minimization =====
    println!("\n--- Example 5: Core Minimization ---");

    // Create a core with duplicates
    let mut dup_core = UnsatCore::empty();
    dup_core.add(NamedAssertion::unnamed(a1));
    dup_core.add(NamedAssertion::unnamed(a2));
    dup_core.add(NamedAssertion::unnamed(a1)); // duplicate

    println!("Core before minimization: {} assertions", dup_core.len());

    dup_core.minimize();
    println!("After minimization: {} assertions", dup_core.len());
    println!("(Duplicates removed)");

    // ===== Example 6: Application - Debugging =====
    println!("\n--- Example 6: Application - Debugging Specifications ---");

    println!("Specification with 20 assertions is UNSAT");
    println!("UNSAT core identifies problematic subset:");
    println!("  Original: 20 assertions");
    println!("  Core: 3 assertions");
    println!("\nBenefit: Focus debugging on 3 assertions instead of 20");

    // ===== Example 7: Application - Abstraction Refinement =====
    println!("\n--- Example 7: Application - Abstraction Refinement (CEGAR) ---");

    println!("Counter-Example Guided Abstraction Refinement:");
    println!("  1. Abstract model is checked");
    println!("  2. Spurious counterexample found");
    println!("  3. UNSAT core identifies blocking clause");
    println!("  4. Refinement based on core");
    println!("\nUNSAT core guides refinement, avoiding over-refinement");

    // ===== Example 8: Core Display =====
    println!("\n--- Example 8: Core Display Format ---");

    let mut display_core = UnsatCore::empty();
    display_core.add(NamedAssertion::named(a1, "assertion_x_gt_10"));
    display_core.add(NamedAssertion::named(a2, "assertion_x_lt_0"));
    display_core.add(NamedAssertion::unnamed(a3));

    println!("{}", display_core);

    // ===== Example 9: Multiple Cores =====
    println!("--- Example 9: Multiple UNSAT Cores ---");

    println!("For formula F = C1 AND C2 AND C3 AND C4 AND C5:");
    println!("  Core 1: {{C1, C2, C3}}");
    println!("  Core 2: {{C1, C4, C5}}");
    println!("  Core 3: {{C2, C4}}");
    println!("\nDifferent cores highlight different reasons for UNSAT");
    println!("Intersection of all cores reveals essential constraints");

    // ===== Example 10: Proof-Based Core Extraction =====
    println!("\n--- Example 10: Proof-Based Core Extraction ---");

    println!("Resolution proof:");
    println!("  1. Derive conflict from assertions");
    println!("  2. Trace back used clauses");
    println!("  3. Core = assertions used in proof");
    println!("\nAdvantage: Core computed during proof construction (no extra SAT calls)");

    // ===== Example 11: Core Visualization =====
    println!("\n--- Example 11: Core Visualization ---");

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
    println!("  6. UnsatCoreBuilder provides a fluent API for construction");
}
