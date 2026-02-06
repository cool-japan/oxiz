//! # Theory Combination Example
//!
//! This example demonstrates theory combination using Nelson-Oppen method.
//! It covers:
//! - Combining multiple theories (EUF + LIA, Arrays + EUF)
//! - Equality propagation between theories
//! - Delayed theory combination
//! - Theory conflict resolution
//! - Performance characteristics
//!
//! ## Nelson-Oppen Method
//! The Nelson-Oppen method enables modular combination of decision procedures
//! for different theories by exchanging equality information.
//!
//! ## Complexity
//! - Equality propagation: O(n^2) worst case
//! - Incremental propagation: O(n log n) amortized
//!
//! ## See Also
//! - [`NelsonOppenCombiner`](oxiz_solver::NelsonOppenCombiner)
//! - [`TheoryCombination`](oxiz_theories::TheoryCombination)
//!
//! Note: This example is a placeholder. The full theory combination API is
//! available in the oxiz_solver and oxiz_theories crates.

fn main() {
    println!("=== OxiZ Solver: Theory Combination ===\n");

    println!("Theory combination capabilities:");
    println!("  - oxiz_solver::Solver - Multi-theory solver");
    println!("  - oxiz_solver::combination - Nelson-Oppen module");
    println!("  - oxiz_solver::NelsonOppenCombiner - Explicit combiner");

    println!("\n--- Example: EUF + LIA ---");
    println!("  x = y");
    println!("  f(x) > 0");
    println!("  f(y) < 0");
    println!("  Result: UNSAT (x = y implies f(x) = f(y), contradiction)");

    println!("\n--- Example: Arrays + EUF ---");
    println!("  a = b");
    println!("  a[i] = 10");
    println!("  b[j] = 20");
    println!("  i = j");
    println!("  Result: UNSAT (a = b and i = j implies a[i] = b[j], but 10 ≠ 20)");

    println!("\n--- Combination Methods ---");
    println!("  Nelson-Oppen: Share equalities between convex theories");
    println!("  Delayed: Lazy theory interaction (more efficient)");
    println!("  Shostak: For theories with canonical forms");

    println!("\n--- Purification ---");
    println!("  Original: f(x + y) = z");
    println!("  Introduce fresh: t = x + y");
    println!("  Purified LIA: t = x + y");
    println!("  Purified EUF: f(t) = z");

    println!("\nSee the module documentation for detailed usage.");
}
