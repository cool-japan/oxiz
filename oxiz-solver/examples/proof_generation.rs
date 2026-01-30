//! # Proof Generation Example
//!
//! This example demonstrates proof generation and verification.
//! It covers:
//! - Resolution proofs for UNSAT results
//! - Proof checking and validation
//! - Proof formats (native, DRAT, LFSC, Lean)
//! - Proof minimization
//! - Certificate generation for verification
//!
//! ## Proof Generation
//! Proofs provide independently checkable certificates of correctness.
//! They're essential for formal verification and high-assurance systems.
//!
//! ## Complexity
//! - Proof generation: O(n) overhead during solving
//! - Proof checking: O(m) where m is proof size
//! - Minimization: O(m^2) worst case
//!
//! ## See Also
//! - [`Proof`](oxiz_solver::Proof) for proof representation
//! - [`ProofChecker`](oxiz_proof) for verification
//!
//! Note: This example is a placeholder. The full proof generation API is
//! available in the oxiz_solver and oxiz_proof crates.

fn main() {
    println!("=== OxiZ Solver: Proof Generation ===\n");

    println!("Proof generation capabilities:");
    println!("  - oxiz_solver::Solver::enable_proof_generation()");
    println!("  - oxiz_solver::Solver::get_proof()");
    println!("  - oxiz_proof::ProofTree - Proof tree representation");
    println!("  - oxiz_proof::coq_export - Export to Coq");
    println!("  - oxiz_proof::lean_export - Export to Lean 4");

    println!("\n--- Resolution Proofs ---");
    println!("  Example: p ∧ ¬p is UNSAT");
    println!("    1. p          [assumption C1]");
    println!("    2. ¬p         [assumption C2]");
    println!("    3. ⊥          [resolution 1, 2]");

    println!("\n--- Theory Lemmas ---");
    println!("  x > 10 ∧ x < 0 is UNSAT");
    println!("    1. x > 10     [assumption]");
    println!("    2. x < 0      [assumption]");
    println!("    3. x ≥ 11     [theory lemma from 1]");
    println!("    4. x ≤ -1     [theory lemma from 2]");
    println!("    5. ⊥          [theory conflict: 3, 4]");

    println!("\n--- Proof Formats ---");
    println!("  Native: Binary, optimized for performance");
    println!("  DRAT: Standard for SAT, checkable by drat-trim");
    println!("  LFSC: SMT-LIB standard, theory-aware");
    println!("  Lean/Coq: Interactive theorem prover format");

    println!("\n--- Proof Checking ---");
    println!("  ✓ All assumptions are asserted formulas");
    println!("  ✓ Resolution steps are valid");
    println!("  ✓ Theory lemmas are sound");
    println!("  ✓ Final step is ⊥ (contradiction)");

    println!("\nSee the module documentation for detailed usage.");
}
