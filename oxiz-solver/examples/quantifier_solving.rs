//! # Quantifier Solving Example
//!
//! This example demonstrates solving formulas with quantifiers.
//! It covers:
//! - Model-Based Quantifier Instantiation (MBQI)
//! - E-matching for pattern-based instantiation
//! - Quantifier elimination preprocessing
//! - Herbrand universe construction
//! - Skolemization
//!
//! ## Quantified Formulas
//! Quantifiers (∀, ∃) make SMT solving undecidable in general.
//! MBQI and E-matching provide incomplete but practical approaches.
//!
//! ## Complexity
//! - MBQI: Potentially non-terminating (semi-decidable)
//! - E-matching: O(n^k) where k is pattern size
//! - QE-lite: O(n^2) heuristic
//!
//! ## See Also
//! - [`MBQISolver`](oxiz_solver::MBQISolver) for model-based instantiation
//! - [`QeLiteSolver`](oxiz_core::qe::QeLiteSolver) for quantifier elimination
//!
//! Note: This example is a placeholder. The full quantifier solving API is
//! available in the oxiz_solver module.

fn main() {
    println!("=== OxiZ Solver: Quantifier Solving ===\n");

    println!("Quantifier solving capabilities:");
    println!("  - oxiz_solver::Solver - Main solver interface");
    println!("  - oxiz_solver::MBQISolver - Model-based quantifier instantiation");
    println!("  - oxiz_core::ast::TermManager::mk_forall - Create universal quantifiers");
    println!("  - oxiz_core::ast::TermManager::mk_exists - Create existential quantifiers");

    println!("\n--- Quantifier Types ---");
    println!("  Universal (∀): ∀x. P(x) - for all x, P(x) holds");
    println!("  Existential (∃): ∃x. P(x) - there exists x such that P(x) holds");

    println!("\n--- Techniques ---");
    println!("  1. Skolemization: Replace ∃x with fresh Skolem constant");
    println!("  2. MBQI: Model-guided instantiation");
    println!("  3. E-matching: Pattern-based instantiation");
    println!("  4. QE (Quantifier Elimination): Remove quantifiers entirely");

    println!("\n--- Decidability ---");
    println!("  QF_LIA: Decidable (quantifier-free linear integer arithmetic)");
    println!("  LIA: Undecidable in general (with quantifiers)");
    println!("  Presburger: Decidable (linear arithmetic without multiplication)");

    println!("\nSee the module documentation for detailed usage.");
}
