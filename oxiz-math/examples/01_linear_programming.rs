//! # Linear Programming Example
//!
//! This example demonstrates linear programming algorithms.
//! It covers:
//! - Simplex method (primal and dual)
//! - Feasibility checking
//! - Optimization (min/max)
//! - Sensitivity analysis
//! - Integer linear programming basics
//!
//! ## Linear Programming
//! Optimize a linear objective subject to linear constraints.
//! Forms the basis for LRA and LIA theory solving in SMT.
//!
//! ## Complexity
//! - Simplex: O(2^n) worst case, polynomial average
//! - Interior point: O(n^3.5) polynomial
//! - Branch-and-bound (ILP): Exponential
//!
//! ## See Also
//! - [`LPSolver`](oxiz_math::lp_core::LPSolver)
//! - [`DualSimplexSolver`](oxiz_math::lp::DualSimplexSolver)
//!
//! Note: This example is a placeholder. The full LP API is available in
//! oxiz_math::lp and oxiz_math::lp_core modules.

fn main() {
    println!("=== OxiZ Math: Linear Programming ===\n");
    println!("Linear programming modules available:");
    println!("  - oxiz_math::lp_core::LPSolver - Full LP solver");
    println!("  - oxiz_math::lp::DualSimplexSolver - Dual simplex method");
    println!("  - oxiz_math::lp::BranchCutSolver - Branch and cut (MIP)");
    println!("  - oxiz_math::lp::CuttingPlaneGenerator - Cutting plane methods");
    println!("  - oxiz_math::lp::FarkasGenerator - Infeasibility certificates");
    println!("\nSee the module documentation for detailed usage.");
}
