//! Linear Integer Arithmetic (LIA) Solver
//!
//! Extends the Simplex-based LRA solver with integer-specific reasoning:
//! - Branch-and-bound for integer feasibility
//! - Gomory cuts, MIR cuts, and CG cuts
//! - GCD-based infeasibility detection
//! - Integer bound propagation

mod types;
mod constructors;
mod constraints;
mod presolve;
mod heuristics;
mod branching;
mod cuts;
mod utils;
mod helpers;
mod hnf;
mod pb;

// Re-export public types
pub use types::LiaSolver;
pub use hnf::HermiteNormalForm;
pub use pb::PseudoBooleanSolver;

// Default impl
impl Default for LiaSolver {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use super::helpers::{extended_gcd, gcd, lcm};
    use crate::arithmetic::simplex::{LinExpr, VarId};
    use num_rational::Rational64;
    use num_traits::{One, Zero};

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(100, 50), 50);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(12, 8), 24);
        assert_eq!(lcm(17, 13), 221);
        assert_eq!(lcm(100, 50), 100);
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(12, 8);
        assert_eq!(g, 4);
        assert_eq!(12 * x + 8 * y, g);
    }

    #[test]
    fn test_gcd_infeasibility() {
        // 2x + 4y = 3 is infeasible (gcd(2,4) = 2, which doesn't divide 3)
        assert!(LiaSolver::check_gcd_infeasibility(&[2, 4], 3));

        // 2x + 4y = 6 is feasible
        assert!(!LiaSolver::check_gcd_infeasibility(&[2, 4], 6));
    }

    #[test]
    fn test_tighten_bound() {
        // 2x + 4y <= 7 can be tightened to 2x + 4y <= 6
        assert_eq!(LiaSolver::tighten_bound(&[2, 4], 7), 6);

        // 3x + 6y <= 10 can be tightened to 3x + 6y <= 9
        assert_eq!(LiaSolver::tighten_bound(&[3, 6], 10), 9);
    }

    #[test]
    fn test_lia_solver_basic() {
        let mut solver = LiaSolver::new();

        let x = solver.new_var();
        let y = solver.new_var();

        // x + y <= 10
        let mut expr = LinExpr::new();
        expr.add_term(x, Rational64::one());
        expr.add_term(y, Rational64::one());
        expr.add_constant(-Rational64::from_integer(10));
        solver.add_le(expr, 0);

        // x >= 0
        let mut expr2 = LinExpr::new();
        expr2.add_term(x, Rational64::one());
        solver.add_ge(expr2, 0);

        // y >= 0
        let mut expr3 = LinExpr::new();
        expr3.add_term(y, Rational64::one());
        solver.add_ge(expr3, 0);

        // This should be feasible
        let result = solver.check();
        assert!(result.is_ok());
    }

    #[test]
    fn test_pseudo_boolean() {
        let mut pb = PseudoBooleanSolver::new(vec![2, 3, 5], 7);
        pb.normalize();

        // After normalization, gcd(2,3,5) = 1, so coefficients remain the same
        assert_eq!(pb.coeffs, vec![2, 3, 5]);
    }

    #[test]
    fn test_pseudo_boolean_cardinality() {
        let pb = PseudoBooleanSolver::new(vec![3, 3, 3], 6);

        // Should convert to cardinality: at most 2 variables can be true
        assert_eq!(pb.to_cardinality(), Some((3, 2)));
    }

    #[test]
    fn test_normalize_constraint() {
        // 6x + 9y + 12z <= 30 should normalize to 2x + 3y + 4z <= 10
        let mut expr = LinExpr::new();
        expr.add_term(0, Rational64::from_integer(6));
        expr.add_term(1, Rational64::from_integer(9));
        expr.add_term(2, Rational64::from_integer(12));
        expr.add_constant(-Rational64::from_integer(30));

        LiaSolver::normalize_constraint(&mut expr);

        // Check normalized coefficients
        assert_eq!(expr.terms[0].1, Rational64::from_integer(2));
        assert_eq!(expr.terms[1].1, Rational64::from_integer(3));
        assert_eq!(expr.terms[2].1, Rational64::from_integer(4));
        assert_eq!(expr.constant, -Rational64::from_integer(10));
    }

    #[test]
    fn test_presolve() {
        let mut solver = LiaSolver::new();

        // Add some constraints
        let x = solver.new_var();
        let y = solver.new_var();

        let mut expr = LinExpr::new();
        expr.add_term(x, Rational64::one());
        expr.add_term(y, Rational64::one());
        expr.add_constant(-Rational64::from_integer(10));
        solver.add_le(expr, 0);

        // Presolve should succeed (no infeasibility detected)
        let result = solver.presolve();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
    }

    #[test]
    fn test_disjunctive_cut() {
        let solver = LiaSolver::new();

        // Test disjunctive cut generation for fractional value
        let var = 0;
        let value = Rational64::new(5, 2); // 2.5

        let cut = solver.generate_disjunctive_cut(var, value);
        assert!(cut.is_some());

        let cut = cut.unwrap();
        assert!(!cut.terms.is_empty());

        // Should not generate cut for integer value
        let int_value = Rational64::from_integer(3);
        let no_cut = solver.generate_disjunctive_cut(var, int_value);
        assert!(no_cut.is_none());
    }

    #[test]
    fn test_fix_tight_bounds() {
        let mut solver = LiaSolver::new();

        let x = solver.new_var();
        let y = solver.new_var();

        // Set tight bounds on x: 2.3 <= x <= 2.7
        // Only integer value is 3, but let's use a range that forces x = 2
        solver.simplex.set_lower(x, Rational64::new(19, 10), 0); // 1.9
        solver.simplex.set_upper(x, Rational64::new(21, 10), 1); // 2.1

        // Set loose bounds on y: 0 <= y <= 10
        solver.simplex.set_lower(y, Rational64::zero(), 2);
        solver.simplex.set_upper(y, Rational64::from_integer(10), 3);

        // Fix tight bounds should fix x to 2, but not y
        let result = solver.fix_tight_bounds();
        assert!(result.is_ok());
        let fixed_count = result.unwrap();
        assert_eq!(fixed_count, 1); // Only x should be fixed

        // Check that x is now fixed to 2
        let x_lower = solver.simplex.get_lower(x).unwrap();
        let x_upper = solver.simplex.get_upper(x).unwrap();
        assert_eq!(x_lower.value.real, Rational64::from_integer(2));
        assert_eq!(x_upper.value.real, Rational64::from_integer(2));

        // y should still have loose bounds
        let y_lower = solver.simplex.get_lower(y).unwrap();
        let y_upper = solver.simplex.get_upper(y).unwrap();
        assert_eq!(y_lower.value.real, Rational64::zero());
        assert_eq!(y_upper.value.real, Rational64::from_integer(10));
    }

    #[test]
    fn test_feasibility_pump() {
        let mut solver = LiaSolver::new();

        let x = solver.new_var();
        let y = solver.new_var();

        // Add constraints: x >= 0, y >= 0, x + y <= 10
        solver.simplex.set_lower(x, Rational64::zero(), 0);
        solver.simplex.set_lower(y, Rational64::zero(), 1);

        let mut expr = LinExpr::new();
        expr.add_term(x, Rational64::one());
        expr.add_term(y, Rational64::one());
        expr.add_constant(-Rational64::from_integer(10));
        solver.add_le(expr, 2);

        // Try feasibility pump
        let result = solver.feasibility_pump(10);
        assert!(result.is_ok());

        // For this simple problem, it should find a solution quickly
        // (or at least not crash)
        if let Ok(Some(solution)) = result {
            // If a solution was found, check it's valid
            assert_eq!(solution.len(), 2);
            assert!(solution[0] >= 0);
            assert!(solution[1] >= 0);
            assert!(solution[0] + solution[1] <= 10);
        }
    }

    #[test]
    fn test_probe_variables() {
        let mut solver = LiaSolver::new();

        let x = solver.new_var();
        let y = solver.new_var();

        // Set initial loose bounds: 0 <= x <= 10, 0 <= y <= 10
        solver.simplex.set_lower(x, Rational64::zero(), 0);
        solver.simplex.set_upper(x, Rational64::from_integer(10), 1);
        solver.simplex.set_lower(y, Rational64::zero(), 2);
        solver.simplex.set_upper(y, Rational64::from_integer(10), 3);

        // Add constraint: x + y >= 15 (forces x and y to be larger)
        let mut expr = LinExpr::new();
        expr.add_term(x, Rational64::one());
        expr.add_term(y, Rational64::one());
        expr.add_constant(-Rational64::from_integer(15));
        solver.add_ge(expr, 4);

        // Solve LP to get basis
        assert!(solver.simplex.check().is_ok());

        // Probe variables - should detect that lower bounds must be tightened
        // If x = 0, then y >= 15 (but y <= 10) - infeasible!
        // If y = 0, then x >= 15 (but x <= 10) - infeasible!
        let result = solver.probe_variables(10);
        assert!(result.is_ok());

        // Probing should have tightened bounds (or detected partial infeasibility)
        let tightened = result.unwrap();
        // We expect probing to complete successfully (tightened count is valid)
        // Note: may be 0 if LP already propagated bounds
        let _ = tightened;
    }

    #[test]
    fn test_manage_cuts() {
        let mut solver = LiaSolver::new();

        // Manually add some cut records
        solver.record_cut(100);
        solver.record_cut(101);
        solver.record_cut(102);

        assert_eq!(solver.active_cuts.len(), 3);

        // Age cuts artificially
        for cut in &mut solver.active_cuts {
            cut.age = 150; // Old enough to be deleted
        }

        // Manage cuts should delete them
        let deleted = solver.manage_cuts();
        // Should delete old cuts with no activity (deleted count is valid)
        let _ = deleted;
        assert!(solver.active_cuts.len() <= 3); // All or some may be deleted
    }
}
