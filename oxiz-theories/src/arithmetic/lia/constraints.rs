//! Constraint addition methods for LIA solver

use super::super::simplex::LinExpr;
use super::types::LiaSolver;

impl LiaSolver {
    /// Add a linear constraint: expr <= 0
    pub fn add_le(&mut self, expr: LinExpr, reason: u32) {
        self.simplex.add_le(expr, reason);
    }

    /// Add a linear constraint: expr >= 0
    pub fn add_ge(&mut self, expr: LinExpr, reason: u32) {
        self.simplex.add_ge(expr, reason);
    }

    /// Add an equality constraint: expr = 0
    ///
    /// For integer constraints, we check GCD-based infeasibility:
    /// If all coefficients share a common GCD that doesn't divide the constant,
    /// the constraint is infeasible over integers.
    ///
    /// Example: 2x + 2y = 7 is infeasible because gcd(2,2) = 2 doesn't divide 7.
    pub fn add_eq(&mut self, expr: LinExpr, reason: u32) {
        // Check for GCD-based infeasibility before adding the constraint
        // Extract integer coefficients from the expression
        let coeffs: Vec<i64> = expr
            .terms
            .iter()
            .filter_map(|(_, c)| {
                // Check if coefficient is an integer (denominator is 1)
                if c.denom() == &1 {
                    Some(*c.numer())
                } else {
                    None
                }
            })
            .collect();

        // Extract the constant term (negated RHS)
        // expr = 0 means sum(a_i * x_i) + constant = 0
        // which is sum(a_i * x_i) = -constant
        let rhs = if expr.constant.denom() == &1 {
            -*expr.constant.numer()
        } else {
            // Non-integer constant in equality - can't be satisfied by integers
            // Mark as infeasible by adding a contradictory constraint
            // x >= 1 and x <= 0 for some variable (if any exists)
            if let Some(&(var, _)) = expr.terms.first() {
                use num_rational::Rational64;
                self.simplex
                    .set_lower(var, Rational64::from_integer(1), reason);
                self.simplex
                    .set_upper(var, Rational64::from_integer(0), reason);
            }
            return;
        };

        // Only check GCD infeasibility if all coefficients are integers
        if !coeffs.is_empty()
            && coeffs.len() == expr.terms.len()
            && Self::check_gcd_infeasibility(&coeffs, rhs)
        {
            // Constraint is GCD-infeasible!
            // Add contradictory constraints to make the problem infeasible
            // This is the standard way to signal infeasibility in incremental solvers
            if let Some(&(var, _)) = expr.terms.first() {
                use num_rational::Rational64;
                // Add x >= 1 and x <= 0, which is clearly infeasible
                self.simplex
                    .set_lower(var, Rational64::from_integer(1), reason);
                self.simplex
                    .set_upper(var, Rational64::from_integer(0), reason);
            }
            return;
        }

        // No GCD infeasibility detected, add the constraint normally
        self.simplex.add_eq(expr, reason);
    }
}
