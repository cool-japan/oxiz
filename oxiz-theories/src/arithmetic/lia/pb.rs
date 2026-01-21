//! Pseudo-Boolean constraint solver

use super::helpers::gcd;

/// Pseudo-Boolean constraint solver for 0-1 linear constraints
pub struct PseudoBooleanSolver {
    /// Coefficients
    pub coeffs: Vec<i64>,
    /// Bound
    pub bound: i64,
}

impl PseudoBooleanSolver {
    /// Create a new pseudo-Boolean solver
    #[must_use]
    pub fn new(coeffs: Vec<i64>, bound: i64) -> Self {
        Self { coeffs, bound }
    }

    /// Check if the constraint is satisfiable
    pub fn check(&self) -> bool {
        // Simple check: if sum of all positive coefficients <= bound, always SAT
        let sum_positive: i64 = self.coeffs.iter().filter(|&&c| c > 0).sum();

        if sum_positive <= self.bound {
            return true;
        }

        // If sum of all coefficients > bound when all vars are 1, might be UNSAT
        // This is a placeholder - full implementation would use specialized PB solving
        false
    }

    /// Normalize the constraint by dividing by GCD
    pub fn normalize(&mut self) {
        if self.coeffs.is_empty() {
            return;
        }

        let g = self.coeffs.iter().fold(0i64, |acc, &c| gcd(acc, c.abs()));

        if g > 1 {
            for c in &mut self.coeffs {
                *c /= g;
            }
            self.bound /= g;
        }
    }

    /// Convert to cardinality constraint if all coefficients are equal
    pub fn to_cardinality(&self) -> Option<(i64, i64)> {
        if self.coeffs.is_empty() {
            return None;
        }

        let first = self.coeffs[0].abs();
        if self.coeffs.iter().all(|&c| c.abs() == first) {
            Some((first, self.bound / first))
        } else {
            None
        }
    }
}

