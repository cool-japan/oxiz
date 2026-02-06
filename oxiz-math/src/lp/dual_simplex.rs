//! Dual Simplex Algorithm for Linear Programming.
//!
//! Implements the dual simplex method for solving LPs, particularly useful
//! for handling infeasibility and for sensitivity analysis.

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use rustc_hash::FxHashMap;

/// Dual simplex solver for linear programming.
pub struct DualSimplexSolver {
    /// Current tableau
    tableau: Vec<Vec<BigRational>>,
    /// Basic variables
    basis: Vec<VarId>,
    /// Non-basic variables
    non_basis: Vec<VarId>,
    /// Objective row
    objective: Vec<BigRational>,
    /// Statistics
    stats: DualSimplexStats,
}

/// Variable identifier
pub type VarId = usize;

/// Dual simplex statistics
#[derive(Debug, Clone, Default)]
pub struct DualSimplexStats {
    /// Number of iterations
    pub iterations: usize,
    /// Number of pivot operations
    pub pivots: usize,
    /// Number of ratio tests
    pub ratio_tests: usize,
}

/// Result of dual simplex solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DualSimplexResult {
    /// Optimal solution found
    Optimal,
    /// Problem is infeasible
    Infeasible,
    /// Problem is unbounded
    Unbounded,
    /// Unknown (iteration limit reached)
    Unknown,
}

impl DualSimplexSolver {
    /// Create a new dual simplex solver.
    pub fn new(num_vars: usize, num_constraints: usize) -> Self {
        Self {
            tableau: vec![vec![BigRational::zero(); num_vars + 1]; num_constraints],
            basis: (0..num_constraints).collect(),
            non_basis: (0..num_vars).collect(),
            objective: vec![BigRational::zero(); num_vars + 1],
            stats: DualSimplexStats::default(),
        }
    }

    /// Solve using dual simplex method.
    pub fn solve(&mut self) -> DualSimplexResult {
        const MAX_ITERATIONS: usize = 10000;

        for _ in 0..MAX_ITERATIONS {
            self.stats.iterations += 1;

            // Check for optimality: all RHS values non-negative
            if self.is_dual_feasible() {
                return DualSimplexResult::Optimal;
            }

            // Select leaving variable (most negative RHS)
            let leaving_idx = match self.select_leaving_variable() {
                Some(idx) => idx,
                None => return DualSimplexResult::Infeasible,
            };

            // Select entering variable (dual ratio test)
            let entering_idx = match self.select_entering_variable(leaving_idx) {
                Some(idx) => idx,
                None => return DualSimplexResult::Unbounded,
            };

            // Perform pivot
            self.pivot(leaving_idx, entering_idx);
            self.stats.pivots += 1;
        }

        DualSimplexResult::Unknown
    }

    /// Check if current solution is dual feasible.
    fn is_dual_feasible(&self) -> bool {
        // All RHS values (last column) should be non-negative
        for row in &self.tableau {
            if let Some(rhs) = row.last()
                && rhs < &BigRational::zero()
            {
                return false;
            }
        }
        true
    }

    /// Select leaving variable (row with most negative RHS).
    fn select_leaving_variable(&self) -> Option<usize> {
        let mut min_rhs = BigRational::zero();
        let mut leaving_idx = None;

        for (i, row) in self.tableau.iter().enumerate() {
            if let Some(rhs) = row.last()
                && rhs < &min_rhs
            {
                min_rhs = rhs.clone();
                leaving_idx = Some(i);
            }
        }

        leaving_idx
    }

    /// Select entering variable using dual ratio test.
    fn select_entering_variable(&mut self, leaving_row: usize) -> Option<usize> {
        self.stats.ratio_tests += 1;

        let mut min_ratio = None;
        let mut entering_idx = None;

        let leaving_row_vec = &self.tableau[leaving_row];

        for (j, coeff) in leaving_row_vec
            .iter()
            .enumerate()
            .take(leaving_row_vec.len() - 1)
        {
            // Only consider negative coefficients
            if coeff >= &BigRational::zero() {
                continue;
            }

            // Compute ratio: objective[j] / |coeff|
            let obj_coeff = &self.objective[j];
            let ratio = obj_coeff / coeff.abs();

            match &min_ratio {
                None => {
                    min_ratio = Some(ratio.clone());
                    entering_idx = Some(j);
                }
                Some(current_min) => {
                    if ratio < *current_min {
                        min_ratio = Some(ratio);
                        entering_idx = Some(j);
                    }
                }
            }
        }

        entering_idx
    }

    /// Perform pivot operation.
    fn pivot(&mut self, leaving_row: usize, entering_col: usize) {
        let pivot_element = self.tableau[leaving_row][entering_col].clone();

        if pivot_element.is_zero() {
            return; // Degenerate pivot
        }

        // Normalize pivot row
        for elem in &mut self.tableau[leaving_row] {
            *elem = &*elem / &pivot_element;
        }

        // Eliminate other rows
        for i in 0..self.tableau.len() {
            if i == leaving_row {
                continue;
            }

            let multiplier = self.tableau[i][entering_col].clone();
            for j in 0..self.tableau[i].len() {
                let pivot_row_elem = &self.tableau[leaving_row][j];
                self.tableau[i][j] = &self.tableau[i][j] - &multiplier * pivot_row_elem;
            }
        }

        // Update objective row
        let obj_multiplier = self.objective[entering_col].clone();
        for j in 0..self.objective.len() {
            let pivot_row_elem = &self.tableau[leaving_row][j];
            self.objective[j] = &self.objective[j] - &obj_multiplier * pivot_row_elem;
        }

        // Update basis
        self.basis[leaving_row] = self.non_basis[entering_col];
        self.non_basis[entering_col] = leaving_row;
    }

    /// Get current solution.
    pub fn get_solution(&self) -> FxHashMap<VarId, BigRational> {
        let mut solution = FxHashMap::default();

        for (i, &var_id) in self.basis.iter().enumerate() {
            if let Some(rhs) = self.tableau[i].last() {
                solution.insert(var_id, rhs.clone());
            }
        }

        solution
    }

    /// Get objective value.
    pub fn get_objective_value(&self) -> BigRational {
        self.objective
            .last()
            .cloned()
            .unwrap_or_else(BigRational::zero)
    }

    /// Get statistics.
    pub fn stats(&self) -> &DualSimplexStats {
        &self.stats
    }

    /// Add constraint to tableau.
    pub fn add_constraint(&mut self, coeffs: Vec<BigRational>, rhs: BigRational) {
        let mut row = coeffs;
        row.push(rhs);
        self.tableau.push(row);
    }

    /// Set objective function.
    pub fn set_objective(&mut self, coeffs: Vec<BigRational>) {
        self.objective = coeffs;
        self.objective.push(BigRational::zero());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_simplex_creation() {
        let solver = DualSimplexSolver::new(3, 2);
        assert_eq!(solver.stats.iterations, 0);
    }

    #[test]
    fn test_is_dual_feasible() {
        let solver = DualSimplexSolver::new(2, 1);
        assert!(solver.is_dual_feasible());
    }
}
