//! SIMD-Optimized Simplex Algorithm.
#![allow(clippy::needless_range_loop)] // Simplex algorithm uses explicit indexing
//!
//! Provides cache-friendly simplex operations for linear programming.

use num_traits::Float;
use std::ops::{Add, Div, Mul, Sub};

/// SIMD-friendly simplex tableau.
#[derive(Debug, Clone)]
pub struct SimplexTableau<T> {
    /// Tableau matrix (includes slack variables and RHS)
    pub tableau: Vec<Vec<T>>,
    /// Basic variable indices
    pub basis: Vec<usize>,
    /// Number of original variables
    pub num_vars: usize,
    /// Number of constraints
    pub num_constraints: usize,
}

impl<T> SimplexTableau<T>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Float,
{
    /// Create a new simplex tableau.
    pub fn new(constraints: Vec<Vec<T>>, objective: Vec<T>, rhs: Vec<T>) -> Self {
        let num_constraints = constraints.len();
        let num_vars = objective.len();

        // Build initial tableau with slack variables
        let total_vars = num_vars + num_constraints;
        let mut tableau = vec![vec![T::zero(); total_vars + 1]; num_constraints + 1];

        // Add constraints
        for (i, constraint) in constraints.iter().enumerate() {
            for (j, &val) in constraint.iter().enumerate() {
                tableau[i][j] = val;
            }
            // Add slack variable
            tableau[i][num_vars + i] = T::one();
            // Add RHS
            tableau[i][total_vars] = rhs[i];
        }

        // Add objective function (negated for maximization)
        for (j, &val) in objective.iter().enumerate() {
            tableau[num_constraints][j] = T::zero() - val;
        }

        // Initial basis is slack variables
        let basis = (num_vars..(num_vars + num_constraints)).collect();

        Self {
            tableau,
            basis,
            num_vars,
            num_constraints,
        }
    }

    /// Perform one iteration of the simplex algorithm.
    pub fn pivot(&mut self) -> Result<bool, SimplexError> {
        // Find entering variable (most negative coefficient in objective row)
        let entering_col = self.find_entering_variable()?;

        if entering_col.is_none() {
            return Ok(true); // Optimal solution found
        }

        let entering = entering_col.unwrap();

        // Find leaving variable using minimum ratio test
        let leaving_row = self.find_leaving_variable(entering)?;

        if leaving_row.is_none() {
            return Err(SimplexError::Unbounded);
        }

        let leaving = leaving_row.unwrap();

        // Perform pivot operation
        self.perform_pivot(entering, leaving);

        // Update basis
        self.basis[leaving] = entering;

        Ok(false) // Not optimal yet
    }

    /// Find entering variable (most negative coefficient).
    fn find_entering_variable(&self) -> Result<Option<usize>, SimplexError> {
        let obj_row = &self.tableau[self.num_constraints];
        let total_vars = self.num_vars + self.num_constraints;

        let mut min_val = T::zero();
        let mut min_idx = None;

        for j in 0..total_vars {
            if obj_row[j] < min_val {
                min_val = obj_row[j];
                min_idx = Some(j);
            }
        }

        Ok(min_idx)
    }

    /// Find leaving variable using minimum ratio test.
    fn find_leaving_variable(&self, entering: usize) -> Result<Option<usize>, SimplexError> {
        let total_vars = self.num_vars + self.num_constraints;
        let rhs_col = total_vars;

        let mut min_ratio = T::infinity();
        let mut min_idx = None;

        for i in 0..self.num_constraints {
            let coeff = self.tableau[i][entering];

            if coeff > T::epsilon() {
                let rhs = self.tableau[i][rhs_col];
                let ratio = rhs / coeff;

                if ratio < min_ratio {
                    min_ratio = ratio;
                    min_idx = Some(i);
                }
            }
        }

        Ok(min_idx)
    }

    /// Perform pivot operation with SIMD-friendly access pattern.
    fn perform_pivot(&mut self, entering: usize, leaving: usize) {
        let total_vars = self.num_vars + self.num_constraints;
        let num_cols = total_vars + 1;

        // Get pivot element
        let pivot = self.tableau[leaving][entering];

        if pivot.abs() < T::epsilon() {
            return; // Avoid division by zero
        }

        // Normalize pivot row
        for j in 0..num_cols {
            self.tableau[leaving][j] = self.tableau[leaving][j] / pivot;
        }

        // Eliminate column in other rows
        // Process in chunks for cache locality
        const CHUNK_SIZE: usize = 8;

        for i in 0..=self.num_constraints {
            if i == leaving {
                continue;
            }

            let factor = self.tableau[i][entering];

            if factor.abs() < T::epsilon() {
                continue;
            }

            // Process columns in chunks
            for chunk_start in (0..num_cols).step_by(CHUNK_SIZE) {
                let chunk_end = (chunk_start + CHUNK_SIZE).min(num_cols);

                for j in chunk_start..chunk_end {
                    let update = self.tableau[leaving][j] * factor;
                    self.tableau[i][j] = self.tableau[i][j] - update;
                }
            }
        }
    }

    /// Get current solution.
    pub fn get_solution(&self) -> Vec<T> {
        let mut solution = vec![T::zero(); self.num_vars];
        let total_vars = self.num_vars + self.num_constraints;
        let rhs_col = total_vars;

        for (row, &var_idx) in self.basis.iter().enumerate() {
            if var_idx < self.num_vars {
                solution[var_idx] = self.tableau[row][rhs_col];
            }
        }

        solution
    }

    /// Get objective value.
    pub fn get_objective_value(&self) -> T {
        let total_vars = self.num_vars + self.num_constraints;
        let rhs_col = total_vars;
        self.tableau[self.num_constraints][rhs_col]
    }

    /// Check if current solution is feasible.
    pub fn is_feasible(&self) -> bool {
        let total_vars = self.num_vars + self.num_constraints;
        let rhs_col = total_vars;

        for i in 0..self.num_constraints {
            if self.tableau[i][rhs_col] < T::zero() - T::epsilon() {
                return false;
            }
        }

        true
    }
}

/// Solve linear program using simplex algorithm.
pub fn simd_simplex_solve<T>(
    constraints: Vec<Vec<T>>,
    objective: Vec<T>,
    rhs: Vec<T>,
    max_iterations: usize,
) -> Result<SimplexSolution<T>, SimplexError>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Float,
{
    let mut tableau = SimplexTableau::new(constraints, objective, rhs);

    if !tableau.is_feasible() {
        return Err(SimplexError::Infeasible);
    }

    let mut iterations = 0;

    loop {
        if iterations >= max_iterations {
            return Err(SimplexError::MaxIterationsReached);
        }

        let optimal = tableau.pivot()?;

        if optimal {
            break;
        }

        iterations += 1;
    }

    Ok(SimplexSolution {
        solution: tableau.get_solution(),
        objective_value: tableau.get_objective_value(),
        iterations,
    })
}

/// Result of simplex optimization.
#[derive(Debug, Clone)]
pub struct SimplexSolution<T> {
    /// Optimal solution
    pub solution: Vec<T>,
    /// Optimal objective value
    pub objective_value: T,
    /// Number of iterations
    pub iterations: usize,
}

/// Errors in simplex algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimplexError {
    /// Problem is infeasible
    Infeasible,
    /// Problem is unbounded
    Unbounded,
    /// Maximum iterations reached
    MaxIterationsReached,
}

impl std::fmt::Display for SimplexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infeasible => write!(f, "linear program is infeasible"),
            Self::Unbounded => write!(f, "linear program is unbounded"),
            Self::MaxIterationsReached => write!(f, "maximum iterations reached"),
        }
    }
}

impl std::error::Error for SimplexError {}

/// Dual simplex algorithm for handling infeasibility.
pub fn simd_dual_simplex<T>(
    constraints: Vec<Vec<T>>,
    objective: Vec<T>,
    rhs: Vec<T>,
    max_iterations: usize,
) -> Result<SimplexSolution<T>, SimplexError>
where
    T: Clone + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Float,
{
    let mut tableau = SimplexTableau::new(constraints, objective, rhs);

    let mut iterations = 0;

    loop {
        if iterations >= max_iterations {
            return Err(SimplexError::MaxIterationsReached);
        }

        // Check if optimal
        if tableau.is_feasible() && is_dual_feasible(&tableau) {
            break;
        }

        // Find leaving variable (most negative RHS)
        let leaving_row = find_dual_leaving_variable(&tableau)?;

        if leaving_row.is_none() {
            return Err(SimplexError::Infeasible);
        }

        let leaving = leaving_row.unwrap();

        // Find entering variable
        let entering_col = find_dual_entering_variable(&tableau, leaving)?;

        if entering_col.is_none() {
            return Err(SimplexError::Infeasible);
        }

        let entering = entering_col.unwrap();

        // Perform pivot
        tableau.perform_pivot(entering, leaving);
        tableau.basis[leaving] = entering;

        iterations += 1;
    }

    Ok(SimplexSolution {
        solution: tableau.get_solution(),
        objective_value: tableau.get_objective_value(),
        iterations,
    })
}

fn is_dual_feasible<T>(tableau: &SimplexTableau<T>) -> bool
where
    T: Clone + Float,
{
    let total_vars = tableau.num_vars + tableau.num_constraints;
    let obj_row = &tableau.tableau[tableau.num_constraints];

    for j in 0..total_vars {
        if obj_row[j] < T::zero() - T::epsilon() {
            return false;
        }
    }

    true
}

fn find_dual_leaving_variable<T>(tableau: &SimplexTableau<T>) -> Result<Option<usize>, SimplexError>
where
    T: Clone + Float,
{
    let total_vars = tableau.num_vars + tableau.num_constraints;
    let rhs_col = total_vars;

    let mut min_val = T::zero();
    let mut min_idx = None;

    for i in 0..tableau.num_constraints {
        let rhs = tableau.tableau[i][rhs_col];
        if rhs < min_val {
            min_val = rhs;
            min_idx = Some(i);
        }
    }

    Ok(min_idx)
}

fn find_dual_entering_variable<T>(
    tableau: &SimplexTableau<T>,
    leaving: usize,
) -> Result<Option<usize>, SimplexError>
where
    T: Clone + Float + Div<Output = T>,
{
    let total_vars = tableau.num_vars + tableau.num_constraints;
    let obj_row = &tableau.tableau[tableau.num_constraints];
    let leaving_row = &tableau.tableau[leaving];

    let mut min_ratio = T::infinity();
    let mut min_idx = None;

    for j in 0..total_vars {
        let coeff = leaving_row[j];

        if coeff < T::zero() - T::epsilon() {
            let obj_coeff = obj_row[j];
            let ratio = obj_coeff / (T::zero() - coeff);

            if ratio < min_ratio {
                min_ratio = ratio;
                min_idx = Some(j);
            }
        }
    }

    Ok(min_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_basic() {
        // Maximize: 3x + 2y
        // Subject to: x + y <= 4, 2x + y <= 5, x, y >= 0
        let constraints = vec![vec![1.0, 1.0], vec![2.0, 1.0]];
        let objective = vec![3.0, 2.0];
        let rhs = vec![4.0, 5.0];

        let result = simd_simplex_solve(constraints, objective, rhs, 100).expect("simplex failed");

        assert!(result.iterations > 0);
        assert!(result.objective_value > 0.0);
    }

    #[test]
    fn test_simplex_unbounded() {
        // Maximize: x + y
        // Subject to: -x + y <= 1
        let constraints = vec![vec![-1.0, 1.0]];
        let objective = vec![1.0, 1.0];
        let rhs = vec![1.0];

        let result = simd_simplex_solve(constraints, objective, rhs, 100);

        assert!(result.is_err());
    }

    #[test]
    fn test_tableau_creation() {
        let constraints = vec![vec![1.0, 1.0]];
        let objective = vec![1.0, 1.0];
        let rhs = vec![5.0];

        let tableau = SimplexTableau::new(constraints, objective, rhs);

        assert_eq!(tableau.num_vars, 2);
        assert_eq!(tableau.num_constraints, 1);
        assert_eq!(tableau.basis.len(), 1);
    }
}
