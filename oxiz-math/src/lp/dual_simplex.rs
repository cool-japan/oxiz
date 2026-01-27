//! Dual Simplex Algorithm for Linear Programming.
//!
//! Implements the dual simplex method for solving linear programs,
//! particularly useful for:
//! - Infeasibility analysis
//! - Re-optimization after adding constraints
//! - Branch-and-bound for MIP
//!
//! ## Algorithm
//!
//! Starting from a dual-feasible (but potentially primal-infeasible) basis:
//! 1. Select a primal-infeasible row (basic variable < 0)
//! 2. Perform dual ratio test to select entering variable
//! 3. Pivot to maintain dual feasibility
//! 4. Repeat until primal feasible or proven unbounded
//!
//! ## References
//!
//! - Dantzig: "Linear Programming and Extensions" (1963)
//! - Z3's `math/lp/lp_dual_simplex.cpp`

use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

/// Variable identifier.
pub type VarId = usize;

/// Constraint identifier.
pub type ConstraintId = usize;

/// Dual simplex configuration.
#[derive(Debug, Clone)]
pub struct DualSimplexConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Enable pivot logging.
    pub log_pivots: bool,
    /// Bland's rule to prevent cycling.
    pub use_blands_rule: bool,
}

impl Default for DualSimplexConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100_000,
            log_pivots: false,
            use_blands_rule: true,
        }
    }
}

/// Dual simplex statistics.
#[derive(Debug, Clone, Default)]
pub struct DualSimplexStats {
    /// Number of iterations performed.
    pub iterations: u64,
    /// Number of pivots.
    pub pivots: u64,
    /// Number of dual ratio tests.
    pub ratio_tests: u64,
}

/// Result of dual simplex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DualSimplexResult {
    /// Optimal solution found.
    Optimal,
    /// Problem is unbounded.
    Unbounded,
    /// Problem is infeasible.
    Infeasible,
    /// Iteration limit reached.
    IterationLimit,
}

/// Dual simplex tableau representation.
#[derive(Debug, Clone)]
pub struct DualTableau {
    /// Number of variables.
    num_vars: usize,
    /// Number of constraints.
    num_constraints: usize,
    /// Tableau matrix (constraints × variables).
    /// Row i, column j: coefficient of variable j in constraint i.
    matrix: Vec<Vec<BigRational>>,
    /// Right-hand side values.
    rhs: Vec<BigRational>,
    /// Objective function coefficients.
    obj: Vec<BigRational>,
    /// Basic variables for each row.
    basis: Vec<VarId>,
    /// Non-basic variables.
    non_basis: Vec<VarId>,
    /// Current objective value.
    obj_value: BigRational,
}

impl DualTableau {
    /// Create a new dual tableau.
    ///
    /// # Arguments
    ///
    /// * `num_vars` - Number of decision variables
    /// * `num_constraints` - Number of constraints
    pub fn new(num_vars: usize, num_constraints: usize) -> Self {
        let matrix = vec![vec![BigRational::zero(); num_vars]; num_constraints];
        let rhs = vec![BigRational::zero(); num_constraints];
        let obj = vec![BigRational::zero(); num_vars];
        let basis = (num_vars..num_vars + num_constraints).collect();
        let non_basis = (0..num_vars).collect();

        Self {
            num_vars,
            num_constraints,
            matrix,
            rhs,
            obj,
            basis,
            non_basis,
            obj_value: BigRational::zero(),
        }
    }

    /// Set a coefficient in the tableau.
    pub fn set_coeff(&mut self, row: usize, var: VarId, coeff: BigRational) {
        if row < self.num_constraints && var < self.num_vars {
            self.matrix[row][var] = coeff;
        }
    }

    /// Set the RHS for a constraint.
    pub fn set_rhs(&mut self, row: usize, value: BigRational) {
        if row < self.num_constraints {
            self.rhs[row] = value;
        }
    }

    /// Set objective coefficient for a variable.
    pub fn set_obj_coeff(&mut self, var: VarId, coeff: BigRational) {
        if var < self.num_vars {
            self.obj[var] = coeff;
        }
    }

    /// Check if tableau is primal feasible.
    pub fn is_primal_feasible(&self) -> bool {
        self.rhs.iter().all(|r| !r.is_negative())
    }

    /// Check if tableau is dual feasible.
    pub fn is_dual_feasible(&self) -> bool {
        // Reduced costs must be non-negative for non-basic variables
        self.non_basis
            .iter()
            .all(|&var| !self.obj[var].is_negative())
    }

    /// Find a primal-infeasible row (basic variable < 0).
    fn find_leaving_row(&self) -> Option<usize> {
        for (i, value) in self.rhs.iter().enumerate() {
            if value.is_negative() {
                return Some(i);
            }
        }
        None
    }

    /// Perform dual ratio test to find entering variable.
    ///
    /// For leaving row r, select entering column j that minimizes:
    /// obj[j] / -matrix[r][j] (for matrix[r][j] < 0)
    fn find_entering_column(&self, leaving_row: usize) -> Option<usize> {
        let mut best_col = None;
        let mut best_ratio: Option<BigRational> = None;

        for (j, &var) in self.non_basis.iter().enumerate() {
            let coeff = &self.matrix[leaving_row][var];
            if coeff.is_negative() {
                let ratio = &self.obj[var] / (-coeff.clone());

                if let Some(ref current_best) = best_ratio {
                    if ratio < *current_best {
                        best_ratio = Some(ratio);
                        best_col = Some(j);
                    }
                } else {
                    best_ratio = Some(ratio);
                    best_col = Some(j);
                }
            }
        }

        best_col
    }

    /// Perform a pivot operation.
    fn pivot(&mut self, leaving_row: usize, entering_col: usize) {
        let entering_var = self.non_basis[entering_col];
        let pivot_element = self.matrix[leaving_row][entering_var].clone();

        // Scale pivot row
        for val in &mut self.matrix[leaving_row] {
            *val = val.clone() / pivot_element.clone();
        }
        self.rhs[leaving_row] = self.rhs[leaving_row].clone() / pivot_element.clone();

        // Update other rows
        for i in 0..self.num_constraints {
            if i != leaving_row {
                let factor = self.matrix[i][entering_var].clone();
                for j in 0..self.num_vars {
                    let update = self.matrix[leaving_row][j].clone() * factor.clone();
                    self.matrix[i][j] = self.matrix[i][j].clone() - update;
                }
                let rhs_update = self.rhs[leaving_row].clone() * factor.clone();
                self.rhs[i] = self.rhs[i].clone() - rhs_update;
            }
        }

        // Update objective
        let obj_factor = self.obj[entering_var].clone();
        for j in 0..self.num_vars {
            let update = self.matrix[leaving_row][j].clone() * obj_factor.clone();
            self.obj[j] = self.obj[j].clone() - update;
        }
        let obj_update = self.rhs[leaving_row].clone() * obj_factor.clone();
        self.obj_value = self.obj_value.clone() - obj_update;

        // Update basis
        let leaving_var = self.basis[leaving_row];
        self.basis[leaving_row] = entering_var;
        self.non_basis[entering_col] = leaving_var;
    }

    /// Get current solution values for basic variables.
    pub fn get_solution(&self) -> FxHashMap<VarId, BigRational> {
        let mut solution = FxHashMap::default();
        for (i, &var) in self.basis.iter().enumerate() {
            solution.insert(var, self.rhs[i].clone());
        }
        solution
    }

    /// Get objective value.
    pub fn get_objective_value(&self) -> BigRational {
        self.obj_value.clone()
    }
}

/// Dual simplex solver.
#[derive(Debug)]
pub struct DualSimplex {
    /// Configuration.
    config: DualSimplexConfig,
    /// Tableau.
    tableau: DualTableau,
    /// Statistics.
    stats: DualSimplexStats,
}

impl DualSimplex {
    /// Create a new dual simplex solver.
    pub fn new(config: DualSimplexConfig, num_vars: usize, num_constraints: usize) -> Self {
        Self {
            config,
            tableau: DualTableau::new(num_vars, num_constraints),
            stats: DualSimplexStats::default(),
        }
    }

    /// Get mutable reference to tableau for setup.
    pub fn tableau_mut(&mut self) -> &mut DualTableau {
        &mut self.tableau
    }

    /// Get reference to tableau.
    pub fn tableau(&self) -> &DualTableau {
        &self.tableau
    }

    /// Solve the LP using dual simplex.
    pub fn solve(&mut self) -> DualSimplexResult {
        for iteration in 0..self.config.max_iterations {
            self.stats.iterations += 1;

            // Check if primal feasible (optimal)
            if self.tableau.is_primal_feasible() {
                return DualSimplexResult::Optimal;
            }

            // Find leaving row (primal infeasible)
            let leaving_row = match self.tableau.find_leaving_row() {
                Some(row) => row,
                None => return DualSimplexResult::Optimal,
            };

            // Find entering column (dual ratio test)
            let entering_col = match self.tableau.find_entering_column(leaving_row) {
                Some(col) => col,
                None => {
                    // No valid entering variable => dual unbounded => primal infeasible
                    return DualSimplexResult::Infeasible;
                }
            };

            // Perform pivot
            self.stats.pivots += 1;
            self.stats.ratio_tests += 1;
            self.tableau.pivot(leaving_row, entering_col);

            if self.config.log_pivots {
                eprintln!(
                    "Dual simplex iteration {}: pivot({}, {})",
                    iteration, leaving_row, entering_col
                );
            }
        }

        DualSimplexResult::IterationLimit
    }

    /// Get statistics.
    pub fn stats(&self) -> &DualSimplexStats {
        &self.stats
    }

    /// Get solution (if optimal).
    pub fn get_solution(&self) -> Option<FxHashMap<VarId, BigRational>> {
        if self.tableau.is_primal_feasible() {
            Some(self.tableau.get_solution())
        } else {
            None
        }
    }

    /// Get objective value.
    pub fn get_objective_value(&self) -> BigRational {
        self.tableau.get_objective_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;

    fn rat(n: i64) -> BigRational {
        BigRational::from_i64(n).unwrap()
    }

    #[test]
    fn test_dual_tableau_creation() {
        let tableau = DualTableau::new(3, 2);
        assert_eq!(tableau.num_vars, 3);
        assert_eq!(tableau.num_constraints, 2);
        assert_eq!(tableau.matrix.len(), 2);
        assert_eq!(tableau.rhs.len(), 2);
        assert_eq!(tableau.obj.len(), 3);
    }

    #[test]
    fn test_dual_tableau_is_primal_feasible() {
        let mut tableau = DualTableau::new(2, 2);
        tableau.set_rhs(0, rat(5));
        tableau.set_rhs(1, rat(3));
        assert!(tableau.is_primal_feasible());

        tableau.set_rhs(1, rat(-1));
        assert!(!tableau.is_primal_feasible());
    }

    #[test]
    fn test_dual_simplex_simple() {
        let config = DualSimplexConfig::default();
        let mut solver = DualSimplex::new(config, 2, 1);

        // Simple LP: minimize -x - y subject to x + y <= 10
        // In standard form with slack: x + y + s = 10
        solver.tableau_mut().set_obj_coeff(0, rat(-1));
        solver.tableau_mut().set_obj_coeff(1, rat(-1));
        solver.tableau_mut().set_coeff(0, 0, rat(1));
        solver.tableau_mut().set_coeff(0, 1, rat(1));
        solver.tableau_mut().set_rhs(0, rat(10));

        let result = solver.solve();
        assert_eq!(result, DualSimplexResult::Optimal);
    }
}
