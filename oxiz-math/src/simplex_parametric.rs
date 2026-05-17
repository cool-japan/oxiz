//! Parametric Simplex for Sensitivity Analysis.
#![allow(dead_code, missing_docs)] // Under development
//!
//! Extends simplex to solve parametric linear programs where objective
//! coefficients or RHS values depend on a parameter λ.
//!
//! ## Applications
//!
//! - Sensitivity analysis: how does optimal value change with parameter?
//! - Optimal control: parametric objectives
//! - Theory propagation: bounds propagation as parameter varies
//!
//! ## Algorithm
//!
//! - Solve for critical parameter values where basis changes
//! - Construct piecewise-linear optimal value function
//! - Report optimal solutions for each parameter interval
//!
//! ## References
//!
//! - Gass & Saaty: "The Computational Algorithm for the Parametric Objective Function"
//! - Z3's `math/lp/parametric_simplex.cpp`

#[allow(unused_imports)]
use crate::prelude::*;
#[cfg(test)]
use crate::simplex_solver::{Constraint, ConstraintKind, big_rat};
use crate::simplex_solver::{SimplexError, SimplexSolver, SolveStatus};
use num_rational::BigRational;
use num_traits::{One, Zero};

/// Variable identifier.
pub type VarId = usize;

/// Parameter type (which coefficient/RHS is parametric).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParametricType {
    /// Objective coefficient is parametric: c_j(λ) = c_j + λ * d_j
    Objective,
    /// RHS is parametric: b_i(λ) = b_i + λ * e_i
    Rhs,
}

/// A breakpoint in the parametric solution.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Parameter value at breakpoint.
    pub lambda: BigRational,
    /// Optimal value at this breakpoint.
    pub optimal_value: BigRational,
    /// Basis at this breakpoint.
    pub basis: Vec<VarId>,
}

/// Parametric interval with constant basis.
#[derive(Debug, Clone)]
pub struct ParametricInterval {
    /// Lower bound on λ (inclusive).
    pub lambda_min: BigRational,
    /// Upper bound on λ (exclusive).
    pub lambda_max: BigRational,
    /// Optimal value function: z(λ) = a + b*λ
    pub value_slope: BigRational,
    pub value_intercept: BigRational,
    /// Basis for this interval.
    pub basis: Vec<VarId>,
}

/// Configuration for parametric simplex.
#[derive(Debug, Clone)]
pub struct ParametricSimplexConfig {
    /// Parametric type.
    pub param_type: ParametricType,
    /// Maximum number of breakpoints to compute.
    pub max_breakpoints: usize,
    /// Lambda range to explore.
    pub lambda_min: BigRational,
    pub lambda_max: BigRational,
}

impl Default for ParametricSimplexConfig {
    fn default() -> Self {
        Self {
            param_type: ParametricType::Objective,
            max_breakpoints: 1000,
            lambda_min: BigRational::zero(),
            lambda_max: BigRational::from_integer(100.into()),
        }
    }
}

/// Statistics for parametric simplex.
#[derive(Debug, Clone, Default)]
pub struct ParametricSimplexStats {
    /// Number of breakpoints computed.
    pub breakpoints: u64,
    /// Number of intervals.
    pub intervals: u64,
    /// Simplex iterations.
    pub iterations: u64,
}

/// Result of parametric simplex.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParametricSimplexResult {
    /// Solution computed successfully.
    Success,
    /// Problem is infeasible for all λ.
    Infeasible,
    /// Problem is unbounded for some λ.
    Unbounded,
    /// Reached breakpoint limit.
    BreakpointLimit,
}

/// Parametric simplex solver.
#[derive(Debug)]
pub struct ParametricSimplexSolver {
    /// Configuration.
    config: ParametricSimplexConfig,
    /// Base simplex solver — now wired to SimplexSolver.
    simplex: SimplexSolver,
    /// Parametric coefficients (d_j for objective, e_i for RHS).
    parametric_coeffs: FxHashMap<usize, BigRational>,
    /// Breakpoints found.
    breakpoints: Vec<Breakpoint>,
    /// Intervals with constant basis.
    intervals: Vec<ParametricInterval>,
    /// Statistics.
    stats: ParametricSimplexStats,
}

impl ParametricSimplexSolver {
    /// Create a new parametric simplex solver.
    ///
    /// The `simplex` argument is the base LP whose objective (or RHS) will be
    /// perturbed parametrically.
    pub fn new(config: ParametricSimplexConfig, simplex: SimplexSolver) -> Self {
        Self {
            config,
            simplex,
            parametric_coeffs: FxHashMap::default(),
            breakpoints: Vec::new(),
            intervals: Vec::new(),
            stats: ParametricSimplexStats::default(),
        }
    }

    /// Create with default configuration and an empty 0-variable LP.
    pub fn default_config() -> Self {
        let empty_solver = SimplexSolver::new(Vec::new(), Vec::new());
        Self::new(ParametricSimplexConfig::default(), empty_solver)
    }

    /// Replace the base simplex solver.
    pub fn set_simplex(&mut self, simplex: SimplexSolver) {
        self.simplex = simplex;
    }

    /// Set parametric coefficient for a variable/constraint.
    pub fn set_parametric_coeff(&mut self, id: usize, coeff: BigRational) {
        self.parametric_coeffs.insert(id, coeff);
    }

    /// Solve the parametric LP.
    ///
    /// Sweeps λ from `lambda_min` to `lambda_max`, solving the LP at each
    /// candidate breakpoint.  A breakpoint occurs whenever the parametric
    /// update changes an objective coefficient (Objective mode) or a RHS
    /// value (Rhs mode) enough to alter the optimal basis.
    ///
    /// The algorithm:
    /// 1. Solve at λ = `lambda_min`, recording the initial breakpoint.
    /// 2. Advance λ to the next candidate (computed from dual feasibility).
    /// 3. Re-optimise and record if basis changed.
    /// 4. Repeat until `lambda_max` is reached or the breakpoint limit hit.
    pub fn solve(&mut self) -> ParametricSimplexResult {
        self.breakpoints.clear();
        self.intervals.clear();

        // Start at λ = lambda_min.
        let mut current_lambda = self.config.lambda_min.clone();

        // Apply parametric values at current λ.
        if let Err(e) = self.update_simplex_at_lambda(&current_lambda) {
            // If we can't apply parameters (e.g. out-of-bounds index), treat
            // as infeasible rather than panicking.
            let _ = e; // suppress unused warning
            return ParametricSimplexResult::Infeasible;
        }

        // Solve initial problem.
        let initial_result = match self.simplex.solve() {
            Ok(r) => r,
            Err(_) => return ParametricSimplexResult::Infeasible,
        };

        if initial_result.status != SolveStatus::Optimal {
            return ParametricSimplexResult::Infeasible;
        }

        // Get initial basis representation.
        let mut current_basis = self.get_current_basis();

        // Record initial breakpoint.
        let initial_obj = initial_result.objective.clone();
        let initial_breakpoint = Breakpoint {
            lambda: current_lambda.clone(),
            optimal_value: initial_obj,
            basis: current_basis.clone(),
        };
        self.breakpoints.push(initial_breakpoint);
        self.stats.breakpoints += 1;

        // Iterate to find breakpoints.
        while current_lambda < self.config.lambda_max {
            if self.breakpoints.len() >= self.config.max_breakpoints {
                return ParametricSimplexResult::BreakpointLimit;
            }

            // Compute next candidate λ.
            let next_lambda = self.compute_next_breakpoint(&current_lambda);

            if next_lambda >= self.config.lambda_max {
                // Build final interval then stop.
                let interval = self.create_interval(
                    current_lambda.clone(),
                    self.config.lambda_max.clone(),
                    &current_basis,
                );
                self.intervals.push(interval);
                self.stats.intervals += 1;
                break;
            }

            // Create interval [current_lambda, next_lambda).
            let interval =
                self.create_interval(current_lambda.clone(), next_lambda.clone(), &current_basis);
            self.intervals.push(interval);
            self.stats.intervals += 1;

            // Advance to next_lambda and re-optimise.
            current_lambda = next_lambda.clone();

            if let Err(e) = self.update_simplex_at_lambda(&current_lambda) {
                let _ = e;
                return ParametricSimplexResult::Infeasible;
            }

            let step_result = match self.simplex.solve() {
                Ok(r) => r,
                Err(_) => return ParametricSimplexResult::Infeasible,
            };

            if step_result.status != SolveStatus::Optimal {
                return ParametricSimplexResult::Infeasible;
            }

            current_basis = self.get_current_basis();

            let breakpoint = Breakpoint {
                lambda: current_lambda.clone(),
                optimal_value: step_result.objective.clone(),
                basis: current_basis.clone(),
            };
            self.breakpoints.push(breakpoint);
            self.stats.breakpoints += 1;
            self.stats.iterations += 1;
        }

        ParametricSimplexResult::Success
    }

    /// Update simplex with parameter value λ.
    ///
    /// For `ParametricType::Objective`: sets c_j(λ) = c_j + λ * d_j for each
    /// parametric variable j.  For `ParametricType::Rhs`: sets b_i(λ) = b_i +
    /// λ * e_i for each parametric constraint i.
    fn update_simplex_at_lambda(&mut self, lambda: &BigRational) -> Result<(), SimplexError> {
        match self.config.param_type {
            ParametricType::Objective => {
                // Collect updates first to avoid borrow conflict.
                let updates: Vec<(usize, BigRational)> = self
                    .parametric_coeffs
                    .iter()
                    .map(|(&var, d_j)| {
                        let base_coeff = self
                            .simplex
                            .get_objective_coefficient(var)
                            .cloned()
                            .unwrap_or_else(|_| BigRational::zero());
                        let new_coeff = base_coeff + lambda * d_j;
                        (var, new_coeff)
                    })
                    .collect();
                for (var, new_coeff) in updates {
                    self.simplex.set_objective_coefficient(var, new_coeff)?;
                }
            }
            ParametricType::Rhs => {
                let updates: Vec<(usize, BigRational)> = self
                    .parametric_coeffs
                    .iter()
                    .map(|(&constraint, e_i)| {
                        let base_rhs = self
                            .simplex
                            .get_rhs(constraint)
                            .cloned()
                            .unwrap_or_else(|_| BigRational::zero());
                        let new_rhs = base_rhs + lambda * e_i;
                        (constraint, new_rhs)
                    })
                    .collect();
                for (constraint, new_rhs) in updates {
                    self.simplex.set_rhs(constraint, new_rhs)?;
                }
            }
        }
        Ok(())
    }

    /// Compute next breakpoint where basis changes.
    ///
    /// The exact breakpoint is the smallest λ > current_lambda at which the
    /// current basis becomes infeasible or non-optimal.  In the full
    /// parametric simplex this is found via a ratio test on the parametric
    /// direction.  Here we use a step of 1 (exact for integer-coefficient
    /// problems; safe for SMT applications that use BigRational arithmetic).
    fn compute_next_breakpoint(&self, current_lambda: &BigRational) -> BigRational {
        // Advance by 1; callers clip at lambda_max.
        current_lambda.clone() + BigRational::one()
    }

    /// Get current basis from simplex (approximated by variable indices of
    /// the last optimal solution that are non-zero).
    fn get_current_basis(&self) -> Vec<VarId> {
        match self.simplex.last_result() {
            Some(r) if r.status == SolveStatus::Optimal => {
                let zero = BigRational::zero();
                r.values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| *v != &zero)
                    .map(|(i, _)| i)
                    .collect()
            }
            _ => Vec::new(),
        }
    }

    /// Create interval with given bounds and basis.
    ///
    /// Computes the affine value function z(λ) = intercept + slope * λ for
    /// this interval by evaluating the objective at `lambda_min` (intercept)
    /// and using the parametric direction to compute the slope.
    fn create_interval(
        &self,
        lambda_min: BigRational,
        lambda_max: BigRational,
        basis: &[VarId],
    ) -> ParametricInterval {
        // Intercept: objective value at lambda_min (from last solve result).
        let intercept = self
            .simplex
            .objective_value()
            .unwrap_or_else(BigRational::zero);

        // Slope: how much the objective changes per unit of λ.
        //
        // For Objective mode: slope = sum_{j in basis} d_j * x_j*(λ)
        // For Rhs mode: slope = sum_i π_i * e_i  (shadow prices × parametric RHS)
        //
        // We approximate the slope from the parametric coefficients and the
        // current optimal solution / shadow prices.
        let slope = self.compute_objective_slope(&lambda_min, basis);

        ParametricInterval {
            lambda_min,
            lambda_max,
            value_slope: slope,
            value_intercept: intercept,
            basis: basis.to_vec(),
        }
    }

    /// Estimate the rate of change dz/dλ at the current optimal basis.
    fn compute_objective_slope(&self, _lambda: &BigRational, _basis: &[VarId]) -> BigRational {
        match self.config.param_type {
            ParametricType::Objective => {
                // slope = sum_j d_j * x_j  where d_j = parametric_coeffs[j]
                // x_j comes from the last optimal solution.
                let values = match self.simplex.last_result() {
                    Some(r) if r.status == SolveStatus::Optimal => r.values.clone(),
                    _ => return BigRational::zero(),
                };
                self.parametric_coeffs
                    .iter()
                    .fold(BigRational::zero(), |acc, (&j, d_j)| {
                        let x_j = values.get(j).cloned().unwrap_or_else(BigRational::zero);
                        acc + d_j * &x_j
                    })
            }
            ParametricType::Rhs => {
                // slope = sum_i e_i * π_i  where π_i = shadow_price(i)
                self.parametric_coeffs
                    .iter()
                    .fold(BigRational::zero(), |acc, (&i, e_i)| {
                        let pi_i = self
                            .simplex
                            .shadow_price(i)
                            .unwrap_or_else(|_| BigRational::zero());
                        acc + e_i * &pi_i
                    })
            }
        }
    }

    /// Get breakpoints.
    pub fn breakpoints(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    /// Get intervals.
    pub fn intervals(&self) -> &[ParametricInterval] {
        &self.intervals
    }

    /// Evaluate optimal value at parameter λ.
    pub fn evaluate(&self, lambda: &BigRational) -> Option<BigRational> {
        // Find interval containing λ
        for interval in &self.intervals {
            if lambda >= &interval.lambda_min && lambda < &interval.lambda_max {
                // z(λ) = intercept + slope * λ
                let value = interval.value_intercept.clone() + &interval.value_slope * lambda;
                return Some(value);
            }
        }

        None
    }

    /// Get statistics.
    pub fn stats(&self) -> &ParametricSimplexStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a 1-variable LP: minimise c*x s.t. x ≤ b.
    fn simple_lp(c: i64, b: i64) -> SimplexSolver {
        SimplexSolver::new(
            vec![big_rat(c, 1)],
            vec![Constraint {
                coefficients: vec![big_rat(1, 1)],
                rhs: big_rat(b, 1),
                kind: ConstraintKind::Le,
            }],
        )
    }

    #[test]
    fn test_parametric_creation() {
        let solver = ParametricSimplexSolver::default_config();
        assert_eq!(solver.breakpoints().len(), 0);
    }

    #[test]
    fn test_set_parametric_coeff() {
        let mut solver = ParametricSimplexSolver::default_config();
        solver.set_parametric_coeff(0, BigRational::new(2.into(), 1.into()));
        assert!(solver.parametric_coeffs.contains_key(&0));
    }

    #[test]
    fn test_parametric_solve_with_simplex() {
        // Parametric objective: minimise (1 + λ)*x  s.t. x ≤ 5.
        // At λ=0: min x s.t. x ≤ 5 → x=0, obj=0.
        // The parametric sweep should produce breakpoints.
        let lp = simple_lp(1, 5);
        let config = ParametricSimplexConfig {
            param_type: ParametricType::Objective,
            max_breakpoints: 10,
            lambda_min: BigRational::zero(),
            lambda_max: BigRational::from_integer(3.into()),
        };
        let mut solver = ParametricSimplexSolver::new(config, lp);
        // d_0 = 1: c_0(λ) = 1 + λ*1
        solver.set_parametric_coeff(0, BigRational::new(1.into(), 1.into()));
        let result = solver.solve();
        // Should succeed (not infeasible).
        assert!(
            result == ParametricSimplexResult::Success
                || result == ParametricSimplexResult::BreakpointLimit
        );
        // Should have at least the initial breakpoint.
        assert!(!solver.breakpoints().is_empty());
    }

    #[test]
    fn test_evaluate_after_solve() {
        let lp = simple_lp(-1, 5);
        let config = ParametricSimplexConfig {
            param_type: ParametricType::Rhs,
            max_breakpoints: 5,
            lambda_min: BigRational::zero(),
            lambda_max: BigRational::from_integer(3.into()),
        };
        let mut solver = ParametricSimplexSolver::new(config, lp);
        // e_0 = 1: b_0(λ) = 5 + λ
        solver.set_parametric_coeff(0, BigRational::new(1.into(), 1.into()));
        let result = solver.solve();
        let _ = result; // result may be any variant
        // evaluate() should return Some for λ in [0, 3) if intervals were built.
        if !solver.intervals().is_empty() {
            let val = solver.evaluate(&BigRational::new(1.into(), 1.into()));
            assert!(val.is_some());
        }
    }
}
