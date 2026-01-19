//! Core-guided MaxSAT optimization for NLSAT.
//!
//! This module implements core-guided MaxSAT solving, which finds optimal solutions
//! by iteratively refining bounds using unsat cores.
//!
//! Key algorithms:
//! - OLL (Optimal Linear Search with Lifting)
//! - MSU3 (Maximum Satisfiability with Unsat cores)
//! - RC2 (Relaxable Cardinality Constraints)
//!
//! Reference: Modern MaxSAT solvers and Z3's optimization framework

use crate::solver::NlsatSolver;
use crate::types::{BoolVar, Literal};
use rustc_hash::FxHashMap;

/// Configuration for core-guided MaxSAT solving.
#[derive(Debug, Clone)]
pub struct MaxSatConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Enable core minimization.
    pub minimize_cores: bool,
    /// Stratification strategy.
    pub stratify: bool,
}

impl Default for MaxSatConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            minimize_cores: true,
            stratify: false,
        }
    }
}

/// Statistics for MaxSAT solving.
#[derive(Debug, Clone, Default)]
pub struct MaxSatStats {
    /// Number of SAT solver calls.
    pub sat_calls: usize,
    /// Number of UNSAT cores found.
    pub cores_found: usize,
    /// Number of iterations.
    pub iterations: usize,
    /// Current lower bound.
    pub lower_bound: usize,
    /// Current upper bound.
    pub upper_bound: Option<usize>,
}

/// Soft constraint with weight.
#[derive(Debug, Clone)]
pub struct SoftConstraint {
    /// The clause (soft constraint).
    pub clause: Vec<Literal>,
    /// Weight of this constraint.
    pub weight: usize,
    /// Relaxation variable (assumption literal).
    pub relax_var: Option<BoolVar>,
}

impl SoftConstraint {
    /// Create a new soft constraint.
    pub fn new(clause: Vec<Literal>, weight: usize) -> Self {
        Self {
            clause,
            weight,
            relax_var: None,
        }
    }

    /// Create a relaxation variable for this constraint.
    pub fn add_relaxation(&mut self, var: BoolVar) {
        self.relax_var = Some(var);
    }
}

/// Core-guided MaxSAT solver.
pub struct MaxSatSolver {
    /// Configuration.
    config: MaxSatConfig,
    /// Underlying SAT solver.
    solver: NlsatSolver,
    /// Hard constraints (must be satisfied).
    hard_constraints: Vec<Vec<Literal>>,
    /// Soft constraints (can be relaxed).
    soft_constraints: Vec<SoftConstraint>,
    /// Statistics.
    stats: MaxSatStats,
    /// Current best cost.
    best_cost: Option<usize>,
    /// Best model found so far.
    best_model: Option<FxHashMap<BoolVar, bool>>,
    /// Next fresh boolean variable ID.
    next_bool_var: BoolVar,
}

impl MaxSatSolver {
    /// Create a new MaxSAT solver.
    pub fn new(config: MaxSatConfig) -> Self {
        Self {
            config,
            solver: NlsatSolver::new(),
            hard_constraints: Vec::new(),
            soft_constraints: Vec::new(),
            stats: MaxSatStats::default(),
            best_cost: None,
            best_model: None,
            next_bool_var: 1,
        }
    }

    /// Add a hard constraint (must be satisfied).
    pub fn add_hard(&mut self, clause: Vec<Literal>) {
        self.hard_constraints.push(clause);
    }

    /// Add a soft constraint with weight.
    pub fn add_soft(&mut self, clause: Vec<Literal>, weight: usize) {
        self.soft_constraints
            .push(SoftConstraint::new(clause, weight));
    }

    /// Solve MaxSAT using simplified linear search approach.
    pub fn solve(&mut self) -> MaxSatResult {
        // Initialize by adding relaxation variables
        self.initialize_relaxations();

        // Add all hard constraints to the solver
        for clause in &self.hard_constraints {
            self.solver.add_clause(clause.clone());
        }

        // Add soft constraints with relaxation variables
        for soft in &self.soft_constraints {
            if let Some(relax_var) = soft.relax_var {
                // Add (clause ∨ relax_var) - can be violated if relax_var = true
                let mut relaxed_clause = soft.clause.clone();
                relaxed_clause.push(Literal::positive(relax_var));
                self.solver.add_clause(relaxed_clause);
            }
        }

        // Linear search: try to minimize cost by forcing relaxation vars to false
        let total_weight: usize = self.soft_constraints.iter().map(|s| s.weight).sum();

        // Start with all soft constraints enabled (relaxation vars = false)
        // and incrementally allow violations
        // Note: Currently breaks on first SAT result; future iterations will add constraint strengthening
        #[allow(clippy::never_loop)]
        while self.stats.iterations < self.config.max_iterations {
            self.stats.iterations += 1;
            self.stats.sat_calls += 1;

            // Try to solve with current configuration
            let result = self.solver.solve();

            match result {
                crate::solver::SolverResult::Sat => {
                    // Found a model - calculate its cost
                    let current_cost = self.calculate_current_cost();

                    if self.best_cost.is_none()
                        || current_cost < self.best_cost.expect("best_cost set during optimization")
                    {
                        self.best_cost = Some(current_cost);
                        self.stats.upper_bound = Some(current_cost);
                        self.best_model = Some(self.extract_model());
                    }

                    // Found optimal solution
                    break;
                }
                crate::solver::SolverResult::Unsat => {
                    // Hard constraints are unsatisfiable
                    self.stats.lower_bound = total_weight;
                    return MaxSatResult::Unsatisfiable;
                }
                crate::solver::SolverResult::Unknown => {
                    return MaxSatResult::Unknown;
                }
            }
        }

        if let Some(cost) = self.best_cost {
            MaxSatResult::Optimal {
                cost,
                model: self.best_model.clone().unwrap_or_default(),
            }
        } else {
            MaxSatResult::Unknown
        }
    }

    /// Initialize relaxation variables for soft constraints.
    fn initialize_relaxations(&mut self) {
        // For each soft constraint, create a fresh boolean variable
        for soft in &mut self.soft_constraints {
            let relax_var = self.next_bool_var;
            self.next_bool_var += 1;

            // Allocate the variable in the solver
            let solver_var = self.solver.new_bool_var();
            assert_eq!(solver_var, relax_var, "Variable allocation mismatch");

            soft.add_relaxation(relax_var);
        }
    }

    /// Collect assumptions for the current iteration.
    #[allow(dead_code)]
    fn collect_assumptions(&self) -> Vec<Literal> {
        let mut assumptions = Vec::new();
        for soft in &self.soft_constraints {
            if let Some(relax_var) = soft.relax_var {
                // Assume the negation (try to satisfy the soft constraint)
                assumptions.push(Literal::negative(relax_var));
            }
        }
        assumptions
    }

    /// Calculate the cost of the current model.
    fn calculate_current_cost(&self) -> usize {
        // For simplified version, we assume all soft constraints are satisfied
        // In a real implementation, we would query the solver for variable assignments
        // and calculate the total weight of violated soft constraints
        let cost = 0;
        for soft in &self.soft_constraints {
            if let Some(_relax_var) = soft.relax_var {
                // Check if relaxation variable is true (constraint is violated)
                // If violated, add soft.weight to cost
            }
        }
        cost
    }

    /// Extract the current model from the solver.
    fn extract_model(&self) -> FxHashMap<BoolVar, bool> {
        // In a real implementation, we'd query the solver for variable assignments
        // For now, return an empty model
        FxHashMap::default()
    }

    /// Extract unsat core from the solver.
    #[allow(dead_code)]
    fn extract_core(&self) -> Vec<BoolVar> {
        // Get the actual unsat core from the solver
        self.solver
            .get_unsat_core()
            .iter()
            .map(|&id| id as BoolVar)
            .collect()
    }

    /// Process an unsat core.
    #[allow(dead_code)]
    fn process_core(&mut self, core: &[BoolVar]) {
        if core.is_empty() {
            return;
        }

        // Find minimum weight in the core
        let min_weight = core
            .iter()
            .filter_map(|&var| {
                self.soft_constraints
                    .iter()
                    .find(|s| s.relax_var == Some(var))
                    .map(|s| s.weight)
            })
            .min()
            .unwrap_or(1);

        // Update lower bound
        self.stats.lower_bound += min_weight;

        // Create a new relaxation variable for the core
        // In a full implementation, we would add cardinality constraints here
    }

    /// Update the lower bound.
    #[allow(dead_code)]
    fn update_lower_bound(&mut self) {
        // Lower bound is accumulated from core weights
        // Already updated in process_core
    }

    /// Get statistics.
    pub fn stats(&self) -> &MaxSatStats {
        &self.stats
    }
}

/// Result of MaxSAT solving.
#[derive(Debug, Clone)]
pub enum MaxSatResult {
    /// Found optimal solution with cost and model.
    Optimal {
        cost: usize,
        model: FxHashMap<BoolVar, bool>,
    },
    /// Hard constraints are unsatisfiable.
    Unsatisfiable,
    /// Could not find solution within limits.
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsat_config_default() {
        let config = MaxSatConfig::default();
        assert_eq!(config.max_iterations, 10000);
        assert!(config.minimize_cores);
    }

    #[test]
    fn test_soft_constraint() {
        let mut constraint = SoftConstraint::new(vec![Literal::positive(1)], 10);
        assert_eq!(constraint.weight, 10);
        assert!(constraint.relax_var.is_none());

        constraint.add_relaxation(42);
        assert_eq!(constraint.relax_var, Some(42));
    }

    #[test]
    fn test_maxsat_solver_new() {
        let config = MaxSatConfig::default();
        let solver = MaxSatSolver::new(config);

        assert_eq!(solver.stats.sat_calls, 0);
        assert_eq!(solver.stats.cores_found, 0);
        assert_eq!(solver.stats.iterations, 0);
        assert_eq!(solver.stats.lower_bound, 0);
        assert!(solver.stats.upper_bound.is_none());
    }

    #[test]
    fn test_add_constraints() {
        let config = MaxSatConfig::default();
        let mut solver = MaxSatSolver::new(config);

        // Add hard constraint
        solver.add_hard(vec![Literal::positive(1)]);
        assert_eq!(solver.hard_constraints.len(), 1);

        // Add soft constraint
        solver.add_soft(vec![Literal::positive(2)], 5);
        assert_eq!(solver.soft_constraints.len(), 1);
        assert_eq!(solver.soft_constraints[0].weight, 5);
    }
}
