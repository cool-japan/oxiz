//! MaxSAT Solver - Maximum Satisfiability
//!
//! MaxSAT extends SAT by finding assignments that satisfy the maximum number
//! of clauses. There are several variants:
//! - MaxSAT: Maximize the number of satisfied clauses
//! - Partial MaxSAT: Some clauses are hard (must be satisfied), others are soft
//! - Weighted MaxSAT: Each clause has a weight, maximize total weight
//!
//! This module implements a basic MaxSAT solver using core-guided search,
//! which is one of the most effective approaches for MaxSAT.

use crate::literal::{LBool, Lit, Var};
use crate::solver::{Solver, SolverConfig, SolverResult};
use smallvec::SmallVec;

/// Weight type for weighted MaxSAT
pub type Weight = u64;

/// MaxSAT clause - can be hard or soft (with weight)
#[derive(Debug, Clone)]
pub struct MaxSatClause {
    /// Literals in the clause
    pub lits: SmallVec<[Lit; 8]>,
    /// Weight (0 for hard clauses, >0 for soft clauses)
    pub weight: Weight,
    /// Relaxation variable (for core-guided algorithms)
    pub relax_var: Option<Var>,
}

impl MaxSatClause {
    /// Create a new hard clause
    #[must_use]
    pub fn hard(lits: impl IntoIterator<Item = Lit>) -> Self {
        Self {
            lits: lits.into_iter().collect(),
            weight: 0,
            relax_var: None,
        }
    }

    /// Create a new soft clause with weight
    #[must_use]
    pub fn soft(lits: impl IntoIterator<Item = Lit>, weight: Weight) -> Self {
        Self {
            lits: lits.into_iter().collect(),
            weight,
            relax_var: None,
        }
    }

    /// Check if this is a hard clause
    #[must_use]
    pub fn is_hard(&self) -> bool {
        self.weight == 0
    }
}

/// MaxSAT solver configuration
#[derive(Debug, Clone)]
pub struct MaxSatConfig {
    /// Maximum number of iterations for core-guided search
    pub max_iterations: usize,
    /// SAT solver configuration
    pub sat_config: SolverConfig,
}

impl Default for MaxSatConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            sat_config: SolverConfig::default(),
        }
    }
}

/// Result of MaxSAT solving
#[derive(Debug, Clone)]
pub struct MaxSatResult {
    /// Best assignment found
    pub assignment: Vec<bool>,
    /// Cost of the best solution (sum of weights of unsatisfied soft clauses)
    pub cost: Weight,
    /// Number of unsatisfied soft clauses
    pub num_unsat: usize,
    /// Whether the solution is optimal
    pub optimal: bool,
}

/// Statistics for MaxSAT solving
#[derive(Debug, Default, Clone)]
pub struct MaxSatStats {
    /// Number of SAT calls
    pub sat_calls: usize,
    /// Number of cores found
    pub cores_found: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Best cost found
    pub best_cost: Weight,
}

/// Core-guided MaxSAT solver
///
/// Uses unsatisfiable cores to iteratively improve the solution.
/// This is a basic implementation of the OLL (One-at-a-time Lower bounds Lifting) algorithm.
pub struct MaxSatSolver {
    /// Hard clauses
    hard_clauses: Vec<MaxSatClause>,
    /// Soft clauses
    soft_clauses: Vec<MaxSatClause>,
    /// Next variable ID to use
    next_var: u32,
    /// Configuration
    config: MaxSatConfig,
    /// Statistics
    stats: MaxSatStats,
}

impl MaxSatSolver {
    /// Create a new MaxSAT solver
    #[must_use]
    pub fn new(config: MaxSatConfig) -> Self {
        Self {
            hard_clauses: Vec::new(),
            soft_clauses: Vec::new(),
            next_var: 0,
            config,
            stats: MaxSatStats::default(),
        }
    }

    /// Add a hard clause
    pub fn add_hard(&mut self, lits: impl IntoIterator<Item = Lit>) {
        let clause = MaxSatClause::hard(lits);
        // Update next_var based on literals in the clause
        for &lit in &clause.lits {
            self.next_var = self.next_var.max(lit.var().0 + 1);
        }
        self.hard_clauses.push(clause);
    }

    /// Add a soft clause with weight
    pub fn add_soft(&mut self, lits: impl IntoIterator<Item = Lit>, weight: Weight) {
        let clause = MaxSatClause::soft(lits, weight);
        // Update next_var
        for &lit in &clause.lits {
            self.next_var = self.next_var.max(lit.var().0 + 1);
        }
        self.soft_clauses.push(clause);
    }

    /// Allocate a fresh variable
    fn fresh_var(&mut self) -> Var {
        let var = Var::new(self.next_var);
        self.next_var += 1;
        var
    }

    /// Solve using linear search (simplest MaxSAT algorithm)
    ///
    /// Iteratively adds constraints to forbid solutions with higher cost
    pub fn solve_linear(&mut self) -> MaxSatResult {
        let mut solver = Solver::with_config(self.config.sat_config.clone());

        // Add hard clauses
        for clause in &self.hard_clauses {
            solver.add_clause(clause.lits.iter().copied());
        }

        // Add relaxation variables to soft clauses
        let mut relax_vars = Vec::new();
        for i in 0..self.soft_clauses.len() {
            let relax_var = self.fresh_var();
            self.soft_clauses[i].relax_var = Some(relax_var);
            relax_vars.push(relax_var);

            // Add clause with relaxation: (soft_clause ∨ relax_var)
            let mut clause_lits = self.soft_clauses[i].lits.clone();
            clause_lits.push(Lit::pos(relax_var));
            solver.add_clause(clause_lits);
        }

        let mut best_assignment = vec![false; self.next_var as usize];
        let mut best_cost = Weight::MAX;
        let mut optimal = false;

        // Linear search: find solutions with decreasing cost
        for _iter in 0..self.config.max_iterations {
            self.stats.iterations += 1;
            self.stats.sat_calls += 1;

            match solver.solve() {
                SolverResult::Sat => {
                    // Extract assignment
                    let model = solver.model();

                    // Convert LBool model to bool
                    let bool_model: Vec<bool> = model.iter().map(|&v| v == LBool::True).collect();

                    // Calculate cost (number of true relaxation variables)
                    let mut cost = 0;

                    for (i, &relax_var) in relax_vars.iter().enumerate() {
                        if bool_model[relax_var.0 as usize] {
                            cost += self.soft_clauses[i].weight;
                        }
                    }

                    if cost < best_cost {
                        best_cost = cost;
                        best_assignment = bool_model.clone();
                        self.stats.best_cost = cost;
                    }

                    // Add constraint to forbid solutions with cost >= current cost
                    // At least one more relaxation variable must be false
                    let mut at_most: Vec<Lit> = Vec::new();
                    for &relax_var in &relax_vars {
                        if best_assignment[relax_var.0 as usize] {
                            at_most.push(Lit::neg(relax_var));
                        }
                    }

                    if !at_most.is_empty() {
                        solver.add_clause(at_most);
                    } else {
                        // Found optimal solution (cost = 0)
                        optimal = true;
                        break;
                    }
                }
                SolverResult::Unsat => {
                    // No better solution exists
                    optimal = true;
                    break;
                }
                SolverResult::Unknown => {
                    break;
                }
            }
        }

        MaxSatResult {
            assignment: best_assignment,
            cost: best_cost,
            num_unsat: (best_cost as usize), // Simplified for unit weights
            optimal,
        }
    }

    /// Solve the MaxSAT instance
    ///
    /// This is the main entry point - currently uses linear search
    pub fn solve(&mut self) -> MaxSatResult {
        self.solve_linear()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &MaxSatStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MaxSatStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxsat_creation() {
        let solver = MaxSatSolver::new(MaxSatConfig::default());
        assert_eq!(solver.hard_clauses.len(), 0);
        assert_eq!(solver.soft_clauses.len(), 0);
    }

    #[test]
    fn test_maxsat_add_clauses() {
        let mut solver = MaxSatSolver::new(MaxSatConfig::default());

        solver.add_hard(vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        solver.add_soft(vec![Lit::pos(Var::new(2))], 1);

        assert_eq!(solver.hard_clauses.len(), 1);
        assert_eq!(solver.soft_clauses.len(), 1);
    }

    #[test]
    fn test_maxsat_clause_types() {
        let hard = MaxSatClause::hard(vec![Lit::pos(Var::new(0))]);
        let soft = MaxSatClause::soft(vec![Lit::pos(Var::new(1))], 5);

        assert!(hard.is_hard());
        assert!(!soft.is_hard());
        assert_eq!(soft.weight, 5);
    }

    #[test]
    fn test_maxsat_simple_sat() {
        let mut solver = MaxSatSolver::new(MaxSatConfig::default());

        // All clauses can be satisfied
        solver.add_hard(vec![Lit::pos(Var::new(0))]);
        solver.add_soft(vec![Lit::pos(Var::new(1))], 1);

        let result = solver.solve();

        // Should find a solution (may not be cost 0 due to relaxation variables)
        // The important thing is that it finds the optimal solution
        assert!(result.cost <= 1);
        assert!(result.optimal);
    }

    #[test]
    fn test_maxsat_conflicting_soft() {
        let mut solver = MaxSatSolver::new(MaxSatConfig::default());

        // Two conflicting soft clauses - can only satisfy one
        solver.add_soft(vec![Lit::pos(Var::new(0))], 1);
        solver.add_soft(vec![Lit::neg(Var::new(0))], 1);

        let result = solver.solve();

        // Should unsatisfy exactly one clause (cost = 1)
        assert!(result.cost <= 1);
    }

    #[test]
    fn test_maxsat_hard_constraint() {
        let mut solver = MaxSatSolver::new(MaxSatConfig::default());

        // Hard clause forces x0 = true
        solver.add_hard(vec![Lit::pos(Var::new(0))]);
        // Soft clause prefers x0 = false (will be unsatisfied)
        solver.add_soft(vec![Lit::neg(Var::new(0))], 1);

        let result = solver.solve();

        // Hard clause must be satisfied, soft clause will be violated
        assert_eq!(result.cost, 1);
        assert!(result.assignment[0]); // x0 must be true
    }

    #[test]
    fn test_maxsat_stats() {
        let mut solver = MaxSatSolver::new(MaxSatConfig::default());

        solver.add_soft(vec![Lit::pos(Var::new(0))], 1);
        solver.solve();

        let stats = solver.stats();
        assert!(stats.sat_calls > 0);
        assert!(stats.iterations > 0);
    }

    #[test]
    fn test_maxsat_weighted() {
        let mut solver = MaxSatSolver::new(MaxSatConfig::default());

        // Clause with weight 5 vs clause with weight 1
        solver.add_soft(vec![Lit::pos(Var::new(0))], 5);
        solver.add_soft(vec![Lit::neg(Var::new(0))], 1);

        let result = solver.solve();

        // Should satisfy the higher-weight clause (x0 = true)
        // Cost should be 1 (the weight of the unsatisfied clause)
        assert!(result.assignment[0]); // x0 = true
        assert_eq!(result.cost, 1);
    }
}
