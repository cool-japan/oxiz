//! UNSAT core extraction and minimization
//!
//! When a SAT solver determines that a formula is unsatisfiable, it's often
//! useful to extract a minimal subset of clauses that are still unsatisfiable.
//! This is called an UNSAT core.
//!
//! Applications:
//! - Debugging: Understanding why a formula is UNSAT
//! - Optimization: Identifying minimal conflicts
//! - Proof generation: Smaller UNSAT cores lead to smaller proofs

use crate::literal::Lit;
use crate::solver::{Solver, SolverResult};
use std::collections::HashSet;

/// UNSAT core extractor
pub struct UnsatCore {
    /// Core clause indices (indices into the original clause set)
    core_clauses: Vec<usize>,
}

impl UnsatCore {
    /// Create a new UNSAT core
    #[must_use]
    pub fn new() -> Self {
        Self {
            core_clauses: Vec::new(),
        }
    }

    /// Extract UNSAT core using assumption-based method
    ///
    /// This method:
    /// 1. Associates each original clause with a unique assumption literal
    /// 2. Solves with all assumptions
    /// 3. If UNSAT, extracts the conflicting assumptions
    /// 4. Maps assumptions back to original clauses
    ///
    /// # Arguments
    ///
    /// * `clauses` - The original clauses
    ///
    /// # Returns
    ///
    /// Vector of clause indices that form an UNSAT core, or None if SAT
    pub fn extract(clauses: &[Vec<Lit>]) -> Option<Vec<usize>> {
        if clauses.is_empty() {
            return None;
        }

        let mut solver = Solver::new();

        // Create assumption literals (one per clause)
        let mut assumptions = Vec::with_capacity(clauses.len());
        for _ in 0..clauses.len() {
            let var = solver.new_var();
            assumptions.push(Lit::pos(var));
        }

        // Add clauses with assumption guards
        // For each clause C, add: assumption => C
        // Which is: ~assumption \/ C
        for (i, clause) in clauses.iter().enumerate() {
            let mut guarded_clause = vec![assumptions[i].negate()];
            guarded_clause.extend_from_slice(clause);
            solver.add_clause(guarded_clause.iter().copied());
        }

        // Solve with all assumptions
        let (result, conflict) = solver.solve_with_assumptions(&assumptions);

        match result {
            SolverResult::Unsat => {
                // Extract conflict core
                if let Some(core) = conflict {
                    let mut core_indices = Vec::new();

                    for lit in core {
                        // Find which assumption this corresponds to
                        if let Some(idx) = assumptions.iter().position(|&a| a == lit) {
                            core_indices.push(idx);
                        }
                    }

                    Some(core_indices)
                } else {
                    // No specific conflict, all clauses are in the core
                    Some((0..clauses.len()).collect())
                }
            }
            _ => None,
        }
    }

    /// Minimize an UNSAT core using deletion-based algorithm
    ///
    /// Tries to remove each clause from the core and checks if it's still UNSAT.
    /// If yes, the clause is not needed; if no, keep it.
    ///
    /// # Arguments
    ///
    /// * `clauses` - The original clauses
    /// * `core` - Initial UNSAT core (indices into clauses)
    ///
    /// # Returns
    ///
    /// Minimized UNSAT core
    pub fn minimize(clauses: &[Vec<Lit>], core: &[usize]) -> Vec<usize> {
        if core.is_empty() {
            return Vec::new();
        }

        let mut current_core: HashSet<usize> = core.iter().copied().collect();

        // Try removing each clause
        for &idx in core {
            // Temporarily remove this clause
            current_core.remove(&idx);

            // Check if still UNSAT
            let mut solver = Solver::new();
            for &i in &current_core {
                solver.add_clause(clauses[i].iter().copied());
            }

            match solver.solve() {
                SolverResult::Unsat => {
                    // Still UNSAT without this clause, keep it removed
                }
                _ => {
                    // Need this clause, add it back
                    current_core.insert(idx);
                }
            }
        }

        let mut result: Vec<usize> = current_core.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Minimize using binary search (faster for large cores)
    ///
    /// Divides the core into two halves and recursively minimizes.
    ///
    /// # Arguments
    ///
    /// * `clauses` - The original clauses
    /// * `core` - Initial UNSAT core (indices into clauses)
    ///
    /// # Returns
    ///
    /// Minimized UNSAT core
    pub fn minimize_binary(clauses: &[Vec<Lit>], core: &[usize]) -> Vec<usize> {
        if core.len() <= 1 {
            return core.to_vec();
        }

        let mid = core.len() / 2;
        let left = &core[..mid];
        let right = &core[mid..];

        // Try left half alone
        if Self::is_unsat(clauses, left) {
            return Self::minimize_binary(clauses, left);
        }

        // Try right half alone
        if Self::is_unsat(clauses, right) {
            return Self::minimize_binary(clauses, right);
        }

        // Need both halves, minimize each recursively
        let mut min_left = Self::minimize_binary(clauses, left);
        let mut min_right = Self::minimize_binary(clauses, right);
        min_left.append(&mut min_right);
        min_left.sort_unstable();
        min_left
    }

    /// Check if a subset of clauses is UNSAT
    fn is_unsat(clauses: &[Vec<Lit>], indices: &[usize]) -> bool {
        let mut solver = Solver::new();
        for &idx in indices {
            solver.add_clause(clauses[idx].iter().copied());
        }
        matches!(solver.solve(), SolverResult::Unsat)
    }

    /// Get the core clause indices
    #[must_use]
    pub fn core(&self) -> &[usize] {
        &self.core_clauses
    }

    /// Compute core size reduction ratio
    #[must_use]
    pub fn reduction_ratio(original_size: usize, core_size: usize) -> f64 {
        if original_size == 0 {
            0.0
        } else {
            1.0 - (core_size as f64 / original_size as f64)
        }
    }
}

impl Default for UnsatCore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_unsat_core_extraction() {
        // Create a simple UNSAT formula
        // (x1) /\ (~x1)
        let clauses = vec![vec![Lit::pos(Var::new(0))], vec![Lit::neg(Var::new(0))]];

        let core = UnsatCore::extract(&clauses);
        assert!(core.is_some());
        let core = core.unwrap();
        // Core should be non-empty for UNSAT formula
        assert!(!core.is_empty());
        // Core should not exceed original clause count
        assert!(core.len() <= clauses.len());
    }

    #[test]
    fn test_unsat_core_with_redundant_clauses() {
        // (x1) /\ (~x1) /\ (x2)
        // The third clause is redundant for UNSAT
        let clauses = vec![
            vec![Lit::pos(Var::new(0))],
            vec![Lit::neg(Var::new(0))],
            vec![Lit::pos(Var::new(1))],
        ];

        let core = UnsatCore::extract(&clauses);
        assert!(core.is_some());
        let core = core.unwrap();

        // Core should not include the redundant clause
        assert!(core.len() <= 2);
    }

    #[test]
    fn test_minimize_deletion() {
        // Create formula: (x1) /\ (~x1) /\ (x2 v x3) /\ (~x2 v ~x3)
        let clauses = vec![
            vec![Lit::pos(Var::new(0))], // Essential
            vec![Lit::neg(Var::new(0))], // Essential
            vec![Lit::pos(Var::new(1)), Lit::pos(Var::new(2))],
            vec![Lit::neg(Var::new(1)), Lit::neg(Var::new(2))],
        ];

        // Start with all clauses as initial core
        let initial_core = vec![0, 1, 2, 3];

        let minimized = UnsatCore::minimize(&clauses, &initial_core);

        // Should contain at least the essential conflict
        assert!(minimized.contains(&0));
        assert!(minimized.contains(&1));
    }

    #[test]
    fn test_is_unsat() {
        let clauses = vec![vec![Lit::pos(Var::new(0))], vec![Lit::neg(Var::new(0))]];

        assert!(UnsatCore::is_unsat(&clauses, &[0, 1]));
        assert!(!UnsatCore::is_unsat(&clauses, &[0]));
        assert!(!UnsatCore::is_unsat(&clauses, &[1]));
    }

    #[test]
    fn test_reduction_ratio() {
        assert!((UnsatCore::reduction_ratio(100, 50) - 0.5).abs() < 1e-6);
        assert!((UnsatCore::reduction_ratio(100, 10) - 0.9).abs() < 1e-6);
        assert!((UnsatCore::reduction_ratio(100, 100) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_sat_formula_no_core() {
        // SAT formula: (x1 v x2)
        let clauses = vec![vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]];

        let core = UnsatCore::extract(&clauses);
        assert!(core.is_none()); // SAT formula has no UNSAT core
    }
}
