//! RC2 (Relaxable Cardinality Constraints) MaxSAT solver.
//!
//! RC2 is a modern core-guided MaxSAT algorithm that uses AtMost-1 cardinality
//! constraints with relaxation variables. It's generally more efficient than
//! traditional algorithms like Fu-Malik or OLL.
//!
//! Reference: Z3's RC2 implementation and the original RC2 paper

use crate::maxsat::{MaxSatResult, SoftClause, SoftId, Weight};
use crate::totalizer::{CardinalityEncoding, encode_at_most_k};
use oxiz_sat::{Lit, Solver as SatSolver, SolverResult, Var};
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Errors specific to RC2
#[derive(Error, Debug)]
pub enum Rc2Error {
    /// SAT solver error
    #[error("SAT solver error: {0}")]
    SolverError(String),
    /// Hard constraints are unsatisfiable
    #[error("hard constraints unsatisfiable")]
    Unsatisfiable,
    /// Resource limit exceeded
    #[error("resource limit exceeded")]
    ResourceLimit,
}

/// Configuration for RC2 solver
#[derive(Debug, Clone)]
pub struct Rc2Config {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Use incremental approach
    pub incremental: bool,
    /// Stratify by weight
    pub stratified: bool,
    /// Cardinality encoding to use
    pub encoding: CardinalityEncoding,
}

impl Default for Rc2Config {
    fn default() -> Self {
        Self {
            max_iterations: 100000,
            incremental: true,
            stratified: true,
            encoding: CardinalityEncoding::Totalizer,
        }
    }
}

/// Statistics from RC2 solving
#[derive(Debug, Clone, Default)]
pub struct Rc2Stats {
    /// Number of SAT solver calls
    pub sat_calls: u32,
    /// Number of cores found
    pub cores_found: u32,
    /// Number of cardinality constraints added
    pub cardinality_constraints: u32,
    /// Total number of relaxation variables
    pub relaxation_vars: u32,
}

/// RC2 MaxSAT solver
pub struct Rc2Solver {
    /// SAT solver
    solver: SatSolver,
    /// Soft clauses
    soft_clauses: Vec<SoftClause>,
    /// Configuration
    config: Rc2Config,
    /// Statistics
    stats: Rc2Stats,
    /// Relaxation variables for each soft clause
    relax_vars: FxHashMap<SoftId, Lit>,
    /// Current cost
    cost: Weight,
    /// Assumption literals for core extraction
    assumptions: Vec<Lit>,
    /// Next variable ID
    next_var: u32,
}

impl Rc2Solver {
    /// Create a new RC2 solver
    pub fn new() -> Self {
        Self::with_config(Rc2Config::default())
    }

    /// Create a new RC2 solver with configuration
    pub fn with_config(config: Rc2Config) -> Self {
        Self {
            solver: SatSolver::new(),
            soft_clauses: Vec::new(),
            config,
            stats: Rc2Stats::default(),
            relax_vars: FxHashMap::default(),
            cost: Weight::zero(),
            assumptions: Vec::new(),
            next_var: 0,
        }
    }

    /// Add a hard clause
    pub fn add_hard(&mut self, lits: impl IntoIterator<Item = Lit>) {
        let clause: Vec<_> = lits.into_iter().collect();
        // Track the maximum variable index
        for lit in &clause {
            let var_idx = lit.var().index() as u32;
            if var_idx >= self.next_var {
                self.next_var = var_idx + 1;
            }
        }
        self.solver.add_clause(clause.iter().copied());
    }

    /// Add a soft clause
    pub fn add_soft(&mut self, id: SoftId, lits: impl IntoIterator<Item = Lit>, weight: Weight) {
        let clause = SoftClause::new(id, lits, weight);
        // Track the maximum variable index
        for lit in &clause.lits {
            let var_idx = lit.var().index() as u32;
            if var_idx >= self.next_var {
                self.next_var = var_idx + 1;
            }
        }
        self.soft_clauses.push(clause);
    }

    /// Initialize the solver
    fn initialize(&mut self) {
        // Collect clause data first to avoid borrow issues
        let clause_data: Vec<_> = self
            .soft_clauses
            .iter()
            .map(|c| (c.id, c.lits.clone()))
            .collect();

        // Add relaxation variables to soft clauses
        for (id, lits) in clause_data {
            let relax_var = self.allocate_var();
            self.relax_vars.insert(id, Lit::pos(relax_var));

            // Add clause with relaxation variable: (clause ∨ relax)
            let mut relaxed_clause = lits.to_vec();
            relaxed_clause.push(Lit::pos(relax_var));
            self.solver.add_clause(relaxed_clause.iter().copied());

            self.stats.relaxation_vars += 1;
        }
    }

    /// Allocate a new variable
    fn allocate_var(&mut self) -> Var {
        let var = Var::new(self.next_var);
        self.next_var += 1;
        var
    }

    /// Solve the MaxSAT instance
    pub fn solve(&mut self) -> Result<MaxSatResult, Rc2Error> {
        self.initialize();

        // Stratify by weight if configured
        if self.config.stratified {
            return self.solve_stratified();
        }

        // Main RC2 loop
        for _ in 0..self.config.max_iterations {
            // Prepare assumptions (negate all relaxation variables)
            self.assumptions.clear();
            for relax_var in self.relax_vars.values() {
                self.assumptions.push(relax_var.negate());
            }

            self.stats.sat_calls += 1;
            let (result, core) = self.solver.solve_with_assumptions(&self.assumptions);

            match result {
                SolverResult::Sat => {
                    // Found satisfying assignment with current cost
                    return Ok(MaxSatResult::Optimal);
                }
                SolverResult::Unsat => {
                    // Extract core and process it
                    let core_lits = core.unwrap_or_default();
                    let core_ids = self.extract_core_from_lits(&core_lits);
                    if core_ids.is_empty() {
                        return Err(Rc2Error::Unsatisfiable);
                    }

                    self.process_core(core_ids)?;
                    self.stats.cores_found += 1;
                }
                SolverResult::Unknown => {
                    return Ok(MaxSatResult::Unknown);
                }
            }
        }

        Ok(MaxSatResult::Unknown)
    }

    /// Extract the unsatisfiable core from failed assumptions
    fn extract_core_from_lits(&self, failed: &[Lit]) -> Vec<SoftId> {
        let mut core = Vec::new();

        for lit in failed {
            // Find which soft clause this relaxation variable belongs to
            for (id, relax_var) in &self.relax_vars {
                if *relax_var == lit.negate() {
                    core.push(*id);
                    break;
                }
            }
        }

        core
    }

    /// Process a core by adding cardinality constraints
    fn process_core(&mut self, core: Vec<SoftId>) -> Result<(), Rc2Error> {
        if core.is_empty() {
            return Ok(());
        }

        // Find minimum weight in the core
        let min_weight = core
            .iter()
            .filter_map(|id| self.soft_clauses.iter().find(|c| c.id == *id))
            .map(|c| &c.weight)
            .min()
            .cloned()
            .unwrap_or_else(Weight::zero);

        // Update cost
        self.cost = self.cost.add(&min_weight);

        // Collect relaxation variables from the core
        let core_lits: Vec<Lit> = core
            .iter()
            .filter_map(|id| self.relax_vars.get(id).copied())
            .collect();

        if core_lits.is_empty() {
            return Ok(());
        }

        // Add AtMost cardinality constraint: at most (|core| - 1) of the relaxation variables can be true
        // This forces at least one soft clause in the core to be satisfied
        let bound = if !core_lits.is_empty() {
            core_lits.len() - 1
        } else {
            0
        };

        // Encode cardinality constraint
        let (clauses, _, new_next_var) =
            encode_at_most_k(&core_lits, bound, self.config.encoding, self.next_var);

        // Add clauses to solver
        for clause in &clauses {
            self.solver.add_clause(clause.lits.iter().copied());
        }

        // Update next_var
        self.next_var = new_next_var;
        self.stats.cardinality_constraints += 1;

        // Create new relaxation variables for soft clauses with reduced weight
        for id in &core {
            if let Some(clause_idx) = self.soft_clauses.iter().position(|c| c.id == *id) {
                let clause = &mut self.soft_clauses[clause_idx];

                // Reduce weight
                clause.weight = clause.weight.sub(&min_weight);

                // If weight is now zero, remove the relaxation variable
                if clause.weight.is_zero() {
                    self.relax_vars.remove(id);
                }
            }
        }

        Ok(())
    }

    /// Solve with stratification by weight
    fn solve_stratified(&mut self) -> Result<MaxSatResult, Rc2Error> {
        // Group soft clauses by weight
        let mut weight_levels: Vec<Weight> =
            self.soft_clauses.iter().map(|c| c.weight.clone()).collect();
        weight_levels.sort_unstable();
        weight_levels.dedup();

        // Solve from highest weight to lowest
        for weight in weight_levels.iter().rev() {
            let relevant_clauses: Vec<_> = self
                .soft_clauses
                .iter()
                .filter(|c| &c.weight == weight)
                .map(|c| c.id)
                .collect();

            // Solve for this weight level
            self.solve_level(&relevant_clauses, weight)?;
        }

        Ok(MaxSatResult::Optimal)
    }

    /// Solve for a specific weight level
    fn solve_level(&mut self, clause_ids: &[SoftId], _weight: &Weight) -> Result<(), Rc2Error> {
        loop {
            // Prepare assumptions for this level
            self.assumptions.clear();
            for id in clause_ids {
                if let Some(relax_var) = self.relax_vars.get(id) {
                    self.assumptions.push(relax_var.negate());
                }
            }

            self.stats.sat_calls += 1;
            let (result, core) = self.solver.solve_with_assumptions(&self.assumptions);

            match result {
                SolverResult::Sat => {
                    return Ok(());
                }
                SolverResult::Unsat => {
                    let core_lits = core.unwrap_or_default();
                    let core_ids = self.extract_core_from_lits(&core_lits);
                    if core_ids.is_empty() {
                        return Err(Rc2Error::Unsatisfiable);
                    }

                    self.process_core(core_ids)?;
                    self.stats.cores_found += 1;
                }
                SolverResult::Unknown => {
                    return Ok(());
                }
            }
        }
    }

    /// Get the current cost
    pub fn cost(&self) -> &Weight {
        &self.cost
    }

    /// Get statistics
    pub fn stats(&self) -> &Rc2Stats {
        &self.stats
    }

    /// Get the model (if SAT)
    pub fn get_model(&self) -> &[oxiz_sat::LBool] {
        self.solver.model()
    }
}

impl Default for Rc2Solver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc2_solver_new() {
        let solver = Rc2Solver::new();
        assert_eq!(solver.stats().sat_calls, 0);
        assert!(solver.cost().is_zero());
    }

    #[test]
    fn test_rc2_simple() {
        let mut solver = Rc2Solver::new();

        // Add hard clauses: (1 ∨ 2)
        solver.add_hard([Lit::from_dimacs(1), Lit::from_dimacs(2)]);

        // Add soft clauses with weight 1 each
        solver.add_soft(SoftId(0), [Lit::from_dimacs(-1)], Weight::from(1)); // Prefer ¬1
        solver.add_soft(SoftId(1), [Lit::from_dimacs(-2)], Weight::from(1)); // Prefer ¬2

        let result = solver.solve();
        assert!(result.is_ok());

        // Should have cost 1 (one soft clause must be violated)
        assert_eq!(*solver.cost(), Weight::from(1));
    }

    #[test]
    fn test_rc2_weighted() {
        let config = Rc2Config {
            max_iterations: 1000,
            incremental: true,
            stratified: false, // Use non-stratified for now
            encoding: CardinalityEncoding::Totalizer,
        };
        let mut solver = Rc2Solver::with_config(config);

        // Add soft clauses with different weights
        solver.add_soft(SoftId(0), [Lit::from_dimacs(1)], Weight::from(5));
        solver.add_soft(SoftId(1), [Lit::from_dimacs(2)], Weight::from(3));
        solver.add_soft(SoftId(2), [Lit::from_dimacs(3)], Weight::from(1));

        // Hard constraint: at most one can be true
        solver.add_hard([Lit::from_dimacs(-1), Lit::from_dimacs(-2)]);
        solver.add_hard([Lit::from_dimacs(-1), Lit::from_dimacs(-3)]);
        solver.add_hard([Lit::from_dimacs(-2), Lit::from_dimacs(-3)]);

        let result = solver.solve();
        assert!(result.is_ok(), "Solve failed: {:?}", result);

        // Should satisfy the clause with weight 5, violating the others (cost = 3 + 1 = 4)
        let cost = solver.cost();
        assert_eq!(*cost, Weight::from(4), "Expected cost 4, got {:?}", cost);
    }

    #[test]
    fn test_rc2_config() {
        let config = Rc2Config {
            max_iterations: 1000,
            incremental: true,
            stratified: false,
            encoding: CardinalityEncoding::Sorted,
        };

        let solver = Rc2Solver::with_config(config);
        assert_eq!(solver.config.max_iterations, 1000);
        assert!(!solver.config.stratified);
    }
}
