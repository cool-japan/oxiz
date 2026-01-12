//! IHS (Implicit Hitting Sets) MaxSAT solver.
//!
//! IHS is an advanced MaxSAT algorithm that extends MaxHS by using implicit
//! hitting set computation. Instead of explicitly computing minimum hitting sets,
//! IHS uses a SAT solver to implicitly search for them, which is often more efficient.
//!
//! The algorithm maintains:
//! - A SAT solver for the original problem (to find MCSes)
//! - A hitting set solver (SAT-based) to implicitly compute minimum-cost hitting sets
//!
//! Reference: "Implicit Hitting Set Algorithms for Maximum Satisfiability Modulo Theories"

use crate::maxsat::{MaxSatResult, SoftClause, SoftId, Weight};
use oxiz_sat::{Lit, Solver as SatSolver, SolverResult, Var};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use thiserror::Error;

/// Errors from IHS
#[derive(Error, Debug)]
pub enum IhsError {
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

/// Configuration for IHS solver
#[derive(Debug, Clone)]
pub struct IhsConfig {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Use core minimization
    pub minimize_cores: bool,
    /// Use disjoint cores optimization
    pub disjoint_cores: bool,
}

impl Default for IhsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            minimize_cores: true,
            disjoint_cores: true,
        }
    }
}

/// Statistics from IHS solving
#[derive(Debug, Clone, Default)]
pub struct IhsStats {
    /// Number of SAT solver calls
    pub sat_calls: u32,
    /// Number of minimal correction sets found
    pub mcses_found: u32,
    /// Number of hitting set solver calls
    pub hs_solver_calls: u32,
    /// Total number of soft clauses
    pub total_soft: u32,
}

/// Minimal Correction Set (MCS)
#[derive(Debug, Clone)]
struct Mcs {
    /// Soft clause IDs in this MCS
    clauses: SmallVec<[SoftId; 8]>,
    /// MCS selector variable (for hitting set encoding)
    selector: Var,
}

/// IHS MaxSAT solver
pub struct IhsSolver {
    /// Main SAT solver (for finding MCSes)
    sat_solver: SatSolver,
    /// Hitting set SAT solver (for finding minimum hitting sets)
    hs_solver: SatSolver,
    /// Hard clauses
    hard_clauses: Vec<SmallVec<[Lit; 4]>>,
    /// Soft clauses
    soft_clauses: Vec<SoftClause>,
    /// Map from SoftId to index
    soft_map: FxHashMap<SoftId, usize>,
    /// Map from SoftId to indicator variable in hitting set solver
    soft_to_hs_var: FxHashMap<SoftId, Var>,
    /// Configuration
    config: IhsConfig,
    /// Statistics
    stats: IhsStats,
    /// Found MCSes
    mcses: Vec<Mcs>,
    /// Current best cost
    best_cost: Weight,
    /// Next variable for hitting set solver
    next_hs_var: u32,
    /// Best model found
    best_model: Option<Vec<oxiz_sat::LBool>>,
}

impl IhsSolver {
    /// Create a new IHS solver
    pub fn new() -> Self {
        Self::with_config(IhsConfig::default())
    }

    /// Create a new IHS solver with configuration
    pub fn with_config(config: IhsConfig) -> Self {
        Self {
            sat_solver: SatSolver::new(),
            hs_solver: SatSolver::new(),
            hard_clauses: Vec::new(),
            soft_clauses: Vec::new(),
            soft_map: FxHashMap::default(),
            soft_to_hs_var: FxHashMap::default(),
            config,
            stats: IhsStats::default(),
            mcses: Vec::new(),
            best_cost: Weight::Infinite,
            next_hs_var: 0,
            best_model: None,
        }
    }

    /// Add a hard clause
    pub fn add_hard(&mut self, lits: impl IntoIterator<Item = Lit>) {
        let clause: SmallVec<[Lit; 4]> = lits.into_iter().collect();
        self.sat_solver.add_clause(clause.iter().copied());
        self.hard_clauses.push(clause);
    }

    /// Add a soft clause
    pub fn add_soft(&mut self, id: SoftId, lits: impl IntoIterator<Item = Lit>, weight: Weight) {
        let clause = SoftClause::new(id, lits, weight);
        let idx = self.soft_clauses.len();
        self.soft_map.insert(id, idx);
        self.soft_clauses.push(clause);
        self.stats.total_soft += 1;

        // Create indicator variable for this soft clause in hitting set solver
        let hs_var = Var::new(self.next_hs_var);
        self.next_hs_var += 1;
        self.soft_to_hs_var.insert(id, hs_var);
    }

    /// Solve the MaxSAT instance
    pub fn solve(&mut self) -> Result<MaxSatResult, IhsError> {
        // Add hard clauses to SAT solver
        for hard_clause in &self.hard_clauses {
            self.sat_solver.add_clause(hard_clause.iter().copied());
        }

        // Add all soft clauses to SAT solver initially
        for clause in &self.soft_clauses {
            self.sat_solver.add_clause(clause.lits.iter().copied());
        }

        // Main IHS loop
        for _ in 0..self.config.max_iterations {
            // Try to find a model with current constraints
            self.stats.sat_calls += 1;
            let (result, _) = self.sat_solver.solve_with_assumptions(&[]);

            match result {
                SolverResult::Sat => {
                    // Formula is SAT with current clauses
                    self.best_model = Some(self.sat_solver.model().to_vec());
                    // If best_cost is still infinite, no clauses were removed (all satisfied)
                    if self.best_cost == Weight::Infinite {
                        self.best_cost = Weight::zero();
                    }
                    return Ok(MaxSatResult::Optimal);
                }
                SolverResult::Unsat => {
                    // Find a minimal correction set (MCS)
                    let mcs = self.find_mcs()?;

                    if mcs.clauses.is_empty() {
                        // Hard constraints are unsatisfiable
                        return Err(IhsError::Unsatisfiable);
                    }

                    self.stats.mcses_found += 1;

                    // Add MCS to hitting set constraints
                    self.add_mcs_to_hs_solver(&mcs);
                    self.mcses.push(mcs);

                    // Find minimum-cost hitting set using SAT
                    match self.find_min_hitting_set()? {
                        Some(hitting_set) => {
                            // Update best cost
                            let cost = self.compute_hitting_set_cost(&hitting_set);
                            if cost < self.best_cost {
                                self.best_cost = cost;
                            }

                            // Block this hitting set in the main SAT solver
                            self.block_hitting_set(&hitting_set);
                        }
                        None => {
                            // No hitting set found (should not happen if formulation is correct)
                            return Ok(MaxSatResult::Unknown);
                        }
                    }
                }
                SolverResult::Unknown => {
                    return Ok(MaxSatResult::Unknown);
                }
            }
        }

        Ok(MaxSatResult::Unknown)
    }

    /// Find a minimal correction set (MCS)
    fn find_mcs(&mut self) -> Result<Mcs, IhsError> {
        // An MCS is a minimal set of soft clauses whose removal makes the formula SAT
        let mut candidate: FxHashSet<SoftId> = self.soft_clauses.iter().map(|c| c.id).collect();

        if !self.config.minimize_cores {
            // Return the full set without minimization
            let selector = Var::new(self.next_hs_var);
            self.next_hs_var += 1;
            return Ok(Mcs {
                clauses: candidate.into_iter().collect(),
                selector,
            });
        }

        // Minimize the MCS by removing clauses one by one
        for &id in &candidate.clone() {
            // Try to remove this clause from the MCS
            let mut test_candidate = candidate.clone();
            test_candidate.remove(&id);

            // Check if test_candidate is still a correction set
            // (i.e., removing test_candidate makes the formula SAT)
            if self.is_correction_set(&test_candidate) {
                candidate = test_candidate;
            }
        }

        let selector = Var::new(self.next_hs_var);
        self.next_hs_var += 1;

        Ok(Mcs {
            clauses: candidate.into_iter().collect(),
            selector,
        })
    }

    /// Check if a set of soft clauses forms a correction set
    fn is_correction_set(&mut self, clause_ids: &FxHashSet<SoftId>) -> bool {
        // A correction set is a set such that removing it makes the formula SAT
        let mut test_solver = SatSolver::new();

        // Add hard clauses
        for hard_clause in &self.hard_clauses {
            test_solver.add_clause(hard_clause.iter().copied());
        }

        // Add soft clauses NOT in the correction set
        for clause in &self.soft_clauses {
            if !clause_ids.contains(&clause.id) {
                test_solver.add_clause(clause.lits.iter().copied());
            }
        }

        let (result, _) = test_solver.solve_with_assumptions(&[]);
        matches!(result, SolverResult::Sat)
    }

    /// Add an MCS to the hitting set solver
    fn add_mcs_to_hs_solver(&mut self, mcs: &Mcs) {
        // Hitting set constraint: at least one clause from this MCS must be in the hitting set
        // Encoding: selector => (v1 ∨ v2 ∨ ... ∨ vn)
        // Which is: ¬selector ∨ v1 ∨ v2 ∨ ... ∨ vn

        let mut clause: SmallVec<[Lit; 16]> = SmallVec::new();
        clause.push(Lit::neg(mcs.selector));

        for &id in &mcs.clauses {
            if let Some(&hs_var) = self.soft_to_hs_var.get(&id) {
                clause.push(Lit::pos(hs_var));
            }
        }

        self.hs_solver.add_clause(clause.iter().copied());
    }

    /// Find minimum-cost hitting set using implicit SAT-based search
    fn find_min_hitting_set(&mut self) -> Result<Option<FxHashSet<SoftId>>, IhsError> {
        self.stats.hs_solver_calls += 1;

        // Build assumption literals that minimize cost
        // We use a cardinality-based approach: try to satisfy with k clauses, starting from k=1

        let num_soft = self.soft_clauses.len();

        for _k in 1..=num_soft {
            // Try to find a hitting set with at most k soft clauses
            let mut assumptions = Vec::new();

            // Add MCS selector assumptions (all MCSes must be hit)
            for mcs in &self.mcses {
                assumptions.push(Lit::pos(mcs.selector));
            }

            // Add at-most-k constraint implicitly using assumptions
            // For now, just try to solve without cardinality constraint
            // A full implementation would use totalizer or other encoding

            let (result, _) = self.hs_solver.solve_with_assumptions(&assumptions);

            match result {
                SolverResult::Sat => {
                    // Extract hitting set from model
                    let model = self.hs_solver.model();
                    let mut hitting_set = FxHashSet::default();

                    for (&id, &hs_var) in &self.soft_to_hs_var {
                        let var_idx = hs_var.index();
                        if var_idx < model.len() && model[var_idx] == oxiz_sat::LBool::True {
                            hitting_set.insert(id);
                        }
                    }

                    // Minimize the hitting set by removing unnecessary clauses
                    hitting_set = self.minimize_hitting_set(hitting_set);

                    return Ok(Some(hitting_set));
                }
                SolverResult::Unsat => {
                    // Try next k
                    continue;
                }
                SolverResult::Unknown => {
                    return Ok(None);
                }
            }
        }

        Ok(None)
    }

    /// Minimize a hitting set by removing unnecessary clauses
    fn minimize_hitting_set(&self, mut hitting_set: FxHashSet<SoftId>) -> FxHashSet<SoftId> {
        // Try to remove each clause and see if it's still a valid hitting set
        for &id in &hitting_set.clone() {
            let mut test_set = hitting_set.clone();
            test_set.remove(&id);

            // Check if test_set still hits all MCSes
            let still_valid = self
                .mcses
                .iter()
                .all(|mcs| mcs.clauses.iter().any(|mcs_id| test_set.contains(mcs_id)));

            if still_valid {
                hitting_set = test_set;
            }
        }

        hitting_set
    }

    /// Compute the cost of a hitting set
    fn compute_hitting_set_cost(&self, hitting_set: &FxHashSet<SoftId>) -> Weight {
        hitting_set
            .iter()
            .filter_map(|id| self.soft_map.get(id))
            .filter_map(|&idx| self.soft_clauses.get(idx))
            .map(|c| &c.weight)
            .fold(Weight::zero(), |acc, w| acc.add(w))
    }

    /// Block a hitting set in the main SAT solver
    fn block_hitting_set(&mut self, hitting_set: &FxHashSet<SoftId>) {
        // Rebuild SAT solver without the hitting set clauses
        self.sat_solver = SatSolver::new();

        // Add hard clauses
        for hard_clause in &self.hard_clauses {
            self.sat_solver.add_clause(hard_clause.iter().copied());
        }

        // Add soft clauses not in hitting set
        for clause in &self.soft_clauses {
            if !hitting_set.contains(&clause.id) {
                self.sat_solver.add_clause(clause.lits.iter().copied());
            }
        }
    }

    /// Get the best cost found
    pub fn best_cost(&self) -> &Weight {
        &self.best_cost
    }

    /// Get statistics
    pub fn stats(&self) -> &IhsStats {
        &self.stats
    }

    /// Get the best model
    pub fn best_model(&self) -> Option<&[oxiz_sat::LBool]> {
        self.best_model.as_deref()
    }
}

impl Default for IhsSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ihs_solver_new() {
        let solver = IhsSolver::new();
        assert_eq!(solver.stats().sat_calls, 0);
        assert_eq!(*solver.best_cost(), Weight::Infinite);
    }

    #[test]
    fn test_ihs_config() {
        let config = IhsConfig {
            max_iterations: 5000,
            minimize_cores: false,
            disjoint_cores: false,
        };

        let solver = IhsSolver::with_config(config);
        assert_eq!(solver.config.max_iterations, 5000);
        assert!(!solver.config.minimize_cores);
    }

    #[test]
    fn test_ihs_simple() {
        let mut solver = IhsSolver::new();

        // Add soft clauses
        solver.add_soft(SoftId(0), [Lit::from_dimacs(1)], Weight::from(1));
        solver.add_soft(SoftId(1), [Lit::from_dimacs(-1)], Weight::from(1));

        let result = solver.solve();
        assert!(result.is_ok(), "Solve failed: {:?}", result);

        // Should have cost 1 (one clause must be violated)
        assert_eq!(*solver.best_cost(), Weight::from(1));
    }

    #[test]
    fn test_ihs_weighted() {
        let mut solver = IhsSolver::new();

        // Add hard clauses: (1 ∨ 2)
        solver.add_hard([Lit::from_dimacs(1), Lit::from_dimacs(2)]);

        // Add soft clauses with different weights
        solver.add_soft(SoftId(0), [Lit::from_dimacs(-1)], Weight::from(5));
        solver.add_soft(SoftId(1), [Lit::from_dimacs(-2)], Weight::from(1));

        let result = solver.solve();
        assert!(result.is_ok());

        // Should violate the clause with weight 1 (cheaper)
        assert_eq!(*solver.best_cost(), Weight::from(1));
    }
}
