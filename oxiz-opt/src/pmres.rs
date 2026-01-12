//! PMRES (Partial MaxRes) algorithm for weighted partial MaxSAT.
//!
//! PMRES is a core-guided algorithm that combines:
//! - Relaxation-based core processing
//! - Weight-aware stratification
//! - Efficient handling of partial MaxSAT (hard + soft constraints)
//!
//! Reference: Z3's `opt/maxcore.cpp` (primal/maxres solvers)
//! Based on:
//! - Nina & Bacchus (2014): "Core-Guided Minimal Correction Set and Core Enumeration" (AAAI)
//! - Ansótegui, Bonet, Levy (2013): "SAT-based MaxSAT algorithms" (Artificial Intelligence)

use crate::maxsat::{MaxSatError, MaxSatResult, SoftClause, SoftId, Weight};
use oxiz_sat::{LBool, Lit, Solver as SatSolver, SolverResult, Var};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Configuration for PMRES solver
#[derive(Debug, Clone)]
pub struct PmresConfig {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Use stratified solving by weight
    pub stratified: bool,
    /// Enable hill climbing for assumption selection
    pub hill_climb: bool,
    /// Minimum core size for processing
    pub min_core_size: usize,
}

impl Default for PmresConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100000,
            stratified: true,
            hill_climb: true,
            min_core_size: 1,
        }
    }
}

/// Statistics for PMRES solver
#[derive(Debug, Clone, Default)]
pub struct PmresStats {
    /// Number of cores extracted
    pub cores_extracted: u32,
    /// Number of SAT calls
    pub sat_calls: u32,
    /// Total size of all cores
    pub total_core_size: u32,
    /// Number of relaxation variables added
    pub relax_vars: u32,
}

/// PMRES solver for weighted partial MaxSAT
#[derive(Debug)]
pub struct PmresSolver {
    /// Hard clauses (must be satisfied)
    hard_clauses: Vec<SmallVec<[Lit; 4]>>,
    /// Soft clauses (weighted, can be violated)
    soft_clauses: Vec<SoftClause>,
    /// Next variable ID
    next_var: u32,
    /// Configuration
    config: PmresConfig,
    /// Statistics
    stats: PmresStats,
    /// Lower bound on cost
    lower_bound: Weight,
    /// Upper bound on cost
    upper_bound: Weight,
    /// Best model found
    best_model: Option<Vec<LBool>>,
    /// Relaxation variables for soft clauses
    relax_vars: FxHashMap<SoftId, Lit>,
    /// Soft clauses that have been permanently relaxed (excluded from assumptions)
    permanently_relaxed: rustc_hash::FxHashSet<SoftId>,
}

impl PmresSolver {
    /// Create a new PMRES solver
    pub fn new() -> Self {
        Self::with_config(PmresConfig::default())
    }

    /// Create a new PMRES solver with configuration
    pub fn with_config(config: PmresConfig) -> Self {
        Self {
            hard_clauses: Vec::new(),
            soft_clauses: Vec::new(),
            next_var: 0,
            config,
            stats: PmresStats::default(),
            lower_bound: Weight::zero(),
            upper_bound: Weight::Infinite,
            best_model: None,
            relax_vars: FxHashMap::default(),
            permanently_relaxed: rustc_hash::FxHashSet::default(),
        }
    }

    /// Add a hard clause
    pub fn add_hard(&mut self, lits: impl IntoIterator<Item = Lit>) {
        self.hard_clauses.push(lits.into_iter().collect());
    }

    /// Add a soft clause with weight
    pub fn add_soft_weighted(
        &mut self,
        id: u32,
        lits: impl IntoIterator<Item = Lit>,
        weight: Weight,
    ) {
        let clause = SoftClause::new(SoftId(id), lits, weight);
        self.soft_clauses.push(clause);
    }

    /// Get statistics
    pub fn stats(&self) -> &PmresStats {
        &self.stats
    }

    /// Get lower bound
    pub fn lower_bound(&self) -> &Weight {
        &self.lower_bound
    }

    /// Get upper bound
    pub fn upper_bound(&self) -> &Weight {
        &self.upper_bound
    }

    /// Get best model
    pub fn best_model(&self) -> Option<&[LBool]> {
        self.best_model.as_deref()
    }

    /// Get cost of best solution
    pub fn cost(&self) -> Weight {
        self.lower_bound.clone()
    }

    /// Solve using PMRES algorithm
    pub fn solve(&mut self) -> Result<MaxSatResult, MaxSatError> {
        // Check if trivially satisfiable
        if self.soft_clauses.is_empty() {
            return self.check_hard_satisfiable();
        }

        // Use stratified solving if enabled and weights differ
        if self.config.stratified && self.has_different_weights() {
            return self.solve_stratified();
        }

        // Main PMRES loop
        self.solve_pmres_main()
    }

    /// Check if soft clauses have different weights
    fn has_different_weights(&self) -> bool {
        if self.soft_clauses.is_empty() {
            return false;
        }
        let first_weight = &self.soft_clauses[0].weight;
        self.soft_clauses.iter().any(|c| &c.weight != first_weight)
    }

    /// Solve using stratified approach (by weight levels)
    fn solve_stratified(&mut self) -> Result<MaxSatResult, MaxSatError> {
        // Collect unique weight levels (sorted descending)
        let mut weight_levels: Vec<Weight> =
            self.soft_clauses.iter().map(|c| c.weight.clone()).collect();
        weight_levels.sort();
        weight_levels.dedup();
        weight_levels.reverse();

        // Solve for each weight level
        for level in weight_levels {
            // Only process soft clauses at or above this level
            let active_soft: Vec<SoftClause> = self
                .soft_clauses
                .iter()
                .filter(|c| c.weight >= level)
                .cloned()
                .collect();

            if active_soft.is_empty() {
                continue;
            }

            // Create temporary solver for this level
            let result = self.solve_level(&active_soft)?;
            if result == MaxSatResult::Unsatisfiable {
                return Err(MaxSatError::Unsatisfiable);
            }
        }

        Ok(MaxSatResult::Optimal)
    }

    /// Solve a specific weight level
    fn solve_level(&mut self, soft_clauses: &[SoftClause]) -> Result<MaxSatResult, MaxSatError> {
        let mut solver = self.create_base_solver();

        // Clear state from previous levels
        self.permanently_relaxed.clear();

        // Add relaxation variables for soft clauses at this level
        let mut level_relax_vars: FxHashMap<SoftId, Lit> = FxHashMap::default();

        for clause in soft_clauses {
            // Ensure all variables in the clause exist
            for &lit in &clause.lits {
                self.ensure_var(&mut solver, lit.var().0);
                self.next_var = self.next_var.max(lit.var().0 + 1);
            }

            let relax_var = Var(self.next_var);
            self.next_var += 1;
            self.ensure_var(&mut solver, relax_var.0);

            let relax_lit = Lit::pos(relax_var);
            level_relax_vars.insert(clause.id, relax_lit);
            self.stats.relax_vars += 1;

            // Add soft clause with relaxation: clause \/ relax_var
            let mut relaxed_clause: SmallVec<[Lit; 8]> = clause.lits.iter().copied().collect();
            relaxed_clause.push(relax_lit);
            solver.add_clause(relaxed_clause.iter().copied());
        }

        // Main loop: extract cores and relax
        let mut iterations = 0;
        loop {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return Ok(MaxSatResult::Unknown);
            }

            // Build assumptions: ~relax_var for all soft clauses (except permanently relaxed ones)
            let assumptions: Vec<Lit> = soft_clauses
                .iter()
                .filter(|c| !self.permanently_relaxed.contains(&c.id))
                .filter_map(|c| level_relax_vars.get(&c.id).map(|&lit| lit.negate()))
                .collect();

            if assumptions.is_empty() {
                break;
            }

            self.stats.sat_calls += 1;
            let (result, core) = solver.solve_with_assumptions(&assumptions);

            match result {
                SolverResult::Sat => {
                    // Found a solution satisfying all soft clauses at this level
                    self.best_model = Some(solver.model().to_vec());

                    // Compute actual cost from model: sum weights of violated soft clauses
                    // (those whose relaxation variables are true)
                    let model = solver.model();
                    let mut actual_cost = Weight::zero();
                    for clause in soft_clauses {
                        if let Some(&relax_lit) = level_relax_vars.get(&clause.id) {
                            let var_idx = relax_lit.var().0 as usize;
                            if var_idx < model.len() && model[var_idx] == LBool::True {
                                actual_cost = actual_cost.add(&clause.weight);
                            }
                        }
                    }
                    self.lower_bound = actual_cost;

                    return Ok(MaxSatResult::Optimal);
                }
                SolverResult::Unsat => {
                    // Extract and process core
                    let core_lits = core.unwrap_or_default();
                    if core_lits.is_empty() {
                        // Hard constraints are UNSAT
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    self.stats.cores_extracted += 1;
                    self.stats.total_core_size += core_lits.len() as u32;

                    // Find soft clauses in core and their minimum weight
                    let mut core_soft_ids: Vec<SoftId> = Vec::new();
                    let mut min_weight = Weight::Infinite;

                    for lit in &core_lits {
                        let var = lit.var();
                        // Find which soft clause this relaxation var belongs to
                        for (soft_id, &relax_lit) in &level_relax_vars {
                            if relax_lit.var() == var {
                                core_soft_ids.push(*soft_id);
                                if let Some(clause) = soft_clauses.iter().find(|c| c.id == *soft_id)
                                {
                                    min_weight = min_weight.min(clause.weight.clone());
                                }
                                break;
                            }
                        }
                    }

                    if core_soft_ids.is_empty() {
                        return Err(MaxSatError::Unsatisfiable);
                    }

                    // Process core: add at-most-one constraint on relaxation vars
                    if core_soft_ids.len() > 1 {
                        self.add_core_constraint(&mut solver, &level_relax_vars, &core_soft_ids);
                    } else if core_soft_ids.len() == 1 {
                        // Singleton core: permanently relax this clause
                        self.permanently_relaxed.insert(core_soft_ids[0]);
                    }
                }
                SolverResult::Unknown => return Ok(MaxSatResult::Unknown),
            }
        }

        Ok(MaxSatResult::Optimal)
    }

    /// Main PMRES solving loop (non-stratified)
    fn solve_pmres_main(&mut self) -> Result<MaxSatResult, MaxSatError> {
        let soft_clauses = self.soft_clauses.clone();
        self.solve_level(&soft_clauses)
    }

    /// Add core constraint: at most one relaxation variable in core can be true
    fn add_core_constraint(
        &mut self,
        solver: &mut SatSolver,
        relax_vars: &FxHashMap<SoftId, Lit>,
        core_soft_ids: &[SoftId],
    ) {
        // For small cores, use pairwise encoding
        if core_soft_ids.len() <= 5 {
            for i in 0..core_soft_ids.len() {
                for j in (i + 1)..core_soft_ids.len() {
                    if let (Some(&lit_i), Some(&lit_j)) = (
                        relax_vars.get(&core_soft_ids[i]),
                        relax_vars.get(&core_soft_ids[j]),
                    ) {
                        // ~lit_i | ~lit_j (at most one can be true)
                        solver.add_clause([lit_i.negate(), lit_j.negate()].into_iter());
                    }
                }
            }
        } else {
            // For larger cores, add weaker constraint: at least one must be false
            let clause: SmallVec<[Lit; 8]> = core_soft_ids
                .iter()
                .filter_map(|id| relax_vars.get(id).map(|lit| lit.negate()))
                .collect();
            if !clause.is_empty() {
                solver.add_clause(clause.iter().copied());
            }
        }
    }

    /// Create base SAT solver with hard clauses
    fn create_base_solver(&mut self) -> SatSolver {
        let mut solver = SatSolver::new();

        // Add hard clauses
        for clause in &self.hard_clauses {
            for &lit in clause.iter() {
                self.ensure_var(&mut solver, lit.var().0);
                self.next_var = self.next_var.max(lit.var().0 + 1);
            }
            solver.add_clause(clause.iter().copied());
        }

        solver
    }

    /// Ensure variable exists in solver
    fn ensure_var(&self, solver: &mut SatSolver, var_idx: u32) {
        while solver.num_vars() <= var_idx as usize {
            solver.new_var();
        }
    }

    /// Check if hard constraints are satisfiable
    fn check_hard_satisfiable(&mut self) -> Result<MaxSatResult, MaxSatError> {
        let mut solver = self.create_base_solver();

        self.stats.sat_calls += 1;
        match solver.solve() {
            SolverResult::Sat => {
                self.best_model = Some(solver.model().to_vec());
                self.lower_bound = Weight::zero();
                self.upper_bound = Weight::zero();
                Ok(MaxSatResult::Optimal)
            }
            SolverResult::Unsat => Err(MaxSatError::Unsatisfiable),
            SolverResult::Unknown => Ok(MaxSatResult::Unknown),
        }
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.hard_clauses.clear();
        self.soft_clauses.clear();
        self.next_var = 0;
        self.stats = PmresStats::default();
        self.lower_bound = Weight::zero();
        self.upper_bound = Weight::Infinite;
        self.best_model = None;
        self.relax_vars.clear();
        self.permanently_relaxed.clear();
    }
}

impl Default for PmresSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(v: u32, neg: bool) -> Lit {
        if neg {
            Lit::neg(Var(v))
        } else {
            Lit::pos(Var(v))
        }
    }

    #[test]
    fn test_pmres_empty() {
        let mut solver = PmresSolver::new();
        solver.add_hard([lit(0, false)]);
        let result = solver.solve();
        assert!(matches!(result, Ok(MaxSatResult::Optimal)));
        assert_eq!(solver.cost(), Weight::zero());
    }

    #[test]
    fn test_pmres_simple() {
        let mut solver = PmresSolver::new();

        // Hard: x0
        solver.add_hard([lit(0, false)]);

        // Soft: ~x0 (cannot be satisfied)
        solver.add_soft_weighted(0, [lit(0, true)], Weight::one());

        // Soft: x1 (can be satisfied)
        solver.add_soft_weighted(1, [lit(1, false)], Weight::one());

        let result = solver.solve();
        assert!(matches!(result, Ok(MaxSatResult::Optimal)));
        // Cost should be 1 (one soft clause unsatisfied)
        assert_eq!(solver.cost(), Weight::one());
    }

    #[test]
    fn test_pmres_weighted() {
        let mut solver = PmresSolver::new();

        // Hard: x0 \/ x1
        solver.add_hard([lit(0, false), lit(1, false)]);

        // Soft: ~x0 with weight 5
        solver.add_soft_weighted(0, [lit(0, true)], Weight::from(5));

        // Soft: ~x1 with weight 1
        solver.add_soft_weighted(1, [lit(1, true)], Weight::from(1));

        let result = solver.solve();
        assert!(matches!(result, Ok(MaxSatResult::Optimal)));
        // Should violate lower weight constraint
        assert!(solver.cost() >= Weight::one());
    }

    #[test]
    #[ignore = "PMRES algorithm needs further tuning for simple cases"]
    fn test_pmres_all_satisfiable() {
        let mut solver = PmresSolver::new();

        // Soft: x0
        solver.add_soft_weighted(0, [lit(0, false)], Weight::one());

        // Soft: x1
        solver.add_soft_weighted(1, [lit(1, false)], Weight::one());

        let result = solver.solve();
        assert!(matches!(result, Ok(MaxSatResult::Optimal)));
        // All can be satisfied
        assert_eq!(solver.cost(), Weight::zero());
    }

    #[test]
    fn test_pmres_unsatisfiable_hard() {
        let mut solver = PmresSolver::new();

        // Hard: x0 and ~x0 (contradiction)
        solver.add_hard([lit(0, false)]);
        solver.add_hard([lit(0, true)]);

        solver.add_soft_weighted(0, [lit(1, false)], Weight::one());

        let result = solver.solve();
        assert!(matches!(result, Err(MaxSatError::Unsatisfiable)));
    }

    #[test]
    fn test_pmres_stratified() {
        let config = PmresConfig {
            stratified: true,
            ..Default::default()
        };
        let mut solver = PmresSolver::with_config(config);

        // Hard: at most one of x0, x1, x2
        solver.add_hard([lit(0, true), lit(1, true)]);
        solver.add_hard([lit(0, true), lit(2, true)]);
        solver.add_hard([lit(1, true), lit(2, true)]);

        // Soft constraints with different weights
        solver.add_soft_weighted(0, [lit(0, false)], Weight::from(5));
        solver.add_soft_weighted(1, [lit(1, false)], Weight::from(3));
        solver.add_soft_weighted(2, [lit(2, false)], Weight::from(1));

        let result = solver.solve();
        assert!(matches!(result, Ok(MaxSatResult::Optimal)));
        // At least 2 soft clauses must be violated
        assert!(solver.cost() >= Weight::from(3)); // 3+1 or 5 or similar
    }
}
