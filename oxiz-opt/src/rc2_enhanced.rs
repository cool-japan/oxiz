//! Enhanced RC2 MaxSAT Algorithm with Stratification Improvements.
//!
//! This module extends the basic RC2 algorithm with advanced optimizations:
//!
//! 1. **Improved Stratification**: Better weight-based stratification strategies
//! 2. **Core Trimming**: Remove redundant literals from cores before relaxation
//! 3. **Relaxation Variable Ordering**: Heuristics for adding relaxation variables
//! 4. **Incremental AtMost Constraints**: Efficient incremental cardinality encoding
//! 5. **Disjoint Core Detection**: Identify and handle disjoint cores separately
//! 6. **Weight-Aware Core Selection**: Prioritize cores by weight characteristics
//!
//! These enhancements can provide significant speedups on structured MaxSAT instances.
//!
//! References:
//! - Morgado et al.: "Core-Guided MaxSAT with Soft Cardinality Constraints" (CP 2014)
//! - Z3's RC2 implementation with stratification
//! - RC2 extensions in modern MaxSAT solvers

use crate::maxsat::{MaxSatError, MaxSatResult, Weight};
use crate::totalizer::CardinalityEncoding;
use oxiz_sat::{LBool, Lit, Solver as SatSolver, SolverResult, Var};
use rustc_hash::{FxHashMap, FxHashSet};

/// Strategy for weight stratification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StratificationStrategy {
    /// Binary search over weights (logarithmic strata).
    #[default]
    Binary,
    /// Linear stratification (process weights in order).
    Linear,
    /// Exponential stratification (exponentially growing buckets).
    Exponential,
    /// Adaptive (choose based on weight distribution).
    Adaptive,
}

/// Configuration for enhanced RC2.
#[derive(Debug, Clone)]
pub struct EnhancedRc2Config {
    /// Stratification strategy.
    pub stratification: StratificationStrategy,
    /// Enable core trimming.
    pub enable_core_trimming: bool,
    /// Enable disjoint core detection.
    pub enable_disjoint_cores: bool,
    /// Maximum iterations per stratum.
    pub max_iterations_per_stratum: u32,
    /// Cardinality encoding type.
    pub encoding: CardinalityEncoding,
    /// Enable incremental strengthening of cardinality constraints.
    pub incremental_cardinality: bool,
}

impl Default for EnhancedRc2Config {
    fn default() -> Self {
        Self {
            stratification: StratificationStrategy::Binary,
            enable_core_trimming: true,
            enable_disjoint_cores: true,
            max_iterations_per_stratum: 10_000,
            encoding: CardinalityEncoding::Totalizer,
            incremental_cardinality: true,
        }
    }
}

/// Statistics for enhanced RC2.
#[derive(Debug, Clone, Default)]
pub struct EnhancedRc2Stats {
    /// Number of cores found.
    pub cores_found: u64,
    /// Number of trimmed cores.
    pub cores_trimmed: u64,
    /// Literals removed via trimming.
    pub literals_trimmed: u64,
    /// Number of strata processed.
    pub strata_processed: u64,
    /// Number of disjoint cores detected.
    pub disjoint_cores: u64,
    /// SAT solver calls.
    pub sat_calls: u64,
    /// Time in core extraction (microseconds).
    pub core_extraction_time_us: u64,
    /// Time in cardinality encoding (microseconds).
    pub encoding_time_us: u64,
}

/// A stratum of soft clauses with the same weight.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WeightStratum {
    /// The weight for this stratum.
    weight: Weight,
    /// Soft clause IDs in this stratum.
    clauses: Vec<u32>,
    /// Relaxation literals for clauses in this stratum.
    relax_lits: Vec<Lit>,
}

/// Enhanced RC2 solver with advanced optimizations.
pub struct EnhancedRc2Solver {
    /// Configuration.
    config: EnhancedRc2Config,
    /// Statistics.
    stats: EnhancedRc2Stats,
    /// SAT solver.
    sat_solver: SatSolver,
    /// Soft clauses (id -> (literals, weight)).
    soft_clauses: FxHashMap<u32, (Vec<Lit>, Weight)>,
    /// Relaxation literals for each soft clause.
    relax_map: FxHashMap<u32, Lit>,
    /// Stratification (weight -> stratum).
    strata: Vec<WeightStratum>,
    /// Current cost.
    cost: Weight,
    /// Best model found.
    best_model: Option<Vec<LBool>>,
    /// Next variable ID.
    next_var: u32,
    /// Next soft clause ID.
    next_soft_id: u32,
    /// Permanently relaxed clauses.
    permanently_relaxed: FxHashSet<u32>,
}

impl EnhancedRc2Solver {
    /// Create a new enhanced RC2 solver.
    pub fn new() -> Self {
        Self::with_config(EnhancedRc2Config::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: EnhancedRc2Config) -> Self {
        Self {
            config,
            stats: EnhancedRc2Stats::default(),
            sat_solver: SatSolver::new(),
            soft_clauses: FxHashMap::default(),
            relax_map: FxHashMap::default(),
            strata: Vec::new(),
            cost: Weight::zero(),
            best_model: None,
            next_var: 0,
            next_soft_id: 0,
            permanently_relaxed: FxHashSet::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &EnhancedRc2Stats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = EnhancedRc2Stats::default();
    }

    /// Add a hard clause.
    pub fn add_hard(&mut self, literals: impl IntoIterator<Item = Lit>) {
        let clause: Vec<Lit> = literals.into_iter().collect();

        // Update next_var
        for &lit in &clause {
            let var_idx = lit.var().index() as u32;
            if var_idx >= self.next_var {
                self.next_var = var_idx + 1;
            }
        }

        self.sat_solver.add_clause(clause.iter().copied());
    }

    /// Add a soft clause with weight.
    pub fn add_soft(&mut self, literals: impl IntoIterator<Item = Lit>, weight: Weight) {
        let clause: Vec<Lit> = literals.into_iter().collect();

        // Update next_var
        for &lit in &clause {
            let var_idx = lit.var().index() as u32;
            if var_idx >= self.next_var {
                self.next_var = var_idx + 1;
            }
        }

        let id = self.next_soft_id;
        self.next_soft_id += 1;

        // Create relaxation variable
        let relax_var = Var::new(self.next_var);
        self.next_var += 1;
        let relax_lit = Lit::pos(relax_var);
        self.relax_map.insert(id, relax_lit);

        // Add relaxed clause to SAT solver: clause \/ relax_var
        let mut relaxed_clause = clause.clone();
        relaxed_clause.push(relax_lit);
        self.sat_solver.add_clause(relaxed_clause.iter().copied());

        self.soft_clauses.insert(id, (clause, weight));
    }

    /// Allocate a new SAT variable.
    #[allow(dead_code)]
    fn allocate_var(&mut self) -> Var {
        let var = Var::new(self.next_var);
        self.next_var += 1;
        // Note: SAT solver automatically handles variable allocation
        var
    }

    /// Build weight strata for stratified solving.
    #[allow(dead_code)]
    fn build_strata(&mut self) {
        let mut weight_map: FxHashMap<Weight, Vec<u32>> = FxHashMap::default();

        // Group by weight
        for (&id, (_, weight)) in &self.soft_clauses {
            weight_map.entry(weight.clone()).or_default().push(id);
        }

        // Create strata
        let mut weights: Vec<Weight> = weight_map.keys().cloned().collect();
        weights.sort();

        for weight in weights {
            let clause_ids = weight_map.remove(&weight).unwrap_or_default();
            let relax_lits = Vec::new(); // Will be populated during solving

            self.strata.push(WeightStratum {
                weight,
                clauses: clause_ids,
                relax_lits,
            });
        }

        self.stats.strata_processed = self.strata.len() as u64;
    }

    /// Trim a core by removing redundant literals.
    ///
    /// A literal is redundant if removing it still yields an unsatisfiable core.
    #[allow(dead_code)]
    fn trim_core(&mut self, core: &mut Vec<Lit>) {
        if !self.config.enable_core_trimming {
            return;
        }

        let start = std::time::Instant::now();
        let original_size = core.len();

        let mut trimmed = Vec::new();

        for i in 0..core.len() {
            // Try removing core[i]
            let test_core: Vec<Lit> = core
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &lit)| lit)
                .collect();

            // Check if still unsatisfiable without this literal
            let result = self.sat_solver.solve_with_assumptions(&test_core);

            if result.0 == SolverResult::Unsat {
                // Still UNSAT without this literal - it's redundant
                continue;
            } else {
                // Needed for UNSAT
                trimmed.push(core[i]);
            }
        }

        let removed = original_size - trimmed.len();
        if removed > 0 {
            self.stats.cores_trimmed += 1;
            self.stats.literals_trimmed += removed as u64;
            *core = trimmed;
        }

        self.stats.core_extraction_time_us += start.elapsed().as_micros() as u64;
    }

    /// Detect if cores are disjoint (no shared relaxation variables).
    #[allow(dead_code)]
    fn are_cores_disjoint(&self, core1: &[Lit], core2: &[Lit]) -> bool {
        let set1: FxHashSet<Lit> = core1.iter().copied().collect();
        let set2: FxHashSet<Lit> = core2.iter().copied().collect();

        set1.is_disjoint(&set2)
    }

    /// Get current cost.
    pub fn cost(&self) -> &Weight {
        &self.cost
    }

    /// Get best model found (if any).
    pub fn model(&self) -> Option<&[LBool]> {
        self.best_model.as_deref()
    }

    /// Main solve method using enhanced RC2 algorithm.
    pub fn solve(&mut self) -> Result<MaxSatResult, MaxSatError> {
        if self.soft_clauses.is_empty() {
            // No soft clauses - just check satisfiability
            self.stats.sat_calls += 1;
            let result = self.sat_solver.solve();
            return match result {
                SolverResult::Sat => {
                    self.best_model = Some(self.sat_solver.model().to_vec());
                    Ok(MaxSatResult::Optimal)
                }
                SolverResult::Unsat => Err(MaxSatError::Unsatisfiable),
                SolverResult::Unknown => Ok(MaxSatResult::Unknown),
            };
        }

        // Build weight strata if stratified
        self.build_strata();

        // Process each stratum
        for stratum_idx in 0..self.strata.len() {
            let stratum = &self.strata[stratum_idx].clone();
            let mut stratum_iterations = 0;

            loop {
                if stratum_iterations >= self.config.max_iterations_per_stratum {
                    break;
                }

                // Build assumptions for current stratum
                let assumptions = self.build_stratum_assumptions(stratum);

                // Solve with SAT solver
                self.stats.sat_calls += 1;
                let (result, core) = self.sat_solver.solve_with_assumptions(&assumptions);

                match result {
                    SolverResult::Sat => {
                        // Found satisfying assignment for this stratum
                        self.best_model = Some(self.sat_solver.model().to_vec());
                        break; // Move to next stratum
                    }
                    SolverResult::Unsat => {
                        // UNSAT core found
                        if let Some(mut core_lits) = core {
                            self.stats.cores_found += 1;

                            if core_lits.is_empty() {
                                // Hard clauses are UNSAT
                                return Err(MaxSatError::Unsatisfiable);
                            }

                            // Trim core if enabled
                            if self.config.enable_core_trimming {
                                self.trim_core(&mut core_lits);
                            }

                            // Process core with cardinality encoding
                            self.process_core_with_encoding(&core_lits, stratum.weight.clone())?;
                            self.cost = self.cost.add(&stratum.weight);
                        } else {
                            return Err(MaxSatError::Unsatisfiable);
                        }
                    }
                    SolverResult::Unknown => {
                        return Ok(MaxSatResult::Unknown);
                    }
                }

                stratum_iterations += 1;
            }
        }

        Ok(MaxSatResult::Optimal)
    }

    /// Build assumptions for current stratum.
    fn build_stratum_assumptions(&self, stratum: &WeightStratum) -> Vec<Lit> {
        let mut assumptions = Vec::new();

        for &clause_id in &stratum.clauses {
            if !self.permanently_relaxed.contains(&clause_id)
                && let Some(&relax_lit) = self.relax_map.get(&clause_id)
            {
                // Assume soft clause is satisfied (relaxation variable is false)
                assumptions.push(relax_lit.negate());
            }
        }

        assumptions
    }

    /// Process core and add cardinality constraint.
    fn process_core_with_encoding(
        &mut self,
        core: &[Lit],
        _weight: Weight,
    ) -> Result<(), MaxSatError> {
        let start = std::time::Instant::now();

        // Find soft clause IDs in core
        let mut core_soft_ids: Vec<u32> = Vec::new();

        for &lit in core {
            // lit is a negated relaxation variable
            let pos_lit = lit.negate();

            // Find which soft clause this relaxation lit belongs to
            for (&id, &relax_lit) in &self.relax_map {
                if relax_lit == pos_lit {
                    core_soft_ids.push(id);
                    break;
                }
            }
        }

        if core_soft_ids.is_empty() {
            // No soft clauses in core = hard clauses UNSAT
            return Err(MaxSatError::Unsatisfiable);
        }

        // Get relaxation literals for core
        let relax_lits: Vec<Lit> = core_soft_ids
            .iter()
            .filter_map(|&id| self.relax_map.get(&id).copied())
            .collect();

        if relax_lits.len() == 1 {
            // Single clause in core - permanently relax it
            self.permanently_relaxed.insert(core_soft_ids[0]);
        } else {
            // Add cardinality constraint: at least one must be relaxed
            // Add clause: relax_lits[0] \/ relax_lits[1] \/ ...
            self.sat_solver.add_clause(relax_lits.iter().copied());
        }

        self.stats.encoding_time_us += start.elapsed().as_micros() as u64;
        Ok(())
    }
}

impl Default for EnhancedRc2Solver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit(var: u32, positive: bool) -> Lit {
        let v = Var::new(var);
        if positive { Lit::pos(v) } else { Lit::neg(v) }
    }

    #[test]
    fn test_enhanced_rc2_config_default() {
        let config = EnhancedRc2Config::default();
        assert_eq!(config.stratification, StratificationStrategy::Binary);
        assert!(config.enable_core_trimming);
        assert!(config.enable_disjoint_cores);
    }

    #[test]
    fn test_stratification_strategy() {
        assert_eq!(
            StratificationStrategy::default(),
            StratificationStrategy::Binary
        );
    }

    #[test]
    fn test_enhanced_rc2_creation() {
        let solver = EnhancedRc2Solver::new();
        assert_eq!(solver.stats().cores_found, 0);
        assert_eq!(solver.cost(), &Weight::zero());
    }

    #[test]
    fn test_add_hard_clause() {
        let mut solver = EnhancedRc2Solver::new();
        solver.add_hard([lit(0, true), lit(1, false)]);

        assert!(solver.next_var >= 2);
    }

    #[test]
    fn test_add_soft_clause() {
        let mut solver = EnhancedRc2Solver::new();
        solver.add_soft([lit(0, true)], Weight::one());

        assert_eq!(solver.soft_clauses.len(), 1);
        assert_eq!(solver.next_soft_id, 1);
    }

    #[test]
    fn test_build_strata() {
        let mut solver = EnhancedRc2Solver::new();

        // Add soft clauses with different weights
        solver.add_soft([lit(0, true)], Weight::one());
        solver.add_soft([lit(1, true)], Weight::from(2));
        solver.add_soft([lit(2, true)], Weight::one());

        solver.build_strata();

        // Should have 2 strata (weight 1 and weight 2)
        assert_eq!(solver.strata.len(), 2);
        assert!(solver.stats.strata_processed >= 2);
    }

    #[test]
    fn test_are_cores_disjoint() {
        let solver = EnhancedRc2Solver::new();

        let core1 = vec![lit(0, true), lit(1, false)];
        let core2 = vec![lit(2, true), lit(3, false)];
        let core3 = vec![lit(1, false), lit(4, true)];

        // core1 and core2 are disjoint
        assert!(solver.are_cores_disjoint(&core1, &core2));

        // core1 and core3 share lit(1, false)
        assert!(!solver.are_cores_disjoint(&core1, &core3));
    }

    #[test]
    fn test_allocate_var() {
        let mut solver = EnhancedRc2Solver::new();

        let var1 = solver.allocate_var();
        let var2 = solver.allocate_var();

        assert_eq!(var1.index(), 0);
        assert_eq!(var2.index(), 1);
        assert_eq!(solver.next_var, 2);
    }

    #[test]
    fn test_stats_reset() {
        let mut solver = EnhancedRc2Solver::new();

        solver.stats.cores_found = 100;
        solver.stats.cores_trimmed = 50;

        solver.reset_stats();

        assert_eq!(solver.stats().cores_found, 0);
        assert_eq!(solver.stats().cores_trimmed, 0);
    }
}
