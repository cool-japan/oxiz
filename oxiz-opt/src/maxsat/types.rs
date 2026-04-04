//! Types for MaxSAT solving: errors, results, configuration, and the solver struct.

use super::core::{SoftClause, SoftId, Weight};
use oxiz_sat::{LBool, Lit};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use thiserror::Error;

/// Errors that can occur during MaxSAT solving
#[derive(Error, Debug)]
pub enum MaxSatError {
    /// No solution exists (hard constraints unsatisfiable)
    #[error("hard constraints unsatisfiable")]
    Unsatisfiable,
    /// Solver error
    #[error("solver error: {0}")]
    SolverError(String),
    /// Resource limit exceeded
    #[error("resource limit exceeded")]
    ResourceLimit,
}

/// Result of MaxSAT solving
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaxSatResult {
    /// Optimal solution found
    Optimal,
    /// Solution found but optimality not proven
    Satisfiable,
    /// No solution exists
    Unsatisfiable,
    /// Could not determine within limits
    Unknown,
}

impl std::fmt::Display for MaxSatResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaxSatResult::Optimal => write!(f, "optimal"),
            MaxSatResult::Satisfiable => write!(f, "satisfiable"),
            MaxSatResult::Unsatisfiable => write!(f, "unsatisfiable"),
            MaxSatResult::Unknown => write!(f, "unknown"),
        }
    }
}

/// MaxSAT solver configuration
#[derive(Debug, Clone)]
pub struct MaxSatConfig {
    /// Maximum number of iterations
    pub max_iterations: u32,
    /// Use stratified solving (by weight levels)
    pub stratified: bool,
    /// Algorithm to use
    pub algorithm: MaxSatAlgorithm,
    /// Enable core minimization (reduce core size after extraction)
    pub core_minimization: bool,
}

impl Default for MaxSatConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100000,
            stratified: true,
            algorithm: MaxSatAlgorithm::FuMalik,
            core_minimization: true,
        }
    }
}

impl MaxSatConfig {
    /// Create a new builder for MaxSatConfig
    pub fn builder() -> MaxSatConfigBuilder {
        MaxSatConfigBuilder::default()
    }
}

/// Builder for MaxSatConfig
#[derive(Debug, Clone)]
pub struct MaxSatConfigBuilder {
    max_iterations: u32,
    stratified: bool,
    algorithm: MaxSatAlgorithm,
    core_minimization: bool,
}

impl Default for MaxSatConfigBuilder {
    fn default() -> Self {
        Self {
            max_iterations: 100000,
            stratified: true,
            algorithm: MaxSatAlgorithm::FuMalik,
            core_minimization: true,
        }
    }
}

impl MaxSatConfigBuilder {
    /// Set the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set whether to use stratified solving
    pub fn stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        self
    }

    /// Set the algorithm to use
    pub fn algorithm(mut self, algorithm: MaxSatAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set whether to enable core minimization
    pub fn core_minimization(mut self, core_minimization: bool) -> Self {
        self.core_minimization = core_minimization;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MaxSatConfig {
        MaxSatConfig {
            max_iterations: self.max_iterations,
            stratified: self.stratified,
            algorithm: self.algorithm,
            core_minimization: self.core_minimization,
        }
    }
}

/// MaxSAT algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaxSatAlgorithm {
    /// Fu-Malik core-guided algorithm
    FuMalik,
    /// Opportunistic Literal Learning
    Oll,
    /// MSU3 iterative relaxation
    Msu3,
    /// Weighted MaxSAT (for weighted instances)
    WMax,
    /// PMRES (Partial MaxSAT Resolution)
    Pmres,
}

/// Statistics from MaxSAT solving
#[derive(Debug, Clone, Default)]
pub struct MaxSatStats {
    /// Number of SAT calls
    pub sat_calls: u32,
    /// Number of cores extracted
    pub cores_extracted: u32,
    /// Number of relaxation variables added
    pub relax_vars_added: u32,
    /// Total core sizes
    pub total_core_size: u32,
    /// Number of cores minimized
    pub cores_minimized: u32,
    /// Total literals removed by core minimization
    pub core_min_lits_removed: u32,
}

/// MaxSAT solver
#[derive(Debug)]
pub struct MaxSatSolver {
    /// Hard clauses
    pub(super) hard_clauses: Vec<SmallVec<[Lit; 4]>>,
    /// Soft clauses
    pub(super) soft_clauses: Vec<SoftClause>,
    /// Next soft ID
    pub(super) next_soft_id: u32,
    /// Configuration
    pub(super) config: MaxSatConfig,
    /// Statistics
    pub(super) stats: MaxSatStats,
    /// Lower bound on cost
    pub(super) lower_bound: Weight,
    /// Upper bound on cost
    pub(super) upper_bound: Weight,
    /// Best model found
    pub(super) best_model: Option<Vec<LBool>>,
    /// Mapping from relaxation variable to soft clause ID
    pub(super) relax_to_soft: FxHashMap<Lit, SoftId>,
}

impl Default for MaxSatSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MaxSatSolver {
    /// Create a new MaxSAT solver
    pub fn new() -> Self {
        Self::with_config(MaxSatConfig::default())
    }

    /// Create a new MaxSAT solver with configuration
    pub fn with_config(config: MaxSatConfig) -> Self {
        Self {
            hard_clauses: Vec::new(),
            soft_clauses: Vec::new(),
            next_soft_id: 0,
            config,
            stats: MaxSatStats::default(),
            lower_bound: Weight::zero(),
            upper_bound: Weight::Infinite,
            best_model: None,
            relax_to_soft: FxHashMap::default(),
        }
    }

    /// Add a hard clause
    pub fn add_hard(&mut self, lits: impl IntoIterator<Item = Lit>) {
        self.hard_clauses.push(lits.into_iter().collect());
    }

    /// Add a soft clause with unit weight
    pub fn add_soft(&mut self, lits: impl IntoIterator<Item = Lit>) -> SoftId {
        self.add_soft_weighted(lits, Weight::one())
    }

    /// Add a soft clause with weight
    pub fn add_soft_weighted(
        &mut self,
        lits: impl IntoIterator<Item = Lit>,
        weight: Weight,
    ) -> SoftId {
        let id = SoftId(self.next_soft_id);
        self.next_soft_id += 1;
        let clause = SoftClause::new(id, lits, weight.clone());
        self.soft_clauses.push(clause);

        // Update upper bound
        self.upper_bound = self.upper_bound.add(&weight);

        id
    }

    /// Get the number of hard clauses
    pub fn num_hard(&self) -> usize {
        self.hard_clauses.len()
    }

    /// Get the number of soft clauses
    pub fn num_soft(&self) -> usize {
        self.soft_clauses.len()
    }

    /// Get the lower bound
    pub fn lower_bound(&self) -> &Weight {
        &self.lower_bound
    }

    /// Get the upper bound
    pub fn upper_bound(&self) -> &Weight {
        &self.upper_bound
    }

    /// Get statistics
    pub fn stats(&self) -> &MaxSatStats {
        &self.stats
    }

    /// Get the best model (if found)
    pub fn best_model(&self) -> Option<&[LBool]> {
        self.best_model.as_deref()
    }

    /// Get the cost of the best solution
    pub fn cost(&self) -> Weight {
        self.lower_bound.clone()
    }

    /// Check if a soft clause is satisfied in the best model
    pub fn is_soft_satisfied(&self, id: SoftId) -> bool {
        self.soft_clauses
            .get(id.0 as usize)
            .is_some_and(|c| c.is_satisfied())
    }

    /// Get satisfied soft clause IDs
    pub fn satisfied_soft(&self) -> impl Iterator<Item = SoftId> + '_ {
        self.soft_clauses
            .iter()
            .filter(|c| c.is_satisfied())
            .map(|c| c.id)
    }

    /// Get unsatisfied soft clause IDs
    pub fn unsatisfied_soft(&self) -> impl Iterator<Item = SoftId> + '_ {
        self.soft_clauses
            .iter()
            .filter(|c| !c.is_satisfied())
            .map(|c| c.id)
    }

    /// Reset the solver
    pub fn reset(&mut self) {
        self.hard_clauses.clear();
        self.soft_clauses.clear();
        self.next_soft_id = 0;
        self.stats = MaxSatStats::default();
        self.lower_bound = Weight::zero();
        self.upper_bound = Weight::Infinite;
        self.best_model = None;
        self.relax_to_soft.clear();
    }
}
