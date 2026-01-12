//! Configuration and parameters for solver tuning
//!
//! This module provides structures for configuring solver behavior,
//! including search parameters, heuristics, and optimizations.

use std::time::Duration;

/// Configuration for the solver
#[derive(Debug, Clone, Default)]
pub struct Config {
    /// General solver parameters
    pub general: GeneralParams,
    /// SAT solver parameters
    pub sat: SatParams,
    /// Simplification parameters
    pub simplify: SimplifyParams,
    /// Resource limits
    pub limits: ResourceLimits,
}

/// General solver parameters
#[derive(Debug, Clone)]
pub struct GeneralParams {
    /// Verbosity level (0 = quiet, 1 = normal, 2 = verbose, 3 = debug)
    pub verbosity: u8,
    /// Random seed for reproducibility
    pub random_seed: u64,
    /// Enable proof production
    pub produce_proofs: bool,
    /// Enable model production
    pub produce_models: bool,
    /// Enable unsat core production
    pub produce_unsat_cores: bool,
    /// Incremental solving mode
    pub incremental: bool,
}

/// SAT solver parameters
#[derive(Debug, Clone)]
pub struct SatParams {
    /// Initial restart interval
    pub restart_base: u32,
    /// Restart interval multiplier
    pub restart_factor: f64,
    /// Clause activity decay factor (0.0 to 1.0)
    pub clause_decay: f64,
    /// Variable activity decay factor (0.0 to 1.0)
    pub var_decay: f64,
    /// Initial number of conflicts before restart
    pub restart_first: u32,
    /// Learned clause deletion strategy
    pub clause_deletion: ClauseDeletionStrategy,
    /// Maximum learnt clause size to keep
    pub max_learnt_size: Option<usize>,
    /// Fraction of clauses to delete during cleanup
    pub clause_deletion_fraction: f64,
    /// Phase saving mode
    pub phase_saving: PhaseSaving,
    /// Use VSIDS (Variable State Independent Decaying Sum) heuristic
    pub use_vsids: bool,
}

/// Simplification parameters
#[derive(Debug, Clone)]
pub struct SimplifyParams {
    /// Enable simplification
    pub enable: bool,
    /// Maximum simplification iterations
    pub max_iterations: usize,
    /// Enable subsumption checking
    pub subsumption: bool,
    /// Enable variable elimination
    pub variable_elimination: bool,
    /// Enable blocked clause elimination
    pub blocked_clause_elimination: bool,
    /// Enable equivalent literal detection
    pub equiv_literals: bool,
}

/// Resource limits for the solver
#[derive(Debug, Clone, Default)]
pub struct ResourceLimits {
    /// Maximum solving time (None = unlimited)
    pub time_limit: Option<Duration>,
    /// Maximum number of decisions (None = unlimited)
    pub decision_limit: Option<u64>,
    /// Maximum number of conflicts (None = unlimited)
    pub conflict_limit: Option<u64>,
    /// Maximum memory usage in bytes (None = unlimited)
    pub memory_limit: Option<u64>,
}

/// Clause deletion strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClauseDeletionStrategy {
    /// Never delete learned clauses
    None,
    /// Delete based on activity
    Activity,
    /// Delete based on LBD (Literal Block Distance)
    Lbd,
    /// Combination of activity and LBD
    Hybrid,
}

/// Phase saving mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseSaving {
    /// No phase saving
    None,
    /// Save last assigned phase
    Last,
    /// Save phase from first assignment
    First,
}

impl Config {
    /// Create a new configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for SAT problems
    #[must_use]
    pub fn for_sat() -> Self {
        let mut config = Self::default();
        config.sat.use_vsids = true;
        config.sat.phase_saving = PhaseSaving::Last;
        config.simplify.enable = true;
        config
    }

    /// Create a configuration optimized for SMT problems
    #[must_use]
    pub fn for_smt() -> Self {
        let mut config = Self::default();
        config.simplify.enable = true;
        config.simplify.max_iterations = 5;
        config
    }

    /// Create a configuration for quick solving (less preprocessing)
    #[must_use]
    pub fn quick() -> Self {
        let mut config = Self::default();
        config.simplify.enable = false;
        config.sat.restart_first = 10;
        config
    }

    /// Create a configuration for thorough solving (more preprocessing)
    #[must_use]
    pub fn thorough() -> Self {
        let mut config = Self::default();
        config.simplify.enable = true;
        config.simplify.max_iterations = 10;
        config.simplify.subsumption = true;
        config.simplify.variable_elimination = true;
        config
    }

    /// Set the verbosity level
    pub fn set_verbosity(&mut self, level: u8) -> &mut Self {
        self.general.verbosity = level;
        self
    }

    /// Set the random seed
    pub fn set_random_seed(&mut self, seed: u64) -> &mut Self {
        self.general.random_seed = seed;
        self
    }

    /// Enable/disable proof production
    pub fn set_produce_proofs(&mut self, enable: bool) -> &mut Self {
        self.general.produce_proofs = enable;
        self
    }

    /// Enable/disable model production
    pub fn set_produce_models(&mut self, enable: bool) -> &mut Self {
        self.general.produce_models = enable;
        self
    }

    /// Enable/disable unsat core production
    pub fn set_produce_unsat_cores(&mut self, enable: bool) -> &mut Self {
        self.general.produce_unsat_cores = enable;
        self
    }

    /// Set time limit
    pub fn set_time_limit(&mut self, limit: Option<Duration>) -> &mut Self {
        self.limits.time_limit = limit;
        self
    }

    /// Set decision limit
    pub fn set_decision_limit(&mut self, limit: Option<u64>) -> &mut Self {
        self.limits.decision_limit = limit;
        self
    }

    /// Set conflict limit
    pub fn set_conflict_limit(&mut self, limit: Option<u64>) -> &mut Self {
        self.limits.conflict_limit = limit;
        self
    }

    /// Set memory limit in bytes
    pub fn set_memory_limit(&mut self, limit: Option<u64>) -> &mut Self {
        self.limits.memory_limit = limit;
        self
    }
}

impl Default for GeneralParams {
    fn default() -> Self {
        Self {
            verbosity: 1,
            random_seed: 0,
            produce_proofs: false,
            produce_models: true,
            produce_unsat_cores: false,
            incremental: false,
        }
    }
}

impl Default for SatParams {
    fn default() -> Self {
        Self {
            restart_base: 100,
            restart_factor: 1.5,
            clause_decay: 0.95,
            var_decay: 0.95,
            restart_first: 100,
            clause_deletion: ClauseDeletionStrategy::Activity,
            max_learnt_size: None,
            clause_deletion_fraction: 0.5,
            phase_saving: PhaseSaving::Last,
            use_vsids: true,
        }
    }
}

impl Default for SimplifyParams {
    fn default() -> Self {
        Self {
            enable: true,
            max_iterations: 3,
            subsumption: false,
            variable_elimination: false,
            blocked_clause_elimination: false,
            equiv_literals: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.general.verbosity, 1);
        assert_eq!(config.general.random_seed, 0);
        assert!(!config.general.produce_proofs);
        assert!(config.general.produce_models);
        assert!(!config.general.produce_unsat_cores);
    }

    #[test]
    fn test_sat_config() {
        let config = Config::for_sat();
        assert!(config.sat.use_vsids);
        assert_eq!(config.sat.phase_saving, PhaseSaving::Last);
        assert!(config.simplify.enable);
    }

    #[test]
    fn test_smt_config() {
        let config = Config::for_smt();
        assert!(config.simplify.enable);
        assert_eq!(config.simplify.max_iterations, 5);
    }

    #[test]
    fn test_quick_config() {
        let config = Config::quick();
        assert!(!config.simplify.enable);
        assert_eq!(config.sat.restart_first, 10);
    }

    #[test]
    fn test_thorough_config() {
        let config = Config::thorough();
        assert!(config.simplify.enable);
        assert_eq!(config.simplify.max_iterations, 10);
        assert!(config.simplify.subsumption);
        assert!(config.simplify.variable_elimination);
    }

    #[test]
    fn test_set_verbosity() {
        let mut config = Config::new();
        config.set_verbosity(3);
        assert_eq!(config.general.verbosity, 3);
    }

    #[test]
    fn test_set_random_seed() {
        let mut config = Config::new();
        config.set_random_seed(12345);
        assert_eq!(config.general.random_seed, 12345);
    }

    #[test]
    fn test_set_produce_proofs() {
        let mut config = Config::new();
        config.set_produce_proofs(true);
        assert!(config.general.produce_proofs);
    }

    #[test]
    fn test_set_limits() {
        let mut config = Config::new();

        config.set_time_limit(Some(Duration::from_secs(60)));
        assert_eq!(config.limits.time_limit, Some(Duration::from_secs(60)));

        config.set_decision_limit(Some(1000));
        assert_eq!(config.limits.decision_limit, Some(1000));

        config.set_conflict_limit(Some(500));
        assert_eq!(config.limits.conflict_limit, Some(500));

        config.set_memory_limit(Some(1024 * 1024 * 100));
        assert_eq!(config.limits.memory_limit, Some(1024 * 1024 * 100));
    }

    #[test]
    fn test_clause_deletion_strategy() {
        assert_ne!(
            ClauseDeletionStrategy::None,
            ClauseDeletionStrategy::Activity
        );
        assert_eq!(ClauseDeletionStrategy::Lbd, ClauseDeletionStrategy::Lbd);
    }

    #[test]
    fn test_phase_saving() {
        assert_ne!(PhaseSaving::None, PhaseSaving::Last);
        assert_eq!(PhaseSaving::First, PhaseSaving::First);
    }

    #[test]
    fn test_builder_pattern() {
        let mut config = Config::new();
        config
            .set_verbosity(2)
            .set_random_seed(42)
            .set_produce_proofs(true)
            .set_time_limit(Some(Duration::from_secs(30)));

        assert_eq!(config.general.verbosity, 2);
        assert_eq!(config.general.random_seed, 42);
        assert!(config.general.produce_proofs);
        assert_eq!(config.limits.time_limit, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert!(limits.time_limit.is_none());
        assert!(limits.decision_limit.is_none());
        assert!(limits.conflict_limit.is_none());
        assert!(limits.memory_limit.is_none());
    }
}
