//! Configuration options for theory solvers
//!
//! This module provides configurable parameters for controlling
//! resource limits, algorithmic choices, and optimization strategies
//! across all theory solvers.

use std::time::Duration;

/// Configuration for simplex-based arithmetic solvers
#[derive(Debug, Clone)]
pub struct SimplexConfig {
    /// Maximum number of pivot operations before timeout
    pub max_pivots: usize,
    /// Pivoting rule to use
    pub pivoting_rule: PivotingRule,
    /// Enable bound tightening optimizations
    pub enable_bound_tightening: bool,
    /// Enable coefficient normalization
    pub enable_normalization: bool,
}

impl Default for SimplexConfig {
    fn default() -> Self {
        Self {
            max_pivots: 10_000,
            pivoting_rule: PivotingRule::Bland,
            enable_bound_tightening: true,
            enable_normalization: true,
        }
    }
}

/// Pivoting rule for simplex algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotingRule {
    /// Bland's rule (anti-cycling, guaranteed termination)
    Bland,
    /// Dantzig's rule (greedy, fastest for most problems)
    Dantzig,
    /// Steepest edge rule (sophisticated, best for hard problems)
    SteepestEdge,
    /// Partial pricing (check subset of candidates, faster for large problems)
    PartialPricing,
}

/// Configuration for linear integer arithmetic solver
#[derive(Debug, Clone)]
pub struct LiaConfig {
    /// Maximum branch-and-bound tree depth
    pub max_depth: usize,
    /// Maximum number of cutting planes to generate
    pub max_cuts: usize,
    /// Enable Gomory cuts
    pub enable_gomory_cuts: bool,
    /// Enable MIR (Mixed Integer Rounding) cuts
    pub enable_mir_cuts: bool,
    /// Enable CG (Chvátal-Gomory) cuts
    pub enable_cg_cuts: bool,
    /// Branching heuristic
    pub branching_heuristic: BranchingHeuristic,
    /// Strong branching: max simplex iterations per candidate evaluation
    pub strong_branching_iterations: usize,
    /// Strong branching: max number of candidates to evaluate (0 = all)
    pub strong_branching_candidates: usize,
}

impl Default for LiaConfig {
    fn default() -> Self {
        Self {
            max_depth: 1000,
            max_cuts: 100,
            enable_gomory_cuts: true,
            enable_mir_cuts: true,
            enable_cg_cuts: true,
            branching_heuristic: BranchingHeuristic::FirstFractional,
            strong_branching_iterations: 10,
            strong_branching_candidates: 5,
        }
    }
}

/// Branching heuristic for integer arithmetic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchingHeuristic {
    /// Select first fractional variable found
    FirstFractional,
    /// Select most fractional variable (closest to 0.5)
    MostFractional,
    /// Select variable with highest pseudo-cost
    PseudoCost,
    /// Strong branching: evaluate both branch directions before committing
    /// Most expensive but often reduces tree size by 20-50%
    StrongBranching,
}

/// Configuration for bit-vector solver
#[derive(Debug, Clone)]
pub struct BvConfig {
    /// Enable word-level propagation before bit-blasting
    pub enable_word_level_propagation: bool,
    /// Enable lazy bit-blasting (only blast when needed)
    pub enable_lazy_blasting: bool,
    /// Bit-width threshold for switching strategies
    pub large_bitvector_threshold: u32,
}

impl Default for BvConfig {
    fn default() -> Self {
        Self {
            enable_word_level_propagation: true,
            enable_lazy_blasting: true,
            large_bitvector_threshold: 64,
        }
    }
}

/// Configuration for theory combination
#[derive(Debug, Clone)]
pub struct CombinationConfig {
    /// Maximum size of lemma cache (0 = unlimited)
    pub max_lemma_cache_size: usize,
    /// Enable lemma subsumption checking
    pub enable_lemma_subsumption: bool,
    /// Enable conflict minimization
    pub enable_conflict_minimization: bool,
    /// Theory combination mode
    pub combination_mode: CombinationMode,
}

impl Default for CombinationConfig {
    fn default() -> Self {
        Self {
            max_lemma_cache_size: 10_000,
            enable_lemma_subsumption: true,
            enable_conflict_minimization: true,
            combination_mode: CombinationMode::NelsonOppen,
        }
    }
}

/// Theory combination mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationMode {
    /// Classic Nelson-Oppen (equality propagation)
    NelsonOppen,
    /// Model-based theory combination (check arrangements)
    ModelBased,
    /// Delayed theory combination (lazy propagation)
    Delayed,
    /// Polite theory combination (more efficient for certain theory classes)
    Polite,
}

/// Global configuration for all theory solvers
#[derive(Debug, Clone, Default)]
pub struct TheoryConfig {
    /// Simplex configuration
    pub simplex: SimplexConfig,
    /// Linear integer arithmetic configuration
    pub lia: LiaConfig,
    /// Bit-vector configuration
    pub bv: BvConfig,
    /// Theory combination configuration
    pub combination: CombinationConfig,
    /// Global timeout (None = no timeout)
    pub timeout: Option<Duration>,
}

impl TheoryConfig {
    /// Create a new configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for performance
    #[must_use]
    pub fn fast() -> Self {
        Self {
            simplex: SimplexConfig {
                max_pivots: 50_000,
                pivoting_rule: PivotingRule::Dantzig,
                enable_bound_tightening: true,
                enable_normalization: true,
            },
            lia: LiaConfig {
                max_depth: 5000,
                max_cuts: 500,
                enable_gomory_cuts: true,
                enable_mir_cuts: true,
                enable_cg_cuts: true,
                branching_heuristic: BranchingHeuristic::StrongBranching,
                strong_branching_iterations: 20,
                strong_branching_candidates: 10,
            },
            bv: BvConfig {
                enable_word_level_propagation: true,
                enable_lazy_blasting: true,
                large_bitvector_threshold: 128,
            },
            combination: CombinationConfig {
                max_lemma_cache_size: 50_000,
                enable_lemma_subsumption: true,
                enable_conflict_minimization: true,
                combination_mode: CombinationMode::ModelBased,
            },
            timeout: Some(Duration::from_secs(300)),
        }
    }

    /// Create a configuration optimized for small problems
    #[must_use]
    pub fn small() -> Self {
        Self {
            simplex: SimplexConfig {
                max_pivots: 1_000,
                pivoting_rule: PivotingRule::Bland,
                enable_bound_tightening: false,
                enable_normalization: false,
            },
            lia: LiaConfig {
                max_depth: 100,
                max_cuts: 10,
                enable_gomory_cuts: false,
                enable_mir_cuts: false,
                enable_cg_cuts: false,
                branching_heuristic: BranchingHeuristic::FirstFractional,
                strong_branching_iterations: 5,
                strong_branching_candidates: 3,
            },
            bv: BvConfig {
                enable_word_level_propagation: false,
                enable_lazy_blasting: false,
                large_bitvector_threshold: 32,
            },
            combination: CombinationConfig {
                max_lemma_cache_size: 100,
                enable_lemma_subsumption: false,
                enable_conflict_minimization: false,
                combination_mode: CombinationMode::NelsonOppen,
            },
            timeout: Some(Duration::from_secs(10)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TheoryConfig::default();
        assert_eq!(config.simplex.max_pivots, 10_000);
        assert_eq!(config.lia.max_depth, 1000);
        assert_eq!(config.combination.max_lemma_cache_size, 10_000);
    }

    #[test]
    fn test_fast_config() {
        let config = TheoryConfig::fast();
        assert_eq!(config.simplex.max_pivots, 50_000);
        assert_eq!(config.lia.max_depth, 5000);
        assert_eq!(config.simplex.pivoting_rule, PivotingRule::Dantzig);
    }

    #[test]
    fn test_small_config() {
        let config = TheoryConfig::small();
        assert_eq!(config.simplex.max_pivots, 1_000);
        assert_eq!(config.lia.max_depth, 100);
        assert!(!config.simplex.enable_bound_tightening);
    }
}
