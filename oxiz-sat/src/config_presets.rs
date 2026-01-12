//! Solver Configuration Presets
//!
//! Pre-configured solver profiles optimized for different problem classes.
//! These presets are based on extensive empirical testing and competition
//! results from modern SAT solvers.

use crate::solver::{RestartStrategy, SolverConfig};

/// Preset categories for different problem types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigPreset {
    /// Default balanced configuration
    Default,
    /// Optimized for industrial/structured problems
    Industrial,
    /// Optimized for random/uniform problems
    Random,
    /// Optimized for cryptographic problems
    Cryptographic,
    /// Optimized for hardware verification
    Hardware,
    /// Aggressive configuration for quick results
    Aggressive,
    /// Conservative configuration for hard problems
    Conservative,
    /// Glucose-style configuration
    Glucose,
    /// MiniSAT-style configuration
    MiniSat,
    /// CaDiCaL-style configuration
    CaDiCaL,
}

impl ConfigPreset {
    /// Get the solver configuration for this preset
    #[must_use]
    pub fn config(self) -> SolverConfig {
        match self {
            Self::Default => Self::default_config(),
            Self::Industrial => Self::industrial_config(),
            Self::Random => Self::random_config(),
            Self::Cryptographic => Self::cryptographic_config(),
            Self::Hardware => Self::hardware_config(),
            Self::Aggressive => Self::aggressive_config(),
            Self::Conservative => Self::conservative_config(),
            Self::Glucose => Self::glucose_config(),
            Self::MiniSat => Self::minisat_config(),
            Self::CaDiCaL => Self::cadical_config(),
        }
    }

    /// Default balanced configuration
    fn default_config() -> SolverConfig {
        SolverConfig::default()
    }

    /// Industrial/structured problems configuration
    ///
    /// Characteristics:
    /// - Heavy use of clause minimization
    /// - Glucose-style restarts
    /// - Aggressive inprocessing
    /// - LRB branching heuristic
    fn industrial_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 100,
            restart_multiplier: 1.5,
            clause_deletion_threshold: 15000,
            var_decay: 0.95,
            clause_decay: 0.999,
            random_polarity_prob: 0.02,
            restart_strategy: RestartStrategy::Glucose,
            enable_lazy_hyper_binary: true,
            use_chb_branching: false,
            use_lrb_branching: true, // LRB for structured problems
            enable_inprocessing: true,
            inprocessing_interval: 5000,
            enable_chronological_backtrack: true,
            chrono_backtrack_threshold: 100,
        }
    }

    /// Random/uniform problems configuration
    ///
    /// Characteristics:
    /// - VSIDS branching (classic)
    /// - Geometric restarts
    /// - Less aggressive preprocessing
    /// - Higher random polarity
    fn random_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 50,
            restart_multiplier: 2.0,
            clause_deletion_threshold: 10000,
            var_decay: 0.90,
            clause_decay: 0.95,
            random_polarity_prob: 0.10, // Higher randomness
            restart_strategy: RestartStrategy::Geometric,
            enable_lazy_hyper_binary: false,
            use_chb_branching: false,
            use_lrb_branching: false,   // VSIDS for random
            enable_inprocessing: false, // Less helpful for random
            inprocessing_interval: 10000,
            enable_chronological_backtrack: false,
            chrono_backtrack_threshold: 100,
        }
    }

    /// Cryptographic problems configuration
    ///
    /// Characteristics:
    /// - XOR-aware techniques
    /// - Longer restart intervals
    /// - CHB branching
    /// - Heavy clause minimization
    fn cryptographic_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 200,
            restart_multiplier: 1.3,
            clause_deletion_threshold: 20000,
            var_decay: 0.98,
            clause_decay: 0.999,
            random_polarity_prob: 0.01,
            restart_strategy: RestartStrategy::Luby,
            enable_lazy_hyper_binary: true,
            use_chb_branching: true, // CHB good for crypto
            use_lrb_branching: false,
            enable_inprocessing: true,
            inprocessing_interval: 10000,
            enable_chronological_backtrack: true,
            chrono_backtrack_threshold: 50,
        }
    }

    /// Hardware verification configuration
    ///
    /// Characteristics:
    /// - Similar to industrial but more aggressive
    /// - Gate detection and exploitation
    /// - LRB branching
    /// - Frequent restarts
    fn hardware_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 80,
            restart_multiplier: 1.4,
            clause_deletion_threshold: 12000,
            var_decay: 0.95,
            clause_decay: 0.999,
            random_polarity_prob: 0.02,
            restart_strategy: RestartStrategy::Glucose,
            enable_lazy_hyper_binary: true,
            use_chb_branching: false,
            use_lrb_branching: true,
            enable_inprocessing: true,
            inprocessing_interval: 3000, // More frequent
            enable_chronological_backtrack: true,
            chrono_backtrack_threshold: 100,
        }
    }

    /// Aggressive configuration for quick results
    ///
    /// Characteristics:
    /// - Frequent restarts
    /// - Aggressive clause deletion
    /// - High random polarity
    /// - Less preprocessing
    fn aggressive_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 30,
            restart_multiplier: 1.1,
            clause_deletion_threshold: 5000,
            var_decay: 0.85,
            clause_decay: 0.90,
            random_polarity_prob: 0.15,
            restart_strategy: RestartStrategy::Geometric,
            enable_lazy_hyper_binary: false,
            use_chb_branching: false,
            use_lrb_branching: false,
            enable_inprocessing: false,
            inprocessing_interval: 20000,
            enable_chronological_backtrack: false,
            chrono_backtrack_threshold: 100,
        }
    }

    /// Conservative configuration for hard problems
    ///
    /// Characteristics:
    /// - Longer restart intervals
    /// - Keep more clauses
    /// - Lower random polarity
    /// - Extensive preprocessing
    fn conservative_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 500,
            restart_multiplier: 2.0,
            clause_deletion_threshold: 50000,
            var_decay: 0.99,
            clause_decay: 0.999,
            random_polarity_prob: 0.01,
            restart_strategy: RestartStrategy::Luby,
            enable_lazy_hyper_binary: true,
            use_chb_branching: false,
            use_lrb_branching: true,
            enable_inprocessing: true,
            inprocessing_interval: 2000,
            enable_chronological_backtrack: true,
            chrono_backtrack_threshold: 200,
        }
    }

    /// Glucose-style configuration
    ///
    /// Based on Glucose SAT solver parameters
    fn glucose_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 100,
            restart_multiplier: 1.5,
            clause_deletion_threshold: 10000,
            var_decay: 0.95,
            clause_decay: 0.999,
            random_polarity_prob: 0.02,
            restart_strategy: RestartStrategy::Glucose,
            enable_lazy_hyper_binary: true,
            use_chb_branching: false,
            use_lrb_branching: false, // VSIDS like Glucose
            enable_inprocessing: false,
            inprocessing_interval: 10000,
            enable_chronological_backtrack: false,
            chrono_backtrack_threshold: 100,
        }
    }

    /// MiniSAT-style configuration
    ///
    /// Based on classic MiniSAT parameters
    fn minisat_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 100,
            restart_multiplier: 1.5,
            clause_deletion_threshold: 8000,
            var_decay: 0.95,
            clause_decay: 0.999,
            random_polarity_prob: 0.0,
            restart_strategy: RestartStrategy::Luby,
            enable_lazy_hyper_binary: false,
            use_chb_branching: false,
            use_lrb_branching: false, // Classic VSIDS
            enable_inprocessing: false,
            inprocessing_interval: 10000,
            enable_chronological_backtrack: false,
            chrono_backtrack_threshold: 100,
        }
    }

    /// CaDiCaL-style configuration
    ///
    /// Based on CaDiCaL SAT solver parameters
    fn cadical_config() -> SolverConfig {
        SolverConfig {
            restart_interval: 100,
            restart_multiplier: 1.4,
            clause_deletion_threshold: 12000,
            var_decay: 0.95,
            clause_decay: 0.999,
            random_polarity_prob: 0.01,
            restart_strategy: RestartStrategy::Glucose,
            enable_lazy_hyper_binary: true,
            use_chb_branching: false,
            use_lrb_branching: false, // VMTF in real CaDiCaL
            enable_inprocessing: true,
            inprocessing_interval: 4000,
            enable_chronological_backtrack: true,
            chrono_backtrack_threshold: 100,
        }
    }

    /// Get a description of this preset
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::Default => "Balanced configuration suitable for most problems",
            Self::Industrial => "Optimized for industrial/structured SAT instances",
            Self::Random => "Optimized for random/uniform SAT instances",
            Self::Cryptographic => "Optimized for cryptographic and XOR-heavy problems",
            Self::Hardware => "Optimized for hardware verification problems",
            Self::Aggressive => "Aggressive settings for quick results",
            Self::Conservative => "Conservative settings for hard/challenging problems",
            Self::Glucose => "Glucose SAT solver style configuration",
            Self::MiniSat => "Classic MiniSAT style configuration",
            Self::CaDiCaL => "CaDiCaL SAT solver style configuration",
        }
    }

    /// List all available presets
    #[must_use]
    pub fn all_presets() -> &'static [ConfigPreset] {
        &[
            Self::Default,
            Self::Industrial,
            Self::Random,
            Self::Cryptographic,
            Self::Hardware,
            Self::Aggressive,
            Self::Conservative,
            Self::Glucose,
            Self::MiniSat,
            Self::CaDiCaL,
        ]
    }
}

impl std::fmt::Display for ConfigPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Default => "Default",
            Self::Industrial => "Industrial",
            Self::Random => "Random",
            Self::Cryptographic => "Cryptographic",
            Self::Hardware => "Hardware",
            Self::Aggressive => "Aggressive",
            Self::Conservative => "Conservative",
            Self::Glucose => "Glucose",
            Self::MiniSat => "MiniSAT",
            Self::CaDiCaL => "CaDiCaL",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_presets_available() {
        let presets = ConfigPreset::all_presets();
        assert_eq!(presets.len(), 10);
    }

    #[test]
    fn test_preset_configs() {
        // Test that all presets can be created
        for preset in ConfigPreset::all_presets() {
            let config = preset.config();
            assert!(config.var_decay > 0.0 && config.var_decay < 1.0);
            assert!(config.clause_decay > 0.0 && config.clause_decay < 1.0);
        }
    }

    #[test]
    fn test_industrial_config() {
        let config = ConfigPreset::Industrial.config();
        assert_eq!(config.restart_strategy, RestartStrategy::Glucose);
        assert!(config.use_lrb_branching);
        assert!(config.enable_inprocessing);
    }

    #[test]
    fn test_random_config() {
        let config = ConfigPreset::Random.config();
        assert_eq!(config.restart_strategy, RestartStrategy::Geometric);
        assert!(!config.use_lrb_branching);
        assert!(!config.enable_inprocessing);
    }

    #[test]
    fn test_aggressive_config() {
        let config = ConfigPreset::Aggressive.config();
        assert!(config.restart_interval < 50);
        assert!(config.clause_deletion_threshold < 10000);
    }

    #[test]
    fn test_conservative_config() {
        let config = ConfigPreset::Conservative.config();
        assert!(config.restart_interval > 200);
        assert!(config.clause_deletion_threshold > 20000);
    }

    #[test]
    fn test_preset_descriptions() {
        for preset in ConfigPreset::all_presets() {
            let desc = preset.description();
            assert!(!desc.is_empty());
            assert!(desc.len() > 10);
        }
    }

    #[test]
    fn test_preset_display() {
        assert_eq!(format!("{}", ConfigPreset::Default), "Default");
        assert_eq!(format!("{}", ConfigPreset::Industrial), "Industrial");
        assert_eq!(format!("{}", ConfigPreset::Glucose), "Glucose");
    }
}
