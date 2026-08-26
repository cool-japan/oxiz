//! Solver Configuration Presets
//!
//! Pre-configured solver profiles optimized for different problem classes.
//! These presets are based on extensive empirical testing and competition
//! results from modern SAT solvers.

#[allow(unused_imports)]
use crate::prelude::*;
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
        SolverConfig {
            // Pinned rather than inherited: the preset docs (and
            // `Solver::check_hyper_binary_resolution`'s) state which presets
            // run lazy hyper-binary resolution, and that statement must not
            // silently change when the struct default is retuned. This one
            // agrees with today's `SolverConfig::default()` on purpose — see
            // the measurement recorded at
            // `SolverConfig::enable_lazy_hyper_binary`.
            enable_lazy_hyper_binary: true,
            ..SolverConfig::default()
        }
    }

    /// Industrial/structured problems configuration
    ///
    /// Characteristics:
    /// - Heavy use of clause minimization
    /// - Glucose-style restarts
    /// - Aggressive inprocessing
    /// - LRB branching heuristic
    /// - Bounded variable elimination (see the `enable_bve` comment below)
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE **on**. Industrial/structured instances are exactly the
            // class SatELite-style variable elimination was designed for:
            // encodings of circuits, schedules and product configurations
            // carry large numbers of low-degree definitional variables whose
            // clauses fold away without growing the formula, and this preset
            // already commits to the base-assertion-level, non-incremental
            // shape BVE requires (aggressive inprocessing, no proof tracing
            // assumed). `Solver::bounded_variable_elimination` still declines
            // on its own whenever that shape does not hold — an active
            // incremental `push`, DRAT/LRAT tracing, or a non-zero decision
            // level — so turning it on here costs nothing in the
            // configurations where it would be unsound. Equivalent-literal
            // substitution stays off because BVE defers entirely to it when
            // both are set (see `SolverConfig::enable_bve`), which would make
            // this flip inert.
            enable_bve: true,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On in every preset, matching `SolverConfig::default()`: the
            // lucky phase runs entirely in scratch buffers, never touches the
            // clause database or the trail, and verifies every candidate
            // against the original clauses before reporting it — so it cannot
            // change a verdict in any configuration, and the only cost on the
            // instances it fails to solve is a few linear sweeps. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE / equivalent-literal substitution / gate congruence all
            // remove variables from the live formula and are therefore only
            // sound at the base assertion level with no incremental `push`
            // in scope (see the doc comments on `SolverConfig`). Left off in
            // this preset; the `Industrial` and `CaDiCaL` presets, whose
            // problem classes and imitated tool both call for it, turn
            // `enable_bve` on (see their bodies for the reasoning).
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On here too, despite this preset's problem class being the
            // one least likely to hit: a uniform-random instance is
            // essentially never satisfied by a structural guess like
            // "all variables true". Left on anyway because the cost is a
            // couple of linear sweeps against a search that will run for
            // millions of propagations, and because "random" is a preset
            // *name*, not a guarantee about the instance — a caller that
            // picks it for a generated formula which happens to be
            // all-positive or Horn gets the answer for free. Turn it off
            // explicitly when measuring the search itself. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE / equivalent-literal substitution / gate congruence all
            // remove variables from the live formula and are therefore only
            // sound at the base assertion level with no incremental `push`
            // in scope (see the doc comments on `SolverConfig`). Left off in
            // this preset; the `Industrial` and `CaDiCaL` presets, whose
            // problem classes and imitated tool both call for it, turn
            // `enable_bve` on (see their bodies for the reasoning).
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On in every preset, matching `SolverConfig::default()`: the
            // lucky phase runs entirely in scratch buffers, never touches the
            // clause database or the trail, and verifies every candidate
            // against the original clauses before reporting it — so it cannot
            // change a verdict in any configuration, and the only cost on the
            // instances it fails to solve is a few linear sweeps. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE / equivalent-literal substitution / gate congruence all
            // remove variables from the live formula and are therefore only
            // sound at the base assertion level with no incremental `push`
            // in scope (see the doc comments on `SolverConfig`). Left off in
            // this preset; the `Industrial` and `CaDiCaL` presets, whose
            // problem classes and imitated tool both call for it, turn
            // `enable_bve` on (see their bodies for the reasoning).
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On in every preset, matching `SolverConfig::default()`: the
            // lucky phase runs entirely in scratch buffers, never touches the
            // clause database or the trail, and verifies every candidate
            // against the original clauses before reporting it — so it cannot
            // change a verdict in any configuration, and the only cost on the
            // instances it fails to solve is a few linear sweeps. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE / equivalent-literal substitution / gate congruence all
            // remove variables from the live formula and are therefore only
            // sound at the base assertion level with no incremental `push`
            // in scope (see the doc comments on `SolverConfig`). Left off in
            // this preset; the `Industrial` and `CaDiCaL` presets, whose
            // problem classes and imitated tool both call for it, turn
            // `enable_bve` on (see their bodies for the reasoning).
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On in every preset, matching `SolverConfig::default()`: the
            // lucky phase runs entirely in scratch buffers, never touches the
            // clause database or the trail, and verifies every candidate
            // against the original clauses before reporting it — so it cannot
            // change a verdict in any configuration, and the only cost on the
            // instances it fails to solve is a few linear sweeps. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE / equivalent-literal substitution / gate congruence all
            // remove variables from the live formula and are therefore only
            // sound at the base assertion level with no incremental `push`
            // in scope (see the doc comments on `SolverConfig`). Left off in
            // this preset; the `Industrial` and `CaDiCaL` presets, whose
            // problem classes and imitated tool both call for it, turn
            // `enable_bve` on (see their bodies for the reasoning).
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On in every preset, matching `SolverConfig::default()`: the
            // lucky phase runs entirely in scratch buffers, never touches the
            // clause database or the trail, and verifies every candidate
            // against the original clauses before reporting it — so it cannot
            // change a verdict in any configuration, and the only cost on the
            // instances it fails to solve is a few linear sweeps. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE / equivalent-literal substitution / gate congruence all
            // remove variables from the live formula and are therefore only
            // sound at the base assertion level with no incremental `push`
            // in scope (see the doc comments on `SolverConfig`). Left off in
            // this preset; the `Industrial` and `CaDiCaL` presets, whose
            // problem classes and imitated tool both call for it, turn
            // `enable_bve` on (see their bodies for the reasoning).
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On in every preset, matching `SolverConfig::default()`: the
            // lucky phase runs entirely in scratch buffers, never touches the
            // clause database or the trail, and verifies every candidate
            // against the original clauses before reporting it — so it cannot
            // change a verdict in any configuration, and the only cost on the
            // instances it fails to solve is a few linear sweeps. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE / equivalent-literal substitution / gate congruence all
            // remove variables from the live formula and are therefore only
            // sound at the base assertion level with no incremental `push`
            // in scope (see the doc comments on `SolverConfig`). Left off in
            // this preset; the `Industrial` and `CaDiCaL` presets, whose
            // problem classes and imitated tool both call for it, turn
            // `enable_bve` on (see their bodies for the reasoning).
            enable_bve: false,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On in every preset, matching `SolverConfig::default()`: the
            // lucky phase runs entirely in scratch buffers, never touches the
            // clause database or the trail, and verifies every candidate
            // against the original clauses before reporting it — so it cannot
            // change a verdict in any configuration, and the only cost on the
            // instances it fails to solve is a few linear sweeps. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
        }
    }

    /// CaDiCaL-style configuration
    ///
    /// Based on CaDiCaL SAT solver parameters, including its default-on
    /// bounded variable elimination (see the `enable_bve` comment below).
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
            use_vmtf: true,
            luby_cap: 64,
            enable_stabilize: true,
            stabilize_base: 5000,
            focused_luby_cap: 16,
            rephase_interval: 50,
            reuse_trail: true,
            external_branching: None,
            // Off by default, matching `SolverConfig::default()`: sound on
            // its own, but a probing-only pass can fully settle a
            // small/dense instance before the main CDCL loop ever runs,
            // changing observable solve behavior even though the verdict
            // itself never changes. Opt-in until a caller wants that shape.
            enable_failed_literal_probing: false,
            // BVE **on**, because the tool this preset imitates runs it: in
            // CaDiCaL, `elim` (bounded variable elimination) is enabled by
            // default and is one of the main reasons its preprocessing
            // reduces industrial formulas as much as it does. A "CaDiCaL-style"
            // preset with variable elimination switched off would not be
            // CaDiCaL-style at all. `Solver::bounded_variable_elimination`
            // keeps its own guards (base assertion level only, never while a
            // proof is traced), so this only takes effect in the
            // configurations where it is sound. Equivalent-literal
            // substitution stays off because BVE defers entirely to it when
            // both are set (see `SolverConfig::enable_bve`), which would make
            // this flip inert.
            enable_bve: true,
            enable_equiv_substitution: false,
            enable_gate_congruence: false,
            // On in every preset: self-subsuming resolution neither removes a
            // variable nor needs model reconstruction, so it carries none of
            // the three fields above's base-assertion-level restriction. It
            // still only runs when `enable_inprocessing` is set, so the
            // presets that turn inprocessing off get it for free as a no-op.
            enable_self_subsumption: true,
            // On, and here it is not merely inherited from the default: the
            // lucky phase is a port of CaDiCaL's own `lucky.cpp`, which that
            // solver runs by name before search. The preset that imitates
            // CaDiCaL is the last place it should be switched off. See
            // `SolverConfig::enable_lucky_phase`.
            enable_lucky_phase: true,
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

impl core::fmt::Display for ConfigPreset {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
