//! Feature Registry — compile-time feature gating for WASM bundle size control.
//!
//! # Bundle size strategy
//!
//! The `minimal` Cargo feature is meant to produce a smaller WASM bundle by
//! excluding subsystems that are large but rarely needed in browser contexts.
//! Exactly one of the six delivers that today, and it is the only one with a
//! measurement behind it:
//!
//! | Excluded subsystem                  | Saving                       | Basis         |
//! |-------------------------------------|------------------------------|---------------|
//! | Nonlinear arithmetic (`oxiz-nlsat`) | **194,260 raw / 68,268 gzip**| measured      |
//! | Proof generation (DAG)              | 0                            | gates nothing |
//! | Craig interpolation                 | 0                            | gates nothing |
//! | Spacer PDR engine                   | 0                            | gates nothing |
//! | ML branching heuristics             | 0                            | gates nothing |
//! | WASM bench harness                  | 0                            | gates nothing |
//!
//! **The estimates this table used to carry — ~180 KB proof, ~90 KB
//! interpolation, ~220 KB Spacer, ~130 KB ML, ~25 KB bench, "~645 KB total",
//! and the manifest's "~35–45 % reduction" — were never true.** `proof`,
//! `interpolation`, `spacer`, `ml_branching` and `wasm_bench` are empty feature
//! lists: they pull in no dependency and gate no code. Choosing `minimal` over
//! `full` therefore could not remove anything before 0.3.3, because there was
//! nothing gated to remove — which is checkable in one command:
//!
//! ```text
//! grep -rn 'cfg(feature\|cfg!(feature' oxiz-wasm/src/
//! ```
//!
//! returns the accessors below and nothing else.
//!
//! `nlsat` (new in 0.3.3) is different in kind: it is a real dependency edge,
//! `oxiz-solver/nlsat` → `oxiz-theories/nlsat` → the `oxiz-nlsat` crate, so
//! dropping it removes compiled code. The figures above were measured on a
//! comparably-shaped browser shim — same `[profile.release]` (`opt-level =
//! "z"`, fat LTO, `codegen-units = 1`, `panic = "abort"`, stripped) and the same
//! `wasm-opt -Oz --converge` pass — at 1,557,287 → 1,363,027 bytes raw and
//! 573,181 → 504,913 after `gzip -9`. Expect a different absolute size here,
//! since this crate's reachable set is not that one's; expect a delta of the
//! same order.
//!
//! What dropping `nlsat` costs is completeness on nonlinear *real* arithmetic:
//! a QF_NRA goal that needs the cell-decomposition core answers `unknown`
//! instead of `sat`/`unsat`. QF_NIA is unaffected — its verdicts come from
//! procedures that were never in `oxiz-nlsat` — and no verdict ever becomes
//! wrong.
//!
//! Always included regardless of feature flags:
//! - Core DPLL(T) solver and SAT engine
//! - The theory solvers (EUF, LIA, LRA, BV, Arrays, Strings, FP, Datatypes,
//!   Sets, Sequences, …). NIA/NRA need `nlsat` to be *complete*; their linear
//!   fragments are decided either way
//! - SMT-LIB2 parser and pretty-printer
//! - Model generation and unsat-core extraction
//! - Incremental solving (push/pop)
//!
//! # Using features
//!
//! ```toml
//! # Cargo.toml — size-optimized WASM build
//! [dependencies]
//! oxiz-wasm = { version = "0.2.2", default-features = false, features = ["minimal"] }
//! ```
//!
//! Or via `wasm-pack`:
//! ```bash
//! wasm-pack build oxiz-wasm --target web --release -- --no-default-features --features minimal
//! ```

#![forbid(unsafe_code)]

/// Describes which optional subsystems are compiled into this binary.
///
/// Populated entirely from Cargo feature flags at compile time — zero runtime
/// overhead (all values are `const`).
pub struct FeatureRegistry;

impl FeatureRegistry {
    /// Returns `true` when proof generation is compiled in.
    ///
    /// Controlled by the `proof` Cargo feature.  Excluded in `minimal` builds.
    pub const fn has_proof() -> bool {
        cfg!(feature = "proof")
    }

    /// Returns `true` when Craig interpolation is compiled in.
    ///
    /// Controlled by the `interpolation` Cargo feature.  Excluded in `minimal` builds.
    pub const fn has_interpolation() -> bool {
        cfg!(feature = "interpolation")
    }

    /// Returns `true` when the Spacer PDR engine is compiled in.
    ///
    /// Controlled by the `spacer` Cargo feature.  Excluded in `minimal` builds.
    pub const fn has_spacer() -> bool {
        cfg!(feature = "spacer")
    }

    /// Returns `true` when ML-guided branching heuristics are compiled in.
    ///
    /// Controlled by the `ml_branching` Cargo feature.  Excluded in `minimal` builds.
    pub const fn has_ml_branching() -> bool {
        cfg!(feature = "ml_branching")
    }

    /// Returns `true` when the WASM benchmark harness is compiled in.
    ///
    /// Controlled by the `wasm_bench` Cargo feature.  Excluded in `minimal` builds.
    pub const fn has_wasm_bench() -> bool {
        cfg!(feature = "wasm_bench")
    }

    /// Returns `true` when the nonlinear-arithmetic solver is compiled in.
    ///
    /// Controlled by the `nlsat` Cargo feature.  Excluded in `minimal` builds.
    ///
    /// Unlike its five neighbours above, this one answers a question about the
    /// binary rather than about a flag: `false` means the `oxiz-nlsat` crate is
    /// not in the dependency graph, and a QF_NRA goal that needs the
    /// cell-decomposition core will answer `unknown`.  See the module docs.
    pub const fn has_nlsat() -> bool {
        cfg!(feature = "nlsat")
    }

    /// Returns `true` when this is a `full` build (all optional features enabled).
    pub const fn is_full() -> bool {
        cfg!(feature = "full")
    }

    /// Returns `true` when this is a `minimal` build.
    ///
    /// A `minimal` build excludes proof, interpolation, spacer, ml_branching,
    /// wasm_bench and nlsat.  Only the last of those currently removes any
    /// code; see the module docs.
    pub const fn is_minimal() -> bool {
        !Self::has_proof()
            && !Self::has_interpolation()
            && !Self::has_spacer()
            && !Self::has_ml_branching()
            && !Self::has_wasm_bench()
            && !Self::has_nlsat()
    }

    /// Human-readable list of enabled optional features.
    pub fn enabled_features() -> Vec<&'static str> {
        let mut features = Vec::new();
        if Self::has_proof() {
            features.push("proof");
        }
        if Self::has_interpolation() {
            features.push("interpolation");
        }
        if Self::has_spacer() {
            features.push("spacer");
        }
        if Self::has_ml_branching() {
            features.push("ml_branching");
        }
        if Self::has_wasm_bench() {
            features.push("wasm_bench");
        }
        if Self::has_nlsat() {
            features.push("nlsat");
        }
        features
    }

    /// Human-readable list of disabled optional features.
    pub fn disabled_features() -> Vec<&'static str> {
        let mut features = Vec::new();
        if !Self::has_proof() {
            features.push("proof");
        }
        if !Self::has_interpolation() {
            features.push("interpolation");
        }
        if !Self::has_spacer() {
            features.push("spacer");
        }
        if !Self::has_ml_branching() {
            features.push("ml_branching");
        }
        if !Self::has_wasm_bench() {
            features.push("wasm_bench");
        }
        if !Self::has_nlsat() {
            features.push("nlsat");
        }
        features
    }

    /// Returns a short build-profile string: `"full"`, `"minimal"`, or `"custom"`.
    pub fn build_profile() -> &'static str {
        if Self::is_full() {
            "full"
        } else if Self::is_minimal() {
            "minimal"
        } else {
            "custom"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_profile_is_string() {
        let profile = FeatureRegistry::build_profile();
        assert!(
            profile == "full" || profile == "minimal" || profile == "custom",
            "unexpected build profile: {profile}"
        );
    }

    #[test]
    fn test_enabled_and_disabled_are_disjoint() {
        let enabled: std::collections::HashSet<&str> =
            FeatureRegistry::enabled_features().into_iter().collect();
        let disabled: std::collections::HashSet<&str> =
            FeatureRegistry::disabled_features().into_iter().collect();
        assert!(
            enabled.is_disjoint(&disabled),
            "enabled and disabled feature sets must be disjoint"
        );
    }

    #[test]
    fn test_enabled_plus_disabled_covers_all() {
        let all_optional = [
            "proof",
            "interpolation",
            "spacer",
            "ml_branching",
            "wasm_bench",
            "nlsat",
        ];
        let enabled: std::collections::HashSet<&str> =
            FeatureRegistry::enabled_features().into_iter().collect();
        let disabled: std::collections::HashSet<&str> =
            FeatureRegistry::disabled_features().into_iter().collect();
        for feat in all_optional {
            assert!(
                enabled.contains(feat) || disabled.contains(feat),
                "feature '{feat}' not covered by enabled/disabled lists"
            );
        }
    }

    #[test]
    fn test_const_methods_are_deterministic() {
        // Call twice; results must be identical (they are const)
        assert_eq!(FeatureRegistry::has_proof(), FeatureRegistry::has_proof());
        assert_eq!(FeatureRegistry::has_spacer(), FeatureRegistry::has_spacer());
    }
}
