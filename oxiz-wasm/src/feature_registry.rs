//! Feature Registry — compile-time feature gating for WASM bundle size control.
//!
//! # Bundle size strategy
//!
//! The `minimal` Cargo feature produces a significantly smaller WASM bundle by
//! excluding subsystems that are large but rarely needed in browser contexts:
//!
//! | Excluded subsystem          | Approx. saving |
//! |-----------------------------|---------------|
//! | Proof generation (DAG)      | ~180 KB        |
//! | Craig interpolation         | ~90 KB         |
//! | Spacer PDR engine           | ~220 KB        |
//! | ML branching heuristics     | ~130 KB        |
//! | WASM bench harness          | ~25 KB         |
//! | **Total (estimated)**       | **~645 KB**    |
//!
//! Always included regardless of feature flags:
//! - Core DPLL(T) solver and SAT engine
//! - All 19 theory solvers (EUF, LIA, LRA, NIA, NRA, BV, Arrays, Strings, FP,
//!   Datatypes, Sets, Sequences, …)
//! - SMT-LIB2 parser and pretty-printer
//! - Model generation and unsat-core extraction
//! - Incremental solving (push/pop)
//!
//! # Using features
//!
//! ```toml
//! # Cargo.toml — size-optimized WASM build
//! [dependencies]
//! oxiz-wasm = { version = "0.2.0", default-features = false, features = ["minimal"] }
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

    /// Returns `true` when this is a `full` build (all optional features enabled).
    pub const fn is_full() -> bool {
        cfg!(feature = "full")
    }

    /// Returns `true` when this is a `minimal` build.
    ///
    /// A `minimal` build excludes proof, interpolation, spacer, ml_branching,
    /// and wasm_bench to reduce the WASM bundle size toward the <2 MB target.
    pub const fn is_minimal() -> bool {
        !Self::has_proof()
            && !Self::has_interpolation()
            && !Self::has_spacer()
            && !Self::has_ml_branching()
            && !Self::has_wasm_bench()
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
