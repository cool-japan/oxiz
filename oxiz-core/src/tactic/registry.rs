//! Tactic Registry for OxiZ.
//!
//! Provides a string-keyed factory map for all concrete tactic implementations
//! in `oxiz-core`. Any crate that can access `oxiz-core` can call
//! [`default_registry`] to obtain a fully populated registry and then call
//! [`TacticRegistry::create`] by name, without knowing the concrete types.
//!
//! # Design
//!
//! - [`TacticRegistry`] is a plain `HashMap`-backed struct — no
//!   `lazy_static`/`once_cell` global state.  Each call to
//!   [`default_registry`] produces an independent instance, which is cheap
//!   because the factories are function pointers wrapped in `Box<dyn Fn>`.
//! - Factories that require a `TermManager` (stateful tactics) are represented
//!   by their *stateless* Newtype wrappers, which implement
//!   [`crate::tactic::core::Tactic`] and internally create a temporary
//!   manager when `apply` is called.
//! - Tactics that have no zero-argument constructor (e.g. `DerTactic`,
//!   `MbpTactic`) are excluded and documented below.
//!
//! # Excluded tactics
//!
//! | Tactic | Reason |
//! |--------|--------|
//! | `DerTactic` / `StatelessDerTactic` | `StatelessDerTactic` does not implement `Tactic`; its `apply` requires a `&mut TermManager` argument. |
//! | `MbpTactic` | No `Tactic` impl; requires a `&mut TermManager` and `MbpEngine` at construction. |
//! | `ScriptableTactic` | Requires a Rhai script string at construction time. |
//! | `CondTactic` / `WhenTactic` / `FailIfTactic` | Require combinator sub-tactics at construction. |
//! | `TseitinCnfTactic` / `NnfTactic` (stateful) | Require `&mut TermManager`. |
//! | `SkolemizationTactic` / `QuantifierInstantiationTactic` / `UniversalEliminationTactic` | Require `&mut TermManager`. |
//! | `ArithBoundsTactic` | Registered as "arith-bounds" using `Default`. |
//! | `FactorTactic` | Registered as "factor" using `Default`. |
//! | `BvArray2UfTactic` | Registered as "bvarray2uf" using `Default`. |

use std::collections::HashMap;

use crate::error::Result;
use crate::tactic::core::{Goal, Tactic, TacticResult};

// ─── Type alias ──────────────────────────────────────────────────────────────

/// Type alias for a boxed tactic factory closure.
type TacticFactory = Box<dyn Fn() -> Box<dyn Tactic> + Send + Sync>;

// ─── Concrete tactic imports ─────────────────────────────────────────────────

// Top-level stateless tactics
use super::ackermann::StatelessAckermannizeTactic;
use super::aggressive_simplify::StatelessAggressiveSimplifyTactic;
use super::bitblast::StatelessBitBlastTactic;
use super::ctx_simplify::StatelessCtxSolverSimplifyTactic;
use super::eliminate::StatelessEliminateUnconstrainedTactic;
use super::pb2bv::StatelessPb2BvTactic;
use super::propagate::StatelessPropagateValuesTactic;
use super::simplify::StatelessSimplifyTactic;
use super::solve_eqs::{
    StatelessCnfTactic, StatelessFourierMotzkinTactic, StatelessNnfTactic, StatelessSolveEqsTactic,
};
use super::split::StatelessSplitTactic;

// Arith tactics
use super::arith::arith_bounds::{ArithBoundsConfig, ArithBoundsTactic};
use super::arith::factor::{FactorTactic, FactorTacticConfig};

// BV tactics
use super::bv::bvarray2uf::{BvArray2UfConfig, BvArray2UfTactic};

// Sub-module tactics with Tactic impl
use super::lia2card::StatelessLia2CardTactic;
use super::nla2bv::StatelessNla2BvTactic;

// ─── TacticRegistry ──────────────────────────────────────────────────────────

/// A registry mapping string names to zero-argument tactic constructor closures.
///
/// Call [`default_registry`] to obtain a pre-populated instance.
pub struct TacticRegistry {
    factories: HashMap<&'static str, TacticFactory>,
}

impl TacticRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
        }
    }

    /// Register a tactic factory under the given canonical name.
    ///
    /// Subsequent calls with the same name overwrite the previous registration.
    pub fn register<F>(&mut self, name: &'static str, factory: F)
    where
        F: Fn() -> Box<dyn Tactic> + Send + Sync + 'static,
    {
        self.factories
            .insert(name, Box::new(factory) as TacticFactory);
    }

    /// Create a fresh tactic instance by name.
    ///
    /// Returns `None` if `name` is not registered.
    #[must_use]
    pub fn create(&self, name: &str) -> Option<Box<dyn Tactic>> {
        self.factories.get(name).map(|f| f())
    }

    /// Returns a sorted list of all registered tactic names.
    #[must_use]
    pub fn names(&self) -> Vec<&'static str> {
        let mut v: Vec<_> = self.factories.keys().copied().collect();
        v.sort_unstable();
        v
    }

    /// Returns `true` if `name` is registered in this registry.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.factories.contains_key(name)
    }
}

impl Default for TacticRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── SkipTactic ──────────────────────────────────────────────────────────────

/// A no-op tactic that always returns `SubGoals` with the goal unchanged.
///
/// Used as the `"skip"` entry in the default registry.
#[derive(Debug, Clone, Default)]
struct SkipTactic;

impl Tactic for SkipTactic {
    fn name(&self) -> &str {
        "skip"
    }

    fn apply(&self, goal: &Goal) -> Result<TacticResult> {
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    fn description(&self) -> &str {
        "A no-op tactic that returns the goal unchanged"
    }
}

// ─── default_registry ────────────────────────────────────────────────────────

/// Build and return the default registry with all known zero-argument tactics
/// registered under their canonical names.
///
/// This is a free function (not a global singleton) so that callers always
/// get an independent, freshly constructed registry — useful for testing and
/// for avoiding `Send`/`Sync` complexity of globals.
#[must_use]
pub fn default_registry() -> TacticRegistry {
    let mut reg = TacticRegistry::new();

    // ── Core simplification tactics ──────────────────────────────────────────
    reg.register("simplify", || Box::new(StatelessSimplifyTactic));
    reg.register("propagate-values", || {
        Box::new(StatelessPropagateValuesTactic)
    });
    reg.register("ctx-solver-simplify", || {
        Box::new(StatelessCtxSolverSimplifyTactic)
    });
    reg.register("aggressive-simplify", || {
        Box::new(StatelessAggressiveSimplifyTactic)
    });

    // ── Bit-blasting and bitvector tactics ───────────────────────────────────
    reg.register("bit-blast", || Box::new(StatelessBitBlastTactic));
    reg.register("bvarray2uf", || {
        Box::new(BvArray2UfTactic::new(BvArray2UfConfig::default()))
    });

    // ── UF (uninterpreted functions) ─────────────────────────────────────────
    reg.register("ackermannize", || Box::new(StatelessAckermannizeTactic));

    // ── Variable elimination and equation solving ─────────────────────────────
    reg.register("elim-uncnstr", || {
        Box::new(StatelessEliminateUnconstrainedTactic)
    });
    reg.register("solve-eqs", || Box::new(StatelessSolveEqsTactic));

    // ── Normal forms and CNF ─────────────────────────────────────────────────
    reg.register("nnf", || Box::new(StatelessNnfTactic));
    reg.register("tseitin-cnf", || Box::new(StatelessCnfTactic));

    // ── Arithmetic tactics ───────────────────────────────────────────────────
    reg.register("fm", || Box::new(StatelessFourierMotzkinTactic));
    reg.register("arith-bounds", || {
        Box::new(ArithBoundsTactic::new(ArithBoundsConfig::default()))
    });
    reg.register("factor", || {
        Box::new(FactorTactic::new(FactorTacticConfig::default()))
    });

    // ── Pseudo-boolean and cardinality ───────────────────────────────────────
    reg.register("pb2bv", || Box::new(StatelessPb2BvTactic));
    reg.register("lia2card", || Box::new(StatelessLia2CardTactic::new()));

    // ── Non-linear arithmetic ────────────────────────────────────────────────
    reg.register("nla2bv", || Box::new(StatelessNla2BvTactic::new()));

    // ── Goal splitting ───────────────────────────────────────────────────────
    reg.register("split", || Box::new(StatelessSplitTactic));

    // ── Utility ──────────────────────────────────────────────────────────────
    reg.register("skip", || Box::new(SkipTactic));

    reg
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_contains_simplify() {
        let reg = default_registry();
        assert!(reg.contains("simplify"));
    }

    #[test]
    fn test_registry_create_simplify_returns_some() {
        let reg = default_registry();
        assert!(reg.create("simplify").is_some());
    }

    #[test]
    fn test_registry_create_unknown_returns_none() {
        let reg = default_registry();
        assert!(reg.create("not-a-real-tactic").is_none());
    }

    #[test]
    fn test_registry_names_sorted() {
        let reg = default_registry();
        let names = reg.names();
        assert!(!names.is_empty());
        assert!(names.contains(&"simplify"));
        // names should be sorted
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_registry_create_produces_independent_instances() {
        let reg = default_registry();
        let _t1 = reg.create("simplify").unwrap();
        let _t2 = reg.create("simplify").unwrap();
        // Two independent instances — just verifying both are Some
    }

    #[test]
    fn test_registry_contains_all_core_tactics() {
        let reg = default_registry();
        let expected = [
            "simplify",
            "propagate-values",
            "bit-blast",
            "ackermannize",
            "ctx-solver-simplify",
            "elim-uncnstr",
            "pb2bv",
            "solve-eqs",
            "fm",
            "nnf",
            "tseitin-cnf",
            "split",
            "aggressive-simplify",
            "lia2card",
            "nla2bv",
            "arith-bounds",
            "factor",
            "bvarray2uf",
            "skip",
        ];
        for name in expected {
            assert!(
                reg.contains(name),
                "default_registry missing tactic: {}",
                name
            );
        }
    }

    #[test]
    fn test_registry_create_skip_returns_subgoals() {
        let reg = default_registry();
        let tactic = reg.create("skip").unwrap();
        let goal = crate::tactic::core::Goal::empty();
        let result = tactic.apply(&goal).expect("skip should not fail");
        assert!(matches!(result, TacticResult::SubGoals(_)));
    }

    #[test]
    fn test_registry_names_count() {
        let reg = default_registry();
        // We register 19 tactics; ensure count is at least 19
        assert!(reg.names().len() >= 19);
    }

    #[test]
    fn test_registry_tactic_names_match_canonical() {
        let reg = default_registry();
        let names = reg.names();
        for name in &names {
            let tactic = reg.create(name).unwrap();
            assert_eq!(
                tactic.name(),
                *name,
                "tactic.name() '{}' does not match registry key '{}'",
                tactic.name(),
                name
            );
        }
    }

    #[test]
    fn test_registry_default_is_empty() {
        let reg = TacticRegistry::default();
        assert!(reg.names().is_empty());
        assert!(!reg.contains("simplify"));
    }

    #[test]
    fn test_registry_register_and_create() {
        let mut reg = TacticRegistry::new();
        reg.register("skip", || Box::new(SkipTactic));
        assert!(reg.contains("skip"));
        let tactic = reg.create("skip").unwrap();
        assert_eq!(tactic.name(), "skip");
    }
}
