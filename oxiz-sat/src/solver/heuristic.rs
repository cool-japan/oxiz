//! Extension trait for plugging in custom branching heuristics.
//!
//! The `BranchingHeuristic` trait lets external crates (e.g. `oxiz-ml`) inject
//! custom variable-selection logic into the CDCL solver without requiring a
//! circular crate dependency. The solver calls `select` when it has a candidate
//! list; the heuristic may return `None` to fall back to the built-in strategy.

use std::sync::{Arc, Mutex};

use crate::literal::Var;

/// A pluggable branching heuristic for the CDCL solver.
///
/// Implementors receive an ordered slice of unassigned variables and
/// their current VSIDS scores, and return the chosen variable or `None`
/// to defer to the built-in VSIDS/LRB/CHB strategy.
///
/// # Thread Safety
///
/// The trait requires `Send + Sync` so the heuristic can be wrapped in
/// `Arc<Mutex<dyn BranchingHeuristic>>` for integration into the solver config.
pub trait BranchingHeuristic: Send + Sync {
    /// Select a variable to branch on from `candidates`.
    ///
    /// `scores` is parallel to `candidates` and contains current VSIDS activity
    /// values (higher = more active). Return `None` to fall back to built-in strategy.
    fn select(&mut self, candidates: &[Var], scores: &[f64]) -> Option<Var>;

    /// Called during conflict analysis for each variable that contributed to the conflict.
    ///
    /// `var` is the variable involved and `level` is the decision level at which it was
    /// assigned. The default implementation is a no-op, preserving backward compatibility
    /// for all existing implementations.
    fn on_conflict_var(&mut self, _var: Var, _level: u32) {}
}

/// A heap-allocated, type-erased branching heuristic.
///
/// This is `Arc<Mutex<dyn BranchingHeuristic>>`, which is both `Clone` and
/// `Send + Sync`. Storing it in `SolverConfig` (which derives `Clone`) works
/// because `Arc` is `Clone` and `Mutex<dyn Trait>` is `Send + Sync` when
/// `dyn Trait: Send + Sync`.
pub type BoxedBranchingHeuristic = Arc<Mutex<dyn BranchingHeuristic>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal heuristic that does NOT override `on_conflict_var`.
    /// This verifies the default no-op implementation compiles and runs cleanly.
    struct MinimalHeuristic;

    impl BranchingHeuristic for MinimalHeuristic {
        fn select(&mut self, candidates: &[Var], _scores: &[f64]) -> Option<Var> {
            candidates.first().copied()
        }
        // on_conflict_var intentionally omitted — exercises the default impl
    }

    #[test]
    fn test_default_on_conflict_var_is_noop() {
        let mut h = MinimalHeuristic;
        // Calling the default on_conflict_var must compile and be a no-op.
        h.on_conflict_var(Var::new(0), 1);
        h.on_conflict_var(Var::new(7), 0);
        // select still works correctly
        let candidates = [Var::new(3), Var::new(5)];
        let chosen = h.select(&candidates, &[0.0, 0.0]);
        assert_eq!(chosen, Some(Var::new(3)));
    }
}
