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

    /// Called during conflict analysis with the LBD of the learned clause.
    ///
    /// `var` is the variable involved and `lbd` is the Literal Block Distance (glue score)
    /// of the learned clause — i.e., the number of distinct decision levels among the
    /// conflict-involved variables (level-0 vars excluded).
    ///
    /// LBD is the gold-standard quality metric for learned clauses in Glucose/MiniSat-style
    /// CDCL solvers: LBD 1 = unit prop at level 0 (keep forever), LBD 2 = "glue" clause
    /// (very valuable), LBD > 6 = likely weak.
    ///
    /// The default implementation delegates to `on_conflict_var(var, lbd)` for backward
    /// compatibility with heuristics that only implement `on_conflict_var`.
    fn on_conflict_var_with_lbd(&mut self, var: Var, lbd: u32) {
        self.on_conflict_var(var, lbd);
    }
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
        // on_conflict_var_with_lbd intentionally omitted — exercises the default delegation impl
    }

    /// A heuristic that records what values `on_conflict_var` was called with.
    struct RecordingHeuristic {
        recorded_levels: Vec<u32>,
    }

    impl BranchingHeuristic for RecordingHeuristic {
        fn select(&mut self, candidates: &[Var], _scores: &[f64]) -> Option<Var> {
            candidates.first().copied()
        }

        fn on_conflict_var(&mut self, _var: Var, level: u32) {
            self.recorded_levels.push(level);
        }
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

    #[test]
    fn test_default_on_conflict_var_with_lbd_delegates_to_on_conflict_var() {
        // When only on_conflict_var is overridden, on_conflict_var_with_lbd should
        // delegate to on_conflict_var, passing lbd as the level argument.
        let mut h = RecordingHeuristic {
            recorded_levels: Vec::new(),
        };
        // Call on_conflict_var_with_lbd — must delegate to on_conflict_var(var, lbd).
        h.on_conflict_var_with_lbd(Var::new(1), 3);
        h.on_conflict_var_with_lbd(Var::new(2), 5);
        assert_eq!(
            h.recorded_levels,
            vec![3, 5],
            "default on_conflict_var_with_lbd must forward lbd to on_conflict_var"
        );
    }
}
