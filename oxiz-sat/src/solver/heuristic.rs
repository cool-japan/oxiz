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
}

/// A heap-allocated, type-erased branching heuristic.
///
/// This is `Arc<Mutex<dyn BranchingHeuristic>>`, which is both `Clone` and
/// `Send + Sync`. Storing it in `SolverConfig` (which derives `Clone`) works
/// because `Arc` is `Clone` and `Mutex<dyn Trait>` is `Send + Sync` when
/// `dyn Trait: Send + Sync`.
pub type BoxedBranchingHeuristic = Arc<Mutex<dyn BranchingHeuristic>>;
