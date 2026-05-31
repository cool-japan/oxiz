//! Adapter exposing MLEnhancedVSIDS as a BranchingHeuristic for oxiz-sat.

use std::sync::{Arc, Mutex};

use oxiz_sat::{BranchingHeuristic, Var};

use super::{MLEnhancedVSIDS, VarId};

/// Wraps [`MLEnhancedVSIDS`] so it can be plugged into `SolverConfig::external_branching`.
///
/// Type bridge: `Var(u32) ↔ VarId(usize)` via `Var::index()` / `Var::new()`.
pub struct MLBranchingHeuristic {
    inner: MLEnhancedVSIDS,
    min_confidence: f64,
}

impl MLBranchingHeuristic {
    /// Wrap an existing heuristic. `min_confidence` defaults to 0.0 (always delegate).
    pub fn new(inner: MLEnhancedVSIDS) -> Self {
        Self {
            inner,
            min_confidence: 0.0,
        }
    }

    /// Gate: if the ML decision has confidence below this, return `None` to defer to VSIDS.
    pub fn with_min_confidence(mut self, threshold: f64) -> Self {
        self.min_confidence = threshold.clamp(0.0, 1.0);
        self
    }

    /// Wrap in `Arc<Mutex<...>>` ready to slot into `SolverConfig::external_branching`.
    pub fn boxed(self) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(self))
    }

    /// Access the inner heuristic.
    pub fn inner(&self) -> &MLEnhancedVSIDS {
        &self.inner
    }

    /// Mutably access the inner heuristic.
    pub fn inner_mut(&mut self) -> &mut MLEnhancedVSIDS {
        &mut self.inner
    }
}

impl BranchingHeuristic for MLBranchingHeuristic {
    fn select(&mut self, candidates: &[Var], _scores: &[f64]) -> Option<Var> {
        if candidates.is_empty() {
            return None;
        }
        let var_ids: Vec<VarId> = candidates.iter().map(|v| v.index()).collect();
        let decision = self.inner.select_variable(&var_ids)?;
        if decision.confidence < self.min_confidence {
            return None;
        }
        Some(Var::new(decision.variable as u32))
    }

    fn on_conflict_var(&mut self, var: Var, level: u32) {
        self.inner.update_conflict(var.index(), level as f64);
    }

    /// Override to pass the real LBD score (not decision level) to the ML model.
    ///
    /// LBD is the gold-standard learned-clause quality metric: lower is better.
    /// Passing it as a `f64` to `update_conflict` gives the ML learner a more
    /// informative signal than the raw decision level that the default delegation
    /// would provide.
    fn on_conflict_var_with_lbd(&mut self, var: Var, lbd: u32) {
        self.inner.update_conflict(var.index(), lbd as f64);
    }
}
