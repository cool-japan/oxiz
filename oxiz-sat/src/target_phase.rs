//! Target-Based Phase Selection
//!
//! This module implements target-based phase selection, an enhancement over
//! simple phase saving. The idea is to maintain both saved phases and target
//! phases, using heuristics to decide which to use.
//!
//! Target phases are set based on:
//! - Recently satisfied literals in conflicts
//! - Literals that appear frequently in learned clauses
//! - Literals that lead to successful propagations
//!
//! Reference: "Target-Oriented Phase Selection" and CryptoMiniSat

use crate::literal::{Lit, Var};

/// Phase selection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseMode {
    /// Use saved phase (last assigned value)
    Saved,
    /// Use target phase (heuristically determined)
    Target,
    /// Random phase
    Random,
}

/// Statistics for target phase selection
#[derive(Debug, Default, Clone)]
pub struct TargetPhaseStats {
    /// Number of times target phase was used
    pub target_used: u64,
    /// Number of times saved phase was used
    pub saved_used: u64,
    /// Number of times random phase was used
    pub random_used: u64,
    /// Number of target phase updates
    pub target_updates: u64,
}

/// Target-based phase selection manager
pub struct TargetPhaseSelector {
    /// Saved phases (last assigned value)
    saved_phase: Vec<bool>,
    /// Target phases (heuristically determined)
    target_phase: Vec<bool>,
    /// Whether to use target phase for each variable
    use_target: Vec<bool>,
    /// Decay factor for target phase confidence
    decay: f64,
    /// Confidence scores for target phases (higher = more confident)
    confidence: Vec<f64>,
    /// Statistics
    stats: TargetPhaseStats,
}

impl TargetPhaseSelector {
    /// Create a new target phase selector
    pub fn new(num_vars: usize, decay: f64) -> Self {
        Self {
            saved_phase: vec![false; num_vars],
            target_phase: vec![false; num_vars],
            use_target: vec![false; num_vars],
            decay,
            confidence: vec![0.0; num_vars],
            stats: TargetPhaseStats::default(),
        }
    }

    /// Resize for a new number of variables
    pub fn resize(&mut self, num_vars: usize) {
        self.saved_phase.resize(num_vars, false);
        self.target_phase.resize(num_vars, false);
        self.use_target.resize(num_vars, false);
        self.confidence.resize(num_vars, 0.0);
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &TargetPhaseStats {
        &self.stats
    }

    /// Save phase for a variable
    pub fn save_phase(&mut self, var: Var, phase: bool) {
        self.saved_phase[var.index()] = phase;
    }

    /// Set target phase for a variable with confidence boost
    pub fn set_target(&mut self, var: Var, phase: bool, confidence_boost: f64) {
        let idx = var.index();
        self.target_phase[idx] = phase;
        self.confidence[idx] += confidence_boost;

        // Enable target phase if confidence is high enough
        if self.confidence[idx] > 1.0 {
            self.use_target[idx] = true;
            self.stats.target_updates += 1;
        }
    }

    /// Update target phase based on a satisfied literal in conflict analysis
    pub fn on_conflict_literal(&mut self, lit: Lit) {
        // When a literal appears in conflict analysis and is satisfied,
        // we want to encourage that polarity
        self.set_target(lit.var(), lit.sign(), 0.5);
    }

    /// Update target phase based on a learned clause
    pub fn on_learned_clause(&mut self, clause: &[Lit]) {
        // Literals in short learned clauses are good candidates for target phases
        if clause.len() <= 5 {
            for &lit in clause {
                self.set_target(lit.var(), lit.sign(), 0.2);
            }
        }
    }

    /// Decay all confidence scores
    pub fn decay_confidence(&mut self) {
        for conf in &mut self.confidence {
            *conf *= self.decay;
            // Reset use_target if confidence drops too low
            if *conf < 0.5 {
                // Don't reset immediately, let it decay naturally
            }
        }
    }

    /// Get the phase for a variable
    pub fn get_phase(&mut self, var: Var, mode: PhaseMode) -> bool {
        let idx = var.index();

        match mode {
            PhaseMode::Saved => {
                self.stats.saved_used += 1;
                self.saved_phase[idx]
            }
            PhaseMode::Target => {
                if self.use_target[idx] && self.confidence[idx] > 0.5 {
                    self.stats.target_used += 1;
                    self.target_phase[idx]
                } else {
                    self.stats.saved_used += 1;
                    self.saved_phase[idx]
                }
            }
            PhaseMode::Random => {
                self.stats.random_used += 1;
                // Simple random using var index
                (idx & 1) == 0
            }
        }
    }

    /// Reset target phases (useful after restarts)
    pub fn reset_targets(&mut self) {
        for use_target in &mut self.use_target {
            *use_target = false;
        }
        for conf in &mut self.confidence {
            *conf = 0.0;
        }
    }

    /// Get confidence for a variable
    #[must_use]
    pub fn get_confidence(&self, var: Var) -> f64 {
        self.confidence[var.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_phase_creation() {
        let selector = TargetPhaseSelector::new(10, 0.95);
        assert_eq!(selector.saved_phase.len(), 10);
        assert_eq!(selector.target_phase.len(), 10);
        assert_eq!(selector.confidence.len(), 10);
    }

    #[test]
    fn test_save_phase() {
        let mut selector = TargetPhaseSelector::new(10, 0.95);
        let var = Var::new(0);

        selector.save_phase(var, true);
        assert!(selector.saved_phase[var.index()]);

        let phase = selector.get_phase(var, PhaseMode::Saved);
        assert!(phase);
    }

    #[test]
    fn test_target_phase() {
        let mut selector = TargetPhaseSelector::new(10, 0.95);
        let var = Var::new(0);

        // Set target with high confidence
        selector.set_target(var, true, 2.0);

        let phase = selector.get_phase(var, PhaseMode::Target);
        assert!(phase);
        assert!(selector.get_confidence(var) > 1.0);
    }

    #[test]
    fn test_confidence_decay() {
        let mut selector = TargetPhaseSelector::new(10, 0.5);
        let var = Var::new(0);

        selector.set_target(var, true, 2.0);
        let initial_conf = selector.get_confidence(var);

        selector.decay_confidence();
        let decayed_conf = selector.get_confidence(var);

        assert!(decayed_conf < initial_conf);
        assert!((decayed_conf - initial_conf * 0.5).abs() < 0.001);
    }

    #[test]
    fn test_on_conflict_literal() {
        let mut selector = TargetPhaseSelector::new(10, 0.95);
        let lit = Lit::pos(Var::new(0));

        selector.on_conflict_literal(lit);
        assert!(selector.get_confidence(lit.var()) > 0.0);
    }

    #[test]
    fn test_reset_targets() {
        let mut selector = TargetPhaseSelector::new(10, 0.95);
        let var = Var::new(0);

        selector.set_target(var, true, 2.0);
        assert!(selector.get_confidence(var) > 0.0);

        selector.reset_targets();
        assert_eq!(selector.get_confidence(var), 0.0);
    }

    #[test]
    fn test_stats() {
        let mut selector = TargetPhaseSelector::new(10, 0.95);
        let var = Var::new(0);

        selector.get_phase(var, PhaseMode::Saved);
        assert_eq!(selector.stats().saved_used, 1);

        selector.get_phase(var, PhaseMode::Random);
        assert_eq!(selector.stats().random_used, 1);
    }
}
