//! Activity management for variables and clauses
//!
//! This module provides efficient activity tracking and decay for both
//! variables and clauses. Activity scores are used for decision heuristics
//! and clause deletion policies.
//!
//! References:
//! - "Chaff: Engineering an Efficient SAT Solver" (VSIDS)
//! - "Glucose: Combining Fast Local Search and Heuristics"

use crate::clause::ClauseId;
use crate::literal::Var;

/// Statistics for activity management
#[derive(Debug, Clone, Default)]
pub struct ActivityStats {
    /// Total number of bumps performed
    pub total_bumps: u64,
    /// Total number of decays performed
    pub total_decays: u64,
    /// Number of rescales performed
    pub rescales: u64,
    /// Current activity increment
    pub current_increment: f64,
}

impl ActivityStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Activity Manager Statistics:");
        println!("  Total bumps: {}", self.total_bumps);
        println!("  Total decays: {}", self.total_decays);
        println!("  Rescales: {}", self.rescales);
        println!("  Current increment: {:.2e}", self.current_increment);
    }
}

/// Variable activity manager
///
/// Manages activity scores for variables using VSIDS-style decay.
#[derive(Debug)]
pub struct VariableActivityManager {
    /// Activity score for each variable
    activities: Vec<f64>,
    /// Activity increment (increased on each decay)
    increment: f64,
    /// Decay factor (< 1.0)
    decay: f64,
    /// Threshold for rescaling
    rescale_threshold: f64,
    /// Statistics
    stats: ActivityStats,
}

impl Default for VariableActivityManager {
    fn default() -> Self {
        Self::new()
    }
}

impl VariableActivityManager {
    /// Create a new variable activity manager
    ///
    /// Default decay factor: 0.95
    #[must_use]
    pub fn new() -> Self {
        Self::with_decay(0.95)
    }

    /// Create with custom decay factor
    ///
    /// Decay should be in range (0.0, 1.0):
    /// - Higher values (e.g., 0.99) decay more slowly
    /// - Lower values (e.g., 0.8) decay more quickly
    #[must_use]
    pub fn with_decay(decay: f64) -> Self {
        Self {
            activities: Vec::new(),
            increment: 1.0,
            decay: decay.clamp(0.0, 1.0),
            rescale_threshold: 1e100,
            stats: ActivityStats {
                current_increment: 1.0,
                ..Default::default()
            },
        }
    }

    /// Resize for new number of variables
    pub fn resize(&mut self, num_vars: usize) {
        self.activities.resize(num_vars, 0.0);
    }

    /// Bump (increase) activity of a variable
    pub fn bump(&mut self, var: Var) {
        let idx = var.index();

        // Ensure we have space
        if idx >= self.activities.len() {
            self.activities.resize(idx + 1, 0.0);
        }

        self.activities[idx] += self.increment;
        self.stats.total_bumps += 1;

        // Rescale if needed
        if self.activities[idx] > self.rescale_threshold {
            self.rescale();
        }
    }

    /// Decay all activities and increase increment
    pub fn decay(&mut self) {
        self.increment /= self.decay;
        self.stats.total_decays += 1;
        self.stats.current_increment = self.increment;

        // Rescale if increment gets too large
        if self.increment > self.rescale_threshold {
            self.rescale();
        }
    }

    /// Rescale all activities to prevent overflow
    fn rescale(&mut self) {
        const RESCALE_FACTOR: f64 = 1e-100;

        for activity in &mut self.activities {
            *activity *= RESCALE_FACTOR;
        }
        self.increment *= RESCALE_FACTOR;
        self.stats.rescales += 1;
        self.stats.current_increment = self.increment;
    }

    /// Get activity of a variable
    #[must_use]
    pub fn activity(&self, var: Var) -> f64 {
        let idx = var.index();
        if idx < self.activities.len() {
            self.activities[idx]
        } else {
            0.0
        }
    }

    /// Set activity of a variable
    pub fn set_activity(&mut self, var: Var, activity: f64) {
        let idx = var.index();
        if idx >= self.activities.len() {
            self.activities.resize(idx + 1, 0.0);
        }
        self.activities[idx] = activity;
    }

    /// Get all activities
    #[must_use]
    pub fn activities(&self) -> &[f64] {
        &self.activities
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &ActivityStats {
        &self.stats
    }

    /// Reset all activities
    pub fn reset(&mut self) {
        for activity in &mut self.activities {
            *activity = 0.0;
        }
        self.increment = 1.0;
        self.stats = ActivityStats {
            current_increment: 1.0,
            ..Default::default()
        };
    }
}

/// Clause activity manager
///
/// Manages activity scores for clauses, used in clause deletion policies.
#[derive(Debug)]
pub struct ClauseActivityManager {
    /// Activity score for each clause
    activities: Vec<f64>,
    /// Activity increment
    increment: f64,
    /// Decay factor
    decay: f64,
    /// Rescale threshold
    rescale_threshold: f64,
    /// Statistics
    stats: ActivityStats,
}

impl Default for ClauseActivityManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ClauseActivityManager {
    /// Create a new clause activity manager
    ///
    /// Default decay factor: 0.999 (slower than variable decay)
    #[must_use]
    pub fn new() -> Self {
        Self::with_decay(0.999)
    }

    /// Create with custom decay factor
    #[must_use]
    pub fn with_decay(decay: f64) -> Self {
        Self {
            activities: Vec::new(),
            increment: 1.0,
            decay: decay.clamp(0.0, 1.0),
            rescale_threshold: 1e20,
            stats: ActivityStats {
                current_increment: 1.0,
                ..Default::default()
            },
        }
    }

    /// Bump activity of a clause
    pub fn bump(&mut self, clause_id: ClauseId) {
        let idx = clause_id.0 as usize;

        // Ensure we have space
        if idx >= self.activities.len() {
            self.activities.resize(idx + 1, 0.0);
        }

        self.activities[idx] += self.increment;
        self.stats.total_bumps += 1;

        if self.activities[idx] > self.rescale_threshold {
            self.rescale();
        }
    }

    /// Decay all clause activities
    pub fn decay(&mut self) {
        self.increment /= self.decay;
        self.stats.total_decays += 1;
        self.stats.current_increment = self.increment;

        if self.increment > self.rescale_threshold {
            self.rescale();
        }
    }

    /// Rescale to prevent overflow
    fn rescale(&mut self) {
        const RESCALE_FACTOR: f64 = 1e-20;

        for activity in &mut self.activities {
            *activity *= RESCALE_FACTOR;
        }
        self.increment *= RESCALE_FACTOR;
        self.stats.rescales += 1;
        self.stats.current_increment = self.increment;
    }

    /// Get activity of a clause
    #[must_use]
    pub fn activity(&self, clause_id: ClauseId) -> f64 {
        let idx = clause_id.0 as usize;
        if idx < self.activities.len() {
            self.activities[idx]
        } else {
            0.0
        }
    }

    /// Set activity of a clause
    pub fn set_activity(&mut self, clause_id: ClauseId, activity: f64) {
        let idx = clause_id.0 as usize;
        if idx >= self.activities.len() {
            self.activities.resize(idx + 1, 0.0);
        }
        self.activities[idx] = activity;
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &ActivityStats {
        &self.stats
    }

    /// Reset all activities
    pub fn reset(&mut self) {
        self.activities.clear();
        self.increment = 1.0;
        self.stats = ActivityStats {
            current_increment: 1.0,
            ..Default::default()
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_activity_manager_creation() {
        let manager = VariableActivityManager::new();
        assert_eq!(manager.decay, 0.95);
        assert_eq!(manager.increment, 1.0);
    }

    #[test]
    fn test_bump_variable() {
        let mut manager = VariableActivityManager::new();
        manager.resize(10);

        manager.bump(Var::new(0));
        assert_eq!(manager.activity(Var::new(0)), 1.0);

        manager.bump(Var::new(0));
        assert_eq!(manager.activity(Var::new(0)), 2.0);
    }

    #[test]
    fn test_decay_variable() {
        let mut manager = VariableActivityManager::with_decay(0.5);
        manager.resize(10);

        manager.bump(Var::new(0));
        let initial_activity = manager.activity(Var::new(0));

        manager.decay();
        manager.bump(Var::new(1));

        // After decay, new bumps should have higher increment
        assert!(manager.activity(Var::new(1)) > initial_activity);
        assert_eq!(manager.stats().total_decays, 1);
    }

    #[test]
    fn test_rescale_variable() {
        let mut manager = VariableActivityManager::new();
        manager.resize(10);
        manager.rescale_threshold = 10.0; // Low threshold for testing

        // Bump until rescale
        for _ in 0..20 {
            manager.bump(Var::new(0));
        }

        // Should have rescaled
        assert!(manager.stats().rescales > 0);
        assert!(manager.activity(Var::new(0)) < 100.0);
    }

    #[test]
    fn test_set_activity() {
        let mut manager = VariableActivityManager::new();
        manager.resize(10);

        manager.set_activity(Var::new(0), 42.0);
        assert_eq!(manager.activity(Var::new(0)), 42.0);
    }

    #[test]
    fn test_reset_variable() {
        let mut manager = VariableActivityManager::new();
        manager.resize(10);

        manager.bump(Var::new(0));
        manager.decay();

        manager.reset();

        assert_eq!(manager.activity(Var::new(0)), 0.0);
        assert_eq!(manager.increment, 1.0);
        assert_eq!(manager.stats().total_bumps, 0);
    }

    #[test]
    fn test_clause_activity_manager_creation() {
        let manager = ClauseActivityManager::new();
        assert_eq!(manager.decay, 0.999);
    }

    #[test]
    fn test_bump_clause() {
        let mut manager = ClauseActivityManager::new();

        manager.bump(ClauseId(0));
        assert_eq!(manager.activity(ClauseId(0)), 1.0);

        manager.bump(ClauseId(0));
        assert_eq!(manager.activity(ClauseId(0)), 2.0);
    }

    #[test]
    fn test_decay_clause() {
        let mut manager = ClauseActivityManager::with_decay(0.5);

        manager.bump(ClauseId(0));
        manager.decay();
        manager.bump(ClauseId(1));

        assert!(manager.activity(ClauseId(1)) > manager.activity(ClauseId(0)));
    }

    #[test]
    fn test_clause_rescale() {
        let mut manager = ClauseActivityManager::new();
        manager.rescale_threshold = 5.0;

        for _ in 0..10 {
            manager.bump(ClauseId(0));
        }

        assert!(manager.stats().rescales > 0);
    }

    #[test]
    fn test_reset_clause() {
        let mut manager = ClauseActivityManager::new();

        manager.bump(ClauseId(0));
        manager.reset();

        assert_eq!(manager.activity(ClauseId(0)), 0.0);
        assert_eq!(manager.increment, 1.0);
    }
}
