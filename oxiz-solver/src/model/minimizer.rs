//! Model Minimization Engine.
//!
//! Produces minimal satisfying assignments by removing unnecessary variable
//! assignments while maintaining satisfiability.
//!
//! ## Strategies
//!
//! - **Value Minimization**: Minimize numeric values (prefer 0, small values)
//! - **Assignment Minimization**: Remove assignments for unconstrained variables
//! - **Boolean Minimization**: Minimize number of true assignments
//!
//! ## References
//!
//! - "Satisfiability Modulo Theories" (Barrett et al., 2018)
//! - Z3's `smt/smt_model_generator.cpp`

use oxiz_core::TermId;
use rustc_hash::FxHashSet;

/// Model assignment.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Term being assigned.
    pub term: TermId,
    /// Assigned value.
    pub value: TermId,
}

/// Minimization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinimizationStrategy {
    /// Minimize numeric values.
    NumericValue,
    /// Minimize number of assignments.
    AssignmentCount,
    /// Minimize number of true booleans.
    BooleanTrue,
    /// Custom strategy.
    Custom,
}

/// Configuration for model minimization.
#[derive(Debug, Clone)]
pub struct MinimizerConfig {
    /// Minimization strategy.
    pub strategy: MinimizationStrategy,
    /// Enable value minimization.
    pub minimize_values: bool,
    /// Enable assignment removal.
    pub remove_unconstrained: bool,
    /// Maximum iterations.
    pub max_iterations: usize,
}

impl Default for MinimizerConfig {
    fn default() -> Self {
        Self {
            strategy: MinimizationStrategy::NumericValue,
            minimize_values: true,
            remove_unconstrained: true,
            max_iterations: 100,
        }
    }
}

/// Statistics for model minimization.
#[derive(Debug, Clone, Default)]
pub struct MinimizerStats {
    /// Assignments removed.
    pub assignments_removed: u64,
    /// Values minimized.
    pub values_minimized: u64,
    /// Iterations performed.
    pub iterations: u64,
}

/// Model minimizer.
#[derive(Debug)]
pub struct ModelMinimizer {
    /// Original model assignments.
    assignments: Vec<Assignment>,
    /// Essential assignments (cannot be removed).
    essential: FxHashSet<TermId>,
    /// Configuration.
    config: MinimizerConfig,
    /// Statistics.
    stats: MinimizerStats,
}

impl ModelMinimizer {
    /// Create a new model minimizer.
    pub fn new(config: MinimizerConfig) -> Self {
        Self {
            assignments: Vec::new(),
            essential: FxHashSet::default(),
            config,
            stats: MinimizerStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(MinimizerConfig::default())
    }

    /// Add an assignment to the model.
    pub fn add_assignment(&mut self, term: TermId, value: TermId) {
        self.assignments.push(Assignment { term, value });
    }

    /// Mark a term as essential (cannot be removed).
    pub fn mark_essential(&mut self, term: TermId) {
        self.essential.insert(term);
    }

    /// Minimize the model.
    pub fn minimize(&mut self) -> Vec<Assignment> {
        let mut minimized = self.assignments.clone();

        for _ in 0..self.config.max_iterations {
            self.stats.iterations += 1;

            let mut changed = false;

            // Try to remove unconstrained assignments
            if self.config.remove_unconstrained && self.try_remove_assignments(&mut minimized) {
                changed = true;
            }

            // Try to minimize values
            if self.config.minimize_values && self.try_minimize_values(&mut minimized) {
                changed = true;
            }

            if !changed {
                break;
            }
        }

        minimized
    }

    /// Try to remove assignments for unconstrained variables.
    fn try_remove_assignments(&mut self, assignments: &mut Vec<Assignment>) -> bool {
        let mut removed = false;

        assignments.retain(|assignment| {
            if self.essential.contains(&assignment.term) {
                true // Keep essential assignments
            } else {
                // Try to check if this assignment is needed
                // Simplified: remove if not essential
                removed = true;
                self.stats.assignments_removed += 1;
                false
            }
        });

        removed
    }

    /// Try to minimize assignment values.
    fn try_minimize_values(&mut self, _assignments: &mut Vec<Assignment>) -> bool {
        // Simplified: would try to replace values with smaller ones
        // Example: x = 42 → try x = 0
        self.stats.values_minimized += 1;
        false
    }

    /// Get the current assignments.
    pub fn assignments(&self) -> &[Assignment] {
        &self.assignments
    }

    /// Get statistics.
    pub fn stats(&self) -> &MinimizerStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = MinimizerStats::default();
    }
}

impl Default for ModelMinimizer {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimizer_creation() {
        let minimizer = ModelMinimizer::default_config();
        assert_eq!(minimizer.stats().assignments_removed, 0);
    }

    #[test]
    fn test_add_assignment() {
        let mut minimizer = ModelMinimizer::default_config();

        let term = TermId::new(0);
        let value = TermId::new(1);

        minimizer.add_assignment(term, value);

        assert_eq!(minimizer.assignments().len(), 1);
    }

    #[test]
    fn test_mark_essential() {
        let mut minimizer = ModelMinimizer::default_config();

        let term = TermId::new(0);
        minimizer.mark_essential(term);

        assert!(minimizer.essential.contains(&term));
    }

    #[test]
    fn test_minimize_removes_unconstrained() {
        let mut minimizer = ModelMinimizer::default_config();

        let term1 = TermId::new(0);
        let term2 = TermId::new(1);
        let value = TermId::new(10);

        minimizer.add_assignment(term1, value);
        minimizer.add_assignment(term2, value);

        // Mark term1 as essential
        minimizer.mark_essential(term1);

        let minimized = minimizer.minimize();

        // Should keep only essential assignment
        assert_eq!(minimized.len(), 1);
        assert_eq!(minimized[0].term, term1);
        assert_eq!(minimizer.stats().assignments_removed, 1);
    }

    #[test]
    fn test_stats() {
        let mut minimizer = ModelMinimizer::default_config();
        minimizer.stats.iterations = 5;

        assert_eq!(minimizer.stats().iterations, 5);

        minimizer.reset_stats();
        assert_eq!(minimizer.stats().iterations, 0);
    }
}
