//! Generic Propagation Trait.
//!
//! Defines interfaces for implementing domain-specific propagators that can
//! deduce new constraints from existing ones.

use crate::TermId;
use crate::literal::Lit;

/// Result of a propagation operation.
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Literals that were propagated.
    pub propagated: Vec<Lit>,
    /// For each propagated literal, the explanation (antecedents).
    pub explanations: Vec<Vec<Lit>>,
}

impl PropagationResult {
    /// Create an empty propagation result.
    pub fn empty() -> Self {
        Self {
            propagated: Vec::new(),
            explanations: Vec::new(),
        }
    }

    /// Check if any literals were propagated.
    pub fn is_empty(&self) -> bool {
        self.propagated.is_empty()
    }

    /// Number of propagated literals.
    pub fn len(&self) -> usize {
        self.propagated.len()
    }

    /// Add a propagation with its explanation.
    pub fn add(&mut self, lit: Lit, explanation: Vec<Lit>) {
        self.propagated.push(lit);
        self.explanations.push(explanation);
    }

    /// Merge another propagation result into this one.
    pub fn merge(&mut self, other: PropagationResult) {
        self.propagated.extend(other.propagated);
        self.explanations.extend(other.explanations);
    }
}

/// Priority level for propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PropagationPriority {
    /// Lowest priority (deferred propagation).
    Low = 0,
    /// Normal priority (theory propagation).
    Normal = 1,
    /// High priority (unit propagation).
    High = 2,
    /// Urgent priority (Boolean constraint propagation).
    Urgent = 3,
}

/// Status of a propagator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagatorStatus {
    /// Propagator is active and may propagate.
    Active,
    /// Propagator is paused (e.g., waiting for more information).
    Paused,
    /// Propagator has found a conflict.
    Conflicted,
}

/// Core trait for implementing propagators.
///
/// A propagator takes current assignments and deduces new facts,
/// either through constraint propagation or theory reasoning.
pub trait Propagator: Send + Sync {
    /// Get the name of this propagator.
    fn name(&self) -> &str;

    /// Reset the propagator to initial state.
    fn reset(&mut self);

    /// Notify the propagator of a new assignment.
    ///
    /// The propagator should update its internal state but not propagate yet.
    fn notify_assigned(&mut self, lit: Lit, level: usize);

    /// Notify the propagator that a literal was unassigned (backtrack).
    fn notify_unassigned(&mut self, lit: Lit);

    /// Attempt to propagate based on current assignments.
    ///
    /// Returns propagated literals with their explanations.
    fn propagate(&mut self) -> PropagationResult;

    /// Check if the propagator can propagate (has pending work).
    fn has_pending(&self) -> bool {
        false
    }

    /// Get the current status of this propagator.
    fn status(&self) -> PropagatorStatus {
        PropagatorStatus::Active
    }

    /// Get the priority of this propagator.
    fn priority(&self) -> PropagationPriority {
        PropagationPriority::Normal
    }

    /// Explain why a literal was propagated.
    ///
    /// Returns the set of literals (antecedents) that justify the propagation.
    fn explain(&self, lit: Lit) -> Vec<Lit>;

    /// Get a conflict explanation if the propagator detected inconsistency.
    fn get_conflict(&self) -> Option<Vec<Lit>> {
        None
    }

    /// Check if this propagator is incremental.
    fn is_incremental(&self) -> bool {
        true
    }

    /// Push a new scope (if incremental).
    fn push(&mut self) {
        // Default: no-op
    }

    /// Pop the last scope (if incremental).
    fn pop(&mut self) {
        // Default: no-op
    }
}

/// Extension trait for watched literal propagators.
///
/// Watched literal propagation is an optimization where constraints only
/// watch a subset of their literals, reducing the overhead of notification.
pub trait WatchedPropagator: Propagator {
    /// Add a watch for a literal in a constraint.
    fn add_watch(&mut self, lit: Lit, constraint_id: usize);

    /// Remove a watch for a literal.
    fn remove_watch(&mut self, lit: Lit, constraint_id: usize);

    /// Get all constraints watching a literal.
    fn get_watches(&self, lit: Lit) -> Vec<usize>;

    /// Process a watched literal becoming assigned.
    ///
    /// Returns:
    /// - `Ok(propagations)` if propagation succeeded
    /// - `Err(conflict)` if a conflict was detected
    fn process_watch(
        &mut self,
        constraint_id: usize,
        lit: Lit,
        value: bool,
    ) -> Result<Vec<Lit>, Vec<Lit>>;
}

/// Extension trait for lazy propagators.
///
/// Lazy propagators defer expensive propagation until absolutely necessary,
/// which can improve performance for constraints that rarely propagate.
pub trait LazyPropagator: Propagator {
    /// Check if propagation is worth attempting.
    ///
    /// Returns `true` if the propagator should run, `false` to skip.
    fn should_propagate(&self) -> bool;

    /// Estimate the cost of running this propagator (in arbitrary units).
    fn propagation_cost(&self) -> u64 {
        100
    }

    /// Get the number of times this propagator has run.
    fn propagation_count(&self) -> u64;

    /// Get the total time spent in propagation (in microseconds).
    fn total_propagation_time(&self) -> u64;
}

/// Extension trait for theory propagators that support model-based reasoning.
pub trait ModelBasedPropagator: Propagator {
    /// Propagate based on a partial model.
    ///
    /// This is called when the SAT solver has a complete assignment,
    /// but the theory solver may deduce additional constraints.
    fn propagate_from_model(&mut self, assignments: &[(Lit, bool)]) -> PropagationResult;

    /// Get relevant terms for the current model.
    fn get_relevant_terms(&self) -> Vec<TermId>;
}

/// Manager for coordinating multiple propagators.
pub struct PropagatorManager {
    /// Registered propagators.
    propagators: Vec<Box<dyn Propagator>>,
    /// Propagation queue (priority, propagator index).
    queue: Vec<(PropagationPriority, usize)>,
}

impl PropagatorManager {
    /// Create a new propagator manager.
    pub fn new() -> Self {
        Self {
            propagators: Vec::new(),
            queue: Vec::new(),
        }
    }

    /// Register a new propagator.
    pub fn register(&mut self, propagator: Box<dyn Propagator>) {
        self.propagators.push(propagator);
    }

    /// Notify all propagators of a new assignment.
    pub fn notify_assigned(&mut self, lit: Lit, level: usize) {
        for (idx, prop) in self.propagators.iter_mut().enumerate() {
            prop.notify_assigned(lit, level);
            if prop.has_pending() {
                let priority = prop.priority();
                self.queue.push((priority, idx));
            }
        }
    }

    /// Run all pending propagations.
    ///
    /// Returns the combined propagation result.
    pub fn propagate_all(&mut self) -> PropagationResult {
        // Sort queue by priority (highest first)
        self.queue.sort_by(|a, b| b.0.cmp(&a.0));

        let mut result = PropagationResult::empty();

        while let Some((_priority, idx)) = self.queue.pop() {
            if let Some(prop) = self.propagators.get_mut(idx)
                && prop.status() == PropagatorStatus::Active
            {
                let prop_result = prop.propagate();
                result.merge(prop_result);
            }
        }

        result
    }

    /// Reset all propagators.
    pub fn reset(&mut self) {
        for prop in &mut self.propagators {
            prop.reset();
        }
        self.queue.clear();
    }

    /// Backtrack all propagators to a decision level.
    pub fn backtrack(&mut self, _level: usize) {
        // Note: This is simplified. A real implementation would track
        // which literals were assigned at which level.
        for prop in &mut self.propagators {
            prop.reset();
        }
        self.queue.clear();
    }

    /// Get the number of registered propagators.
    pub fn num_propagators(&self) -> usize {
        self.propagators.len()
    }
}

impl Default for PropagatorManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock propagator for testing
    struct MockPropagator {
        name: String,
        pending: bool,
        propagated: Vec<Lit>,
    }

    impl MockPropagator {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                pending: false,
                propagated: Vec::new(),
            }
        }
    }

    impl Propagator for MockPropagator {
        fn name(&self) -> &str {
            &self.name
        }

        fn reset(&mut self) {
            self.pending = false;
            self.propagated.clear();
        }

        fn notify_assigned(&mut self, lit: Lit, _level: usize) {
            self.propagated.push(lit);
            self.pending = true;
        }

        fn notify_unassigned(&mut self, _lit: Lit) {
            // No-op
        }

        fn propagate(&mut self) -> PropagationResult {
            self.pending = false;
            PropagationResult::empty()
        }

        fn has_pending(&self) -> bool {
            self.pending
        }

        fn explain(&self, _lit: Lit) -> Vec<Lit> {
            Vec::new()
        }
    }

    #[test]
    fn test_propagation_result() {
        let mut result = PropagationResult::empty();
        assert!(result.is_empty());

        result.add(Lit::positive(1), vec![]);
        assert_eq!(result.len(), 1);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_propagator_manager() {
        let mut manager = PropagatorManager::new();

        let prop1 = Box::new(MockPropagator::new("prop1"));
        let prop2 = Box::new(MockPropagator::new("prop2"));

        manager.register(prop1);
        manager.register(prop2);

        assert_eq!(manager.num_propagators(), 2);
    }

    #[test]
    fn test_notify_and_propagate() {
        let mut manager = PropagatorManager::new();
        manager.register(Box::new(MockPropagator::new("test")));

        manager.notify_assigned(Lit::positive(1), 0);
        let result = manager.propagate_all();

        // Mock propagator doesn't actually propagate anything
        assert!(result.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let priorities = [
            PropagationPriority::Low,
            PropagationPriority::Normal,
            PropagationPriority::High,
            PropagationPriority::Urgent,
        ];

        for i in 0..priorities.len() - 1 {
            assert!(priorities[i] < priorities[i + 1]);
        }
    }
}
