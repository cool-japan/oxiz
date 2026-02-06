//! Theory-Specific Propagator Interface.
//!
//! Defines the interface for theory-specific propagation engines
//! that integrate with the CDCL(T) framework.
//!
//! ## Propagation Types
//!
//! - **Unit propagation**: Theory learns new unit clauses
//! - **Theory propagation**: Theory-specific inference (e.g., bounds)
//! - **Conflict**: Theory detects inconsistency
//!
//! ## References
//!
//! - "DPLL(T): Fast Decision Procedures" (Ganzinger et al., 2004)
//! - Z3's `smt/smt_theory.h`

use oxiz_sat::Lit;

/// Theory identifier.
pub type TheoryId = u8;

/// Propagation result from a theory.
#[derive(Debug, Clone)]
pub enum PropagationResult {
    /// No propagation.
    None,
    /// Theory propagated literals.
    Propagated(Vec<Lit>),
    /// Theory detected a conflict.
    Conflict(Vec<Lit>),
}

/// Explanation for a propagated literal.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// Propagated literal.
    pub literal: Lit,
    /// Reason (antecedent literals).
    pub reason: Vec<Lit>,
    /// Theory that produced this propagation.
    pub theory: TheoryId,
}

/// Interface for theory-specific propagators.
pub trait TheoryPropagator: Send + Sync {
    /// Get the theory ID.
    fn theory_id(&self) -> TheoryId;

    /// Propagate based on current assignment.
    fn propagate(&mut self, assignment: &[Lit]) -> PropagationResult;

    /// Explain why a literal was propagated.
    fn explain(&self, literal: Lit) -> Option<Explanation>;

    /// Backtrack to a given decision level.
    fn backtrack(&mut self, level: usize);

    /// Final check after all literals are assigned.
    fn final_check(&mut self) -> PropagationResult {
        PropagationResult::None
    }

    /// Reset the propagator state.
    fn reset(&mut self);
}

/// Configuration for theory propagator manager.
#[derive(Debug, Clone)]
pub struct PropagatorConfig {
    /// Enable theory propagation.
    pub enable_propagation: bool,
    /// Enable explanation caching.
    pub cache_explanations: bool,
    /// Maximum propagations per call.
    pub max_propagations: usize,
}

impl Default for PropagatorConfig {
    fn default() -> Self {
        Self {
            enable_propagation: true,
            cache_explanations: true,
            max_propagations: 1000,
        }
    }
}

/// Statistics for theory propagator.
#[derive(Debug, Clone, Default)]
pub struct PropagatorStats {
    /// Propagations performed.
    pub propagations: u64,
    /// Conflicts detected.
    pub conflicts: u64,
    /// Explanations generated.
    pub explanations: u64,
    /// Backtrack calls.
    pub backtracks: u64,
}

/// Manager for multiple theory propagators.
pub struct PropagatorManager {
    /// Registered propagators.
    propagators: Vec<Box<dyn TheoryPropagator>>,
    /// Configuration.
    config: PropagatorConfig,
    /// Statistics.
    stats: PropagatorStats,
}

impl PropagatorManager {
    /// Create a new propagator manager.
    pub fn new(config: PropagatorConfig) -> Self {
        Self {
            propagators: Vec::new(),
            config,
            stats: PropagatorStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PropagatorConfig::default())
    }

    /// Register a theory propagator.
    pub fn register(&mut self, propagator: Box<dyn TheoryPropagator>) {
        self.propagators.push(propagator);
    }

    /// Propagate through all theories.
    pub fn propagate_all(&mut self, assignment: &[Lit]) -> PropagationResult {
        if !self.config.enable_propagation {
            return PropagationResult::None;
        }

        let mut all_propagated = Vec::new();

        for propagator in &mut self.propagators {
            match propagator.propagate(assignment) {
                PropagationResult::Conflict(lits) => {
                    self.stats.conflicts += 1;
                    return PropagationResult::Conflict(lits);
                }
                PropagationResult::Propagated(lits) => {
                    self.stats.propagations += lits.len() as u64;
                    all_propagated.extend(lits);

                    // Check max propagations limit
                    if all_propagated.len() >= self.config.max_propagations {
                        break;
                    }
                }
                PropagationResult::None => {}
            }
        }

        if all_propagated.is_empty() {
            PropagationResult::None
        } else {
            PropagationResult::Propagated(all_propagated)
        }
    }

    /// Explain a propagated literal.
    pub fn explain(&mut self, literal: Lit) -> Option<Explanation> {
        for propagator in &self.propagators {
            if let Some(explanation) = propagator.explain(literal) {
                self.stats.explanations += 1;
                return Some(explanation);
            }
        }
        None
    }

    /// Backtrack all theories.
    pub fn backtrack(&mut self, level: usize) {
        for propagator in &mut self.propagators {
            propagator.backtrack(level);
        }
        self.stats.backtracks += 1;
    }

    /// Final check on all theories.
    pub fn final_check(&mut self) -> PropagationResult {
        for propagator in &mut self.propagators {
            if let result @ (PropagationResult::Conflict(_) | PropagationResult::Propagated(_)) =
                propagator.final_check()
            {
                return result;
            }
        }
        PropagationResult::None
    }

    /// Reset all propagators.
    pub fn reset_all(&mut self) {
        for propagator in &mut self.propagators {
            propagator.reset();
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &PropagatorStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PropagatorStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock theory propagator for testing
    struct MockPropagator {
        id: TheoryId,
    }

    impl TheoryPropagator for MockPropagator {
        fn theory_id(&self) -> TheoryId {
            self.id
        }

        fn propagate(&mut self, _assignment: &[Lit]) -> PropagationResult {
            use oxiz_sat::Var;
            // Mock: always propagate one literal
            PropagationResult::Propagated(vec![Lit::pos(Var::new(10))])
        }

        fn explain(&self, _literal: Lit) -> Option<Explanation> {
            use oxiz_sat::Var;
            Some(Explanation {
                literal: Lit::pos(Var::new(10)),
                reason: vec![],
                theory: self.id,
            })
        }

        fn backtrack(&mut self, _level: usize) {}

        fn reset(&mut self) {}
    }

    #[test]
    fn test_manager_creation() {
        let manager = PropagatorManager::default_config();
        assert_eq!(manager.stats().propagations, 0);
    }

    #[test]
    fn test_register_propagator() {
        let mut manager = PropagatorManager::default_config();
        let mock = Box::new(MockPropagator { id: 1 });

        manager.register(mock);
        assert_eq!(manager.propagators.len(), 1);
    }

    #[test]
    fn test_propagate_all() {
        use oxiz_sat::Var;
        let mut manager = PropagatorManager::default_config();
        manager.register(Box::new(MockPropagator { id: 1 }));

        let assignment = vec![Lit::pos(Var::new(0))];
        let result = manager.propagate_all(&assignment);

        match result {
            PropagationResult::Propagated(lits) => {
                assert!(!lits.is_empty());
                assert_eq!(manager.stats().propagations, lits.len() as u64);
            }
            _ => panic!("Expected propagation"),
        }
    }

    #[test]
    fn test_explain() {
        use oxiz_sat::Var;
        let mut manager = PropagatorManager::default_config();
        manager.register(Box::new(MockPropagator { id: 1 }));

        let lit = Lit::pos(Var::new(10));
        let explanation = manager.explain(lit);

        assert!(explanation.is_some());
        assert_eq!(manager.stats().explanations, 1);
    }

    #[test]
    fn test_backtrack() {
        let mut manager = PropagatorManager::default_config();
        manager.register(Box::new(MockPropagator { id: 1 }));

        manager.backtrack(5);
        assert_eq!(manager.stats().backtracks, 1);
    }

    #[test]
    fn test_stats() {
        let mut manager = PropagatorManager::default_config();
        manager.stats.propagations = 100;

        assert_eq!(manager.stats().propagations, 100);

        manager.reset_stats();
        assert_eq!(manager.stats().propagations, 0);
    }
}
