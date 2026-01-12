//! User Propagator Framework
//!
//! Allows users to integrate custom theory solvers and propagation logic into OxiZ.
//!
//! # Overview
//!
//! The user propagator system enables:
//! 1. **Custom Theory Integration**: Define domain-specific reasoning
//! 2. **Event Callbacks**: React to fixed values, equalities, and decisions
//! 3. **Propagation**: Derive and propagate custom consequences
//! 4. **Branching Hints**: Guide the solver's search strategy
//!
//! # Example
//!
//! ```rust,ignore
//! use oxiz_theories::user_propagator::*;
//!
//! struct MyTheory {
//!     // Custom state
//! }
//!
//! impl UserPropagator for MyTheory {
//!     fn on_fixed(&mut self, term: TermId, value: TermId, ctx: &mut PropagatorContext) {
//!         // React to term getting a fixed value
//!         // Can propagate consequences via ctx.propagate(...)
//!     }
//!
//!     fn on_equality(&mut self, lhs: TermId, rhs: TermId, ctx: &mut PropagatorContext) {
//!         // React to equality lhs = rhs
//!     }
//!
//!     fn final_check(&mut self, ctx: &mut PropagatorContext) -> PropagatorResult {
//!         // Perform complete theory check
//!         PropagatorResult::Sat
//!     }
//! }
//! ```

use oxiz_core::ast::TermId;
use std::collections::{HashMap, HashSet, VecDeque};

/// Result from a propagator operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropagatorResult {
    /// Satisfiable - no conflicts found
    Sat,
    /// Unsatisfiable with conflict clause
    Unsat(Vec<TermId>),
    /// Unknown - incomplete check
    Unknown,
}

/// Consequence to propagate
#[derive(Debug, Clone)]
pub struct Consequence {
    /// The consequence term to assert as true
    pub term: TermId,
    /// Justification (antecedent literals)
    pub justification: Vec<TermId>,
}

impl Consequence {
    /// Create a new consequence
    pub fn new(term: TermId, justification: Vec<TermId>) -> Self {
        Self {
            term,
            justification,
        }
    }
}

/// Context provided to user propagators for callbacks
pub struct PropagatorContext<'a> {
    /// Queue of consequences to propagate
    consequences: &'a mut VecDeque<Consequence>,
    /// Fixed terms
    fixed_terms: &'a HashMap<TermId, TermId>,
    /// Equalities
    equalities: &'a HashSet<(TermId, TermId)>,
}

impl<'a> PropagatorContext<'a> {
    /// Create a new propagator context
    pub(crate) fn new(
        consequences: &'a mut VecDeque<Consequence>,
        fixed_terms: &'a HashMap<TermId, TermId>,
        equalities: &'a HashSet<(TermId, TermId)>,
    ) -> Self {
        Self {
            consequences,
            fixed_terms,
            equalities,
        }
    }

    /// Propagate a consequence
    pub fn propagate(&mut self, consequence: Consequence) {
        self.consequences.push_back(consequence);
    }

    /// Get the fixed value for a term, if any
    pub fn get_fixed_value(&self, term: TermId) -> Option<TermId> {
        self.fixed_terms.get(&term).copied()
    }

    /// Check if two terms are equal
    pub fn are_equal(&self, lhs: TermId, rhs: TermId) -> bool {
        self.equalities.contains(&(lhs, rhs)) || self.equalities.contains(&(rhs, lhs))
    }
}

/// Trait for user-defined propagators
///
/// Implement this trait to integrate custom theory reasoning into OxiZ.
pub trait UserPropagator: Send + Sync {
    /// Called when a term gets a fixed value
    ///
    /// # Arguments
    /// * `term` - The term that got fixed
    /// * `value` - The value assigned to the term
    /// * `ctx` - Context for propagating consequences
    fn on_fixed(&mut self, _term: TermId, _value: TermId, _ctx: &mut PropagatorContext) {
        // Default: do nothing
    }

    /// Called when two terms become equal
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side of equality
    /// * `rhs` - Right-hand side of equality
    /// * `ctx` - Context for propagating consequences
    fn on_equality(&mut self, _lhs: TermId, _rhs: TermId, _ctx: &mut PropagatorContext) {
        // Default: do nothing
    }

    /// Called when two terms become disequal
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side of disequality
    /// * `rhs` - Right-hand side of disequality
    /// * `ctx` - Context for propagating consequences
    fn on_disequality(&mut self, _lhs: TermId, _rhs: TermId, _ctx: &mut PropagatorContext) {
        // Default: do nothing
    }

    /// Called when a new term is created
    ///
    /// Allows the propagator to track or register the term.
    fn on_created(&mut self, _term: TermId) {
        // Default: do nothing
    }

    /// Called during final check (SAT-complete check)
    ///
    /// The propagator should perform a complete satisfiability check.
    fn final_check(&mut self, _ctx: &mut PropagatorContext) -> PropagatorResult {
        PropagatorResult::Sat
    }

    /// Called before making a branching decision
    ///
    /// Returns `Some((var, phase))` to guide the decision, or `None` to let the solver decide.
    fn decide(&mut self) -> Option<(TermId, bool)> {
        None
    }

    /// Push a new context level
    fn push(&mut self) {
        // Default: do nothing
    }

    /// Pop context levels
    fn pop(&mut self, _levels: usize) {
        // Default: do nothing
    }

    /// Reset the propagator
    fn reset(&mut self) {
        // Default: do nothing
    }
}

/// Statistics for user propagators
#[derive(Debug, Clone, Default)]
pub struct UserPropagatorStats {
    /// Number of fixed callbacks
    pub num_fixed_callbacks: usize,
    /// Number of equality callbacks
    pub num_eq_callbacks: usize,
    /// Number of disequality callbacks
    pub num_diseq_callbacks: usize,
    /// Number of created callbacks
    pub num_created_callbacks: usize,
    /// Number of final checks
    pub num_final_checks: usize,
    /// Number of propagated consequences
    pub num_propagations: usize,
    /// Number of conflicts found
    pub num_conflicts: usize,
}

impl UserPropagatorStats {
    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Manager for user propagators
pub struct UserPropagatorManager {
    /// Registered propagators
    propagators: Vec<Box<dyn UserPropagator>>,
    /// Fixed terms (term -> value)
    fixed_terms: HashMap<TermId, TermId>,
    /// Known equalities
    equalities: HashSet<(TermId, TermId)>,
    /// Pending consequences
    consequences: VecDeque<Consequence>,
    /// Watched terms
    watched_terms: HashSet<TermId>,
    /// Statistics
    stats: UserPropagatorStats,
    /// Context stack for push/pop
    context_stack: Vec<usize>,
}

impl UserPropagatorManager {
    /// Create a new user propagator manager
    pub fn new() -> Self {
        Self {
            propagators: Vec::new(),
            fixed_terms: HashMap::new(),
            equalities: HashSet::new(),
            consequences: VecDeque::new(),
            watched_terms: HashSet::new(),
            stats: UserPropagatorStats::default(),
            context_stack: Vec::new(),
        }
    }

    /// Register a user propagator
    pub fn register_propagator(&mut self, propagator: Box<dyn UserPropagator>) {
        self.propagators.push(propagator);
    }

    /// Watch a term (trigger callbacks for this term)
    pub fn watch_term(&mut self, term: TermId) {
        self.watched_terms.insert(term);
    }

    /// Notify that a term has a fixed value
    pub fn notify_fixed(&mut self, term: TermId, value: TermId) {
        if !self.watched_terms.contains(&term) {
            return;
        }

        self.fixed_terms.insert(term, value);
        self.stats.num_fixed_callbacks = self.stats.num_fixed_callbacks.saturating_add(1);

        let mut ctx =
            PropagatorContext::new(&mut self.consequences, &self.fixed_terms, &self.equalities);

        for prop in &mut self.propagators {
            prop.on_fixed(term, value, &mut ctx);
        }
    }

    /// Notify that two terms are equal
    pub fn notify_equality(&mut self, lhs: TermId, rhs: TermId) {
        if !self.watched_terms.contains(&lhs) && !self.watched_terms.contains(&rhs) {
            return;
        }

        self.equalities.insert((lhs, rhs));
        self.stats.num_eq_callbacks = self.stats.num_eq_callbacks.saturating_add(1);

        let mut ctx =
            PropagatorContext::new(&mut self.consequences, &self.fixed_terms, &self.equalities);

        for prop in &mut self.propagators {
            prop.on_equality(lhs, rhs, &mut ctx);
        }
    }

    /// Notify that two terms are disequal
    pub fn notify_disequality(&mut self, lhs: TermId, rhs: TermId) {
        if !self.watched_terms.contains(&lhs) && !self.watched_terms.contains(&rhs) {
            return;
        }

        self.stats.num_diseq_callbacks = self.stats.num_diseq_callbacks.saturating_add(1);

        let mut ctx =
            PropagatorContext::new(&mut self.consequences, &self.fixed_terms, &self.equalities);

        for prop in &mut self.propagators {
            prop.on_disequality(lhs, rhs, &mut ctx);
        }
    }

    /// Notify that a new term was created
    pub fn notify_created(&mut self, term: TermId) {
        self.stats.num_created_callbacks = self.stats.num_created_callbacks.saturating_add(1);

        for prop in &mut self.propagators {
            prop.on_created(term);
        }
    }

    /// Perform final check (complete satisfiability)
    pub fn final_check(&mut self) -> PropagatorResult {
        self.stats.num_final_checks = self.stats.num_final_checks.saturating_add(1);

        let mut ctx =
            PropagatorContext::new(&mut self.consequences, &self.fixed_terms, &self.equalities);

        for prop in &mut self.propagators {
            match prop.final_check(&mut ctx) {
                PropagatorResult::Sat => continue,
                PropagatorResult::Unsat(conflict) => {
                    self.stats.num_conflicts = self.stats.num_conflicts.saturating_add(1);
                    return PropagatorResult::Unsat(conflict);
                }
                PropagatorResult::Unknown => return PropagatorResult::Unknown,
            }
        }

        PropagatorResult::Sat
    }

    /// Get the next branching decision from propagators
    pub fn get_decision(&mut self) -> Option<(TermId, bool)> {
        for prop in &mut self.propagators {
            if let Some(decision) = prop.decide() {
                return Some(decision);
            }
        }
        None
    }

    /// Get pending consequences to propagate
    pub fn get_consequences(&mut self) -> Vec<Consequence> {
        let consequences: Vec<_> = self.consequences.drain(..).collect();
        self.stats.num_propagations = self
            .stats
            .num_propagations
            .saturating_add(consequences.len());
        consequences
    }

    /// Check if there are pending consequences
    pub fn has_consequences(&self) -> bool {
        !self.consequences.is_empty()
    }

    /// Push a new context level
    pub fn push(&mut self) {
        self.context_stack.push(self.fixed_terms.len());
        for prop in &mut self.propagators {
            prop.push();
        }
    }

    /// Pop context levels
    pub fn pop(&mut self, levels: usize) {
        if levels == 0 {
            return;
        }

        for _ in 0..levels {
            if let Some(size) = self.context_stack.pop() {
                // Restore fixed terms
                while self.fixed_terms.len() > size {
                    if let Some((term, _)) = self.fixed_terms.iter().next() {
                        let term = *term;
                        self.fixed_terms.remove(&term);
                    }
                }
            }
        }

        for prop in &mut self.propagators {
            prop.pop(levels);
        }
    }

    /// Reset the manager
    pub fn reset(&mut self) {
        self.fixed_terms.clear();
        self.equalities.clear();
        self.consequences.clear();
        self.watched_terms.clear();
        self.context_stack.clear();
        self.stats.reset();

        for prop in &mut self.propagators {
            prop.reset();
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &UserPropagatorStats {
        &self.stats
    }

    /// Get number of registered propagators
    pub fn num_propagators(&self) -> usize {
        self.propagators.len()
    }
}

impl Default for UserPropagatorManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPropagator {
        fixed_count: usize,
        eq_count: usize,
    }

    impl DummyPropagator {
        fn new() -> Self {
            Self {
                fixed_count: 0,
                eq_count: 0,
            }
        }
    }

    impl UserPropagator for DummyPropagator {
        fn on_fixed(&mut self, _term: TermId, _value: TermId, _ctx: &mut PropagatorContext) {
            self.fixed_count = self.fixed_count.saturating_add(1);
        }

        fn on_equality(&mut self, _lhs: TermId, _rhs: TermId, _ctx: &mut PropagatorContext) {
            self.eq_count = self.eq_count.saturating_add(1);
        }

        fn final_check(&mut self, _ctx: &mut PropagatorContext) -> PropagatorResult {
            PropagatorResult::Sat
        }
    }

    #[test]
    fn test_manager_creation() {
        let manager = UserPropagatorManager::new();
        assert_eq!(manager.num_propagators(), 0);
        assert!(!manager.has_consequences());
    }

    #[test]
    fn test_register_propagator() {
        let mut manager = UserPropagatorManager::new();
        manager.register_propagator(Box::new(DummyPropagator::new()));
        assert_eq!(manager.num_propagators(), 1);
    }

    #[test]
    fn test_watch_term() {
        let mut manager = UserPropagatorManager::new();
        let term = TermId::new(1);
        manager.watch_term(term);
        assert!(manager.watched_terms.contains(&term));
    }

    #[test]
    fn test_notify_fixed() {
        let mut manager = UserPropagatorManager::new();
        manager.register_propagator(Box::new(DummyPropagator::new()));

        let term = TermId::new(1);
        let value = TermId::new(2);

        manager.watch_term(term);
        manager.notify_fixed(term, value);

        assert_eq!(manager.stats().num_fixed_callbacks, 1);
        assert_eq!(manager.get_fixed_value(term), Some(value));
    }

    #[test]
    fn test_notify_equality() {
        let mut manager = UserPropagatorManager::new();
        manager.register_propagator(Box::new(DummyPropagator::new()));

        let lhs = TermId::new(1);
        let rhs = TermId::new(2);

        manager.watch_term(lhs);
        manager.notify_equality(lhs, rhs);

        assert_eq!(manager.stats().num_eq_callbacks, 1);
    }

    #[test]
    fn test_final_check() {
        let mut manager = UserPropagatorManager::new();
        manager.register_propagator(Box::new(DummyPropagator::new()));

        let result = manager.final_check();
        assert_eq!(result, PropagatorResult::Sat);
        assert_eq!(manager.stats().num_final_checks, 1);
    }

    #[test]
    fn test_push_pop() {
        let mut manager = UserPropagatorManager::new();
        manager.register_propagator(Box::new(DummyPropagator::new()));

        let term = TermId::new(1);
        let value = TermId::new(2);

        manager.push();
        manager.watch_term(term);
        manager.notify_fixed(term, value);

        assert_eq!(manager.get_fixed_value(term), Some(value));

        manager.pop(1);
        // Note: watched_terms don't get cleared on pop in this simple impl
    }

    #[test]
    fn test_consequence_propagation() {
        let mut manager = UserPropagatorManager::new();

        struct PropagatingPropagator;
        impl UserPropagator for PropagatingPropagator {
            fn on_fixed(&mut self, _term: TermId, _value: TermId, ctx: &mut PropagatorContext) {
                let cons = Consequence::new(TermId::new(100), vec![]);
                ctx.propagate(cons);
            }
        }

        manager.register_propagator(Box::new(PropagatingPropagator));
        let term = TermId::new(1);
        manager.watch_term(term);
        manager.notify_fixed(term, TermId::new(2));

        assert!(manager.has_consequences());
        let consequences = manager.get_consequences();
        assert_eq!(consequences.len(), 1);
    }

    impl UserPropagatorManager {
        fn get_fixed_value(&self, term: TermId) -> Option<TermId> {
            self.fixed_terms.get(&term).copied()
        }
    }
}
