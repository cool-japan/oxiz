//! Premise tracking for proof generation during solving.
//!
//! This module provides infrastructure to track which premises (axioms, assertions)
//! are used in each step of the solving process, enabling precise proof generation.

use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Unique identifier for a premise (assertion, axiom, or assumption).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PremiseId(pub u32);

impl fmt::Display for PremiseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "p{}", self.0)
    }
}

/// A premise (assertion or assumption) in the original problem.
#[derive(Debug, Clone)]
pub struct Premise {
    /// Unique identifier.
    pub id: PremiseId,
    /// String representation of the premise (SMT-LIB term).
    pub term: String,
    /// Whether this premise is an assumption (vs assertion).
    pub is_assumption: bool,
}

/// Tracks premise dependencies during solving.
#[derive(Debug, Clone)]
pub struct PremiseTracker {
    /// All registered premises.
    premises: Vec<Premise>,
    /// Next available premise ID.
    next_id: u32,
    /// Map from terms to premise IDs (for deduplication).
    term_to_id: FxHashMap<String, PremiseId>,
    /// Active premises in the current scope.
    active_premises: FxHashSet<PremiseId>,
}

impl PremiseTracker {
    /// Create a new premise tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            premises: Vec::new(),
            next_id: 0,
            term_to_id: FxHashMap::default(),
            active_premises: FxHashSet::default(),
        }
    }

    /// Register a new premise (assertion or assumption).
    ///
    /// Returns the premise ID. If the term was already registered,
    /// returns the existing ID.
    pub fn add_premise(&mut self, term: impl Into<String>, is_assumption: bool) -> PremiseId {
        let term = term.into();

        // Check if already registered
        if let Some(&id) = self.term_to_id.get(&term) {
            return id;
        }

        let id = PremiseId(self.next_id);
        self.next_id += 1;

        self.premises.push(Premise {
            id,
            term: term.clone(),
            is_assumption,
        });

        self.term_to_id.insert(term, id);
        self.active_premises.insert(id);

        id
    }

    /// Register an assertion.
    pub fn add_assertion(&mut self, term: impl Into<String>) -> PremiseId {
        self.add_premise(term, false)
    }

    /// Register an assumption (for incremental solving).
    pub fn add_assumption(&mut self, term: impl Into<String>) -> PremiseId {
        self.add_premise(term, true)
    }

    /// Get a premise by ID.
    #[must_use]
    pub fn get_premise(&self, id: PremiseId) -> Option<&Premise> {
        self.premises.get(id.0 as usize)
    }

    /// Get the premise ID for a term, if it exists.
    #[must_use]
    pub fn get_id(&self, term: &str) -> Option<PremiseId> {
        self.term_to_id.get(term).copied()
    }

    /// Check if a premise is active.
    #[must_use]
    pub fn is_active(&self, id: PremiseId) -> bool {
        self.active_premises.contains(&id)
    }

    /// Activate a premise.
    pub fn activate(&mut self, id: PremiseId) {
        self.active_premises.insert(id);
    }

    /// Deactivate a premise (e.g., when popping scope).
    pub fn deactivate(&mut self, id: PremiseId) {
        self.active_premises.remove(&id);
    }

    /// Get all active premises.
    pub fn active_premises(&self) -> impl Iterator<Item = PremiseId> + '_ {
        self.active_premises.iter().copied()
    }

    /// Get all premises.
    #[must_use]
    pub fn all_premises(&self) -> &[Premise] {
        &self.premises
    }

    /// Get the number of premises.
    #[must_use]
    pub fn len(&self) -> usize {
        self.premises.len()
    }

    /// Check if there are no premises.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.premises.is_empty()
    }

    /// Clear all premises.
    pub fn clear(&mut self) {
        self.premises.clear();
        self.next_id = 0;
        self.term_to_id.clear();
        self.active_premises.clear();
    }

    /// Push a new scope level.
    ///
    /// Returns the current set of active premises (for restoration on pop).
    #[must_use]
    pub fn push_scope(&self) -> FxHashSet<PremiseId> {
        self.active_premises.clone()
    }

    /// Pop a scope level, restoring the given set of active premises.
    pub fn pop_scope(&mut self, saved_premises: FxHashSet<PremiseId>) {
        self.active_premises = saved_premises;
    }
}

impl Default for PremiseTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks premise dependencies for derived clauses/terms.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PremiseDependency {
    /// The set of premises that this clause/term depends on.
    dependencies: FxHashMap<u32, FxHashSet<PremiseId>>,
    /// Next available clause/term ID.
    next_id: u32,
}

#[allow(dead_code)]
impl PremiseDependency {
    /// Create a new premise dependency tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dependencies: FxHashMap::default(),
            next_id: 0,
        }
    }

    /// Register a derived clause/term with its premise dependencies.
    ///
    /// Returns a unique ID for this derived item.
    pub fn add_derived(&mut self, premises: impl IntoIterator<Item = PremiseId>) -> u32 {
        let id = self.next_id;
        self.next_id += 1;

        let premise_set: FxHashSet<PremiseId> = premises.into_iter().collect();
        self.dependencies.insert(id, premise_set);

        id
    }

    /// Get the premises that a derived item depends on.
    #[must_use]
    pub fn get_dependencies(&self, id: u32) -> Option<&FxHashSet<PremiseId>> {
        self.dependencies.get(&id)
    }

    /// Merge dependencies from multiple derived items.
    ///
    /// Returns the union of all their premise dependencies.
    #[must_use]
    pub fn merge_dependencies(&self, ids: &[u32]) -> FxHashSet<PremiseId> {
        let mut merged = FxHashSet::default();
        for &id in ids {
            if let Some(deps) = self.dependencies.get(&id) {
                merged.extend(deps);
            }
        }
        merged
    }

    /// Clear all dependency information.
    pub fn clear(&mut self) {
        self.dependencies.clear();
        self.next_id = 0;
    }

    /// Get the number of tracked derived items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.dependencies.len()
    }

    /// Check if there are no tracked dependencies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.dependencies.is_empty()
    }
}

impl Default for PremiseDependency {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_premise_tracker_basic() {
        let mut tracker = PremiseTracker::new();

        let p1 = tracker.add_assertion("(> x 5)");
        let p2 = tracker.add_assertion("(< x 10)");
        let p3 = tracker.add_assumption("(= y 7)");

        assert_eq!(tracker.len(), 3);
        assert!(tracker.is_active(p1));
        assert!(tracker.is_active(p2));
        assert!(tracker.is_active(p3));

        let premise = tracker.get_premise(p1).unwrap();
        assert_eq!(premise.term, "(> x 5)");
        assert!(!premise.is_assumption);

        let assumption = tracker.get_premise(p3).unwrap();
        assert!(assumption.is_assumption);
    }

    #[test]
    fn test_premise_deduplication() {
        let mut tracker = PremiseTracker::new();

        let p1 = tracker.add_assertion("(> x 5)");
        let p2 = tracker.add_assertion("(> x 5)");

        assert_eq!(p1, p2);
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_premise_activation() {
        let mut tracker = PremiseTracker::new();

        let p1 = tracker.add_assertion("(> x 5)");
        assert!(tracker.is_active(p1));

        tracker.deactivate(p1);
        assert!(!tracker.is_active(p1));

        tracker.activate(p1);
        assert!(tracker.is_active(p1));
    }

    #[test]
    fn test_premise_scope() {
        let mut tracker = PremiseTracker::new();

        let p1 = tracker.add_assertion("(> x 5)");
        let saved = tracker.push_scope();

        let p2 = tracker.add_assertion("(< x 10)");
        assert!(tracker.is_active(p1));
        assert!(tracker.is_active(p2));

        tracker.pop_scope(saved);
        assert!(tracker.is_active(p1));
        // p2 is still in the tracker but we restored the active set
        assert!(!tracker.is_active(p2));
    }

    #[test]
    fn test_premise_get_id() {
        let mut tracker = PremiseTracker::new();

        let p1 = tracker.add_assertion("(> x 5)");
        let found = tracker.get_id("(> x 5)");

        assert_eq!(found, Some(p1));
        assert_eq!(tracker.get_id("(< x 10)"), None);
    }

    #[test]
    fn test_premise_clear() {
        let mut tracker = PremiseTracker::new();

        tracker.add_assertion("(> x 5)");
        tracker.add_assertion("(< x 10)");

        assert_eq!(tracker.len(), 2);
        tracker.clear();
        assert_eq!(tracker.len(), 0);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_premise_dependency_basic() {
        let mut dep = PremiseDependency::new();

        let p1 = PremiseId(0);
        let p2 = PremiseId(1);

        let d1 = dep.add_derived(vec![p1, p2]);

        let deps = dep.get_dependencies(d1).unwrap();
        assert_eq!(deps.len(), 2);
        assert!(deps.contains(&p1));
        assert!(deps.contains(&p2));
    }

    #[test]
    fn test_premise_dependency_merge() {
        let mut dep = PremiseDependency::new();

        let p1 = PremiseId(0);
        let p2 = PremiseId(1);
        let p3 = PremiseId(2);

        let d1 = dep.add_derived(vec![p1, p2]);
        let d2 = dep.add_derived(vec![p2, p3]);

        let merged = dep.merge_dependencies(&[d1, d2]);
        assert_eq!(merged.len(), 3);
        assert!(merged.contains(&p1));
        assert!(merged.contains(&p2));
        assert!(merged.contains(&p3));
    }

    #[test]
    fn test_premise_dependency_empty() {
        let dep = PremiseDependency::new();

        let merged = dep.merge_dependencies(&[0, 1, 2]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_premise_dependency_clear() {
        let mut dep = PremiseDependency::new();

        dep.add_derived(vec![PremiseId(0)]);
        dep.add_derived(vec![PremiseId(1)]);

        assert_eq!(dep.len(), 2);
        dep.clear();
        assert_eq!(dep.len(), 0);
        assert!(dep.is_empty());
    }

    #[test]
    fn test_premise_id_display() {
        let id = PremiseId(42);
        assert_eq!(format!("{}", id), "p42");
    }

    #[test]
    fn test_premise_id_ordering() {
        let p1 = PremiseId(1);
        let p2 = PremiseId(2);
        let p3 = PremiseId(3);

        assert!(p1 < p2);
        assert!(p2 < p3);
        assert!(p1 < p3);
    }

    #[test]
    fn test_active_premises_iterator() {
        let mut tracker = PremiseTracker::new();

        let p1 = tracker.add_assertion("(> x 5)");
        let p2 = tracker.add_assertion("(< x 10)");
        let _p3 = tracker.add_assertion("(= y 7)");

        tracker.deactivate(p2);

        let active: Vec<PremiseId> = tracker.active_premises().collect();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&p1));
        assert!(!active.contains(&p2));
    }
}
