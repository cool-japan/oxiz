//! Propagation Pipeline for Theory Combination
#![allow(missing_docs)] // Under development
//!
//! This module implements a multi-level propagation framework for CDCL(T):
//! - Unit propagation (Boolean level)
//! - Equality propagation (congruence closure)
//! - Theory-specific propagation (arithmetic, bit-vectors, etc.)
//! - Priority-based scheduling of propagations

use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Placeholder term identifier
pub type TermId = usize;

/// Propagation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PropagationLevel {
    /// Boolean unit propagation
    Boolean,
    /// Equality propagation
    Equality,
    /// Theory-specific propagation
    Theory(TheoryId),
}

/// Theory identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TheoryId {
    Arithmetic,
    BitVector,
    Array,
    Datatype,
    String,
    Uninterpreted,
}

/// Propagation action
#[derive(Debug, Clone)]
pub struct Propagation {
    /// The literal being propagated
    pub literal: TermId,
    /// The level at which this propagation occurs
    pub level: PropagationLevel,
    /// Reason for this propagation
    pub reason: PropagationReason,
    /// Priority (higher = more important)
    pub priority: u32,
}

/// Reason for a propagation
#[derive(Debug, Clone)]
pub enum PropagationReason {
    /// Unit clause
    UnitClause(TermId),
    /// Binary clause
    BinaryClause(TermId, TermId),
    /// General clause
    Clause(Vec<TermId>),
    /// Equality propagation
    Equality { lhs: TermId, rhs: TermId },
    /// Theory propagation
    Theory { explanation: Vec<TermId> },
}

impl Ord for Propagation {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for Propagation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Propagation {
    fn eq(&self, other: &Self) -> bool {
        self.literal == other.literal && self.priority == other.priority
    }
}

impl Eq for Propagation {}

/// Statistics for propagation pipeline
#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    pub total_propagations: u64,
    pub boolean_propagations: u64,
    pub equality_propagations: u64,
    pub theory_propagations: FxHashMap<TheoryId, u64>,
    pub conflicts_detected: u64,
    pub priority_queue_size_max: usize,
}

/// Configuration for propagation pipeline
#[derive(Debug, Clone)]
pub struct PropagationConfig {
    /// Enable priority-based scheduling
    pub use_priority_queue: bool,
    /// Maximum propagations per iteration
    pub max_propagations_per_iter: usize,
    /// Enable lazy propagation (batch updates)
    pub enable_lazy_propagation: bool,
    /// Theory propagation priorities
    pub theory_priorities: FxHashMap<TheoryId, u32>,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        let mut priorities = FxHashMap::default();
        priorities.insert(TheoryId::Arithmetic, 10);
        priorities.insert(TheoryId::BitVector, 9);
        priorities.insert(TheoryId::Array, 8);
        priorities.insert(TheoryId::Datatype, 7);
        priorities.insert(TheoryId::String, 6);
        priorities.insert(TheoryId::Uninterpreted, 5);

        Self {
            use_priority_queue: true,
            max_propagations_per_iter: 1000,
            enable_lazy_propagation: true,
            theory_priorities: priorities,
        }
    }
}

/// Propagation pipeline
pub struct PropagationPipeline {
    config: PropagationConfig,
    stats: PropagationStats,
    /// Priority queue for pending propagations
    pending: BinaryHeap<Propagation>,
    /// Set of propagated literals
    propagated: FxHashSet<TermId>,
    /// Current propagation level
    current_level: usize,
    /// Trail of propagations (for backtracking)
    trail: Vec<Propagation>,
}

impl PropagationPipeline {
    /// Create a new propagation pipeline
    pub fn new(config: PropagationConfig) -> Self {
        Self {
            config,
            stats: PropagationStats::default(),
            pending: BinaryHeap::new(),
            propagated: FxHashSet::default(),
            current_level: 0,
            trail: Vec::new(),
        }
    }

    /// Add a propagation to the pipeline
    pub fn add_propagation(&mut self, propagation: Propagation) {
        // Check if already propagated
        if self.propagated.contains(&propagation.literal) {
            return;
        }

        self.pending.push(propagation);

        // Update max queue size
        if self.pending.len() > self.stats.priority_queue_size_max {
            self.stats.priority_queue_size_max = self.pending.len();
        }
    }

    /// Process all pending propagations
    pub fn propagate(&mut self) -> Result<(), Vec<TermId>> {
        let mut iterations = 0;

        while let Some(propagation) = self.pending.pop() {
            if iterations >= self.config.max_propagations_per_iter {
                // Put it back and stop
                self.pending.push(propagation);
                break;
            }

            // Check if already propagated
            if self.propagated.contains(&propagation.literal) {
                continue;
            }

            // Perform the propagation
            self.perform_propagation(propagation)?;

            iterations += 1;
        }

        Ok(())
    }

    /// Perform a single propagation
    fn perform_propagation(&mut self, propagation: Propagation) -> Result<(), Vec<TermId>> {
        self.stats.total_propagations += 1;

        // Update statistics based on level
        match propagation.level {
            PropagationLevel::Boolean => {
                self.stats.boolean_propagations += 1;
            }
            PropagationLevel::Equality => {
                self.stats.equality_propagations += 1;
            }
            PropagationLevel::Theory(theory_id) => {
                *self.stats.theory_propagations.entry(theory_id).or_insert(0) += 1;
            }
        }

        // Check for conflicts
        if let Some(conflict) = self.check_conflict(propagation.literal)? {
            self.stats.conflicts_detected += 1;
            return Err(conflict);
        }

        // Mark as propagated and add to trail
        self.propagated.insert(propagation.literal);
        self.trail.push(propagation);

        Ok(())
    }

    /// Check if a literal conflicts with current assignment
    fn check_conflict(&self, literal: TermId) -> Result<Option<Vec<TermId>>, Vec<TermId>> {
        // Placeholder: check if negation is already propagated
        let negated = self.negate(literal);

        if self.propagated.contains(&negated) {
            // Conflict detected
            let conflict_clause = vec![literal, negated];
            return Ok(Some(conflict_clause));
        }

        Ok(None)
    }

    /// Negate a literal
    fn negate(&self, literal: TermId) -> TermId {
        // Placeholder: simple negation (even -> odd, odd -> even)
        if literal.is_multiple_of(2) {
            literal + 1
        } else {
            literal - 1
        }
    }

    /// Backtrack to a specific level
    pub fn backtrack(&mut self, target_level: usize) -> Result<(), String> {
        // Remove propagations after target level
        let target_trail_size = self.get_trail_size_at_level(target_level);

        while self.trail.len() > target_trail_size {
            if let Some(prop) = self.trail.pop() {
                self.propagated.remove(&prop.literal);
            }
        }

        // Clear pending queue (all pending propagations are invalidated)
        self.pending.clear();

        self.current_level = target_level;

        Ok(())
    }

    /// Get the trail size at a specific level
    fn get_trail_size_at_level(&self, level: usize) -> usize {
        // Placeholder: would track level boundaries in trail
        if level == 0 { 0 } else { self.trail.len() }
    }

    /// Increment decision level
    pub fn increment_level(&mut self) {
        self.current_level += 1;
    }

    /// Get current decision level
    pub fn current_level(&self) -> usize {
        self.current_level
    }

    /// Check if a literal is propagated
    pub fn is_propagated(&self, literal: TermId) -> bool {
        self.propagated.contains(&literal)
    }

    /// Get the reason for a propagation
    pub fn get_reason(&self, literal: TermId) -> Option<&PropagationReason> {
        self.trail
            .iter()
            .find(|p| p.literal == literal)
            .map(|p| &p.reason)
    }

    /// Create a unit propagation
    pub fn mk_unit_propagation(&self, literal: TermId, clause: TermId) -> Propagation {
        Propagation {
            literal,
            level: PropagationLevel::Boolean,
            reason: PropagationReason::UnitClause(clause),
            priority: 100, // Highest priority
        }
    }

    /// Create an equality propagation
    pub fn mk_equality_propagation(
        &self,
        literal: TermId,
        lhs: TermId,
        rhs: TermId,
    ) -> Propagation {
        Propagation {
            literal,
            level: PropagationLevel::Equality,
            reason: PropagationReason::Equality { lhs, rhs },
            priority: 50,
        }
    }

    /// Create a theory propagation
    pub fn mk_theory_propagation(
        &self,
        literal: TermId,
        theory: TheoryId,
        explanation: Vec<TermId>,
    ) -> Propagation {
        let priority = self
            .config
            .theory_priorities
            .get(&theory)
            .copied()
            .unwrap_or(10);

        Propagation {
            literal,
            level: PropagationLevel::Theory(theory),
            reason: PropagationReason::Theory { explanation },
            priority,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &PropagationStats {
        &self.stats
    }

    /// Reset pipeline
    pub fn reset(&mut self) {
        self.pending.clear();
        self.propagated.clear();
        self.trail.clear();
        self.current_level = 0;
    }
}

/// Propagation watcher for efficient unit propagation
pub struct PropagationWatcher {
    /// Watch lists: literal -> clauses watching it
    watch_lists: FxHashMap<TermId, Vec<TermId>>,
    /// Clause storage
    clauses: FxHashMap<TermId, Vec<TermId>>,
}

impl PropagationWatcher {
    /// Create a new propagation watcher
    pub fn new() -> Self {
        Self {
            watch_lists: FxHashMap::default(),
            clauses: FxHashMap::default(),
        }
    }

    /// Add a clause to watch
    pub fn add_clause(&mut self, clause_id: TermId, literals: Vec<TermId>) -> Result<(), String> {
        if literals.len() < 2 {
            return Err("Clause must have at least 2 literals for watching".to_string());
        }

        // Watch first two literals
        self.watch_lists
            .entry(literals[0])
            .or_default()
            .push(clause_id);

        self.watch_lists
            .entry(literals[1])
            .or_default()
            .push(clause_id);

        self.clauses.insert(clause_id, literals);

        Ok(())
    }

    /// Update watches after a literal is assigned
    pub fn update_watches(
        &mut self,
        assigned_literal: TermId,
        pipeline: &mut PropagationPipeline,
    ) -> Result<(), Vec<TermId>> {
        let clause_ids: Vec<_> = self
            .watch_lists
            .get(&assigned_literal)
            .cloned()
            .unwrap_or_default();

        for clause_id in clause_ids {
            let clause = match self.clauses.get(&clause_id) {
                Some(c) => c.clone(),
                None => continue, // Clause was removed, skip
            };

            // Find a new literal to watch
            if let Some(new_watch) = self.find_new_watch(&clause, assigned_literal, pipeline) {
                // Update watch
                if let Some(watch_list) = self.watch_lists.get_mut(&assigned_literal) {
                    watch_list.retain(|&id| id != clause_id);
                }

                self.watch_lists
                    .entry(new_watch)
                    .or_default()
                    .push(clause_id);
            } else {
                // Check if this is a unit clause
                if let Some(unit_literal) = self.find_unit_literal(&clause, pipeline) {
                    let prop = pipeline.mk_unit_propagation(unit_literal, clause_id);
                    pipeline.add_propagation(prop);
                } else {
                    // Conflict: all literals are false
                    return Err(clause);
                }
            }
        }

        Ok(())
    }

    /// Find a new literal to watch
    fn find_new_watch(
        &self,
        clause: &[TermId],
        old_watch: TermId,
        pipeline: &PropagationPipeline,
    ) -> Option<TermId> {
        clause
            .iter()
            .find(|&&lit| lit != old_watch && !pipeline.is_propagated(pipeline.negate(lit)))
            .copied()
    }

    /// Find a unit literal in the clause
    fn find_unit_literal(
        &self,
        clause: &[TermId],
        pipeline: &PropagationPipeline,
    ) -> Option<TermId> {
        let mut unassigned = None;
        let mut unassigned_count = 0;

        for &lit in clause {
            if !pipeline.is_propagated(lit) && !pipeline.is_propagated(pipeline.negate(lit)) {
                unassigned = Some(lit);
                unassigned_count += 1;

                if unassigned_count > 1 {
                    return None;
                }
            }
        }

        if unassigned_count == 1 {
            unassigned
        } else {
            None
        }
    }
}

impl Default for PropagationWatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = PropagationConfig::default();
        let pipeline = PropagationPipeline::new(config);
        assert_eq!(pipeline.current_level(), 0);
        assert_eq!(pipeline.stats.total_propagations, 0);
    }

    #[test]
    fn test_add_propagation() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        let prop = Propagation {
            literal: 1,
            level: PropagationLevel::Boolean,
            reason: PropagationReason::UnitClause(42),
            priority: 100,
        };

        pipeline.add_propagation(prop);
        assert_eq!(pipeline.pending.len(), 1);
    }

    #[test]
    fn test_unit_propagation() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        let prop = pipeline.mk_unit_propagation(1, 42);
        pipeline.add_propagation(prop);

        let result = pipeline.propagate();
        assert!(result.is_ok());
        assert_eq!(pipeline.stats.boolean_propagations, 1);
        assert!(pipeline.is_propagated(1));
    }

    #[test]
    fn test_conflict_detection() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        // Propagate literal 2
        let prop1 = pipeline.mk_unit_propagation(2, 42);
        pipeline.add_propagation(prop1);
        let _ = pipeline.propagate();

        // Try to propagate negation (3 = NOT 2)
        let prop2 = pipeline.mk_unit_propagation(3, 43);
        pipeline.add_propagation(prop2);

        let result = pipeline.propagate();
        assert!(result.is_err());
        assert_eq!(pipeline.stats.conflicts_detected, 1);
    }

    #[test]
    fn test_equality_propagation() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        let prop = pipeline.mk_equality_propagation(10, 5, 5);
        pipeline.add_propagation(prop);

        let result = pipeline.propagate();
        assert!(result.is_ok());
        assert_eq!(pipeline.stats.equality_propagations, 1);
    }

    #[test]
    fn test_theory_propagation() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        let prop = pipeline.mk_theory_propagation(20, TheoryId::Arithmetic, vec![1, 2, 3]);
        pipeline.add_propagation(prop);

        let result = pipeline.propagate();
        assert!(result.is_ok());
        assert_eq!(
            *pipeline
                .stats
                .theory_propagations
                .get(&TheoryId::Arithmetic)
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_priority_ordering() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        let low_priority = Propagation {
            literal: 1,
            level: PropagationLevel::Boolean,
            reason: PropagationReason::UnitClause(1),
            priority: 10,
        };

        let high_priority = Propagation {
            literal: 2,
            level: PropagationLevel::Boolean,
            reason: PropagationReason::UnitClause(2),
            priority: 100,
        };

        pipeline.add_propagation(low_priority);
        pipeline.add_propagation(high_priority);

        // High priority should be processed first
        let _ = pipeline.propagate();
        assert!(pipeline.is_propagated(2));
    }

    #[test]
    fn test_backtrack() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        pipeline.increment_level();
        let prop = pipeline.mk_unit_propagation(5, 50);
        pipeline.add_propagation(prop);
        let _ = pipeline.propagate();

        assert!(pipeline.is_propagated(5));

        // Backtrack to level 0
        pipeline.backtrack(0).unwrap();
        assert!(!pipeline.is_propagated(5));
        assert_eq!(pipeline.current_level(), 0);
    }

    #[test]
    fn test_watcher_creation() {
        let watcher = PropagationWatcher::new();
        assert_eq!(watcher.watch_lists.len(), 0);
    }

    #[test]
    fn test_add_clause_to_watcher() {
        let mut watcher = PropagationWatcher::new();

        let result = watcher.add_clause(1, vec![10, 20, 30]);
        assert!(result.is_ok());
        assert!(watcher.watch_lists.contains_key(&10));
        assert!(watcher.watch_lists.contains_key(&20));
    }

    #[test]
    fn test_add_short_clause_fails() {
        let mut watcher = PropagationWatcher::new();

        let result = watcher.add_clause(1, vec![10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let config = PropagationConfig::default();
        let mut pipeline = PropagationPipeline::new(config);

        let prop = pipeline.mk_unit_propagation(1, 42);
        pipeline.add_propagation(prop);
        let _ = pipeline.propagate();

        pipeline.reset();
        assert!(!pipeline.is_propagated(1));
        assert_eq!(pipeline.current_level(), 0);
    }
}
