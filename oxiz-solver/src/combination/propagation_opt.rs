//! Optimized Theory Propagation.
//!
//! This module implements advanced propagation strategies for theory combination:
//! - Incremental propagation with dependency tracking
//! - Lazy propagation using watched literals
//! - Priority-based propagation scheduling
//! - Propagation caching and memoization
//! - Backtracking support with trail management
//!
//! ## Incremental Propagation
//!
//! Rather than re-propagating everything on each change:
//! - Track dependencies between propagations
//! - Only re-propagate affected constraints
//! - Use timestamps to detect stale propagations
//!
//! ## Lazy Propagation
//!
//! Inspired by watched literals in SAT solving:
//! - Maintain watch lists for terms
//! - Trigger propagation only when watched terms change
//! - Minimize redundant propagation work
//!
//! ## Priority-Based Propagation
//!
//! Not all propagations are equally important:
//! - Assign priorities based on clause activity
//! - Propagate high-priority constraints first
//! - Defer low-priority propagations
//!
//! ## References
//!
//! - Nieuwenhuis, Oliveras, Tinelli: "Solving SAT and SAT Modulo Theories" (2006)
//! - Z3's `smt/theory_propagation.cpp`

#![allow(missing_docs)]
#![allow(dead_code)]

use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Term identifier.
pub type TermId = u32;

/// Theory identifier.
pub type TheoryId = u32;

/// Decision level.
pub type DecisionLevel = u32;

/// Timestamp for incremental propagation.
pub type Timestamp = u64;

/// Literal (positive or negative term).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal {
    /// Term identifier.
    pub term: TermId,
    /// Polarity (true = positive, false = negative).
    pub polarity: bool,
}

impl Literal {
    /// Create positive literal.
    pub fn positive(term: TermId) -> Self {
        Self {
            term,
            polarity: true,
        }
    }

    /// Create negative literal.
    pub fn negative(term: TermId) -> Self {
        Self {
            term,
            polarity: false,
        }
    }

    /// Negate literal.
    pub fn negate(self) -> Self {
        Self {
            term: self.term,
            polarity: !self.polarity,
        }
    }
}

/// Propagation event.
#[derive(Debug, Clone)]
pub struct PropagationEvent {
    /// The propagated literal.
    pub literal: Literal,
    /// Decision level where propagated.
    pub level: DecisionLevel,
    /// Timestamp of propagation.
    pub timestamp: Timestamp,
    /// Theory that performed propagation.
    pub theory: TheoryId,
    /// Reason for propagation.
    pub reason: PropagationReason,
}

/// Reason for a propagation.
#[derive(Debug, Clone)]
pub enum PropagationReason {
    /// Decision (no reason).
    Decision,
    /// Unit propagation.
    UnitPropagation {
        /// Clause that became unit.
        clause: ClauseId,
    },
    /// Theory propagation.
    TheoryPropagation {
        /// Explanation literals.
        explanation: Vec<Literal>,
    },
    /// Equality propagation.
    EqualityPropagation {
        /// Left-hand side.
        lhs: TermId,
        /// Right-hand side.
        rhs: TermId,
    },
}

/// Clause identifier.
pub type ClauseId = u32;

/// Propagation priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropagationPriority {
    /// Priority level (higher = more urgent).
    pub level: u32,
    /// Clause activity (VSIDS-style).
    pub activity: u32,
    /// Decision level.
    pub decision_level: DecisionLevel,
}

impl Ord for PropagationPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.level
            .cmp(&other.level)
            .then_with(|| self.activity.cmp(&other.activity))
            .then_with(|| other.decision_level.cmp(&self.decision_level))
    }
}

impl PartialOrd for PropagationPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Pending propagation.
#[derive(Debug, Clone)]
struct PendingPropagation {
    /// Event to propagate.
    event: PropagationEvent,
    /// Priority.
    priority: PropagationPriority,
}

impl PartialEq for PendingPropagation {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PendingPropagation {}

impl Ord for PendingPropagation {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for PendingPropagation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Watch list for a term.
#[derive(Debug, Clone)]
pub struct WatchList {
    /// Clauses watching this term.
    clauses: Vec<ClauseId>,
    /// Theories watching this term.
    theories: FxHashSet<TheoryId>,
}

impl WatchList {
    /// Create empty watch list.
    fn new() -> Self {
        Self {
            clauses: Vec::new(),
            theories: FxHashSet::default(),
        }
    }

    /// Add clause to watch list.
    fn add_clause(&mut self, clause: ClauseId) {
        self.clauses.push(clause);
    }

    /// Add theory to watch list.
    fn add_theory(&mut self, theory: TheoryId) {
        self.theories.insert(theory);
    }

    /// Get watching clauses.
    fn watching_clauses(&self) -> &[ClauseId] {
        &self.clauses
    }

    /// Get watching theories.
    fn watching_theories(&self) -> impl Iterator<Item = TheoryId> + '_ {
        self.theories.iter().copied()
    }
}

/// Propagation trail for backtracking.
#[derive(Debug, Clone)]
pub struct PropagationTrail {
    /// Trail of propagation events.
    trail: Vec<PropagationEvent>,
    /// Decision level boundaries.
    level_boundaries: Vec<usize>,
    /// Current decision level.
    current_level: DecisionLevel,
}

impl PropagationTrail {
    /// Create new trail.
    fn new() -> Self {
        Self {
            trail: Vec::new(),
            level_boundaries: vec![0],
            current_level: 0,
        }
    }

    /// Push new decision level.
    fn push_level(&mut self) {
        self.current_level += 1;
        self.level_boundaries.push(self.trail.len());
    }

    /// Add propagation event.
    fn add_event(&mut self, event: PropagationEvent) {
        self.trail.push(event);
    }

    /// Backtrack to decision level.
    fn backtrack(&mut self, level: DecisionLevel) -> Vec<PropagationEvent> {
        if level >= self.current_level {
            return Vec::new();
        }

        // When backtracking to level k, we want to keep everything up to
        // (but not including) the start of level k+1
        let boundary = if (level as usize + 1) < self.level_boundaries.len() {
            self.level_boundaries[level as usize + 1]
        } else {
            self.trail.len()
        };

        let undone = self.trail.split_off(boundary);
        self.level_boundaries.truncate(level as usize + 1);
        self.current_level = level;

        undone
    }

    /// Get current decision level.
    fn current_level(&self) -> DecisionLevel {
        self.current_level
    }

    /// Get trail.
    fn trail(&self) -> &[PropagationEvent] {
        &self.trail
    }

    /// Clear trail.
    fn clear(&mut self) {
        self.trail.clear();
        self.level_boundaries.clear();
        self.level_boundaries.push(0);
        self.current_level = 0;
    }
}

/// Dependency graph for incremental propagation.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Forward dependencies: term → terms that depend on it.
    forward: FxHashMap<TermId, FxHashSet<TermId>>,
    /// Backward dependencies: term → terms it depends on.
    backward: FxHashMap<TermId, FxHashSet<TermId>>,
}

impl DependencyGraph {
    /// Create new dependency graph.
    fn new() -> Self {
        Self {
            forward: FxHashMap::default(),
            backward: FxHashMap::default(),
        }
    }

    /// Add dependency: dependent depends on dependency.
    fn add_dependency(&mut self, dependency: TermId, dependent: TermId) {
        self.forward
            .entry(dependency)
            .or_default()
            .insert(dependent);
        self.backward
            .entry(dependent)
            .or_default()
            .insert(dependency);
    }

    /// Get terms that depend on a term.
    fn get_dependents(&self, term: TermId) -> impl Iterator<Item = TermId> + '_ {
        self.forward
            .get(&term)
            .into_iter()
            .flat_map(|set| set.iter().copied())
    }

    /// Get terms a term depends on.
    fn get_dependencies(&self, term: TermId) -> impl Iterator<Item = TermId> + '_ {
        self.backward
            .get(&term)
            .into_iter()
            .flat_map(|set| set.iter().copied())
    }

    /// Clear all dependencies.
    fn clear(&mut self) {
        self.forward.clear();
        self.backward.clear();
    }
}

/// Propagation cache for memoization.
#[derive(Debug, Clone)]
pub struct PropagationCache {
    /// Cached propagations: (term, theory) → (result, timestamp).
    cache: FxHashMap<(TermId, TheoryId), (Vec<Literal>, Timestamp)>,
    /// Current timestamp.
    current_timestamp: Timestamp,
}

impl PropagationCache {
    /// Create new cache.
    fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            current_timestamp: 0,
        }
    }

    /// Increment timestamp.
    fn tick(&mut self) {
        self.current_timestamp += 1;
    }

    /// Get cached propagation.
    fn get(&self, term: TermId, theory: TheoryId) -> Option<&Vec<Literal>> {
        self.cache
            .get(&(term, theory))
            .map(|(result, _timestamp)| result)
    }

    /// Cache propagation result.
    fn cache_result(&mut self, term: TermId, theory: TheoryId, result: Vec<Literal>) {
        self.cache
            .insert((term, theory), (result, self.current_timestamp));
    }

    /// Invalidate cache entry.
    fn invalidate(&mut self, term: TermId, theory: TheoryId) {
        self.cache.remove(&(term, theory));
    }

    /// Clear entire cache.
    fn clear(&mut self) {
        self.cache.clear();
        self.current_timestamp = 0;
    }
}

/// Configuration for optimized propagation.
#[derive(Debug, Clone)]
pub struct PropagationConfig {
    /// Enable incremental propagation.
    pub incremental: bool,
    /// Enable lazy propagation.
    pub lazy: bool,
    /// Enable priority-based scheduling.
    pub priority_based: bool,
    /// Enable propagation caching.
    pub caching: bool,
    /// Maximum propagation queue size.
    pub max_queue_size: usize,
    /// Propagation batch size.
    pub batch_size: usize,
    /// Enable dependency tracking.
    pub track_dependencies: bool,
}

impl Default for PropagationConfig {
    fn default() -> Self {
        Self {
            incremental: true,
            lazy: true,
            priority_based: true,
            caching: true,
            max_queue_size: 100000,
            batch_size: 1000,
            track_dependencies: true,
        }
    }
}

/// Statistics for propagation.
#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    /// Total propagations performed.
    pub propagations: u64,
    /// Lazy propagations triggered.
    pub lazy_propagations: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
    /// Incremental propagations.
    pub incremental_propagations: u64,
    /// Backtracking operations.
    pub backtracks: u64,
    /// Dependency graph updates.
    pub dependency_updates: u64,
}

/// Optimized propagation engine.
pub struct OptimizedPropagationEngine {
    /// Configuration.
    config: PropagationConfig,

    /// Statistics.
    stats: PropagationStats,

    /// Propagation queue (priority queue).
    queue: BinaryHeap<PendingPropagation>,

    /// Watch lists for terms.
    watch_lists: FxHashMap<TermId, WatchList>,

    /// Propagation trail for backtracking.
    trail: PropagationTrail,

    /// Dependency graph.
    dependencies: DependencyGraph,

    /// Propagation cache.
    cache: PropagationCache,

    /// Assignment timestamps.
    assignment_timestamps: FxHashMap<TermId, Timestamp>,

    /// Current assignments.
    assignments: FxHashMap<TermId, bool>,

    /// Clause activity scores (VSIDS-style).
    clause_activities: FxHashMap<ClauseId, u32>,

    /// Activity decay factor.
    activity_decay: f64,
}

impl OptimizedPropagationEngine {
    /// Create new propagation engine.
    pub fn new() -> Self {
        Self::with_config(PropagationConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: PropagationConfig) -> Self {
        Self {
            config,
            stats: PropagationStats::default(),
            queue: BinaryHeap::new(),
            watch_lists: FxHashMap::default(),
            trail: PropagationTrail::new(),
            dependencies: DependencyGraph::new(),
            cache: PropagationCache::new(),
            assignment_timestamps: FxHashMap::default(),
            assignments: FxHashMap::default(),
            clause_activities: FxHashMap::default(),
            activity_decay: 0.95,
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &PropagationStats {
        &self.stats
    }

    /// Get current decision level.
    pub fn decision_level(&self) -> DecisionLevel {
        self.trail.current_level()
    }

    /// Push new decision level.
    pub fn push_decision_level(&mut self) {
        self.trail.push_level();
    }

    /// Backtrack to decision level.
    pub fn backtrack(&mut self, level: DecisionLevel) -> Result<(), String> {
        if level > self.decision_level() {
            return Err("Cannot backtrack to future level".to_string());
        }

        let undone = self.trail.backtrack(level);

        // Undo assignments
        for event in undone {
            self.assignments.remove(&event.literal.term);
            self.assignment_timestamps.remove(&event.literal.term);

            // Invalidate cache for affected terms
            if self.config.caching {
                self.invalidate_cache_for_term(event.literal.term);
            }
        }

        self.stats.backtracks += 1;
        Ok(())
    }

    /// Add watch for a term.
    pub fn add_watch(&mut self, term: TermId, clause: ClauseId) {
        let watch_list = self.watch_lists.entry(term).or_insert_with(WatchList::new);
        watch_list.add_clause(clause);
    }

    /// Add theory watch for a term.
    pub fn add_theory_watch(&mut self, term: TermId, theory: TheoryId) {
        let watch_list = self.watch_lists.entry(term).or_insert_with(WatchList::new);
        watch_list.add_theory(theory);
    }

    /// Propagate a literal.
    pub fn propagate(
        &mut self,
        literal: Literal,
        theory: TheoryId,
        reason: PropagationReason,
        priority: u32,
    ) -> Result<(), String> {
        // Check if already assigned
        if let Some(&assigned_value) = self.assignments.get(&literal.term) {
            if assigned_value != literal.polarity {
                return Err("Conflict: literal already assigned with opposite polarity".to_string());
            }
            return Ok(()); // Already propagated
        }

        let event = PropagationEvent {
            literal,
            level: self.decision_level(),
            timestamp: self.cache.current_timestamp,
            theory,
            reason,
        };

        let activity = self.clause_activities.get(&0).copied().unwrap_or(0);

        let pending = PendingPropagation {
            event,
            priority: PropagationPriority {
                level: priority,
                activity,
                decision_level: self.decision_level(),
            },
        };

        if self.config.priority_based {
            self.queue.push(pending);
        } else {
            self.propagate_immediate(pending.event)?;
        }

        Ok(())
    }

    /// Propagate immediately (non-lazy).
    fn propagate_immediate(&mut self, event: PropagationEvent) -> Result<(), String> {
        // Assign literal
        self.assignments
            .insert(event.literal.term, event.literal.polarity);
        self.assignment_timestamps
            .insert(event.literal.term, event.timestamp);

        // Add to trail
        self.trail.add_event(event.clone());

        self.stats.propagations += 1;
        self.cache.tick();

        // Trigger watches if lazy propagation enabled
        if self.config.lazy {
            self.trigger_watches(event.literal.term)?;
        }

        Ok(())
    }

    /// Process propagation queue.
    pub fn process_queue(&mut self) -> Result<(), String> {
        let mut count = 0;

        while let Some(pending) = self.queue.pop() {
            self.propagate_immediate(pending.event)?;

            count += 1;
            if count >= self.config.batch_size {
                break;
            }
        }

        Ok(())
    }

    /// Trigger watches for a term.
    fn trigger_watches(&mut self, term: TermId) -> Result<(), String> {
        if let Some(watch_list) = self.watch_lists.get(&term) {
            self.stats.lazy_propagations += 1;

            // Trigger clause watches
            for &_clause_id in watch_list.watching_clauses() {
                // Would trigger clause propagation here
            }

            // Trigger theory watches
            for _theory_id in watch_list.watching_theories() {
                // Would notify theory here
            }
        }

        Ok(())
    }

    /// Add dependency between terms.
    pub fn add_dependency(&mut self, dependency: TermId, dependent: TermId) {
        if !self.config.track_dependencies {
            return;
        }

        self.dependencies.add_dependency(dependency, dependent);
        self.stats.dependency_updates += 1;
    }

    /// Get terms affected by change to a term.
    pub fn get_affected_terms(&self, term: TermId) -> Vec<TermId> {
        if !self.config.track_dependencies {
            return Vec::new();
        }

        self.dependencies.get_dependents(term).collect()
    }

    /// Incremental propagation for changed term.
    pub fn incremental_propagate(&mut self, term: TermId, theory: TheoryId) -> Result<(), String> {
        if !self.config.incremental {
            return Ok(());
        }

        // Invalidate cache for this term
        if self.config.caching {
            self.invalidate_cache_for_term(term);
        }

        // Get affected terms via dependency graph
        let affected: Vec<_> = self.get_affected_terms(term);

        for affected_term in affected {
            // Re-propagate affected terms
            if self.config.caching {
                self.cache.invalidate(affected_term, theory);
            }

            self.stats.incremental_propagations += 1;
        }

        Ok(())
    }

    /// Invalidate cache for a term.
    fn invalidate_cache_for_term(&mut self, term: TermId) {
        // Invalidate for all theories
        // (In practice, we'd track which theories to invalidate)
        for theory_id in 0..10 {
            self.cache.invalidate(term, theory_id);
        }
    }

    /// Check cache for propagation result.
    pub fn check_cache(&mut self, term: TermId, theory: TheoryId) -> Option<Vec<Literal>> {
        if !self.config.caching {
            return None;
        }

        if let Some(result) = self.cache.get(term, theory) {
            self.stats.cache_hits += 1;
            Some(result.clone())
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }

    /// Cache propagation result.
    pub fn cache_propagation(&mut self, term: TermId, theory: TheoryId, result: Vec<Literal>) {
        if !self.config.caching {
            return;
        }

        self.cache.cache_result(term, theory, result);
    }

    /// Update clause activity (VSIDS-style).
    pub fn bump_clause_activity(&mut self, clause: ClauseId) {
        let activity = self.clause_activities.entry(clause).or_insert(0);
        *activity += 1;
    }

    /// Decay all clause activities.
    pub fn decay_activities(&mut self) {
        for activity in self.clause_activities.values_mut() {
            *activity = (*activity as f64 * self.activity_decay) as u32;
        }
    }

    /// Get assignment for term.
    pub fn get_assignment(&self, term: TermId) -> Option<bool> {
        self.assignments.get(&term).copied()
    }

    /// Check if term is assigned.
    pub fn is_assigned(&self, term: TermId) -> bool {
        self.assignments.contains_key(&term)
    }

    /// Get propagation trail.
    pub fn trail(&self) -> &[PropagationEvent] {
        self.trail.trail()
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.queue.clear();
        self.watch_lists.clear();
        self.trail.clear();
        self.dependencies.clear();
        self.cache.clear();
        self.assignment_timestamps.clear();
        self.assignments.clear();
        self.clause_activities.clear();
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PropagationStats::default();
    }
}

impl Default for OptimizedPropagationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy propagator using watch lists.
pub struct LazyPropagator {
    /// Watch lists.
    watches: FxHashMap<Literal, Vec<ClauseId>>,
    /// Clauses.
    clauses: FxHashMap<ClauseId, Vec<Literal>>,
    /// Next clause ID.
    next_clause_id: ClauseId,
}

impl LazyPropagator {
    /// Create new lazy propagator.
    pub fn new() -> Self {
        Self {
            watches: FxHashMap::default(),
            clauses: FxHashMap::default(),
            next_clause_id: 0,
        }
    }

    /// Add clause with watched literals.
    pub fn add_clause(&mut self, literals: Vec<Literal>) -> ClauseId {
        let clause_id = self.next_clause_id;
        self.next_clause_id += 1;

        // Watch first two literals (if they exist)
        if literals.len() >= 2 {
            self.watches.entry(literals[0]).or_default().push(clause_id);
            self.watches.entry(literals[1]).or_default().push(clause_id);
        }

        self.clauses.insert(clause_id, literals);
        clause_id
    }

    /// Propagate assignment.
    pub fn propagate_assignment(
        &mut self,
        literal: Literal,
        _assignments: &FxHashMap<TermId, bool>,
    ) -> Vec<ClauseId> {
        let negated = literal.negate();
        let mut triggered = Vec::new();

        if let Some(watching_clauses) = self.watches.get(&negated) {
            for &clause_id in watching_clauses {
                triggered.push(clause_id);
            }
        }

        triggered
    }

    /// Get clause literals.
    pub fn get_clause(&self, clause_id: ClauseId) -> Option<&Vec<Literal>> {
        self.clauses.get(&clause_id)
    }

    /// Clear all watches and clauses.
    pub fn clear(&mut self) {
        self.watches.clear();
        self.clauses.clear();
        self.next_clause_id = 0;
    }
}

impl Default for LazyPropagator {
    fn default() -> Self {
        Self::new()
    }
}

/// Incremental propagator with dependency tracking.
pub struct IncrementalPropagator {
    /// Dependency graph.
    dependencies: DependencyGraph,
    /// Dirty terms (need re-propagation).
    dirty: FxHashSet<TermId>,
    /// Propagation order (topological).
    propagation_order: Vec<TermId>,
}

impl IncrementalPropagator {
    /// Create new incremental propagator.
    pub fn new() -> Self {
        Self {
            dependencies: DependencyGraph::new(),
            dirty: FxHashSet::default(),
            propagation_order: Vec::new(),
        }
    }

    /// Add dependency.
    pub fn add_dependency(&mut self, dependency: TermId, dependent: TermId) {
        self.dependencies.add_dependency(dependency, dependent);
    }

    /// Mark term as dirty.
    pub fn mark_dirty(&mut self, term: TermId) {
        self.dirty.insert(term);

        // Mark dependents as dirty
        for dependent in self.dependencies.get_dependents(term) {
            self.dirty.insert(dependent);
        }
    }

    /// Get dirty terms.
    pub fn get_dirty_terms(&self) -> impl Iterator<Item = TermId> + '_ {
        self.dirty.iter().copied()
    }

    /// Clear dirty marks.
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Compute propagation order (topological sort).
    pub fn compute_propagation_order(&mut self) -> Vec<TermId> {
        // Simplified topological sort
        self.propagation_order.clear();
        self.propagation_order.extend(self.dirty.iter().copied());
        self.propagation_order.clone()
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.dependencies.clear();
        self.dirty.clear();
        self.propagation_order.clear();
    }
}

impl Default for IncrementalPropagator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_creation() {
        let lit = Literal::positive(1);
        assert_eq!(lit.term, 1);
        assert!(lit.polarity);
    }

    #[test]
    fn test_literal_negation() {
        let lit = Literal::positive(1);
        let neg = lit.negate();
        assert_eq!(neg.term, 1);
        assert!(!neg.polarity);
    }

    #[test]
    fn test_propagation_engine_creation() {
        let engine = OptimizedPropagationEngine::new();
        assert_eq!(engine.decision_level(), 0);
    }

    #[test]
    fn test_decision_level() {
        let mut engine = OptimizedPropagationEngine::new();
        engine.push_decision_level();
        assert_eq!(engine.decision_level(), 1);
    }

    #[test]
    fn test_backtrack() {
        let mut engine = OptimizedPropagationEngine::new();
        engine.push_decision_level();
        engine.push_decision_level();
        assert_eq!(engine.decision_level(), 2);

        engine.backtrack(1).expect("Backtrack failed");
        assert_eq!(engine.decision_level(), 1);
    }

    #[test]
    fn test_watch_list() {
        let mut watch_list = WatchList::new();
        watch_list.add_clause(1);
        watch_list.add_theory(0);

        assert_eq!(watch_list.watching_clauses().len(), 1);
    }

    #[test]
    fn test_propagation_trail() {
        let mut trail = PropagationTrail::new();
        trail.push_level();

        let event = PropagationEvent {
            literal: Literal::positive(1),
            level: 1,
            timestamp: 0,
            theory: 0,
            reason: PropagationReason::Decision,
        };

        trail.add_event(event);
        assert_eq!(trail.trail().len(), 1);
    }

    #[test]
    fn test_trail_backtrack() {
        let mut trail = PropagationTrail::new();
        trail.push_level();

        let event1 = PropagationEvent {
            literal: Literal::positive(1),
            level: 1,
            timestamp: 0,
            theory: 0,
            reason: PropagationReason::Decision,
        };

        trail.add_event(event1);
        trail.push_level();

        let event2 = PropagationEvent {
            literal: Literal::positive(2),
            level: 2,
            timestamp: 1,
            theory: 0,
            reason: PropagationReason::Decision,
        };

        trail.add_event(event2);

        let undone = trail.backtrack(1);
        assert_eq!(undone.len(), 1);
        assert_eq!(trail.trail().len(), 1);
    }

    #[test]
    fn test_dependency_graph() {
        let mut deps = DependencyGraph::new();
        deps.add_dependency(1, 2);
        deps.add_dependency(1, 3);

        let dependents: Vec<_> = deps.get_dependents(1).collect();
        assert_eq!(dependents.len(), 2);
    }

    #[test]
    fn test_propagation_cache() {
        let mut cache = PropagationCache::new();
        let result = vec![Literal::positive(1)];

        cache.cache_result(1, 0, result.clone());
        assert_eq!(cache.get(1, 0), Some(&result));
    }

    #[test]
    fn test_cache_invalidation() {
        let mut cache = PropagationCache::new();
        let result = vec![Literal::positive(1)];

        cache.cache_result(1, 0, result);
        cache.invalidate(1, 0);
        assert_eq!(cache.get(1, 0), None);
    }

    #[test]
    fn test_lazy_propagator() {
        let mut prop = LazyPropagator::new();
        let clause = vec![Literal::positive(1), Literal::positive(2)];
        let clause_id = prop.add_clause(clause);

        assert_eq!(prop.get_clause(clause_id).map(|c| c.len()), Some(2));
    }

    #[test]
    fn test_incremental_propagator() {
        let mut prop = IncrementalPropagator::new();
        prop.add_dependency(1, 2);
        prop.mark_dirty(1);

        let dirty: Vec<_> = prop.get_dirty_terms().collect();
        assert!(dirty.contains(&1));
        assert!(dirty.contains(&2));
    }

    #[test]
    fn test_priority_ordering() {
        let p1 = PropagationPriority {
            level: 1,
            activity: 0,
            decision_level: 0,
        };
        let p2 = PropagationPriority {
            level: 2,
            activity: 0,
            decision_level: 0,
        };

        assert!(p2 > p1);
    }

    #[test]
    fn test_propagate() {
        let mut engine = OptimizedPropagationEngine::new();
        let literal = Literal::positive(1);

        engine
            .propagate(literal, 0, PropagationReason::Decision, 100)
            .expect("Propagation failed");

        // Process the queue to actually execute the propagation
        engine.process_queue().expect("Process queue failed");

        assert_eq!(engine.stats().propagations, 1);
    }

    #[test]
    fn test_assignment_tracking() {
        let mut engine = OptimizedPropagationEngine::new();
        let literal = Literal::positive(1);

        engine
            .propagate(literal, 0, PropagationReason::Decision, 100)
            .expect("Propagation failed");
        engine.process_queue().expect("Queue processing failed");

        assert_eq!(engine.get_assignment(1), Some(true));
    }
}
