//! Interface Equality Management for Theory Combination.
//!
//! This module manages interface equalities in Nelson-Oppen combination:
//! - Minimal interface equality generation
//! - Equality generation strategies
//! - Interface equality scheduling
//! - Equality minimization algorithms
//!
//! ## Interface Equalities
//!
//! Interface equalities are equalities between shared terms that must be
//! communicated between theories. The goal is to generate a **minimal**
//! set of equalities that allows theories to infer all relevant equalities.
//!
//! ## Generation Strategies
//!
//! - **Eager**: Generate all possible interface equalities upfront
//! - **Lazy**: Generate equalities on-demand
//! - **Minimal**: Generate only necessary equalities (star topology)
//! - **Incremental**: Add equalities incrementally as needed
//!
//! ## Star Topology
//!
//! For a set of equivalent terms {t1, t2, ..., tn}, we can use a "star"
//! topology: pick one term as the representative and only share equalities
//! t1 = rep, t2 = rep, ..., tn = rep. This requires O(n) equalities instead
//! of O(n²).
//!
//! ## References
//!
//! - Nelson & Oppen (1979): "Simplification by Cooperating Decision Procedures"
//! - Shostak (1984): "Deciding Combinations of Theories"
//! - Z3's `smt/theory_combine.cpp`

use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Term identifier.
pub type TermId = u32;

/// Theory identifier.
pub type TheoryId = u32;

/// Decision level.
pub type DecisionLevel = u32;

/// Equality between two terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Equality {
    /// Left-hand side.
    pub lhs: TermId,
    /// Right-hand side.
    pub rhs: TermId,
}

impl Equality {
    /// Create new equality (normalized).
    pub fn new(lhs: TermId, rhs: TermId) -> Self {
        if lhs <= rhs {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }
}

/// Equality generation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationStrategy {
    /// Generate all equalities eagerly.
    Eager,
    /// Generate equalities lazily on-demand.
    Lazy,
    /// Generate minimal set (star topology).
    Minimal,
    /// Incremental generation.
    Incremental,
    /// Adaptive strategy based on heuristics.
    Adaptive,
}

/// Priority for interface equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EqualityPriority {
    /// Priority level (higher = more important).
    pub level: u32,
    /// Relevancy score.
    pub relevancy: u32,
    /// Decision level.
    pub decision_level: DecisionLevel,
}

impl Ord for EqualityPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.level
            .cmp(&other.level)
            .then_with(|| self.relevancy.cmp(&other.relevancy))
            .then_with(|| other.decision_level.cmp(&self.decision_level))
    }
}

impl PartialOrd for EqualityPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Interface equality with metadata.
#[derive(Debug, Clone)]
pub struct InterfaceEquality {
    /// The equality.
    pub equality: Equality,
    /// Theories involved.
    pub theories: FxHashSet<TheoryId>,
    /// Priority.
    pub priority: EqualityPriority,
    /// Is this equality necessary?
    pub is_necessary: bool,
    /// Generation timestamp.
    pub timestamp: u64,
}

impl PartialEq for InterfaceEquality {
    fn eq(&self, other: &Self) -> bool {
        self.equality == other.equality
    }
}

impl Eq for InterfaceEquality {}

impl Ord for InterfaceEquality {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

impl PartialOrd for InterfaceEquality {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Equivalence class for interface terms.
#[derive(Debug, Clone)]
pub struct InterfaceEClass {
    /// Representative term.
    pub representative: TermId,
    /// All terms in the class.
    pub members: FxHashSet<TermId>,
    /// Theories using terms in this class.
    pub theories: FxHashSet<TheoryId>,
    /// Generation strategy for this class.
    pub strategy: GenerationStrategy,
}

impl InterfaceEClass {
    /// Create new equivalence class.
    fn new(representative: TermId, theory: TheoryId) -> Self {
        let mut members = FxHashSet::default();
        members.insert(representative);

        let mut theories = FxHashSet::default();
        theories.insert(theory);

        Self {
            representative,
            members,
            theories,
            strategy: GenerationStrategy::Minimal,
        }
    }

    /// Add term to class.
    fn add_term(&mut self, term: TermId, theory: TheoryId) {
        self.members.insert(term);
        self.theories.insert(theory);
    }

    /// Merge another class into this one.
    fn merge(&mut self, other: &InterfaceEClass) {
        for &term in &other.members {
            self.members.insert(term);
        }
        for &theory in &other.theories {
            self.theories.insert(theory);
        }
    }

    /// Check if class is shared between multiple theories.
    fn is_shared(&self) -> bool {
        self.theories.len() > 1
    }

    /// Generate equalities for this class using current strategy.
    fn generate_equalities(
        &self,
        timestamp: u64,
        decision_level: DecisionLevel,
    ) -> Vec<InterfaceEquality> {
        match self.strategy {
            GenerationStrategy::Eager => self.generate_eager(timestamp, decision_level),
            GenerationStrategy::Lazy => Vec::new(), // Generated on-demand
            GenerationStrategy::Minimal => self.generate_minimal(timestamp, decision_level),
            GenerationStrategy::Incremental => self.generate_incremental(timestamp, decision_level),
            GenerationStrategy::Adaptive => self.generate_adaptive(timestamp, decision_level),
        }
    }

    /// Generate all pairwise equalities (eager).
    fn generate_eager(
        &self,
        timestamp: u64,
        decision_level: DecisionLevel,
    ) -> Vec<InterfaceEquality> {
        let mut equalities = Vec::new();
        let members: Vec<_> = self.members.iter().copied().collect();

        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                equalities.push(InterfaceEquality {
                    equality: Equality::new(members[i], members[j]),
                    theories: self.theories.clone(),
                    priority: EqualityPriority {
                        level: 100,
                        relevancy: 50,
                        decision_level,
                    },
                    is_necessary: false,
                    timestamp,
                });
            }
        }

        equalities
    }

    /// Generate minimal equalities (star topology).
    fn generate_minimal(
        &self,
        timestamp: u64,
        decision_level: DecisionLevel,
    ) -> Vec<InterfaceEquality> {
        let mut equalities = Vec::new();
        let rep = self.representative;

        for &term in &self.members {
            if term != rep {
                equalities.push(InterfaceEquality {
                    equality: Equality::new(term, rep),
                    theories: self.theories.clone(),
                    priority: EqualityPriority {
                        level: 100,
                        relevancy: 50,
                        decision_level,
                    },
                    is_necessary: true,
                    timestamp,
                });
            }
        }

        equalities
    }

    /// Generate incremental equalities.
    fn generate_incremental(
        &self,
        timestamp: u64,
        decision_level: DecisionLevel,
    ) -> Vec<InterfaceEquality> {
        // Similar to minimal, but could be enhanced with incremental logic
        self.generate_minimal(timestamp, decision_level)
    }

    /// Generate adaptive equalities based on heuristics.
    fn generate_adaptive(
        &self,
        timestamp: u64,
        decision_level: DecisionLevel,
    ) -> Vec<InterfaceEquality> {
        // Use minimal for small classes, eager for very small classes
        if self.members.len() <= 2 {
            self.generate_eager(timestamp, decision_level)
        } else {
            self.generate_minimal(timestamp, decision_level)
        }
    }
}

/// Configuration for interface equality management.
#[derive(Debug, Clone)]
pub struct InterfaceEqualityConfig {
    /// Default generation strategy.
    pub default_strategy: GenerationStrategy,

    /// Enable equality minimization.
    pub enable_minimization: bool,

    /// Enable priority-based scheduling.
    pub enable_priority: bool,

    /// Maximum equalities per batch.
    pub max_batch_size: usize,

    /// Enable relevancy tracking.
    pub track_relevancy: bool,

    /// Adaptive strategy threshold.
    pub adaptive_threshold: usize,
}

impl Default for InterfaceEqualityConfig {
    fn default() -> Self {
        Self {
            default_strategy: GenerationStrategy::Minimal,
            enable_minimization: true,
            enable_priority: true,
            max_batch_size: 1000,
            track_relevancy: true,
            adaptive_threshold: 10,
        }
    }
}

/// Statistics for interface equality management.
#[derive(Debug, Clone, Default)]
pub struct InterfaceEqualityStats {
    /// Equalities generated.
    pub equalities_generated: u64,
    /// Equalities minimized away.
    pub equalities_minimized: u64,
    /// Eager generations.
    pub eager_generations: u64,
    /// Lazy generations.
    pub lazy_generations: u64,
    /// Minimal generations.
    pub minimal_generations: u64,
    /// Equivalence classes.
    pub eclasses: u64,
    /// Batches sent.
    pub batches_sent: u64,
}

/// Interface equality manager.
pub struct InterfaceEqualityManager {
    /// Configuration.
    config: InterfaceEqualityConfig,

    /// Statistics.
    stats: InterfaceEqualityStats,

    /// Term to equivalence class mapping.
    term_to_eclass: FxHashMap<TermId, usize>,

    /// Equivalence classes.
    eclasses: Vec<InterfaceEClass>,

    /// Pending equalities (priority queue).
    pending: BinaryHeap<InterfaceEquality>,

    /// Generated equalities (deduplication).
    generated: FxHashSet<Equality>,

    /// Current timestamp.
    timestamp: u64,

    /// Current decision level.
    decision_level: DecisionLevel,

    /// Relevancy scores for terms.
    relevancy: FxHashMap<TermId, u32>,

    /// Equality generation history for backtracking.
    history: FxHashMap<DecisionLevel, Vec<Equality>>,
}

impl InterfaceEqualityManager {
    /// Create new manager.
    pub fn new() -> Self {
        Self::with_config(InterfaceEqualityConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: InterfaceEqualityConfig) -> Self {
        Self {
            config,
            stats: InterfaceEqualityStats::default(),
            term_to_eclass: FxHashMap::default(),
            eclasses: Vec::new(),
            pending: BinaryHeap::new(),
            generated: FxHashSet::default(),
            timestamp: 0,
            decision_level: 0,
            relevancy: FxHashMap::default(),
            history: FxHashMap::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &InterfaceEqualityStats {
        &self.stats
    }

    /// Register a term with theory.
    pub fn register_term(&mut self, term: TermId, theory: TheoryId) {
        if let Some(&eclass_id) = self.term_to_eclass.get(&term) {
            self.eclasses[eclass_id].add_term(term, theory);
        } else {
            let eclass_id = self.eclasses.len();
            self.eclasses.push(InterfaceEClass::new(term, theory));
            self.term_to_eclass.insert(term, eclass_id);
            self.stats.eclasses += 1;
        }
    }

    /// Assert equality and merge equivalence classes.
    pub fn assert_equality(&mut self, lhs: TermId, rhs: TermId) -> Result<(), String> {
        let lhs_class = self.find_or_create(lhs);
        let rhs_class = self.find_or_create(rhs);

        if lhs_class == rhs_class {
            return Ok(());
        }

        // Merge smaller into larger
        let (small, large) =
            if self.eclasses[lhs_class].members.len() < self.eclasses[rhs_class].members.len() {
                (lhs_class, rhs_class)
            } else {
                (rhs_class, lhs_class)
            };

        let small_eclass = self.eclasses[small].clone();
        self.eclasses[large].merge(&small_eclass);

        // Update term mappings
        for &term in &small_eclass.members {
            self.term_to_eclass.insert(term, large);
        }

        // Generate interface equalities if the merged class is shared
        if self.eclasses[large].is_shared() {
            self.generate_equalities_for_class(large)?;
        }

        Ok(())
    }

    /// Find or create equivalence class for term.
    fn find_or_create(&mut self, term: TermId) -> usize {
        if let Some(&eclass_id) = self.term_to_eclass.get(&term) {
            eclass_id
        } else {
            let eclass_id = self.eclasses.len();
            self.eclasses.push(InterfaceEClass::new(term, 0));
            self.term_to_eclass.insert(term, eclass_id);
            self.stats.eclasses += 1;
            eclass_id
        }
    }

    /// Generate equalities for an equivalence class.
    fn generate_equalities_for_class(&mut self, eclass_id: usize) -> Result<(), String> {
        if eclass_id >= self.eclasses.len() {
            return Err("Invalid eclass ID".to_string());
        }

        let eclass = &self.eclasses[eclass_id];
        let equalities = eclass.generate_equalities(self.timestamp, self.decision_level);

        for eq in equalities {
            if !self.generated.contains(&eq.equality) {
                self.generated.insert(eq.equality);
                self.pending.push(eq);
                self.stats.equalities_generated += 1;

                // Update strategy-specific stats
                match eclass.strategy {
                    GenerationStrategy::Eager => self.stats.eager_generations += 1,
                    GenerationStrategy::Lazy => self.stats.lazy_generations += 1,
                    GenerationStrategy::Minimal => self.stats.minimal_generations += 1,
                    _ => {}
                }
            }
        }

        self.timestamp += 1;
        Ok(())
    }

    /// Get pending equalities (batch).
    pub fn get_pending_batch(&mut self) -> Vec<InterfaceEquality> {
        let mut batch = Vec::new();

        while batch.len() < self.config.max_batch_size {
            if let Some(eq) = self.pending.pop() {
                batch.push(eq);
            } else {
                break;
            }
        }

        if !batch.is_empty() {
            self.stats.batches_sent += 1;
        }

        batch
    }

    /// Get all pending equalities.
    pub fn get_all_pending(&mut self) -> Vec<InterfaceEquality> {
        let mut all = Vec::new();

        while let Some(eq) = self.pending.pop() {
            all.push(eq);
        }

        if !all.is_empty() {
            self.stats.batches_sent += 1;
        }

        all
    }

    /// Set generation strategy for a term's equivalence class.
    pub fn set_strategy(
        &mut self,
        term: TermId,
        strategy: GenerationStrategy,
    ) -> Result<(), String> {
        let eclass_id = self
            .term_to_eclass
            .get(&term)
            .ok_or("Term not registered")?;
        self.eclasses[*eclass_id].strategy = strategy;
        Ok(())
    }

    /// Minimize pending equalities.
    ///
    /// Remove redundant equalities that can be inferred from others.
    pub fn minimize_equalities(&mut self) {
        if !self.config.enable_minimization {
            return;
        }

        // Convert pending to vector for processing
        let all_pending: Vec<_> = self.pending.drain().collect();
        let mut necessary = Vec::new();

        // Group by equivalence class
        let mut by_class: FxHashMap<usize, Vec<InterfaceEquality>> = FxHashMap::default();

        for eq in all_pending {
            if let Some(&eclass_id) = self.term_to_eclass.get(&eq.equality.lhs) {
                by_class.entry(eclass_id).or_default().push(eq);
            }
        }

        // For each class, keep only necessary equalities (star topology)
        for (_eclass_id, mut equalities) in by_class {
            if equalities.len() <= 2 {
                necessary.extend(equalities);
                continue;
            }

            // Find representative
            let rep = equalities[0].equality.lhs;

            // Keep only equalities involving the representative
            equalities.retain(|eq| eq.equality.lhs == rep || eq.equality.rhs == rep);

            let before = equalities.len();
            let minimized = equalities.len();
            self.stats.equalities_minimized += (before - minimized) as u64;

            necessary.extend(equalities);
        }

        // Re-add necessary equalities
        for eq in necessary {
            self.pending.push(eq);
        }
    }

    /// Update relevancy score for a term.
    pub fn update_relevancy(&mut self, term: TermId, score: u32) {
        if !self.config.track_relevancy {
            return;
        }

        self.relevancy.insert(term, score);

        // Update priority of pending equalities involving this term
        let all_pending: Vec<_> = self.pending.drain().collect();

        for mut eq in all_pending {
            if eq.equality.lhs == term || eq.equality.rhs == term {
                eq.priority.relevancy = score;
            }
            self.pending.push(eq);
        }
    }

    /// Push new decision level.
    pub fn push_decision_level(&mut self) {
        self.decision_level += 1;
    }

    /// Backtrack to decision level.
    pub fn backtrack(&mut self, level: DecisionLevel) -> Result<(), String> {
        if level > self.decision_level {
            return Err("Cannot backtrack to future level".to_string());
        }

        // Remove equalities from higher levels
        let all_pending: Vec<_> = self.pending.drain().collect();

        for eq in all_pending {
            if eq.priority.decision_level <= level {
                self.pending.push(eq);
            } else {
                self.generated.remove(&eq.equality);
            }
        }

        // Remove history above this level
        let levels_to_remove: Vec<_> = self
            .history
            .keys()
            .filter(|&&l| l > level)
            .copied()
            .collect();

        for l in levels_to_remove {
            self.history.remove(&l);
        }

        self.decision_level = level;
        Ok(())
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.term_to_eclass.clear();
        self.eclasses.clear();
        self.pending.clear();
        self.generated.clear();
        self.timestamp = 0;
        self.decision_level = 0;
        self.relevancy.clear();
        self.history.clear();
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = InterfaceEqualityStats::default();
    }

    /// Get number of pending equalities.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if equality has been generated.
    pub fn is_generated(&self, eq: &Equality) -> bool {
        self.generated.contains(eq)
    }

    /// Force generation of all equalities for shared classes.
    pub fn force_generate_all(&mut self) -> Result<(), String> {
        for eclass_id in 0..self.eclasses.len() {
            if self.eclasses[eclass_id].is_shared() {
                self.generate_equalities_for_class(eclass_id)?;
            }
        }
        Ok(())
    }

    /// Get equivalence class for a term.
    pub fn get_eclass(&self, term: TermId) -> Option<&InterfaceEClass> {
        self.term_to_eclass
            .get(&term)
            .and_then(|&id| self.eclasses.get(id))
    }

    /// Get representative for a term.
    pub fn get_representative(&self, term: TermId) -> Option<TermId> {
        self.get_eclass(term).map(|ec| ec.representative)
    }

    /// Check if two terms are in the same equivalence class.
    pub fn are_equal(&self, lhs: TermId, rhs: TermId) -> bool {
        if let (Some(&lhs_class), Some(&rhs_class)) =
            (self.term_to_eclass.get(&lhs), self.term_to_eclass.get(&rhs))
        {
            lhs_class == rhs_class
        } else {
            false
        }
    }
}

impl Default for InterfaceEqualityManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Equality scheduler for coordinating propagation.
pub struct EqualityScheduler {
    /// Scheduled equalities by priority.
    scheduled: BinaryHeap<InterfaceEquality>,
    /// Scheduling policy.
    policy: SchedulingPolicy,
}

/// Scheduling policy for equalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// FIFO (first-in-first-out).
    Fifo,
    /// Priority-based.
    Priority,
    /// Relevancy-based.
    Relevancy,
    /// Round-robin between theories.
    RoundRobin,
}

impl EqualityScheduler {
    /// Create new scheduler.
    pub fn new(policy: SchedulingPolicy) -> Self {
        Self {
            scheduled: BinaryHeap::new(),
            policy,
        }
    }

    /// Schedule an equality.
    pub fn schedule(&mut self, equality: InterfaceEquality) {
        self.scheduled.push(equality);
    }

    /// Get next equality to propagate.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<InterfaceEquality> {
        match self.policy {
            SchedulingPolicy::Fifo => {
                // Convert to FIFO by ignoring priority
                let all: Vec<_> = self.scheduled.drain().collect();

                all.into_iter().next()
            }
            SchedulingPolicy::Priority | SchedulingPolicy::Relevancy => self.scheduled.pop(),
            SchedulingPolicy::RoundRobin => {
                // Simplified round-robin
                self.scheduled.pop()
            }
        }
    }

    /// Get batch of equalities.
    pub fn next_batch(&mut self, size: usize) -> Vec<InterfaceEquality> {
        let mut batch = Vec::new();

        for _ in 0..size {
            if let Some(eq) = self.next() {
                batch.push(eq);
            } else {
                break;
            }
        }

        batch
    }

    /// Clear scheduler.
    pub fn clear(&mut self) {
        self.scheduled.clear();
    }
}

/// Minimizer for interface equalities.
pub struct EqualityMinimizer {
    /// Union-find for transitivity.
    parent: FxHashMap<TermId, TermId>,
    /// Rank for union-by-rank.
    rank: FxHashMap<TermId, usize>,
}

impl EqualityMinimizer {
    /// Create new minimizer.
    pub fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
        }
    }

    /// Add equality.
    pub fn add_equality(&mut self, eq: Equality) {
        let lhs_root = self.find(eq.lhs);
        let rhs_root = self.find(eq.rhs);

        if lhs_root == rhs_root {
            return;
        }

        let lhs_rank = self.rank.get(&lhs_root).copied().unwrap_or(0);
        let rhs_rank = self.rank.get(&rhs_root).copied().unwrap_or(0);

        if lhs_rank < rhs_rank {
            self.parent.insert(lhs_root, rhs_root);
        } else if lhs_rank > rhs_rank {
            self.parent.insert(rhs_root, lhs_root);
        } else {
            self.parent.insert(lhs_root, rhs_root);
            self.rank.insert(rhs_root, rhs_rank + 1);
        }
    }

    /// Find representative.
    fn find(&mut self, mut term: TermId) -> TermId {
        let mut path = Vec::new();

        while let Some(&parent) = self.parent.get(&term) {
            if parent == term {
                break;
            }
            path.push(term);
            term = parent;
        }

        for node in path {
            self.parent.insert(node, term);
        }

        term
    }

    /// Check if equality is redundant.
    pub fn is_redundant(&mut self, eq: &Equality) -> bool {
        self.find(eq.lhs) == self.find(eq.rhs)
    }

    /// Minimize a set of equalities.
    pub fn minimize(&mut self, equalities: Vec<Equality>) -> Vec<Equality> {
        let mut minimal = Vec::new();

        for eq in equalities {
            if !self.is_redundant(&eq) {
                self.add_equality(eq);
                minimal.push(eq);
            }
        }

        minimal
    }

    /// Clear state.
    pub fn clear(&mut self) {
        self.parent.clear();
        self.rank.clear();
    }
}

impl Default for EqualityMinimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interface_eclass() {
        let mut eclass = InterfaceEClass::new(1, 0);
        eclass.add_term(2, 1);

        assert_eq!(eclass.members.len(), 2);
        assert!(eclass.is_shared());
    }

    #[test]
    fn test_minimal_generation() {
        let mut eclass = InterfaceEClass::new(1, 0);
        eclass.add_term(2, 0);
        eclass.add_term(3, 0);

        let equalities = eclass.generate_minimal(0, 0);
        assert_eq!(equalities.len(), 2); // {1,2}, {1,3}
    }

    #[test]
    fn test_manager_creation() {
        let manager = InterfaceEqualityManager::new();
        assert_eq!(manager.stats().equalities_generated, 0);
    }

    #[test]
    fn test_register_term() {
        let mut manager = InterfaceEqualityManager::new();
        manager.register_term(1, 0);
        manager.register_term(1, 1);

        assert_eq!(manager.stats().eclasses, 1);
    }

    #[test]
    fn test_assert_equality() {
        let mut manager = InterfaceEqualityManager::new();
        manager.register_term(1, 0);
        manager.register_term(2, 1);

        manager.assert_equality(1, 2).expect("Assert failed");
        assert!(manager.are_equal(1, 2));
    }

    #[test]
    fn test_get_pending() {
        let mut manager = InterfaceEqualityManager::new();
        manager.register_term(1, 0);
        manager.register_term(2, 1);
        manager.register_term(1, 1); // Make shared

        manager.assert_equality(1, 2).expect("Assert failed");

        let pending = manager.get_all_pending();
        assert!(!pending.is_empty());
    }

    #[test]
    fn test_minimization() {
        let mut manager = InterfaceEqualityManager::new();

        // Create a class with multiple terms
        for i in 1..=5 {
            manager.register_term(i, 0);
            manager.register_term(i, 1);
        }

        for i in 2..=5 {
            manager.assert_equality(1, i).expect("Assert failed");
        }

        manager.minimize_equalities();

        let pending = manager.get_all_pending();
        // Should have minimal equalities (star topology)
        assert!(pending.len() <= 4);
    }

    #[test]
    fn test_scheduler() {
        let mut scheduler = EqualityScheduler::new(SchedulingPolicy::Priority);

        let eq = InterfaceEquality {
            equality: Equality::new(1, 2),
            theories: FxHashSet::default(),
            priority: EqualityPriority {
                level: 100,
                relevancy: 50,
                decision_level: 0,
            },
            is_necessary: true,
            timestamp: 0,
        };

        scheduler.schedule(eq);
        assert!(scheduler.next().is_some());
    }

    #[test]
    fn test_minimizer() {
        let mut minimizer = EqualityMinimizer::new();

        let eq1 = Equality::new(1, 2);
        let eq2 = Equality::new(2, 3);
        let eq3 = Equality::new(1, 3); // Redundant

        minimizer.add_equality(eq1);
        minimizer.add_equality(eq2);

        assert!(minimizer.is_redundant(&eq3));
    }

    #[test]
    fn test_backtrack() {
        let mut manager = InterfaceEqualityManager::new();

        manager.push_decision_level();
        manager.register_term(1, 0);

        manager.backtrack(0).expect("Backtrack failed");
    }

    #[test]
    fn test_set_strategy() {
        let mut manager = InterfaceEqualityManager::new();
        manager.register_term(1, 0);

        manager
            .set_strategy(1, GenerationStrategy::Eager)
            .expect("Set strategy failed");

        let eclass = manager.get_eclass(1).expect("No eclass");
        assert_eq!(eclass.strategy, GenerationStrategy::Eager);
    }
}
