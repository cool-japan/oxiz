//! Theory Propagation Queue with Priority Scheduling
//!
//! This module implements an efficient propagation queue for theory solvers
//! that schedules theory checks based on priority and cost heuristics.
//!
//! Key features:
//! - Priority-based scheduling (cheap theories first)
//! - Batching of similar constraints
//! - Deduplication of redundant propagations
//! - Statistics tracking for tuning

use crate::theory::TheoryId;
use oxiz_core::ast::TermId;
use rustc_hash::FxHashSet;
use std::collections::VecDeque;

/// Priority level for theory propagations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropagationPriority {
    /// Critical propagations (unit clauses, immediate conflicts)
    Critical = 0,
    /// High priority (cheap theories like EUF)
    High = 1,
    /// Normal priority (most theory propagations)
    Normal = 2,
    /// Low priority (expensive theories like LIA)
    Low = 3,
    /// Background (preprocessing, simplification)
    Background = 4,
}

/// A theory propagation item
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Propagation {
    /// The literal to propagate
    pub literal: TermId,
    /// The reason/explanation
    pub reason: Option<TermId>,
    /// Which theory is propagating
    pub theory: TheoryId,
    /// Priority of this propagation
    pub priority: PropagationPriority,
}

impl Propagation {
    /// Create a new propagation
    #[must_use]
    pub fn new(
        literal: TermId,
        reason: Option<TermId>,
        theory: TheoryId,
        priority: PropagationPriority,
    ) -> Self {
        Self {
            literal,
            reason,
            theory,
            priority,
        }
    }

    /// Create a critical propagation
    #[must_use]
    pub fn critical(literal: TermId, reason: Option<TermId>, theory: TheoryId) -> Self {
        Self::new(literal, reason, theory, PropagationPriority::Critical)
    }

    /// Create a high priority propagation
    #[must_use]
    pub fn high(literal: TermId, reason: Option<TermId>, theory: TheoryId) -> Self {
        Self::new(literal, reason, theory, PropagationPriority::High)
    }

    /// Create a normal priority propagation
    #[must_use]
    pub fn normal(literal: TermId, reason: Option<TermId>, theory: TheoryId) -> Self {
        Self::new(literal, reason, theory, PropagationPriority::Normal)
    }

    /// Create a low priority propagation
    #[must_use]
    pub fn low(literal: TermId, reason: Option<TermId>, theory: TheoryId) -> Self {
        Self::new(literal, reason, theory, PropagationPriority::Low)
    }
}

/// Statistics for the propagation queue
#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    /// Total propagations enqueued
    pub total_enqueued: usize,
    /// Propagations dequeued
    pub total_dequeued: usize,
    /// Propagations deduplicated
    pub deduplicated: usize,
    /// Propagations by priority level
    pub by_priority: [usize; 5],
    /// Propagations by theory
    pub by_theory: [usize; 10],
}

impl PropagationStats {
    /// Create new statistics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get deduplication rate
    #[must_use]
    pub fn dedup_rate(&self) -> f64 {
        if self.total_enqueued == 0 {
            0.0
        } else {
            self.deduplicated as f64 / self.total_enqueued as f64
        }
    }
}

/// Priority-based propagation queue
#[derive(Debug)]
pub struct PropagationQueue {
    /// Queues for each priority level
    queues: [VecDeque<Propagation>; 5],
    /// Set of literals already in queue (for deduplication)
    seen: FxHashSet<TermId>,
    /// Statistics
    stats: PropagationStats,
    /// Enable deduplication
    enable_dedup: bool,
}

impl Default for PropagationQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl PropagationQueue {
    /// Create a new propagation queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            queues: [
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ],
            seen: FxHashSet::default(),
            stats: PropagationStats::new(),
            enable_dedup: true,
        }
    }

    /// Create a queue without deduplication
    #[must_use]
    pub fn without_dedup() -> Self {
        let mut queue = Self::new();
        queue.enable_dedup = false;
        queue
    }

    /// Enqueue a propagation
    ///
    /// Returns true if the propagation was added, false if it was deduplicated
    pub fn enqueue(&mut self, prop: Propagation) -> bool {
        self.stats.total_enqueued += 1;
        self.stats.by_priority[prop.priority as usize] += 1;
        self.stats.by_theory[prop.theory as usize] += 1;

        // Check for deduplication
        if self.enable_dedup && self.seen.contains(&prop.literal) {
            self.stats.deduplicated += 1;
            return false;
        }

        if self.enable_dedup {
            self.seen.insert(prop.literal);
        }

        self.queues[prop.priority as usize].push_back(prop);
        true
    }

    /// Dequeue the highest priority propagation
    pub fn dequeue(&mut self) -> Option<Propagation> {
        // Check queues in priority order
        for queue in &mut self.queues {
            if let Some(prop) = queue.pop_front() {
                self.stats.total_dequeued += 1;
                if self.enable_dedup {
                    self.seen.remove(&prop.literal);
                }
                return Some(prop);
            }
        }
        None
    }

    /// Peek at the next propagation without removing it
    #[must_use]
    pub fn peek(&self) -> Option<&Propagation> {
        for queue in &self.queues {
            if let Some(prop) = queue.front() {
                return Some(prop);
            }
        }
        None
    }

    /// Check if the queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(|q| q.is_empty())
    }

    /// Get the total number of propagations in the queue
    #[must_use]
    pub fn len(&self) -> usize {
        self.queues.iter().map(|q| q.len()).sum()
    }

    /// Get the number of propagations at a specific priority
    #[must_use]
    pub fn len_at_priority(&self, priority: PropagationPriority) -> usize {
        self.queues[priority as usize].len()
    }

    /// Clear all propagations
    pub fn clear(&mut self) {
        for queue in &mut self.queues {
            queue.clear();
        }
        self.seen.clear();
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &PropagationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Batch enqueue multiple propagations
    ///
    /// More efficient than calling enqueue multiple times
    pub fn enqueue_batch(&mut self, props: impl IntoIterator<Item = Propagation>) {
        for prop in props {
            self.enqueue(prop);
        }
    }

    /// Filter and keep only propagations matching a predicate
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&Propagation) -> bool,
    {
        for queue in &mut self.queues {
            queue.retain(|prop| {
                let keep = f(prop);
                if !keep && self.enable_dedup {
                    self.seen.remove(&prop.literal);
                }
                keep
            });
        }
    }

    /// Drain all propagations from a specific theory
    pub fn drain_theory(&mut self, theory: TheoryId) -> Vec<Propagation> {
        let mut result = Vec::new();
        for queue in &mut self.queues {
            let mut i = 0;
            while i < queue.len() {
                if queue[i].theory == theory {
                    if let Some(prop) = queue.remove(i) {
                        if self.enable_dedup {
                            self.seen.remove(&prop.literal);
                        }
                        result.push(prop);
                    }
                } else {
                    i += 1;
                }
            }
        }
        result
    }

    /// Get an iterator over all propagations (in priority order)
    pub fn iter(&self) -> impl Iterator<Item = &Propagation> {
        self.queues.iter().flat_map(|q| q.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        let mut queue = PropagationQueue::new();

        // Enqueue in reverse priority order
        queue.enqueue(Propagation::low(TermId::new(1), None, TheoryId::LIA));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::EUF));
        queue.enqueue(Propagation::critical(TermId::new(3), None, TheoryId::EUF));

        // Should dequeue in priority order (critical first)
        assert_eq!(queue.dequeue().unwrap().literal, TermId::new(3));
        assert_eq!(queue.dequeue().unwrap().literal, TermId::new(2));
        assert_eq!(queue.dequeue().unwrap().literal, TermId::new(1));
    }

    #[test]
    fn test_deduplication() {
        let mut queue = PropagationQueue::new();

        let prop1 = Propagation::normal(TermId::new(1), None, TheoryId::EUF);
        let prop2 = Propagation::normal(TermId::new(1), None, TheoryId::LRA);

        assert!(queue.enqueue(prop1));
        assert!(!queue.enqueue(prop2)); // Should be deduplicated

        assert_eq!(queue.len(), 1);
        assert_eq!(queue.stats().deduplicated, 1);
    }

    #[test]
    fn test_without_dedup() {
        let mut queue = PropagationQueue::without_dedup();

        let prop1 = Propagation::normal(TermId::new(1), None, TheoryId::EUF);
        let prop2 = Propagation::normal(TermId::new(1), None, TheoryId::LRA);

        assert!(queue.enqueue(prop1));
        assert!(queue.enqueue(prop2)); // Should NOT be deduplicated

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.stats().deduplicated, 0);
    }

    #[test]
    fn test_empty_queue() {
        let mut queue = PropagationQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn test_peek() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::normal(TermId::new(1), None, TheoryId::EUF));

        // Peek shouldn't remove the item
        assert_eq!(queue.peek().unwrap().literal, TermId::new(1));
        assert_eq!(queue.len(), 1);

        // Dequeue should remove it
        assert_eq!(queue.dequeue().unwrap().literal, TermId::new(1));
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::normal(TermId::new(1), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::LRA));

        assert_eq!(queue.len(), 2);

        queue.clear();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_batch_enqueue() {
        let mut queue = PropagationQueue::new();

        let props = vec![
            Propagation::normal(TermId::new(1), None, TheoryId::EUF),
            Propagation::normal(TermId::new(2), None, TheoryId::LRA),
            Propagation::high(TermId::new(3), None, TheoryId::EUF),
        ];

        queue.enqueue_batch(props);
        assert_eq!(queue.len(), 3);

        // High priority should come first
        assert_eq!(queue.dequeue().unwrap().literal, TermId::new(3));
    }

    #[test]
    fn test_retain() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::normal(TermId::new(1), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::LRA));
        queue.enqueue(Propagation::normal(TermId::new(3), None, TheoryId::EUF));

        // Keep only EUF propagations
        queue.retain(|p| p.theory == TheoryId::EUF);

        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_drain_theory() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::normal(TermId::new(1), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::LRA));
        queue.enqueue(Propagation::normal(TermId::new(3), None, TheoryId::EUF));

        let euf_props = queue.drain_theory(TheoryId::EUF);

        assert_eq!(euf_props.len(), 2);
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.dequeue().unwrap().theory, TheoryId::LRA);
    }

    #[test]
    fn test_statistics() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::critical(TermId::new(1), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::LRA));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::EUF)); // Duplicate

        let stats = queue.stats();
        assert_eq!(stats.total_enqueued, 3);
        assert_eq!(stats.deduplicated, 1);
        assert_eq!(stats.by_priority[PropagationPriority::Critical as usize], 1);
        assert_eq!(stats.by_priority[PropagationPriority::Normal as usize], 2);
    }

    #[test]
    fn test_dedup_rate() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::normal(TermId::new(1), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(1), None, TheoryId::LRA));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::LRA));

        // 2 duplicates out of 4 total = 50%
        assert!((queue.stats().dedup_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_len_at_priority() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::critical(TermId::new(1), None, TheoryId::EUF));
        queue.enqueue(Propagation::critical(TermId::new(2), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(3), None, TheoryId::LRA));

        assert_eq!(queue.len_at_priority(PropagationPriority::Critical), 2);
        assert_eq!(queue.len_at_priority(PropagationPriority::Normal), 1);
        assert_eq!(queue.len_at_priority(PropagationPriority::Low), 0);
    }

    #[test]
    fn test_propagation_constructors() {
        let lit = TermId::new(1);
        let reason = Some(TermId::new(2));
        let theory = TheoryId::EUF;

        let critical = Propagation::critical(lit, reason, theory);
        assert_eq!(critical.priority, PropagationPriority::Critical);

        let high = Propagation::high(lit, reason, theory);
        assert_eq!(high.priority, PropagationPriority::High);

        let normal = Propagation::normal(lit, reason, theory);
        assert_eq!(normal.priority, PropagationPriority::Normal);

        let low = Propagation::low(lit, reason, theory);
        assert_eq!(low.priority, PropagationPriority::Low);
    }

    #[test]
    fn test_iter() {
        let mut queue = PropagationQueue::new();

        queue.enqueue(Propagation::critical(TermId::new(1), None, TheoryId::EUF));
        queue.enqueue(Propagation::normal(TermId::new(2), None, TheoryId::LRA));

        let count = queue.iter().count();
        assert_eq!(count, 2);
    }
}
