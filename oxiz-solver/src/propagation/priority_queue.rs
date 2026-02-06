//! Priority Queue for Propagations.
//!
//! Manages propagation priorities across multiple theories, ensuring
//! efficient propagation order for conflict-driven theory combination.
//!
//! ## Priority Levels
//!
//! - **Boolean (0)**: Highest priority - unit propagation
//! - **Equality (1)**: Shared term equalities
//! - **Theory (2-10)**: Per-theory propagations
//! - **Heuristic (11+)**: Low-priority optional propagations
//!
//! ## References
//!
//! - Z3's `smt/smt_context.cpp` propagation queue

use oxiz_sat::Lit;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Priority level (lower = higher priority).
pub type Priority = u8;

/// Propagation ID.
pub type PropagationId = usize;

/// Type of propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagationType {
    /// Boolean unit propagation.
    Boolean,
    /// Equality propagation between theories.
    Equality,
    /// Theory-specific propagation.
    Theory(u8),
    /// Heuristic propagation (optional).
    Heuristic,
}

impl PropagationType {
    /// Get the priority level for this propagation type.
    pub fn priority(&self) -> Priority {
        match self {
            PropagationType::Boolean => 0,
            PropagationType::Equality => 1,
            PropagationType::Theory(theory_id) => 2 + theory_id,
            PropagationType::Heuristic => 255,
        }
    }
}

/// A propagation entry in the queue.
#[derive(Debug, Clone)]
pub struct Propagation {
    /// Unique ID for this propagation.
    pub id: PropagationId,
    /// Type of propagation.
    pub prop_type: PropagationType,
    /// Priority (computed from type).
    pub priority: Priority,
    /// Literal being propagated (if Boolean).
    pub literal: Option<Lit>,
    /// Timestamp (for FIFO within same priority).
    pub timestamp: u64,
}

impl PartialEq for Propagation {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Propagation {}

impl PartialOrd for Propagation {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Propagation {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (lower priority value = higher priority)
        // For timestamps, also reverse to get FIFO order (lower timestamp = earlier = higher priority)
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.timestamp.cmp(&self.timestamp))
    }
}

/// Configuration for priority queue.
#[derive(Debug, Clone)]
pub struct PriorityQueueConfig {
    /// Enable timestamp ordering within same priority.
    pub use_timestamps: bool,
    /// Maximum queue size (0 = unlimited).
    pub max_size: usize,
}

impl Default for PriorityQueueConfig {
    fn default() -> Self {
        Self {
            use_timestamps: true,
            max_size: 100_000,
        }
    }
}

/// Statistics for priority queue.
#[derive(Debug, Clone, Default)]
pub struct PriorityQueueStats {
    /// Total propagations enqueued.
    pub enqueued: u64,
    /// Total propagations dequeued.
    pub dequeued: u64,
    /// Peak queue size.
    pub peak_size: usize,
    /// Propagations by type.
    pub by_type: FxHashMap<u8, u64>,
}

/// Priority queue for propagations.
#[derive(Debug)]
pub struct PropagationQueue {
    /// Binary heap (min-heap by priority).
    heap: BinaryHeap<Propagation>,
    /// Next propagation ID.
    next_id: PropagationId,
    /// Current timestamp (for FIFO within priority).
    timestamp: u64,
    /// Configuration.
    config: PriorityQueueConfig,
    /// Statistics.
    stats: PriorityQueueStats,
}

impl PropagationQueue {
    /// Create a new priority queue.
    pub fn new(config: PriorityQueueConfig) -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_id: 0,
            timestamp: 0,
            config,
            stats: PriorityQueueStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PriorityQueueConfig::default())
    }

    /// Enqueue a propagation.
    pub fn enqueue(&mut self, prop_type: PropagationType, literal: Option<Lit>) -> PropagationId {
        let id = self.next_id;
        self.next_id += 1;

        let priority = prop_type.priority();
        let timestamp = if self.config.use_timestamps {
            let ts = self.timestamp;
            self.timestamp += 1;
            ts
        } else {
            0
        };

        let prop = Propagation {
            id,
            prop_type,
            priority,
            literal,
            timestamp,
        };

        self.heap.push(prop);
        self.stats.enqueued += 1;

        // Update peak size
        if self.heap.len() > self.stats.peak_size {
            self.stats.peak_size = self.heap.len();
        }

        // Update by-type stats
        *self.stats.by_type.entry(priority).or_insert(0) += 1;

        id
    }

    /// Dequeue the highest-priority propagation.
    pub fn dequeue(&mut self) -> Option<Propagation> {
        let prop = self.heap.pop()?;
        self.stats.dequeued += 1;
        Some(prop)
    }

    /// Peek at the highest-priority propagation without removing it.
    pub fn peek(&self) -> Option<&Propagation> {
        self.heap.peek()
    }

    /// Check if queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get current queue size.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Clear all propagations.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.timestamp = 0;
    }

    /// Get statistics.
    pub fn stats(&self) -> &PriorityQueueStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = PriorityQueueStats::default();
    }
}

impl Default for PropagationQueue {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_creation() {
        let queue = PropagationQueue::default_config();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_priority_ordering() {
        use oxiz_sat::Var;
        let mut queue = PropagationQueue::default_config();

        // Enqueue in reverse priority order
        let _heuristic = queue.enqueue(PropagationType::Heuristic, None);
        let _theory = queue.enqueue(PropagationType::Theory(2), None);
        let _equality = queue.enqueue(PropagationType::Equality, None);
        let boolean = queue.enqueue(PropagationType::Boolean, Some(Lit::pos(Var::new(0))));

        // Boolean should be dequeued first (highest priority)
        let first = queue.dequeue().unwrap();
        assert_eq!(first.id, boolean);
        assert_eq!(first.priority, 0);
    }

    #[test]
    fn test_timestamp_fifo() {
        let mut queue = PropagationQueue::default_config();

        // Enqueue multiple with same priority
        let id1 = queue.enqueue(PropagationType::Theory(0), None);
        let id2 = queue.enqueue(PropagationType::Theory(0), None);
        let id3 = queue.enqueue(PropagationType::Theory(0), None);

        // Should dequeue in FIFO order
        assert_eq!(queue.dequeue().unwrap().id, id1);
        assert_eq!(queue.dequeue().unwrap().id, id2);
        assert_eq!(queue.dequeue().unwrap().id, id3);
    }

    #[test]
    fn test_peek() {
        use oxiz_sat::Var;
        let mut queue = PropagationQueue::default_config();
        let id = queue.enqueue(PropagationType::Boolean, Some(Lit::pos(Var::new(0))));

        let peeked = queue.peek().unwrap();
        assert_eq!(peeked.id, id);
        assert_eq!(queue.len(), 1); // Still in queue

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, id);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_clear() {
        use oxiz_sat::Var;
        let mut queue = PropagationQueue::default_config();
        queue.enqueue(PropagationType::Boolean, Some(Lit::pos(Var::new(0))));
        queue.enqueue(PropagationType::Equality, None);

        assert_eq!(queue.len(), 2);

        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_stats() {
        use oxiz_sat::Var;
        let mut queue = PropagationQueue::default_config();
        queue.enqueue(PropagationType::Boolean, Some(Lit::pos(Var::new(0))));
        queue.enqueue(PropagationType::Boolean, Some(Lit::pos(Var::new(1))));
        queue.dequeue();

        let stats = queue.stats();
        assert_eq!(stats.enqueued, 2);
        assert_eq!(stats.dequeued, 1);
        assert_eq!(stats.peak_size, 2);
    }
}
