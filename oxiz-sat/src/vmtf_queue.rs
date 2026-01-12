//! VMTF (Variable Move-To-Front) queue with bump mechanism
//!
//! This module implements an enhanced VMTF decision heuristic with a bump queue
//! that tracks recently conflicting variables. This is used in modern solvers
//! like Kissat and CaDiCaL for efficient variable ordering.
//!
//! The bump queue maintains variables that have been involved in recent conflicts,
//! prioritizing them for decision making.
//!
//! References:
//! - "The CaDiCaL, Kissat, Paracooba Solvers" (SAT Competition 2020)
//! - "Between SAT and UNSAT: The Fundamental Difference in CDCL SAT" (Biere)

use crate::literal::Var;
use std::collections::VecDeque;

/// Statistics for VMTF bump queue
#[derive(Debug, Clone, Default)]
pub struct VmtfBumpStats {
    /// Total number of bumps performed
    pub total_bumps: u64,
    /// Number of variables currently in queue
    pub queue_size: usize,
    /// Maximum queue size reached
    pub max_queue_size: usize,
    /// Number of queue flushes
    pub flushes: u64,
}

impl VmtfBumpStats {
    /// Display statistics
    pub fn display(&self) {
        println!("VMTF Bump Queue Statistics:");
        println!("  Total bumps: {}", self.total_bumps);
        println!("  Current queue size: {}", self.queue_size);
        println!("  Max queue size: {}", self.max_queue_size);
        println!("  Queue flushes: {}", self.flushes);
        if self.total_bumps > 0 {
            let avg_queue_size = self.max_queue_size as f64 / self.flushes.max(1) as f64;
            println!("  Avg queue size at flush: {:.1}", avg_queue_size);
        }
    }
}

/// VMTF bump queue
///
/// Maintains a queue of recently bumped variables for prioritized decision making.
/// Variables in the queue are prioritized over the regular VMTF order.
#[derive(Debug)]
pub struct VmtfBumpQueue {
    /// Queue of bumped variables (most recent at back)
    queue: VecDeque<Var>,
    /// Track which variables are in queue
    in_queue: Vec<bool>,
    /// Maximum queue size before flush
    max_size: usize,
    /// Number of conflicts before auto-flush
    conflicts_before_flush: u64,
    /// Current conflict count
    conflict_count: u64,
    /// Statistics
    stats: VmtfBumpStats,
}

impl Default for VmtfBumpQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl VmtfBumpQueue {
    /// Create a new VMTF bump queue with default settings
    ///
    /// Default settings:
    /// - Max size: 100 variables
    /// - Auto-flush every 10,000 conflicts
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            in_queue: Vec::new(),
            max_size: 100,
            conflicts_before_flush: 10000,
            conflict_count: 0,
            stats: VmtfBumpStats::default(),
        }
    }

    /// Create with custom settings
    #[must_use]
    pub fn with_config(max_size: usize, conflicts_before_flush: u64) -> Self {
        Self {
            queue: VecDeque::new(),
            in_queue: Vec::new(),
            max_size,
            conflicts_before_flush,
            conflict_count: 0,
            stats: VmtfBumpStats::default(),
        }
    }

    /// Resize for new number of variables
    pub fn resize(&mut self, num_vars: usize) {
        self.in_queue.resize(num_vars, false);
    }

    /// Bump a variable to the queue
    ///
    /// Adds the variable to the end of the queue if not already present.
    /// If queue is full, removes oldest variable.
    pub fn bump(&mut self, var: Var) {
        let var_idx = var.index();

        // Ensure we have space
        if var_idx >= self.in_queue.len() {
            self.in_queue.resize(var_idx + 1, false);
        }

        // Skip if already in queue
        if self.in_queue[var_idx] {
            return;
        }

        // Add to queue
        self.queue.push_back(var);
        self.in_queue[var_idx] = true;
        self.stats.total_bumps += 1;
        self.stats.queue_size = self.queue.len();
        self.stats.max_queue_size = self.stats.max_queue_size.max(self.queue.len());

        // Maintain max size
        while self.queue.len() > self.max_size {
            if let Some(old_var) = self.queue.pop_front() {
                self.in_queue[old_var.index()] = false;
            }
        }

        self.stats.queue_size = self.queue.len();
    }

    /// Get next variable from queue
    ///
    /// Returns None if queue is empty.
    /// The returned variable is removed from the queue.
    pub fn pop(&mut self) -> Option<Var> {
        if let Some(var) = self.queue.pop_back() {
            self.in_queue[var.index()] = false;
            self.stats.queue_size = self.queue.len();
            Some(var)
        } else {
            None
        }
    }

    /// Peek at next variable without removing it
    #[must_use]
    pub fn peek(&self) -> Option<Var> {
        self.queue.back().copied()
    }

    /// Check if a variable is in the queue
    #[must_use]
    pub fn contains(&self, var: Var) -> bool {
        let idx = var.index();
        idx < self.in_queue.len() && self.in_queue[idx]
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get current queue size
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Record a conflict
    ///
    /// Increments conflict counter and auto-flushes if needed
    pub fn on_conflict(&mut self) {
        self.conflict_count += 1;

        if self.conflict_count >= self.conflicts_before_flush {
            self.flush();
            self.conflict_count = 0;
        }
    }

    /// Flush (clear) the queue
    pub fn flush(&mut self) {
        for var in &self.queue {
            self.in_queue[var.index()] = false;
        }
        self.queue.clear();
        self.stats.flushes += 1;
        self.stats.queue_size = 0;
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &VmtfBumpStats {
        &self.stats
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.queue.clear();
        self.in_queue.clear();
        self.conflict_count = 0;
        self.stats = VmtfBumpStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmtf_bump_queue_creation() {
        let queue = VmtfBumpQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_bump_and_pop() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.bump(Var::new(1));
        queue.bump(Var::new(2));

        assert_eq!(queue.len(), 3);
        assert_eq!(queue.pop(), Some(Var::new(2))); // LIFO order
        assert_eq!(queue.pop(), Some(Var::new(1)));
        assert_eq!(queue.pop(), Some(Var::new(0)));
        assert!(queue.is_empty());
    }

    #[test]
    fn test_duplicate_bump() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.bump(Var::new(0)); // Duplicate
        queue.bump(Var::new(0)); // Duplicate

        assert_eq!(queue.len(), 1); // Should only have one copy
    }

    #[test]
    fn test_max_size_limit() {
        let mut queue = VmtfBumpQueue::with_config(3, 10000);
        queue.resize(10);

        for i in 0..10 {
            queue.bump(Var::new(i));
        }

        // Should keep only last 3
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.pop(), Some(Var::new(9)));
        assert_eq!(queue.pop(), Some(Var::new(8)));
        assert_eq!(queue.pop(), Some(Var::new(7)));
    }

    #[test]
    fn test_contains() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.bump(Var::new(2));

        assert!(queue.contains(Var::new(0)));
        assert!(!queue.contains(Var::new(1)));
        assert!(queue.contains(Var::new(2)));
    }

    #[test]
    fn test_peek() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.bump(Var::new(1));

        assert_eq!(queue.peek(), Some(Var::new(1)));
        assert_eq!(queue.len(), 2); // Peek doesn't remove

        queue.pop();
        assert_eq!(queue.peek(), Some(Var::new(0)));
    }

    #[test]
    fn test_auto_flush() {
        let mut queue = VmtfBumpQueue::with_config(100, 5);
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.bump(Var::new(1));

        // Trigger auto-flush after 5 conflicts
        for _ in 0..5 {
            queue.on_conflict();
        }

        assert!(queue.is_empty());
        assert_eq!(queue.stats().flushes, 1);
    }

    #[test]
    fn test_manual_flush() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.bump(Var::new(1));
        queue.bump(Var::new(2));

        queue.flush();

        assert!(queue.is_empty());
        assert!(!queue.contains(Var::new(0)));
        assert!(!queue.contains(Var::new(1)));
        assert!(!queue.contains(Var::new(2)));
    }

    #[test]
    fn test_stats_tracking() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.bump(Var::new(1));
        queue.bump(Var::new(0)); // Duplicate

        let stats = queue.stats();
        assert_eq!(stats.total_bumps, 2); // Only 2 actual bumps
        assert_eq!(stats.queue_size, 2);
    }

    #[test]
    fn test_clear() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(10);

        queue.bump(Var::new(0));
        queue.on_conflict();

        queue.clear();

        assert!(queue.is_empty());
        assert_eq!(queue.stats().total_bumps, 0);
    }

    #[test]
    fn test_resize() {
        let mut queue = VmtfBumpQueue::new();
        queue.resize(5);

        queue.bump(Var::new(4));

        // Resize larger
        queue.resize(10);
        queue.bump(Var::new(9));

        assert_eq!(queue.len(), 2);
    }
}
