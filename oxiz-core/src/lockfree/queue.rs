//! Concurrent queue using parking_lot for low-overhead synchronization.
//!
//! A FIFO queue designed for inter-thread clause sharing in parallel
//! SAT/SMT solving, using parking_lot::Mutex for minimal lock contention.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A concurrent FIFO queue using a lock-based approach with parking_lot.
///
/// Uses parking_lot::Mutex which has extremely low overhead (no system
/// calls in the uncontended case), making it nearly as fast as a
/// lock-free implementation for most workloads.
pub struct LockFreeQueue<T> {
    inner: Mutex<VecDeque<T>>,
    len: AtomicUsize,
}

impl<T> LockFreeQueue<T> {
    /// Create a new empty queue
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(VecDeque::new()),
            len: AtomicUsize::new(0),
        }
    }

    /// Push an item to the back of the queue
    pub fn push(&self, data: T) {
        self.inner.lock().push_back(data);
        self.len.fetch_add(1, Ordering::Relaxed);
    }

    /// Try to pop an item from the front of the queue
    pub fn pop(&self) -> Option<T> {
        let result = self.inner.lock().pop_front();
        if result.is_some() {
            self.len.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.len.load(Ordering::Relaxed) == 0
    }

    /// Get the approximate length of the queue
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}
