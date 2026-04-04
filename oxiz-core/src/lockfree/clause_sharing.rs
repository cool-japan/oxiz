//! Lock-free clause sharing for parallel SAT solving.
//!
//! Provides a mechanism for parallel solver threads to share learned
//! clauses without locks, using the lock-free queue.

use super::queue::LockFreeQueue;
use std::sync::atomic::{AtomicU64, Ordering};

/// A shared clause represented as a vector of literal indices.
/// Uses i32 to match typical SAT solver literal representation.
#[derive(Debug, Clone)]
pub struct SharedClause {
    /// The clause literals (positive = variable, negative = negated variable)
    pub literals: Vec<i32>,
    /// The LBD (Literal Block Distance) quality measure
    pub lbd: u32,
    /// Source solver ID
    pub source_id: u32,
}

/// Lock-free clause sharing for parallel SAT solving.
///
/// Multiple solver threads can concurrently export and import learned
/// clauses through this structure without any locks.
pub struct LockFreeClauseSharing {
    /// Queue of shared clauses
    clause_queue: LockFreeQueue<SharedClause>,
    /// Maximum clause length to share (longer clauses are not useful)
    max_clause_len: usize,
    /// Maximum LBD to share (higher LBD = lower quality)
    max_lbd: u32,
    /// Total clauses exported
    exported_count: AtomicU64,
    /// Total clauses imported
    imported_count: AtomicU64,
    /// Total clauses rejected (too long or low quality)
    rejected_count: AtomicU64,
}

impl LockFreeClauseSharing {
    /// Create a new clause sharing structure with default parameters
    pub fn new() -> Self {
        Self {
            clause_queue: LockFreeQueue::new(),
            max_clause_len: 8,
            max_lbd: 3,
            exported_count: AtomicU64::new(0),
            imported_count: AtomicU64::new(0),
            rejected_count: AtomicU64::new(0),
        }
    }

    /// Create with custom sharing parameters
    pub fn with_params(max_clause_len: usize, max_lbd: u32) -> Self {
        Self {
            clause_queue: LockFreeQueue::new(),
            max_clause_len,
            max_lbd,
            exported_count: AtomicU64::new(0),
            imported_count: AtomicU64::new(0),
            rejected_count: AtomicU64::new(0),
        }
    }

    /// Export a learned clause for sharing with other solvers.
    ///
    /// Returns true if the clause was accepted for sharing,
    /// false if it was rejected due to quality filters.
    pub fn export_clause(&self, literals: Vec<i32>, lbd: u32, source_id: u32) -> bool {
        // Quality filter: reject clauses that are too long or low quality
        if literals.len() > self.max_clause_len || lbd > self.max_lbd {
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let clause = SharedClause {
            literals,
            lbd,
            source_id,
        };
        self.clause_queue.push(clause);
        self.exported_count.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Import a shared clause from another solver.
    ///
    /// Returns the next available shared clause, or None if the queue is empty.
    pub fn import_clause(&self) -> Option<SharedClause> {
        let result = self.clause_queue.pop();
        if result.is_some() {
            self.imported_count.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// Import up to `max_count` clauses at once
    pub fn import_batch(&self, max_count: usize) -> Vec<SharedClause> {
        let mut batch = Vec::with_capacity(max_count);
        for _ in 0..max_count {
            match self.clause_queue.pop() {
                Some(clause) => {
                    self.imported_count.fetch_add(1, Ordering::Relaxed);
                    batch.push(clause);
                }
                None => break,
            }
        }
        batch
    }

    /// Check if there are clauses available to import
    pub fn has_clauses(&self) -> bool {
        !self.clause_queue.is_empty()
    }

    /// Get the number of clauses currently in the queue
    pub fn pending_count(&self) -> usize {
        self.clause_queue.len()
    }

    /// Get sharing statistics: (exported, imported, rejected)
    pub fn statistics(&self) -> (u64, u64, u64) {
        (
            self.exported_count.load(Ordering::Relaxed),
            self.imported_count.load(Ordering::Relaxed),
            self.rejected_count.load(Ordering::Relaxed),
        )
    }
}

impl Default for LockFreeClauseSharing {
    fn default() -> Self {
        Self::new()
    }
}
