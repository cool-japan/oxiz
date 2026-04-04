//! Clause allocation pool for recycling deleted clause memory.
//!
//! `ClausePool` maintains a set of freed clause slots bucketed by literal count.
//! When a new clause is requested, the pool attempts to find a recycled slot
//! of the matching size before falling back to fresh allocation. This reduces
//! allocator pressure during long CDCL searches with frequent clause deletion.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::clause::{Clause, ClauseTier};
use crate::literal::Lit;
use smallvec::SmallVec;

/// Size bucket boundaries for the clause pool.
/// Clauses are grouped into buckets by literal count:
///   bucket 0: 2 literals (binary)
///   bucket 1: 3 literals (ternary)
///   bucket 2: 4-7 literals
///   bucket 3: 8-15 literals
///   bucket 4: 16+ literals
const NUM_BUCKETS: usize = 5;

/// Map a literal count to a bucket index.
fn bucket_for_size(num_lits: usize) -> usize {
    match num_lits {
        0..=2 => 0,
        3 => 1,
        4..=7 => 2,
        8..=15 => 3,
        _ => 4,
    }
}

/// Statistics for the clause pool.
#[derive(Debug, Clone, Default)]
pub struct ClausePoolStats {
    /// Total clauses returned to the pool
    pub total_returned: usize,
    /// Total clauses reused from the pool
    pub total_reused: usize,
    /// Total fresh allocations (pool miss)
    pub total_fresh: usize,
    /// Current number of clauses available in the pool
    pub pool_size: usize,
    /// Per-bucket counts
    pub bucket_sizes: [usize; NUM_BUCKETS],
}

impl ClausePoolStats {
    /// Reuse ratio (0.0 to 1.0); returns 0 if nothing was ever allocated.
    #[must_use]
    pub fn reuse_ratio(&self) -> f64 {
        let total = self.total_reused + self.total_fresh;
        if total == 0 {
            return 0.0;
        }
        self.total_reused as f64 / total as f64
    }
}

/// A recycling pool for clause allocations.
///
/// Deleted clauses are placed into size-based buckets. When a new clause of
/// a given size is requested, the pool pops a matching recycled clause,
/// reinitialises its fields, and returns it without a fresh heap allocation.
#[derive(Debug)]
pub struct ClausePool {
    /// Free clause objects bucketed by size
    buckets: [Vec<Clause>; NUM_BUCKETS],
    /// Maximum number of clauses to keep per bucket
    max_per_bucket: usize,
    /// Statistics
    stats: ClausePoolStats,
}

impl Default for ClausePool {
    fn default() -> Self {
        Self::new()
    }
}

impl ClausePool {
    /// Create a new clause pool with default limits.
    #[must_use]
    pub fn new() -> Self {
        Self::with_max_per_bucket(512)
    }

    /// Create a new clause pool with a specific per-bucket limit.
    #[must_use]
    pub fn with_max_per_bucket(max_per_bucket: usize) -> Self {
        Self {
            buckets: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            max_per_bucket,
            stats: ClausePoolStats::default(),
        }
    }

    /// Return a deleted clause to the pool for later reuse.
    ///
    /// The clause's literal storage is cleared but the underlying allocation
    /// (`SmallVec` heap buffer) is preserved so it can be refilled cheaply.
    pub fn recycle(&mut self, mut clause: Clause) {
        let bucket = bucket_for_size(clause.lits.len());

        if self.buckets[bucket].len() >= self.max_per_bucket {
            // Bucket is full; drop the clause instead of pooling it
            return;
        }

        // Clear mutable state but keep allocation
        clause.lits.clear();
        clause.learned = false;
        clause.activity = 0.0;
        clause.lbd = 0;
        clause.deleted = false;
        clause.tier = ClauseTier::Local;
        clause.usage_count = 0;

        self.buckets[bucket].push(clause);
        self.stats.total_returned += 1;
        self.stats.pool_size += 1;
        self.stats.bucket_sizes[bucket] += 1;
    }

    /// Try to obtain a recycled clause for the given literal list.
    ///
    /// If a matching-size clause is available in the pool, it is reinitialised
    /// with the provided literals and `learned` flag. Otherwise `None` is
    /// returned and the caller should allocate a fresh `Clause`.
    pub fn acquire(&mut self, lits: &[Lit], learned: bool) -> Option<Clause> {
        let bucket = bucket_for_size(lits.len());

        if let Some(mut clause) = self.buckets[bucket].pop() {
            clause.lits.extend_from_slice(lits);
            clause.learned = learned;
            clause.activity = 0.0;
            clause.lbd = 0;
            clause.deleted = false;
            clause.tier = ClauseTier::Local;
            clause.usage_count = 0;

            self.stats.total_reused += 1;
            self.stats.pool_size = self.stats.pool_size.saturating_sub(1);
            self.stats.bucket_sizes[bucket] = self.stats.bucket_sizes[bucket].saturating_sub(1);

            Some(clause)
        } else {
            self.stats.total_fresh += 1;
            None
        }
    }

    /// Build a clause (either recycled or fresh) for the given literals.
    pub fn make_clause(&mut self, lits: impl IntoIterator<Item = Lit>, learned: bool) -> Clause {
        let collected: SmallVec<[Lit; 4]> = lits.into_iter().collect();
        if let Some(clause) = self.acquire(&collected, learned) {
            clause
        } else {
            Clause::new(collected, learned)
        }
    }

    /// Get pool statistics.
    #[must_use]
    pub fn stats(&self) -> &ClausePoolStats {
        &self.stats
    }

    /// Current total number of pooled (available) clauses.
    #[must_use]
    pub fn pool_size(&self) -> usize {
        self.stats.pool_size
    }

    /// Shrink the pool to at most `max` entries across all buckets.
    pub fn shrink_to(&mut self, max: usize) {
        let mut total = self.pool_size();
        if total <= max {
            return;
        }

        // Drain from the largest buckets first
        for bucket in (0..NUM_BUCKETS).rev() {
            while total > max {
                if self.buckets[bucket].pop().is_some() {
                    total -= 1;
                    self.stats.pool_size -= 1;
                    self.stats.bucket_sizes[bucket] =
                        self.stats.bucket_sizes[bucket].saturating_sub(1);
                } else {
                    break;
                }
            }
        }
    }

    /// Clear all pooled clauses.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.stats.pool_size = 0;
        self.stats.bucket_sizes = [0; NUM_BUCKETS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    fn lits(vars: &[u32], signs: &[bool]) -> Vec<Lit> {
        vars.iter()
            .zip(signs.iter())
            .map(|(&v, &s)| {
                if s {
                    Lit::pos(Var::new(v))
                } else {
                    Lit::neg(Var::new(v))
                }
            })
            .collect()
    }

    #[test]
    fn test_pool_creation() {
        let pool = ClausePool::new();
        assert_eq!(pool.pool_size(), 0);
        assert_eq!(pool.stats().total_returned, 0);
        assert_eq!(pool.stats().total_reused, 0);
    }

    #[test]
    fn test_recycle_and_acquire() {
        let mut pool = ClausePool::new();

        // Create a clause and recycle it
        let clause = Clause::learned(lits(&[0, 1, 2], &[true, false, true]));
        assert_eq!(clause.len(), 3);
        pool.recycle(clause);

        assert_eq!(pool.pool_size(), 1);
        assert_eq!(pool.stats().total_returned, 1);

        // Acquire should return a recycled clause for same-bucket size
        let new_lits = lits(&[3, 4, 5], &[false, true, false]);
        let reused = pool.acquire(&new_lits, true);
        assert!(reused.is_some());

        let reused = reused.expect("should have a reused clause");
        assert_eq!(reused.len(), 3);
        assert!(reused.learned);
        assert_eq!(pool.stats().total_reused, 1);
    }

    #[test]
    fn test_acquire_empty_pool() {
        let mut pool = ClausePool::new();
        let new_lits = lits(&[0, 1], &[true, false]);

        let result = pool.acquire(&new_lits, false);
        assert!(result.is_none());
        assert_eq!(pool.stats().total_fresh, 1);
    }

    #[test]
    fn test_make_clause_reuses() {
        let mut pool = ClausePool::new();

        // Recycle a binary clause
        let clause = Clause::original(lits(&[0, 1], &[true, true]));
        pool.recycle(clause);

        // make_clause should reuse it
        let made = pool.make_clause(lits(&[2, 3], &[false, false]), false);
        assert_eq!(made.len(), 2);
        assert!(!made.learned);
        assert_eq!(pool.stats().total_reused, 1);
    }

    #[test]
    fn test_make_clause_fresh() {
        let mut pool = ClausePool::new();

        let made = pool.make_clause(lits(&[0, 1, 2, 3], &[true, true, false, false]), true);
        assert_eq!(made.len(), 4);
        assert!(made.learned);
        assert_eq!(pool.stats().total_fresh, 1);
        assert_eq!(pool.stats().total_reused, 0);
    }

    #[test]
    fn test_pool_stats_reuse_ratio() {
        let mut pool = ClausePool::new();

        // 1 fresh
        let _ = pool.make_clause(lits(&[0, 1], &[true, false]), false);

        // Recycle something
        let c = Clause::original(lits(&[2, 3], &[true, true]));
        pool.recycle(c);

        // 1 reused
        let _ = pool.make_clause(lits(&[4, 5], &[false, false]), false);

        let stats = pool.stats();
        assert_eq!(stats.total_fresh, 1);
        assert_eq!(stats.total_reused, 1);
        // reuse_ratio = 1 / (1+1) = 0.5
        let ratio = stats.reuse_ratio();
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_shrink_to() {
        let mut pool = ClausePool::new();

        // Fill with 10 clauses
        for i in 0..10u32 {
            let c = Clause::original(lits(&[i, i + 10], &[true, false]));
            pool.recycle(c);
        }
        assert_eq!(pool.pool_size(), 10);

        pool.shrink_to(5);
        assert!(pool.pool_size() <= 5);
    }

    #[test]
    fn test_clear_pool() {
        let mut pool = ClausePool::new();

        for i in 0..5u32 {
            let c = Clause::learned(lits(&[i, i + 1, i + 2], &[true, false, true]));
            pool.recycle(c);
        }
        assert_eq!(pool.pool_size(), 5);

        pool.clear();
        assert_eq!(pool.pool_size(), 0);
    }

    #[test]
    fn test_bucket_overflow() {
        let mut pool = ClausePool::with_max_per_bucket(2);

        // Recycle 3 binary clauses; only 2 should be kept
        for i in 0..3u32 {
            let c = Clause::original(lits(&[i, i + 10], &[true, false]));
            pool.recycle(c);
        }

        // Only 2 should be in the pool (max_per_bucket = 2)
        assert!(pool.pool_size() <= 2);
    }

    #[test]
    fn test_different_size_buckets() {
        let mut pool = ClausePool::new();

        // Recycle a binary clause (bucket 0)
        pool.recycle(Clause::original(lits(&[0, 1], &[true, false])));

        // Recycle a ternary clause (bucket 1)
        pool.recycle(Clause::original(lits(&[0, 1, 2], &[true, false, true])));

        assert_eq!(pool.pool_size(), 2);

        // Acquiring a ternary shouldn't get the binary
        let acquired = pool.acquire(&lits(&[3, 4, 5], &[false, true, false]), false);
        assert!(acquired.is_some());
        assert_eq!(pool.pool_size(), 1);

        // The remaining one should be the binary
        let acquired2 = pool.acquire(&lits(&[6, 7], &[true, true]), false);
        assert!(acquired2.is_some());
        assert_eq!(pool.pool_size(), 0);
    }
}
