//! Parallel term operations using Rayon
//!
//! This module provides parallel implementations of common term operations
//! for improved performance on multi-core systems. Operations are automatically
//! parallelized when the input size exceeds a threshold.
//!
//! Reference: Rust's rayon library for data parallelism

use crate::ast::TermId;
use rayon::prelude::*;

/// Threshold for parallelization (terms below this are processed sequentially)
const PARALLEL_THRESHOLD: usize = 100;

/// Parallel reduce operation
///
/// Reduces a collection of terms using a binary operation in parallel.
pub fn parallel_reduce<F>(terms: &[TermId], identity: TermId, op: F) -> TermId
where
    F: Fn(TermId, TermId) -> TermId + Sync,
{
    if terms.is_empty() {
        return identity;
    }

    if terms.len() < PARALLEL_THRESHOLD {
        // Sequential reduce
        terms.iter().fold(identity, |acc, &term| op(acc, term))
    } else {
        // Parallel reduce
        terms
            .par_iter()
            .fold(|| identity, |acc, &term| op(acc, term))
            .reduce(|| identity, &op)
    }
}

/// Parallel find operation
///
/// Finds the first term matching a predicate using parallel search.
pub fn parallel_find<F>(terms: &[TermId], predicate: F) -> Option<TermId>
where
    F: Fn(TermId) -> bool + Sync,
{
    if terms.len() < PARALLEL_THRESHOLD {
        terms.iter().find(|&&t| predicate(t)).copied()
    } else {
        terms.par_iter().find_any(|&&t| predicate(t)).copied()
    }
}

/// Check if terms are pairwise distinct in parallel
///
/// Returns true if all terms are unique.
#[must_use]
pub fn parallel_all_distinct(terms: &[TermId]) -> bool {
    if terms.len() < 2 {
        return true;
    }

    if terms.len() < PARALLEL_THRESHOLD {
        // Sequential check using HashSet
        use rustc_hash::FxHashSet;
        let mut seen = FxHashSet::default();
        terms.iter().all(|t| seen.insert(*t))
    } else {
        // Parallel check: compare all pairs
        // For large sets, this is still efficient due to early termination
        terms
            .par_iter()
            .enumerate()
            .all(|(i, &ti)| terms[i + 1..].par_iter().all(|&tj| ti != tj))
    }
}

/// Filter terms that satisfy a predicate in parallel
///
/// Returns terms where the predicate returns true.
pub fn parallel_filter<F>(terms: &[TermId], predicate: F) -> Vec<TermId>
where
    F: Fn(TermId) -> bool + Sync,
{
    if terms.len() < PARALLEL_THRESHOLD {
        terms.iter().copied().filter(|&t| predicate(t)).collect()
    } else {
        terms
            .par_iter()
            .copied()
            .filter(|&t| predicate(t))
            .collect()
    }
}

/// Map a function over terms in parallel
///
/// Applies the function to each term and collects results.
pub fn parallel_map<F, R>(terms: &[TermId], func: F) -> Vec<R>
where
    F: Fn(TermId) -> R + Sync,
    R: Send,
{
    if terms.len() < PARALLEL_THRESHOLD {
        terms.iter().map(|&t| func(t)).collect()
    } else {
        terms.par_iter().map(|&t| func(t)).collect()
    }
}

/// Partition terms based on a predicate in parallel
///
/// Returns two vectors: (matching, non-matching).
pub fn parallel_partition<F>(terms: &[TermId], predicate: F) -> (Vec<TermId>, Vec<TermId>)
where
    F: Fn(TermId) -> bool + Sync,
{
    if terms.len() < PARALLEL_THRESHOLD {
        terms.iter().copied().partition(|&t| predicate(t))
    } else {
        terms.par_iter().copied().partition(|&t| predicate(t))
    }
}

/// Count terms matching a predicate in parallel
pub fn parallel_count<F>(terms: &[TermId], predicate: F) -> usize
where
    F: Fn(TermId) -> bool + Sync,
{
    if terms.len() < PARALLEL_THRESHOLD {
        terms.iter().filter(|&&t| predicate(t)).count()
    } else {
        terms.par_iter().filter(|&&t| predicate(t)).count()
    }
}

/// Check if any term matches the predicate in parallel
pub fn parallel_any<F>(terms: &[TermId], predicate: F) -> bool
where
    F: Fn(TermId) -> bool + Sync,
{
    if terms.len() < PARALLEL_THRESHOLD {
        terms.iter().any(|&t| predicate(t))
    } else {
        terms.par_iter().any(|&t| predicate(t))
    }
}

/// Check if all terms match the predicate in parallel
pub fn parallel_all<F>(terms: &[TermId], predicate: F) -> bool
where
    F: Fn(TermId) -> bool + Sync,
{
    if terms.len() < PARALLEL_THRESHOLD {
        terms.iter().all(|&t| predicate(t))
    } else {
        terms.par_iter().all(|&t| predicate(t))
    }
}

/// Statistics for parallel operations
#[derive(Debug, Clone, Default)]
pub struct ParallelStats {
    /// Number of parallel operations performed
    pub parallel_ops: usize,
    /// Number of sequential operations performed
    pub sequential_ops: usize,
    /// Total terms processed
    pub terms_processed: usize,
}

impl ParallelStats {
    /// Create new statistics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the parallelization ratio
    #[must_use]
    pub fn parallel_ratio(&self) -> f64 {
        let total = self.parallel_ops + self.sequential_ops;
        if total > 0 {
            self.parallel_ops as f64 / total as f64
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for ParallelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Parallel Statistics:")?;
        writeln!(f, "  Parallel ops:    {}", self.parallel_ops)?;
        writeln!(f, "  Sequential ops:  {}", self.sequential_ops)?;
        writeln!(f, "  Terms processed: {}", self.terms_processed)?;
        writeln!(
            f,
            "  Parallel ratio:  {:.2}%",
            self.parallel_ratio() * 100.0
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_parallel_reduce() {
        let mut tm = TermManager::new();

        let terms = vec![tm.mk_int(1), tm.mk_int(2), tm.mk_int(3)];
        let identity = tm.mk_int(0);

        let result = parallel_reduce(&terms, identity, |a, _b| a);
        assert_eq!(result, identity);
    }

    #[test]
    fn test_parallel_find() {
        let mut tm = TermManager::new();

        let t1 = tm.mk_int(1);
        let t2 = tm.mk_int(2);
        let t3 = tm.mk_int(3);
        let terms = vec![t1, t2, t3];

        let found = parallel_find(&terms, |term| term == t2);
        assert_eq!(found, Some(t2));

        let t99 = tm.mk_int(99);
        let not_found = parallel_find(&terms, |term| term == t99);
        assert_eq!(not_found, None);
    }

    #[test]
    fn test_parallel_all_distinct_empty() {
        assert!(parallel_all_distinct(&[]));
    }

    #[test]
    fn test_parallel_all_distinct_single() {
        let mut tm = TermManager::new();
        let t = tm.mk_int(1);
        assert!(parallel_all_distinct(&[t]));
    }

    #[test]
    fn test_parallel_all_distinct_true() {
        let mut tm = TermManager::new();
        let terms = vec![tm.mk_int(1), tm.mk_int(2), tm.mk_int(3)];
        assert!(parallel_all_distinct(&terms));
    }

    #[test]
    fn test_parallel_all_distinct_false() {
        let mut tm = TermManager::new();
        let t1 = tm.mk_int(1);
        let t2 = tm.mk_int(2);
        let terms = vec![t1, t2, t1];
        assert!(!parallel_all_distinct(&terms));
    }

    #[test]
    fn test_parallel_filter() {
        let mut tm = TermManager::new();

        let t1 = tm.mk_int(1);
        let t2 = tm.mk_int(2);
        let t3 = tm.mk_int(3);
        let terms = vec![t1, t2, t3];

        let filtered = parallel_filter(&terms, |term| term == t2 || term == t3);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_parallel_map() {
        let mut tm = TermManager::new();

        let terms = vec![tm.mk_int(1), tm.mk_int(2), tm.mk_int(3)];
        let mapped = parallel_map(&terms, |t| t.0);
        assert_eq!(mapped.len(), 3);
    }

    #[test]
    fn test_parallel_partition() {
        let mut tm = TermManager::new();

        let t1 = tm.mk_int(1);
        let t2 = tm.mk_int(2);
        let t3 = tm.mk_int(3);
        let terms = vec![t1, t2, t3];

        let (small, large) = parallel_partition(&terms, |term| term == t1);
        assert_eq!(small.len(), 1);
        assert_eq!(large.len(), 2);
    }

    #[test]
    fn test_parallel_count() {
        let mut tm = TermManager::new();

        let t1 = tm.mk_int(1);
        let t2 = tm.mk_int(2);
        let terms = vec![t1, t2, t1, t2, t1];

        let count = parallel_count(&terms, |term| term == t1);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_parallel_any() {
        let mut tm = TermManager::new();

        let t1 = tm.mk_int(1);
        let t2 = tm.mk_int(2);
        let t99 = tm.mk_int(99);
        let terms = vec![t1, t2];

        assert!(parallel_any(&terms, |term| term == t2));
        assert!(!parallel_any(&terms, |term| term == t99));
    }

    #[test]
    fn test_parallel_all() {
        let mut tm = TermManager::new();

        let t1 = tm.mk_int(1);
        let t2 = tm.mk_int(2);
        let terms = vec![t1, t1, t1];

        assert!(parallel_all(&terms, |term| term == t1));
        assert!(!parallel_all(&terms, |term| term == t2));
    }

    #[test]
    fn test_parallel_stats() {
        let mut stats = ParallelStats::new();
        stats.parallel_ops = 80;
        stats.sequential_ops = 20;
        stats.terms_processed = 1000;

        assert_eq!(stats.parallel_ratio(), 0.8);
    }

    #[test]
    fn test_parallel_stats_display() {
        let stats = ParallelStats {
            parallel_ops: 100,
            sequential_ops: 50,
            terms_processed: 5000,
        };

        let display = format!("{}", stats);
        assert!(display.contains("Parallel Statistics"));
        assert!(display.contains("100"));
    }
}
