//! Arena allocator for AST term allocation
//!
//! This module provides a `TermArena` backed by bumpalo for fast, bulk
//! allocation of terms. When the `arena` feature is enabled, `TermManager`
//! routes new allocations through this arena for improved cache locality
//! and reduced allocation overhead.

use super::term::{Term, TermId, TermKind};
use crate::sort::SortId;
use bumpalo::Bump;

/// Arena allocator for efficient term allocation.
///
/// Wraps a `bumpalo::Bump` allocator to provide fast, arena-based term
/// allocation. All allocations are freed together when the arena is dropped,
/// which avoids per-object deallocation overhead.
pub struct TermArena {
    /// The underlying bump allocator
    bump: Bump,
    /// Number of terms allocated through this arena
    alloc_count: usize,
    /// Total bytes allocated
    bytes_allocated: usize,
}

impl core::fmt::Debug for TermArena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TermArena")
            .field("alloc_count", &self.alloc_count)
            .field("bytes_allocated", &self.bytes_allocated)
            .field("bump_allocated", &self.bump.allocated_bytes())
            .finish()
    }
}

/// Statistics from the arena allocator
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Number of individual term allocations
    pub alloc_count: usize,
    /// Approximate bytes used by arena-allocated terms
    pub bytes_allocated: usize,
    /// Total bytes allocated by the bump allocator (includes padding/overhead)
    pub bump_allocated_bytes: usize,
}

impl TermArena {
    /// Create a new term arena with default capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bump: Bump::with_capacity(64 * 1024), // 64 KiB initial
            alloc_count: 0,
            bytes_allocated: 0,
        }
    }

    /// Create a new term arena with specified initial capacity in bytes.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bump: Bump::with_capacity(capacity),
            alloc_count: 0,
            bytes_allocated: 0,
        }
    }

    /// Allocate a term in the arena and return a reference to it.
    ///
    /// The returned reference is valid for the lifetime of the arena.
    pub fn alloc_term(&mut self, id: TermId, kind: TermKind, sort: SortId) -> &Term {
        let term = Term { id, kind, sort };
        self.alloc_count += 1;
        self.bytes_allocated += core::mem::size_of::<Term>();
        self.bump.alloc(term)
    }

    /// Allocate a slice of TermIds in the arena.
    ///
    /// Useful for storing child references contiguously.
    pub fn alloc_slice(&mut self, ids: &[TermId]) -> &[TermId] {
        if ids.is_empty() {
            return &[];
        }
        self.bytes_allocated += core::mem::size_of_val(ids);
        self.bump.alloc_slice_copy(ids)
    }

    /// Allocate a vector of terms in the arena, returning a slice.
    pub fn alloc_term_vec(&mut self, terms: Vec<Term>) -> &[Term] {
        if terms.is_empty() {
            return &[];
        }
        self.alloc_count += terms.len();
        self.bytes_allocated += core::mem::size_of::<Term>() * terms.len();
        self.bump.alloc_slice_fill_iter(terms)
    }

    /// Get arena allocation statistics.
    #[must_use]
    pub fn stats(&self) -> ArenaStats {
        ArenaStats {
            alloc_count: self.alloc_count,
            bytes_allocated: self.bytes_allocated,
            bump_allocated_bytes: self.bump.allocated_bytes(),
        }
    }

    /// Reset the arena, deallocating all memory.
    ///
    /// After calling this, all references obtained from the arena are invalid.
    /// This is useful for bulk cleanup between solving phases.
    pub fn reset(&mut self) {
        self.bump.reset();
        self.alloc_count = 0;
        self.bytes_allocated = 0;
    }

    /// Get the number of allocations.
    #[must_use]
    pub fn alloc_count(&self) -> usize {
        self.alloc_count
    }

    /// Get the total bytes allocated by the bump allocator.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }
}

impl Default for TermArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_arena_creation() {
        let arena = TermArena::new();
        assert_eq!(arena.alloc_count(), 0);
    }

    #[test]
    fn test_arena_with_capacity() {
        let arena = TermArena::with_capacity(1024);
        assert_eq!(arena.alloc_count(), 0);
        // Bump pre-allocates at least the requested capacity
        assert!(arena.allocated_bytes() >= 1024);
    }

    #[test]
    fn test_arena_alloc_term() {
        let mut arena = TermArena::new();
        let tm = TermManager::new();
        let sort = tm.sorts.bool_sort;

        let term_ref = arena.alloc_term(TermId(0), TermKind::True, sort);
        assert_eq!(term_ref.id, TermId(0));
        assert_eq!(term_ref.kind, TermKind::True);
        assert_eq!(arena.alloc_count(), 1);
    }

    #[test]
    fn test_arena_alloc_slice() {
        let mut arena = TermArena::new();
        let ids = [TermId(0), TermId(1), TermId(2), TermId(3)];
        let slice_ref = arena.alloc_slice(&ids);
        assert_eq!(slice_ref.len(), 4);
        assert_eq!(slice_ref[0], TermId(0));
        assert_eq!(slice_ref[3], TermId(3));
    }

    #[test]
    fn test_arena_alloc_empty_slice() {
        let mut arena = TermArena::new();
        let slice_ref = arena.alloc_slice(&[]);
        assert!(slice_ref.is_empty());
    }

    #[test]
    fn test_arena_stats() {
        let mut arena = TermArena::new();
        let tm = TermManager::new();
        let sort = tm.sorts.int_sort;

        for i in 0..100 {
            let _ = arena.alloc_term(TermId(i), TermKind::True, sort);
        }

        let stats = arena.stats();
        assert_eq!(stats.alloc_count, 100);
        assert!(stats.bytes_allocated > 0);
        assert!(stats.bump_allocated_bytes > 0);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = TermArena::new();
        let tm = TermManager::new();
        let sort = tm.sorts.bool_sort;

        for i in 0..50 {
            let _ = arena.alloc_term(TermId(i), TermKind::True, sort);
        }
        assert_eq!(arena.alloc_count(), 50);

        arena.reset();
        assert_eq!(arena.alloc_count(), 0);
        assert_eq!(arena.stats().bytes_allocated, 0);
    }

    #[test]
    fn test_arena_many_allocations() {
        let mut arena = TermArena::with_capacity(4096);
        let tm = TermManager::new();
        let sort = tm.sorts.bool_sort;

        // Allocate many terms to test growth
        for i in 0..1000 {
            let _ = arena.alloc_term(TermId(i), TermKind::True, sort);
        }
        assert_eq!(arena.alloc_count(), 1000);
        assert!(arena.allocated_bytes() > 0);
    }

    #[test]
    fn test_arena_integration_with_term_manager() {
        // When the arena feature is on, TermManager should still work correctly
        let mut tm = TermManager::new();
        let x = tm.mk_var("x", tm.sorts.int_sort);
        let y = tm.mk_var("y", tm.sorts.int_sort);
        let sum = tm.mk_add([x, y]);

        // Arena-based allocation should not affect interning semantics
        let x2 = tm.mk_var("x", tm.sorts.int_sort);
        assert_eq!(x, x2);

        // term retrieval still works
        let term = tm.get(sum);
        assert!(term.is_some());
    }
}
