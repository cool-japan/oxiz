//! Memory pool for efficient term allocation
//!
//! This module provides a chunked memory pool that allocates terms in blocks
//! for better cache locality and reduced allocation overhead.

use super::term::Term;

/// Size of each chunk in the pool (in number of terms)
const CHUNK_SIZE: usize = 4096;

/// A single chunk of pre-allocated terms
struct Chunk {
    /// The actual storage for terms
    terms: Vec<Term>,
    /// Number of terms currently used in this chunk
    used: usize,
}

impl Chunk {
    /// Create a new chunk with the specified capacity
    fn new(capacity: usize) -> Self {
        Self {
            terms: Vec::with_capacity(capacity),
            used: 0,
        }
    }

    /// Try to allocate a term in this chunk
    /// Returns the index if successful, None if chunk is full
    fn try_alloc(&mut self, term: Term) -> Option<usize> {
        if self.used < self.terms.capacity() {
            let idx = self.used;
            if self.terms.len() <= idx {
                self.terms.push(term);
            } else {
                self.terms[idx] = term;
            }
            self.used += 1;
            Some(idx)
        } else {
            None
        }
    }

    /// Get a term by index within this chunk
    fn get(&self, idx: usize) -> Option<&Term> {
        self.terms.get(idx)
    }

    /// Check if this chunk is full
    #[allow(dead_code)]
    fn is_full(&self) -> bool {
        self.used >= self.terms.capacity()
    }

    /// Get the number of used slots
    fn len(&self) -> usize {
        self.used
    }

    /// Get memory usage statistics for this chunk
    fn memory_usage(&self) -> MemoryStats {
        MemoryStats {
            allocated_bytes: self.terms.capacity() * std::mem::size_of::<Term>(),
            used_bytes: self.used * std::mem::size_of::<Term>(),
            num_terms: self.used,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryStats {
    /// Total allocated bytes
    pub allocated_bytes: usize,
    /// Bytes currently in use
    pub used_bytes: usize,
    /// Number of terms
    pub num_terms: usize,
}

impl MemoryStats {
    /// Calculate fragmentation ratio (0.0 = no fragmentation, 1.0 = maximum fragmentation)
    pub fn fragmentation(&self) -> f64 {
        if self.allocated_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.used_bytes as f64 / self.allocated_bytes as f64)
    }
}

/// Chunked memory pool for term allocation
///
/// Allocates terms in large blocks (chunks) to reduce allocation overhead
/// and improve cache locality.
#[derive(Default)]
pub struct TermPool {
    /// All allocated chunks
    chunks: Vec<Chunk>,
    /// Total number of terms allocated
    total_terms: usize,
}

impl TermPool {
    /// Create a new empty term pool
    pub fn new() -> Self {
        Self {
            chunks: vec![Chunk::new(CHUNK_SIZE)],
            total_terms: 0,
        }
    }

    /// Create a pool with a specified initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        let num_chunks = capacity.div_ceil(CHUNK_SIZE);
        let mut chunks = Vec::with_capacity(num_chunks);
        chunks.push(Chunk::new(CHUNK_SIZE));

        Self {
            chunks,
            total_terms: 0,
        }
    }

    /// Allocate a term in the pool
    ///
    /// Returns a global index that can be used to retrieve the term later.
    pub fn alloc(&mut self, term: Term) -> usize {
        // Try to allocate in the last chunk first (hot path)
        if let Some(last_chunk) = self.chunks.last_mut()
            && let Some(_idx) = last_chunk.try_alloc(term.clone())
        {
            let global_idx = self.total_terms;
            self.total_terms += 1;
            return global_idx;
        }

        // Last chunk is full, allocate a new chunk
        let mut new_chunk = Chunk::new(CHUNK_SIZE);
        new_chunk
            .try_alloc(term)
            .expect("New chunk should have space");
        self.chunks.push(new_chunk);

        let global_idx = self.total_terms;
        self.total_terms += 1;
        global_idx
    }

    /// Get a term by its global index
    pub fn get(&self, global_idx: usize) -> Option<&Term> {
        // Find which chunk contains this index
        let mut accumulated = 0;
        for chunk in &self.chunks {
            let chunk_size = chunk.len();
            if global_idx < accumulated + chunk_size {
                let local_idx = global_idx - accumulated;
                return chunk.get(local_idx);
            }
            accumulated += chunk_size;
        }

        None
    }

    /// Get the total number of terms in the pool
    pub fn len(&self) -> usize {
        self.total_terms
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.total_terms == 0
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let mut total_stats = MemoryStats {
            allocated_bytes: 0,
            used_bytes: 0,
            num_terms: 0,
        };

        for chunk in &self.chunks {
            let chunk_stats = chunk.memory_usage();
            total_stats.allocated_bytes += chunk_stats.allocated_bytes;
            total_stats.used_bytes += chunk_stats.used_bytes;
            total_stats.num_terms += chunk_stats.num_terms;
        }

        total_stats
    }

    /// Get the number of chunks allocated
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Clear the pool (for testing/benchmarking)
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.chunks.push(Chunk::new(CHUNK_SIZE));
        self.total_terms = 0;
    }
}

// TermPool is safe to use in single-threaded scenarios

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{TermId, TermKind};
    use crate::sort::SortManager;

    fn make_test_term(id: u32) -> Term {
        use num_bigint::BigInt;
        let sorts = SortManager::new();
        Term {
            id: TermId(id),
            kind: TermKind::IntConst(BigInt::from(id)),
            sort: sorts.int_sort,
        }
    }

    #[test]
    fn test_pool_creation() {
        let pool = TermPool::new();
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert_eq!(pool.num_chunks(), 1);
    }

    #[test]
    fn test_single_allocation() {
        let mut pool = TermPool::new();
        let term = make_test_term(1);

        let idx = pool.alloc(term.clone());
        assert_eq!(idx, 0);
        assert_eq!(pool.len(), 1);

        let retrieved = pool.get(idx).unwrap();
        assert_eq!(retrieved.id, term.id);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut pool = TermPool::new();

        for i in 0..100 {
            let term = make_test_term(i);
            let idx = pool.alloc(term.clone());
            assert_eq!(idx, i as usize);
        }

        assert_eq!(pool.len(), 100);

        // Verify all terms
        for i in 0..100 {
            let term = pool.get(i).unwrap();
            assert_eq!(term.id.0, i as u32);
        }
    }

    #[test]
    fn test_chunk_overflow() {
        let mut pool = TermPool::new();

        // Allocate more than one chunk worth of terms
        let num_terms = CHUNK_SIZE + 100;
        for i in 0..num_terms {
            let term = make_test_term(i as u32);
            pool.alloc(term);
        }

        assert_eq!(pool.len(), num_terms);
        assert!(pool.num_chunks() >= 2);
    }

    #[test]
    fn test_memory_stats() {
        let mut pool = TermPool::new();

        for i in 0..10 {
            let term = make_test_term(i);
            pool.alloc(term);
        }

        let stats = pool.memory_stats();
        assert_eq!(stats.num_terms, 10);
        assert!(stats.allocated_bytes >= stats.used_bytes);
        assert!(stats.fragmentation() > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut pool = TermPool::new();

        for i in 0..100 {
            pool.alloc(make_test_term(i));
        }

        assert_eq!(pool.len(), 100);

        pool.clear();

        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        assert_eq!(pool.num_chunks(), 1);
    }

    #[test]
    fn test_with_capacity() {
        let mut pool = TermPool::with_capacity(10000);

        // Should have pre-allocated chunks
        let initial_chunks = pool.num_chunks();
        assert!(initial_chunks >= 1);

        // Allocate some terms
        for i in 0..100 {
            pool.alloc(make_test_term(i));
        }

        // Should still be in initial chunks
        assert_eq!(pool.num_chunks(), initial_chunks);
    }
}
