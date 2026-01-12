//! Two-watched literal scheme

use crate::clause::ClauseId;
use crate::literal::Lit;
use smallvec::SmallVec;

/// A watcher entry
#[derive(Debug, Clone, Copy)]
pub struct Watcher {
    /// The clause being watched
    pub clause: ClauseId,
    /// The other watched literal (blocking literal)
    pub blocker: Lit,
}

impl Watcher {
    /// Create a new watcher
    #[must_use]
    pub const fn new(clause: ClauseId, blocker: Lit) -> Self {
        Self { clause, blocker }
    }
}

/// Watch lists for the two-watched literal scheme
#[derive(Debug)]
pub struct WatchLists {
    /// Watch list for each literal
    watches: Vec<SmallVec<[Watcher; 8]>>,
}

impl WatchLists {
    /// Create new watch lists for n variables
    #[must_use]
    pub fn new(num_vars: usize) -> Self {
        Self {
            watches: vec![SmallVec::new(); num_vars * 2],
        }
    }

    /// Add a watcher for a literal
    pub fn add(&mut self, lit: Lit, watcher: Watcher) {
        let idx = lit.index();
        if idx >= self.watches.len() {
            self.watches.resize(idx + 1, SmallVec::new());
        }
        self.watches[idx].push(watcher);
    }

    /// Get the watch list for a literal
    #[must_use]
    #[allow(dead_code)]
    pub fn get(&self, lit: Lit) -> &[Watcher] {
        self.watches.get(lit.index()).map_or(&[], |w| w.as_slice())
    }

    /// Get mutable access to the watch list for a literal
    pub fn get_mut(&mut self, lit: Lit) -> &mut SmallVec<[Watcher; 8]> {
        let idx = lit.index();
        if idx >= self.watches.len() {
            self.watches.resize(idx + 1, SmallVec::new());
        }
        &mut self.watches[idx]
    }

    /// Remove all watchers for a clause from a literal's watch list
    #[allow(dead_code)]
    pub fn remove_clause(&mut self, lit: Lit, clause: ClauseId) {
        let idx = lit.index();
        if idx < self.watches.len() {
            self.watches[idx].retain(|w| w.clause != clause);
        }
    }

    /// Resize to support more variables
    pub fn resize(&mut self, num_vars: usize) {
        let new_size = num_vars * 2;
        if new_size > self.watches.len() {
            self.watches.resize(new_size, SmallVec::new());
        }
    }

    /// Clear all watch lists
    pub fn clear(&mut self) {
        for watches in &mut self.watches {
            watches.clear();
        }
    }

    /// Get the number of watchers for a literal
    #[must_use]
    #[allow(dead_code)]
    pub fn count(&self, lit: Lit) -> usize {
        self.watches.get(lit.index()).map_or(0, |w| w.len())
    }
}

/// SIMD-optimized utilities for watched literal processing
///
/// These functions are designed to be auto-vectorized by LLVM for better performance.
/// When compiled with appropriate flags (e.g., -C target-cpu=native), these operations
/// can use SIMD instructions (SSE, AVX, etc.) automatically.
pub mod simd_utils {
    use super::*;
    use crate::literal::LBool;

    /// Check multiple blockers in parallel (optimized for auto-vectorization)
    ///
    /// This function processes watchers in batches and is designed to be
    /// auto-vectorized by LLVM. The compiler can generate SIMD instructions
    /// for the blocker checking when optimization is enabled.
    ///
    /// # Arguments
    /// * `watchers` - Slice of watchers to check
    /// * `lit_values` - Function to get the value of a literal
    ///
    /// # Returns
    /// Indices of watchers that need propagation (blocker is not true)
    #[inline]
    #[allow(dead_code)]
    pub fn find_non_satisfied_watchers<F>(
        watchers: &[Watcher],
        mut lit_values: F,
    ) -> SmallVec<[usize; 16]>
    where
        F: FnMut(Lit) -> LBool,
    {
        let mut result = SmallVec::new();

        // Process in chunks to enable better vectorization
        const CHUNK_SIZE: usize = 8;

        let mut i = 0;
        while i + CHUNK_SIZE <= watchers.len() {
            // Check blockers in batch (LLVM can vectorize this)
            for j in 0..CHUNK_SIZE {
                let watcher = &watchers[i + j];
                if !lit_values(watcher.blocker).is_true() {
                    result.push(i + j);
                }
            }
            i += CHUNK_SIZE;
        }

        // Process remaining watchers
        while i < watchers.len() {
            let watcher = &watchers[i];
            if !lit_values(watcher.blocker).is_true() {
                result.push(i);
            }
            i += 1;
        }

        result
    }

    /// Batch check if literals are satisfied (optimized for auto-vectorization)
    ///
    /// This is optimized for SIMD by processing literals in aligned chunks.
    ///
    /// # Arguments
    /// * `lits` - Slice of literals to check
    /// * `lit_values` - Function to get the value of a literal
    ///
    /// # Returns
    /// true if any literal is satisfied (true)
    #[inline]
    #[allow(dead_code)]
    pub fn any_satisfied<F>(lits: &[Lit], mut lit_values: F) -> bool
    where
        F: FnMut(Lit) -> LBool,
    {
        // Process in chunks for better vectorization
        const CHUNK_SIZE: usize = 8;

        let chunks = lits.chunks(CHUNK_SIZE);
        for chunk in chunks {
            // Check chunk (can be vectorized)
            for &lit in chunk {
                if lit_values(lit).is_true() {
                    return true;
                }
            }
        }

        false
    }

    /// Count unsatisfied literals in a clause (optimized for auto-vectorization)
    ///
    /// # Arguments
    /// * `lits` - Slice of literals to check
    /// * `lit_values` - Function to get the value of a literal
    ///
    /// # Returns
    /// Number of literals that are not satisfied (not true)
    #[inline]
    #[allow(dead_code)]
    pub fn count_unsatisfied<F>(lits: &[Lit], mut lit_values: F) -> usize
    where
        F: FnMut(Lit) -> LBool,
    {
        let mut count = 0;

        // Process in chunks for vectorization
        const CHUNK_SIZE: usize = 8;

        for chunk in lits.chunks(CHUNK_SIZE) {
            for &lit in chunk {
                if !lit_values(lit).is_true() {
                    count += 1;
                }
            }
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_watch_lists() {
        let mut wl = WatchLists::new(5);

        let lit = Lit::pos(Var::new(0));
        let clause = ClauseId::new(0);
        let blocker = Lit::neg(Var::new(1));

        wl.add(lit, Watcher::new(clause, blocker));

        assert_eq!(wl.get(lit).len(), 1);
        assert_eq!(wl.get(lit)[0].clause, clause);
        assert_eq!(wl.get(lit)[0].blocker, blocker);
    }
}
