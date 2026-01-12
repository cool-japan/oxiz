//! Memory pooling for efficient term allocation in WASM
//!
//! This module provides a simple memory pool for allocating strings and small objects
//! efficiently in the WASM environment. The pool reduces the number of allocations
//! and improves cache locality.

use std::cell::RefCell;
use std::collections::VecDeque;

/// A simple memory pool for strings
///
/// This pool maintains a collection of pre-allocated strings that can be reused
/// across multiple solver operations. This is particularly useful in WASM where
/// allocation costs can be higher than on native platforms.
pub struct StringPool {
    /// Pool of available strings
    pool: RefCell<VecDeque<String>>,
    /// Maximum size of the pool
    max_size: usize,
    /// Initial capacity for new strings
    initial_capacity: usize,
}

impl StringPool {
    /// Create a new string pool
    ///
    /// # Parameters
    ///
    /// * `max_size` - Maximum number of strings to keep in the pool
    /// * `initial_capacity` - Initial capacity for new strings
    pub fn new(max_size: usize, initial_capacity: usize) -> Self {
        Self {
            pool: RefCell::new(VecDeque::with_capacity(max_size)),
            max_size,
            initial_capacity,
        }
    }

    /// Acquire a string from the pool
    ///
    /// If the pool is empty, allocates a new string. Otherwise, reuses a string
    /// from the pool.
    ///
    /// # Returns
    ///
    /// A string that can be used and should be returned to the pool when done
    pub fn acquire(&self) -> String {
        self.pool
            .borrow_mut()
            .pop_front()
            .unwrap_or_else(|| String::with_capacity(self.initial_capacity))
    }

    /// Release a string back to the pool
    ///
    /// If the pool is not full, the string is cleared and returned to the pool.
    /// Otherwise, the string is dropped.
    ///
    /// # Parameters
    ///
    /// * `s` - The string to release
    pub fn release(&self, mut s: String) {
        let mut pool = self.pool.borrow_mut();
        if pool.len() < self.max_size {
            s.clear();
            pool.push_back(s);
        }
        // Otherwise, drop the string
    }

    /// Get the current size of the pool
    pub fn size(&self) -> usize {
        self.pool.borrow().len()
    }

    /// Clear all strings from the pool
    pub fn clear(&self) {
        self.pool.borrow_mut().clear();
    }
}

impl Default for StringPool {
    fn default() -> Self {
        // Default: pool up to 64 strings, each with 256 bytes initial capacity
        Self::new(64, 256)
    }
}

/// A simple memory pool for Vec<String>
///
/// This pool maintains a collection of pre-allocated Vec<String> that can be reused
/// across multiple solver operations.
pub struct VecStringPool {
    /// Pool of available vectors
    pool: RefCell<VecDeque<Vec<String>>>,
    /// Maximum size of the pool
    max_size: usize,
    /// Initial capacity for new vectors
    initial_capacity: usize,
}

impl VecStringPool {
    /// Create a new vector pool
    ///
    /// # Parameters
    ///
    /// * `max_size` - Maximum number of vectors to keep in the pool
    /// * `initial_capacity` - Initial capacity for new vectors
    pub fn new(max_size: usize, initial_capacity: usize) -> Self {
        Self {
            pool: RefCell::new(VecDeque::with_capacity(max_size)),
            max_size,
            initial_capacity,
        }
    }

    /// Acquire a vector from the pool
    ///
    /// If the pool is empty, allocates a new vector. Otherwise, reuses a vector
    /// from the pool.
    pub fn acquire(&self) -> Vec<String> {
        self.pool
            .borrow_mut()
            .pop_front()
            .unwrap_or_else(|| Vec::with_capacity(self.initial_capacity))
    }

    /// Release a vector back to the pool
    ///
    /// If the pool is not full, the vector is cleared and returned to the pool.
    /// Otherwise, the vector is dropped.
    pub fn release(&self, mut v: Vec<String>) {
        let mut pool = self.pool.borrow_mut();
        if pool.len() < self.max_size {
            v.clear();
            pool.push_back(v);
        }
        // Otherwise, drop the vector
    }

    /// Get the current size of the pool
    pub fn size(&self) -> usize {
        self.pool.borrow().len()
    }

    /// Clear all vectors from the pool
    pub fn clear(&self) {
        self.pool.borrow_mut().clear();
    }
}

impl Default for VecStringPool {
    fn default() -> Self {
        // Default: pool up to 32 vectors, each with 16 elements initial capacity
        Self::new(32, 16)
    }
}

// Global string pool instance
//
// This is a thread-local global pool that can be used throughout the application.
// Using a global pool reduces the need to pass pool references around.
thread_local! {
    static GLOBAL_STRING_POOL: StringPool = StringPool::default();
    static GLOBAL_VEC_STRING_POOL: VecStringPool = VecStringPool::default();
}

/// Acquire a string from the global pool
#[allow(dead_code)]
pub fn acquire_string() -> String {
    GLOBAL_STRING_POOL.with(|pool| pool.acquire())
}

/// Release a string to the global pool
#[allow(dead_code)]
pub fn release_string(s: String) {
    GLOBAL_STRING_POOL.with(|pool| pool.release(s));
}

/// Acquire a Vec<String> from the global pool
#[allow(dead_code)]
pub fn acquire_vec_string() -> Vec<String> {
    GLOBAL_VEC_STRING_POOL.with(|pool| pool.acquire())
}

/// Release a Vec<String> to the global pool
#[allow(dead_code)]
pub fn release_vec_string(v: Vec<String>) {
    GLOBAL_VEC_STRING_POOL.with(|pool| pool.release(v));
}

/// Get statistics about the global pools
#[allow(dead_code)]
pub fn pool_stats() -> (usize, usize) {
    let string_pool_size = GLOBAL_STRING_POOL.with(|pool| pool.size());
    let vec_pool_size = GLOBAL_VEC_STRING_POOL.with(|pool| pool.size());
    (string_pool_size, vec_pool_size)
}

/// Clear all global pools
#[allow(dead_code)]
pub fn clear_global_pools() {
    GLOBAL_STRING_POOL.with(|pool| pool.clear());
    GLOBAL_VEC_STRING_POOL.with(|pool| pool.clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_pool() {
        let pool = StringPool::new(4, 64);
        assert_eq!(pool.size(), 0);

        // Acquire and release a string
        let s = pool.acquire();
        assert_eq!(pool.size(), 0);
        pool.release(s);
        assert_eq!(pool.size(), 1);

        // Acquire again - should reuse
        let s = pool.acquire();
        assert_eq!(pool.size(), 0);
        pool.release(s);
        assert_eq!(pool.size(), 1);

        // Fill the pool
        for _ in 0..5 {
            pool.release(String::new());
        }
        // Should be capped at max_size
        assert_eq!(pool.size(), 4);
    }

    #[test]
    fn test_vec_string_pool() {
        let pool = VecStringPool::new(4, 16);
        assert_eq!(pool.size(), 0);

        // Acquire and release a vector
        let v = pool.acquire();
        assert_eq!(pool.size(), 0);
        pool.release(v);
        assert_eq!(pool.size(), 1);

        // Acquire again - should reuse
        let v = pool.acquire();
        assert_eq!(pool.size(), 0);
        pool.release(v);
        assert_eq!(pool.size(), 1);

        // Fill the pool
        for _ in 0..5 {
            pool.release(Vec::new());
        }
        // Should be capped at max_size
        assert_eq!(pool.size(), 4);
    }

    #[test]
    fn test_global_string_pool() {
        clear_global_pools();

        let s = acquire_string();
        release_string(s);

        let (string_size, _) = pool_stats();
        assert_eq!(string_size, 1);

        clear_global_pools();
        let (string_size, _) = pool_stats();
        assert_eq!(string_size, 0);
    }

    #[test]
    fn test_global_vec_pool() {
        clear_global_pools();

        let v = acquire_vec_string();
        release_vec_string(v);

        let (_, vec_size) = pool_stats();
        assert_eq!(vec_size, 1);

        clear_global_pools();
        let (_, vec_size) = pool_stats();
        assert_eq!(vec_size, 0);
    }

    #[test]
    fn test_string_pool_clears_content() {
        let pool = StringPool::new(4, 64);

        let mut s = pool.acquire();
        s.push_str("test content");
        pool.release(s);

        let s = pool.acquire();
        assert!(s.is_empty());
        pool.release(s);
    }

    #[test]
    fn test_vec_pool_clears_content() {
        let pool = VecStringPool::new(4, 16);

        let mut v = pool.acquire();
        v.push("test".to_string());
        v.push("content".to_string());
        pool.release(v);

        let v = pool.acquire();
        assert!(v.is_empty());
        pool.release(v);
    }
}
