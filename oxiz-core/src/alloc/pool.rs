//! Object Pool for Reusable Objects.
#![allow(missing_docs)] // Under development - documentation in progress
//!
//! Provides efficient reuse of frequently allocated objects like clauses and literals.

use parking_lot::Mutex;
use std::cell::RefCell;
use std::sync::Arc;

/// Configuration for object pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial pool capacity
    pub initial_capacity: usize,
    /// Maximum pool size (0 = unlimited)
    pub max_size: usize,
    /// Whether to preallocate objects
    pub preallocate: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1000,
            max_size: 10000,
            preallocate: false,
        }
    }
}

/// Thread-local object pool for fast allocation/deallocation.
pub struct ObjectPool<T> {
    pool: RefCell<Vec<T>>,
    config: PoolConfig,
    factory: Box<dyn Fn() -> T>,
    stats: RefCell<PoolStats>,
}

impl<T> ObjectPool<T> {
    /// Create a new object pool with a factory function.
    pub fn new<F>(factory: F, config: PoolConfig) -> Self
    where
        F: Fn() -> T + 'static,
    {
        let mut pool = Vec::with_capacity(config.initial_capacity);

        if config.preallocate {
            for _ in 0..config.initial_capacity {
                pool.push(factory());
            }
        }

        Self {
            pool: RefCell::new(pool),
            config,
            factory: Box::new(factory),
            stats: RefCell::new(PoolStats::default()),
        }
    }

    /// Get an object from the pool, or create a new one.
    pub fn get(&self) -> PoolGuard<'_, T> {
        let obj = self.pool.borrow_mut().pop();

        let obj = match obj {
            Some(obj) => {
                self.stats.borrow_mut().hits += 1;
                obj
            }
            None => {
                self.stats.borrow_mut().misses += 1;
                (self.factory)()
            }
        };

        PoolGuard {
            obj: Some(obj),
            pool: self,
        }
    }

    /// Return an object to the pool.
    fn return_obj(&self, obj: T) {
        let mut pool = self.pool.borrow_mut();

        if self.config.max_size == 0 || pool.len() < self.config.max_size {
            pool.push(obj);
            self.stats.borrow_mut().returns += 1;
        } else {
            self.stats.borrow_mut().discards += 1;
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        *self.stats.borrow()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.borrow_mut() = PoolStats::default();
    }

    /// Get current pool size.
    pub fn size(&self) -> usize {
        self.pool.borrow().len()
    }

    /// Clear the pool.
    pub fn clear(&self) {
        self.pool.borrow_mut().clear();
    }
}

/// Guard that returns object to pool on drop.
pub struct PoolGuard<'a, T> {
    obj: Option<T>,
    pool: &'a ObjectPool<T>,
}

impl<'a, T> PoolGuard<'a, T> {
    /// Get a reference to the pooled object.
    pub fn get(&self) -> &T {
        self.obj.as_ref().expect("pooled object already taken")
    }

    /// Get a mutable reference to the pooled object.
    pub fn get_mut(&mut self) -> &mut T {
        self.obj.as_mut().expect("pooled object already taken")
    }

    /// Take ownership of the object, preventing return to pool.
    pub fn take(mut self) -> T {
        self.obj.take().expect("pooled object already taken")
    }
}

impl<'a, T> Drop for PoolGuard<'a, T> {
    fn drop(&mut self) {
        if let Some(obj) = self.obj.take() {
            self.pool.return_obj(obj);
        }
    }
}

impl<'a, T> std::ops::Deref for PoolGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<'a, T> std::ops::DerefMut for PoolGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

/// Thread-safe object pool using Arc and Mutex.
pub struct SharedObjectPool<T> {
    pool: Arc<Mutex<Vec<T>>>,
    config: PoolConfig,
    factory: Arc<dyn Fn() -> T + Send + Sync>,
    stats: Arc<Mutex<PoolStats>>,
}

impl<T> SharedObjectPool<T>
where
    T: Send,
{
    /// Create a new shared object pool.
    pub fn new<F>(factory: F, config: PoolConfig) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        let mut pool = Vec::with_capacity(config.initial_capacity);

        if config.preallocate {
            for _ in 0..config.initial_capacity {
                pool.push(factory());
            }
        }

        Self {
            pool: Arc::new(Mutex::new(pool)),
            config,
            factory: Arc::new(factory),
            stats: Arc::new(Mutex::new(PoolStats::default())),
        }
    }

    /// Get an object from the pool.
    pub fn get(&self) -> SharedPoolGuard<T> {
        let obj = self.pool.lock().pop();

        let obj = match obj {
            Some(obj) => {
                self.stats.lock().hits += 1;
                obj
            }
            None => {
                self.stats.lock().misses += 1;
                (self.factory)()
            }
        };

        SharedPoolGuard {
            obj: Some(obj),
            pool: Arc::clone(&self.pool),
            stats: Arc::clone(&self.stats),
            max_size: self.config.max_size,
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        *self.stats.lock()
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        *self.stats.lock() = PoolStats::default();
    }

    /// Get current pool size.
    pub fn size(&self) -> usize {
        self.pool.lock().len()
    }

    /// Clear the pool.
    pub fn clear(&self) {
        self.pool.lock().clear();
    }
}

impl<T> Clone for SharedObjectPool<T> {
    fn clone(&self) -> Self {
        Self {
            pool: Arc::clone(&self.pool),
            config: self.config.clone(),
            factory: Arc::clone(&self.factory),
            stats: Arc::clone(&self.stats),
        }
    }
}

/// Guard for shared pool object.
pub struct SharedPoolGuard<T> {
    obj: Option<T>,
    pool: Arc<Mutex<Vec<T>>>,
    stats: Arc<Mutex<PoolStats>>,
    max_size: usize,
}

impl<T> SharedPoolGuard<T> {
    pub fn get(&self) -> &T {
        self.obj.as_ref().expect("pooled object already taken")
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.obj.as_mut().expect("pooled object already taken")
    }

    pub fn take(mut self) -> T {
        self.obj.take().expect("pooled object already taken")
    }
}

impl<T> Drop for SharedPoolGuard<T> {
    fn drop(&mut self) {
        if let Some(obj) = self.obj.take() {
            let mut pool = self.pool.lock();

            if self.max_size == 0 || pool.len() < self.max_size {
                pool.push(obj);
                self.stats.lock().returns += 1;
            } else {
                self.stats.lock().discards += 1;
            }
        }
    }
}

impl<T> std::ops::Deref for SharedPoolGuard<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T> std::ops::DerefMut for SharedPoolGuard<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

/// Statistics for object pool.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Number of pool hits (reused objects)
    pub hits: u64,
    /// Number of pool misses (new objects created)
    pub misses: u64,
    /// Number of returns to pool
    pub returns: u64,
    /// Number of discarded objects (pool full)
    pub discards: u64,
}

impl PoolStats {
    /// Calculate hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            return 0.0;
        }
        self.hits as f64 / (self.hits + self.misses) as f64
    }

    /// Calculate total requests.
    pub fn total_requests(&self) -> u64 {
        self.hits + self.misses
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestObject {
        value: i32,
    }

    impl TestObject {
        fn new() -> Self {
            Self { value: 0 }
        }
    }

    #[test]
    fn test_pool_basic() {
        let config = PoolConfig::default();
        let pool = ObjectPool::new(TestObject::new, config);

        let mut obj = pool.get();
        obj.value = 42;

        drop(obj);

        let obj2 = pool.get();
        assert_eq!(obj2.value, 42); // Reused object
    }

    #[test]
    fn test_pool_stats() {
        let config = PoolConfig::default();
        let pool = ObjectPool::new(TestObject::new, config);

        let _obj1 = pool.get(); // Miss
        drop(_obj1); // Return

        let _obj2 = pool.get(); // Hit

        let stats = pool.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.returns, 1);
    }

    #[test]
    fn test_pool_max_size() {
        let config = PoolConfig {
            initial_capacity: 1,
            max_size: 2,
            preallocate: false,
        };
        let pool = ObjectPool::new(TestObject::new, config);

        let obj1 = pool.get();
        let obj2 = pool.get();
        let obj3 = pool.get();

        drop(obj1);
        drop(obj2);
        drop(obj3); // Should be discarded

        let stats = pool.stats();
        assert_eq!(stats.returns, 2);
        assert_eq!(stats.discards, 1);
    }

    #[test]
    fn test_shared_pool() {
        let config = PoolConfig::default();
        let pool = SharedObjectPool::new(TestObject::new, config);

        let pool_clone = pool.clone();

        let mut obj = pool.get();
        obj.value = 100;
        drop(obj);

        let obj2 = pool_clone.get();
        assert_eq!(obj2.value, 100);
    }

    #[test]
    fn test_pool_take() {
        let config = PoolConfig::default();
        let pool = ObjectPool::new(TestObject::new, config);

        let mut guard = pool.get();
        guard.value = 50;

        let obj = guard.take();
        assert_eq!(obj.value, 50);
        assert_eq!(pool.size(), 0); // Not returned to pool
    }

    #[test]
    fn test_hit_rate() {
        let stats = PoolStats {
            hits: 80,
            misses: 20,
            returns: 80,
            discards: 0,
        };

        assert_eq!(stats.hit_rate(), 0.8);
        assert_eq!(stats.total_requests(), 100);
    }
}
