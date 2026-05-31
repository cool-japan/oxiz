//! LRU (Least Recently Used) cache implementation
//!
//! This module provides a bounded cache with LRU eviction policy.
//! It is used as the simplification memo cache in the aggressive simplifier.

use crate::prelude::*;
use core::hash::Hash;

/// A node in the LRU doubly-linked list
#[derive(Debug)]
struct Node<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// LRU cache with bounded size.
///
/// Capacity of 0 means unlimited (no eviction).
#[derive(Debug)]
pub(crate) struct LruCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    capacity: usize,
    map: FxHashMap<K, usize>,
    nodes: Vec<Node<K, V>>,
    head: Option<usize>,
    tail: Option<usize>,
    free_list: Vec<usize>,
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new LRU cache with given capacity.
    #[must_use]
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: FxHashMap::default(),
            nodes: Vec::new(),
            head: None,
            tail: None,
            free_list: Vec::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get the current number of items.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    /// Get cache statistics: `(hits, misses, evictions)`.
    #[must_use]
    pub(crate) fn stats(&self) -> (usize, usize, usize) {
        (self.hits, self.misses, self.evictions)
    }

    /// Look up a value, updating LRU order and hit/miss stats.
    /// Returns a clone of the cached value, or `None` on miss.
    pub(crate) fn get(&mut self, key: &K) -> Option<V> {
        if let Some(&idx) = self.map.get(key) {
            self.hits += 1;
            self.move_to_front(idx);
            Some(self.nodes[idx].value.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a key-value pair.
    /// Returns `true` for a new insertion, `false` for an update of an existing key.
    pub(crate) fn insert(&mut self, key: K, value: V) -> bool {
        if let Some(&idx) = self.map.get(&key) {
            self.nodes[idx].value = value;
            self.move_to_front(idx);
            return false;
        }

        if self.capacity > 0 && self.map.len() >= self.capacity {
            self.evict_lru();
        }

        let idx = if let Some(free_idx) = self.free_list.pop() {
            self.nodes[free_idx] = Node {
                key: key.clone(),
                value,
                prev: None,
                next: self.head,
            };
            free_idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Node {
                key: key.clone(),
                value,
                prev: None,
                next: self.head,
            });
            idx
        };

        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);

        if self.tail.is_none() {
            self.tail = Some(idx);
        }

        self.map.insert(key, idx);
        true
    }

    /// Clear all entries and reset statistics.
    #[cfg(test)]
    pub(crate) fn clear(&mut self) {
        self.map.clear();
        self.nodes.clear();
        self.free_list.clear();
        self.head = None;
        self.tail = None;
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    fn move_to_front(&mut self, idx: usize) {
        if Some(idx) == self.head {
            return;
        }

        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;

        if let Some(prev_idx) = prev {
            self.nodes[prev_idx].next = next;
        }
        if let Some(next_idx) = next {
            self.nodes[next_idx].prev = prev;
        }
        if Some(idx) == self.tail {
            self.tail = prev;
        }

        self.nodes[idx].prev = None;
        self.nodes[idx].next = self.head;

        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);
    }

    fn evict_lru(&mut self) {
        if let Some(tail_idx) = self.tail {
            let key = self.nodes[tail_idx].key.clone();
            if let Some(idx) = self.map.remove(&key) {
                let prev = self.nodes[idx].prev;
                let next = self.nodes[idx].next;

                if let Some(prev_idx) = prev {
                    self.nodes[prev_idx].next = next;
                } else {
                    self.head = next;
                }
                if let Some(next_idx) = next {
                    self.nodes[next_idx].prev = prev;
                } else {
                    self.tail = prev;
                }

                self.free_list.push(idx);
            }
            self.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_basic() {
        let mut cache: LruCache<u32, u32> = LruCache::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&1), Some(10));
        assert_eq!(cache.get(&3), None);
        let (hits, misses, _) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache: LruCache<u32, u32> = LruCache::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.get(&1); // make 1 most recently used
        cache.insert(3, 30); // should evict 2
        assert_eq!(cache.get(&1), Some(10));
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&3), Some(30));
        let (_, _, evictions) = cache.stats();
        assert_eq!(evictions, 1);
    }

    #[test]
    fn test_lru_cache_bounded() {
        let max_size = 32usize;
        let mut cache: LruCache<u32, u32> = LruCache::new(max_size);
        for i in 0..100u32 {
            cache.insert(i, i * 2);
        }
        assert!(
            cache.len() <= max_size,
            "cache grew beyond max_size: {} > {}",
            cache.len(),
            max_size
        );
        let (_, _, evictions) = cache.stats();
        assert!(evictions > 0);
    }

    #[test]
    fn test_lru_cache_update() {
        let mut cache: LruCache<u32, u32> = LruCache::new(4);
        assert!(cache.insert(1, 10));
        assert!(!cache.insert(1, 99));
        assert_eq!(cache.get(&1), Some(99));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_clear_resets_stats() {
        let mut cache: LruCache<u32, u32> = LruCache::new(4);
        cache.insert(1, 10);
        cache.get(&1);
        cache.get(&99);
        cache.clear();
        // After clear the cache is empty and stats are zeroed.
        assert_eq!(cache.len(), 0);
        // Check stats immediately after clear (before any new get calls).
        let (hits, misses, evictions) = cache.stats();
        assert_eq!((hits, misses, evictions), (0, 0, 0));
        // A subsequent get should return None (cache is empty).
        assert_eq!(cache.get(&1), None);
    }
}
