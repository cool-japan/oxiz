//! LRU (Least Recently Used) cache implementation
//!
//! This module provides a bounded cache with LRU eviction policy,
//! used for theory lemma caching in the theory combination module.

#![allow(dead_code)] // Module ready for integration

use rustc_hash::FxHashMap;
use std::hash::Hash;

/// A node in the LRU doubly-linked list
#[derive(Debug)]
struct Node<K, V> {
    key: K,
    value: V,
    prev: Option<usize>,
    next: Option<usize>,
}

/// LRU cache with bounded size
#[derive(Debug)]
pub struct LruCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Maximum capacity (0 = unlimited)
    capacity: usize,
    /// Map from key to node index
    map: FxHashMap<K, usize>,
    /// Storage for nodes
    nodes: Vec<Node<K, V>>,
    /// Index of most recently used item (head)
    head: Option<usize>,
    /// Index of least recently used item (tail)
    tail: Option<usize>,
    /// Free list for reusing slots
    free_list: Vec<usize>,
    /// Statistics
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new LRU cache with given capacity
    /// Capacity of 0 means unlimited (no eviction)
    #[must_use]
    pub fn new(capacity: usize) -> Self {
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

    /// Get the current number of items
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> (usize, usize, usize) {
        (self.hits, self.misses, self.evictions)
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    /// Get a value from the cache
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if let Some(&idx) = self.map.get(key) {
            self.hits += 1;
            self.move_to_front(idx);
            Some(&self.nodes[idx].value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a key-value pair into the cache
    /// Returns true if this is a new insertion, false if it was an update
    pub fn insert(&mut self, key: K, value: V) -> bool {
        // Check if key already exists
        if let Some(&idx) = self.map.get(&key) {
            // Update existing entry
            self.nodes[idx].value = value;
            self.move_to_front(idx);
            return false;
        }

        // Check if we need to evict (only if capacity is non-zero)
        if self.capacity > 0 && self.map.len() >= self.capacity {
            self.evict_lru();
        }

        // Allocate new node
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

        // Update head
        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);

        // Update tail if this is the first node
        if self.tail.is_none() {
            self.tail = Some(idx);
        }

        self.map.insert(key, idx);
        true
    }

    /// Check if a key exists in the cache
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// Remove a key from the cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(idx) = self.map.remove(key) {
            // Copy node data before modifications
            let value = self.nodes[idx].value.clone();
            let prev = self.nodes[idx].prev;
            let next = self.nodes[idx].next;

            // Update links
            if let Some(prev_idx) = prev {
                self.nodes[prev_idx].next = next;
            } else {
                // This was the head
                self.head = next;
            }

            if let Some(next_idx) = next {
                self.nodes[next_idx].prev = prev;
            } else {
                // This was the tail
                self.tail = prev;
            }

            // Add to free list
            self.free_list.push(idx);

            Some(value)
        } else {
            None
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.map.clear();
        self.nodes.clear();
        self.free_list.clear();
        self.head = None;
        self.tail = None;
    }

    /// Iterate over all key-value pairs (in arbitrary order)
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map
            .iter()
            .map(move |(k, &idx)| (k, &self.nodes[idx].value))
    }

    /// Retain only elements that satisfy the predicate
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let keys_to_remove: Vec<K> = self
            .map
            .iter()
            .filter(|(k, idx)| !f(k, &self.nodes[**idx].value))
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            self.remove(&key);
        }
    }

    /// Move a node to the front (most recently used)
    fn move_to_front(&mut self, idx: usize) {
        // Already at front?
        if Some(idx) == self.head {
            return;
        }

        // Copy the node's links before modifying
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;

        // Remove from current position
        if let Some(prev_idx) = prev {
            self.nodes[prev_idx].next = next;
        }
        if let Some(next_idx) = next {
            self.nodes[next_idx].prev = prev;
        }

        // Update tail if needed
        if Some(idx) == self.tail {
            self.tail = prev;
        }

        // Insert at front
        self.nodes[idx].prev = None;
        self.nodes[idx].next = self.head;

        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);
    }

    /// Evict the least recently used item
    fn evict_lru(&mut self) {
        if let Some(tail_idx) = self.tail {
            let key = self.nodes[tail_idx].key.clone();
            self.remove(&key);
            self.evictions += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut cache = LruCache::new(2);
        assert!(cache.is_empty());

        assert!(cache.insert(1, "one"));
        assert!(cache.insert(2, "two"));
        assert_eq!(cache.len(), 2);

        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&2), Some(&"two"));
        assert_eq!(cache.get(&3), None);
    }

    #[test]
    fn test_eviction() {
        let mut cache = LruCache::new(2);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three"); // Should evict 1

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some(&"two"));
        assert_eq!(cache.get(&3), Some(&"three"));

        let (_, _, evictions) = cache.stats();
        assert_eq!(evictions, 1);
    }

    #[test]
    fn test_lru_order() {
        let mut cache = LruCache::new(2);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.get(&1); // Access 1, making it more recent than 2
        cache.insert(3, "three"); // Should evict 2, not 1

        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&3), Some(&"three"));
    }

    #[test]
    fn test_update() {
        let mut cache = LruCache::new(2);

        assert!(cache.insert(1, "one"));
        assert!(!cache.insert(1, "ONE")); // Update, not insert
        assert_eq!(cache.get(&1), Some(&"ONE"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_unlimited_capacity() {
        let mut cache = LruCache::new(0); // Unlimited

        for i in 0..1000 {
            cache.insert(i, i * 2);
        }

        assert_eq!(cache.len(), 1000);
        let (_, _, evictions) = cache.stats();
        assert_eq!(evictions, 0);
    }

    #[test]
    fn test_remove() {
        let mut cache = LruCache::new(3);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.insert(3, "three");

        assert_eq!(cache.remove(&2), Some("two"));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&2), None);
        assert_eq!(cache.get(&1), Some(&"one"));
        assert_eq!(cache.get(&3), Some(&"three"));
    }

    #[test]
    fn test_clear() {
        let mut cache = LruCache::new(2);

        cache.insert(1, "one");
        cache.insert(2, "two");
        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn test_stats() {
        let mut cache = LruCache::new(2);

        cache.insert(1, "one");
        cache.get(&1); // hit
        cache.get(&2); // miss
        cache.get(&1); // hit

        let (hits, misses, _) = cache.stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_retain() {
        let mut cache = LruCache::new(10);

        for i in 0..5 {
            cache.insert(i, i * 2);
        }

        cache.retain(|&k, _| k % 2 == 0); // Keep only even keys

        assert_eq!(cache.len(), 3); // 0, 2, 4
        assert_eq!(cache.get(&0), Some(&0));
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), Some(&4));
    }
}
