//! Explanation Cache for Conflict Analysis.
//!
//! Caches theory explanations to avoid redundant computation during
//! conflict analysis and proof generation.
//!
//! ## Features
//!
//! - **LRU eviction**: Maintains cache size with least-recently-used policy
//! - **Explanation reuse**: Avoids recomputing identical explanations
//! - **Proof sharing**: Shares proof terms across multiple conflicts
//!
//! ## References
//!
//! - Z3's `smt/smt_theory.cpp` explanation caching

use oxiz_sat::Lit;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Cache key (theory conflict identifier).
pub type CacheKey = u64;

/// Explanation for a theory conflict.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// Literals in the explanation.
    pub literals: Vec<Lit>,
    /// Proof term (optional).
    pub proof: Option<String>,
    /// Timestamp (for LRU).
    timestamp: u64,
}

/// Configuration for explanation cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache entries.
    pub max_entries: usize,
    /// Enable proof caching.
    pub cache_proofs: bool,
    /// Enable LRU eviction.
    pub use_lru: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            cache_proofs: true,
            use_lru: true,
        }
    }
}

/// Statistics for explanation cache.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Cache hits.
    pub hits: u64,
    /// Cache misses.
    pub misses: u64,
    /// Cache evictions.
    pub evictions: u64,
    /// Total explanations cached.
    pub cached: u64,
}

impl CacheStats {
    /// Get hit rate (0.0 - 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Explanation cache with LRU eviction.
#[derive(Debug)]
pub struct ExplanationCache {
    /// Cache map.
    cache: FxHashMap<CacheKey, Explanation>,
    /// LRU queue (most recent at back).
    lru_queue: VecDeque<CacheKey>,
    /// Current timestamp.
    timestamp: u64,
    /// Configuration.
    config: CacheConfig,
    /// Statistics.
    stats: CacheStats,
}

impl ExplanationCache {
    /// Create a new explanation cache.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: FxHashMap::default(),
            lru_queue: VecDeque::new(),
            timestamp: 0,
            config,
            stats: CacheStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Insert an explanation into the cache.
    pub fn insert(&mut self, key: CacheKey, literals: Vec<Lit>, proof: Option<String>) {
        // Check if we need to evict
        if self.cache.len() >= self.config.max_entries && !self.cache.contains_key(&key) {
            self.evict_lru();
        }

        let explanation = Explanation {
            literals,
            proof: if self.config.cache_proofs {
                proof
            } else {
                None
            },
            timestamp: self.timestamp,
        };

        self.cache.insert(key, explanation);
        if self.config.use_lru {
            self.lru_queue.push_back(key);
        }
        self.stats.cached += 1;
        self.timestamp += 1;
    }

    /// Get an explanation from the cache.
    pub fn get(&mut self, key: CacheKey) -> Option<&Explanation> {
        if let Some(explanation) = self.cache.get_mut(&key) {
            // Update timestamp for LRU
            explanation.timestamp = self.timestamp;
            self.timestamp += 1;
            self.stats.hits += 1;

            // Move to back of LRU queue
            if self.config.use_lru {
                self.lru_queue.retain(|&k| k != key);
                self.lru_queue.push_back(key);
            }

            Some(explanation)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        if let Some(key) = self.lru_queue.pop_front() {
            self.cache.remove(&key);
            self.stats.evictions += 1;
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_queue.clear();
        self.timestamp = 0;
    }

    /// Get cache size.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }
}

impl Default for ExplanationCache {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() {
        let cache = ExplanationCache::default_config();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        use oxiz_sat::Var;
        let mut cache = ExplanationCache::default_config();
        let key = 42;
        let literals = vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))];

        cache.insert(key, literals.clone(), None);
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get(key).unwrap();
        assert_eq!(retrieved.literals.len(), 2);
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = ExplanationCache::default_config();
        let result = cache.get(999);
        assert!(result.is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            ..Default::default()
        };
        let mut cache = ExplanationCache::new(config);

        // Insert 3 entries (should evict first)
        cache.insert(1, vec![], None);
        cache.insert(2, vec![], None);
        cache.insert(3, vec![], None);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 1);

        // First entry should be evicted
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = ExplanationCache::default_config();
        cache.insert(1, vec![], None);

        cache.get(1); // hit
        cache.get(2); // miss
        cache.get(1); // hit

        assert_eq!(cache.stats().hits, 2);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.stats().hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut cache = ExplanationCache::default_config();
        cache.insert(1, vec![], None);
        cache.insert(2, vec![], None);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_proof_caching() {
        let mut cache = ExplanationCache::default_config();
        let key = 100;
        let proof = Some("(proof-term ...)".to_string());

        cache.insert(key, vec![], proof.clone());

        let retrieved = cache.get(key).unwrap();
        assert_eq!(retrieved.proof, proof);
    }

    #[test]
    fn test_stats() {
        let mut cache = ExplanationCache::default_config();
        cache.insert(1, vec![], None);
        cache.get(1);

        assert_eq!(cache.stats().cached, 1);
        assert_eq!(cache.stats().hits, 1);

        cache.reset_stats();
        assert_eq!(cache.stats().cached, 0);
        assert_eq!(cache.stats().hits, 0);
    }
}
