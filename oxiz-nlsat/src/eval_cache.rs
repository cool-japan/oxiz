//! Advanced polynomial evaluation cache and sign pattern memoization.
//!
//! This module provides high-performance caching for polynomial evaluations
//! and sign patterns, significantly improving solver performance on large
//! problems with repeated constraint evaluations.
//!
//! Key features:
//! - **Polynomial Evaluation Cache**: Cache polynomial values under specific assignments
//! - **Sign Pattern Memoization**: Cache sign patterns for sets of polynomials
//! - **LRU Eviction**: Automatically evict old entries when cache is full
//! - **Hash-based Indexing**: Fast O(1) lookup using polynomial fingerprints
//!
//! Reference: Z3's caching strategies in nlsat_evaluator and theory solvers

use num_rational::BigRational;
use num_traits::{Signed, Zero};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// A fingerprint for a polynomial and assignment combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EvalFingerprint {
    poly_hash: u64,
    assignment_hash: u64,
}

impl EvalFingerprint {
    /// Create a fingerprint from a polynomial and assignment.
    fn new(poly: &Polynomial, assignment: &FxHashMap<Var, BigRational>) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Hash the polynomial
        let mut poly_hasher = DefaultHasher::new();
        format!("{:?}", poly).hash(&mut poly_hasher);
        let poly_hash = poly_hasher.finish();

        // Hash the assignment (only relevant variables)
        let mut assignment_hasher = DefaultHasher::new();
        let mut vars = poly.vars();
        vars.sort();
        for var in &vars {
            if let Some(value) = assignment.get(var) {
                var.hash(&mut assignment_hasher);
                format!("{:?}", value).hash(&mut assignment_hasher);
            }
        }
        let assignment_hash = assignment_hasher.finish();

        Self {
            poly_hash,
            assignment_hash,
        }
    }
}

/// Sign of a polynomial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachedSign {
    /// Positive value.
    Positive,
    /// Zero.
    Zero,
    /// Negative value.
    Negative,
}

impl CachedSign {
    /// Create from a BigRational value.
    pub fn from_value(value: &BigRational) -> Self {
        if value.is_zero() {
            CachedSign::Zero
        } else if value.is_positive() {
            CachedSign::Positive
        } else {
            CachedSign::Negative
        }
    }
}

/// A pattern of signs for multiple polynomials.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignPattern(Vec<CachedSign>);

impl SignPattern {
    /// Create a sign pattern from a list of signs.
    pub fn new(signs: Vec<CachedSign>) -> Self {
        Self(signs)
    }

    /// Get the sign at the given index.
    pub fn get(&self, index: usize) -> Option<CachedSign> {
        self.0.get(index).copied()
    }

    /// Length of the pattern.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the pattern is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Statistics for the evaluation cache.
#[derive(Debug, Clone, Default)]
pub struct EvalCacheStats {
    /// Number of cache hits for polynomial values.
    pub value_hits: u64,
    /// Number of cache misses for polynomial values.
    pub value_misses: u64,
    /// Number of cache hits for sign patterns.
    pub pattern_hits: u64,
    /// Number of cache misses for sign patterns.
    pub pattern_misses: u64,
    /// Number of evictions due to cache being full.
    pub evictions: u64,
}

impl EvalCacheStats {
    /// Calculate the hit rate for value cache.
    pub fn value_hit_rate(&self) -> f64 {
        let total = self.value_hits + self.value_misses;
        if total == 0 {
            0.0
        } else {
            self.value_hits as f64 / total as f64
        }
    }

    /// Calculate the hit rate for pattern cache.
    pub fn pattern_hit_rate(&self) -> f64 {
        let total = self.pattern_hits + self.pattern_misses;
        if total == 0 {
            0.0
        } else {
            self.pattern_hits as f64 / total as f64
        }
    }
}

/// Configuration for the evaluation cache.
#[derive(Debug, Clone)]
pub struct EvalCacheConfig {
    /// Maximum number of cached polynomial values.
    pub max_value_entries: usize,
    /// Maximum number of cached sign patterns.
    pub max_pattern_entries: usize,
    /// Enable value caching.
    pub enable_value_cache: bool,
    /// Enable pattern caching.
    pub enable_pattern_cache: bool,
}

impl Default for EvalCacheConfig {
    fn default() -> Self {
        Self {
            max_value_entries: 10_000,
            max_pattern_entries: 5_000,
            enable_value_cache: true,
            enable_pattern_cache: true,
        }
    }
}

/// Advanced polynomial evaluation cache.
pub struct EvalCache {
    /// Configuration.
    config: EvalCacheConfig,
    /// Cache for polynomial values.
    value_cache: FxHashMap<EvalFingerprint, BigRational>,
    /// LRU queue for value cache eviction.
    value_lru: VecDeque<EvalFingerprint>,
    /// Cache for sign patterns.
    pattern_cache: FxHashMap<u64, SignPattern>,
    /// LRU queue for pattern cache eviction.
    pattern_lru: VecDeque<u64>,
    /// Statistics.
    stats: EvalCacheStats,
}

impl EvalCache {
    /// Create a new evaluation cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(EvalCacheConfig::default())
    }

    /// Create a new evaluation cache with the given configuration.
    pub fn with_config(config: EvalCacheConfig) -> Self {
        Self {
            config,
            value_cache: FxHashMap::default(),
            value_lru: VecDeque::new(),
            pattern_cache: FxHashMap::default(),
            pattern_lru: VecDeque::new(),
            stats: EvalCacheStats::default(),
        }
    }

    /// Look up a cached polynomial value.
    pub fn lookup_value(
        &mut self,
        poly: &Polynomial,
        assignment: &FxHashMap<Var, BigRational>,
    ) -> Option<BigRational> {
        if !self.config.enable_value_cache {
            return None;
        }

        let fingerprint = EvalFingerprint::new(poly, assignment);

        if let Some(value) = self.value_cache.get(&fingerprint) {
            self.stats.value_hits += 1;
            // Move to front of LRU
            self.value_lru.retain(|&f| f != fingerprint);
            self.value_lru.push_front(fingerprint);
            Some(value.clone())
        } else {
            self.stats.value_misses += 1;
            None
        }
    }

    /// Insert a polynomial value into the cache.
    pub fn insert_value(
        &mut self,
        poly: &Polynomial,
        assignment: &FxHashMap<Var, BigRational>,
        value: BigRational,
    ) {
        if !self.config.enable_value_cache {
            return;
        }

        let fingerprint = EvalFingerprint::new(poly, assignment);

        // Check if we need to evict
        if self.value_cache.len() >= self.config.max_value_entries
            && let Some(old_fingerprint) = self.value_lru.pop_back()
        {
            self.value_cache.remove(&old_fingerprint);
            self.stats.evictions += 1;
        }

        self.value_cache.insert(fingerprint, value);
        self.value_lru.push_front(fingerprint);
    }

    /// Look up a cached sign pattern.
    pub fn lookup_pattern(&mut self, pattern_hash: u64) -> Option<SignPattern> {
        if !self.config.enable_pattern_cache {
            return None;
        }

        if let Some(pattern) = self.pattern_cache.get(&pattern_hash) {
            self.stats.pattern_hits += 1;
            // Move to front of LRU
            self.pattern_lru.retain(|&h| h != pattern_hash);
            self.pattern_lru.push_front(pattern_hash);
            Some(pattern.clone())
        } else {
            self.stats.pattern_misses += 1;
            None
        }
    }

    /// Insert a sign pattern into the cache.
    pub fn insert_pattern(&mut self, pattern_hash: u64, pattern: SignPattern) {
        if !self.config.enable_pattern_cache {
            return;
        }

        // Check if we need to evict
        if self.pattern_cache.len() >= self.config.max_pattern_entries
            && let Some(old_hash) = self.pattern_lru.pop_back()
        {
            self.pattern_cache.remove(&old_hash);
            self.stats.evictions += 1;
        }

        self.pattern_cache.insert(pattern_hash, pattern);
        self.pattern_lru.push_front(pattern_hash);
    }

    /// Compute a hash for a set of polynomials and assignment.
    pub fn compute_pattern_hash(
        polys: &[Polynomial],
        assignment: &FxHashMap<Var, BigRational>,
    ) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        for poly in polys {
            format!("{:?}", poly).hash(&mut hasher);
        }

        // Hash the assignment
        let mut vars: Vec<_> = assignment.keys().copied().collect();
        vars.sort();
        for var in vars {
            if let Some(value) = assignment.get(&var) {
                var.hash(&mut hasher);
                format!("{:?}", value).hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    /// Clear all caches.
    pub fn clear(&mut self) {
        self.value_cache.clear();
        self.value_lru.clear();
        self.pattern_cache.clear();
        self.pattern_lru.clear();
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &EvalCacheStats {
        &self.stats
    }

    /// Get the current size of the value cache.
    pub fn value_cache_size(&self) -> usize {
        self.value_cache.len()
    }

    /// Get the current size of the pattern cache.
    pub fn pattern_cache_size(&self) -> usize {
        self.pattern_cache.len()
    }
}

impl Default for EvalCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_traits::Zero;

    #[test]
    fn test_eval_cache_new() {
        let cache = EvalCache::new();
        assert_eq!(cache.value_cache_size(), 0);
        assert_eq!(cache.pattern_cache_size(), 0);
    }

    #[test]
    fn test_value_cache_basic() {
        let mut cache = EvalCache::new();

        let poly = Polynomial::from_var(0);
        let mut assignment = FxHashMap::default();
        assignment.insert(0, BigRational::from_integer(BigInt::from(5)));

        // Miss on first lookup
        assert!(cache.lookup_value(&poly, &assignment).is_none());

        // Insert value
        let value = BigRational::from_integer(BigInt::from(5));
        cache.insert_value(&poly, &assignment, value.clone());

        // Hit on second lookup
        assert_eq!(cache.lookup_value(&poly, &assignment), Some(value));
        assert_eq!(cache.stats().value_hits, 1);
        assert_eq!(cache.stats().value_misses, 1);
    }

    #[test]
    fn test_value_cache_eviction() {
        let config = EvalCacheConfig {
            max_value_entries: 2,
            ..Default::default()
        };
        let mut cache = EvalCache::with_config(config);

        // Insert 3 entries (should evict the oldest)
        for i in 0..3 {
            let poly = Polynomial::from_var(i);
            let mut assignment = FxHashMap::default();
            assignment.insert(i, BigRational::from_integer(BigInt::from(i as i32)));
            cache.insert_value(&poly, &assignment, BigRational::zero());
        }

        assert_eq!(cache.value_cache_size(), 2);
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_sign_pattern() {
        let pattern = SignPattern::new(vec![
            CachedSign::Positive,
            CachedSign::Zero,
            CachedSign::Negative,
        ]);

        assert_eq!(pattern.len(), 3);
        assert_eq!(pattern.get(0), Some(CachedSign::Positive));
        assert_eq!(pattern.get(1), Some(CachedSign::Zero));
        assert_eq!(pattern.get(2), Some(CachedSign::Negative));
        assert_eq!(pattern.get(3), None);
    }

    #[test]
    fn test_pattern_cache_basic() {
        let mut cache = EvalCache::new();

        let pattern = SignPattern::new(vec![CachedSign::Positive, CachedSign::Negative]);
        let hash = 12345u64;

        // Miss on first lookup
        assert!(cache.lookup_pattern(hash).is_none());

        // Insert pattern
        cache.insert_pattern(hash, pattern.clone());

        // Hit on second lookup
        assert_eq!(cache.lookup_pattern(hash), Some(pattern));
        assert_eq!(cache.stats().pattern_hits, 1);
        assert_eq!(cache.stats().pattern_misses, 1);
    }

    #[test]
    fn test_stats_hit_rate() {
        let stats = EvalCacheStats {
            value_hits: 80,
            value_misses: 20,
            pattern_hits: 60,
            pattern_misses: 40,
            evictions: 5,
        };

        assert_eq!(stats.value_hit_rate(), 0.8);
        assert_eq!(stats.pattern_hit_rate(), 0.6);
    }

    #[test]
    fn test_clear() {
        let mut cache = EvalCache::new();

        let poly = Polynomial::from_var(0);
        let mut assignment = FxHashMap::default();
        assignment.insert(0, BigRational::from_integer(BigInt::from(1)));
        cache.insert_value(&poly, &assignment, BigRational::zero());

        cache.insert_pattern(123, SignPattern::new(vec![CachedSign::Positive]));

        cache.clear();
        assert_eq!(cache.value_cache_size(), 0);
        assert_eq!(cache.pattern_cache_size(), 0);
    }
}
