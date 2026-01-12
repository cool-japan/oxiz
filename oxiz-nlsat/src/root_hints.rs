//! Root approximation hints cache for faster root isolation.
//!
//! This module caches approximate root locations to speed up repeated
//! root isolation operations. When solving similar problems or revisiting
//! polynomials during backtracking, cached hints provide initial bounds
//! that dramatically reduce isolation time.
//!
//! Key features:
//! - **Approximate Root Storage**: Store interval approximations for roots
//! - **Fast Lookup**: Hash-based O(1) lookup by polynomial
//! - **Refinement Support**: Track refinement levels for progressive precision
//! - **LRU Eviction**: Automatic eviction of old hints
//!
//! Reference: Techniques from numerical root-finding and interval arithmetic

use num_rational::BigRational;
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;
use std::collections::{VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

/// An approximate root location with confidence bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootHint {
    /// Lower bound of root location.
    pub lower: BigRational,
    /// Upper bound of root location.
    pub upper: BigRational,
    /// Refinement level (higher = more precise).
    pub refinement_level: u32,
}

impl RootHint {
    /// Create a new root hint.
    pub fn new(lower: BigRational, upper: BigRational, refinement_level: u32) -> Self {
        Self {
            lower,
            upper,
            refinement_level,
        }
    }

    /// Get the interval width.
    pub fn width(&self) -> BigRational {
        &self.upper - &self.lower
    }

    /// Get the midpoint of the hint interval.
    pub fn midpoint(&self) -> BigRational {
        (&self.lower + &self.upper) / BigRational::from_integer(2.into())
    }

    /// Check if this hint contains a value.
    pub fn contains(&self, value: &BigRational) -> bool {
        &self.lower <= value && value <= &self.upper
    }
}

/// Collection of root hints for a polynomial.
#[derive(Debug, Clone)]
pub struct PolynomialRootHints {
    /// Polynomial hash for identification.
    #[allow(dead_code)]
    poly_hash: u64,
    /// Variable the roots are for.
    #[allow(dead_code)]
    variable: Var,
    /// Ordered list of root hints.
    hints: Vec<RootHint>,
    /// Number of times these hints have been used.
    use_count: u64,
}

impl PolynomialRootHints {
    /// Create new polynomial root hints.
    pub fn new(poly_hash: u64, variable: Var, hints: Vec<RootHint>) -> Self {
        Self {
            poly_hash,
            variable,
            hints,
            use_count: 0,
        }
    }

    /// Get the number of roots hinted.
    pub fn num_roots(&self) -> usize {
        self.hints.len()
    }

    /// Get a specific hint by index.
    pub fn get_hint(&self, index: usize) -> Option<&RootHint> {
        self.hints.get(index)
    }

    /// Get all hints.
    pub fn all_hints(&self) -> &[RootHint] {
        &self.hints
    }

    /// Increment usage count.
    fn mark_used(&mut self) {
        self.use_count += 1;
    }
}

/// Statistics for root hints cache.
#[derive(Debug, Clone, Default)]
pub struct RootHintsStats {
    /// Number of hints stored.
    pub num_hints_stored: u64,
    /// Number of cache hits.
    pub num_cache_hits: u64,
    /// Number of cache misses.
    pub num_cache_misses: u64,
    /// Number of hints used for refinement.
    pub num_refinements: u64,
}

impl RootHintsStats {
    /// Calculate cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total = self.num_cache_hits + self.num_cache_misses;
        if total == 0 {
            0.0
        } else {
            self.num_cache_hits as f64 / total as f64
        }
    }
}

/// Configuration for root hints cache.
#[derive(Debug, Clone)]
pub struct RootHintsConfig {
    /// Maximum number of polynomial hint sets to cache.
    pub max_polynomials: usize,
    /// Maximum number of hints per polynomial.
    pub max_hints_per_poly: usize,
    /// Enable hint refinement tracking.
    pub enable_refinement: bool,
}

impl Default for RootHintsConfig {
    fn default() -> Self {
        Self {
            max_polynomials: 1000,
            max_hints_per_poly: 20,
            enable_refinement: true,
        }
    }
}

/// Root approximation hints cache.
pub struct RootHintsCache {
    /// Configuration.
    config: RootHintsConfig,
    /// Cache: polynomial hash -> root hints.
    cache: FxHashMap<u64, PolynomialRootHints>,
    /// LRU queue for eviction.
    lru: VecDeque<u64>,
    /// Statistics.
    stats: RootHintsStats,
}

impl RootHintsCache {
    /// Create a new root hints cache.
    pub fn new() -> Self {
        Self::with_config(RootHintsConfig::default())
    }

    /// Create a new cache with the given configuration.
    pub fn with_config(config: RootHintsConfig) -> Self {
        Self {
            config,
            cache: FxHashMap::default(),
            lru: VecDeque::new(),
            stats: RootHintsStats::default(),
        }
    }

    /// Store root hints for a polynomial.
    pub fn store_hints(&mut self, poly: &Polynomial, variable: Var, hints: Vec<RootHint>) {
        let hash = Self::hash_polynomial(poly, variable);

        // Limit number of hints per polynomial
        let limited_hints: Vec<_> = hints
            .into_iter()
            .take(self.config.max_hints_per_poly)
            .collect();

        let poly_hints = PolynomialRootHints::new(hash, variable, limited_hints);

        // Check if we need to evict
        if self.cache.len() >= self.config.max_polynomials
            && !self.cache.contains_key(&hash)
            && let Some(old_hash) = self.lru.pop_back()
        {
            self.cache.remove(&old_hash);
        }

        self.cache.insert(hash, poly_hints);
        self.lru.retain(|&h| h != hash);
        self.lru.push_front(hash);

        self.stats.num_hints_stored += 1;
    }

    /// Lookup root hints for a polynomial.
    pub fn lookup_hints(&mut self, poly: &Polynomial, variable: Var) -> Option<&[RootHint]> {
        let hash = Self::hash_polynomial(poly, variable);

        if let Some(hints) = self.cache.get_mut(&hash) {
            self.stats.num_cache_hits += 1;
            hints.mark_used();

            // Move to front of LRU
            self.lru.retain(|&h| h != hash);
            self.lru.push_front(hash);

            Some(hints.all_hints())
        } else {
            self.stats.num_cache_misses += 1;
            None
        }
    }

    /// Refine an existing hint with more precise bounds.
    pub fn refine_hint(
        &mut self,
        poly: &Polynomial,
        variable: Var,
        root_index: usize,
        new_lower: BigRational,
        new_upper: BigRational,
    ) -> bool {
        if !self.config.enable_refinement {
            return false;
        }

        let hash = Self::hash_polynomial(poly, variable);

        if let Some(hints) = self.cache.get_mut(&hash)
            && let Some(hint) = hints.hints.get_mut(root_index)
        {
            hint.lower = new_lower;
            hint.upper = new_upper;
            hint.refinement_level += 1;
            self.stats.num_refinements += 1;
            return true;
        }

        false
    }

    /// Get the number of cached root hints for a polynomial.
    pub fn num_roots(&mut self, poly: &Polynomial, variable: Var) -> Option<usize> {
        let hash = Self::hash_polynomial(poly, variable);
        self.cache.get(&hash).map(|h| h.num_roots())
    }

    /// Hash a polynomial and variable for caching.
    fn hash_polynomial(poly: &Polynomial, variable: Var) -> u64 {
        let mut hasher = DefaultHasher::new();
        format!("{:?}", poly).hash(&mut hasher);
        variable.hash(&mut hasher);
        hasher.finish()
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &RootHintsStats {
        &self.stats
    }

    /// Get cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for RootHintsCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn rat(n: i32) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn test_root_hint_new() {
        let hint = RootHint::new(rat(0), rat(10), 1);
        assert_eq!(hint.lower, rat(0));
        assert_eq!(hint.upper, rat(10));
        assert_eq!(hint.refinement_level, 1);
    }

    #[test]
    fn test_root_hint_width() {
        let hint = RootHint::new(rat(0), rat(10), 1);
        assert_eq!(hint.width(), rat(10));
    }

    #[test]
    fn test_root_hint_midpoint() {
        let hint = RootHint::new(rat(0), rat(10), 1);
        assert_eq!(hint.midpoint(), rat(5));
    }

    #[test]
    fn test_root_hint_contains() {
        let hint = RootHint::new(rat(0), rat(10), 1);
        assert!(hint.contains(&rat(5)));
        assert!(hint.contains(&rat(0)));
        assert!(hint.contains(&rat(10)));
        assert!(!hint.contains(&rat(-1)));
        assert!(!hint.contains(&rat(11)));
    }

    #[test]
    fn test_cache_new() {
        let cache = RootHintsCache::new();
        assert_eq!(cache.cache_size(), 0);
    }

    #[test]
    fn test_store_and_lookup() {
        let mut cache = RootHintsCache::new();
        let poly = Polynomial::from_var(0);
        let hints = vec![RootHint::new(rat(0), rat(1), 1)];

        cache.store_hints(&poly, 0, hints.clone());

        let looked_up = cache.lookup_hints(&poly, 0);
        assert!(looked_up.is_some());
        assert_eq!(looked_up.unwrap().len(), 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = RootHintsCache::new();
        let poly = Polynomial::from_var(0);

        let result = cache.lookup_hints(&poly, 0);
        assert!(result.is_none());
        assert_eq!(cache.stats.num_cache_misses, 1);
    }

    #[test]
    fn test_cache_hit() {
        let mut cache = RootHintsCache::new();
        let poly = Polynomial::from_var(0);
        let hints = vec![RootHint::new(rat(0), rat(1), 1)];

        cache.store_hints(&poly, 0, hints);
        let _result = cache.lookup_hints(&poly, 0);

        assert_eq!(cache.stats.num_cache_hits, 1);
    }

    #[test]
    fn test_refine_hint() {
        let mut cache = RootHintsCache::new();
        let poly = Polynomial::from_var(0);
        let hints = vec![RootHint::new(rat(0), rat(10), 1)];

        cache.store_hints(&poly, 0, hints);

        let success = cache.refine_hint(&poly, 0, 0, rat(2), rat(8));
        assert!(success);

        let refined = cache.lookup_hints(&poly, 0).unwrap();
        assert_eq!(refined[0].lower, rat(2));
        assert_eq!(refined[0].upper, rat(8));
        assert_eq!(refined[0].refinement_level, 2);
    }

    #[test]
    fn test_num_roots() {
        let mut cache = RootHintsCache::new();
        let poly = Polynomial::from_var(0);
        let hints = vec![
            RootHint::new(rat(0), rat(1), 1),
            RootHint::new(rat(5), rat(6), 1),
        ];

        cache.store_hints(&poly, 0, hints);

        assert_eq!(cache.num_roots(&poly, 0), Some(2));
    }

    #[test]
    fn test_eviction() {
        let config = RootHintsConfig {
            max_polynomials: 2,
            ..Default::default()
        };
        let mut cache = RootHintsCache::with_config(config);

        // Add 3 polynomials (should evict the first)
        for i in 0..3 {
            let poly = Polynomial::from_var(i);
            let hints = vec![RootHint::new(rat(0), rat(1), 1)];
            cache.store_hints(&poly, i, hints);
        }

        assert_eq!(cache.cache_size(), 2);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = RootHintsCache::new();
        let poly = Polynomial::from_var(0);
        let hints = vec![RootHint::new(rat(0), rat(1), 1)];

        cache.store_hints(&poly, 0, hints);

        // One hit
        let _result1 = cache.lookup_hints(&poly, 0);

        // One miss
        let poly2 = Polynomial::from_var(1);
        let _result2 = cache.lookup_hints(&poly2, 1);

        assert_eq!(cache.stats().hit_rate(), 0.5);
    }

    #[test]
    fn test_clear() {
        let mut cache = RootHintsCache::new();
        let poly = Polynomial::from_var(0);
        let hints = vec![RootHint::new(rat(0), rat(1), 1)];

        cache.store_hints(&poly, 0, hints);
        assert!(cache.cache_size() > 0);

        cache.clear();
        assert_eq!(cache.cache_size(), 0);
    }

    #[test]
    fn test_hints_limit() {
        let config = RootHintsConfig {
            max_hints_per_poly: 2,
            ..Default::default()
        };
        let mut cache = RootHintsCache::with_config(config);

        let poly = Polynomial::from_var(0);
        let hints = vec![
            RootHint::new(rat(0), rat(1), 1),
            RootHint::new(rat(2), rat(3), 1),
            RootHint::new(rat(4), rat(5), 1),
        ];

        cache.store_hints(&poly, 0, hints);

        // Should only store 2 hints
        assert_eq!(cache.num_roots(&poly, 0), Some(2));
    }
}
