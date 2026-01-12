//! Incremental CAD for efficient updates and backtracking.
//!
//! This module provides incremental updates to the CAD decomposition,
//! allowing efficient addition of new constraints and backtracking without
//! full recomputation.
//!
//! ## Key Features
//!
//! - **Projection Cache**: Reuse previously computed projections
//! - **Incremental Updates**: Add new polynomials without full recomputation
//! - **Backtracking**: Save and restore CAD state efficiently
//! - **Lazy Evaluation**: Only compute CAD cells when needed
//!
//! ## Reference
//!
//! - Z3's incremental NLSAT solver
//! - "Incremental CAD" techniques from SMT literature

use crate::cad::{CadConfig, CadDecomposer, CadProjection, ProjectionSet};
use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;
use std::collections::HashMap;

/// State snapshot for backtracking.
#[derive(Debug, Clone)]
pub struct CadSnapshot {
    /// Level at which this snapshot was taken.
    level: usize,
    /// Projection sets at the time of snapshot.
    #[allow(dead_code)]
    projection_sets: Vec<ProjectionSet>,
    /// Polynomials added at this level.
    added_polynomials: Vec<Polynomial>,
}

/// Incremental CAD manager.
///
/// Manages CAD decomposition with support for:
/// - Incremental addition of polynomials
/// - Backtracking to previous states
/// - Caching of projection results
pub struct IncrementalCad {
    /// Base CAD decomposer.
    decomposer: CadDecomposer,
    /// All polynomials (accumulated over all levels).
    all_polynomials: Vec<Polynomial>,
    /// Polynomials added at each decision level.
    level_polynomials: Vec<Vec<Polynomial>>,
    /// Snapshots for backtracking (indexed by level).
    #[allow(dead_code)]
    snapshots: Vec<CadSnapshot>,
    /// Current decision level.
    current_level: usize,
    /// Cache of computed projection sets (polynomial set hash -> projection set).
    projection_cache: FxHashMap<u64, ProjectionSet>,
    /// Cache of resultants (poly1 hash, poly2 hash, var) -> resultant.
    resultant_cache: HashMap<(u64, u64, Var), Polynomial>,
    /// Track which variables each projection cache entry depends on.
    /// Maps cache key -> set of variables.
    projection_vars: FxHashMap<u64, Vec<Var>>,
    /// Enable caching.
    caching_enabled: bool,
}

impl IncrementalCad {
    /// Create a new incremental CAD manager.
    pub fn new(config: CadConfig, var_order: Vec<Var>) -> Self {
        let decomposer = CadDecomposer::new(config.clone(), var_order);
        let caching_enabled = config.cache_projections;

        Self {
            decomposer,
            all_polynomials: Vec::new(),
            level_polynomials: vec![Vec::new()], // Level 0
            snapshots: Vec::new(),
            current_level: 0,
            projection_cache: FxHashMap::default(),
            resultant_cache: HashMap::new(),
            projection_vars: FxHashMap::default(),
            caching_enabled,
        }
    }

    /// Create with automatic variable ordering.
    pub fn with_auto_ordering(config: CadConfig, initial_polynomials: &[Polynomial]) -> Self {
        let decomposer = CadDecomposer::with_auto_ordering(config.clone(), initial_polynomials);
        let caching_enabled = config.cache_projections;

        let mut instance = Self {
            decomposer,
            all_polynomials: Vec::new(),
            level_polynomials: vec![Vec::new()],
            snapshots: Vec::new(),
            current_level: 0,
            projection_cache: FxHashMap::default(),
            resultant_cache: HashMap::new(),
            projection_vars: FxHashMap::default(),
            caching_enabled,
        };

        // Add initial polynomials
        for poly in initial_polynomials {
            instance.add_polynomial(poly.clone());
        }

        instance
    }

    /// Push a new decision level.
    pub fn push_level(&mut self) {
        self.current_level += 1;

        // Ensure we have enough level slots
        while self.level_polynomials.len() <= self.current_level {
            self.level_polynomials.push(Vec::new());
        }
    }

    /// Pop decision level (backtrack).
    ///
    /// Removes all polynomials added at levels > `target_level`.
    pub fn pop_level(&mut self, target_level: usize) {
        if target_level >= self.current_level {
            return; // Nothing to do
        }

        // Remove polynomials added at levels > target_level
        let mut to_remove = Vec::new();
        for level in (target_level + 1)..=self.current_level {
            if level < self.level_polynomials.len() {
                to_remove.append(&mut self.level_polynomials[level]);
            }
        }

        // Remove from all_polynomials
        self.all_polynomials
            .retain(|p| !to_remove.iter().any(|r| polynomials_equal(p, r)));

        // Truncate level_polynomials
        self.level_polynomials.truncate(target_level + 1);

        // Clear cached projections if polynomials changed
        if !to_remove.is_empty() && self.caching_enabled {
            // Fine-grained invalidation: only clear cache entries involving removed variables
            let affected_vars: Vec<Var> = to_remove.iter().flat_map(|p| p.vars()).collect();
            self.invalidate_cache_for_vars(&affected_vars);
        }

        self.current_level = target_level;
    }

    /// Invalidate cache entries that depend on the given variables.
    fn invalidate_cache_for_vars(&mut self, vars: &[Var]) {
        if vars.is_empty() {
            return;
        }

        // Find cache keys to remove
        let keys_to_remove: Vec<u64> = self
            .projection_vars
            .iter()
            .filter_map(|(key, entry_vars)| {
                // Check if any affected variable is used in this cache entry
                if entry_vars.iter().any(|v| vars.contains(v)) {
                    Some(*key)
                } else {
                    None
                }
            })
            .collect();

        // Remove affected cache entries
        for key in keys_to_remove {
            self.projection_cache.remove(&key);
            self.projection_vars.remove(&key);
        }

        // Also clear resultants involving these variables
        self.resultant_cache
            .retain(|&(_, _, var), _| !vars.contains(&var));
    }

    /// Add a polynomial incrementally.
    pub fn add_polynomial(&mut self, poly: Polynomial) {
        if poly.is_zero() || poly.is_constant() {
            return; // Skip trivial polynomials
        }

        // Check if already present
        if self
            .all_polynomials
            .iter()
            .any(|p| polynomials_equal(p, &poly))
        {
            return;
        }

        // Add to current level
        if self.current_level < self.level_polynomials.len() {
            self.level_polynomials[self.current_level].push(poly.clone());
        }

        // Get variables from the new polynomial
        let poly_vars = poly.vars();

        // Add to global set
        self.all_polynomials.push(poly);

        // Invalidate cached projections based on affected variables
        if self.caching_enabled {
            self.invalidate_cache_for_vars(&poly_vars);
        }
    }

    /// Get all active polynomials.
    pub fn polynomials(&self) -> &[Polynomial] {
        &self.all_polynomials
    }

    /// Compute projection with caching.
    pub fn project_with_cache(&mut self, polys: &[Polynomial], var: Var) -> ProjectionSet {
        if !self.caching_enabled {
            // No caching - compute directly
            let projection = CadProjection::new();
            return projection.project(polys, var);
        }

        // Compute hash of polynomial set
        let poly_hash = self.compute_poly_set_hash(polys, var);

        // Check cache
        if let Some(cached) = self.projection_cache.get(&poly_hash) {
            return cached.clone();
        }

        // Not in cache - compute
        let projection = CadProjection::new();
        let result = projection.project(polys, var);

        // Store in cache
        self.projection_cache.insert(poly_hash, result.clone());

        // Track which variables this projection depends on
        let mut vars_set: Vec<Var> = polys.iter().flat_map(|p| p.vars()).collect();
        vars_set.push(var); // Also track the projection variable
        vars_set.sort_unstable();
        vars_set.dedup();
        self.projection_vars.insert(poly_hash, vars_set);

        result
    }

    /// Compute a hash for a set of polynomials.
    fn compute_poly_set_hash(&self, polys: &[Polynomial], var: Var) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        var.hash(&mut hasher);

        // Hash each polynomial
        for poly in polys {
            // Simple hash based on degree and number of terms
            poly.degree(var).hash(&mut hasher);
            poly.num_terms().hash(&mut hasher);
            poly.total_degree().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Compute resultant with caching.
    #[allow(dead_code)]
    pub fn resultant_with_cache(
        &mut self,
        p1: &Polynomial,
        p2: &Polynomial,
        var: Var,
    ) -> Polynomial {
        if !self.caching_enabled {
            return p1.resultant(p2, var);
        }

        let h1 = self.poly_hash(p1);
        let h2 = self.poly_hash(p2);

        // Ensure consistent ordering
        let (hash1, hash2, poly1, poly2) = if h1 <= h2 {
            (h1, h2, p1, p2)
        } else {
            (h2, h1, p2, p1)
        };

        let cache_key = (hash1, hash2, var);

        // Check cache
        if let Some(cached) = self.resultant_cache.get(&cache_key) {
            return cached.clone();
        }

        // Compute
        let result = poly1.resultant(poly2, var);

        // Cache
        self.resultant_cache.insert(cache_key, result.clone());

        result
    }

    /// Compute hash for a polynomial.
    fn poly_hash(&self, poly: &Polynomial) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash degree and number of terms
        poly.total_degree().hash(&mut hasher);
        poly.num_terms().hash(&mut hasher);

        // Hash first few coefficients for uniqueness
        for (i, term) in poly.terms().iter().take(5).enumerate() {
            i.hash(&mut hasher);
            term.coeff.numer().hash(&mut hasher);
            term.coeff.denom().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        CacheStats {
            projection_cache_size: self.projection_cache.len(),
            resultant_cache_size: self.resultant_cache.len(),
            total_polynomials: self.all_polynomials.len(),
            current_level: self.current_level,
        }
    }

    /// Clear all caches.
    pub fn clear_caches(&mut self) {
        self.projection_cache.clear();
        self.resultant_cache.clear();
    }

    /// Get mutable reference to the decomposer (for performing decomposition).
    pub fn decomposer_mut(&mut self) -> &mut CadDecomposer {
        &mut self.decomposer
    }

    /// Get reference to the decomposer.
    pub fn decomposer(&self) -> &CadDecomposer {
        &self.decomposer
    }

    /// Create a snapshot of current state for later restoration.
    pub fn create_snapshot(&self) -> CadSnapshot {
        CadSnapshot {
            level: self.current_level,
            projection_sets: self.decomposer.projection_sets().to_vec(),
            added_polynomials: self.all_polynomials.clone(),
        }
    }

    /// Restore from a snapshot.
    pub fn restore_snapshot(&mut self, snapshot: &CadSnapshot) {
        // Restore level
        self.current_level = snapshot.level;

        // Restore polynomials
        self.all_polynomials = snapshot.added_polynomials.clone();

        // Truncate level_polynomials
        self.level_polynomials.truncate(self.current_level + 1);

        // Clear caches (conservative)
        if self.caching_enabled {
            self.clear_caches();
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of projection sets cached.
    pub projection_cache_size: usize,
    /// Number of resultants cached.
    pub resultant_cache_size: usize,
    /// Total number of polynomials.
    pub total_polynomials: usize,
    /// Current decision level.
    pub current_level: usize,
}

impl CacheStats {
    /// Get cache hit rate estimate (simplistic).
    pub fn cache_efficiency(&self) -> f64 {
        let total_cached = self.projection_cache_size + self.resultant_cache_size;
        if total_cached == 0 {
            return 0.0;
        }
        total_cached as f64 / (self.total_polynomials as f64).max(1.0)
    }
}

/// Check if two polynomials are equal (for deduplication).
fn polynomials_equal(p1: &Polynomial, p2: &Polynomial) -> bool {
    if p1.num_terms() != p2.num_terms() {
        return false;
    }
    if p1.total_degree() != p2.total_degree() {
        return false;
    }
    if p1.max_var() != p2.max_var() {
        return false;
    }

    // More thorough check: compare monic forms
    let p1_monic = p1.make_monic();
    let p2_monic = p2.make_monic();

    p1_monic == p2_monic
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_rational::BigRational;

    fn rat(n: i64) -> BigRational {
        BigRational::from_integer(n.into())
    }

    #[test]
    fn test_incremental_cad_new() {
        let config = CadConfig::default();
        let var_order = vec![0, 1];
        let cad = IncrementalCad::new(config, var_order);

        assert_eq!(cad.current_level, 0);
        assert_eq!(cad.polynomials().len(), 0);
    }

    #[test]
    fn test_incremental_cad_add_polynomial() {
        let config = CadConfig::default();
        let var_order = vec![0];
        let mut cad = IncrementalCad::new(config, var_order);

        // x - 1
        let x = Polynomial::from_var(0);
        let poly = Polynomial::sub(&x, &Polynomial::constant(rat(1)));

        cad.add_polynomial(poly.clone());
        assert_eq!(cad.polynomials().len(), 1);

        // Adding same polynomial again should not duplicate
        cad.add_polynomial(poly);
        assert_eq!(cad.polynomials().len(), 1);
    }

    #[test]
    fn test_incremental_cad_push_pop_level() {
        let config = CadConfig::default();
        let var_order = vec![0];
        let mut cad = IncrementalCad::new(config, var_order);

        let x = Polynomial::from_var(0);
        let poly1 = Polynomial::sub(&x, &Polynomial::constant(rat(1)));

        // Level 0: add poly1
        cad.add_polynomial(poly1.clone());
        assert_eq!(cad.polynomials().len(), 1);
        assert_eq!(cad.current_level, 0);

        // Level 1: add poly2
        cad.push_level();
        assert_eq!(cad.current_level, 1);

        let poly2 = Polynomial::sub(&x, &Polynomial::constant(rat(2)));
        cad.add_polynomial(poly2);
        assert_eq!(cad.polynomials().len(), 2);

        // Backtrack to level 0
        cad.pop_level(0);
        assert_eq!(cad.current_level, 0);
        assert_eq!(cad.polynomials().len(), 1);
        assert!(polynomials_equal(&cad.polynomials()[0], &poly1));
    }

    #[test]
    fn test_incremental_cad_cache_stats() {
        let config = CadConfig::default();
        let var_order = vec![0, 1];
        let cad = IncrementalCad::new(config, var_order);

        let stats = cad.cache_stats();
        assert_eq!(stats.projection_cache_size, 0);
        assert_eq!(stats.resultant_cache_size, 0);
        assert_eq!(stats.total_polynomials, 0);
        assert_eq!(stats.current_level, 0);
    }

    #[test]
    fn test_polynomials_equal() {
        let x = Polynomial::from_var(0);
        let poly1 = Polynomial::sub(&x, &Polynomial::constant(rat(1)));
        let poly2 = Polynomial::sub(&x, &Polynomial::constant(rat(1)));
        let poly3 = Polynomial::sub(&x, &Polynomial::constant(rat(2)));

        assert!(polynomials_equal(&poly1, &poly2));
        assert!(!polynomials_equal(&poly1, &poly3));
    }

    #[test]
    fn test_projection_with_cache() {
        let config = CadConfig::default();
        let var_order = vec![0, 1];
        let mut cad = IncrementalCad::new(config, var_order);

        // x^2 - 1
        let x = Polynomial::from_var(0);
        let x2 = Polynomial::mul(&x, &x);
        let poly = Polynomial::sub(&x2, &Polynomial::constant(rat(1)));

        let polys = vec![poly.clone()];

        // First call - should compute and cache
        let proj1 = cad.project_with_cache(&polys, 0);
        let stats1 = cad.cache_stats();
        assert!(stats1.projection_cache_size > 0);

        // Second call - should hit cache
        let proj2 = cad.project_with_cache(&polys, 0);
        let stats2 = cad.cache_stats();

        // Cache size should be same (same query)
        assert_eq!(stats1.projection_cache_size, stats2.projection_cache_size);

        // Results should be identical
        assert_eq!(proj1.len(), proj2.len());
    }

    #[test]
    fn test_snapshot_restore() {
        let config = CadConfig::default();
        let var_order = vec![0];
        let mut cad = IncrementalCad::new(config, var_order);

        let x = Polynomial::from_var(0);
        let poly1 = Polynomial::sub(&x, &Polynomial::constant(rat(1)));

        cad.add_polynomial(poly1.clone());
        let snapshot = cad.create_snapshot();

        // Add more polynomials
        cad.push_level();
        let poly2 = Polynomial::sub(&x, &Polynomial::constant(rat(2)));
        cad.add_polynomial(poly2);

        assert_eq!(cad.polynomials().len(), 2);

        // Restore snapshot
        cad.restore_snapshot(&snapshot);
        assert_eq!(cad.polynomials().len(), 1);
        assert_eq!(cad.current_level, 0);
    }

    #[test]
    fn test_cache_efficiency() {
        let stats = CacheStats {
            projection_cache_size: 10,
            resultant_cache_size: 5,
            total_polynomials: 20,
            current_level: 2,
        };

        let efficiency = stats.cache_efficiency();
        assert!(efficiency > 0.0);
        assert!(efficiency <= 1.0);
    }
}
