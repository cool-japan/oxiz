//! Efficient substitution with structural sharing
//!
//! This module provides optimized term substitution for E-matching instantiations.
//! Substitutions map bound variables to ground terms, and are applied when
//! instantiating quantified formulas.
//!
//! # Design Principles
//!
//! - **Structural Sharing**: Reuse unchanged subterms to minimize allocations
//! - **Caching**: Cache substitution results to avoid redundant work
//! - **Incremental**: Support incremental substitution updates
//!
//! # Algorithm
//!
//! Based on Z3's substitution implementation in src/ast/substitution.cpp.
//! The walk itself lives in the private `apply` submodule, which replaced a
//! native recursion with an explicit heap work list; see that module's
//! documentation for why it does not delegate to
//! [`crate::ast::TermManager::substitute`].

mod apply;

use crate::ast::{TermId, TermManager};
use crate::error::Result;
use crate::interner::Spur;
#[allow(unused_imports)]
use crate::prelude::*;
use core::fmt;

/// A substitution mapping variables to terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Substitution {
    /// Variable name to term mapping
    bindings: FxHashMap<Spur, TermId>,
}

impl Substitution {
    /// Create an empty substitution
    pub fn new() -> Self {
        Self {
            bindings: FxHashMap::default(),
        }
    }

    /// Create a substitution with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bindings: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Insert a binding
    pub fn insert(&mut self, var: Spur, term: TermId) {
        self.bindings.insert(var, term);
    }

    /// Remove a binding, returning the term it was bound to.
    ///
    /// Used by the E-matching machine to undo a binding recorded on a branch
    /// that later failed (see `quantifier_inst::matcher`).
    pub fn remove(&mut self, var: &Spur) -> Option<TermId> {
        self.bindings.remove(var)
    }

    /// Get a binding
    pub fn get(&self, var: &Spur) -> Option<TermId> {
        self.bindings.get(var).copied()
    }

    /// Check if a variable is bound
    pub fn contains(&self, var: &Spur) -> bool {
        self.bindings.contains_key(var)
    }

    /// Get the number of bindings
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Check if the substitution is empty
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Iterate over bindings
    pub fn iter(&self) -> impl Iterator<Item = (&Spur, &TermId)> {
        self.bindings.iter()
    }

    /// Clear all bindings
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    /// Apply this substitution to a term.
    ///
    /// Returns [`crate::error::OxizError::EmatchError`] if the term -- or any
    /// subterm reached while walking it -- is not present in `manager`.
    pub fn apply(&self, term: TermId, manager: &mut TermManager) -> Result<TermId> {
        apply::apply_bindings(&self.bindings, term, manager)
    }
}

impl Default for Substitution {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Substitution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (var, term)) in self.bindings.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?} -> {:?}", var, term)?;
        }
        write!(f, "}}")
    }
}

/// Builder for constructing substitutions
#[derive(Debug)]
pub struct SubstitutionBuilder {
    subst: Substitution,
}

impl SubstitutionBuilder {
    /// Create a new substitution builder
    pub fn new() -> Self {
        Self {
            subst: Substitution::new(),
        }
    }

    /// Create with initial capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            subst: Substitution::with_capacity(capacity),
        }
    }

    /// Add a binding
    pub fn bind(mut self, var: Spur, term: TermId) -> Self {
        self.subst.insert(var, term);
        self
    }

    /// Build the substitution
    pub fn build(self) -> Substitution {
        self.subst
    }
}

impl Default for SubstitutionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for substitution
#[derive(Debug, Clone)]
pub struct SubstitutionConfig {
    /// Whether to enable caching of substitution results
    pub enable_cache: bool,
    /// Maximum cache size (0 = unlimited)
    pub max_cache_size: usize,
}

impl Default for SubstitutionConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            max_cache_size: 10000,
        }
    }
}

/// Cache for substitution results
#[derive(Debug)]
pub struct SubstitutionCache {
    /// Configuration
    config: SubstitutionConfig,
    /// Cache mapping (term, subst_hash) to result
    cache: FxHashMap<(TermId, u64), TermId>,
    /// Statistics
    stats: SubstitutionStats,
}

/// Statistics about substitutions
#[derive(Debug, Clone, Default)]
pub struct SubstitutionStats {
    /// Number of substitutions applied
    pub substitutions_applied: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Number of terms substituted
    pub terms_substituted: usize,
}

impl SubstitutionCache {
    /// Create a new substitution cache
    pub fn new(config: SubstitutionConfig) -> Self {
        Self {
            config,
            cache: FxHashMap::default(),
            stats: SubstitutionStats::default(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(SubstitutionConfig::default())
    }

    /// Apply a substitution with caching
    pub fn apply(
        &mut self,
        term: TermId,
        subst: &Substitution,
        manager: &mut TermManager,
    ) -> Result<TermId> {
        self.stats.substitutions_applied += 1;

        if !self.config.enable_cache {
            return subst.apply(term, manager);
        }

        // Compute a hash of the substitution for caching
        let subst_hash = self.hash_substitution(subst);
        let key = (term, subst_hash);

        // Check cache
        if let Some(&result) = self.cache.get(&key) {
            self.stats.cache_hits += 1;
            return Ok(result);
        }

        self.stats.cache_misses += 1;

        // Apply substitution
        let result = subst.apply(term, manager)?;

        // Cache result if within limits
        if self.config.max_cache_size == 0 || self.cache.len() < self.config.max_cache_size {
            self.cache.insert(key, result);
        }

        Ok(result)
    }

    /// Hash a substitution for caching
    fn hash_substitution(&self, subst: &Substitution) -> u64 {
        use crate::prelude::hash_map::DefaultHasher;
        use core::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Sort bindings for consistent hashing
        let mut bindings: Vec<_> = subst.iter().map(|(&k, &v)| (k.into_inner(), v)).collect();
        bindings.sort_by_key(|(k, _)| *k);

        for (k, v) in bindings {
            k.hash(&mut hasher);
            v.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.stats = SubstitutionStats::default();
    }

    /// Get statistics
    pub fn stats(&self) -> &SubstitutionStats {
        &self.stats
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.stats.cache_hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{TermKind, TermManager};

    fn setup() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_substitution_creation() {
        let subst = Substitution::new();
        assert!(subst.is_empty());
        assert_eq!(subst.len(), 0);
    }

    #[test]
    fn test_substitution_insert_get() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let x_name = manager.intern_str("x");
        let five = manager.mk_int(5);

        subst.insert(x_name, five);

        assert_eq!(subst.len(), 1);
        assert_eq!(subst.get(&x_name), Some(five));
        assert!(subst.contains(&x_name));
    }

    #[test]
    fn test_substitution_apply_variable() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        let result = subst
            .apply(x, &mut manager)
            .expect("test operation should succeed");
        assert_eq!(result, five);
    }

    #[test]
    fn test_substitution_apply_non_bound_variable() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        // y is not bound, should remain unchanged
        let result = subst
            .apply(y, &mut manager)
            .expect("test operation should succeed");
        assert_eq!(result, y);
    }

    #[test]
    fn test_substitution_apply_complex_term() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add([x, y]);

        let five = manager.mk_int(5);
        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        // sum is (x + y), after substitution should be (5 + y)
        let result = subst
            .apply(sum, &mut manager)
            .expect("test operation should succeed");

        // Verify result is not the same term
        assert_ne!(result, sum);

        // Verify it's structurally (5 + y)
        if let Some(result_term) = manager.get(result) {
            if let TermKind::Add(args) = &result_term.kind {
                assert!(args.contains(&five));
                assert!(args.contains(&y));
            } else {
                panic!("Expected Add term");
            }
        } else {
            panic!("Result term not found");
        }
    }

    #[test]
    fn test_substitution_builder() {
        let mut manager = setup();
        let x_name = manager.intern_str("x");
        let y_name = manager.intern_str("y");
        let five = manager.mk_int(5);
        let ten = manager.mk_int(10);

        let subst = SubstitutionBuilder::new()
            .bind(x_name, five)
            .bind(y_name, ten)
            .build();

        assert_eq!(subst.len(), 2);
        assert_eq!(subst.get(&x_name), Some(five));
        assert_eq!(subst.get(&y_name), Some(ten));
    }

    #[test]
    fn test_substitution_config_default() {
        let config = SubstitutionConfig::default();
        assert!(config.enable_cache);
        assert_eq!(config.max_cache_size, 10000);
    }

    #[test]
    fn test_substitution_cache() {
        let mut manager = setup();
        let mut cache = SubstitutionCache::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        let subst = SubstitutionBuilder::new().bind(x_name, five).build();

        // First application - cache miss
        let result1 = cache
            .apply(x, &subst, &mut manager)
            .expect("test operation should succeed");
        assert_eq!(cache.stats.cache_misses, 1);
        assert_eq!(cache.stats.cache_hits, 0);

        // Second application - cache hit
        let result2 = cache
            .apply(x, &subst, &mut manager)
            .expect("test operation should succeed");
        assert_eq!(cache.stats.cache_hits, 1);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_substitution_structural_sharing() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let f_y = manager.mk_apply("f", [y], int_sort);
        let sum = manager.mk_add([x, f_y]);

        let five = manager.mk_int(5);
        let x_name = manager.intern_str("x");
        subst.insert(x_name, five);

        // Apply substitution: (x + f(y)) becomes (5 + f(y))
        let result = subst
            .apply(sum, &mut manager)
            .expect("test operation should succeed");

        // f(y) should be shared (not re-created)
        if let Some(result_term) = manager.get(result)
            && let TermKind::Add(args) = &result_term.kind
        {
            // One of the args should be f(y) unchanged
            assert!(args.contains(&f_y));
        }
    }

    #[test]
    fn test_substitution_clear() {
        let mut manager = setup();
        let mut subst = Substitution::new();

        let x_name = manager.intern_str("x");
        let five = manager.mk_int(5);

        subst.insert(x_name, five);
        assert_eq!(subst.len(), 1);

        subst.clear();
        assert_eq!(subst.len(), 0);
        assert!(subst.is_empty());
    }

    #[test]
    fn test_substitution_ground_term_unchanged() {
        let mut manager = setup();
        let subst = Substitution::new();

        let five = manager.mk_int(5);
        let result = subst
            .apply(five, &mut manager)
            .expect("test operation should succeed");

        // Ground term should remain unchanged
        assert_eq!(result, five);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut manager = setup();
        let mut cache = SubstitutionCache::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);

        let x_name = manager.intern_str("x");
        let subst = SubstitutionBuilder::new().bind(x_name, five).build();

        // 1 miss, 2 hits
        cache
            .apply(x, &subst, &mut manager)
            .expect("test operation should succeed");
        cache
            .apply(x, &subst, &mut manager)
            .expect("test operation should succeed");
        cache
            .apply(x, &subst, &mut manager)
            .expect("test operation should succeed");

        let hit_rate = cache.hit_rate();
        assert!((hit_rate - 2.0 / 3.0).abs() < 0.01);
    }

    // ===== Regression pins for the iterative rewrite =====

    /// A term far deeper than any native stack could carry: the retired
    /// recursion spent one frame per level and aborted the process here.
    #[test]
    fn test_substitution_deeply_nested_term() {
        let worker = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = setup();
                let int_sort = manager.sorts.int_sort;
                let x = manager.mk_var("x", int_sort);

                // f(f(f(...f(x)...))) nested 6_250 deep.
                let depth = 6_250;
                let mut term = x;
                for _ in 0..depth {
                    term = manager.mk_apply("f", [term], int_sort);
                }

                let five = manager.mk_int(5);
                let x_name = manager.intern_str("x");
                let mut subst = Substitution::new();
                subst.insert(x_name, five);

                let result = subst
                    .apply(term, &mut manager)
                    .expect("deep substitution should succeed");
                assert_ne!(result, term);

                // Walk back down iteratively and confirm the leaf became 5.
                let mut cursor = result;
                for _ in 0..depth {
                    let next = match manager.get(cursor).map(|t| t.kind.clone()) {
                        Some(TermKind::Apply { args, .. }) if args.len() == 1 => args[0],
                        other => panic!("expected unary Apply, got {other:?}"),
                    };
                    cursor = next;
                }
                assert_eq!(cursor, five);
            })
            .expect("spawning the deep-substitution worker should succeed");

        worker.join().expect("deep substitution must not overflow");
    }

    /// A shared subterm must be expanded once, not once per parent edge. The
    /// recursion had no memo, so this DAG (depth 40, every level sharing its
    /// child twice) took 2^40 steps; it now completes immediately.
    #[test]
    fn test_substitution_shared_dag_is_memoized() {
        let mut manager = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        let mut term = x;
        for _ in 0..40 {
            term = manager.mk_apply("g", [term, term], int_sort);
        }

        let five = manager.mk_int(5);
        let x_name = manager.intern_str("x");
        let mut subst = Substitution::new();
        subst.insert(x_name, five);

        let result = subst
            .apply(term, &mut manager)
            .expect("DAG substitution should succeed");
        assert_ne!(result, term);
    }

    /// The retired `_ => Ok(term)` catch-all returned bit-vector nodes
    /// unsubstituted, i.e. produced an instantiation lemma still mentioning
    /// the quantified variable.
    #[test]
    fn test_substitution_reaches_bitvector_children() {
        let mut manager = setup();
        let bv_sort = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv_sort);
        let c = manager.mk_bitvec(7u32, 8);
        let sum = manager.mk_bv_add(x, c);

        let replacement = manager.mk_bitvec(3u32, 8);
        let x_name = manager.intern_str("x");
        let mut subst = Substitution::new();
        subst.insert(x_name, replacement);

        let result = subst
            .apply(sum, &mut manager)
            .expect("bit-vector substitution should succeed");
        assert_ne!(result, sum, "BvAdd child must be substituted, not skipped");

        let free = crate::ast::traversal::collect_subterms(result, &manager);
        assert!(
            !free.contains(&x),
            "the substituted variable must not survive in the result"
        );
    }

    /// Same gap on the string side.
    #[test]
    fn test_substitution_reaches_string_children() {
        let mut manager = setup();
        let str_sort = manager.sorts.string_sort();
        let x = manager.mk_var("x", str_sort);
        let lit = manager.mk_string_lit("ab");
        let concat = manager.mk_str_concat(x, lit);

        let replacement = manager.mk_string_lit("zz");
        let x_name = manager.intern_str("x");
        let mut subst = Substitution::new();
        subst.insert(x_name, replacement);

        let result = subst
            .apply(concat, &mut manager)
            .expect("string substitution should succeed");
        let subterms = crate::ast::traversal::collect_subterms(result, &manager);
        assert!(!subterms.contains(&x));
    }

    /// A binder that rebinds the same name shadows the substitution: the
    /// inner occurrence is the binder's own variable and must survive.
    #[test]
    fn test_substitution_binder_shadowing() {
        let mut manager = setup();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let x = manager.mk_var("x", int_sort);
        let inner = manager.mk_apply("p", [x], bool_sort);
        let forall = manager.mk_forall_with_patterns(
            vec![("x", int_sort)],
            inner,
            Vec::<Vec<TermId>>::new(),
        );

        let five = manager.mk_int(5);
        let x_name = manager.intern_str("x");
        let mut subst = Substitution::new();
        subst.insert(x_name, five);

        let result = subst
            .apply(forall, &mut manager)
            .expect("shadowed substitution should succeed");
        assert_eq!(result, forall, "a shadowed name must not be substituted");
    }

    /// A free occurrence under a binder that binds a *different* name is
    /// still substituted.
    #[test]
    fn test_substitution_under_unrelated_binder() {
        let mut manager = setup();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let inner = manager.mk_apply("p", [x, y], bool_sort);
        let forall = manager.mk_forall_with_patterns(
            vec![("y", int_sort)],
            inner,
            Vec::<Vec<TermId>>::new(),
        );

        let five = manager.mk_int(5);
        let x_name = manager.intern_str("x");
        let mut subst = Substitution::new();
        subst.insert(x_name, five);

        let result = subst
            .apply(forall, &mut manager)
            .expect("substitution under a binder should succeed");
        assert_ne!(result, forall);
        let subterms = crate::ast::traversal::collect_subterms(result, &manager);
        assert!(!subterms.contains(&x));
        assert!(subterms.contains(&y));
    }

    /// A dangling `TermId` is reported, not silently treated as unchanged.
    #[test]
    fn test_substitution_missing_term_errors() {
        let mut manager = setup();
        let subst = Substitution::new();
        let dangling = TermId::new(u32::MAX - 1);
        assert!(subst.apply(dangling, &mut manager).is_err());
    }
}
