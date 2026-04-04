//! Term Manager - Arena allocation for terms

use super::term::{Term, TermId, TermKind};
use super::traversal::get_children;
#[cfg(feature = "arena")]
use crate::ast::arena::TermArena;
use crate::interner::{Rodeo, Spur};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortManager};
use portable_atomic::{AtomicU32, Ordering};

mod builder;
mod query;

/// Statistics for garbage collection
#[derive(Debug, Clone, Default)]
pub struct GCStatistics {
    /// Number of GC runs
    pub gc_count: usize,
    /// Total terms collected across all GC runs
    pub total_collected: usize,
    /// Total cache entries removed across all GC runs
    pub total_cache_removed: usize,
    /// Last GC collection count
    pub last_collected: usize,
    /// Last GC cache removal count
    pub last_cache_removed: usize,
}

/// Manager for term allocation and interning
#[derive(Debug)]
pub struct TermManager {
    /// Arena for term storage
    pub(super) terms: Vec<Term>,
    /// Next term ID
    pub(super) next_id: AtomicU32,
    /// String interner for symbols
    pub(super) interner: Rodeo,
    /// Sort manager
    pub sorts: SortManager,
    /// Cache for structural sharing
    pub(super) cache: FxHashMap<TermKind, TermId>,
    /// True constant
    pub true_id: TermId,
    /// False constant
    pub false_id: TermId,
    /// GC statistics
    pub(super) gc_stats: GCStatistics,
    /// Optional bump arena for fast allocation (feature-gated)
    #[cfg(feature = "arena")]
    pub(super) arena: TermArena,
}

impl Default for TermManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TermManager {
    /// Create a new term manager
    #[must_use]
    pub fn new() -> Self {
        let sorts = SortManager::new();
        let bool_sort = sorts.bool_sort;

        let mut manager = Self {
            terms: Vec::with_capacity(1024),
            next_id: AtomicU32::new(0),
            interner: Rodeo::default(),
            sorts,
            cache: FxHashMap::default(),
            true_id: TermId(0),
            false_id: TermId(1),
            gc_stats: GCStatistics::default(),
            #[cfg(feature = "arena")]
            arena: TermArena::with_capacity(64 * 1024),
        };

        // Pre-allocate true and false
        manager.true_id = manager.intern(TermKind::True, bool_sort);
        manager.false_id = manager.intern(TermKind::False, bool_sort);

        manager
    }

    /// Intern a term kind with an explicit sort, returning its unique ID.
    ///
    /// This is the public-facing version of the internal `intern` method,
    /// intended for use by crates that need to construct term kinds directly
    /// (e.g. when rebuilding quantifiers with substituted bodies).
    pub fn intern_term(&mut self, kind: TermKind, sort: SortId) -> TermId {
        self.intern(kind, sort)
    }

    /// Intern a term, returning its unique ID
    pub(crate) fn intern(&mut self, kind: TermKind, sort: SortId) -> TermId {
        if let Some(&id) = self.cache.get(&kind) {
            return id;
        }

        let id = TermId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let term = Term {
            id,
            kind: kind.clone(),
            sort,
        };
        // When the arena feature is enabled, also allocate in the bump arena
        #[cfg(feature = "arena")]
        {
            let _ = self.arena.alloc_term(id, kind.clone(), sort);
        }

        self.terms.push(term);
        self.cache.insert(kind, id);
        id
    }

    /// Get arena statistics (only available with `arena` feature)
    #[cfg(feature = "arena")]
    #[must_use]
    pub fn arena_stats(&self) -> crate::ast::arena::ArenaStats {
        self.arena.stats()
    }

    /// Reset the arena allocator, freeing all arena memory
    #[cfg(feature = "arena")]
    pub fn reset_arena(&mut self) {
        self.arena.reset();
    }

    /// Get a term by its ID
    #[must_use]
    pub fn get(&self, id: TermId) -> Option<&Term> {
        self.terms.get(id.0 as usize)
    }

    /// Intern a string, returning its key
    pub fn intern_str(&mut self, s: &str) -> Spur {
        self.interner.get_or_intern(s)
    }

    /// Resolve an interned string
    #[must_use]
    pub fn resolve_str(&self, key: Spur) -> &str {
        self.interner.resolve(&key)
    }

    /// Get the number of terms allocated
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if the manager is empty (only contains true/false)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.len() <= 2
    }

    // ========================== Garbage Collection ==========================

    /// Perform garbage collection on unreachable terms
    ///
    /// This method performs a mark-and-sweep garbage collection:
    /// 1. Marks all terms reachable from the given root set
    /// 2. Removes unmarked entries from the cache
    ///
    /// Note: This doesn't actually free memory from the arena (terms vector),
    /// but it does clean up the cache to prevent unbounded growth.
    ///
    /// # Arguments
    /// * `roots` - Set of root term IDs to keep (and their descendants)
    ///
    /// # Returns
    /// Number of cache entries removed
    pub fn gc(&mut self, roots: &FxHashSet<TermId>) -> usize {
        // Mark phase: find all reachable terms
        let mut reachable = FxHashSet::default();
        let mut worklist: Vec<TermId> = roots.iter().copied().collect();

        // Always keep true and false
        worklist.push(self.true_id);
        worklist.push(self.false_id);

        while let Some(id) = worklist.pop() {
            if !reachable.insert(id) {
                continue; // Already visited
            }

            // Mark children as reachable
            if let Some(term) = self.get(id) {
                for child in get_children(&term.kind) {
                    if !reachable.contains(&child) {
                        worklist.push(child);
                    }
                }
            }
        }

        // Sweep phase: remove unreachable entries from cache
        let original_cache_size = self.cache.len();
        self.cache.retain(|_, &mut id| reachable.contains(&id));
        let removed = original_cache_size - self.cache.len();

        // Update statistics
        self.gc_stats.gc_count += 1;
        self.gc_stats.total_cache_removed += removed;
        self.gc_stats.last_cache_removed = removed;
        self.gc_stats.last_collected = removed;
        self.gc_stats.total_collected += removed;

        removed
    }

    /// Perform aggressive garbage collection
    ///
    /// Similar to `gc()` but more thorough. It also shrinks the cache capacity
    /// to fit the retained entries, potentially freeing more memory.
    ///
    /// # Arguments
    /// * `roots` - Set of root term IDs to keep (and their descendants)
    ///
    /// # Returns
    /// Number of cache entries removed
    pub fn gc_aggressive(&mut self, roots: &FxHashSet<TermId>) -> usize {
        let removed = self.gc(roots);
        self.cache.shrink_to_fit();
        removed
    }

    /// Get garbage collection statistics
    #[must_use]
    pub fn gc_statistics(&self) -> &GCStatistics {
        &self.gc_stats
    }

    /// Get the current cache size (number of hash-consed terms)
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get the total number of terms allocated
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Clear all GC statistics
    pub fn reset_gc_stats(&mut self) {
        self.gc_stats = GCStatistics::default();
    }
}

/// Builder for constructing substitutions incrementally with optimizations
///
/// This provides better performance than repeatedly calling substitute when
/// building up complex substitutions, especially when:
/// - Composing multiple substitutions
/// - Applying the same substitution to many terms
/// - Building substitutions incrementally
#[derive(Debug, Clone)]
pub struct SubstitutionBuilder {
    /// The substitution mapping
    mapping: FxHashMap<TermId, TermId>,
    /// Shared cache for substitution results
    cache: FxHashMap<TermId, TermId>,
}

impl SubstitutionBuilder {
    /// Create a new empty substitution builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            mapping: FxHashMap::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create a builder with initial capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            mapping: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            cache: FxHashMap::with_capacity_and_hasher(capacity * 2, Default::default()),
        }
    }

    /// Add a substitution mapping
    pub fn add(&mut self, from: TermId, to: TermId) -> &mut Self {
        // Invalidate cache when adding new mapping
        self.cache.clear();
        self.mapping.insert(from, to);
        self
    }

    /// Add multiple substitution mappings
    pub fn add_many(&mut self, mappings: impl IntoIterator<Item = (TermId, TermId)>) -> &mut Self {
        self.cache.clear();
        self.mapping.extend(mappings);
        self
    }

    /// Compose this substitution with another
    ///
    /// The resulting substitution applies `other` first, then `self`.
    /// This is optimized to share structure where possible.
    pub fn compose(&mut self, other: &SubstitutionBuilder, manager: &mut TermManager) -> &mut Self {
        // For each mapping in self, substitute using other
        let mut new_mapping = FxHashMap::default();
        let mut temp_cache = FxHashMap::default();

        for (&from, &to) in &self.mapping {
            let new_to = if other.mapping.contains_key(&to) {
                manager.substitute_cached(to, &other.mapping, &mut temp_cache)
            } else {
                to
            };
            new_mapping.insert(from, new_to);
        }

        // Add mappings from other that aren't in self
        for (&from, &to) in &other.mapping {
            new_mapping.entry(from).or_insert(to);
        }

        self.mapping = new_mapping;
        self.cache.clear();
        self
    }

    /// Apply the substitution to a term
    ///
    /// This uses a persistent cache across multiple applications,
    /// making it more efficient when substituting many terms.
    pub fn apply(&mut self, id: TermId, manager: &mut TermManager) -> TermId {
        manager.substitute_cached(id, &self.mapping, &mut self.cache)
    }

    /// Apply the substitution to multiple terms efficiently
    ///
    /// Uses the shared cache to avoid redundant work.
    pub fn apply_many(&mut self, ids: &[TermId], manager: &mut TermManager) -> Vec<TermId> {
        ids.iter().map(|&id| self.apply(id, manager)).collect()
    }

    /// Get the underlying mapping
    #[must_use]
    pub fn mapping(&self) -> &FxHashMap<TermId, TermId> {
        &self.mapping
    }

    /// Check if the substitution is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// Get the number of mappings
    #[must_use]
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Clear the substitution
    pub fn clear(&mut self) {
        self.mapping.clear();
        self.cache.clear();
    }

    /// Reset the cache (useful for freeing memory)
    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics (for debugging/optimization)
    #[must_use]
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.cache.capacity())
    }
}

impl Default for SubstitutionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        let manager = TermManager::new();
        assert_ne!(manager.mk_true(), manager.mk_false());
        assert_eq!(manager.mk_bool(true), manager.mk_true());
        assert_eq!(manager.mk_bool(false), manager.mk_false());
    }

    #[test]
    fn test_not_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();

        assert_eq!(manager.mk_not(t), f);
        assert_eq!(manager.mk_not(f), t);

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let not_x = manager.mk_not(x);
        let not_not_x = manager.mk_not(not_x);
        assert_eq!(not_not_x, x);
    }

    #[test]
    fn test_and_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        assert_eq!(manager.mk_and([t, x]), x);
        assert_eq!(manager.mk_and([f, x]), f);
        assert_eq!(manager.mk_and([t, t]), t);
        assert_eq!(manager.mk_and(core::iter::empty()), t);
    }

    #[test]
    fn test_or_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        assert_eq!(manager.mk_or([f, x]), x);
        assert_eq!(manager.mk_or([t, x]), t);
        assert_eq!(manager.mk_or([f, f]), f);
        assert_eq!(manager.mk_or(core::iter::empty()), f);
    }

    #[test]
    fn test_eq_canonicalization() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let eq1 = manager.mk_eq(x, y);
        let eq2 = manager.mk_eq(y, x);
        assert_eq!(eq1, eq2);
    }

    #[test]
    fn test_ite_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);

        assert_eq!(manager.mk_ite(t, x, y), x);
        assert_eq!(manager.mk_ite(f, x, y), y);
        assert_eq!(manager.mk_ite(x, t, f), x);
    }

    #[test]
    fn test_interning() {
        let mut manager = TermManager::new();
        let x1 = manager.mk_var("x", manager.sorts.int_sort);
        let x2 = manager.mk_var("x", manager.sorts.int_sort);
        assert_eq!(x1, x2);

        let y = manager.mk_var("y", manager.sorts.int_sort);
        assert_ne!(x1, y);
    }

    #[test]
    fn test_term_size() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.int_sort);

        assert_eq!(manager.term_size(x), 1);

        let add_xy = manager.mk_add([x, y]);
        assert_eq!(manager.term_size(add_xy), 3);

        let add_xyz = manager.mk_add([x, y, z]);
        assert_eq!(manager.term_size(add_xyz), 4);

        let nested = manager.mk_add([add_xy, z]);
        // add_xy has size 3, z has size 1, outer add has size 1
        // But x and y appear only once each due to hash-consing
        assert_eq!(manager.term_size(nested), 5);
    }

    #[test]
    fn test_term_depth() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        assert_eq!(manager.term_depth(x), 0);

        let add_xy = manager.mk_add([x, y]);
        assert_eq!(manager.term_depth(add_xy), 1);

        let nested = manager.mk_add([add_xy, x]);
        assert_eq!(manager.term_depth(nested), 2);
    }

    #[test]
    fn test_substitute() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let c = manager.mk_int(42);

        let expr = manager.mk_add([x, y]);

        let mut subst = FxHashMap::default();
        subst.insert(x, c);

        let result = manager.substitute(expr, &subst);

        let expected = manager.mk_add([c, y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_free_vars() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let c = manager.mk_int(42);

        let expr = manager.mk_add([x, y, c]);
        let vars = manager.free_vars(expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&x));
        assert!(vars.contains(&y));

        let const_expr = manager.mk_int(100);
        let vars = manager.free_vars(const_expr);
        assert!(vars.is_empty());
    }

    // ==================== Quantifier Pattern Tests ====================

    #[test]
    fn test_forall_without_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let gt_zero = manager.mk_gt(x, zero);

        let forall = manager.mk_forall([("x", int_sort)], gt_zero);
        let term = manager.get(forall).expect("forall term should exist");

        assert_eq!(term.sort, bool_sort);
        match &term.kind {
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                assert_eq!(vars.len(), 1);
                assert_eq!(*body, gt_zero);
                assert!(patterns.is_empty(), "should have no patterns");
            }
            _ => panic!("expected Forall term"),
        }
    }

    #[test]
    fn test_forall_with_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let zero = manager.mk_int(0);
        let gt_zero = manager.mk_gt(f_x, zero);

        let forall = manager.mk_forall_with_patterns([("x", int_sort)], gt_zero, [[f_x]]);
        let term = manager.get(forall).expect("forall term should exist");

        assert_eq!(term.sort, bool_sort);
        match &term.kind {
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                assert_eq!(vars.len(), 1);
                assert_eq!(*body, gt_zero);
                assert_eq!(patterns.len(), 1, "should have 1 pattern");
                assert_eq!(patterns[0].len(), 1, "pattern should have 1 term");
                assert_eq!(patterns[0][0], f_x, "pattern term should be f(x)");
            }
            _ => panic!("expected Forall term"),
        }
    }

    #[test]
    fn test_forall_with_multiple_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let g_x = manager.mk_apply("g", [x], int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_gt(f_x, zero);

        // Two patterns: (f x) and (g x)
        let forall = manager.mk_forall_with_patterns([("x", int_sort)], body, [[f_x], [g_x]]);

        match &manager.get(forall).expect("forall term should exist").kind {
            TermKind::Forall { patterns, .. } => {
                assert_eq!(patterns.len(), 2, "should have 2 patterns");
            }
            _ => panic!("expected Forall term"),
        }
    }

    #[test]
    fn test_forall_with_multi_term_pattern() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let g_y = manager.mk_apply("g", [y], int_sort);
        let body = manager.mk_gt(f_x, g_y);

        // One pattern with two terms: (f x) (g y)
        let forall =
            manager.mk_forall_with_patterns([("x", int_sort), ("y", int_sort)], body, [[f_x, g_y]]);

        match &manager.get(forall).expect("forall term should exist").kind {
            TermKind::Forall { patterns, .. } => {
                assert_eq!(patterns.len(), 1, "should have 1 pattern");
                assert_eq!(patterns[0].len(), 2, "pattern should have 2 terms");
            }
            _ => panic!("expected Forall term"),
        }
    }

    #[test]
    fn test_exists_with_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_gt(f_x, zero);

        let exists = manager.mk_exists_with_patterns([("x", int_sort)], body, [[f_x]]);

        match &manager.get(exists).expect("exists term should exist").kind {
            TermKind::Exists { patterns, .. } => {
                assert_eq!(patterns.len(), 1);
            }
            _ => panic!("expected Exists term"),
        }
    }
}
