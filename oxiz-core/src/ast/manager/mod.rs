//! Term Manager - Arena allocation for terms

use super::term::{Term, TermId, TermKind};
use super::traversal::get_children;
use crate::interner::{Rodeo, Spur};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortManager};
use hashbrown::HashTable;
use portable_atomic::{AtomicU32, Ordering};

mod builder;
pub mod bv_fold;
mod query;
mod rounding_mode;
pub mod str_fold;

/// Hash of an interning key, i.e. of the pair `(kind, sort)`.
///
/// This is deliberately the *same* value the previous
/// `FxHashMap<(TermKind, SortId), TermId>` computed for the same pair:
/// `Hash for (A, B)` is defined as `a.hash(state)` followed by
/// `b.hash(state)`, so hashing the two components in that order into a
/// default-seeded [`rustc_hash::FxHasher`] reproduces the old key hash bit for
/// bit. Storing ids instead of owned keys was meant to move memory and nothing
/// else, so the hash distribution -- and with it every bucket-occupancy and
/// probe-length property the intern table had before -- is held fixed.
///
/// The crate's specialised [`crate::ast::term_hash::TermKindHasher`] is
/// *not* used here on purpose: it summarises large payloads by their length
/// plus first and last bytes, which is a real change in distribution rather
/// than a change in layout. `TermKindHasher`, `BuildTermKindHasher` and
/// `TermHashMap` therefore stay unused by the manager, pending a separate,
/// benchmarked follow-up that can measure the trade properly.
fn hash_term_key(kind: &TermKind, sort: SortId) -> u64 {
    use core::hash::{Hash, Hasher};

    let mut hasher = rustc_hash::FxHasher::default();
    kind.hash(&mut hasher);
    sort.hash(&mut hasher);
    hasher.finish()
}

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
    /// Intern table for structural sharing: hashes of `(kind, sort)` to the
    /// `TermId` of the one term carrying that pair.
    ///
    /// A term's identity is the pair `(kind, sort)`, not the kind alone: two
    /// structurally identical kinds at different sorts -- most notably `Var`
    /// with the same name -- must intern to distinct terms, or same-named
    /// variables of different sorts alias and silently type-confuse (wrong
    /// terms, wrong models).
    ///
    /// Rather than own a *second* copy of each `TermKind` as a map key, this
    /// stores only the id and resolves both lookups and collisions against
    /// `terms[id]`, so each kind is retained exactly once -- in the `terms`
    /// vector, which is where `get()` reads it from anyway. That is why this
    /// is a [`HashTable`] and not a `HashMap`: a `HashMap` has no way to
    /// express "the key lives somewhere else".
    ///
    /// The consequence for `gc` is that pruning this table frees the id's
    /// *slot* and nothing else. See [`TermManager::gc`].
    pub(super) table: HashTable<TermId>,
    /// True constant
    pub true_id: TermId,
    /// False constant
    pub false_id: TermId,
    /// GC statistics
    pub(super) gc_stats: GCStatistics,
    /// Whether a `RoundingMode`-sorted term has been built in this manager.
    ///
    /// Set by [`TermManager::mk_rounding_mode`] and by the SMT-LIB parser when
    /// it accepts `RoundingMode` as a declared sort. The solver layer reads it
    /// through [`TermManager::rounding_mode_used`] to decide whether the
    /// five-modes distinctness axiom is needed at all, so a script that never
    /// mentions a rounding mode pays nothing for the feature.
    rounding_mode_used: bool,
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
            table: HashTable::new(),
            true_id: TermId(0),
            false_id: TermId(1),
            gc_stats: GCStatistics::default(),
            rounding_mode_used: false,
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
    ///
    /// `kind` is *moved* into the interned [`Term`] on a miss and dropped on a
    /// hit, so a term's kind is stored exactly once, in `terms`. The intern
    /// table holds only ids and compares against `terms[id]` (see the `table`
    /// field documentation), which is why both closures below take the
    /// `terms`/`table` fields apart first: each needs `&self.terms` while the
    /// other half of `self` is borrowed.
    pub(crate) fn intern(&mut self, kind: TermKind, sort: SortId) -> TermId {
        // Identity is the pair (kind, sort): identical kinds at distinct sorts
        // must not alias (see the `table` field documentation).
        let hash = hash_term_key(&kind, sort);

        {
            let Self { terms, table, .. } = self;
            if let Some(&id) = table.find(hash, |&id| {
                terms
                    .get(id.0 as usize)
                    .is_some_and(|t| t.sort == sort && t.kind == kind)
            }) {
                return id;
            }
        }

        // Miss: allocate the id, then push the term *before* recording it, so
        // that the id the table stores is already resolvable through `terms`.
        let id = TermId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.terms.push(Term { id, kind, sort });

        let Self { terms, table, .. } = self;
        table.insert_unique(hash, id, |&id| {
            terms
                .get(id.0 as usize)
                .map_or(0, |t| hash_term_key(&t.kind, t.sort))
        });
        id
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
    /// 2. Removes unmarked entries from the intern table
    ///
    /// Note: This doesn't actually free memory from the arena (terms vector),
    /// but it does clean up the cache to prevent unbounded growth. A swept
    /// term therefore stays readable through [`TermManager::get`], and
    /// interning its kind again allocates a *fresh* id rather than returning
    /// the old one.
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

        // Sweep phase: remove unreachable entries from the intern table
        let original_cache_size = self.table.len();
        self.table.retain(|id| reachable.contains(id));
        let removed = original_cache_size - self.table.len();

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
        // Reallocating the table rehashes every retained id, so this needs the
        // same `terms`-resolving hasher that `intern` inserts with.
        let Self { terms, table, .. } = self;
        table.shrink_to_fit(|&id| {
            terms
                .get(id.0 as usize)
                .map_or(0, |t| hash_term_key(&t.kind, t.sort))
        });
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
        self.table.len()
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
    fn test_bv_commutative_ops_canonicalize_operand_order() {
        let mut manager = TermManager::new();
        let bv8 = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv8);
        let y = manager.mk_var("y", bv8);

        assert_eq!(manager.mk_bv_and(x, y), manager.mk_bv_and(y, x));
        assert_eq!(manager.mk_bv_or(x, y), manager.mk_bv_or(y, x));
        assert_eq!(manager.mk_bv_add(x, y), manager.mk_bv_add(y, x));
        assert_eq!(manager.mk_bv_mul(x, y), manager.mk_bv_mul(y, x));
        assert_eq!(manager.mk_bv_xor(x, y), manager.mk_bv_xor(y, x));
    }

    #[test]
    fn test_bv_commutative_ops_canonicalize_const_var_order() {
        let mut manager = TermManager::new();
        let bv8 = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv8);
        let three = manager.mk_bitvec(3i64, 8);

        assert_eq!(manager.mk_bv_mul(three, x), manager.mk_bv_mul(x, three));
        assert_eq!(manager.mk_bv_add(three, x), manager.mk_bv_add(x, three));
    }

    #[test]
    fn test_bv_sub_does_not_canonicalize_operand_order() {
        let mut manager = TermManager::new();
        let bv8 = manager.sorts.bitvec(8);
        let x = manager.mk_var("x", bv8);
        let y = manager.mk_var("y", bv8);

        assert_ne!(manager.mk_bv_sub(x, y), manager.mk_bv_sub(y, x));
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
    fn test_hash_term_key_matches_the_tuple_key_hash() {
        // `hash_term_key` replaced an `FxHashMap<(TermKind, SortId), _>`, and
        // is meant to compute exactly what that map's key hash did -- so this
        // hashes the tuple the old way and demands the same bits. It is the
        // guarantee that moving the key out of the table changed memory only
        // and left the hash distribution untouched.
        use core::hash::{BuildHasher, Hash, Hasher};

        fn tuple_key_hash(kind: &TermKind, sort: SortId) -> u64 {
            let mut hasher = rustc_hash::FxHasher::default();
            (kind.clone(), sort).hash(&mut hasher);
            hasher.finish()
        }

        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let spur = manager.intern_str("x");

        let cases = [
            (TermKind::True, bool_sort),
            (TermKind::False, bool_sort),
            (TermKind::Var(spur), int_sort),
            // Same kind, different sort: must hash differently, or the two
            // would always land in the same bucket.
            (TermKind::Var(spur), bool_sort),
            (TermKind::Add(smallvec::smallvec![x, y]), int_sort),
            (TermKind::StringLit("a longer payload".into()), int_sort),
        ];

        for (kind, sort) in &cases {
            assert_eq!(
                hash_term_key(kind, *sort),
                tuple_key_hash(kind, *sort),
                "hash drifted from the tuple key hash for {kind:?} at {sort:?}"
            );
        }

        // And the same bits an FxHashMap itself would derive for that key.
        let build = rustc_hash::FxBuildHasher;
        for (kind, sort) in &cases {
            assert_eq!(
                hash_term_key(kind, *sort),
                build.hash_one((kind.clone(), *sort))
            );
        }

        assert_ne!(
            hash_term_key(&TermKind::Var(spur), int_sort),
            hash_term_key(&TermKind::Var(spur), bool_sort),
            "the sort must participate in the hash"
        );
    }

    // ==================== Interning identity tests ====================
    //
    // These pin the *identity* contract of `intern`: what counts as "the same
    // term". They are deliberately written against the public API only, so
    // they hold regardless of how the intern table is represented internally.

    #[test]
    fn test_repeated_intern_of_same_key_allocates_one_term() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let spur = manager.intern_str("repeated");

        let before = manager.term_count();
        let first = manager.intern_term(TermKind::Var(spur), int_sort);
        assert_eq!(
            manager.term_count(),
            before + 1,
            "the first intern allocates exactly one term"
        );

        for _ in 0..1000 {
            assert_eq!(manager.intern_term(TermKind::Var(spur), int_sort), first);
        }
        assert_eq!(
            manager.term_count(),
            before + 1,
            "1000 further interns of the same (kind, sort) must allocate nothing"
        );
    }

    #[test]
    fn test_same_named_var_at_different_sorts_stays_distinct() {
        // The reason interning is keyed on (kind, sort) and not on kind alone:
        // `x : Int` and `x : Bool` share a `TermKind::Var(spur)` but must never
        // alias, or every same-named variable of a different sort would type-
        // confuse (wrong terms, wrong models).
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;

        let x_int = manager.mk_var("x", int_sort);
        let x_bool = manager.mk_var("x", bool_sort);
        assert_ne!(x_int, x_bool, "same name at different sorts must not alias");

        assert_eq!(manager.get(x_int).map(|t| t.sort), Some(int_sort));
        assert_eq!(manager.get(x_bool).map(|t| t.sort), Some(bool_sort));

        // ... and each still re-interns to itself, not to the other.
        assert_eq!(manager.mk_var("x", int_sort), x_int);
        assert_eq!(manager.mk_var("x", bool_sort), x_bool);
    }

    #[test]
    fn test_equal_shaped_string_literals_intern_distinctly() {
        // Every literal here has the same length and the same first and last
        // byte -- the shape that a length-plus-endpoints hash would fold
        // together. Structural equality, not the hash, must decide identity.
        let mut manager = TermManager::new();
        let mut ids = FxHashSet::default();
        let mut literals = Vec::new();

        for i in 0..512u32 {
            let literal = format!("A{i:06}Z");
            let id = manager.mk_string_lit(&literal);
            assert!(
                ids.insert(id),
                "literal {literal} collided with an earlier one"
            );
            literals.push((literal, id));
        }
        assert_eq!(ids.len(), 512);

        for (literal, id) in &literals {
            assert_eq!(
                manager.mk_string_lit(literal),
                *id,
                "re-interning {literal} must return the same id"
            );
        }
    }

    // ==================== Garbage collection tests ====================
    //
    // `gc` prunes the intern table only: the `terms` vector is never touched,
    // so a swept term stays readable through `get()` and, if its kind is
    // interned again, comes back as a *fresh* id. These tests pin that exact
    // (slightly surprising) contract.

    #[test]
    fn test_gc_with_empty_roots_keeps_only_true_and_false() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let _sum = manager.mk_add([x, y]);

        let before = manager.cache_size();
        let removed = manager.gc(&FxHashSet::default());

        assert_eq!(
            removed,
            before - manager.cache_size(),
            "the return value is exactly the number of entries dropped"
        );
        assert_eq!(
            manager.cache_size(),
            2,
            "true and false are always rooted, everything else went"
        );
    }

    #[test]
    fn test_gc_keeps_roots_and_their_children() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let sum = manager.mk_add([x, y]);

        let mut roots = FxHashSet::default();
        roots.insert(sum);

        let terms_before = manager.term_count();
        let removed = manager.gc(&roots);

        assert_eq!(
            removed, 0,
            "true, false, x, y and (+ x y) are all reachable"
        );
        assert_eq!(manager.cache_size(), 5);

        // Reachable entries survived, so re-interning hits the table.
        assert_eq!(manager.mk_add([x, y]), sum);
        assert_eq!(manager.mk_var("x", int_sort), x);
        assert_eq!(manager.term_count(), terms_before);
    }

    #[test]
    fn test_gc_prunes_the_table_but_never_the_terms() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let terms_before = manager.term_count();

        let removed = manager.gc(&FxHashSet::default());
        assert_eq!(removed, 1, "only x was unreachable");

        // The term vector is untouched: the swept term is still readable.
        assert_eq!(manager.term_count(), terms_before);
        assert!(manager.get(x).is_some());
        assert!(matches!(
            manager.get(x).map(|t| &t.kind),
            Some(TermKind::Var(_))
        ));
        assert_eq!(manager.get(x).map(|t| t.sort), Some(int_sort));

        // A swept kind re-interns as a FRESH id, and is stable from then on.
        let x_again = manager.mk_var("x", int_sort);
        assert_ne!(x, x_again);
        assert_eq!(manager.term_count(), terms_before + 1);
        assert_eq!(manager.mk_var("x", int_sort), x_again);
    }

    #[test]
    fn test_gc_statistics_track_cache_removals() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let _y = manager.mk_var("y", int_sort);

        let mut roots = FxHashSet::default();
        roots.insert(x);

        let removed = manager.gc(&roots);
        assert_eq!(removed, 1, "y is the only unreachable entry");

        // Every statistic counts *cache* removals, including the two named
        // "collected" -- no term is ever actually freed.
        let stats = manager.gc_statistics();
        assert_eq!(stats.gc_count, 1);
        assert_eq!(stats.last_cache_removed, removed);
        assert_eq!(stats.total_cache_removed, removed);
        assert_eq!(stats.last_collected, removed);
        assert_eq!(stats.total_collected, removed);

        // A second run accumulates rather than replaces the totals.
        let removed2 = manager.gc(&FxHashSet::default());
        let stats = manager.gc_statistics();
        assert_eq!(stats.gc_count, 2);
        assert_eq!(stats.last_cache_removed, removed2);
        assert_eq!(stats.total_cache_removed, removed + removed2);

        manager.reset_gc_stats();
        assert_eq!(manager.gc_statistics().gc_count, 0);
        assert_eq!(manager.gc_statistics().total_cache_removed, 0);
    }

    #[test]
    fn test_gc_aggressive_matches_gc_removal_count() {
        fn populate() -> (TermManager, TermId) {
            let mut manager = TermManager::new();
            let int_sort = manager.sorts.int_sort;
            let x = manager.mk_var("x", int_sort);
            let y = manager.mk_var("y", int_sort);
            let z = manager.mk_var("z", int_sort);
            let sum = manager.mk_add([x, y]);
            let _unreachable = manager.mk_add([sum, z]);
            (manager, sum)
        }

        let (mut plain, sum) = populate();
        let (mut aggressive, sum2) = populate();
        assert_eq!(sum, sum2, "both managers were built identically");

        let mut roots = FxHashSet::default();
        roots.insert(sum);

        assert_eq!(plain.gc(&roots), aggressive.gc_aggressive(&roots));
        assert_eq!(plain.cache_size(), aggressive.cache_size());
        assert_eq!(plain.term_count(), aggressive.term_count());

        // Shrinking must not lose any retained entry.
        assert_eq!(aggressive.mk_add([sum, sum]), plain.mk_add([sum, sum]));
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
