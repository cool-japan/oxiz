//! Hash Consing for Efficient Term Sharing
//!
//! Hash consing is a technique to ensure that structurally equivalent terms
//! share the same memory representation. This provides:
//! - Fast equality checks (pointer equality)
//! - Reduced memory usage
//! - Better cache locality
//!
//! This is particularly important for SMT solvers where the same sub-terms
//! appear many times across different constraints.

use oxiz_core::ast::TermId;
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// A hash-consed term with structural sharing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HcTerm {
    /// Unique identifier for this term
    id: TermId,
}

impl HcTerm {
    /// Create a new hash-consed term
    #[must_use]
    pub const fn new(id: TermId) -> Self {
        Self { id }
    }

    /// Get the term ID
    #[must_use]
    pub const fn id(self) -> TermId {
        self.id
    }
}

/// Hash consing table for managing shared terms
#[derive(Debug)]
pub struct HashConsTable<T>
where
    T: Eq + Hash + Clone,
{
    /// Map from term data to term ID
    table: FxHashMap<T, TermId>,
    /// Map from term ID to term data
    reverse: FxHashMap<TermId, T>,
    /// Next available term ID
    next_id: u32,
    /// Statistics
    hits: usize,
    misses: usize,
}

impl<T> Default for HashConsTable<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> HashConsTable<T>
where
    T: Eq + Hash + Clone,
{
    /// Create a new hash consing table
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: FxHashMap::default(),
            reverse: FxHashMap::default(),
            next_id: 1, // Reserve 0 for special purposes
            hits: 0,
            misses: 0,
        }
    }

    /// Intern a term, returning its unique ID
    ///
    /// If the term already exists, returns the existing ID (cache hit).
    /// Otherwise, allocates a new ID (cache miss).
    pub fn intern(&mut self, term: T) -> HcTerm {
        if let Some(&id) = self.table.get(&term) {
            self.hits += 1;
            HcTerm::new(id)
        } else {
            self.misses += 1;
            let id = TermId::new(self.next_id);
            self.next_id += 1;
            self.table.insert(term.clone(), id);
            self.reverse.insert(id, term);
            HcTerm::new(id)
        }
    }

    /// Get the term data for a given term ID
    #[must_use]
    pub fn get(&self, term: HcTerm) -> Option<&T> {
        self.reverse.get(&term.id())
    }

    /// Get the term ID if it exists (without allocating)
    #[must_use]
    pub fn lookup(&self, term: &T) -> Option<HcTerm> {
        self.table.get(term).map(|&id| HcTerm::new(id))
    }

    /// Check if a term is already interned
    #[must_use]
    pub fn contains(&self, term: &T) -> bool {
        self.table.contains_key(term)
    }

    /// Get the number of unique terms
    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Check if the table is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Get cache statistics (hits, misses)
    #[must_use]
    pub fn stats(&self) -> (usize, usize) {
        (self.hits, self.misses)
    }

    /// Get cache hit rate (0.0 to 1.0)
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    /// Clear the table
    pub fn clear(&mut self) {
        self.table.clear();
        self.reverse.clear();
        self.next_id = 1;
        self.hits = 0;
        self.misses = 0;
    }

    /// Garbage collection: remove terms not in the given set
    ///
    /// This is useful for cleaning up terms that are no longer referenced
    pub fn gc(&mut self, keep: &[HcTerm]) {
        let keep_ids: FxHashMap<TermId, ()> = keep.iter().map(|t| (t.id(), ())).collect();

        // Remove terms not in keep set
        self.table.retain(|_, id| keep_ids.contains_key(id));
        self.reverse.retain(|id, _| keep_ids.contains_key(id));
    }

    /// Iterate over all terms
    pub fn iter(&self) -> impl Iterator<Item = (HcTerm, &T)> {
        self.reverse
            .iter()
            .map(|(&id, term)| (HcTerm::new(id), term))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum TestTerm {
        Var(String),
        App(Box<TestTerm>, Box<TestTerm>),
        Const(i32),
    }

    #[test]
    fn test_basic_interning() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Var("x".to_string());
        let t2 = TestTerm::Var("x".to_string());

        let hc1 = table.intern(t1.clone());
        let hc2 = table.intern(t2);

        // Same term should get same ID
        assert_eq!(hc1, hc2);
        assert_eq!(table.len(), 1);

        // Check statistics
        let (hits, misses) = table.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_different_terms() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Var("x".to_string());
        let t2 = TestTerm::Var("y".to_string());

        let hc1 = table.intern(t1);
        let hc2 = table.intern(t2);

        // Different terms should get different IDs
        assert_ne!(hc1, hc2);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn test_lookup() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Const(42);
        let hc1 = table.intern(t1.clone());

        // Lookup should find the term
        assert_eq!(table.lookup(&t1), Some(hc1));

        // Unknown term should return None
        let t2 = TestTerm::Const(99);
        assert_eq!(table.lookup(&t2), None);
    }

    #[test]
    fn test_get() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Const(42);
        let hc1 = table.intern(t1.clone());

        // Get should retrieve the term data
        assert_eq!(table.get(hc1), Some(&t1));
    }

    #[test]
    fn test_hit_rate() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Const(1);
        table.intern(t1.clone()); // miss
        table.intern(t1.clone()); // hit
        table.intern(t1.clone()); // hit
        table.intern(t1); // hit

        assert!((table.hit_rate() - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_clear() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Var("x".to_string());
        table.intern(t1);

        assert_eq!(table.len(), 1);

        table.clear();
        assert_eq!(table.len(), 0);
        assert_eq!(table.hits, 0);
        assert_eq!(table.misses, 0);
    }

    #[test]
    fn test_gc() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Const(1);
        let t2 = TestTerm::Const(2);
        let t3 = TestTerm::Const(3);

        let hc1 = table.intern(t1);
        let _hc2 = table.intern(t2);
        let hc3 = table.intern(t3);

        assert_eq!(table.len(), 3);

        // Keep only t1 and t3
        table.gc(&[hc1, hc3]);

        assert_eq!(table.len(), 2);
        assert!(table.contains(&TestTerm::Const(1)));
        assert!(!table.contains(&TestTerm::Const(2)));
        assert!(table.contains(&TestTerm::Const(3)));
    }

    #[test]
    fn test_complex_terms() {
        let mut table = HashConsTable::new();

        let x = TestTerm::Var("x".to_string());
        let y = TestTerm::Var("y".to_string());
        let app1 = TestTerm::App(Box::new(x.clone()), Box::new(y.clone()));
        let app2 = TestTerm::App(Box::new(x), Box::new(y));

        let hc1 = table.intern(app1);
        let hc2 = table.intern(app2);

        // Structurally equivalent terms should be shared
        assert_eq!(hc1, hc2);
    }

    #[test]
    fn test_reset_stats() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Const(1);
        table.intern(t1.clone());
        table.intern(t1);

        assert_ne!(table.hits, 0);

        table.reset_stats();
        assert_eq!(table.hits, 0);
        assert_eq!(table.misses, 0);
    }

    #[test]
    fn test_iter() {
        let mut table = HashConsTable::new();

        let t1 = TestTerm::Const(1);
        let t2 = TestTerm::Const(2);

        table.intern(t1.clone());
        table.intern(t2.clone());

        let terms: Vec<_> = table.iter().map(|(_, t)| t.clone()).collect();
        assert_eq!(terms.len(), 2);
        assert!(terms.contains(&t1));
        assert!(terms.contains(&t2));
    }
}
