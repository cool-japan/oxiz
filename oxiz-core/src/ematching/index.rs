//! Term indexing for efficient E-matching
//!
//! This module provides indexing structures for fast lookup of terms during
//! E-matching. Different index types are optimized for different access patterns.

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::{OxizError, Result};
use crate::sort::SortId;
use lasso::Spur;
use rustc_hash::FxHashMap;

/// Configuration for term indexing
#[derive(Debug, Clone)]
pub struct IndexConfig {
    /// Whether to enable fingerprint-based indexing
    pub use_fingerprints: bool,
    /// Whether to index by sort
    pub index_by_sort: bool,
    /// Whether to index by function symbol
    pub index_by_function: bool,
    /// Maximum terms per bucket
    pub max_bucket_size: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            use_fingerprints: true,
            index_by_sort: true,
            index_by_function: true,
            max_bucket_size: 1000,
        }
    }
}

/// Statistics about term indexing
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Total terms indexed
    pub total_terms: usize,
    /// Number of lookups performed
    pub lookups: usize,
    /// Number of successful lookups
    pub hits: usize,
    /// Number of failed lookups
    pub misses: usize,
    /// Average bucket size
    pub avg_bucket_size: f64,
    /// Maximum bucket size
    pub max_bucket_size: usize,
}

/// A term index entry
#[derive(Debug, Clone)]
pub struct TermIndexEntry {
    /// The indexed term
    pub term: TermId,
    /// Additional metadata
    pub metadata: EntryMetadata,
}

/// Metadata associated with an index entry
#[derive(Debug, Clone, Default)]
pub struct EntryMetadata {
    /// Generation when term was added
    pub generation: u64,
    /// Access count
    pub access_count: u32,
    /// Last access generation
    pub last_access: u64,
}

/// Main term index
#[derive(Debug)]
pub struct TermIndex {
    /// Configuration
    config: IndexConfig,
    /// Index by sort
    by_sort: FxHashMap<SortId, Vec<TermIndexEntry>>,
    /// Index by function symbol
    by_function: FxHashMap<Spur, Vec<TermIndexEntry>>,
    /// All indexed terms
    all_terms: Vec<TermIndexEntry>,
    /// Current generation counter
    generation: u64,
    /// Statistics
    stats: IndexStats,
}

impl TermIndex {
    /// Create a new term index
    pub fn new(config: IndexConfig) -> Self {
        Self {
            config,
            by_sort: FxHashMap::default(),
            by_function: FxHashMap::default(),
            all_terms: Vec::new(),
            generation: 0,
            stats: IndexStats::default(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(IndexConfig::default())
    }

    /// Add a term to the index
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) -> Result<()> {
        let Some(t) = manager.get(term) else {
            return Err(OxizError::EmatchError(format!("Term {:?} not found", term)));
        };

        let entry = TermIndexEntry {
            term,
            metadata: EntryMetadata {
                generation: self.generation,
                access_count: 0,
                last_access: 0,
            },
        };

        // Add to all_terms
        self.all_terms.push(entry.clone());

        // Index by sort
        if self.config.index_by_sort {
            self.by_sort.entry(t.sort).or_default().push(entry.clone());
        }

        // Index by function symbol
        if self.config.index_by_function
            && let TermKind::Apply { func, .. } = &t.kind
        {
            self.by_function
                .entry(*func)
                .or_default()
                .push(entry.clone());
        }

        self.stats.total_terms += 1;
        Ok(())
    }

    /// Lookup terms by sort
    pub fn lookup_by_sort(&mut self, sort: SortId) -> &[TermIndexEntry] {
        self.stats.lookups += 1;
        if let Some(entries) = self.by_sort.get(&sort) {
            self.stats.hits += 1;
            entries
        } else {
            self.stats.misses += 1;
            &[]
        }
    }

    /// Lookup terms by function symbol
    pub fn lookup_by_function(&mut self, func: Spur) -> &[TermIndexEntry] {
        self.stats.lookups += 1;
        if let Some(entries) = self.by_function.get(&func) {
            self.stats.hits += 1;
            entries
        } else {
            self.stats.misses += 1;
            &[]
        }
    }

    /// Get all indexed terms
    pub fn all_terms(&self) -> &[TermIndexEntry] {
        &self.all_terms
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.by_sort.clear();
        self.by_function.clear();
        self.all_terms.clear();
        self.generation = 0;
        self.stats = IndexStats::default();
    }

    /// Advance generation
    pub fn advance_generation(&mut self) {
        self.generation += 1;
    }

    /// Get statistics
    pub fn stats(&self) -> &IndexStats {
        &self.stats
    }

    /// Compute index statistics
    pub fn compute_stats(&mut self) {
        let mut total_size = 0;
        let mut max_size = 0;
        let mut bucket_count = 0;

        for bucket in self.by_sort.values() {
            total_size += bucket.len();
            max_size = max_size.max(bucket.len());
            bucket_count += 1;
        }

        for bucket in self.by_function.values() {
            total_size += bucket.len();
            max_size = max_size.max(bucket.len());
            bucket_count += 1;
        }

        self.stats.max_bucket_size = max_size;
        self.stats.avg_bucket_size = if bucket_count > 0 {
            total_size as f64 / bucket_count as f64
        } else {
            0.0
        };
    }
}

/// Inverted index for pattern matching
#[derive(Debug)]
pub struct InvertedIndex {
    /// Map from term patterns to matching terms
    index: FxHashMap<TermPattern, Vec<TermId>>,
    /// Configuration
    #[allow(dead_code)]
    config: IndexConfig,
}

/// A term pattern for indexing
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TermPattern {
    /// Function symbol (if any)
    pub func: Option<Spur>,
    /// Arity
    pub arity: Option<usize>,
    /// Sort
    pub sort: Option<SortId>,
}

impl InvertedIndex {
    /// Create a new inverted index
    pub fn new(config: IndexConfig) -> Self {
        Self {
            index: FxHashMap::default(),
            config,
        }
    }

    /// Add a term to the inverted index
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) -> Result<()> {
        let Some(t) = manager.get(term) else {
            return Err(OxizError::EmatchError(format!("Term {:?} not found", term)));
        };

        let pattern = match &t.kind {
            TermKind::Apply { func, args } => TermPattern {
                func: Some(*func),
                arity: Some(args.len()),
                sort: Some(t.sort),
            },
            _ => TermPattern {
                func: None,
                arity: None,
                sort: Some(t.sort),
            },
        };

        self.index.entry(pattern).or_default().push(term);
        Ok(())
    }

    /// Lookup terms matching a pattern
    pub fn lookup(&self, pattern: &TermPattern) -> &[TermId] {
        self.index.get(pattern).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.index.clear();
    }
}

/// E-graph index for congruence closure
#[derive(Debug)]
pub struct EgraphIndex {
    /// Equivalence classes
    eq_classes: FxHashMap<TermId, TermId>,
    /// Representative to class members
    class_members: FxHashMap<TermId, Vec<TermId>>,
    /// Parent terms
    parents: FxHashMap<TermId, Vec<TermId>>,
}

impl EgraphIndex {
    /// Create a new E-graph index
    pub fn new() -> Self {
        Self {
            eq_classes: FxHashMap::default(),
            class_members: FxHashMap::default(),
            parents: FxHashMap::default(),
        }
    }

    /// Find representative of a term
    pub fn find(&mut self, term: TermId) -> TermId {
        if let Some(&repr) = self.eq_classes.get(&term)
            && repr != term
        {
            let root = self.find(repr);
            self.eq_classes.insert(term, root);
            return root;
        }
        term
    }

    /// Merge two equivalence classes
    pub fn merge(&mut self, t1: TermId, t2: TermId) {
        let r1 = self.find(t1);
        let r2 = self.find(t2);

        if r1 != r2 {
            self.eq_classes.insert(r2, r1);

            // Merge class members
            let members2 = self.class_members.remove(&r2).unwrap_or_default();
            self.class_members.entry(r1).or_default().extend(members2);
        }
    }

    /// Get all members of an equivalence class
    pub fn get_class(&self, term: TermId) -> &[TermId] {
        self.class_members
            .get(&term)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Add parent relationship
    pub fn add_parent(&mut self, child: TermId, parent: TermId) {
        self.parents.entry(child).or_default().push(parent);
    }

    /// Get parents of a term
    pub fn get_parents(&self, term: TermId) -> &[TermId] {
        self.parents.get(&term).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.eq_classes.clear();
        self.class_members.clear();
        self.parents.clear();
    }
}

impl Default for EgraphIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    fn setup() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert!(config.use_fingerprints);
        assert!(config.index_by_sort);
        assert!(config.index_by_function);
    }

    #[test]
    fn test_term_index_creation() {
        let index = TermIndex::new_default();
        assert_eq!(index.stats.total_terms, 0);
    }

    #[test]
    fn test_add_term() {
        let mut manager = setup();
        let mut index = TermIndex::new_default();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        index.add_term(x, &manager).unwrap();

        assert_eq!(index.stats.total_terms, 1);
    }

    #[test]
    fn test_lookup_by_sort() {
        let mut manager = setup();
        let mut index = TermIndex::new_default();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        index.add_term(x, &manager).unwrap();

        let results = index.lookup_by_sort(int_sort);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_inverted_index() {
        let mut manager = setup();
        let mut inv_index = InvertedIndex::new(IndexConfig::default());

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);

        inv_index.add_term(f_x, &manager).unwrap();

        let f_name = manager.intern_str("f");
        let pattern = TermPattern {
            func: Some(f_name),
            arity: Some(1),
            sort: Some(int_sort),
        };

        let results = inv_index.lookup(&pattern);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_egraph_index_find() {
        let mut egraph = EgraphIndex::new();

        let t1 = TermId(1);
        let t2 = TermId(2);

        // Initially, each term is its own representative
        assert_eq!(egraph.find(t1), t1);
        assert_eq!(egraph.find(t2), t2);
    }

    #[test]
    fn test_egraph_index_merge() {
        let mut egraph = EgraphIndex::new();

        let t1 = TermId(1);
        let t2 = TermId(2);

        egraph.merge(t1, t2);

        // After merge, both should have same representative
        let r1 = egraph.find(t1);
        let r2 = egraph.find(t2);
        assert_eq!(r1, r2);
    }
}
