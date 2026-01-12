//! Lemma caching for theory propagation
//!
//! This module implements a cache for theory lemmas to avoid redundant
//! computation of the same theory propagations. This is particularly useful
//! for expensive theory reasoning like arithmetic and arrays.
//!
//! Reference: Z3's theory_cache and lemma management

use crate::ast::TermId;
use rustc_hash::{FxHashMap, FxHashSet};

/// A theory lemma representing a clause or implication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Lemma {
    /// The hypothesis terms (conjunction)
    pub hypotheses: Vec<TermId>,
    /// The conclusion term
    pub conclusion: TermId,
    /// The theory that generated this lemma
    pub theory: TheoryId,
    /// Priority/cost for lemma management
    pub priority: u32,
}

impl Lemma {
    /// Create a new lemma
    #[must_use]
    pub fn new(hypotheses: Vec<TermId>, conclusion: TermId, theory: TheoryId) -> Self {
        Self {
            hypotheses,
            conclusion,
            theory,
            priority: 0,
        }
    }

    /// Create a lemma with priority
    #[must_use]
    pub fn with_priority(
        hypotheses: Vec<TermId>,
        conclusion: TermId,
        theory: TheoryId,
        priority: u32,
    ) -> Self {
        Self {
            hypotheses,
            conclusion,
            theory,
            priority,
        }
    }

    /// Check if this lemma is unit (no hypotheses)
    #[must_use]
    pub fn is_unit(&self) -> bool {
        self.hypotheses.is_empty()
    }

    /// Check if this lemma is binary (one hypothesis)
    #[must_use]
    pub fn is_binary(&self) -> bool {
        self.hypotheses.len() == 1
    }

    /// Get the size (number of literals)
    #[must_use]
    pub fn size(&self) -> usize {
        self.hypotheses.len() + 1
    }
}

/// Theory identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TheoryId {
    /// Core theory (boolean logic)
    Core,
    /// Arithmetic (LIA/LRA/NIA/NRA)
    Arithmetic,
    /// Bitvectors
    BitVector,
    /// Arrays
    Array,
    /// Strings
    String,
    /// Algebraic datatypes
    DataType,
    /// Floating-point
    FloatingPoint,
    /// Uninterpreted functions
    UF,
}

/// Lemma cache for storing and retrieving theory lemmas
#[derive(Debug, Clone)]
pub struct LemmaCache {
    /// Lemmas indexed by their normalized key
    lemmas: FxHashMap<LemmaKey, Vec<Lemma>>,
    /// Lemmas indexed by conclusion term
    by_conclusion: FxHashMap<TermId, Vec<usize>>,
    /// Lemmas indexed by hypothesis terms
    by_hypothesis: FxHashMap<TermId, Vec<usize>>,
    /// All lemmas in insertion order
    all_lemmas: Vec<Lemma>,
    /// Statistics
    hits: u64,
    misses: u64,
    /// Maximum cache size
    max_size: usize,
}

/// Normalized key for lemma lookup
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LemmaKey {
    /// Sorted hypothesis IDs
    hypotheses: Vec<TermId>,
    /// Conclusion ID
    conclusion: TermId,
}

impl LemmaKey {
    fn new(mut hypotheses: Vec<TermId>, conclusion: TermId) -> Self {
        hypotheses.sort_unstable();
        Self {
            hypotheses,
            conclusion,
        }
    }
}

impl LemmaCache {
    /// Create a new lemma cache with default capacity
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(10000)
    }

    /// Create a new lemma cache with specified capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            lemmas: FxHashMap::default(),
            by_conclusion: FxHashMap::default(),
            by_hypothesis: FxHashMap::default(),
            all_lemmas: Vec::new(),
            hits: 0,
            misses: 0,
            max_size: capacity,
        }
    }

    /// Insert a lemma into the cache
    ///
    /// Returns true if the lemma was new (cache miss), false if it already existed (cache hit).
    pub fn insert(&mut self, lemma: Lemma) -> bool {
        let key = LemmaKey::new(lemma.hypotheses.clone(), lemma.conclusion);

        // Check if already cached
        if let Some(existing) = self.lemmas.get(&key)
            && existing.iter().any(|l| l == &lemma)
        {
            self.hits += 1;
            return false; // Already cached
        }

        self.misses += 1;

        // Add to main cache
        let idx = self.all_lemmas.len();
        self.lemmas.entry(key).or_default().push(lemma.clone());

        // Index by conclusion
        self.by_conclusion
            .entry(lemma.conclusion)
            .or_default()
            .push(idx);

        // Index by hypotheses
        for &hyp in &lemma.hypotheses {
            self.by_hypothesis.entry(hyp).or_default().push(idx);
        }

        self.all_lemmas.push(lemma);

        // Evict old lemmas if cache is full
        if self.all_lemmas.len() > self.max_size {
            self.evict_lru();
        }

        true
    }

    /// Look up lemmas with the given hypotheses and conclusion
    #[must_use]
    pub fn lookup(&mut self, hypotheses: &[TermId], conclusion: TermId) -> Option<&[Lemma]> {
        let key = LemmaKey::new(hypotheses.to_vec(), conclusion);

        if let Some(lemmas) = self.lemmas.get(&key) {
            self.hits += 1;
            Some(lemmas)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Get all lemmas with the given conclusion
    #[must_use]
    pub fn by_conclusion(&self, conclusion: TermId) -> Vec<&Lemma> {
        if let Some(indices) = self.by_conclusion.get(&conclusion) {
            indices.iter().map(|&i| &self.all_lemmas[i]).collect()
        } else {
            Vec::new()
        }
    }

    /// Get all lemmas containing the given hypothesis
    #[must_use]
    pub fn by_hypothesis(&self, hypothesis: TermId) -> Vec<&Lemma> {
        if let Some(indices) = self.by_hypothesis.get(&hypothesis) {
            indices.iter().map(|&i| &self.all_lemmas[i]).collect()
        } else {
            Vec::new()
        }
    }

    /// Get all lemmas from a specific theory
    #[must_use]
    pub fn by_theory(&self, theory: TheoryId) -> Vec<&Lemma> {
        self.all_lemmas
            .iter()
            .filter(|l| l.theory == theory)
            .collect()
    }

    /// Get unit lemmas (no hypotheses)
    #[must_use]
    pub fn unit_lemmas(&self) -> Vec<&Lemma> {
        self.all_lemmas.iter().filter(|l| l.is_unit()).collect()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.lemmas.clear();
        self.by_conclusion.clear();
        self.by_hypothesis.clear();
        self.all_lemmas.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get the number of cached lemmas
    #[must_use]
    pub fn len(&self) -> usize {
        self.all_lemmas.len()
    }

    /// Check if the cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.all_lemmas.is_empty()
    }

    /// Get cache statistics
    #[must_use]
    pub fn statistics(&self) -> CacheStatistics {
        CacheStatistics {
            size: self.all_lemmas.len(),
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
            max_size: self.max_size,
        }
    }

    /// Get cache hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Evict least recently used lemmas
    fn evict_lru(&mut self) {
        // Simple LRU: remove first 10% of lemmas
        let to_remove = self.max_size / 10;
        if to_remove == 0 {
            return;
        }

        // Rebuild cache without first to_remove lemmas
        let keep_from = to_remove;
        let kept_lemmas: Vec<Lemma> = self.all_lemmas.drain(keep_from..).collect();

        self.lemmas.clear();
        self.by_conclusion.clear();
        self.by_hypothesis.clear();
        self.all_lemmas.clear();

        // Re-insert kept lemmas
        for lemma in kept_lemmas {
            let key = LemmaKey::new(lemma.hypotheses.clone(), lemma.conclusion);
            let idx = self.all_lemmas.len();

            self.lemmas.entry(key).or_default().push(lemma.clone());

            self.by_conclusion
                .entry(lemma.conclusion)
                .or_default()
                .push(idx);

            for &hyp in &lemma.hypotheses {
                self.by_hypothesis.entry(hyp).or_default().push(idx);
            }

            self.all_lemmas.push(lemma);
        }
    }

    /// Minimize the cache by removing redundant lemmas
    pub fn minimize(&mut self) {
        // Remove subsumed lemmas (lemmas implied by shorter lemmas)
        let mut to_remove = FxHashSet::default();

        for i in 0..self.all_lemmas.len() {
            for j in 0..self.all_lemmas.len() {
                if i == j || to_remove.contains(&i) {
                    continue;
                }

                let lemma_i = &self.all_lemmas[i];
                let lemma_j = &self.all_lemmas[j];

                // If lemma_j subsumes lemma_i, mark i for removal
                if subsumes(lemma_j, lemma_i) {
                    to_remove.insert(i);
                    break;
                }
            }
        }

        // Rebuild without removed lemmas
        let kept_lemmas: Vec<Lemma> = self
            .all_lemmas
            .iter()
            .enumerate()
            .filter(|(i, _)| !to_remove.contains(i))
            .map(|(_, l)| l.clone())
            .collect();

        self.lemmas.clear();
        self.by_conclusion.clear();
        self.by_hypothesis.clear();
        self.all_lemmas.clear();

        for lemma in kept_lemmas {
            self.insert(lemma);
        }
    }
}

impl Default for LemmaCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if lemma a subsumes lemma b (a implies b)
fn subsumes(a: &Lemma, b: &Lemma) -> bool {
    // a subsumes b if:
    // 1. They have the same conclusion
    // 2. a's hypotheses are a subset of b's hypotheses
    if a.conclusion != b.conclusion {
        return false;
    }

    let a_hyp: FxHashSet<_> = a.hypotheses.iter().copied().collect();
    let b_hyp: FxHashSet<_> = b.hypotheses.iter().copied().collect();

    a_hyp.is_subset(&b_hyp)
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    /// Number of lemmas in cache
    pub size: usize,
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f64,
    /// Maximum cache size
    pub max_size: usize,
}

impl std::fmt::Display for CacheStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Lemma Cache Statistics:")?;
        writeln!(f, "  Size:     {}/{}", self.size, self.max_size)?;
        writeln!(f, "  Hits:     {}", self.hits)?;
        writeln!(f, "  Misses:   {}", self.misses)?;
        writeln!(f, "  Hit rate: {:.2}%", self.hit_rate * 100.0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(id: u32) -> TermId {
        TermId(id)
    }

    #[test]
    fn test_empty_cache() {
        let cache = LemmaCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut cache = LemmaCache::new();

        let lemma = Lemma::new(vec![term(1), term(2)], term(3), TheoryId::Arithmetic);

        assert!(cache.insert(lemma.clone()));
        assert_eq!(cache.len(), 1);

        let found = cache.lookup(&[term(1), term(2)], term(3));
        assert!(found.is_some());
        assert_eq!(found.unwrap()[0], lemma);
    }

    #[test]
    fn test_duplicate_insert() {
        let mut cache = LemmaCache::new();

        let lemma = Lemma::new(vec![term(1)], term(2), TheoryId::Core);

        assert!(cache.insert(lemma.clone())); // First insert - miss
        assert!(!cache.insert(lemma)); // Second insert - hit

        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_by_conclusion() {
        let mut cache = LemmaCache::new();

        let lemma1 = Lemma::new(vec![term(1)], term(3), TheoryId::Arithmetic);
        let lemma2 = Lemma::new(vec![term(2)], term(3), TheoryId::Arithmetic);

        cache.insert(lemma1);
        cache.insert(lemma2);

        let found = cache.by_conclusion(term(3));
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_by_hypothesis() {
        let mut cache = LemmaCache::new();

        let lemma1 = Lemma::new(vec![term(1), term(2)], term(3), TheoryId::Core);
        let lemma2 = Lemma::new(vec![term(1), term(4)], term(5), TheoryId::Core);

        cache.insert(lemma1);
        cache.insert(lemma2);

        let found = cache.by_hypothesis(term(1));
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_by_theory() {
        let mut cache = LemmaCache::new();

        cache.insert(Lemma::new(vec![term(1)], term(2), TheoryId::Arithmetic));
        cache.insert(Lemma::new(vec![term(3)], term(4), TheoryId::Arithmetic));
        cache.insert(Lemma::new(vec![term(5)], term(6), TheoryId::BitVector));

        let found = cache.by_theory(TheoryId::Arithmetic);
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_unit_lemmas() {
        let mut cache = LemmaCache::new();

        cache.insert(Lemma::new(vec![], term(1), TheoryId::Core));
        cache.insert(Lemma::new(vec![term(2)], term(3), TheoryId::Core));
        cache.insert(Lemma::new(vec![], term(4), TheoryId::Core));

        let units = cache.unit_lemmas();
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut cache = LemmaCache::new();

        cache.insert(Lemma::new(vec![term(1)], term(2), TheoryId::Core));
        assert!(!cache.is_empty());

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_statistics() {
        let mut cache = LemmaCache::new();

        let lemma = Lemma::new(vec![term(1)], term(2), TheoryId::Arithmetic);

        cache.insert(lemma.clone()); // miss
        cache.insert(lemma); // hit

        let stats = cache.statistics();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[test]
    fn test_eviction() {
        let mut cache = LemmaCache::with_capacity(10);

        // Insert more than capacity
        for i in 0..15 {
            cache.insert(Lemma::new(vec![term(i)], term(i + 100), TheoryId::Core));
        }

        assert!(cache.len() <= 10);
    }

    #[test]
    fn test_subsumes() {
        let a = Lemma::new(vec![term(1)], term(3), TheoryId::Core);
        let b = Lemma::new(vec![term(1), term(2)], term(3), TheoryId::Core);

        assert!(subsumes(&a, &b)); // a has fewer hypotheses
        assert!(!subsumes(&b, &a)); // b has more hypotheses
    }

    #[test]
    fn test_minimize() {
        let mut cache = LemmaCache::new();

        // Lemma a subsumes lemma b
        let a = Lemma::new(vec![term(1)], term(3), TheoryId::Core);
        let b = Lemma::new(vec![term(1), term(2)], term(3), TheoryId::Core);

        cache.insert(a);
        cache.insert(b);

        assert_eq!(cache.len(), 2);

        cache.minimize();

        assert_eq!(cache.len(), 1); // b should be removed
    }

    #[test]
    fn test_lemma_properties() {
        let unit = Lemma::new(vec![], term(1), TheoryId::Core);
        assert!(unit.is_unit());
        assert!(!unit.is_binary());
        assert_eq!(unit.size(), 1);

        let binary = Lemma::new(vec![term(1)], term(2), TheoryId::Core);
        assert!(!binary.is_unit());
        assert!(binary.is_binary());
        assert_eq!(binary.size(), 2);
    }
}
