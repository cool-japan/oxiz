//! Array Theory Eager Expansion.
//!
//! This module implements eager expansion strategies for array theory, where small
//! arrays are completely enumerated rather than using read-over-write axioms.
//!
//! ## When to Use Eager Expansion
//!
//! Eager expansion is beneficial when:
//! - Array index domains are small and known (e.g., arrays of size ≤10)
//! - Many array accesses at different indices
//! - Constraints involve most/all array elements
//!
//! ## Expansion Strategy
//!
//! For an array `a` with index domain `{0, 1, 2}`:
//! ```text
//! Instead of: select(a, i) with read-over-write axioms
//! Expand to: a[0] = v0, a[1] = v1, a[2] = v2
//!            select(a, i) = ite(i=0, v0, ite(i=1, v1, v2))
//! ```
//!
//! ## Benefits
//!
//! - **Completeness**: All elements are explicit
//! - **Simpler Theory**: Reduces to EUF + arithmetic
//! - **Better Propagation**: Direct access to all elements
//! - **Fewer Axiom Instantiations**: No need for read-over-write axioms
//!
//! ## Trade-offs
//!
//! - **Space**: O(domain_size) variables instead of O(accesses)
//! - **Time**: Expansion overhead for large domains
//! - **Scalability**: Limited to small arrays
//!
//! ## References
//!
//! - de Moura & Bjørner: "Z3: An Efficient SMT Solver" (array theory section)
//! - Brummayer & Biere: "Boolector: SMT Solver for Bit-Vectors and Arrays"
//! - CVC4's array solver eager mode

use oxiz_core::ast::{TermId, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::FxHashMap;

/// Configuration for eager expansion.
#[derive(Debug, Clone)]
pub struct EagerExpandConfig {
    /// Enable eager expansion.
    pub enabled: bool,
    /// Maximum index domain size for expansion.
    pub max_domain_size: usize,
    /// Maximum number of arrays to expand.
    pub max_arrays: usize,
    /// Expand arrays with many accesses (access count threshold).
    pub access_threshold: usize,
}

impl Default for EagerExpandConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_domain_size: 10,
            max_arrays: 100,
            access_threshold: 3,
        }
    }
}

/// Statistics for eager expansion.
#[derive(Debug, Clone, Default)]
pub struct EagerExpandStats {
    /// Number of arrays expanded.
    pub arrays_expanded: u64,
    /// Total elements created via expansion.
    pub elements_created: u64,
    /// Number of select operations simplified.
    pub selects_simplified: u64,
    /// Number of store operations simplified.
    pub stores_simplified: u64,
    /// Time in expansion (microseconds).
    pub expansion_time_us: u64,
}

/// Expanded array representation.
#[derive(Debug, Clone)]
pub struct ExpandedArray {
    /// Original array term.
    pub array_term: TermId,
    /// Index domain (concrete values).
    pub domain: Vec<i64>,
    /// Element variables: domain_value -> element_term.
    pub elements: FxHashMap<i64, TermId>,
    /// Array element sort.
    pub element_sort: SortId,
}

/// Eager array expansion engine.
pub struct EagerArrayExpander {
    /// Configuration.
    config: EagerExpandConfig,
    /// Statistics.
    stats: EagerExpandStats,
    /// Expanded arrays.
    expanded: FxHashMap<TermId, ExpandedArray>,
    /// Access counts per array (for heuristics).
    access_counts: FxHashMap<TermId, usize>,
}

impl EagerArrayExpander {
    /// Create a new eager expander.
    pub fn new() -> Self {
        Self::with_config(EagerExpandConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: EagerExpandConfig) -> Self {
        Self {
            config,
            stats: EagerExpandStats::default(),
            expanded: FxHashMap::default(),
            access_counts: FxHashMap::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &EagerExpandStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = EagerExpandStats::default();
    }

    /// Record an access to an array.
    pub fn record_access(&mut self, array: TermId) {
        *self.access_counts.entry(array).or_insert(0) += 1;
    }

    /// Check if an array should be eagerly expanded.
    pub fn should_expand(&self, array: TermId, domain_size: usize) -> bool {
        if !self.config.enabled {
            return false;
        }

        if domain_size > self.config.max_domain_size {
            return false;
        }

        if self.expanded.len() >= self.config.max_arrays {
            return false;
        }

        if self.expanded.contains_key(&array) {
            return false; // Already expanded
        }

        // Check access count heuristic
        let access_count = self.access_counts.get(&array).copied().unwrap_or(0);
        access_count >= self.config.access_threshold
    }

    /// Eagerly expand an array.
    ///
    /// Creates element variables for each index in the domain.
    pub fn expand_array(
        &mut self,
        array: TermId,
        domain: Vec<i64>,
        element_sort: SortId,
        tm: &mut TermManager,
    ) -> Result<(), String> {
        if self.expanded.contains_key(&array) {
            return Ok(()); // Already expanded
        }

        let start = std::time::Instant::now();

        // Create element variables
        let mut elements = FxHashMap::default();

        for &index_val in &domain {
            // Create a fresh variable for this element
            let element_name = format!("array_{}_{}", array.raw(), index_val);
            let element_var = tm.mk_var(&element_name, element_sort);
            elements.insert(index_val, element_var);
        }

        self.stats.elements_created += elements.len() as u64;

        let expanded_array = ExpandedArray {
            array_term: array,
            domain: domain.clone(),
            elements,
            element_sort,
        };

        self.expanded.insert(array, expanded_array);
        self.stats.arrays_expanded += 1;
        self.stats.expansion_time_us += start.elapsed().as_micros() as u64;

        Ok(())
    }

    /// Get element term for an array at a specific index.
    pub fn get_element(&self, array: TermId, index: i64) -> Option<TermId> {
        self.expanded
            .get(&array)
            .and_then(|exp| exp.elements.get(&index).copied())
    }

    /// Check if an array is expanded.
    pub fn is_expanded(&self, array: TermId) -> bool {
        self.expanded.contains_key(&array)
    }

    /// Get the expanded representation of an array.
    pub fn get_expanded(&self, array: TermId) -> Option<&ExpandedArray> {
        self.expanded.get(&array)
    }

    /// Clear all expansions.
    pub fn clear(&mut self) {
        self.expanded.clear();
        self.access_counts.clear();
    }
}

impl Default for EagerArrayExpander {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eager_expand_config_default() {
        let config = EagerExpandConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_domain_size, 10);
    }

    #[test]
    fn test_eager_expander_creation() {
        let expander = EagerArrayExpander::new();
        assert_eq!(expander.stats().arrays_expanded, 0);
    }

    #[test]
    fn test_record_access() {
        let mut expander = EagerArrayExpander::new();

        let array = TermId::new(1);

        expander.record_access(array);
        expander.record_access(array);
        expander.record_access(array);

        assert_eq!(expander.access_counts.get(&array), Some(&3));
    }

    #[test]
    fn test_should_expand_small_domain() {
        let mut expander = EagerArrayExpander::new();

        let array = TermId::new(1);

        // Need enough accesses
        for _ in 0..3 {
            expander.record_access(array);
        }

        assert!(expander.should_expand(array, 5));
        assert!(!expander.should_expand(array, 100)); // Too large
    }

    #[test]
    fn test_should_expand_disabled() {
        let config = EagerExpandConfig {
            enabled: false,
            ..Default::default()
        };
        let expander = EagerArrayExpander::with_config(config);

        assert!(!expander.should_expand(TermId::new(1), 5));
    }

    #[test]
    fn test_is_expanded() {
        let mut expander = EagerArrayExpander::new();
        let mut tm = TermManager::new();

        let array = TermId::new(1);
        let sort = tm.sorts.int_sort;

        assert!(!expander.is_expanded(array));

        let _result = expander.expand_array(array, vec![0, 1, 2], sort, &mut tm);

        assert!(expander.is_expanded(array));
        assert_eq!(expander.stats().arrays_expanded, 1);
        assert_eq!(expander.stats().elements_created, 3);
    }

    #[test]
    fn test_get_element() {
        let mut expander = EagerArrayExpander::new();
        let mut tm = TermManager::new();

        let array = TermId::new(1);
        let sort = tm.sorts.int_sort;

        let _result = expander.expand_array(array, vec![0, 1, 2], sort, &mut tm);

        // Elements should exist
        assert!(expander.get_element(array, 0).is_some());
        assert!(expander.get_element(array, 1).is_some());
        assert!(expander.get_element(array, 2).is_some());
        assert!(expander.get_element(array, 3).is_none()); // Not in domain
    }

    #[test]
    fn test_stats_reset() {
        let mut expander = EagerArrayExpander::new();

        expander.stats.arrays_expanded = 10;
        expander.stats.elements_created = 50;

        expander.reset_stats();

        assert_eq!(expander.stats().arrays_expanded, 0);
        assert_eq!(expander.stats().elements_created, 0);
    }

    #[test]
    fn test_clear() {
        let mut expander = EagerArrayExpander::new();
        let mut tm = TermManager::new();

        let array = TermId::new(1);
        let sort = tm.sorts.int_sort;

        expander
            .expand_array(array, vec![0, 1, 2], sort, &mut tm)
            .unwrap();
        expander.record_access(array);

        assert!(!expander.expanded.is_empty());
        assert!(!expander.access_counts.is_empty());

        expander.clear();

        assert!(expander.expanded.is_empty());
        assert!(expander.access_counts.is_empty());
    }
}
