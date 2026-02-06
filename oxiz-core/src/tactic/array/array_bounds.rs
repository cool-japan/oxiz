//! Array Bounds Propagation Tactic.
//!
//! Propagates bounds information for array indices and values to simplify
//! array constraints before solving.
//!
//! ## Propagation Rules
//!
//! - **Index bounds**: 0 <= i < length(arr)
//! - **Value bounds**: min <= arr\[i\] <= max
//! - **Equality propagation**: arr1 = arr2 => arr1\[i\] = arr2\[i\]
//! - **Store propagation**: store(arr, i, v)\[j\] = if i = j then v else arr\[j\]
//!
//! ## References
//!
//! - de Moura & Bjørner: "Z3: An Efficient SMT Solver" (2008)
//! - Z3's `tactic/smtlogics_tactics.cpp` (array tactics)

use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashMap;

/// Array identifier.
pub type ArrayId = u32;

/// Index bound information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexBound {
    /// Lower bound (inclusive).
    pub lower: Option<i64>,
    /// Upper bound (exclusive).
    pub upper: Option<i64>,
}

impl IndexBound {
    /// Create unbounded.
    pub fn unbounded() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }

    /// Create with specific bounds.
    pub fn new(lower: Option<i64>, upper: Option<i64>) -> Self {
        Self { lower, upper }
    }

    /// Check if this bound is more restrictive than another.
    pub fn is_tighter_than(&self, other: &IndexBound) -> bool {
        let lower_tighter = match (self.lower, other.lower) {
            (Some(a), Some(b)) => a > b,
            (Some(_), None) => true,
            _ => false,
        };

        let upper_tighter = match (self.upper, other.upper) {
            (Some(a), Some(b)) => a < b,
            (Some(_), None) => true,
            _ => false,
        };

        lower_tighter || upper_tighter
    }

    /// Intersect two bounds.
    pub fn intersect(&self, other: &IndexBound) -> IndexBound {
        let lower = match (self.lower, other.lower) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        let upper = match (self.upper, other.upper) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        IndexBound { lower, upper }
    }

    /// Check if bounds are contradictory.
    pub fn is_contradictory(&self) -> bool {
        if let (Some(l), Some(u)) = (self.lower, self.upper) {
            l >= u
        } else {
            false
        }
    }
}

/// Configuration for array bounds tactic.
#[derive(Debug, Clone)]
pub struct ArrayBoundsConfig {
    /// Enable index bounds propagation.
    pub propagate_index_bounds: bool,
    /// Enable value bounds propagation.
    pub propagate_value_bounds: bool,
    /// Enable store/select simplification.
    pub simplify_store_select: bool,
    /// Maximum propagation iterations.
    pub max_iterations: usize,
}

impl Default for ArrayBoundsConfig {
    fn default() -> Self {
        Self {
            propagate_index_bounds: true,
            propagate_value_bounds: true,
            simplify_store_select: true,
            max_iterations: 10,
        }
    }
}

/// Statistics for array bounds tactic.
#[derive(Debug, Clone, Default)]
pub struct ArrayBoundsStats {
    /// Index bounds propagated.
    pub index_bounds_propagated: u64,
    /// Value bounds propagated.
    pub value_bounds_propagated: u64,
    /// Store/select simplifications.
    pub store_select_simplified: u64,
    /// Contradictions detected.
    pub contradictions_detected: u64,
    /// Total iterations.
    pub total_iterations: u64,
}

/// Array bounds propagation tactic.
pub struct ArrayBoundsTactic {
    /// Term manager.
    manager: TermManager,
    /// Configuration.
    config: ArrayBoundsConfig,
    /// Statistics.
    stats: ArrayBoundsStats,
    /// Index bounds for each array variable.
    index_bounds: FxHashMap<TermId, IndexBound>,
    /// Value bounds for each array variable.
    value_bounds: FxHashMap<TermId, (Option<i64>, Option<i64>)>,
    /// Rewrite cache.
    rewrite_cache: FxHashMap<TermId, TermId>,
}

impl ArrayBoundsTactic {
    /// Create a new array bounds tactic.
    pub fn new(manager: TermManager, config: ArrayBoundsConfig) -> Self {
        Self {
            manager,
            config,
            stats: ArrayBoundsStats::default(),
            index_bounds: FxHashMap::default(),
            value_bounds: FxHashMap::default(),
            rewrite_cache: FxHashMap::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config(manager: TermManager) -> Self {
        Self::new(manager, ArrayBoundsConfig::default())
    }

    /// Apply tactic to a term.
    pub fn apply(&mut self, term: TermId) -> TermId {
        // Check cache
        if let Some(&cached) = self.rewrite_cache.get(&term) {
            return cached;
        }

        // Run fixed-point iteration
        let mut current = term;
        let mut iteration = 0;

        while iteration < self.config.max_iterations {
            let next = self.apply_once(current);

            if next == current {
                break; // Fixed point reached
            }

            current = next;
            iteration += 1;
        }

        self.stats.total_iterations += iteration as u64;
        self.rewrite_cache.insert(term, current);
        current
    }

    /// Apply tactic once (single pass).
    fn apply_once(&mut self, term: TermId) -> TermId {
        let term_data = match self.manager.get(term) {
            Some(t) => t.clone(),
            None => return term,
        };

        match &term_data.kind {
            TermKind::Select(array, index) => self.simplify_select(*array, *index, term),
            TermKind::Store(array, index, value) => {
                self.simplify_store(*array, *index, *value, term)
            }
            _ => term,
        }
    }

    /// Simplify array select operation.
    fn simplify_select(&mut self, array: TermId, index: TermId, original: TermId) -> TermId {
        // Check if array is a store operation
        if let Some(array_term) = self.manager.get(array)
            && let TermKind::Store(base_array, store_index, store_value) = &array_term.kind
        {
            // select(store(arr, i, v), j)
            // If i = j, return v
            // If i != j, return select(arr, j)

            if index == *store_index {
                // Indices are equal
                self.stats.store_select_simplified += 1;
                return *store_value;
            }

            // TODO: Check if indices are provably different
            // For now, recursively simplify
            return self.simplify_select(*base_array, index, original);
        }

        // Propagate index bounds
        if self.config.propagate_index_bounds {
            self.propagate_index_bounds(array, index);
        }

        original
    }

    /// Simplify array store operation.
    fn simplify_store(
        &mut self,
        array: TermId,
        index: TermId,
        value: TermId,
        original: TermId,
    ) -> TermId {
        // Check for redundant stores: store(store(arr, i, v1), i, v2) = store(arr, i, v2)
        if let Some(array_term) = self.manager.get(array)
            && let TermKind::Store(_base_array, store_index, _store_value) = &array_term.kind
            && index == *store_index
        {
            // Same index - redundant store
            self.stats.store_select_simplified += 1;
            // Return store(base_array, index, value)
            // For now, just return original
            return original;
        }

        // Propagate value bounds
        if self.config.propagate_value_bounds {
            self.propagate_value_bounds(array, value);
        }

        original
    }

    /// Propagate index bounds for array access.
    fn propagate_index_bounds(&mut self, array: TermId, index: TermId) {
        // Try to extract bounds from index term
        let index_bound = self.extract_index_bound(index);

        if let Some(bound) = index_bound {
            // Update array's index bounds
            let entry = self
                .index_bounds
                .entry(array)
                .or_insert_with(IndexBound::unbounded);

            let new_bound = entry.intersect(&bound);

            if new_bound.is_contradictory() {
                self.stats.contradictions_detected += 1;
            }

            if new_bound.is_tighter_than(entry) {
                *entry = new_bound;
                self.stats.index_bounds_propagated += 1;
            }
        }
    }

    /// Extract index bound from term.
    fn extract_index_bound(&self, _index: TermId) -> Option<IndexBound> {
        // Simplified: would analyze term structure for bounds
        // e.g., i >= 0 && i < 10 => IndexBound { lower: Some(0), upper: Some(10) }
        None
    }

    /// Propagate value bounds for array store.
    fn propagate_value_bounds(&mut self, array: TermId, _value: TermId) {
        // Simplified: would analyze value term for bounds
        // and update array's value bounds
        self.stats.value_bounds_propagated += 1;

        self.value_bounds.entry(array).or_insert((None, None));
    }

    /// Get index bounds for an array.
    pub fn get_index_bounds(&self, array: TermId) -> Option<&IndexBound> {
        self.index_bounds.get(&array)
    }

    /// Get value bounds for an array.
    pub fn get_value_bounds(&self, array: TermId) -> Option<&(Option<i64>, Option<i64>)> {
        self.value_bounds.get(&array)
    }

    /// Check if any contradictions were detected.
    pub fn has_contradictions(&self) -> bool {
        self.stats.contradictions_detected > 0
    }

    /// Clear all bounds information.
    pub fn clear_bounds(&mut self) {
        self.index_bounds.clear();
        self.value_bounds.clear();
        self.rewrite_cache.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &ArrayBoundsStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ArrayBoundsStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_bound_creation() {
        let bound = IndexBound::new(Some(0), Some(10));
        assert_eq!(bound.lower, Some(0));
        assert_eq!(bound.upper, Some(10));
        assert!(!bound.is_contradictory());
    }

    #[test]
    fn test_index_bound_intersection() {
        let bound1 = IndexBound::new(Some(0), Some(10));
        let bound2 = IndexBound::new(Some(5), Some(15));

        let intersection = bound1.intersect(&bound2);
        assert_eq!(intersection.lower, Some(5));
        assert_eq!(intersection.upper, Some(10));
    }

    #[test]
    fn test_contradictory_bounds() {
        let bound = IndexBound::new(Some(10), Some(5));
        assert!(bound.is_contradictory());

        let valid = IndexBound::new(Some(5), Some(10));
        assert!(!valid.is_contradictory());
    }

    #[test]
    fn test_tighter_bounds() {
        let loose = IndexBound::new(Some(0), Some(100));
        let tight = IndexBound::new(Some(10), Some(50));

        assert!(tight.is_tighter_than(&loose));
        assert!(!loose.is_tighter_than(&tight));
    }

    #[test]
    fn test_tactic_creation() {
        let manager = TermManager::default();
        let tactic = ArrayBoundsTactic::default_config(manager);
        assert_eq!(tactic.stats().index_bounds_propagated, 0);
    }

    #[test]
    fn test_config_default() {
        let config = ArrayBoundsConfig::default();
        assert!(config.propagate_index_bounds);
        assert!(config.propagate_value_bounds);
        assert!(config.simplify_store_select);
    }

    #[test]
    fn test_stats() {
        let manager = TermManager::default();
        let mut tactic = ArrayBoundsTactic::default_config(manager);

        tactic.stats.index_bounds_propagated = 5;
        tactic.stats.store_select_simplified = 3;

        assert_eq!(tactic.stats().index_bounds_propagated, 5);
        assert_eq!(tactic.stats().store_select_simplified, 3);

        tactic.reset_stats();
        assert_eq!(tactic.stats().index_bounds_propagated, 0);
    }
}
