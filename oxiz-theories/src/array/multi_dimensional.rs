//! Multi-Dimensional Array Support
//!
//! This module implements support for N-dimensional arrays, including:
//! - N-dimensional indexing and addressing
//! - Dimension tracking and validation
//! - Flattening strategies for multi-dimensional arrays
//! - Specialized lemmas for multi-dimensional reasoning
//!
//! Reference: Z3's array_decl_plugin.cpp and theory_array_full.cpp

#![allow(missing_docs)]

use oxiz_core::error::{OxizError, Result};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// Represents the dimensionality of an array
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArrayDimensions {
    /// Number of dimensions
    pub num_dims: usize,
    /// Domain sorts for each dimension (index types)
    pub domain_sorts: Vec<SortId>,
    /// Range sort (element type)
    pub range_sort: SortId,
}

/// Sort identifier (placeholder for actual sort system)
pub type SortId = u32;

impl ArrayDimensions {
    /// Create a new array dimension descriptor
    pub fn new(domain_sorts: Vec<SortId>, range_sort: SortId) -> Self {
        let num_dims = domain_sorts.len();
        Self {
            num_dims,
            domain_sorts,
            range_sort,
        }
    }

    /// Check if this is a single-dimensional array
    pub fn is_single_dimensional(&self) -> bool {
        self.num_dims == 1
    }

    /// Get the dimension at a specific index
    pub fn get_dimension(&self, idx: usize) -> Option<SortId> {
        self.domain_sorts.get(idx).copied()
    }

    /// Check if dimensions are compatible for operations
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.num_dims == other.num_dims
            && self.domain_sorts == other.domain_sorts
            && self.range_sort == other.range_sort
    }
}

impl fmt::Display for ArrayDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array[")?;
        for (i, sort) in self.domain_sorts.iter().enumerate() {
            if i > 0 {
                write!(f, " × ")?;
            }
            write!(f, "{}", sort)?;
        }
        write!(f, " → {}]", self.range_sort)
    }
}

/// Multi-dimensional select operation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiDimSelect {
    /// The array being indexed
    pub array: u32,
    /// The indices (one per dimension)
    pub indices: Vec<u32>,
    /// Cached hash for performance
    hash: u64,
}

impl MultiDimSelect {
    /// Create a new multi-dimensional select
    pub fn new(array: u32, indices: Vec<u32>) -> Self {
        let hash = Self::compute_hash(array, &indices);
        Self {
            array,
            indices,
            hash,
        }
    }

    /// Compute hash for this select operation
    fn compute_hash(array: u32, indices: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        array.hash(&mut hasher);
        indices.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the number of indices
    pub fn num_indices(&self) -> usize {
        self.indices.len()
    }

    /// Check if this is a single-dimensional select
    pub fn is_single_dimensional(&self) -> bool {
        self.indices.len() == 1
    }

    /// Get a specific index
    pub fn get_index(&self, dim: usize) -> Option<u32> {
        self.indices.get(dim).copied()
    }
}

/// Multi-dimensional store operation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MultiDimStore {
    /// The base array
    pub array: u32,
    /// The indices being written to
    pub indices: Vec<u32>,
    /// The value being written
    pub value: u32,
    /// Cached hash
    hash: u64,
}

impl MultiDimStore {
    /// Create a new multi-dimensional store
    pub fn new(array: u32, indices: Vec<u32>, value: u32) -> Self {
        let hash = Self::compute_hash(array, &indices, value);
        Self {
            array,
            indices,
            value,
            hash,
        }
    }

    /// Compute hash for this store operation
    fn compute_hash(array: u32, indices: &[u32], value: u32) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        array.hash(&mut hasher);
        indices.hash(&mut hasher);
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// Get the number of indices
    pub fn num_indices(&self) -> usize {
        self.indices.len()
    }

    /// Check if this is a single-dimensional store
    pub fn is_single_dimensional(&self) -> bool {
        self.indices.len() == 1
    }
}

/// Array flattening strategy for multi-dimensional arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatteningStrategy {
    /// Row-major ordering (C-style)
    RowMajor,
    /// Column-major ordering (Fortran-style)
    ColumnMajor,
    /// Nested array representation
    Nested,
}

/// Multi-dimensional array manager
pub struct MultiDimArrayManager {
    /// Dimension information for arrays
    array_dims: FxHashMap<u32, ArrayDimensions>,
    /// Multi-dimensional selects
    md_selects: Vec<(u32, MultiDimSelect)>,
    /// Multi-dimensional stores
    md_stores: Vec<(u32, MultiDimStore)>,
    /// Flattening strategy
    flattening_strategy: FlatteningStrategy,
    /// Cache for flattened representations
    flattened_cache: FxHashMap<u32, u32>,
    /// Dimension bounds (for finite arrays)
    dimension_bounds: FxHashMap<u32, Vec<Option<u64>>>,
}

impl Default for MultiDimArrayManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiDimArrayManager {
    /// Create a new multi-dimensional array manager
    pub fn new() -> Self {
        Self {
            array_dims: FxHashMap::default(),
            md_selects: Vec::new(),
            md_stores: Vec::new(),
            flattening_strategy: FlatteningStrategy::RowMajor,
            flattened_cache: FxHashMap::default(),
            dimension_bounds: FxHashMap::default(),
        }
    }

    /// Register an array with its dimensions
    pub fn register_array(&mut self, array: u32, dims: ArrayDimensions) {
        self.array_dims.insert(array, dims);
    }

    /// Get dimensions for an array
    pub fn get_dimensions(&self, array: u32) -> Option<&ArrayDimensions> {
        self.array_dims.get(&array)
    }

    /// Set dimension bounds for an array
    pub fn set_dimension_bounds(&mut self, array: u32, bounds: Vec<Option<u64>>) {
        self.dimension_bounds.insert(array, bounds);
    }

    /// Get dimension bounds for an array
    pub fn get_dimension_bounds(&self, array: u32) -> Option<&[Option<u64>]> {
        self.dimension_bounds
            .get(&array)
            .map(|v: &Vec<Option<u64>>| v.as_slice())
    }

    /// Check if dimensions are valid for an array
    pub fn validate_indices(&self, array: u32, indices: &[u32]) -> Result<()> {
        if let Some(dims) = self.array_dims.get(&array) {
            if indices.len() != dims.num_dims {
                return Err(OxizError::Internal(format!(
                    "Expected {} indices, got {}",
                    dims.num_dims,
                    indices.len()
                )));
            }
            Ok(())
        } else {
            Err(OxizError::Internal(format!("Unknown array: {}", array)))
        }
    }

    /// Register a multi-dimensional select
    pub fn register_select(&mut self, result: u32, array: u32, indices: Vec<u32>) -> Result<()> {
        self.validate_indices(array, &indices)?;
        let select = MultiDimSelect::new(array, indices);
        self.md_selects.push((result, select));
        Ok(())
    }

    /// Register a multi-dimensional store
    pub fn register_store(
        &mut self,
        result: u32,
        array: u32,
        indices: Vec<u32>,
        value: u32,
    ) -> Result<()> {
        self.validate_indices(array, &indices)?;
        let store = MultiDimStore::new(array, indices, value);
        self.md_stores.push((result, store));
        Ok(())
    }

    /// Set the flattening strategy
    pub fn set_flattening_strategy(&mut self, strategy: FlatteningStrategy) {
        self.flattening_strategy = strategy;
    }

    /// Get the flattening strategy
    pub fn get_flattening_strategy(&self) -> FlatteningStrategy {
        self.flattening_strategy
    }

    /// Flatten a multi-dimensional index to a linear index
    ///
    /// For row-major: index = i₀ * (d₁ * d₂ * ... * dₙ) + i₁ * (d₂ * ... * dₙ) + ... + iₙ
    /// For column-major: index = i₀ + i₁ * d₀ + i₂ * (d₀ * d₁) + ...
    pub fn flatten_index(
        &self,
        _array: u32,
        indices: &[u32],
        bounds: &[u64],
    ) -> Option<Vec<(u32, u64)>> {
        if indices.len() != bounds.len() {
            return None;
        }

        match self.flattening_strategy {
            FlatteningStrategy::RowMajor => {
                // Compute multipliers for each dimension
                let mut multipliers = vec![1u64; indices.len()];
                for i in (0..indices.len() - 1).rev() {
                    multipliers[i] = multipliers[i + 1] * bounds[i + 1];
                }
                Some(indices.iter().copied().zip(multipliers).collect())
            }
            FlatteningStrategy::ColumnMajor => {
                // Compute multipliers for column-major
                let mut multipliers = vec![1u64; indices.len()];
                for i in 1..indices.len() {
                    multipliers[i] = multipliers[i - 1] * bounds[i - 1];
                }
                Some(indices.iter().copied().zip(multipliers).collect())
            }
            FlatteningStrategy::Nested => {
                // For nested representation, no flattening needed
                None
            }
        }
    }

    /// Convert a multi-dimensional array to a flattened representation
    pub fn flatten_array(&mut self, array: u32) -> Option<u32> {
        if let Some(&flattened) = self.flattened_cache.get(&array) {
            return Some(flattened);
        }

        // Check if we have dimension bounds
        if let Some(_bounds) = self.dimension_bounds.get(&array) {
            // Create a flattened array node
            let flattened_id = array + 1000000; // Simple ID generation
            self.flattened_cache.insert(array, flattened_id);
            Some(flattened_id)
        } else {
            None
        }
    }

    /// Get all selects on a specific array
    pub fn get_selects_on_array(&self, array: u32) -> Vec<(u32, &MultiDimSelect)> {
        self.md_selects
            .iter()
            .filter(|(_, sel)| sel.array == array)
            .map(|(id, sel)| (*id, sel))
            .collect()
    }

    /// Get all stores on a specific array
    pub fn get_stores_on_array(&self, array: u32) -> Vec<(u32, &MultiDimStore)> {
        self.md_stores
            .iter()
            .filter(|(_, store)| store.array == array)
            .map(|(id, store)| (*id, store))
            .collect()
    }

    /// Check if two index sequences are equal
    pub fn indices_equal(
        &self,
        indices1: &[u32],
        indices2: &[u32],
        equiv_classes: &dyn Fn(u32, u32) -> bool,
    ) -> bool {
        if indices1.len() != indices2.len() {
            return false;
        }
        indices1
            .iter()
            .zip(indices2.iter())
            .all(|(&i1, &i2)| equiv_classes(i1, i2))
    }

    /// Check if two index sequences are provably different at some dimension
    pub fn indices_differ(
        &self,
        indices1: &[u32],
        indices2: &[u32],
        diseq_check: &dyn Fn(u32, u32) -> bool,
    ) -> bool {
        if indices1.len() != indices2.len() {
            return true;
        }
        indices1
            .iter()
            .zip(indices2.iter())
            .any(|(&i1, &i2)| diseq_check(i1, i2))
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.array_dims.clear();
        self.md_selects.clear();
        self.md_stores.clear();
        self.flattened_cache.clear();
        self.dimension_bounds.clear();
    }

    /// Push context (for incremental solving)
    pub fn push(&self) -> MultiDimContext {
        MultiDimContext {
            num_selects: self.md_selects.len(),
            num_stores: self.md_stores.len(),
            num_arrays: self.array_dims.len(),
        }
    }

    /// Pop context
    pub fn pop(&mut self, ctx: &MultiDimContext) {
        self.md_selects.truncate(ctx.num_selects);
        self.md_stores.truncate(ctx.num_stores);

        // Remove arrays added after the context
        if self.array_dims.len() > ctx.num_arrays {
            let to_remove: Vec<_> = self
                .array_dims
                .keys()
                .copied()
                .skip(ctx.num_arrays)
                .collect();
            for key in to_remove {
                self.array_dims.remove(&key);
                self.dimension_bounds.remove(&key);
                self.flattened_cache.remove(&key);
            }
        }
    }
}

/// Context state for multi-dimensional arrays
#[derive(Debug, Clone)]
pub struct MultiDimContext {
    num_selects: usize,
    num_stores: usize,
    num_arrays: usize,
}

/// Lemma generator for multi-dimensional arrays
pub struct MultiDimLemmaGenerator {
    /// Reference to the multi-dimensional manager
    manager: MultiDimArrayManager,
}

impl MultiDimLemmaGenerator {
    /// Create a new lemma generator
    pub fn new(manager: MultiDimArrayManager) -> Self {
        Self { manager }
    }

    /// Generate read-over-write lemmas for multi-dimensional arrays
    ///
    /// For select(store(a, i₁...iₙ, v), j₁...jₙ):
    /// - If all iₖ = jₖ: result = v
    /// - If any iₖ ≠ jₖ: result = select(a, j₁...jₙ)
    pub fn generate_read_over_write_lemmas(&self) -> Vec<MultiDimLemma> {
        let mut lemmas = Vec::new();

        for (store_result, store) in &self.manager.md_stores {
            for (select_result, select) in &self.manager.md_selects {
                // Check if the select reads from this store result
                if select.array == *store_result || select.array == store.array {
                    let lemma = MultiDimLemma::ReadOverWrite {
                        select_result: *select_result,
                        select_array: select.array,
                        select_indices: select.indices.clone(),
                        store_result: *store_result,
                        store_array: store.array,
                        store_indices: store.indices.clone(),
                        store_value: store.value,
                    };
                    lemmas.push(lemma);
                }
            }
        }

        lemmas
    }

    /// Generate extensionality lemmas for multi-dimensional arrays
    ///
    /// If a₁ = a₂, then for all indices i₁...iₙ:
    /// select(a₁, i₁...iₙ) = select(a₂, i₁...iₙ)
    pub fn generate_extensionality_lemmas(&self, array_pairs: &[(u32, u32)]) -> Vec<MultiDimLemma> {
        let mut lemmas = Vec::new();

        for &(array1, array2) in array_pairs {
            if let (Some(dims1), Some(dims2)) = (
                self.manager.get_dimensions(array1),
                self.manager.get_dimensions(array2),
            ) && dims1.is_compatible_with(dims2)
            {
                lemmas.push(MultiDimLemma::Extensionality {
                    array1,
                    array2,
                    dimensions: dims1.clone(),
                });
            }
        }

        lemmas
    }

    /// Generate lemmas for array default values
    pub fn generate_default_value_lemmas(&self) -> Vec<MultiDimLemma> {
        let mut lemmas = Vec::new();

        // For each array, if it has a default value, generate appropriate lemmas
        for (&array, dims) in &self.manager.array_dims {
            // Check if this is a constant array
            let dimensions: ArrayDimensions = dims.clone();
            lemmas.push(MultiDimLemma::DefaultValue { array, dimensions });
        }

        lemmas
    }
}

/// Types of lemmas for multi-dimensional arrays
#[derive(Debug, Clone)]
pub enum MultiDimLemma {
    /// Read-over-write lemma
    ReadOverWrite {
        select_result: u32,
        select_array: u32,
        select_indices: Vec<u32>,
        store_result: u32,
        store_array: u32,
        store_indices: Vec<u32>,
        store_value: u32,
    },
    /// Extensionality lemma
    Extensionality {
        array1: u32,
        array2: u32,
        dimensions: ArrayDimensions,
    },
    /// Default value lemma
    DefaultValue {
        array: u32,
        dimensions: ArrayDimensions,
    },
}

/// Index difference analyzer for multi-dimensional arrays
pub struct IndexDifferenceAnalyzer {
    /// Known index equalities
    index_eqs: FxHashSet<(u32, u32)>,
    /// Known index disequalities
    index_diseqs: FxHashSet<(u32, u32)>,
}

impl Default for IndexDifferenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexDifferenceAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            index_eqs: FxHashSet::default(),
            index_diseqs: FxHashSet::default(),
        }
    }

    /// Assert that two indices are equal
    pub fn assert_equal(&mut self, idx1: u32, idx2: u32) {
        if idx1 != idx2 {
            self.index_eqs.insert((idx1.min(idx2), idx1.max(idx2)));
        }
    }

    /// Assert that two indices are different
    pub fn assert_different(&mut self, idx1: u32, idx2: u32) {
        if idx1 != idx2 {
            self.index_diseqs.insert((idx1.min(idx2), idx1.max(idx2)));
        }
    }

    /// Check if two indices are known to be equal
    pub fn are_equal(&self, idx1: u32, idx2: u32) -> bool {
        idx1 == idx2 || self.index_eqs.contains(&(idx1.min(idx2), idx1.max(idx2)))
    }

    /// Check if two indices are known to be different
    pub fn are_different(&self, idx1: u32, idx2: u32) -> bool {
        self.index_diseqs
            .contains(&(idx1.min(idx2), idx1.max(idx2)))
    }

    /// Analyze index sequences
    pub fn analyze_sequences(&self, indices1: &[u32], indices2: &[u32]) -> IndexSequenceRelation {
        if indices1.len() != indices2.len() {
            return IndexSequenceRelation::DifferentDimensions;
        }

        // AllEqual is reserved for syntactically identical sequences
        if indices1 == indices2 {
            return IndexSequenceRelation::AllEqual;
        }

        let _all_equal = false; // Changed from true since sequences are different
        let mut has_difference = false;
        let mut diff_positions = Vec::new();

        for (pos, (&idx1, &idx2)) in indices1.iter().zip(indices2.iter()).enumerate() {
            if self.are_different(idx1, idx2) {
                has_difference = true;
                diff_positions.push(pos);
            }
        }

        if has_difference {
            IndexSequenceRelation::HasDifference(diff_positions)
        } else {
            IndexSequenceRelation::Unknown
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.index_eqs.clear();
        self.index_diseqs.clear();
    }
}

/// Relationship between two index sequences
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexSequenceRelation {
    /// All indices are known to be equal
    AllEqual,
    /// At least one index is provably different (positions listed)
    HasDifference(Vec<usize>),
    /// Different number of dimensions
    DifferentDimensions,
    /// Relationship is unknown
    Unknown,
}

/// Dimension iterator for bounded arrays
pub struct DimensionIterator {
    /// Current indices
    current: Vec<u64>,
    /// Bounds for each dimension
    bounds: Vec<u64>,
    /// Whether iteration is complete
    done: bool,
}

impl DimensionIterator {
    /// Create a new dimension iterator
    pub fn new(bounds: Vec<u64>) -> Self {
        let current = vec![0; bounds.len()];
        let done = bounds.is_empty() || bounds.contains(&0);
        Self {
            current,
            bounds,
            done,
        }
    }

    /// Get the current indices
    pub fn current(&self) -> &[u64] {
        &self.current
    }

    /// Check if iteration is complete
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Advance to the next index combination
    pub fn next_indices(&mut self) -> Option<Vec<u64>> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        // Increment indices (row-major order)
        let mut carry = true;
        for i in (0..self.current.len()).rev() {
            if carry {
                self.current[i] += 1;
                if self.current[i] >= self.bounds[i] {
                    self.current[i] = 0;
                } else {
                    carry = false;
                }
            }
        }

        if carry {
            self.done = true;
        }

        Some(result)
    }

    /// Get the total number of combinations
    pub fn total_combinations(&self) -> u64 {
        self.bounds.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_dimensions() {
        let dims = ArrayDimensions::new(vec![1, 2], 3);
        assert_eq!(dims.num_dims, 2);
        assert!(!dims.is_single_dimensional());
        assert_eq!(dims.get_dimension(0), Some(1));
        assert_eq!(dims.get_dimension(1), Some(2));
        assert_eq!(dims.get_dimension(2), None);
    }

    #[test]
    fn test_multi_dim_select() {
        let select = MultiDimSelect::new(10, vec![1, 2, 3]);
        assert_eq!(select.num_indices(), 3);
        assert!(!select.is_single_dimensional());
        assert_eq!(select.get_index(1), Some(2));
    }

    #[test]
    fn test_multi_dim_manager() {
        let mut mgr = MultiDimArrayManager::new();
        let dims = ArrayDimensions::new(vec![1, 2], 3);
        mgr.register_array(10, dims);

        assert!(mgr.register_select(100, 10, vec![5, 6]).is_ok());
        assert!(mgr.register_select(101, 10, vec![5]).is_err()); // Wrong number of indices

        let selects = mgr.get_selects_on_array(10);
        assert_eq!(selects.len(), 1);
    }

    #[test]
    fn test_flattening_row_major() {
        let mgr = MultiDimArrayManager::new();
        let indices = vec![1, 2, 3];
        let bounds = vec![10, 20, 30];

        let result = mgr.flatten_index(0, &indices, &bounds);
        assert!(result.is_some());
        let flattened = result.unwrap();
        // For row-major: multipliers are [600, 30, 1]
        // index = 1*600 + 2*30 + 3*1 = 663
        assert_eq!(flattened.len(), 3);
    }

    #[test]
    fn test_index_difference_analyzer() {
        let mut analyzer = IndexDifferenceAnalyzer::new();

        analyzer.assert_equal(1, 2);
        analyzer.assert_different(3, 4);

        assert!(analyzer.are_equal(1, 2));
        assert!(analyzer.are_equal(2, 1));
        assert!(analyzer.are_different(3, 4));
        assert!(!analyzer.are_different(1, 2));
    }

    #[test]
    fn test_index_sequence_relation() {
        let mut analyzer = IndexDifferenceAnalyzer::new();
        analyzer.assert_equal(1, 2);
        analyzer.assert_different(5, 6);

        let rel1 = analyzer.analyze_sequences(&[1, 3], &[2, 3]);
        assert!(matches!(rel1, IndexSequenceRelation::Unknown));

        let rel2 = analyzer.analyze_sequences(&[5, 3], &[6, 3]);
        assert!(matches!(rel2, IndexSequenceRelation::HasDifference(_)));
    }

    #[test]
    fn test_dimension_iterator() {
        let mut iter = DimensionIterator::new(vec![2, 3]);

        let mut count = 0;
        while let Some(_indices) = iter.next_indices() {
            count += 1;
        }

        assert_eq!(count, 6); // 2 * 3 = 6 combinations
        assert_eq!(iter.total_combinations(), 6);
    }

    #[test]
    fn test_dimension_iterator_empty() {
        let mut iter = DimensionIterator::new(vec![]);
        assert!(iter.is_done());
        assert!(iter.next_indices().is_none());
    }

    #[test]
    fn test_multi_dim_store() {
        let store = MultiDimStore::new(10, vec![1, 2], 100);
        assert_eq!(store.num_indices(), 2);
        assert!(!store.is_single_dimensional());
    }

    #[test]
    fn test_dimension_bounds() {
        let mut mgr = MultiDimArrayManager::new();
        mgr.set_dimension_bounds(10, vec![Some(5), Some(10), None]);

        let bounds = mgr.get_dimension_bounds(10);
        assert!(bounds.is_some());
        assert_eq!(bounds.unwrap().len(), 3);
    }

    #[test]
    fn test_push_pop_context() {
        let mut mgr = MultiDimArrayManager::new();
        let dims = ArrayDimensions::new(vec![1], 2);

        mgr.register_array(10, dims.clone());
        let ctx = mgr.push();

        mgr.register_array(20, dims);
        mgr.register_select(100, 10, vec![5]).unwrap();

        mgr.pop(&ctx);
        assert_eq!(mgr.md_selects.len(), 0);
    }

    #[test]
    fn test_compatibility() {
        let dims1 = ArrayDimensions::new(vec![1, 2], 3);
        let dims2 = ArrayDimensions::new(vec![1, 2], 3);
        let dims3 = ArrayDimensions::new(vec![1, 2], 4);

        assert!(dims1.is_compatible_with(&dims2));
        assert!(!dims1.is_compatible_with(&dims3));
    }

    #[test]
    fn test_indices_equal() {
        let mgr = MultiDimArrayManager::new();
        let eq_check = |a: u32, b: u32| a == b;

        assert!(mgr.indices_equal(&[1, 2, 3], &[1, 2, 3], &eq_check));
        assert!(!mgr.indices_equal(&[1, 2, 3], &[1, 2, 4], &eq_check));
    }

    #[test]
    fn test_indices_differ() {
        let mgr = MultiDimArrayManager::new();
        let diseq_check = |a: u32, b: u32| a != b && (a == 5 || b == 5);

        assert!(mgr.indices_differ(&[5, 2], &[6, 2], &diseq_check));
        assert!(!mgr.indices_differ(&[1, 2], &[1, 2], &diseq_check));
    }
}
