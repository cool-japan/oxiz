//! Powerset Operations
//!
//! Handles powerset constraints and operations

#![allow(dead_code)]

use super::{SetConflict, SetVarId};
use rustc_hash::{FxHashMap, FxHashSet};

/// Powerset constraint: S2 = P(S1)
#[derive(Debug, Clone)]
pub struct PowersetConstraint {
    /// Base set
    pub base: SetVarId,
    /// Powerset
    pub powerset: SetVarId,
    /// Decision level when added
    pub level: usize,
}

impl PowersetConstraint {
    /// Create a new powerset constraint
    pub fn new(base: SetVarId, powerset: SetVarId, level: usize) -> Self {
        Self {
            base,
            powerset,
            level,
        }
    }

    /// Check if the cardinality is consistent
    ///
    /// If |base| = n, then |powerset| = 2^n
    pub fn check_cardinality(&self, base_card: i64) -> Option<i64> {
        if !(0..=30).contains(&base_card) {
            // Avoid overflow for large sets
            None
        } else {
            Some(1i64 << base_card)
        }
    }
}

/// Powerset result
pub type PowersetResult<T> = Result<T, SetConflict>;

/// Powerset statistics
#[derive(Debug, Clone, Default)]
pub struct PowersetStats {
    /// Number of powerset constraints
    pub num_constraints: usize,
    /// Number of powersets generated
    pub num_generated: usize,
    /// Total subsets enumerated
    pub total_subsets: usize,
}

/// Powerset builder
#[derive(Debug)]
pub struct PowersetBuilder {
    /// Base set elements
    base_elements: Vec<u32>,
    /// Generated subsets
    subsets: Vec<FxHashSet<u32>>,
    /// Statistics
    stats: PowersetStats,
}

impl PowersetBuilder {
    /// Create a new powerset builder
    pub fn new() -> Self {
        Self {
            base_elements: Vec::new(),
            subsets: Vec::new(),
            stats: PowersetStats::default(),
        }
    }

    /// Set the base set elements
    pub fn with_base(mut self, elements: Vec<u32>) -> Self {
        self.base_elements = elements;
        self
    }

    /// Build the powerset
    pub fn build(mut self) -> Vec<FxHashSet<u32>> {
        self.generate_powerset();
        self.subsets
    }

    fn generate_powerset(&mut self) {
        let n = self.base_elements.len();
        let powerset_size = 1usize << n;

        self.subsets.clear();
        self.subsets.reserve(powerset_size);

        for i in 0..powerset_size {
            let mut subset = FxHashSet::default();
            for (j, &elem) in self.base_elements.iter().enumerate() {
                if (i & (1 << j)) != 0 {
                    subset.insert(elem);
                }
            }
            self.subsets.push(subset);
            self.stats.total_subsets += 1;
        }

        self.stats.num_generated += 1;
    }

    /// Get statistics
    pub fn stats(&self) -> &PowersetStats {
        &self.stats
    }
}

impl Default for PowersetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Powerset iterator (lazy generation)
#[derive(Debug)]
pub struct PowersetIter {
    /// Base set elements
    base: Vec<u32>,
    /// Current index
    index: usize,
    /// Total size (2^n)
    total: usize,
}

impl PowersetIter {
    /// Create a new powerset iterator
    pub fn new(base: Vec<u32>) -> Self {
        let total = 1usize << base.len();
        Self {
            base,
            index: 0,
            total,
        }
    }

    /// Get the current subset count
    pub fn count(&self) -> usize {
        self.total
    }

    /// Check if there are more subsets
    pub fn has_next(&self) -> bool {
        self.index < self.total
    }
}

impl Iterator for PowersetIter {
    type Item = FxHashSet<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }

        let mut subset = FxHashSet::default();
        for (j, &elem) in self.base.iter().enumerate() {
            if (self.index & (1 << j)) != 0 {
                subset.insert(elem);
            }
        }

        self.index += 1;
        Some(subset)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PowersetIter {}

/// Powerset manager
#[derive(Debug)]
pub struct PowersetManager {
    /// Powerset constraints
    #[allow(dead_code)]
    constraints: Vec<PowersetConstraint>,
    /// Generated powersets
    powersets: FxHashMap<SetVarId, Vec<FxHashSet<u32>>>,
    /// Statistics
    stats: PowersetStats,
}

impl PowersetManager {
    /// Create a new powerset manager
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            powersets: FxHashMap::default(),
            stats: PowersetStats::default(),
        }
    }

    /// Add a powerset constraint
    #[allow(dead_code)]
    pub fn add_constraint(&mut self, constraint: PowersetConstraint) {
        self.constraints.push(constraint);
        self.stats.num_constraints += 1;
    }

    /// Generate powerset for a base set
    pub fn generate(
        &mut self,
        base: SetVarId,
        base_elements: Vec<u32>,
    ) -> PowersetResult<Vec<FxHashSet<u32>>> {
        if base_elements.len() > 20 {
            return Err(SetConflict {
                literals: vec![],
                reason: format!(
                    "Powerset too large: base set has {} elements (max 20)",
                    base_elements.len()
                ),
                proof_steps: vec![],
            });
        }

        let builder = PowersetBuilder::new().with_base(base_elements);
        let powerset = builder.build();

        self.powersets.insert(base, powerset.clone());
        self.stats.num_generated += 1;
        self.stats.total_subsets += powerset.len();

        Ok(powerset)
    }

    /// Get generated powerset
    #[allow(dead_code)]
    pub fn get_powerset(&self, base: SetVarId) -> Option<&Vec<FxHashSet<u32>>> {
        self.powersets.get(&base)
    }

    /// Check if a set is a member of a powerset
    pub fn contains(&self, base: SetVarId, subset: &FxHashSet<u32>) -> bool {
        if let Some(powerset) = self.powersets.get(&base) {
            powerset.contains(subset)
        } else {
            false
        }
    }

    /// Get statistics
    #[allow(dead_code)]
    pub fn stats(&self) -> &PowersetStats {
        &self.stats
    }

    /// Reset the manager
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.constraints.clear();
        self.powersets.clear();
        self.stats = PowersetStats::default();
    }
}

impl Default for PowersetManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Powerset cardinality reasoning
#[derive(Debug)]
pub struct PowersetCardinality {
    /// Base set cardinality to powerset cardinality mapping
    cache: FxHashMap<i64, i64>,
}

impl PowersetCardinality {
    /// Create a new powerset cardinality reasoner
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    /// Compute powerset cardinality from base cardinality
    ///
    /// |P(S)| = 2^|S|
    pub fn compute(&mut self, base_card: i64) -> Option<i64> {
        if let Some(&result) = self.cache.get(&base_card) {
            return Some(result);
        }

        if !(0..=60).contains(&base_card) {
            // Avoid overflow
            return None;
        }

        let result = 1i64.checked_shl(base_card as u32)?;
        self.cache.insert(base_card, result);
        Some(result)
    }

    /// Invert: given powerset cardinality, compute possible base cardinality
    ///
    /// If |P(S)| = k, then |S| = log2(k) (if k is a power of 2)
    pub fn invert(&self, powerset_card: i64) -> Option<i64> {
        // Check if power of two: n > 0 && (n & (n-1)) == 0
        if powerset_card <= 0 || (powerset_card & (powerset_card - 1)) != 0 {
            return None;
        }

        Some(powerset_card.trailing_zeros() as i64)
    }

    /// Check if a powerset cardinality is valid
    #[allow(dead_code)]
    pub fn is_valid_powerset_card(&self, card: i64) -> bool {
        // Check if power of two: n > 0 && (n & (n-1)) == 0
        card > 0 && (card & (card - 1)) == 0
    }
}

impl Default for PowersetCardinality {
    fn default() -> Self {
        Self::new()
    }
}

/// Powerset membership checker
#[derive(Debug)]
pub struct PowersetMembership {
    /// Base set
    base: FxHashSet<u32>,
}

impl PowersetMembership {
    /// Create a new powerset membership checker
    pub fn new(base: FxHashSet<u32>) -> Self {
        Self { base }
    }

    /// Check if a set is a subset (member of the powerset)
    pub fn is_member(&self, subset: &FxHashSet<u32>) -> bool {
        subset.is_subset(&self.base)
    }

    /// Get all subsets of a given size
    pub fn subsets_of_size(&self, size: usize) -> Vec<FxHashSet<u32>> {
        if size > self.base.len() {
            return Vec::new();
        }

        let base_vec: Vec<u32> = self.base.iter().copied().collect();
        let mut result = Vec::new();

        self.generate_combinations(&base_vec, size, 0, &mut Vec::new(), &mut result);

        result
    }

    fn generate_combinations(
        &self,
        base: &[u32],
        size: usize,
        start: usize,
        current: &mut Vec<u32>,
        result: &mut Vec<FxHashSet<u32>>,
    ) {
        if current.len() == size {
            let subset: FxHashSet<u32> = current.iter().copied().collect();
            result.push(subset);
            return;
        }

        if start >= base.len() {
            return;
        }

        for i in start..base.len() {
            current.push(base[i]);
            self.generate_combinations(base, size, i + 1, current, result);
            current.pop();
        }
    }

    /// Count subsets of a given size (binomial coefficient)
    pub fn count_subsets_of_size(&self, size: usize) -> usize {
        if size > self.base.len() {
            return 0;
        }

        Self::binomial(self.base.len(), size)
    }

    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k); // Optimization: C(n,k) = C(n,n-k)
        let mut result = 1;

        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }

        result
    }
}

/// Powerset filter for constrained powersets
#[derive(Debug)]
pub struct PowersetFilter {
    /// Minimum subset size
    min_size: Option<usize>,
    /// Maximum subset size
    max_size: Option<usize>,
    /// Elements that must be included
    must_include: FxHashSet<u32>,
    /// Elements that must be excluded
    must_exclude: FxHashSet<u32>,
}

impl PowersetFilter {
    /// Create a new powerset filter
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            min_size: None,
            max_size: None,
            must_include: FxHashSet::default(),
            must_exclude: FxHashSet::default(),
        }
    }

    /// Set minimum subset size
    #[allow(dead_code)]
    pub fn with_min_size(mut self, size: usize) -> Self {
        self.min_size = Some(size);
        self
    }

    /// Set maximum subset size
    #[allow(dead_code)]
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = Some(size);
        self
    }

    /// Add element that must be included in all subsets
    #[allow(dead_code)]
    pub fn must_include(mut self, elem: u32) -> Self {
        self.must_include.insert(elem);
        self
    }

    /// Add element that must be excluded from all subsets
    #[allow(dead_code)]
    pub fn must_exclude(mut self, elem: u32) -> Self {
        self.must_exclude.insert(elem);
        self
    }

    /// Check if a subset passes the filter
    #[allow(dead_code)]
    pub fn check(&self, subset: &FxHashSet<u32>) -> bool {
        // Check size constraints
        if let Some(min) = self.min_size
            && subset.len() < min
        {
            return false;
        }

        if let Some(max) = self.max_size
            && subset.len() > max
        {
            return false;
        }

        // Check must include
        for &elem in &self.must_include {
            if !subset.contains(&elem) {
                return false;
            }
        }

        // Check must exclude
        for &elem in &self.must_exclude {
            if subset.contains(&elem) {
                return false;
            }
        }

        true
    }

    /// Filter a powerset
    #[allow(dead_code)]
    pub fn filter(&self, powerset: &[FxHashSet<u32>]) -> Vec<FxHashSet<u32>> {
        powerset.iter().filter(|s| self.check(s)).cloned().collect()
    }
}

impl Default for PowersetFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_powerset_builder() {
        let builder = PowersetBuilder::new().with_base(vec![1, 2]);
        let powerset = builder.build();

        assert_eq!(powerset.len(), 4); // {}, {1}, {2}, {1,2}
    }

    #[test]
    fn test_powerset_iterator() {
        let iter = PowersetIter::new(vec![1, 2]);
        let powerset: Vec<_> = iter.collect();

        assert_eq!(powerset.len(), 4);
    }

    #[test]
    fn test_powerset_iterator_size_hint() {
        let iter = PowersetIter::new(vec![1, 2, 3]);
        assert_eq!(iter.size_hint(), (8, Some(8)));
    }

    #[test]
    fn test_powerset_manager() {
        let mut manager = PowersetManager::new();

        let base = SetVarId(0);
        let result = manager.generate(base, vec![1, 2]);
        assert!(result.is_ok());

        let powerset = result.unwrap();
        assert_eq!(powerset.len(), 4);
    }

    #[test]
    fn test_powerset_manager_too_large() {
        let mut manager = PowersetManager::new();

        let base = SetVarId(0);
        let elements: Vec<u32> = (0..25).collect();
        let result = manager.generate(base, elements);
        assert!(result.is_err());
    }

    #[test]
    fn test_powerset_contains() {
        let mut manager = PowersetManager::new();

        let base = SetVarId(0);
        manager.generate(base, vec![1, 2]).unwrap();

        let mut subset = FxHashSet::default();
        subset.insert(1);

        assert!(manager.contains(base, &subset));

        subset.insert(3);
        assert!(!manager.contains(base, &subset));
    }

    #[test]
    fn test_powerset_cardinality() {
        let mut card = PowersetCardinality::new();

        assert_eq!(card.compute(0), Some(1));
        assert_eq!(card.compute(3), Some(8));
        assert_eq!(card.compute(10), Some(1024));
    }

    #[test]
    fn test_powerset_cardinality_invert() {
        let card = PowersetCardinality::new();

        assert_eq!(card.invert(1), Some(0));
        assert_eq!(card.invert(8), Some(3));
        assert_eq!(card.invert(1024), Some(10));

        assert_eq!(card.invert(7), None); // Not a power of 2
    }

    #[test]
    fn test_powerset_cardinality_overflow() {
        let mut card = PowersetCardinality::new();

        // Should return None for very large values to avoid overflow
        assert_eq!(card.compute(100), None);
    }

    #[test]
    fn test_powerset_membership() {
        let mut base = FxHashSet::default();
        base.insert(1);
        base.insert(2);
        base.insert(3);

        let checker = PowersetMembership::new(base);

        let mut subset1 = FxHashSet::default();
        subset1.insert(1);
        subset1.insert(2);
        assert!(checker.is_member(&subset1));

        let mut subset2 = FxHashSet::default();
        subset2.insert(1);
        subset2.insert(4);
        assert!(!checker.is_member(&subset2));
    }

    #[test]
    fn test_powerset_subsets_of_size() {
        let mut base = FxHashSet::default();
        base.insert(1);
        base.insert(2);
        base.insert(3);

        let checker = PowersetMembership::new(base);

        let subsets = checker.subsets_of_size(2);
        assert_eq!(subsets.len(), 3); // {1,2}, {1,3}, {2,3}

        for subset in &subsets {
            assert_eq!(subset.len(), 2);
        }
    }

    #[test]
    fn test_powerset_count_subsets_of_size() {
        let mut base = FxHashSet::default();
        base.insert(1);
        base.insert(2);
        base.insert(3);
        base.insert(4);

        let checker = PowersetMembership::new(base);

        assert_eq!(checker.count_subsets_of_size(0), 1); // C(4,0) = 1
        assert_eq!(checker.count_subsets_of_size(1), 4); // C(4,1) = 4
        assert_eq!(checker.count_subsets_of_size(2), 6); // C(4,2) = 6
        assert_eq!(checker.count_subsets_of_size(3), 4); // C(4,3) = 4
        assert_eq!(checker.count_subsets_of_size(4), 1); // C(4,4) = 1
    }

    #[test]
    fn test_powerset_filter() {
        let filter = PowersetFilter::new().with_min_size(1).with_max_size(2);

        let mut subset1 = FxHashSet::default();
        assert!(!filter.check(&subset1)); // Too small

        subset1.insert(1);
        assert!(filter.check(&subset1)); // OK

        subset1.insert(2);
        assert!(filter.check(&subset1)); // OK

        subset1.insert(3);
        assert!(!filter.check(&subset1)); // Too large
    }

    #[test]
    fn test_powerset_filter_must_include() {
        let filter = PowersetFilter::new().must_include(1);

        let mut subset1 = FxHashSet::default();
        subset1.insert(2);
        assert!(!filter.check(&subset1)); // Missing required element

        subset1.insert(1);
        assert!(filter.check(&subset1)); // OK
    }

    #[test]
    fn test_powerset_filter_must_exclude() {
        let filter = PowersetFilter::new().must_exclude(3);

        let mut subset1 = FxHashSet::default();
        subset1.insert(1);
        subset1.insert(2);
        assert!(filter.check(&subset1)); // OK

        subset1.insert(3);
        assert!(!filter.check(&subset1)); // Contains excluded element
    }

    #[test]
    fn test_powerset_filter_apply() {
        let builder = PowersetBuilder::new().with_base(vec![1, 2, 3]);
        let powerset = builder.build();

        let filter = PowersetFilter::new().with_min_size(2);
        let filtered = filter.filter(&powerset);

        // Should filter out sets with size < 2
        for subset in &filtered {
            assert!(subset.len() >= 2);
        }
    }

    #[test]
    fn test_powerset_constraint_check_cardinality() {
        let constraint = PowersetConstraint::new(SetVarId(0), SetVarId(1), 0);

        assert_eq!(constraint.check_cardinality(0), Some(1));
        assert_eq!(constraint.check_cardinality(3), Some(8));
        assert_eq!(constraint.check_cardinality(10), Some(1024));
        assert_eq!(constraint.check_cardinality(50), None); // Too large
    }
}
