//! Finite Set Enumeration
//!
//! Handles finite set enumeration and model generation

#![allow(dead_code)]

use super::{SetConflict, SetVar, SetVarId};
use rustc_hash::{FxHashMap, FxHashSet};

/// Set element (for enumeration)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SetElement(pub u32);

impl SetElement {
    /// Create a new set element
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Get the value
    pub fn value(&self) -> u32 {
        self.0
    }
}

/// Enumerated set (concrete representation)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumSet {
    /// Elements in the set
    pub elements: FxHashSet<u32>,
    /// Is this set infinite?
    pub infinite: bool,
}

impl EnumSet {
    /// Create an empty set
    pub fn empty() -> Self {
        Self {
            elements: FxHashSet::default(),
            infinite: false,
        }
    }

    /// Create a singleton set
    pub fn singleton(elem: u32) -> Self {
        let mut elements = FxHashSet::default();
        elements.insert(elem);
        Self {
            elements,
            infinite: false,
        }
    }

    /// Create a set from elements
    pub fn from_elements(elements: FxHashSet<u32>) -> Self {
        Self {
            elements,
            infinite: false,
        }
    }

    /// Create a set from a range
    pub fn from_range(start: u32, end: u32) -> Self {
        let mut elements = FxHashSet::default();
        for i in start..end {
            elements.insert(i);
        }
        Self {
            elements,
            infinite: false,
        }
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        !self.infinite && self.elements.is_empty()
    }

    /// Get the cardinality
    pub fn cardinality(&self) -> Option<usize> {
        if self.infinite {
            None
        } else {
            Some(self.elements.len())
        }
    }

    /// Check if an element is in the set
    pub fn contains(&self, elem: u32) -> bool {
        self.elements.contains(&elem)
    }

    /// Add an element to the set
    pub fn insert(&mut self, elem: u32) {
        if !self.infinite {
            self.elements.insert(elem);
        }
    }

    /// Remove an element from the set
    pub fn remove(&mut self, elem: u32) {
        self.elements.remove(&elem);
    }

    /// Compute union with another set
    pub fn union(&self, other: &EnumSet) -> EnumSet {
        if self.infinite || other.infinite {
            return EnumSet {
                elements: FxHashSet::default(),
                infinite: true,
            };
        }

        let mut result = self.elements.clone();
        result.extend(&other.elements);
        EnumSet {
            elements: result,
            infinite: false,
        }
    }

    /// Compute intersection with another set
    pub fn intersection(&self, other: &EnumSet) -> EnumSet {
        let elements = self
            .elements
            .intersection(&other.elements)
            .copied()
            .collect();
        EnumSet {
            elements,
            infinite: false,
        }
    }

    /// Compute difference with another set
    pub fn difference(&self, other: &EnumSet) -> EnumSet {
        let elements = self.elements.difference(&other.elements).copied().collect();
        EnumSet {
            elements,
            infinite: self.infinite && !other.infinite,
        }
    }

    /// Check if this is a subset of another set
    pub fn is_subset(&self, other: &EnumSet) -> bool {
        if self.infinite && !other.infinite {
            return false;
        }
        self.elements.is_subset(&other.elements)
    }

    /// Convert to a sorted vector
    pub fn to_sorted_vec(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.elements.iter().copied().collect();
        v.sort_unstable();
        v
    }
}

/// Set enumeration configuration
#[derive(Debug, Clone)]
pub struct SetEnumConfig {
    /// Maximum set size for enumeration
    pub max_size: usize,
    /// Universe size (if finite)
    pub universe_size: Option<usize>,
    /// Enumerate all models or just one
    pub enumerate_all: bool,
}

impl Default for SetEnumConfig {
    fn default() -> Self {
        Self {
            max_size: 100,
            universe_size: None,
            enumerate_all: false,
        }
    }
}

/// Set enumeration result
pub type SetEnumResult<T> = Result<T, SetConflict>;

/// Set enumeration statistics
#[derive(Debug, Clone, Default)]
pub struct SetEnumStats {
    /// Number of sets enumerated
    pub num_enumerated: usize,
    /// Number of models found
    pub num_models: usize,
    /// Total elements considered
    pub total_elements: usize,
}

/// Finite set enumerator
#[derive(Debug)]
pub struct FiniteSetEnumerator {
    /// Configuration
    config: SetEnumConfig,
    /// Statistics
    stats: SetEnumStats,
    /// Universe of elements
    universe: FxHashSet<u32>,
    /// Enumerated sets
    enumerated: FxHashMap<SetVarId, EnumSet>,
}

impl FiniteSetEnumerator {
    /// Create a new enumerator
    pub fn new(config: SetEnumConfig) -> Self {
        let universe = if let Some(size) = config.universe_size {
            (0..size as u32).collect()
        } else {
            FxHashSet::default()
        };

        Self {
            config,
            stats: SetEnumStats::default(),
            universe,
            enumerated: FxHashMap::default(),
        }
    }

    /// Set the universe of elements
    pub fn set_universe(&mut self, universe: FxHashSet<u32>) {
        self.universe = universe;
    }

    /// Add an element to the universe
    pub fn add_to_universe(&mut self, elem: u32) {
        self.universe.insert(elem);
    }

    /// Enumerate a set variable
    pub fn enumerate(&mut self, var: &SetVar) -> SetEnumResult<EnumSet> {
        self.stats.num_enumerated += 1;

        // Start with must_members
        let mut result = EnumSet::from_elements(var.must_members.clone());

        // Check cardinality constraints
        let (lower, upper) = var.cardinality_bounds();

        if lower as usize > self.config.max_size {
            return Err(SetConflict {
                literals: vec![],
                reason: format!(
                    "Set cardinality {} exceeds maximum {}",
                    lower, self.config.max_size
                ),
                proof_steps: vec![],
            });
        }

        // If we have exact cardinality
        if let Some(exact) = upper
            && exact == lower
        {
            // Cardinality is exactly determined
            if result.elements.len() as i64 == exact {
                return Ok(result);
            }

            // Need to add more elements
            let needed = exact as usize - result.elements.len();
            let candidates = self.get_candidates(var);

            if candidates.len() < needed {
                return Err(SetConflict {
                    literals: vec![],
                    reason: format!(
                        "Cannot satisfy cardinality: need {} more elements but only {} candidates",
                        needed,
                        candidates.len()
                    ),
                    proof_steps: vec![],
                });
            }

            // Add the first 'needed' candidates
            for &elem in candidates.iter().take(needed) {
                result.insert(elem);
            }

            return Ok(result);
        }

        // If we have may_members, use them
        if let Some(may) = &var.may_members {
            for &elem in may {
                if !var.must_not_members.contains(&elem) {
                    result.insert(elem);

                    // Check if we've reached the upper bound
                    if let Some(u) = upper
                        && result.elements.len() >= u as usize
                    {
                        break;
                    }
                }
            }
        }

        self.enumerated.insert(var.id, result.clone());
        Ok(result)
    }

    /// Get candidate elements for a set variable
    fn get_candidates(&self, var: &SetVar) -> Vec<u32> {
        if let Some(may) = &var.may_members {
            may.difference(&var.must_members).copied().collect()
        } else {
            self.universe
                .difference(&var.must_members)
                .filter(|e| !var.must_not_members.contains(e))
                .copied()
                .collect()
        }
    }

    /// Enumerate all possible sets for a variable
    pub fn enumerate_all(&mut self, var: &SetVar) -> SetEnumResult<Vec<EnumSet>> {
        let must = &var.must_members;
        let _must_not = &var.must_not_members;
        let (lower, upper) = var.cardinality_bounds();

        let candidates = self.get_candidates(var);

        if candidates.len() + must.len() < lower as usize {
            return Err(SetConflict {
                literals: vec![],
                reason: "Cannot satisfy lower cardinality bound".to_string(),
                proof_steps: vec![],
            });
        }

        let mut results = Vec::new();

        // Generate all subsets of candidates
        let min_additional = (lower as usize).saturating_sub(must.len());
        let max_additional = upper
            .map(|u| (u as usize).saturating_sub(must.len()))
            .unwrap_or(candidates.len())
            .min(candidates.len());

        for size in min_additional..=max_additional {
            self.generate_subsets(&candidates, size, must, &mut results);

            if results.len() >= self.config.max_size {
                break;
            }
        }

        self.stats.num_models = results.len();
        Ok(results)
    }

    fn generate_subsets(
        &self,
        candidates: &[u32],
        size: usize,
        must: &FxHashSet<u32>,
        results: &mut Vec<EnumSet>,
    ) {
        self.generate_subsets_rec(candidates, size, 0, &mut Vec::new(), must, results);
    }

    fn generate_subsets_rec(
        &self,
        candidates: &[u32],
        size: usize,
        start: usize,
        current: &mut Vec<u32>,
        must: &FxHashSet<u32>,
        results: &mut Vec<EnumSet>,
    ) {
        if current.len() == size {
            let mut elements = must.clone();
            elements.extend(current.iter());
            results.push(EnumSet::from_elements(elements));
            return;
        }

        if start >= candidates.len() {
            return;
        }

        // Include candidates[start]
        current.push(candidates[start]);
        self.generate_subsets_rec(candidates, size, start + 1, current, must, results);
        current.pop();

        // Exclude candidates[start]
        self.generate_subsets_rec(candidates, size, start + 1, current, must, results);
    }

    /// Check if an enumerated set satisfies all constraints
    pub fn verify(&self, set: &EnumSet, var: &SetVar) -> bool {
        // Check must members
        for &elem in &var.must_members {
            if !set.contains(elem) {
                return false;
            }
        }

        // Check must not members
        for &elem in &var.must_not_members {
            if set.contains(elem) {
                return false;
            }
        }

        // Check cardinality
        let (lower, upper) = var.cardinality_bounds();
        let card = set.cardinality().unwrap_or(usize::MAX) as i64;

        if card < lower {
            return false;
        }

        if let Some(u) = upper
            && card > u
        {
            return false;
        }

        true
    }

    /// Get enumerated set for a variable
    pub fn get_enumerated(&self, var: SetVarId) -> Option<&EnumSet> {
        self.enumerated.get(&var)
    }

    /// Get statistics
    pub fn stats(&self) -> &SetEnumStats {
        &self.stats
    }

    /// Reset the enumerator
    pub fn reset(&mut self) {
        self.enumerated.clear();
        self.stats = SetEnumStats::default();
    }
}

/// Model generator for set constraints
#[derive(Debug)]
pub struct SetModelGenerator {
    /// Enumerator
    enumerator: FiniteSetEnumerator,
    /// Variable assignments
    assignments: FxHashMap<SetVarId, EnumSet>,
}

impl SetModelGenerator {
    /// Create a new model generator
    pub fn new(config: SetEnumConfig) -> Self {
        Self {
            enumerator: FiniteSetEnumerator::new(config),
            assignments: FxHashMap::default(),
        }
    }

    /// Generate a model for the given variables
    pub fn generate_model(
        &mut self,
        vars: &[SetVar],
    ) -> SetEnumResult<FxHashMap<SetVarId, EnumSet>> {
        self.assignments.clear();

        for var in vars {
            let enumerated = self.enumerator.enumerate(var)?;
            self.assignments.insert(var.id, enumerated);
        }

        Ok(self.assignments.clone())
    }

    /// Verify that a model satisfies all constraints
    #[allow(dead_code)]
    pub fn verify_model(&self, model: &FxHashMap<SetVarId, EnumSet>, vars: &[SetVar]) -> bool {
        for var in vars {
            if let Some(set) = model.get(&var.id) {
                if !self.enumerator.verify(set, var) {
                    return false;
                }
            } else {
                return false; // Missing assignment
            }
        }

        true
    }

    /// Get the enumerator
    #[allow(dead_code)]
    pub fn enumerator(&self) -> &FiniteSetEnumerator {
        &self.enumerator
    }

    /// Get mutable enumerator
    #[allow(dead_code)]
    pub fn enumerator_mut(&mut self) -> &mut FiniteSetEnumerator {
        &mut self.enumerator
    }
}

#[cfg(test)]
mod tests {
    use super::super::SetSort;
    use super::*;

    #[test]
    fn test_enum_set_empty() {
        let set = EnumSet::empty();
        assert!(set.is_empty());
        assert_eq!(set.cardinality(), Some(0));
    }

    #[test]
    fn test_enum_set_singleton() {
        let set = EnumSet::singleton(42);
        assert!(!set.is_empty());
        assert_eq!(set.cardinality(), Some(1));
        assert!(set.contains(42));
    }

    #[test]
    fn test_enum_set_union() {
        let set1 = EnumSet::singleton(1);
        let set2 = EnumSet::singleton(2);

        let union = set1.union(&set2);
        assert_eq!(union.cardinality(), Some(2));
        assert!(union.contains(1));
        assert!(union.contains(2));
    }

    #[test]
    fn test_enum_set_intersection() {
        let mut elements1 = FxHashSet::default();
        elements1.insert(1);
        elements1.insert(2);
        elements1.insert(3);
        let set1 = EnumSet::from_elements(elements1);

        let mut elements2 = FxHashSet::default();
        elements2.insert(2);
        elements2.insert(3);
        elements2.insert(4);
        let set2 = EnumSet::from_elements(elements2);

        let intersection = set1.intersection(&set2);
        assert_eq!(intersection.cardinality(), Some(2));
        assert!(intersection.contains(2));
        assert!(intersection.contains(3));
    }

    #[test]
    fn test_enum_set_difference() {
        let mut elements1 = FxHashSet::default();
        elements1.insert(1);
        elements1.insert(2);
        elements1.insert(3);
        let set1 = EnumSet::from_elements(elements1);

        let mut elements2 = FxHashSet::default();
        elements2.insert(2);
        elements2.insert(3);
        let set2 = EnumSet::from_elements(elements2);

        let difference = set1.difference(&set2);
        assert_eq!(difference.cardinality(), Some(1));
        assert!(difference.contains(1));
    }

    #[test]
    fn test_enum_set_subset() {
        let set1 = EnumSet::singleton(1);

        let mut elements2 = FxHashSet::default();
        elements2.insert(1);
        elements2.insert(2);
        let set2 = EnumSet::from_elements(elements2);

        assert!(set1.is_subset(&set2));
        assert!(!set2.is_subset(&set1));
    }

    #[test]
    fn test_finite_set_enumerator() {
        let config = SetEnumConfig::default();
        let mut enumerator = FiniteSetEnumerator::new(config);

        let var = SetVar::new(SetVarId(0), "S".to_string(), SetSort::IntSet, 0);
        let result = enumerator.enumerate(&var);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enumerate_with_must_members() {
        let config = SetEnumConfig::default();
        let mut enumerator = FiniteSetEnumerator::new(config);

        let mut var = SetVar::new(SetVarId(0), "S".to_string(), SetSort::IntSet, 0);
        var.add_must_member(1);
        var.add_must_member(2);

        let result = enumerator.enumerate(&var).unwrap();
        assert!(result.contains(1));
        assert!(result.contains(2));
    }

    #[test]
    fn test_enumerate_with_cardinality() {
        let config = SetEnumConfig::default();
        let mut enumerator = FiniteSetEnumerator::new(config);

        let mut var = SetVar::new(SetVarId(0), "S".to_string(), SetSort::IntSet, 0);
        var.add_must_member(1);
        var.tighten_lower_card(3);
        var.tighten_upper_card(3);

        let mut may = FxHashSet::default();
        may.insert(1);
        may.insert(2);
        may.insert(3);
        may.insert(4);
        var.may_members = Some(may);

        let result = enumerator.enumerate(&var).unwrap();
        assert_eq!(result.cardinality(), Some(3));
        assert!(result.contains(1));
    }

    #[test]
    fn test_enumerate_all() {
        let config = SetEnumConfig::default();
        let mut enumerator = FiniteSetEnumerator::new(config);

        let mut var = SetVar::new(SetVarId(0), "S".to_string(), SetSort::IntSet, 0);

        let mut may = FxHashSet::default();
        may.insert(1);
        may.insert(2);
        var.may_members = Some(may);

        var.tighten_lower_card(0);
        var.tighten_upper_card(2);

        let results = enumerator.enumerate_all(&var).unwrap();
        // Should generate: {}, {1}, {2}, {1,2}
        assert!(results.len() <= 4);
    }

    #[test]
    fn test_verify() {
        let config = SetEnumConfig::default();
        let enumerator = FiniteSetEnumerator::new(config);

        let mut var = SetVar::new(SetVarId(0), "S".to_string(), SetSort::IntSet, 0);
        var.add_must_member(1);
        var.add_must_not_member(3);

        let mut set = EnumSet::empty();
        set.insert(1);
        set.insert(2);

        assert!(enumerator.verify(&set, &var));

        set.insert(3);
        assert!(!enumerator.verify(&set, &var)); // Contains must_not_member
    }

    #[test]
    fn test_model_generator() {
        let config = SetEnumConfig::default();
        let mut generator = SetModelGenerator::new(config);

        let vars = vec![
            SetVar::new(SetVarId(0), "S1".to_string(), SetSort::IntSet, 0),
            SetVar::new(SetVarId(1), "S2".to_string(), SetSort::IntSet, 0),
        ];

        let model = generator.generate_model(&vars);
        assert!(model.is_ok());

        let model_map = model.unwrap();
        assert_eq!(model_map.len(), 2);
    }

    #[test]
    fn test_enum_set_from_range() {
        let set = EnumSet::from_range(0, 5);
        assert_eq!(set.cardinality(), Some(5));
        assert!(set.contains(0));
        assert!(set.contains(4));
        assert!(!set.contains(5));
    }

    #[test]
    fn test_enum_set_to_sorted_vec() {
        let mut elements = FxHashSet::default();
        elements.insert(3);
        elements.insert(1);
        elements.insert(2);
        let set = EnumSet::from_elements(elements);

        let vec = set.to_sorted_vec();
        assert_eq!(vec, vec![1, 2, 3]);
    }
}
