//! Value Management for Model Construction.
//!
//! Manages theory-specific values during model building, handling type
//! conversions, normalization, and consistency checks.
//!
//! ## Value Types
//!
//! - **Boolean**: True/False
//! - **Integer**: Arbitrary precision integers
//! - **Rational**: Arbitrary precision rationals
//! - **BitVector**: Fixed-width bit strings
//! - **Array**: Function from indices to values
//! - **Datatype**: ADT constructor applications
//! - **Uninterpreted**: Abstract values with equality
//!
//! ## References
//!
//! - Z3's `model/model.cpp`
//! - "Model Construction in SMT" (Barrett et al., 2009)

use num_bigint::BigInt;
use num_rational::BigRational;
use rustc_hash::FxHashMap;
use std::sync::Arc;

use crate::{Sort, Term, TermId};

/// A value in a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelValue {
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(BigInt),
    /// Rational value.
    Rational(BigRational),
    /// Bit-vector value (width, value).
    BitVector(usize, BigInt),
    /// Array value (default + explicit mappings).
    Array {
        default: Box<ModelValue>,
        entries: Vec<(ModelValue, ModelValue)>,
    },
    /// Datatype constructor application.
    Datatype {
        constructor: String,
        fields: Vec<ModelValue>,
    },
    /// Uninterpreted value (sort + identifier).
    Uninterpreted(Sort, usize),
    /// Function interpretation (args -> result).
    Function {
        domain: Vec<Sort>,
        codomain: Sort,
        mappings: Vec<(Vec<ModelValue>, ModelValue)>,
        default: Option<Box<ModelValue>>,
    },
}

impl ModelValue {
    /// Check if this is a boolean value.
    pub fn is_bool(&self) -> bool {
        matches!(self, ModelValue::Bool(_))
    }

    /// Try to extract as boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ModelValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Try to extract as integer.
    pub fn as_int(&self) -> Option<&BigInt> {
        match self {
            ModelValue::Int(i) => Some(i),
            _ => None,
        }
    }

    /// Try to extract as rational.
    pub fn as_rational(&self) -> Option<&BigRational> {
        match self {
            ModelValue::Rational(r) => Some(r),
            _ => None,
        }
    }

    /// Check type compatibility with a sort.
    pub fn compatible_with_sort(&self, sort: &Sort) -> bool {
        // Simplified: would do proper type checking
        true
    }
}

/// Configuration for value management.
#[derive(Debug, Clone)]
pub struct ValueManagerConfig {
    /// Enable value normalization.
    pub normalize_values: bool,
    /// Cache converted values.
    pub enable_caching: bool,
    /// Maximum uninterpreted value ID.
    pub max_uninterpreted_id: usize,
}

impl Default for ValueManagerConfig {
    fn default() -> Self {
        Self {
            normalize_values: true,
            enable_caching: true,
            max_uninterpreted_id: 100000,
        }
    }
}

/// Statistics for value management.
#[derive(Debug, Clone, Default)]
pub struct ValueManagerStats {
    /// Values created.
    pub values_created: u64,
    /// Conversions performed.
    pub conversions: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
}

/// Value manager for model construction.
pub struct ValueManager {
    /// Configuration.
    config: ValueManagerConfig,
    /// Statistics.
    stats: ValueManagerStats,
    /// Term to value mapping.
    term_values: FxHashMap<TermId, ModelValue>,
    /// Next uninterpreted value ID per sort.
    next_uninterp_id: FxHashMap<Sort, usize>,
    /// Value cache (for normalization).
    cache: FxHashMap<ModelValue, ModelValue>,
}

impl ValueManager {
    /// Create a new value manager.
    pub fn new(config: ValueManagerConfig) -> Self {
        Self {
            config,
            stats: ValueManagerStats::default(),
            term_values: FxHashMap::default(),
            next_uninterp_id: FxHashMap::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ValueManagerConfig::default())
    }

    /// Create a boolean value.
    pub fn mk_bool(&mut self, value: bool) -> ModelValue {
        self.stats.values_created += 1;
        ModelValue::Bool(value)
    }

    /// Create an integer value.
    pub fn mk_int(&mut self, value: BigInt) -> ModelValue {
        self.stats.values_created += 1;
        let val = ModelValue::Int(value);

        if self.config.normalize_values {
            self.normalize(val)
        } else {
            val
        }
    }

    /// Create a rational value.
    pub fn mk_rational(&mut self, value: BigRational) -> ModelValue {
        self.stats.values_created += 1;
        let val = ModelValue::Rational(value);

        if self.config.normalize_values {
            self.normalize(val)
        } else {
            val
        }
    }

    /// Create a bit-vector value.
    pub fn mk_bitvector(&mut self, width: usize, value: BigInt) -> ModelValue {
        self.stats.values_created += 1;

        // Mask to width
        let mask = (BigInt::from(1) << width) - 1;
        let masked_value = value & mask;

        ModelValue::BitVector(width, masked_value)
    }

    /// Create a fresh uninterpreted value.
    pub fn mk_uninterpreted(&mut self, sort: Sort) -> ModelValue {
        self.stats.values_created += 1;

        let id = *self.next_uninterp_id.entry(sort.clone()).or_insert(0);
        self.next_uninterp_id.insert(sort.clone(), id + 1);

        ModelValue::Uninterpreted(sort, id)
    }

    /// Create an array value.
    pub fn mk_array(
        &mut self,
        default: ModelValue,
        entries: Vec<(ModelValue, ModelValue)>,
    ) -> ModelValue {
        self.stats.values_created += 1;

        ModelValue::Array {
            default: Box::new(default),
            entries,
        }
    }

    /// Create a datatype value.
    pub fn mk_datatype(&mut self, constructor: String, fields: Vec<ModelValue>) -> ModelValue {
        self.stats.values_created += 1;

        ModelValue::Datatype { constructor, fields }
    }

    /// Normalize a value.
    fn normalize(&mut self, value: ModelValue) -> ModelValue {
        if !self.config.enable_caching {
            return value;
        }

        // Check cache
        if let Some(cached) = self.cache.get(&value) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }

        self.stats.cache_misses += 1;

        // Perform normalization based on value type
        let normalized = match &value {
            ModelValue::Rational(r) => {
                // Reduce fraction to lowest terms (already done by BigRational)
                value.clone()
            }
            ModelValue::BitVector(w, v) => {
                // Ensure value fits in width
                let mask = (BigInt::from(1) << w) - 1;
                ModelValue::BitVector(*w, v & mask)
            }
            _ => value.clone(),
        };

        // Cache the result
        self.cache.insert(value, normalized.clone());

        normalized
    }

    /// Assign a value to a term.
    pub fn assign(&mut self, term: TermId, value: ModelValue) {
        self.term_values.insert(term, value);
    }

    /// Get the value assigned to a term.
    pub fn get_value(&self, term: TermId) -> Option<&ModelValue> {
        self.term_values.get(&term)
    }

    /// Convert a value to a term representation.
    pub fn value_to_term(&mut self, value: &ModelValue) -> Option<TermId> {
        self.stats.conversions += 1;

        // Simplified: would construct actual terms
        // For now, return None
        None
    }

    /// Evaluate a term given current assignments.
    pub fn eval(&self, term: TermId) -> Option<ModelValue> {
        // Simplified: would recursively evaluate term
        self.term_values.get(&term).cloned()
    }

    /// Check if a value satisfies type constraints.
    pub fn check_type(&self, value: &ModelValue, sort: &Sort) -> bool {
        value.compatible_with_sort(sort)
    }

    /// Get number of assigned values.
    pub fn num_assignments(&self) -> usize {
        self.term_values.len()
    }

    /// Clear all assignments.
    pub fn clear(&mut self) {
        self.term_values.clear();
        self.cache.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &ValueManagerStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ValueManagerStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = ValueManager::default_config();
        assert_eq!(manager.stats().values_created, 0);
    }

    #[test]
    fn test_create_bool() {
        let mut manager = ValueManager::default_config();

        let val = manager.mk_bool(true);
        assert_eq!(val.as_bool(), Some(true));
        assert_eq!(manager.stats().values_created, 1);
    }

    #[test]
    fn test_create_int() {
        let mut manager = ValueManager::default_config();

        let val = manager.mk_int(BigInt::from(42));
        assert_eq!(val.as_int(), Some(&BigInt::from(42)));
    }

    #[test]
    fn test_create_bitvector() {
        let mut manager = ValueManager::default_config();

        // Create 8-bit value with value 255
        let val = manager.mk_bitvector(8, BigInt::from(255));

        if let ModelValue::BitVector(width, value) = val {
            assert_eq!(width, 8);
            assert_eq!(value, BigInt::from(255));
        } else {
            panic!("expected bitvector");
        }
    }

    #[test]
    fn test_bitvector_masking() {
        let mut manager = ValueManager::default_config();

        // Create 4-bit value with 255 (should be masked to 15)
        let val = manager.mk_bitvector(4, BigInt::from(255));

        if let ModelValue::BitVector(width, value) = val {
            assert_eq!(width, 4);
            assert_eq!(value, BigInt::from(15)); // 0b1111
        } else {
            panic!("expected bitvector");
        }
    }

    #[test]
    fn test_uninterpreted() {
        let mut manager = ValueManager::default_config();

        let sort = Sort::Uninterpreted("A".to_string());
        let val1 = manager.mk_uninterpreted(sort.clone());
        let val2 = manager.mk_uninterpreted(sort.clone());

        // Should get different IDs
        if let (ModelValue::Uninterpreted(_, id1), ModelValue::Uninterpreted(_, id2)) =
            (&val1, &val2)
        {
            assert_ne!(id1, id2);
        } else {
            panic!("expected uninterpreted values");
        }
    }

    #[test]
    fn test_assign_and_get() {
        let mut manager = ValueManager::default_config();

        let term = TermId::new(0);
        let value = manager.mk_bool(true);

        manager.assign(term, value.clone());

        assert_eq!(manager.get_value(term), Some(&value));
        assert_eq!(manager.num_assignments(), 1);
    }

    #[test]
    fn test_clear() {
        let mut manager = ValueManager::default_config();

        let term = TermId::new(0);
        let value = manager.mk_bool(false);

        manager.assign(term, value);
        assert_eq!(manager.num_assignments(), 1);

        manager.clear();
        assert_eq!(manager.num_assignments(), 0);
    }
}
