//! Value Factory
//!
//! Creates default values for different sorts.

use super::Value;
use crate::sort::SortId;
use num_rational::Rational64;
use std::collections::HashMap;

/// Configuration for value factory
#[derive(Debug, Clone)]
pub struct ValueFactoryConfig {
    /// Default bitvector width
    pub default_bv_width: u32,
    /// Default string value
    pub default_string: String,
    /// Use zero for numerics
    pub zero_numerics: bool,
}

impl Default for ValueFactoryConfig {
    fn default() -> Self {
        Self {
            default_bv_width: 32,
            default_string: String::new(),
            zero_numerics: true,
        }
    }
}

/// Factory for creating default values
#[derive(Debug)]
pub struct ValueFactory {
    config: ValueFactoryConfig,
    /// Uninterpreted sort counters
    uninterpreted_counters: HashMap<SortId, u64>,
    /// Custom default values by sort
    custom_defaults: HashMap<SortId, Value>,
}

impl ValueFactory {
    /// Create a new value factory
    pub fn new() -> Self {
        Self {
            config: ValueFactoryConfig::default(),
            uninterpreted_counters: HashMap::new(),
            custom_defaults: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ValueFactoryConfig) -> Self {
        Self {
            config,
            uninterpreted_counters: HashMap::new(),
            custom_defaults: HashMap::new(),
        }
    }

    /// Get default value for a sort
    pub fn default_value(&mut self, sort: SortId) -> Value {
        // Check custom defaults first
        if let Some(v) = self.custom_defaults.get(&sort).cloned() {
            return v;
        }

        // Built-in sorts (by convention: 0=Bool, 1=Int, 2=Real, 3=String, 4=RegLan, 5+=BV/Array/etc)
        match sort.0 {
            0 => Value::Bool(false),
            1 => {
                if self.config.zero_numerics {
                    Value::Int(0)
                } else {
                    Value::Int(1)
                }
            }
            2 => {
                if self.config.zero_numerics {
                    Value::Rational(Rational64::from_integer(0))
                } else {
                    Value::Rational(Rational64::from_integer(1))
                }
            }
            3 => Value::String(self.config.default_string.clone()),
            4 => Value::Undefined, // RegLan
            _ => self.uninterpreted_value(sort),
        }
    }

    /// Create default boolean value
    pub fn default_bool(&self) -> Value {
        Value::Bool(false)
    }

    /// Create default integer value
    pub fn default_int(&self) -> Value {
        if self.config.zero_numerics {
            Value::Int(0)
        } else {
            Value::Int(1)
        }
    }

    /// Create default rational value
    pub fn default_rational(&self) -> Value {
        if self.config.zero_numerics {
            Value::Rational(Rational64::from_integer(0))
        } else {
            Value::Rational(Rational64::from_integer(1))
        }
    }

    /// Create default bitvector value
    pub fn default_bitvec(&self, width: u32) -> Value {
        Value::BitVec(width, 0)
    }

    /// Create default string value
    pub fn default_string(&self) -> Value {
        Value::String(self.config.default_string.clone())
    }

    /// Create default array value
    pub fn default_array(&mut self, element_sort: SortId) -> Value {
        let default = self.default_value(element_sort);
        Value::Array(Box::new(default), Vec::new())
    }

    /// Create a fresh uninterpreted value
    pub fn uninterpreted_value(&mut self, sort: SortId) -> Value {
        let counter = self.uninterpreted_counters.entry(sort).or_insert(0);
        let id = *counter;
        *counter += 1;
        Value::Uninterpreted(id)
    }

    /// Set custom default value for a sort
    pub fn set_custom_default(&mut self, sort: SortId, value: Value) {
        self.custom_defaults.insert(sort, value);
    }

    /// Remove custom default value
    pub fn remove_custom_default(&mut self, sort: SortId) {
        self.custom_defaults.remove(&sort);
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        self.uninterpreted_counters.clear();
    }

    /// Get current counter for a sort
    pub fn get_counter(&self, sort: SortId) -> u64 {
        self.uninterpreted_counters.get(&sort).copied().unwrap_or(0)
    }
}

impl Default for ValueFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factory_creation() {
        let factory = ValueFactory::new();
        assert_eq!(factory.config.default_bv_width, 32);
        assert!(factory.config.zero_numerics);
    }

    #[test]
    fn test_default_bool() {
        let factory = ValueFactory::new();
        assert_eq!(factory.default_bool(), Value::Bool(false));
    }

    #[test]
    fn test_default_int() {
        let factory = ValueFactory::new();
        assert_eq!(factory.default_int(), Value::Int(0));
    }

    #[test]
    fn test_default_rational() {
        let factory = ValueFactory::new();
        assert_eq!(
            factory.default_rational(),
            Value::Rational(Rational64::from_integer(0))
        );
    }

    #[test]
    fn test_default_bitvec() {
        let factory = ValueFactory::new();
        assert_eq!(factory.default_bitvec(8), Value::BitVec(8, 0));
        assert_eq!(factory.default_bitvec(32), Value::BitVec(32, 0));
    }

    #[test]
    fn test_default_string() {
        let factory = ValueFactory::new();
        assert_eq!(factory.default_string(), Value::String(String::new()));
    }

    #[test]
    fn test_uninterpreted_values() {
        let mut factory = ValueFactory::new();
        let sort = SortId(100);

        let v1 = factory.uninterpreted_value(sort);
        let v2 = factory.uninterpreted_value(sort);
        let v3 = factory.uninterpreted_value(sort);

        assert_eq!(v1, Value::Uninterpreted(0));
        assert_eq!(v2, Value::Uninterpreted(1));
        assert_eq!(v3, Value::Uninterpreted(2));
        assert_eq!(factory.get_counter(sort), 3);
    }

    #[test]
    fn test_custom_default() {
        let mut factory = ValueFactory::new();
        let sort = SortId(100);

        factory.set_custom_default(sort, Value::Int(42));
        assert_eq!(factory.default_value(sort), Value::Int(42));

        factory.remove_custom_default(sort);
        // Should fall back to uninterpreted
        assert!(matches!(
            factory.default_value(sort),
            Value::Uninterpreted(_)
        ));
    }

    #[test]
    fn test_reset() {
        let mut factory = ValueFactory::new();
        let sort = SortId(100);

        factory.uninterpreted_value(sort);
        factory.uninterpreted_value(sort);
        assert_eq!(factory.get_counter(sort), 2);

        factory.reset();
        assert_eq!(factory.get_counter(sort), 0);
    }

    #[test]
    fn test_default_array() {
        let mut factory = ValueFactory::new();
        let int_sort = SortId(1);

        let arr = factory.default_array(int_sort);
        match arr {
            Value::Array(default, exceptions) => {
                assert_eq!(*default, Value::Int(0));
                assert!(exceptions.is_empty());
            }
            _ => panic!("Expected array value"),
        }
    }
}
