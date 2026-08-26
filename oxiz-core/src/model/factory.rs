//! Value Factory
//!
//! Creates default values for different sorts.

use super::Value;
use crate::prelude::HashMap;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortKind, SortManager};
use num_rational::Rational64;

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

    /// Get the default value for `sort`, or `None` if `sort` cannot be
    /// soundly defaulted.
    ///
    /// Dispatches on `sort`'s actual [`SortKind`] (looked up in `sorts`),
    /// never on the raw [`SortId`] integer. `SortManager` only guarantees
    /// fixed ids for the sorts it interns eagerly in `new()` — `Bool` (0),
    /// `Int` (1), `Real` (2) and `RoundingMode` (3). Every other sort,
    /// including `String` and the reserved `RegLan` sort, is interned lazily
    /// in whatever order the input happens to request it, so a caller that
    /// matched on `sort.0 == 3` / `sort.0 == 4` was really matching "whatever
    /// sort got interned 4th / 5th", which can be a `BitVec` or `Array` sort
    /// just as easily as `String`. That handed a term a value of the *wrong
    /// sort* (e.g. `Value::String("")` for a `BitVec`-sorted term) rather
    /// than erroring — silently wrong, not absent.
    ///
    /// Returns `None` (rather than a guessed value) for a sort this factory
    /// cannot soundly default: an unresolved sort parameter, a datatype sort
    /// (picking a base-case constructor and synthesizing its arguments is
    /// out of scope here — see [`crate::model::Value::Datatype`]'s doc), or
    /// a `SortId` not present in `sorts` at all.
    ///
    /// The array case is unrolled with a loop rather than recursion. `None`
    /// is a "cannot default this sort" answer, not an error channel a depth
    /// cap could report through, so a cap here could only ever return a
    /// *wrong* default value. Array-sort nesting is bounded at 512 when it
    /// comes from SMT-LIB text, but `SortManager::array` is `pub` and interns
    /// in constant stack, so an embedder can build an arbitrarily deep sort
    /// and hand it straight to this function.
    pub fn default_value(&mut self, sort: SortId, sorts: &SortManager) -> Option<Value> {
        // Descend the chain of array *range* sorts, then wrap the innermost
        // default back up once per level. A custom default is consulted at
        // every level, exactly as the recursive version did — it short-circuits
        // the descent wherever it is registered.
        let mut array_levels = 0usize;
        let mut current = sort;
        let leaf = loop {
            if let Some(v) = self.custom_defaults.get(&current).cloned() {
                break v;
            }

            // Clone the `SortKind` so `sorts` is no longer borrowed while
            // `self` is used mutably below.
            let kind = sorts.get(current)?.kind.clone();
            match kind {
                SortKind::Bool => break Value::Bool(false),
                SortKind::Int => {
                    break if self.config.zero_numerics {
                        Value::Int(0)
                    } else {
                        Value::Int(1)
                    };
                }
                SortKind::Real => {
                    break if self.config.zero_numerics {
                        Value::Rational(Rational64::from_integer(0))
                    } else {
                        Value::Rational(Rational64::from_integer(1))
                    };
                }
                SortKind::String => break Value::String(self.config.default_string.clone()),
                SortKind::BitVec(width) => break Value::BitVec(width, 0),
                // Positive zero: sign bit clear, all-zero exponent and mantissa.
                SortKind::FloatingPoint { .. } => break Value::FloatingPoint(false, 0, 0),
                // The reserved five-element `RoundingMode` sort. Unlike the
                // uninterpreted arm below it needs no fresh domain element:
                // it has five *named* inhabitants, and IEEE 754's own default
                // mode is round-to-nearest-ties-to-even. Letting it fall into
                // the uninterpreted arm would mint a `Value::Uninterpreted`
                // witness that no `RNE`/`RTZ`/... comparison could ever match.
                SortKind::RoundingMode => {
                    break Value::RoundingMode(crate::ast::RoundingMode::RNE);
                }
                // The default array is the constant array over the range sort's
                // own default, with no stored exceptions. Propagates `None` if
                // the range sort itself cannot be defaulted.
                SortKind::Array { range, .. } => {
                    array_levels += 1;
                    current = range;
                }
                // A declared uninterpreted sort, or an opaque parametric sort
                // application (e.g. `(List Int)`, whose constructors this sort
                // system does not track): both denote *some* domain, so a fresh,
                // per-`SortId` element is as sound a default as `Value` can give
                // without a concrete representation for either. This also covers
                // the reserved `RegLan` sort (modelled as `Uninterpreted("RegLan")`
                // — see `TermManager::mk_regex_op`), which gets the same
                // treatment as any other uninterpreted sort rather than a
                // hardcoded id check.
                SortKind::Uninterpreted(_) | SortKind::Parametric { .. } => {
                    break self.uninterpreted_value(current);
                }
                // A raw sort parameter (e.g. free `T` in a `define-sort` body
                // before instantiation) does not denote a concrete type, and a
                // datatype sort needs a base-case constructor plus synthesized
                // selector arguments to default soundly — neither is something
                // this factory can fabricate without risking a wrong-looking
                // value, so both are honestly "cannot default" rather than a
                // guess.
                SortKind::Parameter(_) | SortKind::Datatype(_) => return None,
            }
        };

        let mut value = leaf;
        for _ in 0..array_levels {
            value = Value::Array(Box::new(value), Vec::new());
        }
        Some(value)
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

    /// Create a default array value over `element_sort`, or `None` if
    /// `element_sort` cannot be soundly defaulted (see [`Self::default_value`]).
    pub fn default_array(&mut self, element_sort: SortId, sorts: &SortManager) -> Option<Value> {
        let default = self.default_value(element_sort, sorts)?;
        Some(Value::Array(Box::new(default), Vec::new()))
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
    use crate::ast::TermManager;

    /// `default_value` unrolls nested array sorts with a loop. It returns
    /// `Option`, whose `None` means "cannot default this sort" — not an error
    /// channel a depth cap could report through — so recursing here could only
    /// abort the process, and `SortManager::array` is `pub` and interns in
    /// constant stack, so nothing bounds the depth an embedder can build.
    ///
    /// Runs on a 128 KiB stack; the assertion is that the call returns.
    #[test]
    fn default_value_survives_a_deeply_nested_array_sort() {
        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                let mut manager = TermManager::new();
                let int_sort = manager.sorts.int_sort;
                let mut sort = int_sort;
                for _ in 0..12_500 {
                    sort = manager.sorts.array(int_sort, sort);
                }
                let mut factory = ValueFactory::new();
                let value = factory.default_value(sort, &manager.sorts);
                // Count the array levels without recursing either.
                let mut levels = 0usize;
                let mut node = value.as_ref();
                while let Some(Value::Array(default, _)) = node {
                    levels += 1;
                    node = Some(default.as_ref());
                }
                (levels, matches!(node, Some(Value::Int(0))))
            })
            .expect("spawn");
        let (levels, innermost_is_int_zero) =
            handle.join().expect("worker thread must not overflow");
        assert_eq!(levels, 12_500);
        assert!(innermost_is_int_zero);
    }

    /// Semantic pin: a shallow array sort defaults exactly as the recursive
    /// version did, and a custom default still short-circuits at every level.
    #[test]
    fn default_value_matches_the_recursive_behaviour_for_shallow_sorts() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bool_sort = manager.sorts.bool_sort;
        let inner = manager.sorts.array(int_sort, bool_sort);
        let outer = manager.sorts.array(bool_sort, inner);

        let mut factory = ValueFactory::new();
        assert_eq!(
            factory.default_value(outer, &manager.sorts),
            Some(Value::Array(
                Box::new(Value::Array(Box::new(Value::Bool(false)), Vec::new())),
                Vec::new()
            ))
        );

        // A custom default registered for the *inner* array sort replaces
        // that whole level, exactly as it did when the recursion consulted
        // `custom_defaults` on entry to each call.
        let mut factory = ValueFactory::new();
        factory.set_custom_default(inner, Value::Int(42));
        assert_eq!(
            factory.default_value(outer, &manager.sorts),
            Some(Value::Array(Box::new(Value::Int(42)), Vec::new()))
        );
    }

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
        let manager = SortManager::new();
        let mut factory = ValueFactory::new();
        let sort = manager.int_sort;

        factory.set_custom_default(sort, Value::Int(42));
        assert_eq!(factory.default_value(sort, &manager), Some(Value::Int(42)));

        factory.remove_custom_default(sort);
        // Falls back to the sort's real default (Int -> 0), not the custom one.
        assert_eq!(factory.default_value(sort, &manager), Some(Value::Int(0)));
    }

    #[test]
    fn test_default_value_none_for_unregistered_sort_id() {
        // A `SortId` that was never interned in `manager` at all: there is
        // genuinely nothing to default, so this must be `None`, not a guess.
        let manager = SortManager::new();
        let mut factory = ValueFactory::new();
        assert_eq!(factory.default_value(SortId(9_999), &manager), None);
    }

    #[test]
    fn test_default_value_for_declared_uninterpreted_sort_mints_fresh_values() {
        let mut manager = SortManager::new();
        let spur = manager.intern_str("MyUninterpretedSort");
        let sort = manager.intern(SortKind::Uninterpreted(spur));

        let mut factory = ValueFactory::new();
        assert_eq!(
            factory.default_value(sort, &manager),
            Some(Value::Uninterpreted(0))
        );
        assert_eq!(
            factory.default_value(sort, &manager),
            Some(Value::Uninterpreted(1))
        );
    }

    #[test]
    fn test_default_value_for_float_sort() {
        let mut manager = SortManager::new();
        let f64_sort = manager.float64_sort();
        let mut factory = ValueFactory::new();
        assert_eq!(
            factory.default_value(f64_sort, &manager),
            Some(Value::FloatingPoint(false, 0, 0))
        );
    }

    #[test]
    fn test_default_value_none_for_datatype_and_parameter_sorts() {
        let mut manager = SortManager::new();
        let color = crate::sort::DataTypeConstructor {
            name: manager.intern_str("red"),
            selectors: smallvec::SmallVec::new(),
        };
        manager.declare_datatype("Color", vec![color]);
        let color_sort = manager.mk_datatype_sort("Color");
        let param_sort = manager.mk_sort_parameter("T");

        let mut factory = ValueFactory::new();
        assert_eq!(factory.default_value(color_sort, &manager), None);
        assert_eq!(factory.default_value(param_sort, &manager), None);
    }

    /// Regression test for: `ValueFactory::default_value` used to dispatch on
    /// the raw `SortId` integer with hardcoded `3 == String`, `4 == RegLan`,
    /// but `SortManager` only guarantees fixed ids for the sorts it interns
    /// eagerly in `new()` — every other sort is interned in whatever order the
    /// caller asks for it.
    ///
    /// The eagerly-interned set is `Bool`/`Int`/`Real`/`RoundingMode`
    /// (0/1/2/3), so the two user-interned sorts below now land on 4 and 5
    /// rather than the 3 and 4 of the original report. The hazard is
    /// unchanged and is in fact demonstrated twice over here: raw id 3, which
    /// the old dispatch called `String`, is now `RoundingMode` and defaults to
    /// a rounding mode; and the ids the old dispatch called `String` and
    /// `RegLan` are now held by sorts of entirely different kinds again.
    /// Whichever way the interning order moves, dispatch must read the
    /// `SortKind`.
    #[test]
    fn test_default_value_dispatches_on_sort_kind_not_raw_sort_id() {
        let mut manager = SortManager::new();
        let bv64 = manager.bitvec(64); // 5th interned sort -> raw id 4
        let arr = manager.array(manager.int_sort, manager.bool_sort); // 6th -> raw id 5
        assert_eq!(bv64.raw(), 4);
        assert_eq!(arr.raw(), 5);

        let mut factory = ValueFactory::new();

        // Raw id 3 — hardcoded to `String` by the old dispatch — is the
        // eagerly-interned `RoundingMode` sort, whose default is a mode.
        assert_eq!(manager.rounding_mode_sort.raw(), 3);
        assert_eq!(
            factory.default_value(manager.rounding_mode_sort, &manager),
            Some(Value::RoundingMode(crate::ast::RoundingMode::RNE))
        );

        // The old magic-integer dispatch handed this `Value::String("")` --
        // silently the wrong *sort* of value -- because `sort.0 == 3`.
        assert_eq!(
            factory.default_value(bv64, &manager),
            Some(Value::BitVec(64, 0))
        );

        // ...and this `Value::Undefined`, because `sort.0 == 4`, rather than
        // a usable Array default. Matched by reference: `Value` implements
        // `Drop` (see `model::Value`), so its fields cannot be moved out of.
        match &factory.default_value(arr, &manager) {
            Some(Value::Array(default, exceptions)) => {
                assert_eq!(**default, Value::Bool(false));
                assert!(exceptions.is_empty());
            }
            other => panic!("expected Some(Value::Array(Bool(false), [])), got {other:?}"),
        }
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
        let manager = SortManager::new();
        let mut factory = ValueFactory::new();
        let int_sort = manager.int_sort;

        let arr = factory
            .default_array(int_sort, &manager)
            .expect("Int has a default");
        // Matched by reference: `Value` implements `Drop` (its structural
        // traits are iterative, see `model::Value`), so its fields cannot be
        // moved out of.
        match &arr {
            Value::Array(default, exceptions) => {
                assert_eq!(**default, Value::Int(0));
                assert!(exceptions.is_empty());
            }
            _ => panic!("Expected array value"),
        }
    }

    #[test]
    fn test_default_array_none_when_element_sort_cannot_be_defaulted() {
        let mut manager = SortManager::new();
        let param_sort = manager.mk_sort_parameter("T");
        let mut factory = ValueFactory::new();
        assert_eq!(factory.default_array(param_sort, &manager), None);
    }
}
