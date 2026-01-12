//! Model Subsystem
//!
//! Provides model construction, evaluation, and manipulation for SMT solving.
//!
//! # Components
//!
//! - **Evaluator**: Evaluates terms under a given model assignment
//! - **Completion**: Completes partial models with default values
//! - **Implicant**: Extracts minimal satisfying assignments (prime implicants)
//! - **Factory**: Creates default values for different sorts
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::model::{Model, ModelEvaluator};
//!
//! let mut model = Model::new();
//! model.assign(x, Value::Int(42));
//! model.assign(y, Value::Bool(true));
//!
//! let evaluator = ModelEvaluator::new(&model);
//! let result = evaluator.eval(expr)?;
//! ```

mod completion;
mod evaluator;
mod factory;
mod implicant;

pub use completion::{ModelCompletion, ModelCompletionConfig};
pub use evaluator::{EvalCache, EvalResult, ModelEvaluator};
pub use factory::{ValueFactory, ValueFactoryConfig};
pub use implicant::{ImplicantConfig, ImplicantExtractor, PrimeImplicant};

use crate::ast::TermId;
use crate::sort::SortId;
use num_rational::Rational64;
use std::collections::HashMap;

/// A value in the model
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Rational value
    Rational(Rational64),
    /// Bitvector value (width, value)
    BitVec(u32, u64),
    /// String value
    String(String),
    /// Array value (default, exceptions)
    Array(Box<Value>, Vec<(Value, Value)>),
    /// Datatype constructor (constructor id, arguments)
    Datatype(u32, Vec<Value>),
    /// Floating-point value (sign, exponent, mantissa)
    FloatingPoint(bool, u64, u64),
    /// Uninterpreted value (id for unique representation)
    Uninterpreted(u64),
    /// Undefined (no assignment)
    Undefined,
}

impl Value {
    /// Check if this is a boolean value
    pub fn is_bool(&self) -> bool {
        matches!(self, Value::Bool(_))
    }

    /// Check if this is an integer value
    pub fn is_int(&self) -> bool {
        matches!(self, Value::Int(_))
    }

    /// Check if this is a rational value
    pub fn is_rational(&self) -> bool {
        matches!(self, Value::Rational(_))
    }

    /// Check if this is a bitvector value
    pub fn is_bitvec(&self) -> bool {
        matches!(self, Value::BitVec(_, _))
    }

    /// Check if this is undefined
    pub fn is_undefined(&self) -> bool {
        matches!(self, Value::Undefined)
    }

    /// Get as boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as rational
    pub fn as_rational(&self) -> Option<Rational64> {
        match self {
            Value::Rational(r) => Some(*r),
            Value::Int(i) => Some(Rational64::from_integer(*i)),
            _ => None,
        }
    }

    /// Get as bitvector (width, value)
    pub fn as_bitvec(&self) -> Option<(u32, u64)> {
        match self {
            Value::BitVec(w, v) => Some((*w, *v)),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Create default value for a sort
    pub fn default_for_sort(sort: SortId) -> Self {
        // Sort IDs: 0=Bool, 1=Int, 2=Real, etc.
        match sort.0 {
            0 => Value::Bool(false),
            1 => Value::Int(0),
            2 => Value::Rational(Rational64::from_integer(0)),
            _ => Value::Undefined,
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Rational(r) => {
                if *r.denom() == 1 {
                    write!(f, "{}", r.numer())
                } else {
                    write!(f, "(/ {} {})", r.numer(), r.denom())
                }
            }
            Value::BitVec(w, v) => write!(f, "#b{:0width$b}", v, width = *w as usize),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Array(def, excs) => {
                if excs.is_empty() {
                    write!(f, "((as const) {})", def)
                } else {
                    write!(f, "(store ... {})", def)
                }
            }
            Value::Datatype(id, args) => {
                if args.is_empty() {
                    write!(f, "C{}", id)
                } else {
                    write!(f, "(C{} ...)", id)
                }
            }
            Value::FloatingPoint(sign, exp, mant) => {
                write!(
                    f,
                    "(fp {} {} {})",
                    if *sign { "#b1" } else { "#b0" },
                    exp,
                    mant
                )
            }
            Value::Uninterpreted(id) => write!(f, "u{}", id),
            Value::Undefined => write!(f, "undefined"),
        }
    }
}

/// A model: assignment of values to terms
#[derive(Debug, Clone, Default)]
pub struct Model {
    /// Term to value assignments
    assignments: HashMap<TermId, Value>,
    /// Sort assignments (for uninterpreted sorts)
    sort_sizes: HashMap<SortId, u64>,
}

impl Model {
    /// Create a new empty model
    pub fn new() -> Self {
        Self::default()
    }

    /// Assign a value to a term
    pub fn assign(&mut self, term: TermId, value: Value) {
        self.assignments.insert(term, value);
    }

    /// Get the value of a term
    pub fn get(&self, term: TermId) -> Option<&Value> {
        self.assignments.get(&term)
    }

    /// Check if a term has an assignment
    pub fn has(&self, term: TermId) -> bool {
        self.assignments.contains_key(&term)
    }

    /// Remove an assignment
    pub fn remove(&mut self, term: TermId) -> Option<Value> {
        self.assignments.remove(&term)
    }

    /// Number of assignments
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Check if model is empty
    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Iterate over assignments
    pub fn iter(&self) -> impl Iterator<Item = (&TermId, &Value)> {
        self.assignments.iter()
    }

    /// Set sort size (for uninterpreted sorts)
    pub fn set_sort_size(&mut self, sort: SortId, size: u64) {
        self.sort_sizes.insert(sort, size);
    }

    /// Get sort size
    pub fn get_sort_size(&self, sort: SortId) -> Option<u64> {
        self.sort_sizes.get(&sort).copied()
    }

    /// Clear all assignments
    pub fn clear(&mut self) {
        self.assignments.clear();
        self.sort_sizes.clear();
    }

    /// Merge another model into this one
    pub fn merge(&mut self, other: &Model) {
        for (term, value) in &other.assignments {
            self.assignments
                .entry(*term)
                .or_insert_with(|| value.clone());
        }
        for (sort, size) in &other.sort_sizes {
            self.sort_sizes.entry(*sort).or_insert(*size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_bool() {
        let v = Value::Bool(true);
        assert!(v.is_bool());
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(format!("{}", v), "true");
    }

    #[test]
    fn test_value_int() {
        let v = Value::Int(42);
        assert!(v.is_int());
        assert_eq!(v.as_int(), Some(42));
        assert_eq!(format!("{}", v), "42");
    }

    #[test]
    fn test_value_rational() {
        let v = Value::Rational(Rational64::new(1, 2));
        assert!(v.is_rational());
        assert_eq!(v.as_rational(), Some(Rational64::new(1, 2)));
        assert_eq!(format!("{}", v), "(/ 1 2)");
    }

    #[test]
    fn test_value_bitvec() {
        let v = Value::BitVec(8, 255);
        assert!(v.is_bitvec());
        assert_eq!(v.as_bitvec(), Some((8, 255)));
        assert_eq!(format!("{}", v), "#b11111111");
    }

    #[test]
    fn test_model_basic() {
        let mut model = Model::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);

        model.assign(t1, Value::Bool(true));
        model.assign(t2, Value::Int(42));

        assert_eq!(model.len(), 2);
        assert!(model.has(t1));
        assert!(model.has(t2));
        assert_eq!(model.get(t1), Some(&Value::Bool(true)));
        assert_eq!(model.get(t2), Some(&Value::Int(42)));
    }

    #[test]
    fn test_model_remove() {
        let mut model = Model::new();
        let t1 = TermId::from(1u32);

        model.assign(t1, Value::Bool(true));
        assert!(model.has(t1));

        model.remove(t1);
        assert!(!model.has(t1));
    }

    #[test]
    fn test_model_merge() {
        let mut m1 = Model::new();
        let mut m2 = Model::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);
        let t3 = TermId::from(3u32);

        m1.assign(t1, Value::Bool(true));
        m1.assign(t2, Value::Int(42));

        m2.assign(t2, Value::Int(100)); // Should not override
        m2.assign(t3, Value::Bool(false));

        m1.merge(&m2);

        assert_eq!(m1.len(), 3);
        assert_eq!(m1.get(t1), Some(&Value::Bool(true)));
        assert_eq!(m1.get(t2), Some(&Value::Int(42))); // Original preserved
        assert_eq!(m1.get(t3), Some(&Value::Bool(false)));
    }

    #[test]
    fn test_value_default_for_sort() {
        assert_eq!(Value::default_for_sort(SortId(0)), Value::Bool(false));
        assert_eq!(Value::default_for_sort(SortId(1)), Value::Int(0));
    }
}
