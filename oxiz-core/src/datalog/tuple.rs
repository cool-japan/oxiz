//! Tuple representation for Datalog relations
//!
//! Tuples are the fundamental data unit in Datalog relations.
//! They support efficient storage, comparison, and hashing.

use lasso::Spur;
use num_rational::BigRational;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::schema::{ColumnId, DataType, Schema};
use crate::ast::TermId;

/// A value that can be stored in a tuple
#[derive(Debug, Clone)]
pub enum Value {
    /// Null value
    Null,
    /// Boolean
    Bool(bool),
    /// 64-bit signed integer
    Int64(i64),
    /// 64-bit unsigned integer
    UInt64(u64),
    /// Rational number
    Rational(BigRational),
    /// Interned symbol
    Symbol(Spur),
    /// Reference to relation
    RelationRef(u64),
    /// SMT term reference
    Term(TermId),
    /// Raw bytes
    Bytes(Arc<[u8]>),
    /// Nested tuple
    Tuple(Box<Tuple>),
}

impl Value {
    /// Check if value is null
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int64(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as u64
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::UInt64(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as rational
    pub fn as_rational(&self) -> Option<&BigRational> {
        match self {
            Value::Rational(r) => Some(r),
            _ => None,
        }
    }

    /// Get as symbol
    pub fn as_symbol(&self) -> Option<Spur> {
        match self {
            Value::Symbol(s) => Some(*s),
            _ => None,
        }
    }

    /// Get as term
    pub fn as_term(&self) -> Option<TermId> {
        match self {
            Value::Term(t) => Some(*t),
            _ => None,
        }
    }

    /// Get as bytes
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bytes(b) => Some(b),
            _ => None,
        }
    }

    /// Check if value matches data type
    pub fn matches_type(&self, dt: &DataType) -> bool {
        match (self, dt) {
            (Value::Null, _) => true, // Null matches any nullable type
            (Value::Bool(_), DataType::Bool) => true,
            (Value::Int64(_), DataType::Int64) => true,
            (Value::UInt64(_), DataType::UInt64) => true,
            (Value::Rational(_), DataType::Rational) => true,
            (Value::Symbol(_), DataType::Symbol) => true,
            (Value::RelationRef(_), DataType::RelationRef) => true,
            (Value::Term(_), DataType::Term) => true,
            (Value::Bytes(_), DataType::Bytes) => true,
            (Value::Tuple(t), DataType::Tuple(types)) => {
                if t.len() != types.len() {
                    return false;
                }
                t.values
                    .iter()
                    .zip(types.iter())
                    .all(|(v, dt)| v.matches_type(dt))
            }
            _ => false,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::UInt64(a), Value::UInt64(b)) => a == b,
            (Value::Rational(a), Value::Rational(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::RelationRef(a), Value::RelationRef(b)) => a == b,
            (Value::Term(a), Value::Term(b)) => a == b,
            (Value::Bytes(a), Value::Bytes(b)) => a == b,
            (Value::Tuple(a), Value::Tuple(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Less,
            (_, Value::Null) => Ordering::Greater,
            (Value::Bool(a), Value::Bool(b)) => a.cmp(b),
            (Value::Int64(a), Value::Int64(b)) => a.cmp(b),
            (Value::UInt64(a), Value::UInt64(b)) => a.cmp(b),
            (Value::Rational(a), Value::Rational(b)) => a.cmp(b),
            (Value::Symbol(a), Value::Symbol(b)) => a.cmp(b),
            (Value::RelationRef(a), Value::RelationRef(b)) => a.cmp(b),
            (Value::Term(a), Value::Term(b)) => a.raw().cmp(&b.raw()),
            (Value::Bytes(a), Value::Bytes(b)) => a.as_ref().cmp(b.as_ref()),
            (Value::Tuple(a), Value::Tuple(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}

impl Hash for Value {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Value::Null => {}
            Value::Bool(b) => b.hash(state),
            Value::Int64(n) => n.hash(state),
            Value::UInt64(n) => n.hash(state),
            Value::Rational(r) => {
                r.numer().hash(state);
                r.denom().hash(state);
            }
            Value::Symbol(s) => s.hash(state),
            Value::RelationRef(r) => r.hash(state),
            Value::Term(t) => t.raw().hash(state),
            Value::Bytes(b) => b.hash(state),
            Value::Tuple(t) => t.hash(state),
        }
    }
}

/// A tuple in a relation
#[derive(Debug, Clone)]
pub struct Tuple {
    /// Values in column order
    values: Vec<Value>,
}

impl Tuple {
    /// Create a new tuple with given values
    pub fn new(values: Vec<Value>) -> Self {
        Self { values }
    }

    /// Create an empty tuple
    pub fn empty() -> Self {
        Self { values: Vec::new() }
    }

    /// Get the arity (number of values)
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if tuple is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get value at index
    pub fn get(&self, idx: usize) -> Option<&Value> {
        self.values.get(idx)
    }

    /// Get value at column ID
    pub fn get_column(&self, col: ColumnId) -> Option<&Value> {
        self.values.get(col.raw())
    }

    /// Get all values
    pub fn values(&self) -> &[Value] {
        &self.values
    }

    /// Project to a subset of columns
    pub fn project(&self, columns: &[ColumnId]) -> Tuple {
        let values: Vec<Value> = columns
            .iter()
            .filter_map(|col| self.get_column(*col).cloned())
            .collect();
        Tuple::new(values)
    }

    /// Concatenate with another tuple
    pub fn concat(&self, other: &Tuple) -> Tuple {
        let mut values = self.values.clone();
        values.extend(other.values.iter().cloned());
        Tuple::new(values)
    }

    /// Check if tuple matches schema
    pub fn matches_schema(&self, schema: &Schema) -> bool {
        if self.len() != schema.arity() {
            return false;
        }
        self.values.iter().zip(schema.columns()).all(|(val, col)| {
            if val.is_null() {
                col.is_nullable()
            } else {
                val.matches_type(col.data_type())
            }
        })
    }

    /// Convert to owned values
    pub fn into_values(self) -> Vec<Value> {
        self.values
    }
}

impl PartialEq for Tuple {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl Eq for Tuple {}

impl PartialOrd for Tuple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Tuple {
    fn cmp(&self, other: &Self) -> Ordering {
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            match a.cmp(b) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        self.values.len().cmp(&other.values.len())
    }
}

impl Hash for Tuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.values.hash(state);
    }
}

/// Reference to a tuple with lifetime
#[derive(Debug)]
pub struct TupleRef<'a> {
    values: &'a [Value],
}

impl<'a> TupleRef<'a> {
    /// Create from slice
    pub fn from_slice(values: &'a [Value]) -> Self {
        Self { values }
    }

    /// Get value at index
    pub fn get(&self, idx: usize) -> Option<&Value> {
        self.values.get(idx)
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Convert to owned tuple
    pub fn to_owned(&self) -> Tuple {
        Tuple::new(self.values.to_vec())
    }

    /// Get values slice
    pub fn values(&self) -> &[Value] {
        self.values
    }
}

impl<'a> PartialEq for TupleRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl<'a> Eq for TupleRef<'a> {}

impl<'a> Hash for TupleRef<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.values.hash(state);
    }
}

/// Builder for constructing tuples incrementally
#[derive(Debug, Default)]
pub struct TupleBuilder {
    values: Vec<Value>,
}

impl TupleBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self { values: Vec::new() }
    }

    /// Create builder with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
        }
    }

    /// Add a null value
    pub fn push_null(mut self) -> Self {
        self.values.push(Value::Null);
        self
    }

    /// Add a boolean value
    pub fn push_bool(mut self, value: bool) -> Self {
        self.values.push(Value::Bool(value));
        self
    }

    /// Add an i64 value
    pub fn push_i64(mut self, value: i64) -> Self {
        self.values.push(Value::Int64(value));
        self
    }

    /// Add a u64 value
    pub fn push_u64(mut self, value: u64) -> Self {
        self.values.push(Value::UInt64(value));
        self
    }

    /// Add a rational value
    pub fn push_rational(mut self, value: BigRational) -> Self {
        self.values.push(Value::Rational(value));
        self
    }

    /// Add a symbol value
    pub fn push_symbol(mut self, value: Spur) -> Self {
        self.values.push(Value::Symbol(value));
        self
    }

    /// Add a term value
    pub fn push_term(mut self, value: TermId) -> Self {
        self.values.push(Value::Term(value));
        self
    }

    /// Add bytes value
    pub fn push_bytes(mut self, value: impl Into<Arc<[u8]>>) -> Self {
        self.values.push(Value::Bytes(value.into()));
        self
    }

    /// Add any value
    pub fn push(mut self, value: Value) -> Self {
        self.values.push(value);
        self
    }

    /// Build the tuple
    pub fn build(self) -> Tuple {
        Tuple::new(self.values)
    }

    /// Current length
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Iterator over tuple values
pub struct TupleIter<'a> {
    values: std::slice::Iter<'a, Value>,
}

impl<'a> Iterator for TupleIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<Self::Item> {
        self.values.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.values.size_hint()
    }
}

impl<'a> IntoIterator for &'a Tuple {
    type Item = &'a Value;
    type IntoIter = TupleIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TupleIter {
            values: self.values.iter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tuple_creation() {
        let t = Tuple::new(vec![Value::Int64(1), Value::Bool(true)]);
        assert_eq!(t.len(), 2);
        assert_eq!(t.get(0), Some(&Value::Int64(1)));
        assert_eq!(t.get(1), Some(&Value::Bool(true)));
    }

    #[test]
    fn test_tuple_builder() {
        let t = TupleBuilder::new()
            .push_i64(42)
            .push_bool(false)
            .push_null()
            .build();

        assert_eq!(t.len(), 3);
        assert_eq!(t.get(0).and_then(|v| v.as_i64()), Some(42));
        assert_eq!(t.get(1).and_then(|v| v.as_bool()), Some(false));
        assert!(t.get(2).map(|v| v.is_null()).unwrap_or(false));
    }

    #[test]
    fn test_tuple_equality() {
        let t1 = TupleBuilder::new().push_i64(1).push_i64(2).build();
        let t2 = TupleBuilder::new().push_i64(1).push_i64(2).build();
        let t3 = TupleBuilder::new().push_i64(1).push_i64(3).build();

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);
    }

    #[test]
    fn test_tuple_ordering() {
        let t1 = TupleBuilder::new().push_i64(1).push_i64(2).build();
        let t2 = TupleBuilder::new().push_i64(1).push_i64(3).build();
        let t3 = TupleBuilder::new().push_i64(2).push_i64(1).build();

        assert!(t1 < t2);
        assert!(t2 < t3);
    }

    #[test]
    fn test_tuple_projection() {
        let t = TupleBuilder::new()
            .push_i64(1)
            .push_i64(2)
            .push_i64(3)
            .build();

        let projected = t.project(&[ColumnId::new(0), ColumnId::new(2)]);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected.get(0), Some(&Value::Int64(1)));
        assert_eq!(projected.get(1), Some(&Value::Int64(3)));
    }

    #[test]
    fn test_tuple_concat() {
        let t1 = TupleBuilder::new().push_i64(1).build();
        let t2 = TupleBuilder::new().push_i64(2).push_i64(3).build();
        let joined = t1.concat(&t2);

        assert_eq!(joined.len(), 3);
        assert_eq!(joined.get(0), Some(&Value::Int64(1)));
        assert_eq!(joined.get(1), Some(&Value::Int64(2)));
        assert_eq!(joined.get(2), Some(&Value::Int64(3)));
    }

    #[test]
    fn test_value_type_matching() {
        assert!(Value::Int64(42).matches_type(&DataType::Int64));
        assert!(Value::Bool(true).matches_type(&DataType::Bool));
        assert!(!Value::Int64(42).matches_type(&DataType::Bool));
        assert!(Value::Null.matches_type(&DataType::Int64));
    }

    #[test]
    fn test_tuple_hash() {
        use std::collections::HashSet;

        let t1 = TupleBuilder::new().push_i64(1).push_i64(2).build();
        let t2 = TupleBuilder::new().push_i64(1).push_i64(2).build();
        let t3 = TupleBuilder::new().push_i64(1).push_i64(3).build();

        let mut set = HashSet::new();
        set.insert(t1.clone());
        set.insert(t2);
        set.insert(t3);

        assert_eq!(set.len(), 2); // t1 and t2 are equal
    }

    #[test]
    fn test_tuple_iter() {
        let t = TupleBuilder::new()
            .push_i64(1)
            .push_i64(2)
            .push_i64(3)
            .build();

        let values: Vec<_> = t.into_iter().collect();
        assert_eq!(values.len(), 3);
    }
}
