//! Schema definitions for Datalog relations
//!
//! Provides typed column definitions and schema management.

use lasso::Spur;
use std::collections::HashMap;
use std::fmt;

/// Unique identifier for a column within a schema
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnId(pub usize);

impl ColumnId {
    /// Create a new column ID
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn raw(&self) -> usize {
        self.0
    }
}

/// Data types supported in Datalog relations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataType {
    /// Boolean value
    Bool,
    /// Signed 64-bit integer
    Int64,
    /// Unsigned 64-bit integer
    UInt64,
    /// Rational number (as pair of big integers)
    Rational,
    /// Interned string symbol
    Symbol,
    /// Reference to another relation (for nested queries)
    RelationRef,
    /// SMT term reference
    Term,
    /// Arbitrary bytes
    Bytes,
    /// Tuple type (for compound keys)
    Tuple(Vec<DataType>),
}

impl DataType {
    /// Check if this type is comparable
    pub fn is_comparable(&self) -> bool {
        matches!(
            self,
            DataType::Int64 | DataType::UInt64 | DataType::Rational | DataType::Symbol
        )
    }

    /// Check if this type supports equality
    pub fn supports_equality(&self) -> bool {
        true // All types support equality
    }

    /// Check if this type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::Int64 | DataType::UInt64 | DataType::Rational
        )
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Bool => write!(f, "Bool"),
            DataType::Int64 => write!(f, "Int64"),
            DataType::UInt64 => write!(f, "UInt64"),
            DataType::Rational => write!(f, "Rational"),
            DataType::Symbol => write!(f, "Symbol"),
            DataType::RelationRef => write!(f, "RelationRef"),
            DataType::Term => write!(f, "Term"),
            DataType::Bytes => write!(f, "Bytes"),
            DataType::Tuple(types) => {
                write!(f, "Tuple<")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ">")
            }
        }
    }
}

/// A column in a relation schema
#[derive(Debug, Clone)]
pub struct Column {
    /// Column name (interned)
    name: Spur,
    /// Column data type
    data_type: DataType,
    /// Whether this column allows nulls
    nullable: bool,
    /// Optional default value index
    default_value: Option<usize>,
    /// Column index within schema
    index: ColumnId,
}

impl Column {
    /// Create a new column
    pub fn new(name: Spur, data_type: DataType, index: ColumnId) -> Self {
        Self {
            name,
            data_type,
            nullable: false,
            default_value: None,
            index,
        }
    }

    /// Create a nullable column
    pub fn nullable(name: Spur, data_type: DataType, index: ColumnId) -> Self {
        Self {
            name,
            data_type,
            nullable: true,
            default_value: None,
            index,
        }
    }

    /// Get column name
    pub fn name(&self) -> Spur {
        self.name
    }

    /// Get column data type
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Check if column is nullable
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }

    /// Get column index
    pub fn index(&self) -> ColumnId {
        self.index
    }

    /// Set default value
    pub fn with_default(mut self, default_idx: usize) -> Self {
        self.default_value = Some(default_idx);
        self
    }

    /// Get default value index
    pub fn default_value(&self) -> Option<usize> {
        self.default_value
    }
}

/// Schema for a relation
#[derive(Debug, Clone)]
pub struct Schema {
    /// Schema name
    name: String,
    /// Columns in order
    columns: Vec<Column>,
    /// Column name to index mapping
    name_to_index: HashMap<Spur, ColumnId>,
    /// Primary key column indices
    primary_key: Vec<ColumnId>,
    /// Unique constraint column sets
    unique_constraints: Vec<Vec<ColumnId>>,
    /// Foreign key references
    foreign_keys: Vec<ForeignKey>,
}

/// Foreign key constraint
#[derive(Debug, Clone)]
pub struct ForeignKey {
    /// Columns in this schema
    pub columns: Vec<ColumnId>,
    /// Referenced schema name
    pub ref_schema: String,
    /// Referenced columns
    pub ref_columns: Vec<ColumnId>,
}

impl Schema {
    /// Create a new empty schema
    pub fn new(name: String) -> Self {
        Self {
            name,
            columns: Vec::new(),
            name_to_index: HashMap::new(),
            primary_key: Vec::new(),
            unique_constraints: Vec::new(),
            foreign_keys: Vec::new(),
        }
    }

    /// Get schema name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Add a column to the schema
    pub fn add_column(&mut self, name: Spur, data_type: DataType) -> ColumnId {
        let index = ColumnId::new(self.columns.len());
        let column = Column::new(name, data_type, index);
        self.columns.push(column);
        self.name_to_index.insert(name, index);
        index
    }

    /// Add a nullable column
    pub fn add_nullable_column(&mut self, name: Spur, data_type: DataType) -> ColumnId {
        let index = ColumnId::new(self.columns.len());
        let column = Column::nullable(name, data_type, index);
        self.columns.push(column);
        self.name_to_index.insert(name, index);
        index
    }

    /// Get column by index
    pub fn column(&self, id: ColumnId) -> Option<&Column> {
        self.columns.get(id.0)
    }

    /// Get column by name
    pub fn column_by_name(&self, name: Spur) -> Option<&Column> {
        self.name_to_index
            .get(&name)
            .and_then(|id| self.column(*id))
    }

    /// Get column index by name
    pub fn column_index(&self, name: Spur) -> Option<ColumnId> {
        self.name_to_index.get(&name).copied()
    }

    /// Get all columns
    pub fn columns(&self) -> &[Column] {
        &self.columns
    }

    /// Get number of columns (arity)
    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    /// Set primary key
    pub fn set_primary_key(&mut self, columns: Vec<ColumnId>) {
        self.primary_key = columns;
    }

    /// Get primary key columns
    pub fn primary_key(&self) -> &[ColumnId] {
        &self.primary_key
    }

    /// Add unique constraint
    pub fn add_unique_constraint(&mut self, columns: Vec<ColumnId>) {
        self.unique_constraints.push(columns);
    }

    /// Get unique constraints
    pub fn unique_constraints(&self) -> &[Vec<ColumnId>] {
        &self.unique_constraints
    }

    /// Add foreign key constraint
    pub fn add_foreign_key(&mut self, fk: ForeignKey) {
        self.foreign_keys.push(fk);
    }

    /// Get foreign keys
    pub fn foreign_keys(&self) -> &[ForeignKey] {
        &self.foreign_keys
    }

    /// Check if schemas are compatible for union/difference
    pub fn is_compatible(&self, other: &Schema) -> bool {
        if self.arity() != other.arity() {
            return false;
        }
        for (c1, c2) in self.columns.iter().zip(other.columns.iter()) {
            if c1.data_type() != c2.data_type() {
                return false;
            }
        }
        true
    }

    /// Get column types as a vector
    pub fn column_types(&self) -> Vec<DataType> {
        self.columns.iter().map(|c| c.data_type().clone()).collect()
    }

    /// Create a projection schema with selected columns
    pub fn project(&self, columns: &[ColumnId]) -> Schema {
        let mut projected = Schema::new(format!("{}_proj", self.name));
        for &col_id in columns {
            if let Some(col) = self.column(col_id) {
                projected.add_column(col.name(), col.data_type().clone());
            }
        }
        projected
    }

    /// Create a join schema (concatenation of columns)
    pub fn join(&self, other: &Schema, suffix: &str) -> Schema {
        let mut joined = Schema::new(format!("{}_{}{}", self.name, other.name, suffix));
        for col in &self.columns {
            joined.add_column(col.name(), col.data_type().clone());
        }
        for col in &other.columns {
            joined.add_column(col.name(), col.data_type().clone());
        }
        joined
    }
}

/// Schema builder for fluent API
pub struct SchemaBuilder {
    schema: Schema,
    interner: lasso::ThreadedRodeo,
}

impl SchemaBuilder {
    /// Create a new schema builder
    pub fn new(name: &str) -> Self {
        Self {
            schema: Schema::new(name.to_string()),
            interner: lasso::ThreadedRodeo::default(),
        }
    }

    /// Add a column
    pub fn column(mut self, name: &str, data_type: DataType) -> Self {
        let spur = self.interner.get_or_intern(name);
        self.schema.add_column(spur, data_type);
        self
    }

    /// Add a nullable column
    pub fn nullable_column(mut self, name: &str, data_type: DataType) -> Self {
        let spur = self.interner.get_or_intern(name);
        self.schema.add_nullable_column(spur, data_type);
        self
    }

    /// Set primary key by column names
    pub fn primary_key(mut self, columns: &[&str]) -> Self {
        let pks: Vec<ColumnId> = columns
            .iter()
            .filter_map(|name| {
                let spur = self.interner.get(name)?;
                self.schema.column_index(spur)
            })
            .collect();
        self.schema.set_primary_key(pks);
        self
    }

    /// Build the schema
    pub fn build(self) -> Schema {
        self.schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let mut schema = Schema::new("test".to_string());
        let interner = lasso::ThreadedRodeo::default();

        let name1 = interner.get_or_intern("col1");
        let name2 = interner.get_or_intern("col2");

        let id1 = schema.add_column(name1, DataType::Int64);
        let id2 = schema.add_column(name2, DataType::Symbol);

        assert_eq!(schema.arity(), 2);
        assert_eq!(id1.raw(), 0);
        assert_eq!(id2.raw(), 1);
    }

    #[test]
    fn test_column_lookup() {
        let mut schema = Schema::new("test".to_string());
        let interner = lasso::ThreadedRodeo::default();

        let name = interner.get_or_intern("col1");
        schema.add_column(name, DataType::Bool);

        let col = schema.column_by_name(name);
        assert!(col.is_some());
        assert_eq!(
            *col.map(|c| c.data_type()).unwrap_or(&DataType::Int64),
            DataType::Bool
        );
    }

    #[test]
    fn test_schema_compatibility() {
        let interner = lasso::ThreadedRodeo::default();
        let name1 = interner.get_or_intern("col1");
        let name2 = interner.get_or_intern("col2");

        let mut s1 = Schema::new("s1".to_string());
        s1.add_column(name1, DataType::Int64);
        s1.add_column(name2, DataType::Symbol);

        let mut s2 = Schema::new("s2".to_string());
        s2.add_column(name1, DataType::Int64);
        s2.add_column(name2, DataType::Symbol);

        assert!(s1.is_compatible(&s2));

        let mut s3 = Schema::new("s3".to_string());
        s3.add_column(name1, DataType::Bool);

        assert!(!s1.is_compatible(&s3));
    }

    #[test]
    fn test_schema_projection() {
        let mut schema = Schema::new("test".to_string());
        let interner = lasso::ThreadedRodeo::default();

        let name1 = interner.get_or_intern("a");
        let name2 = interner.get_or_intern("b");
        let name3 = interner.get_or_intern("c");

        schema.add_column(name1, DataType::Int64);
        let id2 = schema.add_column(name2, DataType::Symbol);
        schema.add_column(name3, DataType::Bool);

        let projected = schema.project(&[id2]);
        assert_eq!(projected.arity(), 1);
    }

    #[test]
    fn test_data_type_properties() {
        assert!(DataType::Int64.is_numeric());
        assert!(DataType::Int64.is_comparable());
        assert!(!DataType::Bytes.is_numeric());
        assert!(DataType::Bytes.supports_equality());
    }

    #[test]
    fn test_schema_builder() {
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Int64)
            .column("name", DataType::Symbol)
            .nullable_column("value", DataType::Rational)
            .primary_key(&["id"])
            .build();

        assert_eq!(schema.arity(), 3);
        assert_eq!(schema.primary_key().len(), 1);
    }
}
