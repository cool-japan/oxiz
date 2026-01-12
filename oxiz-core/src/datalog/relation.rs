//! Relation storage and operations for Datalog
//!
//! Relations are the fundamental data structures in Datalog, representing
//! sets of tuples. This module provides both EDB (Extensional Database)
//! and IDB (Intensional Database) relation types.

use parking_lot::RwLock;
use rustc_hash::FxHashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::index::{IndexId, IndexKey, MultiIndex, TupleId};
use super::schema::{ColumnId, Schema};
use super::tuple::{Tuple, Value};

/// Unique identifier for a relation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationId(pub u64);

impl RelationId {
    /// Create a new relation ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get raw value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Kind of relation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationKind {
    /// Extensional Database - base facts
    Edb,
    /// Intensional Database - derived facts
    Idb,
    /// Delta relation for semi-naive evaluation
    Delta,
    /// New delta for current iteration
    NewDelta,
    /// Temporary relation
    Temp,
}

/// Delta tracking for semi-naive evaluation
#[derive(Debug)]
pub struct DeltaSet {
    /// Current delta tuples
    current: FxHashSet<Tuple>,
    /// New tuples to add in next iteration
    new_tuples: FxHashSet<Tuple>,
    /// Whether delta has changed
    changed: bool,
}

impl DeltaSet {
    /// Create a new delta set
    pub fn new() -> Self {
        Self {
            current: FxHashSet::default(),
            new_tuples: FxHashSet::default(),
            changed: false,
        }
    }

    /// Add a new tuple
    pub fn add(&mut self, tuple: Tuple) -> bool {
        if self.new_tuples.insert(tuple) {
            self.changed = true;
            true
        } else {
            false
        }
    }

    /// Advance to next iteration
    pub fn advance(&mut self) {
        self.current = std::mem::take(&mut self.new_tuples);
        self.changed = false;
    }

    /// Get current delta
    pub fn current(&self) -> &FxHashSet<Tuple> {
        &self.current
    }

    /// Check if changed
    pub fn has_changed(&self) -> bool {
        self.changed || !self.new_tuples.is_empty()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.current.is_empty() && self.new_tuples.is_empty()
    }

    /// Clear all
    pub fn clear(&mut self) {
        self.current.clear();
        self.new_tuples.clear();
        self.changed = false;
    }
}

impl Default for DeltaSet {
    fn default() -> Self {
        Self::new()
    }
}

/// A Datalog relation (table)
#[derive(Debug)]
pub struct Relation {
    /// Relation ID
    id: RelationId,
    /// Relation name
    name: String,
    /// Schema
    schema: Schema,
    /// Kind of relation
    kind: RelationKind,
    /// Tuple storage
    tuples: Vec<Tuple>,
    /// Set for deduplication
    tuple_set: FxHashSet<Tuple>,
    /// Indexes
    indexes: MultiIndex,
    /// Next tuple ID
    next_tuple_id: AtomicU64,
    /// Delta for semi-naive evaluation
    delta: Option<DeltaSet>,
    /// Statistics
    stats: RelationStats,
}

/// Relation statistics
#[derive(Debug, Default)]
pub struct RelationStats {
    /// Total insertions
    pub insertions: u64,
    /// Total deletions
    pub deletions: u64,
    /// Lookups performed
    pub lookups: u64,
    /// Scans performed
    pub scans: u64,
    /// Joins participated in
    pub joins: u64,
}

impl Relation {
    /// Create a new relation
    pub fn new(id: RelationId, name: String, schema: Schema, kind: RelationKind) -> Self {
        Self {
            id,
            name,
            schema,
            kind,
            tuples: Vec::new(),
            tuple_set: FxHashSet::default(),
            indexes: MultiIndex::new(),
            next_tuple_id: AtomicU64::new(0),
            delta: if kind == RelationKind::Idb {
                Some(DeltaSet::new())
            } else {
                None
            },
            stats: RelationStats::default(),
        }
    }

    /// Create an EDB relation
    pub fn edb(id: RelationId, name: String, schema: Schema) -> Self {
        Self::new(id, name, schema, RelationKind::Edb)
    }

    /// Create an IDB relation
    pub fn idb(id: RelationId, name: String, schema: Schema) -> Self {
        Self::new(id, name, schema, RelationKind::Idb)
    }

    /// Get relation ID
    pub fn id(&self) -> RelationId {
        self.id
    }

    /// Get relation name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get kind
    pub fn kind(&self) -> RelationKind {
        self.kind
    }

    /// Get tuple count
    pub fn len(&self) -> usize {
        self.tuples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tuples.is_empty()
    }

    /// Get statistics
    pub fn stats(&self) -> &RelationStats {
        &self.stats
    }

    /// Create a hash index on specified columns
    pub fn create_hash_index(&mut self, columns: Vec<ColumnId>) -> IndexId {
        let idx_id = self.indexes.add_hash_index(columns.clone());

        // Populate index with existing tuples
        if let Some(index) = self.indexes.get_mut(idx_id) {
            for (i, tuple) in self.tuples.iter().enumerate() {
                let key = IndexKey::from_tuple(tuple, &columns);
                index.insert(key, TupleId::new(i as u64));
            }
        }

        idx_id
    }

    /// Create a btree index on specified columns
    pub fn create_btree_index(&mut self, columns: Vec<ColumnId>) -> IndexId {
        let idx_id = self.indexes.add_btree_index(columns.clone());

        // Populate index
        if let Some(index) = self.indexes.get_mut(idx_id) {
            for (i, tuple) in self.tuples.iter().enumerate() {
                let key = IndexKey::from_tuple(tuple, &columns);
                index.insert(key, TupleId::new(i as u64));
            }
        }

        idx_id
    }

    /// Insert a tuple
    pub fn insert(&mut self, tuple: Tuple) -> bool {
        if !tuple.matches_schema(&self.schema) {
            return false;
        }

        // Check for duplicates
        if !self.tuple_set.insert(tuple.clone()) {
            return false;
        }

        let tuple_id = TupleId::new(self.next_tuple_id.fetch_add(1, Ordering::SeqCst));

        // Update indexes
        self.indexes.insert_all(&tuple, tuple_id);

        // Track in delta if IDB
        if let Some(ref mut delta) = self.delta {
            delta.add(tuple.clone());
        }

        self.tuples.push(tuple);
        self.stats.insertions += 1;
        true
    }

    /// Insert multiple tuples
    pub fn insert_all(&mut self, tuples: impl IntoIterator<Item = Tuple>) -> usize {
        let mut count = 0;
        for tuple in tuples {
            if self.insert(tuple) {
                count += 1;
            }
        }
        count
    }

    /// Check if tuple exists
    pub fn contains(&self, tuple: &Tuple) -> bool {
        self.tuple_set.contains(tuple)
    }

    /// Get tuple by index
    pub fn get(&self, idx: usize) -> Option<&Tuple> {
        self.tuples.get(idx)
    }

    /// Iterate over all tuples
    pub fn iter(&self) -> impl Iterator<Item = &Tuple> {
        // Note: stats.scans would be incremented here for profiling
        self.tuples.iter()
    }

    /// Lookup by index
    pub fn lookup(&self, index_id: IndexId, key: &IndexKey) -> Vec<&Tuple> {
        if let Some(index) = self.indexes.get(index_id) {
            index
                .lookup(key)
                .filter_map(|tid| self.tuples.get(tid.raw() as usize))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Lookup by column values (auto-select index)
    pub fn lookup_by_columns(&self, columns: &[ColumnId], values: &[Value]) -> Vec<&Tuple> {
        // Try to find an index
        if let Some(idx_id) = self.indexes.find_index(columns) {
            let key = IndexKey::new(values.to_vec());
            return self.lookup(idx_id, &key);
        }

        // Fall back to scan
        self.tuples
            .iter()
            .filter(|tuple| {
                columns
                    .iter()
                    .zip(values.iter())
                    .all(|(col, val)| tuple.get_column(*col) == Some(val))
            })
            .collect()
    }

    /// Project relation to subset of columns
    pub fn project(&self, columns: &[ColumnId]) -> Relation {
        let new_schema = self.schema.project(columns);
        let mut result = Relation::new(
            RelationId::new(0), // Temp ID
            format!("{}_proj", self.name),
            new_schema,
            RelationKind::Temp,
        );

        for tuple in &self.tuples {
            result.insert(tuple.project(columns));
        }

        result
    }

    /// Select tuples matching predicate
    pub fn select<F>(&self, predicate: F) -> Relation
    where
        F: Fn(&Tuple) -> bool,
    {
        let mut result = Relation::new(
            RelationId::new(0),
            format!("{}_sel", self.name),
            self.schema.clone(),
            RelationKind::Temp,
        );

        for tuple in &self.tuples {
            if predicate(tuple) {
                result.insert(tuple.clone());
            }
        }

        result
    }

    /// Union with another relation
    pub fn union(&self, other: &Relation) -> Option<Relation> {
        if !self.schema.is_compatible(other.schema()) {
            return None;
        }

        let mut result = Relation::new(
            RelationId::new(0),
            format!("{}_union_{}", self.name, other.name),
            self.schema.clone(),
            RelationKind::Temp,
        );

        for tuple in &self.tuples {
            result.insert(tuple.clone());
        }
        for tuple in other.iter() {
            result.insert(tuple.clone());
        }

        Some(result)
    }

    /// Difference (self - other)
    pub fn difference(&self, other: &Relation) -> Option<Relation> {
        if !self.schema.is_compatible(other.schema()) {
            return None;
        }

        let mut result = Relation::new(
            RelationId::new(0),
            format!("{}_diff_{}", self.name, other.name),
            self.schema.clone(),
            RelationKind::Temp,
        );

        for tuple in &self.tuples {
            if !other.contains(tuple) {
                result.insert(tuple.clone());
            }
        }

        Some(result)
    }

    /// Intersection with another relation
    pub fn intersection(&self, other: &Relation) -> Option<Relation> {
        if !self.schema.is_compatible(other.schema()) {
            return None;
        }

        let mut result = Relation::new(
            RelationId::new(0),
            format!("{}_intersect_{}", self.name, other.name),
            self.schema.clone(),
            RelationKind::Temp,
        );

        for tuple in &self.tuples {
            if other.contains(tuple) {
                result.insert(tuple.clone());
            }
        }

        Some(result)
    }

    /// Natural join with another relation
    pub fn natural_join(&self, other: &Relation) -> Relation {
        let joined_schema = self.schema.join(other.schema(), "_join");
        let mut result = Relation::new(
            RelationId::new(0),
            format!("{}_join_{}", self.name, other.name),
            joined_schema,
            RelationKind::Temp,
        );

        // Find common columns by name (simplified - real impl would match properly)
        // For now, do cartesian product
        for t1 in &self.tuples {
            for t2 in other.iter() {
                result.insert(t1.concat(t2));
            }
        }

        result
    }

    /// Hash join with another relation on specified columns
    pub fn hash_join(
        &self,
        other: &Relation,
        self_cols: &[ColumnId],
        other_cols: &[ColumnId],
    ) -> Relation {
        let joined_schema = self.schema.join(other.schema(), "_hjoin");
        let mut result = Relation::new(
            RelationId::new(0),
            format!("{}_hjoin_{}", self.name, other.name),
            joined_schema,
            RelationKind::Temp,
        );

        // Build hash table on smaller relation
        let (build_rel, probe_rel, build_cols, probe_cols) = if self.len() <= other.len() {
            (self, other, self_cols, other_cols)
        } else {
            (other, self, other_cols, self_cols)
        };

        // Build phase
        let mut hash_table: rustc_hash::FxHashMap<IndexKey, Vec<&Tuple>> =
            rustc_hash::FxHashMap::default();
        for tuple in build_rel.iter() {
            let key = IndexKey::from_tuple(tuple, build_cols);
            hash_table.entry(key).or_default().push(tuple);
        }

        // Probe phase
        for probe_tuple in probe_rel.iter() {
            let key = IndexKey::from_tuple(probe_tuple, probe_cols);
            if let Some(matches) = hash_table.get(&key) {
                for build_tuple in matches {
                    // Determine order based on original relation order
                    let (left, right) = if self.len() <= other.len() {
                        (*build_tuple, probe_tuple)
                    } else {
                        (probe_tuple, *build_tuple)
                    };
                    result.insert(left.concat(right));
                }
            }
        }

        result
    }

    /// Get delta set (for semi-naive evaluation)
    pub fn delta(&self) -> Option<&DeltaSet> {
        self.delta.as_ref()
    }

    /// Get mutable delta set
    pub fn delta_mut(&mut self) -> Option<&mut DeltaSet> {
        self.delta.as_mut()
    }

    /// Advance delta to next iteration
    pub fn advance_delta(&mut self) {
        if let Some(ref mut delta) = self.delta {
            delta.advance();
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.tuples.clear();
        self.tuple_set.clear();
        self.indexes.clear_all();
        self.next_tuple_id.store(0, Ordering::SeqCst);
        if let Some(ref mut delta) = self.delta {
            delta.clear();
        }
    }

    /// Clone schema only (create empty relation with same structure)
    pub fn clone_empty(&self) -> Relation {
        Relation::new(
            RelationId::new(0),
            format!("{}_empty", self.name),
            self.schema.clone(),
            RelationKind::Temp,
        )
    }
}

/// Thread-safe shared relation
pub type SharedRelation = Arc<RwLock<Relation>>;

/// Create a shared relation
pub fn shared_relation(relation: Relation) -> SharedRelation {
    Arc::new(RwLock::new(relation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datalog::schema::DataType;
    use crate::datalog::tuple::TupleBuilder;
    use lasso::ThreadedRodeo;

    fn create_test_relation() -> Relation {
        let interner = ThreadedRodeo::default();
        let mut schema = Schema::new("test".to_string());
        schema.add_column(interner.get_or_intern("a"), DataType::Int64);
        schema.add_column(interner.get_or_intern("b"), DataType::Int64);

        Relation::edb(RelationId::new(1), "test".to_string(), schema)
    }

    #[test]
    fn test_relation_insert() {
        let mut rel = create_test_relation();

        let t1 = TupleBuilder::new().push_i64(1).push_i64(2).build();
        let t2 = TupleBuilder::new().push_i64(3).push_i64(4).build();

        assert!(rel.insert(t1.clone()));
        assert!(rel.insert(t2));
        assert!(!rel.insert(t1)); // Duplicate

        assert_eq!(rel.len(), 2);
    }

    #[test]
    fn test_relation_contains() {
        let mut rel = create_test_relation();

        let t1 = TupleBuilder::new().push_i64(1).push_i64(2).build();
        rel.insert(t1.clone());

        assert!(rel.contains(&t1));

        let t2 = TupleBuilder::new().push_i64(3).push_i64(4).build();
        assert!(!rel.contains(&t2));
    }

    #[test]
    fn test_relation_index_lookup() {
        let mut rel = create_test_relation();
        let idx = rel.create_hash_index(vec![ColumnId::new(0)]);

        rel.insert(TupleBuilder::new().push_i64(1).push_i64(2).build());
        rel.insert(TupleBuilder::new().push_i64(1).push_i64(3).build());
        rel.insert(TupleBuilder::new().push_i64(2).push_i64(4).build());

        let key = IndexKey::new(vec![Value::Int64(1)]);
        let results = rel.lookup(idx, &key);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_relation_project() {
        let mut rel = create_test_relation();
        rel.insert(TupleBuilder::new().push_i64(1).push_i64(2).build());
        rel.insert(TupleBuilder::new().push_i64(3).push_i64(4).build());

        let projected = rel.project(&[ColumnId::new(0)]);
        assert_eq!(projected.len(), 2);
        assert_eq!(projected.schema().arity(), 1);
    }

    #[test]
    fn test_relation_select() {
        let mut rel = create_test_relation();
        rel.insert(TupleBuilder::new().push_i64(1).push_i64(2).build());
        rel.insert(TupleBuilder::new().push_i64(3).push_i64(4).build());
        rel.insert(TupleBuilder::new().push_i64(5).push_i64(6).build());

        let selected = rel.select(|t| t.get(0).and_then(|v| v.as_i64()).map_or(false, |n| n > 2));

        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_relation_union() {
        let mut rel1 = create_test_relation();
        rel1.insert(TupleBuilder::new().push_i64(1).push_i64(2).build());

        let mut rel2 = create_test_relation();
        rel2.insert(TupleBuilder::new().push_i64(3).push_i64(4).build());
        rel2.insert(TupleBuilder::new().push_i64(1).push_i64(2).build()); // Duplicate

        let union = rel1.union(&rel2).unwrap();
        assert_eq!(union.len(), 2);
    }

    #[test]
    fn test_relation_difference() {
        let mut rel1 = create_test_relation();
        rel1.insert(TupleBuilder::new().push_i64(1).push_i64(2).build());
        rel1.insert(TupleBuilder::new().push_i64(3).push_i64(4).build());

        let mut rel2 = create_test_relation();
        rel2.insert(TupleBuilder::new().push_i64(1).push_i64(2).build());

        let diff = rel1.difference(&rel2).unwrap();
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn test_relation_hash_join() {
        let interner = ThreadedRodeo::default();

        // Create first relation with columns a, b
        let mut schema1 = Schema::new("r1".to_string());
        schema1.add_column(interner.get_or_intern("a"), DataType::Int64);
        schema1.add_column(interner.get_or_intern("b"), DataType::Int64);
        let mut rel1 = Relation::edb(RelationId::new(1), "r1".to_string(), schema1);
        rel1.insert(TupleBuilder::new().push_i64(1).push_i64(10).build());
        rel1.insert(TupleBuilder::new().push_i64(2).push_i64(20).build());

        // Create second relation with columns c, d
        let mut schema2 = Schema::new("r2".to_string());
        schema2.add_column(interner.get_or_intern("c"), DataType::Int64);
        schema2.add_column(interner.get_or_intern("d"), DataType::Int64);
        let mut rel2 = Relation::edb(RelationId::new(2), "r2".to_string(), schema2);
        rel2.insert(TupleBuilder::new().push_i64(1).push_i64(100).build());
        rel2.insert(TupleBuilder::new().push_i64(3).push_i64(300).build());

        // Join on first column of each
        let joined = rel1.hash_join(&rel2, &[ColumnId::new(0)], &[ColumnId::new(0)]);
        assert_eq!(joined.len(), 1); // Only (1, 10, 1, 100)
    }

    #[test]
    fn test_delta_set() {
        let mut delta = DeltaSet::new();

        assert!(!delta.has_changed());

        delta.add(TupleBuilder::new().push_i64(1).build());
        assert!(delta.has_changed());

        delta.advance();
        assert_eq!(delta.current().len(), 1);
        assert!(!delta.has_changed());
    }
}
