//! Index structures for Datalog relations
//!
//! Provides efficient lookup of tuples based on key columns.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

use super::schema::ColumnId;
use super::tuple::{Tuple, Value};

/// Unique identifier for an index
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexId(pub u64);

impl IndexId {
    /// Create a new index ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get raw value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Type of index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// Hash index for equality lookups
    Hash,
    /// BTree index for range queries
    BTree,
    /// Bitmap index for low-cardinality columns
    Bitmap,
    /// Covering index (stores full tuples)
    Covering,
}

/// Key for hash index lookups
#[derive(Debug, Clone)]
pub struct IndexKey {
    values: Vec<Value>,
}

impl IndexKey {
    /// Create from values
    pub fn new(values: Vec<Value>) -> Self {
        Self { values }
    }

    /// Create from tuple projection
    pub fn from_tuple(tuple: &Tuple, columns: &[ColumnId]) -> Self {
        let values: Vec<Value> = columns
            .iter()
            .filter_map(|col| tuple.get_column(*col).cloned())
            .collect();
        Self { values }
    }

    /// Get values
    pub fn values(&self) -> &[Value] {
        &self.values
    }
}

impl PartialEq for IndexKey {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl Eq for IndexKey {}

impl Hash for IndexKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for val in &self.values {
            val.hash(state);
        }
    }
}

impl PartialOrd for IndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        for (a, b) in self.values.iter().zip(other.values.iter()) {
            match a.cmp(b) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        self.values.len().cmp(&other.values.len())
    }
}

/// Tuple ID for referencing tuples in storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TupleId(pub u64);

impl TupleId {
    /// Create a new tuple ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get raw value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Index trait for different index implementations
pub trait Index: Send + Sync + std::fmt::Debug {
    /// Get index kind
    fn kind(&self) -> IndexKind;

    /// Get indexed columns
    fn columns(&self) -> &[ColumnId];

    /// Insert a tuple into the index
    fn insert(&mut self, key: IndexKey, tuple_id: TupleId);

    /// Remove a tuple from the index
    fn remove(&mut self, key: &IndexKey, tuple_id: TupleId) -> bool;

    /// Lookup exact match
    fn lookup(&self, key: &IndexKey) -> Box<dyn Iterator<Item = TupleId> + '_>;

    /// Check if key exists
    fn contains(&self, key: &IndexKey) -> bool;

    /// Get number of entries
    fn len(&self) -> usize;

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the index
    fn clear(&mut self);
}

/// Hash-based index for equality lookups
#[derive(Debug)]
pub struct HashIndex {
    /// Indexed columns
    columns: Vec<ColumnId>,
    /// Index storage: key -> set of tuple IDs
    data: FxHashMap<IndexKey, FxHashSet<TupleId>>,
    /// Total entry count
    entry_count: usize,
}

impl HashIndex {
    /// Create a new hash index
    pub fn new(columns: Vec<ColumnId>) -> Self {
        Self {
            columns,
            data: FxHashMap::default(),
            entry_count: 0,
        }
    }

    /// Get all keys
    pub fn keys(&self) -> impl Iterator<Item = &IndexKey> {
        self.data.keys()
    }

    /// Get bucket count (for statistics)
    pub fn bucket_count(&self) -> usize {
        self.data.len()
    }
}

impl Index for HashIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Hash
    }

    fn columns(&self) -> &[ColumnId] {
        &self.columns
    }

    fn insert(&mut self, key: IndexKey, tuple_id: TupleId) {
        let set = self.data.entry(key).or_default();
        if set.insert(tuple_id) {
            self.entry_count += 1;
        }
    }

    fn remove(&mut self, key: &IndexKey, tuple_id: TupleId) -> bool {
        if let Some(set) = self.data.get_mut(key)
            && set.remove(&tuple_id)
        {
            self.entry_count -= 1;
            if set.is_empty() {
                self.data.remove(key);
            }
            return true;
        }
        false
    }

    fn lookup(&self, key: &IndexKey) -> Box<dyn Iterator<Item = TupleId> + '_> {
        if let Some(set) = self.data.get(key) {
            Box::new(set.iter().copied())
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn contains(&self, key: &IndexKey) -> bool {
        self.data.contains_key(key)
    }

    fn len(&self) -> usize {
        self.entry_count
    }

    fn clear(&mut self) {
        self.data.clear();
        self.entry_count = 0;
    }
}

/// BTree-based index for range queries
#[derive(Debug)]
pub struct BTreeIndex {
    /// Indexed columns
    columns: Vec<ColumnId>,
    /// Index storage
    data: BTreeMap<IndexKey, BTreeSet<TupleId>>,
    /// Total entry count
    entry_count: usize,
}

impl BTreeIndex {
    /// Create a new btree index
    pub fn new(columns: Vec<ColumnId>) -> Self {
        Self {
            columns,
            data: BTreeMap::new(),
            entry_count: 0,
        }
    }

    /// Range lookup
    pub fn range(&self, start: &IndexKey, end: &IndexKey) -> impl Iterator<Item = TupleId> + '_ {
        self.data
            .range(start..=end)
            .flat_map(|(_, ids)| ids.iter().copied())
    }

    /// Get minimum key
    pub fn min(&self) -> Option<&IndexKey> {
        self.data.first_key_value().map(|(k, _)| k)
    }

    /// Get maximum key
    pub fn max(&self) -> Option<&IndexKey> {
        self.data.last_key_value().map(|(k, _)| k)
    }
}

impl Index for BTreeIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::BTree
    }

    fn columns(&self) -> &[ColumnId] {
        &self.columns
    }

    fn insert(&mut self, key: IndexKey, tuple_id: TupleId) {
        let set = self.data.entry(key).or_default();
        if set.insert(tuple_id) {
            self.entry_count += 1;
        }
    }

    fn remove(&mut self, key: &IndexKey, tuple_id: TupleId) -> bool {
        if let Some(set) = self.data.get_mut(key)
            && set.remove(&tuple_id)
        {
            self.entry_count -= 1;
            if set.is_empty() {
                self.data.remove(key);
            }
            return true;
        }
        false
    }

    fn lookup(&self, key: &IndexKey) -> Box<dyn Iterator<Item = TupleId> + '_> {
        if let Some(set) = self.data.get(key) {
            Box::new(set.iter().copied())
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn contains(&self, key: &IndexKey) -> bool {
        self.data.contains_key(key)
    }

    fn len(&self) -> usize {
        self.entry_count
    }

    fn clear(&mut self) {
        self.data.clear();
        self.entry_count = 0;
    }
}

/// Bitmap index for low-cardinality columns
#[derive(Debug)]
pub struct BitmapIndex {
    /// Indexed columns
    columns: Vec<ColumnId>,
    /// Bitmaps per distinct value
    bitmaps: HashMap<IndexKey, Vec<u64>>,
    /// Maximum tuple ID seen
    max_tuple_id: u64,
    /// Entry count
    entry_count: usize,
}

impl BitmapIndex {
    /// Create a new bitmap index
    pub fn new(columns: Vec<ColumnId>) -> Self {
        Self {
            columns,
            bitmaps: HashMap::new(),
            max_tuple_id: 0,
            entry_count: 0,
        }
    }

    /// Check if bit is set
    fn get_bit(bitmap: &[u64], id: u64) -> bool {
        let word_idx = (id / 64) as usize;
        let bit_idx = id % 64;
        if word_idx < bitmap.len() {
            (bitmap[word_idx] >> bit_idx) & 1 == 1
        } else {
            false
        }
    }

    /// Set bit
    fn set_bit(bitmap: &mut Vec<u64>, id: u64) {
        let word_idx = (id / 64) as usize;
        let bit_idx = id % 64;
        while bitmap.len() <= word_idx {
            bitmap.push(0);
        }
        bitmap[word_idx] |= 1u64 << bit_idx;
    }

    /// Clear bit
    fn clear_bit(bitmap: &mut [u64], id: u64) -> bool {
        let word_idx = (id / 64) as usize;
        let bit_idx = id % 64;
        if word_idx < bitmap.len() {
            let was_set = (bitmap[word_idx] >> bit_idx) & 1 == 1;
            bitmap[word_idx] &= !(1u64 << bit_idx);
            was_set
        } else {
            false
        }
    }

    /// Iterate over set bits
    fn iter_bits(bitmap: &[u64]) -> impl Iterator<Item = TupleId> + '_ {
        bitmap.iter().enumerate().flat_map(|(word_idx, &word)| {
            (0..64).filter_map(move |bit_idx| {
                if (word >> bit_idx) & 1 == 1 {
                    Some(TupleId::new((word_idx as u64) * 64 + bit_idx))
                } else {
                    None
                }
            })
        })
    }

    /// Get distinct value count
    pub fn cardinality(&self) -> usize {
        self.bitmaps.len()
    }
}

impl Index for BitmapIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Bitmap
    }

    fn columns(&self) -> &[ColumnId] {
        &self.columns
    }

    fn insert(&mut self, key: IndexKey, tuple_id: TupleId) {
        let bitmap = self.bitmaps.entry(key).or_default();
        if !Self::get_bit(bitmap, tuple_id.0) {
            Self::set_bit(bitmap, tuple_id.0);
            self.max_tuple_id = self.max_tuple_id.max(tuple_id.0);
            self.entry_count += 1;
        }
    }

    fn remove(&mut self, key: &IndexKey, tuple_id: TupleId) -> bool {
        if let Some(bitmap) = self.bitmaps.get_mut(key)
            && Self::clear_bit(bitmap, tuple_id.0)
        {
            self.entry_count -= 1;
            return true;
        }
        false
    }

    fn lookup(&self, key: &IndexKey) -> Box<dyn Iterator<Item = TupleId> + '_> {
        if let Some(bitmap) = self.bitmaps.get(key) {
            Box::new(Self::iter_bits(bitmap))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn contains(&self, key: &IndexKey) -> bool {
        self.bitmaps.get(key).is_some_and(|bm| !bm.is_empty())
    }

    fn len(&self) -> usize {
        self.entry_count
    }

    fn clear(&mut self) {
        self.bitmaps.clear();
        self.max_tuple_id = 0;
        self.entry_count = 0;
    }
}

/// Covering index that stores full tuples
#[derive(Debug)]
pub struct CoveringIndex {
    /// Indexed columns (key)
    columns: Vec<ColumnId>,
    /// Full tuple storage
    data: FxHashMap<IndexKey, Vec<(TupleId, Tuple)>>,
    /// Entry count
    entry_count: usize,
}

impl CoveringIndex {
    /// Create a new covering index
    pub fn new(columns: Vec<ColumnId>) -> Self {
        Self {
            columns,
            data: FxHashMap::default(),
            entry_count: 0,
        }
    }

    /// Insert tuple with full data
    pub fn insert_tuple(&mut self, key: IndexKey, tuple_id: TupleId, tuple: Tuple) {
        let entries = self.data.entry(key).or_default();
        if !entries.iter().any(|(id, _)| *id == tuple_id) {
            entries.push((tuple_id, tuple));
            self.entry_count += 1;
        }
    }

    /// Lookup with full tuple data
    pub fn lookup_tuples(&self, key: &IndexKey) -> impl Iterator<Item = (&TupleId, &Tuple)> + '_ {
        self.data
            .get(key)
            .into_iter()
            .flat_map(|v| v.iter().map(|(id, t)| (id, t)))
    }
}

impl Index for CoveringIndex {
    fn kind(&self) -> IndexKind {
        IndexKind::Covering
    }

    fn columns(&self) -> &[ColumnId] {
        &self.columns
    }

    fn insert(&mut self, key: IndexKey, tuple_id: TupleId) {
        let entries = self.data.entry(key).or_default();
        if !entries.iter().any(|(id, _)| *id == tuple_id) {
            entries.push((tuple_id, Tuple::empty()));
            self.entry_count += 1;
        }
    }

    fn remove(&mut self, key: &IndexKey, tuple_id: TupleId) -> bool {
        if let Some(entries) = self.data.get_mut(key) {
            let len_before = entries.len();
            entries.retain(|(id, _)| *id != tuple_id);
            let removed = entries.len() < len_before;
            if removed {
                self.entry_count -= 1;
            }
            if entries.is_empty() {
                self.data.remove(key);
            }
            removed
        } else {
            false
        }
    }

    fn lookup(&self, key: &IndexKey) -> Box<dyn Iterator<Item = TupleId> + '_> {
        if let Some(entries) = self.data.get(key) {
            Box::new(entries.iter().map(|(id, _)| *id))
        } else {
            Box::new(std::iter::empty())
        }
    }

    fn contains(&self, key: &IndexKey) -> bool {
        self.data.contains_key(key)
    }

    fn len(&self) -> usize {
        self.entry_count
    }

    fn clear(&mut self) {
        self.data.clear();
        self.entry_count = 0;
    }
}

/// Multi-index container for managing multiple indexes on a relation
#[derive(Debug)]
pub struct MultiIndex {
    /// Indexes by ID
    indexes: HashMap<IndexId, Box<dyn Index>>,
    /// Next index ID
    next_id: u64,
    /// Column set to index ID mapping
    column_to_index: HashMap<Vec<ColumnId>, IndexId>,
}

impl MultiIndex {
    /// Create a new multi-index container
    pub fn new() -> Self {
        Self {
            indexes: HashMap::new(),
            next_id: 0,
            column_to_index: HashMap::new(),
        }
    }

    /// Add a hash index
    pub fn add_hash_index(&mut self, columns: Vec<ColumnId>) -> IndexId {
        if let Some(&existing) = self.column_to_index.get(&columns) {
            return existing;
        }

        let id = IndexId::new(self.next_id);
        self.next_id += 1;
        let index = Box::new(HashIndex::new(columns.clone()));
        self.indexes.insert(id, index);
        self.column_to_index.insert(columns, id);
        id
    }

    /// Add a btree index
    pub fn add_btree_index(&mut self, columns: Vec<ColumnId>) -> IndexId {
        let id = IndexId::new(self.next_id);
        self.next_id += 1;
        let index = Box::new(BTreeIndex::new(columns.clone()));
        self.indexes.insert(id, index);
        self.column_to_index.insert(columns, id);
        id
    }

    /// Get index by ID
    pub fn get(&self, id: IndexId) -> Option<&dyn Index> {
        self.indexes.get(&id).map(|b| b.as_ref())
    }

    /// Get mutable index by ID
    pub fn get_mut(&mut self, id: IndexId) -> Option<&mut Box<dyn Index>> {
        self.indexes.get_mut(&id)
    }

    /// Find index for columns
    pub fn find_index(&self, columns: &[ColumnId]) -> Option<IndexId> {
        self.column_to_index.get(columns).copied()
    }

    /// Insert into all indexes
    pub fn insert_all(&mut self, tuple: &Tuple, tuple_id: TupleId) {
        for index in self.indexes.values_mut() {
            let key = IndexKey::from_tuple(tuple, index.columns());
            index.insert(key, tuple_id);
        }
    }

    /// Remove from all indexes
    pub fn remove_all(&mut self, tuple: &Tuple, tuple_id: TupleId) {
        for index in self.indexes.values_mut() {
            let key = IndexKey::from_tuple(tuple, index.columns());
            index.remove(&key, tuple_id);
        }
    }

    /// Clear all indexes
    pub fn clear_all(&mut self) {
        for index in self.indexes.values_mut() {
            index.clear();
        }
    }

    /// Get all index IDs
    pub fn index_ids(&self) -> impl Iterator<Item = IndexId> + '_ {
        self.indexes.keys().copied()
    }
}

impl Default for MultiIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datalog::tuple::TupleBuilder;

    #[test]
    fn test_hash_index() {
        let mut index = HashIndex::new(vec![ColumnId::new(0)]);

        let key = IndexKey::new(vec![Value::Int64(42)]);
        index.insert(key.clone(), TupleId::new(1));
        index.insert(key.clone(), TupleId::new(2));

        assert!(index.contains(&key));
        assert_eq!(index.len(), 2);

        let results: Vec<_> = index.lookup(&key).collect();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_btree_index_range() {
        let mut index = BTreeIndex::new(vec![ColumnId::new(0)]);

        for i in 0..10 {
            let key = IndexKey::new(vec![Value::Int64(i)]);
            index.insert(key, TupleId::new(i as u64));
        }

        let start = IndexKey::new(vec![Value::Int64(3)]);
        let end = IndexKey::new(vec![Value::Int64(7)]);
        let results: Vec<_> = index.range(&start, &end).collect();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_bitmap_index() {
        let mut index = BitmapIndex::new(vec![ColumnId::new(0)]);

        let key = IndexKey::new(vec![Value::Bool(true)]);
        index.insert(key.clone(), TupleId::new(0));
        index.insert(key.clone(), TupleId::new(64));
        index.insert(key.clone(), TupleId::new(128));

        assert_eq!(index.len(), 3);
        let results: Vec<_> = index.lookup(&key).collect();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_index_key_from_tuple() {
        let tuple = TupleBuilder::new()
            .push_i64(1)
            .push_i64(2)
            .push_i64(3)
            .build();

        let key = IndexKey::from_tuple(&tuple, &[ColumnId::new(0), ColumnId::new(2)]);
        assert_eq!(key.values().len(), 2);
        assert_eq!(key.values()[0], Value::Int64(1));
        assert_eq!(key.values()[1], Value::Int64(3));
    }

    #[test]
    fn test_multi_index() {
        let mut multi = MultiIndex::new();

        let idx1 = multi.add_hash_index(vec![ColumnId::new(0)]);
        let idx2 = multi.add_hash_index(vec![ColumnId::new(1)]);

        let tuple = TupleBuilder::new().push_i64(1).push_i64(2).build();
        multi.insert_all(&tuple, TupleId::new(0));

        // Lookup by first column
        let key1 = IndexKey::new(vec![Value::Int64(1)]);
        let results1: Vec<_> = multi.get(idx1).unwrap().lookup(&key1).collect();
        assert_eq!(results1.len(), 1);

        // Lookup by second column
        let key2 = IndexKey::new(vec![Value::Int64(2)]);
        let results2: Vec<_> = multi.get(idx2).unwrap().lookup(&key2).collect();
        assert_eq!(results2.len(), 1);
    }

    #[test]
    fn test_index_remove() {
        let mut index = HashIndex::new(vec![ColumnId::new(0)]);

        let key = IndexKey::new(vec![Value::Int64(42)]);
        index.insert(key.clone(), TupleId::new(1));
        index.insert(key.clone(), TupleId::new(2));

        assert!(index.remove(&key, TupleId::new(1)));
        assert_eq!(index.len(), 1);

        assert!(index.remove(&key, TupleId::new(2)));
        assert_eq!(index.len(), 0);
        assert!(!index.contains(&key));
    }
}
