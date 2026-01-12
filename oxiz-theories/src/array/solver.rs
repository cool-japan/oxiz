//! Array Theory Solver
//!
//! Implements the theory of extensional arrays using a combination
//! of read-over-write axioms and a delayed lemma approach.

use crate::theory::{Theory, TheoryId, TheoryResult};
use oxiz_core::ast::TermId;
use oxiz_core::error::Result;
use rustc_hash::FxHashMap;

/// Represents a select operation: select(array, index)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SelectOp {
    /// The array being read from
    array: u32,
    /// The index being accessed
    index: u32,
}

/// Represents a store operation: store(array, index, value)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StoreOp {
    /// The original array
    array: u32,
    /// The index being written to
    index: u32,
    /// The value being written
    value: u32,
}

/// An array term (either a base array or a store result)
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ArrayTerm {
    /// A base array variable
    Base(u32),
    /// A store operation result
    Store {
        /// The underlying array
        base: u32,
        /// The store operation that created this
        store: StoreOp,
    },
}

/// A pending lemma to be instantiated
#[derive(Debug, Clone)]
enum PendingLemma {
    /// Read-over-write same: select(store(a, i, v), i) = v
    ReadOverWriteSame {
        /// The store operation
        store: StoreOp,
        /// The result array
        store_result: u32,
    },
    /// Read-over-write different: i ≠ j → select(store(a, i, v), j) = select(a, j)
    ReadOverWriteDiff {
        /// The store operation
        store: StoreOp,
        /// The result array
        store_result: u32,
        /// The select index
        select_index: u32,
    },
}

/// Array Theory Solver
#[derive(Debug)]
pub struct ArraySolver {
    /// Node counter
    next_node: u32,
    /// Term to node mapping
    term_to_node: FxHashMap<TermId, u32>,
    /// Node to term mapping
    node_to_term: Vec<Option<TermId>>,
    /// Array terms
    arrays: Vec<Option<ArrayTerm>>,
    /// Select operations: (select_node, SelectOp)
    selects: Vec<(u32, SelectOp)>,
    /// Store operations: (result_node, StoreOp)
    stores: Vec<(u32, StoreOp)>,
    /// Equivalence classes (simple union-find)
    parent: Vec<u32>,
    /// Trail for undo: (child, old_parent)
    trail: Vec<(u32, u32)>,
    /// Disequalities
    diseqs: Vec<(u32, u32, TermId)>,
    /// Pending lemmas to check
    pending_lemmas: Vec<PendingLemma>,
    /// Context stack for push/pop
    context_stack: Vec<ContextState>,
    /// Current conflicts (if any)
    current_conflict: Option<Vec<TermId>>,
}

/// State for push/pop
#[derive(Debug, Clone)]
struct ContextState {
    num_nodes: usize,
    num_selects: usize,
    num_stores: usize,
    num_diseqs: usize,
    num_pending_lemmas: usize,
    num_trail: usize,
}

impl Default for ArraySolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ArraySolver {
    /// Create a new array solver
    #[must_use]
    pub fn new() -> Self {
        Self {
            next_node: 0,
            term_to_node: FxHashMap::default(),
            node_to_term: Vec::new(),
            arrays: Vec::new(),
            selects: Vec::new(),
            stores: Vec::new(),
            parent: Vec::new(),
            trail: Vec::new(),
            diseqs: Vec::new(),
            pending_lemmas: Vec::new(),
            context_stack: Vec::new(),
            current_conflict: None,
        }
    }

    /// Create a new node
    fn new_node(&mut self) -> u32 {
        let id = self.next_node;
        self.next_node += 1;
        self.parent.push(id); // Initially its own parent
        self.node_to_term.push(None);
        self.arrays.push(None);
        id
    }

    /// Intern a term, returning its node
    pub fn intern(&mut self, term: TermId) -> u32 {
        if let Some(&node) = self.term_to_node.get(&term) {
            return node;
        }

        let node = self.new_node();
        self.term_to_node.insert(term, node);
        self.node_to_term[node as usize] = Some(term);
        node
    }

    /// Intern an array variable
    pub fn intern_array(&mut self, term: TermId) -> u32 {
        let node = self.intern(term);
        if self.arrays[node as usize].is_none() {
            self.arrays[node as usize] = Some(ArrayTerm::Base(node));
        }
        node
    }

    /// Intern a select operation: select(array, index)
    pub fn intern_select(&mut self, term: TermId, array: u32, index: u32) -> u32 {
        let node = self.intern(term);
        let select = SelectOp { array, index };
        self.selects.push((node, select));

        // Check for immediate read-over-write lemmas
        self.check_read_over_write(node, &select);

        node
    }

    /// Intern a store operation: store(array, index, value)
    pub fn intern_store(&mut self, term: TermId, array: u32, index: u32, value: u32) -> u32 {
        let node = self.intern(term);
        let store = StoreOp {
            array,
            index,
            value,
        };
        self.stores.push((node, store));
        self.arrays[node as usize] = Some(ArrayTerm::Store { base: array, store });

        // Add read-over-write-same lemma
        self.pending_lemmas.push(PendingLemma::ReadOverWriteSame {
            store,
            store_result: node,
        });

        node
    }

    /// Check for read-over-write lemmas when a new select is added
    fn check_read_over_write(&mut self, _select_node: u32, select: &SelectOp) {
        // Find all stores on the same array
        for (store_result, store) in &self.stores {
            if self.find(store.array) == self.find(select.array) || *store_result == select.array {
                // Check if indices are the same or different
                if self.find(store.index) == self.find(select.index) {
                    // Same index: select(store(a, i, v), i) = v
                    // Merge select_node with store.value
                    self.pending_lemmas.push(PendingLemma::ReadOverWriteSame {
                        store: *store,
                        store_result: *store_result,
                    });
                } else {
                    // Different indices: may need read-over-write-different lemma
                    self.pending_lemmas.push(PendingLemma::ReadOverWriteDiff {
                        store: *store,
                        store_result: *store_result,
                        select_index: select.index,
                    });
                }
            }
        }
    }

    /// Find the representative of a node (without path compression)
    fn find(&self, mut node: u32) -> u32 {
        while self.parent[node as usize] != node {
            node = self.parent[node as usize];
        }
        node
    }

    /// Find with path compression (when we have mutable access)
    /// Records trail for incremental undo
    fn find_compress(&mut self, mut node: u32) -> u32 {
        let mut root = node;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        // Path compression with trail recording
        while self.parent[node as usize] != root {
            let parent = self.parent[node as usize];
            // Record old parent for undo
            self.trail.push((node, parent));
            self.parent[node as usize] = root;
            node = parent;
        }
        root
    }

    /// Check if two nodes are equivalent
    pub fn are_equal(&self, a: u32, b: u32) -> bool {
        self.find(a) == self.find(b)
    }

    /// Merge two equivalence classes
    pub fn merge(&mut self, a: u32, b: u32, reason: TermId) -> Result<()> {
        let root_a = self.find_compress(a);
        let root_b = self.find_compress(b);

        if root_a != root_b {
            // Record old parent for undo
            self.trail.push((root_b, root_b));
            // Union by rank (simplified: always make root_a the parent)
            self.parent[root_b as usize] = root_a;

            // Check for conflicts with disequalities
            for (lhs, rhs, diseq_reason) in &self.diseqs {
                if self.find(*lhs) == self.find(*rhs) {
                    self.current_conflict = Some(vec![reason, *diseq_reason]);
                    return Ok(());
                }
            }
        }

        Ok(())
    }

    /// Assert a disequality
    pub fn assert_diseq(&mut self, a: u32, b: u32, reason: TermId) {
        self.diseqs.push((a, b, reason));

        // Check for immediate conflict
        if self.find(a) == self.find(b) {
            self.current_conflict = Some(vec![reason]);
        }
    }

    /// Process pending lemmas
    fn process_lemmas(&mut self) -> Result<TheoryResult> {
        let lemmas = std::mem::take(&mut self.pending_lemmas);
        let mut propagations: Vec<(TermId, Vec<TermId>)> = Vec::new();
        let mut pending_merges: Vec<(u32, u32)> = Vec::new();

        for lemma in lemmas {
            match lemma {
                PendingLemma::ReadOverWriteSame {
                    store,
                    store_result,
                } => {
                    // Find select(store(a, i, v), i) and equate with v
                    for (select_node, select) in &self.selects {
                        if self.find(select.array) == self.find(store_result)
                            && self.find(select.index) == self.find(store.index)
                        {
                            // select(store(a, i, v), i) = v
                            if !self.are_equal(*select_node, store.value) {
                                // Generate equality propagation
                                if let (Some(sel_term), Some(val_term)) = (
                                    self.node_to_term[*select_node as usize],
                                    self.node_to_term[store.value as usize],
                                ) {
                                    propagations.push((sel_term, vec![val_term]));
                                }
                                pending_merges.push((*select_node, store.value));
                            }
                        }
                    }
                }
                PendingLemma::ReadOverWriteDiff {
                    store,
                    store_result,
                    select_index,
                } => {
                    // If i ≠ j, then select(store(a, i, v), j) = select(a, j)
                    // Check if indices are proven different
                    if !self.are_equal(store.index, select_index) {
                        // Find the select we need to equate
                        for (select_node, select) in &self.selects {
                            if self.find(select.array) == self.find(store_result)
                                && self.find(select.index) == self.find(select_index)
                            {
                                // Find select(a, j) where a is the base array
                                for (other_select_node, other_select) in &self.selects {
                                    if self.find(other_select.array) == self.find(store.array)
                                        && self.find(other_select.index) == self.find(select_index)
                                        && !self.are_equal(*select_node, *other_select_node)
                                    {
                                        pending_merges.push((*select_node, *other_select_node));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Apply all pending merges
        for (a, b) in pending_merges {
            self.merge(a, b, TermId::new(0))?;
        }

        if let Some(conflict) = self.current_conflict.take() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        if !propagations.is_empty() {
            return Ok(TheoryResult::Propagate(propagations));
        }

        Ok(TheoryResult::Sat)
    }

    /// Check for conflicts
    pub fn check_conflicts(&self) -> Option<Vec<TermId>> {
        for (lhs, rhs, reason) in &self.diseqs {
            if self.find(*lhs) == self.find(*rhs) {
                return Some(vec![*reason]);
            }
        }
        None
    }
}

impl Theory for ArraySolver {
    fn id(&self) -> TheoryId {
        TheoryId::Arrays
    }

    fn name(&self) -> &str {
        "Arrays"
    }

    fn can_handle(&self, _term: TermId) -> bool {
        // Array theory handles select and store operations
        true
    }

    fn assert_true(&mut self, term: TermId) -> Result<TheoryResult> {
        // Assuming term is an equality, intern and merge
        let _ = self.intern(term);
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, term: TermId) -> Result<TheoryResult> {
        // Assuming term is an equality, assert disequality
        let node = self.intern(term);
        self.assert_diseq(node, node, term);
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        // Process any pending lemmas first
        let lemma_result = self.process_lemmas()?;
        if !matches!(lemma_result, TheoryResult::Sat) {
            return Ok(lemma_result);
        }

        // Check for conflicts
        if let Some(conflict) = self.check_conflicts() {
            return Ok(TheoryResult::Unsat(conflict));
        }

        Ok(TheoryResult::Sat)
    }

    fn push(&mut self) {
        self.context_stack.push(ContextState {
            num_nodes: self.next_node as usize,
            num_selects: self.selects.len(),
            num_stores: self.stores.len(),
            num_diseqs: self.diseqs.len(),
            num_pending_lemmas: self.pending_lemmas.len(),
            num_trail: self.trail.len(),
        });
    }

    fn pop(&mut self) {
        if let Some(state) = self.context_stack.pop() {
            // Restore nodes
            self.next_node = state.num_nodes as u32;
            self.node_to_term.truncate(state.num_nodes);
            self.arrays.truncate(state.num_nodes);
            self.parent.truncate(state.num_nodes);

            // Undo union-find operations via trail
            while self.trail.len() > state.num_trail {
                if let Some((node, old_parent)) = self.trail.pop() {
                    self.parent[node as usize] = old_parent;
                }
            }

            // Remove terms added after push
            self.term_to_node
                .retain(|_, &mut v| (v as usize) < state.num_nodes);

            // Restore constraints
            self.selects.truncate(state.num_selects);
            self.stores.truncate(state.num_stores);
            self.diseqs.truncate(state.num_diseqs);
            self.pending_lemmas.truncate(state.num_pending_lemmas);

            // Clear conflict
            self.current_conflict = None;
        }
    }

    fn reset(&mut self) {
        self.next_node = 0;
        self.term_to_node.clear();
        self.node_to_term.clear();
        self.arrays.clear();
        self.selects.clear();
        self.stores.clear();
        self.parent.clear();
        self.trail.clear();
        self.diseqs.clear();
        self.pending_lemmas.clear();
        self.context_stack.clear();
        self.current_conflict = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_basic() {
        let mut solver = ArraySolver::new();

        // Create an array and some indices
        let a = solver.intern_array(TermId::new(1));
        let i = solver.intern(TermId::new(2));
        let v = solver.intern(TermId::new(3));

        // store(a, i, v)
        let a_store = solver.intern_store(TermId::new(10), a, i, v);

        // select(store(a, i, v), i) should equal v
        let select = solver.intern_select(TermId::new(11), a_store, i);

        let result = solver.check().unwrap();
        // Result may be Sat or Propagate (propagating the equality)
        assert!(!matches!(result, TheoryResult::Unsat(_)));

        // After processing lemmas, select should be equal to v
        assert!(solver.are_equal(select, v));
    }

    #[test]
    fn test_array_read_different_index() {
        let mut solver = ArraySolver::new();

        // Create an array and indices
        let a = solver.intern_array(TermId::new(1));
        let i = solver.intern(TermId::new(2));
        let j = solver.intern(TermId::new(3));
        let v = solver.intern(TermId::new(4));

        // Assert i != j
        solver.assert_diseq(i, j, TermId::new(100));

        // store(a, i, v)
        let a_store = solver.intern_store(TermId::new(10), a, i, v);

        // select(a, j) - original array at j
        let select_a_j = solver.intern_select(TermId::new(11), a, j);

        // select(store(a, i, v), j) - should equal select(a, j)
        let select_store_j = solver.intern_select(TermId::new(12), a_store, j);

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));

        // After read-over-write-diff lemma, selects should be equal
        assert!(solver.are_equal(select_a_j, select_store_j));
    }

    #[test]
    fn test_array_conflict() {
        let mut solver = ArraySolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));

        // Assert a != b
        solver.assert_diseq(a, b, TermId::new(100));

        // Then merge a = b
        solver.merge(a, b, TermId::new(101)).unwrap();

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Unsat(_)));
    }

    #[test]
    fn test_array_push_pop() {
        let mut solver = ArraySolver::new();

        let a = solver.intern_array(TermId::new(1));
        let i = solver.intern(TermId::new(2));
        let v = solver.intern(TermId::new(3));

        solver.push();

        let _a_store = solver.intern_store(TermId::new(10), a, i, v);
        assert_eq!(solver.stores.len(), 1);

        solver.pop();

        assert_eq!(solver.stores.len(), 0);
    }

    #[test]
    fn test_incremental_merge_undo() {
        let mut solver = ArraySolver::new();

        let a = solver.intern(TermId::new(1));
        let b = solver.intern(TermId::new(2));
        let c = solver.intern(TermId::new(3));

        // Before merge, a and b are different
        assert!(!solver.are_equal(a, b));

        solver.push();

        // Merge a and b
        solver.merge(a, b, TermId::new(100)).unwrap();
        assert!(solver.are_equal(a, b));

        solver.push();

        // Merge b and c
        solver.merge(b, c, TermId::new(101)).unwrap();
        assert!(solver.are_equal(a, c));

        // Pop should undo b=c but keep a=b
        solver.pop();
        assert!(solver.are_equal(a, b));
        assert!(!solver.are_equal(a, c));

        // Pop should undo a=b
        solver.pop();
        assert!(!solver.are_equal(a, b));
    }

    #[test]
    fn test_incremental_with_pending_lemmas() {
        let mut solver = ArraySolver::new();

        let a = solver.intern_array(TermId::new(1));
        let i = solver.intern(TermId::new(2));
        let v = solver.intern(TermId::new(3));

        solver.push();

        // Create store which adds a pending lemma
        let _a_store = solver.intern_store(TermId::new(10), a, i, v);
        let initial_lemmas = solver.pending_lemmas.len();
        assert!(initial_lemmas > 0);

        solver.push();

        // Add another store
        let j = solver.intern(TermId::new(4));
        let w = solver.intern(TermId::new(5));
        let _a_store2 = solver.intern_store(TermId::new(11), a, j, w);
        assert!(solver.pending_lemmas.len() > initial_lemmas);

        // Pop should restore lemmas count
        solver.pop();
        assert_eq!(solver.pending_lemmas.len(), initial_lemmas);

        solver.pop();
        assert_eq!(solver.pending_lemmas.len(), 0);
    }
}
