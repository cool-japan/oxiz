//! Array theory implementation
//!
//! Implements the theory of arrays with extensionality and read-over-write axioms.
//!
//! Reference: Z3's `src/smt/theory_array.cpp`

use super::combination::{Theory, TheoryResult};
use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortKind, SortManager};

/// Array theory axioms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ArrayAxiom {
    /// Read-over-write axiom (select (store a i v) i) = v
    ReadOverWrite {
        /// Array term
        array: TermId,
        /// Index
        index: TermId,
        /// Value
        value: TermId,
    },
    /// Read-over-write with different indices
    /// i ≠ j ⟹ (select (store a i v) j) = (select a j)
    ReadOverWriteDiff {
        /// Array term
        array: TermId,
        /// First index
        index1: TermId,
        /// Second index (different from index1)
        index2: TermId,
        /// Value
        value: TermId,
    },
    /// Array extensionality axiom
    /// ∀i. (select a i) = (select b i) ⟹ a = b
    Extensionality {
        /// First array
        array1: TermId,
        /// Second array
        array2: TermId,
        /// Skolem witness index for this pair, if one has been chosen
        ///
        /// `None` asks for the quantified axiom. `Some(w)` asks for its
        /// Skolemization, which is only sound when `w` is a *witness chosen
        /// for this pair* — the index at which the two arrays differ if they
        /// differ at all, i.e. a fresh constant. An arbitrary existing index
        /// is not a witness, and the Skolemized form is false for one.
        /// [`ArrayTheory::generate_extensionality`] records `None`.
        witness: Option<TermId>,
    },
}

/// Array theory reasoning engine
#[derive(Debug, Clone)]
pub struct ArrayTheory {
    /// Tracked array terms (maps array term to its sort)
    arrays: FxHashMap<TermId, SortId>,
    /// Select terms: maps (array, index) to select term
    selects: FxHashMap<(TermId, TermId), TermId>,
    /// Store terms: maps (array, index, value) to store term
    stores: FxHashMap<(TermId, TermId, TermId), TermId>,
    /// Pending axiom instantiations
    pending_axioms: Vec<ArrayAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<ArrayAxiom>,
}

impl ArrayTheory {
    /// Create a new array theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            arrays: FxHashMap::default(),
            selects: FxHashMap::default(),
            stores: FxHashMap::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
        }
    }

    /// Register an array term
    pub fn register_array(&mut self, array: TermId, sort: SortId) {
        self.arrays.insert(array, sort);
    }

    /// Register a select term
    pub fn register_select(&mut self, select_term: TermId, array: TermId, index: TermId) {
        self.selects.insert((array, index), select_term);
    }

    /// Register a store term
    pub fn register_store(
        &mut self,
        _store_term: TermId,
        array: TermId,
        index: TermId,
        value: TermId,
    ) {
        self.stores.insert((array, index, value), _store_term);
    }

    /// Add a term to the theory and extract array operations
    pub fn add_term(&mut self, term: TermId, manager: &TermManager, sort_manager: &SortManager) {
        if let Some(t) = manager.get(term) {
            match &t.kind {
                TermKind::Select(array, index) => {
                    self.register_select(term, *array, *index);

                    // Check if array is a store and generate axioms
                    if let Some(array_term) = manager.get(*array)
                        && let TermKind::Store(base_array, store_idx, store_val) = array_term.kind
                    {
                        // Generate read-over-write axioms
                        self.generate_read_over_write_axioms(
                            *array, base_array, store_idx, store_val, *index,
                        );
                    }
                }
                TermKind::Store(array, index, value) => {
                    self.register_store(term, *array, *index, *value);

                    // Register the array (store creates an array)
                    let sort_id = t.sort;
                    self.register_array(term, sort_id);

                    // Check for existing selects and generate axioms
                    self.generate_axioms_for_store(*array, *index, *value);
                }
                TermKind::Var(_) => {
                    // Check if this is an array variable
                    let sort_id = t.sort;
                    if let Some(sort) = sort_manager.get(sort_id)
                        && matches!(sort.kind, SortKind::Array { .. })
                    {
                        self.register_array(term, sort_id);
                    }
                }
                _ => {}
            }
        }
    }

    /// Generate read-over-write axioms for a select on a store
    fn generate_read_over_write_axioms(
        &mut self,
        _store_term: TermId,
        base_array: TermId,
        store_idx: TermId,
        store_val: TermId,
        select_idx: TermId,
    ) {
        // Always generate the read-over-write axiom for same index
        if store_idx == select_idx {
            let axiom = ArrayAxiom::ReadOverWrite {
                array: base_array,
                index: store_idx,
                value: store_val,
            };
            if self.instantiated.insert(axiom.clone()) {
                self.pending_axioms.push(axiom);
            }
        } else {
            // Generate axiom for different indices
            let axiom = ArrayAxiom::ReadOverWriteDiff {
                array: base_array,
                index1: store_idx,
                index2: select_idx,
                value: store_val,
            };
            if self.instantiated.insert(axiom.clone()) {
                self.pending_axioms.push(axiom);
            }
        }
    }

    /// Generate axioms when a new store is encountered
    fn generate_axioms_for_store(&mut self, array: TermId, index: TermId, value: TermId) {
        // For each existing select on this array, generate relevant axioms
        let selects_on_array: Vec<_> = self
            .selects
            .iter()
            .filter(|((arr, _), _)| *arr == array)
            .map(|((_, idx), _)| *idx)
            .collect();

        for select_idx in selects_on_array {
            if select_idx == index {
                let axiom = ArrayAxiom::ReadOverWrite {
                    array,
                    index,
                    value,
                };
                if self.instantiated.insert(axiom.clone()) {
                    self.pending_axioms.push(axiom);
                }
            } else {
                let axiom = ArrayAxiom::ReadOverWriteDiff {
                    array,
                    index1: index,
                    index2: select_idx,
                    value,
                };
                if self.instantiated.insert(axiom.clone()) {
                    self.pending_axioms.push(axiom);
                }
            }
        }
    }

    /// Generate extensionality axiom for two arrays
    pub fn generate_extensionality(&mut self, array1: TermId, array2: TermId) {
        let axiom = ArrayAxiom::Extensionality {
            array1,
            array2,
            witness: None,
        };
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Convert an array axiom to a term
    ///
    /// This creates the SMT formula corresponding to the axiom.
    ///
    /// Returns `None` for an [`ArrayAxiom::Extensionality`] with no witness
    /// whose arrays' domain sort cannot be read from `manager` — the
    /// quantified form needs it to bind the index.
    pub fn axiom_to_term(&self, axiom: &ArrayAxiom, manager: &mut TermManager) -> Option<TermId> {
        Some(match axiom {
            ArrayAxiom::ReadOverWrite {
                array,
                index,
                value,
            } => {
                // (select (store array index value) index) = value
                let store = manager.mk_store(*array, *index, *value);
                let select = manager.mk_select(store, *index);
                manager.mk_eq(select, *value)
            }
            ArrayAxiom::ReadOverWriteDiff {
                array,
                index1,
                index2,
                value,
            } => {
                // (index1 ≠ index2) ⟹ (select (store array index1 value) index2) = (select array index2)
                let store = manager.mk_store(*array, *index1, *value);
                let select_store = manager.mk_select(store, *index2);
                let select_array = manager.mk_select(*array, *index2);
                let equality = manager.mk_eq(select_store, select_array);

                // Create implication: (index1 ≠ index2) ⟹ equality
                let idx_eq = manager.mk_eq(*index1, *index2);
                let not_eq = manager.mk_not(idx_eq);
                manager.mk_implies(not_eq, equality)
            }
            ArrayAxiom::Extensionality {
                array1,
                array2,
                witness,
            } => {
                // With a Skolem witness w for the pair:
                //   (select array1 w) = (select array2 w) ⟹ array1 = array2
                // Without one, the quantified axiom itself:
                //   (∀i. (select array1 i) = (select array2 i)) ⟹ array1 = array2
                let index = match witness {
                    Some(index) => *index,
                    None => {
                        let domain = self.domain_sort_of(*array1, manager)?;
                        // A name no source-level variable can collide with.
                        manager.mk_var("array!ext!i", domain)
                    }
                };

                let select1 = manager.mk_select(*array1, index);
                let select2 = manager.mk_select(*array2, index);
                let select_eq = manager.mk_eq(select1, select2);
                let array_eq = manager.mk_eq(*array1, *array2);

                let premise = if witness.is_some() {
                    select_eq
                } else {
                    let domain = self.domain_sort_of(*array1, manager)?;
                    manager.mk_forall([("array!ext!i", domain)], select_eq)
                };
                manager.mk_implies(premise, array_eq)
            }
        })
    }

    /// Domain sort of an array term, when it has an array sort
    fn domain_sort_of(&self, array: TermId, manager: &TermManager) -> Option<SortId> {
        let sort = manager.get(array)?.sort;
        match manager.sorts.get(sort)?.kind {
            SortKind::Array { domain, .. } => Some(domain),
            _ => None,
        }
    }

    /// Get all pending axioms
    #[must_use]
    pub fn get_pending_axioms(&self) -> &[ArrayAxiom] {
        &self.pending_axioms
    }

    /// Clear pending axioms
    pub fn clear_pending(&mut self) {
        self.pending_axioms.clear();
    }

    /// Get all pending axioms as terms and clear the queue
    ///
    /// Axioms that [`ArrayTheory::axiom_to_term`] cannot build stay queued
    /// rather than being dropped, so a caller that inspects
    /// [`ArrayTheory::get_pending_axioms`] afterwards still sees them.
    pub fn propagate(&mut self, manager: &mut TermManager) -> Vec<TermId> {
        let queued = core::mem::take(&mut self.pending_axioms);

        let mut terms = Vec::with_capacity(queued.len());
        for axiom in queued {
            match self.axiom_to_term(&axiom, manager) {
                Some(term) => terms.push(term),
                None => self.pending_axioms.push(axiom),
            }
        }

        terms
    }

    /// Check whether two arrays are *provably* equal from the registered selects
    ///
    /// This is an equality-establishing predicate, so the conservative
    /// direction is `false`: a `true` answer asserts that the two arrays agree,
    /// and every caller may act on that assertion. It therefore requires
    /// positive evidence for every known index —
    ///
    /// * both arrays must have a registered select at that index, and
    /// * the two select terms must be identical.
    ///
    /// Missing evidence — no index is known at all, or only one of the two
    /// arrays has a select at some index — yields `false` ("not proven
    /// equal"), not `true`. The previous direction returned `true` for exactly
    /// those cases, i.e. it claimed extensional equality for two arrays about
    /// which nothing was known.
    ///
    /// Note that `false` means "not proven equal", never "proven different";
    /// callers must not read it as a disequality.
    pub fn check_extensionality(
        &self,
        array1: TermId,
        array2: TermId,
        _manager: &TermManager,
    ) -> bool {
        if array1 == array2 {
            return true;
        }

        // Get all indices used with these arrays
        let mut indices = FxHashSet::default();

        for (arr, idx) in self.selects.keys() {
            if *arr == array1 || *arr == array2 {
                indices.insert(*idx);
            }
        }

        // With no shared index there is no evidence of agreement at all.
        if indices.is_empty() {
            return false;
        }

        indices.iter().all(|idx| {
            match (
                self.selects.get(&(array1, *idx)),
                self.selects.get(&(array2, *idx)),
            ) {
                // Structural equality of the select terms is the only
                // evidence available without a model.
                (Some(&s1), Some(&s2)) => s1 == s2,
                // One side has no select at this index: nothing is known
                // about the value there, so agreement is not established.
                _ => false,
            }
        })
    }

    /// Get statistics about the theory
    #[must_use]
    pub fn statistics(&self) -> ArrayTheoryStats {
        ArrayTheoryStats {
            num_arrays: self.arrays.len(),
            num_selects: self.selects.len(),
            num_stores: self.stores.len(),
            num_axioms_instantiated: self.instantiated.len(),
            num_pending_axioms: self.pending_axioms.len(),
        }
    }

    /// Reset the theory state
    pub fn reset(&mut self) {
        self.arrays.clear();
        self.selects.clear();
        self.stores.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
    }
}

impl Theory for ArrayTheory {
    fn add_term(&mut self, term: TermId, manager: &TermManager) -> bool {
        ArrayTheory::add_term(self, term, manager, &manager.sorts);

        manager.get(term).is_some_and(|t| {
            matches!(t.kind, TermKind::Select(_, _) | TermKind::Store(_, _, _))
                || manager
                    .sorts
                    .get(t.sort)
                    .is_some_and(|sort| matches!(sort.kind, SortKind::Array { .. }))
        })
    }

    /// This theory contributes axiom instances and does not track equality
    /// classes, so an equality is never new information for it: this always
    /// returns `false`.
    fn assert_equality(&mut self, _a: TermId, _b: TermId) -> bool {
        false
    }

    fn check(&mut self, manager: &mut TermManager) -> TheoryResult {
        let lemmas = self.propagate(manager);

        if lemmas.is_empty() {
            TheoryResult::Sat
        } else {
            TheoryResult::Lemmas(lemmas)
        }
    }

    fn name(&self) -> &str {
        "array"
    }

    fn reset(&mut self) {
        ArrayTheory::reset(self);
    }
}

/// Statistics about array theory reasoning
#[derive(Debug, Default, Clone)]
pub struct ArrayTheoryStats {
    /// Number of array terms
    pub num_arrays: usize,
    /// Number of select operations
    pub num_selects: usize,
    /// Number of store operations
    pub num_stores: usize,
    /// Number of axioms instantiated
    pub num_axioms_instantiated: usize,
    /// Number of pending axioms
    pub num_pending_axioms: usize,
}

impl Default for ArrayTheory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;
    use crate::sort::SortManager;

    #[test]
    fn test_empty_theory() {
        let theory = ArrayTheory::new();
        assert_eq!(theory.arrays.len(), 0);
        assert_eq!(theory.selects.len(), 0);
        assert_eq!(theory.stores.len(), 0);
    }

    #[test]
    fn test_register_select() {
        let mut theory = ArrayTheory::new();
        let array = TermId(1);
        let index = TermId(2);
        let select = TermId(3);

        theory.register_select(select, array, index);
        assert_eq!(theory.selects.len(), 1);
        assert_eq!(theory.selects.get(&(array, index)), Some(&select));
    }

    #[test]
    fn test_register_store() {
        let mut theory = ArrayTheory::new();
        let array = TermId(1);
        let index = TermId(2);
        let value = TermId(3);
        let store = TermId(4);

        theory.register_store(store, array, index, value);
        assert_eq!(theory.stores.len(), 1);
        assert_eq!(theory.stores.get(&(array, index, value)), Some(&store));
    }

    #[test]
    fn test_read_over_write_axiom() {
        let mut manager = TermManager::new();
        let mut theory = ArrayTheory::new();

        // Create array variable
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", array_sort);

        // Create index and value
        let i = manager.mk_var("i", int_sort);
        let v = manager.mk_var("v", int_sort);

        // Create store and select
        let store = manager.mk_store(a, i, v);
        let select = manager.mk_select(store, i);

        // Process terms
        let sort_manager = SortManager::new();
        theory.add_term(store, &manager, &sort_manager);
        theory.add_term(select, &manager, &sort_manager);

        // Should generate read-over-write axiom
        assert!(!theory.pending_axioms.is_empty());
    }

    #[test]
    fn test_axiom_to_term_read_over_write() {
        let mut manager = TermManager::new();
        let theory = ArrayTheory::new();

        let array = TermId(1);
        let index = TermId(2);
        let value = TermId(3);

        let axiom = ArrayAxiom::ReadOverWrite {
            array,
            index,
            value,
        };

        let term = theory
            .axiom_to_term(&axiom, &mut manager)
            .expect("read-over-write should always be buildable");

        // Should create: (select (store array index value) index) = value
        assert!(manager.get(term).is_some());
    }

    #[test]
    fn test_extensionality_without_a_witness_is_quantified() {
        let mut manager = TermManager::new();
        let theory = ArrayTheory::new();

        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", array_sort);
        let b = manager.mk_var("b", array_sort);

        let term = theory
            .axiom_to_term(
                &ArrayAxiom::Extensionality {
                    array1: a,
                    array2: b,
                    witness: None,
                },
                &mut manager,
            )
            .expect("the quantified axiom should be buildable");

        // (forall ((i Int)) (= (select a i) (select b i))) => (= a b)
        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Implies(premise, conclusion)) => {
                assert!(
                    matches!(
                        manager.get(premise).map(|t| t.kind.clone()),
                        Some(TermKind::Forall { .. })
                    ),
                    "the premise must quantify over the index, not fix one"
                );
                match manager.get(conclusion).map(|t| t.kind.clone()) {
                    Some(TermKind::Eq(lhs, rhs)) => {
                        assert!((lhs == a && rhs == b) || (lhs == b && rhs == a));
                    }
                    other => panic!("expected the array equality, got {other:?}"),
                }
            }
            other => panic!("expected an implication, got {other:?}"),
        }
    }

    #[test]
    fn test_extensionality_with_a_witness_is_skolemized() {
        let mut manager = TermManager::new();
        let theory = ArrayTheory::new();

        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", array_sort);
        let b = manager.mk_var("b", array_sort);
        let witness = manager.mk_var("w", int_sort);

        let term = theory
            .axiom_to_term(
                &ArrayAxiom::Extensionality {
                    array1: a,
                    array2: b,
                    witness: Some(witness),
                },
                &mut manager,
            )
            .expect("the Skolemized axiom should be buildable");

        // (= (select a w) (select b w)) => (= a b): an implication, never the
        // bare equality the arrays are not known to satisfy.
        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Implies(premise, _)) => {
                assert!(matches!(
                    manager.get(premise).map(|t| t.kind.clone()),
                    Some(TermKind::Eq(_, _))
                ));
            }
            other => panic!("expected an implication, got {other:?}"),
        }
    }

    #[test]
    fn test_generate_extensionality_records_no_witness() {
        let mut theory = ArrayTheory::new();
        theory.generate_extensionality(TermId(1), TermId(2));

        assert_eq!(
            theory.get_pending_axioms(),
            [ArrayAxiom::Extensionality {
                array1: TermId(1),
                array2: TermId(2),
                witness: None,
            }]
        );
    }

    #[test]
    fn test_statistics() {
        let mut theory = ArrayTheory::new();

        theory.register_array(TermId(1), SortId(0));
        theory.register_select(TermId(2), TermId(1), TermId(3));
        theory.register_store(TermId(4), TermId(1), TermId(3), TermId(5));

        let stats = theory.statistics();
        assert_eq!(stats.num_arrays, 1);
        assert_eq!(stats.num_selects, 1);
        assert_eq!(stats.num_stores, 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = ArrayTheory::new();

        theory.register_array(TermId(1), SortId(0));
        theory.register_select(TermId(2), TermId(1), TermId(3));

        assert!(!theory.arrays.is_empty());

        theory.reset();

        assert!(theory.arrays.is_empty());
        assert!(theory.selects.is_empty());
        assert!(theory.stores.is_empty());
    }

    #[test]
    fn test_propagate_clears_pending() {
        let mut manager = TermManager::new();
        let mut theory = ArrayTheory::new();

        let axiom = ArrayAxiom::ReadOverWrite {
            array: TermId(1),
            index: TermId(2),
            value: TermId(3),
        };

        theory.pending_axioms.push(axiom);
        assert_eq!(theory.pending_axioms.len(), 1);

        let terms = theory.propagate(&mut manager);
        assert_eq!(terms.len(), 1);
        assert!(theory.pending_axioms.is_empty());
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = ArrayTheory::new();

        let axiom = ArrayAxiom::ReadOverWrite {
            array: TermId(1),
            index: TermId(2),
            value: TermId(3),
        };

        // Add the same axiom twice
        if theory.instantiated.insert(axiom.clone()) {
            theory.pending_axioms.push(axiom.clone());
        }

        if theory.instantiated.insert(axiom.clone()) {
            theory.pending_axioms.push(axiom);
        }

        // Should only have one copy
        assert_eq!(theory.pending_axioms.len(), 1);
        assert_eq!(theory.instantiated.len(), 1);
    }

    #[test]
    fn test_extensionality_without_evidence_is_not_proven() {
        let manager = TermManager::new();
        let theory = ArrayTheory::new();
        let a = TermId(1);
        let b = TermId(2);

        // Nothing at all is known about `a` and `b`. Claiming they agree
        // would let a caller merge two unrelated arrays.
        assert!(!theory.check_extensionality(a, b, &manager));
    }

    #[test]
    fn test_extensionality_requires_both_sides_at_every_index() {
        let manager = TermManager::new();
        let mut theory = ArrayTheory::new();
        let a = TermId(1);
        let b = TermId(2);
        let i = TermId(10);
        let select_a_i = TermId(20);

        // Only `a` has a select at `i`: the value of `b[i]` is unknown, so
        // agreement is not established.
        theory.register_select(select_a_i, a, i);
        assert!(!theory.check_extensionality(a, b, &manager));

        // A *different* select term at the same index is not agreement either.
        theory.register_select(TermId(21), b, i);
        assert!(!theory.check_extensionality(a, b, &manager));
    }

    #[test]
    fn test_extensionality_proven_when_all_selects_agree() {
        let manager = TermManager::new();
        let mut theory = ArrayTheory::new();
        let a = TermId(1);
        let b = TermId(2);
        let i = TermId(10);
        let j = TermId(11);
        let shared_i = TermId(20);
        let shared_j = TermId(21);

        theory.register_select(shared_i, a, i);
        theory.register_select(shared_i, b, i);
        theory.register_select(shared_j, a, j);
        theory.register_select(shared_j, b, j);

        assert!(theory.check_extensionality(a, b, &manager));

        // Adding a one-sided index withdraws the proof again.
        theory.register_select(TermId(22), a, TermId(12));
        assert!(!theory.check_extensionality(a, b, &manager));
    }

    #[test]
    fn test_extensionality_is_reflexive() {
        let manager = TermManager::new();
        let theory = ArrayTheory::new();
        let a = TermId(1);
        // An array always agrees with itself, even with no registered selects.
        assert!(theory.check_extensionality(a, a, &manager));
    }
}
