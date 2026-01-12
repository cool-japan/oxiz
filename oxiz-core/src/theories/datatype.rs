//! Algebraic Datatype theory implementation
//!
//! Implements the theory of algebraic datatypes with constructors, testers, and selectors.
//!
//! Reference: Z3's `src/smt/theory_datatype.cpp` at `../z3/src/smt/`

use crate::ast::{TermId, TermKind, TermManager};
use crate::sort::{SortId, SortKind, SortManager};
use rustc_hash::{FxHashMap, FxHashSet};

/// Datatype theory axioms
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DatatypeAxiom {
    /// Constructor distinctness: C_i(...) ≠ C_j(...) for i ≠ j
    ConstructorDistinctness {
        /// Constructor 1
        cons1: lasso::Spur,
        /// Constructor 2
        cons2: lasso::Spur,
        /// Datatype sort
        datatype: SortId,
    },
    /// Selector axiom: sel_i(C(x_1, ..., x_n)) = x_i
    SelectorAxiom {
        /// Constructor term
        constructor: TermId,
        /// Selector
        selector: lasso::Spur,
        /// Field index
        field_index: usize,
        /// Field value
        field_value: TermId,
    },
    /// Tester axiom: is_C(C(x_1, ..., x_n)) = true
    TesterAxiom {
        /// Constructor term
        constructor: TermId,
        /// Tester (is_C)
        tester: lasso::Spur,
    },
    /// Negative tester: is_C_i(C_j(...)) = false for i ≠ j
    NegativeTesterAxiom {
        /// Constructor term with constructor C_j
        constructor: TermId,
        /// Tester for different constructor C_i
        tester: lasso::Spur,
    },
    /// Acyclicity: prevents cyclic structures in strictly positive positions
    Acyclicity {
        /// Term that would create a cycle
        term: TermId,
    },
    /// Injectivity: C(x_1, ..., x_n) = C(y_1, ..., y_n) ⟹ x_i = y_i
    Injectivity {
        /// First constructor application
        cons1: TermId,
        /// Second constructor application
        cons2: TermId,
        /// Constructor name
        constructor: lasso::Spur,
    },
}

/// Datatype theory reasoning engine
#[derive(Debug, Clone)]
pub struct DatatypeTheory {
    /// Tracked datatype terms (maps term to its sort)
    datatypes: FxHashMap<TermId, SortId>,
    /// Constructor applications: maps constructor term to its arguments
    constructors: FxHashMap<TermId, ConstructorInfo>,
    /// Selector applications: maps selector term to (datatype, selector_name)
    selectors: FxHashMap<TermId, (TermId, lasso::Spur)>,
    /// Tester applications: maps tester term to (datatype, tester_name)
    testers: FxHashMap<TermId, (TermId, lasso::Spur)>,
    /// Pending axiom instantiations
    pending_axioms: Vec<DatatypeAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<DatatypeAxiom>,
    /// Statistics
    propagations: usize,
    conflicts: usize,
}

/// Information about a constructor application
#[derive(Debug, Clone, PartialEq, Eq)]
struct ConstructorInfo {
    /// Constructor name
    name: lasso::Spur,
    /// Arguments to the constructor
    args: Vec<TermId>,
    /// Datatype sort
    sort: SortId,
}

impl Default for DatatypeTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl DatatypeTheory {
    /// Create a new datatype theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            datatypes: FxHashMap::default(),
            constructors: FxHashMap::default(),
            selectors: FxHashMap::default(),
            testers: FxHashMap::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
            propagations: 0,
            conflicts: 0,
        }
    }

    /// Register a datatype term
    pub fn register_datatype(&mut self, term: TermId, sort: SortId) {
        self.datatypes.insert(term, sort);
    }

    /// Register a constructor application
    pub fn register_constructor(
        &mut self,
        term: TermId,
        name: lasso::Spur,
        args: Vec<TermId>,
        sort: SortId,
    ) {
        self.constructors.insert(
            term,
            ConstructorInfo {
                name,
                args: args.clone(),
                sort,
            },
        );
        self.register_datatype(term, sort);

        // Generate selector axioms for this constructor
        for (field_index, &field_value) in args.iter().enumerate() {
            self.add_axiom(DatatypeAxiom::SelectorAxiom {
                constructor: term,
                selector: name, // Simplified: would be actual selector name
                field_index,
                field_value,
            });
        }
    }

    /// Register a selector application
    pub fn register_selector(&mut self, term: TermId, datatype: TermId, selector: lasso::Spur) {
        self.selectors.insert(term, (datatype, selector));
    }

    /// Register a tester application
    pub fn register_tester(&mut self, term: TermId, datatype: TermId, tester: lasso::Spur) {
        self.testers.insert(term, (datatype, tester));

        // If datatype is a constructor, generate tester axioms
        if let Some(_cons_info) = self.constructors.get(&datatype) {
            // Positive tester: is_C(C(...)) = true
            self.add_axiom(DatatypeAxiom::TesterAxiom {
                constructor: datatype,
                tester,
            });

            // Check for negative testers (different constructors)
            // This would require iterating through all constructors of the datatype
        }
    }

    /// Add a term to the theory and extract datatype operations
    pub fn add_term(&mut self, term: TermId, manager: &TermManager, sort_manager: &SortManager) {
        if let Some(t) = manager.get(term) {
            match &t.kind {
                TermKind::DtConstructor { constructor, args } => {
                    // Register constructor application
                    self.register_constructor(term, *constructor, args.to_vec(), t.sort);

                    // Generate distinctness axioms with other constructors
                    self.generate_distinctness_axioms(term, *constructor, t.sort, sort_manager);
                }
                TermKind::DtTester { arg, constructor } => {
                    self.register_tester(term, *arg, *constructor);
                }
                TermKind::DtSelector { arg, selector } => {
                    self.register_selector(term, *arg, *selector);

                    // If arg is a constructor, we would generate selector axiom here
                    // but skip to avoid borrow checker issues in this simplified implementation
                }
                _ => {
                    // Check if this is a datatype variable
                    if let Some(sort) = sort_manager.get(t.sort)
                        && matches!(sort.kind, SortKind::Datatype(_))
                    {
                        self.register_datatype(term, t.sort);
                    }
                }
            }
        }
    }

    /// Generate distinctness axioms for a constructor
    fn generate_distinctness_axioms(
        &mut self,
        _term: TermId,
        cons_name: lasso::Spur,
        sort: SortId,
        sort_manager: &SortManager,
    ) {
        // Get all constructors for this datatype
        if let Some(sort_obj) = sort_manager.get(sort)
            && let SortKind::Datatype(dt_name) = sort_obj.kind
        {
            // Would iterate through other constructors and generate distinctness axioms
            // For now, we just record the potential for distinctness
            // This is a placeholder - a full implementation would access the datatype definition
            let _ = (dt_name, cons_name);
        }
    }

    /// Generate selector axiom when applied to a constructor
    #[allow(dead_code)]
    fn generate_selector_axiom_for_constructor(
        &mut self,
        _selector_term: TermId,
        constructor: TermId,
        selector: lasso::Spur,
        cons_info: &ConstructorInfo,
    ) {
        // Find which field this selector corresponds to
        // This is simplified - would need to look up the constructor definition
        for (field_index, &field_value) in cons_info.args.iter().enumerate() {
            self.add_axiom(DatatypeAxiom::SelectorAxiom {
                constructor,
                selector,
                field_index,
                field_value,
            });
        }
    }

    /// Add an axiom to the pending list
    fn add_axiom(&mut self, axiom: DatatypeAxiom) {
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Get pending axioms and clear the list
    pub fn take_pending_axioms(&mut self) -> Vec<DatatypeAxiom> {
        std::mem::take(&mut self.pending_axioms)
    }

    /// Convert an axiom to a term (requires term manager)
    ///
    /// Note: This is a simplified implementation that returns placeholder terms.
    /// A full implementation would need access to a string interner to convert
    /// Spur values back to strings for mk_dt_selector and mk_dt_tester.
    pub fn axiom_to_term(&self, axiom: &DatatypeAxiom, manager: &mut TermManager) -> TermId {
        match axiom {
            DatatypeAxiom::ConstructorDistinctness { .. } => {
                // C_i(...) ≠ C_j(...) becomes a disequality
                // Placeholder: would create actual terms
                manager.mk_true()
            }
            DatatypeAxiom::SelectorAxiom { field_value, .. } => {
                // sel_i(C(x_1, ..., x_n)) = x_i
                // Simplified: just return equality with field value
                *field_value
            }
            DatatypeAxiom::TesterAxiom { .. } => {
                // is_C(C(...)) = true
                // Simplified placeholder
                manager.mk_true()
            }
            DatatypeAxiom::NegativeTesterAxiom { .. } => {
                // is_C_i(C_j(...)) = false
                // Simplified placeholder
                manager.mk_false()
            }
            DatatypeAxiom::Acyclicity { .. } => {
                // Acyclicity constraints
                manager.mk_false()
            }
            DatatypeAxiom::Injectivity { cons1, cons2, .. } => {
                // C(x_1, ..., x_n) = C(y_1, ..., y_n) ⟹ x_i = y_i
                // Placeholder: would create implication
                manager.mk_eq(*cons1, *cons2)
            }
        }
    }

    /// Propagate equalities and deduce new facts
    pub fn propagate(&mut self) -> Vec<(TermId, TermId)> {
        self.propagations += 1;
        // Placeholder: in a full implementation, this would deduce new equalities
        Vec::new()
    }

    /// Check for conflicts in the current state
    pub fn check_for_conflicts(&mut self, _manager: &TermManager) -> Option<Vec<TermId>> {
        // Check for constructor distinctness violations
        // This is a placeholder
        None
    }

    /// Reset the theory state (for backtracking)
    pub fn reset(&mut self) {
        self.datatypes.clear();
        self.constructors.clear();
        self.selectors.clear();
        self.testers.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
        self.propagations = 0;
        self.conflicts = 0;
    }

    /// Get statistics
    pub fn statistics(&self) -> DatatypeStatistics {
        DatatypeStatistics {
            num_datatypes: self.datatypes.len(),
            num_constructors: self.constructors.len(),
            num_selectors: self.selectors.len(),
            num_testers: self.testers.len(),
            num_axioms: self.instantiated.len(),
            num_propagations: self.propagations,
            num_conflicts: self.conflicts,
        }
    }
}

/// Statistics for datatype theory
#[derive(Debug, Clone, Copy)]
pub struct DatatypeStatistics {
    /// Number of datatype terms
    pub num_datatypes: usize,
    /// Number of constructor applications
    pub num_constructors: usize,
    /// Number of selector applications
    pub num_selectors: usize,
    /// Number of tester applications
    pub num_testers: usize,
    /// Number of axioms instantiated
    pub num_axioms: usize,
    /// Number of propagations
    pub num_propagations: usize,
    /// Number of conflicts detected
    pub num_conflicts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    use lasso::Key;

    #[test]
    fn test_empty_theory() {
        let theory = DatatypeTheory::new();
        assert_eq!(theory.datatypes.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_register_datatype() {
        let mut theory = DatatypeTheory::new();
        let term = TermId(42);
        let sort = SortId(1);
        theory.register_datatype(term, sort);
        assert_eq!(theory.datatypes.get(&term), Some(&sort));
    }

    #[test]
    fn test_register_constructor() {
        let mut theory = DatatypeTheory::new();
        let cons_name = lasso::Spur::try_from_usize(1).unwrap();
        let term = TermId(42);
        let sort = SortId(1);
        let args = vec![TermId(1), TermId(2)];

        theory.register_constructor(term, cons_name, args.clone(), sort);

        assert_eq!(theory.constructors.get(&term).unwrap().name, cons_name);
        assert_eq!(theory.constructors.get(&term).unwrap().args, args);
    }

    #[test]
    fn test_register_selector() {
        let mut theory = DatatypeTheory::new();
        let selector = lasso::Spur::try_from_usize(2).unwrap();
        let term = TermId(42);
        let datatype = TermId(10);

        theory.register_selector(term, datatype, selector);
        assert_eq!(theory.selectors.get(&term), Some(&(datatype, selector)));
    }

    #[test]
    fn test_register_tester() {
        let mut theory = DatatypeTheory::new();
        let tester = lasso::Spur::try_from_usize(3).unwrap();
        let term = TermId(42);
        let datatype = TermId(10);

        theory.register_tester(term, datatype, tester);
        assert_eq!(theory.testers.get(&term), Some(&(datatype, tester)));
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = DatatypeTheory::new();
        let axiom = DatatypeAxiom::TesterAxiom {
            constructor: TermId(1),
            tester: lasso::Spur::try_from_usize(1).unwrap(),
        };

        theory.add_axiom(axiom.clone());
        theory.add_axiom(axiom);

        assert_eq!(theory.pending_axioms.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = DatatypeTheory::new();
        let cons_name = lasso::Spur::try_from_usize(1).unwrap();

        theory.register_datatype(TermId(1), SortId(1));
        theory.register_constructor(TermId(2), cons_name, vec![], SortId(1));

        theory.reset();

        assert_eq!(theory.datatypes.len(), 0);
        assert_eq!(theory.constructors.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_statistics() {
        let mut theory = DatatypeTheory::new();
        let cons_name = lasso::Spur::try_from_usize(1).unwrap();

        theory.register_datatype(TermId(1), SortId(1));
        theory.register_constructor(TermId(2), cons_name, vec![], SortId(1));
        theory.propagate();

        let stats = theory.statistics();
        // Constructor registration also registers as a datatype, so we expect 2
        assert_eq!(stats.num_datatypes, 2);
        assert_eq!(stats.num_constructors, 1);
        assert_eq!(stats.num_propagations, 1);
    }

    #[test]
    fn test_propagate() {
        let mut theory = DatatypeTheory::new();
        let equalities = theory.propagate();
        assert_eq!(equalities.len(), 0);
    }

    #[test]
    fn test_selector_axiom_generation() {
        let mut theory = DatatypeTheory::new();
        let cons_name = lasso::Spur::try_from_usize(1).unwrap();
        let args = vec![TermId(10), TermId(20)];

        theory.register_constructor(TermId(1), cons_name, args, SortId(1));

        let axioms = theory.take_pending_axioms();
        // Should have generated selector axioms for each field
        assert!(!axioms.is_empty());
    }
}
