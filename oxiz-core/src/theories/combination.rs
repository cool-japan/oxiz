//! Nelson-Oppen theory combination
//!
//! Implements the Nelson-Oppen framework for combining decision procedures
//! for disjoint theories.
//!
//! Reference: Z3's `src/smt/theory.cpp` and Nelson-Oppen combination framework

use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};

/// Interface for a theory decision procedure
pub trait Theory {
    /// Add a term to the theory
    fn add_term(&mut self, term: TermId, manager: &TermManager);

    /// Check satisfiability and return new equalities or None if UNSAT
    fn check(&mut self, manager: &TermManager) -> TheoryResult;

    /// Get the name of the theory
    fn name(&self) -> &str;

    /// Reset the theory state
    fn reset(&mut self);
}

/// Result of a theory check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TheoryResult {
    /// Theory is satisfiable
    Sat,
    /// Theory found UNSAT
    Unsat,
    /// Theory propagates new equalities
    Propagate(Vec<(TermId, TermId)>),
}

/// Nelson-Oppen theory combiner
///
/// Combines multiple disjoint theories by exchanging equalities
/// between shared variables.
#[derive(Debug)]
pub struct NelsonOppen {
    /// Shared variables across theories (variables that appear in multiple theories)
    shared_vars: FxHashSet<TermId>,
    /// Variables belonging to each theory (theory index -> variables)
    theory_vars: Vec<FxHashSet<TermId>>,
    /// Known equalities between shared variables
    equalities: FxHashSet<(TermId, TermId)>,
    /// Pending equalities to propagate
    pending_equalities: Vec<(TermId, TermId)>,
    /// Number of theories
    num_theories: usize,
}

impl NelsonOppen {
    /// Create a new Nelson-Oppen combiner for the given number of theories
    #[must_use]
    pub fn new(num_theories: usize) -> Self {
        Self {
            shared_vars: FxHashSet::default(),
            theory_vars: vec![FxHashSet::default(); num_theories],
            equalities: FxHashSet::default(),
            pending_equalities: Vec::new(),
            num_theories,
        }
    }

    /// Register a variable with a specific theory
    pub fn register_var(&mut self, var: TermId, theory_idx: usize) {
        if theory_idx >= self.num_theories {
            return; // Invalid theory index
        }

        // Check if this variable is already in another theory
        for (idx, vars) in self.theory_vars.iter().enumerate() {
            if idx != theory_idx && vars.contains(&var) {
                // This is a shared variable
                self.shared_vars.insert(var);
                break;
            }
        }

        self.theory_vars[theory_idx].insert(var);
    }

    /// Add an equality between two terms
    pub fn add_equality(&mut self, a: TermId, b: TermId) {
        let eq = if a.0 < b.0 { (a, b) } else { (b, a) };

        if self.equalities.insert(eq) {
            // This is a new equality
            if self.shared_vars.contains(&a) || self.shared_vars.contains(&b) {
                // At least one of the terms is shared, propagate to all theories
                self.pending_equalities.push(eq);
            }
        }
    }

    /// Get pending equalities to propagate
    #[must_use]
    pub fn get_pending_equalities(&self) -> &[(TermId, TermId)] {
        &self.pending_equalities
    }

    /// Clear pending equalities
    pub fn clear_pending(&mut self) {
        self.pending_equalities.clear();
    }

    /// Get all shared variables
    #[must_use]
    pub fn shared_variables(&self) -> &FxHashSet<TermId> {
        &self.shared_vars
    }

    /// Check if a variable is shared across theories
    #[must_use]
    pub fn is_shared(&self, var: TermId) -> bool {
        self.shared_vars.contains(&var)
    }

    /// Get statistics about theory combination
    #[must_use]
    pub fn statistics(&self) -> CombinationStats {
        CombinationStats {
            num_theories: self.num_theories,
            num_shared_vars: self.shared_vars.len(),
            num_equalities: self.equalities.len(),
            num_pending_equalities: self.pending_equalities.len(),
        }
    }

    /// Reset the combiner state
    pub fn reset(&mut self) {
        self.shared_vars.clear();
        for vars in &mut self.theory_vars {
            vars.clear();
        }
        self.equalities.clear();
        self.pending_equalities.clear();
    }
}

/// Statistics for theory combination
#[derive(Debug, Default, Clone)]
pub struct CombinationStats {
    /// Number of theories being combined
    pub num_theories: usize,
    /// Number of shared variables
    pub num_shared_vars: usize,
    /// Number of known equalities
    pub num_equalities: usize,
    /// Number of pending equalities
    pub num_pending_equalities: usize,
}

/// Theory combiner that coordinates multiple theories
pub struct TheoryCombiner {
    /// Nelson-Oppen combiner
    nelson_oppen: NelsonOppen,
    /// Mapping of terms to theory indices
    term_to_theory: FxHashMap<TermId, usize>,
}

impl TheoryCombiner {
    /// Create a new theory combiner
    #[must_use]
    pub fn new(num_theories: usize) -> Self {
        Self {
            nelson_oppen: NelsonOppen::new(num_theories),
            term_to_theory: FxHashMap::default(),
        }
    }

    /// Classify a term and assign it to appropriate theories
    ///
    /// This analyzes the term structure and determines which theories
    /// should reason about it.
    pub fn classify_term(&mut self, term: TermId, manager: &TermManager) -> Vec<usize> {
        let mut theories = Vec::new();

        if let Some(t) = manager.get(term) {
            match &t.kind {
                // Boolean theory (theory 0)
                TermKind::True
                | TermKind::False
                | TermKind::Not(_)
                | TermKind::And(_)
                | TermKind::Or(_)
                | TermKind::Implies(_, _)
                | TermKind::Xor(_, _) => {
                    theories.push(0);
                }

                // Arithmetic theory (theory 1)
                TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::Add(_)
                | TermKind::Sub(_, _)
                | TermKind::Mul(_)
                | TermKind::Div(_, _)
                | TermKind::Mod(_, _)
                | TermKind::Neg(_)
                | TermKind::Lt(_, _)
                | TermKind::Le(_, _)
                | TermKind::Gt(_, _)
                | TermKind::Ge(_, _) => {
                    theories.push(1);
                }

                // Array theory (theory 2)
                TermKind::Select(_, _) | TermKind::Store(_, _, _) => {
                    theories.push(2);
                }

                // Bit vector theory (theory 3)
                TermKind::BitVecConst { .. } | TermKind::BvNot(_) | TermKind::BvAnd(_, _) => {
                    theories.push(3);
                }

                // Variables and equality can appear in any theory
                TermKind::Var(_) | TermKind::Eq(_, _) => {
                    // Register with all theories
                    for i in 0..self.nelson_oppen.num_theories {
                        theories.push(i);
                    }
                }

                _ => {}
            }
        }

        // Register this term with the identified theories
        for &theory_idx in &theories {
            self.term_to_theory.insert(term, theory_idx);

            // If it's a variable, register it with Nelson-Oppen
            if let Some(t) = manager.get(term)
                && matches!(t.kind, TermKind::Var(_))
            {
                self.nelson_oppen.register_var(term, theory_idx);
            }
        }

        theories
    }

    /// Add an equality to the combiner
    pub fn add_equality(&mut self, a: TermId, b: TermId) {
        self.nelson_oppen.add_equality(a, b);
    }

    /// Get shared variables
    #[must_use]
    pub fn shared_variables(&self) -> &FxHashSet<TermId> {
        self.nelson_oppen.shared_variables()
    }

    /// Get pending equalities
    #[must_use]
    pub fn get_pending_equalities(&self) -> &[(TermId, TermId)] {
        self.nelson_oppen.get_pending_equalities()
    }

    /// Clear pending equalities
    pub fn clear_pending(&mut self) {
        self.nelson_oppen.clear_pending();
    }

    /// Get statistics
    #[must_use]
    pub fn statistics(&self) -> CombinationStats {
        self.nelson_oppen.statistics()
    }

    /// Reset the combiner
    pub fn reset(&mut self) {
        self.nelson_oppen.reset();
        self.term_to_theory.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_nelson_oppen() {
        let no = NelsonOppen::new(2);
        assert_eq!(no.num_theories, 2);
        assert_eq!(no.shared_vars.len(), 0);
    }

    #[test]
    fn test_register_var() {
        let mut no = NelsonOppen::new(2);
        let var1 = TermId(1);

        no.register_var(var1, 0);
        assert_eq!(no.theory_vars[0].len(), 1);
        assert!(!no.is_shared(var1));
    }

    #[test]
    fn test_shared_variable_detection() {
        let mut no = NelsonOppen::new(2);
        let var1 = TermId(1);

        // Register with theory 0
        no.register_var(var1, 0);
        assert!(!no.is_shared(var1));

        // Register with theory 1 - now it's shared
        no.register_var(var1, 1);
        assert!(no.is_shared(var1));
    }

    #[test]
    fn test_add_equality() {
        let mut no = NelsonOppen::new(2);
        let a = TermId(1);
        let b = TermId(2);

        no.shared_vars.insert(a);

        no.add_equality(a, b);
        assert_eq!(no.equalities.len(), 1);
        assert_eq!(no.pending_equalities.len(), 1);
    }

    #[test]
    fn test_equality_normalization() {
        let mut no = NelsonOppen::new(2);
        let a = TermId(1);
        let b = TermId(2);

        no.shared_vars.insert(a);

        // Add in different orders
        no.add_equality(a, b);
        no.add_equality(b, a); // Should be same as (a, b)

        // Should only have one equality
        assert_eq!(no.equalities.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let mut no = NelsonOppen::new(3);
        let var1 = TermId(1);
        let var2 = TermId(2);

        no.register_var(var1, 0);
        no.register_var(var1, 1); // Now shared
        no.shared_vars.insert(var1);
        no.add_equality(var1, var2);

        let stats = no.statistics();
        assert_eq!(stats.num_theories, 3);
        assert_eq!(stats.num_shared_vars, 1);
        assert_eq!(stats.num_equalities, 1);
    }

    #[test]
    fn test_reset() {
        let mut no = NelsonOppen::new(2);
        let var1 = TermId(1);

        no.register_var(var1, 0);
        no.add_equality(var1, TermId(2));

        assert!(!no.equalities.is_empty());

        no.reset();

        assert!(no.equalities.is_empty());
        assert!(no.shared_vars.is_empty());
        assert!(no.pending_equalities.is_empty());
    }

    #[test]
    fn test_theory_combiner_creation() {
        let combiner = TheoryCombiner::new(4);
        assert_eq!(combiner.nelson_oppen.num_theories, 4);
    }

    #[test]
    fn test_classify_boolean_term() {
        let manager = TermManager::new();
        let mut combiner = TheoryCombiner::new(4);

        let t = manager.mk_bool(true);
        let theories = combiner.classify_term(t, &manager);

        assert!(theories.contains(&0)); // Boolean theory
    }

    #[test]
    fn test_classify_arithmetic_term() {
        let mut manager = TermManager::new();
        let mut combiner = TheoryCombiner::new(4);

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let add = manager.mk_add(vec![x, y]);

        let theories = combiner.classify_term(add, &manager);

        assert!(theories.contains(&1)); // Arithmetic theory
    }

    #[test]
    fn test_classify_array_term() {
        let mut manager = TermManager::new();
        let mut combiner = TheoryCombiner::new(4);

        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", array_sort);
        let i = manager.mk_var("i", int_sort);

        let select = manager.mk_select(a, i);
        let theories = combiner.classify_term(select, &manager);

        assert!(theories.contains(&2)); // Array theory
    }

    #[test]
    fn test_classify_variable() {
        let mut manager = TermManager::new();
        let mut combiner = TheoryCombiner::new(4);

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        let theories = combiner.classify_term(x, &manager);

        // Variables should be registered with all theories
        assert_eq!(theories.len(), 4);
    }
}
