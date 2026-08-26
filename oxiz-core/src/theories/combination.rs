//! Nelson-Oppen theory combination
//!
//! Implements the Nelson-Oppen framework for combining decision procedures
//! for disjoint theories.
//!
//! Reference: Z3's `src/smt/theory.cpp` and Nelson-Oppen combination framework

use crate::ast::traversal::get_children;
use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;

/// Interface for a theory decision procedure
///
/// The theories in [`crate::theories`] implement this so that
/// [`TheoryCombiner`] can drive them: it offers every term to every theory,
/// routes the equalities one theory deduces into the theories that share the
/// terms involved, and stops when a theory reports a conflict.
pub trait Theory: core::fmt::Debug {
    /// Offer a term to the theory
    ///
    /// Returns `true` when the term belongs to this theory's language (its
    /// sort or its operator), i.e. when the theory has taken it on. The
    /// combiner uses the answer to work out which terms are shared between
    /// theories, so a theory must not claim terms it cannot reason about.
    fn add_term(&mut self, term: TermId, manager: &TermManager) -> bool;

    /// Tell the theory that two terms are equal
    ///
    /// Returns `true` when this was new information for the theory — both
    /// terms are known to it and they were not already in the same class.
    /// A theory that does not know either term returns `false`.
    fn assert_equality(&mut self, a: TermId, b: TermId) -> bool;

    /// Check the theory's current state
    ///
    /// Takes `&mut TermManager` because a theory may need to build terms in
    /// order to say what it deduced (a folded bit-vector constant, an axiom
    /// instance).
    fn check(&mut self, manager: &mut TermManager) -> TheoryResult;

    /// Get the name of the theory
    fn name(&self) -> &str;

    /// Reset the theory state
    fn reset(&mut self);
}

/// Result of a theory check
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TheoryResult {
    /// Theory found nothing new to say about its current state
    Sat,
    /// Theory found a conflict
    Unsat {
        /// Terms explaining the conflict
        ///
        /// The convention used by the theories in this module is a chain of
        /// terms whose consecutive entries were asserted or deduced equal,
        /// running between the two facts that clash.
        explanation: Vec<TermId>,
    },
    /// Theory deduced new equalities
    Propagate(Vec<(TermId, TermId)>),
    /// Theory produced lemmas that the caller should assert
    ///
    /// Used by the axiom-instantiating theories (arrays, strings), which
    /// contribute formulas rather than equalities between existing terms.
    Lemmas(Vec<TermId>),
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

    /// Add a slot for one more theory, returning its index
    pub fn add_theory_slot(&mut self) -> usize {
        self.theory_vars.push(FxHashSet::default());
        self.num_theories += 1;
        self.num_theories - 1
    }

    /// Add an equality between two terms
    ///
    /// Returns `true` when the equality had not been seen before.
    pub fn add_equality(&mut self, a: TermId, b: TermId) -> bool {
        let eq = if a.0 < b.0 { (a, b) } else { (b, a) };

        if self.equalities.insert(eq) {
            // This is a new equality
            if self.shared_vars.contains(&a) || self.shared_vars.contains(&b) {
                // At least one of the terms is shared, propagate to all theories
                self.pending_equalities.push(eq);
            }
            return true;
        }
        false
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

/// Outcome of a combined theory run
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinerOutcome {
    /// No registered theory reported a conflict, and the equality exchange
    /// reached a fixpoint
    ///
    /// This is "no conflict found by these theories", not a claim that the
    /// input is satisfiable: the theories in this module are incomplete.
    NoConflict,
    /// A registered theory reported a conflict
    Conflict {
        /// Index of the theory that reported it
        theory: usize,
        /// Terms explaining the conflict, in that theory's convention
        explanation: Vec<TermId>,
    },
}

/// Theory combiner that coordinates multiple theories
///
/// There are two ways to use it, which should not be mixed:
///
/// * With registered [`Theory`] objects ([`TheoryCombiner::add_theory`]):
///   [`TheoryCombiner::add_term`] offers terms to every theory and
///   [`TheoryCombiner::run`] drives the Nelson-Oppen equality exchange.
/// * Without them, as a term classifier ([`TheoryCombiner::classify_term`]),
///   which uses the fixed numbering documented on that method.
#[derive(Debug)]
pub struct TheoryCombiner {
    /// Nelson-Oppen combiner
    nelson_oppen: NelsonOppen,
    /// Mapping of terms to the theory indices that claimed them
    term_to_theory: FxHashMap<TermId, Vec<usize>>,
    /// Registered theories, indexed by their Nelson-Oppen slot
    theories: Vec<Box<dyn Theory>>,
    /// Lemmas contributed by the registered theories
    lemmas: Vec<TermId>,
}

impl TheoryCombiner {
    /// Create a new theory combiner
    #[must_use]
    pub fn new(num_theories: usize) -> Self {
        Self {
            nelson_oppen: NelsonOppen::new(num_theories),
            term_to_theory: FxHashMap::default(),
            theories: Vec::new(),
            lemmas: Vec::new(),
        }
    }

    /// Register a theory, returning the index it was given
    ///
    /// The index is the theory's Nelson-Oppen slot, and is what
    /// [`CombinerOutcome::Conflict`] reports.
    pub fn add_theory(&mut self, theory: Box<dyn Theory>) -> usize {
        let index = self.theories.len();
        while self.nelson_oppen.num_theories <= index {
            self.nelson_oppen.add_theory_slot();
        }
        self.theories.push(theory);
        index
    }

    /// Number of registered theories
    #[must_use]
    pub fn num_registered_theories(&self) -> usize {
        self.theories.len()
    }

    /// Name of a registered theory
    #[must_use]
    pub fn theory_name(&self, index: usize) -> Option<&str> {
        self.theories.get(index).map(|theory| theory.name())
    }

    /// Offer a term to every registered theory
    ///
    /// Returns the indices of the theories that claimed it. A term claimed by
    /// more than one theory is a shared term; so are the arguments of a
    /// claimed term, because they are where one theory's language meets
    /// another's (`mk(x)` is a datatype term whose argument `x` may be a
    /// bit-vector one). Both are registered with Nelson-Oppen, which is what
    /// makes an equality between them travel from one theory to the other.
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) -> Vec<usize> {
        let mut owners = Vec::new();

        for index in 0..self.theories.len() {
            if !self.theories[index].add_term(term, manager) {
                continue;
            }

            owners.push(index);
            self.nelson_oppen.register_var(term, index);

            if let Some(t) = manager.get(term) {
                for child in get_children(&t.kind) {
                    self.nelson_oppen.register_var(child, index);
                }
            }
        }

        if !owners.is_empty() {
            self.term_to_theory.insert(term, owners.clone());
        }

        owners
    }

    /// Assert an equality and hand it to every registered theory
    ///
    /// Returns `true` when at least one theory (or the Nelson-Oppen bookkeeping)
    /// learned something new.
    pub fn assert_equality(&mut self, a: TermId, b: TermId) -> bool {
        let mut changed = self.nelson_oppen.add_equality(a, b);

        for theory in &mut self.theories {
            if theory.assert_equality(a, b) {
                changed = true;
            }
        }

        changed
    }

    /// Run the registered theories to a fixpoint, exchanging equalities
    ///
    /// Each round checks every theory; equalities a theory deduces go into the
    /// Nelson-Oppen bookkeeping, and those that involve a shared term are
    /// handed to the other theories before the next round. The loop stops when
    /// a round produces nothing new, or as soon as a theory reports a conflict.
    pub fn run(&mut self, manager: &mut TermManager) -> CombinerOutcome {
        loop {
            let mut changed = false;

            for index in 0..self.theories.len() {
                match self.theories[index].check(manager) {
                    TheoryResult::Sat => {}
                    TheoryResult::Unsat { explanation } => {
                        return CombinerOutcome::Conflict {
                            theory: index,
                            explanation,
                        };
                    }
                    TheoryResult::Propagate(equalities) => {
                        for (a, b) in equalities {
                            if self.nelson_oppen.add_equality(a, b) {
                                changed = true;
                            }
                        }
                    }
                    TheoryResult::Lemmas(terms) => {
                        if !terms.is_empty() {
                            changed = true;
                            self.lemmas.extend(terms);
                        }
                    }
                }
            }

            let pending: Vec<(TermId, TermId)> =
                self.nelson_oppen.get_pending_equalities().to_vec();
            self.nelson_oppen.clear_pending();

            for (a, b) in pending {
                for theory in &mut self.theories {
                    if theory.assert_equality(a, b) {
                        changed = true;
                    }
                }
            }

            if !changed {
                return CombinerOutcome::NoConflict;
            }
        }
    }

    /// Lemmas collected from the registered theories so far
    #[must_use]
    pub fn lemmas(&self) -> &[TermId] {
        &self.lemmas
    }

    /// Take the collected lemmas, leaving the combiner's list empty
    pub fn take_lemmas(&mut self) -> Vec<TermId> {
        core::mem::take(&mut self.lemmas)
    }

    /// Theories that claimed a term, as recorded by [`TheoryCombiner::add_term`]
    #[must_use]
    pub fn theories_of(&self, term: TermId) -> Option<&[usize]> {
        self.term_to_theory.get(&term).map(Vec::as_slice)
    }

    /// Classify a term by its structure, using a fixed theory numbering
    ///
    /// This is the classifier used when no [`Theory`] objects are registered:
    /// 0 is the Boolean theory, 1 arithmetic, 2 arrays, 3 bit-vectors, and
    /// variables and equalities are registered with every theory. The indices
    /// are a convention of this method and are unrelated to the indices handed
    /// out by [`TheoryCombiner::add_theory`], so the two modes should not be
    /// mixed on one combiner.
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
        if !theories.is_empty() {
            self.term_to_theory.insert(term, theories.clone());
        }
        for &theory_idx in &theories {
            // If it's a variable, register it with Nelson-Oppen
            if let Some(t) = manager.get(term)
                && matches!(t.kind, TermKind::Var(_))
            {
                self.nelson_oppen.register_var(term, theory_idx);
            }
        }

        theories
    }

    /// Record an equality in the Nelson-Oppen bookkeeping only
    ///
    /// Use [`TheoryCombiner::assert_equality`] to also hand it to the
    /// registered theories.
    pub fn add_equality(&mut self, a: TermId, b: TermId) -> bool {
        self.nelson_oppen.add_equality(a, b)
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

    /// Reset the combiner and every registered theory
    pub fn reset(&mut self) {
        self.nelson_oppen.reset();
        self.term_to_theory.clear();
        self.lemmas.clear();
        for theory in &mut self.theories {
            theory.reset();
        }
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
