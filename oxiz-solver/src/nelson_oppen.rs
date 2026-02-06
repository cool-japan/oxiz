//! Nelson-Oppen Theory Combination.
#![allow(dead_code, clippy::result_unit_err)] // Under development
//!
//! Implements the Nelson-Oppen framework for combining decision procedures
//! of disjoint theories through equality sharing.

use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::VecDeque;

/// Nelson-Oppen theory combination engine.
pub struct NelsonOppenCombiner {
    /// Shared terms between theories
    shared_terms: FxHashSet<TermId>,
    /// Equality classes for shared terms
    equality_classes: UnionFind,
    /// Pending equalities to propagate
    pending_equalities: VecDeque<(TermId, TermId)>,
    /// Theory assignments for shared terms
    theory_assignments: FxHashMap<TermId, TheoryId>,
    /// Statistics
    stats: NelsonOppenStats,
}

/// Theory identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TheoryId(pub usize);

/// Nelson-Oppen statistics
#[derive(Debug, Clone, Default)]
pub struct NelsonOppenStats {
    /// Number of shared terms
    pub shared_terms_count: usize,
    /// Number of equalities propagated
    pub equalities_propagated: usize,
    /// Number of theory conflicts detected
    pub theory_conflicts: usize,
    /// Number of purification steps
    pub purifications: usize,
}

impl NelsonOppenCombiner {
    /// Create a new Nelson-Oppen combiner.
    pub fn new() -> Self {
        Self {
            shared_terms: FxHashSet::default(),
            equality_classes: UnionFind::new(),
            pending_equalities: VecDeque::new(),
            theory_assignments: FxHashMap::default(),
            stats: NelsonOppenStats::default(),
        }
    }

    /// Register a shared term between theories.
    pub fn register_shared_term(&mut self, term_id: TermId, theory1: TheoryId, _theory2: TheoryId) {
        self.shared_terms.insert(term_id);
        self.theory_assignments.insert(term_id, theory1);
        self.equality_classes.make_set(term_id);
        self.stats.shared_terms_count += 1;
    }

    /// Assert an equality between shared terms.
    ///
    /// Returns Ok(()) if consistent, Err(()) if conflict detected.
    pub fn assert_equality(&mut self, lhs: TermId, rhs: TermId) -> Result<(), ()> {
        if !self.shared_terms.contains(&lhs) || !self.shared_terms.contains(&rhs) {
            return Err(()); // Only shared terms can be equated
        }

        // Check if already in same equivalence class
        if self.equality_classes.find(lhs) == self.equality_classes.find(rhs) {
            return Ok(());
        }

        // Merge equivalence classes
        self.equality_classes.union(lhs, rhs);
        self.pending_equalities.push_back((lhs, rhs));
        self.stats.equalities_propagated += 1;

        Ok(())
    }

    /// Purify a term by introducing fresh variables for sub-terms.
    ///
    /// Purification ensures each theory sees only its own symbols.
    pub fn purify_term(&mut self, term_id: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        self.stats.purifications += 1;

        // Recursively purify sub-terms
        let term = tm.get(term_id).ok_or("term not found")?.clone();

        match &term.kind {
            TermKind::Apply { func: _, args } => {
                let mut purified_args = Vec::new();

                for &arg in args {
                    let purified_arg = self.purify_term(arg, tm)?;
                    purified_args.push(purified_arg);
                }

                // Check if any argument changed theory
                let needs_purification = purified_args
                    .iter()
                    .enumerate()
                    .any(|(i, &purified)| self.get_theory(purified) != self.get_theory(args[i]));

                if needs_purification {
                    // Introduce fresh variable for this sub-term
                    // TODO: TermManager doesn't have fresh_var - needs implementation
                    // let fresh_var = tm.fresh_var(term.sort);
                    // For now, just return the original term
                    // self.register_shared_term(fresh_var, TheoryId(0), TheoryId(1));

                    // Assert equality: fresh_var = (func purified_args)
                    // TODO: mk_apply expects SmallVec, not Vec
                    // let purified_app = tm.mk_apply(*func, purified_args.into());
                    // self.assert_equality(fresh_var, purified_app)?;

                    // Ok(fresh_var)
                    Ok(term_id) // Placeholder
                } else {
                    Ok(term_id)
                }
            }
            _ => Ok(term_id),
        }
    }

    /// Get pending equalities to propagate to theories.
    pub fn get_pending_equalities(&mut self) -> Vec<(TermId, TermId)> {
        let mut result = Vec::new();
        while let Some(eq) = self.pending_equalities.pop_front() {
            result.push(eq);
        }
        result
    }

    /// Check if two terms are in the same equivalence class.
    pub fn are_equal(&self, lhs: TermId, rhs: TermId) -> bool {
        self.equality_classes.find(lhs) == self.equality_classes.find(rhs)
    }

    /// Get all terms in the equivalence class of a term.
    pub fn get_equivalence_class(&self, term_id: TermId) -> Vec<TermId> {
        let rep = self.equality_classes.find(term_id);
        self.shared_terms
            .iter()
            .filter(|&&t| self.equality_classes.find(t) == rep)
            .copied()
            .collect()
    }

    /// Get theory assignment for a term.
    fn get_theory(&self, term_id: TermId) -> Option<TheoryId> {
        self.theory_assignments.get(&term_id).copied()
    }

    /// Convexity closure: generate implied equalities.
    ///
    /// For convex theories, if we have equalities in each class,
    /// we must propagate all pairwise equalities.
    pub fn convexity_closure(&mut self) -> Vec<(TermId, TermId)> {
        let mut implied_equalities = Vec::new();

        // Group terms by equivalence class
        let mut classes: FxHashMap<TermId, Vec<TermId>> = FxHashMap::default();
        for &term in &self.shared_terms {
            let rep = self.equality_classes.find(term);
            classes.entry(rep).or_default().push(term);
        }

        // For each equivalence class with multiple elements
        for (_rep, terms) in classes {
            if terms.len() > 1 {
                // Generate all pairwise equalities
                for i in 0..terms.len() {
                    for j in (i + 1)..terms.len() {
                        implied_equalities.push((terms[i], terms[j]));
                    }
                }
            }
        }

        implied_equalities
    }

    /// Get statistics.
    pub fn stats(&self) -> &NelsonOppenStats {
        &self.stats
    }

    /// Reset for next SMT check.
    pub fn reset(&mut self) {
        self.shared_terms.clear();
        self.equality_classes = UnionFind::new();
        self.pending_equalities.clear();
        self.theory_assignments.clear();
        self.stats = NelsonOppenStats::default();
    }
}

impl Default for NelsonOppenCombiner {
    fn default() -> Self {
        Self::new()
    }
}

/// Union-Find data structure for equivalence classes.
#[derive(Debug, Clone)]
struct UnionFind {
    parent: FxHashMap<TermId, TermId>,
    rank: FxHashMap<TermId, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: FxHashMap::default(),
            rank: FxHashMap::default(),
        }
    }

    fn make_set(&mut self, x: TermId) {
        self.parent.insert(x, x);
        self.rank.insert(x, 0);
    }

    fn find(&self, x: TermId) -> TermId {
        let mut current = x;
        while let Some(&parent) = self.parent.get(&current) {
            if parent == current {
                return current;
            }
            current = parent;
        }
        x // Not found, return itself
    }

    fn union(&mut self, x: TermId, y: TermId) {
        let x_root = self.find(x);
        let y_root = self.find(y);

        if x_root == y_root {
            return;
        }

        let x_rank = *self.rank.get(&x_root).unwrap_or(&0);
        let y_rank = *self.rank.get(&y_root).unwrap_or(&0);

        if x_rank < y_rank {
            self.parent.insert(x_root, y_root);
        } else if x_rank > y_rank {
            self.parent.insert(y_root, x_root);
        } else {
            self.parent.insert(y_root, x_root);
            self.rank.insert(x_root, x_rank + 1);
        }
    }
}

// Placeholder types (these would be defined elsewhere in the codebase)
// Note: Using types from oxiz_core::ast instead
// #[derive(Debug, Clone)]
// struct Term {
//     kind: TermKind,
//     sort: SortId,
// }
//
// #[derive(Debug, Clone)]
// enum TermKind {
//     Var(String),
//     App(FuncId, Vec<TermId>),
//     Const(ConstId),
// }

type SortId = usize;
type FuncId = usize;
type ConstId = usize;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nelson_oppen_creation() {
        let combiner = NelsonOppenCombiner::new();
        assert_eq!(combiner.stats.shared_terms_count, 0);
    }

    #[test]
    fn test_register_shared_term() {
        let mut combiner = NelsonOppenCombiner::new();
        let term_id = TermId(0);

        combiner.register_shared_term(term_id, TheoryId(0), TheoryId(1));

        assert_eq!(combiner.stats.shared_terms_count, 1);
        assert!(combiner.shared_terms.contains(&term_id));
    }

    #[test]
    fn test_assert_equality() {
        let mut combiner = NelsonOppenCombiner::new();
        let t1 = TermId(0);
        let t2 = TermId(1);

        combiner.register_shared_term(t1, TheoryId(0), TheoryId(1));
        combiner.register_shared_term(t2, TheoryId(0), TheoryId(1));

        assert!(combiner.assert_equality(t1, t2).is_ok());
        assert!(combiner.are_equal(t1, t2));
        assert_eq!(combiner.stats.equalities_propagated, 1);
    }

    #[test]
    fn test_convexity_closure() {
        let mut combiner = NelsonOppenCombiner::new();
        let t1 = TermId(0);
        let t2 = TermId(1);
        let t3 = TermId(2);

        combiner.register_shared_term(t1, TheoryId(0), TheoryId(1));
        combiner.register_shared_term(t2, TheoryId(0), TheoryId(1));
        combiner.register_shared_term(t3, TheoryId(0), TheoryId(1));

        combiner.assert_equality(t1, t2).unwrap();
        combiner.assert_equality(t2, t3).unwrap();

        let implied = combiner.convexity_closure();
        assert!(!implied.is_empty());
    }
}
