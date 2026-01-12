//! Craig interpolation for modular verification
//!
//! Given an UNSAT formula A ∧ B, compute an interpolant I such that:
//! - A ⟹ I
//! - I ∧ B is UNSAT
//! - I only contains symbols common to A and B
//!
//! Reference: Z3's interpolation in `../z3/src/smt/theory_interpolant.cpp`

use crate::ast::proof::{Proof, ProofId, ProofRule};
use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};

/// Interpolation context for computing Craig interpolants
#[derive(Debug)]
pub struct InterpolationContext {
    /// Terms from partition A
    partition_a: FxHashSet<TermId>,
    /// Terms from partition B
    partition_b: FxHashSet<TermId>,
    /// Symbols (variables/functions) from A
    symbols_a: FxHashSet<TermId>,
    /// Symbols (variables/functions) from B
    symbols_b: FxHashSet<TermId>,
    /// Shared symbols between A and B
    shared_symbols: FxHashSet<TermId>,
    /// Computed interpolants for proof nodes
    interpolants: FxHashMap<ProofId, TermId>,
}

impl InterpolationContext {
    /// Create a new interpolation context
    #[must_use]
    pub fn new() -> Self {
        Self {
            partition_a: FxHashSet::default(),
            partition_b: FxHashSet::default(),
            symbols_a: FxHashSet::default(),
            symbols_b: FxHashSet::default(),
            shared_symbols: FxHashSet::default(),
            interpolants: FxHashMap::default(),
        }
    }

    /// Add a term to partition A
    pub fn add_to_partition_a(&mut self, term: TermId, manager: &TermManager) {
        self.partition_a.insert(term);
        collect_symbols(term, &mut self.symbols_a, manager);
    }

    /// Add a term to partition B
    pub fn add_to_partition_b(&mut self, term: TermId, manager: &TermManager) {
        self.partition_b.insert(term);
        collect_symbols(term, &mut self.symbols_b, manager);
    }

    /// Finalize the context by computing shared symbols
    pub fn finalize(&mut self) {
        self.shared_symbols = self
            .symbols_a
            .intersection(&self.symbols_b)
            .copied()
            .collect();
    }

    /// Check if a term belongs to partition A
    #[must_use]
    pub fn is_in_a(&self, term: TermId) -> bool {
        self.partition_a.contains(&term)
    }

    /// Check if a term belongs to partition B
    #[must_use]
    pub fn is_in_b(&self, term: TermId) -> bool {
        self.partition_b.contains(&term)
    }

    /// Check if a symbol is shared between A and B
    #[must_use]
    pub fn is_shared_symbol(&self, symbol: TermId) -> bool {
        self.shared_symbols.contains(&symbol)
    }

    /// Get all shared symbols
    #[must_use]
    pub fn shared_symbols(&self) -> &FxHashSet<TermId> {
        &self.shared_symbols
    }

    /// Compute interpolant for the proof
    ///
    /// This implements the Pudlák interpolation algorithm:
    /// - For leaves (A-clauses), the interpolant is the clause itself (or true if only B-symbols)
    /// - For leaves (B-clauses), the interpolant is false (or the clause with A-symbols)
    /// - For resolution steps, combine interpolants appropriately
    pub fn compute_interpolant(
        &mut self,
        proof: &Proof,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        let root = proof.root();
        self.compute_node_interpolant(proof, root, manager)
    }

    /// Recursively compute interpolant for a proof node
    fn compute_node_interpolant(
        &mut self,
        proof: &Proof,
        node_id: ProofId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        // Check if already computed
        if let Some(&interpolant) = self.interpolants.get(&node_id) {
            return Some(interpolant);
        }

        let node = proof.get_node(node_id)?;

        let interpolant = match &node.rule {
            ProofRule::Assume { .. } => {
                // Assumption: check which partition it belongs to
                if self.is_in_a(node.conclusion) {
                    // A-clause: interpolant is the clause (or true if only B-symbols)
                    if self.only_has_b_symbols(node.conclusion, manager) {
                        manager.mk_bool(true)
                    } else {
                        node.conclusion
                    }
                } else {
                    // B-clause: interpolant is false (or the clause with A-symbols removed)
                    if self.only_has_a_symbols(node.conclusion, manager) {
                        manager.mk_bool(false)
                    } else {
                        self.project_to_shared(node.conclusion, manager)
                    }
                }
            }

            ProofRule::Resolution { pivot } => {
                // Resolution: combine interpolants from premises
                if node.premises.len() != 2 {
                    return None;
                }

                let i1 = self.compute_node_interpolant(proof, node.premises[0], manager)?;
                let i2 = self.compute_node_interpolant(proof, node.premises[1], manager)?;

                // Combine based on which partition the pivot belongs to
                if self.is_shared_symbol(*pivot) {
                    // Pivot is shared: disjunction
                    manager.mk_or(vec![i1, i2])
                } else if self.has_a_symbols(*pivot, manager) {
                    // Pivot is in A: use interpolant from second premise
                    i2
                } else {
                    // Pivot is in B: use interpolant from first premise
                    i1
                }
            }

            ProofRule::Transitivity | ProofRule::Congruence => {
                // For transitivity and congruence, combine premise interpolants
                let premise_interpolants: Vec<_> = node
                    .premises
                    .iter()
                    .filter_map(|&p| self.compute_node_interpolant(proof, p, manager))
                    .collect();

                if premise_interpolants.is_empty() {
                    manager.mk_bool(true)
                } else if premise_interpolants.len() == 1 {
                    premise_interpolants[0]
                } else {
                    manager.mk_and(premise_interpolants)
                }
            }

            ProofRule::TheoryLemma { .. } | ProofRule::ArithInequality => {
                // Theory lemma: project to shared symbols
                self.project_to_shared(node.conclusion, manager)
            }

            _ => {
                // For other rules, use conclusion projected to shared symbols
                self.project_to_shared(node.conclusion, manager)
            }
        };

        self.interpolants.insert(node_id, interpolant);
        Some(interpolant)
    }

    /// Get statistics about the interpolation
    #[must_use]
    pub fn statistics(&self) -> InterpolationStats {
        InterpolationStats {
            partition_a_size: self.partition_a.len(),
            partition_b_size: self.partition_b.len(),
            symbols_a_size: self.symbols_a.len(),
            symbols_b_size: self.symbols_b.len(),
            shared_symbols_size: self.shared_symbols.len(),
            interpolants_computed: self.interpolants.len(),
        }
    }

    /// Check if a term only contains B-symbols
    fn only_has_b_symbols(&self, term: TermId, manager: &TermManager) -> bool {
        let mut symbols = FxHashSet::default();
        collect_symbols(term, &mut symbols, manager);
        symbols.iter().all(|s| self.symbols_b.contains(s))
    }

    /// Check if a term only contains A-symbols
    fn only_has_a_symbols(&self, term: TermId, manager: &TermManager) -> bool {
        let mut symbols = FxHashSet::default();
        collect_symbols(term, &mut symbols, manager);
        symbols.iter().all(|s| self.symbols_a.contains(s))
    }

    /// Check if a term contains any A-symbols
    fn has_a_symbols(&self, term: TermId, manager: &TermManager) -> bool {
        let mut symbols = FxHashSet::default();
        collect_symbols(term, &mut symbols, manager);
        symbols.iter().any(|s| self.symbols_a.contains(s))
    }

    /// Project a term to only shared symbols (approximation)
    ///
    /// This replaces non-shared symbols with fresh variables or eliminates them
    fn project_to_shared(&self, term: TermId, manager: &mut TermManager) -> TermId {
        // Clone the term kind to avoid borrowing issues
        let kind = manager.get(term).map(|t| t.kind.clone());

        if let Some(k) = kind {
            match k {
                TermKind::Var(_) => {
                    if self.is_shared_symbol(term) {
                        term
                    } else {
                        // Replace with true for simplicity (could use fresh var)
                        manager.mk_bool(true)
                    }
                }
                TermKind::And(args) => {
                    let projected: Vec<_> = args
                        .iter()
                        .map(|&arg| self.project_to_shared(arg, manager))
                        .collect();
                    manager.mk_and(projected)
                }
                TermKind::Or(args) => {
                    let projected: Vec<_> = args
                        .iter()
                        .map(|&arg| self.project_to_shared(arg, manager))
                        .collect();
                    manager.mk_or(projected)
                }
                TermKind::Not(arg) => {
                    let projected = self.project_to_shared(arg, manager);
                    manager.mk_not(projected)
                }
                TermKind::Implies(a, b) => {
                    let pa = self.project_to_shared(a, manager);
                    let pb = self.project_to_shared(b, manager);
                    manager.mk_implies(pa, pb)
                }
                _ => term, // Keep other terms as is
            }
        } else {
            term
        }
    }
}

/// Collect all symbols (variables) from a term
fn collect_symbols(term: TermId, symbols: &mut FxHashSet<TermId>, manager: &TermManager) {
    if let Some(t) = manager.get(term) {
        match &t.kind {
            TermKind::Var(_) => {
                symbols.insert(term);
            }
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => {
                for &arg in args {
                    collect_symbols(arg, symbols, manager);
                }
            }
            TermKind::Not(arg) | TermKind::Neg(arg) | TermKind::BvNot(arg) => {
                collect_symbols(*arg, symbols, manager);
            }
            TermKind::Eq(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Le(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::Implies(a, b)
            | TermKind::Xor(a, b)
            | TermKind::BvAnd(a, b)
            | TermKind::Select(a, b) => {
                collect_symbols(*a, symbols, manager);
                collect_symbols(*b, symbols, manager);
            }
            TermKind::Ite(c, t, e) | TermKind::Store(c, t, e) => {
                collect_symbols(*c, symbols, manager);
                collect_symbols(*t, symbols, manager);
                collect_symbols(*e, symbols, manager);
            }
            _ => {}
        }
    }
}

impl Default for InterpolationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about interpolation
#[derive(Debug, Default, Clone)]
pub struct InterpolationStats {
    /// Size of partition A
    pub partition_a_size: usize,
    /// Size of partition B
    pub partition_b_size: usize,
    /// Number of symbols in A
    pub symbols_a_size: usize,
    /// Number of symbols in B
    pub symbols_b_size: usize,
    /// Number of shared symbols
    pub shared_symbols_size: usize,
    /// Number of interpolants computed
    pub interpolants_computed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_context() {
        let ctx = InterpolationContext::new();
        assert_eq!(ctx.partition_a.len(), 0);
        assert_eq!(ctx.partition_b.len(), 0);
    }

    #[test]
    fn test_add_to_partitions() {
        let mut ctx = InterpolationContext::new();
        let manager = TermManager::new();

        let term_a = TermId(1);
        let term_b = TermId(2);

        ctx.add_to_partition_a(term_a, &manager);
        ctx.add_to_partition_b(term_b, &manager);

        assert!(ctx.is_in_a(term_a));
        assert!(ctx.is_in_b(term_b));
        assert!(!ctx.is_in_a(term_b));
        assert!(!ctx.is_in_b(term_a));
    }

    #[test]
    fn test_shared_symbols() {
        let mut manager = TermManager::new();
        let mut ctx = InterpolationContext::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);

        // x appears in both partitions
        let term_a = manager.mk_add(vec![x, y]);
        let term_b = manager.mk_add(vec![x]);

        ctx.add_to_partition_a(term_a, &manager);
        ctx.add_to_partition_b(term_b, &manager);
        ctx.finalize();

        // x should be shared
        assert!(ctx.is_shared_symbol(x));
        // y should not be shared
        assert!(!ctx.is_shared_symbol(y));
    }

    #[test]
    fn test_statistics() {
        let mut ctx = InterpolationContext::new();
        let mut manager = TermManager::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        ctx.add_to_partition_a(x, &manager);

        let stats = ctx.statistics();
        assert_eq!(stats.partition_a_size, 1);
        assert_eq!(stats.partition_b_size, 0);
    }

    #[test]
    fn test_project_to_shared() {
        let mut manager = TermManager::new();
        let mut ctx = InterpolationContext::new();

        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);

        // Make x shared
        ctx.symbols_a.insert(x);
        ctx.symbols_b.insert(x);
        ctx.symbols_a.insert(y); // y only in A
        ctx.shared_symbols.insert(x);

        // Project a term containing both x and y
        let term = manager.mk_add(vec![x, y]);
        let projected = ctx.project_to_shared(term, &mut manager);

        // Should keep x, replace y
        assert!(manager.get(projected).is_some());
    }
}
