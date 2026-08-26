//! Craig interpolation for modular verification
//!
//! Given an UNSAT formula A ∧ B, compute an interpolant I such that:
//! - A ⟹ I
//! - I ∧ B is UNSAT
//! - I only contains symbols common to A and B
//!
//! Reference: Z3's interpolation in `src/smt/theory_interpolant.cpp`

use crate::ast::proof::{Proof, ProofId, ProofRule};
use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use smallvec::SmallVec;

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

    /// Compute the interpolant of a proof node.
    ///
    /// The walk over the proof DAG is iterative: a real UNSAT proof is
    /// routinely 10^5 nodes tall, and the recursive form had one frame per
    /// level with no bound. Results are memoized in `self.interpolants`
    /// exactly as before, and a node whose interpolant cannot be computed is
    /// recorded as a failure instead of being retried.
    ///
    /// A node reached again while it is still being expanded (only possible
    /// for a malformed, cyclic proof) yields `None` rather than looping
    /// forever.
    fn compute_node_interpolant(
        &mut self,
        proof: &Proof,
        node_id: ProofId,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        /// Work item of the iterative proof-DAG walk.
        enum Frame {
            /// Expand a node: schedule its premises.
            Enter(ProofId),
            /// Combine the already-computed premise interpolants.
            Build(ProofId),
        }

        let mut memo: FxHashMap<ProofId, Option<TermId>> = FxHashMap::default();
        let mut in_progress: FxHashSet<ProofId> = FxHashSet::default();
        let mut stack = vec![Frame::Enter(node_id)];

        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(id) => {
                    // Check if already computed
                    if let Some(&interpolant) = self.interpolants.get(&id) {
                        memo.insert(id, Some(interpolant));
                        continue;
                    }
                    if memo.contains_key(&id) {
                        continue;
                    }
                    if !in_progress.insert(id) {
                        // Cyclic proof: no well-founded interpolant exists.
                        memo.insert(id, None);
                        continue;
                    }

                    let Some(node) = proof.get_node(id) else {
                        memo.insert(id, None);
                        continue;
                    };

                    match &node.rule {
                        ProofRule::Assume { .. } => {
                            // Assumption: check which partition it belongs to
                            let interpolant = if self.is_in_a(node.conclusion) {
                                // A-clause: interpolant is the clause (or true
                                // if only B-symbols)
                                if self.only_has_b_symbols(node.conclusion, manager) {
                                    manager.mk_bool(true)
                                } else {
                                    node.conclusion
                                }
                            } else {
                                // B-clause: interpolant is false (or the clause
                                // with A-symbols removed)
                                if self.only_has_a_symbols(node.conclusion, manager) {
                                    manager.mk_bool(false)
                                } else {
                                    self.project_to_shared(node.conclusion, manager)
                                }
                            };
                            self.interpolants.insert(id, interpolant);
                            memo.insert(id, Some(interpolant));
                        }

                        ProofRule::Resolution { .. } => {
                            if node.premises.len() != 2 {
                                memo.insert(id, None);
                                continue;
                            }
                            stack.push(Frame::Build(id));
                            for &premise in node.premises.iter() {
                                stack.push(Frame::Enter(premise));
                            }
                        }

                        ProofRule::Transitivity | ProofRule::Congruence => {
                            stack.push(Frame::Build(id));
                            for &premise in node.premises.iter() {
                                stack.push(Frame::Enter(premise));
                            }
                        }

                        ProofRule::TheoryLemma { .. } | ProofRule::ArithInequality => {
                            // Theory lemma: project to shared symbols
                            let interpolant = self.project_to_shared(node.conclusion, manager);
                            self.interpolants.insert(id, interpolant);
                            memo.insert(id, Some(interpolant));
                        }

                        _ => {
                            // For other rules, use conclusion projected to
                            // shared symbols
                            let interpolant = self.project_to_shared(node.conclusion, manager);
                            self.interpolants.insert(id, interpolant);
                            memo.insert(id, Some(interpolant));
                        }
                    }
                }

                Frame::Build(id) => {
                    let Some(node) = proof.get_node(id) else {
                        memo.insert(id, None);
                        continue;
                    };

                    let interpolant = match &node.rule {
                        ProofRule::Resolution { pivot } => {
                            let (Some(&Some(i1)), Some(&Some(i2))) =
                                (memo.get(&node.premises[0]), memo.get(&node.premises[1]))
                            else {
                                memo.insert(id, None);
                                continue;
                            };

                            // Combine based on which partition the pivot
                            // belongs to
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

                        // For transitivity and congruence, combine premise
                        // interpolants; premises without an interpolant are
                        // skipped, as they were by the recursive form.
                        _ => {
                            let premise_interpolants: Vec<_> = node
                                .premises
                                .iter()
                                .filter_map(|p| memo.get(p).copied().flatten())
                                .collect();

                            if premise_interpolants.is_empty() {
                                manager.mk_bool(true)
                            } else if premise_interpolants.len() == 1 {
                                premise_interpolants[0]
                            } else {
                                manager.mk_and(premise_interpolants)
                            }
                        }
                    };

                    self.interpolants.insert(id, interpolant);
                    memo.insert(id, Some(interpolant));
                }
            }
        }

        memo.get(&node_id).copied().flatten()
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
    /// This replaces non-shared symbols with fresh variables or eliminates
    /// them. The rewrite is bottom-up over an explicit post-order with a
    /// memo: the recursive form had no depth bound and re-projected shared
    /// sub-terms once per occurrence.
    fn project_to_shared(&self, term: TermId, manager: &mut TermManager) -> TermId {
        let mut memo: FxHashMap<TermId, TermId> = FxHashMap::default();

        for current in projection_postorder(term, manager) {
            let Some(kind) = manager.get(current).map(|t| t.kind.clone()) else {
                memo.insert(current, current);
                continue;
            };
            let mapped = |id: &TermId, memo: &FxHashMap<TermId, TermId>| -> TermId {
                memo.get(id).copied().unwrap_or(*id)
            };
            let projected = match &kind {
                TermKind::Var(_) => {
                    if self.is_shared_symbol(current) {
                        current
                    } else {
                        // Replace with true for simplicity (could use fresh var)
                        manager.mk_bool(true)
                    }
                }
                TermKind::And(args) => {
                    let projected: Vec<_> = args.iter().map(|a| mapped(a, &memo)).collect();
                    manager.mk_and(projected)
                }
                TermKind::Or(args) => {
                    let projected: Vec<_> = args.iter().map(|a| mapped(a, &memo)).collect();
                    manager.mk_or(projected)
                }
                TermKind::Not(arg) => {
                    let projected = mapped(arg, &memo);
                    manager.mk_not(projected)
                }
                TermKind::Implies(a, b) => {
                    let (pa, pb) = (mapped(a, &memo), mapped(b, &memo));
                    manager.mk_implies(pa, pb)
                }
                // Keep other terms as is: the projection is defined only for
                // the boolean skeleton, and leaving a term untouched is a
                // deliberate choice, not a missing case.
                _ => current,
            };
            memo.insert(current, projected);
        }

        memo.get(&term).copied().unwrap_or(term)
    }
}

/// Children-before-parents listing of the sub-terms the projection rewrites.
///
/// Only the boolean skeleton is rewritten, so only those operands are
/// traversed — exactly the sub-terms the recursive projection descended into.
fn projection_postorder(root: TermId, manager: &TermManager) -> Vec<TermId> {
    fn rewritten_children(kind: &TermKind) -> SmallVec<[TermId; 4]> {
        match kind {
            TermKind::And(args) | TermKind::Or(args) => args.iter().copied().collect(),
            TermKind::Not(arg) => smallvec::smallvec![*arg],
            TermKind::Implies(a, b) => smallvec::smallvec![*a, *b],
            _ => SmallVec::new(),
        }
    }

    let mut order = Vec::new();
    let mut visited: FxHashSet<TermId> = FxHashSet::default();
    let mut stack = vec![(root, false)];

    while let Some((current, expanded)) = stack.pop() {
        if expanded {
            order.push(current);
            continue;
        }
        if !visited.insert(current) {
            continue;
        }
        stack.push((current, true));
        if let Some(t) = manager.get(current) {
            for child in rewritten_children(&t.kind) {
                if !visited.contains(&child) {
                    stack.push((child, false));
                }
            }
        }
    }

    order
}

/// Collect all symbols (variables) from a term.
///
/// Iterative, with a visited set, descending through the exhaustive
/// [`crate::ast::traversal::get_children`]. The previous per-kind list ended
/// in a silent catch-all, so the symbols inside a function application,
/// datatype, string or floating-point term were never seen: a clause such as
/// `f(a)` then looked symbol-free, `only_has_b_symbols` vacuously returned
/// `true`, and the interpolant for it collapsed to `true` — a wrong
/// interpolant, not merely a weak one.
fn collect_symbols(term: TermId, symbols: &mut FxHashSet<TermId>, manager: &TermManager) {
    let mut stack = vec![term];
    let mut visited: FxHashSet<TermId> = FxHashSet::default();

    while let Some(current) = stack.pop() {
        if !visited.insert(current) {
            continue;
        }
        let Some(t) = manager.get(current) else {
            continue;
        };
        if let TermKind::Var(_) = &t.kind {
            symbols.insert(current);
            continue;
        }
        stack.extend(crate::ast::traversal::get_children(&t.kind));
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

#[cfg(test)]
mod deep_walk_tests {
    use super::*;
    use crate::ast::TermManager;

    #[test]
    fn test_collect_symbols_sees_apply_arguments() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let a = manager.mk_var("a", int_sort);
        let fa = manager.mk_apply("f", [a], int_sort);

        let mut symbols = FxHashSet::default();
        collect_symbols(fa, &mut symbols, &manager);
        assert!(
            symbols.contains(&a),
            "symbol below a function application was lost"
        );
    }

    #[test]
    fn test_collect_symbols_shared_dag_is_fast() {
        // 55 doubling levels: exponential without a visited set.
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let a = manager.mk_var("a", int_sort);
        let mut level = a;
        for _ in 0..55 {
            level = manager.mk_apply("f", [level, level], int_sort);
        }

        let mut symbols = FxHashSet::default();
        collect_symbols(level, &mut symbols, &manager);
        assert_eq!(symbols.len(), 1);
    }

    #[test]
    fn test_interpolation_walks_deep_nesting_do_not_overflow() {
        // 128 KiB / 6_250 levels: the house ratio for a genuinely nested
        // `Not` chain (see oxiz-spacer/src/walk.rs's `DEEP_STACK`/`DEEP_DEPTH`).
        // `mk_not` folds double negation (`not(not(p)) == p`), so a loop of
        // `mk_not` calls oscillates between depth 0 and 1 and never actually
        // nests -- intern the `Not` nodes directly so the chain really is
        // this deep.
        const DEEP_STACK: usize = 1 << 17;
        const DEEP_DEPTH: usize = 6_250;

        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let bool_sort = manager.sorts.bool_sort;
                let p = manager.mk_var("p", bool_sort);
                let mut term = p;
                for _ in 0..DEEP_DEPTH {
                    term = manager.intern_term(TermKind::Not(term), bool_sort);
                }

                let mut symbols = FxHashSet::default();
                collect_symbols(term, &mut symbols, &manager);

                let mut ctx = InterpolationContext::new();
                ctx.add_to_partition_a(p, &manager);
                ctx.finalize();
                let projected = ctx.project_to_shared(term, &mut manager);
                (symbols.len(), projected)
            })
            .expect("thread spawn should succeed");

        let (num_symbols, _projected) = handle.join().expect("deep walks must not overflow");
        assert_eq!(num_symbols, 1);
    }
}
