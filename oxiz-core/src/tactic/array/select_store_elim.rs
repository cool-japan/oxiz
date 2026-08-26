//! Select-Store Elimination Tactic for Array Theory.
//!
//! Simplifies array expressions by eliminating redundant select-store patterns:
//! - select(store(a, i, v), i) → v
//! - select(store(a, i, v), j) → select(a, j) when i ≠ j
//! - store(store(a, i, v1), i, v2) → store(a, i, v2)
//! - Array extensionality reasoning

use crate::ast::{TermId, TermKind, TermManager};
#[allow(unused_imports)]
use crate::prelude::*;
use std::rc::Rc;

/// Select-store elimination tactic.
pub struct SelectStoreElimTactic {
    /// Rewrite cache
    cache: FxHashMap<TermId, TermId>,
    /// Store terms reachable from a formula handed to [`SelectStoreElimTactic::apply`].
    ///
    /// Membership — not the chain itself — is what decides whether
    /// extensionality may fire, so the scan phase only has to record the set.
    store_terms: FxHashSet<TermId>,
    /// Store chains, materialized on demand and shared by [`Rc`].
    ///
    /// Only the terms extensionality actually asks about ever get a chain, so
    /// a long store spine costs one entry instead of one per spine node.
    chain_cache: FxHashMap<TermId, Rc<StoreChain>>,
    /// Statistics
    stats: SelectStoreElimStats,
}

/// A chain of store operations on the same base array.
#[derive(Debug, Clone)]
pub struct StoreChain {
    /// Base array (innermost array)
    pub base: TermId,
    /// Sequence of store operations: (index, value)
    pub stores: Vec<(TermId, TermId)>,
}

/// Select-store elimination statistics.
#[derive(Debug, Clone, Default)]
pub struct SelectStoreElimStats {
    /// select(store(a,i,v), i) → v eliminations
    pub select_store_same_index: usize,
    /// select(store(a,i,v), j) → select(a,j) with i≠j
    pub select_store_diff_index: usize,
    /// store(store(a,i,v1), i, v2) → store(a,i,v2)
    pub redundant_store_elim: usize,
    /// Terms rewritten
    pub terms_rewritten: usize,
    /// Extensionality applications
    pub extensionality_apps: usize,
}

impl SelectStoreElimTactic {
    /// Create a new select-store elimination tactic.
    pub fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
            store_terms: FxHashSet::default(),
            chain_cache: FxHashMap::default(),
            stats: SelectStoreElimStats::default(),
        }
    }

    /// Apply the tactic to a formula.
    pub fn apply(&mut self, formula: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        // Phase 1: Record which terms are store spines
        self.collect_store_terms(formula, tm)?;

        // Phase 2: Rewrite formula
        let rewritten = self.rewrite(formula, tm)?;

        Ok(rewritten)
    }

    /// Record every store term reachable from `tid`.
    ///
    /// Iterative (explicit heap stack plus a `visited` set): the term DAG's
    /// depth follows the input formula — a chain of `store`s is exactly the
    /// shape this tactic exists for — so a recursive walk could overflow the
    /// native stack on a deep formula, and would re-walk shared sub-DAGs once
    /// per incoming edge.
    ///
    /// This phase deliberately does *not* materialize chains. Recording a
    /// [`StoreChain`] per store node re-walked the whole spine from each of its
    /// nodes, which is quadratic in time and memory in the chain length; the
    /// sole consumer ([`Self::apply_extensionality`]) asks about a handful of
    /// terms, so chains are built on demand by [`Self::store_chain`]. Terms are
    /// immutable once interned, so an on-demand chain is identical to the one
    /// this phase used to precompute.
    fn collect_store_terms(&mut self, tid: TermId, tm: &TermManager) -> Result<(), String> {
        let mut stack = vec![tid];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }

            let term = tm.get(current).ok_or("term not found")?;
            let kind = &term.kind;

            // Only stores are recorded; every other kind contributes nothing
            // but its children, which `get_children` supplies below.
            if let TermKind::Store(_, _, _) = kind {
                self.store_terms.insert(current);
            }

            stack.extend(self.get_children(kind));
        }

        Ok(())
    }

    /// The store chain of `tid`, if `tid` is a store term seen by
    /// [`Self::collect_store_terms`].
    ///
    /// Chains are memoized and handed out behind an [`Rc`], so repeated
    /// extensionality checks over the same array share one materialization.
    fn store_chain(
        &mut self,
        tid: TermId,
        tm: &TermManager,
    ) -> Result<Option<Rc<StoreChain>>, String> {
        if !self.store_terms.contains(&tid) {
            return Ok(None);
        }

        if let Some(chain) = self.chain_cache.get(&tid) {
            return Ok(Some(Rc::clone(chain)));
        }

        let chain = Rc::new(Self::analyze_store_chain(tid, tm)?);
        self.chain_cache.insert(tid, Rc::clone(&chain));
        Ok(Some(chain))
    }

    /// Analyze a store chain starting from a store term.
    fn analyze_store_chain(store_tid: TermId, tm: &TermManager) -> Result<StoreChain, String> {
        let mut stores = Vec::new();
        let mut current = store_tid;

        loop {
            let term = tm.get(current).ok_or("term not found")?;

            match &term.kind {
                TermKind::Store(array, index, value) => {
                    stores.push((*index, *value));
                    current = *array;
                }
                _ => {
                    // Reached base array
                    return Ok(StoreChain {
                        base: current,
                        stores,
                    });
                }
            }
        }
    }

    /// Rewrite a term applying select-store simplifications.
    ///
    /// Iterative (explicit heap stack): the walk is bottom-up — a node is
    /// rewritten only once its children have been, exactly as the recursive
    /// version did — but the pending work lives on the heap, so a deeply
    /// nested formula can no longer exhaust the native stack. The rewrite
    /// cache is consulted on entry and filled on completion, as before.
    fn rewrite(&mut self, tid: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        /// Work item for the iterative rewriter.
        enum RewriteTask {
            /// Rewrite this term (its children first).
            Enter(TermId),
            /// Rebuild a `select` from the top two results.
            BuildSelect(TermId),
            /// Rebuild a `store` from the top three results.
            BuildStore(TermId),
            /// Rebuild an equality (extensionality) from the top two results.
            BuildEq(TermId),
            /// Rebuild an `and` from the top `n` results.
            BuildAnd(TermId, usize),
            /// Rebuild an `or` from the top `n` results.
            BuildOr(TermId, usize),
            /// Rebuild a `not` from the single result on top.
            BuildNot(TermId),
        }

        let mut tasks = vec![RewriteTask::Enter(tid)];
        let mut results: Vec<TermId> = Vec::new();

        while let Some(task) = tasks.pop() {
            match task {
                RewriteTask::Enter(current) => {
                    if let Some(&cached) = self.cache.get(&current) {
                        results.push(cached);
                        continue;
                    }

                    let term = tm.get(current).ok_or("term not found")?;
                    let kind = term.kind.clone();

                    match kind {
                        TermKind::Select(array, index) => {
                            tasks.push(RewriteTask::BuildSelect(current));
                            tasks.push(RewriteTask::Enter(index));
                            tasks.push(RewriteTask::Enter(array));
                        }
                        TermKind::Store(array, index, value) => {
                            tasks.push(RewriteTask::BuildStore(current));
                            tasks.push(RewriteTask::Enter(value));
                            tasks.push(RewriteTask::Enter(index));
                            tasks.push(RewriteTask::Enter(array));
                        }
                        TermKind::Eq(lhs, rhs) => {
                            tasks.push(RewriteTask::BuildEq(current));
                            tasks.push(RewriteTask::Enter(rhs));
                            tasks.push(RewriteTask::Enter(lhs));
                        }
                        TermKind::And(args) => {
                            tasks.push(RewriteTask::BuildAnd(current, args.len()));
                            tasks.extend(args.iter().rev().map(|&a| RewriteTask::Enter(a)));
                        }
                        TermKind::Or(args) => {
                            tasks.push(RewriteTask::BuildOr(current, args.len()));
                            tasks.extend(args.iter().rev().map(|&a| RewriteTask::Enter(a)));
                        }
                        TermKind::Not(arg) => {
                            tasks.push(RewriteTask::BuildNot(current));
                            tasks.push(RewriteTask::Enter(arg));
                        }
                        // Every other term kind is left as-is, matching the
                        // recursive version's `rewrite_children` fall-through.
                        _ => self.finish_rewrite(current, current, &mut results),
                    }
                }
                RewriteTask::BuildSelect(current) => {
                    let (array, index) = Self::take_two(&mut results)?;
                    let result = self.simplify_select_store(array, index, tm)?;
                    self.finish_rewrite(current, result, &mut results);
                }
                RewriteTask::BuildStore(current) => {
                    let value = Self::take_one(&mut results)?;
                    let (array, index) = Self::take_two(&mut results)?;
                    let result = self.simplify_store_store(array, index, value, tm)?;
                    self.finish_rewrite(current, result, &mut results);
                }
                RewriteTask::BuildEq(current) => {
                    let (lhs, rhs) = Self::take_two(&mut results)?;
                    let result = self.apply_extensionality(lhs, rhs, tm)?;
                    self.finish_rewrite(current, result, &mut results);
                }
                RewriteTask::BuildAnd(current, arity) => {
                    let args = Self::take_n(&mut results, arity)?;
                    let result = tm.mk_and(args);
                    self.finish_rewrite(current, result, &mut results);
                }
                RewriteTask::BuildOr(current, arity) => {
                    let args = Self::take_n(&mut results, arity)?;
                    let result = tm.mk_or(args);
                    self.finish_rewrite(current, result, &mut results);
                }
                RewriteTask::BuildNot(current) => {
                    let arg = Self::take_one(&mut results)?;
                    let result = tm.mk_not(arg);
                    self.finish_rewrite(current, result, &mut results);
                }
            }
        }

        results
            .pop()
            .ok_or_else(|| "rewrite produced no result".to_string())
    }

    /// Record a completed rewrite and make it available to the parent task.
    fn finish_rewrite(&mut self, tid: TermId, result: TermId, results: &mut Vec<TermId>) {
        // A shared DAG node can be entered twice before either completion, so
        // only the first completion counts towards the statistic.
        if self.cache.insert(tid, result).is_none() && result != tid {
            self.stats.terms_rewritten += 1;
        }
        results.push(result);
    }

    /// Detach the single most recent result.
    fn take_one(results: &mut Vec<TermId>) -> Result<TermId, String> {
        results
            .pop()
            .ok_or_else(|| "rewrite result stack underflow".to_string())
    }

    /// Detach the two most recent results, in their original order.
    fn take_two(results: &mut Vec<TermId>) -> Result<(TermId, TermId), String> {
        let second = Self::take_one(results)?;
        let first = Self::take_one(results)?;
        Ok((first, second))
    }

    /// Detach the `n` most recent results, in their original order.
    fn take_n(results: &mut Vec<TermId>, n: usize) -> Result<Vec<TermId>, String> {
        if results.len() < n {
            return Err("rewrite result stack underflow".to_string());
        }
        let start = results.len() - n;
        Ok(results.split_off(start))
    }

    /// Simplify select(store(...), index) patterns.
    fn simplify_select_store(
        &mut self,
        array: TermId,
        index: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        let array_term = tm.get(array).ok_or("array term not found")?;

        match &array_term.kind {
            TermKind::Store(inner_array, store_index, store_value) => {
                // Check if indices are equal
                if self.indices_equal(index, *store_index, tm)? {
                    // select(store(a, i, v), i) → v
                    self.stats.select_store_same_index += 1;
                    return Ok(*store_value);
                }

                // Check if indices are definitely different
                if self.indices_disjoint(index, *store_index, tm)? {
                    // select(store(a, i, v), j) → select(a, j) when i ≠ j
                    self.stats.select_store_diff_index += 1;
                    return Ok(tm.mk_select(*inner_array, index));
                }

                // Can't simplify, reconstruct
                Ok(tm.mk_select(array, index))
            }
            _ => {
                // No simplification
                Ok(tm.mk_select(array, index))
            }
        }
    }

    /// Simplify store(store(...), index, value) patterns.
    fn simplify_store_store(
        &mut self,
        array: TermId,
        index: TermId,
        value: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        let array_term = tm.get(array).ok_or("array term not found")?;

        match &array_term.kind {
            TermKind::Store(inner_array, inner_index, _inner_value) => {
                // Check if indices are equal
                if self.indices_equal(index, *inner_index, tm)? {
                    // store(store(a, i, v1), i, v2) → store(a, i, v2)
                    self.stats.redundant_store_elim += 1;
                    return Ok(tm.mk_store(*inner_array, index, value));
                }

                // Can't simplify, reconstruct
                Ok(tm.mk_store(array, index, value))
            }
            _ => {
                // No simplification
                Ok(tm.mk_store(array, index, value))
            }
        }
    }

    /// Apply array extensionality: (∀i. select(a,i) = select(b,i)) ⇒ a = b.
    ///
    /// Both sides are the *already rewritten* operands: the iterative
    /// [`Self::rewrite`] schedules them as child tasks, so rewriting them again
    /// here would repeat the whole sub-walk.
    fn apply_extensionality(
        &mut self,
        lhs_rewritten: TermId,
        rhs_rewritten: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        // Check if both are arrays with known stores
        let lhs_chain = self.store_chain(lhs_rewritten, tm)?;
        let rhs_chain = self.store_chain(rhs_rewritten, tm)?;
        if let (Some(lhs_chain), Some(rhs_chain)) = (lhs_chain, rhs_chain) {
            // If they have the same base and same stores, they're equal
            if lhs_chain.base == rhs_chain.base
                && self.stores_equivalent(&lhs_chain.stores, &rhs_chain.stores, tm)?
            {
                self.stats.extensionality_apps += 1;
                return Ok(tm.mk_true());
            }
        }

        // No simplification, reconstruct
        Ok(tm.mk_eq(lhs_rewritten, rhs_rewritten))
    }

    /// Check if two store sequences are equivalent.
    fn stores_equivalent(
        &self,
        stores1: &[(TermId, TermId)],
        stores2: &[(TermId, TermId)],
        tm: &TermManager,
    ) -> Result<bool, String> {
        if stores1.len() != stores2.len() {
            return Ok(false);
        }

        // Create maps for easier comparison
        let mut map1: FxHashMap<TermId, TermId> = FxHashMap::default();
        let mut map2: FxHashMap<TermId, TermId> = FxHashMap::default();

        for (idx, val) in stores1 {
            map1.insert(*idx, *val);
        }

        for (idx, val) in stores2 {
            map2.insert(*idx, *val);
        }

        // Check if all indices map to the same values
        for (idx, val1) in &map1 {
            if let Some(val2) = map2.get(idx) {
                if !self.values_equal(*val1, *val2, tm)? {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if two indices are equal.
    fn indices_equal(&self, idx1: TermId, idx2: TermId, tm: &TermManager) -> Result<bool, String> {
        if idx1 == idx2 {
            return Ok(true);
        }

        // Syntactic equality
        let term1 = tm.get(idx1).ok_or("term not found")?;
        let term2 = tm.get(idx2).ok_or("term not found")?;

        // Check if both are constants with the same value
        match (&term1.kind, &term2.kind) {
            (TermKind::IntConst(v1), TermKind::IntConst(v2)) => Ok(v1 == v2),
            (
                TermKind::BitVecConst {
                    value: v1,
                    width: w1,
                },
                TermKind::BitVecConst {
                    value: v2,
                    width: w2,
                },
            ) => Ok(v1 == v2 && w1 == w2),
            _ => Ok(false),
        }
    }

    /// Check if two indices are definitely disjoint.
    fn indices_disjoint(
        &self,
        idx1: TermId,
        idx2: TermId,
        tm: &TermManager,
    ) -> Result<bool, String> {
        if idx1 == idx2 {
            return Ok(false);
        }

        let term1 = tm.get(idx1).ok_or("term not found")?;
        let term2 = tm.get(idx2).ok_or("term not found")?;

        // Check if both are different constants
        match (&term1.kind, &term2.kind) {
            (TermKind::IntConst(v1), TermKind::IntConst(v2)) => Ok(v1 != v2),
            (
                TermKind::BitVecConst {
                    value: v1,
                    width: w1,
                },
                TermKind::BitVecConst {
                    value: v2,
                    width: w2,
                },
            ) => {
                if w1 != w2 {
                    return Ok(false);
                }
                Ok(v1 != v2)
            }
            _ => Ok(false),
        }
    }

    /// Check if two values are equal.
    fn values_equal(&self, val1: TermId, val2: TermId, tm: &TermManager) -> Result<bool, String> {
        self.indices_equal(val1, val2, tm)
    }

    /// Get children of a term kind.
    fn get_children(&self, kind: &TermKind) -> Vec<TermId> {
        match kind {
            TermKind::And(args) | TermKind::Or(args) => args.to_vec(),
            TermKind::Not(arg) => vec![*arg],
            TermKind::Eq(l, r) | TermKind::Le(l, r) | TermKind::Lt(l, r) => vec![*l, *r],
            TermKind::Select(a, i) => vec![*a, *i],
            TermKind::Store(a, i, v) => vec![*a, *i, *v],
            _ => vec![],
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &SelectStoreElimStats {
        &self.stats
    }
}

impl Default for SelectStoreElimTactic {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_store_elim_tactic() {
        let tactic = SelectStoreElimTactic::new();
        assert_eq!(tactic.stats.select_store_same_index, 0);
    }

    #[test]
    fn test_store_chain_analysis() {
        let _tactic = SelectStoreElimTactic::new();
        let chain = StoreChain {
            base: TermId::from(0),
            stores: vec![(TermId::from(1), TermId::from(2))],
        };
        assert_eq!(chain.stores.len(), 1);
    }

    /// Run `body` on a worker thread with a deliberately small (128 KiB) stack,
    /// so a recursive walk over a deep term would abort instead of getting
    /// away with the main thread's much larger stack.
    fn run_with_small_stack<F>(body: F)
    where
        F: FnOnce() + Send + 'static,
    {
        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(body)
            .expect("thread spawn should succeed")
            .join()
            .expect("deep-nesting walk must not overflow the stack");
    }

    #[test]
    fn test_rewrite_and_chain_building_handle_deeply_nested_terms() {
        run_with_small_stack(|| {
            const DEPTH: usize = 6_250;

            let mut tm = TermManager::new();
            let bool_sort = tm.sorts.bool_sort;
            let int_sort = tm.sorts.int_sort;
            let x = tm.mk_var("x", int_sort);
            let zero = tm.mk_int(0);

            // A 50k-deep negation chain. `intern_term` bypasses `mk_not`'s
            // double-negation folding so the depth is real.
            let base = tm.mk_le(zero, x);
            let mut current = base;
            for _ in 0..DEPTH {
                current = tm.intern_term(TermKind::Not(current), bool_sort);
            }

            let mut tactic = SelectStoreElimTactic::new();
            let rewritten = tactic
                .apply(current, &mut tm)
                .expect("deep rewrite should succeed");

            // The rebuild goes through `mk_not`, which folds double negations,
            // so the chain collapses back to the base comparison (DEPTH is
            // even). The point of the test is that neither phase recursed.
            assert_eq!(rewritten, base);
        });
    }

    /// Reference implementation of the original eager chain-building phase:
    /// every reachable store node gets its own fully materialized
    /// [`StoreChain`]. Used to pin the lazy path's chains against the old
    /// path's, which is the only thing extensionality ever observed.
    fn eager_store_chains(
        tactic: &SelectStoreElimTactic,
        tid: TermId,
        tm: &TermManager,
    ) -> FxHashMap<TermId, StoreChain> {
        let mut chains = FxHashMap::default();
        let mut stack = vec![tid];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            let Some(term) = tm.get(current) else {
                continue;
            };
            if let TermKind::Store(_, _, _) = &term.kind {
                let chain = SelectStoreElimTactic::analyze_store_chain(current, tm)
                    .expect("chain analysis should succeed");
                chains.insert(current, chain);
            }
            stack.extend(tactic.get_children(&term.kind));
        }

        chains
    }

    /// A formula with nested stores, aliasing and non-aliasing constant
    /// indices, a variable index, selects, and a boolean skeleton.
    fn nontrivial_nest(tm: &mut TermManager) -> TermId {
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let a = tm.mk_var("a", array_sort);
        let i = tm.mk_var("i", int_sort);
        let one = tm.mk_int(1);
        let two = tm.mk_int(2);
        let ten = tm.mk_int(10);
        let twenty = tm.mk_int(20);

        let s1 = tm.mk_store(a, one, ten);
        let s2 = tm.mk_store(s1, two, twenty);
        let s3 = tm.mk_store(s2, i, ten);

        // Same base, same index→value map, opposite nesting order.
        let permuted_inner = tm.mk_store(a, two, twenty);
        let permuted = tm.mk_store(permuted_inner, one, ten);
        let ext = tm.mk_eq(s2, permuted);

        let sel_same = tm.mk_select(s2, two);
        let sel_diff = tm.mk_select(s2, one);
        let sel_unknown = tm.mk_select(s3, one);

        let eq1 = tm.mk_eq(sel_same, twenty);
        let eq2 = tm.mk_eq(sel_diff, ten);
        let eq3 = tm.mk_eq(sel_unknown, ten);
        let inner = tm.mk_or(vec![eq2, eq3]);
        let negated = tm.mk_not(inner);
        tm.mk_and(vec![eq1, negated, ext])
    }

    #[test]
    fn test_lazy_chains_match_the_eager_chains_on_a_nontrivial_nest() {
        let mut tm = TermManager::new();
        let formula = nontrivial_nest(&mut tm);

        let mut tactic = SelectStoreElimTactic::new();
        let expected = eager_store_chains(&tactic, formula, &tm);
        assert!(
            expected.len() >= 5,
            "the pinned nest must contain several store nodes"
        );

        tactic
            .collect_store_terms(formula, &tm)
            .expect("scan should succeed");

        // Same set of store terms...
        let mut recorded: Vec<TermId> = tactic.store_terms.iter().copied().collect();
        let mut eager: Vec<TermId> = expected.keys().copied().collect();
        recorded.sort();
        eager.sort();
        assert_eq!(recorded, eager);

        // ...and, for each of them, the very same chain.
        for (tid, chain) in &expected {
            let lazy = tactic
                .store_chain(*tid, &tm)
                .expect("chain analysis should succeed")
                .expect("a recorded store term must have a chain");
            assert_eq!(lazy.base, chain.base);
            assert_eq!(lazy.stores, chain.stores);
        }

        // Non-store terms still have no chain, as before.
        let non_store = tm.mk_int(1);
        assert!(
            tactic
                .store_chain(non_store, &tm)
                .expect("lookup should succeed")
                .is_none()
        );
    }

    #[test]
    fn test_extensionality_output_is_pinned() {
        // Permuted stores over disjoint constant indices on a shared base:
        // the two arrays are equal, and the tactic must say so.
        let mut tm = TermManager::new();
        let formula = nontrivial_nest(&mut tm);

        let mut tactic = SelectStoreElimTactic::new();
        let rewritten = tactic.apply(formula, &mut tm).expect("rewrite succeeds");

        let expected = {
            let int_sort = tm.sorts.int_sort;
            let array_sort = tm.sorts.array(int_sort, int_sort);
            let a = tm.mk_var("a", array_sort);
            let i = tm.mk_var("i", int_sort);
            let one = tm.mk_int(1);
            let two = tm.mk_int(2);
            let ten = tm.mk_int(10);
            let twenty = tm.mk_int(20);
            let s1 = tm.mk_store(a, one, ten);
            let s2 = tm.mk_store(s1, two, twenty);
            let s3 = tm.mk_store(s2, i, ten);

            // select(store(_, 2, 20), 2) → 20.
            let eq1 = tm.mk_eq(twenty, twenty);
            // select(store(_, 2, 20), 1) → select(store(a, 1, 10), 1): the
            // pass peels exactly one level, it does not re-simplify.
            let sel_peeled = tm.mk_select(s1, one);
            let eq2 = tm.mk_eq(sel_peeled, ten);
            // select(store(_, i, 10), 1) is untouched: `i` may alias 1.
            let sel_unknown = tm.mk_select(s3, one);
            let eq3 = tm.mk_eq(sel_unknown, ten);
            let inner = tm.mk_or(vec![eq2, eq3]);
            let negated = tm.mk_not(inner);
            // The permuted stores are provably the same array.
            let ext = tm.mk_true();
            tm.mk_and(vec![eq1, negated, ext])
        };
        assert_eq!(rewritten, expected);
        assert_eq!(tactic.stats().extensionality_apps, 1);
        assert_eq!(tactic.stats().select_store_same_index, 1);
        assert_eq!(tactic.stats().select_store_diff_index, 1);
    }

    #[test]
    fn test_extensionality_does_not_fire_on_different_bases() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let a = tm.mk_var("a", array_sort);
        let b = tm.mk_var("b", array_sort);
        let one = tm.mk_int(1);
        let ten = tm.mk_int(10);
        let lhs = tm.mk_store(a, one, ten);
        let rhs = tm.mk_store(b, one, ten);
        let formula = tm.mk_eq(lhs, rhs);

        let mut tactic = SelectStoreElimTactic::new();
        let rewritten = tactic.apply(formula, &mut tm).expect("rewrite succeeds");
        assert_eq!(rewritten, formula);
        assert_eq!(tactic.stats().extensionality_apps, 0);
    }

    #[test]
    fn test_deep_store_chain_is_walked_iteratively() {
        run_with_small_stack(|| {
            // 50k stores: with chains materialized per store node this was
            // quadratic and had to be capped at 1k. Chains are now built only
            // when extensionality asks for one, so both phases are linear.
            const CHAIN: usize = 6_250;

            let mut tm = TermManager::new();
            let int_sort = tm.sorts.int_sort;
            let array_sort = tm.sorts.array(int_sort, int_sort);
            let base = tm.mk_var("a", array_sort);

            let mut array = base;
            for i in 0..CHAIN {
                let index = tm.mk_int(i as i64);
                let value = tm.mk_int((i * 2) as i64);
                array = tm.mk_store(array, index, value);
            }

            let probe = tm.mk_int(0);
            let selected = tm.mk_select(array, probe);

            let mut tactic = SelectStoreElimTactic::new();
            let rewritten = tactic
                .apply(selected, &mut tm)
                .expect("deep store-chain rewrite should succeed");

            // The outermost store has a provably distinct constant index, so
            // one level is peeled off (this pass peels a single level).
            assert_ne!(rewritten, selected);
            assert_eq!(tactic.stats().select_store_diff_index, 1);
        });
    }

    #[test]
    fn test_select_over_store_same_index_yields_value() {
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let base = tm.mk_var("a", array_sort);
        let index = tm.mk_var("i", int_sort);
        let value = tm.mk_int(42);

        let stored = tm.mk_store(base, index, value);
        let selected = tm.mk_select(stored, index);

        let mut tactic = SelectStoreElimTactic::new();
        let rewritten = tactic.apply(selected, &mut tm).expect("rewrite succeeds");
        assert_eq!(rewritten, value);
        assert_eq!(tactic.stats().select_store_same_index, 1);
    }

    #[test]
    fn test_select_over_unknown_index_is_not_peeled() {
        // select(store(a, i, v), j) with i, j unrelated variables must keep
        // the store: the indices may alias.
        let mut tm = TermManager::new();
        let int_sort = tm.sorts.int_sort;
        let array_sort = tm.sorts.array(int_sort, int_sort);
        let base = tm.mk_var("a", array_sort);
        let i = tm.mk_var("i", int_sort);
        let j = tm.mk_var("j", int_sort);
        let value = tm.mk_int(42);

        let stored = tm.mk_store(base, i, value);
        let selected = tm.mk_select(stored, j);

        let mut tactic = SelectStoreElimTactic::new();
        let rewritten = tactic.apply(selected, &mut tm).expect("rewrite succeeds");
        assert_eq!(rewritten, selected);
        assert_eq!(tactic.stats().select_store_same_index, 0);
        assert_eq!(tactic.stats().select_store_diff_index, 0);
    }

    #[test]
    fn test_stats() {
        let mut tactic = SelectStoreElimTactic::new();
        tactic.stats.select_store_same_index = 5;
        tactic.stats.redundant_store_elim = 3;

        assert_eq!(tactic.stats().select_store_same_index, 5);
        assert_eq!(tactic.stats().redundant_store_elim, 3);
    }
}
