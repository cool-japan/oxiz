//! Select-Store Elimination Tactic for Array Theory.
//!
//! Simplifies array expressions by eliminating redundant select-store patterns:
//! - select(store(a, i, v), i) → v
//! - select(store(a, i, v), j) → select(a, j) when i ≠ j
//! - store(store(a, i, v1), i, v2) → store(a, i, v2)
//! - Array extensionality reasoning

use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashMap;

/// Select-store elimination tactic.
pub struct SelectStoreElimTactic {
    /// Rewrite cache
    cache: FxHashMap<TermId, TermId>,
    /// Store chain analysis
    store_chains: FxHashMap<TermId, StoreChain>,
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
            store_chains: FxHashMap::default(),
            stats: SelectStoreElimStats::default(),
        }
    }

    /// Apply the tactic to a formula.
    pub fn apply(&mut self, formula: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        // Phase 1: Build store chains
        self.build_store_chains(formula, tm)?;

        // Phase 2: Rewrite formula
        let rewritten = self.rewrite(formula, tm)?;

        Ok(rewritten)
    }

    /// Build store chains for all array terms.
    fn build_store_chains(&mut self, tid: TermId, tm: &TermManager) -> Result<(), String> {
        let term = tm.get(tid).ok_or("term not found")?;

        match &term.kind {
            TermKind::Store(array, index, value) => {
                // Recursively build chains for nested stores
                self.build_store_chains(*array, tm)?;
                self.build_store_chains(*index, tm)?;
                self.build_store_chains(*value, tm)?;

                // Analyze this store
                let chain = self.analyze_store_chain(tid, tm)?;
                self.store_chains.insert(tid, chain);
            }
            _ => {
                // Recursively process children
                for child in self.get_children(&term.kind) {
                    self.build_store_chains(child, tm)?;
                }
            }
        }

        Ok(())
    }

    /// Analyze a store chain starting from a store term.
    fn analyze_store_chain(
        &self,
        store_tid: TermId,
        tm: &TermManager,
    ) -> Result<StoreChain, String> {
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
    fn rewrite(&mut self, tid: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        // Check cache
        if let Some(&cached) = self.cache.get(&tid) {
            return Ok(cached);
        }

        let term = tm.get(tid).ok_or("term not found")?;
        let kind = term.kind.clone();
        let result = match &kind {
            TermKind::Select(array, index) => {
                // Rewrite array and index first
                let array_rewritten = self.rewrite(*array, tm)?;
                let index_rewritten = self.rewrite(*index, tm)?;

                // Try to simplify select-store pattern
                self.simplify_select_store(array_rewritten, index_rewritten, tm)?
            }

            TermKind::Store(array, index, value) => {
                // Rewrite components
                let array_rewritten = self.rewrite(*array, tm)?;
                let index_rewritten = self.rewrite(*index, tm)?;
                let value_rewritten = self.rewrite(*value, tm)?;

                // Try to eliminate redundant stores
                self.simplify_store_store(array_rewritten, index_rewritten, value_rewritten, tm)?
            }

            TermKind::Eq(lhs, rhs) => {
                // Check for array extensionality
                self.apply_extensionality(*lhs, *rhs, tm)?
            }

            _ => {
                // Recursively rewrite children
                self.rewrite_children(tid, tm)?
            }
        };

        self.cache.insert(tid, result);
        if result != tid {
            self.stats.terms_rewritten += 1;
        }

        Ok(result)
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
    fn apply_extensionality(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        tm: &mut TermManager,
    ) -> Result<TermId, String> {
        // Rewrite both sides
        let lhs_rewritten = self.rewrite(lhs, tm)?;
        let rhs_rewritten = self.rewrite(rhs, tm)?;

        // Check if both are arrays with known stores
        if let (Some(lhs_chain), Some(rhs_chain)) = (
            self.store_chains.get(&lhs_rewritten),
            self.store_chains.get(&rhs_rewritten),
        ) {
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

    /// Rewrite children of a term.
    fn rewrite_children(&mut self, tid: TermId, tm: &mut TermManager) -> Result<TermId, String> {
        let term = tm.get(tid).ok_or("term not found")?;
        let kind = term.kind.clone();

        match kind {
            TermKind::And(args) => {
                let mut new_args = Vec::new();
                for arg in args {
                    new_args.push(self.rewrite(arg, tm)?);
                }
                Ok(tm.mk_and(new_args))
            }
            TermKind::Or(args) => {
                let mut new_args = Vec::new();
                for arg in args {
                    new_args.push(self.rewrite(arg, tm)?);
                }
                Ok(tm.mk_or(new_args))
            }
            TermKind::Not(arg) => {
                let new_arg = self.rewrite(arg, tm)?;
                Ok(tm.mk_not(new_arg))
            }
            _ => Ok(tid),
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

    #[test]
    fn test_stats() {
        let mut tactic = SelectStoreElimTactic::new();
        tactic.stats.select_store_same_index = 5;
        tactic.stats.redundant_store_elim = 3;

        assert_eq!(tactic.stats().select_store_same_index, 5);
        assert_eq!(tactic.stats().redundant_store_elim, 3);
    }
}
