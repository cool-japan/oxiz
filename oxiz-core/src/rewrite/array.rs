//! Array Term Rewriter
//!
//! This module provides rewriting rules for array expressions:
//! - Select-Store simplification (select(store(a, i, v), i) → v)
//! - Store-Store merging (store(store(a, i, v1), i, v2) → store(a, i, v2))
//! - Array extensionality handling

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::ast::{TermId, TermKind, TermManager};

/// Array rewriter
#[derive(Debug, Clone)]
pub struct ArrayRewriter {
    /// Enable store-store merging
    pub merge_stores: bool,
    /// Enable array map simplification
    pub simplify_map: bool,
}

impl Default for ArrayRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ArrayRewriter {
    /// Create a new array rewriter
    pub fn new() -> Self {
        Self {
            merge_stores: true,
            simplify_map: true,
        }
    }

    /// Check if two terms are syntactically equal
    fn terms_equal(&self, t1: TermId, t2: TermId) -> bool {
        t1 == t2
    }

    /// Check if two integer constants are different
    fn indices_different(&self, idx1: TermId, idx2: TermId, manager: &TermManager) -> bool {
        if let (Some(t1), Some(t2)) = (manager.get(idx1), manager.get(idx2)) {
            match (&t1.kind, &t2.kind) {
                (TermKind::IntConst(a), TermKind::IntConst(b)) => a != b,
                (
                    TermKind::BitVecConst {
                        value: a,
                        width: w1,
                    },
                    TermKind::BitVecConst {
                        value: b,
                        width: w2,
                    },
                ) => a != b || w1 != w2,
                _ => false,
            }
        } else {
            false
        }
    }

    /// Rewrite array select
    fn rewrite_select(
        &mut self,
        array: TermId,
        index: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(arr_term) = manager.get(array).cloned() else {
            return RewriteResult::Unchanged(manager.mk_select(array, index));
        };

        match &arr_term.kind {
            // select(store(a, i, v), i) → v  (same index)
            TermKind::Store(_, stored_idx, value) if self.terms_equal(*stored_idx, index) => {
                ctx.stats_mut().record_rule("array_select_store_same");
                RewriteResult::Rewritten(*value)
            }

            // select(store(a, j, v), i) where i ≠ j → select(a, i)
            // This requires disequality proving, so we only do it for constant indices
            TermKind::Store(inner_array, stored_idx, _) => {
                if self.indices_different(index, *stored_idx, manager) {
                    ctx.stats_mut().record_rule("array_select_store_diff");
                    return RewriteResult::Rewritten(manager.mk_select(*inner_array, index));
                }
                RewriteResult::Unchanged(manager.mk_select(array, index))
            }

            // select(ite(c, a1, a2), i) → ite(c, select(a1, i), select(a2, i))
            // Only apply if we're simplifying aggressively
            TermKind::Ite(cond, then_arr, else_arr) => {
                ctx.stats_mut().record_rule("array_select_ite");
                let then_sel = manager.mk_select(*then_arr, index);
                let else_sel = manager.mk_select(*else_arr, index);
                RewriteResult::Rewritten(manager.mk_ite(*cond, then_sel, else_sel))
            }

            _ => RewriteResult::Unchanged(manager.mk_select(array, index)),
        }
    }

    /// Rewrite array store
    fn rewrite_store(
        &mut self,
        array: TermId,
        index: TermId,
        value: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(arr_term) = manager.get(array).cloned() else {
            return RewriteResult::Unchanged(manager.mk_store(array, index, value));
        };

        match &arr_term.kind {
            // store(store(a, i, v1), i, v2) → store(a, i, v2)  (merge stores to same index)
            TermKind::Store(inner_array, stored_idx, _)
                if self.merge_stores && self.terms_equal(*stored_idx, index) =>
            {
                ctx.stats_mut().record_rule("array_store_store_merge");
                RewriteResult::Rewritten(manager.mk_store(*inner_array, index, value))
            }

            // store(a, i, select(a, i)) → a  (storing same value from array)
            _ => {
                if let Some(val_term) = manager.get(value)
                    && let TermKind::Select(sel_array, sel_idx) = &val_term.kind
                    && self.terms_equal(*sel_array, array)
                    && self.terms_equal(*sel_idx, index)
                {
                    ctx.stats_mut().record_rule("array_store_select_same");
                    return RewriteResult::Rewritten(array);
                }
                RewriteResult::Unchanged(manager.mk_store(array, index, value))
            }
        }
    }

    /// Rewrite array equality
    fn rewrite_array_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // a = a → true
        if self.terms_equal(lhs, rhs) {
            ctx.stats_mut().record_rule("array_eq_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        RewriteResult::Unchanged(manager.mk_eq(lhs, rhs))
    }

    /// Check if a term is an array term
    fn is_array_term(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(t.kind, TermKind::Select(_, _) | TermKind::Store(_, _, _))
    }
}

impl Rewriter for ArrayRewriter {
    fn rewrite(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(t) = manager.get(term).cloned() else {
            return RewriteResult::Unchanged(term);
        };

        match &t.kind {
            TermKind::Select(array, index) => self.rewrite_select(*array, *index, ctx, manager),
            TermKind::Store(array, index, value) => {
                self.rewrite_store(*array, *index, *value, ctx, manager)
            }
            TermKind::Eq(lhs, rhs) => {
                // Check if it's an array equality
                if self.is_array_term(*lhs, manager) || self.is_array_term(*rhs, manager) {
                    self.rewrite_array_eq(*lhs, *rhs, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "ArrayRewriter"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::Select(_, _) | TermKind::Store(_, _, _) | TermKind::Eq(_, _)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TermManager, RewriteContext, ArrayRewriter) {
        (
            TermManager::new(),
            RewriteContext::new(),
            ArrayRewriter::new(),
        )
    }

    #[test]
    fn test_select_store_same_index() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let int_sort = manager.sorts.int_sort;
        let arr_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", arr_sort);
        let i = manager.mk_var("i", int_sort);
        let v = manager.mk_int(42);

        let store = manager.mk_store(a, i, v);
        let select = manager.mk_select(store, i);

        let result = rewriter.rewrite(select, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), v);
    }

    #[test]
    fn test_select_store_different_const_index() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let int_sort = manager.sorts.int_sort;
        let arr_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", arr_sort);
        let i1 = manager.mk_int(1);
        let i2 = manager.mk_int(2);
        let v = manager.mk_int(42);

        let store = manager.mk_store(a, i1, v);
        let select = manager.mk_select(store, i2);

        let result = rewriter.rewrite(select, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        // Should be select(a, i2)
        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::Select(_, _)));
    }

    #[test]
    fn test_store_store_merge() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let int_sort = manager.sorts.int_sort;
        let arr_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", arr_sort);
        let i = manager.mk_var("i", int_sort);
        let v1 = manager.mk_int(1);
        let v2 = manager.mk_int(2);

        let store1 = manager.mk_store(a, i, v1);
        let store2 = manager.mk_store(store1, i, v2);

        let result = rewriter.rewrite(store2, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        // Should be store(a, i, v2)
        let t = manager.get(result.term()).expect("term should exist");
        if let TermKind::Store(arr, _, val) = t.kind {
            assert_eq!(arr, a);
            assert_eq!(val, v2);
        } else {
            panic!("Expected Store term");
        }
    }

    #[test]
    fn test_store_select_same() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let int_sort = manager.sorts.int_sort;
        let arr_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", arr_sort);
        let i = manager.mk_var("i", int_sort);

        let select = manager.mk_select(a, i);
        let store = manager.mk_store(a, i, select);

        let result = rewriter.rewrite(store, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), a);
    }

    #[test]
    fn test_array_eq_self() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_eq already simplifies eq(x, x) -> true at term creation
        let int_sort = manager.sorts.int_sort;
        let arr_sort = manager.sorts.array(int_sort, int_sort);
        let a = manager.mk_var("a", arr_sort);

        let eq = manager.mk_eq(a, a);

        // Verify the simplification happened at creation time
        let t = manager.get(eq).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }
}
