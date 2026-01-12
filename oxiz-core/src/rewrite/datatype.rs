//! Datatype/ADT Term Rewriter
//!
//! This module provides rewriting rules for algebraic data type expressions:
//! - Selector-constructor simplification: `sel_i(C(args)) → args[i]`
//! - Tester simplification: `is_C(C(args)) → true`
//! - Constructor equality simplification
//!
//! # Theory Background
//!
//! Algebraic Data Types (ADTs) in SMT-LIB support:
//! - Constructors: Create values of the datatype
//! - Selectors: Extract fields from constructed values
//! - Testers: Test which constructor was used

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::SortId;
use crate::ast::{TermId, TermKind, TermManager};
use lasso::Spur;
use rustc_hash::FxHashMap;

/// Datatype rewriter for ADT simplification
#[derive(Debug, Clone)]
pub struct DatatypeRewriter {
    /// Enable selector-constructor simplification
    pub simplify_selectors: bool,
    /// Enable tester simplification
    pub simplify_testers: bool,
    /// Enable constructor-based equality
    pub constructor_eq: bool,
    /// Cache for constructor-selector mapping: constructor name -> list of selector names
    selector_cache: FxHashMap<Spur, Vec<Spur>>,
    /// Cache for constructor-tester mapping: constructor name -> tester name
    tester_cache: FxHashMap<Spur, Spur>,
}

impl Default for DatatypeRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl DatatypeRewriter {
    /// Create a new datatype rewriter
    pub fn new() -> Self {
        Self {
            simplify_selectors: true,
            simplify_testers: true,
            constructor_eq: true,
            selector_cache: FxHashMap::default(),
            tester_cache: FxHashMap::default(),
        }
    }

    /// Create with all optimizations enabled
    pub fn aggressive() -> Self {
        Self {
            simplify_selectors: true,
            simplify_testers: true,
            constructor_eq: true,
            selector_cache: FxHashMap::default(),
            tester_cache: FxHashMap::default(),
        }
    }

    /// Register constructor info for selector lookup
    pub fn register_constructor(&mut self, constructor: Spur, selectors: Vec<Spur>, tester: Spur) {
        self.selector_cache.insert(constructor, selectors);
        self.tester_cache.insert(constructor, tester);
    }

    /// Check if a term is a constructor application
    fn is_constructor(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };
        matches!(t.kind, TermKind::DtConstructor { .. })
    }

    /// Get constructor name from a constructor application
    #[allow(dead_code)]
    fn get_constructor_name(&self, term: TermId, manager: &TermManager) -> Option<Spur> {
        let t = manager.get(term)?;
        match &t.kind {
            TermKind::DtConstructor { constructor, .. } => Some(*constructor),
            _ => None,
        }
    }

    /// Get constructor arguments
    #[allow(dead_code)]
    fn get_constructor_args(&self, term: TermId, manager: &TermManager) -> Option<Vec<TermId>> {
        let t = manager.get(term)?;
        match &t.kind {
            TermKind::DtConstructor { args, .. } => Some(args.to_vec()),
            _ => None,
        }
    }

    /// Find the index of a selector within a constructor
    fn find_selector_index(&self, constructor: Spur, selector: Spur) -> Option<usize> {
        if let Some(selectors) = self.selector_cache.get(&constructor) {
            return selectors.iter().position(|s| *s == selector);
        }
        None
    }

    /// Rewrite selector application
    /// sel_i(C(a0, a1, ..., an)) → a_i (if sel_i is the i-th selector of C)
    fn rewrite_selector(
        &mut self,
        selector: Spur,
        arg: TermId,
        result_sort: SortId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // Convert Spur to String upfront to avoid borrow issues
        let sel_str = manager.resolve_str(selector).to_string();

        if !self.simplify_selectors {
            return RewriteResult::Unchanged(manager.mk_dt_selector(&sel_str, arg, result_sort));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_dt_selector(&sel_str, arg, result_sort));
        };

        match &t.kind {
            TermKind::DtConstructor {
                constructor, args, ..
            } => {
                // Check if this selector corresponds to a field of this constructor
                if let Some(index) = self.find_selector_index(*constructor, selector)
                    && index < args.len()
                {
                    ctx.stats_mut().record_rule("dt_selector_constructor");
                    return RewriteResult::Rewritten(args[index]);
                }
                RewriteResult::Unchanged(manager.mk_dt_selector(&sel_str, arg, result_sort))
            }
            // sel(ite(c, t, e)) → ite(c, sel(t), sel(e))
            TermKind::Ite(cond, then_br, else_br) => {
                ctx.stats_mut().record_rule("dt_selector_ite");
                let sel_then = manager.mk_dt_selector(&sel_str, *then_br, result_sort);
                let sel_else = manager.mk_dt_selector(&sel_str, *else_br, result_sort);
                RewriteResult::Rewritten(manager.mk_ite(*cond, sel_then, sel_else))
            }
            _ => RewriteResult::Unchanged(manager.mk_dt_selector(&sel_str, arg, result_sort)),
        }
    }

    /// Rewrite tester application
    /// is_C(C(args)) → true
    /// is_C(D(args)) → false (if C ≠ D and same datatype)
    fn rewrite_tester(
        &mut self,
        constructor: Spur,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // Convert Spur to String upfront to avoid borrow issues
        let con_str = manager.resolve_str(constructor).to_string();

        if !self.simplify_testers {
            return RewriteResult::Unchanged(manager.mk_dt_tester(&con_str, arg));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_dt_tester(&con_str, arg));
        };

        match &t.kind {
            TermKind::DtConstructor {
                constructor: arg_con,
                ..
            } => {
                // Check if tester matches constructor
                if *arg_con == constructor {
                    ctx.stats_mut().record_rule("dt_tester_true");
                    return RewriteResult::Rewritten(manager.mk_true());
                }
                // Different constructors → false (assuming same datatype)
                ctx.stats_mut().record_rule("dt_tester_false");
                RewriteResult::Rewritten(manager.mk_false())
            }
            // is_C(ite(cond, t, e)) → ite(cond, is_C(t), is_C(e))
            TermKind::Ite(cond, then_br, else_br) => {
                ctx.stats_mut().record_rule("dt_tester_ite");
                let test_then = manager.mk_dt_tester(&con_str, *then_br);
                let test_else = manager.mk_dt_tester(&con_str, *else_br);
                RewriteResult::Rewritten(manager.mk_ite(*cond, test_then, test_else))
            }
            _ => RewriteResult::Unchanged(manager.mk_dt_tester(&con_str, arg)),
        }
    }

    /// Rewrite constructor equality
    /// C(a1, ..., an) = C(b1, ..., bn) → a1=b1 ∧ ... ∧ an=bn
    /// C(args) = D(args') → false (if C ≠ D)
    fn rewrite_constructor_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        if !self.constructor_eq {
            return RewriteResult::Unchanged(manager.mk_eq(lhs, rhs));
        }

        let (Some(lhs_t), Some(rhs_t)) = (manager.get(lhs).cloned(), manager.get(rhs).cloned())
        else {
            return RewriteResult::Unchanged(manager.mk_eq(lhs, rhs));
        };

        match (&lhs_t.kind, &rhs_t.kind) {
            (
                TermKind::DtConstructor {
                    constructor: lhs_con,
                    args: lhs_args,
                    ..
                },
                TermKind::DtConstructor {
                    constructor: rhs_con,
                    args: rhs_args,
                    ..
                },
            ) => {
                // Same constructor: decompose into field equalities
                if lhs_con == rhs_con && lhs_args.len() == rhs_args.len() {
                    if lhs_args.is_empty() {
                        // Nullary constructor: always equal
                        ctx.stats_mut().record_rule("dt_eq_nullary");
                        return RewriteResult::Rewritten(manager.mk_true());
                    }
                    ctx.stats_mut().record_rule("dt_eq_decompose");
                    let equalities: Vec<_> = lhs_args
                        .iter()
                        .zip(rhs_args.iter())
                        .map(|(&a, &b)| manager.mk_eq(a, b))
                        .collect();
                    return RewriteResult::Rewritten(manager.mk_and(equalities));
                }
                // Different constructors: always false
                if lhs_con != rhs_con {
                    ctx.stats_mut().record_rule("dt_eq_diff_constructor");
                    return RewriteResult::Rewritten(manager.mk_false());
                }
                RewriteResult::Unchanged(manager.mk_eq(lhs, rhs))
            }
            _ => RewriteResult::Unchanged(manager.mk_eq(lhs, rhs)),
        }
    }

    /// Check if this is an equality between datatype terms
    fn is_datatype_eq(&self, lhs: TermId, rhs: TermId, manager: &TermManager) -> bool {
        self.is_constructor(lhs, manager) || self.is_constructor(rhs, manager)
    }
}

impl Rewriter for DatatypeRewriter {
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
            TermKind::DtSelector { selector, arg } => {
                // Get the result sort from the term
                let result_sort = t.sort;
                self.rewrite_selector(*selector, *arg, result_sort, ctx, manager)
            }
            TermKind::DtTester { constructor, arg } => {
                self.rewrite_tester(*constructor, *arg, ctx, manager)
            }
            TermKind::Eq(lhs, rhs) => {
                if self.is_datatype_eq(*lhs, *rhs, manager) {
                    self.rewrite_constructor_eq(*lhs, *rhs, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "DatatypeRewriter"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::DtSelector { .. } | TermKind::DtTester { .. } | TermKind::Eq(_, _)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TermManager, RewriteContext, DatatypeRewriter) {
        (
            TermManager::new(),
            RewriteContext::new(),
            DatatypeRewriter::new(),
        )
    }

    #[test]
    fn test_selector_constructor() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let int_sort = manager.sorts.int_sort;
        let dt_sort = manager.sorts.mk_datatype_sort("Pair");

        let x = manager.mk_int(1);
        let y = manager.mk_int(2);

        let first = manager.intern_str("first");
        let second = manager.intern_str("second");
        let pair = manager.intern_str("Pair");
        let is_pair = manager.intern_str("is_Pair");

        // Register constructor info
        rewriter.register_constructor(pair, vec![first, second], is_pair);

        let constructor = manager.mk_dt_constructor("Pair", vec![x, y], dt_sort);
        let select = manager.mk_dt_selector("first", constructor, int_sort);

        let result = rewriter.rewrite(select, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_tester_true() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let dt_sort = manager.sorts.mk_datatype_sort("Color");
        let red = manager.intern_str("Red");
        let is_red = manager.intern_str("is_Red");

        // Register constructor info
        rewriter.register_constructor(red, vec![], is_red);

        let constructor = manager.mk_dt_constructor("Red", vec![], dt_sort);
        let tester = manager.mk_dt_tester("Red", constructor);

        let result = rewriter.rewrite(tester, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_tester_false() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let dt_sort = manager.sorts.mk_datatype_sort("Color");
        let _red = manager.intern_str("Red");

        let constructor = manager.mk_dt_constructor("Red", vec![], dt_sort);
        let tester = manager.mk_dt_tester("Blue", constructor);

        let result = rewriter.rewrite(tester, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::False));
    }

    #[test]
    fn test_constructor_eq_same() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let dt_sort = manager.sorts.mk_datatype_sort("Pair");
        let int_sort = manager.sorts.int_sort;

        // Use variables instead of constants to avoid mk_eq simplifying the equalities
        let x1 = manager.mk_var("x1", int_sort);
        let y1 = manager.mk_var("y1", int_sort);
        let x2 = manager.mk_var("x2", int_sort);
        let y2 = manager.mk_var("y2", int_sort);

        let con1 = manager.mk_dt_constructor("Pair", vec![x1, y1], dt_sort);
        let con2 = manager.mk_dt_constructor("Pair", vec![x2, y2], dt_sort);
        let eq = manager.mk_eq(con1, con2);

        // The eq term should be an Eq with two DtConstructors
        let eq_term = manager.get(eq).expect("eq term");
        assert!(
            matches!(eq_term.kind, TermKind::Eq(_, _)),
            "Expected Eq term, got {:?}",
            eq_term.kind
        );

        let result = rewriter.rewrite(eq, &mut ctx, &mut manager);
        assert!(result.was_rewritten(), "Expected rewrite to happen");

        // Should decompose to field equalities: x1==x2 ∧ y1==y2
        let t = manager.get(result.term()).expect("term should exist");
        assert!(
            matches!(t.kind, TermKind::And(_)),
            "Expected And term, got {:?}",
            t.kind
        );
    }

    #[test]
    fn test_constructor_eq_nullary() {
        let (mut manager, _ctx, _rewriter) = setup();

        let dt_sort = manager.sorts.mk_datatype_sort("Unit");

        let con1 = manager.mk_dt_constructor("Unit", vec![], dt_sort);
        let con2 = manager.mk_dt_constructor("Unit", vec![], dt_sort);

        // Due to term interning, identical nullary constructors are the same term,
        // so mk_eq returns true directly (no rewriting needed)
        let eq = manager.mk_eq(con1, con2);

        let t = manager.get(eq).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_constructor_eq_diff() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let dt_sort = manager.sorts.mk_datatype_sort("Color");

        let con1 = manager.mk_dt_constructor("Red", vec![], dt_sort);
        let con2 = manager.mk_dt_constructor("Blue", vec![], dt_sort);
        let eq = manager.mk_eq(con1, con2);

        let result = rewriter.rewrite(eq, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::False));
    }
}
