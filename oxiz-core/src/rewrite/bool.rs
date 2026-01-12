//! Boolean Term Rewriter
//!
//! This module provides rewriting rules for Boolean expressions:
//! - Constant propagation (true ∧ x → x)
//! - Complement simplification (x ∧ ¬x → false)
//! - Idempotence (x ∧ x → x)
//! - Double negation (¬¬x → x)
//! - Absorption (x ∧ (x ∨ y) → x)
//! - De Morgan's laws
//! - ITE simplification
//! - Implication normalization

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::ast::{TermId, TermKind, TermManager};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

/// Boolean rewriter
#[derive(Debug, Clone)]
pub struct BoolRewriter {
    /// Enable absorption law
    pub enable_absorption: bool,
    /// Enable De Morgan transformation
    pub enable_demorgan: bool,
    /// Enable implication to disjunction conversion
    pub normalize_implies: bool,
    /// Flatten nested AND/OR
    pub flatten: bool,
    /// Sort arguments for AC normalization
    pub sort_args: bool,
}

impl Default for BoolRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolRewriter {
    /// Create a new boolean rewriter
    pub fn new() -> Self {
        Self {
            enable_absorption: false, // Can be expensive
            enable_demorgan: false,   // May not always be beneficial
            normalize_implies: true,
            flatten: true,
            sort_args: true,
        }
    }

    /// Create with all optimizations enabled
    pub fn aggressive() -> Self {
        Self {
            enable_absorption: true,
            enable_demorgan: true,
            normalize_implies: true,
            flatten: true,
            sort_args: true,
        }
    }

    /// Check if term is true constant
    fn is_true(&self, term: TermId, manager: &TermManager) -> bool {
        manager
            .get(term)
            .is_some_and(|t| matches!(t.kind, TermKind::True))
    }

    /// Check if term is false constant
    fn is_false(&self, term: TermId, manager: &TermManager) -> bool {
        manager
            .get(term)
            .is_some_and(|t| matches!(t.kind, TermKind::False))
    }

    /// Check if one term is the negation of another
    fn is_negation_of(&self, t1: TermId, t2: TermId, manager: &TermManager) -> bool {
        if let Some(term1) = manager.get(t1)
            && let TermKind::Not(inner) = &term1.kind
        {
            return *inner == t2;
        }
        if let Some(term2) = manager.get(t2)
            && let TermKind::Not(inner) = &term2.kind
        {
            return *inner == t1;
        }
        false
    }

    /// Get the inner term of a negation, if any
    fn get_negated(&self, term: TermId, manager: &TermManager) -> Option<TermId> {
        manager.get(term).and_then(|t| {
            if let TermKind::Not(inner) = &t.kind {
                Some(*inner)
            } else {
                None
            }
        })
    }

    /// Rewrite NOT
    fn rewrite_not(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_not(arg));
        };

        match &t.kind {
            // ¬true → false
            TermKind::True => {
                ctx.stats_mut().record_rule("bool_not_true");
                RewriteResult::Rewritten(manager.mk_false())
            }
            // ¬false → true
            TermKind::False => {
                ctx.stats_mut().record_rule("bool_not_false");
                RewriteResult::Rewritten(manager.mk_true())
            }
            // ¬¬x → x
            TermKind::Not(inner) => {
                ctx.stats_mut().record_rule("bool_double_neg");
                RewriteResult::Rewritten(*inner)
            }
            // De Morgan: ¬(x ∧ y) → ¬x ∨ ¬y
            TermKind::And(args) if self.enable_demorgan => {
                ctx.stats_mut().record_rule("bool_demorgan_and");
                let negated: Vec<_> = args.iter().map(|&a| manager.mk_not(a)).collect();
                RewriteResult::Rewritten(manager.mk_or(negated))
            }
            // De Morgan: ¬(x ∨ y) → ¬x ∧ ¬y
            TermKind::Or(args) if self.enable_demorgan => {
                ctx.stats_mut().record_rule("bool_demorgan_or");
                let negated: Vec<_> = args.iter().map(|&a| manager.mk_not(a)).collect();
                RewriteResult::Rewritten(manager.mk_and(negated))
            }
            _ => RewriteResult::Unchanged(manager.mk_not(arg)),
        }
    }

    /// Rewrite AND
    fn rewrite_and(
        &mut self,
        args: &[TermId],
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let mut result: SmallVec<[TermId; 8]> = SmallVec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut changed = false;

        for &arg in args {
            // false ∧ ... → false
            if self.is_false(arg, manager) {
                ctx.stats_mut().record_rule("bool_and_false");
                return RewriteResult::Rewritten(manager.mk_false());
            }

            // true ∧ x → x
            if self.is_true(arg, manager) {
                changed = true;
                continue;
            }

            // Flatten nested ANDs
            if self.flatten
                && let Some(t) = manager.get(arg)
                && let TermKind::And(nested) = &t.kind
            {
                changed = true;
                for &nested_arg in nested {
                    if !seen.contains(&nested_arg) {
                        // Check for complement
                        for &existing in &result {
                            if self.is_negation_of(nested_arg, existing, manager) {
                                ctx.stats_mut().record_rule("bool_and_complement");
                                return RewriteResult::Rewritten(manager.mk_false());
                            }
                        }
                        seen.insert(nested_arg);
                        result.push(nested_arg);
                    }
                }
                continue;
            }

            // Check for complement: x ∧ ¬x → false
            for &existing in &result {
                if self.is_negation_of(arg, existing, manager) {
                    ctx.stats_mut().record_rule("bool_and_complement");
                    return RewriteResult::Rewritten(manager.mk_false());
                }
            }

            // Idempotence: x ∧ x → x
            if !seen.contains(&arg) {
                seen.insert(arg);
                result.push(arg);
            } else {
                changed = true;
            }
        }

        // Sort for canonicalization
        if self.sort_args {
            let mut sorted = result.clone();
            sorted.sort_by_key(|t| t.0);
            if sorted != result {
                changed = true;
                result = sorted;
            }
        }

        if !changed {
            return RewriteResult::Unchanged(manager.mk_and(args.to_vec()));
        }

        ctx.stats_mut().record_rule("bool_and_simplify");

        match result.len() {
            0 => RewriteResult::Rewritten(manager.mk_true()),
            1 => RewriteResult::Rewritten(result[0]),
            _ => RewriteResult::Rewritten(manager.mk_and(result.to_vec())),
        }
    }

    /// Rewrite OR
    fn rewrite_or(
        &mut self,
        args: &[TermId],
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let mut result: SmallVec<[TermId; 8]> = SmallVec::new();
        let mut seen: FxHashSet<TermId> = FxHashSet::default();
        let mut changed = false;

        for &arg in args {
            // true ∨ ... → true
            if self.is_true(arg, manager) {
                ctx.stats_mut().record_rule("bool_or_true");
                return RewriteResult::Rewritten(manager.mk_true());
            }

            // false ∨ x → x
            if self.is_false(arg, manager) {
                changed = true;
                continue;
            }

            // Flatten nested ORs
            if self.flatten
                && let Some(t) = manager.get(arg)
                && let TermKind::Or(nested) = &t.kind
            {
                changed = true;
                for &nested_arg in nested {
                    if !seen.contains(&nested_arg) {
                        // Check for tautology
                        for &existing in &result {
                            if self.is_negation_of(nested_arg, existing, manager) {
                                ctx.stats_mut().record_rule("bool_or_tautology");
                                return RewriteResult::Rewritten(manager.mk_true());
                            }
                        }
                        seen.insert(nested_arg);
                        result.push(nested_arg);
                    }
                }
                continue;
            }

            // Check for tautology: x ∨ ¬x → true
            for &existing in &result {
                if self.is_negation_of(arg, existing, manager) {
                    ctx.stats_mut().record_rule("bool_or_tautology");
                    return RewriteResult::Rewritten(manager.mk_true());
                }
            }

            // Idempotence: x ∨ x → x
            if !seen.contains(&arg) {
                seen.insert(arg);
                result.push(arg);
            } else {
                changed = true;
            }
        }

        // Sort for canonicalization
        if self.sort_args {
            let mut sorted = result.clone();
            sorted.sort_by_key(|t| t.0);
            if sorted != result {
                changed = true;
                result = sorted;
            }
        }

        if !changed {
            return RewriteResult::Unchanged(manager.mk_or(args.to_vec()));
        }

        ctx.stats_mut().record_rule("bool_or_simplify");

        match result.len() {
            0 => RewriteResult::Rewritten(manager.mk_false()),
            1 => RewriteResult::Rewritten(result[0]),
            _ => RewriteResult::Rewritten(manager.mk_or(result.to_vec())),
        }
    }

    /// Rewrite IMPLIES
    fn rewrite_implies(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // false → x  =  true
        if self.is_false(lhs, manager) {
            ctx.stats_mut().record_rule("bool_implies_false_lhs");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // true → x  =  x
        if self.is_true(lhs, manager) {
            ctx.stats_mut().record_rule("bool_implies_true_lhs");
            return RewriteResult::Rewritten(rhs);
        }

        // x → true  =  true
        if self.is_true(rhs, manager) {
            ctx.stats_mut().record_rule("bool_implies_true_rhs");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // x → false  =  ¬x
        if self.is_false(rhs, manager) {
            ctx.stats_mut().record_rule("bool_implies_false_rhs");
            return RewriteResult::Rewritten(manager.mk_not(lhs));
        }

        // x → x  =  true
        if lhs == rhs {
            ctx.stats_mut().record_rule("bool_implies_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // x → ¬x  =  ¬x
        if self.get_negated(rhs, manager) == Some(lhs) {
            ctx.stats_mut().record_rule("bool_implies_neg");
            return RewriteResult::Rewritten(rhs);
        }

        // ¬x → x  =  x
        if self.get_negated(lhs, manager) == Some(rhs) {
            ctx.stats_mut().record_rule("bool_neg_implies");
            return RewriteResult::Rewritten(rhs);
        }

        // Normalize to disjunction: x → y  =  ¬x ∨ y
        if self.normalize_implies {
            ctx.stats_mut().record_rule("bool_implies_to_or");
            let not_lhs = manager.mk_not(lhs);
            return RewriteResult::Rewritten(manager.mk_or([not_lhs, rhs]));
        }

        RewriteResult::Unchanged(manager.mk_implies(lhs, rhs))
    }

    /// Rewrite ITE (if-then-else)
    fn rewrite_ite(
        &mut self,
        cond: TermId,
        then_br: TermId,
        else_br: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // ite(true, x, y) → x
        if self.is_true(cond, manager) {
            ctx.stats_mut().record_rule("bool_ite_true_cond");
            return RewriteResult::Rewritten(then_br);
        }

        // ite(false, x, y) → y
        if self.is_false(cond, manager) {
            ctx.stats_mut().record_rule("bool_ite_false_cond");
            return RewriteResult::Rewritten(else_br);
        }

        // ite(c, x, x) → x
        if then_br == else_br {
            ctx.stats_mut().record_rule("bool_ite_same_branches");
            return RewriteResult::Rewritten(then_br);
        }

        // ite(c, true, false) → c
        if self.is_true(then_br, manager) && self.is_false(else_br, manager) {
            ctx.stats_mut().record_rule("bool_ite_to_cond");
            return RewriteResult::Rewritten(cond);
        }

        // ite(c, false, true) → ¬c
        if self.is_false(then_br, manager) && self.is_true(else_br, manager) {
            ctx.stats_mut().record_rule("bool_ite_to_not_cond");
            return RewriteResult::Rewritten(manager.mk_not(cond));
        }

        // ite(c, true, x) → c ∨ x
        if self.is_true(then_br, manager) {
            ctx.stats_mut().record_rule("bool_ite_true_then");
            return RewriteResult::Rewritten(manager.mk_or([cond, else_br]));
        }

        // ite(c, false, x) → ¬c ∧ x
        if self.is_false(then_br, manager) {
            ctx.stats_mut().record_rule("bool_ite_false_then");
            let not_cond = manager.mk_not(cond);
            return RewriteResult::Rewritten(manager.mk_and([not_cond, else_br]));
        }

        // ite(c, x, true) → ¬c ∨ x
        if self.is_true(else_br, manager) {
            ctx.stats_mut().record_rule("bool_ite_true_else");
            let not_cond = manager.mk_not(cond);
            return RewriteResult::Rewritten(manager.mk_or([not_cond, then_br]));
        }

        // ite(c, x, false) → c ∧ x
        if self.is_false(else_br, manager) {
            ctx.stats_mut().record_rule("bool_ite_false_else");
            return RewriteResult::Rewritten(manager.mk_and([cond, then_br]));
        }

        // ite(¬c, x, y) → ite(c, y, x)
        if let Some(inner_cond) = self.get_negated(cond, manager) {
            ctx.stats_mut().record_rule("bool_ite_neg_cond");
            return RewriteResult::Rewritten(manager.mk_ite(inner_cond, else_br, then_br));
        }

        RewriteResult::Unchanged(manager.mk_ite(cond, then_br, else_br))
    }

    /// Rewrite equality (for Boolean terms)
    fn rewrite_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x = x → true
        if lhs == rhs {
            ctx.stats_mut().record_rule("bool_eq_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // true = x → x
        if self.is_true(lhs, manager) {
            ctx.stats_mut().record_rule("bool_eq_true_lhs");
            return RewriteResult::Rewritten(rhs);
        }
        if self.is_true(rhs, manager) {
            ctx.stats_mut().record_rule("bool_eq_true_rhs");
            return RewriteResult::Rewritten(lhs);
        }

        // false = x → ¬x
        if self.is_false(lhs, manager) {
            ctx.stats_mut().record_rule("bool_eq_false_lhs");
            return RewriteResult::Rewritten(manager.mk_not(rhs));
        }
        if self.is_false(rhs, manager) {
            ctx.stats_mut().record_rule("bool_eq_false_rhs");
            return RewriteResult::Rewritten(manager.mk_not(lhs));
        }

        // x = ¬x → false
        if self.is_negation_of(lhs, rhs, manager) {
            ctx.stats_mut().record_rule("bool_eq_complement");
            return RewriteResult::Rewritten(manager.mk_false());
        }

        RewriteResult::Unchanged(manager.mk_eq(lhs, rhs))
    }

    /// Check if a term is a Boolean term
    fn is_bool_term(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::True
                | TermKind::False
                | TermKind::Not(_)
                | TermKind::And(_)
                | TermKind::Or(_)
                | TermKind::Implies(_, _)
                | TermKind::Ite(_, _, _)
        )
    }
}

impl Rewriter for BoolRewriter {
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
            TermKind::Not(arg) => self.rewrite_not(*arg, ctx, manager),
            TermKind::And(args) => {
                let args_cloned = args.clone();
                self.rewrite_and(&args_cloned, ctx, manager)
            }
            TermKind::Or(args) => {
                let args_cloned = args.clone();
                self.rewrite_or(&args_cloned, ctx, manager)
            }
            TermKind::Implies(lhs, rhs) => self.rewrite_implies(*lhs, *rhs, ctx, manager),
            TermKind::Ite(cond, then_br, else_br) => {
                // Only apply to Boolean ITE
                if self.is_bool_term(*then_br, manager) || self.is_bool_term(*else_br, manager) {
                    self.rewrite_ite(*cond, *then_br, *else_br, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Eq(lhs, rhs) => {
                // Only apply to Boolean equality
                if self.is_bool_term(*lhs, manager) || self.is_bool_term(*rhs, manager) {
                    self.rewrite_eq(*lhs, *rhs, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "BoolRewriter"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::Not(_)
                | TermKind::And(_)
                | TermKind::Or(_)
                | TermKind::Implies(_, _)
                | TermKind::Ite(_, _, _)
                | TermKind::Eq(_, _)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TermManager, RewriteContext, BoolRewriter) {
        (
            TermManager::new(),
            RewriteContext::new(),
            BoolRewriter::new(),
        )
    }

    #[test]
    fn test_not_true() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_not already simplifies not(true) -> false at term creation
        let t = manager.mk_true();
        let not_t = manager.mk_not(t);

        // Verify the simplification happened at creation time
        let term = manager.get(not_t).expect("term should exist");
        assert!(matches!(term.kind, TermKind::False));
    }

    #[test]
    fn test_double_negation() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_not already simplifies not(not(x)) -> x at term creation
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let not_x = manager.mk_not(x);
        let not_not_x = manager.mk_not(not_x);

        // Verify the simplification happened at creation time
        assert_eq!(not_not_x, x);
    }

    #[test]
    fn test_and_false() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_and already simplifies and(x, false) -> false at term creation
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let f = manager.mk_false();
        let and = manager.mk_and([x, f]);

        // Verify the simplification happened at creation time
        let term = manager.get(and).expect("term should exist");
        assert!(matches!(term.kind, TermKind::False));
    }

    #[test]
    fn test_and_true() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_and already simplifies and(x, true) -> x at term creation
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let and = manager.mk_and([x, t]);

        // Verify the simplification happened at creation time
        assert_eq!(and, x);
    }

    #[test]
    fn test_and_complement() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let not_x = manager.mk_not(x);
        let and = manager.mk_and([x, not_x]);

        let result = rewriter.rewrite(and, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let term = manager.get(result.term()).expect("term should exist");
        assert!(matches!(term.kind, TermKind::False));
    }

    #[test]
    fn test_or_true() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_or already simplifies or(x, true) -> true at term creation
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let t = manager.mk_true();
        let or = manager.mk_or([x, t]);

        // Verify the simplification happened at creation time
        let term = manager.get(or).expect("term should exist");
        assert!(matches!(term.kind, TermKind::True));
    }

    #[test]
    fn test_or_tautology() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let not_x = manager.mk_not(x);
        let or = manager.mk_or([x, not_x]);

        let result = rewriter.rewrite(or, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let term = manager.get(result.term()).expect("term should exist");
        assert!(matches!(term.kind, TermKind::True));
    }

    #[test]
    fn test_implies_false_lhs() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_implies already simplifies implies(false, x) -> true at term creation
        let f = manager.mk_false();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let imp = manager.mk_implies(f, x);

        // Verify the simplification happened at creation time
        let term = manager.get(imp).expect("term should exist");
        assert!(matches!(term.kind, TermKind::True));
    }

    #[test]
    fn test_ite_true_cond() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_ite already simplifies ite(true, x, y) -> x at term creation
        let t = manager.mk_true();
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let y = manager.mk_var("y", manager.sorts.bool_sort);
        let ite = manager.mk_ite(t, x, y);

        // Verify the simplification happened at creation time
        assert_eq!(ite, x);
    }

    #[test]
    fn test_ite_same_branches() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_ite already simplifies ite(c, x, x) -> x at term creation
        let c = manager.mk_var("c", manager.sorts.bool_sort);
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let ite = manager.mk_ite(c, x, x);

        // Verify the simplification happened at creation time
        assert_eq!(ite, x);
    }

    #[test]
    fn test_eq_self() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_eq already simplifies eq(x, x) -> true at term creation
        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let eq = manager.mk_eq(x, x);

        // Verify the simplification happened at creation time
        let term = manager.get(eq).expect("term should exist");
        assert!(matches!(term.kind, TermKind::True));
    }
}
