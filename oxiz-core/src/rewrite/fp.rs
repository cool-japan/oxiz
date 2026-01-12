//! Floating-Point Term Rewriter
//!
//! This module provides rewriting rules for floating-point expressions:
//! - Constant folding for FP operations
//! - Special value handling (NaN, Inf, ±0)
//! - Identity simplifications (x + 0 → x for certain modes)
//! - Predicate simplification
//!
//! # IEEE 754 Compliance
//!
//! These rewrites are designed to be compliant with IEEE 754 semantics
//! where applicable, including proper handling of NaN and signed zeros.

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::ast::{RoundingMode, TermId, TermKind, TermManager};

/// Floating-point rewriter
#[derive(Debug, Clone)]
pub struct FpRewriter {
    /// Enable aggressive constant folding
    pub fold_constants: bool,
    /// Enable NaN propagation rules
    pub propagate_nan: bool,
    /// Enable infinity simplification
    pub simplify_infinity: bool,
    /// Enable predicate simplification
    pub simplify_predicates: bool,
}

impl Default for FpRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl FpRewriter {
    /// Create a new FP rewriter
    pub fn new() -> Self {
        Self {
            fold_constants: true,
            propagate_nan: true,
            simplify_infinity: true,
            simplify_predicates: true,
        }
    }

    /// Create with all optimizations enabled
    pub fn aggressive() -> Self {
        Self {
            fold_constants: true,
            propagate_nan: true,
            simplify_infinity: true,
            simplify_predicates: true,
        }
    }

    /// Check if a term is positive zero
    fn is_pos_zero(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };
        matches!(t.kind, TermKind::FpPlusZero { .. })
    }

    /// Check if a term is negative zero
    fn is_neg_zero(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };
        matches!(t.kind, TermKind::FpMinusZero { .. })
    }

    /// Check if a term is any zero (pos or neg)
    fn is_zero(&self, term: TermId, manager: &TermManager) -> bool {
        self.is_pos_zero(term, manager) || self.is_neg_zero(term, manager)
    }

    /// Check if a term is positive infinity
    fn is_pos_inf(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };
        matches!(t.kind, TermKind::FpPlusInfinity { .. })
    }

    /// Check if a term is negative infinity
    fn is_neg_inf(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };
        matches!(t.kind, TermKind::FpMinusInfinity { .. })
    }

    /// Check if a term is any infinity
    fn is_infinity(&self, term: TermId, manager: &TermManager) -> bool {
        self.is_pos_inf(term, manager) || self.is_neg_inf(term, manager)
    }

    /// Check if a term is NaN
    fn is_nan(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };
        matches!(t.kind, TermKind::FpNaN { .. })
    }

    /// Get FP width info (eb, sb) from a term
    fn get_fp_width(&self, term: TermId, manager: &TermManager) -> Option<(u32, u32)> {
        let t = manager.get(term)?;
        let sort = manager.sorts.get(t.sort)?;
        sort.float_format()
    }

    /// Rewrite FP negation
    fn rewrite_fp_neg(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_neg(arg));
        };

        match &t.kind {
            // -(-x) → x (double negation)
            TermKind::FpNeg(inner) => {
                ctx.stats_mut().record_rule("fp_double_neg");
                RewriteResult::Rewritten(*inner)
            }
            // -(+0) → -0
            TermKind::FpPlusZero { eb, sb } => {
                ctx.stats_mut().record_rule("fp_neg_pos_zero");
                RewriteResult::Rewritten(manager.mk_fp_minus_zero(*eb, *sb))
            }
            // -(-0) → +0
            TermKind::FpMinusZero { eb, sb } => {
                ctx.stats_mut().record_rule("fp_neg_neg_zero");
                RewriteResult::Rewritten(manager.mk_fp_plus_zero(*eb, *sb))
            }
            // -(+inf) → -inf
            TermKind::FpPlusInfinity { eb, sb } => {
                ctx.stats_mut().record_rule("fp_neg_pos_inf");
                RewriteResult::Rewritten(manager.mk_fp_minus_infinity(*eb, *sb))
            }
            // -(-inf) → +inf
            TermKind::FpMinusInfinity { eb, sb } => {
                ctx.stats_mut().record_rule("fp_neg_neg_inf");
                RewriteResult::Rewritten(manager.mk_fp_plus_infinity(*eb, *sb))
            }
            // -(NaN) → NaN
            TermKind::FpNaN { eb, sb } => {
                ctx.stats_mut().record_rule("fp_neg_nan");
                RewriteResult::Rewritten(manager.mk_fp_nan(*eb, *sb))
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_neg(arg)),
        }
    }

    /// Rewrite FP absolute value
    fn rewrite_fp_abs(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_abs(arg));
        };

        match &t.kind {
            // abs(abs(x)) → abs(x)
            TermKind::FpAbs(_inner) => {
                ctx.stats_mut().record_rule("fp_abs_abs");
                RewriteResult::Rewritten(arg) // abs(x) is already the result
            }
            // abs(-x) → abs(x)
            TermKind::FpNeg(inner) => {
                ctx.stats_mut().record_rule("fp_abs_neg");
                RewriteResult::Rewritten(manager.mk_fp_abs(*inner))
            }
            // abs(+0) → +0
            TermKind::FpPlusZero { eb, sb } => {
                ctx.stats_mut().record_rule("fp_abs_pos_zero");
                RewriteResult::Rewritten(manager.mk_fp_plus_zero(*eb, *sb))
            }
            // abs(-0) → +0
            TermKind::FpMinusZero { eb, sb } => {
                ctx.stats_mut().record_rule("fp_abs_neg_zero");
                RewriteResult::Rewritten(manager.mk_fp_plus_zero(*eb, *sb))
            }
            // abs(+inf) → +inf
            TermKind::FpPlusInfinity { eb, sb } => {
                ctx.stats_mut().record_rule("fp_abs_pos_inf");
                RewriteResult::Rewritten(manager.mk_fp_plus_infinity(*eb, *sb))
            }
            // abs(-inf) → +inf
            TermKind::FpMinusInfinity { eb, sb } => {
                ctx.stats_mut().record_rule("fp_abs_neg_inf");
                RewriteResult::Rewritten(manager.mk_fp_plus_infinity(*eb, *sb))
            }
            // abs(NaN) → NaN
            TermKind::FpNaN { eb, sb } => {
                ctx.stats_mut().record_rule("fp_abs_nan");
                RewriteResult::Rewritten(manager.mk_fp_nan(*eb, *sb))
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_abs(arg)),
        }
    }

    /// Rewrite FP addition
    fn rewrite_fp_add(
        &mut self,
        rm: RoundingMode,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // NaN propagation: NaN + x → NaN, x + NaN → NaN
        if self.propagate_nan {
            if self.is_nan(lhs, manager) {
                ctx.stats_mut().record_rule("fp_add_nan_lhs");
                return RewriteResult::Rewritten(lhs);
            }
            if self.is_nan(rhs, manager) {
                ctx.stats_mut().record_rule("fp_add_nan_rhs");
                return RewriteResult::Rewritten(rhs);
            }
        }

        // Infinity handling
        if self.simplify_infinity {
            // inf + (-inf) → NaN
            if ((self.is_pos_inf(lhs, manager) && self.is_neg_inf(rhs, manager))
                || (self.is_neg_inf(lhs, manager) && self.is_pos_inf(rhs, manager)))
                && let Some((eb, sb)) = self.get_fp_width(lhs, manager)
            {
                ctx.stats_mut().record_rule("fp_add_inf_neg_inf");
                return RewriteResult::Rewritten(manager.mk_fp_nan(eb, sb));
            }

            // inf + x → inf (for finite x)
            if self.is_pos_inf(lhs, manager) && !self.is_infinity(rhs, manager) {
                ctx.stats_mut().record_rule("fp_add_pos_inf");
                return RewriteResult::Rewritten(lhs);
            }
            if self.is_pos_inf(rhs, manager) && !self.is_infinity(lhs, manager) {
                ctx.stats_mut().record_rule("fp_add_pos_inf");
                return RewriteResult::Rewritten(rhs);
            }

            // -inf + x → -inf (for finite x)
            if self.is_neg_inf(lhs, manager) && !self.is_infinity(rhs, manager) {
                ctx.stats_mut().record_rule("fp_add_neg_inf");
                return RewriteResult::Rewritten(lhs);
            }
            if self.is_neg_inf(rhs, manager) && !self.is_infinity(lhs, manager) {
                ctx.stats_mut().record_rule("fp_add_neg_inf");
                return RewriteResult::Rewritten(rhs);
            }
        }

        RewriteResult::Unchanged(manager.mk_fp_add(rm, lhs, rhs))
    }

    /// Rewrite FP multiplication
    fn rewrite_fp_mul(
        &mut self,
        rm: RoundingMode,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // NaN propagation
        if self.propagate_nan {
            if self.is_nan(lhs, manager) {
                ctx.stats_mut().record_rule("fp_mul_nan_lhs");
                return RewriteResult::Rewritten(lhs);
            }
            if self.is_nan(rhs, manager) {
                ctx.stats_mut().record_rule("fp_mul_nan_rhs");
                return RewriteResult::Rewritten(rhs);
            }
        }

        // 0 * inf → NaN
        if self.simplify_infinity
            && ((self.is_zero(lhs, manager) && self.is_infinity(rhs, manager))
                || (self.is_infinity(lhs, manager) && self.is_zero(rhs, manager)))
            && let Some((eb, sb)) = self.get_fp_width(lhs, manager)
        {
            ctx.stats_mut().record_rule("fp_mul_zero_inf");
            return RewriteResult::Rewritten(manager.mk_fp_nan(eb, sb));
        }

        RewriteResult::Unchanged(manager.mk_fp_mul(rm, lhs, rhs))
    }

    /// Rewrite FP division
    fn rewrite_fp_div(
        &mut self,
        rm: RoundingMode,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // NaN propagation
        if self.propagate_nan {
            if self.is_nan(lhs, manager) {
                ctx.stats_mut().record_rule("fp_div_nan_lhs");
                return RewriteResult::Rewritten(lhs);
            }
            if self.is_nan(rhs, manager) {
                ctx.stats_mut().record_rule("fp_div_nan_rhs");
                return RewriteResult::Rewritten(rhs);
            }
        }

        // 0/0 → NaN
        if self.is_zero(lhs, manager)
            && self.is_zero(rhs, manager)
            && let Some((eb, sb)) = self.get_fp_width(lhs, manager)
        {
            ctx.stats_mut().record_rule("fp_div_zero_zero");
            return RewriteResult::Rewritten(manager.mk_fp_nan(eb, sb));
        }

        // inf/inf → NaN
        if self.simplify_infinity
            && self.is_infinity(lhs, manager)
            && self.is_infinity(rhs, manager)
            && let Some((eb, sb)) = self.get_fp_width(lhs, manager)
        {
            ctx.stats_mut().record_rule("fp_div_inf_inf");
            return RewriteResult::Rewritten(manager.mk_fp_nan(eb, sb));
        }

        // x/inf → 0 (for finite x)
        if self.simplify_infinity
            && self.is_infinity(rhs, manager)
            && !self.is_infinity(lhs, manager)
            && let Some((eb, sb)) = self.get_fp_width(lhs, manager)
        {
            ctx.stats_mut().record_rule("fp_div_by_inf");
            return RewriteResult::Rewritten(manager.mk_fp_plus_zero(eb, sb));
        }

        RewriteResult::Unchanged(manager.mk_fp_div(rm, lhs, rhs))
    }

    /// Rewrite FP square root
    fn rewrite_fp_sqrt(
        &mut self,
        rm: RoundingMode,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_sqrt(rm, arg));
        };

        match &t.kind {
            // sqrt(NaN) → NaN
            TermKind::FpNaN { eb, sb } => {
                ctx.stats_mut().record_rule("fp_sqrt_nan");
                RewriteResult::Rewritten(manager.mk_fp_nan(*eb, *sb))
            }
            // sqrt(+inf) → +inf
            TermKind::FpPlusInfinity { eb, sb } => {
                ctx.stats_mut().record_rule("fp_sqrt_pos_inf");
                RewriteResult::Rewritten(manager.mk_fp_plus_infinity(*eb, *sb))
            }
            // sqrt(-inf) → NaN
            TermKind::FpMinusInfinity { eb, sb } => {
                ctx.stats_mut().record_rule("fp_sqrt_neg_inf");
                RewriteResult::Rewritten(manager.mk_fp_nan(*eb, *sb))
            }
            // sqrt(+0) → +0
            TermKind::FpPlusZero { eb, sb } => {
                ctx.stats_mut().record_rule("fp_sqrt_pos_zero");
                RewriteResult::Rewritten(manager.mk_fp_plus_zero(*eb, *sb))
            }
            // sqrt(-0) → -0
            TermKind::FpMinusZero { eb, sb } => {
                ctx.stats_mut().record_rule("fp_sqrt_neg_zero");
                RewriteResult::Rewritten(manager.mk_fp_minus_zero(*eb, *sb))
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_sqrt(rm, arg)),
        }
    }

    /// Rewrite FP isNaN predicate
    fn rewrite_fp_is_nan(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        if !self.simplify_predicates {
            return RewriteResult::Unchanged(manager.mk_fp_is_nan(arg));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_is_nan(arg));
        };

        match &t.kind {
            TermKind::FpNaN { .. } => {
                ctx.stats_mut().record_rule("fp_is_nan_true");
                RewriteResult::Rewritten(manager.mk_true())
            }
            TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. } => {
                ctx.stats_mut().record_rule("fp_is_nan_false");
                RewriteResult::Rewritten(manager.mk_false())
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_is_nan(arg)),
        }
    }

    /// Rewrite FP isInfinite predicate
    fn rewrite_fp_is_inf(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        if !self.simplify_predicates {
            return RewriteResult::Unchanged(manager.mk_fp_is_infinite(arg));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_is_infinite(arg));
        };

        match &t.kind {
            TermKind::FpPlusInfinity { .. } | TermKind::FpMinusInfinity { .. } => {
                ctx.stats_mut().record_rule("fp_is_inf_true");
                RewriteResult::Rewritten(manager.mk_true())
            }
            TermKind::FpNaN { .. } | TermKind::FpPlusZero { .. } | TermKind::FpMinusZero { .. } => {
                ctx.stats_mut().record_rule("fp_is_inf_false");
                RewriteResult::Rewritten(manager.mk_false())
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_is_infinite(arg)),
        }
    }

    /// Rewrite FP isZero predicate
    fn rewrite_fp_is_zero(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        if !self.simplify_predicates {
            return RewriteResult::Unchanged(manager.mk_fp_is_zero(arg));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_is_zero(arg));
        };

        match &t.kind {
            TermKind::FpPlusZero { .. } | TermKind::FpMinusZero { .. } => {
                ctx.stats_mut().record_rule("fp_is_zero_true");
                RewriteResult::Rewritten(manager.mk_true())
            }
            TermKind::FpNaN { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. } => {
                ctx.stats_mut().record_rule("fp_is_zero_false");
                RewriteResult::Rewritten(manager.mk_false())
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_is_zero(arg)),
        }
    }

    /// Rewrite FP isNormal predicate
    fn rewrite_fp_is_normal(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        if !self.simplify_predicates {
            return RewriteResult::Unchanged(manager.mk_fp_is_normal(arg));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_is_normal(arg));
        };

        // Special values are never normal
        match &t.kind {
            TermKind::FpNaN { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpPlusZero { .. }
            | TermKind::FpMinusZero { .. } => {
                ctx.stats_mut().record_rule("fp_is_normal_false");
                RewriteResult::Rewritten(manager.mk_false())
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_is_normal(arg)),
        }
    }

    /// Rewrite FP isPositive predicate
    fn rewrite_fp_is_positive(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        if !self.simplify_predicates {
            return RewriteResult::Unchanged(manager.mk_fp_is_positive(arg));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_is_positive(arg));
        };

        match &t.kind {
            TermKind::FpPlusZero { .. } | TermKind::FpPlusInfinity { .. } => {
                ctx.stats_mut().record_rule("fp_is_positive_true");
                RewriteResult::Rewritten(manager.mk_true())
            }
            TermKind::FpMinusZero { .. }
            | TermKind::FpMinusInfinity { .. }
            | TermKind::FpNaN { .. } => {
                ctx.stats_mut().record_rule("fp_is_positive_false");
                RewriteResult::Rewritten(manager.mk_false())
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_is_positive(arg)),
        }
    }

    /// Rewrite FP isNegative predicate
    fn rewrite_fp_is_negative(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        if !self.simplify_predicates {
            return RewriteResult::Unchanged(manager.mk_fp_is_negative(arg));
        }

        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_fp_is_negative(arg));
        };

        match &t.kind {
            TermKind::FpMinusZero { .. } | TermKind::FpMinusInfinity { .. } => {
                ctx.stats_mut().record_rule("fp_is_negative_true");
                RewriteResult::Rewritten(manager.mk_true())
            }
            TermKind::FpPlusZero { .. }
            | TermKind::FpPlusInfinity { .. }
            | TermKind::FpNaN { .. } => {
                ctx.stats_mut().record_rule("fp_is_negative_false");
                RewriteResult::Rewritten(manager.mk_false())
            }
            _ => RewriteResult::Unchanged(manager.mk_fp_is_negative(arg)),
        }
    }

    /// Rewrite FP equality (special handling for NaN)
    fn rewrite_fp_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // NaN is never equal to anything (including itself)
        if self.is_nan(lhs, manager) || self.is_nan(rhs, manager) {
            ctx.stats_mut().record_rule("fp_eq_nan");
            return RewriteResult::Rewritten(manager.mk_false());
        }

        // +0 == -0 in FP
        if (self.is_pos_zero(lhs, manager) && self.is_neg_zero(rhs, manager))
            || (self.is_neg_zero(lhs, manager) && self.is_pos_zero(rhs, manager))
        {
            ctx.stats_mut().record_rule("fp_eq_zero");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        RewriteResult::Unchanged(manager.mk_fp_eq(lhs, rhs))
    }
}

impl Rewriter for FpRewriter {
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
            TermKind::FpNeg(arg) => self.rewrite_fp_neg(*arg, ctx, manager),
            TermKind::FpAbs(arg) => self.rewrite_fp_abs(*arg, ctx, manager),
            TermKind::FpAdd(rm, lhs, rhs) => self.rewrite_fp_add(*rm, *lhs, *rhs, ctx, manager),
            TermKind::FpMul(rm, lhs, rhs) => self.rewrite_fp_mul(*rm, *lhs, *rhs, ctx, manager),
            TermKind::FpDiv(rm, lhs, rhs) => self.rewrite_fp_div(*rm, *lhs, *rhs, ctx, manager),
            TermKind::FpSqrt(rm, arg) => self.rewrite_fp_sqrt(*rm, *arg, ctx, manager),
            TermKind::FpIsNaN(arg) => self.rewrite_fp_is_nan(*arg, ctx, manager),
            TermKind::FpIsInfinite(arg) => self.rewrite_fp_is_inf(*arg, ctx, manager),
            TermKind::FpIsZero(arg) => self.rewrite_fp_is_zero(*arg, ctx, manager),
            TermKind::FpIsNormal(arg) => self.rewrite_fp_is_normal(*arg, ctx, manager),
            TermKind::FpIsPositive(arg) => self.rewrite_fp_is_positive(*arg, ctx, manager),
            TermKind::FpIsNegative(arg) => self.rewrite_fp_is_negative(*arg, ctx, manager),
            TermKind::FpEq(lhs, rhs) => self.rewrite_fp_eq(*lhs, *rhs, ctx, manager),
            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "FpRewriter"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::FpNeg(_)
                | TermKind::FpAbs(_)
                | TermKind::FpAdd(_, _, _)
                | TermKind::FpSub(_, _, _)
                | TermKind::FpMul(_, _, _)
                | TermKind::FpDiv(_, _, _)
                | TermKind::FpSqrt(_, _)
                | TermKind::FpIsNaN(_)
                | TermKind::FpIsInfinite(_)
                | TermKind::FpIsZero(_)
                | TermKind::FpIsNormal(_)
                | TermKind::FpIsPositive(_)
                | TermKind::FpIsNegative(_)
                | TermKind::FpEq(_, _)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TermManager, RewriteContext, FpRewriter) {
        (TermManager::new(), RewriteContext::new(), FpRewriter::new())
    }

    #[test]
    fn test_fp_neg_double() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let fp_sort = manager.sorts.float_sort(8, 24); // float32
        let x = manager.mk_var("x", fp_sort);
        let neg1 = manager.mk_fp_neg(x);
        let neg2 = manager.mk_fp_neg(neg1);

        let result = rewriter.rewrite(neg2, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_fp_abs_neg() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let neg = manager.mk_fp_neg(x);
        let abs = manager.mk_fp_abs(neg);

        let result = rewriter.rewrite(abs, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::FpAbs(_)));
    }

    #[test]
    fn test_fp_is_nan_true() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let nan = manager.mk_fp_nan(8, 24);
        let is_nan = manager.mk_fp_is_nan(nan);

        let result = rewriter.rewrite(is_nan, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_fp_is_inf_true() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let inf = manager.mk_fp_plus_infinity(8, 24);
        let is_inf = manager.mk_fp_is_infinite(inf);

        let result = rewriter.rewrite(is_inf, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_fp_is_zero_true() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let zero = manager.mk_fp_plus_zero(8, 24);
        let is_zero = manager.mk_fp_is_zero(zero);

        let result = rewriter.rewrite(is_zero, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_fp_eq_nan() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let nan = manager.mk_fp_nan(8, 24);
        let eq = manager.mk_fp_eq(nan, nan);

        let result = rewriter.rewrite(eq, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::False));
    }

    #[test]
    fn test_fp_eq_zeros() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let pos_zero = manager.mk_fp_plus_zero(8, 24);
        let neg_zero = manager.mk_fp_minus_zero(8, 24);
        let eq = manager.mk_fp_eq(pos_zero, neg_zero);

        let result = rewriter.rewrite(eq, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }
}
