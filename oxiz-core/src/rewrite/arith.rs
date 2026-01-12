//! Arithmetic Term Rewriter
//!
//! This module provides rewriting rules for arithmetic expressions:
//! - Constant folding (2 + 3 → 5)
//! - Identity elimination (x + 0 → x, x * 1 → x)
//! - Zero simplification (x * 0 → 0)
//! - Negation simplification (--x → x)
//! - Distributive law application
//! - Polynomial normalization
//! - Common subexpression elimination
//!
//! # Supported Theories
//!
//! - Linear Integer Arithmetic (LIA)
//! - Linear Real Arithmetic (LRA)
//! - Non-linear Arithmetic (NIA/NRA)

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::ast::{TermId, TermKind, TermManager};
use num_bigint::BigInt;
use num_rational::Rational64;
use num_traits::{One, Zero};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Arithmetic expression rewriter
#[derive(Debug, Clone)]
pub struct ArithRewriter {
    /// Enable distributive law expansion
    pub expand_distributive: bool,
    /// Enable polynomial normalization
    pub normalize_polynomials: bool,
    /// Enable GCD-based simplification
    pub use_gcd: bool,
    /// Enable bound propagation
    pub propagate_bounds: bool,
    /// Cache for integer constant values
    int_cache: FxHashMap<TermId, i64>,
    /// Cache for rational constant values
    rat_cache: FxHashMap<TermId, Rational64>,
}

impl Default for ArithRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ArithRewriter {
    /// Create a new arithmetic rewriter
    pub fn new() -> Self {
        Self {
            expand_distributive: false,
            normalize_polynomials: true,
            use_gcd: true,
            propagate_bounds: false,
            int_cache: FxHashMap::default(),
            rat_cache: FxHashMap::default(),
        }
    }

    /// Create with all optimizations enabled
    pub fn aggressive() -> Self {
        Self {
            expand_distributive: true,
            normalize_polynomials: true,
            use_gcd: true,
            propagate_bounds: true,
            int_cache: FxHashMap::default(),
            rat_cache: FxHashMap::default(),
        }
    }

    /// Try to extract an integer constant value from a term
    fn get_int_constant(&mut self, term: TermId, manager: &TermManager) -> Option<i64> {
        // Check cache first
        if let Some(&val) = self.int_cache.get(&term) {
            return Some(val);
        }

        let t = manager.get(term)?;
        let result: Option<i64> = match &t.kind {
            TermKind::IntConst(n) => n.try_into().ok(),
            _ => None,
        };

        if let Some(val) = result {
            self.int_cache.insert(term, val);
        }

        result
    }

    /// Try to extract a rational constant value from a term
    fn get_rat_constant(&mut self, term: TermId, manager: &TermManager) -> Option<Rational64> {
        // Check cache first
        if let Some(&val) = self.rat_cache.get(&term) {
            return Some(val);
        }

        let t = manager.get(term)?;
        let result = match &t.kind {
            TermKind::IntConst(n) => {
                let val: i64 = n.try_into().ok()?;
                Some(Rational64::from_integer(val))
            }
            TermKind::RealConst(r) => Some(*r),
            _ => None,
        };

        if let Some(val) = result {
            self.rat_cache.insert(term, val);
        }

        result
    }

    /// Check if a term is zero
    fn is_zero(&mut self, term: TermId, manager: &TermManager) -> bool {
        self.get_rat_constant(term, manager)
            .is_some_and(|r| r.is_zero())
    }

    /// Check if a term is one
    fn is_one(&mut self, term: TermId, manager: &TermManager) -> bool {
        self.get_rat_constant(term, manager)
            .is_some_and(|r| r.is_one())
    }

    /// Check if a term is negative one
    fn is_neg_one(&mut self, term: TermId, manager: &TermManager) -> bool {
        self.get_rat_constant(term, manager)
            .is_some_and(|r| r == Rational64::from_integer(-1))
    }

    /// Rewrite addition
    fn rewrite_add(
        &mut self,
        args: &[TermId],
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let mut sum = Rational64::zero();
        let mut non_const: SmallVec<[TermId; 8]> = SmallVec::new();
        let mut changed = false;

        for &arg in args {
            if let Some(val) = self.get_rat_constant(arg, manager) {
                sum += val;
                changed = true;
            } else {
                non_const.push(arg);
            }
        }

        // Filter zeros
        let filtered: SmallVec<[TermId; 8]> = non_const
            .into_iter()
            .filter(|&arg| !self.is_zero(arg, manager))
            .collect();

        if filtered.len() < args.len() - if sum.is_zero() { 0 } else { 1 } {
            changed = true;
        }

        // Build result
        let mut result_args: Vec<TermId> = filtered.into_iter().collect();

        if !sum.is_zero() || result_args.is_empty() {
            if sum.is_integer() {
                result_args.push(manager.mk_int(*sum.numer()));
            } else {
                result_args.push(manager.mk_real(sum));
            }
        }

        if !changed {
            return RewriteResult::Unchanged(manager.mk_add(args.to_vec()));
        }

        ctx.stats_mut().record_rule("arith_add_fold");

        match result_args.len() {
            0 => RewriteResult::Rewritten(manager.mk_int(0)),
            1 => RewriteResult::Rewritten(result_args[0]),
            _ => RewriteResult::Rewritten(manager.mk_add(result_args)),
        }
    }

    /// Rewrite multiplication
    fn rewrite_mul(
        &mut self,
        args: &[TermId],
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let mut product = Rational64::one();
        let mut non_const: SmallVec<[TermId; 8]> = SmallVec::new();
        let mut changed = false;

        for &arg in args {
            // Check for zero - short circuit
            if self.is_zero(arg, manager) {
                ctx.stats_mut().record_rule("arith_mul_zero");
                return RewriteResult::Rewritten(manager.mk_int(0));
            }

            if let Some(val) = self.get_rat_constant(arg, manager) {
                product *= val;
                changed = true;
            } else {
                non_const.push(arg);
            }
        }

        // Filter ones
        let filtered: SmallVec<[TermId; 8]> = non_const
            .into_iter()
            .filter(|&arg| !self.is_one(arg, manager))
            .collect();

        if filtered.len() < args.len() - 1 {
            changed = true;
        }

        // Build result
        let mut result_args: Vec<TermId> = filtered.into_iter().collect();

        if product != Rational64::one() || result_args.is_empty() {
            if product.is_integer() {
                result_args.insert(0, manager.mk_int(*product.numer()));
            } else {
                result_args.insert(0, manager.mk_real(product));
            }
        }

        if !changed {
            return RewriteResult::Unchanged(manager.mk_mul(args.to_vec()));
        }

        ctx.stats_mut().record_rule("arith_mul_fold");

        match result_args.len() {
            0 => RewriteResult::Rewritten(manager.mk_int(1)),
            1 => RewriteResult::Rewritten(result_args[0]),
            _ => RewriteResult::Rewritten(manager.mk_mul(result_args)),
        }
    }

    /// Rewrite subtraction
    fn rewrite_sub(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x - 0 → x
        if self.is_zero(rhs, manager) {
            ctx.stats_mut().record_rule("arith_sub_zero");
            return RewriteResult::Rewritten(lhs);
        }

        // 0 - x → -x
        if self.is_zero(lhs, manager) {
            ctx.stats_mut().record_rule("arith_sub_from_zero");
            return RewriteResult::Rewritten(manager.mk_neg(rhs));
        }

        // x - x → 0
        if lhs == rhs {
            ctx.stats_mut().record_rule("arith_sub_self");
            return RewriteResult::Rewritten(manager.mk_int(0));
        }

        // Constant folding
        if let (Some(l), Some(r)) = (
            self.get_rat_constant(lhs, manager),
            self.get_rat_constant(rhs, manager),
        ) {
            let result = l - r;
            ctx.stats_mut().record_rule("arith_sub_const");
            if result.is_integer() {
                return RewriteResult::Rewritten(manager.mk_int(*result.numer()));
            } else {
                return RewriteResult::Rewritten(manager.mk_real(result));
            }
        }

        RewriteResult::Unchanged(manager.mk_sub(lhs, rhs))
    }

    /// Rewrite negation
    fn rewrite_neg(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let Some(t) = manager.get(arg).cloned() else {
            return RewriteResult::Unchanged(manager.mk_neg(arg));
        };

        match &t.kind {
            // -0 → 0
            TermKind::IntConst(n) if n == &BigInt::from(0) => {
                ctx.stats_mut().record_rule("arith_neg_zero");
                RewriteResult::Rewritten(manager.mk_int(0))
            }
            // -c → -c (constant folding)
            TermKind::IntConst(n) => {
                ctx.stats_mut().record_rule("arith_neg_const");
                RewriteResult::Rewritten(manager.mk_int(-n.clone()))
            }
            TermKind::RealConst(r) => {
                ctx.stats_mut().record_rule("arith_neg_const");
                RewriteResult::Rewritten(manager.mk_real(-r))
            }
            // --x → x
            TermKind::Neg(inner) => {
                ctx.stats_mut().record_rule("arith_double_neg");
                RewriteResult::Rewritten(*inner)
            }
            // -(x - y) → y - x
            TermKind::Sub(x, y) => {
                ctx.stats_mut().record_rule("arith_neg_sub");
                RewriteResult::Rewritten(manager.mk_sub(*y, *x))
            }
            _ => RewriteResult::Unchanged(manager.mk_neg(arg)),
        }
    }

    /// Rewrite division
    fn rewrite_div(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x / 1 → x
        if self.is_one(rhs, manager) {
            ctx.stats_mut().record_rule("arith_div_one");
            return RewriteResult::Rewritten(lhs);
        }

        // 0 / x → 0 (when x ≠ 0)
        if self.is_zero(lhs, manager) && !self.is_zero(rhs, manager) {
            ctx.stats_mut().record_rule("arith_div_zero_num");
            return RewriteResult::Rewritten(manager.mk_int(0));
        }

        // x / -1 → -x
        if self.is_neg_one(rhs, manager) {
            ctx.stats_mut().record_rule("arith_div_neg_one");
            return RewriteResult::Rewritten(manager.mk_neg(lhs));
        }

        // Constant folding for rationals
        if let (Some(l), Some(r)) = (
            self.get_rat_constant(lhs, manager),
            self.get_rat_constant(rhs, manager),
        ) && !r.is_zero()
        {
            let result = l / r;
            ctx.stats_mut().record_rule("arith_div_const");
            if result.is_integer() {
                return RewriteResult::Rewritten(manager.mk_int(*result.numer()));
            } else {
                return RewriteResult::Rewritten(manager.mk_real(result));
            }
        }

        RewriteResult::Unchanged(manager.mk_div(lhs, rhs))
    }

    /// Rewrite modulo
    fn rewrite_mod(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // 0 mod x → 0
        if self.is_zero(lhs, manager) {
            ctx.stats_mut().record_rule("arith_mod_zero_num");
            return RewriteResult::Rewritten(manager.mk_int(0));
        }

        // x mod 1 → 0
        if self.is_one(rhs, manager) {
            ctx.stats_mut().record_rule("arith_mod_one");
            return RewriteResult::Rewritten(manager.mk_int(0));
        }

        // Constant folding
        if let (Some(l), Some(r)) = (
            self.get_int_constant(lhs, manager),
            self.get_int_constant(rhs, manager),
        ) && r != 0
        {
            let result = l % r;
            ctx.stats_mut().record_rule("arith_mod_const");
            return RewriteResult::Rewritten(manager.mk_int(result));
        }

        RewriteResult::Unchanged(manager.mk_mod(lhs, rhs))
    }

    /// Rewrite comparison (less than)
    fn rewrite_lt(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x < x → false
        if lhs == rhs {
            ctx.stats_mut().record_rule("arith_lt_self");
            return RewriteResult::Rewritten(manager.mk_false());
        }

        // Constant comparison
        if let (Some(l), Some(r)) = (
            self.get_rat_constant(lhs, manager),
            self.get_rat_constant(rhs, manager),
        ) {
            ctx.stats_mut().record_rule("arith_lt_const");
            if l < r {
                return RewriteResult::Rewritten(manager.mk_true());
            } else {
                return RewriteResult::Rewritten(manager.mk_false());
            }
        }

        RewriteResult::Unchanged(manager.mk_lt(lhs, rhs))
    }

    /// Rewrite comparison (less or equal)
    fn rewrite_le(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x <= x → true
        if lhs == rhs {
            ctx.stats_mut().record_rule("arith_le_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // Constant comparison
        if let (Some(l), Some(r)) = (
            self.get_rat_constant(lhs, manager),
            self.get_rat_constant(rhs, manager),
        ) {
            ctx.stats_mut().record_rule("arith_le_const");
            if l <= r {
                return RewriteResult::Rewritten(manager.mk_true());
            } else {
                return RewriteResult::Rewritten(manager.mk_false());
            }
        }

        RewriteResult::Unchanged(manager.mk_le(lhs, rhs))
    }

    /// Rewrite comparison (greater than)
    fn rewrite_gt(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x > x → false
        if lhs == rhs {
            ctx.stats_mut().record_rule("arith_gt_self");
            return RewriteResult::Rewritten(manager.mk_false());
        }

        // Constant comparison
        if let (Some(l), Some(r)) = (
            self.get_rat_constant(lhs, manager),
            self.get_rat_constant(rhs, manager),
        ) {
            ctx.stats_mut().record_rule("arith_gt_const");
            if l > r {
                return RewriteResult::Rewritten(manager.mk_true());
            } else {
                return RewriteResult::Rewritten(manager.mk_false());
            }
        }

        RewriteResult::Unchanged(manager.mk_gt(lhs, rhs))
    }

    /// Rewrite comparison (greater or equal)
    fn rewrite_ge(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x >= x → true
        if lhs == rhs {
            ctx.stats_mut().record_rule("arith_ge_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // Constant comparison
        if let (Some(l), Some(r)) = (
            self.get_rat_constant(lhs, manager),
            self.get_rat_constant(rhs, manager),
        ) {
            ctx.stats_mut().record_rule("arith_ge_const");
            if l >= r {
                return RewriteResult::Rewritten(manager.mk_true());
            } else {
                return RewriteResult::Rewritten(manager.mk_false());
            }
        }

        RewriteResult::Unchanged(manager.mk_ge(lhs, rhs))
    }

    /// Rewrite arithmetic equality
    fn rewrite_arith_eq(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        // x = x → true
        if lhs == rhs {
            ctx.stats_mut().record_rule("arith_eq_self");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        // Constant equality
        if let (Some(l), Some(r)) = (
            self.get_rat_constant(lhs, manager),
            self.get_rat_constant(rhs, manager),
        ) {
            ctx.stats_mut().record_rule("arith_eq_const");
            if l == r {
                return RewriteResult::Rewritten(manager.mk_true());
            } else {
                return RewriteResult::Rewritten(manager.mk_false());
            }
        }

        RewriteResult::Unchanged(manager.mk_eq(lhs, rhs))
    }

    /// Check if term is an arithmetic term
    fn is_arith_term(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
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
                | TermKind::Ge(_, _)
        )
    }

    /// Clear the constant cache
    pub fn clear_cache(&mut self) {
        self.int_cache.clear();
        self.rat_cache.clear();
    }
}

impl Rewriter for ArithRewriter {
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
            TermKind::Add(args) => {
                let args_cloned = args.clone();
                self.rewrite_add(&args_cloned, ctx, manager)
            }
            TermKind::Mul(args) => {
                let args_cloned = args.clone();
                self.rewrite_mul(&args_cloned, ctx, manager)
            }
            TermKind::Sub(lhs, rhs) => self.rewrite_sub(*lhs, *rhs, ctx, manager),
            TermKind::Neg(arg) => self.rewrite_neg(*arg, ctx, manager),
            TermKind::Div(lhs, rhs) => self.rewrite_div(*lhs, *rhs, ctx, manager),
            TermKind::Mod(lhs, rhs) => self.rewrite_mod(*lhs, *rhs, ctx, manager),
            TermKind::Lt(lhs, rhs) => self.rewrite_lt(*lhs, *rhs, ctx, manager),
            TermKind::Le(lhs, rhs) => self.rewrite_le(*lhs, *rhs, ctx, manager),
            TermKind::Gt(lhs, rhs) => self.rewrite_gt(*lhs, *rhs, ctx, manager),
            TermKind::Ge(lhs, rhs) => self.rewrite_ge(*lhs, *rhs, ctx, manager),
            TermKind::Eq(lhs, rhs) => {
                // Check if it's an arithmetic equality
                if self.is_arith_term(*lhs, manager) || self.is_arith_term(*rhs, manager) {
                    self.rewrite_arith_eq(*lhs, *rhs, ctx, manager)
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "ArithRewriter"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::Add(_)
                | TermKind::Sub(_, _)
                | TermKind::Mul(_)
                | TermKind::Div(_, _)
                | TermKind::Mod(_, _)
                | TermKind::Neg(_)
                | TermKind::Lt(_, _)
                | TermKind::Le(_, _)
                | TermKind::Gt(_, _)
                | TermKind::Ge(_, _)
                | TermKind::Eq(_, _)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;

    fn setup() -> (TermManager, RewriteContext, ArithRewriter) {
        (
            TermManager::new(),
            RewriteContext::new(),
            ArithRewriter::new(),
        )
    }

    #[test]
    fn test_add_zero() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let add = manager.mk_add([x, zero]);

        let result = rewriter.rewrite(add, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_add_constants() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let two = manager.mk_int(2);
        let three = manager.mk_int(3);
        let add = manager.mk_add([two, three]);

        let result = rewriter.rewrite(add, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(&t.kind, TermKind::IntConst(n) if n == &BigInt::from(5)));
    }

    #[test]
    fn test_mul_zero() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let zero = manager.mk_int(0);
        let mul = manager.mk_mul([x, zero]);

        let result = rewriter.rewrite(mul, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(&t.kind, TermKind::IntConst(n) if n == &BigInt::from(0)));
    }

    #[test]
    fn test_mul_one() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let one = manager.mk_int(1);
        let mul = manager.mk_mul([x, one]);

        let result = rewriter.rewrite(mul, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_sub_self() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let sub = manager.mk_sub(x, x);

        let result = rewriter.rewrite(sub, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(&t.kind, TermKind::IntConst(n) if n == &BigInt::from(0)));
    }

    #[test]
    fn test_double_neg() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let neg = manager.mk_neg(x);
        let double_neg = manager.mk_neg(neg);

        let result = rewriter.rewrite(double_neg, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_div_one() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let one = manager.mk_int(1);
        let div = manager.mk_div(x, one);

        let result = rewriter.rewrite(div, &mut ctx, &mut manager);
        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_lt_self() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let lt = manager.mk_lt(x, x);

        let result = rewriter.rewrite(lt, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::False));
    }

    #[test]
    fn test_le_self() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let le = manager.mk_le(x, x);

        let result = rewriter.rewrite(le, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_lt_constants() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let two = manager.mk_int(2);
        let five = manager.mk_int(5);
        let lt = manager.mk_lt(two, five);

        let result = rewriter.rewrite(lt, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_eq_self() {
        let (mut manager, _ctx, _rewriter) = setup();

        // Note: mk_eq already simplifies eq(x, x) -> true at term creation
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let eq = manager.mk_eq(x, x);

        // Verify the simplification happened at creation time
        let t = manager.get(eq).expect("term should exist");
        assert!(matches!(t.kind, TermKind::True));
    }

    #[test]
    fn test_mod_one() {
        let (mut manager, mut ctx, mut rewriter) = setup();

        let x = manager.mk_var("x", manager.sorts.int_sort);
        let one = manager.mk_int(1);
        let modop = manager.mk_mod(x, one);

        let result = rewriter.rewrite(modop, &mut ctx, &mut manager);
        assert!(result.was_rewritten());

        let t = manager.get(result.term()).expect("term should exist");
        assert!(matches!(&t.kind, TermKind::IntConst(n) if n == &BigInt::from(0)));
    }
}
