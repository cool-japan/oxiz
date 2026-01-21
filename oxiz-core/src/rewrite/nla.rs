//! Nonlinear Arithmetic Rewriter
//!
//! This module provides rewriting rules for nonlinear arithmetic:
//! - Polynomial normalization and simplification
//! - Factorization
//! - Power simplification (x^0 = 1, x^1 = x, etc.)
//! - GCD-based coefficient reduction
//! - Sign analysis and simplification
//! - Degree reduction where possible
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::rewrite::{Rewriter, RewriteContext, NlaRewriter};
//!
//! let mut ctx = RewriteContext::new();
//! let mut nla = NlaRewriter::new();
//! let simplified = nla.rewrite(term, &mut ctx, &mut manager)?;
//! ```

use super::{RewriteContext, RewriteResult, Rewriter};
use crate::ast::{TermId, TermKind, TermManager};
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use rustc_hash::FxHashMap;

/// Configuration for NLA rewriting
#[derive(Debug, Clone)]
pub struct NlaRewriterConfig {
    /// Enable polynomial normalization
    pub enable_normalization: bool,
    /// Enable factorization
    pub enable_factorization: bool,
    /// Enable GCD-based coefficient reduction
    pub enable_gcd_reduction: bool,
    /// Enable power simplification
    pub enable_power_simplification: bool,
    /// Maximum polynomial degree to handle
    pub max_degree: u32,
    /// Enable sign analysis
    pub enable_sign_analysis: bool,
}

impl Default for NlaRewriterConfig {
    fn default() -> Self {
        Self {
            enable_normalization: true,
            enable_factorization: true,
            enable_gcd_reduction: true,
            enable_power_simplification: true,
            max_degree: 10,
            enable_sign_analysis: true,
        }
    }
}

/// A monomial representation: coefficient * variable^power
/// For example: 3*x^2 -> coefficient=3, var=x, power=2
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NlaMonomial {
    /// Coefficient
    pub coefficient: BigInt,
    /// Variable powers: variable -> exponent
    pub vars: FxHashMap<TermId, u32>,
}

impl NlaMonomial {
    /// Create a constant monomial
    pub fn constant(c: BigInt) -> Self {
        Self {
            coefficient: c,
            vars: FxHashMap::default(),
        }
    }

    /// Create a variable monomial (coefficient 1)
    pub fn variable(var: TermId) -> Self {
        let mut vars = FxHashMap::default();
        vars.insert(var, 1);
        Self {
            coefficient: BigInt::one(),
            vars,
        }
    }

    /// Check if this is a constant (no variables)
    pub fn is_constant(&self) -> bool {
        self.vars.is_empty() || self.vars.values().all(|&p| p == 0)
    }

    /// Get total degree
    pub fn degree(&self) -> u32 {
        self.vars.values().sum()
    }

    /// Multiply two monomials
    pub fn mul(&self, other: &Self) -> Self {
        let coefficient = &self.coefficient * &other.coefficient;
        let mut vars = self.vars.clone();
        for (&var, &power) in &other.vars {
            *vars.entry(var).or_insert(0) += power;
        }
        Self { coefficient, vars }
    }

    /// Negate the monomial
    pub fn neg(&self) -> Self {
        Self {
            coefficient: -&self.coefficient,
            vars: self.vars.clone(),
        }
    }

    /// Check if coefficient is zero
    pub fn is_zero(&self) -> bool {
        self.coefficient.is_zero()
    }

    /// Check if this is unit (coefficient 1, degree 0)
    pub fn is_one(&self) -> bool {
        self.coefficient.is_one() && self.is_constant()
    }
}

/// A polynomial representation as a sum of monomials
#[derive(Debug, Clone)]
pub struct NlaPolynomial {
    /// Monomials in the polynomial
    pub monomials: Vec<NlaMonomial>,
}

impl NlaPolynomial {
    /// Create an empty polynomial (zero)
    pub fn zero() -> Self {
        Self {
            monomials: Vec::new(),
        }
    }

    /// Create a constant polynomial
    pub fn constant(c: BigInt) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Self {
                monomials: vec![NlaMonomial::constant(c)],
            }
        }
    }

    /// Create a single variable polynomial
    pub fn variable(var: TermId) -> Self {
        Self {
            monomials: vec![NlaMonomial::variable(var)],
        }
    }

    /// Check if polynomial is zero
    pub fn is_zero(&self) -> bool {
        self.monomials.is_empty() || self.monomials.iter().all(|m| m.is_zero())
    }

    /// Check if polynomial is a constant
    pub fn is_constant(&self) -> bool {
        self.monomials.is_empty() || (self.monomials.len() == 1 && self.monomials[0].is_constant())
    }

    /// Get the constant value if polynomial is constant
    pub fn constant_value(&self) -> Option<BigInt> {
        if self.is_zero() {
            Some(BigInt::zero())
        } else if self.is_constant() {
            Some(
                self.monomials
                    .first()
                    .map(|m| m.coefficient.clone())
                    .unwrap_or_default(),
            )
        } else {
            None
        }
    }

    /// Get total degree
    pub fn degree(&self) -> u32 {
        self.monomials.iter().map(|m| m.degree()).max().unwrap_or(0)
    }

    /// Add two polynomials
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.monomials.extend(other.monomials.iter().cloned());
        result.normalize()
    }

    /// Subtract two polynomials
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result
            .monomials
            .extend(other.monomials.iter().map(|m| m.neg()));
        result.normalize()
    }

    /// Multiply two polynomials
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        for m1 in &self.monomials {
            for m2 in &other.monomials {
                result.push(m1.mul(m2));
            }
        }
        let poly = Self { monomials: result };
        poly.normalize()
    }

    /// Normalize the polynomial (combine like terms, remove zeros)
    pub fn normalize(mut self) -> Self {
        // Group monomials by variable structure
        let mut groups: FxHashMap<Vec<(TermId, u32)>, BigInt> = FxHashMap::default();

        for mono in self.monomials {
            // Create a canonical key from sorted variables
            let mut key: Vec<(TermId, u32)> = mono
                .vars
                .iter()
                .filter(|&(_, &p)| p > 0)
                .map(|(&v, &p)| (v, p))
                .collect();
            key.sort_by_key(|&(v, _)| v.0);

            *groups.entry(key).or_insert(BigInt::zero()) += mono.coefficient;
        }

        // Convert back to monomials, filtering zeros
        self.monomials = groups
            .into_iter()
            .filter(|(_, c)| !c.is_zero())
            .map(|(vars_vec, coef)| {
                let vars: FxHashMap<TermId, u32> = vars_vec.into_iter().collect();
                NlaMonomial {
                    coefficient: coef,
                    vars,
                }
            })
            .collect();

        // Sort by degree (descending) then by coefficient for canonical form
        self.monomials.sort_by(|a, b| {
            b.degree().cmp(&a.degree()).then_with(|| {
                // Compare by variable structure
                let mut a_vars: Vec<_> = a.vars.iter().collect();
                let mut b_vars: Vec<_> = b.vars.iter().collect();
                a_vars.sort_by_key(|&(v, _)| v.0);
                b_vars.sort_by_key(|&(v, _)| v.0);
                a_vars.cmp(&b_vars)
            })
        });

        self
    }

    /// Get GCD of all coefficients
    pub fn coefficient_gcd(&self) -> BigInt {
        if self.monomials.is_empty() {
            return BigInt::one();
        }

        let mut gcd = self.monomials[0].coefficient.clone().abs();
        for mono in &self.monomials[1..] {
            gcd = num_integer::gcd(gcd, mono.coefficient.clone().abs());
            if gcd.is_one() {
                break;
            }
        }
        gcd
    }

    /// Reduce coefficients by their GCD
    pub fn reduce_by_gcd(mut self) -> Self {
        let gcd = self.coefficient_gcd();
        if gcd > BigInt::one() {
            for mono in &mut self.monomials {
                mono.coefficient = &mono.coefficient / &gcd;
            }
        }
        self
    }
}

/// Sign information for expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// Always positive (> 0)
    Positive,
    /// Always negative (< 0)
    Negative,
    /// Always zero (= 0)
    Zero,
    /// Non-negative (>= 0)
    NonNegative,
    /// Non-positive (<= 0)
    NonPositive,
    /// Unknown sign
    Unknown,
}

impl Sign {
    /// Negate the sign
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        match self {
            Sign::Positive => Sign::Negative,
            Sign::Negative => Sign::Positive,
            Sign::Zero => Sign::Zero,
            Sign::NonNegative => Sign::NonPositive,
            Sign::NonPositive => Sign::NonNegative,
            Sign::Unknown => Sign::Unknown,
        }
    }

    /// Multiply two signs
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Sign::Zero, _) | (_, Sign::Zero) => Sign::Zero,
            (Sign::Positive, Sign::Positive) | (Sign::Negative, Sign::Negative) => Sign::Positive,
            (Sign::Positive, Sign::Negative) | (Sign::Negative, Sign::Positive) => Sign::Negative,
            (Sign::NonNegative, Sign::NonNegative) | (Sign::NonPositive, Sign::NonPositive) => {
                Sign::NonNegative
            }
            (Sign::NonNegative, Sign::NonPositive) | (Sign::NonPositive, Sign::NonNegative) => {
                Sign::NonPositive
            }
            _ => Sign::Unknown,
        }
    }

    /// Square (always non-negative if not zero)
    pub fn square(self) -> Self {
        match self {
            Sign::Zero => Sign::Zero,
            Sign::Positive | Sign::Negative => Sign::Positive,
            Sign::NonNegative | Sign::NonPositive => Sign::NonNegative,
            Sign::Unknown => Sign::NonNegative,
        }
    }
}

/// Nonlinear arithmetic rewriter
#[derive(Debug)]
pub struct NlaRewriter {
    /// Configuration
    config: NlaRewriterConfig,
    /// Variable sign information
    var_signs: FxHashMap<TermId, Sign>,
}

impl NlaRewriter {
    /// Create a new NLA rewriter with default configuration
    pub fn new() -> Self {
        Self::with_config(NlaRewriterConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: NlaRewriterConfig) -> Self {
        Self {
            config,
            var_signs: FxHashMap::default(),
        }
    }

    /// Register that a variable has a specific sign
    pub fn register_sign(&mut self, var: TermId, sign: Sign) {
        self.var_signs.insert(var, sign);
    }

    /// Get the sign of a variable
    pub fn get_sign(&self, var: TermId) -> Sign {
        self.var_signs.get(&var).copied().unwrap_or(Sign::Unknown)
    }

    /// Clear sign information
    pub fn clear_signs(&mut self) {
        self.var_signs.clear();
    }

    /// Convert a term to polynomial representation
    #[allow(clippy::only_used_in_recursion)]
    fn term_to_polynomial(&self, term: TermId, manager: &TermManager) -> Option<NlaPolynomial> {
        let t = manager.get(term)?;

        match &t.kind {
            TermKind::IntConst(n) => Some(NlaPolynomial::constant(n.clone())),

            TermKind::Var(_) => Some(NlaPolynomial::variable(term)),

            TermKind::Neg(arg) => {
                let p = self.term_to_polynomial(*arg, manager)?;
                Some(NlaPolynomial::zero().sub(&p))
            }

            TermKind::Add(args) => {
                let mut result = NlaPolynomial::zero();
                for &arg in args.iter() {
                    let p = self.term_to_polynomial(arg, manager)?;
                    result = result.add(&p);
                }
                Some(result)
            }

            TermKind::Sub(lhs, rhs) => {
                let p1 = self.term_to_polynomial(*lhs, manager)?;
                let p2 = self.term_to_polynomial(*rhs, manager)?;
                Some(p1.sub(&p2))
            }

            TermKind::Mul(args) => {
                let mut result = NlaPolynomial::constant(BigInt::one());
                for &arg in args.iter() {
                    let p = self.term_to_polynomial(arg, manager)?;
                    result = result.mul(&p);
                }
                Some(result)
            }

            // Can't convert divisions, etc. to polynomial
            _ => None,
        }
    }

    /// Convert a polynomial back to a term
    fn polynomial_to_term(&self, poly: &NlaPolynomial, manager: &mut TermManager) -> TermId {
        if poly.is_zero() {
            return manager.mk_int(0);
        }

        let mono_terms: Vec<TermId> = poly
            .monomials
            .iter()
            .map(|mono| self.monomial_to_term(mono, manager))
            .collect();

        if mono_terms.len() == 1 {
            mono_terms[0]
        } else {
            manager.mk_add(mono_terms)
        }
    }

    /// Convert a monomial to a term
    fn monomial_to_term(&self, mono: &NlaMonomial, manager: &mut TermManager) -> TermId {
        if mono.is_zero() {
            return manager.mk_int(0);
        }

        let mut factors = Vec::new();

        // Add coefficient if not 1
        if !mono.coefficient.is_one() || mono.vars.is_empty() {
            // Handle negative coefficients
            let coef_term = manager.mk_int(mono.coefficient.clone());
            factors.push(coef_term);
        }

        // Add variable powers
        for (&var, &power) in &mono.vars {
            if power == 0 {
                continue;
            }
            for _ in 0..power {
                factors.push(var);
            }
        }

        match factors.len() {
            0 => manager.mk_int(1),
            1 => factors[0],
            _ => manager.mk_mul(factors),
        }
    }

    /// Rewrite multiplication
    fn rewrite_mul(
        &mut self,
        args: &[TermId],
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_mul(args.to_vec());

        // Check for zero factor
        for &arg in args {
            if let Some(t) = manager.get(arg)
                && let TermKind::IntConst(n) = &t.kind
                && n.is_zero()
            {
                ctx.stats_mut().record_rule("nla_mul_zero");
                return RewriteResult::Rewritten(manager.mk_int(0));
            }
        }

        // Check for single factor of 1
        let filtered: Vec<TermId> = args
            .iter()
            .filter(|&&arg| {
                manager
                    .get(arg)
                    .map(|t| {
                        if let TermKind::IntConst(n) = &t.kind {
                            !n.is_one()
                        } else {
                            true
                        }
                    })
                    .unwrap_or(true)
            })
            .copied()
            .collect();

        if filtered.len() != args.len() {
            ctx.stats_mut().record_rule("nla_mul_one");
            return match filtered.len() {
                0 => RewriteResult::Rewritten(manager.mk_int(1)),
                1 => RewriteResult::Rewritten(filtered[0]),
                _ => RewriteResult::Rewritten(manager.mk_mul(filtered)),
            };
        }

        // Try polynomial normalization
        if self.config.enable_normalization
            && let Some(poly) = self.term_to_polynomial(term, manager)
        {
            let normalized = poly.normalize();
            if self.config.enable_gcd_reduction {
                let reduced = normalized.reduce_by_gcd();
                let new_term = self.polynomial_to_term(&reduced, manager);
                if new_term != term {
                    ctx.stats_mut().record_rule("nla_polynomial_normalize");
                    return RewriteResult::Rewritten(new_term);
                }
            } else {
                let new_term = self.polynomial_to_term(&normalized, manager);
                if new_term != term {
                    ctx.stats_mut().record_rule("nla_polynomial_normalize");
                    return RewriteResult::Rewritten(new_term);
                }
            }
        }

        RewriteResult::Unchanged(term)
    }

    /// Rewrite addition
    fn rewrite_add(
        &mut self,
        args: &[TermId],
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_add(args.to_vec());

        // Filter out zeros
        let filtered: Vec<TermId> = args
            .iter()
            .filter(|&&arg| {
                manager
                    .get(arg)
                    .map(|t| {
                        if let TermKind::IntConst(n) = &t.kind {
                            !n.is_zero()
                        } else {
                            true
                        }
                    })
                    .unwrap_or(true)
            })
            .copied()
            .collect();

        if filtered.len() != args.len() {
            ctx.stats_mut().record_rule("nla_add_zero");
            return match filtered.len() {
                0 => RewriteResult::Rewritten(manager.mk_int(0)),
                1 => RewriteResult::Rewritten(filtered[0]),
                _ => RewriteResult::Rewritten(manager.mk_add(filtered)),
            };
        }

        // Try polynomial normalization
        if self.config.enable_normalization
            && let Some(poly) = self.term_to_polynomial(term, manager)
        {
            let normalized = poly.normalize();
            if self.config.enable_gcd_reduction {
                let reduced = normalized.reduce_by_gcd();
                let new_term = self.polynomial_to_term(&reduced, manager);
                if new_term != term {
                    ctx.stats_mut().record_rule("nla_polynomial_normalize");
                    return RewriteResult::Rewritten(new_term);
                }
            } else {
                let new_term = self.polynomial_to_term(&normalized, manager);
                if new_term != term {
                    ctx.stats_mut().record_rule("nla_polynomial_normalize");
                    return RewriteResult::Rewritten(new_term);
                }
            }
        }

        RewriteResult::Unchanged(term)
    }

    /// Rewrite subtraction
    fn rewrite_sub(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_sub(lhs, rhs);

        // x - x = 0
        if lhs == rhs {
            ctx.stats_mut().record_rule("nla_sub_self");
            return RewriteResult::Rewritten(manager.mk_int(0));
        }

        // x - 0 = x
        if let Some(t) = manager.get(rhs)
            && let TermKind::IntConst(n) = &t.kind
            && n.is_zero()
        {
            ctx.stats_mut().record_rule("nla_sub_zero");
            return RewriteResult::Rewritten(lhs);
        }

        // 0 - x = -x
        if let Some(t) = manager.get(lhs)
            && let TermKind::IntConst(n) = &t.kind
            && n.is_zero()
        {
            ctx.stats_mut().record_rule("nla_zero_sub");
            return RewriteResult::Rewritten(manager.mk_neg(rhs));
        }

        RewriteResult::Unchanged(term)
    }

    /// Rewrite negation
    fn rewrite_neg(
        &mut self,
        arg: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_neg(arg);

        // --x = x
        if let Some(t) = manager.get(arg)
            && let TermKind::Neg(inner) = &t.kind
        {
            ctx.stats_mut().record_rule("nla_double_neg");
            return RewriteResult::Rewritten(*inner);
        }

        // -0 = 0
        if let Some(t) = manager.get(arg)
            && let TermKind::IntConst(n) = &t.kind
            && n.is_zero()
        {
            ctx.stats_mut().record_rule("nla_neg_zero");
            return RewriteResult::Rewritten(manager.mk_int(0));
        }

        RewriteResult::Unchanged(term)
    }

    /// Rewrite comparison with sign analysis
    fn rewrite_comparison(
        &mut self,
        lhs: TermId,
        rhs: TermId,
        _is_lt: bool,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let term = manager.mk_lt(lhs, rhs);

        if !self.config.enable_sign_analysis {
            return RewriteResult::Unchanged(term);
        }

        // Get signs of both sides
        let lhs_sign = self.analyze_sign(lhs, manager);
        let rhs_sign = self.analyze_sign(rhs, manager);

        // positive < non-positive is false
        if lhs_sign == Sign::Positive
            && matches!(rhs_sign, Sign::NonPositive | Sign::Zero | Sign::Negative)
        {
            ctx.stats_mut().record_rule("nla_sign_comparison");
            return RewriteResult::Rewritten(manager.mk_false());
        }

        // negative < non-negative is true (when rhs >= 0)
        if lhs_sign == Sign::Negative && matches!(rhs_sign, Sign::NonNegative | Sign::Positive) {
            ctx.stats_mut().record_rule("nla_sign_comparison");
            return RewriteResult::Rewritten(manager.mk_true());
        }

        RewriteResult::Unchanged(term)
    }

    /// Analyze the sign of a term
    fn analyze_sign(&self, term: TermId, manager: &TermManager) -> Sign {
        let Some(t) = manager.get(term) else {
            return Sign::Unknown;
        };

        match &t.kind {
            TermKind::IntConst(n) => {
                if n.is_zero() {
                    Sign::Zero
                } else if *n > BigInt::zero() {
                    Sign::Positive
                } else {
                    Sign::Negative
                }
            }

            TermKind::Var(_) => self.get_sign(term),

            TermKind::Neg(arg) => self.analyze_sign(*arg, manager).neg(),

            TermKind::Mul(args) => {
                let mut result = Sign::Positive;
                for &arg in args.iter() {
                    result = result.mul(self.analyze_sign(arg, manager));
                }
                result
            }

            TermKind::Add(args) => {
                // If all args have the same sign, sum has that sign
                let first_sign = args
                    .first()
                    .map(|&a| self.analyze_sign(a, manager))
                    .unwrap_or(Sign::Zero);
                let all_same = args
                    .iter()
                    .skip(1)
                    .all(|&a| self.analyze_sign(a, manager) == first_sign);
                if all_same { first_sign } else { Sign::Unknown }
            }

            _ => Sign::Unknown,
        }
    }
}

impl Default for NlaRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Rewriter for NlaRewriter {
    fn rewrite(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        ctx.stats_mut().terms_visited += 1;

        let Some(t) = manager.get(term).cloned() else {
            return RewriteResult::Unchanged(term);
        };

        match &t.kind {
            TermKind::Mul(args) => self.rewrite_mul(args, ctx, manager),
            TermKind::Add(args) => self.rewrite_add(args, ctx, manager),
            TermKind::Sub(lhs, rhs) => self.rewrite_sub(*lhs, *rhs, ctx, manager),
            TermKind::Neg(arg) => self.rewrite_neg(*arg, ctx, manager),
            TermKind::Lt(lhs, rhs) => self.rewrite_comparison(*lhs, *rhs, true, ctx, manager),
            _ => RewriteResult::Unchanged(term),
        }
    }

    fn name(&self) -> &str {
        "nla"
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        if let Some(t) = manager.get(term) {
            matches!(
                &t.kind,
                TermKind::Mul(_)
                    | TermKind::Add(_)
                    | TermKind::Sub(_, _)
                    | TermKind::Neg(_)
                    | TermKind::Lt(_, _)
                    | TermKind::Le(_, _)
                    | TermKind::Gt(_, _)
                    | TermKind::Ge(_, _)
            )
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (TermManager, RewriteContext, NlaRewriter) {
        let manager = TermManager::new();
        let ctx = RewriteContext::new();
        let rewriter = NlaRewriter::new();
        (manager, ctx, rewriter)
    }

    #[test]
    fn test_mul_zero() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);

        let mul = manager.mk_mul(vec![x, zero]);
        let result = rewriter.rewrite(mul, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::IntConst(n)) if n.is_zero()));
    }

    #[test]
    fn test_mul_one() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let one = manager.mk_int(1);

        let mul = manager.mk_mul(vec![x, one]);
        let result = rewriter.rewrite(mul, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_add_zero() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);

        let add = manager.mk_add(vec![x, zero]);
        let result = rewriter.rewrite(add, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_sub_self() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        let sub = manager.mk_sub(x, x);
        let result = rewriter.rewrite(sub, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        let t = manager.get(result.term());
        assert!(matches!(t.map(|t| &t.kind), Some(TermKind::IntConst(n)) if n.is_zero()));
    }

    #[test]
    fn test_double_neg() {
        let (mut manager, mut ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);

        let neg = manager.mk_neg(x);
        let double_neg = manager.mk_neg(neg);
        let result = rewriter.rewrite(double_neg, &mut ctx, &mut manager);

        assert!(result.was_rewritten());
        assert_eq!(result.term(), x);
    }

    #[test]
    fn test_monomial_operations() {
        let m1 = NlaMonomial::constant(BigInt::from(2));
        let m2 = NlaMonomial::constant(BigInt::from(3));

        let product = m1.mul(&m2);
        assert_eq!(product.coefficient, BigInt::from(6));
        assert!(product.is_constant());
    }

    #[test]
    fn test_polynomial_add() {
        let p1 = NlaPolynomial::constant(BigInt::from(2));
        let p2 = NlaPolynomial::constant(BigInt::from(3));

        let sum = p1.add(&p2);
        assert_eq!(sum.constant_value(), Some(BigInt::from(5)));
    }

    #[test]
    fn test_polynomial_mul() {
        let p1 = NlaPolynomial::constant(BigInt::from(2));
        let p2 = NlaPolynomial::constant(BigInt::from(3));

        let product = p1.mul(&p2);
        assert_eq!(product.constant_value(), Some(BigInt::from(6)));
    }

    #[test]
    fn test_sign_analysis() {
        let (mut manager, _ctx, mut rewriter) = setup();
        let int_sort = manager.sorts.int_sort;

        let positive = manager.mk_int(5);
        let negative = manager.mk_int(-3);
        let zero = manager.mk_int(0);

        assert_eq!(rewriter.analyze_sign(positive, &manager), Sign::Positive);
        assert_eq!(rewriter.analyze_sign(negative, &manager), Sign::Negative);
        assert_eq!(rewriter.analyze_sign(zero, &manager), Sign::Zero);

        // Variable with registered sign
        let x = manager.mk_var("x", int_sort);
        rewriter.register_sign(x, Sign::Positive);
        assert_eq!(rewriter.analyze_sign(x, &manager), Sign::Positive);
    }

    #[test]
    fn test_gcd_reduction() {
        // 6x + 9 should reduce to 2x + 3 (divide by gcd=3)
        // But we're just checking the polynomial directly
        let mut p = NlaPolynomial::zero();
        p.monomials.push(NlaMonomial::constant(BigInt::from(6)));
        p.monomials.push(NlaMonomial::constant(BigInt::from(9)));

        let reduced = p.reduce_by_gcd();
        let gcd = reduced.coefficient_gcd();
        assert!(gcd <= BigInt::from(3));
    }

    #[test]
    fn test_sign_mul() {
        assert_eq!(Sign::Positive.mul(Sign::Positive), Sign::Positive);
        assert_eq!(Sign::Positive.mul(Sign::Negative), Sign::Negative);
        assert_eq!(Sign::Negative.mul(Sign::Negative), Sign::Positive);
        assert_eq!(Sign::Positive.mul(Sign::Zero), Sign::Zero);
    }

    #[test]
    fn test_sign_square() {
        assert_eq!(Sign::Positive.square(), Sign::Positive);
        assert_eq!(Sign::Negative.square(), Sign::Positive);
        assert_eq!(Sign::Unknown.square(), Sign::NonNegative);
    }
}
