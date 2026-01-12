//! Term Rewriting Framework
//!
//! This module provides a comprehensive term rewriting framework supporting:
//! - Bottom-up and top-down rewriting strategies
//! - Theory-specific rewriters (arithmetic, bit-vector, array, string)
//! - Rewrite rule compilation and caching
//! - Normalization and canonicalization
//!
//! # Architecture
//!
//! The rewriting system uses a modular architecture:
//! - `Rewriter` trait: Common interface for all rewriters
//! - `RewriteContext`: Shared context with caching and statistics
//! - Specialized rewriters: ArithRewriter, BvRewriter, ArrayRewriter, etc.
//!
//! # Example
//!
//! ```ignore
//! use oxiz_core::rewrite::{Rewriter, RewriteContext, ArithRewriter};
//!
//! let mut ctx = RewriteContext::new();
//! let mut arith = ArithRewriter::new();
//! let simplified = arith.rewrite(term, &mut ctx, &mut manager)?;
//! ```

pub mod arith;
pub mod array;
pub mod bool;
pub mod bv;
pub mod combined;
pub mod datatype;
pub mod fp;
pub mod nla;
pub mod poly;
pub mod quantifier;
pub mod regex;
pub mod string;
pub mod uf;

use crate::ast::{TermId, TermManager};
use rustc_hash::FxHashMap;
use std::fmt::Debug;

/// Result of a rewrite operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteResult {
    /// Term was rewritten to a new term
    Rewritten(TermId),
    /// Term was unchanged
    Unchanged(TermId),
}

impl RewriteResult {
    /// Get the resulting term (whether changed or not)
    #[must_use]
    pub fn term(&self) -> TermId {
        match self {
            Self::Rewritten(t) | Self::Unchanged(t) => *t,
        }
    }

    /// Check if the term was rewritten
    #[must_use]
    pub fn was_rewritten(&self) -> bool {
        matches!(self, Self::Rewritten(_))
    }
}

/// Rewriting strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RewriteStrategy {
    /// Rewrite children first, then parent (bottom-up)
    #[default]
    BottomUp,
    /// Rewrite parent first, then children (top-down)
    TopDown,
    /// Single pass, no recursion
    SinglePass,
    /// Fixed-point iteration until no changes
    FixedPoint,
}

/// Statistics for rewriting operations
#[derive(Debug, Clone, Default)]
pub struct RewriteStats {
    /// Total number of terms visited
    pub terms_visited: u64,
    /// Number of successful rewrites
    pub rewrites_applied: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Number of fixed-point iterations
    pub iterations: u64,
    /// Number of rule applications by rule name
    pub rule_applications: FxHashMap<String, u64>,
}

impl RewriteStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge with another statistics
    pub fn merge(&mut self, other: &RewriteStats) {
        self.terms_visited += other.terms_visited;
        self.rewrites_applied += other.rewrites_applied;
        self.cache_hits += other.cache_hits;
        self.cache_misses += other.cache_misses;
        self.iterations += other.iterations;
        for (rule, count) in &other.rule_applications {
            *self.rule_applications.entry(rule.clone()).or_default() += count;
        }
    }

    /// Record a rule application
    pub fn record_rule(&mut self, rule_name: &str) {
        *self
            .rule_applications
            .entry(rule_name.to_string())
            .or_default() += 1;
        self.rewrites_applied += 1;
    }

    /// Get the cache hit rate
    #[must_use]
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Configuration for rewriting
#[derive(Debug, Clone)]
pub struct RewriteConfig {
    /// Rewriting strategy
    pub strategy: RewriteStrategy,
    /// Maximum number of iterations for fixed-point
    pub max_iterations: usize,
    /// Enable caching
    pub enable_cache: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable aggressive simplification
    pub aggressive: bool,
    /// Sort arguments (for AC normalization)
    pub sort_args: bool,
    /// Flatten nested operations
    pub flatten: bool,
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            strategy: RewriteStrategy::BottomUp,
            max_iterations: 100,
            enable_cache: true,
            max_cache_size: 100_000,
            aggressive: false,
            sort_args: true,
            flatten: true,
        }
    }
}

/// Rewriting context with caching and statistics
#[derive(Debug)]
pub struct RewriteContext {
    /// Configuration
    pub config: RewriteConfig,
    /// Rewrite cache: original term -> rewritten term
    cache: FxHashMap<TermId, TermId>,
    /// Statistics
    stats: RewriteStats,
    /// Current recursion depth
    depth: usize,
    /// Maximum allowed depth
    max_depth: usize,
}

impl Default for RewriteContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteContext {
    /// Create a new rewrite context
    pub fn new() -> Self {
        Self::with_config(RewriteConfig::default())
    }

    /// Create with specific configuration
    pub fn with_config(config: RewriteConfig) -> Self {
        Self {
            config,
            cache: FxHashMap::default(),
            stats: RewriteStats::new(),
            depth: 0,
            max_depth: 1000,
        }
    }

    /// Look up a term in the cache
    pub fn lookup(&mut self, term: TermId) -> Option<TermId> {
        if !self.config.enable_cache {
            return None;
        }
        if let Some(&cached) = self.cache.get(&term) {
            self.stats.cache_hits += 1;
            Some(cached)
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }

    /// Insert a mapping into the cache
    pub fn insert(&mut self, original: TermId, rewritten: TermId) {
        if !self.config.enable_cache {
            return;
        }
        // Evict if cache is too large (LRU would be better, but this is simpler)
        if self.cache.len() >= self.config.max_cache_size {
            self.cache.clear();
        }
        self.cache.insert(original, rewritten);
    }

    /// Get statistics
    pub fn stats(&self) -> &RewriteStats {
        &self.stats
    }

    /// Get mutable statistics
    pub fn stats_mut(&mut self) -> &mut RewriteStats {
        &mut self.stats
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Reset the context
    pub fn reset(&mut self) {
        self.cache.clear();
        self.stats.reset();
        self.depth = 0;
    }

    /// Enter a new recursion level
    pub fn enter(&mut self) -> bool {
        if self.depth >= self.max_depth {
            return false;
        }
        self.depth += 1;
        true
    }

    /// Exit current recursion level
    pub fn exit(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Get current depth
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// Common trait for all rewriters
pub trait Rewriter: Debug + Send + Sync {
    /// Rewrite a term
    ///
    /// Returns the rewritten term (or the original if unchanged)
    fn rewrite(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult;

    /// Get the rewriter name (for debugging/statistics)
    fn name(&self) -> &str;

    /// Check if this rewriter can handle the given term
    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool;
}

/// Composite rewriter that applies multiple rewriters in sequence
#[derive(Debug)]
pub struct CompositeRewriter {
    /// Child rewriters
    rewriters: Vec<Box<dyn Rewriter>>,
    /// Name
    name: String,
}

impl CompositeRewriter {
    /// Create a new composite rewriter
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            rewriters: Vec::new(),
            name: name.into(),
        }
    }

    /// Add a rewriter
    pub fn add<R: Rewriter + 'static>(&mut self, rewriter: R) {
        self.rewriters.push(Box::new(rewriter));
    }

    /// Create from a list of rewriters
    pub fn from_rewriters(name: impl Into<String>, rewriters: Vec<Box<dyn Rewriter>>) -> Self {
        Self {
            rewriters,
            name: name.into(),
        }
    }
}

impl Rewriter for CompositeRewriter {
    fn rewrite(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let mut current = term;
        let mut any_rewritten = false;

        for rewriter in &mut self.rewriters {
            if rewriter.can_handle(current, manager) {
                match rewriter.rewrite(current, ctx, manager) {
                    RewriteResult::Rewritten(new_term) => {
                        current = new_term;
                        any_rewritten = true;
                    }
                    RewriteResult::Unchanged(_) => {}
                }
            }
        }

        if any_rewritten {
            RewriteResult::Rewritten(current)
        } else {
            RewriteResult::Unchanged(term)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        self.rewriters.iter().any(|r| r.can_handle(term, manager))
    }
}

/// Iterating rewriter that applies until fixed point
#[derive(Debug)]
pub struct IteratingRewriter<R: Rewriter> {
    /// Inner rewriter
    inner: R,
    /// Maximum iterations
    max_iterations: usize,
}

impl<R: Rewriter> IteratingRewriter<R> {
    /// Create a new iterating rewriter
    pub fn new(inner: R, max_iterations: usize) -> Self {
        Self {
            inner,
            max_iterations,
        }
    }
}

impl<R: Rewriter> Rewriter for IteratingRewriter<R> {
    fn rewrite(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        let mut current = term;
        let mut any_rewritten = false;

        for _ in 0..self.max_iterations {
            ctx.stats_mut().iterations += 1;

            match self.inner.rewrite(current, ctx, manager) {
                RewriteResult::Rewritten(new_term) => {
                    current = new_term;
                    any_rewritten = true;
                }
                RewriteResult::Unchanged(_) => break,
            }
        }

        if any_rewritten {
            RewriteResult::Rewritten(current)
        } else {
            RewriteResult::Unchanged(term)
        }
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        self.inner.can_handle(term, manager)
    }
}

/// Bottom-up recursive rewriter
#[derive(Debug)]
pub struct BottomUpRewriter<R: Rewriter> {
    /// Inner rewriter
    inner: R,
}

impl<R: Rewriter> BottomUpRewriter<R> {
    /// Create a new bottom-up rewriter
    pub fn new(inner: R) -> Self {
        Self { inner }
    }

    /// Recursively rewrite all children first
    fn rewrite_children(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        use crate::ast::TermKind;

        let Some(t) = manager.get(term).cloned() else {
            return RewriteResult::Unchanged(term);
        };

        // Rewrite based on term kind
        match &t.kind {
            TermKind::Not(arg) => {
                let new_arg = self.rewrite(*arg, ctx, manager).term();
                if new_arg != *arg {
                    RewriteResult::Rewritten(manager.mk_not(new_arg))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::And(args) => {
                let mut changed = false;
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.rewrite(arg, ctx, manager).term();
                        if new_arg != arg {
                            changed = true;
                        }
                        new_arg
                    })
                    .collect();

                if changed {
                    RewriteResult::Rewritten(manager.mk_and(new_args))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Or(args) => {
                let mut changed = false;
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.rewrite(arg, ctx, manager).term();
                        if new_arg != arg {
                            changed = true;
                        }
                        new_arg
                    })
                    .collect();

                if changed {
                    RewriteResult::Rewritten(manager.mk_or(new_args))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Implies(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_implies(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Eq(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_eq(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let new_cond = self.rewrite(*cond, ctx, manager).term();
                let new_then = self.rewrite(*then_br, ctx, manager).term();
                let new_else = self.rewrite(*else_br, ctx, manager).term();
                if new_cond != *cond || new_then != *then_br || new_else != *else_br {
                    RewriteResult::Rewritten(manager.mk_ite(new_cond, new_then, new_else))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            // Arithmetic
            TermKind::Add(args) => {
                let mut changed = false;
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.rewrite(arg, ctx, manager).term();
                        if new_arg != arg {
                            changed = true;
                        }
                        new_arg
                    })
                    .collect();

                if changed {
                    RewriteResult::Rewritten(manager.mk_add(new_args))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Mul(args) => {
                let mut changed = false;
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&arg| {
                        let new_arg = self.rewrite(arg, ctx, manager).term();
                        if new_arg != arg {
                            changed = true;
                        }
                        new_arg
                    })
                    .collect();

                if changed {
                    RewriteResult::Rewritten(manager.mk_mul(new_args))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Sub(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_sub(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Neg(arg) => {
                let new_arg = self.rewrite(*arg, ctx, manager).term();
                if new_arg != *arg {
                    RewriteResult::Rewritten(manager.mk_neg(new_arg))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Div(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_div(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Mod(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_mod(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            // Comparisons
            TermKind::Lt(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_lt(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Le(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_le(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Gt(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_gt(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            TermKind::Ge(lhs, rhs) => {
                let new_lhs = self.rewrite(*lhs, ctx, manager).term();
                let new_rhs = self.rewrite(*rhs, ctx, manager).term();
                if new_lhs != *lhs || new_rhs != *rhs {
                    RewriteResult::Rewritten(manager.mk_ge(new_lhs, new_rhs))
                } else {
                    RewriteResult::Unchanged(term)
                }
            }
            // Leaves and other terms
            _ => RewriteResult::Unchanged(term),
        }
    }
}

impl<R: Rewriter> Rewriter for BottomUpRewriter<R> {
    fn rewrite(
        &mut self,
        term: TermId,
        ctx: &mut RewriteContext,
        manager: &mut TermManager,
    ) -> RewriteResult {
        ctx.stats_mut().terms_visited += 1;

        // Check cache first
        if let Some(cached) = ctx.lookup(term) {
            return RewriteResult::Rewritten(cached);
        }

        // Check recursion depth
        if !ctx.enter() {
            return RewriteResult::Unchanged(term);
        }

        // First, rewrite all children
        let after_children = self.rewrite_children(term, ctx, manager);

        // Then, apply the inner rewriter
        let result = if self.inner.can_handle(after_children.term(), manager) {
            match self.inner.rewrite(after_children.term(), ctx, manager) {
                RewriteResult::Rewritten(new_term) => RewriteResult::Rewritten(new_term),
                RewriteResult::Unchanged(_) => after_children,
            }
        } else {
            after_children
        };

        ctx.exit();

        // Cache the result
        ctx.insert(term, result.term());

        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn can_handle(&self, term: TermId, manager: &TermManager) -> bool {
        self.inner.can_handle(term, manager)
    }
}

// Re-export main types from submodules
pub use arith::ArithRewriter;
pub use array::ArrayRewriter;
pub use bool::BoolRewriter;
pub use bv::BvRewriter;
pub use combined::CombinedRewriter;
pub use datatype::DatatypeRewriter;
pub use fp::FpRewriter;
pub use nla::NlaRewriter;
pub use poly::{Monomial, Polynomial};
pub use quantifier::QuantifierRewriter;
pub use regex::RegexRewriter;
pub use string::StringRewriter;
pub use uf::UfRewriter;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct IdentityRewriter;

    impl Rewriter for IdentityRewriter {
        fn rewrite(
            &mut self,
            term: TermId,
            _ctx: &mut RewriteContext,
            _manager: &mut TermManager,
        ) -> RewriteResult {
            RewriteResult::Unchanged(term)
        }

        fn name(&self) -> &str {
            "identity"
        }

        fn can_handle(&self, _term: TermId, _manager: &TermManager) -> bool {
            true
        }
    }

    #[test]
    fn test_rewrite_result() {
        let term = TermId(1);
        let unchanged = RewriteResult::Unchanged(term);
        assert!(!unchanged.was_rewritten());
        assert_eq!(unchanged.term(), term);

        let new_term = TermId(2);
        let rewritten = RewriteResult::Rewritten(new_term);
        assert!(rewritten.was_rewritten());
        assert_eq!(rewritten.term(), new_term);
    }

    #[test]
    fn test_rewrite_context() {
        let mut ctx = RewriteContext::new();
        let term1 = TermId(1);
        let term2 = TermId(2);

        // Cache miss initially
        assert!(ctx.lookup(term1).is_none());
        assert_eq!(ctx.stats().cache_misses, 1);

        // Insert and lookup
        ctx.insert(term1, term2);
        assert_eq!(ctx.lookup(term1), Some(term2));
        assert_eq!(ctx.stats().cache_hits, 1);

        // Test depth tracking
        assert!(ctx.enter());
        assert_eq!(ctx.depth(), 1);
        ctx.exit();
        assert_eq!(ctx.depth(), 0);
    }

    #[test]
    fn test_rewrite_stats() {
        let mut stats = RewriteStats::new();

        stats.record_rule("test_rule");
        assert_eq!(stats.rewrites_applied, 1);
        assert_eq!(stats.rule_applications.get("test_rule"), Some(&1));

        stats.cache_hits = 10;
        stats.cache_misses = 5;
        assert!((stats.cache_hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_composite_rewriter() {
        let mut manager = TermManager::new();
        let mut ctx = RewriteContext::new();

        let mut composite = CompositeRewriter::new("composite");
        composite.add(IdentityRewriter);

        let term = manager.mk_true();
        let result = composite.rewrite(term, &mut ctx, &mut manager);
        assert!(!result.was_rewritten());
    }

    #[test]
    fn test_iterating_rewriter() {
        let mut manager = TermManager::new();
        let mut ctx = RewriteContext::new();

        let mut iterating = IteratingRewriter::new(IdentityRewriter, 10);

        let term = manager.mk_true();
        let result = iterating.rewrite(term, &mut ctx, &mut manager);
        assert!(!result.was_rewritten());
        assert_eq!(ctx.stats().iterations, 1);
    }
}
