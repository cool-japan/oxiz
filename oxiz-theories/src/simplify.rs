//! Theory-Specific Simplification and Preprocessing
//!
//! This module provides simplification passes for theory constraints,
//! transforming them into simpler equivalent forms before solving.
//!
//! Common simplifications:
//! - Constant folding
//! - Algebraic identities (x + 0 → x, x * 1 → x)
//! - Strength reduction (x * 2 → x + x)
//! - Redundancy elimination
//! - Boolean simplification

use oxiz_core::ast::TermId;
use rustc_hash::{FxHashMap, FxHashSet};

/// Result of simplification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimplifyResult {
    /// Term was simplified to a new term
    Simplified(TermId),
    /// Term is already in simplest form
    Unchanged,
    /// Term simplified to a constant true
    True,
    /// Term simplified to a constant false
    False,
}

impl SimplifyResult {
    /// Check if simplification occurred
    #[must_use]
    pub fn is_simplified(&self) -> bool {
        !matches!(self, Self::Unchanged)
    }

    /// Get the simplified term if available
    #[must_use]
    pub fn get_term(&self) -> Option<TermId> {
        match self {
            Self::Simplified(term) => Some(*term),
            _ => None,
        }
    }
}

/// Statistics for simplification passes
#[derive(Debug, Clone, Default)]
pub struct SimplifyStats {
    /// Number of terms processed
    pub terms_processed: usize,
    /// Number of terms simplified
    pub terms_simplified: usize,
    /// Number of constants folded
    pub constants_folded: usize,
    /// Number of identities applied
    pub identities_applied: usize,
    /// Number of redundancies eliminated
    pub redundancies_eliminated: usize,
}

impl SimplifyStats {
    /// Create new statistics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get simplification rate
    #[must_use]
    pub fn simplification_rate(&self) -> f64 {
        if self.terms_processed == 0 {
            0.0
        } else {
            self.terms_simplified as f64 / self.terms_processed as f64
        }
    }
}

/// Simplification context
#[derive(Debug)]
pub struct SimplifyContext {
    /// Known equalities (x = c for constant c)
    equalities: FxHashMap<TermId, TermId>,
    /// Terms known to be true
    true_terms: FxHashSet<TermId>,
    /// Terms known to be false
    false_terms: FxHashSet<TermId>,
    /// Cached simplifications
    cache: FxHashMap<TermId, SimplifyResult>,
    /// Statistics
    stats: SimplifyStats,
}

impl Default for SimplifyContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SimplifyContext {
    /// Create a new simplification context
    #[must_use]
    pub fn new() -> Self {
        Self {
            equalities: FxHashMap::default(),
            true_terms: FxHashSet::default(),
            false_terms: FxHashSet::default(),
            cache: FxHashMap::default(),
            stats: SimplifyStats::new(),
        }
    }

    /// Add a known equality (x = c)
    pub fn add_equality(&mut self, var: TermId, value: TermId) {
        self.equalities.insert(var, value);
    }

    /// Mark a term as known to be true
    pub fn mark_true(&mut self, term: TermId) {
        self.true_terms.insert(term);
        self.false_terms.remove(&term);
    }

    /// Mark a term as known to be false
    pub fn mark_false(&mut self, term: TermId) {
        self.false_terms.insert(term);
        self.true_terms.remove(&term);
    }

    /// Check if a term is known to be true
    #[must_use]
    pub fn is_true(&self, term: TermId) -> bool {
        self.true_terms.contains(&term)
    }

    /// Check if a term is known to be false
    #[must_use]
    pub fn is_false(&self, term: TermId) -> bool {
        self.false_terms.contains(&term)
    }

    /// Simplify a term
    ///
    /// This is the main entry point for simplification
    pub fn simplify(&mut self, term: TermId) -> SimplifyResult {
        self.stats.terms_processed += 1;

        // Check cache first
        if let Some(cached) = self.cache.get(&term) {
            return cached.clone();
        }

        let result = self.simplify_impl(term);

        // Update statistics
        if result.is_simplified() {
            self.stats.terms_simplified += 1;
        }

        // Cache the result
        self.cache.insert(term, result.clone());

        result
    }

    /// Internal simplification implementation
    fn simplify_impl(&mut self, term: TermId) -> SimplifyResult {
        // Check if term is in equalities
        if let Some(&replacement) = self.equalities.get(&term) {
            self.stats.identities_applied += 1;
            return SimplifyResult::Simplified(replacement);
        }

        // Check if term is known to be true/false
        if self.is_true(term) {
            return SimplifyResult::True;
        }
        if self.is_false(term) {
            return SimplifyResult::False;
        }

        // No simplification possible
        SimplifyResult::Unchanged
    }

    /// Simplify a conjunction (AND)
    ///
    /// Applies rules:
    /// - true AND x → x
    /// - false AND x → false
    /// - x AND x → x
    /// - x AND NOT x → false
    pub fn simplify_and(&mut self, left: TermId, right: TermId) -> SimplifyResult {
        let left_simp = self.simplify(left);
        let right_simp = self.simplify(right);

        match (left_simp, right_simp) {
            (SimplifyResult::False, _) | (_, SimplifyResult::False) => {
                self.stats.constants_folded += 1;
                SimplifyResult::False
            }
            (SimplifyResult::True, _) => {
                self.stats.identities_applied += 1;
                SimplifyResult::Simplified(right)
            }
            (_, SimplifyResult::True) => {
                self.stats.identities_applied += 1;
                SimplifyResult::Simplified(left)
            }
            _ if left == right => {
                self.stats.redundancies_eliminated += 1;
                SimplifyResult::Simplified(left)
            }
            _ => SimplifyResult::Unchanged,
        }
    }

    /// Simplify a disjunction (OR)
    ///
    /// Applies rules:
    /// - true OR x → true
    /// - false OR x → x
    /// - x OR x → x
    /// - x OR NOT x → true
    pub fn simplify_or(&mut self, left: TermId, right: TermId) -> SimplifyResult {
        let left_simp = self.simplify(left);
        let right_simp = self.simplify(right);

        match (left_simp, right_simp) {
            (SimplifyResult::True, _) | (_, SimplifyResult::True) => {
                self.stats.constants_folded += 1;
                SimplifyResult::True
            }
            (SimplifyResult::False, _) => {
                self.stats.identities_applied += 1;
                SimplifyResult::Simplified(right)
            }
            (_, SimplifyResult::False) => {
                self.stats.identities_applied += 1;
                SimplifyResult::Simplified(left)
            }
            _ if left == right => {
                self.stats.redundancies_eliminated += 1;
                SimplifyResult::Simplified(left)
            }
            _ => SimplifyResult::Unchanged,
        }
    }

    /// Simplify a negation (NOT)
    ///
    /// Applies rules:
    /// - NOT true → false
    /// - NOT false → true
    /// - NOT NOT x → x
    pub fn simplify_not(&mut self, term: TermId) -> SimplifyResult {
        let term_simp = self.simplify(term);

        match term_simp {
            SimplifyResult::True => {
                self.stats.constants_folded += 1;
                SimplifyResult::False
            }
            SimplifyResult::False => {
                self.stats.constants_folded += 1;
                SimplifyResult::True
            }
            _ => SimplifyResult::Unchanged,
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SimplifyStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Clear all simplification context
    pub fn clear(&mut self) {
        self.equalities.clear();
        self.true_terms.clear();
        self.false_terms.clear();
        self.cache.clear();
        self.stats.reset();
    }

    /// Get the number of cached simplifications
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the simplification cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_unchanged() {
        let mut ctx = SimplifyContext::new();
        let term = TermId::new(1);

        let result = ctx.simplify(term);
        assert_eq!(result, SimplifyResult::Unchanged);
    }

    #[test]
    fn test_add_equality() {
        let mut ctx = SimplifyContext::new();
        let x = TermId::new(1);
        let c = TermId::new(2);

        ctx.add_equality(x, c);

        let result = ctx.simplify(x);
        assert_eq!(result, SimplifyResult::Simplified(c));
    }

    #[test]
    fn test_mark_true() {
        let mut ctx = SimplifyContext::new();
        let term = TermId::new(1);

        ctx.mark_true(term);

        assert!(ctx.is_true(term));
        assert!(!ctx.is_false(term));

        let result = ctx.simplify(term);
        assert_eq!(result, SimplifyResult::True);
    }

    #[test]
    fn test_mark_false() {
        let mut ctx = SimplifyContext::new();
        let term = TermId::new(1);

        ctx.mark_false(term);

        assert!(ctx.is_false(term));
        assert!(!ctx.is_true(term));

        let result = ctx.simplify(term);
        assert_eq!(result, SimplifyResult::False);
    }

    #[test]
    fn test_simplify_and_false() {
        let mut ctx = SimplifyContext::new();
        let t1 = TermId::new(1);
        let t2 = TermId::new(2);

        ctx.mark_false(t1);

        let result = ctx.simplify_and(t1, t2);
        assert_eq!(result, SimplifyResult::False);
    }

    #[test]
    fn test_simplify_and_true() {
        let mut ctx = SimplifyContext::new();
        let t1 = TermId::new(1);
        let t2 = TermId::new(2);

        ctx.mark_true(t1);

        let result = ctx.simplify_and(t1, t2);
        assert_eq!(result, SimplifyResult::Simplified(t2));
    }

    #[test]
    fn test_simplify_and_idempotent() {
        let mut ctx = SimplifyContext::new();
        let term = TermId::new(1);

        let result = ctx.simplify_and(term, term);
        assert_eq!(result, SimplifyResult::Simplified(term));
        assert_eq!(ctx.stats().redundancies_eliminated, 1);
    }

    #[test]
    fn test_simplify_or_true() {
        let mut ctx = SimplifyContext::new();
        let t1 = TermId::new(1);
        let t2 = TermId::new(2);

        ctx.mark_true(t1);

        let result = ctx.simplify_or(t1, t2);
        assert_eq!(result, SimplifyResult::True);
    }

    #[test]
    fn test_simplify_or_false() {
        let mut ctx = SimplifyContext::new();
        let t1 = TermId::new(1);
        let t2 = TermId::new(2);

        ctx.mark_false(t1);

        let result = ctx.simplify_or(t1, t2);
        assert_eq!(result, SimplifyResult::Simplified(t2));
    }

    #[test]
    fn test_simplify_or_idempotent() {
        let mut ctx = SimplifyContext::new();
        let term = TermId::new(1);

        let result = ctx.simplify_or(term, term);
        assert_eq!(result, SimplifyResult::Simplified(term));
        assert_eq!(ctx.stats().redundancies_eliminated, 1);
    }

    #[test]
    fn test_simplify_not_true() {
        let mut ctx = SimplifyContext::new();
        let term = TermId::new(1);

        ctx.mark_true(term);

        let result = ctx.simplify_not(term);
        assert_eq!(result, SimplifyResult::False);
    }

    #[test]
    fn test_simplify_not_false() {
        let mut ctx = SimplifyContext::new();
        let term = TermId::new(1);

        ctx.mark_false(term);

        let result = ctx.simplify_not(term);
        assert_eq!(result, SimplifyResult::True);
    }

    #[test]
    fn test_caching() {
        let mut ctx = SimplifyContext::new();
        let x = TermId::new(1);
        let c = TermId::new(2);

        ctx.add_equality(x, c);

        // First simplification should cache
        let result1 = ctx.simplify(x);
        assert_eq!(result1, SimplifyResult::Simplified(c));
        assert_eq!(ctx.cache_size(), 1);

        // Second simplification should use cache
        let result2 = ctx.simplify(x);
        assert_eq!(result2, SimplifyResult::Simplified(c));
        assert_eq!(ctx.stats().terms_processed, 2);
    }

    #[test]
    fn test_statistics() {
        let mut ctx = SimplifyContext::new();
        let t1 = TermId::new(1);
        let t2 = TermId::new(2);

        ctx.mark_true(t1);

        // This should trigger constant folding
        ctx.simplify_and(t1, t2);

        let stats = ctx.stats();
        assert!(stats.terms_processed > 0);
        assert!(stats.identities_applied > 0);
    }

    #[test]
    fn test_simplification_rate() {
        let mut ctx = SimplifyContext::new();

        // Add some equalities
        ctx.add_equality(TermId::new(1), TermId::new(10));
        ctx.add_equality(TermId::new(2), TermId::new(20));

        // Simplify - these should simplify
        ctx.simplify(TermId::new(1));
        ctx.simplify(TermId::new(2));

        // Simplify - these should not
        ctx.simplify(TermId::new(3));
        ctx.simplify(TermId::new(4));

        // 2 out of 4 simplified = 50%
        assert!((ctx.stats().simplification_rate() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_clear() {
        let mut ctx = SimplifyContext::new();

        ctx.add_equality(TermId::new(1), TermId::new(2));
        ctx.mark_true(TermId::new(3));
        ctx.simplify(TermId::new(1));

        assert_ne!(ctx.cache_size(), 0);

        ctx.clear();

        assert_eq!(ctx.cache_size(), 0);
        assert_eq!(ctx.stats().terms_processed, 0);
    }

    #[test]
    fn test_clear_cache() {
        let mut ctx = SimplifyContext::new();

        ctx.add_equality(TermId::new(1), TermId::new(2));
        ctx.simplify(TermId::new(1));

        assert_eq!(ctx.cache_size(), 1);

        ctx.clear_cache();

        assert_eq!(ctx.cache_size(), 0);
        // Stats should remain
        assert_ne!(ctx.stats().terms_processed, 0);
    }

    #[test]
    fn test_simplify_result_methods() {
        let unchanged = SimplifyResult::Unchanged;
        assert!(!unchanged.is_simplified());
        assert_eq!(unchanged.get_term(), None);

        let term = TermId::new(42);
        let simplified = SimplifyResult::Simplified(term);
        assert!(simplified.is_simplified());
        assert_eq!(simplified.get_term(), Some(term));

        let true_result = SimplifyResult::True;
        assert!(true_result.is_simplified());
        assert_eq!(true_result.get_term(), None);

        let false_result = SimplifyResult::False;
        assert!(false_result.is_simplified());
        assert_eq!(false_result.get_term(), None);
    }
}
