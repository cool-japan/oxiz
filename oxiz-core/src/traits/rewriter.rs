//! Term Rewriting Trait.
//!
//! Provides a generic interface for implementing term rewriters that transform
//! expressions while preserving semantics.

use crate::{TermId, TermManager};
use rustc_hash::FxHashMap;

/// Result of a rewrite operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RewriteResult {
    /// Term was rewritten to a new term.
    Changed(TermId),
    /// Term was not changed.
    Unchanged,
    /// Rewrite produced a known value (true/false for Boolean terms).
    Value(bool),
}

impl RewriteResult {
    /// Check if the rewrite changed the term.
    pub fn is_changed(&self) -> bool {
        matches!(self, RewriteResult::Changed(_))
    }

    /// Get the rewritten term ID if changed.
    pub fn get_term(&self) -> Option<TermId> {
        match self {
            RewriteResult::Changed(id) => Some(*id),
            _ => None,
        }
    }

    /// Get the Boolean value if rewrite produced a constant.
    pub fn get_value(&self) -> Option<bool> {
        match self {
            RewriteResult::Value(b) => Some(*b),
            _ => None,
        }
    }
}

/// Statistics for rewriting operations.
#[derive(Debug, Clone, Default)]
pub struct RewriteStats {
    /// Number of terms rewritten.
    pub terms_rewritten: u64,
    /// Number of terms visited (including unchanged).
    pub terms_visited: u64,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Total rewrite time (microseconds).
    pub total_time_us: u64,
}

impl RewriteStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Compute cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Configuration for rewriting.
#[derive(Debug, Clone)]
pub struct RewriteConfig {
    /// Maximum depth for recursive rewriting.
    pub max_depth: usize,
    /// Enable caching of rewrite results.
    pub enable_cache: bool,
    /// Maximum number of iterations for fixpoint computation.
    pub max_iterations: usize,
    /// Whether to rewrite children before parent (bottom-up).
    pub bottom_up: bool,
}

impl Default for RewriteConfig {
    fn default() -> Self {
        Self {
            max_depth: 1000,
            enable_cache: true,
            max_iterations: 10,
            bottom_up: true,
        }
    }
}

/// Core trait for term rewriters.
///
/// A rewriter transforms terms according to some rewrite rules,
/// typically for simplification or normalization.
pub trait Rewriter: Send + Sync {
    /// Get the name of this rewriter.
    fn name(&self) -> &str;

    /// Rewrite a single term.
    ///
    /// Returns `RewriteResult` indicating whether the term was changed.
    fn rewrite(&mut self, term: TermId, tm: &mut TermManager) -> RewriteResult;

    /// Rewrite a term and all its children (deep rewriting).
    fn rewrite_deep(&mut self, term: TermId, tm: &mut TermManager) -> RewriteResult {
        self.rewrite_with_config(term, tm, &RewriteConfig::default())
    }

    /// Rewrite with custom configuration.
    fn rewrite_with_config(
        &mut self,
        term: TermId,
        tm: &mut TermManager,
        config: &RewriteConfig,
    ) -> RewriteResult {
        self.rewrite_recursive(term, tm, config, 0)
    }

    /// Recursive rewrite implementation.
    fn rewrite_recursive(
        &mut self,
        term: TermId,
        tm: &mut TermManager,
        config: &RewriteConfig,
        depth: usize,
    ) -> RewriteResult {
        if depth >= config.max_depth {
            return RewriteResult::Unchanged;
        }

        // Simplified implementation: just rewrite the term
        // Bottom-up rewriting with proper children handling should be
        // implemented by specific rewriters that understand term structure
        let result = self.rewrite(term, tm);

        // If term changed and not bottom-up, rewrite result recursively
        if !config.bottom_up
            && result.is_changed()
            && let Some(new_term) = result.get_term()
        {
            return self.rewrite_recursive(new_term, tm, config, depth + 1);
        }

        result
    }

    /// Reset the rewriter state.
    fn reset(&mut self) {
        // Default: no-op
    }

    /// Get statistics for this rewriter.
    fn stats(&self) -> RewriteStats {
        RewriteStats::default()
    }

    /// Check if this rewriter is idempotent (applying twice = applying once).
    fn is_idempotent(&self) -> bool {
        true
    }
}

/// Extension trait for cached rewriters.
pub trait CachedRewriter: Rewriter {
    /// Get the cache.
    fn cache(&self) -> &FxHashMap<TermId, RewriteResult>;

    /// Get mutable cache.
    fn cache_mut(&mut self) -> &mut FxHashMap<TermId, RewriteResult>;

    /// Clear the cache.
    fn clear_cache(&mut self) {
        self.cache_mut().clear();
    }

    /// Rewrite with caching.
    fn rewrite_cached(&mut self, term: TermId, tm: &mut TermManager) -> RewriteResult {
        // Check cache
        if let Some(result) = self.cache().get(&term) {
            return result.clone();
        }

        // Perform rewrite
        let result = self.rewrite(term, tm);

        // Cache result
        self.cache_mut().insert(term, result.clone());

        result
    }
}

/// Extension trait for conditional rewriters.
///
/// Conditional rewriters only apply when certain conditions are met.
pub trait ConditionalRewriter: Rewriter {
    /// Check if this rewriter should apply to the given term.
    fn should_apply(&self, term: TermId, tm: &TermManager) -> bool;

    /// Rewrite only if condition is met.
    fn rewrite_if(&mut self, term: TermId, tm: &mut TermManager) -> RewriteResult {
        if self.should_apply(term, tm) {
            self.rewrite(term, tm)
        } else {
            RewriteResult::Unchanged
        }
    }
}

/// Composite rewriter that applies multiple rewriters in sequence.
pub struct SequentialRewriter {
    /// Name of this composite rewriter.
    name: String,
    /// Rewriters to apply in order.
    rewriters: Vec<Box<dyn Rewriter>>,
    /// Combined statistics.
    stats: RewriteStats,
}

impl SequentialRewriter {
    /// Create a new sequential rewriter.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rewriters: Vec::new(),
            stats: RewriteStats::default(),
        }
    }

    /// Add a rewriter to the sequence.
    pub fn add(&mut self, rewriter: Box<dyn Rewriter>) {
        self.rewriters.push(rewriter);
    }

    /// Get the number of rewriters.
    pub fn len(&self) -> usize {
        self.rewriters.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.rewriters.is_empty()
    }
}

impl Rewriter for SequentialRewriter {
    fn name(&self) -> &str {
        &self.name
    }

    fn rewrite(&mut self, mut term: TermId, tm: &mut TermManager) -> RewriteResult {
        let mut overall_result = RewriteResult::Unchanged;

        for rewriter in &mut self.rewriters {
            self.stats.terms_visited += 1;
            let result = rewriter.rewrite(term, tm);

            match result {
                RewriteResult::Changed(new_term) => {
                    term = new_term;
                    overall_result = RewriteResult::Changed(new_term);
                    self.stats.terms_rewritten += 1;
                }
                RewriteResult::Value(b) => {
                    return RewriteResult::Value(b);
                }
                RewriteResult::Unchanged => {}
            }
        }

        overall_result
    }

    fn reset(&mut self) {
        for rewriter in &mut self.rewriters {
            rewriter.reset();
        }
        self.stats.reset();
    }

    fn stats(&self) -> RewriteStats {
        self.stats.clone()
    }
}

/// Fixpoint rewriter that applies a rewriter until no changes occur.
pub struct FixpointRewriter {
    /// Name of this rewriter.
    name: String,
    /// Underlying rewriter.
    inner: Box<dyn Rewriter>,
    /// Maximum iterations.
    max_iterations: usize,
    /// Statistics.
    stats: RewriteStats,
}

impl FixpointRewriter {
    /// Create a new fixpoint rewriter.
    pub fn new(name: impl Into<String>, inner: Box<dyn Rewriter>) -> Self {
        Self {
            name: name.into(),
            inner,
            max_iterations: 10,
            stats: RewriteStats::default(),
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }
}

impl Rewriter for FixpointRewriter {
    fn name(&self) -> &str {
        &self.name
    }

    fn rewrite(&mut self, mut term: TermId, tm: &mut TermManager) -> RewriteResult {
        let mut overall_changed = false;

        for _iter in 0..self.max_iterations {
            self.stats.terms_visited += 1;
            let result = self.inner.rewrite(term, tm);

            match result {
                RewriteResult::Changed(new_term) => {
                    term = new_term;
                    overall_changed = true;
                    self.stats.terms_rewritten += 1;
                }
                RewriteResult::Value(b) => {
                    return RewriteResult::Value(b);
                }
                RewriteResult::Unchanged => {
                    break; // Fixpoint reached
                }
            }
        }

        if overall_changed {
            RewriteResult::Changed(term)
        } else {
            RewriteResult::Unchanged
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.stats.reset();
    }

    fn stats(&self) -> RewriteStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock rewriter for testing
    struct MockRewriter {
        name: String,
        should_change: bool,
    }

    impl MockRewriter {
        fn new(name: &str, should_change: bool) -> Self {
            Self {
                name: name.to_string(),
                should_change,
            }
        }
    }

    impl Rewriter for MockRewriter {
        fn name(&self) -> &str {
            &self.name
        }

        fn rewrite(&mut self, term: TermId, _tm: &mut TermManager) -> RewriteResult {
            if self.should_change {
                // Return a different term ID
                RewriteResult::Changed(TermId::new(term.raw() + 1))
            } else {
                RewriteResult::Unchanged
            }
        }
    }

    #[test]
    fn test_rewrite_result() {
        let unchanged = RewriteResult::Unchanged;
        assert!(!unchanged.is_changed());
        assert!(unchanged.get_term().is_none());

        let changed = RewriteResult::Changed(TermId::new(42));
        assert!(changed.is_changed());
        assert_eq!(changed.get_term(), Some(TermId::new(42)));

        let value = RewriteResult::Value(true);
        assert_eq!(value.get_value(), Some(true));
    }

    #[test]
    fn test_sequential_rewriter() {
        let mut seq = SequentialRewriter::new("test");

        seq.add(Box::new(MockRewriter::new("r1", false)));
        seq.add(Box::new(MockRewriter::new("r2", false)));

        assert_eq!(seq.len(), 2);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_rewrite_stats() {
        let mut stats = RewriteStats::new();
        assert_eq!(stats.terms_rewritten, 0);

        stats.terms_visited = 100;
        stats.cache_hits = 60;
        stats.cache_misses = 40;

        assert_eq!(stats.cache_hit_rate(), 0.6);
    }

    #[test]
    fn test_config_defaults() {
        let config = RewriteConfig::default();
        assert_eq!(config.max_depth, 1000);
        assert!(config.enable_cache);
        assert!(config.bottom_up);
    }
}
