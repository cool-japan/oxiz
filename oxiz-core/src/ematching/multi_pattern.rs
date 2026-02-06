//! Multi-pattern matching optimization for E-matching
//!
//! This module provides optimized multi-pattern matching for quantifiers
//! that have multiple triggers. Multi-pattern matching is more complex than
//! single-pattern matching because it requires finding combinations of matches
//! that together cover all bound variables.

use crate::ast::{TermId, TermKind, TermManager};
use crate::ematching::index::TermIndex;
use crate::ematching::pattern::Pattern;
use crate::ematching::substitution::Substitution;
use crate::error::{OxizError, Result};
use lasso::Spur;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Configuration for multi-pattern matching
#[derive(Debug, Clone)]
pub struct MultiPatternConfig {
    /// Maximum number of patterns per quantifier
    pub max_patterns: usize,
    /// Whether to enable pattern caching
    pub enable_caching: bool,
    /// Whether to use shared structure optimization
    pub use_shared_structure: bool,
    /// Maximum cache size
    pub max_cache_size: usize,
    /// Enable aggressive filtering
    pub aggressive_filtering: bool,
}

impl Default for MultiPatternConfig {
    fn default() -> Self {
        Self {
            max_patterns: 5,
            enable_caching: true,
            use_shared_structure: true,
            max_cache_size: 10000,
            aggressive_filtering: true,
        }
    }
}

/// Statistics for multi-pattern matching
#[derive(Debug, Clone, Default)]
pub struct MultiPatternStats {
    /// Total patterns processed
    pub patterns_processed: usize,
    /// Multi-pattern matches found
    pub multi_matches: usize,
    /// Single-pattern matches found
    pub single_matches: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
    /// Average patterns per match
    pub avg_patterns_per_match: f64,
    /// Filtered candidates
    pub filtered_candidates: usize,
}

/// A multi-pattern match result
#[derive(Debug, Clone)]
pub struct MultiPatternMatch {
    /// Pattern indices that matched
    pub pattern_indices: SmallVec<[usize; 4]>,
    /// Substitution mapping variables to terms
    pub substitution: Substitution,
    /// Matched terms for each pattern
    pub matched_terms: SmallVec<[TermId; 4]>,
}

/// Multi-pattern matcher
#[derive(Debug)]
pub struct MultiPatternMatcher {
    /// Configuration
    config: MultiPatternConfig,
    /// Pattern sets (each set is a multi-pattern for one quantifier)
    pattern_sets: Vec<PatternSet>,
    /// Match cache
    cache: FxHashMap<CacheKey, Vec<MultiPatternMatch>>,
    /// Statistics
    stats: MultiPatternStats,
}

/// A set of patterns for a quantifier
#[derive(Debug, Clone)]
struct PatternSet {
    /// The quantifier
    #[allow(dead_code)]
    quantifier: TermId,
    /// Patterns in this set
    patterns: SmallVec<[Pattern; 4]>,
    /// All variables that must be bound
    required_vars: FxHashSet<Spur>,
}

/// Cache key for multi-pattern matches
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    /// Quantifier ID
    quant_id: TermId,
    /// Hash of current terms
    terms_hash: u64,
}

impl MultiPatternMatcher {
    /// Create a new multi-pattern matcher
    pub fn new(config: MultiPatternConfig) -> Self {
        Self {
            config,
            pattern_sets: Vec::new(),
            cache: FxHashMap::default(),
            stats: MultiPatternStats::default(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(MultiPatternConfig::default())
    }

    /// Add a pattern set for a quantifier
    pub fn add_pattern_set(&mut self, quant: TermId, patterns: Vec<Pattern>) -> Result<()> {
        if patterns.len() > self.config.max_patterns {
            return Err(OxizError::EmatchError(format!(
                "Too many patterns: {} > {}",
                patterns.len(),
                self.config.max_patterns
            )));
        }

        // Collect all required variables
        let mut required_vars = FxHashSet::default();
        for pattern in &patterns {
            for var in &pattern.variables {
                required_vars.insert(var.name);
            }
        }

        let pattern_set = PatternSet {
            quantifier: quant,
            patterns: patterns.into_iter().collect(),
            required_vars,
        };

        self.pattern_sets.push(pattern_set);
        Ok(())
    }

    /// Find multi-pattern matches
    pub fn find_matches(
        &mut self,
        term_index: &TermIndex,
        manager: &TermManager,
    ) -> Result<Vec<MultiPatternMatch>> {
        let mut all_matches = Vec::new();

        // Clone pattern_sets to avoid borrow conflict
        let pattern_sets = self.pattern_sets.clone();
        for pattern_set in &pattern_sets {
            let matches = self.find_matches_for_set(pattern_set, term_index, manager)?;
            all_matches.extend(matches);
        }

        Ok(all_matches)
    }

    /// Find matches for a specific pattern set
    fn find_matches_for_set(
        &mut self,
        pattern_set: &PatternSet,
        term_index: &TermIndex,
        manager: &TermManager,
    ) -> Result<Vec<MultiPatternMatch>> {
        self.stats.patterns_processed += pattern_set.patterns.len();

        // Single pattern optimization
        if pattern_set.patterns.len() == 1 {
            return self.find_single_pattern_matches(pattern_set, term_index, manager);
        }

        // Multi-pattern case
        self.find_multi_pattern_matches(pattern_set, term_index, manager)
    }

    /// Find matches for a single pattern
    fn find_single_pattern_matches(
        &mut self,
        pattern_set: &PatternSet,
        term_index: &TermIndex,
        manager: &TermManager,
    ) -> Result<Vec<MultiPatternMatch>> {
        let pattern = &pattern_set.patterns[0];
        let mut matches = Vec::new();

        // Get candidate terms from index
        let candidates = self.get_candidates_for_pattern(pattern, term_index, manager)?;

        for &candidate in &candidates {
            if let Some(subst) = self.try_match_pattern(pattern, candidate, manager)? {
                // Check all required variables are bound
                if pattern_set.required_vars.iter().all(|v| subst.contains(v)) {
                    matches.push(MultiPatternMatch {
                        pattern_indices: smallvec::smallvec![0],
                        substitution: subst,
                        matched_terms: smallvec::smallvec![candidate],
                    });
                    self.stats.single_matches += 1;
                }
            }
        }

        Ok(matches)
    }

    /// Find matches for multiple patterns
    fn find_multi_pattern_matches(
        &mut self,
        pattern_set: &PatternSet,
        term_index: &TermIndex,
        manager: &TermManager,
    ) -> Result<Vec<MultiPatternMatch>> {
        let mut matches = Vec::new();

        // Get candidates for each pattern
        let mut pattern_candidates: Vec<Vec<TermId>> = Vec::new();
        for pattern in &pattern_set.patterns {
            let candidates = self.get_candidates_for_pattern(pattern, term_index, manager)?;
            pattern_candidates.push(candidates);
        }

        // Try all combinations (with pruning)
        self.combine_pattern_matches(pattern_set, &pattern_candidates, &mut matches, manager)?;

        self.stats.multi_matches += matches.len();
        Ok(matches)
    }

    /// Combine matches from multiple patterns
    fn combine_pattern_matches(
        &self,
        pattern_set: &PatternSet,
        pattern_candidates: &[Vec<TermId>],
        matches: &mut Vec<MultiPatternMatch>,
        manager: &TermManager,
    ) -> Result<()> {
        // Use backtracking to find compatible combinations
        let mut current_match = MultiPatternMatch {
            pattern_indices: SmallVec::new(),
            substitution: Substitution::new(),
            matched_terms: SmallVec::new(),
        };

        self.backtrack_combine(
            pattern_set,
            pattern_candidates,
            0,
            &mut current_match,
            matches,
            manager,
        )
    }

    /// Backtracking helper for combining patterns
    fn backtrack_combine(
        &self,
        pattern_set: &PatternSet,
        pattern_candidates: &[Vec<TermId>],
        pattern_idx: usize,
        current: &mut MultiPatternMatch,
        results: &mut Vec<MultiPatternMatch>,
        manager: &TermManager,
    ) -> Result<()> {
        // Base case: all patterns matched
        if pattern_idx >= pattern_set.patterns.len() {
            // Check if all required variables are bound
            if pattern_set
                .required_vars
                .iter()
                .all(|v| current.substitution.contains(v))
            {
                results.push(current.clone());
            }
            return Ok(());
        }

        let pattern = &pattern_set.patterns[pattern_idx];
        let candidates = &pattern_candidates[pattern_idx];

        for &candidate in candidates {
            // Try to match this pattern with current substitution
            if let Some(new_subst) =
                self.try_match_with_subst(pattern, candidate, &current.substitution, manager)?
            {
                // Save current state
                let old_subst = current.substitution.clone();
                let old_len_indices = current.pattern_indices.len();
                let old_len_terms = current.matched_terms.len();

                // Update current match
                current.substitution = new_subst;
                current.pattern_indices.push(pattern_idx);
                current.matched_terms.push(candidate);

                // Recurse
                self.backtrack_combine(
                    pattern_set,
                    pattern_candidates,
                    pattern_idx + 1,
                    current,
                    results,
                    manager,
                )?;

                // Restore state
                current.substitution = old_subst;
                current.pattern_indices.truncate(old_len_indices);
                current.matched_terms.truncate(old_len_terms);
            }
        }

        Ok(())
    }

    /// Get candidate terms for a pattern
    fn get_candidates_for_pattern(
        &mut self,
        pattern: &Pattern,
        term_index: &TermIndex,
        manager: &TermManager,
    ) -> Result<Vec<TermId>> {
        // Get terms from index based on pattern structure
        let all_terms = term_index.all_terms();
        let mut candidates = Vec::new();

        for entry in all_terms {
            if self.is_potential_match(pattern, entry.term, manager)? {
                candidates.push(entry.term);
            }
        }

        self.stats.filtered_candidates += all_terms.len() - candidates.len();
        Ok(candidates)
    }

    /// Check if a term could potentially match a pattern
    fn is_potential_match(
        &self,
        pattern: &Pattern,
        term: TermId,
        manager: &TermManager,
    ) -> Result<bool> {
        let Some(term_data) = manager.get(term) else {
            return Ok(false);
        };

        let Some(pattern_data) = manager.get(pattern.root) else {
            return Ok(false);
        };

        // Quick checks
        if term_data.sort != pattern_data.sort {
            return Ok(false);
        }

        // Check term kind compatibility
        match (&pattern_data.kind, &term_data.kind) {
            (TermKind::Apply { func: pf, args: pa }, TermKind::Apply { func: tf, args: ta }) => {
                Ok(pf == tf && pa.len() == ta.len())
            }
            (TermKind::Eq(_, _), TermKind::Eq(_, _)) => Ok(true),
            (TermKind::Lt(_, _), TermKind::Lt(_, _)) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Try to match a pattern against a term
    fn try_match_pattern(
        &self,
        pattern: &Pattern,
        term: TermId,
        manager: &TermManager,
    ) -> Result<Option<Substitution>> {
        let subst = Substitution::new();
        self.match_recursive(pattern.root, term, subst, manager)
    }

    /// Try to match with an existing substitution
    fn try_match_with_subst(
        &self,
        pattern: &Pattern,
        term: TermId,
        existing_subst: &Substitution,
        manager: &TermManager,
    ) -> Result<Option<Substitution>> {
        let subst = existing_subst.clone();
        self.match_recursive(pattern.root, term, subst, manager)
    }

    /// Recursive pattern matching
    fn match_recursive(
        &self,
        pattern: TermId,
        term: TermId,
        mut subst: Substitution,
        manager: &TermManager,
    ) -> Result<Option<Substitution>> {
        let Some(p) = manager.get(pattern) else {
            return Ok(None);
        };

        let Some(t) = manager.get(term) else {
            return Ok(None);
        };

        match &p.kind {
            TermKind::Var(name) => {
                // Variable in pattern - bind or check consistency
                if let Some(existing) = subst.get(name) {
                    if existing == term {
                        Ok(Some(subst))
                    } else {
                        Ok(None) // Inconsistent binding
                    }
                } else {
                    subst.insert(*name, term);
                    Ok(Some(subst))
                }
            }

            TermKind::Apply { func: pf, args: pa } => {
                if let TermKind::Apply { func: tf, args: ta } = &t.kind {
                    if pf != tf || pa.len() != ta.len() {
                        return Ok(None);
                    }

                    // Match all arguments
                    for (p_arg, t_arg) in pa.iter().zip(ta.iter()) {
                        match self.match_recursive(*p_arg, *t_arg, subst, manager)? {
                            Some(new_subst) => subst = new_subst,
                            None => return Ok(None),
                        }
                    }

                    Ok(Some(subst))
                } else {
                    Ok(None)
                }
            }

            _ => {
                // For other terms, require exact match
                if pattern == term {
                    Ok(Some(subst))
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &MultiPatternStats {
        &self.stats
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.pattern_sets.clear();
        self.cache.clear();
        self.stats = MultiPatternStats::default();
    }
}

/// Builder for multi-pattern configurations
#[derive(Debug)]
pub struct MultiPatternBuilder {
    config: MultiPatternConfig,
}

impl MultiPatternBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: MultiPatternConfig::default(),
        }
    }

    /// Set maximum patterns
    pub fn max_patterns(mut self, max: usize) -> Self {
        self.config.max_patterns = max;
        self
    }

    /// Enable caching
    pub fn with_caching(mut self, enabled: bool) -> Self {
        self.config.enable_caching = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> MultiPatternConfig {
        self.config
    }
}

impl Default for MultiPatternBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared pattern cache for optimization
#[derive(Debug)]
pub struct SharedPatternCache {
    /// Cache of pattern match results
    cache: FxHashMap<(TermId, TermId), Option<Substitution>>,
    /// Maximum cache size
    max_size: usize,
}

impl SharedPatternCache {
    /// Create a new shared cache
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: FxHashMap::default(),
            max_size,
        }
    }

    /// Get cached result
    pub fn get(&self, pattern: TermId, term: TermId) -> Option<&Option<Substitution>> {
        self.cache.get(&(pattern, term))
    }

    /// Insert result
    pub fn insert(&mut self, pattern: TermId, term: TermId, result: Option<Substitution>) {
        if self.cache.len() < self.max_size {
            self.cache.insert((pattern, term), result);
        }
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;

    #[allow(dead_code)]
    fn setup() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_config_default() {
        let config = MultiPatternConfig::default();
        assert_eq!(config.max_patterns, 5);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_matcher_creation() {
        let matcher = MultiPatternMatcher::new_default();
        assert_eq!(matcher.pattern_sets.len(), 0);
    }

    #[test]
    fn test_shared_cache() {
        let mut cache = SharedPatternCache::new(100);
        let p = TermId(1);
        let t = TermId(2);

        assert!(cache.get(p, t).is_none());

        cache.insert(p, t, Some(Substitution::new()));
        assert!(cache.get(p, t).is_some());
    }
}
