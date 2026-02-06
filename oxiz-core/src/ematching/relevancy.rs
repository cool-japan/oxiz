//! Relevancy propagation for E-matching
//!
//! This module implements relevancy tracking to focus instantiation on
//! relevant terms. Relevancy helps reduce the search space by identifying which terms
//! are actually needed for the current proof search.

use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use rustc_hash::{FxHashMap, FxHashSet};

/// Relevancy score for a term
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct RelevancyScore(pub f64);

impl RelevancyScore {
    /// Create a new relevancy score
    pub const fn new(score: f64) -> Self {
        Self(score)
    }

    /// Get the score value
    pub const fn value(&self) -> f64 {
        self.0
    }

    /// Maximum relevancy
    pub const fn max() -> Self {
        Self(1.0)
    }

    /// Minimum relevancy
    pub const fn min() -> Self {
        Self(0.0)
    }

    /// Combine two relevancy scores
    pub fn combine(self, other: Self) -> Self {
        Self((self.0 + other.0) / 2.0)
    }
}

/// Configuration for relevancy tracking
#[derive(Debug, Clone)]
pub struct RelevancyConfig {
    /// Whether to enable relevancy tracking
    pub enabled: bool,
    /// Initial relevancy for new terms
    pub initial_score: f64,
    /// Decay factor for propagation
    pub decay_factor: f64,
    /// Minimum score threshold
    pub min_threshold: f64,
    /// Maximum propagation depth
    pub max_depth: usize,
}

impl Default for RelevancyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_score: 1.0,
            decay_factor: 0.9,
            min_threshold: 0.1,
            max_depth: 10,
        }
    }
}

/// Statistics about relevancy tracking
#[derive(Debug, Clone, Default)]
pub struct RelevancyStats {
    /// Number of terms tracked
    pub terms_tracked: usize,
    /// Number of propagations performed
    pub propagations: usize,
    /// Average relevancy score
    pub avg_score: f64,
    /// Number of relevant terms (above threshold)
    pub relevant_terms: usize,
}

/// Relevancy tracker
#[derive(Debug)]
pub struct RelevancyTracker {
    /// Configuration
    config: RelevancyConfig,
    /// Term relevancy scores
    scores: FxHashMap<TermId, RelevancyScore>,
    /// Relevant terms (above threshold)
    relevant: FxHashSet<TermId>,
    /// Statistics
    stats: RelevancyStats,
}

impl RelevancyTracker {
    /// Create a new relevancy tracker
    pub fn new(config: RelevancyConfig) -> Self {
        Self {
            config,
            scores: FxHashMap::default(),
            relevant: FxHashSet::default(),
            stats: RelevancyStats::default(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(RelevancyConfig::default())
    }

    /// Set term relevancy score
    pub fn set_score(&mut self, term: TermId, score: RelevancyScore) {
        self.scores.insert(term, score);

        if score.value() >= self.config.min_threshold {
            self.relevant.insert(term);
        } else {
            self.relevant.remove(&term);
        }

        self.update_stats();
    }

    /// Get term relevancy score
    pub fn get_score(&self, term: TermId) -> RelevancyScore {
        self.scores
            .get(&term)
            .copied()
            .unwrap_or(RelevancyScore::new(self.config.initial_score))
    }

    /// Check if term is relevant
    pub fn is_relevant(&self, term: TermId) -> bool {
        if !self.config.enabled {
            return true;
        }
        self.relevant.contains(&term)
    }

    /// Mark term as relevant
    pub fn mark_relevant(&mut self, term: TermId) {
        self.set_score(term, RelevancyScore::max());
    }

    /// Propagate relevancy from a term to its subterms
    pub fn propagate(&mut self, term: TermId, manager: &TermManager) -> Result<()> {
        self.stats.propagations += 1;
        self.propagate_recursive(term, 0, manager)
    }

    /// Recursive propagation
    fn propagate_recursive(
        &mut self,
        term: TermId,
        depth: usize,
        manager: &TermManager,
    ) -> Result<()> {
        if depth >= self.config.max_depth {
            return Ok(());
        }

        let current_score = self.get_score(term);
        let propagated_score =
            RelevancyScore::new(current_score.value() * self.config.decay_factor);

        if propagated_score.value() < self.config.min_threshold {
            return Ok(());
        }

        let Some(t) = manager.get(term) else {
            return Ok(());
        };

        // Propagate to subterms
        match &t.kind {
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    let old_score = self.get_score(arg);
                    let new_score = old_score.combine(propagated_score);
                    self.set_score(arg, new_score);
                    self.propagate_recursive(arg, depth + 1, manager)?;
                }
            }
            TermKind::Eq(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                for &child in &[*lhs, *rhs] {
                    let old_score = self.get_score(child);
                    let new_score = old_score.combine(propagated_score);
                    self.set_score(child, new_score);
                    self.propagate_recursive(child, depth + 1, manager)?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.terms_tracked = self.scores.len();
        self.stats.relevant_terms = self.relevant.len();

        let total: f64 = self.scores.values().map(|s| s.value()).sum();
        self.stats.avg_score = if !self.scores.is_empty() {
            total / self.scores.len() as f64
        } else {
            0.0
        };
    }

    /// Get statistics
    pub fn stats(&self) -> &RelevancyStats {
        &self.stats
    }

    /// Clear all tracking
    pub fn clear(&mut self) {
        self.scores.clear();
        self.relevant.clear();
        self.stats = RelevancyStats::default();
    }

    /// Get all relevant terms
    pub fn get_relevant_terms(&self) -> &FxHashSet<TermId> {
        &self.relevant
    }
}

/// Relevancy propagator
#[derive(Debug)]
pub struct RelevancyPropagator {
    /// Tracker
    tracker: RelevancyTracker,
    /// Propagation queue
    queue: Vec<TermId>,
}

impl RelevancyPropagator {
    /// Create a new propagator
    pub fn new(config: RelevancyConfig) -> Self {
        Self {
            tracker: RelevancyTracker::new(config),
            queue: Vec::new(),
        }
    }

    /// Add term to propagation queue
    pub fn enqueue(&mut self, term: TermId) {
        self.queue.push(term);
        self.tracker.mark_relevant(term);
    }

    /// Process propagation queue
    pub fn process_queue(&mut self, manager: &TermManager) -> Result<()> {
        while let Some(term) = self.queue.pop() {
            self.tracker.propagate(term, manager)?;
        }
        Ok(())
    }

    /// Get tracker
    pub fn tracker(&self) -> &RelevancyTracker {
        &self.tracker
    }

    /// Get mutable tracker
    pub fn tracker_mut(&mut self) -> &mut RelevancyTracker {
        &mut self.tracker
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
    fn test_relevancy_score() {
        let score = RelevancyScore::new(0.8);
        assert_eq!(score.value(), 0.8);
    }

    #[test]
    fn test_score_combine() {
        let s1 = RelevancyScore::new(0.8);
        let s2 = RelevancyScore::new(0.6);
        let combined = s1.combine(s2);
        assert_eq!(combined.value(), 0.7);
    }

    #[test]
    fn test_relevancy_tracker() {
        let tracker = RelevancyTracker::new_default();
        assert_eq!(tracker.stats.terms_tracked, 0);
    }

    #[test]
    fn test_set_score() {
        let mut tracker = RelevancyTracker::new_default();
        let term = TermId(1);

        tracker.set_score(term, RelevancyScore::new(0.9));
        assert_eq!(tracker.get_score(term).value(), 0.9);
    }

    #[test]
    fn test_is_relevant() {
        let mut tracker = RelevancyTracker::new_default();
        let term = TermId(1);

        tracker.mark_relevant(term);
        assert!(tracker.is_relevant(term));
    }
}
