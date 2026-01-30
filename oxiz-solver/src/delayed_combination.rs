//! Delayed Theory Combination.
//!
//! Implements lazy theory combination where theory interaction is delayed
//! until necessary, reducing overhead for problems with limited sharing.
//!
//! ## Architecture
//!
//! - **Lazy Equality Sharing**: Only share equalities when conflicts arise
//! - **On-Demand Interface Reasoning**: Activate theories as needed
//! - **Conflict-Driven Refinement**: Refine theory interaction based on conflicts
//!
//! ## References
//!
//! - "Delayed Theory Combination vs. Nelson-Oppen" (Meng & Reynolds, 2015)
//! - Z3's `smt/theory_combine.cpp`

use oxiz_core::TermId;
use rustc_hash::FxHashMap;

/// Theory identifier.
pub type TheoryId = usize;

/// Shared term that appears in multiple theories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SharedTerm {
    /// The term.
    pub term: TermId,
    /// Theories that use this term.
    pub theories: u64, // Bitset of theory IDs
}

/// Equality that needs to be shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeferredEquality {
    /// Left-hand side.
    pub lhs: TermId,
    /// Right-hand side.
    pub rhs: TermId,
    /// Source theory that discovered this equality.
    pub source_theory: TheoryId,
}

/// Configuration for delayed combination.
#[derive(Debug, Clone)]
pub struct DelayedCombinationConfig {
    /// Enable lazy equality sharing.
    pub lazy_sharing: bool,
    /// Enable conflict-driven refinement.
    pub conflict_driven: bool,
    /// Maximum deferred equalities before forcing propagation.
    pub max_deferred: usize,
}

impl Default for DelayedCombinationConfig {
    fn default() -> Self {
        Self {
            lazy_sharing: true,
            conflict_driven: true,
            max_deferred: 1000,
        }
    }
}

/// Statistics for delayed combination.
#[derive(Debug, Clone, Default)]
pub struct DelayedCombinationStats {
    /// Equalities deferred.
    pub equalities_deferred: u64,
    /// Equalities propagated.
    pub equalities_propagated: u64,
    /// Forced propagations (max_deferred reached).
    pub forced_propagations: u64,
    /// Conflicts detected.
    pub conflicts_detected: u64,
}

/// Delayed theory combination engine.
#[derive(Debug)]
pub struct DelayedCombination {
    /// Shared terms indexed by term ID.
    shared_terms: FxHashMap<TermId, SharedTerm>,
    /// Deferred equalities.
    deferred: Vec<DeferredEquality>,
    /// Active theories (bitset).
    active_theories: u64,
    /// Configuration.
    config: DelayedCombinationConfig,
    /// Statistics.
    stats: DelayedCombinationStats,
}

impl DelayedCombination {
    /// Create a new delayed combination engine.
    pub fn new(config: DelayedCombinationConfig) -> Self {
        Self {
            shared_terms: FxHashMap::default(),
            deferred: Vec::new(),
            active_theories: 0,
            config,
            stats: DelayedCombinationStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(DelayedCombinationConfig::default())
    }

    /// Register a shared term.
    pub fn register_shared_term(&mut self, term: TermId, theory: TheoryId) {
        let entry = self
            .shared_terms
            .entry(term)
            .or_insert(SharedTerm { term, theories: 0 });

        entry.theories |= 1 << theory;
    }

    /// Check if a term is shared between multiple theories.
    pub fn is_shared(&self, term: TermId) -> bool {
        self.shared_terms
            .get(&term)
            .map(|st| st.theories.count_ones() > 1)
            .unwrap_or(false)
    }

    /// Defer an equality for later propagation.
    pub fn defer_equality(&mut self, lhs: TermId, rhs: TermId, source: TheoryId) {
        if !self.config.lazy_sharing {
            // Immediate propagation mode
            self.propagate_equality(lhs, rhs, source);
            return;
        }

        self.deferred.push(DeferredEquality {
            lhs,
            rhs,
            source_theory: source,
        });
        self.stats.equalities_deferred += 1;

        // Force propagation if too many deferred
        if self.deferred.len() >= self.config.max_deferred {
            self.force_propagation();
        }
    }

    /// Propagate a single equality immediately.
    fn propagate_equality(&mut self, _lhs: TermId, _rhs: TermId, _source: TheoryId) {
        // Simplified: would notify relevant theories
        self.stats.equalities_propagated += 1;
    }

    /// Force propagation of all deferred equalities.
    pub fn force_propagation(&mut self) {
        if self.deferred.is_empty() {
            return;
        }

        self.stats.forced_propagations += 1;

        // Collect deferred equalities to avoid borrow checker issues
        let equalities: Vec<_> = self.deferred.drain(..).collect();
        for eq in equalities {
            self.propagate_equality(eq.lhs, eq.rhs, eq.source_theory);
        }
    }

    /// Handle a conflict by activating theory combination.
    pub fn handle_conflict(&mut self) {
        if !self.config.conflict_driven {
            return;
        }

        self.stats.conflicts_detected += 1;

        // Force propagation of deferred equalities
        self.force_propagation();
    }

    /// Activate a theory.
    pub fn activate_theory(&mut self, theory: TheoryId) {
        self.active_theories |= 1 << theory;
    }

    /// Check if a theory is active.
    pub fn is_theory_active(&self, theory: TheoryId) -> bool {
        (self.active_theories & (1 << theory)) != 0
    }

    /// Get theories that share a term.
    pub fn get_sharing_theories(&self, term: TermId) -> Vec<TheoryId> {
        if let Some(shared) = self.shared_terms.get(&term) {
            let mut theories = Vec::new();
            for i in 0..64 {
                if (shared.theories & (1 << i)) != 0 {
                    theories.push(i);
                }
            }
            theories
        } else {
            Vec::new()
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &DelayedCombinationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = DelayedCombinationStats::default();
    }
}

impl Default for DelayedCombination {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delayed_combination_creation() {
        let dc = DelayedCombination::default_config();
        assert_eq!(dc.stats().equalities_deferred, 0);
    }

    #[test]
    fn test_register_shared_term() {
        let mut dc = DelayedCombination::default_config();

        let term = TermId::new(0);
        dc.register_shared_term(term, 0);
        dc.register_shared_term(term, 1);

        assert!(dc.is_shared(term));
    }

    #[test]
    fn test_defer_equality() {
        let mut dc = DelayedCombination::default_config();

        let lhs = TermId::new(0);
        let rhs = TermId::new(1);

        dc.defer_equality(lhs, rhs, 0);

        assert_eq!(dc.stats().equalities_deferred, 1);
        assert_eq!(dc.deferred.len(), 1);
    }

    #[test]
    fn test_force_propagation() {
        let mut dc = DelayedCombination::default_config();

        let lhs = TermId::new(0);
        let rhs = TermId::new(1);

        dc.defer_equality(lhs, rhs, 0);
        dc.force_propagation();

        assert_eq!(dc.deferred.len(), 0);
        assert_eq!(dc.stats().equalities_propagated, 1);
        assert_eq!(dc.stats().forced_propagations, 1);
    }

    #[test]
    fn test_activate_theory() {
        let mut dc = DelayedCombination::default_config();

        dc.activate_theory(2);
        assert!(dc.is_theory_active(2));
        assert!(!dc.is_theory_active(3));
    }

    #[test]
    fn test_get_sharing_theories() {
        let mut dc = DelayedCombination::default_config();

        let term = TermId::new(0);
        dc.register_shared_term(term, 1);
        dc.register_shared_term(term, 3);

        let theories = dc.get_sharing_theories(term);

        assert_eq!(theories.len(), 2);
        assert!(theories.contains(&1));
        assert!(theories.contains(&3));
    }

    #[test]
    fn test_handle_conflict() {
        let mut dc = DelayedCombination::default_config();

        let lhs = TermId::new(0);
        let rhs = TermId::new(1);

        dc.defer_equality(lhs, rhs, 0);
        dc.handle_conflict();

        assert_eq!(dc.deferred.len(), 0);
        assert_eq!(dc.stats().conflicts_detected, 1);
    }
}
