//! Shared Terms Management for Theory Combination.
#![allow(dead_code)] // Under development
//!
//! Manages terms that appear in multiple theories, enabling efficient
//! equality sharing in Nelson-Oppen combination.
//!
//! ## Architecture
//!
//! - **Term Index**: Fast lookup of shared terms
//! - **Theory Subscriptions**: Theories register interest in terms
//! - **Notification System**: Propagate equality information between theories
//!
//! ## References
//!
//! - Nelson & Oppen: "Simplification by Cooperating Decision Procedures" (1979)
//! - Z3's `smt/theory_combine.cpp`

use oxiz_core::TermId;
use rustc_hash::{FxHashMap, FxHashSet};

/// Theory identifier.
pub type TheoryId = usize;

/// Equality between two terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Equality {
    /// Left-hand side term.
    pub lhs: TermId,
    /// Right-hand side term.
    pub rhs: TermId,
}

impl Equality {
    /// Create a new equality.
    pub fn new(lhs: TermId, rhs: TermId) -> Self {
        // Normalize: smaller TermId first
        if lhs.raw() <= rhs.raw() {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }
}

/// Information about a shared term.
#[derive(Debug, Clone)]
struct SharedTermInfo {
    /// Theories that use this term.
    theories: FxHashSet<TheoryId>,
    /// Representative term in equivalence class.
    representative: TermId,
}

/// Configuration for shared terms manager.
#[derive(Debug, Clone)]
pub struct SharedTermsConfig {
    /// Enable notification batching.
    pub enable_batching: bool,
    /// Maximum batch size before forcing flush.
    pub max_batch_size: usize,
}

impl Default for SharedTermsConfig {
    fn default() -> Self {
        Self {
            enable_batching: true,
            max_batch_size: 1000,
        }
    }
}

/// Statistics for shared terms.
#[derive(Debug, Clone, Default)]
pub struct SharedTermsStats {
    /// Number of shared terms registered.
    pub terms_registered: u64,
    /// Number of theory subscriptions.
    pub subscriptions: u64,
    /// Equalities propagated.
    pub equalities_propagated: u64,
    /// Notification batches sent.
    pub batches_sent: u64,
}

/// Shared terms manager for theory combination.
#[derive(Debug)]
pub struct SharedTermsManager {
    /// Configuration.
    config: SharedTermsConfig,
    /// Shared term information.
    terms: FxHashMap<TermId, SharedTermInfo>,
    /// Equality classes (union-find).
    parent: FxHashMap<TermId, TermId>,
    /// Pending equalities to propagate.
    pending_equalities: Vec<Equality>,
    /// Theories subscribed to each term.
    subscriptions: FxHashMap<TermId, FxHashSet<TheoryId>>,
    /// Statistics.
    stats: SharedTermsStats,
}

impl SharedTermsManager {
    /// Create a new shared terms manager.
    pub fn new(config: SharedTermsConfig) -> Self {
        Self {
            config,
            terms: FxHashMap::default(),
            parent: FxHashMap::default(),
            pending_equalities: Vec::new(),
            subscriptions: FxHashMap::default(),
            stats: SharedTermsStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SharedTermsConfig::default())
    }

    /// Register a shared term.
    pub fn register_term(&mut self, term: TermId, theory: TheoryId) {
        let entry = self.terms.entry(term).or_insert_with(|| {
            self.stats.terms_registered += 1;
            SharedTermInfo {
                theories: FxHashSet::default(),
                representative: term,
            }
        });

        entry.theories.insert(theory);
        self.stats.subscriptions += 1;

        // Also track subscriptions separately for fast lookup
        self.subscriptions.entry(term).or_default().insert(theory);
    }

    /// Check if a term is shared between multiple theories.
    pub fn is_shared(&self, term: TermId) -> bool {
        self.terms
            .get(&term)
            .map(|info| info.theories.len() > 1)
            .unwrap_or(false)
    }

    /// Get theories that use a term.
    pub fn get_theories(&self, term: TermId) -> Vec<TheoryId> {
        self.terms
            .get(&term)
            .map(|info| info.theories.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Assert equality between two terms.
    ///
    /// This merges their equivalence classes and queues notifications.
    pub fn assert_equality(&mut self, lhs: TermId, rhs: TermId) {
        let lhs_rep = self.find(lhs);
        let rhs_rep = self.find(rhs);

        if lhs_rep == rhs_rep {
            return; // Already equal
        }

        // Union: make lhs_rep point to rhs_rep
        self.parent.insert(lhs_rep, rhs_rep);

        // Queue equality for propagation
        let equality = Equality::new(lhs, rhs);
        self.pending_equalities.push(equality);
        self.stats.equalities_propagated += 1;

        // Check if should flush batch
        if self.pending_equalities.len() >= self.config.max_batch_size {
            self.flush_equalities();
        }
    }

    /// Find representative of equivalence class (with path compression).
    fn find(&mut self, term: TermId) -> TermId {
        if let Some(&parent) = self.parent.get(&term)
            && parent != term
        {
            let root = self.find(parent);
            self.parent.insert(term, root); // Path compression
            return root;
        }

        term
    }

    /// Check if two terms are in the same equivalence class.
    pub fn are_equal(&mut self, lhs: TermId, rhs: TermId) -> bool {
        self.find(lhs) == self.find(rhs)
    }

    /// Get pending equalities to propagate.
    pub fn get_pending_equalities(&self) -> &[Equality] {
        &self.pending_equalities
    }

    /// Flush pending equalities (send to theories).
    pub fn flush_equalities(&mut self) {
        if !self.pending_equalities.is_empty() {
            self.stats.batches_sent += 1;
            self.pending_equalities.clear();
        }
    }

    /// Get all shared terms.
    pub fn get_shared_terms(&self) -> Vec<TermId> {
        self.terms
            .iter()
            .filter(|(_, info)| info.theories.len() > 1)
            .map(|(&term, _)| term)
            .collect()
    }

    /// Get statistics.
    pub fn stats(&self) -> &SharedTermsStats {
        &self.stats
    }

    /// Reset manager state.
    pub fn reset(&mut self) {
        self.terms.clear();
        self.parent.clear();
        self.pending_equalities.clear();
        self.subscriptions.clear();
        self.stats = SharedTermsStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(id: u32) -> TermId {
        TermId::new(id)
    }

    #[test]
    fn test_manager_creation() {
        let manager = SharedTermsManager::default_config();
        assert_eq!(manager.stats().terms_registered, 0);
    }

    #[test]
    fn test_register_term() {
        let mut manager = SharedTermsManager::default_config();

        manager.register_term(term(1), 0); // Theory 0
        manager.register_term(term(1), 1); // Theory 1

        assert!(manager.is_shared(term(1)));
        assert_eq!(manager.get_theories(term(1)).len(), 2);
    }

    #[test]
    fn test_equality() {
        let mut manager = SharedTermsManager::default_config();

        manager.assert_equality(term(1), term(2));

        assert!(manager.are_equal(term(1), term(2)));
        assert_eq!(manager.get_pending_equalities().len(), 1);
    }

    #[test]
    fn test_equality_transitivity() {
        let mut manager = SharedTermsManager::default_config();

        manager.assert_equality(term(1), term(2));
        manager.assert_equality(term(2), term(3));

        assert!(manager.are_equal(term(1), term(3)));
    }

    #[test]
    fn test_flush_equalities() {
        let mut manager = SharedTermsManager::default_config();

        manager.assert_equality(term(1), term(2));
        assert_eq!(manager.get_pending_equalities().len(), 1);

        manager.flush_equalities();
        assert_eq!(manager.get_pending_equalities().len(), 0);
    }
}
