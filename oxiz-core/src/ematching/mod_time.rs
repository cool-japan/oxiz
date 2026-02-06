//! Modification time tracking for E-matching optimization
//!
//! This module implements mod-time optimization to avoid redundant instantiations.
//! Each term and quantifier is assigned a generation/modification time, and E-matching
//! only considers terms that have been added or modified since the last instantiation.

use crate::ast::TermId;
use rustc_hash::{FxHashMap, FxHashSet};

/// Modification time (generation counter)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModTime(pub u64);

impl ModTime {
    /// Create a new modification time
    pub const fn new(time: u64) -> Self {
        Self(time)
    }

    /// Get the time value
    pub const fn value(&self) -> u64 {
        self.0
    }

    /// Increment the time
    pub fn increment(&mut self) {
        self.0 += 1;
    }

    /// Check if this time is newer than another
    pub fn is_newer_than(&self, other: ModTime) -> bool {
        self.0 > other.0
    }
}

/// Configuration for mod-time optimization
#[derive(Debug, Clone)]
pub struct ModTimeConfig {
    /// Whether to enable mod-time optimization
    pub enabled: bool,
    /// Whether to track term modifications
    pub track_terms: bool,
    /// Whether to track quantifier instantiations
    pub track_quantifiers: bool,
    /// Cleanup threshold (remove entries older than this)
    pub cleanup_threshold: u64,
}

impl Default for ModTimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            track_terms: true,
            track_quantifiers: true,
            cleanup_threshold: 10000,
        }
    }
}

/// Statistics about mod-time optimization
#[derive(Debug, Clone, Default)]
pub struct ModTimeStats {
    /// Number of terms tracked
    pub terms_tracked: usize,
    /// Number of quantifiers tracked
    pub quantifiers_tracked: usize,
    /// Number of instantiations skipped due to mod-time
    pub instantiations_skipped: usize,
    /// Number of cleanups performed
    pub cleanups_performed: usize,
    /// Current generation
    pub current_generation: u64,
}

/// Modification time manager
#[derive(Debug)]
pub struct ModTimeManager {
    /// Configuration
    config: ModTimeConfig,
    /// Current global time
    current_time: ModTime,
    /// Term modification times
    term_mod_times: FxHashMap<TermId, ModTime>,
    /// Quantifier last instantiation times
    quant_inst_times: FxHashMap<TermId, ModTime>,
    /// Statistics
    stats: ModTimeStats,
}

impl ModTimeManager {
    /// Create a new mod-time manager
    pub fn new(config: ModTimeConfig) -> Self {
        Self {
            config,
            current_time: ModTime::new(0),
            term_mod_times: FxHashMap::default(),
            quant_inst_times: FxHashMap::default(),
            stats: ModTimeStats::default(),
        }
    }

    /// Create with default configuration
    pub fn new_default() -> Self {
        Self::new(ModTimeConfig::default())
    }

    /// Get current time
    pub fn current_time(&self) -> ModTime {
        self.current_time
    }

    /// Advance time
    pub fn tick(&mut self) -> ModTime {
        self.current_time.increment();
        self.stats.current_generation = self.current_time.value();
        self.current_time
    }

    /// Record term modification
    pub fn record_term(&mut self, term: TermId) {
        if !self.config.track_terms {
            return;
        }

        let time = self.current_time;
        self.term_mod_times.insert(term, time);
        self.stats.terms_tracked = self.term_mod_times.len();
    }

    /// Record quantifier instantiation
    pub fn record_instantiation(&mut self, quant: TermId) {
        if !self.config.track_quantifiers {
            return;
        }

        let time = self.current_time;
        self.quant_inst_times.insert(quant, time);
        self.stats.quantifiers_tracked = self.quant_inst_times.len();
    }

    /// Check if term is newer than quantifier's last instantiation
    pub fn is_term_newer(&self, term: TermId, quant: TermId) -> bool {
        if !self.config.enabled {
            return true;
        }

        let term_time = self
            .term_mod_times
            .get(&term)
            .copied()
            .unwrap_or(ModTime::new(0));
        let quant_time = self
            .quant_inst_times
            .get(&quant)
            .copied()
            .unwrap_or(ModTime::new(0));

        term_time.is_newer_than(quant_time)
    }

    /// Check if any term in a set is newer
    pub fn any_newer(&self, terms: &[TermId], quant: TermId) -> bool {
        if !self.config.enabled {
            return true;
        }

        for &term in terms {
            if self.is_term_newer(term, quant) {
                return true;
            }
        }
        false
    }

    /// Cleanup old entries
    pub fn cleanup(&mut self) {
        let threshold_time = ModTime::new(
            self.current_time
                .value()
                .saturating_sub(self.config.cleanup_threshold),
        );

        // Remove old term times
        self.term_mod_times
            .retain(|_, &mut time| time >= threshold_time);

        // Remove old quantifier times
        self.quant_inst_times
            .retain(|_, &mut time| time >= threshold_time);

        self.stats.cleanups_performed += 1;
        self.stats.terms_tracked = self.term_mod_times.len();
        self.stats.quantifiers_tracked = self.quant_inst_times.len();
    }

    /// Get statistics
    pub fn stats(&self) -> &ModTimeStats {
        &self.stats
    }

    /// Clear all tracking
    pub fn clear(&mut self) {
        self.current_time = ModTime::new(0);
        self.term_mod_times.clear();
        self.quant_inst_times.clear();
        self.stats = ModTimeStats::default();
    }
}

/// Tracker for modifications
#[derive(Debug)]
pub struct ModificationTracker {
    /// Modified terms in current round
    modified: FxHashSet<TermId>,
    /// Manager
    manager: ModTimeManager,
}

impl ModificationTracker {
    /// Create a new tracker
    pub fn new(config: ModTimeConfig) -> Self {
        Self {
            modified: FxHashSet::default(),
            manager: ModTimeManager::new(config),
        }
    }

    /// Mark a term as modified
    pub fn mark_modified(&mut self, term: TermId) {
        self.modified.insert(term);
        self.manager.record_term(term);
    }

    /// Get modified terms
    pub fn get_modified(&self) -> &FxHashSet<TermId> {
        &self.modified
    }

    /// Clear modified set (start new round)
    pub fn clear_modified(&mut self) {
        self.modified.clear();
        self.manager.tick();
    }

    /// Check if term was modified
    pub fn is_modified(&self, term: TermId) -> bool {
        self.modified.contains(&term)
    }

    /// Get underlying manager
    pub fn manager(&self) -> &ModTimeManager {
        &self.manager
    }

    /// Get mutable manager
    pub fn manager_mut(&mut self) -> &mut ModTimeManager {
        &mut self.manager
    }
}

/// Mod-time optimization helper
pub struct ModTimeOptimization {
    /// Manager
    manager: ModTimeManager,
    /// Terms to check
    pending_terms: Vec<TermId>,
}

impl ModTimeOptimization {
    /// Create a new optimization helper
    pub fn new(config: ModTimeConfig) -> Self {
        Self {
            manager: ModTimeManager::new(config),
            pending_terms: Vec::new(),
        }
    }

    /// Add pending term
    pub fn add_pending(&mut self, term: TermId) {
        self.pending_terms.push(term);
        self.manager.record_term(term);
    }

    /// Check if instantiation is needed for quantifier
    pub fn needs_instantiation(&self, quant: TermId) -> bool {
        self.manager.any_newer(&self.pending_terms, quant)
    }

    /// Clear pending terms
    pub fn clear_pending(&mut self) {
        self.pending_terms.clear();
        self.manager.tick();
    }

    /// Get manager
    pub fn manager(&self) -> &ModTimeManager {
        &self.manager
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
    fn test_mod_time_creation() {
        let mt = ModTime::new(5);
        assert_eq!(mt.value(), 5);
    }

    #[test]
    fn test_mod_time_increment() {
        let mut mt = ModTime::new(5);
        mt.increment();
        assert_eq!(mt.value(), 6);
    }

    #[test]
    fn test_mod_time_comparison() {
        let mt1 = ModTime::new(5);
        let mt2 = ModTime::new(10);
        assert!(mt2.is_newer_than(mt1));
        assert!(!mt1.is_newer_than(mt2));
    }

    #[test]
    fn test_mod_time_manager() {
        let mut mgr = ModTimeManager::new_default();
        assert_eq!(mgr.current_time().value(), 0);

        mgr.tick();
        assert_eq!(mgr.current_time().value(), 1);
    }

    #[test]
    fn test_record_term() {
        let mut mgr = ModTimeManager::new_default();
        let term = TermId(1);

        mgr.record_term(term);
        assert_eq!(mgr.stats.terms_tracked, 1);
    }

    #[test]
    fn test_is_term_newer() {
        let mut mgr = ModTimeManager::new_default();
        let term = TermId(1);
        let quant = TermId(2);

        mgr.record_instantiation(quant);
        mgr.tick();
        mgr.record_term(term);

        assert!(mgr.is_term_newer(term, quant));
    }
}
