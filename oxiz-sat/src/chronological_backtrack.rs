//! Chronological Backtracking for CDCL SAT Solvers.
//!
//! This module implements chronological backtracking, a modern technique that
//! improves upon traditional non-chronological backjumping in CDCL solvers.
//!
//! ## Background
//!
//! Traditional CDCL backjumps to the asserting level of the learned clause,
//! which can skip many decision levels. While this enables powerful pruning,
//! it can also:
//! - Miss opportunities for early propagation
//! - Cause repeated conflicts at similar parts of the search space
//! - Lose valuable intermediate assignments
//!
//! ## Chronological Backtracking
//!
//! Chronological backtracking backtracks one level at a time (or to a nearby level)
//! instead of always jumping to the asserting level. This hybrid approach:
//!
//! 1. **Preserves Search Progress**: Keeps more assignments, avoiding rework
//! 2. **Enables Propagation**: Learned clauses can propagate at intermediate levels
//! 3. **Reduces Conflicts**: Fewer repeated conflicts in similar search regions
//!
//! ## When to Use Chronological Backtracking
//!
//! Use chronological backtracking when:
//! - Distance from conflict level to asserting level is large
//! - The learned clause has low LBD (good quality)
//! - We've recently done chronological backtracks (locality)
//!
//! Fall back to non-chronological backjumping when:
//! - Distance is small (no benefit)
//! - The learned clause has high LBD (weaker clause)
//! - We need to escape a difficult region
//!
//! ## References
//!
//! - Nadel & Ryvchin: "Chronological Backtracking" (SAT 2018)
//! - CaDiCaL solver implementation
//! - MapleSAT chronological backtracking

// Chronological backtracking implementation

/// Configuration for chronological backtracking.
#[derive(Debug, Clone)]
pub struct ChronoBacktrackConfig {
    /// Enable chronological backtracking.
    pub enabled: bool,
    /// Minimum distance threshold (conflict_level - asserting_level).
    /// Only use chrono backtrack if distance >= this value.
    pub min_distance: u32,
    /// Maximum LBD for chronological backtracking.
    /// Only use chrono backtrack for clauses with LBD <= this value.
    pub max_lbd: u32,
    /// Cutoff level: never backtrack chronologically beyond this many levels.
    pub max_backtrack_distance: u32,
    /// Enable locality-based decision (use chrono if recently used).
    pub enable_locality: bool,
    /// Locality window (number of recent conflicts to consider).
    pub locality_window: usize,
}

impl Default for ChronoBacktrackConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_distance: 10,
            max_lbd: 6,
            max_backtrack_distance: 100,
            enable_locality: true,
            locality_window: 100,
        }
    }
}

/// Statistics for chronological backtracking.
#[derive(Debug, Clone, Default)]
pub struct ChronoBacktrackStats {
    /// Number of chronological backtracks.
    pub chrono_backtracks: u64,
    /// Number of non-chronological backjumps.
    pub nonchrono_backjumps: u64,
    /// Total backtrack distance saved (levels preserved).
    pub levels_saved: u64,
    /// Number of times locality heuristic triggered chrono.
    pub locality_triggers: u64,
    /// Number of times LBD threshold triggered chrono.
    pub lbd_triggers: u64,
    /// Number of times distance threshold triggered chrono.
    pub distance_triggers: u64,
}

/// Backtracking decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BacktrackDecision {
    /// Backtrack chronologically (one or few levels).
    Chronological {
        /// Target level.
        target_level: u32,
    },
    /// Backjump non-chronologically (to asserting level).
    NonChronological {
        /// Target asserting level.
        target_level: u32,
    },
}

/// History entry for locality-based decisions.
#[derive(Debug, Clone, Copy)]
struct BacktrackHistoryEntry {
    /// Was this a chronological backtrack?
    was_chrono: bool,
    /// Conflict LBD.
    #[allow(dead_code)]
    lbd: u32,
}

/// Chronological backtracking decision engine.
pub struct ChronoBacktrackEngine {
    /// Configuration.
    config: ChronoBacktrackConfig,
    /// Statistics.
    stats: ChronoBacktrackStats,
    /// Recent backtrack history (for locality).
    history: Vec<BacktrackHistoryEntry>,
}

impl ChronoBacktrackEngine {
    /// Create a new chronological backtracking engine.
    pub fn new() -> Self {
        Self::with_config(ChronoBacktrackConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: ChronoBacktrackConfig) -> Self {
        Self {
            config,
            stats: ChronoBacktrackStats::default(),
            history: Vec::new(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ChronoBacktrackStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ChronoBacktrackStats::default();
    }

    /// Decide whether to backtrack chronologically or non-chronologically.
    ///
    /// # Arguments
    /// * `conflict_level` - Current decision level where conflict occurred
    /// * `asserting_level` - Level where learned clause becomes unit
    /// * `clause_lbd` - LBD of the learned clause
    pub fn decide_backtrack(
        &mut self,
        conflict_level: u32,
        asserting_level: u32,
        clause_lbd: u32,
    ) -> BacktrackDecision {
        if !self.config.enabled {
            self.stats.nonchrono_backjumps += 1;
            return BacktrackDecision::NonChronological {
                target_level: asserting_level,
            };
        }

        let distance = conflict_level.saturating_sub(asserting_level);

        // Check distance threshold
        if distance < self.config.min_distance {
            self.stats.nonchrono_backjumps += 1;
            self.record_decision(false, clause_lbd);
            return BacktrackDecision::NonChronological {
                target_level: asserting_level,
            };
        }

        // Check LBD threshold (only use chrono for good clauses)
        if clause_lbd > self.config.max_lbd {
            self.stats.nonchrono_backjumps += 1;
            self.record_decision(false, clause_lbd);
            return BacktrackDecision::NonChronological {
                target_level: asserting_level,
            };
        }

        // Check locality (if enabled)
        if self.config.enable_locality && self.should_use_locality() {
            self.stats.locality_triggers += 1;
            self.stats.chrono_backtracks += 1;
            self.stats.levels_saved += distance.saturating_sub(1) as u64;
            self.record_decision(true, clause_lbd);

            // Backtrack one level chronologically
            return BacktrackDecision::Chronological {
                target_level: conflict_level.saturating_sub(1),
            };
        }

        // Use chronological backtracking
        self.stats.distance_triggers += 1;
        self.stats.lbd_triggers += 1;
        self.stats.chrono_backtracks += 1;

        // Compute target level (not too far)
        let max_chrono_dist = self.config.max_backtrack_distance.min(distance);
        let target = conflict_level.saturating_sub(max_chrono_dist.min(distance));
        self.stats.levels_saved += (conflict_level - target) as u64;

        self.record_decision(true, clause_lbd);

        BacktrackDecision::Chronological {
            target_level: target.max(asserting_level),
        }
    }

    /// Check if locality heuristic suggests chronological backtracking.
    fn should_use_locality(&self) -> bool {
        if self.history.is_empty() {
            return false;
        }

        // Count recent chronological backtracks
        let recent = self
            .history
            .iter()
            .rev()
            .take(self.config.locality_window)
            .filter(|e| e.was_chrono)
            .count();

        let total = self.history.len().min(self.config.locality_window);

        // If >50% of recent backtracks were chronological, continue the trend
        recent * 2 > total
    }

    /// Record a backtrack decision in history.
    fn record_decision(&mut self, was_chrono: bool, lbd: u32) {
        self.history.push(BacktrackHistoryEntry { was_chrono, lbd });

        // Keep history bounded
        if self.history.len() > self.config.locality_window * 2 {
            self.history.drain(0..self.config.locality_window);
        }
    }

    /// Clear history (e.g., on restart).
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for ChronoBacktrackEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrono_config_default() {
        let config = ChronoBacktrackConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_distance, 10);
        assert_eq!(config.max_lbd, 6);
    }

    #[test]
    fn test_chrono_engine_creation() {
        let engine = ChronoBacktrackEngine::new();
        assert_eq!(engine.stats().chrono_backtracks, 0);
        assert_eq!(engine.stats().nonchrono_backjumps, 0);
    }

    #[test]
    fn test_disabled_always_nonchrono() {
        let config = ChronoBacktrackConfig {
            enabled: false,
            ..Default::default()
        };
        let mut engine = ChronoBacktrackEngine::with_config(config);

        let decision = engine.decide_backtrack(100, 50, 3);

        match decision {
            BacktrackDecision::NonChronological { target_level } => {
                assert_eq!(target_level, 50);
            }
            _ => panic!("Expected non-chronological backtrack"),
        }

        assert_eq!(engine.stats().nonchrono_backjumps, 1);
        assert_eq!(engine.stats().chrono_backtracks, 0);
    }

    #[test]
    fn test_small_distance_nonchrono() {
        let mut engine = ChronoBacktrackEngine::new();

        // Distance = 5, but min_distance = 10
        let decision = engine.decide_backtrack(20, 15, 3);

        match decision {
            BacktrackDecision::NonChronological { target_level } => {
                assert_eq!(target_level, 15);
            }
            _ => panic!("Expected non-chronological backtrack"),
        }
    }

    #[test]
    fn test_high_lbd_nonchrono() {
        let mut engine = ChronoBacktrackEngine::new();

        // High LBD (bad clause) -> non-chronological
        let decision = engine.decide_backtrack(100, 50, 20);

        match decision {
            BacktrackDecision::NonChronological { .. } => {}
            _ => panic!("Expected non-chronological backtrack for high LBD"),
        }
    }

    #[test]
    fn test_good_conditions_chrono() {
        let mut engine = ChronoBacktrackEngine::new();

        // Distance=50 (>= 10), LBD=3 (<= 6) -> chronological
        let decision = engine.decide_backtrack(100, 50, 3);

        if let BacktrackDecision::Chronological { target_level } = decision {
            // Should backtrack chronologically
            assert!(target_level >= 50);
            assert!(target_level < 100);
        }

        assert!(engine.stats().chrono_backtracks > 0);
    }

    #[test]
    fn test_levels_saved_tracking() {
        let mut engine = ChronoBacktrackEngine::new();

        let _decision = engine.decide_backtrack(100, 50, 3);

        // Should track levels saved
        if engine.stats().chrono_backtracks > 0 {
            assert!(engine.stats().levels_saved > 0);
        }
    }

    #[test]
    fn test_locality_tracking() {
        let mut engine = ChronoBacktrackEngine::new();

        // Make several chronological decisions to build locality
        for _ in 0..5 {
            engine.record_decision(true, 3);
        }

        assert!(engine.should_use_locality());

        engine.clear_history();
        assert!(!engine.should_use_locality());
    }

    #[test]
    fn test_history_management() {
        let config = ChronoBacktrackConfig {
            locality_window: 10,
            ..Default::default()
        };
        let mut engine = ChronoBacktrackEngine::with_config(config);

        // Add many entries
        for i in 0..50 {
            engine.record_decision(i % 2 == 0, (i % 10) as u32);
        }

        // History should be bounded
        assert!(engine.history.len() <= 20); // 2 * locality_window
    }

    #[test]
    fn test_stats_reset() {
        let mut engine = ChronoBacktrackEngine::new();

        engine.stats.chrono_backtracks = 50;
        engine.stats.levels_saved = 200;

        engine.reset_stats();

        assert_eq!(engine.stats().chrono_backtracks, 0);
        assert_eq!(engine.stats().levels_saved, 0);
    }
}
