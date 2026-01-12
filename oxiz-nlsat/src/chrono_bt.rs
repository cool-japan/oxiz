//! Chronological backtracking support.
//!
//! Modern SAT solvers use chronological backtracking instead of always
//! backtracking to the assertion level. This can improve performance by
//! avoiding unnecessary work.
//!
//! Reference: "Chronological Backtracking" by Nadel & Ryvchin (SAT 2018)

use crate::clause::ClauseId;
use crate::types::Literal;

/// Configuration for chronological backtracking.
#[derive(Debug, Clone, Copy)]
pub struct ChronoConfig {
    /// Enable chronological backtracking.
    pub enabled: bool,
    /// Maximum backtrack distance for chronological BT.
    /// If the backjump would skip more than this many levels,
    /// use non-chronological backtracking instead.
    pub max_distance: u32,
}

impl Default for ChronoConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_distance: 100,
        }
    }
}

/// Statistics for chronological backtracking.
#[derive(Debug, Clone, Default)]
pub struct ChronoStats {
    /// Number of chronological backtracks.
    pub chrono_backtracks: u64,
    /// Number of non-chronological backtracks.
    pub non_chrono_backtracks: u64,
}

/// Backtrack decision for conflict resolution.
#[derive(Debug, Clone, Copy)]
pub enum BacktrackDecision {
    /// Chronological backtrack: backtrack one level at a time.
    Chronological { current_level: u32 },
    /// Non-chronological backtrack: jump directly to assertion level.
    NonChronological { target_level: u32 },
}

/// Chronological backtracking manager.
pub struct ChronoBacktracker {
    config: ChronoConfig,
    stats: ChronoStats,
}

impl ChronoBacktracker {
    /// Create a new chronological backtracker.
    pub fn new() -> Self {
        Self::with_config(ChronoConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: ChronoConfig) -> Self {
        Self {
            config,
            stats: ChronoStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ChronoStats {
        &self.stats
    }

    /// Get configuration.
    pub fn config(&self) -> &ChronoConfig {
        &self.config
    }

    /// Decide whether to use chronological or non-chronological backtracking.
    ///
    /// # Arguments
    /// * `current_level` - Current decision level
    /// * `assertion_level` - Level at which the learned clause becomes unit
    /// * `conflict_id` - The conflicting clause (if any)
    ///
    /// # Returns
    /// The backtrack decision.
    pub fn decide_backtrack(
        &mut self,
        current_level: u32,
        assertion_level: u32,
        _conflict_id: Option<ClauseId>,
    ) -> BacktrackDecision {
        if !self.config.enabled {
            // Always use non-chronological if disabled
            self.stats.non_chrono_backtracks += 1;
            return BacktrackDecision::NonChronological {
                target_level: assertion_level,
            };
        }

        // If we're at the assertion level or below, no backtracking needed
        if current_level <= assertion_level {
            return BacktrackDecision::NonChronological {
                target_level: assertion_level,
            };
        }

        let distance = current_level - assertion_level;

        // Use chronological backtracking if distance is small
        if distance <= self.config.max_distance {
            self.stats.chrono_backtracks += 1;
            BacktrackDecision::Chronological { current_level }
        } else {
            // Too far to backtrack chronologically - use non-chronological
            self.stats.non_chrono_backtracks += 1;
            BacktrackDecision::NonChronological {
                target_level: assertion_level,
            }
        }
    }

    /// Get the next backtrack level given the current decision.
    ///
    /// For chronological backtracking, this decrements by one level.
    /// For non-chronological, it returns the target level.
    pub fn next_level(&self, decision: BacktrackDecision) -> u32 {
        match decision {
            BacktrackDecision::Chronological { current_level } => current_level.saturating_sub(1),
            BacktrackDecision::NonChronological { target_level } => target_level,
        }
    }

    /// Check if we should continue chronological backtracking.
    ///
    /// Returns true if we should backtrack another level chronologically.
    pub fn should_continue(
        &self,
        decision: BacktrackDecision,
        current_level: u32,
        assertion_level: u32,
    ) -> bool {
        match decision {
            BacktrackDecision::Chronological { .. } => {
                // Continue until we reach the assertion level
                current_level > assertion_level
            }
            BacktrackDecision::NonChronological { .. } => {
                // Non-chronological is a single jump
                false
            }
        }
    }
}

impl Default for ChronoBacktracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Conflict analysis result with backtrack information.
#[derive(Debug, Clone)]
pub struct ConflictAnalysisResult {
    /// The learned clause.
    pub learned_clause: Vec<Literal>,
    /// The assertion level (level at which learned clause becomes unit).
    pub assertion_level: u32,
    /// The backtrack decision.
    pub backtrack_decision: BacktrackDecision,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chrono_backtracker_new() {
        let bt = ChronoBacktracker::new();
        assert!(bt.config().enabled);
        assert_eq!(bt.stats().chrono_backtracks, 0);
        assert_eq!(bt.stats().non_chrono_backtracks, 0);
    }

    #[test]
    fn test_decide_backtrack_short_distance() {
        let mut bt = ChronoBacktracker::new();

        let decision = bt.decide_backtrack(5, 3, None);

        match decision {
            BacktrackDecision::Chronological { current_level } => {
                assert_eq!(current_level, 5);
            }
            _ => panic!("Expected chronological backtrack"),
        }

        assert_eq!(bt.stats().chrono_backtracks, 1);
    }

    #[test]
    fn test_decide_backtrack_long_distance() {
        let mut bt = ChronoBacktracker::new();

        // Distance of 150 > max_distance (100)
        let decision = bt.decide_backtrack(200, 50, None);

        match decision {
            BacktrackDecision::NonChronological { target_level } => {
                assert_eq!(target_level, 50);
            }
            _ => panic!("Expected non-chronological backtrack"),
        }

        assert_eq!(bt.stats().non_chrono_backtracks, 1);
    }

    #[test]
    fn test_decide_backtrack_disabled() {
        let config = ChronoConfig {
            enabled: false,
            max_distance: 100,
        };
        let mut bt = ChronoBacktracker::with_config(config);

        let decision = bt.decide_backtrack(10, 5, None);

        match decision {
            BacktrackDecision::NonChronological { target_level } => {
                assert_eq!(target_level, 5);
            }
            _ => panic!("Expected non-chronological backtrack when disabled"),
        }
    }

    #[test]
    fn test_next_level_chronological() {
        let bt = ChronoBacktracker::new();

        let decision = BacktrackDecision::Chronological { current_level: 5 };
        assert_eq!(bt.next_level(decision), 4);
    }

    #[test]
    fn test_next_level_non_chronological() {
        let bt = ChronoBacktracker::new();

        let decision = BacktrackDecision::NonChronological { target_level: 3 };
        assert_eq!(bt.next_level(decision), 3);
    }

    #[test]
    fn test_should_continue_chronological() {
        let bt = ChronoBacktracker::new();

        let decision = BacktrackDecision::Chronological { current_level: 5 };

        assert!(bt.should_continue(decision, 5, 3));
        assert!(bt.should_continue(decision, 4, 3));
        assert!(!bt.should_continue(decision, 3, 3));
    }

    #[test]
    fn test_should_continue_non_chronological() {
        let bt = ChronoBacktracker::new();

        let decision = BacktrackDecision::NonChronological { target_level: 3 };

        assert!(!bt.should_continue(decision, 5, 3));
    }
}
