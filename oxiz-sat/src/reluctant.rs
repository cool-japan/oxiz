//! Reluctant Doubling Restart Strategy
//!
//! This module implements the reluctant doubling restart strategy, which
//! is an adaptive restart strategy that balances exploration and exploitation.
//!
//! The strategy works by:
//! - Doubling the restart interval each time
//! - But "reluctantly" - only increasing when no progress is being made
//! - Resetting to base interval when progress is detected
//!
//! This is more adaptive than fixed geometric or Luby sequences.
//!
//! Reference: "Adaptive Restart Strategies for Conflict Driven SAT Solvers"

/// Statistics for reluctant doubling
#[derive(Debug, Default, Clone)]
pub struct ReluctantStats {
    /// Number of restarts performed
    pub restarts: u64,
    /// Number of times interval was doubled
    pub doublings: u64,
    /// Number of times interval was reset
    pub resets: u64,
    /// Current interval value
    pub current_interval: u64,
}

/// Reluctant doubling restart manager
pub struct ReluctantDoubling {
    /// Base interval (starting point)
    base_interval: u64,
    /// Current interval
    current_interval: u64,
    /// Maximum interval (cap to prevent infinite waits)
    max_interval: u64,
    /// Conflict counter since last restart
    conflicts_since_restart: u64,
    /// LBD threshold for detecting progress
    lbd_threshold: f64,
    /// Recent average LBD (to detect progress)
    recent_avg_lbd: f64,
    /// Smoothing factor for LBD average
    lbd_smooth: f64,
    /// Statistics
    stats: ReluctantStats,
}

impl ReluctantDoubling {
    /// Create a new reluctant doubling manager
    pub fn new(base_interval: u64, max_interval: u64, lbd_threshold: f64) -> Self {
        Self {
            base_interval,
            current_interval: base_interval,
            max_interval,
            conflicts_since_restart: 0,
            lbd_threshold,
            recent_avg_lbd: 100.0, // Start high
            lbd_smooth: 0.9,
            stats: ReluctantStats {
                current_interval: base_interval,
                ..Default::default()
            },
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &ReluctantStats {
        &self.stats
    }

    /// Called on each conflict
    ///
    /// Updates LBD average and conflict counter
    pub fn on_conflict(&mut self, lbd: u32) {
        self.conflicts_since_restart += 1;

        // Update recent average LBD with exponential smoothing
        self.recent_avg_lbd =
            self.lbd_smooth * self.recent_avg_lbd + (1.0 - self.lbd_smooth) * lbd as f64;
    }

    /// Check if should restart
    ///
    /// Returns true if conflict count reached the interval
    #[must_use]
    pub fn should_restart(&self) -> bool {
        self.conflicts_since_restart >= self.current_interval
    }

    /// Called when a restart is performed
    ///
    /// Returns true if the interval was doubled (reluctant)
    pub fn on_restart(&mut self) -> bool {
        self.stats.restarts += 1;
        self.conflicts_since_restart = 0;

        // Check if we're making progress (LBD is decreasing)
        let making_progress = self.recent_avg_lbd < self.lbd_threshold;

        if making_progress {
            // Reset to base interval when making progress
            self.current_interval = self.base_interval;
            self.stats.resets += 1;
            self.stats.current_interval = self.current_interval;
            false
        } else {
            // Double interval when not making progress (reluctant)
            let old_interval = self.current_interval;
            self.current_interval = (self.current_interval * 2).min(self.max_interval);

            let doubled = self.current_interval > old_interval;
            if doubled {
                self.stats.doublings += 1;
            }
            self.stats.current_interval = self.current_interval;
            doubled
        }
    }

    /// Reset the restart strategy
    pub fn reset(&mut self) {
        self.current_interval = self.base_interval;
        self.conflicts_since_restart = 0;
        self.recent_avg_lbd = 100.0;
        self.stats.current_interval = self.current_interval;
    }

    /// Get current interval
    #[must_use]
    pub fn current_interval(&self) -> u64 {
        self.current_interval
    }

    /// Get recent average LBD
    #[must_use]
    pub fn recent_avg_lbd(&self) -> f64 {
        self.recent_avg_lbd
    }

    /// Set LBD threshold for progress detection
    pub fn set_lbd_threshold(&mut self, threshold: f64) {
        self.lbd_threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reluctant_creation() {
        let reluctant = ReluctantDoubling::new(100, 10000, 3.0);
        assert_eq!(reluctant.current_interval(), 100);
        assert_eq!(reluctant.base_interval, 100);
        assert_eq!(reluctant.max_interval, 10000);
    }

    #[test]
    fn test_should_restart() {
        let mut reluctant = ReluctantDoubling::new(10, 1000, 3.0);

        assert!(!reluctant.should_restart());

        for _ in 0..9 {
            reluctant.on_conflict(2);
            assert!(!reluctant.should_restart());
        }

        reluctant.on_conflict(2);
        assert!(reluctant.should_restart());
    }

    #[test]
    fn test_restart_with_progress() {
        let mut reluctant = ReluctantDoubling::new(100, 10000, 5.0);

        // Make some conflicts with good LBD (making progress)
        for _ in 0..100 {
            reluctant.on_conflict(2);
        }

        // Should restart and reset interval (not double)
        assert!(reluctant.should_restart());
        let doubled = reluctant.on_restart();
        assert!(!doubled); // Should not double because making progress
        assert_eq!(reluctant.current_interval(), 100); // Reset to base
    }

    #[test]
    fn test_restart_without_progress() {
        let mut reluctant = ReluctantDoubling::new(100, 10000, 3.0);

        // Make some conflicts with poor LBD (not making progress)
        for _ in 0..100 {
            reluctant.on_conflict(10);
        }

        // Should restart and double interval
        assert!(reluctant.should_restart());
        let doubled = reluctant.on_restart();
        assert!(doubled); // Should double because not making progress
        assert_eq!(reluctant.current_interval(), 200); // Doubled
    }

    #[test]
    fn test_max_interval_cap() {
        let mut reluctant = ReluctantDoubling::new(100, 500, 3.0);

        // Force multiple doublings with poor LBD
        for _ in 0..10 {
            for _ in 0..reluctant.current_interval() {
                reluctant.on_conflict(10);
            }
            reluctant.on_restart();
        }

        // Should cap at max_interval
        assert!(reluctant.current_interval() <= 500);
    }

    #[test]
    fn test_reset() {
        let mut reluctant = ReluctantDoubling::new(100, 10000, 3.0);

        for _ in 0..100 {
            reluctant.on_conflict(10);
        }
        reluctant.on_restart();

        assert_eq!(reluctant.current_interval(), 200); // Should have doubled

        reluctant.reset();
        assert_eq!(reluctant.current_interval(), 100); // Reset to base
        assert_eq!(reluctant.conflicts_since_restart, 0);
    }

    #[test]
    fn test_stats() {
        let mut reluctant = ReluctantDoubling::new(100, 10000, 3.0);

        for _ in 0..100 {
            reluctant.on_conflict(10);
        }
        reluctant.on_restart();

        let stats = reluctant.stats();
        assert_eq!(stats.restarts, 1);
        assert_eq!(stats.doublings, 1);
    }

    #[test]
    fn test_lbd_averaging() {
        let mut reluctant = ReluctantDoubling::new(100, 10000, 3.0);

        reluctant.on_conflict(10);
        reluctant.on_conflict(10);
        reluctant.on_conflict(10);

        // Should be close to 10 after several conflicts with LBD=10
        assert!(reluctant.recent_avg_lbd() < 100.0);
        assert!(reluctant.recent_avg_lbd() > 5.0);
    }
}
