//! Smoothed LBD (Literal Block Distance) tracking
//!
//! This module implements exponential moving average tracking for LBD values.
//! Smoothed LBD helps detect trends in clause quality over time and can be
//! used for adaptive restart and clause deletion decisions.
//!
//! References:
//! - "Glucose: Combining Fast Local Search and Heuristics in the SAT Case"
//! - "Predicting Learnt Clauses Quality in Modern SAT Solvers"

/// Statistics for smoothed LBD tracking
#[derive(Debug, Clone, Default)]
pub struct SmoothedLbdStats {
    /// Current smoothed LBD (fast-moving average)
    pub fast_lbd: f64,
    /// Slow-moving average for trend detection
    pub slow_lbd: f64,
    /// Minimum LBD observed
    pub min_lbd: u32,
    /// Maximum LBD observed
    pub max_lbd: u32,
    /// Total clauses tracked
    pub total_clauses: u64,
    /// Number of trend changes detected
    pub trend_changes: usize,
}

impl SmoothedLbdStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Smoothed LBD Statistics:");
        println!("  Fast LBD (EMA): {:.2}", self.fast_lbd);
        println!("  Slow LBD (EMA): {:.2}", self.slow_lbd);
        println!("  Min LBD observed: {}", self.min_lbd);
        println!("  Max LBD observed: {}", self.max_lbd);
        println!("  Total clauses: {}", self.total_clauses);
        println!("  Trend changes: {}", self.trend_changes);
    }
}

/// Smoothed LBD tracker
///
/// Maintains exponential moving averages of LBD values with two different
/// decay rates (fast and slow) to detect quality trends.
#[derive(Debug)]
pub struct SmoothedLbdTracker {
    /// Fast exponential moving average (responsive to recent changes)
    fast_ema: f64,
    /// Slow exponential moving average (stable long-term trend)
    slow_ema: f64,
    /// Decay factor for fast EMA (higher = more weight on recent values)
    fast_decay: f64,
    /// Decay factor for slow EMA (lower = more stable)
    slow_decay: f64,
    /// Previous trend direction (for change detection)
    prev_trend: TrendDirection,
    /// Statistics
    stats: SmoothedLbdStats,
}

/// Trend direction for LBD quality
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrendDirection {
    /// Quality improving (LBD decreasing)
    Improving,
    /// Quality degrading (LBD increasing)
    Degrading,
    /// Neutral (no clear trend)
    Neutral,
}

impl Default for SmoothedLbdTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SmoothedLbdTracker {
    /// Create a new smoothed LBD tracker with default settings
    ///
    /// Default settings:
    /// - Fast decay: 0.9 (responsive to recent changes)
    /// - Slow decay: 0.99 (stable long-term trend)
    #[must_use]
    pub fn new() -> Self {
        Self {
            fast_ema: 0.0,
            slow_ema: 0.0,
            fast_decay: 0.9,
            slow_decay: 0.99,
            prev_trend: TrendDirection::Neutral,
            stats: SmoothedLbdStats {
                fast_lbd: 0.0,
                slow_lbd: 0.0,
                min_lbd: u32::MAX,
                max_lbd: 0,
                total_clauses: 0,
                trend_changes: 0,
            },
        }
    }

    /// Create with custom decay factors
    ///
    /// Fast decay should be lower than slow decay (e.g., 0.9 vs 0.99)
    /// Both should be in range (0.0, 1.0)
    #[must_use]
    pub fn with_decay(fast_decay: f64, slow_decay: f64) -> Self {
        Self {
            fast_ema: 0.0,
            slow_ema: 0.0,
            fast_decay: fast_decay.clamp(0.0, 1.0),
            slow_decay: slow_decay.clamp(0.0, 1.0),
            prev_trend: TrendDirection::Neutral,
            stats: SmoothedLbdStats {
                fast_lbd: 0.0,
                slow_lbd: 0.0,
                min_lbd: u32::MAX,
                max_lbd: 0,
                total_clauses: 0,
                trend_changes: 0,
            },
        }
    }

    /// Record a new LBD value
    ///
    /// Updates both fast and slow exponential moving averages
    pub fn record_lbd(&mut self, lbd: u32) {
        let lbd_f64 = lbd as f64;

        // Initialize EMAs on first value
        if self.stats.total_clauses == 0 {
            self.fast_ema = lbd_f64;
            self.slow_ema = lbd_f64;
        } else {
            // Update fast EMA
            self.fast_ema = self.fast_decay * self.fast_ema + (1.0 - self.fast_decay) * lbd_f64;

            // Update slow EMA
            self.slow_ema = self.slow_decay * self.slow_ema + (1.0 - self.slow_decay) * lbd_f64;
        }

        // Update stats
        self.stats.total_clauses += 1;
        self.stats.fast_lbd = self.fast_ema;
        self.stats.slow_lbd = self.slow_ema;
        self.stats.min_lbd = self.stats.min_lbd.min(lbd);
        self.stats.max_lbd = self.stats.max_lbd.max(lbd);

        // Detect trend changes
        let current_trend = self.current_trend();
        if current_trend != self.prev_trend && current_trend != TrendDirection::Neutral {
            self.stats.trend_changes += 1;
        }
        self.prev_trend = current_trend;
    }

    /// Get current fast-moving average
    #[must_use]
    pub fn fast_ema(&self) -> f64 {
        self.fast_ema
    }

    /// Get current slow-moving average
    #[must_use]
    pub fn slow_ema(&self) -> f64 {
        self.slow_ema
    }

    /// Determine current trend based on fast vs slow EMA
    ///
    /// - If fast < slow: Quality improving (LBD decreasing)
    /// - If fast > slow: Quality degrading (LBD increasing)
    /// - If close: Neutral
    #[must_use]
    fn current_trend(&self) -> TrendDirection {
        let diff = self.fast_ema - self.slow_ema;
        let threshold = 0.5; // Minimum difference to detect trend

        if diff < -threshold {
            TrendDirection::Improving
        } else if diff > threshold {
            TrendDirection::Degrading
        } else {
            TrendDirection::Neutral
        }
    }

    /// Check if quality is improving
    ///
    /// Returns true if fast EMA is significantly lower than slow EMA
    #[must_use]
    pub fn is_improving(&self) -> bool {
        self.current_trend() == TrendDirection::Improving
    }

    /// Check if quality is degrading
    ///
    /// Returns true if fast EMA is significantly higher than slow EMA
    #[must_use]
    pub fn is_degrading(&self) -> bool {
        self.current_trend() == TrendDirection::Degrading
    }

    /// Check if quality is stable
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.current_trend() == TrendDirection::Neutral
    }

    /// Get the divergence between fast and slow EMAs
    ///
    /// Positive value means degrading quality, negative means improving
    #[must_use]
    pub fn divergence(&self) -> f64 {
        self.fast_ema - self.slow_ema
    }

    /// Check if current LBD is better than average
    #[must_use]
    pub fn is_above_average(&self, lbd: u32) -> bool {
        (lbd as f64) < self.fast_ema
    }

    /// Check if current LBD is worse than average
    #[must_use]
    pub fn is_below_average(&self, lbd: u32) -> bool {
        (lbd as f64) > self.fast_ema
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SmoothedLbdStats {
        &self.stats
    }

    /// Reset tracker to initial state
    pub fn reset(&mut self) {
        self.fast_ema = 0.0;
        self.slow_ema = 0.0;
        self.prev_trend = TrendDirection::Neutral;
        self.stats = SmoothedLbdStats {
            fast_lbd: 0.0,
            slow_lbd: 0.0,
            min_lbd: u32::MAX,
            max_lbd: 0,
            total_clauses: 0,
            trend_changes: 0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothed_lbd_tracker_creation() {
        let tracker = SmoothedLbdTracker::new();
        assert_eq!(tracker.fast_ema(), 0.0);
        assert_eq!(tracker.slow_ema(), 0.0);
    }

    #[test]
    fn test_custom_decay() {
        let tracker = SmoothedLbdTracker::with_decay(0.8, 0.95);
        assert_eq!(tracker.fast_decay, 0.8);
        assert_eq!(tracker.slow_decay, 0.95);
    }

    #[test]
    fn test_first_value_initialization() {
        let mut tracker = SmoothedLbdTracker::new();
        tracker.record_lbd(5);

        assert_eq!(tracker.fast_ema(), 5.0);
        assert_eq!(tracker.slow_ema(), 5.0);
        assert_eq!(tracker.stats().min_lbd, 5);
        assert_eq!(tracker.stats().max_lbd, 5);
    }

    #[test]
    fn test_ema_updates() {
        let mut tracker = SmoothedLbdTracker::with_decay(0.5, 0.8);

        tracker.record_lbd(10);
        assert_eq!(tracker.fast_ema(), 10.0);

        tracker.record_lbd(20);
        // Fast EMA: 0.5 * 10 + 0.5 * 20 = 15
        assert_eq!(tracker.fast_ema(), 15.0);
        // Slow EMA: 0.8 * 10 + 0.2 * 20 = 12
        assert_eq!(tracker.slow_ema(), 12.0);
    }

    #[test]
    fn test_improving_trend() {
        let mut tracker = SmoothedLbdTracker::with_decay(0.5, 0.9);

        // Start with high LBD
        tracker.record_lbd(20);
        tracker.record_lbd(20);

        // Then improve (lower LBD)
        for _ in 0..5 {
            tracker.record_lbd(5);
        }

        // Fast EMA should be significantly lower than slow EMA
        assert!(tracker.is_improving());
        assert!(!tracker.is_degrading());
    }

    #[test]
    fn test_degrading_trend() {
        let mut tracker = SmoothedLbdTracker::with_decay(0.5, 0.9);

        // Start with low LBD
        tracker.record_lbd(5);
        tracker.record_lbd(5);

        // Then degrade (higher LBD)
        for _ in 0..5 {
            tracker.record_lbd(20);
        }

        // Fast EMA should be significantly higher than slow EMA
        assert!(tracker.is_degrading());
        assert!(!tracker.is_improving());
    }

    #[test]
    fn test_stable_trend() {
        let mut tracker = SmoothedLbdTracker::new();

        // Record same value multiple times
        for _ in 0..10 {
            tracker.record_lbd(10);
        }

        // Should be stable
        assert!(tracker.is_stable());
    }

    #[test]
    fn test_divergence() {
        let mut tracker = SmoothedLbdTracker::with_decay(0.3, 0.9);

        tracker.record_lbd(10);
        for _ in 0..5 {
            tracker.record_lbd(20);
        }

        // Fast should be higher than slow (degrading)
        assert!(tracker.divergence() > 0.0);
    }

    #[test]
    fn test_is_above_average() {
        let mut tracker = SmoothedLbdTracker::new();

        tracker.record_lbd(10);
        tracker.record_lbd(10);

        // LBD of 5 is better than average (10)
        assert!(tracker.is_above_average(5));
        assert!(!tracker.is_above_average(15));
    }

    #[test]
    fn test_is_below_average() {
        let mut tracker = SmoothedLbdTracker::new();

        tracker.record_lbd(10);
        tracker.record_lbd(10);

        // LBD of 15 is worse than average (10)
        assert!(tracker.is_below_average(15));
        assert!(!tracker.is_below_average(5));
    }

    #[test]
    fn test_min_max_tracking() {
        let mut tracker = SmoothedLbdTracker::new();

        tracker.record_lbd(10);
        tracker.record_lbd(5);
        tracker.record_lbd(20);
        tracker.record_lbd(15);

        assert_eq!(tracker.stats().min_lbd, 5);
        assert_eq!(tracker.stats().max_lbd, 20);
    }

    #[test]
    fn test_reset() {
        let mut tracker = SmoothedLbdTracker::new();

        tracker.record_lbd(10);
        tracker.record_lbd(20);

        tracker.reset();

        assert_eq!(tracker.fast_ema(), 0.0);
        assert_eq!(tracker.slow_ema(), 0.0);
        assert_eq!(tracker.stats().total_clauses, 0);
    }

    #[test]
    fn test_trend_change_detection() {
        let mut tracker = SmoothedLbdTracker::with_decay(0.3, 0.9);

        // Start low
        for _ in 0..3 {
            tracker.record_lbd(5);
        }

        // Go high (trend change to degrading)
        for _ in 0..5 {
            tracker.record_lbd(20);
        }

        // Go low again (trend change to improving)
        for _ in 0..5 {
            tracker.record_lbd(5);
        }

        // Should have detected at least one trend change
        assert!(tracker.stats().trend_changes > 0);
    }
}
