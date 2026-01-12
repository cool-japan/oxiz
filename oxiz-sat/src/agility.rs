//! Agility tracking for adaptive restarts
//!
//! Agility measures how often variable assignments flip between conflicts.
//! This metric is used in modern SAT solvers like Glucose to make adaptive
//! restart decisions. High agility indicates diverse exploration, while low
//! agility suggests the solver might be stuck in a local search area.
//!
//! References:
//! - "Refining Restarts Strategies for SAT and UNSAT" (Glucose)
//! - "Improving Glucose for Incremental SAT Solving"

/// Statistics for agility tracking
#[derive(Debug, Clone, Default)]
pub struct AgilityStats {
    /// Current agility value (0.0 to 1.0)
    pub current_agility: f64,
    /// Number of flips detected
    pub total_flips: u64,
    /// Number of assignments tracked
    pub total_assignments: u64,
    /// Minimum agility observed
    pub min_agility: f64,
    /// Maximum agility observed
    pub max_agility: f64,
}

impl AgilityStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Agility Statistics:");
        println!("  Current agility: {:.4}", self.current_agility);
        println!("  Total flips: {}", self.total_flips);
        println!("  Total assignments: {}", self.total_assignments);
        println!("  Min agility: {:.4}", self.min_agility);
        println!("  Max agility: {:.4}", self.max_agility);
        if self.total_assignments > 0 {
            let flip_rate = self.total_flips as f64 / self.total_assignments as f64;
            println!("  Overall flip rate: {:.4}", flip_rate);
        }
    }
}

/// Agility tracker
///
/// Tracks assignment flip rate using exponential moving average.
/// Agility close to 1.0 indicates high diversity (many flips),
/// while agility close to 0.0 indicates stability (few flips).
#[derive(Debug)]
pub struct AgilityTracker {
    /// Exponential moving average of flip rate
    agility: f64,
    /// Decay factor for exponential moving average (0.0 to 1.0)
    /// Higher values make agility more responsive to recent behavior
    decay: f64,
    /// Last assignment for each variable (to detect flips)
    last_assignment: Vec<Option<bool>>,
    /// Statistics
    stats: AgilityStats,
}

impl Default for AgilityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AgilityTracker {
    /// Create a new agility tracker with default decay
    ///
    /// Default decay is 0.9999, which gives a smoothly changing agility metric
    #[must_use]
    pub fn new() -> Self {
        Self {
            agility: 0.0,
            decay: 0.9999,
            last_assignment: Vec::new(),
            stats: AgilityStats {
                current_agility: 0.0,
                total_flips: 0,
                total_assignments: 0,
                min_agility: 1.0,
                max_agility: 0.0,
            },
        }
    }

    /// Create with custom decay factor
    ///
    /// Decay should be in range (0.0, 1.0):
    /// - Higher values (e.g., 0.9999) make agility change slowly
    /// - Lower values (e.g., 0.99) make agility more responsive
    #[must_use]
    pub fn with_decay(decay: f64) -> Self {
        Self {
            agility: 0.0,
            decay: decay.clamp(0.0, 1.0),
            last_assignment: Vec::new(),
            stats: AgilityStats {
                current_agility: 0.0,
                total_flips: 0,
                total_assignments: 0,
                min_agility: 1.0,
                max_agility: 0.0,
            },
        }
    }

    /// Resize for new number of variables
    pub fn resize(&mut self, num_vars: usize) {
        self.last_assignment.resize(num_vars, None);
    }

    /// Record a variable assignment
    ///
    /// Updates agility based on whether this assignment differs from the last
    pub fn record_assignment(&mut self, var: usize, value: bool) {
        // Ensure we have space
        if var >= self.last_assignment.len() {
            self.last_assignment.resize(var + 1, None);
        }

        self.stats.total_assignments += 1;

        // Check if this is a flip from last assignment
        let is_flip = match self.last_assignment[var] {
            Some(last_value) => last_value != value,
            None => false, // First assignment, not a flip
        };

        // Update exponential moving average
        // agility = decay * agility + (1 - decay) * flip_indicator
        // where flip_indicator is 1.0 for flip, 0.0 for no flip
        let flip_value = if is_flip { 1.0 } else { 0.0 };
        self.agility = self.decay * self.agility + (1.0 - self.decay) * flip_value;

        if is_flip {
            self.stats.total_flips += 1;
        }

        // Store current assignment
        self.last_assignment[var] = Some(value);

        // Update stats
        self.stats.current_agility = self.agility;
        self.stats.min_agility = self.stats.min_agility.min(self.agility);
        self.stats.max_agility = self.stats.max_agility.max(self.agility);
    }

    /// Get current agility value (0.0 to 1.0)
    ///
    /// Values closer to 1.0 indicate high flip rate (diverse exploration)
    /// Values closer to 0.0 indicate low flip rate (focused exploration)
    #[must_use]
    pub fn agility(&self) -> f64 {
        self.agility
    }

    /// Check if agility is high (above threshold)
    ///
    /// Typical threshold is 0.2-0.3 for restart decisions
    #[must_use]
    pub fn is_high(&self, threshold: f64) -> bool {
        self.agility > threshold
    }

    /// Check if agility is low (below threshold)
    #[must_use]
    pub fn is_low(&self, threshold: f64) -> bool {
        self.agility < threshold
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &AgilityStats {
        &self.stats
    }

    /// Reset agility to initial state
    pub fn reset(&mut self) {
        self.agility = 0.0;
        self.last_assignment.clear();
        self.stats = AgilityStats {
            current_agility: 0.0,
            total_flips: 0,
            total_assignments: 0,
            min_agility: 1.0,
            max_agility: 0.0,
        };
    }

    /// Clear assignment history (keeps agility value)
    pub fn clear_assignments(&mut self) {
        for assignment in &mut self.last_assignment {
            *assignment = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agility_tracker_creation() {
        let tracker = AgilityTracker::new();
        assert_eq!(tracker.agility(), 0.0);
        assert_eq!(tracker.decay, 0.9999);
    }

    #[test]
    fn test_custom_decay() {
        let tracker = AgilityTracker::with_decay(0.95);
        assert_eq!(tracker.decay, 0.95);
    }

    #[test]
    fn test_no_flips() {
        let mut tracker = AgilityTracker::with_decay(0.9);
        tracker.resize(2);

        // Assign same values repeatedly
        for _ in 0..10 {
            tracker.record_assignment(0, true);
            tracker.record_assignment(1, false);
        }

        // Agility should be very low (no flips after first assignments)
        assert!(tracker.agility() < 0.2);
        assert_eq!(tracker.stats().total_flips, 0);
    }

    #[test]
    fn test_all_flips() {
        let mut tracker = AgilityTracker::with_decay(0.5);
        tracker.resize(1);

        // First assignment (not a flip)
        tracker.record_assignment(0, true);
        assert_eq!(tracker.stats().total_flips, 0);

        // All subsequent assignments flip
        for i in 0..10 {
            let value = i % 2 == 0;
            tracker.record_assignment(0, value);
        }

        // Should have high agility due to flips
        assert!(tracker.agility() > 0.3);
        // First assignment is not a flip, so 10 assignments = 9 flips
        assert_eq!(tracker.stats().total_flips, 9);
    }

    #[test]
    fn test_mixed_flips() {
        let mut tracker = AgilityTracker::with_decay(0.8);
        tracker.resize(2);

        // var 0: flips frequently
        tracker.record_assignment(0, true);
        tracker.record_assignment(0, false);
        tracker.record_assignment(0, true);
        tracker.record_assignment(0, false);

        // var 1: stable
        tracker.record_assignment(1, true);
        tracker.record_assignment(1, true);
        tracker.record_assignment(1, true);

        // Should have moderate agility
        let agility = tracker.agility();
        assert!(agility > 0.0);
        assert!(agility < 1.0);
    }

    #[test]
    fn test_is_high_is_low() {
        let mut tracker = AgilityTracker::with_decay(0.9);
        tracker.resize(1);

        // Start with stable assignments
        tracker.record_assignment(0, true);
        for _ in 0..5 {
            tracker.record_assignment(0, true);
        }
        assert!(tracker.is_low(0.1));
        assert!(!tracker.is_high(0.1));
    }

    #[test]
    fn test_reset() {
        let mut tracker = AgilityTracker::new();
        tracker.resize(2);
        tracker.record_assignment(0, true);
        tracker.record_assignment(1, false);

        tracker.reset();

        assert_eq!(tracker.agility(), 0.0);
        assert_eq!(tracker.stats().total_assignments, 0);
        assert_eq!(tracker.last_assignment.len(), 0);
    }

    #[test]
    fn test_clear_assignments() {
        let mut tracker = AgilityTracker::new();
        tracker.resize(2);
        tracker.record_assignment(0, true);

        let agility_before = tracker.agility();
        tracker.clear_assignments();

        // Agility value preserved
        assert_eq!(tracker.agility(), agility_before);
        // But assignments cleared
        assert!(tracker.last_assignment.iter().all(|a| a.is_none()));
    }

    #[test]
    fn test_stats_display() {
        let mut tracker = AgilityTracker::new();
        tracker.resize(1);
        tracker.record_assignment(0, true);
        tracker.record_assignment(0, false);

        let stats = tracker.stats();
        assert_eq!(stats.total_assignments, 2);
        assert_eq!(stats.total_flips, 1);
    }
}
