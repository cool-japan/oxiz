//! Learned clause size management
//!
//! This module implements adaptive management of learned clause sizes.
//! Modern SAT solvers limit the size of learned clauses to prevent learning
//! very long clauses that provide little benefit while consuming memory.
//!
//! References:
//! - "Clause Size Management in CDCL SAT Solvers"
//! - "Adaptive Learned Clause Minimization in SAT Solvers"

/// Statistics for clause size management
#[derive(Debug, Clone, Default)]
pub struct SizeManagerStats {
    /// Number of clauses rejected due to size
    pub size_rejected: usize,
    /// Total clauses considered
    pub total_considered: usize,
    /// Average size of accepted clauses
    pub avg_accepted_size: f64,
    /// Average size of rejected clauses
    pub avg_rejected_size: f64,
    /// Current size limit
    pub current_limit: usize,
}

impl SizeManagerStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Clause Size Manager Statistics:");
        println!("  Total considered: {}", self.total_considered);
        println!("  Size rejected: {}", self.size_rejected);
        if self.total_considered > 0 {
            let accept_rate = 100.0 * (self.total_considered - self.size_rejected) as f64
                / self.total_considered as f64;
            println!("  Acceptance rate: {:.1}%", accept_rate);
        }
        println!("  Avg accepted size: {:.1}", self.avg_accepted_size);
        println!("  Avg rejected size: {:.1}", self.avg_rejected_size);
        println!("  Current limit: {}", self.current_limit);
    }
}

/// Strategy for adjusting clause size limit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeAdjustmentStrategy {
    /// Fixed size limit
    Fixed,
    /// Geometric increase (start small, increase over time)
    Geometric,
    /// Adaptive based on solver performance
    Adaptive,
    /// Luby-like sequence
    Luby,
}

/// Clause size manager
///
/// Manages the maximum size of learned clauses to balance between
/// clause quality and memory usage.
#[derive(Debug)]
pub struct ClauseSizeManager {
    /// Current maximum size for learned clauses
    current_limit: usize,
    /// Minimum size limit
    min_limit: usize,
    /// Maximum size limit
    max_limit: usize,
    /// Adjustment strategy
    strategy: SizeAdjustmentStrategy,
    /// Geometric growth factor (for Geometric strategy)
    growth_factor: f64,
    /// Statistics
    stats: SizeManagerStats,
    /// Number of conflicts since last adjustment
    conflicts_since_adjustment: u64,
    /// Adjustment interval (in conflicts)
    adjustment_interval: u64,
    /// Running sum for average calculations
    accepted_size_sum: u64,
    /// Count of accepted clauses
    accepted_count: usize,
    /// Running sum for rejected clauses
    rejected_size_sum: u64,
}

impl Default for ClauseSizeManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ClauseSizeManager {
    /// Create a new clause size manager with default settings
    ///
    /// Defaults:
    /// - Initial limit: 30 literals
    /// - Min limit: 10 literals
    /// - Max limit: 100 literals
    /// - Strategy: Geometric
    #[must_use]
    pub fn new() -> Self {
        Self {
            current_limit: 30,
            min_limit: 10,
            max_limit: 100,
            strategy: SizeAdjustmentStrategy::Geometric,
            growth_factor: 1.1,
            stats: SizeManagerStats::default(),
            conflicts_since_adjustment: 0,
            adjustment_interval: 10000,
            accepted_size_sum: 0,
            accepted_count: 0,
            rejected_size_sum: 0,
        }
    }

    /// Create with custom limits
    #[must_use]
    pub fn with_limits(min: usize, initial: usize, max: usize) -> Self {
        Self {
            current_limit: initial,
            min_limit: min,
            max_limit: max,
            strategy: SizeAdjustmentStrategy::Geometric,
            growth_factor: 1.1,
            stats: SizeManagerStats::default(),
            conflicts_since_adjustment: 0,
            adjustment_interval: 10000,
            accepted_size_sum: 0,
            accepted_count: 0,
            rejected_size_sum: 0,
        }
    }

    /// Set the adjustment strategy
    pub fn set_strategy(&mut self, strategy: SizeAdjustmentStrategy) {
        self.strategy = strategy;
    }

    /// Check if a clause of given size should be learned
    ///
    /// Returns true if the clause size is within the current limit
    pub fn should_learn(&mut self, size: usize, lbd: u32) -> bool {
        self.stats.total_considered += 1;

        // Always learn very high quality clauses (LBD <= 2) regardless of size
        if lbd <= 2 {
            self.accepted_size_sum += size as u64;
            self.accepted_count += 1;
            return true;
        }

        // Check size limit
        if size <= self.current_limit {
            self.accepted_size_sum += size as u64;
            self.accepted_count += 1;
            true
        } else {
            self.stats.size_rejected += 1;
            self.rejected_size_sum += size as u64;
            false
        }
    }

    /// Adjust size limit based on conflicts and strategy
    pub fn adjust_limit(&mut self, conflicts: u64) {
        self.conflicts_since_adjustment += 1;

        if self.conflicts_since_adjustment < self.adjustment_interval {
            return;
        }

        self.conflicts_since_adjustment = 0;

        // Update stats
        if self.accepted_count > 0 {
            self.stats.avg_accepted_size =
                self.accepted_size_sum as f64 / self.accepted_count as f64;
        }
        if self.stats.size_rejected > 0 {
            self.stats.avg_rejected_size =
                self.rejected_size_sum as f64 / self.stats.size_rejected as f64;
        }
        self.stats.current_limit = self.current_limit;

        match self.strategy {
            SizeAdjustmentStrategy::Fixed => {
                // No adjustment
            }
            SizeAdjustmentStrategy::Geometric => {
                // Geometric increase
                let new_limit = (self.current_limit as f64 * self.growth_factor) as usize;
                self.current_limit = new_limit.min(self.max_limit);
            }
            SizeAdjustmentStrategy::Adaptive => {
                // Adjust based on rejection rate
                let rejection_rate = if self.stats.total_considered > 0 {
                    self.stats.size_rejected as f64 / self.stats.total_considered as f64
                } else {
                    0.0
                };

                if rejection_rate > 0.3 {
                    // Too many rejections, increase limit
                    self.current_limit = (self.current_limit + 5).min(self.max_limit);
                } else if rejection_rate < 0.1 {
                    // Few rejections, can decrease limit
                    self.current_limit = (self.current_limit.saturating_sub(2)).max(self.min_limit);
                }
            }
            SizeAdjustmentStrategy::Luby => {
                // Use Luby sequence for limit
                let luby_value = self.luby((conflicts / self.adjustment_interval) as u32);
                self.current_limit = (self.min_limit + luby_value as usize).min(self.max_limit);
            }
        }
    }

    /// Compute Luby sequence value
    /// Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    #[allow(clippy::only_used_in_recursion)]
    fn luby(&self, i: u32) -> u32 {
        let mut power = 1u32;

        // Find the largest power of 2 <= i+1
        while power * 2 <= i + 1 {
            power *= 2;
        }

        if power == i + 1 {
            // i+1 is a power of 2, return previous power (or 1 if power=1)
            if power == 1 { 1 } else { power / 2 }
        } else {
            // Recursively compute
            self.luby(i + 1 - power)
        }
    }

    /// Get current size limit
    #[must_use]
    pub fn current_limit(&self) -> usize {
        self.current_limit
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SizeManagerStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SizeManagerStats::default();
        self.stats.current_limit = self.current_limit;
        self.accepted_size_sum = 0;
        self.accepted_count = 0;
        self.rejected_size_sum = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_manager_creation() {
        let manager = ClauseSizeManager::new();
        assert_eq!(manager.current_limit(), 30);
    }

    #[test]
    fn test_should_learn_within_limit() {
        let mut manager = ClauseSizeManager::new();
        assert!(manager.should_learn(20, 5)); // Within limit
        assert!(manager.should_learn(30, 5)); // At limit
    }

    #[test]
    fn test_should_learn_exceeds_limit() {
        let mut manager = ClauseSizeManager::new();
        assert!(!manager.should_learn(50, 5)); // Exceeds limit
        assert_eq!(manager.stats().size_rejected, 1);
    }

    #[test]
    fn test_should_learn_high_quality() {
        let mut manager = ClauseSizeManager::new();
        // High quality clauses (LBD <= 2) always learned
        assert!(manager.should_learn(100, 2));
        assert_eq!(manager.stats().size_rejected, 0);
    }

    #[test]
    fn test_geometric_growth() {
        let mut manager = ClauseSizeManager::new();
        manager.set_strategy(SizeAdjustmentStrategy::Geometric);

        let initial = manager.current_limit();
        // Manually trigger adjustment by setting conflicts
        manager.conflicts_since_adjustment = manager.adjustment_interval;
        manager.adjust_limit(10000);

        assert!(manager.current_limit() > initial);
    }

    #[test]
    fn test_fixed_strategy() {
        let mut manager = ClauseSizeManager::new();
        manager.set_strategy(SizeAdjustmentStrategy::Fixed);

        let initial = manager.current_limit();
        manager.conflicts_since_adjustment = manager.adjustment_interval;
        manager.adjust_limit(10000);

        assert_eq!(manager.current_limit(), initial);
    }

    #[test]
    fn test_adaptive_increase() {
        let mut manager = ClauseSizeManager::new();
        manager.set_strategy(SizeAdjustmentStrategy::Adaptive);

        // Simulate high rejection rate
        for _ in 0..100 {
            manager.should_learn(50, 5); // Will be rejected
        }

        let initial = manager.current_limit();
        manager.conflicts_since_adjustment = manager.adjustment_interval;
        manager.adjust_limit(10000);

        assert!(manager.current_limit() > initial);
    }

    #[test]
    fn test_adaptive_decrease() {
        let mut manager = ClauseSizeManager::new();
        manager.set_strategy(SizeAdjustmentStrategy::Adaptive);

        // Simulate low rejection rate
        for _ in 0..100 {
            manager.should_learn(10, 5); // Will be accepted
        }

        let initial = manager.current_limit();
        manager.conflicts_since_adjustment = manager.adjustment_interval;
        manager.adjust_limit(10000);

        assert!(manager.current_limit() < initial);
    }

    #[test]
    fn test_luby_strategy_available() {
        let mut manager = ClauseSizeManager::new();
        manager.set_strategy(SizeAdjustmentStrategy::Luby);
        manager.conflicts_since_adjustment = manager.adjustment_interval;
        manager.adjust_limit(10000);
        // Just verify it doesn't crash
        assert!(manager.current_limit() > 0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut manager = ClauseSizeManager::new();

        manager.should_learn(20, 5);
        manager.should_learn(50, 5);
        manager.should_learn(15, 5);

        let stats = manager.stats();
        assert_eq!(stats.total_considered, 3);
        assert_eq!(stats.size_rejected, 1); // The 50-literal clause
    }
}
