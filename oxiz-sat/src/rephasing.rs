//! Rephasing strategies for SAT solving
//!
//! Rephasing periodically resets or modifies phase saving information to help
//! the solver escape local minima. This module implements various rephasing
//! strategies used in modern SAT solvers like CaDiCaL and Kissat.
//!
//! References:
//! - "CaDiCaL, Kissat, Paracooba, Plingeling and Treengeling at SAT Race 2019"
//! - "Everything You Always Wanted to Know About Blocked Sets (But Were Afraid to Ask)"

/// Rephasing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RephasingStrategy {
    /// Use original polarity from formula (positive literals)
    Original,
    /// Invert all current phases
    Inverted,
    /// Assign random phases
    Random,
    /// Set all phases to false
    False,
    /// Set all phases to true
    True,
    /// Use phases from best assignment found so far
    Best,
    /// Use phases from local search or random walk
    Walk,
}

/// Statistics for rephasing operations
#[derive(Debug, Clone, Default)]
pub struct RephasingStats {
    /// Number of rephasing operations performed
    pub rephase_count: usize,
    /// Count of each strategy used
    pub strategy_counts: [usize; 7],
    /// Total conflicts at last rephase
    pub last_rephase_conflicts: u64,
}

impl RephasingStats {
    /// Display statistics
    pub fn display(&self) {
        println!("Rephasing Statistics:");
        println!("  Total rephases: {}", self.rephase_count);
        println!("  Strategy usage:");
        println!("    Original:  {}", self.strategy_counts[0]);
        println!("    Inverted:  {}", self.strategy_counts[1]);
        println!("    Random:    {}", self.strategy_counts[2]);
        println!("    False:     {}", self.strategy_counts[3]);
        println!("    True:      {}", self.strategy_counts[4]);
        println!("    Best:      {}", self.strategy_counts[5]);
        println!("    Walk:      {}", self.strategy_counts[6]);
    }
}

/// Rephasing manager
///
/// Manages periodic rephasing of variable polarities to help escape local minima
/// and improve overall solver performance.
#[derive(Debug)]
pub struct RephasingManager {
    /// Current rephasing interval (in conflicts)
    interval: u64,
    /// Initial interval
    initial_interval: u64,
    /// Interval multiplier for geometric increase
    interval_multiplier: f64,
    /// Maximum interval
    max_interval: u64,
    /// Best assignment found so far (for Best strategy)
    best_assignment: Vec<bool>,
    /// Statistics
    stats: RephasingStats,
    /// RNG state for random rephasing
    rng_state: u64,
}

impl Default for RephasingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RephasingManager {
    /// Create a new rephasing manager with default settings
    ///
    /// Default settings:
    /// - Initial interval: 1000 conflicts
    /// - Multiplier: 1.1 (geometric increase)
    /// - Max interval: 1_000_000 conflicts
    #[must_use]
    pub fn new() -> Self {
        Self {
            interval: 1000,
            initial_interval: 1000,
            interval_multiplier: 1.1,
            max_interval: 1_000_000,
            best_assignment: Vec::new(),
            stats: RephasingStats::default(),
            rng_state: 0x853c_49e6_748f_ea9b, // Random seed
        }
    }

    /// Create with custom interval settings
    #[must_use]
    pub fn with_interval(initial: u64, multiplier: f64, max: u64) -> Self {
        Self {
            interval: initial,
            initial_interval: initial,
            interval_multiplier: multiplier,
            max_interval: max,
            best_assignment: Vec::new(),
            stats: RephasingStats::default(),
            rng_state: 0x853c_49e6_748f_ea9b,
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &RephasingStats {
        &self.stats
    }

    /// Check if rephasing should occur at this conflict count
    #[must_use]
    pub fn should_rephase(&self, conflicts: u64) -> bool {
        if self.stats.last_rephase_conflicts == 0 {
            return conflicts >= self.interval;
        }
        conflicts - self.stats.last_rephase_conflicts >= self.interval
    }

    /// Update best assignment for Best strategy
    pub fn update_best_assignment(&mut self, assignment: &[bool]) {
        self.best_assignment = assignment.to_vec();
    }

    /// Perform rephasing with the given strategy
    ///
    /// Returns new phase values for all variables
    pub fn rephase(
        &mut self,
        strategy: RephasingStrategy,
        num_vars: usize,
        current_phases: &[bool],
        conflicts: u64,
    ) -> Vec<bool> {
        self.stats.rephase_count += 1;
        self.stats.last_rephase_conflicts = conflicts;

        // Update strategy count
        let strategy_idx = match strategy {
            RephasingStrategy::Original => 0,
            RephasingStrategy::Inverted => 1,
            RephasingStrategy::Random => 2,
            RephasingStrategy::False => 3,
            RephasingStrategy::True => 4,
            RephasingStrategy::Best => 5,
            RephasingStrategy::Walk => 6,
        };
        self.stats.strategy_counts[strategy_idx] += 1;

        // Increase interval for next rephasing
        self.interval =
            ((self.interval as f64 * self.interval_multiplier) as u64).min(self.max_interval);

        match strategy {
            RephasingStrategy::Original => {
                // Use positive polarity (false for all variables)
                vec![false; num_vars]
            }
            RephasingStrategy::Inverted => {
                // Invert all current phases
                current_phases.iter().map(|&p| !p).collect()
            }
            RephasingStrategy::Random => {
                // Random phases
                (0..num_vars).map(|_| self.random_bool()).collect()
            }
            RephasingStrategy::False => {
                // All false
                vec![false; num_vars]
            }
            RephasingStrategy::True => {
                // All true
                vec![true; num_vars]
            }
            RephasingStrategy::Best => {
                // Use best assignment if available, otherwise keep current
                if self.best_assignment.len() >= num_vars {
                    self.best_assignment[..num_vars].to_vec()
                } else {
                    current_phases.to_vec()
                }
            }
            RephasingStrategy::Walk => {
                // Random walk: mostly random with some bias toward current
                (0..num_vars)
                    .map(|i| {
                        if self.random_bool() {
                            // 50% keep current
                            current_phases.get(i).copied().unwrap_or(false)
                        } else {
                            // 50% random
                            self.random_bool()
                        }
                    })
                    .collect()
            }
        }
    }

    /// Get the next recommended rephasing strategy
    ///
    /// Uses a cyclic pattern: Original -> Inverted -> Random -> False -> Best
    #[must_use]
    pub fn next_strategy(&self) -> RephasingStrategy {
        let cycle = self.stats.rephase_count % 5;
        match cycle {
            0 => RephasingStrategy::Original,
            1 => RephasingStrategy::Inverted,
            2 => RephasingStrategy::Random,
            3 => RephasingStrategy::False,
            _ => RephasingStrategy::Best,
        }
    }

    /// Reset to initial interval
    pub fn reset(&mut self) {
        self.interval = self.initial_interval;
        self.stats = RephasingStats::default();
    }

    /// Simple xorshift64 random number generator
    fn random_u64(&mut self) -> u64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    /// Generate random boolean
    fn random_bool(&mut self) -> bool {
        (self.random_u64() & 1) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rephasing_manager_creation() {
        let manager = RephasingManager::new();
        assert_eq!(manager.stats.rephase_count, 0);
        assert_eq!(manager.interval, 1000);
    }

    #[test]
    fn test_should_rephase() {
        let manager = RephasingManager::new();
        assert!(!manager.should_rephase(500));
        assert!(manager.should_rephase(1000));
        assert!(manager.should_rephase(1500));
    }

    #[test]
    fn test_rephase_original() {
        let mut manager = RephasingManager::new();
        let current = vec![true, false, true, false];
        let result = manager.rephase(RephasingStrategy::Original, 4, &current, 1000);

        assert_eq!(result, vec![false, false, false, false]);
        assert_eq!(manager.stats.rephase_count, 1);
    }

    #[test]
    fn test_rephase_inverted() {
        let mut manager = RephasingManager::new();
        let current = vec![true, false, true, false];
        let result = manager.rephase(RephasingStrategy::Inverted, 4, &current, 1000);

        assert_eq!(result, vec![false, true, false, true]);
    }

    #[test]
    fn test_rephase_false() {
        let mut manager = RephasingManager::new();
        let current = vec![true, false, true, false];
        let result = manager.rephase(RephasingStrategy::False, 4, &current, 1000);

        assert_eq!(result, vec![false, false, false, false]);
    }

    #[test]
    fn test_rephase_true() {
        let mut manager = RephasingManager::new();
        let current = vec![true, false, true, false];
        let result = manager.rephase(RephasingStrategy::True, 4, &current, 1000);

        assert_eq!(result, vec![true, true, true, true]);
    }

    #[test]
    fn test_rephase_best() {
        let mut manager = RephasingManager::new();
        manager.update_best_assignment(&[true, true, false, false]);

        let current = vec![false, false, false, false];
        let result = manager.rephase(RephasingStrategy::Best, 4, &current, 1000);

        assert_eq!(result, vec![true, true, false, false]);
    }

    #[test]
    fn test_next_strategy() {
        let mut manager = RephasingManager::new();

        assert_eq!(manager.next_strategy(), RephasingStrategy::Original);
        manager.rephase(RephasingStrategy::Original, 1, &[false], 1000);

        assert_eq!(manager.next_strategy(), RephasingStrategy::Inverted);
        manager.rephase(RephasingStrategy::Inverted, 1, &[false], 2000);

        assert_eq!(manager.next_strategy(), RephasingStrategy::Random);
    }

    #[test]
    fn test_interval_increase() {
        let mut manager = RephasingManager::with_interval(100, 2.0, 10000);

        manager.rephase(RephasingStrategy::Original, 1, &[false], 100);
        assert_eq!(manager.interval, 200);

        manager.rephase(RephasingStrategy::Original, 1, &[false], 300);
        assert_eq!(manager.interval, 400);
    }

    #[test]
    fn test_stats_display() {
        let mut manager = RephasingManager::new();
        manager.rephase(RephasingStrategy::Original, 1, &[false], 1000);
        manager.rephase(RephasingStrategy::Inverted, 1, &[false], 2000);

        let stats = manager.stats();
        assert_eq!(stats.rephase_count, 2);
        assert_eq!(stats.strategy_counts[0], 1); // Original
        assert_eq!(stats.strategy_counts[1], 1); // Inverted
    }
}
