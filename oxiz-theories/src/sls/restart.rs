//! Restart strategies for SLS: RestartStrategy and RestartManager.

// ============================================================================
// Restart Strategies
// ============================================================================

/// Restart strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RestartStrategy {
    /// No restarts
    None,
    /// Fixed number of flips per restart
    Fixed(u64),
    /// Geometric sequence (multiply by factor each restart)
    Geometric(u64, f64),
    /// Luby sequence (provably optimal for certain problems)
    Luby(u64),
    /// Glucose-like (based on progress)
    Glucose,
    /// Adaptive (based on conflict analysis)
    Adaptive,
}

impl Default for RestartStrategy {
    fn default() -> Self {
        RestartStrategy::Geometric(1000, 1.5)
    }
}

/// Restart manager
#[derive(Debug)]
pub struct RestartManager {
    /// Strategy
    strategy: RestartStrategy,
    /// Current restart threshold
    current_threshold: u64,
    /// Restart count
    restart_count: u32,
    /// Luby sequence state
    luby_index: u32,
    /// Base unit for Luby
    #[allow(dead_code)]
    luby_unit: u64,
    /// Recent conflict counts (for adaptive)
    recent_conflicts: Vec<u32>,
    /// LBD queue (for Glucose-like)
    lbd_queue: Vec<u32>,
}

impl RestartManager {
    /// Create a new restart manager
    pub fn new(strategy: RestartStrategy) -> Self {
        let (threshold, luby_unit) = match strategy {
            RestartStrategy::None => (u64::MAX, 1),
            RestartStrategy::Fixed(n) => (n, n),
            RestartStrategy::Geometric(base, _) => (base, base),
            RestartStrategy::Luby(unit) => (unit, unit),
            RestartStrategy::Glucose => (50, 50),
            RestartStrategy::Adaptive => (1000, 1000),
        };

        Self {
            strategy,
            current_threshold: threshold,
            restart_count: 0,
            luby_index: 1,
            luby_unit,
            recent_conflicts: Vec::new(),
            lbd_queue: Vec::new(),
        }
    }

    /// Check if should restart
    pub fn should_restart(&self, flips: u64) -> bool {
        flips >= self.current_threshold
    }

    /// Notify of a restart
    pub fn notify_restart(&mut self) {
        self.restart_count += 1;

        self.current_threshold = match self.strategy {
            RestartStrategy::None => u64::MAX,
            RestartStrategy::Fixed(n) => n,
            RestartStrategy::Geometric(base, factor) => {
                let mult = factor.powi(self.restart_count as i32);
                (base as f64 * mult) as u64
            }
            RestartStrategy::Luby(unit) => {
                let luby_val = self.luby(self.luby_index);
                self.luby_index += 1;
                unit * luby_val as u64
            }
            RestartStrategy::Glucose => {
                // Simple Glucose-like: check if recent LBDs are high
                if self.lbd_queue.len() >= 50 {
                    let avg: u32 = self.lbd_queue.iter().sum::<u32>() / 50;
                    if avg > 5 {
                        self.current_threshold / 2
                    } else {
                        self.current_threshold * 2
                    }
                } else {
                    self.current_threshold
                }
            }
            RestartStrategy::Adaptive => {
                // Based on conflict rate
                if self.recent_conflicts.len() >= 10 {
                    let sum: u32 = self.recent_conflicts.iter().sum();
                    let avg = sum / 10;
                    if avg > 100 {
                        self.current_threshold / 2
                    } else {
                        self.current_threshold * 2
                    }
                } else {
                    self.current_threshold
                }
            }
        };
    }

    /// Record conflict (for adaptive strategies)
    pub fn record_conflict(&mut self, conflicts: u32) {
        self.recent_conflicts.push(conflicts);
        if self.recent_conflicts.len() > 100 {
            self.recent_conflicts.remove(0);
        }
    }

    /// Record LBD (for Glucose-like)
    pub fn record_lbd(&mut self, lbd: u32) {
        self.lbd_queue.push(lbd);
        if self.lbd_queue.len() > 100 {
            self.lbd_queue.remove(0);
        }
    }

    /// Luby sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn luby(&self, i: u32) -> u32 {
        if i == 0 {
            return 1;
        }

        // Find k such that 2^k - 1 == i
        let mut k = 1u32;
        let mut power = 2u32;
        while power - 1 < i {
            k += 1;
            power *= 2;
        }

        if power - 1 == i {
            // i is 2^k - 1, return 2^(k-1)
            1u32 << (k - 1)
        } else {
            // Recurse
            self.luby(i - (power / 2) + 1)
        }
    }

    /// Get restart count
    pub fn count(&self) -> u32 {
        self.restart_count
    }

    /// Get current threshold
    pub fn threshold(&self) -> u64 {
        self.current_threshold
    }
}
