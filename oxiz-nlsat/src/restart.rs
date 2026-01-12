//! Restart strategies for the NLSAT solver.
//!
//! Implements various restart strategies including:
//! - Luby sequence restarts
//! - Geometric restarts
//! - Glucose-style LBD-based restarts
//!
//! Reference: Modern SAT solvers like MiniSat, Glucose, and CaDiCaL.

/// Restart strategy for the solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RestartStrategy {
    /// Fixed interval restarts.
    Fixed {
        /// Conflicts between restarts.
        interval: u64,
    },
    /// Geometric sequence restarts (multiplier applied after each restart).
    Geometric {
        /// Initial conflict limit.
        initial: u64,
        /// Multiplier for geometric progression (e.g., 1.5).
        multiplier: f64,
    },
    /// Luby sequence restarts.
    Luby {
        /// Base unit for Luby sequence.
        unit: u64,
    },
    /// Glucose-style LBD-based restarts.
    Glucose {
        /// Size of LBD sliding window.
        window_size: usize,
        /// LBD threshold ratio (K factor).
        threshold_factor: f64,
    },
}

impl Default for RestartStrategy {
    fn default() -> Self {
        RestartStrategy::Luby { unit: 512 }
    }
}

/// Restart manager that tracks restart state.
#[derive(Debug, Clone)]
pub struct RestartManager {
    /// Current restart strategy.
    strategy: RestartStrategy,
    /// Conflicts since last restart.
    conflicts_since_restart: u64,
    /// Current restart limit.
    current_limit: u64,
    /// Number of restarts performed.
    restarts: u64,
    /// LBD history for Glucose strategy (circular buffer).
    lbd_history: Vec<f64>,
    /// Current position in LBD history.
    lbd_pos: usize,
    /// Sum of LBD values in history (for average calculation).
    lbd_sum: f64,
    /// Number of LBD samples recorded (for warm-up).
    lbd_samples: usize,
}

impl RestartManager {
    /// Create a new restart manager with the given strategy.
    pub fn new(strategy: RestartStrategy) -> Self {
        let (current_limit, lbd_history) = match strategy {
            RestartStrategy::Fixed { interval } => (interval, Vec::new()),
            RestartStrategy::Geometric { initial, .. } => (initial, Vec::new()),
            RestartStrategy::Luby { unit } => (luby_sequence(0) * unit, Vec::new()),
            RestartStrategy::Glucose { window_size, .. } => (u64::MAX, vec![0.0; window_size]),
        };

        Self {
            strategy,
            conflicts_since_restart: 0,
            current_limit,
            restarts: 0,
            lbd_history,
            lbd_pos: 0,
            lbd_sum: 0.0,
            lbd_samples: 0,
        }
    }

    /// Check if a restart should be performed.
    ///
    /// For Glucose strategy, provide the current LBD average.
    pub fn should_restart(&mut self, current_lbd: Option<f64>) -> bool {
        match self.strategy {
            RestartStrategy::Fixed { .. }
            | RestartStrategy::Geometric { .. }
            | RestartStrategy::Luby { .. } => self.conflicts_since_restart >= self.current_limit,
            RestartStrategy::Glucose {
                window_size,
                threshold_factor,
            } => {
                if let Some(lbd) = current_lbd {
                    // Update LBD history
                    self.lbd_sum -= self.lbd_history[self.lbd_pos];
                    self.lbd_history[self.lbd_pos] = lbd;
                    self.lbd_sum += lbd;
                    self.lbd_pos = (self.lbd_pos + 1) % window_size;
                    self.lbd_samples += 1;

                    // Check if current LBD exceeds threshold
                    // Only restart if we have filled the window (warm-up period)
                    if self.lbd_samples >= window_size {
                        let avg_lbd = self.lbd_sum / (window_size as f64);
                        lbd > avg_lbd * threshold_factor
                    } else {
                        false // No restart during warm-up
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Record a conflict.
    pub fn record_conflict(&mut self) {
        self.conflicts_since_restart += 1;
    }

    /// Perform a restart and update internal state.
    pub fn restart(&mut self) {
        self.restarts += 1;
        self.conflicts_since_restart = 0;

        // Update limit for next restart
        match self.strategy {
            RestartStrategy::Fixed { .. } => {
                // Limit stays the same
            }
            RestartStrategy::Geometric {
                initial,
                multiplier,
            } => {
                self.current_limit =
                    (initial as f64 * multiplier.powi(self.restarts as i32)) as u64;
            }
            RestartStrategy::Luby { unit } => {
                self.current_limit = luby_sequence(self.restarts) * unit;
            }
            RestartStrategy::Glucose { .. } => {
                // Glucose doesn't use a conflict limit
            }
        }
    }

    /// Get the number of restarts performed.
    pub fn num_restarts(&self) -> u64 {
        self.restarts
    }

    /// Get the current restart limit.
    pub fn current_limit(&self) -> u64 {
        self.current_limit
    }
}

/// Compute the i-th element of the Luby sequence (0-indexed).
///
/// The Luby sequence is: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
///
/// Reference: "Optimal Speedup of Las Vegas Algorithms" by Luby et al.
fn luby_sequence(i: u64) -> u64 {
    // Convert to 1-indexed
    let n = i + 1;

    // Find k such that 2^(k-1) <= n < 2^k
    let k = 64 - (n.leading_zeros() as u64);

    // Check if n == 2^k - 1
    let power_k = 1u64 << k;
    if n == power_k - 1 {
        return 1u64 << (k - 1);
    }

    // Recurse: luby(n - 2^(k-1) + 1), then convert back to 0-indexed
    let power_k_minus_1 = 1u64 << (k - 1);
    luby_sequence(n - power_k_minus_1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luby_sequence() {
        // First 15 elements: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8
        let expected = [1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(luby_sequence(i as u64), exp, "Luby[{}] failed", i);
        }
    }

    #[test]
    fn test_fixed_restart() {
        let mut manager = RestartManager::new(RestartStrategy::Fixed { interval: 100 });

        for i in 0..100 {
            manager.record_conflict();
            if i < 99 {
                assert!(!manager.should_restart(None));
            } else {
                assert!(manager.should_restart(None));
            }
        }

        manager.restart();
        assert_eq!(manager.num_restarts(), 1);
        assert_eq!(manager.conflicts_since_restart, 0);
    }

    #[test]
    fn test_geometric_restart() {
        let mut manager = RestartManager::new(RestartStrategy::Geometric {
            initial: 100,
            multiplier: 1.5,
        });

        assert_eq!(manager.current_limit(), 100);

        for _ in 0..100 {
            manager.record_conflict();
        }
        assert!(manager.should_restart(None));

        manager.restart();
        assert_eq!(manager.current_limit(), 150); // 100 * 1.5
    }

    #[test]
    fn test_luby_restart() {
        let mut manager = RestartManager::new(RestartStrategy::Luby { unit: 10 });

        // First restart at 1*10 = 10
        assert_eq!(manager.current_limit(), 10);

        for _ in 0..10 {
            manager.record_conflict();
        }
        assert!(manager.should_restart(None));

        manager.restart();
        // Second restart at 1*10 = 10
        assert_eq!(manager.current_limit(), 10);

        for _ in 0..10 {
            manager.record_conflict();
        }
        assert!(manager.should_restart(None));

        manager.restart();
        // Third restart at 2*10 = 20
        assert_eq!(manager.current_limit(), 20);
    }

    #[test]
    fn test_glucose_restart() {
        let mut manager = RestartManager::new(RestartStrategy::Glucose {
            window_size: 5,
            threshold_factor: 1.25,
        });

        // Fill history with LBD values
        for lbd in [2.0, 2.5, 3.0, 2.5, 2.0] {
            assert!(!manager.should_restart(Some(lbd)));
        }

        // Average is 2.4, threshold is 2.4 * 1.25 = 3.0
        // LBD > 3.0 should trigger restart
        assert!(manager.should_restart(Some(3.5)));
        assert!(!manager.should_restart(Some(2.0)));
    }

    #[test]
    fn test_restart_manager_default() {
        let manager = RestartManager::new(RestartStrategy::default());
        // Default is Luby with unit 512
        assert_eq!(manager.current_limit(), 512); // luby(0) = 1
    }
}
