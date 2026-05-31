//! Decision heuristics, phase saving, backtracking, and restarts

use super::*;

impl Solver {
    /// Pick next variable to branch on
    pub(super) fn pick_branch_var(&mut self) -> Option<Var> {
        // Try external branching heuristic first.
        if let Some(ref ext) = self.config.external_branching {
            let candidates: Vec<Var> = (0..self.num_vars)
                .map(|i| Var::new(i as u32))
                .filter(|&v| !self.trail.is_assigned(v))
                .collect();
            let scores: Vec<f64> = candidates.iter().map(|&v| self.vsids.activity(v)).collect();
            if let Ok(mut h) = ext.lock()
                && let Some(chosen) = h.select(&candidates, &scores)
            {
                return Some(chosen);
            }
        }

        if self.config.use_lrb_branching {
            // Use LRB branching
            while let Some(var) = self.lrb.select() {
                if !self.trail.is_assigned(var) {
                    self.lrb.on_assign(var);
                    return Some(var);
                }
            }
        } else if self.config.use_chb_branching {
            // Use CHB branching
            // Rebuild heap periodically to reflect score changes
            if self.stats.decisions.is_multiple_of(100) {
                self.chb.rebuild_heap();
            }

            while let Some(var) = self.chb.pop_max() {
                if !self.trail.is_assigned(var) {
                    return Some(var);
                }
            }
        } else {
            // Use VSIDS branching
            while let Some(var) = self.vsids.pop_max() {
                if !self.trail.is_assigned(var) {
                    return Some(var);
                }
            }
        }
        None
    }

    /// Backtrack with phase saving
    pub(super) fn backtrack_with_phase_saving(&mut self, level: u32) {
        // Collect variables that will be unassigned
        let mut unassigned_vars = Vec::new();

        // Save phases before backtracking
        let phase = &mut self.phase;
        let lrb = &mut self.lrb;
        self.trail.backtrack_to_with_callback(level, |lit| {
            let var = lit.var();
            if var.index() < phase.len() {
                phase[var.index()] = lit.is_pos();
            }
            // Re-insert variable into LRB heap
            lrb.unassign(var);
            unassigned_vars.push(var);
        });

        // Re-insert unassigned variables into VSIDS and CHB heaps
        for var in unassigned_vars {
            if !self.vsids.contains(var) {
                self.vsids.insert(var);
            }
            if !self.chb.contains(var) {
                self.chb.insert(var);
            }
        }
    }

    /// Backtrack to a given level without saving phases
    pub(super) fn backtrack(&mut self, level: u32) {
        self.trail.backtrack_to(level);
    }

    /// Compute the Luby sequence value for index i (1-indexed: luby(1)=1, luby(2)=1, ...)
    /// Sequence: 1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, ...
    /// For 0-indexed input, we add 1 internally.
    pub(super) fn luby(i: u64) -> u64 {
        let i = i + 1; // Convert to 1-indexed

        // Find k such that 2^k - 1 >= i
        let mut k = 1u32;
        while (1u64 << k) - 1 < i {
            k += 1;
        }

        let seq_len = (1u64 << k) - 1;

        if i == seq_len {
            // i is exactly 2^k - 1, return 2^(k-1)
            1u64 << (k - 1)
        } else {
            // Recurse: luby(i) = luby(i - (2^(k-1) - 1))
            // The sequence up to 2^k - 1 is: luby(1..2^(k-1)-1), luby(1..2^(k-1)-1), 2^(k-1)
            let half_len = (1u64 << (k - 1)) - 1;
            if i <= half_len {
                Self::luby(i - 1) // Already 0-indexed internally
            } else if i <= 2 * half_len {
                Self::luby(i - half_len - 1)
            } else {
                1u64 << (k - 1)
            }
        }
    }

    /// Restart
    pub(super) fn restart(&mut self) {
        self.stats.restarts += 1;
        self.backtrack_with_phase_saving(0);

        // Calculate next restart threshold based on strategy
        match self.config.restart_strategy {
            RestartStrategy::Luby => {
                self.luby_index += 1;
                self.restart_threshold = self.stats.conflicts
                    + Self::luby(self.luby_index) * self.config.restart_interval;
            }
            RestartStrategy::Geometric => {
                let current_interval = if self.restart_threshold > self.stats.conflicts {
                    self.restart_threshold - self.stats.conflicts
                } else {
                    self.config.restart_interval
                };
                let next_interval =
                    (current_interval as f64 * self.config.restart_multiplier) as u64;
                self.restart_threshold = self.stats.conflicts + next_interval;
            }
            RestartStrategy::Glucose => {
                // Glucose-style dynamic restarts based on LBD
                // Restart when recent average LBD is higher than global average
                // For now, use geometric with dynamic adjustment
                let current_interval = if self.restart_threshold > self.stats.conflicts {
                    self.restart_threshold - self.stats.conflicts
                } else {
                    self.config.restart_interval
                };

                // Adjust based on recent LBD trend
                let next_interval = if self.recent_lbd_count > 50 {
                    let recent_avg = self.recent_lbd_sum / self.recent_lbd_count.max(1);
                    // If recent LBD is low (good), increase interval; if high, decrease
                    if recent_avg < 5 {
                        // Good quality clauses - increase interval
                        ((current_interval as f64) * 1.1) as u64
                    } else {
                        // Poor quality clauses - decrease interval
                        ((current_interval as f64) * 0.9) as u64
                    }
                } else {
                    current_interval
                };

                self.restart_threshold = self.stats.conflicts + next_interval.max(100);
            }
            RestartStrategy::LocalLbd => {
                // Local restarts based on LBD
                // Check if we should do a local restart
                self.conflicts_since_local_restart += 1;

                if self.conflicts_since_local_restart >= 50 && self.should_local_restart() {
                    // Perform local restart - backtrack to a safe level, not to 0
                    let local_level = self.compute_local_restart_level();
                    self.backtrack_with_phase_saving(local_level);
                    self.conflicts_since_local_restart = 0;
                    // Reset recent LBD for next window
                    self.recent_lbd_sum = 0;
                    self.recent_lbd_count = 0;
                } else {
                    // Standard restart if too many conflicts
                    let current_interval = if self.restart_threshold > self.stats.conflicts {
                        self.restart_threshold - self.stats.conflicts
                    } else {
                        self.config.restart_interval
                    };
                    self.restart_threshold = self.stats.conflicts + current_interval;
                }
                return; // Don't do full backtrack to 0
            }
        }

        // Re-add all unassigned variables to VSIDS heap
        for i in 0..self.num_vars {
            let var = Var::new(i as u32);
            if !self.trail.is_assigned(var) && !self.vsids.contains(var) {
                self.vsids.insert(var);
            }
        }
    }

    /// Check if we should perform a local restart
    /// Returns true if recent average LBD is significantly higher than global average
    pub(super) fn should_local_restart(&self) -> bool {
        if self.recent_lbd_count < 50 || self.global_lbd_count < 100 {
            return false;
        }

        let recent_avg = self.recent_lbd_sum / self.recent_lbd_count.max(1);
        let global_avg = self.global_lbd_sum / self.global_lbd_count.max(1);

        // Local restart if recent average is 1.25x higher than global average
        recent_avg * 4 > global_avg * 5
    }

    /// Compute the level to backtrack to for local restart
    /// Use a level that preserves some of the search progress
    pub(super) fn compute_local_restart_level(&self) -> u32 {
        let current_level = self.trail.decision_level();

        // Backtrack to about 20% of current depth to preserve some work
        if current_level > 5 {
            current_level / 5
        } else {
            0
        }
    }

    /// Generate a random u64 using xorshift64
    pub(super) fn rand_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Generate a random f64 in [0, 1)
    pub(super) fn rand_f64(&mut self) -> f64 {
        const MAX: f64 = u64::MAX as f64;
        (self.rand_u64() as f64) / MAX
    }

    /// Generate a random boolean with given probability of being true
    pub(super) fn rand_bool(&mut self, probability: f64) -> bool {
        self.rand_f64() < probability
    }
}
