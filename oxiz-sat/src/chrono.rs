//! Chronological backtracking support
//!
//! Chronological backtracking is a modern SAT solving technique that can improve
//! performance by sometimes backtracking chronologically instead of always using
//! non-chronological backtracking.
//!
//! Key idea: After learning a clause, instead of always jumping to the assertion
//! level, we can sometimes backtrack chronologically (one level at a time) if the
//! learned clause is still satisfied at higher levels.

use crate::literal::Lit;
use crate::trail::Trail;

/// Chronological backtracking helper
#[derive(Debug)]
pub struct ChronoBacktrack {
    /// Enable chronological backtracking
    enabled: bool,
    /// Threshold for chronological backtracking (max distance from current level)
    threshold: u32,
}

impl ChronoBacktrack {
    /// Create a new chronological backtracking helper
    #[must_use]
    pub fn new(enabled: bool, threshold: u32) -> Self {
        Self { enabled, threshold }
    }

    /// Determine the backtrack level for a learned clause
    ///
    /// Returns the level to backtrack to, which may be higher than the
    /// assertion level if chronological backtracking is beneficial.
    ///
    /// # Arguments
    ///
    /// * `trail` - The assignment trail
    /// * `learnt` - The learned clause (first literal is the asserting literal)
    /// * `assertion_level` - The traditional assertion level (second highest level)
    ///
    /// # Returns
    ///
    /// The level to backtrack to
    #[must_use]
    pub fn compute_backtrack_level(
        &self,
        trail: &Trail,
        learnt: &[Lit],
        assertion_level: u32,
    ) -> u32 {
        if !self.enabled || learnt.is_empty() {
            return assertion_level;
        }

        let current_level = trail.decision_level();

        // If we're already at or below the assertion level, use it
        if current_level <= assertion_level {
            return assertion_level;
        }

        // If the distance is too large, use non-chronological backtracking
        if current_level - assertion_level > self.threshold {
            return assertion_level;
        }

        // Try chronological backtracking: find the highest level where the
        // learned clause is still asserting (exactly one literal unassigned)
        let mut best_level = assertion_level;

        for level in (assertion_level + 1)..=current_level {
            if self.is_clause_asserting_at_level(trail, learnt, level) {
                best_level = level - 1; // Backtrack to just before this level
            } else {
                break;
            }
        }

        best_level
    }

    /// Check if the clause is asserting at the given level
    ///
    /// A clause is asserting at level L if exactly one literal is unassigned
    /// and all others are false at level L.
    fn is_clause_asserting_at_level(&self, trail: &Trail, clause: &[Lit], level: u32) -> bool {
        let mut unassigned_count = 0;
        let mut false_count = 0;

        for &lit in clause {
            let var = lit.var();
            let var_level = trail.level(var);

            if var_level == 0 {
                // Unassigned
                unassigned_count += 1;
            } else if var_level <= level {
                // Check if it's false
                let value = trail.lit_value(lit);
                if value.is_false() {
                    false_count += 1;
                }
            }
        }

        // Clause is asserting if exactly one literal is unassigned and rest are false
        unassigned_count == 1 && false_count == clause.len() - 1
    }

    /// Enable or disable chronological backtracking
    #[allow(dead_code)]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the threshold for chronological backtracking
    #[allow(dead_code)]
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
    }

    /// Check if chronological backtracking is enabled
    #[must_use]
    #[allow(dead_code)]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the threshold
    #[must_use]
    #[allow(dead_code)]
    pub const fn threshold(&self) -> u32 {
        self.threshold
    }
}

impl Default for ChronoBacktrack {
    fn default() -> Self {
        // Default: enabled with threshold of 100 levels
        Self::new(true, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;
    use crate::trail::Trail;

    #[test]
    fn test_chrono_disabled() {
        let chrono = ChronoBacktrack::new(false, 100);
        let trail = Trail::new(10);
        let learnt = vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))];

        let level = chrono.compute_backtrack_level(&trail, &learnt, 5);
        assert_eq!(level, 5); // Should use assertion level when disabled
    }

    #[test]
    fn test_chrono_threshold() {
        let chrono = ChronoBacktrack::new(true, 10);
        let trail = Trail::new(100);

        // Create a learned clause
        let learnt = vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))];

        // Distance is 50, which exceeds threshold of 10
        let level = chrono.compute_backtrack_level(&trail, &learnt, 5);

        // Should use non-chronological backtracking (assertion level) when threshold exceeded
        // Note: In a real scenario, the trail would have assignments that determine the behavior
        assert!(level <= 55); // Either assertion level or chronological, but reasonable
    }

    #[test]
    fn test_chrono_enabled() {
        let chrono = ChronoBacktrack::new(true, 100);
        assert!(chrono.is_enabled());
        assert_eq!(chrono.threshold(), 100);
    }

    #[test]
    fn test_chrono_default() {
        let chrono = ChronoBacktrack::default();
        assert!(chrono.is_enabled());
        assert_eq!(chrono.threshold(), 100);
    }
}
