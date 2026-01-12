//! Dynamic LBD (Literal Block Distance) Updates
//!
//! This module implements dynamic LBD updates for learned clauses during search.
//! LBD is a key metric for clause quality - clauses with lower LBD are more likely
//! to be useful for future conflicts.
//!
//! The dynamic LBD update strategy:
//! - Periodically recompute LBD of learned clauses
//! - If LBD decreases, the clause is becoming more "glue-like" and should be kept
//! - Update clause tier based on improved LBD
//!
//! Reference: "Glucose" SAT solver by Gilles Audemard and Laurent Simon

use crate::clause::{ClauseDatabase, ClauseId, ClauseTier};
use crate::literal::Lit;
use std::collections::HashSet;

/// Statistics for dynamic LBD updates
#[derive(Debug, Default, Clone)]
pub struct DynamicLbdStats {
    /// Number of LBD updates performed
    pub updates_performed: u64,
    /// Number of clauses with improved LBD
    pub lbd_improved: u64,
    /// Number of clauses promoted to higher tier due to LBD improvement
    pub tier_promotions: u64,
}

/// Dynamic LBD manager
pub struct DynamicLbdManager {
    /// Interval between LBD updates (in conflicts)
    update_interval: u64,
    /// Conflict counter
    conflicts: u64,
    /// Statistics
    stats: DynamicLbdStats,
    /// Temporary set for LBD computation
    seen_levels: HashSet<u32>,
}

impl DynamicLbdManager {
    /// Create a new dynamic LBD manager
    pub fn new(update_interval: u64) -> Self {
        Self {
            update_interval,
            conflicts: 0,
            stats: DynamicLbdStats::default(),
            seen_levels: HashSet::new(),
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &DynamicLbdStats {
        &self.stats
    }

    /// Increment conflict counter
    pub fn on_conflict(&mut self) {
        self.conflicts += 1;
    }

    /// Check if it's time to update LBDs
    #[must_use]
    pub fn should_update(&self) -> bool {
        self.conflicts.is_multiple_of(self.update_interval)
    }

    /// Compute LBD for a clause given variable decision levels
    ///
    /// LBD (Literal Block Distance) is the number of distinct decision levels
    /// in the clause. Lower LBD means the clause is more "glue-like" and
    /// connects more decision levels.
    fn compute_lbd(&mut self, lits: &[Lit], level: &[u32]) -> u32 {
        self.seen_levels.clear();

        for &lit in lits {
            let var_level = level[lit.var().index()];
            if var_level > 0 {
                self.seen_levels.insert(var_level);
            }
        }

        self.seen_levels.len() as u32
    }

    /// Update LBD for a single clause
    ///
    /// Returns true if the LBD was improved
    pub fn update_clause_lbd(
        &mut self,
        clause_id: ClauseId,
        clauses: &mut ClauseDatabase,
        level: &[u32],
    ) -> bool {
        let Some(clause) = clauses.get(clause_id) else {
            return false;
        };

        // Only update learned clauses
        if !clause.learned || clause.deleted {
            return false;
        }

        let old_lbd = clause.lbd;
        let lits: Vec<Lit> = clause.lits.iter().copied().collect();

        // Compute new LBD
        let new_lbd = self.compute_lbd(&lits, level);

        // Update if improved
        if new_lbd < old_lbd
            && let Some(clause) = clauses.get_mut(clause_id)
        {
            clause.lbd = new_lbd;

            // Promote to Core tier if LBD is very low (glue clause)
            if new_lbd <= 2 && clause.tier != ClauseTier::Core {
                clause.promote_to_core();
                self.stats.tier_promotions += 1;
            }
            // Promote from Local to Mid if LBD improved significantly
            else if new_lbd < old_lbd && clause.tier == ClauseTier::Local {
                clause.tier = ClauseTier::Mid;
                self.stats.tier_promotions += 1;
            }

            self.stats.lbd_improved += 1;
            return true;
        }

        false
    }

    /// Update LBDs for all learned clauses
    ///
    /// Returns the number of clauses with improved LBD
    pub fn update_all_lbds(&mut self, clauses: &mut ClauseDatabase, level: &[u32]) -> u64 {
        let mut improved = 0;

        // Collect clause IDs first to avoid borrow checker issues
        let clause_ids: Vec<ClauseId> = clauses
            .iter_ids()
            .filter(|&id| {
                if let Some(clause) = clauses.get(id) {
                    clause.learned && !clause.deleted
                } else {
                    false
                }
            })
            .collect();

        for clause_id in clause_ids {
            if self.update_clause_lbd(clause_id, clauses, level) {
                improved += 1;
            }
        }

        self.stats.updates_performed += 1;
        improved
    }

    /// Update LBDs if it's time to do so
    ///
    /// Returns the number of clauses with improved LBD, or 0 if not time to update
    pub fn maybe_update(&mut self, clauses: &mut ClauseDatabase, level: &[u32]) -> u64 {
        if self.should_update() {
            self.update_all_lbds(clauses, level)
        } else {
            0
        }
    }

    /// Reset conflict counter
    pub fn reset_conflicts(&mut self) {
        self.conflicts = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_dynamic_lbd_creation() {
        let manager = DynamicLbdManager::new(1000);
        assert_eq!(manager.update_interval, 1000);
        assert_eq!(manager.conflicts, 0);
    }

    #[test]
    fn test_conflict_counter() {
        let mut manager = DynamicLbdManager::new(100);
        manager.on_conflict();
        manager.on_conflict();
        assert_eq!(manager.conflicts, 2);
    }

    #[test]
    fn test_should_update() {
        let mut manager = DynamicLbdManager::new(10);
        assert!(manager.should_update()); // 0 % 10 == 0

        for _ in 0..9 {
            manager.on_conflict();
            assert!(!manager.should_update());
        }

        manager.on_conflict();
        assert!(manager.should_update()); // 10 % 10 == 0
    }

    #[test]
    fn test_compute_lbd() {
        let mut manager = DynamicLbdManager::new(100);
        let lits = vec![
            Lit::pos(Var::new(0)),
            Lit::neg(Var::new(1)),
            Lit::pos(Var::new(2)),
        ];
        let level = vec![0, 1, 2]; // var 0 at level 0, var 1 at level 1, var 2 at level 2

        let lbd = manager.compute_lbd(&lits, &level);
        // Should count levels 1 and 2 (level 0 is not counted)
        assert_eq!(lbd, 2);
    }

    #[test]
    fn test_stats() {
        let manager = DynamicLbdManager::new(100);
        let stats = manager.stats();
        assert_eq!(stats.updates_performed, 0);
        assert_eq!(stats.lbd_improved, 0);
        assert_eq!(stats.tier_promotions, 0);
    }

    #[test]
    fn test_reset_conflicts() {
        let mut manager = DynamicLbdManager::new(100);
        manager.on_conflict();
        manager.on_conflict();
        assert_eq!(manager.conflicts, 2);

        manager.reset_conflicts();
        assert_eq!(manager.conflicts, 0);
    }
}
