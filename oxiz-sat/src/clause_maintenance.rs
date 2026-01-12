//! Advanced clause database maintenance and cleaning
//!
//! This module provides utilities for maintaining clause quality through:
//! - Periodic clause cleaning (duplicate removal, normalization)
//! - Advanced reduction strategies
//! - Clause tier optimization
//! - Memory compaction

use crate::clause::{ClauseDatabase, ClauseId, ClauseTier};
use crate::literal::LBool;

/// Statistics for clause maintenance operations
#[derive(Debug, Clone, Default)]
pub struct MaintenanceStats {
    /// Number of duplicates removed
    pub duplicates_removed: usize,
    /// Number of tautologies detected
    pub tautologies_removed: usize,
    /// Number of clauses strengthened
    pub clauses_strengthened: usize,
    /// Number of tier promotions
    pub tier_promotions: usize,
    /// Number of tier demotions
    pub tier_demotions: usize,
    /// Total maintenance operations
    pub operations: usize,
}

impl MaintenanceStats {
    /// Display maintenance statistics
    pub fn display(&self) {
        println!("Clause Maintenance Statistics:");
        println!("  Operations: {}", self.operations);
        println!("  Duplicates removed: {}", self.duplicates_removed);
        println!("  Tautologies removed: {}", self.tautologies_removed);
        println!("  Clauses strengthened: {}", self.clauses_strengthened);
        println!("  Tier promotions: {}", self.tier_promotions);
        println!("  Tier demotions: {}", self.tier_demotions);
    }
}

/// Clause maintenance manager
#[derive(Debug)]
pub struct ClauseMaintenance {
    /// Statistics
    stats: MaintenanceStats,
    /// Clause IDs to clean
    cleanup_queue: Vec<ClauseId>,
}

impl Default for ClauseMaintenance {
    fn default() -> Self {
        Self::new()
    }
}

impl ClauseMaintenance {
    /// Create a new clause maintenance manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            stats: MaintenanceStats::default(),
            cleanup_queue: Vec::new(),
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &MaintenanceStats {
        &self.stats
    }

    /// Queue a clause for cleaning
    pub fn queue_for_cleanup(&mut self, clause_id: ClauseId) {
        self.cleanup_queue.push(clause_id);
    }

    /// Perform periodic maintenance on the clause database
    ///
    /// This performs various cleaning operations:
    /// - Remove duplicate literals
    /// - Detect and remove tautologies
    /// - Normalize clause representation
    /// - Update tier assignments based on usage
    pub fn periodic_maintenance(
        &mut self,
        clauses: &mut ClauseDatabase,
        assignments: &[LBool],
    ) -> Vec<ClauseId> {
        self.stats.operations += 1;
        let mut removed_clauses = Vec::new();

        // Process cleanup queue
        let queue: Vec<_> = self.cleanup_queue.drain(..).collect();
        for clause_id in queue {
            if let Some(clause) = clauses.get_mut(clause_id) {
                if clause.deleted {
                    continue;
                }

                let old_len = clause.lits.len();

                // Normalize clause (remove duplicates, sort, check tautology)
                if clause.normalize() {
                    // Tautology detected - remove clause
                    clauses.remove(clause_id);
                    removed_clauses.push(clause_id);
                    self.stats.tautologies_removed += 1;
                    continue;
                }

                // Track if duplicates were removed
                if clause.lits.len() < old_len {
                    self.stats.duplicates_removed += old_len - clause.lits.len();
                }

                // Strengthen clause by removing falsified literals
                if self.strengthen_clause(clause, assignments) {
                    self.stats.clauses_strengthened += 1;
                }

                // Optimize tier based on usage and quality
                self.optimize_tier(clause);
            }
        }

        // Compact the database periodically
        clauses.compact();

        removed_clauses
    }

    /// Strengthen a clause by removing falsified literals
    ///
    /// Returns true if the clause was modified
    fn strengthen_clause(&self, clause: &mut crate::clause::Clause, assignments: &[LBool]) -> bool {
        let original_len = clause.lits.len();

        clause.lits.retain(|lit| {
            let var_idx = lit.var().index();
            if var_idx >= assignments.len() {
                return true; // Keep unassigned variables
            }

            let value = assignments[var_idx];

            // Keep literal if variable is undefined
            if value == LBool::Undef {
                return true;
            }

            // A literal is falsified if:
            // - Variable is True and literal is negative
            // - Variable is False and literal is positive
            let is_falsified =
                (value == LBool::True && lit.is_neg()) || (value == LBool::False && !lit.is_neg());

            !is_falsified
        });

        clause.lits.len() < original_len
    }

    /// Optimize clause tier based on usage and quality metrics
    fn optimize_tier(&mut self, clause: &mut crate::clause::Clause) {
        if !clause.learned {
            return;
        }

        let old_tier = clause.tier;

        // Promotion criteria:
        // - High usage count
        // - Low LBD (high quality)
        // - Small size

        if clause.tier == ClauseTier::Local {
            // Promote Local -> Mid if used frequently or has good LBD
            if clause.usage_count >= 3 || (clause.lbd <= 3 && clause.usage_count >= 2) {
                clause.tier = ClauseTier::Mid;
                self.stats.tier_promotions += 1;
            }
        } else if clause.tier == ClauseTier::Mid {
            // Promote Mid -> Core if very high usage or excellent LBD
            if clause.usage_count >= 10
                || clause.lbd <= 2
                || (clause.lbd <= 3 && clause.usage_count >= 5)
            {
                clause.tier = ClauseTier::Core;
                self.stats.tier_promotions += 1;
            }
        }

        // Demotion criteria:
        // - Low activity for extended period
        // - High LBD with low usage

        if clause.tier == ClauseTier::Mid && clause.activity < 0.1 && clause.usage_count < 2 {
            clause.tier = ClauseTier::Local;
            self.stats.tier_demotions += 1;
        }

        // Track tier changes in clause
        if old_tier != clause.tier {
            clause.usage_count = 0; // Reset usage counter on tier change
        }
    }

    /// Advanced clause reduction strategy
    ///
    /// Identifies clauses to delete based on multiple criteria:
    /// - Activity
    /// - LBD
    /// - Tier
    /// - Size
    /// - Age (usage count as proxy)
    ///
    /// Returns a list of clause IDs that should be deleted
    pub fn select_clauses_for_deletion(
        &self,
        clauses: &ClauseDatabase,
        target_count: usize,
    ) -> Vec<ClauseId> {
        let mut candidates: Vec<(ClauseId, f64)> = Vec::new();

        // Collect learned clauses with quality scores
        for i in 0..clauses.num_learned() {
            let clause_id = ClauseId::new(i as u32);
            if let Some(clause) = clauses.get(clause_id) {
                if clause.deleted || !clause.learned {
                    continue;
                }

                // Skip Core tier clauses (protected)
                if clause.tier == ClauseTier::Core {
                    continue;
                }

                // Compute deletion priority (higher = more likely to delete)
                // Factors:
                // - Low activity (weight: 0.4)
                // - High LBD (weight: 0.3)
                // - Large size (weight: 0.2)
                // - Tier (weight: 0.1)

                let activity_score = 1.0 - clause.activity.min(1.0);
                let lbd_score = (clause.lbd as f64 / 20.0).min(1.0);
                let size_score = ((clause.len() - 2) as f64 / 20.0).min(1.0);
                let tier_score = match clause.tier {
                    ClauseTier::Core => 0.0,  // Protected
                    ClauseTier::Mid => 0.3,   // Less likely to delete
                    ClauseTier::Local => 1.0, // Most likely to delete
                };

                let score =
                    activity_score * 0.4 + lbd_score * 0.3 + size_score * 0.2 + tier_score * 0.1;

                candidates.push((clause_id, score));
            }
        }

        // Sort by score (highest first = most deletable)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top candidates up to target_count
        candidates
            .into_iter()
            .take(target_count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MaintenanceStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::Clause;
    use crate::literal::{Lit, Var};

    #[test]
    fn test_maintenance_stats() {
        let mut maintenance = ClauseMaintenance::new();
        maintenance.stats.operations = 10;
        maintenance.stats.duplicates_removed = 5;

        let stats = maintenance.stats();
        assert_eq!(stats.operations, 10);
        assert_eq!(stats.duplicates_removed, 5);
    }

    #[test]
    fn test_tier_optimization() {
        let mut maintenance = ClauseMaintenance::new();
        let mut clause = Clause::learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);

        clause.usage_count = 3;
        clause.lbd = 3;

        maintenance.optimize_tier(&mut clause);

        // Should be promoted from Local to Mid
        assert_eq!(clause.tier, ClauseTier::Mid);
        assert_eq!(maintenance.stats.tier_promotions, 1);
    }

    #[test]
    fn test_clause_strengthening() {
        let maintenance = ClauseMaintenance::new();
        let mut clause = Clause::learned([
            Lit::pos(Var::new(0)),
            Lit::pos(Var::new(1)),
            Lit::pos(Var::new(2)),
        ]);

        let mut assignments = vec![LBool::Undef; 3];
        assignments[1] = LBool::False; // Var(1) is false

        // Lit::pos(Var::new(1)) should be removed since Var(1) is false
        let modified = maintenance.strengthen_clause(&mut clause, &assignments);
        assert!(modified);
        assert_eq!(clause.len(), 2);
    }

    #[test]
    fn test_deletion_selection() {
        let mut db = ClauseDatabase::new();
        let maintenance = ClauseMaintenance::new();

        // Add some learned clauses with different properties
        let mut c1 = Clause::learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);
        c1.activity = 0.1; // Low activity
        c1.lbd = 10; // High LBD
        let _id1 = db.add(c1);

        let mut c2 = Clause::learned([Lit::pos(Var::new(2)), Lit::pos(Var::new(3))]);
        c2.activity = 0.9; // High activity
        c2.lbd = 2; // Low LBD
        c2.promote_to_core(); // Protected
        let _id2 = db.add(c2);

        let to_delete = maintenance.select_clauses_for_deletion(&db, 1);

        // Should select c1 for deletion (low activity, high LBD, not protected)
        assert_eq!(to_delete.len(), 1);
    }
}
