//! Clause Distillation
//!
//! Distillation is a clause strengthening technique that tries to remove
//! literals from clauses through a limited form of search. It's more
//! aggressive than vivification but still efficient.
//!
//! The basic idea:
//! - For each literal L in a clause C
//! - Temporarily assign ~L (trying to falsify L)
//! - Perform limited unit propagation
//! - If we can derive C\{L} (clause without L), then L is redundant
//! - Remove L from C

use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::{LBool, Lit};
use crate::trail::Trail;
use crate::watched::WatchLists;

/// Statistics for distillation
#[derive(Debug, Default, Clone)]
pub struct DistillationStats {
    /// Number of clauses distilled
    pub clauses_distilled: u64,
    /// Number of literals removed
    pub literals_removed: u64,
    /// Number of clauses deleted (became unit/empty)
    pub clauses_deleted: u64,
}

/// Clause distillation manager
pub struct Distillation {
    /// Maximum propagation depth
    #[allow(dead_code)]
    max_depth: u32,
    /// Statistics
    stats: DistillationStats,
}

impl Distillation {
    /// Create a new distillation manager
    pub fn new(max_depth: u32) -> Self {
        Self {
            max_depth,
            stats: DistillationStats::default(),
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &DistillationStats {
        &self.stats
    }

    /// Try to distill a single clause
    ///
    /// Returns true if the clause was strengthened or deleted
    #[allow(clippy::too_many_arguments)]
    pub fn distill_clause(
        &mut self,
        clause_id: ClauseId,
        clauses: &mut ClauseDatabase,
        trail: &mut Trail,
        watches: &mut WatchLists,
        assignment: &[LBool],
    ) -> bool {
        let Some(clause) = clauses.get(clause_id) else {
            return false;
        };

        if clause.deleted || clause.len() <= 2 {
            // Don't distill binary or smaller clauses
            return false;
        }

        let original_lits: Vec<Lit> = clause.lits.iter().copied().collect();
        let mut strengthened = false;

        // Try to remove each literal
        for &lit in &original_lits {
            // Skip if already assigned
            if assignment[lit.var().index()] != LBool::Undef {
                continue;
            }

            // Try assigning ~lit and see if we can derive the clause without lit
            if self.try_remove_literal(lit, clause_id, clauses, trail, watches, assignment) {
                strengthened = true;
                self.stats.literals_removed += 1;
            }
        }

        if strengthened {
            self.stats.clauses_distilled += 1;

            // Check if clause became unit or empty
            if let Some(clause) = clauses.get(clause_id)
                && clause.len() <= 1
            {
                self.stats.clauses_deleted += 1;
                return true;
            }
        }

        strengthened
    }

    /// Try to remove a literal from a clause through limited propagation
    #[allow(clippy::too_many_arguments)]
    fn try_remove_literal(
        &self,
        lit: Lit,
        clause_id: ClauseId,
        clauses: &mut ClauseDatabase,
        trail: &mut Trail,
        _watches: &mut WatchLists,
        assignment: &[LBool],
    ) -> bool {
        // Save current trail level
        let saved_level = trail.decision_level();

        // Make a new decision level for this test
        trail.new_decision_level();

        // Assign ~lit
        let _test_lit = lit.negate();

        // Check if this causes immediate conflict with other literals in the clause
        let Some(clause) = clauses.get(clause_id) else {
            trail.backtrack_to(saved_level);
            return false;
        };

        for &other_lit in &clause.lits {
            if other_lit == lit {
                continue;
            }

            // If other literal is already false under ~lit, we can't remove lit
            if assignment[other_lit.var().index()] == LBool::from(!other_lit.sign()) {
                trail.backtrack_to(saved_level);
                return false;
            }

            // Simple check: if assigning ~lit would make another literal in the
            // clause become true, then lit is potentially redundant
            if assignment[other_lit.var().index()] == LBool::from(other_lit.sign()) {
                // Other literal is already true, so lit is redundant
                if let Some(clause) = clauses.get_mut(clause_id) {
                    clause.lits.retain(|l| *l != lit);
                }
                trail.backtrack_to(saved_level);
                return true;
            }
        }

        // For now, just do a simple check
        // A full implementation would do limited BCP here
        trail.backtrack_to(saved_level);

        // Conservative: don't remove unless we're certain
        false
    }

    /// Distill all learned clauses in the database
    pub fn distill_all(
        &mut self,
        clauses: &mut ClauseDatabase,
        trail: &mut Trail,
        watches: &mut WatchLists,
        assignment: &[LBool],
    ) -> u64 {
        let mut total_strengthened = 0;

        // Only distill learned clauses
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
            if self.distill_clause(clause_id, clauses, trail, watches, assignment) {
                total_strengthened += 1;
            }
        }

        total_strengthened
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_distillation_stats() {
        let distill = Distillation::new(10);
        let stats = distill.stats();
        assert_eq!(stats.clauses_distilled, 0);
        assert_eq!(stats.literals_removed, 0);
    }

    #[test]
    fn test_distillation_creation() {
        let distill = Distillation::new(5);
        assert_eq!(distill.max_depth, 5);
    }

    #[test]
    fn test_distillation_binary_clause() {
        let mut distill = Distillation::new(10);
        let mut db = ClauseDatabase::new();
        let mut trail = Trail::new(10);
        let mut watches = WatchLists::new(10);
        let assignment = vec![LBool::Undef; 10];

        // Add a binary clause - should not be distilled
        let clause_id = db.add_learned([Lit::pos(Var::new(0)), Lit::neg(Var::new(1))]);

        let result =
            distill.distill_clause(clause_id, &mut db, &mut trail, &mut watches, &assignment);
        assert!(!result); // Binary clauses are not distilled
    }
}
