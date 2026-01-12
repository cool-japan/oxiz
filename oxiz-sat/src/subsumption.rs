//! Clause Subsumption Checking
//!
//! This module implements efficient subsumption checking for learned clauses.
//! A clause C subsumes clause D if every literal in C is also in D.
//! When C subsumes D, D can be removed as it's redundant.
//!
//! This is particularly useful for:
//! - Removing redundant learned clauses
//! - Strengthening the clause database
//! - Reducing memory usage
//!
//! Reference: "Efficient Clause Subsumption" and MiniSat

use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::Lit;
use rustc_hash::FxHashSet;

/// Statistics for subsumption checking
#[derive(Debug, Default, Clone)]
pub struct SubsumptionStats {
    /// Number of subsumption checks performed
    pub checks_performed: u64,
    /// Number of clauses subsumed and removed
    pub clauses_subsumed: u64,
    /// Number of forward subsumptions (new clause subsumes old)
    pub forward_subsumptions: u64,
    /// Number of backward subsumptions (old clause subsumes new)
    pub backward_subsumptions: u64,
}

/// Subsumption checker for learned clauses
pub struct SubsumptionChecker {
    /// Temporary set for clause literals (reused across checks)
    clause_lits: FxHashSet<Lit>,
    /// Temporary set for candidate literals
    candidate_lits: FxHashSet<Lit>,
    /// Maximum clause size to check for subsumption
    max_check_size: usize,
    /// Statistics
    stats: SubsumptionStats,
}

impl SubsumptionChecker {
    /// Create a new subsumption checker
    pub fn new(max_check_size: usize) -> Self {
        Self {
            clause_lits: FxHashSet::default(),
            candidate_lits: FxHashSet::default(),
            max_check_size,
            stats: SubsumptionStats::default(),
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &SubsumptionStats {
        &self.stats
    }

    /// Check if clause A subsumes clause B
    ///
    /// A subsumes B if all literals in A are present in B
    fn subsumes(&mut self, clause_a: &[Lit], clause_b: &[Lit]) -> bool {
        // A cannot subsume B if A is larger than B
        if clause_a.len() > clause_b.len() {
            return false;
        }

        // Build set of literals in B
        self.candidate_lits.clear();
        for &lit in clause_b {
            self.candidate_lits.insert(lit);
        }

        // Check if all literals in A are in B
        for &lit in clause_a {
            if !self.candidate_lits.contains(&lit) {
                return false;
            }
        }

        true
    }

    /// Check if a newly learned clause subsumes any existing clauses
    ///
    /// Returns a list of clause IDs that are subsumed by the new clause
    pub fn check_forward_subsumption(
        &mut self,
        new_clause: &[Lit],
        clauses: &ClauseDatabase,
    ) -> Vec<ClauseId> {
        // Don't check if the new clause is too large
        if new_clause.len() > self.max_check_size {
            return Vec::new();
        }

        let mut subsumed = Vec::new();

        // Check against all learned clauses
        for clause_id in clauses.iter_ids() {
            let Some(clause) = clauses.get(clause_id) else {
                continue;
            };

            // Only check learned clauses
            if !clause.learned || clause.deleted {
                continue;
            }

            // Skip if the existing clause is too large
            if clause.len() > self.max_check_size {
                continue;
            }

            self.stats.checks_performed += 1;

            // Check if new clause subsumes this clause
            if self.subsumes(new_clause, &clause.lits) {
                subsumed.push(clause_id);
                self.stats.forward_subsumptions += 1;
            }
        }

        self.stats.clauses_subsumed += subsumed.len() as u64;
        subsumed
    }

    /// Check if a newly learned clause is subsumed by any existing clause
    ///
    /// Returns true if the new clause is subsumed (redundant)
    pub fn check_backward_subsumption(
        &mut self,
        new_clause: &[Lit],
        clauses: &ClauseDatabase,
    ) -> bool {
        // Don't check if the new clause is too large
        if new_clause.len() > self.max_check_size {
            return false;
        }

        // Build set of literals in new clause
        self.clause_lits.clear();
        for &lit in new_clause {
            self.clause_lits.insert(lit);
        }

        // Check against all learned clauses
        for clause_id in clauses.iter_ids() {
            let Some(clause) = clauses.get(clause_id) else {
                continue;
            };

            // Only check learned clauses
            if !clause.learned || clause.deleted {
                continue;
            }

            // Skip if the existing clause is larger (can't subsume)
            if clause.len() > new_clause.len() {
                continue;
            }

            // Skip if the existing clause is too large
            if clause.len() > self.max_check_size {
                continue;
            }

            self.stats.checks_performed += 1;

            // Check if existing clause subsumes new clause
            if self.subsumes(&clause.lits, new_clause) {
                self.stats.backward_subsumptions += 1;
                return true;
            }
        }

        false
    }

    /// Perform full subsumption check for a new clause
    ///
    /// Returns (is_subsumed, subsumed_clauses)
    /// - is_subsumed: true if the new clause is redundant
    /// - subsumed_clauses: list of clause IDs that the new clause makes redundant
    pub fn check_subsumption(
        &mut self,
        new_clause: &[Lit],
        clauses: &ClauseDatabase,
    ) -> (bool, Vec<ClauseId>) {
        // First check backward subsumption (is new clause redundant?)
        let is_subsumed = self.check_backward_subsumption(new_clause, clauses);

        // If new clause is subsumed, don't check forward subsumption
        if is_subsumed {
            return (true, Vec::new());
        }

        // Check forward subsumption (does new clause make others redundant?)
        let subsumed = self.check_forward_subsumption(new_clause, clauses);

        (false, subsumed)
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = SubsumptionStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_subsumption_creation() {
        let checker = SubsumptionChecker::new(10);
        assert_eq!(checker.max_check_size, 10);
    }

    #[test]
    fn test_subsumes() {
        let mut checker = SubsumptionChecker::new(20);

        // {x1, x2} subsumes {x1, x2, x3}
        let clause_a = vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))];
        let clause_b = vec![
            Lit::pos(Var::new(0)),
            Lit::pos(Var::new(1)),
            Lit::pos(Var::new(2)),
        ];

        assert!(checker.subsumes(&clause_a, &clause_b));
        assert!(!checker.subsumes(&clause_b, &clause_a)); // Reverse should be false
    }

    #[test]
    fn test_no_subsumption() {
        let mut checker = SubsumptionChecker::new(20);

        // {x1, x2} does not subsume {x1, x3}
        let clause_a = vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))];
        let clause_b = vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(2))];

        assert!(!checker.subsumes(&clause_a, &clause_b));
    }

    #[test]
    fn test_forward_subsumption() {
        let mut checker = SubsumptionChecker::new(20);
        let mut db = ClauseDatabase::new();

        // Add a learned clause {x1, x2, x3}
        let _old_id = db.add_learned([
            Lit::pos(Var::new(0)),
            Lit::pos(Var::new(1)),
            Lit::pos(Var::new(2)),
        ]);

        // New clause {x1, x2} should subsume the old one
        let new_clause = vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))];

        let subsumed = checker.check_forward_subsumption(&new_clause, &db);
        assert_eq!(subsumed.len(), 1);
    }

    #[test]
    fn test_backward_subsumption() {
        let mut checker = SubsumptionChecker::new(20);
        let mut db = ClauseDatabase::new();

        // Add a learned clause {x1, x2}
        let _old_id = db.add_learned([Lit::pos(Var::new(0)), Lit::pos(Var::new(1))]);

        // New clause {x1, x2, x3} should be subsumed by the old one
        let new_clause = vec![
            Lit::pos(Var::new(0)),
            Lit::pos(Var::new(1)),
            Lit::pos(Var::new(2)),
        ];

        assert!(checker.check_backward_subsumption(&new_clause, &db));
    }

    #[test]
    fn test_stats() {
        let mut checker = SubsumptionChecker::new(20);
        let stats = checker.stats();
        assert_eq!(stats.checks_performed, 0);
        assert_eq!(stats.clauses_subsumed, 0);

        checker.reset_stats();
        let stats = checker.stats();
        assert_eq!(stats.checks_performed, 0);
    }
}
