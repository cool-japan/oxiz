//! Covered Clause Elimination (CCE)
//!
//! CCE is an advanced preprocessing technique that removes redundant clauses.
//! A clause C is "covered" by clause D if there exists a literal l in C such that
//! D is exactly C with l removed. In this case, C is redundant and can be eliminated.
//!
//! More formally: C is covered by D if ∃l ∈ C such that D = C \ {l}
//!
//! Example:
//! - C = (a ∨ b ∨ c)
//! - D = (a ∨ b)
//! - C is covered by D (by removing c)
//!
//! CCE is more powerful than subsumption and can lead to significant formula simplification.

use crate::clause::{ClauseDatabase, ClauseId};
use crate::literal::Lit;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

/// Covered Clause Elimination engine
///
/// Identifies and removes covered clauses from the clause database.
/// This is done by building an index of clause signatures and checking
/// for coverage relationships.
pub struct CoveredClauseElimination {
    /// Clauses indexed by their literals (for fast lookup)
    clause_index: FxHashMap<Lit, Vec<ClauseId>>,
    /// Clauses marked for removal
    to_remove: FxHashSet<ClauseId>,
    /// Statistics
    stats: CceStats,
}

/// Statistics for CCE
#[derive(Debug, Default, Clone)]
pub struct CceStats {
    /// Number of covered clauses eliminated
    pub eliminated: usize,
    /// Number of coverage checks performed
    pub checks: usize,
    /// Number of clauses processed
    pub processed: usize,
}

impl CoveredClauseElimination {
    /// Create a new CCE engine
    #[must_use]
    pub fn new() -> Self {
        Self {
            clause_index: FxHashMap::default(),
            to_remove: FxHashSet::default(),
            stats: CceStats::default(),
        }
    }

    /// Build index of clauses by their literals
    fn build_index(&mut self, clauses: &ClauseDatabase) {
        self.clause_index.clear();

        for id in clauses.iter_ids() {
            if let Some(clause) = clauses.get(id) {
                for &lit in &clause.lits {
                    self.clause_index.entry(lit).or_default().push(id);
                }
            }
        }
    }

    /// Check if clause c1 covers clause c2
    ///
    /// c1 covers c2 if there exists a literal l in c2 such that c1 = c2 \ {l}
    /// In other words, c1 is c2 with one literal removed.
    fn is_covered_by(&mut self, c1_lits: &[Lit], c2_lits: &[Lit]) -> bool {
        self.stats.checks += 1;

        // c1 must be exactly one literal shorter than c2
        if c1_lits.len() + 1 != c2_lits.len() {
            return false;
        }

        // Check if c1 is a subset of c2
        let c1_set: FxHashSet<Lit> = c1_lits.iter().copied().collect();

        // Count how many literals from c2 are in c1
        let common_count = c2_lits.iter().filter(|lit| c1_set.contains(lit)).count();

        // If all literals from c1 are in c2, then c1 covers c2
        common_count == c1_lits.len()
    }

    /// Run CCE elimination on the clause database
    ///
    /// Returns the number of clauses eliminated.
    pub fn eliminate(&mut self, clauses: &mut ClauseDatabase) -> usize {
        self.to_remove.clear();
        self.build_index(clauses);

        // Collect all clause IDs and their literals
        let mut clause_list: Vec<(ClauseId, SmallVec<[Lit; 8]>)> = Vec::new();
        for id in clauses.iter_ids() {
            if let Some(clause) = clauses.get(id) {
                let lits: SmallVec<[Lit; 8]> = clause.lits.iter().copied().collect();
                clause_list.push((id, lits));
            }
        }

        // Sort by clause length (shorter clauses first)
        clause_list.sort_by_key(|(_, lits)| lits.len());

        // For each clause, check if it's covered by any shorter clause
        for i in 0..clause_list.len() {
            let (c2_id, ref c2_lits) = clause_list[i];

            if self.to_remove.contains(&c2_id) {
                continue;
            }

            self.stats.processed += 1;

            // Check against all shorter clauses
            for (c1_id, c1_lits) in clause_list.iter().take(i) {
                if self.to_remove.contains(c1_id) {
                    continue;
                }

                // Check if c2 is covered by c1
                if self.is_covered_by(c1_lits, c2_lits) {
                    self.to_remove.insert(c2_id);
                    self.stats.eliminated += 1;
                    break;
                }
            }
        }

        // Remove covered clauses
        for id in &self.to_remove {
            clauses.remove(*id);
        }

        self.stats.eliminated
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &CceStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CceStats::default();
    }
}

impl Default for CoveredClauseElimination {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::Clause;
    use crate::literal::Var;

    #[test]
    fn test_cce_basic() {
        let mut cce = CoveredClauseElimination::new();
        let mut db = ClauseDatabase::new();

        // Add clause: (a v b v c)
        db.add(Clause::new(
            vec![
                Lit::pos(Var::new(0)),
                Lit::pos(Var::new(1)),
                Lit::pos(Var::new(2)),
            ],
            false,
        ));

        // Add clause: (a v b) - this covers the first clause
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));

        let eliminated = cce.eliminate(&mut db);

        // The first clause should be eliminated
        assert_eq!(eliminated, 1);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_cce_no_coverage() {
        let mut cce = CoveredClauseElimination::new();
        let mut db = ClauseDatabase::new();

        // Add clause: (a v b)
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));

        // Add clause: (c v d) - no coverage relationship
        db.add(Clause::new(
            vec![Lit::pos(Var::new(2)), Lit::pos(Var::new(3))],
            false,
        ));

        let eliminated = cce.eliminate(&mut db);

        // No clauses should be eliminated
        assert_eq!(eliminated, 0);
        assert_eq!(db.len(), 2);
    }

    #[test]
    fn test_cce_multiple_covered() {
        let mut cce = CoveredClauseElimination::new();
        let mut db = ClauseDatabase::new();

        // Add short clause: (a)
        db.add(Clause::new(vec![Lit::pos(Var::new(0))], false));

        // Add covered clauses
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(2))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(3))],
            false,
        ));

        let eliminated = cce.eliminate(&mut db);

        // Three clauses should be eliminated (all covered by (a))
        assert_eq!(eliminated, 3);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_cce_stats() {
        let mut cce = CoveredClauseElimination::new();
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![
                Lit::pos(Var::new(0)),
                Lit::pos(Var::new(1)),
                Lit::pos(Var::new(2)),
            ],
            false,
        ));

        cce.eliminate(&mut db);
        let stats = cce.stats();

        assert_eq!(stats.eliminated, 1);
        assert!(stats.checks > 0);
        assert!(stats.processed > 0);
    }

    #[test]
    fn test_cce_is_covered_by() {
        let mut cce = CoveredClauseElimination::new();

        // (a v b) covers (a v b v c)
        let c1 = vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))];
        let c2 = vec![
            Lit::pos(Var::new(0)),
            Lit::pos(Var::new(1)),
            Lit::pos(Var::new(2)),
        ];

        assert!(cce.is_covered_by(&c1, &c2));

        // (a v b) does not cover (a v c)
        let c3 = vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(2))];

        assert!(!cce.is_covered_by(&c1, &c3));
    }

    #[test]
    fn test_cce_empty_db() {
        let mut cce = CoveredClauseElimination::new();
        let mut db = ClauseDatabase::new();

        let eliminated = cce.eliminate(&mut db);

        assert_eq!(eliminated, 0);
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_cce_single_clause() {
        let mut cce = CoveredClauseElimination::new();
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(vec![Lit::pos(Var::new(0))], false));

        let eliminated = cce.eliminate(&mut db);

        assert_eq!(eliminated, 0);
        assert_eq!(db.len(), 1);
    }
}
