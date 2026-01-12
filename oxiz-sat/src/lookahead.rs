//! Look-ahead Branching Heuristics
//!
//! Look-ahead branching involves probing variables before making branching
//! decisions. This module implements several classic look-ahead heuristics:
//!
//! - **MOMS (Maximum Occurrences in clauses of Minimum Size)**
//!   - Counts occurrences in smallest clauses
//!   - Formula: k × f(x) + (1-k) × f(¬x) where f counts occurrences
//!   - Typical k = 0.5 for balanced selection
//!
//! - **Jeroslow-Wang**
//!   - Exponential weighting by clause size
//!   - Formula: Σ 2^(-|C|) for each clause C containing the literal
//!   - Prefers literals in smaller clauses more strongly
//!
//! - **DLCS (Dynamic Largest Combined Sum)**
//!   - Simple occurrence counting
//!   - Formula: f(x) + f(¬x) where f counts occurrences
//!   - Prefers variables that appear frequently
//!
//! These heuristics are particularly effective for hard combinatorial problems
//! and structured instances.

use crate::clause::ClauseDatabase;
use crate::literal::{LBool, Lit, Var};

/// Look-ahead branching heuristic type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookaheadHeuristic {
    /// Maximum Occurrences in clauses of Minimum Size
    MOMS,
    /// Jeroslow-Wang heuristic with exponential clause size weighting
    JeroslowWang,
    /// Dynamic Largest Combined Sum
    DLCS,
}

/// Statistics for look-ahead branching
#[derive(Debug, Default, Clone)]
pub struct LookaheadStats {
    /// Number of variables selected
    pub selections: u64,
    /// Total time spent computing scores (in operations)
    pub score_computations: u64,
}

/// Look-ahead branching manager
pub struct LookaheadBranching {
    /// Current heuristic
    heuristic: LookaheadHeuristic,
    /// MOMS balance parameter (0.0 to 1.0)
    /// k=0.5 means equal weight to positive and negative occurrences
    moms_k: f64,
    /// Statistics
    stats: LookaheadStats,
}

impl LookaheadBranching {
    /// Create a new look-ahead branching manager
    #[must_use]
    pub fn new(heuristic: LookaheadHeuristic) -> Self {
        Self {
            heuristic,
            moms_k: 0.5,
            stats: LookaheadStats::default(),
        }
    }

    /// Create with custom MOMS k parameter
    #[must_use]
    pub fn with_moms_k(mut self, k: f64) -> Self {
        self.moms_k = k.clamp(0.0, 1.0);
        self
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &LookaheadStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = LookaheadStats::default();
    }

    /// Set the heuristic
    pub fn set_heuristic(&mut self, heuristic: LookaheadHeuristic) {
        self.heuristic = heuristic;
    }

    /// Select the next variable to branch on
    ///
    /// Returns None if no unassigned variables remain
    pub fn select_variable(
        &mut self,
        db: &ClauseDatabase,
        assignment: &[LBool],
        num_vars: usize,
    ) -> Option<Var> {
        self.stats.selections += 1;

        match self.heuristic {
            LookaheadHeuristic::MOMS => self.select_moms(db, assignment, num_vars),
            LookaheadHeuristic::JeroslowWang => self.select_jeroslow_wang(db, assignment, num_vars),
            LookaheadHeuristic::DLCS => self.select_dlcs(db, assignment, num_vars),
        }
    }

    /// Select variable using MOMS heuristic
    fn select_moms(
        &mut self,
        db: &ClauseDatabase,
        assignment: &[LBool],
        num_vars: usize,
    ) -> Option<Var> {
        // Find minimum clause size
        let min_size = self.find_min_clause_size(db, assignment)?;

        // Count occurrences in clauses of minimum size
        let mut pos_count = vec![0; num_vars];
        let mut neg_count = vec![0; num_vars];

        for clause_id in db.iter_ids() {
            if let Some(clause) = db.get(clause_id) {
                if clause.deleted {
                    continue;
                }

                // Count unassigned literals
                let unassigned: Vec<Lit> = clause
                    .lits
                    .iter()
                    .copied()
                    .filter(|&lit| assignment[lit.var().index()] == LBool::Undef)
                    .collect();

                if unassigned.len() == min_size {
                    for lit in unassigned {
                        if lit.is_pos() {
                            pos_count[lit.var().index()] += 1;
                        } else {
                            neg_count[lit.var().index()] += 1;
                        }
                    }
                }
            }
        }

        self.stats.score_computations += 1;

        // Select variable with maximum score: k × f(x) + (1-k) × f(¬x)
        let mut best_var = None;
        let mut best_score = 0.0;

        for v in 0..num_vars {
            if assignment[v] != LBool::Undef {
                continue;
            }

            let score =
                self.moms_k * pos_count[v] as f64 + (1.0 - self.moms_k) * neg_count[v] as f64;
            if score > best_score {
                best_score = score;
                best_var = Some(Var::new(v as u32));
            }
        }

        best_var
    }

    /// Select variable using Jeroslow-Wang heuristic
    fn select_jeroslow_wang(
        &mut self,
        db: &ClauseDatabase,
        assignment: &[LBool],
        num_vars: usize,
    ) -> Option<Var> {
        let mut pos_score = vec![0.0; num_vars];
        let mut neg_score = vec![0.0; num_vars];

        for clause_id in db.iter_ids() {
            if let Some(clause) = db.get(clause_id) {
                if clause.deleted {
                    continue;
                }

                // Count unassigned literals
                let unassigned: Vec<Lit> = clause
                    .lits
                    .iter()
                    .copied()
                    .filter(|&lit| assignment[lit.var().index()] == LBool::Undef)
                    .collect();

                if unassigned.is_empty() {
                    continue;
                }

                // Weight by 2^(-size)
                let weight = 2.0_f64.powi(-(unassigned.len() as i32));

                for lit in unassigned {
                    if lit.is_pos() {
                        pos_score[lit.var().index()] += weight;
                    } else {
                        neg_score[lit.var().index()] += weight;
                    }
                }
            }
        }

        self.stats.score_computations += 1;

        // Select variable with maximum combined score
        let mut best_var = None;
        let mut best_score = 0.0;

        for v in 0..num_vars {
            if assignment[v] != LBool::Undef {
                continue;
            }

            let score = pos_score[v] + neg_score[v];
            if score > best_score {
                best_score = score;
                best_var = Some(Var::new(v as u32));
            }
        }

        best_var
    }

    /// Select variable using DLCS heuristic
    fn select_dlcs(
        &mut self,
        db: &ClauseDatabase,
        assignment: &[LBool],
        num_vars: usize,
    ) -> Option<Var> {
        let mut pos_count = vec![0; num_vars];
        let mut neg_count = vec![0; num_vars];

        for clause_id in db.iter_ids() {
            if let Some(clause) = db.get(clause_id) {
                if clause.deleted {
                    continue;
                }

                // Count unassigned literals
                for &lit in &clause.lits {
                    if assignment[lit.var().index()] == LBool::Undef {
                        if lit.is_pos() {
                            pos_count[lit.var().index()] += 1;
                        } else {
                            neg_count[lit.var().index()] += 1;
                        }
                    }
                }
            }
        }

        self.stats.score_computations += 1;

        // Select variable with maximum combined count
        let mut best_var = None;
        let mut best_count = 0;

        for v in 0..num_vars {
            if assignment[v] != LBool::Undef {
                continue;
            }

            let count = pos_count[v] + neg_count[v];
            if count > best_count {
                best_count = count;
                best_var = Some(Var::new(v as u32));
            }
        }

        best_var
    }

    /// Get the preferred polarity for a variable using the current heuristic
    ///
    /// Returns true for positive polarity, false for negative
    pub fn select_polarity(&self, var: Var, db: &ClauseDatabase, assignment: &[LBool]) -> bool {
        let mut pos_weight = 0.0;
        let mut neg_weight = 0.0;

        for clause_id in db.iter_ids() {
            if let Some(clause) = db.get(clause_id) {
                if clause.deleted {
                    continue;
                }

                // Count unassigned literals
                let unassigned: Vec<Lit> = clause
                    .lits
                    .iter()
                    .copied()
                    .filter(|&lit| assignment[lit.var().index()] == LBool::Undef)
                    .collect();

                if unassigned.is_empty() {
                    continue;
                }

                let weight = match self.heuristic {
                    LookaheadHeuristic::JeroslowWang => 2.0_f64.powi(-(unassigned.len() as i32)),
                    _ => 1.0,
                };

                for lit in unassigned {
                    if lit.var() == var {
                        if lit.is_pos() {
                            pos_weight += weight;
                        } else {
                            neg_weight += weight;
                        }
                    }
                }
            }
        }

        // Prefer the polarity with higher weight
        pos_weight >= neg_weight
    }

    /// Find the minimum clause size (counting only unassigned literals)
    fn find_min_clause_size(&self, db: &ClauseDatabase, assignment: &[LBool]) -> Option<usize> {
        let mut min_size = usize::MAX;

        for clause_id in db.iter_ids() {
            if let Some(clause) = db.get(clause_id) {
                if clause.deleted {
                    continue;
                }

                let unassigned_count = clause
                    .lits
                    .iter()
                    .filter(|&&lit| assignment[lit.var().index()] == LBool::Undef)
                    .count();

                if unassigned_count > 0 && unassigned_count < min_size {
                    min_size = unassigned_count;
                }
            }
        }

        if min_size == usize::MAX {
            None
        } else {
            Some(min_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::Clause;

    #[test]
    fn test_lookahead_creation() {
        let la = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        assert_eq!(la.stats().selections, 0);
    }

    #[test]
    fn test_moms_k_parameter() {
        let la = LookaheadBranching::new(LookaheadHeuristic::MOMS).with_moms_k(0.7);
        assert_eq!(la.moms_k, 0.7);
    }

    #[test]
    fn test_moms_k_clamping() {
        let la = LookaheadBranching::new(LookaheadHeuristic::MOMS).with_moms_k(1.5);
        assert_eq!(la.moms_k, 1.0);

        let la = LookaheadBranching::new(LookaheadHeuristic::MOMS).with_moms_k(-0.5);
        assert_eq!(la.moms_k, 0.0);
    }

    #[test]
    fn test_set_heuristic() {
        let mut la = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        la.set_heuristic(LookaheadHeuristic::JeroslowWang);
        assert_eq!(la.heuristic, LookaheadHeuristic::JeroslowWang);
    }

    #[test]
    fn test_select_variable_moms() {
        let mut la = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(1)), Lit::pos(Var::new(2))],
            false,
        ));

        let assignment = vec![LBool::Undef; 3];
        let var = la.select_variable(&db, &assignment, 3);

        assert!(var.is_some());
        assert_eq!(la.stats().selections, 1);
    }

    #[test]
    fn test_select_variable_jeroslow_wang() {
        let mut la = LookaheadBranching::new(LookaheadHeuristic::JeroslowWang);
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(1)), Lit::pos(Var::new(2))],
            false,
        ));

        let assignment = vec![LBool::Undef; 3];
        let var = la.select_variable(&db, &assignment, 3);

        assert!(var.is_some());
    }

    #[test]
    fn test_select_variable_dlcs() {
        let mut la = LookaheadBranching::new(LookaheadHeuristic::DLCS);
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(1)), Lit::pos(Var::new(2))],
            false,
        ));

        let assignment = vec![LBool::Undef; 3];
        let var = la.select_variable(&db, &assignment, 3);

        assert!(var.is_some());
    }

    #[test]
    fn test_select_variable_all_assigned() {
        let mut la = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));

        let assignment = vec![LBool::True; 2];
        let var = la.select_variable(&db, &assignment, 2);

        assert!(var.is_none());
    }

    #[test]
    fn test_select_polarity() {
        let la = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(2))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(0)), Lit::pos(Var::new(3))],
            false,
        ));

        let assignment = vec![LBool::Undef; 4];
        let polarity = la.select_polarity(Var::new(0), &db, &assignment);

        // Var 0 appears positive twice and negative once, so should prefer positive
        assert!(polarity);
    }

    #[test]
    fn test_find_min_clause_size() {
        let la = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![
                Lit::neg(Var::new(1)),
                Lit::pos(Var::new(2)),
                Lit::pos(Var::new(3)),
            ],
            false,
        ));

        let assignment = vec![LBool::Undef; 4];
        let min_size = la.find_min_clause_size(&db, &assignment);

        assert_eq!(min_size, Some(2));
    }

    #[test]
    fn test_reset_stats() {
        let mut la = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        la.stats.selections = 10;
        la.stats.score_computations = 20;

        la.reset_stats();

        assert_eq!(la.stats().selections, 0);
        assert_eq!(la.stats().score_computations, 0);
    }

    #[test]
    fn test_multiple_heuristics() {
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))],
            false,
        ));
        db.add(Clause::new(
            vec![Lit::neg(Var::new(1)), Lit::pos(Var::new(2))],
            false,
        ));

        let assignment = vec![LBool::Undef; 3];

        let mut la_moms = LookaheadBranching::new(LookaheadHeuristic::MOMS);
        let var_moms = la_moms.select_variable(&db, &assignment, 3);
        assert!(var_moms.is_some());

        let mut la_jw = LookaheadBranching::new(LookaheadHeuristic::JeroslowWang);
        let var_jw = la_jw.select_variable(&db, &assignment, 3);
        assert!(var_jw.is_some());

        let mut la_dlcs = LookaheadBranching::new(LookaheadHeuristic::DLCS);
        let var_dlcs = la_dlcs.select_variable(&db, &assignment, 3);
        assert!(var_dlcs.is_some());
    }
}
