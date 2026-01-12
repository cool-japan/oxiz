//! Clause Vivification
//!
//! Vivification is a lightweight clause strengthening technique that tries to
//! remove literals from clauses through temporary propagation. It's faster
//! than distillation and can be applied more frequently.
//!
//! The basic idea:
//! - For each clause C = (l1 ∨ l2 ∨ ... ∨ ln)
//! - Temporarily assign some literals to false: ¬l1, ¬l2, ..., ¬lk
//! - Perform unit propagation
//! - If we find a conflict, some literals in the prefix can be removed
//! - If we derive some li (i > k), then l1,...,li-1 can be removed
//!
//! Key differences from distillation:
//! - Vivification propagates multiple literals from the clause at once
//! - It's less aggressive but much faster
//! - Can be applied more frequently during search

use crate::clause::{Clause, ClauseDatabase, ClauseId};
use crate::literal::{LBool, Lit};
use std::collections::VecDeque;

/// Statistics for vivification
#[derive(Debug, Default, Clone)]
pub struct VivificationStats {
    /// Number of clauses vivified
    pub clauses_vivified: u64,
    /// Number of literals removed
    pub literals_removed: u64,
    /// Number of clauses deleted (became unit/empty)
    pub clauses_deleted: u64,
    /// Number of propagations performed
    pub propagations: u64,
    /// Number of conflicts found
    pub conflicts: u64,
}

/// Clause vivification manager
pub struct Vivification {
    /// Assignment stack for temporary assignments
    assignment: Vec<LBool>,
    /// Trail of assigned variables (for backtracking)
    trail: Vec<Lit>,
    /// Propagation queue
    prop_queue: VecDeque<Lit>,
    /// Maximum number of propagations per vivification attempt
    max_props: u32,
    /// Statistics
    stats: VivificationStats,
}

impl Vivification {
    /// Create a new vivification manager
    #[must_use]
    pub fn new(num_vars: usize, max_props: u32) -> Self {
        Self {
            assignment: vec![LBool::Undef; num_vars],
            trail: Vec::new(),
            prop_queue: VecDeque::new(),
            max_props,
            stats: VivificationStats::default(),
        }
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &VivificationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = VivificationStats::default();
    }

    /// Resize internal structures when new variables are added
    pub fn resize(&mut self, num_vars: usize) {
        if num_vars > self.assignment.len() {
            self.assignment.resize(num_vars, LBool::Undef);
        }
    }

    /// Try to vivify a single clause
    ///
    /// Returns Some(strengthened_clause) if the clause was strengthened, None otherwise
    pub fn vivify_clause(&mut self, clause: &Clause, db: &ClauseDatabase) -> Option<Vec<Lit>> {
        if clause.deleted || clause.len() <= 2 {
            // Don't vivify binary or smaller clauses
            return None;
        }

        // Clear any previous state
        self.backtrack_all();

        let original_lits: Vec<Lit> = clause.lits.iter().copied().collect();
        let n = original_lits.len();

        // Try vivifying with different prefixes
        // We'll try to falsify literals one by one and see if we can derive
        // any of the remaining literals or find a conflict
        for split_point in 1..n {
            // Clear state for this attempt
            self.backtrack_all();

            // Temporarily assign the first split_point literals to false
            let mut conflict = false;
            for &lit in &original_lits[..split_point] {
                if !self.assign(!lit) {
                    conflict = true;
                    break;
                }
            }

            if conflict {
                // We found a conflict, meaning the clause can be strengthened
                // The literals in the prefix are redundant
                self.stats.conflicts += 1;
                self.stats.literals_removed += split_point as u64;

                let strengthened: Vec<Lit> = original_lits[split_point..].to_vec();
                if !strengthened.is_empty() {
                    self.stats.clauses_vivified += 1;
                    if strengthened.len() == 1 {
                        self.stats.clauses_deleted += 1;
                    }
                    return Some(strengthened);
                }
            }

            // Perform unit propagation
            if self.propagate(db) {
                // Conflict found during propagation
                self.stats.conflicts += 1;
                // The entire clause up to split_point is redundant
                self.stats.literals_removed += split_point as u64;

                let strengthened: Vec<Lit> = original_lits[split_point..].to_vec();
                if !strengthened.is_empty() {
                    self.stats.clauses_vivified += 1;
                    if strengthened.len() == 1 {
                        self.stats.clauses_deleted += 1;
                    }
                    return Some(strengthened);
                }
            }

            // Check if any of the remaining literals are now assigned to true
            for (idx, &lit) in original_lits[split_point..].iter().enumerate() {
                if self.value(lit) == LBool::True {
                    // We derived lit, so everything before it is redundant
                    let removed = split_point + idx;
                    self.stats.literals_removed += removed as u64;

                    let strengthened: Vec<Lit> = original_lits[removed..].to_vec();
                    if strengthened.len() < original_lits.len() {
                        self.stats.clauses_vivified += 1;
                        if strengthened.len() == 1 {
                            self.stats.clauses_deleted += 1;
                        }
                        return Some(strengthened);
                    }
                }
            }
        }

        None
    }

    /// Vivify all clauses in the database
    ///
    /// Returns the number of clauses strengthened
    pub fn vivify_all(&mut self, db: &mut ClauseDatabase) -> usize {
        let mut strengthened_count = 0;

        // Collect clause IDs to avoid borrow checker issues
        let clause_ids: Vec<ClauseId> = db.iter_ids().collect();

        for clause_id in clause_ids {
            if let Some(clause) = db.get(clause_id) {
                if clause.deleted {
                    continue;
                }

                let clause_copy = clause.clone();
                if let Some(strengthened) = self.vivify_clause(&clause_copy, db) {
                    // Update the clause in the database
                    if let Some(clause) = db.get_mut(clause_id) {
                        clause.lits = strengthened.into();
                        strengthened_count += 1;
                    }
                }
            }
        }

        strengthened_count
    }

    /// Assign a literal to true
    ///
    /// Returns false if this causes an immediate conflict
    fn assign(&mut self, lit: Lit) -> bool {
        let var = lit.var();
        let val = self.assignment[var.index()];

        match val {
            LBool::Undef => {
                self.assignment[var.index()] = if lit.is_pos() {
                    LBool::True
                } else {
                    LBool::False
                };
                self.trail.push(lit);
                self.prop_queue.push_back(lit);
                true
            }
            LBool::True if lit.is_pos() => true,
            LBool::False if lit.is_neg() => true,
            _ => false, // Conflict
        }
    }

    /// Get the value of a literal
    fn value(&self, lit: Lit) -> LBool {
        let val = self.assignment[lit.var().index()];
        if val == LBool::Undef {
            return LBool::Undef;
        }
        if lit.is_pos() { val } else { val.negate() }
    }

    /// Perform unit propagation
    ///
    /// Returns true if a conflict was found
    fn propagate(&mut self, db: &ClauseDatabase) -> bool {
        let mut props = 0;

        while let Some(lit) = self.prop_queue.pop_front() {
            if props >= self.max_props {
                break;
            }
            props += 1;
            self.stats.propagations += 1;

            // Check all clauses containing ~lit
            for clause_id in db.iter_ids() {
                if let Some(clause) = db.get(clause_id) {
                    if clause.deleted {
                        continue;
                    }

                    // Count undefined and true literals
                    let mut undef_lit = None;
                    let mut true_count = 0;
                    let mut contains_neg_lit = false;

                    for &l in &clause.lits {
                        if l == !lit {
                            contains_neg_lit = true;
                        }
                        match self.value(l) {
                            LBool::True => {
                                true_count += 1;
                                break;
                            }
                            LBool::Undef => {
                                undef_lit = Some(l);
                            }
                            LBool::False => {}
                        }
                    }

                    if !contains_neg_lit {
                        continue;
                    }

                    if true_count > 0 {
                        // Clause is satisfied
                        continue;
                    }

                    if let Some(unit_lit) = undef_lit {
                        // Unit clause - propagate
                        if !self.assign(unit_lit) {
                            return true; // Conflict
                        }
                    } else {
                        // All literals are false - conflict
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Backtrack all assignments
    fn backtrack_all(&mut self) {
        for &lit in &self.trail {
            self.assignment[lit.var().index()] = LBool::Undef;
        }
        self.trail.clear();
        self.prop_queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::literal::Var;

    #[test]
    fn test_vivification_creation() {
        let viv = Vivification::new(10, 1000);
        assert_eq!(viv.stats().clauses_vivified, 0);
    }

    #[test]
    fn test_vivification_stats() {
        let viv = Vivification::new(10, 1000);
        let stats = viv.stats();
        assert_eq!(stats.clauses_vivified, 0);
        assert_eq!(stats.literals_removed, 0);
    }

    #[test]
    fn test_vivification_resize() {
        let mut viv = Vivification::new(10, 1000);
        viv.resize(20);
        assert_eq!(viv.assignment.len(), 20);
    }

    #[test]
    fn test_vivify_binary_clause() {
        let mut viv = Vivification::new(10, 1000);
        let db = ClauseDatabase::new();

        let clause = Clause::new(vec![Lit::pos(Var::new(0)), Lit::pos(Var::new(1))], false);

        // Binary clauses should not be vivified
        assert!(viv.vivify_clause(&clause, &db).is_none());
    }

    #[test]
    fn test_vivify_clause_basic() {
        let mut viv = Vivification::new(10, 1000);
        let mut db = ClauseDatabase::new();

        // Add a clause that can be strengthened
        // (a ∨ b ∨ c) where we can show that a is redundant
        db.add(Clause::new(
            vec![
                Lit::pos(Var::new(0)),
                Lit::pos(Var::new(1)),
                Lit::pos(Var::new(2)),
            ],
            false,
        ));

        // This test just checks that vivification runs without errors
        let clause = db.get(ClauseId(0)).unwrap().clone();
        let _result = viv.vivify_clause(&clause, &db);
    }

    #[test]
    fn test_vivify_all() {
        let mut viv = Vivification::new(10, 1000);
        let mut db = ClauseDatabase::new();

        db.add(Clause::new(
            vec![
                Lit::pos(Var::new(0)),
                Lit::pos(Var::new(1)),
                Lit::pos(Var::new(2)),
            ],
            false,
        ));

        db.add(Clause::new(
            vec![
                Lit::pos(Var::new(3)),
                Lit::pos(Var::new(4)),
                Lit::pos(Var::new(5)),
            ],
            false,
        ));

        let _count = viv.vivify_all(&mut db);
        // Just verify it runs without errors
    }

    #[test]
    fn test_assign_and_value() {
        let mut viv = Vivification::new(10, 1000);

        let lit = Lit::pos(Var::new(0));
        assert_eq!(viv.value(lit), LBool::Undef);

        assert!(viv.assign(lit));
        assert_eq!(viv.value(lit), LBool::True);
        assert_eq!(viv.value(!lit), LBool::False);
    }

    #[test]
    fn test_assign_conflict() {
        let mut viv = Vivification::new(10, 1000);

        let lit = Lit::pos(Var::new(0));
        assert!(viv.assign(lit));

        // Assigning the negation should cause a conflict
        assert!(!viv.assign(!lit));
    }

    #[test]
    fn test_backtrack() {
        let mut viv = Vivification::new(10, 1000);

        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::pos(Var::new(1));

        viv.assign(lit1);
        viv.assign(lit2);

        assert_eq!(viv.value(lit1), LBool::True);
        assert_eq!(viv.value(lit2), LBool::True);

        viv.backtrack_all();

        assert_eq!(viv.value(lit1), LBool::Undef);
        assert_eq!(viv.value(lit2), LBool::Undef);
    }

    #[test]
    fn test_reset_stats() {
        let mut viv = Vivification::new(10, 1000);
        viv.stats.clauses_vivified = 10;
        viv.stats.literals_removed = 20;

        viv.reset_stats();

        assert_eq!(viv.stats().clauses_vivified, 0);
        assert_eq!(viv.stats().literals_removed, 0);
    }
}
