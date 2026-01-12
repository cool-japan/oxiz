//! Recursive Clause Minimization
//!
//! This module implements recursive minimization of learned clauses.
//! The basic idea is to recursively check if a literal in a learned clause
//! can be removed by examining its reason clause.
//!
//! Reference: "Minimizing Learned Clauses" by Niklas Sörensson and Armin Biere

use crate::clause::ClauseDatabase;
use crate::literal::Lit;
use crate::trail::Reason;
use smallvec::SmallVec;

/// Literal redundancy status for recursive minimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LitStatus {
    /// Literal is known to be redundant
    Redundant,
    /// Literal is known to be non-redundant (essential)
    Essential,
    /// Status is unknown (not yet checked)
    Unknown,
}

/// Statistics for recursive minimization
#[derive(Debug, Default, Clone)]
pub struct RecursiveMinStats {
    /// Number of clauses minimized
    pub clauses_minimized: u64,
    /// Number of literals removed
    pub literals_removed: u64,
    /// Number of recursive checks performed
    pub recursive_checks: u64,
}

/// Recursive clause minimization manager
pub struct RecursiveMinimizer {
    /// Status of each literal during minimization
    status: Vec<LitStatus>,
    /// Stack for depth-first traversal
    stack: Vec<Lit>,
    /// Statistics
    stats: RecursiveMinStats,
}

impl RecursiveMinimizer {
    /// Create a new recursive minimizer
    pub fn new(num_vars: usize) -> Self {
        Self {
            status: vec![LitStatus::Unknown; num_vars * 2],
            stack: Vec::new(),
            stats: RecursiveMinStats::default(),
        }
    }

    /// Resize the minimizer for a new number of variables
    pub fn resize(&mut self, num_vars: usize) {
        self.status.resize(num_vars * 2, LitStatus::Unknown);
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &RecursiveMinStats {
        &self.stats
    }

    /// Minimize a learned clause using recursive minimization
    ///
    /// Returns the minimized clause
    pub fn minimize(
        &mut self,
        clause: &[Lit],
        clauses: &ClauseDatabase,
        reasons: &[Reason],
        level: &[u32],
        current_level: u32,
    ) -> SmallVec<[Lit; 8]> {
        // Reset status for all literals
        for status in &mut self.status {
            *status = LitStatus::Unknown;
        }

        // Mark all literals in the clause that are at the current decision level
        let mut abstract_level = 0u32;
        for &lit in clause {
            let var = lit.var();
            if level[var.index()] == current_level {
                self.status[lit.index()] = LitStatus::Essential;
                abstract_level |= 1 << (level[var.index()] & 31);
            }
        }

        // Try to remove each literal
        let mut result = SmallVec::new();
        let mut removed = 0;

        for &lit in clause {
            let var = lit.var();

            // Skip if at decision level 0 (always keep)
            if level[var.index()] == 0 {
                result.push(lit);
                continue;
            }

            // Skip if already marked as essential
            if self.status[lit.index()] == LitStatus::Essential {
                result.push(lit);
                continue;
            }

            // Check if this literal is redundant
            if self.is_redundant(lit, clauses, reasons, level, abstract_level, current_level) {
                self.status[lit.index()] = LitStatus::Redundant;
                removed += 1;
                self.stats.literals_removed += 1;
            } else {
                self.status[lit.index()] = LitStatus::Essential;
                result.push(lit);
            }
        }

        if removed > 0 {
            self.stats.clauses_minimized += 1;
        }

        result
    }

    /// Check if a literal is redundant using recursive analysis
    fn is_redundant(
        &mut self,
        lit: Lit,
        clauses: &ClauseDatabase,
        reasons: &[Reason],
        level: &[u32],
        abstract_level: u32,
        _current_level: u32,
    ) -> bool {
        self.stats.recursive_checks += 1;

        self.stack.clear();
        self.stack.push(lit);

        while let Some(current_lit) = self.stack.pop() {
            let var = current_lit.var();

            // Check cached status
            match self.status[current_lit.index()] {
                LitStatus::Redundant => continue,
                LitStatus::Essential => return false,
                LitStatus::Unknown => {}
            }

            // Get the reason for this literal
            let reason = &reasons[var.index()];

            match reason {
                Reason::Decision => {
                    // This is a decision literal, not redundant
                    return false;
                }
                Reason::Theory => {
                    // Theory propagation - treat as non-redundant for now
                    return false;
                }
                Reason::Propagation(clause_id) => {
                    // Check all literals in the reason clause
                    let Some(clause) = clauses.get(*clause_id) else {
                        return false;
                    };

                    for &reason_lit in &clause.lits {
                        // Skip the implied literal itself
                        if reason_lit.var() == var {
                            continue;
                        }

                        let reason_var = reason_lit.var();
                        let reason_level = level[reason_var.index()];

                        if reason_level == 0 {
                            // Unit literal, continue
                            continue;
                        }

                        // Check if the level is compatible
                        if (abstract_level & (1 << (reason_level & 31))) == 0 {
                            return false;
                        }

                        // Add to stack for recursive check
                        if self.status[reason_lit.index()] == LitStatus::Unknown {
                            self.stack.push(reason_lit);
                        }
                    }
                }
            }

            // Mark this literal as checked
            self.status[current_lit.index()] = LitStatus::Redundant;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clause::ClauseDatabase;
    use crate::literal::{Lit, Var};
    use crate::trail::Reason;

    #[test]
    fn test_recursive_minimizer_creation() {
        let minimizer = RecursiveMinimizer::new(10);
        assert_eq!(minimizer.status.len(), 20); // 2 * num_vars
    }

    #[test]
    fn test_recursive_minimizer_resize() {
        let mut minimizer = RecursiveMinimizer::new(10);
        minimizer.resize(20);
        assert_eq!(minimizer.status.len(), 40);
    }

    #[test]
    fn test_recursive_minimizer_stats() {
        let minimizer = RecursiveMinimizer::new(10);
        let stats = minimizer.stats();
        assert_eq!(stats.clauses_minimized, 0);
        assert_eq!(stats.literals_removed, 0);
        assert_eq!(stats.recursive_checks, 0);
    }

    #[test]
    fn test_minimize_empty_clause() {
        let mut minimizer = RecursiveMinimizer::new(10);
        let db = ClauseDatabase::new();
        let reasons = vec![Reason::Decision; 10];
        let level = vec![0; 10];

        let clause: Vec<Lit> = vec![];
        let result = minimizer.minimize(&clause, &db, &reasons, &level, 0);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_minimize_unit_clause() {
        let mut minimizer = RecursiveMinimizer::new(10);
        let db = ClauseDatabase::new();
        let reasons = vec![Reason::Decision; 10];
        let level = vec![0; 10];

        let clause = vec![Lit::pos(Var::new(0))];
        let result = minimizer.minimize(&clause, &db, &reasons, &level, 0);
        assert_eq!(result.len(), 1);
    }
}
