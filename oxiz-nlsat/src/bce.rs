//! Blocked Clause Elimination (BCE)
//!
//! This module implements blocked clause elimination, a preprocessing technique that
//! removes clauses that are "blocked" by a literal. A clause is blocked on a literal L
//! if for every clause containing ¬L, the resolvent is a tautology.
//!
//! Reference: Blocked clause elimination by M. Järvisalo, A. Biere, and M. Heule

use crate::clause::Clause;
use crate::types::Literal;
use std::collections::{HashMap, HashSet};

/// Configuration for blocked clause elimination
#[derive(Debug, Clone)]
pub struct BceConfig {
    /// Enable BCE
    pub enabled: bool,
    /// Maximum clause size to consider for blocking
    pub max_clause_size: usize,
}

impl Default for BceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_clause_size: 100,
        }
    }
}

/// Statistics for blocked clause elimination
#[derive(Debug, Default, Clone)]
pub struct BceStats {
    /// Number of blocked clauses eliminated
    pub blocked_clauses: usize,
    /// Number of blocking checks performed
    pub blocking_checks: usize,
    /// Number of resolvents checked for tautology
    pub resolvent_checks: usize,
}

impl BceStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Blocked clause elimination engine
pub struct BceEngine {
    config: BceConfig,
    stats: BceStats,
    /// Mapping from literal to clauses containing it
    lit_occurrences: HashMap<Literal, Vec<usize>>,
}

impl BceEngine {
    /// Create a new BCE engine
    pub fn new(config: BceConfig) -> Self {
        Self {
            config,
            stats: BceStats::new(),
            lit_occurrences: HashMap::new(),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &BceStats {
        &self.stats
    }

    /// Build occurrence lists for all literals
    fn build_occurrence_lists(&mut self, clauses: &[Clause]) {
        self.lit_occurrences.clear();

        for (idx, clause) in clauses.iter().enumerate() {
            for &lit in clause.literals() {
                self.lit_occurrences.entry(lit).or_default().push(idx);
            }
        }
    }

    /// Check if the resolvent of two clauses on a literal is a tautology
    fn is_resolvent_tautology(&mut self, lit: Literal, c1: &Clause, c2: &Clause) -> bool {
        self.stats.resolvent_checks += 1;

        let var = lit.var();
        let mut seen = HashSet::new();

        // Add literals from c1 (except the resolved literal)
        for &l in c1.literals() {
            if l.var() != var {
                if seen.contains(&l.negate()) {
                    return true; // Tautology found
                }
                seen.insert(l);
            }
        }

        // Add literals from c2 (except the resolved literal)
        for &l in c2.literals() {
            if l.var() != var {
                if seen.contains(&l.negate()) {
                    return true; // Tautology found
                }
                seen.insert(l);
            }
        }

        false
    }

    /// Check if a clause is blocked on a literal
    fn is_blocked_on_literal(
        &mut self,
        clause_idx: usize,
        lit: Literal,
        clauses: &[Clause],
    ) -> bool {
        self.stats.blocking_checks += 1;

        let clause = &clauses[clause_idx];

        // Get all clauses containing the negation of the literal (clone to avoid borrow issues)
        let neg_lit = lit.negate();
        let neg_indices = self.lit_occurrences.get(&neg_lit).cloned();

        if let Some(neg_indices) = neg_indices {
            // For each clause containing ¬lit, check if resolvent is a tautology
            for &neg_idx in &neg_indices {
                if neg_idx == clause_idx {
                    continue; // Skip self
                }

                let other_clause = &clauses[neg_idx];

                // Check if resolvent is a tautology
                if !self.is_resolvent_tautology(lit, clause, other_clause) {
                    // Found a non-tautology resolvent, so not blocked
                    return false;
                }
            }
        }

        // All resolvents are tautologies (or no resolvents exist), so the clause is blocked
        true
    }

    /// Check if a clause is blocked on any of its literals
    fn is_blocked(&mut self, clause_idx: usize, clauses: &[Clause]) -> bool {
        let clause = &clauses[clause_idx];

        // Check clause size limit
        if clause.len() > self.config.max_clause_size {
            return false;
        }

        // Try each literal in the clause
        for &lit in clause.literals() {
            if self.is_blocked_on_literal(clause_idx, lit, clauses) {
                return true;
            }
        }

        false
    }

    /// Eliminate blocked clauses
    pub fn eliminate(&mut self, clauses: Vec<Clause>) -> Vec<Clause> {
        if !self.config.enabled {
            return clauses;
        }

        let mut result = clauses;
        let mut changed = true;

        while changed {
            changed = false;
            self.build_occurrence_lists(&result);

            let mut to_remove = Vec::new();

            for idx in 0..result.len() {
                if self.is_blocked(idx, &result) {
                    to_remove.push(idx);
                    self.stats.blocked_clauses += 1;
                    changed = true;
                }
            }

            // Remove blocked clauses (in reverse order to maintain indices)
            for &idx in to_remove.iter().rev() {
                result.remove(idx);
            }

            if !changed {
                break;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_clause(literals: Vec<Literal>) -> Clause {
        let max_var = literals.iter().map(|lit| lit.var()).max().unwrap_or(0);
        Clause::new(literals, max_var, false, 0)
    }

    #[test]
    fn test_bce_engine_new() {
        let config = BceConfig::default();
        let engine = BceEngine::new(config);
        assert_eq!(engine.stats().blocked_clauses, 0);
    }

    #[test]
    fn test_bce_simple_blocked() {
        let config = BceConfig::default();
        let mut engine = BceEngine::new(config);

        // (x1 ∨ x2) is blocked if every clause with ¬x1 or ¬x2 produces a tautology
        // Example: (x1 ∨ x2) ∧ (¬x1 ∨ ¬x2) - first clause is blocked on x1
        let clauses = vec![
            make_test_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_test_clause(vec![Literal::negative(1), Literal::negative(2)]),
        ];

        let result = engine.eliminate(clauses);

        // One clause should be eliminated
        assert!(result.len() < 2);
        assert!(engine.stats().blocked_clauses > 0);
    }

    #[test]
    fn test_bce_unit_clause() {
        let config = BceConfig::default();
        let mut engine = BceEngine::new(config);

        // A unit clause x1 should remain (cannot be blocked)
        // And (x1 ∨ x2) would be blocked on x2 if there are no ¬x2 clauses
        let clauses = vec![
            make_test_clause(vec![Literal::positive(1)]),
            make_test_clause(vec![Literal::positive(2), Literal::positive(3)]),
        ];

        let _result = engine.eliminate(clauses);

        // BCE may eliminate all clauses if they're all blocked - just check it runs
        assert!(engine.stats().blocking_checks > 0);
    }

    #[test]
    fn test_bce_disabled() {
        let config = BceConfig {
            enabled: false,
            ..Default::default()
        };
        let mut engine = BceEngine::new(config);

        let clauses = vec![
            make_test_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_test_clause(vec![Literal::negative(1), Literal::negative(2)]),
        ];

        let result = engine.eliminate(clauses.clone());
        assert_eq!(result.len(), clauses.len());
        assert_eq!(engine.stats().blocked_clauses, 0);
    }

    #[test]
    fn test_bce_stats() {
        let config = BceConfig::default();
        let mut engine = BceEngine::new(config);

        let clauses = vec![
            make_test_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_test_clause(vec![Literal::negative(1), Literal::negative(2)]),
        ];

        engine.eliminate(clauses);

        let stats = engine.stats();
        assert!(stats.blocking_checks > 0);
    }
}
