//! Bounded Variable Elimination (BVE)
//!
//! This module implements bounded variable elimination, a preprocessing technique that
//! eliminates variables by resolving all clauses containing the variable, but only when
//! the number of resulting clauses doesn't exceed a specified bound.
//!
//! Reference: SAT preprocessing techniques, particularly MiniSat's simplification.

use crate::clause::{Clause, ClauseId};
use crate::types::{BoolVar, Literal};
use oxiz_math::polynomial::Var;
use std::collections::{HashMap, HashSet};

/// Configuration for bounded variable elimination
#[derive(Debug, Clone)]
pub struct BveConfig {
    /// Maximum number of new clauses allowed when eliminating a variable
    pub max_clause_increase: usize,
    /// Maximum clause size to consider for elimination
    pub max_clause_size: usize,
    /// Enable variable elimination
    pub enabled: bool,
}

impl Default for BveConfig {
    fn default() -> Self {
        Self {
            max_clause_increase: 100,
            max_clause_size: 50,
            enabled: true,
        }
    }
}

/// Statistics for bounded variable elimination
#[derive(Debug, Default, Clone)]
pub struct BveStats {
    /// Number of variables eliminated
    pub vars_eliminated: usize,
    /// Number of clauses removed
    pub clauses_removed: usize,
    /// Number of clauses added
    pub clauses_added: usize,
    /// Number of elimination attempts
    pub elimination_attempts: usize,
    /// Number of times elimination was blocked by clause increase
    pub blocked_by_increase: usize,
}

impl BveStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Bounded variable elimination engine
pub struct BveEngine {
    config: BveConfig,
    stats: BveStats,
    /// Variables that have been eliminated
    eliminated_vars: HashSet<BoolVar>,
    /// Mapping from variable to clauses containing it (positive)
    pos_occurrences: HashMap<BoolVar, Vec<usize>>,
    /// Mapping from variable to clauses containing it (negative)
    neg_occurrences: HashMap<BoolVar, Vec<usize>>,
    /// Next clause ID
    next_clause_id: ClauseId,
}

impl BveEngine {
    /// Create a new BVE engine
    pub fn new(config: BveConfig) -> Self {
        Self {
            config,
            stats: BveStats::new(),
            eliminated_vars: HashSet::new(),
            pos_occurrences: HashMap::new(),
            neg_occurrences: HashMap::new(),
            next_clause_id: 0,
        }
    }

    /// Create a clause from literals
    fn make_clause(&mut self, literals: Vec<Literal>) -> Clause {
        let max_var = literals.iter().map(|lit| lit.var()).max().unwrap_or(0);
        let id = self.next_clause_id;
        self.next_clause_id = id + 1;
        Clause::new(literals, max_var, false, id)
    }

    /// Get statistics
    pub fn stats(&self) -> &BveStats {
        &self.stats
    }

    /// Check if a variable has been eliminated
    pub fn is_eliminated(&self, var: BoolVar) -> bool {
        self.eliminated_vars.contains(&var)
    }

    /// Build occurrence lists for all variables
    fn build_occurrence_lists(&mut self, clauses: &[Clause]) {
        self.pos_occurrences.clear();
        self.neg_occurrences.clear();

        for (idx, clause) in clauses.iter().enumerate() {
            for &lit in clause.literals() {
                let var = lit.var();
                if lit.is_positive() {
                    self.pos_occurrences.entry(var).or_default().push(idx);
                } else {
                    self.neg_occurrences.entry(var).or_default().push(idx);
                }
            }
        }
    }

    /// Estimate the number of resolvents when eliminating a variable
    fn estimate_resolvents(&self, var: BoolVar, clauses: &[Clause]) -> usize {
        // Worst case: every positive clause resolves with every negative clause
        // But we filter out tautologies
        let mut count = 0;
        let pos_indices = self.pos_occurrences.get(&var);
        let neg_indices = self.neg_occurrences.get(&var);

        if let (Some(pos_idx), Some(neg_idx)) = (pos_indices, neg_indices) {
            for &p_idx in pos_idx {
                for &n_idx in neg_idx {
                    // Create resolvent and check if it's a tautology
                    if !self.is_resolvent_tautology(var, &clauses[p_idx], &clauses[n_idx]) {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Check if the resolvent of two clauses would be a tautology
    fn is_resolvent_tautology(
        &self,
        var: BoolVar,
        pos_clause: &Clause,
        neg_clause: &Clause,
    ) -> bool {
        // Collect all literals from both clauses except the variable being eliminated
        let mut literals = HashSet::new();

        for &lit in pos_clause.literals() {
            if lit.var() != var {
                if literals.contains(&lit.negate()) {
                    return true; // Tautology detected
                }
                literals.insert(lit);
            }
        }

        for &lit in neg_clause.literals() {
            if lit.var() != var {
                if literals.contains(&lit.negate()) {
                    return true; // Tautology detected
                }
                literals.insert(lit);
            }
        }

        false
    }

    /// Create the resolvent of two clauses on a variable
    fn resolve(
        &mut self,
        var: BoolVar,
        pos_clause: &Clause,
        neg_clause: &Clause,
    ) -> Option<Clause> {
        let mut literals = Vec::new();
        let mut seen = HashSet::new();

        // Add literals from positive clause (except the resolved variable)
        for &lit in pos_clause.literals() {
            if lit.var() != var && !seen.contains(&lit) {
                if seen.contains(&lit.negate()) {
                    return None; // Tautology
                }
                literals.push(lit);
                seen.insert(lit);
            }
        }

        // Add literals from negative clause (except the resolved variable)
        for &lit in neg_clause.literals() {
            if lit.var() != var && !seen.contains(&lit) {
                if seen.contains(&lit.negate()) {
                    return None; // Tautology
                }
                literals.push(lit);
                seen.insert(lit);
            }
        }

        Some(self.make_clause(literals))
    }

    /// Try to eliminate a variable
    fn try_eliminate(&mut self, var: BoolVar, clauses: &[Clause]) -> Option<Vec<Clause>> {
        if !self.config.enabled {
            return None;
        }

        self.stats.elimination_attempts += 1;

        // Get clauses containing this variable (clone indices to avoid borrow issues)
        let pos_indices = self.pos_occurrences.get(&var)?.clone();
        let neg_indices = self.neg_occurrences.get(&var)?.clone();

        let total_clauses = pos_indices.len() + neg_indices.len();

        // Check clause size limit
        for &idx in pos_indices.iter().chain(neg_indices.iter()) {
            if clauses[idx].len() > self.config.max_clause_size {
                return None;
            }
        }

        // Estimate number of resolvents
        let num_resolvents = self.estimate_resolvents(var, clauses);

        // Check if elimination is beneficial
        // Allow elimination even if num_resolvents is 0 (all tautologies)
        if num_resolvents > total_clauses + self.config.max_clause_increase {
            self.stats.blocked_by_increase += 1;
            return None;
        }

        // Perform elimination
        let mut new_clauses = Vec::new();

        for &p_idx in &pos_indices {
            for &n_idx in &neg_indices {
                if let Some(resolvent) = self.resolve(var, &clauses[p_idx], &clauses[n_idx]) {
                    new_clauses.push(resolvent);
                }
            }
        }

        self.stats.vars_eliminated += 1;
        self.stats.clauses_removed += total_clauses;
        self.stats.clauses_added += new_clauses.len();
        self.eliminated_vars.insert(var);

        Some(new_clauses)
    }

    /// Eliminate variables from a set of clauses
    pub fn eliminate(&mut self, clauses: Vec<Clause>) -> Vec<Clause> {
        if !self.config.enabled {
            return clauses;
        }

        let mut result = clauses;
        let mut changed = true;

        while changed {
            changed = false;
            self.build_occurrence_lists(&result);

            // Find all variables
            let mut vars: Vec<Var> = self
                .pos_occurrences
                .keys()
                .chain(self.neg_occurrences.keys())
                .copied()
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();

            // Sort by occurrence count (eliminate variables with fewer occurrences first)
            vars.sort_by_key(|&v| {
                let pos = self.pos_occurrences.get(&v).map_or(0, |v| v.len());
                let neg = self.neg_occurrences.get(&v).map_or(0, |v| v.len());
                pos + neg
            });

            for var in vars {
                if self.is_eliminated(var) {
                    continue;
                }

                if let Some(new_clauses) = self.try_eliminate(var, &result) {
                    // Remove clauses containing the eliminated variable
                    result.retain(|c| !c.literals().iter().any(|lit| lit.var() == var));

                    // Add new resolvents
                    result.extend(new_clauses);

                    changed = true;
                    break; // Rebuild occurrence lists
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Literal;

    fn make_test_clause(literals: Vec<Literal>) -> Clause {
        let max_var = literals
            .iter()
            .map(|lit| {
                // Convert BoolVar to arithmetic Var for clause metadata
                lit.var()
            })
            .max()
            .unwrap_or(0);
        Clause::new(literals, max_var, false, 0)
    }

    #[test]
    fn test_bve_engine_new() {
        let config = BveConfig::default();
        let engine = BveEngine::new(config);
        assert_eq!(engine.stats().vars_eliminated, 0);
    }

    #[test]
    fn test_bve_simple_elimination() {
        let config = BveConfig::default();
        let mut engine = BveEngine::new(config);

        // (x1 ∨ x2) ∧ (¬x1 ∨ x3) → (x2 ∨ x3) after eliminating x1
        let clauses = vec![
            make_test_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_test_clause(vec![Literal::negative(1), Literal::positive(3)]),
        ];

        let result = engine.eliminate(clauses);

        // Should have one resolvent: (x2 ∨ x3)
        assert!(engine.is_eliminated(1));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
    }

    #[test]
    fn test_bve_tautology_elimination() {
        let config = BveConfig::default();
        let mut engine = BveEngine::new(config);

        // (x1 ∨ x2) ∧ (¬x1 ∨ ¬x2) → ∅ (tautology resolvent)
        let clauses = vec![
            make_test_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_test_clause(vec![Literal::negative(1), Literal::negative(2)]),
        ];

        let result = engine.eliminate(clauses);

        // At least one variable should be eliminated since resolvents are tautologies
        // BVE may eliminate either x1 or x2 (or both) depending on order
        assert!(engine.is_eliminated(1) || engine.is_eliminated(2));
        // Result should be empty or very small (tautology resolvents aren't added)
        assert!(result.len() <= 1);
    }

    #[test]
    fn test_bve_disabled() {
        let config = BveConfig {
            enabled: false,
            ..Default::default()
        };
        let mut engine = BveEngine::new(config);

        let clauses = vec![make_test_clause(vec![
            Literal::positive(1),
            Literal::positive(2),
        ])];

        let result = engine.eliminate(clauses.clone());
        assert_eq!(result.len(), clauses.len());
        assert!(!engine.is_eliminated(1));
    }

    #[test]
    fn test_bve_stats() {
        let config = BveConfig::default();
        let mut engine = BveEngine::new(config);

        let clauses = vec![
            make_test_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_test_clause(vec![Literal::negative(1), Literal::positive(3)]),
        ];

        engine.eliminate(clauses);

        let stats = engine.stats();
        assert_eq!(stats.vars_eliminated, 1);
        assert_eq!(stats.clauses_removed, 2);
        assert_eq!(stats.clauses_added, 1);
    }
}
