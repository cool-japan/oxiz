//! Clause Subsumption
//!
//! This module implements clause subsumption checking and backward subsumption.
//! A clause C subsumes clause D if C ⊆ D (all literals in C appear in D).
//! Subsumed clauses can be safely removed from the clause database.
//!
//! Reference: Efficient clause subsumption in modern SAT solvers

use crate::clause::Clause;
use crate::types::Literal;
use std::collections::{HashMap, HashSet};

/// Configuration for subsumption
#[derive(Debug, Clone)]
pub struct SubsumptionConfig {
    /// Enable subsumption checking
    pub enabled: bool,
    /// Maximum clause size for subsumption checks (performance limit)
    pub max_clause_size: usize,
    /// Enable backward subsumption (check if new clause subsumes old ones)
    pub backward_subsumption: bool,
}

impl Default for SubsumptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_clause_size: 100,
            backward_subsumption: true,
        }
    }
}

/// Statistics for subsumption
#[derive(Debug, Default, Clone)]
pub struct SubsumptionStats {
    /// Number of clauses subsumed by new clauses
    pub forward_subsumed: usize,
    /// Number of new clauses subsumed by existing clauses
    pub backward_subsumed: usize,
    /// Number of subsumption checks performed
    pub checks: usize,
    /// Number of clauses removed by subsumption
    pub clauses_removed: usize,
}

impl SubsumptionStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Subsumption checker
pub struct SubsumptionChecker {
    config: SubsumptionConfig,
    stats: SubsumptionStats,
    /// Mapping from literal to clause indices containing it
    lit_to_clauses: HashMap<Literal, Vec<usize>>,
}

impl SubsumptionChecker {
    /// Create a new subsumption checker
    pub fn new(config: SubsumptionConfig) -> Self {
        Self {
            config,
            stats: SubsumptionStats::new(),
            lit_to_clauses: HashMap::new(),
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &SubsumptionStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats.reset();
    }

    /// Build occurrence list for all literals
    fn build_occurrence_list(&mut self, clauses: &[Clause]) {
        self.lit_to_clauses.clear();

        for (idx, clause) in clauses.iter().enumerate() {
            for &lit in clause.literals() {
                self.lit_to_clauses.entry(lit).or_default().push(idx);
            }
        }
    }

    /// Check if clause c1 subsumes clause c2 (c1 ⊆ c2)
    fn subsumes(&mut self, c1: &Clause, c2: &Clause) -> bool {
        self.stats.checks += 1;

        // Quick check: c1 cannot subsume c2 if c1 is larger
        if c1.len() > c2.len() {
            return false;
        }

        // Convert c2 literals to a set for fast lookup
        let c2_lits: HashSet<Literal> = c2.literals().iter().copied().collect();

        // Check if all literals in c1 are present in c2
        for &lit in c1.literals() {
            if !c2_lits.contains(&lit) {
                return false;
            }
        }

        true
    }

    /// Check if a new clause is subsumed by any existing clause
    /// Returns true if the new clause is subsumed (and should be discarded)
    pub fn is_subsumed_by_existing(&mut self, new_clause: &Clause, existing: &[Clause]) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Skip very large clauses
        if new_clause.len() > self.config.max_clause_size {
            return false;
        }

        // Find candidate clauses: those that share at least one literal with new_clause
        // and are not larger than new_clause
        let mut candidates = HashSet::new();

        for &lit in new_clause.literals() {
            if let Some(clause_indices) = self.lit_to_clauses.get(&lit) {
                for &idx in clause_indices {
                    if existing[idx].len() <= new_clause.len() {
                        candidates.insert(idx);
                    }
                }
            }
        }

        // Check each candidate
        for &idx in &candidates {
            if self.subsumes(&existing[idx], new_clause) {
                self.stats.forward_subsumed += 1;
                return true;
            }
        }

        false
    }

    /// Find all clauses subsumed by a new clause
    /// Returns indices of clauses that should be removed
    pub fn find_subsumed_clauses(
        &mut self,
        new_clause: &Clause,
        existing: &[Clause],
    ) -> Vec<usize> {
        if !self.config.enabled || !self.config.backward_subsumption {
            return Vec::new();
        }

        // Skip very large clauses
        if new_clause.len() > self.config.max_clause_size {
            return Vec::new();
        }

        let mut subsumed = Vec::new();

        // Find candidate clauses: those that contain all literals from new_clause
        // For efficiency, we look at clauses containing the least frequent literal
        let mut min_occurrences = usize::MAX;
        let mut rarest_lit = None;

        for &lit in new_clause.literals() {
            if let Some(occurrences) = self.lit_to_clauses.get(&lit)
                && occurrences.len() < min_occurrences
            {
                min_occurrences = occurrences.len();
                rarest_lit = Some(lit);
            }
        }

        if let Some(lit) = rarest_lit
            && let Some(clause_indices) = self.lit_to_clauses.get(&lit).cloned()
        {
            for &idx in &clause_indices {
                // new_clause can only subsume clauses that are at least as large
                if existing[idx].len() >= new_clause.len()
                    && self.subsumes(new_clause, &existing[idx])
                {
                    subsumed.push(idx);
                    self.stats.backward_subsumed += 1;
                }
            }
        }

        self.stats.clauses_removed += subsumed.len();
        subsumed
    }

    /// Add a clause to the occurrence list (for incremental updates)
    pub fn add_clause(&mut self, clause: &Clause, idx: usize) {
        for &lit in clause.literals() {
            self.lit_to_clauses.entry(lit).or_default().push(idx);
        }
    }

    /// Remove a clause from the occurrence list
    pub fn remove_clause(&mut self, clause: &Clause, idx: usize) {
        for &lit in clause.literals() {
            if let Some(indices) = self.lit_to_clauses.get_mut(&lit) {
                indices.retain(|&i| i != idx);
            }
        }
    }

    /// Perform full subsumption elimination on a clause set
    pub fn eliminate_subsumed(&mut self, clauses: Vec<Clause>) -> Vec<Clause> {
        if !self.config.enabled {
            return clauses;
        }

        self.build_occurrence_list(&clauses);

        let mut keep = vec![true; clauses.len()];
        let mut changed = true;

        while changed {
            changed = false;

            for i in 0..clauses.len() {
                if !keep[i] {
                    continue;
                }

                for j in (i + 1)..clauses.len() {
                    if !keep[j] {
                        continue;
                    }

                    if self.subsumes(&clauses[i], &clauses[j]) {
                        keep[j] = false;
                        self.stats.clauses_removed += 1;
                        changed = true;
                    } else if self.subsumes(&clauses[j], &clauses[i]) {
                        keep[i] = false;
                        self.stats.clauses_removed += 1;
                        changed = true;
                        break;
                    }
                }
            }
        }

        clauses
            .into_iter()
            .zip(keep.iter())
            .filter_map(|(c, &k)| if k { Some(c) } else { None })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clause(literals: Vec<Literal>) -> Clause {
        let max_var = literals.iter().map(|lit| lit.var()).max().unwrap_or(0);
        Clause::new(literals, max_var, false, 0)
    }

    #[test]
    fn test_subsumption_checker_new() {
        let config = SubsumptionConfig::default();
        let checker = SubsumptionChecker::new(config);
        assert_eq!(checker.stats().checks, 0);
    }

    #[test]
    fn test_subsumes_basic() {
        let config = SubsumptionConfig::default();
        let mut checker = SubsumptionChecker::new(config);

        // (x1) subsumes (x1 ∨ x2)
        let c1 = make_clause(vec![Literal::positive(1)]);
        let c2 = make_clause(vec![Literal::positive(1), Literal::positive(2)]);

        assert!(checker.subsumes(&c1, &c2));
        assert!(!checker.subsumes(&c2, &c1));
    }

    #[test]
    fn test_subsumes_no_subset() {
        let config = SubsumptionConfig::default();
        let mut checker = SubsumptionChecker::new(config);

        // (x1 ∨ x2) does not subsume (x1 ∨ x3)
        let c1 = make_clause(vec![Literal::positive(1), Literal::positive(2)]);
        let c2 = make_clause(vec![Literal::positive(1), Literal::positive(3)]);

        assert!(!checker.subsumes(&c1, &c2));
        assert!(!checker.subsumes(&c2, &c1));
    }

    #[test]
    fn test_is_subsumed_by_existing() {
        let config = SubsumptionConfig::default();
        let mut checker = SubsumptionChecker::new(config);

        let existing = vec![
            make_clause(vec![Literal::positive(1)]),
            make_clause(vec![Literal::positive(2), Literal::positive(3)]),
        ];

        checker.build_occurrence_list(&existing);

        // (x1 ∨ x2) is subsumed by (x1)
        let new_clause = make_clause(vec![Literal::positive(1), Literal::positive(2)]);
        assert!(checker.is_subsumed_by_existing(&new_clause, &existing));

        // (x1 ∨ x4) is subsumed by (x1)
        let new_clause2 = make_clause(vec![Literal::positive(1), Literal::positive(4)]);
        assert!(checker.is_subsumed_by_existing(&new_clause2, &existing));

        // (x4 ∨ x5) is not subsumed
        let new_clause3 = make_clause(vec![Literal::positive(4), Literal::positive(5)]);
        assert!(!checker.is_subsumed_by_existing(&new_clause3, &existing));
    }

    #[test]
    fn test_find_subsumed_clauses() {
        let config = SubsumptionConfig::default();
        let mut checker = SubsumptionChecker::new(config);

        let existing = vec![
            make_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_clause(vec![Literal::positive(1), Literal::positive(3)]),
            make_clause(vec![Literal::positive(2), Literal::positive(3)]),
        ];

        checker.build_occurrence_list(&existing);

        // (x1) subsumes clauses 0 and 1
        let new_clause = make_clause(vec![Literal::positive(1)]);
        let subsumed = checker.find_subsumed_clauses(&new_clause, &existing);

        assert_eq!(subsumed.len(), 2);
        assert!(subsumed.contains(&0));
        assert!(subsumed.contains(&1));
    }

    #[test]
    fn test_eliminate_subsumed() {
        let config = SubsumptionConfig::default();
        let mut checker = SubsumptionChecker::new(config);

        let clauses = vec![
            make_clause(vec![Literal::positive(1)]),
            make_clause(vec![Literal::positive(1), Literal::positive(2)]),
            make_clause(vec![Literal::positive(1), Literal::positive(3)]),
            make_clause(vec![Literal::positive(4)]),
        ];

        let result = checker.eliminate_subsumed(clauses);

        // Should keep only (x1) and (x4)
        assert_eq!(result.len(), 2);
        assert!(
            result
                .iter()
                .any(|c| c.len() == 1 && c.literals()[0] == Literal::positive(1))
        );
        assert!(
            result
                .iter()
                .any(|c| c.len() == 1 && c.literals()[0] == Literal::positive(4))
        );
    }

    #[test]
    fn test_subsumption_disabled() {
        let config = SubsumptionConfig {
            enabled: false,
            ..Default::default()
        };
        let mut checker = SubsumptionChecker::new(config);

        let existing = vec![make_clause(vec![Literal::positive(1)])];
        checker.build_occurrence_list(&existing);

        let new_clause = make_clause(vec![Literal::positive(1), Literal::positive(2)]);
        assert!(!checker.is_subsumed_by_existing(&new_clause, &existing));
    }

    #[test]
    fn test_subsumption_stats() {
        let config = SubsumptionConfig::default();
        let mut checker = SubsumptionChecker::new(config);

        let c1 = make_clause(vec![Literal::positive(1)]);
        let c2 = make_clause(vec![Literal::positive(1), Literal::positive(2)]);

        checker.subsumes(&c1, &c2);

        assert!(checker.stats().checks > 0);
    }
}
