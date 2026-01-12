//! Asymmetric Literal Addition (ALA) for clause strengthening.
//!
//! ALA is a preprocessing technique that strengthens clauses by performing
//! unit propagation on the negation of a clause and checking if any literals
//! can be added to strengthen it.
//!
//! Reference: "Asymmetric Branching" by Heule et al. (SAT 2011)

use crate::types::{Lbool, Literal};
use rustc_hash::FxHashSet;
use std::collections::HashMap;

/// Statistics for ALA operations.
#[derive(Debug, Clone, Default)]
pub struct AlaStats {
    /// Number of clauses processed.
    pub clauses_processed: u64,
    /// Number of literals added.
    pub literals_added: u64,
    /// Number of clauses strengthened.
    pub clauses_strengthened: u64,
}

/// Configuration for ALA.
#[derive(Debug, Clone)]
pub struct AlaConfig {
    /// Enable ALA.
    pub enabled: bool,
    /// Maximum clause size for ALA.
    pub max_clause_size: usize,
    /// Maximum propagation depth.
    pub max_propagation_depth: usize,
}

impl Default for AlaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_clause_size: 30,
            max_propagation_depth: 100,
        }
    }
}

/// Simple assignment for ALA unit propagation.
#[derive(Debug, Clone)]
struct AlaAssignment {
    /// Current assignments (var -> value).
    assignments: HashMap<u32, bool>,
    /// Trail of assignments (for backtracking).
    trail: Vec<Literal>,
}

impl AlaAssignment {
    fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            trail: Vec::new(),
        }
    }

    fn assign(&mut self, lit: Literal) {
        let var = lit.var();
        let value = !lit.is_negated();
        self.assignments.insert(var, value);
        self.trail.push(lit);
    }

    #[allow(dead_code)]
    fn is_assigned(&self, var: u32) -> bool {
        self.assignments.contains_key(&var)
    }

    fn value(&self, lit: Literal) -> Lbool {
        if let Some(&value) = self.assignments.get(&lit.var()) {
            if value != lit.is_negated() {
                Lbool::True
            } else {
                Lbool::False
            }
        } else {
            Lbool::Undef
        }
    }

    fn clear(&mut self) {
        self.assignments.clear();
        self.trail.clear();
    }

    #[allow(dead_code)]
    fn level(&self) -> usize {
        self.trail.len()
    }
}

/// Asymmetric Literal Addition engine.
pub struct AlaEngine {
    config: AlaConfig,
    stats: AlaStats,
    assignment: AlaAssignment,
}

impl AlaEngine {
    /// Create a new ALA engine with default configuration.
    pub fn new() -> Self {
        Self::with_config(AlaConfig::default())
    }

    /// Create a new ALA engine with the given configuration.
    pub fn with_config(config: AlaConfig) -> Self {
        Self {
            config,
            stats: AlaStats::default(),
            assignment: AlaAssignment::new(),
        }
    }

    /// Get the statistics.
    pub fn stats(&self) -> &AlaStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &AlaConfig {
        &self.config
    }

    /// Attempt to strengthen a clause using ALA.
    ///
    /// Returns the strengthened clause (possibly with added literals).
    pub fn strengthen_clause(
        &mut self,
        clause: &[Literal],
        all_clauses: &[Vec<Literal>],
    ) -> Vec<Literal> {
        if !self.config.enabled {
            return clause.to_vec();
        }

        if clause.len() > self.config.max_clause_size {
            return clause.to_vec();
        }

        self.stats.clauses_processed += 1;

        // Try to find literals that can be added
        let mut result = clause.to_vec();
        let original_len = result.len();

        // Collect all literals that appear in other clauses
        let mut candidate_literals = FxHashSet::default();
        for other_clause in all_clauses {
            for &lit in other_clause {
                candidate_literals.insert(lit);
            }
        }

        // Remove literals already in the clause
        for &lit in clause {
            candidate_literals.remove(&lit);
            candidate_literals.remove(&lit.negate());
        }

        // For each candidate literal, check if it can be added asymmetrically
        for &candidate in &candidate_literals {
            if self.can_add_literal(clause, candidate, all_clauses) {
                result.push(candidate);
                self.stats.literals_added += 1;

                // Limit the number of literals added
                if result.len() >= original_len + 5 {
                    break;
                }
            }
        }

        if result.len() > original_len {
            self.stats.clauses_strengthened += 1;
        }

        result
    }

    /// Check if a literal can be added to a clause asymmetrically.
    ///
    /// This is done by assuming all literals in the clause are false,
    /// and checking if unit propagation derives the candidate literal.
    fn can_add_literal(
        &mut self,
        clause: &[Literal],
        candidate: Literal,
        all_clauses: &[Vec<Literal>],
    ) -> bool {
        // Reset assignment
        self.assignment.clear();

        // Assume all literals in the clause are false
        for &lit in clause {
            self.assignment.assign(lit.negate());
        }

        // Perform unit propagation
        let mut propagations = 0;
        loop {
            if propagations >= self.config.max_propagation_depth {
                break;
            }

            let mut propagated = false;

            for clause_lits in all_clauses {
                // Check if this clause is unit
                let mut unassigned_lit = None;
                let mut num_unassigned = 0;
                let mut is_satisfied = false;

                for &lit in clause_lits {
                    match self.assignment.value(lit) {
                        Lbool::True => {
                            is_satisfied = true;
                            break;
                        }
                        Lbool::False => {
                            // Already false, continue
                        }
                        Lbool::Undef => {
                            num_unassigned += 1;
                            unassigned_lit = Some(lit);
                        }
                    }
                }

                if !is_satisfied && num_unassigned == 1 {
                    // Unit clause - propagate
                    if let Some(lit) = unassigned_lit {
                        self.assignment.assign(lit);
                        propagated = true;
                        propagations += 1;

                        // Check if we derived the candidate
                        if lit == candidate {
                            return true;
                        }
                    }
                }

                if !is_satisfied && num_unassigned == 0 {
                    // Conflict - cannot add literal
                    return false;
                }
            }

            if !propagated {
                break;
            }
        }

        false
    }
}

impl Default for AlaEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BoolVar;

    fn make_literal(var: BoolVar, sign: bool) -> Literal {
        Literal::new(var, sign)
    }

    #[test]
    fn test_ala_engine_new() {
        let ala = AlaEngine::new();
        assert!(ala.config().enabled);
        assert_eq!(ala.stats().clauses_processed, 0);
        assert_eq!(ala.stats().literals_added, 0);
    }

    #[test]
    fn test_strengthen_clause_no_change() {
        let mut ala = AlaEngine::new();

        let clause = vec![make_literal(0, true), make_literal(1, true)];
        let all_clauses = vec![clause.clone()];

        let result = ala.strengthen_clause(&clause, &all_clauses);

        // Should not change for simple case
        assert_eq!(result.len(), clause.len());
    }

    #[test]
    fn test_strengthen_clause_disabled() {
        let config = AlaConfig {
            enabled: false,
            ..Default::default()
        };
        let mut ala = AlaEngine::with_config(config);

        let clause = vec![make_literal(0, true)];
        let all_clauses = vec![clause.clone()];

        let result = ala.strengthen_clause(&clause, &all_clauses);

        assert_eq!(result, clause);
    }

    #[test]
    fn test_ala_assignment() {
        let mut assign = AlaAssignment::new();

        let lit = make_literal(0, true);
        assign.assign(lit);

        assert!(assign.is_assigned(0));
        assert_eq!(assign.value(lit), Lbool::True);
        assert_eq!(assign.value(lit.negate()), Lbool::False);
    }

    #[test]
    fn test_ala_assignment_undef() {
        let assign = AlaAssignment::new();

        let lit = make_literal(0, true);
        assert!(!assign.is_assigned(0));
        assert_eq!(assign.value(lit), Lbool::Undef);
    }

    #[test]
    fn test_ala_assignment_clear() {
        let mut assign = AlaAssignment::new();

        assign.assign(make_literal(0, true));
        assign.assign(make_literal(1, false));

        assert!(assign.is_assigned(0));
        assert!(assign.is_assigned(1));

        assign.clear();

        assert!(!assign.is_assigned(0));
        assert!(!assign.is_assigned(1));
    }

    #[test]
    fn test_ala_stats() {
        let mut ala = AlaEngine::new();

        let clause = vec![make_literal(0, true)];
        let all_clauses = vec![clause.clone()];

        ala.strengthen_clause(&clause, &all_clauses);

        assert_eq!(ala.stats().clauses_processed, 1);
    }
}
