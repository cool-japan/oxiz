//! Inprocessing techniques for clause database optimization.
//!
//! This module implements various inprocessing techniques that can be
//! applied during the search to simplify the clause database and improve
//! solver performance. These include subsumption checking, clause strengthening,
//! and other optimizations.
//!
//! Reference: Modern SAT solvers like Glucose, CryptoMiniSat

use crate::clause::{Clause, ClauseId};
use crate::types::Literal;
use rustc_hash::FxHashSet;
use std::collections::HashMap;

/// Statistics for inprocessing operations.
#[derive(Debug, Clone, Default)]
pub struct InprocessStats {
    /// Number of subsumed clauses removed.
    pub subsumed_clauses: u64,
    /// Number of clauses strengthened.
    pub strengthened_clauses: u64,
    /// Number of literals removed by strengthening.
    pub literals_removed: u64,
    /// Number of inprocessing calls.
    pub inprocess_calls: u64,
}

/// Configuration for inprocessing.
#[derive(Debug, Clone)]
pub struct InprocessConfig {
    /// Enable subsumption checking.
    pub enable_subsumption: bool,
    /// Enable clause strengthening.
    pub enable_strengthening: bool,
    /// Maximum clause size for subsumption checks.
    pub max_subsumption_size: usize,
    /// Perform inprocessing every N conflicts.
    pub inprocess_interval: u64,
}

impl Default for InprocessConfig {
    fn default() -> Self {
        Self {
            enable_subsumption: true,
            enable_strengthening: true,
            max_subsumption_size: 30,
            inprocess_interval: 5000,
        }
    }
}

/// Inprocessing engine for clause database optimization.
pub struct Inprocessor {
    config: InprocessConfig,
    stats: InprocessStats,
    /// Map from literal to clause IDs containing it.
    literal_watch: HashMap<Literal, Vec<ClauseId>>,
}

impl Inprocessor {
    /// Create a new inprocessor with default configuration.
    pub fn new() -> Self {
        Self::with_config(InprocessConfig::default())
    }

    /// Create a new inprocessor with the given configuration.
    pub fn with_config(config: InprocessConfig) -> Self {
        Self {
            config,
            stats: InprocessStats::default(),
            literal_watch: HashMap::new(),
        }
    }

    /// Get the statistics.
    pub fn stats(&self) -> &InprocessStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &InprocessConfig {
        &self.config
    }

    /// Check if a clause subsumes another clause.
    ///
    /// Clause C subsumes clause D if every literal in C appears in D.
    pub fn subsumes(clause_c: &[Literal], clause_d: &[Literal]) -> bool {
        if clause_c.len() > clause_d.len() {
            return false;
        }

        let d_set: FxHashSet<_> = clause_d.iter().copied().collect();

        clause_c.iter().all(|&lit| d_set.contains(&lit))
    }

    /// Check if clause C self-subsumes clause D at a literal.
    ///
    /// Self-subsumption: C ∨ x subsumes D ∨ ¬x, allowing us to
    /// strengthen D by removing ¬x.
    ///
    /// Returns Some(literal) if self-subsumption is possible, where
    /// literal is the one that can be removed from D.
    pub fn self_subsumes(clause_c: &[Literal], clause_d: &[Literal]) -> Option<Literal> {
        if clause_c.len() > clause_d.len() {
            return None;
        }

        let d_set: FxHashSet<_> = clause_d.iter().copied().collect();

        // Find complementary literals between C and D
        for &lit_c in clause_c {
            let neg_lit_c = lit_c.negate();

            if d_set.contains(&neg_lit_c) {
                // Check if all other literals in C are in D
                let all_match = clause_c
                    .iter()
                    .filter(|&&lit| lit != lit_c)
                    .all(|&lit| d_set.contains(&lit));

                if all_match {
                    return Some(neg_lit_c); // This literal can be removed from D
                }
            }
        }

        None
    }

    /// Find all subsumptions between clauses.
    ///
    /// Returns a list of (subsumed_id, subsuming_id) pairs.
    pub fn find_subsumptions(&mut self, clauses: &[Clause]) -> Vec<(ClauseId, ClauseId)> {
        if !self.config.enable_subsumption {
            return Vec::new();
        }

        let mut subsumptions = Vec::new();

        // Build literal watch map for efficient lookup
        self.literal_watch.clear();
        for clause in clauses {
            if clause.len() > self.config.max_subsumption_size {
                continue;
            }

            for &lit in clause.literals() {
                self.literal_watch.entry(lit).or_default().push(clause.id());
            }
        }

        // Check each clause for subsumptions
        for i in 0..clauses.len() {
            let clause_c = &clauses[i];

            if clause_c.len() > self.config.max_subsumption_size {
                continue;
            }

            // Find candidate clauses that might be subsumed by clause_c
            // by looking at clauses that contain the first literal of C
            if let Some(&first_lit) = clause_c.literals().first()
                && let Some(candidates) = self.literal_watch.get(&first_lit)
            {
                for &candidate_id in candidates {
                    if candidate_id == clause_c.id() {
                        continue; // Skip self
                    }

                    if let Some(clause_d) = clauses.iter().find(|c| c.id() == candidate_id)
                        && Self::subsumes(clause_c.literals(), clause_d.literals())
                    {
                        subsumptions.push((clause_d.id(), clause_c.id()));
                        self.stats.subsumed_clauses += 1;
                    }
                }
            }
        }

        subsumptions
    }

    /// Find all possible clause strengthening opportunities.
    ///
    /// Returns a list of (clause_id, literal_to_remove) pairs.
    pub fn find_strengthenings(&mut self, clauses: &[Clause]) -> Vec<(ClauseId, Literal)> {
        if !self.config.enable_strengthening {
            return Vec::new();
        }

        let mut strengthenings = Vec::new();

        for i in 0..clauses.len() {
            for j in 0..clauses.len() {
                if i == j {
                    continue;
                }

                let clause_c = &clauses[i];
                let clause_d = &clauses[j];

                if clause_c.len() >= clause_d.len() {
                    continue;
                }

                if clause_c.len() > self.config.max_subsumption_size
                    || clause_d.len() > self.config.max_subsumption_size
                {
                    continue;
                }

                if let Some(lit_to_remove) =
                    Self::self_subsumes(clause_c.literals(), clause_d.literals())
                {
                    strengthenings.push((clause_d.id(), lit_to_remove));
                    self.stats.strengthened_clauses += 1;
                    self.stats.literals_removed += 1;
                }
            }
        }

        strengthenings
    }

    /// Perform inprocessing on the clause database.
    ///
    /// Returns the number of clauses removed.
    pub fn inprocess(&mut self, conflicts: u64) -> bool {
        self.stats.inprocess_calls += 1;

        // Check if we should run inprocessing
        conflicts.is_multiple_of(self.config.inprocess_interval)
    }
}

impl Default for Inprocessor {
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
    fn test_subsumption_basic() {
        // {x1} subsumes {x1, x2}
        let c = vec![make_literal(0, true)];
        let d = vec![make_literal(0, true), make_literal(1, true)];

        assert!(Inprocessor::subsumes(&c, &d));
        assert!(!Inprocessor::subsumes(&d, &c));
    }

    #[test]
    fn test_subsumption_same_size() {
        // {x1, x2} subsumes {x1, x2}
        let c = vec![make_literal(0, true), make_literal(1, true)];
        let d = vec![make_literal(0, true), make_literal(1, true)];

        assert!(Inprocessor::subsumes(&c, &d));
    }

    #[test]
    fn test_subsumption_different_literals() {
        // {x1, x2} does not subsume {x1, x3}
        let c = vec![make_literal(0, true), make_literal(1, true)];
        let d = vec![make_literal(0, true), make_literal(2, true)];

        assert!(!Inprocessor::subsumes(&c, &d));
    }

    #[test]
    fn test_self_subsumption() {
        // {x1} self-subsumes {x1, ~x2} ∪ {x2} = can remove ~x2 from {x1, ~x2}
        // This means: {x1, x2} and {x1, ~x2} -> strengthen to {x1}

        let c = vec![make_literal(0, true), make_literal(1, true)];
        let d = vec![make_literal(0, true), make_literal(1, false)];

        assert_eq!(
            Inprocessor::self_subsumes(&c, &d),
            Some(make_literal(1, false))
        );
    }

    #[test]
    fn test_self_subsumption_none() {
        let c = vec![make_literal(0, true), make_literal(1, true)];
        let d = vec![make_literal(0, true), make_literal(2, true)];

        assert_eq!(Inprocessor::self_subsumes(&c, &d), None);
    }

    #[test]
    fn test_inprocessor_new() {
        let inproc = Inprocessor::new();
        assert_eq!(inproc.stats().subsumed_clauses, 0);
        assert_eq!(inproc.stats().strengthened_clauses, 0);
    }

    #[test]
    fn test_inprocess_interval() {
        let mut inproc = Inprocessor::new();
        assert!(inproc.inprocess(5000));
        assert!(!inproc.inprocess(5001));
        assert!(inproc.inprocess(10000));
    }
}
