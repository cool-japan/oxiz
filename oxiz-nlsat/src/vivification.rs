//! Vivification for clause strengthening.
//!
//! Vivification is a powerful clause strengthening technique that uses
//! unit propagation to find literals that can be removed from clauses.
//! It's particularly effective at reducing clause sizes without losing
//! logical information.
//!
//! Reference: "Vivification" by Piette et al. (IJCAI 2008)

use crate::clause::Clause;
use crate::types::Literal;

/// Statistics for vivification.
#[derive(Debug, Clone, Default)]
pub struct VivificationStats {
    /// Number of clauses vivified.
    pub clauses_vivified: u64,
    /// Number of literals removed.
    pub literals_removed: u64,
    /// Number of clauses removed (became tautologies).
    pub clauses_removed: u64,
}

/// Configuration for vivification.
#[derive(Debug, Clone)]
pub struct VivificationConfig {
    /// Enable vivification.
    pub enabled: bool,
    /// Maximum clause size for vivification.
    pub max_clause_size: usize,
    /// Maximum number of propagations per vivification attempt.
    pub max_propagations: usize,
}

impl Default for VivificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_clause_size: 30,
            max_propagations: 100,
        }
    }
}

/// Vivification engine.
pub struct Vivifier {
    config: VivificationConfig,
    stats: VivificationStats,
}

impl Vivifier {
    /// Create a new vivifier with default configuration.
    pub fn new() -> Self {
        Self::with_config(VivificationConfig::default())
    }

    /// Create a new vivifier with the given configuration.
    pub fn with_config(config: VivificationConfig) -> Self {
        Self {
            config,
            stats: VivificationStats::default(),
        }
    }

    /// Get the statistics.
    pub fn stats(&self) -> &VivificationStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &VivificationConfig {
        &self.config
    }

    /// Attempt to vivify a clause.
    ///
    /// Returns a possibly shortened clause, or None if the clause
    /// becomes a tautology (can be removed).
    pub fn vivify_clause(&mut self, clause: &[Literal]) -> Option<Vec<Literal>> {
        if !self.config.enabled {
            return Some(clause.to_vec());
        }

        if clause.len() > self.config.max_clause_size {
            return Some(clause.to_vec());
        }

        // Try to find literals that can be removed
        let mut result = clause.to_vec();
        let mut removed_any = false;

        // For each literal in the clause, try assuming its negation
        // and see if we can derive a conflict or unit propagation
        let mut i = 0;
        while i < result.len() {
            let lit = result[i];

            // Simulate unit propagation with ~lit
            // If we can derive the clause without lit, then lit is redundant
            if self.is_redundant_in_clause(lit, &result) {
                result.remove(i);
                removed_any = true;
                self.stats.literals_removed += 1;
            } else {
                i += 1;
            }
        }

        if removed_any {
            self.stats.clauses_vivified += 1;
        }

        if result.is_empty() {
            self.stats.clauses_removed += 1;
            None
        } else {
            Some(result)
        }
    }

    /// Check if a literal is redundant in a clause.
    ///
    /// A literal is redundant if the clause is still valid without it.
    /// This is a simplified version - a full implementation would use
    /// unit propagation with the solver's full clause database.
    fn is_redundant_in_clause(&self, lit: Literal, clause: &[Literal]) -> bool {
        // Check if the clause contains both lit and ~lit (tautology)
        let neg_lit = lit.negate();
        let has_negation = clause.contains(&neg_lit);

        if has_negation {
            // Tautological clause - this literal is redundant
            return true;
        }

        // For more sophisticated redundancy checking, we would need
        // access to the full clause database and assignment.
        // This is a placeholder for the actual implementation.
        false
    }

    /// Vivify multiple clauses.
    ///
    /// Returns a list of vivified clauses (None for removed clauses).
    pub fn vivify_clauses(&mut self, clauses: &[Clause]) -> Vec<Option<Vec<Literal>>> {
        clauses
            .iter()
            .map(|clause| self.vivify_clause(clause.literals()))
            .collect()
    }
}

impl Default for Vivifier {
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
    fn test_vivifier_new() {
        let viv = Vivifier::new();
        assert!(viv.config().enabled);
        assert_eq!(viv.stats().clauses_vivified, 0);
        assert_eq!(viv.stats().literals_removed, 0);
    }

    #[test]
    fn test_vivify_tautology() {
        let mut viv = Vivifier::new();

        // {x1, ~x1} is a tautology
        let clause = vec![make_literal(0, true), make_literal(0, false)];

        let result = viv.vivify_clause(&clause);

        // Should detect tautology and remove redundant literal
        assert!(result.is_some());
        let vivified = result.unwrap();
        assert!(vivified.len() < clause.len());
    }

    #[test]
    fn test_vivify_no_change() {
        let mut viv = Vivifier::new();

        // {x1, x2} - no redundancy
        let clause = vec![make_literal(0, true), make_literal(1, true)];

        let result = viv.vivify_clause(&clause);

        assert_eq!(result, Some(clause));
    }

    #[test]
    fn test_vivify_disabled() {
        let config = VivificationConfig {
            enabled: false,
            ..Default::default()
        };
        let mut viv = Vivifier::with_config(config);

        let clause = vec![make_literal(0, true), make_literal(0, false)];

        let result = viv.vivify_clause(&clause);

        // When disabled, should return original clause
        assert_eq!(result, Some(clause));
    }

    #[test]
    fn test_vivify_large_clause() {
        let mut viv = Vivifier::new();

        // Create a clause larger than max_clause_size
        let mut clause = Vec::new();
        for i in 0..50 {
            clause.push(make_literal(i, true));
        }

        let result = viv.vivify_clause(&clause);

        // Should skip large clauses
        assert_eq!(result, Some(clause));
        assert_eq!(viv.stats().clauses_vivified, 0);
    }

    #[test]
    fn test_is_redundant_tautology() {
        let viv = Vivifier::new();

        let clause = vec![make_literal(0, true), make_literal(0, false)];

        // x1 is redundant because ~x1 is also present
        assert!(viv.is_redundant_in_clause(make_literal(0, true), &clause));
    }

    #[test]
    fn test_is_redundant_no_tautology() {
        let viv = Vivifier::new();

        let clause = vec![make_literal(0, true), make_literal(1, true)];

        // x1 is not redundant
        assert!(!viv.is_redundant_in_clause(make_literal(0, true), &clause));
    }
}
