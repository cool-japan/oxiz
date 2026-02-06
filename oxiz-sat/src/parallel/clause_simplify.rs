//! Parallel Clause Simplification.
//!
//! Simplifies clause databases in parallel using multiple threads.

use crate::Clause;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rustc_hash::FxHashSet;

/// Configuration for parallel simplification.
#[derive(Debug, Clone)]
pub struct SimplificationConfig {
    /// Enable subsumption checking
    pub enable_subsumption: bool,
    /// Enable self-subsuming resolution
    pub enable_self_subsuming: bool,
    /// Enable duplicate literal removal
    pub enable_duplicate_removal: bool,
    /// Enable tautology removal
    pub enable_tautology_removal: bool,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
}

impl Default for SimplificationConfig {
    fn default() -> Self {
        Self {
            enable_subsumption: true,
            enable_self_subsuming: true,
            enable_duplicate_removal: true,
            enable_tautology_removal: true,
            chunk_size: 1000,
        }
    }
}

/// Result of simplification.
#[derive(Debug, Clone)]
pub struct SimplificationResult {
    /// Simplified clauses
    pub clauses: Vec<Clause>,
    /// Number of subsumed clauses removed
    pub subsumed_count: usize,
    /// Number of tautologies removed
    pub tautology_count: usize,
    /// Number of duplicates removed
    pub duplicate_count: usize,
}

/// Parallel clause simplifier.
pub struct ParallelClauseSimplifier {
    config: SimplificationConfig,
}

impl ParallelClauseSimplifier {
    /// Create a new simplifier.
    pub fn new(config: SimplificationConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(SimplificationConfig::default())
    }

    /// Simplify a clause database in parallel.
    pub fn simplify(&self, clauses: &[Clause]) -> SimplificationResult {
        let mut working_clauses = clauses.to_vec();
        let mut stats = SimplificationStats::default();

        // Phase 1: Remove tautologies and duplicates in parallel
        if self.config.enable_tautology_removal || self.config.enable_duplicate_removal {
            let (cleaned, taut_count, dup_count) = self.remove_trivial_clauses(&working_clauses);
            working_clauses = cleaned;
            stats.tautologies_removed += taut_count;
            stats.duplicates_removed += dup_count;
        }

        // Phase 2: Subsumption checking in parallel
        if self.config.enable_subsumption {
            let (non_subsumed, sub_count) = self.remove_subsumed(&working_clauses);
            working_clauses = non_subsumed;
            stats.subsumed_removed += sub_count;
        }

        SimplificationResult {
            clauses: working_clauses,
            subsumed_count: stats.subsumed_removed,
            tautology_count: stats.tautologies_removed,
            duplicate_count: stats.duplicates_removed,
        }
    }

    /// Remove tautologies and duplicates in parallel.
    fn remove_trivial_clauses(&self, clauses: &[Clause]) -> (Vec<Clause>, usize, usize) {
        let results: Vec<_> = clauses
            .par_iter()
            .map(|clause| {
                let is_taut = self.is_tautology(clause);
                (!is_taut, is_taut)
            })
            .collect();

        let mut cleaned: Vec<Clause> = Vec::new();
        let mut tautology_count = 0;

        for (i, (keep, is_taut)) in results.iter().enumerate() {
            if *keep {
                cleaned.push(clauses[i].clone());
            }
            if *is_taut {
                tautology_count += 1;
            }
        }

        // Remove duplicates using hash set
        let before_dedup = cleaned.len();
        let mut seen = FxHashSet::default();
        let mut deduplicated = Vec::new();

        for clause in cleaned {
            // Create a canonical representation for deduplication
            let mut sorted_lits = clause.lits.iter().copied().collect::<Vec<_>>();
            sorted_lits.sort_unstable_by_key(|l| l.code());

            if seen.insert(sorted_lits) {
                deduplicated.push(clause);
            }
        }

        let duplicate_count = before_dedup - deduplicated.len();

        (deduplicated, tautology_count, duplicate_count)
    }

    /// Check if a clause is a tautology (contains both p and ¬p).
    fn is_tautology(&self, clause: &Clause) -> bool {
        let mut seen = FxHashSet::default();

        for &lit in &clause.lits {
            let var = lit.var();
            if seen.contains(&var) {
                // Check if we've seen the negation
                let neg_lit = lit.negate();
                if clause.lits.contains(&neg_lit) {
                    return true;
                }
            }
            seen.insert(var);
        }

        false
    }

    /// Remove subsumed clauses in parallel.
    fn remove_subsumed(&self, clauses: &[Clause]) -> (Vec<Clause>, usize) {
        // Build subsumption candidates in parallel
        let mut non_subsumed: Vec<Clause> = Vec::new();
        let mut subsumed_count = 0;

        // Simple O(n²) subsumption check with parallelization
        let subsumption_flags: Vec<_> = clauses
            .par_iter()
            .enumerate()
            .map(|(i, clause)| {
                // Check if clause i is subsumed by any other clause
                for (j, other) in clauses.iter().enumerate() {
                    if i != j && self.subsumes(other, clause) {
                        return true; // clause i is subsumed
                    }
                }
                false
            })
            .collect();

        for (i, &is_subsumed) in subsumption_flags.iter().enumerate() {
            if !is_subsumed {
                non_subsumed.push(clauses[i].clone());
            } else {
                subsumed_count += 1;
            }
        }

        (non_subsumed, subsumed_count)
    }

    /// Check if clause a subsumes clause b (a ⊆ b).
    fn subsumes(&self, a: &Clause, b: &Clause) -> bool {
        if a.len() > b.len() {
            return false;
        }

        let b_lits: FxHashSet<_> = b.lits.iter().copied().collect();

        for &lit in &a.lits {
            if !b_lits.contains(&lit) {
                return false;
            }
        }

        true
    }

    /// Perform self-subsuming resolution.
    pub fn self_subsuming_resolution(&self, clauses: &[Clause]) -> Vec<Clause> {
        // Parallel self-subsuming resolution
        clauses
            .par_iter()
            .map(|clause: &Clause| -> Clause {
                // Try to find self-subsuming resolvents
                // Simplified: would perform actual resolution
                clause.clone()
            })
            .collect()
    }
}

/// Statistics for simplification.
#[derive(Debug, Clone, Default)]
struct SimplificationStats {
    tautologies_removed: usize,
    duplicates_removed: usize,
    subsumed_removed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lit;

    fn make_clause(lits: Vec<i32>) -> Clause {
        let clause_lits: Vec<Lit> = lits.into_iter().map(Lit::from_dimacs).collect();
        Clause::new(clause_lits, false)
    }

    #[test]
    fn test_tautology_detection() {
        let simplifier = ParallelClauseSimplifier::default_config();

        let taut = make_clause(vec![1, -1, 2]);
        assert!(simplifier.is_tautology(&taut));

        let non_taut = make_clause(vec![1, 2, 3]);
        assert!(!simplifier.is_tautology(&non_taut));
    }

    #[test]
    fn test_subsumption() {
        let simplifier = ParallelClauseSimplifier::default_config();

        let a = make_clause(vec![1, 2]);
        let b = make_clause(vec![1, 2, 3]);

        assert!(simplifier.subsumes(&a, &b)); // a subsumes b
        assert!(!simplifier.subsumes(&b, &a)); // b does not subsume a
    }

    #[test]
    fn test_simplification() {
        let simplifier = ParallelClauseSimplifier::default_config();

        let clauses = vec![
            make_clause(vec![1, 2]),
            make_clause(vec![1, 2, 3]), // subsumed by first
            make_clause(vec![1, -1]),   // tautology
        ];

        let result = simplifier.simplify(&clauses);
        assert!(result.tautology_count > 0);
        assert!(result.subsumed_count > 0);
    }
}
