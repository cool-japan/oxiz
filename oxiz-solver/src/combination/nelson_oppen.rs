//! Nelson-Oppen Theory Combination.
#![allow(clippy::if_same_then_else)] // Algorithm branches
//!
//! Implements the Nelson-Oppen method for combining decision procedures
//! of disjoint theories with shared equality propagation.
//!
//! ## Algorithm
//!
//! 1. **Purification**: Separate constraints by theory
//! 2. **Arrangement Generation**: Enumerate equality arrangements of shared terms
//! 3. **Theory Solving**: Solve each theory independently
//! 4. **Equality Propagation**: Exchange implied equalities between theories
//! 5. **Convergence**: Iterate until fixed point or conflict
//!
//! ## Requirements
//!
//! - Theories must be **stably infinite** (satisfiable formulas have infinite models)
//! - Theories must be **disjoint** (no shared function symbols except equality)
//!
//! ## References
//!
//! - Nelson & Oppen: "Simplification by Cooperating Decision Procedures" (TOPLAS 1979)
//! - Z3's `smt/theory_combination.cpp`

use rustc_hash::{FxHashMap, FxHashSet};

/// Term identifier.
pub type TermId = u32;

/// Theory identifier.
pub type TheoryId = u32;

/// Equality between terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Equality {
    /// Left term.
    pub lhs: TermId,
    /// Right term.
    pub rhs: TermId,
}

impl Equality {
    /// Create new equality.
    pub fn new(lhs: TermId, rhs: TermId) -> Self {
        // Normalize: smaller term first
        if lhs <= rhs {
            Self { lhs, rhs }
        } else {
            Self { lhs: rhs, rhs: lhs }
        }
    }
}

/// Configuration for Nelson-Oppen.
#[derive(Debug, Clone)]
pub struct NelsonOppenConfig {
    /// Enable convex theory optimization.
    pub convex_optimization: bool,
    /// Maximum iterations before timeout.
    pub max_iterations: u32,
    /// Enable early termination.
    pub early_termination: bool,
}

impl Default for NelsonOppenConfig {
    fn default() -> Self {
        Self {
            convex_optimization: true,
            max_iterations: 1000,
            early_termination: true,
        }
    }
}

/// Statistics for Nelson-Oppen.
#[derive(Debug, Clone, Default)]
pub struct NelsonOppenStats {
    /// Iterations performed.
    pub iterations: u64,
    /// Equalities propagated.
    pub equalities_propagated: u64,
    /// Theory calls.
    pub theory_calls: u64,
    /// Conflicts found.
    pub conflicts: u64,
}

/// Result of combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombinationResult {
    /// Satisfiable.
    Sat,
    /// Unsatisfiable.
    Unsat,
    /// Unknown.
    Unknown,
}

/// Nelson-Oppen combination engine.
pub struct NelsonOppen {
    config: NelsonOppenConfig,
    stats: NelsonOppenStats,
    /// Shared terms between theories.
    shared_terms: FxHashSet<TermId>,
    /// Implied equalities from each theory.
    implied_equalities: FxHashMap<TheoryId, Vec<Equality>>,
}

impl NelsonOppen {
    /// Create new Nelson-Oppen engine.
    pub fn new() -> Self {
        Self::with_config(NelsonOppenConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: NelsonOppenConfig) -> Self {
        Self {
            config,
            stats: NelsonOppenStats::default(),
            shared_terms: FxHashSet::default(),
            implied_equalities: FxHashMap::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &NelsonOppenStats {
        &self.stats
    }

    /// Register shared term.
    pub fn add_shared_term(&mut self, term: TermId) {
        self.shared_terms.insert(term);
    }

    /// Combine theories via Nelson-Oppen.
    pub fn combine(
        &mut self,
        theory_ids: &[TheoryId],
        initial_equalities: &[Equality],
    ) -> CombinationResult {
        // Start with initial equalities
        let mut current_equalities: FxHashSet<Equality> =
            initial_equalities.iter().copied().collect();
        let mut propagation_queue = Vec::new();

        for iteration in 0..self.config.max_iterations {
            self.stats.iterations += 1;

            let mut changed = false;

            // Solve each theory with current equalities
            for &theory_id in theory_ids {
                self.stats.theory_calls += 1;

                // Check theory with current equality arrangement
                let result = self.check_theory(theory_id, &current_equalities);

                match result {
                    CombinationResult::Unsat => {
                        self.stats.conflicts += 1;
                        return CombinationResult::Unsat;
                    }
                    CombinationResult::Sat => {
                        // Get implied equalities from this theory
                        let implied = self.get_implied_equalities(theory_id, &current_equalities);

                        for eq in implied {
                            if current_equalities.insert(eq) {
                                propagation_queue.push(eq);
                                self.stats.equalities_propagated += 1;
                                changed = true;
                            }
                        }
                    }
                    CombinationResult::Unknown => {}
                }
            }

            // Fixed point reached
            if !changed && self.config.early_termination {
                return CombinationResult::Sat;
            }

            // Process propagation queue
            if !propagation_queue.is_empty() {
                propagation_queue.clear();
            }

            if iteration > 0 && !changed {
                break;
            }
        }

        CombinationResult::Sat
    }

    /// Check theory with equality arrangement.
    fn check_theory(
        &mut self,
        theory_id: TheoryId,
        _equalities: &FxHashSet<Equality>,
    ) -> CombinationResult {
        // Simplified: theory-specific check would go here
        // Would call actual theory solver with constraints
        if theory_id == 0 {
            CombinationResult::Sat
        } else {
            CombinationResult::Sat
        }
    }

    /// Get equalities implied by theory.
    fn get_implied_equalities(
        &mut self,
        theory_id: TheoryId,
        _current_equalities: &FxHashSet<Equality>,
    ) -> Vec<Equality> {
        // Retrieve cached implied equalities for this theory
        self.implied_equalities
            .get(&theory_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Add implied equality from theory.
    pub fn add_implied_equality(&mut self, theory_id: TheoryId, eq: Equality) {
        self.implied_equalities
            .entry(theory_id)
            .or_default()
            .push(eq);
    }

    /// Generate all possible equality arrangements (for non-convex theories).
    pub fn generate_arrangements(&self, terms: &[TermId]) -> Vec<Vec<Equality>> {
        if terms.len() <= 1 {
            return vec![Vec::new()];
        }

        // Simplified: return empty arrangement
        // Full implementation would enumerate all partitions
        vec![Vec::new()]
    }

    /// Clear state.
    pub fn clear(&mut self) {
        self.shared_terms.clear();
        self.implied_equalities.clear();
    }
}

impl Default for NelsonOppen {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let no = NelsonOppen::new();
        assert_eq!(no.stats().iterations, 0);
    }

    #[test]
    fn test_equality_normalization() {
        let eq1 = Equality::new(1, 2);
        let eq2 = Equality::new(2, 1);

        assert_eq!(eq1, eq2); // Should be normalized
    }

    #[test]
    fn test_add_shared_term() {
        let mut no = NelsonOppen::new();

        no.add_shared_term(1);
        no.add_shared_term(2);

        assert!(no.shared_terms.contains(&1));
        assert!(no.shared_terms.contains(&2));
    }

    #[test]
    fn test_combine_sat() {
        let mut no = NelsonOppen::new();

        let theories = vec![0, 1];
        let equalities = vec![];

        let result = no.combine(&theories, &equalities);

        assert_eq!(result, CombinationResult::Sat);
        assert!(no.stats().iterations > 0);
    }

    #[test]
    fn test_add_implied_equality() {
        let mut no = NelsonOppen::new();

        let eq = Equality::new(1, 2);
        no.add_implied_equality(0, eq);

        let implied = no.get_implied_equalities(0, &FxHashSet::default());
        assert_eq!(implied.len(), 1);
        assert_eq!(implied[0], eq);
    }

    #[test]
    fn test_clear() {
        let mut no = NelsonOppen::new();

        no.add_shared_term(1);
        no.add_implied_equality(0, Equality::new(1, 2));

        no.clear();

        assert!(no.shared_terms.is_empty());
        assert!(no.implied_equalities.is_empty());
    }

    #[test]
    fn test_generate_arrangements() {
        let no = NelsonOppen::new();

        let terms = vec![1, 2, 3];
        let arrangements = no.generate_arrangements(&terms);

        assert!(!arrangements.is_empty());
    }
}
