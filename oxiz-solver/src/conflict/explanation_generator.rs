//! Explanation Generation for Theory Conflicts.
//!
//! Generates conflict explanations from theory solvers that are minimal,
//! relevant, and efficiently encoded for clause learning.
//!
//! ## Features
//!
//! - **Minimality**: Remove irrelevant literals from explanations
//! - **Caching**: Reuse explanations for repeated conflicts
//! - **Proof generation**: Optionally track justifications
//!
//! ## References
//!
//! - "Efficient E-matching for SMT Solvers" (de Moura & Bjørner, 2007)
//! - Z3's `smt/theory_explanation.cpp`

use oxiz_sat::Lit;
use rustc_hash::{FxHashMap, FxHashSet};

/// Theory identifier.
pub type TheoryId = u32;

/// An explanation for a theory propagation or conflict.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Explanation {
    /// Literals in the explanation (antecedents).
    pub literals: Vec<Lit>,
    /// Theory that generated this explanation.
    pub theory: TheoryId,
    /// Optional justification (for proof generation).
    pub justification: Option<String>,
}

impl Explanation {
    /// Create a new explanation.
    pub fn new(literals: Vec<Lit>, theory: TheoryId) -> Self {
        Self {
            literals,
            theory,
            justification: None,
        }
    }

    /// Create with justification.
    pub fn with_justification(literals: Vec<Lit>, theory: TheoryId, justification: String) -> Self {
        Self {
            literals,
            theory,
            justification: Some(justification),
        }
    }

    /// Get the size of the explanation.
    pub fn size(&self) -> usize {
        self.literals.len()
    }

    /// Check if explanation is empty.
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Check if explanation contains a literal.
    pub fn contains(&self, lit: Lit) -> bool {
        self.literals.contains(&lit)
    }
}

/// Configuration for explanation generation.
#[derive(Debug, Clone)]
pub struct ExplanationConfig {
    /// Enable explanation minimization.
    pub minimize: bool,
    /// Enable explanation caching.
    pub enable_cache: bool,
    /// Maximum cache size.
    pub max_cache_size: usize,
    /// Generate proof justifications.
    pub generate_proofs: bool,
}

impl Default for ExplanationConfig {
    fn default() -> Self {
        Self {
            minimize: true,
            enable_cache: true,
            max_cache_size: 10000,
            generate_proofs: false,
        }
    }
}

/// Statistics for explanation generation.
#[derive(Debug, Clone, Default)]
pub struct ExplanationStats {
    /// Total explanations generated.
    pub explanations_generated: u64,
    /// Explanations minimized.
    pub explanations_minimized: u64,
    /// Literals removed during minimization.
    pub literals_removed: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
}

/// Explanation generation engine.
pub struct ExplanationGenerator {
    /// Configuration.
    config: ExplanationConfig,
    /// Statistics.
    stats: ExplanationStats,
    /// Cache from conflict to explanation.
    cache: FxHashMap<Vec<Lit>, Explanation>,
}

impl ExplanationGenerator {
    /// Create a new explanation generator.
    pub fn new(config: ExplanationConfig) -> Self {
        Self {
            config,
            stats: ExplanationStats::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ExplanationConfig::default())
    }

    /// Generate an explanation for a conflict.
    ///
    /// Takes a set of conflicting literals and produces a minimal explanation.
    pub fn explain_conflict(&mut self, conflict: Vec<Lit>, theory: TheoryId) -> Explanation {
        self.stats.explanations_generated += 1;

        // Check cache if enabled
        if self.config.enable_cache {
            if let Some(cached) = self.cache.get(&conflict) {
                self.stats.cache_hits += 1;
                return cached.clone();
            }
            self.stats.cache_misses += 1;
        }

        // Generate explanation
        let mut explanation = Explanation::new(conflict.clone(), theory);

        // Minimize if enabled
        if self.config.minimize {
            explanation = self.minimize_explanation(explanation);
        }

        // Add to cache
        if self.config.enable_cache && self.cache.len() < self.config.max_cache_size {
            self.cache.insert(conflict, explanation.clone());
        }

        explanation
    }

    /// Minimize an explanation by removing irrelevant literals.
    fn minimize_explanation(&mut self, explanation: Explanation) -> Explanation {
        let original_size = explanation.size();

        // Simplified minimization: remove duplicate literals
        let mut seen = FxHashSet::default();
        let mut minimized_lits = Vec::new();

        for lit in explanation.literals {
            if seen.insert(lit) {
                minimized_lits.push(lit);
            }
        }

        let removed = original_size - minimized_lits.len();
        if removed > 0 {
            self.stats.explanations_minimized += 1;
            self.stats.literals_removed += removed as u64;
        }

        Explanation {
            literals: minimized_lits,
            theory: explanation.theory,
            justification: explanation.justification,
        }
    }

    /// Explain a theory propagation.
    ///
    /// Returns the antecedents that caused the propagation.
    pub fn explain_propagation(
        &mut self,
        propagated_lit: Lit,
        antecedents: Vec<Lit>,
        theory: TheoryId,
    ) -> Explanation {
        self.stats.explanations_generated += 1;

        let mut explanation = Explanation::new(antecedents, theory);

        if self.config.minimize {
            explanation = self.minimize_explanation(explanation);
        }

        // Add justification if proof generation enabled
        if self.config.generate_proofs {
            explanation.justification =
                Some(format!("Theory {} propagated {:?}", theory, propagated_lit));
        }

        explanation
    }

    /// Merge multiple explanations into one.
    pub fn merge_explanations(&mut self, explanations: Vec<Explanation>) -> Explanation {
        let mut all_literals = Vec::new();
        let mut all_theories = Vec::new();
        let mut all_justifications = Vec::new();

        for exp in explanations {
            all_literals.extend(exp.literals);
            all_theories.push(exp.theory);
            if let Some(just) = exp.justification {
                all_justifications.push(just);
            }
        }

        // Use first theory as representative
        let theory = all_theories.first().copied().unwrap_or(0);

        let mut merged = Explanation::new(all_literals, theory);

        if self.config.minimize {
            merged = self.minimize_explanation(merged);
        }

        if self.config.generate_proofs && !all_justifications.is_empty() {
            merged.justification = Some(all_justifications.join("; "));
        }

        merged
    }

    /// Clear the explanation cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &ExplanationStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ExplanationStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_sat::Var;

    #[test]
    fn test_explanation_creation() {
        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::neg(Var::new(1));

        let exp = Explanation::new(vec![lit1, lit2], 0);

        assert_eq!(exp.size(), 2);
        assert!(exp.contains(lit1));
        assert!(exp.contains(lit2));
        assert_eq!(exp.theory, 0);
    }

    #[test]
    fn test_generator_creation() {
        let generator = ExplanationGenerator::default_config();
        assert_eq!(generator.stats().explanations_generated, 0);
    }

    #[test]
    fn test_explain_conflict() {
        let mut generator = ExplanationGenerator::default_config();

        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::neg(Var::new(1));

        let exp = generator.explain_conflict(vec![lit1, lit2], 0);

        assert_eq!(exp.size(), 2);
        assert_eq!(generator.stats().explanations_generated, 1);
    }

    #[test]
    fn test_minimization() {
        let mut generator = ExplanationGenerator::default_config();

        let lit1 = Lit::pos(Var::new(0));
        let lit2 = Lit::pos(Var::new(0)); // Duplicate

        let exp = generator.explain_conflict(vec![lit1, lit2], 0);

        // Should remove duplicate
        assert_eq!(exp.size(), 1);
        assert!(generator.stats().explanations_minimized > 0);
    }

    #[test]
    fn test_caching() {
        let mut generator = ExplanationGenerator::default_config();

        let lits = vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))];

        // First call - cache miss
        let exp1 = generator.explain_conflict(lits.clone(), 0);
        assert_eq!(generator.stats().cache_misses, 1);

        // Second call - cache hit
        let exp2 = generator.explain_conflict(lits, 0);
        assert_eq!(generator.stats().cache_hits, 1);

        assert_eq!(exp1, exp2);
    }

    #[test]
    fn test_explain_propagation() {
        let config = ExplanationConfig {
            generate_proofs: true,
            ..Default::default()
        };

        let mut generator = ExplanationGenerator::new(config);

        let prop_lit = Lit::pos(Var::new(2));
        let antecedents = vec![Lit::pos(Var::new(0)), Lit::neg(Var::new(1))];

        let exp = generator.explain_propagation(prop_lit, antecedents, 0);

        assert_eq!(exp.size(), 2);
        assert!(exp.justification.is_some());
    }

    #[test]
    fn test_merge_explanations() {
        let mut generator = ExplanationGenerator::default_config();

        let exp1 = Explanation::new(vec![Lit::pos(Var::new(0))], 0);
        let exp2 = Explanation::new(vec![Lit::neg(Var::new(1))], 1);

        let merged = generator.merge_explanations(vec![exp1, exp2]);

        assert_eq!(merged.size(), 2);
    }

    #[test]
    fn test_clear_cache() {
        let mut generator = ExplanationGenerator::default_config();

        generator.explain_conflict(vec![Lit::pos(Var::new(0))], 0);
        assert!(!generator.cache.is_empty());

        generator.clear_cache();
        assert!(generator.cache.is_empty());
    }
}
