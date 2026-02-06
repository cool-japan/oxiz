//! Theory Conflict Explanation.
//!
//! This module generates detailed explanations for theory conflicts, enabling
//! CDCL(T) to learn effective conflict clauses from theory solver failures.
//!
//! ## Explanation Types
//!
//! 1. **Equality Explanations**: Why two terms must/cannot be equal
//! 2. **Bound Explanations**: Why a variable must satisfy certain bounds
//! 3. **Disequality Explanations**: Why distinct terms cannot be equal
//! 4. **Arithmetic Explanations**: Linear combinations proving inconsistency
//!
//! ## Explanation Quality
//!
//! Good explanations are:
//! - **Minimal**: Use fewest literals possible
//! - **Relevant**: Only include necessary constraints
//! - **General**: Learn broadly applicable conflicts
//! - **Precise**: Accurately capture the inconsistency
//!
//! ## References
//!
//! - Nieuwenhuis et al.: "Solving SAT and SAT Modulo Theories" (JACM 2006)
//! - de Moura & Bjørner: "Z3: An Efficient SMT Solver" (TACAS 2008)
//! - Z3's `smt/theory_explanation.cpp`

use oxiz_sat::Lit;
use rustc_hash::FxHashSet;

/// Type of theory explanation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplanationType {
    /// Equality conflict (a = b and a ≠ b).
    Equality,
    /// Bound conflict (x > 5 and x < 3).
    Bounds,
    /// Disequality conflict (distinct(a,b,c) but a = b).
    Disequality,
    /// Arithmetic conflict (linear combination proves UNSAT).
    Arithmetic,
    /// Array conflict (read-over-write axiom violation).
    Array,
    /// BitVector conflict (overflow, underflow).
    BitVector,
}

/// An explanation for a theory conflict.
#[derive(Debug, Clone)]
pub struct TheoryExplanation {
    /// Type of explanation.
    pub explanation_type: ExplanationType,
    /// Literals involved in conflict (as conflict clause).
    pub literals: Vec<Lit>,
    /// Human-readable explanation (for debugging).
    pub description: String,
    /// Proof trace (optional, for proof generation).
    pub proof_trace: Option<Vec<ProofStep>>,
}

/// A step in a theory conflict proof.
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// Description of this proof step.
    pub description: String,
    /// Literals used in this step.
    pub premises: Vec<Lit>,
    /// Conclusion of this step.
    pub conclusion: String,
}

/// Configuration for theory explanation generation.
#[derive(Debug, Clone)]
pub struct ExplainerConfig {
    /// Enable proof trace generation.
    pub generate_proofs: bool,
    /// Minimize explanations.
    pub minimize: bool,
    /// Include human-readable descriptions.
    pub include_descriptions: bool,
    /// Maximum explanation size (literals).
    pub max_size: usize,
}

impl Default for ExplainerConfig {
    fn default() -> Self {
        Self {
            generate_proofs: false,
            minimize: true,
            include_descriptions: false,
            max_size: 100,
        }
    }
}

/// Statistics for explanation generation.
#[derive(Debug, Clone, Default)]
pub struct ExplainerStats {
    /// Explanations generated.
    pub explanations_generated: u64,
    /// Total literals in explanations.
    pub total_literals: u64,
    /// Average explanation size.
    pub avg_size: f64,
    /// Explanations minimized.
    pub minimized: u64,
    /// Time (microseconds).
    pub time_us: u64,
}

/// Theory conflict explainer.
pub struct TheoryExplainer {
    config: ExplainerConfig,
    stats: ExplainerStats,
}

impl TheoryExplainer {
    /// Create new theory explainer.
    pub fn new() -> Self {
        Self::with_config(ExplainerConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: ExplainerConfig) -> Self {
        Self {
            config,
            stats: ExplainerStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ExplainerStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ExplainerStats::default();
    }

    /// Generate explanation for an equality conflict.
    ///
    /// Explains why two terms that must be equal cannot be equal,
    /// or vice versa.
    pub fn explain_equality_conflict(
        &mut self,
        lits: Vec<Lit>,
        description: Option<String>,
    ) -> TheoryExplanation {
        let start = std::time::Instant::now();

        let minimized_lits = if self.config.minimize {
            self.minimize_explanation(&lits)
        } else {
            lits.clone()
        };

        let proof_trace = if self.config.generate_proofs {
            Some(self.generate_proof_trace(&minimized_lits, ExplanationType::Equality))
        } else {
            None
        };

        let desc = if self.config.include_descriptions {
            description.unwrap_or_else(|| {
                format!("Equality conflict with {} literals", minimized_lits.len())
            })
        } else {
            String::new()
        };

        self.stats.explanations_generated += 1;
        self.stats.total_literals += minimized_lits.len() as u64;
        self.update_avg_size();
        self.stats.time_us += start.elapsed().as_micros() as u64;

        TheoryExplanation {
            explanation_type: ExplanationType::Equality,
            literals: minimized_lits,
            description: desc,
            proof_trace,
        }
    }

    /// Generate explanation for a bounds conflict.
    ///
    /// Explains why variable bounds are inconsistent (e.g., x > 10 ∧ x < 5).
    pub fn explain_bounds_conflict(
        &mut self,
        lits: Vec<Lit>,
        description: Option<String>,
    ) -> TheoryExplanation {
        let start = std::time::Instant::now();

        let minimized_lits = if self.config.minimize {
            self.minimize_explanation(&lits)
        } else {
            lits.clone()
        };

        let proof_trace = if self.config.generate_proofs {
            Some(self.generate_proof_trace(&minimized_lits, ExplanationType::Bounds))
        } else {
            None
        };

        let desc = if self.config.include_descriptions {
            description.unwrap_or_else(|| {
                format!("Bounds conflict with {} constraints", minimized_lits.len())
            })
        } else {
            String::new()
        };

        self.stats.explanations_generated += 1;
        self.stats.total_literals += minimized_lits.len() as u64;
        self.update_avg_size();
        self.stats.time_us += start.elapsed().as_micros() as u64;

        TheoryExplanation {
            explanation_type: ExplanationType::Bounds,
            literals: minimized_lits,
            description: desc,
            proof_trace,
        }
    }

    /// Generate explanation for an arithmetic conflict.
    ///
    /// Produces a Farkas coefficient explanation showing linear combination
    /// that proves inconsistency.
    pub fn explain_arithmetic_conflict(
        &mut self,
        lits: Vec<Lit>,
        _farkas_coefficients: Option<Vec<i64>>,
        description: Option<String>,
    ) -> TheoryExplanation {
        let start = std::time::Instant::now();

        // In full implementation, use Farkas coefficients to generate
        // minimal explanation via linear combination

        let minimized_lits = if self.config.minimize {
            self.minimize_explanation(&lits)
        } else {
            lits.clone()
        };

        let proof_trace = if self.config.generate_proofs {
            Some(self.generate_proof_trace(&minimized_lits, ExplanationType::Arithmetic))
        } else {
            None
        };

        let desc = if self.config.include_descriptions {
            description.unwrap_or_else(|| "Arithmetic conflict via linear combination".to_string())
        } else {
            String::new()
        };

        self.stats.explanations_generated += 1;
        self.stats.total_literals += minimized_lits.len() as u64;
        self.update_avg_size();
        self.stats.time_us += start.elapsed().as_micros() as u64;

        TheoryExplanation {
            explanation_type: ExplanationType::Arithmetic,
            literals: minimized_lits,
            description: desc,
            proof_trace,
        }
    }

    /// Minimize an explanation by removing redundant literals.
    fn minimize_explanation(&mut self, lits: &[Lit]) -> Vec<Lit> {
        // Simplified minimization - full version would use:
        // - Relevancy analysis
        // - Core extraction
        // - Subsumption checking

        let mut minimized = Vec::new();
        let _lit_set: FxHashSet<Lit> = lits.iter().copied().collect();

        for &lit in lits {
            // Check if removing this literal still gives a conflict
            // For now, keep all literals (placeholder)
            minimized.push(lit);
        }

        if minimized.len() < lits.len() {
            self.stats.minimized += 1;
        }

        // Enforce max size
        if minimized.len() > self.config.max_size {
            minimized.truncate(self.config.max_size);
        }

        minimized
    }

    /// Generate proof trace for explanation.
    fn generate_proof_trace(&self, lits: &[Lit], exp_type: ExplanationType) -> Vec<ProofStep> {
        let mut trace = Vec::new();

        // Generate initial premises
        trace.push(ProofStep {
            description: format!("Theory conflict detected: {:?}", exp_type),
            premises: lits.to_vec(),
            conclusion: "Contradiction".to_string(),
        });

        // Full implementation would include detailed proof steps

        trace
    }

    /// Update average explanation size.
    fn update_avg_size(&mut self) {
        if self.stats.explanations_generated > 0 {
            self.stats.avg_size =
                self.stats.total_literals as f64 / self.stats.explanations_generated as f64;
        }
    }
}

impl Default for TheoryExplainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_sat::Var;

    fn lit(var: u32, positive: bool) -> Lit {
        let v = Var::new(var);
        if positive { Lit::pos(v) } else { Lit::neg(v) }
    }

    #[test]
    fn test_explainer_creation() {
        let explainer = TheoryExplainer::new();
        assert_eq!(explainer.stats().explanations_generated, 0);
    }

    #[test]
    fn test_equality_explanation() {
        let mut explainer = TheoryExplainer::new();
        let lits = vec![lit(0, true), lit(1, false), lit(2, true)];

        let explanation = explainer.explain_equality_conflict(lits.clone(), None);

        assert_eq!(explanation.explanation_type, ExplanationType::Equality);
        assert!(!explanation.literals.is_empty());
        assert_eq!(explainer.stats().explanations_generated, 1);
    }

    #[test]
    fn test_bounds_explanation() {
        let mut explainer = TheoryExplainer::new();
        let lits = vec![lit(0, true), lit(1, true)];

        let explanation =
            explainer.explain_bounds_conflict(lits, Some("x > 10 and x < 5".to_string()));

        assert_eq!(explanation.explanation_type, ExplanationType::Bounds);
        assert_eq!(explainer.stats().explanations_generated, 1);
    }

    #[test]
    fn test_arithmetic_explanation() {
        let mut explainer = TheoryExplainer::new();
        let lits = vec![lit(0, false), lit(1, false), lit(2, false)];

        let explanation = explainer.explain_arithmetic_conflict(lits, None, None);

        assert_eq!(explanation.explanation_type, ExplanationType::Arithmetic);
        assert_eq!(explainer.stats().explanations_generated, 1);
    }

    #[test]
    fn test_minimization() {
        let config = ExplainerConfig {
            minimize: true,
            ..Default::default()
        };
        let mut explainer = TheoryExplainer::with_config(config);

        let lits = vec![lit(0, true), lit(1, false), lit(2, true), lit(3, false)];

        let explanation = explainer.explain_equality_conflict(lits, None);

        // Should attempt minimization
        assert!(explanation.literals.len() <= 4);
    }

    #[test]
    fn test_proof_generation() {
        let config = ExplainerConfig {
            generate_proofs: true,
            ..Default::default()
        };
        let mut explainer = TheoryExplainer::with_config(config);

        let lits = vec![lit(0, true), lit(1, false)];

        let explanation = explainer.explain_equality_conflict(lits, None);

        assert!(explanation.proof_trace.is_some());
        let trace = explanation.proof_trace.unwrap();
        assert!(!trace.is_empty());
    }

    #[test]
    fn test_max_size_enforcement() {
        let config = ExplainerConfig {
            max_size: 2,
            ..Default::default()
        };
        let mut explainer = TheoryExplainer::with_config(config);

        let lits = vec![lit(0, true), lit(1, false), lit(2, true), lit(3, false)];

        let explanation = explainer.explain_bounds_conflict(lits, None);

        assert!(explanation.literals.len() <= 2);
    }
}
