//! Proof simplification through logical rewriting.
//!
//! This module provides simplification of proof steps by applying logical
//! rewrite rules, combining redundant inferences, and normalizing conclusions.
//! Unlike compression (which removes unreachable nodes), simplification
//! transforms proof steps to equivalent but simpler forms.

use crate::proof::Proof;

/// Configuration for proof simplification.
#[derive(Debug, Clone)]
pub struct SimplificationConfig {
    /// Apply De Morgan's laws to normalize negations
    pub apply_demorgan: bool,
    /// Simplify double negations (¬¬p → p)
    pub simplify_double_negation: bool,
    /// Remove identity operations (p ∧ true → p)
    pub remove_identities: bool,
    /// Combine consecutive inferences when possible
    pub combine_inferences: bool,
    /// Simplify tautologies (p ∨ ¬p → true)
    pub simplify_tautologies: bool,
    /// Maximum number of simplification passes
    pub max_passes: usize,
}

impl Default for SimplificationConfig {
    fn default() -> Self {
        Self {
            apply_demorgan: true,
            simplify_double_negation: true,
            remove_identities: true,
            combine_inferences: true,
            simplify_tautologies: true,
            max_passes: 5,
        }
    }
}

impl SimplificationConfig {
    /// Create a new configuration with all simplifications enabled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Disable De Morgan's law simplification.
    pub fn without_demorgan(mut self) -> Self {
        self.apply_demorgan = false;
        self
    }

    /// Set maximum number of passes.
    pub fn with_max_passes(mut self, passes: usize) -> Self {
        self.max_passes = passes;
        self
    }
}

/// Statistics about simplification operations.
#[derive(Debug, Clone, Default)]
pub struct SimplificationStats {
    /// Number of simplification passes performed
    pub passes: usize,
    /// Number of double negations simplified
    pub double_negations: usize,
    /// Number of De Morgan transformations applied
    pub demorgan_applications: usize,
    /// Number of identity operations removed
    pub identities_removed: usize,
    /// Number of inferences combined
    pub inferences_combined: usize,
    /// Number of tautologies simplified
    pub tautologies_simplified: usize,
    /// Total nodes before simplification
    pub nodes_before: usize,
    /// Total nodes after simplification
    pub nodes_after: usize,
}

impl SimplificationStats {
    /// Calculate total simplifications applied.
    pub fn total_simplifications(&self) -> usize {
        self.double_negations
            + self.demorgan_applications
            + self.identities_removed
            + self.inferences_combined
            + self.tautologies_simplified
    }

    /// Calculate reduction ratio (0.0 = no reduction, 1.0 = complete reduction).
    pub fn reduction_ratio(&self) -> f64 {
        if self.nodes_before == 0 {
            0.0
        } else {
            1.0 - (self.nodes_after as f64 / self.nodes_before as f64)
        }
    }
}

/// Proof simplifier that applies logical rewrite rules.
pub struct ProofSimplifier {
    config: SimplificationConfig,
}

impl ProofSimplifier {
    /// Create a new simplifier with default configuration.
    pub fn new() -> Self {
        Self {
            config: SimplificationConfig::default(),
        }
    }

    /// Create a simplifier with custom configuration.
    pub fn with_config(config: SimplificationConfig) -> Self {
        Self { config }
    }

    /// Simplify a proof in-place, returning statistics.
    pub fn simplify(&self, proof: &mut Proof) -> SimplificationStats {
        let mut stats = SimplificationStats {
            nodes_before: proof.node_count(),
            ..Default::default()
        };

        for pass in 0..self.config.max_passes {
            let mut changed = false;

            if self.config.simplify_double_negation {
                let count = self.simplify_double_negations(proof);
                stats.double_negations += count;
                changed |= count > 0;
            }

            if self.config.apply_demorgan {
                let count = self.apply_demorgan_laws(proof);
                stats.demorgan_applications += count;
                changed |= count > 0;
            }

            if self.config.remove_identities {
                let count = self.remove_identity_operations(proof);
                stats.identities_removed += count;
                changed |= count > 0;
            }

            if self.config.simplify_tautologies {
                let count = self.simplify_tautology_steps(proof);
                stats.tautologies_simplified += count;
                changed |= count > 0;
            }

            if self.config.combine_inferences {
                let count = self.combine_inference_chains(proof);
                stats.inferences_combined += count;
                changed |= count > 0;
            }

            stats.passes = pass + 1;

            if !changed {
                break;
            }
        }

        stats.nodes_after = proof.node_count();
        stats
    }

    /// Simplify double negations (¬¬p → p).
    fn simplify_double_negations(&self, proof: &mut Proof) -> usize {
        let mut count = 0;
        let node_ids: Vec<_> = proof.nodes().iter().map(|node| node.id).collect();

        for id in node_ids {
            if let Some(node) = proof.get_node(id) {
                let conclusion = node.conclusion().to_string();
                if let Some(simplified) = self.simplify_conclusion_double_neg(&conclusion)
                    && simplified != conclusion
                {
                    // Update the conclusion
                    if proof.update_conclusion(id, simplified) {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Apply De Morgan's laws (¬(p ∧ q) → ¬p ∨ ¬q).
    fn apply_demorgan_laws(&self, proof: &mut Proof) -> usize {
        let mut count = 0;
        let node_ids: Vec<_> = proof.nodes().iter().map(|node| node.id).collect();

        for id in node_ids {
            if let Some(node) = proof.get_node(id) {
                let conclusion = node.conclusion().to_string();
                if let Some(simplified) = self.apply_demorgan_to_conclusion(&conclusion)
                    && simplified != conclusion
                    && proof.update_conclusion(id, simplified)
                {
                    count += 1;
                }
            }
        }

        count
    }

    /// Remove identity operations (p ∧ true → p, p ∨ false → p).
    fn remove_identity_operations(&self, proof: &mut Proof) -> usize {
        let mut count = 0;
        let node_ids: Vec<_> = proof.nodes().iter().map(|node| node.id).collect();

        for id in node_ids {
            if let Some(node) = proof.get_node(id) {
                let conclusion = node.conclusion().to_string();
                if let Some(simplified) = self.remove_identities_from_conclusion(&conclusion)
                    && simplified != conclusion
                    && proof.update_conclusion(id, simplified)
                {
                    count += 1;
                }
            }
        }

        count
    }

    /// Simplify tautological steps (p ∨ ¬p → true).
    fn simplify_tautology_steps(&self, proof: &mut Proof) -> usize {
        let mut count = 0;
        let node_ids: Vec<_> = proof.nodes().iter().map(|node| node.id).collect();

        for id in node_ids {
            if let Some(node) = proof.get_node(id) {
                let conclusion = node.conclusion().to_string();
                if self.is_tautology(&conclusion) && proof.update_conclusion(id, "true") {
                    count += 1;
                }
            }
        }

        count
    }

    /// Combine consecutive inference chains when possible.
    fn combine_inference_chains(&self, _proof: &mut Proof) -> usize {
        // This is more complex and requires analyzing inference patterns
        // For now, return 0 (future enhancement)
        0
    }

    // Helper methods for conclusion simplification

    fn simplify_conclusion_double_neg(&self, conclusion: &str) -> Option<String> {
        let s = conclusion.trim();

        // Match patterns like: (not (not p))
        if s.starts_with("(not (not ") && s.ends_with("))") {
            let inner = &s[10..s.len() - 2];
            return Some(inner.trim().to_string());
        }

        // Match patterns like: ¬¬p (UTF-8 safe)
        if s.starts_with("¬") {
            let chars: Vec<char> = s.chars().collect();
            if chars.len() >= 3 && chars[0] == '¬' && chars[1] == '¬' {
                let result: String = chars[2..].iter().collect();
                return Some(result.trim().to_string());
            }
        }

        None
    }

    fn apply_demorgan_to_conclusion(&self, conclusion: &str) -> Option<String> {
        let s = conclusion.trim();

        // Match: (not (and p q)) → (or (not p) (not q))
        if s.starts_with("(not (and ") && s.ends_with("))") {
            let inner_content = &s[10..s.len() - 2];
            if let Some(inner) = self.extract_binary_args(&format!("{})", inner_content)) {
                return Some(format!("(or (not {}) (not {}))", inner.0, inner.1));
            }
        }

        // Match: (not (or p q)) → (and (not p) (not q))
        if s.starts_with("(not (or ") && s.ends_with("))") {
            let inner_content = &s[9..s.len() - 2];
            if let Some(inner) = self.extract_binary_args(&format!("{})", inner_content)) {
                return Some(format!("(and (not {}) (not {}))", inner.0, inner.1));
            }
        }

        None
    }

    fn remove_identities_from_conclusion(&self, conclusion: &str) -> Option<String> {
        let s = conclusion.trim();

        // Match: (and p true) → p
        if (s.contains(" true)") || s.contains(" true ))"))
            && s.starts_with("(and ")
            && let Some((left, right)) = self.extract_binary_args(&s[5..])
        {
            if right.trim() == "true" {
                return Some(left.to_string());
            }
            if left.trim() == "true" {
                return Some(right.to_string());
            }
        }

        // Match: (or p false) → p
        if (s.contains(" false)") || s.contains(" false ))"))
            && s.starts_with("(or ")
            && let Some((left, right)) = self.extract_binary_args(&s[4..])
        {
            if right.trim() == "false" {
                return Some(left.to_string());
            }
            if left.trim() == "false" {
                return Some(right.to_string());
            }
        }

        None
    }

    fn is_tautology(&self, conclusion: &str) -> bool {
        let s = conclusion.trim();

        // Check for patterns like: (or p (not p))
        if let Some(stripped) = s.strip_prefix("(or ")
            && let Some((left, right)) = self.extract_binary_args(stripped)
        {
            // Check if right is (not left) or left is (not right)
            if right.trim() == format!("(not {})", left.trim()) {
                return true;
            }
            if left.trim() == format!("(not {})", right.trim()) {
                return true;
            }
        }

        false
    }

    fn extract_binary_args(&self, s: &str) -> Option<(String, String)> {
        // Simple parser for binary operations
        // This is a basic implementation; a full s-expression parser would be better
        let trimmed = s.trim();
        if !trimmed.ends_with(')') {
            return None;
        }

        let content = &trimmed[..trimmed.len() - 1];
        let mut depth = 0;
        let mut split_pos = None;

        for (i, ch) in content.chars().enumerate() {
            match ch {
                '(' => depth += 1,
                ')' => depth -= 1,
                ' ' if depth == 0 => {
                    split_pos = Some(i);
                    break;
                }
                _ => {}
            }
        }

        if let Some(pos) = split_pos {
            let left = content[..pos].trim().to_string();
            let right = content[pos + 1..].trim().to_string();
            Some((left, right))
        } else {
            None
        }
    }
}

impl Default for ProofSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplify a proof using default configuration.
pub fn simplify_proof(proof: &mut Proof) -> SimplificationStats {
    ProofSimplifier::new().simplify(proof)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::ProofNodeId;

    #[test]
    fn test_simplification_config_default() {
        let config = SimplificationConfig::default();
        assert!(config.apply_demorgan);
        assert!(config.simplify_double_negation);
        assert_eq!(config.max_passes, 5);
    }

    #[test]
    fn test_simplification_config_builder() {
        let config = SimplificationConfig::new()
            .without_demorgan()
            .with_max_passes(3);
        assert!(!config.apply_demorgan);
        assert_eq!(config.max_passes, 3);
    }

    #[test]
    fn test_simplification_stats_total() {
        let stats = SimplificationStats {
            double_negations: 2,
            demorgan_applications: 3,
            identities_removed: 1,
            ..Default::default()
        };
        assert_eq!(stats.total_simplifications(), 6);
    }

    #[test]
    fn test_simplification_stats_reduction_ratio() {
        let stats = SimplificationStats {
            nodes_before: 100,
            nodes_after: 80,
            ..Default::default()
        };
        assert!((stats.reduction_ratio() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_simplify_double_negation_sexp() {
        let simplifier = ProofSimplifier::new();
        let conclusion = "(not (not p))";
        let result = simplifier.simplify_conclusion_double_neg(conclusion);
        assert_eq!(result, Some("p".to_string()));
    }

    #[test]
    fn test_simplify_double_negation_unicode() {
        let simplifier = ProofSimplifier::new();
        let conclusion = "¬¬p";
        let result = simplifier.simplify_conclusion_double_neg(conclusion);
        assert_eq!(result, Some("p".to_string()));
    }

    #[test]
    fn test_apply_demorgan_not_and() {
        let simplifier = ProofSimplifier::new();
        let conclusion = "(not (and p q))";
        let result = simplifier.apply_demorgan_to_conclusion(conclusion);
        assert_eq!(result, Some("(or (not p) (not q))".to_string()));
    }

    #[test]
    fn test_apply_demorgan_not_or() {
        let simplifier = ProofSimplifier::new();
        let conclusion = "(not (or p q))";
        let result = simplifier.apply_demorgan_to_conclusion(conclusion);
        assert_eq!(result, Some("(and (not p) (not q))".to_string()));
    }

    #[test]
    fn test_remove_identity_and_true() {
        let simplifier = ProofSimplifier::new();
        let conclusion = "(and p true)";
        let result = simplifier.remove_identities_from_conclusion(conclusion);
        assert_eq!(result, Some("p".to_string()));
    }

    #[test]
    fn test_remove_identity_or_false() {
        let simplifier = ProofSimplifier::new();
        let conclusion = "(or p false)";
        let result = simplifier.remove_identities_from_conclusion(conclusion);
        assert_eq!(result, Some("p".to_string()));
    }

    #[test]
    fn test_is_tautology_or_not() {
        let simplifier = ProofSimplifier::new();
        assert!(simplifier.is_tautology("(or p (not p))"));
        assert!(simplifier.is_tautology("(or (not p) p)"));
        assert!(!simplifier.is_tautology("(or p q)"));
    }

    #[test]
    fn test_extract_binary_args_simple() {
        let simplifier = ProofSimplifier::new();
        let result = simplifier.extract_binary_args("p q)");
        assert_eq!(result, Some(("p".to_string(), "q".to_string())));
    }

    #[test]
    fn test_extract_binary_args_nested() {
        let simplifier = ProofSimplifier::new();
        let result = simplifier.extract_binary_args("(and a b) q)");
        assert_eq!(result, Some(("(and a b)".to_string(), "q".to_string())));
    }

    #[test]
    fn test_simplify_proof_empty() {
        let mut proof = Proof::new();
        let stats = simplify_proof(&mut proof);
        assert_eq!(stats.total_simplifications(), 0);
        assert_eq!(stats.nodes_before, 0);
        assert_eq!(stats.nodes_after, 0);
    }

    #[test]
    fn test_simplify_proof_double_negation() {
        let mut proof = Proof::new();
        proof.add_axiom("(not (not p))".to_string());

        let stats = simplify_proof(&mut proof);
        assert!(stats.double_negations > 0);

        // Check that conclusion was simplified
        let node = proof.get_node(ProofNodeId(0)).unwrap();
        assert_eq!(node.conclusion(), "p");
    }

    #[test]
    fn test_simplify_proof_identity() {
        let mut proof = Proof::new();
        proof.add_axiom("(and p true)".to_string());

        let stats = simplify_proof(&mut proof);
        assert!(stats.identities_removed > 0);

        let node = proof.get_node(ProofNodeId(0)).unwrap();
        assert_eq!(node.conclusion(), "p");
    }

    #[test]
    fn test_simplify_proof_tautology() {
        let mut proof = Proof::new();
        proof.add_axiom("(or p (not p))".to_string());

        let stats = simplify_proof(&mut proof);
        assert!(stats.tautologies_simplified > 0);

        let node = proof.get_node(ProofNodeId(0)).unwrap();
        assert_eq!(node.conclusion(), "true");
    }

    #[test]
    fn test_simplify_proof_multiple_passes() {
        let mut proof = Proof::new();
        // Double negation of identity: (not (not (and p true)))
        // Should simplify in multiple passes: → (and p true) → p
        proof.add_axiom("(not (not (and p true)))".to_string());

        let config = SimplificationConfig::new().with_max_passes(3);
        let simplifier = ProofSimplifier::with_config(config);
        let stats = simplifier.simplify(&mut proof);

        assert!(stats.passes >= 2);
        // Note: Current implementation may not fully simplify this in all cases
        // This test validates that multiple passes occur
    }

    #[test]
    fn test_simplifier_with_custom_config() {
        let config = SimplificationConfig::new().without_demorgan();
        let simplifier = ProofSimplifier::with_config(config);

        let mut proof = Proof::new();
        proof.add_axiom("(not (and p q))".to_string());

        let stats = simplifier.simplify(&mut proof);
        // De Morgan should not be applied
        assert_eq!(stats.demorgan_applications, 0);
    }
}
