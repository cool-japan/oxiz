//! Advanced proof statistics and analysis.
//!
//! This module provides detailed statistics and metrics for analyzing
//! proof quality, complexity, and structure.

use crate::proof::{Proof, ProofStep};
use crate::theory::{TheoryProof, TheoryRule};
use std::collections::HashMap;

/// Detailed proof statistics.
#[derive(Debug, Clone)]
pub struct DetailedProofStats {
    /// Total number of steps.
    pub total_steps: usize,
    /// Number of axiom/assumption steps.
    pub axioms: usize,
    /// Number of inference steps.
    pub inferences: usize,
    /// Maximum proof depth.
    pub max_depth: usize,
    /// Average depth of leaves.
    pub avg_leaf_depth: f64,
    /// Total number of premises used.
    pub total_premises: usize,
    /// Average premises per inference.
    pub avg_premises: f64,
    /// Number of unique conclusions.
    pub unique_conclusions: usize,
    /// Rule usage counts.
    pub rule_usage: HashMap<String, usize>,
    /// Proof complexity score (higher = more complex).
    pub complexity_score: f64,
}

impl DetailedProofStats {
    /// Compute detailed statistics for a proof.
    #[must_use]
    pub fn compute(proof: &Proof) -> Self {
        let total_steps = proof.node_count();
        let max_depth = proof.depth();
        let leaf_nodes = proof.leaf_nodes();
        let axioms = leaf_nodes.len();
        let inferences = total_steps - axioms;

        let mut total_premises = 0;
        let mut unique_conclusions = std::collections::HashSet::new();
        let mut rule_usage: HashMap<String, usize> = HashMap::new();

        for node in proof.nodes() {
            match &node.step {
                ProofStep::Axiom { conclusion } => {
                    unique_conclusions.insert(conclusion.clone());
                }
                ProofStep::Inference {
                    rule,
                    premises,
                    conclusion,
                    ..
                } => {
                    total_premises += premises.len();
                    unique_conclusions.insert(conclusion.clone());
                    *rule_usage.entry(rule.clone()).or_insert(0) += 1;
                }
            }
        }

        let avg_premises = if inferences > 0 {
            total_premises as f64 / inferences as f64
        } else {
            0.0
        };

        // Compute average leaf depth
        let mut total_leaf_depth = 0;
        if let Some(root) = proof.root() {
            for &leaf in &leaf_nodes {
                total_leaf_depth += compute_depth_to_node(proof, root, leaf);
            }
        }
        let avg_leaf_depth = if !leaf_nodes.is_empty() {
            total_leaf_depth as f64 / leaf_nodes.len() as f64
        } else {
            0.0
        };

        // Compute complexity score
        // Factors: depth, branching, unique rules, premise count
        let depth_factor = max_depth as f64;
        let branching_factor = avg_premises;
        let rule_diversity = rule_usage.len() as f64;
        let complexity_score = (depth_factor * branching_factor * rule_diversity).sqrt();

        Self {
            total_steps,
            axioms,
            inferences,
            max_depth: max_depth as usize,
            avg_leaf_depth,
            total_premises,
            avg_premises,
            unique_conclusions: unique_conclusions.len(),
            rule_usage,
            complexity_score,
        }
    }

    /// Compute detailed statistics for a theory proof.
    #[must_use]
    pub fn compute_theory(proof: &TheoryProof) -> TheoryProofStats {
        let total_steps = proof.len();
        let mut axioms = 0;
        let mut inferences = 0;
        let mut total_premises = 0;
        let mut rule_usage: HashMap<String, usize> = HashMap::new();
        let mut theory_usage: HashMap<String, usize> = HashMap::new();

        for step in proof.steps() {
            if step.premises.is_empty() {
                axioms += 1;
            } else {
                inferences += 1;
                total_premises += step.premises.len();
            }

            let rule_name = format!("{}", step.rule);
            *rule_usage.entry(rule_name).or_insert(0) += 1;

            // Categorize by theory
            let theory = categorize_theory_rule(&step.rule);
            *theory_usage.entry(theory).or_insert(0) += 1;
        }

        let avg_premises = if inferences > 0 {
            total_premises as f64 / inferences as f64
        } else {
            0.0
        };

        TheoryProofStats {
            total_steps,
            axioms,
            inferences,
            total_premises,
            avg_premises,
            rule_usage,
            theory_usage,
        }
    }
}

/// Statistics specific to theory proofs.
#[derive(Debug, Clone)]
pub struct TheoryProofStats {
    /// Total number of steps.
    pub total_steps: usize,
    /// Number of axiom steps.
    pub axioms: usize,
    /// Number of inference steps.
    pub inferences: usize,
    /// Total premises used.
    pub total_premises: usize,
    /// Average premises per inference.
    pub avg_premises: f64,
    /// Rule usage counts.
    pub rule_usage: HashMap<String, usize>,
    /// Theory usage counts (EUF, Arithmetic, Arrays, etc.).
    pub theory_usage: HashMap<String, usize>,
}

/// Compute depth from root to a specific node.
fn compute_depth_to_node(
    proof: &Proof,
    current: crate::proof::ProofNodeId,
    target: crate::proof::ProofNodeId,
) -> usize {
    if current == target {
        return 0;
    }

    if let Some(premises) = proof.premises(current) {
        for &premise in premises {
            let depth = compute_depth_to_node(proof, premise, target);
            if depth > 0 || premise == target {
                return depth + 1;
            }
        }
    }

    0
}

/// Categorize a theory rule by its theory.
fn categorize_theory_rule(rule: &TheoryRule) -> String {
    match rule {
        TheoryRule::Refl
        | TheoryRule::Symm
        | TheoryRule::Trans
        | TheoryRule::Cong
        | TheoryRule::FuncEq
        | TheoryRule::Distinct => "EUF".to_string(),

        TheoryRule::LaGeneric
        | TheoryRule::LaMult
        | TheoryRule::LaAdd
        | TheoryRule::LaTighten
        | TheoryRule::LaDiv
        | TheoryRule::LaTotality
        | TheoryRule::LaDiseq => "Arithmetic".to_string(),

        TheoryRule::BvBlastEq
        | TheoryRule::BvExtract
        | TheoryRule::BvConcat
        | TheoryRule::BvZeroExtend
        | TheoryRule::BvSignExtend
        | TheoryRule::BvBitwise
        | TheoryRule::BvArith
        | TheoryRule::BvCompare
        | TheoryRule::BvOverflow => "BitVector".to_string(),

        TheoryRule::ArrReadWrite1
        | TheoryRule::ArrReadWrite2
        | TheoryRule::ArrExt
        | TheoryRule::ArrConst => "Array".to_string(),

        TheoryRule::ForallElim
        | TheoryRule::ExistsIntro
        | TheoryRule::Skolemize
        | TheoryRule::QuantInst
        | TheoryRule::AlphaEquiv => "Quantifier".to_string(),

        TheoryRule::TheoryConflict | TheoryRule::TheoryProp => "Theory".to_string(),

        TheoryRule::IteElim | TheoryRule::BoolSimp => "Boolean".to_string(),

        TheoryRule::Custom(name) => format!("Custom({})", name),
    }
}

/// Proof quality metrics.
#[derive(Debug, Clone)]
pub struct ProofQuality {
    /// Redundancy ratio (0.0 = no redundancy, 1.0 = highly redundant).
    pub redundancy: f64,
    /// Efficiency score (higher = more efficient).
    pub efficiency: f64,
    /// Compactness score (higher = more compact).
    pub compactness: f64,
    /// Overall quality score (0-100).
    pub overall_score: f64,
}

impl ProofQuality {
    /// Compute quality metrics for a proof.
    #[must_use]
    pub fn compute(stats: &DetailedProofStats) -> Self {
        // Redundancy: ratio of duplicate conclusions
        let redundancy = if stats.total_steps > 0 {
            1.0 - (stats.unique_conclusions as f64 / stats.total_steps as f64)
        } else {
            0.0
        };

        // Efficiency: inversely proportional to average premises
        let efficiency = if stats.avg_premises > 0.0 {
            1.0 / (1.0 + stats.avg_premises)
        } else {
            1.0
        };

        // Compactness: inversely proportional to proof size relative to depth
        let compactness = if stats.total_steps > 0 && stats.max_depth > 0 {
            stats.max_depth as f64 / stats.total_steps as f64
        } else {
            1.0
        };

        // Overall score (0-100)
        let overall_score =
            ((1.0 - redundancy) * 0.3 + efficiency * 0.4 + compactness * 0.3) * 100.0;

        Self {
            redundancy,
            efficiency,
            compactness,
            overall_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detailed_proof_stats() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let _p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        let stats = DetailedProofStats::compute(&proof);

        assert_eq!(stats.total_steps, 3);
        assert_eq!(stats.axioms, 2);
        assert_eq!(stats.inferences, 1);
        assert_eq!(stats.unique_conclusions, 3);
        assert_eq!(stats.rule_usage.get("and"), Some(&1));
    }

    #[test]
    fn test_theory_proof_stats() {
        let mut proof = TheoryProof::new();

        let s1 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= a b)");
        let s2 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= b c)");
        proof.trans(s1, s2, "a", "c");

        let stats = DetailedProofStats::compute_theory(&proof);

        assert_eq!(stats.total_steps, 3);
        assert_eq!(stats.axioms, 2);
        assert_eq!(stats.inferences, 1);
        assert_eq!(stats.theory_usage.get("EUF"), Some(&1));
        assert_eq!(stats.theory_usage.get("Custom(assert)"), Some(&2));
    }

    #[test]
    fn test_categorize_theory_rule() {
        assert_eq!(categorize_theory_rule(&TheoryRule::Refl), "EUF");
        assert_eq!(categorize_theory_rule(&TheoryRule::LaGeneric), "Arithmetic");
        assert_eq!(categorize_theory_rule(&TheoryRule::BvBlastEq), "BitVector");
        assert_eq!(categorize_theory_rule(&TheoryRule::ArrReadWrite1), "Array");
    }

    #[test]
    fn test_proof_quality() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let _p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        let stats = DetailedProofStats::compute(&proof);
        let quality = ProofQuality::compute(&stats);

        assert!(quality.redundancy >= 0.0 && quality.redundancy <= 1.0);
        assert!(quality.efficiency > 0.0);
        assert!(quality.compactness > 0.0);
        assert!(quality.overall_score >= 0.0 && quality.overall_score <= 100.0);
    }

    #[test]
    fn test_complexity_score() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let _p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");

        let stats = DetailedProofStats::compute(&proof);

        assert!(stats.complexity_score > 0.0);
    }

    #[test]
    fn test_avg_leaf_depth() {
        let mut proof = Proof::new();
        let p1 = proof.add_axiom("p");
        let p2 = proof.add_axiom("q");
        let p3 = proof.add_inference("and", vec![p1, p2], "(and p q)");
        let _p4 = proof.add_inference("or", vec![p3], "(or (and p q))");

        let stats = DetailedProofStats::compute(&proof);

        assert!(stats.avg_leaf_depth > 0.0);
    }
}
