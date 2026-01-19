//! Strategy heuristics learned from successful proofs.
//!
//! This module extracts heuristics and strategies from successful proofs
//! to guide the solver in future problem solving.

use crate::proof::{Proof, ProofNodeId, ProofStep};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

/// A heuristic learned from successful proofs.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ProofHeuristic {
    /// Heuristic name
    pub name: String,
    /// Heuristic type
    pub heuristic_type: HeuristicType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Number of proofs supporting this heuristic
    pub support_count: usize,
    /// Average improvement when applied
    pub avg_improvement: f64,
}

/// Types of heuristics that can be learned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeuristicType {
    /// Rule ordering preference
    RuleOrdering { preferred_sequence: Vec<String> },
    /// Branching strategy
    BranchingStrategy { criteria: String },
    /// Lemma selection
    LemmaSelection { pattern: String },
    /// Instantiation preference
    InstantiationPreference { trigger_pattern: String },
    /// Theory combination strategy
    TheoryCombination { theory_order: Vec<String> },
}

impl fmt::Display for HeuristicType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeuristicType::RuleOrdering { preferred_sequence } => {
                write!(f, "RuleOrdering[{}]", preferred_sequence.join(" → "))
            }
            HeuristicType::BranchingStrategy { criteria } => {
                write!(f, "BranchingStrategy[{}]", criteria)
            }
            HeuristicType::LemmaSelection { pattern } => {
                write!(f, "LemmaSelection[{}]", pattern)
            }
            HeuristicType::InstantiationPreference { trigger_pattern } => {
                write!(f, "InstantiationPreference[{}]", trigger_pattern)
            }
            HeuristicType::TheoryCombination { theory_order } => {
                write!(f, "TheoryCombination[{}]", theory_order.join(" + "))
            }
        }
    }
}

impl fmt::Display for ProofHeuristic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Heuristic: {}", self.name)?;
        writeln!(f, "Type: {}", self.heuristic_type)?;
        writeln!(f, "Confidence: {:.2}", self.confidence)?;
        writeln!(f, "Support: {} proofs", self.support_count)?;
        writeln!(f, "Avg improvement: {:.1}%", self.avg_improvement * 100.0)?;
        Ok(())
    }
}

/// Strategy learner for extracting heuristics from proofs.
pub struct StrategyLearner {
    /// Minimum support count for a heuristic
    min_support: usize,
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Learned heuristics
    heuristics: Vec<ProofHeuristic>,
    /// Rule sequence frequency tracker
    rule_sequences: FxHashMap<Vec<String>, usize>,
}

impl Default for StrategyLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategyLearner {
    /// Create a new strategy learner with default settings.
    pub fn new() -> Self {
        Self {
            min_support: 2,
            min_confidence: 0.5,
            heuristics: Vec::new(),
            rule_sequences: FxHashMap::default(),
        }
    }

    /// Set the minimum support count.
    pub fn with_min_support(mut self, support: usize) -> Self {
        self.min_support = support;
        self
    }

    /// Set the minimum confidence threshold.
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Learn heuristics from a collection of successful proofs.
    pub fn learn_from_proofs(&mut self, proofs: &[&Proof], _proof_stats: &[(f64, f64)]) {
        // Extract rule ordering heuristics
        self.learn_rule_ordering(proofs);

        // Extract branching heuristics
        self.learn_branching_strategies(proofs);

        // Extract lemma selection heuristics
        self.learn_lemma_selection(proofs);

        // Extract instantiation heuristics
        self.learn_instantiation_preferences(proofs);

        // Extract theory combination heuristics
        self.learn_theory_combination(proofs);

        // Filter by confidence and support
        self.heuristics
            .retain(|h| h.confidence >= self.min_confidence && h.support_count >= self.min_support);

        // Sort by confidence
        self.heuristics.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get all learned heuristics.
    pub fn get_heuristics(&self) -> &[ProofHeuristic] {
        &self.heuristics
    }

    /// Get heuristics of a specific type.
    pub fn get_heuristics_by_type(&self, type_name: &str) -> Vec<&ProofHeuristic> {
        self.heuristics
            .iter()
            .filter(|h| {
                matches!(
                    (&h.heuristic_type, type_name),
                    (HeuristicType::RuleOrdering { .. }, "rule_ordering")
                        | (HeuristicType::BranchingStrategy { .. }, "branching")
                        | (HeuristicType::LemmaSelection { .. }, "lemma")
                        | (
                            HeuristicType::InstantiationPreference { .. },
                            "instantiation"
                        )
                        | (HeuristicType::TheoryCombination { .. }, "theory")
                )
            })
            .collect()
    }

    /// Get top N heuristics by confidence.
    pub fn get_top_heuristics(&self, n: usize) -> Vec<&ProofHeuristic> {
        self.heuristics.iter().take(n).collect()
    }

    /// Clear all learned heuristics.
    pub fn clear(&mut self) {
        self.heuristics.clear();
        self.rule_sequences.clear();
    }

    // Helper: Learn rule ordering preferences
    fn learn_rule_ordering(&mut self, proofs: &[&Proof]) {
        let mut sequence_freq: FxHashMap<Vec<String>, usize> = FxHashMap::default();

        for proof in proofs {
            let sequences = self.extract_rule_sequences(proof, 3);
            for seq in sequences {
                *sequence_freq.entry(seq).or_insert(0) += 1;
            }
        }

        // Create heuristics for frequent sequences
        for (seq, count) in sequence_freq.iter() {
            if *count >= self.min_support {
                let confidence = (*count as f64) / (proofs.len() as f64);
                if confidence >= self.min_confidence {
                    self.heuristics.push(ProofHeuristic {
                        name: format!("rule_order_{}", seq.join("_")),
                        heuristic_type: HeuristicType::RuleOrdering {
                            preferred_sequence: seq.clone(),
                        },
                        confidence,
                        support_count: *count,
                        avg_improvement: 0.0,
                    });
                }
            }
        }
    }

    // Helper: Extract rule sequences from a proof
    fn extract_rule_sequences(&self, proof: &Proof, length: usize) -> Vec<Vec<String>> {
        let mut sequences = Vec::new();
        let nodes: Vec<ProofNodeId> = proof.nodes().iter().map(|n| n.id).collect();

        if nodes.len() < length {
            return sequences;
        }

        for window in nodes.windows(length) {
            let seq: Vec<String> = window
                .iter()
                .filter_map(|&id| {
                    proof.get_node(id).and_then(|node| {
                        if let ProofStep::Inference { rule, .. } = &node.step {
                            Some(rule.clone())
                        } else {
                            None
                        }
                    })
                })
                .collect();

            if seq.len() == length {
                sequences.push(seq);
            }
        }

        sequences
    }

    // Helper: Learn branching strategies
    fn learn_branching_strategies(&mut self, proofs: &[&Proof]) {
        let mut branching_patterns: FxHashMap<String, usize> = FxHashMap::default();

        for proof in proofs {
            // Look for nodes with multiple dependents (branching points)
            for node in proof.nodes() {
                let dependents = proof.get_children(node.id);
                if dependents.len() > 1 {
                    // This is a branching point
                    let pattern = self.abstract_branching_pattern(node.conclusion());
                    *branching_patterns.entry(pattern).or_insert(0) += 1;
                }
            }
        }

        // Create heuristics for common branching patterns
        for (pattern, count) in branching_patterns.iter() {
            if *count >= self.min_support {
                let confidence = (*count as f64) / (proofs.len() as f64);
                if confidence >= self.min_confidence {
                    self.heuristics.push(ProofHeuristic {
                        name: format!("branch_{}", pattern),
                        heuristic_type: HeuristicType::BranchingStrategy {
                            criteria: pattern.clone(),
                        },
                        confidence,
                        support_count: *count,
                        avg_improvement: 0.0,
                    });
                }
            }
        }
    }

    // Helper: Abstract branching pattern
    fn abstract_branching_pattern(&self, conclusion: &str) -> String {
        // Simple abstraction - in practice would be more sophisticated
        // Check more specific patterns first
        if conclusion.contains("forall") {
            "universal".to_string()
        } else if conclusion.contains("exists") {
            "existential".to_string()
        } else if conclusion.contains(" or ") {
            "disjunction".to_string()
        } else if conclusion.contains(" and ") {
            "conjunction".to_string()
        } else {
            "other".to_string()
        }
    }

    // Helper: Learn lemma selection patterns
    fn learn_lemma_selection(&mut self, proofs: &[&Proof]) {
        let mut lemma_patterns: FxHashMap<String, usize> = FxHashMap::default();

        for proof in proofs {
            for node in proof.nodes() {
                if let ProofStep::Inference { rule, .. } = &node.step
                    && (rule.contains("lemma") || rule.contains("theory"))
                {
                    let pattern = self.extract_lemma_pattern(node.conclusion());
                    *lemma_patterns.entry(pattern).or_insert(0) += 1;
                }
            }
        }

        for (pattern, count) in lemma_patterns.iter() {
            if *count >= self.min_support {
                let confidence = (*count as f64) / (proofs.len() as f64);
                if confidence >= self.min_confidence {
                    self.heuristics.push(ProofHeuristic {
                        name: format!("lemma_{}", pattern),
                        heuristic_type: HeuristicType::LemmaSelection {
                            pattern: pattern.clone(),
                        },
                        confidence,
                        support_count: *count,
                        avg_improvement: 0.0,
                    });
                }
            }
        }
    }

    // Helper: Extract lemma pattern
    fn extract_lemma_pattern(&self, conclusion: &str) -> String {
        // Extract the type of lemma (simplified)
        // Check more specific patterns first
        if conclusion.contains("congruence") {
            "congruence".to_string()
        } else if conclusion.contains("<=") || conclusion.contains(">=") {
            "inequality".to_string()
        } else if conclusion.contains("=") {
            "equality".to_string()
        } else {
            "other".to_string()
        }
    }

    // Helper: Learn instantiation preferences
    fn learn_instantiation_preferences(&mut self, proofs: &[&Proof]) {
        let mut instantiation_patterns: FxHashMap<String, usize> = FxHashMap::default();

        for proof in proofs {
            for node in proof.nodes() {
                if let ProofStep::Inference { rule, .. } = &node.step
                    && (rule.contains("instantiation") || rule.contains("forall_elim"))
                {
                    let pattern = self.extract_trigger_pattern(node.conclusion());
                    *instantiation_patterns.entry(pattern).or_insert(0) += 1;
                }
            }
        }

        for (pattern, count) in instantiation_patterns.iter() {
            if *count >= self.min_support {
                let confidence = (*count as f64) / (proofs.len() as f64);
                if confidence >= self.min_confidence {
                    self.heuristics.push(ProofHeuristic {
                        name: format!("inst_{}", pattern),
                        heuristic_type: HeuristicType::InstantiationPreference {
                            trigger_pattern: pattern.clone(),
                        },
                        confidence,
                        support_count: *count,
                        avg_improvement: 0.0,
                    });
                }
            }
        }
    }

    // Helper: Extract trigger pattern
    fn extract_trigger_pattern(&self, conclusion: &str) -> String {
        // Extract function applications as triggers (simplified)
        if let Some(start) = conclusion.find('(')
            && let Some(end) = conclusion[start..].find(')')
        {
            return conclusion[..start + end + 1].to_string();
        }
        "default".to_string()
    }

    // Helper: Learn theory combination strategies
    fn learn_theory_combination(&mut self, proofs: &[&Proof]) {
        let mut theory_sequences: FxHashMap<Vec<String>, usize> = FxHashMap::default();

        for proof in proofs {
            let theories = self.extract_theory_sequence(proof);
            if !theories.is_empty() {
                *theory_sequences.entry(theories).or_insert(0) += 1;
            }
        }

        for (seq, count) in theory_sequences.iter() {
            if *count >= self.min_support {
                let confidence = (*count as f64) / (proofs.len() as f64);
                if confidence >= self.min_confidence {
                    self.heuristics.push(ProofHeuristic {
                        name: format!("theory_comb_{}", seq.join("_")),
                        heuristic_type: HeuristicType::TheoryCombination {
                            theory_order: seq.clone(),
                        },
                        confidence,
                        support_count: *count,
                        avg_improvement: 0.0,
                    });
                }
            }
        }
    }

    // Helper: Extract theory sequence from proof
    fn extract_theory_sequence(&self, proof: &Proof) -> Vec<String> {
        let mut seen = FxHashSet::default();
        let mut sequence = Vec::new();

        for node in proof.nodes() {
            if let ProofStep::Inference { rule, .. } = &node.step {
                let theory = self.infer_theory_from_rule(rule);
                if !theory.is_empty() && !seen.contains(&theory) {
                    seen.insert(theory.clone());
                    sequence.push(theory);
                }
            }
        }

        sequence
    }

    // Helper: Infer theory from rule name
    fn infer_theory_from_rule(&self, rule: &str) -> String {
        if rule.contains("arith") || rule.contains("farkas") {
            "arithmetic".to_string()
        } else if rule.contains("euf") || rule.contains("congruence") {
            "euf".to_string()
        } else if rule.contains("array") {
            "arrays".to_string()
        } else if rule.contains("bv") || rule.contains("bitvector") {
            "bitvectors".to_string()
        } else {
            String::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_learner_new() {
        let learner = StrategyLearner::new();
        assert_eq!(learner.min_support, 2);
        assert_eq!(learner.min_confidence, 0.5);
        assert!(learner.heuristics.is_empty());
    }

    #[test]
    fn test_strategy_learner_with_settings() {
        let learner = StrategyLearner::new()
            .with_min_support(3)
            .with_min_confidence(0.7);
        assert_eq!(learner.min_support, 3);
        assert_eq!(learner.min_confidence, 0.7);
    }

    #[test]
    fn test_heuristic_type_display() {
        let rule_ordering = HeuristicType::RuleOrdering {
            preferred_sequence: vec!["resolution".to_string(), "unit_prop".to_string()],
        };
        assert_eq!(
            rule_ordering.to_string(),
            "RuleOrdering[resolution → unit_prop]"
        );

        let branching = HeuristicType::BranchingStrategy {
            criteria: "disjunction".to_string(),
        };
        assert_eq!(branching.to_string(), "BranchingStrategy[disjunction]");
    }

    #[test]
    fn test_proof_heuristic_display() {
        let heuristic = ProofHeuristic {
            name: "test_heuristic".to_string(),
            heuristic_type: HeuristicType::RuleOrdering {
                preferred_sequence: vec!["resolution".to_string()],
            },
            confidence: 0.8,
            support_count: 10,
            avg_improvement: 0.15,
        };
        let display = format!("{}", heuristic);
        assert!(display.contains("test_heuristic"));
        assert!(display.contains("0.80"));
        assert!(display.contains("10 proofs"));
    }

    #[test]
    fn test_clear_heuristics() {
        let mut learner = StrategyLearner::new();
        learner.heuristics.push(ProofHeuristic {
            name: "test".to_string(),
            heuristic_type: HeuristicType::RuleOrdering {
                preferred_sequence: vec![],
            },
            confidence: 0.5,
            support_count: 2,
            avg_improvement: 0.0,
        });
        learner.clear();
        assert!(learner.heuristics.is_empty());
    }

    #[test]
    fn test_get_top_heuristics() {
        let mut learner = StrategyLearner::new();
        learner.heuristics.push(ProofHeuristic {
            name: "h1".to_string(),
            heuristic_type: HeuristicType::RuleOrdering {
                preferred_sequence: vec![],
            },
            confidence: 0.9,
            support_count: 2,
            avg_improvement: 0.0,
        });
        learner.heuristics.push(ProofHeuristic {
            name: "h2".to_string(),
            heuristic_type: HeuristicType::RuleOrdering {
                preferred_sequence: vec![],
            },
            confidence: 0.7,
            support_count: 2,
            avg_improvement: 0.0,
        });
        let top = learner.get_top_heuristics(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].name, "h1");
    }

    #[test]
    fn test_abstract_branching_pattern() {
        let learner = StrategyLearner::new();
        assert_eq!(learner.abstract_branching_pattern("x or y"), "disjunction");
        assert_eq!(learner.abstract_branching_pattern("x and y"), "conjunction");
        assert_eq!(
            learner.abstract_branching_pattern("forall x. P(x)"),
            "universal"
        );
    }

    #[test]
    fn test_extract_lemma_pattern() {
        let learner = StrategyLearner::new();
        assert_eq!(learner.extract_lemma_pattern("x = y"), "equality");
        assert_eq!(learner.extract_lemma_pattern("x <= y"), "inequality");
        assert_eq!(
            learner.extract_lemma_pattern("congruence f(x) f(y)"),
            "congruence"
        );
    }

    #[test]
    fn test_infer_theory_from_rule() {
        let learner = StrategyLearner::new();
        assert_eq!(learner.infer_theory_from_rule("arith_lemma"), "arithmetic");
        assert_eq!(learner.infer_theory_from_rule("euf_congruence"), "euf");
        assert_eq!(
            learner.infer_theory_from_rule("array_extensionality"),
            "arrays"
        );
        assert_eq!(learner.infer_theory_from_rule("bv_solve"), "bitvectors");
    }
}
