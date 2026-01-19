//! Natural language proof explanations and analysis.
//!
//! This module provides utilities for generating human-readable explanations
//! of proof steps, useful for debugging, education, and proof understanding.

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use std::fmt;

/// Verbosity level for proof explanations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    /// Minimal explanation (just rule names)
    Minimal,
    /// Concise explanation (rule + premise count)
    Concise,
    /// Detailed explanation (full premise list)
    Detailed,
    /// Verbose explanation (with conclusions)
    Verbose,
}

/// A single explained proof step.
#[derive(Debug, Clone)]
pub struct ExplainedStep {
    /// The node ID
    pub node_id: ProofNodeId,
    /// Natural language explanation
    pub explanation: String,
    /// Depth in the proof tree
    pub depth: u32,
}

impl fmt::Display for ExplainedStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.node_id, self.explanation)
    }
}

/// Proof explanation generator.
pub struct ProofExplainer {
    verbosity: Verbosity,
}

impl ProofExplainer {
    /// Create a new proof explainer with default verbosity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            verbosity: Verbosity::Concise,
        }
    }

    /// Create a proof explainer with specific verbosity.
    #[must_use]
    pub fn with_verbosity(verbosity: Verbosity) -> Self {
        Self { verbosity }
    }

    /// Explain a single proof node.
    #[must_use]
    pub fn explain_node(&self, proof: &Proof, node: &ProofNode) -> ExplainedStep {
        let explanation = match &node.step {
            ProofStep::Axiom { conclusion } => self.explain_axiom(conclusion),
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                args,
            } => self.explain_inference(proof, rule, premises, conclusion, args),
        };

        ExplainedStep {
            node_id: node.id,
            explanation,
            depth: node.depth,
        }
    }

    /// Explain an axiom step.
    fn explain_axiom(&self, conclusion: &str) -> String {
        match self.verbosity {
            Verbosity::Minimal => "Axiom".to_string(),
            Verbosity::Concise => "Assumed as axiom".to_string(),
            Verbosity::Detailed | Verbosity::Verbose => {
                format!("Assumed '{}' as an axiom", conclusion)
            }
        }
    }

    /// Explain an inference step.
    fn explain_inference(
        &self,
        proof: &Proof,
        rule: &str,
        premises: &[ProofNodeId],
        conclusion: &str,
        args: &[String],
    ) -> String {
        let rule_desc = self.describe_rule(rule);

        match self.verbosity {
            Verbosity::Minimal => rule_desc.to_string(),
            Verbosity::Concise => {
                if premises.is_empty() {
                    format!("Applied {}", rule_desc)
                } else {
                    format!("Applied {} using {} premise(s)", rule_desc, premises.len())
                }
            }
            Verbosity::Detailed => {
                if premises.is_empty() {
                    format!("Applied {}", rule_desc)
                } else {
                    let premise_ids: Vec<String> =
                        premises.iter().map(|id| id.to_string()).collect();
                    format!(
                        "Applied {} using premises [{}]",
                        rule_desc,
                        premise_ids.join(", ")
                    )
                }
            }
            Verbosity::Verbose => {
                let mut parts = vec![format!("Applied {}", rule_desc)];

                if !premises.is_empty() {
                    let premise_descs: Vec<String> = premises
                        .iter()
                        .filter_map(|&id| proof.get_node(id).map(|n| (id, n)))
                        .map(|(id, n)| match &n.step {
                            ProofStep::Axiom { conclusion } => format!("{} ({})", id, conclusion),
                            ProofStep::Inference { conclusion, .. } => {
                                format!("{} ({})", id, conclusion)
                            }
                        })
                        .collect();
                    parts.push(format!("from [{}]", premise_descs.join(", ")));
                }

                if !args.is_empty() {
                    parts.push(format!("with arguments [{}]", args.join(", ")));
                }

                parts.push(format!("to derive '{}'", conclusion));
                parts.join(" ")
            }
        }
    }

    /// Describe a rule in natural language.
    fn describe_rule<'a>(&self, rule: &'a str) -> &'a str {
        match rule {
            "and" => "conjunction introduction",
            "or" => "disjunction introduction",
            "not" => "negation introduction",
            "implies" => "implication introduction",
            "resolution" => "resolution",
            "unit_propagation" => "unit propagation",
            "modus_ponens" => "modus ponens",
            "transitivity" => "transitivity",
            "symmetry" => "symmetry",
            "reflexivity" => "reflexivity",
            "congruence" => "congruence",
            "farkas" => "Farkas lemma",
            "split" => "case split",
            "instantiate" => "quantifier instantiation",
            "skolemize" => "skolemization",
            _ => rule,
        }
    }

    /// Explain an entire proof.
    #[must_use]
    pub fn explain_proof(&self, proof: &Proof) -> Vec<ExplainedStep> {
        proof
            .nodes()
            .iter()
            .map(|node| self.explain_node(proof, node))
            .collect()
    }

    /// Generate a step-by-step summary of the proof.
    #[must_use]
    pub fn summarize_proof(&self, proof: &Proof) -> String {
        let explanations = self.explain_proof(proof);

        if explanations.is_empty() {
            return "Empty proof".to_string();
        }

        let mut summary = String::new();
        summary.push_str(&format!("Proof Summary ({} steps):\n", explanations.len()));
        summary.push_str("─".repeat(60).as_str());
        summary.push('\n');

        for (i, step) in explanations.iter().enumerate() {
            let indent = "  ".repeat(step.depth as usize);
            summary.push_str(&format!(
                "{}{}. {} (depth {})\n",
                indent,
                i + 1,
                step.explanation,
                step.depth
            ));
        }

        summary
    }

    /// Identify the critical path in a proof (longest path from axiom to conclusion).
    #[must_use]
    pub fn critical_path(&self, proof: &Proof) -> Vec<ProofNodeId> {
        if proof.is_empty() {
            return Vec::new();
        }

        // Find the node with maximum depth
        let max_depth_node = proof
            .nodes()
            .iter()
            .max_by_key(|n| n.depth)
            .expect("proof has at least one node");

        // Trace back to axioms
        let mut path = vec![max_depth_node.id];
        let mut current = max_depth_node;

        while let ProofStep::Inference { premises, .. } = &current.step {
            if premises.is_empty() {
                break;
            }

            // Find the premise with maximum depth
            let max_premise = premises
                .iter()
                .filter_map(|&id| proof.get_node(id))
                .max_by_key(|n| n.depth);

            if let Some(premise) = max_premise {
                path.push(premise.id);
                current = premise;
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    /// Analyze proof complexity.
    #[must_use]
    pub fn analyze_complexity(&self, proof: &Proof) -> ProofComplexity {
        let total_steps = proof.len();
        let max_depth = proof.depth();

        let axiom_count = proof
            .nodes()
            .iter()
            .filter(|n| matches!(n.step, ProofStep::Axiom { .. }))
            .count();

        let inference_count = total_steps - axiom_count;

        let avg_premises = if inference_count > 0 {
            proof
                .nodes()
                .iter()
                .filter_map(|n| match &n.step {
                    ProofStep::Inference { premises, .. } => Some(premises.len()),
                    _ => None,
                })
                .sum::<usize>() as f64
                / inference_count as f64
        } else {
            0.0
        };

        let branching_factor = if max_depth > 0 {
            total_steps as f64 / max_depth as f64
        } else {
            0.0
        };

        ProofComplexity {
            total_steps,
            max_depth,
            axiom_count,
            inference_count,
            avg_premises,
            branching_factor,
        }
    }
}

impl Default for ProofExplainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complexity analysis of a proof.
#[derive(Debug, Clone)]
pub struct ProofComplexity {
    /// Total number of steps
    pub total_steps: usize,
    /// Maximum depth
    pub max_depth: u32,
    /// Number of axioms
    pub axiom_count: usize,
    /// Number of inferences
    pub inference_count: usize,
    /// Average number of premises per inference
    pub avg_premises: f64,
    /// Branching factor (steps per depth level)
    pub branching_factor: f64,
}

impl fmt::Display for ProofComplexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Proof Complexity Analysis:")?;
        writeln!(f, "  Total steps: {}", self.total_steps)?;
        writeln!(f, "  Maximum depth: {}", self.max_depth)?;
        writeln!(f, "  Axioms: {}", self.axiom_count)?;
        writeln!(f, "  Inferences: {}", self.inference_count)?;
        writeln!(f, "  Avg premises per inference: {:.2}", self.avg_premises)?;
        writeln!(f, "  Branching factor: {:.2}", self.branching_factor)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explain_axiom() {
        let explainer = ProofExplainer::new();
        let mut proof = Proof::new();
        let id = proof.add_axiom("p");
        let node = proof.get_node(id).unwrap();

        let explained = explainer.explain_node(&proof, node);
        assert!(explained.explanation.contains("Assumed"));
    }

    #[test]
    fn test_explain_inference() {
        let explainer = ProofExplainer::new();
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let node = proof.get_node(i1).unwrap();

        let explained = explainer.explain_node(&proof, node);
        assert!(explained.explanation.contains("conjunction"));
    }

    #[test]
    fn test_verbosity_levels() {
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let node = proof.get_node(i1).unwrap();

        let minimal = ProofExplainer::with_verbosity(Verbosity::Minimal);
        let concise = ProofExplainer::with_verbosity(Verbosity::Concise);
        let detailed = ProofExplainer::with_verbosity(Verbosity::Detailed);
        let verbose = ProofExplainer::with_verbosity(Verbosity::Verbose);

        let exp_min = minimal.explain_node(&proof, node);
        let exp_con = concise.explain_node(&proof, node);
        let exp_det = detailed.explain_node(&proof, node);
        let exp_ver = verbose.explain_node(&proof, node);

        // Verbosity should increase explanation length
        assert!(exp_min.explanation.len() < exp_con.explanation.len());
        assert!(exp_det.explanation.len() > exp_con.explanation.len());
        assert!(exp_ver.explanation.len() > exp_det.explanation.len());
    }

    #[test]
    fn test_explain_proof() {
        let explainer = ProofExplainer::new();
        let mut proof = Proof::new();
        proof.add_axiom("p");
        proof.add_axiom("q");

        let explanations = explainer.explain_proof(&proof);
        assert_eq!(explanations.len(), 2);
    }

    #[test]
    fn test_summarize_proof() {
        let explainer = ProofExplainer::new();
        let mut proof = Proof::new();
        proof.add_axiom("p");
        proof.add_axiom("q");

        let summary = explainer.summarize_proof(&proof);
        assert!(summary.contains("Proof Summary"));
        assert!(summary.contains("2 steps"));
    }

    #[test]
    fn test_critical_path() {
        let explainer = ProofExplainer::new();
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        let i1 = proof.add_inference("and", vec![a1, a2], "p /\\ q");
        let i2 = proof.add_inference("not", vec![i1], "!(p /\\ q)");

        let path = explainer.critical_path(&proof);
        assert!(path.len() >= 2);
        assert_eq!(*path.last().unwrap(), i2);
    }

    #[test]
    fn test_analyze_complexity() {
        let explainer = ProofExplainer::new();
        let mut proof = Proof::new();
        let a1 = proof.add_axiom("p");
        let a2 = proof.add_axiom("q");
        proof.add_inference("and", vec![a1, a2], "p /\\ q");

        let complexity = explainer.analyze_complexity(&proof);
        assert_eq!(complexity.total_steps, 3);
        assert_eq!(complexity.axiom_count, 2);
        assert_eq!(complexity.inference_count, 1);
        assert_eq!(complexity.max_depth, 1);
    }

    #[test]
    fn test_empty_proof_complexity() {
        let explainer = ProofExplainer::new();
        let proof = Proof::new();

        let complexity = explainer.analyze_complexity(&proof);
        assert_eq!(complexity.total_steps, 0);
        assert_eq!(complexity.max_depth, 0);
    }

    #[test]
    fn test_rule_descriptions() {
        let explainer = ProofExplainer::new();

        assert_eq!(explainer.describe_rule("and"), "conjunction introduction");
        assert_eq!(explainer.describe_rule("resolution"), "resolution");
        assert_eq!(explainer.describe_rule("farkas"), "Farkas lemma");
        assert_eq!(explainer.describe_rule("unknown"), "unknown");
    }

    #[test]
    fn test_complexity_display() {
        let complexity = ProofComplexity {
            total_steps: 10,
            max_depth: 5,
            axiom_count: 3,
            inference_count: 7,
            avg_premises: 2.5,
            branching_factor: 2.0,
        };

        let display = complexity.to_string();
        assert!(display.contains("Total steps: 10"));
        assert!(display.contains("Maximum depth: 5"));
    }
}
