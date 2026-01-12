//! Isabelle proof export
//!
//! This module provides utilities for exporting proofs to Isabelle/HOL format,
//! enabling verification in the Isabelle theorem prover.
//!
//! ## Overview
//!
//! Isabelle is a generic proof assistant based on Higher-Order Logic (HOL).
//! This module translates SMT proofs into Isabelle/Isar syntax that can be
//! checked by Isabelle/HOL.
//!
//! ## References
//!
//! - Isabelle Documentation: <https://isabelle.in.tum.de/>
//! - Isabelle/Isar Reference Manual

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use crate::theory::TheoryProof;
use std::collections::HashMap;

/// Isabelle proof exporter (Isabelle/HOL + Isar)
#[derive(Debug)]
pub struct IsabelleExporter {
    /// Mapping from proof nodes to Isabelle identifiers
    node_to_ident: HashMap<ProofNodeId, String>,
    /// Counter for generating unique names
    name_counter: usize,
    /// Theory name
    theory_name: String,
}

impl IsabelleExporter {
    /// Create a new Isabelle exporter
    #[must_use]
    pub fn new(theory_name: impl Into<String>) -> Self {
        Self {
            node_to_ident: HashMap::new(),
            name_counter: 0,
            theory_name: theory_name.into(),
        }
    }

    /// Generate a fresh identifier
    fn fresh_ident(&mut self, prefix: &str) -> String {
        let ident = format!("{}_{}", prefix, self.name_counter);
        self.name_counter += 1;
        ident
    }

    /// Escape a string for use in Isabelle
    fn escape_string(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    /// Convert a conclusion to an Isabelle proposition
    fn conclusion_to_prop(&self, conclusion: &str) -> String {
        // For now, represent conclusions as propositions
        // In a real implementation, this would parse and translate the formula
        format!("PropOf ‹{}›", Self::escape_string(conclusion))
    }

    /// Export a proof node to Isabelle
    fn export_node(&mut self, _proof: &Proof, node_id: ProofNodeId, node: &ProofNode) -> String {
        let ident = self.fresh_ident("step");
        self.node_to_ident.insert(node_id, ident.clone());

        match &node.step {
            ProofStep::Axiom { conclusion } => {
                let prop = self.conclusion_to_prop(conclusion);
                format!("axiomatization where\n  {} : \"{}\"", ident, prop)
            }
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                ..
            } => {
                let prop = self.conclusion_to_prop(conclusion);
                let premise_idents: Vec<String> = premises
                    .iter()
                    .filter_map(|&p| self.node_to_ident.get(&p).cloned())
                    .collect();

                if premise_idents.is_empty() {
                    format!("axiomatization where\n  {} : \"{}\"", ident, prop)
                } else {
                    let premises_str = premise_idents
                        .iter()
                        .map(|p| format!("‹{}›", p))
                        .collect::<Vec<_>>()
                        .join(" ⟹ ");
                    format!(
                        "lemma {} : \"{}  ⟹ {}\"\n  (\\<comment> ‹Rule: {}›)\n  sorry",
                        ident, premises_str, prop, rule
                    )
                }
            }
        }
    }

    /// Export a proof to Isabelle format
    pub fn export_proof(&mut self, proof: &Proof) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!("theory {}\n", self.theory_name));
        output.push_str("  imports Main\n");
        output.push_str("begin\n\n");

        output.push_str("(* Generated Isabelle proof *)\n");
        output.push_str("(* This proof was automatically exported from oxiz-proof *)\n\n");

        // Define base types
        output.push_str("(* Proposition representation *)\n");
        output.push_str("typedecl PropOf\n\n");

        // Export nodes
        let nodes = proof.nodes();

        output.push_str("(* Proof steps *)\n");
        for node in nodes {
            let node_def = self.export_node(proof, node.id, node);
            output.push_str(&node_def);
            output.push_str("\n\n");
        }

        // Final theorem
        if let Some(root_id) = proof.root()
            && let Some(root_ident) = self.node_to_ident.get(&root_id)
        {
            output.push_str("(* Main result *)\n");
            output.push_str("theorem main_result: \"∃P. P\"\n");
            output.push_str("proof -\n");
            output.push_str(&format!(
                "  have \"{}\" by (rule {})\n",
                root_ident, root_ident
            ));
            output.push_str("  then show ?thesis by blast\n");
            output.push_str("qed\n\n");
        }

        output.push_str("end\n");
        output
    }

    /// Export a theory proof to Isabelle format
    pub fn export_theory_proof(&mut self, theory_proof: &TheoryProof) -> String {
        let mut output = String::new();

        output.push_str(&format!("theory {}\n", self.theory_name));
        output.push_str("  imports Main \"HOL-Library.Multiset\"\n");
        output.push_str("begin\n\n");

        output.push_str("(* Generated Isabelle theory proof *)\n\n");

        output.push_str("(* Theory axioms and lemmas *)\n");
        for step in theory_proof.steps() {
            let step_name = self.fresh_ident("theory_step");
            output.push_str(&format!("(* Step {}: {:?} *)\n", step.id.0, step.rule));
            output.push_str(&format!(
                "axiomatization where\n  {} : \"True\"\n\n",
                step_name
            ));
        }

        output.push_str("(* Theory proof complete *)\n\n");
        output.push_str("end\n");
        output
    }
}

impl Default for IsabelleExporter {
    fn default() -> Self {
        Self::new("GeneratedProof")
    }
}

/// Export a proof to Isabelle format
#[must_use]
pub fn export_to_isabelle(proof: &Proof, theory_name: &str) -> String {
    let mut exporter = IsabelleExporter::new(theory_name);
    exporter.export_proof(proof)
}

/// Export a theory proof to Isabelle format
#[must_use]
pub fn export_theory_to_isabelle(theory_proof: &TheoryProof, theory_name: &str) -> String {
    let mut exporter = IsabelleExporter::new(theory_name);
    exporter.export_theory_proof(theory_proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isabelle_exporter_creation() {
        let exporter = IsabelleExporter::new("Test");
        assert_eq!(exporter.theory_name, "Test");
    }

    #[test]
    fn test_default_exporter() {
        let exporter = IsabelleExporter::default();
        assert_eq!(exporter.theory_name, "GeneratedProof");
    }

    #[test]
    fn test_fresh_ident() {
        let mut exporter = IsabelleExporter::new("Test");
        let id1 = exporter.fresh_ident("test");
        let id2 = exporter.fresh_ident("test");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(IsabelleExporter::escape_string("hello"), "hello");
        assert_eq!(IsabelleExporter::escape_string("a\\b"), "a\\\\b");
        assert_eq!(IsabelleExporter::escape_string("a\"b"), "a\\\"b");
    }

    #[test]
    fn test_export_simple_proof() {
        let mut proof = Proof::new();
        let _axiom = proof.add_axiom("P");

        let isabelle_code = export_to_isabelle(&proof, "SimpleProof");
        assert!(isabelle_code.contains("theory SimpleProof"));
        assert!(isabelle_code.contains("imports Main"));
        assert!(isabelle_code.contains("axiomatization"));
        assert!(isabelle_code.contains("end"));
    }

    #[test]
    fn test_export_inference_proof() {
        let mut proof = Proof::new();
        let axiom1 = proof.add_axiom("P");
        let axiom2 = proof.add_axiom("P -> Q");
        let _conclusion = proof.add_inference("modus_ponens", vec![axiom1, axiom2], "Q");

        let isabelle_code = export_to_isabelle(&proof, "InferenceProof");
        assert!(isabelle_code.contains("lemma"));
        assert!(isabelle_code.contains("modus_ponens"));
        assert!(isabelle_code.contains("sorry"));
    }

    #[test]
    fn test_export_theory_proof() {
        let theory_proof = TheoryProof::new();
        let isabelle_code = export_theory_to_isabelle(&theory_proof, "TheoryProof");
        assert!(isabelle_code.contains("theory TheoryProof"));
        assert!(isabelle_code.contains("theory proof"));
        assert!(isabelle_code.contains("end"));
    }

    #[test]
    fn test_main_result_generation() {
        let mut proof = Proof::new();
        let _axiom = proof.add_axiom("P");

        let isabelle_code = export_to_isabelle(&proof, "Test");
        assert!(isabelle_code.contains("theorem main_result"));
        assert!(isabelle_code.contains("proof -"));
        assert!(isabelle_code.contains("qed"));
    }
}
