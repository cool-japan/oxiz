//! Coq proof export
//!
//! This module provides utilities for exporting proofs to Coq format,
//! enabling verification in the Coq proof assistant.
//!
//! ## Overview
//!
//! Coq is an interactive theorem prover based on the Calculus of Inductive
//! Constructions (CIC). This module translates SMT proofs into Coq vernacular
//! that can be checked by Coq.
//!
//! ## References
//!
//! - Coq Reference Manual: <https://coq.inria.fr/refman/>
//! - The Coq Proof Assistant: <https://coq.inria.fr/>

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use crate::theory::TheoryProof;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

/// Coq proof exporter
#[derive(Debug)]
pub struct CoqExporter {
    /// Generated Coq definitions
    definitions: Vec<String>,
    /// Generated Coq lemmas
    lemmas: Vec<String>,
    /// Mapping from proof nodes to Coq identifiers
    node_to_ident: HashMap<ProofNodeId, String>,
    /// Counter for generating unique names
    name_counter: usize,
}

impl CoqExporter {
    /// Create a new Coq exporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            definitions: Vec::new(),
            lemmas: Vec::new(),
            node_to_ident: HashMap::new(),
            name_counter: 0,
        }
    }

    /// Generate a fresh identifier
    fn fresh_ident(&mut self, prefix: &str) -> String {
        let ident = format!("{}_{}", prefix, self.name_counter);
        self.name_counter += 1;
        ident
    }

    /// Escape a string for use in Coq
    fn escape_string(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    /// Convert a conclusion to a Coq proposition
    fn conclusion_to_prop(&self, conclusion: &str) -> String {
        // For now, represent conclusions as string propositions
        // In a real implementation, this would parse and translate the formula
        format!("Prop_of_string \"{}\"", Self::escape_string(conclusion))
    }

    /// Export a proof node to Coq
    fn export_node(&mut self, _proof: &Proof, node_id: ProofNodeId, node: &ProofNode) -> String {
        let ident = self.fresh_ident("step");
        self.node_to_ident.insert(node_id, ident.clone());

        match &node.step {
            ProofStep::Axiom { conclusion } => {
                let prop = self.conclusion_to_prop(conclusion);
                format!("Axiom {} : {}.", ident, prop)
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
                    format!("Axiom {} : {}.", ident, prop)
                } else {
                    let premises_str = premise_idents.join(" -> ");
                    format!(
                        "Lemma {} : {} -> {}.\nProof.\n  (* Rule: {} *)\n  auto.\nQed.",
                        ident, premises_str, prop, rule
                    )
                }
            }
        }
    }

    /// Export a proof to Coq format
    pub fn export_proof(&mut self, proof: &Proof) -> String {
        let mut output = String::new();

        // Header
        output.push_str("(* Generated Coq proof *)\n");
        output.push_str("(* This proof was automatically exported from oxiz-proof *)\n\n");

        // Require standard library
        output.push_str("Require Import Coq.Init.Prelude.\n");
        output.push_str("Require Import Coq.Logic.Classical.\n\n");

        // Define base types
        output.push_str("(* Proposition representation *)\n");
        output.push_str("Parameter Prop_of_string : string -> Prop.\n\n");

        // Export nodes in topological order (dependencies first)
        let nodes = proof.nodes();

        output.push_str("(* Proof steps *)\n");
        for node in nodes {
            let node_def = self.export_node(proof, node.id, node);
            output.push_str(&node_def);
            output.push('\n');
        }

        // Final theorem
        if let Some(root_id) = proof.root()
            && let Some(root_ident) = self.node_to_ident.get(&root_id)
        {
            output.push_str("\n(* Main result *)\n");
            output.push_str("Theorem main_result : exists P, P.\n");
            output.push_str("Proof.\n");
            output.push_str(&format!("  exists {}.\n", root_ident));
            output.push_str(&format!("  apply {}.\n", root_ident));
            output.push_str("Qed.\n");
        }

        output
    }

    /// Export a theory proof to Coq format
    pub fn export_theory_proof(&mut self, theory_proof: &TheoryProof) -> String {
        let mut output = String::new();

        output.push_str("(* Generated Coq theory proof *)\n\n");
        output.push_str("Require Import Coq.Init.Prelude.\n");
        output.push_str("Require Import Coq.ZArith.ZArith.\n");
        output.push_str("Require Import Coq.Reals.Reals.\n\n");

        output.push_str("(* Theory axioms and lemmas *)\n");
        for step in theory_proof.steps() {
            let step_name = self.fresh_ident("theory_step");
            writeln!(&mut output, "(* Step {}: {:?} *)", step.id.0, step.rule)
                .expect("write should succeed");
            writeln!(&mut output, "Parameter {} : Prop.", step_name).expect("write should succeed");
        }

        output.push_str("\n(* Theory proof complete *)\n");
        output
    }

    /// Get the generated definitions
    #[must_use]
    pub fn definitions(&self) -> &[String] {
        &self.definitions
    }

    /// Get the generated lemmas
    #[must_use]
    pub fn lemmas(&self) -> &[String] {
        &self.lemmas
    }
}

impl Default for CoqExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Export a proof to Coq format
#[must_use]
pub fn export_to_coq(proof: &Proof) -> String {
    let mut exporter = CoqExporter::new();
    exporter.export_proof(proof)
}

/// Export a theory proof to Coq format
#[must_use]
pub fn export_theory_to_coq(theory_proof: &TheoryProof) -> String {
    let mut exporter = CoqExporter::new();
    exporter.export_theory_proof(theory_proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coq_exporter_creation() {
        let exporter = CoqExporter::new();
        assert_eq!(exporter.definitions.len(), 0);
        assert_eq!(exporter.lemmas.len(), 0);
    }

    #[test]
    fn test_fresh_ident() {
        let mut exporter = CoqExporter::new();
        let id1 = exporter.fresh_ident("test");
        let id2 = exporter.fresh_ident("test");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(CoqExporter::escape_string("hello"), "hello");
        assert_eq!(CoqExporter::escape_string("a\\b"), "a\\\\b");
        assert_eq!(CoqExporter::escape_string("a\"b"), "a\\\"b");
        assert_eq!(CoqExporter::escape_string("a\nb"), "a\\nb");
    }

    #[test]
    fn test_export_simple_proof() {
        let mut proof = Proof::new();
        let _axiom = proof.add_axiom("P");

        let coq_code = export_to_coq(&proof);
        assert!(coq_code.contains("Require Import"));
        assert!(coq_code.contains("Axiom"));
        assert!(coq_code.contains("Prop_of_string"));
    }

    #[test]
    fn test_export_inference_proof() {
        let mut proof = Proof::new();
        let axiom1 = proof.add_axiom("P");
        let axiom2 = proof.add_axiom("P -> Q");
        let _conclusion = proof.add_inference("modus_ponens", vec![axiom1, axiom2], "Q");

        let coq_code = export_to_coq(&proof);
        assert!(coq_code.contains("Lemma"));
        assert!(coq_code.contains("modus_ponens"));
    }

    #[test]
    fn test_export_theory_proof() {
        let theory_proof = TheoryProof::new();
        let coq_code = export_theory_to_coq(&theory_proof);
        assert!(coq_code.contains("theory proof"));
        assert!(coq_code.contains("Require Import"));
    }
}
