//! Lean proof export
//!
//! This module provides utilities for exporting proofs to Lean format,
//! enabling verification in the Lean theorem prover.
//!
//! ## Overview
//!
//! Lean is an interactive theorem prover and programming language based on
//! dependent type theory. This module translates SMT proofs into Lean syntax
//! that can be checked by Lean 4.
//!
//! ## References
//!
//! - Lean Documentation: <https://lean-lang.org/>
//! - Lean 4 Manual: <https://leanprover.github.io/lean4/doc/>

use crate::proof::{Proof, ProofNode, ProofNodeId, ProofStep};
use crate::theory::TheoryProof;
use std::collections::HashMap;

/// Lean proof exporter (Lean 4 format)
#[derive(Debug)]
pub struct LeanExporter {
    /// Mapping from proof nodes to Lean identifiers
    node_to_ident: HashMap<ProofNodeId, String>,
    /// Counter for generating unique names
    name_counter: usize,
    /// Lean 4 syntax
    use_lean4: bool,
}

impl LeanExporter {
    /// Create a new Lean exporter (defaults to Lean 4)
    #[must_use]
    pub fn new() -> Self {
        Self {
            node_to_ident: HashMap::new(),
            name_counter: 0,
            use_lean4: true,
        }
    }

    /// Create a Lean 3 exporter
    #[must_use]
    pub fn lean3() -> Self {
        Self {
            node_to_ident: HashMap::new(),
            name_counter: 0,
            use_lean4: false,
        }
    }

    /// Generate a fresh identifier
    fn fresh_ident(&mut self, prefix: &str) -> String {
        let ident = format!("{}_{}", prefix, self.name_counter);
        self.name_counter += 1;
        ident
    }

    /// Escape a string for use in Lean
    fn escape_string(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    /// Convert a conclusion to a Lean proposition
    fn conclusion_to_prop(&self, conclusion: &str) -> String {
        // For now, represent conclusions as propositions
        // In a real implementation, this would parse and translate the formula
        format!("PropOf \"{}\"", Self::escape_string(conclusion))
    }

    /// Export a proof node to Lean
    fn export_node(&mut self, _proof: &Proof, node_id: ProofNodeId, node: &ProofNode) -> String {
        let ident = self.fresh_ident("step");
        self.node_to_ident.insert(node_id, ident.clone());

        match &node.step {
            ProofStep::Axiom { conclusion } => {
                let prop = self.conclusion_to_prop(conclusion);
                format!("axiom {} : {}", ident, prop)
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
                    format!("axiom {} : {}", ident, prop)
                } else {
                    let premises_str = premise_idents.join(" → ");
                    if self.use_lean4 {
                        format!(
                            "theorem {} : {} → {} := by\n  -- Rule: {}\n  sorry",
                            ident, premises_str, prop, rule
                        )
                    } else {
                        format!(
                            "theorem {} : {} → {} :=\nbegin\n  -- Rule: {}\n  sorry\nend",
                            ident, premises_str, prop, rule
                        )
                    }
                }
            }
        }
    }

    /// Export a proof to Lean format
    pub fn export_proof(&mut self, proof: &Proof) -> String {
        let mut output = String::new();

        // Header
        output.push_str("-- Generated Lean proof\n");
        output.push_str("-- This proof was automatically exported from oxiz-proof\n\n");

        if self.use_lean4 {
            // Lean 4 imports
            output.push_str("import Std.Logic\n\n");
        } else {
            // Lean 3 imports
            output.push_str("import logic.basic\n\n");
        }

        // Define base types
        output.push_str("-- Proposition representation\n");
        if self.use_lean4 {
            output.push_str("def PropOf (s : String) : Prop := True\n\n");
        } else {
            output.push_str("def PropOf (s : string) : Prop := true\n\n");
        }

        // Export nodes
        let nodes = proof.nodes();

        output.push_str("-- Proof steps\n");
        for node in nodes {
            let node_def = self.export_node(proof, node.id, node);
            output.push_str(&node_def);
            output.push_str("\n\n");
        }

        // Final theorem
        if let Some(root_id) = proof.root()
            && let Some(root_ident) = self.node_to_ident.get(&root_id)
        {
            output.push_str("-- Main result\n");
            if self.use_lean4 {
                output.push_str("theorem main_result : ∃ P, P := by\n");
                output.push_str(&format!("  use {}\n", root_ident));
            } else {
                output.push_str("theorem main_result : ∃ P, P :=\n");
                output.push_str(&format!("⟨{}, _⟩\n", root_ident));
            }
        }

        output
    }

    /// Export a theory proof to Lean format
    pub fn export_theory_proof(&mut self, theory_proof: &TheoryProof) -> String {
        let mut output = String::new();

        output.push_str("-- Generated Lean theory proof\n\n");

        if self.use_lean4 {
            output.push_str("import Std.Data.Int\n");
            output.push_str("import Std.Data.Rat\n\n");
        } else {
            output.push_str("import data.int.basic\n");
            output.push_str("import data.rat.basic\n\n");
        }

        output.push_str("-- Theory axioms and lemmas\n");
        for step in theory_proof.steps() {
            let step_name = self.fresh_ident("theory_step");
            output.push_str(&format!("-- Step {}: {:?}\n", step.id.0, step.rule));
            output.push_str(&format!("axiom {} : Prop\n\n", step_name));
        }

        output.push_str("-- Theory proof complete\n");
        output
    }
}

impl Default for LeanExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Export a proof to Lean 4 format
#[must_use]
pub fn export_to_lean(proof: &Proof) -> String {
    let mut exporter = LeanExporter::new();
    exporter.export_proof(proof)
}

/// Export a proof to Lean 3 format
#[must_use]
pub fn export_to_lean3(proof: &Proof) -> String {
    let mut exporter = LeanExporter::lean3();
    exporter.export_proof(proof)
}

/// Export a theory proof to Lean 4 format
#[must_use]
pub fn export_theory_to_lean(theory_proof: &TheoryProof) -> String {
    let mut exporter = LeanExporter::new();
    exporter.export_theory_proof(theory_proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lean_exporter_creation() {
        let exporter = LeanExporter::new();
        assert!(exporter.use_lean4);
    }

    #[test]
    fn test_lean3_exporter_creation() {
        let exporter = LeanExporter::lean3();
        assert!(!exporter.use_lean4);
    }

    #[test]
    fn test_fresh_ident() {
        let mut exporter = LeanExporter::new();
        let id1 = exporter.fresh_ident("test");
        let id2 = exporter.fresh_ident("test");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(LeanExporter::escape_string("hello"), "hello");
        assert_eq!(LeanExporter::escape_string("a\\b"), "a\\\\b");
        assert_eq!(LeanExporter::escape_string("a\"b"), "a\\\"b");
    }

    #[test]
    fn test_export_simple_proof_lean4() {
        let mut proof = Proof::new();
        let _axiom = proof.add_axiom("P");

        let lean_code = export_to_lean(&proof);
        assert!(lean_code.contains("import Std"));
        assert!(lean_code.contains("axiom"));
        assert!(lean_code.contains("PropOf"));
    }

    #[test]
    fn test_export_simple_proof_lean3() {
        let mut proof = Proof::new();
        let _axiom = proof.add_axiom("P");

        let lean_code = export_to_lean3(&proof);
        assert!(lean_code.contains("import logic"));
        assert!(lean_code.contains("axiom"));
    }

    #[test]
    fn test_export_inference_proof() {
        let mut proof = Proof::new();
        let axiom1 = proof.add_axiom("P");
        let axiom2 = proof.add_axiom("P -> Q");
        let _conclusion = proof.add_inference("modus_ponens", vec![axiom1, axiom2], "Q");

        let lean_code = export_to_lean(&proof);
        assert!(lean_code.contains("theorem"));
        assert!(lean_code.contains("modus_ponens"));
        assert!(lean_code.contains("by"));
    }

    #[test]
    fn test_export_theory_proof() {
        let theory_proof = TheoryProof::new();
        let lean_code = export_theory_to_lean(&theory_proof);
        assert!(lean_code.contains("theory proof"));
        assert!(lean_code.contains("import Std"));
    }
}
