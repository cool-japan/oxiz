//! Proof builder for ergonomic proof construction.
//!
//! This module provides a fluent API for building proofs step by step.

use crate::proof::{Proof, ProofNodeId};
use crate::theory::{ProofTerm, TheoryProof, TheoryRule, TheoryStepId};
use std::collections::HashMap;

/// Builder for constructing proofs with a fluent API.
#[allow(dead_code)]
#[derive(Debug)]
pub struct ProofBuilder {
    /// The proof being constructed.
    proof: Proof,
    /// Named nodes for easier reference.
    named_nodes: HashMap<String, ProofNodeId>,
}

#[allow(dead_code)]
impl ProofBuilder {
    /// Create a new proof builder starting with an axiom.
    #[must_use]
    pub fn new(conclusion: impl Into<String>) -> Self {
        let mut proof = Proof::new();
        let root = proof.add_axiom(conclusion);
        let mut named_nodes = HashMap::new();
        named_nodes.insert("root".to_string(), root);
        Self { proof, named_nodes }
    }

    /// Create a new proof builder starting with an inference.
    #[must_use]
    pub fn new_inference(
        rule: impl Into<String>,
        premises: Vec<ProofNodeId>,
        conclusion: impl Into<String>,
    ) -> Self {
        let mut proof = Proof::new();
        let root = proof.add_inference(rule, premises, conclusion);
        let mut named_nodes = HashMap::new();
        named_nodes.insert("root".to_string(), root);
        Self { proof, named_nodes }
    }

    /// Add an axiom node and optionally name it.
    pub fn axiom(&mut self, conclusion: impl Into<String>, name: Option<String>) -> ProofNodeId {
        let id = self.proof.add_axiom(conclusion);
        if let Some(n) = name {
            self.named_nodes.insert(n, id);
        }
        id
    }

    /// Add an inference node and optionally name it.
    pub fn inference(
        &mut self,
        rule: impl Into<String>,
        premises: Vec<ProofNodeId>,
        conclusion: impl Into<String>,
        name: Option<String>,
    ) -> ProofNodeId {
        let id = self.proof.add_inference(rule, premises, conclusion);
        if let Some(n) = name {
            self.named_nodes.insert(n, id);
        }
        id
    }

    /// Get a named node.
    #[must_use]
    pub fn get_named(&self, name: &str) -> Option<ProofNodeId> {
        self.named_nodes.get(name).copied()
    }

    /// Set the root of the proof.
    pub fn set_root(&mut self, root: ProofNodeId) {
        self.named_nodes.insert("root".to_string(), root);
    }

    /// Build the final proof.
    #[must_use]
    pub fn build(self) -> Proof {
        self.proof
    }

    /// Get a reference to the proof being built.
    #[must_use]
    pub fn proof(&self) -> &Proof {
        &self.proof
    }

    /// Get a mutable reference to the proof being built.
    pub fn proof_mut(&mut self) -> &mut Proof {
        &mut self.proof
    }
}

/// Builder for constructing theory proofs with a fluent API.
#[allow(dead_code)]
#[derive(Debug)]
pub struct TheoryProofBuilder {
    /// The theory proof being constructed.
    proof: TheoryProof,
    /// Named steps for easier reference.
    named_steps: HashMap<String, TheoryStepId>,
}

#[allow(dead_code)]
impl TheoryProofBuilder {
    /// Create a new theory proof builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: TheoryProof::new(),
            named_steps: HashMap::new(),
        }
    }

    /// Add an axiom and optionally name it.
    pub fn axiom(
        &mut self,
        rule: TheoryRule,
        conclusion: impl Into<ProofTerm>,
        name: Option<String>,
    ) -> TheoryStepId {
        let id = self.proof.add_axiom(rule, conclusion);
        if let Some(n) = name {
            self.named_steps.insert(n, id);
        }
        id
    }

    /// Add a step and optionally name it.
    pub fn step(
        &mut self,
        rule: TheoryRule,
        premises: Vec<TheoryStepId>,
        conclusion: impl Into<ProofTerm>,
        name: Option<String>,
    ) -> TheoryStepId {
        let id = self.proof.add_step(rule, premises, conclusion);
        if let Some(n) = name {
            self.named_steps.insert(n, id);
        }
        id
    }

    /// Add a reflexivity step.
    pub fn refl(&mut self, term: impl Into<ProofTerm>, name: Option<String>) -> TheoryStepId {
        let id = self.proof.refl(term);
        if let Some(n) = name {
            self.named_steps.insert(n, id);
        }
        id
    }

    /// Add a transitivity step.
    pub fn trans(
        &mut self,
        p1: TheoryStepId,
        p2: TheoryStepId,
        t1: impl Into<ProofTerm>,
        t3: impl Into<ProofTerm>,
        name: Option<String>,
    ) -> TheoryStepId {
        let id = self.proof.trans(p1, p2, t1, t3);
        if let Some(n) = name {
            self.named_steps.insert(n, id);
        }
        id
    }

    /// Get a named step.
    #[must_use]
    pub fn get_named(&self, name: &str) -> Option<TheoryStepId> {
        self.named_steps.get(name).copied()
    }

    /// Build the final theory proof.
    #[must_use]
    pub fn build(self) -> TheoryProof {
        self.proof
    }

    /// Get a reference to the proof being built.
    #[must_use]
    pub fn proof(&self) -> &TheoryProof {
        &self.proof
    }

    /// Get a mutable reference to the proof being built.
    pub fn proof_mut(&mut self) -> &mut TheoryProof {
        &mut self.proof
    }
}

impl Default for TheoryProofBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_builder_basic() {
        let mut builder = ProofBuilder::new("p");
        let root = builder.get_named("root").unwrap();
        let p2 = builder.axiom("q", Some("q_axiom".to_string()));
        let p3 = builder.inference("and", vec![root, p2], "(and p q)", None);

        builder.set_root(p3);

        let proof = builder.build();
        assert_eq!(proof.len(), 3);
    }

    #[test]
    fn test_proof_builder_named() {
        let mut builder = ProofBuilder::new("p");
        builder.axiom("q", Some("q".to_string()));
        builder.axiom("r", Some("r".to_string()));

        let q_id = builder.get_named("q").unwrap();
        let r_id = builder.get_named("r").unwrap();

        assert!(builder.proof.get_node(q_id).is_some());
        assert!(builder.proof.get_node(r_id).is_some());
    }

    #[test]
    fn test_theory_proof_builder() {
        let mut builder = TheoryProofBuilder::new();

        let s1 = builder.axiom(
            TheoryRule::Custom("assert".into()),
            "(= x y)",
            Some("xy_eq".to_string()),
        );
        let s2 = builder.axiom(
            TheoryRule::Custom("assert".into()),
            "(= y z)",
            Some("yz_eq".to_string()),
        );

        builder.trans(s1, s2, "x", "z", Some("xz_eq".to_string()));

        let proof = builder.build();
        assert_eq!(proof.len(), 3);
    }

    #[test]
    fn test_theory_proof_builder_refl() {
        let mut builder = TheoryProofBuilder::new();
        builder.refl("x", Some("x_refl".to_string()));

        assert!(builder.get_named("x_refl").is_some());

        let proof = builder.build();
        assert_eq!(proof.len(), 1);
    }
}
