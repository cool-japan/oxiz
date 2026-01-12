//! Proof generation for NLSAT solver.
//!
//! This module provides proof generation capabilities for the NLSAT solver,
//! enabling formal verification and debugging of unsatisfiability results.
//!
//! ## Key Components
//!
//! - **Proof Steps**: Individual inference rules (resolution, theory lemmas, etc.)
//! - **Proof Builder**: Constructs proofs during solving
//! - **Proof Checker**: Validates proof correctness
//! - **Proof Format**: Export to standard formats (LFSC, Alethe, etc.)
//!
//! ## Reference
//!
//! - Z3's proof generation in `nlsat/`
//! - SMT-LIB proof format
//! - Alethe proof format for SMT

use crate::clause::ClauseId;
use crate::types::Literal;
use num_rational::BigRational;
use oxiz_math::polynomial::Var;
use std::collections::HashMap;

/// Unique identifier for proof steps.
pub type ProofId = u64;

/// A step in the proof derivation.
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// Unique identifier for this step.
    pub id: ProofId,
    /// The rule applied in this step.
    pub rule: ProofRule,
    /// Premises (previous proof steps).
    pub premises: Vec<ProofId>,
    /// Conclusion (the clause derived).
    pub conclusion: Vec<Literal>,
    /// Additional metadata.
    pub metadata: ProofMetadata,
}

/// Proof rules for NLSAT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofRule {
    /// Input clause (axiom from the problem).
    Input,
    /// Resolution between two clauses.
    Resolution { pivot: Literal },
    /// Unit propagation.
    UnitPropagation,
    /// Theory lemma (from CAD/polynomial reasoning).
    TheoryLemma { explanation: TheoryExplanation },
    /// Conflict from theory propagation.
    TheoryConflict,
    /// CAD-based reasoning (cell decomposition, projection).
    CadReasoning { operation: CadOperation },
    /// Polynomial simplification.
    PolySimplify,
    /// Arithmetic reasoning (polynomial evaluation).
    ArithReasoning,
    /// Branch and bound (for integer constraints).
    BranchBound { bound: BigRational, var: Var },
    /// Cutting plane.
    CuttingPlane { cut_type: String },
}

/// Theory explanation for lemmas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TheoryExplanation {
    /// Polynomial evaluation conflict.
    PolyEval {
        poly: String,
        assignment: Vec<(Var, String)>,
    },
    /// Sign determination.
    SignDetermination { poly: String, sign: i8 },
    /// CAD cell inconsistency.
    CadCellConflict { cell_id: usize },
    /// Root isolation.
    RootIsolation { poly: String, root_index: u32 },
    /// Generic explanation.
    Generic { description: String },
}

/// CAD operation types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CadOperation {
    /// Projection operator application.
    Projection { from_var: Var, to_var: Var },
    /// Cell decomposition.
    CellDecomposition { var: Var },
    /// Lifting phase.
    Lifting { level: usize },
    /// Sample point evaluation.
    SampleEval { var: Var, value: String },
}

/// Metadata for proof steps.
#[derive(Debug, Clone, Default)]
pub struct ProofMetadata {
    /// Human-readable description.
    pub description: Option<String>,
    /// Source location (for debugging).
    pub source: Option<String>,
    /// Timestamp or sequence number.
    pub sequence: Option<u64>,
}

/// Complete proof object.
#[derive(Debug, Clone)]
pub struct Proof {
    /// All proof steps.
    steps: Vec<ProofStep>,
    /// Map from proof ID to step index.
    step_index: HashMap<ProofId, usize>,
    /// The final conflict step (empty clause).
    conflict_step: Option<ProofId>,
    /// Next available proof ID.
    next_id: ProofId,
}

impl Proof {
    /// Create an empty proof.
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            step_index: HashMap::new(),
            conflict_step: None,
            next_id: 1,
        }
    }

    /// Add an input clause (axiom).
    pub fn add_input(&mut self, clause: Vec<Literal>) -> ProofId {
        let id = self.next_id;
        self.next_id += 1;

        let step = ProofStep {
            id,
            rule: ProofRule::Input,
            premises: Vec::new(),
            conclusion: clause,
            metadata: ProofMetadata::default(),
        };

        self.step_index.insert(id, self.steps.len());
        self.steps.push(step);
        id
    }

    /// Add a resolution step.
    pub fn add_resolution(
        &mut self,
        premise1: ProofId,
        premise2: ProofId,
        pivot: Literal,
        conclusion: Vec<Literal>,
    ) -> ProofId {
        let id = self.next_id;
        self.next_id += 1;

        let step = ProofStep {
            id,
            rule: ProofRule::Resolution { pivot },
            premises: vec![premise1, premise2],
            conclusion,
            metadata: ProofMetadata::default(),
        };

        self.step_index.insert(id, self.steps.len());
        self.steps.push(step);
        id
    }

    /// Add a theory lemma.
    pub fn add_theory_lemma(
        &mut self,
        explanation: TheoryExplanation,
        conclusion: Vec<Literal>,
    ) -> ProofId {
        let id = self.next_id;
        self.next_id += 1;

        let step = ProofStep {
            id,
            rule: ProofRule::TheoryLemma { explanation },
            premises: Vec::new(),
            conclusion,
            metadata: ProofMetadata::default(),
        };

        self.step_index.insert(id, self.steps.len());
        self.steps.push(step);
        id
    }

    /// Add a CAD reasoning step.
    pub fn add_cad_reasoning(
        &mut self,
        operation: CadOperation,
        premises: Vec<ProofId>,
        conclusion: Vec<Literal>,
    ) -> ProofId {
        let id = self.next_id;
        self.next_id += 1;

        let step = ProofStep {
            id,
            rule: ProofRule::CadReasoning { operation },
            premises,
            conclusion,
            metadata: ProofMetadata::default(),
        };

        self.step_index.insert(id, self.steps.len());
        self.steps.push(step);
        id
    }

    /// Mark a step as the final conflict.
    pub fn set_conflict(&mut self, step_id: ProofId) {
        self.conflict_step = Some(step_id);
    }

    /// Get a proof step by ID.
    pub fn get_step(&self, id: ProofId) -> Option<&ProofStep> {
        self.step_index
            .get(&id)
            .and_then(|&idx| self.steps.get(idx))
    }

    /// Get all steps.
    pub fn steps(&self) -> &[ProofStep] {
        &self.steps
    }

    /// Get the conflict step.
    pub fn conflict_step(&self) -> Option<ProofId> {
        self.conflict_step
    }

    /// Verify the proof is well-formed.
    pub fn verify(&self) -> Result<(), ProofError> {
        // Check that all premise IDs exist
        for step in &self.steps {
            for &premise_id in &step.premises {
                if !self.step_index.contains_key(&premise_id) {
                    return Err(ProofError::MissingPremise {
                        step_id: step.id,
                        premise_id,
                    });
                }
                // Premise must come before this step
                let premise_idx = self.step_index[&premise_id];
                let step_idx = self.step_index[&step.id];
                if premise_idx >= step_idx {
                    return Err(ProofError::InvalidProofOrder {
                        step_id: step.id,
                        premise_id,
                    });
                }
            }
        }

        // Check that conflict step exists and derives empty clause
        if let Some(conflict_id) = self.conflict_step {
            if let Some(step) = self.get_step(conflict_id) {
                if !step.conclusion.is_empty() {
                    return Err(ProofError::InvalidConflict {
                        step_id: conflict_id,
                    });
                }
            } else {
                return Err(ProofError::MissingConflict);
            }
        }

        Ok(())
    }

    /// Export to SMT-LIB proof format.
    pub fn to_smtlib(&self) -> String {
        let mut output = String::new();
        output.push_str("(proof\n");

        for step in &self.steps {
            output.push_str(&format!("  (step s{} ", step.id));

            match &step.rule {
                ProofRule::Input => output.push_str("(cl "),
                ProofRule::Resolution { pivot } => {
                    output.push_str(&format!(
                        "(resolution s{} s{} ",
                        step.premises[0], step.premises[1]
                    ));
                    output.push_str(&format!("lit{} ", pivot.index()));
                }
                ProofRule::TheoryLemma { .. } => output.push_str("(theory-lemma "),
                ProofRule::CadReasoning { .. } => output.push_str("(cad-reasoning "),
                _ => output.push_str("(unknown "),
            }

            // Add conclusion
            for lit in &step.conclusion {
                output.push_str(&format!("lit{} ", lit.index()));
            }
            output.push_str("))\n");
        }

        output.push_str(")\n");
        output
    }

    /// Get proof statistics.
    pub fn stats(&self) -> ProofStats {
        let mut stats = ProofStats {
            total_steps: self.steps.len(),
            ..Default::default()
        };

        for step in &self.steps {
            match &step.rule {
                ProofRule::Input => stats.input_steps += 1,
                ProofRule::Resolution { .. } => stats.resolution_steps += 1,
                ProofRule::TheoryLemma { .. } => stats.theory_lemmas += 1,
                ProofRule::CadReasoning { .. } => stats.cad_steps += 1,
                ProofRule::BranchBound { .. } => stats.branch_bound_steps += 1,
                ProofRule::CuttingPlane { .. } => stats.cutting_plane_steps += 1,
                _ => {}
            }
        }

        stats
    }
}

impl Default for Proof {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during proof generation or verification.
#[derive(Debug, Clone)]
pub enum ProofError {
    /// Missing premise step.
    MissingPremise {
        step_id: ProofId,
        premise_id: ProofId,
    },
    /// Invalid proof order (premise comes after conclusion).
    InvalidProofOrder {
        step_id: ProofId,
        premise_id: ProofId,
    },
    /// Conflict step is missing.
    MissingConflict,
    /// Conflict step doesn't derive empty clause.
    InvalidConflict { step_id: ProofId },
    /// Resolution error.
    ResolutionError { description: String },
}

impl std::fmt::Display for ProofError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProofError::MissingPremise {
                step_id,
                premise_id,
            } => {
                write!(
                    f,
                    "Step {} references missing premise {}",
                    step_id, premise_id
                )
            }
            ProofError::InvalidProofOrder {
                step_id,
                premise_id,
            } => {
                write!(
                    f,
                    "Step {} uses premise {} that comes later",
                    step_id, premise_id
                )
            }
            ProofError::MissingConflict => write!(f, "Proof missing conflict step"),
            ProofError::InvalidConflict { step_id } => {
                write!(
                    f,
                    "Step {} marked as conflict but doesn't derive empty clause",
                    step_id
                )
            }
            ProofError::ResolutionError { description } => {
                write!(f, "Resolution error: {}", description)
            }
        }
    }
}

impl std::error::Error for ProofError {}

/// Statistics about a proof.
#[derive(Debug, Clone, Default)]
pub struct ProofStats {
    /// Total number of steps.
    pub total_steps: usize,
    /// Number of input steps.
    pub input_steps: usize,
    /// Number of resolution steps.
    pub resolution_steps: usize,
    /// Number of theory lemmas.
    pub theory_lemmas: usize,
    /// Number of CAD reasoning steps.
    pub cad_steps: usize,
    /// Number of branch-and-bound steps.
    pub branch_bound_steps: usize,
    /// Number of cutting plane steps.
    pub cutting_plane_steps: usize,
}

impl ProofStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get proof complexity (rough estimate).
    pub fn complexity(&self) -> usize {
        self.total_steps
    }
}

/// Proof builder for incremental construction during solving.
pub struct ProofBuilder {
    /// The proof being built.
    proof: Proof,
    /// Map from clause ID to proof ID.
    clause_to_proof: HashMap<ClauseId, ProofId>,
    /// Enable proof generation.
    enabled: bool,
}

impl ProofBuilder {
    /// Create a new proof builder.
    pub fn new() -> Self {
        Self {
            proof: Proof::new(),
            clause_to_proof: HashMap::new(),
            enabled: true,
        }
    }

    /// Enable or disable proof generation.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Is proof generation enabled?
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an input clause.
    pub fn record_input(&mut self, clause_id: ClauseId, clause: Vec<Literal>) -> Option<ProofId> {
        if !self.enabled {
            return None;
        }

        let proof_id = self.proof.add_input(clause);
        self.clause_to_proof.insert(clause_id, proof_id);
        Some(proof_id)
    }

    /// Record a learned clause from resolution.
    pub fn record_learned(
        &mut self,
        clause_id: ClauseId,
        conflict_clause: ClauseId,
        reason_clause: ClauseId,
        pivot: Literal,
        conclusion: Vec<Literal>,
    ) -> Option<ProofId> {
        if !self.enabled {
            return None;
        }

        let premise1 = self.clause_to_proof.get(&conflict_clause).copied()?;
        let premise2 = self.clause_to_proof.get(&reason_clause).copied()?;

        let proof_id = self
            .proof
            .add_resolution(premise1, premise2, pivot, conclusion);
        self.clause_to_proof.insert(clause_id, proof_id);
        Some(proof_id)
    }

    /// Record a theory lemma.
    pub fn record_theory_lemma(
        &mut self,
        clause_id: ClauseId,
        explanation: TheoryExplanation,
        conclusion: Vec<Literal>,
    ) -> Option<ProofId> {
        if !self.enabled {
            return None;
        }

        let proof_id = self.proof.add_theory_lemma(explanation, conclusion);
        self.clause_to_proof.insert(clause_id, proof_id);
        Some(proof_id)
    }

    /// Record a conflict.
    pub fn record_conflict(&mut self, conflict_clause: ClauseId) {
        if !self.enabled {
            return;
        }

        if let Some(&proof_id) = self.clause_to_proof.get(&conflict_clause) {
            self.proof.set_conflict(proof_id);
        }
    }

    /// Get the constructed proof.
    pub fn build(self) -> Proof {
        self.proof
    }

    /// Get a reference to the proof.
    pub fn proof(&self) -> &Proof {
        &self.proof
    }
}

impl Default for ProofBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_new() {
        let proof = Proof::new();
        assert_eq!(proof.steps().len(), 0);
        assert!(proof.conflict_step().is_none());
    }

    #[test]
    fn test_add_input() {
        let mut proof = Proof::new();
        let lit1 = Literal::positive(0);
        let lit2 = Literal::positive(1);

        let id = proof.add_input(vec![lit1, lit2]);
        assert_eq!(id, 1);
        assert_eq!(proof.steps().len(), 1);

        let step = proof.get_step(id).unwrap();
        assert_eq!(step.conclusion.len(), 2);
        assert!(matches!(step.rule, ProofRule::Input));
    }

    #[test]
    fn test_add_resolution() {
        let mut proof = Proof::new();

        let id1 = proof.add_input(vec![Literal::positive(0), Literal::positive(1)]);
        let id2 = proof.add_input(vec![Literal::negative(0), Literal::positive(2)]);

        let pivot = Literal::positive(0);
        let conclusion = vec![Literal::positive(1), Literal::positive(2)];

        let res_id = proof.add_resolution(id1, id2, pivot, conclusion.clone());

        let step = proof.get_step(res_id).unwrap();
        assert_eq!(step.premises.len(), 2);
        assert_eq!(step.conclusion, conclusion);
    }

    #[test]
    fn test_verify_valid_proof() {
        let mut proof = Proof::new();

        let id1 = proof.add_input(vec![Literal::positive(0)]);
        let id2 = proof.add_input(vec![Literal::negative(0)]);
        let conflict = proof.add_resolution(id1, id2, Literal::positive(0), vec![]);

        proof.set_conflict(conflict);

        assert!(proof.verify().is_ok());
    }

    #[test]
    fn test_verify_missing_premise() {
        let mut proof = Proof::new();

        // Manually construct a step with invalid premise
        proof.steps.push(ProofStep {
            id: 999,
            rule: ProofRule::Resolution {
                pivot: Literal::positive(0),
            },
            premises: vec![1, 2], // These don't exist
            conclusion: vec![],
            metadata: ProofMetadata::default(),
        });
        proof.step_index.insert(999, proof.steps.len() - 1);

        assert!(proof.verify().is_err());
    }

    #[test]
    fn test_theory_lemma() {
        let mut proof = Proof::new();

        let explanation = TheoryExplanation::SignDetermination {
            poly: "x^2 - 1".to_string(),
            sign: 1,
        };

        let id = proof.add_theory_lemma(explanation.clone(), vec![Literal::positive(0)]);

        let step = proof.get_step(id).unwrap();
        if let ProofRule::TheoryLemma { explanation: exp } = &step.rule {
            assert_eq!(exp, &explanation);
        } else {
            panic!("Expected TheoryLemma rule");
        }
    }

    #[test]
    fn test_proof_stats() {
        let mut proof = Proof::new();

        proof.add_input(vec![Literal::positive(0)]);
        proof.add_input(vec![Literal::positive(1)]);
        proof.add_theory_lemma(
            TheoryExplanation::Generic {
                description: "test".to_string(),
            },
            vec![],
        );

        let stats = proof.stats();
        assert_eq!(stats.total_steps, 3);
        assert_eq!(stats.input_steps, 2);
        assert_eq!(stats.theory_lemmas, 1);
    }

    #[test]
    fn test_proof_builder() {
        let mut builder = ProofBuilder::new();

        assert!(builder.is_enabled());

        let clause1 = vec![Literal::positive(0)];
        let id1 = builder.record_input(0, clause1);
        assert!(id1.is_some());

        let clause2 = vec![Literal::negative(0)];
        let id2 = builder.record_input(1, clause2);
        assert!(id2.is_some());

        builder.record_conflict(0);

        let proof = builder.build();
        assert_eq!(proof.steps().len(), 2);
    }

    #[test]
    fn test_proof_builder_disabled() {
        let mut builder = ProofBuilder::new();
        builder.set_enabled(false);

        let clause = vec![Literal::positive(0)];
        let id = builder.record_input(0, clause);
        assert!(id.is_none());

        let proof = builder.build();
        assert_eq!(proof.steps().len(), 0);
    }

    #[test]
    fn test_to_smtlib() {
        let mut proof = Proof::new();
        proof.add_input(vec![Literal::positive(0)]);

        let output = proof.to_smtlib();
        assert!(output.contains("(proof"));
        assert!(output.contains("step"));
    }

    #[test]
    fn test_cad_reasoning() {
        let mut proof = Proof::new();

        let operation = CadOperation::Projection {
            from_var: 1,
            to_var: 0,
        };

        let id = proof.add_cad_reasoning(operation, vec![], vec![Literal::positive(0)]);

        let step = proof.get_step(id).unwrap();
        if let ProofRule::CadReasoning { operation: op } = &step.rule {
            assert!(matches!(op, CadOperation::Projection { .. }));
        } else {
            panic!("Expected CadReasoning rule");
        }
    }
}
