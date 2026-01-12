//! Proof generation and validation for Spacer.
//!
//! This module provides infrastructure for generating and validating
//! inductive invariant proofs from PDR/IC3 runs.
//!
//! Reference: Z3's `muz/spacer/spacer_proof_utils.cpp`

use crate::chc::{PredId, RuleId};
use crate::frames::LemmaId;
use oxiz_core::TermId;
use smallvec::SmallVec;
use std::collections::HashMap;

/// A proof step in the inductive invariant proof
#[derive(Debug, Clone)]
pub enum ProofStep {
    /// Initial lemma from frame 0
    InitialLemma {
        /// The predicate
        pred: PredId,
        /// The lemma formula
        lemma: TermId,
        /// Justification (why this holds initially)
        justification: Justification,
    },
    /// Inductive lemma (holds at all levels)
    InductiveLemma {
        /// The predicate
        pred: PredId,
        /// The lemma formula
        lemma: TermId,
        /// Level at which it became inductive
        level: u32,
        /// Dependencies (other lemmas used in proof)
        dependencies: SmallVec<[LemmaId; 4]>,
    },
    /// Lemma propagation (from level k to k+1)
    Propagation {
        /// The lemma that was propagated
        lemma_id: LemmaId,
        /// From level
        from_level: u32,
        /// To level
        to_level: u32,
    },
    /// Fixpoint detection (all lemmas at level k propagate)
    Fixpoint {
        /// The level at which fixpoint was detected
        level: u32,
        /// Lemmas that form the fixpoint
        lemmas: SmallVec<[LemmaId; 8]>,
    },
}

/// Justification for a proof step
#[derive(Debug, Clone)]
pub enum Justification {
    /// Follows from initial condition
    Initial { rule: RuleId },
    /// Follows from transition + induction
    Inductive { rules: SmallVec<[RuleId; 2]> },
    /// Follows from generalization of counterexample
    Generalization { original: TermId },
    /// Follows from subsumption
    Subsumed { by: LemmaId },
}

/// A complete proof of safety
#[derive(Debug, Clone)]
pub struct SafetyProof {
    /// Sequence of proof steps
    steps: Vec<ProofStep>,
    /// Final inductive invariants per predicate
    invariants: HashMap<PredId, SmallVec<[TermId; 8]>>,
}

impl SafetyProof {
    /// Create a new empty proof
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            invariants: HashMap::new(),
        }
    }

    /// Add a proof step
    pub fn add_step(&mut self, step: ProofStep) {
        self.steps.push(step);
    }

    /// Set final invariants
    pub fn set_invariants(&mut self, invariants: HashMap<PredId, SmallVec<[TermId; 8]>>) {
        self.invariants = invariants;
    }

    /// Get all proof steps
    pub fn steps(&self) -> &[ProofStep] {
        &self.steps
    }

    /// Get final invariants
    pub fn invariants(&self) -> &HashMap<PredId, SmallVec<[TermId; 8]>> {
        &self.invariants
    }

    /// Validate the proof structure
    pub fn validate(&self) -> Result<(), ProofError> {
        // Check that the proof is non-empty
        if self.steps.is_empty() {
            return Err(ProofError::EmptyProof);
        }

        // Check that we have a fixpoint
        let has_fixpoint = self
            .steps
            .iter()
            .any(|step| matches!(step, ProofStep::Fixpoint { .. }));

        if !has_fixpoint {
            return Err(ProofError::MissingFixpoint);
        }

        // Check that all invariants are justified
        if self.invariants.is_empty() {
            return Err(ProofError::NoInvariants);
        }

        Ok(())
    }

    /// Get statistics about the proof
    pub fn stats(&self) -> ProofStats {
        let mut stats = ProofStats::default();

        for step in &self.steps {
            match step {
                ProofStep::InitialLemma { .. } => stats.num_initial += 1,
                ProofStep::InductiveLemma { .. } => stats.num_inductive += 1,
                ProofStep::Propagation { .. } => stats.num_propagations += 1,
                ProofStep::Fixpoint { lemmas, .. } => {
                    stats.num_fixpoints += 1;
                    stats.fixpoint_size = lemmas.len();
                }
            }
        }

        stats.num_predicates = self.invariants.len();
        stats.total_lemmas = self.invariants.values().map(|v| v.len()).sum();

        stats
    }
}

impl Default for SafetyProof {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a proof
#[derive(Debug, Clone, Default)]
pub struct ProofStats {
    /// Number of initial lemmas
    pub num_initial: usize,
    /// Number of inductive lemmas
    pub num_inductive: usize,
    /// Number of propagations
    pub num_propagations: usize,
    /// Number of fixpoints
    pub num_fixpoints: usize,
    /// Size of final fixpoint
    pub fixpoint_size: usize,
    /// Number of predicates with invariants
    pub num_predicates: usize,
    /// Total number of lemmas
    pub total_lemmas: usize,
}

/// Errors that can occur in proof validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    /// Proof is empty
    EmptyProof,
    /// No fixpoint found
    MissingFixpoint,
    /// No invariants in proof
    NoInvariants,
    /// Invalid dependency
    InvalidDependency(LemmaId),
    /// Circular dependency
    CircularDependency,
}

/// Proof builder for constructing proofs during solving
#[derive(Debug)]
pub struct ProofBuilder {
    /// Steps collected so far
    steps: Vec<ProofStep>,
    /// Whether proof building is enabled
    enabled: bool,
}

impl ProofBuilder {
    /// Create a new proof builder
    pub fn new(enabled: bool) -> Self {
        Self {
            steps: Vec::new(),
            enabled,
        }
    }

    /// Record an initial lemma
    pub fn record_initial(&mut self, pred: PredId, lemma: TermId, rule: RuleId) {
        if self.enabled {
            self.steps.push(ProofStep::InitialLemma {
                pred,
                lemma,
                justification: Justification::Initial { rule },
            });
        }
    }

    /// Record an inductive lemma
    pub fn record_inductive(
        &mut self,
        pred: PredId,
        lemma: TermId,
        level: u32,
        dependencies: SmallVec<[LemmaId; 4]>,
    ) {
        if self.enabled {
            self.steps.push(ProofStep::InductiveLemma {
                pred,
                lemma,
                level,
                dependencies,
            });
        }
    }

    /// Record a propagation
    pub fn record_propagation(&mut self, lemma_id: LemmaId, from_level: u32, to_level: u32) {
        if self.enabled {
            self.steps.push(ProofStep::Propagation {
                lemma_id,
                from_level,
                to_level,
            });
        }
    }

    /// Record a fixpoint
    pub fn record_fixpoint(&mut self, level: u32, lemmas: SmallVec<[LemmaId; 8]>) {
        if self.enabled {
            self.steps.push(ProofStep::Fixpoint { level, lemmas });
        }
    }

    /// Build the final proof
    pub fn build(self, invariants: HashMap<PredId, SmallVec<[TermId; 8]>>) -> SafetyProof {
        let mut proof = SafetyProof::new();
        proof.steps = self.steps;
        proof.set_invariants(invariants);
        proof
    }

    /// Check if proof building is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current number of steps
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_creation() {
        let proof = SafetyProof::new();
        assert_eq!(proof.steps().len(), 0);
        assert_eq!(proof.invariants().len(), 0);
    }

    #[test]
    fn test_proof_validation_empty() {
        let proof = SafetyProof::new();
        assert_eq!(proof.validate(), Err(ProofError::EmptyProof));
    }

    #[test]
    fn test_proof_with_fixpoint() {
        let mut proof = SafetyProof::new();

        proof.add_step(ProofStep::Fixpoint {
            level: 5,
            lemmas: SmallVec::from_vec(vec![LemmaId::new(0), LemmaId::new(1)]),
        });

        let mut invariants = HashMap::new();
        invariants.insert(PredId::new(0), SmallVec::from_vec(vec![TermId::new(1)]));
        proof.set_invariants(invariants);

        assert!(proof.validate().is_ok());
    }

    #[test]
    fn test_proof_stats() {
        let mut proof = SafetyProof::new();

        proof.add_step(ProofStep::InitialLemma {
            pred: PredId::new(0),
            lemma: TermId::new(1),
            justification: Justification::Initial {
                rule: RuleId::new(0),
            },
        });

        proof.add_step(ProofStep::InductiveLemma {
            pred: PredId::new(0),
            lemma: TermId::new(2),
            level: 1,
            dependencies: SmallVec::new(),
        });

        proof.add_step(ProofStep::Fixpoint {
            level: 2,
            lemmas: SmallVec::from_vec(vec![LemmaId::new(0)]),
        });

        let stats = proof.stats();
        assert_eq!(stats.num_initial, 1);
        assert_eq!(stats.num_inductive, 1);
        assert_eq!(stats.num_fixpoints, 1);
    }

    #[test]
    fn test_proof_builder() {
        let mut builder = ProofBuilder::new(true);
        assert!(builder.is_enabled());
        assert_eq!(builder.num_steps(), 0);

        builder.record_initial(PredId::new(0), TermId::new(1), RuleId::new(0));
        assert_eq!(builder.num_steps(), 1);

        let invariants = HashMap::new();
        let proof = builder.build(invariants);
        assert_eq!(proof.steps().len(), 1);
    }

    #[test]
    fn test_proof_builder_disabled() {
        let mut builder = ProofBuilder::new(false);
        builder.record_initial(PredId::new(0), TermId::new(1), RuleId::new(0));

        assert_eq!(builder.num_steps(), 0); // Nothing recorded when disabled
    }
}
