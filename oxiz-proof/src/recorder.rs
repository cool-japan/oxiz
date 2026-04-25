//! Generic proof step recorder.
//!
//! This module provides a lightweight recorder over the core [`Proof`] DAG.
//! The existing incremental recorder remains available in [`crate::incremental`].

use crate::proof::{Proof, ProofNodeId, ProofStep};

#[cfg(feature = "arena")]
use bumpalo::Bump;
#[cfg(feature = "arena")]
use std::ptr::NonNull;

/// Arena-backed proof step identifier.
#[cfg(feature = "arena")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArenaProofStepId(pub u32);

/// A lightweight proof recorder over [`ProofStep`] values.
#[derive(Debug, Default)]
pub struct Recorder {
    proof: Proof,
    #[cfg(feature = "arena")]
    arena: Bump,
    #[cfg(feature = "arena")]
    arena_steps: Vec<NonNull<ProofStep>>,
}

impl Recorder {
    /// Create a new recorder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: Proof::new(),
            #[cfg(feature = "arena")]
            arena: Bump::new(),
            #[cfg(feature = "arena")]
            arena_steps: Vec::new(),
        }
    }

    /// Record a proof step into the owned proof DAG.
    pub fn record_step(&mut self, step: ProofStep) -> ProofNodeId {
        match step {
            ProofStep::Axiom { conclusion } => self.proof.add_axiom(conclusion),
            ProofStep::Inference {
                rule,
                premises,
                conclusion,
                args,
            } => self.proof.add_inference_with_args(
                rule,
                premises.into_vec(),
                args.into_vec(),
                conclusion,
            ),
        }
    }

    /// Return the recorded proof.
    #[must_use]
    pub const fn proof(&self) -> &Proof {
        &self.proof
    }

    /// Return the recorded proof mutably.
    pub fn proof_mut(&mut self) -> &mut Proof {
        &mut self.proof
    }

    /// Consume the recorder and return the proof.
    #[must_use]
    pub fn into_proof(self) -> Proof {
        self.proof
    }

    /// Reset the recorder state.
    pub fn clear(&mut self) {
        self.proof.clear();
        #[cfg(feature = "arena")]
        {
            self.arena.reset();
            self.arena_steps.clear();
        }
    }

    /// Record a step in the bump arena and return its arena-local ID.
    #[cfg(feature = "arena")]
    pub fn record_step_arena(&mut self, step: ProofStep) -> Option<ArenaProofStepId> {
        let index = u32::try_from(self.arena_steps.len()).ok()?;
        let allocated = self.arena.alloc(step);
        self.arena_steps.push(NonNull::from(allocated));
        Some(ArenaProofStepId(index))
    }

    /// Fetch an arena-recorded step by ID.
    #[cfg(feature = "arena")]
    #[must_use]
    pub fn get(&self, id: ArenaProofStepId) -> Option<&ProofStep> {
        self.arena_steps.get(id.0 as usize).map(|ptr| {
            // SAFETY: pointers come from `self.arena.alloc` and stay valid until reset/drop.
            unsafe { ptr.as_ref() }
        })
    }
}

#[cfg(all(test, feature = "arena"))]
mod arena_tests {
    use super::*;
    use crate::validation::FormatValidator;
    use smallvec::SmallVec;

    #[test]
    fn arena_recorded_proof_passes_checker() {
        let mut recorder = Recorder::new();

        let left = recorder.record_step(ProofStep::Axiom {
            conclusion: "p".to_string(),
        });
        let right = recorder.record_step(ProofStep::Axiom {
            conclusion: "q".to_string(),
        });

        let arena_id = recorder.record_step_arena(ProofStep::Inference {
            rule: "and_intro".to_string(),
            premises: SmallVec::from_vec(vec![left, right]),
            conclusion: "(and p q)".to_string(),
            args: SmallVec::new(),
        });
        assert!(arena_id.is_some());

        recorder.record_step(ProofStep::Inference {
            rule: "and_intro".to_string(),
            premises: SmallVec::from_vec(vec![left, right]),
            conclusion: "(and p q)".to_string(),
            args: SmallVec::new(),
        });

        let validator = FormatValidator::new();
        assert!(validator.validate_proof(recorder.proof()).is_ok());
    }

    #[test]
    fn arena_step_returns_valid_id() {
        let mut recorder = Recorder::new();

        let maybe_step_id = recorder.record_step_arena(ProofStep::Axiom {
            conclusion: "p".to_string(),
        });

        let Some(step_id) = maybe_step_id else {
            panic!("arena should allocate a proof step id");
        };

        match recorder.get(step_id) {
            Some(ProofStep::Axiom { conclusion }) => assert_eq!(conclusion, "p"),
            _ => panic!("expected arena-recorded axiom"),
        }
    }
}
