//! Proof certificates for theory combination.

use crate::proof::ProofNodeId;
use oxiz_core::TermId;

/// Theory identifier used in theory-combination certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TheoryId(pub u32);

/// One Nelson-Oppen combination step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CombinationStep {
    /// Theory that propagated these equalities.
    pub theory: TheoryId,
    /// Interface equalities propagated at this step.
    pub propagated_equalities: Vec<(TermId, TermId)>,
    /// Proof nodes justifying the propagation.
    pub justification: Vec<ProofNodeId>,
}

/// Structured certificate for a Nelson-Oppen theory-combination trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NelsonOppenCertificate {
    /// Ordered combination steps.
    pub steps: Vec<CombinationStep>,
    /// Theory that concludes the combined derivation.
    pub concluding_theory: TheoryId,
    /// Final contradiction node, when available.
    pub contradiction: ProofNodeId,
}

impl NelsonOppenCertificate {
    /// Create an empty certificate.
    #[must_use]
    pub fn new(concluding_theory: TheoryId, contradiction: ProofNodeId) -> Self {
        Self {
            steps: Vec::new(),
            concluding_theory,
            contradiction,
        }
    }

    /// Append a combination step.
    pub fn add_step(&mut self, step: CombinationStep) {
        self.steps.push(step);
    }

    /// Verify basic certificate shape.
    #[must_use]
    pub fn verify(&self) -> bool {
        let Some(last) = self.steps.last() else {
            return false;
        };

        last.theory == self.concluding_theory
            && (!last.propagated_equalities.is_empty() || !last.justification.is_empty())
    }
}
