use oxiz_core::TermId;
use oxiz_proof::{CombinationStep, CombinationTheoryId, NelsonOppenCertificate, ProofNodeId};

#[test]
fn synthetic_no_conflict_certificate_verifies() {
    let mut certificate = NelsonOppenCertificate::new(CombinationTheoryId(1), ProofNodeId(7));
    certificate.add_step(CombinationStep {
        theory: CombinationTheoryId(1),
        propagated_equalities: vec![(TermId::new(1), TermId::new(2))],
        justification: Vec::new(),
    });

    assert!(certificate.verify());
}

#[test]
fn empty_certificate_fails_verification() {
    let certificate = NelsonOppenCertificate::new(CombinationTheoryId(2), ProofNodeId(0));
    assert!(!certificate.verify());
}
