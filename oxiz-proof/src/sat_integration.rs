//! Integration with oxiz-sat for DRAT proof generation
//!
//! This module provides utilities to integrate DRAT proof generation
//! from the oxiz-proof crate with the oxiz-sat SAT solver.

#[cfg(feature = "sat-integration")]
use crate::drat::{Clause, DratProof, DratProofProducer, Lit};

/// Wrapper that combines oxiz-sat Solver with DRAT proof generation
///
/// This wrapper implements the DratProofProducer trait and can be used
/// to transparently record DRAT proofs during SAT solving.
#[cfg(feature = "sat-integration")]
pub struct ProofRecordingSolver {
    /// The underlying SAT solver
    solver: oxiz_sat::Solver,
    /// DRAT proof being built
    proof: Option<DratProof>,
}

#[cfg(feature = "sat-integration")]
impl ProofRecordingSolver {
    /// Create a new proof-recording solver
    #[must_use]
    pub fn new() -> Self {
        Self {
            solver: oxiz_sat::Solver::new(),
            proof: None,
        }
    }

    /// Create a new proof-recording solver with config
    #[must_use]
    pub fn with_config(config: oxiz_sat::SolverConfig) -> Self {
        Self {
            solver: oxiz_sat::Solver::with_config(config),
            proof: None,
        }
    }

    /// Get a reference to the underlying solver
    #[must_use]
    pub fn solver(&self) -> &oxiz_sat::Solver {
        &self.solver
    }

    /// Get a mutable reference to the underlying solver
    pub fn solver_mut(&mut self) -> &mut oxiz_sat::Solver {
        &mut self.solver
    }

    /// Add a clause to the solver and proof
    pub fn add_clause(&mut self, lits: &[oxiz_sat::Lit]) {
        // Convert oxiz_sat::Lit to our Lit format
        let proof_lits: Vec<Lit> = lits.iter().map(|l| l.to_dimacs()).collect();

        if let Some(proof) = &mut self.proof {
            proof.add_clause(proof_lits);
        }

        self.solver.add_clause(lits.iter().copied());
    }

    /// Notify proof of clause deletion
    pub fn delete_clause(&mut self, lits: &[oxiz_sat::Lit]) {
        let proof_lits: Vec<Lit> = lits.iter().map(|l| l.to_dimacs()).collect();

        if let Some(proof) = &mut self.proof {
            proof.delete_clause(proof_lits);
        }
    }

    /// Solve with proof recording
    pub fn solve(&mut self) -> oxiz_sat::SolverResult {
        self.solver.solve()
    }
}

#[cfg(feature = "sat-integration")]
impl Default for ProofRecordingSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "sat-integration")]
impl DratProofProducer for ProofRecordingSolver {
    fn enable_proof(&mut self) {
        self.proof = Some(DratProof::new());
    }

    fn disable_proof(&mut self) {
        self.proof = None;
    }

    fn get_proof(&self) -> Option<&DratProof> {
        self.proof.as_ref()
    }

    fn take_proof(&mut self) -> Option<DratProof> {
        self.proof.take()
    }
}

/// Convert oxiz-sat literal to DRAT literal format
///
/// This is a utility function for converting between the two formats.
#[cfg(feature = "sat-integration")]
#[must_use]
pub fn sat_lit_to_drat(lit: oxiz_sat::Lit) -> Lit {
    lit.to_dimacs()
}

/// Convert DRAT literal to oxiz-sat literal format
///
/// This is a utility function for converting between the two formats.
#[cfg(feature = "sat-integration")]
#[must_use]
pub fn drat_lit_to_sat(lit: Lit) -> oxiz_sat::Lit {
    oxiz_sat::Lit::from_dimacs(lit)
}

/// Convert a clause from oxiz-sat format to DRAT format
#[cfg(feature = "sat-integration")]
#[must_use]
pub fn sat_clause_to_drat(clause: &[oxiz_sat::Lit]) -> Clause {
    clause.iter().map(|&lit| sat_lit_to_drat(lit)).collect()
}

/// Convert a clause from DRAT format to oxiz-sat format
#[cfg(feature = "sat-integration")]
#[must_use]
pub fn drat_clause_to_sat(clause: &Clause) -> Vec<oxiz_sat::Lit> {
    clause.iter().map(|&lit| drat_lit_to_sat(lit)).collect()
}

#[cfg(all(test, feature = "sat-integration"))]
mod tests {
    use super::*;

    #[test]
    fn test_proof_recording_solver_creation() {
        let solver = ProofRecordingSolver::new();
        assert!(solver.get_proof().is_none());
    }

    #[test]
    fn test_enable_proof() {
        let mut solver = ProofRecordingSolver::new();
        solver.enable_proof();
        assert!(solver.get_proof().is_some());
    }

    #[test]
    fn test_disable_proof() {
        let mut solver = ProofRecordingSolver::new();
        solver.enable_proof();
        solver.disable_proof();
        assert!(solver.get_proof().is_none());
    }

    #[test]
    fn test_take_proof() {
        let mut solver = ProofRecordingSolver::new();
        solver.enable_proof();

        let proof = solver.take_proof();
        assert!(proof.is_some());
        assert!(solver.get_proof().is_none());
    }
}

#[cfg(not(feature = "sat-integration"))]
mod placeholder {
    //! Placeholder implementations when sat-integration feature is disabled
    //!
    //! These stubs allow the module to compile without the feature enabled.
}
