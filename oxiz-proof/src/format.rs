//! Unified proof format conversion API.
//!
//! This module provides a high-level interface for converting proofs to different
//! formats, including Coq, Lean, and Isabelle.

use crate::coq;
use crate::isabelle;
use crate::lean;
use crate::proof::Proof;
use std::fmt;

/// Supported proof formats for export.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofFormat {
    /// Coq proof assistant
    Coq,
    /// Lean 3 theorem prover
    Lean3,
    /// Lean 4 theorem prover
    Lean4,
    /// Isabelle/HOL theorem prover
    Isabelle,
}

impl ProofFormat {
    /// Get file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            ProofFormat::Coq => "v",
            ProofFormat::Lean3 | ProofFormat::Lean4 => "lean",
            ProofFormat::Isabelle => "thy",
        }
    }

    /// Get a human-readable name for this format.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            ProofFormat::Coq => "Coq",
            ProofFormat::Lean3 => "Lean 3",
            ProofFormat::Lean4 => "Lean 4",
            ProofFormat::Isabelle => "Isabelle/HOL",
        }
    }

    /// Convert a proof to this format.
    #[must_use]
    pub fn convert(&self, proof: &Proof) -> String {
        match self {
            ProofFormat::Coq => coq::export_to_coq(proof),
            ProofFormat::Lean3 => lean::export_to_lean3(proof),
            ProofFormat::Lean4 => lean::export_to_lean(proof),
            ProofFormat::Isabelle => isabelle::export_to_isabelle(proof, "Proof"),
        }
    }
}

impl fmt::Display for ProofFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Quick conversion helpers.
pub mod quick {
    use super::*;

    /// Convert proof to Coq format.
    #[must_use]
    pub fn to_coq(proof: &Proof) -> String {
        ProofFormat::Coq.convert(proof)
    }

    /// Convert proof to Lean 3 format.
    #[must_use]
    pub fn to_lean3(proof: &Proof) -> String {
        ProofFormat::Lean3.convert(proof)
    }

    /// Convert proof to Lean 4 format.
    #[must_use]
    pub fn to_lean(proof: &Proof) -> String {
        ProofFormat::Lean4.convert(proof)
    }

    /// Convert proof to Isabelle format.
    #[must_use]
    pub fn to_isabelle(proof: &Proof) -> String {
        ProofFormat::Isabelle.convert(proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_extension() {
        assert_eq!(ProofFormat::Coq.extension(), "v");
        assert_eq!(ProofFormat::Lean4.extension(), "lean");
        assert_eq!(ProofFormat::Isabelle.extension(), "thy");
    }

    #[test]
    fn test_format_name() {
        assert_eq!(ProofFormat::Coq.name(), "Coq");
        assert_eq!(ProofFormat::Lean3.name(), "Lean 3");
        assert_eq!(ProofFormat::Isabelle.name(), "Isabelle/HOL");
    }

    #[test]
    fn test_format_display() {
        assert_eq!(ProofFormat::Coq.to_string(), "Coq");
        assert_eq!(ProofFormat::Lean4.to_string(), "Lean 4");
    }

    #[test]
    fn test_convert_to_coq() {
        let mut proof = Proof::new();
        proof.add_axiom("p");

        let result = ProofFormat::Coq.convert(&proof);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_convert_to_lean() {
        let mut proof = Proof::new();
        proof.add_axiom("p");

        let result = ProofFormat::Lean4.convert(&proof);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_convert_to_isabelle() {
        let mut proof = Proof::new();
        proof.add_axiom("p");

        let result = ProofFormat::Isabelle.convert(&proof);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quick_conversion_coq() {
        let mut proof = Proof::new();
        proof.add_axiom("p");

        let result = quick::to_coq(&proof);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quick_conversion_lean() {
        let mut proof = Proof::new();
        proof.add_axiom("p");

        let result = quick::to_lean(&proof);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_quick_conversion_isabelle() {
        let mut proof = Proof::new();
        proof.add_axiom("p");

        let result = quick::to_isabelle(&proof);
        assert!(!result.is_empty());
    }
}
