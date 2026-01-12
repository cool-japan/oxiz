//! Proof-Carrying Code (PCC) generation
//!
//! This module provides utilities for generating proof-carrying code,
//! which is code accompanied by a formal proof that it satisfies certain
//! safety or correctness properties.
//!
//! ## Overview
//!
//! Proof-carrying code allows producers of code to provide a proof
//! that the code satisfies specified security or safety properties.
//! Consumers can verify this proof without trusting the producer.
//!
//! ## References
//!
//! - Necula, G.C. (1997). "Proof-Carrying Code"
//! - Appel, A.W. (2001). "Foundational Proof-Carrying Code"

use crate::proof::{Proof, ProofNodeId};
use std::collections::HashMap;
use std::fmt;

/// A safety property that code must satisfy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyProperty {
    /// Memory safety (no buffer overflows, use-after-free, etc.)
    MemorySafety,
    /// Type safety (well-typed operations)
    TypeSafety,
    /// Control flow integrity
    ControlFlowIntegrity,
    /// Resource bounds (e.g., bounded memory/time usage)
    ResourceBounds {
        memory: Option<usize>,
        time: Option<usize>,
    },
    /// Custom property with a name and description
    Custom { name: String, description: String },
}

impl fmt::Display for SafetyProperty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MemorySafety => write!(f, "MemorySafety"),
            Self::TypeSafety => write!(f, "TypeSafety"),
            Self::ControlFlowIntegrity => write!(f, "ControlFlowIntegrity"),
            Self::ResourceBounds { memory, time } => {
                write!(f, "ResourceBounds(memory=")?;
                if let Some(m) = memory {
                    write!(f, "{}", m)?;
                } else {
                    write!(f, "∞")?;
                }
                write!(f, ", time=")?;
                if let Some(t) = time {
                    write!(f, "{}", t)?;
                } else {
                    write!(f, "∞")?;
                }
                write!(f, ")")
            }
            Self::Custom { name, .. } => write!(f, "Custom({})", name),
        }
    }
}

/// A verification condition (VC) that must be proven
#[derive(Debug, Clone)]
pub struct VerificationCondition {
    /// Unique identifier for this VC
    pub id: String,
    /// The property being verified
    pub property: SafetyProperty,
    /// The condition that must be proven (as a formula)
    pub condition: String,
    /// Program point where this VC applies
    pub location: CodeLocation,
}

/// Location in the code where a verification condition applies
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeLocation {
    /// Function or procedure name
    pub function: String,
    /// Basic block or statement label
    pub label: Option<String>,
    /// Line number (if available)
    pub line: Option<usize>,
}

impl fmt::Display for CodeLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.function)?;
        if let Some(label) = &self.label {
            write!(f, "::{}", label)?;
        }
        if let Some(line) = self.line {
            write!(f, " (line {})", line)?;
        }
        Ok(())
    }
}

/// Proof-carrying code certificate
///
/// This combines code with proofs of its safety properties.
#[derive(Debug)]
pub struct ProofCarryingCode {
    /// The code (or reference to it)
    code: String,
    /// Safety properties that are certified
    properties: Vec<SafetyProperty>,
    /// Verification conditions
    vcs: Vec<VerificationCondition>,
    /// Proofs for each verification condition
    vc_proofs: HashMap<String, ProofNodeId>,
    /// The underlying proof structure
    proof: Proof,
}

impl ProofCarryingCode {
    /// Create a new proof-carrying code certificate
    #[must_use]
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            properties: Vec::new(),
            vcs: Vec::new(),
            vc_proofs: HashMap::new(),
            proof: Proof::new(),
        }
    }

    /// Add a safety property to be verified
    pub fn add_property(&mut self, property: SafetyProperty) {
        self.properties.push(property);
    }

    /// Add a verification condition
    pub fn add_vc(&mut self, vc: VerificationCondition) {
        self.vcs.push(vc);
    }

    /// Attach a proof for a verification condition
    pub fn attach_proof(&mut self, vc_id: &str, proof_node: ProofNodeId) {
        self.vc_proofs.insert(vc_id.to_string(), proof_node);
    }

    /// Get the underlying proof structure
    #[must_use]
    pub fn proof(&self) -> &Proof {
        &self.proof
    }

    /// Get a mutable reference to the proof
    pub fn proof_mut(&mut self) -> &mut Proof {
        &mut self.proof
    }

    /// Get the code
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Get the certified properties
    #[must_use]
    pub fn properties(&self) -> &[SafetyProperty] {
        &self.properties
    }

    /// Get the verification conditions
    #[must_use]
    pub fn verification_conditions(&self) -> &[VerificationCondition] {
        &self.vcs
    }

    /// Check if all VCs have proofs
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.vcs
            .iter()
            .all(|vc| self.vc_proofs.contains_key(&vc.id))
    }

    /// Get the number of verified VCs
    #[must_use]
    pub fn verified_count(&self) -> usize {
        self.vc_proofs.len()
    }

    /// Get the total number of VCs
    #[must_use]
    pub fn total_vc_count(&self) -> usize {
        self.vcs.len()
    }

    /// Generate a human-readable certificate
    #[must_use]
    pub fn to_certificate(&self) -> String {
        let mut cert = String::new();
        cert.push_str("=== Proof-Carrying Code Certificate ===\n\n");

        cert.push_str("Properties:\n");
        for prop in &self.properties {
            cert.push_str(&format!("  - {}\n", prop));
        }
        cert.push('\n');

        cert.push_str(&format!(
            "Verification Status: {}/{} VCs verified\n\n",
            self.verified_count(),
            self.total_vc_count()
        ));

        cert.push_str("Verification Conditions:\n");
        for vc in &self.vcs {
            cert.push_str(&format!(
                "  [{}] {} at {}\n",
                vc.id, vc.property, vc.location
            ));
            if self.vc_proofs.contains_key(&vc.id) {
                cert.push_str("    Status: VERIFIED ✓\n");
            } else {
                cert.push_str("    Status: UNVERIFIED ✗\n");
            }
        }

        cert.push_str("\n=== End Certificate ===\n");
        cert
    }
}

/// Builder for creating proof-carrying code certificates
pub struct PccBuilder {
    pcc: ProofCarryingCode,
}

impl PccBuilder {
    /// Create a new PCC builder
    #[must_use]
    pub fn new(code: impl Into<String>) -> Self {
        Self {
            pcc: ProofCarryingCode::new(code),
        }
    }

    /// Add a safety property
    pub fn with_property(mut self, property: SafetyProperty) -> Self {
        self.pcc.add_property(property);
        self
    }

    /// Add a verification condition
    pub fn with_vc(mut self, vc: VerificationCondition) -> Self {
        self.pcc.add_vc(vc);
        self
    }

    /// Build the PCC certificate
    #[must_use]
    pub fn build(self) -> ProofCarryingCode {
        self.pcc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcc_creation() {
        let pcc = ProofCarryingCode::new("fn safe_add(a: i32, b: i32) -> i32 { a + b }");
        assert_eq!(pcc.total_vc_count(), 0);
        assert_eq!(pcc.verified_count(), 0);
        assert!(pcc.is_complete());
    }

    #[test]
    fn test_add_property() {
        let mut pcc = ProofCarryingCode::new("code");
        pcc.add_property(SafetyProperty::MemorySafety);
        assert_eq!(pcc.properties().len(), 1);
    }

    #[test]
    fn test_add_vc() {
        let mut pcc = ProofCarryingCode::new("code");
        let vc = VerificationCondition {
            id: "vc1".to_string(),
            property: SafetyProperty::TypeSafety,
            condition: "typeof(x) = int".to_string(),
            location: CodeLocation {
                function: "main".to_string(),
                label: Some("entry".to_string()),
                line: Some(10),
            },
        };
        pcc.add_vc(vc);
        assert_eq!(pcc.total_vc_count(), 1);
        assert!(!pcc.is_complete());
    }

    #[test]
    fn test_attach_proof() {
        let mut pcc = ProofCarryingCode::new("code");
        let vc = VerificationCondition {
            id: "vc1".to_string(),
            property: SafetyProperty::TypeSafety,
            condition: "typeof(x) = int".to_string(),
            location: CodeLocation {
                function: "main".to_string(),
                label: None,
                line: Some(5),
            },
        };
        pcc.add_vc(vc);

        let proof_node = pcc.proof_mut().add_axiom("typeof(x) = int");
        pcc.attach_proof("vc1", proof_node);

        assert!(pcc.is_complete());
        assert_eq!(pcc.verified_count(), 1);
    }

    #[test]
    fn test_pcc_builder() {
        let pcc = PccBuilder::new("safe code")
            .with_property(SafetyProperty::MemorySafety)
            .with_property(SafetyProperty::TypeSafety)
            .with_vc(VerificationCondition {
                id: "vc1".to_string(),
                property: SafetyProperty::MemorySafety,
                condition: "bounds_check(array, index)".to_string(),
                location: CodeLocation {
                    function: "access".to_string(),
                    label: None,
                    line: Some(15),
                },
            })
            .build();

        assert_eq!(pcc.properties().len(), 2);
        assert_eq!(pcc.total_vc_count(), 1);
    }

    #[test]
    fn test_certificate_generation() {
        let mut pcc = ProofCarryingCode::new("test code");
        pcc.add_property(SafetyProperty::MemorySafety);

        let cert = pcc.to_certificate();
        assert!(cert.contains("Proof-Carrying Code Certificate"));
        assert!(cert.contains("MemorySafety"));
    }

    #[test]
    fn test_resource_bounds_display() {
        let prop = SafetyProperty::ResourceBounds {
            memory: Some(1024),
            time: Some(100),
        };
        let s = format!("{}", prop);
        assert!(s.contains("1024"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_code_location_display() {
        let loc = CodeLocation {
            function: "process".to_string(),
            label: Some("loop_head".to_string()),
            line: Some(42),
        };
        let s = format!("{}", loc);
        assert!(s.contains("process"));
        assert!(s.contains("loop_head"));
        assert!(s.contains("42"));
    }
}
