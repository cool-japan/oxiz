//! Format validation utilities for proof formats.
//!
//! This module provides validation for various proof formats to ensure
//! correctness before export or conversion.

use crate::proof::Proof;
use rustc_hash::FxHashSet;
use std::fmt;

/// Result of format validation.
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Errors that can occur during format validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// Missing required field
    MissingField { field: String, location: String },
    /// Invalid step reference
    InvalidReference { step_id: String, reference: String },
    /// Malformed proof structure
    MalformedStructure { reason: String },
    /// Unsupported rule or operation
    UnsupportedFeature { feature: String, format: String },
    /// Invalid conclusion format
    InvalidConclusion { conclusion: String, reason: String },
    /// Empty proof
    EmptyProof,
    /// Circular dependency detected
    CircularDependency { steps: Vec<String> },
    /// Type mismatch in proof
    TypeMismatch { expected: String, found: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::MissingField { field, location } => {
                write!(f, "Missing required field '{}' in {}", field, location)
            }
            ValidationError::InvalidReference { step_id, reference } => {
                write!(f, "Invalid reference '{}' in step '{}'", reference, step_id)
            }
            ValidationError::MalformedStructure { reason } => {
                write!(f, "Malformed proof structure: {}", reason)
            }
            ValidationError::UnsupportedFeature { feature, format } => {
                write!(f, "Unsupported feature '{}' in {} format", feature, format)
            }
            ValidationError::InvalidConclusion { conclusion, reason } => {
                write!(f, "Invalid conclusion '{}': {}", conclusion, reason)
            }
            ValidationError::EmptyProof => write!(f, "Proof is empty"),
            ValidationError::CircularDependency { steps } => {
                write!(f, "Circular dependency detected: {}", steps.join(" -> "))
            }
            ValidationError::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validator for proof formats.
pub struct FormatValidator {
    /// Allow empty proofs
    allow_empty: bool,
    /// Check for circular dependencies
    check_cycles: bool,
    /// Validate conclusion syntax
    validate_syntax: bool,
}

impl Default for FormatValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatValidator {
    /// Create a new format validator with default settings.
    pub fn new() -> Self {
        Self {
            allow_empty: false,
            check_cycles: true,
            validate_syntax: true,
        }
    }

    /// Allow empty proofs.
    pub fn allow_empty(mut self, allow: bool) -> Self {
        self.allow_empty = allow;
        self
    }

    /// Enable/disable cycle checking.
    pub fn check_cycles(mut self, check: bool) -> Self {
        self.check_cycles = check;
        self
    }

    /// Enable/disable syntax validation.
    pub fn validate_syntax(mut self, validate: bool) -> Self {
        self.validate_syntax = validate;
        self
    }

    /// Validate a generic proof.
    pub fn validate_proof(&self, proof: &Proof) -> ValidationResult<()> {
        // Check if proof is empty
        if proof.is_empty() {
            if self.allow_empty {
                return Ok(());
            } else {
                return Err(ValidationError::EmptyProof);
            }
        }

        // Check for circular dependencies
        if self.check_cycles {
            self.check_proof_cycles(proof)?;
        }

        // Validate each node
        for node in proof.nodes() {
            if self.validate_syntax {
                self.validate_conclusion_syntax(node.conclusion())?;
            }
        }

        Ok(())
    }

    // Helper: Check for circular dependencies in proof
    fn check_proof_cycles(&self, proof: &Proof) -> ValidationResult<()> {
        let mut visiting = FxHashSet::default();
        let mut visited = FxHashSet::default();
        let mut path = Vec::new();

        for node in proof.nodes() {
            if !visited.contains(&node.id) {
                Self::visit_node(proof, node.id, &mut visiting, &mut visited, &mut path)?;
            }
        }

        Ok(())
    }

    // Helper: Visit node in DFS for cycle detection
    fn visit_node(
        proof: &Proof,
        node_id: crate::proof::ProofNodeId,
        visiting: &mut FxHashSet<crate::proof::ProofNodeId>,
        visited: &mut FxHashSet<crate::proof::ProofNodeId>,
        path: &mut Vec<String>,
    ) -> ValidationResult<()> {
        if visiting.contains(&node_id) {
            // Cycle detected
            path.push(node_id.to_string());
            return Err(ValidationError::CircularDependency {
                steps: path.clone(),
            });
        }

        if visited.contains(&node_id) {
            return Ok(());
        }

        visiting.insert(node_id);
        path.push(node_id.to_string());

        // Visit premises
        if let Some(node) = proof.get_node(node_id)
            && let crate::proof::ProofStep::Inference { premises, .. } = &node.step
        {
            for &premise_id in premises.iter() {
                Self::visit_node(proof, premise_id, visiting, visited, path)?;
            }
        }

        path.pop();
        visiting.remove(&node_id);
        visited.insert(node_id);

        Ok(())
    }

    // Helper: Validate conclusion syntax
    fn validate_conclusion_syntax(&self, conclusion: &str) -> ValidationResult<()> {
        if conclusion.trim().is_empty() {
            return Err(ValidationError::InvalidConclusion {
                conclusion: conclusion.to_string(),
                reason: "Empty conclusion".to_string(),
            });
        }

        // Check for balanced parentheses
        let mut depth = 0;
        for ch in conclusion.chars() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth < 0 {
                        return Err(ValidationError::InvalidConclusion {
                            conclusion: conclusion.to_string(),
                            reason: "Unbalanced parentheses".to_string(),
                        });
                    }
                }
                _ => {}
            }
        }

        if depth != 0 {
            return Err(ValidationError::InvalidConclusion {
                conclusion: conclusion.to_string(),
                reason: "Unbalanced parentheses".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_new() {
        let validator = FormatValidator::new();
        assert!(!validator.allow_empty);
        assert!(validator.check_cycles);
        assert!(validator.validate_syntax);
    }

    #[test]
    fn test_validator_with_settings() {
        let validator = FormatValidator::new()
            .allow_empty(true)
            .check_cycles(false)
            .validate_syntax(false);
        assert!(validator.allow_empty);
        assert!(!validator.check_cycles);
        assert!(!validator.validate_syntax);
    }

    #[test]
    fn test_validate_empty_proof() {
        let validator = FormatValidator::new();
        let proof = Proof::new();
        assert!(validator.validate_proof(&proof).is_err());

        let validator = FormatValidator::new().allow_empty(true);
        assert!(validator.validate_proof(&proof).is_ok());
    }

    #[test]
    fn test_validate_syntax_balanced_parens() {
        let validator = FormatValidator::new();
        assert!(validator.validate_conclusion_syntax("(x = y)").is_ok());
        assert!(validator.validate_conclusion_syntax("f(x, g(y))").is_ok());
    }

    #[test]
    fn test_validate_syntax_unbalanced_parens() {
        let validator = FormatValidator::new();
        assert!(validator.validate_conclusion_syntax("(x = y").is_err());
        assert!(validator.validate_conclusion_syntax("x = y)").is_err());
    }

    #[test]
    fn test_validate_syntax_empty() {
        let validator = FormatValidator::new();
        assert!(validator.validate_conclusion_syntax("").is_err());
        assert!(validator.validate_conclusion_syntax("   ").is_err());
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::EmptyProof;
        assert_eq!(err.to_string(), "Proof is empty");

        let err = ValidationError::MissingField {
            field: "conclusion".to_string(),
            location: "step 5".to_string(),
        };
        assert!(err.to_string().contains("Missing required field"));
    }

    #[test]
    fn test_validate_nonempty_proof() {
        let validator = FormatValidator::new();
        let mut proof = Proof::new();
        proof.add_axiom("x = x");
        assert!(validator.validate_proof(&proof).is_ok());
    }

    #[test]
    fn test_validate_with_invalid_syntax() {
        let validator = FormatValidator::new();
        assert!(validator.validate_conclusion_syntax("(x = y").is_err());
    }
}
