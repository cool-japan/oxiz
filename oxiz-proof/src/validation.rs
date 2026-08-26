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

/// One step of the iterative cycle-detection DFS.
enum CycleStep {
    /// Enter a node: check it against the current DFS path, then schedule its
    /// premises.
    Enter(crate::proof::ProofNodeId),
    /// Leave a node: its whole premise subtree is done.
    Leave(crate::proof::ProofNodeId),
}

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
    //
    // Driven by an explicit stack rather than recursion. The recursion depth
    // here was the length of the longest premise chain, which grows linearly
    // with the number of learned clauses in a resolution proof — a realistic
    // proof from a long solver run overflowed the stack while merely being
    // *validated*.
    //
    // The `Enter`/`Leave` pair reproduces the recursive pre/post order
    // exactly: `Leave` is scheduled before a node's premises, so it runs only
    // once every premise subtree is finished, which is where the recursive
    // form popped `path` and moved the node from `visiting` to `visited`.
    fn visit_node(
        proof: &Proof,
        node_id: crate::proof::ProofNodeId,
        visiting: &mut FxHashSet<crate::proof::ProofNodeId>,
        visited: &mut FxHashSet<crate::proof::ProofNodeId>,
        path: &mut Vec<String>,
    ) -> ValidationResult<()> {
        let mut stack = vec![CycleStep::Enter(node_id)];

        while let Some(step) = stack.pop() {
            match step {
                CycleStep::Enter(id) => {
                    if visiting.contains(&id) {
                        // Cycle detected
                        path.push(id.to_string());
                        return Err(ValidationError::CircularDependency {
                            steps: path.clone(),
                        });
                    }

                    if visited.contains(&id) {
                        continue;
                    }

                    visiting.insert(id);
                    path.push(id.to_string());
                    stack.push(CycleStep::Leave(id));

                    // Visit premises, pushed in reverse so they are entered
                    // left to right.
                    if let Some(node) = proof.get_node(id)
                        && let crate::proof::ProofStep::Inference { premises, .. } = &node.step
                    {
                        for &premise_id in premises.iter().rev() {
                            stack.push(CycleStep::Enter(premise_id));
                        }
                    }
                }
                CycleStep::Leave(id) => {
                    path.pop();
                    visiting.remove(&id);
                    visited.insert(id);
                }
            }
        }

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

    /// The stack size and `CHAIN_LEN` are scaled together on purpose: what is
    /// pinned is the ratio, ~17 bytes per DFS frame, which no real call frame
    /// fits into. Never raise one without raising the other.
    #[test]
    fn test_cycle_check_deep_premise_chain_does_not_overflow() {
        const CHAIN_LEN: u32 = 7_500;

        let handle = std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(|| {
                // A resolution-style chain: every step's only premise is the
                // previous step, so the DFS depth equals the chain length.
                let mut proof = Proof::new();
                let mut current = proof.add_axiom("c0");
                for i in 1..CHAIN_LEN {
                    current = proof.add_inference("resolve", vec![current], format!("c{i}"));
                }

                FormatValidator::new().validate_proof(&proof)
            })
            .expect("thread spawn should succeed");

        assert!(
            handle
                .join()
                .expect("deep cycle check must not overflow")
                .is_ok(),
            "an acyclic chain must validate"
        );
    }

    #[test]
    fn test_cycle_check_shared_premises_are_visited_once() {
        // A diamond DAG: without the `visited` set this re-expands
        // exponentially. 60 levels would be 2^60 visits.
        // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — this pins
        // 2^60-versus-60 sharing work, not stack depth; scaling the depth
        // down would destroy the test. See TODO.md "v0.3.2 backlog".
        let handle = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                let mut proof = Proof::new();
                let mut left = proof.add_axiom("l0");
                let mut right = proof.add_axiom("r0");
                for i in 1..60u32 {
                    let next_left =
                        proof.add_inference("resolve", vec![left, right], format!("l{i}"));
                    let next_right =
                        proof.add_inference("resolve", vec![right, left], format!("r{i}"));
                    left = next_left;
                    right = next_right;
                }
                FormatValidator::new().validate_proof(&proof)
            })
            .expect("thread spawn should succeed");

        assert!(handle.join().expect("diamond check must terminate").is_ok());
    }

    #[test]
    fn test_cycle_check_still_detects_a_cycle() {
        use crate::proof::ProofNodeId;

        let mut proof = Proof::new();
        // p0, then a mutually-referential pair: p1's premise is p2 (a
        // forward reference that becomes valid once p2 exists) and p2's
        // premise is p1.
        let first = proof.add_axiom("a");
        assert_eq!(first, ProofNodeId(0));
        let second = proof.add_inference("resolve", vec![ProofNodeId(2)], "b");
        let third = proof.add_inference("resolve", vec![ProofNodeId(1)], "c");
        assert_eq!((second, third), (ProofNodeId(1), ProofNodeId(2)));

        let err = FormatValidator::new()
            .validate_proof(&proof)
            .expect_err("a cycle must be reported");
        match err {
            ValidationError::CircularDependency { steps } => {
                let mut seen = FxHashSet::default();
                let repeated = steps.iter().any(|s| !seen.insert(s.clone()));
                assert!(repeated, "the path must revisit a node: {steps:?}");
            }
            other => panic!("expected CircularDependency, got {other:?}"),
        }
    }
}
