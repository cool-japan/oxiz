//! Proof checking infrastructure.
//!
//! This module provides validation and verification of proof steps,
//! ensuring that proof derivations are sound.
//!
//! ## Features
//!
//! - **Syntactic checks**: Validate proof structure (premises exist, etc.)
//! - **Rule validation**: Check that rule applications are well-formed
//! - **Extensible**: Support for custom rule validators
//!
//! ## Example
//!
//! ```
//! use oxiz_proof::checker::{ProofChecker, CheckResult};
//! use oxiz_proof::theory::{TheoryProof, TheoryRule};
//!
//! let mut proof = TheoryProof::new();
//! proof.refl("x");
//!
//! let mut checker = ProofChecker::new();
//! let result = checker.check_theory_proof(&proof);
//! assert!(result.is_valid());
//! ```

use crate::alethe::{AletheProof, AletheRule, AletheStep};
use crate::theory::{TheoryProof, TheoryRule, TheoryStepId};
use std::collections::HashSet;
use std::fmt;

/// Result of checking a proof step
#[derive(Debug, Clone)]
pub enum CheckResult {
    /// The proof is valid
    Valid,
    /// The proof has an error at a specific step
    Invalid {
        /// Step index where the error occurred
        step: u32,
        /// Description of the error
        error: CheckError,
    },
    /// Multiple errors were found
    MultipleErrors(Vec<(u32, CheckError)>),
}

impl CheckResult {
    /// Check if the result indicates a valid proof
    #[must_use]
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Get the error if there is one
    #[must_use]
    pub fn error(&self) -> Option<&CheckError> {
        match self {
            Self::Invalid { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl fmt::Display for CheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Valid => write!(f, "✓ Proof is valid"),
            Self::Invalid { step, error } => {
                writeln!(f, "✗ Proof is invalid")?;
                writeln!(f, "  Step: {}", step)?;
                writeln!(f, "  [{:?}] {}", error.severity(), error)?;
                if let Some(suggestion) = error.suggestion() {
                    writeln!(f, "  Suggestion: {}", suggestion)?;
                }
                Ok(())
            }
            Self::MultipleErrors(errors) => {
                writeln!(f, "✗ Proof has {} error(s):", errors.len())?;
                for (step, error) in errors {
                    writeln!(f, "\n  Step {}:", step)?;
                    writeln!(f, "    [{:?}] {}", error.severity(), error)?;
                    if let Some(suggestion) = error.suggestion() {
                        writeln!(f, "    Suggestion: {}", suggestion)?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// Types of proof checking errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckError {
    /// A referenced premise doesn't exist
    MissingPremise(u32),
    /// Wrong number of premises for the rule
    WrongPremiseCount { expected: usize, got: usize },
    /// Wrong number of arguments for the rule
    WrongArgumentCount { expected: usize, got: usize },
    /// Rule is not applicable
    RuleNotApplicable(String),
    /// Conclusion doesn't follow from premises
    InvalidConclusion(String),
    /// Cyclic dependency in proof
    CyclicDependency,
    /// Empty proof
    EmptyProof,
    /// Malformed term in proof
    MalformedTerm(String),
    /// Unknown rule
    UnknownRule(String),
    /// Custom error
    Custom(String),
}

impl fmt::Display for CheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPremise(id) => {
                write!(
                    f,
                    "Missing premise: step {} does not exist or has not been defined yet. \
                     Premises must be defined before they are referenced.",
                    id
                )
            }
            Self::WrongPremiseCount { expected, got } => {
                write!(
                    f,
                    "Wrong premise count: rule requires {} premise(s), but {} {} provided. \
                     Check the rule definition for the correct number of premises.",
                    expected,
                    got,
                    if *got == 1 { "was" } else { "were" }
                )
            }
            Self::WrongArgumentCount { expected, got } => {
                write!(
                    f,
                    "Wrong argument count: rule expects {} argument(s), but {} {} provided. \
                     Ensure all required arguments are supplied.",
                    expected,
                    got,
                    if *got == 1 { "was" } else { "were" }
                )
            }
            Self::RuleNotApplicable(msg) => {
                write!(
                    f,
                    "Rule not applicable: {}. \
                     Verify that the rule's preconditions are met.",
                    msg
                )
            }
            Self::InvalidConclusion(msg) => {
                write!(
                    f,
                    "Invalid conclusion: {}. \
                     The conclusion does not follow from the premises using the specified rule.",
                    msg
                )
            }
            Self::CyclicDependency => {
                write!(
                    f,
                    "Cyclic dependency detected in proof structure. \
                     A proof step cannot depend on itself (directly or indirectly). \
                     Check for circular references in premise chains."
                )
            }
            Self::EmptyProof => {
                write!(
                    f,
                    "Empty proof: no proof steps provided. \
                     A valid proof must contain at least one step."
                )
            }
            Self::MalformedTerm(msg) => {
                write!(
                    f,
                    "Malformed term: {}. \
                     Check for syntax errors or invalid term structure.",
                    msg
                )
            }
            Self::UnknownRule(name) => {
                write!(
                    f,
                    "Unknown rule: '{}'. \
                     This rule is not recognized by the proof checker. \
                     Verify the rule name is spelled correctly.",
                    name
                )
            }
            Self::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for CheckError {}

impl CheckError {
    /// Get a suggestion for fixing this error
    #[must_use]
    pub fn suggestion(&self) -> Option<&str> {
        match self {
            Self::MissingPremise(_) => {
                Some("Ensure all premise steps are added to the proof before referencing them.")
            }
            Self::WrongPremiseCount { .. } => {
                Some("Consult the rule documentation for the correct number of premises.")
            }
            Self::WrongArgumentCount { .. } => {
                Some("Review the rule definition to determine which arguments are required.")
            }
            Self::RuleNotApplicable(_) => {
                Some("Check that the premise types match what the rule expects.")
            }
            Self::InvalidConclusion(_) => {
                Some("Verify that the rule is being applied correctly to the given premises.")
            }
            Self::CyclicDependency => {
                Some("Reorganize proof steps to eliminate circular dependencies.")
            }
            Self::EmptyProof => Some("Add at least one axiom or assumption to the proof."),
            Self::MalformedTerm(_) => Some("Check the term syntax against the expected format."),
            Self::UnknownRule(_) => {
                Some("Use a standard proof rule or define a custom rule handler.")
            }
            Self::Custom(_) => None,
        }
    }

    /// Get the severity level of this error
    #[must_use]
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::CyclicDependency | Self::EmptyProof => ErrorSeverity::Critical,
            Self::MissingPremise(_) | Self::InvalidConclusion(_) | Self::UnknownRule(_) => {
                ErrorSeverity::Error
            }
            Self::WrongPremiseCount { .. }
            | Self::WrongArgumentCount { .. }
            | Self::RuleNotApplicable(_)
            | Self::MalformedTerm(_) => ErrorSeverity::Warning,
            Self::Custom(_) => ErrorSeverity::Error,
        }
    }
}

/// Error severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Warning - proof may be acceptable
    Warning,
    /// Error - proof is invalid
    Error,
    /// Critical - proof structure is fundamentally broken
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Configuration for proof checking
#[derive(Debug, Clone, Default)]
pub struct CheckerConfig {
    /// Whether to continue checking after the first error
    pub continue_on_error: bool,
    /// Whether to verify conclusion content (not just structure)
    pub verify_conclusions: bool,
    /// Whether to allow cyclic dependencies (for some proof formats)
    pub allow_cycles: bool,
}

/// Proof checker for verifying proof derivations
#[derive(Debug, Default)]
pub struct ProofChecker {
    /// Configuration
    config: CheckerConfig,
    /// Collected errors
    errors: Vec<(u32, CheckError)>,
    /// Validated step IDs (for cycle detection)
    validated: HashSet<u32>,
    /// Currently being validated (for cycle detection)
    in_progress: HashSet<u32>,
}

impl ProofChecker {
    /// Create a new proof checker with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a proof checker with custom configuration
    #[must_use]
    pub fn with_config(config: CheckerConfig) -> Self {
        Self {
            config,
            errors: Vec::new(),
            validated: HashSet::new(),
            in_progress: HashSet::new(),
        }
    }

    /// Reset the checker state
    pub fn reset(&mut self) {
        self.errors.clear();
        self.validated.clear();
        self.in_progress.clear();
    }

    /// Check a theory proof
    pub fn check_theory_proof(&mut self, proof: &TheoryProof) -> CheckResult {
        self.reset();

        if proof.is_empty() {
            return CheckResult::Invalid {
                step: 0,
                error: CheckError::EmptyProof,
            };
        }

        // Check each step
        for step in proof.steps() {
            if let Err(error) = self.check_theory_step(proof, step.id) {
                if self.config.continue_on_error {
                    self.errors.push((step.id.0, error));
                } else {
                    return CheckResult::Invalid {
                        step: step.id.0,
                        error,
                    };
                }
            }
        }

        if self.errors.is_empty() {
            CheckResult::Valid
        } else {
            CheckResult::MultipleErrors(std::mem::take(&mut self.errors))
        }
    }

    /// Check a single theory proof step
    fn check_theory_step(
        &mut self,
        proof: &TheoryProof,
        step_id: TheoryStepId,
    ) -> Result<(), CheckError> {
        // Cycle detection
        if !self.config.allow_cycles {
            if self.in_progress.contains(&step_id.0) {
                return Err(CheckError::CyclicDependency);
            }
            if self.validated.contains(&step_id.0) {
                return Ok(());
            }
            self.in_progress.insert(step_id.0);
        }

        let step = proof
            .get_step(step_id)
            .ok_or(CheckError::MissingPremise(step_id.0))?;

        // Check premises exist
        for premise_id in &step.premises {
            if proof.get_step(*premise_id).is_none() {
                return Err(CheckError::MissingPremise(premise_id.0));
            }

            // Recursively check premises
            if !self.config.allow_cycles {
                self.check_theory_step(proof, *premise_id)?;
            }
        }

        // Check rule-specific requirements
        self.check_theory_rule(&step.rule, step.premises.len(), step.args.len())?;

        // Mark as validated
        if !self.config.allow_cycles {
            self.in_progress.remove(&step_id.0);
            self.validated.insert(step_id.0);
        }

        Ok(())
    }

    /// Check rule-specific requirements for theory proofs
    fn check_theory_rule(
        &self,
        rule: &TheoryRule,
        premise_count: usize,
        arg_count: usize,
    ) -> Result<(), CheckError> {
        match rule {
            // Rules with no premises
            TheoryRule::Refl => {
                if premise_count != 0 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 0,
                        got: premise_count,
                    });
                }
            }

            // Rules with exactly one premise
            TheoryRule::Symm => {
                if premise_count != 1 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 1,
                        got: premise_count,
                    });
                }
            }

            // Rules with exactly two premises
            TheoryRule::Trans => {
                if premise_count != 2 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 2,
                        got: premise_count,
                    });
                }
            }

            // Rules with at least one premise (congruence needs arg equalities)
            TheoryRule::Cong => {
                // Congruence can have zero premises for nullary functions
            }

            // Farkas lemma needs at least 2 premises
            TheoryRule::LaGeneric => {
                if premise_count < 2 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 2,
                        got: premise_count,
                    });
                }
            }

            // Array read-write-same is an axiom
            TheoryRule::ArrReadWrite1 => {
                if premise_count != 0 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 0,
                        got: premise_count,
                    });
                }
            }

            // Array read-write-different needs proof of i ≠ j
            TheoryRule::ArrReadWrite2 => {
                if premise_count != 1 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 1,
                        got: premise_count,
                    });
                }
            }

            // LaMult needs coefficient argument
            TheoryRule::LaMult => {
                if arg_count < 1 {
                    return Err(CheckError::WrongArgumentCount {
                        expected: 1,
                        got: arg_count,
                    });
                }
            }

            // Other rules - flexible checking
            _ => {}
        }

        Ok(())
    }

    /// Check an Alethe proof
    pub fn check_alethe_proof(&mut self, proof: &AletheProof) -> CheckResult {
        self.reset();

        if proof.is_empty() {
            return CheckResult::Invalid {
                step: 0,
                error: CheckError::EmptyProof,
            };
        }

        let steps = proof.steps();
        let mut step_indices: HashSet<u32> = HashSet::new();

        // First pass: collect all step indices
        for step in steps {
            match step {
                AletheStep::Assume { index, .. } => {
                    step_indices.insert(*index);
                }
                AletheStep::Step { index, .. } => {
                    step_indices.insert(*index);
                }
                AletheStep::Anchor { step: index, .. } => {
                    step_indices.insert(*index);
                }
                AletheStep::DefineFun { .. } => {}
            }
        }

        // Second pass: check each step
        for (idx, step) in steps.iter().enumerate() {
            if let Err(error) = self.check_alethe_step(step, &step_indices) {
                if self.config.continue_on_error {
                    self.errors.push((idx as u32, error));
                } else {
                    return CheckResult::Invalid {
                        step: idx as u32,
                        error,
                    };
                }
            }
        }

        if self.errors.is_empty() {
            CheckResult::Valid
        } else {
            CheckResult::MultipleErrors(std::mem::take(&mut self.errors))
        }
    }

    /// Check a single Alethe proof step
    fn check_alethe_step(
        &self,
        step: &AletheStep,
        step_indices: &HashSet<u32>,
    ) -> Result<(), CheckError> {
        match step {
            AletheStep::Assume { .. } => {
                // Assumptions don't need checking
                Ok(())
            }

            AletheStep::Step { rule, premises, .. } => {
                // Check all premises exist
                for premise in premises {
                    if !step_indices.contains(premise) {
                        return Err(CheckError::MissingPremise(*premise));
                    }
                }

                // Check rule-specific requirements
                self.check_alethe_rule(rule, premises.len())
            }

            AletheStep::Anchor { .. } => {
                // Anchors don't need checking
                Ok(())
            }

            AletheStep::DefineFun { .. } => {
                // Definitions don't need checking
                Ok(())
            }
        }
    }

    /// Check rule-specific requirements for Alethe proofs
    fn check_alethe_rule(&self, rule: &AletheRule, premise_count: usize) -> Result<(), CheckError> {
        match rule {
            // Resolution needs at least 2 premises
            AletheRule::Resolution => {
                if premise_count < 2 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 2,
                        got: premise_count,
                    });
                }
            }

            // Reflexivity is an axiom
            AletheRule::Refl => {
                if premise_count != 0 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 0,
                        got: premise_count,
                    });
                }
            }

            // Transitivity needs at least 2 premises
            AletheRule::Trans => {
                if premise_count < 2 {
                    return Err(CheckError::WrongPremiseCount {
                        expected: 2,
                        got: premise_count,
                    });
                }
            }

            // Other rules - flexible checking
            _ => {}
        }

        Ok(())
    }
}

/// Trait for types that can be checked for validity
pub trait Checkable {
    /// Check if the proof is valid
    fn check(&self) -> CheckResult;

    /// Check using a custom checker configuration
    fn check_with_config(&self, config: CheckerConfig) -> CheckResult;
}

impl Checkable for TheoryProof {
    fn check(&self) -> CheckResult {
        ProofChecker::new().check_theory_proof(self)
    }

    fn check_with_config(&self, config: CheckerConfig) -> CheckResult {
        ProofChecker::with_config(config).check_theory_proof(self)
    }
}

impl Checkable for AletheProof {
    fn check(&self) -> CheckResult {
        ProofChecker::new().check_alethe_proof(self)
    }

    fn check_with_config(&self, config: CheckerConfig) -> CheckResult {
        ProofChecker::with_config(config).check_alethe_proof(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::theory::ProofTerm;

    #[test]
    fn test_check_result_is_valid() {
        assert!(CheckResult::Valid.is_valid());
        assert!(
            !CheckResult::Invalid {
                step: 0,
                error: CheckError::EmptyProof
            }
            .is_valid()
        );
    }

    #[test]
    fn test_check_error_display() {
        let err = CheckError::MissingPremise(5);
        let msg = format!("{}", err);
        assert!(msg.contains("5"));
        assert!(msg.contains("does not exist"));

        let err = CheckError::WrongPremiseCount {
            expected: 2,
            got: 1,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("requires 2"));
        assert!(msg.contains("1 was provided"));
    }

    #[test]
    fn test_theory_proof_empty() {
        let proof = TheoryProof::new();
        let result = proof.check();
        assert!(!result.is_valid());
        assert!(matches!(result.error(), Some(CheckError::EmptyProof)));
    }

    #[test]
    fn test_theory_proof_valid_refl() {
        let mut proof = TheoryProof::new();
        proof.refl("x");

        let result = proof.check();
        assert!(result.is_valid());
    }

    #[test]
    fn test_theory_proof_valid_transitivity() {
        let mut proof = TheoryProof::new();
        let s1 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= a b)");
        let s2 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= b c)");
        proof.trans(s1, s2, "a", "c");

        let result = proof.check();
        assert!(result.is_valid());
    }

    #[test]
    fn test_theory_proof_invalid_trans_premises() {
        let mut proof = TheoryProof::new();
        let s1 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= a b)");
        // Trans with only 1 premise should fail
        proof.add_step(TheoryRule::Trans, vec![s1], "(= a c)");

        let result = proof.check();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_theory_proof_missing_premise() {
        let mut proof = TheoryProof::new();
        // Reference a non-existent premise
        proof.add_step(
            TheoryRule::Trans,
            vec![TheoryStepId(99), TheoryStepId(100)],
            "(= a c)",
        );

        let result = proof.check();
        assert!(!result.is_valid());
        assert!(matches!(
            result.error(),
            Some(CheckError::MissingPremise(_))
        ));
    }

    #[test]
    fn test_alethe_proof_empty() {
        let proof = AletheProof::new();
        let result = proof.check();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_alethe_proof_valid() {
        let mut proof = AletheProof::new();
        proof.assume("p");
        proof.step_simple(vec![], AletheRule::Refl);

        let result = proof.check();
        assert!(result.is_valid());
    }

    #[test]
    fn test_checker_continue_on_error() {
        let mut proof = TheoryProof::new();
        // Multiple invalid steps
        proof.add_step(TheoryRule::Trans, vec![TheoryStepId(99)], "(= a b)");
        proof.add_step(TheoryRule::Trans, vec![TheoryStepId(100)], "(= c d)");

        let config = CheckerConfig {
            continue_on_error: true,
            ..Default::default()
        };

        let result = proof.check_with_config(config);
        assert!(matches!(result, CheckResult::MultipleErrors(_)));
    }

    #[test]
    fn test_checker_refl_with_premises_fails() {
        let mut proof = TheoryProof::new();
        let s1 = proof.add_axiom(TheoryRule::Custom("assert".into()), "(= a b)");
        // Refl should have no premises
        proof.add_step(TheoryRule::Refl, vec![s1], "(= x x)");

        let result = proof.check();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_checker_farkas_needs_premises() {
        let mut proof = TheoryProof::new();
        // Farkas with only 1 premise should fail
        let s1 = proof.add_axiom(TheoryRule::Custom("bound".into()), "(>= x 0)");
        proof.add_step(TheoryRule::LaGeneric, vec![s1], "false");

        let result = proof.check();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_checker_arr_read_write_1_axiom() {
        let mut proof = TheoryProof::new();
        // ArrReadWrite1 is an axiom (no premises)
        proof.add_axiom(TheoryRule::ArrReadWrite1, "(= (select (store a i v) i) v)");

        let result = proof.check();
        assert!(result.is_valid());
    }

    #[test]
    fn test_checker_arr_read_write_2_needs_premise() {
        let mut proof = TheoryProof::new();
        // ArrReadWrite2 needs proof of i ≠ j
        proof.add_axiom(
            TheoryRule::ArrReadWrite2,
            "(= (select (store a i v) j) (select a j))",
        );

        let result = proof.check();
        assert!(!result.is_valid());
    }
}
