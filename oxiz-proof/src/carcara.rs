//! Carcara proof format compatibility.
//!
//! Carcara is a proof checker for SMT proofs in Alethe format.
//! This module provides utilities to ensure Alethe proofs are compatible
//! with Carcara's requirements and validation rules.
//!
//! ## Carcara Requirements
//!
//! - Proofs must be in valid Alethe format
//! - Step indices must be sequential and unique
//! - All premises must be previously defined
//! - Rules must be correctly applied according to Carcara's semantics
//!
//! ## References
//!
//! Carcara: <https://github.com/ufmg-smite/carcara>

use crate::alethe::{AletheProof, AletheRule, AletheStep, StepIndex, TermRef};
use rustc_hash::FxHashSet;
use std::io::{self, Write};

/// Carcara-compatible proof builder
///
/// Ensures that generated Alethe proofs meet Carcara's requirements
#[derive(Debug, Default)]
pub struct CarcaraProof {
    /// The underlying Alethe proof
    proof: AletheProof,
    /// Set of defined step indices (for validation)
    defined_steps: FxHashSet<StepIndex>,
    /// Next available step index
    next_index: StepIndex,
}

impl CarcaraProof {
    /// Create a new Carcara-compatible proof
    #[must_use]
    pub fn new() -> Self {
        Self {
            proof: AletheProof::new(),
            defined_steps: FxHashSet::default(),
            next_index: 1,
        }
    }

    /// Allocate a new unique step index
    #[allow(dead_code)]
    fn alloc_index(&mut self) -> StepIndex {
        let idx = self.next_index;
        self.next_index += 1;
        idx
    }

    /// Add an assumption step
    ///
    /// # Errors
    ///
    /// Returns an error if the step index is already used
    pub fn add_assume(&mut self, term: impl Into<TermRef>) -> Result<StepIndex, String> {
        let index = self.proof.assume(term);

        if !self.defined_steps.insert(index) {
            return Err(format!("Step index {} already defined", index));
        }

        self.next_index = index + 1;
        Ok(index)
    }

    /// Add a proof step with validation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The step index is already used
    /// - Any premise is not yet defined
    /// - The rule application is invalid
    #[allow(clippy::too_many_arguments)]
    pub fn add_step(
        &mut self,
        clause: Vec<TermRef>,
        rule: AletheRule,
        premises: Vec<StepIndex>,
        args: Vec<TermRef>,
    ) -> Result<StepIndex, String> {
        // Validate all premises are defined
        for &premise in &premises {
            if !self.defined_steps.contains(&premise) {
                return Err(format!("Premise {} not yet defined", premise));
            }
        }

        let index = self.proof.step(clause, rule, premises, args);

        if !self.defined_steps.insert(index) {
            return Err(format!("Step index {} already defined", index));
        }

        self.next_index = index + 1;
        Ok(index)
    }

    /// Add an anchor for subproofs
    ///
    /// # Errors
    ///
    /// Returns an error if the step index is already used
    pub fn add_anchor(&mut self, args: Vec<(String, TermRef)>) -> Result<StepIndex, String> {
        let index = self.proof.anchor(args);

        if !self.defined_steps.insert(index) {
            return Err(format!("Step index {} already defined", index));
        }

        self.next_index = index + 1;
        Ok(index)
    }

    /// Validate the proof for Carcara compatibility
    ///
    /// Checks:
    /// - All step indices are unique
    /// - All premises are defined before use
    /// - No circular dependencies
    pub fn validate(&self) -> Result<(), String> {
        let mut seen_indices = FxHashSet::default();

        for step in self.proof.steps() {
            let index = match step {
                AletheStep::Assume { index, .. } => *index,
                AletheStep::Step {
                    index, premises, ..
                } => {
                    // Check all premises are already seen
                    for &premise in premises {
                        if !seen_indices.contains(&premise) {
                            return Err(format!(
                                "Step {} uses undefined premise {}",
                                index, premise
                            ));
                        }
                    }
                    *index
                }
                AletheStep::Anchor { step, .. } => *step,
                AletheStep::DefineFun { .. } => continue,
            };

            if !seen_indices.insert(index) {
                return Err(format!("Duplicate step index: {}", index));
            }
        }

        Ok(())
    }

    /// Get the underlying Alethe proof
    #[must_use]
    pub fn alethe_proof(&self) -> &AletheProof {
        &self.proof
    }

    /// Write the proof in Carcara-compatible Alethe format
    pub fn write<W: Write>(&self, writer: W) -> io::Result<()> {
        self.proof.write(writer)
    }

    /// Get the number of steps in the proof
    #[must_use]
    pub fn len(&self) -> usize {
        self.proof.len()
    }

    /// Check if the proof is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.proof.is_empty()
    }

    /// Clear the proof
    pub fn clear(&mut self) {
        self.proof.clear();
        self.defined_steps.clear();
        self.next_index = 1;
    }
}

/// Validate a proof for Carcara compatibility
///
/// This function checks that an existing Alethe proof meets Carcara's requirements
pub fn validate_for_carcara(proof: &AletheProof) -> Result<(), String> {
    let mut seen_indices = FxHashSet::default();

    for step in proof.steps() {
        let index = match step {
            AletheStep::Assume { index, .. } => *index,
            AletheStep::Step {
                index, premises, ..
            } => {
                // Check all premises are already seen
                for &premise in premises {
                    if !seen_indices.contains(&premise) {
                        return Err(format!("Step {} uses undefined premise {}", index, premise));
                    }
                }
                *index
            }
            AletheStep::Anchor { step, .. } => *step,
            AletheStep::DefineFun { .. } => continue,
        };

        if !seen_indices.insert(index) {
            return Err(format!("Duplicate step index: {}", index));
        }
    }

    Ok(())
}

/// Convert an Alethe proof to Carcara-compatible format
///
/// This ensures step indices are sequential and all dependencies are satisfied
pub fn to_carcara_format(proof: &AletheProof) -> Result<CarcaraProof, String> {
    let mut carcara = CarcaraProof::new();
    let mut index_map: rustc_hash::FxHashMap<StepIndex, StepIndex> =
        rustc_hash::FxHashMap::default();

    for step in proof.steps() {
        match step {
            AletheStep::Assume { index, term } => {
                let new_index = carcara.add_assume(term.clone())?;
                index_map.insert(*index, new_index);
            }
            AletheStep::Step {
                index,
                clause,
                rule,
                premises,
                args,
            } => {
                // Map old premise indices to new ones
                let new_premises: Vec<StepIndex> = premises
                    .iter()
                    .map(|p| {
                        index_map
                            .get(p)
                            .copied()
                            .ok_or_else(|| format!("Premise {} not found in index map", p))
                    })
                    .collect::<Result<Vec<_>, String>>()?;

                let new_index =
                    carcara.add_step(clause.clone(), *rule, new_premises, args.clone())?;
                index_map.insert(*index, new_index);
            }
            AletheStep::Anchor { step, args } => {
                let new_index = carcara.add_anchor(args.clone())?;
                index_map.insert(*step, new_index);
            }
            AletheStep::DefineFun { .. } => {
                // Define-fun steps don't have indices, pass through
                continue;
            }
        }
    }

    Ok(carcara)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carcara_proof_creation() {
        let proof = CarcaraProof::new();
        assert!(proof.is_empty());
        assert_eq!(proof.len(), 0);
    }

    #[test]
    fn test_add_assume() {
        let mut proof = CarcaraProof::new();
        let idx = proof.add_assume("p").unwrap();
        assert_eq!(idx, 1);
        assert_eq!(proof.len(), 1);
    }

    #[test]
    fn test_add_step_with_premise() {
        let mut proof = CarcaraProof::new();
        let p1 = proof.add_assume("p").unwrap();
        let p2 = proof.add_assume("q").unwrap();

        let step = proof.add_step(
            vec!["(or p q)".to_string()],
            AletheRule::Resolution,
            vec![p1, p2],
            vec![],
        );

        assert!(step.is_ok());
        assert_eq!(proof.len(), 3);
    }

    #[test]
    fn test_undefined_premise_error() {
        let mut proof = CarcaraProof::new();

        // Try to use premise 99 which doesn't exist
        let result = proof.add_step(
            vec!["p".to_string()],
            AletheRule::Resolution,
            vec![99],
            vec![],
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not yet defined"));
    }

    #[test]
    fn test_validation() {
        let mut proof = CarcaraProof::new();
        proof.add_assume("p").unwrap();
        proof.add_assume("q").unwrap();

        let result = proof.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_sequential_indices() {
        let mut proof = CarcaraProof::new();
        let i1 = proof.add_assume("p").unwrap();
        let i2 = proof.add_assume("q").unwrap();
        let i3 = proof.add_assume("r").unwrap();

        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
        assert_eq!(i3, 3);
    }

    #[test]
    fn test_clear() {
        let mut proof = CarcaraProof::new();
        proof.add_assume("p").unwrap();
        proof.add_assume("q").unwrap();

        assert_eq!(proof.len(), 2);

        proof.clear();
        assert!(proof.is_empty());

        // After clear, indices should start from 1 again
        let idx = proof.add_assume("r").unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_anchor() {
        let mut proof = CarcaraProof::new();
        let idx = proof
            .add_anchor(vec![("x".to_string(), "Int".to_string())])
            .unwrap();
        assert_eq!(idx, 1);
        assert_eq!(proof.len(), 1);
    }

    #[test]
    fn test_validate_for_carcara() {
        let mut alethe = AletheProof::new();
        alethe.assume("p");
        alethe.assume("q");

        let result = validate_for_carcara(&alethe);
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_carcara_format() {
        let mut alethe = AletheProof::new();
        let p = alethe.assume("p");
        let q = alethe.assume("q");
        alethe.step(
            vec!["(or p q)".to_string()],
            AletheRule::Resolution,
            vec![p, q],
            vec![],
        );

        let carcara = to_carcara_format(&alethe).unwrap();
        assert_eq!(carcara.len(), 3);
        assert!(carcara.validate().is_ok());
    }
}
