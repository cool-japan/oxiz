//! Proof format conversion utilities.
//!
//! This module provides converters between different proof formats,
//! enabling interoperability between systems.

use crate::alethe::{AletheProof, AletheRule};
use crate::theory::{TheoryProof, TheoryRule, TheoryStepId};
use std::collections::HashMap;

/// Convert a theory proof to Alethe format.
///
/// This maps theory-specific rules to their Alethe equivalents.
#[allow(dead_code)]
#[must_use]
pub fn theory_to_alethe(theory_proof: &TheoryProof) -> AletheProof {
    let mut alethe = AletheProof::new();
    let mut step_map: HashMap<TheoryStepId, u32> = HashMap::new();

    for step in theory_proof.steps() {
        // Map premise IDs to Alethe step indices
        let alethe_premises: Vec<u32> = step
            .premises
            .iter()
            .filter_map(|&p| step_map.get(&p).copied())
            .collect();

        let alethe_rule = map_theory_rule_to_alethe(&step.rule);

        // Check if this is an assumption (axiom with no premises)
        let alethe_idx = if step.premises.is_empty() && is_assumption(&step.rule) {
            alethe.assume(&step.conclusion.0)
        } else {
            // Create a clause from the conclusion
            let clause = vec![step.conclusion.0.clone()];
            let args: Vec<String> = step.args.iter().map(|a| a.0.clone()).collect();

            alethe.step(clause, alethe_rule, alethe_premises, args)
        };

        step_map.insert(step.id, alethe_idx);
    }

    alethe
}

/// Map a theory rule to its Alethe equivalent.
#[allow(dead_code)]
fn map_theory_rule_to_alethe(rule: &TheoryRule) -> AletheRule {
    match rule {
        // EUF rules
        TheoryRule::Refl => AletheRule::Refl,
        TheoryRule::Symm => AletheRule::Symm,
        TheoryRule::Trans => AletheRule::Trans,
        TheoryRule::Cong => AletheRule::Cong,

        // Arithmetic rules
        TheoryRule::LaGeneric => AletheRule::LaGeneric,
        TheoryRule::LaTighten => AletheRule::LaTightening,
        TheoryRule::LaTotality => AletheRule::LaTotality,
        TheoryRule::LaDiseq => AletheRule::LaDisequality,

        // Array rules
        TheoryRule::ArrReadWrite1 => AletheRule::ArrayRowSame,
        TheoryRule::ArrReadWrite2 => AletheRule::ArrayRowDiff,
        TheoryRule::ArrExt => AletheRule::ArrayExt,

        // General rules
        TheoryRule::TheoryConflict => AletheRule::ThLemma,
        TheoryRule::TheoryProp => AletheRule::ThLemma,

        // Default to theory lemma
        _ => AletheRule::ThLemma,
    }
}

/// Check if a rule represents an assumption/axiom.
#[allow(dead_code)]
fn is_assumption(rule: &TheoryRule) -> bool {
    matches!(
        rule,
        TheoryRule::Custom(_)
            | TheoryRule::Refl
            | TheoryRule::ArrReadWrite1
            | TheoryRule::ArrConst
            | TheoryRule::LaTotality
    )
}

/// Statistics about proof conversion.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ConversionStats {
    /// Number of steps in the source proof.
    pub source_steps: usize,
    /// Number of steps in the target proof.
    pub target_steps: usize,
    /// Number of assumptions/axioms.
    pub assumptions: usize,
    /// Number of inference steps.
    pub inferences: usize,
}

#[allow(dead_code)]
impl ConversionStats {
    /// Compute conversion statistics.
    #[must_use]
    pub fn compute(source: &TheoryProof, target: &AletheProof) -> Self {
        let assumptions = source
            .steps()
            .iter()
            .filter(|s| s.premises.is_empty())
            .count();

        Self {
            source_steps: source.len(),
            target_steps: target.len(),
            assumptions,
            inferences: source.len() - assumptions,
        }
    }
}

/// Convert theory proof to Alethe with statistics.
#[allow(dead_code)]
#[must_use]
pub fn theory_to_alethe_with_stats(theory_proof: &TheoryProof) -> (AletheProof, ConversionStats) {
    let alethe = theory_to_alethe(theory_proof);
    let stats = ConversionStats::compute(theory_proof, &alethe);
    (alethe, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::ProofTerm;

    #[test]
    fn test_theory_to_alethe_simple() {
        let mut theory = TheoryProof::new();
        theory.refl("x");

        let alethe = theory_to_alethe(&theory);
        assert_eq!(alethe.len(), 1);
    }

    #[test]
    fn test_theory_to_alethe_transitivity() {
        let mut theory = TheoryProof::new();

        let s1 = theory.add_axiom(TheoryRule::Custom("assert".into()), "(= a b)");
        let s2 = theory.add_axiom(TheoryRule::Custom("assert".into()), "(= b c)");
        theory.trans(s1, s2, "a", "c");

        let alethe = theory_to_alethe(&theory);
        assert_eq!(alethe.len(), 3);
    }

    #[test]
    fn test_theory_to_alethe_arithmetic() {
        let mut theory = TheoryProof::new();

        let s1 = theory.add_axiom(TheoryRule::Custom("bound".into()), "(>= x 10)");
        let s2 = theory.add_axiom(TheoryRule::Custom("bound".into()), "(<= x 5)");
        theory.farkas(
            vec![s1, s2],
            &[ProofTerm("1".into()), ProofTerm("1".into())],
        );

        let alethe = theory_to_alethe(&theory);
        assert_eq!(alethe.len(), 3);
    }

    #[test]
    fn test_conversion_stats() {
        let mut theory = TheoryProof::new();

        theory.add_axiom(TheoryRule::Custom("assert".into()), "(= x y)");
        theory.add_axiom(TheoryRule::Custom("assert".into()), "(= y z)");
        theory.refl("w");

        let alethe = theory_to_alethe(&theory);
        let stats = ConversionStats::compute(&theory, &alethe);

        assert_eq!(stats.source_steps, 3);
        assert_eq!(stats.assumptions, 3);
        assert_eq!(stats.inferences, 0);
    }

    #[test]
    fn test_theory_to_alethe_with_stats() {
        let mut theory = TheoryProof::new();

        let s1 = theory.add_axiom(TheoryRule::Custom("assert".into()), "(= a b)");
        let s2 = theory.add_axiom(TheoryRule::Custom("assert".into()), "(= b c)");
        theory.trans(s1, s2, "a", "c");

        let (alethe, stats) = theory_to_alethe_with_stats(&theory);

        assert_eq!(alethe.len(), 3);
        assert_eq!(stats.source_steps, 3);
        assert_eq!(stats.assumptions, 2);
        assert_eq!(stats.inferences, 1);
    }

    #[test]
    fn test_map_theory_rule_to_alethe() {
        assert_eq!(
            map_theory_rule_to_alethe(&TheoryRule::Refl),
            AletheRule::Refl
        );
        assert_eq!(
            map_theory_rule_to_alethe(&TheoryRule::Trans),
            AletheRule::Trans
        );
        assert_eq!(
            map_theory_rule_to_alethe(&TheoryRule::LaGeneric),
            AletheRule::LaGeneric
        );
        assert_eq!(
            map_theory_rule_to_alethe(&TheoryRule::ArrReadWrite1),
            AletheRule::ArrayRowSame
        );
    }

    #[test]
    fn test_is_assumption() {
        assert!(is_assumption(&TheoryRule::Refl));
        assert!(is_assumption(&TheoryRule::Custom("test".into())));
        assert!(!is_assumption(&TheoryRule::Trans));
        assert!(!is_assumption(&TheoryRule::Cong));
    }
}
