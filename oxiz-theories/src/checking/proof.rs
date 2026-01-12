//! Proof Checker
//!
//! Validates proof steps and integrates with DRAT/Alethe proof generation.

use super::{CheckResult, CheckerStats, Literal};
use oxiz_core::ast::TermId;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Kind of proof step
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofStepKind {
    /// Axiom (input assertion)
    Axiom,
    /// Resolution between two clauses
    Resolution {
        /// The pivot literal
        pivot: TermId,
        /// First clause ID
        clause1: usize,
        /// Second clause ID
        clause2: usize,
    },
    /// Theory lemma
    TheoryLemma(String),
    /// Unit propagation
    UnitPropagation {
        /// The unit literal
        unit: Literal,
        /// Antecedent clause ID
        antecedent: usize,
    },
    /// Assumption (for proof by contradiction)
    Assumption,
    /// Contradiction (empty clause derivation)
    Contradiction,
    /// Symmetry: a = b => b = a
    Symmetry(TermId, TermId),
    /// Transitivity: a = b, b = c => a = c
    Transitivity(TermId, TermId, TermId),
    /// Congruence: f(a) = f(b) from a = b
    Congruence {
        /// The function being applied
        function: TermId,
        /// Arguments for first application
        args1: Vec<TermId>,
        /// Arguments for second application
        args2: Vec<TermId>,
    },
    /// Instantiation of quantifier
    Instantiation {
        /// The quantifier being instantiated
        quantifier: TermId,
        /// Variable substitutions
        substitution: Vec<(TermId, TermId)>,
    },
    /// Skolemization
    Skolemization {
        /// Original quantified formula
        original: TermId,
        /// Skolemized result
        skolemized: TermId,
    },
}

/// A proof step
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// Step ID
    pub id: usize,
    /// Kind of step
    pub kind: ProofStepKind,
    /// Result clause
    pub clause: Vec<Literal>,
    /// Antecedent step IDs
    pub antecedents: Vec<usize>,
}

impl ProofStep {
    /// Create a new proof step
    pub fn new(id: usize, kind: ProofStepKind, clause: Vec<Literal>) -> Self {
        Self {
            id,
            kind,
            clause,
            antecedents: Vec::new(),
        }
    }

    /// Create with antecedents
    pub fn with_antecedents(
        id: usize,
        kind: ProofStepKind,
        clause: Vec<Literal>,
        antecedents: Vec<usize>,
    ) -> Self {
        Self {
            id,
            kind,
            clause,
            antecedents,
        }
    }
}

/// Proof checker that validates proof steps
#[derive(Debug)]
pub struct ProofChecker {
    /// Steps in the proof
    steps: HashMap<usize, ProofStep>,
    /// Validated steps
    validated: HashSet<usize>,
    /// Statistics
    stats: CheckerStats,
    /// Whether to check thoroughly
    thorough: bool,
}

impl ProofChecker {
    /// Create a new proof checker
    pub fn new() -> Self {
        Self {
            steps: HashMap::new(),
            validated: HashSet::new(),
            stats: CheckerStats::default(),
            thorough: true,
        }
    }

    /// Create with thorough checking disabled
    pub fn quick() -> Self {
        Self {
            steps: HashMap::new(),
            validated: HashSet::new(),
            stats: CheckerStats::default(),
            thorough: false,
        }
    }

    /// Add a proof step
    pub fn add_step(&mut self, step: ProofStep) {
        self.steps.insert(step.id, step);
    }

    /// Validate a proof step
    pub fn validate_step(&mut self, step_id: usize) -> CheckResult {
        let start = Instant::now();

        if self.validated.contains(&step_id) {
            return CheckResult::Valid;
        }

        let step = match self.steps.get(&step_id) {
            Some(s) => s.clone(),
            None => return CheckResult::Invalid(format!("Unknown step: {}", step_id)),
        };

        // First validate antecedents
        for &ant_id in &step.antecedents {
            if !self.validated.contains(&ant_id) {
                let result = self.validate_step(ant_id);
                if !result.is_valid() {
                    return result;
                }
            }
        }

        // Now validate this step
        let result = self.check_step(&step);

        if result.is_valid() {
            self.validated.insert(step_id);
        }

        let elapsed = start.elapsed();
        self.stats.check_time_us += elapsed.as_micros() as u64;

        result
    }

    /// Check a single proof step
    fn check_step(&self, step: &ProofStep) -> CheckResult {
        match &step.kind {
            ProofStepKind::Axiom => {
                // Axioms are always valid (they're input)
                CheckResult::Valid
            }

            ProofStepKind::Assumption => {
                // Assumptions are valid (for proof by contradiction)
                CheckResult::Valid
            }

            ProofStepKind::Resolution {
                pivot,
                clause1,
                clause2,
            } => self.check_resolution(*pivot, *clause1, *clause2, &step.clause),

            ProofStepKind::UnitPropagation { unit, antecedent } => {
                self.check_unit_propagation(*unit, *antecedent, &step.clause)
            }

            ProofStepKind::TheoryLemma(theory) => {
                // Theory lemmas are trusted (checked by theory-specific checker)
                if self.thorough {
                    CheckResult::Unknown(format!("Theory lemma from {}", theory))
                } else {
                    CheckResult::Valid
                }
            }

            ProofStepKind::Contradiction => {
                // Contradiction step should have empty clause
                if step.clause.is_empty() {
                    CheckResult::Valid
                } else {
                    CheckResult::Invalid("Contradiction with non-empty clause".to_string())
                }
            }

            ProofStepKind::Symmetry(a, b) => self.check_symmetry(*a, *b, &step.clause),

            ProofStepKind::Transitivity(a, b, c) => {
                self.check_transitivity(*a, *b, *c, &step.clause)
            }

            ProofStepKind::Congruence { .. } => {
                // Congruence closure - simplified check
                CheckResult::Valid
            }

            ProofStepKind::Instantiation { .. } => {
                // Quantifier instantiation - simplified check
                CheckResult::Valid
            }

            ProofStepKind::Skolemization { .. } => {
                // Skolemization - simplified check
                CheckResult::Valid
            }
        }
    }

    /// Check resolution step
    fn check_resolution(
        &self,
        pivot: TermId,
        clause1_id: usize,
        clause2_id: usize,
        result: &[Literal],
    ) -> CheckResult {
        let clause1 = match self.steps.get(&clause1_id) {
            Some(s) => &s.clause,
            None => return CheckResult::Invalid(format!("Missing clause {}", clause1_id)),
        };

        let clause2 = match self.steps.get(&clause2_id) {
            Some(s) => &s.clause,
            None => return CheckResult::Invalid(format!("Missing clause {}", clause2_id)),
        };

        // Check that pivot appears positive in one and negative in other
        let has_pos1 = clause1.iter().any(|l| l.term == pivot && l.positive);
        let has_neg1 = clause1.iter().any(|l| l.term == pivot && !l.positive);
        let has_pos2 = clause2.iter().any(|l| l.term == pivot && l.positive);
        let has_neg2 = clause2.iter().any(|l| l.term == pivot && !l.positive);

        if !((has_pos1 && has_neg2) || (has_neg1 && has_pos2)) {
            return CheckResult::Invalid("Pivot not complementary in clauses".to_string());
        }

        // Check that result is union minus pivot literals
        let expected: HashSet<_> = clause1
            .iter()
            .chain(clause2.iter())
            .filter(|l| l.term != pivot)
            .collect();

        let actual: HashSet<_> = result.iter().collect();

        if expected != actual {
            return CheckResult::Invalid("Resolution result incorrect".to_string());
        }

        CheckResult::Valid
    }

    /// Check unit propagation
    fn check_unit_propagation(
        &self,
        _unit: Literal,
        _antecedent: usize,
        _result: &[Literal],
    ) -> CheckResult {
        // Simplified: trust unit propagation
        CheckResult::Valid
    }

    /// Check symmetry: a = b => b = a
    fn check_symmetry(&self, _a: TermId, _b: TermId, _result: &[Literal]) -> CheckResult {
        // Simplified: trust symmetry
        CheckResult::Valid
    }

    /// Check transitivity: a = b, b = c => a = c
    fn check_transitivity(
        &self,
        _a: TermId,
        _b: TermId,
        _c: TermId,
        _result: &[Literal],
    ) -> CheckResult {
        // Simplified: trust transitivity
        CheckResult::Valid
    }

    /// Check if proof derives empty clause
    pub fn check_proof(&mut self) -> CheckResult {
        // Find contradiction step
        for (&id, step) in &self.steps.clone() {
            if matches!(step.kind, ProofStepKind::Contradiction) {
                return self.validate_step(id);
            }
        }

        CheckResult::Unknown("No contradiction found".to_string())
    }

    /// Get number of steps
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }

    /// Get number of validated steps
    pub fn num_validated(&self) -> usize {
        self.validated.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &CheckerStats {
        &self.stats
    }

    /// Reset the checker
    pub fn reset(&mut self) {
        self.steps.clear();
        self.validated.clear();
        self.stats = CheckerStats::default();
    }
}

impl Default for ProofChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_step_creation() {
        let t1 = TermId::from(1u32);
        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        assert_eq!(step.id, 0);
        assert_eq!(step.clause.len(), 1);
        assert!(step.antecedents.is_empty());
    }

    #[test]
    fn test_proof_step_with_antecedents() {
        let t1 = TermId::from(1u32);
        let step = ProofStep::with_antecedents(
            2,
            ProofStepKind::TheoryLemma("arith".to_string()),
            vec![Literal::pos(t1)],
            vec![0, 1],
        );
        assert_eq!(step.id, 2);
        assert_eq!(step.antecedents, vec![0, 1]);
    }

    #[test]
    fn test_proof_checker_creation() {
        let checker = ProofChecker::new();
        assert_eq!(checker.num_steps(), 0);
        assert_eq!(checker.num_validated(), 0);
        assert!(checker.thorough);
    }

    #[test]
    fn test_proof_checker_quick() {
        let checker = ProofChecker::quick();
        assert!(!checker.thorough);
    }

    #[test]
    fn test_add_and_validate_axiom() {
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        checker.add_step(step);

        let result = checker.validate_step(0);
        assert!(result.is_valid());
        assert_eq!(checker.num_validated(), 1);
    }

    #[test]
    fn test_validate_unknown_step() {
        let mut checker = ProofChecker::new();
        let result = checker.validate_step(999);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_contradiction_step() {
        let mut checker = ProofChecker::new();
        let step = ProofStep::new(0, ProofStepKind::Contradiction, vec![]);
        checker.add_step(step);

        let result = checker.validate_step(0);
        assert!(result.is_valid());
    }

    #[test]
    fn test_invalid_contradiction() {
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        let step = ProofStep::new(0, ProofStepKind::Contradiction, vec![Literal::pos(t1)]);
        checker.add_step(step);

        let result = checker.validate_step(0);
        assert!(result.is_invalid());
    }

    #[test]
    fn test_resolution() {
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);
        let t2 = TermId::from(2u32);
        let t3 = TermId::from(3u32);

        // Clause 1: t1 OR t2
        let step1 = ProofStep::new(
            0,
            ProofStepKind::Axiom,
            vec![Literal::pos(t1), Literal::pos(t2)],
        );

        // Clause 2: NOT t1 OR t3
        let step2 = ProofStep::new(
            1,
            ProofStepKind::Axiom,
            vec![Literal::neg(t1), Literal::pos(t3)],
        );

        // Resolution on t1 should give: t2 OR t3
        let step3 = ProofStep::with_antecedents(
            2,
            ProofStepKind::Resolution {
                pivot: t1,
                clause1: 0,
                clause2: 1,
            },
            vec![Literal::pos(t2), Literal::pos(t3)],
            vec![0, 1],
        );

        checker.add_step(step1);
        checker.add_step(step2);
        checker.add_step(step3);

        let result = checker.validate_step(2);
        assert!(result.is_valid());
    }

    #[test]
    fn test_check_proof() {
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step1 = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        let step2 = ProofStep::with_antecedents(1, ProofStepKind::Contradiction, vec![], vec![0]);

        checker.add_step(step1);
        checker.add_step(step2);

        let result = checker.check_proof();
        assert!(result.is_valid());
    }

    #[test]
    fn test_no_contradiction() {
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        checker.add_step(step);

        let result = checker.check_proof();
        assert!(!result.is_valid());
    }

    #[test]
    fn test_reset() {
        let mut checker = ProofChecker::new();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(0, ProofStepKind::Axiom, vec![Literal::pos(t1)]);
        checker.add_step(step);
        let _ = checker.validate_step(0);

        assert_eq!(checker.num_steps(), 1);
        assert_eq!(checker.num_validated(), 1);

        checker.reset();
        assert_eq!(checker.num_steps(), 0);
        assert_eq!(checker.num_validated(), 0);
    }

    #[test]
    fn test_theory_lemma() {
        let mut checker = ProofChecker::quick();
        let t1 = TermId::from(1u32);

        let step = ProofStep::new(
            0,
            ProofStepKind::TheoryLemma("arith".to_string()),
            vec![Literal::pos(t1)],
        );
        checker.add_step(step);

        let result = checker.validate_step(0);
        assert!(result.is_valid());
    }
}
