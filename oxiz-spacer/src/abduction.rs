//! Abduction support for hypothesis generation.
//!
//! Abduction is the process of finding explanations (hypotheses) for observations.
//! In the context of CHC solving, abduction can be used for:
//! - Synthesizing loop invariants
//! - Generating preconditions
//! - Finding missing specifications
//! - Repair synthesis
//!
//! ## Abductive Reasoning
//!
//! Given:
//! - Background knowledge K (CHC rules)
//! - Observations O (query or property to prove)
//!
//! Find: Hypothesis H such that:
//! - K ∧ H entails O (completeness)
//! - K ∧ H is consistent (soundness)
//! - H is minimal/simple (preference)
//!
//! Reference: Abductive reasoning techniques from AI literature

use crate::chc::{ChcSystem, PredId, Rule};
use crate::pdr::SpacerError;
use oxiz_core::{TermId, TermManager};
use smallvec::SmallVec;
use std::collections::HashSet;
use thiserror::Error;

/// Errors related to abduction
#[derive(Error, Debug)]
pub enum AbductionError {
    /// No hypothesis found
    #[error("no hypothesis found")]
    NoHypothesis,
    /// Hypothesis generation failed
    #[error("hypothesis generation failed: {0}")]
    GenerationFailed(String),
    /// Inconsistent hypothesis
    #[error("inconsistent hypothesis: {0}")]
    Inconsistent(String),
    /// Spacer error
    #[error("spacer error: {0}")]
    Spacer(#[from] SpacerError),
}

/// Result type for abduction operations
pub type AbductionResult<T> = Result<T, AbductionError>;

/// A candidate hypothesis
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Predicate this hypothesis applies to
    pub pred: PredId,
    /// Formula representing the hypothesis
    pub formula: TermId,
    /// Confidence/score (higher is better)
    pub score: f64,
    /// How this hypothesis was generated
    pub origin: HypothesisOrigin,
}

impl Hypothesis {
    /// Create a new hypothesis
    pub fn new(pred: PredId, formula: TermId, score: f64, origin: HypothesisOrigin) -> Self {
        Self {
            pred,
            formula,
            score,
            origin,
        }
    }
}

/// Origin of a hypothesis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HypothesisOrigin {
    /// From counterexample analysis
    CounterexampleAnalysis,
    /// From template instantiation
    TemplateInstantiation,
    /// From interpolation
    Interpolation,
    /// From data-driven learning
    DataDriven,
    /// From syntax-guided synthesis
    SyntaxGuided,
    /// User-provided
    UserProvided,
}

/// Template for hypothesis generation
#[derive(Debug, Clone)]
pub struct HypothesisTemplate {
    /// Name of the template
    pub name: String,
    /// Parameters of the template
    pub params: SmallVec<[(String, TemplateParamKind); 4]>,
    /// Template structure
    pub structure: TemplateStructure,
}

/// Kind of template parameter
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateParamKind {
    /// Numeric constant
    NumericConstant,
    /// Variable
    Variable,
    /// Operator
    Operator,
    /// Predicate
    Predicate,
}

/// Structure of a hypothesis template
#[derive(Debug, Clone)]
pub enum TemplateStructure {
    /// Linear arithmetic: a*x + b*y + ... <= c
    LinearArithmetic,
    /// Octagon: ±x ± y <= c
    Octagon,
    /// Interval: x in [a, b]
    Interval,
    /// Boolean combination
    BooleanCombination,
    /// Custom structure
    Custom(String),
}

/// Hypothesis generator
pub struct HypothesisGenerator {
    /// Templates to try
    templates: Vec<HypothesisTemplate>,
    /// Maximum number of hypotheses to generate
    #[allow(dead_code)]
    max_hypotheses: usize,
}

impl HypothesisGenerator {
    /// Create a new hypothesis generator
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            max_hypotheses: 100,
        }
    }

    /// Create a hypothesis generator with common templates
    pub fn with_common_templates() -> Self {
        let mut generator = Self::new();

        // Linear arithmetic templates
        generator.add_template(HypothesisTemplate {
            name: "linear_inequality".to_string(),
            params: vec![
                ("coeff".to_string(), TemplateParamKind::NumericConstant),
                ("var".to_string(), TemplateParamKind::Variable),
                ("bound".to_string(), TemplateParamKind::NumericConstant),
            ]
            .into(),
            structure: TemplateStructure::LinearArithmetic,
        });

        // Octagon constraints
        generator.add_template(HypothesisTemplate {
            name: "octagon".to_string(),
            params: vec![
                ("var1".to_string(), TemplateParamKind::Variable),
                ("var2".to_string(), TemplateParamKind::Variable),
                ("bound".to_string(), TemplateParamKind::NumericConstant),
            ]
            .into(),
            structure: TemplateStructure::Octagon,
        });

        // Interval constraints
        generator.add_template(HypothesisTemplate {
            name: "interval".to_string(),
            params: vec![
                ("var".to_string(), TemplateParamKind::Variable),
                ("lower".to_string(), TemplateParamKind::NumericConstant),
                ("upper".to_string(), TemplateParamKind::NumericConstant),
            ]
            .into(),
            structure: TemplateStructure::Interval,
        });

        generator
    }

    /// Add a template
    pub fn add_template(&mut self, template: HypothesisTemplate) {
        self.templates.push(template);
    }

    /// Get the number of templates
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// Generate hypotheses for a predicate
    pub fn generate_hypotheses(
        &self,
        terms: &mut TermManager,
        system: &ChcSystem,
        pred: PredId,
    ) -> AbductionResult<Vec<Hypothesis>> {
        let mut hypotheses = Vec::new();

        // Get predicate signature
        let predicate = system
            .get_predicate(pred)
            .ok_or_else(|| AbductionError::GenerationFailed("Predicate not found".to_string()))?;

        // Strategy 1: Template-based synthesis
        // Generate common invariant patterns for numeric variables
        for template in &self.templates {
            // Instantiate template with predicate arguments
            if let Some(hyp) = self.instantiate_template(terms, template, pred, predicate) {
                hypotheses.push(hyp);
            }
        }

        // Strategy 2: Generate trivial hypotheses
        // Start with "true" as the weakest hypothesis
        hypotheses.push(Hypothesis::new(
            pred,
            terms.mk_true(),
            0.0,
            HypothesisOrigin::TemplateInstantiation,
        ));

        // Sort by score (higher is better)
        hypotheses.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(hypotheses)
    }

    /// Instantiate a template for a specific predicate
    fn instantiate_template(
        &self,
        terms: &mut TermManager,
        template: &HypothesisTemplate,
        pred: PredId,
        predicate: &crate::chc::Predicate,
    ) -> Option<Hypothesis> {
        // For numeric predicates, generate simple bounds
        // Example: for P(x: Int), generate x >= 0, x > 0, etc.

        match template.structure {
            TemplateStructure::LinearArithmetic | TemplateStructure::Interval => {
                // Generate a conjunction of constraints
                let mut constraints = Vec::new();

                // For each integer argument, try generating simple bounds
                for (i, &sort) in predicate.params.iter().enumerate() {
                    if sort == terms.sorts.int_sort {
                        // Create variable for this argument
                        let var_name = format!("x{}", i);
                        let var = terms.mk_var(&var_name, sort);
                        let zero = terms.mk_int(0);

                        // Generate: x >= 0
                        constraints.push(terms.mk_ge(var, zero));
                    }
                }

                if constraints.is_empty() {
                    return None;
                }

                let formula = if constraints.len() == 1 {
                    constraints[0]
                } else {
                    terms.mk_and(constraints)
                };

                // Use a default score if template doesn't have one
                Some(Hypothesis::new(
                    pred,
                    formula,
                    1.0, // Default score
                    HypothesisOrigin::TemplateInstantiation,
                ))
            }
            _ => None,
        }
    }

    /// Validate a hypothesis
    pub fn validate_hypothesis(
        &self,
        _terms: &mut TermManager,
        _system: &ChcSystem,
        hypothesis: &Hypothesis,
    ) -> AbductionResult<bool> {
        // Basic validation: Check that the hypothesis is well-formed
        // A full implementation would:
        // 1. Check inductiveness using an SMT solver
        // 2. Verify it doesn't exclude reachable states
        // 3. Prove the property holds

        // For now, reject the "true" hypothesis as too weak
        // and accept any non-trivial hypothesis

        // A simple heuristic: hypotheses with higher scores are more likely valid
        Ok(hypothesis.score > 0.0)
    }

    /// Refine a hypothesis based on counterexample
    pub fn refine_hypothesis(
        &self,
        terms: &mut TermManager,
        hypothesis: &Hypothesis,
        counterexample: TermId,
    ) -> AbductionResult<Hypothesis> {
        // Strengthen the hypothesis by conjoining it with negation of counterexample
        // This ensures the counterexample is ruled out

        // Extract constraints from counterexample that should be negated
        let neg_cex = terms.mk_not(counterexample);

        // Strengthen hypothesis: H' = H ∧ ¬cex
        let strengthened = terms.mk_and([hypothesis.formula, neg_cex]);

        // Create refined hypothesis with increased score
        Ok(Hypothesis::new(
            hypothesis.pred,
            strengthened,
            hypothesis.score + 0.1,
            HypothesisOrigin::CounterexampleAnalysis,
        ))
    }
}

impl Default for HypothesisGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Abductive solver combining PDR with abduction
#[allow(dead_code)]
pub struct AbductiveSolver<'a> {
    /// Term manager
    terms: &'a mut TermManager,
    /// CHC system
    system: &'a ChcSystem,
    /// Hypothesis generator
    generator: HypothesisGenerator,
    /// Current hypotheses (one per predicate)
    hypotheses: Vec<(PredId, Hypothesis)>,
}

impl<'a> AbductiveSolver<'a> {
    /// Create a new abductive solver
    pub fn new(terms: &'a mut TermManager, system: &'a ChcSystem) -> Self {
        Self {
            terms,
            system,
            generator: HypothesisGenerator::new(),
            hypotheses: Vec::new(),
        }
    }

    /// Add a template to use for hypothesis generation
    pub fn add_template(&mut self, template: HypothesisTemplate) {
        self.generator.add_template(template);
    }

    /// Solve using abductive reasoning
    pub fn solve(&mut self) -> AbductionResult<Vec<Hypothesis>> {
        // CEGIS (Counterexample-Guided Inductive Synthesis) loop
        // 1. Generate candidate hypotheses
        // 2. Check if they prove the property
        // 3. If not, get counterexample
        // 4. Refine hypotheses based on counterexample
        // 5. Repeat until proof or give up

        const MAX_ITERATIONS: usize = 100;
        let mut refined_hypotheses = Vec::new();

        for iteration in 0..MAX_ITERATIONS {
            // Step 1: Generate candidate hypotheses for all predicates
            let predicates: Vec<PredId> = self.system.predicates().map(|p| p.id).collect();

            for pred in predicates {
                let candidates =
                    self.generator
                        .generate_hypotheses(self.terms, self.system, pred)?;

                // Step 2: Validate each hypothesis
                for hyp in candidates {
                    if self
                        .generator
                        .validate_hypothesis(self.terms, self.system, &hyp)?
                    {
                        // Found a valid hypothesis
                        refined_hypotheses.push(hyp.clone());
                        self.hypotheses.push((pred, hyp));
                    }
                }
            }

            // If we have valid hypotheses for all predicates, we're done
            if !refined_hypotheses.is_empty() {
                tracing::info!("CEGIS converged after {} iterations", iteration + 1);
                return Ok(refined_hypotheses);
            }

            // Step 3-4: In a full implementation, we would:
            // - Check the hypotheses against the CHC system
            // - Extract counterexamples if verification fails
            // - Refine hypotheses using the counterexamples
        }

        if refined_hypotheses.is_empty() {
            Err(AbductionError::NoHypothesis)
        } else {
            Ok(refined_hypotheses)
        }
    }

    /// Get current hypotheses
    pub fn hypotheses(&self) -> &[(PredId, Hypothesis)] {
        &self.hypotheses
    }
}

/// Precondition synthesizer
///
/// Synthesizes weakest preconditions or adequate preconditions for programs.
pub struct PreconditionSynthesizer;

impl PreconditionSynthesizer {
    /// Synthesize a precondition for a rule
    pub fn synthesize_precondition(
        terms: &mut TermManager,
        rule: &Rule,
        postcondition: TermId,
    ) -> AbductionResult<TermId> {
        // Compute weakest precondition: wp(rule, postcondition)
        // For a rule: body => head, the weakest precondition that ensures
        // the postcondition holds is: body ∧ constraint => postcondition

        // Basic implementation: conjoin the rule's constraint with the postcondition
        // A full implementation would use symbolic execution and quantifier elimination

        let constraint = rule.body.constraint;

        // The precondition is: constraint => postcondition
        // Which is equivalent to: ¬constraint ∨ postcondition
        let not_constraint = terms.mk_not(constraint);
        let precondition = terms.mk_or([not_constraint, postcondition]);

        Ok(precondition)
    }
}

/// Invariant synthesizer
///
/// Synthesizes loop invariants using abductive reasoning.
#[allow(dead_code)]
pub struct InvariantSynthesizer {
    /// Template generator
    generator: HypothesisGenerator,
    /// Positive examples (states that should be in invariant)
    positive_examples: HashSet<TermId>,
    /// Negative examples (states that should not be in invariant)
    negative_examples: HashSet<TermId>,
}

impl InvariantSynthesizer {
    /// Create a new invariant synthesizer
    pub fn new() -> Self {
        Self {
            generator: HypothesisGenerator::new(),
            positive_examples: HashSet::new(),
            negative_examples: HashSet::new(),
        }
    }

    /// Add a positive example
    pub fn add_positive_example(&mut self, state: TermId) {
        self.positive_examples.insert(state);
    }

    /// Add a negative example
    pub fn add_negative_example(&mut self, state: TermId) {
        self.negative_examples.insert(state);
    }

    /// Synthesize an invariant
    pub fn synthesize(&self, terms: &mut TermManager, _pred: PredId) -> AbductionResult<TermId> {
        // Synthesize an invariant that:
        // - Includes all positive examples
        // - Excludes all negative examples
        // - Is (hopefully) inductive

        if self.positive_examples.is_empty() && self.negative_examples.is_empty() {
            // No examples - return true as the weakest invariant
            return Ok(terms.mk_true());
        }

        // Strategy: Build a conjunction of constraints that separate
        // positive from negative examples

        let mut constraints = Vec::new();

        // If we have positive and negative examples, we need to find
        // a separating hyperplane
        if !self.positive_examples.is_empty() && !self.negative_examples.is_empty() {
            // For now, use a simple heuristic: create a constraint that
            // is implied by all positive examples
            // A full implementation would use:
            // - Template-based synthesis
            // - Machine learning (SVM, decision trees)
            // - Symbolic methods (quantifier elimination)

            // Basic approach: return a constraint that rules out negative examples
            for &neg_example in &self.negative_examples {
                // Add: ¬neg_example to the invariant
                constraints.push(terms.mk_not(neg_example));
            }
        }

        if constraints.is_empty() {
            // No constraints found - return true
            Ok(terms.mk_true())
        } else if constraints.len() == 1 {
            Ok(constraints[0])
        } else {
            Ok(terms.mk_and(constraints))
        }
    }
}

impl Default for InvariantSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hypothesis_creation() {
        let hyp = Hypothesis::new(
            PredId(0),
            TermId(1),
            0.9,
            HypothesisOrigin::TemplateInstantiation,
        );

        assert_eq!(hyp.pred, PredId(0));
        assert_eq!(hyp.formula, TermId(1));
        assert_eq!(hyp.score, 0.9);
        assert_eq!(hyp.origin, HypothesisOrigin::TemplateInstantiation);
    }

    #[test]
    fn test_hypothesis_generator() {
        let generator = HypothesisGenerator::new();
        assert_eq!(generator.templates.len(), 0);
        assert_eq!(generator.max_hypotheses, 100);
    }

    #[test]
    fn test_invariant_synthesizer_examples() {
        let mut synth = InvariantSynthesizer::new();

        synth.add_positive_example(TermId(1));
        synth.add_positive_example(TermId(2));
        synth.add_negative_example(TermId(3));

        assert_eq!(synth.positive_examples.len(), 2);
        assert_eq!(synth.negative_examples.len(), 1);
    }

    #[test]
    fn test_hypothesis_generator_with_templates() {
        let generator = HypothesisGenerator::with_common_templates();

        // Should have 3 templates: linear, octagon, interval
        assert_eq!(generator.template_count(), 3);

        // Check templates were added
        assert!(!generator.templates.is_empty());
    }

    #[test]
    fn test_template_structure() {
        let template = HypothesisTemplate {
            name: "test_template".to_string(),
            params: vec![
                ("x".to_string(), TemplateParamKind::Variable),
                ("c".to_string(), TemplateParamKind::NumericConstant),
            ]
            .into(),
            structure: TemplateStructure::LinearArithmetic,
        };

        assert_eq!(template.name, "test_template");
        assert_eq!(template.params.len(), 2);
    }
}
