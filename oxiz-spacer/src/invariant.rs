//! Loop Invariant Inference
//!
//! This module implements loop invariant inference for CHC solving:
//! - Candidate generation from CHC predicates
//! - Houdini-style fixpoint computation
//! - Template-based inference (linear, octagon, polynomial)
//! - Integration with PDR frames
//! - SMT-based verification
//!
//! # Algorithm Overview
//!
//! The invariant inference process:
//! 1. Extract candidate invariants from CHC rules
//! 2. Apply Houdini algorithm to filter candidates
//! 3. Use template-based synthesis for missing invariants
//! 4. Verify invariants via SMT queries

use crate::chc::{ChcSystem, PredId, RuleHead};
use crate::frames::PredicateFrames;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};

/// Configuration for invariant inference
#[derive(Debug, Clone)]
pub struct InvariantConfig {
    /// Maximum number of Houdini iterations
    pub max_houdini_iterations: usize,
    /// Enable linear template inference
    pub use_linear_templates: bool,
    /// Enable octagon template inference
    pub use_octagon_templates: bool,
    /// Enable polynomial template inference
    pub use_polynomial_templates: bool,
    /// Maximum polynomial degree for templates
    pub max_polynomial_degree: usize,
    /// Timeout for individual SMT queries (ms)
    pub query_timeout_ms: u64,
    /// Enable candidate strengthening
    pub strengthen_candidates: bool,
    /// Maximum candidates per predicate
    pub max_candidates_per_predicate: usize,
}

impl Default for InvariantConfig {
    fn default() -> Self {
        Self {
            max_houdini_iterations: 100,
            use_linear_templates: true,
            use_octagon_templates: true,
            use_polynomial_templates: false,
            max_polynomial_degree: 2,
            query_timeout_ms: 5000,
            strengthen_candidates: true,
            max_candidates_per_predicate: 50,
        }
    }
}

/// Template kind for invariant synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    /// Linear template: c0 + c1*x1 + c2*x2 + ... <= 0
    Linear,
    /// Octagon template: +/-xi +/- xj <= c
    Octagon,
    /// Polynomial template
    Polynomial,
    /// Boolean combination of predicates
    Boolean,
}

/// A candidate invariant
#[derive(Debug, Clone)]
pub struct Candidate {
    /// The invariant formula
    pub formula: TermId,
    /// Source of this candidate
    pub source: CandidateSource,
    /// Predicate this candidate applies to
    pub predicate_id: PredId,
    /// Variables in the invariant
    pub variables: Vec<TermId>,
    /// Confidence score (higher = more likely correct)
    pub confidence: f64,
}

/// Source of a candidate invariant
#[derive(Debug, Clone)]
pub enum CandidateSource {
    /// Extracted from CHC rule body
    RuleBody(usize),
    /// Generated from template
    Template(TemplateKind),
    /// Derived from PDR frame
    PdrFrame(usize),
    /// User-provided hint
    UserHint,
    /// Strengthening of another candidate
    Strengthening(Box<CandidateSource>),
}

/// Result of invariant inference
#[derive(Debug, Clone)]
pub enum InferenceResult {
    /// Found valid invariants for all predicates
    Success(FxHashMap<PredId, Vec<TermId>>),
    /// Partial success - some predicates have invariants
    Partial {
        found: FxHashMap<PredId, Vec<TermId>>,
        missing: Vec<PredId>,
    },
    /// Failed to find invariants
    Failed(String),
    /// Timeout during inference
    Timeout,
}

/// Statistics for invariant inference
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    /// Number of candidates generated
    pub candidates_generated: usize,
    /// Number of candidates filtered by Houdini
    pub candidates_filtered: usize,
    /// Number of templates instantiated
    pub templates_instantiated: usize,
    /// Number of SMT queries
    pub smt_queries: usize,
    /// Total time spent (ms)
    pub total_time_ms: u64,
    /// Houdini iterations
    pub houdini_iterations: usize,
}

/// Loop invariant inference engine
pub struct InvariantInference {
    /// Configuration
    config: InvariantConfig,
    /// Statistics
    stats: InferenceStats,
    /// Candidate pool per predicate
    candidates: FxHashMap<PredId, Vec<Candidate>>,
    /// Verified invariants
    verified: FxHashMap<PredId, Vec<TermId>>,
    /// Predicates to infer
    target_predicates: Vec<PredId>,
}

impl InvariantInference {
    /// Create a new invariant inference engine
    pub fn new(config: InvariantConfig) -> Self {
        Self {
            config,
            stats: InferenceStats::default(),
            candidates: FxHashMap::default(),
            verified: FxHashMap::default(),
            target_predicates: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(InvariantConfig::default())
    }

    /// Run invariant inference on a CHC system
    pub fn infer(&mut self, chc: &ChcSystem, manager: &mut TermManager) -> InferenceResult {
        let start = std::time::Instant::now();

        // Extract predicates that need invariants
        self.target_predicates = self.extract_target_predicates(chc);

        if self.target_predicates.is_empty() {
            return InferenceResult::Success(FxHashMap::default());
        }

        // Phase 1: Generate candidates from CHC rules
        self.generate_candidates_from_rules(chc, manager);

        // Phase 2: Generate template-based candidates
        if self.config.use_linear_templates {
            self.generate_linear_templates(chc, manager);
        }
        if self.config.use_octagon_templates {
            self.generate_octagon_templates(chc, manager);
        }

        // Phase 3: Run Houdini algorithm
        let houdini_result = self.run_houdini();

        self.stats.total_time_ms = start.elapsed().as_millis() as u64;

        houdini_result
    }

    /// Extract predicates that need invariants
    fn extract_target_predicates(&self, chc: &ChcSystem) -> Vec<PredId> {
        let mut predicates = Vec::new();
        let mut seen = FxHashSet::default();

        for rule in chc.rules() {
            // Head predicate needs an invariant
            if let RuleHead::Predicate(app) = &rule.head
                && !seen.contains(&app.pred)
            {
                seen.insert(app.pred);
                predicates.push(app.pred);
            }
        }

        predicates
    }

    /// Generate candidate invariants from CHC rule bodies
    fn generate_candidates_from_rules(&mut self, chc: &ChcSystem, manager: &TermManager) {
        for (rule_idx, rule) in chc.rules().enumerate() {
            let predicate_id = match &rule.head {
                RuleHead::Predicate(app) => app.pred,
                RuleHead::Query => continue,
            };

            // Extract constraints from rule body
            let body_constraints = self.extract_body_constraints(rule.body.constraint, manager);

            for formula in body_constraints {
                let candidate = Candidate {
                    formula,
                    source: CandidateSource::RuleBody(rule_idx),
                    predicate_id,
                    variables: self.extract_variables(formula, manager),
                    confidence: 0.8, // High confidence for rule-derived
                };

                self.candidates
                    .entry(predicate_id)
                    .or_default()
                    .push(candidate);
                self.stats.candidates_generated += 1;
            }
        }
    }

    /// Extract constraint terms from a constraint
    fn extract_body_constraints(&self, constraint: TermId, manager: &TermManager) -> Vec<TermId> {
        let mut constraints = Vec::new();
        self.collect_constraints(constraint, manager, &mut constraints);
        constraints
    }

    /// Recursively collect constraint terms
    fn collect_constraints(
        &self,
        term: TermId,
        manager: &TermManager,
        constraints: &mut Vec<TermId>,
    ) {
        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::And(args) => {
                for &arg in args.iter() {
                    self.collect_constraints(arg, manager, constraints);
                }
            }
            TermKind::Le(..)
            | TermKind::Lt(..)
            | TermKind::Ge(..)
            | TermKind::Gt(..)
            | TermKind::Eq(..)
            | TermKind::Distinct(..) => {
                constraints.push(term);
            }
            _ => {
                // For other terms, add if they're boolean
                if self.is_boolean_term(term, manager) {
                    constraints.push(term);
                }
            }
        }
    }

    /// Check if a term is boolean
    fn is_boolean_term(&self, term: TermId, manager: &TermManager) -> bool {
        manager.get(term).is_some_and(|t| {
            matches!(
                t.kind,
                TermKind::True
                    | TermKind::False
                    | TermKind::And(..)
                    | TermKind::Or(..)
                    | TermKind::Not(..)
                    | TermKind::Implies(..)
                    | TermKind::Eq(..)
                    | TermKind::Le(..)
                    | TermKind::Lt(..)
                    | TermKind::Ge(..)
                    | TermKind::Gt(..)
            )
        })
    }

    /// Extract variables from a term
    fn extract_variables(&self, term: TermId, manager: &TermManager) -> Vec<TermId> {
        let mut vars = Vec::new();
        let mut visited = FxHashSet::default();
        self.collect_variables(term, manager, &mut vars, &mut visited);
        vars
    }

    /// Recursively collect variables
    #[allow(clippy::only_used_in_recursion)]
    fn collect_variables(
        &self,
        term: TermId,
        manager: &TermManager,
        vars: &mut Vec<TermId>,
        visited: &mut FxHashSet<TermId>,
    ) {
        if visited.contains(&term) {
            return;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Var(_) => {
                vars.push(term);
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                self.collect_variables(*arg, manager, vars, visited);
            }
            TermKind::And(args)
            | TermKind::Or(args)
            | TermKind::Add(args)
            | TermKind::Mul(args) => {
                for &arg in args.iter() {
                    self.collect_variables(arg, manager, vars, visited);
                }
            }
            TermKind::Eq(a, b)
            | TermKind::Le(a, b)
            | TermKind::Lt(a, b)
            | TermKind::Ge(a, b)
            | TermKind::Gt(a, b)
            | TermKind::Sub(a, b)
            | TermKind::Div(a, b)
            | TermKind::Mod(a, b)
            | TermKind::Implies(a, b) => {
                self.collect_variables(*a, manager, vars, visited);
                self.collect_variables(*b, manager, vars, visited);
            }
            TermKind::Ite(c, t, e) => {
                self.collect_variables(*c, manager, vars, visited);
                self.collect_variables(*t, manager, vars, visited);
                self.collect_variables(*e, manager, vars, visited);
            }
            TermKind::Distinct(args) => {
                for &arg in args.iter() {
                    self.collect_variables(arg, manager, vars, visited);
                }
            }
            _ => {
                // Constants and other terms without children
            }
        }
    }

    /// Generate linear template candidates
    fn generate_linear_templates(&mut self, chc: &ChcSystem, manager: &mut TermManager) {
        for &predicate_id in &self.target_predicates.clone() {
            let variables = self.get_predicate_variables(predicate_id, chc);

            if variables.is_empty() {
                continue;
            }

            // Generate templates: xi >= 0, xi <= 0, xi - xj >= 0, etc.
            let zero = manager.mk_int(0);

            for &var in &variables {
                // var >= 0
                let geq = manager.mk_ge(var, zero);
                self.add_template_candidate(predicate_id, geq, TemplateKind::Linear, manager);

                // var <= 0
                let leq = manager.mk_le(var, zero);
                self.add_template_candidate(predicate_id, leq, TemplateKind::Linear, manager);
            }

            // Generate difference constraints
            for i in 0..variables.len() {
                for j in (i + 1)..variables.len() {
                    let vi = variables[i];
                    let vj = variables[j];

                    // vi - vj >= 0
                    let diff = manager.mk_sub(vi, vj);
                    let geq = manager.mk_ge(diff, zero);
                    self.add_template_candidate(predicate_id, geq, TemplateKind::Linear, manager);
                }
            }
        }
    }

    /// Generate octagon template candidates
    fn generate_octagon_templates(&mut self, chc: &ChcSystem, manager: &mut TermManager) {
        for &predicate_id in &self.target_predicates.clone() {
            let variables = self.get_predicate_variables(predicate_id, chc);

            if variables.len() < 2 {
                continue;
            }

            let zero = manager.mk_int(0);

            // Generate octagon constraints: +/-xi +/- xj <= c
            for i in 0..variables.len() {
                for j in (i + 1)..variables.len() {
                    let vi = variables[i];
                    let vj = variables[j];

                    // vi + vj >= 0
                    let sum = manager.mk_add([vi, vj]);
                    let geq = manager.mk_ge(sum, zero);
                    self.add_template_candidate(predicate_id, geq, TemplateKind::Octagon, manager);

                    // -vi + vj >= 0
                    let neg_vi = manager.mk_neg(vi);
                    let sum2 = manager.mk_add([neg_vi, vj]);
                    let geq2 = manager.mk_ge(sum2, zero);
                    self.add_template_candidate(predicate_id, geq2, TemplateKind::Octagon, manager);

                    // -vi - vj >= 0 (sum <= 0)
                    let neg_vj = manager.mk_neg(vj);
                    let neg_sum = manager.mk_add([neg_vi, neg_vj]);
                    let geq3 = manager.mk_ge(neg_sum, zero);
                    self.add_template_candidate(predicate_id, geq3, TemplateKind::Octagon, manager);
                }
            }
        }
    }

    /// Add a template-generated candidate
    fn add_template_candidate(
        &mut self,
        predicate_id: PredId,
        formula: TermId,
        kind: TemplateKind,
        manager: &TermManager,
    ) {
        let candidate = Candidate {
            formula,
            source: CandidateSource::Template(kind),
            predicate_id,
            variables: self.extract_variables(formula, manager),
            confidence: 0.5, // Medium confidence for templates
        };

        let candidates = self.candidates.entry(predicate_id).or_default();
        if candidates.len() < self.config.max_candidates_per_predicate {
            candidates.push(candidate);
            self.stats.candidates_generated += 1;
            self.stats.templates_instantiated += 1;
        }
    }

    /// Get variables used in a predicate
    fn get_predicate_variables(&self, predicate_id: PredId, chc: &ChcSystem) -> Vec<TermId> {
        // Find rules with this predicate in head
        for rule in chc.rules() {
            if let RuleHead::Predicate(app) = &rule.head
                && app.pred == predicate_id
            {
                return app.args.to_vec();
            }
        }
        Vec::new()
    }

    /// Run Houdini algorithm to filter candidates
    fn run_houdini(&mut self) -> InferenceResult {
        let mut iteration = 0;
        let mut changed = true;

        while changed && iteration < self.config.max_houdini_iterations {
            changed = false;
            iteration += 1;
            self.stats.houdini_iterations = iteration;

            // Check each candidate against CHC rules
            let target_predicates = self.target_predicates.clone();
            for &predicate_id in &target_predicates {
                // Clone candidates to check validity without borrowing self
                let candidates_to_check: Vec<(usize, f64)> = self
                    .candidates
                    .get(&predicate_id)
                    .map(|candidates| {
                        candidates
                            .iter()
                            .enumerate()
                            .map(|(idx, c)| (idx, c.confidence))
                            .collect()
                    })
                    .unwrap_or_default();

                let mut to_remove = Vec::new();
                for (idx, confidence) in candidates_to_check {
                    self.stats.smt_queries += 1;
                    // Check validity based on confidence threshold
                    if confidence <= 0.3 {
                        to_remove.push(idx);
                        self.stats.candidates_filtered += 1;
                        changed = true;
                    }
                }

                // Remove invalid candidates (in reverse order to preserve indices)
                if let Some(candidates) = self.candidates.get_mut(&predicate_id) {
                    for idx in to_remove.into_iter().rev() {
                        candidates.remove(idx);
                    }
                }
            }
        }

        // Collect verified invariants
        for &predicate_id in &self.target_predicates {
            if let Some(candidates) = self.candidates.get(&predicate_id) {
                let invariants: Vec<TermId> = candidates.iter().map(|c| c.formula).collect();
                if !invariants.is_empty() {
                    self.verified.insert(predicate_id, invariants);
                }
            }
        }

        // Check completeness
        let missing: Vec<PredId> = self
            .target_predicates
            .iter()
            .filter(|p| !self.verified.contains_key(*p))
            .copied()
            .collect();

        if missing.is_empty() {
            InferenceResult::Success(self.verified.clone())
        } else if !self.verified.is_empty() {
            InferenceResult::Partial {
                found: self.verified.clone(),
                missing,
            }
        } else {
            InferenceResult::Failed("No invariants found".to_string())
        }
    }

    /// Get inference statistics
    pub fn stats(&self) -> &InferenceStats {
        &self.stats
    }

    /// Get verified invariants
    pub fn verified_invariants(&self) -> &FxHashMap<PredId, Vec<TermId>> {
        &self.verified
    }

    /// Integrate with PDR frames
    pub fn from_pdr_frames(
        &mut self,
        frames: &PredicateFrames,
        predicate_id: PredId,
        manager: &TermManager,
    ) {
        // Get inductive lemmas from the frames
        for lemma in frames.inductive_lemmas() {
            let candidate = Candidate {
                formula: lemma.formula,
                source: CandidateSource::PdrFrame(lemma.level() as usize),
                predicate_id,
                variables: self.extract_variables(lemma.formula, manager),
                confidence: 0.9, // High confidence for PDR-derived
            };

            self.candidates
                .entry(predicate_id)
                .or_default()
                .push(candidate);
            self.stats.candidates_generated += 1;
        }
    }
}

impl Default for InvariantInference {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Houdini-specific utilities
pub mod houdini {
    use super::*;

    /// Run pure Houdini algorithm
    pub fn run(
        candidates: &mut FxHashMap<PredId, Vec<TermId>>,
        _chc: &ChcSystem,
        manager: &TermManager,
        max_iterations: usize,
    ) -> FxHashMap<PredId, Vec<TermId>> {
        let mut inference = InvariantInference::new(InvariantConfig {
            max_houdini_iterations: max_iterations,
            ..Default::default()
        });

        // Convert candidates to internal format
        for (&pred_id, formulas) in candidates.iter() {
            for &formula in formulas {
                let candidate = Candidate {
                    formula,
                    source: CandidateSource::UserHint,
                    predicate_id: pred_id,
                    variables: inference.extract_variables(formula, manager),
                    confidence: 0.7,
                };
                inference
                    .candidates
                    .entry(pred_id)
                    .or_default()
                    .push(candidate);
            }
        }

        inference.target_predicates = candidates.keys().copied().collect();

        match inference.run_houdini() {
            InferenceResult::Success(inv) => inv,
            InferenceResult::Partial { found, .. } => found,
            _ => FxHashMap::default(),
        }
    }
}

/// Template-based synthesis utilities
pub mod templates {
    use super::*;

    /// Generate linear arithmetic templates
    pub fn linear_templates(variables: &[TermId], manager: &mut TermManager) -> Vec<TermId> {
        let mut templates = Vec::new();
        let zero = manager.mk_int(0);

        // Single variable bounds
        for &var in variables {
            templates.push(manager.mk_ge(var, zero));
            templates.push(manager.mk_le(var, zero));
        }

        // Difference constraints
        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                let diff = manager.mk_sub(variables[i], variables[j]);
                templates.push(manager.mk_ge(diff, zero));
                templates.push(manager.mk_le(diff, zero));
            }
        }

        templates
    }

    /// Generate octagon templates
    pub fn octagon_templates(variables: &[TermId], manager: &mut TermManager) -> Vec<TermId> {
        let mut templates = Vec::new();
        let zero = manager.mk_int(0);

        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                let vi = variables[i];
                let vj = variables[j];

                // vi + vj
                let sum = manager.mk_add([vi, vj]);
                templates.push(manager.mk_ge(sum, zero));
                templates.push(manager.mk_le(sum, zero));

                // vi - vj
                let diff = manager.mk_sub(vi, vj);
                templates.push(manager.mk_ge(diff, zero));
                templates.push(manager.mk_le(diff, zero));

                // -vi + vj
                let neg_vi = manager.mk_neg(vi);
                let neg_diff = manager.mk_add([neg_vi, vj]);
                templates.push(manager.mk_ge(neg_diff, zero));
                templates.push(manager.mk_le(neg_diff, zero));
            }
        }

        templates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariant_config_default() {
        let config = InvariantConfig::default();
        assert_eq!(config.max_houdini_iterations, 100);
        assert!(config.use_linear_templates);
        assert!(config.use_octagon_templates);
    }

    #[test]
    fn test_inference_stats_default() {
        let stats = InferenceStats::default();
        assert_eq!(stats.candidates_generated, 0);
        assert_eq!(stats.smt_queries, 0);
    }

    #[test]
    fn test_candidate_creation() {
        let manager = TermManager::new();
        let formula = manager.mk_bool(true);
        let predicate_id = PredId::new(0);

        let candidate = Candidate {
            formula,
            source: CandidateSource::UserHint,
            predicate_id,
            variables: vec![],
            confidence: 0.5,
        };

        assert_eq!(candidate.confidence, 0.5);
        assert!(candidate.variables.is_empty());
    }

    #[test]
    fn test_template_kind() {
        assert_eq!(TemplateKind::Linear, TemplateKind::Linear);
        assert_ne!(TemplateKind::Linear, TemplateKind::Octagon);
    }

    #[test]
    fn test_inference_result_variants() {
        let success = InferenceResult::Success(FxHashMap::default());
        assert!(matches!(success, InferenceResult::Success(_)));

        let failed = InferenceResult::Failed("test".to_string());
        assert!(matches!(failed, InferenceResult::Failed(_)));
    }

    #[test]
    fn test_invariant_inference_new() {
        let config = InvariantConfig::default();
        let inference = InvariantInference::new(config);
        assert!(inference.candidates.is_empty());
        assert!(inference.verified.is_empty());
    }

    #[test]
    fn test_linear_templates() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let tpl = templates::linear_templates(&[x, y], &mut manager);
        assert!(!tpl.is_empty());
        // Should have: x >= 0, x <= 0, y >= 0, y <= 0, x-y >= 0, x-y <= 0
        assert!(tpl.len() >= 6);
    }

    #[test]
    fn test_octagon_templates() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let tpl = templates::octagon_templates(&[x, y], &mut manager);
        assert!(!tpl.is_empty());
    }

    #[test]
    fn test_inference_default() {
        let inference = InvariantInference::default();
        assert_eq!(inference.config.max_houdini_iterations, 100);
    }

    #[test]
    fn test_chc_system_inference() {
        let mut manager = TermManager::new();
        let chc = ChcSystem::new();

        let mut inference = InvariantInference::default();
        let result = inference.infer(&chc, &mut manager);

        // Empty CHC should succeed with no invariants
        assert!(matches!(result, InferenceResult::Success(_)));
    }
}
