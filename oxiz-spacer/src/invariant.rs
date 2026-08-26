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
use crate::smt::{SmtSolver, canon_cur_vars};
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
        let start = oxiz_time::Instant::now();

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

        // Phase 3: Run Houdini algorithm (with real SMT inductiveness checks)
        let houdini_result = self.run_houdini(chc, manager);

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

    /// Generate candidate invariants from CHC rule bodies.
    ///
    /// Only constraints that are *pure state predicates* over the head
    /// predicate's arguments are kept (constraints mentioning auxiliary /
    /// next-state variables cannot be invariants of the predicate).  Accepted
    /// constraints are normalized into the predicate's canonical current-state
    /// variables so that Houdini's inductiveness queries can rename them
    /// consistently across rules.
    fn generate_candidates_from_rules(&mut self, chc: &ChcSystem, manager: &mut TermManager) {
        for (rule_idx, rule) in chc.rules().enumerate() {
            let head_args: Vec<TermId> = match &rule.head {
                RuleHead::Predicate(app) => app.args.to_vec(),
                RuleHead::Query => continue,
            };
            let predicate_id = match &rule.head {
                RuleHead::Predicate(app) => app.pred,
                RuleHead::Query => continue,
            };

            let canon = canon_cur_vars(manager, chc, predicate_id);
            if canon.len() != head_args.len() {
                continue;
            }
            let head_map: FxHashMap<TermId, TermId> = head_args
                .iter()
                .zip(canon.iter())
                .map(|(&a, &c)| (a, c))
                .collect();

            // Extract constraints from rule body
            let body_constraints = self.extract_body_constraints(rule.body.constraint, manager);

            for formula in body_constraints {
                // Keep only constraints whose variables are all head arguments.
                let vars = self.extract_variables(formula, manager);
                if !vars.iter().all(|v| head_args.contains(v)) {
                    continue;
                }
                let canon_formula = manager.substitute(formula, &head_map);
                let candidate = Candidate {
                    formula: canon_formula,
                    source: CandidateSource::RuleBody(rule_idx),
                    predicate_id,
                    variables: canon.iter().copied().collect(),
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

    /// Collect the constraint atoms of a formula, flattening nested `And`s.
    ///
    /// The `And` tree comes from parsed CHC bodies, so its depth is
    /// unbounded; this walks it with an explicit heap stack
    /// ([`crate::walk::flatten_conjuncts`]) instead of recursing. The
    /// classification of each non-`And` conjunct is unchanged: arithmetic
    /// comparisons and `Distinct` are constraints, anything else is kept
    /// only if it is a Boolean term.
    fn collect_constraints(
        &self,
        term: TermId,
        manager: &TermManager,
        constraints: &mut Vec<TermId>,
    ) {
        for conjunct in crate::walk::flatten_conjuncts(manager, term) {
            let Some(t) = manager.get(conjunct) else {
                continue;
            };
            match &t.kind {
                TermKind::Le(..)
                | TermKind::Lt(..)
                | TermKind::Ge(..)
                | TermKind::Gt(..)
                | TermKind::Eq(..)
                | TermKind::Distinct(..) => {
                    constraints.push(conjunct);
                }
                _ => {
                    // For other terms, add if they're boolean
                    if self.is_boolean_term(conjunct, manager) {
                        constraints.push(conjunct);
                    }
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

    /// Collect the distinct variables of a term, in first-occurrence order.
    ///
    /// The visited set this already carried made the walk linear in DAG
    /// size, but it was still native recursion, so nesting depth alone
    /// could overflow the stack; it is an explicit-stack walk now (see
    /// [`crate::walk`]). Descent is uniform over every `TermKind` child
    /// rather than the previous enumeration, whose `_ => {}` arm dropped
    /// variables occurring under `Apply`, `Select`/`Store`, `Xor`, `Let`,
    /// quantifier bodies and every bitvector/string operator — an
    /// understated variable set makes an inferred invariant candidate refer
    /// to fewer variables than it actually constrains.
    fn collect_variables(
        &self,
        term: TermId,
        manager: &TermManager,
        vars: &mut Vec<TermId>,
        visited: &mut FxHashSet<TermId>,
    ) {
        for var in crate::walk::collect_vars(manager, term) {
            if visited.insert(var) {
                vars.push(var);
            }
        }
    }

    /// Generate linear template candidates
    fn generate_linear_templates(&mut self, chc: &ChcSystem, manager: &mut TermManager) {
        for &predicate_id in &self.target_predicates.clone() {
            let variables = self.get_predicate_variables(predicate_id, chc, manager);

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
            let variables = self.get_predicate_variables(predicate_id, chc, manager);

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

    /// Get the canonical current-state variables of a predicate.
    ///
    /// Templates are built over these canonical variables so that every
    /// candidate for a predicate lives in a single, consistent namespace that
    /// Houdini can rename onto each rule's argument terms.
    fn get_predicate_variables(
        &self,
        predicate_id: PredId,
        chc: &ChcSystem,
        manager: &mut TermManager,
    ) -> Vec<TermId> {
        canon_cur_vars(manager, chc, predicate_id).to_vec()
    }

    /// Run the Houdini algorithm to filter candidates down to an inductive
    /// subset via **real SMT inductiveness queries**.
    ///
    /// A candidate `c` for predicate `P` is dropped whenever some rule
    /// `Q1(a1) ∧ … ∧ Qn(an) ∧ constraint ⇒ P(head)` fails consecution, i.e.
    /// `⋀ᵢ (current candidates of Qᵢ, renamed to aᵢ) ∧ constraint ∧ ¬c[head]`
    /// is satisfiable (or the solver cannot prove it unsatisfiable).  The
    /// process iterates to a greatest fixpoint, so the surviving set is
    /// genuinely inductive — contradictory guesses such as `x ≥ 0` and
    /// `x ≤ 0` are eliminated instead of being reported as invariants.
    fn run_houdini(&mut self, chc: &ChcSystem, manager: &mut TermManager) -> InferenceResult {
        let mut iteration = 0;
        let mut changed = true;

        while changed && iteration < self.config.max_houdini_iterations {
            changed = false;
            iteration += 1;
            self.stats.houdini_iterations = iteration;

            // Snapshot the current candidate formulas per predicate; these form
            // the antecedent context (the assumed invariants of body predicates)
            // for this pass.
            let snapshot: FxHashMap<PredId, Vec<TermId>> = self
                .candidates
                .iter()
                .map(|(&p, cands)| (p, cands.iter().map(|c| c.formula).collect()))
                .collect();

            let target_predicates = self.target_predicates.clone();
            for &predicate_id in &target_predicates {
                let candidate_formulas: Vec<TermId> =
                    snapshot.get(&predicate_id).cloned().unwrap_or_default();

                let mut to_remove = Vec::new();
                for (idx, &formula) in candidate_formulas.iter().enumerate() {
                    self.stats.smt_queries += 1;
                    if !Self::candidate_is_inductive(chc, manager, predicate_id, formula, &snapshot)
                    {
                        to_remove.push(idx);
                        self.stats.candidates_filtered += 1;
                        changed = true;
                    }
                }

                // Remove violated candidates (reverse order to preserve indices).
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

    /// Check whether `candidate` (over `pred`'s canonical current-state
    /// variables) is preserved by **every** rule whose head is `pred`, given
    /// the assumed candidate invariants in `snapshot`.
    ///
    /// Returns `true` only if consecution is *provably* UNSAT for all such
    /// rules.  A SAT result (real violation) or a solver `Unknown` both yield
    /// `false` — we never keep a candidate we could not prove inductive, which
    /// keeps the returned invariant set sound.
    fn candidate_is_inductive(
        chc: &ChcSystem,
        manager: &mut TermManager,
        pred: PredId,
        candidate: TermId,
        snapshot: &FxHashMap<PredId, Vec<TermId>>,
    ) -> bool {
        for rule in chc.rules() {
            let head_app = match &rule.head {
                RuleHead::Predicate(app) if app.pred == pred => app,
                _ => continue,
            };

            // Antecedent: assumed invariants of each body predicate (renamed to
            // the rule's argument terms) plus the rule's own constraint.
            let mut antecedent: Vec<TermId> = Vec::new();
            let mut skip_rule = false;
            for body_app in &rule.body.predicates {
                let body_canon = canon_cur_vars(manager, chc, body_app.pred);
                if body_canon.len() != body_app.args.len() {
                    // Arity mismatch: cannot form a sound antecedent, so we
                    // cannot prove preservation for this rule => drop candidate.
                    skip_rule = true;
                    break;
                }
                let body_map: FxHashMap<TermId, TermId> = body_canon
                    .iter()
                    .zip(body_app.args.iter())
                    .map(|(&c, &a)| (c, a))
                    .collect();
                if let Some(assumed) = snapshot.get(&body_app.pred) {
                    for &inv in assumed {
                        antecedent.push(manager.substitute(inv, &body_map));
                    }
                }
            }
            if skip_rule {
                return false;
            }
            antecedent.push(rule.body.constraint);

            // Consequent: candidate renamed onto the rule's head arguments.
            let head_canon = canon_cur_vars(manager, chc, pred);
            if head_canon.len() != head_app.args.len() {
                return false;
            }
            let head_map: FxHashMap<TermId, TermId> = head_canon
                .iter()
                .zip(head_app.args.iter())
                .map(|(&c, &a)| (c, a))
                .collect();
            let candidate_head = manager.substitute(candidate, &head_map);
            let not_candidate = manager.mk_not(candidate_head);

            // Check SAT(antecedent ∧ ¬candidate_head).
            let mut smt = SmtSolver::new(manager, chc);
            smt.push();
            for a in &antecedent {
                smt.assert(*a);
            }
            smt.assert(not_candidate);
            let sat = smt.check_sat();
            smt.pop();

            match sat {
                Ok(false) => {}           // preserved by this rule
                Ok(true) => return false, // real violation
                Err(_) => return false,   // Unknown: cannot claim inductive
            }
        }
        true
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
        chc: &ChcSystem,
        manager: &mut TermManager,
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

        match inference.run_houdini(chc, manager) {
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

    /// Stack size and nesting depth for the deep-recursion test below.
    ///
    /// The two are scaled together on purpose: what the test actually pins is
    /// the *ratio* -- about 21 bytes of stack per nesting level
    /// (128 KiB / 6_250). A natively recursive flattener needs far more than
    /// that per frame and still overflows, so the regression keeps every bit
    /// of its detection power. The pair used to be 1 MiB / 50_000 -- the same
    /// 21 bytes. `mk_and` flattens its arguments (so `acc = mk_and([acc,
    /// atom])` never actually nests, and is quadratic to boot), so the deep
    /// term below is built with `TermManager::intern_term` directly, which
    /// neither flattens nor re-interns already-built prefixes. Never raise
    /// `DEEP_DEPTH` without raising `DEEP_STACK` by the same factor.
    const DEEP_STACK: usize = 1 << 17;
    const DEEP_DEPTH: u32 = 6_250;

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

    /// A deeply nested `And` body must be flattened without overflowing,
    /// and every leaf constraint must still be collected in order.
    #[test]
    fn extract_body_constraints_survives_deep_nesting() {
        let handle = std::thread::Builder::new()
            .stack_size(DEEP_STACK)
            .spawn(|| {
                let mut manager = TermManager::new();
                let bool_sort = manager.sorts.bool_sort;
                let int_sort = manager.sorts.int_sort;
                let zero = manager.mk_int(0);
                let first = manager.mk_var("v0", int_sort);
                let mut body = manager.mk_ge(first, zero);
                // Interned directly (not via `mk_and`, which would flatten
                // this back into one wide `And`) so the body really is
                // `DEEP_DEPTH` deep.
                for i in 1..DEEP_DEPTH {
                    let v = manager.mk_var(&format!("v{i}"), int_sort);
                    let atom = manager.mk_ge(v, zero);
                    body = manager.intern_term(TermKind::And(vec![body, atom].into()), bool_sort);
                }

                let engine = InvariantInference::default();
                let constraints = engine.extract_body_constraints(body, &manager);
                assert!(
                    constraints.len() >= DEEP_DEPTH as usize,
                    "every conjunct must be collected, got {}",
                    constraints.len()
                );
            })
            .expect("thread spawn should succeed");
        handle
            .join()
            .expect("deep constraint extraction must return");
    }

    /// Order and classification of collected constraints are unchanged.
    #[test]
    fn extract_body_constraints_pins_order() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let zero = manager.mk_int(0);
        let a = manager.mk_ge(x, zero);
        let b = manager.mk_lt(y, zero);
        let inner = manager.mk_and([a, b]);
        let c = manager.mk_eq(x, y);
        let body = manager.mk_and([inner, c]);

        let engine = InvariantInference::default();
        let constraints = engine.extract_body_constraints(body, &manager);
        assert_eq!(constraints, vec![a, b, c]);
    }

    /// Variables occurring under operators the old enumeration skipped must
    /// still be collected.
    #[test]
    fn collect_variables_sees_through_select() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let array_sort = manager.sorts.array(int_sort, int_sort);
        let arr = manager.mk_var("a", array_sort);
        let idx = manager.mk_var("i", int_sort);
        let select = manager.mk_select(arr, idx);

        let engine = InvariantInference::default();
        let vars = engine.extract_variables(select, &manager);
        assert!(
            vars.contains(&arr) && vars.contains(&idx),
            "both the array and the index variable must be collected"
        );
    }
}
