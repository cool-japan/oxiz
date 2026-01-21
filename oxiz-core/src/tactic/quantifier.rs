//! Quantifier tactics for OxiZ
//!
//! This module provides tactics for handling quantified formulas including:
//! - Ground term collection for instantiation candidates
//! - Pattern matching (E-matching) for trigger-based instantiation
//! - Quantifier instantiation tactics
//! - Skolemization tactics

use crate::ast::normal_forms::{eliminate_universal_quantifiers, skolemize};
use crate::ast::traversal::{
    TermVisitor, VisitorAction, collect_free_vars, collect_subterms, traverse,
};
use crate::ast::{TermId, TermKind, TermManager};
use crate::error::Result;
use crate::sort::SortId;
use lasso::Spur;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::{Goal, TacticResult};

/// Collects ground (variable-free) terms by their sort.
///
/// Ground terms are terms that contain no free variables and are not under
/// quantifiers. These terms serve as instantiation candidates for quantified
/// formulas.
#[derive(Debug, Default)]
pub struct GroundTermCollector {
    /// Ground terms indexed by sort
    terms_by_sort: FxHashMap<SortId, Vec<TermId>>,
    /// All collected ground terms (for deduplication)
    all_terms: FxHashSet<TermId>,
}

impl GroundTermCollector {
    /// Create a new ground term collector
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Collect all ground terms from a term
    pub fn collect(&mut self, term_id: TermId, manager: &TermManager) {
        let subterms = collect_subterms(term_id, manager);

        for subterm_id in subterms {
            // Skip if already collected
            if self.all_terms.contains(&subterm_id) {
                continue;
            }

            // Check if the term is ground (no free variables)
            let free_vars = collect_free_vars(subterm_id, manager);
            if free_vars.is_empty() {
                // Get the sort of this term
                if let Some(term) = manager.get(subterm_id) {
                    // Skip boolean constants and quantifiers
                    match &term.kind {
                        TermKind::True
                        | TermKind::False
                        | TermKind::Forall { .. }
                        | TermKind::Exists { .. } => continue,
                        _ => {}
                    }

                    let sort = term.sort;
                    self.all_terms.insert(subterm_id);
                    self.terms_by_sort.entry(sort).or_default().push(subterm_id);
                }
            }
        }
    }

    /// Collect ground terms from a goal (all assertions)
    pub fn collect_from_goal(&mut self, goal: &Goal, manager: &TermManager) {
        for &assertion in &goal.assertions {
            self.collect(assertion, manager);
        }
    }

    /// Get all ground terms of a specific sort
    #[must_use]
    pub fn get_terms(&self, sort: SortId) -> &[TermId] {
        self.terms_by_sort
            .get(&sort)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Get all collected ground terms
    #[must_use]
    pub fn all_terms(&self) -> &FxHashSet<TermId> {
        &self.all_terms
    }

    /// Get the number of ground terms collected
    #[must_use]
    pub fn len(&self) -> usize {
        self.all_terms.len()
    }

    /// Check if no ground terms were collected
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.all_terms.is_empty()
    }

    /// Get all sorts that have ground terms
    pub fn sorts(&self) -> impl Iterator<Item = &SortId> {
        self.terms_by_sort.keys()
    }

    /// Clear all collected terms
    pub fn clear(&mut self) {
        self.terms_by_sort.clear();
        self.all_terms.clear();
    }
}

/// A pattern for quantifier instantiation (trigger)
#[derive(Debug, Clone)]
pub struct Pattern {
    /// The quantifier this pattern belongs to
    pub quantifier: TermId,
    /// The trigger terms (multi-pattern)
    pub triggers: SmallVec<[TermId; 2]>,
    /// Bound variable names with their sorts
    pub bound_vars: SmallVec<[(Spur, SortId); 2]>,
    /// Body of the quantifier
    pub body: TermId,
}

/// A binding from bound variables to ground terms
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// Index of the pattern that produced this binding
    pub pattern_idx: usize,
    /// Substitution from variable names to ground terms
    pub substitution: FxHashMap<Spur, TermId>,
}

/// Pattern matcher for E-matching based quantifier instantiation
#[derive(Debug, Default)]
pub struct PatternMatcher {
    /// Registered patterns
    patterns: Vec<Pattern>,
    /// Already generated bindings (for deduplication)
    generated_bindings: FxHashSet<(usize, Vec<(Spur, TermId)>)>,
}

impl PatternMatcher {
    /// Create a new pattern matcher
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a pattern for a quantifier
    pub fn add_pattern(&mut self, quantifier: TermId, manager: &TermManager) {
        if let Some(term) = manager.get(quantifier)
            && let TermKind::Forall {
                vars,
                body,
                patterns,
            } = &term.kind
        {
            if patterns.is_empty() {
                // No explicit patterns - use heuristic trigger inference
                // For now, just use the body as a fallback (not ideal)
                self.patterns.push(Pattern {
                    quantifier,
                    triggers: SmallVec::new(),
                    bound_vars: vars.clone(),
                    body: *body,
                });
            } else {
                // Use explicit patterns
                for pattern in patterns {
                    self.patterns.push(Pattern {
                        quantifier,
                        triggers: pattern.clone(),
                        bound_vars: vars.clone(),
                        body: *body,
                    });
                }
            }
        }
    }

    /// Try to match a ground term against a pattern trigger
    ///
    /// Returns bindings if successful, None otherwise
    fn try_match_term(
        &self,
        pattern_term: TermId,
        ground_term: TermId,
        bound_vars: &[(Spur, SortId)],
        manager: &TermManager,
    ) -> Option<FxHashMap<Spur, TermId>> {
        let mut bindings = FxHashMap::default();
        let bound_var_names: FxHashSet<Spur> = bound_vars.iter().map(|(n, _)| *n).collect();

        if self.match_recursive(
            pattern_term,
            ground_term,
            &bound_var_names,
            &mut bindings,
            manager,
        ) {
            Some(bindings)
        } else {
            None
        }
    }

    /// Recursive pattern matching
    #[allow(clippy::only_used_in_recursion)]
    fn match_recursive(
        &self,
        pattern: TermId,
        ground: TermId,
        bound_vars: &FxHashSet<Spur>,
        bindings: &mut FxHashMap<Spur, TermId>,
        manager: &TermManager,
    ) -> bool {
        let pattern_term = match manager.get(pattern) {
            Some(t) => t,
            None => return false,
        };
        let ground_term = match manager.get(ground) {
            Some(t) => t,
            None => return false,
        };

        // Check sort compatibility
        if pattern_term.sort != ground_term.sort {
            return false;
        }

        match &pattern_term.kind {
            // Pattern variable - bind it
            TermKind::Var(name) if bound_vars.contains(name) => {
                if let Some(&existing) = bindings.get(name) {
                    // Already bound - check consistency
                    existing == ground
                } else {
                    // New binding
                    bindings.insert(*name, ground);
                    true
                }
            }

            // Non-pattern variable or constant - must match exactly
            TermKind::Var(_)
            | TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_) => pattern == ground,

            // Function application - match head and arguments
            TermKind::Apply { func, args, .. } => {
                if let TermKind::Apply {
                    func: gfunc,
                    args: gargs,
                    ..
                } = &ground_term.kind
                {
                    if func != gfunc || args.len() != gargs.len() {
                        return false;
                    }
                    args.iter()
                        .zip(gargs.iter())
                        .all(|(&p, &g)| self.match_recursive(p, g, bound_vars, bindings, manager))
                } else {
                    false
                }
            }

            // Equality - match both sides
            TermKind::Eq(p1, p2) => {
                if let TermKind::Eq(g1, g2) = &ground_term.kind {
                    (self.match_recursive(*p1, *g1, bound_vars, bindings, manager)
                        && self.match_recursive(*p2, *g2, bound_vars, bindings, manager))
                        || (self.match_recursive(*p1, *g2, bound_vars, bindings, manager)
                            && self.match_recursive(*p2, *g1, bound_vars, bindings, manager))
                } else {
                    false
                }
            }

            // Binary operations - match both operands
            TermKind::Add(args)
            | TermKind::Mul(args)
            | TermKind::And(args)
            | TermKind::Or(args) => {
                let ground_args = match &ground_term.kind {
                    TermKind::Add(a) | TermKind::Mul(a) | TermKind::And(a) | TermKind::Or(a) => a,
                    _ => return false,
                };
                // Check same operation type
                if std::mem::discriminant(&pattern_term.kind)
                    != std::mem::discriminant(&ground_term.kind)
                {
                    return false;
                }
                if args.len() != ground_args.len() {
                    return false;
                }
                args.iter()
                    .zip(ground_args.iter())
                    .all(|(&p, &g)| self.match_recursive(p, g, bound_vars, bindings, manager))
            }

            TermKind::Lt(p1, p2)
            | TermKind::Le(p1, p2)
            | TermKind::Gt(p1, p2)
            | TermKind::Ge(p1, p2)
            | TermKind::Sub(p1, p2)
            | TermKind::Div(p1, p2) => {
                let (g1, g2) = match &ground_term.kind {
                    TermKind::Lt(a, b)
                    | TermKind::Le(a, b)
                    | TermKind::Gt(a, b)
                    | TermKind::Ge(a, b)
                    | TermKind::Sub(a, b)
                    | TermKind::Div(a, b) => (*a, *b),
                    _ => return false,
                };
                if std::mem::discriminant(&pattern_term.kind)
                    != std::mem::discriminant(&ground_term.kind)
                {
                    return false;
                }
                self.match_recursive(*p1, g1, bound_vars, bindings, manager)
                    && self.match_recursive(*p2, g2, bound_vars, bindings, manager)
            }

            TermKind::Not(p1) => {
                if let TermKind::Not(g1) = &ground_term.kind {
                    self.match_recursive(*p1, *g1, bound_vars, bindings, manager)
                } else {
                    false
                }
            }

            TermKind::Neg(p1) => {
                if let TermKind::Neg(g1) = &ground_term.kind {
                    self.match_recursive(*p1, *g1, bound_vars, bindings, manager)
                } else {
                    false
                }
            }

            TermKind::Select(arr, idx) => {
                if let TermKind::Select(garr, gidx) = &ground_term.kind {
                    self.match_recursive(*arr, *garr, bound_vars, bindings, manager)
                        && self.match_recursive(*idx, *gidx, bound_vars, bindings, manager)
                } else {
                    false
                }
            }

            TermKind::Store(arr, idx, val) => {
                if let TermKind::Store(garr, gidx, gval) = &ground_term.kind {
                    self.match_recursive(*arr, *garr, bound_vars, bindings, manager)
                        && self.match_recursive(*idx, *gidx, bound_vars, bindings, manager)
                        && self.match_recursive(*val, *gval, bound_vars, bindings, manager)
                } else {
                    false
                }
            }

            TermKind::Ite(c, t, e) => {
                if let TermKind::Ite(gc, gt, ge) = &ground_term.kind {
                    self.match_recursive(*c, *gc, bound_vars, bindings, manager)
                        && self.match_recursive(*t, *gt, bound_vars, bindings, manager)
                        && self.match_recursive(*e, *ge, bound_vars, bindings, manager)
                } else {
                    false
                }
            }

            // Other terms - fallback to structural equality
            _ => pattern == ground,
        }
    }

    /// Match patterns against ground terms and generate bindings
    pub fn match_against(
        &mut self,
        ground_terms: &GroundTermCollector,
        manager: &TermManager,
    ) -> Vec<Binding> {
        let mut new_bindings = Vec::new();

        for (pattern_idx, pattern) in self.patterns.iter().enumerate() {
            // Skip patterns without triggers (need heuristic inference)
            if pattern.triggers.is_empty() {
                continue;
            }

            // For single-trigger patterns
            if pattern.triggers.len() == 1 {
                let trigger = pattern.triggers[0];

                // Try matching against each ground term
                for &ground_term in ground_terms.all_terms() {
                    if let Some(subst) =
                        self.try_match_term(trigger, ground_term, &pattern.bound_vars, manager)
                    {
                        // Check all bound variables are assigned
                        let all_bound = pattern
                            .bound_vars
                            .iter()
                            .all(|(n, _)| subst.contains_key(n));
                        if all_bound {
                            // Create binding key for deduplication
                            let mut key_vec: Vec<_> = subst.iter().map(|(&k, &v)| (k, v)).collect();
                            key_vec.sort_by_key(|(k, _)| k.into_inner());
                            let key = (pattern_idx, key_vec.clone());

                            if !self.generated_bindings.contains(&key) {
                                self.generated_bindings.insert(key);
                                new_bindings.push(Binding {
                                    pattern_idx,
                                    substitution: subst,
                                });
                            }
                        }
                    }
                }
            }
            // TODO: Multi-trigger patterns require more complex matching
        }

        new_bindings
    }

    /// Instantiate a quantifier body with a binding
    pub fn instantiate(&self, binding: &Binding, manager: &mut TermManager) -> Option<TermId> {
        let pattern = self.patterns.get(binding.pattern_idx)?;

        // Build substitution map from Spur to TermId
        let subst: FxHashMap<TermId, TermId> = pattern
            .bound_vars
            .iter()
            .filter_map(|(name, sort)| {
                let var_name = manager.resolve_str(*name).to_string();
                let var_id = manager.mk_var(&var_name, *sort);
                binding.substitution.get(name).map(|&term| (var_id, term))
            })
            .collect();

        Some(manager.substitute(pattern.body, &subst))
    }

    /// Get the number of patterns
    #[must_use]
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Get the quantifier for a pattern index
    #[must_use]
    pub fn get_quantifier(&self, pattern_idx: usize) -> Option<TermId> {
        self.patterns.get(pattern_idx).map(|p| p.quantifier)
    }

    /// Clear all patterns and bindings
    pub fn clear(&mut self) {
        self.patterns.clear();
        self.generated_bindings.clear();
    }
}

/// Quantifier instantiation tactic
///
/// This tactic finds quantified formulas (∀x. φ(x)), collects ground terms,
/// matches patterns, and generates instantiation lemmas (φ(t) for ground t).
#[derive(Debug)]
pub struct QuantifierInstantiationTactic<'a> {
    manager: &'a mut TermManager,
    /// Maximum number of instantiations per round
    max_instances: usize,
}

impl<'a> QuantifierInstantiationTactic<'a> {
    /// Create a new quantifier instantiation tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            max_instances: 100,
        }
    }

    /// Set the maximum number of instantiations per round
    pub fn with_max_instances(mut self, max: usize) -> Self {
        self.max_instances = max;
        self
    }

    /// Apply the tactic to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        // Phase 1: Collect quantifiers from goal
        let quantifiers = self.collect_quantifiers(goal);
        if quantifiers.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 2: Collect ground terms
        let mut ground_collector = GroundTermCollector::new();
        ground_collector.collect_from_goal(goal, self.manager);

        if ground_collector.is_empty() {
            // No ground terms to instantiate with
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 3: Set up pattern matcher
        let mut matcher = PatternMatcher::new();
        for &quant in &quantifiers {
            matcher.add_pattern(quant, self.manager);
        }

        // Phase 4: Match and generate bindings
        let bindings = matcher.match_against(&ground_collector, self.manager);

        if bindings.is_empty() {
            return Ok(TacticResult::NotApplicable);
        }

        // Phase 5: Generate instantiation lemmas
        let mut new_assertions = goal.assertions.clone();
        let mut count = 0;

        for binding in bindings {
            if count >= self.max_instances {
                break;
            }

            if let Some(instance) = matcher.instantiate(&binding, self.manager) {
                // Add the instance as a new assertion
                // The instantiation lemma is: ∀x.φ(x) → φ(t)
                // Since we assume ∀x.φ(x) is asserted, we can add φ(t)
                new_assertions.push(instance);
                count += 1;
            }
        }

        if count == 0 {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Collect all universal quantifiers from a goal
    fn collect_quantifiers(&self, goal: &Goal) -> Vec<TermId> {
        struct QuantifierCollector {
            quantifiers: Vec<TermId>,
        }

        impl TermVisitor for QuantifierCollector {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id)
                    && matches!(term.kind, TermKind::Forall { .. })
                {
                    self.quantifiers.push(term_id);
                }
                VisitorAction::Continue
            }
        }

        let mut collector = QuantifierCollector {
            quantifiers: Vec::new(),
        };

        for &assertion in &goal.assertions {
            let _ = traverse(assertion, self.manager, &mut collector);
        }

        collector.quantifiers
    }
}

/// Skolemization tactic
///
/// Eliminates existential quantifiers by replacing them with Skolem
/// functions/constants.
#[derive(Debug)]
pub struct SkolemizationTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> SkolemizationTactic<'a> {
    /// Create a new Skolemization tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply the tactic to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());

        for &assertion in &goal.assertions {
            let skolemized = skolemize(assertion, self.manager);
            if skolemized != assertion {
                changed = true;
            }
            new_assertions.push(skolemized);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Universal elimination tactic
///
/// Eliminates universal quantifiers by replacing bound variables with
/// fresh constants.
#[derive(Debug)]
pub struct UniversalEliminationTactic<'a> {
    manager: &'a mut TermManager,
}

impl<'a> UniversalEliminationTactic<'a> {
    /// Create a new universal elimination tactic
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self { manager }
    }

    /// Apply the tactic to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());

        for &assertion in &goal.assertions {
            let eliminated = eliminate_universal_quantifiers(assertion, self.manager);
            if eliminated != assertion {
                changed = true;
            }
            new_assertions.push(eliminated);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }
}

/// Check if a term contains any quantifiers
#[must_use]
pub fn contains_quantifier(term_id: TermId, manager: &TermManager) -> bool {
    struct QuantifierChecker {
        found: bool,
    }

    impl TermVisitor for QuantifierChecker {
        fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
            if let Some(term) = manager.get(term_id)
                && matches!(term.kind, TermKind::Forall { .. } | TermKind::Exists { .. })
            {
                self.found = true;
                return VisitorAction::Stop;
            }
            VisitorAction::Continue
        }
    }

    let mut checker = QuantifierChecker { found: false };
    let _ = traverse(term_id, manager, &mut checker);
    checker.found
}

/// Check if a goal contains any quantifiers
#[must_use]
pub fn goal_has_quantifiers(goal: &Goal, manager: &TermManager) -> bool {
    goal.assertions
        .iter()
        .any(|&a| contains_quantifier(a, manager))
}

// ============================================================================
// Destructive Equality Resolution (DER)
// ============================================================================

/// Configuration for Destructive Equality Resolution
#[derive(Debug, Clone)]
pub struct DerConfig {
    /// Maximum depth to search for eliminable equalities
    pub max_depth: usize,
    /// Whether to apply DER recursively to nested quantifiers
    pub recursive: bool,
    /// Whether to handle disequalities (x ≠ t implies false path)
    pub handle_diseq: bool,
}

impl Default for DerConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            recursive: true,
            handle_diseq: true,
        }
    }
}

/// Represents a discovered eliminable equality
#[derive(Debug, Clone)]
struct EliminableEquality {
    /// The bound variable name
    var_name: Spur,
    /// The term to substitute for the variable
    substitute: TermId,
    /// Whether this is a positive (x = t) or negative (x ≠ t) equality
    /// Reserved for future disequality handling
    #[allow(dead_code)]
    is_positive: bool,
}

/// Destructive Equality Resolution (DER) Tactic
///
/// DER eliminates quantifiers when there are equalities that allow
/// direct variable substitution.
///
/// For universal quantifiers (∀x. φ):
/// - ∀x. (x = t ∨ ψ(x)) where x ∉ FV(t) becomes ψ(t)
/// - ∀x. (x ≠ t → ψ(x)) is equivalent to the above
///
/// For existential quantifiers (∃x. φ):
/// - ∃x. (x = t ∧ ψ(x)) where x ∉ FV(t) becomes ψ(t)
///
/// This is a powerful simplification that can eliminate quantifiers entirely.
#[derive(Debug)]
pub struct DerTactic<'a> {
    manager: &'a mut TermManager,
    config: DerConfig,
}

impl<'a> DerTactic<'a> {
    /// Create a new DER tactic with default configuration
    pub fn new(manager: &'a mut TermManager) -> Self {
        Self {
            manager,
            config: DerConfig::default(),
        }
    }

    /// Create a new DER tactic with custom configuration
    pub fn with_config(manager: &'a mut TermManager, config: DerConfig) -> Self {
        Self { manager, config }
    }

    /// Apply the tactic to a goal
    pub fn apply_mut(&mut self, goal: &Goal) -> Result<TacticResult> {
        let mut changed = false;
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());

        for &assertion in &goal.assertions {
            let simplified = self.apply_der(assertion, 0);
            if simplified != assertion {
                changed = true;
            }
            new_assertions.push(simplified);
        }

        if !changed {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    /// Apply DER to a single term
    fn apply_der(&mut self, term_id: TermId, depth: usize) -> TermId {
        if depth > self.config.max_depth {
            return term_id;
        }

        let term = match self.manager.get(term_id) {
            Some(t) => t.clone(),
            None => return term_id,
        };

        match &term.kind {
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                // First, recursively apply DER to the body if configured
                let processed_body = if self.config.recursive {
                    self.apply_der(*body, depth + 1)
                } else {
                    *body
                };

                // Try to find eliminable equalities
                self.apply_der_forall(vars.clone(), processed_body, patterns.clone())
            }
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                // First, recursively apply DER to the body if configured
                let processed_body = if self.config.recursive {
                    self.apply_der(*body, depth + 1)
                } else {
                    *body
                };

                // Try to find eliminable equalities
                self.apply_der_exists(vars.clone(), processed_body, patterns.clone())
            }
            // For non-quantifier terms, recursively process children if recursive
            TermKind::And(args) if self.config.recursive => {
                let new_args: SmallVec<[TermId; 4]> =
                    args.iter().map(|&a| self.apply_der(a, depth + 1)).collect();
                if new_args == *args {
                    term_id
                } else {
                    self.manager.mk_and(new_args)
                }
            }
            TermKind::Or(args) if self.config.recursive => {
                let new_args: SmallVec<[TermId; 4]> =
                    args.iter().map(|&a| self.apply_der(a, depth + 1)).collect();
                if new_args == *args {
                    term_id
                } else {
                    self.manager.mk_or(new_args)
                }
            }
            TermKind::Not(inner) if self.config.recursive => {
                let new_inner = self.apply_der(*inner, depth + 1);
                if new_inner == *inner {
                    term_id
                } else {
                    self.manager.mk_not(new_inner)
                }
            }
            TermKind::Implies(lhs, rhs) if self.config.recursive => {
                let new_lhs = self.apply_der(*lhs, depth + 1);
                let new_rhs = self.apply_der(*rhs, depth + 1);
                if new_lhs == *lhs && new_rhs == *rhs {
                    term_id
                } else {
                    self.manager.mk_implies(new_lhs, new_rhs)
                }
            }
            _ => term_id,
        }
    }

    /// Apply DER to a universal quantifier
    ///
    /// For ∀x. φ, we look for patterns like:
    /// - (x = t ∨ ψ) → ψ[t/x]
    /// - (x ≠ t → ψ) which is equivalent to (x = t ∨ ψ) → ψ[t/x]
    fn apply_der_forall(
        &mut self,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    ) -> TermId {
        let bound_var_names: FxHashSet<Spur> = vars.iter().map(|(n, _)| *n).collect();

        // Look for eliminable equality in disjunction: x = t ∨ ψ
        if let Some(eq) = self.find_eliminable_equality_in_or(body, &bound_var_names, true) {
            return self.eliminate_variable(vars, body, patterns, &eq, true);
        }

        // Look for negated implication pattern: ¬(x ≠ t) ∨ ψ which is x = t ∨ ψ
        // This is handled by the above

        // Look for implication pattern: x ≠ t → ψ which is x = t ∨ ψ
        if let Some(term) = self.manager.get(body)
            && let TermKind::Implies(lhs, rhs) = &term.kind
        {
            let lhs = *lhs;
            let rhs = *rhs;
            // Check if lhs is x ≠ t (i.e., Not(Eq(x, t)))
            if let Some(eq) = self.extract_diseq_var(lhs, &bound_var_names) {
                // x ≠ t → ψ is equivalent to x = t ∨ ψ
                return self.eliminate_variable_with_substitute(vars, rhs, patterns, &eq);
            }
        }

        // No eliminable equality found - return original or rebuilt
        if vars.is_empty() {
            body
        } else {
            // Convert Spur names to owned strings first
            let var_names: Vec<_> = vars
                .iter()
                .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
                .collect();
            // Now create references for the API
            let var_strs: Vec<_> = var_names
                .iter()
                .map(|(name, sort)| (name.as_str(), *sort))
                .collect();
            self.manager
                .mk_forall_with_patterns(var_strs, body, patterns)
        }
    }

    /// Apply DER to an existential quantifier
    ///
    /// For ∃x. φ, we look for patterns like:
    /// - (x = t ∧ ψ) → ψ[t/x]
    fn apply_der_exists(
        &mut self,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    ) -> TermId {
        let bound_var_names: FxHashSet<Spur> = vars.iter().map(|(n, _)| *n).collect();

        // Look for eliminable equality in conjunction: x = t ∧ ψ
        if let Some(eq) = self.find_eliminable_equality_in_and(body, &bound_var_names) {
            return self.eliminate_variable(vars, body, patterns, &eq, false);
        }

        // No eliminable equality found - return original or rebuilt
        if vars.is_empty() {
            body
        } else {
            // Convert Spur names to owned strings first
            let var_names: Vec<_> = vars
                .iter()
                .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
                .collect();
            // Now create references for the API
            let var_strs: Vec<_> = var_names
                .iter()
                .map(|(name, sort)| (name.as_str(), *sort))
                .collect();
            self.manager
                .mk_exists_with_patterns(var_strs, body, patterns)
        }
    }

    /// Find an eliminable equality in a disjunction (for ∀)
    fn find_eliminable_equality_in_or(
        &self,
        term_id: TermId,
        bound_vars: &FxHashSet<Spur>,
        positive: bool,
    ) -> Option<EliminableEquality> {
        let term = self.manager.get(term_id)?;

        match &term.kind {
            TermKind::Or(args) => {
                // Look through disjuncts for x = t
                for &arg in args.iter() {
                    if let Some(eq) = self.extract_eq_var(arg, bound_vars) {
                        return Some(EliminableEquality {
                            var_name: eq.0,
                            substitute: eq.1,
                            is_positive: positive,
                        });
                    }
                }
                None
            }
            TermKind::Eq(lhs, rhs) => {
                // Direct equality
                self.check_eliminable_eq(*lhs, *rhs, bound_vars)
                    .map(|(var_name, substitute)| EliminableEquality {
                        var_name,
                        substitute,
                        is_positive: positive,
                    })
            }
            _ => None,
        }
    }

    /// Find an eliminable equality in a conjunction (for ∃)
    fn find_eliminable_equality_in_and(
        &self,
        term_id: TermId,
        bound_vars: &FxHashSet<Spur>,
    ) -> Option<EliminableEquality> {
        let term = self.manager.get(term_id)?;

        match &term.kind {
            TermKind::And(args) => {
                // Look through conjuncts for x = t
                for &arg in args.iter() {
                    if let Some(eq) = self.extract_eq_var(arg, bound_vars) {
                        return Some(EliminableEquality {
                            var_name: eq.0,
                            substitute: eq.1,
                            is_positive: true,
                        });
                    }
                }
                None
            }
            TermKind::Eq(lhs, rhs) => {
                // Direct equality
                self.check_eliminable_eq(*lhs, *rhs, bound_vars)
                    .map(|(var_name, substitute)| EliminableEquality {
                        var_name,
                        substitute,
                        is_positive: true,
                    })
            }
            _ => None,
        }
    }

    /// Extract a disequality pattern Not(Eq(x, t)) where x is a bound var
    fn extract_diseq_var(
        &self,
        term_id: TermId,
        bound_vars: &FxHashSet<Spur>,
    ) -> Option<EliminableEquality> {
        let term = self.manager.get(term_id)?;

        if let TermKind::Not(inner) = &term.kind
            && let Some((var_name, substitute)) = self.extract_eq_var(*inner, bound_vars)
        {
            return Some(EliminableEquality {
                var_name,
                substitute,
                is_positive: false,
            });
        }

        None
    }

    /// Extract equality x = t where x is a bound variable
    fn extract_eq_var(
        &self,
        term_id: TermId,
        bound_vars: &FxHashSet<Spur>,
    ) -> Option<(Spur, TermId)> {
        let term = self.manager.get(term_id)?;

        if let TermKind::Eq(lhs, rhs) = &term.kind {
            return self.check_eliminable_eq(*lhs, *rhs, bound_vars);
        }

        None
    }

    /// Check if lhs = rhs is an eliminable equality (one side is a bound var,
    /// other side doesn't contain that var)
    fn check_eliminable_eq(
        &self,
        lhs: TermId,
        rhs: TermId,
        bound_vars: &FxHashSet<Spur>,
    ) -> Option<(Spur, TermId)> {
        // Check if lhs is a bound variable and rhs doesn't contain it
        if let Some(lhs_term) = self.manager.get(lhs)
            && let TermKind::Var(name) = &lhs_term.kind
            && bound_vars.contains(name)
            && !self.term_contains_var(rhs, *name)
        {
            return Some((*name, rhs));
        }

        // Check if rhs is a bound variable and lhs doesn't contain it
        if let Some(rhs_term) = self.manager.get(rhs)
            && let TermKind::Var(name) = &rhs_term.kind
            && bound_vars.contains(name)
            && !self.term_contains_var(lhs, *name)
        {
            return Some((*name, lhs));
        }

        None
    }

    /// Check if a term contains a specific variable
    fn term_contains_var(&self, term_id: TermId, var_name: Spur) -> bool {
        struct VarChecker {
            var_name: Spur,
            found: bool,
        }

        impl TermVisitor for VarChecker {
            fn visit_pre(&mut self, term_id: TermId, manager: &TermManager) -> VisitorAction {
                if let Some(term) = manager.get(term_id)
                    && let TermKind::Var(name) = &term.kind
                    && *name == self.var_name
                {
                    self.found = true;
                    return VisitorAction::Stop;
                }
                VisitorAction::Continue
            }
        }

        let mut checker = VarChecker {
            var_name,
            found: false,
        };
        let _ = traverse(term_id, self.manager, &mut checker);
        checker.found
    }

    /// Eliminate a variable using a discovered equality
    fn eliminate_variable(
        &mut self,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        eq: &EliminableEquality,
        is_forall: bool,
    ) -> TermId {
        // Remove the equality from the body and substitute
        let new_body = if is_forall {
            self.remove_from_or_and_substitute(body, eq)
        } else {
            self.remove_from_and_and_substitute(body, eq)
        };

        // Remove the eliminated variable from the bound vars
        let remaining_vars: SmallVec<[(Spur, SortId); 2]> = vars
            .iter()
            .filter(|(n, _)| *n != eq.var_name)
            .copied()
            .collect();

        // If no variables remain, just return the body
        if remaining_vars.is_empty() {
            return new_body;
        }

        // Rebuild quantifier with remaining variables
        // Convert Spur names to owned strings first
        let var_names: Vec<_> = remaining_vars
            .iter()
            .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
            .collect();
        // Now create references for the API
        let var_strs: Vec<_> = var_names
            .iter()
            .map(|(name, sort)| (name.as_str(), *sort))
            .collect();

        if is_forall {
            self.manager
                .mk_forall_with_patterns(var_strs, new_body, patterns)
        } else {
            self.manager
                .mk_exists_with_patterns(var_strs, new_body, patterns)
        }
    }

    /// Eliminate a variable with a direct substitute (for implication pattern)
    fn eliminate_variable_with_substitute(
        &mut self,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
        eq: &EliminableEquality,
    ) -> TermId {
        // Substitute the variable in the body
        let substituted_body = self.substitute_var(body, eq.var_name, eq.substitute);

        // Remove the eliminated variable from the bound vars
        let remaining_vars: SmallVec<[(Spur, SortId); 2]> = vars
            .iter()
            .filter(|(n, _)| *n != eq.var_name)
            .copied()
            .collect();

        // If no variables remain, just return the body
        if remaining_vars.is_empty() {
            return substituted_body;
        }

        // Rebuild quantifier with remaining variables
        // Convert Spur names to owned strings first
        let var_names: Vec<_> = remaining_vars
            .iter()
            .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
            .collect();
        // Now create references for the API
        let var_strs: Vec<_> = var_names
            .iter()
            .map(|(name, sort)| (name.as_str(), *sort))
            .collect();

        self.manager
            .mk_forall_with_patterns(var_strs, substituted_body, patterns)
    }

    /// Remove the equality from an OR and substitute in remaining disjuncts
    fn remove_from_or_and_substitute(
        &mut self,
        term_id: TermId,
        eq: &EliminableEquality,
    ) -> TermId {
        let term = match self.manager.get(term_id) {
            Some(t) => t.clone(),
            None => return term_id,
        };

        match &term.kind {
            TermKind::Or(args) => {
                // Filter out the equality and substitute in remaining disjuncts
                let mut new_args = Vec::new();
                for &arg in args.iter() {
                    if !self.is_equality_for_var(arg, eq.var_name) {
                        let substituted = self.substitute_var(arg, eq.var_name, eq.substitute);
                        new_args.push(substituted);
                    }
                }

                match new_args.len() {
                    0 => self.manager.mk_false(),
                    1 => new_args[0],
                    _ => self.manager.mk_or(new_args),
                }
            }
            TermKind::Eq(_, _) => {
                // The entire body is the equality - result is true (trivially satisfied)
                self.manager.mk_true()
            }
            _ => {
                // Just substitute
                self.substitute_var(term_id, eq.var_name, eq.substitute)
            }
        }
    }

    /// Remove the equality from an AND and substitute in remaining conjuncts
    fn remove_from_and_and_substitute(
        &mut self,
        term_id: TermId,
        eq: &EliminableEquality,
    ) -> TermId {
        let term = match self.manager.get(term_id) {
            Some(t) => t.clone(),
            None => return term_id,
        };

        match &term.kind {
            TermKind::And(args) => {
                // Filter out the equality and substitute in remaining conjuncts
                let mut new_args = Vec::new();
                for &arg in args.iter() {
                    if !self.is_equality_for_var(arg, eq.var_name) {
                        let substituted = self.substitute_var(arg, eq.var_name, eq.substitute);
                        new_args.push(substituted);
                    }
                }

                match new_args.len() {
                    0 => self.manager.mk_true(),
                    1 => new_args[0],
                    _ => self.manager.mk_and(new_args),
                }
            }
            TermKind::Eq(_, _) => {
                // The entire body is the equality - result is true (equality holds)
                self.manager.mk_true()
            }
            _ => {
                // Just substitute
                self.substitute_var(term_id, eq.var_name, eq.substitute)
            }
        }
    }

    /// Check if a term is an equality involving a specific variable
    fn is_equality_for_var(&self, term_id: TermId, var_name: Spur) -> bool {
        if let Some(term) = self.manager.get(term_id)
            && let TermKind::Eq(lhs, rhs) = &term.kind
        {
            // Check lhs
            if let Some(lhs_term) = self.manager.get(*lhs)
                && let TermKind::Var(name) = &lhs_term.kind
                && *name == var_name
            {
                return true;
            }
            // Check rhs
            if let Some(rhs_term) = self.manager.get(*rhs)
                && let TermKind::Var(name) = &rhs_term.kind
                && *name == var_name
            {
                return true;
            }
        }
        false
    }

    /// Substitute a variable with a term throughout an expression
    fn substitute_var(&mut self, term_id: TermId, var_name: Spur, replacement: TermId) -> TermId {
        let term = match self.manager.get(term_id) {
            Some(t) => t.clone(),
            None => return term_id,
        };

        match &term.kind {
            TermKind::Var(name) if *name == var_name => replacement,

            TermKind::And(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_var(a, var_name, replacement))
                    .collect();
                self.manager.mk_and(new_args)
            }
            TermKind::Or(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_var(a, var_name, replacement))
                    .collect();
                self.manager.mk_or(new_args)
            }
            TermKind::Not(inner) => {
                let new_inner = self.substitute_var(*inner, var_name, replacement);
                self.manager.mk_not(new_inner)
            }
            TermKind::Implies(lhs, rhs) => {
                let new_lhs = self.substitute_var(*lhs, var_name, replacement);
                let new_rhs = self.substitute_var(*rhs, var_name, replacement);
                self.manager.mk_implies(new_lhs, new_rhs)
            }
            TermKind::Eq(lhs, rhs) => {
                let new_lhs = self.substitute_var(*lhs, var_name, replacement);
                let new_rhs = self.substitute_var(*rhs, var_name, replacement);
                self.manager.mk_eq(new_lhs, new_rhs)
            }
            TermKind::Add(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_var(a, var_name, replacement))
                    .collect();
                self.manager.mk_add(new_args)
            }
            TermKind::Mul(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_var(a, var_name, replacement))
                    .collect();
                self.manager.mk_mul(new_args)
            }
            TermKind::Sub(lhs, rhs) => {
                let new_lhs = self.substitute_var(*lhs, var_name, replacement);
                let new_rhs = self.substitute_var(*rhs, var_name, replacement);
                self.manager.mk_sub(new_lhs, new_rhs)
            }
            TermKind::Lt(lhs, rhs) => {
                let new_lhs = self.substitute_var(*lhs, var_name, replacement);
                let new_rhs = self.substitute_var(*rhs, var_name, replacement);
                self.manager.mk_lt(new_lhs, new_rhs)
            }
            TermKind::Le(lhs, rhs) => {
                let new_lhs = self.substitute_var(*lhs, var_name, replacement);
                let new_rhs = self.substitute_var(*rhs, var_name, replacement);
                self.manager.mk_le(new_lhs, new_rhs)
            }
            TermKind::Gt(lhs, rhs) => {
                let new_lhs = self.substitute_var(*lhs, var_name, replacement);
                let new_rhs = self.substitute_var(*rhs, var_name, replacement);
                self.manager.mk_gt(new_lhs, new_rhs)
            }
            TermKind::Ge(lhs, rhs) => {
                let new_lhs = self.substitute_var(*lhs, var_name, replacement);
                let new_rhs = self.substitute_var(*rhs, var_name, replacement);
                self.manager.mk_ge(new_lhs, new_rhs)
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let new_cond = self.substitute_var(*cond, var_name, replacement);
                let new_then = self.substitute_var(*then_br, var_name, replacement);
                let new_else = self.substitute_var(*else_br, var_name, replacement);
                self.manager.mk_ite(new_cond, new_then, new_else)
            }
            TermKind::Apply { func, args } => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_var(a, var_name, replacement))
                    .collect();
                let result_sort = term.sort;
                let func_name = self.manager.resolve_str(*func).to_string();
                self.manager.mk_apply(&func_name, new_args, result_sort)
            }
            TermKind::Select(arr, idx) => {
                let new_arr = self.substitute_var(*arr, var_name, replacement);
                let new_idx = self.substitute_var(*idx, var_name, replacement);
                self.manager.mk_select(new_arr, new_idx)
            }
            TermKind::Store(arr, idx, val) => {
                let new_arr = self.substitute_var(*arr, var_name, replacement);
                let new_idx = self.substitute_var(*idx, var_name, replacement);
                let new_val = self.substitute_var(*val, var_name, replacement);
                self.manager.mk_store(new_arr, new_idx, new_val)
            }
            // For quantifiers, be careful about variable capture
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                // Check if var_name is shadowed
                if vars.iter().any(|(n, _)| *n == var_name) {
                    term_id // Variable is bound here, don't substitute
                } else {
                    let new_body = self.substitute_var(*body, var_name, replacement);
                    let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                        .iter()
                        .map(|p| {
                            p.iter()
                                .map(|&t| self.substitute_var(t, var_name, replacement))
                                .collect()
                        })
                        .collect();
                    // Convert Spur names to owned strings first
                    let var_names: Vec<_> = vars
                        .iter()
                        .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
                        .collect();
                    // Now create references for the API
                    let var_strs: Vec<_> = var_names
                        .iter()
                        .map(|(name, sort)| (name.as_str(), *sort))
                        .collect();
                    self.manager
                        .mk_forall_with_patterns(var_strs, new_body, new_patterns)
                }
            }
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                // Check if var_name is shadowed
                if vars.iter().any(|(n, _)| *n == var_name) {
                    term_id // Variable is bound here, don't substitute
                } else {
                    let new_body = self.substitute_var(*body, var_name, replacement);
                    let new_patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
                        .iter()
                        .map(|p| {
                            p.iter()
                                .map(|&t| self.substitute_var(t, var_name, replacement))
                                .collect()
                        })
                        .collect();
                    // Convert Spur names to owned strings first
                    let var_names: Vec<_> = vars
                        .iter()
                        .map(|(n, s)| (self.manager.resolve_str(*n).to_string(), *s))
                        .collect();
                    // Now create references for the API
                    let var_strs: Vec<_> = var_names
                        .iter()
                        .map(|(name, sort)| (name.as_str(), *sort))
                        .collect();
                    self.manager
                        .mk_exists_with_patterns(var_strs, new_body, new_patterns)
                }
            }
            // For other terms, just return as-is (constants, etc.)
            _ => term_id,
        }
    }
}

/// Stateless wrapper for DER tactic
#[derive(Debug, Clone, Default)]
pub struct StatelessDerTactic {
    config: DerConfig,
}

impl StatelessDerTactic {
    /// Create a new stateless DER tactic
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: DerConfig) -> Self {
        Self { config }
    }

    /// Apply the tactic
    pub fn apply(&self, goal: &Goal, manager: &mut TermManager) -> Result<TacticResult> {
        let mut tactic = DerTactic::with_config(manager, self.config.clone());
        tactic.apply_mut(goal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_manager() -> TermManager {
        TermManager::new()
    }

    #[test]
    fn test_ground_term_collector() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create some terms
        let x = manager.mk_var("x", int_sort);
        let one = manager.mk_int(1);
        let two = manager.mk_int(2);
        let sum = manager.mk_add([one, two]);

        // Collect from multiple terms
        let term_with_var = manager.mk_add([x, one]);

        let mut collector = GroundTermCollector::new();
        // Collect from ground sum (1 + 2)
        collector.collect(sum, &manager);
        // Collect from non-ground term (x + 1)
        collector.collect(term_with_var, &manager);

        // Should have collected 1, 2, and (1 + 2) from ground term
        assert!(collector.all_terms.contains(&one));
        assert!(collector.all_terms.contains(&two));
        assert!(collector.all_terms.contains(&sum)); // ground term (1 + 2)
        // Should NOT have collected x or (x + 1) since they contain free vars
        assert!(!collector.all_terms.contains(&x));
        assert!(!collector.all_terms.contains(&term_with_var));
    }

    #[test]
    fn test_pattern_matching_simple() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create pattern: f(x) where x is bound
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);

        // Create ground term: f(1)
        let one = manager.mk_int(1);
        let f_one = manager.mk_apply("f", [one], int_sort);

        // Create quantifier: forall x. P(f(x)) with pattern f(x)
        let x_name = manager.intern_str("x");
        let bound_vars: SmallVec<[(Spur, SortId); 2]> = smallvec::smallvec![(x_name, int_sort)];

        let matcher = PatternMatcher::new();
        let result = matcher.try_match_term(f_x, f_one, &bound_vars, &manager);

        assert!(result.is_some());
        let bindings = result.expect("should have bindings");
        assert_eq!(bindings.get(&x_name), Some(&one));
    }

    #[test]
    fn test_skolemization_tactic() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: exists x. x > 0
        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_gt(x, zero);
        let exists = manager.mk_exists([("x", int_sort)], body);

        let goal = Goal::new(vec![exists]);

        let mut tactic = SkolemizationTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("tactic should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // The existential should be eliminated
                assert!(
                    !goal_has_quantifiers(&goals[0], &manager)
                        || !goals[0].assertions.iter().any(|&a| {
                            if let Some(t) = manager.get(a) {
                                matches!(t.kind, TermKind::Exists { .. })
                            } else {
                                false
                            }
                        })
                );
            }
            _ => panic!("Expected SubGoals result"),
        }
    }

    #[test]
    fn test_contains_quantifier() {
        let mut manager = setup_manager();
        let bool_sort = manager.sorts.bool_sort;
        let int_sort = manager.sorts.int_sort;

        // Create: forall x. P(x)
        let x = manager.mk_var("x", int_sort);
        let p_x = manager.mk_apply("P", [x], bool_sort);
        let forall = manager.mk_forall([("x", int_sort)], p_x);

        assert!(contains_quantifier(forall, &manager));
        assert!(!contains_quantifier(p_x, &manager));
    }

    // ============================================================================
    // DER (Destructive Equality Resolution) Tests
    // ============================================================================

    #[test]
    fn test_der_config_default() {
        let config = DerConfig::default();
        assert_eq!(config.max_depth, 10);
        assert!(config.recursive);
        assert!(config.handle_diseq);
    }

    #[test]
    fn test_der_forall_with_equality_in_or() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: ∀x. (x = 5 ∨ P(x))
        // Should simplify to: P(5)
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let x_eq_5 = manager.mk_eq(x, five);
        let p_x = manager.mk_apply("P", [x], manager.sorts.bool_sort);
        let body = manager.mk_or([x_eq_5, p_x]);
        let forall = manager.mk_forall([("x", int_sort)], body);

        let goal = Goal::new(vec![forall]);
        let mut tactic = DerTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("DER should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // The quantifier should be eliminated
                assert!(!goal_has_quantifiers(&goals[0], &manager));
            }
            _ => panic!("Expected SubGoals result"),
        }
    }

    #[test]
    fn test_der_exists_with_equality_in_and() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: ∃x. (x = 5 ∧ P(x))
        // Should simplify to: P(5)
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let x_eq_5 = manager.mk_eq(x, five);
        let p_x = manager.mk_apply("P", [x], manager.sorts.bool_sort);
        let body = manager.mk_and([x_eq_5, p_x]);
        let exists = manager.mk_exists([("x", int_sort)], body);

        let goal = Goal::new(vec![exists]);
        let mut tactic = DerTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("DER should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // The quantifier should be eliminated
                assert!(!goal_has_quantifiers(&goals[0], &manager));
            }
            _ => panic!("Expected SubGoals result"),
        }
    }

    #[test]
    fn test_der_not_applicable_no_equality() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: ∀x. P(x)
        // No equality to eliminate - DER should not apply
        let x = manager.mk_var("x", int_sort);
        let p_x = manager.mk_apply("P", [x], manager.sorts.bool_sort);
        let forall = manager.mk_forall([("x", int_sort)], p_x);

        let goal = Goal::new(vec![forall]);
        let mut tactic = DerTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("DER should succeed");

        match result {
            TacticResult::NotApplicable => (),
            _ => panic!("Expected NotApplicable result"),
        }
    }

    #[test]
    fn test_der_symmetric_equality() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: ∃x. (5 = x ∧ P(x))
        // Should also simplify to: P(5) (equality is symmetric)
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let five_eq_x = manager.mk_eq(five, x); // Note: 5 = x instead of x = 5
        let p_x = manager.mk_apply("P", [x], manager.sorts.bool_sort);
        let body = manager.mk_and([five_eq_x, p_x]);
        let exists = manager.mk_exists([("x", int_sort)], body);

        let goal = Goal::new(vec![exists]);
        let mut tactic = DerTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("DER should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // The quantifier should be eliminated
                assert!(!goal_has_quantifiers(&goals[0], &manager));
            }
            _ => panic!("Expected SubGoals result"),
        }
    }

    #[test]
    fn test_der_multiple_bound_vars() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: ∃x y. (x = 5 ∧ P(x, y))
        // Should simplify to: ∃y. P(5, y)
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);
        let x_eq_5 = manager.mk_eq(x, five);
        let p_xy = manager.mk_apply("P", [x, y], manager.sorts.bool_sort);
        let body = manager.mk_and([x_eq_5, p_xy]);
        let exists = manager.mk_exists([("x", int_sort), ("y", int_sort)], body);

        let goal = Goal::new(vec![exists]);
        let mut tactic = DerTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("DER should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                // Should still have quantifier (for y), but x should be eliminated
                let assertion = goals[0].assertions[0];
                if let Some(term) = manager.get(assertion) {
                    if let TermKind::Exists { vars, .. } = &term.kind {
                        // Only y should remain
                        assert_eq!(vars.len(), 1);
                    } else {
                        panic!("Expected Exists term");
                    }
                } else {
                    panic!("Term not found");
                }
            }
            _ => panic!("Expected SubGoals result"),
        }
    }

    #[test]
    fn test_der_var_occurs_in_substitute_fails() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: ∃x. (x = f(x) ∧ P(x))
        // DER should NOT apply because x occurs in f(x)
        let x = manager.mk_var("x", int_sort);
        let f_x = manager.mk_apply("f", [x], int_sort);
        let x_eq_fx = manager.mk_eq(x, f_x);
        let p_x = manager.mk_apply("P", [x], manager.sorts.bool_sort);
        let body = manager.mk_and([x_eq_fx, p_x]);
        let exists = manager.mk_exists([("x", int_sort)], body);

        let goal = Goal::new(vec![exists]);
        let mut tactic = DerTactic::new(&mut manager);
        let result = tactic.apply_mut(&goal).expect("DER should succeed");

        match result {
            TacticResult::NotApplicable => (),
            _ => panic!("Expected NotApplicable result (x occurs in substitute)"),
        }
    }

    #[test]
    fn test_stateless_der_tactic() {
        let mut manager = setup_manager();
        let int_sort = manager.sorts.int_sort;

        // Create: ∃x. (x = 5 ∧ P(x))
        let x = manager.mk_var("x", int_sort);
        let five = manager.mk_int(5);
        let x_eq_5 = manager.mk_eq(x, five);
        let p_x = manager.mk_apply("P", [x], manager.sorts.bool_sort);
        let body = manager.mk_and([x_eq_5, p_x]);
        let exists = manager.mk_exists([("x", int_sort)], body);

        let goal = Goal::new(vec![exists]);
        let tactic = StatelessDerTactic::new();
        let result = tactic
            .apply(&goal, &mut manager)
            .expect("DER should succeed");

        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert!(!goal_has_quantifiers(&goals[0], &manager));
            }
            _ => panic!("Expected SubGoals result"),
        }
    }
}
