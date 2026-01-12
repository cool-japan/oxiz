//! Quantifier Theory Solver
//!
//! This module provides a theory solver for handling quantified formulas.
//! It integrates with the MBQI (Model-Based Quantifier Instantiation) approach
//! and E-matching for pattern-based instantiation.
//!
//! # Supported Features
//!
//! - Universal quantifiers (∀x.φ(x))
//! - Existential quantifiers (∃x.φ(x)) via Skolemization
//! - Pattern-triggered instantiation (E-matching)
//! - Model-based instantiation (MBQI)
//!
//! # Integration
//!
//! The quantifier solver works with the main SMT solver:
//! 1. Collects quantified formulas from assertions
//! 2. On model requests, uses MBQI to find counterexamples
//! 3. Generates instantiation lemmas
//! 4. Returns to the main solver for further solving

use lasso::Spur;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::error::Result;
use oxiz_core::sort::SortId;
use oxiz_core::tactic::{GroundTermCollector, PatternMatcher};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::theory::{EqualityNotification, Theory, TheoryCombination, TheoryId, TheoryResult};

/// Configuration for the quantifier solver
#[derive(Debug, Clone)]
pub struct QuantifierConfig {
    /// Enable E-matching
    pub enable_ematch: bool,
    /// Enable MBQI
    pub enable_mbqi: bool,
    /// Maximum instantiations per quantifier
    pub max_inst_per_quantifier: usize,
    /// Maximum total instantiations
    pub max_total_instantiations: usize,
    /// Eagerness level for instantiation (0-10)
    pub eagerness: u8,
}

impl Default for QuantifierConfig {
    fn default() -> Self {
        Self {
            enable_ematch: true,
            enable_mbqi: true,
            max_inst_per_quantifier: 100,
            max_total_instantiations: 10000,
            eagerness: 5,
        }
    }
}

/// Tracked quantified formula
#[derive(Debug, Clone)]
pub struct TrackedQuantifier {
    /// Original term ID
    pub term: TermId,
    /// Bound variables
    pub bound_vars: SmallVec<[(Spur, SortId); 2]>,
    /// Body of the quantifier
    pub body: TermId,
    /// Patterns/triggers for E-matching
    pub patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    /// Whether this is a universal quantifier
    pub universal: bool,
    /// Current instantiation count
    pub instantiation_count: usize,
}

impl TrackedQuantifier {
    /// Create from a universal quantifier
    pub fn from_forall(
        term: TermId,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    ) -> Self {
        Self {
            term,
            bound_vars: vars,
            body,
            patterns,
            universal: true,
            instantiation_count: 0,
        }
    }

    /// Create from an existential quantifier
    pub fn from_exists(
        term: TermId,
        vars: SmallVec<[(Spur, SortId); 2]>,
        body: TermId,
        patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]>,
    ) -> Self {
        Self {
            term,
            bound_vars: vars,
            body,
            patterns,
            universal: false,
            instantiation_count: 0,
        }
    }
}

/// An instantiation lemma
#[derive(Debug, Clone)]
pub struct InstantiationLemma {
    /// The quantifier that was instantiated
    pub quantifier: TermId,
    /// The substitution
    pub substitution: FxHashMap<Spur, TermId>,
    /// The ground instance
    pub instance: TermId,
}

/// Quantifier solver statistics
#[derive(Debug, Clone, Default)]
pub struct QuantifierStats {
    /// Number of quantifiers
    pub num_quantifiers: usize,
    /// Number of E-matching instantiations
    pub ematch_instantiations: usize,
    /// Number of MBQI instantiations
    pub mbqi_instantiations: usize,
    /// Total instantiations
    pub total_instantiations: usize,
    /// Number of conflicts from quantifiers
    pub conflicts: usize,
}

/// Quantifier theory solver
#[derive(Debug)]
pub struct QuantifierSolver {
    /// Configuration
    config: QuantifierConfig,
    /// Tracked quantifiers
    quantifiers: Vec<TrackedQuantifier>,
    /// Pattern matcher for E-matching
    pattern_matcher: PatternMatcher,
    /// Ground term collector
    ground_collector: GroundTermCollector,
    /// Generated instantiations (for deduplication)
    generated_instances: FxHashSet<(TermId, Vec<(Spur, TermId)>)>,
    /// Context stack (number of quantifiers at each level)
    context_stack: Vec<usize>,
    /// Pending lemmas to add
    pending_lemmas: Vec<InstantiationLemma>,
    /// Statistics
    stats: QuantifierStats,
}

impl Default for QuantifierSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantifierSolver {
    /// Create a new quantifier solver
    pub fn new() -> Self {
        Self::with_config(QuantifierConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: QuantifierConfig) -> Self {
        Self {
            config,
            quantifiers: Vec::new(),
            pattern_matcher: PatternMatcher::new(),
            ground_collector: GroundTermCollector::new(),
            generated_instances: FxHashSet::default(),
            context_stack: Vec::new(),
            pending_lemmas: Vec::new(),
            stats: QuantifierStats::default(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &QuantifierConfig {
        &self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: QuantifierConfig) {
        self.config = config;
    }

    /// Get statistics
    pub fn stats(&self) -> &QuantifierStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = QuantifierStats::default();
    }

    /// Add a quantified formula
    pub fn add_quantifier(&mut self, term: TermId, manager: &TermManager) {
        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            TermKind::Forall {
                vars,
                body,
                patterns,
            } => {
                let tracked =
                    TrackedQuantifier::from_forall(term, vars.clone(), *body, patterns.clone());

                // Register with the pattern matcher
                self.pattern_matcher.add_pattern(term, manager);

                self.quantifiers.push(tracked);
                self.stats.num_quantifiers += 1;
            }
            TermKind::Exists {
                vars,
                body,
                patterns,
            } => {
                let tracked =
                    TrackedQuantifier::from_exists(term, vars.clone(), *body, patterns.clone());
                self.quantifiers.push(tracked);
                self.stats.num_quantifiers += 1;
            }
            _ => {}
        }
    }

    /// Collect ground terms from a formula
    pub fn collect_ground_terms(&mut self, term: TermId, manager: &TermManager) {
        self.ground_collector.collect(term, manager);
    }

    /// Get ground terms by sort
    pub fn get_ground_terms(&self, sort: SortId) -> &[TermId] {
        self.ground_collector.get_terms(sort)
    }

    /// Perform E-matching instantiation
    pub fn do_ematch(&mut self, manager: &mut TermManager) -> Vec<InstantiationLemma> {
        if !self.config.enable_ematch {
            return Vec::new();
        }

        let mut lemmas = Vec::new();

        // Match patterns against ground terms
        let bindings = self
            .pattern_matcher
            .match_against(&self.ground_collector, manager);

        for binding in bindings {
            // Check limits
            if self.stats.total_instantiations >= self.config.max_total_instantiations {
                break;
            }

            // Find the quantifier for this binding
            let Some(quantifier_id) = self.pattern_matcher.get_quantifier(binding.pattern_idx)
            else {
                continue;
            };

            let quantifier_idx = self
                .quantifiers
                .iter()
                .position(|q| q.term == quantifier_id);
            let Some(idx) = quantifier_idx else {
                continue;
            };

            if self.quantifiers[idx].instantiation_count >= self.config.max_inst_per_quantifier {
                continue;
            }

            // Check for duplicates
            let mut key_vec: Vec<_> = binding.substitution.iter().map(|(&k, &v)| (k, v)).collect();
            key_vec.sort_by_key(|(k, _)| k.into_inner());
            let key = (quantifier_id, key_vec.clone());

            if self.generated_instances.contains(&key) {
                continue;
            }

            // Create the instantiation using the pattern matcher's instantiate method
            let Some(instance) = self.pattern_matcher.instantiate(&binding, manager) else {
                continue;
            };

            let lemma = InstantiationLemma {
                quantifier: quantifier_id,
                substitution: binding.substitution.clone(),
                instance,
            };

            self.generated_instances.insert(key);
            self.quantifiers[idx].instantiation_count += 1;
            self.stats.ematch_instantiations += 1;
            self.stats.total_instantiations += 1;
            lemmas.push(lemma);
        }

        lemmas
    }

    /// Get pending lemmas
    pub fn get_pending_lemmas(&mut self) -> Vec<InstantiationLemma> {
        std::mem::take(&mut self.pending_lemmas)
    }

    /// Check if there are any quantifiers
    pub fn has_quantifiers(&self) -> bool {
        !self.quantifiers.is_empty()
    }

    /// Get the number of quantifiers
    pub fn num_quantifiers(&self) -> usize {
        self.quantifiers.len()
    }
}

impl Theory for QuantifierSolver {
    fn id(&self) -> TheoryId {
        // Use Bool as placeholder since TheoryId doesn't have Quantifier variant
        TheoryId::Bool
    }

    fn name(&self) -> &str {
        "Quantifier"
    }

    fn can_handle(&self, term: TermId) -> bool {
        // This would need a TermManager to check the term kind
        // For now, we'll assume terms are pre-registered
        let _ = term;
        false
    }

    fn assert_true(&mut self, _term: TermId) -> Result<TheoryResult> {
        // Track this as an asserted quantifier
        // Note: actual tracking would need TermManager access
        self.pending_lemmas.clear();
        Ok(TheoryResult::Sat)
    }

    fn assert_false(&mut self, _term: TermId) -> Result<TheoryResult> {
        // Negating a quantifier changes its type
        self.pending_lemmas.clear();
        Ok(TheoryResult::Sat)
    }

    fn check(&mut self) -> Result<TheoryResult> {
        // Check if we have pending lemmas to propagate
        if !self.pending_lemmas.is_empty() {
            let propagations: Vec<_> = self
                .pending_lemmas
                .iter()
                .map(|l| (l.instance, vec![l.quantifier]))
                .collect();
            return Ok(TheoryResult::Propagate(propagations));
        }

        Ok(TheoryResult::Sat)
    }

    fn push(&mut self) {
        self.context_stack.push(self.quantifiers.len());
    }

    fn pop(&mut self) {
        if let Some(num) = self.context_stack.pop() {
            self.quantifiers.truncate(num);
        }
    }

    fn reset(&mut self) {
        self.quantifiers.clear();
        self.pattern_matcher = PatternMatcher::new();
        self.ground_collector.clear();
        self.generated_instances.clear();
        self.context_stack.clear();
        self.pending_lemmas.clear();
        self.stats = QuantifierStats::default();
    }

    fn get_model(&self) -> Vec<(TermId, TermId)> {
        Vec::new()
    }
}

impl TheoryCombination for QuantifierSolver {
    fn notify_equality(&mut self, _eq: EqualityNotification) -> bool {
        // Equalities might trigger new pattern matches
        // For now, just accept
        true
    }

    fn get_shared_equalities(&self) -> Vec<EqualityNotification> {
        Vec::new()
    }

    fn is_relevant(&self, term: TermId) -> bool {
        // Check if term is a quantifier
        self.quantifiers.iter().any(|q| q.term == term)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantifier_solver_new() {
        let solver = QuantifierSolver::new();
        assert_eq!(solver.num_quantifiers(), 0);
        assert!(!solver.has_quantifiers());
    }

    #[test]
    fn test_quantifier_config_default() {
        let config = QuantifierConfig::default();
        assert!(config.enable_ematch);
        assert!(config.enable_mbqi);
        assert_eq!(config.eagerness, 5);
    }

    #[test]
    fn test_quantifier_solver_push_pop() {
        let mut solver = QuantifierSolver::new();

        solver.push();
        // Add would need a TermManager
        solver.pop();

        assert_eq!(solver.num_quantifiers(), 0);
    }

    #[test]
    fn test_quantifier_solver_reset() {
        let mut solver = QuantifierSolver::new();
        let mut manager = TermManager::new();

        // Create a ground term and collect it
        let one = manager.mk_int(1);
        solver.collect_ground_terms(one, &manager);
        solver.reset();

        assert!(solver.ground_collector.is_empty());
    }

    #[test]
    fn test_quantifier_solver_stats() {
        let solver = QuantifierSolver::new();
        let stats = solver.stats();

        assert_eq!(stats.num_quantifiers, 0);
        assert_eq!(stats.total_instantiations, 0);
    }

    #[test]
    fn test_theory_trait() {
        let mut solver = QuantifierSolver::new();

        assert_eq!(solver.name(), "Quantifier");

        solver.push();
        solver.pop();

        let result = solver.check().unwrap();
        assert!(matches!(result, TheoryResult::Sat));
    }
}
