//! MBQI Integration with Main Solver
//!
//! This module handles the integration of MBQI with the main SMT solver.
//! It provides callbacks, communication interfaces, and coordination logic.

#![allow(missing_docs)]
#![allow(dead_code)]

use lasso::Spur;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::fmt;
use std::time::{Duration, Instant};

use super::counterexample::CounterExampleGenerator;
use super::finite_model::FiniteModelFinder;
use super::instantiation::InstantiationEngine;
use super::lazy_instantiation::LazyInstantiator;
use super::model_completion::ModelCompleter;
use super::{Instantiation, MBQIResult, MBQIStats, QuantifiedFormula};

/// Callback trait for solver communication
pub trait SolverCallback: fmt::Debug {
    /// Called when new instantiations are generated
    fn on_instantiation(&mut self, inst: &Instantiation);

    /// Called when a conflict is detected
    fn on_conflict(&mut self, quantifier: TermId, reason: &[TermId]);

    /// Called when MBQI starts a new round
    fn on_round_start(&mut self, round: usize);

    /// Called when MBQI completes a round
    fn on_round_end(&mut self, round: usize, result: &MBQIResult);

    /// Check if solver should stop (e.g., timeout)
    fn should_stop(&self) -> bool;
}

/// MBQI integration manager
#[derive(Debug)]
pub struct MBQIIntegration {
    /// Model completer
    model_completer: ModelCompleter,
    /// Instantiation engine
    instantiation_engine: InstantiationEngine,
    /// Lazy instantiator
    lazy_instantiator: LazyInstantiator,
    /// Finite model finder
    finite_model_finder: FiniteModelFinder,
    /// Counterexample generator
    cex_generator: CounterExampleGenerator,
    /// Tracked quantifiers
    quantifiers: Vec<QuantifiedFormula>,
    /// Generated instantiations (for deduplication)
    generated_instantiations: FxHashMap<InstantiationKey, usize>,
    /// Current round number
    current_round: usize,
    /// Maximum rounds
    max_rounds: usize,
    /// Time limit
    time_limit: Option<Duration>,
    /// Start time
    start_time: Option<Instant>,
    /// Statistics
    stats: MBQIStats,
}

impl MBQIIntegration {
    /// Create a new MBQI integration
    pub fn new() -> Self {
        Self {
            model_completer: ModelCompleter::new(),
            instantiation_engine: InstantiationEngine::new(),
            lazy_instantiator: LazyInstantiator::new(),
            finite_model_finder: FiniteModelFinder::new(),
            cex_generator: CounterExampleGenerator::new(),
            quantifiers: Vec::new(),
            generated_instantiations: FxHashMap::default(),
            current_round: 0,
            max_rounds: 100,
            time_limit: Some(Duration::from_secs(60)),
            start_time: None,
            stats: MBQIStats::new(),
        }
    }

    /// Add a quantified formula
    pub fn add_quantifier(&mut self, term: TermId, manager: &TermManager) {
        let Some(t) = manager.get(term) else {
            return;
        };

        match &t.kind {
            oxiz_core::ast::TermKind::Forall { vars, body, .. } => {
                let bound_vars: SmallVec<[(Spur, SortId); 4]> = vars.iter().copied().collect();
                self.quantifiers
                    .push(QuantifiedFormula::new(term, bound_vars, *body, true));
                self.stats.num_quantifiers += 1;
            }
            oxiz_core::ast::TermKind::Exists { vars, body, .. } => {
                let bound_vars: SmallVec<[(Spur, SortId); 4]> = vars.iter().copied().collect();
                self.quantifiers
                    .push(QuantifiedFormula::new(term, bound_vars, *body, false));
                self.stats.num_quantifiers += 1;
            }
            _ => {}
        }
    }

    /// Run MBQI with a partial model implementing the Ge & de Moura (2009) algorithm.
    ///
    /// The loop:
    /// 1. Complete the partial model (fill in defaults, function interps, universes)
    /// 2. For each tracked quantifier, check against the completed model
    /// 3. If counterexamples found, generate instantiation lemmas and return them
    /// 4. If no counterexamples for any quantifier, the model satisfies all -- return SAT
    pub fn run(
        &mut self,
        partial_model: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
        callback: &mut dyn SolverCallback,
    ) -> MBQIResult {
        self.start_time = Some(Instant::now());
        self.current_round = 0;

        if self.quantifiers.is_empty() {
            return MBQIResult::NoQuantifiers;
        }

        // Check round limit
        if self.current_round >= self.max_rounds {
            self.update_final_stats();
            return MBQIResult::Unknown;
        }

        if self.check_timeout() || callback.should_stop() {
            return MBQIResult::Unknown;
        }

        self.current_round += 1;
        callback.on_round_start(self.current_round);
        self.stats.num_checks += 1;

        let round_start = Instant::now();

        // Step 1: Complete the model with proper Ge & de Moura completion
        let completed_model =
            match self
                .model_completer
                .complete(partial_model, &self.quantifiers, manager)
            {
                Ok(model) => {
                    self.stats.num_completions += 1;
                    model
                }
                Err(_) => {
                    callback.on_round_end(self.current_round, &MBQIResult::Unknown);
                    return MBQIResult::Unknown;
                }
            };

        self.stats.completion_time_us += round_start.elapsed().as_micros() as u64;

        // Step 2: Check each quantifier against the completed model
        //         and generate counterexample-based instantiations
        let cex_start = Instant::now();
        let mut all_instantiations = Vec::new();
        // Track whether ALL quantifier evaluations resolved to concrete
        // boolean values.  We can only claim Satisfied when every evaluation
        // across every quantifier was fully ground (i.e. concrete True).
        let mut all_evaluations_fully_ground = true;

        // Collect quantifiers first to avoid borrow checker issues
        let quantifiers: Vec<_> = self.quantifiers.to_vec();
        for quantifier in &quantifiers {
            if !quantifier.can_instantiate() {
                continue;
            }

            // Use the counterexample generator directly to find
            // assignments that falsify the quantifier body
            let cex_result = self
                .cex_generator
                .generate(quantifier, &completed_model, manager);

            if !cex_result.all_evaluations_ground {
                all_evaluations_fully_ground = false;
            }

            self.stats.num_counterexamples += cex_result.counterexamples.len();

            for cex in &cex_result.counterexamples {
                // Build the instantiation lemma: body[x1/v1, ..., xn/vn]
                let ground_body =
                    self.apply_substitution(quantifier.body, &cex.assignment, manager);

                let inst = cex.to_instantiation(ground_body);

                if !self.is_duplicate(&inst) {
                    self.record_instantiation(&inst);
                    callback.on_instantiation(&inst);
                    all_instantiations.push(inst);
                }
            }

            // Also try instantiation engine strategies (pattern-based, enumerative)
            if cex_result.counterexamples.is_empty() {
                let engine_insts =
                    self.instantiation_engine
                        .instantiate(quantifier, &completed_model, manager);

                for inst in engine_insts {
                    if !self.is_duplicate(&inst) {
                        self.record_instantiation(&inst);
                        callback.on_instantiation(&inst);
                        all_instantiations.push(inst);
                    }
                }
            }
        }

        self.stats.cex_search_time_us += cex_start.elapsed().as_micros() as u64;

        // Step 3: Check result
        if all_instantiations.is_empty() {
            if all_evaluations_fully_ground {
                // Every quantifier body evaluated to concrete True under every
                // candidate assignment.  The completed model genuinely satisfies
                // all quantifiers.
                let result = MBQIResult::Satisfied;
                callback.on_round_end(self.current_round, &result);
                self.update_final_stats();
                return result;
            }
            // Some evaluations produced symbolic residuals -- we cannot
            // conclusively say the model satisfies all quantifiers.
            // Conservatively return Unknown instead of the incorrect Satisfied.
            let result = MBQIResult::Unknown;
            callback.on_round_end(self.current_round, &result);
            self.update_final_stats();
            return result;
        }

        // Step 4: Check instantiation limit
        if self.stats.max_instantiations > 0
            && self.stats.total_instantiations >= self.stats.max_instantiations
        {
            let result = MBQIResult::InstantiationLimit;
            callback.on_round_end(self.current_round, &result);
            self.update_final_stats();
            return result;
        }

        // Return the new instantiations to the solver.
        // The solver will add them as lemmas and re-check SAT.
        // On the next call to MBQI, we'll re-complete the model.
        let result = MBQIResult::NewInstantiations(all_instantiations);
        callback.on_round_end(self.current_round, &result);
        self.update_final_stats();
        result
    }

    /// Apply substitution to a term (used for building instantiation lemmas)
    fn apply_substitution(
        &self,
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        let mut cache = FxHashMap::default();
        self.apply_substitution_cached(term, subst, manager, &mut cache)
    }

    fn apply_substitution_cached(
        &self,
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        if let Some(&cached) = cache.get(&term) {
            return cached;
        }

        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        let result = match &t.kind {
            TermKind::Var(name) => subst.get(name).copied().unwrap_or(term),
            TermKind::Not(arg) => {
                let new_arg = self.apply_substitution_cached(*arg, subst, manager, cache);
                manager.mk_not(new_arg)
            }
            TermKind::And(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_and(new_args)
            }
            TermKind::Or(args) => {
                let new_args: Vec<_> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_or(new_args)
            }
            TermKind::Implies(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_implies(new_lhs, new_rhs)
            }
            TermKind::Eq(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_eq(new_lhs, new_rhs)
            }
            TermKind::Lt(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_lt(new_lhs, new_rhs)
            }
            TermKind::Le(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_le(new_lhs, new_rhs)
            }
            TermKind::Gt(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_gt(new_lhs, new_rhs)
            }
            TermKind::Ge(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_ge(new_lhs, new_rhs)
            }
            TermKind::Add(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_add(new_args)
            }
            TermKind::Sub(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_sub(new_lhs, new_rhs)
            }
            TermKind::Mul(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_mul(new_args)
            }
            TermKind::Div(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_div(new_lhs, new_rhs)
            }
            TermKind::Mod(lhs, rhs) => {
                let new_lhs = self.apply_substitution_cached(*lhs, subst, manager, cache);
                let new_rhs = self.apply_substitution_cached(*rhs, subst, manager, cache);
                manager.mk_mod(new_lhs, new_rhs)
            }
            TermKind::Neg(arg) => {
                let new_arg = self.apply_substitution_cached(*arg, subst, manager, cache);
                manager.mk_neg(new_arg)
            }
            TermKind::Ite(cond, then_br, else_br) => {
                let new_cond = self.apply_substitution_cached(*cond, subst, manager, cache);
                let new_then = self.apply_substitution_cached(*then_br, subst, manager, cache);
                let new_else = self.apply_substitution_cached(*else_br, subst, manager, cache);
                manager.mk_ite(new_cond, new_then, new_else)
            }
            TermKind::Apply { func, args } => {
                let func_name = manager.resolve_str(*func).to_string();
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_apply(&func_name, new_args, t.sort)
            }
            // Constants and other terms don't need substitution
            _ => term,
        };

        cache.insert(term, result);
        result
    }

    /// Check if an instantiation is a duplicate
    fn is_duplicate(&self, inst: &Instantiation) -> bool {
        let key = InstantiationKey::from(inst);
        self.generated_instantiations.contains_key(&key)
    }

    /// Record an instantiation
    fn record_instantiation(&mut self, inst: &Instantiation) {
        let key = InstantiationKey::from(inst);
        let count = self.generated_instantiations.entry(key).or_insert(0);
        *count += 1;
        self.stats.total_instantiations += 1;
        self.stats.unique_instantiations = self.generated_instantiations.len();
    }

    /// Check for timeout
    fn check_timeout(&self) -> bool {
        if let (Some(limit), Some(start)) = (self.time_limit, self.start_time) {
            start.elapsed() >= limit
        } else {
            false
        }
    }

    /// Update final statistics
    fn update_final_stats(&mut self) {
        if let Some(start) = self.start_time {
            self.stats.total_time_us = start.elapsed().as_micros() as u64;
        }
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.quantifiers.clear();
        self.generated_instantiations.clear();
        self.current_round = 0;
        self.start_time = None;
        self.instantiation_engine.clear_caches();
        self.lazy_instantiator.clear();
    }

    /// Collect ground terms from trigger patterns
    pub fn collect_ground_terms(&mut self, _term: TermId, _manager: &TermManager) {
        // Stub implementation - would collect ground terms for E-matching
        // This is a placeholder for future E-matching integration
    }

    /// Check quantifiers with a given model
    pub fn check_with_model(
        &mut self,
        model: &FxHashMap<TermId, TermId>,
        manager: &mut TermManager,
    ) -> MBQIResult {
        // Use a no-op callback for this convenience method
        #[derive(Debug)]
        struct NoOpCallback;
        impl SolverCallback for NoOpCallback {
            fn on_instantiation(&mut self, _: &Instantiation) {}
            fn on_round_start(&mut self, _: usize) {}
            fn on_round_end(&mut self, _: usize, _: &MBQIResult) {}
            fn on_conflict(&mut self, _: TermId, _: &[TermId]) {}
            fn should_stop(&self) -> bool {
                false
            }
        }
        let mut callback = NoOpCallback;
        self.run(model, manager, &mut callback)
    }

    /// Get statistics
    pub fn stats(&self) -> &MBQIStats {
        &self.stats
    }

    /// Set maximum rounds
    pub fn set_max_rounds(&mut self, max: usize) {
        self.max_rounds = max;
    }

    /// Set time limit
    pub fn set_time_limit(&mut self, limit: Duration) {
        self.time_limit = Some(limit);
    }

    /// Add a candidate term for model-based instantiation
    pub fn add_candidate(&mut self, _term: TermId, _sort: SortId) {
        // Candidate terms are tracked for model-based instantiation
        // This is a placeholder - full implementation would store candidates
    }

    /// Check if MBQI is enabled
    pub fn is_enabled(&self) -> bool {
        // MBQI is always enabled when the struct exists
        true
    }
}

impl Default for MBQIIntegration {
    fn default() -> Self {
        Self::new()
    }
}

/// Key for instantiation deduplication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InstantiationKey {
    quantifier: TermId,
    binding: Vec<(lasso::Spur, TermId)>,
}

impl From<&Instantiation> for InstantiationKey {
    fn from(inst: &Instantiation) -> Self {
        let mut binding: Vec<_> = inst.substitution.iter().map(|(&k, &v)| (k, v)).collect();
        binding.sort_by_key(|(k, _)| *k);
        Self {
            quantifier: inst.quantifier,
            binding,
        }
    }
}

/// Default callback implementation (no-op)
#[derive(Debug)]
pub struct DefaultCallback {
    stop_requested: bool,
}

impl DefaultCallback {
    pub fn new() -> Self {
        Self {
            stop_requested: false,
        }
    }

    pub fn request_stop(&mut self) {
        self.stop_requested = true;
    }
}

impl Default for DefaultCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl SolverCallback for DefaultCallback {
    fn on_instantiation(&mut self, _inst: &Instantiation) {}
    fn on_conflict(&mut self, _quantifier: TermId, _reason: &[TermId]) {}
    fn on_round_start(&mut self, _round: usize) {}
    fn on_round_end(&mut self, _round: usize, _result: &MBQIResult) {}
    fn should_stop(&self) -> bool {
        self.stop_requested
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbqi_integration_creation() {
        let integration = MBQIIntegration::new();
        assert_eq!(integration.quantifiers.len(), 0);
        assert_eq!(integration.current_round, 0);
    }

    #[test]
    fn test_default_callback() {
        let mut callback = DefaultCallback::new();
        assert!(!callback.should_stop());
        callback.request_stop();
        assert!(callback.should_stop());
    }

    #[test]
    fn test_integration_clear() {
        let mut integration = MBQIIntegration::new();
        integration.current_round = 5;
        integration.clear();
        assert_eq!(integration.current_round, 0);
        assert_eq!(integration.quantifiers.len(), 0);
    }

    #[test]
    fn test_set_max_rounds() {
        let mut integration = MBQIIntegration::new();
        integration.set_max_rounds(50);
        assert_eq!(integration.max_rounds, 50);
    }

    #[test]
    fn test_set_time_limit() {
        let mut integration = MBQIIntegration::new();
        let limit = Duration::from_secs(30);
        integration.set_time_limit(limit);
        assert_eq!(integration.time_limit, Some(limit));
    }
}
