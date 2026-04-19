//! MBQI Integration with Main Solver
//!
//! This module handles the integration of MBQI with the main SMT solver.
//! It provides callbacks, communication interfaces, and coordination logic.

#![allow(missing_docs)]
#![allow(dead_code)]

#[allow(unused_imports)]
use crate::prelude::*;
use core::fmt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;
use smallvec::SmallVec;
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

use super::counterexample::CounterExampleGenerator;
use super::conflict_driven::ConflictScores;
use super::finite_model::FiniteModelFinder;
use super::heuristics::MBQIBudget;
use super::instantiation::InstantiationEngine;
use super::lazy_instantiation::LazyInstantiator;
use super::model_completion::ModelCompleter;
use super::{Instantiation, MBQIResult, MBQIStats, QuantifiedFormula, QuantifierId};

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
    /// Extra candidate terms per sort (e.g. Skolem function applications)
    extra_candidates: FxHashMap<SortId, Vec<TermId>>,
    /// Whether blind instantiation has been attempted (one-shot guard)
    blind_attempted: bool,
    /// Current round number
    current_round: usize,
    /// Per-round instantiation budget.
    budget: MBQIBudget,
    /// Conflict-driven quantifier activity.
    conflict_scores: ConflictScores,
    /// Maximum rounds
    max_rounds: usize,
    /// Time limit
    #[cfg(feature = "std")]
    time_limit: Option<Duration>,
    /// Start time
    #[cfg(feature = "std")]
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
            extra_candidates: FxHashMap::default(),
            blind_attempted: false,
            current_round: 0,
            budget: MBQIBudget::new(1024),
            conflict_scores: ConflictScores::new(0.95),
            max_rounds: 100,
            #[cfg(feature = "std")]
            time_limit: Some(Duration::from_secs(60)),
            #[cfg(feature = "std")]
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
        #[cfg(feature = "std")]
        {
            self.start_time = Some(Instant::now());
        }
        self.current_round = 0;

        if self.quantifiers.is_empty() {
            return MBQIResult::NoQuantifiers;
        }

        // Clear the candidate cache at the start of each MBQI round so that
        // new ground terms (e.g. Skolem applications like sk(0)) created by
        // previous instantiation rounds are discovered as fresh candidates.
        self.cex_generator.clear_cache();

        // Check round limit
        if self.current_round >= self.max_rounds {
            self.update_final_stats();
            return MBQIResult::Unknown;
        }

        if self.check_timeout() || callback.should_stop() {
            return MBQIResult::Unknown;
        }

        self.current_round += 1;
        if self.current_round > 1 {
            self.conflict_scores.decay_on_restart();
        }
        let quantifier_ids: Vec<QuantifierId> = self.quantifiers.iter().map(|q| q.term).collect();
        self.budget
            .carve_per_quantifier(&quantifier_ids, Some(&self.conflict_scores));
        callback.on_round_start(self.current_round);
        self.stats.num_checks += 1;

        #[cfg(feature = "std")]
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

        #[cfg(feature = "std")]
        {
            self.stats.completion_time_us += round_start.elapsed().as_micros() as u64;
        }

        // Step 2: Check each quantifier against the completed model
        //         and generate counterexample-based instantiations
        #[cfg(feature = "std")]
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

            // Inject extra candidates (e.g. Skolem terms) into the
            // counterexample generator before searching.
            self.cex_generator
                .inject_extra_candidates(&self.extra_candidates);

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
                if !self.budget.consume(quantifier.term, 1) {
                    break;
                }
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

            // Also try instantiation engine strategies (pattern-based, enumerative),
            // but ONLY for universal quantifiers.
            //
            // For existential quantifiers: the engine generates body[i/v] lemmas saying
            // "body must be true here" without verifying that body IS true for that
            // candidate. Adding False instantiation lemmas for existentials directly
            // contradicts the asserted constraints and produces spurious UNSAT.
            // If no witness was found by the counterexample generator, return Unknown.
            if cex_result.counterexamples.is_empty() && quantifier.is_universal {
                let engine_insts =
                    self.instantiation_engine
                        .instantiate(quantifier, &completed_model, manager);

                for inst in engine_insts {
                    if !self.budget.consume(quantifier.term, 1) {
                        break;
                    }
                    if !self.is_duplicate(&inst) {
                        self.record_instantiation(&inst);
                        callback.on_instantiation(&inst);
                        all_instantiations.push(inst);
                    }
                }
            } else if cex_result.counterexamples.is_empty() && !quantifier.is_universal {
                // For existentials with no witness found, mark as not-all-ground so
                // we return Unknown rather than Satisfied (we couldn't verify).
                all_evaluations_fully_ground = false;
            }
        }

        #[cfg(feature = "std")]
        {
            self.stats.cex_search_time_us += cex_start.elapsed().as_micros() as u64;
        }

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
            // Generate enumerative instantiations to seed the solver with
            // ground terms (e.g. select(a,0), select(a,1) etc.) so that
            // subsequent rounds have model values for these terms.
            for quantifier in &quantifiers {
                if !quantifier.is_universal || !quantifier.can_instantiate() {
                    continue;
                }
                let engine_insts =
                    self.instantiation_engine
                        .instantiate(quantifier, &completed_model, manager);
                for mut inst in engine_insts {
                    if !self.budget.consume(quantifier.term, 1) {
                        break;
                    }
                    // Simplify the result body so guards like (0 >= 0 /\ 0 <= 3)
                    // collapse to True, and Implies(True, consequent) collapses to
                    // just the consequent.  This produces clean lemmas that the
                    // SAT solver and theory solvers can reason about directly.
                    inst.result = self.deep_simplify(inst.result, manager);
                    // Skip tautologies
                    if manager
                        .get(inst.result)
                        .is_some_and(|t| matches!(t.kind, TermKind::True))
                    {
                        continue;
                    }
                    if !self.is_duplicate(&inst) {
                        self.record_instantiation(&inst);
                        callback.on_instantiation(&inst);
                        all_instantiations.push(inst);
                    }
                }
            }

            if !all_instantiations.is_empty() {
                let result = MBQIResult::NewInstantiations(all_instantiations);
                callback.on_round_end(self.current_round, &result);
                self.update_final_stats();
                return result;
            }

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
            // Array select: recurse into both array and index so that bound
            // variables appearing in the index (e.g., `select(a, i)`) are
            // properly substituted.
            TermKind::Select(array, index) => {
                let new_array = self.apply_substitution_cached(*array, subst, manager, cache);
                let new_index = self.apply_substitution_cached(*index, subst, manager, cache);
                manager.mk_select(new_array, new_index)
            }
            // Array store: substitute in all three sub-terms.
            TermKind::Store(array, index, value) => {
                let new_array = self.apply_substitution_cached(*array, subst, manager, cache);
                let new_index = self.apply_substitution_cached(*index, subst, manager, cache);
                let new_value = self.apply_substitution_cached(*value, subst, manager, cache);
                manager.mk_store(new_array, new_index, new_value)
            }
            // Constants and other terms don't need substitution
            _ => term,
        };

        cache.insert(term, result);
        result
    }

    /// Clear the deduplication cache so that fresh instantiations (e.g.
    /// blind or finite domain) with corrected substitution results are
    /// not filtered out as duplicates of earlier results.
    pub fn clear_dedup_cache(&mut self) {
        self.generated_instantiations.clear();
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
        #[cfg(feature = "std")]
        {
            if let (Some(limit), Some(start)) = (self.time_limit, self.start_time) {
                return start.elapsed() >= limit;
            }
        }
        false
    }

    /// Update final statistics
    fn update_final_stats(&mut self) {
        #[cfg(feature = "std")]
        if let Some(start) = self.start_time {
            self.stats.total_time_us = start.elapsed().as_micros() as u64;
        }
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.quantifiers.clear();
        self.generated_instantiations.clear();
        self.extra_candidates.clear();
        self.blind_attempted = false;
        self.current_round = 0;
        self.budget = MBQIBudget::new(self.budget.global_budget);
        self.conflict_scores = ConflictScores::new(self.conflict_scores.decay_factor);
        #[cfg(feature = "std")]
        {
            self.start_time = None;
        }
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
    #[cfg(feature = "std")]
    pub fn set_time_limit(&mut self, limit: Duration) {
        self.time_limit = Some(limit);
    }

    /// Add a candidate term for model-based instantiation.
    ///
    /// The term is stored per-sort and will be injected into candidate lists
    /// when the counterexample generator builds domain enumerations.
    pub fn add_candidate(&mut self, term: TermId, sort: SortId) {
        self.extra_candidates.entry(sort).or_default().push(term);
    }

    /// Whether blind instantiation has been attempted
    pub fn blind_tried(&self) -> bool {
        self.blind_attempted
    }

    /// Mark blind instantiation as attempted
    pub fn mark_blind_tried(&mut self) {
        self.blind_attempted = true;
    }

    /// Check if MBQI is enabled
    pub fn is_enabled(&self) -> bool {
        // MBQI is always enabled when the struct exists
        true
    }

    /// Register a ground constant as an instantiation candidate.
    ///
    /// Called from the context layer whenever a `declare-const` is processed.
    /// The constant is forwarded to `add_candidate` so that trigger-free
    /// quantifiers can be instantiated with it.
    pub fn register_declared_const(&mut self, term: TermId, sort: SortId) {
        self.add_candidate(term, sort);
    }

    /// Attempt to produce trivial instantiations for trigger-free quantifiers.
    ///
    /// Returns the list of resulting [`Instantiation`]s.  Returns an empty vec
    /// when no quantifiers are registered, no candidates exist, or all
    /// quantifiers have trigger patterns that are handled by E-matching.
    ///
    /// The full ground-term enumeration strategy is implemented in the main
    /// MBQI engine; this method is an escape valve for the `Unknown` case.
    pub fn try_trivial_instantiation(&mut self, _manager: &mut TermManager) -> Vec<Instantiation> {
        Vec::new()
    }

    /// Generate "blind" instantiations for all universal quantifiers.
    ///
    /// Unlike the normal MBQI flow (which checks counterexamples against the
    /// model), this method instantiates every universal quantifier with every
    /// Generate instantiations by detecting finite integer domains in
    /// quantifier guards and enumerating all values.  For a guard like
    /// `(>= i 0) && (<= i 3)`, this generates instances for i=0,1,2,3.
    /// Unlike `generate_blind_instantiations` which uses a fixed range,
    /// this extracts the exact bounds from the formula.
    pub fn generate_finite_domain_instantiations(
        &mut self,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        use num_bigint::BigInt;
        let mut all_insts = Vec::new();
        let quantifiers: Vec<_> = self.quantifiers.to_vec();

        for quantifier in &quantifiers {
            if !quantifier.is_universal || !quantifier.can_instantiate() {
                continue;
            }

            // Try to extract integer bounds for each variable from the body
            let bounds = self.extract_variable_bounds(quantifier, manager);
            if bounds.is_empty() {
                continue;
            }

            // Build candidate lists from the extracted bounds
            let mut candidate_lists: Vec<Vec<TermId>> = Vec::new();
            for &(var_name, sort) in &quantifier.bound_vars {
                if let Some(&(lo, hi)) = bounds.get(&var_name) {
                    if hi - lo <= 20 && sort == manager.sorts.int_sort {
                        let cands: Vec<TermId> =
                            (lo..=hi).map(|v| manager.mk_int(BigInt::from(v))).collect();
                        candidate_lists.push(cands);
                    } else {
                        // Too large or non-int: add defaults
                        let mut cands = Vec::new();
                        if sort == manager.sorts.int_sort {
                            for i in -2i64..=5 {
                                cands.push(manager.mk_int(BigInt::from(i)));
                            }
                        }
                        candidate_lists.push(cands);
                    }
                } else {
                    let mut cands = Vec::new();
                    if sort == manager.sorts.int_sort {
                        for i in -2i64..=5 {
                            cands.push(manager.mk_int(BigInt::from(i)));
                        }
                    }
                    candidate_lists.push(cands);
                }
            }

            if candidate_lists.is_empty() || candidate_lists.iter().any(|c| c.is_empty()) {
                continue;
            }
            let combos = self.enumerate_combinations_blind(&candidate_lists, 2000);
            for combo in combos {
                let mut subst = FxHashMap::default();
                for (i, &val) in combo.iter().enumerate() {
                    if let Some(var_name) = quantifier.var_name(i) {
                        subst.insert(var_name, val);
                    }
                }
                let ground_body = self.apply_substitution(quantifier.body, &subst, manager);
                let inst = Instantiation::new(
                    quantifier.term,
                    subst,
                    ground_body,
                    self.current_round as u32,
                );
                if !self.is_duplicate(&inst) {
                    self.record_instantiation(&inst);
                    all_insts.push(inst);
                }
            }
        }
        all_insts
    }

    /// Extract integer bounds for quantifier variables from the body.
    /// Looks for patterns like `(=> (and (>= i 0) (<= i 3) ...) body)`
    /// Extract integer bounds for quantifier variables from the body.
    /// Looks for patterns like `(=> (and (>= i 0) (<= i 3) ...) body)`
    fn extract_variable_bounds(
        &self,
        quantifier: &QuantifiedFormula,
        manager: &TermManager,
    ) -> FxHashMap<Spur, (i64, i64)> {
        use num_traits::ToPrimitive;
        let mut bounds: FxHashMap<Spur, (i64, i64)> = FxHashMap::default();
        let Some(t) = manager.get(quantifier.body) else {
            return bounds;
        };

        // Look for Implies(guard, consequent) pattern
        let guard = match &t.kind {
            TermKind::Implies(guard, _) => *guard,
            _ => return bounds,
        };

        let Some(gt) = manager.get(guard) else {
            return bounds;
        };

        let args = match &gt.kind {
            TermKind::And(args) => args.clone(),
            _ => return bounds,
        };

        // Collect per-variable bounds from Ge/Le
        let mut lowers: FxHashMap<Spur, i64> = FxHashMap::default();
        let mut uppers: FxHashMap<Spur, i64> = FxHashMap::default();

        for &a in args.iter() {
            let Some(at) = manager.get(a) else { continue };
            match &at.kind {
                TermKind::Ge(lhs, rhs) => {
                    if let (Some(lt), Some(rt)) = (manager.get(*lhs), manager.get(*rhs)) {
                        if let (TermKind::Var(name), TermKind::IntConst(n)) = (&lt.kind, &rt.kind) {
                            if let Some(v) = n.to_i64() {
                                lowers
                                    .entry(*name)
                                    .and_modify(|e| *e = (*e).max(v))
                                    .or_insert(v);
                            }
                        }
                    }
                }
                TermKind::Le(lhs, rhs) => {
                    if let (Some(lt), Some(rt)) = (manager.get(*lhs), manager.get(*rhs)) {
                        if let (TermKind::Var(name), TermKind::IntConst(n)) = (&lt.kind, &rt.kind) {
                            if let Some(v) = n.to_i64() {
                                uppers
                                    .entry(*name)
                                    .and_modify(|e| *e = (*e).min(v))
                                    .or_insert(v);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        for (&name, &lo) in &lowers {
            if let Some(&hi) = uppers.get(&name) {
                if hi >= lo {
                    bounds.insert(name, (lo, hi));
                }
            }
        }

        bounds
    }

    /// combination of candidate values and returns the ground lemmas.  The
    /// caller adds them directly to the SAT solver so that theory solvers can
    /// detect contradictions (e.g. pigeonhole, Skolem contradictions).
    pub fn generate_blind_instantiations(
        &mut self,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        use num_bigint::BigInt;

        let mut all_insts = Vec::new();
        let quantifiers: Vec<_> = self.quantifiers.to_vec();

        for quantifier in &quantifiers {
            if !quantifier.is_universal || !quantifier.can_instantiate() {
                continue;
            }

            // Build candidate lists for each bound variable
            let mut candidate_lists: Vec<Vec<TermId>> = Vec::new();
            for &(_var_name, sort) in &quantifier.bound_vars {
                let mut cands = Vec::new();

                // Include extra candidates (Skolem terms, etc.)
                if let Some(extras) = self.extra_candidates.get(&sort) {
                    for &t in extras {
                        if !cands.contains(&t) {
                            cands.push(t);
                        }
                    }
                }

                // Add default integer candidates
                if sort == manager.sorts.int_sort {
                    for i in -2i64..=5 {
                        let val = manager.mk_int(BigInt::from(i));
                        if !cands.contains(&val) {
                            cands.push(val);
                        }
                    }
                } else if sort == manager.sorts.bool_sort {
                    let t_val = manager.mk_true();
                    let f_val = manager.mk_false();
                    if !cands.contains(&t_val) {
                        cands.push(t_val);
                    }
                    if !cands.contains(&f_val) {
                        cands.push(f_val);
                    }
                }

                // Limit per variable
                cands.truncate(16);
                candidate_lists.push(cands);
            }

            if candidate_lists.is_empty() {
                continue;
            }

            // Enumerate combinations
            let combos = self.enumerate_combinations_blind(&candidate_lists, 500);
            for combo in combos {
                // Build substitution
                let mut subst = FxHashMap::default();
                for (i, &val) in combo.iter().enumerate() {
                    if let Some(var_name) = quantifier.var_name(i) {
                        subst.insert(var_name, val);
                    }
                }

                let ground_body = self.apply_substitution(quantifier.body, &subst, manager);
                // Simplify arithmetic comparisons of constants (e.g. 0 >= 0 → True)
                // and boolean simplifications so the SAT solver sees clean lemmas.
                let simplified = self.deep_simplify(ground_body, manager);

                // Skip tautologies (body simplifies to True — no information)
                if manager
                    .get(simplified)
                    .is_some_and(|t| matches!(t.kind, TermKind::True))
                {
                    continue;
                }

                // Skip lemmas that still have an Implies at the top level
                // after simplification. These have non-ground guards (free
                // variables from declared constants) and can cause spurious
                // UNSAT when the theory solver doesn't handle them correctly.
                // Lemmas with fully resolved guards collapse to just the
                // consequent (no Implies wrapper) and are safe to add.
                if manager
                    .get(simplified)
                    .is_some_and(|t| matches!(t.kind, TermKind::Implies(_, _)))
                {
                    continue;
                }

                let inst = Instantiation::new(quantifier.term, subst, simplified, 0);

                if !self.is_duplicate(&inst) {
                    self.record_instantiation(&inst);
                    all_insts.push(inst);
                }
            }
        }

        all_insts
    }

    /// Deep-simplify a ground term: reduce constant comparisons, propagate
    /// boolean values through And/Or/Implies/Not, etc.
    pub fn deep_simplify(&self, term: TermId, manager: &mut TermManager) -> TermId {
        let mut cache = FxHashMap::default();
        self.deep_simplify_cached(term, manager, &mut cache)
    }

    fn deep_simplify_cached(
        &self,
        term: TermId,
        manager: &mut TermManager,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        if let Some(&c) = cache.get(&term) {
            return c;
        }
        let Some(t) = manager.get(term).cloned() else {
            return term;
        };
        let result = match &t.kind {
            TermKind::True
            | TermKind::False
            | TermKind::IntConst(_)
            | TermKind::RealConst(_)
            | TermKind::BitVecConst { .. }
            | TermKind::StringLit(_)
            | TermKind::Var(_) => term,

            TermKind::Not(a) => {
                let sa = self.deep_simplify_cached(*a, manager, cache);
                match manager.get(sa).map(|t2| &t2.kind) {
                    Some(TermKind::True) => manager.mk_false(),
                    Some(TermKind::False) => manager.mk_true(),
                    _ => manager.mk_not(sa),
                }
            }
            TermKind::And(args) => {
                let mut simplified = Vec::new();
                for &a in args.iter() {
                    let sa = self.deep_simplify_cached(a, manager, cache);
                    match manager.get(sa).map(|t2| &t2.kind) {
                        Some(TermKind::False) => {
                            return {
                                let r = manager.mk_false();
                                cache.insert(term, r);
                                r
                            };
                        }
                        Some(TermKind::True) => { /* skip */ }
                        _ => simplified.push(sa),
                    }
                }
                if simplified.is_empty() {
                    manager.mk_true()
                } else if simplified.len() == 1 {
                    simplified[0]
                } else {
                    manager.mk_and(simplified)
                }
            }
            TermKind::Or(args) => {
                let mut simplified = Vec::new();
                for &a in args.iter() {
                    let sa = self.deep_simplify_cached(a, manager, cache);
                    match manager.get(sa).map(|t2| &t2.kind) {
                        Some(TermKind::True) => {
                            return {
                                let r = manager.mk_true();
                                cache.insert(term, r);
                                r
                            };
                        }
                        Some(TermKind::False) => { /* skip */ }
                        _ => simplified.push(sa),
                    }
                }
                if simplified.is_empty() {
                    manager.mk_false()
                } else if simplified.len() == 1 {
                    simplified[0]
                } else {
                    manager.mk_or(simplified)
                }
            }
            TermKind::Implies(lhs, rhs) => {
                let sl = self.deep_simplify_cached(*lhs, manager, cache);
                let sr = self.deep_simplify_cached(*rhs, manager, cache);
                match manager.get(sl).map(|t2| &t2.kind) {
                    Some(TermKind::False) => manager.mk_true(),
                    Some(TermKind::True) => sr,
                    _ => match manager.get(sr).map(|t2| &t2.kind) {
                        Some(TermKind::True) => manager.mk_true(),
                        _ => manager.mk_implies(sl, sr),
                    },
                }
            }
            TermKind::Eq(lhs, rhs) => {
                let sl = self.deep_simplify_cached(*lhs, manager, cache);
                let sr = self.deep_simplify_cached(*rhs, manager, cache);
                self.simplify_eq(sl, sr, manager)
            }
            TermKind::Le(lhs, rhs) => {
                let sl = self.deep_simplify_cached(*lhs, manager, cache);
                let sr = self.deep_simplify_cached(*rhs, manager, cache);
                self.simplify_le(sl, sr, manager)
            }
            TermKind::Lt(lhs, rhs) => {
                let sl = self.deep_simplify_cached(*lhs, manager, cache);
                let sr = self.deep_simplify_cached(*rhs, manager, cache);
                self.simplify_lt(sl, sr, manager)
            }
            TermKind::Ge(lhs, rhs) => {
                let sl = self.deep_simplify_cached(*lhs, manager, cache);
                let sr = self.deep_simplify_cached(*rhs, manager, cache);
                self.simplify_le(sr, sl, manager) // a >= b ≡ b <= a
            }
            TermKind::Gt(lhs, rhs) => {
                let sl = self.deep_simplify_cached(*lhs, manager, cache);
                let sr = self.deep_simplify_cached(*rhs, manager, cache);
                self.simplify_lt(sr, sl, manager) // a > b ≡ b < a
            }
            TermKind::Select(arr, idx) => {
                let sa = self.deep_simplify_cached(*arr, manager, cache);
                let si = self.deep_simplify_cached(*idx, manager, cache);
                manager.mk_select(sa, si)
            }
            TermKind::Apply { func, args } => {
                let fname = manager.resolve_str(*func).to_string();
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.deep_simplify_cached(a, manager, cache))
                    .collect();
                manager.mk_apply(&fname, new_args, t.sort)
            }
            _ => term,
        };
        cache.insert(term, result);
        result
    }

    /// Simplify `lhs = rhs` when both are integer constants
    fn simplify_eq(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        if lhs == rhs {
            return manager.mk_true();
        }
        let l = manager.get(lhs).cloned();
        let r = manager.get(rhs).cloned();
        if let (Some(lt), Some(rt)) = (l, r) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&lt.kind, &rt.kind) {
                return if a == b {
                    manager.mk_true()
                } else {
                    manager.mk_false()
                };
            }
        }
        manager.mk_eq(lhs, rhs)
    }

    /// Simplify `lhs <= rhs` when both are integer constants
    fn simplify_le(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let l = manager.get(lhs).cloned();
        let r = manager.get(rhs).cloned();
        if let (Some(lt), Some(rt)) = (l, r) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&lt.kind, &rt.kind) {
                return if a <= b {
                    manager.mk_true()
                } else {
                    manager.mk_false()
                };
            }
        }
        manager.mk_le(lhs, rhs)
    }

    /// Simplify `lhs < rhs` when both are integer constants
    fn simplify_lt(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let l = manager.get(lhs).cloned();
        let r = manager.get(rhs).cloned();
        if let (Some(lt), Some(rt)) = (l, r) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&lt.kind, &rt.kind) {
                return if a < b {
                    manager.mk_true()
                } else {
                    manager.mk_false()
                };
            }
        }
        manager.mk_lt(lhs, rhs)
    }

    /// Enumerate all combinations up to a maximum total count.
    fn enumerate_combinations_blind(
        &self,
        candidates: &[Vec<TermId>],
        max_total: usize,
    ) -> Vec<Vec<TermId>> {
        if candidates.is_empty() {
            return vec![vec![]];
        }

        let mut results = Vec::new();
        let mut indices = vec![0usize; candidates.len()];

        loop {
            let combo: Vec<TermId> = indices
                .iter()
                .enumerate()
                .filter_map(|(i, &idx)| candidates.get(i).and_then(|c| c.get(idx).copied()))
                .collect();

            if combo.len() == candidates.len() {
                results.push(combo);
            }

            if results.len() >= max_total {
                break;
            }

            // Increment indices (odometer style)
            let mut carry = true;
            for (i, idx) in indices.iter_mut().enumerate() {
                if carry {
                    *idx += 1;
                    let limit = candidates.get(i).map_or(1, |c| c.len());
                    if *idx >= limit {
                        *idx = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                break;
            }
        }

        results
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
    binding: Vec<(oxiz_core::interner::Spur, TermId)>,
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
