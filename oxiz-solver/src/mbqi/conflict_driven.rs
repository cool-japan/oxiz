//! Conflict-Driven Quantifier Instantiation (CDQI)
//!
//! When theory conflicts involve quantified formulas, this module extracts
//! relevant instances from conflict analysis and uses conflict clause
//! participation to guide which quantifier instances to add.
//!
//! Key ideas:
//! - Analyze conflict clauses to extract terms relevant to quantified formulas
//! - Score quantifier instances by how often they participate in conflicts
//! - Prioritize instances that are likely to resolve or contribute to conflicts
//! - Keep a relevance score that decays over time (activity-style aging)

#![allow(missing_docs)]
#![allow(dead_code)]

#[allow(unused_imports)]
use crate::prelude::*;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;
use oxiz_core::sort::SortId;
use smallvec::SmallVec;

use super::model_completion::CompletedModel;
use super::{Instantiation, InstantiationReason, QuantifiedFormula};

/// Configuration for conflict-driven instantiation
#[derive(Debug, Clone)]
pub struct CDQIConfig {
    /// Maximum instances to generate per conflict
    pub max_instances_per_conflict: usize,
    /// Relevance decay factor (0..1, higher = slower decay)
    pub relevance_decay: f64,
    /// Minimum relevance score to keep tracking
    pub min_relevance_threshold: f64,
    /// Maximum tracked instances (memory limit)
    pub max_tracked_instances: usize,
    /// Enable conflict generalization
    pub generalize_conflicts: bool,
    /// Bonus score for instances matching conflict variables
    pub conflict_variable_bonus: f64,
}

impl Default for CDQIConfig {
    fn default() -> Self {
        Self {
            max_instances_per_conflict: 10,
            relevance_decay: 0.95,
            min_relevance_threshold: 0.01,
            max_tracked_instances: 10000,
            generalize_conflicts: true,
            conflict_variable_bonus: 2.0,
        }
    }
}

/// A tracked instance with relevance scoring
#[derive(Debug, Clone)]
pub struct TrackedInstance {
    /// The quantifier this instance belongs to
    pub quantifier: TermId,
    /// The substitution
    pub substitution: FxHashMap<Spur, TermId>,
    /// The ground body
    pub result: TermId,
    /// Relevance score (activity-style, decays over time)
    pub relevance_score: f64,
    /// Number of conflicts this instance participated in
    pub conflict_count: u64,
    /// Generation when this instance was created
    pub creation_generation: u32,
    /// Last conflict generation this participated in
    pub last_conflict_generation: u32,
}

impl TrackedInstance {
    /// Create a new tracked instance
    pub fn new(
        quantifier: TermId,
        substitution: FxHashMap<Spur, TermId>,
        result: TermId,
        generation: u32,
    ) -> Self {
        Self {
            quantifier,
            substitution,
            result,
            relevance_score: 1.0,
            conflict_count: 0,
            creation_generation: generation,
            last_conflict_generation: 0,
        }
    }

    /// Bump the relevance score when participating in a conflict
    pub fn bump_relevance(&mut self, bonus: f64, generation: u32) {
        self.relevance_score += bonus;
        self.conflict_count += 1;
        self.last_conflict_generation = generation;
    }

    /// Apply decay to the relevance score
    pub fn decay(&mut self, factor: f64) {
        self.relevance_score *= factor;
    }

    /// Convert to an Instantiation
    pub fn to_instantiation(&self) -> Instantiation {
        Instantiation::with_reason(
            self.quantifier,
            self.substitution.clone(),
            self.result,
            self.creation_generation,
            InstantiationReason::Conflict,
        )
    }
}

/// Conflict analysis result
#[derive(Debug, Clone)]
pub struct ConflictAnalysis {
    /// Terms involved in the conflict
    pub conflict_terms: Vec<TermId>,
    /// Variables (bound variable names) that appear in the conflict
    pub conflict_variables: FxHashSet<Spur>,
    /// Ground values found in the conflict
    pub ground_values: FxHashMap<SortId, Vec<TermId>>,
    /// Quantifiers potentially related to the conflict
    pub related_quantifiers: Vec<TermId>,
}

/// Conflict-driven quantifier instantiation engine
#[derive(Debug)]
pub struct ConflictDrivenInstantiator {
    /// Configuration
    config: CDQIConfig,
    /// Tracked instances with relevance scores
    tracked_instances: Vec<TrackedInstance>,
    /// Index: quantifier -> tracked instance indices
    quantifier_index: FxHashMap<TermId, Vec<usize>>,
    /// Deduplication: (quantifier, sorted binding) -> index
    dedup: FxHashMap<(TermId, Vec<(Spur, TermId)>), usize>,
    /// Current generation counter
    generation: u32,
    /// Statistics
    stats: CDQIStats,
}

/// Statistics for conflict-driven instantiation
#[derive(Debug, Clone, Default)]
pub struct CDQIStats {
    /// Total conflicts analyzed
    pub conflicts_analyzed: u64,
    /// Total instances generated from conflicts
    pub instances_from_conflicts: u64,
    /// Total relevance bumps
    pub relevance_bumps: u64,
    /// Total instances pruned (below threshold)
    pub instances_pruned: u64,
    /// Total decay rounds
    pub decay_rounds: u64,
    /// Peak tracked instances
    pub peak_tracked: usize,
}

impl ConflictDrivenInstantiator {
    /// Create a new conflict-driven instantiator
    pub fn new(config: CDQIConfig) -> Self {
        Self {
            config,
            tracked_instances: Vec::new(),
            quantifier_index: FxHashMap::default(),
            dedup: FxHashMap::default(),
            generation: 0,
            stats: CDQIStats::default(),
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(CDQIConfig::default())
    }

    /// Analyze a conflict and generate relevant instantiations.
    ///
    /// When a theory conflict is detected, this method:
    /// 1. Extracts terms and ground values from the conflict clause
    /// 2. Identifies which quantifiers are relevant
    /// 3. Builds instantiations using conflict terms as witnesses
    /// 4. Bumps relevance of existing instances that match conflict terms
    pub fn analyze_conflict(
        &mut self,
        conflict_clause: &[TermId],
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        self.generation += 1;
        self.stats.conflicts_analyzed += 1;

        // Step 1: Analyze the conflict
        let analysis = self.extract_conflict_info(conflict_clause, quantifiers, manager);

        // Step 2: Bump relevance of existing instances that match
        self.bump_matching_instances(&analysis);

        // Step 3: Generate new instances from conflict
        let new_instances =
            self.generate_instances_from_conflict(&analysis, quantifiers, model, manager);

        // Step 4: Apply decay to all tracked instances
        self.apply_decay();

        // Step 5: Prune low-relevance instances
        self.prune_low_relevance();

        new_instances
    }

    /// Extract information from a conflict clause
    fn extract_conflict_info(
        &self,
        conflict_clause: &[TermId],
        quantifiers: &[QuantifiedFormula],
        manager: &TermManager,
    ) -> ConflictAnalysis {
        let mut analysis = ConflictAnalysis {
            conflict_terms: Vec::new(),
            conflict_variables: FxHashSet::default(),
            ground_values: FxHashMap::default(),
            related_quantifiers: Vec::new(),
        };

        // Collect all terms and ground values from the conflict
        let mut visited = FxHashSet::default();
        for &term in conflict_clause {
            self.collect_conflict_terms(term, &mut analysis, &mut visited, manager);
        }

        // Find quantifiers whose body terms overlap with conflict terms
        let conflict_set: FxHashSet<TermId> = analysis.conflict_terms.iter().copied().collect();
        for qf in quantifiers {
            if self.quantifier_overlaps_conflict(qf, &conflict_set, manager) {
                analysis.related_quantifiers.push(qf.term);
            }
        }

        analysis
    }

    /// Recursively collect terms from a conflict clause
    fn collect_conflict_terms(
        &self,
        term: TermId,
        analysis: &mut ConflictAnalysis,
        visited: &mut FxHashSet<TermId>,
        manager: &TermManager,
    ) {
        if visited.contains(&term) {
            return;
        }
        visited.insert(term);
        analysis.conflict_terms.push(term);

        let Some(t) = manager.get(term) else {
            return;
        };

        // Collect ground values by sort
        match &t.kind {
            TermKind::IntConst(_) | TermKind::RealConst(_) | TermKind::BitVecConst { .. } => {
                analysis
                    .ground_values
                    .entry(t.sort)
                    .or_default()
                    .push(term);
            }
            TermKind::True | TermKind::False => {
                analysis
                    .ground_values
                    .entry(t.sort)
                    .or_default()
                    .push(term);
            }
            TermKind::Var(name) => {
                analysis.conflict_variables.insert(*name);
            }
            _ => {}
        }

        // Recurse
        match &t.kind {
            TermKind::Not(a) | TermKind::Neg(a) => {
                self.collect_conflict_terms(*a, analysis, visited, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &a in args {
                    self.collect_conflict_terms(a, analysis, visited, manager);
                }
            }
            TermKind::Eq(l, r)
            | TermKind::Lt(l, r)
            | TermKind::Le(l, r)
            | TermKind::Gt(l, r)
            | TermKind::Ge(l, r)
            | TermKind::Implies(l, r)
            | TermKind::Sub(l, r)
            | TermKind::Div(l, r)
            | TermKind::Mod(l, r) => {
                self.collect_conflict_terms(*l, analysis, visited, manager);
                self.collect_conflict_terms(*r, analysis, visited, manager);
            }
            TermKind::Add(args) | TermKind::Mul(args) => {
                for &a in args.iter() {
                    self.collect_conflict_terms(a, analysis, visited, manager);
                }
            }
            TermKind::Ite(c, t_br, e_br) => {
                self.collect_conflict_terms(*c, analysis, visited, manager);
                self.collect_conflict_terms(*t_br, analysis, visited, manager);
                self.collect_conflict_terms(*e_br, analysis, visited, manager);
            }
            TermKind::Apply { args, .. } => {
                for &a in args.iter() {
                    self.collect_conflict_terms(a, analysis, visited, manager);
                }
            }
            TermKind::Select(arr, idx) => {
                self.collect_conflict_terms(*arr, analysis, visited, manager);
                self.collect_conflict_terms(*idx, analysis, visited, manager);
            }
            TermKind::Store(arr, idx, val) => {
                self.collect_conflict_terms(*arr, analysis, visited, manager);
                self.collect_conflict_terms(*idx, analysis, visited, manager);
                self.collect_conflict_terms(*val, analysis, visited, manager);
            }
            _ => {}
        }
    }

    /// Check if a quantifier's body overlaps with conflict terms
    fn quantifier_overlaps_conflict(
        &self,
        qf: &QuantifiedFormula,
        conflict_set: &FxHashSet<TermId>,
        manager: &TermManager,
    ) -> bool {
        // Check if any function symbols in the quantifier body appear in the conflict
        let body_funcs = self.collect_function_symbols(qf.body, manager);
        let conflict_funcs: FxHashSet<Spur> = conflict_set
            .iter()
            .flat_map(|&t| self.collect_function_symbols(t, manager))
            .collect();

        body_funcs.intersection(&conflict_funcs).next().is_some()
    }

    /// Collect function symbol names from a term
    fn collect_function_symbols(&self, term: TermId, manager: &TermManager) -> FxHashSet<Spur> {
        let mut symbols = FxHashSet::default();
        let mut visited = FxHashSet::default();
        self.collect_func_symbols_rec(term, &mut symbols, &mut visited, manager);
        symbols
    }

    fn collect_func_symbols_rec(
        &self,
        term: TermId,
        symbols: &mut FxHashSet<Spur>,
        visited: &mut FxHashSet<TermId>,
        manager: &TermManager,
    ) {
        if visited.contains(&term) {
            return;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return;
        };

        if let TermKind::Apply { func, args } = &t.kind {
            symbols.insert(*func);
            for &a in args.iter() {
                self.collect_func_symbols_rec(a, symbols, visited, manager);
            }
        }

        match &t.kind {
            TermKind::Not(a) | TermKind::Neg(a) => {
                self.collect_func_symbols_rec(*a, symbols, visited, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &a in args {
                    self.collect_func_symbols_rec(a, symbols, visited, manager);
                }
            }
            TermKind::Eq(l, r) | TermKind::Lt(l, r) | TermKind::Le(l, r) | TermKind::Implies(l, r) => {
                self.collect_func_symbols_rec(*l, symbols, visited, manager);
                self.collect_func_symbols_rec(*r, symbols, visited, manager);
            }
            _ => {}
        }
    }

    /// Bump relevance of tracked instances that match the conflict
    fn bump_matching_instances(&mut self, analysis: &ConflictAnalysis) {
        let conflict_set: FxHashSet<TermId> = analysis.conflict_terms.iter().copied().collect();
        let bonus = self.config.conflict_variable_bonus;
        let current_gen = self.generation;

        for inst in &mut self.tracked_instances {
            // Check if this instance's result term appears in the conflict
            if conflict_set.contains(&inst.result) {
                inst.bump_relevance(bonus, current_gen);
                self.stats.relevance_bumps += 1;
            }
            // Check if any substitution value appears in the conflict
            for &val in inst.substitution.values() {
                if conflict_set.contains(&val) {
                    inst.bump_relevance(bonus * 0.5, current_gen);
                    self.stats.relevance_bumps += 1;
                    break; // only bump once per instance for sub values
                }
            }
        }
    }

    /// Generate new instances from conflict analysis
    fn generate_instances_from_conflict(
        &mut self,
        analysis: &ConflictAnalysis,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        let mut new_instances = Vec::new();

        for qf in quantifiers {
            if !qf.can_instantiate() || !qf.is_universal {
                continue;
            }

            // Only instantiate quantifiers related to the conflict
            if !analysis.related_quantifiers.contains(&qf.term) {
                continue;
            }

            // Build assignments from conflict ground values
            let assignments = self.build_conflict_assignments(qf, analysis, model, manager);

            for assignment in assignments {
                if new_instances.len() >= self.config.max_instances_per_conflict {
                    break;
                }

                // Check deduplication
                let binding_key = Self::make_binding_key(&assignment, qf);
                let dedup_key = (qf.term, binding_key.clone());
                if self.dedup.contains_key(&dedup_key) {
                    continue;
                }

                let ground_body = self.apply_substitution(qf.body, &assignment, manager);

                // Skip tautologies
                if manager
                    .get(ground_body)
                    .is_some_and(|t| matches!(t.kind, TermKind::True))
                {
                    continue;
                }

                let inst = Instantiation::with_reason(
                    qf.term,
                    assignment.clone(),
                    ground_body,
                    self.generation,
                    InstantiationReason::Conflict,
                );

                // Track the instance
                let tracked = TrackedInstance::new(qf.term, assignment, ground_body, self.generation);
                let idx = self.tracked_instances.len();
                self.tracked_instances.push(tracked);
                self.quantifier_index
                    .entry(qf.term)
                    .or_default()
                    .push(idx);
                self.dedup.insert(dedup_key, idx);

                if self.tracked_instances.len() > self.stats.peak_tracked {
                    self.stats.peak_tracked = self.tracked_instances.len();
                }

                new_instances.push(inst);
                self.stats.instances_from_conflicts += 1;
            }
        }

        new_instances
    }

    /// Build assignments from conflict ground values
    fn build_conflict_assignments(
        &self,
        qf: &QuantifiedFormula,
        analysis: &ConflictAnalysis,
        model: &CompletedModel,
        _manager: &TermManager,
    ) -> Vec<FxHashMap<Spur, TermId>> {
        let mut assignments = Vec::new();

        // For each bound variable, collect candidate values from:
        // 1. Ground values in the conflict that match the sort
        // 2. Model values for the sort
        let mut candidates_per_var: Vec<Vec<TermId>> = Vec::new();

        for &(name, sort) in &qf.bound_vars {
            let mut cands = Vec::new();

            // Priority 1: values from conflict
            if let Some(conflict_vals) = analysis.ground_values.get(&sort) {
                cands.extend_from_slice(conflict_vals);
            }

            // Priority 2: values from model universe
            if let Some(universe) = model.universe(sort) {
                for &val in universe {
                    if !cands.contains(&val) {
                        cands.push(val);
                    }
                }
            }

            // Limit candidates
            cands.truncate(self.config.max_instances_per_conflict);
            candidates_per_var.push(cands);

            // Store the variable name for assignment building
            let _ = name;
        }

        // If any variable has no candidates, return empty
        if candidates_per_var.iter().any(|c| c.is_empty()) {
            return assignments;
        }

        // Enumerate combinations (limited)
        let max_combos = self.config.max_instances_per_conflict;
        let mut indices = vec![0usize; qf.bound_vars.len()];

        for _ in 0..max_combos {
            let mut assignment = FxHashMap::default();
            let mut valid = true;

            for (i, &idx) in indices.iter().enumerate() {
                if let Some(cands) = candidates_per_var.get(i) {
                    if let Some(&val) = cands.get(idx) {
                        let (name, _) = qf.bound_vars[i];
                        assignment.insert(name, val);
                    } else {
                        valid = false;
                        break;
                    }
                }
            }

            if valid && assignment.len() == qf.bound_vars.len() {
                assignments.push(assignment);
            }

            // Increment indices (odometer)
            let mut carry = true;
            for (i, idx) in indices.iter_mut().enumerate() {
                if carry {
                    *idx += 1;
                    let limit = candidates_per_var.get(i).map_or(1, |c| c.len());
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

        assignments
    }

    /// Make a sorted binding key for deduplication
    fn make_binding_key(
        assignment: &FxHashMap<Spur, TermId>,
        _qf: &QuantifiedFormula,
    ) -> Vec<(Spur, TermId)> {
        let mut key: Vec<_> = assignment.iter().map(|(&k, &v)| (k, v)).collect();
        key.sort_by_key(|(k, _)| *k);
        key
    }

    /// Apply substitution to a term
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
            TermKind::Not(a) => {
                let sa = self.apply_substitution_cached(*a, subst, manager, cache);
                manager.mk_not(sa)
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
            TermKind::Eq(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_eq(sl, sr)
            }
            TermKind::Implies(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_implies(sl, sr)
            }
            TermKind::Lt(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_lt(sl, sr)
            }
            TermKind::Le(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_le(sl, sr)
            }
            TermKind::Gt(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_gt(sl, sr)
            }
            TermKind::Ge(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_ge(sl, sr)
            }
            TermKind::Add(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_add(new_args)
            }
            TermKind::Mul(args) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_mul(new_args)
            }
            TermKind::Sub(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_sub(sl, sr)
            }
            TermKind::Div(l, r) => {
                let sl = self.apply_substitution_cached(*l, subst, manager, cache);
                let sr = self.apply_substitution_cached(*r, subst, manager, cache);
                manager.mk_div(sl, sr)
            }
            TermKind::Neg(a) => {
                let sa = self.apply_substitution_cached(*a, subst, manager, cache);
                manager.mk_neg(sa)
            }
            TermKind::Ite(c, t_br, e_br) => {
                let sc = self.apply_substitution_cached(*c, subst, manager, cache);
                let st = self.apply_substitution_cached(*t_br, subst, manager, cache);
                let se = self.apply_substitution_cached(*e_br, subst, manager, cache);
                manager.mk_ite(sc, st, se)
            }
            TermKind::Apply { func, args } => {
                let fname = manager.resolve_str(*func).to_string();
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_apply(&fname, new_args, t.sort)
            }
            TermKind::Select(arr, idx) => {
                let sa = self.apply_substitution_cached(*arr, subst, manager, cache);
                let si = self.apply_substitution_cached(*idx, subst, manager, cache);
                manager.mk_select(sa, si)
            }
            TermKind::Store(arr, idx, val) => {
                let sa = self.apply_substitution_cached(*arr, subst, manager, cache);
                let si = self.apply_substitution_cached(*idx, subst, manager, cache);
                let sv = self.apply_substitution_cached(*val, subst, manager, cache);
                manager.mk_store(sa, si, sv)
            }
            _ => term,
        };

        cache.insert(term, result);
        result
    }

    /// Apply decay to all tracked instances
    fn apply_decay(&mut self) {
        let factor = self.config.relevance_decay;
        for inst in &mut self.tracked_instances {
            inst.decay(factor);
        }
        self.stats.decay_rounds += 1;
    }

    /// Prune instances with relevance below threshold
    fn prune_low_relevance(&mut self) {
        let threshold = self.config.min_relevance_threshold;
        let max = self.config.max_tracked_instances;

        if self.tracked_instances.len() <= max {
            return;
        }

        // Remove low-relevance instances
        let before = self.tracked_instances.len();
        self.tracked_instances
            .retain(|inst| inst.relevance_score >= threshold);
        self.stats.instances_pruned += (before - self.tracked_instances.len()) as u64;

        // Rebuild indices
        self.rebuild_indices();
    }

    /// Rebuild internal indices after pruning
    fn rebuild_indices(&mut self) {
        self.quantifier_index.clear();
        self.dedup.clear();

        for (idx, inst) in self.tracked_instances.iter().enumerate() {
            self.quantifier_index
                .entry(inst.quantifier)
                .or_default()
                .push(idx);

            let mut binding: Vec<_> = inst.substitution.iter().map(|(&k, &v)| (k, v)).collect();
            binding.sort_by_key(|(k, _)| *k);
            self.dedup.insert((inst.quantifier, binding), idx);
        }
    }

    /// Get the top-N most relevant instances for a quantifier
    pub fn top_relevant_instances(
        &self,
        quantifier: TermId,
        n: usize,
    ) -> Vec<&TrackedInstance> {
        let Some(indices) = self.quantifier_index.get(&quantifier) else {
            return Vec::new();
        };

        let mut instances: Vec<&TrackedInstance> = indices
            .iter()
            .filter_map(|&idx| self.tracked_instances.get(idx))
            .collect();

        instances.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        instances.truncate(n);
        instances
    }

    /// Get statistics
    pub fn stats(&self) -> &CDQIStats {
        &self.stats
    }

    /// Get current generation
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Number of tracked instances
    pub fn num_tracked(&self) -> usize {
        self.tracked_instances.len()
    }

    /// Clear all state
    pub fn clear(&mut self) {
        self.tracked_instances.clear();
        self.quantifier_index.clear();
        self.dedup.clear();
        self.generation = 0;
    }
}

impl Default for ConflictDrivenInstantiator {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiz_core::interner::Key;

    fn make_qf(term_id: u32, body_id: u32, var_names: &[usize]) -> QuantifiedFormula {
        let bound_vars: SmallVec<[(Spur, SortId); 4]> = var_names
            .iter()
            .map(|&n| {
                (
                    Spur::try_from_usize(n).expect("valid spur"),
                    SortId::new(0),
                )
            })
            .collect();
        QuantifiedFormula::new(TermId::new(term_id), bound_vars, TermId::new(body_id), true)
    }

    #[test]
    fn test_cdqi_creation() {
        let cdqi = ConflictDrivenInstantiator::default_config();
        assert_eq!(cdqi.generation(), 0);
        assert_eq!(cdqi.num_tracked(), 0);
    }

    #[test]
    fn test_cdqi_config() {
        let config = CDQIConfig {
            max_instances_per_conflict: 5,
            relevance_decay: 0.9,
            ..Default::default()
        };
        let cdqi = ConflictDrivenInstantiator::new(config);
        assert_eq!(cdqi.config.max_instances_per_conflict, 5);
    }

    #[test]
    fn test_tracked_instance_creation() {
        let tracked = TrackedInstance::new(
            TermId::new(1),
            FxHashMap::default(),
            TermId::new(2),
            0,
        );
        assert_eq!(tracked.relevance_score, 1.0);
        assert_eq!(tracked.conflict_count, 0);
    }

    #[test]
    fn test_tracked_instance_bump() {
        let mut tracked = TrackedInstance::new(
            TermId::new(1),
            FxHashMap::default(),
            TermId::new(2),
            0,
        );
        tracked.bump_relevance(2.0, 1);
        assert_eq!(tracked.relevance_score, 3.0);
        assert_eq!(tracked.conflict_count, 1);
        assert_eq!(tracked.last_conflict_generation, 1);
    }

    #[test]
    fn test_tracked_instance_decay() {
        let mut tracked = TrackedInstance::new(
            TermId::new(1),
            FxHashMap::default(),
            TermId::new(2),
            0,
        );
        tracked.bump_relevance(9.0, 1); // score = 10.0
        tracked.decay(0.5);
        assert!((tracked.relevance_score - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_tracked_instance_to_instantiation() {
        let mut subst = FxHashMap::default();
        subst.insert(
            Spur::try_from_usize(1).expect("valid spur"),
            TermId::new(10),
        );
        let tracked = TrackedInstance::new(TermId::new(1), subst, TermId::new(2), 5);
        let inst = tracked.to_instantiation();
        assert_eq!(inst.quantifier, TermId::new(1));
        assert_eq!(inst.result, TermId::new(2));
        assert_eq!(inst.generation, 5);
        assert_eq!(inst.reason, InstantiationReason::Conflict);
    }

    #[test]
    fn test_cdqi_analyze_empty_conflict() {
        let mut cdqi = ConflictDrivenInstantiator::default_config();
        let mut manager = TermManager::new();
        let model = CompletedModel::new();

        let result = cdqi.analyze_conflict(&[], &[], &model, &mut manager);
        assert!(result.is_empty());
        assert_eq!(cdqi.stats.conflicts_analyzed, 1);
    }

    #[test]
    fn test_cdqi_generation_increments() {
        let mut cdqi = ConflictDrivenInstantiator::default_config();
        let mut manager = TermManager::new();
        let model = CompletedModel::new();

        let _ = cdqi.analyze_conflict(&[], &[], &model, &mut manager);
        assert_eq!(cdqi.generation(), 1);

        let _ = cdqi.analyze_conflict(&[], &[], &model, &mut manager);
        assert_eq!(cdqi.generation(), 2);
    }

    #[test]
    fn test_cdqi_stats_tracking() {
        let mut cdqi = ConflictDrivenInstantiator::default_config();
        let mut manager = TermManager::new();
        let model = CompletedModel::new();

        for _ in 0..5 {
            let _ = cdqi.analyze_conflict(&[], &[], &model, &mut manager);
        }

        assert_eq!(cdqi.stats().conflicts_analyzed, 5);
        assert_eq!(cdqi.stats().decay_rounds, 5);
    }

    #[test]
    fn test_cdqi_clear() {
        let mut cdqi = ConflictDrivenInstantiator::default_config();
        let mut manager = TermManager::new();
        let model = CompletedModel::new();

        let _ = cdqi.analyze_conflict(&[], &[], &model, &mut manager);
        assert_eq!(cdqi.generation(), 1);

        cdqi.clear();
        assert_eq!(cdqi.generation(), 0);
        assert_eq!(cdqi.num_tracked(), 0);
    }

    #[test]
    fn test_cdqi_conflict_analysis_with_ground_terms() {
        let mut cdqi = ConflictDrivenInstantiator::default_config();
        let mut manager = TermManager::new();
        let model = CompletedModel::new();

        // Create some ground terms in the conflict
        let int_val = manager.mk_int(num_bigint::BigInt::from(42));
        let conflict = vec![int_val];

        let result = cdqi.analyze_conflict(&conflict, &[], &model, &mut manager);
        // No quantifiers, so no instances
        assert!(result.is_empty());
        assert_eq!(cdqi.stats.conflicts_analyzed, 1);
    }

    #[test]
    fn test_cdqi_top_relevant_instances_empty() {
        let cdqi = ConflictDrivenInstantiator::default_config();
        let top = cdqi.top_relevant_instances(TermId::new(1), 5);
        assert!(top.is_empty());
    }

    #[test]
    fn test_conflict_analysis_struct() {
        let analysis = ConflictAnalysis {
            conflict_terms: vec![TermId::new(1), TermId::new(2)],
            conflict_variables: FxHashSet::default(),
            ground_values: FxHashMap::default(),
            related_quantifiers: vec![TermId::new(10)],
        };
        assert_eq!(analysis.conflict_terms.len(), 2);
        assert_eq!(analysis.related_quantifiers.len(), 1);
    }
}
