//! Counter-example Generation and Refinement
//!
//! This module implements counterexample generation for MBQI. A counterexample
//! is an assignment to quantified variables that falsifies the quantifier body
//! under the current model.
//!
//! For a universal quantifier ∀x.φ(x), a counterexample is an assignment σ such
//! that ¬φ(σ(x)) holds in the current model.
//!
//! # Strategy
//!
//! 1. **Model Evaluation**: Evaluate quantifier body under candidate assignments
//! 2. **Satisfiability Checking**: Use auxiliary solver to find counterexamples
//! 3. **Conflict Analysis**: When no counterexamples exist, analyze why
//! 4. **Refinement**: Use counterexamples to refine the search space

use lasso::Spur;
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::fmt;
use std::time::{Duration, Instant};

use super::model_completion::CompletedModel;
use super::{Instantiation, InstantiationReason, QuantifiedFormula};

/// A counter-example to a quantified formula
#[derive(Debug, Clone)]
pub struct CounterExample {
    /// The quantifier this is a counterexample for
    pub quantifier: TermId,
    /// Assignment to bound variables
    pub assignment: FxHashMap<Spur, TermId>,
    /// Witness terms (the concrete values assigned)
    pub witnesses: Vec<TermId>,
    /// Evaluation of the body under this assignment
    pub body_value: Option<TermId>,
    /// Quality score (higher = better counterexample)
    pub quality: f64,
    /// Generation at which this was found
    pub generation: u32,
}

impl CounterExample {
    /// Create a new counter-example
    pub fn new(
        quantifier: TermId,
        assignment: FxHashMap<Spur, TermId>,
        witnesses: Vec<TermId>,
        generation: u32,
    ) -> Self {
        Self {
            quantifier,
            assignment,
            witnesses,
            body_value: None,
            quality: 1.0,
            generation,
        }
    }

    /// Convert to an instantiation
    pub fn to_instantiation(&self, result: TermId) -> Instantiation {
        Instantiation::with_reason(
            self.quantifier,
            self.assignment.clone(),
            result,
            self.generation,
            InstantiationReason::Conflict,
        )
    }

    /// Calculate quality score based on term complexity
    pub fn calculate_quality(&mut self, manager: &TermManager) {
        let mut total_size = 0;
        let mut num_constants = 0;

        for &witness in &self.witnesses {
            let size = self.term_size(witness, manager);
            total_size += size;

            if self.is_constant(witness, manager) {
                num_constants += 1;
            }
        }

        // Prefer simpler terms (smaller size)
        let size_factor = 1.0 / (1.0 + total_size as f64);
        // Prefer more constants (ground terms)
        let const_factor = 1.0 + (num_constants as f64 / self.witnesses.len().max(1) as f64);

        self.quality = size_factor * const_factor;
    }

    fn term_size(&self, term: TermId, manager: &TermManager) -> usize {
        let mut visited = FxHashSet::default();
        self.term_size_rec(term, manager, &mut visited)
    }

    fn term_size_rec(
        &self,
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) -> usize {
        if visited.contains(&term) {
            return 0;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return 1;
        };

        let children_size = match &t.kind {
            TermKind::And(args) | TermKind::Or(args) => args
                .iter()
                .map(|&arg| self.term_size_rec(arg, manager, visited))
                .sum(),
            TermKind::Not(arg) | TermKind::Neg(arg) => self.term_size_rec(*arg, manager, visited),
            TermKind::Eq(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                self.term_size_rec(*lhs, manager, visited)
                    + self.term_size_rec(*rhs, manager, visited)
            }
            _ => 0,
        };

        1 + children_size
    }

    fn is_constant(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };

        matches!(
            t.kind,
            TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
        )
    }
}

/// Counter-example generator
#[derive(Debug)]
pub struct CounterExampleGenerator {
    /// Maximum number of counterexamples to generate per quantifier
    max_cex_per_quantifier: usize,
    /// Maximum number of candidates to try per variable
    max_candidates_per_var: usize,
    /// Maximum total search time
    max_search_time: Duration,
    /// Current generation bound for term selection
    generation_bound: u32,
    /// Statistics
    stats: CexStats,
    /// Candidate cache
    candidate_cache: FxHashMap<SortId, Vec<TermId>>,
}

impl CounterExampleGenerator {
    /// Create a new counterexample generator
    pub fn new() -> Self {
        Self {
            max_cex_per_quantifier: 5,
            max_candidates_per_var: 10,
            max_search_time: Duration::from_secs(1),
            generation_bound: 0,
            stats: CexStats::default(),
            candidate_cache: FxHashMap::default(),
        }
    }

    /// Create with custom limits
    pub fn with_limits(max_cex: usize, max_candidates: usize, max_time: Duration) -> Self {
        let mut generator = Self::new();
        generator.max_cex_per_quantifier = max_cex;
        generator.max_candidates_per_var = max_candidates;
        generator.max_search_time = max_time;
        generator
    }

    /// Generate counterexamples for a quantifier
    pub fn generate(
        &mut self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<CounterExample> {
        let start_time = Instant::now();
        let mut counterexamples = Vec::new();
        self.stats.num_searches += 1;

        // Build candidate lists for each bound variable
        let candidates = self.build_candidate_lists(&quantifier.bound_vars, model, manager);

        // Enumerate combinations of candidates
        let combinations = self.enumerate_combinations(
            &candidates,
            self.max_candidates_per_var,
            self.max_cex_per_quantifier * 20, // Generate more combinations than we need
        );

        self.stats.num_combinations_tried += combinations.len();

        for combo in combinations {
            if start_time.elapsed() > self.max_search_time {
                self.stats.num_timeouts += 1;
                break;
            }

            if counterexamples.len() >= self.max_cex_per_quantifier {
                break;
            }

            // Build assignment from combination
            let mut assignment = FxHashMap::default();
            for (i, &candidate) in combo.iter().enumerate() {
                if let Some(var_name) = quantifier.var_name(i) {
                    assignment.insert(var_name, candidate);
                }
            }

            // Apply substitution and evaluate
            let substituted = self.apply_substitution(quantifier.body, &assignment, manager);
            let evaluated = self.evaluate_under_model(substituted, model, manager);

            // Check if this is a counterexample
            if self.is_counterexample(evaluated, quantifier.is_universal, manager) {
                let mut cex =
                    CounterExample::new(quantifier.term, assignment, combo, model.generation);
                cex.body_value = Some(evaluated);
                cex.calculate_quality(manager);
                counterexamples.push(cex);
                self.stats.num_counterexamples_found += 1;
            }
        }

        // Sort by quality (best first)
        counterexamples.sort_by(|a, b| {
            b.quality
                .partial_cmp(&a.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max
        counterexamples.truncate(self.max_cex_per_quantifier);

        self.stats.total_time += start_time.elapsed();

        counterexamples
    }

    /// Build candidate lists for bound variables
    fn build_candidate_lists(
        &mut self,
        bound_vars: &[(Spur, SortId)],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Vec<TermId>> {
        let mut result = Vec::new();

        for &(_var_name, sort) in bound_vars {
            // Check cache first
            if let Some(cached) = self.candidate_cache.get(&sort) {
                result.push(cached.clone());
                continue;
            }

            let mut candidates = Vec::new();

            // Strategy 1: Use values from the universe (for uninterpreted sorts)
            if let Some(universe) = model.universe(sort) {
                candidates.extend_from_slice(universe);
            }

            // Strategy 2: Use values from the model
            for (&term, &value) in &model.assignments {
                if let Some(t) = manager.get(term)
                    && t.sort == sort
                    && !candidates.contains(&value)
                {
                    candidates.push(value);
                }
            }

            // Strategy 3: Add default values based on sort
            self.add_default_candidates(sort, &mut candidates, manager);

            // Limit candidates
            candidates.truncate(self.max_candidates_per_var);

            // Cache for future use
            self.candidate_cache.insert(sort, candidates.clone());

            result.push(candidates);
        }

        result
    }

    /// Add default candidate values for a sort
    fn add_default_candidates(
        &self,
        sort: SortId,
        candidates: &mut Vec<TermId>,
        manager: &mut TermManager,
    ) {
        if sort == manager.sorts.int_sort {
            // Add small integers
            for i in -2..=5 {
                let val = manager.mk_int(BigInt::from(i));
                if !candidates.contains(&val) {
                    candidates.push(val);
                }
            }
        } else if sort == manager.sorts.bool_sort {
            let true_val = manager.mk_true();
            let false_val = manager.mk_false();
            if !candidates.contains(&true_val) {
                candidates.push(true_val);
            }
            if !candidates.contains(&false_val) {
                candidates.push(false_val);
            }
        }
    }

    /// Enumerate combinations of candidates
    fn enumerate_combinations(
        &self,
        candidates: &[Vec<TermId>],
        max_per_dim: usize,
        max_total: usize,
    ) -> Vec<Vec<TermId>> {
        if candidates.is_empty() {
            return vec![vec![]];
        }

        let mut results = Vec::new();
        let mut indices = vec![0usize; candidates.len()];

        loop {
            // Build current combination
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

            // Increment indices (like odometer)
            let mut carry = true;
            for (i, idx) in indices.iter_mut().enumerate() {
                if carry {
                    *idx += 1;
                    let limit = candidates.get(i).map_or(1, |c| c.len().min(max_per_dim));
                    if *idx >= limit {
                        *idx = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                // Overflow - tried all combinations
                break;
            }
        }

        results
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
            TermKind::Var(name) => {
                // Check if this variable should be substituted
                subst.get(name).copied().unwrap_or(term)
            }
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

    /// Evaluate a term under a model
    fn evaluate_under_model(
        &self,
        term: TermId,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> TermId {
        let mut cache = FxHashMap::default();
        self.evaluate_under_model_cached(term, model, manager, &mut cache)
    }

    fn evaluate_under_model_cached(
        &self,
        term: TermId,
        model: &CompletedModel,
        manager: &mut TermManager,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        if let Some(&cached) = cache.get(&term) {
            return cached;
        }

        // Check if we have a direct model value
        if let Some(val) = model.eval(term) {
            cache.insert(term, val);
            return val;
        }

        let Some(t) = manager.get(term).cloned() else {
            return term;
        };

        let result = match &t.kind {
            TermKind::True | TermKind::False | TermKind::IntConst(_) | TermKind::RealConst(_) => {
                term
            }
            TermKind::Not(arg) => {
                let eval_arg = self.evaluate_under_model_cached(*arg, model, manager, cache);
                if let Some(arg_t) = manager.get(eval_arg) {
                    match arg_t.kind {
                        TermKind::True => manager.mk_false(),
                        TermKind::False => manager.mk_true(),
                        _ => manager.mk_not(eval_arg),
                    }
                } else {
                    manager.mk_not(eval_arg)
                }
            }
            TermKind::And(args) => {
                for &arg in args.iter() {
                    let eval_arg = self.evaluate_under_model_cached(arg, model, manager, cache);
                    if let Some(arg_t) = manager.get(eval_arg)
                        && matches!(arg_t.kind, TermKind::False)
                    {
                        cache.insert(term, manager.mk_false());
                        return manager.mk_false();
                    }
                }
                manager.mk_true()
            }
            TermKind::Or(args) => {
                for &arg in args.iter() {
                    let eval_arg = self.evaluate_under_model_cached(arg, model, manager, cache);
                    if let Some(arg_t) = manager.get(eval_arg)
                        && matches!(arg_t.kind, TermKind::True)
                    {
                        cache.insert(term, manager.mk_true());
                        return manager.mk_true();
                    }
                }
                manager.mk_false()
            }
            TermKind::Eq(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model_cached(*lhs, model, manager, cache);
                let eval_rhs = self.evaluate_under_model_cached(*rhs, model, manager, cache);
                self.eval_eq(eval_lhs, eval_rhs, manager)
            }
            TermKind::Lt(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model_cached(*lhs, model, manager, cache);
                let eval_rhs = self.evaluate_under_model_cached(*rhs, model, manager, cache);
                self.eval_lt(eval_lhs, eval_rhs, manager)
            }
            TermKind::Le(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model_cached(*lhs, model, manager, cache);
                let eval_rhs = self.evaluate_under_model_cached(*rhs, model, manager, cache);
                self.eval_le(eval_lhs, eval_rhs, manager)
            }
            TermKind::Gt(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model_cached(*lhs, model, manager, cache);
                let eval_rhs = self.evaluate_under_model_cached(*rhs, model, manager, cache);
                self.eval_gt(eval_lhs, eval_rhs, manager)
            }
            TermKind::Ge(lhs, rhs) => {
                let eval_lhs = self.evaluate_under_model_cached(*lhs, model, manager, cache);
                let eval_rhs = self.evaluate_under_model_cached(*rhs, model, manager, cache);
                self.eval_ge(eval_lhs, eval_rhs, manager)
            }
            TermKind::Add(args) => {
                let eval_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.evaluate_under_model_cached(a, model, manager, cache))
                    .collect();
                self.eval_add(&eval_args, manager)
            }
            TermKind::Mul(args) => {
                let eval_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.evaluate_under_model_cached(a, model, manager, cache))
                    .collect();
                self.eval_mul(&eval_args, manager)
            }
            _ => {
                // For complex terms, try simplification
                manager.simplify(term)
            }
        };

        cache.insert(term, result);
        result
    }

    /// Evaluate equality
    fn eval_eq(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        if lhs == rhs {
            return manager.mk_true();
        }

        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            match (&l.kind, &r.kind) {
                (TermKind::IntConst(a), TermKind::IntConst(b)) => {
                    if a == b {
                        manager.mk_true()
                    } else {
                        manager.mk_false()
                    }
                }
                (TermKind::RealConst(a), TermKind::RealConst(b)) => {
                    if a == b {
                        manager.mk_true()
                    } else {
                        manager.mk_false()
                    }
                }
                (TermKind::True, TermKind::True) | (TermKind::False, TermKind::False) => {
                    manager.mk_true()
                }
                (TermKind::True, TermKind::False) | (TermKind::False, TermKind::True) => {
                    manager.mk_false()
                }
                _ => manager.mk_eq(lhs, rhs),
            }
        } else {
            manager.mk_eq(lhs, rhs)
        }
    }

    /// Evaluate less-than
    fn eval_lt(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind) {
                if a < b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
            if let (TermKind::RealConst(a), TermKind::RealConst(b)) = (&l.kind, &r.kind) {
                if a < b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
        }

        manager.mk_lt(lhs, rhs)
    }

    /// Evaluate less-than-or-equal
    fn eval_le(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        let lhs_t = manager.get(lhs);
        let rhs_t = manager.get(rhs);

        if let (Some(l), Some(r)) = (lhs_t, rhs_t) {
            if let (TermKind::IntConst(a), TermKind::IntConst(b)) = (&l.kind, &r.kind) {
                if a <= b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
            if let (TermKind::RealConst(a), TermKind::RealConst(b)) = (&l.kind, &r.kind) {
                if a <= b {
                    return manager.mk_true();
                } else {
                    return manager.mk_false();
                }
            }
        }

        manager.mk_le(lhs, rhs)
    }

    /// Evaluate greater-than
    fn eval_gt(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        self.eval_lt(rhs, lhs, manager)
    }

    /// Evaluate greater-than-or-equal
    fn eval_ge(&self, lhs: TermId, rhs: TermId, manager: &mut TermManager) -> TermId {
        self.eval_le(rhs, lhs, manager)
    }

    /// Evaluate addition
    fn eval_add(&self, args: &[TermId], manager: &mut TermManager) -> TermId {
        let mut result = BigInt::from(0);
        let mut all_ints = true;

        for &arg in args {
            if let Some(arg_t) = manager.get(arg) {
                if let TermKind::IntConst(val) = &arg_t.kind {
                    result += val;
                } else {
                    all_ints = false;
                    break;
                }
            } else {
                all_ints = false;
                break;
            }
        }

        if all_ints {
            manager.mk_int(result)
        } else {
            manager.mk_add(args.iter().copied())
        }
    }

    /// Evaluate multiplication
    fn eval_mul(&self, args: &[TermId], manager: &mut TermManager) -> TermId {
        let mut result = BigInt::from(1);
        let mut all_ints = true;

        for &arg in args {
            if let Some(arg_t) = manager.get(arg) {
                if let TermKind::IntConst(val) = &arg_t.kind {
                    result *= val;
                } else {
                    all_ints = false;
                    break;
                }
            } else {
                all_ints = false;
                break;
            }
        }

        if all_ints {
            manager.mk_int(result)
        } else {
            manager.mk_mul(args.iter().copied())
        }
    }

    /// Check if an evaluated term is a counterexample
    fn is_counterexample(
        &self,
        evaluated: TermId,
        is_universal: bool,
        manager: &TermManager,
    ) -> bool {
        let Some(eval_t) = manager.get(evaluated) else {
            return false;
        };

        if is_universal {
            // For ∀x.φ(x), a counterexample is when φ(x) = false
            matches!(eval_t.kind, TermKind::False)
        } else {
            // For ∃x.φ(x), a counterexample is when φ(x) = true
            matches!(eval_t.kind, TermKind::True)
        }
    }

    /// Set generation bound for candidate selection
    pub fn set_generation_bound(&mut self, bound: u32) {
        self.generation_bound = bound;
    }

    /// Clear the candidate cache
    pub fn clear_cache(&mut self) {
        self.candidate_cache.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &CexStats {
        &self.stats
    }
}

impl Default for CounterExampleGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Refinement strategy for narrowing the search space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementStrategy {
    /// No refinement
    None,
    /// Block found counterexamples
    BlockCounterexamples,
    /// Learn from conflicts
    ConflictLearning,
    /// Generalize from counterexamples
    Generalization,
}

/// Statistics for counterexample generation
#[derive(Debug, Clone, Default)]
pub struct CexStats {
    /// Number of search attempts
    pub num_searches: usize,
    /// Number of counterexamples found
    pub num_counterexamples_found: usize,
    /// Number of combinations tried
    pub num_combinations_tried: usize,
    /// Number of timeouts
    pub num_timeouts: usize,
    /// Total time spent
    pub total_time: Duration,
}

impl fmt::Display for CexStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Counterexample Statistics:")?;
        writeln!(f, "  Searches: {}", self.num_searches)?;
        writeln!(f, "  CEX found: {}", self.num_counterexamples_found)?;
        writeln!(f, "  Combinations tried: {}", self.num_combinations_tried)?;
        writeln!(f, "  Timeouts: {}", self.num_timeouts)?;
        writeln!(f, "  Total time: {:.2}ms", self.total_time.as_millis())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counterexample_creation() {
        let cex = CounterExample::new(TermId::new(1), FxHashMap::default(), vec![], 0);
        assert_eq!(cex.quantifier, TermId::new(1));
        assert_eq!(cex.quality, 1.0);
    }

    #[test]
    fn test_cex_generator_creation() {
        let generator = CounterExampleGenerator::new();
        assert_eq!(generator.max_cex_per_quantifier, 5);
        assert_eq!(generator.max_candidates_per_var, 10);
    }

    #[test]
    fn test_cex_generator_with_limits() {
        let generator = CounterExampleGenerator::with_limits(10, 20, Duration::from_secs(2));
        assert_eq!(generator.max_cex_per_quantifier, 10);
        assert_eq!(generator.max_candidates_per_var, 20);
        assert_eq!(generator.max_search_time, Duration::from_secs(2));
    }

    #[test]
    fn test_enumerate_combinations_empty() {
        let generator = CounterExampleGenerator::new();
        let combos = generator.enumerate_combinations(&[], 10, 100);
        assert_eq!(combos.len(), 1);
        assert!(combos[0].is_empty());
    }

    #[test]
    fn test_enumerate_combinations_single() {
        let generator = CounterExampleGenerator::new();
        let candidates = vec![vec![TermId::new(1), TermId::new(2)]];
        let combos = generator.enumerate_combinations(&candidates, 10, 100);
        assert_eq!(combos.len(), 2);
    }

    #[test]
    fn test_enumerate_combinations_multiple() {
        let generator = CounterExampleGenerator::new();
        let candidates = vec![
            vec![TermId::new(1), TermId::new(2)],
            vec![TermId::new(3), TermId::new(4)],
        ];
        let combos = generator.enumerate_combinations(&candidates, 10, 100);
        assert_eq!(combos.len(), 4); // 2 * 2
    }

    #[test]
    fn test_enumerate_combinations_limit() {
        let generator = CounterExampleGenerator::new();
        let candidates = vec![
            vec![TermId::new(1), TermId::new(2), TermId::new(3)],
            vec![TermId::new(4), TermId::new(5), TermId::new(6)],
        ];
        let combos = generator.enumerate_combinations(&candidates, 10, 5);
        assert!(combos.len() <= 5);
    }

    #[test]
    fn test_cex_stats_display() {
        let stats = CexStats {
            num_searches: 10,
            num_counterexamples_found: 5,
            num_combinations_tried: 100,
            num_timeouts: 1,
            total_time: Duration::from_millis(500),
        };
        let display = format!("{}", stats);
        assert!(display.contains("Searches: 10"));
        assert!(display.contains("CEX found: 5"));
    }

    #[test]
    fn test_refinement_strategy() {
        assert_ne!(
            RefinementStrategy::None,
            RefinementStrategy::BlockCounterexamples
        );
    }
}
