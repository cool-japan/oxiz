//! Model-Based Instantiation Engine
//!
//! This module implements the core instantiation logic for MBQI. It handles:
//! - Extracting instantiations from models and counterexamples
//! - Conflict-driven instantiation
//! - Pattern matching and trigger selection
//! - Instantiation deduplication and filtering

#![allow(missing_docs)]
#![allow(dead_code)]

use lasso::Spur;
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::counterexample::CounterExampleGenerator;
use super::model_completion::CompletedModel;
use super::{Instantiation, InstantiationReason, QuantifiedFormula};

/// Context for instantiation
#[derive(Debug)]
pub struct InstantiationContext {
    /// Term manager
    pub manager: TermManager,
    /// Current model
    pub model: CompletedModel,
    /// Generation counter
    pub generation: u32,
    /// E-graph for equality reasoning (simplified)
    pub equalities: FxHashMap<TermId, TermId>,
}

impl InstantiationContext {
    /// Create a new instantiation context
    pub fn new(manager: TermManager) -> Self {
        Self {
            manager,
            model: CompletedModel::new(),
            generation: 0,
            equalities: FxHashMap::default(),
        }
    }

    /// Set the current model
    pub fn set_model(&mut self, model: CompletedModel) {
        self.model = model;
    }

    /// Increment generation
    pub fn next_generation(&mut self) {
        self.generation += 1;
    }

    /// Add an equality
    pub fn add_equality(&mut self, lhs: TermId, rhs: TermId) {
        self.equalities.insert(lhs, rhs);
        self.equalities.insert(rhs, lhs);
    }

    /// Find representative in equality graph
    pub fn find_representative(&self, term: TermId) -> TermId {
        let mut current = term;
        let mut visited = FxHashSet::default();

        while let Some(&next) = self.equalities.get(&current) {
            if visited.contains(&next) {
                break; // Cycle detected
            }
            visited.insert(current);
            current = next;
        }

        current
    }
}

/// Pattern for instantiation (E-matching style)
#[derive(Debug, Clone)]
pub struct InstantiationPattern {
    /// Terms that form the pattern
    pub terms: Vec<TermId>,
    /// Variables that must be matched
    pub vars: FxHashSet<Spur>,
    /// Number of variables
    pub num_vars: usize,
    /// Pattern quality (higher = better)
    pub quality: f64,
}

impl InstantiationPattern {
    /// Create a new pattern
    pub fn new(terms: Vec<TermId>) -> Self {
        Self {
            terms,
            vars: FxHashSet::default(),
            num_vars: 0,
            quality: 1.0,
        }
    }

    /// Extract patterns from a quantified formula
    pub fn extract_patterns(quantifier: &QuantifiedFormula, manager: &TermManager) -> Vec<Self> {
        let mut patterns = Vec::new();

        // Use explicit patterns if available
        if !quantifier.patterns.is_empty() {
            for pattern_terms in &quantifier.patterns {
                let mut pattern = Self::new(pattern_terms.clone());
                pattern.collect_vars(manager);
                pattern.calculate_quality(manager);
                patterns.push(pattern);
            }
        } else {
            // Auto-generate patterns from the body
            let generated = Self::generate_patterns(quantifier.body, manager);
            patterns.extend(generated);
        }

        patterns
    }

    /// Generate patterns from a term
    fn generate_patterns(term: TermId, manager: &TermManager) -> Vec<Self> {
        let mut patterns = Vec::new();
        let candidates = Self::collect_pattern_candidates(term, manager);

        for candidate in candidates {
            let mut pattern = Self::new(vec![candidate]);
            pattern.collect_vars(manager);
            if pattern.num_vars > 0 {
                pattern.calculate_quality(manager);
                patterns.push(pattern);
            }
        }

        patterns
    }

    /// Collect pattern candidates from a term
    fn collect_pattern_candidates(term: TermId, manager: &TermManager) -> Vec<TermId> {
        let mut candidates = Vec::new();
        let mut visited = FxHashSet::default();
        Self::collect_candidates_rec(term, &mut candidates, &mut visited, manager);
        candidates
    }

    fn collect_candidates_rec(
        term: TermId,
        candidates: &mut Vec<TermId>,
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

        // Function applications are good pattern candidates
        if matches!(t.kind, TermKind::Apply { .. }) {
            candidates.push(term);
        }

        // Recurse into children
        match &t.kind {
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    Self::collect_candidates_rec(arg, candidates, visited, manager);
                }
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    Self::collect_candidates_rec(arg, candidates, visited, manager);
                }
            }
            TermKind::Eq(lhs, rhs) | TermKind::Lt(lhs, rhs) | TermKind::Le(lhs, rhs) => {
                Self::collect_candidates_rec(*lhs, candidates, visited, manager);
                Self::collect_candidates_rec(*rhs, candidates, visited, manager);
            }
            _ => {}
        }
    }

    /// Collect variables in the pattern
    fn collect_vars(&mut self, manager: &TermManager) {
        self.vars.clear();
        let mut visited = FxHashSet::default();

        // Collect terms first to avoid borrow checker issues
        let terms: Vec<_> = self.terms.to_vec();
        for term in terms {
            self.collect_vars_rec(term, &mut visited, manager);
        }

        self.num_vars = self.vars.len();
    }

    fn collect_vars_rec(
        &mut self,
        term: TermId,
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

        if let TermKind::Var(name) = t.kind {
            self.vars.insert(name);
        }

        // Recurse into children
        match &t.kind {
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    self.collect_vars_rec(arg, visited, manager);
                }
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                self.collect_vars_rec(*arg, visited, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    self.collect_vars_rec(arg, visited, manager);
                }
            }
            _ => {}
        }
    }

    /// Calculate pattern quality
    fn calculate_quality(&mut self, manager: &TermManager) {
        // Quality factors:
        // - More variables = better (more specific)
        // - Fewer terms = better (simpler)
        // - Contains function applications = better

        let var_factor = 1.0 + (self.num_vars as f64);
        let term_factor = 1.0 / (1.0 + self.terms.len() as f64);
        let func_factor = if self.has_function_applications(manager) {
            2.0
        } else {
            1.0
        };

        self.quality = var_factor * term_factor * func_factor;
    }

    fn has_function_applications(&self, manager: &TermManager) -> bool {
        for &term in &self.terms {
            if let Some(t) = manager.get(term)
                && matches!(t.kind, TermKind::Apply { .. })
            {
                return true;
            }
        }
        false
    }
}

/// Quantifier instantiator
#[derive(Debug)]
pub struct QuantifierInstantiator {
    /// Counterexample generator
    cex_generator: CounterExampleGenerator,
    /// Deduplication cache
    dedup_cache: FxHashSet<InstantiationKey>,
    /// Statistics
    stats: InstantiatorStats,
}

impl QuantifierInstantiator {
    /// Create a new instantiator
    pub fn new() -> Self {
        Self {
            cex_generator: CounterExampleGenerator::new(),
            dedup_cache: FxHashSet::default(),
            stats: InstantiatorStats::default(),
        }
    }

    /// Generate instantiations for a quantifier using model-based approach
    pub fn instantiate_from_model(
        &mut self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        self.stats.num_instantiation_attempts += 1;

        let mut instantiations = Vec::new();

        // Generate counterexamples
        let counterexamples = self.cex_generator.generate(quantifier, model, manager);

        // Convert counterexamples to instantiations
        for cex in counterexamples {
            // Apply substitution to get ground instance
            let ground_body = self.apply_substitution(quantifier.body, &cex.assignment, manager);

            let inst = cex.to_instantiation(ground_body);

            // Check for duplicates
            if self.is_duplicate(&inst) {
                self.stats.num_duplicates_filtered += 1;
                continue;
            }

            self.record_instantiation(&inst);
            instantiations.push(inst);
        }

        self.stats.num_instantiations_generated += instantiations.len();
        instantiations
    }

    /// Generate instantiations using conflict-driven approach
    pub fn instantiate_from_conflict(
        &mut self,
        quantifier: &QuantifiedFormula,
        conflict: &[TermId],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        let mut instantiations = Vec::new();

        // Analyze the conflict to extract relevant terms
        let conflict_terms = self.extract_relevant_terms(conflict, manager);

        // Try to build instantiations from conflict terms
        for assignment in
            self.build_assignments_from_terms(&quantifier.bound_vars, &conflict_terms, manager)
        {
            let ground_body = self.apply_substitution(quantifier.body, &assignment, manager);

            let inst = Instantiation::with_reason(
                quantifier.term,
                assignment,
                ground_body,
                model.generation,
                InstantiationReason::Conflict,
            );

            if !self.is_duplicate(&inst) {
                self.record_instantiation(&inst);
                instantiations.push(inst);
            }
        }

        instantiations
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
            TermKind::Apply { func, args } => {
                let func_name = manager.resolve_str(*func).to_string();
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.apply_substitution_cached(a, subst, manager, cache))
                    .collect();
                manager.mk_apply(&func_name, new_args, t.sort)
            }
            _ => term,
        };

        cache.insert(term, result);
        result
    }

    /// Extract relevant terms from conflict clause
    fn extract_relevant_terms(&self, conflict: &[TermId], manager: &TermManager) -> Vec<TermId> {
        let mut terms = Vec::new();
        let mut visited = FxHashSet::default();

        for &term in conflict {
            self.extract_relevant_terms_rec(term, &mut terms, &mut visited, manager);
        }

        terms
    }

    fn extract_relevant_terms_rec(
        &self,
        term: TermId,
        terms: &mut Vec<TermId>,
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

        // Collect ground terms
        if self.is_ground_value(term, manager) {
            terms.push(term);
        }

        // Recurse into children
        match &t.kind {
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                self.extract_relevant_terms_rec(*arg, terms, visited, manager);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    self.extract_relevant_terms_rec(arg, terms, visited, manager);
                }
            }
            TermKind::Eq(lhs, rhs) | TermKind::Lt(lhs, rhs) => {
                self.extract_relevant_terms_rec(*lhs, terms, visited, manager);
                self.extract_relevant_terms_rec(*rhs, terms, visited, manager);
            }
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    self.extract_relevant_terms_rec(arg, terms, visited, manager);
                }
            }
            _ => {}
        }
    }

    /// Check if a term is a ground value
    fn is_ground_value(&self, term: TermId, manager: &TermManager) -> bool {
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

    /// Build assignments from terms
    fn build_assignments_from_terms(
        &self,
        bound_vars: &[(Spur, SortId)],
        terms: &[TermId],
        manager: &TermManager,
    ) -> Vec<FxHashMap<Spur, TermId>> {
        let mut assignments = Vec::new();

        // Group terms by sort
        let mut terms_by_sort: FxHashMap<SortId, Vec<TermId>> = FxHashMap::default();
        for &term in terms {
            if let Some(t) = manager.get(term) {
                terms_by_sort.entry(t.sort).or_default().push(term);
            }
        }

        // Build candidate lists for each variable
        let mut candidates = Vec::new();
        for &(_name, sort) in bound_vars {
            let sort_terms = terms_by_sort
                .get(&sort)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            candidates.push(sort_terms.to_vec());
        }

        // Enumerate combinations (limited)
        let max_combinations = 10;
        let mut indices = vec![0usize; bound_vars.len()];

        for _ in 0..max_combinations {
            let mut assignment = FxHashMap::default();
            let mut valid = true;

            for (i, &idx) in indices.iter().enumerate() {
                if let Some(cands) = candidates.get(i) {
                    if let Some(&term) = cands.get(idx) {
                        if let Some((name, _)) = bound_vars.get(i) {
                            assignment.insert(*name, term);
                        }
                    } else {
                        valid = false;
                        break;
                    }
                }
            }

            if valid && assignment.len() == bound_vars.len() {
                assignments.push(assignment);
            }

            // Increment indices
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

        assignments
    }

    /// Check if an instantiation is a duplicate
    fn is_duplicate(&self, inst: &Instantiation) -> bool {
        let key = InstantiationKey::from_instantiation(inst);
        self.dedup_cache.contains(&key)
    }

    /// Record an instantiation for deduplication
    fn record_instantiation(&mut self, inst: &Instantiation) {
        let key = InstantiationKey::from_instantiation(inst);
        self.dedup_cache.insert(key);
    }

    /// Clear deduplication cache
    pub fn clear_cache(&mut self) {
        self.dedup_cache.clear();
        self.cex_generator.clear_cache();
    }

    /// Get statistics
    pub fn stats(&self) -> &InstantiatorStats {
        &self.stats
    }
}

impl Default for QuantifierInstantiator {
    fn default() -> Self {
        Self::new()
    }
}

/// Key for instantiation deduplication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct InstantiationKey {
    quantifier: TermId,
    binding: Vec<(Spur, TermId)>,
}

impl InstantiationKey {
    fn from_instantiation(inst: &Instantiation) -> Self {
        let mut binding: Vec<_> = inst.substitution.iter().map(|(&k, &v)| (k, v)).collect();
        binding.sort_by_key(|(k, _)| *k);
        Self {
            quantifier: inst.quantifier,
            binding,
        }
    }
}

/// Instantiation engine that coordinates all instantiation strategies
#[derive(Debug)]
pub struct InstantiationEngine {
    /// Main quantifier instantiator
    instantiator: QuantifierInstantiator,
    /// Pattern matcher
    pattern_matcher: PatternMatcher,
    /// Enumerative instantiation
    enumerative: EnumerativeInstantiator,
    /// Statistics
    stats: EngineStats,
}

impl InstantiationEngine {
    /// Create a new instantiation engine
    pub fn new() -> Self {
        Self {
            instantiator: QuantifierInstantiator::new(),
            pattern_matcher: PatternMatcher::new(),
            enumerative: EnumerativeInstantiator::new(),
            stats: EngineStats::default(),
        }
    }

    /// Generate instantiations for a quantifier
    pub fn instantiate(
        &mut self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        let mut instantiations = Vec::new();

        // Strategy 1: Model-based instantiation
        let model_based = self
            .instantiator
            .instantiate_from_model(quantifier, model, manager);
        instantiations.extend(model_based);

        // Strategy 2: Pattern-based instantiation (if patterns exist)
        if !quantifier.patterns.is_empty() {
            let pattern_based = self
                .pattern_matcher
                .match_patterns(quantifier, model, manager);
            instantiations.extend(pattern_based);
        }

        // Strategy 3: Enumerative instantiation (as fallback, limited)
        if instantiations.is_empty() {
            let enumerative = self.enumerative.enumerate(quantifier, model, manager, 3);
            instantiations.extend(enumerative);
        }

        self.stats.num_instantiations += instantiations.len();
        instantiations
    }

    /// Clear all caches
    pub fn clear_caches(&mut self) {
        self.instantiator.clear_cache();
        self.pattern_matcher.clear_cache();
    }

    /// Get statistics
    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }
}

impl Default for InstantiationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern matcher for E-matching style instantiation
#[derive(Debug)]
struct PatternMatcher {
    /// Match cache
    cache: FxHashMap<TermId, Vec<FxHashMap<Spur, TermId>>>,
}

impl PatternMatcher {
    fn new() -> Self {
        Self {
            cache: FxHashMap::default(),
        }
    }

    fn match_patterns(
        &mut self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        let mut instantiations = Vec::new();

        // Extract patterns
        let patterns = InstantiationPattern::extract_patterns(quantifier, manager);

        // Match each pattern
        for pattern in patterns {
            let matches = self.match_pattern(&pattern, model, manager);
            for assignment in matches {
                let ground_body = self.apply_substitution(quantifier.body, &assignment, manager);
                let inst = Instantiation::with_reason(
                    quantifier.term,
                    assignment,
                    ground_body,
                    model.generation,
                    InstantiationReason::EMatching,
                );
                instantiations.push(inst);
            }
        }

        instantiations
    }

    fn match_pattern(
        &self,
        _pattern: &InstantiationPattern,
        _model: &CompletedModel,
        _manager: &TermManager,
    ) -> Vec<FxHashMap<Spur, TermId>> {
        // Simplified pattern matching
        // A full implementation would use E-matching algorithms
        Vec::new()
    }

    fn apply_substitution(
        &self,
        term: TermId,
        subst: &FxHashMap<Spur, TermId>,
        manager: &mut TermManager,
    ) -> TermId {
        // Reuse implementation from QuantifierInstantiator
        let instantiator = QuantifierInstantiator::new();
        instantiator.apply_substitution(term, subst, manager)
    }

    fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Enumerative instantiator (brute-force small domain)
#[derive(Debug)]
struct EnumerativeInstantiator;

impl EnumerativeInstantiator {
    fn new() -> Self {
        Self
    }

    fn enumerate(
        &self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &mut TermManager,
        max_per_var: usize,
    ) -> Vec<Instantiation> {
        let mut instantiations = Vec::new();

        // Build small domains for each variable
        let domains = self.build_small_domains(&quantifier.bound_vars, model, manager, max_per_var);

        // Enumerate all combinations
        let combinations = self.enumerate_combinations(&domains);

        for combo in combinations {
            let mut assignment = FxHashMap::default();
            for (i, &value) in combo.iter().enumerate() {
                if let Some((name, _)) = quantifier.bound_vars.get(i) {
                    assignment.insert(*name, value);
                }
            }

            let instantiator = QuantifierInstantiator::new();
            let ground_body =
                instantiator.apply_substitution(quantifier.body, &assignment, manager);

            let inst = Instantiation::with_reason(
                quantifier.term,
                assignment,
                ground_body,
                model.generation,
                InstantiationReason::Enumerative,
            );
            instantiations.push(inst);
        }

        instantiations
    }

    fn build_small_domains(
        &self,
        bound_vars: &[(Spur, SortId)],
        model: &CompletedModel,
        manager: &mut TermManager,
        max_per_var: usize,
    ) -> Vec<Vec<TermId>> {
        let mut domains = Vec::new();

        for &(_name, sort) in bound_vars {
            let mut domain = Vec::new();

            // Use universe if available
            if let Some(universe) = model.universe(sort) {
                domain.extend_from_slice(universe);
            }

            // Add default values
            if sort == manager.sorts.int_sort {
                for i in 0..max_per_var.min(3) {
                    domain.push(manager.mk_int(BigInt::from(i as i32)));
                }
            } else if sort == manager.sorts.bool_sort {
                domain.push(manager.mk_true());
                domain.push(manager.mk_false());
            }

            domain.truncate(max_per_var);
            domains.push(domain);
        }

        domains
    }

    fn enumerate_combinations(&self, domains: &[Vec<TermId>]) -> Vec<Vec<TermId>> {
        if domains.is_empty() {
            return vec![vec![]];
        }

        let mut results = Vec::new();
        let mut indices = vec![0usize; domains.len()];
        let max_results = 100; // Limit total combinations

        loop {
            let combo: Vec<TermId> = indices
                .iter()
                .enumerate()
                .filter_map(|(i, &idx)| domains.get(i).and_then(|d| d.get(idx).copied()))
                .collect();

            if combo.len() == domains.len() {
                results.push(combo);
            }

            if results.len() >= max_results {
                break;
            }

            // Increment
            let mut carry = true;
            for (i, idx) in indices.iter_mut().enumerate() {
                if carry {
                    *idx += 1;
                    let limit = domains.get(i).map_or(1, |d| d.len());
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

/// Statistics for instantiator
#[derive(Debug, Clone, Default)]
pub struct InstantiatorStats {
    pub num_instantiation_attempts: usize,
    pub num_instantiations_generated: usize,
    pub num_duplicates_filtered: usize,
}

/// Statistics for engine
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub num_instantiations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use lasso::Key;

    #[test]
    fn test_instantiation_context_creation() {
        let manager = TermManager::new();
        let ctx = InstantiationContext::new(manager);
        assert_eq!(ctx.generation, 0);
    }

    #[test]
    fn test_instantiation_context_generation() {
        let manager = TermManager::new();
        let mut ctx = InstantiationContext::new(manager);
        ctx.next_generation();
        assert_eq!(ctx.generation, 1);
    }

    #[test]
    fn test_instantiation_pattern_creation() {
        let pattern = InstantiationPattern::new(vec![TermId::new(1)]);
        assert_eq!(pattern.terms.len(), 1);
        assert_eq!(pattern.num_vars, 0);
    }

    #[test]
    fn test_quantifier_instantiator_creation() {
        let inst = QuantifierInstantiator::new();
        assert_eq!(inst.stats.num_instantiation_attempts, 0);
    }

    #[test]
    fn test_instantiation_key_equality() {
        let key1 = InstantiationKey {
            quantifier: TermId::new(1),
            binding: vec![(
                Spur::try_from_usize(1).expect("valid spur"),
                TermId::new(10),
            )],
        };
        let key2 = InstantiationKey {
            quantifier: TermId::new(1),
            binding: vec![(
                Spur::try_from_usize(1).expect("valid spur"),
                TermId::new(10),
            )],
        };
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_instantiation_engine_creation() {
        let engine = InstantiationEngine::new();
        assert_eq!(engine.stats.num_instantiations, 0);
    }

    #[test]
    fn test_pattern_matcher_creation() {
        let matcher = PatternMatcher::new();
        assert_eq!(matcher.cache.len(), 0);
    }

    #[test]
    fn test_enumerative_instantiator_creation() {
        let _enum_inst = EnumerativeInstantiator::new();
    }
}
