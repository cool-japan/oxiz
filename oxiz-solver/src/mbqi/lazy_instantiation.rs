//! Lazy Quantifier Instantiation
//!
//! This module implements lazy instantiation strategies that defer creating
//! instantiations until they are needed. This helps manage the explosion of
//! instantiations in complex quantified formulas.
//!
//! # Strategies
//!
//! - **On-Demand**: Generate instantiations only when conflicts occur
//! - **Relevance-Based**: Instantiate only relevant quantifiers
//! - **Cost-Guided**: Prioritize instantiations by estimated cost
//! - **Incremental**: Add instantiations incrementally with backtracking

use lasso::Spur;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::sort::SortId;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BinaryHeap, VecDeque};

use super::counterexample::CounterExampleGenerator;
use super::model_completion::CompletedModel;
use super::{Instantiation, QuantifiedFormula};

/// Lazy instantiation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyStrategy {
    /// Generate all instantiations eagerly
    Eager,
    /// Generate instantiations on-demand when needed
    OnDemand,
    /// Use relevance-based instantiation
    Relevance,
    /// Cost-guided instantiation
    CostGuided,
    /// Incremental instantiation with backtracking
    Incremental,
}

/// Matching context for pattern-based instantiation
#[derive(Debug)]
pub struct MatchingContext {
    /// E-graph (simplified representation)
    pub egraph: EGraph,
    /// Term database for pattern matching
    pub term_db: TermDatabase,
    /// Matching cache
    pub match_cache: FxHashMap<TermId, Vec<Match>>,
}

impl MatchingContext {
    /// Create a new matching context
    pub fn new() -> Self {
        Self {
            egraph: EGraph::new(),
            term_db: TermDatabase::new(),
            match_cache: FxHashMap::default(),
        }
    }

    /// Add a term to the matching context
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) {
        self.term_db.add_term(term, manager);
        self.egraph.add_term(term, manager);
    }

    /// Find matches for a pattern
    pub fn find_matches(&mut self, pattern: TermId, manager: &TermManager) -> Vec<Match> {
        // Check cache first
        if let Some(cached) = self.match_cache.get(&pattern) {
            return cached.clone();
        }

        // Perform matching
        let matches = self.term_db.match_pattern(pattern, manager);

        // Cache results
        self.match_cache.insert(pattern, matches.clone());

        matches
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.match_cache.clear();
    }
}

impl Default for MatchingContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified E-graph for equality reasoning
#[derive(Debug)]
pub struct EGraph {
    /// Equivalence classes (representative -> members)
    classes: FxHashMap<TermId, Vec<TermId>>,
    /// Term to representative mapping
    representatives: FxHashMap<TermId, TermId>,
}

impl EGraph {
    /// Create a new E-graph
    pub fn new() -> Self {
        Self {
            classes: FxHashMap::default(),
            representatives: FxHashMap::default(),
        }
    }

    /// Add a term to the E-graph
    pub fn add_term(&mut self, term: TermId, _manager: &TermManager) {
        if !self.representatives.contains_key(&term) {
            // Create new equivalence class
            self.classes.insert(term, vec![term]);
            self.representatives.insert(term, term);
        }
    }

    /// Merge two equivalence classes
    pub fn merge(&mut self, a: TermId, b: TermId) {
        let rep_a = self.find(a);
        let rep_b = self.find(b);

        if rep_a == rep_b {
            return;
        }

        // Merge smaller class into larger
        let size_a = self.classes.get(&rep_a).map_or(0, |c| c.len());
        let size_b = self.classes.get(&rep_b).map_or(0, |c| c.len());

        let (smaller, larger) = if size_a < size_b {
            (rep_a, rep_b)
        } else {
            (rep_b, rep_a)
        };

        // Move members from smaller to larger
        if let Some(members) = self.classes.remove(&smaller) {
            for &member in &members {
                self.representatives.insert(member, larger);
            }
            self.classes.entry(larger).or_default().extend(members);
        }
    }

    /// Find representative of equivalence class
    pub fn find(&self, term: TermId) -> TermId {
        self.representatives.get(&term).copied().unwrap_or(term)
    }

    /// Get all members of an equivalence class
    pub fn members(&self, term: TermId) -> Vec<TermId> {
        let rep = self.find(term);
        self.classes
            .get(&rep)
            .cloned()
            .unwrap_or_else(|| vec![term])
    }
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Term database for efficient pattern matching
#[derive(Debug)]
pub struct TermDatabase {
    /// Terms indexed by top symbol
    by_symbol: FxHashMap<Spur, Vec<TermId>>,
    /// All ground terms
    ground_terms: Vec<TermId>,
    /// Terms by sort
    by_sort: FxHashMap<SortId, Vec<TermId>>,
}

impl TermDatabase {
    /// Create a new term database
    pub fn new() -> Self {
        Self {
            by_symbol: FxHashMap::default(),
            ground_terms: Vec::new(),
            by_sort: FxHashMap::default(),
        }
    }

    /// Add a term to the database
    pub fn add_term(&mut self, term: TermId, manager: &TermManager) {
        let Some(t) = manager.get(term) else {
            return;
        };

        // Index by sort
        self.by_sort.entry(t.sort).or_default().push(term);

        // Index by top symbol
        if let TermKind::Apply { func, .. } = t.kind {
            self.by_symbol.entry(func).or_default().push(term);
        }

        // Track ground terms
        if self.is_ground(term, manager) {
            self.ground_terms.push(term);
        }
    }

    /// Check if a term is ground (no free variables)
    fn is_ground(&self, term: TermId, manager: &TermManager) -> bool {
        let mut visited = FxHashSet::default();
        self.is_ground_rec(term, manager, &mut visited)
    }

    fn is_ground_rec(
        &self,
        term: TermId,
        manager: &TermManager,
        visited: &mut FxHashSet<TermId>,
    ) -> bool {
        if visited.contains(&term) {
            return true;
        }
        visited.insert(term);

        let Some(t) = manager.get(term) else {
            return true;
        };

        if matches!(t.kind, TermKind::Var(_)) {
            return false;
        }

        // Check children
        match &t.kind {
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    if !self.is_ground_rec(arg, manager, visited) {
                        return false;
                    }
                }
                true
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => self.is_ground_rec(*arg, manager, visited),
            TermKind::And(args) | TermKind::Or(args) => {
                for &arg in args {
                    if !self.is_ground_rec(arg, manager, visited) {
                        return false;
                    }
                }
                true
            }
            _ => true,
        }
    }

    /// Match a pattern against the database
    pub fn match_pattern(&self, pattern: TermId, manager: &TermManager) -> Vec<Match> {
        let mut matches = Vec::new();

        // Try matching against all ground terms
        for &term in &self.ground_terms {
            if let Some(binding) = self.try_match(pattern, term, manager) {
                matches.push(Match {
                    pattern,
                    term,
                    binding,
                });
            }
        }

        matches
    }

    /// Try to match a pattern against a term
    fn try_match(
        &self,
        pattern: TermId,
        term: TermId,
        manager: &TermManager,
    ) -> Option<FxHashMap<Spur, TermId>> {
        let mut binding = FxHashMap::default();
        if self.try_match_rec(pattern, term, &mut binding, manager) {
            Some(binding)
        } else {
            None
        }
    }

    fn try_match_rec(
        &self,
        pattern: TermId,
        term: TermId,
        binding: &mut FxHashMap<Spur, TermId>,
        manager: &TermManager,
    ) -> bool {
        let Some(p) = manager.get(pattern) else {
            return false;
        };

        // Variable matches anything (but must be consistent)
        if let TermKind::Var(var_name) = p.kind {
            if let Some(&bound_term) = binding.get(&var_name) {
                return bound_term == term;
            } else {
                binding.insert(var_name, term);
                return true;
            }
        }

        let Some(t) = manager.get(term) else {
            return false;
        };

        // Structural match
        match (&p.kind, &t.kind) {
            (TermKind::Apply { func: pf, args: pa }, TermKind::Apply { func: tf, args: ta }) => {
                if pf != tf || pa.len() != ta.len() {
                    return false;
                }

                for (parg, targ) in pa.iter().zip(ta.iter()) {
                    if !self.try_match_rec(*parg, *targ, binding, manager) {
                        return false;
                    }
                }
                true
            }
            (TermKind::Not(pa), TermKind::Not(ta)) => {
                self.try_match_rec(*pa, *ta, binding, manager)
            }
            (TermKind::IntConst(pv), TermKind::IntConst(tv)) => pv == tv,
            (TermKind::True, TermKind::True) | (TermKind::False, TermKind::False) => true,
            _ => false,
        }
    }

    /// Get terms by symbol
    pub fn get_by_symbol(&self, symbol: Spur) -> &[TermId] {
        self.by_symbol.get(&symbol).map_or(&[], |v| v.as_slice())
    }

    /// Get terms by sort
    pub fn get_by_sort(&self, sort: SortId) -> &[TermId] {
        self.by_sort.get(&sort).map_or(&[], |v| v.as_slice())
    }
}

impl Default for TermDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// A pattern match
#[derive(Debug, Clone)]
pub struct Match {
    /// The pattern that was matched
    pub pattern: TermId,
    /// The term that matched the pattern
    pub term: TermId,
    /// Variable bindings
    pub binding: FxHashMap<Spur, TermId>,
}

impl Match {
    /// Create a new match
    pub fn new(pattern: TermId, term: TermId, binding: FxHashMap<Spur, TermId>) -> Self {
        Self {
            pattern,
            term,
            binding,
        }
    }
}

/// Lazy instantiator that defers instantiation
#[derive(Debug)]
pub struct LazyInstantiator {
    /// Instantiation strategy
    strategy: LazyStrategy,
    /// Queue of pending instantiations
    pending_queue: VecDeque<PendingInstantiation>,
    /// Priority queue for cost-guided strategy
    priority_queue: BinaryHeap<ScoredInstantiation>,
    /// Matching context
    matching_context: MatchingContext,
    /// Counterexample generator
    cex_generator: CounterExampleGenerator,
    /// Relevance tracker
    relevance: RelevanceTracker,
    /// Statistics
    stats: LazyStats,
}

impl LazyInstantiator {
    /// Create a new lazy instantiator
    pub fn new() -> Self {
        Self {
            strategy: LazyStrategy::OnDemand,
            pending_queue: VecDeque::new(),
            priority_queue: BinaryHeap::new(),
            matching_context: MatchingContext::new(),
            cex_generator: CounterExampleGenerator::new(),
            relevance: RelevanceTracker::new(),
            stats: LazyStats::default(),
        }
    }

    /// Create with specific strategy
    pub fn with_strategy(strategy: LazyStrategy) -> Self {
        let mut inst = Self::new();
        inst.strategy = strategy;
        inst
    }

    /// Process quantifiers and generate instantiations lazily
    pub fn process(
        &mut self,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
        max_instantiations: usize,
    ) -> Vec<Instantiation> {
        self.stats.num_process_calls += 1;

        match self.strategy {
            LazyStrategy::Eager => self.process_eager(quantifiers, model, manager),
            LazyStrategy::OnDemand => {
                self.process_on_demand(quantifiers, model, manager, max_instantiations)
            }
            LazyStrategy::Relevance => self.process_relevance(quantifiers, model, manager),
            LazyStrategy::CostGuided => self.process_cost_guided(quantifiers, model, manager),
            LazyStrategy::Incremental => {
                self.process_incremental(quantifiers, model, manager, max_instantiations)
            }
        }
    }

    /// Eager strategy: generate all instantiations immediately
    fn process_eager(
        &mut self,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        let mut instantiations = Vec::new();

        for quantifier in quantifiers {
            let cexs = self.cex_generator.generate(quantifier, model, manager);

            for cex in cexs {
                let substituted =
                    self.apply_substitution(quantifier.body, &cex.assignment, manager);
                let inst = cex.to_instantiation(substituted);
                instantiations.push(inst);
            }
        }

        self.stats.num_instantiations_generated += instantiations.len();
        instantiations
    }

    /// On-demand strategy: queue instantiations and generate as needed
    fn process_on_demand(
        &mut self,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
        max_instantiations: usize,
    ) -> Vec<Instantiation> {
        // Add quantifiers to pending queue
        for quantifier in quantifiers {
            if quantifier.can_instantiate() {
                self.pending_queue.push_back(PendingInstantiation {
                    quantifier: quantifier.clone(),
                    priority: quantifier.priority_score(),
                });
            }
        }

        // Generate up to max_instantiations
        let mut instantiations = Vec::new();

        while instantiations.len() < max_instantiations {
            let Some(pending) = self.pending_queue.pop_front() else {
                break;
            };

            let cexs = self
                .cex_generator
                .generate(&pending.quantifier, model, manager);

            for cex in cexs {
                if instantiations.len() >= max_instantiations {
                    break;
                }

                let substituted =
                    self.apply_substitution(pending.quantifier.body, &cex.assignment, manager);
                let inst = cex.to_instantiation(substituted);
                instantiations.push(inst);
            }
        }

        self.stats.num_instantiations_generated += instantiations.len();
        instantiations
    }

    /// Relevance-based strategy: only instantiate relevant quantifiers
    fn process_relevance(
        &mut self,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        let mut instantiations = Vec::new();

        // Update relevance based on model
        self.relevance.update_from_model(model, manager);

        for quantifier in quantifiers {
            // Check if quantifier is relevant
            if !self.relevance.is_relevant(quantifier.term) {
                self.stats.num_relevance_filtered += 1;
                continue;
            }

            let cexs = self.cex_generator.generate(quantifier, model, manager);

            for cex in cexs {
                let substituted =
                    self.apply_substitution(quantifier.body, &cex.assignment, manager);
                let inst = cex.to_instantiation(substituted);
                instantiations.push(inst);
            }
        }

        self.stats.num_instantiations_generated += instantiations.len();
        instantiations
    }

    /// Cost-guided strategy: prioritize by estimated cost
    fn process_cost_guided(
        &mut self,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
    ) -> Vec<Instantiation> {
        // Add quantifiers to priority queue
        for quantifier in quantifiers {
            if quantifier.can_instantiate() {
                let cost = self.estimate_cost(quantifier, manager);
                let scored = ScoredInstantiation {
                    quantifier: quantifier.clone(),
                    score: cost,
                };
                self.priority_queue.push(scored);
            }
        }

        let mut instantiations = Vec::new();

        // Process in priority order
        while let Some(scored) = self.priority_queue.pop() {
            let cexs = self
                .cex_generator
                .generate(&scored.quantifier, model, manager);

            for cex in cexs {
                let substituted =
                    self.apply_substitution(scored.quantifier.body, &cex.assignment, manager);
                let inst = cex.to_instantiation(substituted);
                instantiations.push(inst);
            }

            // Limit total instantiations
            if instantiations.len() >= 100 {
                break;
            }
        }

        self.stats.num_instantiations_generated += instantiations.len();
        instantiations
    }

    /// Incremental strategy: add instantiations incrementally
    fn process_incremental(
        &mut self,
        quantifiers: &[QuantifiedFormula],
        model: &CompletedModel,
        manager: &mut TermManager,
        max_per_round: usize,
    ) -> Vec<Instantiation> {
        let mut instantiations = Vec::new();

        for quantifier in quantifiers {
            if instantiations.len() >= max_per_round {
                break;
            }

            let cexs = self.cex_generator.generate(quantifier, model, manager);

            for cex in cexs {
                if instantiations.len() >= max_per_round {
                    break;
                }

                let substituted =
                    self.apply_substitution(quantifier.body, &cex.assignment, manager);
                let inst = cex.to_instantiation(substituted);
                instantiations.push(inst);
            }
        }

        self.stats.num_instantiations_generated += instantiations.len();
        instantiations
    }

    /// Estimate the cost of instantiating a quantifier
    fn estimate_cost(&self, quantifier: &QuantifiedFormula, manager: &TermManager) -> f64 {
        // Factors:
        // - Number of variables (more = higher cost)
        // - Body complexity (larger = higher cost)
        // - Previous instantiation count (more = higher cost to avoid loops)

        let var_cost = quantifier.num_vars() as f64;
        let body_size = self.term_size(quantifier.body, manager) as f64;
        let inst_penalty = quantifier.instantiation_count as f64;

        var_cost + body_size + inst_penalty
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
            TermKind::Not(arg) => self.term_size_rec(*arg, manager, visited),
            _ => 0,
        };

        1 + children_size
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
            _ => term,
        };

        cache.insert(term, result);
        result
    }

    /// Clear all caches and queues
    pub fn clear(&mut self) {
        self.pending_queue.clear();
        self.priority_queue.clear();
        self.matching_context.clear_cache();
        self.cex_generator.clear_cache();
        self.relevance.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> &LazyStats {
        &self.stats
    }
}

impl Default for LazyInstantiator {
    fn default() -> Self {
        Self::new()
    }
}

/// A pending instantiation request
#[derive(Debug, Clone)]
struct PendingInstantiation {
    quantifier: QuantifiedFormula,
    priority: f64,
}

/// A scored instantiation for priority queue
#[derive(Debug, Clone)]
struct ScoredInstantiation {
    quantifier: QuantifiedFormula,
    score: f64,
}

impl PartialEq for ScoredInstantiation {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredInstantiation {}

impl PartialOrd for ScoredInstantiation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredInstantiation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower score = higher priority (min-heap behavior)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Tracks relevance of quantifiers and terms
#[derive(Debug)]
pub struct RelevanceTracker {
    /// Relevant terms
    relevant_terms: FxHashSet<TermId>,
    /// Relevant quantifiers
    relevant_quantifiers: FxHashSet<TermId>,
}

impl RelevanceTracker {
    /// Create a new relevance tracker
    pub fn new() -> Self {
        Self {
            relevant_terms: FxHashSet::default(),
            relevant_quantifiers: FxHashSet::default(),
        }
    }

    /// Mark a term as relevant
    pub fn mark_relevant(&mut self, term: TermId) {
        self.relevant_terms.insert(term);
    }

    /// Mark a quantifier as relevant
    pub fn mark_quantifier_relevant(&mut self, quantifier: TermId) {
        self.relevant_quantifiers.insert(quantifier);
    }

    /// Check if a term is relevant
    pub fn is_relevant(&self, term: TermId) -> bool {
        self.relevant_terms.contains(&term) || self.relevant_quantifiers.contains(&term)
    }

    /// Update relevance from model
    pub fn update_from_model(&mut self, model: &CompletedModel, _manager: &TermManager) {
        // Mark all terms in model as relevant
        for &term in model.assignments.keys() {
            self.mark_relevant(term);
        }
    }

    /// Clear all relevance info
    pub fn clear(&mut self) {
        self.relevant_terms.clear();
        self.relevant_quantifiers.clear();
    }
}

impl Default for RelevanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for lazy instantiation
#[derive(Debug, Clone, Default)]
pub struct LazyStats {
    /// Number of process calls
    pub num_process_calls: usize,
    /// Number of instantiations generated
    pub num_instantiations_generated: usize,
    /// Number of instantiations filtered by relevance
    pub num_relevance_filtered: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn test_lazy_strategy_equality() {
        assert_eq!(LazyStrategy::Eager, LazyStrategy::Eager);
        assert_ne!(LazyStrategy::Eager, LazyStrategy::OnDemand);
    }

    #[test]
    fn test_matching_context_creation() {
        let ctx = MatchingContext::new();
        assert_eq!(ctx.match_cache.len(), 0);
    }

    #[test]
    fn test_egraph_creation() {
        let egraph = EGraph::new();
        assert_eq!(egraph.classes.len(), 0);
    }

    #[test]
    fn test_egraph_find() {
        let mut egraph = EGraph::new();
        let term = TermId::new(1);
        let manager = TermManager::new();
        egraph.add_term(term, &manager);
        assert_eq!(egraph.find(term), term);
    }

    #[test]
    fn test_egraph_merge() {
        let mut egraph = EGraph::new();
        let manager = TermManager::new();
        let term1 = TermId::new(1);
        let term2 = TermId::new(2);
        egraph.add_term(term1, &manager);
        egraph.add_term(term2, &manager);
        egraph.merge(term1, term2);
        assert_eq!(egraph.find(term1), egraph.find(term2));
    }

    #[test]
    fn test_term_database_creation() {
        let db = TermDatabase::new();
        assert_eq!(db.ground_terms.len(), 0);
    }

    #[test]
    fn test_match_creation() {
        let m = Match::new(TermId::new(1), TermId::new(2), FxHashMap::default());
        assert_eq!(m.pattern, TermId::new(1));
        assert_eq!(m.term, TermId::new(2));
    }

    #[test]
    fn test_lazy_instantiator_creation() {
        let inst = LazyInstantiator::new();
        assert_eq!(inst.strategy, LazyStrategy::OnDemand);
    }

    #[test]
    fn test_lazy_instantiator_with_strategy() {
        let inst = LazyInstantiator::with_strategy(LazyStrategy::CostGuided);
        assert_eq!(inst.strategy, LazyStrategy::CostGuided);
    }

    #[test]
    fn test_relevance_tracker_creation() {
        let tracker = RelevanceTracker::new();
        assert!(!tracker.is_relevant(TermId::new(1)));
    }

    #[test]
    fn test_relevance_tracker_mark() {
        let mut tracker = RelevanceTracker::new();
        let term = TermId::new(1);
        tracker.mark_relevant(term);
        assert!(tracker.is_relevant(term));
    }

    #[test]
    fn test_scored_instantiation_ordering() {
        let q1 = QuantifiedFormula::new(TermId::new(1), SmallVec::new(), TermId::new(2), true);
        let q2 = QuantifiedFormula::new(TermId::new(3), SmallVec::new(), TermId::new(4), true);

        let s1 = ScoredInstantiation {
            quantifier: q1,
            score: 1.0,
        };
        let s2 = ScoredInstantiation {
            quantifier: q2,
            score: 2.0,
        };

        // Lower score should be higher priority
        assert!(s1 > s2);
    }
}
