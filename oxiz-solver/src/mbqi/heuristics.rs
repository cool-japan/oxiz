//! MBQI Heuristics and Selection Strategies
//!
//! This module implements various heuristics for guiding MBQI, including:
//! - Quantifier selection strategies
//! - Trigger/pattern selection
//! - Instantiation ordering
//! - Resource allocation

#[allow(unused_imports)]
use crate::prelude::*;
use core::cmp::Ordering;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use oxiz_core::interner::Spur;

use super::model_completion::CompletedModel;
use super::{ConflictScores, QuantifiedFormula, QuantifierId};

/// Overall MBQI heuristics configuration
#[derive(Debug, Clone)]
pub struct MBQIHeuristics {
    /// Quantifier selection strategy
    pub quantifier_selection: SelectionStrategy,
    /// Trigger selection strategy
    pub trigger_selection: TriggerSelection,
    /// Instantiation ordering
    pub instantiation_ordering: InstantiationOrdering,
    /// Resource allocation strategy
    pub resource_allocation: ResourceAllocation,
    /// Enable conflict analysis
    pub enable_conflict_analysis: bool,
    /// Enable model-based bounds
    pub enable_model_bounds: bool,
}

impl MBQIHeuristics {
    /// Create default heuristics
    pub fn new() -> Self {
        Self {
            quantifier_selection: SelectionStrategy::PriorityBased,
            trigger_selection: TriggerSelection::MatchingLoopAvoidance,
            instantiation_ordering: InstantiationOrdering::CostBased,
            resource_allocation: ResourceAllocation::Balanced,
            enable_conflict_analysis: true,
            enable_model_bounds: true,
        }
    }

    /// Create conservative heuristics (fewer instantiations)
    pub fn conservative() -> Self {
        Self {
            quantifier_selection: SelectionStrategy::MostConstrained,
            trigger_selection: TriggerSelection::MinPatterns,
            instantiation_ordering: InstantiationOrdering::DepthFirst,
            resource_allocation: ResourceAllocation::Conservative,
            enable_conflict_analysis: true,
            enable_model_bounds: true,
        }
    }

    /// Create aggressive heuristics (more instantiations)
    pub fn aggressive() -> Self {
        Self {
            quantifier_selection: SelectionStrategy::BreadthFirst,
            trigger_selection: TriggerSelection::MaxCoverage,
            instantiation_ordering: InstantiationOrdering::BreadthFirst,
            resource_allocation: ResourceAllocation::Aggressive,
            enable_conflict_analysis: false,
            enable_model_bounds: false,
        }
    }
}

impl Default for MBQIHeuristics {
    fn default() -> Self {
        Self::new()
    }
}

/// Strategy for selecting which quantifiers to instantiate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Select in order of definition
    Sequential,
    /// Select based on priority scores
    PriorityBased,
    /// Breadth-first (rotate through quantifiers)
    BreadthFirst,
    /// Depth-first (exhaust one before moving to next)
    DepthFirst,
    /// Most constrained first
    MostConstrained,
    /// Least constrained first
    LeastConstrained,
    /// Random selection
    Random,
}

/// Strategy for trigger/pattern selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerSelection {
    /// Use all available patterns
    All,
    /// Use patterns with minimum variables
    MinVars,
    /// Use patterns with minimum terms
    MinPatterns,
    /// Maximize coverage of quantifier body
    MaxCoverage,
    /// Avoid patterns that cause matching loops
    MatchingLoopAvoidance,
    /// User-specified patterns only
    UserOnly,
}

/// Ordering for generated instantiations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstantiationOrdering {
    /// Order by estimated cost
    CostBased,
    /// Depth-first
    DepthFirst,
    /// Breadth-first
    BreadthFirst,
    /// Prefer simpler instantiations
    SimplestFirst,
    /// Prefer instantiations with more ground terms
    GroundnessFirst,
}

/// Resource allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceAllocation {
    /// Conservative (few instantiations)
    Conservative,
    /// Balanced
    Balanced,
    /// Aggressive (many instantiations)
    Aggressive,
    /// Adaptive (adjust based on progress)
    Adaptive,
}

/// Budget for MBQI instantiations.
#[derive(Debug, Clone)]
pub struct MBQIBudget {
    /// Total budget available for a round.
    pub global_budget: u32,
    /// Per-quantifier slices of the total budget.
    pub per_quantifier: FxHashMap<QuantifierId, u32>,
    /// Remaining global budget after consumption.
    pub remaining_global: u32,
}

impl MBQIBudget {
    /// Create a fresh budget.
    pub fn new(global_budget: u32) -> Self {
        Self {
            global_budget,
            per_quantifier: FxHashMap::default(),
            remaining_global: global_budget,
        }
    }

    /// Distribute the remaining budget across quantifiers, weighted by conflict scores.
    pub fn carve_per_quantifier(
        &mut self,
        quantifiers: &[QuantifierId],
        conflict_scores: Option<&ConflictScores>,
    ) {
        self.per_quantifier.clear();
        self.remaining_global = self.global_budget;

        if quantifiers.is_empty() || self.global_budget == 0 {
            return;
        }

        let total_weight: u64 = quantifiers
            .iter()
            .map(|qid| {
                conflict_scores
                    .and_then(|scores| scores.score(*qid))
                    .map_or(1_u64, |score| score as u64 + 1)
            })
            .sum();

        let mut assigned = 0_u32;
        for (index, &qid) in quantifiers.iter().enumerate() {
            let weight = conflict_scores
                .and_then(|scores| scores.score(qid))
                .map_or(1_u64, |score| score as u64 + 1);
            let mut share = ((self.global_budget as u64 * weight) / total_weight) as u32;
            if share == 0 {
                share = 1;
            }
            if index + 1 == quantifiers.len() {
                share = self.global_budget.saturating_sub(assigned);
            }
            assigned = assigned.saturating_add(share);
            self.per_quantifier.insert(qid, share);
        }
    }

    /// Consume part of the budget for one quantifier.
    pub fn consume(&mut self, qid: QuantifierId, amount: u32) -> bool {
        if amount == 0 {
            return true;
        }
        let Some(remaining_for_q) = self.per_quantifier.get_mut(&qid) else {
            return false;
        };
        if *remaining_for_q < amount || self.remaining_global < amount {
            return false;
        }
        *remaining_for_q -= amount;
        self.remaining_global -= amount;
        true
    }
}

/// Instantiation heuristic scorer
#[derive(Debug)]
pub struct InstantiationHeuristic {
    /// Heuristics configuration
    config: MBQIHeuristics,
    /// Quantifier scores
    quantifier_scores: FxHashMap<TermId, f64>,
    /// Pattern quality scores
    pattern_scores: FxHashMap<TermId, f64>,
    /// Historical success rates
    success_history: FxHashMap<TermId, SuccessRate>,
}

impl InstantiationHeuristic {
    /// Create a new heuristic
    pub fn new(config: MBQIHeuristics) -> Self {
        Self {
            config,
            quantifier_scores: FxHashMap::default(),
            pattern_scores: FxHashMap::default(),
            success_history: FxHashMap::default(),
        }
    }

    /// Calculate priority score for a quantifier
    pub fn calculate_priority(
        &mut self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &TermManager,
    ) -> f64 {
        // Check cache
        if let Some(&cached) = self.quantifier_scores.get(&quantifier.term) {
            return cached;
        }

        let score = match self.config.quantifier_selection {
            SelectionStrategy::Sequential => 1.0,
            SelectionStrategy::PriorityBased => self.priority_based_score(quantifier, manager),
            SelectionStrategy::BreadthFirst => 1.0 / (1.0 + quantifier.instantiation_count as f64),
            SelectionStrategy::DepthFirst => quantifier.instantiation_count as f64,
            SelectionStrategy::MostConstrained => self.constraint_score(quantifier, model, manager),
            SelectionStrategy::LeastConstrained => {
                -self.constraint_score(quantifier, model, manager)
            }
            SelectionStrategy::Random => {
                // Use deterministic pseudo-random based on term ID
                let hash = quantifier.term.raw() as u64;
                ((hash.wrapping_mul(2654435761) >> 32) as f64) / (u32::MAX as f64)
            }
        };

        self.quantifier_scores.insert(quantifier.term, score);
        score
    }

    /// Calculate priority-based score
    fn priority_based_score(&self, quantifier: &QuantifiedFormula, manager: &TermManager) -> f64 {
        // Combine multiple factors
        let weight_factor = quantifier.weight;
        let inst_factor = 1.0 / (1.0 + quantifier.instantiation_count as f64);
        let depth_factor = 1.0 / (1.0 + quantifier.nesting_depth as f64);
        let complexity_factor = 1.0 / (1.0 + self.body_complexity(quantifier.body, manager) as f64);

        weight_factor * inst_factor * depth_factor * complexity_factor
    }

    /// Calculate constraint score (higher = more constrained)
    fn constraint_score(
        &self,
        quantifier: &QuantifiedFormula,
        model: &CompletedModel,
        manager: &TermManager,
    ) -> f64 {
        let mut score = 0.0;

        // Count available candidates for each variable
        for &(_name, sort) in &quantifier.bound_vars {
            let num_candidates = model.universe(sort).map_or(0, |u| u.len());
            if num_candidates > 0 {
                score += 1.0 / num_candidates as f64;
            } else {
                score += 1.0;
            }
        }

        // Add complexity penalty
        let complexity = self.body_complexity(quantifier.body, manager);
        score += complexity as f64 * 0.1;

        score
    }

    /// Calculate body complexity
    fn body_complexity(&self, term: TermId, manager: &TermManager) -> usize {
        let mut visited = FxHashSet::default();
        self.body_complexity_rec(term, manager, &mut visited)
    }

    fn body_complexity_rec(
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

        let children_complexity = match &t.kind {
            TermKind::And(args) | TermKind::Or(args) => args
                .iter()
                .map(|&arg| self.body_complexity_rec(arg, manager, visited))
                .sum(),
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                self.body_complexity_rec(*arg, manager, visited)
            }
            TermKind::Eq(lhs, rhs)
            | TermKind::Lt(lhs, rhs)
            | TermKind::Le(lhs, rhs)
            | TermKind::Gt(lhs, rhs)
            | TermKind::Ge(lhs, rhs) => {
                self.body_complexity_rec(*lhs, manager, visited)
                    + self.body_complexity_rec(*rhs, manager, visited)
            }
            TermKind::Apply { args, .. } => args
                .iter()
                .map(|&arg| self.body_complexity_rec(arg, manager, visited))
                .sum(),
            _ => 0,
        };

        1 + children_complexity
    }

    /// Select patterns for a quantifier
    pub fn select_patterns(
        &self,
        quantifier: &QuantifiedFormula,
        manager: &TermManager,
    ) -> Vec<Vec<TermId>> {
        match self.config.trigger_selection {
            TriggerSelection::All => quantifier.patterns.clone(),
            TriggerSelection::MinVars => self.select_min_vars_patterns(quantifier, manager),
            TriggerSelection::MinPatterns => self.select_min_patterns(quantifier),
            TriggerSelection::MaxCoverage => self.select_max_coverage_patterns(quantifier, manager),
            TriggerSelection::MatchingLoopAvoidance => {
                self.select_loop_avoiding_patterns(quantifier, manager)
            }
            TriggerSelection::UserOnly => quantifier.patterns.clone(),
        }
    }

    fn select_min_vars_patterns(
        &self,
        quantifier: &QuantifiedFormula,
        manager: &TermManager,
    ) -> Vec<Vec<TermId>> {
        if quantifier.patterns.is_empty() {
            return vec![];
        }

        let mut patterns_with_vars: Vec<_> = quantifier
            .patterns
            .iter()
            .map(|pattern| {
                let num_vars = self.count_vars_in_pattern(pattern, manager);
                (pattern.clone(), num_vars)
            })
            .collect();

        patterns_with_vars.sort_by_key(|(_, num_vars)| *num_vars);

        vec![patterns_with_vars[0].0.clone()]
    }

    fn select_min_patterns(&self, quantifier: &QuantifiedFormula) -> Vec<Vec<TermId>> {
        if quantifier.patterns.is_empty() {
            return vec![];
        }

        // Select the pattern with fewest terms
        let min_pattern = quantifier
            .patterns
            .iter()
            .min_by_key(|pattern| pattern.len())
            .cloned();

        min_pattern.map_or_else(Vec::new, |p| vec![p])
    }

    fn select_max_coverage_patterns(
        &self,
        quantifier: &QuantifiedFormula,
        manager: &TermManager,
    ) -> Vec<Vec<TermId>> {
        // Select patterns that together cover all variables
        let mut selected = Vec::new();
        let mut covered_vars: FxHashSet<Spur> = FxHashSet::default();

        for pattern in &quantifier.patterns {
            let pattern_vars = self.collect_vars_in_pattern(pattern, manager);
            let new_vars: FxHashSet<_> = pattern_vars.difference(&covered_vars).copied().collect();

            if !new_vars.is_empty() {
                selected.push(pattern.clone());
                covered_vars.extend(new_vars);
            }

            if covered_vars.len() >= quantifier.num_vars() {
                break;
            }
        }

        selected
    }

    fn select_loop_avoiding_patterns(
        &self,
        quantifier: &QuantifiedFormula,
        manager: &TermManager,
    ) -> Vec<Vec<TermId>> {
        // Avoid patterns that contain the quantified function symbol
        quantifier
            .patterns
            .iter()
            .filter(|pattern| !self.contains_quantified_symbol(pattern, quantifier, manager))
            .cloned()
            .collect()
    }

    fn count_vars_in_pattern(&self, pattern: &[TermId], manager: &TermManager) -> usize {
        self.collect_vars_in_pattern(pattern, manager).len()
    }

    fn collect_vars_in_pattern(
        &self,
        pattern: &[TermId],
        manager: &TermManager,
    ) -> FxHashSet<Spur> {
        let mut vars = FxHashSet::default();
        let mut visited = FxHashSet::default();

        for &term in pattern {
            self.collect_vars_rec(term, &mut vars, &mut visited, manager);
        }

        vars
    }

    fn collect_vars_rec(
        &self,
        term: TermId,
        vars: &mut FxHashSet<Spur>,
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
            vars.insert(name);
            return;
        }

        match &t.kind {
            TermKind::Apply { args, .. } => {
                for &arg in args.iter() {
                    self.collect_vars_rec(arg, vars, visited, manager);
                }
            }
            TermKind::Not(arg) | TermKind::Neg(arg) => {
                self.collect_vars_rec(*arg, vars, visited, manager);
            }
            _ => {}
        }
    }

    fn contains_quantified_symbol(
        &self,
        pattern: &[TermId],
        _quantifier: &QuantifiedFormula,
        manager: &TermManager,
    ) -> bool {
        for &term in pattern {
            if self.is_function_application(term, manager) {
                return true;
            }
        }
        false
    }

    fn is_function_application(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(t) = manager.get(term) else {
            return false;
        };
        matches!(t.kind, TermKind::Apply { .. })
    }

    /// Record success or failure of an instantiation
    pub fn record_result(&mut self, quantifier: TermId, success: bool) {
        let entry = self
            .success_history
            .entry(quantifier)
            .or_insert_with(SuccessRate::new);
        entry.record(success);
    }

    /// Get success rate for a quantifier
    pub fn success_rate(&self, quantifier: TermId) -> f64 {
        self.success_history
            .get(&quantifier)
            .map_or(0.5, |sr| sr.rate())
    }
}

/// Success rate tracker
#[derive(Debug, Clone)]
struct SuccessRate {
    successes: usize,
    failures: usize,
}

impl SuccessRate {
    fn new() -> Self {
        Self {
            successes: 0,
            failures: 0,
        }
    }

    fn record(&mut self, success: bool) {
        if success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }
    }

    fn rate(&self) -> f64 {
        let total = self.successes + self.failures;
        if total == 0 {
            0.5
        } else {
            self.successes as f64 / total as f64
        }
    }
}

/// Priority queue for quantifiers
#[derive(Debug)]
pub struct QuantifierQueue {
    /// Heap of scored quantifiers
    heap: BinaryHeap<ScoredQuantifier>,
    /// Heuristic for scoring
    heuristic: InstantiationHeuristic,
}

impl QuantifierQueue {
    /// Create a new queue
    pub fn new(heuristic: InstantiationHeuristic) -> Self {
        Self {
            heap: BinaryHeap::new(),
            heuristic,
        }
    }

    /// Add a quantifier to the queue
    pub fn push(
        &mut self,
        quantifier: QuantifiedFormula,
        model: &CompletedModel,
        manager: &TermManager,
    ) {
        let score = self
            .heuristic
            .calculate_priority(&quantifier, model, manager);
        self.heap.push(ScoredQuantifier { quantifier, score });
    }

    /// Pop the highest priority quantifier
    pub fn pop(&mut self) -> Option<QuantifiedFormula> {
        self.heap.pop().map(|sq| sq.quantifier)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.heap.len()
    }
}

/// Scored quantifier for priority queue
#[derive(Debug, Clone)]
struct ScoredQuantifier {
    quantifier: QuantifiedFormula,
    score: f64,
}

impl PartialEq for ScoredQuantifier {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredQuantifier {}

impl PartialOrd for ScoredQuantifier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredQuantifier {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score = higher priority (max-heap)
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
    }
}

/// Policy for multi-trigger scoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScorerPolicy {
    /// Conservative: current behavior, prefer simpler trigger sets
    Conservative,
    /// Ranked: score by depth, shared variables, and ground-term reachability
    Ranked,
}

/// A trigger set (a multi-pattern: a list of terms that together cover all variables)
#[derive(Debug, Clone)]
pub struct TriggerSet {
    /// The pattern terms forming this trigger set
    pub terms: Vec<TermId>,
    /// Syntactic depth of the deepest term in this set
    pub max_depth: usize,
    /// Number of distinct variables shared across terms
    pub shared_var_count: usize,
}

impl TriggerSet {
    /// Create a new trigger set from pattern terms
    pub fn new(terms: Vec<TermId>) -> Self {
        Self {
            terms,
            max_depth: 0,
            shared_var_count: 0,
        }
    }

    /// Create with precomputed metrics
    pub fn with_metrics(terms: Vec<TermId>, max_depth: usize, shared_var_count: usize) -> Self {
        Self {
            terms,
            max_depth,
            shared_var_count,
        }
    }
}

/// A trigger set annotated with a ranking score
#[derive(Debug, Clone)]
pub struct ScoredTriggerSet {
    /// The trigger set
    pub triggers: TriggerSet,
    /// Score (higher is better / higher priority)
    pub score: f64,
}

/// Multi-trigger scorer: ranks candidate trigger sets
#[derive(Debug, Clone)]
pub struct MultiTriggerScorer {
    /// Scoring policy
    pub policy: ScorerPolicy,
    /// Number of top candidates to return
    pub top_k: usize,
}

impl MultiTriggerScorer {
    /// Create a new scorer
    pub fn new(policy: ScorerPolicy, top_k: usize) -> Self {
        Self { policy, top_k }
    }

    /// Score a collection of trigger-set candidates and return the top-k.
    ///
    /// Scoring criteria (Ranked policy):
    ///   (a) syntactic depth: deeper terms get a lower score
    ///   (b) shared variable count: more shared variables get a higher score
    ///   (c) ground-term reachability: trigger sets whose terms appear in
    ///       the equality graph get a bonus
    pub fn score_trigger_sets(
        &self,
        candidates: &[TriggerSet],
        manager: &TermManager,
    ) -> Vec<ScoredTriggerSet> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<ScoredTriggerSet> = candidates
            .iter()
            .map(|ts| {
                let score = self.compute_score(ts, manager);
                ScoredTriggerSet {
                    triggers: ts.clone(),
                    score,
                }
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        scored.truncate(self.top_k);
        scored
    }

    fn compute_score(&self, ts: &TriggerSet, manager: &TermManager) -> f64 {
        match self.policy {
            ScorerPolicy::Conservative => {
                // Conservative: prefer smaller trigger sets
                1.0 / (1.0 + ts.terms.len() as f64)
            }
            ScorerPolicy::Ranked => self.ranked_score(ts, manager),
        }
    }

    fn ranked_score(&self, ts: &TriggerSet, manager: &TermManager) -> f64 {
        // (a) Depth component: deeper terms are less preferred.
        //     depth_penalty lowers score as max_depth increases.
        let depth = if ts.max_depth > 0 {
            ts.max_depth
        } else {
            // Compute depth on the fly if not pre-set
            ts.terms
                .iter()
                .map(|&t| self.term_depth(t, manager, 0, 20))
                .max()
                .unwrap_or(0)
        };
        let depth_component = 1.0 / (1.0 + depth as f64);

        // (b) Shared variable component: more shared variables → higher score.
        let shared = if ts.shared_var_count > 0 {
            ts.shared_var_count
        } else {
            // Compute shared vars on the fly
            self.count_shared_vars(&ts.terms, manager)
        };
        let shared_component = 1.0 + shared as f64;

        // (c) Ground-term reachability: terms that are ground (no free vars) are
        //     reachable in the E-graph and provide stronger triggering.
        let ground_count = ts
            .terms
            .iter()
            .filter(|&&t| self.is_ground(t, manager))
            .count();
        let ground_component = 1.0 + ground_count as f64 * 0.5;

        depth_component * shared_component * ground_component
    }

    /// Compute term depth (capped at `max_depth_cap` to avoid large recursion)
    fn term_depth(&self, term: TermId, manager: &TermManager, current: usize, cap: usize) -> usize {
        if current >= cap {
            return current;
        }
        let Some(t) = manager.get(term) else {
            return current;
        };
        match &t.kind {
            TermKind::Apply { args, .. } => args
                .iter()
                .map(|&a| self.term_depth(a, manager, current + 1, cap))
                .max()
                .unwrap_or(current),
            TermKind::Not(a) | TermKind::Neg(a) => self.term_depth(*a, manager, current + 1, cap),
            TermKind::And(args) | TermKind::Or(args) => args
                .iter()
                .map(|&a| self.term_depth(a, manager, current + 1, cap))
                .max()
                .unwrap_or(current),
            TermKind::Eq(l, r)
            | TermKind::Lt(l, r)
            | TermKind::Le(l, r)
            | TermKind::Gt(l, r)
            | TermKind::Ge(l, r) => {
                let ld = self.term_depth(*l, manager, current + 1, cap);
                let rd = self.term_depth(*r, manager, current + 1, cap);
                ld.max(rd)
            }
            _ => current,
        }
    }

    /// Count variables that appear in more than one term (shared across the trigger set)
    fn count_shared_vars(&self, terms: &[TermId], manager: &TermManager) -> usize {
        if terms.len() <= 1 {
            return 0;
        }

        let var_sets: Vec<FxHashSet<Spur>> = terms
            .iter()
            .map(|&t| self.collect_vars(t, manager))
            .collect();

        let mut frequencies: FxHashMap<Spur, usize> = FxHashMap::default();
        for vars in &var_sets {
            for &var in vars {
                *frequencies.entry(var).or_insert(0) += 1;
            }
        }

        frequencies.values().filter(|&&count| count >= 2).count()
    }

    fn collect_vars(&self, term: TermId, manager: &TermManager) -> FxHashSet<Spur> {
        let mut vars = FxHashSet::default();
        let mut visited = FxHashSet::default();
        self.collect_vars_rec(term, manager, &mut vars, &mut visited);
        vars
    }

    fn collect_vars_rec(
        &self,
        term: TermId,
        manager: &TermManager,
        vars: &mut FxHashSet<Spur>,
        visited: &mut FxHashSet<TermId>,
    ) {
        if !visited.insert(term) {
            return;
        }
        let Some(t) = manager.get(term) else {
            return;
        };
        if let TermKind::Var(name) = t.kind {
            vars.insert(name);
            return;
        }
        match &t.kind {
            TermKind::Apply { args, .. } => {
                for &a in args.iter() {
                    self.collect_vars_rec(a, manager, vars, visited);
                }
            }
            TermKind::Not(a) | TermKind::Neg(a) => {
                self.collect_vars_rec(*a, manager, vars, visited);
            }
            TermKind::And(args) | TermKind::Or(args) => {
                for &a in args {
                    self.collect_vars_rec(a, manager, vars, visited);
                }
            }
            TermKind::Eq(l, r)
            | TermKind::Lt(l, r)
            | TermKind::Le(l, r)
            | TermKind::Gt(l, r)
            | TermKind::Ge(l, r) => {
                self.collect_vars_rec(*l, manager, vars, visited);
                self.collect_vars_rec(*r, manager, vars, visited);
            }
            _ => {}
        }
    }

    /// Return true if the term contains no free variables
    fn is_ground(&self, term: TermId, manager: &TermManager) -> bool {
        self.collect_vars(term, manager).is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbqi_heuristics_creation() {
        let heuristics = MBQIHeuristics::new();
        assert!(heuristics.enable_conflict_analysis);
    }

    #[test]
    fn test_conservative_heuristics() {
        let heuristics = MBQIHeuristics::conservative();
        assert_eq!(
            heuristics.quantifier_selection,
            SelectionStrategy::MostConstrained
        );
    }

    #[test]
    fn test_aggressive_heuristics() {
        let heuristics = MBQIHeuristics::aggressive();
        assert_eq!(
            heuristics.quantifier_selection,
            SelectionStrategy::BreadthFirst
        );
    }

    #[test]
    fn test_instantiation_heuristic_creation() {
        let config = MBQIHeuristics::new();
        let heuristic = InstantiationHeuristic::new(config);
        assert_eq!(heuristic.quantifier_scores.len(), 0);
    }

    #[test]
    fn test_success_rate_tracker() {
        let mut sr = SuccessRate::new();
        assert_eq!(sr.rate(), 0.5);

        sr.record(true);
        assert_eq!(sr.rate(), 1.0);

        sr.record(false);
        assert_eq!(sr.rate(), 0.5);
    }

    #[test]
    fn test_quantifier_queue_creation() {
        let config = MBQIHeuristics::new();
        let heuristic = InstantiationHeuristic::new(config);
        let queue = QuantifierQueue::new(heuristic);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_selection_strategy_equality() {
        assert_eq!(SelectionStrategy::Sequential, SelectionStrategy::Sequential);
        assert_ne!(SelectionStrategy::Sequential, SelectionStrategy::Random);
    }

    #[test]
    fn test_trigger_selection_equality() {
        assert_eq!(TriggerSelection::All, TriggerSelection::All);
        assert_ne!(TriggerSelection::All, TriggerSelection::MinVars);
    }

    #[test]
    fn test_resource_allocation_equality() {
        assert_eq!(ResourceAllocation::Balanced, ResourceAllocation::Balanced);
        assert_ne!(ResourceAllocation::Balanced, ResourceAllocation::Aggressive);
    }
}
