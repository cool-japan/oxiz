//! MBQI Heuristics and Selection Strategies
//!
//! This module implements various heuristics for guiding MBQI, including:
//! - Quantifier selection strategies
//! - Trigger/pattern selection
//! - Instantiation ordering
//! - Resource allocation

use lasso::Spur;
use oxiz_core::ast::{TermId, TermKind, TermManager};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::QuantifiedFormula;
use super::model_completion::CompletedModel;

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
