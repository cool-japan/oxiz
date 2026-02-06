//! Instantiation heuristics for E-matching

use crate::ast::TermId;
use rustc_hash::FxHashSet;

/// Configuration for instantiation heuristics
#[derive(Debug, Clone)]
pub struct HeuristicConfig {
    /// The heuristic strategy to use
    pub strategy: HeuristicStrategy,
    /// Weight for conflict-driven priority
    pub conflict_weight: f64,
    /// Weight for greedy priority
    pub greedy_weight: f64,
}

/// Heuristic strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeuristicStrategy {
    /// Greedy instantiation based on term size
    Greedy,
    /// Prioritize terms involved in conflicts
    ConflictDriven,
    /// Hybrid approach combining greedy and conflict-driven
    Hybrid,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            strategy: HeuristicStrategy::Hybrid,
            conflict_weight: 0.7,
            greedy_weight: 0.3,
        }
    }
}

/// Priority for instantiation
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct InstantiationPriority(pub f64);

/// Trait for instantiation heuristics
pub trait InstantiationHeuristic {
    /// Compute priority for instantiating a quantifier with given instances
    fn compute_priority(&self, quant: TermId, instances: &[TermId]) -> InstantiationPriority;
    /// Select the top instantiations from candidates up to the given limit
    fn select_instantiations(
        &self,
        candidates: Vec<(TermId, Vec<TermId>)>,
        limit: usize,
    ) -> Vec<(TermId, Vec<TermId>)>;
}

/// Greedy heuristic
#[derive(Debug)]
pub struct GreedyHeuristic {
    #[allow(dead_code)]
    config: HeuristicConfig,
}

impl GreedyHeuristic {
    /// Create a new greedy heuristic with the given configuration
    pub fn new(config: HeuristicConfig) -> Self {
        Self { config }
    }
}

impl InstantiationHeuristic for GreedyHeuristic {
    fn compute_priority(&self, _quant: TermId, instances: &[TermId]) -> InstantiationPriority {
        InstantiationPriority(1.0 / (instances.len() as f64 + 1.0))
    }

    fn select_instantiations(
        &self,
        mut candidates: Vec<(TermId, Vec<TermId>)>,
        limit: usize,
    ) -> Vec<(TermId, Vec<TermId>)> {
        candidates.sort_by(|a, b| {
            let pa = self.compute_priority(a.0, &a.1);
            let pb = self.compute_priority(b.0, &b.1);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        candidates
    }
}

/// Conflict-driven heuristic
#[derive(Debug)]
pub struct ConflictDrivenHeuristic {
    #[allow(dead_code)]
    config: HeuristicConfig,
    conflict_terms: FxHashSet<TermId>,
}

impl ConflictDrivenHeuristic {
    /// Create a new conflict-driven heuristic with the given configuration
    pub fn new(config: HeuristicConfig) -> Self {
        Self {
            config,
            conflict_terms: FxHashSet::default(),
        }
    }

    /// Record that a term was involved in a conflict
    pub fn record_conflict(&mut self, term: TermId) {
        self.conflict_terms.insert(term);
    }
}

impl InstantiationHeuristic for ConflictDrivenHeuristic {
    fn compute_priority(&self, _quant: TermId, instances: &[TermId]) -> InstantiationPriority {
        let conflict_count = instances
            .iter()
            .filter(|t| self.conflict_terms.contains(t))
            .count();
        InstantiationPriority(conflict_count as f64)
    }

    fn select_instantiations(
        &self,
        mut candidates: Vec<(TermId, Vec<TermId>)>,
        limit: usize,
    ) -> Vec<(TermId, Vec<TermId>)> {
        candidates.sort_by(|a, b| {
            let pa = self.compute_priority(a.0, &a.1);
            let pb = self.compute_priority(b.0, &b.1);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        candidates
    }
}

/// Hybrid heuristic
#[derive(Debug)]
pub struct HybridHeuristic {
    config: HeuristicConfig,
    greedy: GreedyHeuristic,
    conflict: ConflictDrivenHeuristic,
}

impl HybridHeuristic {
    /// Create a new hybrid heuristic with the given configuration
    pub fn new(config: HeuristicConfig) -> Self {
        Self {
            greedy: GreedyHeuristic::new(config.clone()),
            conflict: ConflictDrivenHeuristic::new(config.clone()),
            config,
        }
    }

    /// Record that a term was involved in a conflict
    pub fn record_conflict(&mut self, term: TermId) {
        self.conflict.record_conflict(term);
    }
}

impl InstantiationHeuristic for HybridHeuristic {
    fn compute_priority(&self, quant: TermId, instances: &[TermId]) -> InstantiationPriority {
        let greedy_pri = self.greedy.compute_priority(quant, instances);
        let conflict_pri = self.conflict.compute_priority(quant, instances);
        InstantiationPriority(
            self.config.greedy_weight * greedy_pri.0 + self.config.conflict_weight * conflict_pri.0,
        )
    }

    fn select_instantiations(
        &self,
        mut candidates: Vec<(TermId, Vec<TermId>)>,
        limit: usize,
    ) -> Vec<(TermId, Vec<TermId>)> {
        candidates.sort_by(|a, b| {
            let pa = self.compute_priority(a.0, &a.1);
            let pb = self.compute_priority(b.0, &b.1);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = HeuristicConfig::default();
        assert_eq!(config.strategy, HeuristicStrategy::Hybrid);
    }
}
