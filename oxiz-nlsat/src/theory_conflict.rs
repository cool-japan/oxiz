//! Theory conflict tracking for improved decision heuristics.
//!
//! This module tracks which variables and polynomials are involved in
//! theory conflicts (CAD reasoning failures) to improve variable ordering
//! and decision heuristics. Variables that frequently appear in conflicts
//! should be decided earlier.
//!
//! Key features:
//! - **Conflict Variable Tracking**: Track which variables appear in theory conflicts
//! - **Activity Boosting**: Boost VSIDS scores for conflict variables
//! - **Polynomial Conflict Analysis**: Identify problematic polynomial patterns
//! - **Decay Management**: Gradually decay conflict scores over time
//!
//! Reference: Z3's activity-based variable ordering with theory integration

use oxiz_math::polynomial::{Polynomial, Var};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

/// Statistics for theory conflict tracking.
#[derive(Debug, Clone, Default)]
pub struct TheoryConflictStats {
    /// Total number of theory conflicts tracked.
    pub num_conflicts: u64,
    /// Number of variable activity boosts applied.
    pub num_boosts: u64,
    /// Number of conflict polynomials analyzed.
    pub num_polynomials_analyzed: u64,
}

/// Configuration for theory conflict tracking.
#[derive(Debug, Clone)]
pub struct TheoryConflictConfig {
    /// Activity boost factor for variables in conflicts.
    pub boost_factor: f64,
    /// Decay factor for conflict scores (0.0-1.0).
    pub decay_factor: f64,
    /// Maximum history size for recent conflicts.
    pub max_history: usize,
    /// Enable polynomial pattern analysis.
    pub enable_pattern_analysis: bool,
}

impl Default for TheoryConflictConfig {
    fn default() -> Self {
        Self {
            boost_factor: 2.0,
            decay_factor: 0.95,
            max_history: 100,
            enable_pattern_analysis: true,
        }
    }
}

/// Information about a single theory conflict.
#[derive(Debug, Clone)]
pub struct ConflictInfo {
    /// Variables involved in the conflict.
    pub variables: Vec<Var>,
    /// Polynomials involved in the conflict.
    pub polynomials: Vec<Polynomial>,
    /// Conflict score (higher = more important).
    pub score: f64,
}

/// Theory conflict tracker.
pub struct TheoryConflictTracker {
    /// Configuration.
    config: TheoryConflictConfig,
    /// Variable conflict scores.
    var_scores: FxHashMap<Var, f64>,
    /// Recent conflict history.
    conflict_history: VecDeque<ConflictInfo>,
    /// Statistics.
    stats: TheoryConflictStats,
}

impl TheoryConflictTracker {
    /// Create a new theory conflict tracker.
    pub fn new() -> Self {
        Self::with_config(TheoryConflictConfig::default())
    }

    /// Create a new tracker with the given configuration.
    pub fn with_config(config: TheoryConflictConfig) -> Self {
        Self {
            config,
            var_scores: FxHashMap::default(),
            conflict_history: VecDeque::new(),
            stats: TheoryConflictStats::default(),
        }
    }

    /// Record a theory conflict involving the given variables and polynomials.
    pub fn record_conflict(&mut self, variables: Vec<Var>, polynomials: Vec<Polynomial>) {
        self.stats.num_conflicts += 1;
        self.stats.num_polynomials_analyzed += polynomials.len() as u64;

        // Calculate conflict score based on problem size
        let score = self.calculate_conflict_score(&variables, &polynomials);

        // Boost activity for all involved variables
        for &var in &variables {
            let current_score = self.var_scores.get(&var).copied().unwrap_or(0.0);
            let new_score = current_score + score * self.config.boost_factor;
            self.var_scores.insert(var, new_score);
            self.stats.num_boosts += 1;
        }

        // Add to history
        let info = ConflictInfo {
            variables: variables.clone(),
            polynomials,
            score,
        };

        self.conflict_history.push_front(info);

        // Limit history size
        while self.conflict_history.len() > self.config.max_history {
            self.conflict_history.pop_back();
        }

        // Apply decay to old scores
        self.decay_scores();
    }

    /// Calculate the score for a conflict based on its characteristics.
    fn calculate_conflict_score(&self, variables: &[Var], polynomials: &[Polynomial]) -> f64 {
        let mut score = 1.0;

        // Small conflicts are more important (indicate tight constraints)
        if !variables.is_empty() {
            score += 10.0 / variables.len() as f64;
        }

        // High-degree polynomials indicate harder problems
        if self.config.enable_pattern_analysis {
            let max_degree = polynomials
                .iter()
                .map(|p| p.total_degree() as usize)
                .max()
                .unwrap_or(0);
            score += max_degree as f64 * 0.5;
        }

        score
    }

    /// Decay all variable conflict scores.
    fn decay_scores(&mut self) {
        for score in self.var_scores.values_mut() {
            *score *= self.config.decay_factor;
        }
    }

    /// Get the conflict score for a variable.
    pub fn get_score(&self, var: Var) -> f64 {
        self.var_scores.get(&var).copied().unwrap_or(0.0)
    }

    /// Get all variables sorted by conflict score (descending).
    pub fn get_ranked_variables(&self) -> Vec<(Var, f64)> {
        let mut vars: Vec<_> = self.var_scores.iter().map(|(&v, &s)| (v, s)).collect();
        vars.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        vars
    }

    /// Get the most conflicting variables (top N by score).
    pub fn get_top_conflicts(&self, n: usize) -> Vec<Var> {
        let ranked = self.get_ranked_variables();
        ranked.into_iter().take(n).map(|(v, _)| v).collect()
    }

    /// Check if a variable has been involved in recent conflicts.
    pub fn is_conflict_variable(&self, var: Var) -> bool {
        self.get_score(var) > 0.1
    }

    /// Get recent conflict history.
    pub fn get_recent_conflicts(&self, n: usize) -> Vec<&ConflictInfo> {
        self.conflict_history.iter().take(n).collect()
    }

    /// Analyze patterns in recent conflicts.
    pub fn analyze_patterns(&self) -> ConflictPatterns {
        let mut patterns = ConflictPatterns {
            frequent_vars: Vec::new(),
            avg_conflict_size: 0.0,
            max_degree_seen: 0,
            total_conflicts: self.stats.num_conflicts,
        };

        if self.conflict_history.is_empty() {
            return patterns;
        }

        // Find frequently occurring variables
        patterns.frequent_vars = self.get_top_conflicts(10);

        // Calculate average conflict size
        let total_size: usize = self
            .conflict_history
            .iter()
            .map(|c| c.variables.len())
            .sum();
        patterns.avg_conflict_size = total_size as f64 / self.conflict_history.len() as f64;

        // Find maximum polynomial degree
        for conflict in &self.conflict_history {
            for poly in &conflict.polynomials {
                let degree = poly.total_degree() as usize;
                patterns.max_degree_seen = patterns.max_degree_seen.max(degree);
            }
        }

        patterns
    }

    /// Clear all conflict history and scores.
    pub fn clear(&mut self) {
        self.var_scores.clear();
        self.conflict_history.clear();
    }

    /// Get statistics.
    pub fn stats(&self) -> &TheoryConflictStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = TheoryConflictStats::default();
    }
}

impl Default for TheoryConflictTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Patterns observed in conflicts.
#[derive(Debug, Clone)]
pub struct ConflictPatterns {
    /// Most frequently conflicting variables.
    pub frequent_vars: Vec<Var>,
    /// Average number of variables per conflict.
    pub avg_conflict_size: f64,
    /// Maximum polynomial degree seen in conflicts.
    pub max_degree_seen: usize,
    /// Total number of conflicts analyzed.
    pub total_conflicts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    #[allow(dead_code)]
    fn constant(n: i32) -> Polynomial {
        Polynomial::constant(BigRational::from_integer(BigInt::from(n)))
    }

    #[test]
    fn test_tracker_new() {
        let tracker = TheoryConflictTracker::new();
        assert_eq!(tracker.stats.num_conflicts, 0);
    }

    #[test]
    fn test_record_conflict() {
        let mut tracker = TheoryConflictTracker::new();
        let vars = vec![0, 1, 2];
        let polys = vec![Polynomial::from_var(0)];

        tracker.record_conflict(vars, polys);

        assert_eq!(tracker.stats.num_conflicts, 1);
        assert_eq!(tracker.stats.num_boosts, 3); // One boost per variable
        assert!(tracker.get_score(0) > 0.0);
    }

    #[test]
    fn test_score_ordering() {
        let mut tracker = TheoryConflictTracker::new();

        // Variable 0 in one conflict
        tracker.record_conflict(vec![0], vec![Polynomial::from_var(0)]);

        // Variable 1 in two conflicts
        tracker.record_conflict(vec![1], vec![Polynomial::from_var(1)]);
        tracker.record_conflict(vec![1], vec![Polynomial::from_var(1)]);

        // Variable 1 should have higher score
        assert!(tracker.get_score(1) > tracker.get_score(0));
    }

    #[test]
    fn test_ranked_variables() {
        let mut tracker = TheoryConflictTracker::new();

        tracker.record_conflict(vec![0], vec![]);
        tracker.record_conflict(vec![1], vec![]);
        tracker.record_conflict(vec![1], vec![]);
        tracker.record_conflict(vec![2], vec![]);
        tracker.record_conflict(vec![2], vec![]);
        tracker.record_conflict(vec![2], vec![]);

        let ranked = tracker.get_ranked_variables();
        assert_eq!(ranked.len(), 3);
        // Variable 2 should be first (most conflicts)
        assert_eq!(ranked[0].0, 2);
    }

    #[test]
    fn test_top_conflicts() {
        let mut tracker = TheoryConflictTracker::new();

        tracker.record_conflict(vec![0], vec![]);
        tracker.record_conflict(vec![1, 1], vec![]);
        tracker.record_conflict(vec![2, 2, 2], vec![]);

        let top = tracker.get_top_conflicts(2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_is_conflict_variable() {
        let mut tracker = TheoryConflictTracker::new();

        assert!(!tracker.is_conflict_variable(0));

        tracker.record_conflict(vec![0], vec![]);

        assert!(tracker.is_conflict_variable(0));
        assert!(!tracker.is_conflict_variable(1));
    }

    #[test]
    fn test_conflict_history() {
        let mut tracker = TheoryConflictTracker::new();

        tracker.record_conflict(vec![0], vec![]);
        tracker.record_conflict(vec![1], vec![]);

        let recent = tracker.get_recent_conflicts(5);
        assert_eq!(recent.len(), 2);
    }

    #[test]
    fn test_history_limit() {
        let mut config = TheoryConflictConfig::default();
        config.max_history = 3;
        let mut tracker = TheoryConflictTracker::with_config(config);

        for i in 0..5 {
            tracker.record_conflict(vec![i], vec![]);
        }

        assert_eq!(tracker.conflict_history.len(), 3);
    }

    #[test]
    fn test_analyze_patterns() {
        let mut tracker = TheoryConflictTracker::new();

        // x^2 polynomial (degree 2)
        let x = Polynomial::from_var(0);
        let x_squared = Polynomial::mul(&x, &x);

        tracker.record_conflict(vec![0, 1], vec![x_squared]);
        tracker.record_conflict(vec![1, 2], vec![]);

        let patterns = tracker.analyze_patterns();
        assert_eq!(patterns.total_conflicts, 2);
        assert_eq!(patterns.max_degree_seen, 2);
        assert!(patterns.avg_conflict_size > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut tracker = TheoryConflictTracker::new();

        tracker.record_conflict(vec![0], vec![]);
        assert!(tracker.get_score(0) > 0.0);

        tracker.clear();
        assert_eq!(tracker.get_score(0), 0.0);
        assert_eq!(tracker.conflict_history.len(), 0);
    }

    #[test]
    fn test_decay() {
        let config = TheoryConflictConfig {
            decay_factor: 0.5,
            ..Default::default()
        };
        let mut tracker = TheoryConflictTracker::with_config(config);

        tracker.record_conflict(vec![0], vec![]);
        let initial_score = tracker.get_score(0);

        // Record another conflict (this triggers decay)
        tracker.record_conflict(vec![1], vec![]);

        // Score for variable 0 should have decayed
        let decayed_score = tracker.get_score(0);
        assert!(decayed_score < initial_score);
    }
}
