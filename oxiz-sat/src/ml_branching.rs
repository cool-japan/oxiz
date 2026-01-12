//! Machine Learning-Based Branching and Restart Strategies
//!
//! This module implements online learning algorithms to guide SAT solver
//! decisions. The ML system learns from solving history to predict good
//! variable selections and restart timings.
//!
//! Key features:
//! - Online learning from conflict patterns
//! - Exponential weighted averaging for variable scores
//! - Reinforcement learning-style feedback
//! - Pattern recognition for branching heuristics
//! - Adaptive restart prediction
//! - Pure Rust implementation (no external ML libraries)

use crate::literal::{Lit, Var};
use std::collections::HashMap;

/// Statistics for ML-based branching
#[derive(Debug, Default, Clone)]
pub struct MLBranchingStats {
    /// Number of predictions made
    pub predictions: usize,
    /// Number of correct predictions (led to conflicts)
    pub correct_predictions: usize,
    /// Number of incorrect predictions
    pub incorrect_predictions: usize,
    /// Total learning updates
    pub learning_updates: usize,
    /// Average prediction confidence
    pub avg_confidence: f64,
    /// Number of restarts predicted
    pub restart_predictions: usize,
}

impl MLBranchingStats {
    /// Calculate prediction accuracy
    pub fn accuracy(&self) -> f64 {
        if self.predictions == 0 {
            0.0
        } else {
            self.correct_predictions as f64 / self.predictions as f64
        }
    }

    /// Display statistics
    pub fn display(&self) -> String {
        format!(
            "ML Branching Stats:\n\
             - Predictions: {}\n\
             - Accuracy: {:.2}%\n\
             - Correct: {} / Incorrect: {}\n\
             - Learning updates: {}\n\
             - Avg confidence: {:.4}\n\
             - Restart predictions: {}",
            self.predictions,
            self.accuracy() * 100.0,
            self.correct_predictions,
            self.incorrect_predictions,
            self.learning_updates,
            self.avg_confidence,
            self.restart_predictions,
        )
    }
}

/// Feature vector for a variable
#[derive(Debug, Clone)]
struct VariableFeatures {
    /// Conflict participation rate (exponential moving average)
    conflict_rate: f64,
    /// Propagation success rate
    propagation_rate: f64,
    /// Decision depth preference
    depth_preference: f64,
    /// Recent activity score
    activity: f64,
    /// Phase consistency (how often same phase leads to conflicts)
    phase_consistency: f64,
}

impl Default for VariableFeatures {
    fn default() -> Self {
        Self {
            conflict_rate: 0.5,
            propagation_rate: 0.5,
            depth_preference: 0.5,
            activity: 0.0,
            phase_consistency: 0.5,
        }
    }
}

/// Configuration for ML-based branching
#[derive(Debug, Clone)]
pub struct MLBranchingConfig {
    /// Learning rate for exponential weighted averaging
    pub learning_rate: f64,
    /// Discount factor for older observations
    pub discount_factor: f64,
    /// Exploration rate (epsilon for epsilon-greedy)
    pub exploration_rate: f64,
    /// Minimum confidence threshold for predictions
    pub min_confidence: f64,
    /// Enable feature normalization
    pub normalize_features: bool,
}

impl Default for MLBranchingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            discount_factor: 0.95,
            exploration_rate: 0.1,
            min_confidence: 0.6,
            normalize_features: true,
        }
    }
}

/// ML-based branching heuristic
pub struct MLBranching {
    /// Configuration
    config: MLBranchingConfig,
    /// Learned features for each variable
    features: HashMap<Var, VariableFeatures>,
    /// Statistics
    stats: MLBranchingStats,
    /// Conflict history (recent conflicts for pattern learning)
    conflict_history: Vec<Vec<Lit>>,
    /// Maximum history size
    max_history: usize,
    /// Current prediction confidence
    current_confidence: f64,
}

impl MLBranching {
    /// Create a new ML branching heuristic
    pub fn new(config: MLBranchingConfig) -> Self {
        Self {
            config,
            features: HashMap::new(),
            stats: MLBranchingStats::default(),
            conflict_history: Vec::new(),
            max_history: 100,
            current_confidence: 0.5,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(MLBranchingConfig::default())
    }

    /// Predict the best variable to branch on
    ///
    /// Returns (variable, predicted_polarity, confidence)
    pub fn predict_branch(&mut self, candidates: &[Var]) -> Option<(Var, bool, f64)> {
        if candidates.is_empty() {
            return None;
        }

        // Epsilon-greedy: sometimes explore random variables
        if self.should_explore() {
            let idx = self.random_index(candidates.len());
            let var = candidates[idx];
            let polarity = self.predict_polarity(var);
            self.stats.predictions += 1;
            return Some((var, polarity, 0.5));
        }

        // Compute scores for all candidates
        let mut best_var = candidates[0];
        let mut best_score = f64::NEG_INFINITY;

        for &var in candidates {
            let score = self.compute_variable_score(var);
            if score > best_score {
                best_score = score;
                best_var = var;
            }
        }

        let polarity = self.predict_polarity(best_var);
        let confidence = self.score_to_confidence(best_score);

        self.current_confidence = confidence;
        self.stats.predictions += 1;

        Some((best_var, polarity, confidence))
    }

    /// Compute ML-based score for a variable
    fn compute_variable_score(&self, var: Var) -> f64 {
        let features = self.features.get(&var).cloned().unwrap_or_default();

        // Weighted combination of features
        let w_conflict = 0.3;
        let w_propagation = 0.2;
        let w_depth = 0.15;
        let w_activity = 0.25;
        let w_consistency = 0.1;

        let score = w_conflict * features.conflict_rate
            + w_propagation * features.propagation_rate
            + w_depth * features.depth_preference
            + w_activity * features.activity
            + w_consistency * features.phase_consistency;

        if self.config.normalize_features {
            // Sigmoid normalization to [0, 1]
            1.0 / (1.0 + (-score).exp())
        } else {
            score
        }
    }

    /// Predict polarity for a variable based on phase consistency
    fn predict_polarity(&self, var: Var) -> bool {
        self.features
            .get(&var)
            .map(|f| f.phase_consistency > 0.5)
            .unwrap_or(true)
    }

    /// Convert score to confidence level
    fn score_to_confidence(&self, score: f64) -> f64 {
        // Map score to confidence using sigmoid
        let normalized = 1.0 / (1.0 + (-5.0 * (score - 0.5)).exp());
        normalized.clamp(0.0, 1.0)
    }

    /// Update features based on conflict outcome
    pub fn learn_from_conflict(
        &mut self,
        conflict_clause: &[Lit],
        decision_var: Var,
        was_correct: bool,
    ) {
        self.stats.learning_updates += 1;

        if was_correct {
            self.stats.correct_predictions += 1;
        } else {
            self.stats.incorrect_predictions += 1;
        }

        // Update conflict history
        self.conflict_history.push(conflict_clause.to_vec());
        if self.conflict_history.len() > self.max_history {
            self.conflict_history.remove(0);
        }

        // Update features for variables in conflict
        for &lit in conflict_clause {
            let var = lit.var();
            self.update_conflict_rate(var, 1.0);
        }

        // Update decision variable features
        self.update_decision_features(decision_var, was_correct);

        // Update average confidence
        let total = self.stats.correct_predictions + self.stats.incorrect_predictions;
        if total > 0 {
            self.stats.avg_confidence = (self.stats.avg_confidence * (total - 1) as f64
                + self.current_confidence)
                / total as f64;
        }
    }

    /// Update conflict rate for a variable
    fn update_conflict_rate(&mut self, var: Var, reward: f64) {
        let features = self.features.entry(var).or_default();
        let alpha = self.config.learning_rate;

        // Exponential weighted average
        features.conflict_rate = (1.0 - alpha) * features.conflict_rate + alpha * reward;
    }

    /// Update features for a decision variable
    fn update_decision_features(&mut self, var: Var, was_correct: bool) {
        let features = self.features.entry(var).or_default();
        let alpha = self.config.learning_rate;
        let reward = if was_correct { 1.0 } else { 0.0 };

        // Update activity with reward
        features.activity = (1.0 - alpha) * features.activity + alpha * reward;

        // Update phase consistency
        features.phase_consistency = (1.0 - alpha) * features.phase_consistency + alpha * reward;
    }

    /// Update propagation success for a variable
    pub fn update_propagation(&mut self, var: Var, success: bool) {
        let features = self.features.entry(var).or_default();
        let alpha = self.config.learning_rate;
        let reward = if success { 1.0 } else { 0.0 };

        features.propagation_rate = (1.0 - alpha) * features.propagation_rate + alpha * reward;
    }

    /// Update depth preference
    pub fn update_depth_preference(&mut self, var: Var, depth: usize, max_depth: usize) {
        if max_depth == 0 {
            return;
        }

        let features = self.features.entry(var).or_default();
        let alpha = self.config.learning_rate;
        let normalized_depth = depth as f64 / max_depth as f64;

        features.depth_preference =
            (1.0 - alpha) * features.depth_preference + alpha * normalized_depth;
    }

    /// Predict if a restart should occur based on learned patterns
    pub fn predict_restart(&mut self, conflicts_since_restart: usize, lbd_avg: f64) -> bool {
        // Simple heuristic: restart if conflict pattern suggests diminishing returns
        let pattern_score = self.analyze_conflict_patterns();

        // Combine with LBD (lower LBD = better quality clauses = less need to restart)
        let lbd_factor = if lbd_avg > 0.0 { 1.0 / lbd_avg } else { 1.0 };

        let restart_score = pattern_score * lbd_factor;

        // Dynamic threshold based on conflicts since last restart
        let threshold = 0.5 + (conflicts_since_restart as f64 / 1000.0) * 0.3;

        let should_restart = restart_score > threshold;

        if should_restart {
            self.stats.restart_predictions += 1;
        }

        should_restart
    }

    /// Analyze recent conflict patterns for quality trends
    fn analyze_conflict_patterns(&self) -> f64 {
        if self.conflict_history.len() < 2 {
            return 0.0;
        }

        // Compute similarity between recent conflicts (high similarity = stuck pattern)
        let recent = &self.conflict_history[self.conflict_history.len().saturating_sub(10)..];

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..recent.len().saturating_sub(1) {
            for j in (i + 1)..recent.len() {
                total_similarity += self.clause_similarity(&recent[i], &recent[j]);
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    /// Compute similarity between two clauses (Jaccard similarity)
    fn clause_similarity(&self, clause1: &[Lit], clause2: &[Lit]) -> f64 {
        let set1: std::collections::HashSet<_> = clause1.iter().map(|l| l.var()).collect();
        let set2: std::collections::HashSet<_> = clause2.iter().map(|l| l.var()).collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    /// Decay all feature scores (for aging)
    pub fn decay_features(&mut self) {
        let decay = self.config.discount_factor;

        for features in self.features.values_mut() {
            features.activity *= decay;
            features.conflict_rate *= decay;
        }
    }

    /// Check if should explore (epsilon-greedy)
    fn should_explore(&self) -> bool {
        // Simple pseudo-random based on stats
        let pseudo_rand = (self.stats.predictions * 2654435761) % 1000;
        (pseudo_rand as f64 / 1000.0) < self.config.exploration_rate
    }

    /// Get a pseudo-random index
    fn random_index(&self, max: usize) -> usize {
        if max == 0 {
            return 0;
        }
        (self.stats.predictions * 1103515245 + 12345) % max
    }

    /// Get statistics
    pub fn stats(&self) -> &MLBranchingStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MLBranchingStats::default();
    }

    /// Clear all learned features
    pub fn clear(&mut self) {
        self.features.clear();
        self.conflict_history.clear();
        self.stats = MLBranchingStats::default();
    }

    /// Get the number of learned variables
    pub fn num_learned_vars(&self) -> usize {
        self.features.len()
    }

    /// Export learned features for analysis
    pub fn export_features(&self) -> Vec<(Var, f64)> {
        self.features
            .iter()
            .map(|(&var, _features)| (var, self.compute_variable_score(var)))
            .collect()
    }
}

impl Default for MLBranching {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_branching_creation() {
        let ml = MLBranching::default();
        assert_eq!(ml.stats().predictions, 0);
        assert_eq!(ml.num_learned_vars(), 0);
    }

    #[test]
    fn test_predict_branch() {
        let mut ml = MLBranching::default();
        let candidates = vec![Var(0), Var(1), Var(2)];

        let result = ml.predict_branch(&candidates);
        assert!(result.is_some());

        let (var, _polarity, confidence) = result.unwrap();
        assert!(candidates.contains(&var));
        assert!((0.0..=1.0).contains(&confidence));
        assert_eq!(ml.stats().predictions, 1);
    }

    #[test]
    fn test_empty_candidates() {
        let mut ml = MLBranching::default();
        let candidates = vec![];

        let result = ml.predict_branch(&candidates);
        assert!(result.is_none());
    }

    #[test]
    fn test_learn_from_conflict() {
        let mut ml = MLBranching::default();
        let v0 = Var(0);
        let v1 = Var(1);

        let conflict = vec![Lit::pos(v0), Lit::neg(v1)];
        ml.learn_from_conflict(&conflict, v0, true);

        assert_eq!(ml.stats().learning_updates, 1);
        assert_eq!(ml.stats().correct_predictions, 1);
        assert!(ml.num_learned_vars() > 0);
    }

    #[test]
    fn test_update_propagation() {
        let mut ml = MLBranching::default();
        let v0 = Var(0);

        ml.update_propagation(v0, true);
        ml.update_propagation(v0, true);
        ml.update_propagation(v0, false);

        assert!(ml.features.contains_key(&v0));
    }

    #[test]
    fn test_update_depth_preference() {
        let mut ml = MLBranching::default();
        let v0 = Var(0);

        ml.update_depth_preference(v0, 5, 10);

        assert!(ml.features.contains_key(&v0));
        let features = ml.features.get(&v0).unwrap();
        assert!(features.depth_preference > 0.0);
    }

    #[test]
    fn test_predict_restart() {
        let mut ml = MLBranching::default();

        // Add some conflict history
        for i in 0..10 {
            let conflict = vec![Lit::pos(Var(i)), Lit::neg(Var(i + 1))];
            ml.conflict_history.push(conflict);
        }

        let _should_restart = ml.predict_restart(100, 3.0);
        assert!(ml.stats().restart_predictions <= 1);
    }

    #[test]
    fn test_decay_features() {
        let mut ml = MLBranching::default();
        let v0 = Var(0);

        // Set initial activity to a non-zero value
        ml.features.entry(v0).or_default().activity = 1.0;
        ml.features.entry(v0).or_default().conflict_rate = 0.8;

        let initial_activity = ml.features.get(&v0).unwrap().activity;
        let initial_conflict = ml.features.get(&v0).unwrap().conflict_rate;

        ml.decay_features();

        let decayed_activity = ml.features.get(&v0).unwrap().activity;
        let decayed_conflict = ml.features.get(&v0).unwrap().conflict_rate;

        assert!(decayed_activity < initial_activity);
        assert!(decayed_conflict < initial_conflict);
    }

    #[test]
    fn test_clause_similarity() {
        let ml = MLBranching::default();
        let v0 = Var(0);
        let v1 = Var(1);
        let v2 = Var(2);

        let clause1 = vec![Lit::pos(v0), Lit::pos(v1)];
        let clause2 = vec![Lit::pos(v0), Lit::pos(v2)];

        let similarity = ml.clause_similarity(&clause1, &clause2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_identical_clauses_similarity() {
        let ml = MLBranching::default();
        let v0 = Var(0);
        let v1 = Var(1);

        let clause1 = vec![Lit::pos(v0), Lit::pos(v1)];
        let clause2 = vec![Lit::pos(v0), Lit::pos(v1)];

        let similarity = ml.clause_similarity(&clause1, &clause2);
        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_stats_accuracy() {
        let mut stats = MLBranchingStats::default();
        assert_eq!(stats.accuracy(), 0.0);

        stats.predictions = 10;
        stats.correct_predictions = 7;
        stats.incorrect_predictions = 3;

        assert_eq!(stats.accuracy(), 0.7);
    }

    #[test]
    fn test_clear() {
        let mut ml = MLBranching::default();
        let v0 = Var(0);

        ml.update_propagation(v0, true);
        ml.stats.predictions = 10;

        ml.clear();

        assert_eq!(ml.num_learned_vars(), 0);
        assert_eq!(ml.stats().predictions, 0);
    }

    #[test]
    fn test_export_features() {
        let mut ml = MLBranching::default();
        let v0 = Var(0);
        let v1 = Var(1);

        ml.update_propagation(v0, true);
        ml.update_propagation(v1, true);

        let exported = ml.export_features();
        assert_eq!(exported.len(), 2);
    }

    #[test]
    fn test_config_default() {
        let config = MLBranchingConfig::default();
        assert!(config.learning_rate > 0.0);
        assert!(config.discount_factor > 0.0 && config.discount_factor < 1.0);
        assert!(config.exploration_rate >= 0.0 && config.exploration_rate <= 1.0);
    }

    #[test]
    fn test_stats_display() {
        let stats = MLBranchingStats {
            predictions: 100,
            correct_predictions: 75,
            incorrect_predictions: 25,
            learning_updates: 150,
            avg_confidence: 0.85,
            restart_predictions: 10,
        };

        let display = stats.display();
        assert!(display.contains("100"));
        assert!(display.contains("75.00%"));
    }
}
