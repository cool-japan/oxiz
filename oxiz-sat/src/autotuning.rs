//! Auto-tuning Framework
//!
//! This module provides automatic parameter optimization for the SAT solver.
//! It can tune various solver parameters based on performance feedback.
//!
//! Features:
//! - Parameter space exploration
//! - Performance-based parameter adjustment
//! - Adaptive parameter tuning during solving
//! - Configuration scoring and comparison

use std::collections::HashMap;

/// A tunable parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Current value
    pub value: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Step size for adjustments
    pub step: f64,
}

impl Parameter {
    /// Create a new parameter
    #[must_use]
    pub fn new(name: String, value: f64, min: f64, max: f64, step: f64) -> Self {
        Self {
            name,
            value: value.clamp(min, max),
            min,
            max,
            step,
        }
    }

    /// Increase the parameter value
    pub fn increase(&mut self) {
        self.value = (self.value + self.step).min(self.max);
    }

    /// Decrease the parameter value
    pub fn decrease(&mut self) {
        self.value = (self.value - self.step).max(self.min);
    }

    /// Set to a specific value
    pub fn set(&mut self, value: f64) {
        self.value = value.clamp(self.min, self.max);
    }

    /// Get the current value
    #[must_use]
    pub fn get(&self) -> f64 {
        self.value
    }
}

/// Performance metrics for tuning evaluation
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Number of conflicts
    pub conflicts: u64,
    /// Number of decisions
    pub decisions: u64,
    /// Number of propagations
    pub propagations: u64,
    /// Solving time in milliseconds
    pub time_ms: u64,
    /// Whether the instance was solved
    pub solved: bool,
    /// Result (SAT/UNSAT)
    pub satisfiable: Option<bool>,
}

impl PerformanceMetrics {
    /// Compute a score for this configuration (lower is better)
    ///
    /// The score is primarily based on solving time, with bonuses for efficiency
    #[must_use]
    pub fn score(&self) -> f64 {
        if !self.solved {
            return f64::MAX;
        }

        // Base score is time
        let mut score = self.time_ms as f64;

        // Penalty for excessive conflicts
        let conflict_rate = self.conflicts as f64 / self.time_ms.max(1) as f64;
        if conflict_rate > 100.0 {
            score *= 1.0 + (conflict_rate - 100.0) / 100.0;
        }

        // Bonus for high propagation rate
        let prop_rate = self.propagations as f64 / self.time_ms.max(1) as f64;
        if prop_rate > 1000.0 {
            score *= 0.95;
        }

        score
    }
}

/// Configuration with parameters and performance metrics
#[derive(Debug, Clone)]
pub struct Configuration {
    /// Parameter values
    pub parameters: HashMap<String, f64>,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
}

impl Configuration {
    /// Create a new configuration
    #[must_use]
    pub fn new(parameters: HashMap<String, f64>) -> Self {
        Self {
            parameters,
            metrics: None,
        }
    }

    /// Get the score for this configuration
    #[must_use]
    pub fn score(&self) -> f64 {
        self.metrics.as_ref().map_or(f64::MAX, |m| m.score())
    }
}

/// Tuning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuningStrategy {
    /// Grid search - exhaustive search over parameter space
    GridSearch,
    /// Random search - random sampling of parameter space
    RandomSearch,
    /// Hill climbing - local search starting from current parameters
    HillClimbing,
    /// Adaptive tuning - adjust parameters based on runtime feedback
    Adaptive,
}

/// Statistics for auto-tuning
#[derive(Debug, Default, Clone)]
pub struct AutotuningStats {
    /// Number of configurations tried
    pub configurations_tried: u64,
    /// Best score found
    pub best_score: f64,
    /// Number of improvements
    pub improvements: u64,
}

/// Auto-tuning manager
pub struct Autotuner {
    /// Tunable parameters
    parameters: HashMap<String, Parameter>,
    /// Tuning strategy
    strategy: TuningStrategy,
    /// Configuration history
    history: Vec<Configuration>,
    /// Best configuration found
    best_config: Option<Configuration>,
    /// Statistics
    stats: AutotuningStats,
}

impl Autotuner {
    /// Create a new auto-tuner
    #[must_use]
    pub fn new(strategy: TuningStrategy) -> Self {
        Self {
            parameters: HashMap::new(),
            strategy,
            history: Vec::new(),
            best_config: None,
            stats: AutotuningStats::default(),
        }
    }

    /// Add a tunable parameter
    pub fn add_parameter(&mut self, param: Parameter) {
        self.parameters.insert(param.name.clone(), param);
    }

    /// Get current parameter values
    #[must_use]
    pub fn get_parameters(&self) -> HashMap<String, f64> {
        self.parameters
            .iter()
            .map(|(name, param)| (name.clone(), param.value))
            .collect()
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &AutotuningStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = AutotuningStats::default();
        self.stats.best_score = f64::MAX;
    }

    /// Record performance for current configuration
    pub fn record_performance(&mut self, metrics: PerformanceMetrics) {
        let config = Configuration {
            parameters: self.get_parameters(),
            metrics: Some(metrics),
        };

        let score = config.score();
        self.stats.configurations_tried += 1;

        // Update best configuration
        if score < self.stats.best_score || self.best_config.is_none() {
            self.stats.best_score = score;
            self.stats.improvements += 1;
            self.best_config = Some(config.clone());
        }

        self.history.push(config);

        // Apply strategy-specific parameter adjustments
        self.adjust_parameters();
    }

    /// Get the best configuration found
    #[must_use]
    pub fn best_configuration(&self) -> Option<&Configuration> {
        self.best_config.as_ref()
    }

    /// Apply the best configuration
    pub fn apply_best(&mut self) {
        if let Some(best) = &self.best_config {
            for (name, &value) in &best.parameters {
                if let Some(param) = self.parameters.get_mut(name) {
                    param.set(value);
                }
            }
        }
    }

    /// Generate next configuration to try (for grid/random search)
    #[must_use]
    pub fn next_configuration(&mut self) -> HashMap<String, f64> {
        match self.strategy {
            TuningStrategy::GridSearch => self.grid_search_next(),
            TuningStrategy::RandomSearch => self.random_search_next(),
            _ => self.get_parameters(),
        }
    }

    /// Adjust parameters based on strategy
    fn adjust_parameters(&mut self) {
        match self.strategy {
            TuningStrategy::HillClimbing => self.hill_climbing_adjust(),
            TuningStrategy::Adaptive => self.adaptive_adjust(),
            _ => {}
        }
    }

    /// Hill climbing parameter adjustment
    fn hill_climbing_adjust(&mut self) {
        if self.history.len() < 2 {
            return;
        }

        let current = &self.history[self.history.len() - 1];
        let previous = &self.history[self.history.len() - 2];

        let current_score = current.score();
        let previous_score = previous.score();

        // If we improved, continue in the same direction
        // If we got worse, try the opposite direction
        for (name, param) in &mut self.parameters {
            let current_val = current.parameters.get(name).copied().unwrap_or(param.value);
            let previous_val = previous
                .parameters
                .get(name)
                .copied()
                .unwrap_or(param.value);

            if current_score < previous_score {
                // Improved - continue in same direction
                if current_val > previous_val {
                    param.increase();
                } else if current_val < previous_val {
                    param.decrease();
                }
            } else {
                // Got worse - try opposite direction
                if current_val > previous_val {
                    param.decrease();
                } else if current_val < previous_val {
                    param.increase();
                }
            }
        }
    }

    /// Adaptive parameter adjustment based on runtime feedback
    fn adaptive_adjust(&mut self) {
        if self.history.is_empty() {
            return;
        }

        let recent_window = 5.min(self.history.len());
        let recent_configs: Vec<_> = self.history.iter().rev().take(recent_window).collect();

        // Calculate average metrics
        let mut total_conflicts = 0;
        let mut total_decisions = 0;
        let mut count = 0;

        for config in &recent_configs {
            if let Some(ref metrics) = config.metrics {
                total_conflicts += metrics.conflicts;
                total_decisions += metrics.decisions;
                count += 1;
            }
        }

        if count == 0 {
            return;
        }

        let avg_conflicts = total_conflicts / count;
        let avg_decisions = total_decisions / count;

        // Adaptive adjustments based on heuristics
        // High conflict rate -> increase restart aggressiveness
        // High decision rate -> adjust branching heuristic parameters

        // Example: adjust restart_factor based on conflict rate
        if let Some(restart_param) = self.parameters.get_mut("restart_factor") {
            if avg_conflicts > 10000 {
                restart_param.increase();
            } else if avg_conflicts < 1000 {
                restart_param.decrease();
            }
        }

        // Example: adjust variable_decay based on decision rate
        if let Some(decay_param) = self.parameters.get_mut("variable_decay") {
            if avg_decisions > 5000 {
                decay_param.increase();
            } else if avg_decisions < 500 {
                decay_param.decrease();
            }
        }
    }

    /// Grid search - systematically try all combinations
    fn grid_search_next(&self) -> HashMap<String, f64> {
        // Simple implementation: just return current parameters
        // A full grid search would enumerate all combinations
        self.get_parameters()
    }

    /// Random search - randomly sample parameter space
    fn random_search_next(&mut self) -> HashMap<String, f64> {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hash, Hasher};

        let mut result = HashMap::new();
        let state = RandomState::new();

        for (name, param) in &self.parameters {
            // Simple pseudo-random value generation using hash
            let mut hasher = state.build_hasher();
            name.hash(&mut hasher);
            self.history.len().hash(&mut hasher);
            let hash = hasher.finish();

            let range = param.max - param.min;
            let random_factor = (hash % 1000) as f64 / 1000.0;
            let value = param.min + range * random_factor;

            result.insert(name.clone(), value);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_creation() {
        let param = Parameter::new("test".to_string(), 0.5, 0.0, 1.0, 0.1);
        assert_eq!(param.name, "test");
        assert_eq!(param.value, 0.5);
    }

    #[test]
    fn test_parameter_increase_decrease() {
        let mut param = Parameter::new("test".to_string(), 0.5, 0.0, 1.0, 0.1);

        param.increase();
        assert_eq!(param.value, 0.6);

        param.decrease();
        assert_eq!(param.value, 0.5);
    }

    #[test]
    fn test_parameter_bounds() {
        let mut param = Parameter::new("test".to_string(), 0.9, 0.0, 1.0, 0.2);

        param.increase();
        assert_eq!(param.value, 1.0); // Clamped to max

        param.decrease();
        param.decrease();
        param.decrease();
        param.decrease();
        param.decrease();
        param.decrease();
        assert_eq!(param.value, 0.0); // Clamped to min
    }

    #[test]
    fn test_autotuner_creation() {
        let tuner = Autotuner::new(TuningStrategy::HillClimbing);
        assert_eq!(tuner.stats().configurations_tried, 0);
    }

    #[test]
    fn test_add_parameter() {
        let mut tuner = Autotuner::new(TuningStrategy::HillClimbing);
        let param = Parameter::new("test".to_string(), 0.5, 0.0, 1.0, 0.1);

        tuner.add_parameter(param);
        let params = tuner.get_parameters();

        assert_eq!(params.get("test"), Some(&0.5));
    }

    #[test]
    fn test_record_performance() {
        let mut tuner = Autotuner::new(TuningStrategy::HillClimbing);
        let param = Parameter::new("test".to_string(), 0.5, 0.0, 1.0, 0.1);
        tuner.add_parameter(param);

        let metrics = PerformanceMetrics {
            conflicts: 100,
            decisions: 200,
            propagations: 1000,
            time_ms: 50,
            solved: true,
            satisfiable: Some(true),
        };

        tuner.record_performance(metrics);
        assert_eq!(tuner.stats().configurations_tried, 1);
    }

    #[test]
    fn test_best_configuration() {
        let mut tuner = Autotuner::new(TuningStrategy::HillClimbing);
        let param = Parameter::new("test".to_string(), 0.5, 0.0, 1.0, 0.1);
        tuner.add_parameter(param);

        let metrics1 = PerformanceMetrics {
            conflicts: 100,
            decisions: 200,
            propagations: 1000,
            time_ms: 100,
            solved: true,
            satisfiable: Some(true),
        };

        tuner.record_performance(metrics1);

        let metrics2 = PerformanceMetrics {
            conflicts: 50,
            decisions: 100,
            propagations: 500,
            time_ms: 50,
            solved: true,
            satisfiable: Some(true),
        };

        tuner.record_performance(metrics2);

        assert_eq!(tuner.stats().improvements, 2);
        assert!(tuner.best_configuration().is_some());
    }

    #[test]
    fn test_performance_score() {
        let metrics = PerformanceMetrics {
            conflicts: 100,
            decisions: 200,
            propagations: 1000,
            time_ms: 50,
            solved: true,
            satisfiable: Some(true),
        };

        let score = metrics.score();
        assert!(score > 0.0);
        assert!(score < f64::MAX);
    }

    #[test]
    fn test_unsolved_score() {
        let metrics = PerformanceMetrics {
            conflicts: 100,
            decisions: 200,
            propagations: 1000,
            time_ms: 50,
            solved: false,
            satisfiable: None,
        };

        let score = metrics.score();
        assert_eq!(score, f64::MAX);
    }

    #[test]
    fn test_apply_best() {
        let mut tuner = Autotuner::new(TuningStrategy::HillClimbing);
        let param = Parameter::new("test".to_string(), 0.5, 0.0, 1.0, 0.1);
        tuner.add_parameter(param);

        let metrics = PerformanceMetrics {
            conflicts: 100,
            decisions: 200,
            propagations: 1000,
            time_ms: 50,
            solved: true,
            satisfiable: Some(true),
        };

        tuner.record_performance(metrics);

        // Manually change parameter
        if let Some(param) = tuner.parameters.get_mut("test") {
            param.set(0.8);
        }

        // Apply best should revert to best configuration
        tuner.apply_best();
        let params = tuner.get_parameters();
        assert_eq!(params.get("test"), Some(&0.5));
    }

    #[test]
    fn test_different_strategies() {
        let strategies = vec![
            TuningStrategy::GridSearch,
            TuningStrategy::RandomSearch,
            TuningStrategy::HillClimbing,
            TuningStrategy::Adaptive,
        ];

        for strategy in strategies {
            let tuner = Autotuner::new(strategy);
            assert_eq!(tuner.strategy, strategy);
        }
    }
}
