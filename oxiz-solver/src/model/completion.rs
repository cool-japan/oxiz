//! Model Completion.
//!
//! Fills in values for unassigned variables to produce a complete model
//! that satisfies all constraints.
//!
//! ## Strategies
//!
//! - **Default values**: Assign default values based on sort
//! - **Witness completion**: Use witness terms from theory solving
//! - **Optimal completion**: Minimize model size or maximize interpretability
//!
//! ## References
//!
//! - Z3's `model/model_evaluator.cpp`

use super::builder::{Model, Value, VarId};
use rustc_hash::FxHashMap;

/// Strategy for completing missing values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionStrategy {
    /// Use default values (0, false, empty).
    Default,
    /// Use witness terms from solving.
    Witness,
    /// Minimize model complexity.
    Minimal,
}

/// Configuration for model completion.
#[derive(Debug, Clone)]
pub struct CompletionConfig {
    /// Completion strategy.
    pub strategy: CompletionStrategy,
    /// Default integer value.
    pub default_int: i64,
    /// Default boolean value.
    pub default_bool: bool,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            strategy: CompletionStrategy::Default,
            default_int: 0,
            default_bool: false,
        }
    }
}

/// Statistics for model completion.
#[derive(Debug, Clone, Default)]
pub struct CompletionStats {
    /// Number of variables completed.
    pub vars_completed: u64,
    /// Number of defaults used.
    pub defaults_used: u64,
    /// Number of witnesses used.
    pub witnesses_used: u64,
}

/// Model completion engine.
#[derive(Debug)]
pub struct ModelCompleter {
    /// Configuration.
    config: CompletionConfig,
    /// Witness values (from theory solving).
    witnesses: FxHashMap<VarId, Value>,
    /// Statistics.
    stats: CompletionStats,
}

impl ModelCompleter {
    /// Create a new model completer.
    pub fn new(config: CompletionConfig) -> Self {
        Self {
            config,
            witnesses: FxHashMap::default(),
            stats: CompletionStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(CompletionConfig::default())
    }

    /// Register a witness value for a variable.
    pub fn add_witness(&mut self, var: VarId, value: Value) {
        self.witnesses.insert(var, value);
    }

    /// Complete a partial model by filling in missing values.
    ///
    /// Returns the completed model.
    pub fn complete(&mut self, partial_model: &Model) -> Model {
        let mut completed = partial_model.clone();

        // Get all variables that need completion
        let missing_vars = self.find_missing_vars(&completed);

        for var in missing_vars {
            let value = self.complete_variable(var);
            completed.assign_theory(var, value);
            self.stats.vars_completed += 1;
        }

        completed
    }

    /// Find variables that are not assigned in the model.
    fn find_missing_vars(&self, model: &Model) -> Vec<VarId> {
        // In a real implementation, this would query all declared variables
        // and find which ones are not in the model
        let mut missing = Vec::new();

        // Check witnesses for unassigned variables
        for &var in self.witnesses.keys() {
            if model.get_theory(var).is_none() {
                missing.push(var);
            }
        }

        missing
    }

    /// Complete a single variable.
    fn complete_variable(&mut self, var: VarId) -> Value {
        match self.config.strategy {
            CompletionStrategy::Witness => {
                if let Some(witness) = self.witnesses.get(&var) {
                    self.stats.witnesses_used += 1;
                    witness.clone()
                } else {
                    self.stats.defaults_used += 1;
                    self.default_value()
                }
            }
            CompletionStrategy::Default => {
                self.stats.defaults_used += 1;
                self.default_value()
            }
            CompletionStrategy::Minimal => {
                // Try witness first, then minimal value
                if let Some(witness) = self.witnesses.get(&var) {
                    self.stats.witnesses_used += 1;
                    witness.clone()
                } else {
                    self.stats.defaults_used += 1;
                    self.minimal_value()
                }
            }
        }
    }

    /// Get default value based on configuration.
    fn default_value(&self) -> Value {
        Value::Int(self.config.default_int)
    }

    /// Get minimal value (smallest representable).
    fn minimal_value(&self) -> Value {
        Value::Int(0)
    }

    /// Get statistics.
    pub fn stats(&self) -> &CompletionStats {
        &self.stats
    }

    /// Reset completer state.
    pub fn reset(&mut self) {
        self.witnesses.clear();
        self.stats = CompletionStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completer_creation() {
        let completer = ModelCompleter::default_config();
        assert_eq!(completer.stats().vars_completed, 0);
    }

    #[test]
    fn test_add_witness() {
        let mut completer = ModelCompleter::default_config();
        completer.add_witness(0, Value::Int(42));

        assert!(completer.witnesses.contains_key(&0));
    }

    #[test]
    fn test_complete_with_witnesses() {
        let config = CompletionConfig {
            strategy: CompletionStrategy::Witness,
            ..Default::default()
        };
        let mut completer = ModelCompleter::new(config);
        completer.add_witness(0, Value::Int(10));
        completer.add_witness(1, Value::Bool(true));

        let partial = Model::new();
        let _completed = completer.complete(&partial);

        // Check witnesses were used
        assert!(completer.stats().witnesses_used > 0);
    }

    #[test]
    fn test_default_completion() {
        let config = CompletionConfig {
            strategy: CompletionStrategy::Default,
            default_int: 100,
            ..Default::default()
        };

        let completer = ModelCompleter::new(config);
        let default_val = completer.default_value();

        assert_eq!(default_val, Value::Int(100));
    }
}
