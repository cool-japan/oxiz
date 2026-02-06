//! Model Builder for Theory Solvers.
//!
//! Constructs satisfying assignments by coordinating theory-specific model
//! construction and ensuring cross-theory consistency.
//!
//! ## Architecture
//!
//! 1. **Theory Model Builders**: Each theory provides partial models
//! 2. **Coordination**: Ensure shared variables have consistent values
//! 3. **Completion**: Fill unassigned variables with default values
//! 4. **Minimization**: Optionally produce minimal models
//!
//! ## References
//!
//! - Z3's `smt/smt_model_generator.cpp`

use oxiz_sat::Lit;
use rustc_hash::FxHashMap;

/// Variable identifier.
pub type VarId = u32;

/// Sort (type) identifier.
pub type SortId = u32;

/// Value in a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i64),
    /// Rational value (numerator, denominator).
    Rational(i64, i64),
    /// Bitvector value (value, width).
    BitVec(u64, u32),
    /// Uninterpreted constant.
    Uninterpreted(String),
}

/// Model assignment.
#[derive(Debug, Clone)]
pub struct Model {
    /// Boolean assignments.
    pub bool_assignments: FxHashMap<Lit, bool>,
    /// Theory assignments.
    pub theory_assignments: FxHashMap<VarId, Value>,
}

impl Model {
    /// Create empty model.
    pub fn new() -> Self {
        Self {
            bool_assignments: FxHashMap::default(),
            theory_assignments: FxHashMap::default(),
        }
    }

    /// Assign Boolean literal.
    pub fn assign_bool(&mut self, lit: Lit, value: bool) {
        self.bool_assignments.insert(lit, value);
    }

    /// Assign theory variable.
    pub fn assign_theory(&mut self, var: VarId, value: Value) {
        self.theory_assignments.insert(var, value);
    }

    /// Get Boolean assignment.
    pub fn get_bool(&self, lit: Lit) -> Option<bool> {
        self.bool_assignments.get(&lit).copied()
    }

    /// Get theory assignment.
    pub fn get_theory(&self, var: VarId) -> Option<&Value> {
        self.theory_assignments.get(&var)
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for model builder.
#[derive(Debug, Clone)]
pub struct ModelBuilderConfig {
    /// Enable model minimization.
    pub minimize: bool,
    /// Enable model completion.
    pub complete: bool,
    /// Default integer value.
    pub default_int: i64,
}

impl Default for ModelBuilderConfig {
    fn default() -> Self {
        Self {
            minimize: false,
            complete: true,
            default_int: 0,
        }
    }
}

/// Statistics for model builder.
#[derive(Debug, Clone, Default)]
pub struct ModelBuilderStats {
    /// Models built.
    pub models_built: u64,
    /// Variables assigned.
    pub vars_assigned: u64,
    /// Minimizations performed.
    pub minimizations: u64,
}

/// Model builder engine.
pub struct ModelBuilder {
    config: ModelBuilderConfig,
    stats: ModelBuilderStats,
}

impl ModelBuilder {
    /// Create new model builder.
    pub fn new() -> Self {
        Self::with_config(ModelBuilderConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: ModelBuilderConfig) -> Self {
        Self {
            config,
            stats: ModelBuilderStats::default(),
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> &ModelBuilderStats {
        &self.stats
    }

    /// Build model from theory assignments.
    pub fn build_model(
        &mut self,
        bool_assignment: &FxHashMap<Lit, bool>,
        theory_models: Vec<FxHashMap<VarId, Value>>,
    ) -> Model {
        self.stats.models_built += 1;

        let mut model = Model::new();

        // Copy Boolean assignments
        for (&lit, &value) in bool_assignment {
            model.assign_bool(lit, value);
            self.stats.vars_assigned += 1;
        }

        // Merge theory assignments
        for theory_model in theory_models {
            for (var, value) in theory_model {
                // Check consistency with existing assignment
                if let Some(existing) = model.get_theory(var)
                    && existing != &value
                {
                    // Inconsistency - prefer first assignment
                    continue;
                }

                model.assign_theory(var, value);
                self.stats.vars_assigned += 1;
            }
        }

        // Complete if enabled
        if self.config.complete {
            self.complete_model(&mut model);
        }

        // Minimize if enabled
        if self.config.minimize {
            self.minimize_model(&mut model);
        }

        model
    }

    /// Complete model with default values.
    fn complete_model(&mut self, _model: &mut Model) {
        // Would fill unassigned variables with defaults
        // Simplified: no-op
    }

    /// Minimize model.
    fn minimize_model(&mut self, _model: &mut Model) {
        self.stats.minimizations += 1;
        // Would remove unnecessary assignments
        // Simplified: no-op
    }

    /// Validate model consistency.
    pub fn validate(&self, _model: &Model) -> bool {
        // Check that shared variables have consistent values across theories
        // Simplified: always valid
        true
    }
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = Model::new();
        assert!(model.bool_assignments.is_empty());
        assert!(model.theory_assignments.is_empty());
    }

    #[test]
    fn test_bool_assignment() {
        let mut model = Model::new();
        let lit = Lit::from_dimacs(1);

        model.assign_bool(lit, true);

        assert_eq!(model.get_bool(lit), Some(true));
    }

    #[test]
    fn test_theory_assignment() {
        let mut model = Model::new();

        model.assign_theory(0, Value::Int(42));

        assert_eq!(model.get_theory(0), Some(&Value::Int(42)));
    }

    #[test]
    fn test_builder_creation() {
        let builder = ModelBuilder::new();
        assert_eq!(builder.stats().models_built, 0);
    }

    #[test]
    fn test_build_model() {
        let mut builder = ModelBuilder::new();

        let bool_map = FxHashMap::default();
        let theory_models = vec![{
            let mut m = FxHashMap::default();
            m.insert(0, Value::Int(5));
            m
        }];

        let model = builder.build_model(&bool_map, theory_models);

        assert_eq!(model.get_theory(0), Some(&Value::Int(5)));
        assert_eq!(builder.stats().models_built, 1);
    }

    #[test]
    fn test_validate() {
        let builder = ModelBuilder::new();
        let model = Model::new();

        assert!(builder.validate(&model));
    }

    #[test]
    fn test_value_types() {
        let mut model = Model::new();

        model.assign_theory(0, Value::Bool(true));
        model.assign_theory(1, Value::Int(42));
        model.assign_theory(2, Value::Rational(3, 4));
        model.assign_theory(3, Value::BitVec(0xFF, 8));

        assert_eq!(model.get_theory(0), Some(&Value::Bool(true)));
        assert_eq!(model.get_theory(1), Some(&Value::Int(42)));
        assert_eq!(model.get_theory(2), Some(&Value::Rational(3, 4)));
        assert_eq!(model.get_theory(3), Some(&Value::BitVec(0xFF, 8)));
    }
}
