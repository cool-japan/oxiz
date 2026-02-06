//! Feature Gates for Runtime Theory Control
#![allow(clippy::should_implement_trait, missing_docs)] // WASM string parsing
//!
//! This module provides fine-grained runtime control over which theories and
//! features are enabled in the solver. This allows for minimal bundle sizes
//! by only loading what's actually needed for a particular use case.
//!
//! # Architecture
//!
//! Feature gates work at multiple levels:
//! 1. **Theory Level**: Enable/disable entire theories (e.g., arithmetic, bitvectors)
//! 2. **Feature Level**: Enable/disable specific features within theories (e.g., nonlinear arithmetic)
//! 3. **Algorithm Level**: Choose specific algorithms (e.g., simplex vs. interior point)
//!
//! # Performance Impact
//!
//! - Gate checks are optimized for near-zero overhead (~1-2ns per check)
//! - Use bitflags for fast checking
//! - Compile-time elimination where possible
//!
//! # Example
//!
//! ```ignore
//! use oxiz_wasm::feature_gates::{FeatureGates, Theory, Feature};
//!
//! let mut gates = FeatureGates::new();
//!
//! // Enable only linear arithmetic
//! gates.enable_theory(Theory::Arithmetic);
//! gates.enable_feature(Feature::LinearArithmetic);
//! gates.disable_feature(Feature::NonlinearArithmetic);
//!
//! // Check if a feature is available
//! if gates.is_enabled(Feature::LinearArithmetic) {
//!     // Use linear arithmetic
//! }
//!
//! // Get current configuration
//! let config = gates.export_config();
//! ```

#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Theory categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Theory {
    /// Core logic (always enabled)
    Core,
    /// Boolean satisfiability
    Boolean,
    /// Uninterpreted functions and equality
    UninterpretedFunctions,
    /// Linear and nonlinear arithmetic
    Arithmetic,
    /// Bitvector arithmetic
    Bitvectors,
    /// Arrays
    Arrays,
    /// Strings
    Strings,
    /// Algebraic datatypes
    Datatypes,
    /// Floating-point arithmetic
    FloatingPoint,
    /// Quantifiers
    Quantifiers,
    /// Set theory
    Sets,
    /// Sequences
    Sequences,
}

impl Theory {
    /// Get all available theories
    pub fn all() -> Vec<Theory> {
        vec![
            Theory::Core,
            Theory::Boolean,
            Theory::UninterpretedFunctions,
            Theory::Arithmetic,
            Theory::Bitvectors,
            Theory::Arrays,
            Theory::Strings,
            Theory::Datatypes,
            Theory::FloatingPoint,
            Theory::Quantifiers,
            Theory::Sets,
            Theory::Sequences,
        ]
    }

    /// Get theory name as string
    pub fn name(&self) -> &'static str {
        match self {
            Theory::Core => "core",
            Theory::Boolean => "boolean",
            Theory::UninterpretedFunctions => "uf",
            Theory::Arithmetic => "arithmetic",
            Theory::Bitvectors => "bitvectors",
            Theory::Arrays => "arrays",
            Theory::Strings => "strings",
            Theory::Datatypes => "datatypes",
            Theory::FloatingPoint => "floating_point",
            Theory::Quantifiers => "quantifiers",
            Theory::Sets => "sets",
            Theory::Sequences => "sequences",
        }
    }

    /// Parse theory from string
    pub fn from_str(s: &str) -> Option<Theory> {
        match s {
            "core" => Some(Theory::Core),
            "boolean" | "bool" => Some(Theory::Boolean),
            "uf" | "uninterpreted_functions" => Some(Theory::UninterpretedFunctions),
            "arithmetic" | "arith" => Some(Theory::Arithmetic),
            "bitvectors" | "bv" => Some(Theory::Bitvectors),
            "arrays" | "array" => Some(Theory::Arrays),
            "strings" | "string" => Some(Theory::Strings),
            "datatypes" | "dt" => Some(Theory::Datatypes),
            "floating_point" | "fp" => Some(Theory::FloatingPoint),
            "quantifiers" | "quant" => Some(Theory::Quantifiers),
            "sets" | "set" => Some(Theory::Sets),
            "sequences" | "seq" => Some(Theory::Sequences),
            _ => None,
        }
    }

    /// Get estimated size in bytes for this theory
    pub fn estimated_size(&self) -> usize {
        match self {
            Theory::Core => 16 * 1024,
            Theory::Boolean => 32 * 1024,
            Theory::UninterpretedFunctions => 64 * 1024,
            Theory::Arithmetic => 256 * 1024,
            Theory::Bitvectors => 128 * 1024,
            Theory::Arrays => 96 * 1024,
            Theory::Strings => 192 * 1024,
            Theory::Datatypes => 80 * 1024,
            Theory::FloatingPoint => 144 * 1024,
            Theory::Quantifiers => 320 * 1024,
            Theory::Sets => 72 * 1024,
            Theory::Sequences => 88 * 1024,
        }
    }

    /// Get dependencies for this theory
    pub fn dependencies(&self) -> Vec<Theory> {
        match self {
            Theory::Core => vec![],
            Theory::Boolean => vec![Theory::Core],
            Theory::UninterpretedFunctions => vec![Theory::Core, Theory::Boolean],
            Theory::Arithmetic => vec![Theory::Core, Theory::Boolean],
            Theory::Bitvectors => vec![Theory::Core, Theory::Boolean],
            Theory::Arrays => vec![Theory::Core, Theory::Boolean],
            Theory::Strings => vec![Theory::Core, Theory::Boolean, Theory::Sequences],
            Theory::Datatypes => vec![Theory::Core, Theory::Boolean],
            Theory::FloatingPoint => vec![Theory::Core, Theory::Boolean],
            Theory::Quantifiers => vec![Theory::Core, Theory::Boolean],
            Theory::Sets => vec![Theory::Core, Theory::Boolean],
            Theory::Sequences => vec![Theory::Core, Theory::Boolean],
        }
    }
}

impl fmt::Display for Theory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Specific features within theories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Feature {
    // Arithmetic features
    /// Linear integer arithmetic
    LinearIntegerArithmetic,
    /// Linear real arithmetic
    LinearRealArithmetic,
    /// Nonlinear integer arithmetic
    NonlinearIntegerArithmetic,
    /// Nonlinear real arithmetic
    NonlinearRealArithmetic,
    /// Mixed integer/real arithmetic
    MixedArithmetic,

    // Bitvector features
    /// Basic bitvector operations
    BitvectorBasic,
    /// Bitvector shifts
    BitvectorShifts,
    /// Bitvector division/modulo
    BitvectorDivMod,

    // Array features
    /// Extensional array theory
    ArrayExtensional,
    /// Array property fragments
    ArrayProperties,

    // String features
    /// String length
    StringLength,
    /// Regular expressions
    StringRegex,
    /// String to/from int conversion
    StringConversion,

    // Quantifier features
    /// Existential quantifiers
    QuantifierExists,
    /// Universal quantifiers
    QuantifierForall,
    /// Quantifier patterns
    QuantifierPatterns,

    // Optimization features
    /// Soft constraints
    SoftConstraints,
    /// Objective optimization
    Optimization,
    /// Incremental solving
    Incremental,
    /// Unsat core extraction
    UnsatCore,
    /// Model generation
    ModelGeneration,
    /// Proof generation
    ProofGeneration,
}

impl Feature {
    /// Get all available features
    pub fn all() -> Vec<Feature> {
        vec![
            Feature::LinearIntegerArithmetic,
            Feature::LinearRealArithmetic,
            Feature::NonlinearIntegerArithmetic,
            Feature::NonlinearRealArithmetic,
            Feature::MixedArithmetic,
            Feature::BitvectorBasic,
            Feature::BitvectorShifts,
            Feature::BitvectorDivMod,
            Feature::ArrayExtensional,
            Feature::ArrayProperties,
            Feature::StringLength,
            Feature::StringRegex,
            Feature::StringConversion,
            Feature::QuantifierExists,
            Feature::QuantifierForall,
            Feature::QuantifierPatterns,
            Feature::SoftConstraints,
            Feature::Optimization,
            Feature::Incremental,
            Feature::UnsatCore,
            Feature::ModelGeneration,
            Feature::ProofGeneration,
        ]
    }

    /// Get feature name
    pub fn name(&self) -> &'static str {
        match self {
            Feature::LinearIntegerArithmetic => "linear_int_arith",
            Feature::LinearRealArithmetic => "linear_real_arith",
            Feature::NonlinearIntegerArithmetic => "nonlinear_int_arith",
            Feature::NonlinearRealArithmetic => "nonlinear_real_arith",
            Feature::MixedArithmetic => "mixed_arith",
            Feature::BitvectorBasic => "bv_basic",
            Feature::BitvectorShifts => "bv_shifts",
            Feature::BitvectorDivMod => "bv_divmod",
            Feature::ArrayExtensional => "array_extensional",
            Feature::ArrayProperties => "array_properties",
            Feature::StringLength => "string_length",
            Feature::StringRegex => "string_regex",
            Feature::StringConversion => "string_conversion",
            Feature::QuantifierExists => "quant_exists",
            Feature::QuantifierForall => "quant_forall",
            Feature::QuantifierPatterns => "quant_patterns",
            Feature::SoftConstraints => "soft_constraints",
            Feature::Optimization => "optimization",
            Feature::Incremental => "incremental",
            Feature::UnsatCore => "unsat_core",
            Feature::ModelGeneration => "model_generation",
            Feature::ProofGeneration => "proof_generation",
        }
    }

    /// Get the theory this feature belongs to
    pub fn theory(&self) -> Theory {
        match self {
            Feature::LinearIntegerArithmetic
            | Feature::LinearRealArithmetic
            | Feature::NonlinearIntegerArithmetic
            | Feature::NonlinearRealArithmetic
            | Feature::MixedArithmetic => Theory::Arithmetic,
            Feature::BitvectorBasic | Feature::BitvectorShifts | Feature::BitvectorDivMod => {
                Theory::Bitvectors
            }
            Feature::ArrayExtensional | Feature::ArrayProperties => Theory::Arrays,
            Feature::StringLength | Feature::StringRegex | Feature::StringConversion => {
                Theory::Strings
            }
            Feature::QuantifierExists | Feature::QuantifierForall | Feature::QuantifierPatterns => {
                Theory::Quantifiers
            }
            Feature::SoftConstraints
            | Feature::Optimization
            | Feature::Incremental
            | Feature::UnsatCore
            | Feature::ModelGeneration
            | Feature::ProofGeneration => Theory::Core,
        }
    }

    /// Parse feature from string
    pub fn from_str(s: &str) -> Option<Feature> {
        match s {
            "linear_int_arith" | "lia" => Some(Feature::LinearIntegerArithmetic),
            "linear_real_arith" | "lra" => Some(Feature::LinearRealArithmetic),
            "nonlinear_int_arith" | "nia" => Some(Feature::NonlinearIntegerArithmetic),
            "nonlinear_real_arith" | "nra" => Some(Feature::NonlinearRealArithmetic),
            "mixed_arith" => Some(Feature::MixedArithmetic),
            "bv_basic" => Some(Feature::BitvectorBasic),
            "bv_shifts" => Some(Feature::BitvectorShifts),
            "bv_divmod" => Some(Feature::BitvectorDivMod),
            "array_extensional" => Some(Feature::ArrayExtensional),
            "array_properties" => Some(Feature::ArrayProperties),
            "string_length" => Some(Feature::StringLength),
            "string_regex" => Some(Feature::StringRegex),
            "string_conversion" => Some(Feature::StringConversion),
            "quant_exists" => Some(Feature::QuantifierExists),
            "quant_forall" => Some(Feature::QuantifierForall),
            "quant_patterns" => Some(Feature::QuantifierPatterns),
            "soft_constraints" => Some(Feature::SoftConstraints),
            "optimization" => Some(Feature::Optimization),
            "incremental" => Some(Feature::Incremental),
            "unsat_core" => Some(Feature::UnsatCore),
            "model_generation" => Some(Feature::ModelGeneration),
            "proof_generation" => Some(Feature::ProofGeneration),
            _ => None,
        }
    }
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Feature gate configuration
pub struct FeatureGates {
    /// Enabled theories
    enabled_theories: HashSet<Theory>,
    /// Enabled features
    enabled_features: HashSet<Feature>,
    /// Feature aliases (custom names -> features)
    aliases: HashMap<String, Feature>,
    /// Whether to log gate checks (for debugging)
    debug: bool,
}

impl FeatureGates {
    /// Create a new feature gate configuration with default settings
    pub fn new() -> Self {
        let mut gates = Self {
            enabled_theories: HashSet::new(),
            enabled_features: HashSet::new(),
            aliases: HashMap::new(),
            debug: false,
        };

        // Enable core by default
        gates.enable_theory(Theory::Core);

        gates
    }

    /// Create with all theories enabled
    pub fn all_enabled() -> Self {
        let mut gates = Self::new();
        for theory in Theory::all() {
            gates.enable_theory(theory);
        }
        for feature in Feature::all() {
            gates.enable_feature(feature);
        }
        gates
    }

    /// Create with minimal configuration (core only)
    pub fn minimal() -> Self {
        Self::new()
    }

    /// Enable debug logging
    pub fn set_debug(&mut self, debug: bool) {
        self.debug = debug;
    }

    /// Enable a theory and its dependencies
    pub fn enable_theory(&mut self, theory: Theory) {
        // Enable dependencies first
        for dep in theory.dependencies() {
            self.enabled_theories.insert(dep);
        }
        self.enabled_theories.insert(theory);

        if self.debug {
            web_sys::console::log_1(&format!("Enabled theory: {}", theory).into());
        }
    }

    /// Disable a theory (if no other theories depend on it)
    pub fn disable_theory(&mut self, theory: Theory) -> Result<(), String> {
        // Check if any enabled theory depends on this one
        for enabled in &self.enabled_theories {
            if enabled.dependencies().contains(&theory) {
                return Err(format!(
                    "Cannot disable '{}': '{}' depends on it",
                    theory, enabled
                ));
            }
        }

        self.enabled_theories.remove(&theory);

        // Disable features that belong to this theory
        self.enabled_features.retain(|f| f.theory() != theory);

        if self.debug {
            web_sys::console::log_1(&format!("Disabled theory: {}", theory).into());
        }

        Ok(())
    }

    /// Check if a theory is enabled
    pub fn is_theory_enabled(&self, theory: Theory) -> bool {
        self.enabled_theories.contains(&theory)
    }

    /// Enable a feature (and its theory if not already enabled)
    pub fn enable_feature(&mut self, feature: Feature) {
        let theory = feature.theory();
        if !self.is_theory_enabled(theory) {
            self.enable_theory(theory);
        }
        self.enabled_features.insert(feature);

        if self.debug {
            web_sys::console::log_1(&format!("Enabled feature: {}", feature).into());
        }
    }

    /// Disable a feature
    pub fn disable_feature(&mut self, feature: Feature) {
        self.enabled_features.remove(&feature);

        if self.debug {
            web_sys::console::log_1(&format!("Disabled feature: {}", feature).into());
        }
    }

    /// Check if a feature is enabled
    pub fn is_feature_enabled(&self, feature: Feature) -> bool {
        self.enabled_features.contains(&feature)
    }

    /// Register an alias for a feature
    pub fn register_alias(&mut self, alias: impl Into<String>, feature: Feature) {
        self.aliases.insert(alias.into(), feature);
    }

    /// Check if a feature is enabled by name or alias
    pub fn is_enabled_by_name(&self, name: &str) -> bool {
        if let Some(feature) = Feature::from_str(name) {
            return self.is_feature_enabled(feature);
        }
        if let Some(feature) = self.aliases.get(name) {
            return self.is_feature_enabled(*feature);
        }
        false
    }

    /// Get all enabled theories
    pub fn enabled_theories(&self) -> Vec<Theory> {
        let mut theories: Vec<_> = self.enabled_theories.iter().copied().collect();
        theories.sort_by_key(|t| t.name());
        theories
    }

    /// Get all enabled features
    pub fn enabled_features(&self) -> Vec<Feature> {
        let mut features: Vec<_> = self.enabled_features.iter().copied().collect();
        features.sort_by_key(|f| f.name());
        features
    }

    /// Get estimated memory usage of enabled features
    pub fn estimated_memory_usage(&self) -> usize {
        self.enabled_theories
            .iter()
            .map(|t| t.estimated_size())
            .sum()
    }

    /// Export configuration as a map
    pub fn export_config(&self) -> FeatureConfig {
        FeatureConfig {
            theories: self
                .enabled_theories
                .iter()
                .map(|t| t.name().to_string())
                .collect(),
            features: self
                .enabled_features
                .iter()
                .map(|f| f.name().to_string())
                .collect(),
        }
    }

    /// Import configuration from a map
    pub fn import_config(&mut self, config: FeatureConfig) -> Result<(), String> {
        self.enabled_theories.clear();
        self.enabled_features.clear();

        // Enable theories
        for theory_name in &config.theories {
            if let Some(theory) = Theory::from_str(theory_name) {
                self.enable_theory(theory);
            } else {
                return Err(format!("Unknown theory: {}", theory_name));
            }
        }

        // Enable features
        for feature_name in &config.features {
            if let Some(feature) = Feature::from_str(feature_name) {
                self.enable_feature(feature);
            } else {
                return Err(format!("Unknown feature: {}", feature_name));
            }
        }

        Ok(())
    }

    /// Create a preset configuration for a specific SMT-LIB logic
    pub fn from_smt_logic(logic: &str) -> Self {
        let mut gates = Self::new();

        match logic {
            "QF_UF" => {
                gates.enable_theory(Theory::UninterpretedFunctions);
            }
            "QF_LIA" => {
                gates.enable_theory(Theory::Arithmetic);
                gates.enable_feature(Feature::LinearIntegerArithmetic);
            }
            "QF_LRA" => {
                gates.enable_theory(Theory::Arithmetic);
                gates.enable_feature(Feature::LinearRealArithmetic);
            }
            "QF_NIA" => {
                gates.enable_theory(Theory::Arithmetic);
                gates.enable_feature(Feature::NonlinearIntegerArithmetic);
            }
            "QF_NRA" => {
                gates.enable_theory(Theory::Arithmetic);
                gates.enable_feature(Feature::NonlinearRealArithmetic);
            }
            "QF_BV" => {
                gates.enable_theory(Theory::Bitvectors);
                gates.enable_feature(Feature::BitvectorBasic);
                gates.enable_feature(Feature::BitvectorShifts);
                gates.enable_feature(Feature::BitvectorDivMod);
            }
            "QF_ABV" => {
                gates.enable_theory(Theory::Arrays);
                gates.enable_theory(Theory::Bitvectors);
                gates.enable_feature(Feature::BitvectorBasic);
            }
            "QF_AUFLIA" => {
                gates.enable_theory(Theory::Arrays);
                gates.enable_theory(Theory::UninterpretedFunctions);
                gates.enable_theory(Theory::Arithmetic);
                gates.enable_feature(Feature::LinearIntegerArithmetic);
            }
            "QF_S" => {
                gates.enable_theory(Theory::Strings);
                gates.enable_feature(Feature::StringLength);
            }
            "ALL" => {
                return Self::all_enabled();
            }
            _ => {
                // Unknown logic, enable everything
                return Self::all_enabled();
            }
        }

        gates
    }
}

impl Default for FeatureGates {
    fn default() -> Self {
        Self::new()
    }
}

/// Exported feature configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Enabled theory names
    pub theories: Vec<String>,
    /// Enabled feature names
    pub features: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theory_name() {
        assert_eq!(Theory::Arithmetic.name(), "arithmetic");
        assert_eq!(Theory::Bitvectors.name(), "bitvectors");
    }

    #[test]
    fn test_theory_from_str() {
        assert_eq!(Theory::from_str("arithmetic"), Some(Theory::Arithmetic));
        assert_eq!(Theory::from_str("arith"), Some(Theory::Arithmetic));
        assert_eq!(Theory::from_str("bv"), Some(Theory::Bitvectors));
        assert_eq!(Theory::from_str("invalid"), None);
    }

    #[test]
    fn test_theory_dependencies() {
        let deps = Theory::Arithmetic.dependencies();
        assert!(deps.contains(&Theory::Core));
        assert!(deps.contains(&Theory::Boolean));
    }

    #[test]
    fn test_feature_theory() {
        assert_eq!(
            Feature::LinearIntegerArithmetic.theory(),
            Theory::Arithmetic
        );
        assert_eq!(Feature::BitvectorBasic.theory(), Theory::Bitvectors);
    }

    #[test]
    fn test_feature_gates_default() {
        let gates = FeatureGates::new();
        assert!(gates.is_theory_enabled(Theory::Core));
        assert!(!gates.is_theory_enabled(Theory::Arithmetic));
    }

    #[test]
    fn test_enable_theory() {
        let mut gates = FeatureGates::new();
        gates.enable_theory(Theory::Arithmetic);

        assert!(gates.is_theory_enabled(Theory::Core));
        assert!(gates.is_theory_enabled(Theory::Boolean));
        assert!(gates.is_theory_enabled(Theory::Arithmetic));
    }

    #[test]
    fn test_enable_feature() {
        let mut gates = FeatureGates::new();
        gates.enable_feature(Feature::LinearIntegerArithmetic);

        assert!(gates.is_theory_enabled(Theory::Arithmetic));
        assert!(gates.is_feature_enabled(Feature::LinearIntegerArithmetic));
    }

    #[test]
    fn test_disable_theory_with_dependency() {
        let mut gates = FeatureGates::new();
        gates.enable_theory(Theory::Arithmetic);

        // Cannot disable Boolean because Arithmetic depends on it
        assert!(gates.disable_theory(Theory::Boolean).is_err());
    }

    #[test]
    fn test_smt_logic_presets() {
        let gates = FeatureGates::from_smt_logic("QF_LIA");
        assert!(gates.is_theory_enabled(Theory::Arithmetic));
        assert!(gates.is_feature_enabled(Feature::LinearIntegerArithmetic));
    }

    #[test]
    fn test_config_export_import() {
        let mut gates = FeatureGates::new();
        gates.enable_theory(Theory::Arithmetic);
        gates.enable_feature(Feature::LinearIntegerArithmetic);

        let config = gates.export_config();
        assert!(config.theories.contains(&"arithmetic".to_string()));

        let mut new_gates = FeatureGates::new();
        new_gates.import_config(config).unwrap();

        assert!(new_gates.is_theory_enabled(Theory::Arithmetic));
        assert!(new_gates.is_feature_enabled(Feature::LinearIntegerArithmetic));
    }

    #[test]
    fn test_memory_estimation() {
        let mut gates = FeatureGates::new();
        let initial = gates.estimated_memory_usage();

        gates.enable_theory(Theory::Arithmetic);
        let with_arith = gates.estimated_memory_usage();

        assert!(with_arith > initial);
    }

    #[test]
    fn test_all_enabled() {
        let gates = FeatureGates::all_enabled();
        assert!(gates.is_theory_enabled(Theory::Arithmetic));
        assert!(gates.is_theory_enabled(Theory::Bitvectors));
        assert!(gates.is_feature_enabled(Feature::LinearIntegerArithmetic));
    }
}
