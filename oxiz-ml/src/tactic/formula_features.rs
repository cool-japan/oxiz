//! Formula Feature Extraction
#![allow(clippy::too_many_arguments)] // ML feature extraction
//!
//! Extract features from SMT formulas for tactic selection.

use crate::TACTIC_FEATURE_SIZE;

/// Formula features for ML prediction
#[derive(Debug, Clone)]
pub struct FormulaFeatures {
    /// Feature vector
    pub features: Vec<f64>,
}

impl FormulaFeatures {
    /// Create from feature vector
    pub fn from_vec(features: Vec<f64>) -> Self {
        Self { features }
    }

    /// Extract features from formula statistics
    pub fn extract(
        num_variables: usize,
        num_clauses: usize,
        avg_clause_size: f64,
        num_quantifiers: usize,
        num_boolean_vars: usize,
        num_arithmetic_vars: usize,
        num_theory_atoms: usize,
        max_nesting_depth: usize,
        has_arrays: bool,
        has_bitvectors: bool,
        has_uninterpreted_functions: bool,
    ) -> Self {
        let mut features = Vec::with_capacity(TACTIC_FEATURE_SIZE);

        // 1. Number of variables (log scale)
        features.push((1.0 + num_variables as f64).ln() / 20.0);

        // 2. Number of clauses (log scale)
        features.push((1.0 + num_clauses as f64).ln() / 20.0);

        // 3. Average clause size
        features.push(avg_clause_size / 50.0);

        // 4. Clause/variable ratio
        let clause_var_ratio = if num_variables > 0 {
            num_clauses as f64 / num_variables as f64
        } else {
            1.0
        };
        features.push(clause_var_ratio.min(10.0) / 10.0);

        // 5. Number of quantifiers
        features.push((1.0 + num_quantifiers as f64).ln() / 10.0);

        // 6. Boolean variable ratio
        let bool_ratio = if num_variables > 0 {
            num_boolean_vars as f64 / num_variables as f64
        } else {
            1.0
        };
        features.push(bool_ratio);

        // 7. Arithmetic variable ratio
        let arith_ratio = if num_variables > 0 {
            num_arithmetic_vars as f64 / num_variables as f64
        } else {
            0.0
        };
        features.push(arith_ratio);

        // 8. Theory atom density
        let theory_density = if num_clauses > 0 {
            num_theory_atoms as f64 / num_clauses as f64
        } else {
            0.0
        };
        features.push(theory_density);

        // 9. Nesting depth (normalized)
        features.push(max_nesting_depth as f64 / 50.0);

        // 10. Has arrays (binary)
        features.push(if has_arrays { 1.0 } else { 0.0 });

        // 11. Has bitvectors (binary)
        features.push(if has_bitvectors { 1.0 } else { 0.0 });

        // 12. Has uninterpreted functions (binary)
        features.push(if has_uninterpreted_functions {
            1.0
        } else {
            0.0
        });

        // 13. Formula complexity (combined metric)
        let complexity = (num_variables as f64 * num_clauses as f64).sqrt() / 100.0;
        features.push(complexity);

        // 14-20. Reserved for future features
        features.resize(TACTIC_FEATURE_SIZE, 0.0);

        Self { features }
    }
}

impl Default for FormulaFeatures {
    fn default() -> Self {
        Self {
            features: vec![0.0; TACTIC_FEATURE_SIZE],
        }
    }
}

/// Formula feature extractor (stateful)
pub struct FeatureExtractor {
    /// Cached features
    cached_features: Option<FormulaFeatures>,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new() -> Self {
        Self {
            cached_features: None,
        }
    }

    /// Extract features from formula
    pub fn extract_from_formula(&mut self, _formula: &str) -> FormulaFeatures {
        // Placeholder: in real implementation, parse formula and extract stats
        // For now, return default features
        FormulaFeatures::default()
    }

    /// Invalidate cache
    pub fn invalidate_cache(&mut self) {
        self.cached_features = None;
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_features_extract() {
        let features = FormulaFeatures::extract(
            100,   // num_variables
            200,   // num_clauses
            5.0,   // avg_clause_size
            10,    // num_quantifiers
            80,    // num_boolean_vars
            20,    // num_arithmetic_vars
            50,    // num_theory_atoms
            10,    // max_nesting_depth
            true,  // has_arrays
            false, // has_bitvectors
            true,  // has_uninterpreted_functions
        );

        assert_eq!(features.features.len(), TACTIC_FEATURE_SIZE);
        assert!(features.features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_formula_features_default() {
        let features = FormulaFeatures::default();
        assert_eq!(features.features.len(), TACTIC_FEATURE_SIZE);
    }

    #[test]
    fn test_feature_extractor() {
        let mut extractor = FeatureExtractor::new();
        let _features = extractor.extract_from_formula("(and p q)");
        // Currently returns default, but infrastructure is in place
    }
}
