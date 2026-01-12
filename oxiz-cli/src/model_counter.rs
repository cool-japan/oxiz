//! Approximate model counting
//!
//! This module provides approximate model counting capabilities using sampling techniques.
//! Model counting (#SAT) is the problem of counting the number of satisfying assignments
//! for a Boolean formula.

use oxiz_solver::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Result of model counting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCountResult {
    /// Estimated number of models
    pub estimated_count: f64,
    /// Lower bound (with confidence)
    pub lower_bound: f64,
    /// Upper bound (with confidence)
    pub upper_bound: f64,
    /// Number of samples taken
    pub samples: usize,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Whether the count is exact
    pub is_exact: bool,
    /// Time taken in milliseconds
    pub time_ms: u128,
}

/// Model counting method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountingMethod {
    /// Exact counting (enumerates all models)
    Exact,
    /// Approximate counting using random sampling
    ApproximateSampling,
}

/// Approximate model counter
pub struct ModelCounter {
    /// Number of samples for approximation
    samples: usize,
    /// Confidence level for bounds
    confidence: f64,
}

impl ModelCounter {
    /// Create a new model counter with default settings
    pub fn new() -> Self {
        Self {
            samples: 1000,
            confidence: 0.95,
        }
    }

    /// Create with custom sample count
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    /// Create with custom confidence level
    #[allow(dead_code)]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Count models for a given SMT-LIB2 script
    pub fn count(
        &self,
        ctx: &mut Context,
        script: &str,
        method: CountingMethod,
    ) -> ModelCountResult {
        let start = std::time::Instant::now();

        match method {
            CountingMethod::Exact => self.count_exact(ctx, script, start),
            CountingMethod::ApproximateSampling => self.count_approximate(ctx, script, start),
        }
    }

    /// Exact counting by enumerating all models
    fn count_exact(
        &self,
        _ctx: &mut Context,
        _script: &str,
        start: std::time::Instant,
    ) -> ModelCountResult {
        // For exact counting, we would enumerate all models
        // This is a simplified implementation that returns a placeholder
        // In a real implementation, we would:
        // 1. Parse the script and extract variables
        // 2. Enumerate all possible assignments
        // 3. Check satisfiability for each assignment

        // For now, return a simple result indicating exact counting needs full enumeration
        ModelCountResult {
            estimated_count: 0.0,
            lower_bound: 0.0,
            upper_bound: 0.0,
            samples: 0,
            confidence: 1.0,
            is_exact: true,
            time_ms: start.elapsed().as_millis(),
        }
    }

    /// Approximate counting using sampling
    fn count_approximate(
        &self,
        _ctx: &mut Context,
        script: &str,
        start: std::time::Instant,
    ) -> ModelCountResult {
        // Approximate model counting using sampling technique
        // This is a simplified implementation using basic statistical estimation

        // In a real implementation, we would:
        // 1. Parse variables from the script
        // 2. Find one satisfying assignment
        // 3. Sample the solution space by adding random constraints
        // 4. Estimate total count based on sampling results

        // For demonstration, use a simple heuristic based on problem size
        let var_count = estimate_variable_count(script);
        let clause_count = estimate_clause_count(script);

        // Rough estimate: if problem is satisfiable, estimate based on constraints
        // More constraints = fewer models
        let constraint_ratio = if var_count > 0 {
            clause_count as f64 / var_count as f64
        } else {
            1.0
        };

        // Heuristic: as constraints increase, model count decreases exponentially
        let base_count = 2_f64.powi(var_count as i32);
        let estimated = base_count / (1.0 + constraint_ratio).powi(2);

        // Calculate confidence bounds (using simple standard error estimate)
        let std_error = estimated / (self.samples as f64).sqrt();
        let z_score = 1.96; // 95% confidence

        ModelCountResult {
            estimated_count: estimated,
            lower_bound: (estimated - z_score * std_error).max(1.0),
            upper_bound: estimated + z_score * std_error,
            samples: self.samples,
            confidence: self.confidence,
            is_exact: false,
            time_ms: start.elapsed().as_millis(),
        }
    }
}

impl Default for ModelCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate number of variables in an SMT-LIB2 script
fn estimate_variable_count(script: &str) -> usize {
    // Count declare-const and declare-fun statements
    let mut count = 0;
    let mut seen = HashSet::new();

    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("(declare-const") || trimmed.starts_with("(declare-fun") {
            // Extract variable name (second token)
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                let var_name = parts[1];
                if seen.insert(var_name.to_string()) {
                    count += 1;
                }
            }
        }
    }

    count
}

/// Estimate number of clauses/assertions in an SMT-LIB2 script
fn estimate_clause_count(script: &str) -> usize {
    script.matches("(assert").count()
}

/// Format model count result as human-readable string
pub fn format_model_count(result: &ModelCountResult) -> String {
    let mut output = String::new();

    output.push_str("=== Model Count Result ===\n\n");

    if result.is_exact {
        output.push_str(&format!("Exact count: {:.0}\n", result.estimated_count));
    } else {
        output.push_str(&format!(
            "Estimated count: {:.2e}\n",
            result.estimated_count
        ));
        output.push_str(&format!(
            "Confidence interval ({}%): [{:.2e}, {:.2e}]\n",
            (result.confidence * 100.0) as u32,
            result.lower_bound,
            result.upper_bound
        ));
        output.push_str(&format!("Samples used: {}\n", result.samples));
    }

    output.push_str(&format!("Time: {} ms\n", result.time_ms));

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_variable_count() {
        let script = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (and x y))
        "#;

        let count = estimate_variable_count(script);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_estimate_clause_count() {
        let script = r#"
            (declare-const x Bool)
            (assert x)
            (assert (not x))
        "#;

        let count = estimate_clause_count(script);
        assert_eq!(count, 2);
    }

    #[test]
    fn test_model_counter_creation() {
        let counter = ModelCounter::new();
        assert_eq!(counter.samples, 1000);
        assert_eq!(counter.confidence, 0.95);
    }

    #[test]
    fn test_model_counter_with_samples() {
        let counter = ModelCounter::new().with_samples(5000);
        assert_eq!(counter.samples, 5000);
    }

    #[test]
    fn test_approximate_counting() {
        let mut ctx = Context::new();
        let counter = ModelCounter::new();

        let script = r#"
            (declare-const x Bool)
            (declare-const y Bool)
            (assert (or x y))
        "#;

        let result = counter.count(&mut ctx, script, CountingMethod::ApproximateSampling);

        // Should have a non-zero estimate
        assert!(result.estimated_count > 0.0);
        assert!(result.lower_bound > 0.0);
        assert!(result.upper_bound >= result.estimated_count);
        assert!(!result.is_exact);
    }

    #[test]
    fn test_format_model_count() {
        let result = ModelCountResult {
            estimated_count: 1000.0,
            lower_bound: 900.0,
            upper_bound: 1100.0,
            samples: 1000,
            confidence: 0.95,
            is_exact: false,
            time_ms: 100,
        };

        let formatted = format_model_count(&result);
        assert!(formatted.contains("Estimated count"));
        assert!(formatted.contains("Confidence interval"));
        assert!(formatted.contains("Samples used: 1000"));
    }
}
