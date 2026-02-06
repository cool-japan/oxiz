//! Error Recovery Strategies for SMT Solver Operations.
#![allow(missing_docs, clippy::result_large_err)] // Under development - documentation in progress
//!
//! Provides mechanisms to recover from errors during parsing, solving,
//! and theory reasoning operations.

use crate::error_context::ErrorContext;

/// Strategy for recovering from errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Abort on first error
    FailFast,
    /// Continue and collect all errors
    CollectAll,
    /// Try to recover with heuristics
    Heuristic,
    /// Skip problematic items and continue
    Skip,
}

/// Result of error recovery attempt.
#[derive(Debug, Clone)]
pub enum RecoveryResult<T> {
    /// Recovery succeeded with corrected value
    Recovered(T),
    /// Recovery failed, error persists
    Failed(ErrorContext),
    /// Partial recovery with warnings
    Partial { value: T, warnings: Vec<String> },
}

/// Configuration for error recovery.
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Recovery strategy to use
    pub strategy: RecoveryStrategy,
    /// Maximum number of recovery attempts
    pub max_attempts: usize,
    /// Whether to collect detailed diagnostics
    pub collect_diagnostics: bool,
    /// Whether to suggest fixes
    pub suggest_fixes: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            strategy: RecoveryStrategy::Heuristic,
            max_attempts: 3,
            collect_diagnostics: true,
            suggest_fixes: true,
        }
    }
}

/// Statistics for error recovery operations.
#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    /// Number of recovery attempts
    pub attempts: u64,
    /// Number of successful recoveries
    pub successes: u64,
    /// Number of failed recoveries
    pub failures: u64,
    /// Number of partial recoveries
    pub partial: u64,
}

impl RecoveryStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate success rate.
    pub fn success_rate(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.successes as f64 / self.attempts as f64
        }
    }
}

/// Error recovery manager.
pub struct ErrorRecovery {
    /// Configuration
    config: RecoveryConfig,
    /// Statistics
    stats: RecoveryStats,
    /// Collected errors
    errors: Vec<ErrorContext>,
    /// Collected warnings
    warnings: Vec<String>,
}

impl ErrorRecovery {
    /// Create a new error recovery manager.
    pub fn new(config: RecoveryConfig) -> Self {
        Self {
            config,
            stats: RecoveryStats::default(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(RecoveryConfig::default())
    }

    /// Attempt to recover from a parse error.
    pub fn recover_parse_error(
        &mut self,
        error: ErrorContext,
        input: &str,
        position: usize,
    ) -> RecoveryResult<String> {
        self.stats.attempts += 1;

        match self.config.strategy {
            RecoveryStrategy::FailFast => {
                self.stats.failures += 1;
                self.errors.push(error.clone());
                RecoveryResult::Failed(error)
            }
            RecoveryStrategy::CollectAll => {
                self.stats.failures += 1;
                self.errors.push(error.clone());
                RecoveryResult::Failed(error)
            }
            RecoveryStrategy::Heuristic => {
                // Try heuristic recovery strategies
                if let Some(recovered) = self.try_parse_recovery(input, position) {
                    self.stats.successes += 1;
                    RecoveryResult::Recovered(recovered)
                } else {
                    self.stats.failures += 1;
                    self.errors.push(error.clone());
                    RecoveryResult::Failed(error)
                }
            }
            RecoveryStrategy::Skip => {
                // Skip to next valid token
                if let Some(recovered) = self.skip_to_valid(input, position) {
                    self.stats.partial += 1;
                    self.warnings.push("Skipped invalid input".to_string());
                    RecoveryResult::Partial {
                        value: recovered,
                        warnings: vec!["Skipped problematic section".to_string()],
                    }
                } else {
                    self.stats.failures += 1;
                    self.errors.push(error.clone());
                    RecoveryResult::Failed(error)
                }
            }
        }
    }

    /// Try to recover from parse error using heuristics.
    fn try_parse_recovery(&self, input: &str, position: usize) -> Option<String> {
        // Try to find matching parentheses
        if let Some(recovered) = self.recover_unbalanced_parens(input, position) {
            return Some(recovered);
        }

        // Try to recover missing quotes
        if let Some(recovered) = self.recover_missing_quotes(input, position) {
            return Some(recovered);
        }

        None
    }

    /// Recover from unbalanced parentheses.
    fn recover_unbalanced_parens(&self, input: &str, position: usize) -> Option<String> {
        let before = &input[..position];
        let after = &input[position..];

        let open_count = before.chars().filter(|&c| c == '(').count();
        let close_count = before.chars().filter(|&c| c == ')').count();

        if open_count > close_count {
            // Add missing closing parentheses
            let missing = open_count - close_count;
            let recovered = format!("{}{}{}", before, after, ")".repeat(missing));
            Some(recovered)
        } else if close_count > open_count {
            // Remove extra closing parentheses
            let mut result = before.to_string();
            let mut remaining_closes = close_count - open_count;
            for c in after.chars() {
                if c == ')' && remaining_closes > 0 {
                    remaining_closes -= 1;
                    continue;
                }
                result.push(c);
            }
            Some(result)
        } else {
            None
        }
    }

    /// Recover from missing quotes.
    fn recover_missing_quotes(&self, input: &str, position: usize) -> Option<String> {
        let before = &input[..position];
        let after = &input[position..];

        // Check if we have an odd number of quotes
        let quote_count = before.chars().filter(|&c| c == '"').count();

        if quote_count % 2 == 1 {
            // Add missing closing quote
            // Find the end of the string (whitespace or closing paren)
            let end_pos = after
                .find(|c: char| c.is_whitespace() || c == ')' || c == '(')
                .unwrap_or(after.len());

            let recovered = format!("{}{}\"{}", before, &after[..end_pos], &after[end_pos..]);
            Some(recovered)
        } else {
            None
        }
    }

    /// Skip to next valid token.
    fn skip_to_valid(&self, input: &str, position: usize) -> Option<String> {
        let after = &input[position..];

        // Skip to next opening parenthesis or start of identifier
        let skip_pos = after
            .find(|c: char| c == '(' || c.is_alphabetic())
            .unwrap_or(after.len());

        if skip_pos < after.len() {
            Some(format!("{}{}", &input[..position], &after[skip_pos..]))
        } else {
            None
        }
    }

    /// Attempt to recover from a type error.
    pub fn recover_type_error(
        &mut self,
        error: ErrorContext,
        expected: &str,
        found: &str,
    ) -> RecoveryResult<String> {
        self.stats.attempts += 1;

        // Check for common type coercions
        let suggestions = self.suggest_type_coercion(expected, found);

        if !suggestions.is_empty() {
            self.stats.partial += 1;
            RecoveryResult::Partial {
                value: suggestions[0].clone(),
                warnings: suggestions.iter().skip(1).cloned().collect(),
            }
        } else {
            self.stats.failures += 1;
            self.errors.push(error.clone());
            RecoveryResult::Failed(error)
        }
    }

    /// Suggest type coercions.
    fn suggest_type_coercion(&self, expected: &str, found: &str) -> Vec<String> {
        let mut suggestions = Vec::new();

        // Int -> Real coercion
        if expected == "Real" && found == "Int" {
            suggestions.push("Consider using (to_real <value>)".to_string());
        }

        // Real -> Int coercion
        if expected == "Int" && found == "Real" {
            suggestions.push("Consider using (to_int <value>) or (floor <value>)".to_string());
        }

        // BitVec width mismatch
        if expected.starts_with("BitVec") && found.starts_with("BitVec") {
            suggestions.push("Consider using (concat ...) or (extract ...)".to_string());
        }

        suggestions
    }

    /// Get collected errors.
    pub fn errors(&self) -> &[ErrorContext] {
        &self.errors
    }

    /// Get collected warnings.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Check if there are any errors.
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> &RecoveryStats {
        &self.stats
    }

    /// Clear collected errors and warnings.
    pub fn clear(&mut self) {
        self.errors.clear();
        self.warnings.clear();
    }
}

/// Error batch collector for collecting multiple errors before reporting.
#[derive(Debug, Default)]
pub struct ErrorBatch {
    errors: Vec<ErrorContext>,
    max_errors: Option<usize>,
}

impl ErrorBatch {
    /// Create a new error batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with maximum error limit.
    pub fn with_limit(max_errors: usize) -> Self {
        Self {
            errors: Vec::new(),
            max_errors: Some(max_errors),
        }
    }

    /// Add an error to the batch.
    pub fn add(&mut self, error: ErrorContext) -> Result<(), ErrorContext> {
        if let Some(max) = self.max_errors
            && self.errors.len() >= max
        {
            return Err(error);
        }

        self.errors.push(error);
        Ok(())
    }

    /// Check if batch is full.
    pub fn is_full(&self) -> bool {
        if let Some(max) = self.max_errors {
            self.errors.len() >= max
        } else {
            false
        }
    }

    /// Get all errors.
    pub fn errors(&self) -> &[ErrorContext] {
        &self.errors
    }

    /// Get error count.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Take all errors, leaving batch empty.
    pub fn take_errors(&mut self) -> Vec<ErrorContext> {
        std::mem::take(&mut self.errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{OxizError, SourceLocation};

    #[test]
    fn test_recovery_config_default() {
        let config = RecoveryConfig::default();
        assert_eq!(config.strategy, RecoveryStrategy::Heuristic);
        assert!(config.collect_diagnostics);
    }

    #[test]
    fn test_recovery_stats() {
        let mut stats = RecoveryStats::new();
        assert_eq!(stats.success_rate(), 0.0);

        stats.attempts = 10;
        stats.successes = 7;
        assert_eq!(stats.success_rate(), 0.7);
    }

    #[test]
    fn test_error_batch() {
        let mut batch = ErrorBatch::with_limit(2);

        let loc = SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(loc);
        let err1 = ErrorContext::new(OxizError::parse_error(span, "error 1"));
        let err2 = ErrorContext::new(OxizError::parse_error(span, "error 2"));
        let err3 = ErrorContext::new(OxizError::parse_error(span, "error 3"));

        assert!(batch.add(err1).is_ok());
        assert!(batch.add(err2).is_ok());
        assert!(batch.is_full());
        assert!(batch.add(err3).is_err());

        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_unbalanced_parens_recovery() {
        let recovery = ErrorRecovery::default_config();

        // Missing closing paren
        let input = "(assert (> x 0)";
        let recovered = recovery.recover_unbalanced_parens(input, input.len());
        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap(), "(assert (> x 0))");

        // Extra closing paren - need to include the extra ')' in the before portion
        let input2 = "(assert (> x 0)))";
        let recovered2 = recovery.recover_unbalanced_parens(input2, input2.len());
        assert!(recovered2.is_some());
    }

    #[test]
    fn test_type_coercion_suggestions() {
        let recovery = ErrorRecovery::default_config();

        let suggestions = recovery.suggest_type_coercion("Real", "Int");
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("to_real"));

        let suggestions2 = recovery.suggest_type_coercion("Int", "Real");
        assert!(!suggestions2.is_empty());
        assert!(suggestions2[0].contains("to_int"));
    }
}
