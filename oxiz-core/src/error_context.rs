//! Error Context System for Rich Error Reporting.
#![allow(clippy::result_large_err)] // ErrorContext is rich by design
//!
//! Provides error chaining and context accumulation similar to anyhow,
//! but tailored for SMT solver operations.

use crate::error::OxizError;
use std::fmt;

/// Error context providing additional information about where/why an error occurred.
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The underlying error.
    pub error: OxizError,
    /// Stack of context messages (most recent first).
    pub context_stack: Vec<String>,
    /// Optional related term IDs for debugging.
    pub related_terms: Vec<u32>,
    /// Suggestions for fixing the error.
    pub suggestions: Vec<String>,
}

impl ErrorContext {
    /// Create a new error context from an OxizError.
    pub fn new(error: OxizError) -> Self {
        Self {
            error,
            context_stack: Vec::new(),
            related_terms: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Add context to the error.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context_stack.push(context.into());
        self
    }

    /// Add a suggestion for fixing the error.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add a related term ID for debugging.
    pub fn with_related_term(mut self, term_id: u32) -> Self {
        self.related_terms.push(term_id);
        self
    }

    /// Get a fully formatted error message with all context.
    pub fn format_full(&self) -> String {
        let mut output = String::new();

        // Main error
        output.push_str(&format!("Error: {}\n", self.error.detailed_message()));

        // Context stack
        if !self.context_stack.is_empty() {
            output.push_str("\nContext:\n");
            for (i, ctx) in self.context_stack.iter().rev().enumerate() {
                output.push_str(&format!("  {}. {}\n", i + 1, ctx));
            }
        }

        // Related terms
        if !self.related_terms.is_empty() {
            output.push_str(&format!("\nRelated terms: {:?}\n", self.related_terms));
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            output.push_str("\nSuggestions:\n");
            for suggestion in &self.suggestions {
                output.push_str(&format!("  • {}\n", suggestion));
            }
        }

        output
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_full())
    }
}

impl std::error::Error for ErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Extension trait for adding context to Results.
pub trait ResultExt<T> {
    /// Add context to an error.
    fn context(self, context: impl Into<String>) -> Result<T, ErrorContext>;

    /// Add context with a lazy closure (only evaluated on error).
    fn with_context<F>(self, f: F) -> Result<T, ErrorContext>
    where
        F: FnOnce() -> String;
}

impl<T> ResultExt<T> for Result<T, OxizError> {
    fn context(self, context: impl Into<String>) -> Result<T, ErrorContext> {
        self.map_err(|e| ErrorContext::new(e).with_context(context))
    }

    fn with_context<F>(self, f: F) -> Result<T, ErrorContext>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| ErrorContext::new(e).with_context(f()))
    }
}

impl<T> ResultExt<T> for Result<T, ErrorContext> {
    fn context(self, context: impl Into<String>) -> Result<T, ErrorContext> {
        self.map_err(|mut e| {
            e.context_stack.push(context.into());
            e
        })
    }

    fn with_context<F>(self, f: F) -> Result<T, ErrorContext>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|mut e| {
            e.context_stack.push(f());
            e
        })
    }
}

/// Convenience macro for adding context to an operation.
#[macro_export]
macro_rules! context {
    ($result:expr, $($arg:tt)*) => {
        $result.context(format!($($arg)*))
    };
}

/// Convenience macro for returning an error with context.
#[macro_export]
macro_rules! bail {
    ($error:expr) => {
        return Err($crate::error_context::ErrorContext::new($error))
    };
    ($error:expr, $($context:tt)*) => {
        return Err($crate::error_context::ErrorContext::new($error).with_context(format!($($context)*)))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::SourceLocation;

    #[test]
    fn test_error_context_creation() {
        let loc = SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(loc);
        let err = OxizError::parse_error(span, "test error");

        let ctx = ErrorContext::new(err);
        assert!(ctx.context_stack.is_empty());
        assert!(ctx.suggestions.is_empty());
    }

    #[test]
    fn test_add_context() {
        let loc = SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(loc);
        let err = OxizError::parse_error(span, "test error");

        let ctx = ErrorContext::new(err)
            .with_context("while parsing formula")
            .with_context("in check-sat command");

        assert_eq!(ctx.context_stack.len(), 2);
        assert_eq!(ctx.context_stack[0], "while parsing formula");
        assert_eq!(ctx.context_stack[1], "in check-sat command");
    }

    #[test]
    fn test_add_suggestion() {
        let loc = SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(loc);
        let err = OxizError::undefined_symbol(span, "x");

        let ctx = ErrorContext::new(err)
            .with_suggestion("Try declaring 'x' with (declare-const x Int)")
            .with_suggestion("Check for typos in the variable name");

        assert_eq!(ctx.suggestions.len(), 2);
    }

    #[test]
    fn test_related_terms() {
        let err = OxizError::Internal("test".to_string());

        let ctx = ErrorContext::new(err)
            .with_related_term(42)
            .with_related_term(43);

        assert_eq!(ctx.related_terms, vec![42, 43]);
    }

    #[test]
    fn test_result_ext() {
        fn might_fail() -> Result<i32, OxizError> {
            let loc = SourceLocation::start();
            let span = crate::error::SourceSpan::from_location(loc);
            Err(OxizError::parse_error(span, "test"))
        }

        let result = might_fail().context("in test function");

        assert!(result.is_err());
        if let Err(ctx) = result {
            assert_eq!(ctx.context_stack.len(), 1);
            assert_eq!(ctx.context_stack[0], "in test function");
        }
    }

    #[test]
    fn test_chained_context() {
        fn inner() -> Result<i32, OxizError> {
            let loc = SourceLocation::start();
            let span = crate::error::SourceSpan::from_location(loc);
            Err(OxizError::parse_error(span, "inner error"))
        }

        fn middle() -> Result<i32, ErrorContext> {
            inner().context("in middle layer")
        }

        fn outer() -> Result<i32, ErrorContext> {
            middle().context("in outer layer")
        }

        let result = outer();
        assert!(result.is_err());
        if let Err(ctx) = result {
            assert_eq!(ctx.context_stack.len(), 2);
            // Stack is most recent first
            assert!(ctx.context_stack[0].contains("middle"));
            assert!(ctx.context_stack[1].contains("outer"));
        }
    }

    #[test]
    fn test_format_full() {
        let loc = SourceLocation::start();
        let span = crate::error::SourceSpan::from_location(loc);
        let err = OxizError::undefined_symbol(span, "foo");

        let ctx = ErrorContext::new(err)
            .with_context("while type checking")
            .with_suggestion("Declare 'foo' before using it")
            .with_related_term(42);

        let formatted = ctx.format_full();
        assert!(formatted.contains("Error:"));
        assert!(formatted.contains("Context:"));
        assert!(formatted.contains("Suggestions:"));
        assert!(formatted.contains("Related terms:"));
    }
}
