//! Error Utilities and Helper Functions.
#![allow(clippy::result_large_err)] // ErrorContext is rich by design
//!
//! Provides convenience functions for common error handling patterns.

use crate::error::{OxizError, SourceLocation, SourceSpan};
use crate::error_context::ErrorContext;
use crate::{Sort, TermId};

/// Result type with ErrorContext.
pub type ContextResult<T> = Result<T, ErrorContext>;

/// Helper to create a parse error with context.
pub fn parse_error(location: SourceLocation, message: impl Into<String>) -> ErrorContext {
    let span = SourceSpan::from_location(location);
    ErrorContext::new(OxizError::parse_error(span, message.into()))
}

/// Helper to create a type error with context.
pub fn type_error(location: SourceLocation, message: impl Into<String>) -> ErrorContext {
    let span = SourceSpan::from_location(location);
    ErrorContext::new(OxizError::type_error(span, message.into()))
}

/// Helper to create a sort mismatch error with context.
pub fn sort_mismatch(
    location: SourceLocation,
    expected: impl Into<String>,
    found: impl Into<String>,
) -> ErrorContext {
    let span = SourceSpan::from_location(location);
    ErrorContext::new(OxizError::sort_mismatch(
        span,
        expected.into(),
        found.into(),
    ))
}

/// Helper to create an undefined symbol error with context.
pub fn undefined_symbol(location: SourceLocation, symbol: impl Into<String>) -> ErrorContext {
    let span = SourceSpan::from_location(location);
    ErrorContext::new(OxizError::undefined_symbol(span, symbol.into()))
}

/// Helper to create an arity mismatch error with context.
pub fn arity_mismatch(location: SourceLocation, expected: usize, found: usize) -> ErrorContext {
    let span = SourceSpan::from_location(location);
    ErrorContext::new(OxizError::arity_mismatch(span, expected, found))
}

/// Extension trait for Sort error handling.
pub trait SortErrorExt {
    /// Check if two sorts match, returning an error if not.
    fn expect_match(&self, other: &Self, location: SourceLocation) -> ContextResult<()>;

    /// Check if this is a boolean sort.
    fn expect_bool(&self, location: SourceLocation) -> ContextResult<()>;

    /// Check if this is an integer sort.
    fn expect_int(&self, location: SourceLocation) -> ContextResult<()>;
}

impl SortErrorExt for Sort {
    fn expect_match(&self, other: &Self, location: SourceLocation) -> ContextResult<()> {
        if self.kind != other.kind {
            Err(sort_mismatch(
                location,
                format!("{:?}", self.kind),
                format!("{:?}", other.kind),
            ))
        } else {
            Ok(())
        }
    }

    fn expect_bool(&self, location: SourceLocation) -> ContextResult<()> {
        use crate::sort::SortKind;
        match &self.kind {
            SortKind::Bool => Ok(()),
            _ => Err(type_error(
                location,
                format!("expected Bool, found {:?}", self.kind),
            )),
        }
    }

    fn expect_int(&self, location: SourceLocation) -> ContextResult<()> {
        use crate::sort::SortKind;
        match &self.kind {
            SortKind::Int => Ok(()),
            _ => Err(type_error(
                location,
                format!("expected Int, found {:?}", self.kind),
            )),
        }
    }
}

/// Extension trait for Option error handling.
pub trait OptionErrorExt<T> {
    /// Convert None to an error with context.
    fn ok_or_context(self, error: ErrorContext) -> ContextResult<T>;

    /// Convert None to an error using a closure.
    fn ok_or_else_context<F>(self, f: F) -> ContextResult<T>
    where
        F: FnOnce() -> ErrorContext;
}

impl<T> OptionErrorExt<T> for Option<T> {
    fn ok_or_context(self, error: ErrorContext) -> ContextResult<T> {
        self.ok_or(error)
    }

    fn ok_or_else_context<F>(self, f: F) -> ContextResult<T>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.ok_or_else(f)
    }
}

/// Helper to validate argument count.
pub fn validate_arity(
    args: &[TermId],
    expected: usize,
    location: SourceLocation,
) -> ContextResult<()> {
    if args.len() != expected {
        Err(arity_mismatch(location, expected, args.len()))
    } else {
        Ok(())
    }
}

/// Helper to validate argument count (at least N).
pub fn validate_min_arity(
    args: &[TermId],
    min: usize,
    location: SourceLocation,
) -> ContextResult<()> {
    if args.len() < min {
        Err(arity_mismatch(location, min, args.len())
            .with_context(format!("expected at least {} arguments", min)))
    } else {
        Ok(())
    }
}

/// Helper to validate argument count (at most N).
pub fn validate_max_arity(
    args: &[TermId],
    max: usize,
    location: SourceLocation,
) -> ContextResult<()> {
    if args.len() > max {
        Err(arity_mismatch(location, max, args.len())
            .with_context(format!("expected at most {} arguments", max)))
    } else {
        Ok(())
    }
}

/// Helper to validate argument count (range).
pub fn validate_arity_range(
    args: &[TermId],
    min: usize,
    max: usize,
    location: SourceLocation,
) -> ContextResult<()> {
    if args.len() < min || args.len() > max {
        Err(ErrorContext::new(OxizError::Internal(format!(
            "expected {}-{} arguments, found {}",
            min,
            max,
            args.len()
        )))
        .with_context(format!("at {}", location)))
    } else {
        Ok(())
    }
}

/// Suggest common fixes for errors.
pub fn suggest_fixes(error: &mut ErrorContext, error_type: &str) {
    match error_type {
        "undefined_symbol" => {
            error
                .suggestions
                .push("Try declaring the symbol with (declare-const <name> <type>)".to_string());
            error
                .suggestions
                .push("Check for typos in the symbol name".to_string());
        }
        "sort_mismatch" => {
            error
                .suggestions
                .push("Check the types of all arguments".to_string());
            error.suggestions.push(
                "Consider using type conversion functions like (to_real ...) or (to_int ...)"
                    .to_string(),
            );
        }
        "arity_mismatch" => {
            error
                .suggestions
                .push("Check the function signature".to_string());
            error
                .suggestions
                .push("Ensure you're passing the correct number of arguments".to_string());
        }
        "parse_error" => {
            error
                .suggestions
                .push("Check for balanced parentheses".to_string());
            error
                .suggestions
                .push("Ensure all strings are properly quoted".to_string());
            error.suggestions.push("Verify operator syntax".to_string());
        }
        _ => {}
    }
}

/// Create a chain of errors with context.
pub fn error_chain(
    base_error: OxizError,
    contexts: impl IntoIterator<Item = impl Into<String>>,
) -> ErrorContext {
    let mut error = ErrorContext::new(base_error);
    for ctx in contexts {
        error = error.with_context(ctx);
    }
    error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_helper() {
        let loc = SourceLocation::start();
        let err = parse_error(loc, "test error");
        assert!(err.error.to_string().contains("test error"));
    }

    #[test]
    fn test_validate_arity() {
        let loc = SourceLocation::start();
        let args = vec![TermId::new(0), TermId::new(1)];

        assert!(validate_arity(&args, 2, loc).is_ok());
        assert!(validate_arity(&args, 3, loc).is_err());
    }

    #[test]
    fn test_validate_min_arity() {
        let loc = SourceLocation::start();
        let args = vec![TermId::new(0), TermId::new(1), TermId::new(2)];

        assert!(validate_min_arity(&args, 2, loc).is_ok());
        assert!(validate_min_arity(&args, 3, loc).is_ok());
        assert!(validate_min_arity(&args, 4, loc).is_err());
    }

    #[test]
    fn test_option_error_ext() {
        let loc = SourceLocation::start();
        let some_val: Option<i32> = Some(42);
        let none_val: Option<i32> = None;

        assert!(some_val.ok_or_context(parse_error(loc, "error")).is_ok());
        assert!(none_val.ok_or_context(parse_error(loc, "error")).is_err());
    }

    #[test]
    fn test_sort_expect_bool() {
        use crate::sort::{SortId, SortKind};

        let loc = SourceLocation::start();

        // Create Sort instances with proper structure
        let bool_sort = Sort {
            id: SortId::new(0),
            kind: SortKind::Bool,
        };
        let int_sort = Sort {
            id: SortId::new(1),
            kind: SortKind::Int,
        };

        assert!(bool_sort.expect_bool(loc).is_ok());
        assert!(int_sort.expect_bool(loc).is_err());
    }

    #[test]
    fn test_error_chain() {
        let base = OxizError::Internal("base error".to_string());
        let chain = error_chain(base, vec!["context 1", "context 2"]);

        assert_eq!(chain.context_stack.len(), 2);
        assert_eq!(chain.context_stack[0], "context 1");
        assert_eq!(chain.context_stack[1], "context 2");
    }

    #[test]
    fn test_suggest_fixes() {
        let loc = SourceLocation::start();
        let mut err = parse_error(loc, "test");

        suggest_fixes(&mut err, "undefined_symbol");
        assert!(!err.suggestions.is_empty());
        assert!(err.suggestions[0].contains("declare-const"));
    }
}
