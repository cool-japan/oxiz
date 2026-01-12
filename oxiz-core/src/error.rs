//! Error types for OxiZ

use thiserror::Error;

/// Source location information for error reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceLocation {
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub column: usize,
    /// Byte offset in source
    pub offset: usize,
}

impl SourceLocation {
    /// Create a new source location
    #[must_use]
    pub const fn new(line: usize, column: usize, offset: usize) -> Self {
        Self {
            line,
            column,
            offset,
        }
    }

    /// Create a default location (beginning of file)
    #[must_use]
    pub const fn start() -> Self {
        Self {
            line: 1,
            column: 1,
            offset: 0,
        }
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Span of source code (from start to end)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceSpan {
    /// Start location
    pub start: SourceLocation,
    /// End location
    pub end: SourceLocation,
}

impl SourceSpan {
    /// Create a new source span
    #[must_use]
    pub const fn new(start: SourceLocation, end: SourceLocation) -> Self {
        Self { start, end }
    }

    /// Create a span from a single location
    #[must_use]
    pub const fn from_location(loc: SourceLocation) -> Self {
        Self {
            start: loc,
            end: loc,
        }
    }
}

impl std::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.start.line == self.end.line {
            write!(
                f,
                "{}:{}-{}",
                self.start.line, self.start.column, self.end.column
            )
        } else {
            write!(f, "{}-{}", self.start, self.end)
        }
    }
}

/// Main error type for OxiZ operations
#[derive(Debug, Error)]
pub enum OxizError {
    /// Invalid term reference
    #[error("invalid term ID: {0}")]
    InvalidTermId(u32),

    /// Invalid sort reference
    #[error("invalid sort ID: {0}")]
    InvalidSortId(u32),

    /// Sort mismatch during type checking
    #[error("sort mismatch at {location}: expected {expected}, found {found}")]
    SortMismatch {
        /// Location of the error
        location: SourceSpan,
        /// Expected sort
        expected: String,
        /// Found sort
        found: String,
    },

    /// Sort mismatch without location (for legacy code)
    #[error("sort mismatch: expected {expected}, found {found}")]
    SortMismatchSimple {
        /// Expected sort
        expected: String,
        /// Found sort
        found: String,
    },

    /// Parse error with location
    #[error("parse error at {location}: {message}")]
    ParseErrorWithLocation {
        /// Location of the error
        location: SourceSpan,
        /// Error message
        message: String,
    },

    /// Parse error (legacy)
    #[error("parse error at position {position}: {message}")]
    ParseError {
        /// Position in input
        position: usize,
        /// Error message
        message: String,
    },

    /// Undefined symbol error
    #[error("undefined symbol at {location}: {symbol}")]
    UndefinedSymbol {
        /// Location of the error
        location: SourceSpan,
        /// Symbol name
        symbol: String,
    },

    /// Type error
    #[error("type error at {location}: {message}")]
    TypeError {
        /// Location of the error
        location: SourceSpan,
        /// Error message
        message: String,
    },

    /// Arity mismatch
    #[error("arity mismatch at {location}: expected {expected} arguments, found {found}")]
    ArityMismatch {
        /// Location of the error
        location: SourceSpan,
        /// Expected arity
        expected: usize,
        /// Found arity
        found: usize,
    },

    /// Solver returned unknown
    #[error("solver returned unknown: {reason}")]
    Unknown {
        /// Reason for unknown result
        reason: String,
    },

    /// Unsupported operation
    #[error("unsupported at {location}: {message}")]
    UnsupportedWithLocation {
        /// Location of the error
        location: SourceSpan,
        /// Error message
        message: String,
    },

    /// Unsupported operation (legacy)
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// Internal error
    #[error("internal error: {0}")]
    Internal(String),
}

impl OxizError {
    /// Create a sort mismatch error with location
    pub fn sort_mismatch(
        location: SourceSpan,
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        Self::SortMismatch {
            location,
            expected: expected.into(),
            found: found.into(),
        }
    }

    /// Create a parse error with location
    pub fn parse_error(location: SourceSpan, message: impl Into<String>) -> Self {
        Self::ParseErrorWithLocation {
            location,
            message: message.into(),
        }
    }

    /// Create an undefined symbol error
    pub fn undefined_symbol(location: SourceSpan, symbol: impl Into<String>) -> Self {
        Self::UndefinedSymbol {
            location,
            symbol: symbol.into(),
        }
    }

    /// Create a type error
    pub fn type_error(location: SourceSpan, message: impl Into<String>) -> Self {
        Self::TypeError {
            location,
            message: message.into(),
        }
    }

    /// Create an arity mismatch error
    pub fn arity_mismatch(location: SourceSpan, expected: usize, found: usize) -> Self {
        Self::ArityMismatch {
            location,
            expected,
            found,
        }
    }

    /// Create an unsupported operation error with location
    pub fn unsupported(location: SourceSpan, message: impl Into<String>) -> Self {
        Self::UnsupportedWithLocation {
            location,
            message: message.into(),
        }
    }

    /// Get a user-friendly error message with suggestions
    #[must_use]
    pub fn detailed_message(&self) -> String {
        match self {
            OxizError::ParseError { position, message } => {
                format!(
                    "Parsing failed at byte offset {position}: {message}\n\
                     Hint: Check for missing parentheses or invalid syntax near this position."
                )
            }
            OxizError::SortMismatch {
                location,
                expected,
                found,
            } => {
                format!(
                    "Type mismatch at {location}: expected {expected}, but found {found}\n\
                     Hint: Ensure all operands have compatible types. You may need to add explicit type conversions."
                )
            }
            OxizError::UndefinedSymbol { location, symbol } => {
                format!(
                    "Undefined symbol '{symbol}' at {location}\n\
                     Hint: Make sure to declare '{symbol}' with 'declare-const', 'declare-fun', or 'define-fun' before using it."
                )
            }
            OxizError::ArityMismatch {
                location,
                expected,
                found,
            } => {
                format!(
                    "Wrong number of arguments at {location}: expected {expected}, found {found}\n\
                     Hint: Check the function/operator signature and provide exactly {expected} argument(s)."
                )
            }
            _ => self.to_string(),
        }
    }

    /// Check if this is a recoverable error
    #[must_use]
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            OxizError::ParseError { .. }
                | OxizError::ParseErrorWithLocation { .. }
                | OxizError::UndefinedSymbol { .. }
        )
    }
}

/// Result type alias using OxizError
pub type Result<T> = std::result::Result<T, OxizError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation::new(5, 10, 42);
        assert_eq!(loc.to_string(), "5:10");
    }

    #[test]
    fn test_source_span_display_same_line() {
        let start = SourceLocation::new(5, 10, 42);
        let end = SourceLocation::new(5, 20, 52);
        let span = SourceSpan::new(start, end);
        assert_eq!(span.to_string(), "5:10-20");
    }

    #[test]
    fn test_source_span_display_multi_line() {
        let start = SourceLocation::new(5, 10, 42);
        let end = SourceLocation::new(7, 5, 82);
        let span = SourceSpan::new(start, end);
        assert_eq!(span.to_string(), "5:10-7:5");
    }

    #[test]
    fn test_error_constructors() {
        let loc = SourceLocation::new(5, 10, 42);
        let span = SourceSpan::from_location(loc);

        let err = OxizError::sort_mismatch(span, "Int", "Bool");
        assert!(matches!(err, OxizError::SortMismatch { .. }));

        let err = OxizError::parse_error(span, "unexpected token");
        assert!(matches!(err, OxizError::ParseErrorWithLocation { .. }));

        let err = OxizError::undefined_symbol(span, "foo");
        assert!(matches!(err, OxizError::UndefinedSymbol { .. }));

        let err = OxizError::type_error(span, "cannot apply");
        assert!(matches!(err, OxizError::TypeError { .. }));

        let err = OxizError::arity_mismatch(span, 2, 3);
        assert!(matches!(err, OxizError::ArityMismatch { .. }));
    }

    #[test]
    fn test_detailed_error_messages() {
        let loc = SourceLocation::new(5, 10, 42);
        let span = SourceSpan::from_location(loc);

        // Test sort mismatch detailed message
        let err = OxizError::sort_mismatch(span, "Int", "Bool");
        let detailed = err.detailed_message();
        assert!(detailed.contains("Hint"));
        assert!(detailed.contains("Int"));
        assert!(detailed.contains("Bool"));

        // Test undefined symbol detailed message
        let err = OxizError::undefined_symbol(span, "foo");
        let detailed = err.detailed_message();
        assert!(detailed.contains("declare"));
        assert!(detailed.contains("foo"));

        // Test arity mismatch detailed message
        let err = OxizError::arity_mismatch(span, 2, 3);
        let detailed = err.detailed_message();
        assert!(detailed.contains("2"));
        assert!(detailed.contains("3"));
    }

    #[test]
    fn test_is_recoverable() {
        let loc = SourceLocation::new(5, 10, 42);
        let span = SourceSpan::from_location(loc);

        // Recoverable errors
        let err = OxizError::parse_error(span, "test");
        assert!(err.is_recoverable());

        let err = OxizError::undefined_symbol(span, "foo");
        assert!(err.is_recoverable());

        // Non-recoverable errors
        let err = OxizError::Internal("test".to_string());
        assert!(!err.is_recoverable());

        let err = OxizError::sort_mismatch(span, "Int", "Bool");
        assert!(!err.is_recoverable());
    }
}
