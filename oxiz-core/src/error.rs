//! Error types for OxiZ

#[allow(unused_imports)]
use crate::prelude::*;
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

impl core::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

impl core::fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
#[derive(Debug, Clone, Error)]
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
        /// Source location of the mismatch
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
        /// Source location of the parse error
        location: SourceSpan,
        /// Error message
        message: String,
    },
    /// Parse error (legacy)
    #[error("parse error at position {position}: {message}")]
    ParseError {
        /// Byte position of the error
        position: usize,
        /// Error message
        message: String,
    },
    /// Undefined symbol error
    #[error("undefined symbol at {location}: {symbol}")]
    UndefinedSymbol {
        /// Source location where the symbol was referenced
        location: SourceSpan,
        /// The undefined symbol name
        symbol: String,
    },
    /// Type error
    #[error("type error at {location}: {message}")]
    TypeError {
        /// Source location of the type error
        location: SourceSpan,
        /// Error message
        message: String,
    },
    /// Arity mismatch
    #[error("arity mismatch at {location}: expected {expected} arguments, found {found}")]
    ArityMismatch {
        /// Source location of the arity mismatch
        location: SourceSpan,
        /// Expected number of arguments
        expected: usize,
        /// Found number of arguments
        found: usize,
    },
    /// Solver returned unknown
    #[error("solver returned unknown: {reason}")]
    Unknown {
        /// Reason the solver returned unknown
        reason: String,
    },
    /// Unsupported operation
    #[error("unsupported at {location}: {message}")]
    UnsupportedWithLocation {
        /// Source location of the unsupported operation
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
    /// E-matching error
    #[error("e-matching error: {0}")]
    EmatchError(String),
    /// Resource exhausted (timeout, conflict limit, etc.)
    #[error("resource exhausted: {reason}")]
    ResourceExhausted {
        /// Reason for resource exhaustion
        reason: String,
    },
}

/// Enhanced error with optional hint, did-you-mean suggestion, and context snippet.
#[derive(Debug, Clone)]
pub struct EnhancedError {
    /// The underlying error
    pub error: OxizError,
    /// Optional hint for how to fix the error
    pub hint: Option<String>,
    /// Optional "did you mean?" suggestion
    pub did_you_mean: Option<String>,
    /// Optional source context snippet showing the relevant line
    pub context_snippet: Option<String>,
}

impl core::fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.error)?;
        if let Some(ref snippet) = self.context_snippet {
            write!(f, "\n  | {}", snippet)?;
        }
        if let Some(ref suggestion) = self.did_you_mean {
            write!(f, "\n  note: did you mean '{}'?", suggestion)?;
        }
        if let Some(ref hint) = self.hint {
            write!(f, "\n  hint: {}", hint)?;
        }
        Ok(())
    }
}

impl EnhancedError {
    /// Create a new enhanced error from a base error.
    #[must_use]
    pub fn new(error: OxizError) -> Self {
        Self {
            error,
            hint: None,
            did_you_mean: None,
            context_snippet: None,
        }
    }
    /// Attach a hint.
    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }
    /// Attach a "did you mean?" suggestion.
    #[must_use]
    pub fn with_did_you_mean(mut self, suggestion: impl Into<String>) -> Self {
        self.did_you_mean = Some(suggestion.into());
        self
    }
    /// Attach a context snippet.
    #[must_use]
    pub fn with_context_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.context_snippet = Some(snippet.into());
        self
    }
}

/// Compute the Levenshtein edit distance between two strings.
#[must_use]
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let (a_bytes, b_bytes) = (a.as_bytes(), b.as_bytes());
    let (a_len, b_len) = (a_bytes.len(), b_bytes.len());
    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }
    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row = vec![0usize; b_len + 1];
    for i in 1..=a_len {
        curr_row[0] = i;
        for j in 1..=b_len {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };
            curr_row[j] = (prev_row[j] + 1)
                .min(curr_row[j - 1] + 1)
                .min(prev_row[j - 1] + cost);
        }
        core::mem::swap(&mut prev_row, &mut curr_row);
    }
    prev_row[b_len]
}

/// Find the closest match to `name` among `candidates` using Levenshtein distance.
#[must_use]
pub fn find_closest_match<'a>(
    name: &str,
    candidates: impl IntoIterator<Item = &'a str>,
    max_distance: usize,
) -> Option<&'a str> {
    let mut best: Option<(&str, usize)> = None;
    for candidate in candidates {
        let dist = levenshtein_distance(name, candidate);
        if dist <= max_distance {
            match best {
                Some((_, best_dist)) if dist < best_dist => best = Some((candidate, dist)),
                None => best = Some((candidate, dist)),
                _ => {}
            }
        }
    }
    best.map(|(s, _)| s)
}

/// Extract a context snippet from source text at the given location.
#[must_use]
pub fn extract_context_snippet(source: &str, location: &SourceSpan) -> Option<String> {
    let line_num = location.start.line;
    if line_num == 0 {
        return None;
    }
    let line = source.lines().nth(line_num.saturating_sub(1))?;
    let trimmed = line.trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let col = location.start.column.saturating_sub(1);
    let end_col = if location.start.line == location.end.line {
        location.end.column.saturating_sub(1).max(col + 1)
    } else {
        trimmed.len()
    };
    let underline_len = end_col.saturating_sub(col).max(1);
    let mut result = String::new();
    result.push_str(&format!("{:>4} | {}\n", line_num, trimmed));
    result.push_str(&format!(
        "     | {}{}",
        " ".repeat(col),
        "^".repeat(underline_len)
    ));
    Some(result)
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
            OxizError::ParseError { position, message } => format!(
                "Parsing failed at byte offset {position}: {message}\nHint: Check for missing parentheses or invalid syntax near this position."
            ),
            OxizError::ParseErrorWithLocation { location, message } => format!(
                "Parse error at {location}: {message}\nHint: Check the syntax near line {} column {}.",
                location.start.line, location.start.column
            ),
            OxizError::SortMismatch {
                location,
                expected,
                found,
            } => format!(
                "Type mismatch at {location}: expected {expected}, but found {found}\nHint: Ensure all operands have compatible types. You may need to add explicit type conversions."
            ),
            OxizError::UndefinedSymbol { location, symbol } => format!(
                "Undefined symbol '{symbol}' at {location}\nHint: Make sure to declare '{symbol}' with 'declare-const', 'declare-fun', or 'define-fun' before using it."
            ),
            OxizError::ArityMismatch {
                location,
                expected,
                found,
            } => format!(
                "Wrong number of arguments at {location}: expected {expected}, found {found}\nHint: Check the function/operator signature and provide exactly {expected} argument(s)."
            ),
            OxizError::ResourceExhausted { reason } => format!(
                "Resource exhausted: {reason}\nHint: Increase the corresponding limit or simplify the problem."
            ),
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
    /// Create an enhanced error with optional "did you mean?" lookup.
    #[must_use]
    pub fn enhance(self, source: Option<&str>, known_symbols: &[&str]) -> EnhancedError {
        let mut enhanced = EnhancedError::new(self.clone());
        let hint = match &self {
            OxizError::ParseError { .. } | OxizError::ParseErrorWithLocation { .. } => {
                Some("Check for missing parentheses or invalid syntax.".to_string())
            }
            OxizError::SortMismatch {
                expected, found, ..
            } => Some(format!(
                "Ensure operands have compatible types. Expected {expected}, found {found}."
            )),
            OxizError::UndefinedSymbol { symbol, .. } => Some(format!(
                "Declare '{symbol}' with 'declare-const' or 'declare-fun' before using it."
            )),
            OxizError::ArityMismatch { expected, .. } => {
                Some(format!("Provide exactly {expected} argument(s)."))
            }
            _ => None,
        };
        if let Some(h) = hint {
            enhanced.hint = Some(h);
        }
        if let OxizError::UndefinedSymbol { symbol, .. } = &self
            && let Some(closest) = find_closest_match(symbol, known_symbols.iter().copied(), 3)
        {
            enhanced.did_you_mean = Some(closest.to_string());
        }
        if let Some(src) = source {
            let span = match &self {
                OxizError::ParseErrorWithLocation { location, .. }
                | OxizError::SortMismatch { location, .. }
                | OxizError::UndefinedSymbol { location, .. }
                | OxizError::TypeError { location, .. }
                | OxizError::ArityMismatch { location, .. }
                | OxizError::UnsupportedWithLocation { location, .. } => Some(location),
                _ => None,
            };
            if let Some(span) = span
                && let Some(snippet) = extract_context_snippet(src, span)
            {
                enhanced.context_snippet = Some(snippet);
            }
        }
        enhanced
    }
}

/// Result type alias using OxizError
pub type Result<T> = core::result::Result<T, OxizError>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_source_location_display() {
        assert_eq!(SourceLocation::new(5, 10, 42).to_string(), "5:10");
    }
    #[test]
    fn test_source_span_display_same_line() {
        assert_eq!(
            SourceSpan::new(
                SourceLocation::new(5, 10, 42),
                SourceLocation::new(5, 20, 52)
            )
            .to_string(),
            "5:10-20"
        );
    }
    #[test]
    fn test_source_span_display_multi_line() {
        assert_eq!(
            SourceSpan::new(
                SourceLocation::new(5, 10, 42),
                SourceLocation::new(7, 5, 82)
            )
            .to_string(),
            "5:10-7:5"
        );
    }
    #[test]
    fn test_error_constructors() {
        let span = SourceSpan::from_location(SourceLocation::new(5, 10, 42));
        assert!(matches!(
            OxizError::sort_mismatch(span, "Int", "Bool"),
            OxizError::SortMismatch { .. }
        ));
        assert!(matches!(
            OxizError::parse_error(span, "unexpected token"),
            OxizError::ParseErrorWithLocation { .. }
        ));
        assert!(matches!(
            OxizError::undefined_symbol(span, "foo"),
            OxizError::UndefinedSymbol { .. }
        ));
        assert!(matches!(
            OxizError::type_error(span, "cannot apply"),
            OxizError::TypeError { .. }
        ));
        assert!(matches!(
            OxizError::arity_mismatch(span, 2, 3),
            OxizError::ArityMismatch { .. }
        ));
    }
    #[test]
    fn test_detailed_error_messages() {
        let span = SourceSpan::from_location(SourceLocation::new(5, 10, 42));
        let d = OxizError::sort_mismatch(span, "Int", "Bool").detailed_message();
        assert!(d.contains("Hint") && d.contains("Int") && d.contains("Bool"));
        let d = OxizError::undefined_symbol(span, "foo").detailed_message();
        assert!(d.contains("declare") && d.contains("foo"));
        let d = OxizError::arity_mismatch(span, 2, 3).detailed_message();
        assert!(d.contains("2") && d.contains("3"));
    }
    #[test]
    fn test_is_recoverable() {
        let span = SourceSpan::from_location(SourceLocation::new(5, 10, 42));
        assert!(OxizError::parse_error(span, "test").is_recoverable());
        assert!(OxizError::undefined_symbol(span, "foo").is_recoverable());
        assert!(!OxizError::Internal("test".to_string()).is_recoverable());
        assert!(!OxizError::sort_mismatch(span, "Int", "Bool").is_recoverable());
    }
    #[test]
    fn test_levenshtein_distance_identical() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }
    #[test]
    fn test_levenshtein_distance_one_edit() {
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
        assert_eq!(levenshtein_distance("hello", "helloo"), 1);
        assert_eq!(levenshtein_distance("hello", "hella"), 1);
    }
    #[test]
    fn test_levenshtein_distance_empty() {
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("", ""), 0);
    }
    #[test]
    fn test_find_closest_match() {
        assert_eq!(
            find_closest_match(
                "assrt",
                vec!["declare-const", "declare-fun", "assert", "check-sat"].into_iter(),
                2
            ),
            Some("assert")
        );
    }
    #[test]
    fn test_find_closest_match_no_match() {
        assert!(
            find_closest_match(
                "xyz_totally_different",
                vec!["declare-const", "declare-fun"].into_iter(),
                3
            )
            .is_none()
        );
    }
    #[test]
    fn test_extract_context_snippet() {
        let source = "(declare-const x Int)\n(assert (> x 0))\n(check-sat)";
        let span = SourceSpan::new(
            SourceLocation::new(2, 9, 30),
            SourceLocation::new(2, 14, 35),
        );
        let s = extract_context_snippet(source, &span).expect("should have snippet");
        assert!(s.contains("(assert (> x 0))") && s.contains("^"));
    }
    #[test]
    fn test_enhanced_error_display() {
        let span = SourceSpan::from_location(SourceLocation::new(1, 5, 4));
        let enhanced = EnhancedError::new(OxizError::undefined_symbol(span, "foo_bar"))
            .with_hint("declare the symbol first")
            .with_did_you_mean("foo_baz");
        let display = format!("{}", enhanced);
        assert!(
            display.contains("foo_bar")
                && display.contains("did you mean 'foo_baz'")
                && display.contains("hint: declare the symbol first")
        );
    }
    #[test]
    fn test_enhance_undefined_symbol_with_candidates() {
        let span = SourceSpan::from_location(SourceLocation::new(1, 10, 9));
        let enhanced = OxizError::undefined_symbol(span, "varx")
            .enhance(Some("(assert (> varx 0))"), &["var_x", "var_y", "total"]);
        assert_eq!(enhanced.did_you_mean.as_deref(), Some("var_x"));
        assert!(enhanced.hint.is_some() && enhanced.context_snippet.is_some());
    }
    #[test]
    fn test_resource_exhausted_error() {
        let msg = OxizError::ResourceExhausted {
            reason: "timeout".to_string(),
        }
        .detailed_message();
        assert!(msg.contains("timeout") && msg.contains("Hint"));
    }
}
