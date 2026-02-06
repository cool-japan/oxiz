//! Diagnostic System for User-Friendly Error Reporting.
//!
//! Provides rich, colorful error messages with source context,
//! suggestions, and fix-it hints.

use crate::error::{OxizError, SourceSpan};
use crate::error_context::ErrorContext;
use std::fmt;

/// Severity level for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational message
    Info,
    /// Warning message
    Warning,
    /// Error message
    Error,
    /// Fatal error (cannot continue)
    Fatal,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
            Severity::Fatal => write!(f, "fatal"),
        }
    }
}

/// A diagnostic message with source context.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Severity level
    pub severity: Severity,
    /// Main error message
    pub message: String,
    /// Source span where error occurred
    pub span: Option<SourceSpan>,
    /// Additional notes
    pub notes: Vec<String>,
    /// Suggested fixes
    pub fixes: Vec<Fix>,
    /// Related diagnostics
    pub related: Vec<RelatedDiagnostic>,
}

/// A suggested fix for a diagnostic.
#[derive(Debug, Clone)]
pub struct Fix {
    /// Description of the fix
    pub description: String,
    /// Source span to replace
    pub span: SourceSpan,
    /// Replacement text
    pub replacement: String,
}

/// A related diagnostic (for multi-error scenarios).
#[derive(Debug, Clone)]
pub struct RelatedDiagnostic {
    /// Message for this related diagnostic
    pub message: String,
    /// Source span
    pub span: SourceSpan,
}

impl Diagnostic {
    /// Create a new diagnostic.
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            span: None,
            notes: Vec::new(),
            fixes: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Create an error diagnostic.
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(Severity::Error, message)
    }

    /// Create a warning diagnostic.
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, message)
    }

    /// Create an info diagnostic.
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(Severity::Info, message)
    }

    /// Set the source span.
    pub fn with_span(mut self, span: SourceSpan) -> Self {
        self.span = Some(span);
        self
    }

    /// Add a note.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add a fix suggestion.
    pub fn with_fix(
        mut self,
        description: impl Into<String>,
        span: SourceSpan,
        replacement: impl Into<String>,
    ) -> Self {
        self.fixes.push(Fix {
            description: description.into(),
            span,
            replacement: replacement.into(),
        });
        self
    }

    /// Add a related diagnostic.
    pub fn with_related(mut self, message: impl Into<String>, span: SourceSpan) -> Self {
        self.related.push(RelatedDiagnostic {
            message: message.into(),
            span,
        });
        self
    }

    /// Format the diagnostic for display.
    pub fn format(&self, source: Option<&str>) -> String {
        let mut output = String::new();

        // Main message
        output.push_str(&format!("{}: {}\n", self.severity, self.message));

        // Source context
        if let Some(span) = &self.span {
            output.push_str(&format!("  --> {}\n", span));

            if let Some(src) = source
                && let Some(context) = self.format_source_context(src, span)
            {
                output.push_str(&context);
            }
        }

        // Notes
        for note in &self.notes {
            output.push_str(&format!("  note: {}\n", note));
        }

        // Fixes
        for fix in &self.fixes {
            output.push_str(&format!("  help: {}\n", fix.description));
            if let Some(src) = source
                && let Some(fix_preview) = self.format_fix_preview(src, fix)
            {
                output.push_str(&fix_preview);
            }
        }

        // Related diagnostics
        for related in &self.related {
            output.push_str(&format!(
                "  related: {} at {}\n",
                related.message, related.span
            ));
        }

        output
    }

    /// Format source context with error highlighting.
    fn format_source_context(&self, source: &str, span: &SourceSpan) -> Option<String> {
        let lines: Vec<&str> = source.lines().collect();

        if span.start.line == 0 || span.start.line > lines.len() {
            return None;
        }

        let mut output = String::new();
        let line_idx = span.start.line - 1;
        let line = lines[line_idx];

        // Line number and source
        output.push_str(&format!("{:4} | {}\n", span.start.line, line));

        // Error marker
        let spaces = span.start.column.saturating_sub(1);
        let marker_len = if span.start.line == span.end.line {
            span.end.column.saturating_sub(span.start.column).max(1)
        } else {
            line.len().saturating_sub(span.start.column - 1).max(1)
        };

        output.push_str(&format!(
            "     | {}{}\n",
            " ".repeat(spaces),
            "^".repeat(marker_len)
        ));

        Some(output)
    }

    /// Format fix preview.
    fn format_fix_preview(&self, source: &str, fix: &Fix) -> Option<String> {
        let lines: Vec<&str> = source.lines().collect();

        if fix.span.start.line == 0 || fix.span.start.line > lines.len() {
            return None;
        }

        let line_idx = fix.span.start.line - 1;
        let line = lines[line_idx];

        // Show before and after
        let mut output = String::new();
        output.push_str("     | suggested replacement:\n");

        let before_len = fix.span.start.column - 1;
        let replace_len = if fix.span.start.line == fix.span.end.line {
            fix.span.end.column - fix.span.start.column
        } else {
            line.len() - (fix.span.start.column - 1)
        };

        let before = &line[..before_len.min(line.len())];
        let after = &line[(before_len + replace_len).min(line.len())..];

        output.push_str(&format!("     | {}{}{}\n", before, fix.replacement, after));

        Some(output)
    }
}

/// Convert ErrorContext to Diagnostic.
impl From<ErrorContext> for Diagnostic {
    fn from(ctx: ErrorContext) -> Self {
        let mut diag = match &ctx.error {
            OxizError::ParseErrorWithLocation { location, message } => {
                Diagnostic::error(message).with_span(*location)
            }
            OxizError::SortMismatch {
                location,
                expected,
                found,
            } => Diagnostic::error(format!(
                "type mismatch: expected {}, found {}",
                expected, found
            ))
            .with_span(*location)
            .with_note(format!("expected type: {}", expected))
            .with_note(format!("found type: {}", found)),
            OxizError::UndefinedSymbol { location, symbol } => {
                Diagnostic::error(format!("undefined symbol: {}", symbol))
                    .with_span(*location)
                    .with_note(format!(
                        "consider declaring '{}' with (declare-const {} <type>)",
                        symbol, symbol
                    ))
            }
            OxizError::TypeError { location, message } => {
                Diagnostic::error(message).with_span(*location)
            }
            OxizError::ArityMismatch {
                location,
                expected,
                found,
            } => Diagnostic::error(format!(
                "wrong number of arguments: expected {}, found {}",
                expected, found
            ))
            .with_span(*location),
            _ => Diagnostic::error(ctx.error.to_string()),
        };

        // Add context stack as notes
        for context in ctx.context_stack.iter().rev() {
            diag = diag.with_note(context);
        }

        // Add suggestions as fixes (without specific spans)
        for suggestion in &ctx.suggestions {
            diag.notes.push(format!("help: {}", suggestion));
        }

        diag
    }
}

/// Diagnostic emitter for collecting and displaying diagnostics.
#[derive(Debug, Default)]
pub struct DiagnosticEmitter {
    diagnostics: Vec<Diagnostic>,
    error_count: usize,
    warning_count: usize,
}

impl DiagnosticEmitter {
    /// Create a new diagnostic emitter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Emit a diagnostic.
    pub fn emit(&mut self, diagnostic: Diagnostic) {
        match diagnostic.severity {
            Severity::Error | Severity::Fatal => self.error_count += 1,
            Severity::Warning => self.warning_count += 1,
            Severity::Info => {}
        }
        self.diagnostics.push(diagnostic);
    }

    /// Emit an error diagnostic.
    pub fn error(&mut self, message: impl Into<String>) {
        self.emit(Diagnostic::error(message));
    }

    /// Emit a warning diagnostic.
    pub fn warning(&mut self, message: impl Into<String>) {
        self.emit(Diagnostic::warning(message));
    }

    /// Emit an info diagnostic.
    pub fn info(&mut self, message: impl Into<String>) {
        self.emit(Diagnostic::info(message));
    }

    /// Check if there are any errors.
    pub fn has_errors(&self) -> bool {
        self.error_count > 0
    }

    /// Get error count.
    pub fn error_count(&self) -> usize {
        self.error_count
    }

    /// Get warning count.
    pub fn warning_count(&self) -> usize {
        self.warning_count
    }

    /// Get all diagnostics.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Format all diagnostics.
    pub fn format_all(&self, source: Option<&str>) -> String {
        let mut output = String::new();

        for diag in &self.diagnostics {
            output.push_str(&diag.format(source));
            output.push('\n');
        }

        // Summary
        if self.error_count > 0 || self.warning_count > 0 {
            output.push_str(&format!(
                "{} error(s), {} warning(s)\n",
                self.error_count, self.warning_count
            ));
        }

        output
    }

    /// Clear all diagnostics.
    pub fn clear(&mut self) {
        self.diagnostics.clear();
        self.error_count = 0;
        self.warning_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{SourceLocation, SourceSpan};

    #[test]
    fn test_diagnostic_creation() {
        let diag = Diagnostic::error("test error");
        assert_eq!(diag.severity, Severity::Error);
        assert_eq!(diag.message, "test error");
    }

    #[test]
    fn test_diagnostic_with_note() {
        let diag = Diagnostic::error("test").with_note("additional info");
        assert_eq!(diag.notes.len(), 1);
        assert_eq!(diag.notes[0], "additional info");
    }

    #[test]
    fn test_diagnostic_emitter() {
        let mut emitter = DiagnosticEmitter::new();

        emitter.error("error 1");
        emitter.warning("warning 1");
        emitter.info("info 1");

        assert_eq!(emitter.error_count(), 1);
        assert_eq!(emitter.warning_count(), 1);
        assert!(emitter.has_errors());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Fatal);
    }

    #[test]
    fn test_diagnostic_format() {
        let loc = SourceLocation::new(1, 5, 4);
        let span = SourceSpan::from_location(loc);

        let diag = Diagnostic::error("unexpected token")
            .with_span(span)
            .with_note("expected ')'");

        let formatted = diag.format(Some("(foo bar"));
        assert!(formatted.contains("error"));
        assert!(formatted.contains("unexpected token"));
        assert!(formatted.contains("note"));
    }
}
