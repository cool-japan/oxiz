//! Diagnostic mode for problem debugging
//!
//! This module provides comprehensive diagnostic capabilities to help users
//! identify and fix issues in their SMT-LIB2 problems.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Diagnostic result containing issues and suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// List of issues found
    pub issues: Vec<DiagnosticIssue>,
    /// Overall health status
    pub status: DiagnosticStatus,
    /// Summary statistics
    pub summary: DiagnosticSummary,
}

/// Status of the diagnostic check
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticStatus {
    /// No issues found
    Healthy,
    /// Minor issues found (warnings)
    Warning,
    /// Critical issues found (errors)
    Error,
}

/// A single diagnostic issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticIssue {
    /// Severity of the issue
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Description of the issue
    pub description: String,
    /// Location in the input (line number, if available)
    pub location: Option<usize>,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Severity level of an issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational message
    Info,
    /// Warning (may cause problems)
    Warning,
    /// Error (will likely cause failure)
    Error,
}

/// Category of diagnostic issue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    /// Syntax issues
    Syntax,
    /// Symbol declaration/usage issues
    Symbol,
    /// Type-related issues
    Type,
    /// Performance concerns
    Performance,
    /// Complexity issues
    Complexity,
    /// Best practices
    BestPractice,
}

/// Summary statistics from diagnostic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticSummary {
    /// Number of errors found
    pub errors: usize,
    /// Number of warnings found
    pub warnings: usize,
    /// Number of info messages
    pub info: usize,
    /// Total number of assertions checked
    pub assertions_checked: usize,
    /// Total number of symbols checked
    pub symbols_checked: usize,
}

/// Run comprehensive diagnostics on an SMT-LIB2 script
pub fn diagnose_problem(script: &str) -> DiagnosticResult {
    let mut issues = Vec::new();

    // Check for syntax issues
    issues.extend(check_syntax(script));

    // Check for symbol issues
    issues.extend(check_symbols(script));

    // Check for type issues
    issues.extend(check_types(script));

    // Check for performance concerns
    issues.extend(check_performance(script));

    // Check for complexity issues
    issues.extend(check_complexity(script));

    // Check for best practices
    issues.extend(check_best_practices(script));

    // Calculate summary
    let mut errors = 0;
    let mut warnings = 0;
    let mut info = 0;

    for issue in &issues {
        match issue.severity {
            IssueSeverity::Error => errors += 1,
            IssueSeverity::Warning => warnings += 1,
            IssueSeverity::Info => info += 1,
        }
    }

    let status = if errors > 0 {
        DiagnosticStatus::Error
    } else if warnings > 0 {
        DiagnosticStatus::Warning
    } else {
        DiagnosticStatus::Healthy
    };

    let assertions_checked = count_assertions(script);
    let symbols_checked = count_symbols(script);

    DiagnosticResult {
        issues,
        status,
        summary: DiagnosticSummary {
            errors,
            warnings,
            info,
            assertions_checked,
            symbols_checked,
        },
    }
}

/// Check for syntax issues
fn check_syntax(script: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();
    let mut paren_balance = 0;
    let mut line_num = 1;

    for ch in script.chars() {
        if ch == '\n' {
            line_num += 1;
        } else if ch == '(' {
            paren_balance += 1;
        } else if ch == ')' {
            paren_balance -= 1;
            if paren_balance < 0 {
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Error,
                    category: IssueCategory::Syntax,
                    description: "Unmatched closing parenthesis".to_string(),
                    location: Some(line_num),
                    suggestion: Some("Check for extra ')' or missing '('".to_string()),
                });
                paren_balance = 0; // Reset to avoid cascading errors
            }
        }
    }

    if paren_balance > 0 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Error,
            category: IssueCategory::Syntax,
            description: format!("{} unclosed parenthesis(es)", paren_balance),
            location: None,
            suggestion: Some("Check for missing ')' at the end of expressions".to_string()),
        });
    }

    // Check for empty assertions
    if script.contains("(assert)") {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Error,
            category: IssueCategory::Syntax,
            description: "Empty assertion found".to_string(),
            location: None,
            suggestion: Some("Add a boolean expression inside (assert ...)".to_string()),
        });
    }

    issues
}

/// Check for symbol-related issues
fn check_symbols(script: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();
    let mut declared_symbols = HashSet::new();
    let mut used_symbols = HashSet::new();

    // Extract declared symbols
    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("(declare-const") || trimmed.starts_with("(declare-fun") {
            if let Some(symbol) = extract_symbol_from_declaration(trimmed) {
                if declared_symbols.contains(&symbol) {
                    issues.push(DiagnosticIssue {
                        severity: IssueSeverity::Warning,
                        category: IssueCategory::Symbol,
                        description: format!("Symbol '{}' declared multiple times", symbol),
                        location: None,
                        suggestion: Some("Remove duplicate declarations".to_string()),
                    });
                }
                declared_symbols.insert(symbol);
            }
        }
    }

    // Extract used symbols from assertions
    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("(assert") {
            let symbols = extract_symbols_from_expr(trimmed);
            used_symbols.extend(symbols);
        }
    }

    // Check for undeclared symbols
    for symbol in &used_symbols {
        if !declared_symbols.contains(symbol) && !is_builtin_symbol(symbol) {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::Symbol,
                description: format!("Undeclared symbol '{}'", symbol),
                location: None,
                suggestion: Some(format!(
                    "Add declaration: (declare-const {} <type>)",
                    symbol
                )),
            });
        }
    }

    // Check for unused symbols
    for symbol in &declared_symbols {
        if !used_symbols.contains(symbol) {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Info,
                category: IssueCategory::Symbol,
                description: format!("Symbol '{}' declared but never used", symbol),
                location: None,
                suggestion: Some("Consider removing unused declarations".to_string()),
            });
        }
    }

    issues
}

/// Check for type-related issues
fn check_types(script: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();
    let mut symbol_types: HashMap<String, String> = HashMap::new();

    // Extract type information from declarations
    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("(declare-const") {
            if let Some((symbol, ty)) = extract_symbol_and_type(trimmed) {
                symbol_types.insert(symbol, ty);
            }
        }
    }

    // Check for type mismatches in common operations
    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.contains("(= ") {
            // Check that both sides of equality have compatible types
            // This is a simplified check
            if trimmed.contains("true") && trimmed.contains("false") {
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::Type,
                    description: "Comparing boolean constants may be redundant".to_string(),
                    location: None,
                    suggestion: Some("Simplify boolean expressions".to_string()),
                });
            }
        }

        // Check for mixing arithmetic operations with wrong types
        if (trimmed.contains("(+") || trimmed.contains("(-") || trimmed.contains("(*"))
            && (trimmed.contains("true") || trimmed.contains("false"))
        {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Error,
                category: IssueCategory::Type,
                description: "Arithmetic operation on boolean value".to_string(),
                location: None,
                suggestion: Some("Use Int or Real values in arithmetic".to_string()),
            });
        }
    }

    issues
}

/// Check for performance concerns
fn check_performance(script: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();

    // Check for excessive nesting
    let max_nesting = calculate_max_nesting(script);
    if max_nesting > 10 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Warning,
            category: IssueCategory::Performance,
            description: format!("Deep nesting detected (depth: {})", max_nesting),
            location: None,
            suggestion: Some(
                "Consider using let-bindings to reduce nesting and improve readability".to_string(),
            ),
        });
    }

    // Check for very long assertions
    let assertion_count = count_assertions(script);
    if assertion_count > 1000 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Info,
            category: IssueCategory::Performance,
            description: format!("Large number of assertions ({})", assertion_count),
            location: None,
            suggestion: Some(
                "Consider breaking into smaller problems or using incremental solving".to_string(),
            ),
        });
    }

    // Check for quantifiers (can be expensive)
    if script.contains("forall") || script.contains("exists") {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Info,
            category: IssueCategory::Performance,
            description: "Quantifiers detected (may impact performance)".to_string(),
            location: None,
            suggestion: Some("Consider quantifier-free formulations if possible".to_string()),
        });
    }

    issues
}

/// Check for complexity issues
fn check_complexity(script: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();

    let symbol_count = count_symbols(script);

    // High symbol count
    if symbol_count > 100 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Info,
            category: IssueCategory::Complexity,
            description: format!("High symbol count ({})", symbol_count),
            location: None,
            suggestion: Some(
                "Problem may be complex; consider using --auto-tune or --portfolio-mode"
                    .to_string(),
            ),
        });
    }

    // Check for non-linear arithmetic (expensive)
    if script.contains("* ") && (script.contains("Int") || script.contains("Real")) {
        let has_var_mult = check_variable_multiplication(script);
        if has_var_mult {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Warning,
                category: IssueCategory::Complexity,
                description: "Non-linear arithmetic detected (variable multiplication)".to_string(),
                location: None,
                suggestion: Some(
                    "Non-linear arithmetic is undecidable; solving may not terminate".to_string(),
                ),
            });
        }
    }

    issues
}

/// Check for best practices
fn check_best_practices(script: &str) -> Vec<DiagnosticIssue> {
    let mut issues = Vec::new();

    // Check if logic is set
    if !script.contains("(set-logic") {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Info,
            category: IssueCategory::BestPractice,
            description: "Logic not explicitly set".to_string(),
            location: None,
            suggestion: Some("Add (set-logic <LOGIC>) to help the solver optimize".to_string()),
        });
    }

    // Check if check-sat is present
    if !script.contains("(check-sat") {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Info,
            category: IssueCategory::BestPractice,
            description: "No (check-sat) command found".to_string(),
            location: None,
            suggestion: Some("Add (check-sat) to check satisfiability".to_string()),
        });
    }

    // Check for multiple check-sat calls (might want incremental mode)
    let check_sat_count = script.matches("(check-sat").count();
    if check_sat_count > 1 {
        issues.push(DiagnosticIssue {
            severity: IssueSeverity::Info,
            category: IssueCategory::BestPractice,
            description: format!("Multiple check-sat calls ({})", check_sat_count),
            location: None,
            suggestion: Some(
                "Consider using --incremental mode for better performance".to_string(),
            ),
        });
    }

    issues
}

/// Helper function to extract symbol from declaration
fn extract_symbol_from_declaration(decl: &str) -> Option<String> {
    let parts: Vec<&str> = decl.split_whitespace().collect();
    if parts.len() >= 2 {
        Some(parts[1].to_string())
    } else {
        None
    }
}

/// Helper function to extract symbols from an expression
fn extract_symbols_from_expr(expr: &str) -> HashSet<String> {
    let mut symbols = HashSet::new();
    let mut current = String::new();

    for ch in expr.chars() {
        if ch.is_whitespace() || ch == '(' || ch == ')' {
            if !current.is_empty() && !is_operator_or_keyword(&current) {
                symbols.insert(current.clone());
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() && !is_operator_or_keyword(&current) {
        symbols.insert(current);
    }

    symbols
}

/// Check if a symbol is a built-in operator or keyword
fn is_operator_or_keyword(s: &str) -> bool {
    matches!(
        s,
        "assert"
            | "declare-const"
            | "declare-fun"
            | "define-fun"
            | "set-logic"
            | "check-sat"
            | "get-model"
            | "get-proof"
            | "="
            | "+"
            | "-"
            | "*"
            | "/"
            | "and"
            | "or"
            | "not"
            | "=>"
            | "ite"
            | "let"
            | "forall"
            | "exists"
            | "true"
            | "false"
            | "Int"
            | "Bool"
            | "Real"
            | "Array"
            | "BitVec"
            | "<"
            | "<="
            | ">"
            | ">="
    ) || s.parse::<i64>().is_ok()
        || s.parse::<f64>().is_ok()
}

/// Check if a symbol is built-in to SMT-LIB2
fn is_builtin_symbol(s: &str) -> bool {
    is_operator_or_keyword(s)
}

/// Extract symbol and type from declaration
fn extract_symbol_and_type(decl: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = decl.split_whitespace().collect();
    if parts.len() >= 3 {
        let symbol = parts[1].to_string();
        let ty = parts[2].trim_end_matches(')').to_string();
        Some((symbol, ty))
    } else {
        None
    }
}

/// Calculate maximum nesting depth
fn calculate_max_nesting(script: &str) -> usize {
    let mut max_depth: usize = 0;
    let mut current_depth: usize = 0;

    for ch in script.chars() {
        match ch {
            '(' => {
                current_depth += 1;
                max_depth = max_depth.max(current_depth);
            }
            ')' => {
                current_depth = current_depth.saturating_sub(1);
            }
            _ => {}
        }
    }

    max_depth
}

/// Count number of assertions
fn count_assertions(script: &str) -> usize {
    script.matches("(assert").count()
}

/// Count number of unique symbols
fn count_symbols(script: &str) -> usize {
    let mut symbols = HashSet::new();

    for line in script.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("(declare-const") || trimmed.starts_with("(declare-fun") {
            if let Some(symbol) = extract_symbol_from_declaration(trimmed) {
                symbols.insert(symbol);
            }
        }
    }

    symbols.len()
}

/// Check if script contains variable multiplication (non-linear)
fn check_variable_multiplication(script: &str) -> bool {
    // Simple heuristic: look for (* var1 var2) patterns
    // This is not perfect but catches common cases
    script.contains("* ") && !script.contains("* 0") && !script.contains("* 1")
}

/// Format diagnostic result as human-readable string
pub fn format_diagnostic_result(result: &DiagnosticResult) -> String {
    let mut output = String::new();

    output.push_str("=== Diagnostic Report ===\n\n");

    // Status
    let status_str = match result.status {
        DiagnosticStatus::Healthy => "✓ Healthy (no issues found)",
        DiagnosticStatus::Warning => "⚠ Warning (minor issues found)",
        DiagnosticStatus::Error => "✗ Error (critical issues found)",
    };
    output.push_str(&format!("Status: {}\n\n", status_str));

    // Summary
    output.push_str("Summary:\n");
    output.push_str(&format!("  Errors: {}\n", result.summary.errors));
    output.push_str(&format!("  Warnings: {}\n", result.summary.warnings));
    output.push_str(&format!("  Info: {}\n", result.summary.info));
    output.push_str(&format!(
        "  Assertions checked: {}\n",
        result.summary.assertions_checked
    ));
    output.push_str(&format!(
        "  Symbols checked: {}\n\n",
        result.summary.symbols_checked
    ));

    // Issues
    if !result.issues.is_empty() {
        output.push_str("Issues Found:\n\n");

        for (idx, issue) in result.issues.iter().enumerate() {
            let severity_str = match issue.severity {
                IssueSeverity::Error => "ERROR",
                IssueSeverity::Warning => "WARN",
                IssueSeverity::Info => "INFO",
            };

            let category_str = format!("{:?}", issue.category);

            output.push_str(&format!(
                "{}. [{} - {}] {}\n",
                idx + 1,
                severity_str,
                category_str,
                issue.description
            ));

            if let Some(location) = issue.location {
                output.push_str(&format!("   Location: line {}\n", location));
            }

            if let Some(suggestion) = &issue.suggestion {
                output.push_str(&format!("   Suggestion: {}\n", suggestion));
            }

            output.push('\n');
        }
    } else {
        output.push_str("No issues found. Problem looks good!\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_check_balanced() {
        let script = "(declare-const x Int) (assert (= x 5))";
        let issues = check_syntax(script);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_syntax_check_unbalanced() {
        let script = "(declare-const x Int) (assert (= x 5)";
        let issues = check_syntax(script);
        assert!(!issues.is_empty());
        assert_eq!(issues[0].severity, IssueSeverity::Error);
    }

    #[test]
    fn test_symbol_check_undeclared() {
        let script = "(assert (= x 5))";
        let issues = check_symbols(script);
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|i| i.description.contains("Undeclared")));
    }

    #[test]
    fn test_symbol_check_unused() {
        let script = "(declare-const x Int) (declare-const y Int) (assert (= x 5))";
        let issues = check_symbols(script);
        assert!(issues.iter().any(|i| i.description.contains("never used")));
    }

    #[test]
    fn test_full_diagnostic() {
        let script = r#"
            (declare-const x Int)
            (assert (= x 5))
            (check-sat)
        "#;
        let result = diagnose_problem(script);
        assert_eq!(result.status, DiagnosticStatus::Healthy);
    }

    #[test]
    fn test_diagnostic_with_errors() {
        let script = "(assert (= x 5))"; // Undeclared x
        let result = diagnose_problem(script);
        assert_eq!(result.status, DiagnosticStatus::Error);
        assert!(result.summary.errors > 0);
    }
}
