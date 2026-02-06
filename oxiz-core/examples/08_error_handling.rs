//! # Error Handling Example
//!
//! This example demonstrates OxiZ's error handling system.
//! It covers:
//! - Error types and variants
//! - Result handling
//! - Parse errors
//! - Diagnostic messages
//!
//! ## Error Handling Philosophy
//! - Errors are values (Result<T, E>)
//! - Rich context for debugging
//! - Graceful degradation via recovery
//!
//! ## See Also
//! - [`OxizError`](oxiz_core::error::OxizError) for error types
//! - [`Diagnostic`](oxiz_core::diagnostics::Diagnostic)

use oxiz_core::ast::TermManager;
use oxiz_core::config::ResourceLimits;
use oxiz_core::diagnostics::Diagnostic;
use oxiz_core::error::{OxizError, Result, SourceLocation, SourceSpan};
use oxiz_core::resource::{LimitStatus, ResourceManager};
use oxiz_core::smtlib::parse_script;
use oxiz_core::statistics::Statistics;
use std::time::Duration;

fn main() {
    println!("=== OxiZ Core: Error Handling ===\n");

    // ===== Example 1: Basic Error Types =====
    println!("--- Example 1: OxizError Variants ---");

    // Create different error types
    let parse_err = OxizError::ParseError {
        position: 10,
        message: "Unexpected token ')'".to_string(),
    };
    let type_err = OxizError::SortMismatchSimple {
        expected: "Int".to_string(),
        found: "Bool".to_string(),
    };
    let unsupported_err = OxizError::Unsupported("Non-linear arithmetic".to_string());
    let internal_err = OxizError::Internal("Unexpected state".to_string());

    println!("Parse error: {}", parse_err);
    println!("Type error: {}", type_err);
    println!("Unsupported: {}", unsupported_err);
    println!("Internal error: {}", internal_err);

    // ===== Example 2: Parse Error Handling =====
    println!("\n--- Example 2: Parse Error Handling ---");

    let mut tm = TermManager::new();
    let malformed = "(assert (and x))"; // Missing second argument

    match parse_script(malformed, &mut tm) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            println!("Parse error detected:");
            println!("  Error: {}", e);
        }
    }

    // ===== Example 3: Result Type Usage =====
    println!("\n--- Example 3: Result Type Usage ---");

    fn safe_divide(a: i32, b: i32) -> Result<i32> {
        if b == 0 {
            Err(OxizError::Internal("Division by zero".to_string()))
        } else {
            Ok(a / b)
        }
    }

    match safe_divide(10, 2) {
        Ok(result) => println!("10 / 2 = {}", result),
        Err(e) => println!("Error: {}", e),
    }

    match safe_divide(10, 0) {
        Ok(result) => println!("10 / 0 = {}", result),
        Err(e) => println!("10 / 0 = Error: {}", e),
    }

    // ===== Example 4: Diagnostic System =====
    println!("\n--- Example 4: Diagnostic System ---");

    // Error diagnostic using builder pattern
    let error_diag = Diagnostic::error("Variable 'x' is not declared")
        .with_note("Make sure to declare variables before use");

    // Warning diagnostic
    let warning_diag = Diagnostic::warning("Unused variable 'y'");

    // Info diagnostic
    let info_diag = Diagnostic::info("Using QF_LIA logic");

    println!("Diagnostic examples:");
    println!("  [{:?}] {}", error_diag.severity, error_diag.message);
    for note in &error_diag.notes {
        println!("    note: {}", note);
    }
    println!("  [{:?}] {}", warning_diag.severity, warning_diag.message);
    println!("  [{:?}] {}", info_diag.severity, info_diag.message);

    // ===== Example 5: Resource Limit Handling =====
    println!("\n--- Example 5: Resource Limit Handling ---");

    let limits = ResourceLimits {
        time_limit: Some(Duration::from_millis(100)),
        decision_limit: Some(1000),
        conflict_limit: Some(500),
        memory_limit: Some(100 * 1024 * 1024), // 100 MB
    };

    let resources = ResourceManager::new(limits);
    let stats = Statistics::new();

    println!("Resource limits configured:");
    println!("  Time: 100 ms");
    println!("  Decisions: 1000");
    println!("  Conflicts: 500");
    println!("  Memory: 100 MB");

    let status = resources.check_limits(&stats);
    match status {
        LimitStatus::Ok => println!("\nStatus: Within limits"),
        LimitStatus::TimeExceeded => println!("\nStatus: Time limit exceeded"),
        LimitStatus::DecisionExceeded => println!("\nStatus: Decision limit exceeded"),
        LimitStatus::ConflictExceeded => println!("\nStatus: Conflict limit exceeded"),
        LimitStatus::MemoryExceeded => println!("\nStatus: Memory limit exceeded"),
    }

    // ===== Example 6: Error Propagation =====
    println!("\n--- Example 6: Error Propagation ---");

    fn inner_operation() -> Result<i32> {
        Err(OxizError::Internal("Inner failure".to_string()))
    }

    fn outer_operation() -> Result<i32> {
        let value = inner_operation()?; // ? propagates the error
        Ok(value * 2)
    }

    match outer_operation() {
        Ok(v) => println!("Success: {}", v),
        Err(e) => {
            println!("Error propagated from inner operation:");
            println!("  {}", e);
        }
    }

    // ===== Example 7: Valid Parsing =====
    println!("\n--- Example 7: Successful Parsing ---");

    let valid_input = r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (>= x 0))
        (check-sat)
    "#;

    let mut tm2 = TermManager::new();
    match parse_script(valid_input, &mut tm2) {
        Ok(commands) => {
            println!("Parsed {} commands successfully:", commands.len());
            for (i, cmd) in commands.iter().enumerate() {
                println!("  {}: {:?}", i + 1, cmd);
            }
        }
        Err(e) => {
            println!("Parse error: {}", e);
        }
    }

    // ===== Example 8: Custom Error Handling =====
    println!("\n--- Example 8: Custom Error Handling ---");

    #[derive(Debug)]
    enum CustomError {
        InvalidConfiguration(String),
        OutOfRange(String),
    }

    impl std::fmt::Display for CustomError {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                CustomError::InvalidConfiguration(msg) => {
                    write!(f, "Invalid configuration: {}", msg)
                }
                CustomError::OutOfRange(msg) => {
                    write!(f, "Out of range: {}", msg)
                }
            }
        }
    }

    impl std::error::Error for CustomError {}

    fn check_value(value: i32) -> std::result::Result<(), CustomError> {
        if value < 0 {
            Err(CustomError::InvalidConfiguration(
                "Value must be non-negative".to_string(),
            ))
        } else if value > 100 {
            Err(CustomError::OutOfRange("Value must be <= 100".to_string()))
        } else {
            Ok(())
        }
    }

    match check_value(-5) {
        Ok(()) => println!("Value OK"),
        Err(e) => println!("Error: {}", e),
    }

    match check_value(150) {
        Ok(()) => println!("Value OK"),
        Err(e) => println!("Error: {}", e),
    }

    match check_value(50) {
        Ok(()) => println!("Value 50 is OK"),
        Err(e) => println!("Error: {}", e),
    }

    // ===== Example 9: Source Location =====
    println!("\n--- Example 9: Source Location ---");

    let loc1 = SourceLocation::new(5, 12, 50);
    let loc2 = SourceLocation::new(5, 20, 58);
    let span = SourceSpan::new(loc1, loc2);

    println!("Source location: {}", loc1);
    println!("Source span: {}", span);

    // ===== Example 10: Statistics Tracking =====
    println!("\n--- Example 10: Statistics for Debugging ---");

    let mut stats = Statistics::new();

    // Simulate solver activity
    for _ in 0..50 {
        stats.inc_decisions();
        stats.inc_propagations();
        if stats.decisions.is_multiple_of(5) {
            stats.inc_conflicts();
        }
    }

    println!("Statistics after simulation:");
    println!("  Decisions: {}", stats.decisions);
    println!("  Propagations: {}", stats.propagations);
    println!("  Conflicts: {}", stats.conflicts);
    println!(
        "  Conflict ratio: {:.2}%",
        (stats.conflicts as f64 / stats.decisions as f64) * 100.0
    );

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Use Result<T, E> for all fallible operations");
    println!("  2. OxizError provides specific error variants");
    println!("  3. Diagnostics help users understand errors");
    println!("  4. Resource limits prevent unbounded execution");
    println!("  5. Error propagation with ? operator is idiomatic");
    println!("  6. Custom errors implement std::error::Error trait");
}
