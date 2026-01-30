//! # Error Handling and Recovery Example
//!
//! This example demonstrates OxiZ's comprehensive error handling system.
//! It covers:
//! - Error types and hierarchy
//! - Error context and diagnostics
//! - Error recovery strategies
//! - Batch error processing
//! - User-friendly error messages
//!
//! ## Error Handling Philosophy
//! - Errors are values (Result<T, E>)
//! - Rich context for debugging
//! - Suggestions for fixes when possible
//! - Graceful degradation via recovery
//!
//! ## See Also
//! - [`OxizError`](oxiz_core::error::OxizError) for error types
//! - [`ErrorRecovery`](oxiz_core::error_recovery::ErrorRecovery)
//! - [`Diagnostic`](oxiz_core::diagnostics::Diagnostic)

use oxiz_core::ast::TermManager;
use oxiz_core::diagnostics::{Diagnostic, DiagnosticEmitter, Fix, Severity};
use oxiz_core::error::{OxizError, Result};
use oxiz_core::error_context::{ErrorContext, ResultExt};
use oxiz_core::error_recovery::{ErrorBatch, ErrorRecovery, RecoveryConfig, RecoveryStrategy};
use oxiz_core::error_utils::{
    arity_mismatch, parse_error, sort_mismatch, suggest_fixes, type_error, undefined_symbol,
    validate_arity,
};
use oxiz_core::smtlib::parse_script;

fn main() {
    println!("=== OxiZ Core: Error Handling and Recovery ===\n");

    // ===== Example 1: Basic Error Types =====
    println!("--- Example 1: Basic Error Types ---");

    // Parse error
    let mut tm = TermManager::new();
    let malformed = "(assert (and x))"; // Missing second argument

    match parse_script(malformed, &mut tm) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            println!("Parse error:");
            println!("  Error: {}", e);
            println!("  Type: {:?}", e);
        }
    }

    // Type error (sort mismatch)
    let x_int = tm.mk_var("x", tm.sorts.int_sort);
    let y_bool = tm.mk_var("y", tm.sorts.bool_sort);

    // Attempting to add Int and Bool should fail
    let result: Result<()> = Err(sort_mismatch(
        tm.sorts.int_sort,
        tm.sorts.bool_sort,
        "Cannot apply arithmetic to boolean",
    ));

    match result {
        Ok(_) => println!("\nUnexpected success"),
        Err(e) => {
            println!("\nType error:");
            println!("  {}", e);
        }
    }

    // ===== Example 2: Error Context =====
    println!("\n--- Example 2: Error Context ---");

    fn divide_with_context(a: i32, b: i32) -> Result<i32> {
        if b == 0 {
            Err(OxizError::ArithmeticError("Division by zero".to_string()))
                .context(format!("while dividing {} by {}", a, b))?
        } else {
            Ok(a / b)
        }
    }

    match divide_with_context(10, 0) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            println!("Error with context:");
            println!("  {}", e);
            println!("\nContext chain:");
            let mut current = &e as &dyn std::error::Error;
            let mut level = 0;
            while let Some(source) = current.source() {
                level += 1;
                println!("  Level {}: {}", level, source);
                current = source;
            }
        }
    }

    // ===== Example 3: Arity Validation =====
    println!("\n--- Example 3: Arity Validation ---");

    fn check_binary_op(args: &[i32]) -> Result<()> {
        validate_arity(args.len(), 2)?;
        Ok(())
    }

    match check_binary_op(&[1]) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            println!("Arity error:");
            println!("  {}", e);
        }
    }

    match check_binary_op(&[1, 2, 3]) {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            println!("  {}", e);
        }
    }

    // ===== Example 4: Undefined Symbol Error =====
    println!("\n--- Example 4: Undefined Symbol with Suggestions ---");

    let unknown_var = "temperatur"; // Typo
    let defined_vars = vec!["temperature", "pressure", "volume"];

    let err = undefined_symbol(unknown_var, &defined_vars);
    println!("Undefined symbol error:");
    println!("  {}", err);
    println!("\nSuggested fixes:");
    let suggestions = suggest_fixes(unknown_var, &defined_vars, 2);
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!("  {}: Did you mean '{}'?", i + 1, suggestion);
    }

    // ===== Example 5: Diagnostic System =====
    println!("\n--- Example 5: Diagnostic System ---");

    let mut emitter = DiagnosticEmitter::new();

    // Error diagnostic
    emitter.emit(Diagnostic {
        severity: Severity::Error,
        message: "Variable 'x' is not declared".to_string(),
        location: Some("line 5, column 12".to_string()),
        related: vec![],
        fixes: vec![Fix {
            description: "Declare variable with (declare-const x Int)".to_string(),
            replacement: Some("(declare-const x Int)".to_string()),
        }],
    });

    // Warning diagnostic
    emitter.emit(Diagnostic {
        severity: Severity::Warning,
        message: "Unused variable 'y'".to_string(),
        location: Some("line 8".to_string()),
        related: vec![],
        fixes: vec![],
    });

    // Info diagnostic
    emitter.emit(Diagnostic {
        severity: Severity::Info,
        message: "Using QF_LIA logic".to_string(),
        location: None,
        related: vec![],
        fixes: vec![],
    });

    println!("Diagnostics emitted:");
    println!("  Errors: {}", emitter.error_count());
    println!("  Warnings: {}", emitter.warning_count());
    println!("  Total: {}", emitter.diagnostic_count());

    // ===== Example 6: Error Recovery =====
    println!("\n--- Example 6: Error Recovery ---");

    let config = RecoveryConfig {
        strategy: RecoveryStrategy::SkipAndContinue,
        max_errors: Some(10),
        collect_all_errors: true,
    };

    let mut recovery = ErrorRecovery::new(config);

    // Simulate processing multiple inputs with some errors
    let inputs = vec![
        "(declare-const x Int)",
        "(assert (and x))", // Error: arity mismatch
        "(declare-const y Bool)",
        "(assert (+ y 5))", // Error: type mismatch
        "(check-sat)",
    ];

    println!("Processing inputs with error recovery:");
    for (i, input) in inputs.iter().enumerate() {
        match parse_script(input, &mut tm) {
            Ok(cmds) => {
                println!("  Line {}: OK ({} commands)", i + 1, cmds.len());
                recovery.record_success();
            }
            Err(e) => {
                println!("  Line {}: ERROR - {}", i + 1, e);
                recovery.record_error(e);
            }
        }
    }

    let stats = recovery.stats();
    println!("\nRecovery statistics:");
    println!("  Successful: {}", stats.successful);
    println!("  Failed: {}", stats.failed);
    println!("  Recovered: {}", stats.recovered);
    println!("  Recovery rate: {:.1}%", stats.recovery_rate() * 100.0);

    // ===== Example 7: Batch Error Processing =====
    println!("\n--- Example 7: Batch Error Processing ---");

    let mut batch = ErrorBatch::new();

    // Collect multiple errors
    batch.add(parse_error("Unexpected token ')'", Some(3)));
    batch.add(type_error("Expected Bool, got Int"));
    batch.add(arity_mismatch(2, 1, "function 'and'"));

    println!("Batch contains {} errors:", batch.len());
    for (i, err) in batch.iter().enumerate() {
        println!("  {}: {}", i + 1, err);
    }

    // Process all errors
    if batch.has_errors() {
        println!("\nProcessing batch errors:");
        println!("  Total: {}", batch.len());
        println!("  Fatal: {}", batch.fatal_count());
        println!("  Can continue: {}", !batch.has_fatal());
    }

    // ===== Example 8: Chained Error Context =====
    println!("\n--- Example 8: Chained Error Context ---");

    fn parse_and_validate(input: &str) -> Result<()> {
        let mut tm2 = TermManager::new();

        parse_script(input, &mut tm2).context("parsing SMT-LIB script")?;

        // Additional validation
        validate_logic(&tm2).context("validating logic")?;

        Ok(())
    }

    fn validate_logic(_tm: &TermManager) -> Result<()> {
        Err(OxizError::UnsupportedLogic("QF_AUFBV".to_string()))
    }

    match parse_and_validate("(set-logic QF_AUFBV)") {
        Ok(_) => println!("Unexpected success"),
        Err(e) => {
            println!("Chained error:");
            println!("  {}", e);
            println!("\nFull error chain shows the error occurred during:");
            println!("  1. Validating logic");
            println!("  2. While parsing SMT-LIB script");
        }
    }

    // ===== Example 9: Resource Limit Errors =====
    println!("\n--- Example 9: Resource Limit Errors ---");

    use oxiz_core::resource::{LimitStatus, ResourceManager};

    let mut resources = ResourceManager::new();
    resources.set_time_limit(std::time::Duration::from_secs(1));
    resources.set_memory_limit(100 * 1024 * 1024); // 100 MB

    // Simulate timeout
    std::thread::sleep(std::time::Duration::from_millis(1100));

    match resources.check_limits() {
        Ok(_) => println!("Within limits"),
        Err(status) => {
            println!("Resource limit exceeded:");
            match status {
                LimitStatus::Timeout => println!("  Timeout (> 1 second)"),
                LimitStatus::MemoryLimit => println!("  Memory limit (> 100 MB)"),
                LimitStatus::Ok => println!("  OK (unexpected)"),
            }
        }
    }

    // ===== Example 10: Custom Error Types =====
    println!("\n--- Example 10: Custom Error Types ---");

    #[derive(Debug)]
    enum CustomError {
        InvalidConfiguration(String),
        InternalInconsistency(String),
    }

    impl std::fmt::Display for CustomError {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                CustomError::InvalidConfiguration(msg) => {
                    write!(f, "Invalid configuration: {}", msg)
                }
                CustomError::InternalInconsistency(msg) => {
                    write!(f, "Internal inconsistency: {}", msg)
                }
            }
        }
    }

    impl std::error::Error for CustomError {}

    fn check_config(value: i32) -> std::result::Result<(), CustomError> {
        if value < 0 {
            Err(CustomError::InvalidConfiguration(
                "Value must be non-negative".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    match check_config(-5) {
        Ok(_) => println!("Config OK"),
        Err(e) => {
            println!("Custom error:");
            println!("  {}", e);
            println!("  Type: {:?}", e);
        }
    }

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("  1. Use Result<T, E> for all fallible operations");
    println!("  2. Add context to errors for better debugging");
    println!("  3. Provide suggestions for common mistakes");
    println!("  4. Diagnostics help users understand and fix errors");
    println!("  5. Error recovery enables batch processing");
    println!("  6. Resource limits prevent unbounded execution");
}
