//! Output formatting and display utilities for OxiZ CLI

use owo_colors::{OwoColorize, Stream};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// Import from crate modules
use crate::analysis::{ComplexityAnalysis, ProblemClassification};
use crate::dimacs;
use crate::{Args, OutputFormat};

/// Profiling data for a single operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingData {
    /// Operation name
    pub operation: String,
    /// Execution time in microseconds
    pub duration_us: u128,
    /// Memory delta in bytes (if available)
    pub memory_delta_bytes: i64,
}

/// Statistics for solver execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverStats {
    /// Total execution time in milliseconds
    pub execution_time_ms: u128,
    /// Number of files processed
    pub files_processed: usize,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Peak memory usage in bytes (if profiling enabled)
    pub peak_memory_bytes: u64,
    /// Number of successful results
    pub success_count: usize,
    /// Number of errors
    pub error_count: usize,
    /// Profiling data (if profiling enabled)
    pub profiling_data: Option<Vec<ProfilingData>>,
    /// Average time per file in milliseconds
    pub avg_time_per_file_ms: u128,
    /// Min time per file in milliseconds
    pub min_time_ms: u128,
    /// Max time per file in milliseconds
    pub max_time_ms: u128,
    /// SAT solver decisions
    pub decisions: u64,
    /// SAT solver propagations
    pub propagations: u64,
    /// SAT solver conflicts
    pub conflicts: u64,
    /// SAT solver restarts
    pub restarts: u64,
}

/// Result wrapper with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverResult {
    /// File path (if applicable)
    pub file: Option<String>,
    /// Result output
    pub result: String,
    /// Error message (if any)
    pub error: Option<String>,
    /// Execution time in milliseconds
    pub time_ms: u128,
}

/// Print practical usage examples
pub(crate) fn print_examples() {
    println!("OxiZ SMT Solver - Practical Usage Examples");
    println!("==========================================\n");

    println!("BASIC USAGE");
    println!("-----------");
    println!("  # Solve a single SMT-LIB2 file");
    println!("  oxiz problem.smt2\n");
    println!("  # Solve from stdin");
    println!("  echo '(check-sat)' | oxiz\n");
    println!("  # Multiple files");
    println!("  oxiz file1.smt2 file2.smt2 file3.smt2\n");
    println!("  # Glob patterns");
    println!("  oxiz benchmarks/*.smt2\n");

    println!("INTERACTIVE MODE");
    println!("----------------");
    println!("  # Start interactive REPL");
    println!("  oxiz --interactive\n");
    println!("  # In REPL, you can enter SMT-LIB2 commands:");
    println!("  > (declare-const x Int)");
    println!("  > (assert (> x 0))");
    println!("  > (check-sat)\n");

    println!("QUERY ANALYSIS");
    println!("--------------");
    println!("  # Analyze problem complexity without solving");
    println!("  oxiz problem.smt2 --analyze\n");
    println!("  # Get JSON analysis for programmatic use");
    println!("  oxiz problem.smt2 --analyze --format json\n");
    println!("  # Classify problem and get solver recommendations");
    println!("  oxiz problem.smt2 --classify\n");
    println!("  # Automatically tune solver based on problem");
    println!("  oxiz problem.smt2 --auto-tune\n");

    println!("OUTPUT FORMATS");
    println!("--------------");
    println!("  # JSON output");
    println!("  oxiz problem.smt2 --format json\n");
    println!("  # YAML output");
    println!("  oxiz problem.smt2 --format yaml\n");
    println!("  # Save to file");
    println!("  oxiz problem.smt2 -o result.txt\n");

    println!("PERFORMANCE & OPTIMIZATION");
    println!("--------------------------");
    println!("  # Parallel solving with 8 threads");
    println!("  oxiz benchmarks/*.smt2 --parallel --threads 8\n");
    println!("  # Use fast preset for quick results");
    println!("  oxiz problem.smt2 --preset fast\n");
    println!("  # Use thorough preset for maximum completeness");
    println!("  oxiz problem.smt2 --preset thorough\n");
    println!("  # Enable result caching");
    println!("  oxiz problem.smt2 --cache\n");
    println!("  # Set timeout (300 seconds)");
    println!("  oxiz problem.smt2 --timeout 300\n");

    println!("RESOURCE LIMITS");
    println!("---------------");
    println!("  # Limit memory to 2GB");
    println!("  oxiz problem.smt2 --memory-limit 2048\n");
    println!("  # Limit conflicts");
    println!("  oxiz problem.smt2 --conflict-limit 10000\n");
    println!("  # Limit decisions");
    println!("  oxiz problem.smt2 --decision-limit 50000\n");

    println!("ADVANCED FEATURES");
    println!("-----------------");
    println!("  # Enable preprocessing and simplification");
    println!("  oxiz problem.smt2 --simplify\n");
    println!("  # Enumerate all models (up to 10)");
    println!("  oxiz problem.smt2 --enumerate-models --max-models 10\n");
    println!("  # Validate proof for UNSAT results");
    println!("  oxiz problem.smt2 --validate-proof\n");
    println!("  # Minimize satisfying model");
    println!("  oxiz problem.smt2 --minimize-model\n");

    println!("UNSAT CORES & PROOFS");
    println!("--------------------");
    println!("  # Extract UNSAT core (minimal unsatisfiable subset)");
    println!("  oxiz problem.smt2 --unsat-core\n");
    println!("  # Minimize UNSAT core");
    println!("  oxiz problem.smt2 --unsat-core --minimize-core\n");
    println!("  # Generate proof tree visualization (DOT format)");
    println!("  oxiz problem.smt2 --proof-dot proof.dot\n");
    println!("  # Convert DOT to PNG with GraphViz");
    println!("  oxiz problem.smt2 --proof-dot proof.dot && dot -Tpng proof.dot -o proof.png\n");
    println!("  # Validate model against assertions");
    println!("  oxiz problem.smt2 --validate-model\n");

    println!("INCREMENTAL SOLVING");
    println!("-------------------");
    println!("  # Enable incremental mode (supports push/pop)");
    println!("  oxiz --incremental --interactive\n");
    println!("  # In incremental mode:");
    println!("  > (assert (> x 0))");
    println!("  > (push 1)");
    println!("  > (assert (< x 0))");
    println!("  > (check-sat)  ; unsat");
    println!("  > (pop 1)");
    println!("  > (check-sat)  ; sat\n");

    println!("STATISTICS & PROFILING");
    println!("----------------------");
    println!("  # Show statistics");
    println!("  oxiz problem.smt2 --stats\n");
    println!("  # Show timing information");
    println!("  oxiz problem.smt2 --time\n");
    println!("  # Show memory usage");
    println!("  oxiz problem.smt2 --memory\n");
    println!("  # Enable profiling with detailed metrics");
    println!("  oxiz problem.smt2 --profile\n");
    println!("  # Export statistics to CSV");
    println!("  oxiz problem.smt2 --export-stats results.csv\n");

    println!("DIMACS CNF FORMAT");
    println!("-----------------");
    println!("  # Solve DIMACS CNF file");
    println!("  oxiz problem.cnf --dimacs\n");
    println!("  # Convert SMT-LIB2 to DIMACS output");
    println!("  oxiz problem.smt2 --dimacs-output\n");

    println!("VALIDATION & FORMATTING");
    println!("-----------------------");
    println!("  # Validate syntax only (no solving)");
    println!("  oxiz problem.smt2 --validate-only\n");
    println!("  # Format and pretty-print SMT-LIB2");
    println!("  oxiz problem.smt2 --format-smtlib\n");
    println!("  # Format with custom indentation (4 spaces)");
    println!("  oxiz problem.smt2 --format-smtlib --indent-width 4\n");

    println!("WATCH MODE & AUTOMATION");
    println!("-----------------------");
    println!("  # Watch file for changes and re-solve");
    println!("  oxiz problem.smt2 --watch\n");
    println!("  # SMT-COMP compatible output");
    println!("  oxiz problem.smt2 --smtcomp\n");

    println!("IDE INTEGRATION");
    println!("---------------");
    println!("  # Run as LSP server for IDE integration");
    println!("  oxiz --lsp\n");
    println!("  # Generate shell completions (bash)");
    println!("  oxiz --completions bash > ~/.bash_completion.d/oxiz\n");

    println!("COMBINED EXAMPLES");
    println!("-----------------");
    println!("  # Analyze, then solve with auto-tuning and statistics");
    println!("  oxiz problem.smt2 --classify && oxiz problem.smt2 --auto-tune --stats\n");
    println!("  # Process directory recursively with parallel solving and caching");
    println!("  oxiz benchmarks/ --recursive --parallel --threads 16 --cache\n");
    println!("  # Quiet mode with JSON output for scripting");
    println!("  oxiz problem.smt2 --quiet --format json\n");

    println!("\nFor more information, see: oxiz --help");
}

/// Format DIMACS output from SMT-LIB2 result
pub(crate) fn format_dimacs_output(smtlib_output: &str, num_vars: usize) -> String {
    let mut output = String::new();

    if smtlib_output.contains("unsat") {
        output.push_str("s UNSATISFIABLE\n");
    } else if smtlib_output.contains("sat") {
        output.push_str("s SATISFIABLE\n");

        // Extract model if present
        if smtlib_output.contains("define-fun") {
            let assignment = dimacs::DimacsCnf::model_from_smtlib2(smtlib_output, num_vars);
            output.push_str("v ");
            for lit in assignment {
                output.push_str(&format!("{} ", lit));
            }
            output.push_str("0\n");
        }
    } else {
        output.push_str("s UNKNOWN\n");
    }

    output
}

/// Format and pretty-print SMT-LIB2 script
pub(crate) fn format_smtlib_script(script: &str, indent_width: usize) -> String {
    let mut result = String::new();
    let mut indent_level = 0;
    let mut chars = script.chars().peekable();
    let mut in_string = false;
    let mut in_comment = false;
    let mut at_line_start = true;
    let mut pending_space = false;

    while let Some(ch) = chars.next() {
        match ch {
            '\n' => {
                in_comment = false;
                result.push('\n');
                at_line_start = true;
                pending_space = false;
            }
            ';' if !in_string => {
                if pending_space {
                    result.push(' ');
                    pending_space = false;
                }
                in_comment = true;
                result.push(ch);
            }
            '"' if !in_comment => {
                if pending_space {
                    result.push(' ');
                    pending_space = false;
                }
                in_string = !in_string;
                result.push(ch);
            }
            '(' if !in_string && !in_comment => {
                if at_line_start {
                    result.push_str(&" ".repeat(indent_level * indent_width));
                    at_line_start = false;
                } else if pending_space {
                    result.push(' ');
                }
                result.push(ch);
                pending_space = false;

                // Look ahead to see if next non-whitespace is a keyword
                let ahead: String = chars.clone().take(20).collect();
                let ahead_trimmed = ahead.trim_start();

                // Add newline and indent for major commands
                if (ahead_trimmed.starts_with("assert")
                    || ahead_trimmed.starts_with("declare-")
                    || ahead_trimmed.starts_with("define-")
                    || ahead_trimmed.starts_with("set-")
                    || ahead_trimmed.starts_with("check-sat")
                    || ahead_trimmed.starts_with("get-"))
                    && indent_level > 0
                {
                    result.push('\n');
                    result.push_str(&" ".repeat(indent_level * indent_width));
                }

                indent_level += 1;
            }
            ')' if !in_string && !in_comment => {
                indent_level = indent_level.saturating_sub(1);
                result.push(ch);
                pending_space = false;
            }
            ' ' | '\t' | '\r' if !in_string && !in_comment => {
                if !at_line_start && !pending_space {
                    pending_space = true;
                }
            }
            _ if in_comment || in_string => {
                result.push(ch);
            }
            _ => {
                if at_line_start {
                    result.push_str(&" ".repeat(indent_level * indent_width));
                    at_line_start = false;
                } else if pending_space {
                    result.push(' ');
                }
                result.push(ch);
                pending_space = false;
            }
        }
    }

    result
}

/// Format complexity analysis results
pub(crate) fn format_analysis(analysis: &ComplexityAnalysis, args: &Args) -> String {
    if args.format == OutputFormat::Json {
        serde_json::to_string_pretty(analysis).unwrap_or_else(|_| "{}".to_string())
    } else if args.format == OutputFormat::Yaml {
        serde_yaml::to_string(analysis).unwrap_or_else(|_| "".to_string())
    } else {
        let mut result = String::new();
        result.push_str("=== Query Complexity Analysis ===\n\n");
        result.push_str(&format!("Declarations: {}\n", analysis.declarations));
        result.push_str(&format!("Assertions: {}\n", analysis.assertions));
        result.push_str(&format!("Total Commands: {}\n", analysis.commands));
        result.push_str(&format!(
            "Max Nesting Depth: {}\n",
            analysis.max_nesting_depth
        ));
        result.push_str(&format!(
            "Avg Nesting Depth: {:.2}\n",
            analysis.avg_nesting_depth
        ));
        result.push_str(&format!("Quantifiers: {}\n", analysis.quantifiers));
        result.push_str(&format!("Theories: {}\n", analysis.theories.join(", ")));
        result.push_str(&format!(
            "\nEstimated Difficulty: {}\n",
            analysis.estimated_difficulty
        ));
        result.push_str(&format!(
            "Recommended Strategy: {}\n",
            analysis.recommended_strategy
        ));
        result.push_str(&format!(
            "Recommended Timeout: {}s\n",
            analysis.recommended_timeout
        ));

        if !analysis.operators.is_empty() {
            result.push_str("\nOperator Usage:\n");
            let mut ops: Vec<_> = analysis.operators.iter().collect();
            ops.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            for (op, count) in ops.iter().take(10) {
                result.push_str(&format!("  {}: {}\n", op, count));
            }
        }

        result
    }
}

/// Format problem classification results
pub(crate) fn format_classification(
    classification: &ProblemClassification,
    analysis: &ComplexityAnalysis,
    args: &Args,
) -> String {
    if args.format == OutputFormat::Json {
        let combined = serde_json::json!({
            "classification": classification,
            "complexity": analysis
        });
        serde_json::to_string_pretty(&combined).unwrap_or_else(|_| "{}".to_string())
    } else if args.format == OutputFormat::Yaml {
        serde_yaml::to_string(&(classification, analysis)).unwrap_or_else(|_| "".to_string())
    } else {
        let mut result = String::new();
        result.push_str("=== Problem Classification ===\n\n");
        result.push_str(&format!("Logic: {}\n", classification.logic));
        result.push_str(&format!(
            "Primary Theory: {}\n",
            classification.primary_theory
        ));
        result.push_str(&format!(
            "Quantifier-Free: {}\n",
            classification.is_quantifier_free
        ));
        result.push_str(&format!(
            "Complexity Class: {}\n",
            classification.complexity_class
        ));
        result.push_str(&format!(
            "\nEstimated Difficulty: {}\n",
            analysis.estimated_difficulty
        ));
        result.push_str(&format!("Assertions: {}\n", analysis.assertions));
        result.push_str(&format!(
            "Max Nesting Depth: {}\n",
            analysis.max_nesting_depth
        ));

        result.push_str("\n=== Solver Recommendations ===\n\n");
        for (idx, rec) in classification.solver_recommendations.iter().enumerate() {
            result.push_str(&format!("{}. {}\n", idx + 1, rec));
        }

        result
    }
}

/// Pretty-print a model with syntax highlighting
pub(crate) fn pretty_print_model(model_str: &str, args: &Args) -> String {
    // Check if this is a model response
    if !model_str.trim().starts_with('(') {
        return model_str.to_string();
    }

    let mut result = String::new();
    let mut indent_level = 0;
    let mut chars = model_str.chars().peekable();
    let mut in_string = false;
    let mut current_word = String::new();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                in_string = !in_string;
                current_word.push(ch);
            }
            '(' if !in_string => {
                if !current_word.is_empty() {
                    result.push_str(&current_word);
                    current_word.clear();
                }

                // Check if this is a define-fun
                let ahead: String = chars.clone().take(15).collect();
                if ahead.starts_with("define-fun") && indent_level > 0 {
                    result.push('\n');
                    result.push_str(&"  ".repeat(indent_level));
                }

                result.push(ch);
                indent_level += 1;
            }
            ')' if !in_string => {
                if !current_word.is_empty() {
                    result.push_str(&current_word);
                    current_word.clear();
                }
                indent_level = indent_level.saturating_sub(1);
                result.push(ch);
            }
            ' ' | '\t' | '\n' | '\r' if !in_string => {
                if !current_word.is_empty() {
                    // Colorize keywords if colors are enabled
                    let colored = if !args.no_color {
                        match current_word.as_str() {
                            "define-fun" => current_word
                                .if_supports_color(Stream::Stdout, |t| t.blue())
                                .to_string(),
                            "Int" | "Bool" | "Real" | "String" | "Array" | "BitVec" => current_word
                                .if_supports_color(Stream::Stdout, |t| t.cyan())
                                .to_string(),
                            "true" | "false" => current_word
                                .if_supports_color(Stream::Stdout, |t| t.yellow())
                                .to_string(),
                            _ if current_word.chars().all(|c| c.is_ascii_digit() || c == '-') => {
                                current_word
                                    .if_supports_color(Stream::Stdout, |t| t.yellow())
                                    .to_string()
                            }
                            _ => current_word.clone(),
                        }
                    } else {
                        current_word.clone()
                    };
                    result.push_str(&colored);
                    current_word.clear();
                }
                result.push(ch);
            }
            _ => {
                current_word.push(ch);
            }
        }
    }

    if !current_word.is_empty() {
        result.push_str(&current_word);
    }

    result
}

/// Pretty-print a proof with syntax highlighting
pub(crate) fn pretty_print_proof(proof_str: &str, args: &Args) -> String {
    // Check if this is a proof response
    if !proof_str.trim().starts_with('(') {
        return proof_str.to_string();
    }

    let mut result = String::new();
    let mut indent_level = 0;
    let mut chars = proof_str.chars().peekable();
    let mut in_string = false;
    let mut current_word = String::new();
    let mut after_newline = true;

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                in_string = !in_string;
                current_word.push(ch);
            }
            '(' if !in_string => {
                if !current_word.is_empty() {
                    result.push_str(&colorize_proof_word(&current_word, args));
                    current_word.clear();
                }

                // Check if this is a proof step keyword
                let ahead: String = chars.clone().take(20).collect();
                if (ahead.starts_with("step")
                    || ahead.starts_with("assume")
                    || ahead.starts_with("cl"))
                    && !after_newline
                    && indent_level > 0
                {
                    result.push('\n');
                    result.push_str(&"  ".repeat(indent_level));
                }

                result.push(ch);
                indent_level += 1;
                after_newline = false;
            }
            ')' if !in_string => {
                if !current_word.is_empty() {
                    result.push_str(&colorize_proof_word(&current_word, args));
                    current_word.clear();
                }
                indent_level = indent_level.saturating_sub(1);
                result.push(ch);
                after_newline = false;
            }
            ' ' | '\t' if !in_string => {
                if !current_word.is_empty() {
                    result.push_str(&colorize_proof_word(&current_word, args));
                    current_word.clear();
                }
                result.push(ch);
                after_newline = false;
            }
            '\n' | '\r' if !in_string => {
                if !current_word.is_empty() {
                    result.push_str(&colorize_proof_word(&current_word, args));
                    current_word.clear();
                }
                result.push(ch);
                after_newline = true;
            }
            _ => {
                current_word.push(ch);
            }
        }
    }

    if !current_word.is_empty() {
        result.push_str(&colorize_proof_word(&current_word, args));
    }

    result
}

/// Colorize proof keywords (private helper)
fn colorize_proof_word(word: &str, args: &Args) -> String {
    if args.no_color {
        return word.to_string();
    }

    match word {
        "step" | "assume" | "cl" | "proof" | "lemma" | "axiom" => word
            .if_supports_color(Stream::Stdout, |t| t.blue())
            .to_string(),
        "resolution" | "mp" | "and" | "or" | "not" | "implies" | "iff" => word
            .if_supports_color(Stream::Stdout, |t| t.magenta())
            .to_string(),
        "true" | "false" => word
            .if_supports_color(Stream::Stdout, |t| t.yellow())
            .to_string(),
        _ if word.starts_with(':') => word
            .if_supports_color(Stream::Stdout, |t| t.cyan())
            .to_string(),
        _ if word.chars().all(|c| c.is_ascii_digit() || c == '-') => word
            .if_supports_color(Stream::Stdout, |t| t.yellow())
            .to_string(),
        _ => word.to_string(),
    }
}

/// Output results in the requested format
pub(crate) fn output_results(results: &[SolverResult], args: &Args, stats: &SolverStats) {
    match args.format {
        OutputFormat::Smtlib => {
            for result in results {
                if let Some(path) = args.output.as_ref() {
                    if let Err(e) = fs::write(path, &result.result) {
                        eprintln_colored(args, &format!("Error writing output: {}", e));
                    }
                } else if !result.result.is_empty() {
                    println!("{}", result.result);
                }
            }
        }
        OutputFormat::Json => {
            let output = serde_json::json!({
                "results": results,
                "statistics": stats,
            });
            let json = serde_json::to_string_pretty(&output)
                .unwrap_or_else(|e| format!("{{\"error\": \"Failed to serialize JSON: {}\"}}", e));

            if let Some(path) = args.output.as_ref() {
                if let Err(e) = fs::write(path, json) {
                    eprintln_colored(args, &format!("Error writing output: {}", e));
                }
            } else {
                println!("{}", json);
            }
        }
        OutputFormat::Yaml => {
            #[derive(Serialize)]
            struct YamlOutput<'a> {
                results: &'a [SolverResult],
                statistics: &'a SolverStats,
            }

            let output = YamlOutput {
                results,
                statistics: stats,
            };

            let yaml = serde_yaml::to_string(&output)
                .unwrap_or_else(|e| format!("error: \"Failed to serialize YAML: {}\"", e));

            if let Some(path) = args.output.as_ref() {
                if let Err(e) = fs::write(path, yaml) {
                    eprintln_colored(args, &format!("Error writing output: {}", e));
                }
            } else {
                println!("{}", yaml);
            }
        }
    }
}

/// Export statistics to file (CSV or JSON)
pub(crate) fn export_statistics(
    stats: &SolverStats,
    path: &Path,
    args: &Args,
) -> Result<(), String> {
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    match extension {
        "csv" => {
            // Export as CSV
            let mut csv_content = String::new();
            csv_content.push_str("metric,value\n");
            csv_content.push_str(&format!("execution_time_ms,{}\n", stats.execution_time_ms));
            csv_content.push_str(&format!("files_processed,{}\n", stats.files_processed));
            csv_content.push_str(&format!("memory_bytes,{}\n", stats.memory_bytes));
            csv_content.push_str(&format!("peak_memory_bytes,{}\n", stats.peak_memory_bytes));
            csv_content.push_str(&format!("success_count,{}\n", stats.success_count));
            csv_content.push_str(&format!("error_count,{}\n", stats.error_count));
            csv_content.push_str(&format!(
                "avg_time_per_file_ms,{}\n",
                stats.avg_time_per_file_ms
            ));
            csv_content.push_str(&format!("min_time_ms,{}\n", stats.min_time_ms));
            csv_content.push_str(&format!("max_time_ms,{}\n", stats.max_time_ms));
            csv_content.push_str(&format!("decisions,{}\n", stats.decisions));
            csv_content.push_str(&format!("propagations,{}\n", stats.propagations));
            csv_content.push_str(&format!("conflicts,{}\n", stats.conflicts));
            csv_content.push_str(&format!("restarts,{}\n", stats.restarts));

            fs::write(path, csv_content).map_err(|e| format!("Failed to write CSV: {}", e))?;
        }
        "json" => {
            // Export as JSON
            let json = serde_json::to_string_pretty(&stats)
                .map_err(|e| format!("Failed to serialize JSON: {}", e))?;
            fs::write(path, json).map_err(|e| format!("Failed to write JSON: {}", e))?;
        }
        _ => {
            // Default to JSON if extension is not recognized
            if !args.no_color {
                eprintln!(
                    "{}",
                    format!(
                        "Warning: Unknown extension '{}', defaulting to JSON format",
                        extension
                    )
                    .if_supports_color(Stream::Stderr, |t| t.yellow())
                );
            }
            let json = serde_json::to_string_pretty(&stats)
                .map_err(|e| format!("Failed to serialize JSON: {}", e))?;
            fs::write(path, json).map_err(|e| format!("Failed to write JSON: {}", e))?;
        }
    }

    Ok(())
}

/// Print statistics to stdout
pub(crate) fn print_statistics(stats: &SolverStats, args: &Args) {
    println!();
    println_colored(args, "Statistics:", Some(owo_colors::AnsiColors::Yellow));
    println_colored(
        args,
        &format!("  Files processed: {}", stats.files_processed),
        None,
    );
    println_colored(
        args,
        &format!("  Successful: {}", stats.success_count),
        Some(owo_colors::AnsiColors::Green),
    );
    if stats.error_count > 0 {
        println_colored(
            args,
            &format!("  Errors: {}", stats.error_count),
            Some(owo_colors::AnsiColors::Red),
        );
    }
    println_colored(
        args,
        &format!("  Total time: {}ms", stats.execution_time_ms),
        None,
    );

    if args.profile && stats.files_processed > 1 {
        println_colored(
            args,
            &format!("  Average time/file: {}ms", stats.avg_time_per_file_ms),
            None,
        );
        println_colored(args, &format!("  Min time: {}ms", stats.min_time_ms), None);
        println_colored(args, &format!("  Max time: {}ms", stats.max_time_ms), None);
    }

    if stats.memory_bytes > 0 {
        println_colored(
            args,
            &format!("  Memory used: {} MB", stats.memory_bytes / 1_048_576),
            None,
        );
    }

    if args.profile && stats.peak_memory_bytes > 0 {
        println_colored(
            args,
            &format!("  Peak memory: {} MB", stats.peak_memory_bytes / 1_048_576),
            None,
        );
    }

    // Print SAT solver statistics if requested
    if args.stats {
        println!();
        println_colored(
            args,
            "SAT Solver Statistics:",
            Some(owo_colors::AnsiColors::Cyan),
        );
        println_colored(args, &format!("  Decisions: {}", stats.decisions), None);
        println_colored(
            args,
            &format!("  Propagations: {}", stats.propagations),
            None,
        );
        println_colored(args, &format!("  Conflicts: {}", stats.conflicts), None);
        println_colored(args, &format!("  Restarts: {}", stats.restarts), None);
        if stats.decisions > 0 {
            let conflicts_per_decision = stats.conflicts as f64 / stats.decisions as f64;
            println_colored(
                args,
                &format!("  Conflicts/Decision: {:.3}", conflicts_per_decision),
                None,
            );
        }
    }

    // Print detailed profiling data if enabled
    if args.profile
        && let Some(ref profiling_data) = stats.profiling_data
    {
        println!();
        println_colored(args, "Profiling Data:", Some(owo_colors::AnsiColors::Cyan));

        for prof in profiling_data {
            let time_display = if prof.duration_us >= 1000 {
                format!("{:.2}ms", prof.duration_us as f64 / 1000.0)
            } else {
                format!("{}μs", prof.duration_us)
            };

            println_colored(
                args,
                &format!("  {} - {}", prof.operation, time_display),
                None,
            );

            if prof.memory_delta_bytes != 0 {
                let mem_display = if prof.memory_delta_bytes.unsigned_abs() >= 1_048_576 {
                    format!("{:.2} MB", prof.memory_delta_bytes as f64 / 1_048_576.0)
                } else if prof.memory_delta_bytes.unsigned_abs() >= 1024 {
                    format!("{:.2} KB", prof.memory_delta_bytes as f64 / 1024.0)
                } else {
                    format!("{} bytes", prof.memory_delta_bytes)
                };

                println_colored(
                    args,
                    &format!("    Memory delta: {}", mem_display),
                    Some(owo_colors::AnsiColors::BrightBlack),
                );
            }
        }
    }
}

/// Print version information
pub(crate) fn print_version_info(args: &Args) {
    println_colored(
        args,
        &format!("OxiZ SMT Solver v{}", env!("CARGO_PKG_VERSION")),
        Some(owo_colors::AnsiColors::Green),
    );
    println_colored(
        args,
        &format!(
            "========================{}",
            "=".repeat(env!("CARGO_PKG_VERSION").len())
        ),
        Some(owo_colors::AnsiColors::Green),
    );
    println!();
    println_colored(
        args,
        &format!("Version: {}", env!("CARGO_PKG_VERSION")),
        None,
    );
    println_colored(
        args,
        &format!("Authors: {}", env!("CARGO_PKG_AUTHORS")),
        None,
    );
    println_colored(
        args,
        &format!("Repository: {}", env!("CARGO_PKG_REPOSITORY")),
        None,
    );
    println_colored(
        args,
        &format!("License: {}", env!("CARGO_PKG_LICENSE")),
        None,
    );
    println!();
    println_colored(
        args,
        &format!("Rust Version: {}", env!("CARGO_PKG_RUST_VERSION")),
        None,
    );
    println_colored(
        args,
        &format!(
            "Build Profile: {}",
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
        ),
        None,
    );
    println_colored(args, &format!("Target: {}", std::env::consts::ARCH), None);
    println_colored(args, &format!("OS: {}", std::env::consts::OS), None);
    println!();
    println_colored(
        args,
        "A high-performance SMT solver written in pure Rust",
        Some(owo_colors::AnsiColors::Cyan),
    );
}

/// Print interactive help
pub(crate) fn print_help(args: &Args) {
    println_colored(
        args,
        "OxiZ Interactive Mode Help",
        Some(owo_colors::AnsiColors::Green),
    );
    println_colored(
        args,
        "=========================",
        Some(owo_colors::AnsiColors::Green),
    );
    println!();

    println_colored(
        args,
        "Interactive Commands:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  (help)     - Show this help message", None);
    println_colored(args, "  (version)  - Show version information", None);
    println_colored(args, "  (exit)     - Exit the interactive mode", None);
    println_colored(args, "  (quit)     - Exit the interactive mode", None);
    println!();

    println_colored(
        args,
        "SMT-LIB2 Core Commands:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(
        args,
        "  (set-logic <logic>)            - Set the solver logic",
        None,
    );
    println_colored(
        args,
        "  (declare-const <name> <sort>)  - Declare a constant",
        None,
    );
    println_colored(
        args,
        "  (declare-fun <name> (<sorts>) <sort>) - Declare a function",
        None,
    );
    println_colored(
        args,
        "  (assert <formula>)             - Add an assertion",
        None,
    );
    println_colored(
        args,
        "  (check-sat)                    - Check satisfiability",
        None,
    );
    println_colored(
        args,
        "  (get-model)                    - Get satisfying assignment",
        None,
    );
    println_colored(
        args,
        "  (get-value (<terms>))          - Get values of terms",
        None,
    );
    println_colored(
        args,
        "  (get-unsat-core)               - Get unsat core (if enabled)",
        None,
    );
    println_colored(
        args,
        "  (push)                         - Push assertion context",
        None,
    );
    println_colored(
        args,
        "  (pop)                          - Pop assertion context",
        None,
    );
    println!();

    println_colored(args, "Examples:", Some(owo_colors::AnsiColors::Yellow));
    println_colored(
        args,
        "  Simple SAT check:",
        Some(owo_colors::AnsiColors::Cyan),
    );
    println_colored(args, "    (set-logic QF_LIA)", None);
    println_colored(args, "    (declare-const x Int)", None);
    println_colored(args, "    (assert (= x 42))", None);
    println_colored(args, "    (check-sat)", None);
    println_colored(args, "    (get-model)", None);
    println!();

    println_colored(args, "  Boolean logic:", Some(owo_colors::AnsiColors::Cyan));
    println_colored(args, "    (set-logic QF_UF)", None);
    println_colored(args, "    (declare-const p Bool)", None);
    println_colored(args, "    (declare-const q Bool)", None);
    println_colored(args, "    (assert (and p (not q)))", None);
    println_colored(args, "    (check-sat)", None);
    println!();

    println_colored(
        args,
        "  Incremental solving:",
        Some(owo_colors::AnsiColors::Cyan),
    );
    println_colored(args, "    (declare-const x Int)", None);
    println_colored(args, "    (assert (> x 0))", None);
    println_colored(args, "    (push)", None);
    println_colored(args, "    (assert (< x 10))", None);
    println_colored(args, "    (check-sat)", None);
    println_colored(args, "    (pop)", None);
    println_colored(args, "    (assert (> x 100))", None);
    println_colored(args, "    (check-sat)", None);
    println!();

    println_colored(
        args,
        "Supported Logics:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  QF_UF   - Uninterpreted functions", None);
    println_colored(args, "  QF_LIA  - Linear integer arithmetic", None);
    println_colored(args, "  QF_LRA  - Linear real arithmetic", None);
    println_colored(args, "  QF_BV   - Bit-vectors", None);
    println_colored(args, "  ALL     - All theories", None);
    println!();

    println_colored(args, "Tips:", Some(owo_colors::AnsiColors::Yellow));
    println_colored(args, "  - Use arrow keys to navigate history", None);
    println_colored(
        args,
        "  - Multi-line input is supported (auto-detects unbalanced parentheses)",
        None,
    );
    println_colored(args, "  - History is saved to ~/.oxiz_history", None);
    println_colored(args, "  - Use Ctrl+C or Ctrl+D to exit", None);
    println!();

    println_colored(
        args,
        "LSP Server Mode:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(
        args,
        "  Run with --lsp to start as a Language Server Protocol server",
        None,
    );
    println_colored(args, "  Provides IDE integration with:", None);
    println_colored(
        args,
        "    - Real-time syntax validation and diagnostics",
        None,
    );
    println_colored(
        args,
        "    - Hover documentation for SMT-LIB2 keywords",
        None,
    );
    println_colored(args, "    - Auto-completion for commands and sorts", None);
    println_colored(
        args,
        "    - Document symbol outline (functions and constants)",
        None,
    );
    println_colored(args, "  Example: oxiz --lsp", None);
    println!();

    println_colored(
        args,
        "Shell Completion:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(
        args,
        "  Generate shell completion scripts for your shell:",
        None,
    );
    println_colored(args, "    oxiz --completions bash   # For bash", None);
    println_colored(args, "    oxiz --completions zsh    # For zsh", None);
    println_colored(args, "    oxiz --completions fish   # For fish", None);
    println_colored(
        args,
        "    oxiz --completions powershell  # For PowerShell",
        None,
    );
    println!();
    println_colored(args, "  Install completion (bash example):", None);
    println_colored(
        args,
        "    oxiz --completions bash > ~/.local/share/bash-completion/completions/oxiz",
        None,
    );
    println!();

    println_colored(
        args,
        "DIMACS CNF Format:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(
        args,
        "  OxiZ supports DIMACS CNF format for pure SAT problems:",
        None,
    );
    println_colored(
        args,
        "    oxiz --dimacs problem.cnf        # Auto-detect CNF files",
        None,
    );
    println_colored(
        args,
        "    oxiz --dimacs --dimacs-output problem.cnf  # DIMACS input/output",
        None,
    );
    println_colored(
        args,
        "    oxiz --input-format dimacs file.txt  # Force DIMACS parsing",
        None,
    );
    println!();

    println_colored(
        args,
        "Resource Limits:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  Control solver resource usage:", None);
    println_colored(
        args,
        "    oxiz --memory-limit 1024 problem.smt2    # Limit to 1GB RAM",
        None,
    );
    println_colored(
        args,
        "    oxiz --conflict-limit 10000 problem.smt2  # Max 10k conflicts",
        None,
    );
    println_colored(
        args,
        "    oxiz --decision-limit 5000 problem.smt2   # Max 5k decisions",
        None,
    );
    println!();

    println_colored(
        args,
        "Validation & Formatting:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  Check syntax and format SMT-LIB2 files:", None);
    println_colored(
        args,
        "    oxiz --validate-only problem.smt2        # Check syntax only",
        None,
    );
    println_colored(
        args,
        "    oxiz --format-smtlib problem.smt2        # Pretty-print SMT-LIB2",
        None,
    );
    println_colored(
        args,
        "    oxiz --format-smtlib --indent-width 4 problem.smt2  # Custom indentation",
        None,
    );
    println_colored(
        args,
        "  Validates parenthesis matching, string literals, and basic structure",
        None,
    );
    println!();

    println_colored(
        args,
        "Configuration Presets:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(
        args,
        "  Quick solver configurations for common use cases:",
        None,
    );
    println_colored(
        args,
        "    oxiz --preset fast problem.smt2          # Fast: optimize for speed",
        None,
    );
    println_colored(
        args,
        "    oxiz --preset balanced problem.smt2      # Balanced: speed/completeness trade-off",
        None,
    );
    println_colored(
        args,
        "    oxiz --preset thorough problem.smt2      # Thorough: maximize completeness",
        None,
    );
    println_colored(
        args,
        "    oxiz --preset minimal problem.smt2       # Minimal: fastest, least processing",
        None,
    );
    println!();

    println_colored(
        args,
        "Advanced Solving Options:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  Customize solver behavior:", None);
    println_colored(
        args,
        "    oxiz --simplify problem.smt2             # Enable preprocessing",
        None,
    );
    println_colored(
        args,
        "    oxiz --minimize-model problem.smt2       # Find minimal model",
        None,
    );
    println_colored(
        args,
        "    oxiz --validate-proof problem.smt2       # Validate UNSAT proofs",
        None,
    );
    println_colored(
        args,
        "    oxiz --strategy portfolio problem.smt2   # Use portfolio strategy",
        None,
    );
    println_colored(
        args,
        "  Strategies: cdcl, dpll, portfolio, local-search",
        None,
    );
    println!();

    println_colored(
        args,
        "Model Enumeration:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  Find all satisfying assignments:", None);
    println_colored(
        args,
        "    oxiz --enumerate-models problem.smt2           # Find all models",
        None,
    );
    println_colored(
        args,
        "    oxiz --enumerate-models --max-models 10 prob.smt2  # Max 10 models",
        None,
    );
    println!();

    println_colored(
        args,
        "Optimization Mode:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  Maximize/minimize objectives (MaxSMT):", None);
    println_colored(
        args,
        "    oxiz --optimize problem.smt2                   # Enable optimization",
        None,
    );
    println!();

    println_colored(
        args,
        "Statistics Export:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  Export solver statistics to file:", None);
    println_colored(
        args,
        "    oxiz --export-stats stats.csv problem.smt2     # Export as CSV",
        None,
    );
    println_colored(
        args,
        "    oxiz --export-stats stats.json problem.smt2    # Export as JSON",
        None,
    );
    println_colored(
        args,
        "  Includes: execution time, memory, decisions, conflicts, etc.",
        None,
    );
    println!();

    println_colored(
        args,
        "Performance Optimization:",
        Some(owo_colors::AnsiColors::Yellow),
    );
    println_colored(args, "  Cache results and track benchmarks:", None);
    println_colored(
        args,
        "    oxiz --cache problem.smt2                      # Enable caching",
        None,
    );
    println_colored(
        args,
        "    oxiz --cache-dir /path/to/cache problem.smt2   # Custom cache dir",
        None,
    );
    println_colored(
        args,
        "    oxiz --benchmark-file bench.json problem.smt2  # Track performance",
        None,
    );
    println_colored(
        args,
        "    oxiz --theory-opt lia:fastpath problem.smt2    # Theory optimization",
        None,
    );
    println_colored(
        args,
        "    oxiz --enhanced-errors problem.smt2            # Better error messages",
        None,
    );
}

/// Print colored output to stdout
pub(crate) fn println_colored(args: &Args, text: &str, color: Option<owo_colors::AnsiColors>) {
    if args.no_color {
        println!("{}", text);
    } else if let Some(c) = color {
        println!("{}", text.if_supports_color(Stream::Stdout, |t| t.color(c)));
    } else {
        println!("{}", text);
    }
}

/// Print colored error message to stderr
pub(crate) fn eprintln_colored(args: &Args, text: &str) {
    if args.no_color {
        eprintln!("{}", text);
    } else {
        eprintln!("{}", text.if_supports_color(Stream::Stderr, |t| t.red()));
    }
}
