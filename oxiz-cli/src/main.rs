//! OxiZ CLI - Command-line interface for OxiZ SMT Solver

mod analysis;
mod cache;
mod checkpoint;
mod cicd;
mod dashboard;
mod dependency;
mod diagnostic;
mod dimacs;
mod distributed;
mod format;
mod interactive;
mod interpolate;
mod learning;
mod lsp;
mod model_counter;
mod portfolio;
mod processor;
mod proof_checker;
mod server;
mod tptp;
mod tutorial;
mod unsat_core;
mod wasm_bindings;

use clap::{CommandFactory, Parser, ValueEnum};
use clap_complete::{Shell, generate};
use oxiz_solver::Context;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use analysis::{analyze_query_complexity, apply_auto_tune, classify_problem, validate_script};
use format::{
    eprintln_colored, format_analysis, format_classification, format_smtlib_script,
    pretty_print_model, pretty_print_proof, print_examples,
};
use interactive::run_interactive;
use processor::{run_files, run_stdin, run_watch};

/// Configuration file structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CliConfig {
    /// Default verbosity level
    #[serde(default)]
    verbosity: Option<String>,
    /// Default output format
    #[serde(default)]
    format: Option<String>,
    /// Default timeout in seconds
    #[serde(default)]
    timeout: Option<u64>,
    /// Enable colors by default
    #[serde(default)]
    color: Option<bool>,
    /// Default number of threads
    #[serde(default)]
    threads: Option<usize>,
    /// Enable parallel solving by default
    #[serde(default)]
    parallel: Option<bool>,
}

impl CliConfig {
    /// Load configuration from file
    fn load() -> Self {
        let config_path = dirs::home_dir()
            .map(|mut p| {
                p.push(".oxizrc");
                p
            })
            .or_else(|| {
                dirs::config_dir().map(|mut p| {
                    p.push("oxiz");
                    p.push("config.yaml");
                    p
                })
            });

        if let Some(path) = config_path
            && path.exists()
            && let Ok(contents) = fs::read_to_string(&path)
            && let Ok(config) = serde_yaml::from_str(&contents)
        {
            return config;
        }

        Self::default()
    }

    /// Merge configuration with command-line arguments
    fn merge_with_args(&self, args: &mut Args) {
        // Only apply config if arg is not explicitly set
        if args.verbosity == Verbosity::Normal
            && self.verbosity.is_some()
            && let Some(ref v) = self.verbosity
        {
            match v.as_str() {
                "quiet" => args.verbosity = Verbosity::Quiet,
                "verbose" => args.verbosity = Verbosity::Verbose,
                "debug" => args.verbosity = Verbosity::Debug,
                "trace" => args.verbosity = Verbosity::Trace,
                _ => {}
            }
        }

        if self.timeout.is_some() && args.timeout == 0 {
            args.timeout = self.timeout.unwrap_or(0);
        }

        if self.threads.is_some() && args.threads == 4 {
            args.threads = self.threads.unwrap_or(4);
        }

        if self.parallel.is_some() && !args.parallel {
            args.parallel = self.parallel.unwrap_or(false);
        }

        if let Some(color) = self.color
            && !color
        {
            args.no_color = true;
        }
    }
}

/// Output format for results
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    /// SMT-LIB2 format (default)
    Smtlib,
    /// JSON format
    Json,
    /// YAML format
    Yaml,
}

/// Verbosity level
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq, PartialOrd, Ord)]
enum Verbosity {
    /// No output except results
    Quiet,
    /// Minimal output
    Normal,
    /// Detailed output
    Verbose,
    /// Debug output
    Debug,
    /// Trace output
    Trace,
}

/// OxiZ SMT Solver - Next-Generation SMT Solver in Pure Rust
#[derive(Parser, Debug, Clone)]
#[command(name = "oxiz")]
#[command(author = "COOLJAPAN OU (Team KitaSan)")]
#[command(version)]
#[command(about = "A high-performance SMT solver written in pure Rust")]
struct Args {
    /// Input file(s) (SMT-LIB2 format). Supports glob patterns. If not provided, reads from stdin.
    #[arg(value_name = "FILE")]
    input: Vec<PathBuf>,

    /// Output file. If not provided, writes to stdout.
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Set the logic (e.g., QF_LIA, QF_BV, ALL)
    #[arg(short, long)]
    logic: Option<String>,

    /// Verbosity level
    #[arg(short, long, value_enum, default_value = "normal")]
    verbosity: Verbosity,

    /// Enable quiet mode (equivalent to --verbosity quiet)
    #[arg(short, long)]
    quiet: bool,

    /// Run in interactive mode (REPL)
    #[arg(short, long)]
    interactive: bool,

    /// Timeout in seconds (0 = no timeout)
    #[arg(short, long, default_value = "0")]
    timeout: u64,

    /// Enable parallel solving
    #[arg(long)]
    parallel: bool,

    /// Number of threads for parallel solving
    #[arg(long, default_value = "4")]
    threads: usize,

    /// Output format
    #[arg(short = 'f', long, value_enum, default_value = "smtlib")]
    format: OutputFormat,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,

    /// Recursive directory processing
    #[arg(short = 'R', long)]
    recursive: bool,

    /// Show timing information
    #[arg(long)]
    time: bool,

    /// Show statistics
    #[arg(long)]
    stats: bool,

    /// Show memory usage
    #[arg(long)]
    memory: bool,

    /// Watch mode - rerun on file changes
    #[arg(short, long)]
    watch: bool,

    /// Show progress bar for long operations
    #[arg(long)]
    progress: bool,

    /// SMT-COMP compatible output mode
    #[arg(long)]
    smtcomp: bool,

    /// Enable profiling mode with detailed performance metrics
    #[arg(long)]
    profile: bool,

    /// Run as LSP (Language Server Protocol) server for IDE integration
    #[arg(long)]
    lsp: bool,

    /// Run as REST API HTTP server
    #[arg(long)]
    server: bool,

    /// Port for the REST API server (default: 8080)
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Enable web dashboard for monitoring solver progress
    #[arg(long)]
    dashboard: bool,

    /// Port for the web dashboard (default: 8080)
    #[arg(long, default_value = "8080")]
    dashboard_port: u16,

    /// Generate shell completion script for the specified shell
    #[arg(long, value_name = "SHELL")]
    completions: Option<Shell>,

    /// Input format (auto-detect by default)
    #[arg(long, value_enum)]
    input_format: Option<InputFormat>,

    /// Read DIMACS format input (CNF SAT problems)
    #[arg(long)]
    dimacs: bool,

    /// Write output in DIMACS format
    #[arg(long)]
    dimacs_output: bool,

    /// Write output in TPTP SZS status format (Theorem/CounterSatisfiable)
    #[arg(long)]
    tptp_output: bool,

    /// Resource limit: maximum memory in MB (0 = no limit)
    #[arg(long, default_value = "0")]
    memory_limit: u64,

    /// Resource limit: maximum number of conflicts (0 = no limit)
    #[arg(long, default_value = "0")]
    conflict_limit: u64,

    /// Resource limit: maximum number of decisions (0 = no limit)
    #[arg(long, default_value = "0")]
    decision_limit: u64,

    /// Minimize the satisfying model (find minimal solution)
    #[arg(long)]
    minimize_model: bool,

    /// Validate proof after solving (for UNSAT results)
    #[arg(long)]
    validate_proof: bool,

    /// Enable preprocessing and simplification
    #[arg(long)]
    simplify: bool,

    /// Solver strategy: cdcl, dpll, portfolio, or local-search
    #[arg(long)]
    strategy: Option<String>,

    /// Enumerate all models (find all satisfying assignments)
    #[arg(long)]
    enumerate_models: bool,

    /// Maximum number of models to find (0 = unlimited, only with --enumerate-models)
    #[arg(long, default_value = "0")]
    max_models: usize,

    /// Count satisfying models (use --count-method to choose exact or approximate)
    #[arg(long)]
    count_models: bool,

    /// Model counting method: exact or approximate (default: approximate)
    #[arg(long, default_value = "approximate")]
    count_method: String,

    /// Number of samples for approximate counting (default: 1000)
    #[arg(long, default_value = "1000")]
    count_samples: usize,

    /// Export model count to JSON file
    #[arg(long, value_name = "FILE")]
    count_export: Option<PathBuf>,

    /// Enable optimization mode (maximize or minimize objectives)
    #[arg(long)]
    optimize: bool,

    /// Enable result caching (cache solutions for repeated problems)
    #[arg(long)]
    cache: bool,

    /// Cache directory path (default: ~/.oxiz/cache)
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    /// Benchmark tracking file (track and compare performance over time)
    #[arg(long)]
    benchmark_file: Option<PathBuf>,

    /// Theory-specific optimizations (e.g., "lia:fastpath", "bv:bitblast")
    #[arg(long)]
    theory_opt: Vec<String>,

    /// Enhanced error reporting with suggestions and hints
    #[arg(long)]
    enhanced_errors: bool,

    /// Validate syntax only without solving
    #[arg(long)]
    validate_only: bool,

    /// Export statistics to file (CSV or JSON based on extension)
    #[arg(long, value_name = "FILE")]
    export_stats: Option<PathBuf>,

    /// Format and pretty-print SMT-LIB2 files without solving
    #[arg(long)]
    format_smtlib: bool,

    /// Indentation width for formatted output (default: 2)
    #[arg(long, default_value = "2")]
    indent_width: usize,

    /// Use a solver configuration preset (fast, balanced, thorough, minimal)
    #[arg(long)]
    preset: Option<String>,

    /// Analyze query complexity without solving (shows problem statistics and characteristics)
    #[arg(long)]
    analyze: bool,

    /// Show problem classification and recommended solver settings
    #[arg(long)]
    classify: bool,

    /// Automatically tune solver based on problem characteristics
    #[arg(long)]
    auto_tune: bool,

    /// Show practical usage examples for various features
    #[arg(long)]
    examples: bool,

    /// Extract UNSAT core (minimal unsatisfiable subset of assertions)
    #[arg(long)]
    unsat_core: bool,

    /// Minimize UNSAT core (find smallest unsatisfiable subset)
    #[arg(long)]
    minimize_core: bool,

    /// Generate proof tree in DOT format for visualization
    #[arg(long, value_name = "FILE")]
    proof_dot: Option<PathBuf>,

    /// Validate model against original assertions
    #[arg(long)]
    validate_model: bool,

    /// Enable incremental solving mode (supports push/pop)
    #[arg(long)]
    incremental: bool,

    /// Enable parallel portfolio solving (run multiple strategies concurrently)
    #[arg(long)]
    portfolio_mode: bool,

    /// Portfolio timeout in seconds (0 = use default timeout)
    #[arg(long, default_value = "0")]
    portfolio_timeout: u64,

    /// Verify proof correctness (for UNSAT results with proofs)
    #[arg(long)]
    verify_proof: bool,

    /// Proof file to verify (optional, reads from solver output if not specified)
    #[arg(long, value_name = "FILE")]
    proof_file: Option<PathBuf>,

    /// Enable checkpointing for long-running tasks
    #[arg(long)]
    checkpoint: bool,

    /// Checkpoint directory (default: ~/.oxiz/checkpoints)
    #[arg(long)]
    checkpoint_dir: Option<PathBuf>,

    /// Checkpoint interval in seconds (default: 300 = 5 minutes)
    #[arg(long, default_value = "300")]
    checkpoint_interval: u64,

    /// Resume from the latest checkpoint
    #[arg(long)]
    resume: bool,

    /// Resume from a specific checkpoint file
    #[arg(long, value_name = "FILE")]
    resume_from: Option<PathBuf>,

    /// Analyze dependencies between assertions (shows symbol usage and relationships)
    #[arg(long)]
    dependencies: bool,

    /// Show detailed dependency information (per-assertion breakdown)
    #[arg(long)]
    dependencies_detailed: bool,

    /// Export dependency graph to JSON file
    #[arg(long, value_name = "FILE")]
    dependencies_export: Option<PathBuf>,

    /// Run diagnostic checks to identify potential issues in the problem
    #[arg(long)]
    diagnostic: bool,

    /// Export diagnostic report to JSON file
    #[arg(long, value_name = "FILE")]
    diagnostic_export: Option<PathBuf>,

    /// Start interactive tutorial mode (optionally specify section: intro, basic, theories, advanced, cli)
    #[arg(long)]
    tutorial: Option<Option<String>>,

    /// Enable CI/CD mode with machine-readable output
    #[arg(long)]
    cicd: bool,

    /// Export CI/CD report to JSON file
    #[arg(long, value_name = "FILE")]
    cicd_report: Option<PathBuf>,

    /// Exit with non-zero code on any errors
    #[arg(long)]
    cicd_strict: bool,

    /// Enable interpolation mode for Craig interpolant generation
    #[arg(long)]
    interpolate: bool,

    /// Output format for interpolation (smtlib, text, json)
    #[arg(long, value_name = "FORMAT", default_value = "smtlib")]
    interpolate_format: String,

    /// Interpolation algorithm (mcmillan, pudlak, huang)
    #[arg(long, value_name = "ALGORITHM")]
    interpolate_algorithm: Option<String>,

    /// Enable distributed solving mode
    #[arg(long)]
    distributed: bool,

    /// Run as distributed coordinator at HOST:PORT
    #[arg(long, value_name = "HOST:PORT")]
    coordinator: Option<String>,

    /// Run as distributed worker connecting to coordinator at HOST:PORT
    #[arg(long, value_name = "HOST:PORT")]
    worker: Option<String>,

    /// Number of cubes to generate for distributed solving (default: 64)
    #[arg(long, default_value = "64")]
    num_cubes: usize,
}

/// Input format for problems
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum InputFormat {
    /// SMT-LIB2 format (default)
    Smtlib,
    /// DIMACS CNF format
    Dimacs,
    /// QDIMACS (Quantified Boolean Formula) format
    Qdimacs,
    /// TPTP (Thousands of Problems for Theorem Provers) format
    Tptp,
}

#[tokio::main]
async fn main() {
    let mut args = Args::parse();

    // Handle completion generation
    if let Some(shell) = args.completions {
        let mut cmd = Args::command();
        let bin_name = cmd.get_name().to_string();
        generate(shell, &mut cmd, bin_name, &mut std::io::stdout());
        return;
    }

    // Handle examples display
    if args.examples {
        print_examples();
        return;
    }

    // Handle tutorial mode
    if let Some(section_arg) = args.tutorial {
        let section = section_arg
            .as_ref()
            .and_then(|s| tutorial::parse_tutorial_section(s));

        if let Some(ref arg) = section_arg
            && section.is_none()
        {
            eprintln!("Error: Invalid tutorial section '{}'", arg);
            tutorial::list_tutorial_sections();
            std::process::exit(1);
        }

        tutorial::run_tutorial(section);
        return;
    }

    // Handle LSP mode
    if args.lsp {
        if let Err(e) = lsp::run_lsp_server().await {
            eprintln!("LSP server error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Handle REST API server mode
    if args.server {
        if let Err(e) = server::run_server(args.port).await {
            eprintln!("REST API server error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Handle dashboard mode
    if args.dashboard {
        let state = dashboard::DashboardState::new();
        if let Err(e) = dashboard::start_dashboard_server(state, args.dashboard_port).await {
            eprintln!("Dashboard server error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Handle distributed worker mode
    if let Some(ref coordinator_addr) = args.worker {
        let config = distributed::DistributedConfig {
            address: coordinator_addr.clone(),
            num_cubes: args.num_cubes,
            ..Default::default()
        };
        if let Err(e) = distributed::run_worker(&config) {
            eprintln!("Worker error: {}", e);
            std::process::exit(1);
        }
        return;
    }

    // Handle distributed coordinator mode
    if let Some(ref bind_addr) = args.coordinator {
        // Read input script first
        let script = if args.input.is_empty() {
            // Read from stdin
            use std::io::Read;
            let mut script = String::new();
            std::io::stdin()
                .read_to_string(&mut script)
                .expect("Failed to read from stdin");
            script
        } else {
            // Read from first input file
            fs::read_to_string(&args.input[0]).unwrap_or_else(|e| {
                eprintln!("Failed to read input file: {}", e);
                std::process::exit(1);
            })
        };

        let config = distributed::DistributedConfig {
            address: bind_addr.clone(),
            num_cubes: args.num_cubes,
            ..Default::default()
        };

        match distributed::run_coordinator(&script, &config) {
            Ok(result) => {
                println!("{}", distributed::format_distributed_result(&result));
            }
            Err(e) => {
                eprintln!("Coordinator error: {}", e);
                std::process::exit(1);
            }
        }
        return;
    }

    // Load configuration file and merge with args
    let config = CliConfig::load();
    config.merge_with_args(&mut args);

    // Determine verbosity level
    let verbosity = if args.quiet {
        Verbosity::Quiet
    } else {
        args.verbosity
    };

    // Set up logging
    if verbosity >= Verbosity::Debug {
        let level = match verbosity {
            Verbosity::Trace => Level::TRACE,
            Verbosity::Debug => Level::DEBUG,
            _ => Level::INFO,
        };
        let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
        if let Err(e) = tracing::subscriber::set_global_default(subscriber) {
            eprintln_colored(&args, &format!("Failed to set tracing subscriber: {}", e));
            std::process::exit(1);
        }
    }

    // Create solver context
    let mut ctx = Context::new();

    // Set logic if provided
    if let Some(logic) = &args.logic {
        ctx.set_logic(logic);
    }

    // Apply solver options
    apply_solver_options(&mut ctx, &args);

    // Handle input
    if args.interactive {
        run_interactive(&mut ctx, &args, verbosity);
    } else if args.input.is_empty() {
        run_stdin(&mut ctx, &args, verbosity);
    } else if args.watch {
        run_watch(&mut ctx, &args, verbosity);
    } else {
        run_files(&mut ctx, &args, verbosity);
    }
}

/// Apply configuration preset
fn apply_preset(ctx: &mut Context, preset: &str) {
    match preset {
        "fast" => {
            // Fast preset: optimize for speed, minimal checking
            ctx.set_option("simplify", "true");
            ctx.set_option("strategy", "cdcl");
            ctx.set_option("restarts", "frequent");
            ctx.set_option("branching", "vsids");
        }
        "balanced" => {
            // Balanced preset: good trade-off between speed and completeness
            ctx.set_option("simplify", "true");
            ctx.set_option("strategy", "portfolio");
            ctx.set_option("restarts", "moderate");
        }
        "thorough" => {
            // Thorough preset: maximize completeness, slower
            ctx.set_option("simplify", "true");
            ctx.set_option("strategy", "portfolio");
            ctx.set_option("restarts", "rare");
            ctx.set_option("lookahead", "true");
            ctx.set_option("produce-proofs", "true");
        }
        "minimal" => {
            // Minimal preset: minimal processing, fastest
            ctx.set_option("simplify", "false");
            ctx.set_option("strategy", "dpll");
            ctx.set_option("restarts", "never");
        }
        _ => {
            // Unknown preset, ignore
        }
    }
}

/// Apply solver options from command-line arguments
pub(crate) fn apply_solver_options(ctx: &mut Context, args: &Args) {
    // Apply preset first if specified
    if let Some(ref preset) = args.preset {
        apply_preset(ctx, preset);
    }
    // Apply resource limits
    if args.memory_limit > 0 {
        ctx.set_option(
            "memory-limit",
            &format!("{}", args.memory_limit * 1024 * 1024),
        );
    }
    if args.conflict_limit > 0 {
        ctx.set_option("conflict-limit", &format!("{}", args.conflict_limit));
    }
    if args.decision_limit > 0 {
        ctx.set_option("decision-limit", &format!("{}", args.decision_limit));
    }

    // Apply solver options
    if args.simplify {
        ctx.set_option("simplify", "true");
    }
    if args.minimize_model {
        ctx.set_option("minimize-model", "true");
    }
    if args.validate_proof {
        ctx.set_option("produce-proofs", "true");
        ctx.set_option("validate-proofs", "true");
    }
    if let Some(ref strategy) = args.strategy {
        ctx.set_option("strategy", strategy);
    }

    // Model enumeration
    if args.enumerate_models {
        ctx.set_option("enumerate-models", "true");
        if args.max_models > 0 {
            ctx.set_option("max-models", &format!("{}", args.max_models));
        }
    }

    // Optimization mode
    if args.optimize {
        ctx.set_option("optimize", "true");
    }

    // Theory-specific optimizations
    for opt in &args.theory_opt {
        if let Some((theory, setting)) = opt.split_once(':') {
            ctx.set_option(&format!("theory.{}.{}", theory, setting), "true");
        }
    }

    // Enhanced error reporting
    if args.enhanced_errors {
        ctx.set_option("enhanced-errors", "true");
    }
}

pub(crate) fn execute_and_format(ctx: &mut Context, script: &str, args: &Args) -> String {
    // If format-smtlib mode, just format and return
    if args.format_smtlib {
        return format_smtlib_script(script, args.indent_width);
    }

    // If validate-only mode, just validate syntax
    if args.validate_only {
        return match validate_script(script) {
            Ok(msg) => msg,
            Err(e) => format!("(error \"Validation failed: {}\")", e),
        };
    }

    // If analyze mode, analyze query complexity
    if args.analyze {
        let analysis = analyze_query_complexity(script);
        return format_analysis(&analysis, args);
    }

    // If classify mode, classify problem and provide recommendations
    if args.classify {
        let analysis = analyze_query_complexity(script);
        let classification = classify_problem(script, &analysis);
        return format_classification(&classification, &analysis, args);
    }

    // If dependency analysis mode, analyze dependencies between assertions
    if args.dependencies || args.dependencies_detailed || args.dependencies_export.is_some() {
        let graph = dependency::analyze_dependencies(script);

        // Export to JSON if requested
        if let Some(export_path) = &args.dependencies_export {
            if let Err(e) = std::fs::write(
                export_path,
                serde_json::to_string_pretty(&graph).unwrap_or_default(),
            ) {
                eprintln!("Failed to export dependencies: {}", e);
            } else if args.verbosity >= Verbosity::Normal {
                println!("Dependency graph exported to {}", export_path.display());
            }
        }

        // Display dependency information
        if args.dependencies || args.dependencies_detailed {
            let detailed = args.dependencies_detailed;
            let formatted = dependency::format_dependency_graph(&graph, detailed);
            return formatted;
        }

        // If only exporting, return success message
        if args.dependencies_export.is_some() {
            return "Dependency analysis complete".to_string();
        }
    }

    // If diagnostic mode, run comprehensive problem diagnostics
    if args.diagnostic || args.diagnostic_export.is_some() {
        let result = diagnostic::diagnose_problem(script);

        // Export to JSON if requested
        if let Some(export_path) = &args.diagnostic_export {
            if let Err(e) = std::fs::write(
                export_path,
                serde_json::to_string_pretty(&result).unwrap_or_default(),
            ) {
                eprintln!("Failed to export diagnostic report: {}", e);
            } else if args.verbosity >= Verbosity::Normal {
                println!("Diagnostic report exported to {}", export_path.display());
            }
        }

        // Display diagnostic information
        if args.diagnostic {
            let formatted = diagnostic::format_diagnostic_result(&result);
            return formatted;
        }

        // If only exporting, return success message
        if args.diagnostic_export.is_some() {
            return "Diagnostic check complete".to_string();
        }
    }

    // If interpolation mode, compute Craig interpolant
    if args.interpolate {
        let format = interpolate::InterpolateFormat::from_str(&args.interpolate_format)
            .unwrap_or(interpolate::InterpolateFormat::Smtlib);

        let algorithm =
            args.interpolate_algorithm
                .as_ref()
                .and_then(|a| match a.to_lowercase().as_str() {
                    "mcmillan" => Some(oxiz_proof::InterpolationAlgorithm::McMillan),
                    "pudlak" => Some(oxiz_proof::InterpolationAlgorithm::Pudlak),
                    "huang" => Some(oxiz_proof::InterpolationAlgorithm::Huang),
                    _ => None,
                });

        return interpolate::execute_interpolation(script, format, algorithm);
    }

    // If model counting mode, count satisfying models
    if args.count_models || args.count_export.is_some() {
        let method = match args.count_method.as_str() {
            "exact" => model_counter::CountingMethod::Exact,
            "approximate" => model_counter::CountingMethod::ApproximateSampling,
            _ => {
                eprintln!(
                    "Warning: Invalid count method '{}', using approximate",
                    args.count_method
                );
                model_counter::CountingMethod::ApproximateSampling
            }
        };

        let counter = model_counter::ModelCounter::new().with_samples(args.count_samples);

        let result = counter.count(ctx, script, method);

        // Export to JSON if requested
        if let Some(export_path) = &args.count_export {
            if let Err(e) = std::fs::write(
                export_path,
                serde_json::to_string_pretty(&result).unwrap_or_default(),
            ) {
                eprintln!("Failed to export model count: {}", e);
            } else if args.verbosity >= Verbosity::Normal {
                println!("Model count exported to {}", export_path.display());
            }
        }

        // Display count information
        if args.count_models {
            let formatted = model_counter::format_model_count(&result);
            return formatted;
        }

        // If only exporting, return success message
        if args.count_export.is_some() {
            return "Model counting complete".to_string();
        }
    }

    // If auto-tune mode, analyze and apply recommended settings
    if args.auto_tune {
        let analysis = analyze_query_complexity(script);
        let classification = classify_problem(script, &analysis);
        apply_auto_tune(ctx, &analysis, &classification);
        // Continue with normal execution after tuning
    }

    // If portfolio mode is enabled, use parallel strategy execution
    if args.portfolio_mode {
        let timeout = if args.portfolio_timeout > 0 {
            args.portfolio_timeout
        } else if args.timeout > 0 {
            args.timeout
        } else {
            300 // Default 5 minutes
        };

        let logic = args.logic.as_deref();
        match portfolio::solve_portfolio(script, args, logic, ctx, timeout) {
            Ok(result) => {
                // Format the output with strategy information
                let mut output = result.output;
                if args.verbosity >= Verbosity::Verbose {
                    output.insert(
                        0,
                        format!(
                            "; Portfolio solver: {} won in {}ms",
                            result.strategy_name, result.time_ms
                        ),
                    );
                }
                return output.join("\n");
            }
            Err(e) => {
                return format!("(error \"Portfolio solving failed: {}\")", e);
            }
        }
    }

    match ctx.execute_script(script) {
        Ok(mut output) => {
            // Handle UNSAT core extraction if requested
            if args.unsat_core && output.iter().any(|line| line.contains("unsat")) {
                // Add get-unsat-core command if not already present
                let has_core_cmd = script.contains("get-unsat-core");
                if !has_core_cmd {
                    // Execute get-unsat-core command
                    if let Ok(core_output) = ctx.execute_script("(get-unsat-core)") {
                        output.extend(core_output);
                    }
                }
            }

            // Handle model validation if requested
            if args.validate_model
                && output
                    .iter()
                    .any(|line| line.contains("sat") && !line.contains("unsat"))
            {
                // Execute get-model if not already done
                let has_model = output.iter().any(|line| line.contains("define-fun"));
                if !has_model && let Ok(model_output) = ctx.execute_script("(get-model)") {
                    output.extend(model_output);
                }
            }

            // Handle proof DOT generation if requested
            if let Some(ref dot_path) = args.proof_dot
                && let Some(proof_line) = output.iter().find(|line| {
                    line.contains("proof") || line.contains("step") || line.contains("assume")
                })
            {
                if let Err(e) = std::fs::File::create(dot_path).and_then(|file| {
                    unsat_core::generate_proof_dot(proof_line, file).map_err(std::io::Error::other)
                }) {
                    eprintln_colored(args, &format!("Failed to generate proof DOT: {}", e));
                } else if args.verbosity >= Verbosity::Verbose {
                    eprintln_colored(
                        args,
                        &format!("Proof tree written to {}", dot_path.display()),
                    );
                }
            }

            // Handle proof verification if requested
            if args.verify_proof && output.iter().any(|line| line.contains("unsat")) {
                let proof_text = if let Some(ref proof_file) = args.proof_file {
                    // Read proof from file
                    match fs::read_to_string(proof_file) {
                        Ok(text) => text,
                        Err(e) => {
                            eprintln_colored(args, &format!("Failed to read proof file: {}", e));
                            String::new()
                        }
                    }
                } else {
                    // Extract proof from output
                    output
                        .iter()
                        .filter(|line| {
                            line.contains("proof")
                                || line.contains("step")
                                || line.contains("->")
                                || line.contains("axiom")
                                || line.contains("resolution")
                        })
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                if !proof_text.is_empty() {
                    match proof_checker::parse_simple_proof(&proof_text) {
                        Ok(proof) => match proof.verify() {
                            Ok(()) => {
                                if args.verbosity >= Verbosity::Verbose {
                                    eprintln_colored(args, "; Proof verification: VALID");
                                    let core = proof.extract_unsat_core();
                                    eprintln_colored(
                                        args,
                                        &format!("; UNSAT core size: {}", core.len()),
                                    );
                                }
                                output.insert(0, "; Proof verified: VALID".to_string());
                            }
                            Err(e) => {
                                eprintln_colored(
                                    args,
                                    &format!("; Proof verification FAILED: {}", e),
                                );
                                output.insert(0, format!("; Proof verification FAILED: {}", e));
                            }
                        },
                        Err(e) => {
                            if args.verbosity >= Verbosity::Verbose {
                                eprintln_colored(args, &format!("; Failed to parse proof: {}", e));
                            }
                        }
                    }
                }
            }

            if args.smtcomp {
                // SMT-COMP compatible output
                output.join("\n")
            } else {
                // Pretty-print models and proofs
                let formatted: Vec<String> = output
                    .into_iter()
                    .map(|line| {
                        if line.starts_with('(') && line.contains("define-fun") {
                            pretty_print_model(&line, args)
                        } else if line.starts_with('(')
                            && (line.contains("proof")
                                || line.contains("step")
                                || line.contains("assume")
                                || line.contains("cl"))
                        {
                            pretty_print_proof(&line, args)
                        } else {
                            line
                        }
                    })
                    .collect();
                formatted.join("\n")
            }
        }
        Err(e) => format!("(error \"{}\")", e),
    }
}
