//! OxiZ CLI - Command-line interface for OxiZ SMT Solver

mod analysis;
mod cache;
mod checkpoint;
mod checkpointing;
mod cicd;
mod core_min;
mod dashboard;
mod dependency;
mod diagnostic;
mod dimacs;
mod distributed;
mod format;
mod incremental;
mod interactive;
mod interpolate;
mod learning;
mod lsp;
mod memory;
mod ml_tactic;
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

#[derive(Parser, Debug)]
struct PerfDashboardRenderArgs {
    /// Directory containing performance regression history JSON files
    #[arg(long, value_name = "DIR")]
    perf: PathBuf,
    /// Output directory for the rendered static dashboard
    #[arg(long, value_name = "DIR")]
    output: PathBuf,
}

fn maybe_handle_dashboard_render_command() -> anyhow::Result<bool> {
    let raw_args = std::env::args_os().collect::<Vec<_>>();
    let Some(command) = raw_args.get(1).and_then(|arg| arg.to_str()) else {
        return Ok(false);
    };
    let Some(subcommand) = raw_args.get(2).and_then(|arg| arg.to_str()) else {
        return Ok(false);
    };

    if command != "dashboard" || subcommand != "render" {
        return Ok(false);
    }

    let parse_args = std::iter::once(raw_args[0].clone())
        .chain(raw_args.into_iter().skip(3))
        .collect::<Vec<_>>();
    let args = PerfDashboardRenderArgs::try_parse_from(parse_args)?;
    dashboard::perf::render_perf_dashboard(&args.perf, &args.output)?;
    Ok(true)
}

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

    /// Use the ML-guided tactic selector (oxiz-ml) to pick a solver posture
    /// per formula and learn from outcomes (off by default)
    #[arg(long)]
    ml_tactic_selection: bool,

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

    /// Enable binary proof logging; writes a proof log to FILE after each solve
    #[arg(long, value_name = "FILE")]
    proof_log: Option<PathBuf>,

    /// Load a binary proof log and verify it offline.
    /// Exits printing "PROOF VALID" or "PROOF INVALID: `<reason>`"
    #[arg(long, value_name = "FILE")]
    verify_proof_log: Option<PathBuf>,

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
    match maybe_handle_dashboard_render_command() {
        Ok(true) => return,
        Ok(false) => {}
        Err(error) => {
            eprintln!("Dashboard render error: {error}");
            std::process::exit(1);
        }
    }

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
            // Read from stdin. The workspace profile builds with
            // `panic = "abort"`, so an `.expect()` here would abort the
            // whole process on an ordinary I/O error (e.g. a broken pipe)
            // instead of exiting cleanly with a message, unlike every other
            // stdin-reading path in this file (see `processor::run_stdin`).
            use std::io::Read;
            let mut script = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut script) {
                eprintln!("Failed to read from stdin: {}", e);
                std::process::exit(1);
            }
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

    // Enforce a wall-clock `--timeout` from a separate supervisor process for
    // the ordinary solving modes. See `supervise_timeout` for why an
    // out-of-process deadline is reliable where an in-process timer thread can
    // be starved by the abandoned, CPU-bound solver under load.
    if std::env::var_os(TIMEOUT_SUPERVISOR_GUARD).is_some() {
        // We are the supervised child. The parent process owns the real
        // deadline (`args.timeout`) and kills us promptly when it fires, so we
        // solve directly on the main thread with no in-process timer of our own
        // -- that keeps the ordinary (fast) solve off the thread-handoff path,
        // which itself can starve under heavy load.
        //
        // We do, however, arm a lightweight *self-destruct* backstop well past
        // the parent's deadline. Its sole purpose is the pathological case
        // where the parent is itself killed (e.g. a test harness hard-kills the
        // whole process tree): an otherwise-unbounded child would be reparented
        // and keep a core pegged on the abandoned solve. The backstop thread
        // merely sleeps -- it does not compete for CPU with the solve -- and
        // guarantees an orphan eventually terminates instead of leaking a core.
        let backstop = args.timeout.saturating_mul(4).saturating_add(30);
        args.timeout = 0;
        let _ = std::thread::Builder::new()
            .name("oxiz-orphan-guard".to_string())
            .spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(backstop));
                std::process::exit(TIMEOUT_EXIT_CODE);
            });
    } else if args.timeout > 0
        && !args.interactive
        && !args.watch
        && !args.portfolio_mode
        && args.strategy.as_deref() != Some("portfolio")
    {
        // Parent supervisor: terminates the process on completion or deadline.
        // Only returns here if it could not spawn the child, in which case we
        // fall through to the in-process timeout path as an honest fallback.
        supervise_timeout(&args);
    }

    // Determine verbosity level
    let verbosity = if args.quiet {
        Verbosity::Quiet
    } else {
        args.verbosity
    };

    // `--threads N`, when explicitly set (i.e. != the clap default), sizes the
    // global Rayon pool used for parallel *file* processing and — via
    // `execute_and_format` — routes a single-problem solve through the
    // N-worker portfolio. Configure the pool here, once, before any parallel
    // work starts. (`build_global` fails harmlessly if a pool already exists.)
    if args.threads != DEFAULT_THREADS && args.threads >= 1 {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global();
    }

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

    // Handle --verify-proof-log before creating a solving context.
    if let Some(ref log_path) = args.verify_proof_log {
        match Context::verify_proof_log(log_path) {
            Ok(result) => {
                if result.is_valid() {
                    println!("PROOF VALID");
                } else {
                    println!("PROOF INVALID: {}", result);
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("Error verifying proof log: {}", e);
                std::process::exit(1);
            }
        }
        return;
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
    } else if args.incremental {
        // Incremental mode streams top-level commands against a single
        // persistent context (carried across files), honoring push/pop and
        // cross-file declarations — see `incremental::run_incremental`.
        incremental::run_incremental(&mut ctx, &args, verbosity);
    } else if args.input.is_empty() {
        run_stdin(&mut ctx, &args, verbosity);
    } else if args.watch {
        run_watch(&mut ctx, &args, verbosity);
    } else {
        run_files(&mut ctx, &args, verbosity);
    }

    // `--timeout`/config-file timeout is enforced per-script inside
    // `execute_and_format` (see `execute_script_with_timeout`). When a script
    // exceeds its deadline, a dedicated watchdog thread reports "unknown" and
    // terminates the process directly with `TIMEOUT_EXIT_CODE` (the honest,
    // timeout-specific exit code), so there is nothing to reconcile here.
}

/// The clap default for `--threads`. Used as the "was it explicitly set?"
/// sentinel: an explicit `--threads N` with `N != DEFAULT_THREADS` opts into
/// parallel solving (Rayon pool sizing + N-worker portfolio routing), mirroring
/// the same heuristic [`CliConfig::merge_with_args`] uses (`Args` has no
/// `Option<usize>` to distinguish "left at its default" from "explicitly 4").
const DEFAULT_THREADS: usize = 4;

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
    // Wire binary proof logging path if requested.
    if let Some(ref log_path) = args.proof_log {
        ctx.set_proof_log_path(Some(log_path.clone()));
    }
    // Apply preset first if specified
    if let Some(ref preset) = args.preset {
        apply_preset(ctx, preset);
    }
    // Apply resource limits.
    //
    // `--conflict-limit`/`--decision-limit` map to `Context::set_option`'s
    // recognised `max-conflicts`/`max-decisions` keys (see
    // `Context::set_option`'s doc comment for the full list of wired keys);
    // they used to be written under the *unrecognised* keys
    // `conflict-limit`/`decision-limit`, which `set_option` silently records
    // but never consults, so the limits were never actually enforced.
    if args.conflict_limit > 0 {
        ctx.set_option("max-conflicts", &format!("{}", args.conflict_limit));
    }
    if args.decision_limit > 0 {
        ctx.set_option("max-decisions", &format!("{}", args.decision_limit));
    }
    // `--memory-limit` has no corresponding lever in `oxiz-solver` today: a
    // real wall-clock/heap bound would require either an unsafe global
    // allocator with per-thread accounting or OS-level rlimits (both of
    // which are out of scope for `oxiz-cli`, and the latter would need a
    // non-pure-Rust FFI dependency). Rather than silently accepting the flag
    // and doing nothing, tell the user honestly that it is not enforced so
    // they do not mistake "not enforced" for "impossible to exceed".
    if args.memory_limit > 0 && !args.quiet {
        eprintln!(
            "warning: --memory-limit {} is accepted but not enforced by this build \
             (no memory-tracking backend); solving will proceed without a memory bound",
            args.memory_limit
        );
    }

    // Apply solver options
    if args.simplify {
        ctx.set_option("simplify", "true");
    }
    if args.validate_proof {
        ctx.set_option("produce-proofs", "true");
        ctx.set_option("validate-proofs", "true");
    }
    // `--unsat-core`/`--minimize-core` extract a core only after the fact
    // (see `execute_and_format`), but core *tracking* has to be switched on
    // before `check-sat` runs or there is nothing to extract; without this
    // the CLI always reported an error instead of a core.
    if args.unsat_core || args.minimize_core {
        ctx.set_option("produce-unsat-cores", "true");
    }
    if let Some(ref strategy) = args.strategy {
        ctx.set_option("strategy", strategy);
        match strategy.as_str() {
            // "cdcl" is the (only) search engine `oxiz-solver` implements
            // for non-portfolio runs, so requesting it is always honored.
            "cdcl" => {}
            // Routed to the real parallel-portfolio path in
            // `execute_and_format` (see the `args.portfolio_mode ||
            // strategy == "portfolio"` check there).
            "portfolio" => {}
            other => {
                if !args.quiet {
                    eprintln!(
                        "warning: --strategy {other} is not implemented by this build; \
                         falling back to the default CDCL search"
                    );
                }
            }
        }
    }

    // The following flags are accepted for compatibility with the documented
    // CLI surface but currently have no backing implementation in
    // `oxiz-solver`/`oxiz-cli`. Report that honestly instead of silently
    // accepting them and doing nothing (see `format::print_examples` /
    // `--help`, which is intentionally worded to describe intent rather than
    // guaranteed behaviour for these).
    if args.minimize_model && !args.quiet {
        eprintln!(
            "warning: --minimize-model is accepted but model minimization is not yet \
             implemented; the first model found is reported as-is"
        );
    }
    // `--enumerate-models` is implemented for real (see
    // `enumerate_additional_models`, invoked from `execute_and_format`), so
    // it no longer needs an "unimplemented" warning here. `--max-models`
    // only means anything together with `--enumerate-models`; flag the
    // combination that leaves it completely inert, mirroring the other
    // dead-flag warnings in this function.
    if args.max_models > 0 && !args.enumerate_models && !args.quiet {
        eprintln!(
            "warning: --max-models {} has no effect without --enumerate-models",
            args.max_models
        );
    }
    if args.optimize && !args.quiet {
        eprintln!(
            "warning: --optimize is accepted but objective optimization is not yet \
             implemented by oxiz-cli; running a plain satisfiability check instead"
        );
    }
    if !args.theory_opt.is_empty() && !args.quiet {
        eprintln!(
            "warning: --theory-opt is accepted but theory-specific tuning knobs are not yet \
             implemented; the requested settings ({}) have no effect",
            args.theory_opt.join(", ")
        );
    }
}

/// Exit code reported when `--timeout` is exceeded, matching the convention
/// used by the Unix `timeout(1)` utility.
const TIMEOUT_EXIT_CODE: i32 = 124;

/// Environment variable set on the re-executed child by [`supervise_timeout`].
/// Its presence tells the child that a parent supervisor process already owns
/// the wall-clock deadline, so the child must solve directly (in process, with
/// no timeout of its own) rather than recursively re-supervising.
const TIMEOUT_SUPERVISOR_GUARD: &str = "OXIZ_TIMEOUT_SUPERVISED";

/// Enforce a hard wall-clock `--timeout` from a thin *supervisor* process.
///
/// A `check-sat` cannot be cooperatively cancelled mid-search, so enforcing a
/// deadline in-process means abandoning a CPU-bound solver thread while some
/// other thread races to report "unknown" and exit. That other thread has to
/// be scheduled against the still-running solver, and under heavy machine load
/// (e.g. many concurrent solves during a test run) it can be starved for far
/// longer than the deadline -- observed in practice at tens to hundreds of
/// seconds for a 3-second limit.
///
/// A *separate* supervisor process sidesteps that entirely: it does no solving
/// of its own, so its poll-and-kill loop has no CPU-bound sibling to compete
/// with and is scheduled promptly no matter how saturated the machine is.
/// (This is exactly the mechanism the CLI's own integration harness relies on
/// to bound the binary, which is reliable under the same load that starves an
/// in-process timer.)
///
/// The current binary is re-executed with an identical argument vector plus
/// [`TIMEOUT_SUPERVISOR_GUARD`] set, inheriting all standard streams so the
/// child's output flows straight through. On a clean finish the child's exit
/// code is propagated; if the deadline elapses first the child is killed and
/// an honest "unknown" is reported with [`TIMEOUT_EXIT_CODE`].
///
/// Returns to the caller (so it can fall back to in-process enforcement) only
/// if the supervisor could not be started at all; otherwise it terminates the
/// process and never returns.
fn supervise_timeout(args: &Args) {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(_) => return, // cannot re-exec ourselves; fall back in-process
    };
    let child_args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();

    let mut command = std::process::Command::new(exe);
    command.args(&child_args).env(TIMEOUT_SUPERVISOR_GUARD, "1");

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(_) => return, // could not spawn; fall back in-process
    };

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(args.timeout);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Propagate the child's own exit code. A child terminated by a
                // signal (no code) is surfaced as a generic failure rather than
                // a spurious success.
                std::process::exit(status.code().unwrap_or(1));
            }
            Ok(None) => {}
            Err(_) => {
                // We can no longer observe the child; do not risk hanging.
                let _ = child.kill();
                let _ = child.wait();
                std::process::exit(1);
            }
        }

        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            if args.smtcomp {
                println!("unknown");
            } else {
                println!(
                    "; timeout: solver exceeded {}s limit, reporting unknown",
                    args.timeout
                );
                println!("unknown");
            }
            use std::io::Write as _;
            let _ = std::io::stdout().flush();
            std::process::exit(TIMEOUT_EXIT_CODE);
        }

        // Poll on a short, bounded sleep rather than hot-spinning.
        //
        // The supervisor does no solving of its own, so unlike the in-process
        // watchdog it has no CPU-bound sibling to be starved behind -- a short
        // sleep here does not risk the multi-minute wakeup delays that ruled
        // out `sleep` for the abandoned-solver-thread case elsewhere in this
        // file. A `yield_now` busy-loop, by contrast, keeps this thread
        // continuously runnable and burns a full core for the entire timeout
        // window (observed: ~100% CPU on one core for the whole `--timeout`
        // duration), which both wastes power and adds to the very machine
        // load that starves other timed work. Sleeping for a small, fixed
        // interval keeps the deadline check timely (worst-case lateness is
        // one interval) while leaving the core idle between polls.
        std::thread::sleep(std::time::Duration::from_millis(15));
    }
}

/// Outcome of running a script through [`execute_script_with_timeout`].
enum ScriptOutcome {
    /// The script ran to completion (successfully or with an error) within
    /// the deadline.
    Finished(oxiz_core::error::Result<Vec<String>>),
}

/// Run `ctx.execute_script(script)` under a hard wall-clock `timeout`.
///
/// `Context`/`Solver` do expose a cooperative deadline for an in-progress
/// `check-sat` — `SolverConfig::timeout_ms` reaches both SAT engines and the
/// theory callbacks — but it is *cooperative*: it is polled between CDCL loop
/// iterations, and a single `propagate()` over a large clause database is not
/// itself interruptible, so it bounds the search without bounding the process.
/// Enforcing a *hard* wall-clock bound from the CLI therefore still requires
/// moving the solve to its own thread: if it does not finish by the deadline,
/// the thread is abandoned (never joined) and the CLI honestly reports
/// "unknown" instead of hanging forever or fabricating a sat/unsat answer. The
/// abandoned thread keeps running in the background until the process exits.
///
/// # Why a dedicated watchdog thread owns the exit
///
/// An earlier design had the *main* thread do a timed `recv_timeout(timeout)`
/// and, on timeout, print "unknown" and exit. That is fragile: the abandoned
/// solver thread is CPU-bound and churns the (lock-protected, on macOS)
/// system allocator, so when the main thread's timed wait finally fires it
/// contends for CPU *and* the malloc lock against the still-running solver.
/// Empirically this delayed the process exit by minutes (observed 150-300s
/// for a 3s deadline) under load, because the exit path allocates/formats.
///
/// Instead, a dedicated watchdog thread pre-formats its entire output as raw
/// bytes *before* the solve starts, then does nothing but `sleep(timeout)`,
/// write those bytes, and `process::exit`. Its post-wakeup path performs no
/// allocation and takes no contended lock, so it exits promptly even while
/// the abandoned solver thread saturates a core. The main thread blocks on an
/// untimed `recv()`; whichever of {solve completes, watchdog fires} happens
/// first wins via `outcome_claimed`.
///
/// On success (or a script-level error) within the deadline, `ctx` is
/// restored to reflect any state changes the script made (declarations,
/// assertions, the last model, etc.) exactly as a direct
/// `ctx.execute_script(script)` call would have.
fn execute_script_with_timeout(
    ctx: &mut Context,
    script: &str,
    timeout: std::time::Duration,
    timeout_secs: u64,
    smtcomp: bool,
) -> ScriptOutcome {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let owned_ctx = std::mem::replace(ctx, Context::new());
    let script_owned = script.to_string();
    let (tx, rx) = std::sync::mpsc::channel();

    // The first of {main thread receives a result, watchdog deadline fires}
    // to flip this from `false` owns the outcome. This makes the near-deadline
    // race (solve finishes within a hair of the timeout) deterministic instead
    // of double-reporting.
    let outcome_claimed = Arc::new(AtomicBool::new(false));

    let spawn_result = std::thread::Builder::new()
        .name("oxiz-solve".to_string())
        .spawn(move || {
            let mut owned_ctx = owned_ctx;
            let result = owned_ctx.execute_script(&script_owned);
            // Best-effort: if the receiver already gave up (deadline passed
            // and nobody is listening any more), there's nothing to do.
            let _ = tx.send((owned_ctx, result));
        });

    let handle = match spawn_result {
        Ok(handle) => handle,
        Err(e) => {
            // Could not spawn the solver thread at all (OS resource
            // exhaustion). The context that was moved into the failed
            // closure is gone, so `ctx` now holds the fresh placeholder from
            // `mem::replace` above. Report this honestly rather than
            // silently discarding state or fabricating a result.
            return ScriptOutcome::Finished(Err(oxiz_core::error::OxizError::Internal(format!(
                "failed to spawn solver watchdog thread: {e}"
            ))));
        }
    };

    // Pre-format the timeout output as raw bytes now, on the main thread,
    // while nothing is time-critical. The watchdog must not allocate after it
    // wakes (see the doc comment above), so everything it needs is baked in
    // here.
    let timeout_bytes: Vec<u8> = if smtcomp {
        b"unknown\n".to_vec()
    } else {
        format!("; timeout: solver exceeded {timeout_secs}s limit, reporting unknown\nunknown\n")
            .into_bytes()
    };

    let watchdog_claimed = Arc::clone(&outcome_claimed);
    // A best-effort watchdog: if it cannot be spawned we fall back to an
    // untimed wait below (honest: the solve still runs, we just lose the hard
    // bound rather than fabricating an answer).
    let _watchdog = std::thread::Builder::new()
        .name("oxiz-timeout".to_string())
        .spawn(move || {
            std::thread::sleep(timeout);
            if watchdog_claimed.swap(true, Ordering::SeqCst) {
                // The solve already completed and the main thread claimed the
                // outcome first; nothing to do.
                return;
            }
            // We own the outcome: report "unknown" and terminate immediately.
            // Use a raw `write_all` of pre-built bytes (no formatting, no
            // allocation) so this stays off the contended allocator path.
            use std::io::Write as _;
            let stdout = std::io::stdout();
            let mut lock = stdout.lock();
            let _ = lock.write_all(&timeout_bytes);
            let _ = lock.flush();
            std::process::exit(TIMEOUT_EXIT_CODE);
        });

    match rx.recv() {
        Ok((mut returned_ctx, result)) => {
            if outcome_claimed.swap(true, Ordering::SeqCst) {
                // The watchdog already fired and is about to (or did) call
                // `process::exit`; do not also emit a result. Park so we do
                // not race the exit with a second line of output.
                loop {
                    std::thread::park();
                }
            }
            std::mem::swap(ctx, &mut returned_ctx);
            // The solve finished; join is now effectively instantaneous.
            let _ = handle.join();
            ScriptOutcome::Finished(result)
        }
        Err(std::sync::mpsc::RecvError) => {
            // The spawned thread died (panicked) without sending a result.
            let _ = outcome_claimed.swap(true, Ordering::SeqCst);
            ScriptOutcome::Finished(Err(oxiz_core::error::OxizError::Internal(
                "solver thread terminated unexpectedly".to_string(),
            )))
        }
    }
}

/// Cap on additional models enumerated when `--enumerate-models` is used
/// with `--max-models 0` ("unlimited"). Formulas over unbounded domains
/// (`Int`/`Real`) have no natural enumeration bound, so an unbounded loop
/// here would hang the CLI on an explicit "find them all" request; this
/// gives that request a real (if large) stopping point instead of silently
/// refusing to honor it.
const UNLIMITED_ENUMERATION_SAFETY_CAP: usize = 1000;

/// Wall-clock ceiling on the total time spent enumerating additional models
/// beyond the first, so a pathological formula cannot hang the CLI
/// indefinitely even before [`UNLIMITED_ENUMERATION_SAFETY_CAP`] is reached
/// (each round re-solves a growing assertion set, so later rounds get
/// steadily more expensive).
const ENUMERATION_WALL_CLOCK_BUDGET: std::time::Duration = std::time::Duration::from_secs(10);

/// Reconstructs a [`oxiz_core::sort::SortId`] from the subset of
/// [`Context::get_model`]'s sort-name strings [`enumerate_additional_models`]
/// knows how to re-assert equalities over: `Bool`, `Int`, `Real`, and
/// fixed-width `(_ BitVec n)`. Returns `None` for anything else (arrays,
/// datatypes, uninterpreted sorts, strings, floats, ...), which the caller
/// treats the same as an unassigned variable: honestly excluded from the
/// blocking clause rather than guessed at.
fn resolve_basic_sort(ctx: &mut Context, sort_name: &str) -> Option<oxiz_core::sort::SortId> {
    match sort_name {
        "Bool" => Some(ctx.terms.sorts.bool_sort),
        "Int" => Some(ctx.terms.sorts.int_sort),
        "Real" => Some(ctx.terms.sorts.real_sort),
        _ => {
            let width_str = sort_name
                .strip_prefix("(_ BitVec ")?
                .strip_suffix(')')?
                .trim();
            let width: u32 = width_str.parse().ok()?;
            Some(ctx.terms.sorts.bitvec(width))
        }
    }
}

/// Enumerate every satisfying model of the script that led here, up to
/// `--max-models` (`0` means "unlimited", capped by
/// [`UNLIMITED_ENUMERATION_SAFETY_CAP`]), and return the formatted output
/// for all of them (including the one the normal `check-sat` already
/// found -- callers do not need to separately request `(get-model)` for it).
///
/// Implements the classic blocking-clause technique: read back the current
/// model, assert the negation of that exact assignment, and `check-sat`
/// again; each additional `sat` result is one more distinct model, and
/// `unsat` means every model has been found. The blocking equalities are
/// built directly as terms (`TermManager::mk_var`/`mk_eq`/`mk_and`/`mk_not`)
/// rather than through a re-parsed SMT-LIB snippet: `Context::execute_script`
/// parses each call with a fresh, script-local symbol table (see
/// `oxiz_core::smtlib::parser::terms`), so a *separate* `execute_script`
/// call referencing a constant declared by an *earlier* call would not
/// resolve it -- `TermManager::mk_var` is hash-consed on `(name, sort)`,
/// so reconstructing that same pair here yields the identical `TermId` the
/// original `declare-const` produced, with no re-parsing and no risk of
/// re-registering a duplicate declaration.
///
/// The enumeration loop runs inside a single `push`/`pop` pair so the extra
/// blocking assertions (and the `last_result`/model they leave behind)
/// never leak into whatever the caller does with `ctx` afterwards -- the
/// textual model output collected along the way, not any residual solver
/// state, is what the caller actually surfaces to the user.
///
/// A variable whose sort [`resolve_basic_sort`] does not recognize, or that
/// has no assignment to evaluate, is excluded from the blocking clause. If
/// every declared constant in the current model falls into that category,
/// enumeration honestly stops (rather than asserting a vacuous `(not (and))`
/// blocking clause, which would misreport "no more models" after just one
/// round).
fn enumerate_additional_models(ctx: &mut Context, max_models: usize) -> Vec<String> {
    let requested_cap = if max_models == 0 {
        UNLIMITED_ENUMERATION_SAFETY_CAP
    } else {
        max_models
    };
    let mut lines = Vec::new();

    // The caller only invokes this after confirming the script's own
    // `check-sat` reported `sat`, so a model should always be available
    // here; stay honest rather than panicking if that invariant is ever
    // broken.
    if ctx.get_model().is_none() {
        return lines;
    }
    lines.push("; model 1:".to_string());
    lines.push(ctx.format_model());

    if requested_cap <= 1 {
        lines.push(format!(
            "; model enumeration stopped after 1 model(s): reached --max-models {max_models}"
        ));
        return lines;
    }

    ctx.push();
    // `push` is a state-changing command: per SMT-LIB 2.6 §4.1.1 it returns the
    // solver to `assert` mode, so the cached check result (and with it
    // `get_model`/`eval_in_model`) is invalidated.  Re-establish `sat` inside
    // the fresh scope — the assertion set is unchanged, so this simply restores
    // the model the loop below reads to build its first blocking clause.
    if ctx.check_sat() != oxiz_solver::SolverResult::Sat {
        ctx.pop();
        lines.push("; model enumeration stopped after 1 model(s): no model available".to_string());
        return lines;
    }
    let start = std::time::Instant::now();
    let mut found = 1usize;
    let mut stop_reason: Option<String> = None;

    while found < requested_cap {
        if start.elapsed() > ENUMERATION_WALL_CLOCK_BUDGET {
            stop_reason = Some(format!(
                "; model enumeration stopped after {found} model(s): wall-clock budget \
                 ({:.0}s) exceeded",
                ENUMERATION_WALL_CLOCK_BUDGET.as_secs_f64()
            ));
            break;
        }

        let Some(model) = ctx.get_model() else {
            stop_reason = Some(format!(
                "; model enumeration stopped after {found} model(s): no model available"
            ));
            break;
        };

        let mut equalities = Vec::new();
        for (name, sort_name, _value) in &model {
            let Some(sort_id) = resolve_basic_sort(ctx, sort_name) else {
                continue;
            };
            let var_term = ctx.terms.mk_var(name, sort_id);
            let Some(value_term) = ctx.eval_in_model(var_term) else {
                continue;
            };
            // A variable with no real assignment in the solver's model
            // (e.g. genuinely unconstrained -- declared but never
            // decided by any clause) evaluates to itself: `Model::eval`'s
            // fallback for an unassigned `Var` returns the variable term
            // unchanged. Building `(= var var)` on that pair would
            // silently constant-fold to `true` (`TermManager::mk_eq`
            // short-circuits reflexive equalities), and negating it into
            // the blocking clause would assert `false` -- reporting "no
            // further models" after the very next round regardless of
            // how many models actually remain. Exclude it instead: an
            // honest inability to pin this variable down for blocking,
            // not a fabricated constraint.
            if value_term == var_term {
                continue;
            }
            equalities.push(ctx.terms.mk_eq(var_term, value_term));
        }

        if equalities.is_empty() {
            stop_reason = Some(format!(
                "; model enumeration stopped after {found} model(s): no variable in the model \
                 has a sort this enumerator can re-assert as a blocking literal"
            ));
            break;
        }

        let conjunction = ctx.terms.mk_and(equalities);
        let blocking = ctx.terms.mk_not(conjunction);
        ctx.assert(blocking);

        match ctx.check_sat() {
            oxiz_solver::SolverResult::Sat => {
                found += 1;
                lines.push(format!("; model {found}:"));
                lines.push(ctx.format_model());
            }
            oxiz_solver::SolverResult::Unsat => {
                stop_reason = Some(format!(
                    "; model enumeration complete: {found} model(s) total (no further \
                     models exist)"
                ));
                break;
            }
            oxiz_solver::SolverResult::Unknown => {
                stop_reason = Some(format!(
                    "; model enumeration stopped after {found} model(s): solver returned \
                     unknown"
                ));
                break;
            }
        }
    }

    match stop_reason {
        Some(reason) => lines.push(reason),
        None if max_models == 0 => lines.push(format!(
            "; model enumeration stopped after {found} model(s): reached the internal safety \
             cap of {UNLIMITED_ENUMERATION_SAFETY_CAP} (no --max-models limit was given)"
        )),
        None => lines.push(format!(
            "; model enumeration stopped after {found} model(s): reached --max-models {max_models}"
        )),
    }

    ctx.pop();
    lines
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
            Err(e) => format!(
                "(error {})",
                oxiz_core::smtlib::format_string_literal(&format!("Validation failed: {}", e))
            ),
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

    // --resume/--resume-from: replay a previously-checkpointed result for this
    // exact problem instead of re-solving, when a matching checkpoint exists.
    if (args.resume || args.resume_from.is_some())
        && let Some(lines) = checkpointing::try_resume(script, args)
    {
        if args.verbosity >= Verbosity::Verbose {
            eprintln!("; resumed result from checkpoint");
        }
        return lines.join("\n");
    }

    // Portfolio / parallel routing. Reached when the user asked for the
    // portfolio strategy (by name, `--preset`, or `--auto-tune`), enabled
    // `--portfolio-mode`, OR set an explicit `--threads N` (N != the clap
    // default) without `--parallel` file processing — in which case a single
    // problem is solved by N portfolio workers (the real wiring of `--threads`
    // into the parallel-solving entry point). `--strategy portfolio` requested
    // by name thus dispatches to the real parallel-portfolio solver instead of
    // silently falling through to a single-strategy solve.
    let strategy_requests_portfolio = args.strategy.as_deref() == Some("portfolio")
        || ctx.get_option("strategy") == Some("portfolio");
    let threads_request_parallel =
        args.threads != DEFAULT_THREADS && args.threads >= 1 && !args.parallel;
    if args.portfolio_mode || strategy_requests_portfolio || threads_request_parallel {
        let timeout = if args.portfolio_timeout > 0 {
            args.portfolio_timeout
        } else if args.timeout > 0 {
            args.timeout
        } else {
            300 // Default 5 minutes
        };

        let logic = args.logic.as_deref();
        // An explicit `--threads N` bounds the number of concurrent worker
        // strategies to N; otherwise the full default strategy set runs.
        let strategies = if args.threads != DEFAULT_THREADS && args.threads >= 1 {
            portfolio::strategies_for_thread_count(args.threads)
        } else {
            portfolio::get_default_strategies()
        };
        let worker_count = strategies.len();
        match portfolio::solve_portfolio_custom(script, strategies, args, logic, ctx, timeout) {
            Ok(result) => {
                // Format the output with strategy information
                let mut output = result.output;
                if args.verbosity >= Verbosity::Verbose {
                    output.insert(
                        0,
                        format!(
                            "; Portfolio solver: {} won in {}ms ({} worker strategies)",
                            result.strategy_name, result.time_ms, worker_count
                        ),
                    );
                }
                return output.join("\n");
            }
            Err(e) => {
                return format!(
                    "(error {})",
                    oxiz_core::smtlib::format_string_literal(&format!(
                        "Portfolio solving failed: {}",
                        e
                    ))
                );
            }
        }
    }

    // ML-guided tactic selection (opt-in via `--ml-tactic-selection`).
    // Extract formula features, apply a conservative (correctness-preserving)
    // solver option for the recommended tactic, and remember the session so
    // the outcome can be fed back after the solve. Off by default → no effect.
    let mut ml_session = if args.ml_tactic_selection {
        Some(ml_tactic::begin(ctx, script, args))
    } else {
        None
    };
    let ml_start = std::time::Instant::now();

    // Enforce `--timeout`/config-file timeout on the normal solving path.
    // `0` means "no timeout" (the documented default), matching the existing
    // portfolio-mode convention, so the common unbounded case pays no thread
    // overhead at all.
    let script_result = if args.timeout > 0 {
        // On timeout the dedicated watchdog thread inside
        // `execute_script_with_timeout` prints "unknown" and terminates the
        // process with `TIMEOUT_EXIT_CODE`, so this call only ever returns
        // `Finished` (the solve completed within the deadline, or the solver
        // thread failed). See that function's doc comment for why the exit is
        // owned by a pre-formatted, allocation-free watchdog rather than
        // threaded back up through the formatting/printing path here.
        match execute_script_with_timeout(
            ctx,
            script,
            std::time::Duration::from_secs(args.timeout),
            args.timeout,
            args.smtcomp,
        ) {
            ScriptOutcome::Finished(result) => result,
        }
    } else {
        ctx.execute_script(script)
    };

    match script_result {
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

                // Actually validate the model: evaluate every top-level
                // assertion against it and confirm each one holds.
                // Previously this branch only printed the model and never
                // checked anything against it, so "--validate-model" could
                // not distinguish a genuinely sound model from a solver bug.
                let true_id = ctx.terms.mk_true();
                let assertions: Vec<_> = ctx.get_assertions().to_vec();
                let total = assertions.len();
                let mut unverified = 0usize;
                for term in assertions {
                    match ctx.eval_in_model(term) {
                        Some(value) if value == true_id => {}
                        _ => unverified += 1,
                    }
                }
                let validation_message = if unverified == 0 {
                    format!(
                        "; model validation: OK ({total} assertion(s) hold under the reported model)"
                    )
                } else {
                    format!(
                        "; model validation: FAILED ({unverified} of {total} assertion(s) do not \
                         evaluate to true under the reported model)"
                    )
                };
                output.push(validation_message);
            }

            // Handle bounded model enumeration if requested. Only makes
            // sense to run when the script's own `check-sat` reported
            // `sat` -- an `unsat`/`unknown` result has no model to
            // enumerate additional solutions from.
            if args.enumerate_models
                && output
                    .iter()
                    .any(|line| line.contains("sat") && !line.contains("unsat"))
            {
                let extra = enumerate_additional_models(ctx, args.max_models);
                output.extend(extra);
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

            // Real minimal-unsat-core search (deletion-based) when requested.
            // Runs after every ctx-reading post-step above, since it resets and
            // re-asserts subsets of the problem to prove minimality.
            if args.minimize_core && output.iter().any(|line| line.trim() == "unsat") {
                let extra = core_min::minimize_core(ctx);
                output.extend(extra);
            }

            // Close out the ML session: attribute the outcome to the chosen
            // tactic (a definite sat/unsat is a success) and surface the
            // recommendation as a comment on the output.
            if let Some(session) = ml_session.take() {
                let was_successful = output
                    .iter()
                    .any(|line| matches!(line.trim(), "sat" | "unsat"));
                let comment = session.comment().to_string();
                session.finish(was_successful, ml_start.elapsed());
                output.insert(0, comment);
            }

            // Persist a resumable checkpoint of this completed solve if asked.
            if args.checkpoint {
                checkpointing::write(script, args, ctx, &output);
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
        Err(e) => {
            // Close out any ML session even on a failed solve (records the
            // unsuccessful outcome so the model does not over-credit the tactic).
            if let Some(session) = ml_session.take() {
                session.finish(false, ml_start.elapsed());
            }
            if args.enhanced_errors {
                // `OxizError::enhance` (oxiz-core) attaches a source-context
                // snippet, a "did you mean?" suggestion (against currently
                // declared symbols), and an actionable hint. It already
                // existed in oxiz-core but `--enhanced-errors` never called
                // it, so the flag was accepted and silently did nothing.
                let known_symbols: Vec<&str> = ctx.declared_function_names().collect();
                let enhanced = e.enhance(Some(script), &known_symbols);
                format!(
                    "(error {})",
                    oxiz_core::smtlib::format_string_literal(&enhanced.to_string())
                )
            } else {
                format!(
                    "(error {})",
                    oxiz_core::smtlib::format_string_literal(&e.to_string())
                )
            }
        }
    }
}
