//! OxiZ SMT-COMP - Benchmark Infrastructure
//!
//! This crate provides infrastructure for running SMT-COMP style benchmarks,
//! analyzing results, and generating reports compatible with the SMT competition.
//!
//! # Features
//!
//! - Benchmark discovery and loading by logic
//! - Configurable benchmark runner with timeout support
//! - Parallel benchmark execution using rayon
//! - SMT-COMP compatible result reporting (JSON, CSV, Text)
//! - Statistical analysis and solver comparison
//! - Memory limit enforcement
//! - Model verification for SAT results
//! - StarExec format compatibility
//! - Cactus plot and visualization generation
//! - HTML report and dashboard generation
//! - Incremental result saving and resumption
//! - Benchmark filtering and sampling
//! - Virtual best solver calculation
//! - Performance regression detection
//! - CI/CD integration support
//!
//! # Examples
//!
//! ## Basic Benchmark Run
//!
//! ```no_run
//! use oxiz_smtcomp::{Loader, LoaderConfig, Runner, RunnerConfig, Reporter};
//! use std::time::Duration;
//!
//! // Discover benchmarks
//! let config = LoaderConfig::new("/path/to/benchmarks")
//!     .with_logics(vec!["QF_LIA".to_string()])
//!     .with_max_files(100);
//! let loader = Loader::new(config);
//! let benchmarks = loader.discover().expect("Failed to discover benchmarks");
//!
//! // Run benchmarks
//! let runner = Runner::with_timeout(Duration::from_secs(60));
//! let results = runner.run_from_meta(&benchmarks, &loader);
//!
//! // Generate report
//! let reporter = Reporter::text();
//! let report = reporter.to_string(&results).expect("Failed to generate report");
//! println!("{}", report);
//! ```
//!
//! ## Parallel Execution
//!
//! ```no_run
//! use oxiz_smtcomp::parallel::{ParallelRunner, ParallelConfig};
//! use oxiz_smtcomp::Loader;
//! use std::time::Duration;
//!
//! let config = ParallelConfig::new(Duration::from_secs(60))
//!     .with_num_threads(4);
//! let runner = ParallelRunner::new(config);
//! // let results = runner.run_all(&benchmarks);
//! ```
//!
//! ## Comparing Solvers
//!
//! ```
//! use oxiz_smtcomp::{SolverComparison, SingleResult, BenchmarkStatus, BenchmarkMeta};
//! use oxiz_smtcomp::loader::ExpectedStatus;
//! use std::time::Duration;
//! use std::path::PathBuf;
//!
//! // Assuming you have results from two solver runs
//! let meta = BenchmarkMeta {
//!     path: PathBuf::from("/tmp/test.smt2"),
//!     logic: Some("QF_LIA".to_string()),
//!     expected_status: Some(ExpectedStatus::Sat),
//!     file_size: 100,
//!     category: None,
//! };
//!
//! let results_a = vec![SingleResult::new(&meta, BenchmarkStatus::Sat, Duration::from_millis(100))];
//! let results_b = vec![SingleResult::new(&meta, BenchmarkStatus::Sat, Duration::from_millis(200))];
//!
//! let comparison = SolverComparison::compare("SolverA", &results_a, "SolverB", &results_b);
//! println!("{}", comparison.summary());
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Core modules
pub mod benchmark;
pub mod loader;
pub mod reporter;
pub mod statistics;

// High priority modules
pub mod memory;
pub mod model_verify;
pub mod parallel;
pub mod starexec;

// Medium priority modules
pub mod filtering;
pub mod html_report;
pub mod plotting;
pub mod resumption;
pub mod virtual_best;

// Low priority modules
pub mod ci_integration;
pub mod dashboard;
pub mod regression;
pub mod sampling;

// Re-export main types for convenience
pub use benchmark::{
    BenchmarkError, BenchmarkResult, BenchmarkStatus, RunSummary, Runner, RunnerConfig,
    SingleResult,
};

pub use loader::{Benchmark, BenchmarkMeta, Loader, LoaderConfig, LoaderError, LoaderResult};

pub use reporter::{
    Report, ReportFormat, Reporter, ReporterConfig, ReporterError, ReporterResult, ResultEntry,
    SmtCompScore, SolverInfo,
};

pub use statistics::{
    CactusPoint, CategoryStats, DifficultyAnalysis, FullAnalysis, ScatterPoint, SolverComparison,
    Statistics, cactus_data, scatter_data,
};

// Re-export parallel types
pub use parallel::{ParallelConfig, ParallelProgress, ParallelRunner};

// Re-export memory types
pub use memory::{MemoryLimit, MemoryLimiter, MemoryMonitor, MemoryUsage};

// Re-export model verification types
pub use model_verify::{Model, ModelVerifier, VerificationResult, VerificationSummary};

// Re-export StarExec types
pub use starexec::{
    StarExecBenchmarkResult, StarExecJob, StarExecOutput, StarExecReader, StarExecWriter,
};

// Re-export plotting types
pub use plotting::{
    Color, DataSeries, PlotConfig, SvgPlot, generate_cactus_plot, generate_scatter_plot,
};

// Re-export HTML report types
pub use html_report::{HtmlReportConfig, HtmlReportGenerator, generate_comparison_report};

// Re-export resumption types
pub use resumption::{
    Checkpoint, CheckpointConfig, ResultLoader, ResultSaver, SessionInfo, SessionManager,
};

// Re-export filtering types
pub use filtering::{
    ExpectedStatusFilter, FilterCriteria, ResultFilterCriteria, filter_benchmarks, filter_results,
};

// Re-export virtual best types
pub use virtual_best::{
    VirtualBestResult, VirtualBestSolver, VirtualBestStats, calculate_virtual_best,
};

// Re-export CI integration types
pub use ci_integration::{
    CiConfig, CiConfigGenerator, CiSystem, generate_github_summary, generate_junit_xml,
};

// Re-export sampling types
pub use sampling::{SampleSummary, Sampler, SamplingConfig, SamplingStrategy};

// Re-export regression types
pub use regression::{
    RegressionAnalysis, RegressionConfig, RegressionDetector, RegressionFinding, RegressionType,
};

// Re-export dashboard types
pub use dashboard::{DashboardConfig, DashboardData, DashboardGenerator, generate_dashboard};
