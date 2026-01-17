//! OxiZ SMT-COMP - Benchmark Infrastructure
//!
//! This crate provides infrastructure for running SMT-COMP style benchmarks,
//! analyzing results, and generating reports compatible with the SMT competition.
//!
//! # Features
//!
//! - Benchmark discovery and loading by logic
//! - Configurable benchmark runner with timeout support
//! - SMT-COMP compatible result reporting (JSON, CSV, Text)
//! - Statistical analysis and solver comparison
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

pub mod benchmark;
pub mod loader;
pub mod reporter;
pub mod statistics;

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
