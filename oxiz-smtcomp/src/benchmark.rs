//! Benchmark runner with timeout support
//!
//! This module provides the core benchmark execution engine that runs SMT-LIB2
//! benchmarks with configurable timeout and resource limits.

use crate::loader::{Benchmark, BenchmarkMeta, ExpectedStatus};
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::{Command, parse_script};
use oxiz_solver::{Solver, SolverResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Error type for benchmark operations
#[derive(Error, Debug)]
pub enum BenchmarkError {
    /// Parse error in benchmark file
    #[error("Parse error in {path}: {message}")]
    ParseError {
        /// File path
        path: PathBuf,
        /// Error message
        message: String,
    },
    /// Solver error
    #[error("Solver error: {0}")]
    SolverError(String),
    /// Timeout reached
    #[error("Timeout after {0:?}")]
    Timeout(Duration),
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for benchmark operations
pub type BenchmarkResult<T> = Result<T, BenchmarkError>;

/// Result status from running a benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchmarkStatus {
    /// Solver returned SAT
    Sat,
    /// Solver returned UNSAT
    Unsat,
    /// Solver returned unknown
    Unknown,
    /// Benchmark timed out
    Timeout,
    /// Benchmark hit memory limit
    MemoryOut,
    /// Parse or other error occurred
    Error,
}

impl BenchmarkStatus {
    /// Convert to SMT-COMP format string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sat => "sat",
            Self::Unsat => "unsat",
            Self::Unknown => "unknown",
            Self::Timeout => "timeout",
            Self::MemoryOut => "memout",
            Self::Error => "error",
        }
    }

    /// Check if this result matches the expected status
    #[must_use]
    pub fn matches_expected(&self, expected: &ExpectedStatus) -> Option<bool> {
        match (self, expected) {
            (Self::Sat, ExpectedStatus::Sat) => Some(true),
            (Self::Unsat, ExpectedStatus::Unsat) => Some(true),
            (Self::Sat, ExpectedStatus::Unsat) | (Self::Unsat, ExpectedStatus::Sat) => Some(false),
            _ => None, // Unknown/Timeout/Error don't count as correct or incorrect
        }
    }
}

impl From<SolverResult> for BenchmarkStatus {
    fn from(result: SolverResult) -> Self {
        match result {
            SolverResult::Sat => Self::Sat,
            SolverResult::Unsat => Self::Unsat,
            SolverResult::Unknown => Self::Unknown,
        }
    }
}

/// Result of running a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleResult {
    /// Path to the benchmark file
    pub path: PathBuf,
    /// Logic of the benchmark
    pub logic: Option<String>,
    /// Expected status (if known)
    pub expected: Option<ExpectedStatus>,
    /// Actual result status
    pub status: BenchmarkStatus,
    /// Time taken to solve
    pub time: Duration,
    /// Whether result matches expected (None if not comparable)
    pub correct: Option<bool>,
    /// Error message if any
    pub error_message: Option<String>,
    /// Memory used (if tracked)
    pub memory_bytes: Option<u64>,
}

impl SingleResult {
    /// Create a new result
    #[must_use]
    pub fn new(meta: &BenchmarkMeta, status: BenchmarkStatus, time: Duration) -> Self {
        let correct = meta
            .expected_status
            .and_then(|exp| status.matches_expected(&exp));

        Self {
            path: meta.path.clone(),
            logic: meta.logic.clone(),
            expected: meta.expected_status,
            status,
            time,
            correct,
            error_message: None,
            memory_bytes: None,
        }
    }

    /// Create an error result
    #[must_use]
    pub fn error(meta: &BenchmarkMeta, time: Duration, message: String) -> Self {
        Self {
            path: meta.path.clone(),
            logic: meta.logic.clone(),
            expected: meta.expected_status,
            status: BenchmarkStatus::Error,
            time,
            correct: None,
            error_message: Some(message),
            memory_bytes: None,
        }
    }

    /// Check if this result is sound (no wrong answers)
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.correct != Some(false)
    }
}

/// Configuration for the benchmark runner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    /// Timeout per benchmark
    pub timeout: Duration,
    /// Memory limit in bytes (0 = unlimited)
    pub memory_limit: u64,
    /// Logic to set if not specified in benchmark
    pub default_logic: Option<String>,
    /// Whether to continue on error
    pub continue_on_error: bool,
    /// Number of parallel workers (1 = sequential)
    pub num_workers: usize,
    /// Whether to verify models
    pub verify_models: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            memory_limit: 0,
            default_logic: None,
            continue_on_error: true,
            num_workers: 1,
            verify_models: false,
        }
    }
}

impl RunnerConfig {
    /// Create a new config with the given timeout
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Default::default()
        }
    }

    /// Set the memory limit
    #[must_use]
    pub fn with_memory_limit(mut self, limit: u64) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Set the default logic
    #[must_use]
    pub fn with_default_logic(mut self, logic: impl Into<String>) -> Self {
        self.default_logic = Some(logic.into());
        self
    }

    /// Set whether to continue on error
    #[must_use]
    pub fn with_continue_on_error(mut self, cont: bool) -> Self {
        self.continue_on_error = cont;
        self
    }

    /// Set the number of parallel workers
    #[must_use]
    pub fn with_num_workers(mut self, n: usize) -> Self {
        self.num_workers = n.max(1);
        self
    }
}

/// Benchmark runner for executing SMT-LIB2 benchmarks
pub struct Runner {
    config: RunnerConfig,
}

impl Runner {
    /// Create a new runner with the given configuration
    #[must_use]
    pub fn new(config: RunnerConfig) -> Self {
        Self { config }
    }

    /// Create a runner with default configuration
    #[must_use]
    pub fn with_timeout(timeout: Duration) -> Self {
        Self::new(RunnerConfig::new(timeout))
    }

    /// Run a single benchmark
    pub fn run_benchmark(&self, benchmark: &Benchmark) -> SingleResult {
        let start = Instant::now();
        let deadline = start + self.config.timeout;

        debug!("Running benchmark: {}", benchmark.meta.path.display());

        // Parse the benchmark
        let mut tm = TermManager::new();
        let commands = match parse_script(&benchmark.content, &mut tm) {
            Ok(cmds) => cmds,
            Err(e) => {
                warn!("Parse error in {}: {}", benchmark.meta.path.display(), e);
                return SingleResult::error(&benchmark.meta, start.elapsed(), e.to_string());
            }
        };

        // Create solver
        let mut solver = Solver::new();

        // Set default logic if needed
        let mut logic_set = false;

        // Execute commands
        for cmd in commands {
            // Check timeout
            if Instant::now() > deadline {
                info!("Timeout in {}", benchmark.meta.path.display());
                return SingleResult::new(
                    &benchmark.meta,
                    BenchmarkStatus::Timeout,
                    start.elapsed(),
                );
            }

            match cmd {
                Command::SetLogic(logic) => {
                    solver.set_logic(&logic);
                    logic_set = true;
                }
                Command::DeclareConst(name, sort_str) => {
                    let sort = self.parse_sort(&sort_str, &tm);
                    let _var = tm.mk_var(&name, sort);
                }
                Command::DeclareFun(name, arg_sorts, ret_sort) => {
                    if arg_sorts.is_empty() {
                        let sort = self.parse_sort(&ret_sort, &tm);
                        let _var = tm.mk_var(&name, sort);
                    }
                    // Functions with arguments need uninterpreted function support
                }
                Command::Assert(term) => {
                    solver.assert(term, &mut tm);
                }
                Command::CheckSat => {
                    // Set default logic if not set
                    if !logic_set {
                        if let Some(ref logic) = self.config.default_logic {
                            solver.set_logic(logic);
                        } else if let Some(ref logic) = benchmark.meta.logic {
                            solver.set_logic(logic);
                        }
                    }

                    // Solve with remaining timeout
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return SingleResult::new(
                            &benchmark.meta,
                            BenchmarkStatus::Timeout,
                            start.elapsed(),
                        );
                    }

                    // Run solver
                    let result = solver.check(&mut tm);
                    let status = BenchmarkStatus::from(result);

                    return SingleResult::new(&benchmark.meta, status, start.elapsed());
                }
                Command::Push(n) => {
                    for _ in 0..n {
                        solver.push();
                    }
                }
                Command::Pop(n) => {
                    for _ in 0..n {
                        solver.pop();
                    }
                }
                Command::Exit => {
                    break;
                }
                Command::Reset => {
                    solver = Solver::new();
                    tm = TermManager::new();
                    logic_set = false;
                }
                // Ignore other commands for benchmarking
                _ => {}
            }
        }

        // If we got here without a check-sat, return unknown
        SingleResult::new(&benchmark.meta, BenchmarkStatus::Unknown, start.elapsed())
    }

    /// Run multiple benchmarks
    pub fn run_all(&self, benchmarks: &[Benchmark]) -> Vec<SingleResult> {
        benchmarks.iter().map(|b| self.run_benchmark(b)).collect()
    }

    /// Run benchmarks from metadata (loading content as needed)
    pub fn run_from_meta(
        &self,
        meta_list: &[BenchmarkMeta],
        loader: &crate::loader::Loader,
    ) -> Vec<SingleResult> {
        let mut results = Vec::with_capacity(meta_list.len());

        for meta in meta_list {
            match loader.load(meta) {
                Ok(benchmark) => {
                    let result = self.run_benchmark(&benchmark);
                    results.push(result);
                }
                Err(e) => {
                    let elapsed = Duration::ZERO;
                    results.push(SingleResult::error(meta, elapsed, e.to_string()));
                }
            }
        }

        results
    }

    /// Parse a sort string to sort ID
    fn parse_sort(&self, sort_str: &str, tm: &TermManager) -> oxiz_core::sort::SortId {
        match sort_str {
            "Bool" => tm.sorts.bool_sort,
            "Int" => tm.sorts.int_sort,
            "Real" => tm.sorts.real_sort,
            _ => tm.sorts.bool_sort, // Default fallback
        }
    }
}

/// Summary of benchmark run results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RunSummary {
    /// Total number of benchmarks
    pub total: usize,
    /// Number of SAT results
    pub sat: usize,
    /// Number of UNSAT results
    pub unsat: usize,
    /// Number of unknown results
    pub unknown: usize,
    /// Number of timeouts
    pub timeouts: usize,
    /// Number of errors
    pub errors: usize,
    /// Number of memory outs
    pub memouts: usize,
    /// Number of correct results (matching expected)
    pub correct: usize,
    /// Number of incorrect results (wrong answer)
    pub incorrect: usize,
    /// Total time spent
    pub total_time: Duration,
    /// Average time per benchmark
    pub avg_time: Duration,
}

impl RunSummary {
    /// Create a summary from a list of results
    #[must_use]
    pub fn from_results(results: &[SingleResult]) -> Self {
        let mut summary = Self {
            total: results.len(),
            ..Default::default()
        };

        let mut total_nanos = 0u128;

        for result in results {
            total_nanos += result.time.as_nanos();

            match result.status {
                BenchmarkStatus::Sat => summary.sat += 1,
                BenchmarkStatus::Unsat => summary.unsat += 1,
                BenchmarkStatus::Unknown => summary.unknown += 1,
                BenchmarkStatus::Timeout => summary.timeouts += 1,
                BenchmarkStatus::Error => summary.errors += 1,
                BenchmarkStatus::MemoryOut => summary.memouts += 1,
            }

            match result.correct {
                Some(true) => summary.correct += 1,
                Some(false) => summary.incorrect += 1,
                None => {}
            }
        }

        summary.total_time = Duration::from_nanos(total_nanos as u64);
        if !results.is_empty() {
            summary.avg_time = Duration::from_nanos((total_nanos / results.len() as u128) as u64);
        }

        summary
    }

    /// Check if all results are sound (no wrong answers)
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.incorrect == 0
    }

    /// Get the solved count (SAT + UNSAT)
    #[must_use]
    pub fn solved(&self) -> usize {
        self.sat + self.unsat
    }

    /// Get the solve rate as a percentage
    #[must_use]
    pub fn solve_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.solved() as f64 / self.total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_status_as_str() {
        assert_eq!(BenchmarkStatus::Sat.as_str(), "sat");
        assert_eq!(BenchmarkStatus::Unsat.as_str(), "unsat");
        assert_eq!(BenchmarkStatus::Timeout.as_str(), "timeout");
    }

    #[test]
    fn test_benchmark_status_matches_expected() {
        assert_eq!(
            BenchmarkStatus::Sat.matches_expected(&ExpectedStatus::Sat),
            Some(true)
        );
        assert_eq!(
            BenchmarkStatus::Sat.matches_expected(&ExpectedStatus::Unsat),
            Some(false)
        );
        assert_eq!(
            BenchmarkStatus::Timeout.matches_expected(&ExpectedStatus::Sat),
            None
        );
    }

    #[test]
    fn test_run_summary_from_results() {
        let meta = BenchmarkMeta {
            path: PathBuf::from("/tmp/test.smt2"),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };

        let results = vec![
            SingleResult::new(&meta, BenchmarkStatus::Sat, Duration::from_millis(100)),
            SingleResult::new(&meta, BenchmarkStatus::Unsat, Duration::from_millis(200)),
            SingleResult::new(&meta, BenchmarkStatus::Timeout, Duration::from_millis(1000)),
        ];

        let summary = RunSummary::from_results(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.sat, 1);
        assert_eq!(summary.unsat, 1);
        assert_eq!(summary.timeouts, 1);
        assert_eq!(summary.solved(), 2);
    }

    #[test]
    fn test_runner_config_builder() {
        let config = RunnerConfig::new(Duration::from_secs(30))
            .with_memory_limit(1024 * 1024)
            .with_default_logic("QF_LIA")
            .with_continue_on_error(false)
            .with_num_workers(4);

        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.memory_limit, 1024 * 1024);
        assert_eq!(config.default_logic, Some("QF_LIA".to_string()));
        assert!(!config.continue_on_error);
        assert_eq!(config.num_workers, 4);
    }
}
