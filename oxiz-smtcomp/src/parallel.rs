//! Parallel benchmark execution using rayon
//!
//! This module provides parallel execution of benchmarks using rayon
//! for efficient multi-core utilization during benchmark runs.

use crate::benchmark::{BenchmarkStatus, RunnerConfig, SingleResult};
use crate::loader::{Benchmark, BenchmarkMeta, Loader};
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::{Command, parse_script};
use oxiz_solver::Solver;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Configuration for parallel benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Number of worker threads (0 = use all available cores)
    pub num_threads: usize,
    /// Timeout per benchmark
    pub timeout: Duration,
    /// Memory limit per benchmark in bytes (0 = unlimited)
    pub memory_limit: u64,
    /// Default logic to use if not specified
    pub default_logic: Option<String>,
    /// Whether to continue on error
    pub continue_on_error: bool,
    /// Progress callback interval (report every N benchmarks)
    pub progress_interval: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Use all available cores
            timeout: Duration::from_secs(60),
            memory_limit: 0,
            default_logic: None,
            continue_on_error: true,
            progress_interval: 10,
        }
    }
}

impl ParallelConfig {
    /// Create a new config with the given timeout
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            ..Default::default()
        }
    }

    /// Set the number of threads
    #[must_use]
    pub fn with_num_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Set memory limit
    #[must_use]
    pub fn with_memory_limit(mut self, limit: u64) -> Self {
        self.memory_limit = limit;
        self
    }

    /// Set default logic
    #[must_use]
    pub fn with_default_logic(mut self, logic: impl Into<String>) -> Self {
        self.default_logic = Some(logic.into());
        self
    }
}

impl From<&RunnerConfig> for ParallelConfig {
    fn from(config: &RunnerConfig) -> Self {
        Self {
            num_threads: config.num_workers,
            timeout: config.timeout,
            memory_limit: config.memory_limit,
            default_logic: config.default_logic.clone(),
            continue_on_error: config.continue_on_error,
            progress_interval: 10,
        }
    }
}

/// Progress information during parallel execution
#[derive(Debug, Clone)]
pub struct ParallelProgress {
    /// Total number of benchmarks
    pub total: usize,
    /// Number of completed benchmarks
    pub completed: usize,
    /// Number of solved (SAT + UNSAT)
    pub solved: usize,
    /// Number of errors
    pub errors: usize,
    /// Elapsed time
    pub elapsed: Duration,
}

/// Callback type for progress updates
pub type ProgressCallback = Box<dyn Fn(ParallelProgress) + Send + Sync>;

/// Parallel benchmark runner
pub struct ParallelRunner {
    config: ParallelConfig,
}

impl ParallelRunner {
    /// Create a new parallel runner with the given configuration
    #[must_use]
    pub fn new(config: ParallelConfig) -> Self {
        Self { config }
    }

    /// Create a runner with default configuration
    #[must_use]
    pub fn with_timeout(timeout: Duration) -> Self {
        Self::new(ParallelConfig::new(timeout))
    }

    /// Run benchmarks in parallel
    pub fn run_all(&self, benchmarks: &[Benchmark]) -> Vec<SingleResult> {
        self.run_all_with_progress(benchmarks, None)
    }

    /// Run benchmarks in parallel with progress callback
    pub fn run_all_with_progress(
        &self,
        benchmarks: &[Benchmark],
        progress_callback: Option<ProgressCallback>,
    ) -> Vec<SingleResult> {
        let start = Instant::now();
        let total = benchmarks.len();
        let completed = Arc::new(AtomicUsize::new(0));
        let solved = Arc::new(AtomicUsize::new(0));
        let errors = Arc::new(AtomicUsize::new(0));

        // Configure rayon thread pool if needed
        let pool = if self.config.num_threads > 0 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(self.config.num_threads)
                    .build()
                    .expect("Failed to build thread pool"),
            )
        } else {
            None
        };

        let config = &self.config;
        let completed_ref = &completed;
        let solved_ref = &solved;
        let errors_ref = &errors;

        let run_benchmarks = || {
            benchmarks
                .par_iter()
                .map(|benchmark| {
                    let result = Self::run_single_benchmark(benchmark, config);

                    // Update progress
                    let count = completed_ref.fetch_add(1, Ordering::Relaxed) + 1;

                    match result.status {
                        BenchmarkStatus::Sat | BenchmarkStatus::Unsat => {
                            solved_ref.fetch_add(1, Ordering::Relaxed);
                        }
                        BenchmarkStatus::Error => {
                            errors_ref.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {}
                    }

                    // Report progress
                    if let Some(ref callback) = progress_callback
                        && (count.is_multiple_of(config.progress_interval) || count == total)
                    {
                        callback(ParallelProgress {
                            total,
                            completed: count,
                            solved: solved_ref.load(Ordering::Relaxed),
                            errors: errors_ref.load(Ordering::Relaxed),
                            elapsed: start.elapsed(),
                        });
                    }

                    result
                })
                .collect::<Vec<_>>()
        };

        if let Some(pool) = pool {
            pool.install(run_benchmarks)
        } else {
            run_benchmarks()
        }
    }

    /// Run benchmarks from metadata in parallel
    pub fn run_from_meta(&self, meta_list: &[BenchmarkMeta], loader: &Loader) -> Vec<SingleResult> {
        self.run_from_meta_with_progress(meta_list, loader, None)
    }

    /// Run benchmarks from metadata with progress callback
    pub fn run_from_meta_with_progress(
        &self,
        meta_list: &[BenchmarkMeta],
        loader: &Loader,
        progress_callback: Option<ProgressCallback>,
    ) -> Vec<SingleResult> {
        let start = Instant::now();
        let total = meta_list.len();
        let completed = Arc::new(AtomicUsize::new(0));
        let solved = Arc::new(AtomicUsize::new(0));
        let errors = Arc::new(AtomicUsize::new(0));

        let config = &self.config;
        let completed_ref = &completed;
        let solved_ref = &solved;
        let errors_ref = &errors;

        meta_list
            .par_iter()
            .map(|meta| {
                let result = match loader.load(meta) {
                    Ok(benchmark) => Self::run_single_benchmark(&benchmark, config),
                    Err(e) => SingleResult::error(meta, Duration::ZERO, e.to_string()),
                };

                // Update progress
                let count = completed_ref.fetch_add(1, Ordering::Relaxed) + 1;

                match result.status {
                    BenchmarkStatus::Sat | BenchmarkStatus::Unsat => {
                        solved_ref.fetch_add(1, Ordering::Relaxed);
                    }
                    BenchmarkStatus::Error => {
                        errors_ref.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {}
                }

                // Report progress
                if let Some(ref callback) = progress_callback
                    && (count.is_multiple_of(config.progress_interval) || count == total)
                {
                    callback(ParallelProgress {
                        total,
                        completed: count,
                        solved: solved_ref.load(Ordering::Relaxed),
                        errors: errors_ref.load(Ordering::Relaxed),
                        elapsed: start.elapsed(),
                    });
                }

                result
            })
            .collect()
    }

    /// Run a single benchmark with the given configuration
    fn run_single_benchmark(benchmark: &Benchmark, config: &ParallelConfig) -> SingleResult {
        let start = Instant::now();
        let deadline = start + config.timeout;

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
                    let sort = parse_sort(&sort_str, &tm);
                    let _var = tm.mk_var(&name, sort);
                }
                Command::DeclareFun(name, arg_sorts, ret_sort) => {
                    if arg_sorts.is_empty() {
                        let sort = parse_sort(&ret_sort, &tm);
                        let _var = tm.mk_var(&name, sort);
                    }
                }
                Command::Assert(term) => {
                    solver.assert(term, &mut tm);
                }
                Command::CheckSat => {
                    // Set default logic if not set
                    if !logic_set {
                        if let Some(ref logic) = config.default_logic {
                            solver.set_logic(logic);
                        } else if let Some(ref logic) = benchmark.meta.logic {
                            solver.set_logic(logic);
                        }
                    }

                    // Check remaining timeout
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
                Command::Exit => break,
                Command::Reset => {
                    solver = Solver::new();
                    tm = TermManager::new();
                    logic_set = false;
                }
                _ => {}
            }
        }

        // If we got here without a check-sat, return unknown
        SingleResult::new(&benchmark.meta, BenchmarkStatus::Unknown, start.elapsed())
    }
}

/// Parse a sort string to sort ID
fn parse_sort(sort_str: &str, tm: &TermManager) -> oxiz_core::sort::SortId {
    match sort_str {
        "Bool" => tm.sorts.bool_sort,
        "Int" => tm.sorts.int_sort,
        "Real" => tm.sorts.real_sort,
        _ => tm.sorts.bool_sort,
    }
}

/// Convenience function to run benchmarks in parallel
pub fn run_parallel(benchmarks: &[Benchmark], config: ParallelConfig) -> Vec<SingleResult> {
    ParallelRunner::new(config).run_all(benchmarks)
}

/// Convenience function to run benchmarks from metadata in parallel
pub fn run_parallel_from_meta(
    meta_list: &[BenchmarkMeta],
    loader: &Loader,
    config: ParallelConfig,
) -> Vec<SingleResult> {
    ParallelRunner::new(config).run_from_meta(meta_list, loader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::ExpectedStatus;
    use std::path::PathBuf;

    fn make_test_benchmark(content: &str) -> Benchmark {
        Benchmark {
            meta: BenchmarkMeta {
                path: PathBuf::from("/tmp/test.smt2"),
                logic: Some("QF_LIA".to_string()),
                expected_status: Some(ExpectedStatus::Sat),
                file_size: content.len() as u64,
                category: None,
            },
            content: content.to_string(),
        }
    }

    #[test]
    fn test_parallel_config_builder() {
        let config = ParallelConfig::new(Duration::from_secs(30))
            .with_num_threads(4)
            .with_memory_limit(1024 * 1024)
            .with_default_logic("QF_LIA");

        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.memory_limit, 1024 * 1024);
        assert_eq!(config.default_logic, Some("QF_LIA".to_string()));
    }

    #[test]
    fn test_parallel_runner_single_benchmark() {
        let benchmark = make_test_benchmark(
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        );

        let runner = ParallelRunner::with_timeout(Duration::from_secs(10));
        let results = runner.run_all(&[benchmark]);

        assert_eq!(results.len(), 1);
        assert!(matches!(
            results[0].status,
            BenchmarkStatus::Sat | BenchmarkStatus::Unknown
        ));
    }

    #[test]
    fn test_parallel_runner_multiple_benchmarks() {
        let benchmarks: Vec<_> = (0..5)
            .map(|i| {
                make_test_benchmark(&format!(
                    "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x {}))\n(check-sat)",
                    i
                ))
            })
            .collect();

        let runner =
            ParallelRunner::new(ParallelConfig::new(Duration::from_secs(10)).with_num_threads(2));
        let results = runner.run_all(&benchmarks);

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_parallel_progress() {
        let benchmarks: Vec<_> = (0..10)
            .map(|i| {
                make_test_benchmark(&format!(
                    "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x {}))\n(check-sat)",
                    i
                ))
            })
            .collect();

        let progress_count = Arc::new(AtomicUsize::new(0));
        let progress_count_clone = progress_count.clone();

        let callback: ProgressCallback = Box::new(move |_progress| {
            progress_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        let runner =
            ParallelRunner::new(ParallelConfig::new(Duration::from_secs(10)).with_num_threads(2));
        let config = ParallelConfig {
            progress_interval: 2,
            ..ParallelConfig::new(Duration::from_secs(10))
        };
        let runner = ParallelRunner::new(config);
        let _results = runner.run_all_with_progress(&benchmarks, Some(callback));

        // Progress should have been called at least once
        assert!(progress_count.load(Ordering::Relaxed) > 0);
    }
}
