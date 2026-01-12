//! Benchmark Harness
//!
//! Utilities for benchmarking the SAT solver on DIMACS CNF files.
//! Provides timing, memory tracking, and result aggregation.

use crate::config_presets::ConfigPreset;
use crate::dimacs::{DimacsError, DimacsParser};
use crate::solver::{Solver, SolverConfig, SolverResult};
use crate::stats_dashboard::{StatsAggregator, StatsDashboard};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Result of a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Instance name/path
    pub instance: String,
    /// Solver result (SAT/UNSAT/UNKNOWN)
    pub result: SolverResult,
    /// Wall clock time
    pub wall_time: Duration,
    /// Statistics dashboard
    pub stats: StatsDashboard,
    /// Configuration preset used
    pub preset: Option<ConfigPreset>,
}

impl BenchmarkResult {
    /// Check if the result is satisfiable
    #[must_use]
    pub const fn is_sat(&self) -> bool {
        matches!(self.result, SolverResult::Sat)
    }

    /// Check if the result is unsatisfiable
    #[must_use]
    pub const fn is_unsat(&self) -> bool {
        matches!(self.result, SolverResult::Unsat)
    }

    /// Check if the result is unknown
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self.result, SolverResult::Unknown)
    }
}

/// Benchmark harness for running SAT solver on multiple instances
#[derive(Debug)]
pub struct BenchmarkHarness {
    /// Results from completed benchmarks
    results: Vec<BenchmarkResult>,
    /// Timeout in seconds (None for no timeout)
    timeout: Option<u64>,
}

impl BenchmarkHarness {
    /// Create a new benchmark harness
    #[must_use]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            timeout: None,
        }
    }

    /// Set timeout in seconds
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout = Some(seconds);
        self
    }

    /// Run benchmark on a single DIMACS CNF file
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be parsed
    pub fn run_file(
        &mut self,
        path: impl AsRef<Path>,
        config: SolverConfig,
        preset: Option<ConfigPreset>,
    ) -> Result<BenchmarkResult, DimacsError> {
        let path = path.as_ref();
        let instance_name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Parse DIMACS file and create solver
        let mut parser = DimacsParser::new();
        let mut solver = Solver::with_config(config);
        parser.parse_file(path, &mut solver)?;

        // Run solver with timing
        let start = Instant::now();
        let result = solver.solve();
        let wall_time = start.elapsed();

        // Create statistics dashboard
        let stats_dashboard = StatsDashboard::new(solver.stats().clone(), wall_time);

        let bench_result = BenchmarkResult {
            instance: instance_name,
            result,
            wall_time,
            stats: stats_dashboard,
            preset,
        };

        self.results.push(bench_result.clone());

        Ok(bench_result)
    }

    /// Run benchmark on a single file with a configuration preset
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be parsed
    pub fn run_file_with_preset(
        &mut self,
        path: impl AsRef<Path>,
        preset: ConfigPreset,
    ) -> Result<BenchmarkResult, DimacsError> {
        self.run_file(path, preset.config(), Some(preset))
    }

    /// Run benchmark on multiple files
    ///
    /// # Errors
    ///
    /// Returns error if any file cannot be parsed
    pub fn run_files(
        &mut self,
        paths: &[PathBuf],
        config: SolverConfig,
    ) -> Result<Vec<BenchmarkResult>, DimacsError> {
        let mut results = Vec::new();
        for path in paths {
            let result = self.run_file(path, config.clone(), None)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Run benchmark comparing multiple presets on a single file
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be parsed
    pub fn compare_presets(
        &mut self,
        path: impl AsRef<Path>,
        presets: &[ConfigPreset],
    ) -> Result<Vec<BenchmarkResult>, DimacsError> {
        let mut results = Vec::new();
        for &preset in presets {
            let result = self.run_file_with_preset(&path, preset)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Get all benchmark results
    #[must_use]
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Get number of SAT results
    #[must_use]
    pub fn num_sat(&self) -> usize {
        self.results.iter().filter(|r| r.is_sat()).count()
    }

    /// Get number of UNSAT results
    #[must_use]
    pub fn num_unsat(&self) -> usize {
        self.results.iter().filter(|r| r.is_unsat()).count()
    }

    /// Get number of UNKNOWN results
    #[must_use]
    pub fn num_unknown(&self) -> usize {
        self.results.iter().filter(|r| r.is_unknown()).count()
    }

    /// Get total wall time across all benchmarks
    #[must_use]
    pub fn total_time(&self) -> Duration {
        self.results.iter().map(|r| r.wall_time).sum()
    }

    /// Get average wall time
    #[must_use]
    pub fn average_time(&self) -> Duration {
        if self.results.is_empty() {
            return Duration::ZERO;
        }
        let total_nanos: u128 = self.results.iter().map(|r| r.wall_time.as_nanos()).sum();
        Duration::from_nanos((total_nanos / self.results.len() as u128) as u64)
    }

    /// Generate a summary report
    #[must_use]
    pub fn summary_report(&self) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("                   BENCHMARK SUMMARY                           \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str(&format!("Total Instances:    {}\n", self.results.len()));
        report.push_str(&format!("SAT:                {}\n", self.num_sat()));
        report.push_str(&format!("UNSAT:              {}\n", self.num_unsat()));
        report.push_str(&format!("UNKNOWN:            {}\n", self.num_unknown()));
        report.push('\n');

        report.push_str(&format!(
            "Total Time:         {:.3}s\n",
            self.total_time().as_secs_f64()
        ));
        report.push_str(&format!(
            "Average Time:       {:.3}s\n",
            self.average_time().as_secs_f64()
        ));

        if !self.results.is_empty() {
            let min_time = self
                .results
                .iter()
                .map(|r| r.wall_time)
                .min()
                .unwrap_or(Duration::ZERO);
            let max_time = self
                .results
                .iter()
                .map(|r| r.wall_time)
                .max()
                .unwrap_or(Duration::ZERO);

            report.push_str(&format!(
                "Min Time:           {:.3}s\n",
                min_time.as_secs_f64()
            ));
            report.push_str(&format!(
                "Max Time:           {:.3}s\n",
                max_time.as_secs_f64()
            ));
        }

        report.push('\n');
        report.push_str("═══════════════════════════════════════════════════════════════\n");

        report
    }

    /// Generate detailed report with per-instance results
    #[must_use]
    pub fn detailed_report(&self) -> String {
        let mut report = self.summary_report();

        if self.results.is_empty() {
            return report;
        }

        report.push_str("\nPER-INSTANCE RESULTS:\n");
        report.push_str("───────────────────────────────────────────────────────────────\n");
        report.push_str(&format!(
            "{:<30} {:>10} {:>12} {:>12}\n",
            "Instance", "Result", "Time (s)", "Conflicts"
        ));
        report.push_str("───────────────────────────────────────────────────────────────\n");

        for result in &self.results {
            let result_str = match result.result {
                SolverResult::Sat => "SAT",
                SolverResult::Unsat => "UNSAT",
                SolverResult::Unknown => "UNKNOWN",
            };

            report.push_str(&format!(
                "{:<30} {:>10} {:>12.3} {:>12}\n",
                &result.instance,
                result_str,
                result.wall_time.as_secs_f64(),
                result.stats.solver_stats().conflicts
            ));
        }

        report.push_str("═══════════════════════════════════════════════════════════════\n");

        report
    }

    /// Create a statistics aggregator from benchmark results
    #[must_use]
    pub fn to_stats_aggregator(&self) -> StatsAggregator {
        let mut aggregator = StatsAggregator::new();
        for result in &self.results {
            aggregator.add_run(result.stats.clone());
        }
        aggregator
    }

    /// Clear all results
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

impl Default for BenchmarkHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_harness_creation() {
        let harness = BenchmarkHarness::new();
        assert_eq!(harness.results().len(), 0);
        assert_eq!(harness.num_sat(), 0);
        assert_eq!(harness.num_unsat(), 0);
    }

    #[test]
    fn test_benchmark_with_timeout() {
        let harness = BenchmarkHarness::new().with_timeout(10);
        assert_eq!(harness.timeout, Some(10));
    }

    #[test]
    fn test_summary_report_empty() {
        let harness = BenchmarkHarness::new();
        let report = harness.summary_report();
        assert!(report.contains("BENCHMARK SUMMARY"));
        assert!(report.contains("Total Instances:    0"));
    }

    #[test]
    fn test_detailed_report_empty() {
        let harness = BenchmarkHarness::new();
        let report = harness.detailed_report();
        assert!(report.contains("BENCHMARK SUMMARY"));
    }

    #[test]
    fn test_clear_results() {
        let mut harness = BenchmarkHarness::new();
        harness.clear();
        assert_eq!(harness.results().len(), 0);
    }

    #[test]
    fn test_stats_aggregator_creation() {
        let harness = BenchmarkHarness::new();
        let aggregator = harness.to_stats_aggregator();
        assert_eq!(aggregator.num_runs(), 0);
    }

    #[test]
    fn test_time_calculations() {
        let harness = BenchmarkHarness::new();
        assert_eq!(harness.total_time(), Duration::ZERO);
        assert_eq!(harness.average_time(), Duration::ZERO);
    }
}
