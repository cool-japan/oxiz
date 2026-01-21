//! Analysis and comparison statistics
//!
//! This module provides statistical analysis and comparison tools for
//! benchmark results, including solver comparison and performance metrics.

use crate::benchmark::{BenchmarkStatus, RunSummary, SingleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Statistics for a set of benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Statistics {
    /// Number of benchmarks
    pub count: usize,
    /// Minimum time
    pub min_time: Duration,
    /// Maximum time
    pub max_time: Duration,
    /// Mean time
    pub mean_time: Duration,
    /// Median time
    pub median_time: Duration,
    /// Standard deviation of time
    pub std_dev: Duration,
    /// 90th percentile time
    pub p90_time: Duration,
    /// 99th percentile time
    pub p99_time: Duration,
    /// Total time
    pub total_time: Duration,
}

impl Statistics {
    /// Calculate statistics from a list of durations
    #[must_use]
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self::default();
        }

        let mut sorted: Vec<Duration> = durations.to_vec();
        sorted.sort();

        let count = sorted.len();
        let min_time = sorted[0];
        let max_time = sorted[count - 1];

        let total_nanos: u128 = sorted.iter().map(|d| d.as_nanos()).sum();
        let total_time = Duration::from_nanos(total_nanos as u64);
        let mean_nanos = total_nanos / count as u128;
        let mean_time = Duration::from_nanos(mean_nanos as u64);

        // Median
        let median_time = if count.is_multiple_of(2) {
            let mid = count / 2;
            Duration::from_nanos(((sorted[mid - 1].as_nanos() + sorted[mid].as_nanos()) / 2) as u64)
        } else {
            sorted[count / 2]
        };

        // Standard deviation
        let variance: u128 = sorted
            .iter()
            .map(|d| {
                let diff = d.as_nanos() as i128 - mean_nanos as i128;
                (diff * diff) as u128
            })
            .sum::<u128>()
            / count as u128;
        let std_dev = Duration::from_nanos((variance as f64).sqrt() as u64);

        // Percentiles
        let p90_idx = (count as f64 * 0.9).ceil() as usize - 1;
        let p99_idx = (count as f64 * 0.99).ceil() as usize - 1;
        let p90_time = sorted[p90_idx.min(count - 1)];
        let p99_time = sorted[p99_idx.min(count - 1)];

        Self {
            count,
            min_time,
            max_time,
            mean_time,
            median_time,
            std_dev,
            p90_time,
            p99_time,
            total_time,
        }
    }

    /// Calculate statistics from benchmark results
    #[must_use]
    pub fn from_results(results: &[SingleResult]) -> Self {
        let durations: Vec<Duration> = results.iter().map(|r| r.time).collect();
        Self::from_durations(&durations)
    }

    /// Calculate statistics for solved benchmarks only
    #[must_use]
    pub fn from_solved(results: &[SingleResult]) -> Self {
        let durations: Vec<Duration> = results
            .iter()
            .filter(|r| matches!(r.status, BenchmarkStatus::Sat | BenchmarkStatus::Unsat))
            .map(|r| r.time)
            .collect();
        Self::from_durations(&durations)
    }
}

/// Category statistics for breakdown analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryStats {
    /// Total benchmarks in category
    pub total: usize,
    /// Solved count
    pub solved: usize,
    /// SAT count
    pub sat: usize,
    /// UNSAT count
    pub unsat: usize,
    /// Timeout count
    pub timeouts: usize,
    /// Error count
    pub errors: usize,
    /// Timing statistics
    pub timing: Statistics,
}

impl CategoryStats {
    /// Calculate stats from results
    #[must_use]
    pub fn from_results(results: &[SingleResult]) -> Self {
        let summary = RunSummary::from_results(results);
        let timing = Statistics::from_results(results);

        Self {
            total: summary.total,
            solved: summary.solved(),
            sat: summary.sat,
            unsat: summary.unsat,
            timeouts: summary.timeouts,
            errors: summary.errors,
            timing,
        }
    }

    /// Get solve rate as percentage
    #[must_use]
    pub fn solve_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.solved as f64 / self.total as f64) * 100.0
        }
    }
}

/// Comparison between two solver runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverComparison {
    /// Name of solver A
    pub solver_a: String,
    /// Name of solver B
    pub solver_b: String,
    /// Benchmarks only A solved
    pub only_a_solved: usize,
    /// Benchmarks only B solved
    pub only_b_solved: usize,
    /// Benchmarks both solved
    pub both_solved: usize,
    /// Benchmarks neither solved
    pub neither_solved: usize,
    /// A was faster on N benchmarks
    pub a_faster: usize,
    /// B was faster on N benchmarks
    pub b_faster: usize,
    /// Speedup of A over B (geometric mean on commonly solved)
    pub speedup_a_over_b: f64,
    /// Disagreements (different sat/unsat results)
    pub disagreements: Vec<String>,
}

impl SolverComparison {
    /// Compare two solver result sets (must be aligned by benchmark)
    #[must_use]
    pub fn compare(
        solver_a: &str,
        results_a: &[SingleResult],
        solver_b: &str,
        results_b: &[SingleResult],
    ) -> Self {
        let mut comparison = Self {
            solver_a: solver_a.to_string(),
            solver_b: solver_b.to_string(),
            only_a_solved: 0,
            only_b_solved: 0,
            both_solved: 0,
            neither_solved: 0,
            a_faster: 0,
            b_faster: 0,
            speedup_a_over_b: 1.0,
            disagreements: Vec::new(),
        };

        // Build map from path to result for solver B
        let b_map: HashMap<_, _> = results_b.iter().map(|r| (r.path.clone(), r)).collect();

        let mut log_speedups: Vec<f64> = Vec::new();

        for result_a in results_a {
            let Some(result_b) = b_map.get(&result_a.path) else {
                continue;
            };

            let a_solved = matches!(
                result_a.status,
                BenchmarkStatus::Sat | BenchmarkStatus::Unsat
            );
            let b_solved = matches!(
                result_b.status,
                BenchmarkStatus::Sat | BenchmarkStatus::Unsat
            );

            match (a_solved, b_solved) {
                (true, true) => {
                    comparison.both_solved += 1;

                    // Check for disagreement
                    if (result_a.status == BenchmarkStatus::Sat
                        && result_b.status == BenchmarkStatus::Unsat)
                        || (result_a.status == BenchmarkStatus::Unsat
                            && result_b.status == BenchmarkStatus::Sat)
                    {
                        comparison.disagreements.push(
                            result_a
                                .path
                                .file_name()
                                .map(|s| s.to_string_lossy().to_string())
                                .unwrap_or_default(),
                        );
                    }

                    // Compare times
                    let time_a = result_a.time.as_secs_f64();
                    let time_b = result_b.time.as_secs_f64();

                    // Avoid division by zero
                    let time_a = time_a.max(0.001);
                    let time_b = time_b.max(0.001);

                    if time_a < time_b {
                        comparison.a_faster += 1;
                    } else if time_b < time_a {
                        comparison.b_faster += 1;
                    }

                    // Log speedup for geometric mean
                    log_speedups.push((time_b / time_a).ln());
                }
                (true, false) => comparison.only_a_solved += 1,
                (false, true) => comparison.only_b_solved += 1,
                (false, false) => comparison.neither_solved += 1,
            }
        }

        // Calculate geometric mean speedup
        if !log_speedups.is_empty() {
            let mean_log: f64 = log_speedups.iter().sum::<f64>() / log_speedups.len() as f64;
            comparison.speedup_a_over_b = mean_log.exp();
        }

        comparison
    }

    /// Get a summary of the comparison
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = format!("Comparison: {} vs {}\n", self.solver_a, self.solver_b);
        s.push_str(&format!(
            "  Only {} solved: {}\n",
            self.solver_a, self.only_a_solved
        ));
        s.push_str(&format!(
            "  Only {} solved: {}\n",
            self.solver_b, self.only_b_solved
        ));
        s.push_str(&format!("  Both solved: {}\n", self.both_solved));
        s.push_str(&format!("  Neither solved: {}\n", self.neither_solved));
        s.push_str(&format!(
            "  {} faster: {}, {} faster: {}\n",
            self.solver_a, self.a_faster, self.solver_b, self.b_faster
        ));
        s.push_str(&format!(
            "  Speedup ({}/{}): {:.2}x\n",
            self.solver_a, self.solver_b, self.speedup_a_over_b
        ));
        if !self.disagreements.is_empty() {
            s.push_str(&format!("  DISAGREEMENTS: {:?}\n", self.disagreements));
        }
        s
    }
}

/// Scatter plot data point for cactus/scatter plots
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterPoint {
    /// Benchmark name
    pub benchmark: String,
    /// Time for solver A
    pub time_a: Option<f64>,
    /// Time for solver B
    pub time_b: Option<f64>,
    /// Status for solver A
    pub status_a: String,
    /// Status for solver B
    pub status_b: String,
}

/// Generate scatter plot data from two result sets
#[must_use]
pub fn scatter_data(results_a: &[SingleResult], results_b: &[SingleResult]) -> Vec<ScatterPoint> {
    let b_map: HashMap<_, _> = results_b.iter().map(|r| (r.path.clone(), r)).collect();

    let mut points = Vec::new();

    for result_a in results_a {
        let benchmark = result_a
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let (time_b, status_b) = if let Some(rb) = b_map.get(&result_a.path) {
            (Some(rb.time.as_secs_f64()), rb.status.as_str().to_string())
        } else {
            (None, "missing".to_string())
        };

        points.push(ScatterPoint {
            benchmark,
            time_a: Some(result_a.time.as_secs_f64()),
            time_b,
            status_a: result_a.status.as_str().to_string(),
            status_b,
        });
    }

    points
}

/// Cactus plot data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CactusPoint {
    /// Number of benchmarks solved
    pub solved: usize,
    /// Time at which this many were solved
    pub time: f64,
}

/// Generate cactus plot data from results
#[must_use]
pub fn cactus_data(results: &[SingleResult]) -> Vec<CactusPoint> {
    // Collect times for solved benchmarks
    let mut solved_times: Vec<f64> = results
        .iter()
        .filter(|r| matches!(r.status, BenchmarkStatus::Sat | BenchmarkStatus::Unsat))
        .map(|r| r.time.as_secs_f64())
        .collect();

    // Sort by time
    solved_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Generate cactus points
    solved_times
        .into_iter()
        .enumerate()
        .map(|(i, time)| CactusPoint {
            solved: i + 1,
            time,
        })
        .collect()
}

/// Performance analysis by difficulty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifficultyAnalysis {
    /// Easy benchmarks (< 1s)
    pub easy: CategoryStats,
    /// Medium benchmarks (1-10s)
    pub medium: CategoryStats,
    /// Hard benchmarks (10-60s)
    pub hard: CategoryStats,
    /// Very hard benchmarks (60s+)
    pub very_hard: CategoryStats,
}

impl DifficultyAnalysis {
    /// Analyze results by difficulty (based on solve time)
    #[must_use]
    pub fn from_results(results: &[SingleResult]) -> Self {
        let mut easy = Vec::new();
        let mut medium = Vec::new();
        let mut hard = Vec::new();
        let mut very_hard = Vec::new();

        for result in results {
            let secs = result.time.as_secs_f64();
            if secs < 1.0 {
                easy.push(result.clone());
            } else if secs < 10.0 {
                medium.push(result.clone());
            } else if secs < 60.0 {
                hard.push(result.clone());
            } else {
                very_hard.push(result.clone());
            }
        }

        Self {
            easy: CategoryStats::from_results(&easy),
            medium: CategoryStats::from_results(&medium),
            hard: CategoryStats::from_results(&hard),
            very_hard: CategoryStats::from_results(&very_hard),
        }
    }
}

/// Aggregate analysis combining multiple statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullAnalysis {
    /// Overall summary
    pub summary: RunSummary,
    /// Timing statistics
    pub timing: Statistics,
    /// Per-logic breakdown
    pub by_logic: HashMap<String, CategoryStats>,
    /// Per-category breakdown
    pub by_category: HashMap<String, CategoryStats>,
    /// Difficulty analysis
    pub by_difficulty: DifficultyAnalysis,
}

impl FullAnalysis {
    /// Generate full analysis from results
    #[must_use]
    pub fn from_results(results: &[SingleResult]) -> Self {
        let summary = RunSummary::from_results(results);
        let timing = Statistics::from_results(results);

        // Group by logic
        let mut by_logic: HashMap<String, Vec<SingleResult>> = HashMap::new();
        for result in results {
            let logic = result
                .logic
                .clone()
                .unwrap_or_else(|| "UNKNOWN".to_string());
            by_logic.entry(logic).or_default().push(result.clone());
        }

        // Group by category
        let mut by_category: HashMap<String, Vec<SingleResult>> = HashMap::new();
        for result in results {
            if let Some(parent) = result.path.parent() {
                let category = parent
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "root".to_string());
                by_category
                    .entry(category)
                    .or_default()
                    .push(result.clone());
            }
        }

        Self {
            summary,
            timing,
            by_logic: by_logic
                .into_iter()
                .map(|(k, v)| (k, CategoryStats::from_results(&v)))
                .collect(),
            by_category: by_category
                .into_iter()
                .map(|(k, v)| (k, CategoryStats::from_results(&v)))
                .collect(),
            by_difficulty: DifficultyAnalysis::from_results(results),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;

    fn make_result(status: BenchmarkStatus, time_ms: u64) -> SingleResult {
        make_result_with_id(status, time_ms, time_ms)
    }

    fn make_result_with_id(status: BenchmarkStatus, time_ms: u64, id: u64) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/test_{}.smt2", id)),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    #[test]
    fn test_statistics_from_durations() {
        let durations = vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
            Duration::from_millis(400),
            Duration::from_millis(500),
        ];

        let stats = Statistics::from_durations(&durations);

        assert_eq!(stats.count, 5);
        assert_eq!(stats.min_time, Duration::from_millis(100));
        assert_eq!(stats.max_time, Duration::from_millis(500));
        assert_eq!(stats.median_time, Duration::from_millis(300));
    }

    #[test]
    fn test_solver_comparison() {
        // Use same benchmark IDs (1, 2, 3) for both result sets
        let results_a = vec![
            make_result_with_id(BenchmarkStatus::Sat, 100, 1), // bench 1: solved by A
            make_result_with_id(BenchmarkStatus::Unsat, 200, 2), // bench 2: solved by A
            make_result_with_id(BenchmarkStatus::Timeout, 1000, 3), // bench 3: not solved by A
        ];

        let results_b = vec![
            make_result_with_id(BenchmarkStatus::Sat, 150, 1), // bench 1: solved by B
            make_result_with_id(BenchmarkStatus::Timeout, 1000, 2), // bench 2: not solved by B
            make_result_with_id(BenchmarkStatus::Sat, 300, 3), // bench 3: solved by B
        ];

        let comparison = SolverComparison::compare("A", &results_a, "B", &results_b);

        // Bench 1: both solved, A faster (100ms vs 150ms)
        // Bench 2: only A solved
        // Bench 3: only B solved
        assert_eq!(comparison.both_solved, 1);
        assert_eq!(comparison.only_a_solved, 1);
        assert_eq!(comparison.only_b_solved, 1);
    }

    #[test]
    fn test_cactus_data() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, 100),
            make_result(BenchmarkStatus::Sat, 300),
            make_result(BenchmarkStatus::Sat, 200),
            make_result(BenchmarkStatus::Timeout, 1000),
        ];

        let cactus = cactus_data(&results);

        assert_eq!(cactus.len(), 3); // Only solved benchmarks
        assert_eq!(cactus[0].solved, 1);
        assert!(cactus[0].time < cactus[1].time); // Should be sorted
    }

    #[test]
    fn test_difficulty_analysis() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, 100),       // Easy
            make_result(BenchmarkStatus::Sat, 5000),      // Medium
            make_result(BenchmarkStatus::Sat, 30000),     // Hard
            make_result(BenchmarkStatus::Timeout, 70000), // Very hard
        ];

        let analysis = DifficultyAnalysis::from_results(&results);

        assert_eq!(analysis.easy.total, 1);
        assert_eq!(analysis.medium.total, 1);
        assert_eq!(analysis.hard.total, 1);
        assert_eq!(analysis.very_hard.total, 1);
    }
}
