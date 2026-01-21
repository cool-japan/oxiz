//! Virtual best solver calculation
//!
//! This module provides functionality to calculate the virtual best solver (VBS),
//! which represents the theoretical best performance achievable by selecting
//! the best solver for each benchmark.

use crate::benchmark::{BenchmarkStatus, RunSummary, SingleResult};
use crate::statistics::CactusPoint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Virtual best solver result for a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualBestResult {
    /// Benchmark path
    pub path: PathBuf,
    /// Best status achieved
    pub status: BenchmarkStatus,
    /// Best (fastest) time among solving solvers
    pub time: Duration,
    /// Which solver achieved this result
    pub best_solver: String,
    /// All solver times for this benchmark
    pub solver_times: HashMap<String, Duration>,
    /// All solver statuses for this benchmark
    pub solver_statuses: HashMap<String, BenchmarkStatus>,
}

impl VirtualBestResult {
    /// Check if any solver solved this benchmark
    #[must_use]
    pub fn is_solved(&self) -> bool {
        matches!(self.status, BenchmarkStatus::Sat | BenchmarkStatus::Unsat)
    }

    /// Get the speedup factor vs a specific solver
    #[must_use]
    pub fn speedup_vs(&self, solver: &str) -> Option<f64> {
        let solver_time = self.solver_times.get(solver)?;
        let best_time = self.time.as_secs_f64().max(0.001);
        let solver_time = solver_time.as_secs_f64().max(0.001);
        Some(solver_time / best_time)
    }
}

/// Virtual best solver statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualBestStats {
    /// Total benchmarks
    pub total: usize,
    /// Solved by VBS
    pub solved: usize,
    /// SAT results
    pub sat: usize,
    /// UNSAT results
    pub unsat: usize,
    /// Uniquely solved by each solver (solved by only that solver)
    pub unique_solves: HashMap<String, usize>,
    /// Contribution of each solver (how often it was the best)
    pub solver_contributions: HashMap<String, usize>,
    /// Average speedup of VBS over each solver
    pub avg_speedup: HashMap<String, f64>,
    /// Total VBS time
    pub total_time: Duration,
}

impl VirtualBestStats {
    /// Get solve rate
    #[must_use]
    pub fn solve_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.solved as f64 / self.total as f64) * 100.0
        }
    }
}

/// Virtual best solver calculator
pub struct VirtualBestSolver {
    solver_results: HashMap<String, Vec<SingleResult>>,
}

impl VirtualBestSolver {
    /// Create a new VBS calculator
    #[must_use]
    pub fn new() -> Self {
        Self {
            solver_results: HashMap::new(),
        }
    }

    /// Add results from a solver
    pub fn add_solver(&mut self, name: impl Into<String>, results: Vec<SingleResult>) {
        self.solver_results.insert(name.into(), results);
    }

    /// Calculate virtual best solver results
    #[must_use]
    pub fn calculate(&self) -> Vec<VirtualBestResult> {
        // Build map of benchmark path -> solver results
        let mut by_benchmark: HashMap<PathBuf, HashMap<String, &SingleResult>> = HashMap::new();

        for (solver, results) in &self.solver_results {
            for result in results {
                by_benchmark
                    .entry(result.path.clone())
                    .or_default()
                    .insert(solver.clone(), result);
            }
        }

        // Calculate VBS for each benchmark
        let mut vbs_results = Vec::new();

        for (path, solver_results) in by_benchmark {
            let vbs = self.calculate_single(&path, &solver_results);
            vbs_results.push(vbs);
        }

        // Sort by path for consistent ordering
        vbs_results.sort_by(|a, b| a.path.cmp(&b.path));

        vbs_results
    }

    /// Calculate VBS for a single benchmark
    fn calculate_single(
        &self,
        path: &Path,
        solver_results: &HashMap<String, &SingleResult>,
    ) -> VirtualBestResult {
        let mut solver_times = HashMap::new();
        let mut solver_statuses = HashMap::new();

        // Collect all solver results
        for (solver, result) in solver_results {
            solver_times.insert(solver.clone(), result.time);
            solver_statuses.insert(solver.clone(), result.status);
        }

        // Find the best result (solved with fastest time)
        let mut best_solver = String::new();
        let mut best_status = BenchmarkStatus::Unknown;
        let mut best_time = Duration::MAX;

        for (solver, result) in solver_results {
            let solved = matches!(result.status, BenchmarkStatus::Sat | BenchmarkStatus::Unsat);

            // Prefer solved over unsolved
            let is_current_solved =
                matches!(best_status, BenchmarkStatus::Sat | BenchmarkStatus::Unsat);

            if solved && !is_current_solved {
                // This solver solved it but current best didn't
                best_solver = solver.clone();
                best_status = result.status;
                best_time = result.time;
            } else if solved == is_current_solved {
                // Both solved or both didn't - take faster
                if result.time < best_time {
                    best_solver = solver.clone();
                    best_status = result.status;
                    best_time = result.time;
                }
            }
        }

        VirtualBestResult {
            path: path.to_path_buf(),
            status: best_status,
            time: best_time,
            best_solver,
            solver_times,
            solver_statuses,
        }
    }

    /// Calculate VBS statistics
    #[must_use]
    pub fn calculate_stats(&self) -> VirtualBestStats {
        let vbs_results = self.calculate();
        let solver_names: Vec<_> = self.solver_results.keys().cloned().collect();

        let mut stats = VirtualBestStats {
            total: vbs_results.len(),
            solved: 0,
            sat: 0,
            unsat: 0,
            unique_solves: HashMap::new(),
            solver_contributions: HashMap::new(),
            avg_speedup: HashMap::new(),
            total_time: Duration::ZERO,
        };

        // Initialize maps
        for name in &solver_names {
            stats.unique_solves.insert(name.clone(), 0);
            stats.solver_contributions.insert(name.clone(), 0);
        }

        let mut speedup_sums: HashMap<String, f64> = HashMap::new();
        let mut speedup_counts: HashMap<String, usize> = HashMap::new();

        for result in &vbs_results {
            // Count solved
            if result.is_solved() {
                stats.solved += 1;
                stats.total_time += result.time;

                match result.status {
                    BenchmarkStatus::Sat => stats.sat += 1,
                    BenchmarkStatus::Unsat => stats.unsat += 1,
                    _ => {}
                }

                // Track which solver was best
                if let Some(count) = stats.solver_contributions.get_mut(&result.best_solver) {
                    *count += 1;
                }

                // Count unique solves
                let solvers_that_solved: Vec<_> = result
                    .solver_statuses
                    .iter()
                    .filter(|(_, s)| matches!(s, BenchmarkStatus::Sat | BenchmarkStatus::Unsat))
                    .map(|(name, _)| name)
                    .collect();

                if solvers_that_solved.len() == 1
                    && let Some(count) = stats.unique_solves.get_mut(solvers_that_solved[0])
                {
                    *count += 1;
                }

                // Calculate speedup
                for solver in &solver_names {
                    if let Some(speedup) = result.speedup_vs(solver) {
                        *speedup_sums.entry(solver.clone()).or_insert(0.0) += speedup.ln();
                        *speedup_counts.entry(solver.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Calculate average speedups (geometric mean)
        for solver in &solver_names {
            let sum = speedup_sums.get(solver).copied().unwrap_or(0.0);
            let count = speedup_counts.get(solver).copied().unwrap_or(0);
            if count > 0 {
                stats
                    .avg_speedup
                    .insert(solver.clone(), (sum / count as f64).exp());
            } else {
                stats.avg_speedup.insert(solver.clone(), 1.0);
            }
        }

        stats
    }

    /// Generate cactus plot data for VBS
    #[must_use]
    pub fn cactus_data(&self) -> Vec<CactusPoint> {
        let vbs_results = self.calculate();

        let mut solved_times: Vec<f64> = vbs_results
            .iter()
            .filter(|r| r.is_solved())
            .map(|r| r.time.as_secs_f64())
            .collect();

        solved_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        solved_times
            .into_iter()
            .enumerate()
            .map(|(i, time)| CactusPoint {
                solved: i + 1,
                time,
            })
            .collect()
    }

    /// Convert VBS results to SingleResult format
    #[must_use]
    pub fn to_single_results(&self) -> Vec<SingleResult> {
        let vbs_results = self.calculate();

        vbs_results
            .into_iter()
            .map(|vbs| {
                use crate::loader::BenchmarkMeta;

                let meta = BenchmarkMeta {
                    path: vbs.path,
                    logic: None,
                    expected_status: None,
                    file_size: 0,
                    category: None,
                };

                SingleResult::new(&meta, vbs.status, vbs.time)
            })
            .collect()
    }

    /// Get summary for VBS
    #[must_use]
    pub fn summary(&self) -> RunSummary {
        let results = self.to_single_results();
        RunSummary::from_results(&results)
    }

    /// Generate comparison report
    #[must_use]
    pub fn comparison_report(&self) -> String {
        let stats = self.calculate_stats();
        let mut report = String::new();

        report.push_str("=== Virtual Best Solver Analysis ===\n\n");
        report.push_str(&format!("Total benchmarks: {}\n", stats.total));
        report.push_str(&format!(
            "VBS solved: {} ({:.1}%)\n",
            stats.solved,
            stats.solve_rate()
        ));
        report.push_str(&format!("  SAT: {}\n", stats.sat));
        report.push_str(&format!("  UNSAT: {}\n", stats.unsat));
        report.push_str(&format!(
            "Total VBS time: {:.2}s\n\n",
            stats.total_time.as_secs_f64()
        ));

        report.push_str("Solver Contributions (how often each was best):\n");
        let mut contributions: Vec<_> = stats.solver_contributions.iter().collect();
        contributions.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
        for (solver, count) in contributions {
            let pct = (*count as f64 / stats.solved as f64) * 100.0;
            report.push_str(&format!("  {}: {} ({:.1}%)\n", solver, count, pct));
        }

        report.push_str("\nUnique Solves (benchmarks only that solver could solve):\n");
        let mut unique: Vec<_> = stats.unique_solves.iter().collect();
        unique.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
        for (solver, count) in unique {
            report.push_str(&format!("  {}: {}\n", solver, count));
        }

        report.push_str("\nAverage Speedup of VBS over each solver:\n");
        let mut speedups: Vec<_> = stats.avg_speedup.iter().collect();
        speedups.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        for (solver, speedup) in speedups {
            report.push_str(&format!("  {}: {:.2}x\n", solver, speedup));
        }

        report
    }
}

impl Default for VirtualBestSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to calculate VBS from multiple solver results
pub fn calculate_virtual_best(
    solver_results: &HashMap<String, Vec<SingleResult>>,
) -> Vec<VirtualBestResult> {
    let mut vbs = VirtualBestSolver::new();
    for (name, results) in solver_results {
        vbs.add_solver(name.clone(), results.clone());
    }
    vbs.calculate()
}

/// Calculate VBS stats from multiple solver results
pub fn calculate_virtual_best_stats(
    solver_results: &HashMap<String, Vec<SingleResult>>,
) -> VirtualBestStats {
    let mut vbs = VirtualBestSolver::new();
    for (name, results) in solver_results {
        vbs.add_solver(name.clone(), results.clone());
    }
    vbs.calculate_stats()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::time::Duration;

    fn make_result(path: &str, status: BenchmarkStatus, time_ms: u64) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(path),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    #[test]
    fn test_virtual_best_single_solver() {
        let results = vec![
            make_result("/tmp/a.smt2", BenchmarkStatus::Sat, 100),
            make_result("/tmp/b.smt2", BenchmarkStatus::Unsat, 200),
        ];

        let mut vbs = VirtualBestSolver::new();
        vbs.add_solver("Solver A", results);

        let vbs_results = vbs.calculate();
        assert_eq!(vbs_results.len(), 2);

        let stats = vbs.calculate_stats();
        assert_eq!(stats.solved, 2);
    }

    #[test]
    fn test_virtual_best_multiple_solvers() {
        let results_a = vec![
            make_result("/tmp/a.smt2", BenchmarkStatus::Sat, 100),
            make_result("/tmp/b.smt2", BenchmarkStatus::Timeout, 60000),
        ];

        let results_b = vec![
            make_result("/tmp/a.smt2", BenchmarkStatus::Sat, 200), // Slower
            make_result("/tmp/b.smt2", BenchmarkStatus::Sat, 500), // Solves it
        ];

        let mut vbs = VirtualBestSolver::new();
        vbs.add_solver("Solver A", results_a);
        vbs.add_solver("Solver B", results_b);

        let vbs_results = vbs.calculate();
        assert_eq!(vbs_results.len(), 2);

        // Benchmark a: A is faster
        let a = vbs_results
            .iter()
            .find(|r| r.path.ends_with("a.smt2"))
            .unwrap();
        assert_eq!(a.best_solver, "Solver A");
        assert_eq!(a.time, Duration::from_millis(100));

        // Benchmark b: B solves it
        let b = vbs_results
            .iter()
            .find(|r| r.path.ends_with("b.smt2"))
            .unwrap();
        assert_eq!(b.best_solver, "Solver B");
        assert!(b.is_solved());

        let stats = vbs.calculate_stats();
        assert_eq!(stats.solved, 2);
    }

    #[test]
    fn test_unique_solves() {
        let results_a = vec![
            make_result("/tmp/a.smt2", BenchmarkStatus::Sat, 100),
            make_result("/tmp/b.smt2", BenchmarkStatus::Timeout, 60000),
        ];

        let results_b = vec![
            make_result("/tmp/a.smt2", BenchmarkStatus::Timeout, 60000),
            make_result("/tmp/b.smt2", BenchmarkStatus::Sat, 500),
        ];

        let mut vbs = VirtualBestSolver::new();
        vbs.add_solver("Solver A", results_a);
        vbs.add_solver("Solver B", results_b);

        let stats = vbs.calculate_stats();

        // Each solver uniquely solves one benchmark
        assert_eq!(stats.unique_solves.get("Solver A"), Some(&1));
        assert_eq!(stats.unique_solves.get("Solver B"), Some(&1));
    }

    #[test]
    fn test_cactus_data() {
        let results = vec![
            make_result("/tmp/a.smt2", BenchmarkStatus::Sat, 100),
            make_result("/tmp/b.smt2", BenchmarkStatus::Sat, 300),
            make_result("/tmp/c.smt2", BenchmarkStatus::Sat, 200),
        ];

        let mut vbs = VirtualBestSolver::new();
        vbs.add_solver("Solver A", results);

        let cactus = vbs.cactus_data();
        assert_eq!(cactus.len(), 3);
        assert_eq!(cactus[0].solved, 1);
        assert!(cactus[0].time < cactus[1].time);
    }

    #[test]
    fn test_comparison_report() {
        let results_a = vec![make_result("/tmp/a.smt2", BenchmarkStatus::Sat, 100)];
        let results_b = vec![make_result("/tmp/a.smt2", BenchmarkStatus::Sat, 200)];

        let mut vbs = VirtualBestSolver::new();
        vbs.add_solver("Solver A", results_a);
        vbs.add_solver("Solver B", results_b);

        let report = vbs.comparison_report();
        assert!(report.contains("Virtual Best Solver"));
        assert!(report.contains("Solver A"));
        assert!(report.contains("Solver B"));
    }
}
