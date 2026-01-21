//! Performance regression detection
//!
//! This module provides functionality to detect performance regressions
//! between benchmark runs.

use crate::benchmark::{BenchmarkStatus, RunSummary, SingleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Regression detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Threshold for time regression (percentage increase)
    pub time_threshold_pct: f64,
    /// Threshold for solve count regression (percentage decrease)
    pub solve_threshold_pct: f64,
    /// Minimum time difference to consider (avoid noise in fast benchmarks)
    pub min_time_diff_secs: f64,
    /// Maximum allowed new errors
    pub max_new_errors: usize,
    /// Maximum allowed new timeouts
    pub max_new_timeouts: usize,
    /// Fail on soundness regression
    pub fail_on_soundness: bool,
    /// Individual benchmark time multiplier for regression
    pub individual_time_multiplier: f64,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            time_threshold_pct: 10.0,
            solve_threshold_pct: 5.0,
            min_time_diff_secs: 0.1,
            max_new_errors: 0,
            max_new_timeouts: 5,
            fail_on_soundness: true,
            individual_time_multiplier: 2.0,
        }
    }
}

impl RegressionConfig {
    /// Set time threshold
    #[must_use]
    pub fn with_time_threshold(mut self, pct: f64) -> Self {
        self.time_threshold_pct = pct;
        self
    }

    /// Set solve threshold
    #[must_use]
    pub fn with_solve_threshold(mut self, pct: f64) -> Self {
        self.solve_threshold_pct = pct;
        self
    }

    /// Allow some new timeouts
    #[must_use]
    pub fn with_max_new_timeouts(mut self, max: usize) -> Self {
        self.max_new_timeouts = max;
        self
    }
}

/// Type of regression detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionType {
    /// Total solve time increased significantly
    TotalTimeRegression,
    /// Number of solved benchmarks decreased
    SolveCountRegression,
    /// Individual benchmark regressed significantly
    IndividualTimeRegression,
    /// Previously solved benchmark now times out
    NewTimeout,
    /// Previously working benchmark now errors
    NewError,
    /// Wrong answer detected
    SoundnessRegression,
    /// Previously correct answer now wrong
    CorrectnessRegression,
}

/// A single regression finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionFinding {
    /// Type of regression
    pub regression_type: RegressionType,
    /// Severity (0.0-1.0, higher is worse)
    pub severity: f64,
    /// Description of the regression
    pub description: String,
    /// Affected benchmark (if applicable)
    pub benchmark: Option<String>,
    /// Baseline value
    pub baseline_value: Option<String>,
    /// Current value
    pub current_value: Option<String>,
}

impl RegressionFinding {
    /// Create a new finding
    fn new(regression_type: RegressionType, severity: f64, description: impl Into<String>) -> Self {
        Self {
            regression_type,
            severity,
            description: description.into(),
            benchmark: None,
            baseline_value: None,
            current_value: None,
        }
    }

    /// Add benchmark context
    fn with_benchmark(mut self, benchmark: impl Into<String>) -> Self {
        self.benchmark = Some(benchmark.into());
        self
    }

    /// Add values
    fn with_values(mut self, baseline: impl Into<String>, current: impl Into<String>) -> Self {
        self.baseline_value = Some(baseline.into());
        self.current_value = Some(current.into());
        self
    }

    /// Check if this is a critical regression
    #[must_use]
    pub fn is_critical(&self) -> bool {
        matches!(
            self.regression_type,
            RegressionType::SoundnessRegression | RegressionType::CorrectnessRegression
        ) || self.severity >= 0.8
    }
}

/// Regression analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    /// Whether any regressions were found
    pub has_regressions: bool,
    /// Whether critical regressions were found
    pub has_critical_regressions: bool,
    /// All findings
    pub findings: Vec<RegressionFinding>,
    /// Baseline summary
    pub baseline_summary: RunSummary,
    /// Current summary
    pub current_summary: RunSummary,
    /// Improvements found
    pub improvements: Vec<String>,
}

impl RegressionAnalysis {
    /// Check if the analysis passes (no critical regressions)
    #[must_use]
    pub fn passes(&self) -> bool {
        !self.has_critical_regressions
    }

    /// Get critical findings
    #[must_use]
    pub fn critical_findings(&self) -> Vec<&RegressionFinding> {
        self.findings.iter().filter(|f| f.is_critical()).collect()
    }

    /// Generate report string
    #[must_use]
    pub fn report(&self) -> String {
        let mut s = String::new();

        s.push_str("=== Regression Analysis ===\n\n");

        s.push_str("Summary:\n");
        s.push_str(&format!(
            "  Baseline: {} solved ({:.1}%)\n",
            self.baseline_summary.solved(),
            self.baseline_summary.solve_rate()
        ));
        s.push_str(&format!(
            "  Current:  {} solved ({:.1}%)\n",
            self.current_summary.solved(),
            self.current_summary.solve_rate()
        ));

        let solve_diff =
            self.current_summary.solved() as i64 - self.baseline_summary.solved() as i64;
        s.push_str(&format!("  Difference: {:+}\n\n", solve_diff));

        if self.findings.is_empty() {
            s.push_str("No regressions detected.\n");
        } else {
            s.push_str(&format!("Found {} regressions:\n\n", self.findings.len()));

            for (i, finding) in self.findings.iter().enumerate() {
                let severity_label = if finding.is_critical() {
                    "CRITICAL"
                } else if finding.severity >= 0.5 {
                    "WARNING"
                } else {
                    "MINOR"
                };

                s.push_str(&format!(
                    "{}. [{}] {:?}\n",
                    i + 1,
                    severity_label,
                    finding.regression_type
                ));
                s.push_str(&format!("   {}\n", finding.description));

                if let Some(ref bench) = finding.benchmark {
                    s.push_str(&format!("   Benchmark: {}\n", bench));
                }
                if let (Some(baseline), Some(current)) =
                    (&finding.baseline_value, &finding.current_value)
                {
                    s.push_str(&format!(
                        "   Baseline: {} -> Current: {}\n",
                        baseline, current
                    ));
                }
                s.push('\n');
            }
        }

        if !self.improvements.is_empty() {
            s.push_str("Improvements:\n");
            for imp in &self.improvements {
                s.push_str(&format!("  + {}\n", imp));
            }
        }

        s.push_str(&format!(
            "\nResult: {}\n",
            if self.passes() { "PASS" } else { "FAIL" }
        ));

        s
    }
}

/// Regression detector
pub struct RegressionDetector {
    config: RegressionConfig,
}

impl RegressionDetector {
    /// Create a new detector
    #[must_use]
    pub fn new(config: RegressionConfig) -> Self {
        Self { config }
    }

    /// Analyze for regressions
    pub fn analyze(
        &self,
        baseline: &[SingleResult],
        current: &[SingleResult],
    ) -> RegressionAnalysis {
        let baseline_summary = RunSummary::from_results(baseline);
        let current_summary = RunSummary::from_results(current);

        let mut findings = Vec::new();
        let mut improvements = Vec::new();

        // Build maps for comparison
        let baseline_map: HashMap<PathBuf, &SingleResult> =
            baseline.iter().map(|r| (r.path.clone(), r)).collect();
        let current_map: HashMap<PathBuf, &SingleResult> =
            current.iter().map(|r| (r.path.clone(), r)).collect();

        // Check overall solve count regression
        let solve_diff = baseline_summary.solved() as i64 - current_summary.solved() as i64;
        if solve_diff > 0 {
            let pct = (solve_diff as f64 / baseline_summary.solved() as f64) * 100.0;
            if pct >= self.config.solve_threshold_pct {
                findings.push(
                    RegressionFinding::new(
                        RegressionType::SolveCountRegression,
                        (pct / 100.0).min(1.0),
                        format!("Solve count decreased by {} ({:.1}%)", solve_diff, pct),
                    )
                    .with_values(
                        baseline_summary.solved().to_string(),
                        current_summary.solved().to_string(),
                    ),
                );
            }
        } else if solve_diff < 0 {
            improvements.push(format!("Solve count improved by {}", -solve_diff));
        }

        // Check total time regression
        let baseline_time = baseline_summary.total_time.as_secs_f64();
        let current_time = current_summary.total_time.as_secs_f64();
        if baseline_time > 0.0 {
            let time_increase_pct = ((current_time - baseline_time) / baseline_time) * 100.0;
            if time_increase_pct >= self.config.time_threshold_pct {
                findings.push(
                    RegressionFinding::new(
                        RegressionType::TotalTimeRegression,
                        (time_increase_pct / 100.0).min(1.0),
                        format!("Total time increased by {:.1}%", time_increase_pct),
                    )
                    .with_values(
                        format!("{:.2}s", baseline_time),
                        format!("{:.2}s", current_time),
                    ),
                );
            }
        }

        // Check soundness regression
        if current_summary.incorrect > baseline_summary.incorrect {
            let new_incorrect = current_summary.incorrect - baseline_summary.incorrect;
            findings.push(RegressionFinding::new(
                RegressionType::SoundnessRegression,
                1.0,
                format!("{} new incorrect results", new_incorrect),
            ));
        }

        // Check individual benchmark regressions
        let mut new_timeouts = 0;
        let mut new_errors = 0;

        for (path, baseline_result) in &baseline_map {
            if let Some(current_result) = current_map.get(path) {
                let bench_name = path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();

                // Check for new timeout
                let baseline_solved = matches!(
                    baseline_result.status,
                    BenchmarkStatus::Sat | BenchmarkStatus::Unsat
                );
                let current_solved = matches!(
                    current_result.status,
                    BenchmarkStatus::Sat | BenchmarkStatus::Unsat
                );

                if baseline_solved && current_result.status == BenchmarkStatus::Timeout {
                    new_timeouts += 1;
                    findings.push(
                        RegressionFinding::new(
                            RegressionType::NewTimeout,
                            0.7,
                            "Previously solved benchmark now times out",
                        )
                        .with_benchmark(&bench_name)
                        .with_values(
                            format!(
                                "{} in {:.3}s",
                                baseline_result.status.as_str(),
                                baseline_result.time.as_secs_f64()
                            ),
                            "timeout".to_string(),
                        ),
                    );
                }

                // Check for new error
                if baseline_result.status != BenchmarkStatus::Error
                    && current_result.status == BenchmarkStatus::Error
                {
                    new_errors += 1;
                    findings.push(
                        RegressionFinding::new(
                            RegressionType::NewError,
                            0.8,
                            "Benchmark now produces an error",
                        )
                        .with_benchmark(&bench_name),
                    );
                }

                // Check for correctness regression
                if baseline_result.correct == Some(true) && current_result.correct == Some(false) {
                    findings.push(
                        RegressionFinding::new(
                            RegressionType::CorrectnessRegression,
                            1.0,
                            "Previously correct result is now wrong",
                        )
                        .with_benchmark(&bench_name)
                        .with_values(
                            baseline_result.status.as_str().to_string(),
                            current_result.status.as_str().to_string(),
                        ),
                    );
                }

                // Check for individual time regression
                if baseline_solved && current_solved {
                    let baseline_time = baseline_result.time.as_secs_f64();
                    let current_time = current_result.time.as_secs_f64();
                    let time_diff = current_time - baseline_time;

                    if time_diff >= self.config.min_time_diff_secs
                        && current_time >= baseline_time * self.config.individual_time_multiplier
                    {
                        findings.push(
                            RegressionFinding::new(
                                RegressionType::IndividualTimeRegression,
                                0.5,
                                format!(
                                    "Solve time increased {:.1}x",
                                    current_time / baseline_time
                                ),
                            )
                            .with_benchmark(&bench_name)
                            .with_values(
                                format!("{:.3}s", baseline_time),
                                format!("{:.3}s", current_time),
                            ),
                        );
                    }
                }
            }
        }

        // Apply limits for non-critical findings
        if new_errors > self.config.max_new_errors {
            // Promote severity
            for finding in &mut findings {
                if finding.regression_type == RegressionType::NewError {
                    finding.severity = 0.9;
                }
            }
        }
        if new_timeouts > self.config.max_new_timeouts {
            // Promote severity for excessive new timeouts
            for finding in &mut findings {
                if finding.regression_type == RegressionType::NewTimeout {
                    finding.severity = 0.85;
                }
            }
        }

        let has_regressions = !findings.is_empty();
        let has_critical_regressions = findings.iter().any(|f| f.is_critical());

        RegressionAnalysis {
            has_regressions,
            has_critical_regressions,
            findings,
            baseline_summary,
            current_summary,
            improvements,
        }
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new(RegressionConfig::default())
    }
}

/// Quick regression check function
pub fn check_regression(
    baseline: &[SingleResult],
    current: &[SingleResult],
    threshold_pct: f64,
) -> bool {
    let config = RegressionConfig::default()
        .with_time_threshold(threshold_pct)
        .with_solve_threshold(threshold_pct);
    let detector = RegressionDetector::new(config);
    let analysis = detector.analyze(baseline, current);
    !analysis.has_critical_regressions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkStatus;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_result(name: &str, status: BenchmarkStatus, time_ms: u64) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/{}", name)),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    #[test]
    fn test_no_regression() {
        let baseline = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 100),
            make_result("b.smt2", BenchmarkStatus::Unsat, 200),
        ];

        let current = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 90),
            make_result("b.smt2", BenchmarkStatus::Unsat, 180),
        ];

        let detector = RegressionDetector::default();
        let analysis = detector.analyze(&baseline, &current);

        assert!(!analysis.has_regressions);
        assert!(analysis.passes());
    }

    #[test]
    fn test_solve_count_regression() {
        let baseline = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 100),
            make_result("b.smt2", BenchmarkStatus::Unsat, 200),
        ];

        let current = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 100),
            make_result("b.smt2", BenchmarkStatus::Timeout, 60000),
        ];

        let detector = RegressionDetector::default();
        let analysis = detector.analyze(&baseline, &current);

        assert!(analysis.has_regressions);
        assert!(
            analysis
                .findings
                .iter()
                .any(|f| f.regression_type == RegressionType::NewTimeout)
        );
    }

    #[test]
    fn test_soundness_regression() {
        let baseline = vec![make_result("a.smt2", BenchmarkStatus::Sat, 100)];

        let mut wrong_result = make_result("a.smt2", BenchmarkStatus::Unsat, 100);
        wrong_result.correct = Some(false);
        let current = vec![wrong_result];

        let detector = RegressionDetector::default();
        let analysis = detector.analyze(&baseline, &current);

        assert!(analysis.has_critical_regressions);
        assert!(!analysis.passes());
    }

    #[test]
    fn test_time_regression() {
        let baseline = vec![make_result("a.smt2", BenchmarkStatus::Sat, 100)];
        let current = vec![make_result("a.smt2", BenchmarkStatus::Sat, 500)]; // 5x slower

        let config = RegressionConfig::default();
        let detector = RegressionDetector::new(config);
        let analysis = detector.analyze(&baseline, &current);

        assert!(
            analysis
                .findings
                .iter()
                .any(|f| f.regression_type == RegressionType::IndividualTimeRegression)
        );
    }

    #[test]
    fn test_improvement_detection() {
        let baseline = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 100),
            make_result("b.smt2", BenchmarkStatus::Timeout, 60000),
        ];

        let current = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, 100),
            make_result("b.smt2", BenchmarkStatus::Sat, 500), // Now solved
        ];

        let detector = RegressionDetector::default();
        let analysis = detector.analyze(&baseline, &current);

        assert!(!analysis.improvements.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let baseline = vec![make_result("a.smt2", BenchmarkStatus::Sat, 100)];
        let current = vec![make_result("a.smt2", BenchmarkStatus::Timeout, 60000)];

        let detector = RegressionDetector::default();
        let analysis = detector.analyze(&baseline, &current);

        let report = analysis.report();
        assert!(report.contains("Regression Analysis"));
        assert!(report.contains("FAIL") || report.contains("regressions"));
    }
}
