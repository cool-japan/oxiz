//! SMT-COMP format result reporting
//!
//! This module provides functionality to report benchmark results in various
//! formats compatible with SMT-COMP and common analysis tools.

use crate::benchmark::{RunSummary, SingleResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;
use thiserror::Error;

/// Error type for reporter operations
#[derive(Error, Debug)]
pub enum ReporterError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for reporter operations
pub type ReporterResult<T> = Result<T, ReporterError>;

/// Output format for reports
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// Plain text summary
    Text,
    /// SMT-COMP standard format
    SmtComp,
}

/// Configuration for report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReporterConfig {
    /// Output format
    pub format: ReportFormat,
    /// Include detailed per-benchmark results
    pub include_details: bool,
    /// Include timing information
    pub include_timing: bool,
    /// Include memory usage
    pub include_memory: bool,
    /// Solver name for the report
    pub solver_name: String,
    /// Solver version
    pub solver_version: String,
}

impl Default for ReporterConfig {
    fn default() -> Self {
        Self {
            format: ReportFormat::Json,
            include_details: true,
            include_timing: true,
            include_memory: true,
            solver_name: "OxiZ".to_string(),
            solver_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

impl ReporterConfig {
    /// Create a new config with the given format
    #[must_use]
    pub fn new(format: ReportFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Set the solver name
    #[must_use]
    pub fn with_solver_name(mut self, name: impl Into<String>) -> Self {
        self.solver_name = name.into();
        self
    }

    /// Set the solver version
    #[must_use]
    pub fn with_solver_version(mut self, version: impl Into<String>) -> Self {
        self.solver_version = version.into();
        self
    }

    /// Set whether to include details
    #[must_use]
    pub fn with_details(mut self, include: bool) -> Self {
        self.include_details = include;
        self
    }
}

/// Full benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Solver information
    pub solver: SolverInfo,
    /// Summary statistics
    pub summary: RunSummary,
    /// Per-logic summaries
    pub by_logic: HashMap<String, RunSummary>,
    /// Individual results (if include_details is true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<ResultEntry>>,
}

/// Solver information for the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverInfo {
    /// Solver name
    pub name: String,
    /// Solver version
    pub version: String,
}

/// Single result entry for detailed reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultEntry {
    /// Benchmark path (relative or filename)
    pub benchmark: String,
    /// Logic
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logic: Option<String>,
    /// Result status
    pub status: String,
    /// Expected status
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// Whether result is correct
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correct: Option<bool>,
    /// Time in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_secs: Option<f64>,
    /// Error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<&SingleResult> for ResultEntry {
    fn from(result: &SingleResult) -> Self {
        Self {
            benchmark: result
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| result.path.to_string_lossy().to_string()),
            logic: result.logic.clone(),
            status: result.status.as_str().to_string(),
            expected: result.expected.map(|e| e.as_str().to_string()),
            correct: result.correct,
            time_secs: Some(result.time.as_secs_f64()),
            error: result.error_message.clone(),
        }
    }
}

/// Reporter for generating benchmark result reports
pub struct Reporter {
    config: ReporterConfig,
}

impl Reporter {
    /// Create a new reporter with the given configuration
    #[must_use]
    pub fn new(config: ReporterConfig) -> Self {
        Self { config }
    }

    /// Create a reporter for JSON output
    #[must_use]
    pub fn json() -> Self {
        Self::new(ReporterConfig::new(ReportFormat::Json))
    }

    /// Create a reporter for CSV output
    #[must_use]
    pub fn csv() -> Self {
        Self::new(ReporterConfig::new(ReportFormat::Csv))
    }

    /// Create a reporter for text output
    #[must_use]
    pub fn text() -> Self {
        Self::new(ReporterConfig::new(ReportFormat::Text))
    }

    /// Create a reporter for SMT-COMP format
    #[must_use]
    pub fn smtcomp() -> Self {
        Self::new(ReporterConfig::new(ReportFormat::SmtComp))
    }

    /// Generate a report from results
    #[must_use]
    pub fn generate_report(&self, results: &[SingleResult]) -> Report {
        let summary = RunSummary::from_results(results);
        let by_logic = self.group_by_logic(results);

        let detailed_results = if self.config.include_details {
            Some(results.iter().map(ResultEntry::from).collect())
        } else {
            None
        };

        Report {
            solver: SolverInfo {
                name: self.config.solver_name.clone(),
                version: self.config.solver_version.clone(),
            },
            summary,
            by_logic,
            results: detailed_results,
        }
    }

    /// Write report to a writer
    pub fn write_report<W: Write>(
        &self,
        results: &[SingleResult],
        writer: &mut W,
    ) -> ReporterResult<()> {
        match self.config.format {
            ReportFormat::Json => self.write_json(results, writer),
            ReportFormat::Csv => self.write_csv(results, writer),
            ReportFormat::Text => self.write_text(results, writer),
            ReportFormat::SmtComp => self.write_smtcomp(results, writer),
        }
    }

    /// Write report to a file
    pub fn write_to_file(
        &self,
        results: &[SingleResult],
        path: impl AsRef<Path>,
    ) -> ReporterResult<()> {
        let mut file = std::fs::File::create(path)?;
        self.write_report(results, &mut file)
    }

    /// Write report to a string
    pub fn to_string(&self, results: &[SingleResult]) -> ReporterResult<String> {
        let mut buf = Vec::new();
        self.write_report(results, &mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    /// Write JSON format
    fn write_json<W: Write>(&self, results: &[SingleResult], writer: &mut W) -> ReporterResult<()> {
        let report = self.generate_report(results);
        serde_json::to_writer_pretty(writer, &report)?;
        Ok(())
    }

    /// Write CSV format
    fn write_csv<W: Write>(&self, results: &[SingleResult], writer: &mut W) -> ReporterResult<()> {
        // Write header
        writeln!(
            writer,
            "benchmark,logic,status,expected,correct,time_secs,error"
        )?;

        // Write each result
        for result in results {
            let benchmark = result
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let logic = result.logic.as_deref().unwrap_or("");
            let status = result.status.as_str();
            let expected = result.expected.map(|e| e.as_str()).unwrap_or("");
            let correct = result
                .correct
                .map(|c| if c { "true" } else { "false" })
                .unwrap_or("");
            let time = result.time.as_secs_f64();
            let error = result.error_message.as_deref().unwrap_or("");

            // Escape CSV fields
            let benchmark = csv_escape(&benchmark);
            let error = csv_escape(error);

            writeln!(
                writer,
                "{},{},{},{},{},{:.6},{}",
                benchmark, logic, status, expected, correct, time, error
            )?;
        }

        Ok(())
    }

    /// Write plain text format
    fn write_text<W: Write>(&self, results: &[SingleResult], writer: &mut W) -> ReporterResult<()> {
        let summary = RunSummary::from_results(results);

        writeln!(writer, "=== SMT Benchmark Results ===")?;
        writeln!(
            writer,
            "Solver: {} v{}",
            self.config.solver_name, self.config.solver_version
        )?;
        writeln!(writer)?;

        writeln!(writer, "=== Summary ===")?;
        writeln!(writer, "Total benchmarks: {}", summary.total)?;
        writeln!(
            writer,
            "Solved:           {} ({:.1}%)",
            summary.solved(),
            summary.solve_rate()
        )?;
        writeln!(writer, "  SAT:            {}", summary.sat)?;
        writeln!(writer, "  UNSAT:          {}", summary.unsat)?;
        writeln!(writer, "Unknown:          {}", summary.unknown)?;
        writeln!(writer, "Timeout:          {}", summary.timeouts)?;
        writeln!(writer, "Errors:           {}", summary.errors)?;
        writeln!(writer)?;
        writeln!(writer, "Correct:          {}", summary.correct)?;
        writeln!(writer, "Incorrect:        {}", summary.incorrect)?;
        writeln!(
            writer,
            "Sound:            {}",
            if summary.is_sound() { "yes" } else { "NO" }
        )?;
        writeln!(writer)?;
        writeln!(
            writer,
            "Total time:       {:.3}s",
            summary.total_time.as_secs_f64()
        )?;
        writeln!(
            writer,
            "Average time:     {:.3}s",
            summary.avg_time.as_secs_f64()
        )?;

        // Per-logic breakdown
        let by_logic = self.group_by_logic(results);
        if by_logic.len() > 1 {
            writeln!(writer)?;
            writeln!(writer, "=== By Logic ===")?;
            let mut logics: Vec<_> = by_logic.keys().collect();
            logics.sort();
            for logic in logics {
                let s = &by_logic[logic];
                writeln!(
                    writer,
                    "{:15} {:5} benchmarks, {:5} solved ({:5.1}%), {:3} SAT, {:3} UNSAT",
                    logic,
                    s.total,
                    s.solved(),
                    s.solve_rate(),
                    s.sat,
                    s.unsat
                )?;
            }
        }

        // Detailed results if configured
        if self.config.include_details {
            writeln!(writer)?;
            writeln!(writer, "=== Individual Results ===")?;
            for result in results {
                let filename = result
                    .path
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let status = result.status.as_str();
                let time = format!("{:.3}s", result.time.as_secs_f64());
                let correct_str = match result.correct {
                    Some(true) => " [OK]",
                    Some(false) => " [WRONG]",
                    None => "",
                };

                writeln!(
                    writer,
                    "{:50} {:10} {}{}",
                    filename, status, time, correct_str
                )?;
            }
        }

        Ok(())
    }

    /// Write SMT-COMP standard format
    fn write_smtcomp<W: Write>(
        &self,
        results: &[SingleResult],
        writer: &mut W,
    ) -> ReporterResult<()> {
        // SMT-COMP format: one line per benchmark
        // benchmark_name result cpu_time wall_time memory expected correct
        for result in results {
            let benchmark = result
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            let status = result.status.as_str();
            let cpu_time = result.time.as_secs_f64();
            let wall_time = cpu_time; // Same as CPU time for single-threaded
            let memory = result.memory_bytes.unwrap_or(0);
            let expected = result.expected.map(|e| e.as_str()).unwrap_or("-");
            let correct = match result.correct {
                Some(true) => "correct",
                Some(false) => "wrong",
                None => "-",
            };

            writeln!(
                writer,
                "{} {} {:.3} {:.3} {} {} {}",
                benchmark, status, cpu_time, wall_time, memory, expected, correct
            )?;
        }

        Ok(())
    }

    /// Group results by logic
    fn group_by_logic(&self, results: &[SingleResult]) -> HashMap<String, RunSummary> {
        let mut grouped: HashMap<String, Vec<&SingleResult>> = HashMap::new();

        for result in results {
            let logic = result
                .logic
                .clone()
                .unwrap_or_else(|| "UNKNOWN".to_string());
            grouped.entry(logic).or_default().push(result);
        }

        grouped
            .into_iter()
            .map(|(logic, refs)| {
                let owned: Vec<SingleResult> = refs.into_iter().cloned().collect();
                (logic, RunSummary::from_results(&owned))
            })
            .collect()
    }
}

/// Escape a string for CSV output
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// SMT-COMP score calculation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SmtCompScore {
    /// Number of correct answers
    pub correct: usize,
    /// Number of wrong answers
    pub wrong: usize,
    /// Total CPU time for correct answers
    pub cpu_time: Duration,
    /// Score: correct - wrong (negative if unsound)
    pub score: i64,
    /// Sequential performance score
    pub seq_score: f64,
    /// Parallel performance score
    pub par_score: f64,
}

impl SmtCompScore {
    /// Calculate SMT-COMP score from results
    #[must_use]
    pub fn from_results(results: &[SingleResult], timeout: Duration) -> Self {
        let mut score = Self::default();

        for result in results {
            match result.correct {
                Some(true) => {
                    score.correct += 1;
                    score.cpu_time += result.time;
                }
                Some(false) => {
                    score.wrong += 1;
                }
                None => {}
            }
        }

        score.score = score.correct as i64 - score.wrong as i64;

        // Calculate sequential performance score
        // Score = sum of (timeout - time) for each correctly solved benchmark
        let timeout_secs = timeout.as_secs_f64();
        for result in results {
            if result.correct == Some(true) {
                score.seq_score += timeout_secs - result.time.as_secs_f64();
            }
        }

        // Parallel score is same as sequential for single-threaded
        score.par_score = score.seq_score;

        score
    }

    /// Check if the solver is sound (no wrong answers)
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.wrong == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkStatus;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;

    fn make_result(status: BenchmarkStatus, expected: Option<ExpectedStatus>) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from("/tmp/test.smt2"),
            logic: Some("QF_LIA".to_string()),
            expected_status: expected,
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(100))
    }

    #[test]
    fn test_csv_escape() {
        assert_eq!(csv_escape("hello"), "hello");
        assert_eq!(csv_escape("hello,world"), "\"hello,world\"");
        assert_eq!(csv_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn test_report_generation() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, Some(ExpectedStatus::Sat)),
            make_result(BenchmarkStatus::Unsat, Some(ExpectedStatus::Unsat)),
        ];

        let reporter = Reporter::json();
        let report = reporter.generate_report(&results);

        assert_eq!(report.summary.total, 2);
        assert_eq!(report.summary.sat, 1);
        assert_eq!(report.summary.unsat, 1);
        assert_eq!(report.summary.correct, 2);
    }

    #[test]
    fn test_smtcomp_score() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, Some(ExpectedStatus::Sat)),
            make_result(BenchmarkStatus::Unsat, Some(ExpectedStatus::Unsat)),
            make_result(BenchmarkStatus::Sat, Some(ExpectedStatus::Unsat)), // Wrong
        ];

        let score = SmtCompScore::from_results(&results, Duration::from_secs(60));
        assert_eq!(score.correct, 2);
        assert_eq!(score.wrong, 1);
        assert_eq!(score.score, 1);
        assert!(!score.is_sound());
    }

    #[test]
    fn test_text_report() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, Some(ExpectedStatus::Sat)),
            make_result(BenchmarkStatus::Timeout, None),
        ];

        let reporter = Reporter::text();
        let output = reporter.to_string(&results).unwrap();

        assert!(output.contains("Total benchmarks: 2"));
        assert!(output.contains("SAT:"));
        assert!(output.contains("Timeout:"));
    }
}
