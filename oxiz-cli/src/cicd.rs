//! CI/CD Integration helpers
//!
//! This module provides tools and output formats optimized for CI/CD pipelines
//! including GitHub Actions, GitLab CI, Jenkins, and other automation platforms.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// CI/CD report format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CicdReport {
    /// Report format version
    pub version: String,
    /// Timestamp when report was generated
    pub timestamp: String,
    /// Overall status
    pub status: CicdStatus,
    /// Summary statistics
    pub summary: CicdSummary,
    /// Individual test results
    pub results: Vec<CicdResult>,
    /// Environment information
    pub environment: CicdEnvironment,
}

/// Overall CI/CD status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CicdStatus {
    /// All tests passed
    Success,
    /// Some tests failed
    Failure,
    /// Tests had errors (couldn't run)
    Error,
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CicdSummary {
    /// Total number of files processed
    pub total: usize,
    /// Number of SAT results
    pub sat: usize,
    /// Number of UNSAT results
    pub unsat: usize,
    /// Number of UNKNOWN results
    pub unknown: usize,
    /// Number of errors
    pub errors: usize,
    /// Total time in milliseconds
    pub total_time_ms: u128,
    /// Average time per file
    pub avg_time_ms: f64,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CicdResult {
    /// File path
    pub file: String,
    /// Test status
    pub status: String,
    /// Result (sat/unsat/unknown/error)
    pub result: String,
    /// Time taken in milliseconds
    pub time_ms: u128,
    /// Error message if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CicdEnvironment {
    /// OxiZ version
    pub oxiz_version: String,
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
    /// CI platform (if detected)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ci_platform: Option<String>,
}

impl CicdReport {
    /// Create a new CI/CD report
    pub fn new() -> Self {
        let ci_platform = detect_ci_platform();

        Self {
            version: "1.0".to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            status: CicdStatus::Success,
            summary: CicdSummary {
                total: 0,
                sat: 0,
                unsat: 0,
                unknown: 0,
                errors: 0,
                total_time_ms: 0,
                avg_time_ms: 0.0,
            },
            results: Vec::new(),
            environment: CicdEnvironment {
                oxiz_version: env!("CARGO_PKG_VERSION").to_string(),
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                ci_platform,
            },
        }
    }

    /// Add a result to the report
    pub fn add_result(
        &mut self,
        file: String,
        result: String,
        time_ms: u128,
        error: Option<String>,
    ) {
        let status = if error.is_some() {
            "error".to_string()
        } else {
            match result.to_lowercase().as_str() {
                "sat" => "pass",
                "unsat" => "pass",
                "unknown" => "warning",
                _ => "error",
            }
            .to_string()
        };

        self.results.push(CicdResult {
            file,
            status,
            result: result.clone(),
            time_ms,
            error,
        });

        // Update summary
        self.summary.total += 1;
        self.summary.total_time_ms += time_ms;

        match result.to_lowercase().as_str() {
            "sat" => self.summary.sat += 1,
            "unsat" => self.summary.unsat += 1,
            "unknown" => self.summary.unknown += 1,
            _ => self.summary.errors += 1,
        }

        // Update overall status
        if self.summary.errors > 0 {
            self.status = CicdStatus::Error;
        } else if self.summary.unknown > 0 {
            self.status = CicdStatus::Failure;
        }
    }

    /// Finalize the report (calculate averages)
    pub fn finalize(&mut self) {
        if self.summary.total > 0 {
            self.summary.avg_time_ms =
                self.summary.total_time_ms as f64 / self.summary.total as f64;
        }
    }

    /// Get exit code based on status
    pub fn exit_code(&self) -> i32 {
        match self.status {
            CicdStatus::Success => 0,
            CicdStatus::Failure => 1,
            CicdStatus::Error => 2,
        }
    }

    /// Format as JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Print summary to console
    pub fn print_summary(&self) {
        println!("=== CI/CD Test Summary ===");
        println!();
        println!("Status: {:?}", self.status);
        println!("Total: {}", self.summary.total);
        println!("  SAT: {}", self.summary.sat);
        println!("  UNSAT: {}", self.summary.unsat);
        println!("  UNKNOWN: {}", self.summary.unknown);
        println!("  ERRORS: {}", self.summary.errors);
        println!();
        println!("Total time: {} ms", self.summary.total_time_ms);
        println!("Average time: {:.2} ms/file", self.summary.avg_time_ms);
        println!();

        if let Some(ref platform) = self.environment.ci_platform {
            println!("CI Platform: {}", platform);
        }
    }

    /// Generate annotations for CI platforms
    pub fn generate_annotations(&self) -> Vec<String> {
        let mut annotations = Vec::new();

        for result in &self.results {
            if let Some(ref error) = result.error {
                let annotation =
                    format_annotation(&self.environment.ci_platform, "error", &result.file, error);
                annotations.push(annotation);
            } else if result.result.to_lowercase() == "unknown" {
                let annotation = format_annotation(
                    &self.environment.ci_platform,
                    "warning",
                    &result.file,
                    "Solver returned UNKNOWN",
                );
                annotations.push(annotation);
            }
        }

        annotations
    }
}

impl Default for CicdReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect CI platform from environment variables
fn detect_ci_platform() -> Option<String> {
    if std::env::var("GITHUB_ACTIONS").is_ok() {
        Some("GitHub Actions".to_string())
    } else if std::env::var("GITLAB_CI").is_ok() {
        Some("GitLab CI".to_string())
    } else if std::env::var("JENKINS_URL").is_ok() {
        Some("Jenkins".to_string())
    } else if std::env::var("CIRCLECI").is_ok() {
        Some("CircleCI".to_string())
    } else if std::env::var("TRAVIS").is_ok() {
        Some("Travis CI".to_string())
    } else if std::env::var("BUILDKITE").is_ok() {
        Some("Buildkite".to_string())
    } else {
        None
    }
}

/// Format annotation for CI platform
fn format_annotation(platform: &Option<String>, level: &str, file: &str, message: &str) -> String {
    match platform.as_deref() {
        Some("GitHub Actions") => {
            format!("::{} file={}::{}", level, file, message)
        }
        Some("GitLab CI") => {
            // GitLab uses JSON format for code quality reports
            format!(
                "{{\"type\":\"{}\",\"file\":\"{}\",\"description\":\"{}\"}}",
                level, file, message
            )
        }
        _ => {
            // Generic format
            format!("[{}] {}: {}", level.to_uppercase(), file, message)
        }
    }
}

/// Write report to file
pub fn write_report(report: &CicdReport, path: &PathBuf) -> Result<(), String> {
    let json = report.to_json();
    std::fs::write(path, json).map_err(|e| format!("Failed to write CI/CD report: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cicd_report_creation() {
        let report = CicdReport::new();
        assert_eq!(report.version, "1.0");
        assert_eq!(report.status, CicdStatus::Success);
        assert_eq!(report.summary.total, 0);
    }

    #[test]
    fn test_add_result_success() {
        let mut report = CicdReport::new();
        report.add_result("test.smt2".to_string(), "sat".to_string(), 100, None);

        assert_eq!(report.summary.total, 1);
        assert_eq!(report.summary.sat, 1);
        assert_eq!(report.summary.total_time_ms, 100);
        assert_eq!(report.status, CicdStatus::Success);
    }

    #[test]
    fn test_add_result_error() {
        let mut report = CicdReport::new();
        report.add_result(
            "test.smt2".to_string(),
            "error".to_string(),
            100,
            Some("Parse error".to_string()),
        );

        assert_eq!(report.summary.total, 1);
        assert_eq!(report.summary.errors, 1);
        assert_eq!(report.status, CicdStatus::Error);
    }

    #[test]
    fn test_finalize() {
        let mut report = CicdReport::new();
        report.add_result("test1.smt2".to_string(), "sat".to_string(), 100, None);
        report.add_result("test2.smt2".to_string(), "unsat".to_string(), 200, None);
        report.finalize();

        assert_eq!(report.summary.avg_time_ms, 150.0);
    }

    #[test]
    fn test_exit_code() {
        let mut report = CicdReport::new();
        assert_eq!(report.exit_code(), 0);

        report.status = CicdStatus::Failure;
        assert_eq!(report.exit_code(), 1);

        report.status = CicdStatus::Error;
        assert_eq!(report.exit_code(), 2);
    }

    #[test]
    fn test_format_annotation_github() {
        let platform = Some("GitHub Actions".to_string());
        let annotation = format_annotation(&platform, "error", "test.smt2", "Parse error");
        assert!(annotation.starts_with("::error"));
        assert!(annotation.contains("test.smt2"));
    }
}
