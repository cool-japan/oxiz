//! CI/CD integration for regression testing
//!
//! This module provides functionality to integrate SMT benchmark testing
//! into continuous integration pipelines, including configuration generation
//! and result reporting for various CI systems.

use crate::benchmark::{BenchmarkStatus, RunSummary, SingleResult};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;
use thiserror::Error;

/// Error type for CI integration operations
#[derive(Error, Debug)]
pub enum CiError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for CI operations
pub type CiResult<T> = Result<T, CiError>;

/// CI system type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CiSystem {
    /// GitHub Actions
    GitHubActions,
    /// GitLab CI
    GitLabCi,
    /// Jenkins
    Jenkins,
    /// CircleCI
    CircleCi,
    /// Travis CI
    TravisCi,
    /// Generic shell script
    Shell,
}

/// CI configuration for benchmark testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiConfig {
    /// CI system type
    pub system: CiSystem,
    /// Benchmark directory
    pub benchmark_dir: String,
    /// Timeout per benchmark (seconds)
    pub timeout_secs: u64,
    /// Memory limit (MB)
    pub memory_limit_mb: u64,
    /// Logic filter
    pub logics: Vec<String>,
    /// Maximum benchmarks to run
    pub max_benchmarks: Option<usize>,
    /// Fail on regression
    pub fail_on_regression: bool,
    /// Regression threshold (percentage)
    pub regression_threshold: f64,
    /// Baseline results file
    pub baseline_file: Option<String>,
    /// Output directory
    pub output_dir: String,
    /// Parallel execution
    pub parallel: bool,
    /// Number of workers
    pub num_workers: usize,
}

impl Default for CiConfig {
    fn default() -> Self {
        Self {
            system: CiSystem::GitHubActions,
            benchmark_dir: "benchmarks".to_string(),
            timeout_secs: 60,
            memory_limit_mb: 4096,
            logics: Vec::new(),
            max_benchmarks: None,
            fail_on_regression: true,
            regression_threshold: 5.0,
            baseline_file: None,
            output_dir: "results".to_string(),
            parallel: true,
            num_workers: 4,
        }
    }
}

impl CiConfig {
    /// Create new config for a CI system
    #[must_use]
    pub fn new(system: CiSystem) -> Self {
        Self {
            system,
            ..Default::default()
        }
    }

    /// Set benchmark directory
    #[must_use]
    pub fn with_benchmark_dir(mut self, dir: impl Into<String>) -> Self {
        self.benchmark_dir = dir.into();
        self
    }

    /// Set timeout
    #[must_use]
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set logics filter
    #[must_use]
    pub fn with_logics(mut self, logics: Vec<String>) -> Self {
        self.logics = logics;
        self
    }

    /// Set baseline file for regression detection
    #[must_use]
    pub fn with_baseline(mut self, file: impl Into<String>) -> Self {
        self.baseline_file = Some(file.into());
        self
    }
}

/// Generate CI configuration files
pub struct CiConfigGenerator {
    config: CiConfig,
}

impl CiConfigGenerator {
    /// Create a new generator
    #[must_use]
    pub fn new(config: CiConfig) -> Self {
        Self { config }
    }

    /// Generate configuration for the configured CI system
    pub fn generate(&self) -> CiResult<String> {
        match self.config.system {
            CiSystem::GitHubActions => self.generate_github_actions(),
            CiSystem::GitLabCi => self.generate_gitlab_ci(),
            CiSystem::Jenkins => self.generate_jenkinsfile(),
            CiSystem::Shell => self.generate_shell_script(),
            _ => self.generate_shell_script(),
        }
    }

    /// Generate GitHub Actions workflow
    fn generate_github_actions(&self) -> CiResult<String> {
        let logics_arg = if self.config.logics.is_empty() {
            String::new()
        } else {
            format!("--logics {}", self.config.logics.join(","))
        };

        let max_arg = self
            .config
            .max_benchmarks
            .map(|n| format!("--max-files {}", n))
            .unwrap_or_default();

        Ok(format!(
            r#"name: SMT Benchmark Tests

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:

env:
  CARGO_TERM_COLOR: always

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
    - uses: actions/checkout@v4

    - name: Install Rust
      uses: dtolnay/rust-action@stable

    - name: Cache cargo
      uses: actions/cache@v4
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{{{ runner.os }}}}-cargo-${{{{ hashFiles('**/Cargo.lock') }}}}

    - name: Build solver
      run: cargo build --release

    - name: Run benchmarks
      run: |
        cargo run --release -- benchmark \
          --dir {} \
          --timeout {} \
          --output {} \
          {} {} \
          --json {}/results.json

    - name: Check for regressions
      if: ${{ github.event_name == 'pull_request' }}
      run: |
        if [ -f "{}" ]; then
          cargo run --release -- compare \
            --baseline {} \
            --current {}/results.json \
            --threshold {}
        fi

    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: {}
"#,
            self.config.benchmark_dir,
            self.config.timeout_secs,
            self.config.output_dir,
            logics_arg,
            max_arg,
            self.config.output_dir,
            self.config
                .baseline_file
                .as_deref()
                .unwrap_or("baseline.json"),
            self.config
                .baseline_file
                .as_deref()
                .unwrap_or("baseline.json"),
            self.config.output_dir,
            self.config.regression_threshold,
            self.config.output_dir
        ))
    }

    /// Generate GitLab CI configuration
    fn generate_gitlab_ci(&self) -> CiResult<String> {
        Ok(format!(
            r#"stages:
  - build
  - test
  - report

variables:
  CARGO_HOME: $CI_PROJECT_DIR/.cargo

cache:
  paths:
    - .cargo/
    - target/

build:
  stage: build
  script:
    - cargo build --release
  artifacts:
    paths:
      - target/release/oxiz

benchmark:
  stage: test
  timeout: 1 hour
  script:
    - |
      ./target/release/oxiz benchmark \
        --dir {} \
        --timeout {} \
        --output {} \
        --json {}/results.json
  artifacts:
    paths:
      - {}
    reports:
      junit: {}/junit.xml
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH

regression-check:
  stage: report
  script:
    - |
      if [ -f "{}" ]; then
        ./target/release/oxiz compare \
          --baseline {} \
          --current {}/results.json \
          --threshold {}
      fi
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
"#,
            self.config.benchmark_dir,
            self.config.timeout_secs,
            self.config.output_dir,
            self.config.output_dir,
            self.config.output_dir,
            self.config.output_dir,
            self.config
                .baseline_file
                .as_deref()
                .unwrap_or("baseline.json"),
            self.config
                .baseline_file
                .as_deref()
                .unwrap_or("baseline.json"),
            self.config.output_dir,
            self.config.regression_threshold
        ))
    }

    /// Generate Jenkinsfile
    fn generate_jenkinsfile(&self) -> CiResult<String> {
        Ok(format!(
            r#"pipeline {{
    agent any

    environment {{
        CARGO_HOME = "${{WORKSPACE}}/.cargo"
    }}

    stages {{
        stage('Build') {{
            steps {{
                sh 'cargo build --release'
            }}
        }}

        stage('Benchmark') {{
            steps {{
                timeout(time: 60, unit: 'MINUTES') {{
                    sh '''
                        ./target/release/oxiz benchmark \
                            --dir {} \
                            --timeout {} \
                            --output {} \
                            --json {}/results.json
                    '''
                }}
            }}
        }}

        stage('Regression Check') {{
            when {{
                changeRequest()
            }}
            steps {{
                script {{
                    if (fileExists('{}')) {{
                        sh '''
                            ./target/release/oxiz compare \
                                --baseline {} \
                                --current {}/results.json \
                                --threshold {}
                        '''
                    }}
                }}
            }}
        }}
    }}

    post {{
        always {{
            archiveArtifacts artifacts: '{}/**/*', fingerprint: true
            junit '{}/junit.xml'
        }}
    }}
}}
"#,
            self.config.benchmark_dir,
            self.config.timeout_secs,
            self.config.output_dir,
            self.config.output_dir,
            self.config
                .baseline_file
                .as_deref()
                .unwrap_or("baseline.json"),
            self.config
                .baseline_file
                .as_deref()
                .unwrap_or("baseline.json"),
            self.config.output_dir,
            self.config.regression_threshold,
            self.config.output_dir,
            self.config.output_dir
        ))
    }

    /// Generate shell script
    fn generate_shell_script(&self) -> CiResult<String> {
        let logics_arg = if self.config.logics.is_empty() {
            String::new()
        } else {
            format!("--logics {}", self.config.logics.join(","))
        };

        let parallel_arg = if self.config.parallel {
            format!("--parallel --workers {}", self.config.num_workers)
        } else {
            String::new()
        };

        Ok(format!(
            r#"#!/bin/bash
set -e

# SMT Benchmark Test Script
# Generated by oxiz-smtcomp

BENCHMARK_DIR="{}"
TIMEOUT={}
MEMORY_LIMIT={}
OUTPUT_DIR="{}"
BASELINE_FILE="{}"
REGRESSION_THRESHOLD={}

echo "=== Building solver ==="
cargo build --release

echo "=== Running benchmarks ==="
mkdir -p "$OUTPUT_DIR"

./target/release/oxiz benchmark \
    --dir "$BENCHMARK_DIR" \
    --timeout "$TIMEOUT" \
    --memory-limit "$MEMORY_LIMIT" \
    --output "$OUTPUT_DIR" \
    {} {} \
    --json "$OUTPUT_DIR/results.json" \
    --html "$OUTPUT_DIR/report.html"

echo "=== Checking for regressions ==="
if [ -f "$BASELINE_FILE" ]; then
    ./target/release/oxiz compare \
        --baseline "$BASELINE_FILE" \
        --current "$OUTPUT_DIR/results.json" \
        --threshold "$REGRESSION_THRESHOLD"

    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo "ERROR: Performance regression detected!"
        exit 1
    fi
else
    echo "No baseline file found, skipping regression check"
fi

echo "=== Results ==="
cat "$OUTPUT_DIR/summary.txt"

echo "=== Done ==="
"#,
            self.config.benchmark_dir,
            self.config.timeout_secs,
            self.config.memory_limit_mb,
            self.config.output_dir,
            self.config
                .baseline_file
                .as_deref()
                .unwrap_or("baseline.json"),
            self.config.regression_threshold,
            logics_arg,
            parallel_arg
        ))
    }

    /// Write configuration to file
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> CiResult<()> {
        let content = self.generate()?;
        fs::write(path, content)?;
        Ok(())
    }
}

/// CI test result for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiTestResult {
    /// Test name
    pub name: String,
    /// Test passed
    pub passed: bool,
    /// Time in seconds
    pub time_secs: f64,
    /// Error message if failed
    pub error: Option<String>,
    /// Test category
    pub category: Option<String>,
}

impl CiTestResult {
    /// Create from SingleResult
    #[must_use]
    pub fn from_single_result(result: &SingleResult) -> Self {
        let name = result
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| result.path.to_string_lossy().to_string());

        let passed =
            result.correct != Some(false) && !matches!(result.status, BenchmarkStatus::Error);

        Self {
            name,
            passed,
            time_secs: result.time.as_secs_f64(),
            error: result.error_message.clone(),
            category: result.logic.clone(),
        }
    }
}

/// Generate JUnit XML report for CI systems
pub fn generate_junit_xml(results: &[SingleResult], suite_name: &str) -> String {
    let summary = RunSummary::from_results(results);
    let failures = results
        .iter()
        .filter(|r| r.correct == Some(false) || r.status == BenchmarkStatus::Error)
        .count();

    let mut xml = String::new();
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    xml.push('\n');
    xml.push_str(&format!(
        r#"<testsuite name="{}" tests="{}" failures="{}" errors="{}" time="{:.3}">"#,
        suite_name,
        summary.total,
        failures,
        summary.errors,
        summary.total_time.as_secs_f64()
    ));
    xml.push('\n');

    for result in results {
        let name = result
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        let classname = result.logic.as_deref().unwrap_or("unknown");

        xml.push_str(&format!(
            r#"  <testcase name="{}" classname="{}" time="{:.3}">"#,
            xml_escape(&name),
            classname,
            result.time.as_secs_f64()
        ));

        if result.correct == Some(false) {
            xml.push_str(&format!(
                r#"<failure message="Wrong answer: got {} expected {:?}"/>"#,
                result.status.as_str(),
                result.expected
            ));
        } else if result.status == BenchmarkStatus::Error {
            xml.push_str(&format!(
                r#"<error message="{}"/>"#,
                xml_escape(result.error_message.as_deref().unwrap_or("Unknown error"))
            ));
        }

        xml.push_str("</testcase>\n");
    }

    xml.push_str("</testsuite>\n");
    xml
}

/// Escape XML special characters
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Generate GitHub Actions summary (markdown)
pub fn generate_github_summary(results: &[SingleResult]) -> String {
    let summary = RunSummary::from_results(results);

    let mut md = String::new();
    md.push_str("## SMT Benchmark Results\n\n");

    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|-------|\n");
    md.push_str(&format!("| Total | {} |\n", summary.total));
    md.push_str(&format!(
        "| Solved | {} ({:.1}%) |\n",
        summary.solved(),
        summary.solve_rate()
    ));
    md.push_str(&format!("| SAT | {} |\n", summary.sat));
    md.push_str(&format!("| UNSAT | {} |\n", summary.unsat));
    md.push_str(&format!("| Timeout | {} |\n", summary.timeouts));
    md.push_str(&format!("| Errors | {} |\n", summary.errors));
    md.push_str(&format!(
        "| Sound | {} |\n",
        if summary.is_sound() { "Yes" } else { "NO" }
    ));

    if summary.incorrect > 0 {
        md.push_str("\n### Wrong Answers\n\n");
        for result in results.iter().filter(|r| r.correct == Some(false)) {
            let name = result
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            md.push_str(&format!(
                "- `{}`: got `{}` expected `{:?}`\n",
                name,
                result.status.as_str(),
                result.expected
            ));
        }
    }

    md
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;
    use std::time::Duration;

    fn make_result(name: &str, status: BenchmarkStatus, correct: Option<bool>) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/{}", name)),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        let mut result = SingleResult::new(&meta, status, Duration::from_millis(100));
        result.correct = correct;
        result
    }

    #[test]
    fn test_ci_config_builder() {
        let config = CiConfig::new(CiSystem::GitHubActions)
            .with_benchmark_dir("tests/benchmarks")
            .with_timeout(120)
            .with_logics(vec!["QF_LIA".to_string()])
            .with_baseline("baseline.json");

        assert_eq!(config.system, CiSystem::GitHubActions);
        assert_eq!(config.benchmark_dir, "tests/benchmarks");
        assert_eq!(config.timeout_secs, 120);
        assert!(config.baseline_file.is_some());
    }

    #[test]
    fn test_github_actions_generation() {
        let config = CiConfig::new(CiSystem::GitHubActions);
        let generator = CiConfigGenerator::new(config);
        let yaml = generator.generate().unwrap();

        assert!(yaml.contains("name: SMT Benchmark Tests"));
        assert!(yaml.contains("cargo build --release"));
        assert!(yaml.contains("upload-artifact"));
    }

    #[test]
    fn test_shell_script_generation() {
        let config = CiConfig::new(CiSystem::Shell);
        let generator = CiConfigGenerator::new(config);
        let script = generator.generate().unwrap();

        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("cargo build --release"));
    }

    #[test]
    fn test_junit_xml_generation() {
        let results = vec![
            make_result("pass.smt2", BenchmarkStatus::Sat, Some(true)),
            make_result("fail.smt2", BenchmarkStatus::Sat, Some(false)),
        ];

        let xml = generate_junit_xml(&results, "SMT Tests");

        assert!(xml.contains("testsuite"));
        assert!(xml.contains("tests=\"2\""));
        assert!(xml.contains("failures=\"1\""));
        assert!(xml.contains("pass.smt2"));
    }

    #[test]
    fn test_github_summary_generation() {
        let results = vec![
            make_result("a.smt2", BenchmarkStatus::Sat, Some(true)),
            make_result("b.smt2", BenchmarkStatus::Unsat, Some(true)),
        ];

        let md = generate_github_summary(&results);

        assert!(md.contains("## SMT Benchmark Results"));
        assert!(md.contains("Solved"));
        assert!(md.contains("100.0%"));
    }
}
