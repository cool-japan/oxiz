//! StarExec format compatibility
//!
//! This module provides functionality for reading and writing StarExec-compatible
//! benchmark configurations and results, enabling integration with the StarExec
//! execution service used by SMT-COMP.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::benchmark::{BenchmarkStatus, SingleResult};
use crate::loader::BenchmarkMeta;

/// Error type for StarExec operations
#[derive(Error, Debug)]
pub enum StarExecError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),
    /// Invalid format
    #[error("Invalid StarExec format: {0}")]
    InvalidFormat(String),
}

/// Result type for StarExec operations
pub type StarExecResult<T> = Result<T, StarExecError>;

/// StarExec job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarExecJob {
    /// Job name
    pub name: String,
    /// Job description
    pub description: Option<String>,
    /// Space ID (benchmark category)
    pub space_id: u32,
    /// Solver configuration ID
    pub solver_config_id: u32,
    /// Timeout in seconds
    pub cpu_timeout: u32,
    /// Wall clock timeout in seconds
    pub wallclock_timeout: u32,
    /// Memory limit in MB
    pub memory_limit: u32,
    /// Maximum number of benchmarks
    pub max_benchmarks: Option<usize>,
}

impl Default for StarExecJob {
    fn default() -> Self {
        Self {
            name: "OxiZ Benchmark Job".to_string(),
            description: None,
            space_id: 0,
            solver_config_id: 0,
            cpu_timeout: 1200, // 20 minutes
            wallclock_timeout: 1200,
            memory_limit: 4096, // 4 GB
            max_benchmarks: None,
        }
    }
}

impl StarExecJob {
    /// Create a new job with the given name
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the description
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the timeout
    #[must_use]
    pub fn with_timeout(mut self, seconds: u32) -> Self {
        self.cpu_timeout = seconds;
        self.wallclock_timeout = seconds;
        self
    }

    /// Set the memory limit in MB
    #[must_use]
    pub fn with_memory_limit(mut self, mb: u32) -> Self {
        self.memory_limit = mb;
        self
    }
}

/// StarExec benchmark result format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarExecBenchmarkResult {
    /// Pair ID (unique identifier for benchmark-solver pair)
    pub pair_id: u64,
    /// Benchmark path/name
    pub benchmark: String,
    /// Benchmark ID
    pub benchmark_id: u64,
    /// Solver name
    pub solver: String,
    /// Solver configuration
    pub configuration: String,
    /// Result status
    pub status: String,
    /// CPU time in seconds
    pub cpu_time: f64,
    /// Wall clock time in seconds
    pub wallclock_time: f64,
    /// Memory usage in KB
    pub memory_kb: u64,
    /// Expected result (if known)
    pub expected: Option<String>,
    /// Whether result matches expected
    pub correct: Option<bool>,
}

impl StarExecBenchmarkResult {
    /// Create from a SingleResult
    #[must_use]
    pub fn from_single_result(
        result: &SingleResult,
        pair_id: u64,
        benchmark_id: u64,
        solver: &str,
        configuration: &str,
    ) -> Self {
        Self {
            pair_id,
            benchmark: result
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            benchmark_id,
            solver: solver.to_string(),
            configuration: configuration.to_string(),
            status: result.status.as_str().to_string(),
            cpu_time: result.time.as_secs_f64(),
            wallclock_time: result.time.as_secs_f64(),
            memory_kb: result.memory_bytes.unwrap_or(0) / 1024,
            expected: result.expected.map(|e| e.as_str().to_string()),
            correct: result.correct,
        }
    }

    /// Convert status to StarExec format
    #[must_use]
    pub fn starexec_status(&self) -> &str {
        match self.status.as_str() {
            "sat" => "starexec-result=sat",
            "unsat" => "starexec-result=unsat",
            "unknown" => "starexec-result=unknown",
            "timeout" => "starexec-result=timeout (cpu)",
            "memout" => "starexec-result=memout",
            "error" => "starexec-result=error",
            _ => "starexec-result=unknown",
        }
    }
}

/// StarExec format writer
pub struct StarExecWriter {
    solver_name: String,
    configuration: String,
    next_pair_id: u64,
    next_benchmark_id: u64,
}

impl StarExecWriter {
    /// Create a new writer
    #[must_use]
    pub fn new(solver_name: impl Into<String>, configuration: impl Into<String>) -> Self {
        Self {
            solver_name: solver_name.into(),
            configuration: configuration.into(),
            next_pair_id: 1,
            next_benchmark_id: 1,
        }
    }

    /// Write results in StarExec CSV format
    pub fn write_csv<W: Write>(
        &mut self,
        results: &[SingleResult],
        writer: &mut W,
    ) -> StarExecResult<()> {
        // Write header
        writeln!(
            writer,
            "pair id,benchmark,benchmark id,solver,configuration,status,cpu time,wallclock time,memory usage,expected,result"
        )?;

        // Write each result
        for result in results {
            let se_result = StarExecBenchmarkResult::from_single_result(
                result,
                self.next_pair_id,
                self.next_benchmark_id,
                &self.solver_name,
                &self.configuration,
            );

            let correct_str = match se_result.correct {
                Some(true) => "correct",
                Some(false) => "wrong",
                None => "-",
            };

            writeln!(
                writer,
                "{},{},{},{},{},{},{:.3},{:.3},{},{},{}",
                se_result.pair_id,
                se_result.benchmark,
                se_result.benchmark_id,
                se_result.solver,
                se_result.configuration,
                se_result.status,
                se_result.cpu_time,
                se_result.wallclock_time,
                se_result.memory_kb,
                se_result.expected.as_deref().unwrap_or("-"),
                correct_str,
            )?;

            self.next_pair_id += 1;
            self.next_benchmark_id += 1;
        }

        Ok(())
    }

    /// Write results to a CSV file
    pub fn write_csv_file(
        &mut self,
        results: &[SingleResult],
        path: impl AsRef<Path>,
    ) -> StarExecResult<()> {
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_csv(results, &mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Write solver output format (one file per benchmark)
    pub fn write_solver_output(
        &self,
        result: &SingleResult,
        output_dir: impl AsRef<Path>,
    ) -> StarExecResult<PathBuf> {
        let dir = output_dir.as_ref();
        fs::create_dir_all(dir)?;

        let filename = result
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let output_path = dir.join(format!("{}.output", filename));

        let mut file = fs::File::create(&output_path)?;

        // Write StarExec format output
        writeln!(file, "{}", result.status.as_str())?;
        writeln!(file, "starexec-result={}", result.status.as_str())?;
        writeln!(file, "starexec-time={:.3}", result.time.as_secs_f64())?;
        if let Some(mem) = result.memory_bytes {
            writeln!(file, "starexec-memory={}", mem / 1024)?;
        }

        Ok(output_path)
    }
}

/// StarExec format reader
pub struct StarExecReader;

impl StarExecReader {
    /// Parse a StarExec CSV results file
    pub fn parse_csv(path: impl AsRef<Path>) -> StarExecResult<Vec<StarExecBenchmarkResult>> {
        let content = fs::read_to_string(path)?;
        Self::parse_csv_content(&content)
    }

    /// Parse CSV content
    fn parse_csv_content(content: &str) -> StarExecResult<Vec<StarExecBenchmarkResult>> {
        let mut results = Vec::new();
        let mut lines = content.lines();

        // Skip header
        lines.next();

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 10 {
                continue;
            }

            let pair_id = parts[0]
                .parse()
                .map_err(|_| StarExecError::ParseError("Invalid pair_id".to_string()))?;
            let benchmark = parts[1].to_string();
            let benchmark_id = parts[2]
                .parse()
                .map_err(|_| StarExecError::ParseError("Invalid benchmark_id".to_string()))?;
            let solver = parts[3].to_string();
            let configuration = parts[4].to_string();
            let status = parts[5].to_string();
            let cpu_time = parts[6]
                .parse()
                .map_err(|_| StarExecError::ParseError("Invalid cpu_time".to_string()))?;
            let wallclock_time = parts[7]
                .parse()
                .map_err(|_| StarExecError::ParseError("Invalid wallclock_time".to_string()))?;
            let memory_kb = parts[8]
                .parse()
                .map_err(|_| StarExecError::ParseError("Invalid memory".to_string()))?;

            let expected = if parts.len() > 9 && parts[9] != "-" {
                Some(parts[9].to_string())
            } else {
                None
            };

            let correct = if parts.len() > 10 {
                match parts[10] {
                    "correct" => Some(true),
                    "wrong" => Some(false),
                    _ => None,
                }
            } else {
                None
            };

            results.push(StarExecBenchmarkResult {
                pair_id,
                benchmark,
                benchmark_id,
                solver,
                configuration,
                status,
                cpu_time,
                wallclock_time,
                memory_kb,
                expected,
                correct,
            });
        }

        Ok(results)
    }

    /// Parse StarExec solver output file
    pub fn parse_solver_output(path: impl AsRef<Path>) -> StarExecResult<StarExecOutput> {
        let content = fs::read_to_string(path)?;
        Self::parse_solver_output_content(&content)
    }

    /// Parse solver output content
    fn parse_solver_output_content(content: &str) -> StarExecResult<StarExecOutput> {
        let mut output = StarExecOutput::default();

        for line in content.lines() {
            let line = line.trim();

            if line.starts_with("starexec-result=") {
                output.result = Some(line.trim_start_matches("starexec-result=").to_string());
            } else if line.starts_with("starexec-time=") {
                if let Ok(time) = line.trim_start_matches("starexec-time=").parse() {
                    output.time = Some(time);
                }
            } else if line.starts_with("starexec-memory=") {
                if let Ok(mem) = line.trim_start_matches("starexec-memory=").parse() {
                    output.memory_kb = Some(mem);
                }
            } else if output.result.is_none() {
                // First non-starexec line is the status
                output.status = Some(line.to_string());
            }
        }

        Ok(output)
    }
}

/// Parsed StarExec solver output
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StarExecOutput {
    /// Status line (sat/unsat/unknown)
    pub status: Option<String>,
    /// StarExec result annotation
    pub result: Option<String>,
    /// CPU time
    pub time: Option<f64>,
    /// Memory usage in KB
    pub memory_kb: Option<u64>,
}

impl StarExecOutput {
    /// Convert to BenchmarkStatus
    #[must_use]
    pub fn to_benchmark_status(&self) -> BenchmarkStatus {
        let status = self.result.as_deref().or(self.status.as_deref());
        match status {
            Some("sat") => BenchmarkStatus::Sat,
            Some("unsat") => BenchmarkStatus::Unsat,
            Some("unknown") => BenchmarkStatus::Unknown,
            Some(s) if s.contains("timeout") => BenchmarkStatus::Timeout,
            Some(s) if s.contains("memout") => BenchmarkStatus::MemoryOut,
            _ => BenchmarkStatus::Unknown,
        }
    }
}

/// StarExec benchmark space format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarExecSpace {
    /// Space name
    pub name: String,
    /// Space ID
    pub id: u32,
    /// Parent space ID
    pub parent_id: Option<u32>,
    /// Benchmarks in this space
    pub benchmarks: Vec<String>,
    /// Child spaces
    pub children: Vec<StarExecSpace>,
}

impl StarExecSpace {
    /// Create a new space
    #[must_use]
    pub fn new(name: impl Into<String>, id: u32) -> Self {
        Self {
            name: name.into(),
            id,
            parent_id: None,
            benchmarks: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Add a benchmark
    pub fn add_benchmark(&mut self, benchmark: impl Into<String>) {
        self.benchmarks.push(benchmark.into());
    }

    /// Add a child space
    pub fn add_child(&mut self, mut child: StarExecSpace) {
        child.parent_id = Some(self.id);
        self.children.push(child);
    }

    /// Get total benchmark count (recursive)
    #[must_use]
    pub fn total_benchmarks(&self) -> usize {
        self.benchmarks.len()
            + self
                .children
                .iter()
                .map(|c| c.total_benchmarks())
                .sum::<usize>()
    }
}

/// Create benchmark space from metadata
#[must_use]
pub fn create_space_from_benchmarks(name: &str, benchmarks: &[BenchmarkMeta]) -> StarExecSpace {
    let mut space = StarExecSpace::new(name, 1);
    let mut logic_spaces: HashMap<String, StarExecSpace> = HashMap::new();
    let mut next_id = 2;

    for benchmark in benchmarks {
        let logic = benchmark
            .logic
            .clone()
            .unwrap_or_else(|| "UNKNOWN".to_string());

        let logic_space = logic_spaces.entry(logic.clone()).or_insert_with(|| {
            let s = StarExecSpace::new(&logic, next_id);
            next_id += 1;
            s
        });

        logic_space.add_benchmark(benchmark.path.to_string_lossy().to_string());
    }

    for (_, child) in logic_spaces {
        space.add_child(child);
    }

    space
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::ExpectedStatus;
    use std::time::Duration;

    fn make_result(status: BenchmarkStatus, time_ms: u64) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from("/tmp/test.smt2"),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(time_ms))
    }

    #[test]
    fn test_starexec_job_builder() {
        let job = StarExecJob::new("Test Job")
            .with_description("A test job")
            .with_timeout(600)
            .with_memory_limit(8192);

        assert_eq!(job.name, "Test Job");
        assert_eq!(job.description, Some("A test job".to_string()));
        assert_eq!(job.cpu_timeout, 600);
        assert_eq!(job.memory_limit, 8192);
    }

    #[test]
    fn test_starexec_result_from_single() {
        let result = make_result(BenchmarkStatus::Sat, 1234);
        let se_result =
            StarExecBenchmarkResult::from_single_result(&result, 1, 1, "OxiZ", "default");

        assert_eq!(se_result.status, "sat");
        assert!((se_result.cpu_time - 1.234).abs() < 0.001);
    }

    #[test]
    fn test_starexec_csv_write() {
        let results = vec![
            make_result(BenchmarkStatus::Sat, 100),
            make_result(BenchmarkStatus::Unsat, 200),
        ];

        let mut writer = StarExecWriter::new("OxiZ", "default");
        let mut output = Vec::new();
        writer.write_csv(&results, &mut output).unwrap();

        let csv = String::from_utf8(output).unwrap();
        assert!(csv.contains("pair id"));
        assert!(csv.contains("sat"));
        assert!(csv.contains("unsat"));
    }

    #[test]
    fn test_starexec_output_parse() {
        let content = "sat\nstarexec-result=sat\nstarexec-time=1.234\nstarexec-memory=1024\n";
        let output = StarExecReader::parse_solver_output_content(content).unwrap();

        assert_eq!(output.status, Some("sat".to_string()));
        assert_eq!(output.result, Some("sat".to_string()));
        assert_eq!(output.time, Some(1.234));
        assert_eq!(output.memory_kb, Some(1024));
        assert_eq!(output.to_benchmark_status(), BenchmarkStatus::Sat);
    }

    #[test]
    fn test_starexec_space() {
        let mut space = StarExecSpace::new("SMT-LIB", 1);
        let mut qf_lia = StarExecSpace::new("QF_LIA", 2);
        qf_lia.add_benchmark("test1.smt2");
        qf_lia.add_benchmark("test2.smt2");
        space.add_child(qf_lia);

        assert_eq!(space.total_benchmarks(), 2);
        assert_eq!(space.children[0].parent_id, Some(1));
    }
}
