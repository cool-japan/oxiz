//! Incremental result saving and resumption
//!
//! This module provides functionality to save benchmark results incrementally
//! and resume interrupted benchmark runs.

use crate::benchmark::SingleResult;
use crate::loader::BenchmarkMeta;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Error type for resumption operations
#[derive(Error, Debug)]
pub enum ResumptionError {
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// Checkpoint corrupted
    #[error("Checkpoint file corrupted: {0}")]
    Corrupted(String),
}

/// Result type for resumption operations
pub type ResumptionResult<T> = Result<T, ResumptionError>;

/// Checkpoint format for saving progress
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint version
    pub version: u32,
    /// Session ID
    pub session_id: String,
    /// Start time (Unix timestamp)
    pub start_time: u64,
    /// Last update time
    pub last_update: u64,
    /// Total benchmarks in the run
    pub total_benchmarks: usize,
    /// Number completed
    pub completed: usize,
    /// Completed benchmark paths
    pub completed_paths: HashSet<PathBuf>,
    /// Configuration used
    pub config: CheckpointConfig,
}

/// Configuration saved in checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Timeout per benchmark (seconds)
    pub timeout_secs: u64,
    /// Memory limit (bytes)
    pub memory_limit: u64,
    /// Logic filter
    pub logic_filter: Vec<String>,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 60,
            memory_limit: 0,
            logic_filter: Vec::new(),
        }
    }
}

impl Checkpoint {
    /// Create a new checkpoint
    #[must_use]
    pub fn new(session_id: impl Into<String>, total: usize, config: CheckpointConfig) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            version: 1,
            session_id: session_id.into(),
            start_time: now,
            last_update: now,
            total_benchmarks: total,
            completed: 0,
            completed_paths: HashSet::new(),
            config,
        }
    }

    /// Mark a benchmark as completed
    pub fn mark_completed(&mut self, path: &Path) {
        if self.completed_paths.insert(path.to_path_buf()) {
            self.completed += 1;
            self.last_update = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
    }

    /// Check if a benchmark is completed
    #[must_use]
    pub fn is_completed(&self, path: &Path) -> bool {
        self.completed_paths.contains(path)
    }

    /// Get progress percentage
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.total_benchmarks == 0 {
            100.0
        } else {
            (self.completed as f64 / self.total_benchmarks as f64) * 100.0
        }
    }

    /// Get remaining benchmarks count
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.total_benchmarks.saturating_sub(self.completed)
    }

    /// Check if run is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed >= self.total_benchmarks
    }
}

/// Result saver for incremental saving
pub struct ResultSaver {
    /// Path to results file
    results_path: PathBuf,
    /// Path to checkpoint file
    checkpoint_path: PathBuf,
    /// Current checkpoint
    checkpoint: Checkpoint,
    /// Results writer (for JSON Lines format)
    writer: Option<BufWriter<File>>,
}

impl ResultSaver {
    /// Create a new result saver
    pub fn new(
        output_dir: impl AsRef<Path>,
        session_id: impl Into<String>,
        total_benchmarks: usize,
        config: CheckpointConfig,
    ) -> ResumptionResult<Self> {
        let dir = output_dir.as_ref();
        fs::create_dir_all(dir)?;

        let session = session_id.into();
        let results_path = dir.join(format!("{}.jsonl", session));
        let checkpoint_path = dir.join(format!("{}.checkpoint.json", session));

        let checkpoint = Checkpoint::new(&session, total_benchmarks, config);

        // Open results file for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&results_path)?;
        let writer = Some(BufWriter::new(file));

        Ok(Self {
            results_path,
            checkpoint_path,
            checkpoint,
            writer,
        })
    }

    /// Try to resume from existing checkpoint
    pub fn try_resume(
        output_dir: impl AsRef<Path>,
        session_id: impl Into<String>,
    ) -> ResumptionResult<Option<Self>> {
        let dir = output_dir.as_ref();
        let session = session_id.into();
        let checkpoint_path = dir.join(format!("{}.checkpoint.json", session));
        let results_path = dir.join(format!("{}.jsonl", session));

        if !checkpoint_path.exists() {
            return Ok(None);
        }

        // Load checkpoint
        let checkpoint_content = fs::read_to_string(&checkpoint_path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&checkpoint_content)?;

        // Open results file for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&results_path)?;
        let writer = Some(BufWriter::new(file));

        Ok(Some(Self {
            results_path,
            checkpoint_path,
            checkpoint,
            writer,
        }))
    }

    /// Save a single result
    pub fn save_result(&mut self, result: &SingleResult) -> ResumptionResult<()> {
        // Write result as JSON line
        if let Some(ref mut writer) = self.writer {
            let json = serde_json::to_string(result)?;
            writeln!(writer, "{}", json)?;
            writer.flush()?;
        }

        // Update checkpoint
        self.checkpoint.mark_completed(&result.path);

        // Periodically save checkpoint (every 10 results)
        if self.checkpoint.completed.is_multiple_of(10) {
            self.save_checkpoint()?;
        }

        Ok(())
    }

    /// Save multiple results
    pub fn save_results(&mut self, results: &[SingleResult]) -> ResumptionResult<()> {
        for result in results {
            self.save_result(result)?;
        }
        self.save_checkpoint()?;
        Ok(())
    }

    /// Save checkpoint to file
    pub fn save_checkpoint(&self) -> ResumptionResult<()> {
        let json = serde_json::to_string_pretty(&self.checkpoint)?;
        fs::write(&self.checkpoint_path, json)?;
        Ok(())
    }

    /// Finalize and close
    pub fn finalize(mut self) -> ResumptionResult<()> {
        if let Some(ref mut writer) = self.writer {
            writer.flush()?;
        }
        self.save_checkpoint()?;
        Ok(())
    }

    /// Get current checkpoint
    #[must_use]
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    /// Check if benchmark should be skipped (already completed)
    #[must_use]
    pub fn should_skip(&self, path: &Path) -> bool {
        self.checkpoint.is_completed(path)
    }

    /// Get results file path
    #[must_use]
    pub fn results_path(&self) -> &Path {
        &self.results_path
    }

    /// Get checkpoint file path
    #[must_use]
    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }
}

/// Result loader for reading saved results
pub struct ResultLoader;

impl ResultLoader {
    /// Load results from JSON Lines file
    pub fn load(path: impl AsRef<Path>) -> ResumptionResult<Vec<SingleResult>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut results = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let result: SingleResult = serde_json::from_str(&line)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Load checkpoint from file
    pub fn load_checkpoint(path: impl AsRef<Path>) -> ResumptionResult<Checkpoint> {
        let content = fs::read_to_string(path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&content)?;
        Ok(checkpoint)
    }

    /// Get list of completed benchmark paths from checkpoint
    pub fn get_completed_paths(
        checkpoint_path: impl AsRef<Path>,
    ) -> ResumptionResult<HashSet<PathBuf>> {
        let checkpoint = Self::load_checkpoint(checkpoint_path)?;
        Ok(checkpoint.completed_paths)
    }
}

/// Filter benchmarks that haven't been completed
pub fn filter_remaining(
    benchmarks: &[BenchmarkMeta],
    completed: &HashSet<PathBuf>,
) -> Vec<BenchmarkMeta> {
    benchmarks
        .iter()
        .filter(|b| !completed.contains(&b.path))
        .cloned()
        .collect()
}

/// Session manager for handling multiple benchmark runs
pub struct SessionManager {
    output_dir: PathBuf,
}

impl SessionManager {
    /// Create a new session manager
    #[must_use]
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }

    /// Generate a unique session ID
    #[must_use]
    pub fn generate_session_id() -> String {
        use std::time::SystemTime;
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("session_{}", timestamp)
    }

    /// List all sessions
    pub fn list_sessions(&self) -> ResumptionResult<Vec<SessionInfo>> {
        let mut sessions = Vec::new();

        if !self.output_dir.exists() {
            return Ok(sessions);
        }

        for entry in fs::read_dir(&self.output_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|e| e == "json")
                && path.to_string_lossy().contains(".checkpoint")
                && let Ok(checkpoint) = ResultLoader::load_checkpoint(&path)
            {
                let is_complete = checkpoint.is_complete();
                sessions.push(SessionInfo {
                    session_id: checkpoint.session_id,
                    start_time: checkpoint.start_time,
                    completed: checkpoint.completed,
                    total: checkpoint.total_benchmarks,
                    is_complete,
                });
            }
        }

        sessions.sort_by_key(|s| std::cmp::Reverse(s.start_time));
        Ok(sessions)
    }

    /// Get incomplete sessions
    pub fn list_incomplete(&self) -> ResumptionResult<Vec<SessionInfo>> {
        Ok(self
            .list_sessions()?
            .into_iter()
            .filter(|s| !s.is_complete)
            .collect())
    }

    /// Delete a session's files
    pub fn delete_session(&self, session_id: &str) -> ResumptionResult<()> {
        let results_path = self.output_dir.join(format!("{}.jsonl", session_id));
        let checkpoint_path = self
            .output_dir
            .join(format!("{}.checkpoint.json", session_id));

        if results_path.exists() {
            fs::remove_file(results_path)?;
        }
        if checkpoint_path.exists() {
            fs::remove_file(checkpoint_path)?;
        }

        Ok(())
    }
}

/// Information about a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Session ID
    pub session_id: String,
    /// Start time (Unix timestamp)
    pub start_time: u64,
    /// Completed benchmarks
    pub completed: usize,
    /// Total benchmarks
    pub total: usize,
    /// Whether session is complete
    pub is_complete: bool,
}

impl SessionInfo {
    /// Get progress percentage
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            (self.completed as f64 / self.total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::BenchmarkStatus;
    use crate::loader::ExpectedStatus;
    use std::time::Duration;
    use tempfile::tempdir;

    fn make_result(name: &str, status: BenchmarkStatus) -> SingleResult {
        let meta = BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/{}", name)),
            logic: Some("QF_LIA".to_string()),
            expected_status: Some(ExpectedStatus::Sat),
            file_size: 100,
            category: None,
        };
        SingleResult::new(&meta, status, Duration::from_millis(100))
    }

    #[test]
    fn test_checkpoint_creation() {
        let config = CheckpointConfig::default();
        let mut checkpoint = Checkpoint::new("test_session", 100, config);

        assert_eq!(checkpoint.total_benchmarks, 100);
        assert_eq!(checkpoint.completed, 0);
        assert!(!checkpoint.is_complete());
        assert_eq!(checkpoint.progress(), 0.0);

        checkpoint.mark_completed(Path::new("/tmp/test.smt2"));
        assert_eq!(checkpoint.completed, 1);
        assert!(checkpoint.is_completed(Path::new("/tmp/test.smt2")));
    }

    #[test]
    fn test_result_saver() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::default();
        let mut saver = ResultSaver::new(dir.path(), "test", 10, config).unwrap();

        let result = make_result("test.smt2", BenchmarkStatus::Sat);
        saver.save_result(&result).unwrap();

        assert_eq!(saver.checkpoint().completed, 1);
        assert!(saver.should_skip(Path::new("/tmp/test.smt2")));
        assert!(!saver.should_skip(Path::new("/tmp/other.smt2")));

        saver.finalize().unwrap();

        // Verify files were created
        assert!(dir.path().join("test.jsonl").exists());
        assert!(dir.path().join("test.checkpoint.json").exists());
    }

    #[test]
    fn test_result_loader() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::default();

        // Save some results
        {
            let mut saver = ResultSaver::new(dir.path(), "test", 3, config).unwrap();
            saver
                .save_result(&make_result("a.smt2", BenchmarkStatus::Sat))
                .unwrap();
            saver
                .save_result(&make_result("b.smt2", BenchmarkStatus::Unsat))
                .unwrap();
            saver.finalize().unwrap();
        }

        // Load results
        let results = ResultLoader::load(dir.path().join("test.jsonl")).unwrap();
        assert_eq!(results.len(), 2);

        // Load checkpoint
        let checkpoint =
            ResultLoader::load_checkpoint(dir.path().join("test.checkpoint.json")).unwrap();
        assert_eq!(checkpoint.completed, 2);
    }

    #[test]
    fn test_session_resumption() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::default();

        // Start a session
        {
            let mut saver =
                ResultSaver::new(dir.path(), "resume_test", 10, config.clone()).unwrap();
            saver
                .save_result(&make_result("a.smt2", BenchmarkStatus::Sat))
                .unwrap();
            saver.finalize().unwrap();
        }

        // Try to resume
        let resumed = ResultSaver::try_resume(dir.path(), "resume_test").unwrap();
        assert!(resumed.is_some());

        let saver = resumed.unwrap();
        assert_eq!(saver.checkpoint().completed, 1);
        assert!(saver.should_skip(Path::new("/tmp/a.smt2")));
    }

    #[test]
    fn test_filter_remaining() {
        let benchmarks = vec![
            BenchmarkMeta {
                path: PathBuf::from("/tmp/a.smt2"),
                logic: None,
                expected_status: None,
                file_size: 0,
                category: None,
            },
            BenchmarkMeta {
                path: PathBuf::from("/tmp/b.smt2"),
                logic: None,
                expected_status: None,
                file_size: 0,
                category: None,
            },
        ];

        let mut completed = HashSet::new();
        completed.insert(PathBuf::from("/tmp/a.smt2"));

        let remaining = filter_remaining(&benchmarks, &completed);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].path, PathBuf::from("/tmp/b.smt2"));
    }
}
