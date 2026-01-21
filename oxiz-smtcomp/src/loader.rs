//! Benchmark file discovery and loading
//!
//! This module provides functionality to discover and load SMT-LIB2 benchmark files
//! organized by logic (e.g., QF_LIA, QF_BV, QF_UF).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

/// Error type for loader operations
#[derive(Error, Debug)]
pub enum LoaderError {
    /// IO error when reading files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Walk directory error
    #[error("Directory walk error: {0}")]
    WalkDir(#[from] walkdir::Error),
    /// Invalid benchmark file
    #[error("Invalid benchmark file: {0}")]
    InvalidBenchmark(String),
}

/// Result type for loader operations
pub type LoaderResult<T> = Result<T, LoaderError>;

/// Metadata extracted from a benchmark file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeta {
    /// Path to the benchmark file
    pub path: PathBuf,
    /// Logic specified in the file (e.g., QF_LIA)
    pub logic: Option<String>,
    /// Expected status if known (sat, unsat, or unknown)
    pub expected_status: Option<ExpectedStatus>,
    /// File size in bytes
    pub file_size: u64,
    /// Category derived from directory structure
    pub category: Option<String>,
}

/// Expected benchmark result status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedStatus {
    /// Expected to be satisfiable
    Sat,
    /// Expected to be unsatisfiable
    Unsat,
    /// Status unknown
    Unknown,
}

impl ExpectedStatus {
    /// Parse status from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "sat" => Some(Self::Sat),
            "unsat" => Some(Self::Unsat),
            "unknown" => Some(Self::Unknown),
            _ => None,
        }
    }

    /// Convert to SMT-COMP format string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sat => "sat",
            Self::Unsat => "unsat",
            Self::Unknown => "unknown",
        }
    }
}

/// A loaded benchmark with its content
#[derive(Debug, Clone)]
pub struct Benchmark {
    /// Metadata about the benchmark
    pub meta: BenchmarkMeta,
    /// Raw content of the benchmark file
    pub content: String,
}

/// Configuration for the benchmark loader
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    /// Root directory to search for benchmarks
    pub root_dir: PathBuf,
    /// File extension to look for (default: ".smt2")
    pub extension: String,
    /// Maximum file size to load (in bytes, default: 10MB)
    pub max_file_size: u64,
    /// Filter by specific logics (empty means all)
    pub logic_filter: Vec<String>,
    /// Maximum number of files to load (0 means unlimited)
    pub max_files: usize,
    /// Recursive search in subdirectories
    pub recursive: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            root_dir: PathBuf::from("."),
            extension: ".smt2".to_string(),
            max_file_size: 10 * 1024 * 1024, // 10MB
            logic_filter: Vec::new(),
            max_files: 0, // unlimited
            recursive: true,
        }
    }
}

impl LoaderConfig {
    /// Create a new config with the given root directory
    #[must_use]
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
            ..Default::default()
        }
    }

    /// Set the extension filter
    #[must_use]
    pub fn with_extension(mut self, ext: impl Into<String>) -> Self {
        self.extension = ext.into();
        self
    }

    /// Set the maximum file size
    #[must_use]
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size;
        self
    }

    /// Set the logic filter
    #[must_use]
    pub fn with_logics(mut self, logics: Vec<String>) -> Self {
        self.logic_filter = logics;
        self
    }

    /// Set the maximum number of files
    #[must_use]
    pub fn with_max_files(mut self, max: usize) -> Self {
        self.max_files = max;
        self
    }

    /// Set whether to search recursively
    #[must_use]
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }
}

/// Benchmark loader for discovering and loading SMT-LIB2 files
pub struct Loader {
    config: LoaderConfig,
}

impl Loader {
    /// Create a new loader with the given configuration
    #[must_use]
    pub fn new(config: LoaderConfig) -> Self {
        Self { config }
    }

    /// Create a loader with default configuration for the given directory
    #[must_use]
    pub fn for_directory(dir: impl Into<PathBuf>) -> Self {
        Self::new(LoaderConfig::new(dir))
    }

    /// Discover all benchmark files matching the configuration
    pub fn discover(&self) -> LoaderResult<Vec<BenchmarkMeta>> {
        let mut benchmarks = Vec::new();
        let mut walker = WalkDir::new(&self.config.root_dir);

        if !self.config.recursive {
            walker = walker.max_depth(1);
        }

        for entry in walker {
            let entry = entry?;
            let path = entry.path();

            // Check if it's a file with the right extension
            if !path.is_file() {
                continue;
            }
            if path
                .extension()
                .is_none_or(|ext| ext != self.config.extension.trim_start_matches('.'))
            {
                continue;
            }

            // Check file size
            let metadata = fs::metadata(path)?;
            if metadata.len() > self.config.max_file_size {
                continue;
            }

            // Extract metadata from path and file
            let meta = self.extract_metadata(path, metadata.len())?;

            // Apply logic filter
            if !self.config.logic_filter.is_empty() {
                if let Some(ref logic) = meta.logic {
                    if !self.config.logic_filter.contains(logic) {
                        continue;
                    }
                } else {
                    continue; // Skip files without logic if filter is set
                }
            }

            benchmarks.push(meta);

            // Check max files limit
            if self.config.max_files > 0 && benchmarks.len() >= self.config.max_files {
                break;
            }
        }

        Ok(benchmarks)
    }

    /// Discover and group benchmarks by logic
    pub fn discover_by_logic(&self) -> LoaderResult<HashMap<String, Vec<BenchmarkMeta>>> {
        let benchmarks = self.discover()?;
        let mut by_logic: HashMap<String, Vec<BenchmarkMeta>> = HashMap::new();

        for bench in benchmarks {
            let logic = bench.logic.clone().unwrap_or_else(|| "UNKNOWN".to_string());
            by_logic.entry(logic).or_default().push(bench);
        }

        Ok(by_logic)
    }

    /// Load a benchmark file given its metadata
    pub fn load(&self, meta: &BenchmarkMeta) -> LoaderResult<Benchmark> {
        let content = fs::read_to_string(&meta.path)?;
        Ok(Benchmark {
            meta: meta.clone(),
            content,
        })
    }

    /// Load a benchmark file directly from path
    pub fn load_file(&self, path: impl AsRef<Path>) -> LoaderResult<Benchmark> {
        let path = path.as_ref();
        let metadata = fs::metadata(path)?;
        let meta = self.extract_metadata(path, metadata.len())?;
        self.load(&meta)
    }

    /// Extract metadata from a benchmark file
    fn extract_metadata(&self, path: &Path, file_size: u64) -> LoaderResult<BenchmarkMeta> {
        // Try to extract logic from directory structure first (e.g., QF_LIA/...)
        let logic = self.extract_logic_from_path(path);
        let category = self.extract_category_from_path(path);

        // For expected status, try to extract from filename or read file header
        let expected_status = self.extract_expected_status(path);

        Ok(BenchmarkMeta {
            path: path.to_path_buf(),
            logic,
            expected_status,
            file_size,
            category,
        })
    }

    /// Extract logic from path (e.g., /benchmarks/QF_LIA/sat/test.smt2 -> QF_LIA)
    fn extract_logic_from_path(&self, path: &Path) -> Option<String> {
        // Common SMT-COMP logics
        let known_logics = [
            "ALIA",
            "AUFLIA",
            "AUFLIRA",
            "AUFNIRA",
            "BV",
            "LIA",
            "LRA",
            "NIA",
            "NRA",
            "QF_ABV",
            "QF_ALIA",
            "QF_AUFBV",
            "QF_AUFLIA",
            "QF_AX",
            "QF_BV",
            "QF_BVFP",
            "QF_DT",
            "QF_FP",
            "QF_IDL",
            "QF_LIA",
            "QF_LIRA",
            "QF_LRA",
            "QF_NIA",
            "QF_NIRA",
            "QF_NRA",
            "QF_RDL",
            "QF_S",
            "QF_SLIA",
            "QF_UF",
            "QF_UFBV",
            "QF_UFIDL",
            "QF_UFLIA",
            "QF_UFLRA",
            "QF_UFNIA",
            "QF_UFNRA",
            "UF",
            "UFBV",
            "UFDT",
            "UFIDL",
            "UFLIA",
            "UFLRA",
            "UFNIA",
        ];

        for component in path.components() {
            let s = component.as_os_str().to_string_lossy();
            if known_logics.contains(&s.as_ref()) {
                return Some(s.to_string());
            }
        }

        // Also try to read from file header
        if let Ok(content) = fs::read_to_string(path) {
            return Self::extract_logic_from_content(&content);
        }

        None
    }

    /// Extract logic from file content (set-logic command)
    fn extract_logic_from_content(content: &str) -> Option<String> {
        for line in content.lines().take(50) {
            // Check first 50 lines
            let trimmed = line.trim();
            if trimmed.starts_with("(set-logic") {
                // Extract logic name: (set-logic QF_LIA)
                if let Some(start) = trimmed.find("set-logic") {
                    let rest = &trimmed[start + 9..];
                    let logic = rest.trim().trim_start_matches(|c: char| c.is_whitespace());
                    let logic = logic.trim_end_matches(')').trim();
                    if !logic.is_empty() {
                        return Some(logic.to_string());
                    }
                }
            }
        }
        None
    }

    /// Extract category from path structure
    fn extract_category_from_path(&self, path: &Path) -> Option<String> {
        let relative = path.strip_prefix(&self.config.root_dir).ok()?;
        let components: Vec<_> = relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .collect();

        // Skip the last component (filename) and first (logic)
        if components.len() >= 2 {
            let category_parts: Vec<_> = components[..components.len() - 1].to_vec();
            Some(category_parts.join("/"))
        } else {
            None
        }
    }

    /// Extract expected status from filename or content
    fn extract_expected_status(&self, path: &Path) -> Option<ExpectedStatus> {
        // Check filename for hints
        if let Some(stem) = path.file_stem() {
            let name = stem.to_string_lossy().to_lowercase();
            if name.contains("sat") && !name.contains("unsat") {
                return Some(ExpectedStatus::Sat);
            }
            if name.contains("unsat") {
                return Some(ExpectedStatus::Unsat);
            }
        }

        // Check parent directory
        if let Some(parent) = path.parent()
            && let Some(dir_name) = parent.file_name()
        {
            let name = dir_name.to_string_lossy().to_lowercase();
            if name == "sat" {
                return Some(ExpectedStatus::Sat);
            }
            if name == "unsat" {
                return Some(ExpectedStatus::Unsat);
            }
        }

        // Check file content for status annotation
        if let Ok(content) = fs::read_to_string(path) {
            return Self::extract_status_from_content(&content);
        }

        None
    }

    /// Extract expected status from file content (SMT-LIB :status info)
    fn extract_status_from_content(content: &str) -> Option<ExpectedStatus> {
        for line in content.lines().take(50) {
            let trimmed = line.trim();
            // Look for (set-info :status sat/unsat/unknown)
            if trimmed.starts_with("(set-info :status") {
                let rest = trimmed.trim_start_matches("(set-info :status").trim();
                let status = rest.trim_end_matches(')').trim();
                return ExpectedStatus::parse(status);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_status_from_str() {
        assert_eq!(ExpectedStatus::parse("sat"), Some(ExpectedStatus::Sat));
        assert_eq!(ExpectedStatus::parse("unsat"), Some(ExpectedStatus::Unsat));
        assert_eq!(
            ExpectedStatus::parse("unknown"),
            Some(ExpectedStatus::Unknown)
        );
        assert_eq!(ExpectedStatus::parse("SAT"), Some(ExpectedStatus::Sat));
        assert_eq!(ExpectedStatus::parse("invalid"), None);
    }

    #[test]
    fn test_extract_logic_from_content() {
        let content = "(set-logic QF_LIA)\n(declare-const x Int)";
        assert_eq!(
            Loader::extract_logic_from_content(content),
            Some("QF_LIA".to_string())
        );

        let content_spaces = "(set-logic   QF_BV  )";
        assert_eq!(
            Loader::extract_logic_from_content(content_spaces),
            Some("QF_BV".to_string())
        );
    }

    #[test]
    fn test_extract_status_from_content() {
        let content = "(set-info :status sat)\n(declare-const x Int)";
        assert_eq!(
            Loader::extract_status_from_content(content),
            Some(ExpectedStatus::Sat)
        );

        let content_unsat = "(set-info :status unsat)";
        assert_eq!(
            Loader::extract_status_from_content(content_unsat),
            Some(ExpectedStatus::Unsat)
        );
    }

    #[test]
    fn test_loader_config_builder() {
        let config = LoaderConfig::new("/tmp/benchmarks")
            .with_extension(".smt2")
            .with_max_file_size(5 * 1024 * 1024)
            .with_logics(vec!["QF_LIA".to_string()])
            .with_max_files(100)
            .with_recursive(false);

        assert_eq!(config.root_dir, PathBuf::from("/tmp/benchmarks"));
        assert_eq!(config.extension, ".smt2");
        assert_eq!(config.max_file_size, 5 * 1024 * 1024);
        assert_eq!(config.logic_filter, vec!["QF_LIA".to_string()]);
        assert_eq!(config.max_files, 100);
        assert!(!config.recursive);
    }
}
