//! SV-COMP benchmark suite integration
//!
//! This module provides support for reading SV-COMP (Software Verification Competition)
//! task definition YAML files and integrating them with the OxiZ SMT benchmark pipeline.
//!
//! SV-COMP tasks can specify input files of various types; only those pointing at SMT-LIB2
//! (`.smt2`) files are accepted. Tasks referencing only C/C++ source files (`.c`, `.i`, `.cpp`)
//! are skipped with [`SkippedReason::NotSmtLib`]. YAML files that cannot be parsed are recorded
//! as [`SkippedReason::YamlParseError`].
//!
//! # Example
//!
//! ```no_run
//! use oxiz_smtcomp::svcomp::SvCompReader;
//! use std::path::Path;
//!
//! let reader = SvCompReader::discover(Path::new("/benchmarks/sv-comp")).unwrap();
//! println!("Tasks: {}, Skipped: {}", reader.tasks().len(), reader.skipped_count());
//! let metas = reader.to_benchmark_meta();
//! ```

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

use crate::loader::{BenchmarkMeta, ExpectedStatus};

/// Error type for SV-COMP operations
#[derive(Debug, Error)]
pub enum SvCompError {
    /// I/O error encountered during directory traversal or file reading
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// YAML parse failure for a task file (fatal: cannot continue discovery)
    ///
    /// Note: per-file parse failures are captured in [`SkippedReason::YamlParseError`]
    /// and do not surface as this variant during `discover`. This variant is reserved for
    /// future use when a single mandatory task file must be parsed.
    #[error("YAML parse error: {0}")]
    YamlParse(#[from] serde_yaml::Error),
    /// Task YAML was parsed, but no `.smt2` source could be found in it
    #[error("Missing SMT-LIB source in task: {path}")]
    MissingSmtLibSource {
        /// Path to the problematic task file
        path: PathBuf,
    },
}

/// Reason why an SV-COMP task was skipped during discovery
#[derive(Debug, Clone)]
pub enum SkippedReason {
    /// Task's input files do not include any `.smt2` file (e.g., C source only)
    NotSmtLib,
    /// The task YAML file could not be parsed
    YamlParseError(String),
}

/// Raw SV-COMP 2024+ task YAML schema
///
/// Fields are all optional to tolerate forward-compatibility and schema variations
/// across SV-COMP years.
#[derive(Debug, Deserialize)]
struct SvCompTaskYaml {
    /// Format version string (e.g., `"2.0"`)
    format_version: Option<String>,
    /// List of input file paths relative to the YAML file
    input_files: Option<Vec<String>>,
    /// Verification properties
    properties: Option<Vec<SvCompProperty>>,
    /// Top-level expected verdict (`"true"` or `"false"`)
    expected_verdict: Option<String>,
    /// Additional solver or tool options
    options: Option<HashMap<String, String>>,
}

/// A single verification property entry in an SV-COMP task
#[derive(Debug, Deserialize)]
struct SvCompProperty {
    /// Path to the `.prp` property file
    property_file: Option<String>,
    /// Expected verdict for this property (`"true"` or `"false"`)
    expected_verdict: Option<String>,
}

/// A parsed SV-COMP task that contains at least one `.smt2` source file
#[derive(Debug, Clone)]
pub struct SvCompTask {
    /// Human-readable name derived from the YAML filename (without extension)
    pub name: String,
    /// Resolved absolute paths of the `.smt2` input files
    pub sources: Vec<PathBuf>,
    /// Raw expected verdict string (`"true"`, `"false"`, `"sat"`, `"unsat"`, …)
    pub expected_verdict: Option<String>,
    /// Additional key/value options from the task YAML
    pub options: HashMap<String, String>,
    /// The SV-COMP format version declared in the YAML (if present)
    pub format_version: Option<String>,
}

impl SvCompTask {
    /// Map the raw `expected_verdict` string to an [`ExpectedStatus`].
    ///
    /// SV-COMP verdicts use `"true"` (property holds, no counterexample → `Unsat`) and
    /// `"false"` (counterexample found → `Sat`), in addition to the SMT-LIB conventions
    /// `"sat"` / `"unsat"` / `"unknown"`.
    #[must_use]
    pub fn expected_status(&self) -> Option<ExpectedStatus> {
        match self.expected_verdict.as_deref() {
            Some("true") => Some(ExpectedStatus::Unsat),
            Some("false") => Some(ExpectedStatus::Sat),
            Some(s) => ExpectedStatus::parse(s),
            None => None,
        }
    }
}

/// Determines whether a file extension belongs to an SMT-LIB2 benchmark
fn is_smt2_extension(ext: &str) -> bool {
    ext.eq_ignore_ascii_case("smt2")
}

/// Determines whether a file extension is a C/C++ source that disqualifies a task
fn is_c_source_extension(ext: &str) -> bool {
    matches!(
        ext.to_lowercase().as_str(),
        "c" | "i" | "cpp" | "cc" | "cxx" | "c++"
    )
}

/// Reader that discovers and normalises SV-COMP task YAML files
pub struct SvCompReader {
    tasks: Vec<SvCompTask>,
    skipped: Vec<(PathBuf, SkippedReason)>,
}

impl SvCompReader {
    /// Walk `root` recursively, find all `.yml` / `.yaml` files, parse them as SV-COMP
    /// task definitions, and classify each as an accepted [`SvCompTask`] or a skip.
    ///
    /// A task is accepted when at least one of its `input_files` has a `.smt2` extension.
    /// Tasks whose input files are exclusively C-source are marked [`SkippedReason::NotSmtLib`].
    /// Tasks that fail YAML parsing are marked [`SkippedReason::YamlParseError`].
    ///
    /// Returns [`SvCompError::Io`] only for unrecoverable I/O problems such as being unable
    /// to read the root directory itself.
    pub fn discover(root: &Path) -> Result<Self, SvCompError> {
        let mut tasks: Vec<SvCompTask> = Vec::new();
        let mut skipped: Vec<(PathBuf, SkippedReason)> = Vec::new();

        for entry in WalkDir::new(root).follow_links(false) {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    // Silently skip unreadable sub-directories; only fail for the root
                    if err.depth() == 0 {
                        return Err(SvCompError::Io(
                            err.into_io_error()
                                .unwrap_or_else(|| std::io::Error::other("walkdir root error")),
                        ));
                    }
                    continue;
                }
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let path = entry.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            if !ext.eq_ignore_ascii_case("yml") && !ext.eq_ignore_ascii_case("yaml") {
                continue;
            }

            let content = match fs::read_to_string(path) {
                Ok(c) => c,
                Err(err) => {
                    skipped.push((
                        path.to_path_buf(),
                        SkippedReason::YamlParseError(err.to_string()),
                    ));
                    continue;
                }
            };

            let yaml: SvCompTaskYaml = match serde_yaml::from_str(&content) {
                Ok(y) => y,
                Err(err) => {
                    skipped.push((
                        path.to_path_buf(),
                        SkippedReason::YamlParseError(err.to_string()),
                    ));
                    continue;
                }
            };

            // Categorise input files
            let input_files = yaml.input_files.unwrap_or_default();
            let task_dir = path.parent().unwrap_or(root);

            let mut smt2_sources: Vec<PathBuf> = Vec::new();
            let mut has_c_source = false;

            for input in &input_files {
                let file_path = PathBuf::from(input);
                let file_ext = file_path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("");

                if is_smt2_extension(file_ext) {
                    // Resolve relative to the directory that contains the YAML
                    let resolved = if file_path.is_absolute() {
                        file_path
                    } else {
                        task_dir.join(&file_path)
                    };
                    smt2_sources.push(resolved);
                } else if is_c_source_extension(file_ext) {
                    has_c_source = true;
                }
            }

            if !smt2_sources.is_empty() {
                // Derive expected verdict: prefer property-level verdict if all agree,
                // otherwise fall back to the top-level field.
                let expected_verdict = resolve_expected_verdict(&yaml.expected_verdict, &yaml.properties);

                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                tasks.push(SvCompTask {
                    name,
                    sources: smt2_sources,
                    expected_verdict,
                    options: yaml.options.unwrap_or_default(),
                    format_version: yaml.format_version,
                });
            } else if has_c_source || !input_files.is_empty() {
                skipped.push((path.to_path_buf(), SkippedReason::NotSmtLib));
            }
            // YAML with empty input_files is silently ignored (not an error, not a skip)
        }

        Ok(Self { tasks, skipped })
    }

    /// Return a slice of all successfully parsed SMT-LIB tasks
    #[must_use]
    pub fn tasks(&self) -> &[SvCompTask] {
        &self.tasks
    }

    /// Return the number of task files that were skipped during discovery
    #[must_use]
    pub fn skipped_count(&self) -> usize {
        self.skipped.len()
    }

    /// Return a reference to the full list of skipped paths and their reasons
    #[must_use]
    pub fn skipped(&self) -> &[(PathBuf, SkippedReason)] {
        &self.skipped
    }

    /// Convert each accepted [`SvCompTask`] to a [`BenchmarkMeta`].
    ///
    /// One [`BenchmarkMeta`] is produced per `.smt2` source file within a task.
    /// The `logic` field is left as `None`; the caller should run logic detection
    /// (see `oxiz_smtcomp::logic_detector`) if needed.
    ///
    /// `category` is set to `Some("sv-comp")` so downstream reporters can group
    /// these benchmarks separately from standard SMT-LIB suites.
    #[must_use]
    pub fn to_benchmark_meta(&self) -> Vec<BenchmarkMeta> {
        let mut metas = Vec::new();
        for task in &self.tasks {
            let expected_status = task.expected_status();
            for source in &task.sources {
                let file_size = fs::metadata(source).map(|m| m.len()).unwrap_or(0);
                metas.push(BenchmarkMeta {
                    path: source.clone(),
                    logic: None,
                    expected_status,
                    file_size,
                    category: Some("sv-comp".to_string()),
                    structural_features: None,
                });
            }
        }
        metas
    }
}

/// Resolve the expected verdict from the task's top-level and property-level fields.
///
/// If all properties agree on a verdict, that verdict is returned.
/// If they disagree or there are no properties, fall back to the top-level field.
fn resolve_expected_verdict(
    top_level: &Option<String>,
    properties: &Option<Vec<SvCompProperty>>,
) -> Option<String> {
    if let Some(props) = properties {
        // Collect distinct non-None verdicts from properties
        let verdicts: Vec<&str> = props
            .iter()
            .filter_map(|p| p.expected_verdict.as_deref())
            .collect();

        if !verdicts.is_empty() {
            let first = verdicts[0];
            if verdicts.iter().all(|v| *v == first) {
                return Some(first.to_string());
            }
            // Verdicts disagree: fall through to top-level
        }
    }
    top_level.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_temp_dir() -> TempDir {
        tempfile::TempDir::new().expect("temp dir should be created in tests")
    }

    #[test]
    fn test_is_smt2_extension() {
        assert!(is_smt2_extension("smt2"));
        assert!(is_smt2_extension("SMT2"));
        assert!(!is_smt2_extension("c"));
        assert!(!is_smt2_extension("yml"));
    }

    #[test]
    fn test_is_c_source_extension() {
        assert!(is_c_source_extension("c"));
        assert!(is_c_source_extension("i"));
        assert!(is_c_source_extension("cpp"));
        assert!(is_c_source_extension("C"));
        assert!(!is_c_source_extension("smt2"));
    }

    #[test]
    fn test_expected_status_svcomp_verdicts() {
        let task_true = SvCompTask {
            name: "test".into(),
            sources: vec![],
            expected_verdict: Some("true".into()),
            options: HashMap::new(),
            format_version: None,
        };
        assert_eq!(task_true.expected_status(), Some(ExpectedStatus::Unsat));

        let task_false = SvCompTask {
            name: "test".into(),
            sources: vec![],
            expected_verdict: Some("false".into()),
            options: HashMap::new(),
            format_version: None,
        };
        assert_eq!(task_false.expected_status(), Some(ExpectedStatus::Sat));

        let task_sat = SvCompTask {
            name: "test".into(),
            sources: vec![],
            expected_verdict: Some("sat".into()),
            options: HashMap::new(),
            format_version: None,
        };
        assert_eq!(task_sat.expected_status(), Some(ExpectedStatus::Sat));
    }

    #[test]
    fn test_resolve_expected_verdict_top_level() {
        let top = Some("true".to_string());
        let result = resolve_expected_verdict(&top, &None);
        assert_eq!(result, Some("true".to_string()));
    }

    #[test]
    fn test_resolve_expected_verdict_unanimous_properties() {
        let top = Some("true".to_string());
        let props = Some(vec![
            SvCompProperty {
                property_file: None,
                expected_verdict: Some("false".to_string()),
            },
            SvCompProperty {
                property_file: None,
                expected_verdict: Some("false".to_string()),
            },
        ]);
        let result = resolve_expected_verdict(&top, &props);
        assert_eq!(result, Some("false".to_string()));
    }

    #[test]
    fn test_resolve_expected_verdict_disagreeing_properties_falls_back() {
        let top = Some("true".to_string());
        let props = Some(vec![
            SvCompProperty {
                property_file: None,
                expected_verdict: Some("true".to_string()),
            },
            SvCompProperty {
                property_file: None,
                expected_verdict: Some("false".to_string()),
            },
        ]);
        let result = resolve_expected_verdict(&top, &props);
        assert_eq!(result, Some("true".to_string()));
    }

    #[test]
    fn test_discover_empty_directory() {
        let dir = make_temp_dir();
        let reader = SvCompReader::discover(dir.path()).expect("discover should succeed");
        assert_eq!(reader.tasks().len(), 0);
        assert_eq!(reader.skipped_count(), 0);
    }

    #[test]
    fn test_discover_nonexistent_directory() {
        let result = SvCompReader::discover(Path::new("/nonexistent/svcomp/path/12345"));
        // Should return Io error for nonexistent root
        assert!(result.is_err());
    }
}
