//! SMT-COMP 2026 submission preparation infrastructure.
//!
//! This module provides types and functions for generating StarExec-compatible
//! submission packages for the SMT Competition 2026.
//!
//! # Supported Divisions
//!
//! All quantifier-free and quantified logics from SMT-COMP 2026:
//! QF_LIA, QF_LRA, QF_BV, QF_S, QF_FP, QF_DT, QF_A, QF_NIA, QF_NRA,
//! UFLIA, UFLRA, AUFLIA, AUFLIRA, QF_ALIA, QF_AUFBV, QF_ABV, QF_NIRA,
//! QF_IDL, QF_RDL

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::benchmark::{BenchmarkStatus, SingleResult};
use crate::loader::ExpectedStatus;

/// SMT-COMP 2026 competition track.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Track {
    /// Single-query track: one `check-sat` per file.
    SingleQuery,
    /// Incremental track: `push`/`pop` commands present.
    Incremental,
    /// Unsat-core track: solver must output `(get-unsat-core)` after `unsat`.
    UnsatCore,
    /// Model-validation track: solver must output `(get-model)` after `sat`.
    ModelValidation,
    /// Proof-exhibition track: solver must output `(get-proof)` after `unsat`.
    ProofExhibition,
}

impl Track {
    /// Returns the StarExec suffix used in run-script filenames.
    pub fn as_starexec_suffix(&self) -> &'static str {
        match self {
            Track::SingleQuery => "default",
            Track::Incremental => "incremental",
            Track::UnsatCore => "unsat_core",
            Track::ModelValidation => "model_validation",
            Track::ProofExhibition => "proof_exhibition",
        }
    }

    /// Human-readable name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Track::SingleQuery => "Single Query",
            Track::Incremental => "Incremental",
            Track::UnsatCore => "Unsat Core",
            Track::ModelValidation => "Model Validation",
            Track::ProofExhibition => "Proof Exhibition",
        }
    }

    /// All five SMT-COMP 2026 tracks.
    pub fn all() -> &'static [Track] {
        &[
            Track::SingleQuery,
            Track::Incremental,
            Track::UnsatCore,
            Track::ModelValidation,
            Track::ProofExhibition,
        ]
    }
}

/// Errors that can occur during submission preparation.
#[derive(Error, Debug)]
pub enum SubmissionError {
    /// IO error during package generation.
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    /// A declared division has no passing benchmarks.
    #[error("Division '{0}' has no passing benchmarks; cannot include in submission")]
    DivisionNotValidated(String),

    /// An unknown division was declared.
    #[error("Unknown SMT-COMP 2026 division: '{0}'")]
    UnknownDivision(String),

    /// Required field is missing.
    #[error("Required field missing: {0}")]
    MissingField(String),

    /// Benchmark validation failed.
    #[error("Benchmark validation failed for division '{division}': {reason}")]
    BenchmarkValidationFailed {
        /// The logic/division name.
        division: String,
        /// The failure reason.
        reason: String,
    },
}

/// Result type for submission operations.
pub type SubmissionResult<T> = Result<T, SubmissionError>;

/// All SMT-COMP 2026 divisions supported by OxiZ.
pub const SMT_COMP_2026_DIVISIONS: &[&str] = &[
    "QF_LIA",   // Quantifier-free linear integer arithmetic
    "QF_LRA",   // Quantifier-free linear real arithmetic
    "QF_BV",    // Quantifier-free bitvectors
    "QF_S",     // Quantifier-free strings
    "QF_FP",    // Quantifier-free floating point
    "QF_DT",    // Quantifier-free datatypes
    "QF_A",     // Quantifier-free arrays
    "QF_NIA",   // Quantifier-free nonlinear integer arithmetic
    "QF_NRA",   // Quantifier-free nonlinear real arithmetic
    "UFLIA",    // Uninterpreted functions + linear integer arithmetic
    "UFLRA",    // Uninterpreted functions + linear real arithmetic
    "AUFLIA",   // Arrays + uninterpreted functions + linear integer arithmetic
    "AUFLIRA",  // Arrays + uninterpreted functions + linear integer/real arithmetic
    "QF_ALIA",  // Quantifier-free arrays + linear integer arithmetic
    "QF_AUFBV", // Quantifier-free arrays + uninterpreted functions + bitvectors
    "QF_ABV",   // Quantifier-free arrays + bitvectors
    "QF_NIRA",  // Quantifier-free nonlinear integer/real arithmetic
    "QF_IDL",   // Quantifier-free integer difference logic
    "QF_RDL",   // Quantifier-free real difference logic
];

/// Contact information for the SMT-COMP submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactInfo {
    /// Primary contact name.
    pub name: String,
    /// Contact email address.
    pub email: String,
    /// Affiliation.
    pub affiliation: String,
    /// Optional URL for solver website.
    pub url: Option<String>,
}

impl ContactInfo {
    /// Create a new contact info entry.
    pub fn new(
        name: impl Into<String>,
        email: impl Into<String>,
        affiliation: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            email: email.into(),
            affiliation: affiliation.into(),
            url: None,
        }
    }

    /// Set the solver website URL.
    #[must_use]
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }
}

/// SMT-COMP 2026 submission configuration.
///
/// Encapsulates all metadata required for a conforming StarExec submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionConfig {
    /// Solver name (shown in competition results).
    pub solver_name: String,

    /// Solver version string.
    pub version: String,

    /// Description of the solver.
    pub description: Option<String>,

    /// Divisions (logics) the solver participates in.
    pub divisions: Vec<String>,

    /// Competition tracks to participate in.
    ///
    /// Defaults to all five SMT-COMP 2026 tracks when created via
    /// [`SubmissionConfig::default_oxiz_2026`].
    pub tracks: Vec<Track>,

    /// Primary contact information.
    pub contact: ContactInfo,

    /// StarExec solver configuration name.
    pub starexec_config_name: String,

    /// Memory limit in megabytes for the submission.
    pub memory_limit_mb: u64,

    /// CPU timeout in seconds.
    pub cpu_timeout_secs: u64,

    /// Path to the solver binary (relative to package root).
    pub solver_binary: String,

    /// Additional environment variables for StarExec.
    pub env_vars: HashMap<String, String>,
}

impl SubmissionConfig {
    /// Create a new submission configuration with required fields.
    ///
    /// The `tracks` field defaults to all five SMT-COMP 2026 tracks.
    /// The `solver_binary` defaults to `"bin/smtcomp2026"`.
    pub fn new(
        solver_name: impl Into<String>,
        version: impl Into<String>,
        contact: ContactInfo,
    ) -> Self {
        Self {
            solver_name: solver_name.into(),
            version: version.into(),
            description: None,
            divisions: Vec::new(),
            tracks: Track::all().to_vec(),
            contact,
            starexec_config_name: "default".to_string(),
            memory_limit_mb: 8192,
            cpu_timeout_secs: 1200,
            solver_binary: "bin/smtcomp2026".to_string(),
            env_vars: HashMap::new(),
        }
    }

    /// Set the solver description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a division to the submission.
    #[must_use]
    pub fn with_division(mut self, division: impl Into<String>) -> Self {
        self.divisions.push(division.into());
        self
    }

    /// Set all divisions at once.
    #[must_use]
    pub fn with_divisions(mut self, divisions: Vec<String>) -> Self {
        self.divisions = divisions;
        self
    }

    /// Set the StarExec configuration name.
    #[must_use]
    pub fn with_starexec_config(mut self, name: impl Into<String>) -> Self {
        self.starexec_config_name = name.into();
        self
    }

    /// Set the memory limit.
    #[must_use]
    pub fn with_memory_limit_mb(mut self, mb: u64) -> Self {
        self.memory_limit_mb = mb;
        self
    }

    /// Set the CPU timeout.
    #[must_use]
    pub fn with_cpu_timeout_secs(mut self, secs: u64) -> Self {
        self.cpu_timeout_secs = secs;
        self
    }

    /// Set the solver binary path.
    #[must_use]
    pub fn with_solver_binary(mut self, path: impl Into<String>) -> Self {
        self.solver_binary = path.into();
        self
    }

    /// Set the competition tracks for this submission.
    #[must_use]
    pub fn with_tracks(mut self, tracks: Vec<Track>) -> Self {
        self.tracks = tracks;
        self
    }

    /// Add an environment variable.
    #[must_use]
    pub fn with_env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Create a default OxiZ submission config for all 2026 divisions and tracks.
    pub fn default_oxiz_2026() -> Self {
        let contact = ContactInfo::new(
            "COOLJAPAN OU (Team Kitasan)",
            "contact@cooljapan.io",
            "COOLJAPAN OU",
        )
        .with_url("https://github.com/cool-japan/oxiz");

        Self::new("OxiZ", env!("CARGO_PKG_VERSION"), contact)
            .with_description(
                "OxiZ: A next-generation SMT solver written in pure, safe Rust. \
                 Implements the complete SMT-LIB2 standard with support for all \
                 major arithmetic, bitvector, array, and string theories.",
            )
            .with_divisions(
                SMT_COMP_2026_DIVISIONS
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            )
            .with_tracks(Track::all().to_vec())
            .with_starexec_config("oxiz-smtcomp2026")
            .with_memory_limit_mb(8192)
            .with_cpu_timeout_secs(1200)
            .with_solver_binary("bin/smtcomp2026")
    }
}

/// Outcome of benchmark validation for a single division.
#[derive(Debug, Clone)]
pub struct DivisionValidationResult {
    /// The division name.
    pub division: String,
    /// Whether validation passed.
    pub passed: bool,
    /// Number of passing benchmarks in this division.
    pub passing_count: usize,
    /// Number of failing benchmarks.
    pub failing_count: usize,
    /// Number of unknown results.
    pub unknown_count: usize,
    /// Failure reason, if any.
    pub failure_reason: Option<String>,
}

impl DivisionValidationResult {
    /// Create a passing validation result.
    fn passing(division: String, passing: usize, failing: usize, unknown: usize) -> Self {
        Self {
            division,
            passed: true,
            passing_count: passing,
            failing_count: failing,
            unknown_count: unknown,
            failure_reason: None,
        }
    }

    /// Create a failing validation result.
    fn failing(division: String, reason: String) -> Self {
        Self {
            division,
            passed: false,
            passing_count: 0,
            failing_count: 0,
            unknown_count: 0,
            failure_reason: Some(reason),
        }
    }
}

impl fmt::Display for DivisionValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed {
            write!(
                f,
                "{}: OK ({} pass, {} fail, {} unknown)",
                self.division, self.passing_count, self.failing_count, self.unknown_count
            )
        } else {
            write!(
                f,
                "{}: FAIL — {}",
                self.division,
                self.failure_reason.as_deref().unwrap_or("unknown reason")
            )
        }
    }
}

/// Validates that all declared divisions have passing benchmarks.
///
/// For each division in `config.divisions`, this function inspects `results`
/// to ensure at least one benchmark in that logic returned a correct answer
/// (`sat` when expected `sat`, `unsat` when expected `unsat`).
///
/// # Errors
///
/// Returns `SubmissionError::UnknownDivision` if a division is not in the
/// official 2026 division list, or `SubmissionError::DivisionNotValidated`
/// if a division has zero passing benchmarks.
pub fn validate_divisions(
    config: &SubmissionConfig,
    results: &[SingleResult],
) -> SubmissionResult<Vec<DivisionValidationResult>> {
    let valid_divisions: HashSet<&str> =
        SMT_COMP_2026_DIVISIONS.iter().copied().collect();

    let mut validation_results = Vec::with_capacity(config.divisions.len());

    for division in &config.divisions {
        // Check that division is known
        if !valid_divisions.contains(division.as_str()) {
            return Err(SubmissionError::UnknownDivision(division.clone()));
        }

        // Collect results for this division
        let division_results: Vec<&SingleResult> = results
            .iter()
            .filter(|r| {
                r.logic
                    .as_deref()
                    .map(|l| l == division.as_str())
                    .unwrap_or(false)
            })
            .collect();

        if division_results.is_empty() {
            validation_results.push(DivisionValidationResult::failing(
                division.clone(),
                format!("no benchmark results found for division '{}'", division),
            ));
            continue;
        }

        let mut passing = 0usize;
        let mut failing = 0usize;
        let mut unknown = 0usize;

        for result in &division_results {
            match (&result.expected, &result.status) {
                (Some(ExpectedStatus::Sat), BenchmarkStatus::Sat) => passing += 1,
                (Some(ExpectedStatus::Unsat), BenchmarkStatus::Unsat) => passing += 1,
                (None, BenchmarkStatus::Sat | BenchmarkStatus::Unsat) => passing += 1,
                (Some(ExpectedStatus::Unknown), _) => unknown += 1,
                (_, BenchmarkStatus::Unknown | BenchmarkStatus::Timeout) => unknown += 1,
                _ => failing += 1,
            }
        }

        if passing == 0 {
            validation_results.push(DivisionValidationResult::failing(
                division.clone(),
                format!(
                    "{} benchmarks run, {} correct, {} wrong, {} unknown — need at least 1 correct",
                    division_results.len(),
                    passing,
                    failing,
                    unknown
                ),
            ));
        } else {
            validation_results.push(DivisionValidationResult::passing(
                division.clone(),
                passing,
                failing,
                unknown,
            ));
        }
    }

    // Check all divisions passed
    let failed: Vec<&DivisionValidationResult> =
        validation_results.iter().filter(|r| !r.passed).collect();

    if let Some(first_failed) = failed.first() {
        return Err(SubmissionError::DivisionNotValidated(
            first_failed.division.clone(),
        ));
    }

    Ok(validation_results)
}

/// Description of the generated StarExec-compatible submission package.
///
/// The actual package layout is written to disk; this struct describes what
/// was produced and where.
#[derive(Debug, Clone)]
pub struct SubmissionPackage {
    /// Root directory of the submission package.
    pub root_dir: PathBuf,
    /// Path to the generated `starexec_run_default` script (single-query track).
    pub run_script: PathBuf,
    /// Per-track run scripts: `(Track, path)` pairs for each configured track.
    pub track_scripts: Vec<(Track, PathBuf)>,
    /// Path to the generated solver description file.
    pub description_file: PathBuf,
    /// Path to the StarExec configuration XML.
    pub config_xml: PathBuf,
    /// List of divisions included.
    pub divisions: Vec<String>,
}

impl SubmissionPackage {
    /// Returns a summary of the package contents.
    pub fn summary(&self) -> String {
        let track_lines: String = self
            .track_scripts
            .iter()
            .map(|(track, path)| {
                format!("  {:20} {}", track.display_name(), path.display())
            })
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "StarExec submission package at {}\n\
             Run script:   {}\n\
             Per-track scripts:\n{}\n\
             Description:  {}\n\
             Config XML:   {}\n\
             Divisions:    {}",
            self.root_dir.display(),
            self.run_script.display(),
            track_lines,
            self.description_file.display(),
            self.config_xml.display(),
            self.divisions.join(", ")
        )
    }
}

/// Generates a StarExec-compatible submission package on disk.
///
/// Creates the following layout in `output_dir`:
/// ```text
/// output_dir/
///   bin/
///     starexec_run_default          (single-query / backward-compat script)
///     starexec_run_incremental      (incremental track)
///     starexec_run_unsat_core       (unsat-core track)
///     starexec_run_model_validation (model-validation track)
///     starexec_run_proof_exhibition (proof-exhibition track)
///   description.txt
///   starexec_conf.xml
/// ```
///
/// # Errors
///
/// Returns `SubmissionError::Io` for any filesystem failure, or
/// `SubmissionError::MissingField` if required config fields are empty.
pub fn generate_submission_package(
    config: &SubmissionConfig,
    output_dir: &Path,
) -> SubmissionResult<SubmissionPackage> {
    // Validate required fields
    if config.solver_name.is_empty() {
        return Err(SubmissionError::MissingField("solver_name".to_string()));
    }
    if config.version.is_empty() {
        return Err(SubmissionError::MissingField("version".to_string()));
    }
    if config.contact.email.is_empty() {
        return Err(SubmissionError::MissingField("contact.email".to_string()));
    }
    if config.divisions.is_empty() {
        return Err(SubmissionError::MissingField("divisions".to_string()));
    }

    // Create directory structure
    let bin_dir = output_dir.join("bin");
    fs::create_dir_all(&bin_dir)?;

    // Generate the default StarExec run script (backward compatibility — no --track flag).
    let run_script_path = bin_dir.join("starexec_run_default");
    write_run_script(config, &run_script_path)?;

    // Generate per-track run scripts for every configured track.
    // The SingleQuery track reuses `starexec_run_default` (already written above);
    // all other tracks get their own script with `--track <suffix>`.
    let mut track_scripts: Vec<(Track, PathBuf)> = Vec::new();
    for track in &config.tracks {
        let suffix = track.as_starexec_suffix();
        let script_path = bin_dir.join(format!("starexec_run_{}", suffix));
        write_track_run_script(config, track, &script_path)?;
        track_scripts.push((track.clone(), script_path));
    }

    // Generate solver description
    let description_path = output_dir.join("description.txt");
    write_description(config, &description_path)?;

    // Generate StarExec configuration XML
    let config_xml_path = output_dir.join("starexec_conf.xml");
    write_starexec_xml(config, &config_xml_path)?;

    Ok(SubmissionPackage {
        root_dir: output_dir.to_path_buf(),
        run_script: run_script_path,
        track_scripts,
        description_file: description_path,
        config_xml: config_xml_path,
        divisions: config.divisions.clone(),
    })
}

/// Write the StarExec run script (backward-compatible default; no `--track` flag).
fn write_run_script(config: &SubmissionConfig, path: &Path) -> SubmissionResult<()> {
    write_track_run_script(config, &Track::SingleQuery, path)
}

/// Write a track-specific StarExec run script.
///
/// For `Track::SingleQuery` the generated script invokes the binary without a
/// `--track` flag (for backward compatibility with StarExec harnesses that only
/// call `starexec_run_default`).  For all other tracks the `--track <suffix>`
/// flag is appended before the benchmark argument so the binary can perform the
/// appropriate post-processing (model, unsat-core, proof retrieval).
fn write_track_run_script(
    config: &SubmissionConfig,
    track: &Track,
    path: &Path,
) -> SubmissionResult<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    let suffix = track.as_starexec_suffix();

    writeln!(w, "#!/bin/bash")?;
    writeln!(
        w,
        "# Auto-generated StarExec run script for {} — {} track",
        config.solver_name,
        track.display_name()
    )?;
    writeln!(w, "# Version: {}", config.version)?;
    writeln!(w, "# SMT-COMP 2026")?;
    writeln!(w, "#")?;
    writeln!(w, "# Usage: this script is called by StarExec with the benchmark as $1")?;
    writeln!(w)?;

    // Environment variable exports
    for (key, value) in &config.env_vars {
        writeln!(w, "export {}={}", key, value)?;
    }

    writeln!(w)?;
    writeln!(w, "SOLVER_BIN=\"$(dirname \"$0\")/../{}\"", config.solver_binary)?;
    writeln!(w)?;
    writeln!(w, "if [ ! -x \"$SOLVER_BIN\" ]; then")?;
    writeln!(w, "    echo \"error: solver binary not found at $SOLVER_BIN\"")?;
    writeln!(w, "    exit 1")?;
    writeln!(w, "fi")?;
    writeln!(w)?;

    // SingleQuery uses the plain interface (no --track flag) for backward compat.
    // All other tracks pass --track <suffix> so the binary knows which
    // post-processing to perform.
    match track {
        Track::SingleQuery => {
            writeln!(w, "exec \"$SOLVER_BIN\" --smtcomp \"$@\"")?;
        }
        _ => {
            writeln!(
                w,
                "exec \"$SOLVER_BIN\" --smtcomp --track {} \"$@\"",
                suffix
            )?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Write the solver description file.
fn write_description(config: &SubmissionConfig, path: &Path) -> SubmissionResult<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "Solver: {}", config.solver_name)?;
    writeln!(w, "Version: {}", config.version)?;
    writeln!(w)?;

    if let Some(desc) = &config.description {
        writeln!(w, "{}", desc)?;
        writeln!(w)?;
    }

    writeln!(w, "Contact:")?;
    writeln!(w, "  Name:        {}", config.contact.name)?;
    writeln!(w, "  Email:       {}", config.contact.email)?;
    writeln!(w, "  Affiliation: {}", config.contact.affiliation)?;
    if let Some(url) = &config.contact.url {
        writeln!(w, "  URL:         {}", url)?;
    }
    writeln!(w)?;

    writeln!(w, "SMT-COMP 2026 Divisions:")?;
    for division in &config.divisions {
        writeln!(w, "  - {}", division)?;
    }
    writeln!(w)?;

    writeln!(
        w,
        "Supported Logics: {}",
        config.divisions.join(", ")
    )?;

    w.flush()?;
    Ok(())
}

/// Write the StarExec XML configuration.
fn write_starexec_xml(config: &SubmissionConfig, path: &Path) -> SubmissionResult<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")?;
    writeln!(w, "<solverConfig>")?;
    writeln!(
        w,
        "  <name>{}</name>",
        xml_escape(&config.starexec_config_name)
    )?;
    writeln!(
        w,
        "  <solverName>{}</solverName>",
        xml_escape(&config.solver_name)
    )?;
    writeln!(
        w,
        "  <version>{}</version>",
        xml_escape(&config.version)
    )?;
    writeln!(
        w,
        "  <memoryLimit>{}</memoryLimit>",
        config.memory_limit_mb
    )?;
    writeln!(
        w,
        "  <cpuTimeout>{}</cpuTimeout>",
        config.cpu_timeout_secs
    )?;
    writeln!(w, "  <divisions>")?;
    for division in &config.divisions {
        writeln!(w, "    <division>{}</division>", xml_escape(division))?;
    }
    writeln!(w, "  </divisions>")?;
    writeln!(w, "  <contact>")?;
    writeln!(
        w,
        "    <name>{}</name>",
        xml_escape(&config.contact.name)
    )?;
    writeln!(
        w,
        "    <email>{}</email>",
        xml_escape(&config.contact.email)
    )?;
    writeln!(
        w,
        "    <affiliation>{}</affiliation>",
        xml_escape(&config.contact.affiliation)
    )?;
    if let Some(url) = &config.contact.url {
        writeln!(w, "    <url>{}</url>", xml_escape(url))?;
    }
    writeln!(w, "  </contact>")?;
    writeln!(w, "</solverConfig>")?;

    w.flush()?;
    Ok(())
}

/// Minimal XML character escaping.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;
    use crate::benchmark::{BenchmarkStatus, SingleResult};
    use crate::loader::{BenchmarkMeta, ExpectedStatus};

    fn make_meta(logic: &str, expected: ExpectedStatus) -> BenchmarkMeta {
        BenchmarkMeta {
            path: PathBuf::from(format!("/tmp/{}.smt2", logic)),
            logic: Some(logic.to_string()),
            expected_status: Some(expected),
            file_size: 100,
            category: None,
            structural_features: None,
        }
    }

    fn make_result(meta: &BenchmarkMeta, status: BenchmarkStatus) -> SingleResult {
        SingleResult::new(meta, status, Duration::from_millis(50))
    }

    #[test]
    fn test_submission_config_builder() {
        let contact = ContactInfo::new("Test User", "test@example.com", "Test Org");
        let config = SubmissionConfig::new("TestSolver", "1.0.0", contact)
            .with_division("QF_LIA")
            .with_division("QF_BV");

        assert_eq!(config.solver_name, "TestSolver");
        assert_eq!(config.version, "1.0.0");
        assert_eq!(config.divisions.len(), 2);
        assert!(config.divisions.contains(&"QF_LIA".to_string()));
    }

    #[test]
    fn test_validate_divisions_passes() {
        let contact = ContactInfo::new("Test", "t@t.com", "Org");
        let config = SubmissionConfig::new("Solver", "1.0", contact)
            .with_division("QF_LIA");

        let meta = make_meta("QF_LIA", ExpectedStatus::Sat);
        let results = vec![make_result(&meta, BenchmarkStatus::Sat)];

        let validation = validate_divisions(&config, &results)
            .expect("validation should pass");
        assert_eq!(validation.len(), 1);
        assert!(validation[0].passed);
        assert_eq!(validation[0].passing_count, 1);
    }

    #[test]
    fn test_validate_divisions_fails_no_results() {
        let contact = ContactInfo::new("Test", "t@t.com", "Org");
        let config = SubmissionConfig::new("Solver", "1.0", contact)
            .with_division("QF_LIA");

        let err = validate_divisions(&config, &[])
            .expect_err("should fail with no results");
        assert!(matches!(err, SubmissionError::DivisionNotValidated(_)));
    }

    #[test]
    fn test_validate_divisions_rejects_unknown() {
        let contact = ContactInfo::new("Test", "t@t.com", "Org");
        let config = SubmissionConfig::new("Solver", "1.0", contact)
            .with_division("QF_UNKNOWN_LOGIC");

        let err = validate_divisions(&config, &[])
            .expect_err("should fail with unknown division");
        assert!(matches!(err, SubmissionError::UnknownDivision(_)));
    }

    #[test]
    fn test_generate_submission_package() {
        let output_dir = std::env::temp_dir().join("oxiz_smtcomp_test_pkg");
        let _ = fs::remove_dir_all(&output_dir);

        let contact = ContactInfo::new("Test User", "test@example.com", "Test Org")
            .with_url("https://example.com");
        let config = SubmissionConfig::new("OxiZ", "0.2.0", contact)
            .with_division("QF_LIA")
            .with_division("QF_BV");

        let pkg = generate_submission_package(&config, &output_dir)
            .expect("package generation should succeed");

        assert!(pkg.run_script.exists());
        assert!(pkg.description_file.exists());
        assert!(pkg.config_xml.exists());
        assert_eq!(pkg.divisions.len(), 2);

        let _ = fs::remove_dir_all(&output_dir);
    }

    #[test]
    fn test_generate_package_missing_field() {
        let output_dir = std::env::temp_dir().join("oxiz_smtcomp_test_empty");
        let contact = ContactInfo::new("Test", "t@t.com", "Org");
        let config = SubmissionConfig::new("", "1.0", contact);
        // No divisions either, but solver_name checked first
        let err = generate_submission_package(&config, &output_dir)
            .expect_err("should fail with empty solver_name");
        assert!(matches!(err, SubmissionError::MissingField(_)));
    }

    #[test]
    fn test_all_2026_divisions_are_valid() {
        let valid: HashSet<&str> = SMT_COMP_2026_DIVISIONS.iter().copied().collect();
        assert_eq!(valid.len(), SMT_COMP_2026_DIVISIONS.len(), "no duplicate divisions");
        assert!(valid.contains("QF_LIA"));
        assert!(valid.contains("AUFLIRA"));
        assert!(valid.contains("QF_NIRA"));
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a&b<c>d\"e'f"), "a&amp;b&lt;c&gt;d&quot;e&apos;f");
    }

    // --- Track infrastructure tests ---

    #[test]
    fn test_default_config_uses_smtcomp_binary() {
        let cfg = SubmissionConfig::default_oxiz_2026();
        assert_eq!(cfg.solver_binary, "bin/smtcomp2026");
    }

    #[test]
    fn test_default_config_version_matches_cargo() {
        let cfg = SubmissionConfig::default_oxiz_2026();
        assert_eq!(cfg.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_track_enum_round_trip() {
        for track in Track::all() {
            let suffix = track.as_starexec_suffix();
            assert!(!suffix.is_empty());
            // Every suffix must be unique across all tracks.
            let count = Track::all()
                .iter()
                .filter(|t| t.as_starexec_suffix() == suffix)
                .count();
            assert_eq!(count, 1, "suffix '{suffix}' is not unique");
        }
    }

    #[test]
    fn test_generate_emits_per_track_scripts() {
        let dir = std::env::temp_dir()
            .join(format!("oxiz_track_scripts_{}", std::process::id()));
        let cfg = SubmissionConfig::default_oxiz_2026();
        let _pkg = generate_submission_package(&cfg, &dir).expect("package generation failed");
        for track in Track::all() {
            let script = dir
                .join("bin")
                .join(format!("starexec_run_{}", track.as_starexec_suffix()));
            assert!(
                script.exists(),
                "missing run script for track {:?}: {:?}",
                track,
                script
            );
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
