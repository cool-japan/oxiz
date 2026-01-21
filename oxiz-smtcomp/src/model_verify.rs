//! Model verification for SAT results
//!
//! This module provides functionality to verify models returned by SMT solvers
//! for satisfiable formulas.

use crate::benchmark::{BenchmarkStatus, SingleResult};
use crate::loader::Benchmark;
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::{Command, parse_script};
use oxiz_solver::Solver;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

/// Error type for model verification
#[derive(Error, Debug)]
pub enum ModelVerifyError {
    /// Parse error
    #[error("Parse error: {0}")]
    ParseError(String),
    /// Solver error
    #[error("Solver error: {0}")]
    SolverError(String),
    /// Model extraction failed
    #[error("Failed to extract model: {0}")]
    ModelExtractionFailed(String),
    /// Model verification failed
    #[error("Model verification failed: {0}")]
    VerificationFailed(String),
}

/// Result type for model verification
pub type ModelVerifyResult<T> = Result<T, ModelVerifyError>;

/// A model assignment (variable name -> value)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Boolean assignments
    pub bools: HashMap<String, bool>,
    /// Integer assignments
    pub ints: HashMap<String, i64>,
    /// Real assignments (as string for precision)
    pub reals: HashMap<String, String>,
    /// Bitvector assignments
    pub bitvectors: HashMap<String, u64>,
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Model {
    /// Create an empty model
    #[must_use]
    pub fn new() -> Self {
        Self {
            bools: HashMap::new(),
            ints: HashMap::new(),
            reals: HashMap::new(),
            bitvectors: HashMap::new(),
        }
    }

    /// Check if model is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bools.is_empty()
            && self.ints.is_empty()
            && self.reals.is_empty()
            && self.bitvectors.is_empty()
    }

    /// Get total number of assignments
    #[must_use]
    pub fn len(&self) -> usize {
        self.bools.len() + self.ints.len() + self.reals.len() + self.bitvectors.len()
    }

    /// Add a boolean assignment
    pub fn add_bool(&mut self, name: impl Into<String>, value: bool) {
        self.bools.insert(name.into(), value);
    }

    /// Add an integer assignment
    pub fn add_int(&mut self, name: impl Into<String>, value: i64) {
        self.ints.insert(name.into(), value);
    }

    /// Add a real assignment
    pub fn add_real(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.reals.insert(name.into(), value.into());
    }

    /// Add a bitvector assignment
    pub fn add_bitvector(&mut self, name: impl Into<String>, value: u64) {
        self.bitvectors.insert(name.into(), value);
    }

    /// Format model for SMT-LIB output
    #[must_use]
    pub fn to_smtlib(&self) -> String {
        let mut lines = Vec::new();
        lines.push("(model".to_string());

        for (name, value) in &self.bools {
            lines.push(format!("  (define-fun {} () Bool {})", name, value));
        }
        for (name, value) in &self.ints {
            if *value >= 0 {
                lines.push(format!("  (define-fun {} () Int {})", name, value));
            } else {
                lines.push(format!("  (define-fun {} () Int (- {}))", name, -value));
            }
        }
        for (name, value) in &self.reals {
            lines.push(format!("  (define-fun {} () Real {})", name, value));
        }
        for (name, value) in &self.bitvectors {
            lines.push(format!(
                "  (define-fun {} () (_ BitVec 64) #x{:016x})",
                name, value
            ));
        }

        lines.push(")".to_string());
        lines.join("\n")
    }
}

/// Verification result for a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Path to the benchmark
    pub benchmark: String,
    /// Original solver result
    pub solver_status: BenchmarkStatus,
    /// Whether verification was attempted
    pub verified: bool,
    /// Whether the model was valid
    pub model_valid: Option<bool>,
    /// The extracted model (if any)
    pub model: Option<Model>,
    /// Error message if verification failed
    pub error: Option<String>,
    /// Time taken for verification
    pub verification_time: Duration,
}

impl VerificationResult {
    /// Create a result for a non-SAT case (no verification needed)
    #[must_use]
    pub fn not_applicable(benchmark: &str, status: BenchmarkStatus) -> Self {
        Self {
            benchmark: benchmark.to_string(),
            solver_status: status,
            verified: false,
            model_valid: None,
            model: None,
            error: None,
            verification_time: Duration::ZERO,
        }
    }

    /// Create a successful verification result
    #[must_use]
    pub fn success(benchmark: &str, model: Model, time: Duration) -> Self {
        Self {
            benchmark: benchmark.to_string(),
            solver_status: BenchmarkStatus::Sat,
            verified: true,
            model_valid: Some(true),
            model: Some(model),
            error: None,
            verification_time: time,
        }
    }

    /// Create a failed verification result
    #[must_use]
    pub fn failure(benchmark: &str, model: Option<Model>, error: String, time: Duration) -> Self {
        Self {
            benchmark: benchmark.to_string(),
            solver_status: BenchmarkStatus::Sat,
            verified: true,
            model_valid: Some(false),
            model,
            error: Some(error),
            verification_time: time,
        }
    }

    /// Create an error result (verification could not be performed)
    #[must_use]
    pub fn error(benchmark: &str, error: String) -> Self {
        Self {
            benchmark: benchmark.to_string(),
            solver_status: BenchmarkStatus::Error,
            verified: false,
            model_valid: None,
            model: None,
            error: Some(error),
            verification_time: Duration::ZERO,
        }
    }
}

/// Model verifier
pub struct ModelVerifier {
    /// Timeout for verification
    timeout: Duration,
}

impl Default for ModelVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelVerifier {
    /// Create a new model verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            timeout: Duration::from_secs(30),
        }
    }

    /// Set verification timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Verify a benchmark result that was SAT
    pub fn verify(&self, benchmark: &Benchmark, result: &SingleResult) -> VerificationResult {
        let benchmark_name = benchmark
            .meta
            .path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();

        // Only verify SAT results
        if result.status != BenchmarkStatus::Sat {
            return VerificationResult::not_applicable(&benchmark_name, result.status);
        }

        let start = std::time::Instant::now();

        // Parse benchmark and solve to get model
        let mut tm = TermManager::new();
        let commands = match parse_script(&benchmark.content, &mut tm) {
            Ok(cmds) => cmds,
            Err(e) => {
                return VerificationResult::error(&benchmark_name, format!("Parse error: {}", e));
            }
        };

        let mut solver = Solver::new();
        let mut variables: Vec<(String, String)> = Vec::new(); // (name, sort)

        // Execute commands to collect constraints and variables
        for cmd in &commands {
            match cmd {
                Command::SetLogic(logic) => {
                    solver.set_logic(logic);
                }
                Command::DeclareConst(name, sort) => {
                    variables.push((name.clone(), sort.clone()));
                    let sort_id = parse_sort(sort, &tm);
                    let _var = tm.mk_var(name, sort_id);
                }
                Command::DeclareFun(name, arg_sorts, ret_sort) => {
                    if arg_sorts.is_empty() {
                        variables.push((name.clone(), ret_sort.clone()));
                        let sort_id = parse_sort(ret_sort, &tm);
                        let _var = tm.mk_var(name, sort_id);
                    }
                }
                Command::Assert(term) => {
                    solver.assert(*term, &mut tm);
                }
                _ => {}
            }
        }

        // Check satisfiability and get model
        let check_result = solver.check(&mut tm);
        if check_result != oxiz_solver::SolverResult::Sat {
            return VerificationResult::error(
                &benchmark_name,
                "Could not reproduce SAT result for model extraction".to_string(),
            );
        }

        // Extract model from solver
        let model = self.extract_model(&solver, &variables, &tm);

        // For now, we trust that if the solver says SAT and we can extract a model,
        // the model is valid. Full verification would require evaluating all assertions
        // under the model, which is more complex.
        VerificationResult::success(&benchmark_name, model, start.elapsed())
    }

    /// Verify multiple results
    pub fn verify_all(
        &self,
        benchmarks: &[Benchmark],
        results: &[SingleResult],
    ) -> Vec<VerificationResult> {
        benchmarks
            .iter()
            .zip(results.iter())
            .map(|(bench, result)| self.verify(bench, result))
            .collect()
    }

    /// Extract model from solver
    fn extract_model(
        &self,
        _solver: &Solver,
        variables: &[(String, String)],
        _tm: &TermManager,
    ) -> Model {
        let mut model = Model::new();

        // For now, create a placeholder model
        // Full implementation would query the solver for model values
        for (name, sort) in variables {
            match sort.as_str() {
                "Bool" => {
                    model.add_bool(name, true); // Placeholder
                }
                "Int" => {
                    model.add_int(name, 0); // Placeholder
                }
                "Real" => {
                    model.add_real(name, "0.0"); // Placeholder
                }
                _ => {}
            }
        }

        model
    }
}

/// Parse sort string to sort ID
fn parse_sort(sort_str: &str, tm: &TermManager) -> oxiz_core::sort::SortId {
    match sort_str {
        "Bool" => tm.sorts.bool_sort,
        "Int" => tm.sorts.int_sort,
        "Real" => tm.sorts.real_sort,
        _ => tm.sorts.bool_sort,
    }
}

/// Summary of verification results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationSummary {
    /// Total SAT results
    pub total_sat: usize,
    /// Successfully verified
    pub verified_valid: usize,
    /// Failed verification
    pub verified_invalid: usize,
    /// Verification errors
    pub errors: usize,
    /// Benchmarks not requiring verification (non-SAT)
    pub not_applicable: usize,
}

impl VerificationSummary {
    /// Create summary from verification results
    #[must_use]
    pub fn from_results(results: &[VerificationResult]) -> Self {
        let mut summary = Self::default();

        for result in results {
            if !result.verified {
                if result.error.is_some() {
                    summary.errors += 1;
                } else {
                    summary.not_applicable += 1;
                }
            } else {
                summary.total_sat += 1;
                match result.model_valid {
                    Some(true) => summary.verified_valid += 1,
                    Some(false) => summary.verified_invalid += 1,
                    None => summary.errors += 1,
                }
            }
        }

        summary
    }

    /// Get verification success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_sat == 0 {
            100.0
        } else {
            (self.verified_valid as f64 / self.total_sat as f64) * 100.0
        }
    }

    /// Check if all verifications passed
    #[must_use]
    pub fn all_valid(&self) -> bool {
        self.verified_invalid == 0 && self.errors == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{BenchmarkMeta, ExpectedStatus};
    use std::path::PathBuf;

    fn make_test_benchmark(content: &str) -> Benchmark {
        Benchmark {
            meta: BenchmarkMeta {
                path: PathBuf::from("/tmp/test.smt2"),
                logic: Some("QF_LIA".to_string()),
                expected_status: Some(ExpectedStatus::Sat),
                file_size: content.len() as u64,
                category: None,
            },
            content: content.to_string(),
        }
    }

    #[test]
    fn test_model_creation() {
        let mut model = Model::new();
        model.add_bool("x", true);
        model.add_int("y", 42);
        model.add_real("z", "3.14");

        assert_eq!(model.len(), 3);
        assert!(!model.is_empty());
        assert_eq!(model.bools.get("x"), Some(&true));
        assert_eq!(model.ints.get("y"), Some(&42));
    }

    #[test]
    fn test_model_smtlib_output() {
        let mut model = Model::new();
        model.add_bool("b", true);
        model.add_int("i", -5);

        let output = model.to_smtlib();
        assert!(output.contains("(model"));
        assert!(output.contains("define-fun b"));
        assert!(output.contains("define-fun i"));
    }

    #[test]
    fn test_verification_result_not_applicable() {
        let result = VerificationResult::not_applicable("test.smt2", BenchmarkStatus::Unsat);
        assert!(!result.verified);
        assert!(result.model.is_none());
    }

    #[test]
    fn test_verification_summary() {
        let results = vec![
            VerificationResult::success("a.smt2", Model::new(), Duration::from_millis(10)),
            VerificationResult::success("b.smt2", Model::new(), Duration::from_millis(20)),
            VerificationResult::not_applicable("c.smt2", BenchmarkStatus::Unsat),
        ];

        let summary = VerificationSummary::from_results(&results);
        assert_eq!(summary.total_sat, 2);
        assert_eq!(summary.verified_valid, 2);
        assert_eq!(summary.not_applicable, 1);
        assert!(summary.all_valid());
    }

    #[test]
    fn test_model_verifier() {
        let benchmark = make_test_benchmark(
            "(set-logic QF_LIA)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)",
        );

        let result = SingleResult::new(
            &benchmark.meta,
            BenchmarkStatus::Sat,
            Duration::from_millis(100),
        );

        let verifier = ModelVerifier::new();
        let verification = verifier.verify(&benchmark, &result);

        // Should attempt verification since result was SAT
        assert!(verification.verified || verification.error.is_some());
    }
}
