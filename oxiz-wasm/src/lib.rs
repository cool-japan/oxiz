//! OxiZ WASM - WebAssembly bindings for OxiZ SMT Solver
//!
//! Provides JavaScript/TypeScript bindings for running OxiZ in the browser.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod async_utils;
mod pool;
mod string_utils;

pub mod feature_registry;

// Phase 4.2: WebAssembly Optimization modules
pub mod feature_gates;
pub mod js_api;
pub mod lazy_loader;
pub mod module_registry;
pub mod optimize;

use oxiz_solver::Context;
use std::fmt;
use wasm_bindgen::prelude::*;

/// Error types for WASM API
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmErrorKind {
    /// Parse error in SMT-LIB2 script
    ParseError,
    /// Invalid sort name
    InvalidSort,
    /// No model available (check-sat not called or result was unsat)
    NoModel,
    /// No unsat core available
    NoUnsatCore,
    /// No proof available
    NoProof,
    /// Solver is in an invalid state for this operation
    InvalidState,
    /// Invalid input or argument
    InvalidInput,
    /// Operation not supported
    NotSupported,
    /// Unknown error
    Unknown,
}

impl fmt::Display for WasmErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError => write!(f, "ParseError"),
            Self::InvalidSort => write!(f, "InvalidSort"),
            Self::NoModel => write!(f, "NoModel"),
            Self::NoUnsatCore => write!(f, "NoUnsatCore"),
            Self::NoProof => write!(f, "NoProof"),
            Self::InvalidState => write!(f, "InvalidState"),
            Self::InvalidInput => write!(f, "InvalidInput"),
            Self::NotSupported => write!(f, "NotSupported"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Error type for WASM operations
#[derive(Debug, Clone)]
pub struct WasmError {
    kind: WasmErrorKind,
    message: String,
}

impl WasmError {
    /// Create a new error
    pub fn new(kind: WasmErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Get error kind
    pub fn kind(&self) -> WasmErrorKind {
        self.kind
    }

    /// Get error message
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)
    }
}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"kind".into(), &err.kind.to_string().into());
        let _ = js_sys::Reflect::set(&obj, &"message".into(), &err.message.into());
        obj.into()
    }
}

/// WASM-accessible SMT solver
///
/// This is the main interface for using OxiZ from JavaScript/TypeScript.
/// It provides a high-level API for SMT solving operations.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const solver = new WasmSolver();
/// solver.setLogic("QF_LIA");
/// solver.declareConst("x", "Int");
/// solver.assertFormula("(> x 0)");
/// const result = solver.checkSat(); // "sat"
/// const model = solver.getModel();
/// console.log(model.x.value); // prints the value of x
/// ```
#[wasm_bindgen]
pub struct WasmSolver {
    ctx: Context,
    /// Last check-sat result
    last_result: Option<String>,
    /// Cancellation flag
    cancelled: bool,
}

impl Default for WasmSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize panic hook for better error messages in WASM
///
/// This function is automatically called when the WASM module is loaded.
/// It configures better error messages for panics that occur in the WASM code.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Get the version of OxiZ WASM
///
/// Returns the current version string of the OxiZ WASM library.
///
/// # Returns
///
/// A version string in semver format (e.g., "0.1.2")
///
/// # Example (JavaScript)
///
/// ```javascript
/// import { version } from '@cooljapan/oxiz';
/// console.log(`OxiZ WASM version: ${version()}`);
/// ```
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;

    #[test]
    fn test_new_solver() {
        let solver = WasmSolver::new();
        assert!(solver.last_result.is_none());
    }

    #[test]
    fn test_declare_const() {
        let mut solver = WasmSolver::new();
        solver
            .declare_const("x", "Bool")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("z", "Real")
            .expect("test operation should succeed");
        solver
            .declare_const("bv", "BitVec32")
            .expect("test operation should succeed");
    }

    #[test]
    fn test_declare_const_invalid_sort() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_const("x", "InvalidSort");
        assert!(result.is_err());
    }

    #[test]
    fn test_declare_fun_nullary() {
        let mut solver = WasmSolver::new();
        solver
            .declare_fun("f", vec![], "Int")
            .expect("test operation should succeed");
    }

    #[test]
    fn test_check_sat_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.check_sat();
        assert_eq!(result, "sat");
        assert_eq!(solver.last_result.as_deref(), Some("sat"));
    }

    #[test]
    fn test_assert_formula() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        let result = solver.check_sat();
        assert_eq!(result, "sat");
    }

    #[test]
    fn test_assert_formula_unsat() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        solver
            .assert_formula("(not p)")
            .expect("test operation should succeed");
        let result = solver.check_sat();
        assert_eq!(result, "unsat");
    }

    #[test]
    fn test_get_model_before_check() {
        let solver = WasmSolver::new();
        let result = solver.get_model();
        assert!(result.is_err());
    }

    #[test]
    fn test_get_model_string() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        solver.check_sat();
        let model = solver
            .get_model_string()
            .expect("test operation should succeed");
        assert!(model.contains("model"));
    }

    #[test]
    fn test_get_assertions() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        let assertions = solver.get_assertions();
        assert!(assertions.contains("p"));
    }

    #[test]
    fn test_push_pop() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");

        solver.push();
        solver
            .assert_formula("(not p)")
            .expect("test operation should succeed");
        assert_eq!(solver.check_sat(), "unsat");

        solver.pop();
        assert_eq!(solver.check_sat(), "sat");
    }

    #[test]
    fn test_reset() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver.check_sat();
        assert!(solver.last_result.is_some());

        solver.reset();
        assert!(solver.last_result.is_none());
    }

    #[test]
    fn test_reset_assertions() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        solver.check_sat();

        solver.reset_assertions();
        assert!(solver.last_result.is_none());
        let assertions = solver.get_assertions();
        assert_eq!(assertions, "()");
    }

    #[test]
    fn test_set_get_option() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-models", "true");
        assert_eq!(
            solver.get_option("produce-models"),
            Some("true".to_string())
        );
    }

    #[test]
    fn test_simplify() {
        let mut solver = WasmSolver::new();
        let result = solver
            .simplify("(+ 1 2)")
            .expect("test operation should succeed");
        assert_eq!(result, "3");
    }

    #[test]
    fn test_execute() {
        let mut solver = WasmSolver::new();
        let result = solver
            .execute("(check-sat)")
            .expect("test operation should succeed");
        assert_eq!(
            result.as_string().expect("test operation should succeed"),
            "sat"
        );
    }

    #[test]
    fn test_parse_sort_bitvec() {
        let mut solver = WasmSolver::new();
        let sort = solver
            .parse_sort("BitVec8")
            .expect("test operation should succeed");
        assert_eq!(
            solver
                .ctx
                .terms
                .sorts
                .get(sort)
                .expect("key should exist in map")
                .bitvec_width(),
            Some(8)
        );

        let sort = solver
            .parse_sort("BitVec64")
            .expect("test operation should succeed");
        assert_eq!(
            solver
                .ctx
                .terms
                .sorts
                .get(sort)
                .expect("key should exist in map")
                .bitvec_width(),
            Some(64)
        );
    }

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_default() {
        let solver = WasmSolver::default();
        assert!(solver.last_result.is_none());
    }

    #[test]
    fn test_check_sat_assuming() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .declare_const("q", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("(or p q)")
            .expect("test operation should succeed");

        // Check assuming p is true
        let result = solver
            .check_sat_assuming(vec!["p".to_string()])
            .expect("serialization failed");
        assert_eq!(result, "sat");

        // Check assuming both p and q are false
        let result = solver
            .check_sat_assuming(vec!["(not p)".to_string(), "(not q)".to_string()])
            .expect("test operation should succeed");
        assert_eq!(result, "unsat");
    }

    #[test]
    fn test_check_sat_assuming_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.check_sat_assuming(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_define_sort() {
        let mut solver = WasmSolver::new();
        solver
            .define_sort("Word", "BitVec32")
            .expect("test operation should succeed");
        // After defining the sort, we should be able to use it
        // Note: The underlying context may not support custom sort names yet
    }

    #[test]
    fn test_define_fun() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Define a function that doubles its input
        solver
            .define_fun("double", vec!["x Int".to_string()], "Int", "(* 2 x)")
            .expect("test operation should succeed");

        // Note: Actually using the function requires parsing support
    }

    #[test]
    fn test_define_fun_invalid_params() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Invalid parameter format (missing sort)
        let result = solver.define_fun("f", vec!["x".to_string()], "Int", "x");
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_formula() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");

        // Valid formula
        let result = solver.validate_formula("(> x 0)");
        assert!(result.is_ok());

        // Invalid formula (undeclared variable)
        let result = solver.validate_formula("(> y 0)");
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_formula_safe() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");

        // Valid formula
        let result = solver.assert_formula_safe("(> x 0)");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_statistics() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        solver.check_sat();

        let stats = solver
            .get_statistics()
            .expect("test operation should succeed");
        assert!(stats.is_object());
    }

    #[test]
    fn test_get_info() {
        let solver = WasmSolver::new();

        let name = solver
            .get_info("name")
            .expect("test operation should succeed");
        assert_eq!(name, "OxiZ");

        let version = solver
            .get_info("version")
            .expect("test operation should succeed");
        assert!(!version.is_empty());

        let result = solver.get_info("unknown-key");
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_preset() {
        let mut solver = WasmSolver::new();

        // Test valid presets
        solver
            .apply_preset("default")
            .expect("test operation should succeed");
        solver
            .apply_preset("fast")
            .expect("test operation should succeed");
        solver
            .apply_preset("complete")
            .expect("test operation should succeed");
        solver
            .apply_preset("debug")
            .expect("test operation should succeed");
        solver
            .apply_preset("unsat-core")
            .expect("test operation should succeed");
        solver
            .apply_preset("incremental")
            .expect("test operation should succeed");

        // Test invalid preset
        let result = solver.apply_preset("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_debug_dump() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        solver.check_sat();

        let dump = solver.debug_dump();
        assert!(dump.contains("OxiZ"));
        assert!(dump.contains("QF_UF"));
        assert!(dump.contains("sat"));
    }

    #[test]
    fn test_set_tracing() {
        let mut solver = WasmSolver::new();
        solver.set_tracing(true);
        assert_eq!(solver.get_option("trace"), Some("true".to_string()));

        solver.set_tracing(false);
        assert_eq!(solver.get_option("trace"), Some("false".to_string()));
    }

    #[test]
    fn test_get_diagnostics() {
        let solver = WasmSolver::new();
        let warnings = solver.get_diagnostics();
        // Should have warnings about missing logic, etc.
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_check_pattern() {
        let solver = WasmSolver::new();

        let rec = solver.check_pattern("incremental");
        assert!(rec.contains("push/pop"));

        let rec = solver.check_pattern("assumptions");
        assert!(rec.contains("checkSatAssuming"));

        let rec = solver.check_pattern("async");
        assert!(rec.contains("async"));

        let rec = solver.check_pattern("validation");
        assert!(rec.contains("validateFormula"));

        let rec = solver.check_pattern("unknown");
        assert!(rec.contains("Unknown pattern"));
    }

    #[test]
    fn test_minimize() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");
        solver
            .assert_formula("(>= x 0)")
            .expect("test operation should succeed");
        solver
            .assert_formula("(>= y 0)")
            .expect("test operation should succeed");

        // Should succeed
        let result = solver.minimize("(+ x y)");
        assert!(result.is_ok());

        // Empty formula should fail
        let result = solver.minimize("");
        assert!(result.is_err());

        // Whitespace-only should fail
        let result = solver.minimize("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_maximize() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");
        solver
            .assert_formula("(<= x 10)")
            .expect("test operation should succeed");
        solver
            .assert_formula("(<= y 10)")
            .expect("test operation should succeed");

        // Should succeed
        let result = solver.maximize("(+ x y)");
        assert!(result.is_ok());

        // Empty formula should fail
        let result = solver.maximize("");
        assert!(result.is_err());
    }

    #[test]
    fn test_optimize() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");
        solver
            .assert_formula("(>= x 0)")
            .expect("test operation should succeed");
        solver
            .assert_formula("(>= y 0)")
            .expect("test operation should succeed");
        solver
            .assert_formula("(<= (+ x y) 10)")
            .expect("test operation should succeed");
        solver
            .maximize("(+ x y)")
            .expect("test operation should succeed");

        let result = solver.optimize();
        assert!(result.is_ok());

        let obj = result.expect("test operation should succeed");
        assert!(obj.is_object());

        // Check that status field exists
        let status =
            js_sys::Reflect::get(&obj, &"status".into()).expect("test operation should succeed");
        assert!(!status.is_undefined());
    }

    #[test]
    fn test_assert_soft() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .declare_const("q", "Bool")
            .expect("test operation should succeed");

        // Valid soft constraint
        let result = solver.assert_soft("p", "5");
        assert!(result.is_ok());

        // Empty formula should fail
        let result = solver.assert_soft("", "5");
        assert!(result.is_err());

        // Empty weight should fail
        let result = solver.assert_soft("q", "");
        assert!(result.is_err());

        // Invalid weight should fail
        let result = solver.assert_soft("q", "abc");
        assert!(result.is_err());

        // Negative weight should fail
        let result = solver.assert_soft("q", "-5");
        assert!(result.is_err());
    }

    #[test]
    fn test_optimization_unsat() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");
        solver
            .assert_formula("(not p)")
            .expect("test operation should succeed");
        solver.minimize("0").expect("test operation should succeed");

        let result = solver.optimize().expect("test operation should succeed");
        let status =
            js_sys::Reflect::get(&result, &"status".into()).expect("test operation should succeed");
        let status_str = status.as_string().expect("test operation should succeed");
        assert_eq!(status_str, "unsat");
    }

    #[test]
    fn test_get_minimal_model() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("z", "Int")
            .expect("test operation should succeed");
        solver
            .assert_formula("(= (+ x y z) 10)")
            .expect("test operation should succeed");
        solver
            .assert_formula("(= x 5)")
            .expect("test operation should succeed");
        solver
            .assert_formula("(= y 3)")
            .expect("test operation should succeed");

        solver.check_sat();

        // Get minimal model with only x and y
        let minimal = solver.get_minimal_model(vec!["x".to_string(), "y".to_string()]);
        assert!(minimal.is_ok());

        let model = minimal.expect("test operation should succeed");
        assert!(model.is_object());

        // Check that x and y are present
        let x_val =
            js_sys::Reflect::get(&model, &"x".into()).expect("test operation should succeed");
        assert!(!x_val.is_undefined());

        let y_val =
            js_sys::Reflect::get(&model, &"y".into()).expect("test operation should succeed");
        assert!(!y_val.is_undefined());

        // Check that z is not present
        let z_val =
            js_sys::Reflect::get(&model, &"z".into()).expect("test operation should succeed");
        assert!(z_val.is_undefined());
    }

    #[test]
    fn test_get_minimal_model_empty_vars() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");
        solver
            .assert_formula("(= x 5)")
            .expect("test operation should succeed");
        solver
            .assert_formula("(= y 3)")
            .expect("test operation should succeed");

        solver.check_sat();

        // Get minimal model with empty variables array (should return all)
        let minimal = solver.get_minimal_model(vec![]);
        assert!(minimal.is_ok());

        let model = minimal.expect("test operation should succeed");
        assert!(model.is_object());

        // All declared variables should be present
        let x_val =
            js_sys::Reflect::get(&model, &"x".into()).expect("test operation should succeed");
        assert!(!x_val.is_undefined());

        let y_val =
            js_sys::Reflect::get(&model, &"y".into()).expect("test operation should succeed");
        assert!(!y_val.is_undefined());
    }

    #[test]
    fn test_get_minimal_model_before_sat() {
        let solver = WasmSolver::new();

        // Should fail if checkSat not called
        let result = solver.get_minimal_model(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_minimal_model_nonexistent_var() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .assert_formula("(= x 5)")
            .expect("test operation should succeed");
        solver.check_sat();

        // Should fail for nonexistent variable
        let result = solver.get_minimal_model(vec!["nonexistent".to_string()]);
        assert!(result.is_err());
    }

    // Tests for interpolation support

    #[test]
    fn test_compute_interpolant_empty_partition_a() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");

        let result = solver.compute_interpolant(vec![], vec!["(> x 0)".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_interpolant_empty_partition_b() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");

        let result = solver.compute_interpolant(vec!["(> x 0)".to_string()], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_interpolant_no_proof_option() {
        let mut solver = WasmSolver::new();
        // Don't enable proof production

        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["(< x 0)".to_string()]);
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("produce-proofs"));
    }

    #[test]
    fn test_compute_interpolant_sat_formula() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");

        // These formulas are SAT together, not UNSAT
        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["(> x 1)".to_string()]);
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("UNSAT"));
    }

    #[test]
    fn test_compute_interpolant_unsat_formula() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");

        // These formulas are UNSAT together
        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["(< x 0)".to_string()]);

        // Should succeed and return an interpolant
        assert!(result.is_ok());
        let interpolant = result.expect("test operation should succeed");
        assert!(!interpolant.is_empty());
    }

    #[test]
    fn test_compute_interpolant_empty_formula_in_partition() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");

        // Empty formula in partition A
        let result = solver.compute_interpolant(vec!["".to_string()], vec!["(> x 0)".to_string()]);
        assert!(result.is_err());

        // Whitespace-only formula in partition B
        let result =
            solver.compute_interpolant(vec!["(> x 0)".to_string()], vec!["   ".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_interpolant_multiple_formulas() {
        let mut solver = WasmSolver::new();
        solver.set_option("produce-proofs", "true");
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");

        // Multiple formulas in each partition
        let result = solver.compute_interpolant(
            vec!["(> x 0)".to_string(), "(> x 5)".to_string()],
            vec!["(< y 0)".to_string(), "(= x y)".to_string()],
        );

        assert!(result.is_ok());
        let interpolant = result.expect("test operation should succeed");
        assert!(!interpolant.is_empty());
    }

    // Tests for quantifier elimination

    #[test]
    fn test_eliminate_quantifiers_empty_formula() {
        let mut solver = WasmSolver::new();

        let result = solver.eliminate_quantifiers("");
        assert!(result.is_err());
    }

    #[test]
    fn test_eliminate_quantifiers_whitespace_formula() {
        let mut solver = WasmSolver::new();

        let result = solver.eliminate_quantifiers("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_eliminate_quantifiers_no_quantifiers() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Formula without quantifiers
        let result = solver.eliminate_quantifiers("(> x 0)");
        assert!(result.is_ok());
        let qfree = result.expect("test operation should succeed");
        assert_eq!(qfree, "(> x 0)");
    }

    #[test]
    fn test_eliminate_quantifiers_exists() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        // Formula with existential quantifier
        let result = solver.eliminate_quantifiers("(exists ((y Int)) (= x (+ y 1)))");
        // Currently returns NotSupported
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("NotSupported") || err_str.contains("not yet fully implemented"));
    }

    #[test]
    fn test_eliminate_quantifiers_forall() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        // Formula with universal quantifier
        let result = solver.eliminate_quantifiers("(forall ((x Int)) (>= x 0))");
        // Currently returns NotSupported
        assert!(result.is_err());
        let err_str = format!("{:?}", result);
        assert!(err_str.contains("NotSupported") || err_str.contains("not yet fully implemented"));
    }

    #[test]
    fn test_eliminate_quantifiers_nested() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        // Formula with nested quantifiers
        let result =
            solver.eliminate_quantifiers("(exists ((x Int)) (forall ((y Int)) (>= (+ x y) 0)))");
        // Currently returns NotSupported
        assert!(result.is_err());
    }

    #[test]
    fn test_eliminate_quantifiers_error_message() {
        let mut solver = WasmSolver::new();
        solver.set_logic("LIA");

        let result = solver.eliminate_quantifiers("(exists ((x Int)) (> x 0))");
        assert!(result.is_err());

        // Check that error message is informative
        let err_str = format!("{:?}", result);
        assert!(
            err_str.contains("Cooper") || err_str.contains("CAD") || err_str.contains("parser")
        );
    }

    // Tests for new arithmetic/logical operators

    #[test]
    fn test_mk_div() {
        let solver = WasmSolver::new();
        let div = solver
            .mk_div(vec!["x".to_string(), "2".to_string()])
            .expect("test operation should succeed");
        assert_eq!(div, "(/ x 2)");
    }

    #[test]
    fn test_mk_div_multiple_operands() {
        let solver = WasmSolver::new();
        let div = solver
            .mk_div(vec!["100".to_string(), "5".to_string(), "2".to_string()])
            .expect("test operation should succeed");
        assert_eq!(div, "(/ 100 5 2)");
    }

    #[test]
    fn test_mk_div_empty() {
        let solver = WasmSolver::new();
        let result = solver.mk_div(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_div_single_operand() {
        let solver = WasmSolver::new();
        let result = solver.mk_div(vec!["x".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_mod() {
        let solver = WasmSolver::new();
        let modulo = solver
            .mk_mod("x", "5")
            .expect("test operation should succeed");
        assert_eq!(modulo, "(mod x 5)");
    }

    #[test]
    fn test_mk_mod_empty_lhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_mod("", "5");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_mod_empty_rhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_mod("x", "");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_neg() {
        let solver = WasmSolver::new();
        let neg = solver.mk_neg("x").expect("test operation should succeed");
        assert_eq!(neg, "(- x)");
    }

    #[test]
    fn test_mk_neg_expression() {
        let solver = WasmSolver::new();
        let neg = solver
            .mk_neg("(+ x 5)")
            .expect("test operation should succeed");
        assert_eq!(neg, "(- (+ x 5))");
    }

    #[test]
    fn test_mk_neg_empty() {
        let solver = WasmSolver::new();
        let result = solver.mk_neg("");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_xor() {
        let solver = WasmSolver::new();
        let xor = solver
            .mk_xor("p", "q")
            .expect("test operation should succeed");
        assert_eq!(xor, "(xor p q)");
    }

    #[test]
    fn test_mk_xor_empty_lhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_xor("", "q");
        assert!(result.is_err());
    }

    #[test]
    fn test_mk_xor_empty_rhs() {
        let solver = WasmSolver::new();
        let result = solver.mk_xor("p", "");
        assert!(result.is_err());
    }

    // Tests for batch operations

    #[test]
    fn test_declare_funs() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        let result = solver.declare_funs(vec![
            "x Int".to_string(),
            "y Int".to_string(),
            "z Bool".to_string(),
        ]);
        assert!(result.is_ok());

        // Verify we can use the declared variables
        assert!(solver.assert_formula("(> x 0)").is_ok());
        assert!(solver.assert_formula("(< y 10)").is_ok());
        assert!(solver.assert_formula("z").is_ok());
    }

    #[test]
    fn test_declare_funs_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_funs(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_declare_funs_invalid_format() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_funs(vec!["x".to_string()]); // Missing sort
        assert!(result.is_err());
    }

    #[test]
    fn test_declare_funs_invalid_sort() {
        let mut solver = WasmSolver::new();
        let result = solver.declare_funs(vec!["x InvalidSort".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_formulas() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");

        let result = solver.assert_formulas(vec![
            "(> x 0)".to_string(),
            "(< y 10)".to_string(),
            "(< x y)".to_string(),
        ]);
        assert!(result.is_ok());

        // Verify assertions were added
        let assertions = solver.get_assertions();
        assert!(assertions.contains("x"));
        assert!(assertions.contains("y"));
    }

    #[test]
    fn test_assert_formulas_empty() {
        let mut solver = WasmSolver::new();
        let result = solver.assert_formulas(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_formulas_with_error() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");

        // Second formula is invalid (undeclared variable)
        let result = solver.assert_formulas(vec![
            "(> x 0)".to_string(),
            "(> undefined_var 5)".to_string(),
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_named() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.set_option("produce-unsat-cores", "true");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");

        let result = solver.assert_named("constraint1", "(> x 0)");
        assert!(result.is_ok());

        let result = solver.assert_named("constraint2", "(< x 0)");
        assert!(result.is_ok());

        // Check for unsat
        let check_result = solver.check_sat();
        assert_eq!(check_result, "unsat");
    }

    #[test]
    fn test_assert_named_empty_name() {
        let mut solver = WasmSolver::new();
        let result = solver.assert_named("", "(> x 0)");
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_named_empty_formula() {
        let mut solver = WasmSolver::new();
        let result = solver.assert_named("test", "");
        assert!(result.is_err());
    }

    #[test]
    fn test_assert_named_with_unsat_core() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver.set_option("produce-unsat-cores", "true");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");

        solver
            .assert_named("positive", "(> x 0)")
            .expect("test operation should succeed");
        solver
            .assert_named("negative", "(< x 0)")
            .expect("test operation should succeed");

        assert_eq!(solver.check_sat(), "unsat");

        // Try to get unsat core (implementation depends on solver support)
        let core_result = solver.get_unsat_core();
        // We don't assert the exact core content since it depends on solver implementation
        // But the call should not panic
        let _ = core_result;
    }

    // Integration tests combining new features

    #[test]
    fn test_batch_declare_and_assert() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");

        // Batch declare
        solver
            .declare_funs(vec![
                "a Int".to_string(),
                "b Int".to_string(),
                "c Int".to_string(),
            ])
            .expect("test operation should succeed");

        // Batch assert
        solver
            .assert_formulas(vec![
                "(> a 0)".to_string(),
                "(> b a)".to_string(),
                "(= c (+ a b))".to_string(),
            ])
            .expect("test operation should succeed");

        // Check satisfiability
        assert_eq!(solver.check_sat(), "sat");
    }

    #[test]
    fn test_new_operators_integration() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_LIA");
        solver
            .declare_const("x", "Int")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "Int")
            .expect("test operation should succeed");

        // Use new operators to build formulas
        let div_expr = solver
            .mk_div(vec!["x".to_string(), "3".to_string()])
            .expect("test operation should succeed");
        let mod_expr = solver
            .mk_mod("y", "2")
            .expect("test operation should succeed");
        let neg_expr = solver.mk_neg("x").expect("test operation should succeed");

        // Assert formulas using the built expressions
        solver
            .assert_formula(&format!("(= {} 5)", div_expr))
            .expect("test operation should succeed");
        solver
            .assert_formula(&format!("(= {} 0)", mod_expr))
            .expect("test operation should succeed");
        solver
            .assert_formula(&format!("(> {} -20)", neg_expr))
            .expect("test operation should succeed");

        // Should be satisfiable
        assert_eq!(solver.check_sat(), "sat");
    }

    #[test]
    fn test_mk_xor_integration() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_UF");
        solver
            .declare_const("p", "Bool")
            .expect("test operation should succeed");
        solver
            .declare_const("q", "Bool")
            .expect("test operation should succeed");

        let xor_expr = solver
            .mk_xor("p", "q")
            .expect("test operation should succeed");
        solver
            .assert_formula(&xor_expr)
            .expect("test operation should succeed");
        solver
            .assert_formula("p")
            .expect("test operation should succeed");

        // If p is true and (xor p q) is true, then q must be false
        assert_eq!(solver.check_sat(), "sat");

        // Verify the model
        if let Ok(model_val) = solver.get_model() {
            // Model should have p=true and q=false
            let _ = model_val; // Just verify it doesn't panic
        }
    }

    // Tests for bitvector operations

    #[test]
    fn test_mk_bvand() {
        let solver = WasmSolver::new();
        let bvand = solver
            .mk_bvand("x", "y")
            .expect("test operation should succeed");
        assert_eq!(bvand, "(bvand x y)");
    }

    #[test]
    fn test_mk_bvand_empty() {
        let solver = WasmSolver::new();
        assert!(solver.mk_bvand("", "y").is_err());
        assert!(solver.mk_bvand("x", "").is_err());
    }

    #[test]
    fn test_mk_bvor() {
        let solver = WasmSolver::new();
        let bvor = solver
            .mk_bvor("x", "y")
            .expect("test operation should succeed");
        assert_eq!(bvor, "(bvor x y)");
    }

    #[test]
    fn test_mk_bvxor() {
        let solver = WasmSolver::new();
        let bvxor = solver
            .mk_bvxor("x", "y")
            .expect("test operation should succeed");
        assert_eq!(bvxor, "(bvxor x y)");
    }

    #[test]
    fn test_mk_bvnot() {
        let solver = WasmSolver::new();
        let bvnot = solver.mk_bvnot("x").expect("test operation should succeed");
        assert_eq!(bvnot, "(bvnot x)");
    }

    #[test]
    fn test_mk_bvnot_empty() {
        let solver = WasmSolver::new();
        assert!(solver.mk_bvnot("").is_err());
    }

    #[test]
    fn test_mk_bvneg() {
        let solver = WasmSolver::new();
        let bvneg = solver.mk_bvneg("x").expect("test operation should succeed");
        assert_eq!(bvneg, "(bvneg x)");
    }

    #[test]
    fn test_mk_bvadd() {
        let solver = WasmSolver::new();
        let bvadd = solver
            .mk_bvadd("x", "y")
            .expect("test operation should succeed");
        assert_eq!(bvadd, "(bvadd x y)");
    }

    #[test]
    fn test_mk_bvsub() {
        let solver = WasmSolver::new();
        let bvsub = solver
            .mk_bvsub("x", "y")
            .expect("test operation should succeed");
        assert_eq!(bvsub, "(bvsub x y)");
    }

    #[test]
    fn test_mk_bvmul() {
        let solver = WasmSolver::new();
        let bvmul = solver
            .mk_bvmul("x", "y")
            .expect("test operation should succeed");
        assert_eq!(bvmul, "(bvmul x y)");
    }

    #[test]
    fn test_bitvector_integration() {
        let mut solver = WasmSolver::new();
        solver.set_logic("QF_BV");
        solver
            .declare_const("x", "BitVec8")
            .expect("test operation should succeed");
        solver
            .declare_const("y", "BitVec8")
            .expect("test operation should succeed");

        // Build formulas with bitvector operations
        let and_expr = solver
            .mk_bvand("x", "y")
            .expect("test operation should succeed");
        let or_expr = solver
            .mk_bvor("x", "y")
            .expect("test operation should succeed");
        let xor_expr = solver
            .mk_bvxor("x", "y")
            .expect("test operation should succeed");

        // These should parse correctly (we're just testing formula building)
        let _ = solver.validate_formula(&and_expr);
        let _ = solver.validate_formula(&or_expr);
        let _ = solver.validate_formula(&xor_expr);
    }
}
