//! Browser-based tests for oxiz-wasm using wasm-pack test
//!
//! Run these tests with: wasm-pack test --headless --chrome
//! Or with Firefox: wasm-pack test --headless --firefox

#![cfg(target_arch = "wasm32")]

use oxiz_wasm::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_new_solver() {
    let solver = WasmSolver::new();
    assert!(solver.is_cancelled() == false);
}

#[wasm_bindgen_test]
fn test_version() {
    let v = version();
    assert!(!v.is_empty());
    assert!(v.contains('.'));
}

#[wasm_bindgen_test]
fn test_declare_const_bool() {
    let mut solver = WasmSolver::new();
    solver.declare_const("p", "Bool").unwrap();
}

#[wasm_bindgen_test]
fn test_declare_const_int() {
    let mut solver = WasmSolver::new();
    solver.declare_const("x", "Int").unwrap();
}

#[wasm_bindgen_test]
fn test_declare_const_real() {
    let mut solver = WasmSolver::new();
    solver.declare_const("y", "Real").unwrap();
}

#[wasm_bindgen_test]
fn test_declare_const_bitvec() {
    let mut solver = WasmSolver::new();
    solver.declare_const("bv", "BitVec32").unwrap();
    solver.declare_const("bv2", "BitVec8").unwrap();
    solver.declare_const("bv3", "BitVec64").unwrap();
}

#[wasm_bindgen_test]
fn test_declare_const_invalid_sort() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_const("x", "InvalidSort");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_declare_const_empty_name() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_const("", "Int");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_check_sat_empty() {
    let mut solver = WasmSolver::new();
    let result = solver.check_sat();
    assert_eq!(result, "sat");
}

#[wasm_bindgen_test]
fn test_check_sat_simple_sat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    let result = solver.check_sat();
    assert_eq!(result, "sat");
}

#[wasm_bindgen_test]
fn test_check_sat_simple_unsat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.assert_formula("(not p)").unwrap();
    let result = solver.check_sat();
    assert_eq!(result, "unsat");
}

#[wasm_bindgen_test]
fn test_assert_formula_empty() {
    let mut solver = WasmSolver::new();
    let result = solver.assert_formula("");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_get_model_without_check() {
    let solver = WasmSolver::new();
    let result = solver.get_model();
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_get_model_after_sat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();
    let result = solver.get_model();
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn test_get_model_string() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();
    let model = solver.get_model_string().unwrap();
    assert!(model.contains("model") || model.len() > 0);
}

#[wasm_bindgen_test]
fn test_push_pop() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();

    solver.push();
    solver.assert_formula("(not p)").unwrap();
    assert_eq!(solver.check_sat(), "unsat");

    solver.pop();
    assert_eq!(solver.check_sat(), "sat");
}

#[wasm_bindgen_test]
fn test_reset() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    solver.reset();
    let result = solver.check_sat();
    assert_eq!(result, "sat");
}

#[wasm_bindgen_test]
fn test_reset_assertions() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    solver.reset_assertions();
    let assertions = solver.get_assertions();
    assert_eq!(assertions, "()");
}

#[wasm_bindgen_test]
fn test_set_get_option() {
    let mut solver = WasmSolver::new();
    solver.set_option("produce-models", "true");
    assert_eq!(
        solver.get_option("produce-models"),
        Some("true".to_string())
    );
}

#[wasm_bindgen_test]
fn test_simplify() {
    let mut solver = WasmSolver::new();
    let result = solver.simplify("(+ 1 2)").unwrap();
    assert_eq!(result, "3");
}

#[wasm_bindgen_test]
fn test_simplify_empty() {
    let mut solver = WasmSolver::new();
    let result = solver.simplify("");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_execute() {
    let mut solver = WasmSolver::new();
    let result = solver.execute("(check-sat)").unwrap();
    assert_eq!(result.as_string().unwrap(), "sat");
}

#[wasm_bindgen_test]
fn test_execute_empty() {
    let mut solver = WasmSolver::new();
    let result = solver.execute("");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_check_sat_assuming() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.declare_const("q", "Bool").unwrap();
    solver.assert_formula("(or p q)").unwrap();

    let result = solver.check_sat_assuming(vec!["p".to_string()]).unwrap();
    assert_eq!(result, "sat");

    let result = solver
        .check_sat_assuming(vec!["(not p)".to_string(), "(not q)".to_string()])
        .unwrap();
    assert_eq!(result, "unsat");
}

#[wasm_bindgen_test]
fn test_check_sat_assuming_empty() {
    let mut solver = WasmSolver::new();
    let result = solver.check_sat_assuming(vec![]);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_define_sort() {
    let mut solver = WasmSolver::new();
    solver.define_sort("Word", "BitVec32").unwrap();
}

#[wasm_bindgen_test]
fn test_define_sort_invalid() {
    let mut solver = WasmSolver::new();
    let result = solver.define_sort("", "Int");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_define_fun() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver
        .define_fun("double", vec!["x Int".to_string()], "Int", "(* 2 x)")
        .unwrap();
}

#[wasm_bindgen_test]
fn test_define_fun_invalid_params() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    let result = solver.define_fun("f", vec!["x".to_string()], "Int", "x");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_validate_formula() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();

    let result = solver.validate_formula("(> x 0)");
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn test_validate_formula_invalid() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    let result = solver.validate_formula("(> y 0)");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_assert_formula_safe() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();

    let result = solver.assert_formula_safe("(> x 0)");
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
fn test_get_statistics() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    let stats = solver.get_statistics().unwrap();
    assert!(stats.is_object());
}

#[wasm_bindgen_test]
fn test_get_info() {
    let solver = WasmSolver::new();

    let name = solver.get_info("name").unwrap();
    assert_eq!(name, "OxiZ");

    let version = solver.get_info("version").unwrap();
    assert!(!version.is_empty());
}

#[wasm_bindgen_test]
fn test_get_info_invalid() {
    let solver = WasmSolver::new();
    let result = solver.get_info("unknown-key");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_apply_preset() {
    let mut solver = WasmSolver::new();

    solver.apply_preset("default").unwrap();
    solver.apply_preset("fast").unwrap();
    solver.apply_preset("complete").unwrap();
    solver.apply_preset("debug").unwrap();
    solver.apply_preset("unsat-core").unwrap();
    solver.apply_preset("incremental").unwrap();
}

#[wasm_bindgen_test]
fn test_apply_preset_invalid() {
    let mut solver = WasmSolver::new();
    let result = solver.apply_preset("invalid");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_debug_dump() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    let dump = solver.debug_dump();
    assert!(dump.contains("OxiZ"));
    assert!(dump.contains("QF_UF"));
    assert!(dump.contains("sat"));
}

#[wasm_bindgen_test]
fn test_set_tracing() {
    let mut solver = WasmSolver::new();
    solver.set_tracing(true);
    assert_eq!(solver.get_option("trace"), Some("true".to_string()));

    solver.set_tracing(false);
    assert_eq!(solver.get_option("trace"), Some("false".to_string()));
}

#[wasm_bindgen_test]
fn test_get_diagnostics() {
    let solver = WasmSolver::new();
    let warnings = solver.get_diagnostics();
    assert!(!warnings.is_empty());
}

#[wasm_bindgen_test]
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
}

#[wasm_bindgen_test]
fn test_cancel_flag() {
    let mut solver = WasmSolver::new();
    assert!(!solver.is_cancelled());

    solver.cancel();
    assert!(solver.is_cancelled());

    solver.reset();
    assert!(!solver.is_cancelled());
}

#[wasm_bindgen_test]
fn test_get_value() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.assert_formula("(> x 5)").unwrap();
    solver.check_sat();

    let values = solver.get_value(vec!["x".to_string()]);
    assert!(values.is_ok());
}

#[wasm_bindgen_test]
fn test_get_value_empty() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    let result = solver.get_value(vec![]);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_get_unsat_core() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.assert_formula("(not p)").unwrap();
    solver.check_sat();

    let core = solver.get_unsat_core();
    assert!(core.is_ok());
}

#[wasm_bindgen_test]
fn test_get_unsat_core_without_unsat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    let result = solver.get_unsat_core();
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_get_assertions() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();

    let assertions = solver.get_assertions();
    assert!(assertions.contains("p") || assertions == "()");
}

#[wasm_bindgen_test]
fn test_declare_fun_nullary() {
    let mut solver = WasmSolver::new();
    solver.declare_fun("c", vec![], "Int").unwrap();
}

#[wasm_bindgen_test]
fn test_declare_fun_empty_name() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_fun("", vec![], "Int");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn test_bitvec_various_widths() {
    let mut solver = WasmSolver::new();
    solver.declare_const("bv1", "BitVec1").unwrap();
    solver.declare_const("bv16", "BitVec16").unwrap();
    solver.declare_const("bv128", "BitVec128").unwrap();
    solver.declare_const("bv256", "BitVec256").unwrap();
}

#[wasm_bindgen_test]
fn test_bitvec_invalid_width() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_const("bv", "BitVec0");
    assert!(result.is_err());

    let result = solver.declare_const("bv", "BitVecABC");
    assert!(result.is_err());

    let result = solver.declare_const("bv", "BitVec");
    assert!(result.is_err());
}

#[wasm_bindgen_test]
async fn test_check_sat_async() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();

    let result = solver.check_sat_async().await;
    assert_eq!(result, "sat");
}

#[wasm_bindgen_test]
async fn test_execute_async() {
    let mut solver = WasmSolver::new();
    let script = "(check-sat)".to_string();

    let result = solver.execute_async(script).await.unwrap();
    assert_eq!(result.as_string().unwrap(), "sat");
}

#[wasm_bindgen_test]
fn test_logic_setting() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.set_logic("QF_LRA");
    solver.set_logic("QF_UF");
    solver.set_logic("QF_BV");
    solver.set_logic("ALL");
}

#[wasm_bindgen_test]
fn test_integer_arithmetic() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.declare_const("y", "Int").unwrap();
    solver.assert_formula("(= x 5)").unwrap();
    solver.assert_formula("(= y 10)").unwrap();
    solver.assert_formula("(= (+ x y) 15)").unwrap();

    let result = solver.check_sat();
    assert_eq!(result, "sat");
}

#[wasm_bindgen_test]
fn test_boolean_logic() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.declare_const("q", "Bool").unwrap();
    solver.declare_const("r", "Bool").unwrap();

    solver.assert_formula("(=> p q)").unwrap();
    solver.assert_formula("(=> q r)").unwrap();
    solver.assert_formula("p").unwrap();
    solver.assert_formula("(not r)").unwrap();

    let result = solver.check_sat();
    assert_eq!(result, "unsat");
}
