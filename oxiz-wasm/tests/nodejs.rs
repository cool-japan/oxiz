//! Node.js integration tests for oxiz-wasm
//!
//! Run these tests with: wasm-pack test --node

#![cfg(target_arch = "wasm32")]

use oxiz_wasm::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node_js);

#[wasm_bindgen_test]
fn test_nodejs_basic() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    assert_eq!(solver.check_sat(), "sat");
}

#[wasm_bindgen_test]
fn test_nodejs_integer_arithmetic() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.declare_const("y", "Int").unwrap();
    solver.assert_formula("(> x 0)").unwrap();
    solver.assert_formula("(< y 10)").unwrap();
    solver.assert_formula("(= (+ x y) 15)").unwrap();

    assert_eq!(solver.check_sat(), "sat");
    let model = solver.get_model().unwrap();
    assert!(model.is_object());
}

#[wasm_bindgen_test]
fn test_nodejs_push_pop() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.assert_formula("(> x 0)").unwrap();

    solver.push();
    solver.assert_formula("(< x 0)").unwrap();
    assert_eq!(solver.check_sat(), "unsat");

    solver.pop();
    assert_eq!(solver.check_sat(), "sat");
}

#[wasm_bindgen_test]
fn test_nodejs_incremental_solving() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();

    // Solve multiple related problems
    solver.push();
    solver.assert_formula("(> x 5)").unwrap();
    assert_eq!(solver.check_sat(), "sat");
    solver.pop();

    solver.push();
    solver.assert_formula("(< x 0)").unwrap();
    assert_eq!(solver.check_sat(), "sat");
    solver.pop();

    solver.push();
    solver.assert_formula("(= x 10)").unwrap();
    assert_eq!(solver.check_sat(), "sat");
    solver.pop();
}

#[wasm_bindgen_test]
fn test_nodejs_check_sat_assuming() {
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
fn test_nodejs_define_fun() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");

    solver
        .define_fun("double", vec!["x Int".to_string()], "Int", "(* 2 x)")
        .unwrap();
    solver
        .define_fun(
            "max2",
            vec!["a Int".to_string(), "b Int".to_string()],
            "Int",
            "(ite (> a b) a b)",
        )
        .unwrap();
}

#[wasm_bindgen_test]
fn test_nodejs_bitvector() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_BV");
    solver.declare_const("x", "BitVec32").unwrap();
    solver.declare_const("y", "BitVec32").unwrap();

    // We can declare them, but operations may be limited without full BV support
    assert_eq!(solver.check_sat(), "sat");
}

#[wasm_bindgen_test]
fn test_nodejs_options() {
    let mut solver = WasmSolver::new();
    solver.set_option("produce-models", "true");
    solver.set_option("produce-unsat-cores", "true");

    assert_eq!(
        solver.get_option("produce-models"),
        Some("true".to_string())
    );
    assert_eq!(
        solver.get_option("produce-unsat-cores"),
        Some("true".to_string())
    );
}

#[wasm_bindgen_test]
fn test_nodejs_presets() {
    let mut solver = WasmSolver::new();

    solver.apply_preset("default").unwrap();
    solver.apply_preset("fast").unwrap();
    solver.apply_preset("complete").unwrap();
    solver.apply_preset("incremental").unwrap();
}

#[wasm_bindgen_test]
fn test_nodejs_statistics() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    let stats = solver.get_statistics().unwrap();
    assert!(stats.is_object());
}

#[wasm_bindgen_test]
fn test_nodejs_info() {
    let solver = WasmSolver::new();

    let name = solver.get_info("name").unwrap();
    assert_eq!(name, "OxiZ");

    let version = solver.get_info("version").unwrap();
    assert!(!version.is_empty());

    let authors = solver.get_info("authors").unwrap();
    assert!(!authors.is_empty());
}

#[wasm_bindgen_test]
fn test_nodejs_diagnostics() {
    let solver = WasmSolver::new();
    let diagnostics = solver.get_diagnostics();
    assert!(!diagnostics.is_empty());
}

#[wasm_bindgen_test]
fn test_nodejs_debug_dump() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.assert_formula("(> x 0)").unwrap();
    solver.check_sat();

    let dump = solver.debug_dump();
    assert!(dump.contains("OxiZ"));
    assert!(dump.contains("QF_LIA"));
}

#[wasm_bindgen_test]
fn test_nodejs_error_handling() {
    let mut solver = WasmSolver::new();

    // Invalid sort
    assert!(solver.declare_const("x", "InvalidSort").is_err());

    // Empty formula
    assert!(solver.assert_formula("").is_err());

    // Get model before check-sat
    assert!(solver.get_model().is_err());

    // Invalid preset
    assert!(solver.apply_preset("invalid").is_err());
}

#[wasm_bindgen_test]
fn test_nodejs_validate_formula() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();

    assert!(solver.validate_formula("(> x 0)").is_ok());
    assert!(solver.validate_formula("(> y 0)").is_err());
}

#[wasm_bindgen_test]
fn test_nodejs_simplify() {
    let mut solver = WasmSolver::new();

    let result = solver.simplify("(+ 1 2)").unwrap();
    assert_eq!(result, "3");

    let result = solver.simplify("(and true false)").unwrap();
    assert_eq!(result, "false");
}

#[wasm_bindgen_test]
fn test_nodejs_execute_script() {
    let mut solver = WasmSolver::new();
    let script = r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (> x 0))
        (check-sat)
    "#;

    let result = solver.execute(script).unwrap();
    assert!(result.as_string().unwrap().contains("sat"));
}

#[wasm_bindgen_test]
fn test_nodejs_real_arithmetic() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LRA");
    solver.declare_const("x", "Real").unwrap();
    solver.declare_const("y", "Real").unwrap();

    solver.assert_formula("(> x 0.0)").unwrap();
    solver.assert_formula("(< y 1.0)").unwrap();

    assert_eq!(solver.check_sat(), "sat");
}

#[wasm_bindgen_test]
async fn test_nodejs_async_check_sat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();

    let result = solver.check_sat_async().await;
    assert_eq!(result, "sat");
}

#[wasm_bindgen_test]
async fn test_nodejs_async_execute() {
    let mut solver = WasmSolver::new();
    let script = "(check-sat)".to_string();

    let result = solver.execute_async(script).await.unwrap();
    assert_eq!(result.as_string().unwrap(), "sat");
}

#[wasm_bindgen_test]
fn test_nodejs_cancel() {
    let mut solver = WasmSolver::new();
    assert!(!solver.is_cancelled());

    solver.cancel();
    assert!(solver.is_cancelled());

    solver.reset();
    assert!(!solver.is_cancelled());
}

#[wasm_bindgen_test]
fn test_nodejs_tracing() {
    let mut solver = WasmSolver::new();

    solver.set_tracing(true);
    assert_eq!(solver.get_option("trace"), Some("true".to_string()));

    solver.set_tracing(false);
    assert_eq!(solver.get_option("trace"), Some("false".to_string()));
}

#[wasm_bindgen_test]
fn test_nodejs_pattern_checking() {
    let solver = WasmSolver::new();

    let rec = solver.check_pattern("incremental");
    assert!(rec.contains("push/pop"));

    let rec = solver.check_pattern("assumptions");
    assert!(rec.contains("checkSatAssuming"));

    let rec = solver.check_pattern("async");
    assert!(rec.contains("async"));
}

#[wasm_bindgen_test]
fn test_nodejs_stress_push_pop() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();

    // Push and pop multiple times
    for i in 0..10 {
        solver.push();
        solver.assert_formula(&format!("(> x {})", i)).unwrap();
        assert_eq!(solver.check_sat(), "sat");
    }

    for _ in 0..10 {
        solver.pop();
    }

    assert_eq!(solver.check_sat(), "sat");
}

#[wasm_bindgen_test]
fn test_nodejs_multiple_solvers() {
    let mut solver1 = WasmSolver::new();
    let mut solver2 = WasmSolver::new();

    solver1.set_logic("QF_UF");
    solver2.set_logic("QF_LIA");

    solver1.declare_const("p", "Bool").unwrap();
    solver2.declare_const("x", "Int").unwrap();

    solver1.assert_formula("p").unwrap();
    solver2.assert_formula("(> x 0)").unwrap();

    assert_eq!(solver1.check_sat(), "sat");
    assert_eq!(solver2.check_sat(), "sat");
}
