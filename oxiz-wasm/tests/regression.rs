//! Regression tests for oxiz-wasm
//!
//! This file contains tests for specific bugs and edge cases that should not regress.
//! Each test should be well-documented with a comment explaining what issue it prevents.

#![cfg(target_arch = "wasm32")]

use oxiz_wasm::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

// ====================================================================
// REGRESSION: Empty input validation
// ====================================================================

/// Regression test: Ensure empty script is properly rejected
/// Previously, empty scripts might cause unexpected behavior
#[wasm_bindgen_test]
fn test_empty_script_rejected() {
    let mut solver = WasmSolver::new();
    let result = solver.execute("");
    assert!(result.is_err());

    let result = solver.execute("   ");
    assert!(result.is_err());

    let result = solver.execute("\n\n");
    assert!(result.is_err());
}

/// Regression test: Ensure empty formulas are properly rejected
#[wasm_bindgen_test]
fn test_empty_formula_rejected() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");

    let result = solver.assert_formula("");
    assert!(result.is_err());

    let result = solver.assert_formula("   ");
    assert!(result.is_err());
}

/// Regression test: Ensure empty constant names are rejected
#[wasm_bindgen_test]
fn test_empty_constant_name_rejected() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_const("", "Int");
    assert!(result.is_err());

    let result = solver.declare_const("   ", "Int");
    assert!(result.is_err());
}

/// Regression test: Ensure empty sort names are rejected
#[wasm_bindgen_test]
fn test_empty_sort_name_rejected() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_const("x", "");
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: Invalid sort handling
// ====================================================================

/// Regression test: Ensure invalid sort names produce proper errors
#[wasm_bindgen_test]
fn test_invalid_sort_names() {
    let mut solver = WasmSolver::new();

    // Unknown sort
    let result = solver.declare_const("x", "InvalidSort");
    assert!(result.is_err());

    // Typos in common sorts
    let result = solver.declare_const("x", "bool");
    assert!(result.is_err());

    let result = solver.declare_const("x", "Integer");
    assert!(result.is_err());

    let result = solver.declare_const("x", "boolean");
    assert!(result.is_err());
}

/// Regression test: Ensure BitVec requires width
#[wasm_bindgen_test]
fn test_bitvec_requires_width() {
    let mut solver = WasmSolver::new();

    // BitVec without width should fail
    let result = solver.declare_const("bv", "BitVec");
    assert!(result.is_err());
}

/// Regression test: Ensure BitVec width must be positive
#[wasm_bindgen_test]
fn test_bitvec_zero_width_rejected() {
    let mut solver = WasmSolver::new();

    // BitVec with zero width should fail
    let result = solver.declare_const("bv", "BitVec0");
    assert!(result.is_err());
}

/// Regression test: Ensure BitVec width must be numeric
#[wasm_bindgen_test]
fn test_bitvec_invalid_width() {
    let mut solver = WasmSolver::new();

    let result = solver.declare_const("bv", "BitVecXYZ");
    assert!(result.is_err());

    let result = solver.declare_const("bv", "BitVec-1");
    assert!(result.is_err());

    let result = solver.declare_const("bv", "BitVec3.14");
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: Model availability checks
// ====================================================================

/// Regression test: Getting model before check-sat should fail
#[wasm_bindgen_test]
fn test_get_model_before_check_sat() {
    let solver = WasmSolver::new();
    let result = solver.get_model();
    assert!(result.is_err());

    let result = solver.get_model_string();
    assert!(result.is_err());
}

/// Regression test: Getting model after unsat should fail
#[wasm_bindgen_test]
fn test_get_model_after_unsat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.assert_formula("(not p)").unwrap();

    let result = solver.check_sat();
    assert_eq!(result, "unsat");

    let model_result = solver.get_model();
    assert!(model_result.is_err());
}

/// Regression test: Getting value before check-sat should fail
#[wasm_bindgen_test]
fn test_get_value_before_check_sat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();

    let result = solver.get_value(vec!["x".to_string()]);
    assert!(result.is_err());
}

/// Regression test: Getting value with empty terms should fail
#[wasm_bindgen_test]
fn test_get_value_empty_terms() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.assert_formula("(> x 0)").unwrap();
    solver.check_sat();

    let result = solver.get_value(vec![]);
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: Unsat core availability
// ====================================================================

/// Regression test: Getting unsat core before check-sat should fail
#[wasm_bindgen_test]
fn test_get_unsat_core_before_check_sat() {
    let solver = WasmSolver::new();
    let result = solver.get_unsat_core();
    assert!(result.is_err());
}

/// Regression test: Getting unsat core after sat should fail
#[wasm_bindgen_test]
fn test_get_unsat_core_after_sat() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();

    let result = solver.check_sat();
    assert_eq!(result, "sat");

    let core_result = solver.get_unsat_core();
    assert!(core_result.is_err());
}

// ====================================================================
// REGRESSION: State management
// ====================================================================

/// Regression test: Reset should clear last_result
#[wasm_bindgen_test]
fn test_reset_clears_last_result() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    // Model should be available
    let model = solver.get_model();
    assert!(model.is_ok());

    solver.reset();

    // Model should no longer be available
    let model = solver.get_model();
    assert!(model.is_err());
}

/// Regression test: Reset assertions should clear last_result
#[wasm_bindgen_test]
fn test_reset_assertions_clears_last_result() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    // Model should be available
    let model = solver.get_model();
    assert!(model.is_ok());

    solver.reset_assertions();

    // Model should no longer be available
    let model = solver.get_model();
    assert!(model.is_err());
}

/// Regression test: Push/pop should preserve check-sat results
#[wasm_bindgen_test]
fn test_push_pop_preserves_result() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();

    let result = solver.check_sat();
    assert_eq!(result, "sat");

    solver.push();
    solver.pop();

    // Result should still be available
    let model = solver.get_model();
    assert!(model.is_ok());
}

// ====================================================================
// REGRESSION: check-sat-assuming
// ====================================================================

/// Regression test: check-sat-assuming should not modify assertion stack
#[wasm_bindgen_test]
fn test_check_sat_assuming_no_side_effects() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.declare_const("q", "Bool").unwrap();
    solver.assert_formula("(or p q)").unwrap();

    // Get initial assertions
    let assertions_before = solver.get_assertions();

    // Check with assumptions
    solver.check_sat_assuming(vec!["p".to_string()]).unwrap();

    // Assertions should be unchanged
    let assertions_after = solver.get_assertions();
    assert_eq!(assertions_before, assertions_after);
}

/// Regression test: check-sat-assuming with empty array should fail
#[wasm_bindgen_test]
fn test_check_sat_assuming_empty_array() {
    let mut solver = WasmSolver::new();
    let result = solver.check_sat_assuming(vec![]);
    assert!(result.is_err());
}

/// Regression test: check-sat-assuming with empty assumption should fail
#[wasm_bindgen_test]
fn test_check_sat_assuming_empty_assumption() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();

    let result = solver.check_sat_assuming(vec!["".to_string()]);
    assert!(result.is_err());

    let result = solver.check_sat_assuming(vec!["p".to_string(), "   ".to_string()]);
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: define-sort
// ====================================================================

/// Regression test: define-sort with empty name should fail
#[wasm_bindgen_test]
fn test_define_sort_empty_name() {
    let mut solver = WasmSolver::new();
    let result = solver.define_sort("", "Int");
    assert!(result.is_err());

    let result = solver.define_sort("   ", "Int");
    assert!(result.is_err());
}

/// Regression test: define-sort with invalid base sort should fail
#[wasm_bindgen_test]
fn test_define_sort_invalid_base() {
    let mut solver = WasmSolver::new();
    let result = solver.define_sort("MySort", "InvalidSort");
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: define-fun
// ====================================================================

/// Regression test: define-fun with empty name should fail
#[wasm_bindgen_test]
fn test_define_fun_empty_name() {
    let mut solver = WasmSolver::new();
    let result = solver.define_fun("", vec![], "Int", "42");
    assert!(result.is_err());
}

/// Regression test: define-fun with empty body should fail
#[wasm_bindgen_test]
fn test_define_fun_empty_body() {
    let mut solver = WasmSolver::new();
    let result = solver.define_fun("f", vec![], "Int", "");
    assert!(result.is_err());

    let result = solver.define_fun("f", vec![], "Int", "   ");
    assert!(result.is_err());
}

/// Regression test: define-fun with invalid parameter format should fail
#[wasm_bindgen_test]
fn test_define_fun_invalid_params() {
    let mut solver = WasmSolver::new();

    // Parameter missing sort
    let result = solver.define_fun("f", vec!["x".to_string()], "Int", "x");
    assert!(result.is_err());

    // Parameter with too many parts
    let result = solver.define_fun("f", vec!["x Int Extra".to_string()], "Int", "x");
    assert!(result.is_err());

    // Parameter with invalid sort
    let result = solver.define_fun("f", vec!["x InvalidSort".to_string()], "Int", "x");
    assert!(result.is_err());
}

/// Regression test: define-fun with invalid return sort should fail
#[wasm_bindgen_test]
fn test_define_fun_invalid_return_sort() {
    let mut solver = WasmSolver::new();
    let result = solver.define_fun("f", vec![], "InvalidSort", "42");
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: validate-formula
// ====================================================================

/// Regression test: validate-formula should not modify assertion stack
#[wasm_bindgen_test]
fn test_validate_formula_no_side_effects() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.assert_formula("(> x 0)").unwrap();

    let assertions_before = solver.get_assertions();

    // Validate a formula (should not add it)
    let _ = solver.validate_formula("(< x 10)");

    let assertions_after = solver.get_assertions();
    assert_eq!(assertions_before, assertions_after);
}

/// Regression test: validate-formula with empty formula should fail
#[wasm_bindgen_test]
fn test_validate_formula_empty() {
    let mut solver = WasmSolver::new();
    let result = solver.validate_formula("");
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: simplify
// ====================================================================

/// Regression test: simplify with empty expression should fail
#[wasm_bindgen_test]
fn test_simplify_empty() {
    let mut solver = WasmSolver::new();
    let result = solver.simplify("");
    assert!(result.is_err());

    let result = solver.simplify("   ");
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: Presets
// ====================================================================

/// Regression test: apply-preset with invalid preset should fail
#[wasm_bindgen_test]
fn test_apply_preset_invalid() {
    let mut solver = WasmSolver::new();
    let result = solver.apply_preset("invalid-preset");
    assert!(result.is_err());

    let result = solver.apply_preset("");
    assert!(result.is_err());
}

/// Regression test: All valid presets should succeed
#[wasm_bindgen_test]
fn test_apply_preset_all_valid() {
    let presets = vec![
        "default",
        "fast",
        "complete",
        "debug",
        "unsat-core",
        "incremental",
    ];

    for preset in presets {
        let mut solver = WasmSolver::new();
        let result = solver.apply_preset(preset);
        assert!(result.is_ok(), "Preset '{}' should be valid", preset);
    }
}

// ====================================================================
// REGRESSION: get-info
// ====================================================================

/// Regression test: get-info with invalid key should fail
#[wasm_bindgen_test]
fn test_get_info_invalid_key() {
    let solver = WasmSolver::new();
    let result = solver.get_info("invalid-key");
    assert!(result.is_err());
}

/// Regression test: get-info should support both with and without colons
#[wasm_bindgen_test]
fn test_get_info_colon_variants() {
    let solver = WasmSolver::new();

    // With colon
    let result1 = solver.get_info(":name");
    assert!(result1.is_ok());

    // Without colon
    let result2 = solver.get_info("name");
    assert!(result2.is_ok());

    // Should return same value
    assert_eq!(result1.unwrap(), result2.unwrap());
}

// ====================================================================
// REGRESSION: declare-fun
// ====================================================================

/// Regression test: declare-fun with empty name should fail
#[wasm_bindgen_test]
fn test_declare_fun_empty_name() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_fun("", vec![], "Int");
    assert!(result.is_err());
}

/// Regression test: declare-fun with invalid return sort should fail
#[wasm_bindgen_test]
fn test_declare_fun_invalid_return_sort() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_fun("f", vec![], "InvalidSort");
    assert!(result.is_err());
}

/// Regression test: declare-fun with non-nullary args should fail (not yet supported)
#[wasm_bindgen_test]
fn test_declare_fun_non_nullary_not_supported() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_fun("f", vec!["Int".to_string()], "Int");
    assert!(result.is_err());
}

/// Regression test: declare-fun with invalid arg sorts should fail
#[wasm_bindgen_test]
fn test_declare_fun_invalid_arg_sorts() {
    let mut solver = WasmSolver::new();
    let result = solver.declare_fun("f", vec!["InvalidSort".to_string()], "Int");
    assert!(result.is_err());
}

// ====================================================================
// REGRESSION: Cancellation
// ====================================================================

/// Regression test: Cancel flag should persist across operations
#[wasm_bindgen_test]
fn test_cancel_flag_persists() {
    let mut solver = WasmSolver::new();
    assert!(!solver.is_cancelled());

    solver.cancel();
    assert!(solver.is_cancelled());

    // Should still be cancelled
    assert!(solver.is_cancelled());
}

/// Regression test: Reset should clear cancellation flag
#[wasm_bindgen_test]
fn test_reset_clears_cancel_flag() {
    let mut solver = WasmSolver::new();
    solver.cancel();
    assert!(solver.is_cancelled());

    solver.reset();
    assert!(!solver.is_cancelled());
}

// ====================================================================
// REGRESSION: Complex scenarios
// ====================================================================

/// Regression test: Multiple push/pop cycles should work correctly
#[wasm_bindgen_test]
fn test_multiple_push_pop_cycles() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();

    solver.push();
    solver.assert_formula("p").unwrap();
    assert_eq!(solver.check_sat(), "sat");
    solver.pop();

    solver.push();
    solver.assert_formula("(not p)").unwrap();
    assert_eq!(solver.check_sat(), "sat");
    solver.pop();

    // Base level should still be sat with no assertions
    assert_eq!(solver.check_sat(), "sat");
}

/// Regression test: Nested push/pop should work correctly
#[wasm_bindgen_test]
fn test_nested_push_pop() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.declare_const("q", "Bool").unwrap();

    solver.assert_formula("p").unwrap();
    solver.push();
    solver.assert_formula("q").unwrap();
    solver.push();
    solver.assert_formula("(not p)").unwrap();
    assert_eq!(solver.check_sat(), "unsat");
    solver.pop();
    assert_eq!(solver.check_sat(), "sat");
    solver.pop();
    assert_eq!(solver.check_sat(), "sat");
}

/// Regression test: Mixed incremental operations
#[wasm_bindgen_test]
fn test_mixed_incremental_operations() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_LIA");
    solver.declare_const("x", "Int").unwrap();
    solver.declare_const("y", "Int").unwrap();

    // Test 1: Basic assertions
    solver.assert_formula("(> x 0)").unwrap();
    assert_eq!(solver.check_sat(), "sat");

    // Test 2: push/check/pop
    solver.push();
    solver.assert_formula("(< x 0)").unwrap();
    assert_eq!(solver.check_sat(), "unsat");
    solver.pop();

    // Test 3: check-sat-assuming
    let result = solver
        .check_sat_assuming(vec!["(< x 0)".to_string()])
        .unwrap();
    assert_eq!(result, "unsat");

    // Test 4: Assertions should be unchanged
    let assertions = solver.get_assertions();
    assert!(assertions.contains("(> x 0)"));
    assert!(!assertions.contains("(< x 0)"));

    // Test 5: Can still check sat normally
    assert_eq!(solver.check_sat(), "sat");
}

/// Regression test: Diagnostics should provide helpful warnings
#[wasm_bindgen_test]
fn test_diagnostics_warnings() {
    let solver = WasmSolver::new();
    let warnings = solver.get_diagnostics();

    // Should have some warnings for a fresh solver
    assert!(!warnings.is_empty());

    // Should mention logic
    let has_logic_warning = warnings.iter().any(|w| w.contains("logic"));
    assert!(has_logic_warning);
}

/// Regression test: Pattern checking should provide recommendations
#[wasm_bindgen_test]
fn test_pattern_checking() {
    let solver = WasmSolver::new();

    let patterns = vec!["incremental", "assumptions", "async", "validation"];
    for pattern in patterns {
        let rec = solver.check_pattern(pattern);
        assert!(
            !rec.is_empty(),
            "Pattern '{}' should have a recommendation",
            pattern
        );
    }

    // Invalid pattern should return error message
    let rec = solver.check_pattern("invalid-pattern");
    assert!(rec.contains("Unknown pattern"));
}

// ====================================================================
// REGRESSION: Edge cases for options
// ====================================================================

/// Regression test: Getting unset option should return None
#[wasm_bindgen_test]
fn test_get_unset_option() {
    let solver = WasmSolver::new();
    let value = solver.get_option("non-existent-option");
    assert!(value.is_none());
}

/// Regression test: Setting and getting option should work
#[wasm_bindgen_test]
fn test_set_get_option() {
    let mut solver = WasmSolver::new();
    solver.set_option("produce-models", "true");

    let value = solver.get_option("produce-models");
    assert_eq!(value, Some("true".to_string()));
}

// ====================================================================
// REGRESSION: Statistics
// ====================================================================

/// Regression test: Statistics should be available even without check-sat
#[wasm_bindgen_test]
fn test_statistics_without_check_sat() {
    let solver = WasmSolver::new();
    let stats = solver.get_statistics();
    assert!(stats.is_ok());
}

/// Regression test: Statistics should reflect solver state
#[wasm_bindgen_test]
fn test_statistics_reflect_state() {
    let mut solver = WasmSolver::new();
    solver.set_logic("QF_UF");
    solver.declare_const("p", "Bool").unwrap();
    solver.assert_formula("p").unwrap();
    solver.check_sat();

    let stats = solver.get_statistics().unwrap();
    assert!(stats.is_object());
}
