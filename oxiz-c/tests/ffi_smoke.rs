//! Smoke tests for the OxiZ C ABI.
//!
//! All functions are accessed through their exported Rust symbols — no C
//! compiler is required.  Each SMT-LIB2 declaration and its dependent
//! assertions are passed in a single `oxiz_solver_assert_smtlib2` call so that
//! the parser's internal symbol table is populated correctly.

use oxiz_c::context::{OxizContext, oxiz_context_free, oxiz_context_new};
use oxiz_c::error::oxiz_error_string;
use oxiz_c::model::{OxizModelString, oxiz_model_string_data, oxiz_model_string_free};
use oxiz_c::oxiz_version_string;
use oxiz_c::solver::{
    OxizSolver, oxiz_solver_assert_smtlib2, oxiz_solver_check, oxiz_solver_free,
    oxiz_solver_get_model_string, oxiz_solver_new, oxiz_solver_pop, oxiz_solver_push,
};
use std::ffi::CStr;

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Safety wrapper: create a context and return the raw pointer.
fn make_context() -> *mut OxizContext {
    let mut ctx: *mut OxizContext = std::ptr::null_mut();
    let rc = unsafe { oxiz_context_new(&mut ctx) };
    assert_eq!(rc, 0, "oxiz_context_new failed with {rc}");
    assert!(!ctx.is_null());
    ctx
}

/// Safety wrapper: create a solver from a context and return the raw pointer.
fn make_solver(ctx: *mut OxizContext) -> *mut OxizSolver {
    let mut solver: *mut OxizSolver = std::ptr::null_mut();
    let rc = unsafe { oxiz_solver_new(ctx, &mut solver) };
    assert_eq!(rc, 0, "oxiz_solver_new failed with {rc}");
    assert!(!solver.is_null());
    solver
}

/// Assert an SMT-LIB2 string through the C ABI and expect success.
fn assert_smt(solver: *mut OxizSolver, smt: &str) {
    let mut bytes = smt.as_bytes().to_vec();
    bytes.push(0u8);
    let rc = unsafe { oxiz_solver_assert_smtlib2(solver, bytes.as_ptr() as *const _) };
    assert_eq!(
        rc, 0,
        "oxiz_solver_assert_smtlib2 failed with {rc} for: {smt}"
    );
}

/// Call check-sat through the C ABI and return the raw sat_out value.
fn check(solver: *mut OxizSolver) -> std::ffi::c_int {
    let mut sat_out: std::ffi::c_int = -1;
    let rc = unsafe { oxiz_solver_check(solver, &mut sat_out) };
    assert_eq!(rc, 0, "oxiz_solver_check failed with {rc}");
    sat_out
}

// ─── tests ────────────────────────────────────────────────────────────────────

/// Verify that version and error-code strings are non-null and sensible.
#[test]
fn test_version_and_error_strings() {
    // oxiz_version_string is safe (no unsafe extern), no block needed.
    let ver_ptr = oxiz_version_string();
    assert!(!ver_ptr.is_null(), "oxiz_version_string returned NULL");
    let ver = unsafe { CStr::from_ptr(ver_ptr) }.to_str().unwrap_or("");
    assert!(!ver.is_empty(), "version string is empty");

    // Error code 0 → "Ok"
    let ok_ptr = oxiz_error_string(0);
    assert!(!ok_ptr.is_null());
    let ok_str = unsafe { CStr::from_ptr(ok_ptr) }.to_str().unwrap_or("");
    assert_eq!(ok_str, "Ok");

    // Error code 3 → "ParseError"
    let pe_ptr = oxiz_error_string(3);
    let pe_str = unsafe { CStr::from_ptr(pe_ptr) }.to_str().unwrap_or("");
    assert_eq!(pe_str, "ParseError");

    // Out-of-range code → "Unknown"
    let unk_ptr = oxiz_error_string(99);
    let unk_str = unsafe { CStr::from_ptr(unk_ptr) }.to_str().unwrap_or("");
    assert_eq!(unk_str, "Unknown");
}

/// Verify context allocation and deallocation.
#[test]
fn test_context_lifecycle() {
    let ctx = make_context();
    unsafe { oxiz_context_free(ctx) };
    // NULL-free is documented as a no-op.
    unsafe { oxiz_context_free(std::ptr::null_mut()) };
}

/// Verify null-pointer guard on context_new.
#[test]
fn test_context_new_null_guard() {
    let rc = unsafe { oxiz_context_new(std::ptr::null_mut()) };
    assert_eq!(rc, 1, "expected NullPointer(1), got {rc}");
}

/// Full lifecycle: declare, assert two constraints, check → SAT.
///
/// All commands are bundled in one call so the parser's variable table
/// is populated when `(assert ...)` is processed.
#[test]
fn test_solver_assert_check_sat() {
    let ctx = make_context();
    let solver = make_solver(ctx);

    // All commands in a single call to ensure the parser sees the declaration.
    assert_smt(
        solver,
        "(declare-const x Int) (assert (> x 0)) (assert (< x 5))",
    );

    let sat_out = check(solver);
    assert_eq!(sat_out, 1, "expected SAT(1), got {sat_out}");

    unsafe { oxiz_solver_free(solver) };
    unsafe { oxiz_context_free(ctx) };
}

/// Push/pop unsat-core recovery.
///
/// 1. Declare a Bool constant `y` (SAT by itself).
/// 2. Push; assert `y` AND `(not y)` → UNSAT.
/// 3. Pop; check again → SAT (constraint reverted).
#[test]
fn test_push_pop_unsat_recovery() {
    let ctx = make_context();
    let solver = make_solver(ctx);

    // Baseline: (declare-const y Bool) is satisfiable.
    assert_smt(solver, "(declare-const y Bool)");
    assert_eq!(check(solver), 1, "baseline should be SAT");

    // Push and add contradictory constraints.
    let rc = unsafe { oxiz_solver_push(solver) };
    assert_eq!(rc, 0);
    assert_smt(solver, "(assert y) (assert (not y))");
    assert_eq!(check(solver), 0, "contradictory level should be UNSAT");

    // Pop — constraints from the pushed level are discarded.
    let rc = unsafe { oxiz_solver_pop(solver, 1) };
    assert_eq!(rc, 0);
    assert_eq!(check(solver), 1, "after pop should be SAT again");

    unsafe { oxiz_solver_free(solver) };
    unsafe { oxiz_context_free(ctx) };
}

/// Retrieve a model string after a SAT result.
#[test]
fn test_get_model_string() {
    let ctx = make_context();
    let solver = make_solver(ctx);

    assert_smt(solver, "(declare-const z Bool) (assert z)");
    assert_eq!(check(solver), 1, "should be SAT");

    let mut ms: *mut OxizModelString = std::ptr::null_mut();
    let rc = unsafe { oxiz_solver_get_model_string(solver, &mut ms) };
    assert_eq!(rc, 0, "get_model_string failed with {rc}");
    assert!(!ms.is_null(), "model string pointer is NULL");

    let data_ptr = unsafe { oxiz_model_string_data(ms) };
    assert!(!data_ptr.is_null(), "model data pointer is NULL");
    let model_text = unsafe { CStr::from_ptr(data_ptr) }.to_str().unwrap_or("");
    assert!(
        model_text.starts_with("(model"),
        "expected SMT-LIB2 model, got: {model_text}"
    );

    unsafe { oxiz_model_string_free(ms) };
    // NULL free is a no-op.
    unsafe { oxiz_model_string_free(std::ptr::null_mut()) };

    unsafe { oxiz_solver_free(solver) };
    unsafe { oxiz_context_free(ctx) };
}
