//! Solver handle for the OxiZ C API.
//!
//! `OxizSolver` wraps `oxiz_solver::Context`, which bundles the term manager,
//! the CDCL(T) solver, and the SMT-LIB2 execute-script engine.

use crate::context::OxizContext;
use crate::error::OxizError;
use crate::model::OxizModelString;
use oxiz_solver::{Context, SolverResult};
use std::ffi::{CStr, c_int};

/// Opaque solver handle.
///
/// Created with `oxiz_solver_new`; freed with `oxiz_solver_free`.
pub struct OxizSolver {
    ctx: Context,
}

impl OxizSolver {
    fn new() -> Self {
        Self {
            ctx: Context::new(),
        }
    }
}

/// Allocate a new OxiZ solver.
///
/// The solver is independent of `ctx`; the context parameter is reserved for
/// future Phase-2 options.  It may not be NULL, but its internal state is not
/// currently read.
///
/// # Safety
///
/// `ctx` must be a valid (non-NULL, non-freed) `OxizContext` pointer.
/// `solver_out` must point to a valid writable `*mut OxizSolver` location.
/// The returned solver must eventually be freed with `oxiz_solver_free`.
///
/// # Returns
///
/// `0` on success, non-zero on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_solver_new(
    ctx: *const OxizContext,
    solver_out: *mut *mut OxizSolver,
) -> c_int {
    if ctx.is_null() || solver_out.is_null() {
        return OxizError::NullPointer.as_c_int();
    }
    let solver = Box::new(OxizSolver::new());
    unsafe { *solver_out = Box::into_raw(solver) };
    OxizError::Ok.as_c_int()
}

/// Free an OxiZ solver created by `oxiz_solver_new`.
///
/// Passing a NULL pointer is a no-op.
///
/// # Safety
///
/// `solver` must be a pointer returned by `oxiz_solver_new` that has not yet
/// been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_solver_free(solver: *mut OxizSolver) {
    if solver.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(solver)) };
}

/// Parse and execute an SMT-LIB2 command string, asserting any resulting
/// constraints into the solver.
///
/// The string may contain one or more SMT-LIB2 commands such as
/// `(set-logic ...)`, `(declare-const ...)`, or `(assert ...)`.
/// `(check-sat)` is silently accepted but the result is not returned here —
/// use `oxiz_solver_check` for that.
///
/// Multiple calls accumulate assertions.  Because the parser's variable table
/// is rebuilt for each call, declarations and their dependent assertions should
/// be provided in the same call string so the parser can resolve all names.
///
/// # Safety
///
/// `solver` must be a valid (non-NULL, non-freed) solver handle.
/// `smt_cstr` must be a valid NUL-terminated C string.
///
/// # Returns
///
/// `0` on success; `OxizError::NullPointer` if any pointer is NULL;
/// `OxizError::InvalidUtf8` if the string is not valid UTF-8;
/// `OxizError::ParseError` if parsing fails;
/// `OxizError::SolverError` for other solver errors.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_solver_assert_smtlib2(
    solver: *mut OxizSolver,
    smt_cstr: *const std::ffi::c_char,
) -> c_int {
    if solver.is_null() || smt_cstr.is_null() {
        return OxizError::NullPointer.as_c_int();
    }
    let s = match unsafe { CStr::from_ptr(smt_cstr) }.to_str() {
        Ok(s) => s,
        Err(_) => return OxizError::InvalidUtf8.as_c_int(),
    };
    let solver_ref = unsafe { &mut *solver };
    match solver_ref.ctx.execute_script(s) {
        Ok(_) => OxizError::Ok.as_c_int(),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("parse") || msg.contains("Parse") || msg.contains("syntax") {
                OxizError::ParseError.as_c_int()
            } else {
                OxizError::SolverError.as_c_int()
            }
        }
    }
}

/// Check satisfiability of the current set of assertions.
///
/// On success, `sat_out` is written with:
/// - `0` — UNSAT
/// - `1` — SAT
/// - `2` — UNKNOWN
///
/// # Safety
///
/// `solver` must be a valid (non-NULL, non-freed) solver handle.
/// `sat_out` must point to a valid writable `c_int` location.
///
/// # Returns
///
/// `0` on success (result written to `*sat_out`), or an `OxizError` code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_solver_check(solver: *mut OxizSolver, sat_out: *mut c_int) -> c_int {
    if solver.is_null() || sat_out.is_null() {
        return OxizError::NullPointer.as_c_int();
    }
    let solver_ref = unsafe { &mut *solver };
    let result = solver_ref.ctx.check_sat();
    let code: c_int = match result {
        SolverResult::Unsat => 0,
        SolverResult::Sat => 1,
        SolverResult::Unknown => 2,
    };
    unsafe { *sat_out = code };
    OxizError::Ok.as_c_int()
}

/// Push a backtracking scope onto the solver stack.
///
/// # Safety
///
/// `solver` must be a valid (non-NULL, non-freed) solver handle.
///
/// # Returns
///
/// `0` on success, or an `OxizError` code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_solver_push(solver: *mut OxizSolver) -> c_int {
    if solver.is_null() {
        return OxizError::NullPointer.as_c_int();
    }
    let solver_ref = unsafe { &mut *solver };
    solver_ref.ctx.push();
    OxizError::Ok.as_c_int()
}

/// Pop `n` backtracking scopes from the solver stack.
///
/// Popping more levels than have been pushed is silently clamped to zero.
///
/// # Safety
///
/// `solver` must be a valid (non-NULL, non-freed) solver handle.
///
/// # Returns
///
/// `0` on success, or an `OxizError` code.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_solver_pop(solver: *mut OxizSolver, n: u32) -> c_int {
    if solver.is_null() {
        return OxizError::NullPointer.as_c_int();
    }
    let solver_ref = unsafe { &mut *solver };
    for _ in 0..n {
        solver_ref.ctx.pop();
    }
    OxizError::Ok.as_c_int()
}

/// Retrieve the current model as an SMT-LIB2-formatted string.
///
/// Must be called after a successful `oxiz_solver_check` that returned SAT
/// (`*sat_out == 1`).  If no model is available, `*out` is set to NULL and
/// `OxizError::SolverError` is returned.
///
/// The returned `OxizModelString` must be freed with `oxiz_model_string_free`.
///
/// # Safety
///
/// `solver` must be a valid (non-NULL, non-freed) solver handle.
/// `out` must point to a valid writable `*mut OxizModelString` location.
///
/// # Returns
///
/// `0` on success; `OxizError::NullPointer` if any pointer is NULL;
/// `OxizError::SolverError` if the model is not available.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_solver_get_model_string(
    solver: *mut OxizSolver,
    out: *mut *mut OxizModelString,
) -> c_int {
    if solver.is_null() || out.is_null() {
        return OxizError::NullPointer.as_c_int();
    }
    let solver_ref = unsafe { &*solver };
    let model_str = solver_ref.ctx.format_model();
    if model_str.starts_with("(error") {
        unsafe { *out = std::ptr::null_mut() };
        return OxizError::SolverError.as_c_int();
    }
    let ms = OxizModelString::from_string(model_str);
    unsafe { *out = Box::into_raw(Box::new(ms)) };
    OxizError::Ok.as_c_int()
}
