//! Opaque context handle for the OxiZ C API.
//!
//! `OxizContext` is a lightweight configuration container.  All real solver
//! state lives in `OxizSolver` (which wraps `oxiz_solver::Context`).  Callers
//! create a context once and may create multiple solvers from it.

use crate::error::OxizError;
use std::ffi::c_int;

/// Opaque context handle.
///
/// Created with `oxiz_context_new`; freed with `oxiz_context_free`.
/// One context may be used with multiple independent solvers.
pub struct OxizContext {
    // Currently stores no configuration — reserved for Phase 2 options.
    _private: (),
}

impl OxizContext {
    fn new() -> Self {
        Self { _private: () }
    }
}

/// Allocate a new OxiZ context.
///
/// # Safety
///
/// `ctx_out` must point to a valid writable `*mut OxizContext` location.
/// The returned handle must eventually be freed with `oxiz_context_free`.
///
/// # Returns
///
/// `0` on success, non-zero on error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_context_new(ctx_out: *mut *mut OxizContext) -> c_int {
    if ctx_out.is_null() {
        return OxizError::NullPointer.as_c_int();
    }
    let ctx = Box::new(OxizContext::new());
    unsafe { *ctx_out = Box::into_raw(ctx) };
    OxizError::Ok.as_c_int()
}

/// Free an OxiZ context created by `oxiz_context_new`.
///
/// Passing a NULL pointer is a no-op.
///
/// # Safety
///
/// `ctx` must be a pointer returned by `oxiz_context_new` that has not yet
/// been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_context_free(ctx: *mut OxizContext) {
    if ctx.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(ctx)) };
}
