//! # oxiz-c — C API for OxiZ (Phase 1)
//!
//! This crate exposes a minimal `extern "C"` ABI over the OxiZ SMT solver.
//!
//! ## Phase 1 surface (13 functions)
//!
//! | Function | Description |
//! |----------|-------------|
//! | `oxiz_version_string` | Return the crate version as a C string |
//! | `oxiz_error_string` | Map an error code to a human-readable string |
//! | `oxiz_context_new` | Allocate a solver context |
//! | `oxiz_context_free` | Free a solver context |
//! | `oxiz_solver_new` | Allocate a solver |
//! | `oxiz_solver_free` | Free a solver |
//! | `oxiz_solver_assert_smtlib2` | Parse and assert SMT-LIB2 commands |
//! | `oxiz_solver_check` | Check satisfiability |
//! | `oxiz_solver_push` | Push a backtracking scope |
//! | `oxiz_solver_pop` | Pop N backtracking scopes |
//! | `oxiz_solver_get_model_string` | Retrieve the model as an SMT-LIB2 string |
//! | `oxiz_model_string_data` | Get a pointer to model string bytes |
//! | `oxiz_model_string_free` | Free a model string |
//!
//! ## Header generation
//!
//! Set the environment variable `OXIZ_C_GEN_HEADER=1` before `cargo build`
//! to generate `include/oxiz.h` via cbindgen.

pub mod context;
pub mod error;
pub mod model;
pub mod solver;

use std::ffi::c_char;

/// NUL-terminated version string bytes.
static VERSION_BYTES: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// Return the OxiZ C API version string.
///
/// The returned pointer is valid for the lifetime of the process and does not
/// need to be freed.
#[unsafe(no_mangle)]
pub extern "C" fn oxiz_version_string() -> *const c_char {
    VERSION_BYTES.as_ptr() as *const c_char
}
