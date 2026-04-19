//! C-ABI error codes for OxiZ.

use std::ffi::c_int;

/// OxiZ C API error codes.
///
/// Every fallible function returns one of these as a `c_int`.  Zero means
/// success; all other values are failure modes.
#[repr(C)]
pub enum OxizError {
    /// Success — no error.
    Ok = 0,
    /// A required pointer argument was NULL.
    NullPointer = 1,
    /// A C string argument contained invalid UTF-8.
    InvalidUtf8 = 2,
    /// The SMT-LIB2 input could not be parsed.
    ParseError = 3,
    /// An internal solver error occurred.
    SolverError = 4,
    /// The requested operation is not implemented in Phase 1.
    Unimplemented = 5,
    /// Memory allocation failed.
    OutOfMemory = 6,
}

impl OxizError {
    /// Convert to a `c_int` suitable for returning from `extern "C"` functions.
    pub fn as_c_int(self) -> c_int {
        self as c_int
    }
}

/// Return a static human-readable string for an OxiZ error code.
///
/// The returned pointer is valid for the lifetime of the process and does not
/// need to be freed.
#[unsafe(no_mangle)]
pub extern "C" fn oxiz_error_string(err: c_int) -> *const std::ffi::c_char {
    let s: &[u8] = match err {
        0 => b"Ok\0",
        1 => b"NullPointer\0",
        2 => b"InvalidUtf8\0",
        3 => b"ParseError\0",
        4 => b"SolverError\0",
        5 => b"Unimplemented\0",
        6 => b"OutOfMemory\0",
        _ => b"Unknown\0",
    };
    s.as_ptr() as *const std::ffi::c_char
}
