//! Model string handle for the OxiZ C API.

/// An owned, heap-allocated model string returned by `oxiz_solver_get_model_string`.
///
/// The string is NUL-terminated and encoded as UTF-8.  Its contents follow the
/// SMT-LIB2 `(model ...)` format.
pub struct OxizModelString {
    // NUL-terminated bytes.
    bytes: Box<[u8]>,
}

impl OxizModelString {
    /// Construct from a Rust `String`, appending a NUL terminator.
    pub(crate) fn from_string(s: String) -> Self {
        let mut b = s.into_bytes();
        b.push(0u8);
        Self {
            bytes: b.into_boxed_slice(),
        }
    }

    /// Return a raw pointer to the NUL-terminated byte sequence.
    pub(crate) fn as_ptr(&self) -> *const std::ffi::c_char {
        self.bytes.as_ptr() as *const std::ffi::c_char
    }
}

/// Return a pointer to the NUL-terminated model string data.
///
/// The pointer is valid until `oxiz_model_string_free` is called on `ms`.
/// Do not attempt to free the returned pointer directly.
///
/// Passing a NULL `ms` returns NULL.
///
/// # Safety
///
/// `ms` must be a valid (non-freed) `OxizModelString` pointer, or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_model_string_data(
    ms: *const OxizModelString,
) -> *const std::ffi::c_char {
    if ms.is_null() {
        return std::ptr::null();
    }
    unsafe { (*ms).as_ptr() }
}

/// Free an `OxizModelString` returned by `oxiz_solver_get_model_string`.
///
/// Passing a NULL pointer is a no-op.
///
/// # Safety
///
/// `ms` must be a pointer returned by `oxiz_solver_get_model_string` that has
/// not yet been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oxiz_model_string_free(ms: *mut OxizModelString) {
    if ms.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(ms)) };
}
