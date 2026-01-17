//! Fuzz target for the SMT-LIB2 parser
//!
//! This fuzzer tests the parser with arbitrary byte sequences to find
//! crashes, panics, or other unexpected behavior.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxiz_core::ast::TermManager;
use oxiz_core::smtlib::parse_script;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, allowing invalid UTF-8 to be filtered
    if let Ok(input) = std::str::from_utf8(data) {
        let mut tm = TermManager::new();

        // Parse the script - we don't care about the result,
        // we just want to make sure it doesn't crash or panic
        let _ = parse_script(input, &mut tm);
    }
});
