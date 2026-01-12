//! Fuzz target for the execute() method
//!
//! This fuzzer tests the execute method with arbitrary SMT-LIB2 scripts
//! to find crashes, panics, or other unexpected behavior.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxiz_wasm::WasmSolver;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, allowing invalid UTF-8
    if let Ok(script) = std::str::from_utf8(data) {
        let mut solver = WasmSolver::new();

        // Execute the script - we don't care about the result,
        // we just want to make sure it doesn't crash or panic
        let _ = solver.execute(script);
    }
});
