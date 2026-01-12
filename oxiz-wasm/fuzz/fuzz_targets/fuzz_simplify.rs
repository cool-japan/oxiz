//! Fuzz target for the simplify() method
//!
//! This fuzzer tests the simplify method with arbitrary expressions
//! to find crashes, panics, or other unexpected behavior.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxiz_wasm::WasmSolver;

fuzz_target!(|data: &[u8]| {
    if let Ok(expr) = std::str::from_utf8(data) {
        let mut solver = WasmSolver::new();
        solver.set_logic("ALL");

        // Try to declare some variables that might be referenced
        let _ = solver.declare_const("x", "Int");
        let _ = solver.declare_const("y", "Int");
        let _ = solver.declare_const("p", "Bool");
        let _ = solver.declare_const("q", "Bool");
        let _ = solver.declare_const("a", "Real");
        let _ = solver.declare_const("b", "Real");

        // Simplify the expression - should not crash
        let _ = solver.simplify(expr);
    }
});
