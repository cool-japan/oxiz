//! Fuzz target for the assertFormula() method
//!
//! This fuzzer tests the assertFormula method with arbitrary formulas
//! to find crashes, panics, or other unexpected behavior.

#![no_main]

use libfuzzer_sys::fuzz_target;
use oxiz_wasm::WasmSolver;

fuzz_target!(|data: &[u8]| {
    if let Ok(formula) = std::str::from_utf8(data) {
        let mut solver = WasmSolver::new();
        solver.set_logic("ALL");

        // Try to declare some variables that might be referenced
        let _ = solver.declare_const("x", "Int");
        let _ = solver.declare_const("y", "Int");
        let _ = solver.declare_const("z", "Int");
        let _ = solver.declare_const("p", "Bool");
        let _ = solver.declare_const("q", "Bool");
        let _ = solver.declare_const("r", "Bool");
        let _ = solver.declare_const("a", "Real");
        let _ = solver.declare_const("b", "Real");
        let _ = solver.declare_const("bv1", "BitVec32");
        let _ = solver.declare_const("bv2", "BitVec32");

        // Assert the formula - should not crash
        let _ = solver.assert_formula(formula);

        // Also try to check-sat - should not crash
        let _ = solver.check_sat();
    }
});
