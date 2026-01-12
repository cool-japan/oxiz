//! Fuzz target for the declareConst() method
//!
//! This fuzzer tests the declareConst method with arbitrary names and sorts
//! to find crashes, panics, or other unexpected behavior.

#![no_main]

use libfuzzer_sys::{fuzz_target, arbitrary::{Arbitrary, Unstructured}};
use oxiz_wasm::WasmSolver;

#[derive(Debug)]
struct FuzzInput<'a> {
    name: &'a str,
    sort: &'a str,
}

impl<'a> Arbitrary<'a> for FuzzInput<'a> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let name = u.arbitrary()?;
        let sort = u.arbitrary()?;
        Ok(FuzzInput { name, sort })
    }
}

fuzz_target!(|input: FuzzInput| {
    let mut solver = WasmSolver::new();

    // Try to declare a constant with arbitrary name and sort
    // Should not crash regardless of input
    let _ = solver.declare_const(input.name, input.sort);
});
