//! Smoke tests: verify fixture benchmarks execute without panics.
//!
//! Each test instantiates a fresh `Context`, feeds the embedded SMT-LIB2
//! fixture through `execute_script`, and asserts no panic occurs.  The
//! actual solver result (sat / unsat / unknown / error) is intentionally
//! ignored because the goal is purely to confirm that the fixture can be
//! parsed and dispatched without crashing the solver pipeline.

use oxiz_solver::Context;

/// Feed `script` into a fresh context.  No panic == pass.
fn run_fixture(script: &str) {
    let mut ctx = Context::new();
    let _ = ctx.execute_script(script);
}

#[test]
fn test_bv_fixture_runs() {
    run_fixture(bench_regression::fixtures::BV_SIMPLE);
}

#[test]
fn test_lra_fixture_runs() {
    run_fixture(bench_regression::fixtures::LRA_SIMPLE);
}

#[test]
fn test_arrays_fixture_runs() {
    run_fixture(bench_regression::fixtures::ARRAYS_SIMPLE);
}
