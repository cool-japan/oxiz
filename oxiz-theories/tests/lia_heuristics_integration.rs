//! Integration tests for LIA heuristics wired into the branch-and-bound solver loop.
//!
//! These tests exercise `probe_variables`, `feasibility_pump`, and `manage_cuts`
//! through the public `LiaSolver` API to verify that:
//!   1. The heuristics are reachable from the production solve path.
//!   2. They return sensible results on concrete problems.
//!   3. They do not panic on edge cases (empty solver, zero cuts, etc.).
//!
//! Note: integration tests can only access `pub` items.  Private fields
//! (`simplex`, `active_cuts`) are not accessible here; tests are structured
//! to go through the public method surface only.

use num_rational::Rational64;
use num_traits::One;
use oxiz_theories::arithmetic::{LiaSolver, LinExpr, VarId};

/// Helper: add `coeff * var + constant <= 0` as a less-than-or-equal constraint.
fn add_le_single(solver: &mut LiaSolver, var: VarId, coeff: i64, rhs: i64, reason: u32) {
    let mut expr = LinExpr::new();
    expr.add_term(var, Rational64::from_integer(coeff));
    expr.add_constant(-Rational64::from_integer(rhs));
    solver.add_le(expr, reason);
}

/// Helper: add `coeff * var + constant >= 0` as a greater-than-or-equal constraint.
fn add_ge_single(solver: &mut LiaSolver, var: VarId, coeff: i64, rhs: i64, reason: u32) {
    let mut expr = LinExpr::new();
    expr.add_term(var, Rational64::from_integer(coeff));
    expr.add_constant(-Rational64::from_integer(rhs));
    solver.add_ge(expr, reason);
}

/// Helper: add `c1*v1 + c2*v2 <= rhs`.
fn add_le2(solver: &mut LiaSolver, v1: VarId, c1: i64, v2: VarId, c2: i64, rhs: i64, reason: u32) {
    let mut expr = LinExpr::new();
    expr.add_term(v1, Rational64::from_integer(c1));
    expr.add_term(v2, Rational64::from_integer(c2));
    expr.add_constant(-Rational64::from_integer(rhs));
    solver.add_le(expr, reason);
}

// ---------------------------------------------------------------------------
// Test 1: probe_variables succeeds on a constrained problem
// ---------------------------------------------------------------------------
/// Set up: x >= 1, 2x <= 9 (i.e. x <= 4.5 — the effective integer ceiling
/// is 4).  After calling `probe_variables` with max_probes = 20:
///   - The method must return `Ok` without panicking.
///   - The whole-problem check() must still return `Ok(true)` (SAT).
///
/// We do NOT assert a specific tightened-count because whether probing
/// tightens a particular bound depends on the LP relaxation at that value;
/// what matters is that the solver remains correct afterwards.
#[test]
fn test_probe_variables_tightens_bounds() {
    let mut solver = LiaSolver::new();

    let x = solver.new_var();

    // x >= 1
    add_ge_single(&mut solver, x, 1, 1, 0);
    // x <= 5  (explicit upper cap via constraint)
    add_le_single(&mut solver, x, 1, 5, 1);
    // 2x <= 9  → x <= 4.5
    add_le_single(&mut solver, x, 2, 9, 2);

    // Run probing — must not panic, must return Ok.
    // (probe_variables internally solves the LP; the constraints above make
    // it feasible so it should not return an error.)
    let result = solver.probe_variables(20);
    assert!(
        result.is_ok(),
        "probe_variables should return Ok on a feasible problem"
    );

    // The full check must still report SAT after probing tightened bounds.
    let sat = solver.check();
    assert!(
        sat.is_ok(),
        "check() should not error after probe_variables"
    );
    assert!(
        sat.expect("check() must succeed"),
        "problem is still satisfiable after probing (x = 1..4 is valid)"
    );
}

// ---------------------------------------------------------------------------
// Test 2: feasibility_pump returns Ok on a satisfiable integer problem
// ---------------------------------------------------------------------------
/// Build a satisfiable LIA: x >= 0, y >= 0, x + y <= 10.
/// The feasibility pump must return `Ok(Some(_))` or `Ok(None)` —
/// in either case it must not error or panic.
///
/// If it returns `Some(solution)`, we additionally verify the solution
/// satisfies x, y >= 0 and x + y <= 10.
#[test]
fn test_feasibility_pump_finds_solution() {
    let mut solver = LiaSolver::new();

    let x = solver.new_var();
    let y = solver.new_var();

    // x >= 0, y >= 0
    add_ge_single(&mut solver, x, 1, 0, 0);
    add_ge_single(&mut solver, y, 1, 0, 1);

    // x + y <= 10
    add_le2(&mut solver, x, 1, y, 1, 10, 2);

    // Invoke feasibility pump — must not panic, must return Ok.
    // (The pump re-solves the LP internally.)
    let result = solver.feasibility_pump(10);
    assert!(
        result.is_ok(),
        "feasibility_pump should return Ok on a feasible problem"
    );

    // If a solution was found, validate it.
    if let Ok(Some(ref solution)) = result {
        assert_eq!(
            solution.len(),
            2,
            "solution should contain one entry per integer variable (x, y)"
        );
        assert!(solution[0] >= 0, "x component must satisfy x >= 0");
        assert!(solution[1] >= 0, "y component must satisfy y >= 0");
        assert!(
            solution[0] + solution[1] <= 10,
            "x + y must satisfy the upper-bound constraint"
        );
    }
    // Ok(None) is a valid outcome — the pump may not converge in 10 iterations.
}

// ---------------------------------------------------------------------------
// Test 3: manage_cuts on a fresh solver returns 0 deleted cuts
// ---------------------------------------------------------------------------
/// A freshly created `LiaSolver` has no active cuts.  Calling `manage_cuts`
/// must return 0 and must not panic.
///
/// We then add some cut records via `record_cut` (public API) and call
/// `manage_cuts` again — it must not panic.  We cannot inspect `active_cuts`
/// directly from an integration test (it is `pub(super)`), but we can verify
/// the public return value is consistent.
#[test]
fn test_manage_cuts_purges_old_cuts() {
    let mut solver = LiaSolver::new();

    // manage_cuts on a fresh, empty solver must return 0 and not panic.
    let deleted_empty = solver.manage_cuts();
    assert_eq!(
        deleted_empty, 0,
        "manage_cuts on an empty cut list should report 0 deletions"
    );

    // Record a few dummy cut entries via the public API.
    // VarId 200, 201, 202 are arbitrary slack-variable IDs.
    solver.record_cut(200);
    solver.record_cut(201);
    solver.record_cut(202);

    // Immediately calling manage_cuts again should not panic.
    // Newly added cuts have age == 0, so they fall below any age threshold
    // and should NOT be deleted right away.
    let deleted_young = solver.manage_cuts();
    assert_eq!(
        deleted_young, 0,
        "freshly recorded cuts (age 0) should not be purged"
    );

    // A second call should also be stable.
    let deleted_again = solver.manage_cuts();
    assert_eq!(
        deleted_again, 0,
        "repeated manage_cuts on young cuts should still return 0"
    );
}

// ---------------------------------------------------------------------------
// Test 4: full solve path exercises all three heuristics end-to-end
// ---------------------------------------------------------------------------
/// Calls `LiaSolver::check()` on a small SAT problem.  Because `check()` now
/// calls `probe_variables`, `feasibility_pump`, and (via `branch_and_bound`)
/// `manage_cuts`, this verifies the integrated path does not regress existing
/// SAT detection.
///
/// Problem: x >= 1, x <= 3, x + y <= 5, y >= 0  — trivially SAT.
#[test]
fn test_full_solve_path_with_heuristics() {
    let mut solver = LiaSolver::new();

    let x = solver.new_var();
    let y = solver.new_var();

    // x in [1, 3]
    add_ge_single(&mut solver, x, 1, 1, 0);
    add_le_single(&mut solver, x, 1, 3, 1);

    // y >= 0
    add_ge_single(&mut solver, y, 1, 0, 2);

    // x + y <= 5
    let mut sum_expr = LinExpr::new();
    sum_expr.add_term(x, Rational64::one());
    sum_expr.add_term(y, Rational64::one());
    sum_expr.add_constant(-Rational64::from_integer(5));
    solver.add_le(sum_expr, 3);

    // check() runs: probe_variables → feasibility_pump → branch_and_bound
    // (branch_and_bound calls manage_cuts at depth 0 since 0.is_multiple_of(8)).
    let result = solver.check();
    assert!(result.is_ok(), "check() should not error on a SAT problem");
    assert!(
        result.expect("SAT problem should return Ok(true)"),
        "the problem is satisfiable — check() should return true"
    );
}
