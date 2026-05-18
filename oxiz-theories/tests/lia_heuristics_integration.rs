//! Integration tests for LIA heuristics wired into the branch-and-bound solver loop.
//!
//! These tests exercise `probe_variables`, `feasibility_pump`, and `manage_cuts`
//! through the public `LiaSolver` API to verify that:
//!   1. The heuristics are reachable from the production solve path.
//!   2. They return sensible results on concrete problems.
//!   3. They do not panic on edge cases (empty solver, zero cuts, etc.).

use num_rational::Rational64;
use num_traits::One;
use oxiz_theories::arithmetic::{LiaSolver, LinExpr};

/// Helper: build a `LinExpr` with a single term `coeff * var` and a constant.
fn make_expr(var: usize, coeff: i64, constant: i64) -> LinExpr {
    let mut expr = LinExpr::new();
    expr.add_term(var, Rational64::from_integer(coeff));
    expr.add_constant(-Rational64::from_integer(constant));
    expr
}

/// Helper: build a `LinExpr` with two terms `c1*v1 + c2*v2` and a constant.
fn make_expr2(v1: usize, c1: i64, v2: usize, c2: i64, constant: i64) -> LinExpr {
    let mut expr = LinExpr::new();
    expr.add_term(v1, Rational64::from_integer(c1));
    expr.add_term(v2, Rational64::from_integer(c2));
    expr.add_constant(-Rational64::from_integer(constant));
    expr
}

// ---------------------------------------------------------------------------
// Test 1: probe_variables tightens bounds on a constrained problem
// ---------------------------------------------------------------------------
/// Set up: x >= 1, x <= 5, 2x <= 9  (so the effective ceiling for 2x <= 9
/// means x <= 4.5, i.e. x <= 4 for integers).
///
/// After calling `probe_variables` with max_probes = 20:
///   - The LP must still be feasible (x in [1, 4] is non-empty).
///   - The method must return Ok without panicking.
///
/// We do NOT assert that the returned count is non-zero because whether
/// probing tightens a bound depends on whether it detects an infeasibility
/// at the extreme fixed value — our validation is correctness and
/// non-regression, not a specific numeric count.
#[test]
fn test_probe_variables_tightens_bounds() {
    let mut solver = LiaSolver::new();

    let x = solver.new_var();

    // x >= 1
    solver.simplex.set_lower(x, Rational64::from_integer(1), 0);
    // x <= 5
    solver.simplex.set_upper(x, Rational64::from_integer(5), 1);

    // 2x <= 9  → x <= 4.5 (the binding upper constraint for an integer)
    solver.add_le(make_expr(x, 2, 9), 2);

    // The LP must be feasible before probing
    assert!(
        solver.simplex.check().is_ok(),
        "LP should be feasible before probing"
    );

    // Run probing — must not panic, must return Ok
    let result = solver.probe_variables(20);
    assert!(
        result.is_ok(),
        "probe_variables should succeed on a feasible problem"
    );

    // LP must remain feasible after probing (bounds may be tighter, but x = 1..4 is fine)
    let lp_after = solver.simplex.check();
    assert!(
        lp_after.is_ok(),
        "LP should still be feasible after probing; solver state should be consistent"
    );
}

// ---------------------------------------------------------------------------
// Test 2: feasibility_pump returns Ok on a satisfiable integer problem
// ---------------------------------------------------------------------------
/// Build a simple satisfiable LIA: x >= 0, y >= 0, x + y <= 10.
/// The feasibility pump should return `Ok(Some(_))` or `Ok(None)` —
/// in either case it must not error or panic.
///
/// If it returns `Some(solution)`, we additionally verify the solution is
/// integer-valued (all values representable as i64) and satisfies x, y >= 0.
#[test]
fn test_feasibility_pump_finds_solution() {
    let mut solver = LiaSolver::new();

    let x = solver.new_var();
    let y = solver.new_var();

    // x >= 0, y >= 0
    solver.simplex.set_lower(x, Rational64::from_integer(0), 0);
    solver.simplex.set_lower(y, Rational64::from_integer(0), 1);

    // x + y <= 10
    solver.add_le(make_expr2(x, 1, y, 1, 10), 2);

    // Run the LP so the simplex has a basis for the pump to work from
    assert!(
        solver.simplex.check().is_ok(),
        "LP should be feasible before feasibility pump"
    );

    // Invoke feasibility pump — must not panic, must return Ok
    let result = solver.feasibility_pump(10);
    assert!(
        result.is_ok(),
        "feasibility_pump should return Ok on a feasible problem"
    );

    // If a solution was found, validate it
    if let Ok(Some(ref solution)) = result {
        assert_eq!(
            solution.len(),
            2,
            "solution should have one entry per integer variable"
        );
        assert!(solution[0] >= 0, "x solution must satisfy x >= 0");
        assert!(solution[1] >= 0, "y solution must satisfy y >= 0");
        assert!(
            solution[0] + solution[1] <= 10,
            "x + y must satisfy the sum constraint"
        );
    }
    // Ok(None) is also a valid outcome — pump may not converge in 10 iterations
}

// ---------------------------------------------------------------------------
// Test 3: manage_cuts on a fresh solver returns 0 deleted cuts
// ---------------------------------------------------------------------------
/// A freshly created `LiaSolver` has no active cuts.  Calling `manage_cuts`
/// must return 0 and must not panic.
#[test]
fn test_manage_cuts_purges_old_cuts() {
    let mut solver = LiaSolver::new();

    // Sanity: no cuts initially
    assert_eq!(
        solver.active_cuts.len(),
        0,
        "fresh solver should have no active cuts"
    );

    // manage_cuts on an empty cut list must return 0 and not panic
    let deleted = solver.manage_cuts();
    assert_eq!(
        deleted, 0,
        "manage_cuts on an empty list should delete 0 cuts"
    );

    // Additional case: add some cut records, age them past both thresholds,
    // then verify manage_cuts deletes them all.
    solver.record_cut(200);
    solver.record_cut(201);
    solver.record_cut(202);

    assert_eq!(
        solver.active_cuts.len(),
        3,
        "after recording 3 cuts the list should have length 3"
    );

    // Age cuts beyond HIGH_AGE_THRESHOLD (1000) so they are unconditionally deleted
    for cut in &mut solver.active_cuts {
        cut.age = 1001;
    }

    let deleted_old = solver.manage_cuts();
    assert_eq!(
        deleted_old, 3,
        "all 3 over-age cuts should be deleted by manage_cuts"
    );
    assert_eq!(
        solver.active_cuts.len(),
        0,
        "active_cuts should be empty after manage_cuts purges everything"
    );
}

// ---------------------------------------------------------------------------
// Test 4: full solve path exercises all three heuristics end-to-end
// ---------------------------------------------------------------------------
/// This test calls `LiaSolver::check()` on a small SAT problem.  Because
/// `check()` now calls `probe_variables`, `feasibility_pump`, and (via
/// `branch_and_bound`) `manage_cuts`, this verifies the integrated path
/// does not regress existing SAT detection.
///
/// Problem: x >= 1, x <= 3, x + y <= 5, y >= 0  — trivially SAT.
#[test]
fn test_full_solve_path_with_heuristics() {
    let mut solver = LiaSolver::new();

    let x = solver.new_var();
    let y = solver.new_var();

    // x in [1, 3]
    solver.simplex.set_lower(x, Rational64::from_integer(1), 0);
    solver.simplex.set_upper(x, Rational64::from_integer(3), 1);

    // y >= 0
    solver.simplex.set_lower(y, Rational64::from_integer(0), 2);

    // x + y <= 5
    let mut sum_expr = LinExpr::new();
    sum_expr.add_term(x, Rational64::one());
    sum_expr.add_term(y, Rational64::one());
    sum_expr.add_constant(-Rational64::from_integer(5));
    solver.add_le(sum_expr, 3);

    // check() runs: probe_variables, feasibility_pump, branch_and_bound
    // (which calls manage_cuts at depth 0 since 0 % 8 == 0)
    let result = solver.check();
    assert!(result.is_ok(), "check() should not error on a SAT problem");
    assert!(
        result.expect("SAT problem should return Ok(true)"),
        "the problem is satisfiable — check() should return true"
    );
}
