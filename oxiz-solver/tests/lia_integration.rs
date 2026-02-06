//! LIA (Linear Integer Arithmetic) Integration Tests
//!
//! These tests verify that the solver correctly handles integer arithmetic constraints,
//! including GCD-based infeasibility detection and cutting planes.

use oxiz_solver::Context;

/// Test GCD-based infeasibility detection for equality constraints
///
/// For the constraint 2x + 2y = 7:
/// - All coefficients have GCD = 2
/// - The constant 7 is not divisible by 2
/// - Therefore, no integer solution exists
///
/// This test ensures the solver detects this infeasibility immediately
/// during assertion, before even invoking the simplex solver.
///
/// Reference: Schrijver, "Theory of Linear and Integer Programming" (1986)
#[test]
fn test_lia_gcd_infeasibility_basic() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_LIA");

    // Create integer variables
    let x = ctx.declare_const("x", ctx.terms.sorts.int_sort);
    let y = ctx.declare_const("y", ctx.terms.sorts.int_sort);

    // Build the constraint: 2x + 2y = 7
    let two = ctx.terms.mk_int(2);
    let seven = ctx.terms.mk_int(7);
    let two_x = ctx.terms.mk_mul(vec![two, x]);
    let two_y = ctx.terms.mk_mul(vec![two, y]);
    let sum = ctx.terms.mk_add(vec![two_x, two_y]);
    let constraint = ctx.terms.mk_eq(sum, seven);

    ctx.assert(constraint);

    // Also add non-negativity constraints (shouldn't matter for GCD check)
    let zero = ctx.terms.mk_int(0);
    let x_nonneg = ctx.terms.mk_ge(x, zero);
    let y_nonneg = ctx.terms.mk_ge(y, zero);
    ctx.assert(x_nonneg);
    ctx.assert(y_nonneg);

    // The result must be UNSAT due to GCD infeasibility
    let result = ctx.check_sat();
    assert!(
        matches!(result, oxiz_solver::SolverResult::Unsat),
        "Expected UNSAT for 2x + 2y = 7 (GCD infeasibility), got {:?}",
        result
    );
}

/// Test GCD-based infeasibility with larger GCD
///
/// For 6x + 9y + 12z = 5:
/// - GCD(6, 9, 12) = 3
/// - 5 is not divisible by 3
/// - Therefore UNSAT
#[test]
fn test_lia_gcd_infeasibility_larger_gcd() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_LIA");

    let x = ctx.declare_const("x", ctx.terms.sorts.int_sort);
    let y = ctx.declare_const("y", ctx.terms.sorts.int_sort);
    let z = ctx.declare_const("z", ctx.terms.sorts.int_sort);

    // 6x + 9y + 12z = 5
    let six = ctx.terms.mk_int(6);
    let nine = ctx.terms.mk_int(9);
    let twelve = ctx.terms.mk_int(12);
    let five = ctx.terms.mk_int(5);

    let six_x = ctx.terms.mk_mul(vec![six, x]);
    let nine_y = ctx.terms.mk_mul(vec![nine, y]);
    let twelve_z = ctx.terms.mk_mul(vec![twelve, z]);

    let sum = ctx.terms.mk_add(vec![six_x, nine_y, twelve_z]);
    let constraint = ctx.terms.mk_eq(sum, five);

    ctx.assert(constraint);

    let result = ctx.check_sat();
    assert!(
        matches!(result, oxiz_solver::SolverResult::Unsat),
        "Expected UNSAT for 6x + 9y + 12z = 5 (GCD = 3 doesn't divide 5), got {:?}",
        result
    );
}

/// Test that GCD-satisfiable constraints are SAT
///
/// For 2x + 2y = 6:
/// - GCD(2, 2) = 2
/// - 6 is divisible by 2
/// - Therefore SAT (e.g., x=1, y=2 is a solution)
#[test]
fn test_lia_gcd_satisfiable() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_LIA");

    let x = ctx.declare_const("x", ctx.terms.sorts.int_sort);
    let y = ctx.declare_const("y", ctx.terms.sorts.int_sort);

    // 2x + 2y = 6 (SAT: x=1, y=2 works)
    let two = ctx.terms.mk_int(2);
    let six = ctx.terms.mk_int(6);
    let two_x = ctx.terms.mk_mul(vec![two, x]);
    let two_y = ctx.terms.mk_mul(vec![two, y]);
    let sum = ctx.terms.mk_add(vec![two_x, two_y]);
    let constraint = ctx.terms.mk_eq(sum, six);

    ctx.assert(constraint);

    let result = ctx.check_sat();
    assert!(
        matches!(result, oxiz_solver::SolverResult::Sat),
        "Expected SAT for 2x + 2y = 6 (GCD-satisfiable), got {:?}",
        result
    );
}

/// Test mixed equality and inequality constraints
///
/// This is a more complex test that combines GCD reasoning with
/// inequality constraints.
#[test]
fn test_lia_mixed_constraints_with_gcd() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_LIA");

    let x = ctx.declare_const("x", ctx.terms.sorts.int_sort);
    let y = ctx.declare_const("y", ctx.terms.sorts.int_sort);

    // Constraint 1: 2x + 2y = 7 (GCD-infeasible)
    let two = ctx.terms.mk_int(2);
    let seven = ctx.terms.mk_int(7);
    let two_x = ctx.terms.mk_mul(vec![two, x]);
    let two_y = ctx.terms.mk_mul(vec![two, y]);
    let sum = ctx.terms.mk_add(vec![two_x, two_y]);
    let eq_constraint = ctx.terms.mk_eq(sum, seven);

    // Constraint 2: x >= 0
    let zero = ctx.terms.mk_int(0);
    let x_nonneg = ctx.terms.mk_ge(x, zero);

    // Constraint 3: y >= 0
    let y_nonneg = ctx.terms.mk_ge(y, zero);

    ctx.assert(eq_constraint);
    ctx.assert(x_nonneg);
    ctx.assert(y_nonneg);

    let result = ctx.check_sat();
    assert!(
        matches!(result, oxiz_solver::SolverResult::Unsat),
        "Expected UNSAT (GCD infeasibility should dominate), got {:?}",
        result
    );
}

/// Test that fractional constants in equality are detected as infeasible for LIA
#[test]
fn test_lia_fractional_constant_in_equality() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_LIA");

    let x = ctx.declare_const("x", ctx.terms.sorts.int_sort);

    // x = 3.5 should be UNSAT for integer x
    use num_rational::Rational64;
    let three_point_five = ctx.terms.mk_real(Rational64::new(7, 2));
    let constraint = ctx.terms.mk_eq(x, three_point_five);

    ctx.assert(constraint);

    let result = ctx.check_sat();
    // This should be UNSAT because we can't have an integer equal to 3.5
    assert!(
        matches!(result, oxiz_solver::SolverResult::Unsat),
        "Expected UNSAT for x = 3.5 with integer x, got {:?}",
        result
    );
}
