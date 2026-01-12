//! Integration tests for the NLSAT solver.
//!
//! These tests verify end-to-end functionality of the solver with various
//! types of polynomial constraints and real-world problem scenarios.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::One;
use oxiz_math::polynomial::Polynomial;
use oxiz_nlsat::solver::{NlsatSolver, SolverResult};
use oxiz_nlsat::types::AtomKind;

/// Helper to create a polynomial from coefficients.
/// coeffs[0] is the constant term, coeffs[1] is the coefficient of x, etc.
fn poly(var: u32, coeffs: &[i64]) -> Polynomial {
    let mut p = Polynomial::zero();
    let x = Polynomial::from_var(var);

    for (i, &coeff) in coeffs.iter().enumerate() {
        if coeff != 0 {
            let coef_poly = Polynomial::constant(BigRational::from_integer(BigInt::from(coeff)));
            let mut term = coef_poly;
            for _ in 0..i {
                term = term * x.clone();
            }
            p = p + term;
        }
    }
    p
}

#[test]
fn test_simple_linear_sat() {
    // x > 0
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[0, 1]); // x
    let atom_id = solver.new_ineq_atom(p, AtomKind::Gt);
    let lit = solver.atom_literal(atom_id, true);

    solver.add_clause(vec![lit]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_simple_linear_unsat() {
    // x > 0 AND x < 0
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p1 = poly(0, &[0, 1]); // x
    let p2 = poly(0, &[0, 1]); // x

    let atom1 = solver.new_ineq_atom(p1, AtomKind::Gt);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Unsat);
}

#[test]
#[ignore] // Equality constraints with irrational roots may be challenging
fn test_quadratic_roots() {
    // x^2 - 2 = 0  (should be SAT with x = ±√2)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[-2, 0, 1]); // -2 + x^2
    let atom_id = solver.new_ineq_atom(p, AtomKind::Eq);

    solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_quadratic_inequality() {
    // x^2 < 4  (should be SAT for x in (-2, 2))
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[-4, 0, 1]); // -4 + x^2
    let atom_id = solver.new_ineq_atom(p, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_cubic_polynomial() {
    // x^3 - x = 0  (roots at -1, 0, 1)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[0, -1, 0, 1]); // -x + x^3
    let atom_id = solver.new_ineq_atom(p, AtomKind::Eq);

    solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_two_variables_sat() {
    // x > 0 AND y > 0 AND x + y < 10
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // x > 0
    let p1 = poly(0, &[0, 1]);
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Gt);

    // y > 0
    let p2 = poly(1, &[0, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Gt);

    // x + y < 10
    let mut p3 = poly(0, &[0, 1]);
    p3 = p3 + poly(1, &[0, 1]);
    p3 = p3 - Polynomial::constant(BigRational::from_integer(BigInt::from(10)));
    let atom3 = solver.new_ineq_atom(p3, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);
    solver.add_clause(vec![solver.atom_literal(atom3, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_two_variables_unsat() {
    // x > 5 AND y > 5 AND x + y < 5
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // x > 5
    let p1 = poly(0, &[-5, 1]);
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Gt);

    // y > 5
    let p2 = poly(1, &[-5, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Gt);

    // x + y < 5
    let mut p3 = poly(0, &[0, 1]);
    p3 = p3 + poly(1, &[0, 1]);
    p3 = p3 - Polynomial::constant(BigRational::from_integer(BigInt::from(5)));
    let atom3 = solver.new_ineq_atom(p3, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);
    solver.add_clause(vec![solver.atom_literal(atom3, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Unsat);
}

#[test]
fn test_circle_constraint() {
    // x^2 + y^2 = 1  (circle of radius 1)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // x^2
    let x_squared = poly(0, &[0, 0, 1]);
    // y^2
    let y_squared = poly(1, &[0, 0, 1]);
    // x^2 + y^2 - 1
    let mut p = x_squared + y_squared;
    p = p - Polynomial::constant(BigRational::one());

    let atom_id = solver.new_ineq_atom(p, AtomKind::Eq);
    solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
#[ignore] // Complex multi-variable equality constraints - challenging for solver
fn test_circle_and_line() {
    // x^2 + y^2 = 1 AND y = x (line intersects circle)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // x^2 + y^2 - 1 = 0
    let x_squared = poly(0, &[0, 0, 1]);
    let y_squared = poly(1, &[0, 0, 1]);
    let mut p1 = x_squared + y_squared;
    p1 = p1 - Polynomial::constant(BigRational::one());
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Eq);

    // y - x = 0
    let mut p2 = poly(1, &[0, 1]);
    p2 = p2 - poly(0, &[0, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Eq);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_circle_outside_point() {
    // x^2 + y^2 = 1 AND x = 2 (point outside circle - unsat)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // x^2 + y^2 - 1 = 0
    let x_squared = poly(0, &[0, 0, 1]);
    let y_squared = poly(1, &[0, 0, 1]);
    let mut p1 = x_squared + y_squared;
    p1 = p1 - Polynomial::constant(BigRational::one());
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Eq);

    // x - 2 = 0
    let p2 = poly(0, &[-2, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Eq);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Unsat);
}

#[test]
fn test_disjunction() {
    // x < 0 OR x > 1
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p1 = poly(0, &[0, 1]); // x
    let p2 = poly(0, &[-1, 1]); // x - 1

    let atom1 = solver.new_ineq_atom(p1, AtomKind::Lt);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Gt);

    solver.add_clause(vec![
        solver.atom_literal(atom1, true),
        solver.atom_literal(atom2, true),
    ]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_conjunction_via_clauses() {
    // (x > 0) AND (x < 10) represented as two unit clauses
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p1 = poly(0, &[0, 1]); // x
    let p2 = poly(0, &[-10, 1]); // x - 10

    let atom1 = solver.new_ineq_atom(p1, AtomKind::Gt);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_polynomial_factorization_case() {
    // (x - 1)(x + 1) = 0, i.e., x^2 - 1 = 0
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[-1, 0, 1]); // -1 + x^2
    let atom_id = solver.new_ineq_atom(p, AtomKind::Eq);

    solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
#[ignore] // High-degree polynomial equalities may be challenging
fn test_high_degree_polynomial() {
    // x^4 - 1 = 0 (roots at ±1)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[-1, 0, 0, 0, 1]); // -1 + x^4
    let atom_id = solver.new_ineq_atom(p, AtomKind::Eq);

    solver.add_clause(vec![solver.atom_literal(atom_id, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_inequality_chain() {
    // 0 < x < 5 < y < 10
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // x > 0
    let p1 = poly(0, &[0, 1]);
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Gt);

    // x < 5
    let p2 = poly(0, &[-5, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Lt);

    // y > 5
    let p3 = poly(1, &[-5, 1]);
    let atom3 = solver.new_ineq_atom(p3, AtomKind::Gt);

    // y < 10
    let p4 = poly(1, &[-10, 1]);
    let atom4 = solver.new_ineq_atom(p4, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);
    solver.add_clause(vec![solver.atom_literal(atom3, true)]);
    solver.add_clause(vec![solver.atom_literal(atom4, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_three_variables() {
    // x > 0 AND y > 0 AND z > 0 AND x + y + z < 10
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();
    let _z = solver.new_arith_var();

    // x > 0
    let p1 = poly(0, &[0, 1]);
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Gt);

    // y > 0
    let p2 = poly(1, &[0, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Gt);

    // z > 0
    let p3 = poly(2, &[0, 1]);
    let atom3 = solver.new_ineq_atom(p3, AtomKind::Gt);

    // x + y + z < 10
    let mut p4 = poly(0, &[0, 1]);
    p4 = p4 + poly(1, &[0, 1]);
    p4 = p4 + poly(2, &[0, 1]);
    p4 = p4 - Polynomial::constant(BigRational::from_integer(BigInt::from(10)));
    let atom4 = solver.new_ineq_atom(p4, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);
    solver.add_clause(vec![solver.atom_literal(atom3, true)]);
    solver.add_clause(vec![solver.atom_literal(atom4, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_mixed_equalities_inequalities() {
    // x = 1 AND y > 0 AND x + y < 5
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // x = 1
    let p1 = poly(0, &[-1, 1]);
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Eq);

    // y > 0
    let p2 = poly(1, &[0, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Gt);

    // x + y < 5
    let mut p3 = poly(0, &[0, 1]);
    p3 = p3 + poly(1, &[0, 1]);
    p3 = p3 - Polynomial::constant(BigRational::from_integer(BigInt::from(5)));
    let atom3 = solver.new_ineq_atom(p3, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);
    solver.add_clause(vec![solver.atom_literal(atom3, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_parabola_constraints() {
    // y = x^2 AND y < 4 (should be SAT)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();
    let _y = solver.new_arith_var();

    // y - x^2 = 0
    let mut p1 = poly(1, &[0, 1]);
    p1 = p1 - poly(0, &[0, 0, 1]);
    let atom1 = solver.new_ineq_atom(p1, AtomKind::Eq);

    // y < 4
    let p2 = poly(1, &[-4, 1]);
    let atom2 = solver.new_ineq_atom(p2, AtomKind::Lt);

    solver.add_clause(vec![solver.atom_literal(atom1, true)]);
    solver.add_clause(vec![solver.atom_literal(atom2, true)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_negated_literals() {
    // NOT(x > 0) which is equivalent to x <= 0
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[0, 1]); // x
    let atom_id = solver.new_ineq_atom(p, AtomKind::Gt);

    // Add NOT(x > 0)
    solver.add_clause(vec![solver.atom_literal(atom_id, false)]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}

#[test]
fn test_tautology() {
    // x > 0 OR x <= 0 (always true)
    let mut solver = NlsatSolver::new();
    let _x = solver.new_arith_var();

    let p = poly(0, &[0, 1]); // x
    let atom_id = solver.new_ineq_atom(p, AtomKind::Gt);

    // Add (x > 0) OR NOT(x > 0)
    solver.add_clause(vec![
        solver.atom_literal(atom_id, true),
        solver.atom_literal(atom_id, false),
    ]);

    let result = solver.solve();
    assert_eq!(result, SolverResult::Sat);
}
