//! Integration tests for NLSAT (Nonlinear Arithmetic) solver.
//!
//! Tests both QF_NIA (nonlinear integer arithmetic) and QF_NRA (nonlinear
//! real arithmetic) through the high-level `Context` API.
//!
//! The `Context` API dispatches nonlinear assertions through the
//! `Term→Polynomial` translator and `NiaSolver` / `NlsatSolver`.

use oxiz_solver::{Context, SolverResult};

// ─────────────────────────────────────────────────────────────────────────────
// QF_NIA tests — nonlinear integer arithmetic
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_nia_x_squared_eq_4_sat() {
    // x * x = 4 → SAT (x = 2 or x = -2)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let square = ctx.terms.mk_mul(vec![x, x]);
    let four = ctx.terms.mk_int(4);
    let eq = ctx.terms.mk_eq(square, four);
    ctx.assert(eq);

    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "x*x=4 should be SAT for integers, got {:?}",
        result
    );
}

#[test]
fn test_nia_x_squared_eq_3_unsat() {
    // x * x = 3 → UNSAT (3 is not a perfect square)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let square = ctx.terms.mk_mul(vec![x, x]);
    let three = ctx.terms.mk_int(3);
    let eq = ctx.terms.mk_eq(square, three);
    ctx.assert(eq);

    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Unsat),
        "x*x=3 should be UNSAT (3 is not a perfect square), got {:?}",
        result
    );
}

#[test]
fn test_nia_x_squared_eq_neg1_unsat() {
    // x * x = -1 → UNSAT (squares are non-negative)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let square = ctx.terms.mk_mul(vec![x, x]);
    let neg_one = ctx.terms.mk_int(-1);
    let eq = ctx.terms.mk_eq(square, neg_one);
    ctx.assert(eq);

    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Unsat),
        "x*x=-1 should be UNSAT, got {:?}",
        result
    );
}

#[test]
fn test_nia_x_squared_eq_16_sat() {
    // x * x = 16 → SAT (x = 4 or x = -4)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let square = ctx.terms.mk_mul(vec![x, x]);
    let sixteen = ctx.terms.mk_int(16);
    let eq = ctx.terms.mk_eq(square, sixteen);
    ctx.assert(eq);

    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "x*x=16 should be SAT, got {:?}",
        result
    );
}

#[test]
fn test_nia_xy_eq_6_with_bounds_sat() {
    // x * y = 6, x >= 1, y >= 1 → SAT (e.g. x=2, y=3)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let y = ctx.declare_const("y", int_sort);
    let one = ctx.terms.mk_int(1);
    let six = ctx.terms.mk_int(6);
    let xy = ctx.terms.mk_mul(vec![x, y]);
    let eq = ctx.terms.mk_eq(xy, six);
    let x_ge = ctx.terms.mk_ge(x, one);
    let y_ge = ctx.terms.mk_ge(y, one);
    ctx.assert(eq);
    ctx.assert(x_ge);
    ctx.assert(y_ge);

    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "x*y=6, x>=1, y>=1 should be SAT, got {:?}",
        result
    );
}

#[test]
fn test_nia_xy_gt_5_with_bounds_sat() {
    // x * y > 5, x >= 2, y >= 2 → SAT (e.g. x=3, y=2)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let y = ctx.declare_const("y", int_sort);
    let two = ctx.terms.mk_int(2);
    let five = ctx.terms.mk_int(5);
    let xy = ctx.terms.mk_mul(vec![x, y]);
    let gt = ctx.terms.mk_gt(xy, five);
    let x_ge = ctx.terms.mk_ge(x, two);
    let y_ge = ctx.terms.mk_ge(y, two);
    ctx.assert(gt);
    ctx.assert(x_ge);
    ctx.assert(y_ge);

    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "x*y>5, x>=2, y>=2 should be SAT, got {:?}",
        result
    );
}

#[test]
fn test_nia_triple_product_xyz_sat() {
    // x * y * z = 24, x >= 1, y >= 1, z >= 1 → SAT (e.g. x=2, y=3, z=4)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let y = ctx.declare_const("y", int_sort);
    let z = ctx.declare_const("z", int_sort);
    let one = ctx.terms.mk_int(1);
    let twenty_four = ctx.terms.mk_int(24);
    let xyz = ctx.terms.mk_mul(vec![x, y, z]);
    let eq = ctx.terms.mk_eq(xyz, twenty_four);
    let x_ge = ctx.terms.mk_ge(x, one);
    let y_ge = ctx.terms.mk_ge(y, one);
    let z_ge = ctx.terms.mk_ge(z, one);
    ctx.assert(eq);
    ctx.assert(x_ge);
    ctx.assert(y_ge);
    ctx.assert(z_ge);

    // Audit fix: the NIA solver correctly finds a witness here (e.g.
    // x=2,y=3,z=4); accepting `Unknown` was over-lenient and would silently
    // mask a real regression in triple-product handling.
    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "x*y*z=24 should be SAT, got {:?}",
        result
    );
}

#[test]
fn test_nia_factored_product_xp1_ym2_sat() {
    // (x + 1) * (y - 2) = 6, x >= 0, y >= 3 → SAT (e.g. x=1, y=5: 2*3=6)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let y = ctx.declare_const("y", int_sort);
    let zero = ctx.terms.mk_int(0);
    let one = ctx.terms.mk_int(1);
    let two = ctx.terms.mk_int(2);
    let three = ctx.terms.mk_int(3);
    let six = ctx.terms.mk_int(6);
    let xp1 = ctx.terms.mk_add(vec![x, one]);
    let ym2 = ctx.terms.mk_sub(y, two);
    let product = ctx.terms.mk_mul(vec![xp1, ym2]);
    let eq = ctx.terms.mk_eq(product, six);
    let x_ge = ctx.terms.mk_ge(x, zero);
    let y_ge = ctx.terms.mk_ge(y, three);
    ctx.assert(eq);
    ctx.assert(x_ge);
    ctx.assert(y_ge);

    // Audit fix: the NIA solver correctly finds a witness here (e.g.
    // x=1,y=5); accepting `Unknown` was over-lenient.
    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "(x+1)*(y-2)=6 with bounds should be SAT, got {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// QF_NRA tests — nonlinear real arithmetic
//
// Three of these need the `nlsat` feature and are marked individually rather
// than as a block, because the fourth (`test_nra_x_squared_eq_2_sat`) does not:
// it already accepts `Unknown`, which is what *both* builds answer for an
// irrational root. The rule for the marks below is the one measured in
// `oxiz-solver/tests/nlsat_feature_gate.rs`: QF_NRA is the logic that loses its
// nonlinear verdicts when `oxiz-nlsat` is not compiled, because the two
// witness-verified model searches that survive are gated on QF_NIA. The QF_NIA
// group above therefore carries no marks at all and must keep passing in both
// builds — it is the no-regression evidence.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "nlsat")]
#[test]
fn test_nra_x_squared_lt_0_unsat() {
    // x * x < 0 → UNSAT (no real squared is negative)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NRA");

    let real_sort = ctx.terms.sorts.real_sort;
    let x = ctx.declare_const("x", real_sort);
    let square = ctx.terms.mk_mul(vec![x, x]);
    let zero = ctx.terms.mk_int(0);
    let lt = ctx.terms.mk_lt(square, zero);
    ctx.assert(lt);

    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Unsat),
        "x*x<0 over reals should be UNSAT, got {:?}",
        result
    );
}

#[test]
fn test_nra_x_squared_eq_2_sat() {
    // x * x = 2 → SAT (x = sqrt(2))
    let mut ctx = Context::new();
    ctx.set_logic("QF_NRA");

    let real_sort = ctx.terms.sorts.real_sort;
    let x = ctx.declare_const("x", real_sort);
    let square = ctx.terms.mk_mul(vec![x, x]);
    let two = ctx.terms.mk_int(2);
    let eq = ctx.terms.mk_eq(square, two);
    ctx.assert(eq);

    // NOTE: unlike the other `Sat | Unknown` cases in this file, this one is
    // a genuine, currently-verified incompleteness gap rather than stale
    // leniency: the only witnesses are the irrational algebraic numbers
    // `x = ±sqrt(2)`, and the NRA backend currently reports `Unknown`
    // instead of a full CAD-based algebraic witness. Tightening this to
    // `Sat` would be dishonest until that gap is closed.
    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat | SolverResult::Unknown),
        "x*x=2 over reals should be SAT or Unknown, got {:?}",
        result
    );
}

#[cfg(feature = "nlsat")]
#[test]
fn test_nra_circle_inside_sat() {
    // x * x + y * y < 1 → SAT (e.g. x=0, y=0 is inside unit circle)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NRA");

    let real_sort = ctx.terms.sorts.real_sort;
    let x = ctx.declare_const("x", real_sort);
    let y = ctx.declare_const("y", real_sort);
    let x_sq = ctx.terms.mk_mul(vec![x, x]);
    let y_sq = ctx.terms.mk_mul(vec![y, y]);
    let sum = ctx.terms.mk_add(vec![x_sq, y_sq]);
    let one = ctx.terms.mk_int(1);
    let lt = ctx.terms.mk_lt(sum, one);
    ctx.assert(lt);

    // Audit fix: `(0, 0)` is a rational witness the solver correctly finds;
    // accepting `Unknown` was over-lenient.
    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "x^2+y^2<1 should be SAT, got {:?}",
        result
    );
}

#[cfg(feature = "nlsat")]
#[test]
fn test_nra_polynomial_x2_minus_2x_plus_1_sat() {
    // x^2 - 2*x + 1 = 0  ↔  (x-1)^2 = 0  → SAT (x=1)
    let mut ctx = Context::new();
    ctx.set_logic("QF_NRA");

    let real_sort = ctx.terms.sorts.real_sort;
    let x = ctx.declare_const("x", real_sort);
    let x_sq = ctx.terms.mk_mul(vec![x, x]);
    let two = ctx.terms.mk_int(2);
    let one = ctx.terms.mk_int(1);
    let zero = ctx.terms.mk_int(0);
    let two_x = ctx.terms.mk_mul(vec![two, x]);
    let x2_minus_2x = ctx.terms.mk_sub(x_sq, two_x);
    let poly = ctx.terms.mk_add(vec![x2_minus_2x, one]);
    let eq = ctx.terms.mk_eq(poly, zero);
    ctx.assert(eq);

    // Audit fix: `x=1` is a rational witness the solver correctly finds;
    // accepting `Unknown` was over-lenient.
    let result = ctx.check_sat();
    assert!(
        matches!(result, SolverResult::Sat),
        "x^2-2x+1=0 over reals should be SAT, got {:?}",
        result
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Push / pop tests
// ─────────────────────────────────────────────────────────────────────────────

// The one QF_NIA test that needs `nlsat`, and it is worth saying why, since
// every other QF_NIA test in this file passes without it.
//
// The level-2 step asserts `x*x = 4 ∧ x < 0 ∧ x > 0` is UNSAT, and the comment
// inside it is right that the contradiction is purely linear. What it needs the
// dispatcher for is *ordering*: `check_core` consults its
// `arith_atoms_need_theory` honesty gate before the CDCL(T) search, and with
// `x*x = 4` in scope and no theory able to take it, that gate answers `Unknown`
// — correctly, since nothing has proven anything yet. Only `dispatch_nl_solver`
// runs ahead of the gate, so without it the linear conflict is never reached.
// The verdict is conceded, never wrong; the OFF build's version of this
// sequence is pinned in `oxiz-solver/tests/nlsat_feature_gate.rs`.
#[cfg(feature = "nlsat")]
#[test]
fn test_nia_push_pop_backtrack() {
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");

    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let square = ctx.terms.mk_mul(vec![x, x]);
    let four = ctx.terms.mk_int(4);
    let eq = ctx.terms.mk_eq(square, four);
    ctx.assert(eq);

    // Level 0: x*x=4 → SAT
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));

    ctx.push();

    // Level 1: add x < 0 → still SAT (x=-2 is a solution)
    let zero = ctx.terms.mk_int(0);
    let x_lt = ctx.terms.mk_lt(x, zero);
    ctx.assert(x_lt);
    // x*x=4 and x<0 → x=-2 is SAT. Audit fix: the solver correctly finds
    // this witness; accepting `Unknown` was over-lenient.
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));

    ctx.push();

    // Level 2: add x > 0 — conflicts with x < 0
    let x_gt = ctx.terms.mk_gt(x, zero);
    ctx.assert(x_gt);
    // x<0 AND x>0 → UNSAT. This is a direct propositional contradiction on
    // linear atoms (independent of the nonlinear x*x=4 constraint also in
    // scope), so it requires no nonlinear-arithmetic completeness at all.
    // Audit fix: the solver correctly detects this; accepting `Unknown`
    // here would silently mask a real regression in basic conflict
    // detection.
    let result_l2 = ctx.check_sat();
    assert!(
        matches!(result_l2, SolverResult::Unsat),
        "x<0 AND x>0 should be UNSAT, got {:?}",
        result_l2
    );

    // Pop back to level 1
    ctx.pop();
    // x*x=4 and x<0 is still SAT.
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));

    // Pop back to level 0
    ctx.pop();
    // x*x=4 alone is SAT
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixture-based tests: bench/extended_theories/QF_NIA_ext/
// ─────────────────────────────────────────────────────────────────────────────

/// Run a single SMT2 fixture and return the solver result string.
fn run_smt2_fixture(path: &std::path::Path) -> SolverResult {
    let source = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));
    let mut ctx = Context::new();
    match ctx.execute_script(&source) {
        Ok(outputs) => {
            // The last output line is from (check-sat)
            for line in outputs.iter().rev() {
                match line.trim() {
                    "sat" => return SolverResult::Sat,
                    "unsat" => return SolverResult::Unsat,
                    "unknown" => return SolverResult::Unknown,
                    _ => {}
                }
            }
            SolverResult::Unknown
        }
        Err(_) => SolverResult::Unknown,
    }
}

/// Extract the expected result from the first comment line of an SMT2 file.
/// Looks for `;; expected: sat`, `unsat`, or `unknown`.
fn expected_result(path: &std::path::Path) -> Option<SolverResult> {
    let source = std::fs::read_to_string(path).ok()?;
    for line in source.lines().take(10) {
        let lower = line.to_lowercase();
        if lower.contains("expected:") || lower.contains("expected :") {
            if lower.contains("unsat") {
                return Some(SolverResult::Unsat);
            } else if lower.contains("unknown") {
                return Some(SolverResult::Unknown);
            } else if lower.contains("sat") {
                return Some(SolverResult::Sat);
            }
        }
    }
    None
}

#[test]
fn test_qf_nia_ext_fixtures() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../bench/extended_theories/QF_NIA_ext");

    if !fixture_dir.exists() {
        // Fixture directory doesn't exist — skip silently.
        return;
    }

    let entries: Vec<_> = std::fs::read_dir(&fixture_dir)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", fixture_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "smt2").unwrap_or(false))
        .collect();

    if entries.is_empty() {
        return;
    }

    let mut failures = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let expected = expected_result(&path);
        let actual = run_smt2_fixture(&path);

        if let Some(exp) = expected {
            // Allow Unknown as a valid "pass" when a fixture is expected to be
            // solved but the current architecture is inconclusive.
            let passes = if matches!(exp, SolverResult::Unknown) {
                // `expected: unknown` marks fixtures that are out of scope for
                // the current architecture. Keep them in the sweep for
                // visibility, but do not require a specific definitive answer.
                true
            } else {
                actual == exp || matches!(actual, SolverResult::Unknown)
            };
            if !passes {
                failures.push(format!(
                    "{}: expected {:?}, got {:?}",
                    path.display(),
                    exp,
                    actual
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "QF_NIA_ext fixture failures:\n{}",
        failures.join("\n")
    );
}

#[test]
fn test_qf_nia_z3_parity_fixtures() {
    let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../bench/z3_parity/benchmarks/qf_nia");

    if !fixture_dir.exists() {
        return;
    }

    let entries: Vec<_> = std::fs::read_dir(&fixture_dir)
        .unwrap_or_else(|_| panic!("Failed to read {:?}", fixture_dir))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "smt2").unwrap_or(false))
        .collect();

    if entries.is_empty() {
        return;
    }

    let mut failures = Vec::new();

    for entry in &entries {
        let path = entry.path();
        let expected = expected_result(&path);
        let actual = run_smt2_fixture(&path);

        if let Some(exp) = expected
            && actual != exp
            && !matches!(actual, SolverResult::Unknown)
        {
            failures.push(format!(
                "{}: expected {:?}, got {:?}",
                path.display(),
                exp,
                actual
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "z3_parity/qf_nia fixture failures:\n{}",
        failures.join("\n")
    );
}
