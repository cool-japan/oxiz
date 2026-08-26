//! What wiring the NIA-over-LP relaxation engine into the QF_NIA dispatch
//! actually bought, pinned against measurement rather than intent.
//!
//! The engine is `oxiz_theories::arithmetic::nla`; `check_nlsat::
//! dispatch_nl_solver` calls it on QF_NIA goals the cell-decomposition core
//! declined, gated on [`SolverConfig::nonlinear_relaxation_engine`]. Because it
//! runs *after* that core, the only answers it can change are goals that
//! previously came back `unknown` — which is what makes this a completeness
//! gain with no parity exposure, and what these tests are shaped to check.
//!
//! # Why every unsat case here is a paired measurement
//!
//! A test that only asserts `unsat` proves nothing about the wiring: the static
//! pattern detector in `check_core` already refutes several nonlinear shapes
//! (`x*x = -1` among them) and would keep the test green with the engine ripped
//! out. So each goal below is solved *twice* — once with the flag on, once with
//! it off — and both halves are asserted:
//!
//! * flag **on** → `unsat`, the new answer;
//! * flag **off** → `unknown`, which is what this tree measurably answered
//!   before the engine was wired in.
//!
//! The second half is the load-bearing one. It pins the flag as a real switch
//! (so "budget switch" is a claim with a test behind it), and it is also the
//! record of the pre-wiring baseline: if some *other* procedure later learns to
//! refute one of these, the off-half fails and says so, rather than the on-half
//! silently passing for a new reason.
//!
//! Baseline measured on this tree, both feature builds, before the dispatch
//! call existed: all four unsat goals below answered `unknown`; the `sat` goals
//! in the last section answered `sat` exactly as they do now.

use oxiz_core::ast::TermManager;
use oxiz_solver::{Context, Solver, SolverConfig, SolverResult};

/// Run an SMT-LIB2 script the way a consumer does, returning the output lines.
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
        .expect("script should parse and run")
}

/// Solve `build`'s assertions under QF_NIA with the relaxation engine forced on
/// or off, leaving every other knob at its default.
fn solve_with_engine(
    engine_enabled: bool,
    build: impl Fn(&mut TermManager) -> Vec<oxiz_core::ast::TermId>,
) -> SolverResult {
    let mut manager = TermManager::new();
    let assertions = build(&mut manager);
    let config = SolverConfig {
        nonlinear_relaxation_engine: engine_enabled,
        ..SolverConfig::default()
    };
    let mut solver = Solver::with_config(config);
    solver.set_logic("QF_NIA");
    for assertion in assertions {
        solver.assert(assertion, &mut manager);
    }
    solver.check(&mut manager)
}

/// Assert the paired measurement described in the module docs: the engine
/// refutes `build`, and without it the same goal is conceded rather than
/// answered differently.
fn assert_engine_refutes(
    what: &str,
    build: impl Fn(&mut TermManager) -> Vec<oxiz_core::ast::TermId> + Copy,
) {
    assert_eq!(
        solve_with_engine(true, build),
        SolverResult::Unsat,
        "the relaxation engine must refute {what}"
    );
    assert_eq!(
        solve_with_engine(false, build),
        SolverResult::Unknown,
        "with the engine off, {what} must fall back to the pre-wiring `unknown` \
         — not to a different verdict, and not to an answer from somewhere else"
    );
}

// ---------------------------------------------------------------------
// 1. Multivariate goals the engine refutes and nothing before it could.
// ---------------------------------------------------------------------

/// `x ≥ 3 ∧ y ≥ 3 ∧ x*y ≤ 8`.
///
/// Unsatisfiable because the product of two integers each at least 3 is at
/// least 9. No single-variable square pattern sees this, and the linear
/// relaxation alone does not either — it takes the monomial bound `x*y ≥ 9`
/// derived from the two factor bounds, which is exactly what the engine's
/// interval layer contributes.
#[test]
fn bounded_factors_cannot_make_a_small_product() {
    assert_engine_refutes("x >= 3 AND y >= 3 AND x*y <= 8", |manager| {
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let three = manager.mk_int(3);
        let eight = manager.mk_int(8);
        let product = manager.mk_mul(vec![x, y]);
        vec![
            manager.mk_ge(x, three),
            manager.mk_ge(y, three),
            manager.mk_le(product, eight),
        ]
    });
}

/// `x*x + y*y + 1 = 0`.
///
/// The `x*x = -1` shape, but spread over two variables so no atom of the form
/// "this square equals that negative constant" exists to be pattern-matched.
/// Refuting it needs both squares bounded below by 0 at once.
#[test]
fn a_sum_of_squares_is_never_negative() {
    assert_engine_refutes("x*x + y*y + 1 = 0", |manager| {
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let x_sq = manager.mk_mul(vec![x, x]);
        let y_sq = manager.mk_mul(vec![y, y]);
        let one = manager.mk_int(1);
        let zero = manager.mk_int(0);
        let sum = manager.mk_add(vec![x_sq, y_sq, one]);
        vec![manager.mk_eq(sum, zero)]
    });
}

/// `y = x - 5 ∧ y*y < 0`.
///
/// The square is of a *derived* term rather than a declared variable, so the
/// static detector — which matches `x * x` syntactically — has nothing to bite
/// on. The engine linearises `y` as a variable in its own right and the square
/// bound applies.
#[test]
fn a_square_of_a_derived_term_is_never_negative() {
    assert_engine_refutes("y = x - 5 AND y*y < 0", |manager| {
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let five = manager.mk_int(5);
        let zero = manager.mk_int(0);
        let shifted = manager.mk_sub(x, five);
        let y_sq = manager.mk_mul(vec![y, y]);
        vec![manager.mk_eq(y, shifted), manager.mk_lt(y_sq, zero)]
    });
}

/// `x ≥ 2 ∧ y ≥ 2 ∧ z ≥ 2 ∧ x*y*z ≤ 7`.
///
/// A three-factor product chain: the smallest value is 8, so the bound is
/// unreachable. Degree three is past what pairwise reasoning reaches in one
/// step, which is what makes this a different test from the two-factor case
/// above rather than a restatement of it.
#[test]
fn a_three_factor_product_chain_has_a_floor() {
    assert_engine_refutes("x,y,z >= 2 AND x*y*z <= 7", |manager| {
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let z = manager.mk_var("z", int_sort);
        let two = manager.mk_int(2);
        let seven = manager.mk_int(7);
        let product = manager.mk_mul(vec![x, y, z]);
        vec![
            manager.mk_ge(x, two),
            manager.mk_ge(y, two),
            manager.mk_ge(z, two),
            manager.mk_le(product, seven),
        ]
    });
}

// ---------------------------------------------------------------------
// 2. The engine is a completeness gain, so nothing already answered moves.
// ---------------------------------------------------------------------

/// The in-repo parity benchmark `bench/z3_parity/benchmarks/qf_nia/
/// nia_01_simple_mult.smt2`, verbatim. It answered `sat` before the engine was
/// wired in and must still answer `sat`: it is the only QF_NIA instance in the
/// parity suite, hence the suite's entire exposure to this change, reproduced
/// here where a failure names the cause instead of showing up as a parity
/// regression with no obvious owner.
///
/// It is a *verdict* pin, not a model pin. See
/// `already_decided_sat_goals_keep_their_models` below for why an exact
/// `(get-value ...)` string would be the wrong thing to assert here.
#[test]
fn the_parity_benchmark_still_answers_sat() {
    assert_eq!(
        run(
            "(set-logic QF_NIA)(declare-const x Int)(declare-const y Int)\
             (assert (= (* x y) 12))(assert (>= x 1))(assert (<= x 12))\
             (assert (>= y 1))(check-sat)"
        ),
        vec!["sat"],
        "the qf_nia parity benchmark must not move"
    );
}

/// Satisfiable QF_NIA goals that were already decided, checked through the
/// script layer with their models. A `sat` that lost its model, or gained a
/// wrong one, is exactly the regression the "verdicts do not move" claim is
/// supposed to exclude — so the verdict and the model are both asserted.
///
/// The models are asserted as *sets of valid roots*, not as exact strings, and
/// that is deliberate rather than lax. The engine sits above the two searches
/// `nonlinear_model_search` gates, so on a goal both can solve it now answers
/// first and reports its own witness. Measured on this tree, `x*y = 6 ∧
/// x+y = 5` reports (2, 3) in the default build and (3, 2) in the no-`nlsat`
/// build — where the cell decomposition is absent, so the engine is what
/// answers. Both satisfy the script; neither is more correct. Pinning one
/// string would fail in the other build for no defect, which is why the
/// property under test is "a genuine solution", not "this solution".
#[test]
fn already_decided_sat_goals_keep_their_models() {
    let square = run("(set-logic QF_NIA)(declare-const x Int)\
         (assert (= (* x x) 4))(check-sat)(get-value (x))");
    assert_eq!(square.first().map(String::as_str), Some("sat"));
    let value = square.get(1).map(String::as_str).unwrap_or("");
    assert!(
        value == "((x 2))" || value == "((x -2))",
        "x*x = 4 must report a genuine root, got {value:?}"
    );

    let product = run(
        "(set-logic QF_NIA)(declare-const x Int)(declare-const y Int)\
         (assert (= (* x y) 6))(assert (= (+ x y) 5))(check-sat)(get-value (x y))",
    );
    assert_eq!(product.first().map(String::as_str), Some("sat"));
    let values = product.get(1).map(String::as_str).unwrap_or("");
    assert!(
        values == "((x 2)\n (y 3))" || values == "((x 3)\n (y 2))",
        "x*y = 6 AND x+y = 5 must report {{2,3}} in some order, got {values:?}"
    );
}

/// The engine may never invert a verdict — only fill one in. Turning the flag
/// off on a goal that was *already* decided must therefore change nothing at
/// all, which is the complement of the `unsat`/`unknown` pairs above.
#[test]
fn the_flag_does_not_move_an_already_decided_goal() {
    let build = |manager: &mut TermManager| {
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let product = manager.mk_mul(vec![x, y]);
        let twelve = manager.mk_int(12);
        let one = manager.mk_int(1);
        vec![
            manager.mk_eq(product, twelve),
            manager.mk_ge(x, one),
            manager.mk_ge(y, one),
        ]
    };
    assert_eq!(solve_with_engine(true, build), SolverResult::Sat);
    assert_eq!(
        solve_with_engine(false, build),
        SolverResult::Sat,
        "a goal the earlier procedures already decided must not depend on this flag"
    );
}

// ---------------------------------------------------------------------
// 3. The engine is `std`-gated, not `nlsat`-gated.
// ---------------------------------------------------------------------

/// No `cfg`: this test runs in both feature builds and must agree, which is
/// what makes it a proof that the engine survives `--no-default-features
/// --features std,property-tests` rather than two separate expectations.
/// `nlsat_feature_gate.rs` carries the same claim for the goals it pins.
#[test]
fn the_engine_answers_the_same_in_both_feature_builds() {
    assert_eq!(
        run(
            "(set-logic QF_NIA)(declare-const x Int)(declare-const y Int)\
             (assert (>= x 3))(assert (>= y 3))(assert (<= (* x y) 8))(check-sat)"
        ),
        vec!["unsat"],
        "the relaxation engine is compiled in with or without `nlsat`"
    );
}

// ---------------------------------------------------------------------
// 4. QF_NRA is untouched.
// ---------------------------------------------------------------------

/// The dispatch gates the engine on `is_nia`, and the engine itself declines a
/// linearisation carrying a Real-sorted variable (its case splits are
/// tautologies over `Z`, not over `R`). Both guards point the same way, and
/// this pins the outcome: a nonlinear *real* goal is decided by the
/// cell-decomposition core or not at all.
#[cfg(feature = "nlsat")]
#[test]
fn nra_verdicts_are_unchanged() {
    assert_eq!(
        run("(set-logic QF_NRA)(declare-const x Real)(assert (< (* x x) 0.0))(check-sat)"),
        vec!["unsat"],
        "still the cell decomposition's refutation, not the engine's"
    );
    assert_eq!(
        run("(set-logic QF_NRA)(declare-const x Real)(assert (= (* x x) 4.0))(check-sat)"),
        vec!["sat"],
        "and its sat is unchanged too"
    );
}
