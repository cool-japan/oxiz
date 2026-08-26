//! What the `nlsat` feature does, and — more importantly — what it does *not*
//! do to the rest of the solver.
//!
//! `nlsat` is on by default, so the interesting build is the other one:
//!
//! ```text
//! cargo test -p oxiz-solver --no-default-features --features std,property-tests
//! ```
//!
//! which drops the `oxiz-nlsat` crate from the dependency graph entirely (the
//! wasm32 size case this feature was cut for). Three claims have to hold there,
//! and each is asserted below:
//!
//! 1. **Nothing outside nonlinear arithmetic changes.** Boolean, QF_UF and
//!    QF_LIA solving, models and unsat cores are identical to a default build.
//!    Those tests carry no `cfg` at all: they run in both builds and must
//!    produce the same answers, which is what makes them a no-regression proof
//!    rather than two separate expectations.
//! 2. **What is lost is lost honestly.** A goal that needed the
//!    cell-decomposition core answers `unknown` — the same answer this codebase
//!    already gives for a String or FP atom no theory can take (`check_core`'s
//!    honesty gates). It must never fall through to the SAT layer, which would
//!    treat the nonlinear atom as a free Boolean and report a spurious verdict,
//!    and it must never panic.
//! 3. **What survives, survives soundly.** Three things outside `oxiz-nlsat`
//!    still decide nonlinear goals and stay compiled in either way — they are
//!    `std`-gated, and `std` is exactly what the OFF build keeps:
//!
//!    * the static UNSAT pattern detector (`check_nonlinear_constraints`);
//!    * the NIA-over-LP relaxation engine
//!      (`oxiz_theories::arithmetic::nla`), the only one of the three that can
//!      derive `unsat` from a *proof* rather than a syntactic pattern — an LP
//!      infeasibility closure over consequences of the input;
//!    * the two model searches (`nl_repair_search`, `nl_ground_reduce`).
//!
//!    Every `sat` any of them produces is re-verified against the untouched
//!    assertions in exact `BigRational` arithmetic before it is reported
//!    (`adopt_nl_witness` → `nl_eval::holds_under`). All three are gated on
//!    QF_NIA, so the whole QF_NIA group below is *also* uncfg'd: it answers the
//!    same in both builds.
//!
//! Every expectation here was measured against this tree in both feature
//! combinations rather than assumed. Three facts worth stating because they are
//! the ones a reader is most likely to guess wrong:
//!
//! * `(= (* x x) 2.0)` answers `unknown` in **both** builds. The irrational
//!   root is not isolated even with the feature on (`TODO.md`, the
//!   `oxiz-nlsat/src/solver/decide.rs` entry), so it is useless as a probe of
//!   this feature and is deliberately not used as one.
//! * `(get-value ...)` renders one binding per line, so a two-variable answer
//!   contains a newline. The expectations below spell that out.
//! * A **model** for a multi-solution QF_NIA goal is *not* stable across the
//!   two builds, even where the verdict is. The NIA-over-LP relaxation engine
//!   runs above the two model searches, so which procedure answers first — and
//!   therefore which witness is reported — differs once the cell decomposition
//!   is absent: `x*y = 6 ∧ x+y = 5` gives (2, 3) with `nlsat` and (3, 2)
//!   without. Both are re-verified against the assertions and both are correct.
//!   That is why every QF_NIA expectation below is a *verdict*, and why the
//!   exact `get-value` strings pinned in this file are all for goals with a
//!   unique model (QF_LIA) or decided by the cell decomposition (QF_NRA).

use oxiz_solver::Context;
// Most tests here go through `execute_script` and compare rendered lines; the
// two that reach the programmatic API instead do so to keep a change in the
// parser or the script runner from quietly turning them into tautologies. One
// of those two is uncfg'd, so this import is as well.
use oxiz_solver::SolverResult;

/// Run an SMT-LIB2 script the way a consumer does, returning the output lines.
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(e) => panic!("script failed to execute: {e}"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// (a) The core stays correct without the nonlinear solver.
//     No `cfg`: these run in both builds and must agree.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn bool_sat_and_unsat_are_unaffected() {
    assert_eq!(
        run("(declare-const p Bool)(assert p)(assert (not p))(check-sat)"),
        vec!["unsat"],
        "p AND NOT p is unsat with or without nlsat"
    );
    assert_eq!(
        run("(declare-const p Bool)(declare-const q Bool)\
             (assert (or p q))(assert (not p))(check-sat)(get-value (q))"),
        vec!["sat", "((q true))"],
        "the one satisfying assignment must still be found and reported"
    );
    // Pigeonhole-flavoured: three Booleans, at most one true, at least two true.
    assert_eq!(
        run(
            "(declare-const a Bool)(declare-const b Bool)(declare-const c Bool)\
             (assert (not (and a b)))(assert (not (and a c)))(assert (not (and b c)))\
             (assert a)(assert b)(check-sat)"
        ),
        vec!["unsat"],
        "at-most-one plus two forced trues is unsat"
    );
}

#[test]
fn qf_lia_stays_correct() {
    assert_eq!(
        run("(set-logic QF_LIA)(declare-const x Int)\
             (assert (> x 3))(assert (< x 5))(check-sat)(get-value (x))"),
        vec!["sat", "((x 4))"],
        "3 < x < 5 over the integers pins x to 4"
    );
    assert_eq!(
        run("(set-logic QF_LIA)(declare-const x Int)\
             (assert (> x 3))(assert (< x 3))(check-sat)"),
        vec!["unsat"],
        "x > 3 AND x < 3 is unsat"
    );
    assert_eq!(
        run(
            "(set-logic QF_LIA)(declare-const x Int)(declare-const y Int)\
             (assert (= (+ x y) 10))(assert (= (- x y) 4))(check-sat)(get-value (x y))"
        ),
        // One binding per line — the `get-value` renderer's own format, pinned
        // here so a change to it shows up as a visible diff rather than a
        // silent one.
        vec!["sat", "((x 7)\n (y 3))"],
        "a determined 2x2 linear system still solves to its unique model"
    );
}

#[test]
fn qf_uf_unsat_core_stays_correct() {
    // The shape the /oxiz demo's shift board uses: named assertions, an unsat
    // verdict, and a core naming the assertions responsible.
    let lines = run("(set-option :produce-unsat-cores true)(set-logic QF_UF)\
         (declare-const p Bool)(declare-const q Bool)\
         (assert (! (or p q) :named atleast))\
         (assert (! (not p) :named nop))\
         (assert (! (not q) :named noq))\
         (check-sat)(get-unsat-core)");
    assert_eq!(lines.first().map(String::as_str), Some("unsat"));
    let core = lines.get(1).map(String::as_str).unwrap_or("");
    for name in ["atleast", "nop", "noq"] {
        assert!(core.contains(name), "core {core:?} should name {name}");
    }
}

#[test]
fn linear_real_arithmetic_stays_correct() {
    // QF_NRA's *linear* fragment is still fully decided without the feature:
    // dropping `nlsat` costs nonlinear goals, not the logic label.
    assert_eq!(
        run("(set-logic QF_NRA)(declare-const x Real)\
             (assert (> x 1.0))(assert (< x 1.0))(check-sat)"),
        vec!["unsat"],
        "x > 1 AND x < 1 is unsat by linear reasoning alone"
    );
}

#[test]
fn qf_nia_is_decided_identically_in_both_builds() {
    // Claim 3, and the reason it is uncfg'd: everything that decides these
    // goals survives the feature being turned off. The `sat`s come from
    // `nl_repair_search` / `nl_ground_reduce`, whose witnesses are re-checked
    // against the original assertions before the verdict is reported; the
    // `unsat`s come from `check_nonlinear_constraints`' static patterns and
    // from the relaxation engine's proofs. None of those lives in
    // `oxiz-nlsat`.
    //
    // The property survives the engine being wired in, but the *pins* were
    // re-measured rather than assumed: the two multivariate goals at the end
    // answered `unknown` in both builds before the wiring and answer `unsat` in
    // both builds after it, which is the engine's contribution and is pinned
    // here as a both-builds claim rather than left to `qf_nia_relaxation.rs`
    // (which measures it against the flag instead). The four goals above them
    // did not move in either build.
    assert_eq!(
        run("(set-logic QF_NIA)(declare-const x Int)(assert (= (* x x) 4))(check-sat)"),
        vec!["sat"],
        "x*x = 4 has the verified integer witness x = ±2"
    );
    assert_eq!(
        run(
            "(set-logic QF_NIA)(declare-const x Int)(declare-const y Int)\
             (assert (= (* x y) 6))(assert (= (+ x y) 5))(check-sat)"
        ),
        vec!["sat"],
        "x*y = 6 AND x+y = 5 has the verified integer witness (2, 3)"
    );
    assert_eq!(
        run("(set-logic QF_NIA)(declare-const x Int)(assert (= (* x x) 3))(check-sat)"),
        vec!["unsat"],
        "3 is not a perfect square, and the static patterns say so either way"
    );
    assert_eq!(
        run("(set-logic QF_NIA)(declare-const x Int)(assert (= (* x x) (- 1)))(check-sat)"),
        vec!["unsat"],
        "a square is never negative, and the static patterns say so either way"
    );
    // Re-measured after the relaxation engine was wired in: `unknown` in both
    // builds before, `unsat` in both builds now. Neither is reachable by a
    // single-variable square pattern, so these two are the engine talking and
    // nothing else.
    assert_eq!(
        run(
            "(set-logic QF_NIA)(declare-const x Int)(declare-const y Int)\
             (assert (>= x 3))(assert (>= y 3))(assert (<= (* x y) 8))(check-sat)"
        ),
        vec!["unsat"],
        "two integer factors of at least 3 cannot multiply to 8 or less"
    );
    assert_eq!(
        run(
            "(set-logic QF_NIA)(declare-const x Int)(declare-const y Int)\
             (assert (= (+ (* x x) (* y y) 1) 0))(check-sat)"
        ),
        vec!["unsat"],
        "a sum of two squares plus one is never zero"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// (b) The honest failure, and (c) its counterpart in the default build.
//
// QF_NRA is where the two builds part company: the model searches above are
// gated on QF_NIA, so with the feature off there is nothing left to decide a
// nonlinear *real* goal with.
// ─────────────────────────────────────────────────────────────────────────────

/// `x*x = 4` over the reals — satisfiable at ±2, with a rational witness the
/// cell-decomposition core finds.
const NRA_SQUARE_FOUR: &str =
    "(set-logic QF_NRA)(declare-const x Real)(assert (= (* x x) 4.0))(check-sat)";

/// `x*y = 6 AND x+y = 5` over the reals — satisfiable at (2, 3).
const NRA_PRODUCT_SUM: &str = "(set-logic QF_NRA)(declare-const x Real)(declare-const y Real)\
     (assert (= (* x y) 6.0))(assert (= (+ x y) 5.0))(check-sat)";

/// `x*x < 0` over the reals — unsatisfiable, and provably so by the sign
/// reasoning the cell-decomposition core carries.
const NRA_SQUARE_NEGATIVE: &str =
    "(set-logic QF_NRA)(declare-const x Real)(assert (< (* x x) 0.0))(check-sat)";

#[cfg(not(feature = "nlsat"))]
#[test]
fn nra_goals_answer_unknown_without_the_feature() {
    // The honest failure. NOT `sat`, NOT `unsat`, NOT a panic: `unknown`, the
    // same answer this solver already gives for a String or FP atom it has no
    // complete theory for. Note that `NRA_SQUARE_NEGATIVE` is genuinely unsat
    // and this build still declines to say so — the cost is completeness, and
    // conceding it is the point.
    for script in [NRA_SQUARE_FOUR, NRA_PRODUCT_SUM, NRA_SQUARE_NEGATIVE] {
        assert_eq!(
            run(script),
            vec!["unknown"],
            "without `nlsat` a nonlinear real goal must concede, not guess: {script}"
        );
    }
}

#[cfg(not(feature = "nlsat"))]
#[test]
fn a_conceded_goal_declines_get_value_rather_than_panicking() {
    // `panic = "abort"` is the release profile, and the wasm consumer this
    // feature was cut for cannot catch a panic at all: an `unknown` that
    // poisoned the instance on the next `(get-value ...)` would be a worse
    // failure than the missing answer. The renderer must decline in-band.
    let lines = run("(set-logic QF_NRA)(declare-const x Real)\
         (assert (= (* x x) 4.0))(check-sat)(get-value (x))");
    assert_eq!(lines.first().map(String::as_str), Some("unknown"));
    let second = lines.get(1).map(String::as_str).unwrap_or("");
    assert!(
        second.starts_with("(error "),
        "get-value after an unknown should render an in-band error, got {second:?}"
    );
}

#[cfg(not(feature = "nlsat"))]
#[test]
fn nra_unknown_is_reached_through_the_programmatic_api_too() {
    // The same claim without the SMT-LIB2 layer in the way, so a change to the
    // parser or the script runner cannot quietly turn this test into a
    // tautology.
    let mut ctx = Context::new();
    ctx.set_logic("QF_NRA");
    let real_sort = ctx.terms.sorts.real_sort;
    let x = ctx.declare_const("x", real_sort);
    let square = ctx.terms.mk_mul([x, x]);
    let four = ctx.terms.mk_real(num_rational::Rational64::new(4, 1));
    let eq = ctx.terms.mk_eq(square, four);
    ctx.assert(eq);
    assert!(
        matches!(ctx.check_sat(), SolverResult::Unknown),
        "x*x = 4 over the reals must answer Unknown when the nonlinear solver \
         is not compiled in"
    );
}

#[test]
fn push_pop_over_a_nonlinear_atom_never_guesses() {
    // The uncfg'd counterpart of `nlsat_integration.rs`'s
    // `test_nia_push_pop_backtrack`, which is `cfg`'d to the default build.
    //
    // The sequence is `x*x = 4`, push, `x < 0`, push, `x > 0`, pop, pop. The
    // last-but-one state is contradictory on its *linear* atoms alone, and both
    // builds now report `Unsat` for it — which is why this test carries no
    // `cfg` any more.
    //
    // It used to. Before the NIA-over-LP relaxation engine was wired into
    // `dispatch_nl_solver`, the OFF build reached this state with nothing left
    // that could refute it: `dispatch_nl_solver` fell through, `check_core`'s
    // honesty gate spoke, and the answer was `Unknown` — correct, since the
    // solver had proven nothing while `x*x = 4` was still in scope. The engine
    // is `std`-gated rather than `nlsat`-gated, so it is now present in this
    // build too, and it *proves* the contradiction rather than conceding it.
    // The re-pin is a measured improvement, not a relaxed expectation: the
    // verdict moved from "declined" to "refuted", never from one verdict to a
    // different one.
    //
    // What this test exists to catch is the third outcome — the honesty gate
    // being skipped and the SAT layer treating `x*x = 4` as a free Boolean,
    // which would report `Sat` for `x < 0 ∧ x > 0`. That bug is excluded at
    // every level below, not just the contradictory one, and the `!Sat`
    // assertion is deliberately kept separate from the `Unsat` one so a future
    // re-measurement can loosen the second without touching the first.
    let mut ctx = Context::new();
    ctx.set_logic("QF_NIA");
    let int_sort = ctx.terms.sorts.int_sort;
    let x = ctx.declare_const("x", int_sort);
    let square = ctx.terms.mk_mul([x, x]);
    let four = ctx.terms.mk_int(4);
    let eq = ctx.terms.mk_eq(square, four);
    ctx.assert(eq);

    // Level 0 — `x*x = 4`. Still decided, by a re-verified witness.
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));

    ctx.push();
    let zero = ctx.terms.mk_int(0);
    let x_lt = ctx.terms.mk_lt(x, zero);
    ctx.assert(x_lt);
    // Level 1 — `x*x = 4 ∧ x < 0`. Also still decided: x = -2.
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));

    ctx.push();
    let x_gt = ctx.terms.mk_gt(x, zero);
    ctx.assert(x_gt);
    // Level 2 — now contradictory. Refuted, not guessed.
    let at_conflict = ctx.check_sat();
    assert!(
        !matches!(at_conflict, SolverResult::Sat),
        "x < 0 AND x > 0 must never be reported Sat, got {at_conflict:?}"
    );
    assert!(
        matches!(at_conflict, SolverResult::Unsat),
        "both builds refute this: the relaxation engine is compiled in either \
         way, got {at_conflict:?}"
    );

    // And the scopes still unwind to the answers they had on the way in.
    ctx.pop();
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));
    ctx.pop();
    assert!(matches!(ctx.check_sat(), SolverResult::Sat));
}

#[cfg(feature = "nlsat")]
#[test]
fn nra_goals_are_decided_with_the_feature() {
    // The companion measurement: proof that the three `unknown`s above are the
    // feature's absence talking, not a solver that never decided these goals.
    assert_eq!(
        run(NRA_SQUARE_FOUR),
        vec!["sat"],
        "the default build decides x*x = 4 over the reals"
    );
    assert_eq!(
        run(NRA_PRODUCT_SUM),
        vec!["sat"],
        "the default build decides x*y = 6 AND x+y = 5 over the reals"
    );
    assert_eq!(
        run(NRA_SQUARE_NEGATIVE),
        vec!["unsat"],
        "the default build refutes x*x < 0 over the reals"
    );
}

#[cfg(feature = "nlsat")]
#[test]
fn a_decided_goal_still_carries_its_model() {
    // The other half of what the feature buys: not just the verdict, but a
    // model the caller can read back.
    assert_eq!(
        run("(set-logic QF_NRA)(declare-const x Real)\
             (assert (= (* x x) 4.0))(check-sat)(get-value (x))"),
        vec!["sat", "((x -2))"],
        "the default build reports a root of x*x = 4"
    );
}
