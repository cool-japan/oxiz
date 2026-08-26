//! End-to-end pins for issue #40: a candidate model the soundness gate refutes
//! is now *blocked and retried*, not discarded.
//!
//! The gate (`Solver::model_refutes_assertions`) is the last thing between a
//! candidate assignment and a reported `sat`. When it fired, `check_core` threw
//! the model away and answered `unknown` on the spot — even though the search
//! was nowhere near exhausted and the formula might well have another model one
//! decision away. Two things changed:
//!
//! 1. The case-split and array-axiom **repair paths now run before the gate**.
//!    They exist to fix exactly this kind of candidate, and bailing out first
//!    made them unreachable for it.
//! 2. What survives both repairs and still fails the gate is **excluded by a
//!    blocking clause** and the search re-solves, up to a bounded number of
//!    rounds.
//!
//! Because the gate also fires on the *evaluator's* fixed-width arithmetic
//! giving up (`EvalOutcome::Unrepresentable`), a blocking clause is not entailed
//! by the assertions: it restricts the search rather than adding knowledge. So
//! while any such clause is live, an `unsat` from the SAT core is surfaced as
//! `unknown`. The verdict lattice is `unknown -> {sat, unknown}` and never
//! anything worse — which is what the negative cases below are for.
//!
//! Every formula here is written in-house from the diagnosis; none is copied
//! from an upstream benchmark.

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
        .expect("script should parse and run")
}

/// `2^62` is a fine `i64`; `2^62 + 2^62` is not. The evaluator's `EvalVal::Num`
/// is a `Rational64`, so the model `x = 2^62` makes `(+ x x)` report
/// `Unrepresentable` and the gate refuses to certify it — while `x = 1` is a
/// model it certifies without complaint.
const BIG: &str = "4611686018427387904";

/// The headline case. One disjunct leads to a candidate the evaluator cannot
/// certify, the other to a real model. Before the fix, whichever the search
/// reached first decided the verdict and an unlucky order cost a `sat`.
#[test]
fn refuted_candidate_is_retried_not_conceded() {
    let out = run(&format!(
        "(set-logic QF_LIA)
         (declare-const x Int)
         (assert (or (= x 1) (= x {BIG})))
         (assert (>= (+ x x) 0))
         (check-sat)
         (get-model)"
    ));
    assert_eq!(out[0], "sat", "the surviving candidate is a genuine model");
    assert!(
        out[1].contains("(define-fun x () Int 1)"),
        "and it is the one the gate certified, not the one it refused: {}",
        out[1]
    );
}

/// No candidate survives the gate: the loop must run out and concede, and the
/// concession must be `unknown`.
///
/// `unsat` here would be the wrong answer twice over — the formula is
/// satisfiable in the mathematical sense (the evaluator's width limit is the
/// only obstacle), and the refutation would rest on clauses no assertion
/// entails.
#[test]
fn no_certifiable_candidate_concedes_unknown() {
    let out = run(&format!(
        "(set-logic QF_LIA)
         (declare-const x Int)
         (assert (= x {BIG}))
         (assert (>= (+ x x) 0))
         (check-sat)"
    ));
    assert_eq!(out[0], "unknown");
}

/// The same, with several equally uncertifiable candidates: each is blocked in
/// turn and the loop still terminates inside its budget rather than spinning.
#[test]
fn several_refuted_candidates_still_terminate() {
    let out = run(&format!(
        "(set-logic QF_LIA)
         (declare-const x Int)
         (assert (or (= x {BIG}) (= x 4611686018427387905) (= x 4611686018427387906)))
         (assert (>= (+ x x) 0))
         (check-sat)"
    ));
    assert_eq!(out[0], "unknown");
}

/// Regression guard for the failure mode this fix could introduce: an ordinary
/// unsatisfiable goal, which never reaches a candidate model at all, must still
/// answer `unsat`. Nothing about blocking may leak into the common path.
#[test]
fn ordinary_unsat_is_unaffected() {
    let out = run("(set-logic QF_LIA)
         (declare-const x Int)
         (assert (> x 5))
         (assert (< x 3))
         (check-sat)");
    assert_eq!(out[0], "unsat");
}

/// And an ordinary satisfiable goal still answers `sat` with its model.
#[test]
fn ordinary_sat_is_unaffected() {
    let out = run("(set-logic QF_LIA)
         (declare-const x Int)
         (assert (> x 5))
         (assert (< x 8))
         (check-sat)
         (get-model)");
    assert_eq!(out[0], "sat");
    assert!(out[1].contains("define-fun x"), "{}", out[1]);
}

/// Blocking clauses are added at the SAT core's live scope, so `(pop)` retracts
/// them along with the assertions that provoked them — and the solver is free to
/// report `unsat` again afterwards.
///
/// If the counter that tracks live blocking clauses were not rolled back by
/// `pop`, this script would answer `unknown` for a plainly contradictory pair of
/// bounds.
#[test]
fn pop_retracts_the_restriction() {
    let out = run(&format!(
        "(set-logic QF_LIA)
         (declare-const x Int)
         (push 1)
         (assert (or (= x 1) (= x {BIG})))
         (assert (>= (+ x x) 0))
         (check-sat)
         (pop 1)
         (assert (> x 5))
         (assert (< x 3))
         (check-sat)"
    ));
    assert_eq!(out[0], "sat");
    assert_eq!(
        out[1], "unsat",
        "the blocking clauses went with the popped scope"
    );
}
