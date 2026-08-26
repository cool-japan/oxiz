//! End-to-end coverage for `define-fun-rec` / `define-funs-rec`.
//!
//! These scripts used to be rejected outright by the parser. They now run
//! through the fuel-bounded unfolding driver in `context::recfun`, so what has
//! to be pinned here is the *verdict contract*:
//!
//! * `unsat` is final and must be reached whenever the definition refutes the
//!   assertions;
//! * `sat` is published only for a certified model, and the certified cases
//!   below must actually reach it (an over-cautious driver that answered
//!   `unknown` everywhere would be useless);
//! * a definition that does not terminate must answer `unknown` **in finite
//!   time** — the `loop` test fails by timing out, which is exactly the
//!   regression it guards.
//!
//! `(get-assertions)` byte-identity has its own test: the driver asserts
//! instances straight onto the solver, and any leak of them into the context's
//! assertion list would show up there first.

use oxiz_solver::Context;

/// Run a script and return its responses.
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
        .unwrap_or_else(|e| panic!("script must execute: {e}\n---\n{script}"))
}

/// Run a script and return the single verdict it produces.
fn verdict(script: &str) -> String {
    let out = run(script);
    out.into_iter()
        .next()
        .unwrap_or_else(|| panic!("script produced no output\n---\n{script}"))
}

const FACT: &str = "(define-fun-rec fact ((n Int)) Int (ite (<= n 0) 1 (* n (fact (- n 1)))))";

const FIB: &str = "(define-fun-rec fib ((n Int)) Int \
                   (ite (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))";

const PARITY: &str = "(define-funs-rec \
                      ((is-even ((n Int)) Bool) (is-odd ((n Int)) Bool)) \
                      ((ite (<= n 0) true (is-odd (- n 1))) \
                       (ite (<= n 0) false (is-even (- n 1)))))";

// ---------------------------------------------------------------------
// Ground recursion: the answer is decided by the definition alone
// ---------------------------------------------------------------------

#[test]
fn fact_5_is_120() {
    assert_eq!(
        verdict(&format!("{FACT}(assert (= (fact 5) 120))(check-sat)")),
        "sat"
    );
}

#[test]
fn fact_5_is_not_121() {
    assert_eq!(
        verdict(&format!("{FACT}(assert (= (fact 5) 121))(check-sat)")),
        "unsat"
    );
}

#[test]
fn fact_3_is_not_greater_than_1000() {
    assert_eq!(
        verdict(&format!("{FACT}(assert (> (fact 3) 1000))(check-sat)")),
        "unsat"
    );
}

#[test]
fn fib_bounded_sat_and_unsat() {
    assert_eq!(
        verdict(&format!("{FIB}(assert (= (fib 7) 13))(check-sat)")),
        "sat"
    );
    assert_eq!(
        verdict(&format!("{FIB}(assert (= (fib 7) 14))(check-sat)")),
        "unsat"
    );
}

// ---------------------------------------------------------------------
// Mutual recursion
// ---------------------------------------------------------------------

#[test]
fn mutual_parity_even_four() {
    assert_eq!(
        verdict(&format!("{PARITY}(assert (is-even 4))(check-sat)")),
        "sat"
    );
}

#[test]
fn mutual_parity_cannot_be_both() {
    assert_eq!(
        verdict(&format!(
            "{PARITY}(assert (and (is-even 3) (is-odd 3)))(check-sat)"
        )),
        "unsat"
    );
}

// ---------------------------------------------------------------------
// Symbolic argument
// ---------------------------------------------------------------------

const SUM: &str = "(define-fun-rec sum ((n Int)) Int (ite (<= n 0) 0 (+ n (sum (- n 1)))))";

#[test]
fn symbolic_argument_solves_for_the_variable() {
    // Structural unfolding alone never decides this: whatever depth it reaches,
    // the solver can satisfy `sum(k) = 6` by giving the *boundary* application
    // whatever value the arithmetic needs, for any `k`. Reaching `sat` here
    // depends on the refinement loop — the certifier computes `sum` at the
    // rejected model's `k`, and those concrete instances rule that `k` out.
    let script = format!(
        "{SUM}(declare-const k Int)\
         (assert (and (>= k 0) (= (sum k) 6)))\
         (check-sat)(get-value (k))"
    );
    let out = run(&script);
    assert_eq!(out[0], "sat", "a certified model must be found");
    assert!(
        out[1].contains('3'),
        "the only non-negative solution of sum(k) = 6 is k = 3, got {}",
        out[1]
    );
}

#[test]
fn nonlinear_symbolic_argument_is_honestly_unknown() {
    // The `fact` analogue of the test above multiplies two unknowns, and this
    // workspace's integer arithmetic does not solve that: writing the same
    // unfolding out by hand, with no recursive definition anywhere, is
    // `unknown` too. So this pins a *solver* limitation, not a driver one — and
    // it pins that the driver reports it honestly instead of publishing the
    // uncertified `sat` the relaxation offers it.
    let script = format!(
        "{FACT}(declare-const n Int)\
         (assert (and (>= n 0) (= (fact n) 6)))\
         (check-sat)"
    );
    assert_eq!(verdict(&script), "unknown");
}

// ---------------------------------------------------------------------
// Non-termination and inconsistency: honest verdicts, never a hang
// ---------------------------------------------------------------------

#[test]
fn non_terminating_definition_answers_unknown_without_hanging() {
    // `loop` has no base case, so no finite unfolding decides it and no
    // evaluation of `loop(0)` terminates. Both budgets are finite, so the
    // driver must still return — this test fails by *timing out* if either
    // budget is ever removed.
    let script = "(define-fun-rec loop ((x Int)) Int (loop (+ x 1)))\
                  (assert (= (loop 0) 5))(check-sat)";
    assert_eq!(verdict(script), "unknown");
}

#[test]
fn inconsistent_definition_is_unsat() {
    // `f(x) = f(x) + 1` has no model at all: a single instance already
    // contradicts itself, so this must be `unsat`, not `unknown`.
    let script = "(define-fun-rec f ((x Int)) Int (+ (f x) 1))\
                  (assert (= (f 0) 0))(check-sat)";
    assert_eq!(verdict(script), "unsat");
}

// ---------------------------------------------------------------------
// State hygiene: instances must stay out of the user-visible state
// ---------------------------------------------------------------------

#[test]
fn get_assertions_is_unchanged_by_a_recfun_check() {
    let before = format!("{FACT}(assert (= (fact 4) 24))(get-assertions)");
    let after = format!("{FACT}(assert (= (fact 4) 24))(check-sat)(get-assertions)");
    let before_out = run(&before);
    let after_out = run(&after);
    assert_eq!(
        before_out[0], after_out[1],
        "the unfolded instances must never reach (get-assertions)"
    );
}

#[test]
fn two_consecutive_checks_agree() {
    let out = run(&format!(
        "{FACT}(assert (= (fact 5) 120))(check-sat)(check-sat)"
    ));
    assert_eq!(out[0], "sat");
    assert_eq!(
        out[1], out[0],
        "a second check-sat must not be perturbed by the first one's scratch scope"
    );
}

#[test]
fn definition_goes_out_of_scope_on_pop() {
    // Inside the scope the definition decides the verdict; after the pop it is
    // gone, so the same assertion about an *undefined* symbol is satisfiable.
    let out = run(&format!(
        "(push 1){FACT}(assert (= (fact 5) 121))(check-sat)(pop 1)\
         (declare-fun fact (Int) Int)(assert (= (fact 5) 121))(check-sat)"
    ));
    assert_eq!(
        out[0], "unsat",
        "the definition is in scope inside the push"
    );
    assert_eq!(out[1], "sat", "after the pop, fact is an unconstrained UF");
}

#[test]
fn unsat_core_query_after_a_recfun_unsat_does_not_panic() {
    // The scratch scope stays open on `unsat`, so the solver's assertion list
    // holds instances the context's does not. `(get-unsat-core)` indexes that
    // list, and has panicked on an index mismatch before.
    let out = run(&format!(
        "(set-option :produce-unsat-cores true){FACT}\
         (assert (! (= (fact 5) 121) :named a))(check-sat)(get-unsat-core)"
    ));
    assert_eq!(out[0], "unsat");
    assert!(!out[1].is_empty(), "a core response must be produced");
}

// ---------------------------------------------------------------------
// Assumption-guarded solves
// ---------------------------------------------------------------------
//
// `(check-sat-assuming ..)` and `(get-consequences ..)` do not go through
// `check_sat`; they funnel through the context's assumption-guarded check.
// That path has to run the unfolding driver too, or these two commands answer
// with the definition dropped — the same silent-wrong-answer the parser's old
// hard rejection existed to prevent, one command over.

#[test]
fn check_sat_assuming_respects_the_definition() {
    let out = run(&format!(
        "{FACT}(declare-const p Bool)\
         (assert (=> p (= (fact 5) 121)))\
         (check-sat-assuming (p))\
         (check-sat-assuming ((not p)))"
    ));
    assert_eq!(
        out[0], "unsat",
        "assuming p forces fact(5) = 121, which the definition refutes"
    );
    assert_eq!(
        out[1], "sat",
        "assuming (not p) leaves the implication idle"
    );
}

#[test]
fn check_sat_assuming_sees_applications_only_the_assumption_mentions() {
    // `(fact 5)` appears in no assertion at all — only in the assumption — so
    // this fails unless the assumptions are unfolded as roots too.
    let out = run(&format!(
        "{FACT}(declare-const q Bool)\
         (assert (= q (= (fact 5) 121)))\
         (check-sat-assuming (q))"
    ));
    assert_eq!(out[0], "unsat");
}

#[test]
fn get_consequences_respects_the_definition() {
    let out = run(&format!(
        "{FACT}(declare-const p Bool)\
         (assert (= p (= (fact 5) 120)))\
         (get-consequences () (p))"
    ));
    assert_eq!(out[0], "sat");
    assert!(
        out[1].contains(" p)"),
        "p is entailed because fact(5) really is 120, got {}",
        out[1]
    );
}

// ---------------------------------------------------------------------
// Model output
// ---------------------------------------------------------------------

#[test]
fn get_model_reports_the_recursive_definition() {
    let out = run(&format!(
        "{FACT}(assert (= (fact 5) 120))(check-sat)(get-model)"
    ));
    assert_eq!(out[0], "sat");
    assert!(
        out[1].contains("(define-fun-rec fact ((n Int)) Int "),
        "the model must echo the recursive definition, got:\n{}",
        out[1]
    );
}

#[test]
fn get_value_of_a_recursive_application() {
    let out = run(&format!(
        "{FACT}(assert (= (fact 5) 120))(check-sat)(get-value ((fact 5)))"
    ));
    assert_eq!(out[0], "sat");
    assert!(
        out[1].contains("120"),
        "(get-value ((fact 5))) must report 120, got {}",
        out[1]
    );
}

// ---------------------------------------------------------------------
// Nullary recursive definition
// ---------------------------------------------------------------------

#[test]
fn nullary_recursive_definition_saturates() {
    // `c = c + 1` is unsatisfiable on its own, and its unfolding closes after
    // one instance — a nullary definition has exactly one application. The
    // script never mentions `c` again, so this also pins that a nullary
    // definition's axiom is instantiated unconditionally rather than only when
    // some assertion happens to name it.
    let script = "(define-fun-rec c () Int (+ c 1))(check-sat)";
    assert_eq!(verdict(script), "unsat");
}

#[test]
fn nullary_recursive_definition_with_a_base_case_is_sat() {
    // The dual: a nullary definition that *is* satisfiable must not be
    // refuted by the unconditional instance.
    let script = "(define-fun-rec c () Int (+ c 0))(assert (= c 7))(check-sat)";
    assert_eq!(verdict(script), "sat");
}
