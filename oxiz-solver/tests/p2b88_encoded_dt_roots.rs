//! `#P2b-88` — the datatype axioms and the model builder read every assertion
//! as the SAT core encodes it (`Solver::encoded_dt_scan`,
//! `Solver::register_dt_assertion_root`).
//!
//! Numeric purification (`solver::encode::numeric_purification`) hoists the
//! `2` of `(h 2)` into a proxy `v` before an assertion is encoded, so the
//! search decides `((_ is cons) (h v))` beside `(= v 2)`.  The datatype axioms
//! read the assertions as written and axiomatised `(h 2)`, a term no encoded
//! clause names: its testers were free atoms, the model builder valued
//! `(h 2)` from them, and the model gate (`Solver::model_refutes_assertions`)
//! refused each candidate whose phantom tester disagreed with the encoded one.
//! A refusal costs a blocking round, after which a refutation can only surface
//! as `unknown` (`model_blocking`); a candidate the gate let through printed a
//! value nothing had decided, and the honesty net refused the model.  Re-fix
//! pass 18 had re-spelled the eliminated `ite`s alone (`#P2b-90`); reading the
//! encoding itself covers every rewrite of the pre-pass chain.
//!
//! Every script below answered `unknown` at some check on the tree before the
//! fix (`7e4647c`), and is decided now as z3 4.15.4 decides it, with no
//! candidate refused.  Every printed model is judged by exact evaluation
//! (`support/dt_eval.rs`), never by the solver under test.
//!
//! * §1 the smallest shapes: refutations through the proxy's congruence, a
//!   constructor literal that shares the purified numeral, a selector chain
//!   over two purified applications, and the satisfiable twins whose printed
//!   model read the phantom, push / pop included.
//! * §2 `gen_dt.py` seed 30100192 `d00401`, verbatim — decision (24a)'s (B′):
//!   both checks spent all 64 blocking rounds.  Its sibling `d00498` is
//!   `round4_pass18_recheck_pins::a_fresh_datatype_goal_pass_seventeen_decides_is_decided`.
//! * §3 HOLE — `#P2b-95`, a different mechanism this fix's measurement found:
//!   a numeric selector under an uninterpreted function has no theory value,
//!   so the printed function collides at the field's default (never wrong).
//! * §4 HOLE — `#P2b-96`, the model builder's rendering of a consistent search
//!   state as a falsifying model, pinned by property (never wrong) on
//!   `d00090`, which this fix's trajectory lands on it.
//!
//! No test installs a wall clock (decision (16)).

use oxiz_solver::Context;

#[path = "support/dt_eval.rs"]
mod dt_eval;
use dt_eval::{ModelReading, judge};

/// The script's responses (an error folded into one line).
fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// Every `sat` / `unsat` / `unknown` line, in order.
fn verdicts(lines: &[String]) -> Vec<String> {
    lines
        .iter()
        .filter(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .collect()
}

/// Every check of `script` answers `sat`, and every printed model holds.
fn assert_decided_with_models(name: &str, script: &str, checks: usize) {
    let lines = run(script);
    let got = judge(script, &lines);
    assert_eq!(
        got.len(),
        checks,
        "`{name}`: one verdict per check\n{}",
        lines.join("\n")
    );
    for (check, (verdict, reading)) in got.iter().enumerate() {
        assert_eq!(
            verdict,
            "sat",
            "`{name}` check {check} (z3: sat)\n{}",
            lines.join("\n")
        );
        assert_eq!(
            reading,
            &ModelReading::Holds,
            "`{name}` check {check}: the printed model holds\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §1. The smallest shapes.  `h` is an uninterpreted function into a list, so
//     every literal argument of it is purified.
// ---------------------------------------------------------------------------

/// `(h x)` is not purified (a variable is already an interface term), `(h 2)`
/// is: the refutation needs `x = 2` to reach `(h v)` by congruence and the
/// tester over `(h v)` to be the one the axioms constrain.  Refuted after one
/// blocking round before the fix, so `unknown`.
const CONGRUENCE: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(declare-const x Int)\n\
(assert (= x 2))\n\
(assert ((_ is cons) (h x)))\n\
(assert ((_ is nil) (h 2)))\n\
(check-sat)\n";

/// The `1` of `(cons 1 nil)` is the purified numeral too, so the encoded
/// constructor application is `(cons v nil)`, which no axiom spoke of.
const LITERAL_FIELD: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(assert (= (h 1) (cons 1 nil)))\n\
(assert (not (= (hd (h 1)) 1)))\n\
(check-sat)\n";

/// [`LITERAL_FIELD`] as one assertion: one purification covers both halves.
const LITERAL_FIELD_ONE_ASSERTION: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(assert (and (= (h 1) (cons 1 nil)) (not (= (hd (h 1)) 1))))\n\
(check-sat)\n";

/// A selector chain across two purified applications.
const SELECTOR_CHAIN: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(assert (= (tl (h 0)) (h 1)))\n\
(assert ((_ is cons) (h 0)))\n\
(assert ((_ is cons) (h 1)))\n\
(assert (= (hd (tl (h 0))) 7))\n\
(assert (not (= (hd (h 1)) 7)))\n\
(check-sat)\n";

/// A purified application under a constructor, read back through a selector.
const UNDER_A_CONSTRUCTOR: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(declare-const l1 L)\n\
(declare-const x Int)\n\
(assert (= l1 (cons 2 (h 2))))\n\
(assert ((_ is cons) (tl l1)))\n\
(assert (= (hd (h 2)) (+ x 1)))\n\
(assert (= x 4))\n\
(assert (not (= (hd (tl l1)) 5)))\n\
(check-sat)\n";

#[test]
fn a_datatype_refutation_through_a_purified_argument_is_reached() {
    for (name, script) in [
        ("congruence", CONGRUENCE),
        ("literal_field", LITERAL_FIELD),
        ("literal_field_one_assertion", LITERAL_FIELD_ONE_ASSERTION),
        ("selector_chain", SELECTOR_CHAIN),
        ("under_a_constructor", UNDER_A_CONSTRUCTOR),
    ] {
        let lines = run(script);
        assert_eq!(
            verdicts(&lines),
            vec!["unsat"],
            "`{name}` (z3: unsat)\n{}",
            lines.join("\n")
        );
    }
}

/// The smallest satisfiable shape: before the fix the candidate the gate let
/// through printed `(h 2)` from its phantom testers, and the net refused it.
const SELECTOR_VALUE: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(assert (= (hd (h 2)) 5))\n\
(assert ((_ is cons) (h 2)))\n\
(check-sat)\n\
(get-model)\n";

/// Both polarities of one tester across a pop: the root of the popped scope
/// leaves with it.  Before the fix the first check answered `unknown`.
const PUSH_POP: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(push 1)\n\
(assert ((_ is cons) (h 2)))\n\
(assert (= (hd (h 2)) 3))\n\
(check-sat)\n\
(get-model)\n\
(pop 1)\n\
(assert ((_ is nil) (h 2)))\n\
(check-sat)\n\
(get-model)\n";

/// [`SELECTOR_CHAIN`] without its last assertion.
const SELECTOR_CHAIN_SAT: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(assert (= (tl (h 0)) (h 1)))\n\
(assert ((_ is cons) (h 0)))\n\
(assert ((_ is cons) (h 1)))\n\
(assert (= (hd (tl (h 0))) 7))\n\
(check-sat)\n\
(get-model)\n";

/// [`UNDER_A_CONSTRUCTOR`] without its last assertion.
const UNDER_A_CONSTRUCTOR_SAT: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0)) (((nil) (cons (hd Int) (tl L)))))\n\
(declare-fun h (Int) L)\n\
(declare-const l1 L)\n\
(declare-const x Int)\n\
(assert (= l1 (cons 2 (h 2))))\n\
(assert ((_ is cons) (tl l1)))\n\
(assert (= (hd (h 2)) (+ x 1)))\n\
(assert (= x 4))\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn a_datatype_model_over_a_purified_argument_is_printed_and_holds() {
    assert_decided_with_models("selector_value", SELECTOR_VALUE, 1);
    assert_decided_with_models("push_pop", PUSH_POP, 2);
    assert_decided_with_models("selector_chain_sat", SELECTOR_CHAIN_SAT, 1);
    assert_decided_with_models("under_a_constructor_sat", UNDER_A_CONSTRUCTOR_SAT, 1);
}

// ---------------------------------------------------------------------------
// §2. `gen_dt.py` seed 30100192 `d00401`, verbatim: `(not ((_ is cons) (h 2)))`
//     and `(= nil (h 1))` (the `ite` over `(cons 1 l1)` folds to `1`).  Before
//     the fix every one of the first check's 65 candidates set the phantom
//     `((_ is cons) (h 2))` true and the gate refused it; the check answered
//     `unknown` once the 64 blocking rounds were spent, and the second, with
//     nothing asserted in between, answered the same (`c702310` and re-fix
//     pass 14 did not decide them either; re-fix pass 17 did).  z3: `sat`,
//     `sat`.
// ---------------------------------------------------------------------------

const D00401: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun h (Int) L)
(declare-fun g (Int) C)
(declare-fun k (L) Int)
(declare-fun q (C) Int)
(declare-fun w (Int) U)
(declare-fun v (L) L)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const l1 L)
(declare-const l2 L)
(declare-const l3 L)
(declare-const c1 C)
(declare-const c2 C)
(declare-const p P)
(declare-const u1 U)
(declare-const u2 U)
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))
(assert (not ((_ is cons) (h 2))))
(assert (distinct u2 u1))
(assert (distinct l3 (ite ((_ is cons) l2) (tl l2) nil)))
(assert (= nil (h (ite ((_ is cons) (cons 1 l1)) (hd (cons 1 l1)) 0))))
(assert (distinct blue (g x)))
(push 1)
(assert (<= (q c2) (ite ((_ is cons) (cons x (cons 0 l1))) (hd (cons x (cons 0 l1))) 0)))
(check-sat)
(get-model)
(check-sat)
(get-model)
"#;

#[test]
fn a_fresh_datatype_goal_that_spent_its_blocking_budget_is_decided() {
    assert_decided_with_models("d00401", D00401, 2);
}

// ---------------------------------------------------------------------------
// §3. HOLE — `#P2b-95`, found while measuring this fix (the tree before it, `7e4647c`, answers the same):
//     a numeric selector under an uninterpreted function.  `track_theory_vars` does not descend into an
//     application's arguments, so `(px p)` in `(h (px p))` has no arithmetic variable; the Ackermann
//     round (`solver::uf_consistency`) finds no value for it and never sees the collision, and the
//     rebuilt `p` prints the field as its sort default `0` — the point where `(h 0)` is `(cons 6 l3)`.
//     The printed `h` is not a function, the honesty net reads assertion 2 false, and the check answers
//     `unknown` (z3: `sat`).  An `Int` constant in the same position is given a fresh value by the
//     printer (`#P2b-71`) and decided — the control.  Never wrong.  THE HOLE IS CLOSED when the selector
//     shape answers `sat` with a model that holds: invert this pin then, and close `#P2b-95`.
// ---------------------------------------------------------------------------

const SELECTOR_UNDER_A_FUNCTION: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-fun h (Int) L)\n\
(declare-const l3 L)\n\
(declare-const p P)\n\
(assert (= (cons 6 l3) (h 0)))\n\
(assert (= (h (px p)) l3))\n\
(check-sat)\n\
(get-model)\n";

/// [`SELECTOR_UNDER_A_FUNCTION`] with an `Int` constant in place of `(px p)`.
const CONSTANT_UNDER_A_FUNCTION: &str = "(set-logic ALL)\n\
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))\n\
(declare-fun h (Int) L)\n\
(declare-const l3 L)\n\
(declare-const w Int)\n\
(assert (= (cons 6 l3) (h 0)))\n\
(assert (= (h w) l3))\n\
(check-sat)\n\
(get-model)\n";

#[test]
fn a_selector_under_an_uninterpreted_function_is_not_decided() {
    assert_decided_with_models("constant_under_a_function", CONSTANT_UNDER_A_FUNCTION, 1);
    let lines = run(SELECTOR_UNDER_A_FUNCTION);
    let got = judge(SELECTOR_UNDER_A_FUNCTION, &lines);
    assert_eq!(got.len(), 1, "{}", lines.join("\n"));
    let (verdict, reading) = &got[0];
    assert_ne!(
        verdict,
        "unsat",
        "a WRONG unsat (z3: sat)\n{}",
        lines.join("\n")
    );
    if verdict == "sat" {
        assert_eq!(
            reading,
            &ModelReading::Holds,
            "the printed model holds\n{}",
            lines.join("\n")
        );
        panic!(
            "THE HOLE IS CLOSED (`#P2b-95`): a selector under an uninterpreted function is decided \
             with a model that holds — invert this pin\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §4. HOLE — `#P2b-96`, pinned by property and not by a closing tripwire: the datatype model builder renders a
//     search state it did not falsify as a falsifying model.  `gen_dt.py` seed 30093154 `d00090`, verbatim: the
//     tree before this fix (`7e4647c`) printed three models z3 confirms; on this tree's trajectory the search
//     decides `(v (cons x l2))` is `nil` while the printed `v` maps the printed `(cons x l2)` to a cons cell, so
//     the second assertion reads false and every check answers `unknown` with the net's reason (z3: `sat` ×3).
//     Which check a trajectory lands on this mechanism is not a property of the script — a tripwire here would
//     trip on another host the way `d00498`'s did — so the pin is what must hold on any trajectory: never
//     `unsat`, every printed model holds, every check without one names its reason.
// ---------------------------------------------------------------------------

const D00090: &str = r#"(set-logic ALL)
(declare-datatypes ((L 0) (C 0) (P 0)) (((nil) (cons (hd Int) (tl L))) ((red) (green) (blue)) ((mk (px Int) (pc C)))))
(declare-sort U 0)
(declare-fun h (Int) L)
(declare-fun g (Int) C)
(declare-fun k (L) Int)
(declare-fun q (C) Int)
(declare-fun w (Int) U)
(declare-fun v (L) L)
(declare-const x Int)
(declare-const y Int)
(declare-const z Int)
(declare-const l1 L)
(declare-const l2 L)
(declare-const l3 L)
(declare-const c1 C)
(declare-const c2 C)
(declare-const p P)
(declare-const u1 U)
(declare-const u2 U)
(assert (and (<= (- 4) x 4) (<= (- 4) y 4) (<= (- 4) z 4)))
(assert (and (and (distinct (v (v l2)) (ite ((_ is cons) (v (cons x l2))) (tl (v (cons x l2))) nil)) (= c1 red)) (distinct l1 (v l2))))
(assert (distinct green red))
(assert (distinct l3 l2))
(assert (= (ite ((_ is cons) nil) (tl nil) nil) (h y)))
(assert (= (ite ((_ is cons) (cons y l1)) (tl (cons y l1)) nil) (h 0)))
(check-sat)
(get-model)
(check-sat)
(get-model)
(push 1)
(assert (<= 0 (+ 2 (- 1))))
(check-sat)
(get-model)
"#;

#[test]
fn the_rendering_hole_is_never_wrong() {
    let lines = run(D00090);
    let got = judge(D00090, &lines);
    assert_eq!(got.len(), 3, "{}", lines.join("\n"));
    for (check, (verdict, reading)) in got.iter().enumerate() {
        assert_ne!(
            verdict,
            "unsat",
            "check {check}: a WRONG unsat (z3: sat)\n{}",
            lines.join("\n")
        );
        if verdict == "sat" {
            assert!(
                matches!(reading, ModelReading::Holds | ModelReading::Withheld),
                "check {check}: a printed model holds or is withheld: {reading:?}\n{}",
                lines.join("\n")
            );
        }
    }
    let missing = lines
        .iter()
        .filter(|line| line.contains("model not certified"))
        .count();
    let decided = got
        .iter()
        .filter(|(verdict, reading)| verdict == "sat" && *reading == ModelReading::Holds)
        .count();
    assert_eq!(
        missing + decided,
        3,
        "every check prints a model that holds or names why it has none\n{}",
        lines.join("\n")
    );
}
