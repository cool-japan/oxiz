//! Round-4 re-fix pass 15 — the pins for what it closed beyond the holes
//! recheck 14 pinned (those are inverted in `round4_pass14_recheck_pins`).
//!
//! * §1 — `#P2b-75` at every sort a guard can compare a bound variable at:
//!   `Real` (both sides of a boundary and the open interval between two),
//!   bit-vectors (a negated equality over 256 points), a declared sort (a
//!   negated equality beside a `distinct`), a datatype with a field, and a
//!   guard under an `ite`; with the satisfiable controls each repair must
//!   keep deciding (the model of each replays).
//! * §2 — `#P2b-76` at every depth and through every route: literal lists
//!   several cells deep, one constant, two chained constants, a selector
//!   chain; the equal-list control stays `sat`.
//! * §3 — decision (67): a quantifier the SEARCH removed still puts the
//!   `sat` under the certificate — an `exists` Skolemised away, a negated
//!   universal whose binder is vacuous (recheck 14's `s00719` / `s02141`
//!   shapes).  The model published replays, or none is published.
//! * §4 — decisions (68) / (69): a `select` class used as an argument gets
//!   a value of its own, and `(get-value)` answers from the printed model —
//!   a compound over an application folds, a constructor is its own value,
//!   a `Real` keeps its decimal point.
//! * §5 — found by this pass's own z3-judged run of `g14a` (decision (70)):
//!   an `Int` term is an integer under every logic (`#P2b-79`), a refused
//!   quantified candidate that is not a function gets its Ackermann lemma,
//!   and a datatype-indexed read under a vacuous binder is certified.
//!
//! Every model is judged by REPLAY (its `define-fun` lines verbatim beside a
//! closed claim); no test installs a wall clock (decision (16)).

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// The last `sat`/`unsat`/`unknown` line, or `"none"`.
fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// Every `(define-fun …)` line of the first published `(model …)` response.
fn model_definitions(lines: &[String]) -> Option<String> {
    let model = lines
        .iter()
        .find(|line| line.trim_start().starts_with("(model"))?;
    let mut out = String::new();
    for raw in model.lines() {
        let trimmed = raw.trim();
        if trimmed.starts_with("(define-fun ") {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    Some(out)
}

/// Whether the response withheld the model (decision (54)(i)'s error).
fn withheld(lines: &[String]) -> bool {
    lines
        .iter()
        .any(|line| line.contains("model not certified"))
}

/// Replay a published model: its `define-fun` lines (after `prelude`) and one
/// closed claim.
fn replay(prelude: &str, definitions: &str, claim: &str) -> String {
    let script = format!("(set-logic ALL)\n{prelude}{definitions}(assert {claim})\n(check-sat)\n");
    verdict(&run(&script))
}

// ---------------------------------------------------------------------------
// §1. `#P2b-75` beyond the `Int` spellings of recheck 14.
// ---------------------------------------------------------------------------

/// `(name, script)`: each is unsatisfiable (z3 4.15.4 agrees), and every
/// build before re-fix pass 15 answered `sat`.
const GUARD_WRONG_SATS: &[(&str, &str)] = &[
    (
        "real_strict_guard",
        "(assert (forall ((q Real)) (=> (> q 7.0) false)))\n",
    ),
    (
        "real_open_interval",
        "(assert (forall ((q Real)) (=> (and (> q 1.0) (< q 2.0)) false)))\n",
    ),
    (
        "real_guard_above_a_constant",
        "(declare-const m Real)\n(assert (forall ((x Real)) (=> (> x m) false)))\n",
    ),
    (
        "real_two_regions_over_f",
        "(declare-fun f (Real) Real)\n\
         (assert (forall ((x Real)) (=> (> x 7.0) (= (f x) 1.0))))\n\
         (assert (forall ((x Real)) (=> (< x 7.5) (= (f x) 2.0))))\n",
    ),
    (
        "bv8_negated_equality",
        "(assert (forall ((x (_ BitVec 8))) (=> (not (= x #x03)) false)))\n",
    ),
    (
        "declared_sort_negated_equality",
        "(declare-sort U 0)\n(declare-const a U)\n(declare-const b U)\n\
         (assert (forall ((x U)) (=> (not (= x a)) false)))\n(assert (distinct a b))\n",
    ),
    (
        "int_two_regions_over_f",
        "(declare-fun f (Int) Int)\n\
         (assert (forall ((x Int)) (=> (> x 7) (= (f x) 1))))\n\
         (assert (forall ((x Int)) (=> (< x 9) (= (f x) 2))))\n",
    ),
    (
        "datatype_negated_equality",
        "(declare-datatypes ((Lst 0)) (((nil) (cons (hd Int) (tl Lst)))))\n\
         (assert (forall ((x Lst)) (=> (not (= x nil)) false)))\n",
    ),
    (
        "datatype_guard_over_len",
        "(declare-datatypes ((Lst 0)) (((nil) (cons (hd Int) (tl Lst)))))\n\
         (declare-fun len (Lst) Int)\n\
         (assert (forall ((x Lst)) (=> (not (= x nil)) (= (len x) 1))))\n\
         (assert (= (len (cons 1 nil)) 2))\n",
    ),
];

#[test]
fn guards_at_every_sort_get_an_instance_past_their_boundary() {
    for &(name, script) in GUARD_WRONG_SATS {
        let lines = run(&format!("(set-logic ALL)\n{script}(check-sat)\n"));
        assert_eq!(
            verdict(&lines),
            "unsat",
            "`{name}` is unsatisfiable (`#P2b-75`)\n{}",
            lines.join("\n")
        );
    }
}

/// `(name, declarations, assertions, closed claim)`: satisfiable guarded
/// goals the repair must keep deciding, each with the closed instances its
/// model must satisfy.
const GUARD_CONTROLS: &[(&str, &str, &str, &str)] = &[
    (
        "int_guard_beside_a_point_below_it",
        "(declare-fun f (Int) Int)\n",
        "(assert (forall ((x Int)) (=> (> x 7) (= (f x) 1))))\n(assert (= (f 3) 2))\n",
        "(and (= (f 8) 1) (= (f 100) 1) (= (f 3) 2))",
    ),
    (
        "real_open_interval_satisfiable",
        "(declare-fun f (Real) Real)\n",
        "(assert (forall ((x Real)) (=> (and (> x 1.0) (< x 2.0)) (= (f x) 0.0))))\n\
         (assert (= (f 3.0) 1.0))\n",
        "(and (= (f 1.5) 0.0) (= (f 1.9) 0.0) (= (f 3.0) 1.0))",
    ),
    (
        "int_negated_equality_satisfiable",
        "(declare-fun f (Int) Int)\n",
        "(assert (forall ((x Int)) (=> (not (= x 3)) (= (f x) 0))))\n(assert (= (f 3) 5))\n",
        "(and (= (f 2) 0) (= (f 4) 0) (= (f 3) 5))",
    ),
    (
        "bv8_negated_equality_satisfiable",
        "(declare-fun h ((_ BitVec 8)) (_ BitVec 8))\n",
        "(assert (forall ((x (_ BitVec 8))) (=> (not (= x #x03)) (= (h x) #x00))))\n\
         (assert (= (h #x03) #x01))\n",
        "(and (= (h #x02) #x00) (= (h #xff) #x00) (= (h #x03) #x01))",
    ),
    (
        "int_monotone_var_var",
        "(declare-fun f (Int) Int)\n",
        "(assert (forall ((x Int) (y Int)) (=> (<= x y) (<= (f x) (f y)))))\n\
         (assert (< (f 0) (f 5)))\n",
        "(and (<= (f 0) (f 5)) (<= (f 1) (f 2)) (< (f 0) (f 5)))",
    ),
    (
        "real_monotone_var_var",
        "(declare-fun f (Real) Real)\n",
        "(assert (forall ((x Real) (y Real)) (=> (<= x y) (<= (f x) (f y)))))\n\
         (assert (= (f 1.0) 2.0))\n(assert (< (f 0.0) (f 5.0)))\n",
        "(and (<= (f 0.0) (f 1.0)) (<= (f 1.0) (f 5.0)) (= (f 1.0) 2.0) (< (f 0.0) (f 5.0)))",
    ),
    (
        "guard_above_a_declared_constant",
        "(declare-fun f (Int) Int)\n(declare-const m Int)\n",
        "(assert (forall ((x Int)) (=> (> x m) (> (f x) 0))))\n(assert (= m 10))\n\
         (assert (< (f 5) 0))\n",
        "(and (> (f 11) 0) (> (f 50) 0) (= m 10) (< (f 5) 0))",
    ),
];

#[test]
fn satisfiable_guarded_goals_stay_decided_and_their_models_replay() {
    for &(name, decls, body, claim) in GUARD_CONTROLS {
        let lines = run(&format!(
            "(set-logic ALL)\n{decls}{body}(check-sat)\n(get-model)\n"
        ));
        assert_eq!(
            verdict(&lines),
            "sat",
            "`{name}` is satisfiable and every earlier build decided it\n{}",
            lines.join("\n")
        );
        // Certified or absent (decision (67)).  Every one of these models was
        // FALSIFYING on HEAD `c702310` (z3 replay): the search certifies the
        // universal through an interpretation its printed table is not, and
        // the table's else value — the most common entry value — broke it.
        // The certificate now tries the table's other entry values and the
        // goal's literals as its else value (`published::repair_else_values`)
        // and publishes the one it certifies.  A monotone `f` has no constant
        // else value (the interpretation is a step function), so those two
        // are withheld: the named residue of `#P2b-51`.
        if withheld(&lines) {
            assert!(
                MONOTONE_RESIDUE.contains(&name),
                "`{name}`: a certified model is published\n{}",
                lines.join("\n")
            );
            continue;
        }
        let definitions = model_definitions(&lines)
            .unwrap_or_else(|| panic!("`{name}`: a model\n{}", lines.join("\n")));
        assert_eq!(
            replay("", &definitions, claim),
            "sat",
            "`{name}`: the published model satisfies `{claim}`\n{}",
            lines.join("\n")
        );
    }
}

/// The controls no constant else value certifies (`#P2b-51`'s residue).
const MONOTONE_RESIDUE: &[&str] = &["int_monotone_var_var", "real_monotone_var_var"];

/// A datatype guard over a satisfiable goal: before this pass the
/// certificate accepted it without a representative (and, on the unsat
/// twin, certified a wrong `sat`); it is decided over a value no named term
/// denotes now (`sat_certify::datatype_points`).
#[test]
fn a_datatype_guard_is_certified_over_an_unnamed_value() {
    let lines = run("(set-logic ALL)\n\
         (declare-datatypes ((Lst 0)) (((nil) (cons (hd Int) (tl Lst)))))\n\
         (declare-fun len (Lst) Int)\n\
         (assert (forall ((x Lst)) (=> (not (= x nil)) (= (len x) 1))))\n\
         (assert (= (len nil) 0))\n(check-sat)\n");
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
}

// ---------------------------------------------------------------------------
// §2. `#P2b-76` at every depth and through every route.
// ---------------------------------------------------------------------------

const LIST: &str = "(declare-datatypes ((IntList 0)) (((nil) (cons (head Int) (tail IntList)))))\n\
     (declare-const l IntList)\n(declare-const l2 IntList)\n";

#[test]
fn constructor_injectivity_reaches_every_depth_and_route() {
    for (name, body) in [
        (
            "one_level_through_a_constant",
            "(assert (= l (cons 2 nil)))\n(assert (= l (cons 2 (cons 3 nil))))\n",
        ),
        (
            "four_cells_through_a_constant",
            "(assert (= l (cons 1 (cons 2 (cons 3 (cons 4 nil))))))\n\
             (assert (= l (cons 1 (cons 2 (cons 3 (cons 5 nil))))))\n",
        ),
        (
            "two_chained_constants",
            "(assert (= l (cons 1 l2)))\n(assert (= l2 (cons 2 nil)))\n\
             (assert (= l (cons 1 (cons 2 (cons 3 nil)))))\n",
        ),
        (
            "a_selector_chain",
            "(assert (= (tail l) (cons 2 nil)))\n(assert (= (tail l) (cons 2 (cons 3 nil))))\n",
        ),
        (
            "literal_three_levels",
            "(assert (= (cons 0 (cons 1 (cons 2 nil))) (cons 0 (cons 1 (cons 2 (cons 3 nil))))))\n",
        ),
    ] {
        let lines = run(&format!("(set-logic ALL)\n{LIST}{body}(check-sat)\n"));
        assert_eq!(
            verdict(&lines),
            "unsat",
            "`{name}`: two lists that differ in a cell are distinct (`#P2b-76`)\n{}",
            lines.join("\n")
        );
    }
    // The control: equal lists through two constants stay satisfiable, and
    // the model prints both.
    let lines = run(&format!(
        "(set-logic ALL)\n{LIST}(assert (= l (cons 1 (cons 2 nil))))\n\
         (assert (= l2 (cons 1 (cons 2 nil))))\n(assert (= l l2))\n(check-sat)\n(get-model)\n"
    ));
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let definitions = model_definitions(&lines).unwrap_or_default();
    assert_eq!(
        replay(
            "(declare-datatypes ((IntList 0)) (((nil) (cons (head Int) (tail IntList)))))\n",
            &definitions,
            "(and (= l (cons 1 (cons 2 nil))) (= l l2))"
        ),
        "sat",
        "{}",
        lines.join("\n")
    );
}

// ---------------------------------------------------------------------------
// §3. Decision (67): a quantifier the search removed is still certified.
// ---------------------------------------------------------------------------

#[test]
fn a_skolemised_exists_and_a_vacuous_binder_are_certified_as_printed() {
    for (name, decls, body, claim) in [
        (
            "s00719_skolemised_exists",
            "(declare-const b (Array Int Int))\n",
            "(assert (exists ((i Int)) (= (- 1) (select b i))))\n(assert (<= (select b 7) 5))\n",
            // A witness at every index the printed store chain names is
            // checked below; `b[7] <= 5` is closed.
            "(<= (select b 7) 5)",
        ),
        (
            "s02141_vacuous_binder",
            "(declare-fun p (Int) Bool)\n(declare-const a (Array Int Int))\n(declare-const j Int)\n",
            "(assert (not (forall ((i Int)) (not (p (select a j))))))\n",
            "(p (select a j))",
        ),
    ] {
        let lines = run(&format!(
            "(set-logic ALL)\n{decls}{body}(check-sat)\n(get-model)\n"
        ));
        assert_eq!(verdict(&lines), "sat", "`{name}`\n{}", lines.join("\n"));
        if withheld(&lines) {
            continue;
        }
        let definitions = model_definitions(&lines)
            .unwrap_or_else(|| panic!("`{name}`: a model\n{}", lines.join("\n")));
        assert_eq!(
            replay("", &definitions, claim),
            "sat",
            "`{name}`: the published model satisfies `{claim}`\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §4. Decisions (68) / (69): `select` classes and `(get-value)`.
// ---------------------------------------------------------------------------

#[test]
fn a_select_class_used_as_an_argument_gets_its_own_value() {
    // m07: `(select a k)` and `(select a j)` index `b` at two values the
    // search kept apart; before the pass both printed as `0`.
    let decls = "(declare-const a (Array Int Int))\n(declare-const b (Array Int Int))\n\
                 (declare-const k Int)\n(declare-const j Int)\n";
    let body = "(assert (= (select b (select a k)) 1))\n(assert (= (select b (select a j)) 2))\n";
    let lines = run(&format!(
        "(set-logic ALL)\n{decls}{body}(check-sat)\n(get-model)\n"
    ));
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let definitions = model_definitions(&lines).unwrap_or_default();
    assert_eq!(
        replay(
            "",
            &definitions,
            "(and (= (select b (select a k)) 1) (= (select b (select a j)) 2))"
        ),
        "sat",
        "{}",
        lines.join("\n")
    );
}

#[test]
fn get_value_answers_from_the_printed_model() {
    // gv01: every answer a value, and the values those of the printed model.
    let lines = run(
        "(set-logic ALL)\n(declare-fun f (Int) Int)\n(declare-fun g (Int) Int)\n\
         (declare-const j Int)\n(declare-const k Int)\n(declare-const b (Array Int Int))\n\
         (assert (= j 1))\n(assert (= (g k) 3))\n(check-sat)\n\
         (get-value ((+ (f j) 3) (select b (f 10)) (= (g k) (g j))))\n",
    );
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let answer = lines.last().cloned().unwrap_or_default();
    for echo in ["(+ (f j) 3) (+", "(select b (f 10)) (select", "(= 3 (g j))"] {
        assert!(
            !answer.contains(echo),
            "`(get-value)` folds every term to a value (decision (69)): {answer}"
        );
    }
    // A constructor is its own value, and a `Real` prints with its point.
    let lines = run(
        "(set-logic ALL)\n(declare-datatypes ((Color 0)) (((red) (green) (blue))))\n\
         (declare-const v Color)\n(declare-const r Real)\n(assert (= v red))\n(assert (= r 1.0))\n\
         (check-sat)\n(get-value (green blue v r))\n",
    );
    assert_eq!(verdict(&lines), "sat", "{}", lines.join("\n"));
    let answer = lines.last().cloned().unwrap_or_default();
    assert!(
        answer.contains("(green green)") && answer.contains("(blue blue)"),
        "a constructor evaluates to itself: {answer}"
    );
    assert!(answer.contains("(v red)"), "{answer}");
    assert!(
        answer.contains("(r 1.0)"),
        "a `Real` keeps its decimal point: {answer}"
    );
}

// ---------------------------------------------------------------------------
// §5. Decisions (67), (68): the query-free second certificate.
// ---------------------------------------------------------------------------

/// Goals the certificate's queries leave open and exact evaluation decides
/// (`context::model_fmt::printed_eval`): a negated universal (`g14a/s00344`), an
/// `exists` no named point witnesses (`s00755`), a vacuous binder over a
/// datatype (`s01092`), a nonlinear body over a bounded integer box
/// (`bench/extended_theories/UFLIA/05_quantified_lia_bounds`).  `c702310`
/// printed a correct model for each; re-fix pass 15's first build withheld it.
#[test]
fn goals_the_queries_leave_open_are_certified_by_evaluation() {
    for (name, script, claim) in [
        (
            "s00344_negated_universal",
            "(declare-const k Int)\n(declare-const j Int)\n\
             (assert (not (forall ((q Int)) (=> (< q 0) (= j k)))))\n(assert (not (<= j 1)))\n",
            "(and (not (= j k)) (not (<= j 1)))",
        ),
        (
            "s00755_exists_below_every_named_point",
            "(declare-const k Int)\n(declare-const j Int)\n\
             (assert (exists ((i Int)) (=> (>= i (- 3)) (distinct k 10))))\n(assert (= k 10))\n",
            "(= k 10)",
        ),
        (
            "s01092_vacuous_datatype_binder",
            "(declare-datatypes ((Color 0)) (((red) (green) (blue))))\n\
             (declare-fun f (Int) Int)\n(declare-const k Int)\n\
             (assert (exists ((xx Color)) (= k (f (- 1)))))\n",
            "(= k (f (- 1)))",
        ),
        (
            "nonlinear_body_over_a_bounded_box",
            "(declare-fun f (Int) Int)\n\
             (assert (forall ((x Int)) (=> (and (>= x 0) (<= x 10)) (<= (f x) (* x x)))))\n\
             (assert (= (f 3) 5))\n",
            "(and (= (f 3) 5) (<= (f 0) 0) (<= (f 10) 100) (<= (f 7) 49))",
        ),
    ] {
        let lines = run(&format!(
            "(set-logic ALL)\n{script}(check-sat)\n(get-model)\n"
        ));
        assert_eq!(verdict(&lines), "sat", "`{name}`\n{}", lines.join("\n"));
        assert!(
            !withheld(&lines),
            "`{name}`: exact evaluation certifies the printed model\n{}",
            lines.join("\n")
        );
        let definitions = model_definitions(&lines).unwrap_or_default();
        assert_eq!(
            replay("", &definitions, claim),
            "sat",
            "`{name}`: the published model satisfies `{claim}`\n{}",
            lines.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// §5. Found by this pass's own z3-judged run of recheck 14's `g14a`.
// ---------------------------------------------------------------------------

/// `#P2b-79`: an `Int` term is an integer under every logic.  Each script is
/// unsatisfiable (z3 4.15.4 agrees) and answered `sat` on 0.3.3, `c4b04b7`,
/// `c702310` and re-fix pass 14: under `(set-logic ALL)` or no logic the
/// arithmetic solver ran over the reals, so `2x = 1` had the solution `1/2`
/// and the model printed its numerator.
const INTEGRALITY_UNSAT: &[(&str, &str)] = &[
    (
        "all_logic",
        "(set-logic ALL)(declare-const x Int)(assert (= (* 2 x) 1))(check-sat)",
    ),
    (
        "no_logic",
        "(declare-const x Int)(assert (= (* 2 x) 1))(check-sat)",
    ),
    (
        "quantified_path",
        "(set-logic ALL)(declare-const x Int)(declare-fun f (Int) Int)\
         (assert (= (* 2 x) 1))(assert (forall ((i Int)) (>= (f i) (f i))))(check-sat)",
    ),
    (
        "open_interval",
        "(set-logic ALL)(declare-const x Int)(assert (< 0 (* 3 x)))(assert (< (* 3 x) 3))(check-sat)",
    ),
    (
        "parity",
        "(set-logic ALL)(declare-const x Int)(declare-const y Int)\
         (assert (= (* 2 x) (+ (* 2 y) 1)))(check-sat)",
    ),
    (
        "strict_between",
        "(set-logic ALL)(declare-const x Int)(declare-const y Int)\
         (assert (< x y))(assert (< y (+ x 1)))(check-sat)",
    ),
    (
        "mixed_with_a_real",
        "(set-logic ALL)(declare-const x Int)(declare-const r Real)\
         (assert (= (* 2.0 (to_real x)) r))(assert (= r 1.0))(check-sat)",
    ),
    // Every other kind of `Int` leaf the encoder registers (the marks are
    // made where it registers one), and the push/pop and assumption paths.
    (
        "application",
        "(set-logic ALL)(declare-fun f (Int) Int)(assert (= (* 2 (f 0)) 1))(check-sat)",
    ),
    (
        "array_read",
        "(set-logic ALL)(declare-const a (Array Int Int))(assert (= (* 2 (select a 0)) 1))(check-sat)",
    ),
    (
        "datatype_field",
        "(set-logic ALL)(declare-datatypes ((P 0)) (((mk (fst Int) (snd Int)))))(declare-const p P)\
         (assert (= (* 2 (fst p)) 1))(check-sat)",
    ),
    (
        "ite",
        "(set-logic ALL)(declare-const b Bool)(declare-const x Int)(declare-const y Int)\
         (assert (= (* 2 (ite b x y)) 1))(check-sat)",
    ),
    (
        "abs",
        "(set-logic ALL)(declare-const x Int)(assert (= (* 2 (abs x)) 1))(check-sat)",
    ),
    (
        "to_real",
        "(set-logic ALL)(declare-const x Int)(declare-const r Real)\
         (assert (= (to_real x) r))(assert (= (* 2.0 r) 1.0))(check-sat)",
    ),
    (
        "open_unit_interval",
        "(set-logic ALL)(declare-const x Int)(assert (and (> x 0) (< x 1)))(check-sat)",
    ),
    (
        "after_push_pop",
        "(set-logic ALL)(declare-const x Int)(push 1)(assert (= x 0))(check-sat)(pop 1)\
         (assert (= (* 2 x) 1))(check-sat)",
    ),
    (
        "check_sat_assuming",
        "(set-logic ALL)(declare-const x Int)(declare-const q Bool)\
         (assert (=> q (= (* 2 x) 1)))(check-sat-assuming (q))",
    ),
    (
        "universal_over_an_application",
        "(set-logic ALL)(declare-fun f (Int) Int)\
         (assert (forall ((i Int)) (= (* 2 (f i)) (+ (* 2 i) 1))))(check-sat)",
    ),
];

#[test]
fn an_int_term_is_an_integer_under_every_logic() {
    let mut wrong: Vec<String> = Vec::new();
    for (name, script) in INTEGRALITY_UNSAT {
        let got = verdict(&run(script));
        if got != "unsat" {
            wrong.push(format!("{name}: {got}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "integer-infeasible goals not refuted: {wrong:?}"
    );
}

/// The satisfiable controls of the pin above: each stays `sat`, and its model
/// — integers integral, the reals agreeing with them — replays.  (Each model's
/// reals are integral: a printed `(/ 1 2)` does not replay on this solver,
/// whose `/` is `#P2b-80`.)
#[test]
fn a_mixed_integer_goal_publishes_an_integral_model() {
    let cases: &[(&str, &str, &str)] = &[
        (
            "(declare-const x Int)(declare-const r Real)",
            "(assert (= (* 2.0 (to_real x)) r))(assert (> r 1.5))(assert (< r 2.5))",
            "(and (= (* 2.0 (to_real x)) r) (> r 1.5) (< r 2.5))",
        ),
        (
            "(declare-const x Int)(declare-const y Int)",
            "(assert (< (* 3 x) (* 2 y)))(assert (< (* 2 y) (+ (* 3 x) 2)))",
            "(and (< (* 3 x) (* 2 y)) (< (* 2 y) (+ (* 3 x) 2)))",
        ),
        (
            "(declare-const x Int)(declare-const r Real)",
            "(assert (< r 3.0))(assert (> r 1.0))(assert (= r (to_real x)))",
            "(and (< r 3.0) (> r 1.0) (= r (to_real x)))",
        ),
    ];
    for (declarations, assertions, claim) in cases {
        let lines = run(&format!(
            "(set-logic ALL)(set-option :produce-models true){declarations}{assertions}\
             (check-sat)(get-model)"
        ));
        assert_eq!(verdict(&lines), "sat", "{assertions}: {lines:?}");
        let Some(definitions) = model_definitions(&lines) else {
            panic!("{assertions}: no model published: {lines:?}");
        };
        assert_eq!(
            replay("", &definitions, claim),
            "sat",
            "{assertions}: the model does not replay: {definitions}"
        );
    }
}

/// Recheck 14's `g14a/s00761`, verbatim: the second check's candidate gives
/// `f(m)` and `f(5)` two values with `m = 5`, the ground gate refuses it, and
/// the check answered `unknown` on this tree where `c702310` and re-fix pass
/// 14 answered `sat`.  The refused candidate now gets its Ackermann lemma and
/// one more round; both checks answer `sat`, and the second model replays.
#[test]
fn a_refused_candidate_that_is_not_a_function_gets_its_lemma() {
    let script = "(set-logic ALL)(set-option :produce-models true)\
        (declare-const k Int)(declare-const j Int)(declare-const m Int)\
        (declare-fun f (Int) Int)(declare-const a (Array Int Int))\
        (assert (not (forall ((i Int)) (ite (xor (<= (f m) 10) (= k (- 3))) (=> (= j 5) (not (= j 1))) (<= j 1)))))\
        (check-sat)(get-model)(get-value ((f m) k k (- (f m) 10)))\
        (push 1)\
        (assert (ite (ite (distinct (ite (= m j) m k) (ite (= k j) j m)) (not (= m k)) (<= (f j) 0)) (=> (<= m 5) (not (<= k 10))) (= m 1)))\
        (assert (not (< k (f (f m)))))\
        (assert (forall ((q Int)) (=> (>= q (- 1)) (=> (= j 10) (= (f j) (select a (- 3)))))))\
        (check-sat)(get-model)(get-value (j k))";
    let lines = run(script);
    let verdicts: Vec<&str> = lines
        .iter()
        .map(String::as_str)
        .filter(|line| matches!(*line, "sat" | "unsat" | "unknown"))
        .collect();
    assert_eq!(verdicts, vec!["sat", "sat"], "{lines:?}");
    let last_model: Vec<String> = lines
        .iter()
        .rev()
        .find(|line| line.trim_start().starts_with("(model"))
        .cloned()
        .into_iter()
        .collect();
    let Some(definitions) = model_definitions(&last_model) else {
        panic!("no model published: {lines:?}");
    };
    let claim = "(and \
        (not (forall ((i Int)) (ite (xor (<= (f m) 10) (= k (- 3))) (=> (= j 5) (not (= j 1))) (<= j 1)))) \
        (ite (ite (distinct (ite (= m j) m k) (ite (= k j) j m)) (not (= m k)) (<= (f j) 0)) (=> (<= m 5) (not (<= k 10))) (= m 1)) \
        (not (< k (f (f m)))) \
        (forall ((q Int)) (=> (>= q (- 1)) (=> (= j 10) (= (f j) (select a (- 3)))))))";
    assert_eq!(
        replay("", &definitions, claim),
        "sat",
        "the second model does not replay: {definitions}"
    );
}

/// Recheck 14's `g14a/s01255` shape: a negated universal over a vacuous
/// binder reads a datatype-indexed array at a constructor value.  The
/// certificate's evaluator had no value for a datatype index, so the correct
/// model (`e` storing `6` at `(mk 0 red)`, `w = (mk 0 red)`) was withheld on
/// this tree where re-fix pass 14 printed it.  The read is now folded over
/// the printed store chain by value (`printed_eval::datatype_reads`): the
/// model is published and replays.
#[test]
fn a_datatype_indexed_read_under_a_vacuous_binder_is_certified() {
    let declarations = "(declare-datatypes ((Color 0) (Pair 0)) (((red) (green) (blue)) ((mk (fst Int) (snd Color)))))\
        (declare-const u Color)(declare-const w Pair)\
        (declare-const d (Array Color Int))(declare-const e (Array Pair Int))";
    let assertions = "(assert (not (forall ((xx Color)) (<= (select e w) 5))))\
        (assert (< (fst w) (select d u)))";
    let lines = run(&format!(
        "(set-logic ALL)(set-option :produce-models true){declarations}{assertions}(check-sat)(get-model)"
    ));
    assert_eq!(verdict(&lines), "sat", "{lines:?}");
    assert!(!withheld(&lines), "a correct model was withheld: {lines:?}");
    let Some(definitions) = model_definitions(&lines) else {
        panic!("no model published: {lines:?}");
    };
    let datatypes = "(declare-datatypes ((Color 0) (Pair 0)) (((red) (green) (blue)) ((mk (fst Int) (snd Color)))))\n";
    assert_eq!(
        replay(
            datatypes,
            &definitions,
            "(and (not (forall ((xx Color)) (<= (select e w) 5))) (< (fst w) (select d u)))"
        ),
        "sat",
        "the model does not replay: {definitions}"
    );
}

/// Recheck 14's `g14a/s02595`, first check: the congruence closure leaves
/// `(f j)` and `(f m)` apart with `j = m = -2`, so `f`'s printed table keeps
/// one value at `-2` (`#P2b-74`, read as printed on a quantified goal), and
/// `(get-value)` answered `(select a (f m))` from the other: `-2` beside a
/// `(get-model)` whose `f` and `a` give `0` (pre-existing on `c702310`).  A
/// quantified goal's compound terms are now answered from the printed model:
/// every answer replays beside the model's own definitions.
#[test]
fn get_value_of_a_quantified_goal_agrees_with_the_printed_model() {
    let script = "(set-logic ALL)(set-option :produce-models true)\
        (declare-const k Int)(declare-const j Int)(declare-const m Int)\
        (declare-fun f (Int) Int)(declare-fun g (Int) Int)(declare-fun p (Int) Bool)\
        (declare-const a (Array Int Int))\
        (assert (forall ((i Int)) (=> (= i 1) (= (f j) 10))))\
        (assert (or (not (<= k (- 1))) (= 2 m)))\
        (assert (forall ((q Int)) (=> (>= q (- 1)) (not (distinct (select a (select a j)) (f (ite (= (- 1) (- 1)) q k)))))))\
        (assert (not (p m)))\
        (check-sat)(get-model)";
    let queries = [
        "(ite (= m (select a m)) (select a m) m)",
        "(select a (f m))",
        "(g (f 1))",
        "(f j)",
        "(f m)",
    ];
    let mut full = script.to_string();
    for query in &queries {
        full.push_str(&format!("(get-value ({query}))"));
    }
    let lines = run(&full);
    assert_eq!(verdict(&lines), "sat", "{lines:?}");
    assert!(!withheld(&lines), "a correct model was withheld: {lines:?}");
    let Some(definitions) = model_definitions(&lines) else {
        panic!("no model published: {lines:?}");
    };
    for query in &queries {
        let prefix = format!("(({query} ");
        let Some(answer) = lines
            .iter()
            .find_map(|line| line.strip_prefix(&prefix)?.strip_suffix("))"))
        else {
            panic!("no (get-value) answer for {query}: {lines:?}");
        };
        assert_eq!(
            replay("", &definitions, &format!("(= {query} {answer})")),
            "sat",
            "(get-value ({query})) answered {answer}, which the printed model contradicts: {definitions}"
        );
    }
}

/// The regenerated corpus's `qeq120/q0005`: a universal whose body reads a
/// write at the bound variable, `(select (store a1 i #b1) #b0010111)`, is
/// refuted at `i = #b0010111` alone (`a2`'s read makes the right side `#b0`).
/// `c702310` and re-fix pass 14 answered `unsat` only because E-matching
/// matches its triggers against every interned term and a declined
/// certificate round had interned that instance's store; with nothing
/// interned on a declined round (`#P2b-75`'s eligibility-first certificate)
/// it answered `unknown`.  The counterexample search now takes the read's
/// index as a guard point of the write's (`guard_points`).
#[test]
fn a_read_over_a_write_at_the_bound_variable_is_refuted_at_the_read_index() {
    let script = "(set-logic ALL)\
        (declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\
        (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\
        (declare-const a2 (Array (_ BitVec 7) (_ BitVec 1)))\
        (assert (= a2 (store ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1) #b0011010 #b0)))\
        (assert (= (select a0 #b0011000) (bvxor (select a2 #b1110010) #b1)))\
        (assert (forall ((i (_ BitVec 7))) (= (select (store a1 i #b1) #b0010111) (bvxor (select a2 #b1100111) #b1))))\
        (check-sat)";
    assert_eq!(verdict(&run(script)), "unsat");
}

/// `#P2b-79`'s other direction: a branch-and-bound refutation raised under a
/// decision must block only that branch.  Each goal is satisfiable (z3 4.15.4
/// agrees) through a disjunct the fractional branch does not take — a
/// refutation clause too small would answer `unsat`.
#[test]
fn integer_branching_under_a_decision_keeps_every_integer_solution() {
    let cases: &[(&str, &str)] = &[
        (
            "or_with_an_integral_side",
            "(set-logic ALL)(declare-const x Int)(assert (or (= (* 2 x) 1) (= x 4)))(check-sat)",
        ),
        (
            "three_way_or",
            "(set-logic ALL)(declare-const x Int)(assert (or (= (* 2 x) 1) (= (* 3 x) 7) (> x 10)))\
             (assert (< x 12))(check-sat)",
        ),
        (
            "implication",
            "(set-logic ALL)(declare-const x Int)(declare-const p Bool)(assert (=> p (= (* 2 x) 1)))\
             (assert (or p (= x 3)))(check-sat)",
        ),
        (
            "after_a_refuted_scope",
            "(set-logic ALL)(declare-const x Int)(declare-const y Int)(push 1)(assert (= (* 2 x) 1))\
             (check-sat)(pop 1)(assert (or (= (* 2 x) (+ (* 2 y) 1)) (= x (+ y 5))))(check-sat)",
        ),
        (
            "beside_a_universal",
            "(set-logic ALL)(declare-const x Int)(declare-fun f (Int) Int)\
             (assert (or (= (* 2 x) 1) (= (f x) 7)))(assert (forall ((i Int)) (>= (f i) 0)))(check-sat)",
        ),
        (
            "open_interval_or_a_branch",
            "(set-logic ALL)(declare-const x Int)(declare-const y Int)(declare-const q Bool)\
             (assert (or (and (< 0 (* 3 x)) (< (* 3 x) 3)) (and q (= y (* 2 x)))))(assert (=> q (> y 5)))(check-sat)",
        ),
        (
            "parity_or_a_sum",
            "(set-logic ALL)(declare-const a Int)(declare-const b Int)(declare-const c Int)\
             (assert (or (= (+ (* 2 a) (* 4 b)) 3) (= (+ a b c) 1)))(assert (or (= (* 6 c) 5) (> c 0)))(check-sat)",
        ),
    ];
    let mut wrong: Vec<String> = Vec::new();
    for (name, script) in cases {
        let got = verdict(&run(script));
        if got != "sat" {
            wrong.push(format!("{name}: {got}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "satisfiable mixed goals not answered sat: {wrong:?}"
    );
}
