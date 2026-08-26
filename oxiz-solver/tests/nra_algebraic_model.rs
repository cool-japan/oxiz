//! `(get-model)` / `(get-value ..)` for QF_NRA models whose witness is an
//! irrational algebraic number.
//!
//! # Where the expected strings come from
//!
//! Every `root-obj` string asserted below was **captured from `z3 4.15.4`**,
//! not invented: each test names the goal it came from, and running that goal
//! through `z3` reproduces the string character-for-character. Z3 is the
//! reference implementation this project reimplements, so its spelling of
//! `root-obj` — descending monomials, elided unit coefficients, `(- n)` for
//! negative integers, and the bound variable always printed `x` whatever the
//! SMT constant is called — is the specification here.
//!
//! # What "byte-exact" covers, honestly
//!
//! `(get-value ..)` is byte-exact against Z3 and is asserted as a whole line.
//!
//! `(get-model)` is **not**, and no test here claims it is: this solver wraps
//! its entries in `(model …)` and puts each `define-fun` on one line, whereas
//! Z3 4.15.4 emits a bare `( … )` and breaks the body onto its own line. That
//! difference predates algebraic values, applies to every model this solver
//! prints, and changing it would move every other model test in the tree. What
//! *is* pinned byte-exact for `(get-model)` is the **value string** — the
//! `root-obj` term itself — which is the part this work is responsible for.

#![cfg(feature = "nlsat")]

use oxiz_solver::Context;

/// Run a script in a fresh context and return its output lines.
fn run(script: &str) -> Vec<String> {
    Context::new()
        .execute_script(script)
        .expect("the script must execute")
}

/// The `n`th response line of a script's output.
///
/// One entry per response-producing command, in order — `execute_script`
/// parses the whole script in one pass (a declaration does not carry across
/// separate calls), so an incremental test indexes the response list rather
/// than splitting the joined text, which cannot tell a `sat` inside `unsat`
/// from a verdict of its own.
fn response(output: &[String], index: usize) -> &str {
    output
        .get(index)
        .unwrap_or_else(|| panic!("expected at least {} responses, got {output:?}", index + 1))
}

/// `(get-value (x))` for `x² = 2`, exactly as `z3 4.15.4` prints it.
///
/// Z3 picks the *first* real root of `x² − 2` — i.e. `−√2` — when nothing
/// constrains the sign, and this solver's cell decomposition independently
/// picks the same one.
const Z3_NEGATIVE_BRANCH_GET_VALUE: &str = "((x (root-obj (+ (^ x 2) (- 2)) 1)))";

/// The same for `x² = 2 ∧ x > 0`: the *second* real root, `+√2`.
const Z3_POSITIVE_BRANCH_GET_VALUE: &str = "((x (root-obj (+ (^ x 2) (- 2)) 2)))";

/// The `define-fun` body Z3 reports for each branch — the value half of
/// `(define-fun x () Real (root-obj (+ (^ x 2) (- 2)) N))`.
const Z3_NEGATIVE_BRANCH_VALUE: &str = "(root-obj (+ (^ x 2) (- 2)) 1)";
const Z3_POSITIVE_BRANCH_VALUE: &str = "(root-obj (+ (^ x 2) (- 2)) 2)";

const SQRT2: &str = "(set-logic QF_NRA)
(declare-fun x () Real)
(assert (= (* x x) 2.0))
(check-sat)
";

const SQRT2_POSITIVE: &str = "(set-logic QF_NRA)
(declare-fun x () Real)
(assert (= (* x x) 2.0))
(assert (> x 0))
(check-sat)
";

/// The headline pin: `(get-model)` reports `√2` as a `root-obj` term, not as
/// the rounded `0.0` sort default it used to fall through to, and the term is
/// byte-identical to Z3's.
#[test]
fn get_model_root_obj_golden() {
    for (script, expected, branch) in [
        (SQRT2, Z3_NEGATIVE_BRANCH_VALUE, "unconstrained (−√2)"),
        (SQRT2_POSITIVE, Z3_POSITIVE_BRANCH_VALUE, "x > 0 (+√2)"),
    ] {
        let output = run(&format!("{script}(get-model)\n"));
        let model = output.join("\n");
        assert!(
            model.contains("sat"),
            "{branch}: the goal must be sat, got {model}"
        );
        assert!(
            model.contains(&format!("(define-fun x () Real {expected})")),
            "{branch}: (get-model) must report Z3's exact root-obj term \
             `{expected}`, got:\n{model}"
        );
        assert!(
            !model.contains("Real 0.0"),
            "{branch}: reporting the sort default 0.0 for an irrational \
             witness would be a value satisfying none of the assertions; got:\n{model}"
        );
    }
}

/// `(get-value (x))` is byte-exact against Z3 for both branches — the whole
/// response line, not a substring.
#[test]
fn get_value_line_is_byte_exact_with_z3() {
    for (script, expected, branch) in [
        (SQRT2, Z3_NEGATIVE_BRANCH_GET_VALUE, "unconstrained (−√2)"),
        (SQRT2_POSITIVE, Z3_POSITIVE_BRANCH_GET_VALUE, "x > 0 (+√2)"),
    ] {
        let output = run(&format!("{script}(get-value (x))\n"));
        let value_line = output
            .iter()
            .find(|line| line.contains("root-obj"))
            .unwrap_or_else(|| panic!("{branch}: no root-obj line in {output:?}"));
        assert_eq!(
            value_line, expected,
            "{branch}: (get-value (x)) must match z3 4.15.4 byte for byte"
        );
    }
}

/// `(get-model)` and `(get-value ..)` must not disagree about the same
/// constant — the contract `Context::format_get_value` documents. Both are
/// rendered from the one side-channel, so this pins that they stay wired
/// together rather than growing a second renderer.
#[test]
fn get_model_and_get_value_agree_on_the_algebraic_constant() {
    let output = run(&format!("{SQRT2_POSITIVE}(get-model)\n(get-value (x))\n"));
    let joined = output.join("\n");
    assert!(
        joined.contains(&format!(
            "(define-fun x () Real {Z3_POSITIVE_BRANCH_VALUE})"
        )),
        "model half missing in {joined}"
    );
    assert!(
        joined.contains(Z3_POSITIVE_BRANCH_GET_VALUE),
        "get-value half missing in {joined}"
    );
}

/// A QF_NRA goal whose witness *is* rational must be untouched: the ordinary
/// `Model` path answers it, the side-channel stays empty, and no `root-obj`
/// appears anywhere.
#[test]
fn a_rational_nra_model_is_unaffected() {
    let output = run("(set-logic QF_NRA)
(declare-fun x () Real)
(declare-fun y () Real)
(assert (= (* x y) 6.0))
(assert (= (+ x y) 5.0))
(check-sat)
(get-model)
");
    let joined = output.join("\n");
    assert!(joined.contains("sat"), "must be sat: {joined}");
    assert!(
        !joined.contains("root-obj"),
        "a rational witness must not be reported through the algebraic \
         channel: {joined}"
    );
    // 2 and 3 in some order — the pair is what the constraints force.
    assert!(
        joined.contains("(define-fun x () Real 2.0)")
            && joined.contains("(define-fun y () Real 3.0)")
            || joined.contains("(define-fun x () Real 3.0)")
                && joined.contains("(define-fun y () Real 2.0)"),
        "the rational model must still be reported: {joined}"
    );
}

/// A model mixing an algebraic constant, a rationally-constrained one, and a
/// wholly unconstrained one must report all three — and must agree with Z3 on
/// each.
///
/// This is the case that would break if the side-channel were populated only
/// for the *irrational* variables: `y` would then be completed from the `Real`
/// sort default as `0.0`, contradicting `y > 3`. It is populated all-or-nothing
/// precisely so the completion never has to guess at a constrained variable,
/// while `z` — genuinely unconstrained, absent from the translated problem
/// altogether — still gets its sort default, exactly as Z3 gives it.
///
/// Captured from `z3 4.15.4`: `x = (root-obj (+ (^ x 2) (- 2)) 1)`,
/// `y = 4.0`, `z = 0.0`.
#[test]
fn a_mixed_algebraic_rational_and_unconstrained_model_matches_z3() {
    let output = run("(set-logic QF_NRA)
(declare-fun x () Real)
(declare-fun y () Real)
(declare-fun z () Real)
(assert (= (* x x) 2.0))
(assert (> y 3.0))
(check-sat)
(get-model)
");
    assert_eq!(response(&output, 0), "sat");
    let model = response(&output, 1);
    for expected in [
        format!("(define-fun x () Real {Z3_NEGATIVE_BRANCH_VALUE})"),
        "(define-fun y () Real 4.0)".to_string(),
        "(define-fun z () Real 0.0)".to_string(),
    ] {
        assert!(
            model.contains(&expected),
            "missing `{expected}` in:\n{model}"
        );
    }
}

/// A `pop` invalidates the verdict, and the algebraic values are half of that
/// verdict's model. A stale `root-obj` surviving into the next check's model
/// would report `√2` for a variable the new stack pins to `−2`.
#[test]
fn pop_clears_the_algebraic_side_channel() {
    // Responses, in order: [0] check-sat, [1] get-value, [2] check-sat,
    // [3] get-model, [4] get-value — the first two from inside the pushed
    // scope, the last three from the stack that replaced it.
    let output = run("(set-logic QF_NRA)
(declare-fun x () Real)
(push 1)
(assert (= (* x x) 2.0))
(check-sat)
(get-value (x))
(pop 1)
(assert (= (* x x) 4.0))
(check-sat)
(get-model)
(get-value (x))
");
    assert_eq!(response(&output, 0), "sat");
    assert_eq!(
        response(&output, 1),
        Z3_NEGATIVE_BRANCH_GET_VALUE,
        "the pushed scope's algebraic witness must be reported"
    );

    assert_eq!(response(&output, 2), "sat");
    let after_pop = response(&output, 3);
    assert!(
        !after_pop.contains("root-obj"),
        "the popped scope's algebraic witness must not survive into the \
         next check's model: {after_pop}"
    );
    assert!(
        after_pop.contains("(define-fun x () Real -2.0)"),
        "x² = 4 must report its rational root instead: {after_pop}"
    );
    assert!(
        !response(&output, 4).contains("root-obj"),
        "nor into the next check's (get-value ..): {:?}",
        response(&output, 4)
    );
}

/// A fresh assertion on top of an algebraic model invalidates it too — the
/// same `invalidate_results` hook that drops `Model`.
#[test]
fn a_later_assert_clears_the_algebraic_side_channel() {
    // Responses: [0] check-sat, [1] get-value, [2] check-sat, [3] get-model,
    // [4] get-value.
    let output = run("(set-logic QF_NRA)
(declare-fun x () Real)
(assert (= (* x x) 2.0))
(check-sat)
(get-value (x))
(assert (= x 1.0))
(check-sat)
(get-model)
(get-value (x))
");
    assert_eq!(response(&output, 0), "sat");
    assert_eq!(
        response(&output, 1),
        Z3_NEGATIVE_BRANCH_GET_VALUE,
        "the first check must report the algebraic witness"
    );

    // `x² = 2 ∧ x = 1` has no solution, so no model may be published for it —
    // and in particular not the previous check's `√2`.
    assert_ne!(
        response(&output, 2),
        "sat",
        "x² = 2 ∧ x = 1 has no real solution"
    );
    for index in [3, 4] {
        let reply = response(&output, index);
        assert!(
            !reply.contains("root-obj"),
            "the invalidated algebraic witness must not be reported for the \
             refuted stack: {reply}"
        );
        assert!(
            !reply.contains("define-fun"),
            "the refuted stack must publish no model: {reply}"
        );
    }
}

/// A compound query over an algebraic constant must **echo**, never
/// substitute the sort default.
///
/// Z3 answers `2.0` here (it can compute in the algebraic field); this solver
/// cannot yet, and the important property is that it does not answer `0.0`,
/// which is what completing `x` from the `Real` sort default would produce —
/// a fabricated value contradicting the very model it was completed from.
#[test]
fn a_compound_query_over_an_algebraic_constant_does_not_fabricate_a_value() {
    let output = run(&format!("{SQRT2}(get-value ((* x x)))\n"));
    let joined = output.join("\n");
    assert!(
        !joined.contains("0.0"),
        "completing x from the Real sort default would answer 0.0 for \
         (* x x) on a goal whose model says it is 2: {joined}"
    );
    assert!(
        joined.contains("(* x x)"),
        "the unfoldable term must echo back: {joined}"
    );
}

/// `(get-value ..)` and `(get-model)` are `sat`-mode commands. An algebraic
/// model must not make them answerable after an `unsat` verdict.
#[test]
fn no_model_is_published_for_an_unsat_nra_goal() {
    let output = run("(set-logic QF_NRA)
(declare-fun x () Real)
(assert (< (* x x) 0.0))
(check-sat)
(get-model)
(get-value (x))
");
    let joined = output.join("\n");
    assert!(joined.contains("unsat"), "x² < 0 is unsat: {joined}");
    assert!(
        !joined.contains("root-obj"),
        "an unsat goal must publish no values: {joined}"
    );
}
