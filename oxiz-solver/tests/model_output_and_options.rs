//! End-to-end tests for model-value output completeness, the `:print-success`
//! option, named `(get-unsat-core)` minimization, and the SMT-LIB solver-mode
//! rules that decide when a `get-*` query may answer at all — all driven
//! through [`Context::execute_script`].

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    ctx.execute_script(script)
        .expect("script should parse and run")
}

// ---------------------------------------------------------------------
// get-model must print valid SMT-LIB values (never the invalid `?`) for
// FP, Array, and uninterpreted-sort constants.
// ---------------------------------------------------------------------

#[test]
fn get_model_uninterpreted_sort_value_is_valid() {
    let output = run(r#"
        (set-logic QF_UF)
        (declare-sort S 0)
        (declare-const x S)
        (assert true)
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(!model.contains('?'), "model has invalid ? value:\n{model}");
    assert!(
        model.contains("@uc_S_"),
        "expected an uninterpreted-sort witness:\n{model}"
    );
}

#[test]
fn get_model_distinct_uninterpreted_constants_get_distinct_witnesses() {
    let output = run(r#"
        (set-logic QF_UF)
        (declare-sort S 0)
        (declare-const x S)
        (declare-const y S)
        (assert true)
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(
        model.contains("@uc_S_0") && model.contains("@uc_S_1"),
        "distinct uninterpreted constants should get distinct witnesses:\n{model}"
    );
}

#[test]
fn get_model_array_value_is_valid() {
    let output = run(r#"
        (set-logic QF_ALIA)
        (declare-const arr (Array Int Int))
        (assert true)
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(!model.contains('?'), "model has invalid ? value:\n{model}");
    assert!(
        model.contains("as const"),
        "expected a constant-array value:\n{model}"
    );
}

#[test]
fn get_model_fp_value_is_valid() {
    let output = run(r#"
        (set-logic QF_FP)
        (declare-const f (_ FloatingPoint 8 24))
        (assert true)
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(!model.contains('?'), "model has invalid ? value:\n{model}");
    assert!(
        model.contains("+zero"),
        "expected a concrete FP default value:\n{model}"
    );
}

// ---------------------------------------------------------------------
// :print-success — emit `success` after each silently-succeeding command.
// ---------------------------------------------------------------------

#[test]
fn print_success_emits_acknowledgements() {
    let output = run(r#"
        (set-option :print-success true)
        (declare-const p Bool)
        (assert p)
        (check-sat)
    "#);
    // The enabling set-option, the declaration, and the assertion each
    // acknowledge; check-sat reports its own result instead.
    assert_eq!(
        output,
        vec![
            "success".to_string(),
            "success".to_string(),
            "success".to_string(),
            "sat".to_string(),
        ]
    );
}

#[test]
fn print_success_is_off_by_default() {
    let output = run(r#"
        (declare-const p Bool)
        (assert p)
        (check-sat)
    "#);
    assert_eq!(output, vec!["sat".to_string()]);
}

#[test]
fn print_success_get_option_reflects_enabled_state() {
    let output = run(r#"
        (set-option :print-success true)
        (get-option :print-success)
    "#);
    // set-option acknowledges; get-option reports the real `true`.
    assert_eq!(output, vec!["success".to_string(), "true".to_string()]);
}

// ---------------------------------------------------------------------
// (get-unsat-core) reports the named subset actually used in the
// refutation, minimized to exclude irrelevant named assertions.
// ---------------------------------------------------------------------

#[test]
fn named_unsat_core_reports_used_named_assertions() {
    let output = run(r#"
        (set-option :produce-unsat-cores true)
        (set-logic QF_UF)
        (declare-const p Bool)
        (assert (! p :named a1))
        (assert (! (not p) :named a2))
        (check-sat)
        (get-unsat-core)
    "#);
    assert_eq!(output[0], "unsat");
    let core = &output[1];
    assert!(core.contains("a1"), "core should contain a1: {core}");
    assert!(core.contains("a2"), "core should contain a2: {core}");
}

#[test]
fn unsat_core_excludes_irrelevant_named_assertion() {
    // `a3` (asserting `q`) plays no part in the `p ∧ ¬p` contradiction, so
    // deletion-based minimization must drop it from the reported core.
    let output = run(r#"
        (set-option :produce-unsat-cores true)
        (set-logic QF_UF)
        (declare-const p Bool)
        (declare-const q Bool)
        (assert (! p :named a1))
        (assert (! (not p) :named a2))
        (assert (! q :named a3))
        (check-sat)
        (get-unsat-core)
    "#);
    assert_eq!(output[0], "unsat");
    let core = &output[1];
    assert!(core.contains("a1"), "core should contain a1: {core}");
    assert!(core.contains("a2"), "core should contain a2: {core}");
    assert!(
        !core.contains("a3"),
        "minimization must exclude the irrelevant a3: {core}"
    );
}

// ---------------------------------------------------------------------
// SMT-LIB 2.6 §4.1.1 (solver modes, Fig. 4.1): `get-model`, `get-value`,
// `get-assignment`, `get-unsat-core`, `get-proof` and
// `get-unsat-assumptions` are available only in `sat` / `unsat` mode.
// `assert`, `push`, `pop`, `reset-assertions` and `reset` all return the
// solver to `assert` mode, where those queries must report an error
// instead of answering from a superseded assertion stack.
// ---------------------------------------------------------------------

/// Helper: an SMT-LIB error S-expression, whatever its message.
fn is_error(response: &str) -> bool {
    response.starts_with("(error ")
}

#[test]
fn test_get_model_invalidated_by_pop() {
    // `x = 5` only holds inside the pushed scope; after the `pop` the model
    // that proved it no longer describes the current assertion stack.
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (push 1)
        (assert (= x 5))
        (check-sat)
        (pop 1)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        is_error(&output[1]),
        "get-model after pop must not report the pre-pop model: {}",
        output[1]
    );
    assert!(
        !output[1].contains('5'),
        "the stale value must not leak into the error: {}",
        output[1]
    );
}

#[test]
fn test_get_value_invalidated_by_pop() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (push 1)
        (assert (= x 5))
        (check-sat)
        (pop 1)
        (get-value (x))
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        is_error(&output[1]),
        "get-value after pop must not report the pre-pop value: {}",
        output[1]
    );
}

#[test]
fn test_get_model_invalidated_by_assert() {
    // A new assertion supersedes the model built for the previous one.
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (assert (= x 6))
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        is_error(&output[1]),
        "get-model after a further assert must error: {}",
        output[1]
    );
}

#[test]
fn test_get_model_invalidated_by_push() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (push 1)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        is_error(&output[1]),
        "get-model after push must error: {}",
        output[1]
    );
}

/// Control for the invalidation tests: nothing intervenes between the
/// successful `check-sat` and the queries, so both must answer normally.
#[test]
fn test_get_model_after_check_sat_is_available() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (get-model)
        (get-value (x))
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        output[1].contains("(define-fun x () Int 5)"),
        "model must report x = 5: {}",
        output[1]
    );
    assert_eq!(output[2], "((x 5))");
}

/// Control: a `check-sat` *inside* the pushed scope re-establishes `sat`
/// mode, so the query answers again — invalidation is about staleness, not
/// about permanently disabling the command.
#[test]
fn test_get_model_available_again_after_recheck_in_new_scope() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (push 1)
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(output[1], "sat");
    assert!(
        output[2].contains("(define-fun x () Int 5)"),
        "re-checking inside the scope must restore the model: {}",
        output[2]
    );
}

#[test]
fn test_get_model_invalidated_by_reset_assertions() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (reset-assertions)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        is_error(&output[1]),
        "get-model after reset-assertions must error: {}",
        output[1]
    );
}

#[test]
fn test_get_model_invalidated_by_set_logic() {
    // `set-logic` reconfigures the arithmetic engine underneath the cached
    // model, so the verdict it produced no longer describes the solver.
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (set-logic QF_LRA)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        is_error(&output[1]),
        "get-model after set-logic must error: {}",
        output[1]
    );
}

/// A pure *declaration* deliberately does **not** invalidate: it adds no
/// constraint, so the previous model still satisfies every assertion and is
/// simply extended with the new constant's sort default (matching z3).
#[test]
fn test_get_model_survives_a_later_declaration() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (declare-const y Int)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        output[1].contains("(define-fun x () Int 5)")
            && output[1].contains("(define-fun y () Int 0)"),
        "a later declaration must extend, not invalidate, the model: {}",
        output[1]
    );
}

/// A `sat` verdict on an empty assertion stack must still answer `get-model` /
/// `get-value`: with nothing asserted every assignment is a model, so the sort
/// defaults are exact rather than a guess.
#[test]
fn test_get_model_with_no_assertions_reports_defaults() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (check-sat)
        (get-model)
        (get-value (x))
    "#);
    assert_eq!(output[0], "sat");
    assert!(
        output[1].contains("(define-fun x () Int 0)"),
        "an assertion-free sat must still produce a model: {}",
        output[1]
    );
    assert_eq!(output[2], "((x 0))");
}

#[test]
fn test_get_unsat_core_invalidated_by_pop() {
    // Regression: the core recorded before the `pop` indexes the solver's
    // pre-pop assertion vector.  Reporting it against the truncated vector
    // used to panic outright ("index out of bounds") inside
    // `Solver::minimize_unsat_core`.
    let output = run(r#"
        (set-option :produce-unsat-cores true)
        (set-logic QF_LIA)
        (declare-const x Int)
        (push 1)
        (assert (! (= x 5) :named a1))
        (assert (! (= x 6) :named a2))
        (check-sat)
        (pop 1)
        (get-unsat-core)
    "#);
    assert_eq!(output[0], "unsat");
    assert!(
        is_error(&output[1]),
        "get-unsat-core after pop must error: {}",
        output[1]
    );
}

/// Control for the unsat-core invalidation test.
#[test]
fn test_get_unsat_core_after_unsat_is_available() {
    let output = run(r#"
        (set-option :produce-unsat-cores true)
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (! (= x 5) :named a1))
        (assert (! (= x 6) :named a2))
        (check-sat)
        (get-unsat-core)
    "#);
    assert_eq!(output[0], "unsat");
    assert!(
        output[1].contains("a1") && output[1].contains("a2"),
        "core must still be reported when nothing intervened: {}",
        output[1]
    );
}

#[test]
fn test_get_assignment_invalidated_by_pop() {
    let output = run(r#"
        (set-logic QF_UF)
        (declare-const p Bool)
        (push 1)
        (assert p)
        (check-sat)
        (pop 1)
        (get-assignment)
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(
        output[1], "()",
        "get-assignment after pop must not report the pre-pop assignment"
    );
}

/// Control for the assignment invalidation test.
#[test]
fn test_get_assignment_after_check_sat_is_available() {
    let output = run(r#"
        (set-logic QF_UF)
        (declare-const p Bool)
        (assert p)
        (check-sat)
        (get-assignment)
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(output[1], "((p true))");
}

#[test]
fn test_get_unsat_assumptions_invalidated_by_pop() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const p Bool)
        (push 1)
        (assert (=> p (= x 5)))
        (assert (=> p (= x 6)))
        (check-sat-assuming (p))
        (pop 1)
        (get-unsat-assumptions)
    "#);
    assert_eq!(output[0], "unsat");
    assert!(
        is_error(&output[1]),
        "get-unsat-assumptions after pop must error: {}",
        output[1]
    );
}

// ---------------------------------------------------------------------
// (get-value ...) must agree with (get-model): a constant that occurs in
// no assertion has a sort default, not an echo of itself.
// ---------------------------------------------------------------------

#[test]
fn test_get_value_unconstrained_var_has_default() {
    // Sort-agnostic: Bool, Int, Real, BitVec and String all used to answer
    // `((x x))` — the queried term echoed back, which is not a value.
    for (logic, sort, name, expected) in [
        ("QF_LIA", "Int", "i", "0"),
        ("QF_UF", "Bool", "b", "false"),
        ("QF_LRA", "Real", "r", "0.0"),
        ("QF_BV", "(_ BitVec 8)", "v", "#b00000000"),
        ("QF_S", "String", "s", "\"\""),
    ] {
        let script = format!(
            "(set-logic {logic})\n(declare-const {name} {sort})\n(check-sat)\n\
             (get-value ({name}))"
        );
        let output = run(&script);
        assert_eq!(output[0], "sat", "{sort} script should be sat");
        assert_eq!(
            output[1],
            format!("(({name} {expected}))"),
            "unconstrained {sort} constant must get its sort default"
        );
    }
}

#[test]
fn test_get_value_agrees_with_get_model_for_unconstrained_var() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const y Int)
        (assert (= y 3))
        (check-sat)
        (get-value (x y))
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(output[1], "((x 0)\n (y 3))");
    assert!(
        output[2].contains("(define-fun x () Int 0)")
            && output[2].contains("(define-fun y () Int 3)"),
        "get-model must report the same values as get-value: {}",
        output[2]
    );
}

/// Control: completion must not overwrite a value the model really pinned
/// down — a constrained variable still reports its own value.
#[test]
fn test_get_value_constrained_var_reports_real_value() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (declare-const y Int)
        (assert (= x 7))
        (assert (> y 100))
        (check-sat)
        (get-value (x))
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(output[1], "((x 7))");
}

#[test]
fn test_get_value_compound_over_unconstrained_var_is_completed() {
    // The completion is applied inside compound terms too, so `(+ x 1)`
    // reduces to `1` rather than echoing back a partially evaluated term.
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (check-sat)
        (get-value ((+ x 1)))
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(output[1], "(((+ x 1) 1))");
}

// ---------------------------------------------------------------------
// SMT-LIB 2.6 §4.2.5: (reset-assertions) empties the assertion stack but
// keeps the logic, the declarations and the options.
// ---------------------------------------------------------------------

#[test]
fn test_reset_assertions_keeps_declarations() {
    let output = run(r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (reset-assertions)
        (get-assertions)
        (assert (= x 7))
        (check-sat)
        (get-value (x))
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(output[1], "()", "the assertion stack must be empty");
    assert_eq!(output[2], "sat");
    assert_eq!(
        output[3], "((x 7))",
        "`x` must still be declared, and constrained only by the new assertion"
    );
}

// The one test in this file that needs the `nlsat` feature, and it needs it by
// construction: its subject is that `(reset-assertions)` preserves the *NLSAT
// installation* the logic selected, so it can only be written as an assertion
// about a verdict NLSAT alone produces. Without the feature there is nothing to
// install and nothing to preserve, and both scripts below correctly answer
// `unknown` — which `oxiz-solver/tests/nlsat_feature_gate.rs` pins separately.
// Everything else in this file is feature-independent and stays unconditional.
#[cfg(feature = "nlsat")]
#[test]
fn test_reset_assertions_keeps_logic_selected_theory() {
    // The logic selects the arithmetic engine (`QF_NRA` installs NLSAT).
    // `Solver::reset` drops it, so without re-establishing the logic this
    // answered `unknown` after the reset while the identical fresh script
    // answered `sat`.
    let fresh = run(r#"
        (set-logic QF_NRA)
        (declare-const x Real)
        (assert (> (* x x) 2.0))
        (assert (< x 0.0))
        (check-sat)
    "#);
    assert_eq!(fresh, vec!["sat".to_string()]);

    let after_reset = run(r#"
        (set-logic QF_NRA)
        (declare-const x Real)
        (assert true)
        (check-sat)
        (reset-assertions)
        (assert (> (* x x) 2.0))
        (assert (< x 0.0))
        (check-sat)
    "#);
    assert_eq!(
        after_reset,
        vec!["sat".to_string(), "sat".to_string()],
        "reset-assertions must keep the logic's theory selection"
    );
}

#[test]
fn test_reset_assertions_keeps_options() {
    let output = run(r#"
        (set-option :produce-unsat-cores true)
        (set-option :random-seed 42)
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (! (= x 5) :named a1))
        (check-sat)
        (reset-assertions)
        (get-option :produce-unsat-cores)
        (get-option :random-seed)
        (assert (! (= x 5) :named b1))
        (assert (! (= x 6) :named b2))
        (check-sat)
        (get-unsat-core)
    "#);
    assert_eq!(output[0], "sat");
    assert_eq!(output[1], "true");
    assert_eq!(output[2], "42");
    assert_eq!(output[3], "unsat");
    assert!(
        output[4].contains("b1") && output[4].contains("b2"),
        "unsat-core production must survive reset-assertions: {}",
        output[4]
    );
}

#[test]
fn test_reset_assertions_keeps_mbqi_declared_const_candidates() {
    // A trigger-free universal refuted only by instantiating with the
    // still-declared constant `a`.  `Solver::reset` clears the MBQI
    // instantiation candidates, so the surviving declarations have to be
    // re-registered for the refutation to remain reachable.
    let output = run(r#"
        (set-logic ALL)
        (declare-const a Int)
        (declare-fun p (Int) Bool)
        (assert true)
        (check-sat)
        (reset-assertions)
        (assert (forall ((x Int)) (p x)))
        (assert (not (p a)))
        (check-sat)
    "#);
    assert_eq!(output, vec!["sat".to_string(), "unsat".to_string()]);
}

/// Control for `reset-assertions`: `reset` really does clear everything, so
/// the previously declared constant is gone afterwards.
#[test]
fn test_reset_clears_declarations_and_logic() {
    let mut ctx = Context::new();
    let output = ctx
        .execute_script(
            r#"
        (set-logic QF_LIA)
        (declare-const x Int)
        (assert (= x 5))
        (check-sat)
        (reset)
    "#,
        )
        .expect("script should parse and run");
    assert_eq!(output, vec!["sat".to_string()]);
    assert_eq!(ctx.logic(), None, "reset must clear the logic");
    assert!(
        ctx.get_assertions().is_empty(),
        "reset must clear the assertions"
    );
    assert!(
        ctx.get_model().is_none(),
        "reset must invalidate the check result"
    );
}

// ---------------------------------------------------------------------
// Datatype model construction.
//
// A `sat` verdict has to come with a witness that actually satisfies the
// formula.  Before the datatype reconstruction existed, `build_model` assigned
// no datatype-sorted term at all and `(get-model)` completed from the sort
// default — the first nullary constructor — so `((_ is cons) l)` with
// `(= (head l) 7)` was answered `sat` (correctly; z3 agrees) with the witness
// `l = nil`, which satisfies neither conjunct.  The verdict was sound, the
// witness was not.
//
// Every test below therefore checks the witness *by substitution*: it pins each
// reported constant to its reported value and re-solves.  A model that
// satisfies the formula leaves the pinned problem `sat`; one that does not
// makes it `unsat` — which is exactly what the old `l = nil` output did, so
// these tests fail loudly against the previous behaviour rather than merely
// observing that some model was printed.
// ---------------------------------------------------------------------

/// The `List Int` declaration shared by the datatype model tests.
const LST: &str = "(declare-datatype Lst ((nil) (cons (head Int) (tail Lst))))";

/// The `(define-fun NAME () SORT VALUE)` entries of a `(get-model)` response,
/// as `(name, value)` pairs.  `VALUE` may itself be a parenthesised term
/// (`(cons 7 nil)`), so it is taken by scanning to the matching paren.
fn model_entries(model: &str) -> Vec<(String, String)> {
    let bytes: Vec<char> = model.chars().collect();
    let mut entries = Vec::new();
    let mut search = 0usize;
    while let Some(offset) = model[search..].find("(define-fun ") {
        let start = search + offset;
        let mut depth = 0i32;
        let mut end = start;
        for (index, ch) in bytes.iter().enumerate().skip(start) {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = index;
                        break;
                    }
                }
                _ => {}
            }
        }
        let body = &model[start + "(define-fun ".len()..end];
        let mut rest = body.trim_start();
        let name_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let name = rest[..name_end].to_string();
        rest = rest[name_end..].trim_start();
        // Skip the empty argument list `()` and the sort, which may itself be
        // parenthesised (`(_ BitVec 8)`).
        rest = rest.strip_prefix("()").unwrap_or(rest).trim_start();
        let sort_len = if rest.starts_with('(') {
            let mut depth = 0i32;
            let mut len = rest.len();
            for (index, ch) in rest.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            len = index + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            len
        } else {
            rest.find(char::is_whitespace).unwrap_or(rest.len())
        };
        let value = rest[sort_len..].trim().to_string();
        entries.push((name, value));
        search = end + 1;
    }
    entries
}

/// Run `preamble` + `asserts`, then verify by substitution that the printed
/// model satisfies the assertions: every reported constant is pinned to its
/// reported value and the problem re-solved.  Returns the model text so callers
/// can additionally assert on its shape.
fn model_must_satisfy(preamble: &str, asserts: &str) -> String {
    let output = run(&format!("{preamble}\n{asserts}\n(check-sat)\n(get-model)"));
    assert_eq!(output[0], "sat", "expected sat for:\n{preamble}\n{asserts}");
    let model = output[1].clone();
    assert!(
        !model.contains('?'),
        "model contains the invalid `?` placeholder:\n{model}"
    );
    let entries = model_entries(&model);
    assert!(
        !entries.is_empty(),
        "no model entries parsed from:\n{model}"
    );
    let pins: String = entries
        .iter()
        .map(|(name, value)| format!("(assert (= {name} {value}))\n"))
        .collect();
    let recheck = run(&format!("{preamble}\n{asserts}\n{pins}(check-sat)"));
    assert_eq!(
        recheck[0], "sat",
        "the printed model does NOT satisfy the formula \
         (pinning it makes the problem unsat):\nmodel: {model}\npins:\n{pins}"
    );
    model
}

/// Shape 1: a tester-constrained variable.  `l = nil` was printed here.
#[test]
fn get_model_datatype_tester_witness_satisfies_formula() {
    let model = model_must_satisfy(
        &format!("(set-logic ALL) {LST} (declare-const l Lst)"),
        "(assert ((_ is cons) l)) (assert (= (head l) 7))",
    );
    assert!(
        model.contains("(cons 7 nil)"),
        "expected the z3-equivalent witness (cons 7 nil):\n{model}"
    );
}

/// Shape 2: nested reconstruction — two tester levels and a field below both.
#[test]
fn get_model_datatype_nested_witness_satisfies_formula() {
    let model = model_must_satisfy(
        &format!("(set-logic ALL) {LST} (declare-const l Lst)"),
        "(assert ((_ is cons) l)) (assert ((_ is cons) (tail l)))
         (assert (= (head l) 1)) (assert (= (head (tail l)) 3))",
    );
    assert!(
        model.contains("(cons 1 (cons 3 nil))"),
        "expected the z3-equivalent nested witness:\n{model}"
    );
}

/// Shape 3: a record (single, non-nullary constructor).  This printed the
/// invalid `?` before, because the sort default needed a recursive completion.
#[test]
fn get_model_datatype_record_witness_satisfies_formula() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Pair ((mk-pair (fst Int) (snd Int))))
         (declare-const p Pair)",
        "(assert (= (fst p) 4)) (assert (= (snd p) 9))",
    );
    assert!(
        model.contains("(mk-pair 4 9)"),
        "expected the z3-equivalent witness (mk-pair 4 9):\n{model}"
    );
}

/// Shape 4: an enumeration.  `c = red` was printed for a formula that rules
/// `red` out.
#[test]
fn get_model_datatype_enum_witness_satisfies_formula() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Color ((red) (green) (blue)))
         (declare-const c Color)",
        "(assert (not ((_ is red) c))) (assert (not ((_ is green) c)))",
    );
    assert!(
        model.contains("Color blue"),
        "the only remaining colour is blue:\n{model}"
    );
}

/// Shape 5: a datatype with a datatype-typed field.
#[test]
fn get_model_datatype_nested_sort_field_witness_satisfies_formula() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Inner ((mk-inner (val Int))))
         (declare-datatype Outer ((mk-outer (payload Inner) (tag Int))))
         (declare-const o Outer)",
        "(assert (= (val (payload o)) 5)) (assert (= (tag o) 2))",
    );
    assert!(
        model.contains("(mk-outer (mk-inner 5) 2)"),
        "expected the z3-equivalent witness:\n{model}"
    );
}

/// Shape 6: an unconstrained datatype variable — any well-sorted value is
/// legitimate, and the default must still be a *valid* one.  z3 prints `nil`.
#[test]
fn get_model_unconstrained_datatype_uses_a_valid_default() {
    let model = model_must_satisfy(
        &format!("(set-logic ALL) {LST} (declare-const l Lst) (declare-const x Int)"),
        "(assert (= x 3))",
    );
    assert!(
        model.contains("Lst nil"),
        "an unconstrained list defaults to its base constructor:\n{model}"
    );
}

/// An unconstrained datatype with *no* nullary constructor used to print the
/// invalid `?`; it must synthesize a ground constructor application instead.
#[test]
fn get_model_unconstrained_non_nullary_datatype_is_valid() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Inner ((mk-inner (val Int))))
         (declare-datatype Outer ((mk-outer (payload Inner) (tag Int))))
         (declare-const o Outer)
         (declare-const y Int)",
        "(assert (= y 1))",
    );
    assert!(
        model.contains("(mk-outer (mk-inner 0) 0)"),
        "expected a recursively completed ground default:\n{model}"
    );
}

/// A selector applied to the wrong constructor is underspecified in SMT-LIB —
/// `(head nil)` may be any `Int` — so `((_ is nil) l) ∧ (= (head l) 42)` is
/// satisfiable, and the witness must not pretend otherwise.
#[test]
fn get_model_selector_on_wrong_constructor_still_yields_a_valid_witness() {
    let model = model_must_satisfy(
        &format!("(set-logic ALL) {LST} (declare-const l Lst)"),
        "(assert ((_ is nil) l)) (assert (= (head l) 42))",
    );
    assert!(model.contains("Lst nil"), "l is a nil cell:\n{model}");
}

/// A literal constructor application in the formula is reported verbatim.
#[test]
fn get_model_literal_constructor_witness_satisfies_formula() {
    let model = model_must_satisfy(
        &format!("(set-logic ALL) {LST} (declare-const l Lst)"),
        "(assert (= l (cons 1 nil)))",
    );
    assert!(model.contains("(cons 1 nil)"), "{model}");
}

/// A variable asserted equal to a constructor application takes *that* value,
/// not one rebuilt from its own accessors: the equality is the most concrete
/// witness in the class.  Rebuilding it independently reported
/// `p = (mk-pair 0 0)` for `(assert (= p (mk-pair 1 2)))`, because the accessor
/// values are numbers the arithmetic model never had to separate.
#[test]
fn get_model_variable_equal_to_constructor_takes_its_value() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Pair ((mk-pair (first Int) (second Int))))
         (declare-const p Pair)",
        "(assert (= p (mk-pair 1 2)))",
    );
    assert!(model.contains("(mk-pair 1 2)"), "{model}");
}

/// Two datatype constants asserted *equal* must report the same witness — one
/// value per equality class, not one per term.
#[test]
fn get_model_equal_datatype_constants_share_one_witness() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Col ((red) (green) (blue)))
         (declare-datatype Box ((empty) (full (item Col))))
         (declare-const b1 Box)
         (declare-const b2 Box)",
        "(assert (= b1 b2)) (assert ((_ is full) b1)) (assert (= (item b1) blue))",
    );
    let entries = model_entries(&model);
    let value_of = |name: &str| {
        entries
            .iter()
            .find(|(entry, _)| entry == name)
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| panic!("no entry for {name} in:\n{model}"))
    };
    assert_eq!(
        value_of("b1"),
        value_of("b2"),
        "equal constants must share a witness:\n{model}"
    );
    assert_eq!(value_of("b1"), "(full blue)", "{model}");
}

/// Two datatype terms the search proved *distinct* must not be reported with
/// the same value.  For a single-constructor datatype the distinctness lives
/// entirely in the fields, and the linear solver discharges disequalities by
/// case split rather than by separating witnesses — so both sides used to come
/// back `(mk-pair 0 0)`, a witness that violates the very disequality asserted.
#[test]
fn get_model_disequal_records_get_distinct_witnesses() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Pair ((mk-pair (fst Int) (snd Int))))
         (declare-const p Pair)
         (declare-const q Pair)",
        "(assert (not (= p q)))",
    );
    let values: Vec<String> = model_entries(&model)
        .into_iter()
        .map(|(_, value)| value)
        .collect();
    assert_eq!(values.len(), 2, "{model}");
    assert_ne!(values[0], values[1], "disequal records collided:\n{model}");
}

/// The repair above must never move a field the formula pins, and must move a
/// whole equality class together: `q = r` has to survive it.
#[test]
fn get_model_disequal_records_respect_pinned_fields_and_equalities() {
    let model = model_must_satisfy(
        "(set-logic ALL)
         (declare-datatype Pair ((mk-pair (fst Int) (snd Int))))
         (declare-const p Pair)
         (declare-const q Pair)
         (declare-const r Pair)",
        "(assert (= q r)) (assert (= (fst r) 5)) (assert (= (fst p) 5))
         (assert (not (= p q)))",
    );
    let entries = model_entries(&model);
    let value_of = |name: &str| {
        entries
            .iter()
            .find(|(entry, _)| entry == name)
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| panic!("no entry for {name} in:\n{model}"))
    };
    assert_eq!(value_of("q"), value_of("r"), "q = r must be honoured");
    assert_ne!(value_of("p"), value_of("q"), "p != q must be honoured");
}

/// `(get-value ..)` and `(get-model)` must never disagree about the same
/// constant — including a reconstructed datatype value, and including a
/// datatype *sub-term* that only exists inside such a value.
#[test]
fn get_value_and_get_model_agree_on_datatype_values() {
    let output = run(&format!(
        "(set-logic ALL) {LST}
         (declare-datatype Pair ((mk-pair (fst Int) (snd Int))))
         (declare-const l Lst)
         (declare-const p Pair)
         (declare-const u Lst)
         (assert ((_ is cons) l))
         (assert (= (head l) 7))
         (assert (= (fst p) 4))
         (check-sat)
         (get-model)
         (get-value (l))
         (get-value (p))
         (get-value (u))
         (get-value ((tail l)))"
    ));
    assert_eq!(output[0], "sat");
    let entries = model_entries(&output[1]);
    let value_of = |name: &str| {
        entries
            .iter()
            .find(|(entry, _)| entry == name)
            .map(|(_, value)| value.clone())
            .unwrap_or_else(|| panic!("no entry for {name} in:\n{}", output[1]))
    };
    assert_eq!(output[2], format!("((l {}))", value_of("l")));
    assert_eq!(output[3], format!("((p {}))", value_of("p")));
    // An unconstrained constant is answered from `get-model` itself, so the two
    // commands cannot drift apart even for a synthesized default.
    assert_eq!(output[4], format!("((u {}))", value_of("u")));
    // The sub-value printed inside `l` is what `(tail l)` reports.
    assert_eq!(output[5], "(((tail l) nil))");
}

/// A nullary constructor is a plain symbol in SMT-LIB, not a one-element
/// application: `(nil)` is not a term the grammar accepts, so a model printed
/// with it could not be fed back to a solver.
#[test]
fn get_model_nullary_constructor_prints_without_parentheses() {
    let output = run(r#"
        (set-logic ALL)
        (declare-datatype Color ((red) (green) (blue)))
        (declare-const c Color)
        (assert ((_ is green) c))
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(model.contains("Color green"), "{model}");
    assert!(
        !model.contains("(green)"),
        "a nullary constructor must not be printed as an application:\n{model}"
    );
}

// ---------------------------------------------------------------------
// Controls: non-datatype model output is unchanged by the datatype work.
// ---------------------------------------------------------------------

/// Control: arithmetic model output is untouched.
#[test]
fn control_arith_model_output_is_unchanged() {
    let output = run(r#"
        (set-logic ALL)
        (declare-const x Int)
        (declare-const y Real)
        (assert (= x 5))
        (assert (= y 2.5))
        (check-sat)
        (get-model)
        (get-value (x))
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(model.contains("(define-fun x () Int 5)"), "{model}");
    assert!(model.contains("(define-fun y () Real (/ 5 2))"), "{model}");
    assert_eq!(output[2], "((x 5))");
}

/// Control: bit-vector, Bool and String model output is untouched.
#[test]
fn control_bv_bool_string_model_output_is_unchanged() {
    let output = run(r#"
        (set-logic ALL)
        (declare-const b (_ BitVec 4))
        (declare-const p Bool)
        (declare-const s String)
        (assert (= b #b0101))
        (assert p)
        (assert (= s "hi"))
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(
        model.contains("(define-fun b () (_ BitVec 4) #b0101)"),
        "{model}"
    );
    assert!(model.contains("(define-fun p () Bool true)"), "{model}");
    assert!(
        model.contains(r#"(define-fun s () String "hi")"#),
        "{model}"
    );
}

/// Control: an uninterpreted-sort constant still gets its `@uc_S_n` witness and
/// an array constant still gets its `((as const ..) ..)` value — the datatype
/// arm must not have displaced either.
#[test]
fn control_uninterpreted_and_array_defaults_are_unchanged() {
    let output = run(r#"
        (set-logic ALL)
        (declare-sort S 0)
        (declare-const u S)
        (declare-const a (Array Int Int))
        (assert true)
        (check-sat)
        (get-model)
    "#);
    assert_eq!(output[0], "sat");
    let model = &output[1];
    assert!(model.contains("@uc_S_0"), "{model}");
    assert!(model.contains("((as const (Array Int Int)) 0)"), "{model}");
    assert!(!model.contains('?'), "{model}");
}
