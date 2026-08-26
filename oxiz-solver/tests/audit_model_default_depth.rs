//! Regression tests for two solver defects found by the 0.3.1 soundness
//! sweep:
//!
//!   1. `context/model_fmt.rs`: `default_value` and `default_datatype_value`
//!      are mutually recursive -- a datatype's field can be an array, and an
//!      array's range can be a datatype -- but only the second carried a
//!      budget, and the first handed it a literal `0`. The count therefore
//!      restarted at every datatype level and `DEPTH_LIMIT = 16` never fired.
//!      `(declare-datatype D ((c (f (Array Int D)))))` then made `(get-model)`
//!      recurse forever and abort the process. (Genuine non-termination, not
//!      merely deep recursion: the *sort* graph is acyclic, but the walk also
//!      steps through the datatype *definition* table, and that edge closes
//!      the loop.)
//!
//!   2. `solver/mod.rs`: `check_core` consulted the `encode_depth_exceeded`
//!      honesty gate only after running the axiom instantiators, the five
//!      early-conflict collectors and the nonlinear/FP/string model attempts.
//!      Those are recursive walks over the very assertion terms the Tseitin
//!      encoder had already refused as too deep, so the process aborted before
//!      the gate could answer `Unknown`.
//!
//! Every deep structure below is built with an **iterative** loop, and every
//! call runs on a thread with an explicitly small (1 MiB) stack: a stack
//! overflow aborts the whole process, so "the call returned at all" is itself
//! part of the assertion.

use oxiz_core::ast::TermManager;
use oxiz_solver::{Context, Solver, SolverResult};

/// Run `body` on a 1 MiB stack and return its value.
///
/// Frame sizes differ between the debug and release profiles, so the stack is
/// pinned rather than inherited.
fn on_small_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    // STACK-1MIB: deliberately 1 MiB, not swept to 128 KiB — every depth
    // exercised through this helper tops out at a few thousand levels
    // (sub-10,000), so 1 MiB is intentionally generous rather than a
    // tuned minimum. See TODO.md "v0.3.2 backlog".
    std::thread::Builder::new()
        .stack_size(1 << 20)
        .spawn(body)
        .expect("spawn worker thread")
        .join()
        .expect("worker thread must return, not overflow its stack")
}

/// Execute `script` on a small stack and return its response lines.
fn run_script(script: &'static str) -> Vec<String> {
    on_small_stack(move || {
        let mut ctx = Context::new();
        ctx.execute_script(script).expect("script must execute")
    })
}

// ============ Finding 1: default_value / default_datatype_value budget =====

#[test]
fn datatype_whose_field_is_an_array_of_itself_terminates_in_get_model() {
    // `D`'s only field has sort `(Array Int D)`, which is not itself a
    // datatype sort -- so `default_constructor_index` accepts `c` as
    // "bottoming out" and the old code alternated
    // `default_datatype_value(D, 0)` -> `default_value(Array Int D)` ->
    // `default_datatype_value(D, 0)` forever.
    let out = run_script(
        "(declare-datatype D ((c (f (Array Int D)))))\n\
         (declare-const d D)\n\
         (check-sat)\n\
         (get-model)\n",
    );
    assert_eq!(out.first().map(String::as_str), Some("sat"));
    let model = out.get(1).expect("(get-model) response");

    // Terminating is necessary but not sufficient: the output must also stop
    // at the documented budget and say so, rather than silently presenting a
    // plausible-looking finite value for a sort that has no ground value.
    assert!(
        model.contains('?'),
        "the truncation marker must be visible in an ill-founded default: {model}"
    );
    assert_eq!(
        model.matches("as const").count(),
        16,
        "the datatype budget (16) must bound the alternation: {model}"
    );
}

#[test]
fn directly_ill_founded_datatype_still_truncates_at_the_same_budget() {
    // The pre-existing, already-bounded case must be unchanged by threading
    // the budget through the other half of the recursion.
    let out = run_script(
        "(declare-datatype T ((c (f T))))\n\
         (declare-const t T)\n\
         (check-sat)\n\
         (get-model)\n",
    );
    let model = out.get(1).expect("(get-model) response");
    assert_eq!(
        model.matches("(c ").count(),
        16,
        "expected exactly 16 constructor levels then `?`: {model}"
    );
    assert!(model.contains('?'), "unexpected model: {model}");
}

#[test]
fn deeply_nested_array_sorts_still_render_in_full() {
    // The total-step budget is set to the deepest sort the parser can build
    // (`MAX_SORT_PARSE_DEPTH` = 512), so nothing a script can spell loses
    // detail to it. 511 nested arrays is the deepest such sort.
    const LEVELS: usize = 511;
    let value = on_small_stack(|| {
        let mut script = String::from("(declare-const x ");
        for _ in 0..LEVELS {
            script.push_str("(Array Int ");
        }
        script.push_str("Int");
        for _ in 0..LEVELS {
            script.push(')');
        }
        script.push_str(")\n(check-sat)\n(get-model)\n");
        let mut ctx = Context::new();
        let out = ctx.execute_script(&script).expect("script must execute");
        out.get(1).cloned().expect("(get-model) response")
    });
    assert_eq!(
        value.matches("as const").count(),
        LEVELS,
        "a 511-deep array default must not be truncated"
    );
    assert!(
        !value.contains('?'),
        "no truncation marker expected at 511 levels: {value}"
    );
}

#[test]
fn well_founded_datatype_and_array_defaults_are_unchanged() {
    let out = run_script(
        "(declare-datatype P ((mk (a Int) (b Bool))))\n\
         (declare-const p P)\n\
         (declare-const q (Array Int P))\n\
         (check-sat)\n\
         (get-model)\n",
    );
    let model = out.get(1).expect("(get-model) response");
    assert!(model.contains("(mk 0 false)"), "unexpected model: {model}");
    assert!(
        model.contains("((as const (Array Int P)) (mk 0 false))"),
        "unexpected model: {model}"
    );
}

// ============ Finding 2: encode_depth_exceeded consulted before the walks ==

#[test]
fn a_term_deeper_than_the_encoder_allows_answers_unknown_instead_of_aborting() {
    // `ENCODE_DEPTH_LIMIT` is 512. The parser's own bound is 1024 and now
    // genuinely bounds *term* depth, so a script can no longer reach this;
    // the builder API can, and `Solver::assert` accepts whatever it produces.
    //
    // Built iteratively: a recursive helper would overflow before the
    // assertion could run.
    const DEPTH: usize = 5000;
    let result = on_small_stack(|| {
        let mut manager = TermManager::new();
        let string_sort = manager.sorts.string_sort();
        let mut chain = manager.mk_var("v0", string_sort);
        for i in 1..DEPTH {
            let next = manager.mk_var(&format!("v{i}"), string_sort);
            chain = manager.mk_str_concat(chain, next);
        }
        let empty = manager.mk_string_lit("");
        let assertion = manager.mk_eq(empty, chain);

        let mut solver = Solver::new();
        solver.assert(assertion, &mut manager);
        solver.check(&mut manager)
    });

    // The encoder flagged the assertion and skipped it, so the encoding is
    // incomplete: `Unknown` is the only honest answer. Before the fix the
    // process aborted inside `check_string_constraints` -> `eval_ground_bool`
    // on the way to this line.
    assert_eq!(
        result,
        SolverResult::Unknown,
        "an unencodable assertion must degrade to Unknown"
    );
}

#[test]
fn an_ordinary_shallow_string_problem_is_still_decided() {
    // The gate now runs before the early-conflict collectors, so pin that it
    // only fires when the encoder actually gave up: a normal refutable string
    // problem must still come back `unsat`.
    let out = run_script(
        "(declare-const s String)\n\
         (assert (= s \"abc\"))\n\
         (assert (= s \"xyz\"))\n\
         (check-sat)\n",
    );
    assert_eq!(out.first().map(String::as_str), Some("unsat"));
}
