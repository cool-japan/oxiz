//! Pins on the defects the round-4 adversarial recheck (pass 2) found on the
//! `#P2b-39` tree.
//!
//! Every test here is GREEN today and asserts what the tree currently *does*,
//! not what it should do.  Each one fails with `THE HOLE IS CLOSED` when the
//! behaviour becomes correct — that is the signal to invert the assertion and
//! move the entry to the closed list, exactly as
//! `round4_recheck_regressions.rs` did for the previous round.  Two tests are
//! ordinary guards rather than pins and say so.
//!
//! The evidence, and the campaign figures behind each number, are recorded in
//! the recheck's findings file; the shapes are reproduced here verbatim so the
//! repository carries its own reproduction.

use oxiz_solver::Context;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Run a script and return the response lines, folding an `Err` into a single
/// `(error …)` line the way the conformance runner does.
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

/// The `(model …)` block, or the empty string.
fn model_block(lines: &[String]) -> String {
    lines
        .iter()
        .find(|line| line.starts_with("(model"))
        .cloned()
        .unwrap_or_default()
}

/// The `(get-value …)` response, or the empty string.
fn value_block(lines: &[String]) -> String {
    lines
        .iter()
        .find(|line| line.trim_start().starts_with("(("))
        .cloned()
        .unwrap_or_default()
}

/// The body of `(define-fun <name> () <sort> <body>)` in a model block.
fn model_binding(model: &str, name: &str) -> String {
    let needle = format!("(define-fun {name} ()");
    model
        .lines()
        .find(|line| line.trim_start().starts_with(&needle))
        .unwrap_or_default()
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------
// 1. PIN — the const-vs-const refutation dies at one indirection.
// ---------------------------------------------------------------------------

/// `(= a ((as const A) d1))` and `(= a ((as const A) d2))` with `d1 != d2` is
/// unsatisfiable: `a` would have to be two different total functions at once.
/// The rule decision (1c) added — select congruence at each pair's own
/// extensionality witness — refutes it only when the two constants are the
/// *literal operands* of one equality, because each pair is instantiated at
/// its own witness index and the two witnesses never meet.  One declared
/// constant in between is enough to lose it.
///
/// Both independent oracles (a total-table brute force and an enumerating
/// evaluator) say `unsat`, and the answer is `unsat` by hand.  crates.io 0.3.3
/// and the 0.3.4 base answer `sat` too, so this is an unclosed part of
/// `#P2b-37`, not a regression from it.
#[test]
fn a_variable_equal_to_two_different_array_constants_is_still_satisfiable() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "THE HOLE IS CLOSED: a variable pinned to two array constants with \
         different defaults is now refuted — invert this pin and close the \
         indirect half of #P2b-37."
    );
}

/// The same hole with the indirection spread over two variables, which is how
/// a generated corpus reaches it: 349 of 530 exactly-decided scripts of this
/// shape answered `sat` against an exhaustive oracle.
#[test]
fn two_variables_pinned_to_different_array_constants_are_still_satisfiable() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))\n\
         (assert (= b ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n\
         (assert (= a b))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "THE HOLE IS CLOSED: two variables pinned to array constants with \
         different defaults and asserted equal are now refuted — invert this \
         pin."
    );
}

/// The hole closes the moment the script spells any `select`, because the
/// index it names enters the shared index set and the const-read axiom meets
/// congruence there.  A control, not a pin: this one must stay `unsat`.
#[test]
fn one_spelled_select_closes_the_indirect_const_hole() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))\n\
         (assert (= b ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n\
         (assert (= a b))\n\
         (assert (= (select a #b0) (select b #b0)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "unsat",
        "the spelled select must still carry the const-read axiom into the \
         equality (#P2b-37)"
    );
}

/// The two constants spelled directly *are* refuted — the control that shows
/// the rule exists and only the indirection defeats it.
#[test]
fn two_array_constants_spelled_directly_are_refuted() {
    let script = "(set-logic QF_ABV)\n\
         (assert (= ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1) \
                    ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n\
         (check-sat)\n";
    assert_eq!(verdict(&run(script)), "unsat");
}

// ---------------------------------------------------------------------------
// 2. PIN — an array model ignores its own class's `(as const)` member as soon
//    as a witness read is published.
// ---------------------------------------------------------------------------

/// Decision (3) rule 1 is "base = the default of an `(as const d)` member if
/// the class has one".  With a `distinct` in the script the extensionality
/// witness read is published on top of that base and takes its value from the
/// candidate assignment, which the const-read axiom does not constrain at the
/// witness index — the model-side shadow of the hole pinned above.  The
/// printed `a0` then disagrees with `a0 = ((as const A) #b00)` at the witness
/// index, so the model falsifies its own script.
#[test]
fn an_array_model_publishes_a_read_its_own_array_constant_forbids() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a0 (Array (_ BitVec 2) (_ BitVec 2)))\n\
         (declare-const a1 (Array (_ BitVec 2) (_ BitVec 2)))\n\
         (assert (distinct a0 a1))\n\
         (assert (= a0 ((as const (Array (_ BitVec 2) (_ BitVec 2))) #b00)))\n\
         (check-sat)\n\
         (get-model)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat");
    let binding = model_binding(&model_block(&lines), "a0");
    assert!(
        binding.contains("#b01"),
        "THE HOLE IS CLOSED: a0 is pinned to ((as const A) #b00) and its model \
         no longer publishes an entry the constant forbids — invert this pin. \
         Printed binding: {binding}"
    );
}

// ---------------------------------------------------------------------------
// 3. PIN — `(get-model)` prints a quoted symbol unquoted.
// ---------------------------------------------------------------------------

/// A symbol that needs `|…|` in the source needs it in the output too, or the
/// model block is not re-parsable SMT-LIB.  `(get-value)` spells it correctly
/// since `#P2b-39` gave the response the term's source text, so the two
/// commands now disagree about how to *write* the same symbol.  Pre-existing
/// on the 0.3.4 base and on crates.io 0.3.3 for `(get-model)`; the
/// disagreement between the two commands is new.
#[test]
fn get_model_still_prints_a_quoted_symbol_without_its_bars() {
    let script = "(set-logic QF_BV)\n\
         (declare-const |a b| (_ BitVec 1))\n\
         (assert (= |a b| |a b|))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value (|a b|))\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat");
    let model = model_block(&lines);
    assert!(
        model.contains("(define-fun a b ()"),
        "THE HOLE IS CLOSED: (get-model) now quotes a symbol that needs \
         quoting — invert this pin.  Model: {model}"
    );
    assert!(
        value_block(&lines).contains("|a b|"),
        "(get-value) must keep spelling the queried term as written \
         (SMT-LIB 2.6 section 4.1.1)"
    );
}

// ---------------------------------------------------------------------------
// 4. PIN — decision (2)'s canonical class -> value map is not canonical, and
//    its own `debug_assert!` says so.
// ---------------------------------------------------------------------------

/// Two entries of one interpretation with the same evaluated argument tuple
/// and different values are what decision (2) declared impossible.  Asking for
/// `(get-value …)` alongside `(get-model)` makes them happen: in a debug build
/// `get_func_interp_raw`'s `debug_assert!` fires, and in a release build the
/// contradictory interpretation is printed silently.  The same script without
/// its `(get-value)` command does not panic, which is what localises the
/// defect to the `(get-value)` path rather than to the map itself.
///
/// The panic is caught so the test reports the pin rather than aborting the
/// run; `debug_assert!` is compiled out in release, where the assertion below
/// records that the answer still comes back.
#[test]
fn a_get_value_query_beside_get_model_breaks_the_one_reading() {
    let script = "(set-logic QF_AUFLIA)\n\
         (declare-fun f (Int) Int)\n\
         (declare-fun g (Int Int) Int)\n\
         (declare-const j Int)\n\
         (declare-const k Int)\n\
         (declare-const v Int)\n\
         (declare-const w Int)\n\
         (assert (distinct (+ (f (f 0)) (+ (f w) (g 1 3))) w))\n\
         (assert (not (or (distinct (+ 2 1) j) \
                          (= w (f (select (store ((as const (Array Int Int)) 0) j v) k))))))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value (j k v w (f j)))\n";
    let outcome = catch_unwind(AssertUnwindSafe(|| run(script)));
    if cfg!(debug_assertions) {
        assert!(
            outcome.is_err(),
            "THE HOLE IS CLOSED: the canonical class -> value map now survives \
             a (get-value) query beside (get-model) in a debug build — invert \
             this pin and close the #P2b-34 amendment."
        );
        return;
    }
    let lines = match outcome {
        Ok(lines) => lines,
        Err(_) => panic!("release build must not panic here"),
    };
    assert_eq!(
        verdict(&lines),
        "sat",
        "the release build prints the contradictory interpretation rather than \
         failing, which is what makes the debug assertion the only signal"
    );
}

// ---------------------------------------------------------------------------
// 5. GUARD — the store's own-index read (decision (1b)) is observable after
//    all, on the corpus the fix report called blind to it.
// ---------------------------------------------------------------------------

/// `#P2b-37` records rule (1b) as "implemented as specified but not
/// independently observable at these widths".  It is observable: reverting
/// `register_store_own_index_reads` turns this script from `sat` into
/// `unknown` in an isolated tree copy, reproducibly and in about 70 ms, with
/// no wall-clock budget involved.  The 0.3.4 base answers `unknown` here for
/// the same reason.  This is a guard, not a pin: it must keep answering `sat`.
#[test]
fn the_store_own_index_read_is_what_decides_this_script() {
    let script = "(set-logic QF_AUFBV)\n\
         (declare-fun f ((_ BitVec 2)) (_ BitVec 2))\n\
         (declare-const arr (Array (_ BitVec 1) (_ BitVec 2)))\n\
         (declare-const brr (Array (_ BitVec 1) (_ BitVec 2)))\n\
         (declare-const i (_ BitVec 1))\n\
         (declare-const j (_ BitVec 1))\n\
         (declare-const v (_ BitVec 2))\n\
         (declare-const w (_ BitVec 2))\n\
         (assert (or (= (bvsub (f (f w)) (f (f (f (f w))))) (bvand (select brr j) v)) \
                     (distinct w w)))\n\
         (assert (= (f (f (f (f (f #b01))))) \
                    (f (f (select ((as const (Array (_ BitVec 1) (_ BitVec 2))) #b11) i)))))\n\
         (assert (distinct (select (store arr i (bvor v #b01)) (bvadd i #b1)) \
                           (select (store ((as const (Array (_ BitVec 1) (_ BitVec 2))) #b01) \
                                          #b0 (f (f #b11))) #b1)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "reverting the store's own-index read (#P2b-37 rule (1b)) makes this \
         `unknown`; it is the mutation signal #P2b-39 records as absent"
    );
}

// ---------------------------------------------------------------------------
// 6. PIN — `n`-ary `distinct` over array terms still loses decisions to the
//    refinement's wall-clock budget.
// ---------------------------------------------------------------------------

/// `#P2b-39` reports the `n`-ary-`distinct` blow-up as fixed at the root for
/// the four scripts it names and bounded only for campaign B's `s30028_31`.
/// Fresh scripts of the same shape still blow up: 12 of 573 generated `n`-ary
/// array `distinct` scripts are answered by the 0.3.4 base in about 5 ms and
/// hit a ten-second cap here.  This one is given an explicit `:timeout` so the
/// pin costs a second rather than the two-minute floor, and so the outcome is
/// a verdict rather than a wall-clock flake.
///
/// `#[ignore]`d: it asserts a *slow* answer, which is exactly the kind of
/// assertion a loaded machine can flip in either direction.
#[test]
#[ignore = "timing-shaped: asserts that a shape is still slow, run it explicitly"]
fn n_ary_array_distinct_over_store_chains_still_loses_the_verdict() {
    let script = "(set-logic QF_AUFBV)\n\
         (set-option :timeout 1000)\n\
         (declare-const a (Array (_ BitVec 2) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 2) (_ BitVec 1)))\n\
         (declare-const c (Array (_ BitVec 2) (_ BitVec 1)))\n\
         (assert (distinct (store (store c #b00 #b1) #b11 #b0) \
                           ((as const (Array (_ BitVec 2) (_ BitVec 1))) #b0) \
                           (store (store a #b10 #b0) #b00 #b1) a))\n\
         (assert (= (select (store (store c #b00 #b1) #b11 #b0) #b00) \
                    (select (store (store a #b10 #b0) #b00 #b1) #b00)))\n\
         (assert (= (select ((as const (Array (_ BitVec 2) (_ BitVec 1))) #b0) #b01) \
                    (select (store (store a #b10 #b0) #b00 #b1) #b01)))\n\
         (assert (= (select (store (store c #b00 #b1) #b11 #b0) #b11) (select a #b11)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "unknown",
        "THE HOLE IS CLOSED: this `n`-ary array `distinct` is decided inside a \
         one-second budget again — invert this pin and close #P2b-38 strand (b)."
    );
}
