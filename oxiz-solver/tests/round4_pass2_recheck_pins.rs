//! Regression guards for the defects the round-4 adversarial recheck (pass 2)
//! found on the `#P2b-39` tree, inverted by the pass-3 fix (`#P2b-41`).
//!
//! Every test arrived here as a *pin* — green on the defective tree, asserting
//! what the tree did rather than what it should do, and failing with `THE HOLE
//! IS CLOSED` once the behaviour became correct.  All of them have now been
//! inverted: each asserts the correct behaviour and turns red if the defect
//! returns.  The shapes are the recheck's own, reproduced verbatim so the
//! repository carries its own reproduction of each hole.
//!
//! Two entries are *not* inversions and say so where they sit:
//!
//! * [`two_store_chain_equalities_still_answer_unknown`] — pre-existing
//!   completeness residue, identical on the 0.3.4 base and on crates.io 0.3.3,
//!   kept as a pin so the count stays visible;
//! * [`n_ary_array_distinct_over_store_chains_is_slow_but_decided`] — a
//!   cost pin, `#[ignore]`d because it asserts a timing shape.

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
/// its own witness index and the two witnesses never met.  One declared
/// constant in between was enough to lose it — on this tree, on the 0.3.4 base
/// and on crates.io 0.3.3 alike; 52 of 361 oracle-decided scripts in a
/// generated array-constant-indirection corpus were wrong the same way.
///
/// Closed by `array_axioms::build_const_array_witness_congruence`, which reads
/// a constant's default at the extensionality witness of *every* array pair of
/// its sort rather than only of the pairs it is a member of, so the two
/// defaults meet at one index.  GUARD: this must stay `unsat`.
#[test]
fn a_variable_equal_to_two_different_array_constants_is_refuted() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "unsat",
        "a variable pinned to two array constants with different defaults \
         would have to be two different total functions at once (#P2b-37, the \
         indirect half)"
    );
}

/// The same shape with the indirection spread over two variables, which is how
/// a generated corpus reaches it.  GUARD.
#[test]
fn two_variables_pinned_to_different_array_constants_are_refuted() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (assert (= a ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b1)))\n\
         (assert (= b ((as const (Array (_ BitVec 1) (_ BitVec 1))) #b0)))\n\
         (assert (= a b))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "unsat",
        "two variables pinned to array constants with different defaults and \
         then asserted equal are refuted (#P2b-37, the indirect half)"
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
fn an_array_model_publishes_no_read_its_own_array_constant_forbids() {
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
        !binding.contains("#b01"),
        "a0 is pinned to ((as const A) #b00), so no entry of its printed chain \
         may carry another value.  Printed binding: {binding}"
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
fn get_model_prints_a_quoted_symbol_with_its_bars() {
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
        model.contains("(define-fun |a b| ()"),
        "a symbol that needs |…| in the source needs it in the output too, or \
         the model block is not re-parsable SMT-LIB (one `format_symbol` in \
         oxiz-core::smtlib::printer serves both printers).  Model: {model}"
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
// 6. GUARD — `n`-ary `distinct` over array terms is decided, with no clock.
// ---------------------------------------------------------------------------

/// The cost half of decision (10), re-measured and inverted.
///
/// This was a **pin**: the script answered `sat` in about 2.5 s against the
/// 0.3.4 base's 6.2 ms — some 400x, where decision (10) asked for "a small
/// constant factor of the base's time" — so with a one-second `:timeout` it
/// answered `unknown`, and the pin recorded that as an open cost.
///
/// Decision (10)'s enumeration lever changed the measurement, not the verdict
/// rule: over an index sort the solver can write out, extensionality is a
/// finite conjunction at the domain's own elements rather than a fresh Skolem
/// witness per pair whose reads feed the BV<->EUF partition exchange
/// (`array_axioms::ARRAY_INDEX_ENUMERATION_LIMIT`).  Re-measured on this tree
/// in a release build, best of three, both probes back to back: 0.45 ms here
/// against the base's 0.16 ms.
///
/// GUARD, and deliberately with **no `:timeout`**: decision (16) forbids a test
/// that asserts a verdict behind a wall clock, because the colour of such a
/// test is a property of how busy the machine is rather than of the solver.
/// The timing claim lives in the `#[ignore]`d cost pin below.
#[test]
fn n_ary_array_distinct_over_store_chains_is_decided() {
    let script = N_ARY_ARRAY_DISTINCT;
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "this `n`-ary array `distinct` is decided by the deterministic \
         refinement budget, with no clock in the script (#P2b-38 strand (b))"
    );
}

/// The timing claim the guard above deliberately does not make.
///
/// `#[ignore]`d because it asserts a *duration*, which is exactly the kind of
/// assertion a loaded machine flips in either direction — decision (16) keeps
/// such claims out of the gate and here, where they can be run on purpose.
///
/// Release, best of three, tree against `c4b04b7`: 0.45 ms against 0.16 ms.
/// The bound below is the debug build's, with a wide margin: the number to
/// watch is the *ratio* in the report, not this ceiling.
#[test]
#[ignore = "timing-shaped: asserts a duration, run it explicitly"]
fn n_ary_array_distinct_over_store_chains_is_fast() {
    let start = std::time::Instant::now();
    let answer = verdict(&run(N_ARY_ARRAY_DISTINCT));
    let elapsed = start.elapsed();
    assert_eq!(answer, "sat");
    assert!(
        elapsed < std::time::Duration::from_millis(500),
        "the shape took {elapsed:?}; it was 0.45 ms in release and 2.5 s before \
         decision (10)'s enumeration lever"
    );
}

/// The script both of the two above use: four array operands under one
/// `distinct`, two of them store chains over the same base, with three reads
/// pinning them together.
const N_ARY_ARRAY_DISTINCT: &str = "(set-logic QF_AUFBV)\n\
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

// ---------------------------------------------------------------------------
// 7. PIN — solver-minted proxy names are spellable, so a script can capture
//    them and turn a satisfiable formula into `unsat` (the class `#P2b-39`
//    closed for two array prefixes only).
// ---------------------------------------------------------------------------

/// `#P2b-39` reserved exactly two names behind a backslash — the array
/// extensionality witness and the off-chain Skolem index.  The defect class is
/// wider: every fresh symbol the encoder minted was built with `format!` from
/// a `TermId` out of characters SMT-LIB 2.6 §3.1 admits in a *simple* symbol
/// (`$`, `!`, `-`, digits, letters), and `TermManager::mk_var` interns on
/// `(name, sort)`, so a user declaration at the same sort **was** the same
/// term.
///
/// `encode::numeric_purification` mints a proxy for every non-variable numeric
/// argument of an uninterpreted application and asserts the side condition
/// `proxy = arg`.  While the proxy was spelled `$encode-numarg!{arg.0}`,
/// declaring two of those names and asserting them equal asserted
/// `(+ x 1) = (+ x 2)`, which is false, so this script — whose two extra
/// constants are entirely free — answered `unsat`.
///
/// Closed by moving the *class* into `oxiz_core::smtlib::reserved_name`, whose
/// `\oxiz.` prefix neither SMT-LIB symbol form can produce.  The ids 4 and 7
/// are this tree's; they came from a 64-name grid narrowed by delta debugging.
/// GUARD: this satisfiable script must stay `sat`.
#[test]
fn a_numeric_proxy_name_is_not_spellable_and_the_script_stays_sat() {
    let script = "(set-logic QF_UFLIA)\n\
         (declare-fun f (Int) Int)\n\
         (declare-const x Int)\n\
         (assert (distinct (f (+ x 1)) (f (+ x 2))))\n\
         (declare-const $encode-numarg!4 Int)\n\
         (declare-const $encode-numarg!7 Int)\n\
         (assert (= $encode-numarg!4 $encode-numarg!7))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "the two extra constants are free, so the script is satisfiable \
         (x = 0, f(1) = 0, f(2) = 1, both constants 0); an `unsat` means a \
         declaration captured a solver-minted proxy again"
    );
}

/// The same capture through `encode::bool_euf_encoding`'s Boolean argument
/// proxy, whose side condition is the same `mk_eq(v, arg)`.  Kept separate
/// because the two are minted by different modules and a fix that reserved one
/// could have missed the other.  GUARD.
#[test]
fn a_boolean_proxy_name_is_not_spellable_and_the_script_stays_sat() {
    let script = "(set-logic QF_UFBV)\n\
         (declare-fun g (Bool) (_ BitVec 4))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (assert (distinct (g (and p q)) (g (or p q))))\n\
         (declare-const $encode-bool-arg!4 Bool)\n\
         (declare-const $encode-bool-arg!6 Bool)\n\
         (assert (= $encode-bool-arg!4 $encode-bool-arg!6))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "the two extra Boolean constants are free, so the script is \
         satisfiable; an `unsat` means a declaration captured the encoder's \
         Boolean argument proxy again"
    );
}

/// The control that keeps both pins honest: the same shapes with names that
/// merely *look* like the solver's are satisfiable, so it is the collision and
/// nothing else that produces the `unsat` above.  A guard, not a pin.
#[test]
fn the_same_shapes_with_non_colliding_names_are_satisfiable() {
    let numeric = "(set-logic QF_UFLIA)\n\
         (declare-fun f (Int) Int)\n\
         (declare-const x Int)\n\
         (assert (distinct (f (+ x 1)) (f (+ x 2))))\n\
         (declare-const $harmless-numarg!4 Int)\n\
         (declare-const $harmless-numarg!7 Int)\n\
         (assert (= $harmless-numarg!4 $harmless-numarg!7))\n\
         (check-sat)\n";
    let boolean = "(set-logic QF_UFBV)\n\
         (declare-fun g (Bool) (_ BitVec 4))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (assert (distinct (g (and p q)) (g (or p q))))\n\
         (declare-const $harmless-bool-arg!4 Bool)\n\
         (declare-const $harmless-bool-arg!6 Bool)\n\
         (assert (= $harmless-bool-arg!4 $harmless-bool-arg!6))\n\
         (check-sat)\n";
    assert_eq!(verdict(&run(numeric)), "sat", "the control must stay sat");
    assert_eq!(verdict(&run(boolean)), "sat", "the control must stay sat");
}

// ---------------------------------------------------------------------------
// 8. GUARD — decision (9): the array refinement's budget is deterministic.
// ---------------------------------------------------------------------------

/// Decision (9) of this round required the array refinement's re-solve budget
/// to be *deterministic* — refinement rounds, lemma instances or conflicts —
/// so a verdict cannot depend on the machine, the build profile or the load,
/// keeping an explicit user `:timeout` as the only wall-clock limit.
///
/// It used to be `array_refinement_resolve_deadline`, built from
/// `oxiz_time::Instant::now()` with a 120 s floor and a `× 20` factor over the
/// time already spent, armed on every check the caller gave no `:timeout`.
/// Measured consequence, same release binary and the same script with no
/// `:timeout` anywhere: the four-operand `distinct` over store chains of
/// [`n_ary_distinct_over_store_chains_is_decided`] answered `sat` at 77.5 s run
/// alone and `unknown` at 120.0 s with ten copies running at once.
///
/// It is now `ARRAY_REFINEMENT_RESOLVE_CONFLICTS`, a ceiling on
/// `SolverStats::conflicts` counted from the round that asserts the first array
/// lemma.  Two halves are asserted here: the wall-clock budget is gone, and a
/// conflict ceiling is what replaced it.  The behavioural half — the same
/// script answering `sat` alone and under ten-way load — is measured outside
/// the suite, because a test that reproduces it has to create the load.
///
/// This guard was `include_str!` + `.contains(…)` on `check_core.rs` — it
/// checked two identifiers in one file and would have stayed green if a
/// wall-clock budget had returned in another.  Decision (18) replaced it with
/// the behavioural statement it was standing in for: the refinement's work
/// counters, which the search itself advances, are identical on an idle and on
/// a loaded process, and so is the verdict they bound.
///
/// `(get-info :all-statistics)` publishes both currencies
/// (`:array-refinement-rounds`, `:array-lemma-instances`) beside `:conflicts`,
/// so the claim is observable from outside the crate rather than asserted about
/// its source text.
#[test]
fn the_array_refinement_budget_is_deterministic() {
    let script = "(set-logic QF_AUFBV)\n\
         (declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const brr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const i0 (_ BitVec 1))\n\
         (declare-const v0 (_ BitVec 1))\n\
         (assert (distinct (store arr i0 v0) brr))\n\
         (assert (= (select (store arr i0 v0) #b0) (select brr #b0)))\n\
         (assert (= (select (store arr i0 v0) #b1) (select brr #b1)))\n\
         (check-sat)\n\
         (get-info :all-statistics)\n";
    let observe = |lines: &[String]| -> (String, String) {
        let statistics = lines
            .iter()
            .find(|line| line.contains(":conflicts"))
            .cloned()
            .unwrap_or_default();
        (verdict(lines), statistics)
    };

    let idle = observe(&run(script));
    assert_eq!(idle.0, "unsat", "the script is refuted by extensionality");
    assert!(
        idle.1.contains(":array-refinement-rounds") && idle.1.contains(":array-lemma-instances"),
        "the refinement's deterministic currencies are published: {}",
        idle.1
    );

    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let burners: Vec<std::thread::JoinHandle<u64>> = (0..6)
        .map(|_| {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut acc: u64 = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    for i in 0..100_000u64 {
                        acc = acc.wrapping_add(i).rotate_left(7) ^ i;
                    }
                }
                acc
            })
        })
        .collect();
    let loaded = observe(&run(script));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for burner in burners {
        let _ = burner.join();
    }

    assert_eq!(
        idle, loaded,
        "a budget that reads the clock makes the verdict — and the work done to \
         reach it — a property of the machine (decision (9), #P2b-38 strand (c))"
    );
}

// ---------------------------------------------------------------------------
// 9. GUARD — decision (10): `n`-ary `distinct` over store chains is decided.
// ---------------------------------------------------------------------------

/// Decision (10) required `#P2b-38` strand (b) to be fixed *at the root*.
/// The cost centre was the eager extensionality family: all `C(n,2)` pairs of
/// an `n`-ary `distinct` got a fresh witness index and a full
/// `collect_pair_indices` congruence family before anything asked whether the
/// assignment already kept those pairs apart, and every one of those indices
/// is a bit-vector argument term feeding the BV↔EUF partition exchange.
/// `Solver::pair_polarity` now reads that off the SAT trail first
/// (`held_apart` / `held_equal` / `separated_by_reads`), and the chain-index
/// half became its own deferred phase, one pair per round.
///
/// This is the smallest script of the family: index width 1 and element width
/// 1, so the whole array sort has four inhabitants.  It answered nothing
/// inside 10 s before (77.5 s unloaded in a release build) and answers `sat` in
/// about 90 ms now, against the 0.3.4 base's 0.3 ms.  The `:timeout` makes a
/// returning blow-up show up as a verdict rather than as a wall-clock flake.
#[test]
fn n_ary_distinct_over_store_chains_is_decided() {
    let script = "(set-logic QF_AUFBV)\n\
         (declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const brr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const i0 (_ BitVec 1))\n\
         (declare-const i1 (_ BitVec 1))\n\
         (declare-const i2 (_ BitVec 1))\n\
         (declare-const v0 (_ BitVec 1))\n\
         (declare-const v1 (_ BitVec 1))\n\
         (declare-const v2 (_ BitVec 1))\n\
         (assert (distinct \
           (store (store (store (store arr i0 v0) i2 v2) i2 (select arr #b1)) i2 (select arr i0)) \
           (store (store (store brr i2 v0) #b1 (select arr i1)) i0 v1) \
           (store (store (store arr i1 v0) #b0 v2) i0 v0) \
           (store (store (store brr i0 v0) i0 #b0) i1 v1)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "sat",
        "the n-ary `distinct` over store chains must answer the base's verdict, \
         and must do it on the deterministic budget rather than on a clock \
         (#P2b-38 strand (b), decision (16))"
    );
}

/// The control for the pin above: the same two leading operands under a binary
/// `distinct` answer in milliseconds, so it is the `n`-ary expansion and not
/// the store chains alone.  A guard, not a pin.
#[test]
fn the_same_store_chains_under_a_binary_distinct_are_fast() {
    let script = "(set-logic QF_AUFBV)\n\
         (declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const brr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const i0 (_ BitVec 1))\n\
         (declare-const i1 (_ BitVec 1))\n\
         (declare-const i2 (_ BitVec 1))\n\
         (declare-const v0 (_ BitVec 1))\n\
         (declare-const v1 (_ BitVec 1))\n\
         (declare-const v2 (_ BitVec 1))\n\
         (assert (distinct \
           (store (store (store (store arr i0 v0) i2 v2) i2 (select arr #b1)) i2 (select arr i0)) \
           (store (store (store brr i2 v0) #b1 (select arr i1)) i0 v1)))\n\
         (check-sat)\n";
    assert_eq!(verdict(&run(script)), "sat");
}

// ---------------------------------------------------------------------------
// 10. GUARD — decision (11): `#P2b-40` is fixed.
// ---------------------------------------------------------------------------

/// Decision (11) required `#P2b-40` to be fixed by minting `@uc_S_n` witnesses
/// for uninterpreted-sort classes with no declared constant, numbered after
/// the declared ones.  `g` used to print as the constant `@uc_U_0`, so the
/// published model made `(g p) = (g q)` and falsified the script's only
/// assertion — byte-identical on the 0.3.4 base, so residue rather than
/// regression, and fixed here.  `build_class_values` now walks the remaining
/// EUF classes of each uninterpreted sort after the declared constants and
/// mints a witness for each, numbered from the declared count upward so the
/// user-visible numbering of `p` and `q` does not move.  GUARD.
#[test]
fn an_uninterpreted_return_sort_prints_a_distinct_witness_per_class() {
    let script = "(set-logic QF_UF)\n\
         (declare-sort U 0)\n\
         (declare-fun g (U) U)\n\
         (declare-const p U)\n\
         (declare-const q U)\n\
         (assert (distinct (g p) (g q)))\n\
         (check-sat)\n\
         (get-model)\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat");
    let model = model_block(&lines);
    assert!(
        model.contains("(define-fun p () U @uc_U_0)")
            && model.contains("(define-fun q () U @uc_U_1)"),
        "the declared constants keep the numbering they had.  Model:\n{model}"
    );
    assert!(
        !model.contains("(define-fun g ((x!0 U)) U @uc_U_0)"),
        "`g` must not print as a constant function: that model makes \
         (g p) = (g q) and falsifies the script's only assertion (#P2b-40).  \
         Model was:\n{model}"
    );
    // The two applications must land on different witnesses, which is what
    // `(distinct (g p) (g q))` demands of any model of this script.
    assert!(
        model.contains("@uc_U_2") && model.contains("@uc_U_3"),
        "(g p) and (g q) must take two different witnesses.  Model:\n{model}"
    );
}

// ---------------------------------------------------------------------------
// 11. GUARD — `(get-value)` answers a read through an `ite` correctly.
// ---------------------------------------------------------------------------

/// A `select` whose array argument is an `ite` over two arrays used to be
/// answered with the *else* branch's value, contradicting the model the same
/// run printed: `p` is `true` and `a` reads `#b1` at `#b0`, so the answer must
/// be `#b1`, and the answer was `#b0`.
///
/// That was worse than the echo it replaced — the 0.3.4 base answers
/// `((select (ite p a b) #b0) (select (ite p a b) #b0))`, an echo a consumer
/// can detect.  Root cause was `model_fmt::get_value::array_read_value`, whose
/// `array_class_read(...).unwrap_or_else(|| self.default_value(range))` turned
/// "no class for this array term" into the sort default instead of resolving
/// the `ite`.  `resolve_array_branch` folds the condition through the model
/// first, and the sort-default fallback is now gated on the renderer
/// describing the array at all.  GUARD.
#[test]
fn get_value_answers_a_read_through_an_ite_from_the_printed_model() {
    let script = "(set-logic QF_ABV)\n\
         (declare-const a (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const b (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (assert (= (select a #b0) #b1))\n\
         (assert (= (select b #b0) #b0))\n\
         (assert p)\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((select (ite p a b) #b0)))\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat");
    let answer = value_block(&lines);
    assert!(
        answer.contains("((select (ite p a b) #b0) #b1)"),
        "`p` is true and `a` reads #b1 at #b0, so the read through the `ite` \
         must answer #b1 — a (get-value) answer that contradicts the same \
         run's (get-model) is the one failure mode a consumer's model check \
         cannot catch.  Response was: {answer}"
    );
}

/// A datatype selector whose result sort is an *array* used to be echoed while
/// the bit-vector selector beside it folded to a value; decision (4) requires
/// `(get-value)` never to echo.  `array_query_value` routes an array-sorted
/// query term through the same renderer `(get-model)` prints from.  GUARD.
#[test]
fn get_value_answers_an_array_sorted_datatype_selector_with_a_chain() {
    let script = "(set-logic QF_AUFBVDT)\n\
         (declare-datatypes ((Box 0)) \
           (((mk (arr (Array (_ BitVec 1) (_ BitVec 1))) (tag (_ BitVec 2))))))\n\
         (declare-const b Box)\n\
         (assert (= (tag b) #b01))\n\
         (assert (= (select (arr b) #b0) #b1))\n\
         (check-sat)\n\
         (get-value ((arr b) (tag b)))\n";
    let answer = value_block(&run(script));
    assert!(
        !answer.contains("((arr b) (arr b))"),
        "an array-sorted datatype selector must not echo (decision (4)).  \
         Response was: {answer}"
    );
    assert!(
        answer.contains("((arr b) (store ") || answer.contains("((arr b) ((as const "),
        "an array value is a store chain or an `(as const …)` value.  \
         Response was: {answer}"
    );
}

// ---------------------------------------------------------------------------
// 12. GUARD — decision (12): the datatype constructor value reads the model.
// ---------------------------------------------------------------------------

/// Family 1 of decision (12).  The model used to print `b = (mk … #b00)`, so
/// `(tag b)` was `#b00` where the script asserts `#b01`, while
/// `(get-value ((tag b)))` in the same run answered `#b01`: the two printers
/// disagreed and the printed model falsified its own script.  153 of 300
/// generated datatype scripts showed the disagreement, byte-identical on the
/// 0.3.4 base.
///
/// `Context::datatype_class_value` now builds the constructor value from the
/// values this model gives the *selector applications* the script spells,
/// instead of from the sort defaults, so the two printers read one model.
/// GUARD.
#[test]
fn a_datatype_value_agrees_with_get_value_and_with_its_own_script() {
    let script = "(set-logic QF_AUFBVDT)\n\
         (declare-datatypes ((Box 0)) \
           (((mk (arr (Array (_ BitVec 1) (_ BitVec 1))) (tag (_ BitVec 2))))))\n\
         (declare-const b Box)\n\
         (assert (= (tag b) #b01))\n\
         (assert (= (select (arr b) #b0) #b1))\n\
         (check-sat)\n\
         (get-model)\n\
         (get-value ((tag b)))\n";
    let lines = run(script);
    assert_eq!(verdict(&lines), "sat");
    let binding = model_binding(&model_block(&lines), "b");
    assert!(
        binding.contains("#b01"),
        "the printed constructor must carry the asserted tag #b01, not the \
         sort default.  Binding: {binding}"
    );
    assert!(
        value_block(&lines).contains("((tag b) #b01)"),
        "(get-value) answers the asserted value, and (get-model) must agree \
         with it — one reading of one model"
    );
}

// ---------------------------------------------------------------------------
// 13. PIN — completeness residue: two store-chain equalities answer `unknown`.
// ---------------------------------------------------------------------------

/// 159 of 1,200 exhaustively scored array/UF scripts answer `unknown` on
/// inputs the total-table oracle decides (158 of them satisfiable), in under
/// 4 ms — so no wall-clock budget is involved, the refinement simply gives up.
/// This is the minimal member: two equalities between store chains over a
/// one-bit index sort, which has four inhabitants in total.
///
/// `unknown` is sound, and the 0.3.4 base and crates.io 0.3.3 answer `unknown`
/// here too, so this is pre-existing residue and **not** a round-4 regression
/// — the pin exists to keep the count visible.
#[test]
fn two_store_chain_equalities_still_answer_unknown() {
    let script = "(set-logic QF_AUFBV)\n\
         (declare-const arr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const brr (Array (_ BitVec 1) (_ BitVec 1)))\n\
         (declare-const i0 (_ BitVec 1))\n\
         (declare-const v0 (_ BitVec 1))\n\
         (declare-const v1 (_ BitVec 1))\n\
         (declare-fun f ((_ BitVec 1)) (_ BitVec 1))\n\
         (assert (= (store arr #b0 v1) (store brr i0 v0)))\n\
         (assert (= v1 (f #b0)))\n\
         (assert (= (store arr #b0 v0) (store brr i0 v1)))\n\
         (check-sat)\n";
    assert_eq!(
        verdict(&run(script)),
        "unknown",
        "this satisfiable script is decided now — re-run the exhaustive \
         campaign (159 of 1,200 answered `unknown` on decided inputs; the exit \
         is Context::check_sat_core's array honesty gate, not a budget) and \
         invert this pin."
    );
}
