//! Round-4 adversarial recheck, **pass 4** — the holes this pass found open,
//! and guards for the shapes it re-verified closed.
//!
//! # What this file pins
//!
//! The round's decision (10) lever — a full enumeration of the index domain
//! for array pairs whose domain is at or below
//! `solver/array_axioms.rs::ARRAY_INDEX_ENUMERATION_LIMIT` (8 elements) — buys
//! the pigeonhole refutations that this round is right to want, and pays for
//! them with an unbounded amount of work that **no deterministic budget in the
//! tree can see**.  Measured on this tree against the 0.3.4 base `c4b04b7`,
//! release, both probes back to back:
//!
//! | script | tree | `c4b04b7` |
//! |---|---|---|
//! | `(distinct a0 … a19)` over `(Array (_ BitVec 3) (_ BitVec 1))` | no answer in **900.0 s** (888 s user, 277 MB) | `sat` in 0.4 ms |
//! | the same at n = 12 | no answer in 20 s | `sat` in 0.4 ms |
//! | the same at n = 11 | `sat` in 5.4 s | `sat` in 0.4 ms |
//! | the same at index width 4, n = 11 | `unknown` in 0.39 s | `sat` in 0.26 ms |
//! | 10 `store` terms over one base at index width 3 | no answer in 40 s | `sat` in 8.4 ms |
//!
//! Neither `ARRAY_REFINEMENT_LEMMA_BUDGET` (10 000 lemma instances),
//! `ARRAY_REFINEMENT_RESOLVE_CONFLICTS` (50 000 conflicts) nor
//! `REFINEMENT_WORK_CEILING_PROPAGATIONS` fires: at n = 14 the whole run is
//! **one** refinement round with 91 lemma instances and 1 554 conflicts, and
//! the time goes into the embedded `BvSolver::check` of that single round.  A
//! user's `(set-option :max-conflicts 5000)` does not stop it either — only a
//! wall-clock `(set-option :timeout N)` does, which is precisely the
//! machine-dependence decision (9) exists to remove.
//!
//! Attributed by mutation, in an isolated copy of the tree: with
//! `ARRAY_INDEX_ENUMERATION_LIMIT` set to 0, every script in the table above
//! answers `unknown` in 0.12 s to 0.30 s.
//!
//! Above the limit the lever is skipped and the same shapes answer `unknown`
//! on inputs the base decides — a precision regression rather than a cost one,
//! and the cheap, deterministic symptom this file puts in the gate.
//!
//! # Pins versus guards
//!
//! * A **pin** is green because the tree is still wrong.  Each carries a
//!   `THE HOLE IS CLOSED` message, so the pass that fixes the defect sees it
//!   turn red and knows to invert it.
//! * A **guard** asserts the correct behaviour of something this round really
//!   did close, and turns red if it comes back.
//!
//! No test here asserts a verdict behind a wall-clock `(set-option :timeout)`
//! (decision (16)); the two tests that make a *timing* claim are `#[ignore]`d
//! cost pins and make no verdict claim.

use oxiz_solver::Context;
use std::sync::mpsc;
use std::time::{Duration, Instant};

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

/// `n` pairwise-distinct arrays over `(Array (_ BitVec index_width) (_ BitVec
/// 1))`, optionally under a deterministic `:max-conflicts` budget.
///
/// Carried here in Rust rather than left in a scratch generator so the next
/// pass can re-run the whole cliff table from the repository: the sort has
/// `2 ^ (2 ^ index_width)` inhabitants, so the script is satisfiable exactly
/// when `n` is at most that, which makes the family self-scoring — no oracle
/// is needed to know the right answer.
fn distinct_arrays(n: u32, index_width: u32, max_conflicts: Option<u64>) -> String {
    let sort = format!("(Array (_ BitVec {index_width}) (_ BitVec 1))");
    let mut script = String::from("(set-logic QF_AUFBV)\n");
    if let Some(budget) = max_conflicts {
        script.push_str(&format!("(set-option :max-conflicts {budget})\n"));
    }
    for k in 0..n {
        script.push_str(&format!("(declare-const a{k} {sort})\n"));
    }
    script.push_str("(assert (distinct");
    for k in 0..n {
        script.push_str(&format!(" a{k}"));
    }
    script.push_str("))\n(check-sat)\n");
    script
}

/// `n` `store` terms over one shared base array under a single `distinct`.
///
/// The second half of the same cliff: the pairs here share a base, so the
/// enumeration cost is paid per pair just as it is for free array variables.
fn distinct_store_terms(n: u32, index_width: u32) -> String {
    let sort = format!("(Array (_ BitVec {index_width}) (_ BitVec 1))");
    let mut script = format!("(set-logic QF_AUFBV)\n(declare-const base {sort})\n");
    for k in 0..n {
        script.push_str(&format!("(declare-const i{k} (_ BitVec {index_width}))\n"));
        script.push_str(&format!("(declare-const v{k} (_ BitVec 1))\n"));
    }
    script.push_str("(assert (distinct");
    for k in 0..n {
        script.push_str(&format!(" (store base i{k} v{k})"));
    }
    script.push_str("))\n(check-sat)\n");
    script
}

/// Run `script` on a worker thread and report whether it produced a verdict
/// within `limit`.
///
/// Only ever called from an `#[ignore]`d cost pin: it is a wall-clock
/// observation, and decision (16) keeps those out of the gate.  The worker is
/// deliberately abandoned on a timeout — the test binary is one process per
/// test under `cargo nextest`, so returning from the test ends it.
fn answers_within(script: String, limit: Duration) -> Option<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let answer = verdict(&run(&script));
        let _ = tx.send(answer);
    });
    rx.recv_timeout(limit).ok()
}

// ---------------------------------------------------------------------------
// PINS — green because the tree is still wrong.
// ---------------------------------------------------------------------------

/// **PIN.** Above `ARRAY_INDEX_ENUMERATION_LIMIT` the index domain is not
/// enumerated, the Skolem-witness cascade takes over, and eleven pairwise-
/// distinct arrays over a 65 536-inhabitant sort come back `unknown` — on an
/// input the 0.3.4 base decides `sat` in 0.26 ms and that is trivially
/// satisfiable (eleven distinct elements of a set with 65 536 of them).
///
/// This is the cheap, deterministic, clock-free symptom of the blow-up
/// documented in the module header: 0.39 ms … 0.4 s here, against scripts of
/// the same family at index width 3 that never answer at all.
///
/// No `:timeout` and no duration assertion — the pin is the *verdict*.
#[test]
fn n_ary_array_distinct_above_the_enumeration_limit_is_undecided() {
    // The budget only bounds the run; it changes nothing the verdict depends
    // on.  With and without it the tree reports the same counters — 0
    // conflicts, 55 lemma instances, 222 propagations — and the same
    // `unknown`; c4b04b7 reports 0 conflicts and 56 propagations and answers
    // `sat`.  It is here so the gate pays 30 ms instead of 0.4 s (40 s in the
    // debug-assertion build `cargo nextest` produces).
    let answer = verdict(&run(&distinct_arrays(11, 4, Some(200))));
    assert_eq!(
        answer, "unknown",
        "THE HOLE IS CLOSED: 11 pairwise-distinct arrays over \
         (Array (_ BitVec 4) (_ BitVec 1)) now answer `{answer}`.  The sort has \
         65 536 inhabitants, so `sat` is the right answer and the 0.3.4 base \
         c4b04b7 gives it in 0.26 ms with 0 conflicts; this tree answers \
         `unknown` with 0 conflicts too — the pair enumeration is skipped above \
         ARRAY_INDEX_ENUMERATION_LIMIT and the Skolem cascade cannot decide it. \
         Invert this test to assert `sat` once that is fixed."
    );
}

/// **PIN.** The same family *below* the limit, held to a deterministic budget
/// so the gate can run it at all: twenty pairwise-distinct arrays over an
/// 8-element index domain, with `(set-option :max-conflicts 200)`.
///
/// The base answers `sat` in 0.38 ms having used **zero** conflicts.  This
/// tree cannot answer within two hundred, and — the point of the pin — cannot
/// answer within five thousand either, nor within 900 s of wall clock with no
/// budget at all.  `:max-conflicts` is a deterministic currency, not a clock,
/// so this assertion reproduces on any machine.
#[test]
fn a_deterministic_conflict_budget_does_not_bound_the_enumerated_index_domain() {
    let answer = verdict(&run(&distinct_arrays(20, 3, Some(200))));
    assert_eq!(
        answer, "unknown",
        "THE HOLE IS CLOSED: 20 pairwise-distinct arrays over \
         (Array (_ BitVec 3) (_ BitVec 1)) now answer `{answer}` within 200 \
         conflicts.  c4b04b7 answers `sat` in 0.38 ms with 0 conflicts; this \
         tree needed more than 5 000 conflicts and more than 900 s of wall \
         clock, one refinement round deep, with no lemma, round or propagation \
         budget able to see it.  Invert this test to assert `sat` once the \
         index-domain enumeration is bounded."
    );
}

/// **PIN.** `store` terms over a shared base, ten of them at index width 3,
/// held to the same deterministic budget: the base answers `sat` in 8.4 ms,
/// this tree does not answer in 40 s unbudgeted.
#[test]
fn store_terms_under_an_n_ary_distinct_share_the_same_cliff() {
    let mut script = distinct_store_terms(10, 3);
    script = script.replace(
        "(set-logic QF_AUFBV)\n",
        "(set-logic QF_AUFBV)\n(set-option :max-conflicts 200)\n",
    );
    let answer = verdict(&run(&script));
    assert_eq!(
        answer, "unknown",
        "THE HOLE IS CLOSED: 10 `store` terms over one base at index width 3 \
         now answer `{answer}` within 200 conflicts, where c4b04b7 answers \
         `sat` in 8.4 ms and this tree did not answer in 40 s unbudgeted."
    );
}

/// **PIN.** `reject_reserved_symbol` refuses a `@`- or `.`-leading symbol
/// everywhere a *symbol* is parsed — declarations, sorts, binder variables,
/// `let` bindings, `define-fun` parameters, datatype constructors and
/// selectors, and both the bare and the `|…|` form.  It does **not** reach the
/// `:named` annotation of a term, which takes its label through the attribute
/// path instead.
///
/// Not a capture: referring to `@uc_U_0` as a term is refused, so the label
/// cannot be turned into a symbol in scope, and an unsat core prints only on
/// `unsat`, where there is no model to contradict.  It is an inconsistency in
/// the refusal's coverage, pinned so that closing it is deliberate.
#[test]
fn a_named_annotation_escapes_the_reserved_symbol_refusal() {
    let accepted = run("(set-logic QF_BV)\n\
         (declare-const x (_ BitVec 2))\n\
         (assert (! (= x #b00) :named @uc_U_0))\n\
         (check-sat)\n");
    assert_eq!(
        verdict(&accepted),
        "sat",
        "THE HOLE IS CLOSED: `:named @uc_U_0` is no longer accepted — the \
         reserved-symbol refusal now covers the annotation path too.  Invert \
         this test to assert the parse error."
    );

    // The same spelling in every other position is refused, so the class is
    // closed apart from this one door.
    for script in [
        "(set-logic QF_UF)\n(declare-sort U 0)\n(declare-const @uc_U_0 U)\n(check-sat)\n",
        "(set-logic QF_BV)\n(declare-const x (_ BitVec 2))\n\
         (assert (let ((@a x)) (= @a #b00)))\n(check-sat)\n",
        "(set-logic QF_BV)\n\
         (define-fun f ((@x (_ BitVec 2))) (_ BitVec 2) @x)\n(check-sat)\n",
        "(set-logic QF_BV)\n(declare-const x (_ BitVec 2))\n\
         (assert (! (= x #b00) :named @uc_U_0))\n(assert @uc_U_0)\n(check-sat)\n",
    ] {
        let lines = run(script);
        assert!(
            lines
                .iter()
                .any(|line| line.contains("reserves for solver use")),
            "a `@`-leading symbol must be refused here: {lines:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// GUARDS — the correct behaviour of what this round closed.
// ---------------------------------------------------------------------------

/// **GUARD.** The pigeonhole side of the cardinality work, and the clearest
/// single measure of what this round bought: over `(Array (_ BitVec 1) (_
/// BitVec 1))` there are exactly four arrays, so five pairwise-distinct ones
/// are unsatisfiable.  `c4b04b7` answers a wrong `sat` here — and on 52 of the
/// 150 scripts of this pass's cardinality corpus.
#[test]
fn five_distinct_arrays_over_a_four_inhabitant_sort_are_refuted() {
    for n in 5..=8u32 {
        assert_eq!(
            verdict(&run(&distinct_arrays(n, 1, None))),
            "unsat",
            "{n} pairwise-distinct arrays over a sort with four inhabitants \
             are unsatisfiable; c4b04b7 answers a wrong `sat`"
        );
    }
    // And the satisfiable side of the same sort is still decided.
    for n in 2..=4u32 {
        assert_eq!(
            verdict(&run(&distinct_arrays(n, 1, None))),
            "sat",
            "{n} distinct arrays fit in a sort with four inhabitants"
        );
    }
}

/// **GUARD.** Decision (14) composed with the rest of the array fragment.
///
/// The pass-3 pins covered a `select` through an array-sorted `ite`, one such
/// `ite` under a `store`, and two under `distinct`.  These are the
/// compositions they did not cover, each hand-checked and each independently
/// scored `unsat` by this pass's total-table oracle.
#[test]
fn an_array_ite_composes_with_the_rest_of_the_fragment() {
    const SORT: &str = "(Array (_ BitVec 1) (_ BitVec 1))";
    let cases: [(&str, &str); 6] = [
        (
            "a nested `ite` whose three leaves are all pinned",
            "(declare-const a $S)(declare-const b $S)(declare-const c $S)\
             (declare-const p Bool)(declare-const q Bool)\
             (assert (= (select (ite p (ite q a b) c) #b0) #b1))\
             (assert (= (select a #b0) #b0))\
             (assert (= (select b #b0) #b0))\
             (assert (= (select c #b0) #b0))",
        ),
        (
            "an `ite` whose branches are themselves `store`s",
            "(declare-const a $S)(declare-const b $S)(declare-const p Bool)\
             (assert (= (select (ite p (store a #b0 #b1) (store b #b0 #b1)) #b0) #b0))",
        ),
        (
            "an `ite` equated to an array constant",
            "(declare-const a $S)(declare-const p Bool)\
             (assert (= (ite p a ((as const $S) #b0)) ((as const $S) #b1)))\
             (assert (= (select a #b0) #b0))",
        ),
        (
            "two `ite`s and a third array under a three-operand `distinct`",
            "(declare-const a $S)(declare-const b $S)(declare-const c $S)\
             (declare-const p Bool)(declare-const q Bool)\
             (assert (distinct (ite p a b) (ite q a b) c))\
             (assert (= a b))",
        ),
        (
            "an `ite` as the argument of an array-sorted uninterpreted function",
            "(declare-const a $S)(declare-const b $S)(declare-const p Bool)\
             (declare-fun f ($S) (_ BitVec 1))\
             (assert (= a b))\
             (assert (distinct (f (ite p a b)) (f a)))",
        ),
        (
            "an `ite` at the inner level of an array of arrays",
            "(declare-const n (Array (_ BitVec 1) $S))\
             (declare-const x $S)(declare-const y $S)(declare-const p Bool)\
             (assert (= (select n #b0) (ite p x y)))\
             (assert (= (select (select n #b0) #b0) #b1))\
             (assert (= (select x #b0) #b0))\
             (assert (= (select y #b0) #b0))",
        ),
    ];
    for (name, body) in cases {
        let script = format!(
            "(set-logic QF_AUFBV)\n{}\n(check-sat)\n",
            body.replace("$S", SORT)
        );
        assert_eq!(
            verdict(&run(&script)),
            "unsat",
            "{name}: the read through the array-sorted `ite` is one branch's \
             read or the other's, and both are refuted"
        );
    }
}

/// **GUARD.** An array *variable* equated to a `store` is not a store=store
/// pair, so it never reaches `Solver::array_atoms_need_theory`'s honesty gate;
/// the read-over-write machinery has to decide it on its own, in both operand
/// orders and through a `select` of the variable.
#[test]
fn a_variable_equated_to_a_store_is_decided_in_both_orders() {
    const SORT: &str = "(Array (_ BitVec 1) (_ BitVec 1))";
    for (lhs, rhs) in [("a", "(store b #b0 #b1)"), ("(store b #b0 #b1)", "a")] {
        let script = format!(
            "(set-logic QF_ABV)\n\
             (declare-const a {SORT})\n(declare-const b {SORT})\n\
             (assert (= {lhs} {rhs}))\n\
             (assert (= (select a #b0) #b0))\n(check-sat)\n"
        );
        assert_eq!(verdict(&run(&script)), "unsat", "{lhs} = {rhs}");
    }
}

/// **GUARD.** A positive store=store equality under a connective that does not
/// assert it must not be refuted by the store-extensionality conflict rule.
///
/// The rule at `check_array.rs` walks assertions with an explicit polarity;
/// a disjunct, an implication's consequent, an `ite` arm, a Boolean equality's
/// operand and an `xor` operand are all satisfiable with the equality false,
/// and each was a spurious `unsat` waiting to happen.
#[test]
fn a_store_equality_that_is_not_asserted_is_not_refuted() {
    const SORT: &str = "(Array (_ BitVec 1) (_ BitVec 1))";
    const EQ: &str = "(= (store a #b0 #b1) (store b #b0 #b0))";
    for shape in [
        format!("(or {EQ} p)"),
        format!("(=> p {EQ})"),
        format!("(ite p {EQ} true)"),
        format!("(= p {EQ})"),
        format!("(xor p {EQ})"),
        format!("(not (and (not {EQ}) p))"),
        format!("(distinct p {EQ})"),
    ] {
        let script = format!(
            "(set-logic QF_ABV)\n\
             (declare-const a {SORT})\n(declare-const b {SORT})\n\
             (declare-const p Bool)\n(assert {shape})\n(check-sat)\n"
        );
        assert_eq!(
            verdict(&run(&script)),
            "sat",
            "{shape} is satisfiable with the store equality false"
        );
    }
}

// ---------------------------------------------------------------------------
// COST PINS — timing claims, kept out of the gate by `#[ignore]`.
// ---------------------------------------------------------------------------

/// **COST PIN.** The non-termination itself.  No verdict is asserted: the
/// claim is only that the tree produces *no answer at all* inside a generous
/// budget on a script the 0.3.4 base answers in 0.4 ms.
///
/// Measured unbudgeted with `/usr/bin/time`: 900.03 s real, 888.42 s user,
/// 277 MB resident, killed with no answer.
#[test]
#[ignore = "cost pin: asserts a duration; run it explicitly"]
fn the_index_width_three_cardinality_ladder_does_not_answer() {
    let limit = Duration::from_secs(30);
    let answer = answers_within(distinct_arrays(20, 3, None), limit);
    assert!(
        answer.is_none(),
        "THE HOLE IS CLOSED: 20 pairwise-distinct arrays over \
         (Array (_ BitVec 3) (_ BitVec 1)) answered `{}` inside {limit:?}; \
         c4b04b7 answers `sat` in 0.4 ms and this tree gave no answer in 900 s.",
        answer.unwrap_or_default()
    );
}

/// **COST PIN.** The cliff table the module header quotes, re-measured.  It
/// prints; it asserts only the two verdicts that are cheap and stable, so a
/// loaded machine cannot flip it.
#[test]
#[ignore = "cost pin: seconds per script in release, minutes in debug"]
fn the_cardinality_cliff_table() {
    for (width, n) in [(1u32, 5u32), (2, 8), (2, 12), (3, 8), (3, 11), (4, 11)] {
        let start = Instant::now();
        let answer = answers_within(distinct_arrays(n, width, None), Duration::from_secs(60));
        eprintln!(
            "[cliff] index width {width}, n = {n}: {:?} in {:?}",
            answer,
            start.elapsed()
        );
    }
    assert_eq!(verdict(&run(&distinct_arrays(5, 1, None))), "unsat");
    assert_eq!(verdict(&run(&distinct_arrays(3, 3, None))), "sat");
}
