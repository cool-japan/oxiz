//! Round-4 adversarial recheck, **pass 4** — the recheck's pins, inverted into
//! regression guards by re-fix pass 5 (`#P2b-46`), beside the guards the
//! recheck wrote for what the round had already closed.
//!
//! # What this file guards
//!
//! The round's decision (10) lever — a full enumeration of the index domain
//! for array pairs whose domain is at or below
//! `solver/array_axioms.rs::ARRAY_INDEX_ENUMERATION_LIMIT` (8 elements) — buys
//! the pigeonhole refutations that this round is right to want, and paid for
//! them with an amount of work that **no deterministic budget in the tree
//! could see**.  As the recheck measured it against the 0.3.4 base `c4b04b7`,
//! release, both probes back to back:
//!
//! | script | tree, before `#P2b-46` | `c4b04b7` |
//! |---|---|---|
//! | `(distinct a0 … a19)` over `(Array (_ BitVec 3) (_ BitVec 1))` | no answer in **900.0 s** (888 s user, 277 MB) | `sat` in 0.4 ms |
//! | the same at index width 4, n = 11 | `unknown` in 0.39 s | `sat` in 0.26 ms |
//! | 10 `store` terms over one base at index width 3 | no answer in 40 s | `sat` in 8.4 ms |
//!
//! Neither `ARRAY_REFINEMENT_LEMMA_BUDGET` (10,000 lemma instances),
//! `ARRAY_REFINEMENT_RESOLVE_CONFLICTS` (50,000 conflicts) nor
//! `REFINEMENT_WORK_CEILING_PROPAGATIONS` fired: at n = 14 the whole run was
//! **one** refinement round with 91 lemma instances and 1,554 conflicts, and
//! the time went into the embedded `BvSolver::check` of that single round.
//!
//! # What `#P2b-46` changed, and what it did not
//!
//! * **The `unknown` above the enumeration limit is gone.**  The BV↔EUF
//!   partition-lemma exchange's round bound was 512, which cannot rule out the
//!   partitions of the `C(11,2) = 55` Skolem witness indices that eleven
//!   pairwise-distinct arrays mint; it gave up and the verdict was lost.  The
//!   bound is 8,192 now and the script answers `sat`.  Guarded by
//!   `round4_recheck_regressions::array_cardinality_above_the_enumeration_limit_is_decided`,
//!   which is where the inversion of this file's first pin lives (it costs
//!   about 110 s in this profile, so it is not duplicated here).
//! * **The unbounded run is bounded**, in a deterministic currency and not by
//!   a clock: `Statistics::bv_embedded_checks` counts complete checks of the
//!   embedded bit-blasted solver, and `BV_EMBEDDED_CHECK_CEILING` (250,000)
//!   ends the check with `Unknown` when they run out.  `(distinct a0 … a19)`
//!   at index width 3 now answers `unknown` after exactly **250,002** checks,
//!   as does the ten-`store` script, on any machine.
//! * **It is not fast.**  Decision (10) stays open and is reported as open:
//!   250,002 embedded checks is about six minutes here against the base's
//!   0.4 ms.  What changed is that the answer *arrives*, and that the work
//!   before it does is a property of the formula rather than of the host.
//!
//! # Pins versus guards
//!
//! * A **pin** is green because the tree is still wrong.  Each carries a
//!   `THE HOLE IS CLOSED` message, so the pass that fixes the defect sees it
//!   turn red and knows to invert it.  **There are none left in this file.**
//! * A **guard** asserts the correct behaviour of something that is closed,
//!   and turns red if it comes back.
//!
//! No test here asserts a verdict behind a wall-clock `(set-option :timeout)`
//! (decision (16)); the tests that make a *timing* claim are `#[ignore]`d cost
//! pins and make no verdict claim.

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
// GUARDS — the inverted pins of the pass-4 recheck (`#P2b-46`).
// ---------------------------------------------------------------------------

/// **GUARD** (inverted pin).  The enumerated index domain is bounded by a
/// *deterministic* budget.
///
/// The pin this replaces asserted that no budget in the tree could see the
/// blow-up: a fourteen-line `(distinct a0 … a19)` over
/// `(Array (_ BitVec 3) (_ BitVec 1))` ran 900.03 s with no answer, a user's
/// `(set-option :max-conflicts 5000)` did not stop it, and only a wall-clock
/// `:timeout` did.  `Statistics::bv_embedded_checks` is the currency that does
/// see it — it counts complete checks of the embedded bit-blasted solver,
/// which is where that single refinement round spends everything — and
/// `BV_EMBEDDED_CHECK_CEILING` ends the check with `Unknown` when they run
/// out.
///
/// What the gate checks is cheap and clock-free: the counter is published,
/// it is non-zero on this family, and two runs of the same script report the
/// *same* count and the same verdict.  A count that reproduces is the whole
/// claim — a budget denominated in something that moves with machine load
/// would be the wall clock again under another name.  The expensive half, that
/// an unbudgeted twenty-array script really does come back, is the
/// `#[ignore]`d cost pin [`the_index_width_three_cardinality_ladder_terminates`]
/// below.
#[test]
fn the_enumerated_index_domain_is_bounded_by_a_deterministic_budget() {
    let script = format!(
        "{}(get-info :all-statistics)\n",
        distinct_arrays(20, 3, Some(200))
    );
    let first = run(&script);
    let second = run(&script);
    assert_eq!(
        verdict(&first),
        verdict(&second),
        "the same script must reach the same verdict twice"
    );
    let count = |lines: &[String]| -> u64 {
        let line = lines
            .iter()
            .find(|line| line.contains(":bv-embedded-checks"))
            .unwrap_or_else(|| {
                panic!("(get-info :all-statistics) must publish :bv-embedded-checks: {lines:?}")
            })
            .clone();
        let tail = line
            .split(":bv-embedded-checks ")
            .nth(1)
            .unwrap_or_default()
            .to_string();
        tail.trim_end_matches(')')
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("a numeric :bv-embedded-checks count: {line}"))
    };
    let (a, b) = (count(&first), count(&second));
    assert!(
        a > 0,
        "this family really does run embedded checks; a zero count would mean \
         the budget is denominated in something this shape never touches"
    );
    assert_eq!(
        a, b,
        "the budget's currency must be deterministic: two runs of the \
         same script reported {a} and {b} embedded checks"
    );
}

/// **GUARD** (inverted pin).  `store` terms under an `n`-ary `distinct` share
/// the family, and share the bound: ten `store` terms over one base at index
/// width 3, under a deterministic `:max-conflicts`, come back with a verdict
/// rather than running until a clock stops them.
///
/// The recheck's pin asserted the same `unknown` for the opposite reason — the
/// budget cut it off while nothing bounded the unbudgeted run, which did not
/// answer in 40 s.  Unbudgeted it now answers `unknown` after 250,002 embedded
/// checks (124.6 s in release); that measurement is in
/// [`the_store_term_cliff_terminates`].
#[test]
fn store_terms_under_an_n_ary_distinct_are_decided_under_a_deterministic_budget() {
    let mut script = distinct_store_terms(10, 3);
    script = script.replace(
        "(set-logic QF_AUFBV)\n",
        "(set-logic QF_AUFBV)\n(set-option :max-conflicts 200)\n",
    );
    let answer = verdict(&run(&script));
    assert!(
        answer == "unknown" || answer == "sat",
        "a budgeted run must report a verdict, not `{answer}`"
    );
}

/// **GUARD** (inverted pin).  `reject_reserved_symbol` refuses a `@`- or
/// `.`-leading symbol everywhere a *symbol* enters a namespace — declarations,
/// sorts, binder variables, `let` bindings, `define-fun` parameters, datatype
/// constructors and selectors, both the bare and the `|…|` form, **and** the
/// `:named` annotation of a term, which took its label through the attribute
/// path and used to escape (`#P2b-46`, decision (19)).
///
/// The `:named` label was never a capture — referring to `@uc_U_0` as a term
/// was already refused, so the label could not become a symbol in scope, and
/// an unsat core prints only on `unsat`, where there is no model to contradict
/// — but it was an inconsistency in the refusal's coverage, and a rule with a
/// door in it is a rule a later pass has to re-derive.
///
/// Every *other* attribute value is deliberately left alone: `:source`,
/// `:status` and friends name nothing and benchmark files in the wild fill
/// them with arbitrary symbols.
#[test]
fn a_named_annotation_is_covered_by_the_reserved_symbol_refusal() {
    // Every position, including the one that used to escape.
    for script in [
        // The `:named` label itself — the door this closes.
        "(set-logic QF_BV)\n(declare-const x (_ BitVec 2))\n\
         (assert (! (= x #b00) :named @uc_U_0))\n(check-sat)\n",
        // …and its `.`-leading twin, the other reserved class.
        "(set-logic QF_BV)\n(declare-const x (_ BitVec 2))\n\
         (assert (! (= x #b00) :named .hidden))\n(check-sat)\n",
        // The spellings the recheck witnessed refused, re-asserted here so
        // this guard cannot go green by the rule being deleted.
        "(set-logic QF_UF)\n(declare-sort U 0)\n(declare-const @uc_U_0 U)\n(check-sat)\n",
        "(set-logic QF_BV)\n(declare-const x (_ BitVec 2))\n\
         (assert (let ((@a x)) (= @a #b00)))\n(check-sat)\n",
        "(set-logic QF_BV)\n\
         (define-fun f ((@x (_ BitVec 2))) (_ BitVec 2) @x)\n(check-sat)\n",
        "(set-logic QF_BV)\n(declare-const x (_ BitVec 2))\n\
         (assert (! (= x #b00) :named n0))\n(assert @uc_U_0)\n(check-sat)\n",
    ] {
        let lines = run(script);
        assert!(
            lines
                .iter()
                .any(|line| line.contains("reserves for solver use")),
            "a `@`- or `.`-leading symbol must be refused here: {lines:?}"
        );
    }

    // The control: a label that is not in the reserved class still works, so
    // the refusal is not "no `:named` at all".
    let accepted = run("(set-logic QF_BV)\n\
         (declare-const x (_ BitVec 2))\n\
         (assert (! (= x #b00) :named ordinary_label))\n\
         (check-sat)\n");
    assert_eq!(
        verdict(&accepted),
        "sat",
        "an ordinary `:named` label is unaffected: {accepted:?}"
    );
    // And a `@` *inside* a symbol, which SMT-LIB 2.6 allows, still parses.
    let infix = run("(set-logic QF_BV)\n\
         (declare-const x (_ BitVec 2))\n\
         (assert (! (= x #b00) :named a@b))\n\
         (check-sat)\n");
    assert_eq!(
        verdict(&infix),
        "sat",
        "only a *leading* `@` is reserved: {infix:?}"
    );
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

/// **COST PIN** (inverted).  The non-termination is gone, and the measurement
/// that shows it is expensive, so it lives here rather than in the gate.
///
/// Before `#P2b-46` this script produced *no answer at all*: 900.03 s real,
/// 888.42 s user and 277 MB resident under `/usr/bin/time`, killed with
/// nothing printed, where the 0.3.4 base answers `sat` in 0.4 ms.  It now
/// comes back `unknown` after exactly 250,002 embedded bit-blasted checks —
/// `BV_EMBEDDED_CHECK_CEILING` — which took 346.6 s in a release build on the
/// development machine.
///
/// The assertion is the *verdict*, not the duration: the duration is what
/// makes this `#[ignore]`d, and the budget that ends the run is counted in
/// checks, so the same script stops after the same amount of work on any
/// machine.  Decision (10) is **not** closed by this and is not claimed to be:
/// `unknown` in minutes is not `sat` in 0.4 ms.  What changed is that an
/// answer arrives without a wall clock being involved.
///
/// # Release-calibrated, and it declines rather than hangs
///
/// The profile `cargo nextest` builds carries `debug_assertions`, and the
/// bit-blaster's circuit self-check is gated on exactly that: the same script
/// this pin answers in 346.6 s in a release build did not finish in 1,800 s
/// there, and the sibling guard
/// `round4_recheck_regressions::array_cardinality_above_the_enumeration_limit_is_decided`
/// measures the same ratio directly (0.91 s release against 114 s). Rather
/// than ship a pin that cannot pass where it is usually run, it declines under
/// `debug_assertions` with a message. Run it with
/// `cargo nextest run -p oxiz-solver --test round4_pass4_recheck_pins
/// --cargo-profile release --run-ignored all`.
#[test]
#[ignore = "cost pin: minutes per script in release; declines under debug_assertions"]
fn the_index_width_three_cardinality_ladder_terminates() {
    if cfg!(debug_assertions) {
        eprintln!(
            "declined: release-calibrated (346.6 s release, > 1,800 s with \
             debug_assertions); re-run with --cargo-profile release"
        );
        return;
    }
    let limit = Duration::from_secs(3_600);
    let answer = answers_within(distinct_arrays(20, 3, None), limit);
    assert_eq!(
        answer.as_deref(),
        Some("unknown"),
        "20 pairwise-distinct arrays over (Array (_ BitVec 3) (_ BitVec 1)) \
         must come back on the deterministic embedded-check budget; before \
         `#P2b-46` nothing bounded this and it ran 900 s with no answer"
    );
}

/// **COST PIN** (inverted).  The same for the `store`-term spelling of the
/// family: ten `store` terms over one base at index width 3 did not answer in
/// 40 s and had no bound at all; they now answer `unknown` after 250,002
/// embedded checks, 124.6 s in a release build here.
///
/// Release-calibrated and declining under `debug_assertions` for the reason
/// [`the_index_width_three_cardinality_ladder_terminates`] gives.
#[test]
#[ignore = "cost pin: minutes per script in release; declines under debug_assertions"]
fn the_store_term_cliff_terminates() {
    if cfg!(debug_assertions) {
        eprintln!(
            "declined: release-calibrated (124.6 s release, > 1,800 s with \
             debug_assertions); re-run with --cargo-profile release"
        );
        return;
    }
    let limit = Duration::from_secs(3_600);
    let answer = answers_within(distinct_store_terms(10, 3), limit);
    assert_eq!(
        answer.as_deref(),
        Some("unknown"),
        "ten `store` terms under an `n`-ary `distinct` at index width 3 must \
         come back on the deterministic embedded-check budget"
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
