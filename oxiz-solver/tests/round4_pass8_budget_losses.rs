//! Round 4, re-fix pass 8: the verdicts the deterministic budgets lose, named
//! and pinned (`#P2b-38` (b), `#P2b-46` (f)).
//!
//! # What this file is
//!
//! Decision (24a)'s criterion is "no script the BASE decides may become
//! `unknown`", and the round does not meet it: five scripts are known losses.
//! `c4b04b7` answers `sat` to every one of them in under a millisecond; this
//! tree answers `unknown` because the per-check cost of the embedded
//! bit-blasted solver exhausts a deterministic budget first (`#P2b-46` (f):
//! the cost of one embedded check is `O(num_vars)` over a variable table that
//! only grows with the search).
//!
//! The five are named in TODO.md `#P2b-38` (b) with their base and tree
//! verdicts.  This file exists so that the loss cannot silently become
//! something worse, and so that it stays *attributable to the budget*:
//!
//! 1. every one of them answers `unknown` — never a verdict.  All five are
//!    satisfiable (the base decides them `sat`), so an `unsat` here would be a
//!    soundness defect and a `sat` would mean the loss had been repaired and
//!    this file is out of date.  Either way the test reddens.
//! 2. widening the budget decides one of them.  That is what makes "the budget
//!    ran out" a measurement rather than an excuse: if a future change made
//!    these `unknown` for a reason that is *not* the budget, the widened case
//!    would stop deciding and this test would say so.
//!
//! Every budget here is deterministic (`:max-conflicts`,
//! `:max-bv-embedded-checks`) and no script carries a wall clock, so what the
//! file measures is machine-independent.
//!
//! The shapes are the round's own: `n` pairwise-`distinct` arrays over
//! `(Array (_ BitVec w) (_ BitVec 1))` (the `rc4/det` corpus) and a chain of
//! ten pairwise-`distinct` `store`s over one base array (`rf6/cal/st10_w3`).

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

/// `n` pairwise-`distinct` arrays over `(Array (_ BitVec width) (_ BitVec 1))`.
///
/// This is `rc4/det/det_w{width}_n{n}_mc200.smt2`, written out rather than
/// read from disk so the gate carries its own corpus.
fn distinct_arrays(width: u32, n: u32, conflicts: u32, embedded_checks: u32) -> String {
    let mut script = format!(
        "(set-logic QF_AUFBV)\n\
         (set-option :max-conflicts {conflicts})\n\
         (set-option :max-bv-embedded-checks {embedded_checks})\n"
    );
    for k in 0..n {
        script.push_str(&format!(
            "(declare-const a{k} (Array (_ BitVec {width}) (_ BitVec 1)))\n"
        ));
    }
    script.push_str("(assert (distinct");
    for k in 0..n {
        script.push_str(&format!(" a{k}"));
    }
    script.push_str("))\n(check-sat)\n");
    script
}

/// Ten pairwise-`distinct` single-`store` updates of one base array
/// (`rf6/cal/st10_w3.smt2`).
fn distinct_store_chain(width: u32, n: u32, conflicts: u32, embedded_checks: u32) -> String {
    let mut script = format!(
        "(set-logic QF_AUFBV)\n\
         (set-option :max-conflicts {conflicts})\n\
         (set-option :max-bv-embedded-checks {embedded_checks})\n\
         (declare-const base (Array (_ BitVec {width}) (_ BitVec 1)))\n"
    );
    for k in 0..n {
        script.push_str(&format!(
            "(declare-const i{k} (_ BitVec {width}))\n(declare-const v{k} (_ BitVec 1))\n"
        ));
    }
    script.push_str("(assert (distinct");
    for k in 0..n {
        script.push_str(&format!(" (store base i{k} v{k})"));
    }
    script.push_str("))\n(check-sat)\n");
    script
}

/// The five named losses, each under the deterministic budget that loses it.
fn named_losses() -> Vec<(&'static str, String)> {
    vec![
        ("det_w3_n12_mc200", distinct_arrays(3, 12, 200, 2000)),
        ("det_w3_n20_mc200", distinct_arrays(3, 20, 200, 2000)),
        ("det_w4_n11_mc200", distinct_arrays(4, 11, 200, 2000)),
        ("det_w4_n15_mc200", distinct_arrays(4, 15, 200, 2000)),
        ("st10_w3", distinct_store_chain(3, 10, 200, 2000)),
    ]
}

/// **GUARD.**  Every named loss is an honest `unknown`, never a verdict.
///
/// All five are satisfiable — `c4b04b7` answers `sat` to each in under a
/// millisecond — so:
///
/// * `unsat` here is a soundness defect and this test is the alarm;
/// * `sat` here means the loss has been repaired, and TODO.md `#P2b-38` (b)
///   must stop calling it a loss.
#[test]
fn every_named_budget_loss_answers_unknown_and_never_a_verdict() {
    for (name, script) in named_losses() {
        let script = format!("{script}(get-info :all-statistics)\n");
        let lines = run(&script);
        let answer = verdict(&lines);
        assert_eq!(
            answer, "unknown",
            "`{name}` is a NAMED accepted loss (TODO.md #P2b-38 (b)): the base \
             answers `sat` and this tree must answer `unknown`.  It answered \
             `{answer}`, which is either a soundness defect (`unsat` on a \
             satisfiable script) or a repair that TODO.md no longer \
             describes.\n{script}"
        );
        // And the budget it ran out of is *readable*, which it was not before
        // `:bv-embedded-conflicts` was published: `det_w4_n11` and
        // `det_w4_n15` answer `unknown` at `:conflicts 0`, so the outer
        // Boolean counter says nothing about why.  Measured on this tree
        // (debug): 200, 200, 200, 194, 200 embedded conflicts against the
        // `:max-conflicts 200` these scripts carry.  The floor is 190 rather
        // than 200 because `BvSolver::first_solve_allowance` keeps
        // `1 / REVERIFY_RESERVE_DIVISOR` of the remaining allowance back for
        // the `Unsat` re-verification, so the last probe of a run can stop a
        // few conflicts short of the ceiling without the budget being any
        // less spent — `det_w4_n15`'s 194 is that reserve, not slack.
        let spent = embedded_conflicts(&lines).unwrap_or_else(|| {
            panic!("`{name}`: (get-info :all-statistics) must publish :bv-embedded-conflicts")
        });
        assert!(
            spent >= 190,
            "`{name}` must be stopped by the budget it names: \
             `:max-conflicts 200` installs a 200-conflict allowance in the \
             embedded bit-blasted solver and this script spent {spent} of it. \
             A small number here means the `unknown` is NOT the budget and \
             TODO.md #P2b-38 (b)'s attribution is wrong."
        );
    }
}

/// `:bv-embedded-conflicts` from a `(get-info :all-statistics)` line.
fn embedded_conflicts(lines: &[String]) -> Option<u64> {
    lines
        .iter()
        .find(|line| line.contains(":bv-embedded-conflicts"))?
        .split(":bv-embedded-conflicts ")
        .nth(1)?
        .trim_end_matches(')')
        .trim()
        .parse()
        .ok()
}

/// **GUARD.**  One of the five *named* losses, widened, is decided — which is
/// the pin decision (31) asked for by name ("a widening of the budget decides
/// at least one of **them**").
///
/// `det_w4_n11` is the affordable one.  Measured on this tree, debug profile,
/// one script per process:
///
/// | script | `:max-conflicts` | verdict | `:conflicts` | `:bv-embedded-conflicts` | s |
/// |---|---|---|---|---|---|
/// | `det_w4_n11` | 200 | `unknown` | 0 | 200 | 5.1 |
/// | `det_w4_n11` | 20000 | **`sat`** | 0 | **2,561** | **147.1** |
/// | `det_w4_n15` | 20000 | (killed at 774 s) | — | — | — |
/// | `det_w3_n12` | 2000 | `sat` | — | — | 51.6 release |
///
/// So the loss is the budget's, on the script TODO.md names, and the counter
/// that moves is the embedded one — `:conflicts` is `0` on both sides, which
/// is why publishing `:bv-embedded-conflicts` was part of this fix.
///
/// It needs a kill ceiling of its own in `.config/nextest.toml`, exactly like
/// `array_cardinality_above_the_enumeration_limit_is_decided` (113.9 s in the
/// same profile).  The nine-array proxy below stays: it is the same mechanism
/// at 10 ms and 33 ms, so a run that cannot afford this one still checks the
/// attribution.
#[test]
fn widening_the_budget_buys_back_a_named_loss() {
    let tight = format!(
        "{}(get-info :all-statistics)\n",
        distinct_arrays(4, 11, 200, 2000)
    );
    let tight_lines = run(&tight);
    assert_eq!(
        verdict(&tight_lines),
        "unknown",
        "`det_w4_n11` at its named budget is the accepted loss.\n{tight}"
    );
    let wide = format!(
        "{}(get-info :all-statistics)\n",
        distinct_arrays(4, 11, 20_000, 2000)
    );
    let wide_lines = run(&wide);
    assert_eq!(
        verdict(&wide_lines),
        "sat",
        "the identical script at `:max-conflicts 20000` must be decided: that \
         is what makes `det_w4_n11`'s `unknown` attributable to the budget \
         rather than to a defect in the search.\n{wide}"
    );
    let tight_spent = embedded_conflicts(&tight_lines).unwrap_or_default();
    let wide_spent = embedded_conflicts(&wide_lines).unwrap_or_default();
    assert!(
        wide_spent > tight_spent,
        "and the currency has to be the embedded one: {tight_spent} \
         conflicts at the tight budget against {wide_spent} at the wide one. \
         `:conflicts` is 0 on both sides, so a test that watched it would be \
         watching nothing."
    );
}

/// **GUARD.**  The loss is the budget's: the same script, one budget wider, is
/// decided.
///
/// This is the whole content of "accepted loss under `#P2b-46` (f)": the
/// search is *correct* and merely too expensive per check, so more budget buys
/// the verdict back.  If the `unknown` above ever stops being about the
/// budget, this test stops deciding and says so.
///
/// # Why a ninth array *as well as* one of the five named above
///
/// [`widening_the_budget_buys_back_a_named_loss`] is the pin decision (31)
/// asked for and it runs `det_w4_n11` itself, at 147 s in the debug profile a
/// gate runs in.  This one is the cheap sibling: nine arrays over the same
/// sort is the same shape, the same mechanism and the same two budgets
/// (`unknown` at 200, `sat` at 2000) at 10 ms and 33 ms release, so the
/// attribution is still checked on a run that cannot afford two and a half
/// minutes.  The other four named losses stay proxied rather than widened:
/// `det_w4_n15` at `:max-conflicts 20000` was killed at **774 s** in this
/// profile, `det_w3_n12` needs 51.6 s *release*, and `st10_w3` does not finish
/// inside the calibrated ceiling at all.
#[test]
fn widening_the_budget_buys_back_the_verdict_on_the_same_shape() {
    let tight = distinct_arrays(3, 9, 200, 250_000);
    let tight_answer = verdict(&run(&tight));
    assert_eq!(
        tight_answer, "unknown",
        "nine `distinct` arrays over `(Array (_ BitVec 3) (_ BitVec 1))` must \
         exhaust `:max-conflicts 200` on this tree — that is the mechanism the \
         five named losses above share.\n{tight}"
    );
    let wide = distinct_arrays(3, 9, 2_000, 250_000);
    let wide_answer = verdict(&run(&wide));
    assert_eq!(
        wide_answer, "sat",
        "and the identical script at `:max-conflicts 2000` must be decided, \
         which is what makes the `unknown` attributable to the budget and to \
         nothing else.\n{wide}"
    );
}

// ---------------------------------------------------------------------------
// THE ONE PER-SCRIPT REGRESSION IN `bench/` THAT IS NOT LOAD, ATTRIBUTED
// ---------------------------------------------------------------------------

/// **COST PIN.**  `bench/z3_parity/benchmarks/AUFLIA/array_update.smt2` costs
/// what the *quantified* array-refinement path costs, and not a millisecond
/// more.
///
/// The recheck measured this script at 19.8–25.3 ms on the tree over five runs
/// against 1.7–8.3 ms on `c4b04b7` — an order of magnitude, stable, verdict
/// `sat` unchanged, and the only member of the 217-script sweep whose
/// slowdown is not load.
///
/// # The attribution, measured rather than guessed
///
/// Three probes, same script, release:
///
/// | build | conflicts | propagations | array-refinement-rounds | array-lemma-instances | ms |
/// |---|---|---|---|---|---|
/// | `c4b04b7` (base) | 6 | 141 | — | — | 1.4–8.9 |
/// | `00add07` (pass-6 checkpoint) | 17 | 1,308 | 7 | 24 | 20.3–28.0 |
/// | this tree (pass 8) | 17 | 1,308 | 7 | 24 | 17.3–29.5 |
///
/// The pass-6 checkpoint and this tree are **counter-identical**, so re-fix
/// pass 8 contributes nothing to it: the cost arrived with `#P2b-48`, which
/// moved the lazy array refinement out of `check_core`'s `!has_quantifiers`
/// branch so that a script with one `forall` runs array rules at all.  This
/// script has a `forall` *and* a `store`, so it now pays seven refinement
/// rounds — each a full re-solve — and 24 lemma instances that it previously
/// skipped along with the soundness they buy.  It is the price of the fix on
/// the cheapest `bench/` member that shows it, not needless work on a
/// quantifier-free script.
///
/// The pin is on the deterministic counters rather than on the clock: a change
/// that made this script cost *more rounds* moves the test, and a change that
/// made it cost more wall-clock for the same rounds is a machine, not a
/// regression.
#[test]
fn the_auflia_array_update_cost_is_the_quantified_refinement_round() {
    let script = "(set-logic AUFLIA)\n\
         (declare-const a (Array Int Int))\n\
         (declare-const b (Array Int Int))\n\
         (declare-const k Int)\n\
         (declare-const v Int)\n\
         (assert (= b (store a k v)))\n\
         (assert (= k 3))\n\
         (assert (= v 99))\n\
         (assert (forall ((i Int)) (=> (not (= i k)) (= (select b i) (select a i)))))\n\
         (assert (= (select b k) v))\n\
         (assert (= (select a 0) 10))\n\
         (assert (= (select a 1) 20))\n\
         (assert (= (select a 3) 30))\n\
         (assert (= (select b 3) 99))\n\
         (assert (= (select b 0) 10))\n\
         (assert (= (select b 1) 20))\n\
         (check-sat)\n\
         (get-info :all-statistics)\n";
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        "sat",
        "the benchmark's own expected answer is `sat`\n{}",
        lines.join("\n")
    );
    let stats = lines
        .iter()
        .find(|line| line.contains(":array-refinement-rounds"))
        .cloned()
        .unwrap_or_default();
    assert!(
        stats.contains(":array-refinement-rounds 7"),
        "the 20 ms this script costs is seven quantified array-refinement \
         rounds; a different number means the cost moved and its attribution \
         in this test is out of date.  Got: {stats}"
    );
    assert!(
        stats.contains(":array-lemma-instances 24"),
        "and 24 lemma instances.  Got: {stats}"
    );
    assert!(
        stats.contains(":bv-embedded-checks 0"),
        "the embedded bit-blasted solver is not involved here at all, so \
         `#P2b-46` (f) is not this script's cost.  Got: {stats}"
    );
}
