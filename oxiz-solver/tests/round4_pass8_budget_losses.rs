//! Round 4, re-fix pass 8: the verdicts the deterministic budgets lose, named
//! and pinned (`#P2b-38` (b), `#P2b-46` (f)).
//!
//! # What this file is
//!
//! Decision (24a)'s criterion is "no script the BASE decides may become
//! `unknown`", and the round does not meet it: five scripts are known losses.
//! `c4b04b7` answers `sat` to every one of them in under a millisecond; this
//! tree answers `unknown` because a deterministic budget runs out first.
//! Until re-fix pass 12 the reason was the *price* of an embedded check
//! (`#P2b-46` (f): `O(num_vars)` over a variable table that only grew with the
//! search); that price is now flat (root-scoped bit-blasting, decision (45)),
//! and what the four `:max-conflicts 200` scripts still exhaust is that
//! allowance itself — the embedded solver spends all 200 conflicts — while
//! `st10_w3` exhausts the 250,000-check ceiling in 1.1–2.9 s release where
//! it took 447 s.  Re-measured in `TODO.md` `#P2b-38` (b).
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
///
/// The `:max-bv-embedded-checks` cap is a safety net only, and it is set well
/// above what the named `:max-conflicts 200` lets any of the five spend (at
/// most 8,067 checks, measured on re-fix pass 12's tree).  It was 2,000 while
/// an embedded check cost `O(num_vars)`; once the check became cheap the
/// search reached 2,000 checks before 200 conflicts, the net became the
/// budget that stopped three of the five, and this file's attribution
/// assertion rightly failed.
fn named_losses() -> Vec<(&'static str, String)> {
    vec![
        ("det_w3_n12_mc200", distinct_arrays(3, 12, 200, 50_000)),
        ("det_w3_n20_mc200", distinct_arrays(3, 20, 200, 50_000)),
        ("det_w4_n11_mc200", distinct_arrays(4, 11, 200, 50_000)),
        ("det_w4_n15_mc200", distinct_arrays(4, 15, 200, 50_000)),
        ("st10_w3", distinct_store_chain(3, 10, 200, 50_000)),
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
        // Boolean counter says nothing about why.  Measured on re-fix pass
        // 12's tree: 200 embedded conflicts for all five against the
        // `:max-conflicts 200` these scripts carry.  The floor used to be 190
        // because the embedded solver kept a quarter of its remaining
        // allowance back for an `Unsat` re-verification; a check is one solve
        // under assumptions now (decision (45)), there is no reserve, and the
        // allowance is spent to the last conflict.
        let spent = embedded_conflicts(&lines).unwrap_or_else(|| {
            panic!("`{name}`: (get-info :all-statistics) must publish :bv-embedded-conflicts")
        });
        assert!(
            spent >= 200,
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
///
/// **Moved by re-fix pass 12, and attributed:** 7 → **11** rounds and 24 →
/// **37** lemma instances, verdict `sat` unchanged.  The four extra rounds are
/// `#P2b-60`'s repair: the SAT certification no longer concludes `sat` for
/// this script's `∀i. i ≠ k ⇒ b[i] = a[i]` until the instances at every named
/// index value and at a representative of each gap the guard `i ≠ k` orders
/// have been emitted too, and those instances' reads are what the extra
/// refinement rounds axiomatise (measured in an isolated copy with the
/// completion of the relevant set switched off: 7 / 24 again).  It is the price of refusing the
/// wrong `sat` `corpus/named/wsat6` (`a2 = store(K0, 0, 1)` beside
/// `∀i. a2[i] = 1`) answered on every earlier build, on the one `bench/`
/// member that shows it.  The model completion re-fix pass 12 also runs on
/// this `sat` spends only sub-solver work, which these main-solver counters
/// do not see (measured: 24 → 45 ms release, the pinned search declined over
/// an `Int` index sort — see `solver::array_completion_certify::pinned`).
///
/// **Moved again by re-fix pass 12's second half, and attributed:** 11 → **16**
/// rounds and 37 → **53** instances, verdict `sat` unchanged.  The move is
/// decision (43)'s canonical order: the pair a one-pair phase reaches first
/// and the order a round's lemmas are encoded in follow the terms' structural
/// keys, not the assertion order, and on this script the canonical order
/// happens to take five more rounds (measured in an isolated copy with the
/// canonical sorting removed and everything else of this tree kept: 11 / 37
/// again).  Over the two regenerated 120-pair corpora the same ordering is
/// neutral-to-favourable — 344,679 embedded checks and 3,053 rounds against
/// 345,764 and 3,022 without it, the same 239 of 240 decided — which is why it
/// stays; this script is its cost, and `:bv-embedded-checks 0` still says the
/// bit-blaster plays no part.
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
        stats.contains(":array-refinement-rounds 16"),
        "this script's cost is sixteen quantified array-refinement rounds \
         (seven before re-fix pass 12, eleven with `#P2b-60`'s completed \
         relevant set, sixteen in decision (43)'s canonical order); a \
         different number means the cost moved and its attribution in this \
         test is out of date.  Got: {stats}"
    );
    assert!(
        stats.contains(":array-lemma-instances 53"),
        "and 53 lemma instances (24 before re-fix pass 12, 37 after its first \
         half).  Got: {stats}"
    );
    assert!(
        stats.contains(":bv-embedded-checks 0"),
        "the embedded bit-blasted solver is not involved here at all, so \
         `#P2b-46` (f) is not this script's cost.  Got: {stats}"
    );
}
