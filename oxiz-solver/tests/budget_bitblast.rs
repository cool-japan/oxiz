//! U-Z12 — `:timeout` / `:max-conflicts` / `:max-decisions` must bound the
//! bit-blasted solves too.
//!
//! # What OxiZ 0.3.3 / 0.3.4 did before this fix
//!
//! `BvSolver::check()` runs a full `oxiz_sat::Solver::solve()` on its own
//! embedded solver, once per asserted bit-vector atom, from inside the
//! enclosing CDCL(T) search's `on_assignment` callback. Every budget poll lived
//! *outside* that call — `check_core` checked the deadline between MBQI rounds,
//! `TheoryManager` checked it at the entry of a theory callback, and
//! `:max-conflicts` was compared against `Statistics::conflicts`, a counter
//! only ever incremented on a *theory* conflict. Nothing in the workspace ever
//! called `oxiz_sat::Solver::set_max_conflicts`, and `oxiz_sat::Solver` had no
//! deadline at all, so `SolverConfig::max_decisions` was wired to nothing
//! whatsoever.
//!
//! Measured on the 0.3.4 tree at commit `6bdf958`, on the 64x64 multiplier
//! verification condition [`MUL_VC`] below (external `timeout 60`, none of the
//! runs hit it):
//!
//! | option | verdict | wall time |
//! |---|---|---|
//! | none | `unsat` | 15 760 ms |
//! | `(set-option :timeout 100)` | `unsat` | 17 052 ms — 170x the budget |
//! | `(set-option :timeout 1000)` | `unsat` | 6 573 ms |
//! | `(set-option :max-conflicts 50)` | `unsat` | 13 079 ms |
//! | `(set-option :max-decisions 50)` | `unsat` | 5 766 ms |
//!
//! The spread is run-to-run noise, not a budget effect: a honoured
//! `:max-conflicts 50` would have to answer in milliseconds. `unsat` is the
//! *correct* verdict here (`a, b < 2^32`, so the high 64 bits of the 128-bit
//! product are zero), which is why the tests below pin **both** directions:
//! under a budget the answer must become `unknown`, and without one it must
//! still be `unsat`. A regression that answers `unknown` unconditionally would
//! pass only the first half.
//!
//! # The three budgets
//!
//! After this fix `(set-option :max-conflicts N)` is three independent budgets
//! of `N`, one per kind of work, all re-armed once per `(check-sat)`:
//! outer Boolean conflicts, the embedded bit-blasting total across every probe
//! and repair round, and theory conflicts. `(set-option :max-decisions N)`
//! bounds outer Boolean decisions. `(set-option :timeout N)` is one wall-clock
//! deadline shared by all of them.
//!
//! # Wall clock, and why it is the weakest assertion here
//!
//! Every timing tabulated above was taken in an optimized build. `[profile.dev]`
//! is `opt-level = 1`, and this suite runs under `cargo nextest` beside other
//! CPU-bound jobs; the same bounded check then costs several times more. The
//! `:max-conflicts 50` case, whose release bound is 2 s, was measured at 1.45 s
//! in the dev profile at load average ~11 and at 8.67 s at load 24-60 — the
//! release-calibrated bound is simply not a fact about a debug build on a busy
//! machine.
//!
//! So every wall-clock assertion below goes through `bounded_check_ceiling`,
//! which keeps the calibrated release figure and uses a separate, measured dev
//! figure. Nothing is lost by that, because wall time is the *weakest* witness
//! in each of these tests. The load-independent ones are:
//!
//! * the verdict — `unknown` rather than the `unsat` this goal is refutable to;
//! * `(get-info :reason-unknown)` — a real reason, not `"not applicable"`;
//! * `Context::bv_conflicts_spent()` — the budget's own accounting, in the same
//!   unit `(set-option :max-conflicts N)` is expressed in, so it is the *direct*
//!   bound wherever that option is the budget under test;
//! * `Context::stats().propagations` / `.decisions` — the outer engine's
//!   counters.

use oxiz_solver::{Context, SolverResult};
use std::time::{Duration, Instant};

/// The 64x64 multiplier VC: `a, b < 2^32` implies the high half of their
/// 128-bit product is zero, asserted negated — so the goal is `unsat`, and the
/// refutation costs the bit-blaster a full 128-bit multiplier circuit.
///
/// Copied verbatim from the Phase 2 probe's `h20_u64mul` case.
const MUL_VC: &str = concat!(
    "(declare-const a (_ BitVec 64))\n",
    "(declare-const b (_ BitVec 64))\n",
    "(assert (bvult a #x0000000100000000))\n",
    "(assert (bvult b #x0000000100000000))\n",
    "(assert (not (= ((_ extract 127 64) (bvmul ((_ zero_extend 64) a) ",
    "((_ zero_extend 64) b))) #x0000000000000000)))\n",
    "(check-sat)\n",
    "(get-info :reason-unknown)",
);

/// `MUL_VC` with `prelude` (the `(set-option ...)` lines) between the logic and
/// the declarations.
fn mul_vc(prelude: &str) -> String {
    format!("(set-logic QF_BV)\n{prelude}{MUL_VC}")
}

/// Run a script and return every output line joined by newlines.
fn run_script_output(script: &str) -> String {
    let mut ctx = Context::new();
    ctx.execute_script(script).unwrap_or_default().join("\n")
}

/// Everything one script run offers the budget assertions below.
struct Measured {
    /// Every output line of the run, joined by newlines.
    output: String,
    /// The outer SAT engine's propagation count, cumulative over the context.
    ///
    /// This is the witness that a `(check-sat)` really reached `check_core`:
    /// `Solver::check` consults a verdict cache first
    /// (`solver/verdict_cache.rs`), so a second, *identical* check in the same
    /// script replays the first verdict without searching at all — measured,
    /// the counter stays at exactly 3 for one check and for two identical ones,
    /// and rises for two that differ. On this bit-blasted goal it is the only
    /// outer counter that moves (conflicts and decisions stay at 0: the work is
    /// all inside the embedded solver), which is precisely the finding this
    /// file is about.
    propagations: u64,
    /// Embedded bit-blasting conflicts charged to the **last** `(check-sat)` of
    /// the run.
    ///
    /// `check_core` calls `BvSolver::set_budget` once per check, which re-arms
    /// the accumulator, so this is that check's own spend rather than a total
    /// over the script — see `Solver::bv_conflicts_spent`. It is the budget's
    /// own bookkeeping, in the unit `(set-option :max-conflicts N)` is written
    /// in, and it is therefore the same number on a loaded machine as on an
    /// idle one.
    bv_conflicts: u64,
    /// Wall time of the run.
    elapsed: Duration,
}

/// Run a script on a fresh `Context` and measure what it did.
///
/// One script per context, deliberately. A test here that wants a baseline plus
/// a longer run does it as two whole scripts on two contexts rather than as two
/// `execute_script` calls on one, because `Context::execute_script` hands each
/// string to `parse_script`, which parses it in `script_mode` behind a symbol
/// table only *that* string's declarations fill: a follow-up fragment naming a
/// constant an earlier call declared is rejected as undeclared, the `Result` is
/// swallowed by `unwrap_or_default()` here, and the test sees an empty output
/// rather than an error. Two fresh contexts also keep the counter comparisons
/// sound, since every counter starts at zero on both.
fn run_script_measured(script: &str) -> Measured {
    let mut ctx = Context::new();
    let start = Instant::now();
    let output = ctx.execute_script(script).unwrap_or_default().join("\n");
    let elapsed = start.elapsed();
    Measured {
        output,
        propagations: ctx.stats().propagations,
        bv_conflicts: ctx.bv_conflicts_spent(),
        elapsed,
    }
}

/// The wall-clock ceiling a bounded check must respect, by build profile.
///
/// `release` is the figure this file's header table is calibrated against and
/// stays the assertion in an optimized build. `dev` is a separate measured
/// figure for `[profile.dev]` (`opt-level = 1`) under `cargo nextest` on a
/// loaded machine; each call site quotes what it measured.
///
/// The dev figure is deliberately generous. This bound is not what pins the
/// budget — the verdict, the `:reason-unknown` text and
/// `Context::bv_conflicts_spent()` do that, and none of them moves with machine
/// load (see this file's header). All it has to catch is a regression that lets
/// a bounded check run the multiplier refutation to completion, which is 15.8 s
/// in release and 264.3 s in the dev profile — measured by running the
/// `#[ignore]`d `the_same_vc_without_a_budget_is_still_unsat` below at load
/// average 22-37. Every dev ceiling passed in is an order of magnitude under
/// that.
fn bounded_check_ceiling(release: Duration, dev: Duration) -> Duration {
    if cfg!(debug_assertions) { dev } else { release }
}

/// The verdict of the **last** `(check-sat)` in `script`.
fn last_verdict(script: &str) -> SolverResult {
    verdicts(&run_script_output(script))
        .last()
        .copied()
        .unwrap_or(SolverResult::Unknown)
}

/// Every `sat` / `unsat` / `unknown` line of an output, in order.
fn verdicts(output: &str) -> Vec<SolverResult> {
    output
        .lines()
        .filter_map(|line| match line.trim() {
            "sat" => Some(SolverResult::Sat),
            "unsat" => Some(SolverResult::Unsat),
            "unknown" => Some(SolverResult::Unknown),
            _ => None,
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────
// 1. The acceptance case, both directions
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn a_timeout_bounds_the_bit_blasted_multiplier_vc() {
    let run = run_script_measured(&mul_vc("(set-option :timeout 100)\n"));
    let output = &run.output;

    assert_eq!(
        verdicts(output),
        vec![SolverResult::Unknown],
        "`:timeout 100` must cut the bit-blasted solve short; got:\n{output}"
    );
    // The measured pre-fix time is 17 s; 2 s leaves ~8x headroom over the
    // budget's own granularity (the deadline is polled once per 256 CDCL loop
    // iterations, and a single `propagate()` is not itself interruptible) while
    // still failing loudly if the budget stops reaching the embedded solver.
    //
    // Dev: measured 0.47 s at load average ~11 and 2.74 s at 24-60, against
    // 264.3 s for the same goal with the budget removed — so 20 s still
    // separates "bounded" from "ran to completion" by a wide margin. The three
    // `:reason-unknown` assertions below are the load-independent half of this
    // test and are unchanged.
    assert!(
        run.elapsed < bounded_check_ceiling(Duration::from_secs(2), Duration::from_secs(20)),
        "`:timeout 100` let the check run for {:?}",
        run.elapsed
    );
    // `(get-info :reason-unknown)` must now say something: before the fix the
    // verdict was `unsat`, so it answered `"not applicable"`.
    assert!(
        output.contains(":reason-unknown"),
        "the script asks for :reason-unknown; got:\n{output}"
    );
    assert!(
        !output.contains("not applicable"),
        "an `unknown` verdict must report a real reason, not \"not applicable\":\n{output}"
    );
    assert!(
        output.contains("incomplete"),
        "expected `(:reason-unknown incomplete)`; got:\n{output}"
    );
}

/// The other direction: with no budget the same goal must still be refuted.
///
/// `#[ignore]` because it is the one slow test in this file — it runs the
/// multiplier refutation to completion. Measured at 12.2 s in release on the
/// development machine, and at 264.3 s in the dev profile at load average
/// 22-37 (see this file's header table for the pre-fix baseline). That dev
/// figure is what every `bounded_check_ceiling` dev bound is calibrated
/// against: it is the cost of the budget not firing. Run it with
/// `cargo nextest run --release -p oxiz-solver --run-ignored all -E 'test(the_same_vc_without_a_budget_is_still_unsat)'`.
#[test]
#[ignore = "runs the full 64x64 multiplier refutation (~12 s in release)"]
fn the_same_vc_without_a_budget_is_still_unsat() {
    assert_eq!(
        last_verdict(&mul_vc("")),
        SolverResult::Unsat,
        "without a budget the multiplier VC must still be proved"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 2. The conflict budget reaches the bit-blasted search
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn a_conflict_budget_bounds_the_bit_blasted_multiplier_vc() {
    // The allowance under test, in the unit the budget is written in.
    const BUDGET: u64 = 50;

    let run = run_script_measured(&mul_vc(&format!("(set-option :max-conflicts {BUDGET})\n")));
    let output = &run.output;

    assert_eq!(
        verdicts(output),
        vec![SolverResult::Unknown],
        "`:max-conflicts {BUDGET}` must cut the bit-blasted solve short; got:\n{output}"
    );
    // The bound that matters, and the reason a wall clock is not needed to
    // state it: `:max-conflicts` is a budget in conflicts, and
    // `Context::bv_conflicts_spent()` is the solver's own count of the ones it
    // charged to the bit-blasting share of it. Measured: exactly 38, on a
    // loaded machine as on an idle one — `first_solve_allowance` keeps a
    // quarter of the remainder back for the `Unsat` re-verification, so a probe
    // that exhausts its share stops at `BUDGET - BUDGET / 4`. Before the fix
    // nothing charged the embedded solver at all and this check ran to
    // `unsat` in 13 s, which costs thousands of conflicts.
    assert!(
        run.bv_conflicts <= BUDGET,
        "`:max-conflicts {BUDGET}` must bound the bit-blasted search: it spent {} embedded \
         conflicts",
        run.bv_conflicts
    );
    assert!(
        run.bv_conflicts > 0,
        "the check must reach the embedded bit-blasting solver at all, or the bound above says \
         nothing about it"
    );
    // Dev: measured 1.45 s at load average ~11 and 8.67 s at 24-60, against
    // 264.3 s with the budget removed. See `bounded_check_ceiling`.
    assert!(
        run.elapsed < bounded_check_ceiling(Duration::from_secs(2), Duration::from_secs(30)),
        "`:max-conflicts {BUDGET}` let the check run for {:?} (was 13 s before the fix)",
        run.elapsed
    );
    assert!(
        !output.contains("not applicable"),
        "an `unknown` verdict must report a real reason:\n{output}"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 3. A budget must not change a verdict the solver reaches quickly
// ─────────────────────────────────────────────────────────────────────────

/// Easy QF_BV goals, each with its verdict, exercised with and without a
/// generous budget. This is what catches an off-by-one in the
/// remaining-conflicts arithmetic that re-arms the allowance as `0`.
const EASY_GOALS: &[(&str, &str, SolverResult)] = &[
    (
        "bvadd_sat",
        "(declare-const x (_ BitVec 8))\n(assert (= (bvadd x #x01) #x02))\n(check-sat)",
        SolverResult::Sat,
    ),
    (
        "contradictory_constants_unsat",
        "(declare-const x (_ BitVec 8))\n(assert (= x #x01))\n(assert (= x #x02))\n(check-sat)",
        SolverResult::Unsat,
    ),
    (
        "bvult_chain_sat",
        "(declare-const x (_ BitVec 8))\n(declare-const y (_ BitVec 8))\n(declare-const z (_ \
         BitVec 8))\n(assert (bvult x y))\n(assert (bvult y z))\n(check-sat)",
        SolverResult::Sat,
    ),
    (
        "bvult_cycle_unsat",
        "(declare-const x (_ BitVec 8))\n(declare-const y (_ BitVec 8))\n(assert (bvult x \
         y))\n(assert (bvult y x))\n(check-sat)",
        SolverResult::Unsat,
    ),
    (
        "extract_concat_unsat",
        "(declare-const x (_ BitVec 8))\n(assert (not (= (concat ((_ extract 7 4) x) ((_ extract \
         3 0) x)) x)))\n(check-sat)",
        SolverResult::Unsat,
    ),
];

#[test]
fn a_generous_budget_does_not_change_a_fast_verdict() {
    let generous = "(set-option :timeout 60000)\n(set-option :max-conflicts 1000000)\n(set-option \
                    :max-decisions 1000000)\n";
    for (name, body, expected) in EASY_GOALS {
        let unbudgeted = last_verdict(&format!("(set-logic QF_BV)\n{body}"));
        assert_eq!(unbudgeted, *expected, "{name}: baseline verdict changed");
        let budgeted = last_verdict(&format!("(set-logic QF_BV)\n{generous}{body}"));
        assert_eq!(
            budgeted, *expected,
            "{name}: a generous budget must not change the verdict"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// 4. The budget survives `rebase_theory_state`
// ─────────────────────────────────────────────────────────────────────────
//
// `Solver::rebase_theory_state` calls `BvSolver::reset()`, which calls
// `sat.reset()`, which zeroes the embedded solver's statistics — and it runs
// once per check *and* again on every repair round inside one. A budget
// expressed purely in those statistics would silently re-arm each time; the
// `conflicts_spent` accumulator on `BvSolver` is what keeps it a total.

#[test]
fn two_checks_under_one_timeout_are_both_bounded() {
    // The two checks must differ, or the verdict cache replays the first
    // verdict and the second one never reaches `check_core` — this test would
    // then say nothing about it. The extra assertion changes the goal
    // fingerprint; the propagation growth below is what proves the second check
    // really searched.
    let prelude = "(set-logic QF_BV)\n(set-option :timeout 100)\n";
    let one = run_script_measured(&mul_vc("(set-option :timeout 100)\n"));
    assert_eq!(
        verdicts(&one.output),
        vec![SolverResult::Unknown],
        "the single-check baseline must already be bounded; got:\n{}",
        one.output
    );

    let two_script = format!(
        "{prelude}{MUL_VC}\n(assert (bvult a #x00000000ffffffff))\n(check-sat)\n(get-info \
         :reason-unknown)"
    );
    let two = run_script_measured(&two_script);
    assert_eq!(
        verdicts(&two.output),
        vec![SolverResult::Unknown, SolverResult::Unknown],
        "both checks under one `:timeout` must be bounded; got:\n{}",
        two.output
    );
    assert!(
        two.propagations > one.propagations,
        "the second check must actually search ({} -> {} outer propagations); if it does not \
         grow, the verdict cache replayed the first verdict and this test proves nothing",
        one.propagations,
        two.propagations
    );
    // Dev: measured 1.34 s for the two-check script at load average ~11 and
    // 3.95 s for the whole test at 24-60, against 264.3 s for one *unbounded*
    // check (`the_same_vc_without_a_budget_is_still_unsat`). See
    // `bounded_check_ceiling`; the verdicts and the propagation growth above
    // are what pin the budget.
    assert!(
        two.elapsed < bounded_check_ceiling(Duration::from_secs(4), Duration::from_secs(20)),
        "two bounded checks took {:?}",
        two.elapsed
    );
}

#[test]
fn a_budget_survives_push_and_pop() {
    // `Solver::pop` runs `invalidate_results`, which drops the cached verdict,
    // so here the two checks may be identical and the second still runs; the
    // propagation growth pins that rather than assuming it.
    const BUDGET: u64 = 50;

    let prelude = format!("(set-logic QF_BV)\n(set-option :max-conflicts {BUDGET})\n");
    let one = run_script_measured(&format!("{prelude}(push 1)\n{MUL_VC}\n(pop 1)"));
    assert_eq!(
        verdicts(&one.output),
        vec![SolverResult::Unknown],
        "the conflict budget must bound the check inside a scope; got:\n{}",
        one.output
    );

    let two = run_script_measured(&format!(
        "{prelude}(push 1)\n{MUL_VC}\n(pop 1)\n(push 1)\n{MUL_VC}\n(pop 1)"
    ));
    assert_eq!(
        verdicts(&two.output),
        vec![SolverResult::Unknown, SolverResult::Unknown],
        "the conflict budget must survive push/pop; got:\n{}",
        two.output
    );
    assert!(
        two.propagations > one.propagations,
        "the second scope's check must actually search ({} -> {} outer propagations)",
        one.propagations,
        two.propagations
    );
    // The budget's own accounting for the check inside the *second* scope —
    // `bv_conflicts` reports the last `(check-sat)` of a script, and this is the
    // property the test is about: `check_core` re-arms the allowance for every
    // check, so popping the first scope must neither carry the first check's
    // spend over nor hand the second check an unbounded one. Measured: 38 in
    // both the one-scope and the two-scope run, which is `BUDGET - BUDGET / 4`
    // (see `a_conflict_budget_bounds_the_bit_blasted_multiplier_vc`), and the
    // same number whatever the machine is doing.
    assert!(
        two.bv_conflicts <= BUDGET,
        "the embedded allowance must still bound the check after a pop: it spent {} of {BUDGET}",
        two.bv_conflicts
    );
    assert!(
        two.bv_conflicts > 0,
        "the check in the second scope must reach the embedded bit-blasting solver at all, or \
         the bound above says nothing about it"
    );
    // Dev: measured 3.19 s for the two-scope script at load average ~11 and
    // 10.31 s at 24-60, against 264.3 s for one *unbounded* check. See
    // `bounded_check_ceiling`.
    assert!(
        two.elapsed < bounded_check_ceiling(Duration::from_secs(4), Duration::from_secs(30)),
        "two bounded checks around push/pop took {:?}",
        two.elapsed
    );
}

// ─────────────────────────────────────────────────────────────────────────
// 5. The embedded allowance is a total, not a grant per probe
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn the_embedded_conflict_allowance_is_a_total_not_a_per_probe_grant() {
    // Many bit-vector atoms over shared variables: the CDCL(T) loop asserts
    // them one at a time and calls `BvSolver::check()` after each, so this goal
    // needs many probes. Under `:max-conflicts N` their *sum* must stay within
    // `N`; before this fix each probe would have been granted `N` of its own
    // (and in fact none of them was bounded at all).
    //
    // The test is two-sided so the bound cannot pass vacuously: the same goal
    // under a larger budget must spend *more* than `BUDGET`, which is what
    // makes "spent <= BUDGET" evidence that the small budget bound it rather
    // than evidence that the goal was easy. `GENEROUS` is itself a budget
    // rather than "unbounded" only to keep this test's wall time down — the
    // goal runs for tens of seconds if nothing stops it, which is itself the
    // property being relied on.
    //
    // Sizing. This is the one test in this file whose cost is the goal rather
    // than `MUL_VC`, and both numbers below are spent in full by a goal that
    // cannot finish inside them, so the wall time is roughly linear in
    // `GENEROUS`. Measured in the dev profile (`opt-level = 1`) on the
    // development machine at load average ~11, for the two `spend` calls
    // together, over the six-variable goal `multi_probe_goal` builds:
    //
    // | atom width | BUDGET | GENEROUS | bounded | generous | wall |
    // |---|---|---|---|---|---|
    // | 32 | 200 | 5000 | 150 |    — | > 400 s (`cargo nextest` kills it at 180 s) |
    // | 32 | 100 | 1000 |  75 |  750 | 56.7 s |
    // | 16 | 100 | 1000 |  75 |  750 | 12.3 s |
    // | 16 | 100 |  400 |  75 |  300 |  4.5 s |
    //
    // The spends are exact, not timings: `first_solve_allowance` hands a probe
    // `r - r / 4` of the `r` it has left, so a goal that exhausts its allowance
    // spends exactly `BUDGET - BUDGET / 4`. Nothing about the property under
    // test moves with the size — `GENEROUS` only has to be spent *past*
    // `BUDGET` for the bound to be non-vacuous, and 400 witnesses that as
    // surely as 5000 did.
    const BUDGET: u64 = 100;
    const GENEROUS: u64 = 400;

    let goal = multi_probe_goal();
    let spend = |budget: u64| -> (u64, u64) {
        let mut ctx = Context::new();
        let script = format!("(set-logic QF_BV)\n(set-option :max-conflicts {budget})\n{goal}");
        let output = ctx.execute_script(&script).unwrap_or_default().join("\n");
        assert!(
            !verdicts(&output).is_empty(),
            "the script must produce a verdict; got:\n{output}"
        );
        (ctx.bv_conflicts_spent(), ctx.stats().conflicts)
    };

    let (bounded_bv, bounded_outer) = spend(BUDGET);
    let (generous_bv, _generous_outer) = spend(GENEROUS);

    assert!(
        generous_bv > BUDGET,
        "the goal must genuinely need more than {BUDGET} embedded conflicts for the bound below \
         to mean anything; under a {GENEROUS}-conflict budget it spent {generous_bv}"
    );
    assert!(
        bounded_bv <= BUDGET,
        "the embedded bit-blasting allowance is a total across every probe and repair round: \
         spent {bounded_bv} of {BUDGET} (the {GENEROUS}-conflict run spends {generous_bv})"
    );
    // The outer Boolean allowance is the same shape, measured on the cumulative
    // `oxiz_sat` counters relative to where this check started (zero, since the
    // context ran exactly one check).
    assert!(
        bounded_outer <= BUDGET,
        "the outer Boolean allowance is a total: spent {bounded_outer} of {BUDGET}"
    );
}

/// A QF_BV goal that forces many `BvSolver::check()` probes: fifteen
/// multiplier/adder comparisons over six shared 16-bit variables, so the
/// CDCL(T) loop asserts one bit-vector atom at a time and re-checks after each.
///
/// The atoms are 16 bits rather than 32. What this goal has to supply is the
/// *probe count* — six shared variables and fifteen pairwise atoms, both
/// unchanged — and a total cost above the allowance under test, which it still
/// has: under the 400-conflict allowance its caller uses it spends 300 and
/// still answers `unknown`, never reaching a verdict of its own. The width only
/// sets how long each bit-blasted probe takes, and at 32 bits the
/// two runs of `the_embedded_conflict_allowance_is_a_total_not_a_per_probe_grant`
/// took over 400 s in the dev profile, past the 180 s at which the repo's
/// `.config/nextest.toml` terminates a test.
///
/// The threshold constant scales with the width: `#x7fff` is half the widest
/// 16-bit value, as `#x7fffffff` was at 32 bits. Leaving it at `#x7fffffff`
/// would not fit a 16-bit literal at all, and pinning the threshold to the
/// widest value instead would make `(bvugt (bvmul v0 v1) ...)` unsatisfiable by
/// construction — the goal would then cost no conflicts and every assertion
/// over it would pass vacuously.
fn multi_probe_goal() -> String {
    let mut goal = String::new();
    for i in 0..6 {
        goal.push_str(&format!("(declare-const v{i} (_ BitVec 16))\n"));
    }
    for i in 0..6u32 {
        for j in (i + 1)..6 {
            goal.push_str(&format!(
                "(assert (bvult (bvmul v{i} v{j}) (bvadd v{i} v{j})))\n"
            ));
        }
    }
    goal.push_str("(assert (bvugt (bvmul v0 v1) #x7fff))\n(check-sat)\n");
    goal
}

// ─────────────────────────────────────────────────────────────────────────
// 6. `:max-decisions` bounds the outer Boolean search
// ─────────────────────────────────────────────────────────────────────────
//
// By design `:max-decisions` is scoped to the outer CDCL(T) search: the
// embedded bit-blasting solver is bounded by `:timeout` and `:max-conflicts`.
// The goal below therefore has rich *Boolean* structure over bit-vector atoms,
// which is what makes the outer solver take decisions at all.

#[test]
fn a_decision_budget_bounds_the_outer_boolean_search() {
    let mut goal = String::from("(declare-const x (_ BitVec 16))\n");
    for i in 0..12u32 {
        goal.push_str(&format!("(declare-const p{i} Bool)\n"));
    }
    for i in 0..12u32 {
        goal.push_str(&format!(
            "(assert (= p{i} (bvult x #x{:04x})))\n",
            (i + 1) * 0x0100
        ));
    }
    goal.push_str("(assert (or p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11))\n(check-sat)\n");

    let mut ctx = Context::new();
    let script = format!("(set-logic QF_BV)\n(set-option :max-decisions 1)\n{goal}");
    let output = ctx.execute_script(&script).unwrap_or_default().join("\n");
    assert!(
        !verdicts(&output).is_empty(),
        "the script must produce a verdict; got:\n{output}"
    );
    let bounded_decisions = ctx.stats().decisions;

    // Without the budget the same goal is decided normally, and takes more
    // decisions than the budget allowed — so the bound above is not vacuous.
    let mut free_ctx = Context::new();
    let free_output = free_ctx
        .execute_script(&format!("(set-logic QF_BV)\n{goal}"))
        .unwrap_or_default()
        .join("\n");
    assert_eq!(
        verdicts(&free_output).last().copied(),
        Some(SolverResult::Sat),
        "the goal itself is satisfiable; got:\n{free_output}"
    );
    let free_decisions = free_ctx.stats().decisions;

    assert!(
        free_decisions > 2,
        "the goal must need real branching for the bound to mean anything; it took \
         {free_decisions} decisions"
    );
    assert!(
        bounded_decisions <= 2,
        "`:max-decisions 1` must bound the outer search, it made {bounded_decisions} decisions \
         (unbudgeted: {free_decisions})"
    );
}
