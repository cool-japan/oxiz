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

/// Run a script and return its joined output, the outer SAT engine's
/// propagation count and the wall time.
///
/// `propagations` is the witness that a `(check-sat)` really reached
/// `check_core`: `Solver::check` consults a verdict cache first
/// (`solver/verdict_cache.rs`), so a second, *identical* check in the same
/// script replays the first verdict without searching at all — measured, the
/// counter stays at exactly 3 for one check and for two identical ones, and
/// rises for two that differ. On this bit-blasted goal it is the only outer
/// counter that moves (conflicts and decisions stay at 0: the work is all
/// inside the embedded solver), which is precisely the finding this file is
/// about.
fn run_script_measured(script: &str) -> (String, u64, Duration) {
    let mut ctx = Context::new();
    let start = Instant::now();
    let output = ctx.execute_script(script).unwrap_or_default().join("\n");
    let elapsed = start.elapsed();
    (output, ctx.stats().propagations, elapsed)
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
    let start = Instant::now();
    let output = run_script_output(&mul_vc("(set-option :timeout 100)\n"));
    let elapsed = start.elapsed();

    assert_eq!(
        verdicts(&output),
        vec![SolverResult::Unknown],
        "`:timeout 100` must cut the bit-blasted solve short; got:\n{output}"
    );
    // The measured pre-fix time is 17 s; 2 s leaves ~8x headroom over the
    // budget's own granularity (the deadline is polled once per 256 CDCL loop
    // iterations, and a single `propagate()` is not itself interruptible) while
    // still failing loudly if the budget stops reaching the embedded solver.
    assert!(
        elapsed < Duration::from_secs(2),
        "`:timeout 100` let the check run for {elapsed:?}"
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
/// development machine (see this file's header table for the pre-fix
/// baseline); run it with
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
    let start = Instant::now();
    let output = run_script_output(&mul_vc("(set-option :max-conflicts 50)\n"));
    let elapsed = start.elapsed();

    assert_eq!(
        verdicts(&output),
        vec![SolverResult::Unknown],
        "`:max-conflicts 50` must cut the bit-blasted solve short; got:\n{output}"
    );
    assert!(
        elapsed < Duration::from_secs(2),
        "`:max-conflicts 50` let the check run for {elapsed:?} (was 13 s before the fix)"
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
    let (one_output, one_props, _) = run_script_measured(&mul_vc("(set-option :timeout 100)\n"));
    assert_eq!(
        verdicts(&one_output),
        vec![SolverResult::Unknown],
        "the single-check baseline must already be bounded; got:\n{one_output}"
    );

    let two = format!(
        "{prelude}{MUL_VC}\n(assert (bvult a #x00000000ffffffff))\n(check-sat)\n(get-info \
         :reason-unknown)"
    );
    let (two_output, two_props, elapsed) = run_script_measured(&two);
    assert_eq!(
        verdicts(&two_output),
        vec![SolverResult::Unknown, SolverResult::Unknown],
        "both checks under one `:timeout` must be bounded; got:\n{two_output}"
    );
    assert!(
        two_props > one_props,
        "the second check must actually search ({one_props} -> {two_props} outer propagations); \
         if it does not grow, the verdict cache replayed the first verdict and this test proves \
         nothing"
    );
    assert!(
        elapsed < Duration::from_secs(4),
        "two bounded checks took {elapsed:?}"
    );
}

#[test]
fn a_budget_survives_push_and_pop() {
    // `Solver::pop` runs `invalidate_results`, which drops the cached verdict,
    // so here the two checks may be identical and the second still runs; the
    // propagation growth pins that rather than assuming it.
    let prelude = "(set-logic QF_BV)\n(set-option :max-conflicts 50)\n";
    let one = format!("{prelude}(push 1)\n{MUL_VC}\n(pop 1)");
    let (one_output, one_props, _) = run_script_measured(&one);
    assert_eq!(
        verdicts(&one_output),
        vec![SolverResult::Unknown],
        "the conflict budget must bound the check inside a scope; got:\n{one_output}"
    );

    let two = format!("{prelude}(push 1)\n{MUL_VC}\n(pop 1)\n(push 1)\n{MUL_VC}\n(pop 1)");
    let (two_output, two_props, elapsed) = run_script_measured(&two);
    assert_eq!(
        verdicts(&two_output),
        vec![SolverResult::Unknown, SolverResult::Unknown],
        "the conflict budget must survive push/pop; got:\n{two_output}"
    );
    assert!(
        two_props > one_props,
        "the second scope's check must actually search ({one_props} -> {two_props} outer \
         propagations)"
    );
    assert!(
        elapsed < Duration::from_secs(4),
        "two bounded checks around push/pop took {elapsed:?}"
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
    const BUDGET: u64 = 200;
    const GENEROUS: u64 = 5_000;

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
/// multiplier/adder comparisons over six shared 32-bit variables, so the
/// CDCL(T) loop asserts one bit-vector atom at a time and re-checks after each.
fn multi_probe_goal() -> String {
    let mut goal = String::new();
    for i in 0..6 {
        goal.push_str(&format!("(declare-const v{i} (_ BitVec 32))\n"));
    }
    for i in 0..6u32 {
        for j in (i + 1)..6 {
            goal.push_str(&format!(
                "(assert (bvult (bvmul v{i} v{j}) (bvadd v{i} v{j})))\n"
            ));
        }
    }
    goal.push_str("(assert (bvugt (bvmul v0 v1) #x7fffffff))\n(check-sat)\n");
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
