//! U-Z12 — `BvSolver::set_budget` at the theory-solver API level.
//!
//! `oxiz-solver/tests/budget_bitblast.rs` pins the SMT-LIB end of this wiring
//! (`(set-option :timeout N)` / `:max-conflicts N` reaching the bit-blasted
//! solves). These tests pin the two properties of the API itself that the
//! script path relies on but cannot observe directly:
//!
//! * the conflict allowance is a *total*, and its accumulator survives
//!   `Theory::reset` — which `oxiz-solver`'s `rebase_theory_state` calls once
//!   per check and again on every repair round, and which zeroes the embedded
//!   SAT solver's own statistics;
//! * `TheoryCombination::notify_equality` handles an exhausted solve honestly.
//!   That path is off the SMT-LIB script path today (the theory manager only
//!   forwards equality notifications to the arithmetic solver) but is public
//!   API, and before this fix its `_` arm read a model out of a solve that had
//!   never finished.

use oxiz_core::ast::TermId;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::{EqualityNotification, Theory, TheoryCheckResult, TheoryCombination};

/// A goal that needs real search, asserted directly through the bit-blasting
/// API so each `check()` is one probe.
///
/// `vars` 32-bit variables in a ring, each pair constrained so that the modular
/// product `v_i * v_{i+1}` is *below* the sum `v_i + v_{i+1}` — satisfiable
/// only through wraparound — plus one constraint forcing the first product
/// above `0x7fff_ffff`. The tension between "the product wrapped low" and "the
/// product is huge" is what makes the bit-blasted search generate conflicts
/// rather than reading a model straight off the trail; without the last
/// constraint the chain costs zero conflicts and every budget assertion over it
/// passes vacuously.
fn build_multiplier_chain(solver: &mut BvSolver, vars: u32) {
    // 16 bits, not 32.
    //
    // Every assertion in this file is expressed in *conflicts* — the unit the
    // budget itself is expressed in — so the width changes nothing any of them
    // observes; it only sets how long each bit-blasted `solve()` takes.
    // Measured in the dev profile (`[profile.dev]` is `opt-level = 1`) on the
    // development machine, load average ~11, for `spend_under(Some(b), 4, 4)`
    // on an 8-variable chain:
    //
    // | width | b = 150 | b = 400 | b = 2000 |
    // |---|---|---|---|
    // | 32 | 5.44 s | 17.2 s | 116.0 s |
    // | 16 | 0.77 s |  2.70 s |  13.4 s |
    //
    // The spend is identical in both rows (`spent == b` throughout): the
    // allowance, not the arithmetic, is what ends these searches, which is the
    // property the file tests. The same change at the other two call sites,
    // same conditions:
    //
    // | call site | 32 bits | 16 bits | spend |
    // |---|---|---|---|
    // | `set_budget_re_arms_the_allowance` (8 vars, 4 probes, 200) | 8.89 s | 1.87 s | 200 |
    // | `a_generous_allowance_leaves_notify_equality_working` | 17.8 s | 0.35 s | 135 |
    //
    // (the second row is the 4-variable chain under a 20 000 allowance)
    //
    // Under `cargo nextest` on a loaded machine those three tests measured 91 s
    // (past the repo's 60 s SLOW threshold in `.config/nextest.toml`), 18.8 s
    // and 38.3 s at 32 bits.
    const WIDTH: u32 = 16;
    for i in 0..vars {
        solver.new_bv(TermId::new(i + 1), WIDTH);
    }
    for i in 0..vars {
        let a = TermId::new(i + 1);
        let b = TermId::new(((i + 1) % vars) + 1);
        let product = TermId::new(1000 + i);
        let sum = TermId::new(2000 + i);
        solver.new_bv(product, WIDTH);
        solver.new_bv(sum, WIDTH);
        assert!(solver.bv_mul(product, a, b));
        assert!(solver.bv_add(sum, a, b));
        assert!(solver.assert_ult(product, sum));
    }
    let threshold = TermId::new(3000);
    solver.new_bv(threshold, WIDTH);
    // Half the widest value this width can hold (`0x7fff_ffff` at WIDTH = 32,
    // `0x7fff` at 16). It has to scale with the width rather than stay a
    // literal: `assert_const` reads its value modulo `2^WIDTH`, so a
    // hard-coded `0x7fff_ffff` at WIDTH = 16 would pin the threshold to
    // `0xffff`, the *widest* 16-bit value — `threshold < product` would then be
    // unsatisfiable by construction, the chain would cost zero conflicts, and
    // every budget assertion over it would pass vacuously (see this function's
    // doc comment).
    let threshold_value = (1u64 << (WIDTH - 1)) - 1;
    assert!(solver.assert_const(threshold, threshold_value, WIDTH));
    assert!(solver.assert_ult(threshold, TermId::new(1000)));
}

/// Run `probes` probes on a fresh chain under `budget`, resetting between
/// rounds the way `oxiz-solver`'s `rebase_theory_state` does, and return the
/// accumulated embedded conflict spend.
fn spend_under(budget: Option<u64>, rounds: usize, probes: usize) -> u64 {
    let mut solver = BvSolver::new();
    solver.set_budget(budget, None);
    build_multiplier_chain(&mut solver, 8);
    for _ in 0..rounds {
        for _ in 0..probes {
            let _ = solver.check();
        }
        // `reset()` calls `sat.reset()`, which zeroes the embedded statistics;
        // only the `conflicts_spent` accumulator keeps the allowance a total
        // across that, and the owning solver does this on every repair round.
        solver.reset();
        build_multiplier_chain(&mut solver, 8);
    }
    solver.conflicts_spent()
}

#[test]
fn the_conflict_allowance_is_a_total_across_probes_and_survives_reset() {
    const BUDGET: u64 = 150;

    // Two-sided, so the bound cannot pass vacuously: the same work under a
    // larger allowance must spend more than `BUDGET`.  The larger allowance is
    // itself a bound (rather than `None`) only to keep this test's wall time
    // down: unbounded, the chain runs for tens of seconds, which is the
    // property being relied on.
    //
    // 400 rather than 2000: the assertion only needs the larger allowance to be
    // *spent past* `BUDGET`, and it is spent in full at either size (the chain
    // never finishes on its own), so 400 witnesses the same thing at a fifth of
    // the cost.  Measured in the dev profile (`opt-level = 1`), load average
    // ~11, for the two `spend_under` calls of this test together: 32-bit/2000
    // = 121 s, 16-bit/2000 = 14.1 s, 16-bit/400 = 3.5 s. The 32-bit/2000
    // version measured 91 s as a whole test under `cargo nextest`, past its
    // 60 s SLOW threshold.
    const GENEROUS: u64 = 400;
    let unbounded_spend = spend_under(Some(GENEROUS), 4, 4);
    assert!(
        unbounded_spend > BUDGET,
        "the chain must genuinely cost more than {BUDGET} embedded conflicts for the bound below \
         to mean anything; under a {GENEROUS}-conflict allowance it spent {unbounded_spend}"
    );

    let bounded_spend = spend_under(Some(BUDGET), 4, 4);
    assert!(
        bounded_spend <= BUDGET,
        "the allowance is a total across every probe and every reset: spent {bounded_spend} of \
         {BUDGET} (the larger-allowance run spends {unbounded_spend})"
    );
}

#[test]
fn set_budget_re_arms_the_allowance() {
    let mut solver = BvSolver::new();
    solver.set_budget(Some(200), None);
    build_multiplier_chain(&mut solver, 8);
    for _ in 0..4 {
        let _ = solver.check();
    }
    let spent = solver.conflicts_spent();

    // Re-arming zeroes the accumulator: the budget's period is decided by the
    // caller, and `oxiz-solver` re-arms once per `(check-sat)`.
    solver.set_budget(Some(200), None);
    assert_eq!(
        solver.conflicts_spent(),
        0,
        "set_budget must re-arm the allowance (previous spend was {spent})"
    );

    // Clearing the budget leaves the accumulator at zero and stops bounding.
    solver.set_budget(None, None);
    assert_eq!(solver.conflicts_spent(), 0);
}

#[test]
fn an_exhausted_allowance_makes_a_probe_report_unknown_not_sat() {
    // A zero allowance means the first solve stops at its first budget poll.
    // On this chain — which needs thousands of conflicts — the probe must
    // report `Unknown`, never `Sat` (which the CDCL(T) loop would read as "this
    // theory has no objection") and never `Unsat` (nothing was refuted).
    let mut solver = BvSolver::new();
    solver.set_budget(Some(0), None);
    build_multiplier_chain(&mut solver, 8);
    let result = solver.check().expect("check does not fail");
    assert!(
        matches!(result, TheoryCheckResult::Unknown),
        "a probe with no allowance must report Unknown, got {result:?}"
    );
    assert_eq!(
        solver.conflicts_spent(),
        0,
        "a probe that never entered the search cannot have spent conflicts"
    );
}

#[test]
fn notify_equality_under_an_exhausted_allowance_derives_no_equalities() {
    // Both operands are bit-blasted, so `notify_equality` takes the branch that
    // asserts the equality and re-solves. With no allowance left that solve
    // returns `Unknown`, and the method must report "accepted" (it derived no
    // refutation) *without* reading a model out of the unfinished search.
    let mut solver = BvSolver::new();
    build_multiplier_chain(&mut solver, 8);
    solver.set_budget(Some(0), None);

    let lhs = TermId::new(1);
    let rhs = TermId::new(2);
    let accepted = solver.notify_equality(EqualityNotification {
        lhs,
        rhs,
        reason: None,
    });
    assert!(
        accepted,
        "an exhausted solve cannot refute an equality, so it must not report a conflict"
    );
    assert!(
        solver.get_shared_equalities().is_empty(),
        "no shared equality may be derived from a solve that never finished; got {:?}",
        solver.get_shared_equalities().len()
    );
}

#[test]
fn a_generous_allowance_leaves_notify_equality_working() {
    // The control for the test above: with budget the same call reaches the
    // `Sat` arm and does extract model equalities, so the `Unknown` arm is a
    // genuinely different path rather than the only one ever taken.
    let mut solver = BvSolver::new();
    build_multiplier_chain(&mut solver, 4);
    solver.set_budget(Some(20_000), None);

    let accepted = solver.notify_equality(EqualityNotification {
        lhs: TermId::new(1),
        rhs: TermId::new(2),
        reason: None,
    });
    assert!(accepted);
}
