//! Unit tests for the refuted-model blocking loop (issue #40).
//!
//! These are white-box on purpose: the feature's contract is about solver
//! *internals* — which literals a blocking clause names, that the counter is
//! snapshot-scoped, that the repair paths still see the candidate model — and
//! none of that is visible through `Context`'s SMT-LIB surface. The end-to-end
//! verdict behaviour is pinned in `tests/issue40_model_blocking.rs`.
//!
//! # The trigger these tests use
//!
//! `model_refutes_assertions` fires on `EvalOutcome::Unrepresentable` as well
//! as on a definite `Bool(false)`, and the former is by far the easiest thing
//! to reproduce deliberately: `EvalVal::Num` is a fixed-width `Rational64`, so
//! a sum of two model values near `2^62` is not representable even though both
//! operands are. `(or (= x 1) (= x 2^62))` together with `(>= (+ x x) 0)`
//! therefore has one candidate the evaluator refuses to certify and one it
//! accepts — exactly the shape the blocking loop exists to get past.
//!
//! **Which of the two the search reaches first is empirical**, not guaranteed:
//! it falls out of the SAT core's phase and decision heuristics, and it was
//! confirmed by probing both disjunct orders on this tree. That is why the
//! tests below assert `model_blocking_clauses >= 1` alongside the verdict — if a
//! heuristic change ever makes the certifiable candidate come first, the goal is
//! still `sat` and only that assertion fails, which is the signal to re-probe
//! for a fresh refuted-first shape rather than to suspect the blocking loop.

use super::super::*;
use super::MAX_MODEL_BLOCKING_ROUNDS;
use num_bigint::BigInt;
use oxiz_sat::LBool;

/// `2^62`: representable as an `i64`, but `BIG + BIG` is not.
const BIG: i64 = 4_611_686_018_427_387_904;

/// `(or (= x 1) (= x 2^62))` and `(>= (+ x x) 0)`.
///
/// The search reaches the `x = 2^62` candidate first, whose `(+ x x)` the
/// evaluator cannot represent; `x = 1` is a candidate it certifies.
fn overflow_escape_goal(manager: &mut TermManager) -> Vec<TermId> {
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let one = manager.mk_int(1);
    let big = manager.mk_int(BigInt::from(BIG));
    let small_eq = manager.mk_eq(x, one);
    let big_eq = manager.mk_eq(x, big);
    let choice = manager.mk_or(vec![small_eq, big_eq]);
    let sum = manager.mk_add(vec![x, x]);
    let zero = manager.mk_int(0);
    let non_negative = manager.mk_ge(sum, zero);
    vec![choice, non_negative]
}

/// The same goal with **no** representable escape: every candidate value
/// overflows the evaluator, so no amount of blocking can produce a certified
/// model.
fn overflow_only_goal(manager: &mut TermManager) -> Vec<TermId> {
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let zero = manager.mk_int(0);
    let mut choices = Vec::new();
    for offset in 0..4 {
        let value = manager.mk_int(BigInt::from(BIG + offset));
        choices.push(manager.mk_eq(x, value));
    }
    let choice = manager.mk_or(choices);
    let sum = manager.mk_add(vec![x, x]);
    let non_negative = manager.mk_ge(sum, zero);
    vec![choice, non_negative]
}

fn solver_with(config: SolverConfig) -> Solver {
    let mut solver = Solver::with_config(config);
    solver.set_logic("QF_LIA");
    solver
}

// ---------------------------------------------------------------------
// The projection rule
// ---------------------------------------------------------------------

/// A projection that names no literal at all is the **empty clause**, i.e.
/// `false` — it would exclude the entire search space rather than one
/// assignment, and every subsequent solve would report `Unsat` (downgraded to
/// `Unknown`, so not *wrong*, but the goal would have become undecidable for no
/// reason). `block_refuted_model` must decline instead, leaving the solver
/// exactly as it found it.
///
/// This is the point where the rule is the exact opposite of MBQI's
/// all-or-nothing reason clause: there, an unmapped term means "add nothing";
/// here, unmapped variables are dropped freely and only the *degenerate* result
/// is refused.
#[test]
fn all_or_nothing_empty_projection_declines() {
    let mut solver = solver_with(SolverConfig::default());
    let mut manager = TermManager::new();
    // Encode a variable so `var_to_term` is non-empty, but never solve, so
    // nothing on it is assigned.
    let p = manager.mk_var("p", manager.sorts.bool_sort);
    solver.assert(p, &mut manager);
    assert!(
        !solver.var_to_term.is_empty(),
        "the goal must contribute at least one mapped SAT variable, \
         or this test proves nothing about the projection"
    );

    assert!(
        solver.refuted_model_projection().is_empty(),
        "no solve has run, so no mapped variable has a polarity to project"
    );

    let clauses_before = solver.sat.num_clauses();
    assert!(
        !solver.block_refuted_model(),
        "an empty projection must decline rather than add the empty clause"
    );
    assert_eq!(solver.model_blocking_active, 0);
    assert_eq!(solver.statistics.model_blocking_clauses, 0);
    assert_eq!(
        solver.sat.num_clauses(),
        clauses_before,
        "a declined block must leave the clause database untouched"
    );
}

/// A mapped variable the SAT core left `Undef` is dropped, never guessed.
///
/// Both polarities are consistent with what the search committed to, so
/// omitting the variable blocks the candidate for both at once. Adding a guessed
/// literal would leave the sibling assignment — same commitments, opposite guess
/// — unblocked, and the very same refuted candidate could come straight back.
#[test]
fn undef_mapped_var_is_dropped_not_guessed() {
    let mut solver = solver_with(SolverConfig::default());
    let mut manager = TermManager::new();
    let p = manager.mk_var("p", manager.sorts.bool_sort);
    solver.assert(p, &mut manager);
    assert_eq!(solver.check(&mut manager), SolverResult::Sat);

    // Two more variables, introduced *after* the solve: they are mapped, so the
    // projection considers them, but the last model says nothing about them.
    let q = manager.mk_var("q", manager.sorts.bool_sort);
    let r = manager.mk_var("r", manager.sorts.bool_sort);
    let either = manager.mk_or(vec![q, r]);
    solver.assert(either, &mut manager);

    let p_var = *solver.term_to_var.get(&p).expect("p is mapped");
    let q_var = *solver.term_to_var.get(&q).expect("q is mapped");
    assert_eq!(
        solver.sat.model_value(p_var),
        LBool::True,
        "p was asserted, so the solve that just ran must have assigned it true"
    );
    assert_eq!(
        solver.sat.model_value(q_var),
        LBool::Undef,
        "q was created after the solve, so it carries no polarity"
    );

    let projection = solver.refuted_model_projection();
    assert!(
        projection.contains(&Lit::neg(p_var)),
        "an assigned variable is negated into the blocking clause"
    );
    assert!(
        !projection.contains(&Lit::pos(q_var)) && !projection.contains(&Lit::neg(q_var)),
        "an unassigned variable must contribute no literal in either polarity, \
         got {projection:?}"
    );
}

// ---------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------

/// The headline fix: a candidate the gate refuses is excluded and the search
/// resumes, so a goal whose *second* candidate is a real model answers `sat`
/// instead of the `unknown` it used to.
#[test]
fn blocked_retry_finds_real_model() {
    let mut solver = solver_with(SolverConfig::default());
    let mut manager = TermManager::new();
    for assertion in overflow_escape_goal(&mut manager) {
        solver.assert(assertion, &mut manager);
    }

    assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    assert!(
        solver.statistics.model_blocking_clauses >= 1,
        "the first candidate must have been refuted and blocked, \
         or this test is not exercising the loop at all"
    );
    assert!(
        solver.model.is_some(),
        "a reported `Sat` must come with the model that survived the gate"
    );
}

/// A goal with no certifiable candidate at all must stop, not spin: every
/// candidate is blocked, the search runs out, and the verdict is `Unknown`.
///
/// The verdict assertion is doing double duty. Once blocking has started, the
/// SAT core's `Unsat` means "no model outside the excluded region", which is
/// *not* a refutation of the goal — reporting `Unsat` here would be the
/// wrong-answer failure this whole mechanism has to avoid, and is strictly worse
/// than the spurious `Unknown` it set out to fix.
#[test]
fn unsat_terminates_within_budget() {
    let mut solver = solver_with(SolverConfig::default());
    let mut manager = TermManager::new();
    for assertion in overflow_only_goal(&mut manager) {
        solver.assert(assertion, &mut manager);
    }

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unknown,
        "no candidate survives the gate, and an `Unsat` reached over blocking \
         clauses must never be reported as one"
    );
    let budget = u64::try_from(solver.config.max_model_blocking_rounds)
        .expect("the round budget fits in a u64");
    assert!(
        solver.statistics.model_blocking_clauses <= budget,
        "the loop must respect its round budget"
    );
    assert!(
        solver.model.is_none(),
        "the final `Unknown` exit owns clearing the refuted model"
    );
    assert!(
        solver.unsat_core.is_none(),
        "there is no core to hand over for a verdict reached over blocking clauses"
    );
}

/// The blocking clauses outlive the `check` that added them, so a *later*
/// `check` on the same solver is searching a restricted space. Its `Unsat` must
/// still be downgraded — this is the cross-check poisoning that makes the
/// counter a snapshot field rather than a per-search one.
#[test]
fn unsat_downgraded_while_blocking_active() {
    let mut solver = solver_with(SolverConfig::default());
    let mut manager = TermManager::new();
    for assertion in overflow_escape_goal(&mut manager) {
        solver.assert(assertion, &mut manager);
    }
    assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    assert!(
        solver.statistics.model_blocking_clauses >= 1,
        "the first check must have blocked something"
    );
    assert!(solver.blocking_clauses_present());

    // Now make the goal genuinely unsatisfiable, on top of a database that is
    // already restricted.
    let x = manager.mk_var("x", manager.sorts.int_sort);
    let one = manager.mk_int(1);
    let big = manager.mk_int(BigInt::from(BIG));
    let small_eq = manager.mk_eq(x, one);
    let big_eq = manager.mk_eq(x, big);
    let not_small = manager.mk_not(small_eq);
    let not_big = manager.mk_not(big_eq);
    solver.assert(not_small, &mut manager);
    solver.assert(not_big, &mut manager);

    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unknown,
        "an `Unsat` derived from a database carrying model-blocking clauses is \
         not a refutation of the goal"
    );
    assert!(
        solver.unsat_core.is_none(),
        "and it comes with no core, since the proof rests on clauses no \
         assertion entails"
    );
}

/// `pop` retracts the blocking clauses through `sat.pop()`, so the counter that
/// records how many are live has to roll back with them — otherwise the
/// downgrade would keep firing forever and the solver could never report
/// `unsat` again.
#[test]
fn blocking_counter_retracted_by_pop() {
    let mut solver = solver_with(SolverConfig::default());
    let mut manager = TermManager::new();

    solver.push();
    for assertion in overflow_escape_goal(&mut manager) {
        solver.assert(assertion, &mut manager);
    }
    assert_eq!(solver.check(&mut manager), SolverResult::Sat);
    assert!(solver.blocking_clauses_present());
    assert!(solver.model_blocking_active >= 1);

    solver.pop();
    assert_eq!(
        solver.model_blocking_active, 0,
        "the clauses went with `sat.pop()`; the count must go with them"
    );
    assert!(!solver.blocking_clauses_present());

    // And the solver can report a real `unsat` again.
    let p = manager.mk_var("p", manager.sorts.bool_sort);
    let not_p = manager.mk_not(p);
    solver.assert(p, &mut manager);
    solver.assert(not_p, &mut manager);
    assert_eq!(
        solver.check(&mut manager),
        SolverResult::Unsat,
        "with no blocking clause live, an `Unsat` is reported as one"
    );
}

/// The two repair paths run on a *live* candidate model.
///
/// The gate used to sit ahead of them and clear `self.model`, which made the
/// repairs unreachable for refuted candidates and — for the array path, which
/// reads the model to skip instances the candidate already satisfies — would
/// silently degenerate into eager instantiation if it were reached with `None`.
#[test]
fn repair_paths_see_the_model() {
    let mut solver = solver_with(SolverConfig::default());
    let mut manager = TermManager::new();
    for assertion in overflow_escape_goal(&mut manager) {
        solver.assert(assertion, &mut manager);
    }
    assert_eq!(solver.check(&mut manager), SolverResult::Sat);

    assert!(
        solver.statistics.model_blocking_clauses >= 1,
        "at least one candidate must have been refuted, so at least one of the \
         recorded rounds is a round the old order would have bailed out of"
    );
    assert!(
        !solver.repair_paths_saw_model.is_empty(),
        "the ground branch must have reached the repair paths"
    );
    assert!(
        solver.repair_paths_saw_model.iter().all(|&seen| seen),
        "every repair round must have run with the candidate model still in \
         place, got {:?}",
        solver.repair_paths_saw_model
    );
}

// ---------------------------------------------------------------------
// The switch
// ---------------------------------------------------------------------

/// With `enable_model_blocking` off, a refuted candidate is conceded exactly as
/// it was before issue #40: `Unknown`, no model, no core, no clause added.
///
/// Note what the flag does *not* turn off: the reordering that puts the repair
/// paths ahead of the gate is a bug fix, not a feature, and stays live.
#[test]
fn enable_model_blocking_false_is_old_behaviour() {
    let config = SolverConfig {
        enable_model_blocking: false,
        ..SolverConfig::default()
    };
    let mut solver = solver_with(config);
    let mut manager = TermManager::new();
    for assertion in overflow_escape_goal(&mut manager) {
        solver.assert(assertion, &mut manager);
    }

    assert_eq!(solver.check(&mut manager), SolverResult::Unknown);
    assert_eq!(solver.statistics.model_blocking_clauses, 0);
    assert_eq!(solver.model_blocking_active, 0);
    assert!(solver.model.is_none());
    assert!(solver.unsat_core.is_none());
    assert!(
        !solver.repair_paths_saw_model.is_empty()
            && solver.repair_paths_saw_model.iter().all(|&seen| seen),
        "the reorder is not gated by the flag"
    );
}

/// A zero round budget is as complete a disable as the flag is, and it is the
/// shape `SolverConfig::minimal()` ships.
#[test]
fn zero_round_budget_declines() {
    let config = SolverConfig {
        max_model_blocking_rounds: 0,
        ..SolverConfig::default()
    };
    let mut solver = solver_with(config);
    let mut manager = TermManager::new();
    for assertion in overflow_escape_goal(&mut manager) {
        solver.assert(assertion, &mut manager);
    }

    assert_eq!(solver.check(&mut manager), SolverResult::Unknown);
    assert_eq!(solver.statistics.model_blocking_clauses, 0);
}

/// `minimal()` opts out; the other three presets opt in with the module's own
/// constant, so the budget exists in exactly one place.
#[test]
fn presets_agree_with_the_module_constant() {
    for config in [
        SolverConfig::fast(),
        SolverConfig::balanced(),
        SolverConfig::thorough(),
    ] {
        assert!(config.enable_model_blocking);
        assert_eq!(config.max_model_blocking_rounds, MAX_MODEL_BLOCKING_ROUNDS);
    }
    let minimal = SolverConfig::minimal();
    assert!(!minimal.enable_model_blocking);
    assert_eq!(minimal.max_model_blocking_rounds, 0);
}
