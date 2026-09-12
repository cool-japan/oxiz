//! Tests for the wall-clock deadline and the decision budget (finding U-Z12).
//!
//! `Solver::set_max_conflicts` was the only work budget the CDCL loop honoured,
//! and nothing in the workspace ever called it, so a `solve()` on a hard
//! instance was unbounded no matter what `(set-option :timeout N)` or
//! `(set-option :max-decisions N)` said. `Solver::set_deadline` and
//! `Solver::set_max_decisions` are the two levers that fix that; these tests
//! pin their contract at this layer, and `oxiz-solver/tests/budget_bitblast.rs`
//! pins the script-level end of the same wiring.
//!
//! Every instance below is a pigeonhole principle PHP(pigeons, holes):
//! unsatisfiable when `pigeons > holes`, with no unit clauses, so the pre-loop
//! phases of `solve()` (the lucky-phase scan in particular) cannot decide it and
//! the budget check at the top of the CDCL loop is genuinely the thing that ends
//! the search.

use oxiz_sat::{Solver, SolverResult};
use oxiz_time::{Duration, Instant};

/// Add PHP(`pigeons`, `holes`) to `solver`, allocating its variables first.
///
/// Variable `holes * (p - 1) + h` (1-based) means "pigeon `p` sits in hole `h`".
/// Each pigeon is in at least one hole, and no hole takes two pigeons; with
/// `pigeons > holes` that is unsatisfiable, and the refutation is exponentially
/// hard for resolution, which is exactly what a budget test needs.
fn add_pigeonhole(solver: &mut Solver, pigeons: i32, holes: i32) {
    for _ in 0..(pigeons * holes) {
        solver.new_var();
    }
    let var_of = |p: i32, h: i32| holes * (p - 1) + h;
    for p in 1..=pigeons {
        let clause: Vec<i32> = (1..=holes).map(|h| var_of(p, h)).collect();
        solver.add_clause_dimacs(&clause);
    }
    for h in 1..=holes {
        for p in 1..=pigeons {
            for q in (p + 1)..=pigeons {
                solver.add_clause_dimacs(&[-var_of(p, h), -var_of(q, h)]);
            }
        }
    }
}

/// A fresh solver holding PHP(`pigeons`, `holes`).
fn pigeonhole(pigeons: i32, holes: i32) -> Solver {
    let mut solver = Solver::new();
    add_pigeonhole(&mut solver, pigeons, holes);
    solver
}

/// An instant safely in the past. `checked_sub` is `None` only if the process
/// started less than two hours ago *and* the platform clock has no earlier
/// representable point, in which case `Instant::now()` is used: still expired by
/// the time `solve()` polls it, so the test's intent survives either way.
fn two_hours_ago() -> Instant {
    Instant::now()
        .checked_sub(Duration::from_secs(7200))
        .unwrap_or_else(Instant::now)
}

// ---------------------------------------------------------------------------
// (a) A deadline already in the past stops the search.
// ---------------------------------------------------------------------------

#[test]
fn a_deadline_in_the_past_makes_solve_return_unknown() {
    let mut solver = pigeonhole(11, 10);
    solver.set_deadline(Some(two_hours_ago()));
    let start = Instant::now();
    let result = solver.solve();
    let elapsed = start.elapsed();
    assert_eq!(
        result,
        SolverResult::Unknown,
        "an expired deadline must stop the search before it refutes PHP(11,10)"
    );
    assert!(
        elapsed < Duration::from_secs(5),
        "the deadline must be polled promptly, took {elapsed:?}"
    );
}

#[test]
fn a_short_deadline_bounds_a_hard_refutation() {
    // The same instance without a budget takes far longer than 50 ms; with one
    // it must come back `Unknown` rather than run to completion. The assertion
    // is deliberately loose (5 s, not 50 ms): the poll throttle reads the clock
    // once per 256 loop iterations and a single `propagate()` is not itself
    // interruptible, so the deadline is a floor on when the search *may* stop,
    // never a hard real-time guarantee.
    let mut solver = pigeonhole(12, 11);
    let deadline = Instant::now().checked_add(Duration::from_millis(50));
    assert!(deadline.is_some(), "a 50 ms deadline must be representable");
    solver.set_deadline(deadline);
    let start = Instant::now();
    let result = solver.solve();
    let elapsed = start.elapsed();
    assert_eq!(result, SolverResult::Unknown);
    assert!(
        elapsed < Duration::from_secs(5),
        "a 50 ms deadline let the search run for {elapsed:?}"
    );
}

// ---------------------------------------------------------------------------
// (b) Clearing the deadline restores unbounded behaviour.
// ---------------------------------------------------------------------------

#[test]
fn clearing_the_deadline_restores_unbounded_search() {
    let mut solver = pigeonhole(7, 6);
    solver.set_deadline(Some(two_hours_ago()));
    assert_eq!(solver.solve(), SolverResult::Unknown);

    solver.set_deadline(None);
    assert_eq!(
        solver.solve(),
        SolverResult::Unsat,
        "with the deadline cleared the same instance must be refuted"
    );
}

// ---------------------------------------------------------------------------
// (c) `reset()` keeps the budgets and zeroes the statistics.
// ---------------------------------------------------------------------------

#[test]
fn reset_preserves_the_budgets_and_zeroes_the_stats() {
    let mut solver = pigeonhole(8, 7);
    solver.set_max_conflicts(Some(1));
    solver.set_max_decisions(Some(1));
    solver.set_deadline(Some(two_hours_ago()));
    assert_eq!(solver.solve(), SolverResult::Unknown);

    solver.reset();
    assert_eq!(
        solver.stats().conflicts,
        0,
        "reset must zero the cumulative counters"
    );
    assert_eq!(solver.stats().decisions, 0);

    // The budgets survived the reset: the same problem rebuilt on this solver
    // is still cut off. This is the property `BvSolver::reset` depends on — its
    // `sat.reset()` must not silently re-arm an exhausted allowance, which is
    // why `BvSolver` accumulates its spend in a field of its own.
    add_pigeonhole(&mut solver, 8, 7);
    assert_eq!(
        solver.solve(),
        SolverResult::Unknown,
        "the deadline and the conflict/decision budgets must survive reset()"
    );

    // ...and clearing them afterwards really does unbind the search.
    solver.set_deadline(None);
    solver.set_max_conflicts(None);
    solver.set_max_decisions(None);
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

// ---------------------------------------------------------------------------
// (d) A generous deadline does not disturb a normal solve.
// ---------------------------------------------------------------------------

#[test]
fn a_generous_deadline_still_lets_a_satisfiable_instance_answer_sat() {
    // PHP(n, n) is satisfiable: one pigeon per hole.
    let mut solver = pigeonhole(8, 8);
    solver.set_deadline(Instant::now().checked_add(Duration::from_secs(600)));
    assert_eq!(solver.solve(), SolverResult::Sat);
}

#[test]
fn a_generous_deadline_still_lets_an_unsatisfiable_instance_answer_unsat() {
    let mut solver = pigeonhole(7, 6);
    solver.set_deadline(Instant::now().checked_add(Duration::from_secs(600)));
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

// ---------------------------------------------------------------------------
// The decision budget.
// ---------------------------------------------------------------------------

#[test]
fn a_one_decision_budget_stops_a_hard_refutation() {
    let mut solver = pigeonhole(11, 10);
    solver.set_max_decisions(Some(1));
    assert_eq!(solver.solve(), SolverResult::Unknown);
    assert!(
        solver.stats().decisions <= 2,
        "the search must stop at the budget, made {} decisions",
        solver.stats().decisions
    );
}

#[test]
fn clearing_the_decision_budget_restores_unbounded_search() {
    let mut solver = pigeonhole(7, 6);
    solver.set_max_decisions(Some(1));
    assert_eq!(solver.solve(), SolverResult::Unknown);
    solver.set_max_decisions(None);
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

#[test]
fn a_generous_decision_budget_does_not_change_a_verdict() {
    let mut solver = pigeonhole(7, 6);
    solver.set_max_decisions(Some(10_000_000));
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

// ---------------------------------------------------------------------------
// The budgets compose: whichever fires first ends the search.
// ---------------------------------------------------------------------------

#[test]
fn conflict_and_decision_budgets_compose() {
    let mut solver = pigeonhole(11, 10);
    solver.set_max_conflicts(Some(10_000_000));
    solver.set_max_decisions(Some(1));
    assert_eq!(solver.solve(), SolverResult::Unknown);
    assert!(solver.stats().decisions <= 2);
}
