//! Regression tests for the gatekeeper review of the PR #26 port (search
//! core, inprocessing toolkit, LRAT proofs — see the three `pr26_*`
//! regression files alongside this one).
//!
//! SK-1 (blocking, soundness): a variable the one-shot inprocessing toolkit
//! already eliminated from the live formula — via equivalent-literal
//! substitution (ELS) or bounded variable elimination (BVE) — could be
//! *reintroduced* by a later `add_clause` or a `solve_with_assumptions`
//! assumption with no guard at all. `pick_branch_var` skips eliminated
//! variables (see `solver/decide.rs`), so nothing ever assigns them during
//! search, and `save_model`'s reconstruction pass unconditionally overwrites
//! whatever the new clause/assumption demanded. Confirmed repro: `a ≡ b`
//! under `enable_equiv_substitution`; `solve()` folds one into the other;
//! then adding `¬a` and a unit `b` (jointly unconditionally UNSAT, since
//! `a≡b` forces `a == b` in every model) made a second `solve()` wrongly
//! report `Sat`. Same false verdict via
//! `solve_with_assumptions(&[¬a, b])`.
//!
//! Fixed by `Solver::resolve_reintroduced_literal` (see `solver/equiv.rs`):
//! an ELS-substituted literal is rewritten through the substitution map
//! (free and sound — the equivalence was already proven); a BVE-eliminated
//! one has no such cheap rewrite available (its defining clauses are gone,
//! not just renamed), so it poisons the solver with a
//! `SolverError::EliminatedVariableReintroduction` instead — every `solve*`
//! entry point then refuses to answer `Sat`/`Unsat` (see `Solver::error`).

use oxiz_sat::{Lit, Solver, SolverConfig, SolverError, SolverResult, Var};

fn els_enabled_solver() -> Solver {
    Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        // The lucky phase would answer `Sat` on this small instance
        // before equivalent-literal substitution ever ran, leaving the mechanism
        // under test with nothing to show (see
        // `SolverConfig::enable_lucky_phase`).
        enable_lucky_phase: false,
        ..SolverConfig::default()
    })
}

fn bve_enabled_solver() -> Solver {
    Solver::with_config(SolverConfig {
        enable_bve: true,
        // The lucky phase would answer `Sat` on this small instance
        // before BVE ever ran, leaving the mechanism
        // under test with nothing to show (see
        // `SolverConfig::enable_lucky_phase`).
        enable_lucky_phase: false,
        ..SolverConfig::default()
    })
}

/// Wire up `a ≡ b` (via `(¬a∨b)∧(¬b∨a)`) and run the first `solve()`, which
/// (with ELS enabled) folds one variable into the other. Returns the two
/// variables; the caller does not need to know which one ended up as the
/// canonical representative — the fix is symmetric in that choice.
fn solve_and_fold_equivalence(solver: &mut Solver) -> (Lit, Lit) {
    let a = solver.new_var();
    let b = solver.new_var();
    solver.add_clause([Lit::neg(a), Lit::pos(b)]);
    solver.add_clause([Lit::neg(b), Lit::pos(a)]);
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert!(
        solver.var_eliminated(a) || solver.var_eliminated(b),
        "one of the two equivalent variables must be folded into the other"
    );
    (Lit::pos(a), Lit::pos(b))
}

/// Wire up an AND-gate-style formula BVE is known to eliminate the gate
/// variable `v` from (mirrors `bve.rs`'s own
/// `test_pr26_bve_eliminates_and_gate_style_variable`, including the
/// occurrence-count padding via `x`/`y` that makes `v` unambiguously the
/// cheapest variable to eliminate first). Returns `v`.
fn solve_and_eliminate_bve_variable(solver: &mut Solver) -> Var {
    let a = solver.new_var();
    let b = solver.new_var();
    let v = solver.new_var();
    let x = solver.new_var();
    let y = solver.new_var();
    solver.add_clause([Lit::neg(a), Lit::neg(b), Lit::pos(v)]);
    solver.add_clause([Lit::neg(v), Lit::pos(a)]);
    solver.add_clause([Lit::neg(v), Lit::pos(b)]);
    solver.add_clause([Lit::pos(a), Lit::pos(x)]);
    solver.add_clause([Lit::pos(a), Lit::pos(y)]);
    solver.add_clause([Lit::pos(b), Lit::pos(x)]);
    solver.add_clause([Lit::pos(b), Lit::pos(y)]);
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert!(
        solver.var_eliminated(v),
        "v must actually have been eliminated for this test to exercise anything"
    );
    v
}

#[test]
fn test_pr26_gatekeeper_sk1_els_add_clause_reintroduction_is_rewritten_soundly() {
    let mut solver = els_enabled_solver();
    let (a, b) = solve_and_fold_equivalence(&mut solver);

    // ¬a and b together with a≡b is unconditionally UNSAT: a≡b forces a==b
    // in every model, so ¬a implies ¬b, contradicting the unit clause b. The
    // second `add_clause` may catch this immediately (its own trivially_unsat
    // fast path, once b is rewritten to a's already-fixed representative) or
    // defer to `solve()` — either is correct; what must never happen is a
    // later `solve()` reporting `Sat`.
    assert!(solver.add_clause([a.negate()]));
    let _ = solver.add_clause([b]);

    assert_eq!(
        solver.solve(),
        SolverResult::Unsat,
        "a later add_clause naming an ELS-substituted variable must still be \
         checked against the equivalence, not silently ignored"
    );
    assert!(
        solver.error().is_none(),
        "ELS reintroduction has a sound rewrite; it must never poison the solver"
    );
}

#[test]
fn test_pr26_gatekeeper_sk1_els_add_clause_reintroduction_still_finds_sat_models() {
    // Positive control for the fix above: the rewrite must not turn every
    // reintroduction into a spurious UNSAT either. `b ∨ c` (c fresh) is
    // satisfiable together with a≡b; the reconstructed model must actually
    // satisfy it once translated back through b's substitution.
    let mut solver = els_enabled_solver();
    let (_a, b) = solve_and_fold_equivalence(&mut solver);
    let c = solver.new_var();
    assert!(solver.add_clause([b, Lit::pos(c)]));

    assert_eq!(solver.solve(), SolverResult::Sat);
    let b_true = solver.model_value(b.var()) == oxiz_sat::LBool::True;
    let c_true = solver.model_value(c) == oxiz_sat::LBool::True;
    assert!(
        b_true || c_true,
        "reconstructed model must satisfy the clause that reintroduced b"
    );
}

#[test]
fn test_pr26_gatekeeper_sk1_els_assumptions_reintroduction_is_rewritten_soundly() {
    let mut solver = els_enabled_solver();
    let (a, b) = solve_and_fold_equivalence(&mut solver);

    let assumptions = [a.negate(), b];
    let (result, core) = solver.solve_with_assumptions(&assumptions);
    assert_eq!(
        result,
        SolverResult::Unsat,
        "¬a and b together with a≡b is unconditionally UNSAT under assumptions too"
    );
    let core = core.expect("Unsat must carry a core");
    assert!(!core.is_empty());
    assert!(
        core.iter().all(|c| assumptions.contains(c)),
        "the returned core must be a genuine subset of the literals the \
         caller actually passed in, even though `b` was internally rewritten \
         to a's class representative before being decided on: got core \
         {core:?}, assumptions {assumptions:?}"
    );
    assert!(solver.error().is_none());
}

#[test]
fn test_pr26_gatekeeper_sk1_bve_add_clause_reintroduction_is_a_hard_error() {
    let mut solver = bve_enabled_solver();
    let v = solve_and_eliminate_bve_variable(&mut solver);
    assert!(solver.error().is_none());

    let added = solver.add_clause([Lit::pos(v)]);
    assert!(
        !added,
        "add_clause must refuse a clause naming a BVE-eliminated variable"
    );
    match solver.error() {
        Some(SolverError::EliminatedVariableReintroduction { var }) => {
            assert_eq!(*var, v);
        }
        other => panic!("expected EliminatedVariableReintroduction, got {other:?}"),
    }

    assert_eq!(
        solver.solve(),
        SolverResult::Unknown,
        "a poisoned solver must never answer Sat or Unsat -- either could be wrong"
    );
}

#[test]
fn test_pr26_gatekeeper_sk1_bve_assumptions_reintroduction_is_a_hard_error() {
    let mut solver = bve_enabled_solver();
    let v = solve_and_eliminate_bve_variable(&mut solver);

    let (result, core) = solver.solve_with_assumptions(&[Lit::pos(v)]);
    assert_eq!(
        result,
        SolverResult::Unknown,
        "assuming a BVE-eliminated variable must not answer Sat or Unsat"
    );
    assert!(core.is_none());
    match solver.error() {
        Some(SolverError::EliminatedVariableReintroduction { var }) => {
            assert_eq!(*var, v);
        }
        other => panic!("expected EliminatedVariableReintroduction, got {other:?}"),
    }
}

#[test]
fn test_pr26_gatekeeper_sk1_error_is_cleared_by_reset() {
    let mut solver = bve_enabled_solver();
    let v = solve_and_eliminate_bve_variable(&mut solver);
    assert!(!solver.add_clause([Lit::pos(v)]));
    assert!(solver.error().is_some());

    solver.reset();
    assert!(
        solver.error().is_none(),
        "reset() must clear a fatal error along with everything else per-incarnation"
    );
}
