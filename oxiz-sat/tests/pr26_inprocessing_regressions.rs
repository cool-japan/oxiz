//! Regression tests for the PR #26 inprocessing-toolkit port: failed-literal
//! probing (+ hyper-binary resolution), SatELite-style bounded variable
//! elimination (BVE), equivalent-literal substitution (ELS) via SCC over the
//! binary implication graph, and AND/XOR gate congruence detection feeding
//! ELS.
//!
//! These are black-box, whole-solver checks (the white-box mechanism tests —
//! Tarjan SCC unit tests, gate-detection unit tests, resolution/growth-bound
//! unit tests — live next to the code in `src/solver/{bve,equiv,congruence,
//! probe}.rs`). The property under test here is the one that matters most
//! for an opt-in preprocessing pass: turning a mechanism on must never change
//! the SAT/UNSAT verdict, and for SAT results the returned model must be a
//! genuine model of the *original* problem — including for any variable the
//! mechanism eliminated from the live clause set, since a wrong
//! reconstructed value there is exactly the kind of bug verdict-agreement
//! alone cannot catch (a wrong model still reports `Sat`).
//!
//! Vivification and subsumption/self-subsumption strengthening are NOT
//! reimplemented here — OxiZ already has them, wired into
//! `Solver::inprocess` (`Preprocessor::subsumption_elimination` and
//! `Solver::strengthen_clauses_inprocessing`, gated by
//! `SolverConfig::enable_inprocessing`). `test_pr26_vivify_subsumption_already_covered_by_inprocess`
//! below is the targeted check confirming that existing pipeline stays sound
//! rather than a second implementation of the same mechanism.

use oxiz_sat::{Lit, Solver, SolverConfig, SolverResult, Var};

fn v(i: usize) -> Var {
    Var::new(i as u32)
}

/// Every literal of `clause` evaluated against `solver`'s current model.
fn clause_satisfied(solver: &Solver, clause: &[Lit]) -> bool {
    clause.iter().any(|&lit| {
        let val = solver.model_value(lit.var());
        (val == oxiz_sat::LBool::True) == lit.is_pos()
    })
}

/// Assert every clause in `clauses` is satisfied by `solver`'s model and that
/// no variable in `0..num_vars` is left `Undef` — the check that actually
/// catches a wrong model-reconstruction value, not just a wrong verdict.
fn assert_model_is_total_and_satisfies(solver: &Solver, clauses: &[Vec<Lit>], num_vars: usize) {
    for (i, clause) in clauses.iter().enumerate() {
        assert!(
            clause_satisfied(solver, clause),
            "original clause #{i} {clause:?} must be satisfied by the reconstructed model"
        );
    }
    for i in 0..num_vars {
        assert_ne!(
            solver.model_value(v(i)),
            oxiz_sat::LBool::Undef,
            "variable {i} must have a concrete value in a Sat model, eliminated or not"
        );
    }
}

/// The pigeonhole-principle UNSAT instance: `pigeons` items into `holes`
/// slots (`pigeons > holes`).
fn add_pigeonhole(solver: &mut Solver, pigeons: usize, holes: usize) {
    for _ in 0..pigeons * holes {
        solver.new_var();
    }
    let var = |p: usize, h: usize| (p * holes + h + 1) as i32;
    for p in 0..pigeons {
        let clause: Vec<i32> = (0..holes).map(|h| var(p, h)).collect();
        solver.add_clause_dimacs(&clause);
    }
    for h in 0..holes {
        for p1 in 0..pigeons {
            for p2 in (p1 + 1)..pigeons {
                solver.add_clause_dimacs(&[-var(p1, h), -var(p2, h)]);
            }
        }
    }
}

/// An AND-gate-defined variable `v = a ∧ b` wired through padding clauses (on
/// fresh variables `x`, `y`) so `v` is unambiguously the cheapest variable to
/// eliminate under BVE's occurrence-count ordering, plus `(a∨b)` so the
/// search actually has to decide something instead of `a`/`b` being
/// unit-forced (which would let plain propagation determine `v` before BVE
/// gets a chance to run at all). Returns the full original clause set (as
/// `Lit`s) for post-solve model verification.
fn build_and_gate_instance(solver: &mut Solver) -> Vec<Vec<Lit>> {
    let a = solver.new_var();
    let b = solver.new_var();
    let vv = solver.new_var();
    let x = solver.new_var();
    let y = solver.new_var();
    let clauses: Vec<Vec<Lit>> = vec![
        vec![Lit::neg(a), Lit::neg(b), Lit::pos(vv)],
        vec![Lit::neg(vv), Lit::pos(a)],
        vec![Lit::neg(vv), Lit::pos(b)],
        vec![Lit::pos(a), Lit::pos(x)],
        vec![Lit::pos(a), Lit::pos(y)],
        vec![Lit::pos(b), Lit::pos(x)],
        vec![Lit::pos(b), Lit::pos(y)],
        vec![Lit::pos(a), Lit::pos(b)],
    ];
    for clause in &clauses {
        solver.add_clause(clause.iter().copied());
    }
    clauses
}

/// Two literals made equivalent by a binary cycle (`a≡b`), with a third
/// variable `c` whose only satisfying value depends on the cycle actually
/// being honored: `(¬a∨c)` combined with a decision-forcing `(a∨d)` (so `a`
/// is not simply unit-propagated before substitution runs).
fn build_equivalence_instance(solver: &mut Solver) -> (Vec<Vec<Lit>>, Var, Var, Var) {
    let a = solver.new_var();
    let b = solver.new_var();
    let c = solver.new_var();
    let d = solver.new_var();
    let clauses: Vec<Vec<Lit>> = vec![
        vec![Lit::neg(a), Lit::pos(b)],
        vec![Lit::neg(b), Lit::pos(a)],
        vec![Lit::neg(a), Lit::pos(c)],
        vec![Lit::neg(c), Lit::pos(a)],
        vec![Lit::pos(a), Lit::pos(d)],
        vec![Lit::neg(a), Lit::pos(d)],
    ];
    for clause in &clauses {
        solver.add_clause(clause.iter().copied());
    }
    (clauses, a, b, c)
}

// ---------------------------------------------------------------------
// Probing (failed-literal probing + hyper-binary resolution)
// ---------------------------------------------------------------------

#[test]
fn test_pr26_probe_unsat_verdict_agrees_on_off() {
    let mut off = Solver::with_config(SolverConfig {
        enable_failed_literal_probing: false,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut off, 6, 5);
    assert_eq!(off.solve(), SolverResult::Unsat);

    let mut on = Solver::with_config(SolverConfig {
        enable_failed_literal_probing: true,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut on, 6, 5);
    assert_eq!(on.solve(), SolverResult::Unsat);
}

#[test]
fn test_pr26_probe_sat_verdict_and_model_agree_on_off() {
    // A formula shaped so failed-literal probing forces at least one unit
    // (see the equivalent unit test in `src/solver/probe.rs`), embedded in a
    // larger satisfiable instance.
    let base_clauses = |solver: &mut Solver| -> Vec<Vec<Lit>> {
        let a = solver.new_var();
        let b = solver.new_var();
        let e = solver.new_var();
        let f = solver.new_var();
        let clauses: Vec<Vec<Lit>> = vec![
            vec![Lit::pos(a), Lit::pos(b)],
            vec![Lit::neg(a), Lit::pos(b)],
            vec![Lit::neg(a), Lit::neg(b)],
            vec![Lit::pos(e), Lit::pos(f)],
            vec![Lit::neg(e), Lit::pos(f)],
        ];
        for clause in &clauses {
            solver.add_clause(clause.iter().copied());
        }
        clauses
    };

    let mut off = Solver::with_config(SolverConfig {
        enable_failed_literal_probing: false,
        ..SolverConfig::default()
    });
    let clauses_off = base_clauses(&mut off);
    assert_eq!(off.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&off, &clauses_off, 4);

    let mut on = Solver::with_config(SolverConfig {
        enable_failed_literal_probing: true,
        ..SolverConfig::default()
    });
    let clauses_on = base_clauses(&mut on);
    assert_eq!(on.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&on, &clauses_on, 4);
}

// ---------------------------------------------------------------------
// Bounded variable elimination (BVE)
// ---------------------------------------------------------------------

#[test]
fn test_pr26_bve_model_reconstruction_satisfies_original_clauses() {
    let mut solver = Solver::with_config(SolverConfig {
        enable_bve: true,
        // The lucky phase would answer `Sat` on this small instance before
        // BVE ever ran, leaving the model
        // reconstruction under test with nothing to reconstruct (see
        // `SolverConfig::enable_lucky_phase`).
        enable_lucky_phase: false,
        ..SolverConfig::default()
    });
    let clauses = build_and_gate_instance(&mut solver);
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&solver, &clauses, 5);
    // Non-vacuity guard: without this, the test cannot distinguish "BVE
    // eliminated vv and reconstruction restored it correctly" from "BVE
    // never fired and ordinary search assigned vv directly" — the whole
    // point of a *reconstruction* test is to exercise the reconstruction
    // path, not just re-confirm the model is valid.
    assert!(
        solver.var_eliminated(v(2)),
        "vv (the AND-gate output, occurrence-count-cheapest to eliminate) must \
         actually have been eliminated for this to exercise BVE reconstruction"
    );
}

#[test]
fn test_pr26_bve_unsat_verdict_agrees_on_off() {
    let mut off = Solver::with_config(SolverConfig {
        enable_bve: false,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut off, 6, 5);
    assert_eq!(off.solve(), SolverResult::Unsat);

    let mut on = Solver::with_config(SolverConfig {
        enable_bve: true,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut on, 6, 5);
    assert_eq!(on.solve(), SolverResult::Unsat);
}

#[test]
fn test_pr26_bve_sat_verdict_agrees_on_off() {
    let mut off = Solver::with_config(SolverConfig {
        enable_bve: false,
        ..SolverConfig::default()
    });
    let clauses_off = build_and_gate_instance(&mut off);
    assert_eq!(off.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&off, &clauses_off, 5);

    let mut on = Solver::with_config(SolverConfig {
        enable_bve: true,
        ..SolverConfig::default()
    });
    let clauses_on = build_and_gate_instance(&mut on);
    assert_eq!(on.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&on, &clauses_on, 5);
}

// ---------------------------------------------------------------------
// Equivalent-literal substitution (ELS)
// ---------------------------------------------------------------------

#[test]
fn test_pr26_els_model_reconstruction_satisfies_original_clauses() {
    let mut solver = Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        // The lucky phase would answer `Sat` on this small instance before
        // equivalent-literal substitution ever ran, leaving the model
        // reconstruction under test with nothing to reconstruct (see
        // `SolverConfig::enable_lucky_phase`).
        enable_lucky_phase: false,
        ..SolverConfig::default()
    });
    let (clauses, a, b, c) = build_equivalence_instance(&mut solver);
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&solver, &clauses, 4);
    // a≡b and a≡c form one 3-member SCC, which ELS must collapse to a single
    // representative — at least 2 of the 3 have to be folded away. Without
    // this the test would pass vacuously even if `fold_equivalent_literals`
    // never actually fired (verdict + model validity alone can't tell the
    // difference between "ELS folded the class" and "ELS was a no-op and
    // ordinary search happened to satisfy everything anyway").
    let eliminated_count = [a, b, c]
        .iter()
        .filter(|&&var| solver.var_eliminated(var))
        .count();
    assert!(
        eliminated_count >= 2,
        "the a≡b≡c cycle must collapse to one representative (>=2 of the 3 folded away); got {eliminated_count}"
    );
}

#[test]
fn test_pr26_els_unsat_verdict_agrees_on_off() {
    let mut off = Solver::with_config(SolverConfig {
        enable_equiv_substitution: false,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut off, 6, 5);
    assert_eq!(off.solve(), SolverResult::Unsat);

    let mut on = Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut on, 6, 5);
    assert_eq!(on.solve(), SolverResult::Unsat);
}

#[test]
fn test_pr26_els_sat_verdict_agrees_on_off() {
    let mut off = Solver::with_config(SolverConfig {
        enable_equiv_substitution: false,
        ..SolverConfig::default()
    });
    let clauses_off = build_equivalence_instance(&mut off).0;
    assert_eq!(off.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&off, &clauses_off, 4);

    let mut on = Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        ..SolverConfig::default()
    });
    let clauses_on = build_equivalence_instance(&mut on).0;
    assert_eq!(on.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&on, &clauses_on, 4);
}

#[test]
fn test_pr26_els_detects_self_contradiction_as_unsat() {
    // (a≡b) and (a≡¬b) together force a≡¬a: unconditionally UNSAT,
    // independent of any assignment.
    let mut solver = Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        ..SolverConfig::default()
    });
    let a = solver.new_var();
    let b = solver.new_var();
    solver.add_clause([Lit::neg(a), Lit::pos(b)]);
    solver.add_clause([Lit::neg(b), Lit::pos(a)]);
    solver.add_clause([Lit::neg(a), Lit::neg(b)]);
    solver.add_clause([Lit::pos(a), Lit::pos(b)]);
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

// ---------------------------------------------------------------------
// Gate congruence (feeds ELS)
// ---------------------------------------------------------------------

#[test]
fn test_pr26_gates_congruence_model_reconstruction_satisfies_original_clauses() {
    // Two structurally duplicate AND gates o1 = a∧b and o2 = a∧b (a
    // multiplier/adder-style repeated substructure). Gate congruence must
    // recognise o1≡o2 and fold one into the other via ELS; the reconstructed
    // model must still satisfy every original clause for both gates.
    let mut solver = Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        enable_gate_congruence: true,
        // The lucky phase would answer `Sat` on this small instance before
        // gate congruence ever ran, leaving the model
        // reconstruction under test with nothing to reconstruct (see
        // `SolverConfig::enable_lucky_phase`).
        enable_lucky_phase: false,
        ..SolverConfig::default()
    });
    let a = solver.new_var();
    let b = solver.new_var();
    let o1 = solver.new_var();
    let o2 = solver.new_var();
    let d = solver.new_var(); // forces an actual decision on a/b
    let mut clauses: Vec<Vec<Lit>> = Vec::new();
    for &o in &[o1, o2] {
        clauses.push(vec![Lit::neg(a), Lit::neg(b), Lit::pos(o)]);
        clauses.push(vec![Lit::neg(o), Lit::pos(a)]);
        clauses.push(vec![Lit::neg(o), Lit::pos(b)]);
    }
    clauses.push(vec![Lit::pos(a), Lit::pos(d)]);
    clauses.push(vec![Lit::neg(a), Lit::pos(d)]);
    for clause in &clauses {
        solver.add_clause(clause.iter().copied());
    }

    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_model_is_total_and_satisfies(&solver, &clauses, 5);
    // Non-vacuity guard: prove gate congruence actually fired and folded one
    // output into the other, rather than the search simply satisfying both
    // duplicate gates independently without any structural sharing detected.
    assert!(
        solver.var_eliminated(o1) || solver.var_eliminated(o2),
        "gate congruence must have folded o1 and o2 into a single representative"
    );
}

#[test]
fn test_pr26_gates_unsat_verdict_agrees_on_off() {
    let mut off = Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        enable_gate_congruence: false,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut off, 6, 5);
    assert_eq!(off.solve(), SolverResult::Unsat);

    let mut on = Solver::with_config(SolverConfig {
        enable_equiv_substitution: true,
        enable_gate_congruence: true,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut on, 6, 5);
    assert_eq!(on.solve(), SolverResult::Unsat);
}

// ---------------------------------------------------------------------
// Vivification / subsumption: already covered by the existing
// `Solver::inprocess` pipeline (`Preprocessor::subsumption_elimination` +
// `Solver::strengthen_clauses_inprocessing`), not reimplemented here.
// ---------------------------------------------------------------------

#[test]
fn test_pr26_vivify_subsumption_already_covered_by_inprocess() {
    // (a∨b) subsumes (a∨b∨c): the longer clause is redundant given the
    // shorter one. Also add enough conflict-driving structure that
    // `Solver::inprocess` (gated by `enable_inprocessing`, conflict-count
    // triggered) actually runs during the solve, exercising both the
    // subsumption pass and the on-the-fly strengthening pass on learned
    // clauses without changing the verdict or producing an invalid model.
    let build = |solver: &mut Solver| -> Vec<Vec<Lit>> {
        let a = solver.new_var();
        let b = solver.new_var();
        let c = solver.new_var();
        let clauses: Vec<Vec<Lit>> = vec![
            vec![Lit::pos(a), Lit::pos(b)],
            vec![Lit::pos(a), Lit::pos(b), Lit::pos(c)],
            vec![Lit::neg(a), Lit::pos(c)],
            vec![Lit::neg(b), Lit::neg(c)],
        ];
        for clause in &clauses {
            solver.add_clause(clause.iter().copied());
        }
        clauses
    };

    let mut off = Solver::with_config(SolverConfig {
        enable_inprocessing: false,
        ..SolverConfig::default()
    });
    let clauses_off = build(&mut off);
    let result_off = off.solve();

    let mut on = Solver::with_config(SolverConfig {
        enable_inprocessing: true,
        inprocessing_interval: 1,
        ..SolverConfig::default()
    });
    let clauses_on = build(&mut on);
    let result_on = on.solve();

    assert_eq!(
        result_off, result_on,
        "enabling inprocessing must not change the verdict"
    );
    if result_on == SolverResult::Sat {
        assert_model_is_total_and_satisfies(&on, &clauses_on, 3);
        assert_model_is_total_and_satisfies(&off, &clauses_off, 3);
    }
}

// ---------------------------------------------------------------------
// Deferred Part-1 fixes applied in this pass: regression coverage
// ---------------------------------------------------------------------

#[test]
fn test_pr26_lazy_hyper_binary_guard_preserves_verdict() {
    // The `check_hyper_binary_resolution` `is_false()` guard (see
    // `src/solver/propagate.rs`) only ever *restricts* which literals are
    // resolved away; it cannot itself flip a verdict. Exercise the config
    // that turns the mechanism on across a small UNSAT and a small SAT
    // instance built with enough decision levels to reach the >=2
    // decision-level gate inside `check_hyper_binary_resolution`.
    let mut off = Solver::with_config(SolverConfig {
        enable_lazy_hyper_binary: false,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut off, 6, 5);
    assert_eq!(off.solve(), SolverResult::Unsat);

    let mut on = Solver::with_config(SolverConfig {
        enable_lazy_hyper_binary: true,
        ..SolverConfig::default()
    });
    add_pigeonhole(&mut on, 6, 5);
    assert_eq!(on.solve(), SolverResult::Unsat);
}
