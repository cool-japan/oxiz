//! Issue #36: bounded variable elimination must be reachable from this crate.
//!
//! `oxiz_sat::SolverConfig::enable_bve` defaults to `false` and stayed `false`
//! in every one of the nine SAT-level presets, and `Solver::with_config` never
//! mapped anything onto it — so the SAT engine's BVE pass (which carries model
//! reconstruction and its own tests) could not run from an SMT solve at any
//! setting. These tests pin the pass-through and the behavior it must not
//! change.

use oxiz_core::ast::TermManager;
use oxiz_solver::{Solver, SolverConfig, SolverResult};

#[test]
fn test_issue36_enable_bve_is_split_across_the_presets() {
    assert!(
        !SolverConfig::fast().enable_bve,
        "fast() skips preprocessing entirely"
    );
    assert!(
        !SolverConfig::balanced().enable_bve,
        "balanced() is the default and must not surprise a caller who mixes \
         check_sat_only with later incremental assertions"
    );
    assert!(
        !SolverConfig::minimal().enable_bve,
        "minimal() disables essentially everything"
    );
    assert!(
        SolverConfig::thorough().enable_bve,
        "thorough() is the preset that trades solve-shape surprises for \
         preprocessing power"
    );
    assert!(
        SolverConfig::default().enable_bve == SolverConfig::balanced().enable_bve,
        "the default configuration is balanced()"
    );
}

/// `enable_bve` participates in `SolverConfig`'s equality, which the repeated-
/// `check` verdict cache compares to decide whether an earlier answer still
/// applies. Two configurations differing only in this field must not compare
/// equal, or a `thorough` verdict could be served to a `balanced` query.
#[test]
fn test_issue36_enable_bve_participates_in_config_equality() {
    let base = SolverConfig::balanced();
    let flipped = SolverConfig {
        enable_bve: !base.enable_bve,
        ..base.clone()
    };
    assert_ne!(base, flipped);
}

/// Boolean-only formulas exercised through both presets. `thorough()` now runs
/// BVE on the pure-SAT path; the verdicts must be identical to `balanced()`'s.
#[test]
fn test_issue36_thorough_and_balanced_agree_on_boolean_formulas() {
    // (a | b) & (!a | c) & (!b | c) & (!c | a | b), plus an unsatisfiable
    // variant that adds !c.
    let build = |solver: &mut Solver, manager: &mut TermManager, unsat: bool| {
        let bool_sort = manager.sorts.bool_sort;
        let a = manager.mk_var("a", bool_sort);
        let b = manager.mk_var("b", bool_sort);
        let c = manager.mk_var("c", bool_sort);
        let not_a = manager.mk_not(a);
        let not_b = manager.mk_not(b);
        let not_c = manager.mk_not(c);

        let cl0 = manager.mk_or(vec![a, b]);
        let cl1 = manager.mk_or(vec![not_a, c]);
        let cl2 = manager.mk_or(vec![not_b, c]);
        let cl3 = manager.mk_or(vec![not_c, a, b]);
        for clause in [cl0, cl1, cl2, cl3] {
            solver.assert(clause, manager);
        }
        if unsat {
            solver.assert(not_c, manager);
        }
    };

    for (label, unsat, expected) in [
        ("satisfiable", false, SolverResult::Sat),
        ("unsatisfiable", true, SolverResult::Unsat),
    ] {
        for (name, config) in [
            ("balanced", SolverConfig::balanced()),
            ("thorough", SolverConfig::thorough()),
        ] {
            let mut manager = TermManager::new();
            let mut solver = Solver::with_config(config);
            build(&mut solver, &mut manager, unsat);
            assert_eq!(
                solver.check(&mut manager),
                expected,
                "{label}: {name} disagreed"
            );
        }
    }
}

/// The same formulas through `check_sat_only`, which is the entry point that
/// actually reaches `oxiz_sat::Solver::solve` — and therefore the only one
/// where the newly reachable BVE pass runs at all. (The CDCL(T) route used by
/// `check` goes through `solve_with_theory`, which never invokes it.)
#[test]
fn test_issue36_check_sat_only_agrees_across_presets() {
    for (label, negate_last, expected) in [
        ("satisfiable", false, SolverResult::Sat),
        ("unsatisfiable", true, SolverResult::Unsat),
    ] {
        for (name, config) in [
            ("balanced", SolverConfig::balanced()),
            ("thorough", SolverConfig::thorough()),
        ] {
            let mut manager = TermManager::new();
            let mut solver = Solver::with_config(config);

            // A chain of implications p0 -> p1 -> ... -> p7 together with p0,
            // so every pi is forced true; negating p7 makes it unsatisfiable.
            let bool_sort = manager.sorts.bool_sort;
            let vars: Vec<_> = (0..8)
                .map(|i| manager.mk_var(&format!("p{i}"), bool_sort))
                .collect();
            solver.assert(vars[0], &mut manager);
            for window in vars.windows(2) {
                let implication = manager.mk_implies(window[0], window[1]);
                solver.assert(implication, &mut manager);
            }
            if negate_last {
                let last = vars[7];
                let negated = manager.mk_not(last);
                solver.assert(negated, &mut manager);
            }

            assert_eq!(
                solver.check_sat_only(&mut manager),
                expected,
                "{label}: {name} disagreed on check_sat_only"
            );
        }
    }
}

/// Self-subsuming resolution is on by default in the SAT engine and is reached
/// through this crate's `enable_inprocessing` pass-through. Nothing about it
/// may change an SMT verdict.
#[test]
fn test_issue36_self_subsumption_does_not_disturb_smt_solving() {
    let mut manager = TermManager::new();
    let mut solver = Solver::with_config(SolverConfig {
        enable_inprocessing: true,
        inprocessing_interval: 1,
        ..SolverConfig::balanced()
    });

    let int_sort = manager.sorts.int_sort;
    let x = manager.mk_var("x", int_sort);
    let y = manager.mk_var("y", int_sort);
    let one = manager.mk_int(1);
    let x_plus_one = manager.mk_add(vec![x, one]);
    let lhs = manager.mk_le(x_plus_one, y);
    let rhs = manager.mk_lt(y, x);
    solver.assert(lhs, &mut manager);
    solver.assert(rhs, &mut manager);

    assert_eq!(solver.check(&mut manager), SolverResult::Unsat);
}
