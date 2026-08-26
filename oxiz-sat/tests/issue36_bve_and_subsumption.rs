//! Issue #36 regressions: bounded variable elimination reachability, and the
//! quality of the subsumption / self-subsumption passes it feeds.
//!
//! Before this work, `SolverConfig::enable_bve` was `false` in `Default` and
//! in all nine `ConfigPreset` bodies and was not mapped from oxiz-solver's
//! configuration either, so `Solver::bounded_variable_elimination` — a pass
//! that carries full model reconstruction and its own tests — never ran in any
//! shipped configuration. These tests pin the two things that must hold once
//! it does run: every preset must agree with `Default` on the verdict, and a
//! `Sat` model must satisfy the *original* clauses, not merely the reduced
//! ones BVE left behind.

use oxiz_sat::{ConfigPreset, LBool, Lit, Solver, SolverConfig, SolverResult, Var};

fn pos(v: u32) -> Lit {
    Lit::pos(Var::new(v))
}

fn neg(v: u32) -> Lit {
    Lit::neg(Var::new(v))
}

/// Deterministic LCG so the randomized cases are reproducible without pulling
/// in an RNG dependency.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A satisfiable instance with plenty of low-degree definitional variables —
/// exactly the shape bounded variable elimination is meant to fold away. It
/// encodes a small chain of AND gates plus a few side constraints.
fn sat_instance() -> Vec<Vec<Lit>> {
    let mut clauses = Vec::new();
    // g_i <-> (x_i AND x_{i+1}) for i in 0..6, with x_j = var j and g_i = var 10+i.
    for i in 0..6u32 {
        let g = 10 + i;
        clauses.push(vec![neg(g), pos(i)]);
        clauses.push(vec![neg(g), pos(i + 1)]);
        clauses.push(vec![pos(g), neg(i), neg(i + 1)]);
    }
    // At least one gate fires, and two of the inputs are constrained.
    clauses.push(vec![pos(10), pos(11), pos(12), pos(13), pos(14), pos(15)]);
    clauses.push(vec![pos(0), pos(2)]);
    clauses.push(vec![neg(3), pos(5), pos(6)]);
    clauses.push(vec![pos(4), neg(6), pos(1)]);
    clauses
}

/// An unsatisfiable instance: pigeonhole 4-into-3, `p[i][j]` = pigeon `i` in
/// hole `j` at variable `i * 3 + j`.
fn unsat_instance() -> Vec<Vec<Lit>> {
    let var = |i: u32, j: u32| i * 3 + j;
    let mut clauses = Vec::new();
    for i in 0..4u32 {
        clauses.push((0..3).map(|j| pos(var(i, j))).collect());
    }
    for j in 0..3u32 {
        for i in 0..4u32 {
            for k in (i + 1)..4u32 {
                clauses.push(vec![neg(var(i, j)), neg(var(k, j))]);
            }
        }
    }
    clauses
}

fn load(solver: &mut Solver, clauses: &[Vec<Lit>]) {
    let max_var = clauses
        .iter()
        .flatten()
        .map(|lit| lit.var().index() + 1)
        .max()
        .unwrap_or(0);
    solver.ensure_vars(max_var);
    for clause in clauses {
        let _ = solver.add_clause(clause.iter().copied());
    }
}

/// How many variables did the inprocessing toolkit actually remove?
///
/// Used to keep the BVE tests honest: a test that passes because the pass
/// never fired proves nothing about the pass.
fn eliminated_var_count(solver: &Solver) -> usize {
    (0..solver.num_vars())
        .filter(|&i| solver.var_eliminated(Var::new(i as u32)))
        .count()
}

/// Does the reported model satisfy every clause of the ORIGINAL formula?
///
/// This is the property model reconstruction exists for: BVE deletes the
/// clauses that defined an eliminated variable, so a solver that forgot to
/// reconstruct that variable's value would still report `Sat` while handing
/// back an assignment falsifying a clause the caller asserted.
fn model_satisfies(solver: &Solver, clauses: &[Vec<Lit>]) -> Result<(), String> {
    for (idx, clause) in clauses.iter().enumerate() {
        let satisfied = clause.iter().any(|lit| {
            let value = solver.model_value(lit.var());
            match value {
                LBool::True => !lit.is_neg(),
                LBool::False => lit.is_neg(),
                LBool::Undef => false,
            }
        });
        if !satisfied {
            return Err(format!("original clause {idx} ({clause:?}) is falsified"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 1. BVE is reachable, and every preset still agrees with `Default`.
// ---------------------------------------------------------------------------

#[test]
fn test_issue36_bve_is_enabled_in_the_presets_that_claim_it() {
    assert!(
        ConfigPreset::CaDiCaL.config().enable_bve,
        "the CaDiCaL-style preset must run BVE: the tool it imitates enables `elim` by default"
    );
    assert!(
        ConfigPreset::Industrial.config().enable_bve,
        "industrial/structured instances are the class BVE was designed for"
    );
    // BVE defers entirely to equivalent-literal substitution when both are
    // set, so a preset enabling both would make its own `enable_bve` inert.
    for preset in ConfigPreset::all_presets() {
        let config = preset.config();
        assert!(
            !(config.enable_bve && config.enable_equiv_substitution),
            "{preset:?}: enable_bve would be inert alongside enable_equiv_substitution"
        );
    }
}

#[test]
fn test_issue36_every_preset_agrees_with_default_on_the_verdict() {
    for (label, clauses, expected) in [
        ("sat gate chain", sat_instance(), SolverResult::Sat),
        ("pigeonhole 4-into-3", unsat_instance(), SolverResult::Unsat),
    ] {
        // Every solver in this test keeps its preset intact except for the
        // pre-search lucky phase, which is switched off throughout: on this
        // small satisfiable instance it answers `Sat` before BVE ever runs,
        // which would make the `bve_eliminations > 0` assertion below vacuous
        // (see `SolverConfig::enable_lucky_phase`).
        let mut baseline_config = ConfigPreset::Default.config();
        baseline_config.enable_lucky_phase = false;
        let mut baseline = Solver::with_config(baseline_config);
        load(&mut baseline, &clauses);
        let baseline_result = baseline.solve();
        assert_eq!(
            baseline_result, expected,
            "{label}: the Default preset itself must get this right"
        );

        let mut bve_eliminations = 0usize;
        for preset in ConfigPreset::all_presets() {
            let mut config = preset.config();
            config.enable_lucky_phase = false;
            let mut solver = Solver::with_config(config);
            load(&mut solver, &clauses);
            let result = solver.solve();
            assert_eq!(
                result, baseline_result,
                "{label}: preset {preset:?} disagreed with Default"
            );
            if preset.config().enable_bve {
                bve_eliminations += eliminated_var_count(&solver);
            }
            if result == SolverResult::Sat
                && let Err(msg) = model_satisfies(&solver, &clauses)
            {
                panic!("{label}: preset {preset:?} reported Sat but {msg}");
            }
        }
        assert!(
            bve_eliminations > 0,
            "{label}: the BVE-enabled presets must actually eliminate variables \
             here, or the agreement above is vacuous"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. BVE model reconstruction, pinned end to end.
// ---------------------------------------------------------------------------

#[test]
fn test_issue36_bve_sat_model_satisfies_the_original_clauses() {
    let clauses = sat_instance();

    let mut with_bve = Solver::with_config(SolverConfig {
        enable_bve: true,
        // Off, or the lucky phase answers `Sat` on this instance before BVE
        // runs and there is no reconstruction left to test (see
        // `SolverConfig::enable_lucky_phase`).
        enable_lucky_phase: false,
        ..SolverConfig::default()
    });
    load(&mut with_bve, &clauses);
    assert_eq!(with_bve.solve(), SolverResult::Sat);
    assert!(
        eliminated_var_count(&with_bve) > 0,
        "the instance must actually exercise BVE, or this proves nothing about \
         model reconstruction"
    );
    if let Err(msg) = model_satisfies(&with_bve, &clauses) {
        panic!("BVE model reconstruction is incomplete: {msg}");
    }

    // Baseline: the same instance with BVE off eliminates nothing, so the
    // count above is attributable to this switch and not to some other pass.
    let mut without_bve = Solver::with_config(SolverConfig {
        enable_lucky_phase: false,
        ..SolverConfig::default()
    });
    load(&mut without_bve, &clauses);
    assert_eq!(without_bve.solve(), SolverResult::Sat);
    assert_eq!(eliminated_var_count(&without_bve), 0);
}

#[test]
fn test_issue36_bve_agrees_with_plain_search_on_random_instances() {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);

    for round in 0..120u32 {
        let num_vars = 8 + (round % 7);
        let num_clauses = (num_vars as f64 * 4.2) as u32;

        let mut clauses: Vec<Vec<Lit>> = Vec::new();
        for _ in 0..num_clauses {
            let mut clause: Vec<Lit> = Vec::new();
            while clause.len() < 3 {
                let v = rng.below(u64::from(num_vars)) as u32;
                let lit = if rng.below(2) == 0 { pos(v) } else { neg(v) };
                if !clause.iter().any(|&l| l.var() == lit.var()) {
                    clause.push(lit);
                }
            }
            clauses.push(clause);
        }

        let mut plain = Solver::new();
        load(&mut plain, &clauses);
        let expected = plain.solve();

        // Both BVE on its own and the two presets that now enable it.
        let mut configs = vec![(
            "enable_bve".to_string(),
            SolverConfig {
                enable_bve: true,
                ..SolverConfig::default()
            },
        )];
        for preset in [ConfigPreset::CaDiCaL, ConfigPreset::Industrial] {
            configs.push((format!("{preset:?}"), preset.config()));
        }

        for (label, config) in configs {
            let mut solver = Solver::with_config(config);
            load(&mut solver, &clauses);
            let result = solver.solve();
            assert_eq!(
                result, expected,
                "round {round}: {label} disagreed with the plain solver"
            );
            if result == SolverResult::Sat
                && let Err(msg) = model_satisfies(&solver, &clauses)
            {
                panic!("round {round}: {label} reported Sat but {msg}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Self-subsuming resolution, end to end through inprocessing.
// ---------------------------------------------------------------------------

/// A formula containing many `(a ∨ b ∨ c)` / `(a ∨ b ∨ ¬c)` pairs — the
/// canonical self-subsumption shape — bolted onto a pigeonhole core that
/// guarantees a stream of conflicts, so periodic inprocessing actually fires.
/// The pass must remove literals, and the verdict must still be right.
fn strengthenable_instance() -> Vec<Vec<Lit>> {
    // Vars 0..12 are the pigeonhole core; 20.. are the strengthenable block.
    let mut clauses = unsat_instance();
    for i in 0..10u32 {
        let a = 20 + 3 * i;
        let b = 21 + 3 * i;
        let c = 22 + 3 * i;
        clauses.push(vec![pos(a), pos(b), pos(c)]);
        clauses.push(vec![pos(a), pos(b), neg(c)]);
    }
    clauses
}

#[test]
fn test_issue36_self_subsumption_fires_during_inprocessing_and_stays_correct() {
    let clauses = strengthenable_instance();

    let mut solver = Solver::with_config(SolverConfig {
        enable_inprocessing: true,
        inprocessing_interval: 1,
        enable_self_subsumption: true,
        ..SolverConfig::default()
    });
    load(&mut solver, &clauses);
    let result = solver.solve();

    assert_eq!(
        result,
        SolverResult::Unsat,
        "the pigeonhole core makes this instance unsatisfiable"
    );
    assert!(
        solver.stats().self_subsumed_literals > 0,
        "self-subsuming resolution must have removed at least one literal"
    );

    // Same instance with the pass switched off: same verdict, no removals.
    let mut without = Solver::with_config(SolverConfig {
        enable_inprocessing: true,
        inprocessing_interval: 1,
        enable_self_subsumption: false,
        ..SolverConfig::default()
    });
    load(&mut without, &clauses);
    assert_eq!(without.solve(), result);
    assert_eq!(without.stats().self_subsumed_literals, 0);
}

/// The same block of strengthenable pairs on a *satisfiable* instance: the
/// verdict must survive strengthening, and so must the model.
#[test]
fn test_issue36_self_subsumption_preserves_a_satisfiable_model() {
    let mut clauses = sat_instance();
    for i in 0..10u32 {
        let a = 30 + 3 * i;
        let b = 31 + 3 * i;
        let c = 32 + 3 * i;
        clauses.push(vec![pos(a), pos(b), pos(c)]);
        clauses.push(vec![pos(a), pos(b), neg(c)]);
        clauses.push(vec![neg(a), pos(b), pos(c)]);
    }

    let mut solver = Solver::with_config(SolverConfig {
        enable_inprocessing: true,
        inprocessing_interval: 1,
        enable_self_subsumption: true,
        ..SolverConfig::default()
    });
    load(&mut solver, &clauses);
    assert_eq!(solver.solve(), SolverResult::Sat);
    if let Err(msg) = model_satisfies(&solver, &clauses) {
        panic!("self-subsumption broke the model: {msg}");
    }
}

/// Regression for a completeness hole this work surfaced, independent of any
/// of the new passes.
///
/// `Solver::inprocess` runs straight after conflict handling, where the
/// level-0 propagation queue is routinely still non-empty.
/// `strengthen_clauses_inprocessing` (and `vivify_clauses`) then open a probe
/// decision level and call `propagate()`, which drains those pending level-0
/// literals under the probe's assumptions and files their consequences at the
/// probe level — where the following `backtrack` discards them. The genuine
/// level-0 implications among them were never re-derived, leaving live clauses
/// with one unassigned literal and all others false that nothing would ever
/// fire on. Both passes now rewind the propagation head when they finish.
///
/// Reproduces only with a short inprocessing interval (the default 5000
/// conflicts is why this went unnoticed) and is caught by the debug-build
/// fixpoint invariants, so this test carries no assertion of its own beyond
/// running to completion with the right verdicts.
#[test]
fn test_issue36_inprocessing_leaves_no_hanging_unit() {
    let mut rng = Lcg(0x243F_6A88_85A3_08D3);

    for round in 0..200u32 {
        let num_vars = 6 + (round % 6);
        let num_clauses = (num_vars as f64 * 4.4) as u32;

        let mut clauses: Vec<Vec<Lit>> = Vec::new();
        for _ in 0..num_clauses {
            let mut clause: Vec<Lit> = Vec::new();
            while clause.len() < 3 {
                let v = rng.below(u64::from(num_vars)) as u32;
                let lit = if rng.below(2) == 0 { pos(v) } else { neg(v) };
                if !clause.iter().any(|&l| l.var() == lit.var()) {
                    clause.push(lit);
                }
            }
            clauses.push(clause);
        }

        // Baseline: inprocessing off entirely, so none of the probe passes run.
        let mut plain = Solver::new();
        load(&mut plain, &clauses);
        let expected = plain.solve();

        // Inprocessing every conflict, with the probe-based passes doing the
        // work — this is the configuration that used to leave hanging units.
        let mut inprocessed = Solver::with_config(SolverConfig {
            enable_inprocessing: true,
            inprocessing_interval: 1,
            enable_self_subsumption: false,
            ..SolverConfig::default()
        });
        load(&mut inprocessed, &clauses);
        assert_eq!(
            inprocessed.solve(),
            expected,
            "round {round}: inprocessing changed the verdict"
        );
    }
}

#[test]
fn test_issue36_self_subsumption_never_flips_a_verdict() {
    let mut rng = Lcg(0x243F_6A88_85A3_08D3);

    for round in 0..120u32 {
        let num_vars = 6 + (round % 6);
        let num_clauses = (num_vars as f64 * 4.4) as u32;

        let mut clauses: Vec<Vec<Lit>> = Vec::new();
        for _ in 0..num_clauses {
            let mut clause: Vec<Lit> = Vec::new();
            while clause.len() < 3 {
                let v = rng.below(u64::from(num_vars)) as u32;
                let lit = if rng.below(2) == 0 { pos(v) } else { neg(v) };
                if !clause.iter().any(|&l| l.var() == lit.var()) {
                    clause.push(lit);
                }
            }
            clauses.push(clause);
        }

        let base = SolverConfig {
            enable_inprocessing: true,
            inprocessing_interval: 1,
            ..SolverConfig::default()
        };

        let mut off = Solver::with_config(SolverConfig {
            enable_self_subsumption: false,
            ..base.clone()
        });
        load(&mut off, &clauses);
        let expected = off.solve();

        let mut on = Solver::with_config(SolverConfig {
            enable_self_subsumption: true,
            ..base
        });
        load(&mut on, &clauses);
        let result = on.solve();

        assert_eq!(
            result, expected,
            "round {round}: self-subsumption flipped the verdict"
        );
        if result == SolverResult::Sat
            && let Err(msg) = model_satisfies(&on, &clauses)
        {
            panic!("round {round}: reported Sat but {msg}");
        }
    }
}
