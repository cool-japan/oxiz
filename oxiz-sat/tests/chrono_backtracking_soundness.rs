//! Regression tests for chronological backtracking soundness.
//!
//! Chronological backtracking deliberately stops *above* the learned clause's
//! assertion level, keeping the intervening decisions instead of throwing them
//! away (Nadel & Ryvchin, *Chronological Backtracking*, SAT 2018).  That is only
//! sound if the rest of the engine maintains the invariants it implies:
//!
//!  * a learned clause's asserting literal is assigned at its **true**
//!    implication level — the maximum level over the clause's other literals —
//!    and a **unit** learned clause therefore lands at level 0, not at whatever
//!    level the search happens to sit at after the rollback;
//!  * the trail is consequently no longer sorted by decision level, so rollback
//!    filters by level rather than truncating at a level boundary, and the 1-UIP
//!    walk in conflict analysis only resolves on conflict-level literals.
//!
//! When those invariants were violated, a unit lemma was pinned inside a
//! decision level as a second reason-less "decision".  Conflict analysis at that
//! level then hit it while literals were still unresolved, terminated the 1-UIP
//! loop early, and emitted a clause *stronger* than what resolution derives —
//! yielding a false `Unsat` on satisfiable input.
//!
//! Every test here runs with `chrono_backtrack_threshold: 0`, which forces the
//! chronological path on every non-unit conflict, so the code under test is
//! exercised even on instances too small to trigger the production heuristic.

use oxiz_sat::{LBool, Lit, Solver, SolverConfig, SolverResult, Var};

/// A configuration that takes the chronological path on every conflict.
fn chrono_config(threshold: u32) -> SolverConfig {
    SolverConfig {
        enable_chronological_backtrack: true,
        chrono_backtrack_threshold: threshold,
        // Keep the engine minimal so a verdict change is attributable to
        // backtracking rather than to a clause-adding side heuristic.
        enable_lazy_hyper_binary: false,
        enable_inprocessing: false,
        // Likewise off: a pre-search lucky hit returns `Sat` without the
        // search running at all, so the `chrono_used` assertion at the end of
        // the differential test could no longer be met (see
        // `SolverConfig::enable_lucky_phase`).
        enable_lucky_phase: false,
        random_polarity_prob: 0.0,
        restart_interval: 20,
        ..SolverConfig::default()
    }
}

fn solve_cnf(num_vars: usize, cnf: &[&[i32]], config: SolverConfig) -> SolverResult {
    let mut solver = Solver::with_config(config);
    for _ in 0..num_vars {
        solver.new_var();
    }
    for clause in cnf {
        solver.add_clause_dimacs(clause);
    }
    solver.solve()
}

/// Exhaustive oracle over all `2^num_vars` assignments.
fn brute_force_is_sat(num_vars: usize, cnf: &[Vec<i32>]) -> bool {
    assert!(num_vars <= 20, "brute-force oracle is exponential");
    (0u32..(1u32 << num_vars)).any(|mask| {
        cnf.iter().all(|clause| {
            clause.iter().any(|&lit| {
                let var_bit = (mask >> (lit.unsigned_abs() as usize - 1)) & 1 == 1;
                (lit > 0) == var_bit
            })
        })
    })
}

/// The minimal instance reduced from a differential fuzz run against an
/// exhaustive oracle.  Every literal is forced by unit propagation once the
/// first decision is taken — `¬10, 6, ¬3, ¬11, 8, 1, 5` is the unique model — so
/// the correct verdict is unambiguously `Sat`.
///
/// The historical failure ran like this (levels shown as `@level`):
///
/// 1. a conflict at level 5 learns `(10 ∨ ¬11)` with assertion level 1, and the
///    chronological rollback stops at level 3.  `10` is then recorded `@3`
///    although `¬11` — the only other literal of its reason — sits `@1`, so its
///    true implication level is 1.
/// 2. a conflict at level 3 resolves that inflated level into the unit lemma
///    `(¬10)`, which is likewise installed at level 1 instead of level 0, as a
///    reason-less "decision" in the middle of level 1's trail block.
/// 3. the next conflict at level 1 walks the trail backwards, reaches that
///    planted decision while one conflict-level literal is still unresolved,
///    stops early, and emits the unit `(10)` — the exact negation of the lemma
///    from step 2, and not something resolution derives.  `10` and `¬10` are now
///    both units, so the solver reports `Unsat`.
#[test]
fn test_chrono_backtracking_soundness_minimal() {
    let cnf: &[&[i32]] = &[
        &[10, -3],
        &[5],
        &[-10, -7],
        &[8, 11],
        &[3, -11, -6],
        &[7, -10],
        &[11, -8, 1],
        &[10, 6],
    ];

    assert_eq!(
        solve_cnf(11, cnf, chrono_config(0)),
        SolverResult::Sat,
        "instance is satisfiable (x = {{5, 6, 8, 1}} true, {{3, 10, 11}} false); \
         chronological backtracking must not refute it"
    );

    // Same verdict without chronological backtracking, and with the production
    // threshold — the fix must not depend on the heuristic being switched off.
    let mut plain = chrono_config(0);
    plain.enable_chronological_backtrack = false;
    assert_eq!(solve_cnf(11, cnf, plain), SolverResult::Sat);
    assert_eq!(solve_cnf(11, cnf, chrono_config(100)), SolverResult::Sat);
}

/// The model reported for the minimal instance must actually satisfy it — a
/// `Sat` verdict backed by a broken trail would still be a bug.
#[test]
fn test_chrono_backtracking_minimal_model_is_valid() {
    let cnf: &[&[i32]] = &[
        &[10, -3],
        &[5],
        &[-10, -7],
        &[8, 11],
        &[3, -11, -6],
        &[7, -10],
        &[11, -8, 1],
        &[10, 6],
    ];

    let mut solver = Solver::with_config(chrono_config(0));
    for _ in 0..11 {
        solver.new_var();
    }
    for clause in cnf {
        solver.add_clause_dimacs(clause);
    }
    assert_eq!(solver.solve(), SolverResult::Sat);

    for clause in cnf {
        assert!(
            clause.iter().any(|&lit| {
                let value = solver.model_value(Var::new(lit.unsigned_abs() - 1));
                if lit > 0 {
                    value.is_true()
                } else {
                    value.is_false()
                }
            }),
            "reported model falsifies clause {clause:?}"
        );
    }
}

/// A learned clause's asserting literal must be assigned at the maximum level
/// over the clause's *other* literals, not at the level the search sits at after
/// a chronological rollback.
///
/// The instance below forces a conflict whose 1-UIP lemma is a unit: `a` is
/// decided, and `(¬a ∨ b)`, `(¬a ∨ ¬b)` refute it immediately.  The lemma `(¬a)`
/// is a consequence of the formula alone, so it belongs at level 0 — and the
/// solver must end up with `a` permanently false there, whatever level the
/// rollback landed on.
#[test]
fn test_unit_lemma_is_pinned_at_root_level() {
    let mut solver = Solver::with_config(chrono_config(0));
    let a = solver.new_var();
    let b = solver.new_var();
    let c = solver.new_var();

    solver.add_clause([Lit::neg(a), Lit::pos(b)]);
    solver.add_clause([Lit::neg(a), Lit::neg(b)]);
    // Keep the instance satisfiable so the solver has to reach a model rather
    // than refute everything: (a ∨ c).
    solver.add_clause([Lit::pos(a), Lit::pos(c)]);

    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(
        solver.model_value(a),
        LBool::False,
        "(¬a) is entailed, so a must be false in every model"
    );
    assert_eq!(
        solver.model_value(c),
        LBool::True,
        "with a false, (a ∨ c) forces c"
    );
}

/// Bounded randomised differential test: small CNFs solved with chronological
/// backtracking forced on, checked against an exhaustive oracle.
///
/// Deterministic (fixed seed, no wall-clock dependence) and sized to stay well
/// inside a normal CI budget.
#[test]
fn test_chrono_backtracking_matches_reference() {
    // xorshift64* — deterministic and dependency-free.
    let mut state: u64 = 0x2026_0728_C0FF_EE01;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        state
    };

    let mut chrono_used = false;

    for _ in 0..1500 {
        let num_vars = 4 + (next() % 9) as usize; // 4..=12
        let num_clauses = 6 + (next() % 30) as usize; // 6..=35

        let mut cnf: Vec<Vec<i32>> = Vec::with_capacity(num_clauses);
        for _ in 0..num_clauses {
            let len = 1 + (next() % 3) as usize; // 1..=3
            let mut clause: Vec<i32> = Vec::with_capacity(len);
            for _ in 0..len {
                let var = 1 + (next() % num_vars as u64) as i32;
                let lit = if next() % 2 == 0 { var } else { -var };
                if !clause.contains(&lit) && !clause.contains(&(-lit)) {
                    clause.push(lit);
                }
            }
            if !clause.is_empty() {
                cnf.push(clause);
            }
        }

        let expected = brute_force_is_sat(num_vars, &cnf);

        let mut solver = Solver::with_config(chrono_config(0));
        for _ in 0..num_vars {
            solver.new_var();
        }
        for clause in &cnf {
            solver.add_clause_dimacs(clause);
        }
        let got = solver.solve();
        chrono_used |= solver.stats().chrono_backtracks > 0;

        let expected = if expected {
            SolverResult::Sat
        } else {
            SolverResult::Unsat
        };
        assert_eq!(got, expected, "disagreement with oracle on CNF {cnf:?}");
    }

    assert!(
        chrono_used,
        "threshold 0 must actually take the chronological path; otherwise this \
         test is not exercising what it claims to"
    );
}
