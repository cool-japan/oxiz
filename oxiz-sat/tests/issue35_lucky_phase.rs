//! Issue #35: CaDiCaL-style "lucky" pre-solve phase.
//!
//! The reported symptom was a family of instances (`simon-r18-0`, `r21-1`,
//! `r23-1`) that CaDiCaL answers in under 5 ms via a lucky assignment while an
//! unaided CDCL loop times out at 30 s. These tests pin the observable
//! contract of the fix through the *public* API only — the per-scan mechanics
//! live in `solver/lucky.rs`'s own unit tests.
//!
//! What must hold:
//!
//! * a formula one of the scans satisfies is answered `Sat` with no search at
//!   all (zero conflicts, zero decisions);
//! * a formula no scan satisfies is answered exactly as before;
//! * assumptions are honored — a candidate that would violate one is never
//!   reported, and the assumption-driven `Unsat` + core path still works;
//! * the flag switches the whole thing off;
//! * proof tracing is unaffected (a `Sat` verdict has no proof obligation, and
//!   the phase emits nothing).

use oxiz_sat::{LBool, Lit, Solver, SolverConfig, SolverResult, Var};

fn lucky_disabled() -> SolverConfig {
    SolverConfig {
        enable_lucky_phase: false,
        ..SolverConfig::default()
    }
}

fn solver_with(config: SolverConfig, vars: usize, clauses: &[Vec<i32>]) -> Solver {
    let mut solver = Solver::with_config(config);
    for _ in 0..vars {
        solver.new_var();
    }
    for clause in clauses {
        solver.add_clause_dimacs(clause);
    }
    solver
}

fn model_satisfies(solver: &Solver, clauses: &[Vec<i32>]) -> bool {
    clauses.iter().all(|clause| {
        clause.iter().any(|&dimacs| {
            let var = Var::new(dimacs.unsigned_abs() - 1);
            match solver.model_value(var) {
                LBool::True => dimacs > 0,
                LBool::False => dimacs < 0,
                LBool::Undef => false,
            }
        })
    })
}

/// xorshift64*, so the generated families below are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0 = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        self.0
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// Random 3-SAT with every literal positive: satisfied by "everything true",
/// which is the very first scan.
fn all_positive_3sat(vars: usize, clauses: usize, seed: u64) -> Vec<Vec<i32>> {
    let mut rng = Rng(seed);
    (0..clauses)
        .map(|_| {
            (0..3)
                .map(|_| 1 + rng.below(vars as u64) as i32)
                .collect::<Vec<i32>>()
        })
        .collect()
}

/// Random 3-SAT with every literal negative: the second scan's territory.
fn all_negative_3sat(vars: usize, clauses: usize, seed: u64) -> Vec<Vec<i32>> {
    all_positive_3sat(vars, clauses, seed)
        .into_iter()
        .map(|clause| clause.into_iter().map(|lit| -lit).collect())
        .collect()
}

/// A formula none of the six scans solves, but which is satisfiable — the
/// same construction `solver/lucky.rs`'s `unlucky_formula_declines_after_every_scan`
/// uses, expressed in DIMACS.
fn unlucky_instance() -> Vec<Vec<i32>> {
    let trap = |base: i32| -> Vec<Vec<i32>> {
        let (a, b, c) = (base + 1, base + 2, base + 3);
        vec![vec![-a, -b], vec![a, b], vec![a, c], vec![-c, -b]]
    };
    let mut clauses = trap(0);
    let mut mirrored = trap(3);
    mirrored.reverse();
    clauses.extend(mirrored);
    // The negated image on fresh variables 7..=12 defeats the two closures
    // the first half does not.
    let mut negated_half = trap(6);
    let mut negated_mirror = trap(9);
    negated_mirror.reverse();
    negated_half.extend(negated_mirror);
    clauses.extend(
        negated_half
            .into_iter()
            .map(|clause| clause.into_iter().map(|lit| -lit).collect::<Vec<i32>>()),
    );
    clauses
}

#[test]
fn all_positive_instance_is_solved_without_any_search() {
    let clauses = all_positive_3sat(1000, 4000, 0x5EED_0001);
    let mut solver = solver_with(SolverConfig::default(), 1000, &clauses);

    assert_eq!(solver.solve(), SolverResult::Sat);
    let stats = solver.stats();
    assert_eq!(stats.lucky_successes, 1, "the lucky phase must have hit");
    assert_eq!(stats.lucky_attempts, 1, "and on its very first scan");
    assert_eq!(stats.conflicts, 0, "no search may have happened");
    assert_eq!(stats.decisions, 0, "no search may have happened");
    assert!(model_satisfies(&solver, &clauses));
}

#[test]
fn all_negative_instance_is_solved_by_the_second_scan() {
    let clauses = all_negative_3sat(1000, 4000, 0x5EED_0002);
    let mut solver = solver_with(SolverConfig::default(), 1000, &clauses);

    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.stats().lucky_successes, 1);
    assert_eq!(solver.stats().lucky_attempts, 2);
    assert_eq!(solver.stats().decisions, 0);
    assert!(model_satisfies(&solver, &clauses));
}

#[test]
fn instance_no_scan_solves_falls_through_to_the_search() {
    let clauses = unlucky_instance();
    let mut solver = solver_with(SolverConfig::default(), 12, &clauses);

    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.stats().lucky_successes, 0);
    assert_eq!(
        solver.stats().lucky_attempts,
        6,
        "every scan should have been tried and declined"
    );
    assert!(
        solver.stats().decisions > 0,
        "the CDCL loop must actually have run"
    );
    assert!(model_satisfies(&solver, &clauses));
}

#[test]
fn unsat_instance_is_unaffected() {
    // PHP(3,2): 3 pigeons, 2 holes.
    let clauses = vec![
        vec![1, 2],
        vec![3, 4],
        vec![5, 6],
        vec![-1, -3],
        vec![-1, -5],
        vec![-3, -5],
        vec![-2, -4],
        vec![-2, -6],
        vec![-4, -6],
    ];
    let mut solver = solver_with(SolverConfig::default(), 6, &clauses);
    assert_eq!(solver.solve(), SolverResult::Unsat);
    assert_eq!(
        solver.stats().lucky_successes,
        0,
        "an unsatisfiable formula can never be reported Sat by a scan"
    );
}

#[test]
fn disabling_the_flag_bypasses_the_phase_entirely() {
    let clauses = all_positive_3sat(300, 1200, 0x5EED_0003);
    let mut solver = solver_with(lucky_disabled(), 300, &clauses);

    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.stats().lucky_attempts, 0);
    assert_eq!(solver.stats().lucky_successes, 0);
    assert!(
        solver.stats().decisions > 0,
        "without the phase this instance is solved by search"
    );
    assert!(model_satisfies(&solver, &clauses));
}

#[test]
fn verdicts_and_models_match_with_the_phase_on_and_off() {
    // Differential sweep over small random CNFs: the phase may not change a
    // single verdict, and every `Sat` it reports must come with a model that
    // really satisfies the input.
    let mut rng = Rng(0x0BAD_C0DE_1234_5678);
    for _ in 0..400 {
        let vars = 3 + rng.below(8) as usize;
        let num_clauses = 4 + rng.below(20) as usize;
        let clauses: Vec<Vec<i32>> = (0..num_clauses)
            .map(|_| {
                let len = 1 + rng.below(3) as usize;
                let mut clause: Vec<i32> = Vec::with_capacity(len);
                for _ in 0..len {
                    let var = 1 + rng.below(vars as u64) as i32;
                    let lit = if rng.below(2) == 0 { var } else { -var };
                    if !clause.contains(&lit) && !clause.contains(&(-lit)) {
                        clause.push(lit);
                    }
                }
                clause
            })
            .filter(|clause| !clause.is_empty())
            .collect();

        let mut with_lucky = solver_with(SolverConfig::default(), vars, &clauses);
        let mut without_lucky = solver_with(lucky_disabled(), vars, &clauses);
        let got = with_lucky.solve();
        assert_eq!(
            got,
            without_lucky.solve(),
            "verdict changed with the lucky phase on: {clauses:?}"
        );
        if got == SolverResult::Sat {
            assert!(
                model_satisfies(&with_lucky, &clauses),
                "lucky model does not satisfy {clauses:?}"
            );
        }
    }
}

#[test]
fn assumption_conflicting_with_a_lucky_candidate_is_respected() {
    // `(1 ∨ 2)` is solved by the all-true scan, whose candidate sets var 2
    // true. Assuming ¬2 must not be able to produce that candidate.
    let clauses = vec![vec![1, 2], vec![1, 3]];
    let mut solver = solver_with(SolverConfig::default(), 3, &clauses);

    let (result, core) = solver.solve_with_assumptions(&[Lit::neg(Var::new(1))]);
    assert_eq!(result, SolverResult::Sat);
    assert!(core.is_none());
    assert_eq!(
        solver.model_value(Var::new(1)),
        LBool::False,
        "the assumption must hold in the reported model"
    );
    assert!(model_satisfies(&solver, &clauses));
}

#[test]
fn assumptions_that_are_unsatisfiable_still_produce_a_core() {
    // `(1 ∨ 2)` with both assumed false: UNSAT under assumptions. The lucky
    // phase must decline rather than report either verdict itself.
    let clauses = vec![vec![1, 2]];
    let mut solver = solver_with(SolverConfig::default(), 2, &clauses);

    let (result, core) =
        solver.solve_with_assumptions(&[Lit::neg(Var::new(0)), Lit::neg(Var::new(1))]);
    assert_eq!(result, SolverResult::Unsat);
    let core = core.expect("an UNSAT-under-assumptions answer owes a core");
    assert!(
        !core.is_empty() && core.iter().all(|lit| !lit.is_pos()),
        "core must be drawn from the assumptions, got {core:?}"
    );
    assert_eq!(solver.stats().lucky_successes, 0);
}

#[test]
fn assumption_contradicting_a_unit_clause_is_unsat_with_a_core() {
    // The unit `(1)` is a level-0 fact, not a database clause. Assuming ¬1
    // makes the lucky phase decline outright so the ordinary core path runs.
    let clauses = vec![vec![1], vec![1, 2]];
    let mut solver = solver_with(SolverConfig::default(), 2, &clauses);

    let (result, core) = solver.solve_with_assumptions(&[Lit::neg(Var::new(0))]);
    assert_eq!(result, SolverResult::Unsat);
    assert!(core.is_some());
    assert_eq!(solver.stats().lucky_attempts, 0, "the phase must decline");
}

#[test]
fn level_zero_units_are_never_contradicted_by_a_lucky_model() {
    // All-true would satisfy every *clause* here while contradicting the unit
    // `(¬1)`, which lives on the trail rather than in the database.
    let clauses = vec![vec![-1], vec![1, 2], vec![2, 3]];
    let mut solver = solver_with(SolverConfig::default(), 3, &clauses);

    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.model_value(Var::new(0)), LBool::False);
    assert!(model_satisfies(&solver, &clauses));
}

#[test]
fn incremental_reuse_after_a_lucky_hit_stays_correct() {
    // A lucky hit leaves the solver usable: more clauses, another solve, and
    // eventually an unsatisfiable extension.
    let mut solver = Solver::new();
    for _ in 0..3 {
        solver.new_var();
    }
    solver.add_clause_dimacs(&[1, 2]);
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.stats().lucky_successes, 1);

    solver.add_clause_dimacs(&[-1]);
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.model_value(Var::new(0)), LBool::False);
    assert_eq!(solver.model_value(Var::new(1)), LBool::True);

    solver.add_clause_dimacs(&[-2]);
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

#[test]
fn drat_tracing_of_a_lucky_sat_answer_emits_no_conclusion() {
    use std::io::Read as _;

    let path = std::env::temp_dir().join("oxiz_sat_issue35_lucky_sat.drat");
    let mut solver = Solver::new();
    solver
        .enable_drat_proof(&path)
        .expect("DRAT proof file should be creatable");
    for _ in 0..4 {
        solver.new_var();
    }
    for clause in [vec![1, 2], vec![2, 3], vec![3, 4]] {
        solver.add_clause_dimacs(&clause);
    }

    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.stats().lucky_successes, 1);
    solver.disable_drat_proof();

    let mut contents = String::new();
    std::fs::File::open(&path)
        .expect("proof file")
        .read_to_string(&mut contents)
        .expect("readable proof");
    let _ = std::fs::remove_file(&path);
    assert!(
        contents.trim().is_empty(),
        "a Sat verdict carries no proof obligation, got: {contents:?}"
    );
}

#[test]
fn drat_tracing_of_an_unsat_instance_is_unchanged_by_the_phase() {
    use std::io::Read as _;

    let path = std::env::temp_dir().join("oxiz_sat_issue35_lucky_unsat.drat");
    let mut solver = Solver::new();
    solver
        .enable_drat_proof(&path)
        .expect("DRAT proof file should be creatable");
    for _ in 0..6 {
        solver.new_var();
    }
    for clause in [
        vec![1, 2],
        vec![3, 4],
        vec![5, 6],
        vec![-1, -3],
        vec![-1, -5],
        vec![-3, -5],
        vec![-2, -4],
        vec![-2, -6],
        vec![-4, -6],
    ] {
        solver.add_clause_dimacs(&clause);
    }

    assert_eq!(solver.solve(), SolverResult::Unsat);
    solver.disable_drat_proof();

    let mut contents = String::new();
    std::fs::File::open(&path)
        .expect("proof file")
        .read_to_string(&mut contents)
        .expect("readable proof");
    let _ = std::fs::remove_file(&path);
    assert!(
        contents.lines().any(|line| line.trim() == "0"),
        "the refutation must still end in the empty clause, got: {contents:?}"
    );
}

#[test]
fn lrat_tracing_survives_a_lucky_sat_answer() {
    let path = std::env::temp_dir().join("oxiz_sat_issue35_lucky_sat.lrat");
    let mut solver = Solver::new();
    solver
        .enable_lrat_proof(&path)
        .expect("LRAT proof file should be creatable");
    for _ in 0..4 {
        solver.new_var();
    }
    for clause in [vec![1, 2], vec![2, 3]] {
        solver.add_clause_dimacs(&clause);
    }

    // The lucky phase neither adds nor deletes a clause, so it owes the LRAT
    // trace nothing and does not have to gate itself off the way the
    // inprocessing mechanisms do.
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert_eq!(solver.stats().lucky_successes, 1);
    assert!(solver.lrat_proof_enabled());
    solver.disable_lrat_proof();
    let _ = std::fs::remove_file(&path);
}
