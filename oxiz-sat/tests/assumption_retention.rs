//! `Solver::solve_with_assumptions` must never answer `Sat` with a model that
//! falsifies one of its own assumptions, and never `Unsat` with a core that is
//! not itself unsatisfiable together with the clause set.
//!
//! # The defect this pins (found 2026-09-28, re-fix pass 12)
//!
//! The assumptions were decided once, at levels `1..=k`, before the search
//! loop.  Inside the loop a conflict backjumped to `max(bt, 1)` and a restart
//! went to level 0, and nothing re-decided the assumptions the backjump had
//! undone: `pick_branch_var` then chose them like any other variable, and a
//! later propagation could set one of them false.  The model then violated
//! the assumption it was asked to honour, and the call still answered `Sat`.
//!
//! Every script here is built so that forgetting an assumption is observable:
//! assumption `a_i` is tied to a fresh `x_i` by `a_i <-> !x_i`, and the rest
//! of the clause set is a satisfiable random 3-SAT instance near the phase
//! transition that mentions the `x_i`, so the search conflicts, backjumps and
//! restarts, and an undone `a_i` can be propagated false through `x_i`.  An
//! independent brute-force-free check follows: every `Sat` model is verified
//! against the clause set **and** every assumption; every `Unsat` core is
//! re-solved as unit clauses and must be `Unsat`.

use oxiz_sat::{Lit, Solver, SolverResult, Var};

/// A tiny deterministic generator, so the test needs no dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn lit_of(var: u32, positive: bool) -> Lit {
    if positive {
        Lit::pos(Var::new(var))
    } else {
        Lit::neg(Var::new(var))
    }
}

fn holds(model: &[oxiz_sat::LBool], lit: Lit) -> bool {
    let value = model
        .get(lit.var().index())
        .copied()
        .unwrap_or(oxiz_sat::LBool::Undef);
    if lit.is_pos() {
        value.is_true()
    } else {
        value.is_false()
    }
}

/// One instance: `(clauses, assumptions)`.
fn instance(seed: u64, vars: u32, ratio_x100: u64, assumed: u32) -> (Vec<Vec<Lit>>, Vec<Lit>) {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let mut clauses: Vec<Vec<Lit>> = Vec::new();
    let count = u64::from(vars) * ratio_x100 / 100;
    for _ in 0..count {
        let mut clause = Vec::with_capacity(3);
        for _ in 0..3 {
            let var = rng.below(u64::from(vars)) as u32;
            clause.push(lit_of(var, rng.below(2) == 0));
        }
        clauses.push(clause);
    }
    // a_i <-> !x_i, with a_i fresh and x_i one of the instance's variables.
    let mut assumptions = Vec::new();
    for i in 0..assumed {
        let a = vars + i;
        let x = rng.below(u64::from(vars)) as u32;
        clauses.push(vec![lit_of(a, false), lit_of(x, false)]);
        clauses.push(vec![lit_of(a, true), lit_of(x, true)]);
        assumptions.push(lit_of(a, true));
    }
    (clauses, assumptions)
}

fn solver_with(clauses: &[Vec<Lit>]) -> Solver {
    let mut solver = Solver::new();
    for clause in clauses {
        solver.add_clause(clause.iter().copied());
    }
    solver
}

#[test]
fn every_sat_model_honours_every_assumption_and_every_core_is_unsat() {
    let mut sat = 0usize;
    let mut unsat = 0usize;
    for seed in 1..=400u64 {
        let (clauses, assumptions) = instance(seed, 70, 390, 10);
        let mut solver = solver_with(&clauses);
        let (result, core) = solver.solve_with_assumptions(&assumptions);
        match result {
            SolverResult::Sat => {
                sat += 1;
                let model = solver.model().to_vec();
                for clause in &clauses {
                    assert!(
                        clause.iter().any(|&lit| holds(&model, lit)),
                        "seed {seed}: the Sat model falsifies a clause"
                    );
                }
                for &assumption in &assumptions {
                    assert!(
                        holds(&model, assumption),
                        "seed {seed}: the Sat model falsifies the assumption {assumption:?}"
                    );
                }
            }
            SolverResult::Unsat => {
                unsat += 1;
                let core = core.unwrap_or_default();
                assert!(
                    core.iter().all(|lit| assumptions.contains(lit)),
                    "seed {seed}: the core names a literal that was not assumed"
                );
                let mut check = solver_with(&clauses);
                for &lit in &core {
                    check.add_clause([lit]);
                }
                assert_eq!(
                    check.solve(),
                    SolverResult::Unsat,
                    "seed {seed}: the core {core:?} is satisfiable with the clauses"
                );
            }
            SolverResult::Unknown => panic!("seed {seed}: no budget was set"),
        }
    }
    assert!(
        sat > 0 && unsat > 0,
        "both outcomes must occur ({sat} sat / {unsat} unsat)"
    );
}

/// The same instances, re-solved many times on one solver under shifting
/// assumption sets: learned clauses are kept across calls, so a clause learned
/// under one set must never refute a later set it does not follow from.
#[test]
fn repeated_calls_on_one_solver_agree_with_a_fresh_solver() {
    for seed in 1..=60u64 {
        let (clauses, assumptions) = instance(seed, 50, 400, 16);
        let mut solver = solver_with(&clauses);
        for round in 0..8usize {
            let subset: Vec<Lit> = assumptions
                .iter()
                .enumerate()
                .filter(|(i, _)| (i + round) % 3 != 0)
                .map(|(_, &lit)| if round % 2 == 0 { lit } else { lit.negate() })
                .collect();
            let (incremental, _) = solver.solve_with_assumptions(&subset);
            let mut fresh = solver_with(&clauses);
            for &lit in &subset {
                fresh.add_clause([lit]);
            }
            let expected = fresh.solve();
            assert_eq!(
                incremental, expected,
                "seed {seed} round {round}: incremental {incremental:?} vs fresh {expected:?}"
            );
            if incremental == SolverResult::Sat {
                let model = solver.model().to_vec();
                for &lit in &subset {
                    assert!(
                        holds(&model, lit),
                        "seed {seed} round {round}: {lit:?} dropped"
                    );
                }
            }
        }
    }
}

/// Clauses added *between* calls, over fresh variables and old ones alike —
/// the pattern of a core-guided MaxSAT loop (`oxiz-opt`'s PMRES relaxes each
/// core with fresh selector and blocking variables and calls again).  Every
/// answer must agree with a fresh solver given the same clauses and the
/// assumptions as units.
#[test]
fn clauses_added_between_calls_keep_every_answer_right() {
    for seed in 1..=80u64 {
        let (mut clauses, assumptions) = instance(seed, 30, 380, 8);
        let mut solver = solver_with(&clauses);
        let mut rng = Rng(seed ^ 0xdead_beef);
        for round in 0..10usize {
            let subset: Vec<Lit> = assumptions
                .iter()
                .enumerate()
                .filter(|(i, _)| (i + round) % 4 != 0)
                .map(|(_, &lit)| lit)
                .collect();
            let (incremental, core) = solver.solve_with_assumptions(&subset);
            let mut fresh = solver_with(&clauses);
            for &lit in &subset {
                fresh.add_clause([lit]);
            }
            let expected = fresh.solve();
            assert_eq!(
                incremental, expected,
                "seed {seed} round {round}: incremental {incremental:?} vs fresh {expected:?}"
            );
            if incremental == SolverResult::Unsat {
                let core = core.unwrap_or_default();
                let mut check = solver_with(&clauses);
                for &lit in &core {
                    check.add_clause([lit]);
                }
                assert_eq!(
                    check.solve(),
                    SolverResult::Unsat,
                    "seed {seed} round {round}"
                );
            }
            // A relaxation-shaped clause: an old literal or two plus a fresh one.
            // One fresh variable per round, past the instance's 30 + 8.
            let fresh_var = 38 + round as u32;
            let old = rng.below(30) as u32;
            let clause = vec![lit_of(old, rng.below(2) == 0), lit_of(fresh_var, true)];
            solver.add_clause(clause.iter().copied());
            clauses.push(clause);
        }
    }
}

/// Both polarities of one variable assumed: the core must name **both**.
///
/// The core walk keyed assumptions by variable, so the failed `¬x` (added to
/// the core first) shadowed the `x` decided before it and the core came back
/// `{¬x}` — satisfiable on its own, and a false explanation for a caller that
/// turns cores into conflict clauses (found by `oxiz-theories`'
/// `bv_root_scoped_definitions` differential test, where `a <u b` and
/// `a >=u b` assert the same comparison gate in both polarities).
#[test]
fn complementary_assumptions_both_appear_in_the_core() {
    let mut solver = Solver::new();
    // An unrelated clause, so the solver has something to propagate.
    solver.add_clause([pos(1), pos(2)]);
    let x = pos(0);
    let (result, core) = solver.solve_with_assumptions(&[x, pos(1), x.negate()]);
    assert_eq!(result, SolverResult::Unsat);
    let core = core.unwrap_or_default();
    assert!(
        core.contains(&x) && core.contains(&x.negate()),
        "the core must hold both `x` and `¬x`, got {core:?}"
    );
    assert!(!core.contains(&pos(1)), "`y` played no part: {core:?}");
}

fn pos(v: u32) -> Lit {
    Lit::pos(Var::new(v))
}
