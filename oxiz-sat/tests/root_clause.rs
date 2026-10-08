//! `Solver::add_clause_at_root`: a clause that survives every `pop`.
//!
//! The directed tests pin the four cases the module documentation names —
//! registration below the open scopes, a root unit (which lives only on the
//! trail), a root clause the scoped facts falsify, and a clause added while the
//! trail is above decision level 0.  The differential test drives random
//! interleavings of `push`, `pop`, scoped and root clauses and `solve` against
//! a brute-force oracle over at most twelve variables.  The LRAT tests at the
//! end check that a root clause is an *original* of the proof — registered
//! once, never deleted by a `pop`, and still justifying a root unit that a
//! `pop` re-asserts — with `oxiz_proof::lrat_check`'s pure-Rust checker.

use oxiz_proof::lrat_check::check_lrat_proof;
use oxiz_sat::{LBool, Lit, Solver, SolverResult, Var};
use std::path::PathBuf;

fn pos(v: u32) -> Lit {
    Lit::pos(Var::new(v))
}
fn neg(v: u32) -> Lit {
    Lit::neg(Var::new(v))
}

fn holds(model: &[LBool], lit: Lit) -> bool {
    let value = model
        .get(lit.var().index())
        .copied()
        .unwrap_or(LBool::Undef);
    if lit.is_pos() {
        value.is_true()
    } else {
        value.is_false()
    }
}

#[test]
fn a_root_clause_added_inside_a_scope_survives_its_pop() {
    let mut solver = Solver::new();
    solver.push();
    solver.add_clause_at_root([pos(0), pos(1)]);
    solver.pop();
    solver.add_clause([neg(0)]);
    solver.add_clause([neg(1)]);
    assert_eq!(solver.solve(), SolverResult::Unsat);

    // The control: the same clause added with `add_clause` dies with the scope.
    let mut control = Solver::new();
    control.push();
    control.add_clause([pos(0), pos(1)]);
    control.pop();
    control.add_clause([neg(0)]);
    control.add_clause([neg(1)]);
    assert_eq!(control.solve(), SolverResult::Sat);
}

#[test]
fn a_root_unit_added_inside_a_scope_survives_its_pop() {
    let mut solver = Solver::new();
    solver.add_clause([pos(1), pos(2)]);
    solver.push();
    solver.add_clause([neg(2)]);
    solver.add_clause_at_root([pos(0)]);
    solver.pop();
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert!(
        holds(solver.model(), pos(0)),
        "the root unit was lost with the scope"
    );
    solver.add_clause([neg(0)]);
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

#[test]
fn a_root_clause_the_scope_falsifies_is_kept_for_after_the_pop() {
    let mut solver = Solver::new();
    solver.push();
    solver.add_clause([neg(0)]);
    solver.add_clause([neg(1)]);
    assert!(!solver.add_clause_at_root([pos(0), pos(1)]));
    assert_eq!(solver.solve(), SolverResult::Unsat);
    solver.pop();
    assert_eq!(solver.solve(), SolverResult::Sat);
    let model = solver.model().to_vec();
    assert!(holds(&model, pos(0)) || holds(&model, pos(1)));
    solver.add_clause([neg(0)]);
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert!(
        holds(solver.model(), pos(1)),
        "the root clause did not survive"
    );
}

#[test]
fn a_root_unit_the_scope_falsifies_is_reasserted_by_the_pop() {
    let mut solver = Solver::new();
    solver.push();
    solver.add_clause([neg(0)]);
    assert!(!solver.add_clause_at_root([pos(0)]));
    assert_eq!(solver.solve(), SolverResult::Unsat);
    solver.pop();
    assert_eq!(solver.solve(), SolverResult::Sat);
    assert!(holds(solver.model(), pos(0)));
}

#[test]
fn a_root_contradiction_on_permanent_facts_outlives_every_pop() {
    let mut solver = Solver::new();
    solver.add_clause([neg(0)]);
    solver.push();
    assert!(!solver.add_clause_at_root([pos(0)]));
    solver.pop();
    assert_eq!(solver.solve(), SolverResult::Unsat);
}

/// A clause added after `solve` left its model on the trail, falsified by
/// that model: the search must backjump, and the next answer must be right.
#[test]
fn a_root_clause_added_above_level_zero_is_asserted_like_a_learned_clause() {
    for seed in 0..40u32 {
        let mut solver = Solver::new();
        // A satisfiable chain the search has to decide on.
        for v in 0..6 {
            solver.add_clause([pos(v), pos(v + 1), neg((v + seed) % 7)]);
        }
        assert_eq!(solver.solve(), SolverResult::Sat);
        let model = solver.model().to_vec();
        // The clause every variable's model value falsifies.
        let clause: Vec<Lit> = (0..7)
            .map(|v| {
                if holds(&model, pos(v)) {
                    neg(v)
                } else {
                    pos(v)
                }
            })
            .collect();
        solver.add_clause_at_root(clause.iter().copied());
        assert_eq!(solver.solve(), SolverResult::Sat, "seed {seed}");
        let next = solver.model().to_vec();
        assert!(clause.iter().any(|&lit| holds(&next, lit)), "seed {seed}");
    }
}

/// Brute force over `vars` variables.
fn brute_force(vars: u32, clauses: &[Vec<Lit>]) -> bool {
    (0..(1u32 << vars)).any(|bits| {
        clauses.iter().all(|clause| {
            clause.iter().any(|lit| {
                let value = (bits >> lit.var().index()) & 1 == 1;
                if lit.is_pos() { value } else { !value }
            })
        })
    })
}

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

#[test]
fn random_interleavings_agree_with_brute_force() {
    const VARS: u32 = 10;
    for seed in 1..=300u64 {
        let mut rng = Rng(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) | 1);
        let mut solver = Solver::new();
        let mut root: Vec<Vec<Lit>> = Vec::new();
        let mut scopes: Vec<Vec<Vec<Lit>>> = vec![Vec::new()];
        for step in 0..60 {
            let op = rng.below(10);
            if op < 3 {
                let len = 1 + rng.below(3) as usize;
                let clause: Vec<Lit> = (0..len)
                    .map(|_| {
                        let v = rng.below(u64::from(VARS)) as u32;
                        if rng.below(2) == 0 { pos(v) } else { neg(v) }
                    })
                    .collect();
                solver.add_clause_at_root(clause.iter().copied());
                root.push(clause);
            } else if op < 6 {
                let len = 1 + rng.below(3) as usize;
                let clause: Vec<Lit> = (0..len)
                    .map(|_| {
                        let v = rng.below(u64::from(VARS)) as u32;
                        if rng.below(2) == 0 { pos(v) } else { neg(v) }
                    })
                    .collect();
                solver.add_clause(clause.iter().copied());
                if let Some(top) = scopes.last_mut() {
                    top.push(clause);
                }
            } else if op < 7 {
                solver.push();
                scopes.push(Vec::new());
            } else if op < 8 {
                if scopes.len() > 1 {
                    solver.pop();
                    scopes.pop();
                }
            } else {
                let mut all: Vec<Vec<Lit>> = root.clone();
                for scope in &scopes {
                    all.extend(scope.iter().cloned());
                }
                let expected = brute_force(VARS, &all);
                let got = solver.solve();
                assert_eq!(
                    got == SolverResult::Sat,
                    expected,
                    "seed {seed} step {step}: solver {got:?}, brute force sat = {expected}"
                );
                if got == SolverResult::Sat {
                    let model = solver.model().to_vec();
                    for clause in &all {
                        assert!(
                            clause.iter().any(|&lit| holds(&model, lit)),
                            "seed {seed} step {step}: the model falsifies {clause:?}"
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// LRAT tracing through `add_clause_at_root`.
// ---------------------------------------------------------------------

/// A fresh, uniquely named LRAT output path under `std::env::temp_dir()`.
fn unique_lrat_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "oxiz_sat_root_clause_{tag}_{}_{nanos}.lrat",
        std::process::id()
    ))
}

/// Run `build` under LRAT tracing, require `Unsat` from `solve`, and return
/// the emitted proof text.
fn refute_with_lrat(tag: &str, build: impl FnOnce(&mut Solver)) -> String {
    let path = unique_lrat_path(tag);
    let mut solver = Solver::new();
    let enabled = solver.enable_lrat_proof(&path);
    assert!(enabled.is_ok(), "enable_lrat_proof: {enabled:?}");
    build(&mut solver);
    assert_eq!(solver.solve(), SolverResult::Unsat, "{tag}");
    solver.disable_lrat_proof();
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    text
}

fn dimacs(clause: &[Lit]) -> Vec<i32> {
    clause.iter().map(|lit| lit.to_dimacs()).collect()
}

/// Four pigeons into three holes, every clause a root clause, no scope.
#[test]
fn a_refutation_over_root_clauses_carries_a_checkable_lrat_proof() {
    let mut original: Vec<Vec<i32>> = Vec::new();
    let proof = refute_with_lrat("pigeonhole", |solver| {
        let var = |p: u32, h: u32| p * 3 + h;
        for p in 0..4 {
            let clause: Vec<Lit> = (0..3).map(|h| pos(var(p, h))).collect();
            solver.add_clause_at_root(clause.iter().copied());
            original.push(dimacs(&clause));
        }
        for h in 0..3 {
            for p1 in 0..4 {
                for p2 in (p1 + 1)..4 {
                    let clause = [neg(var(p1, h)), neg(var(p2, h))];
                    solver.add_clause_at_root(clause);
                    original.push(dimacs(&clause));
                }
            }
        }
    });
    let report = check_lrat_proof(&original, &proof);
    assert!(report.verified, "failure: {:?}", report.failure);
}

/// A root clause added inside a scope is not among the `pop`'s deletions:
/// the refutation after the `pop` cites it and the checker accepts that.
#[test]
fn a_root_clause_kept_across_a_pop_is_still_an_original_of_the_proof() {
    let proof = refute_with_lrat("across_pop", |solver| {
        solver.push();
        solver.add_clause_at_root([pos(0), pos(1)]);
        solver.add_clause([pos(2), pos(3)]);
        solver.pop();
        solver.add_clause([neg(0), pos(4)]);
        solver.add_clause([neg(1), pos(4)]);
        solver.add_clause([neg(4)]);
    });
    let original = vec![
        dimacs(&[pos(0), pos(1)]),
        dimacs(&[pos(2), pos(3)]),
        dimacs(&[neg(0), pos(4)]),
        dimacs(&[neg(1), pos(4)]),
        dimacs(&[neg(4)]),
    ];
    let report = check_lrat_proof(&original, &proof);
    assert!(report.verified, "failure: {:?}", report.failure);
}

/// A root unit lives on the trail only; the `pop` unassigns it, clears its
/// justification, and re-asserts it.  The refutation after the `pop` rests on
/// that replayed unit, so its hint chain needs the unit's original id back.
#[test]
fn a_root_unit_replayed_by_a_pop_keeps_its_lrat_justification() {
    let proof = refute_with_lrat("replayed_unit", |solver| {
        solver.push();
        solver.add_clause([pos(5), pos(6)]);
        solver.add_clause_at_root([pos(0)]);
        solver.pop();
        solver.add_clause([neg(0), pos(1)]);
        solver.add_clause([neg(0), neg(1), pos(2)]);
        solver.add_clause([neg(2), neg(1)]);
    });
    let original = vec![
        dimacs(&[pos(5), pos(6)]),
        dimacs(&[pos(0)]),
        dimacs(&[neg(0), pos(1)]),
        dimacs(&[neg(0), neg(1), pos(2)]),
        dimacs(&[neg(2), neg(1)]),
    ];
    let report = check_lrat_proof(&original, &proof);
    assert!(report.verified, "failure: {:?}", report.failure);
}

/// A root unit contradicting a permanent fact closes the proof at once.
#[test]
fn a_root_unit_contradicting_a_permanent_fact_closes_the_lrat_proof() {
    let proof = refute_with_lrat("unit_contradiction", |solver| {
        assert!(solver.add_clause([pos(0)]));
        solver.push();
        assert!(!solver.add_clause_at_root([neg(0)]));
        solver.pop();
    });
    let report = check_lrat_proof(&[vec![1], vec![-1]], &proof);
    assert!(report.verified, "failure: {:?}", report.failure);
}
