//! Root-scoped bit-blasting (`#P2b-46` (f), decision (45), `bv/solver/scope.rs`):
//! definitions are installed once and outlive every `pop`, only assertions are
//! scoped, and a refutation is explained by its failed-assumption core.
//!
//! * The variable-count pins are the leak itself: before re-fix pass 12 every
//!   `pop` retracted the circuits encoded inside the scope, every re-assertion
//!   re-encoded them with fresh variables, and `assert_neq` minted one fresh
//!   variable per bit per call — so the embedded variable count, and with it
//!   the `O(num_vars)` price of one check, grew with the number of outer
//!   backtracks rather than with the formula.
//! * The differential test drives random `push` / `pop` / assertion / `check`
//!   interleavings over three narrow bit-vectors against brute force, because
//!   the scoping of assertions is now the whole soundness argument of `pop`,
//!   and checks every `Unsat` explanation (the failed-assumption core, mapped
//!   back to the blamed terms) is itself unsatisfiable.  That last check is
//!   what found `oxiz-sat`'s core walk collapsing `x` and `¬x` into one
//!   literal when both were assumed (`a <u b` beside `a >=u b`).

use oxiz_core::ast::TermId;
use oxiz_theories::bv::BvSolver;
use oxiz_theories::{Theory, TheoryCheckResult};

#[test]
fn re_asserting_a_disequality_after_its_pop_mints_no_variable() {
    let mut solver = BvSolver::new();
    let (a, b) = (TermId::new(1), TermId::new(2));
    solver.new_bv(a, 8);
    solver.new_bv(b, 8);
    solver.push();
    assert!(solver.assert_neq(a, b));
    assert!(matches!(
        solver.check().expect("check"),
        TheoryCheckResult::Sat
    ));
    solver.pop();
    let after_first = solver.embedded_variable_count();
    for _ in 0..50 {
        solver.push();
        assert!(solver.assert_neq(a, b));
        assert!(solver.assert_neq(a, b));
        assert!(matches!(
            solver.check().expect("check"),
            TheoryCheckResult::Sat
        ));
        solver.pop();
    }
    assert_eq!(
        solver.embedded_variable_count(),
        after_first,
        "the equality gate `assert_neq` asserts false is defined once"
    );
}

#[test]
fn re_encoding_a_circuit_after_its_pop_mints_no_variable() {
    let mut solver = BvSolver::new();
    let (x, y, z) = (TermId::new(1), TermId::new(2), TermId::new(3));
    solver.new_bv(x, 8);
    solver.new_bv(y, 8);
    let mut counts = Vec::new();
    for round in 0..20u64 {
        solver.push();
        // The encoder calls this only for a term with no circuit yet; after
        // the pop the circuit is still there, so it is not rebuilt.
        if solver.get_bv(z).is_none() {
            assert!(solver.bv_mul(z, x, y));
        }
        assert!(solver.assert_const(x, round + 1, 8));
        assert!(solver.assert_const(y, 3, 8));
        assert!(solver.assert_ult(x, z));
        assert!(matches!(
            solver.check().expect("check"),
            TheoryCheckResult::Sat
        ));
        assert_eq!(solver.get_value(z), Some((round + 1) * 3));
        solver.pop();
        counts.push(solver.embedded_variable_count());
    }
    assert!(
        counts.windows(2).all(|w| w[0] == w[1]),
        "the variable count must be flat across scopes: {counts:?}"
    );
}

/// Brute-force oracle over three bit-vectors of width `W`.
const W: u32 = 3;

#[derive(Clone, Copy, Debug)]
enum Atom {
    Eq(usize, usize),
    Neq(usize, usize),
    Ult(usize, usize),
    Ule(usize, usize),
    Slt(usize, usize),
    Const(usize, u64),
}

/// An assertion checked before any term was recorded for it stays unblamed
/// (the `blame_from` watermark behind `BvSolver::record_constraint_term`), so
/// a later, unrelated term never becomes its explanation.  `a <u b` is
/// asserted with no term and checked while consistent; only then is
/// `b <u a` asserted and recorded as `later`.  The refutation needs both, and
/// `b <u a` alone is satisfiable, so an explanation of `[later]` alone would
/// be a false theory lemma; the unblamed literal sends the explanation to the
/// sound fallback instead — every recorded term, `earlier` included.
#[test]
fn an_assertion_checked_before_its_term_is_never_charged_to_a_later_term() {
    let mut solver = BvSolver::new();
    let (a, b, p, q) = (
        TermId::new(1),
        TermId::new(2),
        TermId::new(3),
        TermId::new(4),
    );
    for term in [a, b, p, q] {
        solver.new_bv(term, 4);
    }
    assert!(solver.assert_ult(p, q));
    let earlier = TermId::new(101);
    solver.record_constraint_term(earlier);
    assert!(solver.assert_ult(a, b));
    assert!(matches!(
        solver.check().expect("check"),
        TheoryCheckResult::Sat
    ));
    assert!(solver.assert_ult(b, a));
    let later = TermId::new(102);
    solver.record_constraint_term(later);
    match solver.check().expect("check") {
        TheoryCheckResult::Unsat(terms) => {
            assert!(
                terms.contains(&later) && terms.contains(&earlier),
                "the unblamed `a <u b` must send the explanation to every \
                 recorded term, not charge it to `later`: {terms:?}"
            );
        }
        other => panic!("`a <u b` beside `b <u a` is unsat, got {other:?}"),
    }
}

fn holds(atom: Atom, values: [u64; 3]) -> bool {
    let signed = |v: u64| -> i64 {
        if v >> (W - 1) & 1 == 1 {
            v as i64 - (1i64 << W)
        } else {
            v as i64
        }
    };
    match atom {
        Atom::Eq(i, j) => values[i] == values[j],
        Atom::Neq(i, j) => values[i] != values[j],
        Atom::Ult(i, j) => values[i] < values[j],
        Atom::Ule(i, j) => values[i] <= values[j],
        Atom::Slt(i, j) => signed(values[i]) < signed(values[j]),
        Atom::Const(i, c) => values[i] == c,
    }
}

fn satisfiable(atoms: &[Atom]) -> bool {
    let n = 1u64 << W;
    (0..n * n * n).any(|code| {
        let values = [code % n, (code / n) % n, code / (n * n)];
        atoms.iter().all(|&atom| holds(atom, values))
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
fn random_scoped_assertions_agree_with_brute_force() {
    let vars = [TermId::new(1), TermId::new(2), TermId::new(3)];
    for seed in 1..=3000u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        let mut solver = BvSolver::new();
        for &v in &vars {
            solver.new_bv(v, W);
        }
        let mut scopes: Vec<Vec<(TermId, Atom)>> = vec![Vec::new()];
        let mut next_guard = 100u32;
        for step in 0..40 {
            match rng.below(9) {
                0 => {
                    solver.push();
                    scopes.push(Vec::new());
                }
                1 => {
                    if scopes.len() > 1 {
                        solver.pop();
                        scopes.pop();
                    }
                }
                2 | 3 => {
                    let atoms: Vec<Atom> = scopes.iter().flatten().map(|&(_, atom)| atom).collect();
                    let expected = satisfiable(&atoms);
                    match solver.check().expect("check") {
                        TheoryCheckResult::Sat => {
                            assert!(expected, "seed {seed} step {step}: wrong sat {atoms:?}");
                            let values = [
                                solver.get_value(vars[0]).unwrap_or(0),
                                solver.get_value(vars[1]).unwrap_or(0),
                                solver.get_value(vars[2]).unwrap_or(0),
                            ];
                            for &atom in &atoms {
                                assert!(
                                    holds(atom, values),
                                    "seed {seed} step {step}: model {values:?} falsifies {atom:?}"
                                );
                            }
                        }
                        TheoryCheckResult::Unsat(blamed) => {
                            assert!(!expected, "seed {seed} step {step}: wrong unsat {atoms:?}");
                            // The explanation is the core: the atoms it blames
                            // are unsatisfiable on their own.
                            let core: Vec<Atom> = scopes
                                .iter()
                                .flatten()
                                .filter(|(guard, _)| blamed.contains(guard))
                                .map(|&(_, atom)| atom)
                                .collect();
                            assert!(
                                !satisfiable(&core),
                                "seed {seed} step {step}: the explanation {core:?} is satisfiable"
                            );
                        }
                        other => panic!("seed {seed} step {step}: {other:?}"),
                    }
                }
                _ => {
                    let i = rng.below(3) as usize;
                    let j = rng.below(3) as usize;
                    let atom = match rng.below(6) {
                        0 => Atom::Eq(i, j),
                        1 => Atom::Neq(i, j),
                        2 => Atom::Ult(i, j),
                        3 => Atom::Ule(i, j),
                        4 => Atom::Slt(i, j),
                        _ => Atom::Const(i, rng.below(1 << W)),
                    };
                    let (a, b) = match atom {
                        Atom::Eq(i, j)
                        | Atom::Neq(i, j)
                        | Atom::Ult(i, j)
                        | Atom::Ule(i, j)
                        | Atom::Slt(i, j) => (vars[i], vars[j]),
                        Atom::Const(i, _) => (vars[i], vars[i]),
                    };
                    let asserted = match atom {
                        Atom::Eq(..) => solver.assert_eq(a, b),
                        Atom::Neq(..) => solver.assert_neq(a, b),
                        Atom::Ult(..) => solver.assert_ult(a, b),
                        Atom::Ule(..) => solver.assert_ule(a, b),
                        Atom::Slt(..) => solver.assert_slt(a, b),
                        Atom::Const(_, c) => solver.assert_const(a, c, W),
                    };
                    assert!(asserted, "seed {seed} step {step}: {atom:?} not asserted");
                    let guard = TermId::new(next_guard);
                    solver.record_constraint_term(guard);
                    next_guard += 1;
                    if let Some(top) = scopes.last_mut() {
                        top.push((guard, atom));
                    }
                }
            }
        }
    }
}
