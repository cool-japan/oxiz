//! Regression tests for the two push/pop soundness defects the cargo-formal
//! Phase 2b differential fuzz found in the 0.3.4 working tree on 2026-09-14.
//!
//! # Defect 1 — learned clauses outlived the scope they were derived in
//!
//! `oxiz_sat::Solver::learn_clause` (`oxiz-sat/src/solver/learn.rs`) registered
//! every clause it learned in `learned_clause_ids` but — alone among the six
//! clause-installing sites in the crate — **not** in
//! `assertion_clause_ids.last_mut()`, the per-assertion-level list
//! `oxiz_sat::Solver::pop` deletes.  A clause learned while the clauses of a
//! pushed level were in the database therefore survived the matching `pop`,
//! while the clauses that entailed it did not.  The retained clause is a
//! consequence of `base ∧ pushed`, not of `base`, so it over-constrains
//! everything asserted after the `pop` and can refute a satisfiable goal:
//! a **false proof**.
//!
//! It is a *regression* introduced by the U-Z10 undo-journal fix in the same
//! working tree only in the sense that the journals made the popped bit-vector
//! circuit be re-encoded, which is what turned the previously-inert stale
//! learned clauses into active over-constraints; crates.io `oxiz` 0.3.3 answers
//! all thirteen witnesses below `sat` where the fixed tree now does.  The
//! missing registration itself predates the journals.
//!
//! Fix: `learn_clause` registers each learned clause at the currently open
//! assertion level, exactly like `probe.rs`, `propagate.rs`, `add_clause.rs`
//! and the two in-search sites in `solver/mod.rs` already did.  Assertion
//! levels are strictly LIFO, so the deepest open level at learn time is an
//! upper bound on every level the clause's derivation could have touched —
//! which is why per-level registration is as sound as discarding every learned
//! clause on `pop`, and far cheaper.
//!
//! # Defect 2 — a bare `push`/`pop` discarded a base-level contradiction
//!
//! `oxiz_sat::Solver::pop` ended with an unconditional
//! `self.trivially_unsat = false;`.  That flag is the "the clause database
//! contains an outright contradiction" latch, set by `add_clause` (empty
//! clause, a unit contradicting a level-0 fact, a fully falsified clause) and
//! by the pre-search inprocessing passes.  Clearing it on every `pop`
//! regardless of the level it was *set* at threw away a contradiction proved
//! before the `push` ever happened, so
//! `(assert (distinct b b)) (push 1) (pop 1) (check-sat)` answered `sat`.
//!
//! `(distinct x x)` is the shortest trigger because its Tseitin encoding pins
//! the contradiction at level 0 inside `add_clause`: `(= x x)` folds to `true`,
//! so the binary clause `¬result ∨ ¬true` forces `¬result` at level 0 and the
//! assertion's own unit `result` then contradicts a level-0 fact.  The defect
//! is sort-independent (bit-vector, `Int` and `Bool` spellings all lost it) and
//! is **not** a 0.3.4 regression: `oxiz` 0.3.3 drops it too.
//!
//! Fix: `push` snapshots the flag onto `assertion_trivially_unsat` and `pop`
//! restores that snapshot instead of zeroing.  A contradiction latched *below*
//! the retracted scope survives; one latched *inside* it is dropped, exactly as
//! before.
//!
//! # Measured before / after (this file's own scripts)
//!
//! On the unfixed tree the thirteen `wrong_*` witnesses below produced 12 wrong
//! `unsat` and 1 wrong `sat`; all four `minimal_*` scripts were wrong.  After
//! both fixes every script in this file matches its brute-force truth, and the
//! 12,000-trial Family A differential campaign that produced the witnesses
//! reports 0 wrong `sat` and 0 wrong `unsat`.

use oxiz_solver::Context;

/// Run one SMT-LIB2 script and return **every** `sat` / `unsat` / `unknown`
/// verdict it printed, in order.
///
/// The whole list matters here: each of these scripts asks two or three
/// `(check-sat)` questions and the defects only show up in the later ones.
fn verdicts(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    let outputs = ctx.execute_script(script).unwrap_or_default();
    outputs
        .iter()
        .map(|line| line.trim().to_string())
        .filter(|line| line == "sat" || line == "unsat" || line == "unknown")
        .collect()
}

/// Assert that `script` answers exactly `expected`.
///
/// `unknown` is never accepted: every script here is a width-8 two-variable
/// bit-vector problem the solver decides outright, so an `unknown` would be a
/// silent loss of the property under test, not a resource limit.
fn assert_verdicts(script: &str, expected: &[&str]) {
    let got = verdicts(script);
    assert_eq!(
        got, expected,
        "verdict list mismatch\n--- script ---\n{script}\n--- expected {expected:?}, got {got:?}"
    );
}

// ---------------------------------------------------------------------------
// 1. The four minimal reproducers
// ---------------------------------------------------------------------------

/// Defect 1, minimised.  The second `(check-sat)` sees exactly `{A, C}`; `B`
/// was popped.  `{A, B}` is unsatisfiable (`A` pins `a = #xc5` and then
/// `b = #xb3`, which `B` contradicts), so the first answer is `unsat` — and the
/// clauses the search learned proving it outlived the `pop`, refuting the
/// satisfiable `{A, C}`.  Spelled flat (no `push`/`pop`) the same `{A, C}`
/// answers `sat` with `a = #xc5, b = #xb3`, so the unfixed tree contradicted
/// itself.  0.3.3 answers `["sat", "sat"]` — wrong on the *first* check, which
/// is the U-Z10 defect this tree already fixed.
#[test]
fn minimal_pushpop_false_proof() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (and (or (= b #xb3) (= a #x55)) (= a #xc5) (=> (= b #xde) (bvugt a #x01))))
(push 1)
(assert (= (bvnot b) #x7b))
(check-sat)
(pop 1)
(assert (not (bvsle (bvshl b b) a)))
(check-sat)
"#,
        &["unsat", "sat"],
    );
}

/// Defect 2, minimised, bit-vector spelling.  `(distinct b b)` is
/// unsatisfiable on its own; an empty `push`/`pop` pair around nothing at all
/// made the unfixed tree answer `sat`.
#[test]
fn minimal_pushpop_drops_distinct_bv() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (distinct b b))
(push 1)
(pop 1)
(check-sat)
"#,
        &["unsat"],
    );
}

/// Defect 2 is sort-independent: the same script over `Int` (no bit-blaster,
/// no `QF_BV` logic) lost the contradiction in exactly the same way.
#[test]
fn minimal_pushpop_drops_distinct_int() {
    assert_verdicts(
        r#"(declare-const i Int)
(assert (distinct i i))
(push 1)
(pop 1)
(check-sat)
"#,
        &["unsat"],
    );
}

/// Defect 2, in its most direct form: the *same* goal answered `unsat` before
/// the `push`/`pop` pair and `sat` after it.  One script, two contradictory
/// answers, nothing asserted or retracted in between.
#[test]
fn minimal_pushpop_flips_unsat_to_sat() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const b (_ BitVec 8))
(assert (distinct b b))
(check-sat)
(push 1)
(pop 1)
(check-sat)
"#,
        &["unsat", "unsat"],
    );
}

// ---------------------------------------------------------------------------
// 2. The thirteen campaign witnesses
//
// Every script below came out of the 12,000-trial Family A differential
// campaign (`probe A <seed> 1000` for seeds 1..=12, shapes (ii) one-level
// push/pop and (iii) two-level push/pop).  The expected verdict list is the
// campaign's brute-force ground truth over all 65,536 assignments of the two
// width-8 variables, transcribed from each file's own `all_expected=[..]`
// evidence header (`true` = `sat`).
//
// Twelve are wrong `unsat` (defect 1); `wrong_005` is the single wrong `sat`
// and belongs to defect 2 — its base-level assertion is `(distinct b b)`.
// ---------------------------------------------------------------------------

/// Campaign witness `wrong-001.smt2` — WRONG_UNSAT, shape (ii) one-level push/pop,
/// seed 1, trial 610.  Unfixed tree answered `[unsat, unsat]`.
#[test]
fn wrong_001() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (and (xor (bvuge (bvlshr (bvor b #xd5) (bvsub b b)) (bvurem b b)) (distinct (bvor a a) (bvlshr (bvadd b a) (bvor a b)))) (ite (bvsle b (bvsub a #x84)) (= b #x0c) (bvugt a #x30)) (or (= b #x49) (bvugt #x28 b) (= a #x04))))
(push 1)
(assert (ite (and (bvule a a) (bvugt a (bvsub #xb4 (bvneg #x4b)))) (ite (bvslt b b) (= a #x16) (= (bvsub a b) b)) (or (bvule #xbe (bvadd (bvashr b a) (bvsub (bvneg #xba) (bvudiv b a)))) (bvugt (bvshl b b) (bvmul a (bvlshr #xb0 (bvnot a)))) (= a #x07))))
(check-sat)
(pop 1)
(assert (or (or (bvsgt (bvnot (bvand a #xdc)) (bvadd (bvadd (bvashr #xfa #x90) (bvneg b)) (bvxor (bvor b #xb8) (bvand #x94 a)))) (bvuge (bvadd (bvmul #xdd a) (bvashr a b)) (bvnot #x11))) (= a #xd9) (= a #xf0)))
(check-sat)
"#,
        &["unsat", "sat"],
    );
}

/// Campaign witness `wrong-002.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 1, trial 785.  Unfixed tree answered `[sat, unsat, unsat]`.
#[test]
fn wrong_002() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (or (and (bvsge (bvand a #xd8) (bvmul a a)) (bvugt (bvshl a a) a) (bvugt (bvmul (bvashr (bvsub b b) (bvand a a)) b) (bvurem (bvsub #x3f #xae) (bvmul b a)))) (and (= a #xa0) (distinct a (bvlshr a #xa8)))))
(push 1)
(assert (ite (not (bvuge a (bvnot (bvashr #x32 b)))) (and (distinct a b) (bvsle a a)) (ite (bvsle (bvudiv a a) a) (bvugt b (bvshl b a)) (bvslt (bvsub a a) a))))
(check-sat)
(push 1)
(assert (and (or (= b #xff) (= (bvor b (bvlshr (bvshl b a) (bvadd a a))) (bvxor a b)) (bvult a (bvadd (bvxor b a) (bvor a b)))) (or (bvult b (bvudiv a a)) (bvsle (bvxor (bvsub (bvor a a) (bvsub a #x10)) (bvudiv (bvxor a a) (bvmul b #xf9))) (bvneg (bvneg a))) (bvsgt a (bvneg (bvmul (bvxor a #xe9) a))))))
(check-sat)
(pop 1)
(pop 1)
(assert (not (= b #x3c)))
(check-sat)
"#,
        &["sat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-003.smt2` — WRONG_UNSAT, shape (ii) one-level push/pop,
/// seed 2, trial 383.  Unfixed tree answered `[unsat, unsat]`.
#[test]
fn wrong_003() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (and (or (= b #xb3) (= a #x55)) (= a #xc5) (=> (= b #xde) (bvugt a (bvsub a #x62)))))
(push 1)
(assert (= (bvadd (bvnot (bvadd b a)) (bvnot (bvneg a))) #x7b))
(check-sat)
(pop 1)
(assert (not (ite (bvsle (bvxor (bvshl #xac a) (bvshl b b)) (bvxor a #x2f)) (= a #xc1) (bvsge b (bvsub a b)))))
(check-sat)
"#,
        &["unsat", "sat"],
    );
}

/// Campaign witness `wrong-004.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 3, trial 579.  Unfixed tree answered `[sat, unsat, unsat]`.
#[test]
fn wrong_004() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (xor (or (bvsgt (bvashr b (bvand a (bvxor b a))) (bvand (bvor (bvadd #x08 b) (bvnot #x84)) (bvashr b (bvand a a)))) (distinct (bvxor (bvneg (bvand a a)) (bvor (bvnot b) (bvsub #xa4 #xd4))) (bvurem #xae a)) (bvugt (bvshl (bvand (bvor b a) (bvlshr a a)) (bvor (bvurem #xfe b) (bvmul a a))) (bvashr (bvlshr (bvnot b) b) (bvlshr (bvor b b) (bvurem b a))))) (or (distinct (bvxor (bvlshr (bvxor b a) (bvadd b a)) (bvnot #x41)) (bvand #x75 (bvudiv a b))) (= b #x9c) (bvsle #xe9 (bvsub (bvnot (bvashr b a)) b)))))
(push 1)
(assert (bvuge b (bvshl (bvneg (bvadd a a)) (bvxor (bvadd #x6c a) (bvneg a)))))
(check-sat)
(push 1)
(assert (and (not (bvule (bvudiv (bvnot (bvmul b a)) (bvudiv a #x25)) b)) (and (bvsge (bvurem (bvshl b b) (bvsub b #xc3)) (bvxor (bvor (bvmul a b) (bvnot a)) #xf3)) (= (bvneg (bvsub (bvlshr b b) (bvudiv #x35 #xef))) (bvneg a))) (or (bvugt (bvlshr a (bvmul a b)) (bvand (bvadd a b) (bvlshr #xe2 a))) (bvsgt (bvneg (bvxor (bvnot a) (bvmul #x86 a))) (bvsub b b)))))
(check-sat)
(pop 1)
(pop 1)
(assert (and (and (bvuge (bvxor (bvsub (bvnot a) (bvudiv b b)) (bvudiv (bvxor a b) a)) (bvor #x07 #x43)) (bvsge (bvnot #x62) (bvsub #xe6 b))) (not (= (bvmul (bvadd b (bvsub b a)) (bvmul (bvxor a a) (bvor b #xef))) (bvor (bvudiv b #x68) a))) (not (bvule (bvor b a) (bvnot a)))))
(check-sat)
"#,
        &["sat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-005.smt2` — WRONG_SAT, shape (ii) one-level push/pop,
/// seed 5, trial 579.  Unfixed tree answered `[unsat, sat]`.
#[test]
fn wrong_005() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (distinct b b))
(push 1)
(assert (xor (and (bvsgt b (bvsub (bvadd a (bvxor #xe8 a)) (bvshl (bvashr b a) (bvadd a b)))) (distinct (bvnot a) a) (= b #x45)) (and (= b b) (bvult (bvnot a) (bvudiv b (bvurem #x03 #xff))))))
(check-sat)
(pop 1)
(assert (not (not (= a #xaa))))
(check-sat)
"#,
        &["unsat", "unsat"],
    );
}

/// Campaign witness `wrong-006.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 5, trial 962.  Unfixed tree answered `[unsat, unsat, unsat]`.
#[test]
fn wrong_006() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (or (and (= b #x5c) (bvult (bvmul b b) (bvurem (bvshl #xff b) (bvmul a b)))) (bvsgt (bvshl (bvand a a) (bvmul b a)) (bvurem (bvxor (bvand b #x93) (bvashr #x49 a)) (bvneg (bvmul a a)))) (and (bvsle a (bvxor #x26 a)) (= b #xff) (= a #xaf))))
(push 1)
(assert (= a #x8c))
(check-sat)
(push 1)
(assert (and (and (= b #x41) (bvsgt (bvsub (bvadd a b) (bvashr b a)) (bvadd b #x8a)) (= (bvneg (bvor #xea a)) (bvand b a))) (and (bvuge (bvshl (bvor #xf7 #xea) (bvneg b)) #x4e) (= a #x53) (bvsge (bvashr (bvor a a) (bvnot b)) (bvadd #x66 a))) (=> (= a #x2f) (= b #x1c))))
(check-sat)
(pop 1)
(pop 1)
(assert (and (= b #x1a) (=> (= a #x9e) (bvsle (bvlshr #x63 b) a)) (not (bvslt (bvlshr (bvnot (bvshl #x0a b)) (bvsub (bvadd b a) (bvxor #xc6 #x39))) (bvshl (bvxor b #x88) (bvand b #x73))))))
(check-sat)
"#,
        &["unsat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-007.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 6, trial 26.  Unfixed tree answered `[sat, unsat, unsat]`.
#[test]
fn wrong_007() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (xor (and (= a (bvadd (bvlshr b #x49) #x55)) (bvugt a (bvsub (bvashr (bvor a a) (bvor #x54 #x2d)) (bvxor #x76 (bvand b #x64)))) (bvslt (bvadd (bvashr b #xe4) b) (bvxor b (bvxor b a)))) (= a #x1e)))
(push 1)
(assert (bvuge (bvneg a) b))
(check-sat)
(push 1)
(assert (ite (not (= b #x4a)) (and (bvslt a (bvashr (bvudiv a #xfb) (bvneg b))) (bvult (bvmul b (bvlshr b a)) a)) (bvult (bvadd b b) b)))
(check-sat)
(pop 1)
(pop 1)
(assert (=> (bvuge b a) (and (= b #xed) (bvule (bvand (bvashr (bvadd b #x7b) a) (bvsub (bvand b b) (bvashr b b))) (bvor (bvneg (bvsub b #x6e)) (bvxor (bvxor #xfd #xaa) (bvor b b)))) (= (bvadd a #x99) #xed))))
(check-sat)
"#,
        &["sat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-008.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 8, trial 619.  Unfixed tree answered `[sat, unsat, unsat]`.
#[test]
fn wrong_008() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (xor (xor (bvult (bvxor #x78 #xbd) a) (= b #xa3)) (and (bvule a (bvadd (bvudiv (bvmul b b) (bvnot b)) (bvshl (bvsub b b) (bvmul #x6c a)))) (distinct a b) (bvslt (bvor a #xb9) (bvashr (bvadd a b) (bvnot b))))))
(push 1)
(assert (xor (not (bvslt (bvor (bvshl (bvand a b) (bvsub a #x14)) (bvand (bvmul #xa9 a) b)) (bvadd (bvadd a #x85) (bvor #xb2 a)))) (=> (= a #x79) (= b #x56))))
(check-sat)
(push 1)
(assert (and (xor (= b #x44) (bvsle (bvnot a) (bvneg #x69))) (and (bvslt (bvurem b #x11) (bvudiv (bvneg b) (bvsub #x79 a))) (bvsge #x9c a) (bvsgt (bvsub b a) b))))
(check-sat)
(pop 1)
(pop 1)
(assert (=> (not (bvsgt b (bvashr (bvneg (bvsub a b)) (bvadd (bvurem a #x40) b)))) (ite (bvult #x4b #x22) (bvugt (bvneg (bvor (bvor a b) (bvand b #x0c))) a) (= (bvxor b b) (bvashr (bvneg a) (bvshl a a))))))
(check-sat)
"#,
        &["sat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-009.smt2` — WRONG_UNSAT, shape (ii) one-level push/pop,
/// seed 9, trial 302.  Unfixed tree answered `[sat, unsat]`.
#[test]
fn wrong_009() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (and (xor (= a #x3b) (= b (bvmul (bvadd b a) (bvand b b)))) (ite (= (bvsub (bvnot (bvlshr a a)) (bvnot (bvlshr b b))) b) (= (bvor (bvmul #xfa a) (bvneg #x87)) a) (bvult (bvadd #xef b) (bvor #x90 b)))))
(push 1)
(assert (bvule b (bvudiv (bvand (bvmul a a) (bvnot #x96)) (bvneg (bvsub #x5d a)))))
(check-sat)
(pop 1)
(assert (and (or (= (bvmul b b) a) (= b #x63)) (or (bvule (bvneg #xe5) (bvshl a a)) (bvsgt a (bvadd (bvand b b) (bvsub #x60 a))) (= a #x9c))))
(check-sat)
"#,
        &["sat", "sat"],
    );
}

/// Campaign witness `wrong-010.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 9, trial 361.  Unfixed tree answered `[sat, unsat, unsat]`.
// Skipped only in a `debug_assertions` build.  This is the most expensive of
// the thirteen scripts: 1.95 s as a release binary, but 363 s with debug
// assertions on, because `Solver::debug_check_invariants` re-sweeps the whole
// clause database at every state transition (~186x).  The workspace's own gate
// is a release run (`cargo nextest run --release --workspace`), where this runs
// normally; a dev-profile run would otherwise be terminated by nextest's
// 180 s cap and report a timeout instead of a verdict.  Run it explicitly with
// `cargo nextest run --release -p oxiz-solver --test bv_scope_rollback_pushpop`.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "363 s under debug_assertions vs 1.95 s in release; run with --release"
)]
fn wrong_010() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (and (or (= #x32 (bvlshr a b)) (= a #x05) (= b #x82)) (or (= b #x9b) (bvsle (bvnot b) b) (bvslt (bvashr (bvadd #xc9 a) (bvxor b a)) b))))
(push 1)
(assert (or (= b #x2a) (and (distinct (bvsub (bvmul b a) b) (bvadd (bvshl (bvlshr a b) (bvor a b)) (bvor (bvurem b a) (bvsub #xf6 #x66)))) (bvuge (bvsub a a) (bvlshr b #x10))) (ite (= (bvnot #x1f) (bvand (bvnot b) (bvlshr (bvxor #x80 #xb9) (bvsub b #x33)))) (bvsge (bvmul b b) (bvneg b)) (bvult a (bvadd b a)))))
(check-sat)
(push 1)
(assert (xor (ite (bvugt #x1e (bvurem #xe7 a)) (bvsgt #x99 #x1b) (= (bvnot a) (bvmul (bvashr #x85 a) (bvsub a #xd7)))) (xor (= a #xbf) (bvuge b (bvadd b a)))))
(check-sat)
(pop 1)
(pop 1)
(assert (and (ite (bvsge (bvashr (bvlshr (bvmul b a) a) b) (bvneg (bvsub (bvxor a #x73) (bvsub a b)))) (bvslt (bvadd (bvand (bvadd b #x6e) (bvsub #x43 b)) (bvudiv (bvadd #x9c b) (bvshl b b))) (bvxor #x47 #x56)) (bvsgt #xb3 (bvnot (bvnot (bvsub a #x47))))) (ite (bvsgt (bvand (bvand b b) (bvsub b b)) (bvadd (bvsub a b) (bvlshr a #x74))) (bvult (bvudiv (bvadd (bvneg b) (bvneg b)) b) a) (bvuge (bvurem (bvlshr b b) (bvand b a)) a))))
(check-sat)
"#,
        &["sat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-011.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 10, trial 509.  Unfixed tree answered `[sat, unsat, unsat]`.
#[test]
fn wrong_011() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (or (=> (= (bvashr (bvor (bvnot b) (bvnot a)) (bvnot (bvadd #xaa #x38))) (bvxor (bvxor a a) (bvneg a))) (bvugt (bvashr (bvurem (bvand a a) #x80) b) a)) (not (bvslt (bvneg a) a))))
(push 1)
(assert (bvslt (bvadd (bvlshr #xbc #xb0) (bvashr a a)) (bvor b (bvsub (bvmul b #x6c) (bvshl b #x6e)))))
(check-sat)
(push 1)
(assert (ite (and (bvslt (bvxor a b) (bvshl b (bvadd #x7d b))) (bvult (bvurem #xaf b) b) (distinct (bvlshr (bvand b a) (bvadd b b)) (bvor (bvsub (bvsub #x7f #xb6) (bvnot b)) (bvneg (bvxor #xa5 a))))) (and (bvule (bvadd #x6a b) (bvand (bvsub b #xd9) (bvshl b #x7a))) (= a a) (bvslt (bvand (bvnot a) (bvxor a b)) (bvudiv (bvsub (bvashr #x98 b) (bvadd b #x1a)) (bvor #x3e (bvsub a a))))) (and (= (bvor b #x1a) b) (= b #x14) (bvsle (bvadd a b) (bvsub (bvnot (bvand b b)) (bvmul (bvsub a b) (bvneg #x13)))))))
(check-sat)
(pop 1)
(pop 1)
(assert (and (xor (bvsgt b b) (bvsge a #x4a)) (or (distinct b #xb4) (bvugt b (bvlshr a (bvsub b a))))))
(check-sat)
"#,
        &["sat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-012.smt2` — WRONG_UNSAT, shape (iii) two-level push/pop,
/// seed 10, trial 862.  Unfixed tree answered `[sat, unsat, unsat]`.
#[test]
fn wrong_012() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (or (or (= a #xca) (= b #xd9)) (and (bvule (bvudiv a b) a) (= (bvudiv #xcc b) (bvashr (bvadd (bvlshr b a) (bvadd #x72 a)) (bvadd (bvashr b b) (bvxor b b)))))))
(push 1)
(assert (not (distinct a (bvxor (bvneg (bvudiv #xed a)) b))))
(check-sat)
(push 1)
(assert (not (not (bvult (bvmul (bvand (bvsub b #xf3) b) #x59) (bvudiv (bvlshr #x5c b) b)))))
(check-sat)
(pop 1)
(pop 1)
(assert (= b #x72))
(check-sat)
"#,
        &["sat", "unsat", "sat"],
    );
}

/// Campaign witness `wrong-013.smt2` — WRONG_UNSAT, shape (ii) one-level push/pop,
/// seed 12, trial 412.  Unfixed tree answered `[unsat, unsat]`.
#[test]
fn wrong_013() {
    assert_verdicts(
        r#"(set-logic QF_BV)
(declare-const a (_ BitVec 8))
(declare-const b (_ BitVec 8))
(assert (ite (ite (bvugt (bvand (bvand a b) (bvshl a #x48)) (bvand b a)) (bvuge (bvnot a) (bvashr #x3c a)) (= a b)) (=> (bvule (bvmul b a) (bvadd b a)) (= a #x81)) (or (= b #xbf) (= b #x75))))
(push 1)
(assert (not (not (= b #x71))))
(check-sat)
(pop 1)
(assert (ite (=> (= b #x96) (= b #xd0)) (and (bvsgt a (bvor (bvashr (bvlshr #x10 b) (bvor #xbc a)) (bvadd (bvshl b #xf4) a))) (= b #xdf) (= b #x33)) (not (bvslt (bvsub (bvmul (bvsub a #x53) (bvsub b b)) b) (bvand b a)))))
(check-sat)
"#,
        &["unsat", "sat"],
    );
}

// ---------------------------------------------------------------------------
// 3. A bounded, self-contained differential test
//
// The thirteen witnesses above pin thirteen *known* counterexamples.  This
// section re-runs the search that found them, in miniature and in-process:
// 200 trials of the two push/pop shapes over two width-8 bit-vectors, each
// scored against a brute-force oracle that evaluates the generated formula
// over all 65,536 assignments with its own SMT-LIB `FixedSizeBitVectors`
// semantics.  Nothing here consults the solver for the ground truth, so a
// verdict that disagrees is a solver bug by construction.
//
// Deterministic: one fixed seed, a counter-based PRNG, and no clock or thread
// input anywhere.  The generator, the evaluator and the printer are ports of
// the external probe crate that produced the witnesses, including its
// seed-avalanche fix (splitmix64 advances its state by a *fixed* increment, so
// seeding with `seed + K` directly would make consecutive seeds produce the
// same stream shifted by one draw instead of independent streams).
// ---------------------------------------------------------------------------

/// splitmix64, with the seed run through the finalizer first (see above).
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        Rng {
            state: z ^ (z >> 33),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn byte(&mut self) -> u8 {
        self.next_u64() as u8
    }

    fn weighted(&mut self, weights: &[u32]) -> usize {
        let total: u32 = weights.iter().sum();
        let mut r = self.below(u64::from(total)) as u32;
        for (i, w) in weights.iter().enumerate() {
            if r < *w {
                return i;
            }
            r -= *w;
        }
        weights.len() - 1
    }
}

/// A width-8 bit-vector term over the two free variables `a` and `b`.
#[derive(Clone)]
enum Bv {
    VarA,
    VarB,
    Const(u8),
    Un(&'static str, Box<Bv>),
    Bin(&'static str, Box<Bv>, Box<Bv>),
}

/// A quantifier-free Boolean formula over bit-vector atoms.
#[derive(Clone)]
enum F {
    Atom(&'static str, Bv, Bv),
    Not(Box<F>),
    And(Vec<F>),
    Or(Vec<F>),
    Imp(Box<F>, Box<F>),
    Xor(Box<F>, Box<F>),
    Ite(Box<F>, Box<F>, Box<F>),
}

const BV_BIN: [&str; 11] = [
    "bvadd", "bvsub", "bvmul", "bvand", "bvor", "bvxor", "bvshl", "bvlshr", "bvashr", "bvudiv",
    "bvurem",
];
const BV_BIN_W: [u32; 11] = [4, 4, 3, 3, 3, 3, 2, 2, 2, 1, 1];
const BV_UN: [&str; 2] = ["bvnot", "bvneg"];
const ATOMS: [&str; 10] = [
    "=", "distinct", "bvule", "bvult", "bvuge", "bvugt", "bvsle", "bvslt", "bvsge", "bvsgt",
];
const ATOMS_W: [u32; 10] = [4, 2, 3, 3, 3, 3, 2, 2, 2, 2];

/// SMT-LIB `FixedSizeBitVectors` semantics at width 8, written out here rather
/// than delegated to the workspace so the oracle is genuinely independent of
/// the code under test: a shift at or beyond the width yields `0` (logical) or
/// the sign fill (arithmetic), division by zero is all-ones, and remainder by
/// zero is the dividend.
fn eval_bv(t: &Bv, a: u8, b: u8) -> u8 {
    match t {
        Bv::VarA => a,
        Bv::VarB => b,
        Bv::Const(c) => *c,
        Bv::Un(op, x) => {
            let u = eval_bv(x, a, b);
            match *op {
                "bvnot" => !u,
                "bvneg" => u.wrapping_neg(),
                _ => unreachable!("unary bv op {op}"),
            }
        }
        Bv::Bin(op, x, y) => {
            let u = eval_bv(x, a, b);
            let v = eval_bv(y, a, b);
            match *op {
                "bvadd" => u.wrapping_add(v),
                "bvsub" => u.wrapping_sub(v),
                "bvmul" => u.wrapping_mul(v),
                "bvand" => u & v,
                "bvor" => u | v,
                "bvxor" => u ^ v,
                "bvshl" => {
                    if v >= 8 {
                        0
                    } else {
                        u.wrapping_shl(u32::from(v))
                    }
                }
                "bvlshr" => {
                    if v >= 8 {
                        0
                    } else {
                        u.wrapping_shr(u32::from(v))
                    }
                }
                "bvashr" => {
                    if v >= 8 {
                        if u & 0x80 != 0 { 0xff } else { 0x00 }
                    } else {
                        ((u as i8).wrapping_shr(u32::from(v))) as u8
                    }
                }
                // SMT-LIB `FixedSizeBitVectors`: division by zero is all-ones
                // and remainder by zero is the dividend, so both are total.
                "bvudiv" => u.checked_div(v).unwrap_or(0xff),
                "bvurem" => u.checked_rem(v).unwrap_or(u),
                _ => unreachable!("binary bv op {op}"),
            }
        }
    }
}

fn eval_f(f: &F, a: u8, b: u8) -> bool {
    match f {
        F::Atom(op, x, y) => {
            let u = eval_bv(x, a, b);
            let v = eval_bv(y, a, b);
            let (su, sv) = (u as i8, v as i8);
            match *op {
                "=" => u == v,
                "distinct" => u != v,
                "bvule" => u <= v,
                "bvult" => u < v,
                "bvuge" => u >= v,
                "bvugt" => u > v,
                "bvsle" => su <= sv,
                "bvslt" => su < sv,
                "bvsge" => su >= sv,
                "bvsgt" => su > sv,
                _ => unreachable!("atom op {op}"),
            }
        }
        F::Not(x) => !eval_f(x, a, b),
        F::And(xs) => xs.iter().all(|x| eval_f(x, a, b)),
        F::Or(xs) => xs.iter().any(|x| eval_f(x, a, b)),
        F::Imp(x, y) => !eval_f(x, a, b) || eval_f(y, a, b),
        F::Xor(x, y) => eval_f(x, a, b) != eval_f(y, a, b),
        F::Ite(c, t, e) => {
            if eval_f(c, a, b) {
                eval_f(t, a, b)
            } else {
                eval_f(e, a, b)
            }
        }
    }
}

fn print_bv(t: &Bv, out: &mut String) {
    use core::fmt::Write as _;
    match t {
        Bv::VarA => out.push('a'),
        Bv::VarB => out.push('b'),
        Bv::Const(c) => {
            let _ = write!(out, "#x{c:02x}");
        }
        Bv::Un(op, x) => {
            let _ = write!(out, "({op} ");
            print_bv(x, out);
            out.push(')');
        }
        Bv::Bin(op, x, y) => {
            let _ = write!(out, "({op} ");
            print_bv(x, out);
            out.push(' ');
            print_bv(y, out);
            out.push(')');
        }
    }
}

fn print_f(f: &F, out: &mut String) {
    use core::fmt::Write as _;
    match f {
        F::Atom(op, x, y) => {
            let _ = write!(out, "({op} ");
            print_bv(x, out);
            out.push(' ');
            print_bv(y, out);
            out.push(')');
        }
        F::Not(x) => {
            out.push_str("(not ");
            print_f(x, out);
            out.push(')');
        }
        F::And(xs) => {
            out.push_str("(and");
            for x in xs {
                out.push(' ');
                print_f(x, out);
            }
            out.push(')');
        }
        F::Or(xs) => {
            out.push_str("(or");
            for x in xs {
                out.push(' ');
                print_f(x, out);
            }
            out.push(')');
        }
        F::Imp(x, y) => {
            out.push_str("(=> ");
            print_f(x, out);
            out.push(' ');
            print_f(y, out);
            out.push(')');
        }
        F::Xor(x, y) => {
            out.push_str("(xor ");
            print_f(x, out);
            out.push(' ');
            print_f(y, out);
            out.push(')');
        }
        F::Ite(c, t, e) => {
            out.push_str("(ite ");
            print_f(c, out);
            out.push(' ');
            print_f(t, out);
            out.push(' ');
            print_f(e, out);
            out.push(')');
        }
    }
}

fn gen_bv(rng: &mut Rng, depth: u32) -> Bv {
    if depth == 0 || rng.below(5) == 0 {
        return match rng.weighted(&[4, 4, 3]) {
            0 => Bv::VarA,
            1 => Bv::VarB,
            _ => Bv::Const(rng.byte()),
        };
    }
    if rng.below(6) == 0 {
        let op = BV_UN[rng.below(2) as usize];
        Bv::Un(op, Box::new(gen_bv(rng, depth - 1)))
    } else {
        let op = BV_BIN[rng.weighted(&BV_BIN_W)];
        let x = gen_bv(rng, depth - 1);
        let y = gen_bv(rng, depth - 1);
        Bv::Bin(op, Box::new(x), Box::new(y))
    }
}

/// Largest bit-vector operand tree an atom may carry, exclusive.
///
/// The external probe crate that found the witnesses draws this from `0..4`.
/// Here it is `0..2`, which is the one deliberate difference from that
/// generator and is purely a cost control: a depth-3 operand tree routinely
/// contains a nested 8-bit `bvmul` or `bvudiv`, whose bit-blasted circuit
/// dominates the whole trial.  Capping it leaves this test's subject — Boolean
/// structure over bit-vector atoms across `push`/`pop`, which is where both
/// defects live — completely intact while making 200 trials affordable in a
/// `debug_assertions` build, where `Solver::debug_check_invariants` re-sweeps
/// the entire clause database at every state transition (measured ~190x slower
/// than the same scripts in a release build).  The uncapped generator still
/// runs, at 12,000 trials, in the external campaign.
const OPERAND_DEPTH_BOUND: u64 = 2;

/// One atom.  With probability 1/4 it is a "pin" (`var = constant`), which is
/// what makes a useful share of the generated conjunctions genuinely
/// unsatisfiable instead of trivially satisfiable.
fn gen_atom(rng: &mut Rng) -> F {
    if rng.below(4) == 0 {
        let v = if rng.below(2) == 0 {
            Bv::VarA
        } else {
            Bv::VarB
        };
        return F::Atom("=", v, Bv::Const(rng.byte()));
    }
    let op = ATOMS[rng.weighted(&ATOMS_W)];
    let d1 = rng.below(OPERAND_DEPTH_BOUND) as u32;
    let d2 = rng.below(OPERAND_DEPTH_BOUND) as u32;
    F::Atom(op, gen_bv(rng, d1), gen_bv(rng, d2))
}

fn gen_f(rng: &mut Rng, depth: u32) -> F {
    if depth == 0 {
        return gen_atom(rng);
    }
    match rng.weighted(&[5, 4, 3, 2, 2, 2, 3]) {
        0 => {
            let n = 2 + rng.below(2) as usize;
            F::And((0..n).map(|_| gen_f(rng, depth - 1)).collect())
        }
        1 => {
            let n = 2 + rng.below(2) as usize;
            F::Or((0..n).map(|_| gen_f(rng, depth - 1)).collect())
        }
        2 => F::Not(Box::new(gen_f(rng, depth - 1))),
        3 => F::Imp(
            Box::new(gen_f(rng, depth - 1)),
            Box::new(gen_f(rng, depth - 1)),
        ),
        4 => F::Xor(
            Box::new(gen_f(rng, depth - 1)),
            Box::new(gen_f(rng, depth - 1)),
        ),
        5 => F::Ite(
            Box::new(gen_f(rng, depth - 1)),
            Box::new(gen_f(rng, depth - 1)),
            Box::new(gen_f(rng, depth - 1)),
        ),
        _ => gen_atom(rng),
    }
}

/// Brute force over all 65,536 assignments, early-exiting on the first model.
/// `mask` permutes the scan order so the oracle never systematically tries
/// `a = 0, b = 0` first (which would hide an off-by-one in the solver's model
/// search behind a lucky common answer).
fn truth_sat(fs: &[&F], mask: u16) -> bool {
    for i in 0..=u16::MAX {
        let idx = i ^ mask;
        let a = (idx >> 8) as u8;
        let b = (idx & 0xff) as u8;
        if fs.iter().all(|f| eval_f(f, a, b)) {
            return true;
        }
    }
    false
}

const DECLS_8: &str =
    "(set-logic QF_BV)\n(declare-const a (_ BitVec 8))\n(declare-const b (_ BitVec 8))\n";

/// 200 trials over the two push/pop shapes, scored against the brute-force
/// oracle.  Any mismatch fails with the offending script and both verdict
/// lists, so the failure message is itself a minimisable reproducer.
#[test]
fn bounded_differential_pushpop_matches_brute_force_oracle() {
    const TRIALS: usize = 200;
    // Seed 2, chosen by scanning seeds 0..12 against the *unfixed* sources:
    // this one exposes two wrong `unsat` answers there, so the test genuinely
    // witnesses defect 1 rather than merely guarding against it.  (At this
    // trial count most seeds see none: the campaign's measured rate was 13
    // wrong verdicts in 24,054, so 200 trials is a guard, not a search.)
    const SEED: u64 = 2;

    let mut rng = Rng::new(SEED);
    let mut wrong_sat = 0usize;
    let mut wrong_unsat = 0usize;
    let mut unknown = 0usize;
    let mut truth_unsat_seen = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for trial in 0..TRIALS {
        let mask = rng.next_u64() as u16;
        // Alternate the two push/pop shapes deterministically so both are
        // exercised the same number of times whatever the PRNG does.
        let two_level = trial % 2 == 1;

        let mut script = String::from(DECLS_8);
        let fa = gen_f(&mut rng, 2);
        let fb = gen_f(&mut rng, 2);
        let fc = gen_f(&mut rng, 2);
        let fd = gen_f(&mut rng, 2);

        let expected: Vec<&str> = if two_level {
            // (iii) assert A / push / assert B / check / push / assert C /
            //       check / pop / pop / assert D / check
            script.push_str("(assert ");
            print_f(&fa, &mut script);
            script.push_str(")\n(push 1)\n(assert ");
            print_f(&fb, &mut script);
            script.push_str(")\n(check-sat)\n(push 1)\n(assert ");
            print_f(&fc, &mut script);
            script.push_str(")\n(check-sat)\n(pop 1)\n(pop 1)\n(assert ");
            print_f(&fd, &mut script);
            script.push_str(")\n(check-sat)\n");
            [
                truth_sat(&[&fa, &fb], mask),
                truth_sat(&[&fa, &fb, &fc], mask),
                truth_sat(&[&fa, &fd], mask),
            ]
            .iter()
            .map(|t| if *t { "sat" } else { "unsat" })
            .collect()
        } else {
            // (ii) assert A / push / assert B / check / pop / assert C / check
            script.push_str("(assert ");
            print_f(&fa, &mut script);
            script.push_str(")\n(push 1)\n(assert ");
            print_f(&fb, &mut script);
            script.push_str(")\n(check-sat)\n(pop 1)\n(assert ");
            print_f(&fc, &mut script);
            script.push_str(")\n(check-sat)\n");
            [truth_sat(&[&fa, &fb], mask), truth_sat(&[&fa, &fc], mask)]
                .iter()
                .map(|t| if *t { "sat" } else { "unsat" })
                .collect()
        };

        truth_unsat_seen += expected.iter().filter(|t| **t == "unsat").count();

        let got = verdicts(&script);
        assert_eq!(
            got.len(),
            expected.len(),
            "trial {trial}: expected {} verdicts, got {got:?}\n{script}",
            expected.len()
        );
        for (k, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            match (g.as_str(), *e) {
                ("unknown", _) => unknown += 1,
                ("sat", "unsat") => {
                    wrong_sat += 1;
                    failures.push(format!(
                        "trial {trial} check-sat#{k}: WRONG_SAT (oracle says unsat)\n\
                         expected {expected:?} got {got:?}\n{script}"
                    ));
                }
                ("unsat", "sat") => {
                    wrong_unsat += 1;
                    failures.push(format!(
                        "trial {trial} check-sat#{k}: WRONG_UNSAT (oracle found a model)\n\
                         expected {expected:?} got {got:?}\n{script}"
                    ));
                }
                _ => {}
            }
        }
    }

    // The oracle must actually have produced unsatisfiable questions, or a
    // "0 wrong" result would be vacuous.
    assert!(
        truth_unsat_seen >= 20,
        "the generator produced only {truth_unsat_seen} truly-unsat questions in \
         {TRIALS} trials; the test would be vacuous"
    );
    assert_eq!(
        unknown, 0,
        "every question here is a decidable width-8 bit-vector problem; \
         {unknown} came back `unknown`"
    );
    assert!(
        failures.is_empty(),
        "{wrong_sat} wrong `sat` and {wrong_unsat} wrong `unsat` in {TRIALS} trials\n\n{}",
        failures.join("\n\n")
    );
}
