//! Round-4 re-fix pass 12 — `#P2b-60`: an array fully determined by a ground
//! equality, read under a binder above the finite-expansion budget, was
//! answered a WRONG `sat`.
//!
//! Found on 2026-09-28 by the rebuilt `score.py` (decision (42)(c)): its pin
//! re-check asserts a published model into the quantified script and re-solves
//! it, and on 19 of the 120 regenerated width-7/8 pairs the tree *confirmed* a
//! model the solver-free exact evaluation refutes.  The minimal shape needs no
//! model at all:
//!
//! ```text
//! (assert (= a1 (store ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0) #b0000000 #b1)))
//! (assert (forall ((i (_ BitVec 7))) (= (select a1 i) #b1)))
//! ```
//!
//! `a1` reads `#b0` at `#b0000001`, so this is `unsat`; `c4b04b7` and crates.io
//! 0.3.3 answer `unknown`, the tree answered `sat`.  The Int-index spelling
//! (`wsat6` below) was a wrong `sat` on all three.  Cause and repair:
//! `mbqi::sat_certify::unnamed_region` — the relevant instantiation set of an
//! array index position carries a representative of the region no index term
//! names, so the certifier's saturation can no longer vouch for a universal
//! that has only been checked where the goal happens to read.
//!
//! Every script here is decided by construction (a store over a constant array
//! pins every index), so no oracle is needed; each refutation has a
//! satisfiable neighbour beside it so "answer `unsat` to everything" fails as
//! loudly as the wrong `sat` did.

use oxiz_solver::Context;

fn run(script: &str) -> Vec<String> {
    let mut ctx = Context::new();
    match ctx.execute_script(script) {
        Ok(lines) => lines,
        Err(err) => vec![format!("(error \"{err}\")")],
    }
}

/// The last `sat`/`unsat`/`unknown` line, or `"none"`.
fn verdict(lines: &[String]) -> String {
    lines
        .iter()
        .rev()
        .find(|line| matches!(line.as_str(), "sat" | "unsat" | "unknown"))
        .cloned()
        .unwrap_or_else(|| "none".to_string())
}

fn assert_verdict(script: &str, expected: &str, why: &str) {
    let lines = run(script);
    assert_eq!(
        verdict(&lines),
        expected,
        "{why}\n--- script ---\n{script}--- response ---\n{}",
        lines.join("\n")
    );
}

const BV7: &str = "(declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n\
     (declare-const b (Array (_ BitVec 7) (_ BitVec 1)))\n";

const K0: &str = "((as const (Array (_ BitVec 7) (_ BitVec 1))) #b0)";
const K1: &str = "((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1)";

/// The three bit-vector spellings of the family found on 2026-09-28
/// (`corpus/named/wsat1`, `wsat3`, `wsat4`): each was `sat` on the tree and
/// `unknown` on `c4b04b7` and 0.3.3.
#[test]
fn an_array_pinned_by_a_ground_equality_is_refuted_under_a_binder() {
    let cases = [
        (
            "wsat1: the universal reads the pinned array directly",
            format!(
                "(assert (= a1 (store {K0} #b0000000 #b1)))\n\
                 (assert (forall ((i (_ BitVec 7))) (= (select a1 i) #b1)))\n"
            ),
        ),
        (
            "wsat3: the pinned array against an array constant",
            format!(
                "(assert (= a1 (store {K0} #b0000000 #b1)))\n\
                 (assert (forall ((i (_ BitVec 7))) (distinct (select {K0} i) (select a1 i))))\n"
            ),
        ),
        (
            "wsat4: pinned through a second equality",
            format!(
                "(assert (= a1 (store b #b0000000 #b1)))\n\
                 (assert (= b {K0}))\n\
                 (assert (forall ((i (_ BitVec 7))) (= (select a1 i) #b1)))\n"
            ),
        ),
    ];
    for (what, body) in cases {
        assert_verdict(
            &format!("(set-logic ALL)\n{BV7}{body}(check-sat)\n"),
            "unsat",
            &format!("#P2b-60 ({what}): `a1` reads `#b0` at `#b0000001`"),
        );
    }
}

/// The satisfiable neighbours: the same shapes with the constant array's
/// default flipped, so the universal holds at every index.
#[test]
fn the_satisfiable_neighbours_of_the_pinned_family_stay_sat() {
    for body in [
        format!(
            "(assert (= a1 (store {K1} #b0000000 #b1)))\n\
             (assert (forall ((i (_ BitVec 7))) (= (select a1 i) #b1)))\n"
        ),
        format!(
            "(assert (= a1 (store {K1} #b0000000 #b1)))\n\
             (assert (forall ((i (_ BitVec 7))) (distinct (select {K0} i) (select a1 i))))\n"
        ),
        format!(
            "(assert (= a1 (store b #b0000000 #b1)))\n\
             (assert (= b {K1}))\n\
             (assert (forall ((i (_ BitVec 7))) (= (select a1 i) #b1)))\n"
        ),
    ] {
        assert_verdict(
            &format!("(set-logic ALL)\n{BV7}{body}(check-sat)\n"),
            "sat",
            "the pinned array reads `#b1` everywhere; the universal holds",
        );
    }
}

/// The `Int`-index spelling (`corpus/named/wsat6`) — a wrong `sat` on
/// `c4b04b7`, on crates.io 0.3.3 and on the tree alike — and the ordered
/// guards `Int` admits, whose regions each need their own representative.
#[test]
fn an_int_indexed_pinned_array_is_refuted_in_every_guard_region() {
    let decls = "(declare-const a2 (Array Int Int))\n";
    let pinned = "(assert (= a2 (store ((as const (Array Int Int)) 0) 5 1)))\n";
    for (what, quantifier, expected) in [
        (
            "wsat6 shape: false at every index but the stored one",
            "(forall ((i Int)) (= (select a2 i) 1))",
            "unsat",
        ),
        (
            "a guard that admits the store index",
            "(forall ((i Int)) (=> (>= i 5) (= (select a2 i) 0)))",
            "unsat",
        ),
        (
            "a guard below the store index",
            "(forall ((i Int)) (=> (< i 5) (= (select a2 i) 1)))",
            "unsat",
        ),
        (
            "a guard that excludes the store index",
            "(forall ((i Int)) (=> (> i 5) (= (select a2 i) 0)))",
            "sat",
        ),
    ] {
        assert_verdict(
            &format!("(set-logic ALL)\n{decls}{pinned}(assert {quantifier})\n(check-sat)\n"),
            expected,
            &format!("#P2b-60, Int index ({what})"),
        );
    }
}

/// The `Real` and declared-sort spellings (`corpus/named/wreal`, `wusort2`),
/// found while closing the bit-vector one: `wreal` was a wrong `sat` on the
/// tree (`unknown` on `c4b04b7` and 0.3.3), `wusort2` on all three.  Over a
/// declared sort the universe needs no element beyond the ground terms the
/// goal spells, so the completed relevant set is every one of them; without
/// `v`, the certifier's projection mapped `v` onto `u`, where the `store`
/// writes `1`.  The satisfiable neighbour (`u` alone, which a one-element
/// universe satisfies) stays `sat`, and so does the `Bool`-index spelling's.
#[test]
fn the_real_and_declared_sort_spellings_are_refuted_and_their_neighbours_stay_sat() {
    for (what, script, expected) in [
        (
            "wreal: Real index",
            "(declare-const a (Array Real Int))
             (assert (= a (store ((as const (Array Real Int)) 0) 0.0 1)))
             (assert (forall ((x Real)) (= (select a x) 1)))
",
            "unsat",
        ),
        (
            "wusort2: declared index sort, two distinct elements",
            "(declare-sort U 0)
(declare-const u U)
(declare-const v U)
             (assert (distinct u v))
             (declare-const a (Array U Int))
             (assert (= a (store ((as const (Array U Int)) 0) u 1)))
             (assert (forall ((x U)) (= (select a x) 1)))
",
            "unsat",
        ),
        (
            "wusort: one named element, a one-element universe satisfies it",
            "(declare-sort U 0)
(declare-const u U)
             (declare-const a (Array U Int))
             (assert (= a (store ((as const (Array U Int)) 0) u 1)))
             (assert (forall ((x U)) (= (select a x) 1)))
",
            "sat",
        ),
        (
            "wbool: Bool index",
            "(declare-const a (Array Bool Int))
             (assert (= a (store ((as const (Array Bool Int)) 0) true 1)))
             (assert (forall ((x Bool)) (= (select a x) 1)))
",
            "unsat",
        ),
    ] {
        assert_verdict(
            &format!("(set-logic ALL)\n{script}(check-sat)\n"),
            expected,
            &format!("#P2b-60 ({what})"),
        );
    }
}

/// The pin re-check of `corpus/qmbqi120/q0000` (seed 20260928) with the
/// model the tree published before re-fix pass 12 asserted into it — the
/// shape `score.py` counted as `pinned_wrong_sat`.  `a1` is pinned to a store
/// over the all-`#b0` constant and the universal demands `a1[i] ≠ #b0`
/// everywhere.
#[test]
fn a_falsifying_model_pinned_into_its_own_script_is_refuted() {
    assert_verdict(
        &format!(
            "(set-logic ALL)\n\
             (declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\n\
             (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n\
             (declare-const a2 (Array (_ BitVec 7) (_ BitVec 1)))\n\
             (declare-const p Bool)\n(declare-const q Bool)\n\
             (declare-const d (_ BitVec 1))\n\
             (assert (forall ((i!q (_ BitVec 7))) (distinct (select {K0} i!q) (select a1 i!q))))\n\
             (assert (= a0 {K0}))\n\
             (assert (= a1 (store {K0} #b0000000 #b1)))\n\
             (assert (= a2 {K0}))\n\
             (assert (= p false))\n(assert (= q false))\n(assert (= d #b0))\n\
             (check-sat)\n"
        ),
        "unsat",
        "the pinned `a1` reads `#b0` at `#b0000001`, where the universal \
         demands `#b1`",
    );
}

/// The price of the repair, bought back: `corpus/qeq120/q0033` (gen_qmbqi.py
/// seed 20260931), verbatim.
///
/// HEAD `c702310` answers `sat` in 12 ms (16 refinement rounds, 972 embedded
/// checks); with the unnamed-region instances added to every MBQI round it
/// gave no answer in 60 s — the extra index term is what an array search over
/// `ite`-selected bases pays for most.  The instances are therefore added only
/// when the relevant set is saturated, and at that point a certified model
/// completion (`Solver::certify_at_mbqi_saturation`) is tried first, since a
/// `sat` that rests on the certificate needs none of them.  Measured: `sat`
/// with the same 16 rounds / 972 checks as HEAD; with the saturation-time
/// completion mutated away, TIMEOUT at 20 s.
#[test]
fn a_corpus_member_the_unnamed_region_would_have_cost_is_decided() {
    assert_verdict(
        "(set-logic ALL)\n\
         (declare-const a0 (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const a1 (Array (_ BitVec 7) (_ BitVec 1)))\n\
         (declare-const p Bool)\n\
         (declare-const q Bool)\n\
         (declare-const d (_ BitVec 1))\n\
         (assert (= a1 (store ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1) #b1000010 #b1)))\n\
         (assert (or (= #b0 (select (store a1 #b1001001 #b0) #b0101011)) (= #b1 (bvxor (select \
         (ite q (store a0 #b1001010 #b0) (ite q a1 a1)) #b1100001) #b1))))\n\
         (assert (forall ((i!q (_ BitVec 7))) (= (select (ite q a0 a1) i!q) (select (ite p a0 \
         ((as const (Array (_ BitVec 7) (_ BitVec 1))) #b1)) #b1011000))))\n\
         (check-sat)\n",
        "sat",
        "`c4b04b7` and HEAD `c702310` answer `sat`; the unnamed-region repair \
         must not cost it",
    );
}
