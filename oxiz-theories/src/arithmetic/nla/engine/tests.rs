// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the nonlinear engine, kept in a sibling file the way
//! `arithmetic/simplex` and the other `nla` modules do.
//!
//! The tests are written against [`check_assertions`] rather than against
//! [`NlaEngine`] directly wherever the property under test is about the *public
//! contract* — that is where the incompleteness gate and the witness
//! re-verification live, and testing one layer down would miss both.

use super::super::{NlaConfig, NlaVerdict, check_assertions};
use super::*;
use num_bigint::BigInt;
use oxiz_core::ast::{TermId, TermManager};

// --- construction helpers ---------------------------------------------------

fn int_var(m: &mut TermManager, name: &str) -> TermId {
    let s = m.sorts.int_sort;
    m.mk_var(name, s)
}

fn ic(m: &mut TermManager, n: i64) -> TermId {
    m.mk_int(BigInt::from(n))
}

/// `lhs >= rhs` as a term.
fn ge(m: &mut TermManager, lhs: TermId, rhs: i64) -> TermId {
    let r = ic(m, rhs);
    m.mk_ge(lhs, r)
}

/// `lhs <= rhs` as a term.
fn le(m: &mut TermManager, lhs: TermId, rhs: i64) -> TermId {
    let r = ic(m, rhs);
    m.mk_le(lhs, r)
}

/// `lhs = rhs` as a term.
fn eq(m: &mut TermManager, lhs: TermId, rhs: i64) -> TermId {
    let r = ic(m, rhs);
    m.mk_eq(lhs, r)
}

fn decide(assertions: &[TermId], m: &TermManager) -> NlaVerdict {
    check_assertions(assertions, m, &NlaConfig::default())
}

// --- satisfiable goals ------------------------------------------------------

#[test]
fn product_equal_twelve_is_sat_with_a_consistent_witness() {
    // x * y = 12 ∧ 1 <= x <= 12 ∧ y >= 1
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 12);
    let b = ge(&mut m, x, 1);
    let c = le(&mut m, x, 12);
    let d = ge(&mut m, y, 1);

    let NlaVerdict::Sat(interp) = decide(&[a, b, c, d], &m) else {
        panic!("x*y = 12 with a bounded x is satisfiable");
    };

    // The witness must be product-consistent, not merely present. Evaluating
    // the *original* assertion set is the real check, and `check_assertions`
    // has already done it — so assert the individual values line up too, which
    // is what "product-consistent" actually means.
    let xv = interp.num_of(x).expect("x pinned").clone();
    let yv = interp.num_of(y).expect("y pinned").clone();
    assert!(xv.is_integer() && yv.is_integer());
    assert_eq!(
        xv.clone() * yv.clone(),
        num_rational::BigRational::from_integer(BigInt::from(12)),
        "witness must satisfy x*y = 12, got x={xv}, y={yv}"
    );
    assert!(xv >= num_rational::BigRational::from_integer(BigInt::from(1)));
    assert!(yv >= num_rational::BigRational::from_integer(BigInt::from(1)));
}

#[test]
fn a_satisfiable_square_is_sat() {
    // x * x = 9 ∧ x >= 0  ⇒  x = 3
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let p = m.mk_mul(vec![x, x]);
    let a = eq(&mut m, p, 9);
    let b = ge(&mut m, x, 0);
    let c = le(&mut m, x, 100);
    assert!(
        matches!(decide(&[a, b, c], &m), NlaVerdict::Sat(_)),
        "x*x = 9 with x >= 0 is satisfiable at x = 3"
    );
}

// --- refutations ------------------------------------------------------------

#[test]
fn square_equal_negative_one_is_unsat() {
    // x * x = -1 — refuted by forward interval propagation alone: `x^2 >= 0`
    // is a tautology the monic layer derives with an empty reason set.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let p = m.mk_mul(vec![x, x]);
    let a = eq(&mut m, p, -1);
    assert_eq!(decide(&[a], &m), NlaVerdict::Unsat);
}

#[test]
fn product_of_two_large_factors_cannot_be_small() {
    // x >= 3 ∧ y >= 3 ∧ x*y <= 8 — forward propagation gives x*y >= 9.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = ge(&mut m, x, 3);
    let b = ge(&mut m, y, 3);
    let c = le(&mut m, p, 8);
    assert_eq!(decide(&[a, b, c], &m), NlaVerdict::Unsat);
}

#[test]
fn two_is_not_a_perfect_square_over_the_integers() {
    // x * x = 2. Over `Z` there is no root: forward propagation bounds `|x|`
    // by the integer square root, and the two remaining candidates are then
    // closed by the sign split.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let p = m.mk_mul(vec![x, x]);
    let a = eq(&mut m, p, 2);
    assert_eq!(decide(&[a], &m), NlaVerdict::Unsat);
}

#[test]
fn a_refutation_survives_a_dropped_disjunction() {
    // (x*x = -1) ∧ (x >= 0 ∨ x <= 0)
    //
    // The disjunction is outside the linearisation grammar and is dropped,
    // setting `incomplete`. Dropping a conjunct *weakens* the problem, so a
    // refutation of what remains still refutes the input — this is the
    // soundness-of-dropping property, and it must not be gated the way `sat`
    // is.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let p = m.mk_mul(vec![x, x]);
    let hard = eq(&mut m, p, -1);
    let l = ge(&mut m, x, 0);
    let r = le(&mut m, x, 0);
    let disjunction = m.mk_or(vec![l, r]);
    let conj = m.mk_and(vec![hard, disjunction]);

    let lin = super::super::linearize::linearize(&[conj], &m).expect("arithmetic content");
    assert!(lin.incomplete, "the Or conjunct must be dropped");
    assert_eq!(
        decide(&[conj], &m),
        NlaVerdict::Unsat,
        "unsat must survive a dropped conjunct"
    );
}

// --- the incompleteness gate ------------------------------------------------

#[test]
fn an_engine_sat_under_an_incomplete_linearization_is_suppressed() {
    // (x*y = 12 ∧ 1 <= x <= 12 ∧ y >= 1) ∧ (x >= 0 ∨ y >= 99)
    //
    // The satisfiable core would answer `sat`, but the dropped disjunction
    // means the relaxation is weaker than the input, so the witness says
    // nothing about the goal and the verdict must degrade to `unknown`.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 12);
    let b = ge(&mut m, x, 1);
    let c = le(&mut m, x, 12);
    let d = ge(&mut m, y, 1);

    // Sanity: without the disjunction this is the `sat` case.
    assert!(matches!(decide(&[a, b, c, d], &m), NlaVerdict::Sat(_)));

    let l = ge(&mut m, x, 0);
    let r = ge(&mut m, y, 99);
    let disjunction = m.mk_or(vec![l, r]);
    let conj = m.mk_and(vec![a, b, c, d, disjunction]);

    let lin = super::super::linearize::linearize(&[conj], &m).expect("arithmetic content");
    assert!(lin.incomplete, "the Or conjunct must be dropped");
    assert_eq!(
        decide(&[conj], &m),
        NlaVerdict::Unknown,
        "sat must be suppressed when a conjunct was dropped"
    );
}

// --- budgets ----------------------------------------------------------------

#[test]
fn a_zero_node_budget_yields_unknown_never_a_verdict() {
    // A goal that genuinely needs a case split, run with no splits allowed.
    // The engine must answer `unknown`; answering `unsat` because it could not
    // explore is exactly the failure this budget discipline exists to prevent.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 12);
    let b = ge(&mut m, x, -20);
    let c = le(&mut m, x, 20);
    let d = ge(&mut m, y, -20);
    let e = le(&mut m, y, 20);
    let assertions = [a, b, c, d, e];

    let starved = NlaConfig {
        max_nodes: 0,
        ..NlaConfig::default()
    };
    let verdict = check_assertions(&assertions, &m, &starved);
    assert!(
        !matches!(verdict, NlaVerdict::Unsat),
        "a starved budget must never refute; got {verdict:?}"
    );
}

#[test]
fn every_budget_of_zero_is_survivable() {
    // No budget may panic or be read as a verdict when set to zero. `max_nodes`
    // has its own test above; this sweeps the rest.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 12);
    let b = ge(&mut m, x, 1);
    let c = le(&mut m, x, 12);
    let assertions = [a, b, c];

    let base = NlaConfig::default();
    for starved in [
        NlaConfig {
            max_rounds: 0,
            ..base.clone()
        },
        NlaConfig {
            max_depth: 0,
            ..base.clone()
        },
        NlaConfig {
            max_tangent_cuts: 0,
            ..base.clone()
        },
        NlaConfig {
            max_pivots: 0,
            ..base.clone()
        },
        NlaConfig {
            max_lia_depth: 0,
            ..base.clone()
        },
    ] {
        // Any answer is acceptable except a panic; a refutation would need the
        // search to have actually closed every case, which a zero budget
        // cannot have done for a satisfiable goal.
        let verdict = check_assertions(&assertions, &m, &starved);
        assert!(
            !matches!(verdict, NlaVerdict::Unsat),
            "a starved budget refuted a satisfiable goal: {starved:?}"
        );
    }
}

#[test]
fn the_verdict_is_a_function_of_the_input() {
    // Two identical runs must agree. The search reads bounds out of hash maps
    // and its shape depends on assertion order, so a stray hash-order
    // dependency would show up here as a flapping verdict.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let z = int_var(&mut m, "z");
    let p = m.mk_mul(vec![x, y]);
    let q = m.mk_mul(vec![y, z]);
    let a = eq(&mut m, p, 12);
    let b = ge(&mut m, x, 1);
    let c = le(&mut m, x, 12);
    let d = ge(&mut m, y, 1);
    let e = le(&mut m, q, 40);
    let f = ge(&mut m, z, 1);
    let assertions = [a, b, c, d, e, f];

    let first = decide(&assertions, &m);
    for _ in 0..4 {
        assert_eq!(
            decide(&assertions, &m),
            first,
            "the verdict must not depend on iteration order"
        );
    }
}

// --- scope discipline -------------------------------------------------------

#[test]
fn the_search_restores_its_entry_scope() {
    // `solve` carries a debug assertion for this; the test makes it observable
    // in release builds too, and pins the invariant against a future edit that
    // adds an early return between a push and its pop.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 7);
    let b = ge(&mut m, x, -10);
    let c = le(&mut m, x, 10);
    let d = ge(&mut m, y, -10);
    let e = le(&mut m, y, 10);

    let lin = super::super::linearize::linearize(&[a, b, c, d, e], &m).expect("arithmetic");
    let config = NlaConfig::default();
    let mut engine = NlaEngine::new(&lin, &config).expect("engine builds");
    let before = engine.lia.scope_depth();
    let _ = engine.solve();
    assert_eq!(
        engine.lia.scope_depth(),
        before,
        "solve must be scope balanced"
    );
}

// --- exact arithmetic -------------------------------------------------------

#[test]
fn monic_consistency_is_checked_beyond_i64() {
    // A product of two factors near 2^31 overflows an i64 while remaining an
    // ordinary model. The consistency check runs in BigInt, so a model that
    // assigns the product anything other than the true value must be rejected
    // — a wrapped comparison here is the one overflow that could manufacture a
    // wrong `sat`.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let big = 3_037_000_500_i64; // > 2^31, and big*big > i64::MAX
    let a = eq(&mut m, x, big);
    let b = eq(&mut m, y, big);
    let c = eq(&mut m, p, 1); // plainly false for those factors

    // Whatever the engine answers, it must not be `sat`: no model satisfies
    // this. `unsat` or `unknown` are both acceptable (the coefficients may not
    // be representable in the LP), a witness is not.
    assert!(
        !matches!(decide(&[a, b, c], &m), NlaVerdict::Sat(_)),
        "an unsatisfiable goal must never produce a witness"
    );
}

// --- witness verification ---------------------------------------------------

#[test]
fn a_returned_witness_satisfies_the_original_assertions() {
    // The public contract calls `sat` advisory and re-verifies before
    // returning. This checks the re-verification is really wired, by running
    // `holds_under` again on what came back.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 6);
    let b = ge(&mut m, x, 1);
    let c = le(&mut m, x, 6);
    let d = ge(&mut m, y, 1);
    let assertions = [a, b, c, d];

    let NlaVerdict::Sat(interp) = decide(&assertions, &m) else {
        panic!("x*y = 6 with a bounded x is satisfiable");
    };
    assert!(
        crate::nl_eval::holds_under(&assertions, &m, &interp),
        "a returned witness must satisfy the input it was returned for"
    );
}

#[test]
fn a_goal_with_no_arithmetic_is_unknown() {
    let mut m = TermManager::new();
    let b = m.sorts.bool_sort;
    let p = m.mk_var("p", b);
    assert_eq!(decide(&[p], &m), NlaVerdict::Unknown);
}

// --- the sign lemma tables, on satisfiable instances -------------------------
//
// These are the branch-local lemmas emitted under a sign premise. A flipped
// convention in `sign` or `proportion` asserts a *false* lemma inside a case,
// which closes a case that has a model — a wrong `unsat`, the one failure mode
// no later check recovers from. Every other satisfiable test in this file has a
// non-negative solution, so without these the negative arms of both tables are
// never exercised in the direction that would expose a flip.

#[test]
fn a_product_of_two_negatives_is_sat() {
    // x*y = 12 ∧ -12 <= x <= -1 ∧ y <= -1, satisfied at (-1, -12).
    // Drives Neg/Neg: `sign` must conclude `product > 0`, and `proportion`
    // must pick the `y < 0, x <= -1 ⇒ v >= -y` form.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 12);
    let b = ge(&mut m, x, -12);
    let c = le(&mut m, x, -1);
    let d = le(&mut m, y, -1);
    let assertions = [a, b, c, d];

    let NlaVerdict::Sat(interp) = decide(&assertions, &m) else {
        panic!("x*y = 12 with both factors negative is satisfiable at (-1, -12)");
    };
    assert!(crate::nl_eval::holds_under(&assertions, &m, &interp));
}

#[test]
fn a_product_of_mixed_signs_is_sat() {
    // x*y = -12 ∧ 1 <= x <= 12 ∧ y <= -1, satisfied at (1, -12).
    // Drives Pos/Neg: `sign` must conclude `product < 0` — the other half of
    // the table from the test above.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, -12);
    let b = ge(&mut m, x, 1);
    let c = le(&mut m, x, 12);
    let d = le(&mut m, y, -1);
    let assertions = [a, b, c, d];

    let NlaVerdict::Sat(interp) = decide(&assertions, &m) else {
        panic!("x*y = -12 with mixed signs is satisfiable at (1, -12)");
    };
    assert!(crate::nl_eval::holds_under(&assertions, &m, &interp));
}

#[test]
fn a_negative_square_root_is_sat() {
    // x*x = 9 ∧ x <= 0 ⇒ x = -3. The square shape under a negative sign
    // premise: an even exponent must come out positive whatever the base's
    // sign, which is the branch `product_sign` handles separately.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let p = m.mk_mul(vec![x, x]);
    let a = eq(&mut m, p, 9);
    let b = le(&mut m, x, 0);
    let c = ge(&mut m, x, -100);
    let assertions = [a, b, c];

    let NlaVerdict::Sat(interp) = decide(&assertions, &m) else {
        panic!("x*x = 9 with x <= 0 is satisfiable at x = -3");
    };
    assert!(crate::nl_eval::holds_under(&assertions, &m, &interp));
}

#[test]
fn a_zero_factor_annihilates_without_blocking_a_model() {
    // x*y = 0 ∧ x = 0 ∧ y = 7. `emit_zero_lemmas` fires here; if it ever
    // attached `v = 0` to the wrong monic this model would be lost.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, p, 0);
    let b = eq(&mut m, x, 0);
    let c = eq(&mut m, y, 7);
    let assertions = [a, b, c];

    let NlaVerdict::Sat(interp) = decide(&assertions, &m) else {
        panic!("x*y = 0 with x = 0 and y = 7 is satisfiable");
    };
    assert!(crate::nl_eval::holds_under(&assertions, &m, &interp));
}

#[test]
fn a_zero_factor_forces_a_zero_product() {
    // x = 0 ∧ x*y = 5 is unsat, and only the annihilation lemma shows it:
    // the LP alone sees an unconstrained product variable.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = eq(&mut m, x, 0);
    let b = eq(&mut m, p, 5);
    let c = ge(&mut m, y, -50);
    let d = le(&mut m, y, 50);
    assert_eq!(decide(&[a, b, c, d], &m), NlaVerdict::Unsat);
}

#[test]
fn a_negative_product_of_two_positives_is_unsat() {
    // x >= 1 ∧ y >= 1 ∧ x*y <= -1. The Pos/Pos sign lemma is what closes
    // this: forward interval propagation also reaches it, so the value is in
    // having both routes agree rather than in either alone.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let a = ge(&mut m, x, 1);
    let b = ge(&mut m, y, 1);
    let c = le(&mut m, p, -1);
    assert_eq!(decide(&[a, b, c], &m), NlaVerdict::Unsat);
}
