// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super`], kept in a sibling file the way
//! `arithmetic/simplex` and `arithmetic/solver` do. Same module tree, so
//! `use super::*` still reaches the crate-private items under test.

use super::*;
use num_bigint::BigInt;

fn int_var(m: &mut TermManager, name: &str) -> TermId {
    let s = m.sorts.int_sort;
    m.mk_var(name, s)
}

fn ic(m: &mut TermManager, n: i64) -> TermId {
    m.mk_int(BigInt::from(n))
}

fn lin(m: &TermManager, asserts: &[TermId]) -> Linearization {
    linearize(asserts, m).expect("expected arithmetic content")
}

#[test]
fn plain_linear_atom_has_no_monics() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let sum = m.mk_add(vec![x, y]);
    let two = ic(&mut m, 2);
    let a = m.mk_le(sum, two);
    let l = lin(&m, &[a]);
    assert!(l.monics.is_empty());
    assert_eq!(l.atoms.len(), 1);
    assert_eq!(l.atoms[0].kind, LinAtomKind::Le);
    assert!(!l.incomplete);
    assert_eq!(l.num_vars, 2);
}

#[test]
fn product_of_two_vars_becomes_one_monic() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let z = ic(&mut m, 0);
    let a = m.mk_ge(p, z);
    let l = lin(&m, &[a]);
    assert_eq!(l.monics.len(), 1);
    assert_eq!(l.monics[0].degree(), 2);
    assert_eq!(l.monics[0].factors.len(), 2);
    assert!(l.term_of_var.contains_key(&l.monics[0].product));
}

#[test]
fn repeated_factor_collapses_to_a_power() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let p = m.mk_mul(vec![x, x]);
    let one = ic(&mut m, 1);
    let a = m.mk_ge(p, one);
    let l = lin(&m, &[a]);
    assert_eq!(l.monics.len(), 1);
    assert_eq!(l.monics[0].factors.len(), 1);
    assert_eq!(l.monics[0].factors[0].1, 2);
}

#[test]
fn commuted_products_share_one_monic() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let p = m.mk_mul(vec![x, y]);
    let q = m.mk_mul(vec![y, x]);
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(p, zero);
    let b = m.mk_le(q, zero);
    let l = lin(&m, &[a, b]);
    assert_eq!(l.monics.len(), 1, "x*y and y*x must dedupe");
}

#[test]
fn nested_product_is_spliced_into_one_monic() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let z = int_var(&mut m, "z");
    let inner = m.mk_mul(vec![x, y]);
    let outer = m.mk_mul(vec![inner, z]);
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(outer, zero);
    let l = lin(&m, &[a]);
    let top = l
        .monics
        .iter()
        .max_by_key(|mo| mo.degree())
        .expect("a monic");
    assert_eq!(top.degree(), 3, "x*y*z must be one degree-3 monic");
    assert_eq!(top.factors.len(), 3);
}

#[test]
fn compound_factor_gets_an_aux_definition() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let one = ic(&mut m, 1);
    let xp1 = m.mk_add(vec![x, one]);
    let p = m.mk_mul(vec![xp1, y]);
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(p, zero);
    let l = lin(&m, &[a]);
    // x, y, aux(x+1), product  => 4 variables.
    assert_eq!(l.num_vars, 4);
    let defs = l
        .atoms
        .iter()
        .filter(|at| at.kind == LinAtomKind::Eq)
        .count();
    assert_eq!(defs, 1, "one aux definition expected");
}

#[test]
fn deeply_nested_term_does_not_blow_the_stack() {
    // `mk_add` does not flatten, so this is a genuinely 20_000-deep term.
    // A native-recursion walker overflows here; the explicit work stack
    // must not.
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let one = ic(&mut m, 1);
    let mut acc = x;
    for _ in 0..20_000 {
        acc = m.mk_add(vec![acc, one]);
        acc = m.mk_neg(acc);
        acc = m.mk_neg(acc);
    }
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(acc, zero);
    let l = lin(&m, &[a]);
    assert!(!l.incomplete);
    assert_eq!(l.atoms.len(), 1);
    // x + 20_000 >= 0
    assert_eq!(l.atoms[0].expr.constant, Rational64::from_integer(20_000));
}

#[test]
fn every_top_level_conjunct_is_translated() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(x, zero);
    let b = m.mk_le(y, zero);
    let conj = m.mk_and(vec![a, b]);
    let l = lin(&m, &[conj]);
    assert_eq!(l.atoms.len(), 2);
    assert!(!l.incomplete);
}

#[test]
fn out_of_grammar_conjunct_is_dropped_and_flagged() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let zero = ic(&mut m, 0);
    let good = m.mk_ge(x, zero);
    let d = m.mk_div(x, y);
    let bad = m.mk_ge(d, zero);
    let conj = m.mk_and(vec![good, bad]);
    let l = lin(&m, &[conj]);
    assert!(l.incomplete, "Div must set incomplete");
    assert_eq!(l.atoms.len(), 1, "only the good conjunct survives");
}

#[test]
fn disjunction_is_dropped_and_flagged() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(x, zero);
    let b = m.mk_le(x, zero);
    let or = m.mk_or(vec![a, b]);
    let good = m.mk_ge(x, zero);
    let conj = m.mk_and(vec![good, or]);
    let l = lin(&m, &[conj]);
    assert!(l.incomplete);
    assert_eq!(l.atoms.len(), 1);
}

#[test]
fn no_arithmetic_content_returns_none() {
    let mut m = TermManager::new();
    let b = m.sorts.bool_sort;
    let p = m.mk_var("p", b);
    assert!(linearize(&[p], &m).is_none());
}

#[test]
fn strict_int_atom_tightens_to_non_strict() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let five = ic(&mut m, 5);
    let a = m.mk_lt(x, five);
    let l = lin(&m, &[a]);
    assert_eq!(l.atoms[0].kind, LinAtomKind::Le);
    // x - 5 < 0  ==>  x - 4 <= 0
    assert_eq!(l.atoms[0].expr.constant, Rational64::from_integer(-4));
}

#[test]
fn strict_real_atom_stays_strict() {
    let mut m = TermManager::new();
    let rs = m.sorts.real_sort;
    let x = m.mk_var("rx", rs);
    let five = m.mk_real(Rational64::from_integer(5));
    let a = m.mk_lt(x, five);
    let l = lin(&m, &[a]);
    assert_eq!(l.atoms[0].kind, LinAtomKind::Lt);
}

#[test]
fn non_integral_real_const_is_dropped() {
    let mut m = TermManager::new();
    let rs = m.sorts.real_sort;
    let x = m.mk_var("rx", rs);
    let half = m.mk_real(Rational64::new(1, 2));
    let bad = m.mk_lt(x, half);
    let zero = m.mk_real(Rational64::from_integer(0));
    let good = m.mk_ge(x, zero);
    let conj = m.mk_and(vec![good, bad]);
    let l = lin(&m, &[conj]);
    assert!(l.incomplete);
    assert_eq!(l.atoms.len(), 1);
}

#[test]
fn huge_int_constant_is_dropped_not_wrapped() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let huge = m.mk_int(BigInt::from(i64::MAX) * BigInt::from(1_000_000));
    let bad = m.mk_le(x, huge);
    let zero = ic(&mut m, 0);
    let good = m.mk_ge(x, zero);
    let conj = m.mk_and(vec![good, bad]);
    let l = lin(&m, &[conj]);
    assert!(l.incomplete, "unrepresentable constant must set incomplete");
    assert_eq!(l.atoms.len(), 1);
}

#[test]
fn zero_coefficient_product_needs_no_monic() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let zero = ic(&mut m, 0);
    let p = m.mk_mul(vec![zero, x, y]);
    let a = m.mk_ge(p, zero);
    let l = lin(&m, &[a]);
    assert!(l.monics.is_empty(), "0 * x * y folds to 0");
}

#[test]
fn scaled_product_keeps_its_coefficient() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let three = ic(&mut m, 3);
    let p = m.mk_mul(vec![three, x, y]);
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(p, zero);
    let l = lin(&m, &[a]);
    assert_eq!(l.monics.len(), 1);
    let atom = l
        .atoms
        .iter()
        .find(|at| at.kind == LinAtomKind::Ge)
        .expect("the relation atom");
    assert_eq!(atom.expr.terms.len(), 1);
    assert_eq!(atom.expr.terms[0].1, Rational64::from_integer(3));
    // The product variable is `x*y`, not `3*x*y`, so it must NOT claim to be
    // the `(* 3 x y)` term -- a later model read-back would be off by 3.
    assert!(
        !l.term_of_var.contains_key(&l.monics[0].product),
        "a scaled product must not name its monic after the scaled term"
    );
}

#[test]
fn scaled_and_unscaled_products_share_a_monic_without_aliasing_terms() {
    let mut m = TermManager::new();
    let x = int_var(&mut m, "x");
    let y = int_var(&mut m, "y");
    let two = ic(&mut m, 2);
    let bare = m.mk_mul(vec![x, y]);
    let scaled = m.mk_mul(vec![two, x, y]);
    let zero = ic(&mut m, 0);
    let a = m.mk_ge(bare, zero);
    let b = m.mk_le(scaled, zero);
    let l = lin(&m, &[a, b]);
    assert_eq!(l.monics.len(), 1, "both name the same monomial");
    let p = l.monics[0].product;
    // The bare product *is* the monomial, so it may claim the variable; the
    // scaled one must not have overwritten that mapping.
    assert_eq!(l.term_of_var.get(&p), Some(&bare));
}
