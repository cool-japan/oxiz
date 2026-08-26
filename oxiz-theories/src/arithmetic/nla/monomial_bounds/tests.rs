// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for [`super`], kept in a sibling file the way
//! `arithmetic/simplex` and `arithmetic/solver` do. Same module tree, so
//! `use super::*` still reaches the crate-private items under test.

use super::*;

fn r(n: i64) -> Rational64 {
    Rational64::from_integer(n)
}

fn iv(lo: i64, hi: i64, tag: u32) -> Interval {
    Interval::closed(r(lo), r(hi), tag)
}

fn lo_val(i: &Interval) -> Option<Rational64> {
    i.lo.as_ref().map(|b| b.value)
}
fn hi_val(i: &Interval) -> Option<Rational64> {
    i.hi.as_ref().map(|b| b.value)
}

#[test]
fn forward_product_of_two_intervals() {
    // x in [2, 3], y in [-1, 4]  =>  x*y in [-3, 12]
    let p = mul(&iv(2, 3, 0), &iv(-1, 4, 1)).expect("no overflow");
    assert_eq!(lo_val(&p), Some(r(-3)));
    assert_eq!(hi_val(&p), Some(r(12)));
}

#[test]
fn forward_unions_the_reasons_of_both_operands() {
    let p = mul(&iv(2, 3, 7), &iv(-1, 4, 9)).expect("no overflow");
    let lo = p.lo.as_ref().expect("a lower bound");
    assert_eq!(lo.reasons, vec![7, 9]);
}

#[test]
fn even_power_of_a_straddling_interval_is_non_negative() {
    let p = pow(&iv(-1, 4, 0), 2).expect("no overflow");
    assert_eq!(lo_val(&p), Some(r(0)));
    assert_eq!(hi_val(&p), Some(r(16)));
    let lo = p.lo.as_ref().expect("a lower bound");
    assert!(lo.reasons.is_empty(), "x^2 >= 0 depends on nothing");
}

#[test]
fn even_power_of_a_negative_interval_flips_endpoints() {
    let p = pow(&iv(-5, -2, 0), 2).expect("no overflow");
    assert_eq!(lo_val(&p), Some(r(4)));
    assert_eq!(hi_val(&p), Some(r(25)));
}

#[test]
fn odd_power_is_monotone() {
    let p = pow(&iv(-2, 3, 0), 3).expect("no overflow");
    assert_eq!(lo_val(&p), Some(r(-8)));
    assert_eq!(hi_val(&p), Some(r(27)));
}

#[test]
fn power_beats_repeated_multiplication_in_precision() {
    let square = pow(&iv(-1, 4, 0), 2).expect("no overflow");
    let naive = mul(&iv(-1, 4, 0), &iv(-1, 4, 0)).expect("no overflow");
    assert_eq!(lo_val(&square), Some(r(0)));
    assert_eq!(lo_val(&naive), Some(r(-4)));
}

#[test]
fn zero_times_a_half_open_interval_stays_finite() {
    let unbounded_above = Interval {
        lo: Some(Bound::tagged(r(2), 0)),
        hi: None,
    };
    let point_zero = iv(0, 0, 1);
    let p = mul(&point_zero, &unbounded_above).expect("no overflow");
    assert_eq!(lo_val(&p), Some(r(0)));
    assert_eq!(hi_val(&p), Some(r(0)));
}

#[test]
fn non_negative_times_half_open_is_half_open() {
    let unbounded_above = Interval {
        lo: Some(Bound::tagged(r(2), 0)),
        hi: None,
    };
    let p = mul(&iv(0, 1, 1), &unbounded_above).expect("no overflow");
    assert_eq!(lo_val(&p), Some(r(0)));
    assert_eq!(hi_val(&p), None);
}

#[test]
fn backward_needs_a_cofactor_excluding_zero() {
    let product = iv(6, 12, 0);
    assert!(backward(&product, &iv(-1, 4, 1)).is_none());
    let d = backward(&product, &iv(2, 3, 1)).expect("cofactor excludes zero");
    // [6,12] / [2,3] = [2, 6]
    assert_eq!(lo_val(&d), Some(r(2)));
    assert_eq!(hi_val(&d), Some(r(6)));
}

#[test]
fn backward_with_a_negative_cofactor() {
    // v = x*y, v in [6, 12], y in [-3, -2]  =>  x in [-6, -2]
    let d = backward(&iv(6, 12, 0), &iv(-3, -2, 1)).expect("excludes zero");
    assert_eq!(lo_val(&d), Some(r(-6)));
    assert_eq!(hi_val(&d), Some(r(-2)));
}

#[test]
fn recip_of_a_half_open_positive_interval_is_closed_at_zero() {
    let a = Interval {
        lo: Some(Bound::tagged(r(2), 0)),
        hi: None,
    };
    let inv = recip(&a).expect("sign is known");
    assert_eq!(lo_val(&inv), Some(r(0)));
    assert_eq!(hi_val(&inv), Some(Rational64::new(1, 2)));
}

#[test]
fn tighten_reports_progress_then_fixpoint() {
    let mut a = iv(0, 10, 0);
    assert_eq!(a.tighten(&iv(2, 8, 1)), PropOutcome::Progress);
    assert_eq!(a.tighten(&iv(1, 9, 2)), PropOutcome::Fixpoint);
    assert_eq!(lo_val(&a), Some(r(2)));
    assert_eq!(hi_val(&a), Some(r(8)));
}

#[test]
fn tighten_detects_an_empty_interval() {
    let mut a = iv(0, 1, 0);
    assert_eq!(a.tighten(&iv(5, 9, 1)), PropOutcome::Conflict);
}

#[test]
fn propagate_monic_runs_both_directions() {
    // v = x*y with x in [2,3], y in [1,4]: forward gives v in [2,12].
    let mut product = Interval::unbounded();
    let mut factors = vec![(iv(2, 3, 0), 1u32), (iv(1, 4, 1), 1u32)];
    assert_eq!(
        propagate_monic(&mut product, &mut factors),
        PropOutcome::Progress
    );
    assert_eq!(lo_val(&product), Some(r(2)));
    assert_eq!(hi_val(&product), Some(r(12)));

    // Now clamp v to [10, 12]: backward must lift y's lower bound.
    assert_eq!(product.tighten(&iv(10, 12, 2)), PropOutcome::Progress);
    assert_eq!(
        propagate_monic(&mut product, &mut factors),
        PropOutcome::Progress
    );
    let y_lo = lo_val(&factors[1].0).expect("y has a lower bound");
    assert!(
        y_lo >= Rational64::new(10, 3),
        "y >= v/x >= 10/3, got {y_lo}"
    );
}

#[test]
fn propagate_monic_reaches_a_fixpoint() {
    let mut product = Interval::unbounded();
    let mut factors = vec![(iv(2, 3, 0), 1u32), (iv(1, 4, 1), 1u32)];
    assert_eq!(
        propagate_monic(&mut product, &mut factors),
        PropOutcome::Progress
    );
    assert_eq!(
        propagate_monic(&mut product, &mut factors),
        PropOutcome::Fixpoint
    );
}

#[test]
fn propagate_monic_reports_a_conflict() {
    // v = x*y forced to 100 while x,y in [1,2] can reach at most 4.
    let mut product = iv(100, 100, 2);
    let mut factors = vec![(iv(1, 2, 0), 1u32), (iv(1, 2, 1), 1u32)];
    assert_eq!(
        propagate_monic(&mut product, &mut factors),
        PropOutcome::Conflict
    );
}

#[test]
fn overflow_is_reported_not_panicked() {
    let big = Rational64::from_integer(i64::MAX / 2);
    let a = Interval {
        lo: Some(Bound::tagged(big, 0)),
        hi: Some(Bound::tagged(big, 0)),
    };
    assert!(mul(&a, &a).is_none(), "must report, not wrap");
    assert!(pow(&a, 3).is_none());
    let mut product = Interval::unbounded();
    let mut factors = vec![(a.clone(), 1u32), (a, 1u32)];
    assert_eq!(
        propagate_monic(&mut product, &mut factors),
        PropOutcome::Overflow
    );
    // The failure was raised by the very first forward pass, so nothing had
    // been written yet. That is a property of *this* path, not a contract of
    // the function -- see `propagate_monic`'s "What `Overflow` promises".
    assert_eq!(product, Interval::unbounded());
}

#[test]
fn higher_powers_are_skipped_by_the_backward_step() {
    // v = x^2 with x in [-4, 4] and v pinned to [9, 9]: forward gives
    // v in [0, 16] (already satisfied), and no root extraction happens,
    // so x keeps its interval rather than acquiring a bogus one.
    let mut product = iv(9, 9, 2);
    let mut factors = vec![(iv(-4, 4, 0), 2u32)];
    let out = propagate_monic(&mut product, &mut factors);
    assert!(matches!(out, PropOutcome::Fixpoint | PropOutcome::Progress));
    assert_eq!(lo_val(&factors[0].0), Some(r(-4)));
    assert_eq!(hi_val(&factors[0].0), Some(r(4)));
}

#[test]
fn forward_over_a_three_factor_monic() {
    // x in [1,2], y in [3,4], z in [-1,1]  =>  xyz in [-8, 8]
    let f = vec![
        (iv(1, 2, 0), 1u32),
        (iv(3, 4, 1), 1u32),
        (iv(-1, 1, 2), 1u32),
    ];
    let p = forward(&f).expect("no overflow");
    assert_eq!(lo_val(&p), Some(r(-8)));
    assert_eq!(hi_val(&p), Some(r(8)));
}
