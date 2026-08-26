// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for exact integer-root backward propagation.

use super::*;
use crate::arithmetic::nla::monomial_bounds::Interval;

fn r(n: i64) -> Rational64 {
    Rational64::from_integer(n)
}

fn closed(lo: i64, hi: i64) -> Interval {
    Interval::closed(r(lo), r(hi), 0)
}

fn lo_of(iv: &Interval) -> Option<i64> {
    iv.lo.as_ref().map(|b| b.value.to_integer())
}

fn hi_of(iv: &Interval) -> Option<i64> {
    iv.hi.as_ref().map(|b| b.value.to_integer())
}

// --- the roots themselves ---------------------------------------------------

#[test]
fn floor_root_is_exact_on_perfect_powers() {
    for n in 0_i64..40 {
        for e in 2_u32..5 {
            let p = n.pow(e);
            assert_eq!(
                nth_root_floor(&BigInt::from(p), e),
                BigInt::from(n),
                "floor root of {n}^{e}"
            );
        }
    }
}

#[test]
fn floor_root_rounds_down_between_powers() {
    // 2^3 = 8, 3^3 = 27: everything in [8, 26] floors to 2.
    for n in 8_i64..27 {
        assert_eq!(
            nth_root_floor(&BigInt::from(n), 3),
            BigInt::from(2),
            "n={n}"
        );
    }
}

#[test]
fn ceil_root_rounds_up_between_powers() {
    // Anything in [9, 27] needs a cube of at least 3.
    for n in 9_i64..=27 {
        assert_eq!(nth_root_ceil(&BigInt::from(n), 3), BigInt::from(3), "n={n}");
    }
}

#[test]
fn roots_agree_with_brute_force_over_a_range() {
    // The defining property, checked directly: `floor` is the largest `r` with
    // `r^e <= n`, and `ceil` the smallest with `r^e >= n`.
    for n in 0_i64..200 {
        for e in 2_u32..6 {
            let big = BigInt::from(n);
            let f = nth_root_floor(&big, e);
            assert!(pow_big(&f, e) <= big, "floor {n}^(1/{e}) too large");
            assert!(
                pow_big(&(f + BigInt::one()), e) > big,
                "floor {n}^(1/{e}) too small"
            );
            let c = nth_root_ceil(&big, e);
            assert!(pow_big(&c, e) >= big, "ceil {n}^(1/{e}) too small");
            if c > BigInt::zero() {
                assert!(
                    pow_big(&(c - BigInt::one()), e) < big,
                    "ceil {n}^(1/{e}) too large"
                );
            }
        }
    }
}

#[test]
fn signed_roots_handle_negatives_for_odd_powers() {
    // (-3)^3 = -27, and the cube root map is a bijection on Z.
    assert_eq!(signed_root_floor(&BigInt::from(-27), 3), BigInt::from(-3));
    assert_eq!(signed_root_ceil(&BigInt::from(-27), 3), BigInt::from(-3));
    // -26 sits between (-3)^3 and (-2)^3, so floor is -3 and ceil is -2.
    assert_eq!(signed_root_floor(&BigInt::from(-26), 3), BigInt::from(-3));
    assert_eq!(signed_root_ceil(&BigInt::from(-26), 3), BigInt::from(-2));
}

#[test]
fn roots_are_exact_far_beyond_i64() {
    // 2^100 is a perfect square whose root no machine word holds.
    let big = BigInt::one() << 100_usize;
    assert_eq!(nth_root_floor(&big, 2), BigInt::one() << 50_usize);
}

// --- the propagation rule ---------------------------------------------------

#[test]
fn even_power_upper_bound_bounds_the_base_both_ways() {
    // v = x^2, v <= 9  ⇒  -3 <= x <= 3.
    let derived = power_backward(&closed(0, 9), 2);
    assert_eq!(lo_of(&derived), Some(-3));
    assert_eq!(hi_of(&derived), Some(3));
}

#[test]
fn even_power_bound_rounds_toward_the_looser_interval() {
    // v = x^2, v <= 10  ⇒  -3 <= x <= 3 (not ±sqrt(10)).
    let derived = power_backward(&closed(0, 10), 2);
    assert_eq!(lo_of(&derived), Some(-3));
    assert_eq!(hi_of(&derived), Some(3));
}

#[test]
fn even_power_lower_bound_derives_nothing() {
    // v = x^2, v >= 4 is `x <= -2 ∨ x >= 2` — a disjunction, not an interval.
    // Deriving an interval here would be unsound; deriving nothing is correct.
    let iv = Interval {
        lo: Some(Bound::tagged(r(4), 0)),
        hi: None,
    };
    let derived = power_backward(&iv, 2);
    assert_eq!(lo_of(&derived), None);
    assert_eq!(hi_of(&derived), None);
}

#[test]
fn odd_power_maps_each_side_to_the_matching_side() {
    // v = x^3, -27 <= v <= 27  ⇒  -3 <= x <= 3.
    let derived = power_backward(&closed(-27, 27), 3);
    assert_eq!(lo_of(&derived), Some(-3));
    assert_eq!(hi_of(&derived), Some(3));
}

#[test]
fn odd_power_rounds_outward() {
    // v = x^3, -26 <= v <= 26  ⇒  -2 <= x <= 2, since ±3 cubes past ±26.
    let derived = power_backward(&closed(-26, 26), 3);
    assert_eq!(lo_of(&derived), Some(-2));
    assert_eq!(hi_of(&derived), Some(2));
}

#[test]
fn a_degree_one_power_derives_nothing() {
    // Degree one is not a monic at all; `propagate_monic`'s division handles
    // it, and this module declines rather than duplicating that.
    let derived = power_backward(&closed(1, 5), 1);
    assert_eq!(lo_of(&derived), None);
    assert_eq!(hi_of(&derived), None);
}

#[test]
fn a_negative_upper_bound_on_an_even_power_derives_nothing() {
    // v = x^2 <= -1 is a contradiction, but the *forward* direction is what
    // reports it (`x^2 >= 0` crossing `v <= -1`). Backward propagation has no
    // real root to offer and must not invent one.
    let derived = power_backward(&closed(-5, -1), 2);
    assert_eq!(lo_of(&derived), None);
    assert_eq!(hi_of(&derived), None);
}

#[test]
fn the_two_from_the_square_root_test_is_bounded_tightly() {
    // The propagation that makes `x*x = 2` refutable: v = 2 bounds |x| by 1,
    // and 1^2 = 1 != 2 closes both remaining cases.
    let derived = power_backward(&closed(2, 2), 2);
    assert_eq!(lo_of(&derived), Some(-1));
    assert_eq!(hi_of(&derived), Some(1));
}

#[test]
fn derived_bounds_inherit_their_antecedent_tags() {
    // An explanation built from a derived bound must name a real antecedent.
    let iv = Interval::closed(r(0), r(9), 7);
    let derived = power_backward(&iv, 2);
    let hi = derived.hi.as_ref().expect("an upper bound");
    assert_eq!(hi.reasons, vec![7], "the tag of the bound it came from");
    let lo = derived.lo.as_ref().expect("a lower bound");
    assert_eq!(lo.reasons, vec![7]);
}

#[test]
fn an_unrepresentable_endpoint_is_dropped_not_clamped() {
    // The root of a value whose root does not fit an i64 is not derivable as a
    // `Rational64` bound. Dropping it is sound; clamping would not be.
    //
    // `Rational64` cannot hold 2^100 in the first place, so the reachable form
    // of this is an unbounded side, which must simply derive nothing there.
    let iv = Interval {
        lo: None,
        hi: Some(Bound::tagged(r(i64::MAX), 0)),
    };
    let derived = power_backward(&iv, 3);
    // An i64::MAX cube root is ~2.1e6, comfortably representable.
    assert!(derived.hi.is_some());
    assert!(
        derived.lo.is_none(),
        "no lower bound was available to invert"
    );
}
