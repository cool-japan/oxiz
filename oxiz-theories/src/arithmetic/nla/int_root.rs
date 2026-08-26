// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Backward propagation through an integer power, by exact integer roots.
//!
//! [`super::monomial_bounds::propagate_monic`] deliberately declines to invert
//! a factor raised to a power above one: over the rationals that needs a root,
//! not a division, and there is no exact rational `e`-th root to divide by.
//! Over the *integers* there is a perfectly exact answer, and it is worth
//! having — without it `x * x = 2` is unrefutable, because nothing else ever
//! puts a finite bound on `x`.
//!
//! What this module adds is the one-factor case `v = x^e`. That is the shape
//! where the inversion is unconditional: `x ↦ x^e` is a bijection on `Z` for
//! odd `e`, and two-to-one through the origin for even `e`, so a bound on `v`
//! translates into a bound on `x` with no cofactor to reason about.
//!
//! # Why this is sound
//!
//! For odd `e` the map is strictly increasing, so `x^e ≤ U ⟺ x ≤ ⌊U^{1/e}⌋`
//! and `x^e ≥ L ⟺ x ≥ ⌈L^{1/e}⌉`, both over `Z`.
//!
//! For even `e` the map is symmetric: `x^e ≤ U` with `U ≥ 0` gives
//! `|x| ≤ ⌊U^{1/e}⌋`, an interval. The dual, `x^e ≥ L` with `L > 0`, gives
//! `|x| ≥ ⌈L^{1/e}⌉`, which is a *disjunction* (`x ≤ -c ∨ x ≥ c`) and not
//! representable as one interval, so it is not derived here — the engine's
//! sign split is what turns that case into two interval cases.
//!
//! Every root is computed on [`BigInt`] by binary search, so nothing wraps.
//! A derived endpoint that does not fit an `i64` is dropped rather than
//! clamped: a weaker bound is always sound, a wrong one never is.

use super::monomial_bounds::{Bound, Interval};
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use num_rational::Rational64;
use num_traits::{One, Signed, Zero};

/// `n^e` for a non-negative exponent, exactly.
fn pow_big(n: &BigInt, e: u32) -> BigInt {
    let mut acc = BigInt::one();
    for _ in 0..e {
        acc *= n;
    }
    acc
}

/// The largest integer `r` with `r^e <= n`, for `n >= 0` and `e >= 1`.
fn nth_root_floor(n: &BigInt, e: u32) -> BigInt {
    debug_assert!(
        !n.is_negative(),
        "nth_root_floor needs a non-negative input"
    );
    if e == 1 || n.is_zero() || n.is_one() {
        return n.clone();
    }
    // `2^(bits/e + 1)` raised to `e` is at least `2^(bits + 1)`, which exceeds
    // any value of `bits` bits — so it is a valid strict upper bound to search
    // down from.
    let bits = n.bits() as usize;
    let mut lo = BigInt::zero();
    let mut hi = BigInt::one() << (bits / (e as usize) + 1);
    while lo < hi {
        let mid = (&lo + &hi + BigInt::one()) >> 1_usize;
        if pow_big(&mid, e) <= *n {
            lo = mid;
        } else {
            hi = mid - BigInt::one();
        }
    }
    lo
}

/// The smallest integer `r` with `r^e >= n`, for `n >= 0` and `e >= 1`.
fn nth_root_ceil(n: &BigInt, e: u32) -> BigInt {
    let r = nth_root_floor(n, e);
    if pow_big(&r, e) < *n {
        r + BigInt::one()
    } else {
        r
    }
}

/// `⌊n^{1/e}⌋` over all of `Z` for odd `e`: the root of a negative value is
/// the negated ceiling root of its magnitude.
fn signed_root_floor(n: &BigInt, e: u32) -> BigInt {
    if n.is_negative() {
        -nth_root_ceil(&-n, e)
    } else {
        nth_root_floor(n, e)
    }
}

/// `⌈n^{1/e}⌉` over all of `Z` for odd `e`.
fn signed_root_ceil(n: &BigInt, e: u32) -> BigInt {
    if n.is_negative() {
        -nth_root_floor(&-n, e)
    } else {
        nth_root_ceil(n, e)
    }
}

/// A `BigInt` as a `Rational64`, or `None` when it does not fit.
fn to_rational(n: &BigInt) -> Option<Rational64> {
    let v: i64 = i64::try_from(n).ok()?;
    Some(Rational64::from_integer(v))
}

/// The floor of a rational, as an exact integer.
fn floor_int(r: Rational64) -> BigInt {
    BigInt::from(r.floor().to_integer())
}

/// The ceiling of a rational, as an exact integer.
fn ceil_int(r: Rational64) -> BigInt {
    BigInt::from(r.ceil().to_integer())
}

/// The interval `x` must lie in, given `v = x^e` and `v`'s interval, over the
/// integers.
///
/// Returns [`Interval::unbounded`] when nothing can be derived, which
/// [`Interval::tighten`] then treats as a no-op. Reasons are inherited from the
/// bound on `v` that produced each endpoint, so an explanation built from the
/// result still names a real antecedent.
#[must_use]
pub(crate) fn power_backward(product: &Interval, e: u32) -> Interval {
    if e < 2 {
        return Interval::unbounded();
    }
    let mut out = Interval::unbounded();

    if e.is_multiple_of(2) {
        // `x^e <= U` with `U >= 0` bounds `|x|`. A negative `U` is a
        // contradiction rather than a bound, and the forward direction
        // (`x^e >= 0`) already reports it, so nothing is derived here.
        if let Some(hi) = &product.hi {
            let u = floor_int(hi.value);
            if !u.is_negative() {
                let r = nth_root_floor(&u, e);
                if let (Some(pos), Some(neg)) = (to_rational(&r), to_rational(&-r.clone())) {
                    out.lo = Some(Bound::new(neg, hi.reasons.clone()));
                    out.hi = Some(Bound::new(pos, hi.reasons.clone()));
                }
            }
        }
        // `x^e >= L` with `L > 0` is the disjunction `x <= -c ∨ x >= c`; not an
        // interval, so not derived. See the module docs.
        return out;
    }

    // Odd `e`: strictly increasing, so each side maps to the matching side.
    if let Some(hi) = &product.hi {
        let r = signed_root_floor(&floor_int(hi.value), e);
        out.hi = to_rational(&r).map(|v| Bound::new(v, hi.reasons.clone()));
    }
    if let Some(lo) = &product.lo {
        let r = signed_root_ceil(&ceil_int(lo.value), e);
        out.lo = to_rational(&r).map(|v| Bound::new(v, lo.reasons.clone()));
    }
    out
}

#[cfg(test)]
mod tests;
