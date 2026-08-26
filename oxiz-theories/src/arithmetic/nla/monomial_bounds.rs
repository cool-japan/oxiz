// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Checked interval arithmetic over monics, with explanations.
//!
//! For a monic `v = x^a * y^b * ...` the linear relaxation knows nothing about
//! the product, but bounds on the factors still constrain `v` and — when the
//! cofactor is bounded away from zero — bounds on `v` constrain the factors.
//! That two-way propagation is what this module computes.
//!
//! # Explanations
//!
//! A derived bound is worthless to a conflict analysis unless it can say which
//! input bounds produced it, so every endpoint carries a [`Bound::reasons`]
//! tag list in the same `u32`-tag idiom Simplex already uses for its bound
//! antecedents. Reasons are *unioned* along the derivation: an over-large
//! reason set weakens the explanation but can never make it unsound, whereas a
//! missing reason would.
//!
//! An endpoint justified by a tautology (`x^2 >= 0`) carries an empty reason
//! set, which is correct — it depends on no assertion at all.
//!
//! # Overflow
//!
//! Every product routes through [`checked_mul_r64`]. The moment one cannot be
//! represented, propagation reports [`PropOutcome::Overflow`] and *nothing is
//! written back*: deriving no bound is always sound, deriving a wrapped one
//! never is.

use super::{checked_mul_r64, checked_pow_r64, checked_recip_r64};
#[allow(unused_imports)]
use crate::prelude::*;
use core::cmp::Ordering;
use num_rational::Rational64;
use num_traits::Zero;

/// A finite bound together with the input tags that justify it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Bound {
    /// The bounding value.
    pub(crate) value: Rational64,
    /// Input bound tags whose conjunction entails this bound.
    pub(crate) reasons: Vec<u32>,
}

impl Bound {
    /// A bound justified by the given tags.
    pub(crate) fn new(value: Rational64, reasons: Vec<u32>) -> Self {
        let mut reasons = reasons;
        reasons.sort_unstable();
        reasons.dedup();
        Self { value, reasons }
    }

    /// A bound depending on a single input.
    pub(crate) fn tagged(value: Rational64, tag: u32) -> Self {
        Self {
            value,
            reasons: vec![tag],
        }
    }

    /// A bound that follows from nothing (a tautology such as `x^2 >= 0`).
    pub(crate) fn axiom(value: Rational64) -> Self {
        Self {
            value,
            reasons: Vec::new(),
        }
    }
}

/// A closed interval; `None` on a side means unbounded there.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Interval {
    /// Lower bound, `None` for `-inf`.
    pub(crate) lo: Option<Bound>,
    /// Upper bound, `None` for `+inf`.
    pub(crate) hi: Option<Bound>,
}

/// What one propagation round achieved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PropOutcome {
    /// At least one bound was tightened.
    Progress,
    /// Nothing changed; the round is a no-op and iteration can stop.
    Fixpoint,
    /// A derived interval is empty — the current bounds are inconsistent.
    Conflict,
    /// A coefficient could not be represented; nothing was written back.
    Overflow,
}

impl Interval {
    /// The whole line.
    pub(crate) fn unbounded() -> Self {
        Self::default()
    }

    /// `[lo, hi]` from raw values, both justified by `tag`.
    ///
    /// Used by the tests of this module and of `int_root`; the engine builds
    /// its intervals endpoint-by-endpoint from the solver's bounds, where the
    /// two sides carry different tags and so cannot share one.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn closed(lo: Rational64, hi: Rational64, tag: u32) -> Self {
        Self {
            lo: Some(Bound::tagged(lo, tag)),
            hi: Some(Bound::tagged(hi, tag)),
        }
    }

    /// Is this interval empty (`lo > hi`)?
    pub(crate) fn is_empty(&self) -> bool {
        match (&self.lo, &self.hi) {
            (Some(l), Some(h)) => l.value > h.value,
            _ => false,
        }
    }

    /// Does the interval provably exclude zero? Needed before dividing by it.
    pub(crate) fn excludes_zero(&self) -> bool {
        self.lo
            .as_ref()
            .is_some_and(|b| b.value > Rational64::zero())
            || self
                .hi
                .as_ref()
                .is_some_and(|b| b.value < Rational64::zero())
    }

    /// Intersect `other`'s bounds into `self`, keeping the tighter side.
    ///
    /// Returns [`PropOutcome::Progress`] if either side moved,
    /// [`PropOutcome::Conflict`] if the result is empty, otherwise
    /// [`PropOutcome::Fixpoint`].
    pub(crate) fn tighten(&mut self, other: &Interval) -> PropOutcome {
        let mut moved = false;
        if let Some(nl) = &other.lo {
            let better = match &self.lo {
                Some(cur) => nl.value > cur.value,
                None => true,
            };
            if better {
                self.lo = Some(nl.clone());
                moved = true;
            }
        }
        if let Some(nh) = &other.hi {
            let better = match &self.hi {
                Some(cur) => nh.value < cur.value,
                None => true,
            };
            if better {
                self.hi = Some(nh.clone());
                moved = true;
            }
        }
        if self.is_empty() {
            PropOutcome::Conflict
        } else if moved {
            PropOutcome::Progress
        } else {
            PropOutcome::Fixpoint
        }
    }
}

// --- extended endpoints -----------------------------------------------------

/// An interval endpoint on the extended real line.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Ext {
    NegInf,
    Fin(Rational64, Vec<u32>),
    PosInf,
}

impl Ext {
    fn lo_of(iv: &Interval) -> Ext {
        match &iv.lo {
            Some(b) => Ext::Fin(b.value, b.reasons.clone()),
            None => Ext::NegInf,
        }
    }
    fn hi_of(iv: &Interval) -> Ext {
        match &iv.hi {
            Some(b) => Ext::Fin(b.value, b.reasons.clone()),
            None => Ext::PosInf,
        }
    }
    fn into_bound(self) -> Option<Bound> {
        match self {
            Ext::Fin(v, r) => Some(Bound::new(v, r)),
            _ => None,
        }
    }
}

fn union_reasons(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

/// Extended multiplication.
///
/// The one delicate case is `0 * inf`. Interval multiplication by endpoint
/// enumeration is correct precisely when that product is taken to be `0`: for
/// `[0, 1] * [2, +inf)` the candidates then become `{0, 0, 2, +inf}`, giving
/// `[0, +inf)`, which is the true image. Any other convention loses the
/// finite side.
fn ext_mul(x: &Ext, y: &Ext) -> Option<Ext> {
    match (x, y) {
        (Ext::Fin(a, ra), Ext::Fin(b, rb)) => {
            let p = checked_mul_r64(*a, *b)?;
            Some(Ext::Fin(p, union_reasons(ra, rb)))
        }
        (Ext::Fin(a, ra), inf) | (inf, Ext::Fin(a, ra)) => {
            if a.is_zero() {
                return Some(Ext::Fin(Rational64::zero(), ra.clone()));
            }
            let flip = *a < Rational64::zero();
            Some(match (inf, flip) {
                (Ext::PosInf, false) | (Ext::NegInf, true) => Ext::PosInf,
                _ => Ext::NegInf,
            })
        }
        (Ext::PosInf, Ext::PosInf) | (Ext::NegInf, Ext::NegInf) => Some(Ext::PosInf),
        _ => Some(Ext::NegInf),
    }
}

fn ext_cmp(x: &Ext, y: &Ext) -> Ordering {
    match (x, y) {
        (Ext::NegInf, Ext::NegInf) | (Ext::PosInf, Ext::PosInf) => Ordering::Equal,
        (Ext::NegInf, _) | (_, Ext::PosInf) => Ordering::Less,
        (_, Ext::NegInf) | (Ext::PosInf, _) => Ordering::Greater,
        (Ext::Fin(a, _), Ext::Fin(b, _)) => a.cmp(b),
    }
}

fn ext_min(cands: &[Ext]) -> Ext {
    let mut best = Ext::PosInf;
    for c in cands {
        if ext_cmp(c, &best) == Ordering::Less {
            best = c.clone();
        }
    }
    best
}

fn ext_max(cands: &[Ext]) -> Ext {
    let mut best = Ext::NegInf;
    for c in cands {
        if ext_cmp(c, &best) == Ordering::Greater {
            best = c.clone();
        }
    }
    best
}

// --- forward: factors => product -------------------------------------------

/// The image of `a * b`. `None` on overflow.
pub(crate) fn mul(a: &Interval, b: &Interval) -> Option<Interval> {
    let cands = [
        ext_mul(&Ext::lo_of(a), &Ext::lo_of(b))?,
        ext_mul(&Ext::lo_of(a), &Ext::hi_of(b))?,
        ext_mul(&Ext::hi_of(a), &Ext::lo_of(b))?,
        ext_mul(&Ext::hi_of(a), &Ext::hi_of(b))?,
    ];
    Some(Interval {
        lo: ext_min(&cands).into_bound(),
        hi: ext_max(&cands).into_bound(),
    })
}

fn ext_pow(x: &Ext, e: u32) -> Option<Ext> {
    Some(match x {
        Ext::PosInf => Ext::PosInf,
        Ext::NegInf => {
            if e.is_multiple_of(2) {
                Ext::PosInf
            } else {
                Ext::NegInf
            }
        }
        Ext::Fin(v, r) => Ext::Fin(checked_pow_r64(*v, e)?, r.clone()),
    })
}

/// The image of `a^e`, computed exactly rather than by repeated
/// multiplication — `x * x` treats the two occurrences as independent and
/// would lose the correlation (`[-1, 4]^2` would come out as `[-4, 16]`).
///
/// An even exponent forces a non-negative lower bound. When the base straddles
/// zero that `0` is a tautology and is tagged with no reasons at all.
pub(crate) fn pow(a: &Interval, e: u32) -> Option<Interval> {
    if e == 0 {
        return Some(Interval {
            lo: Some(Bound::axiom(Rational64::new_raw(1, 1))),
            hi: Some(Bound::axiom(Rational64::new_raw(1, 1))),
        });
    }
    if e == 1 {
        return Some(a.clone());
    }
    let lo = Ext::lo_of(a);
    let hi = Ext::hi_of(a);
    let zero = Ext::Fin(Rational64::zero(), Vec::new());
    let plo = ext_pow(&lo, e)?;
    let phi = ext_pow(&hi, e)?;

    if !e.is_multiple_of(2) {
        // Strictly increasing: endpoints map to endpoints.
        return Some(Interval {
            lo: plo.into_bound(),
            hi: phi.into_bound(),
        });
    }
    let nonneg_base = ext_cmp(&lo, &zero) != Ordering::Less;
    let nonpos_base = ext_cmp(&hi, &zero) != Ordering::Greater;
    Some(if nonneg_base {
        Interval {
            lo: plo.into_bound(),
            hi: phi.into_bound(),
        }
    } else if nonpos_base {
        Interval {
            lo: phi.into_bound(),
            hi: plo.into_bound(),
        }
    } else {
        Interval {
            lo: Some(Bound::axiom(Rational64::zero())),
            hi: ext_max(&[plo, phi]).into_bound(),
        }
    })
}

/// The image of `prod_i factors[i].0 ^ factors[i].1`. `None` on overflow.
pub(crate) fn forward(factors: &[(Interval, u32)]) -> Option<Interval> {
    let one = Rational64::new_raw(1, 1);
    let mut acc = Interval {
        lo: Some(Bound::axiom(one)),
        hi: Some(Bound::axiom(one)),
    };
    for (iv, e) in factors {
        let p = pow(iv, *e)?;
        acc = mul(&acc, &p)?;
    }
    Some(acc)
}

// --- backward: product / cofactor => factor ---------------------------------

/// `1 / a`, for an interval that provably excludes zero.
///
/// A one-sided interval such as `[2, +inf)` reciprocates to `(0, 1/2]`; the
/// open `0` is reported as the closed bound `0`, which is weaker and sound.
/// Returns `None` when the sign of `a` is not determined, or on overflow.
pub(crate) fn recip(a: &Interval) -> Option<Interval> {
    let zero = Rational64::zero();
    if a.lo.as_ref().is_some_and(|b| b.value > zero) {
        let l = a.lo.as_ref()?;
        let new_hi = Bound::new(checked_recip_r64(l.value)?, l.reasons.clone());
        let new_lo = match &a.hi {
            Some(h) => Bound::new(checked_recip_r64(h.value)?, h.reasons.clone()),
            None => Bound::new(zero, l.reasons.clone()),
        };
        return Some(Interval {
            lo: Some(new_lo),
            hi: Some(new_hi),
        });
    }
    if a.hi.as_ref().is_some_and(|b| b.value < zero) {
        let h = a.hi.as_ref()?;
        let new_lo = Bound::new(checked_recip_r64(h.value)?, h.reasons.clone());
        let new_hi = match &a.lo {
            Some(l) => Bound::new(checked_recip_r64(l.value)?, l.reasons.clone()),
            None => Bound::new(zero, h.reasons.clone()),
        };
        return Some(Interval {
            lo: Some(new_lo),
            hi: Some(new_hi),
        });
    }
    None
}

/// `product / cofactor`, the interval a factor must lie in.
///
/// `None` when the cofactor is not bounded away from zero (division would be
/// unbounded and derive nothing) or on overflow.
pub(crate) fn backward(product: &Interval, cofactor: &Interval) -> Option<Interval> {
    if !cofactor.excludes_zero() {
        return None;
    }
    let inv = recip(cofactor)?;
    mul(product, &inv)
}

/// One round of two-way propagation for a single monic.
///
/// Forward first (factors constrain the product), then backward for each
/// exponent-one factor whose cofactor excludes zero. A factor raised to a
/// higher power is skipped: inverting `x^e` needs a root, not a division, and
/// deriving nothing is the sound choice.
///
/// # What `Overflow` promises
///
/// No *unsound* bound is ever written: the coefficient that could not be
/// represented is simply not turned into a bound. It does **not** mean the
/// arguments are untouched — an overflow raised while building a cofactor
/// happens after the forward step has already tightened `product`, and any
/// bound written before that point came from a successful, exact computation.
/// A caller must therefore treat [`PropOutcome::Overflow`] as "partially
/// applied, then gave up", and re-read `product` and `factors` rather than
/// assuming they still hold their pre-call values.
pub(crate) fn propagate_monic(
    product: &mut Interval,
    factors: &mut [(Interval, u32)],
) -> PropOutcome {
    let Some(fwd) = forward(factors) else {
        return PropOutcome::Overflow;
    };
    let mut progress = false;
    match product.tighten(&fwd) {
        PropOutcome::Conflict => return PropOutcome::Conflict,
        PropOutcome::Progress => progress = true,
        _ => {}
    }

    for i in 0..factors.len() {
        if factors[i].1 != 1 {
            continue;
        }
        let mut cofactor = Vec::with_capacity(factors.len().saturating_sub(1));
        for (j, f) in factors.iter().enumerate() {
            if j != i {
                cofactor.push(f.clone());
            }
        }
        let Some(co) = forward(&cofactor) else {
            return PropOutcome::Overflow;
        };
        let Some(derived) = backward(product, &co) else {
            continue;
        };
        // `tighten` only ever answers Progress / Fixpoint / Conflict.
        match factors[i].0.tighten(&derived) {
            PropOutcome::Conflict => return PropOutcome::Conflict,
            PropOutcome::Progress => progress = true,
            PropOutcome::Fixpoint | PropOutcome::Overflow => {}
        }
    }

    if progress {
        PropOutcome::Progress
    } else {
        PropOutcome::Fixpoint
    }
}

#[cfg(test)]
mod tests;
