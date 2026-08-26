// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Linear consequences of multiplication.
//!
//! The relaxation built by [`super::linearize`] has forgotten that
//! `v = x * y` means anything. Each constructor here hands back a *linear*
//! fact that the product does entail, so the relaxation can be strengthened
//! until either it becomes infeasible (the input is unsat) or its model
//! happens to respect every monic (the input is sat).
//!
//! # Scope is part of the lemma
//!
//! Almost none of these are unconditional. `x > 0 ∧ y > 0 ⟹ xy > 0` is a
//! consequence *of its premises*, and asserting the conclusion outside the
//! branch where the premises hold would be plain unsound. Each [`Lemma`]
//! therefore carries a [`LemmaScope`] and the tags of the bounds it leaned on;
//! only [`LemmaScope::Global`] lemmas — the square tangent
//! `v = x² ⟹ v ≥ 2ax − a²`, valid for every `a` and every `x` — may be added
//! to the root.
//!
//! Callers are expected to supply premises they have actually established from
//! the current bounds; nothing here re-derives them.
//!
//! # Checked coefficients
//!
//! Every coefficient is a product of user bounds and can overflow. A
//! constructor that cannot represent one of its coefficients returns `None`:
//! not emitting a lemma costs precision, emitting a wrapped one costs
//! soundness.

use super::super::simplex::{LinExpr, VarId};
use super::linearize::{LinAtom, LinAtomKind};
use super::{checked_add_r64, checked_mul_r64, checked_neg_r64};
#[allow(unused_imports)]
use crate::prelude::*;
use num_rational::Rational64;
use num_traits::Zero;
use smallvec::SmallVec;

/// Where a lemma may be asserted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LemmaScope {
    /// Valid unconditionally; safe to add at the root and keep forever.
    Global,
    /// Valid only while the premises recorded on the lemma hold; must be
    /// retracted when the branch that established them is popped.
    BranchLocal,
}

/// Which construction produced a lemma. Useful for dedup and for telling a
/// cascade that keeps making progress from one that is spinning.
///
/// Some variants are only constructed by the families the engine does not yet
/// emit (see the note above [`order`]). They are part of the enum's meaning
/// rather than of its current use, so the allow is on the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(crate) enum LemmaFamily {
    /// Sign of a product from the signs of its factors.
    Sign,
    /// A zero factor annihilates the product.
    Zero,
    /// A unit factor makes the product equal to the cofactor.
    Neutral,
    /// `|xy| >= |y|` when `|x| >= 1`.
    Proportion,
    /// Multiplying an inequality by a positive quantity preserves it.
    Order,
    /// Products of ordered non-negative quantities stay ordered.
    Monotonicity,
    /// The bilinear tangent plane at a point.
    Tangent,
    /// The four McCormick envelopes of a bilinear term over a box.
    McCormick,
}

/// A sign premise on a factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Sign {
    /// Strictly negative.
    Neg,
    /// Strictly positive.
    Pos,
}

/// One emitted lemma: a conjunction of linear atoms, plus where it is valid.
#[derive(Debug, Clone)]
pub(crate) struct Lemma {
    /// The atoms, implicitly conjoined; each read as `expr ⋈ 0`.
    pub(crate) atoms: SmallVec<[LinAtom; 4]>,
    /// Whether the lemma survives a backtrack.
    pub(crate) scope: LemmaScope,
    /// The construction it came from. Carried for debugging and for the
    /// dedup/progress accounting a future cascade will want; the current
    /// driver keys nothing off it.
    #[allow(dead_code)]
    pub(crate) family: LemmaFamily,
    /// Tags of the bounds the caller used to establish the premises.
    pub(crate) premises: SmallVec<[u32; 4]>,
}

impl Lemma {
    fn branch(family: LemmaFamily, premises: &[u32], atoms: SmallVec<[LinAtom; 4]>) -> Self {
        Self {
            atoms,
            scope: LemmaScope::BranchLocal,
            family,
            premises: premises.iter().copied().collect(),
        }
    }
}

/// Build `sum(coef * var) + constant ⋈ 0`, merging repeated variables through
/// checked addition so a caller may pass the same variable twice (`v = x * x`
/// makes that the normal case). `None` on overflow.
fn atom(kind: LinAtomKind, terms: &[(VarId, Rational64)], constant: Rational64) -> Option<LinAtom> {
    let mut expr = LinExpr::new();
    for (v, c) in terms {
        if c.is_zero() {
            continue;
        }
        match expr.terms.iter_mut().find(|(w, _)| w == v) {
            Some(slot) => slot.1 = checked_add_r64(slot.1, *c)?,
            None => expr.terms.push((*v, *c)),
        }
    }
    expr.terms.retain(|(_, c)| !c.is_zero());
    expr.constant = constant;
    Some(LinAtom { expr, kind })
}

fn one() -> Rational64 {
    Rational64::new_raw(1, 1)
}

// --- basics -----------------------------------------------------------------

/// Sign of a product: `sx(x) ∧ sy(y) ⟹ v > 0` when the signs agree and
/// `v < 0` when they differ, for `v = x * y`.
///
/// Both premises are strict, so the conclusion is strict too — that is the
/// whole value of the lemma, since a non-strict `v >= 0` is already implied by
/// the McCormick envelope over non-negative boxes.
pub(crate) fn sign(v: VarId, sx: Sign, sy: Sign, premises: &[u32]) -> Option<Lemma> {
    let kind = if sx == sy {
        LinAtomKind::Gt
    } else {
        LinAtomKind::Lt
    };
    let a = atom(kind, &[(v, one())], Rational64::zero())?;
    Some(Lemma::branch(
        LemmaFamily::Sign,
        premises,
        smallvec::smallvec![a],
    ))
}

/// The sign of a product stated directly, for a caller that has already worked
/// out which sign follows.
///
/// [`sign`] takes the two *factor* signs and derives the product's; for a monic
/// of arbitrary degree the caller has to count negative odd-power factors
/// itself, and then wants to assert the conclusion it computed rather than
/// re-encode it as a pair of factor signs. Passing a synthesised pair through
/// [`sign`] would produce the same atom, but it would read as a claim about two
/// factors that do not exist.
///
/// `result` is the established sign of `v`: `Pos` emits `v > 0`, `Neg` emits
/// `v < 0`. Strict either way, which is the point — the premises are strict.
pub(crate) fn product_sign(v: VarId, result: Sign, premises: &[u32]) -> Option<Lemma> {
    let kind = match result {
        Sign::Pos => LinAtomKind::Gt,
        Sign::Neg => LinAtomKind::Lt,
    };
    let a = atom(kind, &[(v, one())], Rational64::zero())?;
    Some(Lemma::branch(
        LemmaFamily::Sign,
        premises,
        smallvec::smallvec![a],
    ))
}

/// A zero factor annihilates: `x = 0 ⟹ v = 0` for `v = x * y * ...`.
pub(crate) fn zero(v: VarId, premises: &[u32]) -> Option<Lemma> {
    let a = atom(LinAtomKind::Eq, &[(v, one())], Rational64::zero())?;
    Some(Lemma::branch(
        LemmaFamily::Zero,
        premises,
        smallvec::smallvec![a],
    ))
}

/// A unit factor is neutral: `y = 1 ⟹ v = x` for `v = x * y`, emitted as
/// `v - x = 0`. `cofactor` is the `x` that remains once the unit is dropped.
pub(crate) fn neutral(v: VarId, cofactor: VarId, premises: &[u32]) -> Option<Lemma> {
    let a = atom(
        LinAtomKind::Eq,
        &[(v, one()), (cofactor, checked_neg_r64(one())?)],
        Rational64::zero(),
    )?;
    Some(Lemma::branch(
        LemmaFamily::Neutral,
        premises,
        smallvec::smallvec![a],
    ))
}

/// Multiplying by something of magnitude at least one cannot shrink:
/// `|x| >= 1 ⟹ |v| >= |y|` for `v = x * y`.
///
/// `|·|` is not linear, so the caller must say which quadrant it is in:
/// `x_sign` is `Pos` for the premise `x >= 1` and `Neg` for `x <= -1`, and
/// `y_sign` is the established sign of `y`. The four combinations give the
/// four linear forms of the same fact.
pub(crate) fn proportion(
    v: VarId,
    y: VarId,
    x_sign: Sign,
    y_sign: Sign,
    premises: &[u32],
) -> Option<Lemma> {
    let neg_one = checked_neg_r64(one())?;
    // y > 0, x >= 1  =>  v >= y            (v - y >= 0)
    // y > 0, x <= -1 =>  v <= -y           (v + y <= 0)
    // y < 0, x >= 1  =>  v <= y            (v - y <= 0)
    // y < 0, x <= -1 =>  v >= -y           (v + y >= 0)
    let (y_coef, kind) = match (y_sign, x_sign) {
        (Sign::Pos, Sign::Pos) => (neg_one, LinAtomKind::Ge),
        (Sign::Pos, Sign::Neg) => (one(), LinAtomKind::Le),
        (Sign::Neg, Sign::Pos) => (neg_one, LinAtomKind::Le),
        (Sign::Neg, Sign::Neg) => (one(), LinAtomKind::Ge),
    };
    let a = atom(kind, &[(v, one()), (y, y_coef)], Rational64::zero())?;
    Some(Lemma::branch(
        LemmaFamily::Proportion,
        premises,
        smallvec::smallvec![a],
    ))
}

// --- order and monotonicity -------------------------------------------------
//
// These two relate *pairs* of monics that share a factor, which is a different
// shape of premise from everything above: the engine must first establish an
// ordering between two variables (`a <= b`) and then find two monics whose
// factor lists differ only in that position. The current driver reasons about
// one monic at a time and so has no site that can discharge those premises;
// the constructors are kept, tested, and unused rather than deleted, because
// the pairing pass that will use them is a scheduled follow-up and re-deriving
// the (delicate) sign conventions later would be pure waste.
//
// `#[allow(dead_code)]` here is scoped to exactly the two items it excuses,
// not to the module, so anything else that falls out of use still shows up.

/// `a <= b ∧ c > 0 ⟹ ac <= bc`, emitted as `ac - bc <= 0` over the two
/// product variables.
///
/// Strictly branch-local: it holds only under the sign context that
/// established `c > 0`, and the ordering premise on `a` and `b`.
#[allow(dead_code)]
pub(crate) fn order(prod_ac: VarId, prod_bc: VarId, premises: &[u32]) -> Option<Lemma> {
    let a = atom(
        LinAtomKind::Le,
        &[(prod_ac, one()), (prod_bc, checked_neg_r64(one())?)],
        Rational64::zero(),
    )?;
    Some(Lemma::branch(
        LemmaFamily::Order,
        premises,
        smallvec::smallvec![a],
    ))
}

/// `0 <= a <= b ∧ 0 <= c <= d ⟹ ac <= bd`, emitted as `ac - bd <= 0`.
///
/// Unused for the same reason as [`order`]; see the note above it.
#[allow(dead_code)]
pub(crate) fn monotonicity(prod_ac: VarId, prod_bd: VarId, premises: &[u32]) -> Option<Lemma> {
    let a = atom(
        LinAtomKind::Le,
        &[(prod_ac, one()), (prod_bd, checked_neg_r64(one())?)],
        Rational64::zero(),
    )?;
    Some(Lemma::branch(
        LemmaFamily::Monotonicity,
        premises,
        smallvec::smallvec![a],
    ))
}

// --- tangents ---------------------------------------------------------------

/// The bilinear tangent plane at `(a, b)` for `v = x * y`.
///
/// `v - b·x - a·y + a·b` *is* `(x - a)(y - b)`, so the plane sits below the
/// surface exactly where that product is non-negative. `above` selects which
/// half the caller has established: `true` for `(x-a)(y-b) >= 0` (giving
/// `v >= b·x + a·y - a·b`), `false` for the other side.
///
/// Currently unused: the engine relaxes a bilinear term with the four
/// [`mccormick`] envelopes, which over a finite box are strictly tighter (they
/// are the convex and concave hulls). This constructor is what a *boxless*
/// bilinear term would need — the caller establishes the sign of
/// `(x-a)(y-b)` itself instead of reading a box — and is kept for that case.
#[allow(dead_code)]
pub(crate) fn tangent(
    v: VarId,
    x: VarId,
    y: VarId,
    a: Rational64,
    b: Rational64,
    above: bool,
    premises: &[u32],
) -> Option<Lemma> {
    let kind = if above {
        LinAtomKind::Ge
    } else {
        LinAtomKind::Le
    };
    let at = atom(
        kind,
        &[
            (v, one()),
            (x, checked_neg_r64(b)?),
            (y, checked_neg_r64(a)?),
        ],
        checked_mul_r64(a, b)?,
    )?;
    Some(Lemma::branch(
        LemmaFamily::Tangent,
        premises,
        smallvec::smallvec![at],
    ))
}

/// The square tangent `v = x² ⟹ v >= 2a·x - a²`, emitted as
/// `v - 2a·x + a² >= 0`.
///
/// This one is [`LemmaScope::Global`]: it is `(x - a)² >= 0` rewritten, valid
/// for every real `a` and every `x`, with no premise whatsoever. It is the
/// only unconditional family here.
pub(crate) fn square_tangent(v: VarId, x: VarId, a: Rational64) -> Option<Lemma> {
    let two = Rational64::new_raw(2, 1);
    let two_a = checked_mul_r64(two, a)?;
    let at = atom(
        LinAtomKind::Ge,
        &[(v, one()), (x, checked_neg_r64(two_a)?)],
        checked_mul_r64(a, a)?,
    )?;
    Some(Lemma {
        atoms: smallvec::smallvec![at],
        scope: LemmaScope::Global,
        family: LemmaFamily::Tangent,
        premises: SmallVec::new(),
    })
}

/// The rectangle a bilinear term is relaxed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Box2 {
    /// Lower bound on `x`.
    pub(crate) xl: Rational64,
    /// Upper bound on `x`.
    pub(crate) xu: Rational64,
    /// Lower bound on `y`.
    pub(crate) yl: Rational64,
    /// Upper bound on `y`.
    pub(crate) yu: Rational64,
}

/// The four McCormick envelopes of `v = x * y` over the box
/// `[xl, xu] × [yl, yu]`.
///
/// Each is one of the four products of a non-negative slack with another
/// expanded and rearranged:
///
/// | slack product | envelope |
/// |---|---|
/// | `(x - xl)(y - yl) >= 0` | `v >= xl·y + yl·x - xl·yl` |
/// | `(xu - x)(yu - y) >= 0` | `v >= xu·y + yu·x - xu·yu` |
/// | `(xu - x)(y - yl) >= 0` | `v <= xu·y + yl·x - xu·yl` |
/// | `(x - xl)(yu - y) >= 0` | `v <= xl·y + yu·x - xl·yu` |
///
/// Together they are the convex and concave hulls of the bilinear surface over
/// the box, which is the tightest linear relaxation a box can give.
pub(crate) fn mccormick(v: VarId, x: VarId, y: VarId, b: &Box2, premises: &[u32]) -> Option<Lemma> {
    let (xl, xu, yl, yu) = (b.xl, b.xu, b.yl, b.yu);
    let cut = |kind, cx: Rational64, cy: Rational64| -> Option<LinAtom> {
        // v - cx*y - cy*x + cx*cy  ⋈  0, where (cx, cy) are the corner
        // coordinates the slack product was taken at.
        atom(
            kind,
            &[
                (v, one()),
                (y, checked_neg_r64(cx)?),
                (x, checked_neg_r64(cy)?),
            ],
            checked_mul_r64(cx, cy)?,
        )
    };
    let mut atoms: SmallVec<[LinAtom; 4]> = SmallVec::new();
    atoms.push(cut(LinAtomKind::Ge, xl, yl)?);
    atoms.push(cut(LinAtomKind::Ge, xu, yu)?);
    atoms.push(cut(LinAtomKind::Le, xu, yl)?);
    atoms.push(cut(LinAtomKind::Le, xl, yu)?);
    Some(Lemma::branch(LemmaFamily::McCormick, premises, atoms))
}

#[cfg(test)]
mod tests;
