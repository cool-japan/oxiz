//! Exact witness *values* for a nonlinear real model.
//!
//! # Why this module exists at all
//!
//! [`crate::nl_eval::Interpretation`] — the witness carrier every other
//! nonlinear procedure in this crate uses — pins a term to a
//! [`BigRational`]. That is the right value language for it: `holds_under`
//! re-evaluates an interpretation against the original assertions in exact
//! rational arithmetic, and every consumer downstream of it (the solver's
//! `Model`, `(get-value ..)`'s ordinary path) stores a rational too.
//!
//! A cell decomposition does not always have a rational to offer. The only
//! value satisfying `x² = 2` is `√2`, which is exactly representable — as a
//! *root of a polynomial*, not as a fraction. Rounding it into one would
//! report a number that satisfies none of the constraints the real witness
//! did, so `crate::nlsat`'s real dispatcher has always declined to report
//! anything at all in that case, leaving the model unset.
//!
//! This module is the value language that lets it report the witness
//! honestly instead: `NlWitnessValue` is either an ordinary rational or an
//! [`AlgebraicValue`] — a defining integer polynomial plus the index of the
//! root it denotes, which is precisely what SMT-LIB's `root-obj` notation
//! spells and what Z3 prints for the same goals.
//!
//! # Why it is not inside `crate::nlsat`
//!
//! `crate::nlsat` is gated on the `nlsat` feature, because it is the only
//! module reaching the `oxiz-nlsat` dependency. These types reach nothing:
//! they are plain data. Keeping them ungated is what lets `oxiz-solver` hold
//! a field of this type, classify it on its scope-restoration checklist and
//! render it, all without a `cfg` at every one of those sites — only the
//! *population* of that field is feature-dependent.
//!
//! # This is not a second model
//!
//! A value here is never mixed with a rational `Interpretation` witness for
//! the same check. `crate::nlsat::dispatch_nra_constraints` produces exactly
//! one of the two: an interpretation when every variable is rational, and an
//! all-or-nothing map of these values when at least one is not (and then for
//! *every* variable, rationals included). See that function for the
//! reasoning; the consumer relies on it to avoid completing a partial
//! algebraic model with sort defaults, which would print a model that does
//! not satisfy the assertions.

#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use num_rational::BigRational;

/// An exact real-algebraic number: the `root_index`-th real root of a
/// univariate integer polynomial.
///
/// This is SMT-LIB's `root-obj` in struct form. `(root-obj (+ (^ x 2) (- 2))
/// 2)` — Z3's answer for `x² = 2 ∧ x > 0` — is `coefficients = [-2, 0, 1]`
/// (index = degree, so `x² − 2`) with `root_index = 2`.
///
/// # Normal form
///
/// `coefficients` is **primitive with a positive leading coefficient**: the
/// denominators are cleared, the integer content is divided out, and the
/// polynomial is negated if that leaves the leading coefficient negative.
/// All three operations preserve the real-root *set* and its left-to-right
/// order, so `root_index` is invariant under them — which is what makes the
/// normalisation safe to apply after the index was determined rather than
/// before.
///
/// The same normal form is what Z3 prints, verified against `z3 4.15.4`:
/// `2x² = 4` and `−x² + 2 = 0` both report `(+ (^ x 2) (- 2))`, and
/// `x² = 1/2` reports `(+ (* 2 (^ x 2)) (- 1))` rather than a monic form.
///
/// # What it deliberately is *not*
///
/// It is not necessarily the value's **minimal** polynomial. Z3 factors
/// before printing, so `x⁴ = 4` reports `√2` as `(root-obj (+ (^ x 2) (- 2))
/// 2)` where this representation carries the unfactored `x⁴ − 4` (whose
/// second real root is the same number). Both denote `√2` correctly; only
/// the spelling differs. Factoring here would need a univariate integer
/// factoriser on the model-reporting path, which is a separate piece of work
/// — recorded rather than faked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgebraicValue {
    /// Integer coefficients of the defining polynomial: `coefficients[i]` is
    /// the coefficient of `x^i`, so the last entry is the leading one. Never
    /// empty, and always of degree at least 1 (a constant polynomial has no
    /// roots to index).
    pub coefficients: Vec<BigInt>,
    /// Which real root this value is, counting from the left, **1-based** —
    /// the same convention as `root-obj` and as
    /// `oxiz_nlsat::cad::CadPoint::Algebraic::index`.
    pub root_index: u32,
    /// Lower end of an isolating interval containing the root.
    ///
    /// Diagnostics only: nothing in the reported value depends on it. It is
    /// carried because the polynomial and index alone say nothing legible to
    /// a human reading a debug dump, and because a caller wanting a decimal
    /// approximation has no other way to get one without re-isolating.
    pub lower: BigRational,
    /// Upper end of the isolating interval. See [`Self::lower`].
    pub upper: BigRational,
}

impl AlgebraicValue {
    /// The midpoint of the isolating interval — a rational *approximation*,
    /// never the value itself.
    ///
    /// Deliberately not named `value` and deliberately not used for anything
    /// a caller could mistake for an exact answer: the whole reason this type
    /// exists is that no rational equals the number it denotes.
    #[must_use]
    pub fn approximation(&self) -> BigRational {
        (&self.lower + &self.upper) / BigRational::from_integer(2.into())
    }
}

/// One variable's exact value in a nonlinear real model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NlWitnessValue {
    /// The value is exactly this rational.
    Rational(BigRational),
    /// The value is irrational and is this algebraic number.
    Algebraic(AlgebraicValue),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> BigInt {
        BigInt::from(n)
    }

    fn rat(n: i64, d: i64) -> BigRational {
        BigRational::new(int(n), int(d))
    }

    #[test]
    fn approximation_is_the_bracket_midpoint() {
        let value = AlgebraicValue {
            coefficients: vec![int(-2), int(0), int(1)],
            root_index: 2,
            lower: rat(3, 4),
            upper: rat(3, 2),
        };
        assert_eq!(value.approximation(), rat(9, 8));
    }

    /// The two arms are distinguishable and comparable — the map that carries
    /// them is compared for equality inside `NlDispatchResult`.
    #[test]
    fn witness_values_compare_structurally() {
        let a = NlWitnessValue::Rational(rat(1, 2));
        let b = NlWitnessValue::Rational(rat(2, 4));
        let c = NlWitnessValue::Algebraic(AlgebraicValue {
            coefficients: vec![int(-2), int(0), int(1)],
            root_index: 1,
            lower: rat(-3, 2),
            upper: rat(-3, 4),
        });
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
