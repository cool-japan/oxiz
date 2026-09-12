//! The bit-vector half of the model-verification gate's evaluator.
//!
//! [`super::model_eval`] is the dispatcher: it walks the term DAG on an
//! explicit frame stack and decides *which* operator a term is.  This module
//! is everything that happens once both operands of that operator have values
//! — the width rules, and the adaptation to the workspace's folding rules.
//!
//! # Why the gate was blind to bit-vectors
//!
//! Until this module existed, [`super::EvalVal`] had two variants, `Bool` and
//! `Num`, and every `Bv*` [`TermKind`](oxiz_core::ast::TermKind) fell into
//! `open_in_model`'s closing `_ =>` arm, which looks the term up in the model
//! and hands the witness to `parse_value_term` — which answered
//! `Undetermined` for a `BitVecConst`.  So a candidate model for a QF_BV
//! formula was checked by a gate that could not read a single one of its
//! values, and `model_refutes_assertions` returned `false` for every
//! bit-vector problem, whatever the model said.
//!
//! That is not an abstract gap.  On cargo-formal's `u01` fixture
//!
//! ```text
//! (assert (= a #x0f))
//! (assert (not (and (bvule a #x0f) (bvule a #x10))))
//! ```
//!
//! OxiZ 0.3.3 and 0.3.4 answer `sat` with `a = #b00001111` — a model that
//! makes both `bvule`s true, the conjunction true and the assertion **false**.
//! The verdict is wrong; the gate is what should have noticed and downgraded
//! it to `unknown`.  With this module it does.  (The *root* cause is the
//! bit-vector solver's scope rollback, fixed separately; this evaluator is the
//! backstop that turns any future wrong `sat` of that shape into an honest
//! `unknown` rather than a fabricated counterexample.)
//!
//! # Where the semantics live
//!
//! In [`oxiz_core::ast::bv_fold`], the workspace's single definition of the
//! SMT-LIB `FixedSizeBitVectors` folding rules — the same one the term
//! builder, the rewriter, the bit-blaster and the array cross-theory check
//! ([`super::check_array::eval_bv`]) route through.  No operator's arithmetic
//! is written here.  What *is* written here, because `bv_fold` cannot know it,
//! is:
//!
//! * **Reduction.**  `bv_fold` takes operands already in `[0, 2^width)` as a
//!   precondition and `TermManager::mk_bitvec` does not enforce it, so every
//!   leaf goes through [`bv_fold::bv_wrap_unsigned`] on the way in (see
//!   [`leaf`]).  That is also the invariant [`super::EvalVal::Bv`] documents,
//!   and every rule below preserves it.
//! * **Width agreement.**  `bv_fold` takes one width per call and cannot tell
//!   that two operands disagreed.  Each rule below states its own.
//! * **"No answer."**  `bv_fold` is total on its domain; reporting
//!   [`EvalOutcome::Undetermined`] for an ill-sorted term is this module's job.
//!
//! # Which direction is safe
//!
//! The gate acts on a definite `Bool(false)` and shrugs at
//! `Undetermined`, so *declining* to fold costs at most a missed refutation
//! (a wrong `sat` stays wrong) while folding something *incorrectly* can
//! invent a refutation and turn a genuine `sat` into `unknown`.  Every rule
//! here therefore answers `Undetermined` the moment its width precondition
//! fails, and never guesses a width.
//!
//! # Bit-vector values are exact, unlike arithmetic ones
//!
//! [`super::model_eval::combine_eq`] deliberately answers `Undetermined` when
//! two *numeric* operands collide, and `cmp_strict` softens `<` / `>` at the
//! boundary, because the arithmetic solver's LP model can collide two
//! variables it never asserted equal and reports a strict bound at its
//! boundary value.  Neither caveat applies to a bit-vector: the model witness
//! is a `BitVecConst` naming every bit, produced by a bit-blasted decision
//! procedure, so equality and the four comparisons are trustworthy in **both**
//! directions.  Making equality weaker than `bvule` would also be incoherent —
//! `(= a b)` is `(and (bvule a b) (bvule b a))`.

use super::EvalVal;
use super::model_eval::EvalOutcome;
#[allow(unused_imports)]
use crate::prelude::*;
use num_bigint::BigInt;
use oxiz_core::ast::bv_fold;

/// A binary operator whose two operands and whose result are bit-vectors.
///
/// The four comparisons are [`BvCompareOp`] instead, because their result is a
/// truth value and not a bit-vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BvBinaryOp {
    /// `bvand`
    And,
    /// `bvor`
    Or,
    /// `bvxor`
    Xor,
    /// `bvadd`
    Add,
    /// `bvsub`
    Sub,
    /// `bvmul`
    Mul,
    /// `bvudiv`
    Udiv,
    /// `bvsdiv`
    Sdiv,
    /// `bvurem`
    Urem,
    /// `bvsrem`
    Srem,
    /// `bvshl`
    Shl,
    /// `bvlshr`
    Lshr,
    /// `bvashr`
    Ashr,
    /// `concat`
    Concat,
}

/// A comparison between two bit-vector operands, producing a truth value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BvCompareOp {
    /// `bvult`
    Ult,
    /// `bvule`
    Ule,
    /// `bvslt`
    Slt,
    /// `bvsle`
    Sle,
}

/// A bit-vector leaf: an interned `BitVecConst`, or a model witness read back
/// for a variable or an opaque application.
///
/// The value is reduced into `[0, 2^width)` here and nowhere else — that is
/// [`EvalVal::Bv`]'s invariant, and it matters: `mk_bitvec` interns whatever
/// integer it is handed, so a negative or oversized literal is reachable
/// through the builder API, and handing one to `bv_fold` produces a result
/// that is not a bit-vector value at all.
pub(super) fn leaf(value: &BigInt, width: u32) -> EvalOutcome {
    EvalOutcome::bits(bv_fold::bv_wrap_unsigned(value, width), width)
}

/// `bvnot`: flip every bit within the width.  The width is unchanged.
pub(super) fn complement(value: &BigInt, width: u32) -> EvalOutcome {
    EvalOutcome::bits(bv_fold::bv_not(value, width), width)
}

/// `(_ extract high low)`: the bits `high ..= low`, right-aligned, in a sort
/// of `high - low + 1` bits.
///
/// Malformed indices (`high` at or past the operand's width, or `low` above
/// `high`) produce no value.  That matches `TermManager::mk_bv_extract`, which
/// leaves them "for the parser's sort check rather than silently folding to a
/// fabricated value", and the arithmetic is checked so a malformed pair can
/// only reach `Undetermined`, never a wrapped width.
pub(super) fn extract(high: u32, low: u32, value: &BigInt, width: u32) -> EvalOutcome {
    if high >= width {
        return EvalOutcome::UNDETERMINED;
    }
    let Some(span) = high.checked_sub(low).and_then(|span| span.checked_add(1)) else {
        return EvalOutcome::UNDETERMINED;
    };
    EvalOutcome::bits(bv_fold::bv_extract(value, high, low), span)
}

/// Apply a binary bit-vector operator to two folded operands.
///
/// Two width rules, not one:
///
/// * `concat` produces the *sum* of the two widths and so requires no
///   agreement (the checked addition is what keeps a pathological pair of
///   widths from wrapping `u32` into a narrow result);
/// * **every other operator, the three shifts included, requires the two
///   widths to agree.**  SMT-LIB requires it of all of them — a shift distance
///   is a bit-vector of the shifted operand's own sort — so a disagreement
///   means the term was never well sorted and no answer for it would be
///   meaningful.  This is deliberately stricter than
///   [`super::check_array::eval_bv`], which folds a mismatched shift from the
///   left operand's width: that evaluator's answers feed a conflict check
///   where declining costs a conflict, while these answers feed the *gate*,
///   where guessing can invent one.
pub(super) fn binary(
    op: BvBinaryOp,
    left: &BigInt,
    left_width: u32,
    right: &BigInt,
    right_width: u32,
) -> EvalOutcome {
    if op == BvBinaryOp::Concat {
        let Some(joined) = left_width.checked_add(right_width) else {
            return EvalOutcome::UNDETERMINED;
        };
        return EvalOutcome::bits(bv_fold::bv_concat(left, right, right_width), joined);
    }
    if left_width != right_width {
        return EvalOutcome::UNDETERMINED;
    }
    let width = left_width;
    let value = match op {
        BvBinaryOp::And => bv_fold::bv_and(left, right, width),
        BvBinaryOp::Or => bv_fold::bv_or(left, right, width),
        BvBinaryOp::Xor => bv_fold::bv_xor(left, right, width),
        BvBinaryOp::Add => bv_fold::bv_add(left, right, width),
        BvBinaryOp::Sub => bv_fold::bv_sub(left, right, width),
        BvBinaryOp::Mul => bv_fold::bv_mul(left, right, width),
        BvBinaryOp::Udiv => bv_fold::bv_udiv(left, right, width),
        BvBinaryOp::Sdiv => bv_fold::bv_sdiv(left, right, width),
        BvBinaryOp::Urem => bv_fold::bv_urem(left, right, width),
        BvBinaryOp::Srem => bv_fold::bv_srem(left, right, width),
        BvBinaryOp::Shl => bv_fold::bv_shl(left, right, width),
        BvBinaryOp::Lshr => bv_fold::bv_lshr(left, right, width),
        BvBinaryOp::Ashr => bv_fold::bv_ashr(left, right, width),
        // Handled above, before the width check, because it is the one
        // operator whose operands need not agree.
        BvBinaryOp::Concat => return EvalOutcome::UNDETERMINED,
    };
    EvalOutcome::bits(value, width)
}

/// Compare two folded operands.
///
/// The unsigned comparisons compare the reduced values directly; the signed
/// ones reinterpret both through [`bv_fold::to_signed`] first, so `#xff < #x01`
/// is false unsigned and true signed.  Unequal widths produce no value.
pub(super) fn compare(
    op: BvCompareOp,
    left: &BigInt,
    left_width: u32,
    right: &BigInt,
    right_width: u32,
) -> EvalOutcome {
    if left_width != right_width {
        return EvalOutcome::UNDETERMINED;
    }
    let width = left_width;
    EvalOutcome::boolean(match op {
        BvCompareOp::Ult => left < right,
        BvCompareOp::Ule => left <= right,
        BvCompareOp::Slt => bv_fold::to_signed(left, width) < bv_fold::to_signed(right, width),
        BvCompareOp::Sle => bv_fold::to_signed(left, width) <= bv_fold::to_signed(right, width),
    })
}

/// `=` between two bit-vector operands.
///
/// Definite in both directions at equal widths — see the module documentation
/// for why a bit-vector collision is evidence where a numeric one is not — and
/// [`EvalOutcome::Undetermined`] at unequal widths, which is an ill-sorted term
/// and therefore the parser's problem rather than the gate's.
pub(super) fn equal(
    left: &BigInt,
    left_width: u32,
    right: &BigInt,
    right_width: u32,
) -> EvalOutcome {
    if left_width != right_width {
        return EvalOutcome::UNDETERMINED;
    }
    EvalOutcome::boolean(left == right)
}

/// `distinct` over operands that are **all** bit-vector values of the same
/// width.
///
/// The caller ([`super::model_eval`]'s `Op::Distinct` frame) only reaches here
/// when every operand folded to an [`EvalVal::Bv`]; anything else ends the
/// frame as `Undetermined` before this is called, which is exactly the answer
/// the gate gave for *every* `distinct` before bit-vectors were evaluable, so
/// no arithmetic `distinct` changes verdict.
///
/// Fewer than two operands is `Undetermined` rather than the vacuous `true`:
/// there is nothing to refute either way, and the parser never builds one.
pub(super) fn all_distinct(values: &[EvalVal]) -> EvalOutcome {
    if values.len() < 2 {
        return EvalOutcome::UNDETERMINED;
    }
    let mut seen: Vec<&BigInt> = Vec::with_capacity(values.len());
    let mut common_width: Option<u32> = None;
    for value in values {
        let EvalVal::Bv { value, width } = value else {
            return EvalOutcome::UNDETERMINED;
        };
        match common_width {
            None => common_width = Some(*width),
            Some(known) if known == *width => {}
            Some(_) => return EvalOutcome::UNDETERMINED,
        }
        seen.push(value);
    }
    // Sorting and scanning for an adjacent repeat is O(n log n) and needs no
    // hasher; `n` is the arity of one `distinct`, so this is never hot.
    seen.sort_unstable();
    EvalOutcome::boolean(seen.windows(2).all(|pair| pair[0] != pair[1]))
}

#[cfg(test)]
mod tests {
    //! # What OxiZ 0.3.3 / 0.3.4 answered
    //!
    //! For **every** test in this module: `Undetermined`, or the `false` that
    //! follows from it.  `EvalVal` had no bit-vector variant, so
    //! `parse_value_term` could not read a `BitVecConst` witness and every
    //! `Bv*` term fell into `open_in_model`'s closing `_ =>` arm.  Each
    //! operator test below therefore had *no* answer to check before this
    //! module, and every gate test below answered "the model is fine" —
    //! including `the_u01_model_is_refused`, where 0.3.3 and 0.3.4 reported
    //! `sat` with exactly the `a = #b00001111` model built there.

    use super::{
        BvBinaryOp, BvCompareOp, all_distinct, binary, compare, complement, equal, extract, leaf,
    };
    use crate::solver::model_eval::EvalOutcome;
    use crate::solver::types::Model;
    use crate::solver::{EvalVal, Solver};
    use num_bigint::BigInt;
    use oxiz_core::ast::{TermId, TermManager};

    // ─────────────────────────────────────────────────────────────────
    // A reference implementation, independent of `bv_fold`
    // ─────────────────────────────────────────────────────────────────
    //
    // Every rule below is written directly on `u128`, so it shares no code
    // with the `BigInt` implementation under test.  `u128` covers every width
    // this reference is used at (1, 7, 8, 64, 65, 128); the wider widths are
    // pinned separately by `wide_widths_*` with hand-written expectations.

    /// The all-ones `width`-bit value as a `u128`.
    fn mask(width: u32) -> u128 {
        if width >= 128 {
            u128::MAX
        } else {
            (1u128 << width) - 1
        }
    }

    /// Reinterpret an unsigned `width`-bit value as two's complement.
    fn signed(value: u128, width: u32) -> i128 {
        if width >= 128 {
            // The bit pattern *is* the two's-complement reading at 128 bits.
            value as i128
        } else if (value >> (width - 1)) & 1 == 1 {
            (value as i128) - (1i128 << width)
        } else {
            value as i128
        }
    }

    /// Reduce a two's-complement value back into `[0, 2^width)`.
    fn unsigned(value: i128, width: u32) -> u128 {
        (value as u128) & mask(width)
    }

    /// The reference answer for a binary operator, at equal widths.
    fn reference_binary(op: BvBinaryOp, a: u128, b: u128, width: u32) -> u128 {
        let m = mask(width);
        match op {
            BvBinaryOp::And => a & b,
            BvBinaryOp::Or => a | b,
            BvBinaryOp::Xor => a ^ b,
            BvBinaryOp::Add => a.wrapping_add(b) & m,
            BvBinaryOp::Sub => a.wrapping_sub(b) & m,
            BvBinaryOp::Mul => a.wrapping_mul(b) & m,
            // Both are total in SMT-LIB: division by zero is the all-ones
            // vector, remainder by zero is the dividend.
            BvBinaryOp::Udiv => a.checked_div(b).unwrap_or(m),
            BvBinaryOp::Urem => a.checked_rem(b).unwrap_or(a),
            BvBinaryOp::Sdiv => {
                let (x, y) = (signed(a, width), signed(b, width));
                if y == 0 {
                    if x < 0 { 1 } else { m }
                } else {
                    unsigned(x.wrapping_div(y), width)
                }
            }
            BvBinaryOp::Srem => {
                let (x, y) = (signed(a, width), signed(b, width));
                if y == 0 {
                    a
                } else {
                    unsigned(x.wrapping_rem(y), width)
                }
            }
            BvBinaryOp::Shl => {
                if b >= u128::from(width) {
                    0
                } else {
                    (a << b) & m
                }
            }
            BvBinaryOp::Lshr => {
                if b >= u128::from(width) {
                    0
                } else {
                    a >> b
                }
            }
            BvBinaryOp::Ashr => {
                let negative = (a >> (width - 1)) & 1 == 1;
                if b >= u128::from(width) {
                    if negative { m } else { 0 }
                } else {
                    let shifted = a >> b;
                    if negative {
                        // Re-introduce the `b` sign bits the logical shift
                        // dropped: the top `b` bits of the `width`-bit word.
                        #[allow(clippy::cast_possible_truncation)]
                        let kept = mask(width - b as u32);
                        shifted | (m ^ kept)
                    } else {
                        shifted
                    }
                }
            }
            // `concat` has its own reference: its operands need not share a
            // width and its result is wider than either.
            BvBinaryOp::Concat => unreachable!("concat has its own reference"),
        }
    }

    /// The reference answer for a comparison, at equal widths.
    fn reference_compare(op: BvCompareOp, a: u128, b: u128, width: u32) -> bool {
        match op {
            BvCompareOp::Ult => a < b,
            BvCompareOp::Ule => a <= b,
            BvCompareOp::Slt => signed(a, width) < signed(b, width),
            BvCompareOp::Sle => signed(a, width) <= signed(b, width),
        }
    }

    /// Every binary operator except `concat`, which is exercised separately.
    const BINARY_OPS: [BvBinaryOp; 13] = [
        BvBinaryOp::And,
        BvBinaryOp::Or,
        BvBinaryOp::Xor,
        BvBinaryOp::Add,
        BvBinaryOp::Sub,
        BvBinaryOp::Mul,
        BvBinaryOp::Udiv,
        BvBinaryOp::Sdiv,
        BvBinaryOp::Urem,
        BvBinaryOp::Srem,
        BvBinaryOp::Shl,
        BvBinaryOp::Lshr,
        BvBinaryOp::Ashr,
    ];

    const COMPARE_OPS: [BvCompareOp; 4] = [
        BvCompareOp::Ult,
        BvCompareOp::Ule,
        BvCompareOp::Slt,
        BvCompareOp::Sle,
    ];

    /// The widths the design names: 1 (degenerate), 7 (odd, sub-byte), 8, 64
    /// (the limb boundary), 65 (one bit past it — where the 0.3.3/0.3.4
    /// `encode_add_const` shift bug lived) and 128.
    const WIDTHS: [u32; 6] = [1, 7, 8, 64, 65, 128];

    /// A deterministic xorshift, so the sampled pairs are the same on every
    /// run and on every platform without pulling in a random-number crate.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn next_u128(&mut self) -> u128 {
            (u128::from(self.next_u64()) << 64) | u128::from(self.next_u64())
        }
    }

    /// The values every width is exercised at: the structural corners, plus
    /// deterministic random samples.
    fn samples(width: u32, rng: &mut Rng) -> Vec<u128> {
        let m = mask(width);
        let sign_bit = 1u128 << (width - 1);
        let mut values = vec![
            0,
            1 & m,
            2 & m,
            m,
            sign_bit,
            (sign_bit.wrapping_sub(1)) & m,
            (sign_bit.wrapping_add(1)) & m,
            u128::from(width) & m,
            (u128::from(width).wrapping_add(1)) & m,
            (u128::from(width).wrapping_sub(1)) & m,
        ];
        for _ in 0..8 {
            values.push(rng.next_u128() & m);
        }
        values.sort_unstable();
        values.dedup();
        values
    }

    /// The bit-vector payload of an outcome, or `None` for anything else.
    fn bits_of(outcome: EvalOutcome) -> Option<(BigInt, u32)> {
        match outcome {
            EvalOutcome::Value(EvalVal::Bv { value, width }) => Some((value, width)),
            _ => None,
        }
    }

    /// The truth payload of an outcome, or `None` for anything else.
    fn truth_of(outcome: EvalOutcome) -> Option<bool> {
        match outcome {
            EvalOutcome::Value(EvalVal::Bool(b)) => Some(b),
            _ => None,
        }
    }

    /// An [`EvalVal::Bv`] built the way [`leaf`] builds one.
    fn bv(value: u128, width: u32) -> EvalVal {
        EvalVal::Bv {
            value: BigInt::from(value),
            width,
        }
    }

    /// Every binary operator, at every width, against the `u128` reference.
    ///
    /// Before this evaluator existed the gate had no bit-vector arm at all —
    /// OxiZ 0.3.3/0.3.4 answered `Undetermined` for every one of these terms,
    /// which is why `u01` came back `sat` with a model that falsifies it.
    #[test]
    fn every_binary_operator_matches_the_reference() {
        let mut rng = Rng(0x2026_0908_0000_0001);
        for width in WIDTHS {
            let values = samples(width, &mut rng);
            for &a in &values {
                for &b in &values {
                    let (left, right) = (BigInt::from(a), BigInt::from(b));
                    for op in BINARY_OPS {
                        let expected = reference_binary(op, a, b, width);
                        let got = bits_of(binary(op, &left, width, &right, width));
                        assert_eq!(
                            got,
                            Some((BigInt::from(expected), width)),
                            "{op:?} at width {width} on {a:#x}, {b:#x}"
                        );
                    }
                }
            }
        }
    }

    /// Every comparison, at every width, against the `u128` reference.
    #[test]
    fn every_comparison_matches_the_reference() {
        let mut rng = Rng(0x2026_0908_0000_0002);
        for width in WIDTHS {
            let values = samples(width, &mut rng);
            for &a in &values {
                for &b in &values {
                    let (left, right) = (BigInt::from(a), BigInt::from(b));
                    for op in COMPARE_OPS {
                        assert_eq!(
                            truth_of(compare(op, &left, width, &right, width)),
                            Some(reference_compare(op, a, b, width)),
                            "{op:?} at width {width} on {a:#x}, {b:#x}"
                        );
                    }
                    assert_eq!(
                        truth_of(equal(&left, width, &right, width)),
                        Some(a == b),
                        "= at width {width} on {a:#x}, {b:#x}"
                    );
                }
            }
        }
    }

    /// `bvnot` at every width.
    #[test]
    fn complement_matches_the_reference() {
        let mut rng = Rng(0x2026_0908_0000_0003);
        for width in WIDTHS {
            for value in samples(width, &mut rng) {
                assert_eq!(
                    bits_of(complement(&BigInt::from(value), width)),
                    Some((BigInt::from((!value) & mask(width)), width)),
                    "bvnot at width {width} on {value:#x}"
                );
            }
        }
    }

    /// `(_ extract high low)` over every legal index pair of the narrow
    /// widths, plus the two boundaries of the wide ones.
    #[test]
    fn extract_matches_the_reference() {
        let mut rng = Rng(0x2026_0908_0000_0004);
        for width in WIDTHS {
            for value in samples(width, &mut rng) {
                let indices: Vec<u32> = if width <= 8 {
                    (0..width).collect()
                } else {
                    vec![0, 1, width / 2, width - 2, width - 1]
                };
                for &high in &indices {
                    for &low in &indices {
                        if low > high {
                            continue;
                        }
                        let span = high - low + 1;
                        let expected = (value >> low) & mask(span);
                        assert_eq!(
                            bits_of(extract(high, low, &BigInt::from(value), width)),
                            Some((BigInt::from(expected), span)),
                            "extract {high}:{low} at width {width} on {value:#x}"
                        );
                    }
                }
            }
        }
    }

    /// `concat` at width pairs whose sum still fits the `u128` reference.
    #[test]
    fn concat_matches_the_reference() {
        let mut rng = Rng(0x2026_0908_0000_0005);
        for left_width in [1u32, 7, 8, 64] {
            for right_width in [1u32, 7, 8, 64] {
                let joined = left_width + right_width;
                if joined > 128 {
                    continue;
                }
                for a in samples(left_width, &mut rng) {
                    for b in samples(right_width, &mut rng) {
                        let expected = ((a << right_width) | b) & mask(joined);
                        assert_eq!(
                            bits_of(binary(
                                BvBinaryOp::Concat,
                                &BigInt::from(a),
                                left_width,
                                &BigInt::from(b),
                                right_width,
                            )),
                            Some((BigInt::from(expected), joined)),
                            "concat {left_width}+{right_width} on {a:#x}, {b:#x}"
                        );
                    }
                }
            }
        }
    }

    /// The division family at a zero divisor, spelled out at every width
    /// rather than left to the sampled sweep.
    ///
    /// SMT-LIB makes all four total: `bvudiv s 0` is all ones, `bvurem s 0` is
    /// `s`, `bvsdiv s 0` is `1` for a negative `s` and all ones otherwise, and
    /// `bvsrem s 0` is `s`.
    #[test]
    fn division_by_zero_follows_the_standard() {
        for width in WIDTHS {
            let zero = BigInt::ZERO;
            let ones = BigInt::from(mask(width));
            let negative = BigInt::from(mask(width)); // all ones is -1 signed
            let positive = BigInt::from(mask(width) >> 1); // the largest positive

            assert_eq!(
                bits_of(binary(BvBinaryOp::Udiv, &positive, width, &zero, width)),
                Some((ones.clone(), width))
            );
            assert_eq!(
                bits_of(binary(BvBinaryOp::Urem, &positive, width, &zero, width)),
                Some((positive.clone(), width))
            );
            assert_eq!(
                bits_of(binary(BvBinaryOp::Sdiv, &negative, width, &zero, width)),
                Some((BigInt::from(1u8), width)),
                "bvsdiv of a negative dividend by zero is 1, at width {width}"
            );
            assert_eq!(
                bits_of(binary(BvBinaryOp::Sdiv, &zero, width, &zero, width)),
                Some((ones.clone(), width))
            );
            assert_eq!(
                bits_of(binary(BvBinaryOp::Srem, &negative, width, &zero, width)),
                Some((negative.clone(), width))
            );
        }
    }

    /// Shifting by at least the width is specified, not undefined: `bvshl` and
    /// `bvlshr` vanish, `bvashr` saturates to the sign bit.  The distance is
    /// itself a full-width bit-vector, so at width 65 and 128 it can be far
    /// larger than any `u32` — the shape that broke a duplicate folder in this
    /// crate before (`bvshl x (_ bv2^64 65)` folded to `x`).
    #[test]
    fn shifts_at_and_beyond_the_width_saturate() {
        for width in WIDTHS {
            let ones = BigInt::from(mask(width));
            let zero = BigInt::ZERO;
            let distances = [
                BigInt::from(width),
                BigInt::from(width + 1),
                BigInt::from(mask(width)),
                // 2^64 reduced into the width: zero below 65 bits, a genuinely
                // huge distance at 65 and 128.
                BigInt::from(u128::from(u64::MAX) + 1) % (BigInt::from(1u8) << width),
            ];
            for distance in distances {
                if distance < BigInt::from(width) {
                    continue;
                }
                assert_eq!(
                    bits_of(binary(BvBinaryOp::Shl, &ones, width, &distance, width)),
                    Some((zero.clone(), width)),
                    "bvshl by {distance} at width {width}"
                );
                assert_eq!(
                    bits_of(binary(BvBinaryOp::Lshr, &ones, width, &distance, width)),
                    Some((zero.clone(), width)),
                    "bvlshr by {distance} at width {width}"
                );
                assert_eq!(
                    bits_of(binary(BvBinaryOp::Ashr, &ones, width, &distance, width)),
                    Some((ones.clone(), width)),
                    "bvashr of all-ones by {distance} at width {width}"
                );
                assert_eq!(
                    bits_of(binary(BvBinaryOp::Ashr, &zero, width, &distance, width)),
                    Some((zero.clone(), width)),
                    "bvashr of zero by {distance} at width {width}"
                );
            }
        }
    }

    /// Widths past `u128`, where the reference implementation cannot follow.
    ///
    /// The expectations are written structurally (`2^200 - 1`, `2^199`) rather
    /// than computed, so they do not restate the implementation.  A 64-bit
    /// window onto these values — the shape of every wide-BV bug this crate
    /// has fixed — gets all four of them wrong.
    #[test]
    fn wide_widths_are_not_truncated_to_64_bits() {
        const WIDTH: u32 = 200;
        let one = BigInt::from(1u8);
        let modulus: BigInt = one.clone() << WIDTH;
        let ones: BigInt = modulus.clone() - &one;
        let high: BigInt = one.clone() << (WIDTH - 1); // 2^199, the sign bit
        let two_to_the_64: BigInt = one.clone() << 64u32;

        // 2^199 + 2^199 = 2^200 = 0 (mod 2^200).
        assert_eq!(
            bits_of(binary(BvBinaryOp::Add, &high, WIDTH, &high, WIDTH)),
            Some((BigInt::ZERO, WIDTH))
        );
        // 2^64 * 2^64 = 2^128, which a 64-bit window folds to 0.
        assert_eq!(
            bits_of(binary(
                BvBinaryOp::Mul,
                &two_to_the_64,
                WIDTH,
                &two_to_the_64,
                WIDTH
            )),
            Some((one.clone() << 128u32, WIDTH))
        );
        // A shift distance of 2^64 is >= the width, so `bvshl` vanishes and
        // `bvashr` of a negative value saturates.
        assert_eq!(
            bits_of(binary(BvBinaryOp::Shl, &ones, WIDTH, &two_to_the_64, WIDTH)),
            Some((BigInt::ZERO, WIDTH))
        );
        assert_eq!(
            bits_of(binary(
                BvBinaryOp::Ashr,
                &ones,
                WIDTH,
                &two_to_the_64,
                WIDTH
            )),
            Some((ones.clone(), WIDTH))
        );
        // 2^64 is neither 0 nor equal to itself modulo 2^64 only: unsigned
        // comparison must see the whole value.
        assert_eq!(
            truth_of(compare(
                BvCompareOp::Ult,
                &BigInt::ZERO,
                WIDTH,
                &two_to_the_64,
                WIDTH
            )),
            Some(true)
        );
        assert_eq!(
            truth_of(equal(&BigInt::ZERO, WIDTH, &two_to_the_64, WIDTH)),
            Some(false)
        );
        // All ones is -1 signed at any width.
        assert_eq!(
            truth_of(compare(
                BvCompareOp::Slt,
                &ones,
                WIDTH,
                &BigInt::ZERO,
                WIDTH
            )),
            Some(true)
        );
        // bvudiv by zero is all ones, at 200 bits as at 8.
        assert_eq!(
            bits_of(binary(
                BvBinaryOp::Udiv,
                &two_to_the_64,
                WIDTH,
                &BigInt::ZERO,
                WIDTH
            )),
            Some((ones, WIDTH))
        );
    }

    /// A leaf is reduced into `[0, 2^width)` on the way in: `mk_bitvec`
    /// interns a negative or oversized literal unchanged, and every rule here
    /// has the reduced range as a precondition.
    #[test]
    fn leaves_are_reduced_into_range() {
        assert_eq!(
            bits_of(leaf(&BigInt::from(-1), 8)),
            Some((BigInt::from(255u8), 8))
        );
        assert_eq!(
            bits_of(leaf(&BigInt::from(256), 8)),
            Some((BigInt::ZERO, 8))
        );
        assert_eq!(
            bits_of(leaf(&(BigInt::from(1u8) << 200u32), 65)),
            Some((BigInt::ZERO, 65))
        );
    }

    /// Every rule declines rather than guesses when the two operand widths
    /// disagree — `concat`, whose operands need not agree, excepted.
    #[test]
    fn unequal_widths_produce_no_value() {
        let a = BigInt::from(3u8);
        let b = BigInt::from(3u8);
        for op in BINARY_OPS {
            assert_eq!(
                binary(op, &a, 8, &b, 16),
                EvalOutcome::Undetermined,
                "{op:?} at mismatched widths"
            );
        }
        for op in COMPARE_OPS {
            assert_eq!(compare(op, &a, 8, &b, 16), EvalOutcome::Undetermined);
        }
        assert_eq!(equal(&a, 8, &b, 16), EvalOutcome::Undetermined);
        assert_eq!(
            all_distinct(&[bv(1, 8), bv(1, 16)]),
            EvalOutcome::Undetermined
        );
        // `concat` is the exception and folds to the joined width.
        assert_eq!(
            bits_of(binary(BvBinaryOp::Concat, &a, 8, &b, 16)),
            Some((BigInt::from(3u32 + (3 << 16)), 24))
        );
    }

    /// A malformed `extract` index pair produces no value rather than a
    /// fabricated one.
    #[test]
    fn malformed_extract_indices_produce_no_value() {
        let value = BigInt::from(0xffu32);
        assert_eq!(extract(8, 0, &value, 8), EvalOutcome::Undetermined);
        assert_eq!(extract(3, 5, &value, 8), EvalOutcome::Undetermined);
        assert_eq!(extract(u32::MAX, 0, &value, 8), EvalOutcome::Undetermined);
    }

    /// `distinct` is decided only when every operand is a bit-vector of one
    /// width; a single non-bit-vector operand leaves it inconclusive, which is
    /// what the gate answered for every `distinct` before this evaluator.
    #[test]
    fn distinct_decides_only_uniform_bit_vectors() {
        assert_eq!(
            all_distinct(&[bv(1, 8), bv(2, 8), bv(3, 8)]),
            EvalOutcome::Value(EvalVal::Bool(true))
        );
        assert_eq!(
            all_distinct(&[bv(1, 8), bv(2, 8), bv(1, 8)]),
            EvalOutcome::Value(EvalVal::Bool(false))
        );
        assert_eq!(
            all_distinct(&[bv(1, 8), EvalVal::Bool(true)]),
            EvalOutcome::Undetermined
        );
        assert_eq!(all_distinct(&[bv(1, 8)]), EvalOutcome::Undetermined);
    }

    // ─────────────────────────────────────────────────────────────────
    // The gate itself
    // ─────────────────────────────────────────────────────────────────

    /// A solver holding exactly `assertions` and `model`, ready for the gate.
    fn solver_with(assertions: Vec<TermId>, model: Model) -> Solver {
        let mut solver = Solver::new();
        solver.assertions = assertions;
        solver.model = Some(model);
        solver
    }

    /// `u01`'s shape: an 8-bit variable `a` whose model witness is `value`,
    /// asserted `(not (and (bvule a #x0f) (bvule a #x10)))`.
    fn u01_gate(witness: u32) -> bool {
        let mut manager = TermManager::new();
        let sort = manager.sorts.bitvec(8);
        let a = manager.mk_var("a", sort);
        let fifteen = manager.mk_bitvec(0x0f, 8);
        let sixteen = manager.mk_bitvec(0x10, 8);
        let low = manager.mk_bv_ule(a, fifteen);
        let high = manager.mk_bv_ule(a, sixteen);
        let both = manager.mk_and([low, high]);
        let assertion = manager.mk_not(both);

        let witness_term = manager.mk_bitvec(witness, 8);
        let mut model = Model::new();
        model.set(a, witness_term);
        solver_with(vec![assertion], model).model_refutes_assertions(&manager)
    }

    /// The `u01` model OxiZ 0.3.3/0.3.4 reported: `a = #b00001111`.  Both
    /// `bvule`s hold (15 <= 15 and 15 <=u 16), so the conjunction is true and
    /// the assertion is **false** — the model provably violates it and the
    /// gate must refuse the verdict.
    ///
    /// Before this evaluator the gate answered `false` here: `EvalVal` had no
    /// bit-vector variant, every `Bv*` term fell into `open_in_model`'s
    /// closing arm, and the assertion evaluated `Undetermined`.  0.3.3 and
    /// 0.3.4 therefore reported `sat` with exactly this model.
    #[test]
    fn the_u01_model_is_refused() {
        assert!(
            u01_gate(0x0f),
            "a = #x0f makes (not (and (bvule a #x0f) (bvule a #x10))) false"
        );
    }

    /// The control: a model that genuinely satisfies the same assertion must
    /// pass.  `a = #x20` (32) is above both bounds, so both `bvule`s are
    /// false, the conjunction is false and the assertion is true.  Without
    /// this the test above would also pass if the gate simply refused every
    /// bit-vector model.
    #[test]
    fn a_satisfying_model_of_the_same_assertion_passes() {
        assert!(!u01_gate(0x20), "a = #x20 satisfies the assertion");
    }

    /// A definite bit-vector refutation nested under arithmetic-free Boolean
    /// structure still reaches the gate, and one the model satisfies does not.
    #[test]
    fn bit_vector_structure_refutes_only_when_it_should() {
        let mut manager = TermManager::new();
        let sort = manager.sorts.bitvec(64);
        let x = manager.mk_var("x", sort);
        let y = manager.mk_var("y", sort);
        let sum = manager.mk_bv_add(x, y);
        let zero = manager.mk_bitvec(0, 64);
        // (= (bvadd x y) #x0000000000000000) with x = 1, y = 1: false.
        let assertion = manager.mk_eq(sum, zero);
        let one_term = manager.mk_bitvec(1, 64);
        let mut model = Model::new();
        model.set(x, one_term);
        model.set(y, one_term);
        assert!(
            solver_with(vec![assertion], model).model_refutes_assertions(&manager),
            "1 + 1 is 2, not 0"
        );

        // The wrap-around witness the same assertion does have: x = 1,
        // y = 2^64 - 1.
        let all_ones = manager.mk_bitvec(u64::MAX, 64);
        let mut model = Model::new();
        model.set(x, one_term);
        model.set(y, all_ones);
        assert!(
            !solver_with(vec![assertion], model).model_refutes_assertions(&manager),
            "1 + (2^64 - 1) wraps to 0"
        );
    }

    /// An equality between operands of *different* widths is an ill-sorted
    /// term, and the gate has nothing to say about it — never a refutation.
    ///
    /// Built through the manager rather than the parser, which rejects it.
    #[test]
    fn an_unequal_width_equality_stays_inconclusive() {
        let mut manager = TermManager::new();
        let narrow_sort = manager.sorts.bitvec(8);
        let wide_sort = manager.sorts.bitvec(16);
        let a = manager.mk_var("a", narrow_sort);
        let b = manager.mk_var("b", wide_sort);
        let assertion = manager.mk_eq(a, b);

        let narrow = manager.mk_bitvec(1, 8);
        let wide = manager.mk_bitvec(2, 16);
        let mut model = Model::new();
        model.set(a, narrow);
        model.set(b, wide);
        assert!(
            !solver_with(vec![assertion], model).model_refutes_assertions(&manager),
            "1 (8 bits) and 2 (16 bits) are different numbers, but the term is \
             ill sorted and the gate must not rule on it"
        );
    }

    /// A bit-vector variable the model does not pin at all leaves the
    /// assertion inconclusive rather than defaulting it to zero.
    ///
    /// The control in the second half matters: `(bvult a #x00)` would also
    /// pass the first assertion, but only because the term builder folds it to
    /// `false` outright.  Pinning a witness that genuinely falsifies the
    /// assertion proves the term reaches the evaluator at all.
    #[test]
    fn an_unpinned_bit_vector_variable_stays_inconclusive() {
        let mut manager = TermManager::new();
        let sort = manager.sorts.bitvec(8);
        let a = manager.mk_var("a", sort);
        let five = manager.mk_bitvec(5, 8);
        let assertion = manager.mk_bv_ult(a, five);
        assert!(
            !solver_with(vec![assertion], Model::new()).model_refutes_assertions(&manager),
            "the model pins no witness for `a`, so `(bvult a #x05)` has no value"
        );

        let nine = manager.mk_bitvec(9, 8);
        let mut model = Model::new();
        model.set(a, nine);
        assert!(
            solver_with(vec![assertion], model).model_refutes_assertions(&manager),
            "9 is not <u 5"
        );
    }
}
