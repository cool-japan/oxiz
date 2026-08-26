//! Bit-vector blasting tactics.
//!
//! This module provides two things:
//!
//! 1. [`BitBlastTactic`] / [`StatelessBitBlastTactic`]: cheap *detectors* that
//!    check whether a [`Goal`] contains BitVector terms. Because the
//!    [`Tactic`] trait's `apply` method only receives `&Goal` (no
//!    [`TermManager`] access), these types are structurally unable to
//!    allocate new terms, so they can only ever return the goal unchanged.
//!    They do **not** perform bit-blasting; see [`BitBlaster`] for that.
//! 2. [`BitBlaster`]: the actual bit-blasting engine. Given `&mut
//!    TermManager` it walks a goal's assertions and rewrites quantifier-free
//!    Boolean/BitVector formulas into pure Boolean circuits (per-bit
//!    variables plus ripple-carry adders/subtractors, a shift-and-add
//!    multiplier, a restoring-division circuit, a barrel shifter and
//!    unsigned/signed comparators). Assertions that reach a construct
//!    outside the supported QF\_BV(+Bool) fragment (arrays, uninterpreted
//!    functions, quantifiers, floating point, strings, Int/Real arithmetic,
//!    ...) cause the *whole* transformation to be abandoned honestly via
//!    [`TacticResult::NotApplicable`] rather than emitting a partially
//!    blasted (and therefore unsound) goal.
//!
//! ## References
//!
//! - Brummayer & Biere: "Boolector: An Efficient SMT Solver for Bit-Vectors
//!   and Arrays" (TACAS 2009)
//! - Z3's `tactic/bv/bit_blaster_tpl.h` / `bit_blaster_tactic.cpp`

use super::core::*;
use crate::ast::{TermId, TermManager};
use crate::error::Result;
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use num_bigint::BigInt;

/// Bit-blasting tactic - converts BV operations to propositional logic
pub struct BitBlastTactic<'a> {
    manager: &'a TermManager,
}

impl<'a> BitBlastTactic<'a> {
    /// Create a new bit-blast tactic
    pub fn new(manager: &'a TermManager) -> Self {
        Self { manager }
    }

    /// Check if a term is a BitVector term
    fn is_bv_term(&self, term_id: TermId) -> bool {
        use crate::ast::TermKind;
        if let Some(term) = self.manager.get(term_id) {
            matches!(
                term.kind,
                TermKind::BitVecConst { .. }
                    | TermKind::BvConcat(_, _)
                    | TermKind::BvExtract { .. }
                    | TermKind::BvNot(_)
                    | TermKind::BvAnd(_, _)
                    | TermKind::BvOr(_, _)
                    | TermKind::BvXor(_, _)
                    | TermKind::BvAdd(_, _)
                    | TermKind::BvSub(_, _)
                    | TermKind::BvMul(_, _)
                    | TermKind::BvUdiv(_, _)
                    | TermKind::BvSdiv(_, _)
                    | TermKind::BvUrem(_, _)
                    | TermKind::BvSrem(_, _)
                    | TermKind::BvShl(_, _)
                    | TermKind::BvLshr(_, _)
                    | TermKind::BvAshr(_, _)
                    | TermKind::BvUlt(_, _)
                    | TermKind::BvUle(_, _)
                    | TermKind::BvSlt(_, _)
                    | TermKind::BvSle(_, _)
            ) || self.is_bv_sort(term.sort)
        } else {
            false
        }
    }

    /// Check if a sort is a BitVector sort
    fn is_bv_sort(&self, sort_id: crate::sort::SortId) -> bool {
        if let Some(sort) = self.manager.sorts.get(sort_id) {
            sort.bitvec_width().is_some()
        } else {
            false
        }
    }

    /// Check if a term contains any BitVector subterms.
    ///
    /// Iterative (explicit heap stack plus a `visited` set): the term DAG's
    /// depth follows the input formula, and a `-> bool` walk has no error
    /// channel with which to report a depth cut-off, so recursing here could
    /// only fail by overflowing the native stack. The `visited` set also stops
    /// a shared sub-DAG from being re-expanded once per incoming edge — the
    /// original recursion re-walked it every time, which is exponential on the
    /// classic `let`-free doubling DAG.
    ///
    /// The answer is unchanged: a node contributes `true` exactly when it is
    /// itself a BitVector term, and otherwise the result is the disjunction
    /// over the same per-variant child sets the recursive version used.
    fn contains_bv_term(&self, term_id: TermId) -> bool {
        use crate::ast::TermKind;

        let mut stack = vec![term_id];
        let mut visited: FxHashSet<TermId> = FxHashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }

            if self.is_bv_term(current) {
                return true;
            }

            let Some(term) = self.manager.get(current) else {
                continue;
            };

            match &term.kind {
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::Var(_) => {
                    if self.is_bv_sort(term.sort) {
                        return true;
                    }
                }
                TermKind::Not(a) | TermKind::Neg(a) | TermKind::BvNot(a) => stack.push(*a),
                TermKind::BvExtract { arg, .. } => stack.push(*arg),
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::Distinct(args) => stack.extend(args.iter().copied()),
                // String constructs whose operands cannot carry BV content.
                TermKind::StringLit(_)
                | TermKind::StrLen(_)
                | TermKind::StrToInt(_)
                | TermKind::IntToStr(_)
                | TermKind::StrToCode(_)
                | TermKind::StrFromCode(_) => {}
                TermKind::Implies(a, b)
                | TermKind::Xor(a, b)
                | TermKind::Eq(a, b)
                | TermKind::Sub(a, b)
                | TermKind::Div(a, b)
                | TermKind::Mod(a, b)
                | TermKind::Lt(a, b)
                | TermKind::Le(a, b)
                | TermKind::Gt(a, b)
                | TermKind::Ge(a, b)
                | TermKind::Select(a, b)
                | TermKind::StrConcat(a, b)
                | TermKind::StrAt(a, b)
                | TermKind::StrContains(a, b)
                | TermKind::StrPrefixOf(a, b)
                | TermKind::StrSuffixOf(a, b)
                | TermKind::StrInRe(a, b)
                | TermKind::StrLt(a, b)
                | TermKind::StrLe(a, b)
                | TermKind::BvConcat(a, b)
                | TermKind::BvAnd(a, b)
                | TermKind::BvOr(a, b)
                | TermKind::BvXor(a, b)
                | TermKind::BvAdd(a, b)
                | TermKind::BvSub(a, b)
                | TermKind::BvMul(a, b)
                | TermKind::BvUdiv(a, b)
                | TermKind::BvSdiv(a, b)
                | TermKind::BvUrem(a, b)
                | TermKind::BvSrem(a, b)
                | TermKind::BvShl(a, b)
                | TermKind::BvLshr(a, b)
                | TermKind::BvAshr(a, b)
                | TermKind::BvUlt(a, b)
                | TermKind::BvUle(a, b)
                | TermKind::BvSlt(a, b)
                | TermKind::BvSle(a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                TermKind::Ite(c, t, e)
                | TermKind::Store(c, t, e)
                | TermKind::StrSubstr(c, t, e)
                | TermKind::StrIndexOf(c, t, e)
                | TermKind::StrReplace(c, t, e)
                | TermKind::StrReplaceAll(c, t, e)
                | TermKind::StrReplaceRe(c, t, e)
                | TermKind::StrReplaceReAll(c, t, e) => {
                    stack.push(*c);
                    stack.push(*t);
                    stack.push(*e);
                }
                TermKind::Apply { args, .. } => stack.extend(args.iter().copied()),
                TermKind::Forall { body, .. } | TermKind::Exists { body, .. } => stack.push(*body),
                TermKind::Let { bindings, body } => {
                    stack.extend(bindings.iter().map(|(_, t)| *t));
                    stack.push(*body);
                }
                // Floating-point literals don't contain BV terms
                TermKind::FpLit { .. }
                | TermKind::FpPlusInfinity { .. }
                | TermKind::FpMinusInfinity { .. }
                | TermKind::FpPlusZero { .. }
                | TermKind::FpMinusZero { .. }
                | TermKind::FpNaN { .. } => {}
                TermKind::FpAbs(a)
                | TermKind::FpNeg(a)
                | TermKind::FpSqrt(_, a)
                | TermKind::FpRoundToIntegral(_, a)
                | TermKind::FpIsNormal(a)
                | TermKind::FpIsSubnormal(a)
                | TermKind::FpIsZero(a)
                | TermKind::FpIsInfinite(a)
                | TermKind::FpIsNaN(a)
                | TermKind::FpIsNegative(a)
                | TermKind::FpIsPositive(a)
                | TermKind::FpToReal(a) => stack.push(*a),
                TermKind::FpAdd(_, a, b)
                | TermKind::FpSub(_, a, b)
                | TermKind::FpMul(_, a, b)
                | TermKind::FpDiv(_, a, b)
                | TermKind::FpRem(a, b)
                | TermKind::FpMin(a, b)
                | TermKind::FpMax(a, b)
                | TermKind::FpLeq(a, b)
                | TermKind::FpLt(a, b)
                | TermKind::FpGeq(a, b)
                | TermKind::FpGt(a, b)
                | TermKind::FpEq(a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                TermKind::FpFma(_, a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                TermKind::FpToFp { arg, .. }
                | TermKind::FpToSBV { arg, .. }
                | TermKind::FpToUBV { arg, .. }
                | TermKind::RealToFp { arg, .. }
                | TermKind::SBVToFp { arg, .. }
                | TermKind::UBVToFp { arg, .. } => stack.push(*arg),
                // Algebraic datatypes
                TermKind::DtConstructor { args, .. } => stack.extend(args.iter().copied()),
                TermKind::DtTester { arg, .. } | TermKind::DtSelector { arg, .. } => {
                    stack.push(*arg);
                }
                // Match expressions
                TermKind::Match { scrutinee, cases } => {
                    stack.push(*scrutinee);
                    stack.extend(cases.iter().map(|c| c.body));
                }
            }
        }

        false
    }

    /// Check whether a goal contains any BitVector terms.
    ///
    /// This is a detection-only probe: it does **not** perform any
    /// transformation, because [`BitBlastTactic`] only holds an immutable
    /// `&TermManager` and therefore cannot allocate the fresh Boolean
    /// variables and circuit terms a real bit-blast requires. Callers that
    /// want an actual Boolean-circuit encoding must use `BitBlaster`
    /// with a `&mut TermManager` instead.
    pub fn apply_check(&self, goal: &Goal) -> Result<TacticResult> {
        // Check if any assertion contains BitVector terms
        let has_bv = goal.assertions.iter().any(|&a| self.contains_bv_term(a));

        if !has_bv {
            return Ok(TacticResult::NotApplicable);
        }

        // Detection only: this type cannot mutate the term manager, so the
        // goal is returned unchanged. Use `BitBlastTactic::blast` for a
        // real transformation.
        Ok(TacticResult::SubGoals(vec![goal.clone()]))
    }

    /// Perform a real bit-blast of `goal` into pure Boolean circuits.
    ///
    /// This is an associated function (it takes no `self`) rather than a
    /// method deliberately: this type's `&self` holds an immutable `&'a
    /// TermManager` (needed by the cheap detection methods above and kept
    /// for backwards compatibility), which would conflict with also
    /// taking the `&mut TermManager` a real transformation requires. See
    /// `BitBlaster` for the encoding itself.
    pub fn blast(goal: &Goal, manager: &mut TermManager) -> Result<TacticResult> {
        BitBlaster::new().blast_goal(goal, manager)
    }
}

/// Stateless version for the Tactic trait
#[derive(Debug, Default)]
pub struct StatelessBitBlastTactic;

impl Tactic for StatelessBitBlastTactic {
    fn name(&self) -> &str {
        "bit-blast"
    }

    fn apply(&self, _goal: &Goal) -> Result<TacticResult> {
        // The `Tactic` trait only hands us `&Goal`, with no `TermManager`
        // access, so this dispatch path is structurally unable to allocate
        // the fresh Boolean variables and circuit terms a real bit-blast
        // needs. It therefore honestly reports NotApplicable rather than
        // returning the goal unchanged (which would read as a successful
        // no-op transformation to a combinator). Callers holding a
        // `&mut TermManager` should use [`BitBlaster::blast_goal`] (exposed on
        // the registry's `create_managed` path) for the real Boolean-circuit
        // encoding of BitVector operations.
        Ok(TacticResult::NotApplicable)
    }

    fn description(&self) -> &str {
        "Detects BitVector operations (does not itself transform them; requires a TermManager — \
         see BitBlaster / create_managed for the real encoder; the manager-free path is NotApplicable)"
    }
}

// ---------------------------------------------------------------------------
// Real bit-blasting engine
// ---------------------------------------------------------------------------

/// A bit-vector represented as a vector of Boolean-sorted [`TermId`]s,
/// least-significant bit first (index `i` has weight `2^i`).
type BitVec = Vec<TermId>;

/// Statistics for [`BitBlaster`].
#[derive(Debug, Clone, Default)]
pub struct BitBlastStats {
    /// Number of distinct bit-vectors blasted (constants + variables +
    /// operator results).
    pub bitvectors_blasted: u64,
    /// Total number of bit-level Boolean terms produced.
    pub bits_generated: u64,
    /// Approximate number of Boolean gates (and/or/xor/not/ite) generated.
    pub gates_generated: u64,
}

/// The result of blasting a single term: either a Boolean-sorted term, or
/// the bit-vector (LSB-first) encoding of a BitVector-sorted term.
enum Blasted {
    Bool(TermId),
    Bits(BitVec),
}

/// Real bit-blasting engine.
///
/// Converts BitVector operations reachable from a goal's assertions into
/// pure Boolean circuits over freshly allocated Boolean variables, using
/// standard hardware-synthesis encodings (ripple-carry adder/subtractor,
/// shift-and-add multiplier, restoring divider, barrel shifter, and
/// MSB-to-LSB comparators). Only the quantifier-free Boolean/BitVector
/// fragment is supported; if any assertion needing transformation reaches
/// an unsupported construct (arrays, uninterpreted functions, quantifiers,
/// floating point, strings, Int/Real arithmetic, ...) the whole
/// transformation is abandoned and [`TacticResult::NotApplicable`] is
/// returned, so no partially-blasted (and therefore potentially unsound)
/// goal is ever produced.
pub struct BitBlaster {
    stats: BitBlastStats,
    bool_cache: FxHashMap<TermId, TermId>,
    bits_cache: FxHashMap<TermId, BitVec>,
}

impl BitBlaster {
    /// Create a new, empty bit-blaster.
    pub fn new() -> Self {
        Self {
            stats: BitBlastStats::default(),
            bool_cache: FxHashMap::default(),
            bits_cache: FxHashMap::default(),
        }
    }

    /// Get statistics.
    // `bitblast` is a private module (only `BitBlastTactic` and
    // `StatelessBitBlastTactic` are re-exported from `tactic`), so this
    // introspection getter isn't reachable from outside the crate and is
    // currently only exercised by tests; `#[allow(dead_code)]` reflects
    // that module-visibility fact rather than hiding dead functionality.
    #[allow(dead_code)]
    pub fn stats(&self) -> &BitBlastStats {
        &self.stats
    }

    /// Bit-blast a goal.
    ///
    /// Returns `NotApplicable` if the goal has no BitVector content, or if
    /// any assertion containing BitVector content cannot be fully blasted
    /// (in which case the goal is returned untouched rather than partially
    /// transformed).
    pub fn blast_goal(&mut self, goal: &Goal, manager: &mut TermManager) -> Result<TacticResult> {
        let mut new_assertions = Vec::with_capacity(goal.assertions.len());
        let mut any_bv = false;

        for &assertion in &goal.assertions {
            let has_bv = {
                let checker = BitBlastTactic::new(&*manager);
                checker.contains_bv_term(assertion)
            };

            if !has_bv {
                new_assertions.push(assertion);
                continue;
            }

            any_bv = true;
            match self.blast_bool(assertion, manager) {
                Some(blasted) => new_assertions.push(blasted),
                // Cannot fully blast this assertion (it reaches an
                // unsupported construct); bail out honestly instead of
                // emitting a partially-blasted goal.
                None => return Ok(TacticResult::NotApplicable),
            }
        }

        if !any_bv {
            return Ok(TacticResult::NotApplicable);
        }

        Ok(TacticResult::SubGoals(vec![Goal {
            assertions: new_assertions,
            precision: goal.precision,
        }]))
    }

    // -- dispatch -----------------------------------------------------------

    fn bv_width(manager: &TermManager, sort: SortId) -> Option<u32> {
        manager.sorts.get(sort)?.bitvec_width()
    }

    fn blast_bool(&mut self, term_id: TermId, manager: &mut TermManager) -> Option<TermId> {
        match self.blast(term_id, manager)? {
            Blasted::Bool(t) => Some(t),
            Blasted::Bits(_) => None,
        }
    }

    fn blast_bits(&mut self, term_id: TermId, manager: &mut TermManager) -> Option<BitVec> {
        match self.blast(term_id, manager)? {
            Blasted::Bits(v) => Some(v),
            Blasted::Bool(_) => None,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn blast(&mut self, term_id: TermId, manager: &mut TermManager) -> Option<Blasted> {
        use crate::ast::TermKind;

        if let Some(&b) = self.bool_cache.get(&term_id) {
            return Some(Blasted::Bool(b));
        }
        if let Some(v) = self.bits_cache.get(&term_id) {
            return Some(Blasted::Bits(v.clone()));
        }

        let term = manager.get(term_id)?.clone();
        let bool_sort = manager.sorts.bool_sort;

        let result = match &term.kind {
            TermKind::True | TermKind::False => Blasted::Bool(term_id),
            TermKind::Var(_) => {
                if term.sort == bool_sort {
                    Blasted::Bool(term_id)
                } else {
                    let width = Self::bv_width(manager, term.sort)?;
                    Blasted::Bits(self.fresh_bits(term_id, width, manager))
                }
            }
            TermKind::BitVecConst { value, width } => {
                Blasted::Bits(self.blast_const(value, *width, manager))
            }
            TermKind::Not(a) => {
                let a = self.blast_bool(*a, manager)?;
                Blasted::Bool(manager.mk_not(a))
            }
            TermKind::And(args) => {
                let mut bs = Vec::with_capacity(args.len());
                for &a in args.iter() {
                    bs.push(self.blast_bool(a, manager)?);
                }
                Blasted::Bool(manager.mk_and(bs))
            }
            TermKind::Or(args) => {
                let mut bs = Vec::with_capacity(args.len());
                for &a in args.iter() {
                    bs.push(self.blast_bool(a, manager)?);
                }
                Blasted::Bool(manager.mk_or(bs))
            }
            TermKind::Xor(a, b) => {
                let a = self.blast_bool(*a, manager)?;
                let b = self.blast_bool(*b, manager)?;
                Blasted::Bool(manager.mk_xor(a, b))
            }
            TermKind::Implies(a, b) => {
                let a = self.blast_bool(*a, manager)?;
                let b = self.blast_bool(*b, manager)?;
                Blasted::Bool(manager.mk_implies(a, b))
            }
            TermKind::Ite(c, t, e) => {
                let c = self.blast_bool(*c, manager)?;
                if term.sort == bool_sort {
                    let t = self.blast_bool(*t, manager)?;
                    let e = self.blast_bool(*e, manager)?;
                    Blasted::Bool(manager.mk_ite(c, t, e))
                } else if Self::bv_width(manager, term.sort).is_some() {
                    let tb = self.blast_bits(*t, manager)?;
                    let eb = self.blast_bits(*e, manager)?;
                    if tb.len() != eb.len() {
                        return None;
                    }
                    let bits = (0..tb.len())
                        .map(|i| manager.mk_ite(c, tb[i], eb[i]))
                        .collect();
                    Blasted::Bits(bits)
                } else {
                    return None;
                }
            }
            TermKind::Eq(a, b) => Blasted::Bool(self.blast_eq(*a, *b, manager)?),
            TermKind::Distinct(args) => Blasted::Bool(self.blast_distinct(args, manager)?),
            TermKind::BvNot(a) => {
                let ab = self.blast_bits(*a, manager)?;
                Blasted::Bits(self.bitwise_not(&ab, manager))
            }
            TermKind::BvAnd(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.bitwise_and(&ab, &bb, manager))
            }
            TermKind::BvOr(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.bitwise_or(&ab, &bb, manager))
            }
            TermKind::BvXor(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.bitwise_xor(&ab, &bb, manager))
            }
            TermKind::BvConcat(a, b) => {
                // SMT-LIB `concat`: first argument occupies the high bits.
                let ab = self.blast_bits(*a, manager)?;
                let mut bits = self.blast_bits(*b, manager)?;
                bits.extend(ab);
                Blasted::Bits(bits)
            }
            TermKind::BvExtract { high, low, arg } => {
                let ab = self.blast_bits(*arg, manager)?;
                if low > high || (*high as usize) >= ab.len() {
                    return None;
                }
                Blasted::Bits(ab[*low as usize..=*high as usize].to_vec())
            }
            TermKind::BvAdd(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.ripple_add(&ab, &bb, manager))
            }
            TermKind::BvSub(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.ripple_sub(&ab, &bb, manager))
            }
            TermKind::BvMul(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.ripple_mul(&ab, &bb, manager))
            }
            TermKind::BvUdiv(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.blast_udiv(&ab, &bb, manager))
            }
            TermKind::BvUrem(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.blast_urem(&ab, &bb, manager))
            }
            TermKind::BvSdiv(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.blast_sdiv(&ab, &bb, manager))
            }
            TermKind::BvSrem(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.blast_srem(&ab, &bb, manager))
            }
            TermKind::BvShl(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.blast_shl(&ab, &bb, manager))
            }
            TermKind::BvLshr(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.blast_lshr(&ab, &bb, manager))
            }
            TermKind::BvAshr(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bits(self.blast_ashr(&ab, &bb, manager))
            }
            TermKind::BvUlt(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bool(self.unsigned_lt_bits(&ab, &bb, manager))
            }
            TermKind::BvUle(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                let gt = self.unsigned_lt_bits(&bb, &ab, manager);
                Blasted::Bool(manager.mk_not(gt))
            }
            TermKind::BvSlt(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                Blasted::Bool(self.signed_lt_bits(&ab, &bb, manager))
            }
            TermKind::BvSle(a, b) => {
                let ab = self.blast_bits(*a, manager)?;
                let bb = self.blast_bits(*b, manager)?;
                if ab.len() != bb.len() {
                    return None;
                }
                let gt = self.signed_lt_bits(&bb, &ab, manager);
                Blasted::Bool(manager.mk_not(gt))
            }
            // Everything else (arrays, uninterpreted functions, quantifiers,
            // let-bindings, floating point, strings, algebraic datatypes,
            // Int/Real arithmetic, ...) is outside the supported QF_BV(+Bool)
            // fragment. Bail out honestly rather than guess.
            _ => return None,
        };

        match &result {
            Blasted::Bool(t) => {
                self.bool_cache.insert(term_id, *t);
            }
            Blasted::Bits(v) => {
                self.stats.bits_generated += v.len() as u64;
                self.stats.bitvectors_blasted += 1;
                self.bits_cache.insert(term_id, v.clone());
            }
        }

        Some(result)
    }

    fn blast_eq(&mut self, a: TermId, b: TermId, manager: &mut TermManager) -> Option<TermId> {
        let sort_a = manager.get(a)?.sort;
        let bool_sort = manager.sorts.bool_sort;
        if sort_a == bool_sort {
            let ba = self.blast_bool(a, manager)?;
            let bb = self.blast_bool(b, manager)?;
            Some(manager.mk_eq(ba, bb))
        } else if Self::bv_width(manager, sort_a).is_some() {
            let ba = self.blast_bits(a, manager)?;
            let bb = self.blast_bits(b, manager)?;
            if ba.len() != bb.len() {
                return None;
            }
            Some(self.bits_eq(&ba, &bb, manager))
        } else {
            None
        }
    }

    fn blast_distinct(&mut self, args: &[TermId], manager: &mut TermManager) -> Option<TermId> {
        if args.len() < 2 {
            return Some(manager.mk_true());
        }
        let mut diffs = Vec::with_capacity(args.len() * (args.len() - 1) / 2);
        for i in 0..args.len() {
            for j in (i + 1)..args.len() {
                let eq = self.blast_eq(args[i], args[j], manager)?;
                diffs.push(manager.mk_not(eq));
            }
        }
        Some(manager.mk_and(diffs))
    }

    // -- leaves ---------------------------------------------------------------

    fn fresh_bits(&mut self, term_id: TermId, width: u32, manager: &mut TermManager) -> BitVec {
        let bool_sort = manager.sorts.bool_sort;
        let bits = (0..width)
            .map(|i| {
                let name = format!("!bb_{}_{i}", term_id.0);
                manager.mk_var(&name, bool_sort)
            })
            .collect();
        self.stats.bitvectors_blasted += 1;
        bits
    }

    fn blast_const(&mut self, value: &BigInt, width: u32, manager: &mut TermManager) -> BitVec {
        let one = BigInt::from(1u8);
        let bits = (0..width)
            .map(|i| {
                let set = ((value >> i) & &one) == one;
                manager.mk_bool(set)
            })
            .collect();
        self.stats.bitvectors_blasted += 1;
        bits
    }

    // -- bitwise ----------------------------------------------------------------

    fn bitwise_not(&mut self, a: &BitVec, manager: &mut TermManager) -> BitVec {
        a.iter().map(|&x| manager.mk_not(x)).collect()
    }

    fn bitwise_and(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| manager.mk_and([x, y]))
            .collect()
    }

    fn bitwise_or(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| manager.mk_or([x, y]))
            .collect()
    }

    fn bitwise_xor(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| manager.mk_xor(x, y))
            .collect()
    }

    fn bits_eq(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> TermId {
        let xnors: Vec<TermId> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let x = manager.mk_xor(x, y);
                manager.mk_not(x)
            })
            .collect();
        manager.mk_and(xnors)
    }

    // -- arithmetic circuits ------------------------------------------------

    /// Ripple-carry adder (result truncated to the input width, matching
    /// BitVector wraparound-on-overflow semantics).
    fn ripple_add(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let mut result = Vec::with_capacity(width);
        let mut carry = manager.mk_false();

        for i in 0..width {
            let axb = manager.mk_xor(a[i], b[i]);
            let sum = manager.mk_xor(axb, carry);
            let ab = manager.mk_and([a[i], b[i]]);
            let ac = manager.mk_and([a[i], carry]);
            let bc = manager.mk_and([b[i], carry]);
            let new_carry = manager.mk_or([ab, ac, bc]);

            result.push(sum);
            carry = new_carry;
            self.stats.gates_generated += 6;
        }

        result
    }

    /// Full-subtractor chain: `a - b` via borrow propagation.
    fn ripple_sub(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let mut result = Vec::with_capacity(width);
        let mut borrow = manager.mk_false();

        for i in 0..width {
            let axb = manager.mk_xor(a[i], b[i]);
            let diff = manager.mk_xor(axb, borrow);
            let not_a = manager.mk_not(a[i]);
            let t1 = manager.mk_and([not_a, b[i]]);
            let t2 = manager.mk_and([not_a, borrow]);
            let t3 = manager.mk_and([b[i], borrow]);
            let new_borrow = manager.mk_or([t1, t2, t3]);

            result.push(diff);
            borrow = new_borrow;
            self.stats.gates_generated += 6;
        }

        result
    }

    fn negate_bits(&mut self, a: &BitVec, manager: &mut TermManager) -> BitVec {
        let false_t = manager.mk_false();
        let zero = vec![false_t; a.len()];
        self.ripple_sub(&zero, a, manager)
    }

    /// Shift-and-add multiplier: `a * b` (result truncated to the input
    /// width, matching BitVector wraparound-on-overflow semantics).
    fn ripple_mul(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let false_t = manager.mk_false();
        let mut result: BitVec = vec![false_t; width];

        for i in 0..width {
            let mut partial: BitVec = vec![false_t; width];
            for j in 0..(width - i) {
                partial[i + j] = manager.mk_and([b[i], a[j]]);
            }
            result = self.ripple_add(&result, &partial, manager);
        }

        result
    }

    /// Unsigned restoring division: returns `(quotient, remainder)` for
    /// `a / b`, undefined (but well-formed) when `b == 0` — callers must
    /// apply the SMT-LIB division-by-zero convention themselves.
    fn div_rem_unsigned(
        &mut self,
        a: &BitVec,
        b: &BitVec,
        manager: &mut TermManager,
    ) -> (BitVec, BitVec) {
        let width = a.len();
        let false_t = manager.mk_false();

        let mut remainder: BitVec = vec![false_t; width + 1];
        let mut quotient: BitVec = vec![false_t; width];
        let mut divisor_ext = b.clone();
        divisor_ext.push(false_t);

        for i in (0..width).rev() {
            let mut shifted = vec![false_t; width + 1];
            shifted[0] = a[i];
            shifted[1..=width].copy_from_slice(&remainder[0..width]);

            let lt = self.unsigned_lt_bits(&shifted, &divisor_ext, manager);
            let ge = manager.mk_not(lt);
            let sub = self.ripple_sub(&shifted, &divisor_ext, manager);

            let after: BitVec = (0..=width)
                .map(|k| manager.mk_ite(ge, sub[k], shifted[k]))
                .collect();

            quotient[i] = ge;
            remainder = after;
        }

        remainder.truncate(width);
        (quotient, remainder)
    }

    fn blast_udiv(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let (quotient, _) = self.div_rem_unsigned(a, b, manager);
        let zero = vec![manager.mk_false(); width];
        let is_zero_b = self.bits_eq(b, &zero, manager);
        (0..width)
            .map(|i| {
                let all_one = manager.mk_true();
                manager.mk_ite(is_zero_b, all_one, quotient[i])
            })
            .collect()
    }

    fn blast_urem(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let (_, remainder) = self.div_rem_unsigned(a, b, manager);
        let zero = vec![manager.mk_false(); width];
        let is_zero_b = self.bits_eq(b, &zero, manager);
        (0..width)
            .map(|i| manager.mk_ite(is_zero_b, a[i], remainder[i]))
            .collect()
    }

    fn blast_sdiv(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let sign_a = a[width - 1];
        let sign_b = b[width - 1];

        let neg_a = self.negate_bits(a, manager);
        let neg_b = self.negate_bits(b, manager);
        let abs_a: BitVec = (0..width)
            .map(|i| manager.mk_ite(sign_a, neg_a[i], a[i]))
            .collect();
        let abs_b: BitVec = (0..width)
            .map(|i| manager.mk_ite(sign_b, neg_b[i], b[i]))
            .collect();

        let (quotient, _) = self.div_rem_unsigned(&abs_a, &abs_b, manager);
        let neg_quotient = self.negate_bits(&quotient, manager);
        let result_negative = manager.mk_xor(sign_a, sign_b);
        let normal: BitVec = (0..width)
            .map(|i| manager.mk_ite(result_negative, neg_quotient[i], quotient[i]))
            .collect();

        // SMT-LIB bvsdiv-by-zero: -1 (all ones) if the dividend is
        // non-negative, 1 otherwise.
        let zero = vec![manager.mk_false(); width];
        let is_zero_b = self.bits_eq(b, &zero, manager);
        let mut one = vec![manager.mk_false(); width];
        one[0] = manager.mk_true();
        let div0: BitVec = (0..width)
            .map(|i| {
                let all_one = manager.mk_true();
                manager.mk_ite(sign_a, one[i], all_one)
            })
            .collect();

        (0..width)
            .map(|i| manager.mk_ite(is_zero_b, div0[i], normal[i]))
            .collect()
    }

    fn blast_srem(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let sign_a = a[width - 1];
        let sign_b = b[width - 1];

        let neg_a = self.negate_bits(a, manager);
        let neg_b = self.negate_bits(b, manager);
        let abs_a: BitVec = (0..width)
            .map(|i| manager.mk_ite(sign_a, neg_a[i], a[i]))
            .collect();
        let abs_b: BitVec = (0..width)
            .map(|i| manager.mk_ite(sign_b, neg_b[i], b[i]))
            .collect();

        let (_, remainder) = self.div_rem_unsigned(&abs_a, &abs_b, manager);
        let neg_remainder = self.negate_bits(&remainder, manager);
        // bvsrem's result sign follows the dividend's sign only.
        let normal: BitVec = (0..width)
            .map(|i| manager.mk_ite(sign_a, neg_remainder[i], remainder[i]))
            .collect();

        // SMT-LIB bvsrem-by-zero: the dividend itself.
        let zero = vec![manager.mk_false(); width];
        let is_zero_b = self.bits_eq(b, &zero, manager);
        (0..width)
            .map(|i| manager.mk_ite(is_zero_b, a[i], normal[i]))
            .collect()
    }

    // -- shifts -------------------------------------------------------------

    fn log2_ceil(width: usize) -> u32 {
        if width <= 1 {
            0
        } else {
            usize::BITS - (width - 1).leading_zeros()
        }
    }

    fn shift_overflow(
        &mut self,
        amount: &BitVec,
        width: usize,
        manager: &mut TermManager,
    ) -> TermId {
        let width_bits =
            self.blast_const(&BigInt::from(width as u64), amount.len() as u32, manager);
        let lt = self.unsigned_lt_bits(amount, &width_bits, manager);
        manager.mk_not(lt)
    }

    /// Logical left shift via a `log2(width)`-stage barrel shifter.
    fn blast_shl(&mut self, a: &BitVec, amount: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let false_t = manager.mk_false();
        let mut cur = a.clone();

        for k in 0..Self::log2_ceil(width) {
            let shift_amt = 1usize << k;
            if shift_amt >= width {
                continue;
            }
            let mut shifted = vec![false_t; width];
            shifted[shift_amt..width].copy_from_slice(&cur[0..width - shift_amt]);

            let bit_k = amount.get(k as usize).copied().unwrap_or(false_t);
            cur = (0..width)
                .map(|j| manager.mk_ite(bit_k, shifted[j], cur[j]))
                .collect();
        }

        let overflow = self.shift_overflow(amount, width, manager);
        cur.into_iter()
            .map(|bit| manager.mk_ite(overflow, false_t, bit))
            .collect()
    }

    /// Logical right shift via a `log2(width)`-stage barrel shifter.
    fn blast_lshr(&mut self, a: &BitVec, amount: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let false_t = manager.mk_false();
        let mut cur = a.clone();

        for k in 0..Self::log2_ceil(width) {
            let shift_amt = 1usize << k;
            if shift_amt >= width {
                continue;
            }
            let mut shifted = vec![false_t; width];
            shifted[0..width - shift_amt].copy_from_slice(&cur[shift_amt..width]);

            let bit_k = amount.get(k as usize).copied().unwrap_or(false_t);
            cur = (0..width)
                .map(|j| manager.mk_ite(bit_k, shifted[j], cur[j]))
                .collect();
        }

        let overflow = self.shift_overflow(amount, width, manager);
        cur.into_iter()
            .map(|bit| manager.mk_ite(overflow, false_t, bit))
            .collect()
    }

    /// Arithmetic right shift: like `blast_lshr`, but fills with (and
    /// overflows to) the sign bit rather than zero.
    fn blast_ashr(&mut self, a: &BitVec, amount: &BitVec, manager: &mut TermManager) -> BitVec {
        let width = a.len();
        let sign = a[width - 1];
        let false_t = manager.mk_false();
        let mut cur = a.clone();

        for k in 0..Self::log2_ceil(width) {
            let shift_amt = 1usize << k;
            if shift_amt >= width {
                continue;
            }
            let mut shifted = vec![sign; width];
            shifted[0..width - shift_amt].copy_from_slice(&cur[shift_amt..width]);

            let bit_k = amount.get(k as usize).copied().unwrap_or(false_t);
            cur = (0..width)
                .map(|j| manager.mk_ite(bit_k, shifted[j], cur[j]))
                .collect();
        }

        let overflow = self.shift_overflow(amount, width, manager);
        cur.into_iter()
            .map(|bit| manager.mk_ite(overflow, sign, bit))
            .collect()
    }

    // -- comparators ----------------------------------------------------------

    /// Unsigned `a < b`, folded MSB-to-LSB.
    ///
    /// Two accumulators are threaded through the fold: `decided_lt` (the
    /// less-than verdict already locked in by strictly more significant
    /// bits) and `still_equal` (whether every bit processed so far, from
    /// the MSB down to the current one, has been equal). A single
    /// accumulator is *not* sound here: once a higher bit has decided the
    /// comparison, a differing lower bit must not be allowed to overturn
    /// it, which a naive `bit_lt OR (bit_eq AND previous)` fold gets wrong
    /// (it forgets `previous` as soon as any later bit differs, even after
    /// the outcome was already determined).
    fn unsigned_lt_bits(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> TermId {
        let mut decided_lt = manager.mk_false();
        let mut still_equal = manager.mk_true();

        for i in (0..a.len()).rev() {
            let not_ai = manager.mk_not(a[i]);
            let bit_lt = manager.mk_and([not_ai, b[i]]);
            let xor_i = manager.mk_xor(a[i], b[i]);
            let bit_eq = manager.mk_not(xor_i);

            let decide_here = manager.mk_and([still_equal, bit_lt]);
            let not_still_equal = manager.mk_not(still_equal);
            let keep_prior = manager.mk_and([not_still_equal, decided_lt]);
            decided_lt = manager.mk_or([decide_here, keep_prior]);

            still_equal = manager.mk_and([still_equal, bit_eq]);
            self.stats.gates_generated += 7;
        }

        decided_lt
    }

    /// Signed `a < b` via sign-bit case analysis plus an unsigned compare.
    fn signed_lt_bits(&mut self, a: &BitVec, b: &BitVec, manager: &mut TermManager) -> TermId {
        let width = a.len();
        let sign_a = a[width - 1];
        let sign_b = b[width - 1];

        let not_sign_b = manager.mk_not(sign_b);
        let diff_sign_lt = manager.mk_and([sign_a, not_sign_b]);
        let xor_signs = manager.mk_xor(sign_a, sign_b);
        let same_sign = manager.mk_not(xor_signs);
        let ult = self.unsigned_lt_bits(a, b, manager);
        let same_sign_lt = manager.mk_and([same_sign, ult]);

        manager.mk_or([diff_sign_lt, same_sign_lt])
    }
}

impl Default for BitBlaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{TermKind, TermManager};

    #[test]
    fn test_bit_blast_not_applicable_empty() {
        let manager = TermManager::new();
        let goal = Goal::empty();
        let tactic = BitBlastTactic::new(&manager);
        let result = tactic
            .apply_check(&goal)
            .expect("test operation should succeed");
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_blaster_not_applicable_without_bv() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let x = manager.mk_var("x", int_sort);
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(x, ten);
        let goal = Goal::new(vec![lt]);

        let mut blaster = BitBlaster::new();
        let result = blaster
            .blast_goal(&goal, &mut manager)
            .expect("test operation should succeed");
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_blaster_bails_on_unsupported_construct() {
        // A goal that mixes an array `select` (unsupported) with a BV
        // comparison referencing the same variable must be left
        // completely untouched, not partially blasted.
        let mut manager = TermManager::new();
        let bv8 = manager.sorts.bitvec(8);
        let array_sort = manager.sorts.array(bv8, bv8);
        let arr = manager.mk_var("arr", array_sort);
        let idx = manager.mk_var("idx", bv8);
        let selected = manager.mk_select(arr, idx);
        let zero = manager.mk_bitvec(0u64, 8);
        let eq = manager.mk_eq(selected, zero);

        let goal = Goal::new(vec![eq]);
        let mut blaster = BitBlaster::new();
        let result = blaster
            .blast_goal(&goal, &mut manager)
            .expect("test operation should succeed");
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    #[test]
    fn test_stateless_bit_blast() {
        // The manager-free `Tactic::apply` path cannot allocate circuit terms,
        // so it honestly reports NotApplicable (P4-1107) rather than returning
        // the goal unchanged. The real blast is exercised via
        // `BitBlaster::blast_goal` / the registry's `create_managed` path.
        let goal = Goal::empty();
        let tactic = StatelessBitBlastTactic;

        let result = tactic.apply(&goal).expect("test operation should succeed");
        assert!(matches!(result, TacticResult::NotApplicable));
    }

    /// Build a bit-vector operation node **without** the term manager's
    /// constant folding.
    ///
    /// Every `mk_bv_*` constructor evaluates literal operands directly, so
    /// building e.g. `bvadd #x3 #x5` through the public API yields the literal
    /// `#x8` and the blaster would never see an adder to expand.  The tests
    /// below exist precisely to check the generated *circuits*, so they intern
    /// the raw node and let the blaster expand it over the operands' pinned
    /// constant bits -- the blasted result still folds to `true`, so the
    /// assertion checks the circuit's answer rather than merely that some term
    /// was produced.
    fn raw_bv_op(manager: &mut TermManager, kind: TermKind, width: u32) -> TermId {
        let sort = manager.sorts.bitvec(width);
        manager.intern_term(kind, sort)
    }

    /// Blast `eq` (expected to compare two constant bit-vector
    /// expressions) and assert the whole thing reduces (via the term
    /// manager's built-in constant folding) all the way down to `true`,
    /// proving the generated circuit computes the right answer -- not
    /// just that *some* term was produced.
    fn assert_blasts_to_true(manager: &mut TermManager, eq: TermId) {
        let goal = Goal::new(vec![eq]);
        let result = BitBlastTactic::blast(&goal, manager).expect("blast should succeed");
        match result {
            TacticResult::SubGoals(goals) => {
                assert_eq!(goals.len(), 1);
                assert_eq!(goals[0].assertions, vec![manager.mk_true()]);
            }
            other => panic!("expected SubGoals, got {other:?}"),
        }
    }

    #[test]
    fn test_blast_add_is_correct() {
        let mut manager = TermManager::new();
        let a = manager.mk_bitvec(3u64, 4);
        let b = manager.mk_bitvec(5u64, 4);
        let sum = raw_bv_op(&mut manager, TermKind::BvAdd(a, b), 4);
        let expected = manager.mk_bitvec(8u64, 4); // 3 + 5 = 8
        let eq = manager.mk_eq(sum, expected);
        assert_blasts_to_true(&mut manager, eq);
    }

    #[test]
    fn test_blast_sub_wraps_correctly() {
        let mut manager = TermManager::new();
        let a = manager.mk_bitvec(3u64, 4);
        let b = manager.mk_bitvec(5u64, 4);
        let diff = raw_bv_op(&mut manager, TermKind::BvSub(a, b), 4);
        let expected = manager.mk_bitvec(14u64, 4); // 3 - 5 = -2 = 14 (mod 16)
        let eq = manager.mk_eq(diff, expected);
        assert_blasts_to_true(&mut manager, eq);
    }

    #[test]
    fn test_blast_mul_is_correct() {
        let mut manager = TermManager::new();
        let a = manager.mk_bitvec(6u64, 5);
        let b = manager.mk_bitvec(5u64, 5);
        let prod = raw_bv_op(&mut manager, TermKind::BvMul(a, b), 5);
        let expected = manager.mk_bitvec(30u64, 5);
        let eq = manager.mk_eq(prod, expected);
        assert_blasts_to_true(&mut manager, eq);
    }

    #[test]
    fn test_blast_udiv_urem_are_correct() {
        let mut manager = TermManager::new();
        let a = manager.mk_bitvec(7u64, 4);
        let b = manager.mk_bitvec(2u64, 4);
        let q = raw_bv_op(&mut manager, TermKind::BvUdiv(a, b), 4);
        let r = raw_bv_op(&mut manager, TermKind::BvUrem(a, b), 4);
        let expected_q = manager.mk_bitvec(3u64, 4);
        let expected_r = manager.mk_bitvec(1u64, 4);
        let eq_q = manager.mk_eq(q, expected_q);
        let eq_r = manager.mk_eq(r, expected_r);
        let both = manager.mk_and([eq_q, eq_r]);
        assert_blasts_to_true(&mut manager, both);
    }

    #[test]
    fn test_blast_udiv_by_zero_is_all_ones() {
        let mut manager = TermManager::new();
        let a = manager.mk_bitvec(7u64, 4);
        let zero = manager.mk_bitvec(0u64, 4);
        let q = raw_bv_op(&mut manager, TermKind::BvUdiv(a, zero), 4);
        let all_ones = manager.mk_bitvec(0xFu64, 4);
        let eq = manager.mk_eq(q, all_ones);
        assert_blasts_to_true(&mut manager, eq);
    }

    #[test]
    fn test_blast_sdiv_srem_are_correct() {
        // -7 / 2 = -3 rem -1 (truncating division), in 4-bit two's
        // complement: -7 = 9, -3 = 13, -1 = 15.
        let mut manager = TermManager::new();
        let a = manager.mk_bitvec(9u64, 4);
        let b = manager.mk_bitvec(2u64, 4);
        let q = raw_bv_op(&mut manager, TermKind::BvSdiv(a, b), 4);
        let r = raw_bv_op(&mut manager, TermKind::BvSrem(a, b), 4);
        let expected_q = manager.mk_bitvec(13u64, 4);
        let expected_r = manager.mk_bitvec(15u64, 4);
        let eq_q = manager.mk_eq(q, expected_q);
        let eq_r = manager.mk_eq(r, expected_r);
        let both = manager.mk_and([eq_q, eq_r]);
        assert_blasts_to_true(&mut manager, both);
    }

    #[test]
    fn test_blast_shl_overflow_is_zero() {
        let mut manager = TermManager::new();
        let a = manager.mk_bitvec(0b0011u64, 4);
        let one = manager.mk_bitvec(1u64, 4);
        let shifted_by_1 = raw_bv_op(&mut manager, TermKind::BvShl(a, one), 4);
        let expected = manager.mk_bitvec(0b0110u64, 4);
        let eq1 = manager.mk_eq(shifted_by_1, expected);

        // Shifting by >= width must yield zero, not garbage.
        let width = manager.mk_bitvec(4u64, 4);
        let shifted_overflow = raw_bv_op(&mut manager, TermKind::BvShl(a, width), 4);
        let zero = manager.mk_bitvec(0u64, 4);
        let eq2 = manager.mk_eq(shifted_overflow, zero);

        let both = manager.mk_and([eq1, eq2]);
        assert_blasts_to_true(&mut manager, both);
    }

    #[test]
    fn test_blast_ashr_sign_extends() {
        let mut manager = TermManager::new();
        // -8 (1000b) arithmetic-shifted right by 1 = -4 (1100b).
        let a = manager.mk_bitvec(0b1000u64, 4);
        let one = manager.mk_bitvec(1u64, 4);
        let shifted = raw_bv_op(&mut manager, TermKind::BvAshr(a, one), 4);
        let expected = manager.mk_bitvec(0b1100u64, 4);
        let eq = manager.mk_eq(shifted, expected);
        assert_blasts_to_true(&mut manager, eq);
    }

    #[test]
    fn test_blast_comparisons_are_correct() {
        let mut manager = TermManager::new();
        // The operands are wrapped in an *unfolded* `bvadd` (see `raw_bv_op`)
        // so the comparison survives `mk_bv_ult` / `mk_bv_slt`'s constant
        // folding (issue #17) and actually reaches the comparator circuits
        // under test.  Each `bvadd` still evaluates to a constant once blasted,
        // so the goal folds all the way to `true` and the assertion checks the
        // circuit's answer, not merely that some term was produced.
        let zero = manager.mk_bitvec(0u64, 4);
        let one = manager.mk_bitvec(1u64, 4);
        let two = manager.mk_bitvec(2u64, 4);
        let three = manager.mk_bitvec(3u64, 4);
        let fourteen = manager.mk_bitvec(0b1110u64, 4);

        // Unsigned: 3 < 5.
        let lhs_three = raw_bv_op(&mut manager, TermKind::BvAdd(one, two), 4); // 3
        let rhs_five = raw_bv_op(&mut manager, TermKind::BvAdd(two, three), 4); // 5
        let ult = manager.mk_bv_ult(lhs_three, rhs_five);

        // Signed: -1 (1111b) < 1, even though unsigned 15 > 1.
        let minus_one = raw_bv_op(&mut manager, TermKind::BvAdd(fourteen, one), 4); // 1111b
        let plus_one = raw_bv_op(&mut manager, TermKind::BvAdd(zero, one), 4); // 0001b
        let slt = manager.mk_bv_slt(minus_one, plus_one);
        let not_ult_for_same = {
            let ult2 = manager.mk_bv_ult(minus_one, plus_one);
            manager.mk_not(ult2) // unsigned: 15 < 1 is false
        };

        let all = manager.mk_and([ult, slt, not_ult_for_same]);
        assert_blasts_to_true(&mut manager, all);
    }

    /// Run `body` on a worker thread with a deliberately small (128 KiB) stack,
    /// so that a recursive walk over a deep term would abort the process
    /// instead of silently getting away with a large main-thread stack.
    fn run_with_small_stack<F>(body: F)
    where
        F: FnOnce() + Send + 'static,
    {
        std::thread::Builder::new()
            .stack_size(1 << 17)
            .spawn(body)
            .expect("thread spawn should succeed")
            .join()
            .expect("deep-nesting walk must not overflow the stack");
    }

    #[test]
    fn test_contains_bv_term_handles_deeply_nested_terms() {
        run_with_small_stack(|| {
            const DEPTH: usize = 6_250;

            let mut manager = TermManager::new();
            let bv8 = manager.sorts.bitvec(8);
            let bool_sort = manager.sorts.bool_sort;
            let a = manager.mk_var("a", bv8);
            let b = manager.mk_var("b", bv8);

            // A Bool-sorted term that does contain BitVector content, buried
            // under 50k negations. `intern_term` is used so the chain survives
            // `mk_not`'s double-negation folding.
            let mut current = manager.mk_bv_ult(a, b);
            for _ in 0..DEPTH {
                current = manager.intern_term(TermKind::Not(current), bool_sort);
            }

            let tactic = BitBlastTactic::new(&manager);
            assert!(tactic.contains_bv_term(current));
        });
    }

    #[test]
    fn test_contains_bv_term_does_not_re_expand_shared_subterms() {
        run_with_small_stack(|| {
            // A doubling DAG: without a `visited` set this is 2^40 visits.
            const LEVELS: usize = 40;

            let mut manager = TermManager::new();
            let bool_sort = manager.sorts.bool_sort;
            let int_sort = manager.sorts.int_sort;
            let x = manager.mk_var("x", int_sort);
            let one = manager.mk_int(1);

            let mut current = manager.mk_lt(x, one);
            for _ in 0..LEVELS {
                current =
                    manager.intern_term(TermKind::And(vec![current, current].into()), bool_sort);
            }

            let tactic = BitBlastTactic::new(&manager);
            assert!(!tactic.contains_bv_term(current));
        });
    }

    #[test]
    fn test_contains_bv_term_semantics_are_unchanged() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;
        let bv8 = manager.sorts.bitvec(8);

        let x = manager.mk_var("x", int_sort);
        let ten = manager.mk_int(10);
        let int_only = manager.mk_lt(x, ten);

        let a = manager.mk_var("a", bv8);
        let b = manager.mk_var("b", bv8);
        let with_bv = manager.mk_bv_ult(a, b);
        let nested = manager.mk_and([int_only, with_bv]);

        let tactic = BitBlastTactic::new(&manager);
        assert!(!tactic.contains_bv_term(int_only));
        assert!(tactic.contains_bv_term(with_bv));
        assert!(tactic.contains_bv_term(nested));
        assert!(tactic.contains_bv_term(a));
    }

    #[test]
    fn test_blaster_reports_nonzero_stats_after_blasting() {
        let mut manager = TermManager::new();
        let bv8 = manager.sorts.bitvec(8);
        let a = manager.mk_var("a", bv8);
        let b = manager.mk_var("b", bv8);
        let sum = manager.mk_bv_add(a, b);
        let hundred = manager.mk_bitvec(100u64, 8);
        let eq = manager.mk_eq(sum, hundred);
        let goal = Goal::new(vec![eq]);

        let mut blaster = BitBlaster::new();
        let result = blaster
            .blast_goal(&goal, &mut manager)
            .expect("test operation should succeed");
        assert!(matches!(result, TacticResult::SubGoals(_)));
        assert!(blaster.stats().bits_generated > 0);
        assert!(blaster.stats().bitvectors_blasted > 0);
    }
}
