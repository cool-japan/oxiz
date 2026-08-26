//! FloatingPoint theory implementation
//!
//! Implements a lightweight theory of IEEE 754 floating-point terms: it tracks
//! the registered floating-point terms, recognises the literals and the
//! special values among them, instantiates the axioms it can state as terms —
//! with the guards that make them true, since most of the "obvious" identities
//! fail for NaN or for negative zero — keeps equality classes, and reports a
//! conflict when two different literal values are forced into one class.
//!
//! It does not evaluate rounded arithmetic: only the exact, rounding-free
//! operations (`fp.neg`, `fp.abs`) are folded. This is `oxiz-core`'s
//! self-contained layer, not the solver's floating-point theory — that is
//! `oxiz_theories::fp` — and finding no conflict here says nothing about
//! satisfiability.
//!
//! Reference: Z3's `src/smt/theory_fpa.cpp`

use super::combination::{Theory, TheoryResult};
use super::eq_classes::EqClasses;
use crate::ast::traversal::get_children;
use crate::ast::{RoundingMode, TermId, TermKind, TermManager, bv_wrap_unsigned};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::{SortId, SortKind, SortManager};
use num_bigint::BigInt;

/// FloatingPoint theory axioms and properties
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FloatingPointAxiom {
    /// Addition identity: fp.add(rm, x, +0) = x
    AddIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The rounding mode used in the operation
        rounding_mode: RoundingMode,
    },
    /// Multiplication identity: fp.mul(rm, x, 1.0) = x
    MulIdentity {
        /// The term this axiom applies to
        term: TermId,
        /// The rounding mode used in the operation
        rounding_mode: RoundingMode,
    },
    /// Multiplication by zero: fp.mul(rm, x, +0) = +0 (for normal x)
    MulZero {
        /// The term this axiom applies to
        term: TermId,
        /// The rounding mode used in the operation
        rounding_mode: RoundingMode,
    },
    /// Negation involution: fp.neg(fp.neg(x)) = x
    NegInvolution {
        /// The term this axiom applies to
        term: TermId,
    },
    /// Absolute value: fp.abs(x) >= +0
    AbsNonNegative {
        /// The term this axiom applies to
        term: TermId,
    },
    /// NaN propagation: any operation with NaN produces NaN
    NaNPropagation {
        /// The term this axiom applies to
        term: TermId,
        /// The floating-point operation that propagates NaN
        operation: FloatingPointOp,
    },
    /// Infinity properties
    InfinityAxiom {
        /// The term this axiom applies to
        term: TermId,
        /// The specific infinity property being asserted
        property: InfinityProperty,
    },
    /// Comparison with NaN always false (except fp.isNaN)
    CompareNaN {
        /// The left-hand side of the comparison
        lhs: TermId,
        /// The right-hand side of the comparison
        rhs: TermId,
    },
}

/// FloatingPoint operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FloatingPointOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Remainder
    Rem,
    /// Fused multiply-add
    Fma,
    /// Square root
    Sqrt,
    /// Round to integral
    RoundToIntegral,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Absolute value
    Abs,
    /// Negation
    Neg,
}

/// Infinity properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InfinityProperty {
    /// +∞ + +∞ = +∞
    AddPosInf,
    /// +∞ * x = +∞ (for x > 0)
    MulPosInf,
    /// +∞ > x (for finite x)
    CompareGreater,
}

/// FloatingPoint theory reasoning engine
#[derive(Debug, Clone)]
pub struct FloatingPointTheory {
    /// Tracked floating-point terms (maps term to its sort)
    floats: FxHashMap<TermId, SortId>,
    /// Special values (NaN, +Inf, -Inf, +0, -0)
    special_values: FxHashMap<SortId, SpecialValues>,
    /// Equality classes over the registered floating-point terms
    classes: EqClasses,
    /// Equalities already reported by `propagate`, so each is reported once
    reported: FxHashSet<(TermId, TermId)>,
    /// Pending axiom instantiations
    pending_axioms: Vec<FloatingPointAxiom>,
    /// Already instantiated axioms (to avoid duplicates)
    instantiated: FxHashSet<FloatingPointAxiom>,
    /// Statistics
    propagations: usize,
    conflicts: usize,
}

/// The value denoted by a floating-point literal
///
/// Two literals denote the same value exactly when this is equal, which is why
/// every comparison goes through it: `(fp #b1 #b00000000 #b0…0)` and
/// `(_ -zero 8 24)` are two spellings of the same number, and SMT-LIB has a
/// single NaN, so all NaN spellings are one value too.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FloatValue {
    /// The one NaN value
    NaN,
    /// Positive infinity
    PosInfinity,
    /// Negative infinity
    NegInfinity,
    /// Positive zero
    PosZero,
    /// Negative zero
    NegZero,
    /// A finite non-zero value, as sign / biased exponent / trailing significand
    Finite {
        /// Sign bit
        sign: bool,
        /// Biased exponent field
        exponent: BigInt,
        /// Trailing significand field
        significand: BigInt,
    },
}

/// The special values of one floating-point sort that have been registered
///
/// Each field holds the term that introduced that value, once one has been
/// seen; `None` means no term of this sort denoting it has been registered.
/// Several spellings of one value collapse here, so this is a per-sort index
/// of "has the value shown up, and under which term".
#[derive(Debug, Clone, Default)]
struct SpecialValues {
    /// Positive zero
    pos_zero: Option<TermId>,
    /// Negative zero
    neg_zero: Option<TermId>,
    /// Positive infinity
    pos_inf: Option<TermId>,
    /// Negative infinity
    neg_inf: Option<TermId>,
    /// Not-a-Number
    nan: Option<TermId>,
}

/// Which special value a term denotes, if any
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialValue {
    /// Positive zero
    PosZero,
    /// Negative zero
    NegZero,
    /// Positive infinity
    PosInfinity,
    /// Negative infinity
    NegInfinity,
    /// Not-a-Number
    NaN,
}

impl Default for FloatingPointTheory {
    fn default() -> Self {
        Self::new()
    }
}

impl FloatingPointTheory {
    /// Create a new floating-point theory instance
    #[must_use]
    pub fn new() -> Self {
        Self {
            floats: FxHashMap::default(),
            special_values: FxHashMap::default(),
            classes: EqClasses::new(),
            reported: FxHashSet::default(),
            pending_axioms: Vec::new(),
            instantiated: FxHashSet::default(),
            propagations: 0,
            conflicts: 0,
        }
    }

    /// Register a floating-point term
    pub fn register_float(&mut self, term: TermId, sort: SortId) {
        self.floats.insert(term, sort);
        self.classes.add(term);
    }

    /// Add a term to the theory and extract floating-point operations
    ///
    /// Returns `true` when the term has a floating-point sort, i.e. when this
    /// theory has taken it on. Predicates over floating-point terms
    /// (`fp.isNaN`, `fp.lt`, …) are Boolean and are not claimed here.
    pub fn add_term(
        &mut self,
        term: TermId,
        manager: &TermManager,
        sort_manager: &SortManager,
    ) -> bool {
        if let Some(t) = manager.get(term)
            && let Some(sort) = sort_manager.get(t.sort)
            && matches!(sort.kind, SortKind::FloatingPoint { .. })
        {
            let sort_id = t.sort;
            self.register_float(term, sort_id);
            self.record_special_value(term, sort_id, manager);

            // Generate axioms based on term structure
            let kind = t.kind.clone();
            self.generate_axioms_for_term(term, &kind, manager);
            return true;
        }
        false
    }

    /// Note a term that denotes one of its sort's special values
    ///
    /// The first term seen for a value is the one kept, so the index points at
    /// a term that is actually registered.
    fn record_special_value(&mut self, term: TermId, sort: SortId, manager: &TermManager) {
        let Some((_, value)) = Self::literal_value(term, manager) else {
            return;
        };

        let entry = self.special_values.entry(sort).or_default();
        let slot = match value {
            FloatValue::PosZero => &mut entry.pos_zero,
            FloatValue::NegZero => &mut entry.neg_zero,
            FloatValue::PosInfinity => &mut entry.pos_inf,
            FloatValue::NegInfinity => &mut entry.neg_inf,
            FloatValue::NaN => &mut entry.nan,
            FloatValue::Finite { .. } => return,
        };
        slot.get_or_insert(term);
    }

    /// The registered term denoting a special value of a sort, if one was seen
    ///
    /// Returns the first term of that sort registered for that value. `None`
    /// means no registered term of this sort denotes it — never that no such
    /// value exists.
    #[must_use]
    pub fn special_value_term(&self, sort: SortId, value: SpecialValue) -> Option<TermId> {
        let entry = self.special_values.get(&sort)?;
        match value {
            SpecialValue::PosZero => entry.pos_zero,
            SpecialValue::NegZero => entry.neg_zero,
            SpecialValue::PosInfinity => entry.pos_inf,
            SpecialValue::NegInfinity => entry.neg_inf,
            SpecialValue::NaN => entry.nan,
        }
    }

    /// Record that two registered floating-point terms of the same sort are equal
    ///
    /// Returns `true` when this was new information. Equalities about unknown
    /// terms, and equalities between terms of different floating-point sorts
    /// (which are not well sorted), are ignored.
    pub fn assert_equality(&mut self, a: TermId, b: TermId) -> bool {
        let (Some(&sort_a), Some(&sort_b)) = (self.floats.get(&a), self.floats.get(&b)) else {
            return false;
        };
        if sort_a != sort_b {
            return false;
        }

        self.classes.union(a, b)
    }

    /// Whether two terms are currently known to be equal by this theory
    ///
    /// Takes `&mut self` because the lookup compresses the union-find paths.
    pub fn known_equal(&mut self, a: TermId, b: TermId) -> bool {
        self.classes.are_equal(a, b)
    }

    /// Generate axioms for a floating-point term
    fn generate_axioms_for_term(&mut self, term: TermId, kind: &TermKind, manager: &TermManager) {
        match kind {
            TermKind::FpAdd(rm, lhs, rhs) => {
                // Check for addition with zero
                if self.is_fp_pos_zero(*rhs, manager) || self.is_fp_pos_zero(*lhs, manager) {
                    self.add_axiom(FloatingPointAxiom::AddIdentity {
                        term,
                        rounding_mode: *rm,
                    });
                }
                // Check for NaN
                if self.is_fp_nan(*lhs, manager) || self.is_fp_nan(*rhs, manager) {
                    self.add_axiom(FloatingPointAxiom::NaNPropagation {
                        term,
                        operation: FloatingPointOp::Add,
                    });
                }
            }
            TermKind::FpMul(rm, lhs, rhs) => {
                // Check for multiplication by zero or one
                if self.is_fp_pos_zero(*rhs, manager) || self.is_fp_pos_zero(*lhs, manager) {
                    self.add_axiom(FloatingPointAxiom::MulZero {
                        term,
                        rounding_mode: *rm,
                    });
                } else if self.is_fp_one(*rhs, manager) || self.is_fp_one(*lhs, manager) {
                    self.add_axiom(FloatingPointAxiom::MulIdentity {
                        term,
                        rounding_mode: *rm,
                    });
                }
                // Check for NaN
                if self.is_fp_nan(*lhs, manager) || self.is_fp_nan(*rhs, manager) {
                    self.add_axiom(FloatingPointAxiom::NaNPropagation {
                        term,
                        operation: FloatingPointOp::Mul,
                    });
                }
            }
            TermKind::FpNeg(inner) => {
                // Check for double negation
                if let Some(inner_term) = manager.get(*inner)
                    && let TermKind::FpNeg(_) = inner_term.kind
                {
                    self.add_axiom(FloatingPointAxiom::NegInvolution { term });
                }
            }
            TermKind::FpAbs(_) => {
                self.add_axiom(FloatingPointAxiom::AbsNonNegative { term });
            }
            _ => {}
        }
    }

    /// The value denoted by a floating-point literal, with its format
    ///
    /// Returns `None` for terms that are not literals. The literal's fields
    /// are normalised into their bit-widths first, and an all-ones exponent is
    /// resolved into an infinity or a NaN, so that every spelling of a value
    /// maps to the same [`FloatValue`].
    fn literal_value(term: TermId, manager: &TermManager) -> Option<((u32, u32), FloatValue)> {
        match &manager.get(term)?.kind {
            TermKind::FpNaN { eb, sb } => Some(((*eb, *sb), FloatValue::NaN)),
            TermKind::FpPlusInfinity { eb, sb } => Some(((*eb, *sb), FloatValue::PosInfinity)),
            TermKind::FpMinusInfinity { eb, sb } => Some(((*eb, *sb), FloatValue::NegInfinity)),
            TermKind::FpPlusZero { eb, sb } => Some(((*eb, *sb), FloatValue::PosZero)),
            TermKind::FpMinusZero { eb, sb } => Some(((*eb, *sb), FloatValue::NegZero)),
            TermKind::FpLit {
                sign,
                exp,
                sig,
                eb,
                sb,
            } => {
                let exponent = bv_wrap_unsigned(exp, *eb);
                let significand = bv_wrap_unsigned(sig, sb.saturating_sub(1));
                let all_ones = crate::ast::bv_fold::all_ones(*eb);

                let value = if exponent == all_ones {
                    if significand == BigInt::ZERO {
                        if *sign {
                            FloatValue::NegInfinity
                        } else {
                            FloatValue::PosInfinity
                        }
                    } else {
                        FloatValue::NaN
                    }
                } else if exponent == BigInt::ZERO && significand == BigInt::ZERO {
                    if *sign {
                        FloatValue::NegZero
                    } else {
                        FloatValue::PosZero
                    }
                } else {
                    FloatValue::Finite {
                        sign: *sign,
                        exponent,
                        significand,
                    }
                };

                Some(((*eb, *sb), value))
            }
            _ => None,
        }
    }

    /// Check if a term is positive zero
    fn is_fp_pos_zero(&self, term: TermId, manager: &TermManager) -> bool {
        matches!(
            Self::literal_value(term, manager),
            Some((_, FloatValue::PosZero))
        )
    }

    /// Check if a term is NaN
    fn is_fp_nan(&self, term: TermId, manager: &TermManager) -> bool {
        matches!(
            Self::literal_value(term, manager),
            Some((_, FloatValue::NaN))
        )
    }

    /// Check if a term is the literal `1.0`
    ///
    /// `1.0` is the value whose exponent field is the bias `2^(eb-1) - 1` and
    /// whose trailing significand is zero.
    fn is_fp_one(&self, term: TermId, manager: &TermManager) -> bool {
        let Some(((eb, _), value)) = Self::literal_value(term, manager) else {
            return false;
        };
        let FloatValue::Finite {
            sign,
            exponent,
            significand,
        } = value
        else {
            return false;
        };

        !sign && significand == BigInt::ZERO && exponent == bias(eb)
    }

    /// Add an axiom to the pending list
    fn add_axiom(&mut self, axiom: FloatingPointAxiom) {
        if self.instantiated.insert(axiom.clone()) {
            self.pending_axioms.push(axiom);
        }
    }

    /// Get pending axioms and clear the list
    pub fn take_pending_axioms(&mut self) -> Vec<FloatingPointAxiom> {
        core::mem::take(&mut self.pending_axioms)
    }

    /// Build the formula stated by an axiom
    ///
    /// The identities of floating-point arithmetic mostly need a side
    /// condition to be true, and the terms built here carry it:
    ///
    /// * `AddIdentity` — `not (fp.isZero x) ⟹ fp.add(rm, x, +0) = x`. The
    ///   guard is not decoration: `fp.add(RNE, -0, +0)` is `+0`, not `-0`.
    /// * `MulIdentity` — `fp.mul(rm, x, 1.0) = x`, which needs no guard.
    /// * `MulZero` — `fp.isPositive x ∧ not (fp.isInfinite x) ⟹
    ///   fp.mul(rm, x, +0) = +0`. Without the guard it is false for negative
    ///   `x` (which gives `-0`), for infinities and for NaN.
    /// * `NegInvolution` — `fp.neg(fp.neg(x)) = x`.
    /// * `AbsNonNegative` — `not (fp.isNaN x) ⟹ fp.geq(fp.abs x, +0)`;
    ///   every comparison against NaN is false, including this one.
    /// * `NaNPropagation` — `term = NaN`.
    /// * `InfinityAxiom` — the guarded form of the named property.
    /// * `CompareNaN` — the conjunction stating that all five ordering
    ///   predicates are false for the pair.
    ///
    /// Returns `None` when the axiom does not fit its term: the term is not in
    /// `manager`, does not have the operator the axiom claims, has no
    /// floating-point sort, or its operands are not the special values the
    /// axiom names.
    pub fn axiom_to_term(
        &self,
        axiom: &FloatingPointAxiom,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        match axiom {
            FloatingPointAxiom::AddIdentity {
                term,
                rounding_mode,
            } => {
                let TermKind::FpAdd(rm, lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                if rm != *rounding_mode {
                    return None;
                }
                let other = self.other_operand(lhs, rhs, manager, Self::is_fp_pos_zero)?;

                let equation = manager.mk_eq(*term, other);
                let is_zero = manager.mk_fp_is_zero(other);
                let guard = manager.mk_not(is_zero);
                Some(manager.mk_implies(guard, equation))
            }
            FloatingPointAxiom::MulIdentity {
                term,
                rounding_mode,
            } => {
                let TermKind::FpMul(rm, lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                if rm != *rounding_mode {
                    return None;
                }
                let other = self.other_operand(lhs, rhs, manager, Self::is_fp_one)?;
                Some(manager.mk_eq(*term, other))
            }
            FloatingPointAxiom::MulZero {
                term,
                rounding_mode,
            } => {
                let TermKind::FpMul(rm, lhs, rhs) = manager.get(*term)?.kind else {
                    return None;
                };
                if rm != *rounding_mode {
                    return None;
                }
                let other = self.other_operand(lhs, rhs, manager, Self::is_fp_pos_zero)?;
                let (eb, sb) = self.format_of(*term, manager)?;

                let zero = manager.mk_fp_plus_zero(eb, sb);
                let equation = manager.mk_eq(*term, zero);
                let positive = manager.mk_fp_is_positive(other);
                let infinite = manager.mk_fp_is_infinite(other);
                let finite = manager.mk_not(infinite);
                let guard = manager.mk_and([positive, finite]);
                Some(manager.mk_implies(guard, equation))
            }
            FloatingPointAxiom::NegInvolution { term } => {
                let TermKind::FpNeg(inner) = manager.get(*term)?.kind else {
                    return None;
                };
                let TermKind::FpNeg(innermost) = manager.get(inner)?.kind else {
                    return None;
                };
                Some(manager.mk_eq(*term, innermost))
            }
            FloatingPointAxiom::AbsNonNegative { term } => {
                let TermKind::FpAbs(inner) = manager.get(*term)?.kind else {
                    return None;
                };
                let (eb, sb) = self.format_of(*term, manager)?;

                let zero = manager.mk_fp_plus_zero(eb, sb);
                let comparison = manager.mk_fp_geq(*term, zero);
                let is_nan = manager.mk_fp_is_nan(inner);
                let guard = manager.mk_not(is_nan);
                Some(manager.mk_implies(guard, comparison))
            }
            FloatingPointAxiom::NaNPropagation { term, .. } => {
                // The result is NaN because an operand is; without such an
                // operand there is nothing to state.
                let kind = manager.get(*term)?.kind.clone();
                let operands = get_children(&kind);
                if operands.is_empty()
                    || !operands
                        .iter()
                        .any(|&operand| self.is_fp_nan(operand, manager))
                {
                    return None;
                }

                let (eb, sb) = self.format_of(*term, manager)?;
                let nan = manager.mk_fp_nan(eb, sb);
                Some(manager.mk_eq(*term, nan))
            }
            FloatingPointAxiom::InfinityAxiom { term, property } => {
                self.infinity_axiom_term(*term, *property, manager)
            }
            FloatingPointAxiom::CompareNaN { lhs, rhs } => {
                if !self.is_fp_nan(*lhs, manager) && !self.is_fp_nan(*rhs, manager) {
                    return None;
                }

                let comparisons = [
                    manager.mk_fp_lt(*lhs, *rhs),
                    manager.mk_fp_leq(*lhs, *rhs),
                    manager.mk_fp_gt(*lhs, *rhs),
                    manager.mk_fp_geq(*lhs, *rhs),
                    manager.mk_fp_eq(*lhs, *rhs),
                ];
                let negations: Vec<TermId> = comparisons
                    .into_iter()
                    .map(|comparison| manager.mk_not(comparison))
                    .collect();
                Some(manager.mk_and(negations))
            }
        }
    }

    /// Build the guarded form of an infinity property
    fn infinity_axiom_term(
        &self,
        term: TermId,
        property: InfinityProperty,
        manager: &mut TermManager,
    ) -> Option<TermId> {
        match property {
            InfinityProperty::AddPosInf => {
                let TermKind::FpAdd(_, lhs, rhs) = manager.get(term)?.kind else {
                    return None;
                };
                let (eb, sb) = self.format_of(term, manager)?;
                let infinity = manager.mk_fp_plus_infinity(eb, sb);
                if lhs != infinity || rhs != infinity {
                    return None;
                }
                Some(manager.mk_eq(term, infinity))
            }
            InfinityProperty::MulPosInf => {
                let TermKind::FpMul(_, lhs, rhs) = manager.get(term)?.kind else {
                    return None;
                };
                let (eb, sb) = self.format_of(term, manager)?;
                let infinity = manager.mk_fp_plus_infinity(eb, sb);
                let other = if lhs == infinity {
                    rhs
                } else if rhs == infinity {
                    lhs
                } else {
                    return None;
                };

                let equation = manager.mk_eq(term, infinity);
                let positive = manager.mk_fp_is_positive(other);
                let is_zero = manager.mk_fp_is_zero(other);
                let nonzero = manager.mk_not(is_zero);
                let guard = manager.mk_and([positive, nonzero]);
                Some(manager.mk_implies(guard, equation))
            }
            InfinityProperty::CompareGreater => {
                let TermKind::FpGt(lhs, rhs) = manager.get(term)?.kind else {
                    return None;
                };
                let (eb, sb) = self.format_of(lhs, manager)?;
                let infinity = manager.mk_fp_plus_infinity(eb, sb);
                if lhs != infinity {
                    return None;
                }

                let is_nan = manager.mk_fp_is_nan(rhs);
                let not_nan = manager.mk_not(is_nan);
                let is_infinite = manager.mk_fp_is_infinite(rhs);
                let finite = manager.mk_not(is_infinite);
                let guard = manager.mk_and([not_nan, finite]);
                Some(manager.mk_implies(guard, term))
            }
        }
    }

    /// The operand that is *not* the special value the caller is looking for
    fn other_operand(
        &self,
        lhs: TermId,
        rhs: TermId,
        manager: &TermManager,
        is_special: fn(&Self, TermId, &TermManager) -> bool,
    ) -> Option<TermId> {
        if is_special(self, rhs, manager) {
            Some(lhs)
        } else if is_special(self, lhs, manager) {
            Some(rhs)
        } else {
            None
        }
    }

    /// Exponent and significand widths of a term's floating-point sort
    fn format_of(&self, term: TermId, manager: &TermManager) -> Option<(u32, u32)> {
        let sort = manager.get(term)?.sort;
        match manager.sorts.get(sort)?.kind {
            SortKind::FloatingPoint { eb, sb } => Some((eb, sb)),
            _ => None,
        }
    }

    /// Deduce new equalities between the registered floating-point terms
    ///
    /// Two rules are applied to a fixpoint:
    ///
    /// * *exact evaluation* — `fp.neg` and `fp.abs` of a literal are computed.
    ///   They are the only floating-point operations that need no rounding, so
    ///   they are the only ones folded here; `fp.add` and friends are left to
    ///   `oxiz_theories::fp`.
    /// * *congruence* — two applications of the same operator (with the same
    ///   rounding mode) whose arguments are pairwise known equal are equal.
    ///
    /// Each deduced equality is merged into this theory's classes and returned
    /// once; later calls return only what is new since the previous one.
    pub fn propagate(&mut self, manager: &mut TermManager) -> Vec<(TermId, TermId)> {
        let mut deduced = Vec::new();

        loop {
            let mut round = self.evaluate_exact_operations(manager);
            round.extend(self.congruent_pairs(manager));

            let mut progressed = false;
            for (a, b) in round {
                let key = if a.0 < b.0 { (a, b) } else { (b, a) };
                let merged = self.classes.union(a, b);
                if self.reported.insert(key) {
                    deduced.push(key);
                    progressed = true;
                } else if merged {
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }

        self.propagations += deduced.len();
        deduced
    }

    /// One term per class that is a floating-point literal, keyed by representative
    fn literal_representatives(&mut self, manager: &TermManager) -> FxHashMap<TermId, TermId> {
        let mut literals = FxHashMap::default();

        for class in self.classes.classes() {
            for &term in &class {
                if Self::literal_value(term, manager).is_some() {
                    let representative = self.classes.find(term);
                    literals.entry(representative).or_insert(term);
                    break;
                }
            }
        }

        literals
    }

    /// Equalities from evaluating `fp.neg` and `fp.abs` on known literals
    fn evaluate_exact_operations(&mut self, manager: &mut TermManager) -> Vec<(TermId, TermId)> {
        let literals = self.literal_representatives(manager);
        let mut terms: Vec<TermId> = self.floats.keys().copied().collect();
        terms.sort_unstable_by_key(|term| term.0);

        let mut evaluated = Vec::new();
        for term in terms {
            let Some(kind) = manager.get(term).map(|t| t.kind.clone()) else {
                continue;
            };
            let (argument, negate) = match kind {
                TermKind::FpNeg(argument) => (argument, true),
                TermKind::FpAbs(argument) => (argument, false),
                _ => continue,
            };

            let representative = self.classes.find(argument);
            let Some(&literal) = literals.get(&representative) else {
                continue;
            };
            let Some((format, value)) = Self::literal_value(literal, manager) else {
                continue;
            };

            let result = if negate {
                negated_value(&value)
            } else {
                absolute_value(&value)
            };
            let built = build_literal(format, &result, manager);

            let (eb, sb) = format;
            let sort = manager.sorts.float_sort(eb, sb);
            self.register_float(built, sort);
            if built != term {
                evaluated.push((term, built));
            }
        }

        evaluated
    }

    /// Equalities between applications of the same operator to equal arguments
    fn congruent_pairs(&mut self, manager: &TermManager) -> Vec<(TermId, TermId)> {
        let mut terms: Vec<TermId> = self.floats.keys().copied().collect();
        terms.sort_unstable_by_key(|term| term.0);

        let mut seen: FxHashMap<(u16, u16, Vec<TermId>), TermId> = FxHashMap::default();
        let mut congruent = Vec::new();

        for term in terms {
            let Some(kind) = manager.get(term).map(|t| t.kind.clone()) else {
                continue;
            };
            let Some((tag, rounding)) = fp_operator_key(&kind) else {
                continue;
            };
            let arguments: Vec<TermId> = get_children(&kind)
                .into_iter()
                .map(|child| self.classes.find(child))
                .collect();

            let key = (tag, rounding, arguments);
            match seen.get(&key) {
                Some(&other) => {
                    if !self.classes.are_equal(term, other) {
                        congruent.push((other, term));
                    }
                }
                None => {
                    seen.insert(key, term);
                }
            }
        }

        congruent
    }

    /// Check for conflicts in the current state
    ///
    /// A class holding two literals that denote different values cannot be
    /// satisfied. Spellings are compared by value, so `-0` and `+0` do clash
    /// (they are different values), while two different spellings of NaN do
    /// not (SMT-LIB has one NaN).
    ///
    /// The returned chain starts at one literal and ends at the other, and
    /// every consecutive pair in it was asserted or deduced equal.
    ///
    /// `None` means "no conflict found by this theory", never "satisfiable".
    pub fn check_for_conflicts(&mut self, manager: &TermManager) -> Option<Vec<TermId>> {
        for class in self.classes.classes() {
            // Bucketed by format: only literals of one format are comparable,
            // and a literal of another must not shadow the first one of its
            // own, or a real clash further along would be missed.
            let mut witnesses: FxHashMap<(u32, u32), (TermId, FloatValue)> = FxHashMap::default();

            for term in class {
                let Some((format, value)) = Self::literal_value(term, manager) else {
                    continue;
                };

                match witnesses.get(&format) {
                    Some((first, first_value)) => {
                        if *first_value != value {
                            let first = *first;
                            self.conflicts += 1;
                            let explanation = self.classes.explain(first, term);
                            return Some(if explanation.is_empty() {
                                vec![first, term]
                            } else {
                                explanation
                            });
                        }
                    }
                    None => {
                        witnesses.insert(format, (term, value));
                    }
                }
            }
        }

        None
    }

    /// Reset the theory state (for backtracking)
    pub fn reset(&mut self) {
        self.floats.clear();
        self.special_values.clear();
        self.classes.reset();
        self.reported.clear();
        self.pending_axioms.clear();
        self.instantiated.clear();
        self.propagations = 0;
        self.conflicts = 0;
    }

    /// Get statistics
    pub fn statistics(&self) -> FloatingPointStatistics {
        FloatingPointStatistics {
            num_floats: self.floats.len(),
            num_axioms: self.instantiated.len(),
            num_equality_nodes: self.classes.len(),
            num_propagations: self.propagations,
            num_conflicts: self.conflicts,
        }
    }
}

/// The exponent bias of a format, `2^(eb-1) - 1`
fn bias(eb: u32) -> BigInt {
    if eb == 0 {
        return BigInt::ZERO;
    }
    (BigInt::from(1u8) << (eb as usize - 1)) - 1
}

/// `fp.neg` of a value: the sign bit flips, NaN stays the one NaN
fn negated_value(value: &FloatValue) -> FloatValue {
    match value {
        FloatValue::NaN => FloatValue::NaN,
        FloatValue::PosInfinity => FloatValue::NegInfinity,
        FloatValue::NegInfinity => FloatValue::PosInfinity,
        FloatValue::PosZero => FloatValue::NegZero,
        FloatValue::NegZero => FloatValue::PosZero,
        FloatValue::Finite {
            sign,
            exponent,
            significand,
        } => FloatValue::Finite {
            sign: !*sign,
            exponent: exponent.clone(),
            significand: significand.clone(),
        },
    }
}

/// `fp.abs` of a value: the sign bit clears, NaN stays the one NaN
fn absolute_value(value: &FloatValue) -> FloatValue {
    match value {
        FloatValue::NaN => FloatValue::NaN,
        FloatValue::PosInfinity | FloatValue::NegInfinity => FloatValue::PosInfinity,
        FloatValue::PosZero | FloatValue::NegZero => FloatValue::PosZero,
        FloatValue::Finite {
            exponent,
            significand,
            ..
        } => FloatValue::Finite {
            sign: false,
            exponent: exponent.clone(),
            significand: significand.clone(),
        },
    }
}

/// Build the literal term for a value in a given format
fn build_literal(format: (u32, u32), value: &FloatValue, manager: &mut TermManager) -> TermId {
    let (eb, sb) = format;
    match value {
        FloatValue::NaN => manager.mk_fp_nan(eb, sb),
        FloatValue::PosInfinity => manager.mk_fp_plus_infinity(eb, sb),
        FloatValue::NegInfinity => manager.mk_fp_minus_infinity(eb, sb),
        FloatValue::PosZero => manager.mk_fp_plus_zero(eb, sb),
        FloatValue::NegZero => manager.mk_fp_minus_zero(eb, sb),
        FloatValue::Finite {
            sign,
            exponent,
            significand,
        } => manager.mk_fp_lit(*sign, exponent.clone(), significand.clone(), eb, sb),
    }
}

/// Operator identity of a floating-point term, for congruence lookup
///
/// The second component distinguishes rounding modes; it is zero for the
/// operators that take none. Returns `None` for terms that are not
/// floating-point operations.
fn fp_operator_key(kind: &TermKind) -> Option<(u16, u16)> {
    let rounding = |mode: RoundingMode| match mode {
        RoundingMode::RNE => 1,
        RoundingMode::RNA => 2,
        RoundingMode::RTP => 3,
        RoundingMode::RTN => 4,
        RoundingMode::RTZ => 5,
    };

    Some(match kind {
        TermKind::FpAbs(_) => (1, 0),
        TermKind::FpNeg(_) => (2, 0),
        TermKind::FpSqrt(mode, _) => (3, rounding(*mode)),
        TermKind::FpRoundToIntegral(mode, _) => (4, rounding(*mode)),
        TermKind::FpAdd(mode, _, _) => (5, rounding(*mode)),
        TermKind::FpSub(mode, _, _) => (6, rounding(*mode)),
        TermKind::FpMul(mode, _, _) => (7, rounding(*mode)),
        TermKind::FpDiv(mode, _, _) => (8, rounding(*mode)),
        TermKind::FpRem(_, _) => (9, 0),
        TermKind::FpMin(_, _) => (10, 0),
        TermKind::FpMax(_, _) => (11, 0),
        TermKind::FpFma(mode, _, _, _) => (12, rounding(*mode)),
        _ => return None,
    })
}

impl Theory for FloatingPointTheory {
    fn add_term(&mut self, term: TermId, manager: &TermManager) -> bool {
        FloatingPointTheory::add_term(self, term, manager, &manager.sorts)
    }

    fn assert_equality(&mut self, a: TermId, b: TermId) -> bool {
        FloatingPointTheory::assert_equality(self, a, b)
    }

    /// Propagates, then reports a conflict, then states the axioms it can
    ///
    /// An axiom that [`FloatingPointTheory::axiom_to_term`] cannot build stays
    /// queued rather than being dropped, so
    /// [`FloatingPointTheory::take_pending_axioms`] still hands it to the
    /// caller. It produces no lemma, so leaving it queued does not keep the
    /// combiner's loop running.
    fn check(&mut self, manager: &mut TermManager) -> TheoryResult {
        let deduced = self.propagate(manager);

        if let Some(explanation) = self.check_for_conflicts(manager) {
            return TheoryResult::Unsat { explanation };
        }

        if !deduced.is_empty() {
            return TheoryResult::Propagate(deduced);
        }

        let queued = self.take_pending_axioms();
        let mut lemmas = Vec::with_capacity(queued.len());
        for axiom in queued {
            match self.axiom_to_term(&axiom, manager) {
                Some(term) => lemmas.push(term),
                None => self.pending_axioms.push(axiom),
            }
        }

        if lemmas.is_empty() {
            TheoryResult::Sat
        } else {
            TheoryResult::Lemmas(lemmas)
        }
    }

    fn name(&self) -> &str {
        "floatingpoint"
    }

    fn reset(&mut self) {
        FloatingPointTheory::reset(self);
    }
}

/// Statistics for floating-point theory
#[derive(Debug, Clone, Copy)]
pub struct FloatingPointStatistics {
    /// Number of floating-point terms
    pub num_floats: usize,
    /// Number of axioms instantiated
    pub num_axioms: usize,
    /// Number of terms held in the equality classes
    pub num_equality_nodes: usize,
    /// Number of equalities deduced by `propagate`
    pub num_propagations: usize,
    /// Number of conflicts detected
    pub num_conflicts: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TermManager;
    use crate::sort::SortManager;

    #[test]
    fn test_empty_theory() {
        let theory = FloatingPointTheory::new();
        assert_eq!(theory.floats.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
    }

    #[test]
    fn test_register_float() {
        let mut theory = FloatingPointTheory::new();
        let term = TermId(42);
        let sort = SortId(1);
        theory.register_float(term, sort);
        assert_eq!(theory.floats.get(&term), Some(&sort));
    }

    #[test]
    fn test_no_duplicate_axioms() {
        let mut theory = FloatingPointTheory::new();
        let axiom = FloatingPointAxiom::NegInvolution { term: TermId(1) };

        theory.add_axiom(axiom.clone());
        theory.add_axiom(axiom);

        assert_eq!(theory.pending_axioms.len(), 1);
    }

    #[test]
    fn test_reset() {
        let mut theory = FloatingPointTheory::new();
        theory.register_float(TermId(1), SortId(1));
        theory.add_axiom(FloatingPointAxiom::NegInvolution { term: TermId(1) });

        theory.reset();

        assert_eq!(theory.floats.len(), 0);
        assert_eq!(theory.pending_axioms.len(), 0);
        assert_eq!(theory.instantiated.len(), 0);
    }

    #[test]
    fn test_statistics_counts_deduced_equalities() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let minus_zero = manager.mk_fp_minus_zero(8, 24);
        let negated = manager.mk_fp_neg(minus_zero);
        for term in [minus_zero, negated] {
            theory.add_term(term, &manager, &manager.sorts);
        }

        let deduced = theory.propagate(&mut manager);
        let stats = theory.statistics();
        assert_eq!(stats.num_propagations, deduced.len());
        assert!(stats.num_equality_nodes >= 2);
    }

    #[test]
    fn test_propagate_with_nothing_known_deduces_nothing() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let negated = manager.mk_fp_neg(x);
        for term in [x, negated] {
            theory.add_term(term, &manager, &manager.sorts);
        }

        assert!(theory.propagate(&mut manager).is_empty());
    }

    #[test]
    fn test_rounding_modes() {
        use RoundingMode::*;
        let modes = [RNE, RNA, RTP, RTN, RTZ];
        assert_eq!(modes.len(), 5);
    }

    #[test]
    fn test_literal_recognition() {
        let theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let plus_zero = manager.mk_fp_plus_zero(8, 24);
        let minus_zero = manager.mk_fp_minus_zero(8, 24);
        let nan = manager.mk_fp_nan(8, 24);
        // A NaN spelled out as a literal: all-ones exponent, non-zero significand.
        let spelled_nan = manager.mk_fp_lit(false, 255, 1, 8, 24);
        // 1.0 in Float32: exponent field = bias = 127, significand = 0.
        let one = manager.mk_fp_lit(false, 127, 0, 8, 24);

        assert!(theory.is_fp_pos_zero(plus_zero, &manager));
        assert!(!theory.is_fp_pos_zero(minus_zero, &manager));
        assert!(theory.is_fp_nan(nan, &manager));
        assert!(theory.is_fp_nan(spelled_nan, &manager));
        assert!(theory.is_fp_one(one, &manager));
        assert!(!theory.is_fp_one(plus_zero, &manager));
    }

    #[test]
    fn test_two_spellings_of_nan_are_not_a_conflict() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let nan = manager.mk_fp_nan(8, 24);
        let spelled_nan = manager.mk_fp_lit(true, 255, 3, 8, 24);
        for term in [nan, spelled_nan] {
            theory.add_term(term, &manager, &manager.sorts);
        }
        assert!(theory.assert_equality(nan, spelled_nan));

        assert!(theory.check_for_conflicts(&manager).is_none());
    }

    #[test]
    fn test_conflict_on_positive_and_negative_zero() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let plus_zero = manager.mk_fp_plus_zero(8, 24);
        let minus_zero = manager.mk_fp_minus_zero(8, 24);
        for term in [plus_zero, minus_zero] {
            theory.add_term(term, &manager, &manager.sorts);
        }
        assert!(theory.assert_equality(plus_zero, minus_zero));

        let explanation = theory
            .check_for_conflicts(&manager)
            .expect("+0 and -0 are different values");
        assert!(explanation.contains(&plus_zero));
        assert!(explanation.contains(&minus_zero));
    }

    #[test]
    fn test_negation_of_a_literal_is_evaluated() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let minus_zero = manager.mk_fp_minus_zero(8, 24);
        let negated = manager.mk_fp_neg(minus_zero);
        for term in [minus_zero, negated] {
            theory.add_term(term, &manager, &manager.sorts);
        }

        let deduced = theory.propagate(&mut manager);
        assert!(!deduced.is_empty(), "fp.neg(-0) should evaluate");

        let plus_zero = manager.mk_fp_plus_zero(8, 24);
        assert!(theory.known_equal(negated, plus_zero));
    }

    #[test]
    fn test_absolute_value_of_a_literal_is_evaluated() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let negative = manager.mk_fp_lit(true, 127, 0, 8, 24);
        let absolute = manager.mk_fp_abs(negative);
        for term in [negative, absolute] {
            theory.add_term(term, &manager, &manager.sorts);
        }

        theory.propagate(&mut manager);
        let positive = manager.mk_fp_lit(false, 127, 0, 8, 24);
        assert!(theory.known_equal(absolute, positive));
    }

    #[test]
    fn test_congruence_deduces_an_equality() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let y = manager.mk_var("y", fp_sort);
        let z = manager.mk_var("z", fp_sort);
        let left = manager.mk_fp_add(RoundingMode::RNE, x, z);
        let right = manager.mk_fp_add(RoundingMode::RNE, y, z);

        for term in [x, y, z, left, right] {
            theory.add_term(term, &manager, &manager.sorts);
        }
        assert!(theory.assert_equality(x, y));

        theory.propagate(&mut manager);
        assert!(theory.known_equal(left, right));
    }

    #[test]
    fn test_different_rounding_modes_are_not_congruent() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let y = manager.mk_var("y", fp_sort);
        let nearest = manager.mk_fp_add(RoundingMode::RNE, x, y);
        let toward_zero = manager.mk_fp_add(RoundingMode::RTZ, x, y);

        for term in [x, y, nearest, toward_zero] {
            theory.add_term(term, &manager, &manager.sorts);
        }

        theory.propagate(&mut manager);
        assert!(!theory.known_equal(nearest, toward_zero));
    }

    #[test]
    fn test_add_identity_axiom_is_guarded() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let zero = manager.mk_fp_plus_zero(8, 24);
        let sum = manager.mk_fp_add(RoundingMode::RNE, x, zero);

        assert!(theory.add_term(sum, &manager, &manager.sorts));
        let axioms = theory.take_pending_axioms();
        assert!(axioms.contains(&FloatingPointAxiom::AddIdentity {
            term: sum,
            rounding_mode: RoundingMode::RNE,
        }));

        let term = theory
            .axiom_to_term(
                &FloatingPointAxiom::AddIdentity {
                    term: sum,
                    rounding_mode: RoundingMode::RNE,
                },
                &mut manager,
            )
            .expect("the guarded identity should be buildable");

        // not (fp.isZero x) => (fp.add RNE x +0) = x
        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Implies(guard, conclusion)) => {
                assert!(matches!(
                    manager.get(guard).map(|t| t.kind.clone()),
                    Some(TermKind::Not(_))
                ));
                assert!(matches!(
                    manager.get(conclusion).map(|t| t.kind.clone()),
                    Some(TermKind::Eq(_, _))
                ));
            }
            other => panic!("expected an implication, got {other:?}"),
        }
    }

    #[test]
    fn test_mul_identity_axiom_needs_no_guard() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let one = manager.mk_fp_lit(false, 127, 0, 8, 24);
        let product = manager.mk_fp_mul(RoundingMode::RNE, x, one);

        theory.add_term(product, &manager, &manager.sorts);
        let term = theory
            .axiom_to_term(
                &FloatingPointAxiom::MulIdentity {
                    term: product,
                    rounding_mode: RoundingMode::RNE,
                },
                &mut manager,
            )
            .expect("fp.mul(rm, x, 1.0) = x should be buildable");

        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Eq(lhs, rhs)) => {
                assert!((lhs == product && rhs == x) || (lhs == x && rhs == product));
            }
            other => panic!("expected an equality, got {other:?}"),
        }
    }

    #[test]
    fn test_nan_propagation_axiom_equates_with_nan() {
        let theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let nan = manager.mk_fp_nan(8, 24);
        let sum = manager.mk_fp_add(RoundingMode::RNE, x, nan);

        let term = theory
            .axiom_to_term(
                &FloatingPointAxiom::NaNPropagation {
                    term: sum,
                    operation: FloatingPointOp::Add,
                },
                &mut manager,
            )
            .expect("the NaN propagation equation should be buildable");

        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::Eq(lhs, rhs)) => {
                assert!(lhs == nan || rhs == nan, "one side should be NaN");
            }
            other => panic!("expected an equality, got {other:?}"),
        }

        // Without a NaN operand the result is not NaN, so there is nothing to
        // state: an addition of two plain variables is not forced to be NaN.
        let y = manager.mk_var("y", fp_sort);
        let plain = manager.mk_fp_add(RoundingMode::RNE, x, y);
        assert!(
            theory
                .axiom_to_term(
                    &FloatingPointAxiom::NaNPropagation {
                        term: plain,
                        operation: FloatingPointOp::Add,
                    },
                    &mut manager
                )
                .is_none()
        );
    }

    #[test]
    fn test_compare_nan_axiom_negates_every_comparison() {
        let theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let nan = manager.mk_fp_nan(8, 24);

        let term = theory
            .axiom_to_term(
                &FloatingPointAxiom::CompareNaN { lhs: x, rhs: nan },
                &mut manager,
            )
            .expect("comparisons against NaN should be buildable");

        match manager.get(term).map(|t| t.kind.clone()) {
            Some(TermKind::And(conjuncts)) => assert_eq!(conjuncts.len(), 5),
            other => panic!("expected a conjunction of five negations, got {other:?}"),
        }

        // Without a NaN operand there is nothing to state.
        let y = manager.mk_var("y", fp_sort);
        assert!(
            theory
                .axiom_to_term(
                    &FloatingPointAxiom::CompareNaN { lhs: x, rhs: y },
                    &mut manager
                )
                .is_none()
        );
    }

    #[test]
    fn test_axiom_to_term_rejects_a_mismatched_term() {
        let theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);
        let y = manager.mk_var("y", fp_sort);
        let sum = manager.mk_fp_add(RoundingMode::RNE, x, y);

        // Neither operand is +0, so there is no identity to state.
        assert!(
            theory
                .axiom_to_term(
                    &FloatingPointAxiom::AddIdentity {
                        term: sum,
                        rounding_mode: RoundingMode::RNE,
                    },
                    &mut manager
                )
                .is_none()
        );

        // A term that is not an addition at all.
        assert!(
            theory
                .axiom_to_term(
                    &FloatingPointAxiom::AddIdentity {
                        term: x,
                        rounding_mode: RoundingMode::RNE,
                    },
                    &mut manager
                )
                .is_none()
        );
    }

    #[test]
    fn test_special_values_are_indexed_per_sort() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let fp_sort = manager.sorts.float_sort(8, 24);
        let nan = manager.mk_fp_nan(8, 24);
        let spelled_minus_zero = manager.mk_fp_lit(true, 0, 0, 8, 24);

        theory.add_term(nan, &manager, &manager.sorts);
        theory.add_term(spelled_minus_zero, &manager, &manager.sorts);

        assert_eq!(
            theory.special_value_term(fp_sort, SpecialValue::NaN),
            Some(nan)
        );
        // Recognised through its bit pattern, not its constructor.
        assert_eq!(
            theory.special_value_term(fp_sort, SpecialValue::NegZero),
            Some(spelled_minus_zero)
        );
        assert_eq!(
            theory.special_value_term(fp_sort, SpecialValue::PosInfinity),
            None
        );
    }

    #[test]
    fn test_add_term_registration() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();
        let mut sort_manager = SortManager::new();

        // Create a Float32 sort (exponent=8, significand=24)
        let fp_sort = sort_manager.float_sort(8, 24);
        let x = manager.mk_var("x", fp_sort);

        assert!(theory.add_term(x, &manager, &sort_manager));
        assert!(theory.floats.contains_key(&x));

        let int_sort = manager.sorts.int_sort;
        let n = manager.mk_var("n", int_sort);
        assert!(!theory.add_term(n, &manager, &manager.sorts));
    }

    #[test]
    fn test_theory_trait_reports_conflict() {
        let mut theory = FloatingPointTheory::new();
        let mut manager = TermManager::new();

        let plus_zero = manager.mk_fp_plus_zero(8, 24);
        let minus_zero = manager.mk_fp_minus_zero(8, 24);
        for term in [plus_zero, minus_zero] {
            Theory::add_term(&mut theory, term, &manager);
        }
        assert!(Theory::assert_equality(&mut theory, plus_zero, minus_zero));

        assert!(matches!(
            Theory::check(&mut theory, &mut manager),
            TheoryResult::Unsat { .. }
        ));
        assert_eq!(Theory::name(&theory), "floatingpoint");
    }
}
