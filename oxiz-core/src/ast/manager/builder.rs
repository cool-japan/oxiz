//! Term builder methods for TermManager — all mk_* constructors

use super::super::term::{RoundingMode, TermId, TermKind};
#[allow(unused_imports)]
use crate::prelude::*;
use crate::sort::SortId;
use num_bigint::BigInt;
use num_rational::Rational64;
use smallvec::SmallVec;

use super::TermManager;

impl TermManager {
    /// Create the boolean true constant
    #[must_use]
    pub fn mk_true(&self) -> TermId {
        self.true_id
    }

    /// Create the boolean false constant
    #[must_use]
    pub fn mk_false(&self) -> TermId {
        self.false_id
    }

    /// Create a boolean constant
    #[must_use]
    pub fn mk_bool(&self, value: bool) -> TermId {
        if value { self.true_id } else { self.false_id }
    }

    /// Create an integer constant
    pub fn mk_int(&mut self, value: impl Into<BigInt>) -> TermId {
        let sort = self.sorts.int_sort;
        self.intern(TermKind::IntConst(value.into()), sort)
    }

    /// Create a rational constant
    pub fn mk_real(&mut self, value: Rational64) -> TermId {
        let sort = self.sorts.real_sort;
        self.intern(TermKind::RealConst(value), sort)
    }

    /// Create a bit vector constant
    pub fn mk_bitvec(&mut self, value: impl Into<BigInt>, width: u32) -> TermId {
        let sort = self.sorts.bitvec(width);
        self.intern(
            TermKind::BitVecConst {
                value: value.into(),
                width,
            },
            sort,
        )
    }

    /// Create a named variable
    pub fn mk_var(&mut self, name: &str, sort: SortId) -> TermId {
        let spur = self.intern_str(name);
        self.intern(TermKind::Var(spur), sort)
    }

    /// Create a logical NOT
    pub fn mk_not(&mut self, arg: TermId) -> TermId {
        // Simplify double negation
        if let Some(term) = self.get(arg) {
            if let TermKind::Not(inner) = term.kind {
                return inner;
            }
            if let TermKind::True = term.kind {
                return self.false_id;
            }
            if let TermKind::False = term.kind {
                return self.true_id;
            }
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Not(arg), sort)
    }

    /// Create a logical AND
    pub fn mk_and(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let mut flat_args: SmallVec<[TermId; 4]> = SmallVec::new();

        for arg in args {
            if let Some(term) = self.get(arg) {
                match &term.kind {
                    TermKind::False => return self.false_id,
                    TermKind::True => continue,
                    TermKind::And(inner) => flat_args.extend(inner.iter().copied()),
                    _ => flat_args.push(arg),
                }
            } else {
                flat_args.push(arg);
            }
        }

        match flat_args.len() {
            0 => self.true_id,
            1 => flat_args[0],
            _ => {
                let sort = self.sorts.bool_sort;
                self.intern(TermKind::And(flat_args), sort)
            }
        }
    }

    /// Create a logical OR
    pub fn mk_or(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let mut flat_args: SmallVec<[TermId; 4]> = SmallVec::new();

        for arg in args {
            if let Some(term) = self.get(arg) {
                match &term.kind {
                    TermKind::True => return self.true_id,
                    TermKind::False => continue,
                    TermKind::Or(inner) => flat_args.extend(inner.iter().copied()),
                    _ => flat_args.push(arg),
                }
            } else {
                flat_args.push(arg);
            }
        }

        match flat_args.len() {
            0 => self.false_id,
            1 => flat_args[0],
            _ => {
                let sort = self.sorts.bool_sort;
                self.intern(TermKind::Or(flat_args), sort)
            }
        }
    }

    /// Create a logical implication
    pub fn mk_implies(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Simplifications
        if let Some(term) = self.get(lhs) {
            if let TermKind::False = term.kind {
                return self.true_id;
            }
            if let TermKind::True = term.kind {
                return rhs;
            }
        }
        if let Some(term) = self.get(rhs)
            && let TermKind::True = term.kind
        {
            return self.true_id;
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Implies(lhs, rhs), sort)
    }

    /// Create a logical XOR
    pub fn mk_xor(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        // Simplifications
        if lhs == rhs {
            return self.false_id;
        }
        if let Some(term) = self.get(lhs) {
            if let TermKind::False = term.kind {
                return rhs;
            }
            if let TermKind::True = term.kind {
                return self.mk_not(rhs);
            }
        }
        if let Some(term) = self.get(rhs) {
            if let TermKind::False = term.kind {
                return lhs;
            }
            if let TermKind::True = term.kind {
                return self.mk_not(lhs);
            }
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Xor(lhs, rhs), sort)
    }

    /// Create an if-then-else
    pub fn mk_ite(&mut self, cond: TermId, then_branch: TermId, else_branch: TermId) -> TermId {
        // Simplifications
        if let Some(term) = self.get(cond) {
            if let TermKind::True = term.kind {
                return then_branch;
            }
            if let TermKind::False = term.kind {
                return else_branch;
            }
        }
        if then_branch == else_branch {
            return then_branch;
        }
        // ite(c, true, false) => c
        let then_is_true = self
            .get(then_branch)
            .is_some_and(|t| matches!(t.kind, TermKind::True));
        let else_is_false = self
            .get(else_branch)
            .is_some_and(|t| matches!(t.kind, TermKind::False));
        if then_is_true && else_is_false {
            return cond;
        }

        let sort = self
            .get(then_branch)
            .map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::Ite(cond, then_branch, else_branch), sort)
    }

    /// Create an equality
    pub fn mk_eq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        if lhs == rhs {
            return self.true_id;
        }

        // Check for constant comparisons
        let lhs_kind = self.get(lhs).map(|t| t.kind.clone());
        let rhs_kind = self.get(rhs).map(|t| t.kind.clone());

        match (&lhs_kind, &rhs_kind) {
            // Integer constants
            (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) => {
                return self.mk_bool(a == b);
            }
            // Boolean constants
            (Some(TermKind::True), Some(TermKind::True)) => return self.true_id,
            (Some(TermKind::False), Some(TermKind::False)) => return self.true_id,
            (Some(TermKind::True), Some(TermKind::False)) => return self.false_id,
            (Some(TermKind::False), Some(TermKind::True)) => return self.false_id,
            // BitVec constants
            (
                Some(TermKind::BitVecConst {
                    value: v1,
                    width: w1,
                }),
                Some(TermKind::BitVecConst {
                    value: v2,
                    width: w2,
                }),
            ) => {
                return self.mk_bool(v1 == v2 && w1 == w2);
            }
            _ => {}
        }

        // Canonicalize order
        let (lhs, rhs) = if lhs.0 <= rhs.0 {
            (lhs, rhs)
        } else {
            (rhs, lhs)
        };

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Eq(lhs, rhs), sort)
    }

    /// Create a distinct constraint
    pub fn mk_distinct(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();

        if args.len() <= 1 {
            return self.true_id;
        }

        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Distinct(args), sort)
    }

    /// Create an addition
    pub fn mk_add(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();

        match args.len() {
            0 => self.mk_int(0),
            1 => args[0],
            _ => {
                let sort = self.get(args[0]).map_or(self.sorts.int_sort, |t| t.sort);
                self.intern(TermKind::Add(args), sort)
            }
        }
    }

    /// Create a subtraction
    pub fn mk_sub(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Sub(lhs, rhs), sort)
    }

    /// Create arithmetic negation
    pub fn mk_neg(&mut self, arg: TermId) -> TermId {
        let sort = self.get(arg).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Neg(arg), sort)
    }

    /// Create a multiplication
    pub fn mk_mul(&mut self, args: impl IntoIterator<Item = TermId>) -> TermId {
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();

        match args.len() {
            0 => self.mk_int(1),
            1 => args[0],
            _ => {
                let sort = self.get(args[0]).map_or(self.sorts.int_sort, |t| t.sort);
                self.intern(TermKind::Mul(args), sort)
            }
        }
    }

    /// Create a division
    pub fn mk_div(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Div(lhs, rhs), sort)
    }

    /// Create a modulo operation
    pub fn mk_mod(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Mod(lhs, rhs), sort)
    }

    /// Create a less-than comparison
    pub fn mk_lt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Lt(lhs, rhs), sort)
    }

    /// Create a less-than-or-equal comparison
    pub fn mk_le(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Le(lhs, rhs), sort)
    }

    /// Create a greater-than comparison
    pub fn mk_gt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Gt(lhs, rhs), sort)
    }

    /// Create a greater-than-or-equal comparison
    pub fn mk_ge(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::Ge(lhs, rhs), sort)
    }

    /// Create a greater-than-or-equal comparison (alias for mk_ge)
    pub fn mk_geq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_ge(lhs, rhs)
    }

    /// Create a less-than-or-equal comparison (alias for mk_le)
    pub fn mk_leq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        self.mk_le(lhs, rhs)
    }

    /// Create an array select operation
    pub fn mk_select(&mut self, array: TermId, index: TermId) -> TermId {
        // Get the range sort from the array's sort
        let sort = if let Some(term) = self.get(array) {
            if let Some(array_sort) = self.sorts.get(term.sort) {
                if let crate::sort::SortKind::Array { range, .. } = array_sort.kind {
                    range
                } else {
                    self.sorts.int_sort
                }
            } else {
                self.sorts.int_sort
            }
        } else {
            self.sorts.int_sort
        };
        self.intern(TermKind::Select(array, index), sort)
    }

    /// Create an array store operation
    pub fn mk_store(&mut self, array: TermId, index: TermId, value: TermId) -> TermId {
        let sort = self.get(array).map_or(self.sorts.int_sort, |t| t.sort);
        self.intern(TermKind::Store(array, index, value), sort)
    }

    /// Create a string literal
    pub fn mk_string_lit(&mut self, value: &str) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StringLit(value.to_string()), string_sort)
    }

    /// Create a string concatenation
    pub fn mk_str_concat(&mut self, s1: TermId, s2: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrConcat(s1, s2), string_sort)
    }

    /// Create a string length operation
    pub fn mk_str_len(&mut self, s: TermId) -> TermId {
        let int_sort = self.sorts.int_sort;
        self.intern(TermKind::StrLen(s), int_sort)
    }

    /// Create a substring operation
    pub fn mk_str_substr(&mut self, s: TermId, start: TermId, len: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrSubstr(s, start, len), string_sort)
    }

    /// Create a character at index operation
    pub fn mk_str_at(&mut self, s: TermId, i: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrAt(s, i), string_sort)
    }

    /// Create a contains substring operation
    pub fn mk_str_contains(&mut self, s: TermId, sub: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrContains(s, sub), bool_sort)
    }

    /// Create a prefix check operation
    pub fn mk_str_prefixof(&mut self, prefix: TermId, s: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrPrefixOf(prefix, s), bool_sort)
    }

    /// Create a suffix check operation
    pub fn mk_str_suffixof(&mut self, suffix: TermId, s: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrSuffixOf(suffix, s), bool_sort)
    }

    /// Create an index of operation
    pub fn mk_str_indexof(&mut self, s: TermId, sub: TermId, offset: TermId) -> TermId {
        let int_sort = self.sorts.int_sort;
        self.intern(TermKind::StrIndexOf(s, sub, offset), int_sort)
    }

    /// Create a string replace operation
    pub fn mk_str_replace(&mut self, s: TermId, pattern: TermId, replacement: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::StrReplace(s, pattern, replacement), string_sort)
    }

    /// Create a replace all operation
    pub fn mk_str_replace_all(
        &mut self,
        s: TermId,
        pattern: TermId,
        replacement: TermId,
    ) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(
            TermKind::StrReplaceAll(s, pattern, replacement),
            string_sort,
        )
    }

    /// Create a string to integer conversion
    pub fn mk_str_to_int(&mut self, s: TermId) -> TermId {
        let int_sort = self.sorts.int_sort;
        self.intern(TermKind::StrToInt(s), int_sort)
    }

    /// Create an integer to string conversion
    pub fn mk_int_to_str(&mut self, i: TermId) -> TermId {
        let string_sort = self.sorts.string_sort();
        self.intern(TermKind::IntToStr(i), string_sort)
    }

    /// Create a string in regex operation
    pub fn mk_str_in_re(&mut self, s: TermId, re: TermId) -> TermId {
        let bool_sort = self.sorts.bool_sort;
        self.intern(TermKind::StrInRe(s, re), bool_sort)
    }

    // Floating-point operations

    /// Create a floating-point literal from components
    pub fn mk_fp_lit(
        &mut self,
        sign: bool,
        exp: impl Into<BigInt>,
        sig: impl Into<BigInt>,
        eb: u32,
        sb: u32,
    ) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(
            TermKind::FpLit {
                sign,
                exp: exp.into(),
                sig: sig.into(),
                eb,
                sb,
            },
            sort,
        )
    }

    /// Create floating-point positive infinity
    pub fn mk_fp_plus_infinity(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpPlusInfinity { eb, sb }, sort)
    }

    /// Create floating-point negative infinity
    pub fn mk_fp_minus_infinity(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpMinusInfinity { eb, sb }, sort)
    }

    /// Create floating-point positive zero
    pub fn mk_fp_plus_zero(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpPlusZero { eb, sb }, sort)
    }

    /// Create floating-point negative zero
    pub fn mk_fp_minus_zero(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpMinusZero { eb, sb }, sort)
    }

    /// Create floating-point NaN
    pub fn mk_fp_nan(&mut self, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpNaN { eb, sb }, sort)
    }

    /// Create floating-point absolute value
    pub fn mk_fp_abs(&mut self, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpAbs(arg), sort)
    }

    /// Create floating-point negation
    pub fn mk_fp_neg(&mut self, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpNeg(arg), sort)
    }

    /// Create floating-point square root
    pub fn mk_fp_sqrt(&mut self, rm: RoundingMode, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpSqrt(rm, arg), sort)
    }

    /// Create floating-point round to integral
    pub fn mk_fp_round_to_integral(&mut self, rm: RoundingMode, arg: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(arg).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpRoundToIntegral(rm, arg), sort)
    }

    /// Create floating-point addition
    pub fn mk_fp_add(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpAdd(rm, lhs, rhs), sort)
    }

    /// Create floating-point subtraction
    pub fn mk_fp_sub(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpSub(rm, lhs, rhs), sort)
    }

    /// Create floating-point multiplication
    pub fn mk_fp_mul(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpMul(rm, lhs, rhs), sort)
    }

    /// Create floating-point division
    pub fn mk_fp_div(&mut self, rm: RoundingMode, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpDiv(rm, lhs, rhs), sort)
    }

    /// Create floating-point remainder
    pub fn mk_fp_rem(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpRem(lhs, rhs), sort)
    }

    /// Create floating-point minimum
    pub fn mk_fp_min(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpMin(lhs, rhs), sort)
    }

    /// Create floating-point maximum
    pub fn mk_fp_max(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(lhs).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpMax(lhs, rhs), sort)
    }

    /// Create floating-point less than or equal comparison
    pub fn mk_fp_leq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpLeq(lhs, rhs), sort)
    }

    /// Create floating-point less than comparison
    pub fn mk_fp_lt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpLt(lhs, rhs), sort)
    }

    /// Create floating-point greater than or equal comparison
    pub fn mk_fp_geq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpGeq(lhs, rhs), sort)
    }

    /// Create floating-point greater than comparison
    pub fn mk_fp_gt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpGt(lhs, rhs), sort)
    }

    /// Create floating-point equality comparison
    pub fn mk_fp_eq(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpEq(lhs, rhs), sort)
    }

    /// Create floating-point fused multiply-add: (x * y) + z
    pub fn mk_fp_fma(&mut self, rm: RoundingMode, x: TermId, y: TermId, z: TermId) -> TermId {
        let default_sort = self.sorts.float32_sort();
        let sort = self.get(x).map_or(default_sort, |t| t.sort);
        self.intern(TermKind::FpFma(rm, x, y, z), sort)
    }

    /// Create floating-point is-normal predicate
    pub fn mk_fp_is_normal(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsNormal(arg), sort)
    }

    /// Create floating-point is-subnormal predicate
    pub fn mk_fp_is_subnormal(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsSubnormal(arg), sort)
    }

    /// Create floating-point is-zero predicate
    pub fn mk_fp_is_zero(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsZero(arg), sort)
    }

    /// Create floating-point is-infinite predicate
    pub fn mk_fp_is_infinite(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsInfinite(arg), sort)
    }

    /// Create floating-point is-NaN predicate
    pub fn mk_fp_is_nan(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsNaN(arg), sort)
    }

    /// Create floating-point is-negative predicate
    pub fn mk_fp_is_negative(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsNegative(arg), sort)
    }

    /// Create floating-point is-positive predicate
    pub fn mk_fp_is_positive(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::FpIsPositive(arg), sort)
    }

    /// Convert floating-point to another FP format
    pub fn mk_fp_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::FpToFp { rm, arg, eb, sb }, sort)
    }

    /// Convert floating-point to signed bitvector
    pub fn mk_fp_to_sbv(&mut self, rm: RoundingMode, arg: TermId, width: u32) -> TermId {
        let sort = self.sorts.bitvec(width);
        self.intern(TermKind::FpToSBV { rm, arg, width }, sort)
    }

    /// Convert floating-point to unsigned bitvector
    pub fn mk_fp_to_ubv(&mut self, rm: RoundingMode, arg: TermId, width: u32) -> TermId {
        let sort = self.sorts.bitvec(width);
        self.intern(TermKind::FpToUBV { rm, arg, width }, sort)
    }

    /// Convert floating-point to real
    pub fn mk_fp_to_real(&mut self, arg: TermId) -> TermId {
        let sort = self.sorts.real_sort;
        self.intern(TermKind::FpToReal(arg), sort)
    }

    /// Convert real to floating-point
    pub fn mk_real_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::RealToFp { rm, arg, eb, sb }, sort)
    }

    /// Convert signed bitvector to floating-point
    pub fn mk_sbv_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::SBVToFp { rm, arg, eb, sb }, sort)
    }

    /// Convert unsigned bitvector to floating-point
    pub fn mk_ubv_to_fp(&mut self, rm: RoundingMode, arg: TermId, eb: u32, sb: u32) -> TermId {
        let sort = self.sorts.float_sort(eb, sb);
        self.intern(TermKind::UBVToFp { rm, arg, eb, sb }, sort)
    }

    /// Create a function application
    pub fn mk_apply(
        &mut self,
        func: &str,
        args: impl IntoIterator<Item = TermId>,
        sort: SortId,
    ) -> TermId {
        let func_spur = self.intern_str(func);
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();
        self.intern(
            TermKind::Apply {
                func: func_spur,
                args,
            },
            sort,
        )
    }

    // Algebraic datatypes

    /// Create a datatype constructor application
    ///
    /// Constructs a datatype value using the specified constructor.
    /// For example, `cons(1, nil)` for a list.
    pub fn mk_dt_constructor(
        &mut self,
        constructor: &str,
        args: impl IntoIterator<Item = TermId>,
        sort: SortId,
    ) -> TermId {
        let constructor_spur = self.intern_str(constructor);
        let args: SmallVec<[TermId; 4]> = args.into_iter().collect();
        self.intern(
            TermKind::DtConstructor {
                constructor: constructor_spur,
                args,
            },
            sort,
        )
    }

    /// Create a datatype tester/discriminator
    ///
    /// Tests if a term was constructed with a specific constructor.
    /// For example, `is-cons(x)` tests if `x` is a cons cell.
    pub fn mk_dt_tester(&mut self, constructor: &str, arg: TermId) -> TermId {
        let constructor_spur = self.intern_str(constructor);
        let bool_sort = self.sorts.bool_sort;
        self.intern(
            TermKind::DtTester {
                constructor: constructor_spur,
                arg,
            },
            bool_sort,
        )
    }

    /// Create a datatype selector/accessor
    ///
    /// Extracts a field from a datatype value.
    /// For example, `head(x)` extracts the first element of a cons cell.
    pub fn mk_dt_selector(&mut self, selector: &str, arg: TermId, result_sort: SortId) -> TermId {
        let selector_spur = self.intern_str(selector);
        self.intern(
            TermKind::DtSelector {
                selector: selector_spur,
                arg,
            },
            result_sort,
        )
    }

    /// Create a universal quantifier without patterns
    pub fn mk_forall<'a>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
    ) -> TermId {
        self.mk_forall_with_patterns(vars, body, core::iter::empty::<Vec<TermId>>())
    }

    /// Create a universal quantifier with instantiation patterns
    ///
    /// Patterns are lists of terms that guide quantifier instantiation.
    /// Each pattern is a conjunction of terms that must match for instantiation.
    ///
    /// # Example
    /// ```ignore
    /// // (forall ((x Int)) (! (> (f x) 0) :pattern ((f x))))
    /// let x_var = manager.mk_var("x", int_sort);
    /// let fx = manager.mk_apply("f", [x_var], int_sort);
    /// let body = manager.mk_gt(fx, zero);
    /// let forall = manager.mk_forall_with_patterns(
    ///     [("x", int_sort)],
    ///     body,
    ///     [[fx]],  // pattern: (f x)
    /// );
    /// ```
    pub fn mk_forall_with_patterns<'a, P, Q>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
        patterns: P,
    ) -> TermId
    where
        P: IntoIterator<Item = Q>,
        Q: IntoIterator<Item = TermId>,
    {
        use crate::interner::Spur;
        let vars: SmallVec<[(Spur, SortId); 2]> = vars
            .into_iter()
            .map(|(name, sort)| (self.intern_str(name), sort))
            .collect();

        if vars.is_empty() {
            return body;
        }

        let patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
            .into_iter()
            .map(|p| p.into_iter().collect())
            .collect();

        let sort = self.sorts.bool_sort;
        self.intern(
            TermKind::Forall {
                vars,
                body,
                patterns,
            },
            sort,
        )
    }

    /// Create an existential quantifier without patterns
    pub fn mk_exists<'a>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
    ) -> TermId {
        self.mk_exists_with_patterns(vars, body, core::iter::empty::<Vec<TermId>>())
    }

    /// Create an existential quantifier with instantiation patterns
    pub fn mk_exists_with_patterns<'a, P, Q>(
        &mut self,
        vars: impl IntoIterator<Item = (&'a str, SortId)>,
        body: TermId,
        patterns: P,
    ) -> TermId
    where
        P: IntoIterator<Item = Q>,
        Q: IntoIterator<Item = TermId>,
    {
        use crate::interner::Spur;
        let vars: SmallVec<[(Spur, SortId); 2]> = vars
            .into_iter()
            .map(|(name, sort)| (self.intern_str(name), sort))
            .collect();

        if vars.is_empty() {
            return body;
        }

        let patterns: SmallVec<[SmallVec<[TermId; 2]>; 2]> = patterns
            .into_iter()
            .map(|p| p.into_iter().collect())
            .collect();

        let sort = self.sorts.bool_sort;
        self.intern(
            TermKind::Exists {
                vars,
                body,
                patterns,
            },
            sort,
        )
    }

    /// Create a let expression
    pub fn mk_let<'a>(
        &mut self,
        bindings: impl IntoIterator<Item = (&'a str, TermId)>,
        body: TermId,
    ) -> TermId {
        use crate::interner::Spur;
        let bindings: SmallVec<[(Spur, TermId); 2]> = bindings
            .into_iter()
            .map(|(name, term)| (self.intern_str(name), term))
            .collect();

        if bindings.is_empty() {
            return body;
        }

        let sort = self.get(body).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::Let { bindings, body }, sort)
    }

    // BitVector operations

    /// Create a bit vector concatenation
    pub fn mk_bv_concat(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let width = self
            .get(lhs)
            .and_then(|t| self.sorts.get(t.sort))
            .and_then(|s| s.bitvec_width())
            .unwrap_or(32)
            + self
                .get(rhs)
                .and_then(|t| self.sorts.get(t.sort))
                .and_then(|s| s.bitvec_width())
                .unwrap_or(32);
        let sort = self.sorts.bitvec(width);
        self.intern(TermKind::BvConcat(lhs, rhs), sort)
    }

    /// Create a bit vector extraction
    pub fn mk_bv_extract(&mut self, high: u32, low: u32, arg: TermId) -> TermId {
        let width = high - low + 1;
        let sort = self.sorts.bitvec(width);
        self.intern(TermKind::BvExtract { high, low, arg }, sort)
    }

    /// Create a bit vector NOT
    pub fn mk_bv_not(&mut self, arg: TermId) -> TermId {
        let sort = self.get(arg).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvNot(arg), sort)
    }

    /// Create a bit vector AND
    pub fn mk_bv_and(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvAnd(lhs, rhs), sort)
    }

    /// Create a bit vector OR
    pub fn mk_bv_or(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvOr(lhs, rhs), sort)
    }

    /// Create a bit vector addition
    pub fn mk_bv_add(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvAdd(lhs, rhs), sort)
    }

    /// Create a bit vector subtraction
    pub fn mk_bv_sub(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvSub(lhs, rhs), sort)
    }

    /// Create a bit vector multiplication
    pub fn mk_bv_mul(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvMul(lhs, rhs), sort)
    }

    /// Create a bit vector unsigned less-than
    pub fn mk_bv_ult(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvUlt(lhs, rhs), sort)
    }

    /// Create a bit vector signed less-than
    pub fn mk_bv_slt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvSlt(lhs, rhs), sort)
    }

    /// Create a bit vector unsigned less-than-or-equal
    pub fn mk_bv_ule(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvUle(lhs, rhs), sort)
    }

    /// Create a bit vector signed less-than-or-equal
    pub fn mk_bv_sle(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.sorts.bool_sort;
        self.intern(TermKind::BvSle(lhs, rhs), sort)
    }

    /// Create a bit vector negation (two's complement)
    /// Implemented as 0 - arg
    pub fn mk_bv_neg(&mut self, arg: TermId) -> TermId {
        // Get the width from the argument's sort
        let sort = self.get(arg).map_or(self.sorts.bool_sort, |t| t.sort);
        let width = self
            .sorts
            .get(sort)
            .and_then(|s| s.bitvec_width())
            .unwrap_or(32);
        let zero = self.mk_bitvec(0i64, width);
        self.intern(TermKind::BvSub(zero, arg), sort)
    }

    /// Create an unsigned bit vector division
    pub fn mk_bv_udiv(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvUdiv(lhs, rhs), sort)
    }

    /// Create a signed bit vector division
    pub fn mk_bv_sdiv(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvSdiv(lhs, rhs), sort)
    }

    /// Create an unsigned bit vector remainder
    pub fn mk_bv_urem(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvUrem(lhs, rhs), sort)
    }

    /// Create a signed bit vector remainder
    pub fn mk_bv_srem(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map_or(self.sorts.bool_sort, |t| t.sort);
        self.intern(TermKind::BvSrem(lhs, rhs), sort)
    }

    /// Create a bit vector XOR
    pub fn mk_bv_xor(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvXor(lhs, rhs), sort)
    }

    /// Create a bit vector shift left
    pub fn mk_bv_shl(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvShl(lhs, rhs), sort)
    }

    /// Create a bit vector logical shift right
    pub fn mk_bv_lshr(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvLshr(lhs, rhs), sort)
    }

    /// Create a bit vector arithmetic shift right
    pub fn mk_bv_ashr(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let sort = self.get(lhs).map(|t| t.sort);
        let sort = sort.unwrap_or_else(|| self.sorts.bitvec(32));
        self.intern(TermKind::BvAshr(lhs, rhs), sort)
    }
}
