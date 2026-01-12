//! Term Manager - Arena allocation for terms

use super::term::{RoundingMode, Term, TermId, TermKind};
use super::traversal::get_children;
use crate::sort::{SortId, SortManager};
use lasso::{Rodeo, Spur};
use num_bigint::BigInt;
use num_rational::Rational64;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU32, Ordering};

/// Statistics for garbage collection
#[derive(Debug, Clone, Default)]
pub struct GCStatistics {
    /// Number of GC runs
    pub gc_count: usize,
    /// Total terms collected across all GC runs
    pub total_collected: usize,
    /// Total cache entries removed across all GC runs
    pub total_cache_removed: usize,
    /// Last GC collection count
    pub last_collected: usize,
    /// Last GC cache removal count
    pub last_cache_removed: usize,
}

/// Manager for term allocation and interning
#[derive(Debug)]
pub struct TermManager {
    /// Arena for term storage
    terms: Vec<Term>,
    /// Next term ID
    next_id: AtomicU32,
    /// String interner for symbols
    interner: Rodeo,
    /// Sort manager
    pub sorts: SortManager,
    /// Cache for structural sharing
    cache: FxHashMap<TermKind, TermId>,
    /// True constant
    pub true_id: TermId,
    /// False constant
    pub false_id: TermId,
    /// GC statistics
    gc_stats: GCStatistics,
}

impl Default for TermManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TermManager {
    /// Create a new term manager
    #[must_use]
    pub fn new() -> Self {
        let sorts = SortManager::new();
        let bool_sort = sorts.bool_sort;

        let mut manager = Self {
            terms: Vec::with_capacity(1024),
            next_id: AtomicU32::new(0),
            interner: Rodeo::default(),
            sorts,
            cache: FxHashMap::default(),
            true_id: TermId(0),
            false_id: TermId(1),
            gc_stats: GCStatistics::default(),
        };

        // Pre-allocate true and false
        manager.true_id = manager.intern(TermKind::True, bool_sort);
        manager.false_id = manager.intern(TermKind::False, bool_sort);

        manager
    }

    /// Intern a term, returning its unique ID
    pub(crate) fn intern(&mut self, kind: TermKind, sort: SortId) -> TermId {
        if let Some(&id) = self.cache.get(&kind) {
            return id;
        }

        let id = TermId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let term = Term {
            id,
            kind: kind.clone(),
            sort,
        };
        self.terms.push(term);
        self.cache.insert(kind, id);
        id
    }

    /// Get a term by its ID
    #[must_use]
    pub fn get(&self, id: TermId) -> Option<&Term> {
        self.terms.get(id.0 as usize)
    }

    /// Intern a string, returning its key
    pub fn intern_str(&mut self, s: &str) -> Spur {
        self.interner.get_or_intern(s)
    }

    /// Resolve an interned string
    #[must_use]
    pub fn resolve_str(&self, key: Spur) -> &str {
        self.interner.resolve(&key)
    }

    // Builder methods

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
        self.mk_forall_with_patterns(vars, body, std::iter::empty::<Vec<TermId>>())
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
        self.mk_exists_with_patterns(vars, body, std::iter::empty::<Vec<TermId>>())
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

    /// Get the number of terms allocated
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if the manager is empty (only contains true/false)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.len() <= 2
    }

    // ===== Term Analysis =====

    /// Compute the size (number of nodes) of a term
    #[must_use]
    pub fn term_size(&self, id: TermId) -> usize {
        self.term_size_cached(id, &mut FxHashMap::default())
    }

    /// Compute the size with memoization
    fn term_size_cached(&self, id: TermId, cache: &mut FxHashMap<TermId, usize>) -> usize {
        if let Some(&size) = cache.get(&id) {
            return size;
        }

        let size = match self.get(id).map(|t| &t.kind) {
            None => 1,
            Some(
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_)
                | TermKind::Var(_),
            ) => 1,
            Some(
                TermKind::Not(arg)
                | TermKind::Neg(arg)
                | TermKind::BvNot(arg)
                | TermKind::StrLen(arg)
                | TermKind::StrToInt(arg)
                | TermKind::IntToStr(arg),
            ) => 1 + self.term_size_cached(*arg, cache),
            Some(TermKind::BvExtract { arg, .. }) => 1 + self.term_size_cached(*arg, cache),
            Some(
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::Distinct(args),
            ) => {
                1 + args
                    .iter()
                    .map(|&a| self.term_size_cached(a, cache))
                    .sum::<usize>()
            }
            Some(
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
                | TermKind::BvSle(a, b),
            ) => 1 + self.term_size_cached(*a, cache) + self.term_size_cached(*b, cache),
            Some(
                TermKind::Ite(c, t, e)
                | TermKind::Store(c, t, e)
                | TermKind::StrSubstr(c, t, e)
                | TermKind::StrIndexOf(c, t, e)
                | TermKind::StrReplace(c, t, e)
                | TermKind::StrReplaceAll(c, t, e),
            ) => {
                1 + self.term_size_cached(*c, cache)
                    + self.term_size_cached(*t, cache)
                    + self.term_size_cached(*e, cache)
            }
            Some(TermKind::Apply { args, .. }) => {
                1 + args
                    .iter()
                    .map(|&a| self.term_size_cached(a, cache))
                    .sum::<usize>()
            }
            Some(TermKind::Forall { body, .. } | TermKind::Exists { body, .. }) => {
                1 + self.term_size_cached(*body, cache)
            }
            Some(TermKind::Let { bindings, body }) => {
                1 + bindings
                    .iter()
                    .map(|(_, t)| self.term_size_cached(*t, cache))
                    .sum::<usize>()
                    + self.term_size_cached(*body, cache)
            }
            // Floating-point operations - calculate size recursively
            Some(_) => self.get(id).map_or(0, |term| {
                1 + get_children(&term.kind)
                    .iter()
                    .map(|&child| self.term_size_cached(child, cache))
                    .sum::<usize>()
            }),
        };

        cache.insert(id, size);
        size
    }

    /// Compute the depth of a term
    #[must_use]
    pub fn term_depth(&self, id: TermId) -> usize {
        self.term_depth_cached(id, &mut FxHashMap::default())
    }

    /// Compute the depth with memoization
    fn term_depth_cached(&self, id: TermId, cache: &mut FxHashMap<TermId, usize>) -> usize {
        if let Some(&depth) = cache.get(&id) {
            return depth;
        }

        let depth = match self.get(id).map(|t| &t.kind) {
            None => 0,
            Some(
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_)
                | TermKind::Var(_),
            ) => 0,
            Some(
                TermKind::Not(arg)
                | TermKind::Neg(arg)
                | TermKind::BvNot(arg)
                | TermKind::StrLen(arg)
                | TermKind::StrToInt(arg)
                | TermKind::IntToStr(arg),
            ) => 1 + self.term_depth_cached(*arg, cache),
            Some(TermKind::BvExtract { arg, .. }) => 1 + self.term_depth_cached(*arg, cache),
            Some(
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::Distinct(args),
            ) => {
                1 + args
                    .iter()
                    .map(|&a| self.term_depth_cached(a, cache))
                    .max()
                    .unwrap_or(0)
            }
            Some(
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
                | TermKind::BvSle(a, b),
            ) => {
                1 + self
                    .term_depth_cached(*a, cache)
                    .max(self.term_depth_cached(*b, cache))
            }
            Some(
                TermKind::Ite(c, t, e)
                | TermKind::Store(c, t, e)
                | TermKind::StrSubstr(c, t, e)
                | TermKind::StrIndexOf(c, t, e)
                | TermKind::StrReplace(c, t, e)
                | TermKind::StrReplaceAll(c, t, e),
            ) => {
                1 + self
                    .term_depth_cached(*c, cache)
                    .max(self.term_depth_cached(*t, cache))
                    .max(self.term_depth_cached(*e, cache))
            }
            Some(TermKind::Apply { args, .. }) => {
                1 + args
                    .iter()
                    .map(|&a| self.term_depth_cached(a, cache))
                    .max()
                    .unwrap_or(0)
            }
            Some(TermKind::Forall { body, .. } | TermKind::Exists { body, .. }) => {
                1 + self.term_depth_cached(*body, cache)
            }
            Some(TermKind::Let { bindings, body }) => {
                let binding_depth = bindings
                    .iter()
                    .map(|(_, t)| self.term_depth_cached(*t, cache))
                    .max()
                    .unwrap_or(0);
                1 + binding_depth.max(self.term_depth_cached(*body, cache))
            }
            // Floating-point operations - calculate depth recursively
            Some(_) => self.get(id).map_or(0, |term| {
                1 + get_children(&term.kind)
                    .iter()
                    .map(|&child| self.term_depth_cached(child, cache))
                    .max()
                    .unwrap_or(0)
            }),
        };

        cache.insert(id, depth);
        depth
    }

    /// Substitute variables in a term according to a mapping
    pub fn substitute(&mut self, id: TermId, subst: &FxHashMap<TermId, TermId>) -> TermId {
        self.substitute_cached(id, subst, &mut FxHashMap::default())
    }

    /// Substitute with memoization
    fn substitute_cached(
        &mut self,
        id: TermId,
        subst: &FxHashMap<TermId, TermId>,
        cache: &mut FxHashMap<TermId, TermId>,
    ) -> TermId {
        // Check if this term is directly substituted
        if let Some(&replacement) = subst.get(&id) {
            return replacement;
        }

        // Check cache
        if let Some(&result) = cache.get(&id) {
            return result;
        }

        let result = match self.get(id).map(|t| t.kind.clone()) {
            None => id,
            Some(
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::Var(_),
            ) => id,
            Some(TermKind::Not(arg)) => {
                let new_arg = self.substitute_cached(arg, subst, cache);
                if new_arg == arg {
                    id
                } else {
                    self.mk_not(new_arg)
                }
            }
            Some(TermKind::And(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_cached(a, subst, cache))
                    .collect();
                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    id
                } else {
                    self.mk_and(new_args)
                }
            }
            Some(TermKind::Or(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_cached(a, subst, cache))
                    .collect();
                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    id
                } else {
                    self.mk_or(new_args)
                }
            }
            Some(TermKind::Implies(lhs, rhs)) => {
                let new_lhs = self.substitute_cached(lhs, subst, cache);
                let new_rhs = self.substitute_cached(rhs, subst, cache);
                if new_lhs == lhs && new_rhs == rhs {
                    id
                } else {
                    self.mk_implies(new_lhs, new_rhs)
                }
            }
            Some(TermKind::Eq(lhs, rhs)) => {
                let new_lhs = self.substitute_cached(lhs, subst, cache);
                let new_rhs = self.substitute_cached(rhs, subst, cache);
                if new_lhs == lhs && new_rhs == rhs {
                    id
                } else {
                    self.mk_eq(new_lhs, new_rhs)
                }
            }
            Some(TermKind::Ite(cond, then_br, else_br)) => {
                let new_cond = self.substitute_cached(cond, subst, cache);
                let new_then = self.substitute_cached(then_br, subst, cache);
                let new_else = self.substitute_cached(else_br, subst, cache);
                if new_cond == cond && new_then == then_br && new_else == else_br {
                    id
                } else {
                    self.mk_ite(new_cond, new_then, new_else)
                }
            }
            Some(TermKind::Add(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_cached(a, subst, cache))
                    .collect();
                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    id
                } else {
                    self.mk_add(new_args)
                }
            }
            Some(TermKind::Sub(lhs, rhs)) => {
                let new_lhs = self.substitute_cached(lhs, subst, cache);
                let new_rhs = self.substitute_cached(rhs, subst, cache);
                if new_lhs == lhs && new_rhs == rhs {
                    id
                } else {
                    self.mk_sub(new_lhs, new_rhs)
                }
            }
            Some(TermKind::Mul(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.substitute_cached(a, subst, cache))
                    .collect();
                if new_args.iter().zip(args.iter()).all(|(a, b)| a == b) {
                    id
                } else {
                    self.mk_mul(new_args)
                }
            }
            Some(TermKind::Lt(lhs, rhs)) => {
                let new_lhs = self.substitute_cached(lhs, subst, cache);
                let new_rhs = self.substitute_cached(rhs, subst, cache);
                if new_lhs == lhs && new_rhs == rhs {
                    id
                } else {
                    self.mk_lt(new_lhs, new_rhs)
                }
            }
            Some(TermKind::Le(lhs, rhs)) => {
                let new_lhs = self.substitute_cached(lhs, subst, cache);
                let new_rhs = self.substitute_cached(rhs, subst, cache);
                if new_lhs == lhs && new_rhs == rhs {
                    id
                } else {
                    self.mk_le(new_lhs, new_rhs)
                }
            }
            Some(TermKind::Gt(lhs, rhs)) => {
                let new_lhs = self.substitute_cached(lhs, subst, cache);
                let new_rhs = self.substitute_cached(rhs, subst, cache);
                if new_lhs == lhs && new_rhs == rhs {
                    id
                } else {
                    self.mk_gt(new_lhs, new_rhs)
                }
            }
            Some(TermKind::Ge(lhs, rhs)) => {
                let new_lhs = self.substitute_cached(lhs, subst, cache);
                let new_rhs = self.substitute_cached(rhs, subst, cache);
                if new_lhs == lhs && new_rhs == rhs {
                    id
                } else {
                    self.mk_ge(new_lhs, new_rhs)
                }
            }
            Some(TermKind::Select(arr, idx)) => {
                let new_arr = self.substitute_cached(arr, subst, cache);
                let new_idx = self.substitute_cached(idx, subst, cache);
                if new_arr == arr && new_idx == idx {
                    id
                } else {
                    self.mk_select(new_arr, new_idx)
                }
            }
            Some(TermKind::Store(arr, idx, val)) => {
                let new_arr = self.substitute_cached(arr, subst, cache);
                let new_idx = self.substitute_cached(idx, subst, cache);
                let new_val = self.substitute_cached(val, subst, cache);
                if new_arr == arr && new_idx == idx && new_val == val {
                    id
                } else {
                    self.mk_store(new_arr, new_idx, new_val)
                }
            }
            // For complex terms, just return as-is for now
            Some(_) => id,
        };

        cache.insert(id, result);
        result
    }

    /// Simplify a term by applying rewrite rules
    ///
    /// This performs bottom-up simplification including:
    /// - Constant folding for arithmetic
    /// - Boolean simplifications
    /// - Identity/annihilator rules
    pub fn simplify(&mut self, id: TermId) -> TermId {
        let mut cache = FxHashMap::default();
        self.simplify_cached(id, &mut cache)
    }

    fn simplify_cached(&mut self, id: TermId, cache: &mut FxHashMap<TermId, TermId>) -> TermId {
        if let Some(&result) = cache.get(&id) {
            return result;
        }

        let result = match self.get(id).map(|t| t.kind.clone()) {
            None
            | Some(
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::Var(_),
            ) => id,

            Some(TermKind::Not(arg)) => {
                let new_arg = self.simplify_cached(arg, cache);
                self.mk_not(new_arg)
            }
            Some(TermKind::And(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.simplify_cached(a, cache))
                    .collect();
                self.mk_and(new_args)
            }
            Some(TermKind::Or(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.simplify_cached(a, cache))
                    .collect();
                self.mk_or(new_args)
            }
            Some(TermKind::Implies(lhs, rhs)) => {
                let new_lhs = self.simplify_cached(lhs, cache);
                let new_rhs = self.simplify_cached(rhs, cache);
                self.mk_implies(new_lhs, new_rhs)
            }
            Some(TermKind::Eq(lhs, rhs)) => {
                let new_lhs = self.simplify_cached(lhs, cache);
                let new_rhs = self.simplify_cached(rhs, cache);
                self.mk_eq(new_lhs, new_rhs)
            }
            Some(TermKind::Ite(cond, then_br, else_br)) => {
                let new_cond = self.simplify_cached(cond, cache);
                let new_then = self.simplify_cached(then_br, cache);
                let new_else = self.simplify_cached(else_br, cache);
                self.mk_ite(new_cond, new_then, new_else)
            }
            Some(TermKind::Add(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.simplify_cached(a, cache))
                    .collect();
                self.simplify_add(new_args)
            }
            Some(TermKind::Sub(lhs, rhs)) => {
                let new_lhs = self.simplify_cached(lhs, cache);
                let new_rhs = self.simplify_cached(rhs, cache);
                self.simplify_sub(new_lhs, new_rhs)
            }
            Some(TermKind::Mul(args)) => {
                let new_args: SmallVec<[TermId; 4]> = args
                    .iter()
                    .map(|&a| self.simplify_cached(a, cache))
                    .collect();
                self.simplify_mul(new_args)
            }
            Some(TermKind::Neg(arg)) => {
                let new_arg = self.simplify_cached(arg, cache);
                self.simplify_neg(new_arg)
            }
            Some(TermKind::Lt(lhs, rhs)) => {
                let new_lhs = self.simplify_cached(lhs, cache);
                let new_rhs = self.simplify_cached(rhs, cache);
                self.simplify_lt(new_lhs, new_rhs)
            }
            Some(TermKind::Le(lhs, rhs)) => {
                let new_lhs = self.simplify_cached(lhs, cache);
                let new_rhs = self.simplify_cached(rhs, cache);
                self.simplify_le(new_lhs, new_rhs)
            }
            Some(TermKind::Gt(lhs, rhs)) => {
                let new_lhs = self.simplify_cached(lhs, cache);
                let new_rhs = self.simplify_cached(rhs, cache);
                self.simplify_gt(new_lhs, new_rhs)
            }
            Some(TermKind::Ge(lhs, rhs)) => {
                let new_lhs = self.simplify_cached(lhs, cache);
                let new_rhs = self.simplify_cached(rhs, cache);
                self.simplify_ge(new_lhs, new_rhs)
            }
            // For other terms, just return as-is
            Some(_) => id,
        };

        cache.insert(id, result);
        result
    }

    /// Simplify addition with constant folding
    fn simplify_add(&mut self, args: SmallVec<[TermId; 4]>) -> TermId {
        let mut constant_sum = BigInt::from(0);
        let mut other_args: SmallVec<[TermId; 4]> = SmallVec::new();

        for arg in args {
            if let Some(TermKind::IntConst(n)) = self.get(arg).map(|t| &t.kind) {
                constant_sum += n;
            } else {
                other_args.push(arg);
            }
        }

        let zero = BigInt::from(0);
        if other_args.is_empty() {
            return self.intern(TermKind::IntConst(constant_sum), self.sorts.int_sort);
        }

        if constant_sum != zero {
            other_args.push(self.intern(TermKind::IntConst(constant_sum), self.sorts.int_sort));
        }

        if other_args.len() == 1 {
            return other_args[0];
        }

        self.mk_add(other_args)
    }

    /// Simplify subtraction with constant folding
    fn simplify_sub(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        let zero = BigInt::from(0);
        match (
            self.get(lhs).map(|t| t.kind.clone()),
            self.get(rhs).map(|t| t.kind.clone()),
        ) {
            (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) => {
                self.intern(TermKind::IntConst(a - b), self.sorts.int_sort)
            }
            (_, Some(TermKind::IntConst(n))) if n == zero => lhs,
            (Some(TermKind::IntConst(n)), _) if n == zero => self.simplify_neg(rhs),
            _ => self.mk_sub(lhs, rhs),
        }
    }

    /// Simplify multiplication with constant folding
    fn simplify_mul(&mut self, args: SmallVec<[TermId; 4]>) -> TermId {
        let mut constant_product = BigInt::from(1);
        let mut other_args: SmallVec<[TermId; 4]> = SmallVec::new();
        let zero = BigInt::from(0);
        let one = BigInt::from(1);

        for arg in args {
            if let Some(TermKind::IntConst(n)) = self.get(arg).map(|t| &t.kind) {
                if *n == zero {
                    return self.mk_int(0);
                }
                constant_product *= n;
            } else {
                other_args.push(arg);
            }
        }

        if other_args.is_empty() {
            return self.intern(TermKind::IntConst(constant_product), self.sorts.int_sort);
        }

        if constant_product == zero {
            return self.mk_int(0);
        }

        if constant_product != one {
            other_args.insert(
                0,
                self.intern(TermKind::IntConst(constant_product), self.sorts.int_sort),
            );
        }

        if other_args.len() == 1 {
            return other_args[0];
        }

        self.mk_mul(other_args)
    }

    /// Simplify negation
    fn simplify_neg(&mut self, arg: TermId) -> TermId {
        match self.get(arg).map(|t| t.kind.clone()) {
            Some(TermKind::IntConst(n)) => self.intern(TermKind::IntConst(-n), self.sorts.int_sort),
            Some(TermKind::Neg(inner)) => inner,
            _ => {
                let sort = self.get(arg).map_or(self.sorts.int_sort, |t| t.sort);
                self.intern(TermKind::Neg(arg), sort)
            }
        }
    }

    /// Simplify less-than with constant comparison
    fn simplify_lt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        match (
            self.get(lhs).map(|t| &t.kind),
            self.get(rhs).map(|t| &t.kind),
        ) {
            (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) => self.mk_bool(a < b),
            _ => self.mk_lt(lhs, rhs),
        }
    }

    /// Simplify less-or-equal with constant comparison
    fn simplify_le(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        match (
            self.get(lhs).map(|t| &t.kind),
            self.get(rhs).map(|t| &t.kind),
        ) {
            (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) => self.mk_bool(a <= b),
            _ => self.mk_le(lhs, rhs),
        }
    }

    /// Simplify greater-than with constant comparison
    fn simplify_gt(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        match (
            self.get(lhs).map(|t| &t.kind),
            self.get(rhs).map(|t| &t.kind),
        ) {
            (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) => self.mk_bool(a > b),
            _ => self.mk_gt(lhs, rhs),
        }
    }

    /// Simplify greater-or-equal with constant comparison
    fn simplify_ge(&mut self, lhs: TermId, rhs: TermId) -> TermId {
        match (
            self.get(lhs).map(|t| &t.kind),
            self.get(rhs).map(|t| &t.kind),
        ) {
            (Some(TermKind::IntConst(a)), Some(TermKind::IntConst(b))) => self.mk_bool(a >= b),
            _ => self.mk_ge(lhs, rhs),
        }
    }

    /// Collect all free variables in a term
    pub fn free_vars(&self, id: TermId) -> Vec<TermId> {
        let mut vars = Vec::new();
        let mut visited = FxHashMap::default();
        self.collect_free_vars(id, &mut vars, &mut visited);
        vars
    }

    fn collect_free_vars(
        &self,
        id: TermId,
        vars: &mut Vec<TermId>,
        visited: &mut FxHashMap<TermId, ()>,
    ) {
        if visited.contains_key(&id) {
            return;
        }
        visited.insert(id, ());

        match self.get(id).map(|t| &t.kind) {
            None => {}
            Some(TermKind::Var(_)) => {
                if !vars.contains(&id) {
                    vars.push(id);
                }
            }
            Some(
                TermKind::True
                | TermKind::False
                | TermKind::IntConst(_)
                | TermKind::RealConst(_)
                | TermKind::BitVecConst { .. }
                | TermKind::StringLit(_),
            ) => {}
            Some(
                TermKind::Not(arg)
                | TermKind::Neg(arg)
                | TermKind::BvNot(arg)
                | TermKind::StrLen(arg)
                | TermKind::StrToInt(arg)
                | TermKind::IntToStr(arg),
            ) => {
                self.collect_free_vars(*arg, vars, visited);
            }
            Some(TermKind::BvExtract { arg, .. }) => {
                self.collect_free_vars(*arg, vars, visited);
            }
            Some(
                TermKind::And(args)
                | TermKind::Or(args)
                | TermKind::Add(args)
                | TermKind::Mul(args)
                | TermKind::Distinct(args),
            ) => {
                for &arg in args {
                    self.collect_free_vars(arg, vars, visited);
                }
            }
            Some(
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
                | TermKind::BvSle(a, b),
            ) => {
                self.collect_free_vars(*a, vars, visited);
                self.collect_free_vars(*b, vars, visited);
            }
            Some(
                TermKind::Ite(c, t, e)
                | TermKind::Store(c, t, e)
                | TermKind::StrSubstr(c, t, e)
                | TermKind::StrIndexOf(c, t, e)
                | TermKind::StrReplace(c, t, e)
                | TermKind::StrReplaceAll(c, t, e),
            ) => {
                self.collect_free_vars(*c, vars, visited);
                self.collect_free_vars(*t, vars, visited);
                self.collect_free_vars(*e, vars, visited);
            }
            Some(TermKind::Apply { args, .. }) => {
                for &arg in args {
                    self.collect_free_vars(arg, vars, visited);
                }
            }
            Some(TermKind::Forall { body, .. } | TermKind::Exists { body, .. }) => {
                // Note: This is simplified - we should track bound vars
                self.collect_free_vars(*body, vars, visited);
            }
            Some(TermKind::Let { bindings, body }) => {
                for (_, term) in bindings {
                    self.collect_free_vars(*term, vars, visited);
                }
                self.collect_free_vars(*body, vars, visited);
            }
            // Floating-point operations - collect vars from children
            Some(_) => {
                if let Some(term) = self.get(id) {
                    for &child in &get_children(&term.kind) {
                        self.collect_free_vars(child, vars, visited);
                    }
                }
            }
        }
    }

    // ========================== Garbage Collection ==========================

    /// Perform garbage collection on unreachable terms
    ///
    /// This method performs a mark-and-sweep garbage collection:
    /// 1. Marks all terms reachable from the given root set
    /// 2. Removes unmarked entries from the cache
    ///
    /// Note: This doesn't actually free memory from the arena (terms vector),
    /// but it does clean up the cache to prevent unbounded growth.
    ///
    /// # Arguments
    /// * `roots` - Set of root term IDs to keep (and their descendants)
    ///
    /// # Returns
    /// Number of cache entries removed
    pub fn gc(&mut self, roots: &FxHashSet<TermId>) -> usize {
        // Mark phase: find all reachable terms
        let mut reachable = FxHashSet::default();
        let mut worklist: Vec<TermId> = roots.iter().copied().collect();

        // Always keep true and false
        worklist.push(self.true_id);
        worklist.push(self.false_id);

        while let Some(id) = worklist.pop() {
            if !reachable.insert(id) {
                continue; // Already visited
            }

            // Mark children as reachable
            if let Some(term) = self.get(id) {
                for child in get_children(&term.kind) {
                    if !reachable.contains(&child) {
                        worklist.push(child);
                    }
                }
            }
        }

        // Sweep phase: remove unreachable entries from cache
        let original_cache_size = self.cache.len();
        self.cache.retain(|_, &mut id| reachable.contains(&id));
        let removed = original_cache_size - self.cache.len();

        // Update statistics
        self.gc_stats.gc_count += 1;
        self.gc_stats.total_cache_removed += removed;
        self.gc_stats.last_cache_removed = removed;
        self.gc_stats.last_collected = removed;
        self.gc_stats.total_collected += removed;

        removed
    }

    /// Perform aggressive garbage collection
    ///
    /// Similar to `gc()` but more thorough. It also shrinks the cache capacity
    /// to fit the retained entries, potentially freeing more memory.
    ///
    /// # Arguments
    /// * `roots` - Set of root term IDs to keep (and their descendants)
    ///
    /// # Returns
    /// Number of cache entries removed
    pub fn gc_aggressive(&mut self, roots: &FxHashSet<TermId>) -> usize {
        let removed = self.gc(roots);
        self.cache.shrink_to_fit();
        removed
    }

    /// Get garbage collection statistics
    #[must_use]
    pub fn gc_statistics(&self) -> &GCStatistics {
        &self.gc_stats
    }

    /// Get the current cache size (number of hash-consed terms)
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get the total number of terms allocated
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Clear all GC statistics
    pub fn reset_gc_stats(&mut self) {
        self.gc_stats = GCStatistics::default();
    }
}

/// Builder for constructing substitutions incrementally with optimizations
///
/// This provides better performance than repeatedly calling substitute when
/// building up complex substitutions, especially when:
/// - Composing multiple substitutions
/// - Applying the same substitution to many terms
/// - Building substitutions incrementally
#[derive(Debug, Clone)]
pub struct SubstitutionBuilder {
    /// The substitution mapping
    mapping: FxHashMap<TermId, TermId>,
    /// Shared cache for substitution results
    cache: FxHashMap<TermId, TermId>,
}

impl SubstitutionBuilder {
    /// Create a new empty substitution builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            mapping: FxHashMap::default(),
            cache: FxHashMap::default(),
        }
    }

    /// Create a builder with initial capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            mapping: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            cache: FxHashMap::with_capacity_and_hasher(capacity * 2, Default::default()),
        }
    }

    /// Add a substitution mapping
    pub fn add(&mut self, from: TermId, to: TermId) -> &mut Self {
        // Invalidate cache when adding new mapping
        self.cache.clear();
        self.mapping.insert(from, to);
        self
    }

    /// Add multiple substitution mappings
    pub fn add_many(&mut self, mappings: impl IntoIterator<Item = (TermId, TermId)>) -> &mut Self {
        self.cache.clear();
        self.mapping.extend(mappings);
        self
    }

    /// Compose this substitution with another
    ///
    /// The resulting substitution applies `other` first, then `self`.
    /// This is optimized to share structure where possible.
    pub fn compose(&mut self, other: &SubstitutionBuilder, manager: &mut TermManager) -> &mut Self {
        // For each mapping in self, substitute using other
        let mut new_mapping = FxHashMap::default();
        let mut temp_cache = FxHashMap::default();

        for (&from, &to) in &self.mapping {
            let new_to = if other.mapping.contains_key(&to) {
                manager.substitute_cached(to, &other.mapping, &mut temp_cache)
            } else {
                to
            };
            new_mapping.insert(from, new_to);
        }

        // Add mappings from other that aren't in self
        for (&from, &to) in &other.mapping {
            new_mapping.entry(from).or_insert(to);
        }

        self.mapping = new_mapping;
        self.cache.clear();
        self
    }

    /// Apply the substitution to a term
    ///
    /// This uses a persistent cache across multiple applications,
    /// making it more efficient when substituting many terms.
    pub fn apply(&mut self, id: TermId, manager: &mut TermManager) -> TermId {
        manager.substitute_cached(id, &self.mapping, &mut self.cache)
    }

    /// Apply the substitution to multiple terms efficiently
    ///
    /// Uses the shared cache to avoid redundant work.
    pub fn apply_many(&mut self, ids: &[TermId], manager: &mut TermManager) -> Vec<TermId> {
        ids.iter().map(|&id| self.apply(id, manager)).collect()
    }

    /// Get the underlying mapping
    #[must_use]
    pub fn mapping(&self) -> &FxHashMap<TermId, TermId> {
        &self.mapping
    }

    /// Check if the substitution is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }

    /// Get the number of mappings
    #[must_use]
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Clear the substitution
    pub fn clear(&mut self) {
        self.mapping.clear();
        self.cache.clear();
    }

    /// Reset the cache (useful for freeing memory)
    pub fn reset_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics (for debugging/optimization)
    #[must_use]
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.len(), self.cache.capacity())
    }
}

impl Default for SubstitutionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        let manager = TermManager::new();
        assert_ne!(manager.mk_true(), manager.mk_false());
        assert_eq!(manager.mk_bool(true), manager.mk_true());
        assert_eq!(manager.mk_bool(false), manager.mk_false());
    }

    #[test]
    fn test_not_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();

        assert_eq!(manager.mk_not(t), f);
        assert_eq!(manager.mk_not(f), t);

        let x = manager.mk_var("x", manager.sorts.bool_sort);
        let not_x = manager.mk_not(x);
        let not_not_x = manager.mk_not(not_x);
        assert_eq!(not_not_x, x);
    }

    #[test]
    fn test_and_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        assert_eq!(manager.mk_and([t, x]), x);
        assert_eq!(manager.mk_and([f, x]), f);
        assert_eq!(manager.mk_and([t, t]), t);
        assert_eq!(manager.mk_and(std::iter::empty()), t);
    }

    #[test]
    fn test_or_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();
        let x = manager.mk_var("x", manager.sorts.bool_sort);

        assert_eq!(manager.mk_or([f, x]), x);
        assert_eq!(manager.mk_or([t, x]), t);
        assert_eq!(manager.mk_or([f, f]), f);
        assert_eq!(manager.mk_or(std::iter::empty()), f);
    }

    #[test]
    fn test_eq_canonicalization() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        let eq1 = manager.mk_eq(x, y);
        let eq2 = manager.mk_eq(y, x);
        assert_eq!(eq1, eq2);

        assert_eq!(manager.mk_eq(x, x), manager.mk_true());
    }

    #[test]
    fn test_ite_simplification() {
        let mut manager = TermManager::new();
        let t = manager.mk_true();
        let f = manager.mk_false();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        assert_eq!(manager.mk_ite(t, x, y), x);
        assert_eq!(manager.mk_ite(f, x, y), y);
        assert_eq!(manager.mk_ite(t, x, x), x);
    }

    #[test]
    fn test_interning() {
        let mut manager = TermManager::new();
        let x1 = manager.mk_var("x", manager.sorts.int_sort);
        let x2 = manager.mk_var("x", manager.sorts.int_sort);
        assert_eq!(x1, x2);

        let int1 = manager.mk_int(42);
        let int2 = manager.mk_int(42);
        assert_eq!(int1, int2);
    }

    #[test]
    fn test_term_size() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // Single variable has size 1
        assert_eq!(manager.term_size(x), 1);

        // x + y has size 3 (add node + 2 children)
        let add = manager.mk_add([x, y]);
        assert_eq!(manager.term_size(add), 3);

        // (x + y) < 10 has size 5
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(add, ten);
        assert_eq!(manager.term_size(lt), 5);
    }

    #[test]
    fn test_term_depth() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);

        // Variables have depth 0
        assert_eq!(manager.term_depth(x), 0);

        // x + y has depth 1
        let add = manager.mk_add([x, y]);
        assert_eq!(manager.term_depth(add), 1);

        // (x + y) < 10 has depth 2
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(add, ten);
        assert_eq!(manager.term_depth(lt), 2);
    }

    #[test]
    fn test_substitute() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let five = manager.mk_int(5);

        // Create x + y
        let add = manager.mk_add([x, y]);

        // Substitute x -> 5
        let mut subst = FxHashMap::default();
        subst.insert(x, five);
        let result = manager.substitute(add, &subst);

        // Result should be 5 + y
        let expected = manager.mk_add([five, y]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_free_vars() {
        let mut manager = TermManager::new();
        let x = manager.mk_var("x", manager.sorts.int_sort);
        let y = manager.mk_var("y", manager.sorts.int_sort);
        let z = manager.mk_var("z", manager.sorts.bool_sort);

        // x + y has free vars {x, y}
        let add = manager.mk_add([x, y]);
        let vars = manager.free_vars(add);
        assert!(vars.contains(&x));
        assert!(vars.contains(&y));
        assert!(!vars.contains(&z));

        // (x + y) < 10 has free vars {x, y}
        let ten = manager.mk_int(10);
        let lt = manager.mk_lt(add, ten);
        let vars = manager.free_vars(lt);
        assert_eq!(vars.len(), 2);
    }

    // ==================== Quantifier Pattern Tests ====================

    #[test]
    fn test_forall_without_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_gt(x, zero);

        let forall = manager.mk_forall([("x", int_sort)], body);

        // Check it's a forall without patterns
        if let Some(term) = manager.get(forall) {
            if let TermKind::Forall { patterns, .. } = &term.kind {
                assert!(patterns.is_empty());
            } else {
                panic!("Expected Forall");
            }
        }
    }

    #[test]
    fn test_forall_with_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        // Create (f x) as the pattern
        let x = manager.mk_var("x", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_gt(fx, zero);

        // (forall ((x Int)) (! (> (f x) 0) :pattern ((f x))))
        let forall = manager.mk_forall_with_patterns(
            [("x", int_sort)],
            body,
            [vec![fx]], // Single pattern with (f x)
        );

        // Check it's a forall with the correct pattern
        if let Some(term) = manager.get(forall) {
            if let TermKind::Forall { patterns, .. } = &term.kind {
                assert_eq!(patterns.len(), 1);
                assert_eq!(patterns[0].len(), 1);
                assert_eq!(patterns[0][0], fx);
            } else {
                panic!("Expected Forall");
            }
        }
    }

    #[test]
    fn test_forall_with_multiple_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let gx = manager.mk_apply("g", [x], int_sort);
        let body = manager.mk_true();

        // Multiple patterns: :pattern (f x) :pattern (g x)
        let forall = manager.mk_forall_with_patterns([("x", int_sort)], body, [vec![fx], vec![gx]]);

        if let Some(term) = manager.get(forall) {
            if let TermKind::Forall { patterns, .. } = &term.kind {
                assert_eq!(patterns.len(), 2);
            } else {
                panic!("Expected Forall");
            }
        }
    }

    #[test]
    fn test_forall_with_multi_term_pattern() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let y = manager.mk_var("y", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let gy = manager.mk_apply("g", [y], int_sort);
        let body = manager.mk_true();

        // Multi-term pattern: :pattern ((f x) (g y))
        let forall = manager.mk_forall_with_patterns(
            [("x", int_sort), ("y", int_sort)],
            body,
            [vec![fx, gy]],
        );

        if let Some(term) = manager.get(forall) {
            if let TermKind::Forall { patterns, .. } = &term.kind {
                assert_eq!(patterns.len(), 1);
                assert_eq!(patterns[0].len(), 2);
            } else {
                panic!("Expected Forall");
            }
        }
    }

    #[test]
    fn test_exists_with_patterns() {
        let mut manager = TermManager::new();
        let int_sort = manager.sorts.int_sort;

        let x = manager.mk_var("x", int_sort);
        let fx = manager.mk_apply("f", [x], int_sort);
        let zero = manager.mk_int(0);
        let body = manager.mk_eq(fx, zero);

        let exists = manager.mk_exists_with_patterns([("x", int_sort)], body, [vec![fx]]);

        if let Some(term) = manager.get(exists) {
            if let TermKind::Exists { patterns, .. } = &term.kind {
                assert_eq!(patterns.len(), 1);
            } else {
                panic!("Expected Exists");
            }
        }
    }
}
